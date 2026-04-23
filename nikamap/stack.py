from concurrent.futures import ProcessPoolExecutor
from functools import partial

import astropy.units as u
import numpy as np
from astropy.convolution.kernels import _round_up_to_odd_integer
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D, InverseVariance, StdDevUncertainty
from astropy.table import Table
from astropy.utils.console import ProgressBar
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from photutils.datasets import make_model_image

from .utils import (
    _shuffled_average,
    cpu_count,
    make_kernel_image,
)


def skycoord_to_catalog(coords, wcs, x_name="x_mean", y_name="y_mean"):
    """Convert SkyCoord objects into a minimal source catalog in pixel coordinates."""
    x_pix, y_pix = skycoord_to_pixel(coords, wcs)
    x_pix = np.asarray(x_pix, dtype=float)
    y_pix = np.asarray(y_pix, dtype=float)

    cat = Table(masked=True)
    cat["amplitude"] = np.ones(len(coords), dtype=float)
    cat[x_name] = x_pix
    cat[y_name] = y_pix
    return cat


def _kernel_to_model(kernel, catalog, x_name="x_mean", y_name="y_mean"):
    """Build a source table compatible with make_model_image from a Kernel2D."""
    if not hasattr(kernel, "model"):
        raise TypeError("kernel must provide a 'model' attribute for non-fast mode")

    model = kernel.model.copy()
    model_catalog = catalog.copy()

    for param_name in model.param_names:
        if param_name in ("amplitude", x_name, y_name):
            continue
        if param_name not in model_catalog.colnames:
            param_value = float(np.asarray(getattr(model, param_name)))
            model_catalog[param_name] = np.full(len(model_catalog), param_value, dtype=float)

    return model, model_catalog


def cat_to_hit_maps(catalogs, shape, kernel, x_name="x_mean", y_name="y_mean", fast=True):
    """Convert a source catalog to a list of hit maps, one per source.

    Each hit map is the convolution of a single source (with amplitude=1) with
    the kernel.  The resulting hit maps can be used as regressors in a linear
    fit to reconstruct the true map from the source positions and amplitudes.

    Parameters
    ----------
    catalogs : `~astropy.table.Table` or list of `~astropy.table.Table`
        Source catalog with columns for x/y positions and amplitude.
    shape : (ny, nx)
        Desired shape of the output hit maps.
    kernel : `~astropy.convolution.Kernel2D`
        Convolution kernel (e.g. beam PSF).
    x_name, y_name : str
        Column names for the pixel x / y positions.

    fast : bool, optional
        If True, use the FFT-based implementation (`make_kernel_image`).
        If False, use `photutils.datasets.make_model_image` with `kernel.model`.

    Returns
    -------
    hit_maps : list of 2D ndarrays
        List of hit maps, one per source, each convolved with the kernel.
    """
    if isinstance(catalogs, Table):
        catalogs = [catalogs]

    name_kwargs = {"x_name": x_name, "y_name": y_name}

    hit_maps = []
    for cat in catalogs:
        _cat = cat.copy()
        _cat["amplitude"] = 1  # unit amplitude for hit map
        if fast:
            hit_map = make_kernel_image(shape, kernel, _cat, **name_kwargs)
        else:
            args = _kernel_to_model(kernel, _cat, **name_kwargs)
            hit_map = make_model_image(shape, *args, model_shape=np.asarray(kernel).shape, **name_kwargs)
        hit_maps.append(np.asarray(hit_map))
    return hit_maps




def _prepare_simstack(
    data,
    wcs,
    kernel,
    coords,
    mask=None,
    weight=None,
    add_offset=False,
    fast=True,
):
    """
    Prepare arrays (X, y, s) for simstack from map and SkyCoord(s).

    See Viero et al. 2013 (https://arxiv.org/abs/1304.0446) for the original simstack method

    Parameters
    ----------
    data : 2D ndarray
        Input map data.
    wcs : WCS
        WCS for pixel conversion.
    kernel : Kernel2D
        Convolution kernel to represent the beam.
    coords : SkyCoord or list of SkyCoord
        Source positions (single population or several populations).
    mask : ndarray, optional
        Boolean mask for invalid pixels.
    weight : ndarray, optional
        Weights for each pixel.
    add_offset : bool, optional
        If True, add a constant offset regressor.
    fast : bool, optional
        If True, use FFT-based hit maps.

    Returns
    -------
    X : ndarray
        Design matrix (n_valid_pixels, n_groups [+1 if offset]).
    y : ndarray
        Data vector (n_valid_pixels,).
    w : ndarray or None
        Weights for valid pixels, or None.

    Raises
    ------
    ValueError
        If no valid pixels or input shapes are inconsistent.
    """
    true_map = np.asarray(data, dtype=float)
    if true_map.ndim != 2:
        raise ValueError("data must be a 2D array")

    if isinstance(coords, SkyCoord):
        coord_groups = [coords]
    else:
        coord_groups = list(coords)

    name_kwargs = {"x_name": "x_mean", "y_name": "y_mean"}

    catalogs = [skycoord_to_catalog(coords, wcs, **name_kwargs) for coords in coord_groups]
    hit_maps = cat_to_hit_maps(catalogs, true_map.shape, kernel, **name_kwargs, fast=fast)

    if len(hit_maps) == 0:
        raise ValueError("no hit map generated from coords")

    valid = np.isfinite(true_map)
    for h in hit_maps:
        valid &= np.isfinite(h)

    if mask is not None:
        valid &= ~mask

    if weight is not None:
        if weight.shape != true_map.shape:
            raise ValueError("weight must have the same shape as data")
        valid &= np.isfinite(weight) & (weight > 0)

    if not np.any(valid):
        raise ValueError("no valid pixel available for regression")

    y = true_map[valid]
    X = np.column_stack([h[valid] for h in hit_maps])
    if add_offset:
        X = np.column_stack([X, np.ones_like(y)])

    w = weight[valid] if weight is not None else None

    return X, y, w


def _solve_simstack(X, y, w=None):
    """
    Solve (weighted) linear least squares and return coefficients and 1-sigma errors.

    Parameters
    ----------
    X : ndarray
        Design matrix (n_samples, n_coeffs).
    y : ndarray
        Data vector (n_samples,).
    w : ndarray, optional
        Weights for weighted fit (n_samples,).

    Returns
    -------
    coeffs : ndarray
        Fitted coefficients.
    err : ndarray
        1-sigma errors

    Raises
    ------
    np.linalg.LinAlgError
        If the design matrix is rank deficient.
    """
    if w is not None:
        A = X * w[:, None]
        b = y * w
    else:
        A = X
        b = y

    coeffs, chi2, rank, _ = np.linalg.lstsq(A, b, rcond=None)
    if rank < A.shape[1]:
        raise np.linalg.LinAlgError("design matrix is rank deficient; coefficients are not uniquely constrained")

    dof = max(1, y.size - X.shape[1])
    cov = np.linalg.pinv(A.T @ A)
    cov_scaled = cov * (chi2 / dof)

    return coeffs, np.sqrt(np.diag(cov_scaled))


def simstack(
    data,
    wcs,
    kernel,
    coords,
    mask=None,
    weight=None,
    add_offset=False,
    fast=True,
):
    """
    Perform simultaneaous stacking on map.

    See Viero et al. 2013 (https://arxiv.org/abs/1304.0446) for the original simstack method

    Parameters
    ----------
    data : 2D ndarray
        Input map data.
    wcs : WCS
        WCS for pixel conversion.
    kernel : Kernel2D
        Convolution kernel to represent the beam.
    coords : SkyCoord or list of SkyCoord
        Source positions (single population or several populations).
    mask : ndarray, optional
        Boolean mask for invalid pixels.
    weight : ndarray, optional
        Weights for each pixel.
    add_offset : bool, optional
        If True, add a constant offset regressor.
    fast : bool, optional
        If True, use FFT-based hit maps.

    Returns
    -------
    mean_fluxes : ndarray
        mean fluxes of each population.
    err : ndarray
        1-sigma errors for each population

    Raises
    ------
    ValueError
        If no valid pixels or input shapes are inconsistent.
    """
    X, y, w = _prepare_simstack(
        data,
        wcs,
        kernel,
        coords,
        mask=mask,
        weight=weight,
        add_offset=add_offset,
        fast=fast,
    )
    return _solve_simstack(X, y, w=w)


def _shuffle_coords(coords):
    """Shuffle the coordinates."""
    shuffled_coords = []
    for coord in coords:
        shuffled_index = np.random.choice(len(coord), size=len(coord), replace=True)
        shuffled_coords.append(coord[shuffled_index])
    return shuffled_coords


def _shuffle_pixel_solve_simstack(*args, X=None, y=None, w=None):
    """Helper function to solve simstack for a bootstrap sample."""
    shuffled_index = np.random.choice(len(y), size=len(y), replace=True)
    coeffs = _solve_simstack(X[shuffled_index], y[shuffled_index], w=w[shuffled_index] if w is not None else None)[0]
    return coeffs


def gen_cutout2d(coord, data=None, shape=None, wcs=None, mode="partial", fill_value=np.nan):
    """Generate a 2D cutout around one sky coordinate.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
        Central coordinate of the cutout.
    data : ndarray, optional
        Input 2D array from which the cutout is extracted.
    shape : tuple, optional
        Output cutout shape as ``(ny, nx)`` in pixels.
    wcs : `~astropy.wcs.WCS`, optional
        World coordinate system associated with ``data``.
    mode : {"trim", "partial", "strict"}, optional
        Edge handling mode passed to `~astropy.nddata.Cutout2D`.
    fill_value : float, optional
        Fill value used when ``mode="partial"`` and the cutout extends beyond
        the input array.

    Returns
    -------
    ndarray
        Cutout data array.
    """
    return Cutout2D(data, coord, shape, wcs=wcs, mode=mode, fill_value=fill_value).data


def gen_reproject(coord, output_wcs=None, data=None, weights=None, shape=None, wcs=None, func=None):
    """Reproject data and weights onto a cutout centred on one coordinate.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
        Central sky position of the output cutout.
    output_wcs : `~astropy.wcs.WCS`, optional
        Output WCS template. Its reference world coordinates are updated in
        place to match ``coord``.
    data : ndarray, optional
        Input 2D data array.
    weights : ndarray, optional
        Input 2D weights array aligned with ``data``.
    shape : tuple, optional
        Output shape as ``(ny, nx)`` in pixels.
    wcs : `~astropy.wcs.WCS`, optional
        WCS associated with the input arrays.
    func : callable, optional
        Reprojection function following the reproject package API.

    Returns
    -------
    array_new : ndarray
        Reprojected data cutout.
    weight_new : ndarray
        Reprojected weights cutout with invalid pixels forced to zero.
    """
    output_wcs.wcs.crval = (coord.ra.to("deg").value, coord.dec.to("deg").value)
    array_new, footprint = func((data, wcs), output_wcs, shape, return_footprint=True)
    weight_new = func((weights, wcs), output_wcs, shape, return_footprint=False)

    array_new[footprint == 0] = np.nan
    weight_new[np.isnan(array_new)] = 0

    return array_new, weight_new


class StackMixin:
    """Mixin holding stacking orchestration logic for ContMap-like objects.

    Notes
    -----
    This mixin provides stack orchestration and the two built-in map
    generation methods: ``cutout2d`` and ``reproject``.
    """

    def _gen_cutouts(self, *args, method="cutout2d", **kwargs):
        """Dispatch cutout generation to the requested backend.

        Parameters
        ----------
        coords : `~astropy.coordinates.SkyCoord`
            Coordinates to extract or reproject around.
        size : `~astropy.units.Quantity` or sequence of `~astropy.units.Quantity`
            Angular size of each cutout.
        method : {"cutout2d", "reproject"}, optional
            Backend used to generate the cutouts.
        **kwargs
            Additional keyword arguments forwarded to the selected backend.

        Returns
        -------
        datas : ndarray
            Array of extracted data cutouts.
        weights : ndarray
            Array of extracted weight cutouts.
        wcs : `~astropy.wcs.WCS`
            Output WCS shared by the generated cutouts.

        Raises
        ------
        ValueError
            If ``method`` is not supported.
        """
        if method == "cutout2d":
            return self._gen_cutout2d(*args, **kwargs)
        if method == "reproject":
            return self._gen_reproject(*args, **kwargs)
        raise ValueError("method should be cutout2d or reproject")

    def _gen_cutout2d(self, coords, size, progress=False, **kwargs):
        """Generate simple 2D cutouts from a catalog of sky coordinates.

        Parameters
        ----------
        coords : `~astropy.coordinates.SkyCoord`
            Coordinates defining the centre of each cutout.
        size : `~astropy.units.Quantity` or sequence of `~astropy.units.Quantity`
            Angular size of the output cutout. A scalar quantity is applied to
            both axes. A two-element sequence sets ``(size_y, size_x)``.
        progress : bool, optional
            If True, use tqdm's process map to display a progress bar.
        **kwargs
            Unused placeholder for API compatibility.

        Returns
        -------
        data_cutouts : ndarray
            Array of data cutouts, one per input coordinate.
        weights_cutouts : ndarray
            Array of weight cutouts. Pixels that are NaN in the corresponding
            data cutout are forced to zero.
        output_wcs : `~astropy.wcs.WCS`
            WCS describing the common output cutout frame, centred at
            ``(0, 0)`` world coordinates.

        Raises
        ------
        ValueError
            If ``size`` has more than two elements or if its units are not
            angular.

        Notes
        -----
        The cutouts have an odd number of pixels along each axis and are
        centred on the pixel containing the requested coordinates.
        """
        size = np.atleast_1d(size)
        if len(size) == 1:
            size = np.repeat(size, 2)

        if len(size) > 2:
            raise ValueError("size must have at most two elements")

        pixel_scales = u.Quantity(
            [scale * u.Unit(unit) for scale, unit in zip(proj_plane_pixel_scales(self.wcs), self.wcs.wcs.cunit)]
        )

        shape = np.zeros(2).astype(int)
        for axis, side in enumerate(size):
            if side.unit.physical_type == "angle":
                shape[axis] = int(_round_up_to_odd_integer((side / pixel_scales[axis]).decompose()))
            else:
                raise ValueError("size must contains only Quantities with angular units")

        input_wcs = self.wcs
        input_array = self.__array__().filled(np.nan)

        chunksize = max(1, len(coords) // (cpu_count() * 10))  # heuristic for chunk size

        _ = partial(gen_cutout2d, data=input_array, shape=shape, wcs=input_wcs)

        if progress:
            from tqdm.contrib.concurrent import process_map
        else:
            from .utils import process_map

        data_cutouts = process_map(_, coords, chunksize=chunksize)
        data_cutouts = np.array(data_cutouts)

        weights = self.weights
        if np.any(self.mask):
            weights[self.mask] = 0

        _ = partial(gen_cutout2d, data=weights, shape=shape, wcs=input_wcs)
        if progress:
            from tqdm.contrib.concurrent import process_map

            weights_cutouts = process_map(_, coords, chunksize=chunksize)
        else:
            with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
                weights_cutouts = list(executor.map(_, coords, chunksize=chunksize))

        weights_cutouts = np.array(weights_cutouts)
        weights_cutouts[np.isnan(data_cutouts)] = 0

        output_wcs = Cutout2D(self, coords[0], shape, mode="partial").wcs
        output_wcs.wcs.crval = (0, 0)
        output_wcs.wcs.crpix = (shape - 1) / 2 + 1

        return data_cutouts, weights_cutouts, output_wcs

    def _gen_reproject(self, coords, size, type="interp", pixel_scales=None, progress=False, **kwargs):
        """Generate reprojected 2D cutouts from sky coordinates.

        Parameters
        ----------
        coords : `~astropy.coordinates.SkyCoord`
            Coordinates defining the centre of each reprojected cutout.
        size : `~astropy.units.Quantity` or sequence of `~astropy.units.Quantity`
            Angular size of the output cutout along each axis.
        type : {"interp", "adaptive", "exact"}, optional
            Reprojection algorithm to use.
        pixel_scales : `~astropy.units.Quantity` or sequence of `~astropy.units.Quantity`, optional
            Pixel scale of the output WCS. When omitted, the input map pixel
            scale is reused.
        progress : bool, optional
            If True, use tqdm's process map to display progress.
        **kwargs
            Unused placeholder for API compatibility.

        Returns
        -------
        data_cutouts : ndarray
            Reprojected data cutouts.
        weights_cutouts : ndarray
            Reprojected weight cutouts.
        output_wcs : `~astropy.wcs.WCS`
            WCS describing the common output cutout frame.

        Raises
        ------
        ValueError
            If ``size`` or ``pixel_scales`` have invalid dimensions, or if an
            unsupported reprojection type is requested.
        """
        if type.lower() == "interp":
            from reproject import reproject_interp as _reproject
        elif type.lower() == "adaptive":
            from reproject import reproject_adaptive

            _reproject = partial(reproject_adaptive, kernel="gaussian", boundary_mode="strict", conserve_flux=True)
        elif type.lower() == "exact":
            from reproject import reproject_exact as _reproject
        else:
            raise ValueError("Reprojection should be (``interp`` | ``adaptive`` | ``exact``)")

        size = np.atleast_1d(size)
        if len(size) == 1:
            size = np.repeat(size, 2)

        if len(size) > 2:
            raise ValueError("size must have at most two elements")

        if pixel_scales is None:
            pixel_scales = u.Quantity(
                [scale * u.Unit(unit) for scale, unit in zip(proj_plane_pixel_scales(self.wcs), self.wcs.wcs.cunit)]
            )
        else:
            pixel_scales = np.atleast_1d(pixel_scales)
            if len(pixel_scales) == 1:
                pixel_scales = np.repeat(pixel_scales, 2)

            if len(pixel_scales) > 2:
                raise ValueError("pixel_scale must have at most two elements")

        shape = np.zeros(2).astype(int)
        cdelt = np.zeros(2)
        for axis, side in enumerate(size):
            if side.unit.physical_type == "angle":
                cdelt[axis] = pixel_scales[axis].to(u.deg).value * np.sign(self.wcs.wcs.cdelt[axis])
                shape[axis] = int(_round_up_to_odd_integer((side / pixel_scales[axis]).decompose()))
            else:
                raise ValueError("size must contains only Quantities with angular units")

        output_wcs = WCS(naxis=2)
        output_wcs.wcs.ctype = self.wcs.wcs.ctype
        output_wcs.wcs.crpix = (shape - 1) / 2 + 1
        output_wcs.wcs.cdelt = cdelt

        input_array = self.__array__().filled(np.nan)
        input_weights = self.weights
        input_wcs = self.wcs

        chunksize = max(1, len(coords) // (cpu_count() * 10))  # heuristic for chunk size

        _ = partial(
            gen_reproject,
            output_wcs=output_wcs,
            data=input_array,
            weights=input_weights,
            shape=shape,
            wcs=input_wcs,
            func=_reproject,
        )

        if progress:
            from tqdm.contrib.concurrent import process_map
        else:
            from .utils import process_map

        results = process_map(_, coords, chunksize=chunksize)

        data_cutouts = np.array([result[0] for result in results])
        weights_cutouts = np.array([result[1] for result in results])
        del results

        output_wcs.wcs.crval = (0, 0)

        return np.array(data_cutouts), np.array(weights_cutouts), output_wcs

    def _gen_stack_output(self, datas, weights, wcs, ncoords, n_bootstrap=None):
        """Build a stacked map from cutout data and weights.

        Parameters
        ----------
        datas : ndarray
            Input cutout data cube with shape ``(ncoords, ny, nx)``.
        weights : ndarray
            Weight cube aligned with ``datas``.
        wcs : `~astropy.wcs.WCS`
            WCS describing the stacked cutout frame.
        ncoords : int
            Number of stacked coordinates, stored in the output history.
        n_bootstrap : int, optional
            Number of bootstrap realizations used to estimate uncertainties.
            If omitted, a weighted mean and inverse-variance uncertainty are
            computed directly.

        Returns
        -------
        object
            New instance of the map class containing the stacked data,
            uncertainty and metadata.
        """
        if np.any(np.isnan(datas)):
            nan_mask = np.isnan(datas)
            datas[nan_mask] = 0
            weights[nan_mask] = 0

        header = self.header.copy()
        header["HISTORY"] = "Stacked on {} coordinates".format(ncoords)

        if n_bootstrap is None:
            # np.ma.average handles 0-weights in the final map
            data, weight = np.ma.average(datas, weights=weights, axis=0, returned=True)
            uncertainty = InverseVariance(weight)
        else:
            # Quick and dirty bootstrap, not sure if this is the best way to do it but it works and is fast enough for now.
            # The idea is to shuffle the data and weights together and compute the average for each bootstrap sample.
            _ = partial(_shuffled_average, datas=datas, weights=weights)

            bs_array = ProgressBar.map(_, np.array_split(np.arange(n_bootstrap), cpu_count()), multiprocess=True)
            bs_array = np.concatenate(bs_array)

            data = np.mean(bs_array, axis=0)
            uncertainty = StdDevUncertainty(np.std(bs_array, axis=0, ddof=1))

        return self.__class__(
            data,
            mask=np.isnan(data),
            uncertainty=uncertainty,
            unit=self.unit,
            wcs=wcs,
            meta=header,
        )

    def stack(self, coords, size, method="cutout2d", n_bootstrap=None, **kwargs):
        """Stack cutouts centred on the provided coordinates.

        Parameters
        ----------
        coords : `~astropy.coordinates.SkyCoord`
            Coordinates to stack.
        size : `~astropy.units.Quantity` or sequence of `~astropy.units.Quantity`
            Angular size of each cutout.
        method : {"cutout2d", "reproject"}, optional
            Cutout generation backend.
        n_bootstrap : int, optional
            Number of bootstrap realizations used to estimate the stack
            uncertainty.
        **kwargs
            Additional keyword arguments forwarded to the selected cutout
            backend. For the ``reproject`` method this includes options such
            as ``pixel_scales`` and the reprojection ``type``.

        Returns
        -------
        object
            Stacked map of the same class as ``self``. The output keeps the
            parent map unit and records the number of stacked coordinates in
            the FITS header history.
        """
        datas, weights, wcs = self._gen_cutouts(coords, size, method=method, **kwargs)
        return self._gen_stack_output(datas, weights, wcs, len(coords), n_bootstrap=n_bootstrap)

    def simstack(self, coords, kernel=None, add_offset=False, fast=True, n_bootstrap=None, **kwargs):
        """Return the mean flux and error of populations of sources using the simstack method.

        Parameters
        ----------
        coords : SkyCoords or list of SkyCoords
            One group (single coefficient) or several groups (multi-coefficient).
        kernel : Kernel2D, optional
            Beam/kernel used to build the hit maps, default to the image beam.
        add_offset : bool, optional
            If True, include a constant offset term in the regression.
        fast : bool, optional
            If True, build hit maps with `make_model_image_fft`.
            If False, build hit maps with `make_model_image` using `kernel.model`.
        n_bootstrap : int, optional
            If provided, estimate the coefficient uncertainties from bootstrap
            realizations of the valid map pixels instead of the analytic
            covariance matrix.
        **kwargs
            Reserved for future extensions. Currently unused.

        Returns
        -------
        res : ndarray
            Mean flux density recovered for each source population. When
            ``add_offset`` is True, the constant offset is appended as the
            last coefficient.
        err : ndarray
            One-sigma uncertainty on each returned coefficient.
        """
        if kernel is None:
            # Default to the beam
            kernel = self.beam.as_kernel((1 * u.pix).to("arcsec", equivalencies=self._pixel_scale))

        weight = None
        if self.uncertainty is not None:
            weight = self.uncertainty.represent_as(InverseVariance).array

        simstack_args = (self.data, self.wcs, kernel)
        simstack_kwargs = {"mask": self.mask, "weight": weight, "add_offset": add_offset, "fast": fast}

        X, y, w = _prepare_simstack(*simstack_args, coords, **simstack_kwargs)

        if n_bootstrap is None:
            # np.ma.average handles 0-weights in the final map
            res, err = _solve_simstack(X, y, w=w)
        else:
            # Quick and dirty bootstrap, not sure if this is the best way to do it but it works and is fast enough for now.
            # The idea is to shuffle the data and weights together and compute the average for each bootstrap sample.

            _ = partial(_shuffle_pixel_solve_simstack, X=X, y=y, w=w)
            bs_array = ProgressBar.map(_, range(n_bootstrap), multiprocess=True)
            bs_array = np.array(bs_array)

            res = np.mean(bs_array, axis=0)
            err = np.std(bs_array, axis=0, ddof=len(coords) + (1 if add_offset else 0))
        return res, err