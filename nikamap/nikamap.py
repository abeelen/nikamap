from __future__ import absolute_import, division, print_function

from itertools import product
from collections import MutableMapping
from pathlib import Path

import numpy as np
from copy import deepcopy

from astropy.io import fits, registry
from astropy import units as u
from astropy.wcs import WCS, InconsistentAxisTypesError
from astropy.coordinates import match_coordinates_sky
from astropy.nddata import NDDataArray, NDUncertainty
from astropy.nddata import StdDevUncertainty, InverseVariance, VarianceUncertainty
from astropy.modeling import models
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.convolution import Kernel2D, Box2DKernel, Gaussian2DKernel

from astropy.table import Table, MaskedColumn, Column

from astropy.nddata.ccddata import _known_uncertainties, _uncertainty_unit_equivalent_to_parent
from astropy.nddata.ccddata import _unc_name_to_cls, _unc_cls_to_name

import photutils
from photutils.psf import BasicPSFPhotometry
from photutils.psf import DAOGroup
from photutils.background import MedianBackground
from photutils.datasets import make_gaussian_sources_image
from photutils.centroids import centroid_2dg, centroid_com

from radio_beam import Beam

from scipy import signal
from scipy.optimize import curve_fit

import warnings
from astropy.utils.exceptions import AstropyWarning

from .utils import CircularGaussianPSF, _round_up_to_odd_integer
from .utils import pos_uniform, cat_to_sc
from .utils import powspec_k
from .utils import update_header
from .utils import shrink_mask
from .utils import setup_ax

Jy_beam = u.Jy / u.beam

__all__ = ["NikaBeam", "NikaMap", "NikaFits"]


class NikaBeam(Kernel2D):
    """NikaBeam describe the beam of a NikaMap.

    By default the beams are gaussian function, but the class should be able to handle arbitrary beam. It also add an internal pixel_scale which allow for :class:`astropy.units.Quantity` arguments

    Parameters
    ----------
    fwhm : :class:`astropy.units.Quantity`
        Full width half maximum of the Gaussian kernel.
    pixel_scale : `astropy.units.equivalencies.pixel_scale`
        The pixel scale either in units of angle/pixel or pixel/angle.

    See also
    --------
    :class:`astropy.convolution.Gaussian2DKernel`

    """

    def __init__(self, fwhm=None, pixel_scale=None, **kwargs):
        self._pixel_scale = pixel_scale
        self._fwhm = fwhm

        if kwargs.get("array", None) is None:
            stddev = gaussian_fwhm_to_sigma * fwhm.to(u.pixel, equivalencies=self._pixel_scale).value
            self._model = models.Gaussian2D(1.0, 0, 0, stddev, stddev)
            self._default_size = _round_up_to_odd_integer(8 * stddev)

        super(NikaBeam, self).__init__(**kwargs)
        self._truncation = np.abs(1.0 - self._array.sum())

    def __repr__(self):
        return "<NikaBeam(fwhm={}, pixel_scale={:.2f} / pixel)".format(
            self._fwhm.to(u.arcsec), (1 * u.pixel).to(u.arcsec, equivalencies=self._pixel_scale)
        )

    @property
    def fwhm(self):
        warnings.warn("fwhm is deprecated, use major/minor", DeprecationWarning)
        return self._fwhm

    @property
    def fwhm_pix(self):
        warnings.warn("fwhm_pix is deprecated, use major/minor.to(u.pixel, u=pixel_scale)", DeprecationWarning)
        return self._fwhm.to(u.pixel, equivalencies=self._pixel_scale)

    @property
    def major(self):
        return self._fwhm

    @property
    def minor(self):
        return self._fwhm

    @property
    def sigma(self):
        warnings.warn("sigma is deprecated, use major/minor * gaussian_fwhm_to_sigma", DeprecationWarning)
        return self._fwhm * gaussian_fwhm_to_sigma

    @property
    def sigma_pix(self):
        warnings.warn(
            "sigma is deprecated, use major/minor.to(u.pixel, u=pixel_scale)* gaussian_fwhm_to_sigma",
            DeprecationWarning,
        )
        return self._fwhm.to(u.pixel, equivalencies=self._pixel_scale) * gaussian_fwhm_to_sigma

    @property
    def area(self):
        warnings.warn("area is deprecated, use sr", DeprecationWarning)
        return 2 * np.pi * (self._fwhm * gaussian_fwhm_to_sigma) ** 2

    @property
    def sr(self):
        return 2 * np.pi * (self._fwhm * gaussian_fwhm_to_sigma) ** 2

    @property
    def area_pix(self):
        warnings.warn("area_pix is deprecated, use np.sqrt(self.sr).to(u.pix, pixel_scale)**2", DeprecationWarning)
        return 2 * np.pi * self.sigma_pix ** 2

    def convolve(self, kernel):

        if isinstance(kernel, Beam):
            pixscale = (1 * u.pix).to(u.arcsec, self._pixel_scale)
            kernel = kernel.as_kernel(pixscale)

        kernel.normalize("integral")

        # Assuming the same pixel_scale
        if isinstance(self.model, models.Gaussian2D) & isinstance(kernel.model, models.Gaussian2D):
            fwhm = np.sqrt(self.model.x_fwhm ** 2 + kernel.model.x_fwhm ** 2) * u.pixel
            fwhm = fwhm.to(u.arcsec, equivalencies=self._pixel_scale)
            return NikaBeam(fwhm, pixel_scale=self._pixel_scale)
        else:
            # Using scipy.signal.convolve to extend the beam if necessary
            return NikaBeam(array=signal.convolve(self.array, kernel.array), pixel_scale=self._pixel_scale)


# TODO: Take care of operations (add/subtract/...) to add extra parameters...


class NikaMap(NDDataArray):
    """A NikaMap object represent a nika map with additionnal capabilities.

    It contains the metadata, wcs, and all attribute (data/stddev/time/unit/mask) as well as potential source list detected in these maps.

    Parameters
    ----------
    data : :class:`~numpy.ndarray` or :class:`astropy.nddata.NDData`
        The actual data contained in this `NDData` object. Not that this
        will always be copies by *reference* , so you should make copy
        the ``data`` before passing it in if that's the  desired behavior.
    uncertainty : :class:`astropy.nddata.NDUncertainty`, optional
        Uncertainties on the data.
    mask : :class:`~numpy.ndarray`-like, optional
        Mask for the data, given as a boolean Numpy array or any object that
        can be converted to a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If ``data`` is a numpy masked array, providing
        ``mask`` here will causes the mask from the masked array to be
        ignored.
    hits : :class:`~numpy.ndarray`-like, optional
        The hit per pixel on the map
    sampling_freq : float or :class:`~astropy.units.Quantity`
        the sampling frequency of the experiment, default 1 Hz
    wcs : undefined, optional
        WCS-object containing the world coordinate system for the data.
    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.
    unit : :class:`astropy.units.UnitBase` instance or str, optional
        The units of the data.
    beam : :class:`radio_beam.Beam`
        The beam corresponding to the data, by default a gaussian
        constructed from the header 'BMAJ' 'BMIN', 'PA' keyword.
    fake_source : :class:`astropy.table.Table`, optional
        The table of potential fake sources included in the data

        .. note::
            The table must contain at least 3 columns: ['ID', 'ra', 'dec']

    sources : :class`astropy.table.Table`, optional
        The table of detected sources in the data.

    """

    def __init__(self, data, *args, **kwargs):

        if "meta" not in kwargs:
            kwargs["meta"] = kwargs.pop("header", None)
        if "header" in kwargs:
            raise ValueError("can't have both header and meta.")

        # Arbitrary unit by default
        if "unit" not in kwargs:
            kwargs["unit"] = data.unit if isinstance(data, (u.Quantity, NikaMap)) else "adu"

        # Must be set AFTER the super() call
        self.primary_header = kwargs.pop("primary_header", None)
        self.fake_sources = kwargs.pop("fake_sources", None)
        self.sources = kwargs.pop("sources", None)
        self.sampling_freq = kwargs.pop("sampling_freq", None)

        self.hits = kwargs.pop("hits", None)
        self.beam = kwargs.pop("beam", None)

        super().__init__(data, *args, **kwargs)

        if isinstance(data, NikaMap):
            if self.hits is None and data.hits is not None:
                self.hits = data.hits
            if self.beam is None and data.beam is not None:
                self.beam = data.beam

        if isinstance(self.wcs, WCS):
            pixsize = np.abs(self.wcs.wcs.cdelt[0]) * u.deg
        else:
            pixsize = np.abs(self.meta.get("CDELT1", 1)) * u.deg

        self._pixel_scale = u.pixel_scale(pixsize / u.pixel)

        if self.beam is None:
            # Default BMAJ 1 deg...
            header = self.meta
            if "BMAJ" not in header:
                header["BMAJ"] = 1
            self.beam = Beam.from_fits_header(fits.Header(header))

    @property
    def header(self):
        return self._meta

    @header.setter
    def header(self, value):
        self.meta = value

    @property
    def time(self):
        if self.hits is not None and self.sampling_freq is not None:
            return (self.hits / self.sampling_freq).to(u.s)
        else:
            return None

    def compressed(self):
        return self.data[~self.mask] * self.unit

    def uncertainty_compressed(self):
        return self.uncertainty.array[~self.mask] * self.uncertainty.unit

    def __array__(self):
        """
        This allows code that requests a Numpy array to use an NDData
        object as a Numpy array.

        Notes
        -----
        Overrite NDData.__array__ to force for MaskedArray output
        """
        return np.ma.array(self.data, mask=self.mask)

    def __u_array__(self):
        """Retrieve uncertainty array as masked array"""
        return np.ma.array(self.uncertainty.array, mask=self.mask)

    def __t_array__(self):
        """Retrieve hit array as maskedQuantity"""
        return np.ma.array(self.time, mask=self.mask, fill_value=0)

    def surface(self, box_size=None):
        """Retrieve surface covered by unmasked pixels
        Parameters
        ----------
        box_size : scalar or tuple, optional
            The edge of the map is cropped by the box_size if not None.
            Default is None.

        Returns
        -------
        :class:`astropy.units.Quantity`
            Surface covered by unmasked pixels

        Notes
        -------
            Default value for box_size in detect_sources is 5"""

        nvalid = np.prod(self.data.shape)

        if self.mask is not None:
            mask = self.mask
            if box_size is not None:
                box_kernel = Box2DKernel(box_size)
                mask = shrink_mask(mask, box_kernel)

            nvalid = np.sum(~mask)

        conversion = (u.pix.to(u.arcsec, equivalencies=self._pixel_scale)) ** 2

        return nvalid * conversion * u.arcsec ** 2

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            # Ugly trick to overcome bug in NDDataArray uncertainty setter
            unit = getattr(value, "unit", None)
            _class = value.__class__
            if isinstance(value, (np.ndarray, u.Quantity)):
                if self.unit and unit:
                    value = value.to(self.unit).value
                _class = StdDevUncertainty
            elif isinstance(value, NDUncertainty):
                # Ugly trick to overcome bug in NDDataArray uncertainty setter
                if self.unit and unit:
                    value = (value.array * unit).to(self.unit).value
            else:
                raise TypeError("uncertainty must be an instance of a NDUncertainty object or a numpy array.")

            value = _class(value, unit=None)
            if value.array is not None and value.array.shape != self.shape:
                raise ValueError("uncertainty must have same shape as data.")

            NDDataArray.uncertainty.__set__(self, value)
        else:
            self._uncertainty = value

    @property
    def snr(self):
        if isinstance(self.uncertainty, InverseVariance):
            snr = self.data * np.sqrt(self.uncertainty.array)
        elif isinstance(self.uncertainty, StdDevUncertainty):
            snr = self.data / self.uncertainty.array
        elif isinstance(self.uncertainty, VarianceUncertainty):
            snr = self.data / np.sqrt(self.uncertainty.array)
        else:
            raise ValueError("Unknown uncertainty type")

        return np.ma.array(snr, mask=self.mask)

    def _to_ma(self, item=None):
        """Get masked array quantities from object.

        Parameters
        ----------
        item : str, optional (None|signal,uncertainty|snr)
            The quantity to retrieve, by default None, ie signal

        Returns
        -------
        data : ~np.ma.MaskedArray
            The corresponding item as masked quantity
        label : str
            The corresponding label

        Raises
        ------
        ValueError
            When item is not in list
        """

        if item == "snr":
            data = self.snr
            label = "SNR"
        elif item == "uncertainty":
            data = np.ma.array(self.uncertainty.array * self.unit, mask=self.mask)
            label = "Uncertainty"
        elif item in ["signal", None]:
            data = np.ma.array(self.data * self.unit, mask=self.mask)
            label = "Brightness"
        else:
            raise ValueError("must be in (None|signal|uncertainty|snr)")

        return data, label

    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, value):
        if value is None or isinstance(value, Beam):
            self._beam = value
        elif isinstance(value, NikaBeam):
            warnings.warn("Using deprecated NikaBeam", DeprecationWarning)
            self._beam = value
        else:
            raise ValueError("Can not handle this beam type {}".format(type(value)))

    def _slice(self, item):
        # slice all normal attributes
        kwargs = super(NikaMap, self)._slice(item)
        # The arguments for creating a new instance are saved in kwargs
        # so we need to add another keyword "flags" and add the sliced flags
        kwargs["hits"] = self.hits[item] if self.hits is not None else None
        kwargs["beam"] = self.beam

        kwargs["fake_sources"] = self.fake_sources
        kwargs["sources"] = self.sources

        return kwargs  # these must be returned

    def trim(self):
        """Remove masked region on the edges

        Returns
        -------
        :class:`NikaMap`
            return a trimmed NikaMap object

        """
        mask = self.mask
        axis_slice = []
        for axis in [1, 0]:
            good_pix = np.argwhere(np.mean(mask, axis=axis) != 1)
            axis_slice.append(slice(np.min(good_pix), np.max(good_pix) + 1))

        output = self[axis_slice[0], axis_slice[1]]
        return output

    def add_gaussian_sources(self, within=(0, 1), cat_gen=pos_uniform, **kwargs):
        """Add gaussian sources into the map.

        Parameters
        ----------
        within : tuple of 2 int
            force the sources within this relative range in the map
        cat_gen : function (`pos_uniform`|`pos_gridded`|`pos_list`|...)
            the function used to generate the pixel positions and flux of the sources (see Notes below)
        **kwargs
            any keyword arguments to be passed to the `cat_gen` function

        Notes
        -----
        the `cat_gen` function is used to generate the list of x, y pixel positions and fluxes
        and must at least support the `shape=None, within=(0, 1), mask=None` arguments.
        """
        shape = self.shape

        x_mean, y_mean, peak_flux = cat_gen(shape=shape, within=within, mask=self.mask, **kwargs)

        nsources = x_mean.shape[0]

        sources = Table(masked=True)

        sources["amplitude"] = peak_flux.to(self.unit * u.beam)

        sources["x_mean"] = x_mean
        sources["y_mean"] = y_mean

        sources["x_stddev"] = (
            np.ones(nsources) * self.beam.major.to(u.pix, self._pixel_scale).value * gaussian_fwhm_to_sigma
        )
        sources["y_stddev"] = (
            np.ones(nsources) * self.beam.minor.to(u.pix, self._pixel_scale).value * gaussian_fwhm_to_sigma
        )
        sources["theta"] = np.zeros(nsources)

        # Crude check to be within the finite part of the map
        if self.mask is not None:
            within_coverage = ~self.mask[sources["y_mean"].astype(int), sources["x_mean"].astype(int)]
            sources = sources[within_coverage]

        # Gaussian sources...
        self._data += make_gaussian_sources_image(shape, sources)

        # Add an ID column
        sources.add_column(Column(np.arange(len(sources)), name="fake_id"), 0)

        # Transform pixel to world coordinates
        a, d = self.wcs.pixel_to_world_values(sources["x_mean"], sources["y_mean"])
        sources.add_columns([Column(a * u.deg, name="ra"), Column(d * u.deg, name="dec")])

        sources["_ra"] = sources["ra"]
        sources["_dec"] = sources["dec"]

        # Remove unnecessary columns
        sources.remove_columns(["x_mean", "y_mean", "x_stddev", "y_stddev", "theta"])

        self.fake_sources = sources

    def detect_sources(self, threshold=3, box_size=5):
        """Detect sources with find local peaks above a specified threshold value.

        The detection is made on the SNR map, and return an :class`astropy.table.Table` with columns ``ID, ra, dec, SNR``.
        If fake sources are present, a match is made with a distance threshold of ``beam_fwhm / 3``

        Parameters
        ----------
        threshold : float
            The data value or pixel-wise data values to be used for the
            detection threshold.
        box_size : scalar or tuple, optional
            The size of the local region to search for peaks at every point
            in ``data``.  If ``box_size`` is a scalar, then the region shape
            will be ``(box_size, box_size)``.

        Notes
        -----
        The edge of the map is cropped by the box_size in order to insure proper subpixel fitting.
        """
        detect_on = self.snr.filled(0)

        if self.mask is not None:
            # Make sure that there is no detection on the edge of the map
            box_kernel = Box2DKernel(box_size)
            detect_mask = shrink_mask(self.mask, box_kernel)
            detect_on[detect_mask] = np.nan

        # TODO: Have a look at
        # ~photutils.psf.IterativelySubtractedPSFPhotometry

        # See #667 of photutils
        try:
            # To avoid bad fit warnings...
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                sources = photutils.find_peaks(
                    detect_on,
                    threshold=threshold,
                    mask=self.mask,
                    wcs=self.wcs,
                    centroid_func=centroid_2dg,  # or centroid_com for faster/less precise values (see phoutils #655)
                    box_size=box_size,
                )
        except TypeError:  # See #1295 of photutils
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                sources = photutils.find_peaks(
                    detect_on,
                    threshold=threshold,
                    mask=self.mask,
                    wcs=self.wcs,
                    centroid_func=centroid_com,  # or centroid_com for faster/less precise values (see phoutils #655)
                    box_size=box_size,
                )
        except InconsistentAxisTypesError:
            sources = []

        if sources is not None and len(sources) > 0:
            # Transform to masked Table here to avoid future warnings
            sources = Table(sources, masked=True)
            sources.meta["method"] = "find_peak"
            sources.meta["threshold"] = threshold

            if "skycoord_centroid" in sources.colnames:
                # If the centroid was properly computed
                # Extract skycoord centroid positions
                sources["ra"] = sources["skycoord_centroid"].ra
                sources["dec"] = sources["skycoord_centroid"].dec
                sources.remove_columns(["x_centroid", "y_centroid", "x_peak", "y_peak"])
                sources.remove_columns(["skycoord_centroid", "skycoord_peak"])
            elif "skycoord_peak" in sources.colnames:
                # The centroid where not computed...
                # Extract skycoord from peak position...
                sources["ra"] = sources["skycoord_peak"].ra
                sources["dec"] = sources["skycoord_peak"].dec
                sources.remove_columns(["x_peak", "y_peak"])
                sources.remove_columns(["skycoord_peak"])
            else:
                raise ValueError("No centroid nor peak positions found")

            # For compatibility issues
            sources["_ra"] = sources["ra"]
            sources["_dec"] = sources["dec"]

            # Sort by decreasing SNR
            sources.rename_column("peak_value", "SNR")
            sources.sort("SNR")
            sources.reverse()

            sources.add_column(Column(np.arange(len(sources)), name="ID"), 0)

        if self.fake_sources:
            # Match to the fake catalog
            fake_sources = self.fake_sources
            dist_threshold = self.beam.major / 3

            if sources is None or len(sources) == 0:
                fake_sources["find_peak"] = MaskedColumn(np.ones(len(fake_sources), dtype=np.int), mask=True)
            else:

                fake_sc = cat_to_sc(fake_sources)
                sources_sc = cat_to_sc(sources)

                idx, sep2d, _ = match_coordinates_sky(fake_sc, sources_sc)
                mask = sep2d > dist_threshold
                fake_sources["find_peak"] = MaskedColumn(sources[idx]["ID"], mask=mask)

                idx, sep2d, _ = match_coordinates_sky(sources_sc, fake_sc)
                mask = sep2d > dist_threshold
                sources["fake_id"] = MaskedColumn(fake_sources[idx]["fake_id"], mask=mask)

        if sources is not None and len(sources) > 0:
            self.sources = sources
        else:
            self.sources = None

    def match_sources(self, catalogs, dist_threshold=None):

        if dist_threshold is None:
            dist_threshold = self.beam.major / 3

        if not isinstance(catalogs, list):
            catalogs = [catalogs]

        for cat, ref_cat in product([self.sources], catalogs):
            cat_sc = cat_to_sc(cat)
            ref_sc = cat_to_sc(ref_cat)
            idx, sep2d, _ = match_coordinates_sky(cat_sc, ref_sc)
            mask = sep2d > dist_threshold
            cat[ref_cat.meta["name"]] = MaskedColumn(idx, mask=mask)

    def phot_sources(self, sources=None, peak=True, psf=True):

        if sources is None:
            sources = self.sources

        xx, yy = self.wcs.world_to_pixel_values(sources["ra"], sources["dec"])

        x_idx = np.floor(xx + 0.5).astype(int)
        y_idx = np.floor(yy + 0.5).astype(int)

        if peak:
            # Crude Peak Photometry
            # From pixel indexes to array indexing

            sources["flux_peak"] = Column(self.data[y_idx, x_idx], unit=self.unit * u.beam).to(u.mJy)
            sources["eflux_peak"] = Column(self.uncertainty.array[y_idx, x_idx], unit=self.unit * u.beam).to(u.mJy)

        if psf:
            # BasicPSFPhotometry with fixed positions

            sigma_psf = self.beam.major.to(u.pix, self._pixel_scale).value * gaussian_fwhm_to_sigma

            # Using an IntegratedGaussianPRF can cause biais in the photometry
            # TODO: Check the NIKA2 calibration scheme
            # from photutils.psf import IntegratedGaussianPRF
            # psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
            psf_model = CircularGaussianPSF(sigma=sigma_psf)

            psf_model.x_0.fixed = True
            psf_model.y_0.fixed = True

            daogroup = DAOGroup(3 * self.beam.major.to(u.pix, self._pixel_scale).value)
            mmm_bkg = MedianBackground()

            photometry = BasicPSFPhotometry(
                group_maker=daogroup, bkg_estimator=mmm_bkg, psf_model=psf_model, fitter=LevMarLSQFitter(), fitshape=9
            )

            positions = Table(
                [Column(xx, name="x_0"), Column(yy, name="y_0"), Column(self.data[y_idx, x_idx], name="flux_0")]
            )

            # Fill the mask with nan to perform correct photometry on the edge
            # of the mask, and catch numpy & astropy warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", AstropyWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                result_tab = photometry(
                    image=np.ma.array(self.data, mask=self.mask).filled(np.nan), init_guesses=positions
                )

            result_tab.sort("id")
            for _source, _tab in zip(["flux_psf", "eflux_psf"], ["flux_fit", "flux_unc"]):
                # Sometimes the returning fluxes has no uncertainty....
                if _tab in result_tab.colnames:
                    sources[_source] = Column(result_tab[_tab] * psf_model(0, 0), unit=self.unit * u.beam).to(u.mJy)
            sources["group_id"] = result_tab["group_id"]

        self.sources = sources

    def match_filter(self, kernel):
        """Return a matched filtered version of the map.

        Parameters
        ----------
        kernel : :class:`radio_beam.Beam` or :class:`astropy.convolution.Gaussian2Dkernel` or :class:`nikamap.NikaBeam`
            the kernel used for filtering

        Returns
        -------
        :class:`nikamap.NikaMap`
            the resulting match filtered nikamap object


        Notes
        -----
        This compute the match filtered :math:`MF` map as :

        .. math::

            MF = \\frac{B * (W M)}{B^2 * W}

        with :math:`B` the beam, :math:`W` the weights (inverse variance) and :math:`M` the signal map

        Peak photometry is conserved for data and e_data

        Resultings maps have a different mask

        >>> npix, std = 500, 4
        >>> kernel = Gaussian2DKernel(std)
        >>> mask = np.zeros((npix,npix))
        >>> data = np.random.normal(0, 1, size=mask.shape)
        >>> data[(npix-std*8)//2:(npix+std*8)//2+1,(npix-std*8)//2:(npix+std*8)//2+1] += kernel.array/kernel.array.max()
        >>> data = NikaMap(data, uncertainty=StdDevUncertainty(np.ones_like(data)), time=np.ones_like(data)*u.s, mask=mask)
        >>> mf_data = data.match_filter(kernel)
        >>> import matplotlib.pypot as plt
        >>> plt.ion()
        >>> fig, axes = plt.subplots(ncols=2)
        >>> axes[0].imshow(data) ; axes[1].imshow(mf_data)
        """

        if isinstance(kernel, Gaussian2DKernel):
            bmaj = (kernel.model.x_stddev.value * u.pix).to(u.arcsec, self._pixel_scale)
            bmin = (kernel.model.y_stddev.value * u.pix).to(u.arcsec, self._pixel_scale)
            pa = kernel.model.theta.value * u.radian
            kernel = Beam(bmaj, bmin, pa)
        elif isinstance(kernel, NikaBeam) and isinstance(kernel.model, models.Gaussian2D):
            bmaj = kernel.fwhm
            kernel = Beam(bmaj)
        elif isinstance(kernel, Kernel2D):
            pixscale = (1 * u.pix).to(u.arcsec, self._pixel_scale)
            beam = self.beam.as_kernel(pixscale)
            mf_beam = NikaBeam(array=signal.convolve(beam.array, kernel.array), pixel_scale=self._pixel_scale)
        elif not isinstance(kernel, Beam):
            raise ValueError("Can not handle this kernel type yet")

        if isinstance(kernel, Beam):
            mf_beam = self.beam.convolve(kernel)
            pixscale = (1 * u.pix).to(u.arcsec, self._pixel_scale)
            kernel = kernel.as_kernel(pixscale)

        kernel.normalize("integral")

        # Convolve the mask and retrieve the fully sampled region, this
        # will remove one kernel width on the edges
        # mf_mask = ~np.isclose(convolve(~self.mask, kernel, normalize_kernel=False), 1)
        if self.mask is not None:
            mf_mask = shrink_mask(self.mask, kernel)
        else:
            mf_mask = None

        # Convolve the time (integral for time)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        #     mf_time = convolve(self.time, kernel, normalize_kernel=False)*self.time.unit
        if self.hits is not None:
            mf_hits = signal.fftconvolve(np.asarray(self.hits), kernel, mode="same")
            if mf_mask is not None:
                mf_hits[mf_mask] = 0
        else:
            mf_hits = None

        # Convolve the data (peak for unit conservation)
        kernel.normalize("peak")
        kernel_sqr = kernel.array ** 2

        # ma.filled(0) required for the fft convolution
        weights = 1.0 / self.uncertainty.array ** 2
        if self.mask is not None:
            weights[self.mask] = 0

        with np.errstate(invalid="ignore", divide="ignore"):
            mf_uncertainty = 1 / np.sqrt(signal.fftconvolve(weights, kernel_sqr, mode="same"))
        if mf_mask is not None:
            mf_uncertainty[mf_mask] = np.nan

        # Units are not propagated in masked arrays...
        mf_data = signal.fftconvolve(weights * self.__array__().filled(0), kernel, mode="same") * mf_uncertainty ** 2

        mf_data = NikaMap(
            mf_data,
            mask=mf_mask,
            hits=mf_hits,
            uncertainty=StdDevUncertainty(mf_uncertainty),
            beam=mf_beam,
            unit=self.unit,
            sampling_freq=self.sampling_freq,
            wcs=self.wcs,
            meta=self.meta,
            primary_header=self.primary_header,
            fake_sources=self.fake_sources,
        )

        return mf_data

    def plot(self, to_plot=None, ax=None, cbar=False, cat=None, levels=None, **kwargs):
        """Convenience routine to plot the dataset.

        Parameters
        ----------
        to_plot : str, optionnal (snr|uncertainty|None)
            Choose which quantity to plot, by default None (signal)
        ax : :class:`matplotlib.axes.Axes`, optional
            Axe to plot the power spectrum
        cbar: boolean, optionnal
            Draw a colorbar (ax must be None)
        cat : boolean of list of tuple [(cat, kwargs)], optionnal
            If True, overplot the current self.source catalog
            with '^' as marker.
            Otherwise overplot the given catalogs on the map, with kwargs.
        levels: array_like, optionnal
            Overplot levels contours, add negative contours as dashed line
        **kwargs
            Arbitrary keyword arguments for :func:`matplotib.pyplot.imshow `

        Returns
        -------
        image : `~matplotlib.image.AxesImage`

        Notes
        -----
        * if a fake_sources property is present, it will be overplotted with 'o' as marker
        * each catalog *must* have '_ra' & '_dec' column

        """
        try:
            data, cbar_label = self._to_ma(item=to_plot)
        except ValueError as e:
            raise ValueError("to_plot {}".format(e))

        if isinstance(data.data, u.Quantity):
            # Remove unit to avoid matplotlib problems
            data = np.ma.array(data.data.to(self.unit).value, mask=data.mask)
            cbar_label = "{} [{}]".format(cbar_label, self.unit)

        ax = setup_ax(ax, self.wcs)

        iax = ax.imshow(data, origin="lower", interpolation="none", **kwargs)

        if levels is not None:
            ax.contour(data, levels=levels, alpha=0.8, colors="w")
            ax.contour(data, levels=-levels[::-1], alpha=0.8, colors="w", linestyles="dashed")

        if cbar:
            fig = ax.get_figure()
            cbar = fig.colorbar(iax, ax=ax)
            cbar.set_label(cbar_label)

        if cat is True and self.sources is not None:
            cat = [(self.sources, {"marker": "^", "color": "red"})]
        else:
            cat = []

        # In case of fake sources, overplot them
        if self.fake_sources:
            fake_cat = [(self.fake_sources, {"marker": "o", "c": "red", "alpha": 0.8})]
            cat += fake_cat

        if cat != []:
            for _cat, _kwargs in list(cat):
                label = _cat.meta.get("method") or _cat.meta.get("name") or _cat.meta.get("NAME") or "Unknown"
                cat_sc = cat_to_sc(_cat)
                x, y = self.wcs.world_to_pixel_values(cat_sc.ra, cat_sc.dec)
                if _kwargs is None:
                    _kwargs = {"alpha": 0.8}
                ax.scatter(x, y, **_kwargs, label=label)

        ax.set_xlim(0, self.shape[1])
        ax.set_ylim(0, self.shape[0])

        return iax

    def plot_SNR(self, vmin=-3, vmax=5, **kwargs):
        """Convenience method to plot the signal to noise map.

        Notes
        -----
        See :func:`nikamap.plot`for additionnal keywords
        """
        return self.plot(to_plot="snr", vmin=vmin, vmax=vmax, **kwargs)

    def check_SNR(self, ax=None, bins=100):
        """Perform normality test on SNR map.

        Parameters
        ----------
        ax : :class:`~matplotlib.axes.Axes`, optional
            axe to plot the histogram and fits
        bins: int
            number of bins for the histogram. Default 100.

        Returns
        -------
        std : float
            return the robust standard deviation of the SNR

        Notes
        -----
        To recover the normality you must multiply the uncertainty array by the returned stddev value

        >>> std = data.check_SNR()
        >>> data.uncertainty.array *= std
        """
        SN = self.snr.compressed()
        hist, bin_edges = np.histogram(SN, bins=bins, density=True, range=(-5, 5))

        # is biased if signal is present
        # is biased if trimmed
        # mu, std = norm.fit(SN)

        bin_center = (bin_edges[1:] + bin_edges[:-1]) / 2

        # Clip to 3 sigma, this will biais the result
        robust = (-6 < bin_center) & (bin_center < 3)

        def gauss(x, a, c, s):
            return a * np.exp(-((x - c) ** 2) / (2 * s ** 2))

        popt, pcov = curve_fit(gauss, bin_center[robust].astype(np.float32), hist[robust].astype(np.float32))
        mu, std = popt[1:]

        if ax is not None:
            ax.plot(bin_center, hist, drawstyle="steps-mid")
            ax.plot(bin_center, gauss(bin_center, *popt))

        return std

    def plot_PSD(self, to_plot=None, ax=None, bins=100, range=None, apod_size=None, **kwargs):
        """Plot the power spectrum of the map.

        Parameters
        ----------
        to_plot : str, optionnal (snr|uncertainty|signal|None)
            Choose which quantity to plot, by default None (signal)
        ax : :class:`matplotlib.axes.Axes`, optional
            Axe to plot the power spectrum
        bins : int
            Number of bins for the histogram. Default 100.
        range : (float, float), optional
            The lower and upper range of the bins. (see `~numpy.histogram`)


        Returns
        -------
        powspec_k : :class:`astropy.units.quantity.Quantity`
            The value of the power spectrum
        bin_edges : :class:`astropy.units.quantity.Quantity`
            Return the bin edges ``(length(hist)+1)``.
        """
        try:
            data, label = self._to_ma(item=to_plot)
        except ValueError as e:
            raise ValueError("to_plot {}".format(e))

        res = (1 * u.pixel).to(u.arcsec, equivalencies=self._pixel_scale)
        powspec, bin_edges = powspec_k(data, res=res, bins=bins, range=range, apod_size=apod_size)

        if to_plot == "snr":
            powspec /= res ** 2
        else:
            pk_unit = u.Jy ** 2 / u.sr
            powspec /= (self.beam.sr / u.beam) ** 2
            powspec = powspec.to(pk_unit)
            label = "P(k) {} [{}]".format(label, pk_unit)

        if ax is not None:
            bin_center = (bin_edges[1:] + bin_edges[:-1]) / 2
            ax.loglog(bin_center, powspec, **kwargs)
            ax.set_xlabel(r"k [arcsec$^{-1}$]")
            ax.set_ylabel(label)

        return powspec, bin_edges

    def get_square_slice(self, start=None):
        """Retrieve the slice to get the maximum unmasked square.

        Parameters
        ----------
        start : (int, int)
            define the center (y, x) of the starting point (default: center of the image)

        Returns
        -------
        islice : slice
            to be applied on the object itself on both dimension data[islice, islice]

        Notes
        -----
        Simply growth a square symetrically from the starting point
        """
        if start is None:
            start = np.asarray(self.shape) // 2
        else:
            assert isinstance(start, (list, tuple, np.ndarray)), "start should have a length of 2"
            assert len(start) == 2, "start should have a length of 2"

        islice = slice(*(np.asarray(start) + [0, 1]))

        while np.all(~self.mask[islice, islice]):
            islice = slice(islice.start - 1, islice.stop)
        islice = slice(islice.start + 1, islice.stop)

        while np.all(~self.mask[islice, islice]):
            islice = slice(islice.start, islice.stop + 1)
        islice = slice(islice.start, islice.stop - 1)

        return islice

    def to_hdus(
        self,
        hdu_data="DATA",
        hdu_mask="MASK",
        hdu_uncertainty="UNCERT",
        hdu_hits="HITS",
        wcs_relax=True,
        key_uncertainty_type="UTYPE",
    ):
        """Creates an HDUList object from a NikaMap object.
        Parameters
        ----------
        hdu_data, hdu_mask, hdu_uncertainty, hdu_hits : str or None, optional
            If it is a string append this attribute to the HDUList as
            `~astropy.io.fits.ImageHDU` with the string as extension name.
            Default is ``'DATA'`` for data, ``'MASK'`` for mask, ``'UNCERT'``
            for uncertainty and ``HITS`` for hits.
        wcs_relax : bool
            Value of the ``relax`` parameter to use in converting the WCS to a
            FITS header using `~astropy.wcs.WCS.to_header`. The common
            ``CTYPE`` ``RA---TAN-SIP`` and ``DEC--TAN-SIP`` requires
            ``relax=True`` for the ``-SIP`` part of the ``CTYPE`` to be
            preserved.
        key_uncertainty_type : str, optional
            The header key name for the class name of the uncertainty (if any)
            that is used to store the uncertainty type in the uncertainty hdu.
            Default is ``UTYPE``.

        Raises
        -------
        ValueError
            - If ``self.mask`` is set but not a `numpy.ndarray`.
            - If ``self.uncertainty`` is set but not a astropy uncertainty type.
            - If ``self.uncertainty`` is set but has another unit then
              ``self.data``.

        Returns
        -------
        hdulist : `~astropy.io.fits.HDUList`
        """
        if isinstance(self.header, fits.Header):
            # Copy here so that we can modify the HDU header by adding WCS
            # information without changing the header of the CCDData object.
            header = self.header.copy()
        else:
            # Because _insert_in_metadata_fits_safe is written as a method
            # we need to create a dummy CCDData instance to hold the FITS
            # header we are constructing. This probably indicates that
            # _insert_in_metadata_fits_safe should be rewritten in a more
            # sensible way...
            header = deepcopy(self.header)
            history = header.pop("history", None)
            comment = header.pop("comment", None)

            dummy_data = NikaMap([1], meta=fits.Header(), unit="")
            for k, v in header.items():
                dummy_data._insert_in_metadata_fits_safe(k, str(v))
            header = dummy_data.header

            if history is not None:
                for item in history:
                    header["history"] = item
            if comment is not None:
                for item in comment:
                    header["comment"] = item

            for key, comment in FITS_HEADER_COMMENT.items():
                if key in header:
                    header.set(key, comment=comment)

        if self.unit is not u.dimensionless_unscaled:
            header["bunit"] = self.unit.to_string()

        if self.wcs:
            # Simply extending the FITS header with the WCS can lead to
            # duplicates of the WCS keywords; iterating over the WCS
            # header should be safer.
            #
            # Turns out if I had read the io.fits.Header.extend docs more
            # carefully, I would have realized that the keywords exist to
            # avoid duplicates and preserve, as much as possible, the
            # structure of the commentary cards.
            #
            # Note that until astropy/astropy#3967 is closed, the extend
            # will fail if there are comment cards in the WCS header but
            # not header.
            wcs_header = self.wcs.to_header(relax=wcs_relax)
            header.extend(wcs_header, useblanks=False, update=True)

        hdus = [fits.ImageHDU(self.data, header, name=hdu_data)]

        if hdu_mask and self.mask is not None:
            # Always assuming that the mask is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.mask, "shape"):
                raise ValueError("only a numpy.ndarray mask can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduMask = fits.ImageHDU(self.mask.astype(np.uint8), header, name=hdu_mask)
            hdus.append(hduMask)

        if hdu_uncertainty and self.uncertainty is not None:
            # We need to save some kind of information which uncertainty was
            # used so that loading the HDUList can infer the uncertainty type.
            # No idea how this can be done so only allow StdDevUncertainty.
            uncertainty_cls = self.uncertainty.__class__
            if uncertainty_cls not in _known_uncertainties:
                raise ValueError("only uncertainties of type {} can be saved.".format(_known_uncertainties))
            uncertainty_name = _unc_cls_to_name[uncertainty_cls]

            hdr_uncertainty = fits.Header(header)
            hdr_uncertainty[key_uncertainty_type] = uncertainty_name

            # Assuming uncertainty is an StdDevUncertainty save just the array
            # this might be problematic if the Uncertainty has a unit differing
            # from the data so abort for different units. This is important for
            # astropy > 1.2
            if hasattr(self.uncertainty, "unit") and self.uncertainty.unit is not None:
                if not _uncertainty_unit_equivalent_to_parent(uncertainty_cls, self.uncertainty.unit, self.unit):
                    raise ValueError(
                        "saving uncertainties with a unit that is not "
                        "equivalent to the unit from the data unit is not "
                        "supported."
                    )

            hduUncert = fits.ImageHDU(self.uncertainty.array, hdr_uncertainty, name=hdu_uncertainty)
            hdus.append(hduUncert)

        if hdu_hits and self.hits is not None:
            # Always assuming that the hits is a np.ndarray (check that it has
            # a 'shape').
            if not hasattr(self.hits, "shape"):
                raise ValueError("only a numpy.ndarray hits can be saved.")

            # Convert boolean mask to uint since io.fits cannot handle bool.
            hduHits = fits.ImageHDU(self.hits.astype(np.uint8), header, name=hdu_hits)
            hdus.append(hduHits)

        hdulist = fits.HDUList(hdus)

        return hdulist


def retrieve_primary_keys(filename, band="1mm", **kwd):
    """Retrieve usefulle keys in primary header."""

    assert band in ["1mm", "2mm", "1", "2", "3"], "band should be either '1mm', '2mm', '1', '2', '3'"

    with fits.open(filename, **kwd) as hdus:
        # Fiddling to "fix" the fits file
        # extension params and info
        # hdus[14].header['EXTNAME'] = 'Param'
        # hdus[15].header['EXTNAME'] = 'Info'
        f_sampling = hdus[0].header["f_sampli"] * u.Hz
        if band in ["1mm", "1", "3"]:
            bmaj = hdus[0].header["FWHM_260"] * u.arcsec
        elif band in ["2mm", "2"]:
            bmaj = hdus[0].header["FWHM_150"] * u.arcsec

    return f_sampling, bmaj


def idl_fits_nikamap_reader(filename, band="1mm", revert=False, **kwd):
    """NIKA2 IDL Pipeline Map reader.

    Parameters
    ----------
    filename : str
        the fits filename
    band : str (1mm | 2mm | 1 | 2 | 3)
        the requested band
    revert : boolean
         use if to return -1 * data
    """

    f_sampling, bmaj = retrieve_primary_keys(filename, band, **kwd)

    with fits.open(filename, **kwd) as hdus:
        primary_header = hdus[0].header

        brightness_key = "Brightness_{}".format(band)
        stddev_key = "Stddev_{}".format(band)
        hits_key = "Nhits_{}".format(band)

        data = hdus[brightness_key].data.astype(np.float)
        header = hdus[brightness_key].header
        e_data = hdus[stddev_key].data.astype(np.float)
        hits = hdus[hits_key].data.astype(np.int)

    header = update_header(header, bmaj)

    # time = (hits / f_sampling).to(u.h)

    # Mask unobserved regions
    unobserved = hits == 0
    data[unobserved] = np.nan
    e_data[unobserved] = np.nan

    if revert:
        data *= -1

    data = NikaMap(
        data,
        mask=unobserved,
        uncertainty=StdDevUncertainty(e_data),
        hits=hits,
        sampling_freq=f_sampling,
        unit=header["UNIT"],
        wcs=WCS(header),
        header=header,
        primary_header=primary_header,
    )

    return data


def idl_fits_nikamap_writer(nm_data, filename, band="1mm", append=False, **kwd):
    """Write NikaMap object on IDL Pipeline fits file format.

    Parameters
    ----------
    filename : str
        the fits filename
    band : str (1mm | 2mm | 1 | 2 | 3)
        the output band
    append : boolean
        append nikamap to file
    """
    assert band in ["1mm", "2mm", "1", "2", "3"], "band should be either '1mm', '2mm', '1', '2', '3'"

    if append:
        hdus = fits.HDUList.fromfile(filename, mode="update")
    else:
        hdus = fits.HDUList([fits.PrimaryHDU(None, getattr(nm_data, "primary_header", None))])

    for hdu in nm_data.to_hdus(
        hdu_data="Brightness_{}".format(band),
        hdu_mask=None,
        hdu_uncertainty="Stddev_{}".format(band),
        hdu_hits="Nhits_{}".format(band),
    ):
        hdus.append(hdu)

    if append:
        hdus.flush()
    else:
        hdus.writeto(filename, **kwd)


def piic_fits_nikamap_reader(filename, band=None, revert=False, unit="mJy/beam", **kwd):
    """NIKA2 PIIC Pipeline Map reader.

    Parameters
    ----------
    filename : str or `Path
        the fits data filename
    band : str (1mm | 2mm | 1 | 2 | 3 )
        the corresponding band
    revert : boolean
         use if to return -1 * data
    unit : str
         unit of the data (assuming mJy/beam)
    Notes
    -----
    the snr filenames is assumed to be in the same directory ending in '_snr.fits'
    """
    data_file = Path(filename)
    rgw_file = data_file.parent / (data_file.with_suffix("").name + "rgw.fits")

    assert data_file.exists() & rgw_file.exists(), "Either {} or {} could not be found".format(
        data_file.name, rgw_file.name
    )

    with fits.open(data_file) as data_hdu, fits.open(rgw_file) as rgw_hdu:
        header = data_hdu[0].header
        data = data_hdu[0].data.astype(np.float)
        rgw = rgw_hdu[0].data.astype(np.float)
        rgw_header = rgw_hdu[0].header

    assert WCS(rgw_header).to_header() == WCS(header).to_header(), "{} and {} do not share the same WCS".format(
        data_file.name, rgw_file.name
    )
    with np.errstate(invalid="ignore", divide="ignore"):
        e_data = 1 / np.sqrt(rgw)

    unobserved = np.isnan(data) | np.isnan(e_data)

    if revert:
        data *= -1

    data = NikaMap(
        data,
        mask=unobserved,
        uncertainty=StdDevUncertainty(e_data),
        unit=unit,
        wcs=WCS(header),
        meta={"header": header, "primary_header": None, "band": band},
        hit=None,
    )

    return data


def identify_piic(origin, *args, **kwargs):
    data_file = Path(args[0])
    rgw_file = data_file.parent / (data_file.with_suffix("").name + "rgw.fits")
    check = data_file.exists() & rgw_file.exists()
    if check:
        check &= fits.connect.is_fits("read", data_file.parent, data_file.open(mode="rb"))
        check &= fits.connect.is_fits("read", rgw_file.parent, rgw_file.open(mode="rb"))
    return check


with registry.delay_doc_updates(NikaMap):
    registry.register_reader("piic", NikaMap, piic_fits_nikamap_reader)
    registry.register_identifier("piic", NikaMap, identify_piic)

    registry.register_reader("idl", NikaMap, idl_fits_nikamap_reader)
    registry.register_writer("idl", NikaMap, idl_fits_nikamap_writer)
    registry.register_identifier("idl", NikaMap, fits.connect.is_fits)


class NikaFits(MutableMapping):
    def __init__(self, filename=None, **kwd):
        self._filename = filename
        self._kwd = kwd
        self.__data = {"1mm": None, "2mm": None, "1": None, "2": None, "3": None}
        with fits.open(filename, **kwd) as hdus:
            self.primary_header = hdus[0].header

    def __repr__(self):
        return "<NIKAFits(filename={})>".format(self._filename)

    def __getitem__(self, key):
        value = self.__data[key]
        if not isinstance(value, NikaMap):
            value = NikaMap.read(self._filename, band=key, **self._kwd)
            self.__data[key] = value
        return value

    def __delitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        if key not in self.__data:
            raise KeyError(key)

    def __iter__(self):
        return iter(self.__data)

    def __len__(self):
        return len(self.__data)

    @classmethod
    def read(cls, *args, **kwargs):
        """NIKA2 IDL Pipeline Map reader

        Parameters
        ----------
        filename : str
            the fits filename
        revert : boolean
             use if to return -1 * data
        """
        return cls(*args, **kwargs)

    def write(self, filename, *args, **kwargs):
        """Write NikaFits object on IDL Pipeline fits file format

        Parameters
        ----------
        filename : str
            the fits filename
        """
        hdus = [fits.PrimaryHDU(None, self.primary_header)]
        for item in self.values():
            hdus += item.to_hdus()
        hdus = fits.HDUList(hdus)
        hdus.writeto(filename, **kwargs)
