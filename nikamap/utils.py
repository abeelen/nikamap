from __future__ import absolute_import, division, print_function

import json
import multiprocessing
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.convolution import CustomKernel, convolve_fft
from astropy.convolution.kernels import _round_up_to_odd_integer
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import Fittable2DModel, Parameter
from astropy.nddata import StdDevUncertainty
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from scipy import signal

Jy_beam = u.Jy / u.beam

__all__ = [
    "fake_data",
    "cat_to_sc",
    "CircularGaussianPSF",
    "pos_uniform",
    "pos_uniform_no_overlap",
    "pos_gridded",
    "pos_list",
]


# from radio_beam.utils.convolve
# https://github.com/radio-astro-tools/radio-beam/blob/master/radio_beam/utils.py


def beam_convolve(beam, other):  # pragma: no cover
    """Convolve two elliptical Gaussian beams.

    Parameters
    ----------
    beam : object
        First beam-like object exposing ``major``, ``minor`` and ``pa``
        attributes.
    other : object
        Second beam-like object exposing the same attributes.

    Returns
    -------
    new_major : `~astropy.units.Quantity`
        Major axis of the convolved beam.
    new_minor : `~astropy.units.Quantity`
        Minor axis of the convolved beam.
    new_pa : `~astropy.units.Quantity`
        Position angle of the convolved beam.

    Notes
    -----
    The implementation follows the MIRIAD Gaussian parameter combination
    formulae.
    """
    # blame: https://github.com/pkgw/carma-miriad/blob/CVSHEAD/src/subs/gaupar.for
    # (github checkin of MIRIAD, code by Sault)

    alpha = (
        (beam.major * np.cos(beam.pa)) ** 2
        + (beam.minor * np.sin(beam.pa)) ** 2
        + (other.major * np.cos(other.pa)) ** 2
        + (other.minor * np.sin(other.pa)) ** 2
    )

    beta = (
        (beam.major * np.sin(beam.pa)) ** 2
        + (beam.minor * np.cos(beam.pa)) ** 2
        + (other.major * np.sin(other.pa)) ** 2
        + (other.minor * np.cos(other.pa)) ** 2
    )

    gamma = 2 * (
        (beam.minor**2 - beam.major**2) * np.sin(beam.pa) * np.cos(beam.pa)
        + (other.minor**2 - other.major**2) * np.sin(other.pa) * np.cos(other.pa)
    )

    s = alpha + beta
    t = np.sqrt((alpha - beta) ** 2 + gamma**2)

    new_major = np.sqrt(0.5 * (s + t))
    new_minor = np.sqrt(0.5 * (s - t))
    # absolute tolerance needs to be <<1 microarcsec
    if np.isclose(((abs(gamma) + abs(alpha - beta)) ** 0.5).to(u.arcsec).value, 1e-7):
        new_pa = 0.0 * u.deg
    else:
        new_pa = 0.5 * np.arctan2(-1.0 * gamma, alpha - beta)

    return new_major, new_minor, new_pa


class CircularGaussianPSF(Fittable2DModel):
    r"""
    Circular Gaussian model, not integrated, un-normalized.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian PSF.
    flux : float (default 1)
        Total integrated flux over the entire PSF
    x_0 : float (default 0)
        Position of the peak in x direction.
    y_0 : float (default 0)
        Position of the peak in y direction.

    """

    flux = Parameter(default=1)
    x_0 = Parameter(default=0)
    y_0 = Parameter(default=0)
    sigma = Parameter(default=1, fixed=True)

    _erf = None
    fit_deriv = None

    @property
    def bounding_box(self):
        halfwidth = 4 * self.sigma
        return (
            (int(self.y_0 - halfwidth), int(self.y_0 + halfwidth)),
            (int(self.x_0 - halfwidth), int(self.x_0 + halfwidth)),
        )

    def __init__(self, sigma=sigma.default, x_0=x_0.default, y_0=y_0.default, flux=flux.default, **kwargs):
        if self._erf is None:
            from scipy.special import erf

            self.__class__._erf = erf

        super(CircularGaussianPSF, self).__init__(n_models=1, sigma=sigma, x_0=x_0, y_0=y_0, flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """Model function Gaussian PSF model."""

        return flux * np.exp(-((x - x_0) ** 2 + (y - y_0) ** 2) / (2 * sigma**2))


def fake_header(shape=(512, 512), beam_fwhm=12.5 * u.arcsec, pixsize=2 * u.arcsec):
    """Build a minimal synthetic FITS header for a test map.

    Parameters
    ----------
    shape : tuple of int, optional
        Image shape as ``(ny, nx)``.
    beam_fwhm : `~astropy.units.Quantity`, optional
        Circular beam full width at half maximum.
    pixsize : `~astropy.units.Quantity`, optional
        Pixel angular size.

    Returns
    -------
    `~astropy.io.fits.Header`
        Header describing a tangent-plane sky projection and beam keywords.
    """

    header = fits.Header()
    header["NAXIS"] = (2, "Number of data axes")
    header["NAXIS1"] = (shape[1], "")
    header["NAXIS2"] = (shape[0], "")

    header["CTYPE1"] = ("RA---TAN", "Coordinate Type")
    header["CTYPE2"] = ("DEC--TAN", "Coordinate Type")
    header["EQUINOX"] = (2000, "Equinox of Ref. Coord.")

    header["CRPIX1"] = (shape[1] / 2, "Reference Pixel in X")
    header["CRPIX2"] = (shape[0] / 2, "Reference Pixel in Y")

    header["CRVAL1"] = (189, "R.A. (degrees) of reference pixel")
    header["CRVAL2"] = (62, "Declination of reference pixel")

    header["CDELT1"] = (-pixsize.to(u.deg).value, "Degrees / Pixel")
    header["CDELT2"] = (pixsize.to(u.deg).value, "Degrees / Pixel")

    header["OBJECT"] = ("fake", "Name of the object")

    update_header(header, beam_fwhm)

    return header


def update_header(header, bmaj):
    """Ensure beam keywords are present in a FITS header.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        Header to update.
    bmaj : `~astropy.units.Quantity`
        Beam major axis written to the ``BMAJ`` and ``BMIN`` keywords when
        missing.

    Returns
    -------
    `~astropy.io.fits.Header`
        Updated header.
    """
    if "BMAJ" not in header:  # pragma: no cover  # old file format
        header["BMAJ"] = (bmaj.to(u.deg).value, "[deg],  Beam major axis")
        header["BMIN"] = (bmaj.to(u.deg).value, "[deg],  Beam minor axis")

    return header


def cat_to_sc(cat):
    """Extract positions from cat and return corresponding SkyCoord

    Parameters
    ----------
    cat : :class:`astropy.table.Table`
        a table containing sky columns with units

    Returns
    -------
    :class:`astropy.coordinates.SkyCoord`
        the corresponding SkyCoord object

    Notes
    -----
    Look for _ra/_dec first and then ra/dec
    """

    if "_ra" in cat.keys() and "_dec" in cat.keys():
        cols = ["_ra", "_dec"]
    elif "ra" in cat.keys() and "dec" in cat.keys():
        cols = ["ra", "dec"]

    coords = SkyCoord(cat[cols[0]], cat[cols[1]], unit=(cat[cols[0]].unit, cat[cols[1]].unit))

    return coords



def pos_in_mask(pos, mask=None, nsources=1):
    """Check if pos is in mask, issue warning with less than nsources

    Parameters
    ----------
    pos : array_like (N, 2)
        pixel indexes (y, x) to be checked in mask
    mask : 2D boolean array_like
        corresponding mask
    nsources : int
        the requested number of sources

    Returns
    -------
    :class:`numpy.ndarray`
        the pixel indexes within the mask
    """
    pos = np.asarray(pos)

    if mask is not None:
        pos_idx = np.floor(pos + 0.5).astype(int)
        inside = ~mask[pos_idx[:, 0], pos_idx[:, 1]]
        pos = pos[inside]

    if pos.shape[0] < nsources:
        warnings.warn("Only {} positions".format(pos.shape[0]), UserWarning)

    return pos


def pos_too_close(pos, dist_threshold=0):
    """Remove sources which are too close

    Parameters
    ----------
    pos : array_like (N, 2)
        pixel indexes (y, x) to be checked in mask
    dist_threshold : float
        the distance threshold to remove the sources

    Returns
    -------
    :class:`numpy.ndarray`
        the filtered positions

    Notes
    -----
    Based on Euclidian distances
    """
    dist_mask = np.ones(len(pos), dtype=bool)

    while not np.all(~dist_mask):
        # Computing pixel distances between all sources
        dist = np.sqrt(np.sum((pos.reshape(len(pos), 1, 2) - pos) ** 2, 2))

        # Filter 0 distances and find minima
        i = np.arange(len(pos))
        dist[i, i] = np.inf
        arg_min_dist = np.argmin(dist, 1)
        min_dist = dist[i, arg_min_dist]

        # This will mask pair of sources with dist < dist_threshold
        dist_mask = min_dist < dist_threshold

        # un-mask the second source
        for idx, arg_min in enumerate(arg_min_dist):
            if dist_mask[idx]:
                dist_mask[arg_min] = False

        pos = pos[~dist_mask]

    return pos


def pos_uniform(shape=None, within=(0, 1), mask=None, nsources=1, dist_threshold=0, max_loop=10):
    """Generate x, y uniform position within a mask, with a minimum distance between them

    Notes
    -----
    depending on the distance threshold and the number of loop, the requested number of sources might not be returned
    """

    pos = np.array([[], []], dtype=float).T

    i_loop = 0
    while i_loop < max_loop and len(pos) < nsources:
        i_loop += 1

        # note that these are pixels 0-indexes
        pos = np.concatenate((pos, np.random.uniform(within[0], within[1], (nsources, 2)) * np.asarray(shape) - 0.5))

        # Filter sources inside the mask
        pos = pos_in_mask(pos, mask, 0)

        # Removing too close sources
        pos = pos_too_close(pos, dist_threshold)

        pos = pos[0:nsources]

    if i_loop == max_loop and len(pos) < nsources:
        warnings.warn("Maximum of loops reached, only have {} positions".format(len(pos)), UserWarning)

    return pos[:, 1], pos[:, 0]


def pos_uniform_no_overlap(
    shape=None,
    within=(0, 1),
    mask=None,
    nsources=1,
    dist_threshold=0,
    oversample=5,
    max_loop=10,
):
    """Generate x, y uniform positions with pairwise distance constraints.

    Parameters
    ----------
    shape : tuple
        Output map shape as (ny, nx).
    within : tuple
        Fractional bounds in [0, 1] applied on each axis.
    mask : 2D boolean array_like, optional
        Mask where True values are excluded.
    nsources : int
        Requested number of sources.
    dist_threshold : float
        Minimum Euclidian distance in pixel units between sources.
    oversample : int
        Number of candidates generated per requested source at each loop.
    max_loop : int
        Maximum number of generation/refinement loops.

    Returns
    -------
    x, y, flux : ndarray, ndarray, ndarray
        Pixel coordinates and repeated flux values.

    Notes
    -----
    Depending on constraints and map geometry, fewer than ``nsources`` may be
    returned. In this case a UserWarning is emitted.
    """
    pos = np.array([[], []], dtype=float).T

    i_loop = 0
    while i_loop < max_loop and len(pos) < nsources:
        i_loop += 1

        n_candidates = max(1, int(oversample) * int(nsources))
        candidates = np.random.uniform(within[0], within[1], (n_candidates, 2)) * np.asarray(shape) - 0.5
        pos = np.concatenate((pos, candidates))

        # Keep only valid candidates then remove close pairs.
        pos = pos_in_mask(pos, mask, 0)
        pos = pos_too_close(pos, dist_threshold)
        pos = pos[0:nsources]

    if i_loop == max_loop and len(pos) < nsources:
        warnings.warn("Maximum of loops reached, only have {} positions".format(len(pos)), UserWarning)

    return pos[:, 1], pos[:, 0]


def pos_gridded(shape=None, within=(0, 1), mask=None, nsources=2**2, wobble=False, wobble_frac=1):
    """Generate x, y gridded position within a mask

    Parameters
    ----------
    wobble : boolean
        Add a random offset with fwhm = grid_step * wobble_frac

    Notes
    -----
    requested number of sources might not be returned"""

    sq_sources = int(np.sqrt(nsources))
    assert sq_sources**2 == nsources, "nsources must be a squared number"
    assert nsources > 1, "nsouces can not be 1"

    # square distribution with step margin on the side
    within_step = (within[1] - within[0]) / (sq_sources + 1)
    pos = np.indices([sq_sources] * 2, dtype=float) * within_step + within[0] + within_step

    if wobble:
        # With some wobbling if needed
        pos += np.random.normal(0, within_step * wobble_frac * gaussian_fwhm_to_sigma, pos.shape)

    pos = pos.reshape(2, nsources).T

    # wobbling can push sources outside the shape
    inside = np.sum((pos >= 0) & (pos <= 1), 1) == 2
    pos = pos[inside]

    pos = pos * np.asarray(shape) - 0.5

    pos = pos_in_mask(pos, mask, nsources)

    return pos[:, 1], pos[:, 0]


def pos_list(shape=None, within=(0, 1), mask=None, nsources=1, x_mean=None, y_mean=None):
    """Return positions within a mask

    Notes
    -----
    requested number of sources might not be returned"""

    assert x_mean is not None and y_mean is not None, "you must provide x_mean & y_mean"
    assert len(x_mean) == len(y_mean), "x_mean and y_mean must have the same dimension"
    assert nsources <= len(x_mean), "x_mean must contains at least {} sources".format(nsources)

    pos = np.array([y_mean, x_mean]).T

    # within
    limits = shape * np.asarray(within)[:, np.newaxis]
    inside = np.sum((pos >= limits[0]) & (pos <= limits[1] - 1), 1) == 2
    pos = pos[inside]

    pos = pos_in_mask(pos, mask, nsources)

    return pos[:, 1], pos[:, 0]


def centered_circular_gaussian(fwhm, shape):
    """Generate a centred circular Gaussian exposure map.

    Parameters
    ----------
    fwhm : float or array_like
        Gaussian full width at half maximum expressed as a fraction of the
        image size along each axis.
    shape : tuple of int
        Output array shape as ``(ny, nx)``.

    Returns
    -------
    ndarray
        Normalized Gaussian profile sampled on the image grid.
    """
    y_idx, x_idx = np.indices(shape, dtype=float)
    sigma = gaussian_fwhm_to_sigma * fwhm * np.asarray(shape)
    delta_x = (x_idx - shape[1] / 2) ** 2 / (2 * sigma[1] ** 2)
    delta_y = (y_idx - shape[0] / 2) ** 2 / (2 * sigma[0] ** 2)

    return np.exp(-(delta_x + delta_y))


def fake_data(
    shape=(512, 512),
    beam_fwhm=12.5 * u.arcsec,
    pixsize=2 * u.arcsec,
    nefd=50e-3 * Jy_beam * u.s**0.5,
    sampling_freq=25 * u.Hz,
    time_fwhm=1.0 / 5,
    jk_data=None,
    e_data=None,
    primary_header=None,
    nsources=32,
    peak_flux=None,
    pos_gen=pos_uniform,
    **kwargs,
):
    """Build a synthetic NikaMap-like dataset for tests and examples.

    Parameters
    ----------
    shape : tuple of int, optional
        Output map shape as ``(ny, nx)``.
    beam_fwhm : `~astropy.units.Quantity`, optional
        Circular beam full width at half maximum.
    pixsize : `~astropy.units.Quantity`, optional
        Pixel angular size.
    nefd : `~astropy.units.Quantity`, optional
        Noise-equivalent flux density used to derive the map noise.
    sampling_freq : `~astropy.units.Quantity`, optional
        Sampling frequency used to derive hit counts from integration time.
    time_fwhm : float, optional
        Fractional FWHM of the integration-time Gaussian profile. If None, a
        uniform coverage map is used.
    jk_data : object, optional
        Existing jackknife-like map used as a template for data, mask and hits.
    e_data : `~astropy.units.Quantity`, optional
        Per-pixel noise standard deviation map.
    primary_header : `~astropy.io.fits.Header`, optional
        Primary FITS header stored on the returned map.
    nsources : int, optional
        Number of Gaussian sources to inject. Set to 0 to return pure noise.
    peak_flux : `~astropy.units.Quantity`, optional
        Peak flux of the injected sources. If omitted, a nominal 3-sigma value
        at the field centre is used.
    pos_gen : callable, optional
        Position generator used when injecting sources.
    **kwargs
        Additional keyword arguments forwarded to ``add_gaussian_sources``.

    Returns
    -------
    `~nikamap.nikamap.NikaMap`
        Synthetic map populated with optional Gaussian sources.
    """

    # To avoid import loops
    from .nikamap import NikaMap

    if jk_data is not None:
        # JK data, extract all...
        data = jk_data.data
        e_data = jk_data.uncertainty
        mask = jk_data.mask
        hits = jk_data.hits
        shape = data.shape
        primary_header = data.primary_header
        sampling_freq = data.sampling_freq
    elif e_data is not None:
        # Only gave e_data
        mask = np.isnan(e_data)
        time = ((e_data / nefd) ** (-1.0 / 0.5)).to(u.h)
        hits = (time / sampling_freq).decompose().value.astype(int)
        e_data = e_data.to(Jy_beam).value

        data = np.random.normal(0, 1, size=shape) * e_data

    else:
        # Regular gaussian noise
        if time_fwhm is not None:
            # Time as a centered gaussian
            time = centered_circular_gaussian(time_fwhm, shape) * u.h
        else:
            # Time is uniform
            time = np.ones(shape) * u.h

        hits = (time / sampling_freq).decompose().value.astype(int)
        mask = time < 1 * u.s
        time[mask] = 0
        hits[mask] = 0

        e_data = (nefd * time ** (-0.5)).to(Jy_beam).value

        # White noise plus source
        data = np.random.normal(0, 1, size=shape) * e_data

    header = fake_header(shape, beam_fwhm, pixsize)
    header["NEFD"] = (nefd.to(Jy_beam * u.s**0.5).value, "[Jy/beam sqrt(s)], NEFD")

    # min flux which should be recoverable at the center of the field at 3
    # sigma
    if peak_flux is None:
        peak_flux = 3 * (nefd / np.sqrt(np.nanmax(time)) * u.beam).to(u.mJy)

    data = NikaMap(
        data,
        mask=mask,
        unit=Jy_beam,
        uncertainty=StdDevUncertainty(e_data),
        wcs=WCS(header),
        meta=header,
        hits=hits,
        primary_header=primary_header,
    )

    if nsources:
        data.add_gaussian_sources(nsources=nsources, cat_gen=pos_gen, peak_flux=peak_flux, **kwargs)

    return data


def shrink_mask(mask, kernel):
    """Shrink mask wrt to a kernel

    Parameters
    ----------
    mask : 2D boolean array_like
        the mask to be shrinked by...
    kernel : 2D float array_like
        ... the corresponding array

    Returns
    -------
    2D boolean array
        the corresponding shrunk mask

    Notes
    -----
    The kernel sum must be normalized
    """
    return ~np.isclose(signal.fftconvolve(~mask, kernel, mode="same"), 1)


def fft_2d_hanning(mask, size=2):
    """Build a 2D Hanning apodization map from a mask.

    Parameters
    ----------
    mask : ndarray of bool
        Input validity mask where True values mark masked pixels.
    size : int, optional
        Radius of the radial Hanning kernel in pixels.

    Returns
    -------
    ndarray
        Apodization map with smoothly tapered edges.
    """
    assert np.min(mask.shape) > size * 2 + 1
    assert size > 1

    idx = np.linspace(-0.5, 0.5, size * 2 + 1, endpoint=True)
    xx, yy = np.meshgrid(idx, idx)
    n = np.sqrt(xx**2 + yy**2)
    hann_kernel = (1 + np.cos(2 * np.pi * n)) / 2
    hann_kernel[n > 0.5] = 0

    hann_kernel = CustomKernel(hann_kernel)
    hann_kernel.normalize("integral")

    # Reduce mask size to apodize on the edge
    apod = ~shrink_mask(mask, hann_kernel)

    # Final convolution goes to 0 on the edge
    apod = convolve_fft(apod, hann_kernel)

    return apod


def setup_ax(ax=None, wcs=None):
    """Setup a axe for plotting.

    Parameters
    ----------
    ax : ~matplotlib.Axes, optional
        potential axe, by default None
    wcs : ~astropy.wcs.WCS, optional
        potential wcs, by default None

    Returns
    -------
    ~matplotlib.Axes
        the necessary axe.
    """

    if not ax:
        fig = plt.figure()
        if wcs is not None:
            ax = fig.add_subplot(111, projection=getattr(wcs, "low_level_wcs", wcs))
        else:
            ax = fig.add_subplot(111)

    return ax


def meta_to_header(meta):
    """Transform a meta object into a fits Header

    Parameters
    ----------
    meta : dict-like
        a meta object

    Returns
    -------
    header : :class:`~astropy.io.fits.Header`
        the corresponding header
    """
    header = fits.Header()
    for key, value in meta.items():
        if key in ["history", "comment", "HISTORY", "COMMENT"]:
            continue
        if not isinstance(value, (int, float, str, complex, np.floating, np.integer, np.complexfloating, np.bool_)):
            value = json.dumps(value)

        # from from CCDData._insert_in_metadata_fits_safe
        if len(key) > 9 and len(str(value)) > 72:
            short_name = key[:8]
            header[f"HIERARCH {key.upper()}"] = (short_name, f"Shortened name for {key}")
            header[short_name] = value
        else:
            header[key] = value

    for key in ["history", "comment"]:
        if key in meta:
            for item in meta[key]:
                header[key] = item

    return header


def cpu_count():
    """Proper cpu count on a SLURM cluster."""
    try:
        ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    except KeyError:
        ncpus = multiprocessing.cpu_count()

    return ncpus


def _read_first_int(path):
    """Read the first integer from a text file, returning None on failure."""
    try:
        with open(path, "r") as handle:
            token = handle.read().strip().split()[0]
    except (OSError, IndexError):
        return None

    if token == "max":
        return None

    try:
        return int(token)
    except ValueError:
        return None


def available_memory_bytes():
    """Best-effort estimate of available memory in bytes.

    Priority order:
    1. cgroups memory limits/usage (v2 then v1), useful on SLURM/HPC nodes
       and containers.
    2. psutil virtual memory available bytes.
    3. POSIX sysconf fallback.
    """
    # cgroup v2
    cgv2_limit = _read_first_int("/sys/fs/cgroup/memory.max")
    cgv2_used = _read_first_int("/sys/fs/cgroup/memory.current")
    if cgv2_limit is not None and cgv2_used is not None and cgv2_limit > 0:
        return max(0, cgv2_limit - cgv2_used)

    # cgroup v1
    cgv1_limit = _read_first_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    cgv1_used = _read_first_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if cgv1_limit is not None and cgv1_used is not None and cgv1_limit > 0:
        # Ignore absurdly large "unlimited" sentinel values from cgroup v1.
        if cgv1_limit < (1 << 60):
            return max(0, cgv1_limit - cgv1_used)

    # psutil fallback
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        pass

    # Last-resort POSIX fallback
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None

    return int(page_size * available_pages)


def shuffled_average(datas, weights, n_shuffle=1):
    len_data = datas.shape[0]
    outputs = []
    for _ in range(n_shuffle):
        shuffled_index = np.floor(np.random.uniform(0, len_data, len_data)).astype(int)
        # np.ma.average is needed as some of the pixels have zero weights (should be masked)
        outputs.append(np.ma.average(datas[shuffled_index], weights=weights[shuffled_index], axis=0, returned=False))
    return outputs


def _shuffled_average(*args, datas=None, weights=None):
    """Worker function to produce shuffled averages

    To be used with ProgressBar.map and multiprocess=True
    """
    if len(args) > 0:
        n_shuffle = len(args[0])
    else:
        n_shuffle = 1

    outputs = shuffled_average(datas, weights, n_shuffle)

    return outputs


def xy_to_world(sources, wcs, x_key, y_key):
    """Add world-coordinate columns from pixel-coordinate columns.

    Parameters
    ----------
    sources : `~astropy.table.Table`
        Source table containing pixel coordinates.
    wcs : `~astropy.wcs.WCS`
        WCS used to convert pixels to world coordinates.
    x_key : str
        Name of the pixel x-coordinate column.
    y_key : str
        Name of the pixel y-coordinate column.

    Returns
    -------
    `~astropy.table.Table`
        Input table with added world-coordinate columns and compatibility
        aliases for right ascension and declination when available.
    """
    # Transform pixel coordinates column to world coordinates
    lonlat = wcs.low_level_wcs.pixel_to_world_values(sources[x_key], sources[y_key])
    for key, item, unit in zip(wcs.world_axis_physical_types, lonlat, wcs.world_axis_units):
        sources[key] = item * u.Unit(unit)

    if "pos.eq.ra" in sources.colnames and "pos.eq.dec" in sources.colnames:
        for key in ["ra", "dec"]:
            if key in sources.colnames:
                del sources[key]
        sources.rename_columns(["pos.eq.ra", "pos.eq.dec"], ["ra", "dec"])
        # For compatibility issues
        sources["_ra"] = sources["ra"]
        sources["_dec"] = sources["dec"]

    return sources


def make_kernel_image(
    shape,
    kernel,
    sources,
    x_name="x_mean",
    y_name="y_mean",
    memory_fraction=0.5,
):
    """Make a model image using FFT convolution with an astropy Kernel2D.

    Builds the image as the sum of amplitude-weighted Dirac deltas convolved
    with a kernel (e.g. a beam PSF).  The entire computation is done in Fourier
    space:

        FT{ A_k * delta(x-x_k, y-y_k) } = A_k * exp(-2πi (u x_k + v y_k))

    Summing over all sources gives the Fourier-space delta map; multiplying by
    the kernel's FFT and inverse-transforming yields the final image.  Sub-pixel
    source positions are handled exactly through the complex-exponential phase.

    Parameters
    ----------
    shape : (ny, nx)
    kernel : `~astropy.convolution.Kernel2D`
        Convolution kernel (e.g. ``Gaussian2DKernel``).  Its ``.array``
        attribute is used directly; the kernel is assumed to be centred.
    sources : `~astropy.table.Table`
        Must contain ``"amplitude"``, *x_name*, *y_name*.
    x_name, y_name : str
        Column names for the pixel x / y positions.
    psf_nsigma : float, optional
        For `~astropy.convolution.Gaussian2DKernel`, rebuild the kernel with a
        larger support of +/- psf_nsigma * sigma along each axis before FFT.
        This reduces PSF truncation. Ignored for non-Gaussian kernels.
    pixel_scale : `~astropy.units.Quantity`
        Not used in the computation but kept for API consistency.

    Returns
    -------
    convolved_image : ndarray (or Quantity)
        Real part of the inverse FFT of the product of the Dirac Fourier map
        and the kernel FFT.
    """
    ny, nx = shape

    # Build an odd-sized, zero-padded working grid to reduce circular FFT
    # wrap-around from sources close to map boundaries.
    kernel.normalize("peak")
    kernel_array = np.array(kernel, dtype=float)
    ky, kx = kernel_array.shape

    ky_odd = int(_round_up_to_odd_integer(ky))
    kx_odd = int(_round_up_to_odd_integer(kx))
    if ky_odd > ky or kx_odd > kx:
        kernel_array = np.pad(
            kernel_array,
            ((0, ky_odd - ky), (0, kx_odd - kx)),
            mode="constant",
            constant_values=0,
        )
    ky, kx = ky_odd, kx_odd

    margin_y = ky // 2
    margin_x = kx // 2
    ny_work = ny + 2 * margin_y
    nx_work = nx + 2 * margin_x

    ny_fft = int(_round_up_to_odd_integer(ny_work))
    nx_fft = int(_round_up_to_odd_integer(nx_work))

    # ------------------------------------------------------------------
    # 1. Fourier-space map of Dirac delta functions
    # ------------------------------------------------------------------
    # For a delta at position (x0, y0) weighted by amplitude A:
    #   FT[u, v] = A * exp(-2πi (u*x0 + v*y0))
    # Summing over all sources gives the combined Fourier map.

    u_freq = np.fft.fftfreq(nx_fft)  # shape (nx_fft,)
    v_freq = np.fft.fftfreq(ny_fft)  # shape (ny_fft,)
    uu, vv = np.meshgrid(u_freq, v_freq)  # shape (ny_fft, nx_fft)

    amp_values = np.asarray(sources["amplitude"], dtype=float)
    x_values = np.asarray(sources[x_name], dtype=float) + margin_x
    y_values = np.asarray(sources[y_name], dtype=float) + margin_y

    # Use the separability of the Fourier kernel to avoid looping over pixels:
    # exp(-2j*pi*(u*x + v*y)) = exp(-2j*pi*u*x) * exp(-2j*pi*v*y).
    # Depending on available memory, compute in one pass or split over source
    # chunks to reduce peak RAM.
    n_sources = len(amp_values)
    available_mem = available_memory_bytes()
    phase_bytes_per_source = np.dtype(np.complex128).itemsize * (nx_fft + ny_fft)
    max_sources_per_chunk = n_sources

    if available_mem is not None and n_sources > 0:
        memory_budget = max(1, int(float(memory_fraction) * available_mem))
        max_sources_per_chunk = max(1, memory_budget // phase_bytes_per_source)

    if n_sources <= max_sources_per_chunk:
        phase_x = np.exp(-2j * np.pi * np.outer(x_values, u_freq))
        phase_y = np.exp(-2j * np.pi * np.outer(v_freq, y_values))
        delta_fft = phase_y @ (amp_values[:, None] * phase_x)
    else:
        chunk_size = int(max_sources_per_chunk)
        delta_fft = np.zeros((ny_fft, nx_fft), dtype=np.complex128)
        for start in range(0, n_sources, chunk_size):
            end = min(n_sources, start + chunk_size)
            phase_x = np.exp(-2j * np.pi * np.outer(x_values[start:end], u_freq))
            phase_y = np.exp(-2j * np.pi * np.outer(v_freq, y_values[start:end]))
            delta_fft += phase_y @ (amp_values[start:end, None] * phase_x)

    # ------------------------------------------------------------------
    # 2. Kernel FFT
    # ------------------------------------------------------------------
    # Pad the (small, centred) kernel array to the full image shape, then
    # use ifftshift to move its centre to pixel (0, 0) — the origin
    # expected by numpy's FFT convention — before transforming.

    # Build kernel image on the same odd-sized grid used for FFT.
    pad_image = np.zeros((ny_fft, nx_fft))
    # Center the kernel in the full image (critical to avoid introducing shifts)
    start_y = (ny_fft - ky) // 2
    start_x = (nx_fft - kx) // 2
    pad_image[start_y : start_y + ky, start_x : start_x + kx] = kernel_array
    # ifftshift moves the kernel centre to (0, 0)
    kernel_fft = np.fft.fft2(np.fft.ifftshift(pad_image))

    # ------------------------------------------------------------------
    # 3. Multiply in Fourier space and back-transform
    # ------------------------------------------------------------------
    convolved_image = np.fft.ifft2(delta_fft * kernel_fft).real

    # Crop back to requested output size (central region of the padded map).
    return convolved_image[margin_y : margin_y + ny, margin_x : margin_x + nx]

def process_map(fn, *iterables, max_workers=None, chunksize=1, max_length=200):
    """Process a map with a function in parallel.

    Parameters
    ----------
    fn : callable
        Function to apply to each chunk of the map.
    iterables : iterable of iterables
        Iterables to be passed to the function. Each iterable should have the same length.
    max_workers : int, optional
        Maximum number of worker processes to use. If None, it will use the number of CPUs in the system.
    chunksize : int, optional
        Number of items to process in each chunk. Default is 1. None will chunck the map in as many chunks as max_workers.
    max_length : int, optional
        Maximum length of the iterables before chunking. Default is 200.
    **kwargs : dict
        Additional keyword arguments to pass to the function.

    Returns
    -------
    list
        List of results from applying the function to each chunk.
    """
    from concurrent.futures import ProcessPoolExecutor
    from operator import length_hint

    if max_workers is None:
        max_workers = cpu_count()

    if iterables:
        longest_iterable_len = max(map(length_hint, iterables))
        if longest_iterable_len < max_workers:
            max_workers = longest_iterable_len

    if iterables and chunksize is None:
        chunksize = 1

        if longest_iterable_len > max_length:
            chunksize = longest_iterable_len // max_workers

        if chunksize == 1:
            return list(map(fn, *iterables))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(fn, *iterables, chunksize=chunksize))
