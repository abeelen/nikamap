from __future__ import absolute_import, division, print_function

import os
import numpy as np
import warnings

from scipy import signal

import matplotlib.pyplot as plt
import multiprocessing

from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.modeling import Parameter, Fittable2DModel
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.nddata import StdDevUncertainty
from astropy.convolution import CustomKernel, convolve_fft

Jy_beam = u.Jy / u.beam

__all__ = ["fake_data", "cat_to_sc", "CircularGaussianPSF", "pos_uniform", "pos_gridded", "pos_list"]


# from radio_beam.utils.convolve
# https://github.com/radio-astro-tools/radio-beam/blob/master/radio_beam/utils.py


def beam_convolve(beam, other):  # pragma: no cover
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
    """Build fake header"""

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
    """Update header if 'BMAJ' not present"""
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


def pos_uniform(shape=None, within=(0, 1), mask=None, nsources=1, peak_flux=1 * u.mJy, dist_threshold=0, max_loop=10):
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

    return pos[:, 1], pos[:, 0], np.repeat(peak_flux, len(pos))


def pos_gridded(
    shape=None, within=(0, 1), mask=None, nsources=2**2, peak_flux=1 * u.mJy, wobble=False, wobble_frac=1
):
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

    return pos[:, 1], pos[:, 0], np.repeat(peak_flux, len(pos))


def pos_list(shape=None, within=(0, 1), mask=None, nsources=1, peak_flux=1 * u.mJy, x_mean=None, y_mean=None):
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

    return pos[:, 1], pos[:, 0], np.repeat(peak_flux, len(pos))


def centered_circular_gaussian(fwhm, shape):
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
    nsources=32,
    peak_flux=None,
    pos_gen=pos_uniform,
    **kwargs
):
    """Build fake dataset"""

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
        data, mask=mask, unit=Jy_beam, uncertainty=StdDevUncertainty(e_data), wcs=WCS(header), meta=header, hits=hits
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

    header = {key: value for key, value in meta.items() if key not in ["history", "comment", "HISTORY", "COMMENT"]}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)
        header = fits.Header(header)

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
    # Transform pixel coordinates column to world coordinates
    lonlat = wcs.pixel_to_world_values(sources[x_key], sources[y_key])
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
