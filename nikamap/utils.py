import numpy as np
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.modeling import Parameter, Fittable2DModel
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.nddata import StdDevUncertainty

Jy_beam = u.Jy / u.beam

__all__ = ['fake_data']


# Forking from astropy.convolution.kernels
def _round_up_to_odd_integer(value):
    i = int(np.ceil(value))  # TODO: int() call is only needed for six.PY2
    if i % 2 == 0:
        return i + 1
    else:
        return i


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
        return ((int(self.y_0 - halfwidth), int(self.y_0 + halfwidth)),
                (int(self.x_0 - halfwidth), int(self.x_0 + halfwidth)))

    def __init__(self, sigma=sigma.default,
                 x_0=x_0.default, y_0=y_0.default, flux=flux.default,
                 **kwargs):
        if self._erf is None:
            from scipy.special import erf
            self.__class__._erf = erf

        super(CircularGaussianPSF, self).__init__(n_models=1, sigma=sigma,
                                                  x_0=x_0, y_0=y_0,
                                                  flux=flux, **kwargs)

    def evaluate(self, x, y, flux, x_0, y_0, sigma):
        """Model function Gaussian PSF model."""

        return flux * np.exp(-((x - x_0)**2 + (y - y_0)**2) / (2*sigma**2))


def fake_header(shape=(512, 512), beam_fwhm=12.5 * u.arcsec, pixsize=2 * u.arcsec):
    """Build fake header"""

    header = fits.Header()
    header['NAXIS'] = (2, 'Number of data axes')
    header['NAXIS1'] = (shape[1], '')
    header['NAXIS2'] = (shape[0], '')

    header['CTYPE1'] = ('RA---TAN', 'Coordinate Type')
    header['CTYPE2'] = ('DEC--TAN', 'Coordinate Type')
    header['EQUINOX'] = (2000, 'Equinox of Ref. Coord.')

    header['CRPIX1'] = (shape[1] / 2, 'Reference Pixel in X')
    header['CRPIX2'] = (shape[0] / 2, 'Reference Pixel in Y')

    header['CRVAL1'] = (189, 'R.A. (degrees) of reference pixel')
    header['CRVAL2'] = (62, 'Declination of reference pixel')

    header['CDELT1'] = (-pixsize.to(u.deg).value, 'Degrees / Pixel')
    header['CDELT2'] = (pixsize.to(u.deg).value, 'Degrees / Pixel')

    header['OBJECT'] = ('fake', 'Name of the object')
    header['BMAJ'] = (beam_fwhm.to(u.deg).value, '[deg],  Beam major axis')
    header['BMIN'] = (beam_fwhm.to(u.deg).value, '[deg],  Beam major axis')

    return header


def fake_data(shape=(512, 512), beam_fwhm=12.5 * u.arcsec, pixsize=2 * u.arcsec, NEFD=50e-3 * Jy_beam * u.s**0.5,
              nsources=32, grid=False, wobble=False, peak_flux=None, time_fwhm=1. / 5, jk_data=None, e_data=None):
    """Build fake dataset"""

    # To avoid import loops
    from .nikamap import NikaMap

    if jk_data is not None:
        # JK data, extract all...
        data = jk_data.data
        e_data = jk_data.uncertainty
        mask = jk_data.mask
        time = jk_data.time
        header = jk_data.wcs.to_header()
        shape = data.shape
    elif e_data is not None:
        # Only gave e_data
        mask = np.isnan(e_data)
        time = ((e_data / NEFD)**(-1. / 0.5)).to(u.h)
        e_data = e_data.to(Jy_beam).value

        data = np.random.normal(0, 1, size=shape) * e_data

    else:
        # Regular gaussian noise
        if time_fwhm is not None:
            # Time as a centered gaussian
            y_idx, x_idx = np.indices(shape, dtype=np.float)
            time = np.exp(-((x_idx - shape[1] / 2)**2 / (2 * (gaussian_fwhm_to_sigma * time_fwhm * shape[1])**2) +
                            (y_idx - shape[0] / 2)**2 / (2 * (gaussian_fwhm_to_sigma * time_fwhm * shape[0])**2))) * u.h
        else:
            # Time is uniform
            time = np.ones(shape) * u.h

        mask = time < 1 * u.s
        time[mask] = np.nan

        e_data = (NEFD * time**(-0.5)).to(Jy_beam).value

        # White noise plus source
        data = np.random.normal(0, 1, size=shape) * e_data

    header = fake_header(shape, beam_fwhm, pixsize)
    header['NEFD'] = (NEFD.to(Jy_beam * u.s**0.5).value,
                      '[Jy/beam sqrt(s)], NEFD')

    # min flux which should be recoverable at the center of the field at 3 sigma
    if peak_flux is None:
        peak_flux = 3 * (NEFD / np.sqrt(np.nanmax(time)) * u.beam).to(u.mJy)

    data = NikaMap(data, mask=mask, unit=Jy_beam, uncertainty=StdDevUncertainty(
        e_data), wcs=WCS(header), meta=header, time=time)

    if nsources:
        data.add_gaussian_sources(nsources=nsources, peak_flux=peak_flux, grid=grid, wobble=wobble)

    return data
