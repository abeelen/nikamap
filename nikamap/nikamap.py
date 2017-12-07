from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits, registry
from astropy import units as u
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.nddata import NDDataArray, StdDevUncertainty, NDUncertainty
from astropy.modeling import models
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.convolution import Kernel2D, Box2DKernel

from astropy.table import Table, MaskedColumn, Column


import photutils
from photutils.datasets import make_gaussian_sources_image

# from scipy.signal import convolve
from scipy import signal

import warnings
from astropy.utils.exceptions import AstropyWarning


plt.ion()

Jy_beam = u.Jy / u.beam

__all__ = ['NikaBeam', 'NikaMap', 'fake_data']


# Forking from astropy.convolution.kernels
def _round_up_to_odd_integer(value):
    i = int(np.ceil(value))  # TODO: int() call is only needed for six.PY2
    if i % 2 == 0:
        return i + 1
    else:
        return i


class NikaBeam(Kernel2D):
    def __init__(self, fwhm=None, pixel_scale=None, **kwargs):

        self._pixel_scale = pixel_scale
        self._fwhm = fwhm

        if kwargs.get('array', None) is None:
            stddev = gaussian_fwhm_to_sigma * \
                fwhm.to(u.pixel, equivalencies=self._pixel_scale).value
            self._model = models.Gaussian2D(1., 0, 0, stddev, stddev)
            self._default_size = _round_up_to_odd_integer(8 * stddev)

        super(NikaBeam, self).__init__(**kwargs)
        self._truncation = np.abs(1. - self._array.sum())

    @property
    def fwhm(self):
        return self._fwhm

    @property
    def fwhm_pix(self):
        return self._fwhm.to(u.pixel, equivalencies=self._pixel_scale)


class NikaMap(NDDataArray):
    def __init__(self, *args, **kwargs):

        # Must be set AFTER the super() call
        time = kwargs.pop('time', None)
        beam = kwargs.pop('beam', None)
        fake_sources = kwargs.pop('fake_sources', None)
        sources = kwargs.pop('sources', None)

        super(NikaMap, self).__init__(*args, **kwargs)

        pixsize = np.abs(self.meta.get('CDELT1', 1)) * u.deg
        self._pixel_scale = u.pixel_scale(pixsize / u.pixel)

        if time is not None:
            self.time = time
        else:
            self.time = np.zeros(self.data.shape) * u.s

        if beam is None:
            # Default gaussian beam
            bmaj = self.meta.get('BMAJ', 1) * u.deg
            self.beam = NikaBeam(bmaj, pixel_scale=self._pixel_scale)
        else:
            self.beam = beam

        self.fake_sources = fake_sources
        self.sources = sources

    def compressed(self):
        return self.data[~self.mask] * self.unit

    def uncertainty_compressed(self):
        return self.uncertainty.array[~self.mask] * self.uncertainty.unit

    def __array__(self):
        return np.ma.array(self.data * self.unit, mask=self.mask)

    def __u_array__(self):
        return np.ma.array(self.uncertainty.array * self.unit, mask=self.mask)

    def __t_array__(self):
        return np.ma.array(self.time, mask=self.mask, fill_value=0)

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            if isinstance(value, NDUncertainty):
                if getattr(value, '_parent_nddata', None) is not None:
                    value = value.__class__(value, copy=False)
                self._uncertainty = value
            elif isinstance(value, np.ndarray):
                if value.shape != self.shape:
                    raise ValueError("uncertainty must have same shape as "
                                     "data.")
                self._uncertainty = StdDevUncertainty(value)
            else:
                raise TypeError("uncertainty must be an instance of a "
                                "NDUncertainty object or a numpy array.")
            self._uncertainty.parent_nddata = self
        else:
            self._uncertainty = value

    @property
    def SNR(self):
        return np.ma.array(self.data / self.uncertainty.array, mask=self.mask)

    @property
    def beam(self):
        beam = self._beam
        beam.normalize('peak')
        return beam

    @beam.setter
    def beam(self, value):
        self._beam = value

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        value = u.Quantity(value)
        if not value.unit.is_equivalent(u.s):
            raise ValueError('time unit must be equivalent to seconds')
        if value.shape != self.data.shape:
            raise ValueError('time must have the same shape as the data.')
        self._time = value

    def _slice(self, item):
        # slice all normal attributes
        kwargs = super(NikaMap, self)._slice(item)
        # The arguments for creating a new instance are saved in kwargs
        # so we need to add another keyword "flags" and add the sliced flags
        kwargs['time'] = self.time[item]
        kwargs['beam'] = self.beam

        kwargs['fake_sources'] = self.fake_sources
        kwargs['sources'] = self.sources

        return kwargs  # these must be returned

    def detect_sources(self, threshold=3, clean=True, box_size=4):
        """Detect sources with IRAF Star Finder and DAO StarFinder"""

        # To avoid fitting warnings...
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)

            # Make sure that there is no detection on the edge of the map
            box_kernel = Box2DKernel(box_size)
            detect_mask = ~np.isclose(signal.fftconvolve(
                ~self.mask, box_kernel, mode='same'), 1)
            detect_on = self.SNR.filled(0)
            detect_on[detect_mask] = 0

            sources = photutils.find_peaks(
                detect_on, threshold=threshold, mask=self.mask, wcs=self.wcs, subpixel=True, box_size=box_size)
        sources.meta['method'] = 'find_peak'
        sources.meta['threshold'] = threshold

        if clean:
            # Filter values on threshold
            sources = sources[sources['peak_value'] > threshold]

        # Transform to masked Table here to avoid future warnings
        sources = Table(sources, masked=True)

        if len(sources) > 0:
            # pixels values are irrelevant
            sources.remove_columns(
                ['x_centroid', 'y_centroid', 'x_peak', 'y_peak'])
            # Only keep fitted value
            sources.remove_columns(
                ['icrs_ra_peak', 'icrs_dec_peak', 'peak_value'])

            # Rename usefull columns
            sources.rename_column('icrs_ra_centroid', 'ra')
            sources.rename_column('icrs_dec_centroid', 'dec')

            # Sort by decreasing SNR
            sources['fit_peak_value'].name = 'SNR'
            sources.sort('SNR')
            sources.reverse()

            sources.add_column(Column(np.arange(len(sources)), name='ID'), 0)

        if self.fake_sources:
            # Match to the fake catalog
            fake_sources = self.fake_sources
            dist_threshold = self.beam.fwhm / 3

            if len(sources) == 0:
                fake_sources['find_peak'] = MaskedColumn(
                    np.ones(len(fake_sources)), mask=True)
            else:

                fake_SkyCoord = SkyCoord(
                    fake_sources['ra'], fake_sources['dec'])
                sources_SkyCoord = SkyCoord(sources['ra'], sources['dec'])

                idx, sep2d, _ = match_coordinates_sky(
                    fake_SkyCoord, sources_SkyCoord)
                mask = sep2d > dist_threshold
                fake_sources['find_peak'] = MaskedColumn(
                    sources[idx]['ID'], mask=mask)

                idx, sep2d, _ = match_coordinates_sky(
                    sources_SkyCoord, fake_SkyCoord)
                mask = sep2d > dist_threshold
                sources['fake_sources'] = MaskedColumn(
                    fake_sources[idx]['ID'], mask=mask)

        self.sources = sources

    def match_sources(self, catalogs, dist_threshold=4 * u.arcsec):

        for cat, ref_cat in product([self.sources], catalogs):
            cat_SkyCoord = SkyCoord(cat['ra'], cat['dec'], unit=(cat['ra'].unit, cat['dec'].unit))
            ref_SkyCoord = SkyCoord(ref_cat['ra'], ref_cat['dec'], unit=(ref_cat['ra'].unit, ref_cat['dec'].unit))
            idx, sep2d, _ = match_coordinates_sky(cat_SkyCoord, ref_SkyCoord)
            mask = sep2d > dist_threshold
            cat[ref_cat.meta['name']] = MaskedColumn(idx, mask=mask)

    def phot_sources(self, sources=None, peak=True, psf=True):

        if sources is None:
            sources = self.sources

        xx, yy = self.wcs.wcs_world2pix(sources['ra'], sources['dec'], 0)

        if peak:
            # Crude Peak Photometry
            sources['flux_peak'] = Column([self.data[np.int(y), np.int(
                x)] for x, y in zip(xx, yy)], unit=self.unit * u.beam).to(u.mJy)
            sources['eflux_peak'] = Column([self.uncertainty.array[np.int(y), np.int(
                x)] for x, y in zip(xx, yy)], unit=self.unit * u.beam).to(u.mJy)

        if psf:
            # BasicPSFPhotometry with fixed positions
            from photutils.psf import IntegratedGaussianPRF, BasicPSFPhotometry
            from astropy.modeling.fitting import LevMarLSQFitter
            from photutils.psf import DAOGroup
            from photutils.background import MedianBackground

            sigma_psf = self.beam.fwhm_pix.value * gaussian_fwhm_to_sigma

            psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
            psf_model.x_0.fixed = True
            psf_model.y_0.fixed = True

            daogroup = DAOGroup(2.0 * self.beam.fwhm_pix.value)
            mmm_bkg = MedianBackground()

            photometry = BasicPSFPhotometry(group_maker=daogroup, bkg_estimator=mmm_bkg,
                                            psf_model=psf_model, fitter=LevMarLSQFitter(),
                                            fitshape=(21, 21))

            positions = Table([Column(xx, name="x_0"), Column(yy, name="y_0")])

            result_tab = photometry(image=np.ma.array(
                self.data, mask=self.mask).filled(0), init_guesses=positions)

            sources['flux_psf'] = Column(
                result_tab['flux_fit'] / (2 * np.pi * sigma_psf**2), unit=self.unit * u.beam).to(u.mJy)
            sources['eflux_psf'] = Column(
                result_tab['flux_unc'] / (2 * np.pi * sigma_psf**2), unit=self.unit * u.beam).to(u.mJy)

        self.sources = sources

    def match_filter(self, kernel):
        """Return a matched filtered version of the map

        Notes
        -----
        Peak photometry is conserved for data and e_data
        Resultings maps have a different mask

        >>> npix = 500
        >>> std = 4
        >>> kernel = Gaussian2DKernel(std)
        >>> mask = np.zeros((npix,npix))
        >>> data = np.random.normal(0, 1, size=mask.shape)
        >>> data[(npix-std*8)//2:(npix+std*8)//2+1,(npix-std*8)//2:(npix+std*8)//2+1] += kernel.array/kernel.array.max()
        >>> data = NikaMap(data, uncertainty=StdDevUncertainty(np.ones_like(data)), time=np.ones_like(data)*u.s, mask=mask)
        >>> mf_data = data.match_filter(kernel)
        >>> fig, axes = plt.subplots(ncols=2)
        >>> axes[0].imshow(data) ; axes[1].imshow(mf_data)
        """

        # Is peak normalizerd on get
        beam = self.beam

        kernel.normalize('integral')

        # Assuming the same pixel_scale
        if isinstance(beam.model, models.Gaussian2D) & isinstance(kernel.model, models.Gaussian2D):
            fwhm = np.sqrt(beam.model.x_fwhm**2 +
                           kernel.model.x_fwhm**2) * u.pixel
            fwhm = fwhm.to(u.arcsec, equivalencies=beam._pixel_scale)
            mf_beam = NikaBeam(fwhm, pixel_scale=beam._pixel_scale)
        else:
            # Using scipy.signal.convolve to extend the beam if necessary
            mf_beam = NikaBeam(array=signal.convolve(
                beam.array, kernel.array), pixel_scale=beam._pixel_scale)

        # Convolve the mask and retrieve the fully sampled region, this
        # will remove one kernel width on the edges
        # mf_mask = ~np.isclose(convolve(~self.mask, kernel, normalize_kernel=False), 1)
        mf_mask = ~np.isclose(signal.fftconvolve(
            ~self.mask, kernel, mode='same'), 1)

        # Convolve the time (integral for time)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        #     mf_time = convolve(self.time, kernel, normalize_kernel=False)*self.time.unit
        mf_time = signal.fftconvolve(self.__t_array__().filled(
            0), kernel, mode='same') * self.time.unit
        mf_time[mf_mask] = 0

        # Convolve the data (peak for unit conservation)
        kernel.normalize('peak')
        kernel_sqr = kernel.array**2

        # ma.filled(0) required for the fft convolution
        weights = 1. / self.uncertainty.array**2
        weights[self.mask] = 0

        # with np.errstate(divide='ignore'):
        #     mf_uncertainty = np.sqrt(convolve(weights, kernel_sqr, normalize_kernel=False))**-1
        with np.errstate(invalid='ignore', divide='ignore'):
            mf_uncertainty = np.sqrt(signal.fftconvolve(
                weights, kernel_sqr, mode='same'))**-1
        mf_uncertainty[mf_mask] = np.nan

        # Units are not propagated in masked arrays...
        # mf_data = convolve(weights*data, kernel, normalize_kernel=False) * mf_uncertainty**2
        mf_data = signal.fftconvolve(
            weights * self.__array__().filled(0), kernel, mode='same') * mf_uncertainty**2

        mf_data = NikaMap(mf_data, unit=self.unit, mask=mf_mask, time=mf_time,
                          uncertainty=StdDevUncertainty(mf_uncertainty), wcs=self.wcs,
                          meta=self.meta, fake_sources=self.fake_sources, beam=mf_beam)

        return mf_data

    def plot_SNR(self, clim=None, levels=None, title=None, ax=None, cat=None):

        SNR = self.SNR.data
        if clim is None:
            clim = (-4, 4)

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=self.wcs, title=title)

        # interval = ZScaleInterval()
        # vmin, vmax = interval.get_limits(SNR)
        #   vmin, vmax = MinMaxInterval().get_limits(mf_SNR)
        vmin, vmax = -3, 5
        snr = ax.imshow(SNR, vmin=vmin, vmax=vmax, origin='lower')
        snr.set_clim(clim)
        if levels is not None:
            levels = np.sqrt(2)**np.arange(levels[0], levels[1])
            levels = np.concatenate((-levels[::-1], levels))
            ax.contour(SNR, levels=levels, origin='lower', colors='w')

        # In case of fake sources, overplot them
        if self.fake_sources:
            x, y = self.wcs.wcs_world2pix(self.fake_sources['ra'], self.fake_sources['dec'], 0)
            ax.scatter(x, y, marker='o', c='red', alpha=0.8)

        if cat is True:
            cat = [(self.sources, '^')]

        if cat is not None:
            for _cat, _mark in list(cat):
                label = _cat.meta.get('method') or _cat.meta.get('name')
                cat_SkyCoord = SkyCoord(_cat['ra'], _cat['dec'], unit=(_cat['ra'].unit, _cat['dec'].unit))
                x, y = self.wcs.wcs_world2pix(cat_SkyCoord.ra, cat_SkyCoord.dec, 0)
                ax.scatter(x, y, marker=_mark, alpha=0.8, label=label)

        ax.set_xlim(0, self.shape[1])
        ax.set_ylim(0, self.shape[0])
        ax.legend(loc='best', frameon=False)

        return ax


def fits_nikamap_reader(filename, band="1mm", revert=False, **kwd):
    """Retrieve the required data from the fits file"""

    with fits.open(filename, **kwd) as hdus:
        # Fiddling to "fix" the fits file
        # extension params and info
        # hdus[14].header['EXTNAME'] = 'Param'
        # hdus[15].header['EXTNAME'] = 'Info'
        f_sampling = hdus[0].header['f_sampli'] * u.Hz
        if band == "1mm":
            bmaj = hdus[0].header['FWHM_260'] * u.arcsec
        elif band == "2mm":
            bmaj = hdus[0].header['FWHM_150'] * u.arcsec

        data, header = hdus['Brightness_{}'.format(
            band)].data, hdus['Brightness_{}'.format(band)].header
        e_data = hdus['Stddev_{}'.format(band)].data
        hits = hdus['Nhits_{}'.format(band)].data

        header['BMAJ'] = (bmaj.to(u.deg).value, '[deg],  Beam major axis')
        header['BMIN'] = (bmaj.to(u.deg).value, '[deg],  Beam minor axis')

        time = (hits / f_sampling).to(u.h)

        # Mask unobserved regions
        unobserved = hits == 0
        data[unobserved] = np.nan
        e_data[unobserved] = np.nan

        if revert:
            data *= -1

        data = NikaMap(data, mask=unobserved, uncertainty=StdDevUncertainty(
            e_data), unit=header['UNIT'], wcs=WCS(header), meta=header, time=time)

    return data


with registry.delay_doc_updates(NikaMap):
    registry.register_reader('fits', NikaMap, fits_nikamap_reader)
    registry.register_identifier('fits', NikaMap, fits.connect.is_fits)


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


def fake_source(data, beam_fwhm=12.5 * u.arcsec, pixsize=2 * u.arcsec, nsources=10, peak_flux=1 * u.mJy, within=(1. / 4, 3. / 4), grid=None, wobble=False):

    shape = data.shape

    beam_std_pix = (beam_fwhm / pixsize).decompose().value * \
        gaussian_fwhm_to_sigma

    sources = Table(masked=True)
    sources['amplitude'] = (np.ones(nsources) * peak_flux).to(Jy_beam * u.beam)
    if grid is None:
        x_mean = np.random.uniform(within[0], within[1], size=nsources)
        y_mean = np.random.uniform(within[0], within[1], size=nsources)
    else:
        sq_sources = int(np.sqrt(nsources))
        assert sq_sources**2 == nsources, 'nsources must be a squared number'
        y_mean, x_mean = np.indices(
            [sq_sources] * 2) / (sq_sources - 1) * (within[1] - within[0]) + within[0]
        if wobble:
            step = (np.max(within) - np.min(within)) / (sq_sources - 1)
            x_mean += np.random.normal(0, step / 2 *
                                       gaussian_fwhm_to_sigma, size=x_mean.shape)
            y_mean += np.random.normal(0, step / 2 *
                                       gaussian_fwhm_to_sigma, size=y_mean.shape)

    sources['x_mean'] = x_mean.flatten() * shape[1]
    sources['y_mean'] = y_mean.flatten() * shape[0]

    sources['x_stddev'] = np.ones(nsources) * beam_std_pix
    sources['y_stddev'] = np.ones(nsources) * beam_std_pix
    sources['theta'] = np.zeros(nsources)

    # Crude check to be within the finite part of the map
    within_coverage = np.isfinite(
        data[sources['y_mean'].astype(int), sources['x_mean'].astype(int)])

    sources = sources[within_coverage]

    # Add an ID column
    sources.add_column(Column(np.arange(len(sources)), name='ID'), 0)

    return data + make_gaussian_sources_image(shape, sources), sources


def fake_data(shape=(512, 512), beam_fwhm=12.5 * u.arcsec, pixsize=2 * u.arcsec, NEFD=50e-3 * Jy_beam * u.s**0.5,
              nsources=32, grid=False, wobble=False, peak_flux=None, time_fwhm=1. / 5, jk_data=None, e_data=None):
    """Build fake dataset"""

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

    if nsources:
        data, sources = fake_source(data, beam_fwhm=beam_fwhm, pixsize=pixsize,
                                    nsources=nsources, peak_flux=peak_flux, grid=grid, wobble=wobble)
        a, d = WCS(header).wcs_pix2world(
            sources['x_mean'], sources['y_mean'], 0)
        sources.remove_columns(['x_mean', 'y_mean'])
        sources.add_columns([Column(a * u.deg, name='ra'),
                             Column(d * u.deg, name='dec')])
    else:
        sources = None

    data = NikaMap(data, mask=mask, unit=Jy_beam, uncertainty=StdDevUncertainty(
        e_data), wcs=WCS(header), meta=header, time=time, fake_sources=sources)

    return data
