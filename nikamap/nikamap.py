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
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.convolution import Kernel2D, Box2DKernel

from astropy.table import Table, MaskedColumn, Column

from photutils.psf import BasicPSFPhotometry
from photutils.psf import DAOGroup
from photutils.background import MedianBackground

import photutils
from photutils.datasets import make_gaussian_sources_image

# from scipy.signal import convolve
from scipy import signal
from scipy.optimize import curve_fit

import warnings
from astropy.utils.exceptions import AstropyWarning

from .utils import CircularGaussianPSF, _round_up_to_odd_integer
from .utils import pos_uniform, pos_gridded
from .utils import powspec_k

Jy_beam = u.Jy / u.beam

__all__ = ['NikaBeam', 'NikaMap']


class NikaBeam(Kernel2D):
    """NikaBeam describe the beam of a NikaMap

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

        if kwargs.get('array', None) is None:
            stddev = gaussian_fwhm_to_sigma * \
                fwhm.to(u.pixel, equivalencies=self._pixel_scale).value
            self._model = models.Gaussian2D(1., 0, 0, stddev, stddev)
            self._default_size = _round_up_to_odd_integer(8 * stddev)

        super(NikaBeam, self).__init__(**kwargs)
        self._truncation = np.abs(1. - self._array.sum())

    def __repr__(self):
        return "<NikaBeam(fwhm={}, pixel_scale={:.2f} / pixel)".format(self.fwhm.to(u.arcsec), (1 * u.pixel).to(u.arcsec, equivalencies=self._pixel_scale))

    @property
    def fwhm(self):
        return self._fwhm

    @property
    def fwhm_pix(self):
        return self._fwhm.to(u.pixel, equivalencies=self._pixel_scale)

    @property
    def sigma(self):
        return self._fwhm * gaussian_fwhm_to_sigma

    @property
    def sigma_pix(self):
        return self._fwhm.to(u.pixel, equivalencies=self._pixel_scale) * gaussian_fwhm_to_sigma

    @property
    def area(self):
        return 2 * np.pi * self.sigma**2

    @property
    def area_pix(self):
        return 2 * np.pi * self.sigma_pix**2


class NikaMap(NDDataArray):
    """A NikaMap object represent a nika map with additionnal capabilities.

    It contains the metadata, wcs, and all attribute (data/stddev/time/unit/mask) as well as potential source list detected in these maps.

    Parameters
    -----------
    data : :class:`~numpy.ndarray` or :class:`astropy.nddata.NDData`
        The actual data contained in this `NDData` object. Not that this
        will always be copies by *reference* , so you should make copy
        the ``data`` before passing it in if that's the  desired behavior.
    uncertainty : :class:`astropy.nddata.NDUncertainty`, optional
        Uncertainties on the data.
    mask : `~numpy.ndarray`-like, optional
        Mask for the data, given as a boolean Numpy array or any object that
        can be converted to a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If ``data`` is a numpy masked array, providing
        ``mask`` here will causes the mask from the masked array to be
        ignored.
    time : :class:`astropy.units.quantity.Quantity` array
        The time spent per pixel on the map, must have unit equivalent to time
        and shape equivalent to data
    wcs : undefined, optional
        WCS-object containing the world coordinate system for the data.
    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.
    unit : :class:`astropy.units.UnitBase` instance or str, optional
        The units of the data.
    beam : :class:`nikamap.NikaBeam`
        The beam corresponding to the data, by default a circular gaussian
        constructed from the header 'BMAJ' keyword.
    fake_source : :class:`astropy.table.Table`, optional
        The table of potential fake sources included in the data

        .. note::
            The table must contain at least 3 columns: ['ID', 'ra', 'dec']

    sources : :class`astropy.table.Table`, optional
        The table of detected sources in the data.

    """

    def __init__(self, *args, **kwargs):

        # Must be set AFTER the super() call
        time = kwargs.pop('time', None)
        beam = kwargs.pop('beam', None)
        fake_sources = kwargs.pop('fake_sources', None)
        sources = kwargs.pop('sources', None)

        super(NikaMap, self).__init__(*args, **kwargs)

        if isinstance(self.wcs, WCS):
            pixsize = np.abs(self.wcs.wcs.cdelt[0]) * u.deg
        else:
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
                                "StdDevUncertainty object or a numpy array.")
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

    def trim(self):
        """Remove masked region on the edges"""

        mask = self.mask
        axis_slice = []
        for axis in [1, 0]:
            good_pix = np.argwhere(np.mean(mask, axis=axis) != 1)
            axis_slice.append(slice(np.min(good_pix), np.max(good_pix) + 1))

        return self[axis_slice[0], axis_slice[1]]

    def add_gaussian_sources(self, nsources=10, peak_flux=1 * u.mJy, within=(0, 1), pos_gen=pos_uniform, **kwargs):

        shape = self.shape

        # TODO: Non gaussian beam
        beam_std_pix = self.beam.sigma_pix.value

        x_mean, y_mean = pos_gen(
            nsources=nsources, shape=shape, within=within, mask=self.mask, **kwargs)

        nsources = len(x_mean)

        sources = Table(masked=True)

        sources['amplitude'] = (
            np.ones(nsources) * peak_flux).to(self.unit * u.beam)

        sources['x_mean'] = x_mean
        sources['y_mean'] = y_mean

        sources['x_stddev'] = np.ones(nsources) * beam_std_pix
        sources['y_stddev'] = np.ones(nsources) * beam_std_pix
        sources['theta'] = np.zeros(nsources)

        # Crude check to be within the finite part of the map
        if self.mask is not None:
            within_coverage = ~self.mask[sources['y_mean'].astype(
                int), sources['x_mean'].astype(int)]
            sources = sources[within_coverage]

        # Gaussian sources...
        self._data += make_gaussian_sources_image(shape, sources)

        # Add an ID column
        sources.add_column(Column(np.arange(len(sources)), name='ID'), 0)

        # Transform pixel to world coordinates
        a, d = self.wcs.wcs_pix2world(sources['x_mean'], sources['y_mean'], 0)
        sources.add_columns([Column(a * u.deg, name='ra'),
                             Column(d * u.deg, name='dec')])

        # Remove unnecessary columns
        sources.remove_columns(
            ['x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta'])

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

        detect_on = self.SNR.filled(0)

        if self.mask is not None:
                # Make sure that there is no detection on the edge of the map
            box_kernel = Box2DKernel(box_size)
            detect_mask = ~np.isclose(signal.fftconvolve(
                ~self.mask, box_kernel, mode='same'), 1)
            detect_on[detect_mask] = 0

        # TODO: Have a look at  ~photutils.psf.IterativelySubtractedPSFPhotometry

        # To avoid bad fit warnings...
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', AstropyWarning)
            sources = photutils.find_peaks(
                detect_on, threshold=threshold, mask=self.mask, wcs=self.wcs, subpixel=True, box_size=box_size)

        if len(sources) > 0:
            # Transform to masked Table here to avoid future warnings
            sources = Table(sources, masked=True)
            sources.meta['method'] = 'find_peak'
            sources.meta['threshold'] = threshold

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

        if len(sources) > 0:
            self.sources = sources
        else:
            self.sources = None

    def match_sources(self, catalogs, dist_threshold=None):

        if dist_threshold is None:
            dist_threshold = self.beam.fwhm / 3

        if not isinstance(catalogs, list):
            catalogs = [catalogs]

        for cat, ref_cat in product([self.sources], catalogs):
            cat_SkyCoord = SkyCoord(cat['ra'], cat['dec'], unit=(
                cat['ra'].unit, cat['dec'].unit))
            ref_SkyCoord = SkyCoord(ref_cat['ra'], ref_cat['dec'], unit=(
                ref_cat['ra'].unit, ref_cat['dec'].unit))
            idx, sep2d, _ = match_coordinates_sky(cat_SkyCoord, ref_SkyCoord)
            mask = sep2d > dist_threshold
            cat[ref_cat.meta['name']] = MaskedColumn(idx, mask=mask)

    def phot_sources(self, sources=None, peak=True, psf=True):

        if sources is None:
            sources = self.sources

        xx, yy = self.wcs.wcs_world2pix(sources['ra'], sources['dec'], 0)

        x_idx = np.floor(xx + 0.5).astype(int)
        y_idx = np.floor(yy + 0.5).astype(int)

        if peak:
            # Crude Peak Photometry
            # From pixel indexes to array indexing

            sources['flux_peak'] = Column(
                self.data[y_idx, x_idx], unit=self.unit * u.beam).to(u.mJy)
            sources['eflux_peak'] = Column(
                self.uncertainty.array[y_idx, x_idx], unit=self.unit * u.beam).to(u.mJy)

        if psf:
            # BasicPSFPhotometry with fixed positions

            sigma_psf = self.beam.sigma_pix.value

            # Using an IntegratedGaussianPRF can cause biais in the photometry
            # TODO: Check the NIKA2 calibration scheme
            # from photutils.psf import IntegratedGaussianPRF
            # psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
            psf_model = CircularGaussianPSF(sigma=sigma_psf)

            psf_model.x_0.fixed = True
            psf_model.y_0.fixed = True

            daogroup = DAOGroup(3 * self.beam.fwhm_pix.value)
            mmm_bkg = MedianBackground()

            photometry = BasicPSFPhotometry(group_maker=daogroup, bkg_estimator=mmm_bkg,
                                            psf_model=psf_model, fitter=LevMarLSQFitter(),
                                            fitshape=9)

            positions = Table([Column(xx, name="x_0"),
                               Column(yy, name="y_0"),
                               Column(self.data[y_idx, x_idx], name="flux_0")])

            result_tab = photometry(image=np.ma.array(
                self.data, mask=self.mask).filled(0), init_guesses=positions)

            # sources['flux_psf'] = Column(
            #     result_tab['flux_fit'] / (2 * np.pi * sigma_psf**2), unit=self.unit * u.beam).to(u.mJy)
            # sources['eflux_psf'] = Column(
            #     result_tab['flux_unc'] / (2 * np.pi * sigma_psf**2), unit=self.unit * u.beam).to(u.mJy)

            result_tab.sort('id')
            sources['flux_psf'] = Column(
                result_tab['flux_fit'] * psf_model(0, 0), unit=self.unit * u.beam).to(u.mJy)
            sources['eflux_psf'] = Column(
                result_tab['flux_unc'] * psf_model(0, 0), unit=self.unit * u.beam).to(u.mJy)
            sources['group_id'] = result_tab['group_id']

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
        if self.mask is not None:
            mf_mask = ~np.isclose(signal.fftconvolve(
                ~self.mask, kernel, mode='same'), 1)
        else:
            mf_mask = None

        # Convolve the time (integral for time)
        # with warnings.catch_warnings():
        #     warnings.simplefilter('ignore', AstropyWarning)
        #     mf_time = convolve(self.time, kernel, normalize_kernel=False)*self.time.unit
        mf_time = signal.fftconvolve(self.__t_array__().filled(0),
                                     kernel, mode='same') * self.time.unit

        if mf_mask is not None:
            mf_time[mf_mask] = 0

        # Convolve the data (peak for unit conservation)
        kernel.normalize('peak')
        kernel_sqr = kernel.array**2

        # ma.filled(0) required for the fft convolution
        weights = 1. / self.uncertainty.array**2
        if self.mask is not None:
            weights[self.mask] = 0

        # with np.errstate(divide='ignore'):
        #     mf_uncertainty = np.sqrt(convolve(weights, kernel_sqr, normalize_kernel=False))**-1
        with np.errstate(invalid='ignore', divide='ignore'):
            mf_uncertainty = np.sqrt(signal.fftconvolve(weights,
                                                        kernel_sqr, mode='same'))**-1
        if mf_mask is not None:
            mf_uncertainty[mf_mask] = np.nan

        # Units are not propagated in masked arrays...
        # mf_data = convolve(weights*data, kernel, normalize_kernel=False) * mf_uncertainty**2
        mf_data = signal.fftconvolve(weights * self.__array__().filled(0),
                                     kernel, mode='same') * mf_uncertainty**2

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
            ax = fig.add_subplot(111, projection=self.wcs)
        else:
            fig = None

        if title is not None:
            ax.set_title(title)

        # interval = ZScaleInterval()
        # vmin, vmax = interval.get_limits(SNR)
        #   vmin, vmax = MinMaxInterval().get_limits(mf_SNR)
        vmin, vmax = -3, 5
        snr = ax.imshow(SNR, vmin=vmin, vmax=vmax,
                        origin='lower', interpolation='none')
        snr.set_clim(clim)
        if levels is not None:
            levels = np.sqrt(2)**np.arange(levels[0], levels[1])
            levels = np.concatenate((-levels[::-1], levels))
            ax.contour(SNR, levels=levels, alpha=0.8, colors='w')

        # In case of fake sources, overplot them
        if self.fake_sources:
            x, y = self.wcs.wcs_world2pix(
                self.fake_sources['ra'], self.fake_sources['dec'], 0)
            ax.scatter(x, y, marker='o', c='red', alpha=0.8)

        if cat is True:
            cat = [(self.sources, '^')]

        if cat is not None:
            for _cat, _mark in list(cat):
                label = _cat.meta.get('method') or _cat.meta.get(
                    'name') or 'Unknown'
                cat_SkyCoord = SkyCoord(_cat['ra'], _cat['dec'], unit=(
                    _cat['ra'].unit, _cat['dec'].unit))
                x, y = self.wcs.wcs_world2pix(
                    cat_SkyCoord.ra, cat_SkyCoord.dec, 0)
                ax.scatter(x, y, marker=_mark, alpha=0.8, label=label)

        ax.set_xlim(0, self.shape[1])
        ax.set_ylim(0, self.shape[0])

        if cat is not None:
            ax.legend(loc='best', frameon=False)

        if fig:
            return fig

    def check_SNR(self, ax=None, bins=100):
        """Perform normality test on SNR map

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

        SN = self.SNR.compressed()
        hist, bin_edges = np.histogram(
            SN, bins=bins, normed=True, range=(-5, 5))

        # is biased if signal is presmf_beament
        # is biased if trimmed
        # mu, std = norm.fit(SN)

        bin_center = (bin_edges[1:] + bin_edges[:-1]) / 2

        # Clip to 3 sigma, this will biais the result
        robust = (-6 < bin_center) & (bin_center < 3)

        def gauss(x, a, c, s):
            return a * np.exp(-(x - c)**2 / (2 * s**2))

        popt, pcov = curve_fit(gauss, bin_center[robust], hist[robust])
        mu, std = popt[1:]

        if ax is not None:
            ax.plot(bin_center, hist, drawstyle='steps-mid')
            ax.plot(bin_center, gauss(bin_center, *popt))

        return std

    def plot_PSD(self, ax=None, bins=100, range=None, apod_size=None, snr=False, **kwargs):
        """Plot the power spectrum of the map

        Parameters
        ----------
        ax : :class:`matplotlib.axes.Axes`, optional
            Axe to plot the power spectrum
        bins : int
            Number of bins for the histogram. Default 100.
        range : (float, float), optional
            The lower and upper range of the bins. (see `~numpy.histogram`)
        snr : boolean
            use the SNR map

        Returns
        -------
        powspec_k : :class:`astropy.units.quantity.Quantity`
            The value of the power spectrum
        bin_edges : :class:`astropy.units.quantity.Quantity`
            Return the bin edges ``(length(hist)+1)``.
        """

        if snr:
            data = self.SNR
        else:
            data = self.__array__()

        res = (1 * u.pixel).to(u.arcsec, equivalencies=self._pixel_scale)
        powspec, bin_edges = powspec_k(
            data, res=res, bins=bins, range=range, apod_size=apod_size)

        if snr:
            powspec /= res**2
        else:
            powspec /= (self.beam.area / u.beam)**2
            powspec = powspec.to(u.Jy**2 / u.sr)

        if ax is not None:
            bin_center = (bin_edges[1:] + bin_edges[:-1]) / 2
            ax.loglog(bin_center, powspec, **kwargs)
            ax.set_xlabel(r'k [arcsec$^{-1}$]')
            ax.set_ylabel('P(k) [{}]'.format(powspec.unit))

        return powspec, bin_edges

    def get_square_slice(self, start=None):
        """Retrieve the slice to get the maximum unmasked square

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
            assert isinstance(start, (list, tuple, np.ndarray)
                              ), "start should have a length of 2"
            assert len(start) == 2, "start should have a length of 2"

        islice = slice(*(np.asarray(start) + [0, 1]))

        while np.all(~self.mask[islice, islice]):
            islice = slice(islice.start - 1, islice.stop)
        islice = slice(islice.start + 1, islice.stop)

        while np.all(~self.mask[islice, islice]):
            islice = slice(islice.start, islice.stop + 1)
        islice = slice(islice.start, islice.stop - 1)

        return islice


def retrieve_primary_keys(filename, band="1mm", **kwd):
    """Retrieve usefulle keys in primary header"""

    with fits.open(filename, **kwd) as hdus:
        # Fiddling to "fix" the fits file
        # extension params and info
        # hdus[14].header['EXTNAME'] = 'Param'
        # hdus[15].header['EXTNAME'] = 'Info'
        f_sampling = hdus[0].header['f_sampli'] * u.Hz
        if band in ["1mm", '1', '3']:
            bmaj = hdus[0].header['FWHM_260'] * u.arcsec
        elif band in ["2mm", '2']:
            bmaj = hdus[0].header['FWHM_150'] * u.arcsec
        else:
            bmaj = None

    return f_sampling, bmaj


def fits_nikamap_reader(filename, band="1mm", revert=False, **kwd):
    """NIKA2 IDL Pipeline Map reader

    Parameters
    ----------
    filenames : list
        the list of fits files to produce the Jackknifes
    band : str (1mm | 2mm | 1 | 2 | 3)
        the requested band
    revert : boolean
         use if to return -1 * data
    """

    assert band in ['1mm', '2mm', '1', '2',
                    '3'], "band should be either '1mm', '2mm', '1', '2', '3'"

    f_sampling, bmaj = retrieve_primary_keys(filename, band, **kwd)

    with fits.open(filename, **kwd) as hdus:

        data = hdus['Brightness_{}'.format(band)].data
        header = hdus['Brightness_{}'.format(band)].header
        e_data = hdus['Stddev_{}'.format(band)].data
        hits = hdus['Nhits_{}'.format(band)].data

    if 'BMAJ' not in header:  # pragma: no cover  # old file format
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
