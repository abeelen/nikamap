from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np

import matplotlib.pyplot as plt

from astropy.io import fits, registry
from astropy import units as u
from astropy.wcs import WCS, InconsistentAxisTypesError
from astropy.coordinates import match_coordinates_sky
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

from scipy import signal
from scipy.optimize import curve_fit

import warnings
from astropy.utils.exceptions import AstropyWarning

from .utils import CircularGaussianPSF, _round_up_to_odd_integer
from .utils import pos_uniform, cat_to_sc
from .utils import powspec_k
from .utils import update_header
from .utils import shrink_mask

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
        return self._fwhm.to(
            u.pixel, equivalencies=self._pixel_scale) * gaussian_fwhm_to_sigma

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
        """Retrieve time array as maskedQuantity"""
        return np.ma.array(self.time, mask=self.mask, fill_value=0)

    @property
    def uncertainty(self):
        return self._uncertainty

    @uncertainty.setter
    def uncertainty(self, value):
        if value is not None:
            # Ugly trick to overcome bug in NDDataArray uncertainty setter
            unit = getattr(value, 'unit', None)
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
    def SNR(self):
        return np.ma.array((self.data / self.uncertainty.array),
                           mask=self.mask)

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

        return self[axis_slice[0], axis_slice[1]]

    def add_gaussian_sources(self, within=(0, 1), cat_gen=pos_uniform, **kwargs):
        """Add gaussian sources into the map

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

        sources['amplitude'] = peak_flux.to(self.unit * u.beam)

        sources['x_mean'] = x_mean
        sources['y_mean'] = y_mean

        sources['x_stddev'] = np.ones(nsources) * self.beam.sigma_pix.value
        sources['y_stddev'] = np.ones(nsources) * self.beam.sigma_pix.value
        sources['theta'] = np.zeros(nsources)

        # Crude check to be within the finite part of the map
        if self.mask is not None:
            within_coverage = ~self.mask[sources['y_mean'].astype(int), sources['x_mean'].astype(int)]
            sources = sources[within_coverage]

        # Gaussian sources...
        self._data += make_gaussian_sources_image(shape, sources)

        # Add an ID column
        sources.add_column(Column(np.arange(len(sources)), name='ID'), 0)

        # Transform pixel to world coordinates
        a, d = self.wcs.wcs_pix2world(sources['x_mean'], sources['y_mean'], 0)
        sources.add_columns([Column(a * u.deg, name='ra'),
                             Column(d * u.deg, name='dec')])

        sources['_ra'] = sources['ra']
        sources['_dec'] = sources['dec']

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
            detect_mask = shrink_mask(self.mask, box_kernel)
            detect_on[detect_mask] = np.nan

        # TODO: Have a look at
        # ~photutils.psf.IterativelySubtractedPSFPhotometry

        # See #667 of photutils
        try:
            # To avoid bad fit warnings...
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                sources = photutils.find_peaks(
                    detect_on,
                    threshold=threshold,
                    mask=self.mask,
                    wcs=self.wcs,
                    subpixel=True,
                    box_size=box_size)
        except InconsistentAxisTypesError:
            sources = []

        if len(sources) > 0:
            # Transform to masked Table here to avoid future warnings
            sources = Table(sources, masked=True)
            sources.meta['method'] = 'find_peak'
            sources.meta['threshold'] = threshold

            # pixels values are irrelevant
            sources.remove_columns(['x_centroid', 'y_centroid', 'x_peak', 'y_peak'])
            # Only keep fitted value
            sources.remove_columns(['icrs_ra_peak', 'icrs_dec_peak', 'peak_value'])

            # Rename usefull columns
            sources.rename_column('icrs_ra_centroid', 'ra')
            sources.rename_column('icrs_dec_centroid', 'dec')

            # Copy column for compatibility
            sources['_ra'] = sources['ra']
            sources['_dec'] = sources['dec']

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
                fake_sources['find_peak'] = MaskedColumn(np.ones(len(fake_sources)), mask=True)
            else:

                fake_sc = cat_to_sc(fake_sources)
                sources_sc = cat_to_sc(sources)

                idx, sep2d, _ = match_coordinates_sky(fake_sc, sources_sc)
                mask = sep2d > dist_threshold
                fake_sources['find_peak'] = MaskedColumn(sources[idx]['ID'], mask=mask)

                idx, sep2d, _ = match_coordinates_sky(sources_sc, fake_sc)
                mask = sep2d > dist_threshold
                sources['fake_sources'] = MaskedColumn(fake_sources[idx]['ID'], mask=mask)

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
            cat_sc = cat_to_sc(cat)
            ref_sc = cat_to_sc(ref_cat)
            idx, sep2d, _ = match_coordinates_sky(cat_sc, ref_sc)
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

            sources['flux_peak'] = Column(self.data[y_idx, x_idx], unit=self.unit * u.beam).to(u.mJy)
            sources['eflux_peak'] = Column(self.uncertainty.array[y_idx, x_idx], unit=self.unit * u.beam).to(u.mJy)

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

            photometry = BasicPSFPhotometry(
                group_maker=daogroup,
                bkg_estimator=mmm_bkg,
                psf_model=psf_model,
                fitter=LevMarLSQFitter(),
                fitshape=9)

            positions = Table([Column(xx, name="x_0"),
                               Column(yy, name="y_0"),
                               Column(self.data[y_idx, x_idx], name="flux_0")])

            # Fill the mask with nan to perform correct photometry on the edge
            # of the mask, and catch numpy & astropy warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', AstropyWarning)
                warnings.simplefilter('ignore', RuntimeWarning)
                result_tab = photometry(image=np.ma.array(self.data, mask=self.mask).filled(np.nan), init_guesses=positions)

            result_tab.sort('id')
            for _source, _tab in zip(['flux_psf', 'eflux_psf'], ['flux_fit', 'flux_unc']):
                sources[_source] = Column(result_tab[_tab] * psf_model(0, 0), unit=self.unit * u.beam).to(u.mJy)
            sources['group_id'] = result_tab['group_id']

        self.sources = sources

    def match_filter(self, kernel):
        """Return a matched filtered version of the map

        Parameters
        ----------
        kernel : :class:`nikamap.NikaBeam`
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
        >>> fig, axes = plt.subplots(ncols=2)
        >>> axes[0].imshow(data) ; axes[1].imshow(mf_data)
        """

        # Is peak normalizerd on get
        beam = self.beam

        kernel.normalize('integral')

        # Assuming the same pixel_scale
        if isinstance(beam.model, models.Gaussian2D) & isinstance(kernel.model, models.Gaussian2D):
            fwhm = np.sqrt(beam.model.x_fwhm**2 + kernel.model.x_fwhm**2) * u.pixel
            fwhm = fwhm.to(u.arcsec, equivalencies=beam._pixel_scale)
            mf_beam = NikaBeam(fwhm, pixel_scale=beam._pixel_scale)
        else:
            # Using scipy.signal.convolve to extend the beam if necessary
            mf_beam = NikaBeam(array=signal.convolve(beam.array, kernel.array), pixel_scale=beam._pixel_scale)

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
        mf_time = signal.fftconvolve(self.__t_array__().filled(0), kernel, mode='same') * self.time.unit

        if mf_mask is not None:
            mf_time[mf_mask] = 0

        # Convolve the data (peak for unit conservation)
        kernel.normalize('peak')
        kernel_sqr = kernel.array**2

        # ma.filled(0) required for the fft convolution
        weights = 1. / self.uncertainty.array**2
        if self.mask is not None:
            weights[self.mask] = 0

        with np.errstate(invalid='ignore', divide='ignore'):
            mf_uncertainty = 1 / np.sqrt(signal.fftconvolve(weights, kernel_sqr, mode='same'))
        if mf_mask is not None:
            mf_uncertainty[mf_mask] = np.nan

        # Units are not propagated in masked arrays...
        mf_data = signal.fftconvolve(weights * self.__array__().filled(0), kernel, mode='same') * mf_uncertainty**2

        mf_data = NikaMap(
            mf_data,
            unit=self.unit,
            mask=mf_mask,
            time=mf_time,
            uncertainty=StdDevUncertainty(mf_uncertainty),
            wcs=self.wcs,
            meta=self.meta,
            fake_sources=self.fake_sources,
            beam=mf_beam)

        return mf_data

    def plot(self, snr=False, ax=None, cbar=False, cat=None, levels=None, **kwargs):
        """Convenience routine to plot the dataset

        Parameters
        ----------
        snr : boolean, optionnal
            Plot the signal to noise ratio instead of the signal (default: False)
        ax : :class:`matplotlib.axes.Axes`, optional
            Axe to plot the power spectrum
        cbar: boolean, optionnal
            Draw a colorbar (ax must be None)
        cat : boolean of list of tuple [(cat, symbol)], optionnal
            If True, overplot the current self.source catalog with '^' as marker.
            Otherwise overplot the given catalogs on the map.
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

        if snr:
            data = self.SNR.data
            cbar_label = 'SNR'
        else:
            data = self.__array__()
            cbar_label = 'Brightness [{}]'.format(self.unit)

        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection=self.wcs)

        cax = ax.imshow(data, origin='lower', interpolation='none', **kwargs)

        if levels is not None:
            ax.contour(data, levels=levels, alpha=0.8, colors='w')
            ax.contour(data, levels=-levels[::-1], alpha=0.8, colors='w', linestyles='dashed')

        if cbar:
            fig = ax.get_figure()
            cbar = fig.colorbar(cax)
            cbar.set_label(cbar_label)

        # In case of fake sources, overplot them
        if self.fake_sources:
            x, y = self.wcs.wcs_world2pix(self.fake_sources['_ra'], self.fake_sources['_dec'], 0)
            ax.scatter(x, y, marker='o', c='red', alpha=0.8)

        if cat is True:
            cat = [(self.sources, '^')]

        if cat is not None:
            for _cat, _mark in list(cat):
                label = _cat.meta.get('method') or _cat.meta.get('name') or 'Unknown'
                cat_sc = cat_to_sc(_cat)
                x, y = self.wcs.wcs_world2pix(cat_sc.ra, cat_sc.dec, 0)
                ax.scatter(x, y, marker=_mark, alpha=0.8, label=label)

        ax.set_xlim(0, self.shape[1])
        ax.set_ylim(0, self.shape[0])

        if cat is not None:
            ax.legend(loc='best', frameon=False)

        return cax

    def plot_SNR(self, vmin=-3, vmax=5, **kwargs):
        """Convenience method to plot the signal to noise map"""
        return self.plot(snr=True, vmin=vmin, vmax=vmax, **kwargs)

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

    def plot_PSD(self, snr=False, ax=None, bins=100, range=None, apod_size=None, **kwargs):
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
            data = np.ma.array(self.data * self.unit, mask=self.mask)

        res = (1 * u.pixel).to(u.arcsec, equivalencies=self._pixel_scale)
        powspec, bin_edges = powspec_k(data, res=res, bins=bins, range=range, apod_size=apod_size)

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

    assert band in ['1mm', '2mm', '1', '2', '3'], "band should be either '1mm', '2mm', '1', '2', '3'"

    f_sampling, bmaj = retrieve_primary_keys(filename, band, **kwd)

    with fits.open(filename, **kwd) as hdus:

        data = hdus['Brightness_{}'.format(band)].data
        header = hdus['Brightness_{}'.format(band)].header
        e_data = hdus['Stddev_{}'.format(band)].data
        hits = hdus['Nhits_{}'.format(band)].data

    header = update_header(header, bmaj)

    time = (hits / f_sampling).to(u.h)

    # Mask unobserved regions
    unobserved = hits == 0
    data[unobserved] = np.nan
    e_data[unobserved] = np.nan

    if revert:
        data *= -1

    data = NikaMap(data, mask=unobserved, uncertainty=StdDevUncertainty(e_data), unit=header['UNIT'], wcs=WCS(header), meta=header, time=time)

    return data


with registry.delay_doc_updates(NikaMap):
    registry.register_reader('fits', NikaMap, fits_nikamap_reader)
    registry.register_identifier('fits', NikaMap, fits.connect.is_fits)
