from __future__ import absolute_import, division, print_function

from itertools import product
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
from astropy.modeling.utils import ellipse_extent
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats.funcs import gaussian_fwhm_to_sigma, gaussian_sigma_to_fwhm
from astropy.convolution import Kernel2D, Box2DKernel, Gaussian2DKernel
from astropy.convolution.kernels import _round_up_to_odd_integer

from astropy.table import Table, MaskedColumn, Column

from astropy.nddata.ccddata import _known_uncertainties, _uncertainty_unit_equivalent_to_parent
from astropy.nddata.ccddata import _unc_cls_to_name, _unc_name_to_cls

import photutils
from photutils.psf import BasicPSFPhotometry
from photutils.psf import DAOGroup
from photutils.background import MedianBackground
from photutils.datasets import make_gaussian_sources_image
from photutils.centroids import centroid_2dg, centroid_sources


from scipy import signal
from scipy.optimize import curve_fit

import warnings
from astropy.utils.exceptions import AstropyWarning

from .utils import beam_convolve
from .utils import CircularGaussianPSF
from .utils import pos_uniform, cat_to_sc
from .utils import powspec_k
from .utils import update_header
from .utils import shrink_mask
from .utils import setup_ax
from .utils import meta_to_header

Jy_beam = u.Jy / u.beam

__all__ = ["ContMap"]


class ContBeam(Kernel2D):
    """ContBeam describe the beam of a ContMap.

    By default the beams are derived from :class:`astropy.convolution.Kernel2D` but follow the api of :class:`radio_beam.Beam'
    and implement 2D gaussian function by default, but the class should be able to handle arbitrary beam.

    Parameters
    ----------

    See also
    --------
    :class:`astropy.convolution.Gaussian2DKernel`

    """

    _major = None
    _minor = None
    _pa = None
    _pixscale = None
    default_unit = None
    support_scaling = 8

    def __init__(
        self,
        major=None,
        minor=None,
        pa=None,
        area=None,
        default_unit=u.arcsec,
        meta=None,
        pixscale=None,
        array=None,
        support_scaling=8,
        **kwargs,
    ):
        """
        Create a new Gaussian beam

        Parameters
        ----------
        major : :class:`~astropy.units.Quantity` with angular equivalency
            The FWHM major axis
        minor : :class:`~astropy.units.Quantity` with angular equivalency
            The FWHM minor axis
        pa : :class:`~astropy.units.Quantity` with angular equivalency
            The beam position angle
        area : :class:`~astropy.units.Quantity` with steradian equivalency
            The area of the beam.  This is an alternative to specifying the
            major/minor/PA, and will create those values assuming a circular
            Gaussian beam.
        default_unit : :class:`~astropy.units.Unit`
            The unit to impose on major, minor if they are specified as floats
        pixscale : :class:`~astropy.units.Quantity` with angular equivalency
            the size of the pixel
        array : array_like, optional
            replace the gaussian beam by this array

        """
        if area is not None:
            if major is not None:
                raise ValueError("Can only specify one of {major,minor,pa} " "and {area}")
            if not area.unit.is_equivalent(u.sr):
                raise ValueError("Area unit should be equivalent to steradian.")
            rad = np.sqrt(area / (2 * np.pi))
            major = rad * gaussian_sigma_to_fwhm
            minor = rad * gaussian_sigma_to_fwhm
            pa = 0.0 * u.deg

        # give specified values priority
        if major is not None:
            if u.deg.is_equivalent(major):
                major = major
            else:
                warnings.warn("Assuming major axis has been specified in degrees")
                major = major * u.deg
        if minor is not None:
            if u.deg.is_equivalent(minor):
                minor = minor
            else:
                warnings.warn("Assuming minor axis has been specified in degrees")
                minor = minor * u.deg
        if pa is not None:
            if u.deg.is_equivalent(pa):
                pa = pa
            else:
                warnings.warn("Assuming position angle has been specified in degrees")
                pa = pa * u.deg
        else:
            pa = 0.0 * u.deg

        # some sensible defaults
        if minor is None:
            minor = major

        if major is not None and minor > major:
            raise ValueError("Minor axis greater than major axis.")

        if meta is None:
            self.meta = {}
        elif isinstance(meta, dict):
            self.meta = meta
        else:
            raise TypeError("metadata must be a dictionary")

        self._major = major
        self._minor = minor
        self._pa = pa
        self.default_unit = default_unit
        self.support_scaling = support_scaling

        self._pixscale = pixscale

        if self._pixscale is None:
            raise ValueError("You must define pixscale.")

        if self._major is not None:

            stddev_maj = (self.stddev_maj / self.pixscale).decompose()
            stddev_min = (self.stddev_min / self.pixscale).decompose()
            angle = (90 * u.deg + self.pa).to(u.radian).value

            self._model = models.Gaussian2D(
                1 / (2 * np.pi * stddev_maj * stddev_min), 0, 0, x_stddev=stddev_maj, y_stddev=stddev_min, theta=angle
            )

            max_extent = np.max(ellipse_extent(stddev_maj, stddev_min, angle))
            self._default_size = _round_up_to_odd_integer(self.support_scaling * 2 * max_extent)

            super(ContBeam, self).__init__(**kwargs)
        elif array is not None:
            super(ContBeam, self).__init__(array=array, **kwargs)
        else:
            raise TypeError("Must specify either major or array")

    def __repr__(self):
        repr = "ContBeam: "
        if self._major is not None:
            repr += "BMAJ={0} BMIN={1} BPA={2} as ".format(
                self.major.to(self.default_unit), self.minor.to(self.default_unit), self.pa.to(u.deg)
            )
        repr += "{} Kernel2D at pixscale {}".format(self._array.shape, self._pixscale)
        return repr

    def to_header_keywords(self):  # pragma: no cover
        return {
            "BMAJ": self.major.to(u.deg).value,
            "BMIN": self.minor.to(u.deg).value,
            "BPA": self.pa.to(u.deg).value,
        }

    def ellipse_to_plot(self, xcen, ycen, pixscale):  # pragma: no cover
        """
        Return a matplotlib ellipse for plotting
        Parameters
        ----------
        xcen : int
            Center pixel in the x-direction.
        ycen : int
            Center pixel in the y-direction.
        pixscale : `~astropy.units.Quantity`
            Conversion from degrees to pixels.
        Returns
        -------
        ~matplotlib.patches.Ellipse
            Ellipse patch object centered on the given pixel coordinates.
        """
        from matplotlib.patches import Ellipse

        return Ellipse(
            (xcen, ycen),
            width=(self.major.to(u.deg) / pixscale).to(u.dimensionless_unscaled).value,
            height=(self.minor.to(u.deg) / pixscale).to(u.dimensionless_unscaled).value,
            # PA is 90 deg offset from x-y axes by convention
            # (it is angle from NCP)
            angle=(self.pa + 90 * u.deg).to(u.deg).value,
        )

    @property
    def major(self):
        """Beam FWHM Major Axis"""
        return self._major

    @property
    def stddev_maj(self):
        """Beam Stddev Major Axis"""
        return self._major * gaussian_fwhm_to_sigma

    @property
    def minor(self):
        """Beam FWHM Minor Axis"""
        return self._minor

    @property
    def stddev_min(self):
        """Beam Stddev Minor Axis"""
        return self._minor * gaussian_fwhm_to_sigma

    @property
    def pa(self):
        return self._pa

    @property
    def pixscale(self):
        return self._pixscale

    @property
    def sr(self):
        if self.major is not None:
            return (2 * np.pi * (self.major * self.minor) * gaussian_fwhm_to_sigma ** 2).to(u.sr)
        else:
            return (self._array.sum() / self._array.max() * (self.pixscale ** 2)).to(u.sr)

    def as_kernel(self, pixscale, **kwargs):
        """
        Returns an elliptical Gaussian kernel of the beam.
        .. warning::
            This method is not aware of any misalignment between pixel
            and world coordinates.
        Parameters
        ----------
        pixscale : `~astropy.units.Quantity`
            Conversion from angular to pixel size.
        kwargs : passed to EllipticalGaussian2DKernel
        """
        if pixscale == self._pixscale:
            return self
        elif self.major is not None:
            return ContBeam(
                major=self.major,
                minor=self.minor,
                pa=self.pa,
                meta=self.meta,
                pixscale=pixscale,
                support_scaling=self.support_scaling,
                **kwargs,
            )
        else:
            raise ValueError("Do not rescale array kernel with different pixscale")

    def convolve(self, other):
        """
        Convolve one beam with another.
        Parameters
        ----------
        other : `ContBeam` or `Beam` or `Kernel2D`
            The beam to convolve with
        Returns
        -------
        new_beam : `ContBeam`
            The convolved Beam
        """
        if self.major is not None and getattr(other, "major", None) is not None:
            # other could be a ContBeam
            new_major, new_minor, new_pa = beam_convolve(self, other)
            return ContBeam(
                major=new_major,
                minor=new_minor,
                pa=new_pa,
                meta=self.meta,
                pixscale=self.pixscale,
                support_scaling=self.support_scaling,
            )
        elif self.major is not None and isinstance(other, Gaussian2DKernel):
            warnings.warn("Assuming same pixelscale")
            major = other.model.x_fwhm.value * self.pixscale
            minor = other.model.y_fwhm.value * self.pixscale
            pa = other.model.theta.value * u.radian - 90 * u.deg
            new_major, new_minor, new_pa = beam_convolve(
                self, ContBeam(major=major, minor=minor, pa=pa, pixscale=self.pixscale)
            )
            return ContBeam(
                major=new_major,
                minor=new_minor,
                pa=new_pa,
                meta=self.meta,
                pixscale=self.pixscale,
                support_scaling=self.support_scaling,
            )
        elif isinstance(other, Kernel2D):
            other_pixscale = getattr(other, "pixscale", None)
            if other_pixscale is None:
                warnings.warn("Assuming same pixelscale")
            elif self.pixscale != other_pixscale:
                raise ValueError("Do not know hot to handle different pixscale")

            return ContBeam(array=signal.convolve(self.array, other.array), pixscale=self.pixscale)
        else:
            ValueError("Do not know how to handle {}".format(type(other)))

    @classmethod
    def from_fits_header(cls, hdr, unit=u.deg, pixscale=None):  # pragma: no cover
        """
        Instantiate the beam from a header. Attempts to extract the
        beam from standard keywords. Failing that, it looks for an
        AIPS-style HISTORY entry.
        """
        # ... given a file try to make a fits header
        # assume a string refers to a filename on disk
        if not isinstance(hdr, fits.Header):
            if isinstance(hdr, str):
                if hdr.lower().endswith((".fits", ".fits.gz", ".fit", ".fit.gz", ".fits.Z", ".fit.Z")):
                    hdr = fits.getheader(hdr)
                else:
                    raise TypeError("Unrecognized extension.")
            else:
                raise TypeError("Header is not a FITS header or a filename")

        # If we find a major axis keyword then we are in keyword
        # mode. Else look to see if there is an AIPS header.
        if "BMAJ" in hdr:
            major = hdr["BMAJ"] * unit
        else:
            hist_beam = cls.from_fits_history(hdr, pixscale=pixscale)
            if hist_beam is not None:
                return hist_beam
            else:
                raise ValueError("No BMAJ found and does not appear to be a CASA/AIPS header.")

        # Fill out the minor axis and position angle if they are
        # present. Else they will default .
        if "BMIN" in hdr:
            minor = hdr["BMIN"] * unit
        else:
            minor = None
        if "BPA" in hdr:
            pa = hdr["BPA"] * unit
        else:
            pa = None

        return cls(major=major, minor=minor, pa=pa, pixscale=pixscale)

    @classmethod
    def from_fits_history(cls, hdr, pixscale=None):  # pragma: no cover
        """
        Instantiate the beam from an AIPS header. AIPS holds the beam
        in history. This method of initializing uses the last such
        entry.
        """
        # a line looks like
        # HISTORY AIPS   CLEAN BMAJ=  1.7599E-03 BMIN=  1.5740E-03 BPA=   2.61
        if "HISTORY" not in hdr:
            return None

        aipsline = None
        for line in hdr["HISTORY"]:
            if "BMAJ" in line:
                aipsline = line

        # a line looks like
        # HISTORY Sat May 10 20:53:11 2014
        # HISTORY imager::clean() [] Fitted beam used in
        # HISTORY > restoration: 1.34841 by 0.830715 (arcsec)
        #        at pa 82.8827 (deg)

        casaline = None
        for line in hdr["HISTORY"]:
            if ("restoration" in line) and ("arcsec" in line):
                casaline = line
        # assert precedence for CASA style over AIPS
        #        this is a dubious choice

        if casaline is not None:
            bmaj = float(casaline.split()[2]) * u.arcsec
            bmin = float(casaline.split()[4]) * u.arcsec
            bpa = float(casaline.split()[8]) * u.deg
            return cls(major=bmaj, minor=bmin, pa=bpa, pixscale=None)

        elif aipsline is not None:
            bmaj = float(aipsline.split()[3]) * u.deg
            bmin = float(aipsline.split()[5]) * u.deg
            bpa = float(aipsline.split()[7]) * u.deg
            return cls(major=bmaj, minor=bmin, pa=bpa, pixscale=None)

        else:
            return None


class ContMap(NDDataArray):
    """A ContMap object represent a continuum map with additionnal capabilities.

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
    beam : :class:`~nikamap.contmap.ContBeam`
        The beam corresponding to the data, by default a gaussian
        constructed from the header 'BMAJ' 'BMIN', 'PA' keyword.
    fake_source : :class:`astropy.table.Table`, optional
        The table of potential fake sources included in the data

        .. note::
            The table must contain at least 3 columns: ['ID', 'ra', 'dec']

    sources : :class`astropy.table.Table`, optional
        The table of detected sources in the data.


    """

    _residual = None

    def __init__(self, data, *args, **kwargs):

        if "meta" not in kwargs:
            kwargs["meta"] = kwargs.pop("header", None)
        if "header" in kwargs:
            raise ValueError("can't have both header and meta.")

        # Arbitrary unit by default
        if "unit" not in kwargs:
            kwargs["unit"] = data.unit if isinstance(data, (u.Quantity, ContMap)) else "adu"

        # Must be set AFTER the super() call
        self.fake_sources = kwargs.pop("fake_sources", None)
        self.sources = kwargs.pop("sources", None)
        self.sampling_freq = kwargs.pop("sampling_freq", None)

        self.hits = kwargs.pop("hits", None)
        self.beam = kwargs.pop("beam", None)

        super().__init__(data, *args, **kwargs)

        if isinstance(data, ContMap):
            if self.hits is None and data.hits is not None:
                self.hits = data.hits
            if self.beam is None and data.beam is not None:
                self.beam = data.beam

        if isinstance(self.wcs, WCS):
            pixscale = np.abs(self.wcs.wcs.cdelt[0]) * u.deg
        else:
            pixscale = np.abs(self.meta.get("CDELT1", 1)) * u.deg

        self._pixel_scale = u.pixel_scale(pixscale / u.pixel)

        if self.beam is None:
            # Default BMAJ 1 deg...
            header = meta_to_header(self.meta)
            if "BMAJ" not in header:
                header["BMAJ"] = 1
            self.beam = ContBeam.from_fits_header(header, pixscale=pixscale)

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
            if isinstance(value, NDUncertainty):
                if getattr(value, "_parent_nddata", None) is not None:
                    value = value.__class__(value, copy=False)
                self._uncertainty = value
            elif isinstance(value, np.ndarray):
                if value.shape != self.shape:
                    raise ValueError("uncertainty must have same shape as " "data.")
                self._uncertainty = StdDevUncertainty(value)
                warnings.warn("array provided for uncertainty; assuming it is a " "StdDevUncertainty.")
            else:
                raise TypeError("uncertainty must be an instance of a " "NDUncertainty object or a numpy array.")
            self._uncertainty.parent_nddata = self
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
        item : str, optional (None|signal|uncertainty|snr|residual)
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
        elif item == 'residual':
            data = np.ma.array(self._residual * self.unit, mask=self.mask)
            label = "Residual"
        else:
            raise ValueError("must be in (None|signal|uncertainty|snr|residual)")

        return data, label

    @property
    def beam(self):
        return self._beam

    @beam.setter
    def beam(self, value):
        if value is None or isinstance(value, ContBeam):
            self._beam = value
        else:
            raise ValueError("Can not handle this beam type {}".format(type(value)))

    def _slice(self, item):
        # slice all normal attributes
        kwargs = super()._slice(item)
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
        :class:`ContMap`
            return a trimmed ContMap object

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

        sources["x_stddev"] = np.ones(nsources) * self.beam.stddev_maj.to(u.pix, self._pixel_scale).value
        sources["y_stddev"] = np.ones(nsources) * self.beam.stddev_min.to(u.pix, self._pixel_scale).value
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
                    box_size=box_size,
                )
        except InconsistentAxisTypesError:
            sources = []

        if sources is not None and len(sources) > 0:

            sources.rename_column("peak_value", "SNR")

            # To avoid #1294 photutils issue, compute the centroid outside of find_peak
            x_centroids, y_centroids = centroid_sources(
                detect_on,
                sources["x_peak"],
                sources["y_peak"],
                box_size=box_size,
                mask=self.mask,
                centroid_func=centroid_2dg,
            )

            sources["x_centroid"] = x_centroids
            sources["y_centroid"] = y_centroids

            lonlat = self.wcs.pixel_to_world_values(x_centroids, y_centroids)

            for key, item, unit in zip(self.wcs.world_axis_physical_types, lonlat, self.wcs.world_axis_units):
                sources[key] = item * u.Unit(unit)

            if "pos.eq.ra" in sources.colnames and "pos.eq.dec" in sources.colnames:
                sources.rename_columns(["pos.eq.ra", "pos.eq.dec"], ["ra", "dec"])
                # For compatibility issues
                sources["_ra"] = sources["ra"]
                sources["_dec"] = sources["dec"]

            # Transform to masked Table here to avoid future warnings
            sources = Table(sources, masked=True)
            sources.meta["method"] = "find_peak"
            sources.meta["threshold"] = threshold

            # Sort by decreasing SNR
            sources.sort("SNR")
            sources.reverse()

            sources.add_column(Column(np.arange(len(sources)), name="ID"), 0)

        if self.fake_sources:
            # Match to the fake catalog
            fake_sources = self.fake_sources
            dist_threshold = self.beam.major / 3

            if sources is None or len(sources) == 0:
                fake_sources["find_peak"] = MaskedColumn(np.ones(len(fake_sources), dtype=int), mask=True)
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

            sigma_psf = self.beam.stddev_maj.to(u.pix, self._pixel_scale).value

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
                group_maker=daogroup, bkg_estimator=mmm_bkg, psf_model=psf_model, fitter=LevMarLSQFitter(), fitshape=11
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

            self._residual = photometry.get_residual_image()

        self.sources = sources

    def match_filter(self, kernel):
        """Return a matched filtered version of the map.

        Parameters
        ----------
        kernel : :class:`nikamap.contmap.ContBeam` or any :class:`astropy.convolution.kernel2D`
            the kernel used for filtering

        Returns
        -------
        :class:`nikamap.ContMap`
            the resulting match filtered ContMap object

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
        >>> data = ContMap(data, uncertainty=StdDevUncertainty(np.ones_like(data)), time=np.ones_like(data)*u.s, mask=mask)
        >>> mf_data = data.match_filter(kernel)
        >>> import matplotlib.pypot as plt
        >>> plt.ion()
        >>> fig, axes = plt.subplots(ncols=2)
        >>> axes[0].imshow(data) ; axes[1].imshow(mf_data)
        """

        mf_beam = self.beam.convolve(kernel)

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
        if isinstance(self.uncertainty, InverseVariance):
            weights = self.uncertainty.array
        elif isinstance(self.uncertainty, StdDevUncertainty):
            weights = 1 / self.uncertainty.array ** 2
        elif isinstance(self.uncertainty, VarianceUncertainty):
            weights = 1 / self.uncertainty.array
        else:
            raise ValueError("Unknown uncertainty type")

        if self.mask is not None:
            weights[self.mask] = 0

        with np.errstate(invalid="ignore", divide="ignore"):
            mf_uncertainty = 1 / np.sqrt(signal.fftconvolve(weights, kernel_sqr, mode="same"))
        if mf_mask is not None:
            mf_uncertainty[mf_mask] = np.nan

        # Units are not propagated in masked arrays...
        mf_data = signal.fftconvolve(weights * self.__array__().filled(0), kernel, mode="same") * mf_uncertainty ** 2

        mf_data = ContMap(
            mf_data,
            mask=mf_mask,
            hits=mf_hits,
            uncertainty=StdDevUncertainty(mf_uncertainty),
            beam=mf_beam,
            unit=self.unit,
            sampling_freq=self.sampling_freq,
            wcs=self.wcs,
            meta=self.meta,
            fake_sources=self.fake_sources,
        )

        return mf_data

    def plot(self, to_plot=None, ax=None, cbar=False, cat=None, levels=None, **kwargs):
        """Convenience routine to plot the dataset.

        Parameters
        ----------
        to_plot : str, optionnal (None|signal|uncertainty|snr|residual)
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
        elif cat is None:
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
        See :func:`nikamap.ContMap.plot`for additionnal keywords
        """
        return self.plot(to_plot="snr", vmin=vmin, vmax=vmax, **kwargs)

    def check_SNR(self, ax=None, bins=100):
        """Perform normality test on SNR map.

        This perform a gaussian fit on snr pixels histogram clipped between -6 and 3

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
        To recover the normality you must multiply the uncertainty array by the returned stddev value,
        if uncertainty is StdDevUncertainty.

        >>> std = data.check_SNR()
        >>> data.uncertainty.array *= std
        """
        snr = self.snr.compressed()
        hist, bin_edges = np.histogram(snr, bins=bins, density=True, range=(-5, 5))

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

    def check_SNR_simple(self):
        """Perform normality test on SNR maps

        This perform a simple mad absolute deviation on snr pixels
        """
        snr = self.snr.compressed()
        return np.median(np.abs(snr - np.median(snr)))

    def normalize_uncertainty(self, factor=None, method="check_SNR_simple"):
        """Normalize the uncertainty.value

        Parameters
        ----------
        factor : float, optionnal
            the factor which normalize the snr distribution
        method : str, (check_SNR_simple|check_SNR),
            the method to compute this factor if not provided, by default `check_SNR_simple`
        """
        assert method in ("check_SNR_simple", "check_SNR", None)

        if factor is None:
            if method is None:
                raise ValueError("You must provide either `factor` or `method`.")
            elif method == "check_SNR_simple":
                factor = self.check_SNR_simple()
            elif method == "check_SNR":
                factor = self.check_SNR()

        if isinstance(self.uncertainty, StdDevUncertainty):
            self.uncertainty.array *= factor
        elif isinstance(self.uncertainty, InverseVariance):
            self.uncertainty.array /= factor ** 2
        elif isinstance(self.uncertainty, VarianceUncertainty):
            self.uncertainty.array *= factor ** 2
        else:
            raise ValueError("Unknown uncertainty type")

        # Add the factor in the meta
        if "FACTOR" in self.meta:
            self.meta["FACTOR"] *= factor
        else:
            self.meta["FACTOR"] = factor

    def plot_PSD(self, to_plot=None, ax=None, bins=100, range=None, apod_size=None, **kwargs):
        """Plot the power spectrum of the map.

        Parameters
        ----------
        to_plot : str, optionnal (None|signal|uncertainty|snr|residual)
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

    def _arithmetic(self, operation, operand, *args, **kwargs):
        # take all args and kwargs to allow arithmetic on the other properties
        # to work like before.
        # do the arithmetics on the flags (pop the relevant kwargs, if any!!!)
        if self.hits is not None and operand.hits is not None:
            result_hits = operation(self.hits, operand.hits)
            # np.logical_or is just a suggestion you can do what you want
        else:
            if self.hits is not None:
                result_hits = deepcopy(self.hits)
            else:
                result_hits = deepcopy(operand.hits)

        # Let the superclass do all the other attributes note that
        # this returns the result and a dictionary containing other attributes
        result, kwargs = super()._arithmetic(operation, operand, *args, **kwargs)
        # The arguments for creating a new instance are saved in kwargs
        # so we need to add another keyword "flags" and add the processed flags
        kwargs["hits"] = result_hits
        return result, kwargs  # these must be returned

    def to_hdus(
        self,
        hdu_data="DATA",
        hdu_mask="MASK",
        hdu_uncertainty="UNCERT",
        hdu_hits="HITS",
        wcs_relax=True,
        key_uncertainty_type="UTYPE",
        fits_header_comment=None,
    ):
        """Creates an HDUList object from a ContMap object.
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
        fits_header_comment : dict, optional
            A dictionnary (key, comment) to update the fits header comments.

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
            header = meta_to_header(self.header)

            if fits_header_comment is not None:
                for key, comment in fits_header_comment.items():
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


def fits_contmap_reader(
    filename,
    unit=None,
    hdu_data="DATA",
    hdu_uncertainty="UNCERT",
    hdu_mask="MASK",
    hdu_hits="HITS",
    key_uncertainty_type="UTYPE",
    **kwd,
):

    with fits.open(filename, **kwd) as hdus:

        hdr = hdus[0].header

        sampling_freq = hdr.get("sampling_freq", None)
        if sampling_freq is not None:
            sampling_freq = sampling_freq * u.Hz

        if hdu_data is not None and hdu_data in hdus:
            data = hdus[hdu_data].data
            wcs = WCS(hdus[hdu_data].header)
            if unit is None:
                unit = hdus[hdu_data].header.get("BUNIT", None)
        else:
            data = None
            wcs = None
        if hdu_uncertainty is not None and hdu_uncertainty in hdus:
            unc_hdu = hdus[hdu_uncertainty]
            stored_unc_name = unc_hdu.header.get(key_uncertainty_type, "None")

            unc_type = _unc_name_to_cls.get(stored_unc_name, StdDevUncertainty)
            uncertainty = unc_type(unc_hdu.data)
        else:
            uncertainty = None
        if hdu_mask is not None and hdu_mask in hdus:
            # Mask is saved as uint but we want it to be boolean.
            mask = hdus[hdu_mask].data.astype(np.bool_)
        else:
            mask = None
        if hdu_hits is not None and hdu_hits in hdus:
            # hits is saved as uint but we want it to be boolean.
            hits = hdus[hdu_hits].data
        else:
            hits = None

    c_data = ContMap(
        data, wcs=wcs, uncertainty=uncertainty, mask=mask, hits=hits, meta=hdr, unit=unit, sampling_freq=sampling_freq
    )

    return c_data


def fits_contmap_writer(
    c_data, filename, hdu_mask="MASK", hdu_uncertainty="UNCERT", hdu_hits="HITS", key_uncertainty_type="UTYPE", **kwd
):
    """
    Write ContMap object to FITS file.
    Parameters
    ----------
    filename : str
        Name of file.
    hdu_mask, hdu_uncertainty, hdu_hits : str or None, optional
        If it is a string append this attribute to the HDUList as
        `~astropy.io.fits.ImageHDU` with the string as extension name.
        Flags are not supported at this time. If ``None`` this attribute
        is not appended.
        Default is ``'MASK'`` for mask, ``'UNCERT'`` for uncertainty and
        ``HITS`` for flags.
    key_uncertainty_type : str, optional
        The header key name for the class name of the uncertainty (if any)
        that is used to store the uncertainty type in the uncertainty hdu.
        Default is ``UTYPE``.
        .. versionadded:: 3.1
    kwd :
        All additional keywords are passed to :py:mod:`astropy.io.fits`
    Raises
    -------
    ValueError
        - If ``self.mask`` is set but not a `numpy.ndarray`.
        - If ``self.uncertainty`` is set but not a
          `~astropy.nddata.StdDevUncertainty`.
        - If ``self.uncertainty`` is set but has another unit then
          ``self.data``.
    NotImplementedError
        Saving flags is not supported.
    """
    # Build the primary header with history and comments

    header = meta_to_header(c_data.header)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)
        if c_data.sampling_freq is not None:
            header["sampling_freq"] = c_data.sampling_freq.to(u.Hz).value

    hdu = [fits.PrimaryHDU(None, header=header)]
    hdu += c_data.to_hdus(
        hdu_mask=hdu_mask,
        hdu_uncertainty=hdu_uncertainty,
        key_uncertainty_type=key_uncertainty_type,
        hdu_hits=hdu_hits,
    )

    hdu = fits.HDUList(hdu)
    hdu.writeto(filename, **kwd)


with registry.delay_doc_updates(ContMap):
    registry.register_reader("fits", ContMap, fits_contmap_reader)
    registry.register_writer("fits", ContMap, fits_contmap_writer)
    registry.register_identifier("fits", ContMap, fits.connect.is_fits)


def contmap_average(continuum_datas, normalize=False):
    """Return weighted average of severak ContMap, using inverse variance as the weights

    Parameters
    ----------
    continuum_datas: list of class:`kidsdata.continuum_data.ContMap`
        the list of ContMap object
    normalize : bool
        normalize the uncertainty such that the snr std is 1, default False

    Returns
    -------
    data : class:`kidsdata.continuum_data.ContMap`
        the resulting combined filtered ContMap object
    """

    datas = np.array([item.data for item in continuum_datas])
    masks = np.array([item.mask for item in continuum_datas])
    hits = np.array([item.hits for item in continuum_datas])

    wcs = [item.wcs for item in continuum_datas]

    assert all([wcs[0].wcs == item.wcs for item in wcs[1:]]), "All wcs must be equal"

    weights = []
    for item in continuum_datas:
        if isinstance(item.uncertainty, InverseVariance):
            weight = item.uncertainty.array
        elif isinstance(item.uncertainty, StdDevUncertainty):
            weight = 1 / item.uncertainty.array ** 2
        elif isinstance(item.uncertainty, VarianceUncertainty):
            weight = 1 / item.uncertainty.array
        else:
            raise ValueError("Unknown uncertainty type")
        weights.append(weight)

    weights = np.array(weights)

    datas[masks] = 0.0
    weights[masks] = 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        weight = np.sum(weights, axis=0)
        data = np.sum(datas * weights, axis=0) / weight

    mask = np.isnan(data)
    hits = np.sum(hits, axis=0)

    output = ContMap(data=data, uncertainty=InverseVariance(weight), wcs=wcs[0], hits=hits, mask=mask)

    if normalize:
        output.normalize_uncertainty()

    return output
