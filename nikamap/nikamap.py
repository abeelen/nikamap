from __future__ import absolute_import, division, print_function

from itertools import product
from collections.abc import MutableMapping
from pathlib import Path

import numpy as np
from copy import deepcopy

from astropy.io import fits, registry
from astropy import units as u
from astropy.wcs import WCS
from astropy.nddata import StdDevUncertainty

from scipy import signal
from scipy.optimize import curve_fit

import warnings
from astropy.utils.exceptions import AstropyWarning

from .contmap import ContMap
from .utils import update_header

Jy_beam = u.Jy / u.beam

__all__ = ["NikaMap", "NikaFits"]


# TODO: Take care of operations (add/subtract/...) to add extra parameters...


class NikaMap(ContMap):
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

        self.primary_header = kwargs.pop("primary_header", None)

        super().__init__(data, *args, **kwargs)

        if isinstance(data, NikaMap):
            if self.primary_header is None and data.primary_header is not None:
                self.primary_header = data.primary_header


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

        data = hdus[brightness_key].data.astype(float)
        header = hdus[brightness_key].header
        e_data = hdus[stddev_key].data.astype(float)
        hits = hdus[hits_key].data.astype(int)

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
        data = data_hdu[0].data.astype(float)
        rgw = rgw_hdu[0].data.astype(float)
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
        if isinstance(data, ContMap):
            if self.hits is None and data.hits is not None:
                self.hits = data.hits
            if self.beam is None and data.beam is not None:
                self.beam = data.beam

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
        for band, item in self.items():
            hdus += item.to_hdus(
                hdu_data="Brightness_{}".format(band),
                hdu_mask=None,
                hdu_uncertainty="Stddev_{}".format(band),
                hdu_hits="Nhits_{}".format(band),
            )
        hdus = fits.HDUList(hdus)
        hdus.writeto(filename, **kwargs)
