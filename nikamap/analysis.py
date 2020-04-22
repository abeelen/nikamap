from __future__ import absolute_import, division, print_function

import os
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import StdDevUncertainty
from astropy.utils.console import ProgressBar

from .nikamap import retrieve_primary_keys, NikaMap
from .utils import update_header

__all__ = ["Jackknife", "bootstrap"]


def check_filenames(filenames, band="1mm", n=None):
    """Check the existence and compatibility of a list of NIKA IDL fits

    Parameters
    ----------
    filenames : list
        the list of NIKA fits files
    band : str (1mm | 2mm | 1 | 2 | 3)
        the requested band
    n : int or None, optionnal
        if not None check parity of filenames

    Returns
    -------
    line
        the curated list of filenames
    """

    assert band in ["1mm", "2mm", "1", "2", "3"], "band should be either '1mm', '2mm', '1', '2', '3'"

    assert isinstance(n, (int, np.int32, np.int64)) or n is None, "n must be an int or None"

    # Chek for existence
    checked_filenames = []
    for filename in filenames:
        if os.path.isfile(filename):
            checked_filenames.append(filename)
        else:
            warnings.warn("{} does not exist, removing from list".format(filename), UserWarning)

    filenames = checked_filenames
    header = fits.getheader(filenames[0], "Brightness_{}".format(band))
    w = WCS(header)

    # Checking all header for consistency
    for filename in filenames:
        _header = fits.getheader(filename, "Brightness_{}".format(band))
        _w = WCS(_header)
        assert w.wcs == _w.wcs, "{} has a different header".format(filename)
        for key in ['UNIT', 'NAXIS1', 'NAXIS2']:
            assert header[key] == _header[key], "{} has a different key".format(filename, key)

    if n is not None and len(filenames) % 2:
        warnings.warn("Even number of files, dropping the last one", UserWarning)
        filenames = filenames[:-1]

    return filenames


class Jackknife:
    """A class to create weighted Jackknife maps from a list of IDL fits files.

    This acts as a python lazy iterator and/or a callable

    Parameters
    ----------
    filenames : list
        the list of fits files to produce the Jackknifes
    band : str (1mm | 2mm | 1 | 2 | 3)
        the requested band
    n : int
        the number of Jackknifes maps to be produced

            if set to `None`, produce one weighted average of the maps

    parity_threshold : float
        mask threshold between 0 and 1 to keep partially jackknifed area
        * 1 pure jackknifed
        * 0 partially jackknifed, keep all

    Notes
    -----
    A crude check is made on the wcs of each map when instanciated
    """

    def __init__(self, filenames, band="1mm", n=1, parity_threshold=1, **kwd):

        self.i = 0
        self.n = n
        self.band = band
        self.parity_threshold = parity_threshold

        filenames = check_filenames(filenames, band=band, n=n)
        assert len(filenames) > 1, "Less than 2 existing files in filenames"

        self.filenames = filenames

        self.primary_header = fits.getheader(filenames[0])

        header = fits.getheader(filenames[0], "Brightness_{}".format(band))

        # Retrieve common keywords
        f_sampling, bmaj = retrieve_primary_keys(filenames[0], band, **kwd)
        header = update_header(header, bmaj)

        self.header = header
        self.shape = (header["NAXIS2"], header["NAXIS1"])

        # This is a low_mem=False case ...
        # TODO: How to refactor that for low_mem=True ?
        datas = np.zeros((len(filenames),) + self.shape)
        weights = np.zeros((len(filenames),) + self.shape)
        time = np.zeros(self.shape) * u.h

        for i, filename in enumerate(filenames):

            with fits.open(filename, **kwd) as hdus:

                f_sampling = hdus[0].header["f_sampli"] * u.Hz
                nhits = hdus["Nhits_{}".format(band)].data

                # Time always adds up
                time += nhits / f_sampling

                datas[i, :, :] = hdus["Brightness_{}".format(band)].data
                with np.errstate(invalid="ignore", divide="ignore"):
                    weights[i, :, :] = hdus["Stddev_{}".format(band)].data ** -2

                weights[i, nhits == 0] = 0

        unobserved = time == 0
        # Base jackknife weights
        jk_weights = np.ones(len(filenames))

        if n is not None:
            jk_weights[::2] *= -1

        self.datas = datas
        self.weights = weights
        self.time = time
        self.mask = unobserved
        self.jk_weights = jk_weights

    @property
    def parity_threshold(self):
        return self._parity

    @parity_threshold.setter
    def parity_threshold(self, value):
        if value is not None and isinstance(value, (int, float)) and 0 <= value <= 1:
            self._parity = value
        else:
            raise TypeError("parity must be between 0 and 1")

    def __len__(self):
        # to retrieve the legnth of the iterator, enable ProgressBar on it
        return self.n

    def __iter__(self):
        # Iterators are iterables too.
        # Adding this functions to make them so.
        return self

    def __call__(self):
        """Compute a jackknifed dataset

        Returns
        -------
        :class:`nikamap.NikaMap`
            a jackknifed data set
        """
        np.random.shuffle(self.jk_weights)

        with np.errstate(invalid="ignore", divide="ignore"):
            e_data = 1 / np.sqrt(np.sum(self.weights, axis=0))
            data = np.sum(self.datas * self.weights * self.jk_weights[:, np.newaxis, np.newaxis], axis=0) * e_data ** 2
            parity = np.mean((self.weights != 0) * self.jk_weights[:, np.newaxis, np.newaxis], axis=0)
            weighted_parity = np.sum(self.weights * self.jk_weights[:, np.newaxis, np.newaxis], axis=0) * e_data ** 2

        if self.n is not None:
            mask = (1 - np.abs(parity)) < self.parity_threshold
        else:
            mask = parity < self.parity_threshold

        mask = mask | self.mask

        data[mask] = np.nan
        e_data[mask] = np.nan

        # TBC: time will have a different mask here....
        data = NikaMap(
            data,
            mask=mask,
            uncertainty=StdDevUncertainty(e_data),
            unit=self.header["UNIT"],
            wcs=WCS(self.header),
            meta={"header": self.header, "primary_header": self.primary_header},
            time=self.time,
        )

        return data  # , weighted_parity

    def __next__(self):
        """Iterator on the Jackknife object"""
        if self.n is None or self.i < self.n:
            # Produce Jackkife data until last iter
            self.i += 1
            data = self.__call__()
        else:
            raise StopIteration()

        return data


def bootstrap(filenames, band="1mm", n_bootstrap=200, wmean=False, ipython_widget=False):
    """Perform Bootstrap analysis on a set of IDL nika fits files"""

    filenames = check_filenames(filenames, band=band, n=None)

    n_scans = len(filenames)
    header = fits.getheader(filenames[0], "Brightness_{}".format(band))
    primary_header = fits.getheader(filenames[0])

    f_sampling, bmaj = retrieve_primary_keys(filenames[0], band)
    header = update_header(header, bmaj)

    shape = (header["NAXIS2"], header["NAXIS1"])

    datas = np.zeros((n_scans,) + tuple(shape), dtype=np.float)
    hits = np.zeros(shape, dtype=np.float)

    bs_array = np.zeros((n_bootstrap,) + tuple(shape), dtype=np.float)

    # To avoid large memory allocation
    if wmean:
        weights = np.zeros((n_scans,) + tuple(shape))

    for index, filename in enumerate(filenames):
        with fits.open(filename, "readonly") as fits_file:
            datas[index] = fits_file["Brightness_{}".format(band)].data
            hits += fits_file["Nhits_{}".format(band)].data
            if wmean:
                stddev = fits_file["Stddev_{}".format(band)].data
                with np.errstate(divide="ignore"):
                    weights[index] = 1 / stddev ** 2

    if wmean:
        mask = ~np.isfinite(weights)
        datas = np.ma.array(datas, mask=mask)
        weights[mask] = 0

    # This is where the magic happens
    for index in ProgressBar(np.arange(n_bootstrap), ipython_widget=ipython_widget):
        shuffled_index = np.floor(np.random.uniform(0, n_scans, n_scans)).astype(np.int)
        if wmean:
            bs_array[index, :, :] = np.ma.average(
                datas[shuffled_index, :, :], weights=weights[shuffled_index, :, :], axis=0, returned=False
            )
        else:
            bs_array[index, :, :] = np.mean(datas[shuffled_index, :, :], axis=0)

    data = np.mean(bs_array, axis=0)
    e_data = np.std(bs_array, axis=0, ddof=1)

    time = (hits / f_sampling).to(u.h)

    # Mask unobserved regions
    unobserved = hits == 0
    data[unobserved] = np.nan
    e_data[unobserved] = np.nan

    data = NikaMap(
        data,
        mask=unobserved,
        uncertainty=StdDevUncertainty(e_data),
        unit=header["UNIT"],
        wcs=WCS(header),
        meta={"header": header, "primary_header": primary_header},
        time=time,
    )

    return data
