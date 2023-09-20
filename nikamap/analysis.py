from __future__ import absolute_import, division, print_function

import os
import warnings
import numpy as np
from functools import partial

from astropy import units as u
from astropy.nddata import StdDevUncertainty, InverseVariance, VarianceUncertainty
from astropy.utils.console import ProgressBar
from astropy.wcs import WCS

from .nikamap import NikaMap
from .utils import _shuffled_average, cpu_count

__all__ = ["HalfDifference", "Jackknife", "Bootstrap"]


def compare_header(header_ref, header_target):
    """Crude comparison of two header

    Parameters
    ----------
    header_ref : astropy.io.fits.Header
        the reference header
    header_target : astropy.io.fits.Header
        the target header to check

    Notes
    -----
    This will raise assertion error if the two header are not equivalent
    """
    wcs_ref = WCS(header_ref)
    wcs_target = WCS(header_target)

    assert wcs_ref.wcs == wcs_target.wcs, "Different header found"
    for key in ["UNIT", "NAXIS1", "NAXIS2"]:
        if key in header_ref:
            assert header_ref[key] == header_target[key], "Different key found"


def check_filenames(filenames):
    """check filenames existence

    Parameters
    ----------
    filenames : list of str
        filenames list to be checked

    Returns
    -------
    list of str
        curated list of files
    """
    _filenames = []
    for filename in filenames:
        if os.path.isfile(filename):
            _filenames.append(filename)
        else:
            warnings.warn("{} does not exist, removing from list".format(filename), UserWarning)
    return _filenames


class MultiScans(object):
    """A class to hold multi single scans from a list of fits files.

    This acts as a python lazy iterator and/or a callable

    Parameters
    ----------
    filenames : list or `~MultiScans` object
        the list of fits files to produce the Jackknifes or an already filled object
    ipython_widget : bool, optional
        If True, the progress bar will display as an IPython notebook widget.
    ignore_header : bool, optional
        if True, the check on header is ignored
    n : int
        the number of iteration for the iterator

    Notes
    -----
    A crude check is made on the wcs of each map when instanciated
    """

    dataclass = None
    filenames = None
    header = None
    unit = None
    shape = None
    datas = None
    weights = None
    hits = None
    mask = None
    extra_kwargs = dict()

    def __init__(self, filenames, n=None, ipython_widget=False, ignore_header=False, dataclass=NikaMap, **kwd):
        self.i = 0
        self.n = n
        self.dataclass = dataclass
        self.kwargs = kwd
        self.ipython_widget = ipython_widget

        if isinstance(filenames, MultiScans):
            data = filenames

            self.filenames = data.filenames
            self.header = data.header
            self.unit = data.unit
            self.shape = data.shape
            self.datas = data.datas
            self.weights = data.weights
            self.hits = data.hits
            self.mask = data.mask

            for key in ["sampling_freq", "primary_header"]:
                if hasattr(data, key):
                    self.extra_kwargs[key] = getattr(data, key)
        else:
            self.filenames = check_filenames(filenames)

            nm = self.dataclass.read(self.filenames[0], **kwd)

            self.header = nm.meta
            self.unit = nm.unit
            self.shape = nm.shape

            for key in ["sampling_freq", "primary_header"]:
                if hasattr(nm, key):
                    self.extra_kwargs[key] = getattr(nm, key)

            # This is a low_mem=False case ...
            # TODO: How to refactor that for low_mem=True ?
            datas = np.zeros((len(self.filenames),) + self.shape)
            weights = np.zeros((len(self.filenames),) + self.shape)
            hits = np.zeros(self.shape)

            for i, filename in enumerate(ProgressBar(self.filenames, ipython_widget=self.ipython_widget)):
                nm = self.dataclass.read(filename, **kwd)
                try:
                    compare_header(self.header, nm.meta)
                except AssertionError as e:
                    if ignore_header:
                        warnings.warn("{} for {}".format(e, filename), UserWarning)
                    else:
                        raise ValueError("{} for {}".format(e, filename))

                datas[i, :, :] = nm.data
                with np.errstate(invalid="ignore", divide="ignore"):
                    weights[i, :, :] = nm.uncertainty.array**-2
                hits += nm.hits

                # make sure that we do not have nans in the data
                unobserved = nm.hits == 0
                datas[i, unobserved] = 0
                weights[i, unobserved] = 0

            self.datas = datas
            self.weights = weights
            self.hits = hits
            self.mask = hits == 0

    def __len__(self):
        # to retrieve the legnth of the iterator, enable ProgressBar on it
        return self.n

    def __iter__(self):
        # Iterators are iterables too.
        # Adding this functions to make them so.
        return self

    def __call__(self):
        """The main method which should be overrided

        should return a  :class:`nikamap.NikaMap`
        """
        pass

    def __next__(self):
        """Iterator on the objects"""
        if self.n is None or self.i < self.n:
            # Produce data until last iter
            self.i += 1
            data = self.__call__()
        else:
            raise StopIteration()

        return data


class HalfDifference(MultiScans):
    """A class to create weighted half differences uncertainty maps from a list of scans.

    This acts as a python lazy iterator and/or a callable

    Parameters
    ----------
    filenames : list
        the list of fits files to produce the Jackknifes
    ipython_widget : bool, optional
        If True, the progress bar will display as an IPython notebook widget.
    n : int
        the number of Jackknifes maps to be produced in the iterator

            if set to `None`, produce only one weighted average of the maps

    parity_threshold : float
        mask threshold between 0 and 1 to keep partially jackknifed area
        * 1 pure jackknifed
        * 0 partially jackknifed, keep all


    Notes
    -----
    A crude check is made on the wcs of each map when instanciated
    """

    def __init__(self, filenames, parity_threshold=1, **kwd):
        super(HalfDifference, self).__init__(filenames, **kwd)
        self.parity_threshold = parity_threshold

        # Create weights for Half differences
        jk_weights = np.ones(len(self.filenames))

        if self.n is not None:
            jk_weights[::2] *= -1

        if self.n is not None and len(self.filenames) % 2:
            warnings.warn("Even number of files, dropping a random file", UserWarning)
            jk_weights[-1] = 0

        assert np.sum(jk_weights != 0), "Less than 2 existing files in filenames"

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

    def __call__(self):
        """Compute a Half Difference dataset

        Returns
        -------
        :class:`nikamap.NikaMap`
            a half difference data set
        """
        np.random.shuffle(self.jk_weights)

        with np.errstate(invalid="ignore", divide="ignore"):
            e_data = 1 / np.sqrt(np.sum(self.weights, axis=0))
            data = np.sum(self.datas * self.weights * self.jk_weights[:, np.newaxis, np.newaxis], axis=0) * e_data**2
            parity = np.mean((self.weights != 0) * self.jk_weights[:, np.newaxis, np.newaxis], axis=0)
            # TBC: In principle we should use a weighted parity to avoid different scans/weights problems
            # weighted_parity = np.sum(self.weights * self.jk_weights[:, np.newaxis, np.newaxis], axis=0) * e_data ** 2

        if self.n is not None:
            mask = (1 - np.abs(parity)) < self.parity_threshold
        else:
            mask = parity < self.parity_threshold

        mask = mask | self.mask

        data[mask] = np.nan
        e_data[mask] = np.nan

        # TBC: hits will have a different mask here....
        data = self.dataclass(
            data,
            mask=mask,
            uncertainty=StdDevUncertainty(e_data),
            hits=self.hits,
            unit=self.unit,
            wcs=WCS(self.header),
            meta=self.header,
            **self.extra_kwargs,
        )

        return data  # , weighted_parity


class Jackknife(MultiScans):
    """A class to create weighted Jackknife maps from a list of scans.

    This acts as a python lazy iterator and/or a callable

    Parameters
    ----------
    filenames : list
        the list of fits files to produce the Jackknifes
    n_samples : int
        The number of (sub) samples to use (from 2 to len(filenames))
    parity_threshold : float
        mask threshold between 0 and 1 to keep partially jackknifed area
        * 1 pure jackknifed
        * 0 partially jackknifed, keep all
    ipython_widget : bool, optional
        If True, the progress bar will display as an IPython notebook widget.
    n : int
        the number of Jackknifes maps to be produced by the iterator


    Notes
    -----
    A crude check is made on the wcs of each map when instanciated
    """

    def __init__(self, filenames, n_samples=None, parity_threshold=1, **kwd):
        super(Jackknife, self).__init__(filenames, **kwd)

        assert len(self.filenames) > 1, "Less than 2 existing files in filenames"

        self.n_samples = n_samples  # Will create the indexes for the sub-samples
        self.parity_threshold = parity_threshold

    @property
    def parity_threshold(self):
        return self._parity

    @parity_threshold.setter
    def parity_threshold(self, value):
        if value is not None and isinstance(value, (int, float)) and 0 <= value <= 1:
            self._parity = value
        else:
            raise TypeError("parity must be between 0 and 1")

    @property
    def n_samples(self):
        return self._n_samples

    @n_samples.setter
    def n_samples(self, value):
        if value is None:
            value = len(self.filenames)

        assert (2 <= value) and (value <= len(self.filenames)), "n_samples must be between 2 and the number of scans"

        self._n_samples = value

        # Check compatibility between n_samples and filenames length
        n_filenames = len(self.filenames)
        remainder = n_filenames % value

        if remainder:
            warnings.warn(
                "Remainder in number of files for {} samples, dropping the last {}".format(value, remainder),
                UserWarning,
            )
            n_filenames -= remainder

        assert n_filenames, "Less than 2 existing files in filenames"

        # Create the indexes for the sub-samples
        indexes = np.repeat(np.arange(value), n_filenames // value)

        if remainder:
            indexes = np.concatenate([indexes, np.full(remainder, np.nan)])

        self.indexes = indexes

    def __call__(self):
        """Compute a jackknifed dataset

        Returns
        -------
        :class:`nikamap.NikaMap`
            a jackknifed data set
        """
        np.random.shuffle(self.indexes)

        with np.errstate(invalid="ignore", divide="ignore"):
            # Compute sub-samples
            sub_datas = []
            sub_weights = []
            for idx in range(self.n_samples):
                mask = self.indexes == idx
                data, weight = np.ma.average(self.datas[mask], weights=self.weights[mask], axis=0, returned=True)
                sub_datas.append(data)
                sub_weights.append(weight)

            sub_datas = np.ma.array(sub_datas)
            sub_weights = np.ma.array(sub_weights)

            data = np.ma.average(sub_datas, weights=sub_weights, axis=0)
            # unweighted sample variance
            V1 = self.n_samples
            e_data = np.sqrt(np.sum((sub_datas - data) ** 2, axis=0) / (V1 * (V1 - 1)))
            # TODO : weighted sample variance (NOT WORKING !!!)
            # V1 = np.sum(sub_weights, axis=0)
            # V2 = np.sum(sub_weights**2, axis=0)
            # e_data = np.sqrt(np.sum(sub_weights * (sub_datas - data)**2, axis=0)  / (V1 - V2 / V1) )
            # e_data = e_data.filled(np.nan)

            parity = np.mean(sub_weights != 0, axis=0)

            # TBC: In principle we should use a weighted parity to avoid different scans/weights problems

            mask = parity < self.parity_threshold

        mask = mask | self.mask

        data[mask] = np.nan
        e_data[mask] = np.nan

        # TBC: hits will have a different mask here....
        data = self.dataclass(
            data,
            mask=mask,
            uncertainty=StdDevUncertainty(e_data),
            hits=self.hits,
            unit=self.unit,
            wcs=WCS(self.header),
            meta=self.header,
            **self.extra_kwargs,
        )

        return data  # , weighted_parity


class Bootstrap(MultiScans):
    """A class to create bootstraped maps from a list of scans.

    This acts as a python lazy iterator and/or a callable

    Parameters
    ----------
    filenames : list
        the list of fits files to produce the Jackknifes
    n_bootstrap : int
        the number of realization to produce a bootsrapped map, by default 20 times the length of the input filename list
    ipython_widget : bool, optional
        If True, the progress bar will display as an IPython notebook widget.
    n : int
        the number of bootstrap maps to be produced by the iterator

    Notes
    -----
    A crude check is made on the wcs of each map when instanciated
    """

    def __init__(self, filenames, n_bootstrap=None, **kwd):
        super(Bootstrap, self).__init__(filenames, **kwd)

        if n_bootstrap is None:
            n_bootstrap = 50 * len(self.filenames)

        self.n_bootstrap = n_bootstrap

    def __call__(self):
        """Compute a bootstraped map

        Returns
        -------
        :class:`nikamap.NikaMap`
            a bootstraped data set
        """

        _ = partial(_shuffled_average, datas=self.datas, weights=self.weights)

        bs_array = np.concatenate(
            ProgressBar.map(
                _,
                np.array_split(np.arange(self.n_bootstrap), cpu_count()),
                ipython_widget=self.ipython_widget,
                multiprocess=True,
            )
        )

        bs_array[bs_array == 0] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            data = np.nanmean(bs_array, axis=0)
            e_data = np.nanstd(bs_array, axis=0)

        # Mask unobserved regions
        unobserved = self.hits == 0
        data[unobserved] = np.nan
        e_data[unobserved] = np.nan

        data = self.dataclass(
            data,
            mask=unobserved,
            uncertainty=StdDevUncertainty(e_data),
            hits=self.hits,
            unit=self.unit,
            wcs=WCS(self.header),
            meta=self.header,
            **self.extra_kwargs,
        )

        return data
