from __future__ import absolute_import, division, print_function

import os
import warnings
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units as u
from astropy.nddata import StdDevUncertainty

from .nikamap import retrieve_primary_keys, NikaMap

__all__ = ['jackknife', 'bootstrap']


def check_filenames(filenames, band="1mm"):
    """Check the existence and compatibility of a list of NIKA IDL fits"""

    assert band in ['1mm', '2mm', '1', '2', '3'], "band should be either '1mm', '2mm', '1', '2', '3'"

    # Chek for existence
    checked_filenames = []
    for filename in filenames:
        if os.path.isfile(filename):
            checked_filenames.append(filename)
        else:
            warnings.warn('{} does not exist, removing from list'.format(
                filename), UserWarning)

    filenames = checked_filenames
    header = fits.getheader(filenames[0], 'Brightness_{}'.format(band))

    # Checking all header for consistency
    for filename in filenames:
        _header = fits.getheader(filename, 'Brightness_{}'.format(band))
        assert WCS(header).wcs == WCS(
            _header).wcs, '{} has a different header'.format(filename)
        assert header['UNIT'] == _header['UNIT'], '{} has a different uni'.format(filename)
        assert WCS(header)._naxis1 == WCS(
            _header)._naxis1, '{} has a different shape'.format(filename)
        assert WCS(header)._naxis2 == WCS(
            _header)._naxis2, '{} has a different shape'.format(filename)

    return filenames


class jackknife:
    """A class to create weighted Jackknife maps from a list of fits files.

    This acts as a python generator.

    Parameters
    ----------
    filenames : list
        the list of fits files to produce the Jackknifes
    band : str (1mm | 2mm | 1 | 2 | 3)
        the requested band
    n : int
        the number of Jackknifes maps to be produced

            if set to `None`, produce one weighted average of the maps

    Notes
    -----
    A crude check is made on the wcs of each map when instanciated
    """

    def __init__(self, filenames, band='1mm', n=10, low_mem=False, **kwd):

        self._iter = iter(self)  # Py2-style
        self.i = 0
        self.n = n
        self.band = band
        self.low_mem = False

        filenames = check_filenames(filenames, band=band)
        assert len(filenames) > 1, 'Less than 2 existing files in filenames'

        if len(filenames) % 2 and n is not None:
            warnings.warn('Even number of files, dropping the last one', UserWarning)
            filenames = filenames[:-1]

        self.filenames = filenames

        header = fits.getheader(filenames[0], 'Brightness_{}'.format(band))

        # Retrieve common keywords
        f_sampling, bmaj = retrieve_primary_keys(filenames[0], band, **kwd)

        header['BMAJ'] = (bmaj.to(u.deg).value, '[deg],  Beam major axis')
        header['BMIN'] = (bmaj.to(u.deg).value, '[deg],  Beam minor axis')

        self.header = header

        # This is low_mem=False case ...
        datas = np.zeros((len(filenames), header['NAXIS2'], header['NAXIS1']))
        weights = np.zeros((len(filenames), header['NAXIS2'], header['NAXIS1']))
        time = np.zeros((header['NAXIS2'], header['NAXIS1'])) * u.h

        for i, filename in enumerate(filenames):

            with fits.open(filename, **kwd) as hdus:

                f_sampling = hdus[0].header['f_sampli'] * u.Hz
                nhits = hdus['Nhits_{}'.format(band)].data

                # Time always adds up
                time += nhits / f_sampling

                datas[i, :, :] = hdus['Brightness_{}'.format(band)].data
                with np.errstate(invalid='ignore', divide='ignore'):
                    weights[i, :, :] = hdus['Stddev_{}'.format(band)].data**-2

                weights[i, nhits == 0] = 0

        unobserved = time == 0
        # Base jackknife weights
        jk_weights = np.ones(len(filenames))
        jk_weights[::2] *= -1

        self.datas = datas
        self.weights = weights
        self.time = time
        self.mask = unobserved
        self.jk_weights = jk_weights

    def __iter__(self):
        # Iterators are iterables too.
        # Adding this functions to make them so.
        return self

    def next(self):  # pragma: no cover        # Py2-style
        return self._iter.__next__()

    def __next__(self):
        if self.n is None:
            # No Jackknife, just co-addition
            self.i = self.n = 0
            with np.errstate(invalid='ignore', divide='ignore'):
                e_data = np.sum(self.weights, axis=0)**(-0.5)
                data = np.sum(self.datas * self.weights, axis=0) * e_data**2

        elif self.i < self.n:

            # Produce Jackkife data until last iter
            self.i += 1
            np.random.shuffle(self.jk_weights)
            with np.errstate(invalid='ignore', divide='ignore'):
                e_data = np.sum(self.weights, axis=0)**(-0.5)
                data = np.sum(self.datas * self.weights *
                              self.jk_weights[:, np.newaxis, np.newaxis], axis=0) * e_data**2

        else:
            raise StopIteration()

        data[self.mask] = np.nan
        e_data[self.mask] = np.nan

        data = NikaMap(data, mask=self.mask,
                       uncertainty=StdDevUncertainty(e_data),
                       unit=self.header['UNIT'], wcs=WCS(self.header),
                       meta=self.header, time=self.time)

        return data


def bootstrap(filenames, band="1mm", n_bootstrap=200, wmean=False):
    """Perform Bootstrap analysis on a set of IDL nika files"""

    filenames = check_filenames(filenames, band=band)

    f_sampling, bmaj = retrieve_primary_keys(filenames[0], band)

    n_scans = len(filenames)
    header = fits.getheader(filenames[0], 'Brightness_{}'.format(band))

    if 'BMAJ' not in header:  # pragma: no cover  # old file format
        header['BMAJ'] = (bmaj.to(u.deg).value, '[deg],  Beam major axis')
        header['BMIN'] = (bmaj.to(u.deg).value, '[deg],  Beam minor axis')

    shape = (header['NAXIS2'], header['NAXIS1'])

    datas = np.zeros((n_scans,) + tuple(shape), dtype=np.float)
    hits = np.zeros(shape, dtype=np.float)

    bs_array = np.zeros((n_bootstrap,) + tuple(shape), dtype=np.float)

    # To avoid large memory allocation
    if wmean:
        weights = np.zeros((n_scans,) + tuple(shape))

    for index, filename in enumerate(filenames):
        with fits.open(filename, 'readonly') as fits_file:
            datas[index] = fits_file['Brightness_{}'.format(band)].data
            hits += fits_file['Nhits_{}'.format(band)].data
            if wmean:
                stddev = fits_file['Stddev_{}'.format(band)].data
                with np.errstate(divide='ignore'):
                    weights[index] = 1 / stddev**2

    if wmean:
        mask = ~np.isfinite(weights)
        datas = np.ma.array(datas, mask=mask)
        weights[mask] = 0

    # This is where the magic happens
    for index in np.arange(n_bootstrap):
        shuffled_index = np.floor(np.random.uniform(
            0, n_scans, n_scans)).astype(np.int)
        if wmean:
            bs_array[index, :, :] = np.ma.average(datas[shuffled_index, :, :],
                                                  weights=weights[shuffled_index, :, :],
                                                  axis=0, returned=False)
        else:
            bs_array[index, :, :] = np.mean(datas[shuffled_index, :, :], axis=0)

    data = np.mean(bs_array, axis=0)
    e_data = np.std(bs_array, axis=0, ddof=1)

    time = (hits / f_sampling).to(u.h)

    # Mask unobserved regions
    unobserved = hits == 0
    data[unobserved] = np.nan
    e_data[unobserved] = np.nan

    data = NikaMap(data, mask=unobserved, uncertainty=StdDevUncertainty(
        e_data), unit=header['UNIT'], wcs=WCS(header), meta=header, time=time)

    return data
