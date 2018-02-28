"""
===========
Basic usage
===========

This example shows the basic operation on the :class:`nikamap.NikaMap` object

"""
import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord, Angle

from nikamap import NikaMap

###############################################################################
# Read the data
# -------------
#
# By default the read routine will read the 1mm band, but any band can
# be read
#
# .. note:: This fake dataset as been generated by the :mod:`fake_map.py` script

data_path = os.getcwd()
nm = NikaMap.read(os.path.join(data_path, 'fake_map.fits'))


################################################################################
# NikaMap is derived from the `astropy.NDData` class and thus you can access and and manipulate the data the same way
#
# * `nm.data` : an np.array containing the brightness
# * `nm.wcs` : a WCS object describing the astrometry of the image
# * `nm.uncertainy.array` : a np.array containing the uncertainty array
# * `nm.mask` : a boolean mask of the observations
# * `nm.meta` : a copy of the header of the original map

print(nm)

###############################################################################
#
print(nm.wcs)

################################################################################
# NikaMap objects support slicing like numpy arrays, thus one can access
# part of the dataset

print(nm[96:128, 96:128])

################################################################################
# Basic Plotting
# --------------
# thus they can be plotted directly using maplotlib routines

plt.imshow(nm.data)

################################################################################
# or using the convience routine of :class:`nikamap.NikaMap`
#

fig, axes = plt.subplots(ncols=2, subplot_kw={'projection': nm.wcs})
levels = np.logspace(np.log10(2 * 1e-3), np.log10(10e-3), 4)
nm[275:300, 270:295].plot(ax=axes[0], levels=levels)
nm[210:260, 260:310].plot(ax=axes[1], levels=levels)

################################################################################
#
nm.plot_SNR(cbar=True)

################################################################################
# or the power spectrum density of the data :

fig, ax = plt.subplots()
powspec, bins = nm.plot_PSD(ax=ax)

islice = nm.get_square_slice()
_ = nm[islice, islice].plot_PSD(ax=ax)

################################################################################
# Beware that these PSD are based on an non-uniform noise, thus dominated by the
# largest noise part of the map

################################################################################
# Match filtering
# ---------------
#
# A match filter algorithm can be applied to the data to improve
# the detectability of sources. Here using the gaussian beam as the filter

mf_nm = nm.match_filter(nm.beam)
mf_nm.plot_SNR()

################################################################################
# Source detection & photometry
# -----------------------------
#
# A peak finding algorithm can be applied to the SNR datasets

mf_nm.detect_sources(threshold=3)

################################################################################
# The resulting catalog is stored in the `sources` property of the :class:`nikamap.NikaMap` object

print(mf_nm.sources)

################################################################################
# and can be overploted on the SNR maplotlib
mf_nm.plot_SNR(cat=True)

################################################################################
# There is two available photometries :
# * **peak_flux** : to retrieve point sources flux directly on the pixel value of the map, ideadlly on the matched filtered map
# * **psf_flux** : which perfom psf fitting on the pixels at the given position

mf_nm.phot_sources(peak=True, psf=False)

################################################################################
# catalog which can be transfered to the un-filtered dataset, where psf fitting can be performed

nm.phot_sources(sources=mf_nm.sources, peak=False, psf=True)

################################################################################
# the `sources` attribute now contains both photometries

print(nm.sources)

################################################################################
# which can be compared to the original fake source catalog
fake_sources = Table.read('fake_map.fits', 'FAKE_SOURCES')
fake_sources.meta['name'] = 'fake sources'
nm.plot_SNR(cat=[(fake_sources, '^'), (nm.sources, '+')])

################################################################################
# or in greater details :

fake_coords = SkyCoord(fake_sources['ra'], fake_sources['dec'], unit="deg")
detected_coords = SkyCoord(nm.sources['ra'], nm.sources['dec'], unit="deg")

idx, sep2d, _ = fake_coords.match_to_catalog_sky(detected_coords)
good = sep2d < 10 * u.arcsec
idx = idx[good]
sep2d = sep2d[good]

ra_off = Angle(fake_sources[good]['ra'] - nm.sources[idx]['ra'], 'deg')
dec_off = Angle(fake_sources[good]['dec'] - nm.sources[idx]['dec'], 'deg')

fig, axes = plt.subplots(ncols=2)
for method in ['flux_psf', 'flux_peak']:
    axes[0].errorbar(fake_sources[good]['amplitude'],
                     nm.sources[idx][method],
                     yerr=nm.sources[idx]['e{}'.format(method)],
                     fmt='o',
                     label=method)
axes[0].legend(loc='best')
axes[0].set_xlabel('input flux [mJy]')
axes[0].set_ylabel('detected flux [mJy]')

axes[1].scatter(ra_off.arcsecond, dec_off.arcsecond)
axes[1].set_xlabel('R.A. off [arcsec]')
axes[1].set_ylabel('Dec. off [arcsec]')