"""
=======================
Create a fake fits file
=======================

Create a fully fake fits file for test purposes, note that this is basically
a copy of the function :func:`nikamap.fake_data`

"""
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import gaussian_fwhm_to_sigma

from photutils.datasets import make_gaussian_sources_image

Jypb = u.Jy / u.beam
mJypb = u.mJy / u.beam

np.random.seed(1)

##########################################################################
# Create a fake dataset
# ---------------------
#
# Create a fake dataset with uniformly distributed point sources
#
# .. note:: This is mostly inspired by :func:`nikamap.fake_data`


def create_dataset(shape=(512, 512),
                   fwhm=12.5 * u.arcsec, pixsize=2 * u.arcsec,
                   noise_level=1 * mJypb,
                   n_sources=50, flux_min=1 * mJypb, flux_max=10 * mJypb,
                   filename='fake_map.fits'):

    hits, uncertainty, mask = create_ancillary(
        shape, fwhm=None, noise_level=1 * mJypb)
    wcs = create_wcs(shape, pixsize=2 * u.arcsec, center=None)

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    sources = create_fake_source(shape, wcs, beam_std_pix, flux_min=flux_min, flux_max=flux_max, n_sources=n_sources)

    sources_map = make_gaussian_sources_image(shape, sources) * sources['amplitude'].unit

    data = add_noise(sources_map, uncertainty)
    hdus = create_hdulist(data, hits, uncertainty, mask,
                          wcs, sources, fwhm, noise_level)

    hdus.writeto(filename, overwrite=True)

##########################################################################
# Define the hits and uncertainty map
#


def create_ancillary(shape, fwhm=None, noise_level=1 * mJypb):
    if fwhm is None:
        fwhm = np.asarray(shape) / 2.5

    y_idx, x_idx = np.indices(shape, dtype=np.float)
    hits = np.exp(-((x_idx - shape[1] / 2)**2 / (2 * (gaussian_fwhm_to_sigma * fwhm[1])**2) +
                    (y_idx - shape[0] / 2)**2 / (2 * (gaussian_fwhm_to_sigma * fwhm[0])**2)))

    uncertainty = noise_level.to(u.Jy / u.beam).value / np.sqrt(hits)

    # with a circle for the mask
    xx, yy = np.indices(shape)
    mask = np.sqrt((xx - (shape[1] - 1) / 2)**2 +
                   (yy - (shape[0] - 1) / 2)**2) > shape[0] / 2

    return hits, uncertainty, mask

##########################################################################
# and a fake :class:`astropy.wcs.WCS`


def create_wcs(shape, pixsize=2 * u.arcsec, center=None):
    if center is None:
        center = np.asarray([0, 0]) * u.deg

    wcs = WCS(naxis=2)
    wcs.wcs.crval = center.to(u.deg).value
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize.to(u.deg)
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    return wcs

##########################################################################
# Construct a fake source catalog
# -------------------------------
#
# This define an uniformly distribued catalog of sources in a disk as an :class:`astropy.table.Table`


def create_fake_source(shape, wcs, beam_std_pix,
                       flux_min=1 * mJypb, flux_max=10 * mJypb, n_sources=1):

    peak_fluxes = np.random.uniform(flux_min.to(Jypb).value, flux_max.to(Jypb).value, n_sources) * Jypb

    sources = Table(masked=True)
    sources['amplitude'] = peak_fluxes

    # Uniformly distributed sources on a disk
    theta = np.random.uniform(0, 2 * np.pi, n_sources)
    r = np.sqrt(np.random.uniform(0, 1, n_sources)) * (shape[0] / 2 - 10)

    sources['x_mean'] = r * np.cos(theta) + shape[1] / 2
    sources['y_mean'] = r * np.sin(theta) + shape[0] / 2

    sources['x_stddev'] = np.ones(n_sources) * beam_std_pix
    sources['y_stddev'] = np.ones(n_sources) * beam_std_pix
    sources['theta'] = np.zeros(n_sources)

    ra, dec = wcs.all_pix2world(sources['x_mean'], sources['y_mean'], 0)
    sources['ra'] = ra * u.deg
    sources['dec'] = dec * u.deg
    sources['_ra'] = ra * u.deg
    sources['_dec'] = dec * u.deg
    sources.meta = {'name': 'fake catalog'}

    return sources

##########################################################################
# Construct the map with noise
# ----------------------------
#


def add_noise(sources_map, uncertainty):
    data = sources_map.to(Jypb).value + np.random.normal(loc=0, scale=1, size=sources_map.shape) * uncertainty
    return data


##########################################################################
# Pack everything into :class:`astropy.io.fits.HDUList`
#

def create_hdulist(data, hits, uncertainty, mask, wcs, sources, fwhm, noise_level):

    data[mask] = np.nan
    hits[mask] = 0
    uncertainty[mask] = 0

    header = wcs.to_header()
    header['UNIT'] = "Jy / beam", 'Fake Unit'

    primary_header = fits.header.Header()
    primary_header['f_sampli'] = 10., 'Fake the f_sampli keyword'

    # Old keyword for compatilibity
    primary_header['FWHM_260'] = fwhm.to(u.arcsec).value, '[arcsec] Fake the FWHM_260 keyword'
    primary_header['FWHM_150'] = fwhm.to(u.arcsec).value, '[arcsec] Fake the FWHM_150 keyword'

    # Traceback of the fake sources
    primary_header['nsources'] = len(sources), 'Number of fake sources'
    primary_header['noise'] = noise_level.to(u.Jy / u.beam).value, '[Jy/beam] noise level per map'

    primary = fits.hdu.PrimaryHDU(header=primary_header)

    hdus = fits.hdu.HDUList(hdus=[primary])
    for band in ['1mm', '2mm']:
        hdus.append(fits.hdu.ImageHDU(data, header=header, name='Brightness_{}'.format(band)))
        hdus.append(fits.hdu.ImageHDU(uncertainty, header=header, name='Stddev_{}'.format(band)))
        hdus.append(fits.hdu.ImageHDU(hits, header=header, name='Nhits_{}'.format(band)))
        hdus.append(fits.hdu.BinTableHDU(sources, name="fake_sources"))

    return hdus


##########################################################################
#  Finally, run the :func:`create_dataset` function, only if called directly

if __name__ == '__main__':
    create_dataset()
