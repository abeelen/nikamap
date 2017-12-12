from __future__ import absolute_import, division, print_function

import pytest

import os.path as op

import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

from astropy.nddata import StdDevUncertainty
from astropy.modeling import models
from astropy.stats.funcs import gaussian_fwhm_to_sigma

import numpy.testing as npt
# from nikamap import nikamap as nm
# from nikamap.nikamap import NikaMap

import nikamap as nm
from ..nikamap import NikaMap


data_path = op.join(nm.__path__[0], 'data')


def test_nikamap_init():
    data = [1, 2, 3]
    nm = NikaMap(data)
    assert np.all(nm.data == np.array(data))

    # Should default to empty wcs and no unit
    assert nm.wcs is None
    assert nm.unit is None
    assert nm.uncertainty is None

    # time "empty"
    assert np.all(nm.time == 0*u.s)

    # Default pixsize 1*u.deg
    assert (1*u.pixel).to(u.deg, equivalencies=nm._pixel_scale) == 1*u.deg

    # Default beam fwhm 1*u.deg
    assert nm.beam.fwhm == 1*u.deg


def test_nikamap_init_quantity():
    data = np.array([1, 2, 3])*u.Jy/u.beam
    nm = NikaMap(data)
    assert nm.unit == u.Jy/u.beam


def test_nikamap_init_time():
    data = np.array([1, 2, 3])*u.Jy/u.beam

    time = np.array([1, 2])*u.s
    with pytest.raises(ValueError):
        nm = NikaMap(data, time=time)

    time = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        nm = NikaMap(data, time=time)

    time = np.array([1, 2, 3])*u.Hz
    with pytest.raises(ValueError):
        nm = NikaMap(data, time=time)

    time = np.array([1, 2, 3])*u.h
    nm = NikaMap(data, time=time)
    assert nm.time.unit == u.h


def test_nikamap_init_meta():
    data = np.array([1, 2, 3])
    meta = fits.header.Header()

    meta['CDELT1'] = -1./3600, 'pixel size used for pixel_scale'
    meta['BMAJ'] = 1./3600, 'Beam Major Axis'
    nm = NikaMap(data, meta=meta)
    assert (1*u.pixel).to(u.deg, equivalencies=nm._pixel_scale) == 1*u.arcsec
    assert nm.beam.fwhm == 1*u.arcsec

    # Full header
    meta['CRPIX1'] = 1
    meta['CRPIX2'] = 2
    meta['CDELT1'] = -1/3600
    meta['CDELT2'] = 1/3600
    meta['CRVAL1'] = 0
    meta['CRVAL2'] = 0
    meta['CTYPE1'] = 'RA---TAN'
    meta['CTYPE2'] = 'DEC--TAN'

    nm = NikaMap(data, meta=meta, wcs=WCS(meta))
    assert nm.wcs is not None


def test_nikamap_init_uncertainty():
    data = np.array([1, 2, 3])
    uncertainty = np.array([1, 1, 1])

    # Default to StdDevUncertainty...
    nm = NikaMap(data, uncertainty=uncertainty)
    assert isinstance(nm.uncertainty, StdDevUncertainty)
    assert np.all(nm.uncertainty.array == np.array([1, 1, 1]))

    nm_mean = nm.add(nm).divide(2)
    assert np.all(nm_mean.data == nm.data)
    npt.assert_allclose(nm_mean.uncertainty.array, np.array([1, 1, 1])/np.sqrt(2))


@pytest.fixture()
def single_source():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (26, 26)
    pixsize = 1/3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape)/2-0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1])*pixsize
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    nm = NikaMap(data, uncertainty=np.ones_like(data)/4, wcs=wcs, unit=u.Jy/u.beam)

    # Additionnal attribute just for the tests...
    nm.x = np.asarray([shape[1]/2])
    nm.y = np.asarray([shape[0]/2])
    nm.add_gaussian_sources(nsources=1, peak_flux=1*u.Jy,
                            within=(nm.y[0]/shape[0], nm.x[0]/shape[1]))
    return nm


@pytest.fixture()
def single_source_mask():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (26, 26)
    pixsize = 1/3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape)/2-0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1])*pixsize
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    xx, yy = np.indices(shape)
    mask = np.sqrt((xx-shape[1]/2)**2 + (yy-shape[0]/2)**2) > 10

    nm = NikaMap(data, uncertainty=np.ones_like(data)/4, mask=mask, wcs=wcs, unit=u.Jy/u.beam)

    # Additionnal attribute just for the tests...
    nm.x = np.asarray([shape[1]/2])
    nm.y = np.asarray([shape[0]/2])
    nm.add_gaussian_sources(nsources=1, peak_flux=1*u.Jy,
                            within=(nm.y[0]/shape[0], nm.x[0]/shape[1]))
    return nm


@pytest.fixture()
def grid_sources():
    # Larger shape to allow for wobbling
    # as beam needs to be much smaller than the map at some point..
    shape = (28, 28)
    pixsize = 1/3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape)/2-0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1])*pixsize
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    nm = NikaMap(data, uncertainty=np.ones_like(data)/4, wcs=wcs, unit=u.Jy/u.beam)

    # Additionnal attribute just for the tests...
    nm.add_gaussian_sources(nsources=2**2, peak_flux=1*u.Jy,
                            grid=True)

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources['ra'], nm.fake_sources['dec'], 0)

    nm.x = x
    nm.y = y

    return nm


@pytest.fixture()
def wobble_grid_sources():
    # Even Larger shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (60, 60)
    pixsize = 1/3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape)/2-0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1])*pixsize
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    nm = NikaMap(data, uncertainty=np.ones_like(data)/4, wcs=wcs, unit=u.Jy/u.beam)

    np.random.seed(0)
    # Additionnal attribute just for the tests...
    nm.add_gaussian_sources(nsources=2**2, peak_flux=1*u.Jy,
                            grid=True, wobble=True,
                            within=(1/3, 2/3))

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources['ra'], nm.fake_sources['dec'], 0)

    nm.x = x
    nm.y = y

    return nm


@pytest.fixture(params=['single_source', 'single_source_mask', 'grid_sources', 'wobble_grid_sources'])
def nms(request):
    return request.getfuncargvalue(request.param)


def test_nikamap_trim(single_source_mask):

    nm = single_source_mask
    nm_trimed = nm.trim()
    assert nm_trimed.shape == (21, 21)

    assert np.any(nm_trimed.mask[0, :])
    assert np.any(nm_trimed.mask[-1, :])
    assert np.any(nm_trimed.mask[:, 0])
    assert np.any(nm_trimed.mask[:, -1])


def test_nikamap_add_gaussian_sources(nms):

    nm = nms
    shape = nm.shape
    pixsize = np.abs(nm.wcs.wcs.cdelt[0])

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(1, nm.y[0], nm.x[0], stddev, stddev)
    for item_x, item_y in zip(nm.y[1:], nm.x[1:]):
        g += models.Gaussian2D(1, item_x, item_y, stddev, stddev)
    npt.assert_allclose(nm.data, g(xx, yy))

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources['ra'], nm.fake_sources['dec'], 0)
    npt.assert_allclose([x, y], [nm.x, nm.y])


def test_nikamap_detect_sources(nms):

    nm = nms
    nm.detect_sources()

    ordering = nm.fake_sources['find_peak']

    npt.assert_allclose(nm.fake_sources['ra'], nm.sources['ra'][ordering])
    npt.assert_allclose(nm.fake_sources['dec'], nm.sources['dec'][ordering])
    npt.assert_allclose(nm.sources['SNR'], [4] * len(nm.sources))


def test_nikamap_phot_sources(nms):

    nm = nms
    nm.detect_sources()
    nm.phot_sources()

    # Relative and absolute tolerance are really bad here for the case where the sources are not centered on pixels... Otherwise it give perfect answer when there is no noise
    npt.assert_allclose(nm.sources['flux_peak'].to(u.Jy).value, [1] * len(nm.sources), atol=1e-2, rtol=1e-1)
    # Relative tolerance is rather low to pass the case of multiple sources...
    # TODO: Should not be that high !!! (See Issue #1)
    npt.assert_allclose(nm.sources['flux_psf'].to(u.Jy).value, [1] * len(nm.sources), rtol=1e-4)


def test_nikamap_match_filter(nms):

    nm = nms
    mf_nm = nm.match_filter(nm.beam)

    x_idx = np.floor(nm.x + 0.5).astype(int)
    y_idx = np.floor(nm.y + 0.5).astype(int)

    npt.assert_allclose(mf_nm.data[y_idx, x_idx], nm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    npt.assert_allclose((nm.beam.fwhm*np.sqrt(2)).to(u.arcsec), mf_nm.beam.fwhm.to(u.arcsec))


def test_nikamap_match_sources(nms):

    nm = nms
    nm.detect_sources()
    sources = nm.sources
    sources.meta['name'] = 'to_match'
    nm.match_sources([sources])

    assert np.all(nm.sources['ID'] == nm.sources['to_match'])
