from __future__ import absolute_import, division, print_function

import pytest

import os.path as op

import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, Column

from astropy.nddata import StdDevUncertainty
from astropy.modeling import models
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from photutils.datasets import make_gaussian_sources_image
from astropy.convolution import MexicanHat2DKernel

import numpy.testing as npt

# from nikamap import nikamap as nm
# from nikamap.nikamap import NikaMap, jk_nikamap

import nikamap as nm
from ..nikamap import NikaMap, jk_nikamap


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

    # Wrong size
    with pytest.raises(ValueError):
        nm = NikaMap(data, uncertainty=uncertainty[1:])

    # Wrong TypeError
    with pytest.raises(TypeError):
        nm = NikaMap(data, uncertainty=list(uncertainty))


def test_nikamap_compressed():
    data = np.array([1, 2, 3])
    uncertainty = np.array([10, 1, 1])
    mask = np.array([True, False, False])
    time = np.ones(3)*u.h

    nm = NikaMap(data, uncertainty=uncertainty, mask=mask, time=time, unit=u.Jy)

    assert np.all(nm.compressed() == np.array([2, 3]) * u.Jy)
    assert np.all(nm.uncertainty_compressed() == np.array([1, 1]) * u.Jy)

    assert np.all(nm.__array__() == np.ma.array(data * u.Jy, mask=mask))
    assert np.all(nm.__u_array__() == np.ma.array(uncertainty * u.Jy, mask=mask))
    assert np.all(nm.__t_array__() == np.ma.array(time, mask=mask))


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
def single_source_side():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (26, 26)
    pixsize = 1/3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape)/2-0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1])*pixsize
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    fake_sources = Table(masked=True)
    fake_sources['ID'] = [1]
    fake_sources['x_mean'] = [0]
    fake_sources['y_mean'] = [13]

    ra, dec = wcs.wcs_pix2world(fake_sources['x_mean'], fake_sources['y_mean'], 0)
    fake_sources['ra'] = ra * u.deg
    fake_sources['dec'] = dec * u.deg

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(1, fake_sources['y_mean'], fake_sources['x_mean'], stddev, stddev)

    data += g(xx, yy)

    nm = NikaMap(data, uncertainty=np.ones_like(data)/4, wcs=wcs, unit=u.Jy/u.beam, fake_sources=fake_sources)

    nm.x = fake_sources['x_mean']
    nm.y = fake_sources['y_mean']

    return nm


@pytest.fixture()
def blended_sources():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (26, 26)
    pixsize = 1/3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape)/2-0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1])*pixsize
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    fake_sources = Table(masked=True)
    fake_sources['ID'] = [1, 2]
    fake_sources['x_mean'] = [13.6, 15.1]
    fake_sources['y_mean'] = [13.6, 15.1]

    ra, dec = wcs.wcs_pix2world(fake_sources['x_mean'], fake_sources['y_mean'], 0)
    fake_sources['ra'] = ra * u.deg
    fake_sources['dec'] = dec * u.deg

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(1, fake_sources['y_mean'][0], fake_sources['x_mean'][0], stddev, stddev)
    for source in fake_sources[1:]:
        g += models.Gaussian2D(1, source['y_mean'], source['x_mean'], stddev, stddev)

    data += g(xx, yy)

    nm = NikaMap(data, uncertainty=np.ones_like(data)/4, wcs=wcs, unit=u.Jy/u.beam, fake_sources=fake_sources)

    nm.x = fake_sources['x_mean']
    nm.y = fake_sources['y_mean']

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
    # Shape was too small to allow for a proper background estimation
    # shape = (28, 28)
    shape = (60, 60)
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

    np.random.seed(1)
    # Additionnal attribute just for the tests...
    nm.add_gaussian_sources(nsources=2**2, peak_flux=1*u.Jy,
                            grid=True, wobble=True,
                            within=(1/3, 2/3))

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources['ra'], nm.fake_sources['dec'], 0)

    nm.x = x
    nm.y = y

    return nm


@pytest.fixture(scope='session')
def generate_nikamaps(tmpdir_factory):
    # Generate several maps with sources and noise... only one band...

    tmpdir = tmpdir_factory.mktemp("nm_maps")

    band = "1mm"

    shape = (60, 60)
    pixsize = 1/3 * u.deg
    noise_level = 1 * u.Jy / u.beam
    nmaps = 10
    nsources = 5
    fwhm = 1*u.deg

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape)/2-0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1])*pixsize
    wcs.wcs.ctype = ('RA---TAN', 'DEC--TAN')

    # Fake sources for all maps
    np.random.seed(3)
    sources = Table(masked=True)
    sources['amplitude'] = np.random.uniform(1, 10, size=nsources) * u.Jy
    sources['x_mean'] = np.random.uniform(1/4, 3/4, size=nsources) * shape[1]
    sources['y_mean'] = np.random.uniform(1/4, 3/4, size=nsources) * shape[0]

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    sources['x_stddev'] = np.ones(nsources) * beam_std_pix
    sources['y_stddev'] = np.ones(nsources) * beam_std_pix
    sources['theta'] = np.zeros(nsources)

    data_sources = make_gaussian_sources_image(shape, sources) * u.Jy / u.beam

    a, d = wcs.wcs_pix2world(sources['x_mean'], sources['y_mean'], 0)
    sources.add_columns([Column(a * u.deg, name='ra'),
                         Column(d * u.deg, name='dec')])
    sources.remove_columns(['x_mean', 'y_mean', 'x_stddev', 'y_stddev', 'theta'])
    sources.sort('amplitude')
    sources.reverse()
    sources.add_column(Column(np.arange(len(sources)), name='ID'), 0)

    xx, yy = np.indices(shape)
    mask = np.sqrt((xx-shape[1]/2)**2 + (yy-shape[0]/2)**2) >= shape[0]/2

    primary_header = fits.header.Header()
    primary_header['f_sampli'] = 10., 'Fake the f_sampli keyword'
    primary_header['FWHM_260'] = fwhm.to(u.arcsec).value, '[arcsec] Fake the FWHM_260 keyword'

    primary_header['nsources'] = nsources, 'Number of fake sources'
    primary_header['pixsize'] = pixsize.to(u.deg).value, '[deg] pixel size'
    primary_header['nmaps'] = nmaps, 'number of maps produced'
    primary_header['shape0'] = shape[0], '[0] of map shape'
    primary_header['shape1'] = shape[1], '[1] of map shape'
    primary_header['noise'] = noise_level.to(u.Jy/u.beam).value, '[Jy/beam] noise level per map'

    primary = fits.hdu.PrimaryHDU(header=primary_header)

    filenames = []

    for i_map in range(nmaps):

        filename = str(tmpdir.join('map_{}.fits'.format(i_map)))

        hits = np.ones(shape=shape, dtype=np.float)
        uncertainty = np.ones(shape=shape, dtype=np.float) * noise_level
        data = np.random.normal(loc=0, scale=1, size=shape) * uncertainty

        data += data_sources
        data[mask] = 0
        hits[mask] = 0
        uncertainty[mask] = 0

        header = wcs.to_header()
        header['UNIT'] = "Jy / beam", 'Fake Unit'

        hdus = fits.hdu.HDUList(hdus=[primary])
        hdus.append(fits.hdu.ImageHDU(data.value, header=header, name='Brightness_{}'.format(band)))
        hdus.append(fits.hdu.ImageHDU(uncertainty.value, header=header, name='Stddev_{}'.format(band)))
        hdus.append(fits.hdu.ImageHDU(hits, header=header, name='Nhits_{}'.format(band)))
        hdus.append(fits.hdu.BinTableHDU(sources, name="fake_sources"))
        hdus.writeto(filename, overwrite=True)

        filenames.append(filename)

    return filenames


@pytest.fixture(params=['single_source', 'single_source_side', 'single_source_mask',
                        'grid_sources', 'wobble_grid_sources'])
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
    # We are actually only testing the tolerance on x,y -> ra, dec -> x, y
    npt.assert_allclose([x, y], [nm.x, nm.y], atol=1e-13)


def test_nikamap_detect_sources(nms):

    nm = nms
    nm.detect_sources()

    ordering = nm.fake_sources['find_peak']

    npt.assert_allclose(nm.fake_sources['ra'], nm.sources['ra'][ordering])
    npt.assert_allclose(nm.fake_sources['dec'], nm.sources['dec'][ordering])
    npt.assert_allclose(nm.sources['SNR'], [4] * len(nm.sources))

    x_fake, y_fake = nm.wcs.wcs_world2pix(nm.fake_sources['ra'], nm.fake_sources['dec'], 0)
    x, y = nm.wcs.wcs_world2pix(nm.sources['ra'], nm.sources['dec'], 0)

    # Tolerance coming from round wcs transformations
    npt.assert_allclose(x_fake, x[ordering], atol=1e-11)
    npt.assert_allclose(y_fake, y[ordering], atol=1e-11)

    # Fake empy data to fake no found sources
    nm._data *= 0
    nm.detect_sources()
    assert nm.sources is None
    assert np.all(nm.fake_sources['find_peak'].mask)


def test_nikamap_phot_sources(nms):

    nm = nms
    nm.detect_sources()
    nm.phot_sources()

    # Relative and absolute tolerance are really bad here for the case where the sources are not centered on pixels... Otherwise it give perfect answer when there is no noise
    npt.assert_allclose(nm.sources['flux_peak'].to(u.Jy).value, [1] * len(nm.sources), atol=1e-2, rtol=1e-1)
    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(nm.sources['flux_psf'].to(u.Jy).value, [1] * len(nm.sources))


def test_nikamap_match_filter(nms):

    nm = nms
    mf_nm = nm.match_filter(nm.beam)

    x_idx = np.floor(nm.x + 0.5).astype(int)
    y_idx = np.floor(nm.y + 0.5).astype(int)

    npt.assert_allclose(mf_nm.data[y_idx, x_idx], nm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    npt.assert_allclose((nm.beam.fwhm*np.sqrt(2)).to(u.arcsec), mf_nm.beam.fwhm.to(u.arcsec))

    mh_nm = nm.match_filter(MexicanHat2DKernel(nm.beam.fwhm_pix.value * gaussian_fwhm_to_sigma))
    npt.assert_allclose(mh_nm.data[y_idx, x_idx], nm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    assert mh_nm.beam.fwhm is None


def test_nikamap_match_sources(nms):

    nm = nms
    nm.detect_sources()
    sources = nm.sources
    sources.meta['name'] = 'to_match'
    nm.match_sources(sources)

    assert np.all(nm.sources['ID'] == nm.sources['to_match'])


def test_nikamap_read(generate_nikamaps):

    filenames = generate_nikamaps
    primary_header = fits.getheader(filenames[0], 0)
    data = NikaMap.read(filenames[0])

    assert np.all(data.uncertainty.array[~data.mask] == primary_header['NOISE'])
    assert data.shape == (primary_header['SHAPE0'], primary_header['SHAPE1'])
    assert data.beam.fwhm.to(u.arcsec).value == primary_header['FWHM_260']
    assert np.all(data.time[~data.mask].value == ((primary_header['F_SAMPLI']*u.Hz)**-1).to(u.h).value)

    data_revert = NikaMap.read(filenames[0], revert=True)
    assert np.all(data_revert._data[~data_revert.mask] == -1 * data._data[~data.mask])


def test_jk_nikmap(generate_nikamaps):

    filenames = generate_nikamaps
    primary_header = fits.getheader(filenames[0], 0)

    weighed_noise = primary_header['NOISE'] / np.sqrt(primary_header['NMAPS'])

    # Weighted average
    data = next(jk_nikamap(filenames, n=None))
    assert np.all(data.uncertainty.array[~data.mask] == weighed_noise)

    # Produce one jackknife
    data = next(jk_nikamap(filenames, n=1))
    assert np.all(data.uncertainty.array[~data.mask] == weighed_noise)
    npt.assert_allclose(np.std(data.data[~data.mask]), weighed_noise, rtol=1e-2)

    iterator = jk_nikamap(filenames, n=10)
    assert len(list(iterator)) == 10

    with pytest.raises(StopIteration):
        next(iterator)

    with pytest.warns(UserWarning):
        iterator = jk_nikamap(filenames[1:], n=1)


def test_blended_sources(blended_sources):

    nm = blended_sources
    nm.detect_sources()
    nm.phot_sources()

    # Cannot recover all sources :
    assert len(nm.sources) != len(nm.fake_sources)

    # But still prior photometry can recover the flux
    nm.phot_sources(nm.fake_sources)
    npt.assert_allclose(nm.fake_sources['flux_psf'].to(u.Jy).value, [1] * len(nm.fake_sources))
