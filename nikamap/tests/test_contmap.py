from __future__ import absolute_import, division, print_function

import pytest
import warnings
import numpy as np
import numpy.testing as npt

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from astropy.nddata import StdDevUncertainty, InverseVariance
from astropy.modeling import models
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.convolution import RickerWavelet2DKernel

from photutils.datasets import make_gaussian_sources_image


import astropy.units as u
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.convolution import Kernel2D, Gaussian2DKernel

import matplotlib.pyplot as plt

from ..contmap import ContBeam, ContMap
from ..utils import pos_gridded


def test_contbeam_init():
    # TODO: What if we init with an array ?
    fwhm = 18 * u.arcsec
    array = np.ones((10, 10))
    pixscale = 2 * u.arcsec

    ref_kernel = Gaussian2DKernel(fwhm * gaussian_fwhm_to_sigma / pixscale, x_size=63, y_size=63)

    with pytest.raises(ValueError):
        beam = ContBeam()
        beam = ContBeam(fwhm)

    beam = ContBeam(fwhm, pixscale=pixscale)
    kernel = beam.as_kernel(pixscale)

    assert beam.major == fwhm
    assert (
        str(beam)
        == "ContBeam: BMAJ=18.0 arcsec BMIN=18.0 arcsec BPA=0.0 deg as (63, 63) Kernel2D at pixscale 2.0 arcsec"
    )
    assert isinstance(kernel, Kernel2D)
    assert np.all((ref_kernel.array - kernel.array) == 0)
    assert beam.sr == (2 * np.pi * (fwhm * gaussian_fwhm_to_sigma) ** 2).to(u.sr)

    beam = ContBeam(array=ref_kernel.array, pixscale=pixscale)
    assert beam.major is None
    assert str(beam) == "ContBeam: (63, 63) Kernel2D at pixscale 2.0 arcsec"
    npt.assert_almost_equal(beam.sr.value, (2 * np.pi * (fwhm * gaussian_fwhm_to_sigma) ** 2).to(u.sr).value)
    with pytest.raises(TypeError):
        beam.as_kernel()

    kernel = beam.as_kernel(pixscale)
    assert np.all(kernel.array == ref_kernel.array)


def test_contbeam_convolve():
    fwhm = 18 * u.arcsec
    pixscale = 2 * u.arcsec

    ref_kernel = Gaussian2DKernel(fwhm * gaussian_fwhm_to_sigma / pixscale, x_size=63, y_size=63)

    beam = ContBeam(fwhm, pixscale=pixscale)
    beam_convolve = beam.convolve(beam)
    npt.assert_almost_equal(beam_convolve.major.to(u.arcsec).value, (np.sqrt(2) * fwhm).to(u.arcsec).value)

    with pytest.warns(UserWarning):
        beam_refconvolve = beam.convolve(ref_kernel)

    center = (beam_refconvolve.shape[0] - 1) // 2
    size = (beam_convolve.shape[0] - 1) // 2
    _slice = slice(center - size, center + size + 1)
    npt.assert_almost_equal(beam_refconvolve.array[_slice, _slice], beam_convolve.array)


def test_contmap_init():
    data = [1, 2, 3]
    nm = ContMap(data)
    assert np.all(nm.data == np.array(data))

    # Should default to empty wcs and no unit
    assert nm.wcs is None
    assert nm.unit is u.adu
    assert nm.uncertainty is None

    # time "empty"
    assert nm.time is None

    # Default pixsize 1*u.deg
    assert (1 * u.pixel).to(u.deg, equivalencies=nm._pixel_scale) == 1 * u.deg

    # Default beam fwhm 1*u.deg
    assert nm.beam.major == 1 * u.deg


def test_contmap_init_quantity():
    data = np.array([1, 2, 3]) * u.Jy / u.beam
    nm = ContMap(data)
    assert nm.unit == u.Jy / u.beam


def test_contmap_init_meta():
    data = np.array([1, 2, 3])
    header = fits.header.Header()

    header["CDELT1"] = -1.0 / 3600, "pixel size used for pixel_scale"
    header["BMAJ"] = 1.0 / 3600, "Beam Major Axis"
    nm = ContMap(data, meta=header)
    assert (1 * u.pixel).to(u.deg, equivalencies=nm._pixel_scale) == 1 * u.arcsec
    assert nm.beam.major == 1 * u.arcsec
    assert nm.beam.minor == 1 * u.arcsec

    # Full header
    header["CRPIX1"] = 1
    header["CRPIX2"] = 2
    header["CDELT1"] = -1 / 3600
    header["CDELT2"] = 1 / 3600
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"

    nm = ContMap(data, meta=header, wcs=WCS(header))
    assert nm.wcs is not None


def test_contmap_init_uncertainty():
    data = np.array([1, 2, 3])
    uncertainty = np.array([1, 1, 1])

    # Default to StdDevUncertainty...
    with pytest.warns(UserWarning):
        nm = ContMap(data, uncertainty=uncertainty)
    assert isinstance(nm.uncertainty, StdDevUncertainty)
    assert np.all(nm.uncertainty.array == np.array([1, 1, 1]))

    nm_mean = nm.add(nm).divide(2)
    assert np.all(nm_mean.data == nm.data)
    npt.assert_allclose(nm_mean.uncertainty.array, np.array([1, 1, 1]) / np.sqrt(2))

    # Wrong size
    with pytest.raises(ValueError):
        nm = ContMap(data, uncertainty=uncertainty[1:])

    # Wrong TypeError
    with pytest.raises(TypeError):
        nm = ContMap(data, uncertainty=list(uncertainty))

    iv_uncertainty = InverseVariance(uncertainty)
    nm = ContMap(data, uncertainty=iv_uncertainty)

def test_contmap_compressed():
    data = np.array([1, 2, 3])
    uncertainty = np.array([10, 1, 1], dtype=float)
    mask = np.array([True, False, False])
    hits = np.ones(3)
    sampling_freq = 1 * u.Hz

    nm = ContMap(data, uncertainty=uncertainty, mask=mask, hits=hits, sampling_freq=sampling_freq, unit=u.Jy)

    assert np.all(nm.compressed() == np.array([2, 3]) * u.Jy)
    assert np.all(nm.uncertainty_compressed() == np.array([1, 1]) * u.Jy)

    assert np.all(nm.__array__() == np.ma.array(data, mask=mask))
    assert np.all(nm.__u_array__() == np.ma.array(uncertainty, mask=mask))

    # To insure compatilibity with Astropy 3.0, maskedQuantity cannot evaluate
    # truth value of quantities
    assert np.all(nm.__t_array__().data == hits / sampling_freq)
    assert np.all(nm.__t_array__().mask == mask)


# from pytest-django #393
def getfixturevalue(request, value):
    if hasattr(request, "getfixturevalue"):
        return request.getfixturevalue(value)

    return request.getfuncargvalue(value)


@pytest.fixture()
def no_source():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    np.random.seed(0)
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.random.normal(size=shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    nm = ContMap(data, uncertainty=np.ones(shape), wcs=wcs, unit=u.Jy / u.beam)

    return nm


def test_no_source(no_source):
    nm = no_source
    nm.detect_sources(threshold=5)
    assert nm.sources is None


@pytest.fixture()
def single_source():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    nm = ContMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam)

    # Additionnal attribute just for the tests...
    nm.x = np.asarray([shape[1] / 2 - 0.5])
    nm.y = np.asarray([shape[0] / 2 - 0.5])
    nm.add_gaussian_sources(nsources=1, peak_flux=1 * u.Jy, within=(1 / 2, 1 / 2))
    return nm


@pytest.fixture()
def single_source_side():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    fake_sources = Table(masked=True)
    fake_sources["fake_id"] = [1]
    fake_sources["x_mean"] = [0]
    fake_sources["y_mean"] = [13]

    ra, dec = wcs.wcs_pix2world(fake_sources["x_mean"], fake_sources["y_mean"], 0)
    fake_sources["ra"] = ra * u.deg
    fake_sources["dec"] = dec * u.deg

    fake_sources["_ra"] = fake_sources["ra"]
    fake_sources["_dec"] = fake_sources["dec"]

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(1, fake_sources["y_mean"], fake_sources["x_mean"], stddev, stddev)

    data += g(xx, yy)

    nm = ContMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam, fake_sources=fake_sources)

    nm.x = fake_sources["x_mean"]
    nm.y = fake_sources["y_mean"]

    return nm


@pytest.fixture()
def blended_sources():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    fake_sources = Table(masked=True)
    fake_sources["fake_id"] = [1, 2]
    fake_sources["x_mean"] = [13.6, 15.1]
    fake_sources["y_mean"] = [13.6, 15.1]

    ra, dec = wcs.wcs_pix2world(fake_sources["x_mean"], fake_sources["y_mean"], 0)
    fake_sources["ra"] = ra * u.deg
    fake_sources["dec"] = dec * u.deg

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(1, fake_sources["y_mean"][0], fake_sources["x_mean"][0], stddev, stddev)
    for source in fake_sources[1:]:
        g += models.Gaussian2D(1, source["y_mean"], source["x_mean"], stddev, stddev)

    data += g(xx, yy)

    nm = ContMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam, fake_sources=fake_sources)

    nm.x = fake_sources["x_mean"]
    nm.y = fake_sources["y_mean"]

    return nm


@pytest.fixture()
def single_source_mask():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    xx, yy = np.indices(shape)
    mask = np.sqrt((xx - (shape[1] - 1) / 2) ** 2 + (yy - (shape[0] - 1) / 2) ** 2) > 10

    data[mask] = np.nan

    nm = ContMap(data, uncertainty=np.ones_like(data) / 4, mask=mask, wcs=wcs, unit=u.Jy / u.beam)

    # Additionnal attribute just for the tests...
    nm.x = np.asarray([shape[1] / 2 - 0.5])
    nm.y = np.asarray([shape[0] / 2 - 0.5])
    nm.add_gaussian_sources(nsources=1, peak_flux=1 * u.Jy, within=(1 / 2, 1 / 2))
    return nm


@pytest.fixture()
def single_source_mask_edge():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    mask = np.zeros(shape, dtype=bool)
    mask[0 : (shape[0] - 1) // 2, :] = True  # noqa: E203

    data[mask] = np.nan

    fake_sources = Table(masked=True)
    fake_sources["fake_id"] = [1]
    fake_sources["x_mean"] = [13]
    fake_sources["y_mean"] = [13]

    ra, dec = wcs.wcs_pix2world(fake_sources["x_mean"], fake_sources["y_mean"], 0)
    fake_sources["ra"] = ra * u.deg
    fake_sources["dec"] = dec * u.deg

    fake_sources["_ra"] = fake_sources["ra"]
    fake_sources["_dec"] = fake_sources["dec"]

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(1, fake_sources["y_mean"], fake_sources["x_mean"], stddev, stddev)

    data += g(xx, yy)

    nm = ContMap(
        data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam, mask=mask, fake_sources=fake_sources
    )

    nm.x = fake_sources["x_mean"]
    nm.y = fake_sources["y_mean"]

    return nm


@pytest.fixture()
def grid_sources():
    # Larger shape to allow for wobbling
    # as beam needs to be much smaller than the map at some point..
    # Shape was too small to allow for a proper background estimation
    # shape = (28, 28)
    shape = (60, 60)
    pixsize = 1 / 3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    nm = ContMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam)

    # Additionnal attribute just for the tests...
    nm.add_gaussian_sources(nsources=2 ** 2, peak_flux=1 * u.Jy, cat_gen=pos_gridded, within=(1 / 4, 3 / 4))

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources["ra"], nm.fake_sources["dec"], 0)

    nm.x = x
    nm.y = y

    return nm


@pytest.fixture()
def wobble_grid_sources():
    # Even Larger shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (60, 60)
    pixsize = 1 / 3
    data = np.zeros(shape)
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    nm = ContMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam)

    np.random.seed(0)
    # Additionnal attribute just for the tests...
    nm.add_gaussian_sources(nsources=2 ** 2, peak_flux=1 * u.Jy, cat_gen=pos_gridded, wobble=True, wobble_frac=0.2)

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources["ra"], nm.fake_sources["dec"], 0)

    nm.x = x
    nm.y = y

    return nm


@pytest.fixture()
def large_map_source():

    np.random.seed(0)

    shape = (256, 256)
    pixsize = 1 / 3 * u.deg
    peak_flux = 1 * u.Jy
    noise_level = 0.1 * u.Jy / u.beam
    fwhm = 1 * u.deg
    nsources = 1

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    xx, yy = np.indices(shape)
    mask = np.sqrt((xx - (shape[1] - 1) / 2) ** 2 + (yy - (shape[0] - 1) / 2) ** 2) > shape[0] / 2

    sources = Table(masked=True)
    sources["amplitude"] = np.ones(nsources) * peak_flux
    sources["x_mean"] = [shape[1] / 2]
    sources["y_mean"] = [shape[0] / 2]

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    sources["x_stddev"] = np.ones(nsources) * beam_std_pix
    sources["y_stddev"] = np.ones(nsources) * beam_std_pix
    sources["theta"] = np.zeros(nsources)

    data = make_gaussian_sources_image(shape, sources)

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level.to(u.Jy / u.beam).value
    data += np.random.normal(loc=0, scale=1, size=shape) * uncertainty
    data[mask] = np.nan
    hits[mask] = 0
    uncertainty[mask] = 0

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"

    nm = ContMap(data, uncertainty=uncertainty, wcs=wcs, unit=u.Jy / u.beam)

    return nm


# Special case to avoid detection, which would fail here (ShrinkMask)
def test_contmap_phot_mask_edge(single_source_mask_edge):
    nm = single_source_mask_edge
    nm.sources = nm.fake_sources
    nm.phot_sources(peak=True, psf=False)
    # Relative and absolute tolerance are really bad here for the case where
    # the sources are not centered on pixels... Otherwise it give perfect
    # answer when there is no noise
    npt.assert_allclose(nm.sources["flux_peak"].to(u.Jy).value, [1] * len(nm.sources), atol=1e-1, rtol=1e-1)

    nm.phot_sources(peak=False, psf=True)
    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(nm.sources["flux_psf"].to(u.Jy).value, [1] * len(nm.sources), rtol=1e-6)


@pytest.fixture(
    params=["single_source", "single_source_side", "single_source_mask", "grid_sources", "wobble_grid_sources"]
)
def nms(request):
    return getfixturevalue(request, request.param)


def test_contmap_trim(single_source_mask):

    nm = single_source_mask
    nm_trimed = nm.trim()
    assert nm_trimed.shape == (21, 21)

    assert np.any(nm_trimed.mask[0, :])
    assert np.any(nm_trimed.mask[-1, :])
    assert np.any(nm_trimed.mask[:, 0])
    assert np.any(nm_trimed.mask[:, -1])


def test_contmap_add_gaussian_sources(nms):

    nm = nms
    shape = nm.shape
    pixsize = np.abs(nm.wcs.wcs.cdelt[0])

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(1, nm.y[0], nm.x[0], stddev, stddev)
    for item_x, item_y in zip(nm.y[1:], nm.x[1:]):
        g += models.Gaussian2D(1, item_x, item_y, stddev, stddev)

    if nm.mask is None:
        npt.assert_allclose(nm.data, g(xx, yy))
    else:
        npt.assert_allclose(nm.data[~nm.mask], g(xx, yy)[~nm.mask])

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources["ra"], nm.fake_sources["dec"], 0)
    # We are actually only testing the tolerance on x,y -> ra, dec -> x, y
    npt.assert_allclose([x, y], [nm.x, nm.y], atol=1e-13)


def test_contmap_detect_sources(nms):

    nm = nms
    nm.detect_sources()

    ordering = nm.fake_sources["find_peak"]

    npt.assert_allclose(nm.fake_sources["ra"], nm.sources["ra"][ordering])
    npt.assert_allclose(nm.fake_sources["dec"], nm.sources["dec"][ordering])

    # When sources are exactly at the center of 4 pixels the basic peak finder will fail
    if len(nms.fake_sources) != 4:
        npt.assert_allclose(nm.sources["SNR"], [4] * len(nm.sources))
    else:
        npt.assert_allclose(nm.sources["SNR"], [4] * len(nm.sources), atol=0.6)

    x_fake, y_fake = nm.wcs.wcs_world2pix(nm.fake_sources["ra"], nm.fake_sources["dec"], 0)
    x, y = nm.wcs.wcs_world2pix(nm.sources["ra"], nm.sources["dec"], 0)
    nm = nms
    nm.detect_sources()
    nm.phot_sources(peak=True, psf=False)
    # Relative and absolute tolerance are really bad here for the case where
    # the sources are not centered on pixels... Otherwise it give perfect
    # answer when there is no noise
    npt.assert_allclose(nm.sources["flux_peak"].to(u.Jy).value, [1] * len(nm.sources), atol=1e-1, rtol=1e-1)

    nm.phot_sources(peak=False, psf=True)
    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(nm.sources["flux_psf"].to(u.Jy).value, [1] * len(nm.sources), rtol=1e-6)

    # Tolerance coming from round wcs transformations
    npt.assert_allclose(x_fake, x[ordering], atol=1e-11)
    npt.assert_allclose(y_fake, y[ordering], atol=1e-11)

    # Fake empy data to fake no found sources
    nm._data *= 0
    nm.detect_sources()
    assert nm.sources is None
    assert np.all(nm.fake_sources["find_peak"].mask)


def test_contmap_phot_sources(nms):

    nm = nms
    nm.detect_sources()
    nm.phot_sources(peak=True, psf=False)
    # Relative and absolute tolerance are really bad here for the case where
    # the sources are not centered on pixels... Otherwise it give perfect
    # answer when there is no noise
    npt.assert_allclose(nm.sources["flux_peak"].to(u.Jy).value, [1] * len(nm.sources), atol=1e-1, rtol=1e-1)

    nm.phot_sources(peak=False, psf=True)
    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(nm.sources["flux_psf"].to(u.Jy).value, [1] * len(nm.sources), rtol=1e-6)


def test_contmap_match_filter(nms):

    nm = nms
    mf_nm = nm.match_filter(nm.beam)

    x_idx = np.floor(nm.x + 0.5).astype(int)
    y_idx = np.floor(nm.y + 0.5).astype(int)

    npt.assert_allclose(mf_nm.data[y_idx, x_idx], nm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    npt.assert_allclose((nm.beam.major * np.sqrt(2)).to(u.arcsec), mf_nm.beam.major.to(u.arcsec))

    mh_nm = nm.match_filter(
        RickerWavelet2DKernel(nm.beam.major.to(u.pix, nm._pixel_scale).value * gaussian_fwhm_to_sigma)
    )
    npt.assert_allclose(mh_nm.data[y_idx, x_idx], nm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    assert mh_nm.beam.major is None


def test_contmap_match_sources(nms):

    nm = nms
    nm.detect_sources()
    sources = nm.sources
    sources.meta["name"] = "to_match"
    nm.match_sources(sources)

    assert np.all(nm.sources["ID"] == nm.sources["to_match"])


def test_contmap_match_sources_threshold(nms):

    nm = nms
    nm.detect_sources()
    sources = nm.sources
    sources.meta["name"] = "to_match"
    nm.match_sources(sources, dist_threshold=nm.beam.major)

    assert np.all(nm.sources["ID"] == nm.sources["to_match"])


def test_contmap_match_sources_list(nms):

    nm = nms
    nm.detect_sources()
    sources = nm.sources.copy()
    sources.meta["name"] = "to_match_1"
    sources2 = nm.sources.copy()
    sources2.meta["name"] = "to_match_2"

    nm.match_sources([sources, sources2])

    assert np.all(nm.sources["ID"] == nm.sources["to_match_1"])
    assert np.all(nm.sources["ID"] == nm.sources["to_match_2"])


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_contmap_plot(nms):

    nm = nms
    cax = nm.plot()

    return cax.get_figure()


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_contmap_plot_SNR(nms):

    nm = nms
    cax = nm.plot_SNR(cbar=True)

    return cax.get_figure()


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_contmap_plot_ax(nms):

    nm = nms
    fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": nm.wcs})
    axes = axes.flatten()
    nm.plot(ax=axes[0], vmin=-1, vmax=3)
    nm.plot(ax=axes[1], levels=np.logspace(np.log10(0.1), np.log10(5), 5))
    nm.plot(ax=axes[2], cat=[(nm.fake_sources, {"marker": "+", "color": "red"})])
    nm.fake_sources = None
    nm.detect_sources()
    nm.plot(ax=axes[3], cat=True)

    for ax in axes:
        ax.legend(loc="best", frameon=False)

    return fig


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_contmap_plot_PSD(nms):

    nm = nms
    fig, axes = plt.subplots(nrows=4, sharex=True)
    nm.plot_PSD(ax=axes[0])
    nm.plot_PSD(ax=axes[1], apod_size=5)
    nm.plot_PSD(ax=axes[2], bins=50)
    nm.plot_PSD(ax=axes[3], to_plot="snr")

    powspec, bins = nm.plot_PSD()

    return fig


def test_contmap_check_SNR(large_map_source):

    nm = large_map_source

    std = nm.check_SNR()
    # Tolerance comes from the fact that we biased the result using the SNR
    # cut for the fit
    npt.assert_allclose(std, 1, rtol=1e-2)


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_contmap_check_SNR_ax(large_map_source):

    nm = large_map_source

    fig, ax = plt.subplots()
    std = nm.check_SNR(ax=ax)

    return fig


def test_blended_sources(blended_sources):

    nm = blended_sources
    nm.detect_sources()
    nm.phot_sources()

    # Cannot recover all sources :
    assert len(nm.sources) != len(nm.fake_sources)

    # But still prior photometry can recover the flux
    nm.phot_sources(nm.fake_sources)
    npt.assert_allclose(nm.fake_sources["flux_psf"].to(u.Jy).value, [1] * len(nm.fake_sources))


def test_get_square_slice(single_source_mask):

    nm = single_source_mask
    islice = nm.get_square_slice()

    radius = 10
    assert np.floor(np.sqrt(2) * radius) == islice.stop - islice.start - 1
    assert np.floor(nm.shape[0] / 2 - np.sqrt(2) * radius / 2) == islice.start
    assert np.floor(nm.shape[0] / 2 + np.sqrt(2) * radius / 2 + 1) == islice.stop


def test_get_square_slice_start(single_source_mask):

    nm = single_source_mask

    with pytest.raises(AssertionError):
        islice = nm.get_square_slice(start=14)
        islice = nm.get_square_slice(start=[14, 14, 14])

    islice = nm.get_square_slice(start=(14, 14))

    radius = 10
    assert np.floor(np.sqrt(2) * radius) == islice.stop - islice.start - 1
    assert np.floor(nm.shape[0] / 2 - np.sqrt(2) * radius / 2) == islice.start
    assert np.floor(nm.shape[0] / 2 + np.sqrt(2) * radius / 2 + 1) == islice.stop


def test_surface():
    shape = (2, 2)
    data = np.ones(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    wcs = WCS()
    wcs.wcs.cdelt = np.array([-2 / 60 ** 2, 2 / 60 ** 2])
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]

    nm = ContMap(data=data, mask=mask, wcs=wcs)
    surface = nm.surface()
    assert np.isclose(surface.to_value(u.arcsec ** 2), 8)


def test_surface_shrink():
    shape = (5, 5)
    data = np.ones(shape)
    mask = np.ones(shape, dtype=bool)
    mask[1:-1, 1:-1] = False

    wcs = WCS()
    wcs.wcs.cdelt = np.array([-2 / 60 ** 2, 2 / 60 ** 2])
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]

    nm = ContMap(data=data, mask=mask, wcs=wcs)
    surface = nm.surface(box_size=1.001)
    assert np.isclose(surface.to_value(u.arcsec ** 2), 4)


@pytest.fixture(scope="session")
def generate_fits(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("cm_map")
    filename = str(tmpdir.join("map.fits"))
    # Larger map to perform check_SNR

    np.random.seed(0)

    shape = (256, 256)
    pixsize = 1 / 3 * u.deg
    peak_flux = 1 * u.Jy
    noise_level = 0.1 * u.Jy / u.beam
    fwhm = 1 * u.deg
    nsources = 1

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    xx, yy = np.indices(shape)
    mask = np.sqrt((xx - (shape[1] - 1) / 2) ** 2 + (yy - (shape[0] - 1) / 2) ** 2) > shape[0] / 2

    sources = Table(masked=True)
    sources["amplitude"] = np.ones(nsources) * peak_flux
    sources["x_mean"] = [shape[1] / 2]
    sources["y_mean"] = [shape[0] / 2]

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    sources["x_stddev"] = np.ones(nsources) * beam_std_pix
    sources["y_stddev"] = np.ones(nsources) * beam_std_pix
    sources["theta"] = np.zeros(nsources)

    data = make_gaussian_sources_image(shape, sources)

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level.to(u.Jy / u.beam).value
    data += np.random.normal(loc=0, scale=1, size=shape) * uncertainty
    data[mask] = np.nan
    hits[mask] = 0
    uncertainty[mask] = 0

    header = wcs.to_header()
    header["BUNIT"] = "Jy / beam", "Fake Unit"

    primary_header = fits.Header()
    primary_header["HISTORY"] = "this"
    primary_header["HISTORY"] = "and that"
    primary_header["COMMENT"] = "or that"
    primary_header["COMMENT"] = "and this"
    primary_header["BMAJ"] = fwhm.to(u.deg).value
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=fits.verify.VerifyWarning)
        primary_header["sampling_freq"] = 10

    hdus = [fits.hdu.PrimaryHDU(None, primary_header)]
    hdus.append(fits.hdu.ImageHDU(data, header=header, name="DATA"))
    hdus.append(fits.hdu.ImageHDU(uncertainty, header=header, name="UNCERT"))
    hdus.append(fits.hdu.ImageHDU(hits, header=header, name="HITS"))
    # hdus.append(fits.hdu.ImageHDU(mask, header=header, name="MASK"))

    hdus = fits.hdu.HDUList(hdus)

    hdus.writeto(filename, overwrite=True)

    return filename


def test_contmap_read(generate_fits):

    filename = generate_fits

    data = ContMap.read(filename)
    assert data.sampling_freq == 10 * u.Hz
    assert data.shape == (256, 256)
    assert str(data.unit) == "Jy / beam"
    assert data.beam.major.to(u.arcsec).value == 3600
    assert list(data.meta["HISTORY"]) == ["this", "and that"]
    assert list(data.meta["COMMENT"]) == ["or that", "and this"]
    assert data.hits is not None


def test_nikamap_write(generate_fits):
    filename = generate_fits

    data = ContMap.read(filename)

    outfilename = filename.replace("map.fits", "map2.fits")
    data.write(outfilename)
