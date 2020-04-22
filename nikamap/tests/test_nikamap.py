from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table

from astropy.nddata import StdDevUncertainty
from astropy.modeling import models
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.convolution import MexicanHat2DKernel

from photutils.datasets import make_gaussian_sources_image

import numpy.testing as npt

import matplotlib.pyplot as plt

# from nikamap.nikamapNDDataArray, import NikaMap, jk_nikamap

# import nikamap as nm
# data_path = op.join(nm.__path__[0], 'data')

from ..nikamap import NikaMap, NikaBeam, NikaFits, retrieve_primary_keys
from ..utils import pos_gridded


# from pytest-django #393
def getfixturevalue(request, value):
    if hasattr(request, "getfixturevalue"):
        return request.getfixturevalue(value)

    return request.getfuncargvalue(value)


def test_nikabeam_exceptions():
    # TODO: Should probably be assertions at the __init__ stage...

    fwhm = 18 * u.arcsec

    with pytest.raises(AttributeError):
        beam = NikaBeam()

    with pytest.raises(AttributeError):
        beam = NikaBeam(fwhm.value)

    with pytest.raises(TypeError):
        beam = NikaBeam(fwhm, fwhm)


def test_nikabeam_init():
    # TODO: What if we init with an array ?
    fwhm = 18 * u.arcsec
    pix_scale = u.equivalencies.pixel_scale(2 * u.arcsec / u.pixel)

    beam = NikaBeam(fwhm, pix_scale)

    assert beam.fwhm == fwhm
    assert beam.fwhm_pix == fwhm.to(u.pixel, equivalencies=pix_scale)

    assert beam.sigma == fwhm * gaussian_fwhm_to_sigma
    assert beam.sigma_pix == fwhm.to(u.pixel, equivalencies=pix_scale) * gaussian_fwhm_to_sigma

    assert beam.area == 2 * np.pi * (fwhm * gaussian_fwhm_to_sigma) ** 2
    assert beam.area_pix == 2 * np.pi * (fwhm.to(u.pixel, equivalencies=pix_scale) * gaussian_fwhm_to_sigma) ** 2

    beam.normalize("peak")
    npt.assert_allclose(beam.area_pix.value, np.sum(beam.array), rtol=1e-4)

    assert str(beam) == "<NikaBeam(fwhm=18.0 arcsec, pixel_scale=2.00 arcsec / pixel)"


def test_nikamap_init():
    data = [1, 2, 3]
    nm = NikaMap(data)
    assert np.all(nm.data == np.array(data))

    # Should default to empty wcs and no unit
    assert nm.wcs is None
    assert nm.unit is None
    assert nm.uncertainty is None

    # time "empty"
    assert np.all(nm.time == 0 * u.s)

    # Default pixsize 1*u.deg
    assert (1 * u.pixel).to(u.deg, equivalencies=nm._pixel_scale) == 1 * u.deg

    # Default beam fwhm 1*u.deg
    assert nm.beam.fwhm == 1 * u.deg


def test_nikamap_init_quantity():
    data = np.array([1, 2, 3]) * u.Jy / u.beam
    nm = NikaMap(data)
    assert nm.unit == u.Jy / u.beam


def test_nikamap_init_time():
    data = np.array([1, 2, 3]) * u.Jy / u.beam

    time = np.array([1, 2]) * u.s
    with pytest.raises(ValueError):
        nm = NikaMap(data, time=time)

    time = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        nm = NikaMap(data, time=time)

    time = np.array([1, 2, 3]) * u.Hz
    with pytest.raises(ValueError):
        nm = NikaMap(data, time=time)

    time = np.array([1, 2, 3]) * u.h
    nm = NikaMap(data, time=time)
    assert nm.time.unit == u.h


def test_nikamap_init_meta():
    data = np.array([1, 2, 3])
    header = fits.header.Header()

    header["CDELT1"] = -1.0 / 3600, "pixel size used for pixel_scale"
    header["BMAJ"] = 1.0 / 3600, "Beam Major Axis"
    nm = NikaMap(data, meta={"header": header})
    assert (1 * u.pixel).to(u.deg, equivalencies=nm._pixel_scale) == 1 * u.arcsec
    assert nm.beam.fwhm == 1 * u.arcsec

    # Full header
    header["CRPIX1"] = 1
    header["CRPIX2"] = 2
    header["CDELT1"] = -1 / 3600
    header["CDELT2"] = 1 / 3600
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"

    nm = NikaMap(data, meta={"header": header}, wcs=WCS(header))
    assert nm.wcs is not None


def test_nikamap_init_uncertainty():
    data = np.array([1, 2, 3])
    uncertainty = np.array([1, 1, 1])

    # Default to StdDevUncertainty...
    nm = NikaMap(data, uncertainty=uncertainty)
    assert isinstance(nm.uncertainty, StdDevUncertainty)
    assert np.all(nm.uncertainty.array == np.array([1, 1, 1]))
    assert nm.unit == nm.uncertainty.unit

    nm_mean = nm.add(nm).divide(2)
    assert np.all(nm_mean.data == nm.data)
    npt.assert_allclose(nm_mean.uncertainty.array, np.array([1, 1, 1]) / np.sqrt(2))

    # Wrong size
    with pytest.raises(ValueError):
        nm = NikaMap(data, uncertainty=uncertainty[1:])

    # Wrong TypeError
    with pytest.raises(TypeError):
        nm = NikaMap(data, uncertainty=list(uncertainty))

    # Different Units
    st_uncertainty = StdDevUncertainty(uncertainty * 1e-3, unit=u.Jy)
    nm = NikaMap(data * u.mJy, uncertainty=st_uncertainty)
    assert nm.uncertainty.unit == nm.unit
    npt.assert_equal(nm.uncertainty.array, uncertainty)


def test_nikamap_compressed():
    data = np.array([1, 2, 3])
    uncertainty = np.array([10, 1, 1], dtype=float)
    mask = np.array([True, False, False])
    time = np.ones(3) * u.h

    nm = NikaMap(data, uncertainty=uncertainty, mask=mask, time=time, unit=u.Jy)

    assert np.all(nm.compressed() == np.array([2, 3]) * u.Jy)
    assert np.all(nm.uncertainty_compressed() == np.array([1, 1]) * u.Jy)

    assert np.all(nm.__array__() == np.ma.array(data, mask=mask))
    assert np.all(nm.__u_array__() == np.ma.array(uncertainty, mask=mask))

    # To insure compatilibity with Astropy 3.0, maskedQuantity cannot evaluate
    # truth value of quantities
    assert np.all(nm.__t_array__().data == time)
    assert np.all(nm.__t_array__().mask == mask)


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

    nm = NikaMap(data, uncertainty=np.ones(shape), wcs=wcs, unit=u.Jy / u.beam)

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

    nm = NikaMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam)

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

    nm = NikaMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam, fake_sources=fake_sources)

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

    nm = NikaMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam, fake_sources=fake_sources)

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

    nm = NikaMap(data, uncertainty=np.ones_like(data) / 4, mask=mask, wcs=wcs, unit=u.Jy / u.beam)

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

    nm = NikaMap(
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

    nm = NikaMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam)

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

    nm = NikaMap(data, uncertainty=np.ones_like(data) / 4, wcs=wcs, unit=u.Jy / u.beam)

    np.random.seed(0)
    # Additionnal attribute just for the tests...
    nm.add_gaussian_sources(nsources=2 ** 2, peak_flux=1 * u.Jy, cat_gen=pos_gridded, wobble=True, wobble_frac=0.2)

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources["ra"], nm.fake_sources["dec"], 0)

    nm.x = x
    nm.y = y

    return nm


@pytest.fixture(scope="session")
def generate_fits(tmpdir_factory):

    tmpdir = tmpdir_factory.mktemp("nm_map")
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

    hits = np.ones(shape=shape, dtype=np.float)
    uncertainty = np.ones(shape, dtype=np.float) * noise_level.to(u.Jy / u.beam).value
    data += np.random.normal(loc=0, scale=1, size=shape) * uncertainty
    data[mask] = np.nan
    hits[mask] = 0
    uncertainty[mask] = 0

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"

    primary_header = fits.header.Header()
    primary_header["f_sampli"] = 10.0, "Fake the f_sampli keyword"
    primary_header["FWHM_260"] = fwhm.to(u.arcsec).value, "[arcsec] Fake the FWHM_260 keyword"
    primary_header["FWHM_150"] = fwhm.to(u.arcsec).value, "[arcsec] Fake the FWHM_150 keyword"

    primary_header["nsources"] = 1, "Number of fake sources"
    primary_header["noise"] = noise_level.to(u.Jy / u.beam).value, "[Jy/beam] noise level per map"

    primary = fits.hdu.PrimaryHDU(header=primary_header)

    hdus = fits.hdu.HDUList(hdus=[primary])
    for band in ["1mm", "2mm"]:
        hdus.append(fits.hdu.ImageHDU(data, header=header, name="Brightness_{}".format(band)))
        hdus.append(fits.hdu.ImageHDU(uncertainty, header=header, name="Stddev_{}".format(band)))
        hdus.append(fits.hdu.ImageHDU(hits, header=header, name="Nhits_{}".format(band)))
        hdus.append(fits.hdu.BinTableHDU(sources, name="fake_sources"))

    hdus.writeto(filename, overwrite=True)

    return filename


# Special case to avoid detection, which would fail here (ShrinkMask)
def test_nikamap_phot_mask_edge(single_source_mask_edge):
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

    if nm.mask is None:
        npt.assert_allclose(nm.data, g(xx, yy))
    else:
        npt.assert_allclose(nm.data[~nm.mask], g(xx, yy)[~nm.mask])

    x, y = nm.wcs.wcs_world2pix(nm.fake_sources["ra"], nm.fake_sources["dec"], 0)
    # We are actually only testing the tolerance on x,y -> ra, dec -> x, y
    npt.assert_allclose([x, y], [nm.x, nm.y], atol=1e-13)


def test_nikamap_detect_sources(nms):

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


def test_nikamap_phot_sources(nms):

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


def test_nikamap_match_filter(nms):

    nm = nms
    mf_nm = nm.match_filter(nm.beam)

    x_idx = np.floor(nm.x + 0.5).astype(int)
    y_idx = np.floor(nm.y + 0.5).astype(int)

    npt.assert_allclose(mf_nm.data[y_idx, x_idx], nm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    npt.assert_allclose((nm.beam.fwhm * np.sqrt(2)).to(u.arcsec), mf_nm.beam.fwhm.to(u.arcsec))

    mh_nm = nm.match_filter(MexicanHat2DKernel(nm.beam.fwhm_pix.value * gaussian_fwhm_to_sigma))
    npt.assert_allclose(mh_nm.data[y_idx, x_idx], nm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    assert mh_nm.beam.fwhm is None


def test_nikamap_match_sources(nms):

    nm = nms
    nm.detect_sources()
    sources = nm.sources
    sources.meta["name"] = "to_match"
    nm.match_sources(sources)

    assert np.all(nm.sources["ID"] == nm.sources["to_match"])


def test_nikamap_match_sources_threshold(nms):

    nm = nms
    nm.detect_sources()
    sources = nm.sources
    sources.meta["name"] = "to_match"
    nm.match_sources(sources, dist_threshold=nm.beam.fwhm)

    assert np.all(nm.sources["ID"] == nm.sources["to_match"])


def test_nikamap_match_sources_list(nms):

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
def test_nikamap_plot(nms):

    nm = nms
    cax = nm.plot()

    return cax.get_figure()


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_nikamap_plot_SNR(nms):

    nm = nms
    cax = nm.plot_SNR(cbar=True)

    return cax.get_figure()


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_nikamap_plot_ax(nms):

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
        ax.legend(loc='best', frameon=False)

    return fig


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_nikamap_plot_PSD(nms):

    nm = nms
    fig, axes = plt.subplots(nrows=4, sharex=True)
    nm.plot_PSD(ax=axes[0])
    nm.plot_PSD(ax=axes[1], apod_size=5)
    nm.plot_PSD(ax=axes[2], bins=50)
    nm.plot_PSD(ax=axes[3], snr=True)

    powspec, bins = nm.plot_PSD()

    return fig


def test_nikamap_check_SNR(generate_fits):

    filename = generate_fits
    nm = NikaMap.read(filename)

    std = nm.check_SNR()
    # Tolerance comes from the fact that we biased the result using the SNR
    # cut for the fit
    npt.assert_allclose(std, 1, rtol=1e-2)


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_nikamap_check_SNR_ax(generate_fits):

    filename = generate_fits
    nm = NikaMap.read(filename)

    fig, ax = plt.subplots()
    std = nm.check_SNR(ax=ax)

    return fig


def test_retrieve_primary_keys(generate_fits):

    filename = generate_fits

    with pytest.raises(AssertionError):
        retrieve_primary_keys(filename, band="toto")

    f_sampling, bmaj = retrieve_primary_keys(filename, band="1mm")
    assert f_sampling == 10.0 * u.Hz
    assert bmaj == 3600.0 * u.arcsec
    f_sampling, bmaj = retrieve_primary_keys(filename, band="2mm")
    assert f_sampling == 10.0 * u.Hz
    assert bmaj == 3600.0 * u.arcsec


def test_nikamap_read(generate_fits):

    filename = generate_fits
    primary_header = fits.getheader(filename, 0)

    data = NikaMap.read(filename)
    data_2mm = NikaMap.read(filename, band="2mm")
    data_1mm = NikaMap.read(filename, band="1mm")

    assert np.all(data._data[~data.mask] == data_1mm._data[~data_1mm.mask])
    assert np.all(data._data[~data.mask] == data_2mm._data[~data_2mm.mask])

    assert data.beam.fwhm.to(u.arcsec).value == primary_header["FWHM_260"]
    assert np.all(data.time[~data.mask].value == ((primary_header["F_SAMPLI"] * u.Hz) ** -1).to(u.h).value)

    data_revert = NikaMap.read(filename, revert=True)
    assert np.all(data_revert._data[~data_revert.mask] == -1 * data._data[~data.mask])


def test_nikamap_write(generate_fits):
    filename = generate_fits

    data = NikaMap.read(filename)
    data_2mm = NikaMap.read(filename, band="2mm")
    data_1mm = NikaMap.read(filename, band="1mm")

    outfilename = filename.replace("map.fits", "map2.fits")
    data.write(outfilename)
    data_1mm.write(outfilename, overwrite=True)
    data_2mm.write(outfilename, append=True)


def test_nikafits_read(generate_fits):
    filename = generate_fits
    primary_header = fits.getheader(filename, 0)

    data = NikaFits.read(filename)
    assert data.primary_header == primary_header
    assert len(data) == 5
    assert isinstance(data["1mm"], NikaMap)
    assert list(data.keys()) == ["1mm", "2mm", "1", "2", "3"]
    nm = NikaMap.read(filename, band="1mm")
    assert np.nanstd((data["1mm"].subtract(nm))) == 0


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

    nm = NikaMap(data=data, mask=mask, wcs=wcs)
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

    nm = NikaMap(data=data, mask=mask, wcs=wcs)
    surface = nm.surface(box_size=1.001)
    assert np.isclose(surface.to_value(u.arcsec ** 2), 4)
