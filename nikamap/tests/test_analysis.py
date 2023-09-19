from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table, Column

from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.coordinates import SkyCoord
from photutils.datasets import make_gaussian_sources_image

import numpy.testing as npt

# import nikamap as nm
# data_path = op.join(nm.__path__[0], 'data')

from ..analysis import HalfDifference, MultiScans, Jackknife, Bootstrap, StackMap
from ..contmap import ContMap, contmap_average
from ..nikamap import NikaMap

Jybeam = u.Jy / u.beam


def generate_nikamaps(
    tmpdir_factory,
    shape=(257, 257),
    pixsize=1 / 3 * u.deg,
    noise_level=1 * Jybeam,
    nmaps=10,
    nsources=5,
    fwhm=1 * u.deg,
    nrots=4,
):
    # Generate several maps with sources and noise... only one band...

    tmpdir = tmpdir_factory.mktemp("nm_maps")

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    # Fake sources for all maps
    np.random.seed(0)

    sources = Table(masked=True)
    sources["amplitude"] = np.random.uniform(1, 10, size=nsources) * u.Jy
    sources["x_mean"] = np.random.uniform(1 / 4, 3 / 4, size=nsources) * shape[1]
    sources["y_mean"] = np.random.uniform(1 / 4, 3 / 4, size=nsources) * shape[0]

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    sources["x_stddev"] = np.ones(nsources) * beam_std_pix
    sources["y_stddev"] = np.ones(nsources) * beam_std_pix
    sources["theta"] = np.zeros(nsources)

    data_sources = make_gaussian_sources_image(shape, sources) * u.Jy / u.beam

    a, d = wcs.wcs_pix2world(sources["x_mean"], sources["y_mean"], 0)
    sources.add_columns([Column(a * u.deg, name="ra"), Column(d * u.deg, name="dec")])
    sources.remove_columns(["x_mean", "y_mean", "x_stddev", "y_stddev", "theta"])
    sources.sort("amplitude")
    sources.reverse()
    sources.add_column(Column(np.arange(len(sources)), name="ID"), 0)

    # Elliptical gaussian mask
    def elliptical_mask_rot(shape, sigma, theta, limit):
        xx, yy = np.indices(shape)

        xx_arr = xx - (shape[1] - 1) / 2
        yy_arr = yy - (shape[0] - 1) / 2

        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s), (s, c)))

        xx_arr, yy_arr = np.dot(rot, [xx_arr.flatten(), yy_arr.flatten()])

        xx_arr = (xx_arr.reshape(shape) / (2 * sigma[1] ** 2)) ** 2
        yy_arr = (yy_arr.reshape(shape) / (2 * sigma[0] ** 2)) ** 2

        mask = np.sqrt(xx_arr + yy_arr) > limit
        return mask

    #  mask = mask_rot(shape, (np.sqrt(0.5), np.sqrt(0.5)), 0, (shape[0] - 1) / 2)

    primary_header = fits.header.Header()
    primary_header["f_sampli"] = 10.0, "Fake the f_sampli keyword"
    primary_header["FWHM_260"] = (
        fwhm.to(u.arcsec).value,
        "[arcsec] Fake the FWHM_260 keyword",
    )
    primary_header["FWHM_150"] = (
        fwhm.to(u.arcsec).value,
        "[arcsec] Fake the FWHM_150 keyword",
    )

    primary_header["nsources"] = nsources, "Number of fake sources"
    primary_header["pixsize"] = pixsize.to(u.deg).value, "[deg] pixel size"
    primary_header["nmaps"] = nmaps, "number of maps produced"
    primary_header["nrots"] = nmaps, "number of rotations"
    primary_header["shape0"] = shape[0], "[0] of map shape"
    primary_header["shape1"] = shape[1], "[1] of map shape"
    primary_header["noise"] = (
        noise_level.to(u.Jy / u.beam).value,
        "[Jy/beam] noise level per map",
    )

    primary = fits.hdu.PrimaryHDU(header=primary_header)

    filenames = []

    for i_map in range(nmaps):
        # Rotated asymetric mask
        mask = elliptical_mask_rot(
            shape,
            (np.sqrt(1 / 2), np.sqrt(2 / 5)),
            i_map * np.pi / nrots,
            (shape[0] - 1) / 2,
        )

        filename = str(tmpdir.join("map_{}.fits".format(i_map)))

        hits = np.ones(shape=shape, dtype=float)
        uncertainty = np.ones(shape=shape, dtype=float) * noise_level
        data = np.random.normal(loc=0, scale=1, size=shape) * uncertainty

        data += data_sources
        data[mask] = 0
        hits[mask] = 0
        uncertainty[mask] = 0

        header = wcs.to_header()
        header["UNIT"] = "Jy / beam", "Fake Unit"

        hdus = fits.hdu.HDUList(hdus=[primary])

        for band in ["1mm", "2mm"]:
            hdus.append(fits.hdu.ImageHDU(data.value, header=header, name="Brightness_{}".format(band)))
            hdus.append(fits.hdu.ImageHDU(uncertainty.value, header=header, name="Stddev_{}".format(band)))
            hdus.append(fits.hdu.ImageHDU(hits, header=header, name="Nhits_{}".format(band)))
            hdus.append(fits.hdu.BinTableHDU(sources, name="fake_sources"))

        hdus.writeto(filename, overwrite=True)

        filenames.append(filename)

    return filenames


@pytest.fixture(name="generate_nikamaps", scope="session")
def generate_nikamaps_fixture(tmpdir_factory):
    # default arguments
    return generate_nikamaps(tmpdir_factory)


def test_contmap_average(generate_nikamaps):
    filenames = generate_nikamaps

    # Read the nikamap as contmap
    nms = [NikaMap.read(filename) for filename in filenames]
    nm = contmap_average(nms)
    assert nm.shape == nms[0].shape
    npt.assert_allclose(nm.check_SNR(range=(-6, 1)), 1, atol=1e-2)

    nm = contmap_average(nms, normalize=True)
    assert nm.shape == nms[0].shape
    npt.assert_allclose(nm.check_SNR(range=(-6, 1)), 1, atol=1e-2)

def test_MultiScans_init(generate_nikamaps):
    filenames = generate_nikamaps

    ms = MultiScans(filenames, n=None)

    ms2 = MultiScans(ms, n=1)
    assert ms.filenames == ms2.filenames
    npt.assert_equal(ms.datas, ms2.datas)
    npt.assert_equal(ms.weights, ms2.weights)
    npt.assert_equal(ms.hits, ms2.hits)
    npt.assert_equal(ms.mask, ms2.mask)


def test_MultiScans_iterator(generate_nikamaps):
    filenames = generate_nikamaps

    iterator = MultiScans(filenames, n=10)
    assert len(iterator) == 10

    assert len(list(iterator)) == 10

    with pytest.raises(StopIteration):
        next(iterator)


def test_HalfDifference_average(generate_nikamaps):
    filenames = generate_nikamaps

    primary_header = fits.getheader(filenames[0], 0)
    weighted_noise = primary_header["NOISE"] / np.sqrt(primary_header["NMAPS"])

    # Weighted average
    hd = HalfDifference(filenames, n=None, parity_threshold=1)
    data = next(hd)
    assert np.all(data.uncertainty.array[~data.mask] == weighted_noise)

    data_call = hd()
    npt.assert_equal(data.data[~data.mask], data_call.data[~data_call.mask])


def test_HalfDifference_call(generate_nikamaps):
    filenames = generate_nikamaps

    primary_header = fits.getheader(filenames[0], 0)
    weighted_noise = primary_header["NOISE"] / np.sqrt(primary_header["NMAPS"])

    # Produce one HalfDifference
    hd = HalfDifference(filenames, n=1, parity_threshold=1)
    data = hd()

    shape = data.shape
    norm = data.hits / data.hits[(shape[1] - 1) // 2, (shape[0] - 1) // 2]
    npt.assert_allclose((data.uncertainty.array * norm**0.5)[~data.mask], weighted_noise)


def test_HalfDifference_parity_set(generate_nikamaps):
    filenames = generate_nikamaps

    hd = HalfDifference(filenames, parity_threshold=0)
    assert hd.parity_threshold == 0

    hd = HalfDifference(filenames, parity_threshold=0.4)
    assert hd.parity_threshold == 0.4

    hd = HalfDifference(filenames)
    assert hd.parity_threshold == 1

    with pytest.raises(TypeError):
        hd = HalfDifference(filenames, parity_threshold=-0.1)

    with pytest.raises(TypeError):
        hd = HalfDifference(filenames, parity_threshold=1.1)


def test_HalfDifference_parity(generate_nikamaps):
    filenames = generate_nikamaps

    primary_header = fits.getheader(filenames[0], 0)
    weighted_noises = primary_header["NOISE"] / np.sqrt(np.arange(1, primary_header["NMAPS"] + 1))

    hd = HalfDifference(filenames, n=None, parity_threshold=0)
    data = hd()
    uncertainties = np.unique(data.uncertainty.array[~data.mask])

    assert np.all([True if uncertainty in weighted_noises else False for uncertainty in uncertainties])

    hd.parity_threshold = 0.5
    data = hd()
    uncertainties = np.unique(data.uncertainty.array[~data.mask])

    assert np.all([True if uncertainty in weighted_noises else False for uncertainty in uncertainties])

    hd = HalfDifference(filenames, n=1, parity_threshold=0)
    data = hd()
    uncertainties = np.unique(data.uncertainty.array[~data.mask])

    assert np.all([True if uncertainty in weighted_noises else False for uncertainty in uncertainties])

    hd.parity_threshold = 0.5
    data = hd()
    uncertainties = np.unique(data.uncertainty.array[~data.mask])

    assert np.all([True if uncertainty in weighted_noises else False for uncertainty in uncertainties])


def test_HalfDifference_odd(generate_nikamaps):
    filenames = generate_nikamaps

    # Odd number
    with pytest.warns(UserWarning):
        _ = HalfDifference(filenames[1:], n=1)


def test_HalfDifference_assert(generate_nikamaps):
    filenames = generate_nikamaps

    # Non existent files
    with pytest.warns(UserWarning):
        _ = HalfDifference([filenames[0], filenames[1], "toto.fits"], n=1)

    # Non existent files
    with pytest.raises(AssertionError):
        _ = HalfDifference([filenames[0]], n=1)

    # Non existent files
    with pytest.warns(UserWarning):
        with pytest.raises(AssertionError):
            _ = HalfDifference([filenames[0], "toto.fits"], n=1)


def test_Bootstrap(generate_nikamaps):
    filenames = generate_nikamaps

    primary_header = fits.getheader(filenames[0], 0)
    weighted_noises = primary_header["NOISE"] / np.sqrt(np.arange(1, primary_header["NMAPS"] + 1))

    # Weighted average
    bs = Bootstrap(filenames)
    data = bs()

    # Most likely weight and median absolute deviation should be the minimal noise
    full_coverage = data.hits==len(filenames)
    med = np.nanmedian(data.uncertainty.array[full_coverage])
    mad = np.nanmedian(np.abs(data.uncertainty.array[full_coverage] - med))
    assert (weighted_noises.min() - med) < mad

    # Trouble to find a proper test for this
    bs = Bootstrap(filenames, n=10)
    data = bs()


def test_Jackknife(generate_nikamaps):
    filenames = generate_nikamaps

    primary_header = fits.getheader(filenames[0], 0)
    weighted_noises = primary_header["NOISE"] / np.sqrt(np.arange(1, primary_header["NMAPS"] + 1))

    # Weighted average
    jk = Jackknife(filenames, n=None)
    data = jk()

    # Most likely weight and median absolute deviation should be the minimal noise
    full_coverage = data.hits==len(filenames)
    med = np.nanmedian(data.uncertainty.array[full_coverage])
    mad = np.nanmedian(np.abs(data.uncertainty.array[full_coverage] - med))
    assert (weighted_noises.min() - med) < mad

    # Trouble to find a proper test for this


def uniform_no_overlap(nsources, shape, marging=1 / 8, min_dist=None, oversample=5):
    x = np.random.uniform(size=oversample * nsources, low=shape[1] * marging, high=shape[1] * (1 - marging))
    y = np.random.uniform(size=oversample * nsources, low=shape[0] * marging, high=shape[0] * (1 - marging))

    for idx in range(0, nsources):
        remainder_to_keep = np.sqrt((x[idx] - x[idx + 1 :]) ** 2 + (y[idx] - y[idx + 1 :]) ** 2) > min_dist
        x = np.concatenate([x[: idx + 1], x[idx + 1 :][remainder_to_keep]])
        y = np.concatenate([y[: idx + 1], y[idx + 1 :][remainder_to_keep]])

    x = x[: idx + 1]
    y = y[: idx + 1]
    assert x.shape == (nsources,)

    return x, y


@pytest.fixture()
def large_map_sources_centered():
    np.random.seed(42)

    shape = (512, 512)
    pixsize = 1 / 3 * u.arcsec
    peak_flux = 1 * u.Jy
    noise_level = 0.1 * u.Jy / u.beam
    fwhm = 1 * u.arcsec
    nsources = 10

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize.to("deg").value
    wcs.wcs.crval = (0, 0)
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    xx, yy = np.indices(shape)
    mask = np.zeros(shape, dtype=bool)
    # mask = np.sqrt((xx - (shape[1] - 1) / 2) ** 2 + (yy - (shape[0] - 1) / 2) ** 2) > shape[0] / 2

    # Sources will fall at the center of each pixel
    sources = Table(masked=True)
    sources["amplitude"] = np.ones(nsources) * peak_flux
    x, y = uniform_no_overlap(nsources, shape, 1 / 8, min_dist=(fwhm / pixsize).decompose().value * 3)

    # Put them at the center of the pixel to ease the tests !
    sources["x_mean"] = x.astype(int)
    sources["y_mean"] = y.astype(int)

    ra, dec = wcs.wcs_pix2world(sources["x_mean"], sources["y_mean"], 0)
    sources["ra"] = ra * u.deg
    sources["dec"] = dec * u.deg

    sources["_ra"] = sources["ra"]
    sources["_dec"] = sources["dec"]

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    sources["x_stddev"] = np.ones(nsources) * beam_std_pix
    sources["y_stddev"] = np.ones(nsources) * beam_std_pix
    sources["theta"] = np.zeros(nsources)

    data = make_gaussian_sources_image(shape, sources)

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level.to(u.Jy / u.beam).value
    # data += np.random.normal(loc=0, scale=1, size=shape) * uncertainty
    # data[mask] = np.nan
    # hits[mask] = 0
    # uncertainty[mask] = 0

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"
    header["BMAJ"] = fwhm.to("deg").value
    header["BMIN"] = fwhm.to("deg").value
    header["BPA"] = 0

    cm = ContMap(
        data,
        uncertainty=uncertainty,
        wcs=wcs,
        meta=header,
        hits=hits,
        unit=u.Jy / u.beam,
        mask=mask,
        fake_sources=sources,
    )

    return cm


def test_StackMap(large_map_sources_centered):
    cm = large_map_sources_centered
    coords = SkyCoord(cm.fake_sources["_ra"], cm.fake_sources["_dec"])
    size = 10 * u.arcsec

    npix = int(size.to(u.pixel, equivalencies=cm._pixel_scale).value) + 1
    center_pix = (npix - 1) // 2

    sm = StackMap(cm)

    datas, weights, wcs = sm._gen_cutout2d(coords, size)
    assert datas.shape == (len(coords), npix, npix)
    assert datas.shape == weights.shape
    assert np.all(sm.wcs.wcs.cdelt == wcs.wcs.cdelt)
    # To avoid overlap in 2 sources
    npt.assert_allclose(datas[:, center_pix, center_pix], 1)

    with pytest.raises(ValueError):
        sm._gen_cutout2d(coords, 1 * u.m)

    datas, weights, wcs = sm._gen_reproject(coords, size)
    assert datas.shape == (len(coords), npix, npix)
    assert datas.shape == weights.shape
    assert np.all(sm.wcs.wcs.cdelt == wcs.wcs.cdelt)
    # To avoid overlap in 2 sources
    npt.assert_allclose(datas[:, center_pix, center_pix], 1)

    datas, weights, wcs = sm._gen_reproject(coords[0:5], size, type="adaptive")
    assert datas.shape == (len(coords[0:5]), npix, npix)
    assert datas.shape == weights.shape
    assert np.all(sm.wcs.wcs.cdelt == wcs.wcs.cdelt)
    # To avoid overlap in 2 sources
    # flux is NOT conserved !!!!
    # npt.assert_allclose(datas[:, center_pix, center_pix], 1)

    datas, weights, wcs = sm._gen_reproject(coords[0:5], size, type="exact")
    assert datas.shape == (len(coords[0:5]), npix, npix)
    assert datas.shape == weights.shape
    assert np.all(sm.wcs.wcs.cdelt == wcs.wcs.cdelt)
    npt.assert_allclose(datas[:, center_pix, center_pix], 1)

    stack = sm.stack(coords, size, method="reproject")
    npt.assert_allclose(stack.data[center_pix, center_pix], 1)
    npt.assert_allclose(np.sqrt(np.median(sm.weights)) * np.sqrt(len(coords)), np.sqrt(np.median(stack.weights)))

    stack = sm.stack(coords, size, method="reproject", n_bootstrap=100)
    npt.assert_allclose(1 / np.sqrt(np.median(stack.weights)), 0, atol=1e-25)

    pixel_scale = 1 / 10 * u.arcsec
    npix = int((size / pixel_scale).decompose().value) + 1
    center_pix = (npix - 1) // 2

    stack = sm.stack(coords, size, method="reproject", pixel_scales=pixel_scale)
    npt.assert_allclose(stack.data[center_pix, center_pix], 1)

    with pytest.raises(ValueError):
        sm.stack(coords, size, method="reproject", pixel_scales=u.Quantity((pixel_scale, pixel_scale, pixel_scale)))

    with pytest.raises(ValueError):
        sm.stack(coords, u.Quantity([size, size, size]), method="cutout2d")

    with pytest.raises(ValueError):
        sm.stack(coords, u.Quantity([size, size, size]), method="reproject")

    with pytest.raises(ValueError):
        sm.stack(coords, 1 * u.m, method="reproject")

    with pytest.raises(ValueError):
        sm.stack(coords, 1 * u.arcsec, method="toto")

    with pytest.raises(ValueError):
        sm.stack(coords, 1 * u.arcsec, method="reproject", type="toto")


# To be run interactively to get a fixture for debugging
def interactive_fixture():
    import py.path

    tmpdir = py.path.local()
    tmpdir.mktemp = tmpdir.mkdir
    filenames = generate_nikamaps(tmpdir)
    return filenames
