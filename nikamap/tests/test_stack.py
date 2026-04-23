from __future__ import absolute_import, division, print_function

import astropy.units as u
import numpy as np
import numpy.testing as npt
import pytest
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty, VarianceUncertainty
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.table import Table, vstack
from astropy.wcs import WCS
from photutils.datasets import make_model_image

from ..contmap import ContMap
from ..utils import cat_to_sc, pos_uniform


def gen_fake_sources(
    nsources, shape, wcs, beam_std_pix, pos_gen=pos_uniform, flux_gen=np.ones, peak_flux=1 * u.mJy, **kwargs
):
    x, y = pos_gen(nsources=nsources, shape=shape, within=(0.1, 0.9), **kwargs)
    amplitude = peak_flux  * flux_gen(nsources)
    # Source centered on pixel for the tests
    x, y = x.astype(int), y.astype(int)
    coord = wcs.pixel_to_world(x, y)
    ra = coord.ra
    dec = coord.dec
    x_stddev = np.ones_like(x) * beam_std_pix
    y_stddev = np.ones_like(x) * beam_std_pix
    theta = np.zeros_like(x)
    fake_sources = Table(
        [x, y, ra, dec, ra, dec, amplitude, x_stddev, y_stddev, theta],
        names=["x_mean", "y_mean", "ra", "dec", "_ra", "_dec", "amplitude", "x_stddev", "y_stddev", "theta"],
        masked=True,
    )
    return fake_sources


def large_map_sources(
    shape=(512, 512),
    pixsize=1 / 3 * u.arcsec,
    peak_flux=1 * u.Jy,
    noise_level=0.1 * u.Jy / u.beam,
    fwhm=1 * u.arcsec,
    dist_threshold=10 * u.arcsec,
    nsources=10,
    add_noise=False,
):
    np.random.seed(42)

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize.to("deg").value
    wcs.wcs.crval = (0, 0)
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    mask = np.zeros(shape, dtype=bool)
    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    dist_threshold_pix = (dist_threshold / pixsize).decompose().value
    fake_sources = gen_fake_sources(
        nsources,
        shape,
        wcs,
        beam_std_pix,
        pos_gen=pos_uniform,
        flux_gen=np.ones,
        peak_flux=peak_flux,
        dist_threshold=dist_threshold_pix,
    )

    data = (
        make_model_image(shape, models.Gaussian2D(), fake_sources, model_shape=shape, x_name="x_mean", y_name="y_mean")
        * peak_flux.unit
        / u.beam
    )

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"
    header["NOISE"] = noise_level.to_value(u.Jy / u.beam)
    header["BMAJ"] = fwhm.to("deg").value
    header["BMIN"] = fwhm.to("deg").value
    header["BPA"] = 0

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        wcs=wcs,
        meta=header,
        hits=hits,
        unit=u.Jy / u.beam,
        mask=mask,
        fake_sources=fake_sources,
    )

    if add_noise:
        cm._data += np.random.normal(loc=0, scale=1, size=cm.shape) * cm.uncertainty.array

    return cm


def large_map_multisources(nsources=1000, add_noise=False, zero_centered=False, flux_gen=np.ones):
    np.random.seed(42)

    shape = (128, 128)
    pixsize = 1 / 3 * u.arcsec
    peak_fluxes = np.array([1, 0.5]) * u.Jy
    noise_level = 0.1 * u.Jy / u.beam
    fwhm = 1 * u.arcsec

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize.to("deg").value
    wcs.wcs.crval = (0, 0)
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    mask = np.zeros(shape, dtype=bool)
    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma

    gen_fake_args = (nsources, shape, wcs, beam_std_pix)
    gen_fake_kwargs = {"pos_gen": pos_uniform, "flux_gen": flux_gen}
    fake_sources = [
        gen_fake_sources(*gen_fake_args, **gen_fake_kwargs, peak_flux=peak_flux) for peak_flux in peak_fluxes
    ]

    coords = [cat_to_sc(fake_source) for fake_source in fake_sources]
    amplitudes = [fake_source["amplitude"] for fake_source in fake_sources]

    fake_sources = vstack(fake_sources)
    fake_sources["fake_id"] = np.arange(len(fake_sources))

    data = make_model_image(
        shape, models.Gaussian2D(), fake_sources, model_shape=shape, x_name="x_mean", y_name="y_mean"
    )

    if zero_centered:
        source_area = 2 * np.pi * beam_std_pix**2
        mean_level = np.sum(peak_fluxes.value * nsources * source_area) / np.prod(shape)
        data -= mean_level

    data *= peak_fluxes.unit / u.beam

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"
    header["NOISE"] = noise_level.to_value(u.Jy / u.beam)
    header["BMAJ"] = fwhm.to("deg").value
    header["BMIN"] = fwhm.to("deg").value
    header["BPA"] = 0

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        wcs=wcs,
        meta=header,
        hits=hits,
        unit=u.Jy / u.beam,
        mask=mask,
        fake_sources=fake_sources,
    )
    cm.coords = coords
    cm.peak_fluxes = peak_fluxes
    cm.amplitudes = amplitudes

    if add_noise:
        cm._data += np.random.normal(loc=0, scale=1, size=cm.shape) * cm.uncertainty.array

    return cm


@pytest.fixture()
def f_large_map_sources():
    return large_map_sources(nsources=10, add_noise=False)


@pytest.fixture()
def f_large_map_sources_with_noise():
    return large_map_sources(nsources=50, add_noise=True)


@pytest.fixture()
def f_large_map_multisources():
    return large_map_multisources(nsources=1000, add_noise=False)


@pytest.fixture()
def f_large_map_multisources_zero_centered():
    return large_map_multisources(nsources=1000, add_noise=False, zero_centered=True)


@pytest.fixture()
def f_large_map_multisources_with_noise():
    return large_map_multisources(nsources=1000, add_noise=True)


def _stack_test_context(cm, size=10 * u.arcsec):
    coords = cat_to_sc(cm.fake_sources)
    npix = int(size.to(u.pixel, equivalencies=cm._pixel_scale).value) + 1
    center_pix = (npix - 1) // 2
    return coords, size, npix, center_pix


def test_stackmixin_gen_cutout2d(f_large_map_sources):
    cm = f_large_map_sources
    coords, size, npix, center_pix = _stack_test_context(cm)

    datas, weights, wcs = cm._gen_cutout2d(coords, size)
    assert datas.shape == (len(coords), npix, npix)
    assert datas.shape == weights.shape
    assert np.all(cm.wcs.wcs.cdelt == wcs.wcs.cdelt)
    # To avoid overlap in 2 sources
    npt.assert_allclose(datas[:, center_pix, center_pix], 1)


def test_stackmixin_gen_cutout2d_invalid_size(f_large_map_sources):
    cm = f_large_map_sources
    coords, size, _, _ = _stack_test_context(cm)
    del size

    with pytest.raises(ValueError):
        cm._gen_cutout2d(coords, 1 * u.m)


def test_stackmixin_gen_reproject_default(f_large_map_sources):
    pytest.importorskip("reproject")

    cm = f_large_map_sources
    coords, size, npix, center_pix = _stack_test_context(cm)

    datas, weights, wcs = cm._gen_reproject(coords, size)
    assert datas.shape == (len(coords), npix, npix)
    assert datas.shape == weights.shape
    assert np.all(cm.wcs.wcs.cdelt == wcs.wcs.cdelt)
    # To avoid overlap in 2 sources
    npt.assert_allclose(datas[:, center_pix, center_pix], 1)


@pytest.mark.parametrize("reproject_type", ["adaptive", "exact", "interp"])
def test_stackmixin_gen_reproject_subset(f_large_map_sources, reproject_type):
    pytest.importorskip("reproject")

    cm = f_large_map_sources
    coords, size, npix, center_pix = _stack_test_context(cm)

    subset = coords[0:5]
    datas, weights, wcs = cm._gen_reproject(subset, size, type=reproject_type)
    assert datas.shape == (len(subset), npix, npix)
    assert datas.shape == weights.shape
    assert np.all(cm.wcs.wcs.cdelt == wcs.wcs.cdelt)

    if reproject_type == "exact":
        npt.assert_allclose(datas[:, center_pix, center_pix], 1)


def test_stackmixin_stack_reproject(f_large_map_sources):
    pytest.importorskip("reproject")
    cm = f_large_map_sources
    coords, size, _, center_pix = _stack_test_context(cm)

    stack = cm.stack(coords, size, method="reproject")
    npt.assert_allclose(stack.data[center_pix, center_pix], 1)
    npt.assert_allclose(np.sqrt(np.median(stack.weights)), np.sqrt(np.median(cm.weights)) * np.sqrt(len(coords)))


def test_stackmixin_stack_cutout2d(f_large_map_sources_with_noise):
    cm = f_large_map_sources_with_noise
    coords, size, _, _ = _stack_test_context(cm)

    expected_std = cm.header["noise"] / np.sqrt(len(coords))

    stack = cm.stack(coords, size, method="cutout2d")
    stack_std = stack.uncertainty.represent_as(StdDevUncertainty).array
    npt.assert_allclose(stack_std, expected_std)

    stack = cm.stack(coords, size, method="cutout2d", n_bootstrap=1000)
    stack_std = stack.uncertainty.represent_as(StdDevUncertainty).array
    npt.assert_allclose(np.median(stack_std), expected_std, rtol=0.1)


def test_stackmixin_stack_reproject_custom_pixel_scale(f_large_map_sources):
    pytest.importorskip("reproject")
    cm = f_large_map_sources
    coords, size, _, _ = _stack_test_context(cm)

    pixel_scale = 1 / 10 * u.arcsec
    npix = int((size / pixel_scale).decompose().value) + 1
    center_pix = (npix - 1) // 2

    stack = cm.stack(coords, size, method="reproject", pixel_scales=pixel_scale)
    npt.assert_allclose(stack.data[center_pix, center_pix], 1)


@pytest.mark.parametrize(
    "stack_kwargs",
    [
        {"size": 10 * u.arcsec, "method": "reproject", "pixel_scales": u.Quantity((1 / 10, 1 / 10, 1 / 10), u.arcsec)},
        {"size": u.Quantity([10, 10, 10], u.arcsec), "method": "cutout2d"},
        {"size": u.Quantity([10, 10, 10], u.arcsec), "method": "reproject"},
        {"size": 1 * u.m, "method": "reproject"},
        {"size": 1 * u.arcsec, "method": "toto"},
        {"size": 1 * u.arcsec, "method": "reproject", "type": "toto"},
    ],
)
def test_stackmixin_stack_invalid_arguments(f_large_map_sources, stack_kwargs):
    if stack_kwargs["method"] == "reproject":
        pytest.importorskip("reproject")
    cm = f_large_map_sources
    coords = cat_to_sc(cm.fake_sources)

    with pytest.raises(ValueError):
        cm.stack(coords, **stack_kwargs)


def test_stackmixin_stack_simstack_fast(f_large_map_multisources):
    cm = f_large_map_multisources
    coords = cm.coords
    peak_fluxes = cm.peak_fluxes.value
    # fake_source_by_amplitude = cm.fake_sources.group_by('amplitude')
    # n_sources = [len(item) for item in fake_source_by_amplitude.groups]
    # amplitudes = fake_source_by_amplitude.groups.keys['amplitude']

    kernel = cm.beam.as_kernel((1 * u.pix).to("arcsec", equivalencies=cm._pixel_scale))
    st, _ = cm.simstack(coords, kernel=kernel, add_offset=False, fast=True)
    npt.assert_allclose(st[:2], peak_fluxes, atol=1e-10)
    st, _ = cm.simstack(coords, kernel=kernel, add_offset=True, fast=True)
    npt.assert_allclose(st[:2], peak_fluxes, atol=1e-10)

    # No kernel... should default to the beam and give the same result
    st, _ = cm.simstack(coords)
    npt.assert_allclose(st[:2], peak_fluxes, atol=1e-10)


def test_stackmixin_stack_simstack_slow(f_large_map_multisources):
    cm = f_large_map_multisources
    coords = cm.coords
    peak_fluxes = cm.peak_fluxes.value

    st, _ = cm.simstack(coords, add_offset=False, fast=False)
    npt.assert_allclose(st[:2], peak_fluxes, atol=1e-10)
    st, _ = cm.simstack(coords, add_offset=True, fast=False)
    npt.assert_allclose(st[:2], peak_fluxes, atol=1e-10)


def test_stackminxin_stack_simstack_uncertainty_zero_centered(f_large_map_multisources_zero_centered):
    cm = f_large_map_multisources_zero_centered
    coords = cm.coords
    peak_fluxes = cm.peak_fluxes.value

    st, _ = cm.simstack(coords, add_offset=True, fast=True)
    npt.assert_allclose(st[:2], peak_fluxes, atol=1e-4)

    st, _ = cm.simstack(coords, add_offset=False, fast=True)
    assert np.max(np.abs(st[:2] - peak_fluxes)) > 1e-4


def test_stackmixin_stack_simstack_uncertainty(f_large_map_multisources_with_noise):
    cm = f_large_map_multisources_with_noise
    coords = cm.coords

    beam_std_pix = cm.beam.major.to("pix", cm._pixel_scale).decompose().value * gaussian_fwhm_to_sigma
    beam_sqr_area_pix = np.pi * beam_std_pix**2

    n_sources = [len(item) for item in cm.coords]
    single_phot_variance = np.median(cm.uncertainty.represent_as(VarianceUncertainty).array) / beam_sqr_area_pix
    st_err_expected = np.sqrt(single_phot_variance / n_sources)

    _, st_err = cm.simstack(coords, add_offset=True, fast=True)
    npt.assert_allclose(st_err_expected, st_err[:2], atol=1e-3)

    _, st_err = cm.simstack(coords, add_offset=True, fast=True, n_bootstrap=1000)
    npt.assert_allclose(st_err_expected, st_err[:2], atol=1e-3)


def side_cat_to_fake_sources(cat, wcs, model, amplitude_name="SNIKA1200", x_name="x_mean", y_name="y_mean"):
    from astropy.coordinates import SkyCoord

    cat["coords"] = SkyCoord(cat["ra"], cat["dec"])

    fake_sources = dict()
    fake_sources["amplitude"] = cat[amplitude_name]
    x, y = wcs.world_to_pixel(cat["coords"])
    fake_sources[x_name] = x
    fake_sources[y_name] = y

    for item, value in zip(model.param_names, model.parameters):
        if item not in ["amplitude", x_name, y_name]:
            fake_sources[item] = np.ones_like(x) * value

    return Table(fake_sources)


def read_side_cat(filename):
    from astropy.table import QTable

    cat = QTable.read(filename)
    for key in ["ra", "dec"]:
        cat[key].unit = u.deg
    return cat


def read_side_map(filename):
    from astropy.io import fits
    from astropy.wcs import WCS

    data_map, hdr_map = fits.getdata(filename, header=True)
    data = ContMap(data_map, uncertainty=np.ones_like(data_map), wcs=WCS(hdr_map), meta=hdr_map)
    return data


def check_side():
    from astropy.coordinates import SkyCoord

    from nikamap import NikaMap

    cat_filename = "/data/PHOTOM/piic/reduction/sim/COSMOS_cat_13.fits"
    img_filename = "/data/PHOTOM/piic/reduction/sim/COSMOS_a13_sim_13.fits"
    sim_filename = "/data/PHOTOM/piic/reduction/COSMOS/sim_10_13_COSMOS_a13/red/COSMOS_a13_MP90_00naydwfyyBE0_0t1l17s0_0_0o0_0_0_10_50meS12_0sn4sm0nz0i10n10.fits"

    cat = read_side_cat(cat_filename)
    data = read_side_map(img_filename)
    sim_data = NikaMap.read(sim_filename, format="piic")

    # Stack high-z sources:
    z_mask = (cat["redshift"] > 3) & (cat["redshift"] < 3.2)
    sub_cat = cat[z_mask]
    coords = SkyCoord(sub_cat["ra"], sub_cat["dec"])

    x, y = sim_data.wcs.world_to_pixel(coords)  # Just to check that the coords are within the map
    within_sim = (x >= 0) & (x < sim_data.shape[1]) & (y >= 0) & (y < sim_data.shape[0])
    coords = coords[within_sim]

    stack = data.stack(coords, 40 * u.arcsec)
    stack.phot_sources(Table({"ra": [0] * u.deg, "dec": [0] * u.deg}), peak=False, psf=True, background=True)
    result = data.simstack(coords, add_offset=True, fast=True)

    sim_stack = sim_data.stack(coords, 40 * u.arcsec)
    sim_stack.phot_sources(Table({"ra": [0] * u.deg, "dec": [0] * u.deg}), peak=False, psf=True, background=True)

    # fwhm_major_pix = data.beam.major.to_value(u.pix, equivalencies=data._pixel_scale)
    # fwhm_minor_pix = data.beam.minor.to_value(u.pix, equivalencies=data._pixel_scale)

    # model = models.Gaussian2D(
    #     x_stddev=fwhm_major_pix * gaussian_fwhm_to_sigma, y_stddev=fwhm_minor_pix * gaussian_fwhm_to_sigma
    # )

    # params_table = side_cat_to_fake_sources(
    #     cat,
    #     data.wcs,
    #     model=model,
    #     amplitude_name="SNIKA1200",
    # )

    # name_kwargs = {"x_name": "x_mean", "y_name": "y_mean"}
    # # fake_img = make_model_image(data.shape, model, params_table, model_shape=data.shape, **name_kwargs)

    # kernel = Model2DKernel(model, x_size=11, y_size=11)

    # # Too much memory... split...
    # nsplits = 100
    # fake_img = []
    # for sub_params in np.array_split(params_table, nsplits):
    #     _fake_img = make_kernel_image(data.shape, kernel, Table(sub_params), **name_kwargs)
    #     fake_img.append(_fake_img)
    # fake_img = np.sum(fake_img, axis=0)

    # fake_img = make_kernel_image(data.shape, kernel, params_table, **name_kwargs)


def check_bootstrap_alg():
    from functools import partial

    import matplotlib.pyplot as plt
    import numpy as np
    from numpy.random import RandomState
    from tqdm.contrib.concurrent import process_map

    plt.ion()

    uncertainty = 2
    intrinsinc_scale = 1

    nsims = 50
    nsample = 100
    n_bootstraps = 30

    bootstraps = np.unique((np.logspace(np.log10(0.1), np.log10(1_000), n_bootstraps) * nsample).astype(int))

    def shuffled_average(samples, weights):
        nsample = len(samples)
        indices = np.random.choice(nsample, size=nsample, replace=True)
        return np.average(samples[indices], weights=weights[indices])

    def bootstrap_std(test, noise, weight, n_bootstrap):
        bs_array = np.array([shuffled_average(test, weight) for _ in range(n_bootstrap)])
        bs_std = np.std(bs_array)
        bs_array = np.array([shuffled_average(noise, weight) for _ in range(n_bootstrap)])
        bs_noise_std = np.std(bs_array)
        return bs_std, bs_noise_std

    def gen_bootstrap_data(intrinsinc_scale, uncertainty, nsample, distrib="normal"):
        prng = RandomState()
        noise = prng.normal(loc=0, scale=uncertainty, size=(nsample))
        if distrib == "normal":
            test = prng.normal(loc=0, scale=intrinsinc_scale, size=(nsample))
        elif distrib == "exponential":
            test = prng.standard_exponential(size=nsample)
        else:
            raise ValueError("Invalid distribution")
        test += noise
        weight = np.full(nsample, 1 / uncertainty**2)
        return test, noise, weight

    def bootstrap_item(intrinsinc_scale, uncertainty, nsample, bootstraps, *args, distrib="normal"):
        test, noise, weight = gen_bootstrap_data(intrinsinc_scale, uncertainty, nsample, distrib=distrib)

        _bs_std = np.zeros(len(bootstraps))
        _bs_noise_std = np.zeros(len(bootstraps))
        for k, n_bootstrap in enumerate(bootstraps):
            _bs_std[k], _bs_noise_std[k] = bootstrap_std(test, noise, weight, n_bootstrap)
        return _bs_std, _bs_noise_std

    _ = partial(bootstrap_item, intrinsinc_scale, uncertainty, nsample, bootstraps, distrib="normal")
    results = process_map(_, range(nsims), max_workers=12, chunksize=1)

    results = np.array(results)

    theory_noise_std = np.sqrt(uncertainty**2 / nsample)  # + intrinsinc_scale**2 / nsample)
    theory_std = np.sqrt(uncertainty**2 / nsample + intrinsinc_scale**2 / nsample)

    bs_std = results[:, 0]
    bs_std_mean = bs_std.mean(0)
    bs_std_std = bs_std.std(0)
    bs_noise_std = results[:, 1]
    bs_noise_std_mean = bs_noise_std.mean(0)
    bs_noise_std_std = bs_noise_std.std(0)

    x = bootstraps / nsample
    plt.figure()
    c = plt.fill_between(x, bs_std_mean - bs_std_std, bs_std_mean + bs_std_std, alpha=0.3)
    plt.plot(x, bs_std.T, c=c.get_facecolor()[0], alpha=0.1)
    plt.plot(x, bs_std_mean, label="Bootstrap Std", c=c.get_facecolor()[0], linewidth=5, alpha=1)

    c = plt.fill_between(x, bs_noise_std_mean - bs_noise_std_std, bs_noise_std_mean + bs_noise_std_std, alpha=0.3)
    plt.plot(x, bs_noise_std.T, c=c.get_facecolor()[0], alpha=0.1)
    plt.plot(x, bs_noise_std_mean, label="Bootstrap Noise Std", c=c.get_facecolor()[0], linewidth=5, alpha=1)

    plt.axhline(theory_std, color="k", ls="--", label="Theoretical Std")
    plt.axhline(theory_noise_std, color="b", ls="--", label="Theoretical Noise Std")
    plt.xscale("log")
    plt.xlabel("Number of Bootstraps / Sample Size")
    plt.ylabel("Bootstrap Std")
    plt.legend()