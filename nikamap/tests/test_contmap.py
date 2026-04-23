from __future__ import absolute_import, division, print_function

import warnings

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import numpy.testing as npt
import pytest
from astropy.convolution import Gaussian2DKernel, Kernel2D, RickerWavelet2DKernel
from astropy.io import fits
from astropy.modeling import models
from astropy.nddata import InverseVariance, StdDevUncertainty, VarianceUncertainty
from astropy.stats.funcs import gaussian_fwhm_to_sigma
from astropy.table import Table
from astropy.wcs import WCS
from photutils.datasets import make_model_image
from scipy import signal

from ..contmap import ContBeam, ContMap
from ..utils import pos_gridded, shrink_mask


def prototype_match_filter_corrected(cm, kernel, apply_flux_correction=True):
    """Prototype version of ``match_filter`` with explicit flux/uncertainty correction.

    This function mirrors the current ``ContMap.match_filter`` implementation and
    makes the following changes explicit:


    Parameters
    ----------
    cm : ContMap
        Input map.
    kernel : ContBeam or Kernel2D
        Filtering kernel.
    apply_flux_correction : bool, optional
        If True, apply the explicit correction factor. If False, this function
        is equivalent to the current implementation (up to numerical precision).
    """
    from copy import deepcopy

    kernel = deepcopy(kernel)

    # Same beam update as in current match_filter.
    mf_beam = cm.beam.convolve(kernel)

    # Same mask shrink behavior as in current match_filter.
    kernel.normalize("integral")
    if cm.mask is not None:
        mf_mask = shrink_mask(cm.mask, kernel)
    else:
        mf_mask = None

    # Estimator normalization choice:
    # - peak: current production behavior
    # - integral: possible alternative, but not unit-preserving by itself
    kernel.normalize("peak")
    kernel_sqr = kernel.array**2

    if cm.hits is not None:
        mf_hits = signal.fftconvolve(np.asarray(cm.hits), kernel_sqr, mode="same")
        if mf_mask is not None:
            mf_hits[mf_mask] = 0
    else:
        mf_hits = None

    weights = cm.weights
    if cm.mask is not None:
        weights[cm.mask] = 0

    with np.errstate(invalid="ignore", divide="ignore"):
        mf_uncertainty = 1 / np.sqrt(signal.fftconvolve(weights, kernel_sqr, mode="same"))
    if mf_mask is not None:
        mf_uncertainty[mf_mask] = np.nan

    mf_data = signal.fftconvolve(weights * cm.__array__().filled(0), kernel, mode="same") * mf_uncertainty**2

    # Explicit correction for kernel != beam.
    # Important: scale data and uncertainty identically.
    #
    # For uniform weights and a point source with beam template B (peak=1),
    # the MF peak response is:
    #     R = <K, B> / <K, K>
    # where K is the effective estimator kernel (with chosen normalization).
    # So the exact correction is:
    #     C = 1 / R = <K, K> / <K, B>
    #
    # In the Gaussian+peak-normalized limit this reduces to:
    #     C = (sigma_b^2 + sigma_k^2) / (2 sigma_b^2)
    if apply_flux_correction:
        cm.beam.normalize("peak")
        beam_template = cm.beam.array

        kernel.normalize("peak")
        kk = np.nansum(kernel.array**2)

        kb_map = signal.fftconvolve(beam_template, kernel, mode="same")
        cy, cx = (kb_map.shape[0] - 1) // 2, (kb_map.shape[1] - 1) // 2
        kb = kb_map[cy, cx]

        if np.isfinite(kk) and np.isfinite(kb) and kk > 0 and kb != 0:
            flux_correction = kk / kb
            mf_data *= flux_correction
            mf_uncertainty *= flux_correction
        else:
            warnings.warn(
                "prototype_match_filter_corrected: flux correction skipped (invalid <K,K> or <K,B>)",
                UserWarning,
            )

    return ContMap(
        mf_data,
        mask=mf_mask,
        hits=mf_hits,
        uncertainty=StdDevUncertainty(mf_uncertainty),
        beam=mf_beam,
        unit=cm.unit,
        sampling_freq=cm.sampling_freq,
        wcs=cm.wcs,
        meta=cm.meta,
        fake_sources=cm.fake_sources,
    )


def test_contbeam_init():
    # TODO: What if we init with an array ?
    fwhm = 18 * u.arcsec
    # array = np.ones((10, 10))
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
    npt.assert_allclose(ref_kernel.array, kernel.array)
    assert beam.sr == (2 * np.pi * (fwhm * gaussian_fwhm_to_sigma) ** 2).to(u.sr)

    beam = ContBeam(array=ref_kernel.array, pixscale=pixscale)
    assert beam.major is None
    assert str(beam) == "ContBeam: (63, 63) Kernel2D at pixscale 2.0 arcsec"
    npt.assert_almost_equal(beam.sr.value, (2 * np.pi * (fwhm * gaussian_fwhm_to_sigma) ** 2).to(u.sr).value)
    with pytest.raises(TypeError):
        kernel = beam.as_kernel()
        kernel = beam.as_kernel(2 * pixscale)

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
        beam_refconvolve = beam.convolve(ContBeam(fwhm, pixscale=2 * pixscale))

    center = (beam_refconvolve.shape[0] - 1) // 2
    size = (beam_convolve.shape[0] - 1) // 2
    _slice = slice(center - size, center + size + 1)
    npt.assert_almost_equal(beam_refconvolve.array[_slice, _slice], beam_convolve.array)


def test_contmap_init():
    data = [1, 2, 3]
    cm = ContMap(data)
    assert np.all(cm.data == np.array(data))

    # Should default to empty wcs and no unit
    assert cm.wcs is None
    assert cm.unit is u.adu
    assert cm.uncertainty is None

    # time "empty"
    assert cm.time is None

    # Default pixsize 1*u.deg
    assert (1 * u.pixel).to(u.deg, equivalencies=cm._pixel_scale) == 1 * u.deg

    # Default beam fwhm 1*u.deg
    assert cm.beam.major == 1 * u.deg


def test_contmap_init_quantity():
    data = np.array([1, 2, 3]) * u.Jy / u.beam
    cm = ContMap(data)
    assert cm.unit == u.Jy / u.beam


def test_contmap_init_meta():
    data = np.array([1, 2, 3])
    header = fits.header.Header()

    header["CDELT1"] = -1.0 / 3600, "pixel size used for pixel_scale"
    header["BMAJ"] = 1.0 / 3600, "Beam Major Axis"
    cm = ContMap(data, meta=header)
    assert (1 * u.pixel).to(u.deg, equivalencies=cm._pixel_scale) == 1 * u.arcsec
    assert cm.beam.major == 1 * u.arcsec
    assert cm.beam.minor == 1 * u.arcsec

    # Full header
    header["CRPIX1"] = 1
    header["CRPIX2"] = 2
    header["CDELT1"] = -1 / 3600
    header["CDELT2"] = 1 / 3600
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"

    cm = ContMap(data, meta=header, wcs=WCS(header))
    assert cm.wcs is not None


def test_contmap_init_uncertainty():
    data = np.array([1, 2, 3])
    uncertainty = np.array([1, 1, 1])

    # Default to StdDevUncertainty...
    with pytest.warns(UserWarning):
        cm = ContMap(data, uncertainty=uncertainty)
    assert isinstance(cm.uncertainty, StdDevUncertainty)
    assert np.all(cm.uncertainty.array == np.array([1, 1, 1]))

    cm_mean = cm.add(cm).divide(2)
    assert np.all(cm_mean.data == cm.data)
    npt.assert_allclose(cm_mean.uncertainty.array, np.array([1, 1, 1]) / np.sqrt(2))

    # Wrong size
    with pytest.raises(ValueError):
        cm = ContMap(data, uncertainty=uncertainty[1:])

    # Wrong TypeError
    with pytest.raises(TypeError):
        cm = ContMap(data, uncertainty=list(uncertainty))

    iv_uncertainty = InverseVariance(uncertainty)
    cm = ContMap(data, uncertainty=iv_uncertainty)
    assert np.all(cm.snr == data)

    v_uncertainty = VarianceUncertainty(uncertainty)
    cm = ContMap(data, uncertainty=v_uncertainty)
    assert np.all(cm.snr == data)


def test_contmap_compressed():
    data = np.array([1, 2, 3])
    uncertainty = np.array([10, 1, 1], dtype=float)
    mask = np.array([True, False, False])
    hits = np.ones(3)
    sampling_freq = 1 * u.Hz

    cm = ContMap(data, uncertainty=uncertainty, mask=mask, hits=hits, sampling_freq=sampling_freq, unit=u.Jy)

    assert np.all(cm.compressed() == np.array([2, 3]) * u.Jy)
    assert np.all(cm.uncertainty_compressed() == np.array([1, 1]) * u.Jy)

    assert np.all(cm.__array__() == np.ma.array(data, mask=mask))
    assert np.all(cm.__u_array__() == np.ma.array(uncertainty, mask=mask))

    # To insure compatilibity with Astropy 3.0, maskedQuantity cannot evaluate
    # truth value of quantities
    assert np.all(cm.__t_array__().data == hits / sampling_freq)
    assert np.all(cm.__t_array__().mask == mask)


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

    cm = ContMap(data, uncertainty=np.ones(shape), wcs=wcs, unit=u.Jy / u.beam)

    return cm


def test_no_source(no_source):
    cm = no_source
    cm.detect_sources(threshold=5)
    assert cm.sources is None


@pytest.fixture()
def single_source():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    uncertainty = np.ones_like(data) / 4

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    cm = ContMap(
        data, uncertainty=StdDevUncertainty(uncertainty), wcs=wcs, unit=u.Jy / u.beam, hits=np.ones_like(data, int)
    )

    # Additionnal attribute just for the tests...
    cm.x = np.asarray([shape[1] / 2 - 0.5])
    cm.y = np.asarray([shape[0] / 2 - 0.5])
    cm.add_gaussian_sources(nsources=1, peak_flux=1 * u.Jy, within=(1 / 2, 1 / 2))
    return cm


@pytest.fixture()
def single_source_side():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    uncertainty = np.ones_like(data) / 4
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    fake_sources = Table(masked=True)
    fake_sources["fake_id"] = [1]
    fake_sources["amplitude"] = [1]
    fake_sources["x_mean"] = [0]
    fake_sources["y_mean"] = [13]

    ra, dec = wcs.wcs_pix2world(fake_sources["x_mean"], fake_sources["y_mean"], 0)
    fake_sources["ra"] = ra * u.deg
    fake_sources["dec"] = dec * u.deg

    fake_sources["_ra"] = fake_sources["ra"]
    fake_sources["_dec"] = fake_sources["dec"]

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(fake_sources["amplitude"], fake_sources["y_mean"], fake_sources["x_mean"], stddev, stddev)

    data += g(xx, yy)

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        wcs=wcs,
        unit=u.Jy / u.beam,
        fake_sources=fake_sources,
        hits=np.ones_like(data, int),
    )

    cm.x = fake_sources["x_mean"]
    cm.y = fake_sources["y_mean"]

    return cm


@pytest.fixture()
def blended_sources():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    uncertainty = np.ones_like(data) / 4
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    fake_sources = Table(masked=True)
    fake_sources["fake_id"] = [1, 2]
    fake_sources["amplitude"] = [1, 1]
    fake_sources["x_mean"] = [13.6, 15.1]
    fake_sources["y_mean"] = [13.6, 15.1]

    ra, dec = wcs.wcs_pix2world(fake_sources["x_mean"], fake_sources["y_mean"], 0)
    fake_sources["ra"] = ra * u.deg
    fake_sources["dec"] = dec * u.deg

    xx, yy = np.indices(shape)
    stddev = 1 / pixsize * gaussian_fwhm_to_sigma
    g = models.Gaussian2D(
        fake_sources["amplitude"][0], fake_sources["y_mean"][0], fake_sources["x_mean"][0], stddev, stddev
    )
    for source in fake_sources[1:]:
        g += models.Gaussian2D(source["amplitude"], source["y_mean"], source["x_mean"], stddev, stddev)

    data += g(xx, yy)

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        wcs=wcs,
        unit=u.Jy / u.beam,
        fake_sources=fake_sources,
        hits=np.ones_like(data, int),
    )

    cm.x = fake_sources["x_mean"]
    cm.y = fake_sources["y_mean"]

    return cm


@pytest.fixture()
def single_source_mask():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    uncertainty = np.ones_like(data) / 4
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    xx, yy = np.indices(shape)
    mask = np.sqrt((xx - (shape[1] - 1) / 2) ** 2 + (yy - (shape[0] - 1) / 2) ** 2) > 10

    data[mask] = np.nan

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        mask=mask,
        wcs=wcs,
        unit=u.Jy / u.beam,
        hits=np.ones_like(data, int),
    )

    # Additionnal attribute just for the tests...
    cm.x = np.asarray([shape[1] / 2 - 0.5])
    cm.y = np.asarray([shape[0] / 2 - 0.5])
    cm.add_gaussian_sources(nsources=1, peak_flux=1 * u.Jy, within=(1 / 2, 1 / 2))
    return cm


@pytest.fixture()
def single_source_mask_edge():
    # Large shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (27, 27)
    pixsize = 1 / 3
    data = np.zeros(shape)
    uncertainty = np.ones_like(data) / 4
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

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        wcs=wcs,
        unit=u.Jy / u.beam,
        mask=mask,
        fake_sources=fake_sources,
        hits=np.ones_like(data, int),
    )

    cm.x = fake_sources["x_mean"]
    cm.y = fake_sources["y_mean"]

    return cm


@pytest.fixture()
def grid_sources():
    # Larger shape to allow for wobbling
    # as beam needs to be much smaller than the map at some point..
    # Shape was too small to allow for a proper background estimation
    # shape = (28, 28)
    shape = (60, 60)
    pixsize = 1 / 3
    data = np.zeros(shape)
    uncertainty = np.ones_like(data) / 4
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        wcs=wcs,
        unit=u.Jy / u.beam,
        hits=np.ones_like(data, int),
    )

    # Additionnal attribute just for the tests...
    cm.add_gaussian_sources(nsources=2**2, peak_flux=1 * u.Jy, cat_gen=pos_gridded, within=(1 / 4, 3 / 4))

    x, y = cm.wcs.wcs_world2pix(cm.fake_sources["ra"], cm.fake_sources["dec"], 0)

    cm.x = x
    cm.y = y

    return cm


@pytest.fixture()
def wobble_grid_sources():
    # Even Larger shape to allow for psf fitting
    # as beam needs to be much smaller than the map at some point..
    shape = (60, 60)
    pixsize = 1 / 3
    data = np.zeros(shape)
    uncertainty = np.ones_like(data) / 4
    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    cm = ContMap(
        data, uncertainty=StdDevUncertainty(uncertainty), wcs=wcs, unit=u.Jy / u.beam, hits=np.ones_like(data, int)
    )

    np.random.seed(0)
    # Additionnal attribute just for the tests...
    cm.add_gaussian_sources(nsources=2**2, peak_flux=1 * u.Jy, cat_gen=pos_gridded, wobble=True, wobble_frac=0.2)

    x, y = cm.wcs.wcs_world2pix(cm.fake_sources["ra"], cm.fake_sources["dec"], 0)

    cm.x = x
    cm.y = y

    return cm


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

    fake_sources = Table(masked=True)
    fake_sources["fake_id"] = np.arange(nsources) + 1
    fake_sources["amplitude"] = np.ones(nsources) * peak_flux
    fake_sources["x_mean"] = [shape[1] / 2]
    fake_sources["y_mean"] = [shape[0] / 2]

    ra, dec = wcs.wcs_pix2world(fake_sources["x_mean"], fake_sources["y_mean"], 0)
    fake_sources["ra"] = ra * u.deg
    fake_sources["dec"] = dec * u.deg

    fake_sources["_ra"] = fake_sources["ra"]
    fake_sources["_dec"] = fake_sources["dec"]

    beam_std_pix = (fwhm / pixsize).decompose().value * gaussian_fwhm_to_sigma
    fake_sources["x_stddev"] = np.ones(nsources) * beam_std_pix
    fake_sources["y_stddev"] = np.ones(nsources) * beam_std_pix
    fake_sources["theta"] = np.zeros(nsources)

    data = (
        make_model_image(shape, models.Gaussian2D(), fake_sources, model_shape=shape, x_name="x_mean", y_name="y_mean")
        * u.Jy
        / u.beam
    )

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level.to(u.Jy / u.beam)
    data += np.random.normal(loc=0, scale=1, size=shape) * uncertainty
    data[mask] = np.nan
    hits[mask] = 0
    uncertainty[mask] = 0

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"
    header["NOISE"] = noise_level.to_value(u.Jy / u.beam)

    cm = ContMap(
        data,
        uncertainty=StdDevUncertainty(uncertainty),
        hits=hits,
        mask=mask,
        wcs=wcs,
        unit=u.Jy / u.beam,
        fake_sources=fake_sources,
        meta=header,
    )

    cm.x = fake_sources["x_mean"]
    cm.y = fake_sources["y_mean"]

    return cm


@pytest.fixture()
def large_map_nosource():
    np.random.seed(0)

    shape = (256, 256)
    pixsize = 1 / 3 * u.deg
    noise_level = 0.1 * u.Jy / u.beam

    wcs = WCS()
    wcs.wcs.crpix = np.asarray(shape) / 2 - 0.5  # Center of pixel
    wcs.wcs.cdelt = np.asarray([-1, 1]) * pixsize
    wcs.wcs.ctype = ("RA---TAN", "DEC--TAN")

    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level.to(u.Jy / u.beam)
    data = np.random.normal(loc=0, scale=1, size=shape) * uncertainty

    header = wcs.to_header()
    header["UNIT"] = "Jy / beam", "Fake Unit"
    header["NOISE"] = noise_level.to_value(u.Jy / u.beam)

    cm = ContMap(data, uncertainty=StdDevUncertainty(uncertainty), hits=hits, wcs=wcs, unit=u.Jy / u.beam)

    return cm


# Special case to avoid detection, which would fail here (ShrinkMask)
def test_contmap_phot_mask_edge(single_source_mask_edge):
    cm = single_source_mask_edge
    cm.sources = cm.fake_sources
    cm.phot_sources(peak=True, psf=False)
    # Relative and absolute tolerance are really bad here for the case where
    # the sources are not centered on pixels... Otherwise it give perfect
    # answer when there is no noise
    npt.assert_allclose(cm.sources["flux_peak"].to(u.Jy).value, [1] * len(cm.sources), atol=1e-1, rtol=1e-1)

    cm.phot_sources(peak=False, psf=True)
    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(cm.sources["flux_psf"].to(u.Jy).value, [1] * len(cm.sources), rtol=1e-6)


@pytest.fixture(
    params=["single_source", "single_source_side", "single_source_mask", "grid_sources", "wobble_grid_sources"]
)
def cms(request):
    return getfixturevalue(request, request.param)


def test_contmap_trim(single_source_mask):
    cm = single_source_mask
    cm_trimed = cm.trim()
    assert cm_trimed.shape == (21, 21)

    assert np.any(cm_trimed.mask[0, :])
    assert np.any(cm_trimed.mask[-1, :])
    assert np.any(cm_trimed.mask[:, 0])
    assert np.any(cm_trimed.mask[:, -1])


def test_contmap_add_gaussian_sources(cms):
    cm = cms
    shape = cm.shape
    pixsize = np.abs(cm.wcs.wcs.cdelt[0])

    xx, yy = np.indices(shape)
    stddev = gaussian_fwhm_to_sigma / pixsize
    g = models.Gaussian2D(1, cm.y[0], cm.x[0], stddev, stddev)
    for item_x, item_y in zip(cm.y[1:], cm.x[1:]):
        g += models.Gaussian2D(1, item_x, item_y, stddev, stddev)

    if cm.mask is None:
        # atol for other sources
        npt.assert_allclose(cm.data, g(xx, yy))
    else:
        npt.assert_allclose(cm.data[~cm.mask], g(xx, yy)[~cm.mask])

    x, y = cm.wcs.wcs_world2pix(cm.fake_sources["ra"], cm.fake_sources["dec"], 0)
    # We are actually only testing the tolerance on x,y -> ra, dec -> x, y
    npt.assert_allclose([x, y], [cm.x, cm.y], atol=1e-13)


def test_contmap_detect_sources(cms):
    cm = cms
    cm.detect_sources()

    ordering = cm.fake_sources["find_peak"]

    npt.assert_allclose(cm.fake_sources["ra"], cm.sources["ra"][ordering])
    npt.assert_allclose(cm.fake_sources["dec"], cm.sources["dec"][ordering])

    # When sources are exactly at the center of 4 pixels the basic peak finder will fail
    if len(cms.fake_sources) != 4:
        npt.assert_allclose(cm.sources["SNR"], [4] * len(cm.sources))
    else:
        npt.assert_allclose(cm.sources["SNR"], [4] * len(cm.sources), atol=0.6)

    x_fake, y_fake = cm.wcs.wcs_world2pix(cm.fake_sources["ra"], cm.fake_sources["dec"], 0)
    x, y = cm.wcs.wcs_world2pix(cm.sources["ra"], cm.sources["dec"], 0)

    # Tolerance coming from round wcs transformations
    npt.assert_allclose(x_fake, x[ordering], atol=1e-11)
    npt.assert_allclose(y_fake, y[ordering], atol=1e-11)

    # Fake empy data to fake no found sources
    cm._data *= 0
    cm.detect_sources()
    assert cm.sources is None
    assert np.all(cm.fake_sources["find_peak"].mask)


def test_contmap_phot_sources(cms):
    cm = cms

    # Beam area
    beam_stddev_pix = cm.beam.major.to("pix", cm._pixel_scale).decompose().value * gaussian_fwhm_to_sigma
    beam_sqr_area_pix = np.pi * beam_stddev_pix**2

    cm.detect_sources()

    cm.phot_sources(peak=True, psf=False)
    # Relative and absolute tolerance are really bad here for the case where
    # the sources are not centered on pixels... Otherwise it give perfect
    # answer when there is no noise
    npt.assert_allclose(cm.sources["flux_peak"].to(u.Jy).value, [1] * len(cm.sources), atol=1e-1, rtol=1e-1)
    npt.assert_allclose(cm.sources["eflux_peak"].to(u.Jy).value, [1 / 4] * len(cm.sources), atol=1e-1, rtol=1e-1)

    cm.phot_sources(peak=False, psf=True)
    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(cm.sources["flux_psf"].to(u.Jy).value, [1] * len(cm.sources), rtol=1e-6)
    npt.assert_allclose(
        cm.sources["eflux_psf"].to(u.Jy).value,
        [1 / 4 / np.sqrt(beam_sqr_area_pix)] * len(cm.sources),
        atol=1e-1,
        rtol=1e-1,
    )

    # Without background estimation
    cm.phot_sources(peak=False, psf=True, background=False)
    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(cm.sources["flux_psf"].to(u.Jy).value, [1] * len(cm.sources), rtol=1e-6)
    npt.assert_allclose(
        cm.sources["eflux_psf"].to(u.Jy).value,
        [1 / 4 / np.sqrt(beam_sqr_area_pix) * np.sqrt(2)] * len(cm.sources),
        atol=1e-1,
        rtol=1e-1,
    )

    # Without fixed positions
    cm.phot_sources(peak=False, psf=True, fixed_positions=False)
    assert "x_fit" in cm.sources.colnames
    assert "y_fit" in cm.sources.colnames
    npt.assert_allclose(cm.sources["x_fit"], cm.sources["x_centroid"], atol=1e-10)
    npt.assert_allclose(cm.sources["y_fit"], cm.sources["y_centroid"], atol=1e-10)

    # Without fixed sigma
    cm.phot_sources(peak=False, psf=True, fixed_sigma=False)
    assert "fwhm_fit" in cm.sources.colnames
    # High tolerance for the grid sources...
    npt.assert_allclose(cm.sources["fwhm_fit"].to("deg").value, cm.beam._major.to("deg").value, atol=1e-5)
    npt.assert_allclose(cm.sources["fwhm_fit"].to("deg").value, cm.beam._minor.to("deg").value, atol=1e-5)

    ordering = cm.fake_sources["find_peak"]

    # Relative tolerance is rather low to pass the case of multiple sources...
    npt.assert_allclose(cm.fake_sources["ra"], cm.sources["ra"][ordering], rtol=1e-6)
    npt.assert_allclose(cm.fake_sources["dec"], cm.sources["dec"][ordering], rtol=1e-6)
    npt.assert_allclose(cm.sources["flux_psf"].to(u.Jy).value, [1] * len(cm.sources), rtol=1e-5)


def test_contmap_match_filter(cms):
    cm = cms
    mf_cm = cm.match_filter(cm.beam)

    # Beam area
    beam_stddev_pix = cm.beam.major.to("pix", cm._pixel_scale).value * gaussian_fwhm_to_sigma
    beam_sqr_area_pix = np.pi * beam_stddev_pix**2

    x_idx = np.floor(cm.x + 0.5).astype(int)
    y_idx = np.floor(cm.y + 0.5).astype(int)

    npt.assert_allclose(mf_cm.data[y_idx, x_idx], cm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    npt.assert_allclose(
        mf_cm.uncertainty.array[y_idx, x_idx],
        cm.uncertainty.array[y_idx, x_idx] / np.sqrt(beam_sqr_area_pix),
        atol=1e-2,
        rtol=1e-1,
    )
    npt.assert_allclose((cm.beam.major * np.sqrt(2)).to(u.arcsec), mf_cm.beam.major.to(u.arcsec))

    hit_factor = (
        cm.beam.major / cm.beam.pixscale * gaussian_fwhm_to_sigma
    ) ** 2 * np.pi  # as it scale as the kernel_sqr (/2 wrt to gaussian size)
    npt.assert_allclose(
        np.median(mf_cm.hits[mf_cm.hits != 0]), np.median(cm.hits[cm.hits != 0]) * hit_factor, atol=1e-2, rtol=1e-1
    )
    with pytest.warns(UserWarning):
        mh_cm = cm.match_filter(
            RickerWavelet2DKernel(cm.beam.major.to(u.pix, cm._pixel_scale).value * gaussian_fwhm_to_sigma)
        )
    npt.assert_allclose(mh_cm.data[y_idx, x_idx], cm.data[y_idx, x_idx], atol=1e-2, rtol=1e-1)
    assert mh_cm.beam.major is None


def wip_test_contmap_correlated_noise(cms):
    cm = cms

    sigma_beam_pix = cm.beam.major.to_value("pix", cm._pixel_scale) * gaussian_fwhm_to_sigma

    r = np.linspace(-25, 25, 51)
    yy, xx = np.meshgrid(r, r)
    rr = np.sqrt(xx * xx + yy * yy)
    kernel = np.sinc(rr / np.pi)

    # fwhm 1 / 2 beam
    r = np.linspace(-10, 10, 21)
    yy, xx = np.meshgrid(r, r)
    rr = np.sqrt(xx * xx + yy * yy)
    sigma_pix = sigma_beam_pix / 2
    kernel = np.exp(-(rr**2) / (2 * sigma_pix**2))
    kernel = Kernel2D(array=kernel)
    kernel = Gaussian2DKernel(x_stddev=sigma_pix)

    # # fwhm = 1 beam
    sigma_pix = sigma_beam_pix
    kernel = np.exp(-(rr**2) / (2 * sigma_pix**2))
    kernel = Kernel2D(array=kernel)
    kernel = Gaussian2DKernel(x_stddev=sigma_pix)

    # Eq 37 from Condon 1997
    int_C = 2 * np.pi * sigma_pix**2
    int_C2 = np.pi * sigma_pix**2
    expected_std = np.sqrt(int_C2 / int_C**2) * cm.uncertainty.represent_as(StdDevUncertainty).array

    from astropy.convolution import convolve

    test_data = np.asanyarray(cm).filled(0)
    corr_data = convolve(test_data, kernel, boundary="wrap", normalize_kernel=True)

    # If we want to have the same peak / beam intensity we need to correct for the different beam area
    beam_corr = 1 + (sigma_pix / sigma_beam_pix) ** 2
    corr_data *= beam_corr
    expected_std *= beam_corr

    # match filter, works ONLY for kernel == beam, this is not satisfactory....
    kernel.normalize("peak")
    kernel_sqr = kernel.array**2

    weights = cm.uncertainty.represent_as(InverseVariance).array

    mf_uncertainty = 1 / np.sqrt(convolve(weights, kernel_sqr, boundary="wrap", normalize_kernel=False))
    convolve(weights * test_data, kernel, boundary="wrap", normalize_kernel=False) * mf_uncertainty**2
    npt.assert_allclose(mf_uncertainty, expected_std, rtol=1e-6)

    mf_cm = cm.match_filter(kernel)
    test_mf_cm = prototype_match_filter_corrected(cm, kernel)

    plt.clf()
    plt.plot(test_data[128])
    plt.plot(corr_data[128])
    plt.plot(mf_cm.data[128])
    plt.plot(test_mf_cm.data[128])

    # Not working
    # # Another way...
    # kernel.normalize("integral")
    # kernel_sqr = Kernel2D(array=kernel.array**2)
    # weights = cm.uncertainty.represent_as(InverseVariance).array

    # mf_uncertainty = 1 / np.sqrt(convolve(weights, kernel_sqr, boundary="wrap", normalize_kernel=True))
    # mf_data = convolve(weights * test_data, kernel, boundary="wrap", normalize_kernel=True) * mf_uncertainty**2

    # mf_corr = 1 + kernel._array.sum() / cm.beam._array.sum().value
    # mf_uncertainty *= mf_corr
    # mf_data *= mf_corr

    # WIP....
    corr_data *= 0

    cm_corr = cm.match_filter(Kernel2D(array=kernel))
    del cm_corr

    # Additional analytical checks at the end of the test bench.
    x_idx = np.floor(cm.x + 0.5).astype(int)
    y_idx = np.floor(cm.y + 0.5).astype(int)

    # For peak-normalized Gaussians and uniform weights:
    # MF_peak / input_peak = 2 * sigma_b^2 / (sigma_b^2 + sigma_k^2)
    for sigma_factor in (1.0, 0.5):
        sigma_pix = sigma_beam_pix * sigma_factor
        kernel_check = Gaussian2DKernel(sigma_pix)
        mf_cm = cm.match_filter(kernel_check)

        int_c = 2 * np.pi * sigma_pix**2
        int_c2 = np.pi * sigma_pix**2
        expected_std = np.sqrt(int_c2 / int_c**2) * cm.uncertainty.represent_as(StdDevUncertainty).array
        npt.assert_allclose(
            mf_cm.uncertainty.array[y_idx, x_idx],
            expected_std[y_idx, x_idx],
            atol=1e-2,
            rtol=2e-1,
        )

        measured_peak_ratio = np.nacmedian(mf_cm.data[y_idx, x_idx] / cm.data[y_idx, x_idx])
        expected_peak_ratio = 2 * sigma_beam_pix**2 / (sigma_beam_pix**2 + sigma_pix**2)
        npt.assert_allclose(measured_peak_ratio, expected_peak_ratio, atol=2e-2, rtol=8e-2)

        flux_correction = (sigma_beam_pix**2 + sigma_pix**2) / (2 * sigma_beam_pix**2)
        npt.assert_allclose(measured_peak_ratio * flux_correction, 1.0, atol=2e-2, rtol=8e-2)


def test_contmap_match_sources(cms):
    cm = cms
    cm.detect_sources()
    sources = cm.sources
    sources.meta["name"] = "to_match"
    cm.match_sources(sources)

    assert np.all(cm.sources["ID"] == cm.sources["to_match"])


def test_contmap_match_sources_threshold(cms):
    cm = cms
    cm.detect_sources()
    sources = cm.sources
    sources.meta["name"] = "to_match"
    cm.match_sources(sources, dist_threshold=cm.beam.major)

    assert np.all(cm.sources["ID"] == cm.sources["to_match"])


def test_contmap_match_sources_list(cms):
    cm = cms
    cm.detect_sources()
    sources = cm.sources.copy()
    sources.meta["name"] = "to_match_1"
    sources2 = cm.sources.copy()
    sources2.meta["name"] = "to_match_2"

    cm.match_sources([sources, sources2])

    assert np.all(cm.sources["ID"] == cm.sources["to_match_1"])
    assert np.all(cm.sources["ID"] == cm.sources["to_match_2"])


# Different Freetype version on travis... 2.8.0 vs 2.6.1 -> tolerance 20
# Different Freetype version on circleci... 2.12.1 vs 2.6.1 -> tolerance 21
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=21)
def test_contmap_plot(cms):
    cm = cms
    cax = cm.plot()

    return cax.get_figure()


# Different Freetype version on travis... 2.8.0 vs 2.6.1 -> tolerance 20
# Different Freetype version on circleci... 2.12.1 vs 2.6.1 -> tolerance 21
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=21)
def test_contmap_plot_SNR(cms):
    cm = cms
    cax = cm.plot_SNR(cbar=True)

    return cax.get_figure()


# Different Freetype version on travis... 2.8.0 vs 2.6.1 -> tolerance 20
# Different Freetype version on circleci... 2.12.1 vs 2.6.1 -> tolerance 21
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=21)
def test_contmap_plot_beam(cms):
    cm = cms
    cax = cm.plot(beam=True)

    return cax.get_figure()


# Different Freetype version on travis... 2.8.0 vs 2.6.1 -> tolerance 20
# Different Freetype version on circleci... 2.12.1 vs 2.6.1 -> tolerance 21
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=21)
def test_contmap_plot_ax(cms):
    cm = cms
    fig, axes = plt.subplots(nrows=2, ncols=2, subplot_kw={"projection": cm.wcs})
    axes = axes.flatten()
    cm.plot(ax=axes[0], vmin=-1, vmax=3)
    cm.plot(ax=axes[1], levels=np.logspace(np.log10(0.1), np.log10(5), 5))
    cm.plot(ax=axes[2], cat=[(cm.fake_sources, {"marker": "+", "color": "red"})])
    cm.fake_sources = None
    cm.detect_sources()
    cm.plot(ax=axes[3], cat=True)

    for ax in axes:
        ax.legend(loc="best", frameon=False)

    return fig


# Different Freetype version on travis... 2.8.0 vs 2.6.1 -> tolerance 20
# Different Freetype version on circleci... 2.12.1 vs 2.6.1 -> tolerance 21
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=21)
def test_contmap_plot_PSD(cms):
    pytest.importorskip("powspec")

    cm = cms
    fig, axes = plt.subplots(nrows=4, sharex=True)
    cm.plot_PSD(ax=axes[0])
    cm.plot_PSD(ax=axes[1], apod_size=5)
    cm.plot_PSD(ax=axes[2], bins=50)
    cm.plot_PSD(ax=axes[3], to_plot="snr")

    powspec, bins = cm.plot_PSD()

    return fig


def test_contmap_check_SNR(large_map_source):
    cm = large_map_source

    std = cm.check_SNR()
    # Tolerance comes from the fact that we biased the result using the SNR
    # cut for the fit
    npt.assert_allclose(std, 1, rtol=1e-2)

    std, mu = cm.check_SNR(return_mean=True)
    npt.assert_allclose(std, 1, rtol=1e-2)
    npt.assert_allclose(mu, 0, atol=1e-2)


# Different Freetype version on travis... 2.8.0 vs 2.6.1
@pytest.mark.mpl_image_compare(remove_text=True, tolerance=20)
def test_contmap_check_SNR_ax(large_map_source):
    cm = large_map_source

    fig, ax = plt.subplots()
    _ = cm.check_SNR(ax=ax)

    return fig


def test_blended_sources(blended_sources):
    cm = blended_sources
    cm.detect_sources()
    cm.phot_sources()

    # Cannot recover all sources :
    assert len(cm.sources) != len(cm.fake_sources)

    # But still prior photometry can recover the flux
    cm.phot_sources(cm.fake_sources)
    npt.assert_allclose(cm.fake_sources["flux_psf"].to(u.Jy).value, [1] * len(cm.fake_sources))


def test_get_square_slice(single_source_mask):
    cm = single_source_mask
    islice = cm.get_square_slice()

    radius = 10
    assert np.floor(np.sqrt(2) * radius) == islice.stop - islice.start - 1
    assert np.floor(cm.shape[0] / 2 - np.sqrt(2) * radius / 2) == islice.start
    assert np.floor(cm.shape[0] / 2 + np.sqrt(2) * radius / 2 + 1) == islice.stop


def test_get_square_slice_start(single_source_mask):
    cm = single_source_mask

    with pytest.raises(AssertionError):
        islice = cm.get_square_slice(start=14)
        islice = cm.get_square_slice(start=[14, 14, 14])

    islice = cm.get_square_slice(start=(14, 14))

    radius = 10
    assert np.floor(np.sqrt(2) * radius) == islice.stop - islice.start - 1
    assert np.floor(cm.shape[0] / 2 - np.sqrt(2) * radius / 2) == islice.start
    assert np.floor(cm.shape[0] / 2 + np.sqrt(2) * radius / 2 + 1) == islice.stop


def test_surface():
    shape = (2, 2)
    data = np.ones(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    wcs = WCS()
    wcs.wcs.cdelt = np.array([-2 / 60**2, 2 / 60**2])
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]

    cm = ContMap(data=data, mask=mask, wcs=wcs, meta={"BMAJ": 1 / 3600})
    surface = cm.surface()
    assert np.isclose(surface.to_value(u.arcsec**2), 8)


def test_surface_shrink():
    shape = (5, 5)
    data = np.ones(shape)
    mask = np.ones(shape, dtype=bool)
    mask[1:-1, 1:-1] = False

    wcs = WCS()
    wcs.wcs.cdelt = np.array([-2 / 60**2, 2 / 60**2])
    wcs.wcs.ctype = ["RA---AIR", "DEC--AIR"]

    cm = ContMap(data=data, mask=mask, wcs=wcs, meta={"BMAJ": 1 / 3600})
    surface = cm.surface(box_size=1.001)
    assert np.isclose(surface.to_value(u.arcsec**2), 4)


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

    data = make_model_image(shape, models.Gaussian2D(), sources, model_shape=shape, x_name="x_mean", y_name="y_mean")
    hits = np.ones(shape=shape, dtype=float)
    uncertainty = np.ones(shape, dtype=float) * noise_level.to_value(u.Jy / u.beam)
    data += np.random.normal(loc=0, scale=1, size=shape) * uncertainty
    data[mask] = np.nan
    hits[mask] = 0
    uncertainty[mask] = 0

    header = wcs.to_header()
    header["BUNIT"] = "Jy / beam", "Fake Unit"
    header["NOISE"] = noise_level.to_value(u.Jy / u.beam)

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
