from __future__ import absolute_import, division, print_function

import numpy as np
import numpy.testing as npt
import pytest
from astropy.convolution import Gaussian2DKernel
from astropy.modeling import models
from astropy.table import Table
from photutils.datasets import make_model_image

from ..utils import (
    cat_to_sc,
    fake_data,
    fft_2d_hanning,
    make_kernel_image,
    meta_to_header,
    pos_gridded,
    pos_in_mask,
    pos_list,
    pos_uniform,
    pos_uniform_no_overlap,
    shrink_mask,
)


def test_shrink_mask():
    kernel_size = 3
    mask_size = 48

    mask = np.ones((2 * mask_size, 2 * mask_size), bool)
    center_slice = slice(mask_size - mask_size // 3, mask_size + mask_size // 3)
    mask[center_slice, center_slice] = False

    result = np.ones((2 * mask_size, 2 * mask_size), bool)
    center_slice = slice(mask_size - mask_size // 3 + kernel_size, mask_size + mask_size // 3 - kernel_size)
    result[center_slice, center_slice] = False

    xx = np.arange(2 * kernel_size + 1) - kernel_size
    kernel = np.exp(-(xx**2 + xx[:, np.newaxis] ** 2) / 2)
    kernel /= kernel.sum()

    shrinked_mask = shrink_mask(mask, kernel)

    assert np.all(result == shrinked_mask)


def test_pos_in_mask():
    mask = np.asarray([[True, False], [False, False]])
    pos = [[0, 0], [0.5, 0.5], [1, 1]]

    result = pos_in_mask(pos)
    npt.assert_equal(result, pos)

    result = pos_in_mask(pos, mask)
    npt.assert_equal(result, pos[1:])


def test_cat_to_sc():
    cat = Table(data=[[0, 1], [0, 1]], names=["ra", "dec"], dtype=[float, float])
    cat["ra"].unit = "deg"
    cat["dec"].unit = "deg"
    coords = cat_to_sc(cat)
    npt.assert_equal(coords.ra.deg, cat["ra"].data)
    npt.assert_equal(coords.dec.deg, cat["dec"].data)

    cat["_ra"] = cat["ra"] * 2
    cat["_dec"] = cat["dec"] * 2

    # _ra/_dec superseed ra/dec
    coords = cat_to_sc(cat)
    npt.assert_equal(coords.ra.deg, cat["_ra"].data)
    npt.assert_equal(coords.dec.deg, cat["_dec"].data)


def test_pos_uniform():
    np.random.seed(0)
    shape = (5, 20)
    x, y = pos_uniform(nsources=100, shape=shape)
    assert -0.5 < x.min() or x.max() < shape[1] - 0.5, "pixel coordinate outside boundaries"
    assert -0.5 < y.min() or y.max() < shape[0] - 0.5, "pixel coordinate outside boundaries"

    mask = np.zeros(shape, dtype=bool)
    mask[:, :5] = True
    x, y = pos_uniform(nsources=100, shape=shape, mask=mask)
    assert 4 < np.floor(x.min() + 0.5), "pixel coordinate inside max"

    x, y = pos_uniform(nsources=100, shape=shape, within=(0.5, 1))
    assert shape[1] * 0.5 - 1 < np.floor(x.min() + 0.5), "pixel coordinate outside within"
    assert shape[0] * 0.5 - 1 < np.floor(y.min() + 0.5), "pixel coordinate outside within"

    x, y = pos_uniform(nsources=30, shape=shape, dist_threshold=1)
    dist = np.sqrt((x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2)
    i = np.arange(len(x))
    dist[i, i] = np.inf
    assert np.all(np.min(dist, 1) > 1), "sources too close"

    with pytest.warns(UserWarning):
        x, y = pos_uniform(nsources=100, shape=shape, dist_threshold=1)


def test_pos_gridded():
    shape = (9, 21)
    x, y = pos_gridded(nsources=3**2, shape=shape)
    assert np.all(
        x.reshape(3, 3) == np.linspace(shape[1] / 4, shape[1] * 3 / 4, 3) - 0.5
    ), "unexpected pixel coordinate"
    assert np.all(
        y.reshape(3, 3).T == np.linspace(shape[0] / 4, shape[0] * 3 / 4, 3) - 0.5
    ), "unexpected pixel coordinate"

    mask = np.zeros(shape, dtype=bool)
    mask[:, :5] = True
    with pytest.warns(UserWarning):
        x, y = pos_gridded(nsources=10**2, shape=shape, mask=mask)
    assert 4 < np.floor(x.min() + 0.5), "pixel coordinate inside max"

    with pytest.raises(AssertionError):
        x, y = pos_gridded(nsources=1)

    with pytest.raises(AssertionError):
        x, y = pos_gridded(nsources=2)

    np.random.seed(26)
    # This can raise an exception
    with pytest.warns(UserWarning):
        x, y = pos_gridded(nsources=3**2, shape=shape, wobble=True)

    x, y = pos_gridded(nsources=3**2, shape=shape, wobble=True, within=(1 / 3, 2 / 3))
    assert -0.5 < x.min() or x.max() < shape[1] - 0.5, "pixel coordinate outside boundaries"
    assert -0.5 < y.min() or y.max() < shape[0] - 0.5, "pixel coordinate outside boundaries"


def test_pos_uniform_no_overlap():
    np.random.seed(42)
    shape = (64, 64)

    x, y = pos_uniform_no_overlap(nsources=20, shape=shape, within=(1 / 8, 7 / 8), dist_threshold=4, oversample=5)
    assert x.shape == (20,)
    assert y.shape == (20,)

    dist = np.sqrt((x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2)
    i = np.arange(len(x))
    dist[i, i] = np.inf
    assert np.all(np.min(dist, 1) > 4), "sources too close"


def test_pos_uniform_no_overlap_warns_when_overconstrained():
    np.random.seed(0)
    shape = (5, 5)

    with pytest.warns(UserWarning):
        x, y = pos_uniform_no_overlap(
            nsources=20,
            shape=shape,
            within=(0.4, 0.6),
            dist_threshold=3,
            max_loop=2,
        )

    assert len(x) < 20
    assert len(y) < 20


def test_pos_list():
    shape = (5, 20)
    nsources = 20
    x_mean = np.linspace(0, 19, nsources)
    y_mean = np.ones(nsources) * 2.5

    with pytest.raises(AssertionError):
        x, y = pos_list(nsources=nsources, shape=shape)

    with pytest.raises(AssertionError):
        x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean[:-2])

    x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean)
    assert np.all(x == x_mean), "should be identical"
    assert np.all(y == y_mean), "should be identical"

    mask = np.zeros(shape, dtype=bool)
    mask[:, :5] = True
    with pytest.warns(UserWarning):
        x, y = pos_list(nsources=nsources, shape=shape, mask=mask, x_mean=x_mean, y_mean=y_mean)

    assert np.all(x == x_mean[5:]), "should be identical"
    assert np.all(y == y_mean[5:]), "should be identical"

    with pytest.warns(UserWarning):
        x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean, within=(1 / 4, 3 / 4))

    assert np.all(x == x_mean[shape[1] // 4 : shape[1] * 3 // 4]), "should be identical"  # noqa: E203

    x_mean = np.linspace(-1, 18, nsources)

    with pytest.warns(UserWarning):
        x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean)

    assert np.all(x == x_mean[1:]), "should be identical"


def test_fft_2d_hanning_assertion():
    shape = 5
    mask = np.ones((shape, shape), dtype=bool)

    with pytest.raises(AssertionError):
        fft_2d_hanning(mask, size=2)

    with pytest.raises(AssertionError):
        fft_2d_hanning(mask, size=1)


def test_fft_2d_hanning():
    # Min hann filter is 5 x 5
    shape = 15
    size = 5
    apod_size = 2

    mask = np.ones((shape, shape), dtype=bool)
    islice = slice(((shape - 1) - (size - 1)) // 2, ((shape - 1) + (size - 1)) // 2 + 1)
    mask[islice, islice] = False

    apod = fft_2d_hanning(mask, size=apod_size)
    # Nothing outside the mask
    assert np.all((apod > 1e-15) == ~mask)
    # Only unchanged pixel at the center
    unchanged = slice(
        ((shape - 1) - (size - 1) + (apod_size * 2 + 1 - 1)) // 2,
        ((shape - 1) + (size - 1) - (apod_size * 2 + 1 - 1)) // 2 + 1,
    )
    npt.assert_allclose(apod[unchanged, unchanged], 1)

    shape = 45
    size = 15
    apod_size = 2

    mask = np.ones((shape, shape), dtype=bool)
    islice = slice(((shape - 1) - (size - 1)) // 2, ((shape - 1) + (size - 1)) // 2 + 1)
    mask[islice, islice] = False
    apod = fft_2d_hanning(mask, size=2)

    assert np.all((apod > 1e-15) == ~mask)
    unchanged = slice(
        ((shape - 1) - (size - 1) + (apod_size * 2 + 1 - 1)) // 2,
        ((shape - 1) + (size - 1) - (apod_size * 2 + 1 - 1)) // 2 + 1,
    )
    npt.assert_allclose(apod[unchanged, unchanged], 1)


def P(k, alpha=-11.0 / 3, fknee=1):
    """Simple power law formula"""
    return (k / fknee) ** alpha


def gen_pkfield(npix=32, alpha=-11.0 / 3, fknee=1, res=1):
    """Generate a 2D square map with P(k) field"""

    ufreq = np.fft.fftfreq(npix, d=res)
    kfreq = np.sqrt(ufreq[:, np.newaxis] ** 2 + ufreq**2)

    with np.errstate(divide="ignore"):
        psd = 2 * P(kfreq, alpha=alpha, fknee=fknee)
    psd[0, 0] = 0

    pha = np.random.uniform(low=-np.pi, high=np.pi, size=(npix, npix))

    fft_img = np.sqrt(psd) * (np.cos(pha) + 1j * np.sin(pha))
    return np.real(np.fft.ifft2(fft_img)) * npix / res**2


def test_fake_data():
    # Dummy test for now
    _ = fake_data()


def test_meta_to_header():
    meta = {"toto": "tata"}

    hdr = meta_to_header(meta)
    assert hdr["toto"] == meta["toto"]

    meta["history"] = ["first", "second"]
    hdr = meta_to_header(meta)
    assert list(hdr["history"]) == meta["history"]

    meta["comment"] = ["first", "second"]
    hdr = meta_to_header(meta)
    assert list(hdr["comment"]) == meta["comment"]

    meta["array"] = [1, 2, 3]
    hdr = meta_to_header(meta)
    assert hdr["array"] == "[1, 2, 3]"

    meta["way to long key"] = "toto"
    hdr = meta_to_header(meta)
    assert hdr["way to long key"] == "toto"


def test_make_kernel_image_chunked_matches_photutils_make_model_image():
    np.random.seed(1)
    shape = (48, 64)
    nsrc = 60
    sigma = 1.2
    sources = Table()
    sources["x_mean"] = np.random.uniform(0, shape[1] - 1, nsrc)
    sources["y_mean"] = np.random.uniform(0, shape[0] - 1, nsrc)
    sources["amplitude"] = np.random.normal(0.0, 1.0, nsrc)
    sources["x_stddev"] = np.full(nsrc, sigma)
    sources["y_stddev"] = np.full(nsrc, sigma)
    sources["theta"] = np.zeros(nsrc)

    kernel = Gaussian2DKernel(sigma)
    model_shape = kernel.array.shape

    img_photutils = make_model_image(
        shape,
        models.Gaussian2D(),
        sources,
        model_shape=model_shape,
        x_name="x_mean",
        y_name="y_mean",
    )

    img_chunked = make_kernel_image(shape, kernel, sources)

    npt.assert_allclose(img_chunked, img_photutils, rtol=0.0, atol=1e-3)

    # A tiny memory fraction ensures the chunked path is exercised.
    img_chunked = make_kernel_image(shape, kernel, sources, memory_fraction=1e-12)

    npt.assert_allclose(img_chunked, img_photutils, rtol=0.0, atol=1e-3)
