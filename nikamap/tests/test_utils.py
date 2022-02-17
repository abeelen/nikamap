from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
import numpy.testing as npt

from ..utils import pos_in_mask, cat_to_sc
from ..utils import pos_uniform, pos_gridded, pos_list
from ..utils import fft_2d_hanning, powspec_k
from ..utils import fake_data
from ..utils import shrink_mask
from ..utils import meta_to_header

from astropy.table import Table
import astropy.units as u


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
    kernel = np.exp(-(xx ** 2 + xx[:, np.newaxis] ** 2) / 2)
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
    x, y, f = pos_uniform(nsources=100, shape=shape)
    assert -0.5 < x.min() or x.max() < shape[1] - 0.5, "pixel coordinate outside boundaries"
    assert -0.5 < y.min() or y.max() < shape[0] - 0.5, "pixel coordinate outside boundaries"
    assert np.all(f == np.repeat(1 * u.mJy, f.shape))

    mask = np.zeros(shape, dtype=bool)
    mask[:, :5] = True
    x, y, f = pos_uniform(nsources=100, shape=shape, mask=mask)
    assert 4 < np.floor(x.min() + 0.5), "pixel coordinate inside max"

    x, y, f = pos_uniform(nsources=100, shape=shape, within=(0.5, 1))
    assert shape[1] * 0.5 - 1 < np.floor(x.min() + 0.5), "pixel coordinate outside within"
    assert shape[0] * 0.5 - 1 < np.floor(y.min() + 0.5), "pixel coordinate outside within"

    x, y, f = pos_uniform(nsources=30, shape=shape, dist_threshold=1)
    dist = np.sqrt((x[:, np.newaxis] - x) ** 2 + (y[:, np.newaxis] - y) ** 2)
    i = np.arange(len(x))
    dist[i, i] = np.inf
    assert np.all(np.min(dist, 1) > 1), "sources too close"

    with pytest.warns(UserWarning):
        x, y, f = pos_uniform(nsources=100, shape=shape, dist_threshold=1)


def test_pos_gridded():

    shape = (9, 21)
    x, y, f = pos_gridded(nsources=3 ** 2, shape=shape)
    assert np.all(
        x.reshape(3, 3) == np.linspace(shape[1] / 4, shape[1] * 3 / 4, 3) - 0.5
    ), "unexpected pixel coordinate"
    assert np.all(
        y.reshape(3, 3).T == np.linspace(shape[0] / 4, shape[0] * 3 / 4, 3) - 0.5
    ), "unexpected pixel coordinate"
    assert np.all(f == np.repeat(1 * u.mJy, f.shape))

    mask = np.zeros(shape, dtype=bool)
    mask[:, :5] = True
    with pytest.warns(UserWarning):
        x, y, f = pos_gridded(nsources=10 ** 2, shape=shape, mask=mask)
    assert 4 < np.floor(x.min() + 0.5), "pixel coordinate inside max"

    with pytest.raises(AssertionError):
        x, y, f = pos_gridded(nsources=1)

    with pytest.raises(AssertionError):
        x, y, f = pos_gridded(nsources=2)

    np.random.seed(26)
    # This can raise an exception
    with pytest.warns(UserWarning):
        x, y, f = pos_gridded(nsources=3 ** 2, shape=shape, wobble=True)

    x, y, f = pos_gridded(nsources=3 ** 2, shape=shape, wobble=True, within=(1 / 3, 2 / 3))
    assert -0.5 < x.min() or x.max() < shape[1] - 0.5, "pixel coordinate outside boundaries"
    assert -0.5 < y.min() or y.max() < shape[0] - 0.5, "pixel coordinate outside boundaries"


def test_pos_list():

    shape = (5, 20)
    nsources = 20
    x_mean = np.linspace(0, 19, nsources)
    y_mean = np.ones(nsources) * 2.5

    with pytest.raises(AssertionError):
        x, y, f = pos_list(nsources=nsources, shape=shape)

    with pytest.raises(AssertionError):
        x, y, f = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean[:-2])

    x, y, f = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean)
    assert np.all(x == x_mean), "should be identical"
    assert np.all(y == y_mean), "should be identical"

    mask = np.zeros(shape, dtype=bool)
    mask[:, :5] = True
    with pytest.warns(UserWarning):
        x, y, f = pos_list(nsources=nsources, shape=shape, mask=mask, x_mean=x_mean, y_mean=y_mean)

    assert np.all(x == x_mean[5:]), "should be identical"
    assert np.all(y == y_mean[5:]), "should be identical"

    with pytest.warns(UserWarning):
        x, y, f = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean, within=(1 / 4, 3 / 4))

    assert np.all(x == x_mean[shape[1] // 4 : shape[1] * 3 // 4]), "should be identical"  # noqa: E203

    x_mean = np.linspace(-1, 18, nsources)

    with pytest.warns(UserWarning):
        x, y, f = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean)

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
    kfreq = np.sqrt(ufreq[:, np.newaxis] ** 2 + ufreq ** 2)

    with np.errstate(divide="ignore"):
        psd = 2 * P(kfreq, alpha=alpha, fknee=fknee)
    psd[0, 0] = 0

    pha = np.random.uniform(low=-np.pi, high=np.pi, size=(npix, npix))

    fft_img = np.sqrt(psd) * (np.cos(pha) + 1j * np.sin(pha))
    return np.real(np.fft.ifft2(fft_img)) * npix / res ** 2


def test_powspec_k():

    npix = 128
    res = 50
    img = gen_pkfield(npix=npix, res=res)
    powspec, bin_edges = powspec_k(img, res=res, bins=npix)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    def gen_pk(npix, res):
        img = gen_pkfield(npix=npix, res=res)
        powspec, _ = powspec_k(img, res=res, bins=npix)
        return powspec

    realization = list(map(lambda i: gen_pk(npix, res), range(100)))
    mean_Pk = np.mean(realization, axis=0)
    std_Pk = np.std(realization, axis=0)

    # plt.close('all')
    # plt.loglog(bin_centers[1:], mean_Pk[1:])
    # plt.loglog(bin_centers[1:], mean_Pk[1:]+std_Pk[1:])
    # plt.loglog(bin_centers[1:], mean_Pk[1:]-std_Pk[1:])
    # plt.loglog(bin_centers, P(bin_centers) / res**2)

    assert np.all((mean_Pk[1:] - P(bin_centers[1:])) < std_Pk[1:])


def test_powspec_k_unit():

    npix = 1024
    nsub = 128
    alpha = -1  # For alpha=-3, the P(k) is dominated by the step edges...
    res = 2 * u.arcsec

    np.random.seed(1)

    img = gen_pkfield(npix=npix, res=res, alpha=alpha, fknee=1 / u.arcsec) * u.Jy

    with pytest.raises(AssertionError):
        dummy = powspec_k(img, res=res, range=(0, 1))

    bins = np.linspace(2, nsub // 2, nsub // 2 - 2) / (res * nsub)
    powspec_full, bin_full = powspec_k(img, res=res, bins=bins)

    bin_centers = (bin_full[1:] + bin_full[:-1]) / 2

    powspecs = u.Quantity(
        [
            powspec_k(img[i : i + nsub, j : j + nsub], res=res, bins=bins)[0]  # noqa: E203
            for i, j in np.random.randint(size=(128, 2), low=0, high=npix - nsub)
        ]
    ).to(u.Jy ** 2 / u.sr)

    # plt.close('all')
    # plt.loglog(bins[1:], powspec_full.to(u.Jy**2/u.sr), c='k')
    # plt.loglog(bins[1:], np.mean(powspecs, axis=0))
    # plt.loglog(bins[1:], np.mean(powspecs, axis=0) + np.std(powspecs, axis=0), linestyle='dashed')
    # plt.loglog(bins[1:], np.mean(powspecs, axis=0) - np.std(powspecs, axis=0), linestyle='dashed')
    # plt.loglog(bins, (P(bins, alpha=alpha, fknee=1/u.arcsec) / res **2 * u.Jy**2).to(u.Jy**2 / u.sr))

    assert np.all((np.mean(powspecs, axis=0) - powspec_full.to(u.Jy ** 2 / u.sr)) < np.std(powspecs, axis=0))


def test_fake_data():

    # Dummy test for now
    nm = fake_data()

def test_meta_to_header():
    meta = {'toto': 'tata'}

    hdr = meta_to_header(meta)
    assert hdr['toto'] == meta['toto']

    meta['history'] = ['first', 'second']
    hdr = meta_to_header(meta)
    assert list(hdr['history']) == meta['history']

    meta['comment'] = ['first', 'second']
    hdr = meta_to_header(meta)
    assert list(hdr['comment']) == meta['comment']
