from __future__ import absolute_import, division, print_function

import pytest
import numpy as np
import numpy.testing as npt

from ..utils import pos_in_mask
from ..utils import pos_uniform, pos_gridded, pos_list
from ..utils import fft_2D_hanning, powspec_k


def test_pos_in_mask():

    mask = np.asarray([[True, False], [False, False]])
    pos = [[0, 0], [0.5, 0.5], [1, 1]]

    result = pos_in_mask(pos)
    npt.assert_equal(result, pos)

    result = pos_in_mask(pos, mask)
    npt.assert_equal(result, pos[1:])


def test_pos_uniform():
    np.random.seed(0)
    shape = (5, 20)
    x, y = pos_uniform(nsources=100, shape=shape)
    assert -0.5 < x.min() or x.max() < shape[1] - 0.5, 'pixel coordinate outside boundaries'
    assert -0.5 < y.min() or y.max() < shape[0] - 0.5, 'pixel coordinate outside boundaries'

    mask = np.zeros(shape, dtype=np.bool)
    mask[:, :5] = True
    x, y = pos_uniform(nsources=100, shape=shape, mask=mask)
    assert 4 < np.floor(x.min() + 0.5), 'pixel coordinate inside max'

    x, y = pos_uniform(nsources=100, shape=shape, within=(0.5, 1))
    assert shape[1] * 0.5 - 1 < np.floor(x.min() + 0.5), 'pixel coordinate outside within'
    assert shape[0] * 0.5 - 1 < np.floor(y.min() + 0.5), 'pixel coordinate outside within'

    x, y = pos_uniform(nsources=30, shape=shape, dist_threshold=1)
    dist = np.sqrt((x[:, np.newaxis] - x)**2 + (y[:, np.newaxis] - y)**2)
    i = np.arange(len(x))
    dist[i, i] = np.inf
    assert np.all(np.min(dist, 1) > 1), 'sources too close'

    with pytest.warns(UserWarning):
        x, y = pos_uniform(nsources=100, shape=shape, dist_threshold=1)


def test_pos_gridded():

    shape = (9, 21)
    x, y = pos_gridded(nsources=3**2, shape=shape)
    assert np.all(x.reshape(3, 3) == np.linspace(shape[1] / 4, shape[1] * 3 / 4, 3) - 0.5), 'unexpected pixel coordinate'
    assert np.all(y.reshape(3, 3).T == np.linspace(shape[0] / 4, shape[0] * 3 / 4, 3) - 0.5), 'unexpected pixel coordinate'

    mask = np.zeros(shape, dtype=np.bool)
    mask[:, :5] = True
    with pytest.warns(UserWarning):
        x, y = pos_gridded(nsources=10**2, shape=shape, mask=mask)
    assert 4 < np.floor(x.min() + 0.5), 'pixel coordinate inside max'

    with pytest.raises(AssertionError):
        x, y = pos_gridded(nsources=1)

    with pytest.raises(AssertionError):
        x, y = pos_gridded(nsources=2)

    np.random.seed(26)
    # This can raise an exception
    with pytest.warns(UserWarning):
        x, y = pos_gridded(nsources=3**2, shape=shape, wobble=True)

    x, y = pos_gridded(nsources=3**2, shape=shape, wobble=True, within=(1 / 3, 2 / 3))
    assert -0.5 < x.min() or x.max() < shape[1] - 0.5, 'pixel coordinate outside boundaries'
    assert -0.5 < y.min() or y.max() < shape[0] - 0.5, 'pixel coordinate outside boundaries'


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
    assert np.all(x == x_mean), 'should be identical'
    assert np.all(y == y_mean), 'should be identical'

    mask = np.zeros(shape, dtype=np.bool)
    mask[:, :5] = True
    with pytest.warns(UserWarning):
        x, y = pos_list(nsources=nsources, shape=shape, mask=mask, x_mean=x_mean, y_mean=y_mean)

    assert np.all(x == x_mean[5:]), 'should be identical'
    assert np.all(y == y_mean[5:]), 'should be identical'

    with pytest.warns(UserWarning):
        x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean, within=(1 / 4, 3 / 4))

    assert np.all(x == x_mean[shape[1] // 4:shape[1] * 3 // 4]), 'should be identical'
    x_mean = np.linspace(-1, 18, nsources)

    with pytest.warns(UserWarning):
        x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean)

    assert np.all(x == x_mean[1:]), 'should be identical'


def test_fft_2D_hanning_assertion():

    shape = 5
    mask = np.ones((shape, shape), dtype=bool)

    with pytest.raises(AssertionError):
        fft_2D_hanning(mask, size=2)

    with pytest.raises(AssertionError):
        fft_2D_hanning(mask, size=1)


def test_fft_2D_hanning():

    # Min hann filter is 5 x 5
    shape = 15
    size = 5
    apod_size = 2

    mask = np.ones((shape, shape), dtype=bool)
    islice = slice(((shape - 1) - (size - 1)) // 2,
                   ((shape - 1) + (size - 1)) // 2 + 1)
    mask[islice, islice] = False

    apod = fft_2D_hanning(mask, size=apod_size)
    # Nothing outside the mask
    assert np.all((apod > 1e-15) == ~mask)
    # Only unchanged pixel at the center
    unchanged = slice(((shape - 1) - (size - 1) + (apod_size * 2 + 1 - 1)) // 2,
                      ((shape - 1) + (size - 1) - (apod_size * 2 + 1 - 1)) // 2 + 1)
    npt.assert_equal(apod[unchanged, unchanged], 1)

    shape = 45
    size = 15
    apod_size = 2

    mask = np.ones((shape, shape), dtype=bool)
    islice = slice(((shape - 1) - (size - 1)) // 2,
                   ((shape - 1) + (size - 1)) // 2 + 1)
    mask[islice, islice] = False
    apod = fft_2D_hanning(mask, size=2)

    assert np.all((apod > 1e-15) == ~mask)
    unchanged = slice(((shape - 1) - (size - 1) + (apod_size * 2 + 1 - 1)) // 2,
                      ((shape - 1) + (size - 1) - (apod_size * 2 + 1 - 1)) // 2 + 1)
    npt.assert_allclose(apod[unchanged, unchanged], 1)


def P(k, alpha=-11. / 3, fknee=1):
    """Simple power law formula"""
    return (k / fknee)**alpha


def gen_pkfield(npix=32, alpha=-11. / 3, fknee=1, res=1):
    """Generate a 2D square map with P(k) field"""

    ufreq = np.fft.fftfreq(npix, d=res)
    kfreq = np.sqrt(ufreq[:, np.newaxis]**2 + ufreq**2)

    with np.errstate(divide='ignore'):
        psd = 2 * P(kfreq, alpha=alpha, fknee=fknee)
    psd[0, 0] = 0

    pha = np.random.uniform(low=-np.pi, high=np.pi, size=(npix, npix))

    fft_img = np.sqrt(psd) * (np.cos(pha) + 1j * np.sin(pha))
    return np.real(np.fft.ifft2(fft_img)) * npix


def test_powspec_k():

    npix = 128
    res = 1
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

    # plt.plot(bin_centers[1:], mean_Pk[1:])
    # plt.plot(bin_centers[1:], mean_Pk[1:]+std_Pk[1:])
    # plt.plot(bin_centers[1:], mean_Pk[1:]-std_Pk[1:])
    # plt.plot(bin_centers[1:], P(bin_centers[1:]))

    assert np.all((mean_Pk[1:] - P(bin_centers[1:])) < std_Pk[1:])
