from __future__ import absolute_import, division, print_function

import pytest
import numpy as np

from ..utils import pos_uniform, pos_gridded, pos_list


def test_pos_uniform():
    np.random.seed(0)
    shape = (5, 20)
    x, y = pos_uniform(nsources=100, shape=shape)
    assert -0.5 < x.min() or x.max() < shape[1]-0.5, 'pixel coordinate outside boundaries'
    assert -0.5 < y.min() or y.max() < shape[0]-0.5, 'pixel coordinate outside boundaries'

    mask = np.zeros(shape, dtype=np.bool)
    mask[:, :5] = True
    x, y = pos_uniform(nsources=100, shape=shape, mask=mask)
    assert 4 < np.floor(x.min()+0.5), 'pixel coordinate inside max'

    x, y = pos_uniform(nsources=100, shape=shape, within=(0.5, 1))
    assert shape[1] * 0.5 - 1 < np.floor(x.min()+0.5), 'pixel coordinate outside within'
    assert shape[0] * 0.5 - 1 < np.floor(y.min()+0.5), 'pixel coordinate outside within'

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
    assert np.all(x.reshape(3, 3) == np.linspace(shape[1]/4, shape[1]*3/4, 3) - 0.5), 'unexpected pixel coordinate'
    assert np.all(y.reshape(3, 3).T == np.linspace(shape[0]/4, shape[0]*3/4, 3) - 0.5), 'unexpected pixel coordinate'

    mask = np.zeros(shape, dtype=np.bool)
    mask[:, :5] = True
    with pytest.warns(UserWarning):
        x, y = pos_gridded(nsources=10**2, shape=shape, mask=mask)
    assert 4 < np.floor(x.min()+0.5), 'pixel coordinate inside max'

    with pytest.raises(AssertionError):
        x, y = pos_gridded(nsources=1)

    with pytest.raises(AssertionError):
        x, y = pos_gridded(nsources=2)

    np.random.seed(26)
    # This can raise an exception
    with pytest.warns(UserWarning):
        x, y = pos_gridded(nsources=3**2, shape=shape, wobble=True)

    x, y = pos_gridded(nsources=3**2, shape=shape, wobble=True, within=(1/3, 2/3))
    assert -0.5 < x.min() or x.max() < shape[1]-0.5, 'pixel coordinate outside boundaries'
    assert -0.5 < y.min() or y.max() < shape[0]-0.5, 'pixel coordinate outside boundaries'


def test_pos_list():

    shape = (5, 20)
    nsources = 20
    x_mean = np.linspace(0, 19, nsources)
    y_mean = np.ones(nsources)*2.5

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
        x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean, within=(1/4, 3/4))

    assert np.all(x == x_mean[shape[1]//4:shape[1]*3//4]), 'should be identical'

    x_mean = np.linspace(-1, 18, nsources)

    with pytest.warns(UserWarning):
        x, y = pos_list(nsources=nsources, shape=shape, x_mean=x_mean, y_mean=y_mean)

    assert np.all(x == x_mean[1:]), 'should be identical'
