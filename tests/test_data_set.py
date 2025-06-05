import pytest

import jax.random as jr
import jax.numpy as jnp

from bcp.data_set import DataSet, load_coco, load_synth, load_heteroskedastic


@pytest.fixture
def scores():
    key = jr.key(0)
    return jr.uniform(key, (100, 10))


@pytest.fixture
def labels(scores):
    return scores >= 0.5


@pytest.fixture
def data_set(scores, labels):
    return DataSet(scores, labels)


def test_dataset_basic(scores, labels):
    assert isinstance(DataSet(scores, labels), DataSet)


def test_len(data_set):
    assert len(data_set) == 100


def test_slice(data_set):
    ds1 = data_set[:10]
    ds2 = data_set[10:]
    assert len(ds1) == 10
    assert len(ds2) == 90


def test_permute():
    scores = jnp.array([1, 2, 4, 6, 8])
    labels = jnp.array([True, False, False, True, True])
    data_set = DataSet(scores, labels)
    shuffled = data_set[jnp.array([4, 2, 3, 1, 0])]
    assert jnp.array_equal(shuffled.scores, jnp.array([8, 4, 6, 2, 1]))
    assert jnp.array_equal(shuffled.labels, jnp.array([True, False, True, False, True]))


def test_permute_split(data_set):
    n = int(len(data_set) * 0.8)
    m = len(data_set) - n

    perm = jr.permutation(jr.key(1), len(data_set))
    ds_shuffled = data_set[perm]
    ds1 = ds_shuffled[:n]
    ds2 = ds_shuffled[n:]

    assert len(ds1) == n
    assert len(ds2) == m


def test_random_split(data_set):
    n = int(len(data_set) * 0.8)
    m = len(data_set) - n

    ds1, ds2 = data_set.random_split(jr.key(1), n)
    assert len(ds1) == n
    assert len(ds2) == m


def test_coco():
    dataset = load_coco()
    assert dataset.scores.shape == (4952, 80)
    assert dataset.labels.shape == (4952, 80)


def test_synth():
    dataset = load_synth()
    assert dataset.scores.shape == (10010, 0)
    assert dataset.labels.shape == (10010, 4)


def test_heteroskedastic():
    dataset = load_heteroskedastic()
    assert dataset.scores.shape == (5200, 1)
    assert dataset.labels.shape == (5200, 1)
