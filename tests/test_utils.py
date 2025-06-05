import math
import pytest

import jax.numpy as jnp
import jax.random as jr

from bcp.utils import predict_set


@pytest.fixture
def scores():
    key = jr.key(0)
    return jr.uniform(key, (100, 10))


def test_predict_scalar(scores):
    prediction_set = predict_set(scores, 0.5)
    assert prediction_set.dtype == bool
    assert prediction_set.shape == (100, 10)


def test_predict_singleton(scores):
    prediction_set = predict_set(scores, jnp.array([0.5]))
    assert prediction_set.dtype == bool
    assert prediction_set.shape == (1, 100, 10)


def test_predict_vector(scores):
    prediction_set = predict_set(scores, jnp.array([0.25, 0.5, 0.75]))
    assert prediction_set.dtype == bool
    assert prediction_set.shape == (3, 100, 10)
