import pytest

import jax.numpy as jnp
import jax.random as jr

from bcp.losses import false_negative_rate, scaled_count_loss, miscoverage_loss
from bcp.utils import predict_set


@pytest.fixture
def fnr_scores():
    key = jr.key(0)
    return jr.uniform(key, (100, 10))


@pytest.fixture
def fnr_labels(fnr_scores):
    return fnr_scores >= 0.5


@pytest.fixture
def sc_scores():
    return jnp.zeros((100, 0))


@pytest.fixture
def sc_labels():
    key = jr.key(0)
    return jr.uniform(key, (100, 4))


@pytest.fixture
def hs_scores():
    key = jr.key(0)
    return jnp.abs(jr.normal(key, (100, 1)))


@pytest.fixture
def hs_labels(hs_scores):
    return hs_scores


def test_false_negative_rate(fnr_scores, fnr_labels):
    losses = false_negative_rate(predict_set(fnr_scores, 0.25), fnr_labels)
    assert losses.dtype == jnp.float32
    assert losses.shape == (100,)
    assert jnp.min(losses) >= 0.0
    assert jnp.max(losses) <= 1.0


def test_scaled_count_loss_worst(sc_scores, sc_labels):
    losses = scaled_count_loss(jnp.zeros(sc_scores.shape[0]), sc_labels)

    assert losses.dtype == jnp.float32
    assert losses.shape == (100,)
    assert jnp.min(losses) == 1.0
    assert jnp.max(losses) == 1.0


def test_scaled_count_loss_best(sc_scores, sc_labels):
    losses = scaled_count_loss(jnp.ones(sc_scores.shape[0]), sc_labels)

    assert losses.dtype == jnp.float32
    assert losses.shape == (100,)
    assert jnp.min(losses) == 0.0
    assert jnp.max(losses) == 0.0


def test_scaled_count_loss_mixed(sc_scores, sc_labels):
    thresholds = jnp.concatenate(
        [jnp.zeros(sc_scores.shape[0] // 2), jnp.ones(sc_scores.shape[0] // 2)], 0
    )
    losses = scaled_count_loss(thresholds, sc_labels)

    assert losses.dtype == jnp.float32
    assert losses.shape == (100,)
    assert jnp.min(losses) == 0.0
    assert jnp.max(losses) == 1.0

    assert jnp.allclose(1 - thresholds, losses)


def test_scaled_count_loss_random(sc_scores, sc_labels):
    key = jr.key(1)
    thresholds = jr.uniform(key, sc_scores.shape[0])
    losses = scaled_count_loss(thresholds, sc_labels)

    assert losses.dtype == jnp.float32
    assert losses.shape == (100,)
    assert jnp.min(losses) >= 0.0
    assert jnp.max(losses) <= 1.0
    assert jnp.mean(losses) > 0.0
    assert jnp.mean(losses) < 1.0


def test_miscoverage_loss(hs_scores, hs_labels):
    thresholds = 1.0 * jnp.ones(hs_scores.shape[0])

    losses = miscoverage_loss(thresholds, hs_labels)

    assert losses.dtype == jnp.float32
    assert losses.shape == (100,)
    assert jnp.min(losses) == 0.0
    assert jnp.max(losses) == 1.0
    assert jnp.abs(losses).sum() == (1.0 * (losses > 0)).sum()
    assert jnp.mean(losses) > 0.0
    assert jnp.mean(losses) < 1.0
