import pytest

import jax.random as jr
import jax.numpy as jnp

from bcp.data_set import DataSet
from bcp.thresholds import (
    conformal_risk_control_threshold,
    hpd_threshold,
    rcps_threshold,
)
from bcp.losses import false_negative_rate, scaled_count_loss
from bcp.utils import predict_set, predict_identity


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


@pytest.fixture
def synth_data_set():
    key = jr.key(0)
    scores = jnp.zeros((100, 0))
    labels = jr.uniform(key, (100, 4))
    return DataSet(scores, labels)


def test_crc(data_set):
    lambda_hat = conformal_risk_control_threshold(
        data_set, predict_set, false_negative_rate, 0.3, 1.0, 1.0
    )
    assert jnp.isscalar(lambda_hat)


def test_hpd(data_set):
    lambda_hat = hpd_threshold(
        data_set,
        predict_set,
        false_negative_rate,
        0.3,
        1.0,
        1.0,
        0.95,
        100,
        key=jr.key(1),
    )
    assert jnp.isscalar(lambda_hat)


def test_rcps(data_set):
    lambda_hat = rcps_threshold(
        data_set, predict_set, false_negative_rate, 0.3, 1.0, 1.0, 0.05
    )
    assert jnp.isscalar(lambda_hat)


def test_synth_crc(synth_data_set):
    lambda_hat = conformal_risk_control_threshold(
        synth_data_set, predict_identity, scaled_count_loss, 0.3, 1.0, 1.0
    )
    assert jnp.isscalar(lambda_hat)


def test_synth_hpd(synth_data_set):
    lambda_hat = hpd_threshold(
        synth_data_set,
        predict_identity,
        scaled_count_loss,
        0.3,
        1.0,
        1.0,
        0.95,
        100,
        key=jr.key(1),
    )
    assert jnp.isscalar(lambda_hat)


def test_synth_rcps(synth_data_set):
    lambda_hat = rcps_threshold(
        synth_data_set, predict_identity, scaled_count_loss, 0.3, 1.0, 0.05, 1.0
    )
    assert jnp.isscalar(lambda_hat)
