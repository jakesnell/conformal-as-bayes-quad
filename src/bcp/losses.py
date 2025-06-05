from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp

from bcp.utils import validate_array_like


def false_negative_rate(prediction_set: ArrayLike, labels: ArrayLike) -> Array:
    prediction_set = validate_array_like(prediction_set)
    labels = validate_array_like(labels)

    if prediction_set.dtype != bool:
        raise ValueError("prediction_set must be bool dtype")

    if labels.dtype != bool:
        raise ValueError("labels must be bool dtype")

    return jnp.sum(~prediction_set & labels, axis=-1) / jnp.sum(labels, axis=-1)


def scaled_count_loss(thresholds: ArrayLike, labels: ArrayLike) -> Array:
    thresholds = validate_array_like(thresholds)
    labels = validate_array_like(labels)

    if thresholds.dtype not in [jnp.float32, jnp.float64]:
        raise ValueError("thresholds must be float dtype")

    if labels.dtype not in [jnp.float32, jnp.float64]:
        raise ValueError("labels must be float dtype")

    return jnp.mean(labels > thresholds[..., jnp.newaxis], axis=-1)


def miscoverage_loss(thresholds: ArrayLike, labels: ArrayLike) -> Array:
    thresholds = validate_array_like(thresholds)
    labels = validate_array_like(labels)

    if thresholds.dtype not in [jnp.float32, jnp.float64]:
        raise ValueError("thresholds must be float dtype")

    if labels.dtype not in [jnp.float32, jnp.float64]:
        raise ValueError("labels must be float dtype")

    return jnp.mean(labels > thresholds[..., jnp.newaxis], axis=-1)
