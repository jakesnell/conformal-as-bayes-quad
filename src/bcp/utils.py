from typing import Callable

import jax
from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp


def validate_array_like(arr: ArrayLike) -> Array:
    if not isinstance(arr, ArrayLike):
        raise TypeError(f"Expected arraylike, got {arr}")
    return jnp.asarray(arr)


def predict_set(scores: ArrayLike, threshold: ArrayLike) -> Array:
    scores = jnp.asarray(scores)
    threshold = jnp.asarray(threshold)
    return jnp.asarray(scores) >= jnp.reshape(
        threshold, threshold.shape + (1,) * scores.ndim
    )


def predict_identity(scores: ArrayLike, threshold: ArrayLike) -> Array:
    scores = jnp.asarray(scores)
    threshold = jnp.asarray(threshold)
    return jnp.reshape(threshold, threshold.shape + (1,) * (scores.ndim - 1))


# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75
def tree_stack(trees):
    return jax.tree.map(lambda *v: jnp.stack(v), *trees)


def tree_concat(trees):
    return jax.tree.map(lambda *v: jnp.concat(v), *trees)
