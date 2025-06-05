from dataclasses import dataclass, field

from jax import Array
from jax.typing import ArrayLike

import jax.numpy as jnp
import jax.random as jr

from bcp.utils import validate_array_like


@dataclass
class DataSet:
    scores: Array
    labels: Array
    aux: dict = field(default_factory=lambda: {})

    def __post_init__(self):
        self.scores = validate_array_like(self.scores)
        self.labels = validate_array_like(self.labels)

    def __len__(self):
        return self.scores.shape[0]

    def __getitem__(self, key):
        return DataSet(self.scores[key], self.labels[key])

    def random_split(self, key, n: int):
        perm = jr.permutation(key, len(self))
        return self[perm[:n]], self[perm[n:]]


def load_coco():
    data = jnp.load("data/coco/coco-tresnetxl.npz")

    return DataSet(data["sgmd"], data["labels"].astype(bool))


def load_synth(key=jr.key(0)):
    n_cal = 10
    n_test = 10000
    n_event = 4

    u = jr.uniform(key, (n_cal + n_test, n_event))

    return DataSet(jnp.zeros((n_cal + n_test, 0)), u)


def load_heteroskedastic(key=jr.key(0)):
    n_cal = 200
    n_test = 5000

    key, subkey = jr.split(key)
    x = jr.uniform(subkey, (n_cal + n_test, 1)) * 4

    key, subkey = jr.split(key)
    y = jr.normal(subkey, (n_cal + n_test, 1)) * jnp.abs(x)

    return DataSet(jnp.abs(y), jnp.abs(y))
