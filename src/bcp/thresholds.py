import math
from typing import Callable

from jax import Array
import jax.numpy as jnp
import jax.random as jr

from jaxopt import Bisection

from bcp.data_set import DataSet


def conformal_risk_control_threshold(
    data_set: DataSet,
    predict: Callable,
    loss: Callable,
    target_risk: float,
    max_risk: float,
    max_threshold: float,
) -> Array:
    n = len(data_set)

    def optimality_fun(threshold):
        empirical_risk = loss(
            predict(data_set.scores, threshold), data_set.labels
        ).mean(axis=-1)
        return (n / (n + 1.0)) * empirical_risk + max_risk / (n + 1.0) - target_risk

    solver = Bisection(
        optimality_fun=optimality_fun, lower=0, upper=max_threshold, check_bracket=False
    )
    return solver.run().params


def hpd_threshold(
    data_set: DataSet,
    predict: Callable,
    loss: Callable,
    target_risk: float,
    max_risk: float,
    max_threshold: float,
    hpd_level: float,
    num_dir: int,
    key,
) -> Array:
    n = len(data_set)

    bin_widths = jr.dirichlet(key, jnp.ones(n + 1), shape=(num_dir,))

    def optimality_fun(threshold):
        observed_losses = loss(predict(data_set.scores, threshold), data_set.labels)
        l_plus = bin_widths[:, :-1] @ observed_losses + bin_widths[:, 0] * max_risk
        return jnp.quantile(l_plus, hpd_level) - target_risk

    solver = Bisection(
        optimality_fun=optimality_fun, lower=0, upper=max_threshold, check_bracket=False
    )
    return solver.run().params


def hoeffding_ucb(loss: Array, delta: float):
    """The Hoeffding upper confidence bound for losses in [0, 1].

    Args:
        loss (Tensor): A tensor of loss values with shape (N), where N is the
                       number of examples.
        delta (float): The probability of error for the UCB.

    Returns:
        float: The UCB value.
    """
    assert loss.ndim == 1
    num_train = loss.shape[-1]
    mean_loss = jnp.mean(loss)
    # compute upper confidence bound on risk for each hypothesis
    return mean_loss + math.sqrt(math.log(1.0 / delta) / (2 * num_train))


def rcps_threshold(
    data_set: DataSet,
    predict: Callable,
    loss: Callable,
    target_risk: float,
    max_risk: float,
    max_threshold: float,
    delta: float,
) -> Array:
    # TODO: Fix UCB to take into account max_risk
    n = len(data_set)

    def optimality_fun(threshold):
        observed_losses = loss(predict(data_set.scores, threshold), data_set.labels)
        return hoeffding_ucb(observed_losses, delta) - target_risk

    solver = Bisection(
        optimality_fun=optimality_fun, lower=0, upper=max_threshold, check_bracket=False
    )
    return solver.run().params
