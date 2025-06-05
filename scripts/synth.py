import click
from functools import partial

import pandas as pd

import jax
import jax.random as jr

from bcp.data_set import load_synth
from bcp.losses import scaled_count_loss
from bcp.utils import predict_identity, tree_concat
from bcp.thresholds import (
    conformal_risk_control_threshold,
    hpd_threshold,
    rcps_threshold,
)

jax.config.update("jax_enable_x64", True)


@click.group()
def cli():
    pass


@cli.command()
def diagnostics():
    data_set = load_synth()
    print(f"Dataset contains {len(data_set)} examples")


@cli.command()
@click.option("--method", required=True, help="Method to run (options include: crc)")
@click.option("--n", default=10, help="Number of calibration examples")
@click.option("--target_risk", default=0.4, help="Target risk")
@click.option("--num_trials", default=1000, help="Number of random trials")
@click.option("--batch_size", default=100, help="Number of trials to process per batch")
@click.option("--hpd_level", default=0.95, help="HPD level")
@click.option("--delta", default=0.05, help="probability of failure for rcps")
@click.option("--num_dir", default=1000, help="Number of Dirichlet samples")
@click.option("--seed", default=0, help="Random seed")
@click.option("--out_file", required=False)
def run(
    method: str,
    n: int,
    target_risk: float,
    num_trials: int,
    batch_size: int,
    hpd_level: float,
    delta: float,
    num_dir: int,
    seed: int,
    out_file: str,
):
    key = jr.key(seed)

    data_set = load_synth()

    if method == "crc":
        threshold_fun = conformal_risk_control_threshold
    elif method == "hpd":
        key, subkey = jr.split(key)
        threshold_fun = partial(
            hpd_threshold, hpd_level=hpd_level, num_dir=num_dir, key=subkey
        )
    elif method == "rcps":
        threshold_fun = partial(rcps_threshold, delta=delta)
    else:
        raise ValueError(f"unknown method {method}")

    get_threshold = lambda ds: threshold_fun(
        ds, predict_identity, scaled_count_loss, target_risk, 1.0, 1.0
    )

    def run_trial(subkey):
        cal_set, val_set = data_set.random_split(subkey, n)

        lambda_hat = get_threshold(cal_set)

        predictions = predict_identity(val_set.scores, lambda_hat)
        val_risk = scaled_count_loss(predictions, val_set.labels).mean(axis=-1)

        return lambda_hat, val_risk

    run_batch = jax.jit(jax.vmap(run_trial))

    results = []

    num_processed = 0
    while num_processed < num_trials:
        key, subkey = jr.split(key)
        results.append(run_batch(jr.split(subkey, batch_size)))
        num_processed += batch_size
        print(num_processed)

    lambda_hat, val_risk = tree_concat(results)

    df = pd.DataFrame(dict(lambda_hat=lambda_hat, val_risk=val_risk))

    print(df)

    if out_file is not None:
        df.to_csv(out_file)


if __name__ == "__main__":
    cli()
