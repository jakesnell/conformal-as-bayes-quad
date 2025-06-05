import click
import scipy

import pandas as pd


@click.group()
def cli():
    pass


@cli.command()
@click.option("--method", required=True, help="Method to run (options include: crc)")
@click.option("--target_risk", required=True, type=float, help="The target risk")
def run(method: str, target_risk: float):
    df = pd.read_csv(f"output/heteroskedastic_{method}.csv", index_col=0)

    failure = df.val_risk > target_risk
    print(
        f"{failure.sum()}/{len(failure)} trials ({failure.mean()*100:0.2f}%) exceed target risk of {target_risk}"
    )
    test_result = scipy.stats.binomtest(failure.sum(), failure.size)
    ci = test_result.proportion_ci(confidence_level=0.95)

    print(
        f"failure rate = {100*(test_result.k / test_result.n):0.2f}%. 95% CI: [{100*ci.low:0.2f}%, {100*ci.high:0.2f}%]"
    )

    print(f"mean prediction interval length = {2 * df.lambda_hat.mean():0.2f}")


if __name__ == "__main__":
    cli()
