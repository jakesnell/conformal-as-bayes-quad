<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Conformal Prediction as Bayesian Quadrature</h1>

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2502.13228" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
    <a style="text-decoration:none !important;" href="https://opensource.org/licenses/MIT" alt="License"><img src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
</p>

Code to accompany "Conformal Prediction as Bayesian Quadrature" by [Jake Snell](https://jakesnell.com) & [Tom Griffiths](https://cocosci.princeton.edu/tom/tom.php) (accepted as a spotlight at ICML 2025).

## Dependencies

- [uv](https://github.com/astral-sh/uv) for managing python packages and dependencies
- [just](https://github.com/casey/just) for running commands
- [gdown](https://github.com/wkentaro/gdown) for downloading MS-COCO data


## Running Tests

Run `just test`.

## Running Synthetic Binomial Experiments

1. Be sure that the `output` directory exists (e.g. by running `mkdir output`).
2. Run `just synth-run {method}`, where `{method}` is `crc` for Conformal Risk Control, `rcps` for Risk-controlling Prediction Sets, or `hpd` for our highest posterior density method. This will create a CSV file in `output` that contains the results of the experiment.
3. To summarize the results, run `just synth-analyze {method}`.

## Running Synthetic Heteroskedastic Experiments

Follow the same steps as the synthetic binomial experiments, but replace `synth` with `heteroskedastic`.

1. Be sure that the `output` directory exists (e.g. by running `mkdir output`).
2. Run `just heteroskedastic-run {method}`, where `{method}` is `crc` for Conformal Risk Control, `rcps` for Risk-controlling Prediction Sets, or `hpd` for our highest posterior density method. This will create a CSV file in `output` that contains the results of the experiment.
3. To summarize the results, run `just heteroskedastic-analyze {method}`.

## Running MS-COCO Experiments

First, run `just fetch` to download the necessary data[^1].  Then, follow the same steps as the heteroskedastic experiments above but replace `heteroskedastic` with `coco`.

1. Be sure that the `output` directory exists (e.g. by running `mkdir output`).
2. Run `just coco-run {method}`, where `{method}` is `crc` for Conformal Risk Control, `rcps` for Risk-controlling Prediction Sets, or `hpd` for our highest posterior density method. This will create a CSV file in `output` that contains the results of the experiment.
3. To summarize the results, run `just coco-analyze {method}`.

[^1]: Data credit: [conformal prediction](https://github.com/aangelopoulos/conformal-prediction) repository (Angelopoulos & Bates).
