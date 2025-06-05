<h1 align="center" style="margin-bottom:0px; border-bottom:0px; padding-bottom:0px">Conformal Prediction as Bayesian Quadrature</h1>

<p align="center">
    <a style="text-decoration:none !important;" href="https://arxiv.org/abs/2502.13228" alt="arXiv"><img src="https://img.shields.io/badge/paper-arXiv-red" /></a>
</p>

Code to accompany "Conformal Prediction as Bayesian Quadrature" (accepted as a spotlight at ICML 2025).

## Dependencies

- [uv](https://github.com/astral-sh/uv) for managing python packages and dependencies
- [just](https://github.com/casey/just) for running commands
- [gdown](https://github.com/wkentaro/gdown) for downloading MS-COCO data

## Preparing Data

Run `just fetch`. This will download the data needed to run MS-COCO experiments.

Data credit: [conformal prediction](https://github.com/aangelopoulos/conformal-prediction) repository (Angelopoulos & Bates).

## Running Tests

Run `just test`.
