<div align=center>
<picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jmineau/fips/main/docs/_static/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/jmineau/fips/main/docs/_static/logo.png">
    <img alt="fips logo" src="https://raw.githubusercontent.com/jmineau/fips/main/docs/_static/logo.png" width="600">
</picture>
</div>

[![Tests](https://github.com/jmineau/fips/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/fips/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/fips/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/fips/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/fips/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/fips/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/fips/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/fips)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22258484.svg)](https://doi.org/10.5281/zenodo.22258484)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyrefly](https://img.shields.io/badge/types-pyrefly-blue.svg)](https://pyrefly.org)

> **⚠️ Beta**: fips is in beta, and its API may still change between releases. The [changelog](https://github.com/jmineau/fips/blob/main/CHANGELOG.md) says what changed, and the [migration guide](https://jmineau.github.io/fips/stable/migration.html) how to move code from 0.1. Please [report any issues or edge cases](https://github.com/jmineau/fips/issues) you encounter.

## Bridging Physics and Data
Inverse problems in geophysics and atmospheric science often involve large state spaces, heterogeneous observation networks, and explicit matrix algebra. General-purpose optimization tools tend to make researchers strip away the labels of their data (times, places, instruments) and translate their physical models into bare arrays.

**fips (Flexible Inverse Problem Solver)** solves linear inverse problems without losing the labels of your data:

- **Everything is matched by label**: every value carries a `pandas` index (time, latitude, site, ...). The inputs of a problem are matched by their labels, never by position, and a label on one side only is an error that names it.

- **Built from named blocks**: a state of fluxes and offsets, observations from towers and satellites. Each block keeps its own labels, and a matrix is built block by block.

- **The standard linear model**: observations are the forward operator times the state, plus an offset, plus an error. Plug in your forward operator (a Jacobian you computed), the prior and its covariance, and the observation covariance. Covariances are written as correlation models (exponential, Gaussian, Matérn, Kronecker products) with standard deviations.

- **Fast and exact**: each matrix is kept in its cheapest form (dense, sparse, diagonal, or Kronecker factors), and fips computes the exact Gaussian posterior with direct linear algebra, not sampling, for large state spaces in seconds.

## Installation

```bash
pip install fips            # the solver
pip install "fips[viz]"     # with plot_comparison (matplotlib)
```

For development:

```bash
git clone https://github.com/jmineau/fips.git
cd fips
uv sync
```

## Usage

### A first problem: rays through a line of cells

Six rays cross a line of four cells, timed by two instruments whose clocks may be off. A ray's travel time is the sum, over the cells it crosses, of its length in the cell times the cell's slowness, plus its instrument's clock delay. What are the slowness of each cell and the delay of each clock?

```python
import numpy as np
import pandas as pd

import fips

# The state: what we want to know. The slowness (s/km) of four cells along a
# line, labeled by where each cell is (km), and the clock delay (s) of each of
# the two instruments that timed the rays.
cells = pd.Index([0.5, 1.5, 2.5, 3.5], name="x")
instruments = pd.Index(["east", "west"], name="instrument")
prior = fips.Vector.from_blocks(
    {
        "slowness": pd.Series(0.25, index=cells),
        "delay": pd.Series(0.0, index=instruments),
    }
)

# The observations: the travel time (s) of six rays, each timed by one instrument.
rays = pd.MultiIndex.from_tuples(
    [("east", 1), ("east", 2), ("east", 3), ("west", 4), ("west", 5), ("west", 6)],
    names=["instrument", "ray"],
)
times = pd.Series([0.53, 0.76, 1.02, 0.25, 0.48, 0.80], index=rays)
obs = fips.Vector.from_series(times, block="time")

# The forward operator: how the state makes the observations. A ray's time is
# its length (km) in each cell times the cell's slowness, plus its instrument's
# delay. Each block of the operator is labeled, and placed by its labels.
lengths = pd.DataFrame(
    [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1]],
    index=rays,
    columns=cells,
    dtype=float,
)
delays = pd.DataFrame(
    [[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]],
    index=rays,
    columns=instruments,
    dtype=float,
)
forward = fips.Operator.from_blocks(
    {("time", "slowness"): lengths, ("time", "delay"): delays},
    rows=obs.space,
    cols=prior.space,
)

# The covariances: how far the prior and the observations may be off. Neighboring
# cells are alike (correlated over 1 km); the delays are independent of each other.
prior_cov = fips.Covariance.block_diag(
    slowness=fips.Exponential("x", 1.0).with_std(0.05),
    delay=fips.Independent().with_std(0.02),
)
obs_cov = fips.Independent().with_std(0.01)

problem = fips.Problem(prior, prior_cov, obs, obs_cov, forward)
post = fips.solve(problem)

# The answer, labeled as the prior was, and its standard deviation
print(post.estimate["slowness"].to_series().round(3))
print(post.estimate["delay"].to_series().round(3))
print(post.variances["slowness"].to_series().pow(0.5).rename("std").round(3))

# The posterior of an aggregate: the mean slowness of the two halves of the line
halves = pd.Series(["near", "near", "far", "far"], index=cells, name="half")
W = fips.aggregate.mean(prior.space, by=halves, blocks="slowness")
print(post.cov(W)["slowness", "slowness"].to_frame().round(6))
```

The prior and the observations are `Vector`s, the forward operator is an `Operator`, the covariances are models put on the labels by the `Problem`, and `fips.solve` returns the `Posterior`. The rows of `lengths` and `delays` could come in any order: everything is matched by label.

### The same objects at scale: an atmospheric flux inversion

The state is monthly fluxes on a grid, and the observations CO₂ every 6 hours at two sites. The prior covariance is correlated in time and in space and kept as its two Kronecker factors, so the full matrix is never formed. A transport Jacobian that covers only the cells a footprint touched is placed on the whole grid by its labels.

```python
import numpy as np
import pandas as pd

import fips

rng = np.random.default_rng(0)

# The state: monthly fluxes on a 3 x 3 grid, one block named "flux"
cells = pd.MultiIndex.from_product(
    [
        pd.date_range("2024-01-01", periods=3, freq="MS"),
        [40.6, 40.7, 40.8],
        [-112.0, -111.9, -111.8],
    ],
    names=["time", "lat", "lon"],
)
prior = fips.Vector.from_series(pd.Series(1.0, index=cells), block="flux")

# The observations: CO2 every 6 hours at two sites, one block named "co2"
labels = pd.MultiIndex.from_product(
    [["UOU", "WBB"], pd.date_range("2024-01-01", periods=360, freq="6h")],
    names=["site", "obs_time"],
)
co2 = pd.Series(rng.normal(420.0, 1.0, len(labels)), index=labels)
obs = fips.Vector.from_series(co2, block="co2")

# The transport Jacobian has columns only for the cells a footprint touched:
# from_blocks places it by its labels, and the other cells get zeros
touched = cells[rng.random(len(cells)) < 0.7]
jacobian = pd.DataFrame(
    rng.uniform(0.0, 0.5, (len(labels), len(touched))), index=labels, columns=touched
)
forward = fips.Operator.from_blocks(
    {("co2", "flux"): jacobian}, rows=obs.space, cols=prior.space
)

# The prior covariance, correlated in time and in space and kept as its two
# Kronecker factors; the observation covariance, correlated within a site's day
prior_cov = fips.Kronecker(
    fips.Exponential("time", "30D"),
    fips.Exponential(["lat", "lon"], 10.0, metric="haversine"),
).with_std(0.5)
obs_cov = fips.Independent().with_std(1.0) + fips.Exponential(
    "obs_time", "12h", within=["site", "obs_time.date"]
).with_std(0.5)

problem = fips.Problem(
    prior=prior,
    prior_cov=prior_cov,
    obs=obs,
    obs_cov=obs_cov,
    forward=forward,
    offset=415.0,  # the background: the part of the observations the fluxes do not explain
)
post = fips.solve(problem)

print(post.estimate["flux"].to_xarray())  # the posterior fluxes on (time, lat, lon)
print(post.summary())  # DOFS, chi-squared, R², RMSE, ... in one row

# The posterior of each month's total flux, without forming the full covariance
W = fips.aggregate.sum(prior.space, by="time")
print(post.cov(W).to_frame().round(3))
```

The [user guide](https://jmineau.github.io/fips/stable/usage.html) has more: offsets solved for with the state, aggregated observations, part of the state held at its prior, and repeated inversions.

## Documentation

Full documentation is available at [https://jmineau.github.io/fips/](https://jmineau.github.io/fips/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Use of AI coding agents

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science.
