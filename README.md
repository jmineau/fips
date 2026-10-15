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

> **⚠️ Beta Release**: FIPS is currently in beta (v0.1.0b1). The core API is stable and ready for testing, but may evolve based on user feedback. Please [report any issues or edge cases](https://github.com/jmineau/fips/issues) you encounter.

## Bridging Physics and Data
Inverse problems in geophysics and atmospheric science are incredibly complex, often involving massive state spaces, deeply heterogeneous observational networks, and explicit matrix-algebra requirements. While many general-purpose optimization tools exist, they often force researchers to strip away critical spatiotemporal metadata or translate pre-computed physical models into rigid, abstract array structures.

**FIPS (Flexible Inverse Problem Solver)** is built from the ground up to solve linear, matrix-based inverse problems without losing the context of your data. It provides the structural flexibility to handle messy, real-world realities seamlessly:

- **Native N-Dimensional Alignment**: FIPS natively utilizes `pandas.MultiIndex` to smoothly align heterogeneous datasets across any dimension. Whether you are mixing temporal, spatial, spectral, or sensor-specific data, your coordinates are never dropped or misaligned.

- **Modular Block Architecture**: Avoid wrangling monolithic arrays. Construct massive, multi-source state spaces and observation networks piece-by-piece from named blocks: a `Space` of labels, and an `Operator` built block by block.

- **Speak Your Domain's Language**: Built explicitly around the standard `y = Hx + error` paradigm. Directly plug in your pre-computed forward operators ($H$), prior covariances ($S_0$), and model-data mismatches ($S_z$).

- **Analytical Speed & Sparse Support**: FIPS is built for scale. Each matrix is kept in its cheapest form (dense, sparse, diagonal, or Kronecker factors), and by using direct linear algebra rather than expensive sampling algorithms, it computes exact analytical Maximum A Posteriori (MAP) estimates for massive state spaces in seconds.

## Installation

### From PyPI
```bash
pip install fips
```

### For development

```bash
git clone https://github.com/jmineau/fips.git
cd fips
uv sync --dev
```

## Usage

### Single-block — one observation source

```python
import numpy as np
import pandas as pd
import fips
from fips import Covariance, Operator, Problem, Vector

# State: gridded prior fluxes (time × lat × lon), one block named "flux"
flux_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=3, freq="MS"), [37.0, 38.0], [-112.0, -111.0]],
    names=["time", "lat", "lon"],
)
prior = Vector.from_series(pd.Series(np.ones(12) * 1.5, index=flux_idx), block="flux")

# Observations: tower concentration measurements
obs_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=8, freq="2W"), ["UOU"]],
    names=["time", "site"],
)
obs = Vector.from_series(pd.Series(np.ones(8) * 400.0, index=obs_idx), block="concentration")

# Forward operator (Jacobian), flux error covariance, obs error covariance
H   = Operator(np.random.rand(8, 12), obs.space, prior.space)
S_0 = Covariance.diagonal(0.5, prior.space)
S_z = Covariance.diagonal(0.1, obs.space)

problem = Problem(prior=prior, prior_cov=S_0, obs=obs, obs_cov=S_z, forward=H)
post = fips.solve(problem)

print(post.estimate["flux"].to_series())  # posterior fluxes, labeled by (time, lat, lon)
print(post.reduced_chi2)                  # reduced chi-squared statistic
```

### Multi-block — combined station + satellite observations

```python
import numpy as np
import pandas as pd
import fips
from fips import Covariance, Operator, Problem, Space, Vector

# State: same gridded prior fluxes (from above)
N_f = 12

# Obs block 1: ground station in-situ concentrations
station_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=8, freq="2W"), ["UOU"]],
    names=["time", "site"],
)

# Obs block 2: satellite column-average concentrations
satellite_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=3, freq="MS"), [37.5], [-111.5]],
    names=["time", "lat", "lon"],
)

# The observations: one vector of two blocks
obs = Vector.from_blocks({
    "station": pd.Series(np.ones(8) * 400.0, index=station_idx),
    "satellite": pd.Series(np.ones(3) * 0.00400, index=satellite_idx),
})

# Jacobian: one block per obs type, both mapping from the "flux" state block
H = Operator.from_blocks({
    ("station", "flux"): pd.DataFrame(np.random.rand(8, N_f),
                                      index=station_idx, columns=flux_idx),
    ("satellite", "flux"): pd.DataFrame(np.random.rand(3, N_f),
                                        index=satellite_idx, columns=flux_idx),
})

# Prior error covariance: independent flux errors
S_0 = Covariance.diagonal(0.5, Space({"flux": flux_idx}))

# Model-data mismatch covariance: block-diagonal with separate error levels for stations vs. satellite
S_z = Covariance.block_diag(
    Covariance.diagonal(0.1, Space({"station": station_idx})),
    Covariance.diagonal(0.2, Space({"satellite": satellite_idx})),
)

# Put everything in a Problem, which checks the labels, and solve it
problem = Problem(prior=prior, prior_cov=S_0, obs=obs, obs_cov=S_z, forward=H)
post = fips.solve(problem)

print(post.estimate["flux"].to_series())  # posterior fluxes, labeled by (time, lat, lon)
print(post.reduced_chi2)                  # reduced chi-squared statistic
```

## Documentation

Full documentation is available at [https://jmineau.github.io/fips/](https://jmineau.github.io/fips/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Use of AI coding agents

This project is developed with the help of AI coding agents, directed and
reviewed by the maintainer, who owns the design and the science.
