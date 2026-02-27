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
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/pyright-checked-brightgreen.svg)](https://github.com/microsoft/pyright)

## Bridging Physics and Data
Inverse problems in geophysics and atmospheric science are incredibly complex, often involving massive state spaces, deeply heterogeneous observational networks, and explicit matrix-algebra requirements. While many general-purpose optimization tools exist, they often force researchers to strip away critical spatiotemporal metadata or translate pre-computed physical models into rigid, abstract array structures.

**FIPS (Flexible Inverse Problem Solver)** is built from the ground up to solve linear, matrix-based inverse problems without losing the context of your data. It provides the structural flexibility to handle messy, real-world realities seamlessly:

- **Native N-Dimensional Alignment**: FIPS natively utilizes `pandas.MultiIndex` to smoothly align heterogeneous datasets across any dimension. Whether you are mixing temporal, spatial, spectral, or sensor-specific data, your coordinates are never dropped or misaligned.

- **Modular Block Architecture**: Avoid wrangling monolithic arrays. Construct massive, multi-source state spaces and observation networks piece-by-piece using specialized `Block` and `MatrixBlock` objects.

- **Speak Your Domain's Language**: Built explicitly around the standard `y = Hx + error` paradigm. Directly plug in your pre-computed forward operators ($H$), prior covariances ($S_x$), and model-data mismatches ($S_z$).

- **Analytical Speed & Sparse Support**: FIPS is built for scale. By leveraging optimized sparse data structures and direct linear algebra rather than expensive sampling algorithms, it computes exact analytical Maximum A Posteriori (MAP) estimates for massive state spaces in seconds.

## Installation

### From GitHub
```bash
pip install git+https://github.com/jmineau/fips
```

### From Source

```bash
git clone https://github.com/jmineau/fips.git
cd fips
pip install -e .
```

## Usage

### Single-block — one observation source

```python
import numpy as np
import pandas as pd
from fips import Block, CovarianceMatrix
from fips.problems.flux import FluxProblem

# State: gridded prior fluxes (time × lat × lon)
flux_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=3, freq="MS"), [37.0, 38.0], [-112.0, -111.0]],
    names=["time", "lat", "lon"],
)
prior = pd.Series(np.ones(12) * 1.5, index=flux_idx, name="flux")

# Observations: tower concentration measurements
obs_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=8, freq="2W"), ["UOU"]],
    names=["time", "site"],
)
obs = pd.Series(np.ones(8) * 400.0, index=obs_idx, name="concentration")

# Forward operator (Jacobian), flux error covariance, obs error covariance
H   = pd.DataFrame(np.random.rand(8, 12), index=obs_idx,  columns=flux_idx)
S_0 = pd.DataFrame(np.eye(12) * 0.5,  index=flux_idx, columns=flux_idx)
S_z = pd.DataFrame(np.eye(8)  * 0.1,  index=obs_idx,  columns=obs_idx)

problem = FluxProblem(
    obs=obs, prior=prior,
    forward_operator=H, prior_error=S_0, modeldata_mismatch=S_z,
).solve()

print(problem.posterior_fluxes)        # posterior pd.Series indexed by (time, lat, lon)
print(problem.estimator.reduced_chi2)  # reduced chi-squared statistic
```

### Multi-block — combined station + satellite observations

```python
import numpy as np
import pandas as pd
from fips import Block, Vector, Matrix, MatrixBlock, CovarianceMatrix, InverseProblem

# State: same gridded prior fluxes (from above)
N_f = 12

# Obs block 1: ground station in-situ concentrations
station_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=8, freq="2W"), ["UOU"]],
    names=["time", "site"],
)
station_obs = Block(pd.Series(np.ones(8) * 400.0,
                              index=station_idx, name="station"))

# Obs block 2: satellite column-average concentrations
satellite_idx = pd.MultiIndex.from_product(
    [pd.date_range("2023-01", periods=3, freq="MS"), [37.5], [-111.5]],
    names=["time", "lat", "lon"],
)
satellite_obs = Block(pd.Series(np.ones(3) * 0.00400,
                                index=satellite_idx, name="satellite"))

# Combine obs blocks into a list
obs_blks = [station_obs, satellite_obs]

# Jacobian: one MatrixBlock per obs type, both mapping to the "flux" state block
H_blks = [
    MatrixBlock(
        pd.DataFrame(np.random.rand(8, N_f),
        index=station_idx, columns=flux_idx),
        row_block="station", col_block="flux",
    ),
    MatrixBlock(
        pd.DataFrame(np.random.rand(3, N_f),
        index=satellite_idx, columns=flux_idx),
        row_block="satellite", col_block="flux",
    ),]

# Prior error covariance: only flux errors, no cross-block covariances
S_0 = CovarianceMatrix(np.eye(N_f) * 0.5, index=flux_idx, columns=flux_idx)

# Model-data mismatch covariance: block-diagonal with separate error levels for stations vs. satellite
S_z_blks = [
    CovarianceMatrix(np.eye(8) * 0.1, index=station_idx, columns=station_idx),
    CovarianceMatrix(np.eye(3) * 0.2, index=satellite_idx, columns=satellite_idx),
]

# Pass blocks to the InverseProblem and solve
problem = InverseProblem(
    obs=obs_blks, prior=prior,
    forward_operator=H_blks, prior_error=S_0, modeldata_mismatch=S_z_blks,
).solve()

print(problem.posterior['flux'])       # posterior pd.Series indexed by (time, lat, lon)
print(problem.estimator.reduced_chi2)  # reduced chi-squared statistic
```

## Documentation

Full documentation is available at [https://jmineau.github.io/fips/](https://jmineau.github.io/fips/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
