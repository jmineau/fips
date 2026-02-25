<picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/_static/logo_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="/docs/_static/logo.png">
    <img alt="fips logo" src="/docs/_static/logo.png">
</picture>

# Flexible Inverse Problem Solver (FIPS)

[![Tests](https://github.com/jmineau/fips/actions/workflows/tests.yml/badge.svg)](https://github.com/jmineau/fips/actions/workflows/tests.yml)
[![Documentation](https://github.com/jmineau/fips/actions/workflows/docs.yml/badge.svg)](https://github.com/jmineau/fips/actions/workflows/docs.yml)
[![Code Quality](https://github.com/jmineau/fips/actions/workflows/quality.yml/badge.svg)](https://github.com/jmineau/fips/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/jmineau/fips/branch/main/graph/badge.svg)](https://codecov.io/gh/jmineau/fips)
[![PyPI version](https://badge.fury.io/py/fips.svg)](https://badge.fury.io/py/fips)
[![Python Version](https://img.shields.io/pypi/pyversions/fips.svg)](https://pypi.org/project/fips/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://img.shields.io/badge/pyright-checked-brightgreen.svg)](https://github.com/microsoft/pyright)

## Installation

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
S_x = pd.DataFrame(np.eye(12) * 0.5,  index=flux_idx, columns=flux_idx)
S_z = pd.DataFrame(np.eye(8)  * 0.1,  index=obs_idx,  columns=obs_idx)

problem = FluxProblem(
    obs=obs, prior=prior,
    forward_operator=H, prior_error=S_x, modeldata_mismatch=S_z,
).solve()

print(problem.posterior_fluxes)        # posterior pd.Series indexed by (time, lat, lon)
print(problem.estimator.reduced_chi2)  # reduced chi-squared statistic
```

### Multi-block — combined station + satellite observations

```python
import numpy as np
import pandas as pd
from fips import Block, Vector, Matrix, MatrixBlock, CovarianceMatrix, InverseProblem
from fips.covariance import

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
S_x = CovarianceMatrix(np.eye(N_f) * 0.5, index=flux_idx, columns=flux_idx)

# Model-data mismatch covariance: block-diagonal with separate error levels for stations vs. satellite
S_z_blks = [
    CovarianceMatrix(np.eye(8) * 0.1, index=station_idx, columns=station_idx),
    CovarianceMatrix(np.eye(3) * 0.2, index=satellite_idx, columns=satellite_idx),
]

# Pass blocks to the InverseProblem and solve
problem = InverseProblem(
    obs=obs_blks, prior=prior,
    forward_operator=H_blks, prior_error=S_x, modeldata_mismatch=S_z_blks,
).solve()

print(problem.posterior['flux'])       # posterior pd.Series indexed by (time, lat, lon)
print(problem.estimator.reduced_chi2)  # reduced chi-squared statistic
```

## Documentation

Full documentation is available at [https://jmineau.github.io/fips/](https://jmineau.github.io/fips/)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**James Mineau** - [jmineau](https://github.com/jmineau)
