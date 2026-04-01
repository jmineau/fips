# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Desroziers diagnostic** (`Estimator.desroziers`, `InverseProblem.desroziers`): Desroziers et al. (2005) diagnosed observation error covariance, estimated from the innovation and analysis departure vectors. Symmetrized for single-realization use. Available as a numpy array on the estimator and as a `CovarianceMatrix` with pandas indexing on the problem.
- **Enhanced flux inversion summary** (`FluxInversionPipeline.summarize`): Comprehensive statistical summary including negative cell count/percentage, per-timestep mean flux table with prior/posterior/change columns, and posterior flux trend via linear regression (slope, R², p-value).

### Fixed
- **Flux map edge clipping** (`FluxPlotter.fluxes`): Map extent now expands by half a cell width beyond the outermost cell centers, so edge cells are fully visible instead of being clipped at their centers.

### Changed
- **Concentration plot improvements** (`FluxPlotter.concentrations`): Overhauled the concentration timeseries plot for better readability:
  - Raw observations shown as subtle background dots (smaller, more transparent)
  - Smoothed lines use a time-based rolling window (`rolling_window`, default 30 days) instead of a fixed fraction of data length
  - Grey shaded spans highlight data gaps exceeding `gap_threshold` (default 7 days), with rolling mean lines broken at gaps
  - Y-axis auto-scaled to the 1st–99th percentile range to prevent outlier stretching
  - Clean legend built from smoothed lines only, with a "no data" entry when gaps exist

## [0.1.0b2] - 2026-03-23

### Added
- **Terminology & Notation documentation page**: Added comprehensive `docs/terminology.rst` page explaining:
  - Mathematical notation framework (lowercase for vectors, uppercase for matrices)
  - Subscript conventions (_0 for *a priori* / prior, _z for observation space)
  - Hat notation (^ for *a posteriori* / posterior)
  - Forward model relationship (z - c = y, y = Hx)
  - Quick reference table for all mathematical symbols
  - Diagnostic metrics (DOFS, χ², R²)
  - Inverse problem terminology glossary
- **Cross-references to terminology**: Added links to the terminology page from `getting_started.rst`, `usage.rst`, `reference/estimators.rst`, and `reference/inverse.rst` for easy navigation
- **Regularization factor (gamma) for BayesianSolver**: Added `gamma` parameter to control the balance between fitting observations and staying close to the prior. The regularization factor (γ) multiplies the observation term in the cost function: `J(x) = (x - x_0)^T S_0^{-1} (x - x_0) + γ*(z - Hx)^T S_z^{-1} (z - Hx)`. Values > 1 increase weight on data fitting (less regularization), while values < 1 decrease weight on data fitting (more regularization). Default is γ=1.0 (standard Bayesian inversion).
- **Estimator kwargs in pipeline**: Added `estimator_kwargs` parameter to `InversionPipeline.run()` and `FluxInversionPipeline.run()` methods to allow passing estimator-specific parameters (e.g., `gamma` for BayesianSolver) during the solve step.

### Changed
- **Pipeline step ordering** (`fips.pipeline`): Reordered `InversionPipeline.get_inputs()` to load the constant/background term before building the model-data mismatch covariance. This ensures that enhancement values (background-subtracted observations) are available when calculating the model-data mismatch covariance, which is essential for flux inversion problems where the MDM may depend on the enhancement rather than raw concentrations.

### Fixed
- **Pandas 3.x compatibility**: Fixed compatibility issues with pandas 3.x for datetime subtypes and MultiIndex operations:
  - **MultiIndex coordinate rounding** (`fips.problems.flux.transport.stilt.builder`): Updated coordinate rounding to use `get_level_values().round()` pattern instead of direct rounding, which is required for pandas 3.x MultiIndex operations
  - **DateTime subtype casting** (`fips.aggregators`): Added automatic dtype casting when binning datetime data to handle pandas 3.x datetime64 subtypes (e.g., datetime64[us] vs datetime64[ns]), preventing ValueError when dtypes differ between time data and bin edges
  - **Test coverage**: Added comprehensive test suite in `test_stilt_builder.py` to verify MultiIndex rounding behavior across pandas versions
- Handled series vs dataframe when reordering levels during reindexing in `fips.base.Structure`.

## [0.1.0b1] - 2026-03-02

### Added

This is the first beta release of FIPS (Flexible Inverse Problem Solver). The project has transitioned from Calendar Versioning (2025.10.0) to Semantic Versioning to better reflect development maturity and API stability.

#### Core Framework
- **Vector & Matrix structures**: Native support for state vectors and observation vectors with pandas MultiIndex for N-dimensional alignment
- **Operators**: Forward operators (H matrix) with flexible block-based construction and sparse matrix support
- **Covariance structures**: Prior covariance (S_a), observation error covariance (S_z), and posterior covariance computation
- **Estimators**: Bayesian MAP estimation with analytical solutions for linear inverse problems

#### Data Structures
- **Block architecture**: Modular `Block` and `MatrixBlock` objects for constructing large-scale problems from heterogeneous data sources
- **Index management**: Specialized indexing tools for aligning spatiotemporal coordinates across datasets
- **Sparse matrix support**: Optimized sparse data structures for memory-efficient large-scale computations

#### Processing & Analysis
- **Filters**: Data filtering and quality control tools
- **Aggregators**: Spatial and temporal aggregation of results with coordinate preservation
- **Kernels**: Covariance kernel functions for spatial and temporal correlations
- **Metrics**: Diagnostic metrics for inverse problem solution quality

#### Specialized Applications
- **Flux inversion problem**: Domain-specific implementation for atmospheric flux estimation
- **Pipeline support**: Workflow orchestration for multi-step inverse problem solutions

#### Visualization & I/O
- **Visualization tools**: Plotting utilities for state vectors, operators, and covariance structures
- **Serialization**: Native pandas/xarray serialization for results

#### Development Tools
- Comprehensive test suite with 18 test modules covering all major components
- Full documentation with Sphinx, including API reference and example notebooks
- CI/CD workflows for testing, code quality, and documentation deployment
- Pre-commit hooks with ruff (linting/formatting) and pyright (type checking)

### Changed
- **Versioning scheme**: Migrated from Calendar Versioning (CalVer) to Semantic Versioning (SemVer)
- **Development status**: Updated from Alpha to Beta

### Notes
- This beta release is ready for testing and feedback from colleagues
- The core API is stable but may evolve based on user feedback before 1.0.0
- Installation available via pip from GitHub: `pip install git+https://github.com/jmineau/fips`
- Please report issues and edge cases at https://github.com/jmineau/fips/issues

[unreleased]: https://github.com/jmineau/fips/compare/v0.1.0b1...HEAD
[0.1.0b1]: https://github.com/jmineau/fips/releases/tag/v0.1.0b1
