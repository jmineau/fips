# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **A scalar passed to `Matrix` or `CovarianceMatrix` builds a diagonal**
  (breaking): `Matrix(5.0, index=idx)` returned a full matrix with 5.0 in every
  entry, which for a `CovarianceMatrix` has rank 1 and is not what "a
  covariance of 5" means. It now returns the square matrix with 5.0 on the
  diagonal and zeros elsewhere, the same as `Matrix([5.0, 5.0, 5.0], index=idx)`
  does for a 1D input. As for a 1D input, `columns` is not used, so a scalar no
  longer builds a rectangular matrix. For the old result, build the full array
  yourself: `Matrix(np.full((n, m), 5.0), index=rows, columns=cols)`.
  ([#50](https://github.com/jmineau/fips/issues/50))
- `FluxInversionPipeline.filter_state_space` raises a `TypeError` that says what
  is wrong when it has to infer the frequency of flux times that are not
  datetimes (the config has no `flux_freq`). Before, `pd.infer_freq` ran on
  whatever the `time` level was: integers raised pandas's own `TypeError`,
  strings and mixed time zones a `ValueError`, and timedeltas were accepted and
  dropped every observation, since no interval of timedeltas holds an
  observation time. `FluxInversionPipeline.summarize` raises the same kind of
  `TypeError` where it raised `AttributeError: 'Index' object has no attribute
  'year'`: the flux has more than two timesteps and its `time` level is not
  datetimes. A `time` level of timestamps stored as objects is not affected; a
  `Vector` already reads it as datetimes.
  ([#43](https://github.com/jmineau/fips/issues/43))

### Fixed

- A scalar passed to `Matrix` or `CovarianceMatrix` without an index raises
  "Scalar data requires an index.", as a 1D array does, in place of "All
  levels in the row index of Matrix must be named." A 0-d array
  (`np.array(5.0)`) is treated as a scalar; it raised pandas' "Must pass 2-d
  input".
  ([#50](https://github.com/jmineau/fips/issues/50))
- A `Structure1D` (`Vector`, `Block`) built from 2-D data that cannot be reduced
  to 1-D, such as a 2 x 2 DataFrame, raises a `ValueError` naming the class and
  the shape on every pandas version. On pandas 2 it raised "The truth value of
  a DataFrame is ambiguous".
  ([#51](https://github.com/jmineau/fips/issues/51))
- Selecting a block out of a `Matrix` with `matrix[row, col]` or
  `matrix.blocks[row, col]` (and so `InverseProblem.get_block` and the
  `FluxProblem` properties such as `jacobian`) kept the column levels that are
  NaN throughout the block, the levels only the other blocks have, while the
  rows were already cleaned. The columns are now cleaned too, so the block
  comes back with the levels it was built with.
  ([#46](https://github.com/jmineau/fips/issues/46))
- `Matrix` and `CovarianceMatrix` accept 1D diagonal values with a pandas
  index and preserve it on both axes. A 1D array or list without an index
  now raises an explicit error instead of creating an invalid unnamed index.
  ([#45](https://github.com/jmineau/fips/issues/45))
- An unnamed `Vector` keeps its missing name when copied, reindexed, or its
  index is rounded, instead of acquiring the string name `"None"`.
  Non-string names are converted to strings during construction so these
  operations preserve the stored name as well.
- A `Block` or `Vector` with a single element could not be built from a
  Series: `Block(pd.Series([0.1], index=..., name="bias"))` raised "All levels
  in the row index of Block must be named", because the one element was
  squeezed to a scalar and its index was lost. A state component with one entry
  (one bias for one site, one region) now builds, from a Series or from a
  one-row, one-column DataFrame.
  ([#38](https://github.com/jmineau/fips/issues/38))

## [0.1.0b9] - 2026-10-09

### Added

- `fips.__version__`.
- The API reference has a page for each class, with tables of its attributes and
  methods, and a subclass links to the members it inherits. The base classes
  (`fips.base`) and `Estimator` have reference pages.

### Changed

- **Requires Python 3.11 or newer** (breaking): Python 3.10 reaches its end of
  life in October 2026. Tested on 3.11 through 3.14.
- The version now comes from git tags (setuptools-scm). An install from git
  between releases reports a development version such as
  `0.1.0b9.dev3+g1a2b3c4` rather than the last release's number.

### Removed

- **`JacobianBuilder`** (breaking): it called PYSTILT's `Model` API, which
  PYSTILT 0.1.0a23 removed. PYSTILT now builds the Jacobian itself
  (`project.jacobian()`); wrap its matrix in a `MatrixBlock` (see Usage). The
  `flux` extra no longer installs `pystilt`.
  ([#5](https://github.com/jmineau/fips/issues/5))
- **`fips.problems.flux.transport`** (breaking), which held only
  `transport.stilt.errors`. It read STILT-R column names from PYSTILT's old
  `by-id` simulation folders; PYSTILT computes the transport error itself
  (`sim.transport_error`).
- The `typing-extensions` dependency (`typing.Self` is in the standard library
  from Python 3.11).

## [0.1.0b8] - 2026-09-29

### Changed

- **`JacobianBuilder` selects PYSTILT simulations by variant** (breaking;
  requires `pystilt>=0.1.0a21`). PYSTILT projects are
  now receptors crossed with named variants, each with at most one footprint,
  so `build_from_target(target, flux_times, variant)` takes the variant name
  (`"hrrr"`) where it took a footprint name. Regenerating footprints from
  stored particles moves to the keyword `footprint=FootprintConfig(...)`.
  `mets` is gone (a variant fixes its met). Selection goes through
  `model.simulations.sel(...)`, so `subset_hours` filters on the receptor's
  time instead of parsing directory names, and an unknown variant raises
  `KeyError`. `build_from_grid` and `build_from_coords` follow.

### Added

- `InverseProblem.prior_obs_error` (and `FluxProblem.prior_concentration_error`):
  the prior error covariance mapped into observation space, `H S_0 H^T`, as a
  labelled `CovarianceMatrix` on the observation index. Its diagonal is the
  prior's share of each observation's error budget; the solve already
  computed it, so this only exposes it (`Estimator.S_y0`).

### Fixed

- `transport.stilt.errors` used `matplotlib` in a return annotation without
  importing it; it is now imported under `TYPE_CHECKING`.
- Restored the Code Quality gates, which had failed on `main` since 0.1.0b6:
  docstrings in `transport.stilt.errors` are in imperative mood, and the
  remaining private helpers and nested functions have one-line docstrings
  (docstring coverage is back to 100%). The `@overload` stubs in `filters`
  carry a docstr-coverage excuse rather than a docstring, which ruff forbids.

## [0.1.0b7] - 2026-09-17

### Added
- `JacobianBuilder.build_from_target(target, flux_times, footprint)`: the
  primary Jacobian entry point. `target` is any PYSTILT spatial target
  (`stilt.Grid`, `stilt.Mesh`, `stilt.Zones`, an xarray grid, or a coords
  list), and the Jacobian's column index is the target's state index
  (`(cell, time)` for meshes and zones). Pass a `stilt.FootprintConfig` as `footprint` to
  regenerate each footprint from stored trajectories on that grid instead of
  regridding a stored raster. `build_from_grid` and `build_from_coords` are
  now thin aliases. Requires PYSTILT with spatial targets (> 0.1.0a8).

## [0.1.0b6] - 2026-09-02

### Added
- `.zenodo.json` citation metadata; releases are archived (and DOI-minted) on Zenodo

### Fixed
- Flaky Windows Tk test failures (force Agg matplotlib backend in tests)

## [0.1.0b5] - 2026-06-17

### Fixed
- **Singular-matrix failures in covariance solves** (`Estimator`): the Kalman gain `K`, `leverage`, `cost`, and `reduced_chi2` solved covariance systems with `scipy.linalg.solve(..., assume_a="pos")`, which forces a Cholesky factorization that requires *numerical* positive-definiteness. For `A = H S_0 H^T + S_z`, when `A` is ill-conditioned (a wide eigenvalue spread, with `H S_0 H^T` large relative to `S_z`) roundoff in forming `A` can tip it just off positive-definite even though it is positive-definite in exact arithmetic, raising `LinAlgError: singular matrix`. Covariance solves now go through a shared `_solve_psd` helper that keeps the fast Cholesky path and falls back to a backward-stable symmetric `LDL^T` solve (`assume_a="sym"`) with a warning. This generalizes and replaces the machine-epsilon jitter previously added in `reduced_chi2` (0.1.0b3).

## [0.1.0b4] - 2026-06-17

### Added
- **`JacobianBuilder.build_from_grid`** (`fips.problems.flux.transport.stilt`): Build the Jacobian from an xarray target grid (`lon`/`lat` or `x`/`y`; `NaN`-masked cells) or a coords list, conservatively regridding each footprint onto it. `build_from_coords` is now a thin wrapper that forwards to it. Advances the "build Jacobian from geometries or nested grid" roadmap item.

### Changed
- **PYSTILT dependency bumped to `>=0.1.0a3`** (`pyproject.toml`): requires the conservative `Footprint.aggregate` (the old nearest-point sampling undercounted coarse-grid sensitivities), which `build_from_grid` and `build_from_coords` rely on.

### Fixed
- **Parallel footprint loading in JacobianBuilder** (`build_from_coords`): Footprints are now loaded inside each parallel worker instead of serially before dispatch, eliminating a major NFS I/O bottleneck. Hours filtering is applied up-front by parsing the sim_id to avoid dispatching workers for footprints that would be discarded.
- **Unnamed time column causing MatrixBlock validation error** (`_build_jacobian_row_from_path`): `Footprint.aggregate` returns a DataFrame whose column index has `name=None`; this caused "All levels in the columns must be named." Fixed by setting `agg.columns.name = "time"` before stacking.

## [0.1.0b3] - 2026-05-12

### Added
- **Publish workflow** (`.github/workflows/publish.yml`): Added GitHub Actions workflow that builds the sdist and wheel, publishes to PyPI via OIDC trusted publishing, and creates a GitHub Release with the changelog entry — all triggered on `v*.*.*` tag pushes.
- **Desroziers diagnostic** (`Estimator.desroziers`, `InverseProblem.desroziers`): Desroziers et al. (2005) diagnosed observation error covariance, estimated from the innovation and analysis departure vectors. Symmetrized for single-realization use. Available as a numpy array on the estimator and as a `CovarianceMatrix` with pandas indexing on the problem.
- **Enhanced flux inversion summary** (`FluxInversionPipeline.summarize`): Comprehensive statistical summary including negative cell count/percentage, per-timestep mean flux table with prior/posterior/change columns, and posterior flux trend via linear regression (slope, R², p-value).

### Fixed
- **Singular matrix handling in chi-squared calculation** (`Estimator.reduced_chi2`): Added machine epsilon regularization to the diagonal of S_z and S_0 covariance matrices before solving to prevent singular matrix errors when observations are heavily aggregated (e.g., 3-day windows). This stabilizes the linear solve without significantly affecting the chi-squared value.
- **Flux map edge clipping** (`FluxPlotter.fluxes`): Map extent now expands by half a cell width beyond the outermost cell centers, so edge cells are fully visible instead of being clipped at their centers.
- **Jacobian builder error handling** (`JacobianBuilder.build_from_coords`): Added validation to raise helpful error when no Jacobian rows are produced, with counts of failed simulations and resolution info.

### Changed
- **PYSTILT dependency migrated to PyPI** (`pyproject.toml`): Replaced git source reference with `pystilt>=0.1.0a1` now that pystilt is published as an alpha package on PyPI. Updated `JacobianBuilder` to use the new `model.footprints[name].load(...)` API (replacing the removed `model.get_footprints()` method).
- **PYSTILT API rename** (`fips.problems.flux.transport.stilt`): Updated call sites to match renamed PYSTILT methods — `Simulation.read` → `Simulation.from_directory`, `Trajectory.read` → `Trajectory.from_parquet`.
- **Concentration plot improvements** (`FluxPlotter.concentrations`): Overhauled the concentration timeseries plot for better readability:
  - Raw observations shown as subtle background dots (smaller, more transparent)
  - Smoothed lines use a time-based rolling window (`rolling_window`, default 30 days) instead of a fixed fraction of data length
  - Grey shaded spans highlight data gaps exceeding `gap_threshold` (default 7 days), with rolling mean lines broken at gaps
  - Y-axis auto-scaled to the 1st–99th percentile range to prevent outlier stretching
  - Clean legend built from smoothed lines only, with a "no data" entry when gaps exist
- **location_mapper optimization** (`fips.problems.flux.transport.stilt.builder`): Refactored to use `index.to_frame()` instead of `reset_index()`/`set_index()` for more efficient index manipulation.
- **Jacobian row assembly delegates to `Footprint.aggregate`** (`build_jacobian_row_from_coords`): Coordinate rounding, spatial filtering, and time-bin integration are now handled by `stilt.Footprint.aggregate(coords, time_bins)`. The builder is now a thin assembler: call `fp.aggregate()`, stack time into the index, attach the `(obs_location, obs_time)` row index. Removed `calc_digits` and the direct `integrate_over_time_bins` import from this module.
- **PYSTILT `resolution` parameter renamed to `footprint`** (`build_from_coords`, `build_jacobian_row_from_coords`, `get_footprint`, `load_footprint`): Accepts a named footprint config key (string) or `None` for the finest available footprint. Removes the old `"{xres}x{yres}"` resolution string convention.
- **`errors.py` received from PYSTILT** (`fips.problems.flux.transport.stilt.errors`): Transport error estimation module (particle ensemble error analysis) moved here from the `stilt` package where it did not belong.

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
