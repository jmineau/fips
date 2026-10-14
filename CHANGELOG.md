# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`diagonal`, a constructor for diagonal matrices**, on `Matrix`,
  `CovarianceMatrix`, `ForwardOperator` and `MatrixBlock`:
  `CovarianceMatrix.diagonal(0.25, index=idx)` is the matrix with 0.25 on the
  diagonal and zeros elsewhere, labeled by `idx` on both axes. The values are a
  scalar, a 1D array or list (one value per element), or a `Series`, `Block`
  or `Vector`, which brings its own index, so
  `CovarianceMatrix.diagonal(S.variances)` is the diagonal part of `S`. With
  `sparse=True` the dense matrix is never formed. A `MatrixBlock` also takes
  its block names: `MatrixBlock.diagonal(0.01, index=idx, row_block="bias",
  col_block="bias")`.
  ([#61](https://github.com/jmineau/fips/issues/61))
- `InverseProblem` (and so `FluxProblem`) accepts a scalar for `prior_error`
  and `modeldata_mismatch`: one variance for every element of the state or of
  the observations. A `Series` or a `Vector` of variances with a `block` level
  gives one for each, as a `Series` did before.
  ([#61](https://github.com/jmineau/fips/issues/61))
- **Oracle tests** (`tests/test_oracle.py`), for every registered estimator:
  the posterior mean and covariance, Kalman gain, averaging kernel, DOFS,
  modeled observations and reduced chi-squared against the closed-form
  Gaussian posterior written independently in the test (both covariance
  forms, 20 seeds, several sizes, at 1e-10); `gamma` against a mismatch
  scaled by hand; diagonal covariances given as variances, a scalar or a full
  matrix; a multi-block problem read back block by block; a permutation test
  (each input in its own random order gives the same labeled posterior); and
  dot tests of `convolve` and of `ObsAggregator`'s aggregation. No library
  code changed. ([#82](https://github.com/jmineau/fips/issues/82))

### Changed

- **`Matrix`, `CovarianceMatrix`, `ForwardOperator` and `MatrixBlock` take 2D
  data only** (breaking). A scalar or 1D input now raises a `ValueError` that
  names the replacement, where the constructor used to decide from the shape of
  `data` what to build:
  - `CovarianceMatrix(series)` built a diagonal from a Series. Use
    `CovarianceMatrix.diagonal(series)`, with `name=` if the matrix should
    have one: it no longer takes the Series' name.
  - `Matrix(5.0, index=idx)` and `MatrixBlock(5.0, "r", "c", index=idx)` filled
    every entry with 5.0, which for a `CovarianceMatrix` has rank 1 and is not
    what "a covariance of 5" means. Use `.diagonal(5.0, index=idx)` for the
    diagonal, or pass the array, `np.full((n, m), 5.0)`, for a matrix of one
    value.
  - A 1D array or list raised already, from pandas or for its unnamed index;
    it raises the same `ValueError` now.

  ([#61](https://github.com/jmineau/fips/issues/61),
  [#50](https://github.com/jmineau/fips/issues/50),
  [#45](https://github.com/jmineau/fips/issues/45))
- `FluxInversionPipeline.filter_state_space` raises a `TypeError` that says what
  is wrong when it has to infer the frequency of flux times that are not
  datetimes (the config has no `flux_freq`). Before, `pd.infer_freq` ran on
  whatever the `time` level was: integers raised pandas's own `TypeError`,
  strings and mixed time zones a `ValueError`, and timedeltas were accepted and
  dropped every observation, since no interval of timedeltas holds an
  observation time. A `time` level of timestamps stored as objects is not
  affected; a `Vector` already reads it as datetimes.
  ([#43](https://github.com/jmineau/fips/issues/43))
- **`Matrix.scale` keeps the matrix's name, and a sum of two matrices keeps it
  when both have the same one.** The result of `scale` (on `Matrix`,
  `CovarianceMatrix` and `ForwardOperator`) and of `a + b` always had no name
  (`name` was `None`). A scaled matrix now has the name of the original, and
  `a + b` is named like `a` and `b` when their names are equal and has no name
  when they differ or either has none, as the sum of two pandas Series does. A
  sum with a scalar still has no name. The values, the index and the storage are
  as before.
  ([#72](https://github.com/jmineau/fips/issues/72))
- **`Matrix`, `CovarianceMatrix`, `ForwardOperator` and `MatrixBlock` turn a
  name that is not `None` into a string when they are built**, as `Block` and
  `Vector` do. `Matrix(data, name=5).name` is `"5"` where it was `5`, and a
  tuple or a float is its `str`. `None` stays `None`. Compare `matrix.name` with
  the string if you passed a number. `ObsAggregator.apply` names its result like
  its input, so it returns the string name too, and two `MatrixBlock`s named `1`
  and `"1"` are now duplicates when they are assembled into a `Matrix`, as two
  such `Block`s already were in a `Vector`.
  ([#72](https://github.com/jmineau/fips/issues/72))
- **`InversionPipeline.get_inputs` and `run` log their progress instead of
  printing it.** The line each step prints as it starts ("Loading
  observations..."), its elapsed time and the total are now `INFO` records of the
  `fips.pipeline` logger, with the same text. **With no logging configured they
  no longer appear**, in a terminal or in a batch job's output, because Python
  shows only warnings by default. Call `logging.basicConfig(level=logging.INFO)`
  to see them again, and read them from the log, not from standard output, if
  you captured it. `FluxInversionPipeline.summarize` still prints its report, and
  so does a subclass that prints in its own `run`.
  ([#8](https://github.com/jmineau/fips/issues/8))

### Fixed

- **`KroneckerError.build` gives the right covariance for any row order of
  its index.** It laid the Kronecker product of the marginal correlations over
  the rows as if they were the product of the marginal coordinates in the
  kernels' order, and never checked: the same labels sorted another way, by
  `(lat, lon, time)` against kernels given as `[time, (lat, lon)]`, gave entries
  off by up to 0.37 on a correlation scale, with no warning, and an index with
  a cell missing at only some times raised a broadcast error. Each entry is now
  `std_i * std_j` times the product of the marginal correlations between rows i
  and j, for any row order and any subset of the product. An index that is the
  product in the kernels' order, with or without the same cells left out at
  every time, gets the same matrix as before, bit for bit. Each kernel still
  receives the unique coordinates of its levels in order of first appearance.
  `build` raises a `ValueError` when two rows have the same coordinates on
  every kernel's level (two blocks on one grid would be perfectly correlated;
  this raised a broadcast error before) or a kernel returns a matrix of the
  wrong shape. It also holds one n x n array fewer while it builds (a 5.6 GB
  peak instead of 8.3 GB at n = 18,720, in the same time).
  ([#76](https://github.com/jmineau/fips/issues/76))
- **A sparse frame whose `fill_value` is neither 0.0 nor NaN raises a
  `ValueError`** when it is given to `Matrix`, `CovarianceMatrix`,
  `ForwardOperator`, `MatrixBlock`, or `ObsAggregator.apply` as the forward
  operator. It was accepted, and fips reads the implicit entries of a sparse
  matrix as zeros: on pandas 3, `Matrix.values` (which the solve reads) and the
  aggregation of a bare sparse operator took the implicit entries of a frame
  filled with 1.0 as 0.0, though `to_numpy()` gave the cells, and a frame that
  mixed dense columns with a sparse column of such a fill had the cells that
  fill stood for changed to 0.0 on both pandas 2 and 3. On pandas 2, `.values`
  raised a pandas error. **If you pass a sparse frame with another fill, build
  it with `fill_value=0.0` or hand over a dense frame** (`sparse=True` stores it
  sparse); `astype(pd.SparseDtype(float, fill_value=0.0))` does not convert it,
  it keeps the stored entries and turns the implicit ones into zeros. Frames
  filled with 0.0, and with NaN (read as 0.0), are accepted as before.
  ([#67](https://github.com/jmineau/fips/issues/67))
- `FluxInversionPipeline.summarize` leaves out the flux trend when the `time`
  level of the flux is not datetimes, where it raised `AttributeError: 'Index'
  object has no attribute 'year'`. The trend is a rate per year, so it needs
  datetimes, and only a flux with more than two timesteps has one: a pipeline
  whose flux times are integer steps, or intervals, failed at the end of `run`,
  after the solve. The rest of the report is unchanged, and the skip is logged
  at `INFO` level.
- `plot_comparison` accepts `None` as the error of a series when another series
  has one: `plot_comparison(a, b, errors=[None, err])` raised `KeyError` (and
  `errors=[None]` or all `None` raised `ValueError`). A series with no error is
  plotted without a band or error bars, and the others keep theirs. The error
  you pass is no longer renamed after its series, which also made
  `errors=[err, err]` (one error for two series) raise `KeyError`, and an
  unnamed series is matched with its own error by position.
  ([#57](https://github.com/jmineau/fips/issues/57))
- `FluxPlotter.concentrations` no longer warns on pandas 3 when called with its
  default windows. `rolling_window` and `gap_threshold` defaulted to `"30d"` and
  `"7d"`, and pandas 3 deprecates the lowercase `d` alias (`Pandas4Warning`);
  they default to `"30D"` and `"7D"`. A call that passes `"30d"` itself still
  warns: write `"30D"`.
  ([#55](https://github.com/jmineau/fips/issues/55))
- On pandas 3, a column of a sparse frame that stores every row kept
  `fill_value=NaN`, where every other column filled with 0.0, so a frame such
  as `DataFrame.sparse.from_spmatrix(csr_matrix([[1.0, 0.0], [3.0, 4.0]]))`
  handed to `Matrix` had mixed fills and warned ("Concatenating sparse arrays
  with multiple fill values"). The same happened to a covariance from
  `CovarianceBuilder.build(sparse=True)` or an operator from `ObsAggregator`
  with a column that has no zero. Every column now fills with 0.0. No cell
  changes: a column that stores every row has no implicit entry. pandas 2 was
  not affected.
  ([#64](https://github.com/jmineau/fips/issues/64))
- `MatrixBlock.reindex`, `round_index` and `copy` raised "MatrixBlock must have
  both row_block and col_block defined", as did `to_sparse` (it copies) and
  `to_dense` of a sparse block: the new block was built without its block names.
  They return a block with the same `row_block`, `col_block` and name, and
  `reindex(..., inplace=True)` and the `Matrix` classes were not affected.
  Present in 0.1.0b9.
  ([#65](https://github.com/jmineau/fips/issues/65))
- On pandas 3, a NaN stored in a sparse matrix was read as a zero, where the
  dense path raises "Data contains NaN values.". A NaN variance given to
  `CovarianceBuilder.build(sparse=True)`, a NaN in a sparse forward operator
  aggregated by `ObsAggregator`, and a sparse frame built with
  `DataFrame.sparse.from_spmatrix` from a matrix that stores a NaN and handed
  to `Matrix` or another 2D structure all came out as zeros, so an inversion
  could run with an element that has no uncertainty. They now raise the same
  `ValueError` as the dense path. The implicit entries of a sparse frame are
  still zeros. Since 0.1.0b8; pandas 2 was not affected.
  ([#63](https://github.com/jmineau/fips/issues/63))
- `plot_comparison` unpacks only a `Block` or a `Vector`. It took the `data`
  attribute of anything that had one, and pandas makes an index label or a
  column called `data` an attribute. A Series whose index had the label `data`
  raised `TypeError` as a series, `ValueError` as the truth and `AttributeError`
  as an error; an error DataFrame with a column `data` plotted that column
  instead of the one named like its series.
  ([#43](https://github.com/jmineau/fips/issues/43))
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
