# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

fips 0.2 replaces the pandas-backed structures of 0.1 with labeled linear
algebra: a `Space` of labels, values on it (`Vector`), matrices between spaces
(`Operator`, `Covariance`) kept in their cheapest form, covariances written as
models, a `Problem` that checks its labels once, and `fips.solve`, which returns
a `Posterior`. Almost every name of 0.1.0b9 changes: the
[migration guide](https://jmineau.github.io/fips/stable/migration.html)
(`docs/migration.rst`) gives each one and its replacement. The numbers of a
solve do not change: the analytic method gives the posterior the Bayesian
estimator gave, to 1e-10 and on the paper-1 reference run to 6e-14.
([#82](https://github.com/jmineau/fips/issues/82))

### Added

- **`Space` and `Vector`.** A `Space` is the labels of a state, of the
  observations or of one axis of a matrix: ordered named blocks, each labeled by
  its own pandas index, together one `MultiIndex` with the `block` level first
  (a level one block lacks is NaN on its rows, and every level keeps its dtype).
  `select(block)`, `slice(block)`, `offsets`, `from_index`, `align(other)` (an
  exact join that names the labels on one side only) and `round_labels`. A
  `Vector` is read-only values on a space: `from_series`, `from_blocks`,
  `v["flux"]` (a `Vector` of that block, labeled as it was given), `to_series`,
  `to_xarray`, `align`, `combine_first` and `round_labels`. ([#82](https://github.com/jmineau/fips/issues/82))
- **`Operator`, a matrix with a `Space` on each axis**, in its cheapest form (a
  dense array, a SciPy sparse matrix, a diagonal, Kronecker factors):
  `Operator(matrix, rows, cols)`, `Operator.from_frame(frame)`,
  `Operator.from_blocks({(row_block, col_block): block}, rows=, cols=)` (a
  labeled block is placed by its labels: a label it has no entry for is zero, a
  label the space lacks raises, a sparse block stays sparse), `op["co2",
  "flux"]`, `op @ vector`, `op @ op`, `+`, `-`, a scalar `*` or `/`, `T`,
  `align`, `round_labels`, `dense()` and `to_frame()`. Products and sums first
  match the labels they meet on, in any order. ([#82](https://github.com/jmineau/fips/issues/82))
- **`Covariance`, an `Operator` from a space to itself**, with
  `Covariance.diagonal(variances, space)` (kept as the diagonal),
  `Covariance.block_diag` (each part keeps its form; parts by block name
  too), `variances`, `solve`, `logdet`, `quadratic_form` and `sqrt`, each done
  the way its form makes cheap. ([#82](https://github.com/jmineau/fips/issues/82))
- **`fips.linalg`, the unlabeled forms**: `DenseOperator`, `SparseOperator`,
  `DiagonalOperator`, `BlockDiagonalOperator` and `KroneckerOperator`
  (a covariance kept as its factors, for any row order and any subset of the
  product), subclasses of SciPy's `LinearOperator` with `dense`, `diagonal`,
  `trace`, `solve`, `logdet` and `sqrt`; `A @ B` and `A + B` keep the cheapest
  exact form, and a sparse product past a quarter full is dense.
  `as_linear_operator` and `dottest`. A form copies its input and makes the copy
  read-only. ([#82](https://github.com/jmineau/fips/issues/82), [#6](https://github.com/jmineau/fips/issues/6))
- **Covariance models**, exported from `fips`: a correlation model times
  standard deviations, `model.with_std(std)`. `Exponential`, `Gaussian` and
  `Matern` decay with the distance along some levels (a datetime level with a
  time scale such as `"30D"`, numeric levels, or latitude and longitude with
  `metric="haversine"` or `"chordal"`); `Constant`; `Independent`, the identity;
  `Kronecker(a, b, ...)`, kept as its factors. `within=` keeps a model inside
  groups of labels and sparse between them (`within=["site",
  "obs_time.date"]`, `within="time.hour"`). Parameters are checked when a model
  is made. `with_std` takes the standard deviations as a `Vector` (the
  `Covariance` on its space, at once), a number, or a pandas Series on the
  labels of the block it is built on (lined up by label; a missing or extra
  label raises); a plain array raises, since it has no labels. Models add,
  `Covariance.block_diag` takes them by block name, and
  `build(space)` puts one on a space (`fips.covariance.CovarianceModel`); a new
  kind is a subclass of `fips.covariance.CorrelationModel`.
  `fips.covariance.distances.chordal_matrix`. ([#82](https://github.com/jmineau/fips/issues/82))
- **`Problem(prior, prior_cov, obs, obs_cov, forward, offset=None)`**, the
  inputs of a solve, their labels checked once by an exact join, each held in
  its own form; a covariance model is built on the prior's or the observations'
  space. `predict(x)` is `H x + c` for any state; `aggregate(W)` is the problem
  of aggregated observations (`W z`, `W H`, `W S_z W^T`, `W c`); `restrict(keep)`
  solves for part of the state and holds the rest at its prior (its `H x_0` moves
  into the offset). ([#82](https://github.com/jmineau/fips/issues/82))
- **`fips.solve(problem, method="analytic", **options)`** returns a `Posterior`;
  each method declares its options once (`fips.methods.analytic.AnalyticOptions`:
  `gamma`, positive and finite), an unknown method or option raises with the
  names it knows, and `fips.available_methods()` lists the methods. ([#82](https://github.com/jmineau/fips/issues/82), [#29](https://github.com/jmineau/fips/issues/29), [#12](https://github.com/jmineau/fips/issues/12))
- **`Posterior`**, the answer, labeled: `estimate` (and `mean`), `variances`,
  `cov(W)` and `averaging_kernel(W)` (of the state, or of aggregates of it
  without the `n x n` matrices; the averaging kernel of aggregates is
  `W A W*`, as IMI computes it), `kalman_gain`, `modeled_obs`, `innovation_cov`
  (`H S_0 H^T + S_z / gamma`), `desroziers(by=)` (a DataFrame of the
  diagnosed and the specified observation error variances), `leverage(x)`, the
  scalars `dofs`, `chi2`, `reduced_chi2`, `r2`, `rmse` and
  `uncertainty_reduction`, and `summary()`, a one-row DataFrame that stacks over
  many solves. `with_gamma(g)` and `with_data(obs)` reuse the factorization
  instead of solving again. A method's solution answers the
  `fips.posterior.Solution` protocol, so a new method gets all of it.
  ([#82](https://github.com/jmineau/fips/issues/82), [#22](https://github.com/jmineau/fips/issues/22), [#23](https://github.com/jmineau/fips/issues/23), [#8](https://github.com/jmineau/fips/issues/8))
- **The analytic method's information form, chosen by size.** With fewer
  state elements than observations (`n < m`) and a model-data mismatch that
  is diagonal, sparse or block diagonal, `fips.solve` factors the prior
  instead of the observations: with `S_0 = R R^T` (the form's own square root;
  through the factors' Cholesky factors for a Kronecker `S_0`), it
  eigendecomposes the `n x n` matrix `R^T H^T S_z^-1 H R` instead of the
  `m x m` one, holds `n x n` instead of `m x m`, and gives the same posterior
  (both forms are held to the closed-form oracle at 1e-10; `with_gamma` and
  `with_data` work in both). Otherwise, and when `S_0` has no Cholesky
  factor, the whitened form solves as before, bit for bit. There is no option:
  the choice is logged at DEBUG. The warning about a large solve now fires
  only for a whitened solve with more than 5,000 observations and more than
  twice as many observations as state elements, and says why the information
  form was not used (before, every solve with `m > n` and `m >= 1000`
  warned). `KroneckerOperator.sqrt` gives its root through the factors when
  its rows are the whole product.
  With `n = 300`, `m = 10,000` and a diagonal `S_z` on one core, a solve
  with the variances, DOFS and chi-squared takes 0.14 s at a 238 MB peak
  (194 s and 3.4 GB in the whitened form); at `n = 500`, `m = 5,000` a
  posterior holds 2 MB after the solve instead of 220 MB.
  ([#90](https://github.com/jmineau/fips/issues/90))
- **`fips.aggregate.sum` and `fips.aggregate.mean`**: an aggregation `W`, a
  sparse `Operator` from a space to groups of its labels (levels, a
  `pandas.Grouper` with a frequency, an `IntervalIndex` of bins, or a Series
  that maps a level's values to groups). **`fips.offset_block(obs_space, by)`**:
  a block of unknown offsets for the state, one per group of observations, with
  the forward block that adds each to its observations. ([#82](https://github.com/jmineau/fips/issues/82), [#28](https://github.com/jmineau/fips/issues/28))
- **Oracle tests** (`tests/test_oracle.py`): every method against the
  closed-form Gaussian posterior, and the permutation test (inputs in any order
  give the same labeled posterior). ([#82](https://github.com/jmineau/fips/issues/82))
- **Documentation for 0.2**: the README leads with a small problem from no
  particular field; the getting-started guide, the user guide, the terminology
  (each word defined once) and both example notebooks are rewritten on the new
  objects; a migration guide gives each name of 0.1 and its replacement; every
  public attribute has a reference page. ([#82](https://github.com/jmineau/fips/issues/82), [#14](https://github.com/jmineau/fips/issues/14))

### Changed

- **The solve whitens and keeps its factorization.** The analytic method
  whitens with the Cholesky factor of the observation covariance (or the square
  root of its diagonal), eigendecomposes one `m x m` matrix (the `n x n` matrix
  of the information form, under Added, when that is smaller and the
  observation covariance allows it), and derives the estimate, the variances,
  the DOFS and the chi-squared from the factors without forming anything
  `n x n` in the whitened form. A sum `H S_0 H^T + S_z` that roundoff has tipped off
  positive-definite no longer matters, but the observation covariance itself
  must be positive-definite (one that is not raises `LinAlgError`). A whitened
  solve with more than 5,000 observations and more than twice as many
  observations as state elements logs a warning. On an slv-like problem
  (n = 18,720, m = 2,858, a Kronecker prior covariance) a solve with the
  variances, DOFS and chi-squared took 6.7 s at a 1.6 GB peak (20.8 s and
  9.2 GB before) and each further gamma 0.11 s. ([#22](https://github.com/jmineau/fips/issues/22), [#23](https://github.com/jmineau/fips/issues/23))
- **The inputs are matched by label, exactly** (breaking). `InverseProblem`
  filled the labels a matrix lacked with zeros (with a "Partial overlap"
  warning), dropped the extra ones, and filled a misspelled block name with
  zeros; a `Problem` raises and names them. A covariance that lacks a label's
  variance raises the same way. A Jacobian that covers only some cells is
  placed with `Operator.from_blocks`. ([#24](https://github.com/jmineau/fips/issues/24), [#80](https://github.com/jmineau/fips/issues/80), [#82](https://github.com/jmineau/fips/issues/82))
- **Labels are kept as given** (breaking): no conversion to numbers (site codes
  `"010"` stay strings) and no rounding (`round_index=6` is gone); round on
  purpose with `round_labels`. Duplicate, missing and unnamed labels raise a
  `ValueError` that names them. ([#77](https://github.com/jmineau/fips/issues/77), [#78](https://github.com/jmineau/fips/issues/78))
- **New names, from one table** (breaking): `obs`, `prior`, `forward`,
  `prior_cov`, `obs_cov`, `offset` and `estimate` everywhere, the method
  `"analytic"` (was the estimator `"bayesian"`), and lowercase statistics
  (`dofs`, `r2`, `rmse`). `Problem` takes `(prior, prior_cov, obs, obs_cov,
  forward, offset)`, where `InverseProblem` took `(obs, prior, forward_operator,
  modeldata_mismatch, prior_error, constant)`: pass them by name.
- **Covariances take standard deviations**, where the error components took
  variances. ([#82](https://github.com/jmineau/fips/issues/82))
- **Vectors and operators are read-only and own their data**: each copies its
  input, so changing the original afterwards leaves it as it was. ([#82](https://github.com/jmineau/fips/issues/82))
- **A problem is not solved in place**: `fips.solve(problem)` returns the
  posterior and leaves the problem as it was. ([#12](https://github.com/jmineau/fips/issues/12))
- **The `flux` extra is renamed `viz`** and installs matplotlib only, for
  `plot_comparison`; cartopy and `h5py` are no longer installed, and `joblib` is
  no longer a dependency. ([#82](https://github.com/jmineau/fips/issues/82))
- **fips logs and never prints**: the solve's records go to the `fips.methods`
  logger, and statistics are returned (`Posterior.summary()`). ([#8](https://github.com/jmineau/fips/issues/8))
- **A kernel's `within=` is built for all its groups at once.** `Exponential`,
  `Gaussian`, `Matern` and `Constant` with `within=` (the correlated part of a
  model-data mismatch, as errors shared by a site's observations on one day)
  used to select each group's rows with pandas and evaluate the kernel on a
  small matrix, group by group. They now list every pair of labels in the same
  group, all groups together, and evaluate the kernel once on the distances of
  those pairs. The matrix is the same bit for bit, for every kernel, metric and
  kind of level. With 11,000 hourly observations in 2,752 groups of a site and a
  day, an `Exponential` builds in 0.009 s instead of 1.1 s and a `Constant` in
  0.008 s instead of 0.45 s; with 40,000 observations in 0.03 s instead of
  4.1 s and 1.6 s. `Problem()` for such a problem takes 0.04 s instead of 1.6 s.
  A few very large groups take about as long as before (an `Exponential` of 7
  groups of 1,570 observations 1.6 s instead of 1.2 s) in a fifth less memory.
  Nothing in the API changes.
- **The analytic solve's eigendecomposition uses LAPACK's `evd` driver, and
  redoes it with `evr` when there is not enough memory.** Both forms of the
  solve eigendecompose one symmetric matrix (`m x m` or `n x n`) with SciPy's
  `eigh`, whose default driver (`evr`) slows down when many eigenvalues are
  equal, as they are in the whitened form's `m x m` matrix when there are
  fewer state elements than observations (it has rank `n`, so `m - n`
  eigenvalues are zero). `evd` is not slowed by that: with `m = 6,000` on 8
  cores, a matrix with three quarters of its eigenvalues zero took 235.7 s with
  `evr` and 17.9 s with `evd`, and one with all distinct eigenvalues 20.0 s and
  19.2 s. `evd` needs more workspace: at `m = 10,000` the peak above the matrix
  itself is 1.52 GB with `evr` and 2.30 GB with `evd`. When SciPy cannot
  allocate it, it raises `MemoryError` before the decomposition starts, and
  the solve redoes it with `evr`, as before, logging one INFO line. There is
  no option. The estimates of the two drivers agree to 1.8e-13, not bit for
  bit: the eigenvectors are rounded differently.
- **A large product of a sparse matrix and a dense one runs on threads.**
  SciPy multiplies a sparse matrix by a dense one on a single thread, and the
  solve does it with `H S_0 H^T` (a dense matrix times `H^T`) and, for a sparse
  `H`, with `H @ X`. Above about ten million multiply-adds, `SparseOperator`
  now gives each thread a block of the sparse matrix's rows; the result is
  SciPy's bit for bit (every row's sum is the same, in the same order), in the
  same memory layout. The threads are the CPUs the process may run on (a Slurm
  job's, not the node's), and no more than `OMP_NUM_THREADS` when that is
  set, as for NumPy's BLAS: `OMP_NUM_THREADS=1` keeps a run on one thread, as
  before. On 8 threads, `H S_0 H^T` with 96,000 state elements and 2,751
  observations takes 1.6 s instead of 10.4 s, and with 18,720 state elements and
  10,003 observations 2.8 s instead of 25 s; the whole solve takes 17.4 s
  instead of 20.7 s and, with the 10,003, 87 s instead of 92 to 97 s. Nothing in
  the API changes.
- **The distance functions drop the options nothing used and are typed.**
  `haversine_matrix(lats, lons)` takes degrees and gives kilometers on a sphere
  of radius 6371 km: its `deg=` and `earth_radius=` are gone (for radians, pass
  `np.degrees(lats)`), and `time_diff_matrix(times)` always gives the absolute
  differences: its `absolute=` is gone. The fips kernels passed none of them.
  `haversine_matrix` and `chordal_matrix` share the one half-angle term they
  both write, and every distance function has annotations. No distance
  changes. ([#82](https://github.com/jmineau/fips/issues/82))

### Removed

Each removed name and its replacement is in the migration guide.

- **The 0.1 structures**: `Block`, the old `Vector`, `Matrix`, `MatrixBlock`,
  `ForwardOperator` and `CovarianceMatrix`, with `fips.base` (`Structure`,
  `Structure1D`, `Structure2D`, the block mixins, `xselect`, `Pickleable`),
  `fips.matrix`, `fips.operators`, `fips.indexes` and `convolve`. Use `Vector`,
  `Operator` and `Covariance`; save with `pickle`. 0.1 pickles do not load.
  ([#82](https://github.com/jmineau/fips/issues/82), [#77](https://github.com/jmineau/fips/issues/77))
- **`InverseProblem`, `Estimator`, `BayesianSolver`, `ESTIMATOR_REGISTRY` and
  `available_estimators`** (`fips.estimators`): use `Problem`, `fips.solve` and
  `available_methods`. ([#82](https://github.com/jmineau/fips/issues/82), [#12](https://github.com/jmineau/fips/issues/12))
- **The error components and kernel factories**: `ErrorComponent`,
  `DiagonalError`, `BlockDecayError`, `KroneckerError`, `CovarianceBuilder`,
  `ConstantCorrelation`, `RaggedTimeDecay`, `GridTimeDecay` and
  `GridSpatialDecay`; `fips.metrics` moved to `fips.covariance.distances`. The
  covariance models give the same matrices, bit for bit. ([#82](https://github.com/jmineau/fips/issues/82))
- **`ObsAggregator` and `integrate_over_time_bins`** (`fips.aggregators`): use
  `fips.aggregate` and `Problem.aggregate`. ([#82](https://github.com/jmineau/fips/issues/82))
- **`InversionPipeline`, the subpackage `fips.problems` (`FluxProblem`,
  `FluxInversionPipeline`, `FluxPlotter`) and `fips.filters`**: fips knows no
  science domain. A flux inversion is a `Problem` whose blocks are read by name;
  its workflow and its maps are the caller's. ([#82](https://github.com/jmineau/fips/issues/82), [#8](https://github.com/jmineau/fips/issues/8), [#9](https://github.com/jmineau/fips/issues/9), [#10](https://github.com/jmineau/fips/issues/10), [#11](https://github.com/jmineau/fips/issues/11), [#13](https://github.com/jmineau/fips/issues/13))
- **`fips.visualization.compute_credible_interval` and `plot_error_norm`**:
  `plot_comparison` is the one plot. ([#82](https://github.com/jmineau/fips/issues/82), [#20](https://github.com/jmineau/fips/issues/20))

### Fixed

Bugs of 0.1.0b9 that the objects replacing them do not have:

- `Estimator.leverage` raised on every state vector; `Posterior.leverage`
  gives the leverage matrix. ([#81](https://github.com/jmineau/fips/issues/81))
- `KroneckerError` built a wrong covariance, with no warning, when the rows
  were not in the order of its kernels; a `Kronecker` model is right for any
  order and any subset of the product. ([#76](https://github.com/jmineau/fips/issues/76))
- Two labels that rounding made equal became one, and the posterior had fewer
  entries than the state; duplicate labels now raise. ([#78](https://github.com/jmineau/fips/issues/78))
- A sparse prior error or model-data mismatch raised or returned an
  `np.matrix`; dense and sparse inputs now work in any combination. ([#79](https://github.com/jmineau/fips/issues/79))
- A pandas-sparse frame with a fill other than 0 or NaN, or with a stored NaN,
  was read with zeros in their place; `Operator.from_frame` raises. ([#67](https://github.com/jmineau/fips/issues/67), [#63](https://github.com/jmineau/fips/issues/63))
- A block selected out of a vector or a matrix came back with float labels for
  integer ones, or with the other blocks' levels; a block now comes back
  labeled as it was given. ([#46](https://github.com/jmineau/fips/issues/46), [#82](https://github.com/jmineau/fips/issues/82))
- A block of one element could not be built from a Series, and 2-D data gave a
  pandas error on pandas 2. ([#38](https://github.com/jmineau/fips/issues/38), [#51](https://github.com/jmineau/fips/issues/51))
- `plot_comparison` refused `None` as the error of one series, and took the
  `data` attribute of anything that had one. ([#57](https://github.com/jmineau/fips/issues/57), [#43](https://github.com/jmineau/fips/issues/43))
- `ObsAggregator` with one frequency key gave a row of zeros for an empty bin
  and mixed up the bins of labels out of time order; `fips.aggregate` does not.
  ([#98](https://github.com/jmineau/fips/issues/98))
- The README's multi-block example raised. ([#93](https://github.com/jmineau/fips/issues/93))
- **Labels at every door** (the branch review's fixes;
  [#82](https://github.com/jmineau/fips/issues/82)):
  `Covariance` refuses a dense or sparse matrix that is not symmetric (its
  largest `|A - A^T|` past 1e-10 of its largest entry; the message names the
  two entries), and `KroneckerOperator` a factor that is not: an upper
  triangle used to solve silently to a wrong answer. `Operator(op, rows,
  cols)`, `Covariance(cov, space)` and `Vector(v, space)` refuse a labeled
  object, whose labels they ignored, and point to `align` or `.data` /
  `.values`. `plot_comparison` draws `truth` by label, as it does `errors`
  (it drew it by position). `Operator.from_blocks` without `rows`/`cols`
  raises when two labeled matrices of a block have different labels, in
  either order of the keys (the first one's labels won). `v[0]` finds a
  block named `0` (kept as `"0"`); `Covariance.block_diag(name=model)` raises
  that `name` is the covariance's name; a mapping to NaN in `aggregate`
  names the mapping; `Exponential("lag", "1h")` and the other decay kernels
  take a timedelta level.
- `haversine_matrix` gave NaN for some exactly antipodal points: the
  half-angle term is clipped to [0, 1] against roundoff, as `chordal_matrix`'s
  already was. ([#108](https://github.com/jmineau/fips/issues/108))
- **Leftovers of the posterior redesign** (the branch review's fixes;
  [#82](https://github.com/jmineau/fips/issues/82)): the arrays of the
  analytic method's factors (`Whitened`) are read-only, also after a pickle or
  a copy (they were writable and shared by `with_gamma` and `with_data`).
  `with_data` builds its problem with the constructor, and one `Operator._on`
  puts an operator on two spaces. No number changes.
- **The `fips.linalg` forms pickle and copy on SciPy 1.18.** A
  `KroneckerOperator` (and so a `Covariance`, `Problem` or `Posterior` holding
  one) failed to pickle with "cannot pickle 'module' object", and a dense,
  sparse, diagonal or block diagonal form that had been through a pickle or a
  copy raised on its first product. SciPy 1.18 keeps its array namespace in
  each operator; a form's pickle now leaves SciPy's own attributes out and
  loading sets them again, so a pickle written on any SciPy version loads on
  any other. The lock's SciPy is 1.18.1 on Python 3.12 and newer (1.17.1 on
  3.11), so CI runs both. ([#113](https://github.com/jmineau/fips/issues/113))
- **`Posterior.r2` is the R² of the enhancement**, `corr(z - c, H x_hat)²`:
  the observations less the offset `c`, against `H x_hat`. 0.1.0b9's
  `Estimator.R2`, documented as `corr(z, H x_hat)²`, computed
  `corr(z, H x_hat + c)²`, the offset included, so a background that varies
  from day to day raised it whatever the fit of the state: with 200 synthetic
  observations and a background of standard deviation 1, it gave 0.78 for a
  fit whose enhancement R² is 0.27. Without an offset, or with a constant one,
  the number is the same as before.
  ([#116](https://github.com/jmineau/fips/issues/116))

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
