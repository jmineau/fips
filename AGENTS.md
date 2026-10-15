> **Keep this file current.** If you change the module layout, the build/test
> commands, or learn a new invariant or gotcha, update the matching section in
> the same change. A section that no longer matches the code is worse than no
> section: fix it or delete it.
>
> Personal or machine-specific notes (local paths, cluster setup) belong in an
> untracked file, not here. Anything matching `*.local.md` is gitignored for
> this (e.g. `CLAUDE.local.md`); add your tool's local files to `.gitignore` if
> they are not covered. Some agents stop reading `AGENTS.md` once a local
> instruction file exists, so import or reference it from yours.

# AGENTS.md — Developer and Agent Guide for fips

`fips` is the **Flexible Inverse Problem Solver**: a Pythonic framework for
solving *linear* Bayesian inverse problems where state and observation spaces
are organized as labeled (`pandas.MultiIndex`) blocks. It computes analytical
MAP estimates (no sampling) for `y = Hx + error`. Beta software (v0.1.0b*) —
API may evolve.

PyPI name and import name are both `fips`. Source lives in `src/fips/`.

## What this package is *not*

- Not a sampler (no MCMC). Estimators are analytical / regularized.
- Not nonlinear-problem aware. The framework assumes linearity around the
  prior; nonlinear iterations are the caller's responsibility.
- Not opinionated about file formats. Inputs are pandas/xarray; serialization
  is via `pickle` through the `Pickleable` mixin.

## Module layout

```
src/fips/
  __init__.py        public exports (Space, Field, Operator, Covariance, the
                     covariance models, Block, Vector, Estimator,
                     InverseProblem, Posterior, solve, InversionPipeline,
                     convolve)
  space.py           Space — ordered named blocks, the padded MultiIndex,
                     select, from_index, exact align, round_labels
  field.py           Field — values on a Space (read-only float array)
  operator.py        Operator — a matrix (a linalg form) with a Space on each
                     axis: from_frame, from_blocks, op[r, c], @ / + by exact
                     join, align, to_frame; convolve(state, H)
  covariance.py      Covariance — Operator on one space: diagonal, block_diag
                     (in order, or by block name with models), variances,
                     solve, logdet, quadratic_form, sqrt
  correlation.py     CorrelationModel (base; with_std), Independent, Kronecker
  kernels.py         the kernels: Exponential, Gaussian, Matern, Constant, and
                     within= (zero between groups of labels, sparse)
  models.py          CovarianceModel — a covariance waiting for its space
                     (with_std of a scalar or array, sums, block_diag with a
                     model); build(space) gives the Covariance
  linalg.py          the unlabeled forms the solvers compute with:
                     LinearOperator (scipy LinearOperator) and Dense-,
                     Sparse-, Diagonal-, BlockDiagonalOperator;
                     as_linear_operator, dottest, _solve_psd
  kronecker.py       KroneckerOperator — a covariance kept as its factors
  base.py            Pickleable, Structure/Structure1D, the block mixins —
                     what the deprecated Block and Vector build on
  vector.py          Block, Vector — 1D labeled containers (state, obs);
                     deprecated for Field, and built through Space and Field
  estimators.py      Estimator ABC, ESTIMATOR_REGISTRY, available_estimators();
                     BayesianSolver, which factors through _whitened.py
  _whitened.py       Whitened — the factorization BayesianSolver keeps: the
                     estimate, variances, gain, DOFS, chi-squared for any gamma
  problem.py         InverseProblem — top-level orchestrator that combines
                     obs, prior, H, S_0, S_z; solve(problem) fits an
                     Estimator and returns a Posterior
  posterior.py       Posterior — the estimator's outputs, labeled with the
                     problem's state and obs spaces, each built once
  pipeline.py        InversionPipeline ABC — workflow template for repeated
                     inversions (load data, build covariances, solve)
  aggregators.py     Time/space binning (e.g. integrate_over_time_bins)
  filters.py         Selection helpers (e.g. enough_obs_per_interval)
  indexes.py         overlap check and round_index
  metrics.py         haversine_matrix, chordal_matrix, time_diff_matrix
  visualization.py   Generic plotting helpers
  py.typed           ships type hints
  problems/          domain-specific problem subpackages
    flux/            atmospheric flux inversion (see below)
tests/               pytest suite (no integration markers in use)
docs/                Sphinx (pydata-sphinx-theme + numpy-style docstrings)
archive/             local-only, not tracked
```

### `fips.problems.flux`

Specialization for atmospheric flux inversion. Public exports:

- `FluxProblem` — `InverseProblem` subclass with flux-specific defaults
- `FluxInversionPipeline` — `InversionPipeline` subclass
- `FluxPlotter` — visualization

The `flux` extra pulls in `cartopy`, `h5py`, `matplotlib`. fips does not
build transport Jacobians: PYSTILT's `project.jacobian()` does, and the
caller puts a `Space` on each axis of its sparse matrix (`Operator(J.data,
rows, cols)`).

## Public API summary

From `fips.__init__`:

| Symbol | What it is |
|---|---|
| `Space`, `Field` | the labels of a vector (named blocks), and values on them |
| `Block`, `Vector` | 1D labeled data (state vector, observations); deprecated for `Field` |
| `Operator` | a matrix with a `Space` on each axis, in its cheapest form (H, an aggregation) |
| `Covariance` | an `Operator` from a space to itself (`S_0`, `S_z`), with solve/logdet/sqrt |
| `Exponential`, `Gaussian`, `Matern`, `Constant`, `Independent`, `Kronecker` | correlation models; `model.with_std(std)` is a covariance |
| `convolve` | apply H to a state vector given as a Series, by label |
| `Estimator`, `available_estimators` | swappable solvers via registry |
| `InverseProblem` | orchestrator; `solve()` keeps a `Posterior` as `result` and returns the problem |
| `solve`, `Posterior` | `fips.solve(problem)` returns the labeled posterior and diagnostics |
| `InversionPipeline` | repeatable workflow template |

## Mental model

1. **Wrap your inputs in Blocks.** Observations and state become `Block`s
   (or lists of `Block`s for multi-source obs / multi-region state). Each
   Block carries a `pandas.MultiIndex` describing dims (time, lat, lon,
   site, etc.).
2. **Express H as an `Operator`.** `Operator.from_blocks({(obs_block,
   state_block): frame})` stitches a block-structured Jacobian together;
   `Operator(matrix, rows, cols)` labels a matrix you have (a SciPy sparse one
   stays sparse); `Operator.from_frame` reads a labeled DataFrame.
3. **Build covariances from models.** A correlation model and standard
   deviations: `Kronecker(Exponential("time", "30D"), Exponential(["lat",
   "lon"], 20.0, metric="haversine")).with_std(std)` (a Kronecker prior error
   stays as its factors), `Independent().with_std(s) + Exponential("obs_time",
   "2.8h", within=["site", "obs_time.date"]).with_std(t)`, parts by block in
   `Covariance.block_diag(flux=..., bias=...)`; `.build(space)` puts a model on
   the labels (a `Field` std gives the `Covariance` at once). Or
   `Covariance.diagonal(variances, space)` and `Covariance.from_frame`. `H`,
   `S_0` and `S_z` may each be dense or sparse, and the solve keeps each in its
   form (see "Sparse paths").
4. **Hand everything to `InverseProblem`.** It validates index alignment;
   `fips.solve(problem)` picks an estimator from `ESTIMATOR_REGISTRY` and
   returns a `Posterior` (estimate, covariance, diagnostics such as
   `reduced_chi2`), and `problem.solve()` keeps that posterior as
   `problem.result`, whose outputs `problem.posterior` and friends forward to.
5. **For repeated inversions, subclass `InversionPipeline`** so loading,
   covariance build, and solve are templated.

## Invariants to respect

- **Names.** `Structure1D`, `Field` and `Operator` convert a name that is not
  `None` to a string, so a name is one spelling everywhere; an unnamed object
  keeps `None`. A sum of two operators keeps the name when both have the same
  one and is `None` otherwise (pandas' rule for a sum of two Series).
- **MultiIndex preservation.** The whole design hinges on labels surviving.
  Do not silently `.values` or `.to_numpy()` away from Series/DataFrame.
- **Labels are kept as given.** Nothing converts labels to numbers or rounds
  them unless the caller asks (`round_index=`, `Space.round_labels`); duplicate,
  missing (NaN) and unnamed labels of a block raise.
- **NaN-only levels.** Assembling blocks whose indexes have different levels
  fills the missing levels with NaN, and `Space` is the one place that does it
  in both directions: `Space.index` pads (keeping each level's dtype, through
  the MultiIndex codes) and `Space.from_index` / `Space.select` strip. Vectors,
  fields and both axes of every operator go through it; `op[r, c]` gives a
  block on single-block spaces, labeled as the block was given.
- **An operator's labels are beside its matrix.** `Operator.data` is a form of
  `linalg.py` with rows and columns in the order of `rows` and `cols`. Every
  product, sum and placement puts the other side in this side's order with
  `Space.align` (an exact join) before computing, and a method that changes
  labels or data builds its result with `_like` (or `_build` in a
  constructor), which keeps a `Covariance` a `Covariance` when both axes are
  one space. Reorder through `form._select(rows, cols)`, which keeps a
  Kronecker or block-diagonal form when the rows and columns are the same
  positions.
- **Block alignment.** `InverseProblem.__init__` puts the obs and prior in
  their `Space` order (each block's rows together) and places `H`, `S_0` and
  `S_z` on those spaces with zero fill (`problem._place`): labels missing within
  a known block are zeros with a "Partial overlap" warning (a STILT Jacobian
  covers only the cells a footprint touched, so this must stay). It raises on a
  matrix block name that matches no block of the space and on duplicate obs or
  state labels, before and after `round_index`. `Posterior` labels the
  estimator's arrays with `Space.from_index` of the problem's indexes, and
  raises if a block's rows are apart.
- **A sparse matrix is a SciPy matrix, never a pandas-sparse frame.** The
  one place a pandas-sparse frame enters is `Operator.from_frame` (and
  `ObsAggregator.apply` with a frame), through `operator._frame_matrix`: the
  entries a sparse column does not store are zeros, so its fill value must be
  0 or NaN (pandas 3's `DataFrame.sparse.from_spmatrix` fills with NaN,
  pandas-dev/pandas#59212); any other fill raises, and a NaN stored in a sparse
  column raises as one in a dense column does (`tests/test_sparse.py`).
  `to_frame()` is dense.
- **Sparse paths.** Covariances and large operators may be sparse. Don't
  densify in helpers without a sparse fallback — flux-scale problems are
  GB-class. Compute with `Operator.data` (a `DenseOperator`, `SparseOperator`,
  `DiagonalOperator`, `BlockDiagonalOperator` or `KroneckerOperator`), never
  with `dense()` or `to_frame()`. Products and sums keep the cheapest exact form
  (the rule is in `linalg.py`'s docstring). The tests are the dense/sparse
  combinations in `tests/test_linalg.py` and the oracle's sparse closed form.
  The estimator keeps its matrices as forms under private names (`_H_op`,
  `_S_0_op`, `_S_z_op`, `_S_z_orig_op`) and computes only with those; its
  public `H`, `S_0` and `S_z` give an array (read-only) or a CSR matrix, so
  never compute with them inside the estimator.
- **Covariance models.** A model is a frozen dataclass whose parameters are
  checked in `__init__`; it reads its levels from the labels of the space it
  is built on (`models._coords`, which raises when a block lacks one) and gives
  its correlation in its cheapest form (`_form`: `Independent` a diagonal, a
  kernel `within` groups sparse, `Kronecker` a `KroneckerOperator` with each
  factor evaluated once on the unique coordinates of its levels, in first
  appearance order). `with_std` scales that form without forming the matrix
  (`models._scaled`). Each covariance the removed builders made is pinned in
  `tests/test_covariance.py` (`tests/data/old_builders.npz`, written on pandas
  3; bit for bit there, and to the last bit on pandas 2, whose time
  differences are in nanoseconds): keep
  the order of the products (`(std_i * C_ij) * std_j`) and the distance
  functions of `metrics.py` when changing a kernel.
- **`Pickleable` contract.** Anything user-facing should round-trip through
  pickle. New stateful classes should inherit from `Pickleable` (in `base.py`)
  or document why not.
- **Logging, not printing.** Every module uses
  `logger = logging.getLogger(__name__)`; `__init__.py` attaches a
  `NullHandler`. Don't `print()` in library code.
- **Docstring convention.** `pydocstyle` (numpy convention) is enforced via
  ruff (`D` rules). Public functions need at least a one-line docstring.

## Dev commands

Driven by `just` + `uv`; CI runs the same recipes.

| Command | What it does |
|---|---|
| `just sync` | `uv sync`: fips, its `flux` extra, and the dev tools |
| `just quality-check` | `lint` + `type-check` + `docstr` + `test`: what CI checks |
| `just test` | pytest in parallel (up to 8 workers), skipping `network`/`slow`; extra args go to pytest (`-n 0` for serial) |
| `just lint` / `just format` | ruff check and format check / fix and format |
| `just type-check` | pyrefly, against `pyrefly-baseline.json` |
| `just build-docs` | Sphinx HTML into `docs/_build`, running every example; warnings are errors |
| `just docs-serve` | live docs preview on port 8000 |
| `just changelog` | draft CHANGELOG entries from the commits since the last tag |
| `just pre-commit` | every hook on every file |
| `just dist` | build and check the sdist and wheel |
| `just release X.Y.Z` | tag and push a release (the maintainer runs it, never an agent) |
| `just clean` | wipe build, dist, coverage, caches, `__pycache__`, docs build |

Workflows in `.github/workflows/`: `tests.yml` (Linux/macOS/Windows x Python
3.11-3.14), `quality.yml`, `docs.yml` (versioned docs: it pushes the gh-pages branch, which GitHub Pages serves; `dev/`
from main, one folder per release, `stable/`), and `publish.yml` (a `vX.Y.Z`
tag builds the release, uploads it to PyPI by trusted publishing, and creates
the GitHub Release from its CHANGELOG section). The version comes from git tags
(setuptools-scm); never bump, tag or push unless the maintainer asks.

The tooling comes from
[jmineau/python-template](https://github.com/jmineau/python-template) via copier
(`.copier-answers.yml`); `copier update` pulls in template changes.

## Conventions and tooling

- **Python**: 3.11+ (`requires-python`; ruff takes its target from it).
- **Linting**: ruff selects `E, F, UP, B, SIM, I, D, D213, NPY, RUF100` and
  ignores `E501, D200, D212, D400`; tests and the example notebooks may use
  numpy's legacy seeded RNG (`NPY002`). The whole repo is linted, tests included
  (see `pyproject.toml`).
- **Types**: pyrefly on `src/`, checked against `pyrefly-baseline.json`: the
  errors that predate the switch from pyright. New errors fail the check; prune
  fixed ones with `uv run pyrefly check --prune-baseline`. `py.typed` is
  shipped — keep new public API fully typed. Fix a listed error where the type
  comes from (a correct annotation, a narrower return type), not with
  `typing.cast` or `Any`. Where pandas-stubs are wrong or too narrow (pandas
  takes the call, as with `DataFrame.sparse`, which they leave untyped), put
  `# pyrefly: ignore[<code>]` on the line above the error, after a comment saying
  why, and keep it off the baseline. The aim is an empty baseline.
- **Docstrings**: numpy convention, format docstring code blocks
  (`docstring-code-format = true`).
- **Docs**: `docs/reference/` groups the API by topic, as pandas does. Each
  class also gets its own page, with tables of its attributes and methods, and
  each member a page of its own; a subclass lists what it defines and links to
  what it inherits (`docs/_templates/autosummary/` and `docs/_ext/api_pages.py`,
  from python-template). Document a new base class in `reference/base.rst`, or
  its inherited members show as plain names. Give each public property a
  docstring: an undocumented one shows an empty row. Examples run when the docs
  build (see "Examples in the docs").
- **Coverage**: configured in `pyproject.toml`; `tests/` is the only
  testpath. `coverage.exclude_also` skips `__repr__`, abstract methods,
  `if __name__ == "__main__":`, etc.
- **Dependency groups**: runtime deps in `[project]`; the `flux` extra adds
  cartopy/h5py/matplotlib; the `dev` group is what `just sync`
  syncs.

## Examples in the docs

Examples run when the docs build, so they show real output, and the build fails
when one breaks.

- **A figure in a docstring:** put a `.. plot::` directive in the Examples
  section. Its code runs (doctest `>>>` lines work), and the figure appears
  under it.

  ```rst
  Examples
  --------
  .. plot::

     >>> import matplotlib.pyplot as plt
     >>> _ = plt.plot([0, 1, 2], [0, 1, 4])
  ```

- **Code in an `.rst` page:** a `.. ipython:: python` block runs, and shows each
  line with its output, as in an IPython session. Changing a
  `.. code-block:: python` to `.. ipython:: python` is enough to make it run. For
  a figure, put `@savefig name.png` on the line above the plotting call.

  ```rst
  .. ipython:: python

     import matplotlib.pyplot as plt
     import fips

     fips.__version__

     @savefig squares.png width=5in
     plt.plot([0, 1, 2], [0, 1, 4]);
  ```

  A block that raises or warns fails the build; `:okexcept:` or `:okwarning:`
  under the directive allows one that is meant to. All `.rst` pages share one
  session, so each page imports and defines what it uses.
- **Code cells in a Markdown page:** a page (`.md`) under `docs/` can mix prose
  with code cells that run, each showing what it prints and plots, as in a
  notebook. The header makes the cells run:

  ````markdown
  ---
  file_format: mystnb
  kernelspec:
    name: python3
  ---

  # Getting started

  Some prose.

  ```{code-cell}
  import fips

  print(fips.__version__)
  ```
  ````

- **A notebook:** an `.ipynb` under `docs/` (the worked examples are in
  `docs/examples/`), listed in a toctree like any page. Commit it without
  outputs: the build runs every cell.
- **When they run:** `just build-docs` runs everything, as CI does.
  `just docs-serve` reruns a Markdown page or notebook only when its code
  changes, and an `.rst` page's blocks whenever that page is rebuilt. A cell
  times out after 30 s; for a slower one, set `mystnb: {execution_timeout: 120}`
  in the page's header, or `"mystnb": {"execution_timeout": 120}` in the
  notebook's metadata.
- **Data:** the docs build on GitHub Actions, so an example uses synthetic data,
  a small file in the repository, or a public download, never a machine-specific
  path. Seed random numbers so the figures don't change from build to build.
- **Optional dependencies:** the `dev` group already lists `fips[flux]` (cartopy,
  h5py, matplotlib), so `uv sync` installs them here and in CI. If an example
  needs another extra, add `fips[extra]` the same way.
- **A notebook that can't run in CI** (it needs data only one machine has) is
  committed with its outputs and `"mystnb": {"execution_mode": "off"}` in its
  metadata. The page then shows the outputs from its last run, so rerun it when
  the code it uses changes.

## Dependency policy — what fips owns and what it delegates

Decided 2026-10-05 after surveying the inverse-problem libraries in other
fields (pyemu, pyOptimalEstimation, CoFI, CUQIpy, hippylib, pylops, MNE, CIF,
IMI, FLEXINVERT, LUMIA). None of them provides a labeled
linear-Gaussian solve with posterior covariance at flux scale, and the ones
that come close use dense `inv`. So:

- **fips owns the core solve.** The analytic estimator is ~50 lines of
  `scipy.linalg` (Cholesky, triangular solves, `eigh`); the value of fips is
  the labeled/block layer around it, and routing arrays through an external
  solver is exactly where labels get lost. Do not add a dependency to replace
  `BayesianSolver`; improve it in place.
- **Delegate factorizations and operator plumbing to scipy.** `scipy.linalg`
  for dense factorizations, `scipy.sparse` for storage,
  `scipy.sparse.linalg.LinearOperator` as the matvec abstraction when
  covariances or H become factored/matrix-free, and
  `scipy.sparse.linalg.eigsh`/`lobpcg` for low-rank posteriors before porting
  any randomized eigensolver.
- **pylops is read, not installed or copied** (LGPL; decided 2026-10-14): fips
  writes the few operators it needs on `LinearOperator`, as `KroneckerOperator`.
- **Optional, not required:** `GSTools` (covariance models, only if
  `kernels.py` outgrows its exponential, Gaussian and Matérn kernels). If adopted, it goes in an extra,
  never in `[project]` dependencies.
- **Never write a sampler.** If MCMC lands, use PyMC or CUQIpy for
  sampling and ArviZ for diagnostics, as `openghg_inversions` does. Same for a
  bounded/variational optimizer: `scipy.optimize`, not custom code.
- **The reference libraries are test oracles, not dependencies.** Their
  closed-form identities (pyOE's χ² eigenvalue forms, hippylib's `d/(1+d)`
  posterior, pyemu's Schur posterior, the Yadav & Michalak supplement code) are
  for cross-checking new estimators on small random problems in `tests/`.

## Common workflows

### Single-block solve (one obs source, one state block)
See README "Single-block" example. Pass `obs`, `prior`, `forward_operator`,
`prior_error`, `modeldata_mismatch` directly to `FluxProblem(...).solve()`.

### Multi-block solve (heterogeneous obs)
Wrap each obs source in its own `Block`. Build `H` with
`Operator.from_blocks`, one entry per (row_block, col_block) pair. A
block-diagonal `modeldata_mismatch` (`Covariance.block_diag`) is the common
pattern (no inter-source error covariance). See README "Multi-block" example.

### Adding a new estimator
1. Subclass `Estimator` in `estimators.py`.
2. Implement the abstract solve method.
3. Register via the `EstimatorRegistry` decorator (`@ESTIMATOR_REGISTRY.register("name")`).
4. Now available via `available_estimators()` and selectable in `fips.solve`
   and `InverseProblem.solve`. `Posterior` labels the outputs every estimator
   has (`x_hat`, `S_hat`, `y_hat`, `y_0`, `S_y0`, `K`, `A`, `desroziers`) and
   knows which index goes on each axis; an output a new estimator adds is
   reached unlabeled through `post.estimator` until `Posterior` gets a
   property for it.
5. `tests/test_oracle.py` is parametrized over the registry, so the new
   estimator is held to the closed-form Gaussian posterior and the
   permutation test at once. An estimator whose answer is not that posterior
   (bounded, lognormal, sampled) says so there, with its own reference.

### Adding a new problem domain
1. Create `src/fips/problems/<domain>/` with `__init__.py` re-exporting the
   public surface.
2. Subclass `InverseProblem` for the domain-specific orchestration; subclass
   `InversionPipeline` for the workflow template.
3. Mirror the `flux` package's layout (`problem.py`, `pipeline.py`,
   `visualization.py`).

## Gotchas

- **An `Operator` takes its matrix by position, a DataFrame by
  `from_frame`.** The constructor refuses a pandas object, since its labels
  would be ignored; a diagonal is `Covariance.diagonal(values, space)` (kept
  as a `DiagonalOperator`). An `Operator` multiplies an `Operator` or a
  `Field`, never an array (no labels to check): compute by position with
  `op.data`.
- Check an optional pandas index with `is None`, since evaluating its truth
  value raises an error.
- **The pyrefly baseline matches an error by file, kind and column**, not by
  line or message (seen with pyrefly 1.3.2). A new error is therefore hidden
  when a listed one has the same kind at the same column of that file, and
  `--prune-baseline` keeps a fixed entry for as long as another error shares
  its file, kind and column: remove that one by hand. To see every error with
  its line, the listed ones included, run
  `uv run pyrefly check --baseline-error-level error --output-format min-text`;
  add `--min-severity warn` for the warnings, which `pyrefly check` does not
  show. pyrefly types `DataFrame.columns` as `Index[str]`, so it calls the
  `str(col)` in `plot_comparison` an unnecessary conversion, but a column label
  is an integer when a Series has no name: keep the `str()`.
- **SciPy is not annotated**: pyrefly takes the private class a `LinearOperator`
  default returns as the type an override must return, and a sparse `shape` as
  possibly `None`, and finds no `__getitem__` on a sparse array. `linalg.py`
  marks those lines with an ignore and the reason; a new form that
  overrides only `_transpose` needs none. `form.T` is typed as SciPy's private
  transposed class: call `form._transpose()` to keep a fips form.
- **pyrefly (1.3.2) reads `type[X]` written inside a base class's arguments as
  `type[Any]`**: after `class R(dict[str, type[E]])` a lookup in `R()` is
  `type[Any]`, though `typing.Type[E]` or an alias of `type[E]` gives
  `type[E]`. `EstimatorRegistry` names its value type through an alias
  (`_EstimatorClass`) for this reason; do the same where a subclass of a
  container holds classes.
- `convolve(state, H)` returns a labeled `pandas.Series`, and `H @ field` a
  `Field`, not a numpy array. Don't expect raw arrays out of public APIs.
- `reduced_chi2` lives on the *estimator* and the *posterior*, not the problem
  (`problem.estimator.reduced_chi2`, `problem.result.reduced_chi2`).
- `InverseProblem` pickles its estimator under `_estimator`, not the
  `Posterior`, and rebuilds the posterior on load, so a pickle reads the same
  with or without `posterior.py`. A `Posterior` pickles the estimator and the
  two indexes only; its labeled outputs are rebuilt when asked for. A
  `BayesianSolver` pickles its factorization as a dict of arrays and drops
  `S_hat`, `A`, `K` and `H S_0`, which the factors give again.
- **`BayesianSolver` never factors `H S_0 H^T + S_z`.** It whitens with the
  Cholesky factor of `S_z` alone (or the square root of its diagonal), so `S_z`
  must be positive-definite (it raises otherwise), and eigendecomposes the
  `m x m` matrix `L^-1 H S_0 H^T L^-T`, which is positive semidefinite: roundoff
  can only leave eigenvalues slightly below zero, which are clipped (a clip
  beyond roundoff is logged). A smooth temporal prior that makes `S_0` near-low-rank
  (what used to tip the sum off positive-definite and block sub-yearly
  inversions) therefore cannot stop the solve. Everything is derived from the
  factors (`fips/_whitened.py`'s docstring has the formulas); nothing `n x n`
  is formed unless `S_hat` or `A` is read, and `tests/test_whitened.py` fails if
  `fips.solve` with the variances, DOFS or chi-squared allocates one. With `m > n`
  the state-space form would be cheaper; it does not exist, and a large such
  solve logs a warning.
- Other covariance solves (`cost`, `KroneckerOperator.solve`, the base
  `Estimator`'s gain) go through `LinearOperator.solve`, whose dense form is
  `_solve_psd(A, B)` (in `linalg.py`), **not**
  `scipy.linalg.solve(..., assume_a="pos")` directly: it keeps the fast
  Cholesky path and falls back, with a warning, to a symmetric `LDL^T` solve when
  roundoff has tipped a matrix that is PD in exact arithmetic off-PD. Use it for
  any new covariance-like Cholesky solve here. A sparse operator solves with
  SuperLU (`splu`): SciPy has no sparse Cholesky, and an LU needs no
  positive-definiteness.
