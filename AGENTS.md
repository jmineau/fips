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

- Not a sampler (no MCMC). The solve methods are analytical / regularized.
- Not nonlinear-problem aware. The framework assumes linearity around the
  prior; nonlinear iterations are the caller's responsibility.
- Not opinionated about file formats. Inputs are pandas/xarray; serialization
  is plain `pickle` (a `Problem` and a `Posterior` pickle).

## Module layout

```
src/fips/
  __init__.py        public exports (Space, Vector, Operator, Covariance, the
                     covariance models, Problem, solve, available_methods,
                     Posterior, the aggregate module, offset_block)
  space.py           Space — ordered named blocks, the padded MultiIndex,
                     select, from_index, exact align, round_labels
  vector.py          Vector — values on a Space (read-only float array);
                     from_series, from_blocks, v[block], to_series, to_xarray,
                     combine_first
  operator.py        Operator — a matrix (a linalg form) with a Space on each
                     axis: from_frame, from_blocks, op[r, c], @ / + by exact
                     join, align, to_frame
  problem.py         Problem — prior, prior_cov, obs, obs_cov, forward,
                     offset, checked once by exact join; models built there;
                     predict(x) = H x + c; restrict (part of the state
                     held at the prior) and aggregate (the observations
                     through one W)
  posterior.py       Posterior — the problem, the options and the method's
                     solution; asks the solution for unlabeled arrays, labels
                     them, builds each once, and writes the rest (averaging
                     kernel, innovation_cov, desroziers, leverage, scalars)
                     from them; summary() returns the diagnostics as one row.
                     Solution — the Protocol a method's kept object answers
                     (estimate, variances, cov(W), gain(W), dofs, chi2;
                     optional with_gamma, with_data, estimate_is_mean);
                     Options — what it reads from the options (gamma)
  aggregate.py       sum, mean: an aggregation W (sparse Operator) from a
                     Space to groups of its labels (levels, a pandas Grouper,
                     an IntervalIndex, a Series mapping); offset_block, the
                     same groups as a block of offsets in the state
  visualization.py   plot_comparison (the `viz` extra: matplotlib): several
                     series on one axes, with error bands or bars
  covariance/        covariances, and the models that write them down
                     (public as `fips.covariance`)
    __init__.py      re-exports Covariance and the models
    covariance.py    Covariance — Operator on one space: diagonal, block_diag
                     (in order, or by block name; a part that is a model goes
                     to models.py through a hook), variances, solve, logdet,
                     quadratic_form, sqrt
    models.py        CorrelationModel (base; with_std), Independent, Kronecker;
                     CovarianceModel — a covariance waiting for its space
                     (with_std of a scalar or array, sums, block_diag with a
                     model); build(space) gives the Covariance
    kernels.py       the kernels: Exponential, Gaussian, Matern, Constant, and
                     within= (zero between groups of labels, sparse)
    distances.py     haversine_matrix, chordal_matrix, time_diff_matrix
  methods/           how a Problem is solved (`fips.solve` is the function)
    __init__.py      solve(problem, method, **options), the method registry
                     (_METHODS, available_methods)
    analytic.py      AnalyticOptions, and the analytic method: Whitened —
                     its Solution, the factorization of the whitened system
                     read at one gamma (estimate, variances, gain, DOFS,
                     chi-squared; with_gamma and with_data share the
                     factors) — and _analytic, which returns
                     Posterior(problem, Whitened.factor(...), options)
  linalg/            the unlabeled forms the solvers compute with (public as
                     `fips.linalg`)
    __init__.py      re-exports the forms; its docstring says what a form does
    forms.py         LinearOperator (scipy LinearOperator) and the Dense-,
                     Sparse- and DiagonalOperator it builds, products and sums
                     (_product, _sum), as_linear_operator, dottest, _adopt,
                     _solve_psd
    block.py         BlockDiagonalOperator
    kronecker.py     KroneckerOperator — a covariance kept as its factors
  py.typed           ships type hints
tests/               pytest suite (no integration markers in use)
docs/                Sphinx (pydata-sphinx-theme + numpy-style docstrings)
archive/             local-only, not tracked
```

Imports run one way: `space` < `vector` and `linalg` < `operator` <
`covariance` < `problem` < `posterior` < `methods`, with `aggregate` and
`visualization` on top of `operator` and `covariance`, and no module imports
one to its right (no import inside a function to avoid a cycle, and none under
`TYPE_CHECKING`). `Covariance`
does not import the models: `models.py` hands `Covariance.block_diag` the
function that makes the block diagonal of parts some of which are models
(`Covariance._block_diag_models`), so `import fips.covariance` always sets it.
`posterior.py` imports nothing from `fips.methods`: a method's solution and
options are typed by the `Solution` and `Options` protocols defined beside
`Posterior`, and a method imports `posterior` to build one. Whatever only one
method can do (the analytic factorization, `Whitened`) lives in that method's
module; `Posterior` reaches it only through the protocol, and through
`with_gamma`, `with_data` and `mean`, which raise for a solution without them.
`linalg/forms.py` holds `LinearOperator` and the
three forms it builds in one module, since the base class names its
subclasses.

### No domain subpackages

fips knows no science domain: a flux inversion is a `Problem` whose blocks
are named `flux` and `concentration`, read by name (`post.estimate["flux"]`),
and its workflow and its plots live in its user's package as functions.
fips is grid-agnostic: nothing in its public names, defaults or docstrings
(beyond an example) is flux science or geography (no maps, no lat/lon
defaults, no cartopy). Its one plot is `plot_comparison`; anything else is
drawn from `Vector.to_series()` or `to_xarray()` with pandas or xarray. The
`viz` extra pulls in `matplotlib`. fips does not build transport Jacobians: PYSTILT's
`project.jacobian()` does, and the caller places its sparse matrix with its
labels (`Operator.from_blocks({("concentration", "flux"): (J.data, rows,
cols)}, rows=obs.space, cols=prior.space)`).

## Public API summary

From `fips.__init__`:

| Symbol | What it is |
|---|---|
| `Space`, `Vector` | the labels of a vector (named blocks), and values on them |
| `Operator` | a matrix with a `Space` on each axis, in its cheapest form (H, an aggregation) |
| `Covariance` | an `Operator` from a space to itself (`S_0`, `S_z`), with solve/logdet/sqrt |
| `Exponential`, `Gaussian`, `Matern`, `Constant`, `Independent`, `Kronecker` | correlation models; `model.with_std(std)` is a covariance |
| `Problem` | the inputs of a solve, their labels checked once by exact join |
| `solve`, `available_methods` | `fips.solve(problem, method="analytic", **options)` returns a `Posterior` |
| `Posterior` | the estimate and every diagnostic, labeled, from what the method kept (the analytic method: its factorization) |
| `aggregate.sum`, `aggregate.mean` | an aggregation `W` of a space by group: `problem.aggregate(W)` on the observations, `post.cov(W)` on the state |
| `offset_block` | a block of offsets in the state, one per group of observations, and its forward block |

The one plot is `fips.visualization.plot_comparison` (not exported from
`fips`).

## Mental model

1. **Wrap your inputs in Vectors.** Observations and state become `Vector`s
   (`Vector.from_series(series, block="flux")` for one block,
   `Vector.from_blocks({...})` for several). Each block carries its own
   pandas index describing dims (time, lat, lon, site, etc.).
2. **Express H as an `Operator`.** `Operator.from_blocks({(obs_block,
   state_block): frame})` stitches a block-structured Jacobian together and
   places each labeled block (a DataFrame, or `(matrix, row_labels,
   col_labels)`) by its labels: a label it lacks is zero, one its block lacks
   raises;
   `Operator(matrix, rows, cols)` labels a matrix you have (a SciPy sparse one
   stays sparse); `Operator.from_frame` reads a labeled DataFrame.
3. **Build covariances from models.** A correlation model and standard
   deviations: `Kronecker(Exponential("time", "30D"), Exponential(["lat",
   "lon"], 20.0, metric="haversine")).with_std(std)` (a Kronecker prior error
   stays as its factors), `Independent().with_std(s) + Exponential("obs_time",
   "2.8h", within=["site", "obs_time.date"]).with_std(t)`, parts by block in
   `Covariance.block_diag(flux=..., bias=...)`; `.build(space)` puts a model on
   the labels (a `Vector` std gives the `Covariance` at once). Or
   `Covariance.diagonal(variances, space)` and `Covariance.from_frame`. `H`,
   `S_0` and `S_z` may each be dense or sparse, and the solve keeps each in its
   form (see "Sparse paths").
4. **Hand everything to `Problem`.** `Problem(prior, prior_cov, obs,
   obs_cov, forward, offset=None)` checks the labels once (exact join) and
   builds a covariance model on its space. `fips.solve(problem)` returns a
   `Posterior` (estimate, `cov(W)`, `variances`, diagnostics such as `dofs`
   and `reduced_chi2`; `post.summary()` gives them as one row); the problem
   has no `solve` method and is not changed.
5. **A repeated inversion is a function** that builds the `Problem` from its
   settings, then `fips.solve`; the rows of `post.summary()` stack over a
   sweep.

## Invariants to respect

- **Names.** `Vector` and `Operator` convert a name that is not
  `None` to a string, so a name is one spelling everywhere; an unnamed object
  keeps `None`. A sum of two operators keeps the name when both have the same
  one and is `None` otherwise (pandas' rule for a sum of two Series). A
  `Problem` names its covariances `"prior_cov"` and `"obs_cov"` whether it was
  given a model or a `Covariance` (a renamed copy sharing the form: the
  caller's object keeps its name).
- **A structure owns its data.** `Vector` and `Operator` (every `fips.linalg`
  form) copy their input and make the copy read-only. Code that builds a form
  from an array fips just made, or from another form's read-only arrays, uses
  `_adopt` (no second copy); the public constructors are for what a caller
  hands over.
- **A covariance is symmetric.** `Covariance(...)` (so `from_frame` and
  `from_blocks`) checks a dense or sparse matrix to 1e-10 of its largest
  entry, without densifying, and `KroneckerOperator` its factors. A covariance
  fips computes from covariances or models (a sum, a reordering, a block
  diagonal, `W S W^T`, a posterior) is built with `Covariance._computed`,
  which skips the check: its roundoff can come near 1e-10.
- **MultiIndex preservation.** The whole design hinges on labels surviving.
  Do not silently `.values` or `.to_numpy()` away from Series/DataFrame.
- **Labels are kept as given.** Nothing converts labels to numbers or rounds
  them unless the caller asks (`Space.round_labels`); duplicate,
  missing (NaN) and unnamed labels of a block raise.
- **NaN-only levels.** Assembling blocks whose indexes have different levels
  fills the missing levels with NaN, and `Space` is the one place that does it
  in both directions: `Space.index` pads (keeping each level's dtype, through
  the MultiIndex codes) and `Space.from_index` / `Space.select` strip. Vectors
  and both axes of every operator go through it; `v[block]` and `op[r, c]`
  give a block on single-block spaces, labeled as the block was given. A block
  labeled by a MultiIndex of one level is held as that level's plain Index.
- **An operator's labels are beside its matrix.** `Operator.data` is a form of
  `fips.linalg` with rows and columns in the order of `rows` and `cols`. Every
  product, sum and placement puts the other side in this side's order with
  `Space.align` (an exact join) before computing, and a method that changes
  labels or data builds its result with `_like` (or `_build` in a
  constructor), which keeps a `Covariance` a `Covariance` when both axes are
  one space. Reorder through `form._select(rows, cols)`, which keeps a
  Kronecker or block-diagonal form when the rows and columns are the same
  positions.
- **The exact join of a problem.** `Problem.__init__` is the one place a
  solve's inputs are checked: `forward`'s rows and columns, each covariance and
  the offset must have exactly the labels of the observations or the prior (in
  any order), or it raises naming the blocks and labels on one side only
  (`Space._align`, which `Space.align` calls with its own wording). Nothing is
  zero-filled here and nothing is rounded: a Jacobian that covers only the
  cells a footprint touched is put on the whole state before, by
  `Operator.from_blocks`, which places a labeled block by its labels (a label
  the block lacks is zero, a label the space lacks raises; `operator._placed`
  keeps a sparse block sparse and a block that covers its labels in its form).
  Or the state is restricted to what the Jacobian covers (`Problem.restrict`,
  which holds the rest at the prior through the offset).
  Each input is then held in the order of `prior.space` and
  `obs.space`, through `form._select` (which keeps a Kronecker form), and a
  covariance model is built there (`model.build(space, name=...)`). The solve
  and the `Posterior` compute by position on those orders and label every
  output with those two spaces.
- **The problem is the question, the posterior the answer.** `Problem` holds
  what needs no solve and no `m x m` matrix (`predict`), and makes other
  problems from itself (`aggregate`, `restrict`); `Posterior` holds what
  depends on the solve or on the `m x m` matrix it factors (`innovation_cov`,
  the estimate and every diagnostic). `Posterior.modeled_obs` is the one
  shortcut to a problem method: `problem.predict(post.estimate)`, kept.
- **A sparse matrix is a SciPy matrix, never a pandas-sparse frame.** The
  one place a pandas-sparse frame enters is `Operator.from_frame`, through
  `operator._frame_matrix`: the
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
  (the rule is in the docstring of `linalg/__init__.py`). The tests are the dense/sparse
  combinations in `tests/test_linalg.py` and the oracle's sparse closed form.
  The analytic method and the `Posterior` compute with `problem.forward.data`,
  `prior_cov.data` and `obs_cov.data`, never with their dense matrices.
- **Covariance models.** A model is a frozen dataclass whose parameters are
  checked in `__init__`; it reads its levels from the labels of the space it
  is built on (`covariance/models.py`'s `_coords`, which raises when a block lacks one) and gives
  its correlation in its cheapest form (`_form`: `Independent` a diagonal, a
  kernel `within` groups sparse, `Kronecker` a `KroneckerOperator` with each
  factor evaluated once on the unique coordinates of its levels, in first
  appearance order). `with_std` scales that form without forming the matrix
  (`models._scaled`). Each covariance the removed builders made is pinned in
  `tests/test_covariance.py` (`tests/data/old_builders.npz`, written on pandas
  3; bit for bit there, and to the last bit on pandas 2, whose time
  differences are in nanoseconds): keep
  the order of the products (`(std_i * C_ij) * std_j`) and the distance
  functions of `covariance/distances.py` when changing a kernel.
- **Aggregations.** `fips.aggregate.sum` and `mean` build `W` block by
  block: each aggregated block's groups are its rows, sorted, and a block not
  named in `blocks=` passes through (one row per label); `offset_block` forms
  its groups over the named blocks together and gives other blocks zero rows.
  A group exists only if a label falls in it, so `W` has no empty row (`mean`
  divides by the count, and `averaging_kernel(W)` needs `W W^T` invertible). A
  label a key cannot place raises. `Problem.aggregate` and `restrict` return a
  new problem of the same class, built through `__init__` (the exact join
  runs again); `restrict` keeps each covariance's form (`form._select`).
- **Pickling.** Anything user-facing should round-trip through plain
  `pickle`: `Problem` and `Posterior` define `__getstate__` and `__setstate__`
  for it. A `Posterior` pickles the problem, the options and the method's
  solution (the analytic method's `Whitened` refers to the same problem, which
  pickle stores once), never an output, which is rebuilt when read.
- **Logging, not printing.** Every module uses
  `logger = logging.getLogger(__name__)`; `__init__.py` attaches a
  `NullHandler`. Don't `print()` in library code: a report is returned
  (`Posterior.summary()` is a DataFrame), and a plot returns its figure or
  axes without `plt.show()`.
- **Docstring convention.** `pydocstyle` (numpy convention) is enforced via
  ruff (`D` rules). Public functions need at least a one-line docstring.

## Dev commands

Driven by `just` + `uv`; CI runs the same recipes.

| Command | What it does |
|---|---|
| `just sync` | `uv sync`: fips, its `viz` extra, and the dev tools |
| `just quality-check` | `lint` + `type-check` + `docstr` + `test`: what CI checks |
| `just test` | pytest in parallel (up to 8 workers), skipping `network`/`slow`; extra args go to pytest (`-n 0` for serial) |
| `just lint` / `just format` | ruff check and format check / fix and format |
| `just type-check` | pyrefly on `src/` |
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
- **Types**: pyrefly on `src/`, with no baseline: every error fails the check.
  `py.typed` is shipped — keep new public API fully typed. Fix an error where
  the type comes from (a correct annotation, a narrower return type), not with
  `typing.cast` or `Any`. Where pandas-stubs are wrong or too narrow (pandas
  takes the call, as with `DataFrame.sparse`, which they leave untyped), put
  `# pyrefly: ignore[<code>]` on the line above the error, after a comment
  saying why.
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
- **Dependency groups**: runtime deps in `[project]`; the `viz` extra adds
  matplotlib; the `dev` group is what `just sync`
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
- **Optional dependencies:** the `dev` group already lists `fips[viz]`
  (matplotlib), so `uv sync` installs them here and in CI. If an example
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

- **fips owns the core solve.** The analytic method is ~50 lines of
  `scipy.linalg` (Cholesky, triangular solves, `eigh`); the value of fips is
  the labeled/block layer around it, and routing arrays through an external
  solver is exactly where labels get lost. Do not add a dependency to replace
  the analytic method (`methods/analytic.py`, with `Whitened`); improve it in place.
- **Delegate factorizations and operator plumbing to scipy.** `scipy.linalg`
  for dense factorizations, `scipy.sparse` for storage,
  `scipy.sparse.linalg.LinearOperator` as the matvec abstraction when
  covariances or H become factored/matrix-free, and
  `scipy.sparse.linalg.eigsh`/`lobpcg` for low-rank posteriors before porting
  any randomized eigensolver.
- **pylops is read, not installed or copied** (LGPL; decided 2026-10-14): fips
  writes the few operators it needs on `LinearOperator`, as `KroneckerOperator`.
- **Optional, not required:** `GSTools` (covariance models, only if
  `covariance/kernels.py` outgrows its exponential, Gaussian and Matérn kernels). If adopted, it goes in an extra,
  never in `[project]` dependencies.
- **Never write a sampler.** If MCMC lands, use PyMC or CUQIpy for
  sampling and ArviZ for diagnostics, as `openghg_inversions` does. Same for a
  bounded/variational optimizer: `scipy.optimize`, not custom code.
- **The reference libraries are test oracles, not dependencies.** Their
  closed-form identities (pyOE's χ² eigenvalue forms, hippylib's `d/(1+d)`
  posterior, pyemu's Schur posterior, the Yadav & Michalak supplement code) are
  for cross-checking new methods on small random problems in `tests/`.

## Common workflows

### Single-block solve (one obs source, one state block)
See README "Single-block" example: `Problem(prior=..., prior_cov=...,
obs=..., obs_cov=..., forward=...)`, then `fips.solve(problem)`.

### Multi-block solve (heterogeneous obs)
Put each obs source in its own block (`Vector.from_blocks`). Build `H` with
`Operator.from_blocks`, one entry per (row_block, col_block) pair. A
block-diagonal `obs_cov` (`Covariance.block_diag`) is the common pattern (no
inter-source error covariance). See README "Multi-block" example.

### Adding a new method
1. Write its options as a frozen dataclass in a new `methods/<name>.py` (the
   only place they are declared: `solve` checks the caller's keywords against
   its fields and lists them for an unknown one; check values in
   `__post_init__`). It has a `gamma` field (`posterior.Options`).
2. Write its solution in the same file: the object it keeps from a solve,
   answering `posterior.Solution` (`estimate()`, `variances()`, `cov(W)`,
   `gain(W)`, `dofs()`, `chi2()`, unlabeled arrays in the problem's order).
   Give it `with_gamma` or `with_data` only if it can answer without solving
   again, and `estimate_is_mean = True` only if its estimate is the mean;
   without them `Posterior` raises for those features. The docstring of
   `methods/__init__.py` says how a lognormal or an ensemble method would fit.
3. Write the method as a function `(problem, options) -> Posterior(problem,
   solution, options)` in the same file and list it in `_METHODS` in
   `methods/__init__.py`; `available_methods()` and `fips.solve(problem,
   "name")` then find it. The one `Posterior` labels every method's answer.
5. `tests/test_oracle.py` is parametrized over `available_methods()`, so the
   new method is held to the closed-form Gaussian posterior and the
   permutation test at once. A method whose answer is not that posterior says
   so there, with its own reference.

### A new problem domain
fips gets no subpackage, `Problem` subclass or pipeline for it (see "No
domain subpackages"): the domain names its blocks, writes its workflow as
functions in its own package, and reads the outputs by block name; its plots
stay there too. A statistic that any problem could use goes on `Posterior`.

## Gotchas

- **An `Operator` takes its matrix by position, a DataFrame by
  `from_frame`.** The constructor refuses a pandas object or an `Operator`
  (and `Vector` a `Vector`), since its labels would be ignored; a diagonal is `Covariance.diagonal(values, space)` (kept
  as a `DiagonalOperator`). An `Operator` multiplies an `Operator` or a
  `Vector`, never an array (no labels to check): compute by position with
  `op.data`.
- Check an optional pandas index with `is None`, since evaluating its truth
  value raises an error.
- **pyrefly's warnings** are not shown by `pyrefly check`: run
  `uv run pyrefly check --min-severity warn --output-format min-text` to see
  them with their lines. pyrefly types `DataFrame.columns` as `Index[str]`, so it calls the
  `str(col)` in `plot_comparison` an unnecessary conversion, but a column label
  is an integer when a Series has no name: keep the `str()`.
- **SciPy is not annotated**: pyrefly takes the private class a `LinearOperator`
  default returns as the type an override must return, and a sparse `shape` as
  possibly `None`, and finds no `__getitem__` on a sparse array. `linalg/forms.py`
  marks those lines with an ignore and the reason; a new form that
  overrides only `_transpose` needs none. `form.T` is typed as SciPy's private
  transposed class: call `form._transpose()` to keep a fips form.
- **pyrefly (1.3.2) reads `type[X]` written inside a base class's arguments as
  `type[Any]`**: after `class R(dict[str, type[E]])` a lookup in `R()` is
  `type[Any]`, though `typing.Type[E]` or an alias of `type[E]` gives
  `type[E]`. Use `typing.Type[E]` or an alias where a subclass of a container
  holds classes.
- **A `groupby` on one `pd.Grouper(freq=...)` numbers its bins in time order,
  empty bins included**: `ngroup()` skips the number of an empty bin, and on
  labels that are not in time order it does not follow the labels' order (seen
  with pandas 2.0.3 and 3.0.6). `aggregate._key_labels` bins the labels sorted
  and reads each label's bin from `size().index`; do the same anywhere a single
  frequency key is grouped.
- `H @ vector` is a `Vector`, not a numpy array. Don't expect raw arrays out
  of public APIs.
- **The analytic method never factors `H S_0 H^T + S_z`.** It whitens with the
  Cholesky factor of `S_z` alone (or the square root of its diagonal), so `S_z`
  must be positive-definite (it raises otherwise), and eigendecomposes the
  `m x m` matrix `L^-1 H S_0 H^T L^-T`, which is positive semidefinite: roundoff
  can only leave eigenvalues slightly below zero, which are clipped (a clip
  beyond roundoff is logged). A smooth temporal prior that makes `S_0` near-low-rank
  (what used to tip the sum off positive-definite and block sub-yearly
  inversions) therefore cannot stop the solve. Everything is derived from the
  factors (`Whitened`'s docstring in `fips/methods/analytic.py` has the formulas); nothing `n x n`
  is formed unless `post.cov()` or `post.averaging_kernel()` is read, and
  `tests/test_whitened.py` fails if `fips.solve` with the variances, DOFS or
  chi-squared allocates one. With `m > n`
  the state-space form would be cheaper; it does not exist, and a large such
  solve logs a warning.
- Other covariance solves (`Covariance.solve`, `KroneckerOperator.solve`,
  `Posterior.leverage`) go through `LinearOperator.solve` or `_solve_psd`, whose dense form is
  `_solve_psd(A, B)` (in `linalg/forms.py`), **not**
  `scipy.linalg.solve(..., assume_a="pos")` directly: it keeps the fast
  Cholesky path and falls back, with a warning, to a symmetric `LDL^T` solve when
  roundoff has tipped a matrix that is PD in exact arithmetic off-PD. Use it for
  any new covariance-like Cholesky solve here. A sparse operator solves with
  SuperLU (`splu`): SciPy has no sparse Cholesky, and an LU needs no
  positive-definiteness.
