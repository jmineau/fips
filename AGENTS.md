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
  __init__.py        public exports (re-exports Block, Vector, Matrix,
                     MatrixBlock, CovarianceMatrix, ForwardOperator,
                     Estimator, InverseProblem, InversionPipeline, convolve)
  base.py            Pickleable, MultiBlockMixin, SingleBlockMixin,
                     Structure1D/2D — foundational mixins all blocks build on
  vector.py          Block, Vector — 1D labeled containers (state, obs)
  matrix.py          MatrixBlock, Matrix — 2D labeled containers
  covariance.py      CovarianceMatrix — Matrix subclass; kernel-driven build,
                     sparse (csr_matrix) + joblib-parallel construction
  operators.py       ForwardOperator (Matrix subclass), convolve(state, H)
  estimators.py      Estimator ABC, ESTIMATOR_REGISTRY, available_estimators()
  problem.py         InverseProblem — top-level orchestrator that combines
                     obs, prior, H, S_0, S_z and dispatches to an Estimator
  pipeline.py        InversionPipeline ABC — workflow template for repeated
                     inversions (load data, build covariances, solve)
  aggregators.py     Time/space binning (e.g. integrate_over_time_bins)
  filters.py         Selection helpers (e.g. enough_obs_per_interval)
  _sparse.py         sparse fill_value normalization (pandas 3 workaround)
  indexes.py         Index sanity / promotion / overlap helpers
  kernels.py         Covariance kernels (exponential decay, time decay, ...)
  metrics.py         haversine_matrix, time_diff_matrix
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
- `JacobianBuilder` — optional, imported only if `pystilt` is available
  (lives in `problems/flux/transport/stilt/builder.py`)

The `flux` extra pulls in `pystilt`, `cartopy`, `h5py`, `matplotlib`.

## Public API summary

From `fips.__init__`:

| Symbol | What it is |
|---|---|
| `Block`, `Vector` | 1D labeled data (state vector, observations) |
| `MatrixBlock`, `Matrix` | 2D labeled data (operators, covariances) |
| `CovarianceMatrix` | Covariance with kernel-driven construction |
| `ForwardOperator` | Matrix specialization for the Jacobian H |
| `convolve` | apply H to a state vector |
| `Estimator`, `available_estimators` | swappable solvers via registry |
| `InverseProblem` | orchestrator (`solve()` returns posterior + diagnostics) |
| `InversionPipeline` | repeatable workflow template |

## Mental model

1. **Wrap your inputs in Blocks.** Observations and state become `Block`s
   (or lists of `Block`s for multi-source obs / multi-region state). Each
   Block carries a `pandas.MultiIndex` describing dims (time, lat, lon,
   site, etc.).
2. **Express H as `MatrixBlock`s.** Each `MatrixBlock` declares its
   `row_block` and `col_block`, so the framework knows how to stitch a
   block-structured Jacobian together.
3. **Build covariances.** `CovarianceMatrix` can be hand-built from arrays
   or composed from `kernels.py` (exponential decay, time decay, etc.).
   Use sparse (`csr_matrix`) when blocks are large; the package supports it
   throughout.
4. **Hand everything to `InverseProblem`.** It validates index alignment,
   picks an estimator from `ESTIMATOR_REGISTRY`, and `.solve()` returns
   posterior plus diagnostics (e.g. `reduced_chi2`).
5. **For repeated inversions, subclass `InversionPipeline`** so loading,
   covariance build, and solve are templated.

## Invariants to respect

- **MultiIndex preservation.** The whole design hinges on labels surviving.
  Do not silently `.values` or `.to_numpy()` away from Series/DataFrame.
- **Block alignment.** `MatrixBlock`s must declare correct `row_block`/
  `col_block` names; `InverseProblem` cross-checks against the obs/state
  block names. Stale names → silent misalignment.
- **Sparse fill_value is always 0.0.** The implicit entries of a covariance or
  a Jacobian are zeros, not missing data: `Structure2D.values` calls
  `.sparse.to_coo()` and `_validate` rejects NaN. pandas 3's
  `DataFrame.sparse.from_spmatrix` builds float frames with `fill_value=NaN`
  (pandas-dev/pandas#59212), so **every** sparse frame entering fips goes
  through `fips._sparse.normalize_fill_value` first — the two `from_spmatrix`
  call sites (`CovarianceBuilder.build(sparse=True)`,
  `ObsAggregator._build_operator`) and `Structure2D.__init__`, which is what
  lets a user hand in a pandas-native sparse frame instead of being told it
  "contains NaN". It fills rather than re-typing: assigning a new SparseDtype
  would keep the NaNs as stored entries. Pinned by `TestSparseFillValue` in
  `tests/test_sparse.py` — do not drop it as redundant when pandas fixes the
  regression; drop it when the minimum pandas has the fix.
- **Sparse paths.** `CovarianceMatrix` and large operators may be sparse.
  Don't densify in helpers without a sparse fallback — flux-scale problems
  are GB-class.
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
| `just build-docs` | Sphinx HTML into `docs/_build`; warnings are errors (needs pandoc) |
| `just docs-serve` | live docs preview on port 8000 |
| `just changelog` | draft CHANGELOG entries from the commits since the last tag |
| `just pre-commit` | every hook on every file |
| `just dist` | build and check the sdist and wheel |
| `just release X.Y.Z` | tag and push a release (the maintainer runs it, never an agent) |
| `just clean` | wipe build, dist, coverage, caches, `__pycache__`, docs build |

Workflows in `.github/workflows/`: `tests.yml` (Linux/macOS/Windows x Python
3.11-3.14), `quality.yml`, `docs.yml` (versioned docs on GitHub Pages: `dev/`
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
  shipped — keep new public API fully typed.
- **Docstrings**: numpy convention, format docstring code blocks
  (`docstring-code-format = true`).
- **Docs**: `docs/reference/` groups the API by topic, as pandas does. Each
  class also gets its own page, with tables of its attributes and methods, and
  each member a page of its own; a subclass lists what it defines and links to
  what it inherits (`docs/_templates/autosummary/` and `docs/_ext/api_pages.py`,
  from python-template). Document a new base class in `reference/base.rst`, or
  its inherited members show as plain names. Give each public property a
  docstring: an undocumented one shows an empty row.
- **Coverage**: configured in `pyproject.toml`; `tests/` is the only
  testpath. `coverage.exclude_also` skips `__repr__`, abstract methods,
  `if __name__ == "__main__":`, etc.
- **Dependency groups**: runtime deps in `[project]`; the `flux` extra adds
  cartopy/h5py/matplotlib/pystilt; the `dev` group is what `just sync`
  syncs.

## Dependency policy — what fips owns and what it delegates

Decided 2026-10-05 after surveying the inverse-problem libraries in other
fields (pyemu, pyOptimalEstimation, CoFI, CUQIpy, hippylib, pylops, MNE, CIF,
IMI, FLEXINVERT, LUMIA; see issues #21–#30). None of them provides a labeled
linear-Gaussian solve with posterior covariance at flux scale, and the ones
that come close use dense `inv`. So:

- **fips owns the core solve.** The analytic estimator is ~50 lines of
  `scipy.linalg` (Cholesky, triangular solves, `eigh`); the value of fips is
  the labeled/block layer around it, and routing arrays through an external
  solver is exactly where labels get lost. Do not add a dependency to replace
  `BayesianSolver`; improve it in place (#22).
- **Delegate factorizations and operator plumbing to scipy.** `scipy.linalg`
  for dense factorizations, `scipy.sparse` for storage,
  `scipy.sparse.linalg.LinearOperator` as the matvec abstraction when
  covariances or H become factored/matrix-free (#6), and
  `scipy.sparse.linalg.eigsh`/`lobpcg` for low-rank posteriors before porting
  any randomized eigensolver.
- **Optional, not required:** `pylops` (ready-made `Kronecker`/`BlockDiag`/
  `VStack` operators — decide at #6 whether to write the few we need instead),
  `GSTools` (covariance models, only if `kernels.py` outgrows exponential
  decay). If adopted, they go in an extra, never in `[project]` dependencies.
- **Never write a sampler.** If MCMC lands (#18), use PyMC or CUQIpy for
  sampling and ArviZ for diagnostics, as `openghg_inversions` does. Same for a
  bounded/variational optimiser (#15, #19): `scipy.optimize`, not custom code.
- **The reference libraries are test oracles, not dependencies.** Their
  closed-form identities (pyOE's χ² eigenvalue forms, hippylib's `d/(1+d)`
  posterior, pyemu's Schur posterior, the Yadav & Michalak supplement code) are
  for cross-checking new estimators on small random problems in `tests/`.

## Common workflows

### Single-block solve (one obs source, one state block)
See README "Single-block" example. Pass `obs`, `prior`, `forward_operator`,
`prior_error`, `modeldata_mismatch` directly to `FluxProblem(...).solve()`.

### Multi-block solve (heterogeneous obs)
Wrap each obs source in its own `Block`. Build one `MatrixBlock` per
(row_block, col_block) pair with explicit names. Block-diagonal
`modeldata_mismatch` is the common pattern (no inter-source error
covariance). See README "Multi-block" example.

### Adding a new estimator
1. Subclass `Estimator` in `estimators.py`.
2. Implement the abstract solve method.
3. Register via the `EstimatorRegistry` decorator (`@ESTIMATOR_REGISTRY.register("name")`).
4. Now available via `available_estimators()` and selectable on
   `InverseProblem`.

### Adding a new problem domain
1. Create `src/fips/problems/<domain>/` with `__init__.py` re-exporting the
   public surface.
2. Subclass `InverseProblem` for the domain-specific orchestration; subclass
   `InversionPipeline` for the workflow template.
3. Mirror the `flux` package's layout (`problem.py`, `pipeline.py`,
   `visualization.py`, optional `transport/`).

## Gotchas

- The `flux` extra hard-depends on `pystilt`; without it, `JacobianBuilder`
  is silently omitted from `fips.problems.flux.__all__` (try/except in
  `problems/flux/__init__.py`). Don't surprise users — keep the optional
  import contract.
- `convolve(state, H)` returns a `Vector`, not a numpy array. Don't expect
  raw arrays out of public APIs.
- `reduced_chi2` lives on the *estimator*, not the problem
  (`problem.estimator.reduced_chi2`).
- Covariance solves in `estimators.py` go through `_solve_psd(A, B)`, **not**
  `scipy.linalg.solve(..., assume_a="pos")` directly. `A = H S_0 H^T + S_z` is
  PD in exact arithmetic, but when it is ill-conditioned (wide eigenvalue
  spread, `H S_0 H^T` large relative to `S_z`) roundoff can tip it numerically
  off-PD, making bare Cholesky raise `LinAlgError: singular matrix`. The module
  is pure linear algebra and knows nothing about the problem domain — the
  conditioning comes entirely from the caller's `H`/`S_0`/`S_z` (downstream this
  surfaced as a near-low-rank `S_0` from a smooth temporal prior, which blocked
  sub-yearly flux inversions). `_solve_psd` keeps the fast Cholesky path and
  falls back to a symmetric `LDL^T` solve (`assume_a="sym"`) with a warning. Use
  it for any new covariance-like Cholesky solve here.
