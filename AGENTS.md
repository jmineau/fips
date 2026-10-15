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

# AGENTS.md: fips

Orientation for contributors and coding agents. Contribution mechanics are in
[CONTRIBUTING.md](CONTRIBUTING.md); the words (state, observation, forward
operator, prior covariance, ...) are defined in `docs/terminology.rst`.

## What fips is

The **Flexible Inverse Problem Solver**: the exact Gaussian posterior of a
*linear* inverse problem, `z = H x + c + e`, whose state and observations are
labeled by pandas indexes in named blocks. Labels are matched, never positions;
each matrix is kept in its cheapest form (dense, sparse, diagonal, Kronecker
factors). Beta: the API may change between releases. PyPI and import name `fips`.

It is not a sampler, not a nonlinear solver (the caller linearizes), not a file
format, and not tied to a science domain: a flux inversion is a `Problem` whose
blocks are named `flux` and `concentration`, and its workflow, maps and STILT
code live in its user's package (no domain subpackages, no lat/lon defaults, no
cartopy). fips does not build transport Jacobians; the caller places one with
`Operator.from_blocks`.

## Layout

```
src/fips/
  __init__.py       the public names: Space, Vector, Operator, Covariance, the models,
                    Problem, solve, available_methods, Posterior, aggregate, offset_block
  space.py          Space: ordered named blocks, the padded MultiIndex, select, align
  vector.py         Vector: values on a Space (read-only), from_series/from_blocks
  operator.py       Operator: a fips.linalg form with a Space on each axis
  problem.py        Problem: the inputs, checked once; predict, aggregate, restrict
  posterior.py      Posterior (labels a method's answer); the Solution and Options protocols
  aggregate.py      sum, mean (an aggregation W), offset_block
  visualization.py  plot_comparison, the one plot (the `viz` extra: matplotlib)
  covariance/       Covariance and the models that write one down (public)
    covariance.py   Covariance: diagonal, block_diag, variances, solve, logdet, sqrt
    models.py       CorrelationModel, Independent, Kronecker, CovarianceModel
    kernels.py      Exponential, Gaussian, Matern, Constant; within= groups
    distances.py    haversine_matrix, chordal_matrix, time_diff_matrix
  methods/          solve, the method table (_METHODS), one module per method
    analytic.py     AnalyticOptions; Whitened, the whitened factorization (its Solution)
  linalg/           the unlabeled forms (public as fips.linalg)
    forms.py        LinearOperator and the Dense-, Sparse- and DiagonalOperator it builds
    block.py        BlockDiagonalOperator
    kronecker.py    KroneckerOperator: a covariance kept as its factors
tests/              pytest; test_oracle.py holds every method to the closed form
docs/               Sphinx: guides, reference/ (by topic), examples/ (notebooks)
```

Imports run one way: `space` < `vector`, `linalg` < `operator` < `covariance` <
`problem` < `posterior` < `methods`, with `aggregate` and `visualization` on
top. No import inside a function to dodge a cycle, none under `TYPE_CHECKING`.
`Covariance` does not import the models: `models.py` sets
`Covariance._block_diag_models`, so `Covariance.block_diag` takes model parts.
`posterior.py` imports no method: a method imports `posterior` and returns
`Posterior(problem, solution, options)`.

## Mental model

Wrap the prior and the observations in `Vector`s (`from_series`, `from_blocks`);
express `H` as an `Operator` (by position, `from_frame`, or `from_blocks`,
which places labeled blocks by label); write each covariance as a correlation
model times standard deviations (`Kronecker(...).with_std(std)`, sums,
`Covariance.block_diag(flux=..., bias=...)`); hand them to `Problem`, which
checks the labels and builds the models; `fips.solve(problem)` returns the
`Posterior`. A repeated inversion is a function that builds the `Problem`.

## Invariants

- **Labels are matched exactly, and kept as given.** `Problem.__init__` is the
  one place a solve's inputs are checked: `forward`, each covariance and the
  offset must have exactly the labels of the observations or the prior, in any
  order, or it raises naming the labels on one side only (`Space._align`).
  Nothing is zero-filled, converted to numbers or rounded unless the caller
  asks (`round_labels`). Duplicate, missing and unnamed labels raise.
- **`block` is the first level, and `Space` owns the padding.** A level one
  block lacks is NaN on its rows; `Space.index` pads (keeping each level's
  dtype) and `Space.from_index`/`select` strip, so `v[block]` and `op[r, c]`
  come back labeled as given. A one-level MultiIndex is held as its plain Index.
- **An operator's labels are beside its matrix.** `Operator.data` is a
  `fips.linalg` form in the order of `rows` and `cols`; every product, sum and
  placement aligns the other side first, and a result is built with `_like`
  (which keeps a `Covariance` a `Covariance`). Reorder with `form._select`,
  which keeps a Kronecker or block-diagonal form.
- **A structure owns its data.** `Vector`, `Operator` and every form copy their
  input and make the copy read-only. A form built from arrays fips just made,
  or from a parent's read-only arrays, uses `_adopt` (no second copy).
- **Sparse stays sparse.** A sparse matrix is a SciPy matrix, never a
  pandas-sparse frame (one enters only through `Operator.from_frame`, with a
  fill of 0 or NaN). Products and sums keep the cheapest exact form; a product
  past a quarter full is dense (the rule is in `linalg/__init__.py`). Compute
  with `.data`, not `dense()` or `to_frame()`, except where a dense matrix is
  the point: the analytic method densifies `H S_0` (`m x n`), `H S_0 H^T` and a
  non-diagonal `S_z` (`m x m`), and `post.cov()` forms `n x n` only when read.
- **The problem is the question, the posterior the answer.** `Problem` holds
  what needs no solve and no `m x m` matrix (`predict`, `aggregate`,
  `restrict`, each returning a new problem through `__init__`); `Posterior`
  holds everything that depends on the solve, built once and kept.
- **The posterior labels; the method computes.** A method's solution answers
  `posterior.Solution` with unlabeled arrays in the problem's order;
  `Posterior` labels them and writes the rest (averaging kernel, innovation
  covariance, Desroziers, leverage, scalars) from them.
- **Covariance models** are frozen dataclasses checked in `__init__`. The 0.1
  builders' covariances are pinned bit for bit in `tests/test_covariance.py`:
  keep the order of the products (`(std_i * C_ij) * std_j`).
- An aggregation has one row per group that holds a label; `Problem` and
  `Posterior` pickle with plain `pickle` (a posterior never stores an output).
- **Logging, not printing.** A module that logs uses
  `logger = logging.getLogger(__name__)`; `__init__.py` adds a `NullHandler`.
  A report is returned (`summary()` is a DataFrame), and a plot returns its
  figure and axes without `plt.show()`.

## Adding a method

A module `methods/<name>.py` with three things: its options (a frozen dataclass
with a `gamma` field, values checked in `__post_init__`), its solution (answers
`posterior.Solution`; `with_gamma`, `with_data` and `estimate_is_mean` only if
it can), and a function `(problem, options) -> Posterior`. Add a line to
`_METHODS` in `methods/__init__.py`; its docstring says how a lognormal or an
ensemble method fits. `tests/test_oracle.py` runs every method against the
closed-form posterior and the permutation test.

## Dev commands

Driven by `just` and `uv`; CI runs the same recipes.

| Command | What it does |
|---|---|
| `just sync` | `uv sync`: fips, its `viz` extra, and the dev tools |
| `just quality-check` | `lint` + `type-check` + `docstr` + `test`: what CI checks |
| `just test` | pytest in parallel, skipping `network`/`slow` (`-n 0` for serial) |
| `just lint` / `just format` | ruff check and format check / fix and format |
| `just type-check` | pyrefly (no baseline: every error fails) |
| `just docstr` | docstr-coverage of the public API |
| `just build-docs` | Sphinx, running every example; warnings are errors |
| `just pre-commit`, `just dist` | every hook on every file; build and check the dists |

`just test` does not run docstring examples: `uv run pytest --doctest-modules
src/fips` does. Workflows: `tests.yml`, `quality.yml`, `docs.yml` (versioned
docs on gh-pages), `publish.yml` (a `vX.Y.Z` tag publishes). The version comes
from git tags; the maintainer cuts releases. Tooling comes from
[jmineau/python-template](https://github.com/jmineau/python-template) (copier).

## Conventions

- Python 3.11+; ruff (`E, F, UP, B, SIM, I, D, D213, NPY, RUF100`), the whole
  repo including tests and notebooks; NumPy docstrings, code in them formatted.
- pyrefly with no baseline. Fix a type where it comes from, not with `cast` or
  `Any`; where pandas-stubs or SciPy are wrong, `# pyrefly: ignore[<code>]` on
  the line above, after a comment saying why.
- `docs/reference/` groups the API by topic; each class gets a page, and each
  documented member a page of its own (`docs/_templates/autosummary/`,
  `docs/_ext/api_pages.py`). A new public attribute needs a docstring.
- Examples run when the docs build (python-template's rules): `.. plot::` and
  `.. ipython:: python` (one session for all pages), and notebooks in
  `docs/examples/` committed without outputs; synthetic, seeded data only.

## Dependencies

fips owns the core solve (the labeled layer around a few `scipy.linalg` calls;
an outside solver loses the labels) and delegates factorizations and storage to
SciPy (`scipy.linalg`, `scipy.sparse`, `LinearOperator`, `eigsh`).
pylops is read, not installed or copied (LGPL). GSTools only in an extra,
if `covariance/kernels.py` outgrows its kernels. Never write a sampler: PyMC or CUQIpy and ArviZ if
MCMC comes. The other inverse-problem libraries (pyOE, hippylib, pyemu, the
Yadav and Michalak code) are test oracles, not dependencies. A new runtime
dependency needs a reason, proposed first.

## Gotchas

- **Never run `python` from inside `src/fips`**: `operator.py` there shadows
  the standard library's `operator`.
- **Docs:** an `.. ipython::` block can drop a long statement near its tenth
  (split long blocks); napoleon reads a colon in an attribute docstring's first
  line as a type; docstr-coverage counts each `@overload` (give it a one-liner
  with `# noqa: D418`); a `.. currentmodule::` goes after an `.. automodule::`.
- **SciPy is not annotated**: pyrefly takes a `LinearOperator` default's private
  return class as the type an override must return. `linalg/forms.py` marks
  those lines; call `form._transpose()` rather than `form.T` to keep a fips form.
- **One `pd.Grouper(freq=...)` key numbers its bins in time order, empty bins
  included**: `aggregate._key_labels` reads each label's bin from
  `size().index`; do the same wherever one frequency key is grouped.
- **The analytic method whitens with the Cholesky factor of `S_z`**, so `S_z`
  must be positive-definite; it never factors `H S_0 H^T + S_z`. Other
  covariance solves go through `LinearOperator.solve` or `_solve_psd`
  (Cholesky, then `LDL^T` with a warning), never `scipy.linalg.solve` directly;
  a sparse form solves with SuperLU.
- Check an optional pandas index with `is None`: its truth value raises.
