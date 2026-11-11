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

The **Flexible Inverse Problem Solver**: the posterior of an inverse problem
with a Gaussian prior and Gaussian errors, `z = H x + c + e` (or `F(x)`, a
`ForwardModel` the caller writes with its Jacobian), whose state and
observations are labeled by pandas indexes in named blocks: exact for a linear
problem; with bounds on the state, its mode and draws of the Gaussian truncated
to the bounds; with a forward model, its mode by Gauss-Newton and draws of its
posterior; with priors on the covariance settings, draws of the settings and
of the state at each (`fips.hierarchical`); for a model that can only be run,
an ensemble square-root update of members drawn from the prior (`"ensemble"`); too large to factor, CG and a
low-rank posterior from products alone (`"variational"`). fips samples the posteriors it
cannot give in closed form. Labels
are matched, never positions; each matrix is kept in its cheapest form (dense,
sparse, diagonal, Kronecker factors). Beta: the API may change between
releases. PyPI and import name `fips`.

It is not a general optimizer, not a file format, and not tied to a science
domain: a flux inversion is a `Problem` whose blocks are named `flux` and
`concentration`, and its workflow, maps and STILT code live in its user's
package (no domain subpackages, no lat/lon defaults, no cartopy). fips builds
no Jacobian: the caller places one with `Operator.from_blocks`, or writes it.

## Layout

```
src/fips/
  __init__.py       the public names (`__all__`), each from the module below that owns it
  _errors.py        FipsWarning, and `warn`, which logs a warning and issues it
  space.py          Space: ordered named blocks, the padded MultiIndex, select, align
  vector.py         Vector: values on a Space (read-only), from_series/from_blocks
  operator.py       Operator: a fips.linalg form with a Space on each axis
  forward.py        Forward (what problem.forward is), ForwardModel (F and its Jacobian, or its products
                    jvp/vjp), taylortest (and the adjoint)
  transforms.py     LogNormalForward, SoftplusForward: H on a positive T(u) of one block; prior_for
                    (softplus moments by panels split at the bend: Gauss-Hermite fails past 2 scales)
  problem.py        Problem: the inputs, checked once; predict, linearized, aggregate, restrict
  posterior.py      Posterior (labels a method's answer); the Solution and Options protocols
  aggregate.py      sum, mean (an aggregation W), offset_block
  distributions.py  LogNormal, HalfNormal, Uniform: the priors of a setting, on `_Distribution`
  settings.py       hierarchical, Hierarchical: the settings by Metropolis on the exact evidence,
                    the state drawn at each kept step; `_Built` (the caller's function) or
                    `_Scales` (a Problem: the fast path)
  visualization.py  plot_comparison, the one plot (the `viz` extra: matplotlib)
  covariance/       Covariance and the models that write one down (public)
    covariance.py   Covariance: diagonal, block_diag, variances, solve, logdet, sqrt, sample,
                    aggregate (W S W^T)
    models.py       CorrelationModel, Independent, Kronecker, CovarianceModel
                    (`with_std` takes a number, a Vector or a Series, never an array)
    kernels.py      Exponential, Gaussian, Matern, Constant; within= groups
    distances.py    haversine_matrix, chordal_matrix, time_diff_matrix
  methods/          solve, the method table (_METHODS), one module per method
    analytic.py     AnalyticOptions; its two Solutions, Whitened (m x m) and
                    Information (n x n); `_choose` picks the form by size
    bounded.py      BoundedOptions; Bounded: the mode by an active set on the analytic
                    posterior, draws by exact HMC (_Sampler) or rectified (`_mode` at analytic draws)
    gauss_newton.py GaussNewtonOptions; GaussNewton: the mode by linearizations (Newton for a transform), draws by HMC
    ensemble.py     EnsembleOptions; Ensemble: batch EnSRF of the members and their runs (no Jacobian)
    variational.py  VariationalOptions; Variational: CG on the whitened system, Lanczos pairs (rank)
    _draws.py       a chain's autocorrelation and the error of its standard deviations; the warning past 10%
  linalg/           the unlabeled forms (public as fips.linalg)
    forms.py        LinearOperator and the Dense-, Sparse-, Diagonal- and MatrixFreeOperator it builds
    block.py        BlockDiagonalOperator
    kronecker.py    KroneckerOperator: a covariance kept as its factors
tests/              pytest; test_oracle.py holds every exact method to the closed form
docs/               Sphinx: guides, reference/ (by topic), examples/ (notebooks)
```

Imports run one way: `space` < `vector`, `linalg` < `operator` < `forward` < `transforms`, `covariance` <
`problem` < `posterior` < `methods` < `settings`, with `aggregate` and `visualization` on top; `distributions`
and `_errors` import nothing of fips's. No import inside a function to dodge a cycle, none under `TYPE_CHECKING`.
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
`Posterior` (`"gauss_newton"` for a `ForwardModel` in place of `H`). **An `Operator` is a
forward model** whose Jacobian is itself (`H(x)` is `H @ x`, `H.jacobian(x)` is `H`), so
`Problem.forward` is one type, `fips.forward.Forward`, a protocol both answer. A repeated
inversion is a function that builds the `Problem`.

## Invariants

- **Labels are matched exactly, and kept as given.** `Problem.__init__` is the
  one place a solve's inputs are checked: `forward` (or a model's `rows` and
  `cols`), each covariance and the offset must have exactly the labels of the
  observations or the prior, in any order, or it raises naming the labels on
  one side only (`Space._align`).
  Nothing is zero-filled, converted to numbers or rounded unless the caller
  asks (`round_labels`). Duplicate, missing and unnamed labels raise. A prior
  error covariance that is zero everywhere raises; some zero variances do not.
- **`block` is the first level, and `Space` owns the padding.** A level one
  block lacks is NaN on its rows; `Space.index` pads (keeping each level's
  dtype) and `Space.from_index`/`select` strip, so `v[block]` and `op[r, c]`
  come back labeled as given. A one-level MultiIndex is held as its plain Index.
- **An operator's labels are beside its matrix.** `Operator.data` is a
  `fips.linalg` form in the order of `rows` and `cols`; every product, sum and
  placement aligns the other side first, and a result is built with `_like`
  (which keeps a `Covariance` a `Covariance`). Reorder with `form._select`,
  which keeps a Kronecker or block-diagonal form. A `ForwardModel` forward is
  kept as given; code that needs `H` reads `problem._operator` (refuses it).
- **A structure owns its data.** `Vector`, `Operator` and every form copy their
  input and make the copy read-only. A form built from arrays fips just made,
  or from a parent's read-only arrays, uses `_adopt` (no second copy). A `MatrixFreeOperator`
  keeps the SciPy operator it wraps; what is built from it stays products (`_MatrixFree*`).
- **A covariance is symmetric and positive semidefinite.** `Covariance(...)`
  (so `from_frame`, `from_blocks` and `Covariance.diagonal`) checks a dense or
  sparse matrix's symmetry to 1e-10 of its largest entry, without densifying,
  and `KroneckerOperator` its factors. It then builds `form._root`, which
  refuses an eigenvalue below zero by more than `n * eps * max|lambda|`, for
  every fips form, and keeps it for `sample`. What fips computes from
  covariances or models (a sum, a reordering, a block diagonal, `W S W^T`, a
  posterior, a model's `build`) is built with `Covariance._computed`, which
  skips both checks: its roundoff can come near 1e-10, and a model is
  positive semidefinite by construction. A variance a posterior reports as a
  standard deviation goes through `posterior._std`: below zero by up to
  `sqrt(eps)` of `(sum_i |x_i| sigma_i)^2` it is 0, further below it is NaN, logged.
- **Sparse stays sparse.** A sparse matrix is a SciPy matrix, never a
  pandas-sparse frame (one enters only through `Operator.from_frame`, with a
  fill of 0 or NaN). Products and sums keep the cheapest exact form; a product
  past a quarter full is dense (the rule is in `linalg/__init__.py`). Compute
  with `.data`, not `dense()` or `to_frame()`, except where a dense matrix is
  the point: the analytic method densifies `H S_0` (`m x n`), `H S_0 H^T` and a
  non-diagonal `S_z` (`m x m`; Gauss-Newton, each step's `J S_0`, `J S_0 J^T`),
  and `post.cov()` forms `n x n` only when read. `"variational"` forms nothing `n x n` or `m x m`.
- **The problem is the question, the posterior the answer.** `Problem` holds
  what needs no solve and no `m x m` matrix (`predict`; `linearized`,
  `aggregate`, `restrict` make a new problem through `__init__`; `restrict`
  refuses a model); `Posterior` what depends on the solve, built once, kept.
- **The posterior labels; the method computes.** A method's solution answers
  `posterior.Solution` with unlabeled arrays in the problem's order;
  `Posterior` labels them and writes the rest (averaging kernel, innovation
  covariance, Desroziers, leverage, scalars) from them.
- **A sampled answer's mean and intervals are its draws'**: a solution that keeps draws (`kept()`: `"bounded"` with a
  bound, `"gauss_newton"` with a model) gives `mean` as their mean beside the mode `estimate`, and every interval
  (`uncertainty`'s and `trials`' `lower`, `upper`: `Posterior._interval`) their quantiles; any other's is `estimate +- z sd`.
- **Covariance models** are frozen dataclasses checked in `__init__`. The 0.1
  builders' covariances are pinned bit for bit in `tests/test_covariance.py`:
  keep the order of the products (`(std_i * C_ij) * std_j`).
- **`rtol=` conditions each matrix a kernel builds** (`_Kernel._conditioned`:
  the whole correlation, each `within=` group, a Kronecker factor on its own
  coordinates): eigenvalues below `rtol * max` are raised to that floor and
  the matrix is rescaled to a unit diagonal, with a WARNING if that shrinks
  correlations past 1%. Without `rtol` no value changes. **Only a solve warns
  of a nearly singular matrix**, on the final mismatch (`_warn_if_ill_conditioned`,
  above condition number `1e10`, of the correlation: standard deviations cost
  no digits), since a kernel cannot tell a prior from a mismatch or see a sum.
  `_solve_psd`'s `LDL^T` stays the last fallback.
- An aggregation has one row per group that holds a label; `Problem` and
  `Posterior` pickle with plain `pickle` (a posterior never stores an output).
- **Logging, not printing.** A module that logs uses
  `logger = logging.getLogger(__name__)`; `__init__.py` adds a `NullHandler`.
  A warning goes through `_errors.warn(logger, msg, *args)`, which logs it and
  issues a `FipsWarning` at the user's line (never `logger.warning` alone; INFO
  and DEBUG stay log-only). The tests ignore `FipsWarning` (`pyproject.toml`) and
  check a warning's log line with `caplog`; `tests/test_warnings.py` the warning.
  A report is returned (`summary()` is a DataFrame), and a plot returns its
  figure and axes without `plt.show()`.

## Adding a method

A module `methods/<name>.py` with three things: its options (a frozen dataclass
with a `gamma` field, values checked in `__post_init__`), its solution (answers
`posterior.Solution`; `with_gamma`, `with_data`, `sample`, `kept` (a sampler's kept draws), `at_bound`, `estimate_is_mean`,
`estimate_is_linear`, `evidence_is_laplace` (`no_evidence`, why not), and `innovation_precision` with `self_sensitivity` (for
`Posterior.data_worth`) only if it can; a member it lacks raises `fips.UnavailableError`), and a function `(problem, options) -> Posterior`. Add a line to
`_METHODS` in `methods/__init__.py`; its docstring says how a lognormal or an
ensemble method fits. **A solve has a `seed` (an int, in its options), a draw takes an
`rng`** (`int | Generator | None`, through `np.random.default_rng`): `sample`, `simulate`,
`ensemble`, `trials`, `taylortest`, `dottest`. **A method's default options on a linear, unbounded
problem give the analytic posterior** (`"bounded"` with no finite bound and
`"gauss_newton"` with an operator return the analytic solution itself):
`tests/test_oracle.py` runs every registered method, with its defaults, against the closed-form posterior and the
permutation test, except one registered `exact=False` (`"ensemble"`), held instead
to its update's exactness for its own covariance and a `1/sqrt(N)` curve (`tests/test_ensemble.py`);
an iterative one is solved with its `_Method.oracle` options, held at its `oracle_rtol` (`"variational"`). A method that samples is held to exact answers in small cases
(`truncnorm`, integration; Monte Carlo errors counting autocorrelation), to the
truth's rank among its draws in its own world, and to wrong variants that fail
them (`tests/test_bounded.py`, `tests/test_gauss_newton.py`).

## Dev commands

Driven by `just` and `uv`; CI runs the same recipes.

| Command | What it does |
|---|---|
| `just sync` | `uv sync`: fips, its `viz` extra, and the dev tools |
| `just quality-check` | `lint` + `type-check` + `docstr` + `test`: what CI checks |
| `just test` | pytest in parallel, skipping `network`/`slow` (`-n 0` for serial); also the docstring examples |
| `just lint` / `just format` | ruff check and format check / fix and format |
| `just type-check` | pyrefly (no baseline: every error fails) |
| `just docstr` | docstr-coverage of the public API |
| `just build-docs` | Sphinx, running every example; warnings are errors |
| `just pre-commit`, `just dist` | every hook on every file; build and check the dists |

`just test` also runs the `>>>` examples in docstrings under `src/`
(`--doctest-modules`), so an example that stops matching the code fails the
tests; one that needs data CI lacks gets `# doctest: +SKIP` on its `>>>` line.
Workflows: `tests.yml`, `quality.yml`, `docs.yml` (versioned docs on gh-pages),
`lock-upgrade.yml` (monthly `uv lock --upgrade` pull request), `publish.yml` (a
`vX.Y.Z` tag publishes). The version comes from git tags; the maintainer cuts
releases. A pull request leaves `CHANGELOG.md` alone: at release `just
changelog` drafts the entries from the commit messages (`cliff.toml`), so write
each message as the line a user would read. Tooling comes from
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
if `covariance/kernels.py` outgrows its kernels. fips writes the samplers its
methods need (HMC: exact for the bounded posterior, split at the Gaussian at the mode for a model) on NumPy and SciPy. The other inverse-problem libraries (pyOE, hippylib, pyemu, the
Yadav and Michalak code) are test oracles, not dependencies. A new runtime
dependency needs a reason, proposed first.

## Gotchas

- **Never run `python` from inside `src/fips`**: `operator.py` there shadows
  the standard library's `operator`.
- **A draw is `form._root @ w`.** `LinearOperator._root` (cached, left out of
  `__getstate__`) is built by `_make_root`: the Cholesky factor of the dense
  matrix, or its `eigh` root when it is only semi-definite (one INFO line); a
  diagonal, sparse (one factor per connected group), block diagonal and
  Kronecker form (`n x N`, on the whole grid, then its rows) override it so a
  draw never forms `n x n`. A new form with a cheap root overrides `_make_root`.
  `Covariance(...)` builds the root (its positive semidefinite check), so its
  INFO line comes when the covariance is built, not at the first draw.
- **Docs:** an `.. ipython::` block runs only the first line of a multi-line
  statement that is its 10th (or 100th), so split long blocks; napoleon reads a colon in an attribute docstring's first
  line as a type; docstr-coverage counts each `@overload` (give it a one-liner
  with `# noqa: D418`); a `.. currentmodule::` goes after an `.. automodule::`.
- **SciPy is not annotated**: pyrefly takes a `LinearOperator` default's private
  return class as the type an override must return. `linalg/forms.py` marks
  those lines; call `form._transpose()` rather than `form.T` to keep a fips form.
- **One `pd.Grouper(freq=...)` key numbers its bins in time order, empty bins
  included**: `aggregate._key_labels` reads each label's bin from
  `size().index`; do the same wherever one frequency key is grouped.
- **The analytic method has two forms and chooses by size, with no option.**
  With fewer state elements than observations (and an `S_z` that is dense,
  diagonal, sparse, block diagonal or Kronecker, applied through its
  factors), it takes the information form (`Information`): `S_0 = R R^T` by
  the form's own `sqrt` (through the factors for a Kronecker `S_0`), and it
  factors the `n x n` matrix `R^T H^T S_z^-1 H R`, holding no `m x m`
  of its own. Otherwise, or when `S_0` has no Cholesky factor, the whitened form. Both give the same
  posterior; the oracle and `tests/test_whitened.py` run every case in both
  (the `analytic_form` fixture replaces `_choose`).
- **A solve factors by Cholesky at its gamma; the first `with_gamma`
  eigendecomposes** (`_decompose`). Both forms need `(A + I/gamma)^-1` for
  their matrix `A` (`M` or `N`). `fips.solve` takes it from a Cholesky
  (eigenvalues None, weights 1, DOFS computed with it), after checking that
  `A` is positive semidefinite to roundoff with a second Cholesky of `A + tau I`
  (`_cholesky_inverse`); an `A` that fails is eigendecomposed, so `_clip`
  warns for the same priors as before. The first `with_gamma` eigendecomposes
  (`_eigh`: `evd`, redone with `evr` on `MemoryError`) and keeps it outside
  the frozen fields (not pickled). Every output of the Cholesky must equal the
  eigendecomposition's at the same gamma: `tests/test_whitened.py` holds them at 1e-10.
  Only a whitened solve with more than `_LARGE_M` observations and more
  than twice `n` logs a warning, saying why the information form was not used.
- **The whitened form whitens with the Cholesky factor of `S_z`**, so `S_z`
  must be positive-definite; it never factors `H S_0 H^T + S_z`. Other
  covariance solves go through `LinearOperator.solve` or `_solve_psd`
  (Cholesky, then `LDL^T` with a warning), never `scipy.linalg.solve` directly;
  a sparse form solves with SuperLU. A form keeps no factorization of its own
  (Kronecker keeps its factors); a loop of solves with one `S_z` takes
  `S_z._solver()`, which factors once and returns the substitution.
- **A constructor refuses a labeled object.** `Operator(op, rows, cols)`,
  `Covariance(cov, space)` and `Vector(v, space)` refuse a pandas object, an
  `Operator` or a `Vector`, whose labels they would ignore: use `from_frame`,
  `align`, or `.data` / `.values`.
- Check an optional pandas index with `is None`: its truth value raises.
- **The bounded method needs `S_hat` only at some columns** (`_Columns`, from
  `analytic._cov_times`, kept; one cache for every `with_data`): the active set's
  and those the sampler's paths (or the rectified draws' active sets) reach. They are computed in fixed blocks of 64
  neighbors, since a product's last bits depend on its shape: a chain is then
  the same after a pickle. The draws are made at the first `variances`/`cov`
  and kept outside the frozen fields (not pickled). A draw costs one `O(n)`
  reflection per bound its path reaches, about as many as there are elements
  near a bound, so `trials` of a bounded posterior runs a sampler per trial.
- **A Gauss-Newton posterior holds the user's problem, with the model** (`modeled_obs`, `chi2`, the
  misfits: the model's at the mode; `with_data` solves again from the last linearization), and the
  Gaussian at the mode, the analytic solution linearized at the last iterate (gain, DOFS, the `H` of `A`
  and `innovation_cov`: `Posterior._linear`). `variances`, `cov(W)`, `sample`: draws (`_Sampler`, HMC
  split at that Gaussian, a Metropolis test), kept as `"bounded"`'s. Iterates and draws are `x_0 + S_0 y`:
  no `S_0^-1`. A draw is a few `F`, `J^T v` (`ForwardModel._gradient`), `S_hat v`. A step that raises the
  cost is halved. A transform's `_curvature` (diagonal; None for other models) adds Newton's step (CG
  preconditioned by `S_hat`), the lower cost taken; only a whole Gauss-Newton step (and Newton's) stops.
- **`hierarchical`'s fast path reads every step from one eigendecomposition** (`_Factors._evidence_terms`,
  `_draws_at`): at a prior-std factor `s` and a mismatch-covariance factor `c` the problem is the unscaled
  one at `gamma = s^2 / c`, with chi2 over `s^2` and `m ln c` in the evidence (the information form's chi2
  is `gamma q - gamma sum w b^2`, `q = d^T S_z^-1 d`). Its draws take each draw's normals in turn, as
  `sample(1)` would, so one seed gives the fast and the general path the same chain and draws
  (`tests/test_hierarchical.py`). The chain and the state's draws have their own streams (`spawn(2)`), and
  `_accept` takes one uniform a step, so the chain does not depend on how the state is drawn.
