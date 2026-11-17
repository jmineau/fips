# The reference suite

fips's nonlinear solves (`"gauss_newton"` with a forward model, `LogNormalForward`
or `SoftplusForward`) give their answer from draws of the posterior. The tests
beside the code check those draws against references written next to them; this
suite holds them to answers fips did not compute, on problems from two elements to
900, and reports one table a release can quote.

It is not part of the default or slow runs (`just test`, `just test-slow`) and not
of CI: its tests are marked `reference`, and it takes a few hours of solves.

## What it checks

| File | Problems | Held to |
|---|---|---|
| `test_curves.py` | issue #203's two curves of two elements (`e^a + 2 e^b`; the user guide's two cells) | the posterior on a grid |
| `test_outside.py` | the four ten-element problems of `tests/test_outside_sampler.py` (lognormal and softplus, mild and strong) | emcee's runs and nutpie's long runs, each a reference file |
| `test_sweep.py` | 100 to 900 elements, both transforms: a base setting of 400 cells and four changes of it (`cases.SWEEP`) | nutpie's long runs, a reference file |
| `test_calibration.py` | the ten-element problems and the 100-cell ones | their own definition: simulation-based calibration (Talts et al., 2018) |

Every solve is a default `"gauss_newton"` solve with its own seed, read as a user
reads it: each element's and each aggregate's mean, standard deviation and 5% and
95% quantiles from the kept draws, every `FipsWarning` kept. A solve's statistic
is **off** where a standard deviation differs from the reference's by more than
10%, or an end of the 90% interval by more than 10% of the interval's width; a
solve **misses** where an aggregate is off or more than 5% of the elements' spreads
are, and a miss **with no warning** is the failure the suite exists to find. A
reference statistic whose own Monte Carlo error is above 2.5% is reported as
unsettled and left out.

In calibration each world's truth and data are drawn from the problem by NumPy and
solved with the world's own seed (a seed shared by every world repeats its error in
each, and the worlds do not average it out); the truth's rank among every 20th kept
draw is tested for uniformity, element by element and aggregate by aggregate.

The table at the end has a row for each problem and reference: the elements' spreads
off, the aggregates off and the worst one, the misses with no warning, the model
runs a solve, and a row for each calibration.

## Running it

```bash
just test-reference
```

runs it on this machine, spreading the solves over every core it may use
(`FIPS_REFERENCE_WORKERS` sets how many; pytest itself runs serially, with no `-n`).
On a SLURM cluster, submit it from the repository's root:

```bash
sbatch -A <account> -p <partition> tests/reference/run.sbatch
```

which asks for 32 cores for 6 hours, and writes the table to `reference-<job id>.md`.
A known failure is an expected one (`KNOWN` in its test module, with the reason),
so the suite turns red when it passes.

The two references of the ten-element problems agree on every statistic but one:
the standard deviation of element 0 in the strongly nonlinear problems. That element
has a thin arm towards large negative values, which a run visits rarely. Its 5% and
95% quantiles agree, but single runs scatter (in the softplus problem, nutpie's runs
gave 0.21 to 0.32 and emcee's 0.14 to 0.19). emcee's runs mostly miss the arm
together, so its error there is understated: in the lognormal problem it is 6.5%
below nutpie's, at 4.5 of their combined errors. In the softplus problem both
references mark it unsettled, and it is left out. In the lognormal problem nutpie's
value is the one to trust.

What the table leaves out there is still worth knowing: in the softplus problem,
fips's single solves gave element 0 a spread 25% to 45% below nutpie's in 9 of 16
solves, 6 of them with no warning. A solve of 2,000 draws usually misses the arm too.
Its 90% interval agrees with both references.

A seed fixes a solve on one machine, not across machines. Floating-point rounding
differs between processors (and their BLAS), and a long chain amplifies it, so two
machines give different counts near a threshold. A known failure that a run's
Monte Carlo error moves across is marked as one that comes and goes (`Known(...,
strict=False)`).

## Making the references again

The problems are in `cases.py`, each as a fips problem and as NumPy arrays with its
log density, which the reference samplers draw. `generate.py` makes the reference
files of `data/` (`<problem>.<sampler>.csv`: each statistic and its Monte Carlo
error, under a header saying how they were made). A file whose problem has changed
is refused by its fingerprint, and must be made again:

```bash
n=$(uv run --with nutpie==0.16.11 python -m tests.reference.generate jobs | wc -l)
sbatch -A <account> -p <partition> --array=0-$((n - 1)) tests/reference/generate.sbatch <directory>
uv run --with nutpie==0.16.11 python -m tests.reference.generate combine <directory>
```

The first line counts the runs (independent emcee and nutpie runs of each problem),
the second makes each in a job of its own, writing its summary into `<directory>`
(minutes for the ten-element problems, up to a few hours for the 900-cell ones), and
the last, once they have all ended, writes the reference files. nutpie is not one of
fips's dependencies; `--with` installs it for these commands only.

## References

Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2018).
Validating Bayesian inference algorithms with simulation-based calibration.
arXiv:1804.06788.
