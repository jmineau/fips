"""
Make the reference files of ``tests/reference/data`` again, with samplers that share no code with fips.

Each reference is several independent runs of a sampler on the problem's log
density written in NumPy (:meth:`~tests.reference.cases.Arrays.log_density`),
summarized one by one and then combined: the runs' mean, and their spread
over the root of their number as each statistic's Monte Carlo error.

- ``emcee`` (the ten-element problems only): emcee's ensemble sampler with
  its differential-evolution move, run as ``tests/test_outside_sampler.py``
  runs it.
- ``nutpie``: nutpie's NUTS (the Rust sampler behind PyMC's fast backend),
  with small steps (target acceptance 0.99) and a long warm-up, in
  coordinates that whiten the prior (the ten-element problems) or the Gaussian
  at the mode found by L-BFGS here (the sweep's problems). A change of
  coordinates changes the cost of the draws, not their distribution.

Usage, from the repository's root (nutpie is not one of fips's dependencies)::

    uv run --with nutpie python -m tests.reference.generate jobs
    uv run --with nutpie python -m tests.reference.generate run <k> <directory>
    uv run --with nutpie python -m tests.reference.generate combine <directory>

``jobs`` lists the runs, one a line, numbered from 0; ``run`` makes run
``k`` of that list and writes its summary into `directory`; ``combine``
writes the reference file of every problem whose runs are all in
`directory`. A run takes minutes (emcee, nutpie on ten elements) to two hours
(nutpie on the largest), on one core: run the list as an array of jobs, as
``tests/reference/README.md`` shows.
"""

import datetime
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from tests.reference.cases import OUTSIDE, SWEEP, Arrays, case
from tests.reference.summaries import (
    combine,
    fingerprint,
    summarize,
    write_reference,
)

EMCEE = dict(runs=8, walkers=32, burn=5_000, thin=10)
"""emcee's runs: as many, as wide and as long as ``tests/test_outside_sampler.py``'s; the steps are by problem (:func:`_emcee_steps`)."""

NUTPIE = {
    "outside": dict(runs=8, chains=4, draws=10_000, tune=2_000, coords="prior"),
    "sweep": dict(runs=4, chains=4, draws=5_000, tune=1_500, coords="laplace"),
}
"""nutpie's runs, by family of problem: each run's chains, draws a chain, warm-up a chain and coordinates."""

TARGET = 0.99
"""nutpie's target acceptance: small steps, which a curved posterior needs."""


def _emcee_steps(name: str) -> int:
    """Emcee's steps a run: 120,000 for a strongly nonlinear problem, 30,000 for a mild one."""
    return 120_000 if name.endswith("strong") else 30_000


def jobs() -> list[tuple[str, str, int]]:
    """Every run to make: ``(problem, sampler, run)``."""
    listed = []
    for name in OUTSIDE:
        listed += [(name, "emcee", k) for k in range(EMCEE["runs"])]
        listed += [(name, "nutpie", k) for k in range(NUTPIE["outside"]["runs"])]
    for name in SWEEP:
        listed += [(name, "nutpie", k) for k in range(NUTPIE["sweep"]["runs"])]
    return listed


def _settings(name: str, sampler: str) -> dict:
    """The settings of `sampler`'s runs on problem `name`."""
    if sampler == "emcee":
        return {**EMCEE, "steps": _emcee_steps(name)}
    return {**NUTPIE["outside" if name in OUTSIDE else "sweep"], "target": TARGET}


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """The split R-hat of each row of `chains` (``chains x draws x n``), Gelman et al.'s, on halves of each chain."""
    half = chains.shape[1] // 2
    parts = np.concatenate([chains[:, :half], chains[:, half : 2 * half]])
    within = parts.var(axis=1, ddof=1).mean(axis=0)
    between = half * parts.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((half - 1) / half * within + between / half) / within)


def run_emcee(name: str, seed: int, settings: dict) -> tuple:
    """One emcee run (``tests/test_outside_sampler.py``'s), from a ball about the mode: its draws (``n x draws``) and what to report of it."""
    from tests.test_outside_sampler import REGIMES, Transformed, emcee_draws

    made = Transformed.drawn(**REGIMES[name], seed=7)
    mode = made.mode()
    spread = np.sqrt(np.diag(made.laplace(mode)))
    draws, tau = emcee_draws(
        made.log_prob,
        mode,
        0.1 * spread,
        seed,
        settings["walkers"],
        settings["steps"],
        settings["burn"],
    )
    kept = draws[:, :: settings["thin"]]
    return kept, dict(tau=round(tau, 1), steps_over_tau=round(settings["steps"] / tau))


def run_nutpie(arrays: Arrays, seed: int, settings: dict) -> tuple:
    """
    One nutpie run in whitened coordinates ``u = m + L w``: its draws (``n x draws``) and what to report of it.

    ``m`` and ``L L^T`` are the prior's (``coords="prior"``) or the Gaussian
    at the mode's (``"laplace"``). The chains start near the mode.
    """
    import nutpie
    from nutpie.compiled_pyfunc import from_pyfunc

    n = len(arrays.u_0)
    mode, laplace = arrays.laplace()
    if settings["coords"] == "prior":
        m, L = arrays.u_0, np.linalg.cholesky(arrays.S_0)
    else:
        m, L = mode, np.linalg.cholesky(laplace)
    w_mode = np.linalg.solve(L, mode - m)

    def make_logp():
        def logp(w):
            try:
                with np.errstate(over="raise", invalid="raise"):
                    value, gradient = arrays.log_density(m + L @ w)
            except FloatingPointError:
                return -np.inf, np.zeros(n)
            return value, L.T @ gradient

        return logp

    def make_expand(*_):
        return lambda w: {"u": m + L @ w}

    def initial(start_seed):
        rng = np.random.default_rng([seed, start_seed])
        return w_mode + 0.1 * rng.standard_normal(n)

    model = from_pyfunc(
        n,
        make_logp,
        make_expand,
        [np.dtype("float64")],
        [(n,)],
        ["u"],
        make_initial_point_fn=initial,
    )
    trace = nutpie.sample(
        model,
        draws=settings["draws"],
        tune=settings["tune"],
        chains=settings["chains"],
        cores=1,
        seed=seed,
        progress_bar=False,
        save_warmup=False,
        target_accept=settings["target"],
    )
    chains = trace["posterior"]["u"].values  # chain x draw x n
    stats = trace["sample_stats"]
    report = dict(
        nutpie=nutpie.__version__,
        rhat=round(float(split_rhat(chains).max()), 4),
        divergences=int(stats["diverging"].values.sum()),
        steps=round(float(stats["n_steps"].values.mean()), 1),
    )
    return np.concatenate(list(chains), axis=0).T, report


def run(k: int, directory: Path) -> Path:
    """Make run `k` of :func:`jobs` and write its summary into `directory`."""
    name, sampler, index = jobs()[k]
    arrays = case(name).arrays
    settings = _settings(name, sampler)
    seed = 1000 + index
    start = time.perf_counter()
    if sampler == "emcee":
        draws, report = run_emcee(name, seed, settings)
    else:
        draws, report = run_nutpie(arrays, seed, settings)
    report["seconds"] = round(time.perf_counter() - start)
    directory.mkdir(parents=True, exist_ok=True)
    return write_reference(
        summarize(arrays, draws),
        name,
        f"{sampler}.{index}",
        {"run": report},
        directory,
    )


def combine_all(directory: Path) -> list[Path]:
    """Write the reference file of every problem whose runs are all in `directory`."""
    made = []
    by_problem: dict[tuple[str, str], list[int]] = {}
    for name, sampler, index in jobs():
        by_problem.setdefault((name, sampler), []).append(index)
    for (name, sampler), indices in by_problem.items():
        paths = [directory / f"{name}.{sampler}.{k}.csv" for k in indices]
        if not all(path.exists() for path in paths):
            continue
        reports = [
            path.read_text().splitlines()[0].removeprefix("# run: ") for path in paths
        ]
        runs = [pd.read_csv(path, comment="#", index_col="quantity") for path in paths]
        arrays = case(name).arrays
        header = {
            "problem": name,
            "sampler": sampler,
            "settings": _settings(name, sampler),
            "runs": "; ".join(reports),
            "made": datetime.date.today().isoformat(),
            "fingerprint": fingerprint(arrays),
        }
        made.append(write_reference(combine(runs), name, sampler, header))
    return made


def main(argv: list[str]) -> None:
    """The command line: ``jobs``, ``run <k> <directory>`` or ``combine <directory>``."""
    if argv[:1] == ["jobs"]:
        for k, job in enumerate(jobs()):
            print(k, *job)
    elif argv[:1] == ["run"] and len(argv) == 3:
        print(run(int(argv[1]), Path(argv[2])))
    elif argv[:1] == ["combine"] and len(argv) == 2:
        for path in combine_all(Path(argv[1])):
            print(path)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
