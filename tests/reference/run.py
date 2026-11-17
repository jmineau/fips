"""
One fips solve of a reference problem, as a user makes it, run in a worker process of the suite's pool.

Each function takes names and numbers, not objects, so that a worker builds
the problem itself; it returns plain values. A solve is a default
``"gauss_newton"`` solve with its own seed. Its answer is read as a user
reads it: the kept draws, which its mean, variances and intervals come from
(the solve's check of its chains runs when they are made), and the intervals
of the elements and of the aggregates' sums of the state
(:meth:`fips.Posterior.uncertainty`, which checks the draws in their tails,
and on the aggregates reads :meth:`fips.Posterior.cov`, which checks the
chains on each). Every :class:`fips.FipsWarning` met is kept, and every run
of the model counted.
"""

import time
import warnings

import numpy as np

import fips
from fips.problem import Problem
from tests.reference.cases import case
from tests.reference.summaries import summarize

MORE_STEPS = 500
"""The Gauss-Newton steps allowed a calibration world that does not converge in the default 50."""


class _Counted:
    """Count the runs of the model of every problem in this process (``Problem._apply``, which the method calls for each)."""

    def __enter__(self):
        self.calls = 0
        self._apply = Problem._apply
        apply = self._apply

        def counted(problem, values):
            self.calls += 1
            return apply(problem, values)

        Problem._apply = counted  # pyrefly: ignore[bad-assignment]
        return self

    def __exit__(self, *exc):
        Problem._apply = self._apply  # pyrefly: ignore[bad-assignment]


def _solved(problem: Problem, seed: int, W, **options) -> tuple[np.ndarray, dict]:
    """The kept draws of a default solve of `problem` with `seed`, and what it warned and cost."""
    with _Counted() as counted, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", fips.FipsWarning)
        start = time.perf_counter()
        post = fips.solve(problem, "gauss_newton", seed=seed, **options)
        kept = np.asarray(post._solution.kept())  # the draws fips reads its answer from
        post.uncertainty()  # the intervals a user reads, which check the tails' draws
        post.uncertainty(W)  # and the aggregates', which read cov(W) first
        seconds = time.perf_counter() - start
    warned = [
        str(w.message).splitlines()[0]
        for w in caught
        if issubclass(w.category, fips.FipsWarning)
    ]
    return kept, dict(warned=warned, calls=counted.calls, seconds=seconds)


def solve(name: str, seed: int, **options) -> dict:
    """A default solve of problem `name` with `seed` (and `options`): its summary, as a dict, and what it warned and cost."""
    made = case(name)
    kept, info = _solved(made.problem, seed, made.state_aggregation(), **options)
    return dict(summary=summarize(made.arrays, kept).to_dict(), **info)


def rank(name: str, world: int, thin: int) -> dict:
    """
    A calibration world of problem `name`: a truth and data drawn from the problem (:meth:`~tests.reference.cases.Arrays.world`), a default solve of the data with the seed ``world + 1``, and the truth's rank among every `thin`-th kept draw, for each element and aggregate.

    The seed is the world's own: with one seed in every world, a seed's error
    repeats in each and the worlds do not average it out. A world whose
    Gauss-Newton iterations do not converge in the default steps is solved
    again with :data:`MORE_STEPS` and counted (``unconverged``): leaving it out
    would leave out the worlds where the posterior is hardest, and bias the
    ranks.
    """
    made = case(name)
    truth, arrays = made.arrays.world(10_000 + world)
    problem, W = made.with_data(arrays.z), made.state_aggregation()
    try:
        kept, info = _solved(problem, world + 1, W)
        unconverged = False
    except fips.ConvergenceError:  # as its message advises, more steps
        kept, info = _solved(problem, world + 1, W, max_iterations=MORE_STEPS)
        unconverged = True
    draws = kept[:, thin - 1 :: thin]
    values = arrays.value(draws)
    truth_value = arrays.value(truth)
    aggregates = list(arrays.aggregates.values())
    ranks = [int((row < t).sum()) for row, t in zip(draws, truth, strict=True)]
    ranks += [int((w @ values < w @ truth_value).sum()) for w in aggregates]
    return dict(ranks=ranks, draws=draws.shape[1], unconverged=unconverged, **info)
