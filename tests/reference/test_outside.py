"""
The four ten-element problems of ``tests/test_outside_sampler.py``, against two references: emcee and nutpie.

Lognormal and softplus, each mildly and strongly nonlinear. That module
holds 16 default solves to emcee runs it makes itself; here the same solves
are held to stored emcee runs and to nutpie's long runs
(:mod:`tests.reference.generate`), so a disagreement can be told from an
error of one reference: in the strong lognormal problem, two sets of eight
emcee runs once differed by 4.4 of their combined errors on one element.

Two checks: the precise one of ``tests/test_outside_sampler.py`` (each
element's mean and standard deviation, and the total's 5% and 95%
quantiles, over the 16 solves, within 4.5 combined Monte Carlo errors of the
reference's, and known to 2.5% of their scale), and, solve by solve, the
suite's 10% rule, for the table.
"""

import numpy as np
import pandas as pd
import pytest

from tests.reference import report
from tests.reference.cases import OUTSIDE, case
from tests.reference.known import Known, check
from tests.reference.run import solve
from tests.reference.summaries import compare, read_reference
from tests.test_outside_sampler import (
    PRECISION,
    UNCHECKED,
    Disagrees,
    Estimates,
    assert_agree,
)

SOLVES = 16
"""Default solves of each problem, with the seeds 100 to 115."""


def _statistics(summary: pd.DataFrame, n: int) -> tuple[list[str], np.ndarray]:
    """The statistics ``tests/test_outside_sampler.py`` compares, from a summary: their names and values (or, for a reference, the columns given)."""
    names = [f"mean[{i}]" for i in range(n)] + [f"std[{i}]" for i in range(n)]
    names += ["total q05", "total q95"]
    elements = [f"u[{i}]" for i in range(n)]
    values = np.concatenate(
        [
            summary.loc[elements, "mean"],
            summary.loc[elements, "sd"],
            summary.loc[["total"], "q05"],
            summary.loc[["total"], "q95"],
        ]
    )
    return names, values


def _errors(reference: pd.DataFrame, n: int) -> np.ndarray:
    """The reference's Monte Carlo errors of the statistics of :func:`_statistics`."""
    elements = [f"u[{i}]" for i in range(n)]
    return np.concatenate(
        [
            reference.loc[elements, "mean_se"],
            reference.loc[elements, "sd_se"],
            reference.loc[["total"], "q05_se"],
            reference.loc[["total"], "q95_se"],
        ]
    )


@pytest.mark.parametrize("source", ["emcee", "nutpie"])
@pytest.mark.parametrize("name", OUTSIDE)
def test_a_ten_element_problem_against_a_reference(name, source, pool):
    """16 default solves: their statistics within 4.5 combined errors of the reference's and known to 2.5% of their scale; no solve off by 10% with no warning."""
    made = case(name)
    reference = read_reference(name, source, made.arrays)
    results = [
        future.result()
        for future in [pool.submit(solve, name, 100 + k) for k in range(SOLVES)]
    ]
    summaries = [pd.DataFrame(r["summary"]) for r in results]
    comparisons = [compare(s, reference) for s in summaries]
    warned = [bool(r["warned"]) for r in results]
    report.ROWS.append(
        report.Row(
            name, source, 2_000, comparisons, warned, [r["calls"] for r in results]
        )
    )
    silent = [c for c, w in zip(comparisons, warned, strict=True) if c.missed and not w]
    failures = [
        f"{len(silent)} of {len(comparisons)} solves off with no warning"
    ] * bool(silent)

    n = len(made.arrays.u_0)
    names, theirs = _statistics(reference, n)
    ours = Estimates.of([_statistics(s, n)[1] for s in summaries])
    std = theirs[n : 2 * n]
    width = (theirs[-1] - theirs[-2]) / 3.29
    scale = np.concatenate([std, std, [width, width]])
    try:
        assert_agree(
            names,
            ours,
            Estimates(theirs, _errors(reference, n)),
            scale,
            UNCHECKED.get(name, ()),
            PRECISION,
        )
    except Disagrees as error:
        failures.append(str(error))
    check(failures, KNOWN.get((name, source)))


_ARM = Known(
    "in a single default solve the spread of element 0 (a thin arm, which 2,000 draws "
    "visit rarely) or of element 2 is 10% to 16% off with no warning in 3 solves of 16 "
    "(of 4 to 6 off); over the 16 solves every statistic agrees with the reference"
)
KNOWN = {
    ("lognormal-strong", "emcee"): _ARM,
    ("lognormal-strong", "nutpie"): _ARM,
    ("softplus-strong", "nutpie"): Known(
        "element 1's spread is 11% to 12% off in one solve of 16: its Monte Carlo "
        "error at the threshold (element 0 is left out: tests/reference/README.md)",
        strict=False,
    ),
}
"""The problems and references known to fail, and why."""
