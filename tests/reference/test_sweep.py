"""
Problems of 100 to 900 elements, lognormal and softplus, against nutpie's long runs.

An area of square cells seen by observations upwind of three receptors
(:data:`tests.reference.cases.SWEEP`): a base setting of 400 cells, and
100 cells, 900 cells, a wider prior (log standard deviation 1.5) and precise
data (error 0.1). Each default solve is held, element by element and on six
sums of the values (the total, the cells observed, four quarters), to a
reference of four independent nutpie runs of 20,000 draws.
"""

import pandas as pd
import pytest

from tests.reference import report
from tests.reference.cases import SWEEP, case
from tests.reference.known import Known, check
from tests.reference.run import solve
from tests.reference.summaries import compare, read_reference

SEEDS = range(4)
"""The default solves of each problem: a seed each."""

CELLS = 0.02
"""The share of the elements' spreads, over a problem's solves, that may be more than 10% off the reference's."""


@pytest.mark.parametrize("name", SWEEP)
def test_a_larger_problem_against_nutpie(name, pool):
    """Default solves: at most 2% of the elements' spreads more than 10% off the reference's, and no solve off with no warning."""
    made = case(name)
    reference = read_reference(name, "nutpie", made.arrays)
    results = [
        future.result() for future in [pool.submit(solve, name, seed) for seed in SEEDS]
    ]
    comparisons = [compare(pd.DataFrame(r["summary"]), reference) for r in results]
    warned = [bool(r["warned"]) for r in results]
    report.ROWS.append(
        report.Row(
            name, "nutpie", 2_000, comparisons, warned, [r["calls"] for r in results]
        )
    )
    silent = [c for c, w in zip(comparisons, warned, strict=True) if c.missed and not w]
    off = sum(c.cells_off for c in comparisons) / sum(c.cells for c in comparisons)
    failures = []
    if silent:
        failures.append(f"missed with no warning: {silent}")
    if off > CELLS:
        failures.append(f"{100 * off:.1f}% of the elements' spreads off")
    check(failures, KNOWN.get(name))


KNOWN: dict[str, Known] = {}
"""The problems known to fail, and why."""
