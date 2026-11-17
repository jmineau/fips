"""
Issue #203's two curves, against the posterior on a grid.

``e^a + 2 e^b`` observed as 4 with error 0.15 (a :class:`~fips.ForwardModel`
the test writes) and the user guide's two cells, whose total ``e^a + e^b`` is
observed as 2 to 2.5% (:class:`~fips.LogNormalForward`): the states that give
the observation lie on a curve that bends across the prior's spread. Each
default solve, at 2,000 and 20,000 draws, is held to the grid's mean,
standard deviation and 5% and 95% quantiles of each element and of the total.
"""

import numpy as np
import pandas as pd
import pytest

from tests.reference import report
from tests.reference.cases import CURVES, Arrays, case
from tests.reference.known import Known, check
from tests.reference.run import solve
from tests.reference.summaries import compare

SEEDS = {2_000: range(8), 20_000: range(6)}
"""The solves of each length: a seed each."""


def grid(arrays: Arrays, lower=-8.0, upper=4.0, points=2401) -> pd.DataFrame:
    """The posterior of a two-element problem on a grid (the trapezoidal rule, exact to roundoff on this smooth density), summarized as the suite summarizes draws."""
    axis = np.linspace(lower, upper, points)
    a, b = np.meshgrid(axis, axis, indexing="ij")
    u = np.stack([a.ravel(), b.ravel()])
    P_0, P_z = arrays.precisions
    d = u - arrays.u_0[:, None]
    r = arrays.z[:, None] - arrays.H @ arrays.value(u) - arrays.c[:, None]
    log_p = -0.5 * np.einsum("ik,ij,jk->k", d, P_0, d) - 0.5 * np.einsum(
        "ik,ij,jk->k", r, P_z, r
    )
    w = np.exp(log_p - log_p.max())
    w /= w.sum()
    values = arrays.value(u)
    rows = [u[0], u[1], *(weights @ values for weights in arrays.aggregates.values())]
    names = ["u[0]", "u[1]", *arrays.aggregates]
    summary = {}
    for name, row in zip(names, rows, strict=True):
        mean = float(w @ row)
        order = np.argsort(row)
        q05, q95 = np.interp([0.05, 0.95], np.cumsum(w[order]), row[order])
        summary[name] = [mean, float(np.sqrt(w @ (row - mean) ** 2)), q05, q95]
    return pd.DataFrame.from_dict(
        summary, orient="index", columns=["mean", "sd", "q05", "q95"]
    ).rename_axis("quantity")


@pytest.mark.parametrize("draws", list(SEEDS))
@pytest.mark.parametrize("name", CURVES)
def test_a_curve_against_its_grid(name, draws, pool):
    """
    Each default solve of a curve: no element's or total's spread more than 10% off the grid's with no warning, and none at all at 20,000 draws.

    Before issue #203's fix (the sampler split at the Gaussian at the mode),
    ``b``'s spread on ``e^a + 2 e^b`` at 20,000 draws was 28% to 40% low in
    three seeds of six, with no warning.
    """
    made = case(name)
    reference = grid(made.arrays)
    results = [
        future.result()
        for future in [
            pool.submit(solve, name, seed, draws=draws) for seed in SEEDS[draws]
        ]
    ]
    comparisons = [compare(pd.DataFrame(r["summary"]), reference) for r in results]
    warned = [bool(r["warned"]) for r in results]
    report.ROWS.append(
        report.Row(
            name, "grid", draws, comparisons, warned, [r["calls"] for r in results]
        )
    )
    silent = [c for c, w in zip(comparisons, warned, strict=True) if c.missed and not w]
    failures = [
        f"{len(silent)} of {len(comparisons)} solves off with no warning"
    ] * bool(silent)
    if draws >= 20_000 and any(c.missed for c in comparisons):
        failures.append(f"{sum(c.missed for c in comparisons)} solves off")
    check(failures, KNOWN.get((name, draws)))


KNOWN: dict[tuple[str, int], Known] = {}
"""The curves and lengths known to fail, and why."""
