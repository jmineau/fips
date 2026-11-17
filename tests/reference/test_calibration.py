"""
Simulation-based calibration (Talts et al., 2018): in worlds drawn from the problem, the truth's rank among a solve's draws is uniform.

In each world a truth is drawn from the prior and data from it, by NumPy
(:meth:`~tests.reference.cases.Arrays.world`), and a default solve of the
data, with a seed of the world's own, gives its kept draws. If the draws are
the posterior's, the truth is one more draw of it, so its rank among them
(every 20th of the 2,000, 100 draws) is uniform on 0 to 100 over the
worlds, for each element and each aggregate. Draws too narrow pile the ranks
at both ends, draws off center at one. The ranks are counted in 10 bins and
tested by a chi-square test; a quantity is rejected below a p of 1% over the
number of quantities (Bonferroni), so a correct sampler fails a problem in
about one run of a hundred.

This needs no reference sampler: it holds the solves to the problem's own
definition. It cannot see an error that is the same in every world.

References
----------
Talts, S., Betancourt, M., Simpson, D., Vehtari, A., and Gelman, A. (2018).
Validating Bayesian inference algorithms with simulation-based calibration.
arXiv:1804.06788.
"""

import numpy as np
import pytest
from scipy.stats import chisquare

from tests.reference import report
from tests.reference.cases import OUTSIDE
from tests.reference.run import rank

THIN = 20
"""Every 20th kept draw is ranked against: 100 of the 2,000, less correlated than neighbors."""

BINS = 10
"""The rank histogram's bins."""

FAMILY = 0.01
"""The chance that a correct sampler fails a problem: each quantity is tested at this over their number."""

WORLDS = {**dict.fromkeys(OUTSIDE, 1_000), "lognormal-100": 400, "softplus-100": 400}
"""The calibrated problems and their worlds: the ten-element problems and the sweep's 100-cell ones."""


def uniformity(ranks: np.ndarray, draws: int) -> np.ndarray:
    """The chi-square test's p of each column of `ranks` (``worlds x quantities``, each in 0 to `draws`) against the uniform, in :data:`BINS` bins."""
    edges = np.linspace(0, draws + 1, BINS + 1)
    expected = np.diff(np.floor(edges)) / (draws + 1) * len(ranks)
    p = []
    for column in ranks.T:
        counts = np.histogram(column, bins=np.floor(edges))[0]
        p.append(chisquare(counts, expected).pvalue)
    return np.array(p)


@pytest.mark.parametrize("name", list(WORLDS))
def test_the_truths_rank_is_uniform(name, pool):
    """No element's or aggregate's rank histogram is rejected."""
    worlds = WORLDS[name]
    results = [
        future.result()
        for future in [pool.submit(rank, name, world, THIN) for world in range(worlds)]
    ]
    ranks = np.array([r["ranks"] for r in results])
    p = uniformity(ranks, results[0]["draws"])
    rejected = int((p < FAMILY / len(p)).sum())
    report.CALIBRATION.append(
        report.Calibration(
            name,
            worlds,
            len(p),
            float(p.min()),
            rejected,
            sum(bool(r["warned"]) for r in results),
            sum(r["unconverged"] for r in results),
            [r["calls"] for r in results],
        )
    )
    assert not rejected, f"{rejected} of {len(p)} rejected; smallest p {p.min():.2g}"
