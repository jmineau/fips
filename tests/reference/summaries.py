"""
What the suite reads from draws, the reference files that hold it, and how two summaries are compared.

A summary has a row for each element of the state, ``u[i]``, and for each
aggregate of the values ``T(u)`` (``total``, ...), and the columns of
:data:`STATISTICS`: the mean, the standard deviation, and the 5% and 95%
quantiles. A reference adds each one's Monte Carlo standard error
(``mean_se``, ...): the spread of independent runs over the root of their
number.
"""

import dataclasses
import math
from pathlib import Path

import numpy as np
import pandas as pd

from tests.reference.cases import Arrays

STATISTICS = ("mean", "sd", "q05", "q95")
"""The statistics of each element and aggregate."""

OFF = 0.10
"""
A summary is off its reference where a standard deviation differs from the reference's by more than this fraction of it, or an end of the 90% interval by more than this fraction of the interval's width (a third of a standard deviation, for a Gaussian).

The issue that started this suite (#203) was spreads 16% to 40% low.
"""

SETTLED = 0.25
"""A reference's statistic is settled when its Monte Carlo error is under this fraction of :data:`OFF` (2.5% of the standard deviation or of the width); one that is not is not compared."""

DATA = Path(__file__).parent / "data"
"""The reference files, ``<problem>.<sampler>.csv``."""


def summarize(arrays: Arrays, u: np.ndarray) -> pd.DataFrame:
    """The summary of draws `u` (``n x draws``) of the state of `arrays`'s problem: each element's, then each aggregate's of the values."""
    values = arrays.value(u)
    names = list(arrays.aggregates)
    rows = np.vstack([u, np.array([arrays.aggregates[a] @ values for a in names])])
    q05, q95 = np.quantile(rows, [0.05, 0.95], axis=1)
    index = pd.Index([f"u[{i}]" for i in range(len(u))] + names, name="quantity")
    return pd.DataFrame(
        {
            "mean": rows.mean(axis=1),
            "sd": rows.std(axis=1, ddof=1),
            "q05": q05,
            "q95": q95,
        },
        index=index,
    )


def combine(runs: list[pd.DataFrame]) -> pd.DataFrame:
    """The mean of independent runs' summaries, and each statistic's standard error: the runs' spread over the root of their number."""
    stack = np.stack([run[list(STATISTICS)].to_numpy() for run in runs])
    error = stack.std(axis=0, ddof=1) / math.sqrt(len(runs))
    frame = pd.DataFrame(stack.mean(axis=0), runs[0].index, list(STATISTICS))
    for k, name in enumerate(STATISTICS):
        frame[f"{name}_se"] = error[:, k]
    return frame


def fingerprint(arrays: Arrays) -> list[float]:
    """A few sums of the problem's arrays, kept in its reference files, so that a reference for another problem is refused."""
    return [
        float(arrays.u_0.sum()),
        float(arrays.S_0.sum()),
        float(arrays.H.sum()),
        float(arrays.z.sum()),
        float(arrays.c.sum()),
        float(arrays.r.sum()),
    ]


def write_reference(
    frame: pd.DataFrame, name: str, sampler: str, header: dict, directory=DATA
) -> Path:
    """Write a reference file: its header (``# key: value`` lines, the fingerprint among them), then the summary as CSV."""
    path = Path(directory) / f"{name}.{sampler}.csv"
    with path.open("w") as file:
        for key, value in header.items():
            file.write(f"# {key}: {value}\n")
        frame.to_csv(file, float_format="%.7g")
    return path


def read_reference(name: str, sampler: str, arrays: Arrays) -> pd.DataFrame:
    """
    The reference of problem `name` by `sampler`, after checking that it was made for these arrays.

    Raises
    ------
    AssertionError
        If its fingerprint is not that of `arrays`: the problem changed since
        the reference was made, and it must be made again
        (:mod:`tests.reference.generate`).
    """
    path = DATA / f"{name}.{sampler}.csv"
    header = {}
    with path.open() as file:
        for line in file:
            if not line.startswith("# "):
                break
            key, _, value = line[2:].partition(": ")
            header[key] = value.strip()
    kept = [float(x) for x in header["fingerprint"].strip("[]").split(",")]
    np.testing.assert_allclose(
        kept,
        fingerprint(arrays),
        rtol=1e-9,
        err_msg=f"{path.name} was made for another problem: make it again",
    )
    return pd.read_csv(path, comment="#", index_col="quantity")


@dataclasses.dataclass
class Comparison:
    """How far one solve's summary is from a reference: counts over the elements and the aggregates that the reference settles."""

    cells: int
    """The elements compared."""
    cells_off: int
    """The elements whose standard deviation is off (:data:`OFF`)."""
    intervals_off: int
    """The elements an end of whose 90% interval is off."""
    aggregates: int
    """The aggregates compared."""
    aggregates_off: list[str]
    """The aggregates whose standard deviation or interval is off."""
    unsettled: int
    """The elements and aggregates the reference does not settle (:data:`SETTLED`), left out."""
    worst: float
    """The largest relative error of an aggregate's standard deviation."""

    @property
    def missed(self) -> bool:
        """Whether the solve is wrong where a user reads it: an aggregate off, or more than 5% of the elements' spreads (where fips's own check of the chains warns)."""
        return bool(self.aggregates_off) or self.cells_off > 0.05 * self.cells


def compare(ours: pd.DataFrame, reference: pd.DataFrame) -> Comparison:
    """Compare a solve's summary with a reference's, row by row (the rows of both are the same)."""
    ours = ours.loc[reference.index]
    width = reference["q95"] - reference["q05"]
    ratio = ours["sd"] / reference["sd"] - 1.0
    sd_off = ratio.abs() > OFF
    end = np.maximum(
        (ours["q05"] - reference["q05"]).abs(), (ours["q95"] - reference["q95"]).abs()
    )
    end_off = end > OFF * width
    settled = pd.Series(True, index=reference.index)
    if "sd_se" in reference:
        settled = (reference["sd_se"] < SETTLED * OFF * reference["sd"]) & (
            np.maximum(reference["q05_se"], reference["q95_se"]) < SETTLED * OFF * width
        )
    cell = reference.index.str.startswith("u[")
    aggregate = ~cell & settled
    agg_off = aggregate & (sd_off | end_off)
    return Comparison(
        cells=int((cell & settled).sum()),
        cells_off=int((cell & settled & sd_off).sum()),
        intervals_off=int((cell & settled & end_off).sum()),
        aggregates=int(aggregate.sum()),
        aggregates_off=list(reference.index[agg_off]),
        unsettled=int((~settled).sum()),
        worst=float(ratio[aggregate].abs().max()) if aggregate.any() else 0.0,
    )
