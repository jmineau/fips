"""The table the reference suite ends with: one row for each problem and reference, then one for each calibration."""

import dataclasses

import numpy as np

from tests.reference.summaries import Comparison


@dataclasses.dataclass
class Row:
    """A problem's solves against one reference."""

    problem: str
    reference: str
    draws: int
    comparisons: list[Comparison]
    warned: list[bool]
    calls: list[int]

    def cells(self) -> str:
        """The share of the elements' spreads more than 10% off, over every solve."""
        off = sum(c.cells_off for c in self.comparisons)
        total = sum(c.cells for c in self.comparisons)
        return f"{off}/{total} ({100 * off / max(total, 1):.1f}%)"

    def aggregates(self) -> str:
        """The aggregates off, over every solve, and the worst one's spread."""
        off = sum(len(c.aggregates_off) for c in self.comparisons)
        total = sum(c.aggregates for c in self.comparisons)
        worst = max(c.worst for c in self.comparisons)
        return f"{off}/{total} (worst {100 * worst:.0f}%)"

    def silent(self) -> str:
        """The solves that miss (:attr:`Comparison.missed`) with no warning, of those that miss."""
        missed = [c.missed for c in self.comparisons]
        silent = sum(m and not w for m, w in zip(missed, self.warned, strict=True))
        return f"{silent} of {sum(missed)} missed"

    def unsettled(self) -> int:
        """The quantities the reference does not settle, which are left out."""
        return max(c.unsettled for c in self.comparisons)


@dataclasses.dataclass
class Calibration:
    """A problem's calibration: the uniformity of the truth's ranks, element by element and aggregate by aggregate."""

    problem: str
    worlds: int
    quantities: int
    smallest_p: float
    rejected: int
    warned: int
    unconverged: int
    calls: list[int]


ROWS: list[Row] = []
"""Every comparison of the run, in order."""

CALIBRATION: list[Calibration] = []
"""Every calibration of the run, in order."""


def table() -> str:
    """The run's table, in Markdown."""
    lines = [
        "| problem | reference | solves x draws | cells off >10% | aggregates off >10% "
        "| misses with no warning | model calls a solve (median) | reference unsettled |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in ROWS:
        lines.append(
            f"| {row.problem} | {row.reference} | {len(row.comparisons)} x {row.draws} "
            f"| {row.cells()} | {row.aggregates()} | {row.silent()} "
            f"| {int(np.median(row.calls)):,} | {row.unsettled()} |"
        )
    if CALIBRATION:
        lines += [
            "",
            "| calibration | worlds | elements and aggregates | smallest p | rejected "
            "| solves warned | solves past 50 steps | model calls a solve (median) |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for c in CALIBRATION:
            lines.append(
                f"| {c.problem} | {c.worlds} | {c.quantities} | {c.smallest_p:.2g} "
                f"| {c.rejected} | {c.warned} | {c.unconverged} "
                f"| {int(np.median(c.calls)):,} |"
            )
    return "\n".join(lines)
