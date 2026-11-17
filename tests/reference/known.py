"""Known failures: a test's failures, expected for a reason, so that the suite stays green on them and says why."""

import dataclasses

import pytest


@dataclasses.dataclass(frozen=True)
class Known:
    """
    A failure a test expects, and why.

    A strict one turns the suite red once the test passes, so that the entry is
    removed when the cause is fixed. One that comes and goes (a count near its
    threshold, which the Monte Carlo error of a run moves across) is not strict.
    """

    reason: str
    strict: bool = True


def check(failures: list[str], known: Known | None) -> None:
    """Fail with `failures`, unless they are `known`: then an expected failure (and, if strict, a failure when there are none)."""
    if known is None:
        assert not failures, failures
    elif failures:
        pytest.xfail(f"{known.reason} ({'; '.join(failures)})")
    elif known.strict:
        pytest.fail("a known failure passes: remove it from KNOWN")
