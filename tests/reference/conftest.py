"""
The reference suite's fixtures: the ``reference`` mark on every test here, a pool of worker processes, and the table printed at the end.

Every test of this folder is marked ``reference``, which the default and slow
runs leave out (``just test``, ``just test-slow``); ``just test-reference``
runs only them. The solves are independent, so each test hands them to one
pool of processes, one per core the run may use (``FIPS_REFERENCE_WORKERS``,
or every core this process may run on).
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from tests.reference import report


def pytest_collection_modifyitems(items):
    """Mark every test of this folder ``reference``."""
    here = Path(__file__).parent
    for item in items:
        if here in Path(item.path).parents:
            item.add_marker(pytest.mark.reference)


@pytest.fixture(scope="session")
def pool():
    """A pool of worker processes, each one BLAS thread, started afresh (not forked)."""
    workers = int(
        os.environ.get("FIPS_REFERENCE_WORKERS", len(os.sched_getaffinity(0)))
    )
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"  # read by the workers' NumPy when it starts
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=context) as executor:
        yield executor


def pytest_terminal_summary(terminalreporter):
    """Print the suite's table, and write it to ``FIPS_REFERENCE_REPORT`` if that names a file."""
    if not report.ROWS and not report.CALIBRATION:
        return
    text = report.table()
    terminalreporter.write_sep("=", "reference suite")
    terminalreporter.write_line(text)
    path = os.environ.get("FIPS_REFERENCE_REPORT")
    if path:
        Path(path).write_text(text + "\n")
