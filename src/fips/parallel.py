"""Parallel execution utilities for multiprocessing tasks.

This module provides decorators for parallelizing function calls
across iterables with optional timeout support.
"""

import multiprocessing
import signal
from collections.abc import Callable
from functools import partial
from typing import Any, Literal


def exec_with_timeout(func, timeout, kwargs, item):
    """Execute a function with a timeout using signals.

    Parameters
    ----------
    func : callable
        Function to execute.
    timeout : float
        Timeout in seconds.
    kwargs : dict
        Keyword arguments for func.
    item : any
        Item to pass to func.

    Returns
    -------
    any
        Result of func(item, **kwargs).
    """

    def handler(signum, frame):
        raise TimeoutError(f"Task timed out after {timeout} seconds")

    # Register the signal function handler
    signal.signal(signal.SIGALRM, handler)
    # Define a timeout for the function (supports floats)
    signal.setitimer(signal.ITIMER_REAL, timeout)

    try:
        return func(item, **kwargs)
    finally:
        # Disable the alarm
        signal.setitimer(signal.ITIMER_REAL, 0)


def parallelize(
    func: Callable,
    num_processes: int | Literal["max"] = 1,
    timeout: float | None = None,
) -> Callable:
    """Parallelize a function across an iterable.

    Parameters
    ----------
    func : callable
        The function to parallelize.
    num_processes : int or 'max', optional
        Number of processes to use. If 'max', uses all available CPUs. Default is 1.
    timeout : float, optional
        Maximum time (in seconds) for each item. Default is None (no timeout).

    Returns
    -------
    callable
        Parallelized function that accepts an iterable and **kwargs.
    """

    def parallelized(iterable, **kwargs) -> list[Any]:
        """
        Execute the input function in parallel across an iterable.

        Parameters
        ----------
        iterable : iterable
            The iterable to parallelize the function across.
        **kwargs : dict
            Additional keyword arguments to pass to the function.

        Returns
        -------
        results : list
            The results of the function applied to each item in the iterable.
        """
        if len(iterable) == 0:
            raise ValueError("Iterable is empty; nothing to parallelize.")

        # Determine the number of processes to use
        cpu_count = multiprocessing.cpu_count()
        if num_processes == "max" or num_processes > cpu_count:
            processes = cpu_count
        else:
            processes = num_processes

        if processes > len(iterable):
            processes = len(iterable)

        # If only one process is requested, execute the function sequentially
        if processes == 1:
            if timeout is not None:
                results = [
                    exec_with_timeout(func, timeout, kwargs, i) for i in iterable
                ]
            else:
                results = [func(i, **kwargs) for i in iterable]
            return results

        # Create a multiprocessing Pool
        pool = multiprocessing.Pool(processes=processes)

        try:
            # Use the pool to map the function across the iterable
            if timeout is not None:
                # Use partial to bind func, timeout, and kwargs.
                # The iterable item is passed as the last argument by pool.map
                worker = partial(exec_with_timeout, func, timeout, kwargs)
                results = pool.map(func=worker, iterable=iterable)
            else:
                results = pool.map(func=partial(func, **kwargs), iterable=iterable)
        except Exception:
            pool.terminate()
            raise
        else:
            # Close the pool to free resources
            pool.close()
        finally:
            pool.join()

        return results

    return parallelized


__all__ = ["exec_with_timeout", "parallelize"]
