from __future__ import annotations

import os
import sys
from typing import NamedTuple


DEFAULT_MAX_CPU_THREADS = 4
MAX_CPU_THREADS_ENV = "EM3DFOLD_MAX_CPU_THREADS"

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


class CpuResourceConfig(NamedTuple):
    max_threads: int
    affinity_before: tuple[int, ...]
    affinity_after: tuple[int, ...]


def _read_max_cpu_threads() -> int:
    raw_value = os.environ.get(
        MAX_CPU_THREADS_ENV,
        str(DEFAULT_MAX_CPU_THREADS),
    )
    try:
        max_threads = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{MAX_CPU_THREADS_ENV} must be a positive integer, got {raw_value!r}"
        ) from exc
    if max_threads < 1:
        raise ValueError(
            f"{MAX_CPU_THREADS_ENV} must be a positive integer, got {raw_value!r}"
        )
    return min(max_threads, DEFAULT_MAX_CPU_THREADS)


def _cap_thread_environment(max_threads: int) -> None:
    for variable_name in _THREAD_ENV_VARS:
        current_value = os.environ.get(variable_name)
        if current_value is not None:
            try:
                thread_count = min(int(current_value), max_threads)
            except ValueError:
                thread_count = max_threads
        else:
            thread_count = max_threads
        os.environ[variable_name] = str(max(1, thread_count))


def configure_cpu_resources() -> CpuResourceConfig:
    """Limit EM3DFold to a bounded number of concurrently active CPU threads.

    Thread-pool environment variables are set before numerical libraries are
    imported. On Linux, CPU affinity supplies the hard upper bound even when a
    third-party library creates additional worker threads. Existing scheduler
    or taskset restrictions are respected by selecting CPUs only from the
    process's current affinity mask.
    """

    max_threads = _read_max_cpu_threads()
    _cap_thread_environment(max_threads)

    affinity_before: tuple[int, ...] = ()
    affinity_after: tuple[int, ...] = ()
    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        affinity_before = tuple(sorted(os.sched_getaffinity(0)))
        affinity_after = affinity_before[:max_threads]
        if affinity_after and len(affinity_after) < len(affinity_before):
            os.sched_setaffinity(0, affinity_after)
        else:
            affinity_after = affinity_before

    return CpuResourceConfig(
        max_threads=max_threads,
        affinity_before=affinity_before,
        affinity_after=affinity_after,
    )


def configure_loaded_thread_pools(max_threads: int) -> None:
    """Apply the same bound to libraries imported by the selected command."""

    torch = sys.modules.get("torch")
    if torch is not None:
        torch.set_num_threads(max_threads)
        try:
            torch.set_num_interop_threads(max_threads)
        except RuntimeError:
            # PyTorch permits setting this only before inter-op work starts.
            # CPU affinity still enforces the hard process-wide bound.
            pass

    numba = sys.modules.get("numba")
    if numba is not None:
        try:
            numba.set_num_threads(min(max_threads, numba.get_num_threads()))
        except ValueError:
            pass
