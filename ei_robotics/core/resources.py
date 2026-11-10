# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""What the job's container may use. Studio jobs run in CPU- and memory-limited containers, where
os.cpu_count() reports the whole host."""

import os


def available_cpus():
    """CPUs this container may use: the affinity mask, capped by the cgroup v2 quota."""
    n = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    try:  # e.g. "400000 100000" = 4 CPUs
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            n = min(n, max(int(int(quota) / int(period)), 1))
    except (OSError, ValueError):
        pass
    return max(n, 1)


def available_memory():
    """Bytes this container may use: the cgroup v2 limit if set, else MemAvailable; None if unknown."""
    try:
        limit = open("/sys/fs/cgroup/memory.max").read().strip()
        if limit != "max":
            return int(limit)
    except (OSError, ValueError):
        pass
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def default_workers(worker_bytes, main_bytes):
    """One worker process per CPU, but no more than fit in memory: too many and the kernel kills workers.

    worker_bytes: real memory per worker; main_bytes: what to leave for the main process.
    """
    n = available_cpus()
    mem = available_memory()
    if mem is not None:
        fit = max(int((mem - main_bytes) // worker_bytes), 1)
        if fit < n:
            print(f"{n} CPUs, but {mem / 1e9:.1f} GB of memory fits {fit} workers")
            n = fit
    return n
