from __future__ import annotations
import os, json, time, tracemalloc
from pathlib import Path
import numpy as np
import psutil

def process_memory_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def measure_callable(fn, repeats=1):
    times = []
    outputs = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        outputs = fn()
        times.append(time.perf_counter() - t0)
    return outputs, {
        "mean_ms": 1000 * float(np.mean(times)),
        "p50_ms": 1000 * float(np.percentile(times, 50)),
        "p95_ms": 1000 * float(np.percentile(times, 95)),
    }

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")
