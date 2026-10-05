# Copyright (c) 2026 Gil Rodrigues
"""RSS delta of loading libexl3_accept.so (fresh process): exl3_accept_rss.py LIB."""

import ctypes
import json
import sys
from pathlib import Path

STATUS_KEYS = frozenset({"VmRSS", "RssAnon", "RssFile", "VmSize"})


def status() -> dict[str, int]:
    """Read this process's memory counters.

    Returns:
        The selected /proc/self/status fields, in KiB.

    """
    out: dict[str, int] = {}
    with Path("/proc/self/status").open(encoding="utf-8") as lines:
        for line in lines:
            key, _, rest = line.partition(":")
            if key in STATUS_KEYS:
                out[key] = int(rest.split()[0])
    return out


lib = sys.argv[1]
cells = (ctypes.c_int64 * 23)(7, 4096, 0, 1, 248046, 0, 0, 0, *range(8), *range(7))
# Warm the ctypes load/call machinery on an already-mapped library so the
# delta below is the library itself, not first-touch of ctypes code paths.
for kind in (ctypes.CDLL, ctypes.PyDLL):
    warm = kind("libc.so.6", mode=ctypes.RTLD_LOCAL).labs
    warm.argtypes = [ctypes.POINTER(ctypes.c_int64)]
    warm.restype = ctypes.c_int32
    warm(cells)
before = status()
handle = ctypes.CDLL(lib, mode=ctypes.RTLD_LOCAL)
fn = handle.elpis_exl3_accept
fn.argtypes = [ctypes.POINTER(ctypes.c_int64)]
fn.restype = ctypes.c_int32
loaded = status()
result = fn(cells)
called = status()
sys.stdout.write(
    json.dumps({
        "result": result,
        "load_kib": {k: loaded[k] - before[k] for k in before},
        "first_call_kib": {k: called[k] - loaded[k] for k in before},
    })
    + "\n"
)
