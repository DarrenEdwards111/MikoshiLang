#!/usr/bin/env python3
"""Benchmark machine-array operations (packed arrays vs element-by-element lists).

Run from a checkout:  python scripts/benchmark_packed.py [--quick]
Times are the best of a few runs, in seconds, on the machine you run it on. Operations that a checkout does
not support (ToPackedArray before interop phase 2) are reported as "n/a".
"""
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mikoshilang import Session  # noqa: E402


def best(f, repeat):
    out = None
    for _ in range(repeat):
        t = time.perf_counter()
        f()
        dt = time.perf_counter() - t
        out = dt if out is None else min(out, dt)
    return out


def main():
    quick = "--quick" in sys.argv
    s = Session()
    s.evaluate("SeedRandom[1]; a60 = RandomReal[1, {60, 60}]; a200 = RandomReal[1, {200, 200}]; "
               "v = RandomReal[1, 100000]; w = RandomReal[1, 100000];")
    try:
        s.evaluate("p200 = ToPackedArray[a200]; pv = ToPackedArray[v]; pw = ToPackedArray[w];")
        packed = bool(s.evaluate("PackedArrayQ[p200]"))
    except Exception:          # older checkouts: no ToPackedArray
        packed = False
    A = np.random.default_rng(1).random((200, 200))
    ml = s.ns
    cases = [
        ("Dot 60x60 machine reals (language)", lambda: s.evaluate("a60 . a60"), 3),
        ("Dot 200x200 machine reals (language)", lambda: s.evaluate("a200 . a200"), 1 if quick else 3),
        ("Total of 10^5 reals (language)", lambda: s.evaluate("Total[v]"), 3),
        ("Mean of 10^5 reals (language)", lambda: s.evaluate("Mean[v]"), 3),
        ("Dot 200x200 from NumPy (Session.ns)", lambda: ml.Dot(A, A), 1 if quick else 3),
        ("Dot 200x200 packed", (lambda: s.evaluate("p200 . p200")) if packed else None, 3),
        ("v + w, 10^5 packed", (lambda: s.evaluate("pv + pw")) if packed else None, 3),
        ("Sin[v], 10^5 packed", (lambda: s.evaluate("Sin[pv]")) if packed else None, 3),
        ("v + w, 10^5 unpacked (element-wise, unchanged)", lambda: s.evaluate("v + w"), 1),
        ("NumPy reference: A @ A, 200x200", lambda: A @ A, 5),
    ]
    width = max(len(n) for n, _f, _r in cases)
    for name, f, repeat in cases:
        t = "n/a" if f is None else f"{best(f, repeat):.4f}"
        print(f"{name:<{width}}  {t}")


if __name__ == "__main__":
    main()
