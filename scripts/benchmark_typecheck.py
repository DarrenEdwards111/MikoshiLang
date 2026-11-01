#!/usr/bin/env python3
"""Overhead of the opt-in type checker.

Run from a checkout:  python scripts/benchmark_typecheck.py [--repeat N]
Prints the best-of-N time (seconds) of a dispatch-heavy workload with checking off, and with
Session(check_types=True) when the checkout has it. Compare "off" against a checkout without the checker
to measure the cost of the gate.
"""
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mikoshilang import Session  # noqa: E402

WORKLOAD = [
    # library heads (wl_* modules), many small calls
    "Total[Table[BitLength[k] + BitAnd[k, 7] + PolygonalNumber[3, k], {k, 1, 1500}]]",
    "Length[Table[StringRiffle[{\"a\", ToString[k]}, \"-\"], {k, 1, 800}]]",
    "Total[Table[AlternatingFactorial[Mod[k, 12]], {k, 1, 1500}]]",
    # user definitions (DownValues), deep recursion
    "fibB[0] = 0; fibB[1] = 1; fibB[n_] := fibB[n - 1] + fibB[n - 2]; fibB[17]",
    # kernel arithmetic and lists
    "Total[Map[#^2 + 1 &, Range[4000]]]",
]


def run(session):
    for src in WORKLOAD:
        session.evaluate(src)


def best(make, repeat):
    out = None
    for _ in range(repeat):
        s = make()
        t = time.perf_counter()
        run(s)
        dt = time.perf_counter() - t
        out = dt if out is None else min(out, dt)
    return out


def main():
    repeat = 7
    if "--repeat" in sys.argv:
        repeat = int(sys.argv[sys.argv.index("--repeat") + 1])
    run(Session())                               # warm up imports and caches
    print(f"checking off                 {best(Session, repeat):.4f}")
    try:
        Session(check_types=True)
    except TypeError:
        print("checking on                  n/a (this checkout has no type checker)")
        return
    print(f"checking on (check_types)    {best(lambda: Session(check_types=True), repeat):.4f}")


if __name__ == "__main__":
    main()
