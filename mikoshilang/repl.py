"""Interactive REPL for MikoshiLang."""

from __future__ import annotations
import readline
import sys
from .expr import Expr, Symbol, symbols
from .evaluate import evaluate
from .builtins import BUILTIN_FUNCTIONS
from .math_ops import simplify, expand, factor, diff, integrate, solve, limit, series
from .numerical import N, Pi, E, I, Infinity, GoldenRatio
from .data import List, Table, Map, Select, Sort, Reverse, Take, Drop, Part, Range, Total, Mean, Median, StandardDeviation
from .linalg import Matrix, Dot, Det, Inverse, Eigenvalues, Eigenvectors, Transpose

_HELP = {
    "Sin": "Sin[x] — sine of x",
    "Cos": "Cos[x] — cosine of x",
    "Tan": "Tan[x] — tangent of x",
    "Exp": "Exp[x] — exponential e^x",
    "Log": "Log[x] — natural logarithm",
    "diff": "diff(expr, var) — differentiate",
    "integrate": "integrate(expr, var) — integrate",
    "solve": "solve(expr, var) — solve equation",
    "simplify": "simplify(expr) — algebraic simplification",
    "expand": "expand(expr) — expand products",
    "factor": "factor(expr) — factorize",
    "N": "N(expr) or N(expr, precision) — numerical evaluation",
    "Matrix": "Matrix([a,b],[c,d]) — create matrix",
    "List": "List(a, b, c) — ordered collection",
    "Range": "Range(n) or Range(a,b) — generate range",
    "Table": "Table(fn, (min, max)) — generate list from function",
}

_ALL_NAMES = list(BUILTIN_FUNCTIONS.keys()) + [
    "simplify", "expand", "factor", "diff", "integrate", "solve", "limit", "series",
    "N", "Pi", "E", "I", "Infinity", "GoldenRatio",
    "List", "Table", "Map", "Select", "Sort", "Reverse", "Take", "Drop", "Part", "Range",
    "Total", "Mean", "Median", "StandardDeviation",
    "Matrix", "Dot", "Det", "Inverse", "Eigenvalues", "Eigenvectors", "Transpose",
    "symbols", "evaluate",
]


def _completer(text, state):
    matches = [n for n in _ALL_NAMES if n.startswith(text)]
    return matches[state] if state < len(matches) else None


def main():
    """Run the MikoshiLang interactive REPL."""
    print("MikoshiLang v0.1.0 — Symbolic Computation Language")
    print("Built by Mikoshi Ltd")
    print('Type "quit" to exit, "?Name" for help.\n')

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")

    history_in = {}
    history_out = {}
    counter = 1

    # Build evaluation namespace
    ns = {}
    ns.update(BUILTIN_FUNCTIONS)
    ns.update({
        "simplify": simplify, "expand": expand, "factor": factor,
        "diff": diff, "integrate": integrate, "solve": solve,
        "limit": limit, "series": series,
        "N": N, "Pi": Pi, "E": E, "I": I,
        "Infinity": Infinity, "GoldenRatio": GoldenRatio,
        "List": List, "Table": Table, "Map": Map, "Select": Select,
        "Sort": Sort, "Reverse": Reverse, "Take": Take, "Drop": Drop,
        "Part": Part, "Range": Range, "Total": Total, "Mean": Mean,
        "Median": Median, "StandardDeviation": StandardDeviation,
        "Matrix": Matrix, "Dot": Dot, "Det": Det, "Inverse": Inverse,
        "Eigenvalues": Eigenvalues, "Eigenvectors": Eigenvectors,
        "Transpose": Transpose,
        "symbols": symbols, "Symbol": Symbol, "Expr": Expr,
        "evaluate": evaluate,
    })

    while True:
        try:
            line = input(f"In[{counter}]:= ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        line = line.strip()
        if not line:
            continue
        if line.lower() in ("quit", "exit"):
            print("Goodbye!")
            break

        if line.startswith("?"):
            name = line[1:].strip()
            if name in _HELP:
                print(f"  {_HELP[name]}")
            else:
                print(f"  No help available for {name}")
            continue

        history_in[counter] = line

        try:
            result = eval(line, {"__builtins__": {}}, ns)
            history_out[counter] = result
            ns["In"] = history_in
            ns["Out"] = history_out
            ns[f"Out{counter}"] = result
            print(f"Out[{counter}]= {result}")
        except Exception as ex:
            print(f"  Error: {ex}")

        counter += 1
        print()


if __name__ == "__main__":
    main()
