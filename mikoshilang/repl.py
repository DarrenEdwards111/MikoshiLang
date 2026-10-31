"""Interactive REPL for MikoshiLang — now with Wolfram-style syntax parser."""

from __future__ import annotations
import readline
import sys
from .parser import parse, ParseError
from .runtime import Session
from . import __version__
from .expr import Expr, Symbol


_HELP = {
    "Sin": "Sin[x] — sine of x",
    "Cos": "Cos[x] — cosine of x",
    "Tan": "Tan[x] — tangent of x",
    "Exp": "Exp[x] — exponential e^x",
    "Log": "Log[x] — natural logarithm",
    "Diff": "Diff[expr, x] — differentiate",
    "Integrate": "Integrate[expr, x] — integrate",
    "Solve": "Solve[expr == 0, x] — solve equation",
    "Simplify": "Simplify[expr] — algebraic simplification",
    "Expand": "Expand[expr] — expand products",
    "Factor": "Factor[expr] — factorize",
    "N": "N[expr] or N[expr, precision] — numerical evaluation",
    "Plot": "Plot[expr, {x, min, max}] — 2D function plot",
    "Matrix": "{{a, b}, {c, d}} — create matrix",
    "List": "{a, b, c} — ordered collection",
    "Range": "Range[n] or Range[a, b] — generate range",
    "Table": "Table[expr, {i, min, max}] — generate list",
    "Quantity": "Quantity[value, \"unit\"] — physical quantity",
    "Element": "Element[\"H\"] — element data",
    "DFT": "DFT[{1, 2, 3, 4}] — discrete Fourier transform",
}

_ALL_NAMES = list(_HELP.keys()) + [
    "Pi", "E", "I", "Infinity", "True", "False",
    "ArcSin", "ArcCos", "ArcTan", "Log2", "Log10",
    "Det", "Inverse", "Transpose", "Eigenvalues", "Dot",
    "Map", "Select", "Sort", "Reverse", "Take", "Drop", "Part",
    "Total", "Mean", "Median", "StandardDeviation",
    "Prime", "PrimeQ", "GCD", "LCM", "Mod",
    "Factorial", "Binomial", "Fibonacci",
    "UnitConvert", "MolecularMass", "BalanceEquation",
    "FourierTransform", "Convolve", "HammingWindow",
]


def _completer(text, state):
    matches = [n for n in _ALL_NAMES if n.startswith(text)]
    return matches[state] if state < len(matches) else None


def main():
    """Run the MikoshiLang interactive REPL."""
    print(f"MikoshiLang {__version__} — Symbolic Computation Language")
    session = Session()
    print("Built by Mikoshi Ltd")
    print("Type Wolfram-style expressions. \"quit\" to exit, \"?Name\" for help.\n")

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")

    history_out = {}
    counter = 1

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

        try:
            # the Wolfram main loop: In[n], Out[n] (%), $Line, $Pre, $Post, $PrePrint
            for n, result in session.evaluate_input(line):
                history_out[n] = result
                if result is not None:  # Null output is not printed, as in Wolfram
                    from .library.wl_forms import display_text   # display wrappers (NumberForm, Grid, ...) as text
                    result = session.output_form(result)
                    shown = result if isinstance(result, str) else display_text(result)
                    print(f"Out[{n}]= {shown}")
        except ParseError as ex:
            print(f"  Parse error: {ex}")
        except Exception as ex:
            print(f"  Error: {ex}")

        counter = session.state.own.get("$Line", counter + 1)
        print()
    try:
        session.end()
    except Exception as ex:
        print(f"  Error in $Epilog: {ex}")


if __name__ == "__main__":
    main()
