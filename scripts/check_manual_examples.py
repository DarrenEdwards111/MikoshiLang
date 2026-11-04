#!/usr/bin/env python
"""Check the manual's authored examples; optionally regenerate its TeX chapter.

Run from any directory with the project installed (or from this checkout):
    python scripts/check_manual_examples.py
    python scripts/check_manual_examples.py --write

Only core dependencies are used. Each example runs in a fresh, timeout-limited
Python process. Expected stdout is authored, not captured and blessed at runtime.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
CHAPTER = ROOT / "manual" / "current-examples.tex"

# (title, explanatory LaTeX, executable Python, exact expected stdout)
EXAMPLES = [
    (
        "Session: Python data and exact arithmetic",
        r"Use \texttt{Session.eval(source)} for Python data. Exact rationals become "
        r"\texttt{fractions.Fraction}; an inexact zero remains a Python float. "
        r"\texttt{Session.evaluate(source)} instead returns the raw language result.",
        '''
        from fractions import Fraction
        from mikoshilang import Session

        s = Session()
        assert s.eval("1 + 2") == 3
        q = s.eval("1/3")
        assert type(q) is Fraction and q == Fraction(1, 3)
        z = s.eval("0.5 - 0.5")
        assert type(z) is float and z == 0.0
        print(s.eval("1 + 2"))
        print(repr(q))
        print(repr(z))
        ''',
        "3\nFraction(1, 3)\n0.0\n",
    ),
    (
        "Session: definitions, patterns, and isolation",
        r"Definitions and pattern dispatch belong to one session. The integer "
        r"pattern is more specific than the general pattern, and the literal "
        r"definition is more specific still.",
        '''
        from mikoshilang import Session

        s = Session()
        s.eval('f[x_] := "general"; f[n_Integer] := n^2; f[0] = 7')
        values = s.eval(' {f[0], f[3], f["x"]} ')
        assert values == [7, 9, "general"]
        assert str(Session().eval("f[3]")) == "f[3]"
        print(values)
        print(Session().eval("f[3]"))
        ''',
        "[7, 9, 'general']\nf[3]\n",
    ),
    (
        "Session: solution rules, ODEs, and series",
        r"The language heads \texttt{Solve} and \texttt{DSolve} return nested "
        r"rule lists, not bare roots or Python dictionaries. Here "
        r"\texttt{evaluate} preserves the raw language braces; \texttt{eval} "
        r"converts the list containers to Python lists while leaving rules "
        r"as expressions. Other symbolic results remain expressions by default. "
        r"Series includes the requested order; "
        r"\texttt{Normal} removes the remainder term.",
        '''
        from mikoshilang import Session

        s = Session()
        print(s.evaluate("Solve[{x + y == 3, x - y == 1}, {x, y}]"))
        print(s.evaluate("DSolve[{y'[x] == y[x], y[0] == 1}, y[x], x]"))
        print(s.eval("Series[Exp[x], {x, 0, 3}]"))
        print(s.eval("Normal[Series[Exp[x], {x, 0, 3}]]"))
        assert s.eval("{x, y} /. First[Solve[{x+y==3, x-y==1}, {x,y}]]") == [2, 1]
        assert s.eval("Normal[Series[Exp[x], {x,0,3}]] /. x -> 0") == 1
        ''',
        "{{x -> 2, y -> 1}}\n{{y[x] -> E^x}}\n"
        "1 + x + x^2/2 + x^3/6 + O[x]^4\n1 + x + x^2/2 + x^3/6\n",
    ),
    (
        "Session: verification status and strict type predicates",
        r"Only the outer status is displayed here, not the full evidence. "
        r"\texttt{Certified} means an exact certificate was checked; "
        r"\texttt{Checked} means symbolic or numerical evidence, not a proof; "
        r"\texttt{Unchecked} means no applicable or conclusive checker. A failed "
        r"check returns \texttt{Failed}; computation errors still propagate. "
        r"\texttt{TypeMatchQ} is a strict predicate: a symbolic value is not "
        r"known to be an integer. This differs from opt-in call-boundary type "
        r"checking, which permits symbolic arguments.",
        '''
        from mikoshilang import Session

        s = Session()
        print(s.evaluate("Head[Verify[GCD[12, 18]]]"))
        print(s.evaluate("Head[Verify[Integrate[x^2, x]]]"))
        print(s.evaluate("Head[Verify[Sort[{3, 1, 2}]]]"))
        matches = s.eval("{TypeMatchQ[{1,2}, ListOf[Integer]], TypeMatchQ[x, Integer]}")
        assert matches == [True, False]
        print(matches)
        ''',
        "Certified\nChecked\nUnchecked\n[True, False]\n",
    ),
    (
        "Session: expose a typed Python function locally",
        r"The host deliberately exposes a function with \texttt{@s.function}; "
        r"this needs no Python-import permission and does not register a global "
        r"library head. Type annotations constrain concrete inputs. The error "
        r"example prints only the checked exception class.",
        '''
        from mikoshilang import Session

        s = Session()

        @s.function(listable=True, check="exact")
        def square_plus_one(n: int) -> int:
            return n * n + 1

        values = s.eval("SquarePlusOne[{1, 2, 3}]")
        assert values == [2, 5, 10]
        assert str(Session().eval("SquarePlusOne[2]")) == "SquarePlusOne[2]"
        print(values)
        try:
            s.eval("SquarePlusOne[2.5]")
        except TypeError as exc:
            print(type(exc).__name__)
        else:
            raise AssertionError("expected a TypeError")
        ''',
        "[2, 5, 10]\nTypeError\n",
    ),
    (
        "Session: explicit permission for Python calls",
        r"Python imports/calls are disabled by default. An allow-list enables "
        r"the named modules and their submodules; it is not a security sandbox. "
        r"Do not execute untrusted input without an external process sandbox. "
        r"This offline example uses only the standard library.",
        '''
        from mikoshilang import Session

        call = 'PythonFunction["math.gcd"][12, 18]'
        try:
            Session().eval(call)
        except PermissionError as exc:
            print(type(exc).__name__)
        else:
            raise AssertionError("Python calls must be gated")
        s = Session(allow_python={"math"})
        assert s.eval(call) == 6
        print(s.eval(call))
        ''',
        "PermissionError\n6\n",
    ),
    (
        "Direct Python: List, Take, Drop, and Part",
        r"These imported helpers are not \texttt{Session} language calls. "
        r"\texttt{Take(lst, n)}, \texttt{Drop(lst, n)}, and "
        r"\texttt{Part(lst, *indices)} accept a \texttt{List} expression, not "
        r"an ordinary Python list. Negative counts select/drop from the end; "
        r"zero counts select nothing/drop nothing. Part indices are one-based, "
        r"negative indices count from the end, and index zero returns the head.",
        '''
        from mikoshilang import List, Take, Drop, Part

        xs = List(10, 20, 30, 40)
        assert Take(xs, -2).args == (30, 40)
        assert Drop(xs, -2).args == (10, 20)
        assert Take(xs, 0).args == ()
        assert Drop(xs, 0) == xs
        assert Part(xs, -1) == 40
        print(Take(xs, -2))
        print(Drop(xs, -2))
        print(Take(xs, 0))
        print(Drop(xs, 0))
        print(Part(xs, 0))
        print(Part(xs, -1))
        try:
            Take([10, 20], 1)
        except TypeError as exc:
            print(type(exc).__name__)
        else:
            raise AssertionError("expected a List expression requirement")
        ''',
        "{30, 40}\n{10, 20}\n{}\n{10, 20, 30, 40}\nList\n40\nTypeError\n",
    ),
    (
        "Direct Python: finite Range and callable Table",
        r"\texttt{Range(n)}, \texttt{Range(a, b)}, and "
        r"\texttt{Range(a, b, step)} use integer bounds with an inclusive end. "
        r"Direct \texttt{Table(fn, spec)} takes a Python callable and either "
        r"a count, \texttt{(min, max)}, or \texttt{(min, max, step)}; it does "
        r"not take a language expression and iterator symbol. Wrong-direction "
        r"steps return an empty list rather than looping. Table rejects zero, "
        r"nonfinite, and nonadvancing steps. The verifier runs this entire "
        r"example in a timeout-limited child process; the checks below include "
        r"wrong-direction and nonadvancing cases.",
        '''
        from mikoshilang import Range, Table

        assert Range(3, 1, -1).args == (3, 2, 1)
        assert Range(1, 3, -1).args == ()
        assert Table(lambda i: i*i, (3, 1, -1)).args == (9, 4, 1)
        assert Table(lambda i: i, (1, 3, -1)).args == ()
        print(Range(3, 1, -1))
        print(Range(1, 3, -1))
        print(Table(lambda i: i*i, (3, 1, -1)))
        print(Table(lambda i: i, (1, 3, -1)))
        for spec in [(1, 3, 0), (1, 3, float("inf")),
                     (1e20, 1e20 + 1e6, 1.0)]:
            try:
                Table(lambda i: i, spec)
            except ValueError as exc:
                print(type(exc).__name__)
            else:
                raise AssertionError("expected invalid-step rejection")
        ''',
        "{3, 2, 1}\n{}\n{9, 4, 1}\n{}\nValueError\nValueError\nValueError\n",
    ),
    (
        "Direct Python: tiny determinants and complex eigenvalues",
        r"Direct linear-algebra helpers use \texttt{Matrix(*rows)}, whereas "
        r"Session language calls use nested lists. Integer determinant input "
        r"is exact; floating input is numerical and a tiny nonzero determinant "
        r"must not be rounded to zero. Complex eigenvalues keep their imaginary "
        r"parts. Formatting below normalizes insignificant numerical variation; "
        r"the assertions check values, not just display or shape.",
        '''
        from math import isclose
        from mikoshilang import Matrix, Det, Eigenvalues

        assert Det(Matrix([2, 0], [0, 3])) == 6
        tiny = Det(Matrix([1e-12, 0.0], [0.0, 1.0]))
        assert tiny != 0 and isclose(tiny, 1e-12, rel_tol=1e-12, abs_tol=0)
        values = Eigenvalues(Matrix([1j, 0], [0, -1j])).args
        assert len(values) == 2 and set(values) == {-1j, 1j}
        print(f"{tiny:.0e}")
        print([complex(v) == -1j or complex(v) == 1j for v in values])
        print([v.imag for v in values])
        ''',
        "1e-12\n[True, True]\n[-1.0, 1.0]\n",
    ),
    (
        "Direct Python: units, balancing, and complex convolution",
        r"\texttt{UnitConvert(Quantity(value, unit), target)} returns a Python "
        r"Quantity object. Offset-temperature conversion is supported, but "
        r"offset-temperature arithmetic is rejected: convert to kelvin first. "
        r"\texttt{BalanceEquation(equation)} accepts an equation string; "
        r"\texttt{Convolve(a, b)} accepts List expressions and preserves "
        r"complex components. These examples require no SciPy extra.",
        '''
        from mikoshilang import (
            Quantity, UnitConvert, BalanceEquation, Convolve, List,
        )

        distance = UnitConvert(Quantity(1, "km"), "m")
        assert distance.value == 1000.0 and distance.unit_name == "m"
        temperature = UnitConvert(Quantity(0, "C"), "K")
        assert temperature.value == 273.15 and temperature.unit_name == "K"
        print(distance)
        print(temperature)
        try:
            Quantity(0, "C") + Quantity(1, "C")
        except ValueError as exc:
            print(type(exc).__name__)
        else:
            raise AssertionError("offset-temperature arithmetic must be rejected")
        balanced = BalanceEquation("H2 + O2 -> H2O")
        assert balanced == "2H2 + O2 -> 2H2O"
        print(balanced)
        result = Convolve(List(1j, 1), List(1, 1j))
        assert result.args == (1j, 0, 1j)
        print(result.args)
        ''',
        'Quantity[1000, "m"]\nQuantity[273.15, "K"]\nValueError\n'
        '2H2 + O2 -> 2H2O\n(1j, 0, 1j)\n',
    ),
]


def source_code(source):
    return textwrap.dedent(source).strip() + "\n"


def render_chapter():
    parts = [
        "% Generated by scripts/check_manual_examples.py --write; edit that source.\n",
        "\\section{Verified runnable examples}\n",
        r"\label{chap:current-examples}" + "\n",
        "These examples are checked against the current checkout by "
        "\\texttt{scripts/check\\_manual\\_examples.py}. "
        "Run it with Python and the core package dependencies installed; "
        "no network, credentials, or optional scientific extras are needed. "
        "Each listing is a standalone Python program. The output listings are "
        "exact stdout checked by the script, including the explicit formatting "
        "of numerical results. Exception examples deliberately display only "
        "the exception class, not unstable full error messages.\n\n",
        "\\begin{lstlisting}[style=mikoshi]\n"
        "python scripts/check_manual_examples.py\n"
        "python scripts/check_manual_examples.py --write\n"
        "\\end{lstlisting}\n",
        "Without \\texttt{--write}, the script also checks that this generated "
        "chapter has not drifted. With \\texttt{--write}, it regenerates the "
        "chapter only after all examples pass. This is a bounded documentation "
        "check, not a claim to verify the whole library.\n\n",
    ]
    for title, explanation, source, expected in EXAMPLES:
        parts.extend([
            "\\subsection{" + title + "}\n",
            explanation + "\n\n",
            "\\begin{lstlisting}[style=mikoshi]\n", source_code(source),
            "\\end{lstlisting}\n",
            "\\noindent Expected standard output:\n",
            "\\begin{lstlisting}[style=mikoshi]\n", expected,
            "\\end{lstlisting}\n\n",
        ])
    return "".join(parts).rstrip() + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the checked TeX chapter")
    args = parser.parse_args()
    # A fresh process both isolates decorator/definition state and bounds potential
    # iterator regressions. -u makes any failure diagnostics available immediately.
    for index, (title, _, source, expected) in enumerate(EXAMPLES, 1):
        try:
            run = subprocess.run(
                [sys.executable, "-u", "-c", source_code(source)],
                cwd=str(ROOT), capture_output=True, text=True, timeout=45,
            )
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(f"FAIL {index}: {title}: exceeded 45 seconds") from exc
        if run.returncode or run.stdout != expected or run.stderr:
            raise SystemExit(
                f"FAIL {index}: {title}\nreturn code: {run.returncode}\n"
                f"expected stdout: {expected!r}\nactual stdout: {run.stdout!r}\n"
                f"stderr: {run.stderr}"
            )
        print(f"PASS {index}: {title}")
    chapter = render_chapter()
    if args.write:
        with CHAPTER.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(chapter)
    elif not CHAPTER.exists() or CHAPTER.read_text(encoding="utf-8") != chapter:
        raise SystemExit("FAIL: chapter missing or stale; run with --write")
    verb = "written" if args.write else "matches"
    print(f"PASS: {len(EXAMPLES)} example programs; exact stdout and assertions checked; chapter {verb}.")


if __name__ == "__main__":
    main()
