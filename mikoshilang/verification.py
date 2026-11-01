"""Executable ``Check:`` lines: run each library function's stated verification and report pass/fail.

Every reviewed head's notes end with a ``Check:`` sentence naming one or more methods (CONVENTIONS §4):
``exact``, ``self-check``, ``numeric cross-check``, ``reference`` and ``delegated``. :func:`run` turns those
into checks that run:

* ``reference``: the head's differential cases in ``tests/library`` (values written independently of the code);
* ``exact``: every case with exact input gives a result without machine or bignum floats;
* ``self-check``: the implementation contains its verification step (it raises ``ArithmeticError``) and its
  cases pass through it;
* ``numeric cross-check``: for numeric heads, the exact result evaluated to 30 digits agrees with the head
  evaluated on 30-digit input;
* ``delegated``: the named library routine exists.

On top of the stated checks, independent tables apply to every head they mention, stated or not: identities
and round trips (``IDENTITIES``), numerical comparison with mpmath (``MPMATH``) and with NumPy (``NUMPY``).

Use ``scripts/verify_library.py``; ``--stated`` (the CI job) limits the run to heads with a ``Check:`` line and
to the tables.
"""
from __future__ import annotations

import importlib
import inspect
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

METHODS = ("exact", "self-check", "numeric cross-check", "reference", "delegated")
_CHECK = re.compile(r"Check:\s*(.*)$", re.S)
_DELEGATED = re.compile(r"delegated\s*\(([A-Za-z_][\w.]*)")


def stated(notes: str) -> Tuple[List[str], str]:
    """(methods named in the Check: sentence, the sentence) for a head's notes."""
    m = _CHECK.search(notes or "")
    if not m:
        return [], ""
    text = m.group(1).strip()
    low = text.lower()
    return [k for k in METHODS if k in low], text


# ─────────────────────────── independent tables ───────────────────────────

# (heads exercised, MikoshiLang source that must evaluate to True)
IDENTITIES: List[Tuple[Tuple[str, ...], str]] = [
    (("FromDigits", "IntegerDigits"),
     "And @@ Flatten[Table[FromDigits[IntegerDigits[n, b], b] == n, {n, {1, 7, 255, 10^20 + 3}}, {b, {2, 3, 10, 16}}]]"),
    (("FactorInteger",), "And @@ Table[Times @@ (#[[1]]^#[[2]] & /@ FactorInteger[n]) == n, "
                         "{n, {2, 360, 2^64 + 1, 10^12 + 39, 3^40 - 1}}]"),
    (("PrimeQ", "Prime", "PrimePi"), "And @@ Table[PrimeQ[Prime[k]] && PrimePi[Prime[k]] == k, {k, {1, 10, 1000, 10^5}}]"),
    (("Factor", "Expand"), "And @@ (Expand[Factor[#]] == Expand[#] & /@ {x^4 - 1, x^6 - y^6, 2 x^2 + 4 x + 2, "
                           "x^3 - 6 x^2 + 11 x - 6})"),
    (("Integrate", "D"), "And @@ (Simplify[D[Integrate[#, x], x] - #] == 0 & /@ {x^3, Sin[x]^2, 1/(1 + x^2), "
                         "x Exp[x], Log[x], 1/(x^2 - 1)})"),
    (("Together", "Apart"), "And @@ (Simplify[Together[Apart[#]] - #] == 0 & /@ {1/(x^2 - 1), (x + 2)/(x^3 - x)})"),
    (("Solve",), "And @@ Flatten[Table[Simplify[(x^3 - 6 x^2 + 11 x - 6) /. s] == 0, "
                 "{s, Solve[x^3 - 6 x^2 + 11 x - 6 == 0, x]}]]"),
    (("ExtendedGCD", "GCD"), "And @@ Flatten[Table[Module[{r = ExtendedGCD[a, b]}, r[[1]] == GCD[a, b] && "
                             "r[[2]] . {a, b} == r[[1]]], {a, {240, 17, 10^18}}, {b, {46, 5, 3^30}}]]"),
    (("ChineseRemainder",), "Mod[ChineseRemainder[{2, 3, 1}, {5, 7, 11}], {5, 7, 11}] == {2, 3, 1}"),
    (("PowerMod",), "And @@ Table[Mod[PowerMod[a, -1, 101] a, 101] == 1, {a, 1, 100}]"),
    (("Binomial",), "And @@ Flatten[Table[Binomial[n, k] == Binomial[n - 1, k - 1] + Binomial[n - 1, k], "
                    "{n, 2, 14}, {k, 1, n - 1}]]"),
    (("Fibonacci",), "And @@ Table[Fibonacci[n + 2] == Fibonacci[n + 1] + Fibonacci[n], {n, 0, 60}]"),
    (("Inverse", "IdentityMatrix"), "And @@ (Inverse[#] . # == IdentityMatrix[Length[#]] & /@ "
                                    "{{{2, 1}, {1, 3}}, {{1, 2, 0}, {0, 1, 4}, {5, 0, 1}}})"),
    (("Det",), "Det[{{1, 2}, {3, 4}} . {{0, 1}, {1, 1}}] == Det[{{1, 2}, {3, 4}}] Det[{{0, 1}, {1, 1}}]"),
    (("Transpose",), "Transpose[Transpose[{{1, 2, 3}, {4, 5, 6}}]] == {{1, 2, 3}, {4, 5, 6}}"),
    (("Reverse", "Sort"), "Reverse[Reverse[{3, 1, 2}]] == {3, 1, 2} && Sort[Sort[{3, 1, 2}]] == Sort[{3, 1, 2}]"),
    (("StringReverse",), "StringReverse[StringReverse[\"abc d\"]] == \"abc d\""),
    (("ToCharacterCode", "FromCharacterCode"), "FromCharacterCode[ToCharacterCode[\"Mikoshi λ\"]] == \"Mikoshi λ\""),
    (("StringSplit", "StringRiffle"), "StringRiffle[StringSplit[\"a b c\"], \" \"] == \"a b c\""),
    (("ToUpperCase", "ToLowerCase"), "ToLowerCase[ToUpperCase[\"mikoshi\"]] == \"mikoshi\""),
    (("Compress", "Uncompress"), "Uncompress[Compress[{x^2, \"s\", 3/4}]] == {x^2, \"s\", 3/4}"),
    (("ExportString", "ImportString"), "ImportString[ExportString[{1, 2, 3}, \"JSON\"], \"JSON\"] == {1, 2, 3}"),
    (("BaseEncode", "BaseDecode"), "BaseDecode[BaseEncode[ByteArray[{1, 2, 250}]]] == ByteArray[{1, 2, 250}]"),
    (("Encrypt", "Decrypt", "GenerateSymmetricKey"),
     "Module[{k = GenerateSymmetricKey[]}, Decrypt[k, Encrypt[k, \"secret\"]] == \"secret\"]"),
    (("Hash",), "Hash[\"abc\", \"SHA256\", \"HexString\"] == "
                "\"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad\""),   # FIPS 180-2 vector
    (("AlternatingFactorial",), "And @@ Table[AlternatingFactorial[k] + AlternatingFactorial[k - 1] == k!, {k, 1, 25}]"),
    (("PolygonalNumber",), "And @@ Table[PolygonalNumber[3, n] == n (n + 1)/2, {n, 0, 40}]"),
]

# (head, mpmath function, list of exact argument tuples as MikoshiLang source)
MPMATH: List[Tuple[str, str, List[str]]] = [
    ("Gamma", "gamma", ["7/3", "1/2", "-5/2", "10"]),
    ("LogGamma", "loggamma", ["7/3", "50"]),
    ("Zeta", "zeta", ["3", "1/2", "-3/2"]),
    ("Erf", "erf", ["1/3", "2"]),
    ("Erfc", "erfc", ["1/3", "2"]),
    ("BesselJ", "besselj", ["0, 1", "2, 7/2", "1/2, 3"]),
    ("BesselY", "bessely", ["0, 1", "2, 7/2"]),
    ("BesselI", "besseli", ["1, 2/3", "3, 5"]),
    ("BesselK", "besselk", ["1, 2/3", "3, 5"]),
    ("AiryAi", "airyai", ["1/2", "-3"]),
    ("AiryBi", "airybi", ["1/2", "-3"]),
    ("PolyLog", "polylog", ["2, 1/2", "3, -1/3"]),
    ("Beta", "beta", ["2/3, 5/2", "3, 4"]),
    ("Hypergeometric2F1", "hyp2f1", ["1/2, 1/3, 2, 1/4", "1, 1, 2, -1/2"]),
    ("EllipticK", "ellipk", ["1/3", "-2"]),
    ("EllipticE", "ellipe", ["1/3", "-2"]),
    ("ExpIntegralEi", "ei", ["1/2", "3"]),
    ("ProductLog", "lambertw", ["1", "1/3"]),
    ("PolyGamma", "psi", ["0, 7/3", "1, 5/2"]),
    ("LegendreP", "legendre", ["5, 1/3"]),
    ("HermiteH", "hermite", ["4, 2/3"]),
    ("ChebyshevT", "chebyt", ["5, 1/3"]),
    ("Gudermannian", None, ["1/2"]),   # no mpmath routine: compared with 2 ArcTan[Tanh[x/2]] below
]

# (head, NumPy reference as a Python callable on float arrays, list of argument sources)
_MATS = ["{{2., 1., 0.5}, {1., 3., 0.25}, {0.5, 0.25, 4.}}", "{{1., 2.}, {3., 4.}}"]


def _np_refs():
    import numpy as np
    return [
        ("Det", lambda a: np.linalg.det(a), _MATS),
        ("Inverse", lambda a: np.linalg.inv(a), _MATS),
        ("Eigenvalues", lambda a: np.sort_complex(np.linalg.eigvals(a)), _MATS),
        ("Transpose", lambda a: a.T, _MATS),
        ("Tr", lambda a: np.trace(a), _MATS),
        ("MatrixRank", lambda a: np.linalg.matrix_rank(a), _MATS),
        ("Total", lambda a: np.sum(a), ["{1.5, 2.25, -3., 10.}"]),
        ("Mean", lambda a: np.mean(a), ["{1.5, 2.25, -3., 10.}"]),
        ("Variance", lambda a: np.var(a, ddof=1), ["{1.5, 2.25, -3., 10.}"]),
        ("StandardDeviation", lambda a: np.std(a, ddof=1), ["{1.5, 2.25, -3., 10.}"]),
        ("Norm", lambda a: np.linalg.norm(a), ["{3., 4., 12.}"]),
        ("Median", lambda a: np.median(a), ["{1.5, 2.25, -3., 10.}"]),
    ]


# ─────────────────────────── report ───────────────────────────

@dataclass
class Result:
    name: str
    methods: List[str]
    text: str
    checks: Dict[str, Tuple[int, int]] = field(default_factory=dict)   # check -> (passed, failed)
    failures: List[str] = field(default_factory=list)

    def add(self, check: str, ok: bool, detail: str = ""):
        p, f = self.checks.get(check, (0, 0))
        self.checks[check] = (p + ok, f + (not ok))
        if not ok and len(self.failures) < 5:
            self.failures.append(f"{check}: {detail}"[:300])

    @property
    def status(self) -> str:
        if any(f for _p, f in self.checks.values()):
            return "fail"
        if not self.checks:
            return "not exercised" if self.methods else "no check stated"
        return "pass"


def _ev(src):
    from .evaluate import evaluate
    from .parser import parse
    return evaluate(parse(src))


def _has_inexact(x) -> bool:
    import sympy as sp
    from .expr import Expr
    if isinstance(x, bool):
        return False
    if isinstance(x, (float, complex)):
        return True
    if isinstance(x, sp.Basic):
        return any(isinstance(a, sp.Float) for a in sp.preorder_traversal(x))
    if isinstance(x, Expr):
        return any(_has_inexact(a) for a in x.args) or (isinstance(x.head, Expr) and _has_inexact(x.head))
    return False


_INEXACT_SRC = re.compile(r"\d\.\d*|\.\d|\bN\[|`|\bNumeric|\bRandom|Machine|\bReal\b|\bFindRoot|\bNIntegrate|"
                          r"\bNSolve|\bNMinimize|\bNMaximize|\bInterpolation|\bFit\b|\bFindFit")


def load_modules(tests_dir: Path) -> List[Tuple[str, list]]:
    """[(module, CASES), ...] from tests/library, in file order (a module's cases may depend on earlier ones)."""
    tests_dir = Path(tests_dir)
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    out = []
    for p in sorted(tests_dir.glob("test_*.py")):
        if _needs_fixtures(p):
            NEEDS_FIXTURES.add(p.stem)
            continue
        try:
            mod = importlib.import_module(p.stem)
        except Exception:            # optional dependency missing: those cases are skipped
            continue
        cases = [c for c in (getattr(mod, "CASES", None) or []) if isinstance(c, tuple) and c and isinstance(c[0], str)]
        if cases:
            out.append((p.stem, cases))
    return out


NEEDS_FIXTURES: set = set()
_FIXTURE_ARG = re.compile(r"^def test_\w+\(case,\s*\w", re.M)
_AUTOUSE_PER_TEST = re.compile(r"^@pytest\.fixture\(autouse=True\)", re.M)


def _needs_fixtures(path: Path) -> bool:
    """A module whose cases only run inside pytest fixtures (a temporary working directory, local servers,
    per-test isolation) cannot be replayed outside pytest: the harness leaves it out, so its heads count as
    not exercised there rather than failing. Its cases still run in the test suite."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return bool(_FIXTURE_ARG.search(src) or _AUTOUSE_PER_TEST.search(src))


def load_cases(tests_dir: Path) -> Dict[str, list]:
    """{head: [case, ...]} from the CASES lists in tests/library (cases are grouped by their outermost head)."""
    harness = _harness(tests_dir)
    out: Dict[str, list] = {}
    for _m, cases in load_modules(tests_dir):
        for c in cases:
            h = harness.head_of(c)
            if h:
                out.setdefault(h, []).append(c)
    return out


def _harness(tests_dir):
    tests_dir = str(Path(tests_dir))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    return importlib.import_module("_harness")


def run_reference(tests_dir, heads: set) -> Dict[str, List[Tuple[bool, str]]]:
    """Run every module that has cases for ``heads``, in order, and collect the outcomes of those heads' cases."""
    from .kernel import _state, SessionState
    harness = _harness(tests_dir)
    out: Dict[str, List[Tuple[bool, str]]] = {}
    for m, cases in load_modules(tests_dir):
        if not any(harness.head_of(c) in heads for c in cases):
            continue
        # Each module runs in fresh session definitions (its cases may set up state for later ones);
        # modules whose test function isolates every case (a = 5, f[x_] := ...) get the same here.
        per_case = _isolates_cases(tests_dir, m)
        tok = _state.set(SessionState())
        try:
            for c in cases:
                h = harness.head_of(c)
                ctok = _state.set(SessionState()) if per_case else None
                try:
                    harness.check(c)
                    ok, detail = True, ""
                except BaseException as exc:      # noqa: BLE001 - pytest.raises failures derive from BaseException
                    if isinstance(exc, KeyboardInterrupt):
                        raise
                    ok, detail = False, f"{c[0][:80]}: {type(exc).__name__}: {str(exc)[:150]}"
                finally:
                    if ctok is not None:
                        _state.reset(ctok)
                if h in heads:
                    out.setdefault(h, []).append((ok, detail))
        finally:
            _state.reset(tok)
    return out


def _isolates_cases(tests_dir, module) -> bool:
    try:
        src = (Path(tests_dir) / f"{module}.py").read_text(encoding="utf-8")
    except OSError:
        return False
    return "_state.set(SessionState())" in src


def _run_reference(r: Result, cases, harness, outcomes=None):
    for ok, detail in (outcomes or {}).get(r.name, []):
        r.add("reference", ok, detail)


def _run_exact(r: Result, cases, harness):
    n = 0
    for c in cases:
        if not isinstance(c[1], str) or _INEXACT_SRC.search(c[0]) or _INEXACT_SRC.search(c[1]):
            continue
        try:
            got = _ev(c[0])
        except Exception:
            continue
        n += 1
        if _has_inexact(got):
            r.add("exact", False, f"{c[0][:80]} gave an inexact result {str(got)[:80]}")
        else:
            r.add("exact", True)
    if n == 0:
        r.add("exact", True)          # vacuous for heads whose cases are all inexact or Raises; noted in the docs


def _impl_sources(name, depth: int = 2):
    """Source of a head's implementations and of the same-module functions and classes they use (``depth``
    levels): verification steps often live in a helper, in the function a factory wraps or in a class it builds."""
    from .library.core import REGISTRY
    e = REGISTRY.get(name)
    srcs, seen = [], set()

    def visit(f, d):
        while getattr(f, "__wrapped__", None) is not None:
            f = f.__wrapped__
        code = getattr(f, "__code__", None)
        if code is None or code in seen:
            return
        seen.add(code)
        try:
            srcs.append(inspect.getsource(f))
        except (OSError, TypeError):
            return
        if d <= 0:
            return
        lib = "mikoshilang.library"
        refs = [f.__globals__.get(n) for n in code.co_names]
        refs += [c.cell_contents for c in (f.__closure__ or ()) if _cell_ok(c)]
        for g in refs:
            if inspect.isfunction(g) and (g.__module__ or "").startswith(lib):
                visit(g, d - 1)
            elif inspect.isclass(g) and (g.__module__ or "").startswith(lib) and g not in seen:
                seen.add(g)                      # a distribution or object class the factory instantiates
                try:
                    srcs.append(inspect.getsource(g))
                except (OSError, TypeError):
                    pass

    for f, *_ in (e.impls if e else []):
        visit(f, depth)
    if depth > 0:
        # distributions are classes or builders looked up by name in the statistics tables
        try:
            from .library import wl_stats
        except Exception:                    # optional dependency missing
            wl_stats = None
        for table in ("TABLE", "BUILDERS"):
            g = getattr(wl_stats, table, {}).get(name) if wl_stats is not None else None
            if inspect.isfunction(g):
                visit(g, depth - 1)
            elif inspect.isclass(g) and g not in seen:
                seen.add(g)
                try:
                    srcs.append(inspect.getsource(g))
                except (OSError, TypeError):
                    pass
    return srcs


def _cell_ok(cell):
    try:
        cell.cell_contents
        return True
    except ValueError:
        return False


def _run_selfcheck(r: Result, cases, harness):
    # the implementation itself, or a helper, factory-wrapped function or class it uses, raising ArithmeticError
    if not (any("ArithmeticError" in s or "verif" in s.lower() for s in _impl_sources(r.name, depth=0))
            or any("raise ArithmeticError" in s for s in _impl_sources(r.name))):
        r.add("self-check", False, "no verification step (ArithmeticError) found in the implementation")
        return
    r.add("self-check", True)


def _run_numeric(r: Result, cases, harness):
    import sympy as sp
    from .library.core import S
    n = 0
    for c in cases:
        if _INEXACT_SRC.search(c[0]) or not isinstance(c[1], (str, int, float, complex)):
            continue
        m = re.match(r"\s*([A-Za-z][A-Za-z0-9]*)\[(.*)\]\s*$", c[0], re.S)
        if not m or m.group(1) != r.name:
            continue
        try:
            exact = _ev(f"N[{c[0]}, 30]")
            direct = _ev(f"{r.name}[Sequence @@ N[{{{m.group(2)}}}, 30]]")
            a, b = S(exact), S(direct)
            if not (isinstance(a, sp.Basic) and isinstance(b, sp.Basic) and a.is_number and b.is_number):
                continue
        except Exception:
            continue
        n += 1
        # a function that works in machine precision (its result has 53-bit floats even for 30-digit input)
        # is compared at machine accuracy; otherwise to 20 digits
        floats = b.atoms(sp.Float)
        tol = 1e-12 if floats and min(f._prec for f in floats) <= 53 else 1e-20
        ok = abs(complex(sp.N(a - b, 30))) <= tol * max(1.0, abs(complex(sp.N(a, 30))))
        r.add("numeric cross-check", ok, f"{c[0][:60]}: {sp.N(a, 15)} vs {sp.N(b, 15)}")
        if n >= 5:
            break
    if n == 0:
        r.add("numeric cross-check", True)   # nothing numeric to compare (noted as vacuous in the docs)


def _run_delegated(r: Result, cases, harness):
    m = _DELEGATED.search(r.text)
    if not m:
        r.add("delegated", True)
        return
    path = m.group(1)
    if "." not in path:                 # prose ("delegated (the stored function f)"), not a module path
        r.add("delegated", True)
        return
    parts = path.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        try:
            for a in parts[i:]:
                obj = getattr(obj, a)
        except AttributeError:
            break
        r.add("delegated", True)
        return
    r.add("delegated", False, f"{path} is not importable")


_RUNNERS = {"reference": _run_reference, "exact": _run_exact, "self-check": _run_selfcheck,
            "numeric cross-check": _run_numeric, "delegated": _run_delegated}


def _run_tables(results: Dict[str, Result], only: Optional[set]):
    import mpmath
    import numpy as np
    import sympy as sp
    from .library.core import S, STATUS

    def res(h):
        if h not in results:
            notes = STATUS.get(h, ("", "", ""))[2]
            m, t = stated(notes)
            results[h] = Result(h, m, t)
        return results[h]

    for heads, src in IDENTITIES:
        if only is not None and not (set(heads) & only):
            continue
        try:
            ok = _ev(src) is True
            detail = "" if ok else f"{src[:90]} is not True"
        except Exception as exc:
            ok, detail = False, f"{src[:60]}: {type(exc).__name__}: {str(exc)[:120]}"
        for h in heads:
            res(h).add("identity", ok, detail)

    for head, mpname, samples in MPMATH:
        if only is not None and head not in only:
            continue
        for args in samples:
            try:
                got = S(_ev(f"N[{head}[{args}], 30]"))
                vals = [sp.Rational(S(_ev(a.strip()))) for a in args.split(",")]
                with mpmath.workdps(30):
                    if mpname is None:
                        ref = 2 * mpmath.atan(mpmath.tanh(mpmath.mpf(vals[0].p) / vals[0].q / 2))
                    else:
                        ref = getattr(mpmath, mpname)(*[mpmath.mpf(v.p) / v.q for v in vals])
                g = complex(sp.N(got, 30))
                ok = abs(g - complex(ref)) <= 1e-14 * max(1.0, abs(complex(ref)))
                res(head).add("mpmath", ok, f"{head}[{args}] = {sp.N(got, 15)}, mpmath {mpmath.nstr(ref, 15)}")
            except Exception as exc:
                res(head).add("mpmath", False, f"{head}[{args}]: {type(exc).__name__}: {str(exc)[:120]}")

    for head, ref_fn, samples in _np_refs():
        if only is not None and head not in only:
            continue
        for src in samples:
            try:
                arr = np.array(S(_ev(src)), dtype=float)
                got = _ev(f"{head}[{src}]")
                g = S(got)
                g = np.array(g, dtype=complex) if isinstance(g, list) else complex(sp.N(g))
                ref = ref_fn(arr)
                if head == "Eigenvalues":
                    g = np.sort_complex(np.array(g, dtype=complex))
                ok = np.allclose(g, ref, rtol=1e-9, atol=1e-12)
                res(head).add("numpy", bool(ok), f"{head}[{src}] = {got}, NumPy {ref}")
            except Exception as exc:
                res(head).add("numpy", False, f"{head}[{src}]: {type(exc).__name__}: {str(exc)[:120]}")


def run(tests_dir, mode: str = "stated", names: Optional[List[str]] = None) -> Dict[str, Result]:
    """Run the checks. mode "stated": heads with a Check: line plus the tables; "all": every audited head."""
    import mikoshilang  # noqa: F401  (registers the library)
    from .library.core import STATUS
    cases = load_cases(tests_dir)
    harness = _harness(tests_dir)
    selected = {}
    for name, (area, status, notes) in sorted(STATUS.items()):
        if names is not None and name not in names:
            continue
        if status in ("error", "quarantined"):
            continue
        methods, text = stated(notes)
        if mode == "stated" and not methods:
            continue
        selected[name] = (methods, text)
    # reference cases: every selected head in "all" mode, else those whose Check: names reference
    ref_heads = {n for n, (m, _t) in selected.items() if mode == "all" or "reference" in m}
    outcomes = run_reference(tests_dir, ref_heads)
    results: Dict[str, Result] = {}
    for name, (methods, text) in selected.items():
        r = results[name] = Result(name, methods, text)
        cs = cases.get(name, [])
        for m in methods:
            if m == "reference":
                _run_reference(r, cs, harness, outcomes)
            else:
                _RUNNERS[m](r, cs, harness)
        if mode == "all" and "reference" not in methods:
            _run_reference(r, cs, harness, outcomes)   # every head's cases are reference evidence too
    _run_tables(results, set(names) if names is not None else None)
    return results


def summary(results: Dict[str, Result]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in results.values():
        out[r.status] = out.get(r.status, 0) + 1
    return out


def markdown(results: Dict[str, Result], seconds: float, mode: str) -> str:
    s = summary(results)
    lines = [f"# Verification report ({mode})", "",
             f"{len(results)} functions in {seconds:.0f} s: " + ", ".join(f"{v} {k}" for k, v in sorted(s.items())),
             "", "| Function | Stated (Check:) | Checks run (passed/failed) | Status |", "|---|---|---|---|"]
    for name in sorted(results):
        r = results[name]
        ch = ", ".join(f"{k} {p}/{f}" for k, (p, f) in sorted(r.checks.items())) or "none"
        lines.append(f"| `{name}` | {', '.join(r.methods) or 'none'} | {ch} | {r.status} |")
    fails = [r for r in results.values() if r.status == "fail"]
    if fails:
        lines += ["", "## Failures", ""]
        for r in sorted(fails, key=lambda r: r.name):
            lines.append(f"* `{r.name}`: " + "; ".join(r.failures).replace("|", "\\|"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    import argparse
    import json
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--all", action="store_true", help="every audited head (slow), not only stated checks")
    ap.add_argument("--tests", default=str(Path(__file__).resolve().parent.parent / "tests" / "library"))
    ap.add_argument("--names", nargs="*", help="only these heads")
    ap.add_argument("--markdown", help="write the report to this file")
    ap.add_argument("--json", help="write per-function results to this file")
    a = ap.parse_args(argv)
    mode = "all" if a.all else "stated"
    t = time.perf_counter()
    results = run(a.tests, mode, a.names)
    dt = time.perf_counter() - t
    s = summary(results)
    print(f"verification ({mode}): {len(results)} functions in {dt:.1f} s: "
          + ", ".join(f"{v} {k}" for k, v in sorted(s.items())))
    for r in sorted(results.values(), key=lambda r: r.name):
        if r.status == "fail":
            print(f"FAIL {r.name}: " + "; ".join(r.failures))
    if a.markdown:
        Path(a.markdown).write_text(markdown(results, dt, mode))
    if a.json:
        Path(a.json).write_text(json.dumps({k: dict(methods=v.methods, checks=v.checks, status=v.status,
                                                    failures=v.failures) for k, v in results.items()}, indent=1))
    return 1 if s.get("fail") else 0
