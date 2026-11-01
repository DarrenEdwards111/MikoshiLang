"""Wolfram coverage batch W09 -- number theory, algebraic number fields, finite fields, domains, group theory and
fractals.

This module holds the shared helpers (arithmetic hooks for special numeric objects, wrappers that extend existing
heads with new argument forms) and the number-theory part of the batch.  Finite fields live in
wl_numbertheory2_ff.py, algebraic numbers and number fields in wl_numbertheory2_alg.py, group theory in
wl_numbertheory2_groups.py and fractals in wl_numbertheory2_fractals.py.

Every function follows the Wolfram Language reference semantics and is checked by the differential tests in
tests/library/test_wolfram_w09.py.  Argument forms that are not implemented raise a clear error instead of returning
a guessed answer.
"""
from __future__ import annotations

import math
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
import importlib
E = importlib.import_module("mikoshilang.evaluate")
from .core import fn, record, Unevaluated, REGISTRY, S, R, L, is_list, has_float, is_numeric_value
from .wl_common import ev, apply, is_sym, split_options, int_value

AREA = "number theory (W09)"
ADDED = dict(area=AREA, status="added")
UNH = K.UNHANDLED


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


def _rule_opts(args, names, fname):
    return split_options(args, names, fname)


def _opt_name(v):
    if isinstance(v, str):
        return v
    if isinstance(v, Symbol):
        return v.name
    return None


# ─────────────────────────── wrappers around existing heads ───────────────────────────

class _NotMine(Exception):
    pass


NOT_MINE = _NotMine


def wrap_registry(name, handler):
    """Give the library head ``name`` a pre-handler: handler(args) returns a result or raises NOT_MINE."""
    e = REGISTRY.get(name)
    if e is None:
        raise RuntimeError(f"wrap_registry: {name} is not a library head")
    first = e.impls[0][0]

    def pre(*args: Expr) -> Expr:
        try:
            return handler(args)
        except _NotMine:
            pass
        n = len(args)
        for f, lo, hi in e.impls_orig:
            if n >= lo and (hi is None or n <= hi):
                return f(*args)
        raise ValueError(f"{name} called with {n} argument(s)")
    if not hasattr(e, "impls_orig"):
        e.impls_orig = list(e.impls)
    e.impls = [(pre, 0, None)]
    return first


def wrap_builtin(name, handler):
    """Give the kernel built-in ``name`` a pre-handler: handler(args, ev, expr) returns a result or raises NOT_MINE."""
    orig = K._BUILTINS[name]

    def pre(a, ev_, expr):
        try:
            return handler(list(a), ev_, expr)
        except _NotMine:
            return orig(a, ev_, expr)
    K._BUILTINS[name] = pre
    return orig


# ─────────────────────────── arithmetic hooks for special numeric objects ───────────────────────────
#
# Objects such as FiniteFieldElement[...], AlgebraicNumber[...], Overflow[] and Underflow[] combine under
# Plus, Times and Power.  The hook runs just before the kernel built-ins (after user upvalues, which take
# precedence as in the Wolfram Language).

SPECIAL_HEADS = set()
ARITH_HANDLERS = []          # fn(expr, ev) -> result or UNH
_ARITH_HEADS = {"Plus", "Times", "Power", "Equal", "Unequal", "Sqrt", "Less", "Greater", "LessEqual",
                "GreaterEqual", "SameQ"}
GUARDS = {}                  # head -> fn(expr) raising for unsupported special arguments


def _is_special(a):
    return isinstance(a, Expr) and isinstance(a.head, str) and a.head in SPECIAL_HEADS


def _contains_special(x, depth=0):
    if _is_special(x):
        return True
    if isinstance(x, Expr) and depth < 40:
        return any(_contains_special(a, depth + 1) for a in x.args)
    return False


def _hook(expr, ev_):
    h = expr.head
    if h in _ARITH_HEADS and any(_is_special(a) or (isinstance(a, Expr) and a.head == "Times" and
                                                    any(_is_special(b) for b in a.args)) for a in expr.args):
        for hd in ARITH_HANDLERS:
            r = hd(expr, ev_)
            if r is not UNH:
                return r
    g = GUARDS.get(h)
    if g is not None and any(_contains_special(a) for a in expr.args):
        g(expr)
    return UNH


def _install_hook():
    from . import wl_evaluation as WE
    prev = WE.apply_upvalues

    def chained(expr, ev_):
        r = prev(expr, ev_)
        if r is not UNH:
            return r
        return _hook(expr, ev_)
    chained._w09 = True
    if not getattr(prev, "_w09", False):
        WE.apply_upvalues = chained
        E.UPVALUE_HOOK = chained


_install_hook()

# N[...] pre-pass: heads whose numeric value needs their unevaluated arguments (e.g. RootSum[f, form]).
N_PRE = {}


def _n_pre_walk(x, prec):
    if isinstance(x, Expr):
        if isinstance(x.head, str) and x.head in N_PRE:
            r = N_PRE[x.head](x, prec)
            if r is not None:
                return r
        args = [_n_pre_walk(a, prec) for a in x.args]
        if any(n is not o for n, o in zip(args, x.args)):
            return Expr(x.head, *args)
    return x


def _has_n_pre(x, depth=0):
    if isinstance(x, Expr) and depth < 60:
        if isinstance(x.head, str) and x.head in N_PRE:
            return True
        return any(_has_n_pre(a, depth + 1) for a in x.args)
    return False


def _n_pre(a, ev_, expr):
    if not a or not _has_n_pre(a[0]):
        raise _NotMine()
    prec = 15
    if len(a) >= 2:
        p = a[1]
        if isinstance(p, (int, float)) and not isinstance(p, bool):
            prec = int(p)
        elif isinstance(p, sp.Basic) and p.is_number:
            prec = int(p)
    new = _n_pre_walk(a[0], prec)
    if new is a[0]:
        raise _NotMine()
    return ev_(Expr("N", new, *a[1:]))




def number_of(x):
    """Exact SymPy number or float for a numeric Mikoshi value, else None."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float, complex)):
        return x
    if isinstance(x, sp.Basic):
        return x if x.is_number else None
    if isinstance(x, Expr) and x.head in ("Rational", "Complex"):
        return S(x)
    if isinstance(x, (Expr, Symbol)) and is_numeric_value(x):
        try:
            v = S(x)
            return v if v.is_number else None
        except Exception:
            return None
    return None


# ─────────────────────────── Overflow[] and Underflow[] ───────────────────────────

SPECIAL_HEADS.update({"Overflow", "Underflow"})


@fn("Overflow", **ADDED, notes="Overflow[] represents a number too large to represent; Overflow[] + x = Overflow[], "
    "c Overflow[] = Sign[c] Overflow[] for real c != 0, 1/Overflow[] = Underflow[]; NumberQ-like tests treat it as "
    "a real number. MikoshiLang's arbitrary-precision numbers never overflow, so it only arises when entered"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _overflow(*a: Expr) -> Expr:
    if a:
        raise ValueError("Overflow takes no arguments")
    raise Unevaluated()


@fn("Underflow", **ADDED, notes="Underflow[] represents a number too small to represent; x + Underflow[] = x for "
    "numbers x != 0 (as a machine real), c Underflow[] = Sign[c] Underflow[], 1/Underflow[] = Overflow[]"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _underflow(*a: Expr) -> Expr:
    if a:
        raise ValueError("Underflow takes no arguments")
    raise Unevaluated()


def _flow_kind(a):
    """('Overflow'|'Underflow', sign) for +-Overflow[]/Underflow[] (possibly Times[-1, ...]), else None."""
    if isinstance(a, Expr) and a.head in ("Overflow", "Underflow") and not a.args:
        return a.head, 1
    if isinstance(a, Expr) and a.head == "Times" and len(a.args) == 2 and a.args[0] == -1 and \
            isinstance(a.args[1], Expr) and a.args[1].head in ("Overflow", "Underflow") and not a.args[1].args:
        return a.args[1].head, -1
    return None


def _flow(kind, sign):
    o = Expr(kind)
    return o if sign > 0 else Expr("Times", -1, o)


def _real_sign(v):
    if isinstance(v, (int, float)):
        return (v > 0) - (v < 0)
    if isinstance(v, complex):
        return None
    if v.is_extended_real:
        return 1 if v.is_positive else (-1 if v.is_negative else 0)
    return None


def _flow_arith(expr, ev_):
    h = expr.head
    kinds = [_flow_kind(a) for a in expr.args]
    if not any(kinds):
        return UNH
    if h == "Plus":
        overs = {s for k, s in (x for x in kinds if x) if k == "Overflow"}
        unders = [x for x in kinds if x and x[0] == "Underflow"]
        others = [a for a, k in zip(expr.args, kinds) if not k]
        nums = [number_of(a) for a in others]
        if any(n is None for n in nums):
            return UNH
        if overs:
            if len(overs) == 2:
                return Symbol("Indeterminate")
            return _flow("Overflow", overs.pop())
        total = sum((sp.nsimplify(n) if isinstance(n, float) else n) for n in nums) if nums else 0
        if nums and total != 0:
            return float(sp.N(total)) if sp.sympify(total).is_real else complex(sp.N(total))
        if len(unders) == 1 and not nums:
            return UNH
        sgn = {s for _, s in unders}
        return _flow("Underflow", sgn.pop()) if len(sgn) == 1 else Symbol("Indeterminate")
    if h == "Times":
        sign, kind = 1, None
        for a, k in zip(expr.args, kinds):
            if k:
                if kind and kind != k[0]:
                    return Symbol("Indeterminate")
                kind = k[0]
                sign *= k[1]
            else:
                n = number_of(a)
                if n is None:
                    return UNH
                s = _real_sign(sp.sympify(n))
                if s is None:
                    return UNH
                if s == 0:
                    return 0 if not isinstance(n, float) else 0.0
                sign *= s
        if len(expr.args) == 2 and expr.args[0] == -1:
            return UNH                       # canonical -Overflow[]
        return _flow(kind, sign)
    if h == "Power" and len(expr.args) == 2 and kinds[0]:
        kind, sign = kinds[0]
        n = number_of(expr.args[1])
        if n is None or not _intq(n) and not isinstance(n, (float, sp.Rational, sp.Float)):
            return UNH
        n = sp.sympify(n)
        if n == 0:
            return 1
        inv = {"Overflow": "Underflow", "Underflow": "Overflow"}
        k2 = kind if n > 0 else inv[kind]
        if sign < 0:
            if not n.is_integer:
                return UNH
            sign = -1 if int(n) % 2 else 1
        return _flow(k2, sign)
    return UNH


ARITH_HANDLERS.append(_flow_arith)


# ─────────────────────────── system precision variables ───────────────────────────

def _sysvar(name, value, note):
    E.SYSTEM_VARIABLES[name] = (lambda v=value: v)

    def assign(lhs, rhs, ev_):
        v = ev_(rhs)
        if K.same(v, value) if isinstance(v, (Expr, Symbol)) else (v == value):
            return
        raise NotImplementedError(f"{name}: MikoshiLang uses a fixed-precision model (precision is the minimum of "
                                  f"the inputs' precisions, with no extra-precision steps), so {name} cannot be "
                                  f"changed from {value}")
    K.ASSIGN_HOOKS[name] = assign
    record(name, AREA, "added", note)


_sysvar("$MaxPrecision", Symbol("Infinity"), "maximum precision of arbitrary-precision numbers: Infinity (no cap); "
        "read-only in MikoshiLang (assigning another value raises, because the fixed-precision model has no "
        "precision clamping)")
_sysvar("$MinPrecision", 0, "minimum precision of arbitrary-precision numbers: 0 (no floor); read-only in "
        "MikoshiLang (assigning another value raises)")


# ─────────────────────────── real-number predicates and parts ───────────────────────────

def _is_real_number_atom(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float)):
        return not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    if isinstance(x, sp.Float):
        return True
    if isinstance(x, Expr) and x.head in ("Rational", "Times", "Power") and not has_float(x):
        try:
            return bool(S(x).is_Rational)
        except Exception:
            return False
    if isinstance(x, (sp.Integer, sp.Rational)):
        return True
    if _flow_kind(x):
        return True
    return False


@fn("RealValuedNumberQ", **ADDED, notes="True for integers, rationals, approximate reals and Overflow[]/"
    "Underflow[]; False for complex numbers (even with zero imaginary part), exact numeric expressions such as "
    "Sqrt[2] and symbols"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _rvnq(x: Expr) -> Boolean:
    return _is_real_number_atom(x)


def _numeric_real(v):
    """True/False/None: is the numeric SymPy expression v real?"""
    if v.is_extended_real is True:
        return bool(v.is_finite)
    if v.is_extended_real is False or v.is_infinite:
        return False
    for prec in (40, 120):
        try:
            z = sp.N(v, prec)
        except Exception:
            return None
        re_, im_ = z.as_real_imag()
        if not (re_.is_number and im_.is_number):
            return None
        if im_ == 0:
            return True
        if abs(im_) > sp.Float(10) ** (-(prec // 2)) * max(1, abs(re_)):
            return False
    return True


@fn("RealValuedNumericQ", **ADDED, notes="True when expr is numeric (NumericQ) and real-valued: exact quantities "
    "such as Sin[Sqrt[2]], E or Exp[I] - I Sin[1] (decided exactly when possible, otherwise numerically at 40 and "
    "120 digits); False for complex numbers with a zero inexact imaginary part, Infinity and non-numeric input"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _rvnumq(x: Expr) -> Boolean:
    if _is_real_number_atom(x):
        return True
    if isinstance(x, (bool, str, complex)):
        return False
    if isinstance(x, Symbol) and x.name in ("Infinity", "ComplexInfinity", "Indeterminate"):
        return False
    if not is_numeric_value(x):
        return False
    if has_float(x) and isinstance(x, Expr):
        try:
            v = S(x)
        except Exception:
            return False
        if isinstance(v, sp.Basic) and v.is_number:
            if any(isinstance(a, sp.Float) for a in sp.preorder_traversal(v)) and v.as_real_imag()[1] != 0:
                return False
    try:
        v = S(x)
    except Exception:
        return False
    if not isinstance(v, sp.Basic) or not v.is_number or v.has(sp.zoo, sp.oo, sp.nan):
        return False
    r = _numeric_real(v)
    return bool(r)


@fn("RealSign", **ADDED, listable=True, notes="sign of a real number: -1, 0 or 1 (an exact integer, also for "
    "approximate input); RealSign[Infinity] = 1, RealSign[-Infinity] = -1; complex and symbolic arguments stay "
    "unevaluated"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _realsign(x: Expr) -> Expr:
    if isinstance(x, Symbol) and x.name == "Infinity":
        return 1
    if isinstance(x, Expr) and K.same(x, Expr("Times", -1, Symbol("Infinity"))):
        return -1
    k = _flow_kind(x)
    if k:
        return k[1]
    if not is_numeric_value(x) or isinstance(x, complex):
        raise Unevaluated()
    v = S(x)
    if not isinstance(v, sp.Basic) or not v.is_number:
        raise Unevaluated()
    r = _numeric_real(v)
    if r is not True:
        raise Unevaluated()
    if v.is_zero:
        return 0
    if v.is_positive is not None:
        return 1 if v.is_positive else -1
    return 1 if sp.N(v, 50) > 0 else -1


_MACHINE_ZERO_EXPONENT = math.log10(2.2250738585072014e-308)


@fn("RealExponent", **ADDED, listable=True, notes="RealExponent[x] = Log10[Abs[x]], RealExponent[x, b] = "
    "Log[b, Abs[x]], as machine reals; RealExponent[0] = -Infinity; RealExponent[0.] = -307.653 (minus the "
    "accuracy of a machine zero); symbolic arguments stay unevaluated"
    ". Check: reference (differential cases in tests/library).")
def _realexp(x: Expr, b: Expr = 10) -> Expr:
    if not is_numeric_value(x) or (not is_numeric_value(b)):
        raise Unevaluated()
    vb = S(b)
    if isinstance(x, float) and x == 0.0 or isinstance(x, sp.Float) and x == 0:
        lv = _MACHINE_ZERO_EXPONENT
        return lv / math.log10(float(vb)) if vb != 10 else lv
    v = S(x)
    if v == 0:
        return Expr("Times", -1, Symbol("Infinity"))
    a = abs(complex(sp.N(v, 30)))
    fb = complex(sp.N(vb, 30))
    if fb.imag != 0 or fb.real <= 0 or fb.real == 1:
        raise ValueError(f"RealExponent: the base {b!r} must be a positive real number other than 1")
    return float(sp.N(sp.log(sp.Abs(v)) / sp.log(vb), 20)) if a != float("inf") else float(
        mathlog(a) / math.log(fb.real))


def mathlog(a):
    return math.log(a)


@fn("MixedFractionParts", **ADDED, listable=True, notes="{IntegerPart[x], FractionalPart[x]} (both parts carry "
    "the sign of x; complex numbers are split componentwise); symbolic arguments stay unevaluated"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _mixedfrac(x: Expr) -> ListOf[Expr]:
    if not is_numeric_value(x):
        raise Unevaluated()
    ip = ev(Expr("IntegerPart", x))
    fp = ev(Expr("FractionalPart", x))
    if isinstance(ip, Expr) and ip.head == "IntegerPart":
        raise Unevaluated()
    return L(ip, fp)


@fn("NumeratorDenominator", **ADDED, notes="{Numerator[expr], Denominator[expr]}; lists give the pair of lists. "
    "The Modulus and Trig options are not supported (raise)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _numden(x: Expr, *opts: Option) -> ListOf[Expr]:
    if opts:
        raise NotImplementedError("NumeratorDenominator: the Modulus and Trig options are not supported")
    return L(ev(Expr("Numerator", x)), ev(Expr("Denominator", x)))


def _num_list(x, fname, what):
    if not is_list(x):
        raise ValueError(f"{fname}: {what} must be a list")
    out = []
    for v in x.args:
        if not is_numeric_value(v):
            raise Unevaluated()
        out.append(v)
    return out


def _sp_or_float(v):
    if isinstance(v, (float, complex)):
        return v
    return S(v)


@fn("NumberDecompose", **ADDED, notes="NumberDecompose[x, {u1, ..., un}]: coefficients ci with x == Sum[ci ui]; "
    "c1..c(n-1) are integers taken greedily with IntegerPart (so the signs follow x) and cn = remainder/un may be "
    "fractional; exact inputs give exact results"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _numdecomp(x: Expr, units: ListOf[Expr]) -> ListOf[Expr]:
    us = _num_list(units, "NumberDecompose", "the units")
    if not us:
        raise ValueError("NumberDecompose: the list of units must not be empty")
    if not is_numeric_value(x):
        raise Unevaluated()
    inexact = has_float(x) or any(has_float(u) for u in us)
    r = _sp_or_float(x)
    out = []
    for u in us[:-1]:
        uv = _sp_or_float(u)
        if uv == 0:
            raise ValueError("NumberDecompose: units must be non-zero")
        q = r / uv
        if isinstance(q, complex) or (isinstance(q, sp.Basic) and q.is_extended_real is False):
            raise ValueError("NumberDecompose: x and the units must be real")
        c = int(sp.Integer(sp.sign(q) * sp.floor(abs(q)))) if isinstance(q, sp.Basic) else int(q)
        out.append(c)
        r = r - c * uv
        if isinstance(r, sp.Basic):
            r = sp.expand(r)
    last = r / _sp_or_float(us[-1])
    if isinstance(last, sp.Basic):
        last = sp.expand(last)
        if inexact and last.is_number:
            last = float(last) if last.is_real else complex(last)
    out.append(last)
    return L(*[c if isinstance(c, (int, float, complex)) else R(c) for c in out])


@fn("NumberCompose", **ADDED, notes="NumberCompose[{c1, ..., ck}, {u1, ..., un}] = Sum[ci u(n-k+i)]: the "
    "coefficients are aligned with the last k units (k <= n)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _numcompose(cs: Expr, units: Expr) -> Expr:
    if not is_list(cs) or not is_list(units):
        raise ValueError("NumberCompose: both arguments must be lists")
    c, u = list(cs.args), list(units.args)
    if len(c) > len(u):
        raise ValueError("NumberCompose: more coefficients than units")
    u = u[len(u) - len(c):]
    return ev(Expr("Plus", *[Expr("Times", a, b) for a, b in zip(c, u)])) if c else 0


# ─────────────────────────── MixedRadix ───────────────────────────

def _mixed_bases(mr, fname):
    if not (isinstance(mr, Expr) and mr.head == "MixedRadix" and len(mr.args) == 1 and is_list(mr.args[0])):
        raise _NotMine()
    bs = []
    for b in mr.args[0].args:
        if not _intq(b) or int(b) < 1:
            raise ValueError(f"{fname}: MixedRadix bases must be positive integers")
        bs.append(int(b))
    if not bs:
        raise ValueError(f"{fname}: MixedRadix needs at least one base")
    return bs


@fn("MixedRadix", **ADDED, notes="MixedRadix[{b1, ..., bn}] mixed-radix base for IntegerDigits, FromDigits, "
    "IntegerLength and IntegerReverse; the leading digit is unbounded")
def _mixedradix(bs: Expr) -> Expr:
    if is_list(bs):
        for b in bs.args:
            if _intq(b) and int(b) < 1:
                raise ValueError("MixedRadix: bases must be positive integers")
    raise Unevaluated()


def mixed_digits(n, bs):
    n = abs(n)
    ds = []
    for b in reversed(bs):
        if b == 1:
            ds.append(0)
            continue
        n, d = divmod(n, b)
        ds.append(d)
    if n:
        ds.append(n)
    ds.reverse()
    # strip leading zeros (but keep at least one digit)
    while len(ds) > 1 and ds[0] == 0:
        ds.pop(0)
    return ds


def mixed_value(ds, bs, fname):
    if len(ds) > len(bs) + 1:
        raise ValueError(f"{fname}: more digits than MixedRadix bases allow ({len(bs) + 1})")
    radices = list(reversed(bs))
    acc, mult = sp.Integer(0), sp.Integer(1)
    for i, d in enumerate(reversed(ds)):
        acc += mult * d
        if i < len(radices):
            mult *= radices[i]
    return acc


def _intdigits_mr(args):
    if len(args) < 2:
        raise _NotMine()
    bs = _mixed_bases(args[1], "IntegerDigits")
    if not _intq(args[0]):
        if is_numeric_value(args[0]):
            raise ValueError("IntegerDigits: the number must be an integer")
        raise Unevaluated()
    ds = mixed_digits(int(args[0]), bs)
    if len(args) == 3:
        ln = int_value(args[2], "IntegerDigits", "length", 0)
        ds = ([0] * ln + ds)[-ln:] if ln else []
    elif len(args) > 3:
        raise ValueError("IntegerDigits takes at most 3 arguments")
    return L(*ds)


def _fromdigits_mr(args):
    if len(args) != 2:
        raise _NotMine()
    bs = _mixed_bases(args[1], "FromDigits")
    if not is_list(args[0]):
        raise ValueError("FromDigits: the digits must be a list")
    ds = [S(d) for d in args[0].args]
    return R(sp.expand(mixed_value(ds, bs, "FromDigits")))


def _intlength_mr(args):
    if len(args) != 2:
        raise _NotMine()
    bs = _mixed_bases(args[1], "IntegerLength")
    if not _intq(args[0]):
        raise Unevaluated()
    if int(args[0]) == 0:
        return 0
    return len(mixed_digits(int(args[0]), bs))


def _intreverse_mr(args):
    if len(args) != 2:
        raise _NotMine()
    bs = _mixed_bases(args[1], "IntegerReverse")
    if not _intq(args[0]):
        raise Unevaluated()
    ds = mixed_digits(int(args[0]), bs)
    return int(mixed_value(list(reversed(ds)), bs, "IntegerReverse"))


wrap_registry("IntegerDigits", _intdigits_mr)
wrap_registry("FromDigits", _fromdigits_mr)
for _nm, _h in (("IntegerLength", _intlength_mr), ("IntegerReverse", _intreverse_mr)):
    if _nm in REGISTRY:
        wrap_registry(_nm, _h)


# ─────────────────────────── domains ───────────────────────────

_BASE_DOMAINS = ("Reals", "Integers", "Rationals", "Complexes", "Algebraics", "Primes", "Booleans")
_SIGNED_DOMAINS = {}
for _sg, _rel in (("Positive", "Greater"), ("Negative", "Less"), ("NonNegative", "GreaterEqual"),
                  ("NonPositive", "LessEqual")):
    for _base in ("Integers", "Rationals", "Reals"):
        _SIGNED_DOMAINS[_sg + _base] = (_base, _rel)

for _d in _BASE_DOMAINS:
    record(_d, AREA, "added", f"the domain of {_d.lower()}, used in Element, assumptions (Refine, Simplify, "
           "Integrate, ...) and as a domain argument (Solve, Reduce, FindInstance)")
for _d, (_b, _r) in _SIGNED_DOMAINS.items():
    record(_d, AREA, "added", f"the domain of {_d[: -len(_b)].lower()} {_b.lower()}: Element[x, {_d}] is "
           f"Element[x, {_b}] && x {dict(Greater='>', Less='<', GreaterEqual='>=', LessEqual='<=')[_r]} 0; "
           "usable in Element, $Assumptions and Assuming")


def _element_one(x, dom):
    """True/False/None for a single (non-list, non-Alternatives) element."""
    r = ev(Expr("Element", x, Symbol(dom)))
    if r is True or r is False:
        return r
    return None


def _signed_member(x, dom):
    base, rel = _SIGNED_DOMAINS[dom]
    if isinstance(x, (bool, str)):
        return False
    b = _element_one(x, base)
    if b is False:
        return False
    if not is_numeric_value(x):
        return None
    c = ev(Expr(rel, x, 0))
    if c is False:
        return False
    if b is True and c is True:
        return True
    return None


def _element_pre(args):
    if len(args) != 2 or not isinstance(args[1], Symbol):
        raise _NotMine()
    x, dom = args
    d = dom.name
    signed = d in _SIGNED_DOMAINS
    alts = isinstance(x, Expr) and x.head == "Alternatives"
    if not signed and not alts:
        raise _NotMine()
    if not signed and d not in _BASE_DOMAINS:
        raise _NotMine()
    items = list(x.args) if (alts or is_list(x)) else [x]
    res = [(_signed_member(it, d) if signed else _element_one(it, d)) for it in items]
    if any(r is False for r in res):
        return False
    if all(r is True for r in res):
        return True
    rest = [it for it, r in zip(items, res) if r is not True]
    if alts:
        rest_x = rest[0] if len(rest) == 1 else Expr("Alternatives", *rest)
        if len(rest) == len(items):
            raise Unevaluated()
        return Expr("Element", rest_x, dom)
    raise Unevaluated()


wrap_registry("Element", _element_pre)


def _install_assume_domains():
    from . import assume as A
    orig = A.Assumed._walk

    def walk(self, c):
        if isinstance(c, Expr) and c.head == "Element" and len(c.args) == 2 and isinstance(c.args[1], Symbol) \
                and c.args[1].name in _SIGNED_DOMAINS:
            base, rel = _SIGNED_DOMAINS[c.args[1].name]
            xs = c.args[0]
            xs = xs.args if isinstance(xs, Expr) and xs.head in ("List", "Alternatives") else (xs,)
            for x in xs:
                orig(self, Expr("Element", x, Symbol(base)))
                orig(self, Expr(rel, x, 0))
            return
        if isinstance(c, Expr) and c.head == "Element" and len(c.args) == 2 and isinstance(c.args[0], Expr) \
                and c.args[0].head == "Alternatives":
            return orig(self, Expr("Element", Expr("List", *c.args[0].args), c.args[1]))
        return orig(self, c)
    if not getattr(orig, "_w09", False):
        walk._w09 = True
        A.Assumed._walk = walk


_install_assume_domains()


# ─────────────────────────── GaussianIntegers, Modulus, Extension ───────────────────────────

record("GaussianIntegers", AREA, "added", "option for Factor, FactorInteger, PrimeQ, Divisors, ... : "
       "GaussianIntegers -> True works over the Gaussian integers Z[I]")
record("Modulus", AREA, "added", "option for Factor, PolynomialGCD, PolynomialQuotient/Remainder, Expand, "
       "Together, Det, Inverse, ... : Modulus -> p works modulo p (Det accepts any modulus, Inverse a prime "
       "or a modulus coprime to the determinant)")
record("Extension", AREA, "added", "option for Factor, PolynomialGCD, ... : Extension -> a (or {a1, ...}) "
       "works over the algebraic extension Q(a); Extension -> Automatic uses the algebraic numbers in the input")


def _gauss_of(x):
    """(a, b) for an exact Gaussian integer a + b I, else None."""
    if _intq(x):
        return int(x), 0
    if isinstance(x, (Expr, Symbol)) and not isinstance(x, bool):
        try:
            v = S(x)
        except Exception:
            return None
        if isinstance(v, sp.Basic) and v.is_number:
            re_, im_ = v.as_real_imag()
            if re_.is_Integer and im_.is_Integer:
                return int(re_), int(im_)
    return None


def _gmul(a, b):
    return a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0]


def _gdivexact(a, b):
    """a / b in Z[i] or None."""
    n = b[0] * b[0] + b[1] * b[1]
    re_ = a[0] * b[0] + a[1] * b[1]
    im_ = a[1] * b[0] - a[0] * b[1]
    if re_ % n or im_ % n:
        return None
    return re_ // n, im_ // n


def _gnormalize(z):
    """Associate of z in the first quadrant: re > 0, im >= 0."""
    a, b = z
    for _ in range(4):
        if a > 0 and b >= 0:
            return a, b
        a, b = -b, a
    return z


def _sum_two_squares(p):
    """x, y with x^2 + y^2 = p (p prime, p % 4 == 1)."""
    # Cornacchia / Hermite-Serret
    t = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            t = pow(c, (p - 1) // 4, p)
            break
    a, b = p, t
    while b * b > p:
        a, b = b, a % b
    return b, math.isqrt(p - b * b)


def gaussian_factor(z):
    """[(unit, prime, exponent)...] for a nonzero Gaussian integer z -> (unit, [(prime, e)])."""
    a, b = z
    n = a * a + b * b
    facs = []
    rem = z
    for p, e in sorted(sp.factorint(n).items()):
        if p == 2:
            pi = (1, 1)
            k = 0
            while True:
                q = _gdivexact(rem, pi)
                if q is None:
                    break
                rem, k = q, k + 1
            facs.append((pi, k))
        elif p % 4 == 3:
            k = e // 2
            for _ in range(k):
                rem = _gdivexact(rem, (p, 0))
            facs.append(((p, 0), k))
        else:
            x, y = _sum_two_squares(p)
            for pi in {_gnormalize((x, y)), _gnormalize((x, -y))}:
                k = 0
                while True:
                    q = _gdivexact(rem, pi)
                    if q is None:
                        break
                    rem, k = q, k + 1
                if k:
                    facs.append((pi, k))
    facs.sort(key=lambda t: (t[0][0], t[0][1]))
    return rem, facs


def _gexpr(z):
    a, b = z
    return R(sp.Integer(a) + sp.Integer(b) * sp.I)


def gaussian_primeq(z):
    a, b = z
    if a == 0 or b == 0:
        m = abs(a + b)
        return bool(sp.isprime(m)) and m % 4 == 3
    return bool(sp.isprime(a * a + b * b))


def _gauss_opt(args, fname):
    pos, opts = _rule_opts(args, ("GaussianIntegers",), fname)
    g = opts.get("GaussianIntegers")
    if g is None:
        return pos, None
    if g is not True and g is not False:
        raise ValueError(f"{fname}: GaussianIntegers must be True or False")
    return pos, g


def _factorinteger_pre(args):
    pos, g = _gauss_opt(args, "FactorInteger")
    if len(pos) != 1:
        raise _NotMine()
    z = _gauss_of(pos[0])
    if g is None:
        if z is None or z[1] == 0:
            raise _NotMine()
        g = True
    if not g:
        if z is not None and z[1] != 0:
            raise ValueError("FactorInteger: a Gaussian integer needs GaussianIntegers -> True")
        return ev(Expr("FactorInteger", pos[0]))
    if z is None:
        if is_numeric_value(pos[0]):
            raise ValueError("FactorInteger: the argument must be a Gaussian integer")
        raise Unevaluated()
    if z == (0, 0):
        return L(L(0, 1))
    unit, facs = gaussian_factor(z)
    out = []
    if unit != (1, 0) or not facs:
        out.append(L(_gexpr(unit), 1))
    out += [L(_gexpr(p), k) for p, k in facs]
    return L(*out)


wrap_registry("FactorInteger", _factorinteger_pre)


def _primeq_pre(a, ev_, expr):
    pos, g = _gauss_opt(a, "PrimeQ")
    if len(pos) != 1:
        if len(pos) > 1:
            raise ValueError("PrimeQ takes one argument (plus the GaussianIntegers option)")
        raise _NotMine()
    z = _gauss_of(pos[0])
    if g is None:
        if z is None or z[1] == 0:
            raise _NotMine()
        g = True
    if not g:
        return bool(z and z[1] == 0 and sp.isprime(abs(z[0])))
    if z is None:
        return False
    return gaussian_primeq(z)


wrap_builtin("PrimeQ", _primeq_pre)


def _modmat(m, fname):
    if not is_list(m) or not m.args or not all(is_list(r) for r in m.args):
        raise ValueError(f"{fname}: the first argument must be a matrix")
    rows = []
    for r in m.args:
        row = []
        for v in r.args:
            if _intq(v):
                row.append(int(v))
            elif isinstance(v, Expr) and v.head == "Rational":
                row.append(Fraction(int(v.args[0]), int(v.args[1])))
            else:
                raise ValueError(f"{fname}: with Modulus the matrix entries must be integers or rationals")
        rows.append(row)
    return rows


def _modulus_opt(a, fname):
    pos, opts = _rule_opts(a, ("Modulus",), fname)
    if "Modulus" not in opts:
        raise _NotMine()
    m = opts["Modulus"]
    if not _intq(m) or int(m) < 0:
        raise ValueError(f"{fname}: Modulus must be a non-negative integer")
    return pos, int(m)


def _to_mod(v, m):
    if isinstance(v, Fraction):
        return v.numerator * pow(v.denominator, -1, m) % m
    return v % m


def _det_pre(a, ev_, expr):
    pos, m = _modulus_opt(a, "Det")
    if len(pos) != 1:
        raise ValueError("Det takes one matrix argument (plus Modulus)")
    if m == 0:
        return ev_(Expr("Det", pos[0]))
    rows = _modmat(pos[0], "Det")
    if len(set(map(len, rows))) != 1 or len(rows[0]) != len(rows):
        raise ValueError("Det: the matrix must be square")
    M = sp.Matrix([[_to_mod(v, m) for v in r] for r in rows])
    return int(M.det(method="bareiss")) % m


def _inverse_pre(a, ev_, expr):
    pos, m = _modulus_opt(a, "Inverse")
    if len(pos) != 1:
        raise ValueError("Inverse takes one matrix argument (plus Modulus)")
    if m == 0:
        return ev_(Expr("Inverse", pos[0]))
    rows = _modmat(pos[0], "Inverse")
    if len(set(map(len, rows))) != 1 or len(rows[0]) != len(rows):
        raise ValueError("Inverse: the matrix must be square")
    M = sp.Matrix([[_to_mod(v, m) for v in r] for r in rows])
    d = int(M.det(method="bareiss")) % m
    if math.gcd(d, m) != 1:
        raise ValueError(f"Inverse: the matrix is singular modulo {m}")
    inv = M.inv_mod(m)
    return L(*[L(*[int(inv[i, j]) % m for j in range(inv.cols)]) for i in range(inv.rows)])


wrap_builtin("N", _n_pre)
wrap_builtin("Det", _det_pre)
wrap_builtin("Inverse", _inverse_pre)


# ─────────────────────────── polynomials over the integers and Z/p ───────────────────────────

def _poly_var(poly, fname, var=None):
    if var is not None:
        return S(var)
    fs = sorted(S(poly).free_symbols, key=str)
    if len(fs) != 1:
        raise ValueError(f"{fname}: the polynomial must be univariate")
    return fs[0]


@fn("Cyclotomic", **ADDED, notes="Cyclotomic[n, x] is the n-th cyclotomic polynomial (product of x - zeta over "
    "the primitive n-th roots of unity); Cyclotomic[0, x] = 1; any x, including numbers and matrices of "
    "symbols, is substituted"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _cyclotomic(n: Integer, x: Expr) -> Expr:
    if not _intq(n):
        raise Unevaluated()
    k = int(n)
    if k < 0:
        raise ValueError("Cyclotomic: n must be a non-negative integer")
    t = sp.Symbol("_w09t")
    p = sp.Integer(1) if k == 0 else sp.cyclotomic_poly(k, t)
    if isinstance(x, (Symbol, Expr)) and not is_list(x) or isinstance(x, (int, float, complex)):
        return R(sp.expand(p.subs(t, S(x))) if not isinstance(x, Symbol) else p.subs(t, S(x)))
    raise Unevaluated()


def _gf_poly_coeffs(poly, p, fname, var=None):
    """Coefficient list (high to low) of poly modulo p, made monic; None if not a polynomial."""
    x = _poly_var(poly, fname, var)
    e = sp.together(S(poly))
    num, den = sp.fraction(e)
    P = sp.Poly(num, x)
    D = sp.Poly(den, x)
    if D.degree() > 0:
        raise ValueError(f"{fname}: {poly} is not a polynomial")
    d = int(D.LC()) % p
    if d == 0:
        raise ValueError(f"{fname}: the denominator vanishes modulo {p}")
    cs = [(int(sp.Rational(c).p) * pow(int(sp.Rational(c).q) * d, -1, p)) % p for c in P.all_coeffs()]
    while cs and cs[0] == 0:
        cs.pop(0)
    if not cs:
        raise ValueError(f"{fname}: the polynomial vanishes modulo {p}")
    inv = pow(cs[0], -1, p)
    return [c * inv % p for c in cs]


def _gf_mulmod(a, b, m, p):
    """a*b mod m over GF(p); polynomials as low-to-high lists, m monic."""
    n = len(m) - 1
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] = (r[i + j] + x * y) % p
    for k in range(len(r) - 1, n - 1, -1):
        c = r[k]
        if c:
            for t in range(n + 1):
                r[k - n + t] = (r[k - n + t] - c * m[t]) % p
    r = r[:n] + [0] * max(0, n - len(r))
    return r


def _gf_powmod(base, e, m, p):
    n = len(m) - 1
    result = [1] + [0] * (n - 1)
    b = (list(base) + [0] * n)[:max(n, len(base))]
    while e:
        if e & 1:
            result = _gf_mulmod(result, b, m, p)
        b = _gf_mulmod(b, b, m, p)
        e >>= 1
    return result


def gf_primitive(cs_high, p):
    """Is the monic polynomial (coefficients high to low) primitive over GF(p)?"""
    m = list(reversed(cs_high))
    n = len(m) - 1
    if n < 1 or m[0] % p == 0:
        return False
    N = p ** n - 1
    one = [1] + [0] * (n - 1)
    x = [0, 1] if n > 1 else [(-m[0]) % p]
    if _gf_powmod(x, N, m, p) != one:
        return False
    for q in sp.factorint(N):
        if _gf_powmod(x, N // q, m, p) == one:
            return False
    return True


@fn("PrimitivePolynomialQ", **ADDED, notes="PrimitivePolynomialQ[poly, p]: True if the univariate poly, reduced "
    "modulo the prime p and made monic, is primitive over GF(p) (irreducible with x of order p^n - 1)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _primpolyq(poly: Expr, p: Expr) -> Boolean:
    if not _intq(p) or not sp.isprime(int(p)):
        raise ValueError("PrimitivePolynomialQ: the modulus must be a prime")
    if not isinstance(poly, (Expr, Symbol)) or isinstance(poly, bool):
        return False
    try:
        cs = _gf_poly_coeffs(poly, int(p), "PrimitivePolynomialQ")
    except (ValueError, sp.PolynomialError, sp.GeneratorsNeeded):
        return False
    return gf_primitive(cs, int(p))


# ─────────────────────────── continued fractions ───────────────────────────

K.BUILTIN_ATTRIBUTES["ContinuedFractionK"] = {"HoldAll", "Protected"}


def _iter_spec(spec, fname):
    if not is_list(spec) or not (1 <= len(spec.args) <= 3) or not isinstance(spec.args[0], Symbol):
        raise ValueError(f"{fname}: the iterator must be {{i, imax}} or {{i, imin, imax}}")
    i = spec.args[0]
    if len(spec.args) == 2:
        lo, hi = 1, ev(spec.args[1])
    else:
        lo, hi = ev(spec.args[1]), ev(spec.args[2])
    return i, lo, hi


def _at(f, i, v):
    return ev(K.substitute(f, {i.name: v}))


def _free_of(f, i):
    if isinstance(f, Symbol):
        return f.name != i.name
    if isinstance(f, Expr):
        return _free_of(f.head, i) if not isinstance(f.head, str) else True and all(_free_of(a, i) for a in f.args)
    return True


@fn("ContinuedFractionK", **ADDED, hold=True, notes="ContinuedFractionK[g, {i, imin, imax}] = 1/(g_imin + "
    "1/(g_imin+1 + ...)) and ContinuedFractionK[f, g, {i, imin, imax}] = f_imin/(g_imin + f_imin+1/(g_imin+1 + "
    "...)); finite numeric limits are evaluated exactly; infinite fractions with constant terms give the closed "
    "form 2 f/(g + Sqrt[g^2 + 4 f]) (for numeric terms the attracting fixed point, unevaluated when the fraction "
    "oscillates); other infinite or symbolic-length fractions stay unevaluated"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _cfk(*args: Held) -> Expr:
    pos, opts = split_options(args, ("Assumptions", "GenerateConditions", "Method", "VerifyConvergence"),
                              "ContinuedFractionK")
    gc = opts.get("GenerateConditions")
    if gc is not None and ev(gc) is True:
        raise NotImplementedError("ContinuedFractionK: GenerateConditions -> True is not supported")
    if len(pos) == 2:
        f, g, spec = 1, pos[0], pos[1]
    elif len(pos) == 3:
        f, g, spec = pos
    else:
        raise ValueError("ContinuedFractionK takes 2 or 3 arguments")
    if is_list(spec) and len(spec.args) and is_list(spec.args[0]):
        raise NotImplementedError("ContinuedFractionK: only one iterator is supported")
    i, lo, hi = _iter_spec(spec, "ContinuedFractionK")
    inf = isinstance(hi, Symbol) and hi.name == "Infinity"
    if inf:
        if not (_free_of(f, i) and _free_of(g, i)):
            raise Unevaluated()
        fv, gv = ev(f), ev(g)
        a, b = S(fv), S(gv)
        if a.is_number and b.is_number:
            disc = sp.nsimplify(b ** 2 + 4 * a) if not has_float(fv) and not has_float(gv) else b ** 2 + 4 * a
            r1 = 2 * a / (b + sp.sqrt(disc))
            r2 = 2 * a / (b - sp.sqrt(disc)) if disc != 0 else r1
            m1 = abs(complex(sp.N(b + a / (b + r1) if b + r1 != 0 else sp.oo))) if False else None
            # attracting fixed point of x -> a/(b + x): |b + x| is the larger one
            v1, v2 = complex(sp.N(b + r1)), complex(sp.N(b - sp.sqrt(disc)) if disc != 0 else sp.N(b + r1))
            if abs(abs(v1) - abs(v2)) < 1e-12 * max(1.0, abs(v1)) and disc != 0:
                raise Unevaluated()
            res = r1 if abs(v1) >= abs(v2) else r2
            out = R(sp.radsimp(res) if not res.has(sp.Float) else res)
            return ev(out) if not has_float(fv) and not has_float(gv) else ev(Expr("N", out))
        return ev(Expr("Times", 2, fv, Expr("Power", Expr("Plus", gv, Expr("Power", Expr(
            "Plus", Expr("Power", gv, 2), Expr("Times", 4, fv)), Expr("Rational", 1, 2) if False else
            Expr("Times", 1, Expr("Power", 2, -1)))), -1)))
    if not (_intq(lo) and _intq(hi)):
        if is_numeric_value(lo) and is_numeric_value(hi):
            lo_i, hi_i = int(sp.floor(S(lo))), int(sp.floor(S(hi)))
        else:
            raise Unevaluated()
    else:
        lo_i, hi_i = int(lo), int(hi)
    if hi_i - lo_i > 100000:
        raise ValueError("ContinuedFractionK: too many terms")
    v = 0
    for k in range(hi_i, lo_i - 1, -1):
        fk = _at(f, i, k) if not isinstance(f, int) else f
        gk = _at(g, i, k)
        v = ev(Expr("Times", fk, Expr("Power", Expr("Plus", gk, v), -1)))
    return ev(Expr("Together", v)) if not has_float(v) and isinstance(v, Expr) and not is_numeric_value(v) else v


# ─────────────────────────── sequences ───────────────────────────

def _fkm(k, n):
    """Lexicographically least de Bruijn sequence B(k, n) (Fredricksen-Kessler-Maiorana)."""
    a = [0] * (k * n)
    seq = []

    def db(t, p):
        if t > n:
            if n % p == 0:
                seq.extend(a[1:p + 1])
        else:
            a[t] = a[t - p]
            db(t + 1, p)
            for j in range(a[t - p] + 1, k):
                a[t] = j
                db(t + 1, t)
    db(1, 1)
    return seq


@fn("DeBruijnSequence", **ADDED, notes="DeBruijnSequence[list, n]: the lexicographically first de Bruijn "
    "sequence (FKM algorithm) in the order of the elements of list (duplicates removed, first occurrence kept); "
    "DeBruijnSequence[k, n] uses Range[0, k - 1]; a string gives a string. The positions-based form "
    "DeBruijnSequence[list, n, positions] is not supported"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _debruijn(lst: Expr, n: Expr) -> Expr:
    if not _intq(n) or int(n) < 0:
        raise ValueError("DeBruijnSequence: n must be a non-negative integer")
    n = int(n)
    if isinstance(lst, str):
        alph = list(dict.fromkeys(lst))
        if n == 0:
            return ""
        return "".join(alph[i] for i in _fkm(len(alph), n))
    if _intq(lst):
        if int(lst) < 1:
            raise ValueError("DeBruijnSequence: k must be a positive integer")
        alph = list(range(int(lst)))
    elif is_list(lst):
        alph = []
        for x in lst.args:
            if not any(K.same(x, y) for y in alph):
                alph.append(x)
    else:
        raise ValueError("DeBruijnSequence: the first argument must be a list, a string or a positive integer")
    if n == 0 or not alph:
        return L()
    if len(alph) ** n > 10 ** 7:
        raise ValueError("DeBruijnSequence: the sequence is too long")
    return L(*[alph[i] for i in _fkm(len(alph), n)])


def _default_lfsr_poly(n):
    """Exponents (descending, without n and 0) of the feedback polynomial used by ShiftRegisterSequence[n]:
    the primitive trinomial x^n + x^k + 1 with the least k, else the primitive pentanomial x^n + x^a + x^b +
    x^c + 1 with the lexicographically least (c, b, a)."""
    for k in range(1, n):
        cs = [0] * (n + 1)
        cs[0] = cs[n] = 1
        cs[k] = 1
        if gf_primitive(list(reversed(cs)), 2):
            return [k]
    import itertools
    for c in range(1, n):
        for b in range(c + 1, n):
            for a in range(b + 1, n):
                cs = [0] * (n + 1)
                cs[0] = cs[n] = cs[a] = cs[b] = cs[c] = 1
                if gf_primitive(list(reversed(cs)), 2):
                    return [a, b, c]
    raise ValueError(f"ShiftRegisterSequence: no primitive polynomial of degree {n} with at most 5 terms")


def _poly_lowhigh(poly, p, fname):
    x = _poly_var(poly, fname)
    P = sp.Poly(S(poly), x)
    cs = list(reversed([int(c) % p for c in P.all_coeffs()]))
    if any(not sp.sympify(c).is_Integer for c in P.all_coeffs()):
        raise ValueError(f"{fname}: the polynomial must have integer coefficients")
    if cs[-1] % p != 1:
        raise ValueError(f"{fname}: the polynomial must be monic")
    return cs, x


def _srs_run(step, state, length):
    out = []
    s = list(state)
    if length > 10 ** 7:
        raise ValueError("ShiftRegisterSequence: the sequence is too long")
    for _ in range(length):
        out.append(s[-1])
        s = step(s)
    return out


@fn("ShiftRegisterSequence", **ADDED, notes="linear-feedback shift register sequences: ShiftRegisterSequence[n] "
    "(maximal length over GF(2), feedback polynomial x^n + x^k + 1 with least k or else the lexicographically "
    "first primitive pentanomial), [{n, {t1, ...}}] (Fibonacci register: the new bit is the sum mod 2 of the "
    "bits at the tap positions), [{n, taps, f}] (feedback function f, result taken mod 2), [poly] (Galois register), with optional "
    "initial state (list, or polynomial for the Galois form) and length (an integer or All, default one full "
    "period p^n - 1, or 2^n terms with a feedback function); options Method -> \"Fibonacci\"|\"Galois\" and Modulus -> p (Galois form)")
def _srs(*args: Expr) -> ListOf[Expr]:
    pos, opts = split_options(args, ("Method", "Modulus"), "ShiftRegisterSequence")
    if not pos or len(pos) > 3:
        raise ValueError("ShiftRegisterSequence takes 1 to 3 arguments")
    spec = pos[0]
    p = opts.get("Modulus", 2)
    if not _intq(p) or not sp.isprime(int(p)):
        raise ValueError("ShiftRegisterSequence: Modulus must be a prime")
    p = int(p)
    method = opts.get("Method")
    method = method if isinstance(method, str) or method is None else _opt_name(method)
    init = pos[1] if len(pos) == 3 else None
    length = pos[2] if len(pos) == 3 else (pos[1] if len(pos) == 2 else None)
    if len(pos) == 2 and is_list(pos[1]):
        init, length = pos[1], None
    if length is not None:
        if isinstance(length, Symbol) and length.name == "All":
            length = None
        elif _intq(length) and int(length) >= 0:
            length = int(length)
        else:
            raise ValueError("ShiftRegisterSequence: the length must be a non-negative integer or All")
    feedback = None
    if _intq(spec):
        n = int(spec)
        if n < 1:
            raise ValueError("ShiftRegisterSequence: the register length must be positive")
        if p != 2:
            raise NotImplementedError("ShiftRegisterSequence[n, Modulus -> p] (p > 2) is not supported; give the "
                                      "polynomial")
        cs = [0] * (n + 1)
        cs[0] = cs[n] = 1
        for e in _default_lfsr_poly(n):
            cs[e] = 1
        form, taps = "Galois", None
    elif is_list(spec):
        if len(spec.args) not in (2, 3) or not _intq(spec.args[0]) or not is_list(spec.args[1]):
            raise ValueError("ShiftRegisterSequence: the register specification must be {n, {taps...}} or "
                             "{n, taps, f}")
        n = int(spec.args[0])
        taps = [int(t) for t in spec.args[1].args if _intq(t)]
        if len(taps) != len(spec.args[1].args) or any(not 1 <= t <= n for t in taps):
            raise ValueError("ShiftRegisterSequence: taps must be integers between 1 and n")
        feedback = spec.args[2] if len(spec.args) == 3 else None
        form = "Fibonacci"
        cs = [0] * (n + 1)
        cs[n] = 1
        for t in taps:
            cs[t - 1] = (cs[t - 1] + 1) % 2
    else:
        if p == 2 and isinstance(spec, (Expr, Symbol)):
            pass
        cs, _x = _poly_lowhigh(spec, p, "ShiftRegisterSequence")
        n = len(cs) - 1
        form, taps = "Galois", None
        if n < 1:
            raise ValueError("ShiftRegisterSequence: the polynomial must have positive degree")
    if method is not None:
        if method not in ("Fibonacci", "Galois"):
            raise ValueError("ShiftRegisterSequence: Method must be \"Fibonacci\" or \"Galois\"")
        if method != form:
            if feedback is not None:
                raise ValueError("ShiftRegisterSequence: a feedback function requires the Fibonacci method")
            if method == "Fibonacci":
                taps = [e + 1 for e in range(n) if cs[e] % p]
            form = method
    if form == "Fibonacci" and p != 2:
        raise NotImplementedError("ShiftRegisterSequence: the Fibonacci form is supported only modulo 2")
    # initial state
    if init is None:
        state = [1] + [0] * (n - 1)
    elif is_list(init):
        state = [int(v) % p if _intq(v) else None for v in init.args]
        if len(state) != n or None in state:
            raise ValueError(f"ShiftRegisterSequence: the initial state must be a list of {n} integers")
    else:
        x = _poly_var(init, "ShiftRegisterSequence") if not _intq(init) else sp.Symbol("x")
        P = sp.Poly(S(init), x)
        icoef = list(reversed([int(c) % p for c in P.all_coeffs()]))
        if len(icoef) > n:
            raise ValueError("ShiftRegisterSequence: the initial-state polynomial has too high a degree")
        state = icoef + [0] * (n - len(icoef))
    if form == "Fibonacci":
        if feedback is None:
            def step(s):
                return s[1:] + [sum(s[t - 1] for t in taps) % 2]
        else:
            def step(s):
                v = ev(K.call(feedback, L(*[s[t - 1] for t in taps])))
                if not _intq(v):
                    raise ValueError(f"ShiftRegisterSequence: the feedback function returned {v}, not an integer")
                return s[1:] + [int(v) % 2]
    else:
        def step(s):
            o = s[-1]
            s = [0] + s[:-1]
            if o:
                for t in range(n):
                    if cs[t]:
                        s[t] = (s[t] - o * cs[t]) % p
            return s
    if length is None:          # one full period of a maximal-length register (p^n - 1; 2^n with a feedback function)
        length = p ** n - 1 if feedback is None else 2 ** n
    return L(*_srs_run(step, state, length))


# ─────────────────────────── counting groups ───────────────────────────

def _holder_squarefree(n):
    ps = sp.primefactors(n)
    total = 0
    for k in range(len(ps) + 1):
        import itertools
        for d_primes in itertools.combinations(ps, k):
            d = math.prod(d_primes)
            m = n // d
            mprimes = [q for q in ps if m % q == 0]
            prod = 1
            for p in d_primes:
                c = sum(1 for q in mprimes if q % p == 1)
                prod *= (p ** c - 1) // (p - 1)
            total += prod
    return total


@fn("FiniteGroupCount", **ADDED, listable=True, notes="number of non-isomorphic groups of order n: tabulated "
    "for n <= 2047 (SmallGroups library, OEIS A000001); beyond the table exact formulas for cyclic numbers "
    "(gcd(n, EulerPhi[n]) = 1 gives 1), square-free n (Hoelder), p, p^2, p^3, p^4 and p^5 (p >= 5); other orders "
    "raise"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fgc(n: Expr) -> Expr:
    if not _intq(n):
        if is_numeric_value(n):
            raise ValueError("FiniteGroupCount: the argument must be a positive integer")
        raise Unevaluated()
    n = int(n)
    if n < 1:
        raise ValueError("FiniteGroupCount: the argument must be a positive integer")
    from ._w09_groupcounts import GROUP_COUNTS
    if n <= len(GROUP_COUNTS):
        return GROUP_COUNTS[n - 1]
    if math.gcd(n, int(sp.totient(n))) == 1:
        return 1
    f = sp.factorint(n)
    if all(e == 1 for e in f.values()):
        return _holder_squarefree(n)
    if len(f) == 1:
        (p, e), = f.items()
        if e <= 3:
            return [1, 2, 5][e - 1]
        if e == 4:
            return 14 if p == 2 else 15
        if e == 5 and p >= 5:
            return 61 + 2 * p + 2 * math.gcd(p - 1, 3) + math.gcd(p - 1, 4)
    raise NotImplementedError(f"FiniteGroupCount: the number of groups of order {n} is not available")


@fn("FiniteAbelianGroupCount", **ADDED, listable=True, notes="number of abelian groups of order n: the product "
    "of PartitionsP[e] over the prime-power exponents e of n"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fagc(n: Expr) -> Expr:
    if not _intq(n):
        if is_numeric_value(n):
            raise ValueError("FiniteAbelianGroupCount: the argument must be a positive integer")
        raise Unevaluated()
    n = int(n)
    if n < 1:
        raise ValueError("FiniteAbelianGroupCount: the argument must be a positive integer")
    return math.prod(int(sp.partition(e)) for e in sp.factorint(n).values())


# ─────────────────────────── Groupings ───────────────────────────

def _compositions(n, k, ok):
    """Compositions of n into k parts with every part satisfying ok, lexicographically descending."""
    if k == 1:
        if ok(n):
            yield (n,)
        return
    for first in range(n - (k - 1), 0, -1):
        if ok(first):
            for rest in _compositions(n - first, k - 1, ok):
                yield (first,) + rest


def _groupings(leaves, k, head, limit):
    n = len(leaves)
    if n == 1:
        return [leaves[0]]
    ok = lambda m: (m - 1) % (k - 1) == 0    # noqa: E731
    out = []
    for comp in _compositions(n, k, ok):
        parts, i = [], 0
        for c in comp:
            parts.append(leaves[i:i + c])
            i += c
        subs = [_groupings(p, k, head, limit) for p in parts]
        import itertools
        for combo in itertools.product(*subs):
            out.append(Expr(head, *combo))
            if len(out) > limit:
                raise ValueError("Groupings: too many groupings")
    return out


@fn("Groupings", **ADDED, notes="Groupings[list, k] (or Groupings[n, k] for Range[n]): all trees of k-ary "
    "List nodes with the given leaves in order (compositions of the leaves lexicographically descending); "
    "Groupings[list, h -> k] uses head h; Groupings[{list1, list2, ...}, spec] gives the groupings of every "
    "list, grouping structure outermost; a third argument wraps each result (e.g. HoldForm). Several heads or "
    "arities, arity 1 and Orderless/Flat specifications are not supported (raise)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _groupings_fn(lst: Expr, spec: Expr, wrapper: Fn = None) -> ListOf[Expr]:
    if _intq(lst):
        if int(lst) < 0:
            raise ValueError("Groupings: n must be non-negative")
        lists = [list(range(1, int(lst) + 1))]
    elif is_list(lst):
        if lst.args and all(is_list(a) for a in lst.args):
            lists = [list(a.args) for a in lst.args]
            if len({len(x) for x in lists}) != 1:
                raise ValueError("Groupings: the lists must have the same length")
        else:
            lists = [list(lst.args)]
    else:
        raise ValueError("Groupings: the first argument must be a list or a non-negative integer")
    head, k = "List", None
    if _intq(spec):
        k = int(spec)
    elif isinstance(spec, Expr) and spec.head in ("Rule", "RuleDelayed") and _intq(spec.args[1]) and \
            not is_list(spec.args[0]):
        head, k = spec.args[0], int(spec.args[1])
    else:
        raise NotImplementedError("Groupings: only a single arity k or a single head h -> k is supported")
    if k == 1:
        raise NotImplementedError("Groupings: arity 1 is not supported")
    if k < 1:
        raise ValueError("Groupings: the arity must be a positive integer")
    n = len(lists[0])
    if n == 0:
        return L()
    if (n - 1) % (k - 1):
        return L()
    shapes = _groupings(list(range(n)), k, "_shape", 10 ** 6)

    def build(shape, leaves):
        if isinstance(shape, int):
            return leaves[shape]
        kids = [build(s, leaves) for s in shape.args]
        return Expr(head, *kids) if isinstance(head, str) else K.call(head, *kids)
    out = []
    for sh in shapes:
        for lv in lists:
            e = build(sh, lv)
            out.append(K.call(wrapper, e) if wrapper is not None else e)
    return L(*out)


# ─────────────────────────── RandomComplex ───────────────────────────

def _complex_bounds(spec, fname):
    if is_list(spec):
        if len(spec.args) != 2:
            raise ValueError(f"{fname}: the range must be {{zmin, zmax}}")
        lo, hi = (complex(sp.N(S(v))) for v in spec.args)
    else:
        lo, hi = 0j, complex(sp.N(S(spec)))
    return lo, hi


@fn("RandomComplex", **ADDED, notes="RandomComplex[] in the unit square, RandomComplex[zmax] in the rectangle "
    "0..zmax, RandomComplex[{zmin, zmax}], with a count or dimension list; uniform real and imaginary parts "
    "from the session generator (SeedRandom reproducible); WorkingPrecision -> p gives p-digit numbers")
def _randcomplex(*args: Expr) -> Expr:
    pos, opts = split_options(args, ("WorkingPrecision",), "RandomComplex")
    if len(pos) > 2:
        raise ValueError("RandomComplex takes at most 2 arguments")
    spec = pos[0] if pos else L(0, Expr("Plus", 1, Symbol("I")))
    if not pos:
        lo, hi = 0j, 1 + 1j
    else:
        if not is_numeric_value(spec) and not is_list(spec):
            raise ValueError("RandomComplex: the range must be numeric")
        lo, hi = _complex_bounds(spec, "RandomComplex")
    wp = opts.get("WorkingPrecision")
    prec = None
    if wp is not None and not (isinstance(wp, Symbol) and wp.name == "MachinePrecision"):
        if not is_numeric_value(wp):
            raise ValueError("RandomComplex: WorkingPrecision must be a positive number")
        prec = int(sp.ceiling(S(wp)))
    from .statistics import _RNG as rng

    def one():
        a, b = rng.random(), rng.random()
        if prec is None:
            return complex(lo.real + a * (hi.real - lo.real), lo.imag + b * (hi.imag - lo.imag))
        z = (sp.Float(lo.real, prec + 5) + sp.Float(a, prec + 5) * sp.Float(hi.real - lo.real, prec + 5)) + \
            sp.I * (sp.Float(lo.imag, prec + 5) + sp.Float(b, prec + 5) * sp.Float(hi.imag - lo.imag, prec + 5))
        return R(sp.N(z, prec))
    if len(pos) < 2:
        return one()
    dims = pos[1]
    if _intq(dims):
        dims = [int(dims)]
    elif is_list(dims) and all(_intq(d) for d in dims.args):
        dims = [int(d) for d in dims.args]
    else:
        raise ValueError("RandomComplex: the count must be a non-negative integer or a list of them")
    if any(d < 0 for d in dims):
        raise ValueError("RandomComplex: dimensions must be non-negative")

    def build(ds):
        if not ds:
            return one()
        return L(*[build(ds[1:]) for _ in range(ds[0])])
    return build(dims)


# ─────────────────────────── Dirichlet characters and series ───────────────────────────

def _char_table(k, j):
    """{residue: exponent r} with chi(n) = Exp[2 Pi I r] for n coprime to k (r a Rational in [0, 1))."""
    if k == 1:
        if j != 1:
            raise ValueError("DirichletCharacter: the index must be 1 for modulus 1")
        return {0: sp.Integer(0)}
    if k == 2:
        if j != 1:
            raise ValueError("DirichletCharacter: the index must be 1 for modulus 2")
        return {1: sp.Integer(0)}
    if k == 4:
        if j not in (1, 2):
            raise ValueError("DirichletCharacter: the index must be 1 or 2 for modulus 4")
        return {1: sp.Integer(0), 3: sp.Integer(0) if j == 1 else sp.Rational(1, 2)}
    if sp.isprime(k):
        if not 1 <= j <= k - 1:
            raise ValueError(f"DirichletCharacter: the index must be between 1 and {k - 1}")
        g = int(sp.primitive_root(k))
        tab, x = {}, 1
        for a in range(k - 1):
            tab[x] = sp.Rational((j - 1) * a, k - 1) % 1
            x = x * g % k
        return tab
    raise NotImplementedError(f"DirichletCharacter: modulus {k} is not supported (only 1, 2, 4 and primes; the "
                              "Wolfram Language's ordering of characters for other moduli is not documented)")


def _char_value(r):
    if r == 0:
        return 1
    if r == sp.Rational(1, 2):
        return -1
    return R(sp.exp(2 * sp.pi * sp.I * r))


def _kj(k, j, fname):
    if not (_intq(k) and _intq(j)):
        raise Unevaluated()
    k, j = int(k), int(j)
    if k < 1:
        raise ValueError(f"{fname}: the modulus must be a positive integer")
    return k, j


@fn("DirichletCharacter", **ADDED, notes="DirichletCharacter[k, j, n]: the j-th Dirichlet character modulo k "
    "(j = 1 is the principal character); for a prime modulus chi_j(g^a) = Exp[2 Pi I (j - 1) a/(k - 1)] with g "
    "the least primitive root (the Wolfram ordering); moduli 1, 2, 4 and primes are supported, other moduli "
    "raise; zero when n is not coprime to k; listable in n"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _dirichletchar(k: Expr, j: Expr, n: Expr) -> Expr:
    k, j = _kj(k, j, "DirichletCharacter")
    if is_list(n):
        return L(*[_dirichletchar(k, j, x) for x in n.args])
    tab = _char_table(k, j)
    if not _intq(n):
        if is_numeric_value(n):
            raise ValueError("DirichletCharacter: n must be an integer")
        raise Unevaluated()
    m = int(n) % k
    if math.gcd(m, k) != 1 and k > 1:
        return 0
    return _char_value(tab[m % k if k > 1 else 0])


def _gen_bernoulli(k, tab, m):
    """B_{m, chi} = k^(m-1) Sum[chi(a) B_m(a/k), {a, 1, k}]."""
    x = sp.Symbol("x")
    bm = sp.bernoulli(m, x)
    tot = 0
    for a in range(1, k + 1):
        if a % k in tab or (k == 1):
            r = tab[a % k] if k > 1 else 0
            tot += sp.exp(2 * sp.pi * sp.I * r) * bm.subs(x, sp.Rational(a, k))
    return sp.nsimplify(sp.expand(k ** (m - 1) * tot)) if False else sp.expand(k ** (m - 1) * tot)


def _dirichlet_numeric(k, j, s, prec):
    import mpmath
    tab = _char_table(k, j)
    with mpmath.workdps(prec + 10):
        chi = []
        for a in range(k):
            if k == 1:
                chi.append(1)
            elif a in tab:
                chi.append(mpmath.expjpi(2 * mpmath.mpf(tab[a].p) / tab[a].q))
            else:
                chi.append(0)
        return mpmath.dirichlet(s, chi)


@fn("DirichletL", **ADDED, listable=True, notes="DirichletL[k, j, s]: the Dirichlet L-function of "
    "DirichletCharacter[k, j, n]; DirichletL[1, 1, s] = Zeta[s]; the principal character gives Zeta[s] "
    "Product[1 - p^-s, {p | k}]; non-positive integers s give -B[1 - s, chi]/(1 - s) (generalized Bernoulli "
    "numbers); inexact s is evaluated with mpmath; other exact arguments stay unevaluated (N evaluates them)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _dirichletl(k: Expr, j: Expr, s: Expr) -> Expr:
    k, j = _kj(k, j, "DirichletL")
    tab = _char_table(k, j)
    if k == 1:
        return ev(Expr("Zeta", s))
    if is_numeric_value(s) and has_float(s):
        return numeric_eval(lambda mp_s, dps: _dirichlet_numeric(k, j, mp_s, dps), s)
    principal = all(r == 0 for r in tab.values())
    if principal:
        fac = Expr("Times", *[Expr("Plus", 1, Expr("Times", -1, Expr("Power", int(p), Expr("Times", -1, s))))
                              for p in sp.primefactors(k)])
        return ev(Expr("Times", Expr("Zeta", s), fac))
    if _intq(s) and int(s) <= 0:
        m = 1 - int(s)
        return R(sp.simplify(-_gen_bernoulli(k, tab, m) / m))
    raise Unevaluated()


def mp_to_value(v, dps):
    """mpmath number -> Mikoshi number (machine for dps <= 15, else a SymPy Float of dps digits)."""
    import mpmath
    if isinstance(v, mpmath.mpc):
        if v.imag == 0:
            v = v.real
        else:
            if dps <= 15:
                return complex(v)
            return R(sp.Float(v.real, dps) + sp.I * sp.Float(v.imag, dps))
    if dps <= 15:
        return float(v)
    return sp.Float(v, dps)


def numeric_eval(f, *xs):
    """Evaluate f(*mp_args, dps) at the precision of the inexact inputs xs and convert back."""
    import mpmath
    from ..precision import input_precision
    pin = input_precision(list(xs))
    dps = 15 if pin in (None, "m") else int(mpmath.libmp.prec_to_dps(pin))
    with mpmath.workdps(dps + 10):
        args = []
        for x in xs:
            v = S(x)
            args.append(mpmath.mpmathify(complex(v)) if dps <= 15 else
                        mpmath.mpc(*[mpmath.mpf(str(sp.N(t, dps + 5))) for t in v.as_real_imag()]))
        args = [a.real if isinstance(a, mpmath.mpc) and a.imag == 0 else a for a in args]
        out = f(*args, dps)
    return mp_to_value(out, dps)


def _as_int(x):
    v = S(x)
    r = int(round(float(v))) if v.is_number and v.is_real else None
    if r is None or abs(v - r) > 1e-9:
        raise Unevaluated()
    return r


def _mp_dirichletl(k, j, s):
    k, j = _as_int(k), _as_int(j)
    sv = s if not isinstance(s, (int, sp.Integer)) else sp.Float(s)
    return S(numeric_eval(lambda mp_s, dps: _dirichlet_numeric(k, j, mp_s, dps), R(sv) if isinstance(sv, sp.Basic)
                          else sv))


def _register_numeric():
    from .wl_special import MP_NUMERIC
    MP_NUMERIC["DirichletL"] = _mp_dirichletl


_register_numeric()


def _sigma(k, m):
    return Expr("DivisorSigma", k, m)


def _poly_terms(f, n):
    """[(coeff, exponent)] when f = Sum c n^a with c free of n, else None."""
    e = sp.expand(S(f))
    nn = S(n)
    out = []
    for t in sp.Add.make_args(e):
        c, a = sp.S(1), sp.S(0)
        for fac in sp.Mul.make_args(t):
            if fac == nn:
                a += 1
            elif fac.is_Pow and fac.base == nn and not fac.exp.has(nn):
                a += fac.exp
            elif fac.has(nn):
                return None
            else:
                c *= fac
        out.append((c, a))
    return out


def _is_delta1(f, n):
    """Is f KroneckerDelta[n - 1] (in any of its evaluated forms)?"""
    if not (isinstance(f, Expr) and f.head == "KroneckerDelta" and len(f.args) in (1, 2)):
        return False
    d = S(f.args[0]) if len(f.args) == 1 else S(f.args[1]) - S(f.args[0])
    return sp.expand(d - (S(n) - 1)) == 0 or sp.expand(d + (S(n) - 1)) == 0


@fn("DirichletConvolve", **ADDED, notes="DirichletConvolve[f, g, n, m] = Sum[f(d) g(m/d), {d | m}]: explicit for "
    "a positive integer m; for symbolic m, sums of powers of n give DivisorSigma closed forms (n^a * n^b -> "
    "m^Min[a,b] DivisorSigma[|a - b|, m]), KroneckerDelta[n - 1] is the identity, and other inputs stay "
    "unevaluated"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _dirconv(f: Expr, g: Expr, n: Expr, m: Expr) -> Expr:
    if not isinstance(n, Symbol):
        raise ValueError("DirichletConvolve: the variable must be a symbol")
    if _intq(m):
        mm = int(m)
        if mm < 1:
            raise ValueError("DirichletConvolve: m must be a positive integer")
        terms = [Expr("Times", K.substitute(f, {n.name: d}), K.substitute(g, {n.name: mm // d}))
                 for d in sp.divisors(mm)]
        return ev(Expr("Plus", *terms))
    if is_numeric_value(m):
        raise ValueError("DirichletConvolve: m must be a positive integer")
    if _is_delta1(f, n):
        return ev(K.substitute(g, {n.name: m}))
    if _is_delta1(g, n):
        return ev(K.substitute(f, {n.name: m}))
    try:
        pf, pg = _poly_terms(f, n), _poly_terms(g, n)
    except Exception:
        pf = pg = None
    if pf is None or pg is None:
        raise Unevaluated()
    mm = S(m)
    tot = 0
    for c1, a in pf:
        for c2, b in pg:
            if not (a - b).is_number:
                raise Unevaluated()
            lo, d = (b, a - b) if a >= b else (a, b - a)
            tot += c1 * c2 * mm ** lo * sp.Function("DivisorSigma")(d, mm)
    return ev(R(sp.expand(tot)))


_PL = sp.Function("PolyLog")


def _dt_term(t, n, s):
    """Transform of one product term (no sums), or None."""
    nn = S(n)
    if not t.has(nn):
        return t * sp.zeta(s)
    c = sp.S(1)
    rest = []
    a = sp.S(0)
    for fac in sp.Mul.make_args(t):
        if not fac.has(nn):
            c *= fac
        elif fac == nn:
            a += 1
        elif fac.is_Pow and fac.base == nn and not fac.exp.has(nn):
            a += fac.exp
        else:
            rest.append(fac)
    ss = s - a
    if not rest:
        return c * sp.zeta(ss)
    if len(rest) != 1:
        return None
    r = rest[0]
    if r.is_Pow and not r.base.has(nn) and r.exp == nn:
        return c * _PL(ss, r.base)
    if r.is_Pow and not r.base.has(nn) and r.exp == -nn:
        return c * _PL(ss, 1 / r.base)
    if isinstance(r, sp.exp) and (r.args[0] / nn).is_number is False and False:
        return None
    if isinstance(r, sp.exp):
        q = sp.simplify(r.args[0] / nn)
        if not q.has(nn):
            return c * _PL(ss, sp.exp(q))
    if isinstance(r, sp.cos):
        q = sp.simplify(r.args[0] / nn)
        if not q.has(nn):
            return c * (_PL(ss, sp.exp(-sp.I * q)) + _PL(ss, sp.exp(sp.I * q))) / 2
    if isinstance(r, sp.sin):
        q = sp.simplify(r.args[0] / nn)
        if not q.has(nn):
            return c * sp.I * (_PL(ss, sp.exp(-sp.I * q)) - _PL(ss, sp.exp(sp.I * q))) / 2
    if isinstance(r, sp.Mod) and r.args[0] == nn and r.args[1].is_Integer and r.args[1] > 0:
        k = int(r.args[1])
        return c * k ** (-ss) * sp.Add(*[rr * sp.zeta(ss, sp.Rational(rr, k)) for rr in range(1, k)])
    return None


_DT_HEADS = {
    "MoebiusMu": lambda s: Expr("Power", Expr("Zeta", s), -1),
    "EulerPhi": lambda s: Expr("Times", Expr("Zeta", Expr("Plus", s, -1)), Expr("Power", Expr("Zeta", s), -1)),
    "LiouvilleLambda": lambda s: Expr("Times", Expr("Zeta", Expr("Times", 2, s)), Expr("Power", Expr("Zeta", s),
                                                                                        -1)),
}


def _dt(f, n, s):
    """DirichletTransform of f (Mikoshi expr) or None."""
    if isinstance(f, Expr) and f.head == "Plus":
        parts = [_dt(a, n, s) for a in f.args]
        return None if any(p is None for p in parts) else Expr("Plus", *parts)
    if isinstance(f, Expr) and f.head == "Times":
        consts = [a for a in f.args if _free_of(a, n)]
        others = [a for a in f.args if not _free_of(a, n)]
        if consts and len(others) == 1:
            r = _dt(others[0], n, s)
            return None if r is None else Expr("Times", *consts, r)
    if isinstance(f, Expr) and isinstance(f.head, str):
        h = f.head
        if h in _DT_HEADS and len(f.args) == 1 and K.same(f.args[0], n):
            return _DT_HEADS[h](s)
        if h == "DivisorSigma" and len(f.args) == 2 and K.same(f.args[1], n) and _free_of(f.args[0], n):
            return Expr("Times", Expr("Zeta", s), Expr("Zeta", Expr("Plus", s, Expr("Times", -1, f.args[0]))))
        if h == "DirichletCharacter" and len(f.args) == 3 and K.same(f.args[2], n):
            return Expr("DirichletL", f.args[0], f.args[1], s)
        if h == "KroneckerDelta" and _is_delta1(f, n):
            return 1
        if h == "DivisorSum" and len(f.args) == 2 and K.same(f.args[0], n):
            inner = ev(K.call(f.args[1], n))
            r = _dt(inner, n, s)
            if r is None:
                r = Expr("DirichletTransform", inner, n, s)
            return Expr("Times", Expr("Zeta", s), r)
    try:
        e = S(f)
    except Exception:
        return None
    if any(isinstance(a, sp.core.function.AppliedUndef) for a in sp.preorder_traversal(e)):
        return None
    out = 0
    for t in sp.Add.make_args(sp.expand(e)):
        r = _dt_term(t, n, S(s))
        if r is None:
            return None
        out += r
    return R(out)


@fn("DirichletTransform", **ADDED, notes="DirichletTransform[f, n, s] = Sum[f(n)/n^s, {n, 1, Infinity}] in "
    "closed form for linear combinations of powers of n, a^n (PolyLog), Cos/Sin/Exp of multiples of n, Mod[n, "
    "k] (Hurwitz zeta), MoebiusMu, EulerPhi, LiouvilleLambda, DivisorSigma, DirichletCharacter (DirichletL), "
    "KroneckerDelta[n - 1] and DivisorSum[n, f] (Zeta[s] times the transform of f); other functions stay "
    "unevaluated"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _dirtransform(f: Expr, n: Expr, s: Expr, *opts: Option) -> Expr:
    if not isinstance(n, Symbol):
        raise ValueError("DirichletTransform: the variable must be a symbol")
    r = _dt(f, n, s)
    if r is None:
        raise Unevaluated()
    return ev(r)


# ─────────────────────────── integer relations and lattices ───────────────────────────

def _is_exact_vec(items):
    return not any(has_float(v) for v in items)


@fn("FindIntegerNullVector", **ADDED, notes="FindIntegerNullVector[{x1, ..., xn}] finds integers ai, not all "
    "zero, with a1 x1 + ... + an xn == 0 (PSLQ at the input precision, or at 3 n + 30 digits for exact input); "
    "FindIntegerNullVector[x, d] restricts to vectors of norm <= d and raises when none exists; for exact input "
    "the relation is verified to 60 more digits")
def _findintnull(xs: Expr, d: Expr = None, *opts: Option) -> ListOf[Expr]:
    import mpmath
    if not is_list(xs) or len(xs.args) < 2:
        raise ValueError("FindIntegerNullVector: the argument must be a list of at least two numbers")
    pos, o = split_options((d,) + opts if d is not None else opts, ("WorkingPrecision", "ZeroTest"),
                           "FindIntegerNullVector")
    d = pos[0] if pos else None
    items = list(xs.args)
    for v in items:
        if not is_numeric_value(v):
            raise ValueError("FindIntegerNullVector: the elements must be numbers")
    vals = [S(v) for v in items]
    exact = _is_exact_vec(items)
    wp = o.get("WorkingPrecision")
    if exact:
        dps = int(S(wp)) if wp is not None else 3 * len(items) + 40
    else:
        precs = [x._prec for v in vals for x in sp.preorder_traversal(v) if isinstance(x, sp.Float)]
        dps = int(mpmath.libmp.prec_to_dps(min(precs))) if precs else 15
        if all(not isinstance(v, sp.Float) or v._prec <= 53 for v in vals):
            dps = 15
    maxcoeff = int(sp.floor(S(d))) if d is not None else 10 ** max(3, dps // len(items))
    has_complex = any(complex(sp.N(v)).imag != 0 for v in vals)
    with mpmath.workdps(dps):
        if has_complex:
            rows = [mpmath.mpmathify(complex(sp.N(v, dps))) if dps <= 15 else mpmath.mpc(
                *[mpmath.mpf(str(sp.N(p, dps))) for p in v.as_real_imag()]) for v in vals]
            rel = _pslq_complex(rows, maxcoeff, dps)
        else:
            rows = [mpmath.mpf(str(sp.N(v, dps + 5))) for v in vals]
            tol = mpmath.mpf(10) ** (-(dps - max(3, dps // 5)))
            rel = mpmath.pslq(rows, tol=tol, maxcoeff=maxcoeff, maxsteps=10 ** 5)
    if rel is None:
        raise ValueError("FindIntegerNullVector: no integer null vector found" +
                         (f" with norm <= {d}" if d is not None else ""))
    if d is not None and math.sqrt(sum(r * r for r in rel)) > float(S(d)):
        raise ValueError(f"FindIntegerNullVector: no integer null vector with norm <= {d}")
    if exact:
        tot = sum(r * v for r, v in zip(rel, vals))
        z = sp.N(tot, dps + 60)
        if abs(z) > sp.Float(10) ** (-(dps + 30)) and not sp.simplify(tot) == 0:
            raise ValueError("FindIntegerNullVector: no exact integer relation found (the candidate fails a "
                             "high-precision check)")
    return L(*[int(r) for r in rel])


def _pslq_complex(zs, maxcoeff, dps):
    import mpmath
    # real relation on (Re, Im) stacked with a large weight
    n = len(zs)
    w = mpmath.mpf(10) ** (dps // 2)
    rows = [[mpmath.mpf(1) if i == j else mpmath.mpf(0) for j in range(n)] + [w * mpmath.re(zs[i]),
                                                                                w * mpmath.im(zs[i])]
            for i in range(n)]
    M = sp.Matrix([[int(mpmath.nint(x)) for x in r] for r in rows])
    red = M.lll() if hasattr(M, "lll") else None
    if red is None:
        from sympy.polys.matrices import DomainMatrix
        red = DomainMatrix.from_Matrix(M).convert_to(sp.ZZ).lll().to_Matrix()
    for i in range(red.rows):
        v = [int(red[i, j]) for j in range(n)]
        if any(v) and red[i, n] == 0 and red[i, n + 1] == 0 and max(abs(x) for x in v) <= maxcoeff:
            return v
    return None


@fn("LatticeReduce", **ADDED, notes="LatticeReduce[{v1, v2, ...}] gives an LLL-reduced basis (delta = 3/4) of "
    "the lattice spanned by the integer (or rational) vectors vi; linearly dependent inputs drop to the rank; "
    "approximate vectors raise"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _latticereduce(m: Expr) -> ListOf[Expr]:
    if not is_list(m) or not m.args or not all(is_list(r) for r in m.args):
        raise ValueError("LatticeReduce: the argument must be a list of vectors")
    rows = []
    for r in m.args:
        row = []
        for v in r.args:
            sv = S(v) if not isinstance(v, (int, float)) else v
            if isinstance(v, float) or (isinstance(sv, sp.Basic) and not sv.is_Rational):
                raise ValueError("LatticeReduce: the vectors must have integer or rational entries")
            row.append(sp.Rational(sv))
        rows.append(row)
    if len({len(r) for r in rows}) != 1:
        raise ValueError("LatticeReduce: the vectors must have the same length")
    den = sp.ilcm(*[x.q for r in rows for x in r])
    from sympy.polys.matrices import DomainMatrix
    M = sp.Matrix([[int(x * den) for x in r] for r in rows])
    # remove dependencies with an HNF-free approach: LLL on full-rank input; for dependent rows use the
    # rational row space basis first
    rank = M.rank()
    if rank < M.rows:
        M = _integer_row_basis(M)
    red = DomainMatrix.from_Matrix(M).convert_to(sp.ZZ).lll().to_Matrix()
    return L(*[L(*[R(sp.Rational(int(red[i, j]), den)) for j in range(red.cols)]) for i in range(red.rows)])


def _integer_row_basis(M):
    """A basis of the integer row lattice of M (via the Hermite normal form)."""
    from sympy.matrices.normalforms import hermite_normal_form
    H = hermite_normal_form(M.T)  # columns of H span the column lattice of M.T, i.e. the row lattice of M
    rows = [r for r in (H.col(j).T for j in range(H.cols)) if any(r)]
    B = sp.Matrix.vstack(*rows) if rows else sp.zeros(0, M.cols)
    if B.rank() != M.rank():
        raise ValueError("LatticeReduce: could not compute a basis of the lattice")
    return B
