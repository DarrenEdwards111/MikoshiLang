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

    def pre(*args):
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
    "a real number. MikoshiLang's arbitrary-precision numbers never overflow, so it only arises when entered")
def _overflow(*a):
    if a:
        raise ValueError("Overflow takes no arguments")
    raise Unevaluated()


@fn("Underflow", **ADDED, notes="Underflow[] represents a number too small to represent; x + Underflow[] = x for "
    "numbers x != 0 (as a machine real), c Underflow[] = Sign[c] Underflow[], 1/Underflow[] = Overflow[]")
def _underflow(*a):
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
    "Sqrt[2] and symbols")
def _rvnq(x):
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
    "120 digits); False for complex numbers with a zero inexact imaginary part, Infinity and non-numeric input")
def _rvnumq(x):
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
    "unevaluated")
def _realsign(x):
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
    "accuracy of a machine zero); symbolic arguments stay unevaluated")
def _realexp(x, b=10):
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
    "the sign of x; complex numbers are split componentwise); symbolic arguments stay unevaluated")
def _mixedfrac(x):
    if not is_numeric_value(x):
        raise Unevaluated()
    ip = ev(Expr("IntegerPart", x))
    fp = ev(Expr("FractionalPart", x))
    if isinstance(ip, Expr) and ip.head == "IntegerPart":
        raise Unevaluated()
    return L(ip, fp)


@fn("NumeratorDenominator", **ADDED, notes="{Numerator[expr], Denominator[expr]}; lists give the pair of lists. "
    "The Modulus and Trig options are not supported (raise)")
def _numden(x, *opts):
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
    "fractional; exact inputs give exact results")
def _numdecomp(x, units):
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
    "coefficients are aligned with the last k units (k <= n)")
def _numcompose(cs, units):
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
def _mixedradix(bs):
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


wrap_builtin("Det", _det_pre)
wrap_builtin("Inverse", _inverse_pre)
