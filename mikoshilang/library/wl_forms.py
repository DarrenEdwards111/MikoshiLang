"""Coverage batch W04: output forms, number formatting, layout wrappers and typesetting (Wolfram semantics).

Display wrappers stay inert, as in Wolfram: ``NumberForm[1.23456, 3]`` evaluates to itself (its arguments are
evaluated) and only changes how the expression is *shown*.  The text a wrapper shows as is what
``ToString[...]`` (and the REPL) produce, via :func:`output_text`:

* number forms (NumberForm, ScientificForm, EngineeringForm, AccountingForm, PaddedForm, DecimalForm,
  PercentForm, BaseForm) follow the documented option semantics (DigitBlock, ExponentFunction, ExponentStep,
  NumberFormat, NumberMultiplier, NumberPadding, NumberPoint, NumberSeparator, NumberSigns,
  ScientificNotationThreshold, SignPadding, DefaultPrintPrecision).  Mantissas are rounded half-up from the exact
  binary value; machine numbers show at most 16 digits, trailing zeros of machine numbers are dropped, and exponents
  are written on one line as ``1.25×10^8`` (Wolfram's two-dimensional OutputForm puts the exponent on a separate
  line; MikoshiLang's text output is always one line per row);
* TeXForm, CForm, FortranForm and MathMLForm produce the corresponding source text; TraditionalForm, StandardForm
  and OutputForm produce linear text;
* layout wrappers (Row, Column, Grid, TableForm, Multicolumn, Labeled, Framed, Overlay, Style, Text, Item, Placed,
  TextElement) render as aligned plain-text blocks.  Grid and TableForm use Wolfram's OutputForm layout: columns
  separated by three spaces and rows by a blank line, left aligned;
* StringForm substitutes its arguments; Short/Skeleton/StringSkeleton elide long output as ``<<k>>``;
* TextString gives a human-readable string (decimal numbers, ListFormat, MissingString, BooleanStrings).

Option and position names (Left, Top, Alignment, ColumnSpacings, HeaderStyle, ...) are inert symbols, as in
Wolfram.  Those that affect text layout are honoured by the wrappers above; the rest (front-end or Dataset styling
options such as Background, FontFamily, HeaderBackground) are accepted, validated where Wolfram documents a value
domain, and have no effect on plain text.
"""
from __future__ import annotations

import math
import re
from decimal import Decimal, localcontext, ROUND_HALF_UP
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import formatting as F
from .core import Unevaluated, fn, record, L, is_list
from .wl_common import ev, is_sym

AREA = "output forms & formatting (W04)"
INF = float("inf")


def _sym(name):
    return Symbol(name)


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def _opt_name(x):
    if _is_rule(x):
        k = x.args[0]
        if isinstance(k, Symbol):
            return k.name
        if isinstance(k, str):
            return k
    return None


def split_opts(args, allowed, fname, strict=True):
    """Trailing options -> dict; unknown option names raise (a clear error rather than silently ignored)."""
    args = list(args)
    opts = {}
    while args and _opt_name(args[-1]) is not None:
        r = args.pop()
        name = _opt_name(r)
        if name not in allowed:
            if strict:
                raise ValueError(f"{fname}: unknown option {name}")
            continue
        opts.setdefault(name, r.args[1])
    return args, opts


def _is_real(x):
    return (isinstance(x, (float, sp.Float)) and not isinstance(x, bool))


def _is_int(x):
    return F.is_int(x)


def _is_bignum(x):
    return isinstance(x, sp.Float) and x._prec > 53


def _dps(x):
    import mpmath
    return int(mpmath.libmp.prec_to_dps(x._prec))


def _exact_decimal(x):
    """Exact decimal value of a float or SymPy Float (as Decimal)."""
    if isinstance(x, float):
        return Decimal(x)
    sign, man, exp, bc = x._mpf_ if x._mpf_[1] or x._mpf_ != (0, 0, 0, 0) else (0, 0, 0, 0)
    if man == 0:
        return Decimal(0)
    with localcontext() as c:
        c.prec = max(60, bc + abs(exp) + 10)
        d = Decimal(man) * (Decimal(2) ** exp)
    return -d if sign else d


def _num_value(x):
    """Python float, int, or None for machine/exact numbers usable in comparisons."""
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return x
    if isinstance(x, (sp.Integer,)):
        return int(x)
    if isinstance(x, sp.Float):
        return float(x)
    return None


def _str_arg(x, fname, what):
    if not isinstance(x, str):
        raise ValueError(f"{fname}: value for option {what} -> {F.input_form(x)} is expected to be a string")
    return x


# ═════════════════════════════════════ number formatting ═════════════════════════════════════

NUMBER_FORMS = ("NumberForm", "ScientificForm", "EngineeringForm", "AccountingForm", "PaddedForm",
                "DecimalForm", "PercentForm")
NUMBER_OPTIONS = ("DefaultPrintPrecision", "DigitBlock", "ExponentFunction", "ExponentStep", "NumberFormat",
                  "NumberMultiplier", "NumberPadding", "NumberPoint", "NumberSeparator", "NumberSigns",
                  "ScientificNotationThreshold", "SignPadding")

_DEFAULTS = {
    "DefaultPrintPrecision": Symbol("Automatic"), "DigitBlock": INF, "ExponentFunction": Symbol("Automatic"),
    "ExponentStep": 1, "NumberFormat": Symbol("Automatic"), "NumberMultiplier": "\u00d7",
    "NumberPadding": L("", "0"), "NumberPoint": ".", "NumberSeparator": L(",", " "),
    "NumberSigns": L("-", ""), "ScientificNotationThreshold": L(-5, 6), "SignPadding": False,
}
FORM_DEFAULTS = {
    "NumberForm": {},
    "ScientificForm": {"ExponentFunction": "identity"},
    "EngineeringForm": {"ExponentFunction": "identity", "ExponentStep": 3},
    "AccountingForm": {"ExponentFunction": "none", "NumberSigns": L(L("(", ")"), "")},
    "PaddedForm": {"NumberPadding": L(" ", "0")},
    "DecimalForm": {"ExponentFunction": "none"},
    "PercentForm": {},
}


def _pair(v, fname, name, kind):
    if isinstance(v, str):
        return (v, v)
    if is_list(v) and len(v.args) == 2 and all(isinstance(t, str) for t in v.args):
        return tuple(v.args)
    raise ValueError(f"{fname}: value for option {name} -> {F.input_form(v)} should be {kind}")


class NumOpts:
    def __init__(self, form, opts, fname):
        o = dict(_DEFAULTS)
        o.update({k: v for k, v in FORM_DEFAULTS[form].items()})
        o.update(opts)
        self.form = form
        dpp = o["DefaultPrintPrecision"]
        if is_sym(dpp, "Automatic"):
            self.dpp = 6
        elif _is_int(dpp) and int(dpp) > 0:
            self.dpp = int(dpp)
        else:
            raise ValueError(f"{fname}: DefaultPrintPrecision must be a positive integer or Automatic")
        db = o["DigitBlock"]
        self.block = (self._block(db, fname))
        ef = o["ExponentFunction"]
        self.ef = ef
        st = o["ExponentStep"]
        if not _is_int(st) or int(st) <= 0:
            raise ValueError(f"{fname}: value of option ExponentStep -> {F.input_form(st)} is not a positive integer")
        self.step = int(st)
        self.nformat = o["NumberFormat"]
        self.mult = _str_arg(o["NumberMultiplier"], fname, "NumberMultiplier")
        self.pad = _pair(o["NumberPadding"], fname, "NumberPadding", "a string or a pair of strings")
        self.point = _str_arg(o["NumberPoint"], fname, "NumberPoint")
        self.sep = _pair(o["NumberSeparator"], fname, "NumberSeparator", "a string or a pair of strings")
        ns = o["NumberSigns"]
        if is_list(ns) and len(ns.args) == 2:
            def part(p):
                if isinstance(p, str):
                    return (p, "")
                if is_list(p) and len(p.args) == 2 and all(isinstance(t, str) for t in p.args):
                    return tuple(p.args)
                raise ValueError(f"{fname}: value for option NumberSigns should be a pair of strings or two pairs "
                                 "of strings")
            self.signs = (part(ns.args[0]), part(ns.args[1]))
        else:
            raise ValueError(f"{fname}: value for option NumberSigns -> {F.input_form(ns)} should be a pair of "
                             "strings or two pairs of strings")
        th = o["ScientificNotationThreshold"]
        if not (is_list(th) and len(th.args) == 2 and all(_is_int(t) or t == INF or t == -INF or
                                                            is_sym(t, "Infinity") for t in th.args)):
            if not (is_list(th) and len(th.args) == 2):
                raise ValueError(f"{fname}: ScientificNotationThreshold must be a pair {{lo, hi}} of integers")
        self.thresh = tuple(self._thr(t, fname) for t in th.args)
        sp_ = o["SignPadding"]
        if sp_ is not True and sp_ is not False:
            raise ValueError(f"{fname}: SignPadding must be True or False")
        self.sign_padding = sp_

    @staticmethod
    def _thr(t, fname):
        if _is_int(t):
            return int(t)
        if t in (INF, -INF):
            return t
        if is_sym(t, "Infinity"):
            return INF
        if isinstance(t, Expr) and t.head == "Times" and t.args and t.args[0] == -1 and is_sym(t.args[-1], "Infinity"):
            return -INF
        if isinstance(t, Expr) and t.head == "DirectedInfinity" and t.args == (-1,):
            return -INF
        raise ValueError(f"{fname}: ScientificNotationThreshold must be a pair {{lo, hi}} of integers")

    @staticmethod
    def _block(db, fname):
        def one(v):
            if v == INF or is_sym(v, "Infinity"):
                return None
            if _is_int(v) and int(v) > 0:
                return int(v)
            raise ValueError(f"{fname}: value for option DigitBlock should be a positive integer, Infinity, or a "
                             "pair of positive integers")
        if is_list(db):
            if len(db.args) != 2:
                raise ValueError(f"{fname}: value for option DigitBlock should be a positive integer, Infinity, "
                                 "or a pair of positive integers")
            return (one(db.args[0]), one(db.args[1]))
        b = one(db)
        return (b, b)

    def exponent(self, e):
        """Exponent to factor out for a number whose leading digit has decimal exponent e (None: no exponent)."""
        r = (e // self.step) * self.step
        ef = self.ef
        if ef == "identity":
            return r
        if ef == "none":
            return None
        if is_sym(ef, "Automatic"):
            lo, hi = self.thresh
            return None if lo <= r < hi else r
        v = ev(Expr(ef, r) if isinstance(ef, Symbol) else K.call(ef, r))
        if v is None or is_sym(v, "Null"):
            return None
        if not _is_int(v):
            raise ValueError(f"ExponentFunction: {F.input_form(ef)} returned {F.input_form(v)}, not an integer "
                             "or Null")
        return int(v)


def _round_sig(d, n):
    """Round positive Decimal d half-up to n significant digits (Decimal with that many digits)."""
    with localcontext() as c:
        c.prec = max(80, n + 40)
        e = d.adjusted()
        r = d.quantize(Decimal(1).scaleb(e - n + 1), rounding=ROUND_HALF_UP)
        if r.adjusted() > e:          # 9.99 -> 10.0: keep n digits
            r = d.quantize(Decimal(1).scaleb(e - n + 2), rounding=ROUND_HALF_UP)
        return r


def _quant(d, decimals):
    with localcontext() as c:
        c.prec = max(80, abs(decimals) + d.adjusted() + 40)
        return d.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def _blocks(s, size, sep, from_left):
    if not size or not sep or len(s) <= size:
        return s
    if from_left:
        return sep.join(s[i:i + size] for i in range(0, len(s), size))
    head = len(s) % size
    parts = ([s[:head]] if head else []) + [s[i:i + size] for i in range(head, len(s), size)]
    return sep.join(parts)


def format_number(x, n, f, o: NumOpts, percent=False):
    """Text of a real or integer x under a number form (n significant digits or None, f decimals or None)."""
    with localcontext() as c:
        c.prec = 1000
        return _format_number(x, n, f, o, percent)


def _format_number(x, n, f, o, percent):
    is_int = _is_int(x)
    if is_int:
        v = int(x)
        neg = v < 0
        d = Decimal(abs(v))
        if percent:
            d = d * 100
    else:
        d = _exact_decimal(x)
        neg = d < 0
        d = d.copy_abs()
        if percent:
            d = d * 100
    machine = not is_int and not _is_bignum(x)
    if is_int:
        nd = len(str(d)) if n is None else n
    else:
        cap = 16 if machine else _dps(x)
        nd = (o.dpp if machine else cap) if n is None else n
        nd = max(1, min(nd, cap))
    pexp = None
    if is_int:
        left, right, has_point = str(int(d)), "", False
        if f is not None:
            has_point = True
    else:
        if d == 0:
            e = 0
            r = Decimal(0)
        else:
            r = _round_sig(d, nd)
            e = r.adjusted()
        pexp = o.exponent(e)
        m = d.scaleb(-pexp) if pexp is not None else d
        if f is not None and m != 0:
            em = _round_sig(m, nd).adjusted()
            decimals = min(nd - 1 - em, f)
            if decimals >= 0:
                r = _quant(m, decimals)
            else:
                r = _round_sig(m, nd)
        elif m != 0:
            r = _round_sig(m, nd)
        else:
            r = Decimal(0) if not _is_bignum(x) else Decimal(0)
        txt = format(r, "f")
        left, _, right = txt.partition(".")
        if machine:
            right = right.rstrip("0")
        left = left or "0"
        has_point = True
    if f is not None and o.pad[1]:
        right = right + o.pad[1] * max(0, f - len(right))
    ndig = len(left) + len(right)
    left_b = _blocks(left, o.block[0], o.sep[0], False)
    right_b = _blocks(right, o.block[1], o.sep[1], True)
    if percent and not right and f is None:
        has_point = False                 # PercentForm[0.5] is 50%, not 50.%
    body = left_b + (o.point + right_b if has_point else "")
    pre, suf = o.signs[0] if neg and (d != 0) else o.signs[1]
    maxsign = max(len(o.signs[0][0]), len(o.signs[1][0]))
    want = nd if n is not None or not is_int else ndig
    if n is not None:
        want = n
    i = len(pre) + ndig - maxsign
    padcount = 0
    if o.pad[0]:
        if i < want:
            padcount = want - i
        elif len(pre) < maxsign:
            padcount = maxsign - len(pre)
    padding = o.pad[0] * padcount
    mant = (pre + padding if o.sign_padding else padding + pre) + body + suf
    exp_s = "" if pexp is None else str(pexp)
    nf = o.nformat
    if not is_sym(nf, "Automatic"):
        res = ev(K.call(nf, mant, "10", exp_s))
        return output_text(res)
    if exp_s:
        mant = mant + o.mult + "10^" + exp_s
    return mant + ("%" if percent else "")


def _digits_spec(spec, fname):
    """n or {n, f} -> (n, f); a non-positive or non-integer spec raises."""
    if spec is None:
        return None, None
    if _is_int(spec) and int(spec) > 0:
        return int(spec), None
    if is_list(spec) and len(spec.args) == 2 and all(_is_int(t) for t in spec.args) and int(spec.args[0]) > 0 \
            and int(spec.args[1]) >= 0:
        return int(spec.args[0]), int(spec.args[1])
    raise ValueError(f"{fname}: formatting specification {F.input_form(spec)} should be a positive integer or a "
                     "pair of positive integers")


def _number_form_parts(e):
    """(expr, n, f, NumOpts, percent) for a number-form wrapper."""
    h = e.head
    args, opts = split_opts(e.args, NUMBER_OPTIONS, h)
    if not args or len(args) > 2:
        raise ValueError(f"{h} called with {len(args)} positional argument(s); expected 1 or 2")
    n, f = _digits_spec(args[1] if len(args) == 2 else None, h)
    if h == "PaddedForm" and n is None:
        n = None
    return args[0], n, f, NumOpts(h, opts, h), h == "PercentForm"


def _validate_number_form(head):
    def impl(*a: Expr) -> Expr:
        tmp = Expr(head, *a)
        _number_form_parts(tmp)     # raises for a malformed specification or option value
        raise Unevaluated()
    return impl


_NUMBER_FORM_NOTES = {
    "NumberForm": "NumberForm[expr, n | {n, f}, opts] inert display wrapper; ToString/REPL show approximate reals to n "
                  "significant digits (f decimals), with DigitBlock, ExponentFunction, ExponentStep, NumberFormat, "
                  "NumberMultiplier, NumberPadding, NumberPoint, NumberSeparator, NumberSigns, "
                  "ScientificNotationThreshold, SignPadding, DefaultPrintPrecision",
    "ScientificForm": "ScientificForm[expr, n | {n, f}, opts]: one digit before the point and an explicit exponent",
    "EngineeringForm": "EngineeringForm[expr, n | {n, f}, opts]: exponents that are multiples of 3",
    "AccountingForm": "AccountingForm[expr, n | {n, f}, opts]: no exponents, negative numbers in parentheses",
    "PaddedForm": "PaddedForm[expr, n | {n, f}, opts]: numbers left-padded with spaces to n digits (room for a sign)",
    "DecimalForm": "DecimalForm[expr, n | {n, f}, opts]: approximate reals always in decimal notation (no exponent)",
    "PercentForm": "PercentForm[expr, n | {n, f}, opts]: numbers shown as percentages (x 100, then %)",
}
for _h in NUMBER_FORMS:
    fn(_h, area=AREA, status="added", notes=_NUMBER_FORM_NOTES[_h])(_validate_number_form(_h))
for _o in NUMBER_OPTIONS:
    record(_o, AREA, "added", f"option of NumberForm and related number forms (inert symbol; default "
                              f"{F.input_form(_DEFAULTS[_o]) if _o != 'DigitBlock' else 'Infinity'})")


# ═════════════════════════════════════ text rendering ═════════════════════════════════════

class Ctx:
    """Rendering context: how atoms are shown."""

    def __init__(self, mode="output", num=None, quote=False, list_format=None, missing=None, booleans=None):
        self.mode = mode            # "output" (OutputForm/StandardForm text), "text" (TextString)
        self.num = num              # callable(number) -> text, or None for the default display
        self.quote = quote          # strings in quotes (InputForm-like)
        self.list_format = list_format
        self.missing = missing      # TextString MissingString: None (unchanged), "auto" (reason), or a string
        self.booleans = booleans

    def with_num(self, num):
        c = Ctx(self.mode, num, self.quote, self.list_format, self.missing, self.booleans)
        c.assoc_format = getattr(self, "assoc_format", None)
        return c


def _default_real_text(x, decimal_only=False):
    """Wolfram's default display of an approximate real: 6 significant digits (bignums: their precision)."""
    o = NumOpts("DecimalForm" if decimal_only else "NumberForm", {}, "OutputForm")
    return format_number(x, None, None, o)


def _block(text):
    return text.split("\n")


def _width(lines):
    return max((len(s) for s in lines), default=0)


def _align(s, w, how):
    if how == "Right":
        return s.rjust(w)
    if how == "Center":
        tot = w - len(s)
        lft = tot // 2
        return " " * lft + s + " " * (tot - lft)
    return s.ljust(w)


class _Walker:
    """Replace atoms and wrappers that need special text by placeholder symbols, render the rest with InputForm
    (so operators, precedence and signs are handled exactly as in InputForm), then substitute the texts back."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.texts = {}

    def ph(self, text):
        k = f"\x01{len(self.texts)}\x02"
        self.texts[k] = text
        return Symbol(k)

    def neg_number(self, x):
        return (_is_real(x) or _is_int(x)) and x < 0

    def number(self, x):
        c = self.ctx
        if c.num is not None:
            return c.num(x)
        if _is_int(x):
            return str(int(x))
        return _default_real_text(x, decimal_only=(c.mode == "text"))

    def walk(self, e):
        c = self.ctx
        if e is True or e is False:
            if c.booleans is not None:
                return self.ph(c.booleans[0] if e else c.booleans[1])
            return e
        if isinstance(e, str):
            return self.ph(F.input_form(e) if c.quote else e)
        if _is_real(e) or (_is_int(e) and c.num is not None):
            if c.num is not None and getattr(c.num, "signed", False):
                return self.ph(self.number(e))
            if e < 0:
                return Expr("Times", -1, self.ph(self.number(-e)))
            return self.ph(self.number(e))
        if not isinstance(e, Expr):
            return e
        q = F.rational_value(e)
        if q is not None and not F.is_int(e):
            if c.num is not None and getattr(c.num, "rational", None):
                return self.ph(c.num.rational(q))
            if c.mode == "text":
                v = float(q)
                return self.ph(_default_real_text(v, decimal_only=True))
            return e
        h = e.head
        if isinstance(h, str):
            r = RENDER.get(h)
            if r is not None:
                try:
                    t = r(e, c)
                except Unevaluated:
                    t = None
                if t is not None:
                    return self.ph(t)
            if c.mode == "text":
                if h == "Missing" and c.missing is not None:
                    if c.missing == "auto":
                        return self.ph(e.args[0] if e.args and isinstance(e.args[0], str) else "")
                    return self.ph(c.missing)
                if h == "List" and c.list_format is not None:
                    lf = c.list_format
                    return self.ph(lf[0] + lf[1].join(output_text(a, c) for a in e.args) + lf[2])
                af = getattr(c, "assoc_format", None)
                if h == "Association" and af is not None and all(
                        isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2
                        for r in e.args):
                    return self.ph(af[0] + af[1].join(output_text(r.args[0], c) + af[3] + output_text(r.args[1], c)
                                                      for r in e.args) + af[2])
        if h == "Times":
            args, sign = [], 1
            for a in e.args:
                if self.neg_number(a) and not (c.num is not None and getattr(c.num, "signed", False)):
                    sign = -sign
                    a = -a
                    if a == 1 and _is_int(a):
                        continue
                w = self.walk(a)
                if isinstance(w, Expr) and w.head == "Times" and w.args and w.args[0] == -1:
                    sign = -sign
                    args.extend(w.args[1:])
                else:
                    args.append(w)
            if sign < 0:
                args.insert(0, -1)
            return Expr("Times", *args) if len(args) != 1 else args[0]
        head = self.walk(h) if isinstance(h, Expr) else h
        return Expr(head, *(self.walk(a) for a in e.args))

    def render(self, e):
        t = F.input_form(self.walk(e))
        if self.texts:
            t = re.sub("\x01\\d+\x02", lambda m: self.texts[m.group(0)], t)
        return t


def output_text(e, ctx=None):
    """Linear OutputForm-style text of e (strings unquoted, wrappers rendered)."""
    ctx = ctx or Ctx()
    if isinstance(e, str):
        return F.input_form(e) if ctx.quote else e
    if e is None:
        return "Null" if ctx.mode != "output" else ""
    return _Walker(ctx).render(e)


def _txt(e, c):
    return output_text(e, c)


RENDER = {}


def renders(*heads):
    def deco(f):
        for h in heads:
            RENDER[h] = f
        return f
    return deco


# ── number forms ──

def _num_renderer(n, f, o, percent):
    def num(x):
        return format_number(x, n, f, o, percent)
    # with the default sign and no left padding, a minus sign is handled by the surrounding operator (x - 2.5 y)
    num.signed = not (o.signs[0] == ("-", "") and o.signs[1] == ("", "") and not o.pad[0])
    if percent:
        def rational(q):
            return format_number(float(q), n, f, o, True) if (q * 100).denominator != 1 \
                else format_number(int(q * 100), n, f, o, False) + "%"
        num.rational = rational
    return num


@renders(*NUMBER_FORMS)
def _r_numberform(e, c):
    x, n, f, o, pct = _number_form_parts(e)
    return output_text(x, c.with_num(_num_renderer(n, f, o, pct)))


# ── BaseForm ──

_DIG = "0123456789abcdefghijklmnopqrstuvwxyz"


def _int_base(v, b):
    if v == 0:
        return "0"
    s = []
    v = abs(v)
    while v:
        v, r = divmod(v, b)
        s.append(_DIG[r])
    return "".join(reversed(s))


def base_digits(x, b):
    """Digits of a number in base b (no subscript)."""
    if _is_int(x):
        v = int(x)
        return ("-" if v < 0 else "") + _int_base(v, b)
    d = _exact_decimal(x)
    neg = d < 0
    fr = Fraction(d.copy_abs())
    machine = not _is_bignum(x)
    prec10 = 6 if machine else _dps(x)
    ndig = max(1, round(prec10 * math.log(10) / math.log(b)))
    ip = int(fr)
    istr = _int_base(ip, b)
    nfrac = max(0, ndig - (len(istr) if ip else 0))
    if ip == 0 and fr:
        # leading zeros after the point do not count as significant digits
        lead = 0
        t = fr
        while t * b < 1 and lead < 4000:
            t *= b
            lead += 1
        nfrac = lead + ndig
    scaled = fr * (b ** nfrac)
    r = int(scaled)
    if scaled - r >= Fraction(1, 2):
        r += 1
    digits = _int_base(r, b).rjust(nfrac + 1, "0") if nfrac else _int_base(r, b)
    ipart = digits[:-nfrac] if nfrac else digits
    fpart = digits[-nfrac:] if nfrac else ""
    fpart = fpart.rstrip("0")
    return ("-" if neg else "") + (ipart or "0") + "." + fpart


def _base_arg(e):
    if len(e.args) != 2:
        raise ValueError(f"BaseForm called with {len(e.args)} argument(s); expected 2")
    b = e.args[1]
    if not _is_int(b) or int(b) <= 0:
        raise ValueError(f"BaseForm: positive machine-sized integer expected at position 2 in "
                         f"{F.input_form(e)}")
    b = int(b)
    if not 2 <= b <= 36:
        raise ValueError(f"BaseForm: requested base {b} must be between 2 and 36")
    return b


@fn("BaseForm", area=AREA, status="added",
    notes="BaseForm[expr, b] inert wrapper (2 <= b <= 36): integers and approximate reals in expr show as digits in "
          "base b with subscript _b (e.g. 100001_2); reals keep as many base-b digits as their decimal precision "
          "(6 digits for machine numbers) and drop trailing zeros")
def _baseform(*a: Expr) -> Expr:
    _base_arg(Expr("BaseForm", *a))
    raise Unevaluated()


@renders("BaseForm")
def _r_baseform(e, c):
    b = _base_arg(e)

    def num(x):
        return base_digits(x, b) + f"_{b}"
    num.signed = True
    num.rational = lambda q: f"{_int_base(q.numerator, b) if q.numerator >= 0 else '-' + _int_base(q.numerator, b)}" \
                             f"_{b}/{_int_base(q.denominator, b)}_{b}"
    return output_text(e.args[0], c.with_num(num))


# ═════════════════════════════════════ ToString, character encodings ═════════════════════════════════════

TEXT_FORMS = ("OutputForm", "StandardForm", "TraditionalForm", "TeXForm", "CForm", "FortranForm", "MathMLForm",
              "InputForm", "FullForm", "TextForm")
TOSTRING_OPTIONS = ("FormatType", "PageWidth", "TotalWidth", "CharacterEncoding", "NumberMarks")

_NAMED_CHARS = {
    "\u00d7": "Times", "\u03c0": "Pi", "\u221e": "Infinity", "\u2192": "Rule", "\u29f4": "RuleDelayed",
    "\u2264": "LessEqual", "\u2265": "GreaterEqual", "\u2260": "NotEqual", "\u00ab": "LeftSkeleton",
    "\u00bb": "RightSkeleton", "\u2026": "Ellipsis", "\u00b0": "Degree", "\u2208": "Element", "\u2227": "And",
    "\u2228": "Or", "\u00ac": "Not", "\u2061": "InvisibleApplication", "\u2062": "InvisibleTimes",
    "\u00b7": "CenterDot", "\u00e0": "AGrave", "\u00e9": "EAcute", "\u00e8": "EGrave", "\u00fc": "UDoubleDot",
    "\u00f6": "ODoubleDot", "\u00e4": "ADoubleDot", "\u00f1": "NTilde", "\u00e7": "CCedilla", "\u00df": "SZ",
    "\u2202": "PartialD", "\u2211": "Sum", "\u220f": "Product", "\u222b": "Integral", "\u221a": "Sqrt",
}
_GREEK = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu",
          "Nu", "Xi", "Omicron", "Pi", "Rho", "FinalSigma", "Sigma", "Tau", "Upsilon", "Phi", "Chi", "Psi", "Omega"]
for _i, _g in enumerate(_GREEK):
    if _g != "FinalSigma":
        _NAMED_CHARS.setdefault(chr(0x3b1 + _i), _g)
        _NAMED_CHARS.setdefault(chr(0x391 + _i), "Capital" + _g)
    else:
        _NAMED_CHARS.setdefault(chr(0x3b1 + _i), "FinalSigma")
_ENCODINGS = {"UTF8": "utf-8", "UTF-8": "utf-8", "Unicode": "utf-8", "ASCII": "ascii", "PrintableASCII": "ascii",
              "ISOLatin1": "latin-1", "ISO8859-1": "latin-1", "UTF16": "utf-8", "UTF-16": "utf-8",
              "WindowsANSI": "cp1252", "CP1252": "cp1252"}


def encode_text(s, enc, fname="ToString"):
    """Characters outside the encoding become \\[Name] (Wolfram's named characters) or \\:xxxx escapes."""
    if is_sym(enc, "None") or is_sym(enc, "Automatic"):
        return s
    if not isinstance(enc, str) or enc not in _ENCODINGS:
        raise ValueError(f"{fname}: {F.input_form(enc)} is not a known character encoding "
                         f"(one of {', '.join(sorted(_ENCODINGS))})")
    codec = _ENCODINGS[enc]
    if codec == "utf-8":
        return s
    out = []
    for ch in s:
        try:
            ch.encode(codec)
            if enc == "PrintableASCII" and (ord(ch) < 32 and ch not in "\n\t"):
                raise UnicodeEncodeError(codec, ch, 0, 1, "")
            out.append(ch)
        except UnicodeEncodeError:
            nm = _NAMED_CHARS.get(ch)
            out.append(f"\\[{nm}]" if nm else (f"\\:{ord(ch):04x}" if ord(ch) <= 0xFFFF else f"\\|{ord(ch):06x}"))
    return "".join(out)


def _wolfram_real_input(x, marks):
    """InputForm text of an approximate real with Wolfram number marks (`) and *^ exponents."""
    if _is_bignum(x):
        p = _dps(x)
        d = _exact_decimal(x)
        r = _round_sig(d.copy_abs(), p) if d != 0 else Decimal(0)
        neg = d < 0
        e = r.adjusted() if d != 0 else 0
        digits = r.as_tuple().digits if d != 0 else (0,)
        ds = "".join(map(str, digits))
        mark = f"`{p}." if marks is not False else ""
    else:
        v = float(x)
        neg = v < 0
        rep = repr(abs(v))
        if "e" in rep or "E" in rep:
            mant, ex = rep.lower().split("e")
        else:
            mant, ex = rep, "0"
        dd = Decimal(mant).scaleb(int(ex))
        if dd == 0:
            return ("-" if neg else "") + "0." + ("`" if marks is True else "")
        e = dd.adjusted()
        ds = "".join(map(str, dd.normalize().as_tuple().digits))
        mark = "`" if marks is True else ""
    ds = ds.rstrip("0") or "0"
    if -5 <= e <= 5:
        if e >= 0:
            ip = ds[:e + 1].ljust(e + 1, "0")
            fp = ds[e + 1:]
        else:
            ip, fp = "0", "0" * (-e - 1) + ds
        s = ip + "." + fp + mark
    else:
        s = ds[0] + "." + ds[1:] + mark + f"*^{e}"
    return ("-" if neg else "") + s


def input_text(e, marks=None):
    """InputForm text with Wolfram-style approximate numbers (NumberMarks: True, False or Automatic/None)."""
    if marks is None:
        return F.input_form(e)

    class W(_Walker):
        def walk(self, x):
            if _is_real(x):
                if x < 0:
                    return Expr("Times", -1, self.ph(_wolfram_real_input(-x, marks)))
                return self.ph(_wolfram_real_input(x, marks))
            if isinstance(x, Expr):
                return Expr(self.walk(x.head) if isinstance(x.head, Expr) else x.head, *(self.walk(a) for a in x.args))
            return x
    return W(Ctx()).render(e)


def _marks_value(v, fname):
    if v is True or v is False:
        return v
    if is_sym(v, "Automatic"):
        return "auto"
    raise ValueError(f"{fname}: NumberMarks must be True, False or Automatic")


def has_wrapper(e, depth=0):
    if isinstance(e, Expr):
        if isinstance(e.head, str) and (e.head in RENDER or e.head in WRAPPERS):
            return True
        if depth > 200:
            return False
        return any(has_wrapper(a, depth + 1) for a in e.args) or (isinstance(e.head, Expr) and has_wrapper(e.head))
    return False


WRAPPERS = set()


def form_text(x, form, opts=None, fname="ToString"):
    """Text of x in the named form ("OutputForm", "InputForm", "TeXForm", ...)."""
    opts = opts or {}
    marks = _marks_value(opts["NumberMarks"], fname) if "NumberMarks" in opts else None
    if form in ("OutputForm", "StandardForm", "TextForm"):
        t = output_text(x)
    elif form == "TraditionalForm":
        t = traditional_text(x)
    elif form == "InputForm":
        t = F.input_form(x) if marks is None else input_text(x, marks)
    elif form == "FullForm":
        t = F.full_form(x)
    elif form == "TeXForm":
        t = tex_text(x)
    elif form == "CForm":
        t = c_text(x)
    elif form == "FortranForm":
        t = fortran_text(x)
    elif form == "MathMLForm":
        t = mathml_text(x)
    else:
        raise ValueError(f"{fname}: unknown form {form}")
    if "TotalWidth" in opts:
        tw = opts["TotalWidth"]
        if not (tw == INF or is_sym(tw, "Infinity")):
            if not _is_int(tw) or int(tw) <= 0:
                raise ValueError(f"{fname}: TotalWidth must be a positive integer or Infinity")
            t = shorten_text(x, form, int(tw))
    if "PageWidth" in opts:
        pw = opts["PageWidth"]
        if not (pw == INF or is_sym(pw, "Infinity")):
            if not _is_int(pw) or int(pw) <= 0:
                raise ValueError(f"{fname}: PageWidth must be a positive integer or Infinity")
            t = break_lines(t, int(pw))
    if "CharacterEncoding" in opts:
        t = encode_text(t, opts["CharacterEncoding"], fname)
    return t


def _form_name(v, fname):
    if isinstance(v, Symbol) and v.name in TEXT_FORMS:
        return v.name
    raise ValueError(f"{fname}: {F.input_form(v)} is not a known format type ({', '.join(TEXT_FORMS)})")


_ORIG_TOSTRING = K._BUILTINS["ToString"]


def _tostring(a, ev_, expr):
    if not a:
        return _ORIG_TOSTRING(a, ev_, expr)
    x, rest = a[0], list(a[1:])
    rest, opts = split_opts(rest, TOSTRING_OPTIONS, "ToString")
    if len(rest) > 1:
        raise ValueError("ToString called with too many arguments; expected ToString[expr, form, options]")
    form = _form_name(rest[0], "ToString") if rest else None
    if "FormatType" in opts:
        form = _form_name(opts.pop("FormatType"), "ToString")
    if isinstance(x, str) and form in (None, "OutputForm", "StandardForm", "TextForm", "TraditionalForm") \
            and not opts:
        return x
    if form is None and not opts and not has_wrapper(x):
        return _ORIG_TOSTRING([x], ev_, expr)
    if form is None:
        form = "OutputForm" if has_wrapper(x) or isinstance(x, str) else "InputForm"
    return form_text(x, form, opts)


K._BUILTINS["ToString"] = _tostring


def display_text(e):
    """What the REPL shows for an output: wrappers in OutputForm text, everything else in InputForm."""
    if isinstance(e, str):
        return e
    if has_wrapper(e):
        return output_text(e)
    return F.input_form(e)


for _n, _d in (("CharacterEncoding", "option of ToString and other output functions: the raw character encoding "
                                      "(\"ASCII\" writes other characters as \\[Name] or \\:xxxx escapes)"),
               ("FormatType", "option of ToString and Text: the format type (InputForm, OutputForm, TeXForm, ...)"),
               ("PageWidth", "option of ToString: lines longer than n characters are broken at spaces"),
               ("TotalWidth", "option of ToString: output longer than n characters is shortened with <<k>> skeletons"),
               ("NumberMarks", "option of ToString[..., InputForm]: whether approximate numbers carry ` marks (True: "
                               "all; Automatic: arbitrary-precision numbers; False: none)")):
    record(_n, AREA, "added", _d)
SYSTEM_VALUES = {"$CharacterEncoding": "UTF-8", "$SystemCharacterEncoding": "UTF-8", "$OutputSizeLimit": 2 ** 20,
                 "$NumberMarks": Symbol("Automatic")}


# ═════════════════════════════════════ line breaking, Short / Skeleton ═════════════════════════════════════

def insert_linebreaks(s, n):
    """Replace whitespace by newlines so no line is longer than n; break longer words every n characters."""
    out_lines = []
    for para in s.split("\n"):
        words = re.split(r"[ \t]+", para)
        line = ""
        for w in words:
            while len(w) > n:
                if line:
                    out_lines.append(line)
                    line = ""
                out_lines.append(w[:n])
                w = w[n:]
            if not line:
                line = w
            elif len(line) + 1 + len(w) <= n:
                line = line + " " + w
            else:
                out_lines.append(line)
                line = w
        out_lines.append(line)
    return "\n".join(out_lines)


def break_lines(t, n):
    """PageWidth: break lines longer than n at spaces (tokens longer than n are kept whole)."""
    res = []
    for line in t.split("\n"):
        while len(line) > n:
            cut = line.rfind(" ", 0, n + 1)
            if cut <= 0:
                cut = line.find(" ", n)
                if cut <= 0:
                    break
            res.append(line[:cut].rstrip())
            line = line[cut + 1:]
        res.append(line)
    return "\n".join(res)


def _skeleton_text(k):
    return f"<<{k}>>"


def _shorten(e, budget, form):
    """An expression whose text fits in about `budget` characters, with Skeleton[k] for omitted elements."""
    def skel(k):
        return Expr("Skeleton", k) if form != "InputForm" else Symbol(f"<<{k}>>")

    def size(x):
        return len(form_text(x, form))
    if size(e) <= budget:
        return e
    if isinstance(e, str):
        keep = max(2, budget - 5)
        a = keep // 2
        b = keep - a
        k = len(e) - a - b
        if k <= 0:
            return e
        return Expr("StringSkeletonJoin", e[:a], k, e[len(e) - b:])
    if not isinstance(e, Expr) or not e.args:
        return e
    args = list(e.args)
    n = len(args)
    lo, hi = 0, n          # binary search on the number of kept elements (split between front and back)
    best = Expr(e.head, skel(n))
    while lo <= hi:
        mid = (lo + hi) // 2
        a = (mid + 1) // 2
        b = mid - a
        if mid >= n:
            cand = e
        else:
            cand = Expr(e.head, *args[:a], skel(n - mid), *(args[n - b:] if b else []))
        if size(cand) <= budget:
            best = cand
            lo = mid + 1
        else:
            hi = mid - 1
    if n == 1 and best.args and len(best.args) == 1 and not (best.args[0] is args[0]):
        inner = _shorten(args[0], budget - len(F.input_form(Expr(e.head, Symbol("_")))) + 1, form)
        return Expr(e.head, inner)
    return best


def shorten_text(x, form, budget):
    return form_text(_shorten(x, budget, form), form)


@renders("Skeleton")
def _r_skeleton(e, c):
    if len(e.args) != 1:
        return None
    return _skeleton_text(_txt(e.args[0], c))


@renders("StringSkeleton")
def _r_stringskeleton(e, c):
    return "\u2026" if len(e.args) == 1 else None


@renders("StringSkeletonJoin")
def _r_ssj(e, c):
    a, k, b = e.args
    return a + "\u2026" + b


def _short_args(e):
    if not e.args or len(e.args) > 2:
        raise ValueError(f"Short called with {len(e.args)} argument(s); expected 1 or 2")
    n = 1
    if len(e.args) == 2:
        v = _num_value(e.args[1])
        if v is None or v <= 0:
            raise ValueError("Short: the number of lines must be a positive number")
        n = v
    return e.args[0], n


@fn("Short", area=AREA, status="added",
    notes="Short[expr, n] inert wrapper: shows expr in about n lines (n * 78 characters), omitted runs of k "
          "elements written <<k>> (Skeleton[k]) and long strings elided with an ellipsis")
def _short(*a: Expr) -> Expr:
    _short_args(Expr("Short", *a))
    raise Unevaluated()


@renders("Short")
def _r_short(e, c):
    x, n = _short_args(e)
    budget = max(10, int(78 * n))
    return output_text(_shorten(x, budget, "OutputForm"), c)


@fn("Skeleton", area=AREA, status="added",
    notes="Skeleton[k] inert: k omitted elements, shown as <<k>> (what Short inserts)")
def _skeleton(k: NonNegInteger) -> Expr:
    raise Unevaluated()


@fn("StringSkeleton", area=AREA, status="added",
    notes="StringSkeleton[k] inert: k omitted characters of a string, shown as an ellipsis")
def _stringskeleton(k: NonNegInteger) -> Expr:
    raise Unevaluated()


@fn("InsertLinebreaks", area=AREA, status="added",
    notes="InsertLinebreaks[s, n] (default 78) replaces whitespace by newlines so no line exceeds n characters; "
          "words longer than n are broken every n characters")
def _insertlinebreaks(s: String, n: PosInteger = 78) -> String:
    if not isinstance(s, str):
        if isinstance(s, (Symbol, Expr)):
            raise Unevaluated()
        raise ValueError("InsertLinebreaks: the first argument must be a string")
    if not _is_int(n) or int(n) <= 0:
        raise ValueError("InsertLinebreaks: the line width must be a positive integer")
    return insert_linebreaks(s, int(n))


# ═════════════════════════════════════ layout ═════════════════════════════════════

H_ALIGN = {"Left": "Left", "Right": "Right", "Center": "Center", "Automatic": "Left", "Before": "Left",
           "After": "Right"}
V_ALIGN = {"Top": "Top", "Bottom": "Bottom", "Center": "Center", "Baseline": "Top", "Automatic": "Top",
           "Axis": "Center"}


def _halign(v, fname):
    if isinstance(v, Symbol) and v.name in H_ALIGN:
        return H_ALIGN[v.name]
    if isinstance(v, str) and v in (".", ","):
        return "Left"
    raise ValueError(f"{fname}: {F.input_form(v)} is not a valid horizontal alignment (Left, Center, Right)")


def _valign(v, fname):
    if isinstance(v, Symbol) and v.name in V_ALIGN:
        return V_ALIGN[v.name]
    raise ValueError(f"{fname}: {F.input_form(v)} is not a valid vertical alignment (Top, Center, Bottom, "
                     "Baseline)")


def _cell(x, c):
    """(lines, alignment override) of a grid entry; Item[expr, Alignment -> a] carries its own alignment."""
    if isinstance(x, Expr) and x.head == "Item" and x.args:
        args, opts = split_opts(x.args, ITEM_OPTIONS, "Item")
        al = None
        if "Alignment" in opts:
            a = opts["Alignment"]
            al = _halign(a.args[0] if is_list(a) else a, "Item")
        return _block(_txt(args[0], c)), al
    return _block(_txt(x, c)), None


def layout_grid(rows, c, colsep=3, rowsep=1, halign="Left", valign="Top", col_aligns=None, row_aligns=None,
                col_widths=None, col_lines=False, row_lines=False, min_height=None, frame=False):
    """Text of a 2-D arrangement; rows is a list of lists of expressions (ragged rows allowed)."""
    cells = [[_cell(x, c) for x in r] for r in rows]
    ncol = max((len(r) for r in cells), default=0)
    if ncol == 0:
        return ""
    widths = [0] * ncol
    for r in cells:
        for j, (lines, _a) in enumerate(r):
            widths[j] = max(widths[j], _width(lines))
    if col_widths:
        widths = [max(w, col_widths[j] if j < len(col_widths) and col_widths[j] else 0) for j, w in enumerate(widths)]
    sep = (" " * (colsep // 2) + "|" + " " * (colsep - colsep // 2 - 1)) if col_lines and colsep >= 1 else \
        ("|" if col_lines else " " * colsep)
    out_rows = []
    for i, r in enumerate(cells):
        h = max((len(lines) for lines, _a in r), default=1)
        if min_height:
            h = max(h, int(math.ceil(min_height)))
        va = row_aligns[i] if row_aligns and i < len(row_aligns) and row_aligns[i] else valign
        cols = []
        for j in range(ncol):
            lines, al = r[j] if j < len(r) else ([""], None)
            ha = al or (col_aligns[j] if col_aligns and j < len(col_aligns) and col_aligns[j] else halign)
            pad = h - len(lines)
            if va == "Bottom":
                lines = [""] * pad + lines
            elif va == "Center":
                lines = [""] * (pad // 2) + lines + [""] * (pad - pad // 2)
            else:
                lines = lines + [""] * pad
            cols.append([_align(s, widths[j], ha) for s in lines])
        block = [sep.join(cols[j][k] for j in range(ncol)) for k in range(h)]
        out_rows.append(block)
    total = len(out_rows[0][0]) if out_rows and out_rows[0] else 0
    lines = []
    for i, b in enumerate(out_rows):
        if i:
            if row_lines:
                lines.extend([""] * ((rowsep - 1) // 2))
                lines.append("-" * total)
                lines.extend([""] * (rowsep - 1 - (rowsep - 1) // 2))
            else:
                lines.extend([""] * rowsep)
        lines.extend(b)
    if frame:
        lines = _frame_lines(lines)
    else:
        lines = [s.rstrip() for s in lines]
    return "\n".join(lines)


def _frame_lines(lines):
    w = _width(lines)
    return ["+" + "-" * (w + 2) + "+"] + ["| " + s.ljust(w) + " |" for s in lines] + ["+" + "-" * (w + 2) + "+"]


def _list_or_raise(x, fname, what="list"):
    if is_list(x):
        return list(x.args)
    raise ValueError(f"{fname}: the first argument must be a {what}")


def _per_item(v, fn_, fname):
    """Alignment-like spec: a single value or {v1, v2, ...} (one per column)."""
    if is_list(v):
        return [fn_(t, fname) for t in v.args]
    return fn_(v, fname)


def _spacing_value(v, fname, what):
    x = _num_value(v)
    if x is None or x < 0:
        raise ValueError(f"{fname}: {what} must be a non-negative number")
    return int(round(x))


GRID_OPTIONS = ("Alignment", "Background", "BaselinePosition", "BaseStyle", "Dividers", "Frame", "FrameStyle",
                "ItemSize", "ItemStyle", "Spacings", "AllowedDimensions", "AutoDelete", "DefaultElement",
                "ColumnAlignments", "ColumnLines", "ColumnSpacings", "ColumnWidths", "RowAlignments", "RowLines",
                "RowSpacings", "RowMinHeight", "GridDefaultElement", "ColumnsEqual", "RowsEqual",
                "Appearance", "Pagination")
ITEM_OPTIONS = ("Alignment", "Background", "Frame", "FrameStyle", "ItemSize", "ItemStyle", "BaseStyle",
                "Dividers", "Spacings")


def _bool_opt(v, fname, name):
    if v is True or v is False:
        return v
    if is_sym(v, "All"):
        return True
    if is_sym(v, "None"):
        return False
    if is_list(v):
        return any(_bool_opt(t, fname, name) for t in v.args)
    raise ValueError(f"{fname}: {name} must be True, False, All or None")


def _grid_layout(rows, opts, fname, c, colsep=3, rowsep=1, halign="Left"):
    kw = dict(colsep=colsep, rowsep=rowsep, halign=halign)
    if "Alignment" in opts:
        a = opts["Alignment"]
        if is_list(a) and len(a.args) == 2 and not is_list(a.args[0]) and isinstance(a.args[1], Symbol) and \
                a.args[1].name in V_ALIGN and a.args[0] is not None and isinstance(a.args[0], Symbol) and \
                a.args[0].name in H_ALIGN:
            kw["halign"] = _halign(a.args[0], fname)
            kw["valign"] = _valign(a.args[1], fname)
        elif is_list(a) and a.args and is_list(a.args[0]):
            kw["col_aligns"] = [_halign(t, fname) for t in a.args[0].args]
            if len(a.args) > 1:
                kw["valign"] = _valign(a.args[1], fname)
        else:
            kw["halign"] = _halign(a, fname)
    if "ColumnAlignments" in opts:
        v = _per_item(opts["ColumnAlignments"], _halign, fname)
        if isinstance(v, list):
            kw["col_aligns"] = v
        else:
            kw["halign"] = v
    if "RowAlignments" in opts:
        v = _per_item(opts["RowAlignments"], _valign, fname)
        if isinstance(v, list):
            kw["row_aligns"] = v
        else:
            kw["valign"] = v
    if "ColumnSpacings" in opts:
        kw["colsep"] = _spacing_value(opts["ColumnSpacings"], fname, "ColumnSpacings")
    if "RowSpacings" in opts:
        kw["rowsep"] = _spacing_value(opts["RowSpacings"], fname, "RowSpacings")
    if "Spacings" in opts:
        s = opts["Spacings"]
        if is_list(s) and len(s.args) == 2:
            kw["colsep"] = _spacing_value(s.args[0], fname, "Spacings")
            kw["rowsep"] = _spacing_value(s.args[1], fname, "Spacings")
        elif not is_sym(s, "Automatic"):
            kw["colsep"] = _spacing_value(s, fname, "Spacings")
    if "ColumnWidths" in opts:
        w = opts["ColumnWidths"]
        if is_list(w):
            kw["col_widths"] = [_spacing_value(t, fname, "ColumnWidths") if not is_sym(t, "Automatic") else 0
                                for t in w.args]
        elif not is_sym(w, "Automatic"):
            kw["col_widths"] = [_spacing_value(w, fname, "ColumnWidths")] * max(len(r) for r in rows)
    if "ColumnLines" in opts:
        kw["col_lines"] = _bool_opt(opts["ColumnLines"], fname, "ColumnLines")
    if "RowLines" in opts:
        kw["row_lines"] = _bool_opt(opts["RowLines"], fname, "RowLines")
    if "Dividers" in opts:
        d = opts["Dividers"]
        if is_sym(d, "All") or d is True:
            kw["col_lines"] = kw["row_lines"] = True
    if "RowMinHeight" in opts:
        v = _num_value(opts["RowMinHeight"])
        if v is None or v < 0:
            raise ValueError(f"{fname}: RowMinHeight must be a non-negative number")
        kw["min_height"] = v
    if "Frame" in opts:
        kw["frame"] = _bool_opt(opts["Frame"], fname, "Frame")
    return layout_grid(rows, c, **kw)


def _grid_rows(x, fname):
    items = _list_or_raise(x, fname, "list of lists")
    return [list(r.args) if is_list(r) else [r] for r in items]


@fn("Grid", area=AREA, status="added",
    notes="Grid[{{a, b}, ...}, opts] inert layout object; text: columns 3 spaces apart, rows separated by a blank "
          "line, left aligned (Wolfram's OutputForm), with Alignment, Spacings, Frame, Dividers and the GridBox "
          "options ColumnAlignments, ColumnSpacings, ColumnWidths, ColumnLines, RowAlignments, RowSpacings, "
          "RowLines, RowMinHeight honoured")
def _grid(x: Expr, *opts: Option) -> Expr:
    rest, o = split_opts(opts, GRID_OPTIONS, "Grid")
    if rest:
        raise ValueError("Grid: options expected after the list of rows")
    if not is_list(x):
        raise Unevaluated() if isinstance(x, Symbol) else ValueError("Grid: the first argument must be a list "
                                                                      "of lists")
    _grid_layout(_grid_rows(x, "Grid"), o, "Grid", Ctx())   # validates the options
    raise Unevaluated()


@renders("Grid")
def _r_grid(e, c):
    if not e.args or not is_list(e.args[0]):
        return None
    rest, o = split_opts(e.args[1:], GRID_OPTIONS, "Grid")
    return _grid_layout(_grid_rows(e.args[0], "Grid"), o, "Grid", c)


@fn("Column", area=AREA, status="added",
    notes="Column[{e1, e2, ...}, alignment, spacing, opts] inert layout: elements one per line (spacing blank "
          "lines between), aligned Left, Center or Right")
def _column(x: Expr, *rest: Expr) -> Expr:
    _column_parts(Expr("Column", x, *rest))
    raise Unevaluated()


def _column_parts(e):
    if not e.args:
        raise ValueError("Column called with 0 arguments")
    x = e.args[0]
    if not is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("Column: the first argument must be a list")
    rest, o = split_opts(e.args[1:], GRID_OPTIONS, "Column")
    if len(rest) > 2:
        raise ValueError("Column called with too many arguments")
    al = _halign(rest[0], "Column") if rest else "Left"
    if "Alignment" in o:
        a = o.pop("Alignment")
        al = _halign(a.args[0] if is_list(a) else a, "Column")
    sp_ = _spacing_value(rest[1], "Column", "the spacing") if len(rest) > 1 else 0
    if "Spacings" in o:
        sp_ = _spacing_value(o.pop("Spacings"), "Column", "Spacings")
    return list(x.args), al, sp_, o


@renders("Column")
def _r_column(e, c):
    items, al, sp_, o = _column_parts(e)
    return _grid_layout([[t] for t in items], o, "Column", c, colsep=0, rowsep=sp_, halign=al)


@fn("Row", area=AREA, status="added",
    notes="Row[{e1, e2, ...}] / Row[list, sep] inert layout: the elements' text concatenated (with sep between); "
          "multi-line elements are placed side by side")
def _row(x: Expr, *rest: Expr) -> Expr:
    _row_parts(Expr("Row", x, *rest))
    raise Unevaluated()


def _row_parts(e):
    x = e.args[0]
    if not is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("Row: the first argument must be a list")
    rest, o = split_opts(e.args[1:], GRID_OPTIONS, "Row")
    if len(rest) > 1:
        raise ValueError("Row called with too many arguments")
    return list(x.args), (rest[0] if rest else None)


def side_by_side(blocks, valign="Top"):
    h = max((len(b) for b in blocks), default=1)
    cols = []
    for b in blocks:
        w = _width(b)
        pad = h - len(b)
        if valign == "Bottom":
            b = [""] * pad + b
        elif valign == "Center":
            b = [""] * (pad // 2) + b + [""] * (pad - pad // 2)
        else:
            b = b + [""] * pad
        cols.append([s.ljust(w) for s in b])
    return "\n".join("".join(col[k] for col in cols).rstrip() if h > 1 else "".join(col[k] for col in cols)
                     for k in range(h))


@renders("Row")
def _r_row(e, c):
    if not e.args or not is_list(e.args[0]):
        return None
    items, sep = _row_parts(e)
    blocks = []
    for i, t in enumerate(items):
        if i and sep is not None:
            blocks.append(_block(_txt(sep, c)))
        blocks.append(_block(_txt(t, c)))
    if not blocks:
        return ""
    if all(len(b) == 1 for b in blocks):
        return "".join(b[0] for b in blocks)
    return side_by_side(blocks)


TABLE_OPTIONS = ("TableAlignments", "TableDepth", "TableDirections", "TableHeadings", "TableSpacing")


def _depth_of(x):
    if not is_list(x):
        return 0
    if not x.args:
        return 1
    return 1 + min(_depth_of(a) for a in x.args)


def _table_parts(e):
    if not e.args:
        raise ValueError("TableForm called with 0 arguments")
    x = e.args[0]
    rest, o = split_opts(e.args[1:], TABLE_OPTIONS, "TableForm")
    if rest:
        raise ValueError("TableForm: options expected after the list")
    depth = INF
    if "TableDepth" in o:
        d = o["TableDepth"]
        if d == INF or is_sym(d, "Infinity"):
            depth = INF
        elif _is_int(d) and int(d) >= 1:
            depth = int(d)
        else:
            raise ValueError("TableForm: TableDepth must be a positive integer or Infinity")
    dirs = None
    if "TableDirections" in o:
        d = o["TableDirections"]
        def one(t):
            if is_sym(t, "Row") or is_sym(t, "Column"):
                return t.name
            raise ValueError("TableForm: TableDirections must be Row, Column or a list of them")
        dirs = [one(t) for t in d.args] if is_list(d) else [one(d)]
    spacing = [1, 3]
    if "TableSpacing" in o:
        s = o["TableSpacing"]
        if is_list(s):
            spacing = [_spacing_value(t, "TableForm", "TableSpacing") for t in s.args]
        elif is_sym(s, "Automatic"):
            pass
        else:
            v = _spacing_value(s, "TableForm", "TableSpacing")
            spacing = [v, v]
    aligns = ["Left"]
    if "TableAlignments" in o:
        a = o["TableAlignments"]
        def one_a(t):
            if isinstance(t, Symbol) and t.name in ("Left", "Right", "Center", "Top", "Bottom", "Automatic",
                                                     "Baseline"):
                return t.name
            raise ValueError("TableForm: TableAlignments must be Left, Right, Center, Top, Bottom or a list of "
                             "them")
        aligns = [one_a(t) for t in a.args] if is_list(a) else [one_a(a)]
    heads = None
    if "TableHeadings" in o:
        hd = o["TableHeadings"]
        if is_sym(hd, "None"):
            heads = None
        elif is_sym(hd, "Automatic"):
            heads = ["auto", "auto"]
        elif is_list(hd):
            heads = []
            for t in hd.args:
                if is_sym(t, "None"):
                    heads.append(None)
                elif is_sym(t, "Automatic"):
                    heads.append("auto")
                elif is_list(t):
                    heads.append(list(t.args))
                else:
                    raise ValueError("TableForm: TableHeadings must be None, Automatic or {rowheads, colheads}")
        else:
            raise ValueError("TableForm: TableHeadings must be None, Automatic or {rowheads, colheads}")
    return x, depth, dirs, spacing, aligns, heads


@fn("TableForm", area=AREA, status="added",
    notes="TableForm[list, opts] inert display wrapper; text: rows separated by a blank line and columns by 3 spaces "
          "(Wolfram's OutputForm), with TableDepth, TableDirections, TableSpacing, TableAlignments and "
          "TableHeadings")
def _tableform(*a: Expr) -> Expr:
    _table_parts(Expr("TableForm", *a))
    raise Unevaluated()


def _dir(dirs, level):
    if dirs is None:
        return "Column" if level % 2 == 0 else "Row"
    if level < len(dirs):
        return dirs[level]
    last = dirs[-1]
    # continue alternating after the explicit settings
    k = level - len(dirs) + 1
    return last if k % 2 == 0 else ("Row" if last == "Column" else "Column")


def _table_text(x, c, level, depth, dirs, spacing, aligns, heads=None):
    if not is_list(x) or level >= depth or (level > 0 and not x.args):
        return _txt(x, c)
    items = list(x.args)
    d = _dir(dirs, level)
    sub = [_table_text(t, c, level + 1, depth, dirs, spacing, aligns) for t in items]
    halign = next((a for a in aligns if a in ("Left", "Right", "Center")), "Left")
    rowsep = spacing[0] if len(spacing) > 0 else 1
    colsep = spacing[1] if len(spacing) > 1 else 3
    # a list whose elements are rows (direction Column) and whose elements' elements are laid along a row
    if d == "Column":
        rows = []
        all_rows = all(is_list(t) and level + 1 < depth and _dir(dirs, level + 1) == "Row" for t in items) and items
        if all_rows:
            rows = [[_table_text(u, c, level + 2, depth, dirs, spacing, aligns) for u in t.args] for t in items]
        else:
            rows = [[s] for s in sub]
        return _layout_text_rows(rows, colsep, rowsep, halign, heads)
    rowsv = [[s for s in sub]]
    if all(is_list(t) and level + 1 < depth and _dir(dirs, level + 1) == "Column" for t in items) and items:
        cols = [[_table_text(u, c, level + 2, depth, dirs, spacing, aligns) for u in t.args] for t in items]
        nrow = max(len(col) for col in cols)
        rowsv = [[col[i] if i < len(col) else "" for col in cols] for i in range(nrow)]
    return _layout_text_rows(rowsv, colsep, rowsep, halign, heads)


def _layout_text_rows(rows, colsep, rowsep, halign, heads):
    """rows of already-rendered texts -> text (texts are wrapped as strings so they are not re-rendered)."""
    if heads:
        rh = heads[0] if len(heads) > 0 else None
        ch = heads[1] if len(heads) > 1 else None
        nrow = len(rows)
        ncol = max((len(r) for r in rows), default=0)
        if ch is not None:
            labels = [str(i + 1) for i in range(ncol)] if ch == "auto" else [output_text(t) for t in ch]
            rows = [labels + [""] * (ncol - len(labels))] + rows
        if rh is not None:
            labels = [str(i + 1) for i in range(nrow)] if rh == "auto" else [output_text(t) for t in rh]
            labels = labels + [""] * (nrow - len(labels))
            rows = [([""] if ch is not None else []) + rows[0]] + rows[1:] if False else rows
            off = 1 if ch is not None else 0
            rows = [([""] + r) if i < off else ([labels[i - off] if i - off < len(labels) else ""] + r)
                    for i, r in enumerate(rows)]
    return layout_grid([[_Pre(s) for s in r] for r in rows], Ctx(), colsep=colsep, rowsep=rowsep, halign=halign)


class _Pre(str):
    """An already-rendered text cell."""


@renders("TableForm", "MatrixForm")
def _r_tableform(e, c):
    if e.head == "MatrixForm":
        if not e.args:
            return None
        return _table_text(e.args[0], c, 0, INF, None, [1, 3], ["Left"])
    x, depth, dirs, spacing, aligns, heads = _table_parts(e)
    if is_list(x) and not x.args:
        return ""
    return _table_text(x, c, 0, depth, dirs, spacing, aligns, heads)


# ── Style, Framed, Labeled, Placed, Text, TextElement, Overlay, Item, Multicolumn ──

STYLE_OPTIONS = ("FontFamily", "FontSize", "FontWeight", "FontSlant", "FontColor", "Background", "FontVariations",
                 "FontTracking", "LineSpacing", "TextAlignment", "ScriptLevel", "ShowStringCharacters",
                 "SingleLetterItalics", "Magnification", "PrintPrecision", "FormatType", "BaseStyle",
                 "Visible", "LineIndent", "TextJustification", "Hyphenation", "FontOpacity")
_FONT_SIZES = {"Tiny", "Small", "Medium", "Large", "Larger", "Smaller", "Inherited", "Automatic"}
_DIRECTIVE_SYMBOLS = {"Bold", "Italic", "Plain", "Underlined", "Struckthrough", "Larger", "Smaller", "Tiny",
                      "Small", "Medium", "Large", "Black", "White", "Red", "Green", "Blue", "Gray", "Cyan", "Magenta",
                      "Yellow", "Brown", "Orange", "Pink", "Purple", "LightGray", "LightRed", "LightBlue",
                      "LightGreen", "LightYellow", "LightOrange", "LightPurple", "LightPink", "LightBrown",
                      "LightCyan", "LightMagenta", "Transparent", "Thick", "Thin", "Dashed", "Dotted", "DotDashed",
                      "Opacity", "None", "Automatic", "Inherited"}


def _check_style_opt(name, v, fname):
    if name == "FontSize":
        x = _num_value(v)
        if x is not None:
            if x <= 0:
                raise ValueError(f"{fname}: FontSize must be positive")
            return
        if isinstance(v, Symbol) and v.name in _FONT_SIZES:
            return
        if isinstance(v, Expr) and v.head in ("Scaled", "Offset"):
            return
        raise ValueError(f"{fname}: FontSize must be a positive number, Scaled[s], Tiny, Small, Medium or Large")
    if name == "FontFamily":
        if not isinstance(v, str) and not is_sym(v, "Automatic") and not is_sym(v, "Inherited"):
            raise ValueError(f"{fname}: FontFamily must be a string such as \"Helvetica\"")
    if name == "Visible" and v is not True and v is not False:
        raise ValueError(f"{fname}: Visible must be True or False")
    if name == "ScriptLevel":
        if not (_is_int(v) and int(v) >= 0) and not is_sym(v, "Automatic") and not is_sym(v, "Inherited"):
            raise ValueError(f"{fname}: ScriptLevel must be a non-negative integer")


def _style_parts(e):
    if not e.args:
        raise ValueError("Style called with 0 arguments")
    for s in e.args[1:]:
        nm = _opt_name(s)
        if nm is not None:
            _check_style_opt(nm, s.args[1], "Style")
        elif isinstance(s, str) or _num_value(s) is not None:
            if _num_value(s) is not None and _num_value(s) <= 0:
                raise ValueError("Style: a font size must be positive")
        elif isinstance(s, Symbol) or isinstance(s, Expr):
            pass
    return e.args[0]


@fn("Style", area=AREA, status="added",
    notes="Style[expr, styles...] inert display wrapper (Bold, Italic, colours, font size n, \"stylename\", "
          "FontSize -> n, FontFamily -> \"name\", Background -> c, ...); plain text shows expr unchanged")
def _style(*a: Expr) -> Expr:
    _style_parts(Expr("Style", *a))
    raise Unevaluated()


@renders("Style")
def _r_style(e, c):
    return _txt(_style_parts(e), c)


@fn("Framed", area=AREA, status="added",
    notes="Framed[expr, opts] inert display wrapper; text draws an ASCII box (+---+ / | expr |) around expr")
def _framed(x: Expr, *opts: Option) -> Expr:
    rest, _o = split_opts(opts, ("Background", "FrameMargins", "FrameStyle", "ImageMargins", "ImageSize",
                                 "RoundingRadius", "Alignment", "BaselinePosition", "BaseStyle", "ContentPadding",
                                 "FrameLabel", "Frame"), "Framed")
    if rest:
        raise ValueError("Framed called with too many arguments")
    raise Unevaluated()


@renders("Framed")
def _r_framed(e, c):
    if not e.args:
        return None
    return "\n".join(_frame_lines(_block(_txt(e.args[0], c))))


_LABEL_POS = {"Bottom": "Bottom", "Top": "Top", "Left": "Left", "Right": "Right", "Before": "Left",
              "After": "Right", "Above": "Top", "Below": "Bottom"}


def _labeled_parts(e):
    a, o = split_opts(e.args, ("LabelStyle", "Spacings", "Frame", "FrameStyle", "Background", "RoundingRadius",
                               "FrameMargins", "Alignment", "BaseStyle"), "Labeled")
    if len(a) < 2 or len(a) > 3:
        raise ValueError(f"Labeled called with {len(a)} argument(s); expected 2 or 3")
    x, lbl = a[0], a[1]
    if len(a) == 3:
        pos = a[2]
        if is_sym(pos, "All"):
            if not (is_list(lbl) and len(lbl.args) == 4):
                raise ValueError("Labeled: with position All the labels must be a list {bottom, left, top, right}")
            return x, list(zip(lbl.args, ["Bottom", "Left", "Top", "Right"]))
        if is_list(pos):
            if not (is_list(lbl) and len(lbl.args) == len(pos.args)):
                raise ValueError("Labeled: the lists of labels and positions must have the same length")
            return x, [(l_, _label_pos(p)) for l_, p in zip(lbl.args, pos.args)]
        return x, [(lbl, _label_pos(pos))]
    return x, [(lbl, "Bottom")]


def _label_pos(p):
    if isinstance(p, Symbol) and p.name in _LABEL_POS:
        return _LABEL_POS[p.name]
    raise ValueError(f"Labeled: {F.input_form(p)} is not a valid label position (Top, Bottom, Left, Right, Before, "
                     "After, Above, Below)")


@fn("Labeled", area=AREA, status="added",
    notes="Labeled[expr, lbl, pos] inert display wrapper (pos Bottom by default; Top, Left, Right, Before, After, "
          "Above, Below, a list of positions, or All for {bottom, left, top, right}); text places the label "
          "blocks around expr, centred above/below")
def _labeled(*a: Expr) -> Expr:
    _labeled_parts(Expr("Labeled", *a))
    raise Unevaluated()


@renders("Labeled")
def _r_labeled(e, c):
    x, labels = _labeled_parts(e)
    core = _block(_txt(x, c))
    for lbl, pos in labels:
        lb = _block(_txt(lbl, c))
        if pos in ("Left", "Right"):
            blocks = [lb, [" "], core] if pos == "Left" else [core, [" "], lb]
            core = _block(side_by_side(blocks, "Center"))
        else:
            w = max(_width(core), _width(lb))
            core = [_align(s, w, "Center").rstrip() for s in core]
            lb = [_align(s, w, "Center").rstrip() for s in lb]
            core = (lb + core) if pos == "Top" else (core + lb)
    return "\n".join(core)


@fn("Placed", area=AREA, status="added",
    notes="Placed[expr, pos] / Placed[{e1, ...}, pos, f] inert: an annotation at relative position pos (Top, "
          "Bottom, Left, Right, Center, Above, Below, Before, After, {x, y}, ...); text shows expr (f applied to each "
          "ei when given)")
def _placed(x: Expr, pos: Expr, *f: Expr) -> Expr:
    if len(f) > 1:
        raise ValueError("Placed called with too many arguments")
    raise Unevaluated()


@renders("Placed")
def _r_placed(e, c):
    if len(e.args) == 3:
        x, _pos, f = e.args
        items = list(x.args) if is_list(x) else [x]
        res = [ev(K.call(f, t)) for t in items]
        return _txt(L(*res) if is_list(x) else res[0], c)
    if len(e.args) == 2:
        return _txt(e.args[0], c)
    return None


TEXT_OPTIONS = ("FormatType", "Background", "BaseStyle", "BaselinePosition", "Alignment") + STYLE_OPTIONS


@fn("Text", area=AREA, status="added",
    notes="Text[expr] shows expr as plain text; Text[expr, coords, offset, dir] is an inert graphics primitive "
          "(coordinates a list of 2 or 3 numbers, Scaled[...] or Offset[...])")
def _text(x: Expr, *rest: Expr) -> Expr:
    a, o = split_opts(rest, TEXT_OPTIONS, "Text")
    if len(a) > 3:
        raise ValueError("Text called with too many arguments")
    if a:
        p = a[0]
        ok = (is_list(p) and len(p.args) in (2, 3)) or (isinstance(p, Expr) and p.head in ("Scaled", "Offset",
                                                                                          "ImageScaled"))
        if not ok and not isinstance(p, Symbol):
            raise ValueError("Text: coordinates must be {x, y}, {x, y, z}, Scaled[...] or Offset[...]")
    if "FormatType" in o:
        _form_name(o["FormatType"], "Text")
    raise Unevaluated()


@renders("Text")
def _r_text(e, c):
    if not e.args:
        return None
    a, o = split_opts(e.args[1:], TEXT_OPTIONS, "Text")
    if a:
        return None      # a graphics primitive: shown as the expression itself
    if "FormatType" in o:
        return form_text(e.args[0], _form_name(o["FormatType"], "Text"))
    return _txt(e.args[0], c)


@fn("TextElement", area=AREA, status="added",
    notes="TextElement[text, props] / TextElement[{e1, e2, ...}, props] inert: an element of text with a property "
          "association; text shows the elements' strings concatenated")
def _textelement(x: Expr, *props: Expr) -> Expr:
    if len(props) > 1:
        raise ValueError("TextElement called with too many arguments")
    if props and not (isinstance(props[0], Expr) and props[0].head == "Association"):
        raise ValueError("TextElement: properties must be an association")
    raise Unevaluated()


@renders("TextElement")
def _r_textelement(e, c):
    if not e.args:
        return None
    x = e.args[0]
    if is_list(x):
        return "".join(_txt(t, c) for t in x.args)
    return _txt(x, c)


@fn("Overlay", area=AREA, status="added",
    notes="Overlay[{e1, e2, ...}, {i, j, ...}] inert: the expressions drawn on top of each other (top-left aligned); "
          "text superimposes the blocks, later ones overwriting earlier ones where they are not blank")
def _overlay(x: Expr, *rest: Expr) -> Expr:
    _overlay_parts(Expr("Overlay", x, *rest))
    raise Unevaluated()


def _overlay_parts(e):
    a, _o = split_opts(e.args, ("Alignment", "Background", "BaseStyle", "BaselinePosition", "ImageSize"), "Overlay")
    if not a or not is_list(a[0]):
        raise ValueError("Overlay: the first argument must be a list")
    items = list(a[0].args)
    idx = list(range(1, len(items) + 1))
    if len(a) >= 2:
        s = a[1]
        if is_sym(s, "All"):
            pass
        elif is_list(s) and all(_is_int(t) and 1 <= int(t) <= len(items) for t in s.args):
            idx = [int(t) for t in s.args]
        else:
            raise ValueError("Overlay: the layers must be a list of indices into the list of expressions")
    if len(a) > 3:
        raise ValueError("Overlay called with too many arguments")
    return [items[i - 1] for i in idx]


@renders("Overlay")
def _r_overlay(e, c):
    layers = [_block(_txt(t, c)) for t in _overlay_parts(e)]
    h = max((len(b) for b in layers), default=0)
    w = max((_width(b) for b in layers), default=0)
    canvas = [[" "] * w for _ in range(h)]
    for b in layers:
        for i, line in enumerate(b):
            for j, ch in enumerate(line):
                if ch != " ":
                    canvas[i][j] = ch
    return "\n".join("".join(r).rstrip() for r in canvas)


@fn("Item", area=AREA, status="added",
    notes="Item[expr, opts] inert: an entry of Grid, Column, ... with its own options (Alignment honoured in text)")
def _item(x: Expr, *opts: Option) -> Expr:
    rest, o = split_opts(opts, ITEM_OPTIONS, "Item")
    if rest:
        raise ValueError("Item called with too many arguments")
    if "Alignment" in o:
        a = o["Alignment"]
        _halign(a.args[0] if is_list(a) else a, "Item")
    raise Unevaluated()


@renders("Item")
def _r_item(e, c):
    return _txt(e.args[0], c) if e.args else None


@fn("Multicolumn", area=AREA, status="added",
    notes="Multicolumn[list, cols | {rows, Automatic} | {rows, cols}] evaluates to the corresponding Grid, filling "
          "the first column first (Appearance -> \"Horizontal\" fills by rows); Multicolumn[list] uses about "
          "Sqrt[n] columns; {rows, cols} drops elements beyond rows*cols")
def _multicolumn(x: Expr, *rest: Expr) -> Expr:
    a, o = split_opts(rest, GRID_OPTIONS, "Multicolumn")
    if not is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("Multicolumn: the first argument must be a list")
    items = list(x.args)
    n = len(items)
    if len(a) > 1:
        raise ValueError("Multicolumn called with too many arguments")
    spec = a[0] if a else None
    if spec is None or is_sym(spec, "Automatic"):
        cols = max(1, math.ceil(math.sqrt(n)))
        rows = max(1, math.ceil(n / cols))
    elif _is_int(spec) and int(spec) > 0:
        cols = int(spec)
        rows = max(1, math.ceil(n / cols))
    elif is_list(spec) and len(spec.args) == 2:
        r, cc = spec.args
        if _is_int(r) and int(r) > 0 and is_sym(cc, "Automatic"):
            rows = int(r)
            cols = max(1, math.ceil(n / rows))
        elif _is_int(r) and int(r) > 0 and _is_int(cc) and int(cc) > 0:
            rows, cols = int(r), int(cc)
        elif is_sym(r, "Automatic") and _is_int(cc) and int(cc) > 0:
            cols = int(cc)
            rows = max(1, math.ceil(n / cols))
        else:
            raise ValueError("Multicolumn: the shape must be cols, {rows, Automatic} or {rows, cols}")
    else:
        raise ValueError("Multicolumn: the shape must be cols, {rows, Automatic} or {rows, cols}")
    horizontal = False
    if "Appearance" in o:
        ap = o.pop("Appearance")
        if ap == "Horizontal":
            horizontal = True
        elif ap not in ("Vertical",) and not is_sym(ap, "Automatic"):
            raise ValueError("Multicolumn: Appearance must be \"Vertical\" or \"Horizontal\"")
    o.pop("Pagination", None)
    items = items[:rows * cols]
    grid = []
    for i in range(rows):
        row = []
        for j in range(cols):
            k = (i * cols + j) if horizontal else (j * rows + i)
            if k < len(items):
                row.append(items[k])
        if row:
            grid.append(L(*row))
    extra = [Expr("Rule", Symbol(k), v) for k, v in o.items()]
    return Expr("Grid", L(*grid), *extra)


# ── StringForm ──

def string_form_text(tmpl, args, c=None):
    if not isinstance(tmpl, str):
        raise ValueError("StringForm: the first argument must be a string")
    out, i, nxt = [], 0, 0
    while i < len(tmpl):
        ch = tmpl[i]
        if ch == "\\" and i + 1 < len(tmpl) and tmpl[i + 1] == "`":
            out.append("`")
            i += 2
            continue
        if ch == "`":
            j = tmpl.find("`", i + 1)
            if j < 0:
                raise ValueError(f"StringForm: unmatched backquote in {tmpl}")
            body = tmpl[i + 1:j]
            if body == "":
                nxt += 1
                k = nxt
            elif re.fullmatch(r"\d+", body):
                k = int(body)
                nxt = k
            else:
                raise ValueError(f"StringForm: `{body}` is not a valid placeholder (use `` or `n`)")
            if not 1 <= k <= len(args):
                raise ValueError(f"StringForm: item {k} requested in \"{tmpl}\" out of range; {len(args)} items "
                                 "available")
            out.append(_txt(args[k - 1], c or Ctx()))
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


@fn("StringForm", area=AREA, status="added",
    notes="StringForm[\"... `` ... `n` ...\", e1, e2, ...] inert: shows the template with `` replaced by successive "
          "ei and `n` by en (\\` is a literal backquote); out-of-range or unmatched placeholders raise")
def _stringform(tmpl: String, *args: Expr) -> Expr:
    if not isinstance(tmpl, str):
        if isinstance(tmpl, Symbol):
            raise Unevaluated()
        raise ValueError("StringForm: the first argument must be a string")
    string_form_text(tmpl, args)
    raise Unevaluated()


@renders("StringForm")
def _r_stringform(e, c):
    return string_form_text(e.args[0], e.args[1:], c)


# ── DisableFormatting, OutputForm, StandardForm, TraditionalForm, TeXForm, CForm, FortranForm, MathMLForm ──

@fn("DisableFormatting", area=AREA, status="added",
    notes="DisableFormatting[expr] gives expr as soon as it is evaluated (it only matters inside held expressions)")
def _disableformatting(x: Expr) -> Expr:
    return x


def _one_arg_form(name, renderer, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def impl(x: Expr, *opts: Option) -> Expr:
        if len(opts) > 0 and any(_opt_name(t) is None for t in opts):
            raise ValueError(f"{name} called with {1 + len(opts)} arguments; expected 1")
        raise Unevaluated()

    @renders(name)
    def r(e, c):
        if len(e.args) < 1:
            return None
        return renderer(e.args[0])
    return impl


_one_arg_form("OutputForm", lambda x: output_text(x),
              "OutputForm[expr] inert: shows expr in linear OutputForm text (strings without quotes)")
_one_arg_form("StandardForm", lambda x: output_text(x),
              "StandardForm[expr] inert: shows expr in standard (linear) text")
_one_arg_form("TraditionalForm", lambda x: traditional_text(x),
              "TraditionalForm[expr] inert: shows expr in traditional mathematical notation (f(x), sin(x), e^x, "
              "\u03c0, i, polynomials in decreasing degree)")
_one_arg_form("TeXForm", lambda x: tex_text(x),
              "TeXForm[expr] inert: shows the LaTeX of expr (TraditionalForm conventions: \\frac, \\sqrt, \\sin (x), "
              "x^2+1); ToString[TeXForm[expr]] gives the TeX string")
_one_arg_form("CForm", lambda x: c_text(x),
              "CForm[expr] inert: C source for expr (Power(x,2), Sin(x), rationals as decimals, 1.5e10)")
_one_arg_form("FortranForm", lambda x: fortran_text(x),
              "FortranForm[expr] inert: Fortran source for expr (x**2, Sin(x), rationals as decimals)")
_one_arg_form("MathMLForm", lambda x: mathml_text(x),
              "MathMLForm[expr] inert: presentation MathML for expr (<math> with mi, mn, mo, msup, mfrac, msqrt, "
              "mrow)")
for _h in ("OutputForm", "StandardForm", "TraditionalForm", "TeXForm", "CForm", "FortranForm", "MathMLForm"):
    WRAPPERS.add(_h)


# ── TextString ──

TEXTSTRING_OPTIONS = ("ListFormat", "MissingString", "BooleanStrings", "AssociationFormat", "ElidedForms",
                      "DateFormat", "TimeFormat", "TimeZone")


def _association_format(v):
    """AssociationFormat -> Automatic | "sep" | {"left", "sep", "right"} | {"left", "sep", "right", "arrow"}."""
    if is_sym(v, "Automatic"):
        return ("<|", ", ", "|>", " -> ")
    if isinstance(v, str):
        return ("", v, "", " -> ")
    if is_list(v) and len(v.args) in (3, 4) and all(isinstance(t, str) for t in v.args):
        return tuple(v.args) + ((" -> ",) if len(v.args) == 3 else ())
    raise ValueError("TextString: AssociationFormat must be Automatic, a separator string or "
                     "{\"left\", \"sep\", \"right\"[, \"arrow\"]}")


def text_string(x, opts):
    lf = None
    if "ListFormat" in opts:
        v = opts["ListFormat"]
        if is_sym(v, "Automatic"):
            lf = ("{", ", ", "}")
        elif is_list(v) and len(v.args) == 3 and all(isinstance(t, str) for t in v.args):
            lf = tuple(v.args)
        elif isinstance(v, str):
            lf = ("", v, "")
        else:
            raise ValueError("TextString: ListFormat must be Automatic or {\"left\", \"sep\", \"right\"}")
    else:
        lf = ("{", ", ", "}")
    ms = ""
    if "MissingString" in opts:
        v = opts["MissingString"]
        if isinstance(v, str):
            ms = v
        elif is_sym(v, "Automatic"):
            ms = "auto"
        elif is_sym(v, "None"):
            ms = None
        else:
            raise ValueError("TextString: MissingString must be a string, Automatic or None")
    bs = None
    if "BooleanStrings" in opts:
        v = opts["BooleanStrings"]
        if is_list(v) and len(v.args) == 2 and all(isinstance(t, str) for t in v.args):
            bs = tuple(v.args)
        else:
            raise ValueError("TextString: BooleanStrings must be a pair of strings {\"true\", \"false\"}")
    ctx = Ctx("text", list_format=lf, missing=ms, booleans=bs)
    ctx.assoc_format = _association_format(opts.get("AssociationFormat", Symbol("Automatic")))
    if isinstance(x, str):
        return x
    return output_text(x, ctx)


@fn("TextString", area=AREA, status="added",
    notes="TextString[expr, opts] human-readable string: strings unchanged, numbers in decimal notation (6 digits "
          "for machine reals, rationals as decimals), symbols by name, Missing per MissingString (default \"\"), lists "
          "per ListFormat, Row/Column/TableForm/Grid as text, otherwise InputForm; BooleanStrings")
def _textstring(x: Expr, *opts: Option) -> String:
    rest, o = split_opts(opts, TEXTSTRING_OPTIONS, "TextString")
    if rest:
        raise ValueError("TextString called with too many arguments")
    return text_string(x, o)


record("ListFormat", AREA, "added", "option of TextString: {\"left\", \"sep\", \"right\"} for lists (Automatic: "
                                    "{\"{\", \", \", \"}\"})")
record("AssociationFormat", AREA, "added", "option of TextString: Automatic ({\"<|\", \", \", \"|>\", \" -> \"}), "
                                          "a separator string or {\"left\", \"sep\", \"right\"[, \"arrow\"]} "
                                          "for associations; other values raise ValueError. Check: exact; "
                                          "reference strings in tests.")
record("MissingString", AREA, "added", "option of TextString: the string for Missing[...] (default \"\"; Automatic: "
                                       "the reason; None: unchanged)")


# ═════════════════════════════════════ typesetting: TeX, traditional text, C, Fortran, MathML ═════════════════

P_REL, P_SUM, P_PROD, P_POW, P_ATOM = 1, 2, 3, 4, 5

_GREEK_TEX = {g: "\\" + g.lower() for g in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta",
                                            "Iota", "Kappa", "Lambda", "Mu", "Nu", "Xi", "Rho", "Sigma", "Tau",
                                            "Upsilon", "Phi", "Chi", "Psi", "Omega"]}
_GREEK_TEX.update({"Capital" + g: "\\" + g for g in ["Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi", "Sigma",
                                                     "Upsilon", "Phi", "Psi", "Omega"]})
_TEX_CONST = {"Pi": "\\pi", "E": "e", "I": "i", "Infinity": "\\infty", "ComplexInfinity": "\\tilde{\\infty}",
              "Degree": "{}^{\\circ}", "GoldenRatio": "\\phi", "EulerGamma": "\\gamma", "Catalan": "C",
              "Indeterminate": "\\text{Indeterminate}", "Reals": "\\mathbb{R}", "Integers": "\\mathbb{Z}",
              "Complexes": "\\mathbb{C}", "Rationals": "\\mathbb{Q}", "Primes": "\\mathbb{P}", "True": "\\text{True}",
              "False": "\\text{False}", "Null": "\\text{Null}"}
_TEX_FUNCS = {"Sin": "\\sin", "Cos": "\\cos", "Tan": "\\tan", "Cot": "\\cot", "Sec": "\\sec", "Csc": "\\csc",
              "Sinh": "\\sinh", "Cosh": "\\cosh", "Tanh": "\\tanh", "Coth": "\\coth", "Sech": "\\text{sech}",
              "Csch": "\\text{csch}", "Log": "\\log", "Max": "\\max", "Min": "\\min", "Det": "\\det",
              "GCD": "\\gcd", "Arg": "\\arg", "Gamma": "\\Gamma", "Zeta": "\\zeta", "Erf": "\\text{erf}",
              "Erfc": "\\text{erfc}", "Re": "\\Re", "Im": "\\Im", "Sign": "\\text{sgn}", "Exp": "\\exp",
              "PolyGamma": "\\psi", "Beta": "B", "LCM": "\\text{lcm}", "Mod": "\\text{mod}",
              "EllipticK": "K", "EllipticE": "E", "LogGamma": "\\log \\Gamma", "Tr": "\\text{tr}",
              "DiracDelta": "\\delta", "HeavisideTheta": "\\theta", "KroneckerDelta": "\\delta",
              "PrimePi": "\\pi", "EulerPhi": "\\phi", "MoebiusMu": "\\mu", "DivisorSigma": "\\sigma",
              "Prime": "p", "Fibonacci": "F", "LucasL": "L", "HarmonicNumber": "H", "BernoulliB": "B",
              "EulerE": "E", "CatalanNumber": "C", "Pochhammer": "", "Hypergeometric2F1": "\\, _2F_1"}
_TEX_INV = {"ArcSin": "\\sin", "ArcCos": "\\cos", "ArcTan": "\\tan", "ArcCot": "\\cot", "ArcSec": "\\sec",
            "ArcCsc": "\\csc", "ArcSinh": "\\sinh", "ArcCosh": "\\cosh", "ArcTanh": "\\tanh", "ArcCoth": "\\coth",
            "ArcSech": "\\text{sech}", "ArcCsch": "\\text{csch}"}
_TEX_SUBFUNC = {"BesselJ": "J", "BesselY": "Y", "BesselI": "I", "BesselK": "K", "LegendreP": "P",
                "LegendreQ": "Q", "HermiteH": "H", "LaguerreL": "L", "ChebyshevT": "T", "ChebyshevU": "U",
                "StruveH": "\\pmb{H}", "StruveL": "\\pmb{L}", "HankelH1": "H^{(1)}", "HankelH2": "H^{(2)}",
                "PolyLog": "\\text{Li}", "SphericalBesselJ": "j", "SphericalBesselY": "y"}
_TEX_REL = {"Equal": "=", "Unequal": "\\neq ", "Less": "<", "Greater": ">", "LessEqual": "\\leq ",
            "GreaterEqual": "\\geq ", "SameQ": "===", "UnsameQ": "=!=", "Rule": "\\to ", "RuleDelayed": ":\\to ",
            "And": "\\land ", "Or": "\\lor ", "Implies": "\\Rightarrow ", "Equivalent": "\\Leftrightarrow ",
            "Xor": "\\veebar ", "Element": "\\in ", "NotElement": "\\notin ", "Set": "=", "SetDelayed": ":=",
            "Congruent": "\\equiv ", "Proportional": "\\propto ", "Union": "\\cup ", "Intersection": "\\cap ",
            "Subset": "\\subset ", "SubsetEqual": "\\subseteq ", "Superset": "\\supset ",
            "TwoWayRule": "\\leftrightarrow ", "Composition": "\\circ ", "Dot": ".", "Cross": "\\times ",
            "Distributed": "\\approx "}


def _is_num(x):
    return _is_int(x) or _is_real(x) or (isinstance(x, Expr) and F.rational_value(x) is not None)


def _num_neg(x):
    if _is_int(x) or _is_real(x):
        return x < 0
    q = F.rational_value(x) if isinstance(x, Expr) else None
    return q is not None and q < 0


def _degree(t):
    if isinstance(t, Symbol):
        return 0 if t.name in _TEX_CONST else 1
    if _is_num(t):
        return 0
    if isinstance(t, Expr):
        if t.head == "Times":
            return sum(_degree(a) for a in t.args)
        if t.head == "Power" and len(t.args) == 2:
            ex = t.args[1]
            q = F.rational_value(ex) if isinstance(ex, Expr) else (Fraction(int(ex)) if _is_int(ex) else None)
            if q is None and _is_real(ex):
                q = Fraction(float(ex))
            if q is not None:
                return _degree(t.args[0]) * q
            return 1
    return 0


_NUMERIC_CONSTANTS = {"Pi", "E", "EulerGamma", "GoldenRatio", "Catalan", "Degree", "Khinchin", "Glaisher"}


def _numeric_q(t):
    if _is_num(t):
        return True
    if isinstance(t, Symbol):
        return t.name in _NUMERIC_CONSTANTS
    if isinstance(t, Expr) and t.head in ("Times", "Power", "Sqrt") and isinstance(t.head, str):
        return all(_numeric_q(a) for a in t.args)
    return False


def _single_var(t):
    """The variable of a monomial c x^k in one symbol, or None."""
    if isinstance(t, Symbol) and not _numeric_q(t):
        return t.name
    if isinstance(t, Expr) and t.head == "Power" and len(t.args) == 2 and _is_num(t.args[1]):
        return _single_var(t.args[0])
    if isinstance(t, Expr) and t.head == "Times":
        vs = {_single_var(a) for a in t.args if not _numeric_q(a)}
        return vs.pop() if len(vs) == 1 and None not in vs else None
    return None


def traditional_order(args):
    """Wolfram TraditionalForm term order: numeric terms last; a polynomial in one variable in decreasing degree;
    otherwise the standard order."""
    terms = list(args)
    nums = [t for t in terms if _numeric_q(t)]
    rest = [t for t in terms if not _numeric_q(t)]
    vs = {_single_var(t) for t in rest}
    if len(vs) == 1 and None not in vs:
        rest.sort(key=lambda t: -_degree(t))
    return rest + nums


def _split_times(e):
    """(sign, numerator factors, denominator factors) of a product (rational coefficients split)."""
    factors = []
    for a in (e.args if isinstance(e, Expr) and e.head == "Times" else [e]):
        if isinstance(a, Expr) and a.head == "Times" and F.rational_value(a) is None:
            factors.extend(a.args)
        else:
            factors.append(a)
    sign = 1
    num, den = [], []
    for f in factors:
        q = F.rational_value(f) if not _is_real(f) else None
        if q is not None:
            if q < 0:
                sign = -sign
                q = -q
            if q.numerator != 1:
                num.insert(0, q.numerator)
            if q.denominator != 1:
                den.insert(0, q.denominator)
            continue
        if _is_real(f) and f < 0:
            sign = -sign
            f = -f
            if f == 1.0:
                num.insert(0, f)
                continue
        if isinstance(f, Expr) and f.head == "Power" and len(f.args) == 2 and _num_neg(f.args[1]):
            b, ex = f.args
            nex = _negate_num(ex)
            den.append(b if (_is_int(nex) and int(nex) == 1) else Expr("Power", b, nex))
            continue
        num.append(f)
    return sign, num, den


def _negate_num(x):
    if _is_int(x) or _is_real(x):
        return -x
    q = F.rational_value(x)
    return F._rational_expr(-q)


class TeX:
    def __init__(self):
        pass

    def __call__(self, e):
        return self.tex(e)[0]

    def wrap(self, tp, prec):
        t, p = tp
        return f"\\left({t}\\right)" if p < prec else t

    def number(self, x):
        if _is_int(x):
            return str(int(x)), (P_ATOM if x >= 0 else P_SUM)
        s = _default_real_text(abs(x)) if _is_real(x) else None
        if s is not None:
            if "\u00d710^" in s:
                m, ex = s.split("\u00d710^")
                s = f"{m}\\times 10^{{{ex}}}"
                pr = P_PROD
            else:
                pr = P_ATOM
            if x < 0:
                return "-" + s, P_SUM
            return s, pr
        return str(x), P_ATOM

    def symbol(self, name):
        if name in _TEX_CONST:
            return _TEX_CONST[name]
        if name in _GREEK_TEX:
            return _GREEK_TEX[name]
        if len(name) == 1:
            return name
        return "\\text{" + name + "}"

    def args(self, a):
        return ",".join(self.tex(x)[0] for x in a)

    def call(self, f, a):
        return f"{f}({self.args(a)})", P_ATOM

    def tex(self, e):
        if e is None:
            return "\\text{Null}", P_ATOM
        if e is True or e is False:
            return "\\text{" + ("True" if e else "False") + "}", P_ATOM
        if isinstance(e, str):
            return "\\text{" + _tex_escape(e) + "}", P_ATOM
        if _is_int(e) or _is_real(e):
            return self.number(e)
        if isinstance(e, Symbol):
            return self.symbol(e.name), P_ATOM
        if not isinstance(e, Expr):
            return str(e), P_ATOM
        q = F.rational_value(e)
        if q is not None and not F.is_int(e):
            t = f"\\frac{{{abs(q.numerator)}}}{{{q.denominator}}}"
            return ("-" + t, P_SUM) if q < 0 else (t, P_PROD)
        h, a = e.head, e.args
        if isinstance(h, Expr):
            if h.head == "Derivative" and len(h.args) == 1 and _is_int(h.args[0]) and len(a) == 1:
                k = int(h.args[0])
                fname = self.tex(a[0])[0]
                return (fname + "'" * k if k <= 3 else f"{fname}^{{({k})}}"), P_ATOM
            if isinstance(h.head, Expr) and h.head.head == "Derivative" and len(h.head.args) == 1 and \
                    _is_int(h.head.args[0]) and len(h.args) == 1:
                k = int(h.head.args[0])
                fname = self.tex(h.args[0])[0]
                return f"{fname}{chr(39) * k if k <= 3 else '^{(' + str(k) + ')}'}({self.args(a)})", P_ATOM
            return f"{self.tex(h)[0]}({self.args(a)})", P_ATOM
        if h in RENDER_TEX:
            return RENDER_TEX[h](self, e)
        if h == "Plus":
            terms = traditional_order(a)
            out = ""
            for i, t in enumerate(terms):
                s, p = self.tex(t)
                if i == 0:
                    out = s
                elif s.startswith("-"):
                    out += s
                else:
                    out += "+" + s
            return out, P_SUM
        if h == "Times":
            return self.times(e)
        if h == "Power" and len(a) == 2:
            return self.power(a[0], a[1])
        if h == "List":
            return "\\{" + self.args(a) + "\\}", P_ATOM
        if h == "Association":
            return "\\langle |" + ",".join(self.tex(x)[0] for x in a) + "| \\rangle", P_ATOM
        if h in _TEX_REL and len(a) >= 2:
            return _TEX_REL[h].join(self.wrap(self.tex(x), P_SUM) for x in a), P_REL
        if h == "Not" and len(a) == 1:
            return "\\neg " + self.wrap(self.tex(a[0]), P_ATOM), P_PROD
        if h == "Sqrt" and len(a) == 1:
            return f"\\sqrt{{{self.tex(a[0])[0]}}}", P_ATOM
        if h == "Abs" and len(a) == 1:
            return f"\\left| {self.tex(a[0])[0]}\\right| ", P_ATOM
        if h == "Norm" and len(a) == 1:
            return f"\\left\\| {self.tex(a[0])[0]}\\right\\| ", P_ATOM
        if h == "Floor" and len(a) == 1:
            return f"\\lfloor {self.tex(a[0])[0]}\\rfloor ", P_ATOM
        if h == "Ceiling" and len(a) == 1:
            return f"\\lceil {self.tex(a[0])[0]}\\rceil ", P_ATOM
        if h == "Factorial" and len(a) == 1:
            return self.wrap(self.tex(a[0]), P_ATOM) + "!", P_POW
        if h == "Factorial2" and len(a) == 1:
            return self.wrap(self.tex(a[0]), P_ATOM) + "!!", P_POW
        if h == "Binomial" and len(a) == 2:
            return f"\\binom{{{self.tex(a[0])[0]}}}{{{self.tex(a[1])[0]}}}", P_ATOM
        if h == "Conjugate" and len(a) == 1:
            return self.wrap(self.tex(a[0]), P_ATOM) + "^*", P_POW
        if h == "Subscript" and len(a) >= 2:
            sub = self.args(a[1:])
            return f"{self.wrap(self.tex(a[0]), P_ATOM)}_{sub if len(sub) == 1 else '{' + sub + '}'}", P_ATOM
        if h == "Superscript" and len(a) == 2:
            return f"{self.wrap(self.tex(a[0]), P_ATOM)}^{{{self.tex(a[1])[0]}}}", P_POW
        if h == "Subsuperscript" and len(a) == 3:
            b_, c_ = self.tex(a[1])[0], self.tex(a[2])[0]
            b_ = b_ if len(b_) == 1 else "{" + b_ + "}"
            c_ = c_ if len(c_) == 1 else "{" + c_ + "}"
            return f"{self.wrap(self.tex(a[0]), P_ATOM)}_{b_}^{c_}", P_POW
        if h in ("Integrate", "Sum", "Product") and len(a) >= 2:
            return self.bigop(h, a)
        if h == "Limit" and len(a) == 2 and isinstance(a[1], Expr) and a[1].head == "Rule":
            v, pt = a[1].args
            return (f"\\underset{{{self.tex(v)[0]}\\to {self.tex(pt)[0]}}}{{\\text{{lim}}}}"
                    f"{self.wrap(self.tex(a[0]), P_PROD)}"), P_PROD
        if h == "D" and len(a) == 2:
            return f"\\frac{{\\partial {self.wrap(self.tex(a[0]), P_POW)}}}{{\\partial {self.tex(a[1])[0]}}}", P_PROD
        if h in _TEX_INV and len(a) == 1:
            return f"{_TEX_INV[h]} ^{{-1}}({self.tex(a[0])[0]})", P_ATOM
        if h in _TEX_SUBFUNC and len(a) >= 2:
            return f"{_TEX_SUBFUNC[h]}_{{{self.args(a[:-1])}}}({self.tex(a[-1])[0]})", P_ATOM
        if h == "Log" and len(a) == 2:
            return f"\\log _{{{self.tex(a[0])[0]}}}({self.tex(a[1])[0]})", P_ATOM
        if h in _TEX_FUNCS:
            f = _TEX_FUNCS[h]
            return f"{f} ({self.args(a)})" if f.startswith("\\") and f[1:].isalpha() and f not in (
                "\\Gamma", "\\zeta", "\\psi", "\\delta", "\\theta", "\\pi", "\\phi", "\\mu", "\\sigma", "\\Re",
                "\\Im") else f"{f}({self.args(a)})", P_ATOM
        if h in ("HoldForm", "Style", "Item", "DisableFormatting", "Text", "Placed", "Framed", "OutputForm",
                 "StandardForm", "TraditionalForm", "TeXForm", "Labeled", "Short", "InputForm") and a:
            inner = a[0]
            if h in ("InputForm",):
                return "\\text{" + _tex_escape(F.input_form(inner)) + "}", P_ATOM
            return self.tex(inner)
        if h == "Row" and a and is_list(a[0]):
            sep = self.tex(a[1])[0] if len(a) > 1 else ""
            return sep.join(self.tex(x)[0] for x in a[0].args), P_ATOM
        if h in ("Grid", "TableForm", "MatrixForm", "Column") and a and is_list(a[0]):
            rows = [list(r.args) if is_list(r) else [r] for r in a[0].args]
            if h == "Column":
                rows = [[r] for r in a[0].args]
            ncol = max((len(r) for r in rows), default=1)
            body = "".join(" " + " & ".join(self.tex(x)[0] for x in r) + " \\\\\n" for r in rows)
            arr = f"\\begin{{array}}{{{'c' * ncol}}}\n{body}\\end{{array}}"
            if h == "MatrixForm":
                arr = "\\left(\n" + arr + "\n\\right)"
            return arr, P_ATOM
        if h in NUMBER_FORMS or h == "BaseForm":
            t = RENDER[h](e, Ctx())
            t = re.sub(r"\u00d710\^(-?\d+)", lambda m: f"\\times 10^{{{m.group(1)}}}", t)
            return t, P_ATOM
        if h in ("StringForm",):
            return "\\text{" + _tex_escape(RENDER[h](e, Ctx())) + "}", P_ATOM
        fname = self.symbol(h) if h not in _TEX_CONST else "\\text{" + h + "}"
        return self.call(fname, a)

    def times(self, e):
        sign, num, den = _split_times(e)
        nt = self.product(num, in_frac=bool(den)) if num else ("1", P_ATOM)
        if den:
            dt = self.product(den, in_frac=True)
            t = f"\\frac{{{nt[0]}}}{{{dt[0]}}}"
            p = P_PROD
        else:
            t, p = nt
        if sign < 0:
            return "-" + (t if p >= P_PROD else f"\\left({t}\\right)"), P_SUM
        return t, p

    def product(self, fs, in_frac=False):
        parts = []
        for i, f in enumerate(fs):
            s, p = self.tex(f)
            if len(fs) == 1 and in_frac:
                parts.append(s)
                continue
            if p < P_PROD or (p == P_SUM):
                s = f"\\left({s}\\right)"
            if i and _is_num(f) and parts:
                parts.append("\\times " + s)
            else:
                parts.append(s)
        if len(parts) == 1:
            return parts[0], self.tex(fs[0])[1] if len(fs) == 1 else P_PROD
        return " ".join(parts).replace(" \\times ", " \\times "), P_PROD

    def power(self, b, x):
        q = F.rational_value(x) if isinstance(x, Expr) else (Fraction(int(x)) if _is_int(x) else None)
        if q is not None and q < 0:
            return self.times(Expr("Times", Expr("Power", b, x)))
        if q is not None and q.numerator == 1 and q.denominator == 2:
            return f"\\sqrt{{{self.tex(b)[0]}}}", P_ATOM
        if q is not None and q.numerator == 1 and q.denominator > 2:
            return f"\\sqrt[{q.denominator}]{{{self.tex(b)[0]}}}", P_ATOM
        if isinstance(x, Expr) and _num_neg_expr(x):
            return self.times(Expr("Times", Expr("Power", b, x)))
        xs = self.tex(x)[0]
        ex = xs if len(xs) == 1 else "{" + xs + "}"
        if isinstance(b, Expr) and isinstance(b.head, str) and len(b.args) == 1 and \
                (b.head in _TEX_FUNCS and _TEX_FUNCS[b.head] in ("\\sin", "\\cos", "\\tan", "\\cot", "\\sec",
                                                                 "\\csc", "\\sinh", "\\cosh", "\\tanh", "\\coth",
                                                                 "\\log")):
            return f"{_TEX_FUNCS[b.head]} ^{ex}({self.tex(b.args[0])[0]})", P_ATOM
        bt = self.tex(b)
        bs = bt[0] if bt[1] >= P_ATOM and not bt[0].startswith("-") else f"\\left({bt[0]}\\right)"
        if bt[1] >= P_ATOM and bs.endswith("'"):
            bs = "{" + bs + "}"
        return f"{bs}^{ex}", P_POW

    def bigop(self, h, a):
        body = a[0]
        specs = a[1:]
        sym = {"Integrate": "\\int", "Sum": "\\sum", "Product": "\\prod"}[h]
        inner = self.wrap(self.tex(body), P_SUM if h == "Integrate" else P_PROD)
        if h == "Integrate":
            pre, post = "", ""
            for s in specs:
                if is_list(s) and len(s.args) == 3:
                    pre += f"\\int_{{{self.tex(s.args[1])[0]}}}^{{{self.tex(s.args[2])[0]}}} "
                    post += f" \\, d{self.tex(s.args[0])[0]}"
                else:
                    pre += "\\int "
                    post += f" \\, d{self.tex(s)[0]}"
            pre = re.sub(r"_\{(.)\}", r"_\1", pre)
            pre = re.sub(r"\^\{(.)\}", r"^\1", pre)
            return f"{pre}{inner}{post}", P_SUM
        pre = ""
        for s in specs:
            if is_list(s) and len(s.args) == 3:
                lo = f"{self.tex(s.args[0])[0]}={self.tex(s.args[1])[0]}"
                hi = self.tex(s.args[2])[0]
                hi = hi if len(hi) == 1 else "{" + hi + "}"
                pre += f"{sym} _{{{lo}}}^{hi} "
            elif is_list(s) and len(s.args) == 2:
                hi = self.tex(s.args[1])[0]
                hi = hi if len(hi) == 1 else "{" + hi + "}"
                pre += f"{sym} _{{{self.tex(s.args[0])[0]}=1}}^{hi} "
            else:
                pre += f"{sym} _{{{self.tex(s)[0]}}} "
        return f"{pre}{inner}", P_SUM


def _num_neg_expr(x):
    """True for a product with a negative numeric coefficient (x^(-n y) is written as a fraction)."""
    if isinstance(x, Expr) and x.head == "Times" and x.args and _is_num(x.args[0]):
        return _num_neg(x.args[0])
    return False


def _tex_escape(s):
    rep = {"\\": "\\backslash ", "{": "\\{", "}": "\\}", "$": "\\$", "&": "\\&", "%": "\\%", "#": "\\#",
           "_": "\\_", "^": "\\^{}", "~": "\\~{}"}
    return "".join(rep.get(ch, ch) for ch in s)


RENDER_TEX = {}


def tex_text(e):
    return TeX()(e)


# ── traditional linear text ──

_TRAD_CONST = {"Pi": "\u03c0", "E": "e", "I": "i", "Infinity": "\u221e", "Degree": "\u00b0",
               "GoldenRatio": "\u03d5", "EulerGamma": "\u03b3"}
_TRAD_FUNCS = {"Sin": "sin", "Cos": "cos", "Tan": "tan", "Cot": "cot", "Sec": "sec", "Csc": "csc",
               "Sinh": "sinh", "Cosh": "cosh", "Tanh": "tanh", "Coth": "coth", "Log": "log", "ArcSin": "sin^-1",
               "ArcCos": "cos^-1", "ArcTan": "tan^-1", "Gamma": "\u0393", "Zeta": "\u03b6", "Max": "max",
               "Min": "min", "Exp": "exp", "Erf": "erf", "Re": "Re", "Im": "Im", "Arg": "arg", "Sign": "sgn"}


def traditional_text(e):
    def t(x, parent=0):
        if isinstance(x, str):
            return x
        if x is True or x is False:
            return "True" if x else "False"
        if _is_int(x) or _is_real(x):
            s = str(int(x)) if _is_int(x) else _default_real_text(x)
            return f"({s})" if s.startswith("-") and parent >= P_PROD else s
        if isinstance(x, Symbol):
            if x.name in _TRAD_CONST:
                return _TRAD_CONST[x.name]
            if x.name in _GREEK_TEX:
                return chr(0x3b1 + _GREEK.index(x.name.replace("Capital", ""))) if not x.name.startswith(
                    "Capital") else chr(0x391 + _GREEK.index(x.name[7:]))
            return x.name
        if not isinstance(x, Expr):
            return str(x)
        q = F.rational_value(x)
        if q is not None and not F.is_int(x):
            s = f"{abs(q.numerator)}/{q.denominator}"
            s = "-" + s if q < 0 else s
            return f"({s})" if parent >= P_PROD else s
        h, a = x.head, x.args
        if isinstance(h, str) and h in RENDER and h not in ("TraditionalForm",):
            return output_text(x)
        if h == "Plus":
            terms = traditional_order(a)
            out = ""
            for i, term in enumerate(terms):
                s = t(term, P_SUM)
                if i == 0:
                    out = s
                elif s.startswith("-"):
                    out += " - " + s[1:]
                else:
                    out += " + " + s
            return f"({out})" if parent > P_SUM else out
        if h == "Times":
            sign, num, den = _split_times(x)
            ns = " ".join(t(f, P_PROD) for f in num) if num else "1"
            if den:
                ds = " ".join(t(f, P_PROD) for f in den)
                if len(den) > 1:
                    ds = f"({ds})"
                if len(num) > 1:
                    ns = f"({ns})"
                s = f"{ns}/{ds}"
            else:
                s = ns
            if sign < 0:
                s = "-" + s
            return f"({s})" if parent > P_PROD or (sign < 0 and parent >= P_PROD) else s
        if h == "Power" and len(a) == 2:
            b, ex = a
            qq = F.rational_value(ex) if isinstance(ex, Expr) else (Fraction(int(ex)) if _is_int(ex) else None)
            if qq is not None and qq < 0:
                return t(Expr("Times", x), parent)
            if qq == Fraction(1, 2):
                return f"\u221a({t(b)})"
            if isinstance(b, Symbol) and b.name == "E":
                return f"e^{t(ex, P_ATOM)}"
            return f"{t(b, P_ATOM)}^{t(ex, P_ATOM)}"
        if h == "List":
            return "{" + ", ".join(t(y) for y in a) + "}"
        if h in ("Equal", "Less", "Greater", "LessEqual", "GreaterEqual", "Unequal", "Rule", "And", "Or"):
            op = {"Equal": "=", "Less": "<", "Greater": ">", "LessEqual": "\u2264", "GreaterEqual": "\u2265",
                  "Unequal": "\u2260", "Rule": "\u2192", "And": "\u2227", "Or": "\u2228"}[h]
            return f" {op} ".join(t(y, P_REL + 1) for y in a)
        if h == "Abs" and len(a) == 1:
            return f"|{t(a[0])}|"
        if h == "Factorial" and len(a) == 1:
            return t(a[0], P_ATOM) + "!"
        if isinstance(h, str):
            name = _TRAD_FUNCS.get(h, h)
            return f"{name}(" + ", ".join(t(y) for y in a) + ")"
        if isinstance(h, Expr) and h.head == "Derivative" and len(h.args) == 1 and _is_int(h.args[0]):
            return t(a[0]) + "'" * int(h.args[0]) if len(a) == 1 else output_text(x)
        return t(h) + "(" + ", ".join(t(y) for y in a) + ")"
    return t(e)


# ── C and Fortran ──

def _c_real(x, exp_char="e"):
    s = _wolfram_real_input(abs(x), False)
    s = s.replace("*^", exp_char)
    return ("-" if x < 0 else "") + s


def _c_like(e, lang):
    pw = (lambda b, x: f"Power({b},{x})") if lang == "C" else None

    def t(x, parent=0):
        if isinstance(x, str):
            return '"' + x.replace("\\", "\\\\").replace('"', '\\"') + '"'
        if x is True or x is False:
            return "True" if x else "False"
        if _is_int(x):
            s = str(int(x))
            return f"({s})" if s.startswith("-") and parent > P_SUM else s
        if _is_real(x):
            s = _c_real(x, "e" if lang == "C" else "e")
            return f"({s})" if s.startswith("-") and parent > P_SUM else s
        if isinstance(x, Symbol):
            return x.name
        if not isinstance(x, Expr):
            return str(x)
        q = F.rational_value(x)
        if q is not None and not F.is_int(x):
            s = _c_real(float(q))
            return f"({s})" if s.startswith("-") and parent > P_SUM else s
        h, a = x.head, x.args
        if h == "Plus":
            out = ""
            for i, term in enumerate(a):
                s = t(term, P_SUM)
                if i == 0:
                    out = s
                elif s.startswith("-"):
                    out += " - " + s[1:]
                else:
                    out += " + " + s
            return f"({out})" if parent > P_SUM else out
        if h == "Times":
            sign, num, den = _split_times(x)
            numx = list(num)
            ns = "*".join(t(f, P_PROD) for f in numx) if numx else "1"
            if den:
                dints = [d for d in den if _is_int(d)]
                dother = [d for d in den if not _is_int(d)]
                if dints and not dother:
                    dv = 1
                    for d in dints:
                        dv *= int(d)
                    ds = f"{dv}."
                else:
                    ds = "*".join(t(f, P_PROD) for f in den)
                    if len(den) > 1:
                        ds = f"({ds})"
                if len(numx) > 1:
                    ns = f"({ns})"
                s = f"{ns}/{ds}"
            else:
                s = ns
            if sign < 0:
                s = "-" + s
                return f"({s})" if parent > P_SUM else s
            return f"({s})" if parent > P_PROD else s
        if h == "Power" and len(a) == 2:
            b, ex = a
            qq = F.rational_value(ex) if isinstance(ex, Expr) else (Fraction(int(ex)) if _is_int(ex) else None)
            if qq == -1:
                return t(Expr("Times", x), parent)
            if qq == Fraction(1, 2):
                return f"Sqrt({t(b)})"
            if qq == Fraction(-1, 2):
                return f"1/Sqrt({t(b)})" if parent <= P_PROD else f"(1/Sqrt({t(b)}))"
            if lang == "C":
                return pw(t(b), t(ex))
            es = t(ex, P_POW + 1)
            if es.startswith("-"):
                es = f"({es})"
            return f"{t(b, P_POW + 1)}**{es}"
        if h == "List":
            return "List(" + ",".join(t(y) for y in a) + ")"
        ops = {"Equal": "==", "Unequal": "!=", "Less": "<", "Greater": ">", "LessEqual": "<=",
               "GreaterEqual": ">=", "And": "&&", "Or": "||"}
        if lang == "Fortran":
            ops.update({"Unequal": "/=", "And": ".AND.", "Or": ".OR."})
        if h in ops and len(a) >= 2:
            return f" {ops[h]} ".join(t(y, P_REL + 1) for y in a)
        if h == "Not" and len(a) == 1:
            return ("!" if lang == "C" else ".NOT.") + t(a[0], P_ATOM)
        if h == "Rule" and len(a) == 2:
            return f"Rule({t(a[0])},{t(a[1])})"
        if isinstance(h, str):
            return f"{h}(" + ",".join(t(y) for y in a) + ")"
        return t(h) + "(" + ",".join(t(y) for y in a) + ")"
    return t(e)


def c_text(e):
    return _c_like(e, "C")


def fortran_text(e):
    return _c_like(e, "Fortran")


# ── MathML ──

def _xml(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_MML_CONST = {"Pi": "&#960;", "E": "&#8519;", "I": "&#8520;", "Infinity": "&#8734;", "Degree": "&#176;"}


def mathml_text(e):
    def m(x):
        if isinstance(x, str):
            return f"<ms>{_xml(x)}</ms>"
        if x is True or x is False:
            return f"<mi>{'True' if x else 'False'}</mi>"
        if _is_int(x) or _is_real(x):
            neg = x < 0
            s = str(abs(int(x))) if _is_int(x) else _default_real_text(abs(x))
            if "\u00d710^" in s:
                mm, ex = s.split("\u00d710^")
                core = f"<mrow><mn>{mm}</mn><mo>&#215;</mo><msup><mn>10</mn><mn>{ex}</mn></msup></mrow>"
            else:
                core = f"<mn>{s}</mn>"
            return f"<mrow><mo>-</mo>{core}</mrow>" if neg else core
        if isinstance(x, Symbol):
            return f"<mi>{_MML_CONST.get(x.name, _xml(x.name))}</mi>"
        if not isinstance(x, Expr):
            return f"<mi>{_xml(str(x))}</mi>"
        q = F.rational_value(x)
        if q is not None and not F.is_int(x):
            core = f"<mfrac><mn>{abs(q.numerator)}</mn><mn>{q.denominator}</mn></mfrac>"
            return f"<mrow><mo>-</mo>{core}</mrow>" if q < 0 else core
        h, a = x.head, x.args
        if h == "Plus":
            parts = []
            for i, term in enumerate(traditional_order(a)):
                if i and _neg_term(term):
                    parts.append("<mo>-</mo>" + m(_abs_term(term)))
                else:
                    parts.append(("<mo>+</mo>" if i else "") + m(term))
            return "<mrow>" + "".join(parts) + "</mrow>"
        if h == "Times":
            sign, num, den = _split_times(x)
            def prod(fs):
                inner = "<mo>&#8290;</mo>".join(_mml_paren(m(f), f) for f in fs)
                return inner if len(fs) == 1 else f"<mrow>{inner}</mrow>"
            ns = prod(num) if num else "<mn>1</mn>"
            core = f"<mfrac>{ns}{prod(den)}</mfrac>" if den else ns
            return f"<mrow><mo>-</mo>{core}</mrow>" if sign < 0 else core
        if h == "Power" and len(a) == 2:
            b, ex = a
            qq = F.rational_value(ex) if isinstance(ex, Expr) else (Fraction(int(ex)) if _is_int(ex) else None)
            if qq is not None and qq < 0:
                return m(Expr("Times", x))
            if qq == Fraction(1, 2):
                return f"<msqrt>{m(b)}</msqrt>"
            if qq is not None and qq.numerator == 1 and qq.denominator > 2:
                return f"<mroot>{m(b)}<mn>{qq.denominator}</mn></mroot>"
            return f"<msup>{_mml_paren(m(b), b, power=True)}{m(ex)}</msup>"
        if h == "Sqrt" and len(a) == 1:
            return f"<msqrt>{m(a[0])}</msqrt>"
        if h == "List":
            return "<mrow><mo>{</mo>" + "<mo>,</mo>".join(m(y) for y in a) + "<mo>}</mo></mrow>"
        rel = {"Equal": "=", "Unequal": "&#8800;", "Less": "&lt;", "Greater": "&gt;", "LessEqual": "&#8804;",
               "GreaterEqual": "&#8805;", "Rule": "&#8594;", "And": "&#8743;", "Or": "&#8744;"}
        if h in rel and len(a) >= 2:
            return "<mrow>" + f"<mo>{rel[h]}</mo>".join(m(y) for y in a) + "</mrow>"
        if h in ("Grid", "TableForm", "MatrixForm") and a and is_list(a[0]):
            rows = [list(r.args) if is_list(r) else [r] for r in a[0].args]
            tab = "<mtable>" + "".join("<mtr>" + "".join(f"<mtd>{m(c_)}</mtd>" for c_ in r) + "</mtr>"
                                       for r in rows) + "</mtable>"
            return f"<mrow><mo>(</mo>{tab}<mo>)</mo></mrow>" if h == "MatrixForm" else tab
        if h in ("Style", "HoldForm", "Item", "Text", "Framed", "TraditionalForm", "StandardForm") and a:
            return m(a[0])
        if isinstance(h, str) and h in RENDER:
            return f"<mtext>{_xml(output_text(x))}</mtext>"
        fname = _TRAD_FUNCS.get(h, h) if isinstance(h, str) else None
        fm = f"<mi>{_xml(fname)}</mi>" if fname is not None else m(h)
        return (f"<mrow>{fm}<mo>&#8289;</mo><mrow><mo>(</mo>" + "<mo>,</mo>".join(m(y) for y in a)
                + "<mo>)</mo></mrow></mrow>")
    return "<math xmlns='http://www.w3.org/1998/Math/MathML'>" + m(e) + "</math>"


def _neg_term(t):
    if _is_num(t):
        return _num_neg(t)
    return isinstance(t, Expr) and t.head == "Times" and t.args and _is_num(t.args[0]) and _num_neg(t.args[0])


def _abs_term(t):
    if _is_num(t):
        return _negate_num(t)
    c = _negate_num(t.args[0])
    rest = t.args[1:]
    if _is_int(c) and int(c) == 1:
        return rest[0] if len(rest) == 1 else Expr("Times", *rest)
    return Expr("Times", c, *rest)


def _mml_paren(s, x, power=False):
    if isinstance(x, Expr) and (x.head == "Plus" or (power and x.head in ("Times", "Power"))):
        return f"<mrow><mo>(</mo>{s}<mo>)</mo></mrow>"
    return s


# ═════════════════════════════════════ inert option and position symbols, system values ═══════════════════

_INERT = {
    # positions and alignments
    "Left": "position/alignment symbol: Column/Grid/Item alignment, Labeled position (label to the left)",
    "Right": "position/alignment symbol: Column/Grid/Item alignment, Labeled position (label to the right)",
    "Center": "position/alignment symbol: centred alignment in Column, Grid, Item, TableForm",
    "Top": "position/alignment symbol: Labeled position (label above), vertical alignment in Grid rows",
    "Bottom": "position/alignment symbol: Labeled position (default, label below), vertical alignment",
    "Above": "position symbol: Labeled/Placed position, same as Top",
    "Below": "position symbol: Labeled/Placed position, same as Bottom",
    "Before": "position symbol: Labeled/Placed position, same as Left",
    "After": "position symbol: Labeled/Placed position, same as Right",
    "Baseline": "alignment symbol: vertical alignment on the text baseline (top line in plain text)",
    # style directives and options
    "Bold": "style directive for Style (FontWeight -> \"Bold\"); no effect on plain text",
    "Italic": "style directive for Style (FontSlant -> \"Italic\"); no effect on plain text",
    "Background": "option of Style, Grid, Framed, Item, ...: background colour; no effect on plain text",
    "FontFamily": "option of Style: font family name (a string); validated, no effect on plain text",
    "FontSize": "option of Style: font size (positive number, Scaled[s], Tiny/Small/Medium/Large); validated, "
                "no effect on plain text",
    "ScriptLevel": "option of Style: script nesting level (non-negative integer); validated, no effect on text",
    "Visible": "option of Style and notebooks: whether shown (True/False); validated",
    # layout options honoured in text
    "Alignment": "option of Column, Grid, Item, Multicolumn: Left, Center, Right or {h, v} (honoured in text)",
    "ColumnAlignments": "option of Grid (GridBox): per-column horizontal alignment (honoured in text)",
    "ColumnLines": "option of Grid (GridBox): draw | between columns (honoured in text)",
    "ColumnSpacings": "option of Grid (GridBox): spaces between columns (honoured in text)",
    "ColumnWidths": "option of Grid (GridBox): minimum column widths in characters (honoured in text)",
    "RowAlignments": "option of Grid (GridBox): vertical alignment of multi-line rows (honoured in text)",
    "RowLines": "option of Grid (GridBox): draw a line between rows (honoured in text)",
    "RowSpacings": "option of Grid (GridBox): blank lines between rows (honoured in text)",
    "RowMinHeight": "option of Grid (GridBox): minimum row height in lines (honoured in text)",
    "TableAlignments": "option of TableForm: Left, Center, Right (honoured in text)",
    "TableDepth": "option of TableForm: levels laid out as a table; deeper levels shown as lists",
    "TableDirections": "option of TableForm: Column/Row for successive dimensions (Column: first dimension down)",
    "TableHeadings": "option of TableForm: {row headings, column headings} (None, Automatic or lists)",
    "TableSpacing": "option of TableForm: {blank lines between rows, spaces between columns} (default {1, 3})",
    "Pagination": "option of Multicolumn/GalleryView: items per page (accepted; plain text shows every item)",
    # front-end / Dataset options accepted but not used by plain text
    "AlignmentPoint": "option of Inset and related: alignment point of an object (inert)",
    "BaselinePosition": "option of Grid, Row, Column, Framed, ...: baseline for alignment with text (accepted)",
    "GridDefaultElement": "option of GridBox: element inserted for a new grid entry (accepted)",
    "ItemAspectRatio": "option of GraphicsGrid: aspect ratio of each item's region (inert)",
    "HeaderAlignment": "option of Dataset: alignment of headers (inert)",
    "HeaderBackground": "option of Dataset: header background colour (inert)",
    "HeaderDisplayFunction": "option of Dataset: function applied to headers before display (inert)",
    "HeaderLines": "option of SemanticImport and related import functions: number of header rows (inert)",
    "HeaderSize": "option of Dataset: widths and heights of headers (inert)",
    "HeaderStyle": "option of Dataset: style of headers (inert)",
    "HiddenItems": "option of Dataset: positions of items to hide (inert)",
    "ItemDisplayFunction": "option of Dataset: function applied to items before display (inert)",
    "MissingStyle": "option of GeoRegionValuePlot and related: style for missing data (inert)",
    "Delimiters": "option of various functions: the delimiters to use or look for (inert)",
    "SurdForm": "option of RadicalBox/SqrtBox: whether the radical is a Surd (inert)",
}
for _n, _d in _INERT.items():
    record(_n, AREA, "added", "inert symbol: " + _d)

from .wl_numerics import SYSTEM_CONSTANTS as _SC  # noqa: E402

_SC.update(SYSTEM_VALUES)
record("$CharacterEncoding", AREA, "added", "system variable: default raw character encoding, \"UTF-8\"")
record("$SystemCharacterEncoding", AREA, "added", "system variable: the operating system's encoding, \"UTF-8\"")
record("$OutputSizeLimit", AREA, "added", "system variable: maximum output size in bytes before eliding, 2^20")
record("$NumberMarks", AREA, "added", "system variable: default NumberMarks setting, Automatic")
