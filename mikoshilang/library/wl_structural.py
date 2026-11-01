"""Wolfram coverage batch W03 -- patterns, structural and functional extras, ordering, arrays and Nearest.

Pattern objects (PatternSequence, OrderlessPatternSequence, Verbatim, KeyValuePattern) are implemented in the
matcher (kernel.match / kernel._match_seq).  Trees live in wl_trees.py and data structures in wl_datastruct.py.
Every function follows the Wolfram Language reference semantics and is checked by the differential tests in
tests/library/test_wolfram_w03.py.  Argument forms that are not implemented raise a clear error rather than
returning a guessed answer.
"""
from __future__ import annotations

import functools
import itertools
import math
import unicodedata

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated, REGISTRY, S
from .wl_common import (L, ev, apply, true, is_sym, is_list, is_int, is_assoc, is_rule, same, items_of,
                        canonical, tester, split_options, int_value, real_number)
from . import wl_lists as WL

AREA = "structural (W03)"
ADDED = dict(area=AREA, status="added")


def _sym(x):
    return isinstance(x, Symbol)


def _str_opt(v, fname, what):
    if isinstance(v, str):
        return v
    if _sym(v):
        return v.name
    raise ValueError(f"{fname}: invalid {what} {v!r}")


def _missing(*a):
    return Expr("Missing", *a)


# ─────────────────────────── pattern objects (matcher in kernel.py) ───────────────────────────

for _n, _note in [
        ("PatternSequence", "PatternSequence[p1, p2, ...] matches a sequence of arguments (also inside "
                            "Alternatives and Repeated, and named x : PatternSequence[...]); PatternSequence[] "
                            "matches the empty sequence"),
        ("OrderlessPatternSequence", "matches a sequence of arguments matching p1, p2, ... in any order (the order of "
                                     "the matched arguments is preserved; at most 8 patterns)"),
        ("Verbatim", "Verbatim[expr] matches expr literally (blanks inside are not patterns)"),
        ("KeyValuePattern", "an association or list of rules with a different element matching each pi, in any "
                            "order; KeyValuePattern[p] = KeyValuePattern[{p}]")]:
    record(_n, AREA, "added", _note)


# ─────────────────────────── inert operators and simple objects ───────────────────────────

@fn("Wedge", **ADDED, notes="x \\[Wedge] y: an operator with no built-in meaning (stays unevaluated)")
def _wedge(*args: Expr) -> Expr:
    raise Unevaluated()


@fn("SuchThat", **ADDED, notes="x \\[SuchThat] y: an operator with no built-in meaning (stays unevaluated)")
def _suchthat(*args: Expr) -> Expr:
    raise Unevaluated()


@fn("Cyclic", **ADDED, notes="Cyclic[{x1, ..., xn}] cyclic specification; Cyclic[{...}][k] = x[[Mod[k, n, 1]]]")
def _cyclic(spec: ListOf[Expr]) -> Expr:
    if not is_list(spec):
        raise ValueError("Cyclic: the argument must be a list {x1, x2, ...}")
    raise Unevaluated()


def _cyclic_apply(head, args, ev_):
    if len(head.args) != 1 or not is_list(head.args[0]) or len(args) != 1:
        return K.UNHANDLED
    items = head.args[0].args
    if not items:
        raise ValueError("Cyclic: cannot index an empty cyclic list")
    k = args[0]
    if not is_int(k):
        if isinstance(k, (Symbol, Expr)) and not K.is_number(k):
            return K.UNHANDLED
        raise ValueError(f"Cyclic: index {k!r} is not an integer")
    return items[(int(k) - 1) % len(items)]


KL.COMPOUND_HEAD_HOOKS["Cyclic"] = _cyclic_apply


# ─────────────────────────── Dispatch ───────────────────────────

def _dispatch_rules(r, fname="Dispatch"):
    if isinstance(r, Expr) and r.head == "Dispatch" and len(r.args) == 1:
        return _dispatch_rules(r.args[0], fname)
    if is_assoc(r):
        return list(r.args)
    if is_rule(r):
        return [r]
    if is_list(r) and all(is_rule(x) for x in r.args):
        return list(r.args)
    raise ValueError(f"{fname}: {r!r} is not a rule, a list of rules or an association")


@fn("Dispatch", **ADDED, notes="Dispatch[rules] is an optimised rule set usable wherever the rule list is (ReplaceAll, "
                               "Replace, ReplaceRepeated, ...); Normal[Dispatch[rules]] gives the rules; "
                               "Dispatch[<|...|>] takes the association's rules")
def _dispatch(r: Expr) -> Expr:
    rules = _dispatch_rules(r)
    if is_list(r) and len(rules) == len(r.args):
        raise Unevaluated()
    return Expr("Dispatch", L(*rules))


# ─────────────────────────── ordering ───────────────────────────

_EXPAND = {"æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ß": "ss", "ĳ": "ij", "Ĳ": "IJ", "ø": "o\u0338",
           "Ø": "O\u0338", "ł": "l\u0335", "Ł": "L\u0335", "đ": "d\u0335", "Đ": "D\u0335", "þ": "th", "Þ": "TH"}
_LANGS = {"English", "Automatic"}


def _collation_key(s, case_order, ignore_case, ignore_diacritics, ignore_punct):
    """(primary, secondary, tertiary) collation key in the spirit of the Unicode collation algorithm (English
    tailoring): punctuation < digits < letters; letters compare by base letter, then diacritics, then case."""
    s = "".join(_EXPAND.get(c, c) for c in s)
    d = unicodedata.normalize("NFD", s)
    prim, sec, ter = [], [], []
    for c in d:
        if unicodedata.combining(c) or c in "\u0338\u0335":
            if sec:
                sec[-1] = sec[-1] + (ord(c),)
            continue
        if c.isalpha():
            low = c.lower()
            prim.append((2, ord(low)))
            sec.append(())
            if c != low:
                ter.append(1 if case_order == "LowerFirst" else 0)
            else:
                ter.append(0 if case_order == "LowerFirst" else 1)
        elif c.isdigit():
            prim.append((1, unicodedata.digit(c, 0), ord(c)))
            sec.append(())
            ter.append(0)
        else:
            if ignore_punct:
                continue
            prim.append((0, ord(c)))
            sec.append(())
            ter.append(0)
    key = [tuple(prim)]
    if not ignore_diacritics:
        key.append(tuple(sec))
    if not ignore_case:
        key.append(tuple(ter))
    return tuple(key)


_ALPHA_OPTS = {"CaseOrdering", "IgnoreCase", "IgnoreDiacritics", "IgnorePunctuation", "Language"}


def _alpha_settings(lang, opts, fname):
    if lang is None:
        lang = opts.get("Language")
    if lang is not None:
        name = _str_opt(lang, fname, "language")
        if name not in _LANGS:
            raise NotImplementedError(f"{fname}: only the English (default) collation is implemented, not {name}")
    co = opts.get("CaseOrdering")
    case_order = "LowerFirst"
    if co is not None and not is_sym(co, "Automatic"):
        case_order = _str_opt(co, fname, "CaseOrdering")
        if case_order not in ("LowerFirst", "UpperFirst"):
            raise ValueError(f"{fname}: CaseOrdering must be \"LowerFirst\", \"UpperFirst\" or Automatic")
    flags = []
    for o in ("IgnoreCase", "IgnoreDiacritics", "IgnorePunctuation"):
        v = opts.get(o, False)
        if v not in (True, False):
            raise ValueError(f"{fname}: {o} must be True or False")
        flags.append(v)
    return (case_order, *flags)


def _alpha_cmp(a, b, settings, fname):
    if not (isinstance(a, str) and isinstance(b, str)):
        raise ValueError(f"{fname}: strings expected")
    ka, kb = _collation_key(a, *settings), _collation_key(b, *settings)
    if ka == kb:
        if settings[1] or settings[2] or settings[3]:
            return 0
        return 0 if a == b else (1 if a < b else -1)
    return 1 if ka < kb else -1


@fn("AlphabeticOrder", **ADDED,
    notes="1 / -1 / 0 for alphabetical (dictionary) order of two strings: punctuation < digits < letters, letters by "
          "base letter, then diacritics, then case (CaseOrdering \"LowerFirst\" default or \"UpperFirst\"); options "
          "IgnoreCase, IgnoreDiacritics, IgnorePunctuation; English collation only (other languages raise); "
          "AlphabeticOrder[] / AlphabeticOrder[\"English\"] / AlphabeticOrder[opts] operator forms")
def _alphaorder(*args: Expr) -> Expr:
    pos, opts = split_options(args, _ALPHA_OPTS, "AlphabeticOrder")
    if len(pos) < 2:
        if len(pos) == 1 and not isinstance(pos[0], (str, Symbol)):
            raise ValueError("AlphabeticOrder: invalid language specification")
        raise Unevaluated()
    if len(pos) > 3:
        raise ValueError("AlphabeticOrder called with too many arguments")
    a, b = pos[0], pos[1]
    if not (isinstance(a, str) and isinstance(b, str)):
        if isinstance(a, Symbol) or isinstance(b, Symbol):
            raise Unevaluated()
        raise ValueError("AlphabeticOrder: strings expected")
    return _alpha_cmp(a, b, _alpha_settings(pos[2] if len(pos) == 3 else None, opts, "AlphabeticOrder"),
                      "AlphabeticOrder")


def _alphaorder_op(head, args, ev_):
    if len(args) != 2:
        return K.UNHANDLED
    return ev_(Expr("AlphabeticOrder", args[0], args[1], *head.args))


KL.COMPOUND_HEAD_HOOKS["AlphabeticOrder"] = _alphaorder_op


@fn("AlphabeticSort", **ADDED,
    notes="sorts strings into alphabetical order (the AlphabeticOrder collation; stable); AlphabeticSort[list, "
          "\"English\"] and the CaseOrdering / IgnoreCase / IgnoreDiacritics / IgnorePunctuation options")
def _alphasort(x: Expr, *rest: Expr) -> ListOf[Expr]:
    pos, opts = split_options(rest, _ALPHA_OPTS, "AlphabeticSort")
    if len(pos) > 1:
        raise ValueError("AlphabeticSort called with too many arguments")
    if not is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("AlphabeticSort: a list of strings is expected")
    if not all(isinstance(s, str) for s in x.args):
        raise ValueError("AlphabeticSort: every element must be a string")
    st = _alpha_settings(pos[0] if pos else None, opts, "AlphabeticSort")
    return L(*sorted(x.args, key=functools.cmp_to_key(lambda u, v: -_alpha_cmp(u, v, st, "AlphabeticSort"))))


record("CaseOrdering", AREA, "added",
       "option of AlphabeticOrder and AlphabeticSort: \"LowerFirst\" (default), \"UpperFirst\" or Automatic")


def _order_val(r, fname):
    if r is True:
        return 1
    if r is False:
        return -1
    if is_int(r) and int(r) in (-1, 0, 1):
        return int(r)
    raise ValueError(f"{fname}: the ordering function must return 1, 0, -1, True or False, not {r!r}")


@fn("LexicographicOrder", **ADDED,
    notes="Order of the first differing pair of elements (canonical Order or ordering function p); if one list "
          "is a prefix of the other the shorter comes first; strings compare by Characters; any common head; "
          "LexicographicOrder[p] operator form")
def _lexorder(a: Expr, b: Expr, p: Fn = None) -> Expr:
    if isinstance(a, str) and isinstance(b, str):
        xa, xb = list(a), list(b)
    else:
        for v in (a, b):
            if not isinstance(v, Expr) or KL.atomq(v):
                if isinstance(v, Symbol):
                    raise Unevaluated()
                raise ValueError("LexicographicOrder: lists (or expressions with the same head) are expected")
        if a.head != b.head:
            raise ValueError("LexicographicOrder: the two expressions must have the same head")
        xa, xb = list(a.args), list(b.args)
    for u, v in zip(xa, xb):
        if p is None:
            r = WL._order(u, v)
        else:
            r = _order_val(apply(p, u, v), "LexicographicOrder")
        if r != 0:
            return r
    return 0 if len(xa) == len(xb) else (1 if len(xa) < len(xb) else -1)


def _lexorder_op(head, args, ev_):
    if len(head.args) != 1 or len(args) != 2:
        return K.UNHANDLED
    return ev_(Expr("LexicographicOrder", args[0], args[1], head.args[0]))


KL.COMPOUND_HEAD_HOOKS["LexicographicOrder"] = _lexorder_op


def _num_value(x):
    """Exact SymPy number for a numeric expression, else None."""
    if isinstance(x, bool) or isinstance(x, str):
        return None
    if isinstance(x, Expr) and x.head in ("DateObject", "TimeObject", "Quantity"):
        raise NotImplementedError("NumericalOrder: dates, times and quantities are not supported")
    if not K.is_number(x) and not (isinstance(x, (Symbol, Expr)) and _numericq(x)):
        return None
    try:
        v = K.to_sp(x)
    except Exception:
        return None
    if isinstance(v, sp.Basic) and v.is_number and v.is_finite is not False:
        return v
    return None


def _numericq(x):
    try:
        return ev(Expr("NumericQ", x)) is True
    except Exception:
        return False


def _cmp_real(u, v):
    """Sign of u - v for real SymPy numbers (exact when possible, else at 50 digits)."""
    d = sp.nsimplify(0) + (u - v)
    if d.is_zero:
        return 0
    if d.is_positive:
        return 1
    if d.is_negative:
        return -1
    n = sp.N(d, 50)
    if n == 0:
        return 0
    return 1 if n > 0 else -1


@fn("NumericalOrder", **ADDED,
    notes="1 if e1 < e2, -1 if e1 > e2, 0 if numerically equal (no tolerance); complex numbers by real part then "
          "|imaginary part|; numeric expressions precede non-numeric ones; otherwise canonical Order (dates and "
          "quantities raise)")
def _numorder(a: Expr, b: Expr) -> Expr:
    va, vb = _num_value(a), _num_value(b)
    if va is None and vb is None:
        return WL._order(a, b)
    if va is None:
        return -1
    if vb is None:
        return 1
    ra, ia = sp.re(va), sp.im(va)
    rb, ib = sp.re(vb), sp.im(vb)
    c = _cmp_real(ra, rb)
    if c == 0:
        c = _cmp_real(abs(ia), abs(ib))
    if c == 0:
        if _cmp_real(ia, ib) == 0:
            return 0
        return WL._order(a, b)
    return -c


# ─────────────────────────── functional ───────────────────────────

def _fold_while(f, x, items, test, m, n, fname, keep):
    if m is None:
        mm = 1
    elif is_sym(m, "All"):
        mm = "all"
    else:
        mm = int_value(m, fname, "number of results", 1)
    out = [x]
    k = 0
    while True:
        if mm == "all":
            ok = true(test, *out)
        elif len(out) < mm:
            ok = True
        else:
            ok = true(test, *out[-mm:])
        if not ok or k >= len(items):
            break
        out.append(apply(f, out[-1], items[k]))
        k += 1
    if n is not None:
        extra = int_value(n, fname, "extra steps")
        if extra > 0:
            for _ in range(extra):
                if k >= len(items):
                    break
                out.append(apply(f, out[-1], items[k]))
                k += 1
        elif extra < 0:
            out = out[:max(len(out) + extra, 1)]
    return L(*out) if keep else out[-1]


def _fold_while_args(args, fname):
    """(f, x, items, test, m, n) from FoldWhile[f, x, list, test, m, n] or FoldWhile[f, list, test]."""
    if len(args) == 3:
        f, lst, test = args
        items = items_of(lst, fname)
        if not items:
            raise ValueError(f"{fname}: the list must not be empty in the three-argument form")
        return f, items[0], items[1:], test, None, None
    if 4 <= len(args) <= 6:
        f, x, lst, test = args[:4]
        return f, x, items_of(lst, fname), test, (args[4] if len(args) > 4 else None), \
            (args[5] if len(args) > 5 else None)
    raise ValueError(f"{fname} called with {len(args)} arguments; expected 3 to 6")


@fn("FoldWhile", **ADDED,
    notes="folds f over the list while test of the most recent m (default 1, or All) results is True; returns the "
          "first result failing the test (or the last one); FoldWhile[f, list, test]; extra n / -n steps; "
          "FoldWhile[f, test] operator form")
def _foldwhile(*args: Expr) -> Expr:
    if len(args) == 2:
        raise Unevaluated()
    return _fold_while(*_fold_while_args(args, "FoldWhile"), "FoldWhile", False)


@fn("FoldWhileList", **ADDED, notes="the list of FoldWhile results, starting with x")
def _foldwhilelist(*args: Expr) -> Expr:
    if len(args) == 2:
        raise Unevaluated()
    return _fold_while(*_fold_while_args(args, "FoldWhileList"), "FoldWhileList", True)


def _foldwhile_op(head, args, ev_):
    if len(head.args) != 2 or len(args) not in (1, 2):
        return K.UNHANDLED
    f, test = head.args
    return ev_(Expr(head.head, f, *args, test))


KL.COMPOUND_HEAD_HOOKS["FoldWhile"] = _foldwhile_op
KL.COMPOUND_HEAD_HOOKS["FoldWhileList"] = _foldwhile_op


def _sequence_fold(f, seeds, lst, k, fname):
    if not is_list(seeds):
        raise ValueError(f"{fname}: the seeds must be a list {{x1, ..., xn}}")
    xs = list(seeds.args)
    items = items_of(lst, fname)
    n = len(xs)
    kk = n + 1 if k is None else int_value(k, fname, "number of arguments", 0)
    if kk < n:
        raise ValueError(f"{fname}: the number of arguments {kk} is smaller than the number of seeds {n}")
    if kk > n + 1:
        raise NotImplementedError(f"{fname}: taking more than one list element per step (k > n + 1) is not "
                                  "supported")
    out = list(xs)
    steps = len(items) if kk == n + 1 else len(items) + 1
    for s in range(steps):
        window = out[len(out) - n:] if n else []
        extra = [items[s]] if kk == n + 1 else []
        out.append(apply(f, *window, *extra))
    return out


@fn("SequenceFold", **ADDED,
    notes="SequenceFold[f, {x1, ..., xn}, list]: f of the last n results and the next element (a recurrence); "
          "{x} gives Fold, {} applies f to each element; k = n (f of the last n results only, len + 1 steps); "
          "k > n + 1 raises")
def _seqfold(f: Fn, seeds: Expr, lst: Expr, k: Expr = None) -> Expr:
    out = _sequence_fold(f, seeds, lst, k, "SequenceFold")
    if not out:
        raise ValueError("SequenceFold: no result (empty seeds and empty list)")
    return out[-1]


@fn("SequenceFoldList", **ADDED, notes="the seeds followed by the successive results of SequenceFold")
def _seqfoldlist(f: Fn, seeds: Expr, lst: Expr, k: Expr = None) -> ListOf[Expr]:
    return L(*_sequence_fold(f, seeds, lst, k, "SequenceFoldList"))


@fn("ReplaceAt", **ADDED,
    notes="ReplaceAt[expr, rules, n | {i, j, ...} | {{...}, ...}]: Replace (level 0, first matching rule) at the "
          "given parts; negative indices, All, Span, keys; repeated positions apply the rules repeatedly; "
          "ReplaceAt[rules, pos] operator form")
def _replaceat(x: Expr, rules: Expr, spec: Expr) -> Expr:
    rl = _dispatch_rules(rules, "ReplaceAt")

    def rep(e):
        for r in rl:
            b = K.first_match(r.args[0], e, ev)
            if b is not None:
                return ev(K.substitute(r.args[1], b))
        return e
    out = x
    for p in WL._positions(spec, "ReplaceAt"):
        for q in WL._expand_all(x, p, "ReplaceAt"):
            out = WL._modify_at(out, WL._resolve(out, q, "ReplaceAt"), rep)
    return ev(out)


KL.operator_form("ReplaceAt", 2, "first")


def _subset_positions(x, spec, fname):
    """Explicit positions for SubsetMap: {i, j, ...} (ints), {pos1, pos2, ...}, i ;; j, or a part spec with All /
    Span such as {All, 2}."""
    if isinstance(spec, Expr) and spec.head == "Span":
        return WL._expand_all(x, [spec], fname)
    if is_int(spec):
        return [[spec]]
    if not is_list(spec):
        raise ValueError(f"{fname}: invalid position specification {spec!r}")
    if all(is_int(i) or isinstance(i, str) or (isinstance(i, Expr) and i.head == "Key") for i in spec.args):
        return [[i] for i in spec.args]
    if all(is_list(p) for p in spec.args):
        out = []
        for p in spec.args:
            out.extend(WL._expand_all(x, list(p.args), fname))
        return out
    return WL._expand_all(x, list(spec.args), fname)


@fn("SubsetMap", **ADDED,
    notes="SubsetMap[f, list, {i, j, ...}] replaces the elements at the given positions by the elements of "
          "f[{e_i, e_j, ...}] (f must return a list of the same length); positions {pos1, ...}, i ;; j or part "
          "specs with All / Span; SubsetMap[f, pos] operator form")
def _subsetmap(f: Fn, x: Expr, spec: Expr) -> Expr:
    poss = _subset_positions(x, spec, "SubsetMap")
    resolved = [WL._resolve(x, q, "SubsetMap") for q in poss]
    vals = [KL.part(x, q, ev) for q in poss]
    new = apply(f, L(*vals))
    if not is_list(new) or len(new.args) != len(vals):
        raise ValueError(f"SubsetMap: {f!r} must return a list of length {len(vals)}")
    out = x
    for r, v in zip(resolved, new.args):
        out = WL._modify_at(out, r, lambda e, v=v: v)
    return ev(out)


KL.operator_form("SubsetMap", 2, "middle")


@fn("AssociationComap", **ADDED,
    notes="AssociationComap[{f1, f2, ...}, x] = <|f1 -> f1[x], ...|> (a repeated key keeps its last value); "
          "AssociationComap[fs] operator form")
def _assoccomap(fs: Expr, x: Expr) -> Expr:
    if is_assoc(fs):
        raise ValueError("AssociationComap: the keys must be given as a list")
    keys = list(fs.args) if is_list(fs) else None
    if keys is None:
        raise ValueError("AssociationComap: the first argument must be a list of functions")
    return KL.make_assoc([(k, apply(k, x)) for k in keys])


KL.operator_form("AssociationComap", 1, "last")


@fn("AllSameBy", **ADDED,
    notes="True when f[e] is the same (SameQ) for every element (short-circuits; lists of length 0 or 1 give "
          "True); AllSameBy[f] operator form")
def _allsameby(x: Expr, f: Fn) -> Boolean:
    items = items_of(x, "AllSameBy")
    if len(items) < 2:
        return True
    first = apply(f, items[0])
    for e in items[1:]:
        if not same(apply(f, e), first):
            return False
    return True


KL.operator_form("AllSameBy", 1, "first")


@fn("UniqueElements", **ADDED,
    notes="for each list, the elements (without repeats, in order) that are not in any other list; "
          "UniqueElements[lists, test] uses test for equivalence")
def _uniqueelements(lists: ListOf[Expr], test: Fn = None) -> ListOf[Expr]:
    if not is_list(lists):
        raise ValueError("UniqueElements: a list of lists is expected")
    ls = list(lists.args)
    for l_ in ls:
        if not isinstance(l_, Expr) or KL.atomq(l_):
            raise ValueError("UniqueElements: every element must be a list (or expression)")
    heads = {l_.head for l_ in ls}
    if len(heads) > 1:
        raise ValueError("UniqueElements: the lists must have the same head")
    t = tester(test)
    out = []
    for i, l_ in enumerate(ls):
        others = [e for j, o in enumerate(ls) if j != i for e in o.args]
        kept = []
        for e in l_.args:
            if any(t(e, o) for o in others) or any(t(e, k) for k in kept):
                continue
            kept.append(e)
        out.append(Expr(l_.head, *kept))
    return L(*out)


@fn("Discard", **ADDED,
    notes="removes the elements for which crit is True (opposite of Select); Discard[data, crit, n] removes only "
          "the first n; data may be a list, any expression or an association; crit -> \"Element\" | \"Index\" | "
          "{props}; Discard[crit] operator form")
def _discard(x: Expr, crit: Fn, n: Expr = None) -> Expr:
    prop = None
    if is_rule(crit) and not (isinstance(crit.args[0], Expr) and crit.args[0].head in ("Rule",)):
        crit, prop = crit.args
    limit = None
    if n is not None:
        if is_sym(n, "Infinity"):
            limit = None
        else:
            limit = int_value(n, "Discard", "count", 0)
    if is_assoc(x):
        entries = list(x.args)
        values = [KL.rule_value(r, ev) for r in entries]
    elif isinstance(x, Expr) and not KL.atomq(x):
        entries = list(x.args)
        values = entries
    else:
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("Discard: a list, expression or association is expected")
    keep, removed = [], 0
    for i, v in enumerate(values):
        if (limit is None or removed < limit) and true(crit, v):
            removed += 1
            continue
        keep.append(i)
    if prop is None or prop == "Element":
        return Expr(x.head, *(entries[i] for i in keep))

    def one(p):
        if p == "Element":
            return Expr(x.head, *(entries[i] for i in keep))
        if p == "Index":
            if is_assoc(x):
                return L(*(Expr("Key", entries[i].args[0]) for i in keep))
            return L(*(i + 1 for i in keep))
        raise NotImplementedError(f"Discard: property {p!r} is not supported (\"Element\" and \"Index\" are)")
    if is_list(prop):
        return L(*(one(p) for p in prop.args))
    return one(prop)


KL.operator_form("Discard", 1, "first")


def _elementwise(f, args):
    lists = [a for a in args if is_list(a)]
    if not lists:
        return apply(f, *args)
    lens = {len(a.args) for a in lists}
    if len(lens) != 1:
        raise ValueError("Elementwise: the lists have incompatible lengths")
    n = lens.pop()
    return L(*(_elementwise(f, [a.args[i] if is_list(a) else a for a in args]) for i in range(n)))


@fn("Elementwise", **ADDED,
    notes="Elementwise[f][args...] threads f over all lists among the arguments to the deepest common level "
          "(makes non-listable functions behave as listable)")
def _elementwise_head(f: Expr) -> Expr:
    raise Unevaluated()


def _elementwise_apply(head, args, ev_):
    if len(head.args) != 1:
        return K.UNHANDLED
    return _elementwise(head.args[0], list(args))


KL.COMPOUND_HEAD_HOOKS["Elementwise"] = _elementwise_apply


# ─────────────────────────── Subset* (sublists matching in any order) ───────────────────────────

_SUBSET_BUDGET = 200_000


def _len_bounds(p):
    """(min, max) number of list elements a list-element pattern can take (max None = unbounded)."""
    q = p
    while isinstance(q, Expr) and q.head in ("Condition", "PatternTest") and len(q.args) == 2:
        q = q.args[0]
    if isinstance(q, Expr) and q.head == "Pattern" and len(q.args) == 2:
        q = q.args[1]
    if not isinstance(q, Expr):
        return 1, 1
    h = q.head
    if h == "BlankSequence":
        return 1, None
    if h == "BlankNullSequence":
        return 0, None
    if h in ("Repeated", "RepeatedNull"):
        lo, hi = K._repeat_bounds(q)
        ilo, ihi = _len_bounds(q.args[0])
        return lo * ilo, (None if ihi is None or hi >= 10 ** 9 else hi * ihi)
    if h in ("PatternSequence", "OrderlessPatternSequence"):
        lo, hi = 0, 0
        for a in q.args:
            a_lo, a_hi = _len_bounds(a)
            lo += a_lo
            hi = None if hi is None or a_hi is None else hi + a_hi
        return lo, hi
    if h == "Alternatives":
        bs = [_len_bounds(a) for a in q.args]
        return min(b[0] for b in bs), (None if any(b[1] is None for b in bs) else max(b[1] for b in bs))
    if h == "Optional":
        return 0, 1
    return 1, 1


def _sizes(p, n):
    q = p.args[0] if isinstance(p, Expr) and p.head == "HoldPattern" and len(p.args) == 1 else p
    if is_list(q):
        lo, hi = 0, 0
        for a in q.args:
            a_lo, a_hi = _len_bounds(a)
            lo += a_lo
            hi = None if hi is None or a_hi is None else hi + a_hi
        hi = n if hi is None else min(hi, n)
        return range(hi, lo - 1, -1)
    return range(n, -1, -1)


class _Budget:
    def __init__(self, fname):
        self.n = 0
        self.fname = fname

    def tick(self):
        self.n += 1
        if self.n > _SUBSET_BUDGET:
            raise NotImplementedError(f"{self.fname}: the subset search is too large")


def _first_subset(items, avail, patt, budget):
    """First (indices in pattern order, bindings): combinations by decreasing size, then in lexicographic order of
    positions; the first ordering of a combination that matches."""
    for size in _sizes(patt, len(avail)):
        for combo in itertools.combinations(avail, size):
            seen = set()
            for perm in itertools.permutations(combo):
                key = tuple(K.tag(items[i]) for i in perm)
                if key in seen:
                    continue
                seen.add(key)
                budget.tick()
                b = K.first_match(patt, L(*(items[i] for i in perm)), ev)
                if b is not None:
                    return list(perm), b
    return None


def _all_subsets(items, patt, budget):
    out = []
    n = len(items)
    for size in _sizes(patt, n):
        for combo in itertools.combinations(range(n), size):
            seen = set()
            for perm in itertools.permutations(combo):
                key = tuple(K.tag(items[i]) for i in perm)
                if key in seen:
                    continue
                seen.add(key)
                budget.tick()
                b = K.first_match(patt, L(*(items[i] for i in perm)), ev)
                if b is not None:
                    out.append((list(perm), b))
                    break
    return out


def _branches(patt):
    return list(patt.args) if isinstance(patt, Expr) and patt.head == "Alternatives" else [patt]


def _subset_matches(items, patt, overlaps, limit, fname):
    budget = _Budget(fname)
    brs = _branches(patt)
    out = []
    if overlaps:
        for br in brs:
            for m in _all_subsets(items, br, budget):
                out.append(m)
                if limit is not None and len(out) >= limit:
                    return out
        return out
    avail = list(range(len(items)))
    while limit is None or len(out) < limit:
        found = None
        for br in brs:
            found = _first_subset(items, avail, br, budget)
            if found is not None:
                break
        if found is None or not found[0]:
            break
        out.append(found)
        avail = [i for i in avail if i not in found[0]]
    return out


def _subset_args(x, rest, fname, default_overlaps):
    pos, opts = split_options(rest, {"Overlaps"}, fname)
    if len(pos) > 1:
        raise ValueError(f"{fname} called with too many arguments")
    limit = None
    if pos:
        if is_sym(pos[0], "Infinity"):
            limit = None
        else:
            limit = int_value(pos[0], fname, "number of matches", 0)
    ov = opts.get("Overlaps", default_overlaps)
    if ov not in (True, False):
        raise NotImplementedError(f"{fname}: Overlaps must be True or False")
    items = items_of(x, fname) if not is_assoc(x) else None
    if items is None:
        raise ValueError(f"{fname}: a list is expected")
    return items, limit, ov


@fn("SubsetCases", **ADDED,
    notes="sublists (in any order, not necessarily contiguous) matching a list pattern, given in the pattern's order; "
          "longest matches first, non-overlapping by default (Overlaps -> True gives all); patt -> rhs; n matches; "
          "Alternatives tried in order")
def _subsetcases(x: Expr, patt: Expr, *rest: Expr) -> ListOf[Expr]:
    items, limit, ov = _subset_args(x, rest, "SubsetCases", False)
    rhs = None
    if is_rule(patt):
        patt, rhs = patt.args[0], patt
    out = []
    for perm, b in _subset_matches(items, patt, ov, limit, "SubsetCases"):
        if rhs is None:
            out.append(L(*(items[i] for i in perm)))
        else:
            out.append(ev(K.substitute(rhs.args[1], b)))
    return L(*out)


@fn("SubsetCount", **ADDED, notes="number of sublists matching patt in any order (non-overlapping by default; "
                                  "Overlaps -> True counts all)")
def _subsetcount(x: Expr, patt: Expr, *rest: Expr) -> Integer:
    items, limit, ov = _subset_args(x, rest, "SubsetCount", False)
    if limit is not None:
        raise ValueError("SubsetCount called with too many arguments")
    return len(_subset_matches(items, patt, ov, None, "SubsetCount"))


@fn("SubsetPosition", **ADDED, notes="positions {i, j, ...} (in the pattern's order) of sublists matching patt in any "
                                     "order; overlapping by default (Overlaps -> False excludes); first n")
def _subsetposition(x: Expr, patt: Expr, *rest: Expr) -> ListOf[Expr]:
    items, limit, ov = _subset_args(x, rest, "SubsetPosition", True)
    return L(*(L(*(i + 1 for i in perm)) for perm, _b in _subset_matches(items, patt, ov, limit, "SubsetPosition")))


@fn("SubsetReplace", **ADDED,
    notes="replaces sublists matching a rule's pattern (in any order) by its right-hand side, inserted at the position "
          "of the element matched first; rules applied in order to untransformed elements; first n replacements; "
          "SubsetReplace[rules] operator form")
def _subsetreplace(x: Expr, rules: Expr, n: Expr = None) -> Expr:
    items = items_of(x, "SubsetReplace")
    rl = _dispatch_rules(rules, "SubsetReplace")
    limit = None if n is None or is_sym(n, "Infinity") else int_value(n, "SubsetReplace", "count", 0)
    avail = list(range(len(items)))
    inserted, removed = {}, set()
    count = 0
    for r in rl:
        budget = _Budget("SubsetReplace")
        while limit is None or count < limit:
            found = None
            for br in _branches(r.args[0]):
                found = _first_subset(items, avail, br, budget)
                if found is not None:
                    break
            if found is None or not found[0]:
                break
            perm, b = found
            inserted[perm[0]] = ev(K.substitute(r.args[1], b))
            removed.update(perm[1:])
            avail = [i for i in avail if i not in perm]
            count += 1
    out = []
    for i, e in enumerate(items):
        if i in inserted:
            out.append(inserted[i])
        elif i not in removed:
            out.append(e)
    return Expr(x.head if isinstance(x, Expr) else "List", *out)


KL.operator_form("SubsetReplace", 1, "first")


# ─────────────────────────── arrays ───────────────────────────

def _full_dims(x):
    """Dimensions of a full (rectangular) array of lists; [] for a non-list."""
    if not is_list(x):
        return []
    if not x.args:
        return [0]
    subs = [_full_dims(e) for e in x.args]
    first = subs[0]
    common = []
    for k in range(len(first)):
        if all(len(s) > k and s[k] == first[k] for s in subs):
            common.append(first[k])
        else:
            break
    return [len(x.args)] + common


def _to_nested(x, depth):
    if depth == 0:
        return x
    return [_to_nested(e, depth - 1) for e in x.args]


def _from_nested(v, depth):
    if depth == 0:
        return v
    return L(*(_from_nested(e, depth - 1) for e in v))


_PAD_NAMES = {"Fixed", "Periodic", "Reflected", "Reversed"}


def _pad_index(i, n, kind):
    if kind == "Fixed":
        return min(max(i, 0), n - 1)
    if kind == "Periodic":
        return i % n
    if kind == "Reflected":
        if n == 1:
            return 0
        p = 2 * n - 2
        j = i % p
        return p - j if j >= n else j
    if kind == "Reversed":
        j = i % (2 * n)
        return 2 * n - 1 - j if j >= n else j
    raise AssertionError(kind)


def _padding_kind(pad, fname):
    if isinstance(pad, str):
        if pad in _PAD_NAMES:
            return pad
        raise NotImplementedError(f"{fname}: padding \"{pad}\" is not supported (constants, \"Fixed\", \"Periodic\", "
                                  "\"Reflected\" and \"Reversed\" are)")
    if is_list(pad):
        raise NotImplementedError(f"{fname}: cyclic constant padding {{c1, c2, ...}} is not supported")
    return ("const", pad)


def _shape_const(c, dims):
    if not dims:
        return c
    return [_shape_const(c, dims[1:]) for _ in range(dims[0])]


def _pad_nested(a, specs, kind, dims):
    """Pad nested Python lists a (dims) by specs [(before, after), ...] for the leading levels."""
    if not specs:
        return a
    (m, n), rest = specs[0], specs[1:]
    inner = [_pad_nested(e, rest, kind, dims[1:]) for e in a]
    N = dims[0]
    if isinstance(kind, tuple):
        inner_dims = [d + s[0] + s[1] for d, s in zip(dims[1:], rest)] + dims[1 + len(rest):]
        filler = _shape_const(kind[1], inner_dims)
        out = []
        for i in range(-m, N + n):
            out.append(inner[i] if 0 <= i < N else filler)
        return out
    if N == 0:
        raise ValueError("cannot pad an empty array with a named padding")
    return [inner[_pad_index(i, N, kind)] for i in range(-m, N + n)]


def _pad_specs(spec, depth, fname):
    def pair(v):
        if is_int(v):
            return int(v), int(v)
        if is_list(v) and len(v.args) == 1 and is_int(v.args[0]):
            return int(v.args[0]), int(v.args[0])
        if is_list(v) and len(v.args) == 2 and all(is_int(u) for u in v.args):
            return int(v.args[0]), int(v.args[1])
        raise ValueError(f"{fname}: invalid padding specification {v!r}")
    if is_int(spec):
        return [(int(spec), int(spec))] * depth
    if is_list(spec) and len(spec.args) == 2 and all(is_int(u) for u in spec.args):
        return [(int(spec.args[0]), int(spec.args[1]))] * depth
    if is_list(spec) and all(is_list(v) for v in spec.args):
        if len(spec.args) > depth:
            raise ValueError(f"{fname}: the padding specification is deeper than the array")
        return [pair(v) for v in spec.args]
    raise ValueError(f"{fname}: invalid padding specification {spec!r}")


@fn("ArrayPad", **ADDED,
    notes="ArrayPad[array, m | {m, n} | {{m1, n1}, ...}, padding]: constant padding (default 0) or \"Fixed\", "
          "\"Periodic\", \"Reflected\", \"Reversed\"; negative amounts trim; any depth of full array (cyclic "
          "constant lists and the differences / extrapolated paddings raise)")
def _arraypad(a: Expr, spec: Expr, pad: Expr = 0) -> Expr:
    dims = _full_dims(a)
    if not dims:
        if isinstance(a, Symbol):
            raise Unevaluated()
        raise ValueError("ArrayPad: a full array is expected")
    specs = _pad_specs(spec, len(dims), "ArrayPad")
    for (m, n), d in zip(specs, dims):
        if d + m + n < 0:
            raise ValueError("ArrayPad: cannot trim more elements than the array has")
    kind = _padding_kind(pad, "ArrayPad")
    return _from_nested(_pad_nested(_to_nested(a, len(dims)), specs, kind, dims), len(dims))


@fn("CenterArray", **ADDED,
    notes="CenterArray[a, n | {n1, n2, ...}, pad]: a centred in an array of the given dimensions (extra padding on "
          "the right; Inherited keeps a dimension; a lower-rank a is centred at the deepest levels); CenterArray[n] "
          "has a single 1; non-list a means {a}; trimming by an odd amount raises; padding as in ArrayPad")
def _centerarray(a: Expr, n: Expr = None, pad: Expr = 0, *rest: Expr) -> Expr:
    if n is None:
        a, n = L(1), a
    if rest:
        if len(rest) == 1 and is_sym(rest[0], "SparseArray"):
            pass          # MikoshiLang's SparseArray is the equivalent dense array
        else:
            raise ValueError("CenterArray called with too many arguments")
    if not is_list(a):
        a = L(a)
    targets = list(n.args) if is_list(n) else [n]
    dims = _full_dims(a)
    while len(dims) < len(targets):
        a = L(a)
        dims = _full_dims(a)
    specs = []
    for k, t in enumerate(targets):
        if is_sym(t, "Inherited"):
            specs.append((0, 0))
            continue
        tt = int_value(t, "CenterArray", "dimension", 0)
        diff = tt - dims[k]
        if diff < 0 and diff % 2:
            raise NotImplementedError("CenterArray: trimming by an odd number of elements is not supported")
        left = diff // 2
        specs.append((left, diff - left))
    kind = _padding_kind(pad, "CenterArray")
    return _from_nested(_pad_nested(_to_nested(a, len(dims)), specs, kind, dims), len(dims))


def _block(arr, idx, ranges, mask=None, path=()):
    """Nested block of arr (padded nested lists) at idx with half-widths ranges."""
    if not ranges:
        if mask is not None and not _mask_at(mask, path):
            return Expr("Missing")
        return arr
    i0 = idx[0]
    r = ranges[0]
    return [_block(arr[i0 + d], idx[1:], ranges[1:], mask, path + (d + r,)) for d in range(-r, r + 1)]


def _mask_at(mask, path):
    v = mask
    for p in path:
        v = v[p]
    return v != 0


def _blocks_over(padded, out_dims, ranges, offset, f, mask, level=0, idx=()):
    if level == len(out_dims):
        blk = _block(padded, [i + o for i, o in zip(idx, offset)], ranges, mask)
        return apply(f, _from_nested_any(blk))
    return L(*(_blocks_over(padded, out_dims, ranges, offset, f, mask, level + 1, idx + (i,))
               for i in range(out_dims[level])))


def _from_nested_any(v):
    if isinstance(v, list):
        return L(*(_from_nested_any(e) for e in v))
    return v


@fn("ArrayFilter", **ADDED,
    notes="ArrayFilter[f, array, r | {r1, r2, ...} | template] applies f to the range-r blocks around every element "
          "(template 0s become Missing[]); Padding -> \"Fixed\" (default), None (smaller result), a constant, "
          "\"Periodic\", \"Reflected\" or \"Reversed\"")
def _arrayfilter(f: Expr, a: Expr, r: Expr, *rest: Expr) -> Expr:
    pos, opts = split_options(rest, {"Padding"}, "ArrayFilter")
    if pos:
        raise ValueError("ArrayFilter called with too many arguments")
    dims = _full_dims(a)
    if not dims or 0 in dims:
        if isinstance(a, Symbol):
            raise Unevaluated()
        raise ValueError("ArrayFilter: a non-empty full array is expected")
    mask = None
    if is_int(r):
        ranges = [int_value(r, "ArrayFilter", "range", 0)] * len(dims)
    elif is_list(r) and all(is_int(v) for v in r.args) and (len(r.args) == len(dims) or not (
            len(r.args) % 2 == 1 and all(int(v) in (0, 1) for v in r.args) and len(dims) == 1)):
        ranges = [int_value(v, "ArrayFilter", "range", 0) for v in r.args]
    elif is_list(r):
        tdims = _full_dims(r)
        if any(d % 2 == 0 for d in tdims):
            raise ValueError("ArrayFilter: the template must have odd dimensions")
        ranges = [(d - 1) // 2 for d in tdims]
        mask = _to_nested(r, len(tdims))
    else:
        raise ValueError("ArrayFilter: the range must be an integer, a list of integers or a template array")
    if len(ranges) != len(dims):
        raise NotImplementedError("ArrayFilter: give one range per array dimension")
    pad = opts.get("Padding", "Fixed")
    nested = _to_nested(a, len(dims))
    if pad is None or is_sym(pad, "None"):
        out_dims = [d - 2 * q for d, q in zip(dims, ranges)]
        if any(d <= 0 for d in out_dims):
            return L()
        return _blocks_over(nested, out_dims, ranges, ranges, f, mask)
    kind = _padding_kind(pad, "ArrayFilter")
    padded = _pad_nested(nested, [(q, q) for q in ranges], kind, dims)
    return _blocks_over(padded, dims, ranges, ranges, f, mask)


record("Padding", AREA, "added",
       "option of ArrayFilter (and CenterArray-style padding values): a constant, \"Fixed\", \"Periodic\", "
       "\"Reflected\", \"Reversed\" or None")


def _reduce_groups(spec, depth, fname):
    def lvl(v):
        k = int_value(v, fname, "dimension")
        if k < 0:
            k = depth + 1 + k
        if not 1 <= k <= depth:
            raise ValueError(f"{fname}: dimension {v!r} does not exist")
        return k - 1
    if is_int(spec):
        groups = [[lvl(spec)]]
    elif isinstance(spec, Expr) and spec.head == "Span":
        idx = KL.seq_indices(spec, depth, fname)
        groups = [list(idx)]
    elif is_list(spec) and spec.args and all(is_list(g) for g in spec.args):
        groups = [[lvl(v) for v in g.args] for g in spec.args]
    elif is_list(spec):
        groups = [[lvl(v) for v in spec.args]]
    else:
        raise ValueError(f"{fname}: invalid dimension specification {spec!r}")
    flat = [d for g in groups for d in g]
    if len(set(flat)) != len(flat):
        raise ValueError(f"{fname}: a dimension is repeated")
    return groups


@fn("ArrayReduce", **ADDED,
    notes="ArrayReduce[f, array, n | n1 ;; n2 | {n1, n2, ...} | {{...}, {...}}]: f applied to the vectors (or "
          "arrays) along the given dimensions, which are removed; f's result takes the place of each vector")
def _arrayreduce(f: Fn, a: Expr, spec: Expr) -> Expr:
    dims = _full_dims(a)
    if not dims:
        if isinstance(a, Symbol):
            raise Unevaluated()
        raise ValueError("ArrayReduce: a full array is expected")
    D = len(dims)
    groups = _reduce_groups(spec, D, "ArrayReduce")
    reduced = {d for g in groups for d in g}
    keep = [d for d in range(D) if d not in reduced]
    nested = _to_nested(a, D)

    def get(idx):
        v = nested
        for i in idx:
            v = v[i]
        return v

    def sub(fixed):
        def grp(gi, partial):
            if gi == len(groups):
                return get([partial[d] for d in range(D)])
            g = groups[gi]
            vals = []
            for combo in itertools.product(*(range(dims[d]) for d in g)):
                p = dict(partial)
                p.update(zip(g, combo))
                vals.append(grp(gi + 1, p))
            return L(*vals)
        return grp(0, fixed)

    def build(k, fixed):
        if k == len(keep):
            return apply(f, sub(fixed))
        d = keep[k]
        return L(*(build(k + 1, {**fixed, d: i}) for i in range(dims[d])))
    return build(0, {})


@fn("UnitVector", **ADDED, notes="UnitVector[n, k]: length-n vector with 1 at position k; UnitVector[k] = "
                                 "UnitVector[2, k]")
def _unitvector(n: Expr, k: Expr = None) -> ListOf[Expr]:
    if k is None:
        n, k = 2, n
    nn = int_value(n, "UnitVector", "length", 1)
    kk = int_value(k, "UnitVector", "position", 1)
    if kk > nn:
        raise ValueError(f"UnitVector: position {kk} exceeds the length {nn}")
    return L(*(1 if i == kk else 0 for i in range(1, nn + 1)))


@fn("RangeSpace", **ADDED,
    notes="basis of the range (column space) of a matrix: for exact and symbolic matrices the nonzero rows of "
          "RowReduce[Transpose[a]] (generic values for symbols); for approximate matrices the left singular vectors "
          "of the nonzero singular values (Tolerance option)")
def _rangespace(a: Expr, *rest: Expr) -> Expr:
    pos, opts = split_options(rest, {"Tolerance", "Method", "Modulus", "ZeroTest"}, "RangeSpace")
    if pos:
        raise ValueError("RangeSpace called with too many arguments")
    for o in ("Modulus", "ZeroTest", "Method"):
        if o in opts and not (o == "Modulus" and opts[o] == 0) and not is_sym(opts[o], "Automatic"):
            raise NotImplementedError(f"RangeSpace: the {o} option is not supported")
    dims = _full_dims(a)
    if len(dims) != 2:
        if isinstance(a, Symbol):
            raise Unevaluated()
        raise ValueError("RangeSpace: a rectangular matrix is expected")
    if dims[0] == 0 or dims[1] == 0:
        return L()
    M = sp.Matrix([[K.to_sp(v) for v in row.args] for row in a.args])
    if any(isinstance(v, sp.Float) for v in M.atoms(sp.Float)):
        import numpy as np
        A = np.array(M.evalf(), dtype=complex)
        if not np.iscomplexobj(A) or np.all(A.imag == 0):
            A = A.real
        U, s, _vh = np.linalg.svd(A)
        tol = opts.get("Tolerance")
        if tol is None or is_sym(tol, "Automatic"):
            t = max(A.shape) * np.finfo(float).eps * (s[0] if len(s) else 0)
        else:
            t = float(K.to_sp(tol)) * (s[0] if len(s) else 0)
        r = int(np.sum(s > t))
        vecs = []
        for v in U[:, :r].T:
            nz = [c for c in v if abs(c) > 1e-14]
            if nz and (nz[0].real if np.iscomplexobj(v) else nz[0]) < 0:
                v = -v
            vecs.append(v)
        from .core import R as _R
        return _R(np.array(vecs)) if r else L()
    R_, _piv = M.T.rref()
    rows = [R_.row(i) for i in range(R_.rows) if any(v != 0 for v in R_.row(i))]
    return L(*(L(*(K.from_sp(sp.simplify(v)) for v in row)) for row in rows))


# ─────────────────────────── Nearest ───────────────────────────

def _levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_num(x):
    return not isinstance(x, (bool, str)) and K.is_number(x)


def _is_vec(x):
    return is_list(x) and x.args and all(_is_num(v) for v in x.args)


def _kind(elems):
    if all(_is_num(e) for e in elems):
        return "num"
    if all(isinstance(e, str) for e in elems):
        return "str"
    if all(_is_vec(e) for e in elems) and len({len(e.args) for e in elems}) == 1:
        return "vec"
    if all(is_list(e) and all(v in (True, False) for v in e.args) for e in elems) and \
            len({len(e.args) for e in elems}) == 1:
        return "bool"
    return None


def _auto_distance(u, v, kind):
    if kind == "num":
        return sp.Abs(K.to_sp(u) - K.to_sp(v))
    if kind == "vec":
        if not _is_vec(v) or len(v.args) != len(u.args):
            raise ValueError("Nearest: the query point has the wrong dimension")
        return sp.sqrt(sum(sp.Abs(K.to_sp(a) - K.to_sp(b)) ** 2 for a, b in zip(u.args, v.args)))
    if kind == "str":
        if not isinstance(v, str):
            raise ValueError("Nearest: a string query is expected")
        return sp.Integer(_levenshtein(u, v))
    if kind == "bool":
        # JaccardDissimilarity: (c_TF + c_FT) / (c_TT + c_TF + c_FT)
        tt = sum(1 for a, b in zip(u.args, v.args) if a is True and b is True)
        diff = sum(1 for a, b in zip(u.args, v.args) if a is not b)
        return sp.Integer(0) if tt + diff == 0 else sp.Rational(diff, tt + diff)
    raise AssertionError(kind)


def _nearest_data(data, fname):
    """(elements, outputs, prop) from the data specification."""
    prop = None
    if is_assoc(data):
        elems = [KL.rule_value(r, ev) for r in data.args]
        outs = [r.args[0] for r in data.args]
    elif is_rule(data) and is_list(data.args[0]):
        elems = list(data.args[0].args)
        rhs = data.args[1]
        if is_list(rhs) and not all(isinstance(p, str) for p in rhs.args) or \
                (is_list(rhs) and len(rhs.args) == len(elems) and not all(
                    p in ("Element", "Index", "Distance") for p in rhs.args)):
            if len(rhs.args) != len(elems):
                raise ValueError(f"{fname}: elements and values have different lengths")
            outs = list(rhs.args)
        else:
            outs = list(elems)
            prop = rhs
    elif is_list(data) and data.args and all(is_rule(r) for r in data.args):
        elems = [r.args[0] for r in data.args]
        outs = [r.args[1] for r in data.args]
    elif is_list(data):
        elems = list(data.args)
        outs = list(elems)
    else:
        if isinstance(data, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: invalid data {data!r}")
    if not elems:
        raise ValueError(f"{fname}: the data must not be empty")
    return elems, outs, prop


def _sortable(d):
    v = sp.N(d, 30)
    if not v.is_real:
        raise ValueError(f"Nearest: the distance {d} is not a real number")
    return v


def _nearest_query(elems, outs, prop, x, spec, dist, fname):
    kind = _kind(elems) if dist is None else "custom"
    if dist is None and kind is None:
        raise NotImplementedError(f"{fname}: give a DistanceFunction for this kind of data")
    # threading over a list of query points
    if kind == "num" and is_list(x):
        return L(*(_nearest_query(elems, outs, prop, q, spec, dist, fname) for q in x.args))
    if kind in ("vec", "bool") and is_list(x) and x.args and all(is_list(q) for q in x.args):
        return L(*(_nearest_query(elems, outs, prop, q, spec, dist, fname) for q in x.args))
    if kind == "str" and is_list(x):
        return L(*(_nearest_query(elems, outs, prop, q, spec, dist, fname) for q in x.args))
    if dist is None:
        ds = [_auto_distance(e, x, kind) for e in elems]
    else:
        ds = []
        for e in elems:
            r = apply(dist, e, x)
            try:
                ds.append(K.to_sp(r))
            except Exception:
                raise ValueError(f"{fname}: the distance function gave {r!r}, not a number")
    keys = [_sortable(d) for d in ds]
    order = sorted(range(len(elems)), key=lambda i: (keys[i], i))
    n, radius = None, None
    if spec is not None:
        if is_list(spec) and len(spec.args) == 2:
            n = None if is_sym(spec.args[0], "All") else int_value(spec.args[0], fname, "count", 0)
            radius = _sortable(K.to_sp(spec.args[1]))
        elif is_sym(spec, "All"):
            n = len(elems)
        else:
            n = int_value(spec, fname, "count", 0)
    if spec is None:
        best = keys[order[0]]
        chosen = [i for i in order if keys[i] == best or sp.simplify(ds[i] - ds[order[0]]) == 0]
    else:
        chosen = [i for i in order if radius is None or keys[i] <= radius]
        if n is not None:
            chosen = chosen[:n]

    def one(p, i):
        if p == "Element":
            return outs[i]
        if p == "Index":
            return i + 1
        if p == "Distance":
            return K.from_sp(ds[i]) if not isinstance(ds[i], sp.Float) else float(ds[i])
        raise ValueError(f"{fname}: unknown property {p!r}")
    if prop is None:
        return L(*(outs[i] for i in chosen))
    if is_sym(prop, "All"):
        return L(*(KL.make_assoc([("Element", outs[i]), ("Index", i + 1), ("Distance", one("Distance", i))])
                   for i in chosen))
    if is_list(prop):
        return L(*(L(*(one(p, i) for p in prop.args)) for i in chosen))
    return L(*(one(prop, i) for i in chosen))


def _nearest_dist_opt(opts, fname):
    for o in ("Method", "WorkingPrecision"):
        if o in opts and not is_sym(opts[o], "Automatic"):
            raise NotImplementedError(f"{fname}: the {o} option is not supported")
    d = opts.get("DistanceFunction")
    return None if d is None or is_sym(d, "Automatic") else d


_NEAREST_OPTS = {"DistanceFunction", "Method", "WorkingPrecision"}


@fn("Nearest", **ADDED,
    notes="Nearest[data, x] all elements at the smallest distance; Nearest[data, x, n | {n, r} | {All, r}]; data as a "
          "list, {e1 -> v1, ...}, {e1, ...} -> {v1, ...}, {e1, ...} -> \"Element\" | \"Index\" | \"Distance\" | "
          "{props} | All, or an association; lists of query points thread; distances: |x - y| for numbers, the "
          "Euclidean norm for vectors, EditDistance for strings, Jaccard dissimilarity for Boolean vectors, or "
          "DistanceFunction -> f; Nearest[data] gives a NearestFunction")
def _nearest(data: Expr, *rest: Expr) -> Expr:
    pos, opts = split_options(rest, _NEAREST_OPTS, "Nearest")
    dist = _nearest_dist_opt(opts, "Nearest")
    if not pos:
        _nearest_data(data, "Nearest")
        o = [Expr("Rule", Symbol("DistanceFunction"), dist)] if dist is not None else []
        return Expr("NearestFunction", data, *o)
    if len(pos) > 2:
        raise ValueError("Nearest called with too many arguments")
    elems, outs, prop = _nearest_data(data, "Nearest")
    return _nearest_query(elems, outs, prop, pos[0], pos[1] if len(pos) == 2 else None, dist, "Nearest")


@fn("NearestFunction", **ADDED,
    notes="NearestFunction[data] (made by Nearest[data]) applied as nf[x], nf[{x1, ...}], nf[x, n], nf[x, {n, r}]")
def _nearestfunction(*args: Expr) -> Expr:
    if not args:
        raise ValueError("NearestFunction: data is required")
    raise Unevaluated()


def _nf_apply(head, args, ev_):
    if not head.args or not 1 <= len(args) <= 2:
        return K.UNHANDLED
    return ev_(Expr("Nearest", head.args[0], *args, *head.args[1:]))


KL.COMPOUND_HEAD_HOOKS["NearestFunction"] = _nf_apply


@fn("NearestTo", **ADDED,
    notes="operator form: NearestTo[x][elems] = Nearest[elems, x]; NearestTo[x, n | {n, r}]; applied to a "
          "NearestFunction nf it gives nf[x, ...]")
def _nearestto(*args: Expr) -> Expr:
    if not 1 <= len([a for a in args if not is_rule(a)]) <= 2:
        raise ValueError("NearestTo takes a point x and optionally a count n or {n, r}")
    raise Unevaluated()


def _nearestto_apply(head, args, ev_):
    if len(args) != 1 or not head.args:
        return K.UNHANDLED
    target = args[0]
    pos = [a for a in head.args if not is_rule(a)]
    opts = [a for a in head.args if is_rule(a)]
    if isinstance(target, Expr) and target.head == "NearestFunction":
        return ev_(Expr(target, *pos))
    return ev_(Expr("Nearest", target, *pos, *opts))


KL.COMPOUND_HEAD_HOOKS["NearestTo"] = _nearestto_apply


# ─────────────────────────── NumericArray ───────────────────────────

_NA_INT = {"Integer8": (-2 ** 7, 2 ** 7 - 1), "UnsignedInteger8": (0, 2 ** 8 - 1),
           "Integer16": (-2 ** 15, 2 ** 15 - 1), "UnsignedInteger16": (0, 2 ** 16 - 1),
           "Integer32": (-2 ** 31, 2 ** 31 - 1), "UnsignedInteger32": (0, 2 ** 32 - 1),
           "Integer64": (-2 ** 63, 2 ** 63 - 1), "UnsignedInteger64": (0, 2 ** 64 - 1)}
_NA_REAL = {"Real32", "Real64"}
_NA_COMPLEX = {"ComplexReal32", "ComplexReal64"}
_NA_TYPES = set(_NA_INT) | _NA_REAL | _NA_COMPLEX
_NA_METHODS = {"Check", "Round", "ClipAndCheck", "ClipAndRound", "Coerce", "ClipAndCoerce"}
_F32_MAX = 3.4028234663852886e38


def _round_half_even(v):
    return int(sp.Integer(sp.floor(v + sp.Rational(1, 2)))) if (v - sp.floor(v)) != sp.Rational(1, 2) else \
        int(sp.floor(v)) + (int(sp.floor(v)) % 2)


def _na_convert_real(v, typ, method, fname):
    lo, hi = _NA_INT.get(typ, (None, None))
    if typ in _NA_INT:
        if v.is_Integer:
            iv = int(v)
        else:
            if method in ("Check", "ClipAndCheck"):
                raise ValueError(f"{fname}: {v} is not an integer (type {typ}, method {method})")
            iv = None
        if method.startswith("Clip"):
            if v < lo:
                return lo
            if v > hi:
                return hi
        if iv is None:
            iv = _round_half_even(sp.nsimplify(v) if isinstance(v, sp.Float) else v)
        if not lo <= iv <= hi:
            raise ValueError(f"{fname}: {v} is outside the range of {typ}")
        return iv
    f = float(v)
    if typ.endswith("32"):
        if abs(f) > _F32_MAX:
            if method.startswith("Clip"):
                f = math.copysign(_F32_MAX, f)
            else:
                raise ValueError(f"{fname}: {v} is outside the range of {typ}")
        import numpy as np
        f = float(np.float32(f))
    return f


def _na_convert(x, typ, method, fname):
    if is_list(x):
        return L(*(_na_convert(e, typ, method, fname) for e in x.args))
    if isinstance(x, bool) or isinstance(x, str) or not K.is_number(x):
        raise ValueError(f"{fname}: {x!r} is not a number")
    v = K.to_sp(x)
    re_, im_ = sp.re(v), sp.im(v)
    if typ in _NA_COMPLEX:
        a = _na_convert_real(re_, typ.replace("Complex", ""), method, fname)
        b = _na_convert_real(im_, typ.replace("Complex", ""), method, fname)
        return K.from_sp(sp.Float(a) + sp.I * sp.Float(b)) if b != 0 else \
            K.from_sp(sp.Float(a) + sp.I * sp.Float(0.0))
    if im_ != 0:
        raise ValueError(f"{fname}: complex value {x!r} does not fit type {typ}")
    return _na_convert_real(re_, typ, method, fname)


@fn("NumericArray", **ADDED,
    notes="NumericArray[array, type, method] typed numeric array (types Integer8 ... UnsignedInteger64, Real32, "
          "Real64, ComplexReal32, ComplexReal64; methods ClipAndRound (default), Round, Check, ClipAndCheck; "
          "Coerce methods raise); stored as NumericArray[values, type]; Normal, Length and Dimensions read it; "
          "NumericArray[array] without a type raises")
def _numericarray(a: Expr, typ: Expr = None, method: Expr = None) -> Expr:
    if typ is None:
        raise NotImplementedError("NumericArray: give the element type, e.g. NumericArray[list, \"Integer64\"]")
    if not isinstance(typ, str) or typ not in _NA_TYPES:
        raise ValueError(f"NumericArray: unknown type {typ!r}")
    if _na_valid(Expr("NumericArray", a, typ)) and method is None:
        from ..packed import is_packed, pack
        packed = None if typ in _NA_COMPLEX or is_packed(a) else pack(a)
        if packed is None:
            raise Unevaluated()              # already in normal form
        return Expr("NumericArray", packed, typ)
    m = "ClipAndRound" if method is None else method
    if not isinstance(m, str) or m not in _NA_METHODS:
        raise ValueError(f"NumericArray: unknown conversion method {m!r}")
    if m in ("Coerce", "ClipAndCoerce"):
        raise NotImplementedError(f"NumericArray: the \"{m}\" conversion method is not supported")
    if isinstance(a, Expr) and a.head == "NumericArray" and len(a.args) == 2:
        a = a.args[0]
    dims = _full_dims(a)
    if not dims or 0 in dims:
        raise ValueError("NumericArray: a non-empty full array of numbers is expected")
    vals = _na_convert(a, typ, m, "NumericArray")
    if typ not in _NA_COMPLEX:
        from ..packed import pack
        vals = pack(vals) or vals            # stored as one NumPy array (packed List) when it fits int64/float64
    return Expr("NumericArray", vals, typ)


def _na_valid(x):
    if not (isinstance(x, Expr) and x.head == "NumericArray" and len(x.args) == 2):
        return False
    vals, typ = x.args
    if not isinstance(typ, str) or typ not in _NA_TYPES:
        return False
    dims = _full_dims(vals)
    if not dims or 0 in dims:
        return False

    def ok(v):
        if typ in _NA_INT:
            lo, hi = _NA_INT[typ]
            return is_int(v) and lo <= int(v) <= hi
        if typ in _NA_REAL:
            return isinstance(v, float) or isinstance(v, sp.Float)
        return K.is_number(v) and not isinstance(v, (bool, str)) and (
            isinstance(v, (float, complex)) or bool(K.to_sp(v).atoms(sp.Float)))

    def walk(v, d):
        if d == 0:
            return ok(v)
        return all(walk(e, d - 1) for e in v.args)
    return walk(vals, len(dims))


@fn("NumericArrayQ", **ADDED,
    notes="True for a valid NumericArray; NumericArrayQ[expr, tpatt] also tests the type against a pattern, "
          "NumericArrayQ[expr, tpatt, dpatt] the array depth")
def _numericarrayq(x: Expr, tpatt: Expr = None, dpatt: Expr = None) -> Boolean:
    if not _na_valid(x):
        return False
    if tpatt is not None and K.first_match(tpatt, x.args[1], ev) is None:
        return False
    if dpatt is not None and K.first_match(dpatt, len(_full_dims(x.args[0])), ev) is None:
        return False
    return True


@fn("NumericArrayType", **ADDED, notes="the element type string of a NumericArray")
def _numericarraytype(x: Expr) -> Expr:
    if not _na_valid(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("NumericArrayType: a valid NumericArray is expected")
    return x.args[1]


# ─────────────────────────── symmetrized arrays ───────────────────────────

def _perm_from(spec, r, fname):
    """A permutation of 1..r (image list) from a list or Cycles[...]."""
    if isinstance(spec, Expr) and spec.head == "Cycles" and len(spec.args) == 1:
        img = list(range(1, r + 1))
        for cyc in spec.args[0].args:
            c = [int_value(v, fname, "cycle element", 1) for v in cyc.args]
            for i, v in enumerate(c):
                if v > r:
                    raise ValueError(f"{fname}: the symmetry acts on slot {v} beyond the rank {r}")
                img[v - 1] = c[(i + 1) % len(c)]
        return tuple(img)
    if is_list(spec) and all(is_int(v) for v in spec.args):
        p = [int(v) for v in spec.args]
        if sorted(p) != list(range(1, len(p) + 1)):
            raise ValueError(f"{fname}: {spec!r} is not a permutation list")
        if len(p) > r:
            raise ValueError(f"{fname}: the permutation {spec!r} is longer than the rank {r}")
        return tuple(p + list(range(len(p) + 1, r + 1)))
    raise ValueError(f"{fname}: invalid permutation {spec!r}")


def _slots(spec, r, fname):
    if is_sym(spec, "All"):
        return list(range(1, r + 1))
    if is_list(spec) and all(is_int(v) for v in spec.args):
        s = [int(v) for v in spec.args]
        if len(set(s)) != len(s) or any(not 1 <= v <= r for v in s):
            raise ValueError(f"{fname}: invalid slots {spec!r} for rank {r}")
        return s
    raise ValueError(f"{fname}: invalid slot specification {spec!r}")


def _sym_generators(sym, r, fname):
    """Generators [(perm, phase)] of a symmetry: Symmetric[slots], Antisymmetric[slots], {} or {{perm, phase}, ...}."""
    if isinstance(sym, Expr) and sym.head in ("Symmetric", "Antisymmetric") and len(sym.args) == 1:
        s = _slots(sym.args[0], r, fname)
        phase = 1 if sym.head == "Symmetric" else -1
        gens = []
        for a, b in zip(s, s[1:]):
            img = list(range(1, r + 1))
            img[a - 1], img[b - 1] = b, a
            gens.append((tuple(img), sp.Integer(phase)))
        return gens
    if is_list(sym):
        gens = []
        for g in sym.args:
            if not (is_list(g) and len(g.args) == 2):
                raise ValueError(f"{fname}: a symmetry generator must be {{permutation, phase}}")
            gens.append((_perm_from(g.args[0], r, fname), K.to_sp(g.args[1])))
        return gens
    raise ValueError(f"{fname}: invalid symmetry {sym!r}")


def _compose(p, q):
    """Permutation p after q (image lists, 1-based): (p∘q)(i) = p(q(i))."""
    return tuple(p[q[i] - 1] for i in range(len(q)))


def _group(gens, r, fname):
    ident = tuple(range(1, r + 1))
    elems = {ident: sp.Integer(1)}
    frontier = [ident]
    while frontier:
        nxt = []
        for g in frontier:
            for p, ph in gens:
                h = _compose(p, g)
                v = sp.simplify(ph * elems[g])
                if h in elems:
                    if sp.simplify(elems[h] - v) != 0:
                        raise ValueError(f"{fname}: the symmetry phases are inconsistent")
                    continue
                elems[h] = v
                nxt.append(h)
        frontier = nxt
        if len(elems) > 40320:
            raise NotImplementedError(f"{fname}: the symmetry group is too large")
    return elems


def _permute(idx, p):
    """Permute[idx, p]: element k of idx moves to position p[k]."""
    out = [None] * len(idx)
    for k, v in enumerate(idx):
        out[p[k] - 1] = v
    return tuple(out)


class _SymInfo:
    def __init__(self, dims, sym, fname):
        self.dims = tuple(dims)
        self.r = len(dims)
        self.group = _group(_sym_generators(sym, self.r, fname), self.r, fname)
        for p in self.group:
            for k in range(self.r):
                if dims[k] != dims[p[k] - 1]:
                    raise ValueError(f"{fname}: the symmetry is incompatible with the dimensions {list(dims)}")

    def orbit(self, idx):
        """{position: phase} with A[position] = phase * A[idx]."""
        out = {}
        for p, ph in self.group.items():
            j = _permute(idx, p)
            if j in out:
                if out[j] is not None and sp.simplify(out[j] - ph) != 0:
                    out[j] = None          # vanishing component
            else:
                out[j] = ph
        return out

    def rep(self, idx):
        """(representative, phase, vanishing) with A[idx] = phase * A[rep]."""
        orb = self.orbit(idx)
        vanish = any(v is None for v in orb.values())
        rep = min(orb)
        if vanish:
            return rep, sp.Integer(0), True
        # A[rep] = orb[rep] * A[idx]  =>  A[idx] = A[rep] / orb[rep]
        return rep, sp.simplify(1 / orb[rep]), False

    def independent(self):
        out = []
        for idx in itertools.product(*(range(1, d + 1) for d in self.dims)):
            rep, _ph, vanish = self.rep(idx)
            if rep == idx and not vanish:
                out.append(idx)
        return out


def _sa_parts(sa, fname):
    if not (isinstance(sa, Expr) and sa.head == "SymmetrizedArray" and len(sa.args) == 1 and
            isinstance(sa.args[0], Expr) and sa.args[0].head == "StructuredData" and len(sa.args[0].args) == 2):
        return None
    dims_e, inner = sa.args[0].args
    dims = [int(v) for v in dims_e.args]
    rules, sym = inner.args
    vals = {tuple(int(i) for i in r.args[0].args): r.args[1] for r in rules.args}
    return dims, vals, sym


def _sa_make(dims, vals, sym):
    rules = [Expr("Rule", L(*idx), v) for idx, v in sorted(vals.items()) if not _is_zero(v)]
    return Expr("SymmetrizedArray", Expr("StructuredData", L(*dims), L(L(*rules), sym)))


def _is_zero(v):
    return (is_int(v) and int(v) == 0) or (isinstance(v, float) and v == 0.0)


def _sa_value(info, vals, idx):
    rep, ph, vanish = info.rep(idx)
    if vanish:
        return 0
    v = vals.get(rep, 0)
    if _is_zero(v):
        return 0
    return ev(Expr("Times", K.from_sp(ph), v))


def _sa_normal(dims, vals, sym, fname="Normal"):
    info = _SymInfo(dims, sym, fname)

    def build(prefix):
        if len(prefix) == len(dims):
            return _sa_value(info, vals, tuple(prefix))
        return L(*(build(prefix + [i]) for i in range(1, dims[len(prefix)] + 1)))
    return build([])


def _assign_rules(info, rules, fname, base=None):
    """Independent values from rules pos -> val (explicit positions are mapped to their representative; pattern
    positions are evaluated for each independent component they match)."""
    vals = dict(base or {})
    explicit = [r for r in rules if is_list(r.args[0]) and all(is_int(v) for v in r.args[0].args)]
    patterned = [r for r in rules if r not in explicit]
    done = set()
    for r in explicit:
        idx = tuple(int(v) for v in r.args[0].args)
        if len(idx) != info.r or any(not 1 <= i <= d for i, d in zip(idx, info.dims)):
            raise ValueError(f"{fname}: position {list(idx)} is outside the dimensions {list(info.dims)}")
        rep, ph, vanish = info.rep(idx)
        v = ev(r.args[1]) if r.head == "RuleDelayed" else r.args[1]
        if vanish:
            if not _is_zero(v):
                raise ValueError(f"{fname}: component {list(idx)} vanishes by symmetry but was given {v!r}")
            continue
        if rep in done:
            continue
        done.add(rep)
        vals[rep] = ev(Expr("Times", K.from_sp(1 / ph), v))
    if patterned:
        for idx in info.independent():
            if idx in done:
                continue
            for r in patterned:
                b = K.first_match(r.args[0], L(*idx), ev)
                if b is not None:
                    vals[idx] = ev(K.substitute(r.args[1], b))
                    done.add(idx)
                    break
    return vals


def _rules_from(x, fname):
    if is_rule(x):
        return [x]
    if is_list(x) and all(is_rule(r) for r in x.args):
        return list(x.args)
    raise ValueError(f"{fname}: a list of rules pos -> val is expected")


def _detect_symmetry(arr, dims):
    """Symmetric[slots] / Antisymmetric[slots] / {} / generator list for the slot-permutation symmetry of arr."""
    r = len(dims)
    nested = arr

    def get(idx):
        v = nested
        for i in idx:
            v = v.args[i - 1]
        return v
    idxs = list(itertools.product(*(range(1, d + 1) for d in dims)))
    found = []
    for p in itertools.permutations(range(1, r + 1)):
        if p == tuple(range(1, r + 1)):
            continue
        if any(dims[k] != dims[p[k] - 1] for k in range(r)):
            continue
        for phase in (1, -1):
            if all(same(get(_permute(i, p)), ev(Expr("Times", phase, get(i)))) for i in idxs):
                found.append((p, phase))
                break
    if not found:
        return L()
    moved = sorted({k + 1 for p, _ in found for k in range(r) if p[k] != k + 1})
    nfull = math.factorial(len(moved))
    if len(found) + 1 == nfull:
        signs = {ph == _perm_sign(p) for p, ph in found}
        if all(ph == 1 for _, ph in found):
            return Expr("Symmetric", L(*moved) if len(moved) < r else Symbol("All"))
        if signs == {True}:
            return Expr("Antisymmetric", L(*moved) if len(moved) < r else Symbol("All"))
    return L(*(L(L(*p), ph) for p, ph in found))


def _perm_sign(p):
    p = list(p)
    s = 1
    for i in range(len(p)):
        while p[i] != i + 1:
            j = p[i] - 1
            p[i], p[j] = p[j], p[i]
            s = -s
    return s


@fn("SymmetrizedArray", **ADDED,
    notes="SymmetrizedArray[{pos -> val, ...}, dims, sym] with sym Symmetric[slots], Antisymmetric[slots], {} or "
          "{{perm, phase}, ...}: stored as SymmetrizedArray[StructuredData[dims, {independent rules, sym}]] (the "
          "lexicographically smallest member of each orbit is independent; dependent positions are converted with "
          "their phase; pattern positions evaluated per independent component; unspecified entries 0); "
          "SymmetrizedArray[list] detects the slot-permutation symmetry; Normal gives the full array")
def _symarray(x: Expr, dims: Expr = None, sym: Expr = None) -> Expr:
    if isinstance(x, Expr) and x.head == "StructuredData":
        raise Unevaluated()
    if dims is None:
        d = _full_dims(x)
        if not d or 0 in d:
            raise ValueError("SymmetrizedArray: a non-empty full array or a list of rules is expected")
        s = _detect_symmetry(x, d)
        info = _SymInfo(d, s, "SymmetrizedArray")
        vals = {}
        for idx in info.independent():
            v = x
            for i in idx:
                v = v.args[i - 1]
            vals[idx] = v
        return _sa_make(d, vals, s)
    rules = _rules_from(x, "SymmetrizedArray")
    d = [int_value(v, "SymmetrizedArray", "dimension", 1) for v in (dims.args if is_list(dims) else [dims])]
    s = L() if sym is None else sym
    info = _SymInfo(d, s, "SymmetrizedArray")
    return _sa_make(d, _assign_rules(info, rules, "SymmetrizedArray"), s)


@fn("SymmetrizedArrayRules", **ADDED,
    notes="independent nonzero rules of a SymmetrizedArray followed by {_, _, ...} -> 0; for an ordinary array, the "
          "rules of the array symmetrized with sym (Symmetrize: the phase-weighted group average); without sym the "
          "array's own nonzero rules")
def _symarrayrules(x: Expr, sym: Expr = None) -> ListOf[Expr]:
    parts = _sa_parts(x, "SymmetrizedArrayRules")
    if parts is not None:
        dims, vals, _s = parts
    else:
        dims = _full_dims(x)
        if not dims or 0 in dims:
            if isinstance(x, Symbol):
                raise Unevaluated()
            raise ValueError("SymmetrizedArrayRules: a symmetrized array or a full array is expected")
        s = L() if sym is None else sym
        info = _SymInfo(dims, s, "SymmetrizedArrayRules")

        def get(idx):
            v = x
            for i in idx:
                v = v.args[i - 1]
            return v
        n = len(info.group)
        vals = {}
        for idx in info.independent():
            # Symmetrize: S[i] = (1/|G|) sum_g phase_g^-1 A[Permute^-1 ...]; with A[Permute(i, p)] = ph S[i] this is
            # the average of conj-phase-weighted orbit values
            terms = [Expr("Times", K.from_sp(1 / ph), get(_permute(idx, p))) for p, ph in info.group.items()]
            vals[idx] = ev(Expr("Times", Expr("Power", n, -1), Expr("Plus", *terms)))
    rules = [Expr("Rule", L(*idx), v) for idx, v in sorted(vals.items()) if not _is_zero(v)]
    rules.append(Expr("Rule", L(*([Expr("Blank")] * len(dims))), 0))
    return L(*rules)


@fn("SymmetrizedIndependentComponents", **ADDED,
    notes="independent components (lexicographically smallest orbit members, vanishing orbits excluded) of an array "
          "with dimensions dims and symmetry sym, in lexicographic order")
def _symindep(dims: Expr, sym: Expr) -> ListOf[Expr]:
    d = [int_value(v, "SymmetrizedIndependentComponents", "dimension", 1) for v in dims.args] if is_list(dims) else None
    if d is None:
        raise ValueError("SymmetrizedIndependentComponents: dimensions must be a list of positive integers")
    info = _SymInfo(d, sym, "SymmetrizedIndependentComponents")
    return L(*(L(*idx) for idx in info.independent()))


@fn("SymmetrizedDependentComponents", **ADDED,
    notes="sorted list of the components related to comp by the symmetry sym (its orbit, including comp)")
def _symdep(comp: Expr, sym: Expr) -> ListOf[Expr]:
    if not (is_list(comp) and comp.args and all(is_int(v) and int(v) >= 1 for v in comp.args)):
        raise ValueError("SymmetrizedDependentComponents: the component must be a list of positive integers")
    idx = tuple(int(v) for v in comp.args)
    info = _SymInfo([max(idx)] * len(idx), sym, "SymmetrizedDependentComponents")
    return L(*(L(*j) for j in sorted(info.orbit(idx))))


def _same_array(a, b) -> bool:
    """Elementwise equality of two nested lists (entries compared after expanding their difference)."""
    if is_list(a) and is_list(b):
        return len(a.args) == len(b.args) and all(_same_array(p, q) for p, q in zip(a.args, b.args))
    if is_list(a) or is_list(b):
        return False
    if a == b:
        return True
    try:
        return sp.expand(S(a) - S(b)) == 0
    except Exception:
        return False


@fn("SymmetrizedReplacePart", **ADDED,
    notes="SymmetrizedReplacePart[sa, {pos -> val, ...}] replaces independent values (dependent positions are mapped "
          "to their independent component); SymmetrizedReplacePart[array, rules, sym] needs an array that already has the "
          "symmetry sym (non-symmetric input raises instead of being averaged). Check: reference; tests compare "
          "hand-derived symmetric/antisymmetric results and the raised non-symmetric cases.")
def _symreplace(x: Expr, rules: Expr, sym: Expr = None) -> Expr:
    parts = _sa_parts(x, "SymmetrizedReplacePart")
    if parts is None:
        if sym is None:
            raise ValueError("SymmetrizedReplacePart: give the symmetry for an ordinary array")
        base = _symarrayrules(x, sym)
        dims = _full_dims(x)
        vals = {tuple(int(i) for i in r.args[0].args): r.args[1] for r in base.args[:-1]}
        s = sym
        if not _same_array(_sa_normal(dims, vals, s), x):
            raise ValueError("SymmetrizedReplacePart: the array does not have the given symmetry (it is not "
                             "symmetrized implicitly; use SymmetrizedArray[array, dims, sym] to symmetrize first)")
    else:
        dims, vals, s = parts
    info = _SymInfo(dims, s, "SymmetrizedReplacePart")
    new = _sa_make(dims, _assign_rules(info, _rules_from(rules, "SymmetrizedReplacePart"), "SymmetrizedReplacePart",
                                       vals), s)
    return new if parts is not None else _sa_normal(dims, _sa_parts(new, "")[1], s)


# ─────────────────────────── Normal / Length / Dimensions for the new raw objects ───────────────────────────

_prev_normal = K._BUILTINS.get("Normal")


def _w03_normal(a, ev_, expr):
    if len(a) == 1 and isinstance(a[0], Expr):
        x = a[0]
        if x.head == "NumericArray" and _na_valid(x):
            return x.args[0]
        if x.head == "Dispatch" and len(x.args) == 1:
            return x.args[0]
        parts = _sa_parts(x, "Normal")
        if parts is not None:
            return _sa_normal(*parts)
    return _prev_normal(a, ev_, expr)


K._BUILTINS["Normal"] = _w03_normal

_prev_length = K._BUILTINS.get("Length")


def _w03_length(a, ev_, expr):
    if len(a) == 1 and isinstance(a[0], Expr) and a[0].head == "NumericArray" and _na_valid(a[0]):
        return len(a[0].args[0].args)
    return _prev_length(a, ev_, expr)


K._BUILTINS["Length"] = _w03_length


def _wrap_dimensions():
    e = REGISTRY.get("Dimensions")
    if e is None:
        return
    impls = list(e.impls)

    def dims(x, n=None):
        if isinstance(x, Expr) and x.head == "NumericArray" and _na_valid(x):
            d = _full_dims(x.args[0])
            return L(*(d if n is None else d[:int_value(n, "Dimensions", "depth", 0)]))
        parts = _sa_parts(x, "Dimensions") if isinstance(x, Expr) else None
        if parts is not None:
            return L(*(parts[0] if n is None else parts[0][:int_value(n, "Dimensions", "depth", 0)]))
        for f_, lo, hi in impls:
            if (1 if n is None else 2) >= lo and (hi is None or (1 if n is None else 2) <= hi):
                return f_(x) if n is None else f_(x, n)
        raise ValueError("Dimensions called with the wrong number of arguments")
    e.impls[:] = [(dims, 1, 2)]


_wrap_dimensions()


# ─────────────────────────── option symbols of other functions ───────────────────────────

for _n, _note in [
        ("SortedBy", "option specifying a function by which to sort the results of a computation; an inert option "
                     "name (functions accepting it validate it)"),
        ("ComponentKeys", "option naming the data components of a time or event series; an inert option name"),
        ("DeleteContents", "option for DeleteDirectory (whether to delete the directory contents); an inert option "
                           "name"),
        ("KeyCollisionFunction", "option for JoinAcross (how to handle same-key elements not being joined); an inert "
                                 "option name"),
        ("PaddingSize", "option of AudioDelay and other functions giving the amount of padding; an inert option name"),
        ("PartBehavior", "option of Query and related functions for nonexistent parts; an inert option name"),
        ("WrapAround", "option for NotebookFind (continue past the end of a document); an inert option name"),
        ("RowsEqual", "option for GridBox (equal row heights); an inert option name"),
        ("ColumnsEqual", "option for GridBox (equal column widths); an inert option name")]:
    record(_n, AREA, "added", _note)


# ─────────────────────────── interpreter / form element specifications ───────────────────────────

def _count_bound(v, fname):
    if is_sym(v, "Infinity"):
        return math.inf
    return int_value(v, fname, "count", 0)


def _check_range(spec, fname):
    """max | {min, max} | {n, {min, max}} | {{i, n}, {min, max}}"""
    if is_int(spec) or is_sym(spec, "Infinity"):
        _count_bound(spec, fname)
        return
    if is_list(spec) and len(spec.args) == 2:
        a, b = spec.args
        if is_list(b) and len(b.args) == 2:
            lo, hi = (_count_bound(v, fname) for v in b.args)
            if lo > hi:
                raise ValueError(f"{fname}: the minimum count exceeds the maximum")
            if is_list(a):
                if len(a.args) != 2 or not isinstance(a.args[0], Symbol):
                    raise ValueError(f"{fname}: expected {{{{i, n}}, {{min, max}}}}")
                n = _count_bound(a.args[1], fname)
            else:
                n = _count_bound(a, fname)
            if not lo <= n <= hi:
                raise ValueError(f"{fname}: the initial count must lie between min and max")
            return
        lo, hi = _count_bound(a, fname), _count_bound(b, fname)
        if lo > hi:
            raise ValueError(f"{fname}: the minimum count exceeds the maximum")
        return
    raise ValueError(f"{fname}: invalid count specification {spec!r}")


@fn("RepeatingElement", **ADDED,
    notes="RepeatingElement[spec], RepeatingElement[spec, max | {min, max} | {n, {min, max}} | {{i, n}, {min, max}}]: "
          "an inert interpreter/form specification (count specifications are validated; Interpreter and forms are "
          "not available)")
def _repeating(spec: Expr, rng: Expr = None) -> Expr:
    if rng is not None:
        _check_range(rng, "RepeatingElement")
    raise Unevaluated()


@fn("AnySubset", **ADDED,
    notes="AnySubset[{c1, ...} | {lab -> c, ...} | EntityClass[...]], AnySubset[choices, max | {min, max}]: an inert "
          "interpreter/form specification (arguments are validated)")
def _anysubset(choices: Expr, rng: Expr = None) -> Expr:
    if not (is_list(choices) or (isinstance(choices, Expr) and choices.head == "EntityClass")
            or isinstance(choices, Symbol)):
        raise ValueError("AnySubset: the choices must be a list {c1, c2, ...} or {lab1 -> c1, ...}")
    if rng is not None:
        if is_int(rng) or is_sym(rng, "Infinity"):
            _count_bound(rng, "AnySubset")
        elif is_list(rng) and len(rng.args) == 2 and not any(is_list(v) for v in rng.args):
            _check_range(rng, "AnySubset")
        else:
            raise ValueError(f"AnySubset: invalid count specification {rng!r}")
    raise Unevaluated()


# ─────────────────────────── ArrayResample ───────────────────────────

def _rs_count(spec, m, fname):
    if is_sym(spec, "All"):
        return m
    if isinstance(spec, Expr) and spec.head == "Scaled" and len(spec.args) == 1:
        s = real_number(spec.args[0], fname, "scale factor")
        if s <= 0:
            raise ValueError(f"{fname}: the scale factor must be positive")
        return int(sp.floor(s * (m - 1) + sp.Rational(1, 2))) + 1 if m > 1 else 1
    if is_sym(spec, "Automatic"):
        raise NotImplementedError(f"{fname}: Automatic dimension specifications are not supported; give each "
                                  "dimension explicitly")
    return int_value(spec, fname, "dimension", 1)


def _rs_line(vals, n, kind, fname):
    m = len(vals)
    if n == m:
        return list(vals)
    if m == 1 or n == 1:
        raise NotImplementedError(f"{fname}: resampling to or from a single sample is not supported")
    out = []
    for k in range(n):
        p = sp.Rational(k * (m - 1), n - 1)          # 0-based sample position
        i = int(sp.floor(p))
        t = p - i
        if t == 0:
            out.append(vals[i])
            continue
        a, b = vals[i], vals[i + 1]
        if kind == "Linear":
            out.append(ev(Expr("Plus", Expr("Times", K.from_sp(1 - t), a), Expr("Times", K.from_sp(t), b))))
        elif kind == "Nearest":
            if t == sp.Rational(1, 2):
                out.append(ev(Expr("Times", K.from_sp(sp.Rational(1, 2)), Expr("Plus", a, b))))
            else:
                out.append(a if t < sp.Rational(1, 2) else b)
        elif kind == "NearestLeft":
            out.append(a if t <= sp.Rational(1, 2) else b)
        else:                                          # NearestRight
            out.append(a if t < sp.Rational(1, 2) else b)
    return out


def _rs_dim(nested, axis, n, kind, fname, below):
    if axis == 0:
        vals = [_from_nested_any(v) for v in nested]
        return [_to_nested(v, below) for v in _rs_line(vals, n, kind, fname)]
    return [_rs_dim(sub, axis - 1, n, kind, fname, below) for sub in nested]


@fn("ArrayResample", **ADDED,
    notes="ArrayResample[array, n | Scaled[s] | All | {d1, d2, ...}] and ArrayResample[array, dspec, \"Point\"]: "
          "point sampling on 1..m (n samples at 1 + (k-1)(m-1)/(n-1); Scaled[s] gives Round[s (m-1)] + 1 samples), "
          "separably along each dimension, exact for exact data; Resampling -> \"Linear\" (default), \"Nearest\" "
          "(halfway averages), \"NearestLeft\", \"NearestRight\". \"Bin\" schemes, subranges, Automatic dimensions, "
          "DataRange, Antialiasing -> True and other kernels raise")
def _arrayresample(a: Expr, dspec: Expr, *rest: Expr) -> Expr:
    fname = "ArrayResample"
    pos, opts = split_options(rest, {"Resampling", "Padding", "DataRange", "Antialiasing"}, fname)
    if pos:
        scheme = pos[0]
        if not (scheme == "Point" or (is_list(scheme) and scheme.args and scheme.args[0] == "Point")):
            raise NotImplementedError(f"{fname}: only the \"Point\" sampling scheme is supported")
        if len(pos) > 1:
            raise NotImplementedError(f"{fname}: resampling a subrange is not supported")
    kind = opts.get("Resampling", "Linear")
    if is_sym(kind, "Automatic"):
        kind = "Linear"
    if kind not in ("Linear", "Nearest", "NearestLeft", "NearestRight"):
        raise NotImplementedError(f"{fname}: Resampling -> {kind!r} is not supported (use \"Linear\", \"Nearest\", "
                                  "\"NearestLeft\" or \"NearestRight\")")
    if "DataRange" in opts and not is_sym(opts["DataRange"], "Automatic"):
        raise NotImplementedError(f"{fname}: the DataRange option is not supported")
    if opts.get("Antialiasing", False) not in (False,):
        raise NotImplementedError(f"{fname}: Antialiasing -> True is not supported")
    dims = _full_dims(a)
    if not dims or 0 in dims:
        if isinstance(a, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: a non-empty full array is expected")
    if is_list(dspec):
        specs = list(dspec.args)
        if len(specs) > len(dims):
            raise ValueError(f"{fname}: more dimensions specified than the array has")
        specs += [Symbol("All")] * (len(dims) - len(specs))
    elif len(dims) == 1 or is_sym(dspec, "All"):
        specs = [dspec] * len(dims)
    elif isinstance(dspec, Expr) and dspec.head == "Scaled":
        specs = [dspec] * len(dims)
    else:
        raise NotImplementedError(f"{fname}: for a multidimensional array give {{n1, n2, ...}} (n alone means "
                                  "{n, Automatic, ...}, which is not supported)")
    nested = _to_nested(a, len(dims))
    for axis, (s, m) in enumerate(zip(specs, dims)):
        n = _rs_count(s, m, fname)
        nested = _rs_dim(nested, axis, n, kind, fname, len(dims) - axis - 1)
    return _from_nested_any(nested)


# ─────────────────────────── incremental functions and objects ───────────────────────────
#
# MikoshiLang has no Wolfram Compiler, so IncrementalFunction[f][args] creates the running instance directly
# (Wolfram requires FunctionCompile[IncrementalFunction[f]][args]).  The instance runs f in its own thread that
# strictly alternates with the caller, so only one of them evaluates at any time.

import contextvars as _cv
import threading as _th

_INC_LOCAL = _th.local()


def _inc_failure(tag, msg):
    return Expr("Failure", tag, Expr("Association", Expr("Rule", "MessageTemplate", msg)))


class _IncrementalFunctionInstance:
    type = "IncrementalFunction"

    def __init__(self, f, args):
        self.f, self.args = f, args
        self.state = "new"            # new | paused | done
        self.msg = None
        self.resume = None            # (sent?, value)
        self.to_worker = _th.Semaphore(0)
        self.to_caller = _th.Semaphore(0)

    def copy(self):
        raise ValueError("IncrementalFunction: a running incremental function cannot be copied")

    def normal(self):
        raise ValueError("IncrementalFunction: an incremental function instance has no Normal form")

    def length(self):
        raise ValueError("IncrementalFunction: an incremental function instance has no Length")

    def _worker(self):
        _INC_LOCAL.current = self
        try:
            apply(self.f, *self.args)
            self.msg = ("done", None)
        except BaseException as e:          # noqa: BLE001 -- re-raised in the caller's thread
            self.msg = ("error", e)
        self.to_caller.release()

    def pause(self, kind, payload):
        self.msg = (kind, payload)
        self.to_caller.release()
        self.to_worker.acquire()
        return self.resume

    def _run(self, sent, value):
        if self.state == "done":
            return _inc_failure("IncrementalFunctionTerminated", "The incremental function has terminated.")
        self.resume = (sent, value)
        if self.state == "new":
            ctx = _cv.copy_context()
            old = _th.stack_size()
            _th.stack_size(256 * 1024 * 1024)
            try:
                t = _th.Thread(target=ctx.run, args=(self._worker,), daemon=True)
                t.start()
            finally:
                _th.stack_size(old)
        else:
            self.to_worker.release()
        self.to_caller.acquire()
        kind, payload = self.msg
        if kind == "yield":
            self.state = "paused"
            return payload
        if kind == "receive":
            self.state = "paused"
            return None
        self.state = "done"
        if kind == "error":
            raise payload
        return _inc_failure("IncrementalFunctionTerminated", "The incremental function has terminated.")

    def op_Next(self):
        return self._run(False, None)

    def op_Send(self, x):
        if self.state == "new":
            raise ValueError("IncrementalFunction: an incremental function must be started with \"Next\" "
                             "before values can be sent to it")
        return self._run(True, x)


def _inc_current(fname):
    cur = getattr(_INC_LOCAL, "current", None)
    if cur is None:
        raise ValueError(f"{fname}: can only be used inside a running IncrementalFunction")
    return cur


@fn("IncrementalFunction", **ADDED,
    notes="IncrementalFunction[f] is an inert function that can pause; IncrementalFunction[f][args] creates a running "
          "instance DataStructure[\"IncrementalFunction\", ...] (no FunctionCompile step is needed). ds[\"Next\"] runs "
          "to the next IncrementalYield (giving its value) or IncrementalReceive (giving Null); ds[\"Send\", x] "
          "resumes, making the pending IncrementalReceive return x; once the code ends the call gives "
          "Failure[\"IncrementalFunctionTerminated\", ...]. Errors inside the function are raised to the caller")
def _incfunction(f: Expr) -> Expr:
    raise Unevaluated()


def _incfunction_apply(head, args, ev_):
    if len(head.args) != 1:
        return K.UNHANDLED
    from . import wl_datastruct as DS
    return DS._wrap(_IncrementalFunctionInstance(head.args[0], list(args)))


KL.COMPOUND_HEAD_HOOKS["IncrementalFunction"] = _incfunction_apply


@fn("IncrementalYield", **ADDED,
    notes="IncrementalYield[e] pauses the enclosing running IncrementalFunction, which then gives e; outside one it "
          "raises")
def _incyield(e: Expr) -> Expr:
    _inc_current("IncrementalYield").pause("yield", e)
    return None


@fn("IncrementalReceive", **ADDED,
    notes="IncrementalReceive[] / IncrementalReceive[default] pauses the enclosing running IncrementalFunction; when "
          "resumed with \"Send\" it returns the sent value, with \"Next\" the default (Null if none); outside one it "
          "raises")
def _increceive(default: Expr = None) -> Expr:
    sent, value = _inc_current("IncrementalReceive").pause("receive", None)
    return value if sent else default


_INC_OBJECTS = {}
_INC_COUNTER = itertools.count(1)
_INC_NAMES = ("Identity", "Range", "Select", "Take", "Map", "FoldList", "Permutations", "Subsets", "Tuples")
_DONE = object()


def _inc_obj(x):
    if isinstance(x, Expr) and x.head == "IncrementalObject" and len(x.args) == 2 and \
            isinstance(x.args[1], Expr) and x.args[1].head == "IncrementalObjectHandle" and \
            isinstance(x.args[1].args[0], int):
        return _INC_OBJECTS.get(x.args[1].args[0])
    return None


def _inc_fn_instance(x):
    from . import wl_datastruct as DS
    if DS.is_ds(x) and x.args[0] == "IncrementalFunction":
        return DS._get(x, "NextValue")
    return None


def _source_iter(src, fname):
    """A Python iterator over a source: an expression's elements, an incremental object or function instance."""
    g = _inc_obj(src)
    if g is not None:
        def it_obj():
            while True:
                v = next(g, _DONE)
                if v is _DONE:
                    return
                yield v
        return it_obj()
    inst = _inc_fn_instance(src)
    if inst is not None:
        def it_fn():
            while True:
                v = inst.op_Next()
                if isinstance(v, Expr) and v.head == "Failure":
                    return
                yield v
        return it_fn()
    if is_assoc(src):
        return iter(KL.assoc_values(src, ev))
    if isinstance(src, Expr):
        return iter(src.args)
    raise ValueError(f"{fname}: {src!r} is not a list, an expression, an incremental object or an incremental "
                     "function instance")


def _finite_items(src, fname):
    if isinstance(src, Expr) and _inc_obj(src) is None and _inc_fn_instance(src) is None and not is_assoc(src):
        return list(src.args)
    raise ValueError(f"{fname}: the source must be a list")


def _inc_range(args, fname):
    if not 1 <= len(args) <= 3:
        raise ValueError(f"{fname}: \"Range\" takes 1 to 3 arguments as in Range")
    vals = [real_number(a, fname, "Range bound") if not is_sym(a, "Infinity") else sp.oo for a in args]
    if len(vals) == 1:
        a, b, d = sp.Integer(1), vals[0], sp.Integer(1)
    elif len(vals) == 2:
        (a, b), d = vals, sp.Integer(1)
    else:
        a, b, d = vals
    if d == 0:
        raise ValueError(f"{fname}: the Range step must be nonzero")
    exact = all(not isinstance(v, sp.Float) for v in (a, b, d))

    def gen():
        k = 0
        while True:
            v = a + k * d
            if (d > 0 and v > b) or (d < 0 and v < b):
                return
            yield K.from_sp(v if exact else sp.Float(v))
            k += 1
    return gen()


def _inc_permutations(items):
    tags = [K.tag(x) for x in items]
    distinct = len(set(tags)) == len(tags)
    seen = set()
    for p in itertools.permutations(range(len(items))):
        if not distinct:
            key = tuple(tags[i] for i in p)
            if key in seen:
                continue
            seen.add(key)
        yield L(*(items[i] for i in p))


def _make_generator(name, args):
    fname = f"IncrementalObject[\"{name}\"]"
    if name == "Identity":
        if len(args) != 1:
            raise ValueError(f"{fname}: one source expected")
        return _source_iter(args[0], fname)
    if name == "Range":
        return _inc_range(args, fname)
    if name == "Select":
        if len(args) != 2:
            raise ValueError(f"{fname}: expected [source, crit]")
        src, crit = _source_iter(args[0], fname), args[1]
        return (x for x in src if true(crit, x))
    if name == "Take":
        if len(args) != 2:
            raise ValueError(f"{fname}: expected [source, n]")
        return itertools.islice(_source_iter(args[0], fname), int_value(args[1], fname, "count", 0))
    if name == "Map":
        if len(args) != 2:
            raise ValueError(f"{fname}: expected [f, source]")
        f, src = args[0], _source_iter(args[1], fname)
        return (apply(f, x) for x in src)
    if name == "FoldList":
        if len(args) != 3:
            raise ValueError(f"{fname}: expected [f, x, source]")
        f, x0, src = args[0], args[1], _source_iter(args[2], fname)

        def fold():
            acc = x0
            yield acc
            for x in src:
                acc = apply(f, acc, x)
                yield acc
        return fold()
    if name == "Permutations":
        if len(args) != 1:
            raise ValueError(f"{fname}: expected [list]")
        return _inc_permutations(_finite_items(args[0], fname))
    if name == "Subsets":
        if len(args) != 1:
            raise ValueError(f"{fname}: expected [list]")
        items = _finite_items(args[0], fname)
        return (L(*c) for k in range(len(items) + 1) for c in itertools.combinations(items, k))
    if name == "Tuples":
        if len(args) == 2:
            items = _finite_items(args[0], fname)
            return (L(*t) for t in itertools.product(items, repeat=int_value(args[1], fname, "length", 0)))
        if len(args) == 1:
            lists = [_finite_items(x, fname) for x in _finite_items(args[0], fname)]
            return (L(*t) for t in itertools.product(*lists))
        raise ValueError(f"{fname}: expected [list, n] or [{{list1, list2, ...}}]")
    raise ValueError(f"IncrementalObject: unknown name {name!r}; known: {', '.join(_INC_NAMES)}")


@fn("IncrementalObject", **ADDED,
    notes="IncrementalObject[\"name\"][args] for \"Identity\" [expr], \"Range\" [Range args], \"Select\" [src, crit], "
          "\"Take\" [src, n], \"Map\" [f, src], \"FoldList\" [f, x, src], \"Permutations\" [list], \"Subsets\" [list], "
          "\"Tuples\" [list, n | {lists}]: a lazy stateful iterator; sources may be expressions, incremental objects "
          "or incremental function instances (consumed, not copied). Values are taken with NextValue; Table/Do "
          "iteration over an incremental object is not supported")
def _incobject(name: Expr, handle: Expr = None) -> Expr:
    if handle is not None:
        if isinstance(handle, Expr) and handle.head == "IncrementalObjectHandle":
            raise Unevaluated()
        raise ValueError("IncrementalObject: expected IncrementalObject[\"name\"]")
    if name not in _INC_NAMES:
        raise ValueError(f"IncrementalObject: unknown name {name!r}; known: {', '.join(_INC_NAMES)}")
    raise Unevaluated()


def _incobject_apply(head, args, ev_):
    if len(head.args) != 1 or not isinstance(head.args[0], str):
        return K.UNHANDLED
    name = head.args[0]
    gen = _make_generator(name, list(args))
    h = next(_INC_COUNTER)
    _INC_OBJECTS[h] = gen
    return Expr("IncrementalObject", name, Expr("IncrementalObjectHandle", h))


KL.COMPOUND_HEAD_HOOKS["IncrementalObject"] = _incobject_apply


@fn("NextValue", **ADDED,
    notes="NextValue[obj] gives the next value of an incremental object (Failure[\"IncrementalObjectTerminated\", "
          "...] when exhausted) or of a running incremental function instance (as ds[\"Next\"])")
def _nextvalue(obj: Expr) -> Expr:
    g = _inc_obj(obj)
    if g is not None:
        v = next(g, _DONE)
        if v is _DONE:
            return _inc_failure("IncrementalObjectTerminated", "The incremental object has no more values.")
        return v
    inst = _inc_fn_instance(obj)
    if inst is not None:
        return inst.op_Next()
    if isinstance(obj, Symbol):
        raise Unevaluated()
    raise ValueError(f"NextValue: {obj!r} is not an incremental object")
