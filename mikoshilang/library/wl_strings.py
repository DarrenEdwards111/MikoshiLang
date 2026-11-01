"""Batch 1 (strings): Wolfram string patterns and string functions.

String patterns (StringExpression ``~~``, ``_``/``__``/``___``, named patterns with back-references,
Alternatives, lists of alternatives, Repeated, Shortest/Longest, Except, character classes,
StartOfString/EndOfString/StartOfLine/EndOfLine/WordBoundary, Whitespace, NumberString,
RegularExpression and ``_?test``) compile to Python regular expressions.  Defaults follow Wolfram:
matching is greedy (Longest), StringCases/StringCount/StringReplace scan without overlaps and
StringPosition with Overlaps -> True.  Conditions (``p /; test``) inside string patterns are not
supported and raise.
"""
from __future__ import annotations

import re
import unicodedata

from ..expr import Expr, Symbol
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated, nonneg_int, pos_int, int_arg, string_arg
from .wl_common import L, ev, apply, is_sym, is_list, is_rule, split_options

AREA = "strings (batch 1)"
_FAILED = Symbol("$Failed")


def _s(x, fname, what="argument"):
    return string_arg(x, fname, what)


# ---------------------------------------------------------------- pattern compiler
_CLASSES = {
    "DigitCharacter": "[0-9]",
    "LetterCharacter": r"[^\W\d_]",
    "WordCharacter": r"[^\W_]",
    "WhitespaceCharacter": r"\s",
    "PunctuationCharacter": r"[!-/:-@\[-`{-~\u00a1\u00a7\u00ab\u00b6\u00b7\u00bb\u00bf\u2010-\u2027\u2030-\u205e]",
    "HexadecimalCharacter": "[0-9A-Fa-f]",
    "StartOfString": r"\A",
    "EndOfString": r"\Z",
    "StartOfLine": r"(?m:^)",
    "EndOfLine": r"(?m:$)",
    "WordBoundary": r"\b",
    "Whitespace": r"\s+",
    "NumberString": r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
}
_SINGLE = {"DigitCharacter", "LetterCharacter", "WordCharacter", "WhitespaceCharacter", "PunctuationCharacter",
           "HexadecimalCharacter"}
# character predicates whose whole-string test equals a per-character test
_CHAR_TESTS = {"LetterQ": r"[^\W\d_]", "DigitQ": "[0-9]", "UpperCaseQ": None, "LowerCaseQ": None}


class _Ctx:
    def __init__(self, subject):
        self.subject = subject            # used to build classes for _?test from the characters present
        self.names = {}                   # Wolfram pattern name -> regex group name
        self.regex_groups = False

    def group(self, name):
        g = self.names.get(name)
        if g is None:
            g = self.names[name] = f"g{len(self.names)}"
            return g, True
        return g, False


def _blank(head, lazy):
    q = {"Blank": "", "BlankSequence": "+", "BlankNullSequence": "*"}[head]
    return "." + q + ("?" if lazy and q else "")


def _char_class(test, ctx):
    """Regex class of the subject's characters c with test[c] True."""
    name = test.name if isinstance(test, Symbol) else None
    if name == "LetterQ":
        return r"[^\W\d_]"
    if name == "DigitQ":
        return "[0-9]"
    chars = sorted(set(ctx.subject))
    ok = [c for c in chars if apply(test, c) is True]
    if not ok:
        return "(?!)"
    return "[" + "".join(re.escape(c) for c in ok) + "]"


def _compile(p, ctx, lazy=False):
    if isinstance(p, str):
        return re.escape(p)
    if isinstance(p, Symbol):
        if p.name in _CLASSES:
            return _CLASSES[p.name]
        raise NotImplementedError(f"string pattern: {p.name} is not a string pattern object")
    if not isinstance(p, Expr) or not isinstance(p.head, str):
        raise NotImplementedError(f"string pattern: unsupported object {p!r}")
    h, a = p.head, p.args
    if h == "StringExpression":
        return "".join(_compile(x, ctx, lazy) for x in a)
    if h in ("List", "Alternatives"):
        return "(?:" + "|".join(_compile(x, ctx, lazy) for x in a) + ")"
    if h in ("Blank", "BlankSequence", "BlankNullSequence"):
        if a:
            raise NotImplementedError(f"string pattern: {h} with a head is not supported in strings")
        return _blank(h, lazy)
    if h == "Pattern" and len(a) == 2 and isinstance(a[0], Symbol):
        g, new = ctx.group(a[0].name)
        if not new:
            return f"(?P={g})"
        return f"(?P<{g}>" + _compile(a[1], ctx, lazy) + ")"
    if h == "Shortest" and len(a) == 1:
        return _compile(a[0], ctx, True)
    if h == "Longest" and len(a) == 1:
        return _compile(a[0], ctx, False)
    if h in ("Repeated", "RepeatedNull") and 1 <= len(a) <= 2:
        inner = "(?:" + _compile(a[0], ctx, lazy) + ")"
        lo = 1 if h == "Repeated" else 0
        if len(a) == 1:
            q = "+" if lo else "*"
        else:
            spec = a[1]
            if is_list(spec) and len(spec.args) == 1:
                n = int_arg(spec.args[0], "Repeated", "count")
                q = f"{{{n}}}"
            elif is_list(spec) and len(spec.args) == 2:
                q = f"{{{int_arg(spec.args[0], 'Repeated', 'min')},{int_arg(spec.args[1], 'Repeated', 'max')}}}"
            else:
                q = f"{{{lo},{int_arg(spec, 'Repeated', 'max')}}}"
        return inner + q + ("?" if lazy else "")
    if h == "Except" and len(a) in (1, 2):
        neg = "(?!" + _compile(a[0], ctx, lazy) + ")"
        return "(?:" + neg + ("." if len(a) == 1 else _compile(a[1], ctx, lazy)) + ")"
    if h == "PatternTest" and len(a) == 2 and isinstance(a[0], Expr) and a[0].head in (
            "Blank", "BlankSequence", "BlankNullSequence") and not a[0].args:
        t = a[1]
        if a[0].head != "Blank" and not (isinstance(t, Symbol) and t.name in _CHAR_TESTS):
            raise NotImplementedError("string pattern: __?test is only supported for LetterQ, DigitQ, UpperCaseQ "
                                      "and LowerCaseQ")
        cls = _char_class(t, ctx)
        q = {"Blank": "", "BlankSequence": "+", "BlankNullSequence": "*"}[a[0].head]
        return cls + q + ("?" if lazy and q else "")
    if h == "RegularExpression" and len(a) == 1 and isinstance(a[0], str):
        ctx.regex_groups = True
        return "(?:" + a[0] + ")"
    if h == "Characters" and len(a) == 1:
        return "[" + "".join(re.escape(c) for c in _s(a[0], "Characters")) + "]"
    if h == "Condition":
        raise NotImplementedError("string pattern: conditions (p /; test) are not supported")
    raise NotImplementedError(f"string pattern: {h} is not supported")


def _rx(p, subject, ignore_case=False, metachars=False):
    """(compiled regex, context) for a Wolfram string pattern."""
    ctx = _Ctx(subject)
    if metachars and isinstance(p, str):
        src = "".join(".*" if c == "*" else "[^A-Z]*" if c == "@" else re.escape(c) for c in p)
    else:
        src = _compile(p, ctx)
    flags = re.DOTALL | (re.IGNORECASE if ignore_case else 0)
    try:
        return re.compile(src, flags), ctx
    except re.error as e:
        raise ValueError(f"invalid string pattern: {e}")


def _ignore_case(opts):
    v = opts.get("IgnoreCase")
    return v is True


def _is_rule(r):
    return isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2


def _subst(expr, binds):
    if isinstance(expr, Symbol) and expr.name in binds:
        return binds[expr.name]
    if isinstance(expr, Expr):
        h = _subst(expr.head, binds) if not isinstance(expr.head, str) else expr.head
        return Expr(h, *(_subst(x, binds) for x in expr.args))
    return expr


def _rhs(rule, m, ctx):
    rhs = rule.args[1]
    if isinstance(rhs, str) and ctx.regex_groups and "$" in rhs:
        def grp(mm):
            v = m.group(int(mm.group(1)))
            return v or ""
        return re.sub(r"\$(\d+)", grp, rhs)
    binds = {n: m.group(g) for n, g in ctx.names.items() if m.group(g) is not None}
    if not binds:
        return ev(rhs) if rule.head == "RuleDelayed" else rhs
    return ev(_subst(rhs, binds))


def _matches(rx, st, overlaps=False, limit=None):
    """Non-empty matches, left to right (Overlaps -> False/True)."""
    out = []
    if overlaps:
        for i in range(len(st)):
            m = rx.match(st, i)
            if m and m.end() > m.start():
                out.append(m)
                if limit is not None and len(out) >= limit:
                    break
        return out
    for m in rx.finditer(st):
        if m.end() > m.start():
            out.append(m)
            if limit is not None and len(out) >= limit:
                break
    return out


def _overlaps(opts, default):
    v = opts.get("Overlaps", default)
    if v is True or v is False:
        return v
    raise NotImplementedError("Overlaps -> All is not supported")


def _rules_and_patterns(p):
    """List of (pattern, rule or None) for a pattern, rule, or list of them."""
    items = list(p.args) if is_list(p) and any(_is_rule(x) for x in p.args) else [p]
    out = []
    for x in items:
        out.append((x.args[0], x) if _is_rule(x) else (x, None))
    return out


def _strmap(s, f):
    """Thread a string function over a list of strings in its first argument."""
    if is_list(s):
        return L(*(f(x) for x in s.args))
    return f(s)


_OPTS = {"IgnoreCase", "Overlaps"}


# ---------------------------------------------------------------- matching functions
@fn("StringMatchQ", area=AREA, status="added",
    notes="whole-string match against a string pattern; literal patterns use the metacharacters * (any "
          "characters) and @ (characters other than upper-case letters); IgnoreCase; threads over lists"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _smatchq(s: Expr, p: Expr, *opts: Option) -> Boolean | ListOf[Boolean]:
    _, o = split_options(opts, _OPTS, "StringMatchQ")

    def one(x):
        if not isinstance(x, str):
            return False
        rx, _ = _rx(p, x, _ignore_case(o), metachars=True)
        return rx.fullmatch(x) is not None
    return _strmap(s, one)


def _search(fname, s, p, opts, anchor):
    _, o = split_options(opts, _OPTS, fname)

    def one(x):
        st = _s(x, fname)
        rx, _ = _rx(p, st, _ignore_case(o))
        if anchor == "start":
            return rx.match(st) is not None
        if anchor == "end":
            return any(rx.fullmatch(st, i) for i in range(len(st) + 1))
        return rx.search(st) is not None
    return _strmap(s, one)


@fn("StringContainsQ", area=AREA, status="fixed",
    notes="string patterns (legacy: literal substrings only); IgnoreCase; threads over lists"
          ". Check: reference (differential cases in tests/library).")
def _scontains(s: Expr, p: Expr, *opts: Option) -> Boolean | ListOf[Boolean]:
    return _search("StringContainsQ", s, p, opts, None)


@fn("StringFreeQ", area=AREA, status="added", notes="not StringContainsQ; string patterns; IgnoreCase"
                                                    ". Check: reference (differential cases in tests/library).")
def _sfree(s: Expr, p: Expr, *opts: Option) -> Boolean | ListOf[Boolean]:
    r = _search("StringFreeQ", s, p, opts, None)
    return L(*(not v for v in r.args)) if is_list(r) else (not r)


@fn("StringStartsQ", area=AREA, status="fixed", notes="string patterns (legacy: literal prefix only); IgnoreCase"
                                                      ". Check: reference (differential cases in tests/library).")
def _sstarts(s: Expr, p: Expr, *opts: Option) -> Boolean | ListOf[Boolean]:
    return _search("StringStartsQ", s, p, opts, "start")


@fn("StringEndsQ", area=AREA, status="fixed", notes="string patterns (legacy: literal suffix only); IgnoreCase"
                                                    ". Check: reference (differential cases in tests/library).")
def _sends(s: Expr, p: Expr, *opts: Option) -> Boolean | ListOf[Boolean]:
    return _search("StringEndsQ", s, p, opts, "end")


def _scan(fname, s, p, n, opts, default_overlaps, emit):
    pos, o = split_options(opts, _OPTS, fname)
    if pos:
        raise ValueError(f"{fname}: unexpected arguments")
    lim = None if n is None else nonneg_int(n, fname, "count")

    def one(x):
        st = _s(x, fname)
        alts = _rules_and_patterns(p)
        if len(alts) == 1:
            rx, ctx = _rx(alts[0][0], st, _ignore_case(o))
            ms = [(m, alts[0][1], ctx) for m in _matches(rx, st, _overlaps(o, default_overlaps), lim)]
        else:
            # several rules: one alternation, then find which rule matched
            compiled = [(_rx(pt, st, _ignore_case(o)), r) for pt, r in alts]
            ms, i = [], 0
            ov = _overlaps(o, default_overlaps)
            while i < len(st) and (lim is None or len(ms) < lim):
                hit = None
                for (rx, ctx), r in compiled:
                    m = rx.match(st, i)
                    if m and m.end() > m.start():
                        hit = (m, r, ctx)
                        break
                if hit:
                    ms.append(hit)
                    i = i + 1 if ov else hit[0].end()
                else:
                    i += 1
        return emit(st, ms)
    return _strmap(s, one)


@fn("StringCases", area=AREA, status="added",
    notes="substrings matching a string pattern, or rule right-hand sides with named parts substituted; "
          "StringCases[s, p, n]; Overlaps -> False by default; IgnoreCase"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _scases(s: Expr, p: Expr, *rest: Expr) -> Expr:
    n, opts = _count_arg(rest)

    def emit(st, ms):
        return L(*(m.group(0) if r is None else _rhs(r, m, ctx) for m, r, ctx in ms))
    return _scan("StringCases", s, p, n, opts, False, emit)


def _count_arg(rest):
    rest = list(rest)
    n = None
    if rest and not _is_rule(rest[0]):
        n = rest.pop(0)
    return n, rest


@fn("StringPosition", area=AREA, status="added",
    notes="{{start, end}, ...} 1-based; Overlaps -> True by default (one match per start position), "
          "StringPosition[s, p, n]; IgnoreCase"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _spos(s: Expr, p: Expr, *rest: Expr) -> Expr:
    n, opts = _count_arg(rest)

    def emit(st, ms):
        return L(*(L(m.start() + 1, m.end()) for m, _r, _c in ms))
    return _scan("StringPosition", s, p, n, opts, True, emit)


@fn("StringCount", area=AREA, status="fixed",
    notes="non-overlapping matches of a string pattern (legacy: literal substrings only); Overlaps -> True "
          "counts one match per start position; IgnoreCase; the empty literal raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _scount(s: Expr, p: Expr, *opts: Option) -> Expr:
    if p == "":
        raise ValueError("StringCount: substring must be non-empty")

    def emit(st, ms):
        return len(ms)
    return _scan("StringCount", s, p, None, opts, False, emit)


def _replace(fname, st, rules, lim, o):
    alts = _rules_and_patterns(rules)
    for pt, r in alts:
        if r is None:
            raise ValueError(f"{fname}: expected a rule pattern -> replacement")
        if pt == "":
            raise ValueError("cannot replace the empty string")
    compiled = [(_rx(pt, st, _ignore_case(o)), r) for pt, r in alts]
    pieces, i, last, n = [], 0, 0, 0
    while i <= len(st) and (lim is None or n < lim):
        hit = None
        for (rx, ctx), r in compiled:
            m = rx.match(st, i)
            if m and m.end() > m.start():
                hit = (m, r, ctx)
                break
        if hit is None:
            i += 1
            continue
        m, r, ctx = hit
        if m.start() > last:
            pieces.append(st[last:m.start()])
        pieces.append(_rhs(r, m, ctx))
        n += 1
        i = last = m.end()
    if last < len(st):
        pieces.append(st[last:])
    if all(isinstance(x, str) for x in pieces):
        return "".join(pieces)
    out = []
    for x in pieces:                      # merge adjacent strings, keep non-string replacements
        if isinstance(x, str) and out and isinstance(out[-1], str):
            out[-1] += x
        else:
            out.append(x)
    return Expr("StringExpression", *out)


@fn("StringReplace", area=AREA, status="fixed",
    notes="StringReplace[s, p -> r], [s, {rules}], [s, rules, n] with string patterns; named pattern parts "
          "substitute into the right-hand side; RegularExpression rules accept \"$1\" group references; "
          "non-string replacements give a StringExpression; the legacy form StringReplace[s, old, new] raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sreplace(s: Expr, r: Expr, *rest: Expr) -> Expr:
    if isinstance(r, str) and rest and isinstance(rest[0], str):
        raise ValueError("StringReplace[s, old, new] is not Wolfram syntax; use StringReplace[s, old -> new]")
    n, opts = _count_arg(rest)
    _, o = split_options(opts, _OPTS, "StringReplace")
    lim = None if n is None else nonneg_int(n, "StringReplace", "count")
    return _strmap(s, lambda x: _replace("StringReplace", _s(x, "StringReplace"), r, lim, o))


@fn("StringDelete", area=AREA, status="fixed",
    notes="deletes every match of a string pattern (or list of patterns); IgnoreCase; the legacy slice form "
          "StringDelete[s, i, j] raises (use StringDrop[s, {i, j}])"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sdelete(s: Expr, p: Expr, *rest: Expr) -> Expr:
    if rest and not all(is_rule(x) for x in rest):
        raise ValueError("StringDelete[s, i, j] is not Wolfram syntax; use StringDrop[s, {i, j}]")
    _, o = split_options(rest, _OPTS, "StringDelete")
    pats = list(p.args) if is_list(p) else [p]
    rules = L(*(Expr("Rule", x, "") for x in pats))
    return _strmap(s, lambda x: _replace("StringDelete", _s(x, "StringDelete"), rules, None, o))


@fn("StringTrim", area=AREA, status="fixed",
    notes="StringTrim[s] removes leading and trailing whitespace; StringTrim[s, p] removes a leading and a "
          "trailing match of the string pattern p"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _strim(s: Expr, p: Expr = None, *opts: Option) -> Expr:
    if is_rule(p) and isinstance(p.args[0], Symbol) and p.args[0].name == "IgnoreCase":
        opts, p = (p,) + opts, None
    _, o = split_options(opts, {"IgnoreCase"}, "StringTrim")

    def one(x):
        st = _s(x, "StringTrim")
        if p is None:
            return re.sub(r"\A\s+|\s+\Z", "", st)
        rx, _ = _rx(p, st, _ignore_case(o))
        m = rx.match(st)
        if m:
            st = st[m.end():]
        for i in range(len(st)):
            if rx.fullmatch(st, i):
                return st[:i]
        return st
    return _strmap(s, one)


@fn("StringSplit", area=AREA, status="added",
    notes="splits at whitespace or at matches of a string pattern; empty pieces at the ends are dropped "
          "(kept with StringSplit[s, p, All]), interior ones kept; p -> r inserts r at each split; "
          "StringSplit[s, p, n] gives at most n pieces; IgnoreCase; threads over lists"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _ssplit(s: Expr, p: Expr = None, *rest: Expr) -> Expr:
    rest = list(rest)
    n = rest.pop(0) if rest and not is_rule(rest[0]) else None
    _, o = split_options(rest, _OPTS, "StringSplit")
    keep_all = is_sym(n, "All")
    lim = None if n is None or keep_all else pos_int(n, "StringSplit", "count")

    def one(x):
        st = _s(x, "StringSplit")
        pat = Symbol("Whitespace") if p is None else p
        alts = _rules_and_patterns(pat)
        compiled = [(_rx(pt, st, _ignore_case(o)), r) for pt, r in alts]
        pieces, i, last = [], 0, 0
        while i < len(st) and (lim is None or sum(1 for q in pieces if q[0] == "s") < lim - 1):
            hit = None
            for (rx, ctx), r in compiled:
                m = rx.match(st, i)
                if m and m.end() > m.start():
                    hit = (m, r, ctx)
                    break
            if hit is None:
                i += 1
                continue
            m, r, ctx = hit
            pieces.append(("s", st[last:m.start()]))
            if r is not None:
                pieces.append(("r", _rhs(r, m, ctx)))
            i = last = m.end()
        pieces.append(("s", st[last:]))
        if not keep_all:
            if pieces and pieces[0] == ("s", ""):
                pieces = pieces[1:]
            if pieces and pieces[-1] == ("s", ""):
                pieces = pieces[:-1]
        return L(*(v for _k, v in pieces))
    return _strmap(s, one)


# ---------------------------------------------------------------- structural string functions
def _tostr(x):
    return x if isinstance(x, str) else ev(Expr("ToString", x))


@fn("StringRiffle", area=AREA, status="added",
    notes="StringRiffle[list] joins with spaces (nested lists: spaces at the lowest level, one more newline per "
          "level above), StringRiffle[list, sep], "
          "StringRiffle[list, {left, sep, right}], StringRiffle[list, sep1, sep2, ...] per level; non-strings "
          "are converted with ToString"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sriffle(lst: ListOf[Expr], *seps: Expr) -> Expr:
    if not is_list(lst):
        raise ValueError("StringRiffle: the first argument must be a list")
    depth = 1
    probe = lst
    while is_list(probe) and probe.args and all(is_list(e) for e in probe.args):
        depth += 1
        probe = probe.args[0]
    if not seps:
        seps = tuple("\n" * (depth - 1 - i) for i in range(depth - 1)) + (" ",)

    def sep_parts(sp):
        if is_list(sp):
            if len(sp.args) != 3 or not all(isinstance(v, str) for v in sp.args):
                raise ValueError("StringRiffle: {left, sep, right} must be three strings")
            return sp.args
        return "", _s(sp, "StringRiffle", "separator"), ""

    def go(x, k):
        if not is_list(x):
            return _tostr(x)
        left, sep, right = sep_parts(seps[min(k, len(seps) - 1)])
        return left + sep.join(go(e, k + 1) for e in x.args) + right
    return go(lst, 0)


@fn("StringPartition", area=AREA, status="added",
    notes="substrings of length n (incomplete final piece dropped), offset d, or UpTo[n] keeping the remainder"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _spartition(s: String, n: Expr, d: Expr = None) -> ListOf[Expr]:
    st = _s(s, "StringPartition")
    if isinstance(n, Expr) and n.head == "UpTo" and len(n.args) == 1:
        k = pos_int(n.args[0], "StringPartition", "length")
        step = k if d is None else pos_int(d, "StringPartition", "offset")
        out = []
        for i in range(0, len(st), step):
            out.append(st[i:i + k])
            if i + k >= len(st):
                break
        return L(*out)
    k = pos_int(n, "StringPartition", "length")
    step = k if d is None else pos_int(d, "StringPartition", "offset")
    return L(*(st[i:i + k] for i in range(0, len(st) - k + 1, step)))


def _rotate(st, n):
    if not st:
        return st
    n %= len(st)
    return st[n:] + st[:n]


@fn("StringRotateLeft", area=AREA, status="added", notes="cyclic rotation by n characters (default 1)"
                                                         ". Check: exact (exact input gives exact results), "
                                                         "reference (differential cases in tests/library).")
def _srotl(s: Expr, n: Expr = 1) -> Expr:
    return _strmap(s, lambda x: _rotate(_s(x, "StringRotateLeft"), int_arg(n, "StringRotateLeft", "count")))


@fn("StringRotateRight", area=AREA, status="added", notes="cyclic rotation by n characters (default 1)"
                                                          ". Check: exact (exact input gives exact results), "
                                                          "reference (differential cases in tests/library).")
def _srotr(s: Expr, n: Expr = 1) -> Expr:
    return _strmap(s, lambda x: _rotate(_s(x, "StringRotateRight"), -int_arg(n, "StringRotateRight", "count")))


@fn("StringTakeDrop", area=AREA, status="added", notes="{StringTake[s, spec], StringDrop[s, spec]}"
                                                       ". Check: exact (exact input gives exact results), reference "
                                                       "(differential cases in tests/library).")
def _stakedrop(s: Expr, spec: Expr) -> ListOf[Expr]:
    return L(ev(Expr("StringTake", s, spec)), ev(Expr("StringDrop", s, spec)))


@fn("StringReplacePart", area=AREA, status="added",
    notes="StringReplacePart[s, new, {m, n}], a list of ranges with one new string, or a list of new strings "
          "matched to the ranges; negative positions count from the end"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sreplacepart(s: String, new: Expr, spec: Expr) -> String:
    st = _s(s, "StringReplacePart")
    if is_list(spec) and spec.args and all(is_list(x) for x in spec.args):
        ranges = list(spec.args)
    else:
        ranges = [spec]
    news = list(new.args) if is_list(new) else [new] * len(ranges)
    if len(news) != len(ranges):
        raise ValueError("StringReplacePart: the numbers of strings and ranges differ")
    spans = []
    for r, nw in zip(ranges, news):
        if not (is_list(r) and len(r.args) == 2):
            raise ValueError("StringReplacePart: ranges are {m, n}")
        m, k = (int_arg(v, "StringReplacePart", "position") for v in r.args)
        m = m if m > 0 else len(st) + m + 1
        k = k if k > 0 else len(st) + k + 1
        if not (1 <= m <= len(st) + 1 and m - 1 <= k <= len(st)):
            raise ValueError("StringReplacePart: range out of bounds")
        spans.append((m - 1, k, _s(nw, "StringReplacePart", "new string")))
    out, last = [], 0
    for a, b, nw in sorted(spans):
        if a < last:
            raise ValueError("StringReplacePart: overlapping ranges")
        out.append(st[last:a])
        out.append(nw)
        last = b
    out.append(st[last:])
    return "".join(out)


# ---------------------------------------------------------------- case and character predicates
@fn("ToUpperCase", area=AREA, status="added", notes="upper-cases every letter; threads over lists"
                                                    ". Check: exact (exact input gives exact results), reference "
                                                    "(differential cases in tests/library).")
def _toupper(s: Expr) -> Expr:
    return _strmap(s, lambda x: _s(x, "ToUpperCase").upper())


@fn("ToLowerCase", area=AREA, status="added", notes="lower-cases every letter; threads over lists"
                                                    ". Check: exact (exact input gives exact results), reference "
                                                    "(differential cases in tests/library).")
def _tolower(s: Expr) -> Expr:
    return _strmap(s, lambda x: _s(x, "ToLowerCase").lower())


@fn("Capitalize", area=AREA, status="added",
    notes="upper-cases the first character and leaves the rest; Capitalize[s, \"AllWords\"] does that for every "
          "word, \"LongWords\" for words longer than three characters, \"FirstWord\" = default (\"TitleCase\" with "
          "its small-word rules is not supported)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _capitalize(s: Expr, mode: Expr = None) -> Expr:
    def one(x):
        st = _s(x, "Capitalize")
        if mode is None or mode == "FirstWord":
            return st[:1].upper() + st[1:]
        if mode in ("AllWords", "LongWords"):
            short = 3 if mode == "LongWords" else 0
            return re.sub(r"(?<![^\W_'])[^\W_][\w']*",
                          lambda m: m.group(0) if len(m.group(0)) <= short else m.group(0)[:1].upper() + m.group(0)[1:],
                          st)
        raise NotImplementedError(f"Capitalize: mode {mode!r} is not supported")
    return _strmap(s, one)


@fn("Decapitalize", area=AREA, status="added", notes="lower-cases the first character only"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _decapitalize(s: Expr) -> Expr:
    return _strmap(s, lambda x: _s(x, "Decapitalize")[:1].lower() + _s(x, "Decapitalize")[1:])


def _all_chars(name, pred):
    @fn(name, area=AREA, status="added",
        notes="True when every character of the string satisfies the test (so True for \"\"); non-strings False")
    def impl(s: Expr) -> Boolean:
        if not isinstance(s, str):
            return False
        return all(pred(c) for c in s)
    return impl


_all_chars("UpperCaseQ", lambda c: c.isupper())
_all_chars("LowerCaseQ", lambda c: c.islower())
_all_chars("LetterQ", lambda c: c.isalpha())
_all_chars("DigitQ", lambda c: "0" <= c <= "9")
_all_chars("PrintableASCIIQ", lambda c: " " <= c <= "~")


# ---------------------------------------------------------------- conversion
@fn("ToExpression", area=AREA, status="added",
    notes="parses and evaluates InputForm text; ToExpression[s, InputForm, h] wraps the unevaluated parse in h "
          "(e.g. Hold); invalid syntax gives $Failed; non-strings are returned evaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _toexpr(s: Expr, form: Expr = None, h: Expr = None) -> Expr:
    if not isinstance(s, str):
        return s
    if form is not None and not is_sym(form, "InputForm") and not is_sym(form, "StandardForm"):
        raise NotImplementedError("ToExpression: only InputForm text is supported")
    from ..parser import parse
    try:
        e = parse(s)
    except Exception:
        return _FAILED
    if h is not None:
        return ev(Expr(h, e)) if not is_sym(h, "Hold") else Expr("Hold", e)
    return ev(e)


_ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
          (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def _roman(n):
    if not 1 <= n <= 4999:
        raise ValueError("Roman numerals are supported for 1 to 4999")
    out = []
    for v, sym in _ROMAN:
        while n >= v:
            out.append(sym)
            n -= v
    return "".join(out)


@fn("IntegerString", area=AREA, status="added",
    notes="digits of |n| (the sign is dropped, as Wolfram) in base 2-36 with lower-case letters, padded with "
          "zeros or truncated to the last len digits; IntegerString[n, \"Roman\"]; threads over lists"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _intstring(n: Expr, b: Expr = 10, length: Expr = None) -> Expr:
    if is_list(n):
        return L(*(_intstring(x, b, length) for x in n.args))
    k = int_arg(n, "IntegerString", "integer")
    if b == "Roman":
        return _roman(abs(k))
    base = int_arg(b, "IntegerString", "base")
    if not 2 <= base <= 36:
        raise ValueError("IntegerString: base must be between 2 and 36")
    k = abs(k)
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while True:
        out = digits[k % base] + out
        k //= base
        if k == 0:
            break
    if length is not None:
        w = nonneg_int(length, "IntegerString", "length")
        out = out.rjust(w, "0")[-w:] if w else ""
    return out


@fn("RomanNumeral", area=AREA, status="added",
    notes="Roman numeral string of n for 1-4999 (sign dropped); RomanNumeral[0] gives \"N\"; threads over "
          "lists; the overbar forms for n >= 5000 are not supported"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _romannum(n: Expr) -> Expr:
    if is_list(n):
        return L(*(_romannum(x) for x in n.args))
    k = abs(int_arg(n, "RomanNumeral", "integer"))
    return "N" if k == 0 else _roman(k)


@fn("FromRomanNumeral", area=AREA, status="added", notes="integer value of a Roman numeral string (\"N\" -> 0)"
                                                         ". Check: exact (exact input gives exact results), "
                                                         "reference (differential cases in tests/library).")
def _fromroman(s: String) -> Expr:
    st = _s(s, "FromRomanNumeral").upper()
    if st == "N":
        return 0
    i, total = 0, 0
    for v, sym in _ROMAN:
        while st.startswith(sym, i):
            total += v
            i += len(sym)
    if i != len(st) or not st or _roman(total) != st:
        raise ValueError(f"FromRomanNumeral: {s!r} is not a valid Roman numeral")
    return total


@fn("TextWords", area=AREA, status="added",
    notes="words of the text: runs of letters/digits, keeping internal apostrophes and hyphens (rule-based; "
          "Wolfram's tokenizer may differ on unusual punctuation)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _textwords(s: String, n: Expr = None) -> ListOf[Expr]:
    words = re.findall(r"[^\W_]+(?:['\u2019\-][^\W_]+)*", _s(s, "TextWords"))
    if n is not None:
        words = words[:nonneg_int(n, "TextWords", "count")]
    return L(*words)


@fn("Alphabet", area=AREA, status="added",
    notes="Alphabet[] and Alphabet[\"English\"] give the 26 lower-case letters; other languages are not "
          "supported (curated data)")
def _alphabet(lang: Expr = "English") -> ListOf[Expr]:
    if lang not in ("English", Symbol("Automatic")) and not is_sym(lang, "Automatic"):
        raise NotImplementedError(f"Alphabet: only English is supported, not {lang!r}")
    return L(*"abcdefghijklmnopqrstuvwxyz")


@fn("LetterNumber", area=AREA, status="added",
    notes="position of a letter in the English alphabet (either case), 0 for other characters; a string of "
          "several characters gives a list; threads over lists"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _letternumber(s: Expr) -> Expr:
    def one(x):
        st = _s(x, "LetterNumber")
        vals = [ord(c.lower()) - 96 if c.isascii() and c.isalpha() else 0 for c in st]
        return vals[0] if len(vals) == 1 else L(*vals)
    return _strmap(s, one)


@fn("FromLetterNumber", area=AREA, status="added",
    notes="the n-th lower-case English letter for 1 <= n <= 26 (negative n counts from z, 0 gives \" \"); other "
          "values raise; "
          "threads over lists"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fromletternumber(n: Expr) -> Expr:
    if is_list(n):
        return L(*(_fromletternumber(x) for x in n.args))
    k = int_arg(n, "FromLetterNumber", "position")
    if k == 0:
        return " "
    if 1 <= k <= 26:
        return chr(96 + k)
    if -26 <= k <= -1:
        return chr(123 + k)
    raise ValueError(f"FromLetterNumber: {k} is not a letter position")


@fn("RemoveDiacritics", area=AREA, status="added",
    notes="drops combining marks after Unicode NFKD decomposition (ligatures such as \u00e6 are kept)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _removediacritics(s: Expr) -> Expr:
    def one(x):
        st = unicodedata.normalize("NFKD", _s(x, "RemoveDiacritics"))
        return unicodedata.normalize("NFC", "".join(c for c in st if not unicodedata.combining(c)))
    return _strmap(s, one)


@fn("CharacterRange", area=AREA, status="added",
    notes="characters from c1 to c2 by code point (strings or integer codes); empty when c2 < c1"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _charrange(a: Expr, b: Expr) -> ListOf[Expr]:
    def code(x):
        if isinstance(x, str):
            if len(x) != 1:
                raise ValueError("CharacterRange: arguments must be single characters")
            return ord(x)
        return nonneg_int(x, "CharacterRange", "code")
    return L(*(chr(c) for c in range(code(a), code(b) + 1)))


@fn("SymbolName", area=AREA, status="added", notes="name of a symbol as a string (no context prefix)"
                                                   ". Check: exact (exact input gives exact results), reference "
                                                   "(differential cases in tests/library).")
def _symbolname(x: Expr) -> Expr:
    if isinstance(x, Symbol):
        return x.name.split("`")[-1]
    if isinstance(x, bool) or x is None:
        return {True: "True", False: "False", None: "Null"}[x]
    raise Unevaluated()


@fn("Symbol", area=AREA, status="added", notes="Symbol[\"name\"] gives the symbol (evaluated); invalid names raise"
                                               ". Check: exact (exact input gives exact results), reference "
                                               "(differential cases in tests/library).")
def _symbol(name: String) -> Expr:
    st = _s(name, "Symbol", "name")
    if not re.fullmatch(r"\$?[A-Za-z][A-Za-z0-9$]*|\$[A-Za-z0-9$]+", st):
        raise ValueError(f"Symbol: {st!r} is not a valid symbol name")
    return ev(Symbol(st))


# ---------------------------------------------------------------- distances and sequences
def _seq(x, fname):
    if isinstance(x, str):
        return list(x), True
    if is_list(x):
        return list(x.args), False
    raise ValueError(f"{fname}: arguments must be strings or lists")


def _eq(a, b, ignore_case):
    if ignore_case and isinstance(a, str) and isinstance(b, str):
        return a.lower() == b.lower()
    return KL.K.tag(a) == KL.K.tag(b) if not (isinstance(a, str) and isinstance(b, str)) else a == b


@fn("EditDistance", area=AREA, status="added",
    notes="Levenshtein distance (insertions, deletions, substitutions) between strings or lists; IgnoreCase"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _editdistance(a: Expr, b: Expr, *opts: Option) -> Expr:
    _, o = split_options(opts, {"IgnoreCase"}, "EditDistance")
    ic = _ignore_case(o)
    x, _ = _seq(a, "EditDistance")
    y, _ = _seq(b, "EditDistance")
    prev = list(range(len(y) + 1))
    for i, ca in enumerate(x, 1):
        cur = [i]
        for j, cb in enumerate(y, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if _eq(ca, cb, ic) else 1)))
        prev = cur
    return prev[-1]


@fn("HammingDistance", area=AREA, status="added",
    notes="number of differing positions between equal-length strings or lists (unequal lengths raise); "
          "IgnoreCase"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _hamming(a: Expr, b: Expr, *opts: Option) -> Expr:
    _, o = split_options(opts, {"IgnoreCase"}, "HammingDistance")
    x, _ = _seq(a, "HammingDistance")
    y, _ = _seq(b, "HammingDistance")
    if len(x) != len(y):
        raise ValueError("HammingDistance: the arguments must have the same length")
    return sum(1 for p, q in zip(x, y) if not _eq(p, q, _ignore_case(o)))


def _wrap(items, is_str):
    return "".join(items) if is_str else L(*items)


@fn("LongestCommonSubsequence", area=AREA, status="added",
    notes="longest contiguous common run of two strings or lists (Wolfram's meaning; the non-contiguous one is "
          "LongestCommonSequence); ties go to the earliest position in the first argument"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _lcsubstr(a: Expr, b: Expr) -> Expr:
    x, sx = _seq(a, "LongestCommonSubsequence")
    y, _ = _seq(b, "LongestCommonSubsequence")
    best, end = 0, 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if _eq(x[i - 1], y[j - 1], False):
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best, end = cur[j], i
        prev = cur
    return _wrap(x[end - best:end], sx)


@fn("LongestCommonSequence", area=AREA, status="added",
    notes="longest common (not necessarily contiguous) subsequence of two strings or lists by dynamic "
          "programming; among equal-length answers the one found by the standard back-trace is returned"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _lcseq(a: Expr, b: Expr) -> Expr:
    x, sx = _seq(a, "LongestCommonSequence")
    y, _ = _seq(b, "LongestCommonSequence")
    n, m = len(x), len(y)
    t = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            t[i][j] = t[i + 1][j + 1] + 1 if _eq(x[i], y[j], False) else max(t[i + 1][j], t[i][j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if _eq(x[i], y[j], False):
            out.append(x[i])
            i += 1
            j += 1
        elif t[i + 1][j] >= t[i][j + 1]:
            i += 1
        else:
            j += 1
    return _wrap(out, sx)


@fn("PalindromeQ", area=AREA, status="added",
    notes="True for a string, list or non-negative integer (decimal digits) equal to its reverse; other "
          "expressions False"
          ". Check: reference (differential cases in tests/library).")
def _palindromeq(x: Expr) -> Boolean:
    if isinstance(x, str):
        return x == x[::-1]
    if is_list(x):
        tags = [KL.K.tag(e) for e in x.args]
        return tags == tags[::-1]
    if isinstance(x, int) and not isinstance(x, bool):
        d = str(abs(x))
        return d == d[::-1]
    return False



# String pattern objects and options interpreted by the functions above (not functions themselves).
for _n in ("StringExpression", "Shortest", "Longest", "RegularExpression", "DigitCharacter", "LetterCharacter",
           "WordCharacter", "WhitespaceCharacter", "PunctuationCharacter", "HexadecimalCharacter", "StartOfString",
           "EndOfString", "StartOfLine", "EndOfLine", "WordBoundary", "Whitespace", "NumberString", "IgnoreCase",
           "Overlaps"):
    record(_n, AREA, "added", "string pattern object / option interpreted by the batch-1 string functions")


@fn("StringExtract", area=AREA, status="added",
    notes="StringExtract[s, n] gives the n-th whitespace-separated piece (spans, lists and negative n allowed); "
          "StringExtract[s, sep -> n] splits at sep"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _stringextract(s: String, spec: Expr) -> Expr:
    st = _s(s, "StringExtract")
    if _is_rule(spec):
        pieces = ev(Expr("StringSplit", st, spec.args[0]))
        spec = spec.args[1]
    else:
        pieces = ev(Expr("StringSplit", st))
    return ev(Expr("Part", pieces, spec))


@fn("StringReplaceList", area=AREA, status="added",
    notes="every string obtained by replacing a single match of the rules (one result per match position and "
          "rule, left to right); StringReplaceList[s, rules, n] keeps at most n"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _stringreplacelist(s: Expr, r: Expr, n: Expr = None) -> Expr:
    def one(x):
        st = _s(x, "StringReplaceList")
        alts = _rules_and_patterns(r)
        compiled = [(_rx(pt, st), rr) for pt, rr in alts]
        if any(rr is None for _c, rr in compiled):
            raise ValueError("StringReplaceList: expected rules")
        out = []
        for i in range(len(st)):
            for (rx, ctx), rr in compiled:
                m = rx.match(st, i)
                if m and m.end() > m.start():
                    v = _rhs(rr, m, ctx)
                    out.append(st[:i] + v + st[m.end():] if isinstance(v, str)
                               else Expr("StringExpression", st[:i], v, st[m.end():]))
        if n is not None:
            out = out[:nonneg_int(n, "StringReplaceList", "count")]
        return L(*out)
    return _strmap(s, one)

# Check: sentences (CONVENTIONS §4) for heads registered in a loop or by a factory, whose notes are
# built in code. Each names only methods that pass in scripts/verify_library.py.
from .core import add_check as _add_check  # noqa: E402

_add_check("reference (differential cases in tests/library).",
          "DigitQ", "LetterQ", "LowerCaseQ", "PrintableASCIIQ", "UpperCaseQ")
