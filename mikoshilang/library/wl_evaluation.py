"""Coverage batch W01: Wolfram Language evaluation control, attributes, definitions, contexts and packages.

What is here (each name is registered with status ``added`` and differential tests in
tests/library/test_wolfram_w01.py):

* evaluation control: ``Unevaluated``, ``HoldAllComplete``/``HoldCompleteForm``, ``Defer``, ``Inactive``/
  ``Inactivate``/``Activate``/``IgnoringInactive``, ``Until``, ``Goto``/``Label``, ``Update``,
  ``$RecursionLimit``, ``$IterationLimit``, ``Stack``/``StackBegin``/``StackInhibit``/``StackComplete``,
  ``TracePrint``/``TraceScan`` and the ``Trace`` options ``TraceDepth``, ``TraceForward``, ``TraceBackward``,
  ``TraceAbove``, ``TraceOn``;
* attributes: ``Protect``/``Unprotect``, ``Locked``, ``ReadProtected``, ``Constant``, ``Temporary``, ``Stub``,
  ``NHoldAll``/``NHoldFirst``/``NHoldRest``, ``SubValuesHoldAll``, ``NonThreadable``;
* definitions: ``UpSet``/``UpSetDelayed``/``TagSet``/``TagSetDelayed``/``TagUnset``, ``OwnValues``,
  ``DownValues``, ``SubValues``, ``UpValues``, ``DefaultValues``, ``Default``, ``Definition``,
  ``FullDefinition``, ``Information``, ``ValueQ``, ``Remove``, ``Unique``, ``$ModuleNumber``, ``ApplyTo``,
  ``Once``, ``LocalSymbol``;
* options: ``Options``, ``SetOptions``, ``OptionsPattern``, ``OptionValue``, ``FilterRules``,
  ``ArgumentsOptions``, ``CheckArguments`` and the inert option values ``Automatic``, ``Full``, ``Manual``,
  ``Inherited``, ``Method``, ...;
* contexts and packages: ``$Context``, ``$ContextPath``, ``$ContextAliases``, ``Context``, ``Contexts``,
  ``Begin``/``End``, ``BeginPackage``/``EndPackage``, ``Needs``, ``Get``, ``DeclarePackage``, ``$Packages``,
  ``Names``, ``NameQ``, ``$NewSymbol``, ``ResolveContextAliases``;
* the main loop (``Session.evaluate_input``): ``In``, ``Out`` (``%``), ``InString``, ``$Line``,
  ``$HistoryLength``, ``$Pre``, ``$Post``, ``$PrePrint``, ``$PreRead``, ``$Epilog``;
* functional forms: ``OperatorApplied``, ``CurryApplied``, ``Threaded``, ``Distribute``, ``InverseFunction``,
  ``Indexed``, ``Prefix``/``Postfix`` (display wrappers).

Symbols are created when input is parsed, as in Wolfram: names in ``System``` and ``Global``` are kept as plain
names; symbols in any other context carry their full name (``Foo`bar``).  Within one parsed input the contexts in
force when parsing began are used (so ``Begin["a`"]; x; End[]`` on one line creates ``Global`x``, as in Wolfram);
``Get`` and ``Session.evaluate_input`` parse and evaluate one top-level expression at a time.

Where Wolfram's behaviour depends on the notebook front end, the compiler, the cloud or persistence locations,
the name is not provided (see ``SKIPPED`` at the end of this module).
"""
from __future__ import annotations

import itertools
import json
import os
import re

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
import importlib
E = importlib.import_module("mikoshilang.evaluate")
from .core import fn, record, Unevaluated, REGISTRY, STATUS, L

AREA = "evaluation control, definitions, contexts (W01)"
UNH = K.UNHANDLED
NULL = None


def ev(x):
    return E.evaluate(x, 1)


def st():
    return K.state()


def S_(name):
    return Symbol(name)


def is_sym(x, name=None):
    return isinstance(x, Symbol) and (name is None or x.name == name)


def is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def ext(kind):
    return st().ext.setdefault(kind, {})


def _added(name, notes):
    record(name, AREA, "added", notes)


def held(*names, attrs=None, notes=""):
    """Register a special form that receives its arguments unevaluated (kernel._HELD)."""
    def deco(f):
        for n in names:
            K._HELD[n] = f
            if attrs is not None:
                K.BUILTIN_ATTRIBUTES[n] = set(attrs) | {"Protected"}
            if notes:
                _added(n, notes)
        return f
    return deco


def set_attrs(name, *attrs):
    K.BUILTIN_ATTRIBUTES[name] = set(attrs) | {"Protected"}


def _need(a, lo, name, hi=None):
    hi = lo if hi is None else hi
    if not (lo <= len(a) <= (hi if hi != -1 else 10 ** 9)):
        exp = f"{lo}" if hi == lo else (f"{lo} or more" if hi == -1 else f"{lo}-{hi}")
        raise ValueError(f"{name}: called with {len(a)} argument(s); {exp} expected")


def _innermost_head(e):
    h = e.head if isinstance(e, Expr) else None
    while isinstance(h, Expr):
        h = h.head
    return h if isinstance(h, str) else None


def _tag_of(e):
    """Symbol that carries upvalues for e: e itself if a symbol, else its innermost head symbol."""
    if isinstance(e, Symbol):
        return e.name
    if isinstance(e, Expr):
        return _innermost_head(e)
    return None


def _sym_name(x, fname, allow_string=True):
    if isinstance(x, Symbol):
        return x.name
    if allow_string and isinstance(x, str):
        return resolve_name(x)
    raise ValueError(f"{fname}: {x!r} is not a symbol" + (" or a string" if allow_string else ""))


def _strip_holdpattern(x):
    while isinstance(x, Expr) and x.head == "HoldPattern" and len(x.args) == 1:
        x = x.args[0]
    return x


def _hp(x):
    return Expr("HoldPattern", x)


def _rd(lhs, rhs):
    return Expr("RuleDelayed", _hp(lhs), rhs)


# ─────────────────────────── system symbols ───────────────────────────

INERT = {}          # inert System symbols added here: name -> notes
_SYS_CACHE = [None, -1]


def system_names():
    """Every name MikoshiLang implements as a System` symbol."""
    from .. import library
    from ..library.wl_numerics import SYSTEM_CONSTANTS
    n = len(library.REGISTRY) + len(K._BUILTINS) + len(K._HELD) + len(library.STATUS)
    if _SYS_CACHE[0] is None or _SYS_CACHE[1] != n:
        names = set(library.REGISTRY) | set(K._BUILTINS) | set(K._HELD) | set(K.BUILTIN_ATTRIBUTES)
        names |= set(getattr(K, "STRUCTURAL_HEADS", ())) | set(SYSTEM_CONSTANTS) | set(E.SYSTEM_VARIABLES)
        names |= {h for h, (_a, s, _n) in library.STATUS.items() if s != "quarantined"}
        names |= set(INERT) | set(K.KNOWN_ATTRIBUTES) | {"Null", "True", "False", "$Assumptions", "General"}
        _SYS_CACHE[0], _SYS_CACHE[1] = frozenset(names), n
    return _SYS_CACHE[0]


def is_system(name):
    return "`" not in name and name in system_names()


# Variables a user may assign although they are System` symbols
_ASSIGNABLE = set()


def _protected_system(name):
    return is_system(name) and name not in _ASSIGNABLE and not name.startswith("$")


_orig_is_protected = K.is_protected


def is_protected(name):
    if _orig_is_protected(name):
        return True
    return _protected_system(name) and name not in st().ext.get("Unprotected", ())


K.is_protected = is_protected


# ─────────────────────────── contexts ───────────────────────────
#
# $Context, $ContextPath, $ContextAliases, $Packages, $Line, $HistoryLength, ... are kept as OwnValues (so Block
# localises them); without an OwnValue they have their default value.  The symbol table (which symbols exist in
# which context) is session-wide data in state().ext["\x00ctx"].

_CTX_RE = re.compile(r"^(?:[A-Za-z$][A-Za-z0-9$]*`)+$")
_NAME_RE = re.compile(r"^`?(?:[A-Za-z$][A-Za-z0-9$]*`)*[A-Za-z$][A-Za-z0-9$]*$")


def _ctxdata():
    d = st().ext.get("\x00ctx")
    if d is None:
        d = st().ext["\x00ctx"] = {"globals": set(), "symbols": set(), "begin": [], "package": [],
                                   "unique": {}, "removed": set()}
    return d


def _own(name, default):
    v = st().own.get(name, UNH)
    if v is UNH:
        return default
    return v


def cur_context():
    v = _own("$Context", "Global`")
    if not isinstance(v, str):
        v = ev(v)
    if not isinstance(v, str) or not _CTX_RE.match(v):
        raise ValueError(f"$Context: {v!r} is not a valid context name")
    return v


def context_path():
    v = _own("$ContextPath", None)
    if v is None:
        return ["System`", "Global`"]
    if not K.is_list(v):
        v = ev(v)
    if not K.is_list(v) or not all(isinstance(c, str) for c in v.args):
        raise ValueError("$ContextPath must be a list of context names")
    return list(v.args)


def aliases():
    v = _own("$ContextAliases", None)
    if v is None:
        return {}
    if not (isinstance(v, Expr) and v.head == "Association"):
        v = ev(v)
    out = {}
    for r in v.args:
        if is_rule(r) and isinstance(r.args[0], str) and isinstance(r.args[1], str):
            out[r.args[0]] = r.args[1]
    return out


def packages():
    v = _own("$Packages", None)
    if v is None:
        return ["System`", "Global`"]
    return [c for c in v.args if isinstance(c, str)]


def _set_own(name, value):
    st().own[name] = value


def _assoc_of(d):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in d.items()))


def _check_context(c, fname):
    if not isinstance(c, str) or not _CTX_RE.match(c):
        raise ValueError(f"{fname}: {c!r} is not a valid context name (a string such as \"Name`\")")
    return c


def _context_of_full(full):
    i = full.rfind("`")
    return full[:i + 1]


def full_name(name):
    """Full name of an internal symbol name (bare names are in System` or Global`)."""
    if "`" in name:
        return name
    return ("System`" if is_system(name) else "Global`") + name


def _short(full):
    c, s = _context_of_full(full), full[full.rfind("`") + 1:]
    return c, s


def _internal(full):
    """Internal name for a full symbol name: System` and Global` symbols are bare."""
    c, s = _short(full)
    if c in ("Global`", "System`"):
        return s
    return full


def _exists_in(ctx, short):
    d = _ctxdata()
    if ctx == "System`":
        return is_system(short)
    if ctx == "Global`":
        return short in d["globals"] or short in _defined_bare()
    return (ctx + short) in d["symbols"]


def _defined_bare():
    s = st()
    names = set(s.own) | set(s.down) | set(s.attrs)
    for l, _r in s.exact.values():
        k = K._def_key(l)
        names.add(k)
    for d in s.ext.values():
        names |= {k for k in d if isinstance(k, str)}
    return {n for n in names if "`" not in n}


def _expand_alias(full, use_aliases=True):
    if use_aliases:
        for a, c in aliases().items():
            if full.startswith(a):
                return c + full[len(a):]
    return full


def _new_symbol_hook(short, ctx):
    f = st().own.get("$NewSymbol")
    if f is None:
        return
    ev(K.call(f, short, ctx))


def resolve_name(name, create=True):
    """Internal symbol name for a name as typed (short, qualified or `relative), creating it if needed."""
    d = _ctxdata()
    if name.startswith("`"):
        full = cur_context() + name[1:]
    elif "`" in name:
        full = _expand_alias(name)
    else:
        for c in context_path() + [cur_context()]:
            if _exists_in(c, name):
                return name if c in ("System`", "Global`") else c + name
        ctx = cur_context()
        full = ctx + name
        if create:
            _new_symbol_hook(name, ctx)
            _register(full)
        return _internal(full)
    if create and not _exists_in(_context_of_full(full), _short(full)[1]):
        _new_symbol_hook(_short(full)[1], _context_of_full(full))
        _register(full)
    return _internal(full)


def _register(full):
    d = _ctxdata()
    c, s = _short(full)
    d["removed"].discard(_internal(full))
    if c == "System`":
        return
    if c == "Global`":
        d["globals"].add(s)
    else:
        d["symbols"].add(full)


def _default_contexts():
    o = st().own
    return "$Context" not in o and "$ContextPath" not in o and "$ContextAliases" not in o and \
        "$NewSymbol" not in o


def _collect_symbols(t, out):
    if isinstance(t, Symbol):
        out.add(t.name)
    elif isinstance(t, Expr):
        if isinstance(t.head, str):
            out.add(t.head)
        else:
            _collect_symbols(t.head, out)
        for a in t.args:
            _collect_symbols(a, out)


def _rename_tree(t, m):
    if isinstance(t, Symbol):
        n = m.get(t.name)
        return Symbol(n) if n is not None and n != t.name else t
    if isinstance(t, Expr):
        h = t.head
        if isinstance(h, str):
            h2 = m.get(h, h)
        else:
            h2 = _rename_tree(h, m)
        args = tuple(_rename_tree(a, m) for a in t.args)
        if h2 is h and all(x is y for x, y in zip(args, t.args)):
            return t
        return Expr(h2, *args)
    return t


def post_parse(expr):
    """Create the symbols of a parsed expression and resolve their names against the current contexts."""
    names = set()
    _collect_symbols(expr, names)
    if not names:
        return expr
    d = _ctxdata()
    if _default_contexts() and not any("`" in n for n in names):
        sysn = system_names()
        new = {n for n in names if n not in sysn}
        d["globals"] |= new
        if d["removed"]:
            d["removed"] -= new
        return expr
    m = {}
    for n in names:
        r = resolve_name(n)
        if r != n:
            m[n] = r
    return _rename_tree(expr, m) if m else expr


def _install_parse_hook():
    from .. import parser
    parser.POST_PARSE_HOOK = post_parse


def all_symbols():
    """Full names of every symbol that exists (System` names included)."""
    d = _ctxdata()
    out = {"System`" + n for n in system_names()}
    out |= {"Global`" + n for n in (d["globals"] | _defined_bare()) if not is_system(n)}
    out |= {n for n in d["symbols"]}
    for k in list(st().own) + list(st().down) + list(st().attrs):
        if "`" in k:
            out.add(k)
    removed = {full_name(r) for r in d["removed"]}
    return out - removed


def _display_name(full):
    """Short name if its context is $Context or on $ContextPath (and not shadowed), else the full name."""
    c, s = _short(full)
    path = context_path() + [cur_context()]
    if c in path:
        for c2 in path:
            if c2 == c:
                return s
            if _exists_in(c2, s):
                return full
    return full


def _abbrev_regex(p, ignore_case=False, star="[^`]*"):
    src = "".join(star if ch == "*" else "[^A-Z`]+" if ch == "@" else re.escape(ch) for ch in p)
    return re.compile(src + r"\Z", re.DOTALL | (re.IGNORECASE if ignore_case else 0))


def _string_pattern_regex(p, ignore_case=False):
    from .wl_strings import _rx
    rx, _ = _rx(p, "", ignore_case)
    return re.compile("(?:" + rx.pattern + r")\Z", rx.flags)


def _name_matcher(p, use_aliases=True, ignore_case=False):
    """Predicate on full names for a Names-style specification."""
    if isinstance(p, str):
        if "`" in p:
            pats = [p]
            if use_aliases:
                ex = _expand_alias(p)
                if ex != p:
                    pats.append(ex)
            if p.startswith("`"):
                pats = [cur_context() + p[1:]]
            rxs = [_abbrev_regex(q, ignore_case) for q in pats]
            return lambda full: any(r.match(full) for r in rxs)
        rx = _abbrev_regex(p, ignore_case)
        path = set(context_path()) | {cur_context()}
        return lambda full: _context_of_full(full) in path and bool(rx.match(_short(full)[1]))
    if K.is_list(p):
        ms = [_name_matcher(q, use_aliases, ignore_case) for q in p.args]
        return lambda full: any(m(full) for m in ms)
    if isinstance(p, Expr) and p.head in ("StringExpression", "RegularExpression", "Alternatives"):
        rx = _string_pattern_regex(p, ignore_case)
        return lambda full: bool(rx.match(full)) or bool(rx.match(_short(full)[1]))
    raise ValueError(f"Names: {p!r} is not a string pattern")


def _opts_from(args, allowed, fname):
    pos, opts = [], {}
    for a in args:
        if is_rule(a) and isinstance(a.args[0], Symbol) and a.args[0].name in allowed:
            opts[a.args[0].name] = a.args[1]
        elif is_rule(a) and isinstance(a.args[0], Symbol):
            raise ValueError(f"{fname}: unknown option {a.args[0].name}")
        else:
            pos.append(a)
    return pos, opts


def names(spec=None, use_aliases=True, ignore_case=False):
    syms = all_symbols()
    if spec is None:
        return sorted(_display_name(f) for f in syms)
    m = _name_matcher(spec, use_aliases, ignore_case)
    out = sorted({_display_name(f) for f in syms if m(f)})
    return out


@fn("Names", area=AREA, status="added",
    notes="Names[\"pattern\"] (metacharacters * and @, a context prefix such as \"Global`*\", StringExpression or "
          "RegularExpression), Names[{p1, ...}], Names[] ; options IgnoreCase, ResolveContextAliases, "
          "SpellingCorrection -> False. Names are short when their context is $Context or on $ContextPath. "
          "System` contains the symbols MikoshiLang implements")
def _names(*args: Expr) -> ListOf[Expr]:
    pos, o = _opts_from(args, {"IgnoreCase", "ResolveContextAliases", "SpellingCorrection"}, "Names")
    if o.get("SpellingCorrection", False) is True:
        raise NotImplementedError("Names: SpellingCorrection -> True is not supported")
    if len(pos) > 1:
        raise ValueError("Names: expected one pattern")
    r = names(pos[0] if pos else None, o.get("ResolveContextAliases", True) is not False,
              o.get("IgnoreCase", False) is True)
    return L(*r)


@fn("NameQ", area=AREA, status="added",
    notes="NameQ[\"string\"]: True if any symbol name matches (abbreviated string patterns); "
          "ResolveContextAliases option")
def _nameq(*args: Expr) -> Boolean:
    pos, o = _opts_from(args, {"ResolveContextAliases", "IgnoreCase"}, "NameQ")
    if len(pos) != 1:
        raise ValueError("NameQ: expected one string")
    p = pos[0]
    if not isinstance(p, str):
        raise Unevaluated()
    return bool(names(p, o.get("ResolveContextAliases", True) is not False, o.get("IgnoreCase") is True))


def _all_contexts():
    out = {"System`", "Global`"} | set(context_path()) | set(packages()) | {cur_context()}
    for f in all_symbols():
        c = _context_of_full(f)
        parts = c.split("`")[:-1]
        for i in range(1, len(parts) + 1):
            out.add("`".join(parts[:i]) + "`")
    for c in list(out):
        parts = c.split("`")[:-1]
        for i in range(1, len(parts)):
            out.add("`".join(parts[:i]) + "`")
    return out


@fn("Contexts", area=AREA, status="added",
    notes="Contexts[] and Contexts[\"pattern\"]: the contexts of all existing symbols (and their parent contexts), "
          "$ContextPath and $Packages, sorted; ResolveContextAliases option")
def _contexts(*args: Expr) -> ListOf[Expr]:
    pos, o = _opts_from(args, {"ResolveContextAliases"}, "Contexts")
    cs = _all_contexts()
    use = o.get("ResolveContextAliases", True) is not False
    if pos:
        p = pos[0]
        if not isinstance(p, str):
            raise ValueError("Contexts: the pattern must be a string")
        pats = [p]
        if use:
            ex = _expand_alias(p)
            if ex != p:
                pats.append(ex)
            for a in aliases():
                if _abbrev_regex(p, star=".*").match(a):
                    cs = cs | {a}
        rxs = [_abbrev_regex(q, star=".*") for q in pats]
        cs = {c for c in cs if any(r.match(c) for r in rxs)}
    return L(*sorted(cs))


def _ctx_of_name(name):
    full = full_name(name)
    return _context_of_full(full)


@held("Context", attrs={"HoldFirst"}, notes="Context[] gives $Context; Context[symbol] and Context[\"name\"] the "
      "context of a symbol (System` for built-ins, Global` for unqualified user symbols); a name that does not "
      "exist raises")
def _h_context(a, ev_, expr):
    _need(a, 0, "Context", 1)
    if not a:
        return cur_context()
    x = a[0]
    if isinstance(x, Symbol):
        return _ctx_of_name(x.name)
    x = ev_(x)
    if isinstance(x, Symbol):
        return _ctx_of_name(x.name)
    if isinstance(x, str):
        if not _NAME_RE.match(x):
            raise ValueError(f"Context: {x!r} is not a valid symbol name")
        r = resolve_name(x, create=False)
        full = full_name(r)
        if full not in all_symbols():
            raise ValueError(f"Context::notfound: Symbol {x} not found")
        return _context_of_full(full)
    raise ValueError(f"Context: {x!r} is not a symbol or a string")


def _resolve_context_arg(c, fname):
    if not isinstance(c, str):
        raise ValueError(f"{fname}: {c!r} is not a context name")
    if c.startswith("`"):
        c = cur_context() + c[1:]
    return _check_context(c, fname)


@fn("Begin", area=AREA, status="added",
    notes="Begin[\"ctx`\"] sets $Context (a context starting with ` extends the current one) and returns it; "
          "End[] restores $Context and $ContextAliases. Affects how later inputs are parsed")
def _begin(c: String) -> Expr:
    c = _resolve_context_arg(c, "Begin")
    d = _ctxdata()
    o = st().own
    d["begin"].append((o.get("$Context", UNH), o.get("$ContextAliases", UNH)))
    _set_own("$Context", c)
    return c


def _restore_own(name, v):
    if v is UNH:
        st().own.pop(name, None)
    else:
        st().own[name] = v


@fn("End", area=AREA, status="added",
    notes="End[] returns the current context and restores $Context and $ContextAliases to their values before "
          "the matching Begin; an unmatched End raises")
def _end() -> Expr:
    d = _ctxdata()
    if not d["begin"]:
        raise ValueError("End::noctx: No previous context defined")
    c = cur_context()
    ctx, al = d["begin"].pop()
    _restore_own("$Context", ctx)
    _restore_own("$ContextAliases", al)
    return c


@fn("BeginPackage", area=AREA, status="added",
    notes="BeginPackage[\"ctx`\"] sets $Context to ctx` and $ContextPath to {ctx`, System`}, adds ctx` to "
          "$Packages; BeginPackage[\"ctx`\", {\"need1`\", ...}] calls Needs on each and puts them on the path")
def _beginpackage(c: String, needs: Expr = None) -> Expr:
    c = _resolve_context_arg(c, "BeginPackage")
    need_list = []
    if needs is not None:
        if isinstance(needs, str):
            need_list = [needs]
        elif K.is_list(needs) and all(isinstance(x, str) for x in needs.args):
            need_list = list(needs.args)
        else:
            raise ValueError("BeginPackage: the second argument must be a list of context names")
    d = _ctxdata()
    o = st().own
    d["package"].append((c, o.get("$Context", UNH), o.get("$ContextPath", UNH)))
    for n in need_list:
        _needs(n)
    _set_own("$ContextPath", L(c, *need_list, "System`"))
    _set_own("$Context", c)
    if c not in packages():
        _set_own("$Packages", L(c, *packages()))
    return NULL


@fn("EndPackage", area=AREA, status="added",
    notes="EndPackage[] restores $Context and $ContextPath to their values before BeginPackage and prepends the "
          "package context to $ContextPath")
def _endpackage() -> Expr:
    d = _ctxdata()
    if not d["package"]:
        raise ValueError("EndPackage::noctx: No previous context defined")
    c, ctx, path = d["package"].pop()
    _restore_own("$Context", ctx)
    _restore_own("$ContextPath", path)
    p = context_path()
    if c not in p:
        _set_own("$ContextPath", L(c, *p))
    else:
        _set_own("$ContextPath", L(c, *[x for x in p if x != c]))
    return NULL


# package files ------------------------------------------------------------------------------------------------

PACKAGE_PATH = []          # extra directories searched by Get["ctx`"] and Needs (after the working directory)


def _search_dirs():
    dirs = [os.getcwd()] + list(PACKAGE_PATH)
    env = os.environ.get("MIKOSHILANG_PATH")
    if env:
        dirs += [p for p in env.split(os.pathsep) if p]
    return dirs


def find_package_file(c):
    parts = c.rstrip("`").split("`")
    rel = os.path.join(*parts)
    for base in _search_dirs():
        for cand in (rel + ".wl", rel + ".m", os.path.join(rel, "init.wl"), os.path.join(rel, "init.m"),
                     os.path.join(rel, "Kernel", "init.wl"), os.path.join(rel, "Kernel", "init.m")):
            p = os.path.join(base, cand)
            if os.path.isfile(p):
                return p
    return None


def split_statements(text):
    """Split source text into top-level expressions (one per line unless brackets or an operator continue it)."""
    from ..parser import tokenize, ParseError
    lines = text.split("\n")
    out, buf = [], []
    for line in lines:
        buf.append(line)
        chunk = "\n".join(buf)
        if not chunk.strip():
            buf = []
            continue
        try:
            toks = tokenize(chunk)
        except ParseError:
            if '"' in chunk or "(*" in chunk:
                continue            # an unterminated string or comment continues on the next line
            raise
        if not toks:
            buf = []
            continue
        depth = 0
        for t in toks:
            if t.type in ("LPAREN", "LSQUARE", "LBRACE", "LASSOC"):
                depth += 1
            elif t.type in ("RPAREN", "RSQUARE", "RBRACE", "RASSOC"):
                depth -= 1
        last = toks[-1].type
        if depth > 0 or last in _CONTINUES:
            continue
        out.append(chunk)
        buf = []
    if buf and "\n".join(buf).strip():
        out.append("\n".join(buf))
    return out


_CONTINUES = {"PLUS", "MINUS", "TIMES", "DIVIDE", "POWER", "SET", "SETDELAYED", "UPSET", "UPSETDELAYED", "TAGSET",
              "RULE", "RULEDELAYED", "AND", "OR", "COMMA", "AT", "APPLY", "MAP", "MAPALL", "APPLYLEVEL1",
              "REPLACEALL", "REPLACEREPEATED", "CONDITION", "EQUAL", "UNEQUAL", "LESS", "GREATER", "LEQ", "GEQ",
              "SAMEQ", "UNSAMEQ", "STRINGJOIN", "STRINGEXPR", "ALTERNATIVES", "DOT", "POSTFIXAPP", "COLON",
              "TWOWAYRULE", "ADDTO", "SUBTRACTFROM", "TIMESBY", "DIVIDEBY", "APPLYTO", "COMPOSITION",
              "RIGHTCOMPOSITION", "SPAN"}


def run_source(text):
    """Parse and evaluate text one top-level expression at a time (as Get does); the last result."""
    from ..parser import parse
    result = NULL
    for chunk in split_statements(text):
        e = parse(chunk)
        if e is None:
            continue
        result = E.evaluate(e)
    return result


def _get_file(path):
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        raise ValueError(f"Get::noopen: Cannot open {path}.")
    return run_source(text)


@fn("Get", area=AREA, status="added",
    notes="Get[\"file\"] (<<file) reads a file of Wolfram Language input and evaluates each expression in turn, "
          "returning the last result (a relative file name is also searched for in PACKAGE_PATH and "
          "$MIKOSHILANG_PATH); Get[\"ctx`\"] loads ctx.wl, ctx.m, ctx/init.m or ctx/Kernel/init.m from the "
          "working directory, PACKAGE_PATH or $MIKOSHILANG_PATH. A file that cannot be opened raises Get::noopen")
def _get(name, *opts):
    if opts:
        raise NotImplementedError("Get: options are not supported")
    if not isinstance(name, str):
        if isinstance(name, Expr) and name.head == "File" and len(name.args) == 1 and isinstance(name.args[0], str):
            name = name.args[0]
        else:
            raise ValueError(f"Get: {name!r} is not a file name or a context")
    if _CTX_RE.match(name):
        p = find_package_file(name)
        if p is None:
            raise ValueError(f"Get::noopen: Cannot open {name}.")
        return _get_file(p)
    path = os.path.expanduser(name)
    if not os.path.isabs(path) and not os.path.exists(path):
        for d in _search_dirs():
            if os.path.exists(os.path.join(d, path)):
                path = os.path.join(d, path)
                break
    return _get_file(path)


def _needs(c, file=None):
    target = c
    alias = UNH
    if is_rule(c):
        target, alias = c.args
    target = _resolve_context_arg(target, "Needs")
    path_before = context_path()
    if target not in packages():
        if file is not None:
            if not isinstance(file, str):
                raise ValueError("Needs: the file name must be a string")
            _get_file(os.path.expanduser(file))
        else:
            p = find_package_file(target)
            if p is None:
                raise ValueError(f"Needs::nocont / Get::noopen: Cannot open {target}.")
            _get_file(p)
        if target not in packages():
            _set_own("$Packages", L(target, *packages()))
    if alias is UNH:
        p = context_path()
        if target not in p:
            _set_own("$ContextPath", L(target, *p))
    else:
        if target not in path_before:       # the package's EndPackage put it on the path: undo that
            _set_own("$ContextPath", L(*[x for x in context_path() if x != target]))
    if alias is UNH:
        pass
    elif isinstance(alias, str):
        _check_context(alias, "Needs")
        al = aliases()
        al[alias] = target
        _set_own("$ContextAliases", _assoc_of(al))
    elif not is_sym(alias, "None"):
        raise ValueError("Needs: the alias must be a context name or None")
    return NULL


@fn("Needs", area=AREA, status="added",
    notes="Needs[\"ctx`\"] loads the package (Get[\"ctx`\"]) unless ctx` is in $Packages and puts ctx` on "
          "$ContextPath; Needs[\"ctx`\" -> \"alias`\"] adds a context alias instead; Needs[\"ctx`\" -> None] only "
          "loads; Needs[\"ctx`\", \"file\"] loads the given file")
def _needs_fn(c: Expr, file: Expr = None) -> Expr:
    return _needs(c, file)


@fn("DeclarePackage", area=AREA, status="added",
    notes="DeclarePackage[\"ctx`\", {\"name1\", ...}] creates ctx`name with attribute Stub and prepends ctx` to "
          "$ContextPath; the first use of such a symbol calls Needs[\"ctx`\"] and evaluates it again")
def _declarepackage(c: String, nms: Expr) -> Expr:
    c = _resolve_context_arg(c, "DeclarePackage")
    if isinstance(nms, str):
        nms = L(nms)
    if not K.is_list(nms) or not all(isinstance(n, str) and _NAME_RE.match(n) and "`" not in n for n in nms.args):
        raise ValueError("DeclarePackage: the second argument must be a list of symbol names")
    stubs = ext("Stub")
    for n in nms.args:
        full = c + n
        _register(full)
        stubs[_internal(full)] = c
        st().attrs.setdefault(_internal(full), set()).add("Stub")
    p = context_path()
    _set_own("$ContextPath", L(c, *[x for x in p if x != c]))
    _install_arg_hook()
    return NULL


def _load_stub(name):
    stubs = st().ext.get("Stub", {})
    c = stubs.get(name)
    if c is None:
        return False
    for k in [k for k, v in stubs.items() if v == c]:
        del stubs[k]
        a = st().attrs.get(k)
        if a:
            a.discard("Stub")
    _needs(c)
    return True


def _stub_symbol(expr, depth):
    _load_stub(expr.name)
    return E.evaluate(expr, depth + 1)


E.STUB_HOOK = _stub_symbol


# ─────────────────────────── system variables ───────────────────────────

def _getter(name, default):
    def g():
        return default() if callable(default) else default
    E.SYSTEM_VARIABLES[name] = g
    _ASSIGNABLE.add(name)


_getter("$Context", "Global`")
_getter("$ContextPath", lambda: L("System`", "Global`"))
_getter("$ContextAliases", lambda: Expr("Association"))
_getter("$Packages", lambda: L("System`", "Global`"))
_getter("$Line", 1)
_getter("$HistoryLength", lambda: Symbol("Infinity"))
_getter("$RecursionLimit", E._MAX_DEPTH)
_getter("$IterationLimit", 4096)
E.SYSTEM_VARIABLES["$ModuleNumber"] = lambda: st().counter + 1
_ASSIGNABLE.add("$ModuleNumber")
for _n in ("$Pre", "$Post", "$PrePrint", "$PreRead", "$Epilog", "$NewSymbol"):
    _ASSIGNABLE.add(_n)
    INERT[_n] = "session hook variable"


def _validate_limit(v, name):
    if is_sym(v, "Infinity") or (isinstance(v, Expr) and v.head == "DirectedInfinity"):
        return Symbol("Infinity")
    if K.is_int(v) and int(v) >= 20:
        return int(v)
    raise ValueError(f"{name}: cannot set {name} to {v!r}; value must be Infinity or an integer at least 20")


def _assign_var(lhs, rhs, ev_):
    """Validated assignment to a system variable (stored as an OwnValue so that Block localises it)."""
    name = lhs.name if isinstance(lhs, Symbol) else lhs.head
    if isinstance(lhs, Expr):
        if name == "$ContextAliases" and len(lhs.args) == 1:
            k = ev_(lhs.args[0])
            v = ev_(rhs)
            _check_alias(k, v)
            al = aliases()
            al[k] = v
            _set_own(name, _assoc_of(al))
            return
        raise ValueError(f"Set: cannot assign to {lhs!r}")
    v = ev_(rhs) if not isinstance(rhs, (str, int)) else rhs
    if name == "$Context":
        _check_context(v, "$Context")
    elif name in ("$ContextPath", "$Packages"):
        if not (K.is_list(v) and all(isinstance(c, str) and _CTX_RE.match(c) for c in v.args)):
            raise ValueError(f"{name} must be a list of context names (strings such as \"Global`\")")
    elif name == "$ContextAliases":
        if not (isinstance(v, Expr) and v.head == "Association"):
            raise ValueError("$ContextAliases must be an association \"alias`\" -> \"context`\"")
        for r in v.args:
            _check_alias(r.args[0], r.args[1])
    elif name in ("$RecursionLimit", "$IterationLimit"):
        v = _validate_limit(v, name)
    elif name == "$Line":
        if not (K.is_int(v) and int(v) >= 0):
            raise ValueError("$Line must be a non-negative integer")
    elif name == "$HistoryLength":
        if not ((K.is_int(v) and int(v) >= 0) or is_sym(v, "Infinity")):
            raise ValueError("$HistoryLength must be a non-negative integer or Infinity")
    elif name == "$ModuleNumber":
        if not (K.is_int(v) and int(v) >= 1):
            raise ValueError("$ModuleNumber must be a positive integer")
        st().counter = int(v) - 1
        return
    _set_own(name, v)


def _check_alias(k, v):
    if not (isinstance(k, str) and _CTX_RE.match(k)):
        raise ValueError(f"$ContextAliases: {k!r} is not a valid alias (a string such as \"a`\")")
    if not (isinstance(v, str) and _CTX_RE.match(v)):
        raise ValueError(f"$ContextAliases: {v!r} is not a fully qualified context name")


for _n in ("$Context", "$ContextPath", "$ContextAliases", "$Packages", "$RecursionLimit", "$IterationLimit",
           "$Line", "$HistoryLength", "$ModuleNumber"):
    K.ASSIGN_HOOKS[_n] = _assign_var

for _n, _note in [
        ("$Context", "current context (default \"Global`\"); set by Begin, BeginPackage, End, EndPackage; "
                     "assignable (validated); Block localises it"),
        ("$ContextPath", "contexts searched (before $Context) when a name is parsed; default {\"System`\", "
                         "\"Global`\"}; modified by BeginPackage, EndPackage, Needs, DeclarePackage"),
        ("$ContextAliases", "association \"alias`\" -> \"context`\" used when parsing qualified names; "
                            "$ContextAliases[\"a`\"] = \"ctx`\" and =. update it; Needs[ctx -> alias] adds one"),
        ("$Packages", "contexts of loaded packages (BeginPackage adds; Needs consults it)"),
        ("$RecursionLimit", "maximum evaluation depth (default 1024, Infinity or an integer >= 20); exceeding it "
                            "raises $RecursionLimit::reclim. MikoshiLang counts nested evaluator calls"),
        ("$IterationLimit", "maximum length of an evaluation chain (default 4096): each rewrite of an expression by "
                            "a definition or rule counts once; exceeding it raises $IterationLimit::itlim"),
        ("$ModuleNumber", "the number the next Module or Unique symbol gets (x$nnn); assignable"),
        ("$Line", "number of the current input line in the main loop (Session.evaluate_input); assignable"),
        ("$HistoryLength", "number of In/Out lines kept by the main loop (default Infinity)")]:
    _added(_n, _note)


_orig_unset = K._HELD["Unset"]


def _unset_hook(lhs):
    """=. for system variables and special left-hand sides; UNH if not special."""
    o = st().own
    if isinstance(lhs, Symbol) and lhs.name in _ASSIGNABLE:
        if lhs.name == "$ModuleNumber":
            raise ValueError("$ModuleNumber cannot be unset")
        o.pop(lhs.name, None)
        return NULL
    if isinstance(lhs, Expr) and isinstance(lhs.head, str):
        if lhs.head == "$ContextAliases" and len(lhs.args) == 1:
            k = ev(lhs.args[0])
            al = aliases()
            al.pop(k, None)
            _set_own("$ContextAliases", _assoc_of(al))
            return NULL
        h = UNSET_HOOKS.get(lhs.head)
        if h is not None:
            return h(lhs)
    return UNH


UNSET_HOOKS = {}


def _h_unset(a, ev_, expr):
    if len(a) != 1:
        raise ValueError("Unset expects lhs =.")
    r = _unset_hook(a[0])
    if r is not UNH:
        return r
    lhs = a[0]
    name = lhs.name if isinstance(lhs, Symbol) else _innermost_head(lhs) if isinstance(lhs, Expr) else None
    if name is not None and is_protected(name):
        raise ValueError(f"Unset: {name} is protected")
    return _orig_unset(a, ev_, expr)


K._HELD["Unset"] = _h_unset


# ─────────────────────────── attributes ───────────────────────────

NEW_ATTRIBUTES = {"HoldAllComplete", "NHoldAll", "NHoldFirst", "NHoldRest", "Stub", "Temporary",
                  "SubValuesHoldAll", "NonThreadable", "Locked", "Constant", "ReadProtected"}
K.KNOWN_ATTRIBUTES |= NEW_ATTRIBUTES
set_attrs("Unevaluated", "HoldAllComplete")
K.SUPPORTED_USER_ATTRIBUTES |= NEW_ATTRIBUTES

for _n, _note in [
        ("HoldAllComplete", "attribute: no argument is evaluated, Sequence is not spliced, Unevaluated is kept and "
                            "upvalues are not used"),
        ("NHoldAll", "attribute: N does not affect any argument of f"),
        ("NHoldFirst", "attribute: N does not affect the first argument of f"),
        ("NHoldRest", "attribute: N affects only the first argument of f"),
        ("Stub", "attribute set by DeclarePackage: using the symbol loads its package"),
        ("Temporary", "attribute of Module local symbols (and Unique[x, {Temporary}]); reported by Attributes"),
        ("SubValuesHoldAll", "attribute: in f[a...][s...], the subvalue arguments s are held (Unevaluated "
                             "stripped, Sequence spliced)"),
        ("NonThreadable", "attribute: arithmetic and other Listable functions do not thread lists over f or "
                          "f[...]; the threadable arguments still combine"),
        ("Locked", "attribute: the attributes of the symbol can no longer be changed (Protect/Unprotect, "
                   "SetAttributes, ClearAttributes, Attributes[f] = ... raise)"),
        ("Constant", "attribute: Dt treats the symbol as a constant (Dt[c] = 0, Dt[c[x], x] = 0)"),
        ("ReadProtected", "attribute: Definition, FullDefinition and Information show only the attributes; the "
                          "values are still used in evaluation and DownValues etc. still return them")]:
    _added(_n, _note)


def _temporary(name):
    return name in st().temporaries


def full_attributes(name):
    a = set(K.attributes(name))
    if _protected_system(name) and name not in st().ext.get("Unprotected", ()):
        a.add("Protected")
    if name in st().ext.get("Unprotected", ()):
        a.discard("Protected")
    if _temporary(name):
        a.add("Temporary")
    return a


def _attr_names(x):
    items = x.args if K.is_list(x) else (x,)
    out = []
    for i in items:
        if not isinstance(i, Symbol):
            raise ValueError(f"Attributes: {i!r} is not an attribute name")
        if i.name not in K.KNOWN_ATTRIBUTES:
            raise ValueError(f"Attributes: unknown attribute {i.name}")
        out.append(i.name)
    return out


def _target_names(x, ev_, fname):
    """Symbols named by x (a symbol, a string, a string pattern or a list of these), x held."""
    if isinstance(x, Symbol):
        return [x.name]
    if K.is_list(x):
        out = []
        for y in x.args:
            out.extend(_target_names(y, ev_, fname))
        return out
    v = ev_(x)
    if isinstance(v, Symbol):
        return [v.name]
    if isinstance(v, str):
        if any(c in v for c in "*@"):
            return [resolve_name(n, create=False) for n in names(v)]
        return [resolve_name(v)]
    if K.is_list(v):
        return _target_names(v, lambda y: y, fname)
    raise ValueError(f"{fname}: {x!r} is not a symbol or a string")


def _locked(name):
    return "Locked" in st().attrs.get(name, ())


def _check_unlocked(name, fname):
    if _locked(name):
        raise ValueError(f"{fname}::locked: Symbol {name} is locked")


@held("Attributes", attrs={"HoldAll", "Listable"},
      notes="Attributes[f], Attributes[\"f\"], Attributes[{f, g}] (built-in Protected included, Temporary for "
            "live Module symbols, Stub, user attributes); Attributes[f] = {...} replaces them")
def _h_attributes(a, ev_, expr):
    _need(a, 1, "Attributes")
    x = a[0]
    if K.is_list(x):
        return L(*(_h_attributes((y,), ev_, expr) for y in x.args))
    if isinstance(x, Symbol):
        nm = x.name
    else:
        v = ev_(x)
        if isinstance(v, str):
            nm = resolve_name(v, create=False)
        elif isinstance(v, Symbol):
            nm = v.name
        elif K.is_list(v):
            return L(*(_h_attributes((y,), ev_, expr) for y in v.args))
        else:
            raise ValueError("Attributes expects a symbol or a string")
    return L(*(Symbol(n) for n in sorted(full_attributes(nm))))


def _assign_attributes(lhs, rhs, ev_):
    if len(lhs.args) != 1:
        raise ValueError("Attributes[f] = {...} needs one symbol")
    names_ = _target_names(lhs.args[0], ev_, "Attributes")
    attrs = _attr_names(ev_(rhs))
    for n in names_:
        _check_unlocked(n, "Attributes")
        if is_system(n):
            raise ValueError(f"Attributes: {n} is a protected built-in symbol")
        st().attrs[n] = set(attrs)
    _note_new_attrs(attrs)
    return NULL


K.ASSIGN_HOOKS["Attributes"] = _assign_attributes

_orig_setattributes = K._h_setattributes


def _h_setattr(a, ev_, expr, clear=False):
    if len(a) != 2:
        raise ValueError("SetAttributes expects SetAttributes[symbol, attrs]")
    syms = _target_names(a[0], ev_, "SetAttributes")
    attrs = _attr_names(ev_(a[1]))
    for s in syms:
        _check_unlocked(s, "ClearAttributes" if clear else "SetAttributes")
        if is_system(s) and is_protected(s) and not (clear and attrs == ["Protected"]):
            raise ValueError(f"{'ClearAttributes' if clear else 'SetAttributes'}: {s} is protected")
    _orig_setattributes((L(*(Symbol(s) for s in syms)), L(*(Symbol(x) for x in attrs))), lambda y: y, expr,
                        clear=clear)
    if not clear:
        _note_new_attrs(attrs)
    return NULL


def _note_new_attrs(attrs):
    if "NonThreadable" in attrs:
        _install_arg_hook()


K._HELD["SetAttributes"] = _h_setattr
K._HELD["ClearAttributes"] = lambda a, ev_, e: _h_setattr(a, ev_, e, clear=True)


def _protect(a, ev_, on):
    changed = []
    for x in a:
        for n in _target_names(x, ev_, "Protect" if on else "Unprotect"):
            _check_unlocked(n, "Protect" if on else "Unprotect")
            was = is_protected(n)
            if on:
                if n in st().ext.get("Unprotected", {}):
                    del st().ext["Unprotected"][n]
                if not is_protected(n):
                    st().attrs.setdefault(n, set()).add("Protected")
            else:
                st().attrs.get(n, set()).discard("Protected")
                if is_protected(n):
                    ext("Unprotected")[n] = True
            if was != is_protected(n):
                changed.append(_display_name(full_name(n)))
    return L(*changed)


@held("Protect", attrs={"HoldAll"}, notes="Protect[s1, s2, ...] (symbols, names or string patterns) sets the "
      "Protected attribute (assignments then raise); returns the names whose protection changed")
def _h_protect(a, ev_, expr):
    return _protect(a, ev_, True)


@held("Unprotect", attrs={"HoldAll"}, notes="Unprotect[s1, ...] removes Protected, also from built-in symbols "
      "(definitions for them are then used before the built-in rules); returns the names whose protection changed. "
      "Special forms evaluated by the kernel itself (If, Set, Module, ...) still cannot get definitions")
def _h_unprotect(a, ev_, expr):
    return _protect(a, ev_, False)


_orig_assign = K._assign


def _assign_checked(lhs, rhs, ev_):
    if isinstance(lhs, Expr) and isinstance(lhs.head, str) and lhs.head in K._HELD and lhs.head not in K.ASSIGN_HOOKS \
            and lhs.head not in ("List", "Part"):
        raise ValueError(f"{lhs.head} is a kernel special form; definitions for it are not supported")
    return _orig_assign(lhs, rhs, ev_)


K._assign = _assign_checked


# ─────────────────────────── upvalues ───────────────────────────

def _store_upvalue(tag, lhs, rhs):
    if is_protected(tag):
        raise ValueError(f"TagSet: tag {tag} is protected")
    st().note_write(tag, lhs, rhs)
    defs = ext("UpValues").setdefault(tag, [])
    lt = (K.tag(lhs), K._condition_tag(rhs))
    for i, (l, r, g) in enumerate(defs):
        if (K.tag(l), K._condition_tag(r)) == lt:
            defs[i] = (lhs, rhs, g)
            return
    g = K._generality(lhs) if K.has_pattern(lhs) else -1.0
    pos = len(defs)
    for i, (_, _, gi) in enumerate(defs):
        if gi > g:
            pos = i
            break
    defs.insert(pos, (lhs, rhs, g))
    E.UPVALUE_HOOK = apply_upvalues


def apply_upvalues(expr, ev_):
    ups = K.state().ext.get("UpValues")
    if not ups or not isinstance(expr.head, str):
        return UNH
    seen = set()
    for a in expr.args:
        t = _tag_of(a)
        if t is None or t in seen:
            continue
        seen.add(t)
        defs = ups.get(t)
        if not defs:
            continue
        for lhs, rhs, _ in list(defs):
            for b in K.match(lhs, expr, {}, ev_):
                r = rhs
                if K.OPTION_VALUE_HOOK is not None:
                    r = K.OPTION_VALUE_HOOK(lhs, r, b, expr)
                if isinstance(r, Expr) and r.head == "Condition" and len(r.args) == 2:
                    if ev_(K.substitute(r.args[1], b)) is not True:
                        continue
                    return K.substitute(r.args[0], b)
                return K.substitute(r, b)
    return UNH


def _eval_lhs_args(lhs, ev_):
    """Evaluate the arguments of a pattern-free lhs (respecting its head's Hold attributes), as Set does."""
    if not isinstance(lhs, Expr) or K.has_pattern(lhs):
        return lhs
    head = lhs.head
    hold = K.attributes(head) if isinstance(head, str) else set()
    args = list(lhs.args)
    for i, x in enumerate(args):
        if "HoldAll" in hold or "HoldAllComplete" in hold or (i == 0 and "HoldFirst" in hold) or \
                (i > 0 and "HoldRest" in hold):
            continue
        args[i] = ev_(x)
    return Expr(head, *args)


def _arg_tag(a):
    """Upvalue tag of one argument of a left-hand side: a symbol, the head of x_h / _h, or an innermost head."""
    a = _strip_holdpattern(a)
    if isinstance(a, Symbol):
        return a.name
    if not isinstance(a, Expr):
        return None
    if a.head in ("Condition", "PatternTest") and len(a.args) == 2:
        return _arg_tag(a.args[0])
    if a.head == "Pattern" and len(a.args) == 2:
        return _arg_tag(a.args[1])
    if a.head in ("Blank", "BlankSequence", "BlankNullSequence"):
        return a.args[0].name if a.args and isinstance(a.args[0], Symbol) else None
    if a.head in ("Alternatives", "Optional", "Repeated", "RepeatedNull", "Except", "OptionsPattern"):
        return None
    return _innermost_head(a)


def _upset(lhs, rhs, ev_, fname):
    lhs = _strip_holdpattern(lhs)
    if not isinstance(lhs, Expr):
        raise ValueError(f"{fname}: the left-hand side {lhs!r} has no arguments to attach the rule to")
    lhs = _eval_lhs_args(lhs, ev_)
    tags = []
    for a in lhs.args:
        t = _arg_tag(a)
        if t is not None and t not in tags:
            tags.append(t)
    if not tags:
        raise ValueError(f"{fname}::nosym: {lhs} does not contain a symbol to attach a rule to")
    for t in tags:
        _store_upvalue(t, lhs, rhs)


@held("UpSet", attrs={"HoldFirst", "SequenceHold"},
      notes="lhs ^= rhs: rhs is evaluated and the rule is attached (as an upvalue) to every symbol that is an "
            "argument of lhs or the head of an argument (h[x_] in f[h[x_]] attaches to h); returns rhs")
def _h_upset(a, ev_, expr):
    _need(a, 2, "UpSet")
    rhs = ev_(a[1])
    _upset(a[0], rhs, ev_, "UpSet")
    return rhs


@held("UpSetDelayed", attrs={"HoldAll", "SequenceHold"},
      notes="lhs ^:= rhs: like UpSet with rhs kept unevaluated until the rule is used; returns Null")
def _h_upsetdelayed(a, ev_, expr):
    _need(a, 2, "UpSetDelayed")
    _upset(a[0], a[1], ev_, "UpSetDelayed")
    return NULL


def _tag_positions(tag, lhs):
    """'down' if tag is the (innermost) head of lhs, 'up' if an argument or an argument's head, else None."""
    if _innermost_head(lhs) == tag:
        return "down"
    for a in lhs.args:
        if _arg_tag(a) == tag:
            return "up"
    return None


def _tagset(a, ev_, delayed, fname):
    _need(a, 3, fname)
    f, lhs, rhs = a
    if not isinstance(f, Symbol):
        raise ValueError(f"{fname}: the tag {f!r} must be a symbol")
    lhs = _strip_holdpattern(lhs)
    if isinstance(lhs, Symbol):
        if lhs.name != f.name:
            raise ValueError(f"{fname}::tagnf: Tag {f.name} not found in {lhs}")
        v = rhs if delayed else ev_(rhs)
        K._assign(lhs, v, ev_)
        return NULL if delayed else v
    if not isinstance(lhs, Expr):
        raise ValueError(f"{fname}: invalid left-hand side {lhs!r}")
    pos = _tag_positions(f.name, lhs)
    if pos is None:
        raise ValueError(f"{fname}::tagnf: Tag {f.name} not found in {lhs}")
    v = rhs if delayed else ev_(rhs)
    if pos == "down":
        K._assign(lhs, v, ev_)
    else:
        _store_upvalue(f.name, _eval_lhs_args(lhs, ev_), v)
    return NULL if delayed else v


@held("TagSet", attrs={"HoldAll", "SequenceHold"},
      notes="f /: lhs = rhs: attaches the rule to f, which must be the head of lhs (a DownValue or SubValue) or "
            "an argument of lhs or an argument's head (an UpValue); otherwise TagSet::tagnf is raised")
def _h_tagset(a, ev_, expr):
    return _tagset(a, ev_, False, "TagSet")


@held("TagSetDelayed", attrs={"HoldAll", "SequenceHold"},
      notes="f /: lhs := rhs (delayed TagSet); returns Null")
def _h_tagsetdelayed(a, ev_, expr):
    return _tagset(a, ev_, True, "TagSetDelayed")


@held("TagUnset", attrs={"HoldAll"},
      notes="f /: lhs =. removes the rule for lhs attached to f (an UpValue, DownValue or SubValue); raises if "
            "there is none")
def _h_tagunset(a, ev_, expr):
    _need(a, 2, "TagUnset")
    f, lhs = a
    if not isinstance(f, Symbol):
        raise ValueError("TagUnset: the tag must be a symbol")
    lhs = _strip_holdpattern(lhs)
    if isinstance(lhs, Expr) and _tag_positions(f.name, lhs) == "up":
        lhs2 = _eval_lhs_args(lhs, ev_)
        defs = st().ext.get("UpValues", {}).get(f.name, [])
        keep = [d for d in defs if not K.same(d[0], lhs2)]
        if len(keep) == len(defs):
            raise ValueError(f"TagUnset::norep: Assignment on {f.name} for {lhs} not found")
        ext("UpValues")[f.name] = keep
        return NULL
    if isinstance(lhs, (Symbol, Expr)) and (_tag_positions(f.name, lhs) == "down" if isinstance(lhs, Expr)
                                            else lhs.name == f.name):
        return K._HELD["Unset"]((lhs,), ev_, expr)
    raise ValueError(f"TagUnset::tagnf: Tag {f.name} not found in {lhs}")


# ─────────────────────────── value lists ───────────────────────────

def _own_values(n):
    o = st().own
    return [_rd(Symbol(n), o[n])] if n in o else []


def _downs(n, sub):
    s = st()
    out = []
    for l, r in s.exact.values():
        if K._def_key(l) == n and isinstance(l, Expr) and (isinstance(l.head, Expr) == sub):
            out.append(_rd(l, r))
    for l, r, _ in s.down.get(n, ()):
        if isinstance(l, Expr) and (isinstance(l.head, Expr) == sub):
            out.append(_rd(l, r))
    return out


def _ups(n):
    return [_rd(l, r) for l, r, _ in st().ext.get("UpValues", {}).get(n, ())]


def _default_values(n):
    d = st().ext.get("Default", {}).get(n, {})
    out = []
    for key in sorted(d, key=len):
        out.append(_rd(Expr("Default", Symbol(n), *key), d[key]))
    return out


def _sym_arg(a, ev_, fname):
    _need(a, 1, fname)
    x = a[0]
    if isinstance(x, Symbol):
        return x.name
    v = ev_(x)
    if isinstance(v, str):
        return resolve_name(v, create=False)
    if isinstance(v, Symbol):
        return v.name
    raise ValueError(f"{fname}: {x!r} is not a symbol or a string")


def _values_fn(name, getter, note):
    @held(name, attrs={"HoldAll"}, notes=note)
    def h(a, ev_, expr):
        return L(*getter(_sym_arg(a, ev_, name)))
    return h


_values_fn("OwnValues", _own_values, "OwnValues[x]: {HoldPattern[x] :> value} for a symbol with a value, else {}; "
           "OwnValues[x] = {...} sets it")
_values_fn("DownValues", lambda n: _downs(n, False),
           "DownValues[f]: the rules HoldPattern[f[...]] :> rhs in the order they are tried (pattern-free "
           "definitions first, then by specificity); DownValues[f] = {...} replaces them")
_values_fn("SubValues", lambda n: _downs(n, True),
           "SubValues[f]: rules for f[...][...] (HoldPattern[lhs] :> rhs); SubValues[f] = {...} replaces them")
_values_fn("UpValues", _ups, "UpValues[f]: rules attached to f by UpSet/TagSet (HoldPattern[lhs] :> rhs); "
           "UpValues[f] = {...} replaces them")
_values_fn("DefaultValues", _default_values,
           "DefaultValues[f]: {HoldPattern[Default[f, ...]] :> value, ...} set by Default[f, ...] = value")


def _rules_of(v, fname):
    if not K.is_list(v) or not all(is_rule(r) for r in v.args):
        raise ValueError(f"{fname}: the value must be a list of rules lhs :> rhs")
    return [(_strip_holdpattern(r.args[0]), r.args[1]) for r in v.args]


def _assign_values(kind):
    def hook(lhs, rhs, ev_):
        _need(lhs.args, 1, kind)
        n = _sym_arg(lhs.args, ev_, kind)
        if is_protected(n):
            raise ValueError(f"{kind}: {n} is protected")
        rules = _rules_of(ev_(rhs), kind)
        s = st()
        if kind == "OwnValues":
            if len(rules) > 1 or (rules and not is_sym(rules[0][0], n)):
                raise ValueError(f"OwnValues[{n}] must be {{}} or {{HoldPattern[{n}] :> value}}")
            s.own.pop(n, None)
            for _l, r in rules:
                s.own[n] = r
            return NULL
        if kind in ("DownValues", "SubValues"):
            sub = kind == "SubValues"
            for k in [k for k, (l, _r) in s.exact.items() if K._def_key(l) == n and
                      isinstance(l, Expr) and isinstance(l.head, Expr) == sub]:
                del s.exact[k]
            s.down[n] = [d for d in s.down.get(n, []) if isinstance(d[0].head, Expr) != sub]
            for l, r in rules:
                if not isinstance(l, Expr) or _innermost_head(l) != n or isinstance(l.head, Expr) != sub:
                    raise ValueError(f"{kind}: {l} is not a {'sub' if sub else 'down'}value of {n}")
                K.store_definition(l, r)
            return NULL
        if kind == "UpValues":
            ext("UpValues")[n] = []
            for l, r in rules:
                if not isinstance(l, Expr) or _tag_positions(n, l) != "up":
                    raise ValueError(f"UpValues: {l} is not an upvalue of {n}")
                _store_upvalue(n, l, r)
            return NULL
        if kind == "DefaultValues":
            ext("Default")[n] = {}
            for l, r in rules:
                if not (isinstance(l, Expr) and l.head == "Default" and l.args and is_sym(l.args[0], n)):
                    raise ValueError(f"DefaultValues: {l} is not a default value of {n}")
                ext("Default")[n][tuple(l.args[1:])] = r
            return NULL
        raise AssertionError(kind)
    return hook


for _k in ("OwnValues", "DownValues", "SubValues", "UpValues", "DefaultValues"):
    K.ASSIGN_HOOKS[_k] = _assign_values(_k)


# ─────────────────────────── Default ───────────────────────────

_BUILTIN_DEFAULTS = {"Plus": {(): 0}, "Times": {(): 1}, "Power": {(2,): 1}}


@fn("Default", area=AREA, status="added",
    notes="Default[f], Default[f, i], Default[f, i, n]: the value an omitted x_. takes in f[...] (user values set "
          "by Default[f, ...] = v; Plus 0, Times 1, Power exponent 1); unevaluated when none is defined")
def _default(f: Expr, *pos: Expr) -> Expr:
    if not isinstance(f, Symbol):
        raise ValueError("Default: the first argument must be a symbol")
    if len(pos) > 2 or not all(K.is_int(p) and int(p) >= 1 for p in pos):
        raise ValueError("Default: positions must be positive integers")
    key = tuple(int(p) for p in pos)
    d = st().ext.get("Default", {}).get(f.name) or _BUILTIN_DEFAULTS.get(f.name, {})
    for k in (key, key[:1], ()):
        if k in d:
            if f.name == "Power" and k == (2,) and key[:1] != (2,):
                break
            return d[k]
    raise Unevaluated()


def _assign_default(lhs, rhs, ev_):
    if not lhs.args or not isinstance(lhs.args[0], Symbol):
        raise ValueError("Default: Default[f, ...] = value needs a symbol f")
    n = lhs.args[0].name
    pos = [ev_(p) for p in lhs.args[1:]]
    if len(pos) > 2 or not all(K.is_int(p) and int(p) >= 1 for p in pos):
        raise ValueError("Default: positions must be positive integers")
    if is_protected(n):
        raise ValueError(f"Default: {n} is protected")
    ext("Default").setdefault(n, {})[tuple(int(p) for p in pos)] = rhs if not isinstance(rhs, Expr) else ev_(rhs)
    return NULL


K.ASSIGN_HOOKS["Default"] = _assign_default
UNSET_HOOKS["Default"] = lambda lhs: (st().ext.get("Default", {}).get(
    lhs.args[0].name if lhs.args and isinstance(lhs.args[0], Symbol) else None, {}).pop(
    tuple(int(ev(p)) for p in lhs.args[1:]), None), NULL)[1]


# ─────────────────────────── argument hook: Stub heads, NonThreadable, Threaded ───────────────────────────

def _array_depth(x):
    d = 0
    while K.is_list(x) and x.args:
        lens = {len(y.args) if K.is_list(y) else -1 for y in x.args}
        d += 1
        if len(lens) != 1 or -1 in lens:
            return d
        x = x.args[0]
    return d


def _non_threadable(x):
    if isinstance(x, Symbol):
        return "NonThreadable" in st().attrs.get(x.name, ())
    if isinstance(x, Expr):
        h = _innermost_head(x)
        if h is not None and "NonThreadable" in st().attrs.get(h, ()):
            return True
        if isinstance(x.head, str) and "Listable" in K.attributes(x.head) and x.head != "List":
            return any(_non_threadable(a) for a in x.args)
    return False


def _threaded(head, args, ev_):
    ti = [i for i, a in enumerate(args) if isinstance(a, Expr) and a.head == "Threaded"]
    if len(ti) != 1:
        raise NotImplementedError(f"{head}: only one Threaded argument is supported")
    i = ti[0]
    t = args[i]
    if not 1 <= len(t.args) <= 2:
        raise ValueError("Threaded[b] or Threaded[b, alev] expected")
    b = t.args[0]
    lists = [j for j, a in enumerate(args) if j != i and K.is_list(a)]
    if not lists:
        return ev_(Expr(head, *(b if j == i else a for j, a in enumerate(args))))
    if len(lists) > 1:
        raise NotImplementedError(f"{head}: Threaded with more than one other array is not supported")
    j = lists[0]
    a = args[j]
    da, db = _array_depth(a), _array_depth(b)
    if len(t.args) == 1:
        lev = da - db
    else:
        spec = t.args[1]
        if K.is_int(spec):
            alev = int(spec)
        elif is_rule(spec) and K.is_int(spec.args[0]) and int(spec.args[0]) == 1 and K.is_int(spec.args[1]):
            alev = int(spec.args[1])
        else:
            raise NotImplementedError("Threaded[b, blev -> alev] is supported only for blev = 1")
        if alev < 0:
            alev = da + alev + 1
        if not 1 <= alev <= da + 1:
            raise ValueError(f"Threaded: level {spec} is not a level of the array")
        lev = alev - 1
    if lev < 0:
        raise ValueError("Threaded: the threaded array is deeper than the array it is combined with")

    def go(x, k):
        if k == 0:
            return Expr(head, *(x if m == j else b if m == i else args[m] for m in range(len(args))))
        if not K.is_list(x):
            raise ValueError("Threaded: the arrays are not rectangular")
        return L(*(go(y, k - 1) for y in x.args))
    return ev_(go(a, lev))


def _pre_listable(head, new_args, attrs, ev_):
    if "Listable" not in attrs and not K.state().ext:
        return UNH
    stubs = K.state().ext.get("Stub")
    if stubs and head in stubs:
        _load_stub(head)
        return ev_(Expr(head, *new_args))
    if "Listable" not in attrs:
        return UNH
    if any(isinstance(a, Expr) and a.head == "Threaded" for a in new_args):
        return _threaded(head, new_args, ev_)
    if any(K.is_list(a) for a in new_args) and any(_non_threadable(a) for a in new_args):
        lists = [a for a in new_args if K.is_list(a)]
        rest = [a for a in new_args if not K.is_list(a)]
        if len(lists) > 1:
            lists = [ev_(Expr(head, *lists))]
        if rest and head in ("Plus", "Times"):
            core = ev_(Expr(head, *rest))
            return Expr(head, core, *lists) if not (isinstance(core, Expr) and core.head == head) else \
                Expr(head, *core.args, *lists)
        return Expr(head, *new_args)
    return UNH


def _install_arg_hook():
    E.PRE_LISTABLE_HOOK = _pre_listable


_install_arg_hook()


_added("Threaded", "f[a, Threaded[b]] for Listable f combines b with the lowest-level subarrays of a; "
       "Threaded[b, alev] and Threaded[b, 1 -> alev] choose the level (other blev raise)")
INERT["Threaded"] = "inert wrapper"


# ─────────────────────────── options ───────────────────────────

def _opt_key(k):
    return k.name if isinstance(k, Symbol) else k if isinstance(k, str) else None


def _flat_opts(xs):
    out = []
    for x in xs:
        if K.is_list(x):
            out.extend(_flat_opts(x.args))
        elif is_rule(x):
            out.append(x)
        else:
            raise ValueError(f"{x!r} is not a rule")
    return out


def _options_of(name):
    return list(st().ext.get("Options", {}).get(name, []))


def _known_options(name):
    o = st().ext.get("Options", {})
    if name in o:
        return o[name]
    if is_system(name):
        raise NotImplementedError(f"Options[{name}]: the option lists of built-in functions are not catalogued")
    s = st()
    if name[:1].isupper() and "`" not in name and name not in s.own and name not in s.down and \
            name not in s.attrs and not any(K._def_key(l) == name for l, _ in s.exact.values()):
        # a capitalised name with no user definitions may be a Wolfram built-in MikoshiLang lacks: do not
        # claim it has no options
        raise NotImplementedError(f"Options[{name}]: {name} has no user definitions and built-in option lists "
                                  "are not catalogued")
    return []


def _select_opts(rules, keys):
    ks = {_opt_key(k) for k in keys}
    return [r for r in rules if _opt_key(r.args[0]) in ks]


@fn("Options", area=AREA, status="added",
    notes="Options[f] gives the default option rules of a user symbol (set by Options[f] = {...} or "
          "SetOptions); Options[f, name] or Options[f, {names}] selects; Options[expr] gives the options at the end "
          "of expr (f[x, a -> 1] gives {a -> 1}). Built-in option lists are not catalogued and raise")
def _options(x, *names_):
    if len(names_) > 1:
        raise ValueError("Options: 1 or 2 arguments expected")
    if isinstance(x, Symbol):
        rules = _known_options(x.name)
    elif isinstance(x, str):
        rules = _known_options(resolve_name(x, create=False))
    elif isinstance(x, Expr):
        args = list(x.args)
        rules = []
        while args and K._is_option(args[-1]):
            rules = _flat_opts([args.pop()]) + rules
    else:
        raise ValueError(f"Options: {x!r} is not a symbol or an expression")
    if names_:
        n = names_[0]
        keys = n.args if K.is_list(n) else (n,)
        return L(*_select_opts(rules, keys))
    return L(*rules)


def _assign_options(lhs, rhs, ev_):
    if len(lhs.args) != 1:
        raise ValueError("Options[f] = {...} needs one symbol")
    n = _sym_arg(lhs.args, ev_, "Options")
    if is_protected(n):
        raise ValueError(f"Options: {n} is protected")
    v = ev_(rhs)
    rules = _flat_opts([v])
    ext("Options")[n] = rules
    return L(*rules)


K.ASSIGN_HOOKS["Options"] = _assign_options
UNSET_HOOKS["Options"] = lambda lhs: (st().ext.get("Options", {}).pop(
    lhs.args[0].name if lhs.args and isinstance(lhs.args[0], Symbol) else None, None), NULL)[1]


@fn("SetOptions", area=AREA, status="added",
    notes="SetOptions[f, name -> value, ...] resets defaults that f already has (SetOptions::optnf for an unknown "
          "option), returning the new Options[f]; SetOptions[{f, g}, ...] sets several")
def _setoptions(f: Expr, *rules: Expr) -> ListOf[Expr]:
    if K.is_list(f):
        return L(*(_setoptions(g, *rules) for g in f.args))
    if not isinstance(f, Symbol):
        raise ValueError("SetOptions: the first argument must be a symbol")
    n = f.name
    if is_protected(n) and is_system(n):
        raise NotImplementedError(f"SetOptions[{n}, ...]: built-in options are not catalogued")
    if is_protected(n):
        raise ValueError(f"SetOptions: {n} is protected")
    cur = list(_known_options(n))
    for r in _flat_opts(rules):
        k = _opt_key(r.args[0])
        for i, c in enumerate(cur):
            if _opt_key(c.args[0]) == k:
                cur[i] = Expr(c.head, c.args[0], r.args[1])
                break
        else:
            raise ValueError(f"SetOptions::optnf: {r.args[0]} is not a known option for {n}")
    ext("Options")[n] = cur
    return L(*cur)


def _lookup(rules, key):
    k = _opt_key(key)
    for r in rules:
        if _opt_key(r.args[0]) == k:
            return True, r.args[1]
    return False, None


def _option_value(fs, opts, name):
    """OptionValue[f, opts, name]: explicit opts first, then the defaults of the functions fs."""
    if K.is_list(name):
        return L(*(_option_value(fs, opts, n) for n in name.args))
    if _opt_key(name) is None:
        raise ValueError(f"OptionValue: {name!r} is not an option name")
    found, v = _lookup(opts, name)
    if found:
        return v
    for f in fs:
        found, v = _lookup(_known_options(f), name)
        if found:
            return v
    raise ValueError(f"OptionValue::optnf: {name} is not a known option for {', '.join(fs) or 'the function'}")


def _fs(f):
    if isinstance(f, Symbol):
        return [f.name]
    if K.is_list(f) and all(isinstance(g, Symbol) for g in f.args):
        return [g.name for g in f.args]
    raise ValueError(f"OptionValue: {f!r} is not a symbol or a list of symbols")


@fn("OptionValue", area=AREA, status="added",
    notes="OptionValue[name] inside f[..., OptionsPattern[]] := rhs gives the value of option name in the call "
          "(or f's default); OptionValue[f, name], OptionValue[f, opts, name], lists of names; "
          "OptionsPattern[{f, g}] searches several defaults")
def _optionvalue(*a: Expr) -> Expr:
    if len(a) == 2:
        return _option_value(_fs(a[0]), [], a[1])
    if len(a) == 3:
        return _option_value(_fs(a[0]), _flat_opts([a[1]]), a[2])
    if len(a) == 1:
        raise ValueError("OptionValue[name] can only be used in a definition with OptionsPattern[]")
    raise ValueError("OptionValue: 1 to 3 arguments expected")


def _find_options_patterns(lhs, out):
    if isinstance(lhs, Expr):
        if lhs.head == "OptionsPattern":
            out.append((None, lhs))
            return
        if lhs.head == "Pattern" and len(lhs.args) == 2 and isinstance(lhs.args[1], Expr) and \
                lhs.args[1].head == "OptionsPattern":
            out.append((lhs.args[0].name, lhs.args[1]))
            return
        if isinstance(lhs.head, Expr):
            _find_options_patterns(lhs.head, out)
        for x in lhs.args:
            _find_options_patterns(x, out)


_OV_CACHE = {}


def _option_value_hook(lhs, rhs, b, expr):
    if not isinstance(rhs, Expr) or not K.contains(rhs, lambda x: isinstance(x, Expr) and x.head == "OptionValue"):
        return rhs
    key = id(lhs)
    pats = _OV_CACHE.get(key)
    if pats is None or pats[0] is not lhs:
        found = []
        _find_options_patterns(lhs, found)
        pats = _OV_CACHE[key] = (lhs, found)
    found = pats[1]
    if not found:
        return rhs
    name, op = found[0]
    bound = b.get(name or "\x00opts", ())
    opts = _flat_opts(bound if isinstance(bound, tuple) else (bound,))
    fsym = L(op.args[0]) if op.args and isinstance(op.args[0], Symbol) else (
        op.args[0] if op.args else L(Symbol(_innermost_head(lhs))))
    optlist = L(*opts)

    def rw(t):
        if isinstance(t, Expr):
            if t.head == "OptionValue" and len(t.args) == 1:
                return Expr("OptionValue", fsym, optlist, rw(t.args[0]))
            if t.head == "OptionValue" and len(t.args) == 2 and isinstance(t.args[0], Symbol) and \
                    not K.is_list(t.args[1]) and False:
                return t
            return Expr(rw(t.head) if isinstance(t.head, Expr) else t.head, *(rw(x) for x in t.args))
        return t
    return rw(rhs)


K.OPTION_VALUE_HOOK = _option_value_hook

_added("OptionsPattern", "OptionsPattern[] matches a sequence of zero or more rules or lists of rules; "
       "OptionsPattern[f] or OptionsPattern[{f, ...}] names where OptionValue finds defaults")


@fn("FilterRules", area=AREA, status="added",
    notes="FilterRules[rules, patt] keeps rules whose lhs matches patt; FilterRules[rules, {k1, ...}] keeps the "
          "named options; FilterRules[rules, Except[...]] removes them; FilterRules[rules, Options[f]] keeps f's")
def _filterrules(rules: Expr, patt: Expr) -> ListOf[Expr]:
    rs = _flat_opts([rules])

    def keep_pred(p):
        if isinstance(p, Expr) and p.head == "Except" and len(p.args) == 1:
            inner = keep_pred(p.args[0])
            return lambda k: not inner(k)
        if K.is_list(p):
            preds = []
            for x in p.args:
                preds.append(keep_pred(x.args[0] if is_rule(x) else x))
            return lambda k: any(q(k) for q in preds)
        return lambda k: any(True for _ in K.match(p, k, {}, ev)) or (
            isinstance(p, (Symbol, str)) and _opt_key(p) == _opt_key(k))
    pred = keep_pred(patt)
    return L(*(r for r in rs if pred(r.args[0])))


# ─────────────────────────── ValueQ, Remove, Unique ───────────────────────────

def _has_value(e):
    s = st()
    if isinstance(e, Symbol):
        return e.name in s.own or e.name in E.SYSTEM_VARIABLES or e.name in s.ext.get("Stub", ())
    return False


@held("ValueQ", attrs={"HoldAll"},
      notes="ValueQ[expr] is True when expr has a value: a symbol with an OwnValue, or an expression changed by "
            "evaluation (Method -> \"Legacy\" default compares the result; \"SymbolDefinitionsPresent\" and "
            "\"TrialEvaluation\" also accepted)")
def _h_valueq(a, ev_, expr):
    _need(a, 1, "ValueQ", 2)
    method = "Legacy"
    if len(a) == 2:
        o = ev_(a[1])
        if not (is_rule(o) and is_sym(o.args[0], "Method") and isinstance(o.args[1], str)):
            raise ValueError("ValueQ: the second argument must be Method -> \"name\"")
        method = o.args[1]
        if method not in ("Legacy", "SymbolDefinitionsPresent", "TrialEvaluation", "Automatic"):
            raise ValueError(f"ValueQ: unknown method {method}")
    x = a[0]
    if isinstance(x, Symbol):
        return _has_value(x)
    if not isinstance(x, Expr):
        return False
    if method == "SymbolDefinitionsPresent":
        h = _innermost_head(x)
        s = st()
        return bool(h and (s.down.get(h) or any(K._def_key(l) == h for l, _ in s.exact.values()) or
                           s.ext.get("UpValues", {}).get(h)))
    try:
        r = ev_(x)
    except Exception:
        return True
    return not K.same(r, x)


@held("Remove", attrs={"HoldAll"},
      notes="Remove[s1, ...] removes symbols completely (values, attributes, options, the name: Names no longer "
            "lists it); Remove[\"ctx`*\"] removes by pattern; protected symbols raise Remove::rmptc")
def _h_remove(a, ev_, expr):
    tgt = []
    for x in a:
        tgt.extend(_target_names(x, ev_, "Remove"))
    for n in tgt:
        if is_protected(n) or is_system(n):
            raise ValueError(f"Remove::rmptc: Symbol {n} is Protected and cannot be removed")
    for n in tgt:
        _check_unlocked(n, "Remove")
        K.clear_symbol(n, all_=True)
        d = _ctxdata()
        f = full_name(n)
        d["symbols"].discard(f)
        if _short(f)[0] == "Global`":
            d["globals"].discard(_short(f)[1])
        d["removed"].add(n)
    return NULL


_UNIQUE_N = {}


@held("Unique", attrs={"HoldAll"},
      notes="Unique[] gives $nnn, Unique[x] gives x$nnn (using $ModuleNumber, then incrementing it), "
            "Unique[\"xxx\"] gives xxxn with n counting up from 1 per prefix, Unique[{x, y}] gives a list; "
            "Unique[x, {attrs}] sets attributes")
def _h_unique(a, ev_, expr):
    _need(a, 0, "Unique", 2)
    s = st()

    def mk(x):
        if isinstance(x, Symbol):
            s.counter += 1
            return Symbol(f"{x.name}${s.counter}")
        if isinstance(x, str):
            if not re.match(r"^[A-Za-z$][A-Za-z0-9$]*$", x):
                raise ValueError(f"Unique: {x!r} is not a valid symbol name")
            n = _UNIQUE_N.get(x, 0)
            while True:
                n += 1
                cand = f"{x}{n}"
                if cand not in s.own and cand not in s.down and not is_system(cand) and \
                        cand not in _ctxdata()["globals"]:
                    break
            _UNIQUE_N[x] = n
            return Symbol(cand)
        raise ValueError(f"Unique: {x!r} is not a symbol or a string")
    if not a:
        s.counter += 1
        return Symbol(f"${s.counter}")
    x = a[0]
    if not isinstance(x, (Symbol, str)) and not K.is_list(x):
        x = ev_(x)
    out = L(*(mk(y) for y in x.args)) if K.is_list(x) else mk(x)
    if len(a) == 2:
        attrs = _attr_names(ev_(a[1]))
        for y in (out.args if K.is_list(out) else (out,)):
            s.attrs.setdefault(y.name, set()).update(attrs)
            if "Temporary" in attrs:
                s.temporaries.add(y.name)
    return out


# ─────────────────────────── Definition, FullDefinition, Information ───────────────────────────

_KIND = [None]          # "=" or ":=" while Set / SetDelayed is assigning


def _kinds():
    return ext("\x00kind")


def _wrap_set(name, kind):
    orig = K._HELD[name]

    def h(a, ev_, expr):
        old = _KIND[0]
        _KIND[0] = kind
        try:
            return orig(a, ev_, expr)
        finally:
            _KIND[0] = old
    K._HELD[name] = h


_wrap_set("Set", "=")
_wrap_set("SetDelayed", ":=")
_orig_store = K.store_definition


def _store_definition(lhs, rhs):
    if _KIND[0] is not None:
        _kinds()[K.tag(lhs)] = _KIND[0]
    return _orig_store(lhs, rhs)


K.store_definition = _store_definition
_orig_assign2 = K._assign


def _assign_kind(lhs, rhs, ev_):
    if isinstance(lhs, Symbol) and _KIND[0] is not None:
        _kinds()[("own", lhs.name)] = _KIND[0]
    return _orig_assign2(lhs, rhs, ev_)


K._assign = _assign_kind


def _kind_of(key, lhs):
    k = st().ext.get("\x00kind", {}).get(key)
    if k is not None:
        return k
    return ":=" if K.has_pattern(lhs) else "="


def _fmt(x):
    from ..formatting import input_form
    return input_form(x)


def _definition_lines(n):
    s = st()
    attrs = sorted(full_attributes(n))
    lines = []
    if attrs:
        lines.append(f"Attributes[{_display(n)}] = {{{', '.join(attrs)}}}")
    if "ReadProtected" in attrs:
        return lines
    if n in s.own:
        lines.append(f"{_display(n)} {_kind_of(('own', n), Symbol(n))} {_fmt(s.own[n])}")
    for sub in (False, True):
        for l, r in s.exact.values():
            if K._def_key(l) == n and isinstance(l, Expr) and isinstance(l.head, Expr) == sub:
                lines.append(f"{_fmt(l)} {_kind_of(K.tag(l), l)} {_fmt(r)}")
        for l, r, _ in s.down.get(n, ()):
            if isinstance(l, Expr) and isinstance(l.head, Expr) == sub:
                lines.append(f"{_fmt(l)} {_kind_of(K.tag(l), l)} {_fmt(r)}")
    for l, r, _ in s.ext.get("UpValues", {}).get(n, ()):
        op = "^:=" if K.has_pattern(l) else "^="
        lines.append(f"{_fmt(l)} {op} {_fmt(r)}")
    for k, v in sorted(s.ext.get("Default", {}).get(n, {}).items(), key=lambda kv: len(kv[0])):
        lines.append(f"Default[{', '.join([_display(n)] + [str(i) for i in k])}] = {_fmt(v)}")
    if s.ext.get("Options", {}).get(n):
        lines.append(f"Options[{_display(n)}] = {_fmt(L(*s.ext['Options'][n]))}")
    return lines


def _display(n):
    return _display_name(full_name(n))


def _deps(n, seen, excluded):
    """Symbols that the definitions of n mention, recursively (for FullDefinition)."""
    out = []
    stack = [n]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        out.append(m)
        found = set()
        for e in st().definitions_of(m):
            _collect_symbols(e, found)
        for d in st().ext.get("UpValues", {}).get(m, ()):
            _collect_symbols(d[0], found)
            _collect_symbols(d[1], found)
        for f in sorted(found):
            if f in seen or is_system(f) or K.has_pattern(Symbol(f)):
                continue
            if any(_context_of_full(full_name(f)) == c for c in excluded):
                continue
            stack.append(f)
    return out


def _def_names(x, ev_, fname):
    if isinstance(x, Symbol):
        return [x.name]
    v = ev_(x)
    if isinstance(v, str):
        return [resolve_name(v, create=False)] if not any(c in v for c in "*@") else \
            [resolve_name(m, create=False) for m in names(v)]
    if isinstance(v, Symbol):
        return [v.name]
    raise ValueError(f"{fname}: {x!r} is not a symbol or a string")


@held("Definition", attrs={"HoldAll"},
      notes="Definition[f] stays as a held object that prints the definitions of f (Attributes, values with = or "
            ":=, upvalues with ^= / ^:=, Default, Options), separated by blank lines as in Wolfram; ToString gives "
            "the text; ReadProtected hides everything but the attributes")
def _h_definition(a, ev_, expr):
    _need(a, 1, "Definition", -1)
    for x in a:
        if not isinstance(x, Symbol):
            _def_names(x, ev_, "Definition")
    return expr


@held("FullDefinition", attrs={"HoldAll"},
      notes="FullDefinition[f] prints the definitions of f and of every symbol they refer to, recursively "
            "(System` symbols excluded); options ExcludedContexts and IncludedContexts")
def _h_fulldefinition(a, ev_, expr):
    _need(a, 1, "FullDefinition", -1)
    for x in a:
        if is_rule(x):
            k = x.args[0]
            if not (is_sym(k, "ExcludedContexts") or is_sym(k, "IncludedContexts")):
                raise ValueError(f"FullDefinition: unknown option {k}")
    return expr


def _definition_text(e, full):
    syms = []
    excluded, included = ["System`"], None
    for x in e.args:
        if is_rule(x):
            v = ev(x.args[1])
            lst = [c for c in (v.args if K.is_list(v) else (v,))]
            if is_sym(x.args[0], "ExcludedContexts"):
                excluded = ["System`"] + [c if c.endswith("`") else c + "`" for c in lst if isinstance(c, str)]
            else:
                included = [c if c.endswith("`") else c + "`" for c in lst if isinstance(c, str)]
            continue
        syms.extend(_def_names(x, ev, "Definition"))
    if full:
        seen, allsyms = set(), []
        for n in syms:
            allsyms.extend(_deps(n, seen, excluded))
        if included is not None:
            allsyms = [n for n in allsyms if n in syms or _context_of_full(full_name(n)) in included]
        syms = allsyms
    blocks = []
    for n in syms:
        ls = _definition_lines(n)
        if ls:
            blocks.append("\n \n".join(ls))
    return "\n \n".join(blocks) if blocks else "Null"


def _info_assoc(n, long_form=True):
    s = st()

    def vals(lst):
        return L(*lst) if lst else Symbol("None")
    full = full_name(n)
    items = [("ObjectType", "Symbol"),
             ("Usage", Expr("Missing", "NotAvailable")),
             ("Documentation", Symbol("None") if not is_system(n) else
              "https://reference.wolfram.com/language/ref/" + n + ".html"),
             ("OwnValues", vals(_own_values(n) if "ReadProtected" not in full_attributes(n) else [])),
             ("UpValues", vals(_ups(n))),
             ("DownValues", vals(_downs(n, False) if "ReadProtected" not in full_attributes(n) else [])),
             ("SubValues", vals(_downs(n, True))),
             ("DefaultValues", vals(_default_values(n))),
             ("NValues", Symbol("None")),
             ("FormatValues", Symbol("None")),
             ("Options", vals(s.ext.get("Options", {}).get(n, []))),
             ("Attributes", L(*(Symbol(x) for x in sorted(full_attributes(n))))),
             ("FullName", full)]
    if not long_form:
        items = [it for it in items if it[0] in ("ObjectType", "Usage", "Documentation", "FullName")]
    return Expr("Association", *(Expr("Rule", k, v) for k, v in items))


@held("Information", attrs={"HoldAll"},
      notes="Information[symbol] (also ?f, ??f) gives InformationData[<|ObjectType, Usage, Documentation, "
            "OwnValues, UpValues, DownValues, SubValues, DefaultValues, NValues, FormatValues, Options, Attributes, "
            "FullName|>]; Information[s, \"prop\"] gives one property; Information[\"patt*\"] a list; LongForm -> "
            "False keeps only the summary. Usage messages (f::usage) are not available")
def _h_information(a, ev_, expr):
    _need(a, 1, "Information", 3)
    long_form = True
    prop = None
    for x in a[1:]:
        v = ev_(x)
        if is_rule(v) and is_sym(v.args[0], "LongForm"):
            long_form = v.args[1] is True
        elif isinstance(v, str) or K.is_list(v):
            prop = v
        else:
            raise ValueError(f"Information: invalid argument {v!r}")
    x = a[0]
    if isinstance(x, Symbol):
        ns = [x.name]
        single = True
    else:
        v = ev_(x)
        if isinstance(v, str):
            single = not any(c in v for c in "*@")
            ns = [resolve_name(v, create=False)] if single else [resolve_name(m, create=False) for m in names(v)]
            if single and not _exists(ns[0]):
                raise ValueError(f"Information::notfound: Symbol {v} not found")
        elif isinstance(v, Symbol):
            ns, single = [v.name], True
        else:
            raise NotImplementedError("Information: only symbols and names are supported")
    out = []
    for n in ns:
        d = _info_assoc(n, long_form)
        if prop is not None:
            keys = prop.args if K.is_list(prop) else (prop,)
            got = []
            for k in keys:
                for r in d.args:
                    if r.args[0] == k:
                        got.append(r.args[1])
                        break
                else:
                    got.append(Expr("Missing", "UnknownProperty", k))
            out.append(L(*got) if K.is_list(prop) else got[0])
        else:
            out.append(Expr("InformationData", d, long_form is False))
    return out[0] if single else L(*out)


def _exists(n):
    return is_system(n) or full_name(n) in all_symbols()


@fn("InformationData", area=AREA, status="added",
    notes="InformationData[assoc, False][\"prop\"] gives a property of an Information result")
def _informationdata(*a: Expr) -> Expr:
    raise Unevaluated()


def _infodata_call(head, args, ev_):
    if isinstance(head, Expr) and head.head == "InformationData" and head.args and len(args) == 1:
        d = head.args[0]
        for r in d.args:
            if r.args[0] == args[0]:
                return r.args[1]
        return Expr("Missing", "UnknownProperty", args[0])
    return UNH


_added("ExcludedContexts", "FullDefinition option: contexts whose symbols are not followed (System` always)")
_added("IncludedContexts", "FullDefinition option: only symbols in these contexts are followed")


# ─────────────────────────── evaluation control ───────────────────────────

import importlib as _il
KL = _il.import_module("mikoshilang.kernel_lists")


@held("Defer", attrs={"HoldAll"},
      notes="Defer[expr] keeps expr unevaluated and prints as expr itself (InputForm/ToString show the contents); "
            "evaluating the printed form later evaluates it")
def _h_defer(a, ev_, expr):
    return expr


@held("HoldCompleteForm", attrs={"HoldAllComplete"},
      notes="HoldCompleteForm[expr] keeps expr completely unevaluated (no Sequence splicing, no upvalues, "
            "Unevaluated kept) and prints as expr without the wrapper")
def _h_holdcompleteform(a, ev_, expr):
    return expr


_added("Unevaluated", "Unevaluated[expr] passes expr unevaluated as an argument of a function; the wrapper is "
       "removed when the function uses it, kept when the function returns the expression unchanged "
       "(Length[Unevaluated[1 + 2]] = 2)")


@held("Inactive", attrs={"HoldFirst"},
      notes="Inactive[f][args] is an inert form of f[args] (the arguments are evaluated, f is not applied); "
            "Inactive[atom] gives atom for atoms other than symbols")
def _h_inactive(a, ev_, expr):
    _need(a, 1, "Inactive")
    x = a[0]
    if not isinstance(x, (Symbol, Expr)):
        return x
    return expr


_INACTIVATE_EXCLUDE = {"List", "Rule", "RuleDelayed", "Blank", "BlankSequence", "BlankNullSequence", "Pattern",
                       "Optional", "Condition", "PatternTest", "Alternatives", "Repeated", "RepeatedNull",
                       "HoldPattern", "Verbatim", "Inactive", "Association", "Slot", "SlotSequence",
                       "OptionsPattern", "Except", "Hold", "HoldForm", "HoldComplete", "Unevaluated", "Complex",
                       "Rational", "Sequence", "Function", "Evaluate", "CompoundExpression", "Set", "SetDelayed",
                       "Missing", "Inactivate", "Activate", "Defer"}


def _heads_opt(opts, fname):
    heads = True
    for o in opts:
        if is_rule(o) and is_sym(o.args[0], "Heads") and isinstance(o.args[1], bool):
            heads = o.args[1]
        else:
            raise ValueError(f"{fname}: unknown option {o!r}")
    return heads


def _patt_pred(patt):
    if patt is None:
        return lambda name: True
    return lambda name: any(True for _ in K.match(patt, Symbol(name), {}, ev))


def _inactivate(t, pred, heads):
    if isinstance(t, Expr):
        h = t.head
        args = tuple(_inactivate(x, pred, heads) for x in t.args)
        if isinstance(h, str):
            if h not in _INACTIVATE_EXCLUDE and pred(h):
                h = Expr("Inactive", Symbol(h))
        elif heads:
            h = _inactivate(h, pred, heads)
        return Expr(h, *args)
    return t


@held("Inactivate", attrs={"HoldFirst"},
      notes="Inactivate[expr] replaces every head f (except List, Rule, Blank and other structural heads) by "
            "Inactive[f] before evaluation; Inactivate[expr, patt] only heads matching patt (a symbol, "
            "Alternatives, Except[...]); Heads -> False leaves compound heads alone")
def _h_inactivate(a, ev_, expr):
    _need(a, 1, "Inactivate", -1)
    rest = [ev_(x) for x in a[1:]]
    patt = None
    if rest and not is_rule(rest[0]):
        patt = rest.pop(0)
    heads = _heads_opt(rest, "Inactivate")
    return ev_(_inactivate(a[0], _patt_pred(patt), heads))


def _activate(t, pred, heads):
    if isinstance(t, Expr):
        h = t.head
        if isinstance(h, Expr) and h.head == "Inactive" and len(h.args) == 1:
            inner = h.args[0]
            name = inner.name if isinstance(inner, Symbol) else None
            if name is not None and pred(name):
                h = inner.name
            elif not isinstance(inner, Symbol):
                h = Expr("Inactive", _activate(inner, pred, heads)) if heads else h
        elif isinstance(h, Expr) and heads:
            h = _activate(h, pred, heads)
        return Expr(h, *(_activate(x, pred, heads) for x in t.args))
    return t


@fn("Activate", area=AREA, status="added",
    notes="Activate[expr] replaces Inactive[f] by f everywhere in expr and evaluates; Activate[expr, patt] only "
          "for f matching patt; Heads -> False does not enter heads")
def _activate_fn(e: Expr, *rest: Expr) -> Expr:
    rest = list(rest)
    patt = None
    if rest and not is_rule(rest[0]):
        patt = rest.pop(0)
    heads = _heads_opt(rest, "Activate")
    return ev(_activate(e, _patt_pred(patt), heads))


_added("Heads", "option Heads -> True/False of Inactivate, Activate (and the level functions)")


def _strip_inactive(t):
    if isinstance(t, Expr):
        h = t.head
        if isinstance(h, Expr) and h.head == "Inactive" and len(h.args) == 1:
            h = h.args[0].name if isinstance(h.args[0], Symbol) else _strip_inactive(h.args[0])
        elif isinstance(h, Expr):
            h = _strip_inactive(h)
        if h == "Inactive" and len(t.args) == 1:
            return _strip_inactive(t.args[0])
        return Expr(h, *(_strip_inactive(x) for x in t.args))
    return t


def _ignoring_inactive(pat, expr, b, ev_):
    if len(pat.args) != 1:
        return
    yield from K.match(_strip_inactive(pat.args[0]), _strip_inactive(expr), b, ev_)


K.PATTERN_HOOKS["IgnoringInactive"] = _ignoring_inactive
_added("IgnoringInactive", "IgnoringInactive[patt] matches as patt with Inactive removed from both the pattern "
       "and the expression (Cases[{Sin[m], Inactive[Sin][p]}, IgnoringInactive[Sin[a_]]] gives both)")
K._PATTERN_HEADS.update({"IgnoringInactive", "OptionsPattern"})


@held("Until", attrs={"HoldAll"},
      notes="Until[test, body] evaluates body then test, repeating until test is True (at least once); Until[test] "
            "has an empty body; Break[] exits, Continue[] skips to the test; returns Null")
def _h_until(a, ev_, expr):
    _need(a, 1, "Until", 2)
    for _ in range(K.MAX_ITERATIONS):
        if len(a) == 2:
            try:
                ev_(a[1])
            except K.BreakSignal:
                return NULL
            except K.ContinueSignal:
                pass
        if ev_(a[0]) is True:
            return NULL
    raise RecursionError(f"Until exceeded {K.MAX_ITERATIONS} iterations")


class GotoSignal(ValueError):
    def __init__(self, tag):
        super().__init__(f"Goto::nolabel: Label {tag} not found")
        self.tag = tag


@fn("Goto", area=AREA, status="added",
    notes="Goto[tag] continues evaluation after Label[tag] in the enclosing compound expression (the innermost "
          "one containing that label, searching outwards); a missing label raises Goto::nolabel")
def _goto(tag: Expr) -> Expr:
    raise GotoSignal(tag)


@held("Label", attrs={"HoldFirst"}, notes="Label[tag] marks a point in a compound expression for Goto[tag]; "
                                          "evaluates to Null")
def _h_label(a, ev_, expr):
    _need(a, 1, "Label")
    return NULL


_orig_compound = K._HELD["CompoundExpression"]


def _h_compound(a, ev_, expr):
    if not any(isinstance(x, Expr) and x.head == "Label" for x in a):
        return _orig_compound(a, ev_, expr)
    i, result, n = 0, None, 0
    while i < len(a):
        try:
            result = ev_(a[i])
            i += 1
        except GotoSignal as g:
            for j, x in enumerate(a):
                if isinstance(x, Expr) and x.head == "Label" and len(x.args) == 1 and \
                        (K.same(x.args[0], g.tag) or K.same(ev_(x.args[0]), g.tag)):
                    i = j + 1
                    result = None
                    break
            else:
                raise
            n += 1
            if n > K.MAX_ITERATIONS:
                raise RecursionError(f"Goto exceeded {K.MAX_ITERATIONS} jumps")
    return result


K._HELD["CompoundExpression"] = _h_compound


@fn("Update", area=AREA, status="added",
    notes="Update[s] / Update[] tell the system that values depending on s may have changed; MikoshiLang keeps "
          "no evaluation caches that could go stale, so this returns Null")
def _update(*s: Expr) -> Expr:
    if len(s) > 1 or (s and not isinstance(s[0], Symbol)):
        raise ValueError("Update expects Update[] or Update[symbol]")
    return NULL


@held("ApplyTo", attrs={"HoldFirst"},
      notes="x //= f sets x to f[x] and returns the new value (like x = f[x]); x must have a value")
def _h_applyto(a, ev_, expr):
    _need(a, 2, "ApplyTo")
    x = a[0]
    cur = ev_(x)
    if K.same(cur, x) and isinstance(x, Symbol):
        raise ValueError(f"ApplyTo: {x} has no value")
    v = ev_(Expr(K.as_head(ev_(a[1])), cur))
    K._assign(x, v, ev_)
    return v


# ─────────────────────────── functional forms ───────────────────────────

def _curry_spec(spec, fname, default):
    """(k, order, opts) for an arity specification."""
    if spec is None:
        return default
    if K.is_int(spec):
        n = int(spec)
        if n < 0:
            raise ValueError(f"{fname}: the arity must be a non-negative integer")
        return n, list(range(1, n + 1)), []
    if is_rule(spec) and K.is_int(spec.args[0]):
        k = int(spec.args[0])
        _, order, opts = _curry_spec(spec.args[1], fname, default)
        if any(i > k for i in order):
            raise ValueError(f"{fname}: indices larger than the arity {k}")
        return k, order, opts
    if K.is_list(spec):
        order = [x for x in spec.args if not is_rule(x)]
        opts = [x for x in spec.args if is_rule(x)]
        if not all(K.is_int(i) and int(i) >= 1 for i in order):
            raise ValueError(f"{fname}: the indices must be positive integers")
        order = [int(i) for i in order]
        return (max(order) if order else 0), order, opts
    raise ValueError(f"{fname}: invalid arity specification {spec!r}")


def _collect(head, args, name):
    groups = [list(args)]
    h = head
    while isinstance(h, Expr) and not (h.head == name):
        groups.append(list(h.args))
        h = h.head
    return h, [x for g in reversed(groups) for x in g]


def _curry_apply(name, default):
    def hook(head, args, ev_):
        base, xs = _collect(head, args, name)
        if not isinstance(base, Expr) or base.head != name or not base.args:
            return UNH
        if name == "CurryApplied" and len(base.args) == 1:
            return UNH
        f = base.args[0]
        spec = base.args[1] if len(base.args) > 1 else None
        k, order, opts = _curry_spec(spec, name, default)
        if len(xs) < k:
            return UNH
        r = Expr(K.as_head(f), *(xs[i - 1] for i in order), *opts)
        if len(xs) > k:
            r = Expr(r, *xs[k:])
        return ev_(r)
    return hook


for _nm, _d in (("OperatorApplied", (2, [2, 1], [])), ("CurryApplied", None)):
    KL.COMPOUND_HEAD_HOOKS[_nm] = _curry_apply(_nm, _d)
    KL.DEEP_HEAD_HOOKS[_nm] = _curry_apply(_nm, _d)


@fn("OperatorApplied", area=AREA, status="added",
    notes="OperatorApplied[f][y][x] = f[x, y]; OperatorApplied[f, n][x1]...[xn] = f[x1, ..., xn]; "
          "OperatorApplied[f, {i1, ...}] and k -> {i1, ...} reorder; bracket structure is irrelevant")
def _operatorapplied(f: Expr, spec: Expr = None) -> Expr:
    _curry_spec(spec, "OperatorApplied", (2, [2, 1], []))
    raise Unevaluated()


@fn("CurryApplied", area=AREA, status="added",
    notes="CurryApplied[f, n][x1]...[xn] = f[x1, ..., xn]; CurryApplied[n][f] = CurryApplied[f, n]; "
          "CurryApplied[f, {i1, ...}] and k -> {i1, ...} reorder")
def _curryapplied(f: Expr, spec: Expr = None) -> Expr:
    if spec is None:
        if not (K.is_int(f) or K.is_list(f) or (is_rule(f) and K.is_int(f.args[0]))):
            raise ValueError("CurryApplied[f] needs an arity: CurryApplied[f, n] or CurryApplied[n][f]")
        _curry_spec(f, "CurryApplied", None)
        raise Unevaluated()
    _curry_spec(spec, "CurryApplied", None)
    raise Unevaluated()


def _curry_op(head, args, ev_):
    if len(head.args) == 1 and len(args) == 1:
        return ev_(Expr("CurryApplied", args[0], head.args[0]))
    return UNH


_prev_curry = KL.COMPOUND_HEAD_HOOKS["CurryApplied"]
KL.COMPOUND_HEAD_HOOKS["CurryApplied"] = lambda h, a, e: _curry_op(h, a, e) if len(h.args) == 1 else \
    _prev_curry(h, a, e)

_INVERSES = {"Sin": "ArcSin", "Cos": "ArcCos", "Tan": "ArcTan", "Cot": "ArcCot", "Sec": "ArcSec", "Csc": "ArcCsc",
             "Sinh": "ArcSinh", "Cosh": "ArcCosh", "Tanh": "ArcTanh", "Coth": "ArcCoth", "Sech": "ArcSech",
             "Csch": "ArcCsch", "Exp": "Log", "Log": "Exp", "Identity": "Identity", "Gudermannian":
             "InverseGudermannian", "Erf": "InverseErf", "Erfc": "InverseErfc", "Haversine": "InverseHaversine"}
for _k, _v in list(_INVERSES.items()):
    _INVERSES.setdefault(_v, _k)
_INVERSES["Log"] = "Exp"
_INVERSES["Exp"] = "Log"


def _pure_inverse(f, n, tot):
    from .core import S, R
    body = f.args[0] if len(f.args) == 1 else None
    if body is None:
        if len(f.args) == 2 and isinstance(f.args[0], Symbol):
            vars_ = [f.args[0]]
        elif len(f.args) == 2 and K.is_list(f.args[0]):
            vars_ = list(f.args[0].args)
        else:
            return None
        body = f.args[1]
        slots = {v.name: Expr("Slot", i + 1) for i, v in enumerate(vars_)}
        body = K.substitute(body, slots)
    names_ = [Symbol(f"inv$x{i}") for i in range(1, tot + 1)]
    b = K.substitute(body, {}) if False else body

    def sub(t):
        if isinstance(t, Expr):
            if t.head == "Slot" and len(t.args) == 1 and K.is_int(t.args[0]):
                i = int(t.args[0])
                return names_[i - 1] if 1 <= i <= tot else t
            return Expr(sub(t.head) if isinstance(t.head, Expr) else t.head, *(sub(x) for x in t.args))
        return t
    b = sub(b)
    if K.contains(b, lambda x: isinstance(x, Expr) and x.head in ("Slot", "SlotSequence")):
        return None
    y = sp.Symbol("inv$y")
    try:
        sols = sp.solve(sp.Eq(S(b), y), sp.Symbol(names_[n - 1].name))
    except Exception:
        return None
    if len(sols) != 1:
        return None
    m = {"inv$y": Expr("Slot", n if tot > 1 else 1)}
    for i, nm in enumerate(names_):
        if i != n - 1:
            m[nm.name] = Expr("Slot", i + 1)
    r = K.substitute(R(sols[0]), m)
    return Expr("Function", ev(r))


@fn("InverseFunction", area=AREA, status="added",
    notes="InverseFunction[f]: ArcSin for Sin etc. (named inverses), the solved inverse of a pure function with "
          "a unique inverse (InverseFunction[2 # + 3 &] = (#1 - 3)/2 &), InverseFunction[InverseFunction[f]] = f; "
          "InverseFunction[f, n, tot] for the n-th of tot arguments; otherwise stays symbolic "
          "(InverseFunction[f][x])")
def _inversefunction(f: Expr, *nt: Expr) -> Expr:
    if len(nt) not in (0, 2):
        raise ValueError("InverseFunction expects InverseFunction[f] or InverseFunction[f, n, tot]")
    n, tot = (1, 1) if not nt else nt
    if not (K.is_int(n) and K.is_int(tot) and 1 <= int(n) <= int(tot)):
        raise ValueError("InverseFunction: n and tot must be integers with 1 <= n <= tot")
    n, tot = int(n), int(tot)
    if isinstance(f, Expr) and f.head == "InverseFunction" and len(f.args) == 1 and tot == 1:
        return f.args[0]
    if tot == 1 and isinstance(f, Symbol) and f.name in _INVERSES:
        return Symbol(_INVERSES[f.name])
    if isinstance(f, Expr) and f.head == "Function":
        r = _pure_inverse(f, n, tot)
        if r is not None:
            return r
    if nt and (n, tot) == (1, 1):
        return Expr("InverseFunction", f)
    raise Unevaluated()


@fn("Indexed", area=AREA, status="added",
    notes="Indexed[list, i] gives list[[i]], Indexed[list, {i, j, ...}] indexes recursively; for a symbolic "
          "expr, Indexed[x, i] stays as the component x_i (Indexed[x, {i}] is the same as Indexed[x, i])")
def _indexed(e: Expr, i: Expr) -> Expr:
    idx = list(i.args) if K.is_list(i) else [i]
    if not idx:
        return e
    if K.is_list(e):
        k = idx[0]
        if not K.is_int(k):
            raise Unevaluated()
        k = int(k)
        n = len(e.args)
        if not (1 <= k <= n or -n <= k <= -1):
            raise ValueError(f"Indexed::partw: Part {k} of {e} does not exist")
        sub = e.args[k - 1] if k > 0 else e.args[n + k]
        return sub if len(idx) == 1 else ev(Expr("Indexed", sub, L(*idx[1:])))
    if K.is_list(i) and len(idx) == 1:
        return Expr("Indexed", e, idx[0])
    raise Unevaluated()


def _distribute(e, g="Plus", f=None, gp=None, fp=None):
    if not isinstance(e, Expr) or not isinstance(e.head, str):
        return e
    f = f or e.head
    if e.head != f:
        return e
    gp, fp = gp or g, fp or f
    if not any(isinstance(x, Expr) and x.head == g for x in e.args):
        return e
    choices = [list(x.args) if isinstance(x, Expr) and x.head == g else [x] for x in e.args]
    total = 1
    for c in choices:
        total *= max(len(c), 1)
        if total > K.MAX_TABLE:
            raise ValueError("Distribute: the result would be too large")
    return Expr(gp, *(Expr(fp, *combo) for combo in itertools.product(*choices)))


def _hname(x, fname):
    if isinstance(x, Symbol):
        return x.name
    raise ValueError(f"{fname}: {x!r} is not a symbol")


@fn("Distribute", area=AREA, status="added",
    notes="Distribute[f[a + b, c + d]] = f[a, c] + f[a, d] + f[b, c] + f[b, d]; Distribute[expr, g] over g; "
          "Distribute[expr, g, f] only when the head is f; Distribute[expr, g, f, gp, fp] uses gp and fp in the "
          "result")
def _distribute_fn(e: Expr, *hs: Expr) -> Expr:
    if len(hs) > 4:
        raise ValueError("Distribute: 1 to 5 arguments expected")
    names_ = [_hname(h, "Distribute") for h in hs]
    return _distribute(e, *names_)


@fn("Prefix", area=AREA, status="added",
    notes="Prefix[f[x]] prints as f@x and Prefix[f[x], h] as h<>x (display wrapper; the expression inside is "
          "evaluated)")
def _prefix(e: Expr, *rest: Expr) -> Expr:
    if len(rest) > 3:
        raise ValueError("Prefix: 1 to 4 arguments expected")
    raise Unevaluated()


@fn("Postfix", area=AREA, status="added",
    notes="Postfix[f[x]] prints as x//f and Postfix[f[x], h] as x<>h (display wrapper)")
def _postfix(e: Expr, *rest: Expr) -> Expr:
    if len(rest) > 3:
        raise ValueError("Postfix: 1 to 4 arguments expected")
    raise Unevaluated()


# ─────────────────────────── N holding, Constant ───────────────────────────

_NHOLD = ("NHoldAll", "NHoldFirst", "NHoldRest")
_orig_n = K._BUILTINS["N"]


def _has_nhold(t):
    return K.contains(t, lambda x: isinstance(x, Expr) and isinstance(x.head, str) and
                      bool(st().attrs.get(x.head, set()) & set(_NHOLD)))


def _n_walk(t, prec, ev_):
    if not isinstance(t, Expr):
        return ev_(Expr("N", t, *prec))
    h = t.head
    attrs = st().attrs.get(h, set()) if isinstance(h, str) else set()
    args = []
    for i, x in enumerate(t.args):
        hold = "NHoldAll" in attrs or (i == 0 and "NHoldFirst" in attrs) or (i > 0 and "NHoldRest" in attrs)
        args.append(x if hold else _n_walk(x, prec, ev_))
    r = ev_(Expr(h, *args))
    if isinstance(r, Expr) and not _has_nhold(r) and not K.same(r, t):
        return _orig_n((r, *prec), ev_, Expr("N", r, *prec))
    return r


def _b_n(a, ev_, expr):
    if a and st().attrs and _has_nhold(a[0]):
        return _n_walk(a[0], tuple(a[1:]), ev_)
    return _orig_n(a, ev_, expr)


K._BUILTINS["N"] = _b_n

_dt_entry = REGISTRY["Dt"]
_orig_dt_impls = list(_dt_entry.impls)


def _dt_constants(f: Expr, *args: Expr) -> Expr:
    consts = sorted({n for n in st().attrs if "Constant" in st().attrs[n]})
    if consts:
        found = set()
        _collect_symbols(f, found)
        consts = [c for c in consts if c in found]
    if consts:
        args = list(args)
        for i, o in enumerate(args):
            if is_rule(o) and is_sym(o.args[0], "Constants"):
                v = o.args[1]
                items = list(v.args) if K.is_list(v) else [v]
                args[i] = Expr(o.head, o.args[0], L(*items, *(Symbol(c) for c in consts)))
                break
        else:
            args.append(Expr("Rule", Symbol("Constants"), L(*(Symbol(c) for c in consts))))
    for fn_, lo, hi in _orig_dt_impls:
        if 1 + len(args) >= lo and (hi is None or 1 + len(args) <= hi):
            return fn_(f, *args)
    raise ValueError("Dt: wrong number of arguments")


_dt_entry.impls[:] = [(_dt_constants, 1, None)]
for _n in ("Constants", "NonConstants"):
    _added(_n, f"option {_n} -> {{...}} of Dt and D (symbols treated as constant / as depending on the variable)")


KL.COMPOUND_HEAD_HOOKS["InformationData"] = _infodata_call


# ─────────────────────────── evaluation stack ───────────────────────────

import contextvars as _cv

_STACK_COMPLETE = _cv.ContextVar("w01_stack_complete", default=False)


def _frame_tag(e):
    if isinstance(e, Expr):
        h = _innermost_head(e)
        return Symbol(h) if h is not None else e.head
    return e


@held("Stack", attrs={"HoldFirst"},
      notes="Stack[] gives the tags (heads) of the evaluations in progress, outermost first, excluding Stack "
            "itself; Stack[patt] the expressions matching patt wrapped in HoldCompleteForm (Stack[_] all); "
            "MikoshiLang's internal evaluations made by library code also appear")
def _h_stack(a, ev_, expr):
    _need(a, 0, "Stack", 1)
    frames = E._STACK.get() or []
    complete = _STACK_COMPLETE.get()
    items = [f[0] for f in frames if f[1] or complete]
    if items and items[-1] is expr:
        items = items[:-1]
    elif items and isinstance(items[-1], Expr) and items[-1].head == "Stack":
        items = items[:-1]
    if not a:
        return L(*(_frame_tag(x) for x in items))
    patt = a[0]
    return L(*(Expr("HoldCompleteForm", x) for x in items if any(True for _ in K.match(patt, x, {}, ev_))))


@held("StackBegin", attrs={"HoldFirst"}, notes="StackBegin[expr] evaluates expr with a fresh evaluation stack "
                                               "(outer evaluations are not shown by Stack)")
def _h_stackbegin(a, ev_, expr):
    _need(a, 1, "StackBegin")
    tok = E._STACK.set([])
    try:
        return ev_(a[0])
    finally:
        E._STACK.reset(tok)


@held("StackInhibit", attrs={"HoldFirst"}, notes="StackInhibit[expr] evaluates expr without adding its "
                                                 "evaluations to the stack")
def _h_stackinhibit(a, ev_, expr):
    _need(a, 1, "StackInhibit")
    frames = E._STACK.get()
    mine = frames[-1] if frames and frames[-1][0] is expr else None
    if mine is not None:
        mine[1] = False                 # StackInhibit itself is not shown either
    tok = E._STACK_INHIBIT.set(True)
    try:
        return ev_(a[0])
    finally:
        E._STACK_INHIBIT.reset(tok)


@held("StackComplete", attrs={"HoldFirst"},
      notes="StackComplete[expr] evaluates expr keeping the earlier expressions of each evaluation chain on the "
            "stack (f[2] and g[4] both appear after f[x_] := g[x^2])")
def _h_stackcomplete(a, ev_, expr):
    _need(a, 1, "StackComplete")
    tok = _STACK_COMPLETE.set(True)
    try:
        return ev_(a[0])
    finally:
        _STACK_COMPLETE.reset(tok)


# ─────────────────────────── Trace options, TracePrint, TraceScan ───────────────────────────

from . import linalg as _linalg  # noqa: E402

_TRACE_OPTS = {"TraceDepth", "TraceForward", "TraceBackward", "TraceAbove", "TraceOn", "TraceOff", "TraceOriginal"}


def _first_form(chain):
    for el in chain:
        if not isinstance(el, list):
            return el
    return None


def _m(patt, x):
    return any(True for _ in K.match(patt, x, {}, lambda y: y))


def _depth_cut(chain, n, d=1):
    out = []
    for el in chain:
        if isinstance(el, list):
            if d < n:
                out.append(_depth_cut(el, n, d + 1))
        else:
            out.append(el)
    return out


def _trace_off(chain, patt):
    """Forms after a form matching patt in its chain are not traced (the chain's final result is kept)."""
    out = []
    for i, el in enumerate(chain):
        if isinstance(el, list):
            out.append(_trace_off(el, patt))
        else:
            out.append(el)
            if _m(patt, el):
                last = chain[-1]
                if i < len(chain) - 1 and not isinstance(last, list):
                    out.append(last)
                return out
    return out


def _trace_on(chain, patt):
    """Only the parts of chains from a form matching patt onwards."""
    out = []
    for i, el in enumerate(chain):
        if isinstance(el, list):
            sub = _trace_on(el, patt)
            if sub:
                out.append(sub)
        elif _m(patt, el):
            return out + list(chain[i:])
    return out


def _filter(chain, patt, fwd, bwd, above):
    """(filtered chain, whether a match occurs anywhere inside)."""
    idx = [i for i, el in enumerate(chain) if not isinstance(el, list) and _m(patt, el)]
    first_m = idx[0] if idx else None
    last_m = idx[-1] if idx else None
    out = []
    nested_hit = False
    for i, el in enumerate(chain):
        if isinstance(el, list):
            sub, hit = _filter(el, patt, fwd, bwd, above)
            nested_hit |= hit
            full = (fwd == "All" and first_m is not None and i > first_m) or \
                   (bwd == "All" and last_m is not None and i < last_m)
            if full:
                out.append(el)
            elif sub:
                out.append(sub)
        else:
            if i in idx or (fwd and first_m is not None and i > first_m) or \
                    (bwd and last_m is not None and i < last_m):
                out.append(el)
    if above and nested_hit:
        if above == "All":
            res, j = [], 0
            for el in chain:
                if isinstance(el, list):
                    while j < len(out) and not isinstance(out[j], list):
                        j += 1
                    if j < len(out):
                        res.append(out[j])
                        j += 1
                else:
                    res.append(el)
            out = res
        else:
            forms = [el for el in chain if not isinstance(el, list)]
            if forms:
                if not (out and not isinstance(out[0], list) and K.same(out[0], forms[0])):
                    out.insert(0, forms[0])
                if len(forms) > 1 and not (out and not isinstance(out[-1], list) and K.same(out[-1], forms[-1])):
                    out.append(forms[-1])
    return out, bool(idx) or nested_hit


def _tf(v, name):
    if v is True or v is False:
        return v
    if is_sym(v, "All"):
        return "All"
    raise ValueError(f"Trace: {name} must be True, False or All")


def _trace_parse(rest, fname):
    form, opts = None, {}
    pos = []
    for x in rest:
        if is_rule(x) and isinstance(x.args[0], Symbol) and x.args[0].name in _TRACE_OPTS:
            opts[x.args[0].name] = ev(x.args[1]) if x.args[0].name not in ("TraceOn", "TraceOff") else x.args[1]
        elif is_rule(x) and isinstance(x.args[0], Symbol):
            raise ValueError(f"{fname}: unknown option {x.args[0].name}")
        else:
            pos.append(x)
    if len(pos) > 1:
        raise NotImplementedError(f"{fname}[expr, form, s] (a third argument) is not supported")
    if pos:
        form = pos[0]
        if isinstance(form, Symbol) and (st().down.get(form.name) or
                                         any(K._def_key(l) == form.name for l, _ in st().exact.values())):
            form = Expr("Alternatives", form, Expr(form.name, Expr("BlankNullSequence")))
    if "TraceOriginal" in opts and opts["TraceOriginal"] is not False:
        raise NotImplementedError(f"{fname}: TraceOriginal -> True is not supported")
    return form, opts


def _traced_chain(e, form, opts):
    chain, r = E.trace_evaluation(e)
    if "TraceOff" in opts:
        chain = _trace_off(chain, opts["TraceOff"])
    if "TraceOn" in opts:
        p = opts["TraceOn"]
        f0 = _first_form(chain)
        if chain and not isinstance(chain[0], list) and f0 is not None and _m(p, f0):
            pass
        else:
            chain = _trace_on(chain, p)
    if "TraceDepth" in opts:
        d = opts["TraceDepth"]
        if is_sym(d, "Infinity"):
            pass
        elif K.is_int(d) and int(d) >= 1:
            chain = _depth_cut(chain, int(d))
        else:
            raise ValueError("Trace: TraceDepth must be a positive integer or Infinity")
    if form is not None:
        fwd = _tf(opts.get("TraceForward", False), "TraceForward")
        bwd = _tf(opts.get("TraceBackward", False), "TraceBackward")
        above = _tf(opts.get("TraceAbove", False), "TraceAbove")
        chain, _ = _filter(chain, form, fwd, bwd, above)
    return chain, r


def _trace_new(e: Expr, *rest: Expr) -> Expr:
    form, opts = _trace_parse(rest, "Trace")
    chain, _ = _traced_chain(e, form, opts)
    return _linalg._trace_out(chain)


REGISTRY["Trace"].impls[:] = [(_trace_new, 1, None)]
STATUS["Trace"] = (STATUS["Trace"][0], STATUS["Trace"][1], STATUS["Trace"][2] +
                   "; options TraceDepth, TraceForward, TraceBackward, TraceAbove, TraceOn, TraceOff (W01)")
for _n, _note in [
        ("TraceDepth", "Trace option: TraceDepth -> n keeps only the first n levels of nested lists"),
        ("TraceForward", "Trace option: True also keeps the later forms of a chain in which form matched; All also "
                         "their subevaluations"),
        ("TraceBackward", "Trace option: True also keeps the earlier forms of a chain in which form matched; All "
                          "also their subevaluations"),
        ("TraceAbove", "Trace option: True adds the first and last forms of every chain containing a match; All "
                       "adds all forms of those chains"),
        ("TraceOn", "Trace option: TraceOn -> patt traces only inside evaluations of forms matching patt"),
        ("TraceOff", "Trace option: TraceOff -> patt does not trace inside evaluations of forms matching patt "
                     "(the forms themselves and their results are kept)")]:
    _added(_n, _note)
E._HOLD_ALL.update({"TracePrint", "TraceScan"})

OUTPUT = [None]     # callable(text) receiving printed lines (TracePrint); default print()


def emit(text):
    (OUTPUT[0] or print)(text)


def _walk_forms(chain, d, out):
    for el in chain:
        if isinstance(el, list):
            _walk_forms(el, d + 1, out)
        else:
            out.append((d, el))


@fn("TracePrint", area=AREA, status="added",
    notes="TracePrint[expr] prints the forms of the evaluation of expr indented by nesting level and returns "
          "the value; TracePrint[expr, form] only matching forms; Trace options accepted (forms are printed "
          "after evaluation completes)")
def _traceprint(e: Expr, *rest: Expr) -> Expr:
    form, opts = _trace_parse(rest, "TracePrint")
    if "TraceBackward" in opts:
        raise ValueError("TracePrint does not support the TraceBackward option")
    chain, r = _traced_chain(e, form, opts)
    forms = []
    _walk_forms(chain, 1, forms)
    for d, f in forms:
        emit(" " * (2 * d - 1) + _fmt(f))
    return r


@fn("TraceScan", area=AREA, status="added",
    notes="TraceScan[f, expr] applies f to HoldForm of every form in the evaluation of expr and returns its "
          "value; TraceScan[f, expr, form] only matching forms; TraceScan[f, expr, form, fp] also applies fp to "
          "(HoldForm[form], HoldForm[result]) of each matching chain; f runs after the evaluation completes")
def _tracescan(f: Expr, e: Expr, *rest: Expr) -> Expr:
    f = ev(f)
    fp = None
    rest = list(rest)
    if len([x for x in rest if not is_rule(x)]) == 2:
        k = [i for i, x in enumerate(rest) if not is_rule(x)][1]
        fp = ev(rest.pop(k))
    form, opts = _trace_parse(rest, "TraceScan")
    chain, r = _traced_chain(e, form, opts)
    forms = []
    _walk_forms(chain, 1, forms)
    for _d, x in forms:
        ev(Expr(K.as_head(f), Expr("HoldForm", x)))
    if fp is not None:
        def chains(c):
            fs = [el for el in c if not isinstance(el, list)]
            if fs:
                ev(Expr(K.as_head(fp), Expr("HoldForm", fs[0]), Expr("HoldForm", fs[-1])))
            for el in c:
                if isinstance(el, list):
                    chains(el)
        chains(chain)
    return r


# ─────────────────────────── main loop: In, Out, $Line, $Pre, $Post, ... ───────────────────────────

def _hist():
    return ext("\x00history")     # n -> {"in": expr, "string": str, "out": value}


def _line():
    v = st().own.get("$Line", 1)
    return int(v) if K.is_int(v) else 1


def _hval(name):
    v = st().own.get(name)
    return v


# Optional hook (text, parse) -> expr used by main_loop to parse each input (W29: $Echo, $SyntaxHandler).
INPUT_HOOK = None


def main_loop(source, whole=True):
    """Evaluate input lines as the Wolfram main loop does; returns [(n, value)] for each line.

    Per line: $PreRead applied to the text, InString[n] and In[n] recorded, $Pre applied to the input,
    evaluation (inside StackBegin), $Post applied to the result, Out[n] recorded, $Line incremented, history
    trimmed to $HistoryLength.
    """
    from ..parser import parse
    pre_read = _hval("$PreRead")
    if pre_read is not None and isinstance(source, str):
        source = ev(Expr(K.as_head(pre_read), source))
        if not isinstance(source, str):
            raise ValueError("$PreRead must return a string")
    texts = [source] if whole else split_statements(source)
    results = []
    for text in texts:
        if not text.strip():
            continue
        n = _line()
        h = _hist()
        h[n] = {"string": text}
        expr = parse(text) if INPUT_HOOK is None else INPUT_HOOK(text, parse)
        h[n]["in"] = expr
        tok = E._STACK.set([])
        try:
            pre = _hval("$Pre")
            value = E.evaluate(Expr(K.as_head(pre), expr)) if pre is not None else E.evaluate(expr)
            post = _hval("$Post")
            if post is not None:
                value = E.evaluate(Expr(K.as_head(post), value))
            h[n]["out"] = value
        finally:
            E._STACK.reset(tok)
            if _line() == n:
                st().own["$Line"] = n + 1
        hl = st().own.get("$HistoryLength")
        if K.is_int(hl):
            for k in [k for k in h if k <= n - int(hl)]:
                del h[k]
        results.append((n, value))
    return results


def preprint(value):
    """The form shown for an output: $PrePrint applied (display only)."""
    pp = _hval("$PrePrint")
    return ev(Expr(K.as_head(pp), value)) if pp is not None else value


def epilog():
    """Evaluate $Epilog (called by the REPL when the session ends)."""
    v = st().own.get("$Epilog")
    if v is not None:
        ev(v)


def _hist_index(a, fname):
    n = _line()
    if not a:
        return n - 1
    k = a[0]
    if not K.is_int(k):
        raise Unevaluated()
    k = int(k)
    return n + k if k <= 0 else k


@fn("Out", area=AREA, status="added",
    notes="Out[n] (%n) is the n-th output of the main loop (Session.evaluate_input / REPL / Jupyter), Out[] (%) "
          "the last, Out[-k] (%%...) the k-th previous; unevaluated when there is no such output")
def _out(*a: Expr) -> Expr:
    if len(a) > 1:
        raise ValueError("Out expects at most one argument")
    i = _hist_index(a, "Out")
    rec = st().ext.get("\x00history", {}).get(i)
    if rec is None or "out" not in rec:
        raise Unevaluated()
    return rec["out"]


@fn("InString", area=AREA, status="added",
    notes="InString[n] is the text of input line n of the main loop; InString[] the last, InString[-k] earlier")
def _instring(*a: Expr) -> Expr:
    if len(a) > 1:
        raise ValueError("InString expects at most one argument")
    i = _hist_index(a, "InString") if a else _line() - 1
    rec = st().ext.get("\x00history", {}).get(i)
    if rec is None:
        raise Unevaluated()
    return rec["string"]


@held("In", attrs={"HoldAll"},
      notes="In[n] re-evaluates input line n of the main loop; In[] the last line, In[-k] earlier lines; "
            "unevaluated when there is no such line")
def _h_in(a, ev_, expr):
    _need(a, 0, "In", 1)
    n = _line()
    if not a:
        i = n - 1
    else:
        k = ev_(a[0])
        if not K.is_int(k):
            return expr
        i = n + int(k) if int(k) <= 0 else int(k)
    rec = st().ext.get("\x00history", {}).get(i)
    if rec is None or "in" not in rec or i == n:
        return expr
    return ev_(rec["in"])


for _n, _note in [
        ("$Line", "number of the current main-loop input line; assignable"),
        ("$HistoryLength", "number of In/Out lines kept by the main loop (default Infinity)"),
        ("$Pre", "function applied to each main-loop input before evaluation (unevaluated if it holds)"),
        ("$Post", "function applied to each main-loop result before it is stored in Out"),
        ("$PrePrint", "function applied to each main-loop result before it is printed (not stored)"),
        ("$PreRead", "function applied to each main-loop input string before it is parsed"),
        ("$Epilog", "expression evaluated when the REPL session ends")]:
    _added(_n, _note)


# ─────────────────────────── Once, LocalSymbol ───────────────────────────

@held("Once", attrs={"HoldFirst"},
      notes="Once[expr] evaluates expr the first time and afterwards returns the cached result for the rest of "
            "the session (Once[expr, \"KernelSession\"]); other persistence locations raise; Once[expr] =. clears")
def _h_once(a, ev_, expr):
    _need(a, 1, "Once", -1)
    loc = "KernelSession"
    for x in a[1:]:
        v = ev_(x)
        if is_rule(v):
            if is_sym(v.args[0], "ExpirationDate") or is_sym(v.args[0], "PersistenceTime"):
                if not is_sym(v.args[1], "None"):
                    raise NotImplementedError("Once: ExpirationDate and PersistenceTime are not supported")
                continue
            raise ValueError(f"Once: unknown option {v.args[0]}")
        loc = v
    if loc != "KernelSession":
        raise NotImplementedError("Once: only the \"KernelSession\" location is supported")
    cache = ext("\x00once")
    key = K.tag(a[0])
    if key in cache:
        return cache[key]
    r = ev_(a[0])
    cache[key] = r
    return r


UNSET_HOOKS["Once"] = lambda lhs: (ext("\x00once").pop(K.tag(lhs.args[0]), None), NULL)[1] if lhs.args else NULL


def local_symbol_base():
    return os.environ.get("MIKOSHILANG_LOCALSYMBOL_BASE") or os.path.join(os.path.expanduser("~"), ".mikoshilang",
                                                                         "LocalSymbols")


def _ls_path(args, fname):
    if not args or not isinstance(args[0], str):
        raise ValueError(f"{fname}: LocalSymbol[\"name\"] or LocalSymbol[\"name\", \"dir\"] expected")
    base = local_symbol_base()
    if len(args) == 2:
        b = args[1]
        if isinstance(b, Expr) and b.head == "File" and b.args and isinstance(b.args[0], str):
            b = b.args[0]
        if not isinstance(b, str):
            raise ValueError(f"{fname}: the base must be a directory name")
        base = b
    elif len(args) > 2:
        raise ValueError(f"{fname}: too many arguments")
    import hashlib
    name = args[0]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:60]
    return os.path.join(base, f"{safe}-{hashlib.sha1(name.encode()).hexdigest()[:10]}.json")


@fn("LocalSymbol", area=AREA, status="added",
    notes="LocalSymbol[\"name\"] gives the value stored persistently (a file under ~/.mikoshilang/LocalSymbols, "
          "or $MIKOSHILANG_LOCALSYMBOL_BASE) by LocalSymbol[\"name\"] = val or := val; unevaluated when never "
          "assigned; LocalSymbol[\"name\", \"dir\"] uses another base; =. deletes it")
def _localsymbol(*a: Expr) -> Expr:
    p = _ls_path(a, "LocalSymbol")
    if not os.path.exists(p):
        raise Unevaluated()
    from ..parser import parse
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    return ev(parse(d["value"]))


def _assign_localsymbol(lhs, rhs, ev_):
    args = [ev_(x) for x in lhs.args]
    p = _ls_path(args, "LocalSymbol")
    delayed = _KIND[0] == ":="
    v = rhs
    text = _fmt(v)
    from ..parser import parse
    try:
        back = parse(text)
    except Exception:
        back = None
    if back is None or not K.same(back, v):
        raise ValueError("LocalSymbol: the value cannot be stored as text that reads back identically")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"name": args[0], "delayed": delayed, "value": text}, fh)
    return v


def _unset_localsymbol(lhs):
    p = _ls_path([ev(x) for x in lhs.args], "LocalSymbol")
    if os.path.exists(p):
        os.remove(p)
    return NULL


K.ASSIGN_HOOKS["LocalSymbol"] = _assign_localsymbol
UNSET_HOOKS["LocalSymbol"] = _unset_localsymbol


# ─────────────────────────── CheckArguments, ArgumentsOptions, SyntaxInformation ───────────────────────────

def _split_args(call, spec, assoc, fname):
    if not isinstance(call, Expr) or not isinstance(call.head, str):
        raise ValueError(f"{fname}: the first argument must be f[args]")
    f = call.head
    if K.is_int(spec):
        lo = hi = int(spec)
    elif K.is_list(spec) and len(spec.args) == 2 and K.is_int(spec.args[0]) and (
            K.is_int(spec.args[1]) or is_sym(spec.args[1], "Infinity")):
        lo = int(spec.args[0])
        hi = 10 ** 9 if is_sym(spec.args[1], "Infinity") else int(spec.args[1])
    else:
        raise ValueError(f"{fname}: the argument count must be n or {{min, max}}")
    extra, mode = [], "Longest"
    if assoc is not None:
        if not (isinstance(assoc, Expr) and assoc.head == "Association"):
            raise ValueError(f"{fname}: the third argument must be an association")
        for r in assoc.args:
            k, v = r.args
            if k == "ExtraOptions":
                for x in (v.args if K.is_list(v) else (v,)):
                    if isinstance(x, Symbol):
                        extra.extend(_known_options(x.name))
                    elif is_rule(x):
                        extra.append(x)
                    else:
                        raise ValueError(f"{fname}: invalid ExtraOptions entry {x!r}")
            elif k == "OptionsMode":
                mode = v if isinstance(v, str) else "None" if is_sym(v, "None") else None
                if mode not in ("Longest", "Shortest", "None"):
                    raise ValueError(f"{fname}: OptionsMode must be \"Longest\", \"Shortest\" or None")
            elif k == "Head":
                pass
            else:
                raise ValueError(f"{fname}: unknown key {k!r}")
    known = {_opt_key(r.args[0]) for r in list(_known_options(f)) + extra}
    args = list(call.args)
    opts = []
    if mode != "None":
        while args and K._is_option(args[-1]) and len(args) > (lo if mode == "Shortest" else 0):
            o = _flat_opts([args[-1]])
            if mode == "Shortest" and not all(_opt_key(r.args[0]) in known for r in o):
                break
            opts = o + opts
            args.pop()
    if not lo <= len(args) <= hi:
        exp = str(lo) if lo == hi else f"between {lo} and {hi}"
        return None, f"{f}::argx: {f} called with {len(args)} arguments; {exp} expected"
    for r in opts:
        if _opt_key(r.args[0]) not in known:
            return None, f"{f}::optx: Unknown option {_fmt(r.args[0])} in {_fmt(call)}"
    return (args, opts), None


@held("CheckArguments", attrs={"HoldFirst"},
      notes="CheckArguments[f[args], n] is True when args is n positional arguments followed by options known "
            "to f (Options[f]), else False; {min, max} counts; <|\"ExtraOptions\" -> ..., \"OptionsMode\" -> "
            "...|>")
def _h_checkarguments(a, ev_, expr):
    _need(a, 2, "CheckArguments", 3)
    call = a[0]
    if isinstance(call, Expr):
        call = Expr(call.head, *(ev_(x) for x in call.args))
    res, _msg = _split_args(call, ev_(a[1]), ev_(a[2]) if len(a) == 3 else None, "CheckArguments")
    return res is not None


@held("ArgumentsOptions", attrs={"HoldFirst"},
      notes="ArgumentsOptions[f[args], n] gives {{positional args}, {options}} (checked against Options[f]); a "
            "Failure[\"ArgumentsOptions\", ...] when the count or an option is wrong; {min, max}; \"Head\" key")
def _h_argumentsoptions(a, ev_, expr):
    _need(a, 2, "ArgumentsOptions", 3)
    call = a[0]
    if isinstance(call, Expr):
        call = Expr(call.head, *(ev_(x) for x in call.args))
    assoc = ev_(a[2]) if len(a) == 3 else None
    res, msg = _split_args(call, ev_(a[1]), assoc, "ArgumentsOptions")
    if res is None:
        return Expr("Failure", "ArgumentsOptions", Expr("Association", Expr("Rule", "MessageTemplate", msg)))
    head = "List"
    if assoc is not None:
        for r in assoc.args:
            if r.args[0] == "Head":
                head = K.as_head(r.args[1])
    return L(Expr(head, *res[0]), Expr(head, *res[1]))


@held("SyntaxInformation", attrs={"HoldAll"},
      notes="SyntaxInformation[f] gives the syntax properties set by SyntaxInformation[f] = {\"ArgumentsPattern\" "
            "-> {...}, ...} ({} if none); used only for advisories, not evaluation. Built-in syntax data is not "
            "catalogued and raises")
def _h_syntaxinformation(a, ev_, expr):
    _need(a, 1, "SyntaxInformation")
    n = _sym_arg(a, ev_, "SyntaxInformation")
    d = st().ext.get("SyntaxInformation", {})
    if n in d:
        return d[n]
    if is_system(n):
        raise NotImplementedError(f"SyntaxInformation[{n}]: built-in syntax information is not catalogued")
    return L()


def _assign_syntaxinformation(lhs, rhs, ev_):
    n = _sym_arg(lhs.args, ev_, "SyntaxInformation")
    v = ev_(rhs)
    if not K.is_list(v) or not all(is_rule(r) and isinstance(r.args[0], str) for r in v.args):
        raise ValueError("SyntaxInformation[f] must be a list of \"property\" -> value rules")
    allowed = {"ArgumentsPattern", "LocalVariables", "ColorEqualSigns", "OptionNames"}
    for r in v.args:
        if r.args[0] not in allowed:
            raise ValueError(f"SyntaxInformation: unknown property {r.args[0]!r}")
    if is_system(n):
        raise ValueError(f"SyntaxInformation: {n} is protected")
    ext("SyntaxInformation")[n] = v
    return v


K.ASSIGN_HOOKS["SyntaxInformation"] = _assign_syntaxinformation


# ─────────────────────────── display wrappers (formatting hooks) ───────────────────────────

def _fmt_prefix(e, parent):
    from ..formatting import input_form
    if not e.args or not isinstance(e.args[0], Expr) or len(e.args[0].args) != 1:
        return None
    inner = e.args[0]
    if len(e.args) >= 2 and isinstance(e.args[1], str):
        return e.args[1] + input_form(inner.args[0], 100)
    return input_form(inner.head if isinstance(inner.head, Expr) else Symbol(inner.head), 100) + "@" + \
        input_form(inner.args[0], 100)


def _fmt_postfix(e, parent):
    from ..formatting import input_form
    if not e.args or not isinstance(e.args[0], Expr) or len(e.args[0].args) != 1:
        return None
    inner = e.args[0]
    if len(e.args) >= 2 and isinstance(e.args[1], str):
        return input_form(inner.args[0], 100) + e.args[1]
    return input_form(inner.args[0], 1) + "//" + input_form(
        inner.head if isinstance(inner.head, Expr) else Symbol(inner.head), 1)


def _fmt_contents(e, parent):
    from ..formatting import input_form
    if len(e.args) == 1:
        return input_form(e.args[0], parent)
    return None


def _install_format_hooks():
    from .. import formatting
    hooks = formatting.FORMAT_HOOKS
    hooks["Defer"] = _fmt_contents
    hooks["HoldCompleteForm"] = _fmt_contents
    hooks["Prefix"] = _fmt_prefix
    hooks["Postfix"] = _fmt_postfix
    hooks["Definition"] = lambda e, p: _definition_text(e, False)
    hooks["FullDefinition"] = lambda e, p: _definition_text(e, True)


_install_format_hooks()


# ─────────────────────────── inert symbols ───────────────────────────

set_attrs("IgnoringInactive", "HoldAll")
for _n, _note in [
        ("Automatic", "inert option value: chosen automatically by the function that uses it"),
        ("Full", "inert option value (e.g. PlotRange -> Full)"),
        ("Manual", "inert option value: selected manually"),
        ("Inherited", "inert option value: inherited from an enclosing setting"),
        ("Method", "option name Method -> \"name\" / {\"name\", opts} / Automatic for algorithm choices"),
        ("General", "symbol to which general messages are attached (General::tag)"),
        ("ActivateResult", "option name: whether Inactive results are activated before being returned"),
        ("AllowedHeads", "option name: heads of subexpressions into which a function may descend"),
        ("Application", "Application[f, g] is a formal application with no built-in meaning (stays unevaluated)"),
        ("ResolveContextAliases", "option of Names, NameQ and Contexts: False ignores $ContextAliases"),
        ("InverseFunctions", "option name of Solve and Reduce: whether InverseFunction objects may be used"),
        ("$NewSymbol", "if set, $NewSymbol[name, context] is evaluated before a new symbol is created by the "
                       "parser")]:
    INERT[_n] = _note
    _added(_n, _note)
for _n in ("Automatic", "Full", "Manual", "Inherited", "Method", "General", "ActivateResult", "AllowedHeads",
           "Application", "ResolveContextAliases", "InverseFunctions", "Heads", "Threaded", "Constants", "NonConstants",
           "ExcludedContexts", "IncludedContexts", "TraceDepth", "TraceForward", "TraceBackward", "TraceAbove",
           "TraceOn", "TraceOff"):
    K.BUILTIN_ATTRIBUTES.setdefault(_n, set()).add("Protected")


# ─────────────────────────── not provided ───────────────────────────

SKIPPED = {
    "Restricted": "a form for Interpreter/FormObject (curated interpreters, not available offline)",
    "PlaceholderReplace": "notebook front-end placeholder option (Paste)",
    "InheritScope": "DynamicModule/front-end option",
    "UnsavedVariables": "DynamicModule/front-end option",
    "Initialization": "DynamicModule/Manipulate front-end option",
    "IncludeDefinitions": "Wolfram Cloud deployment option",
    "PartProtection": "Wolfram Cloud (CloudExpression) option",
    "Anonymous": "Wolfram Cloud (permissions) setting",
    "InertEvaluate": "compiled-code (FunctionCompile) construct",
    "InertExpression": "compiled-code (FunctionCompile) construct",
    "DownValuesFunction": "compiled-code (FunctionCompile) construct",
    "RuntimeAttributes": "Compile option (compiler not part of this batch)",
    "RuntimeOptions": "Compile option (compiler not part of this batch)",
    "EvaluationObject": "parallel-kernel evaluation objects (ParallelSubmit)",
    "InitializationObject": "persistence framework (PersistenceLocation) not available",
    "InitializationObjects": "persistence framework (PersistenceLocation) not available",
    "InitializationValue": "persistence framework (PersistenceLocation) not available",
    "Initialize": "persistence framework (PersistenceLocation) not available",
    "$Initialization": "persistence/kernel start-up file settings",
    "$PreInitialization": "persistence/kernel start-up file settings",
    "Symmetrize": "needs SymmetrizedArray and Symmetric/Antisymmetric symmetry objects (other batches)",
    "AbsoluteOptions": "needs the graphics option catalogue (Graphics batch)",
    "PackageExported": "Structured Package Format (PackageScope/PackageExport files)",
    "PackageImport": "Structured Package Format (PackageScope/PackageExport files)",
    "PackageScoped": "Structured Package Format (PackageScope/PackageExport files)",
    "PackageInitialize": "Structured Package Format (PackageScope/PackageExport files)",
    "TraceOriginal": "MikoshiLang's trace records only forms that change, so the original forms cannot be tested",
}


_install_parse_hook()
