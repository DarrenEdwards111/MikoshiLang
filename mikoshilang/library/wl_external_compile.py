"""Wolfram batch W30 (part 2): Compile, FunctionCompile and the compiler's type language.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). MikoshiLang has no machine-code compiler: Compile and FunctionCompile keep Wolfram's
*semantics* (argument types checked and coerced, machine-number results, documented Failure objects for bad
arguments, CompilerRuntimeErrorAction for integer overflow, declarations and IfCompiled) and evaluate the body
directly, so results agree with Wolfram's compiled code where it succeeds and with its documented fallback to the
Wolfram Engine where it does not. Speed is not reproduced.
"""
from __future__ import annotations

import itertools

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
import importlib
from .core import Unevaluated, fn, record, L, is_list
from .wl_common import ev, is_sym, apply

E = importlib.import_module("mikoshilang.evaluate")

AREA = "external interfaces: compilation (W30)"
NONE = Symbol("None")


def _show(x):
    try:
        from .. import formatting as F
        return F.input_form(x)
    except Exception:
        return repr(x)


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def _key(k):
    return k if isinstance(k, str) else (k.name if isinstance(k, Symbol) else None)


def _split_opts(args, names, fname):
    args = list(args)
    opts = {}
    while args and _is_rule(args[-1]) and _key(args[-1].args[0]) is not None:
        k = _key(args[-1].args[0])
        if k not in names:
            raise ValueError(f"{fname}: {k} is not a known option (expected one of {', '.join(sorted(names))})")
        opts[k] = ev(args.pop().args[1])
    return args, opts


def _assoc(pairs):
    return KL.make_assoc([Expr("Rule", k, v) for k, v in pairs])


def _assoc_get(a, k, default=None):
    for r in a.args:
        if r.args[0] == k:
            return r.args[1]
    return default


# ═══════════════════════════════════════ types ═══════════════════════════════════════

_INT_RANGES = {"Integer8": (-2 ** 7, 2 ** 7 - 1), "Integer16": (-2 ** 15, 2 ** 15 - 1),
               "Integer32": (-2 ** 31, 2 ** 31 - 1), "Integer64": (-2 ** 63, 2 ** 63 - 1),
               "Integer128": (-2 ** 127, 2 ** 127 - 1), "UnsignedInteger8": (0, 2 ** 8 - 1),
               "UnsignedInteger16": (0, 2 ** 16 - 1), "UnsignedInteger32": (0, 2 ** 32 - 1),
               "UnsignedInteger64": (0, 2 ** 64 - 1)}
_ALIASES = {"MachineInteger": "Integer64", "MachineReal": "Real64", "Complex128": "ComplexReal64",
            "MachineComplex": "ComplexReal64", "Expression": "InertExpression"}
NATIVE = set(_INT_RANGES) | {"Real32", "Real64", "ComplexReal64", "Boolean", "String", "InertExpression", "Void"}
_COMPOUND = {"PackedArray": 2, "ListVector": 1, "NumericArray": 2}


# X01 extension points (wl_core_extras_compile): rewrites of the compiled view of a body, extra declaration kinds
# and extra type kinds. PREPARE_HOOKS: f(expr, walk) -> replacement or None. DECL_HOOKS[head]: f(decl, aliases,
# fname) registering the declaration. TYPE_CONFORMERS[kind]: f(v, t, b, fname) -> (ok, value) for a parsed type
# tuple (kind, ...). TYPE_EXPRS[kind]: f(t) -> Wolfram form of the type.
PREPARE_HOOKS = []
DECL_HOOKS = {}
TYPE_CONFORMERS = {}
TYPE_EXPRS = {}


class TVar:
    def __init__(self, name):
        self.name = name

    def __eq__(self, o):
        return isinstance(o, TVar) and o.name == self.name

    def __hash__(self):
        return hash(("TVar", self.name))

    def __repr__(self):
        return f"TVar({self.name})"


def _tvar_name(x):
    if isinstance(x, Symbol):
        return x.name
    if isinstance(x, str):
        return x
    return None


class _AnyName(dict):
    """Alias table that accepts every unknown type name (used when validating Typed/TypeSpecifier, whose type
    names may come from a TypeDeclaration)."""

    def __contains__(self, k):
        return isinstance(k, str) and _ALIASES.get(k, k) not in NATIVE and _ALIASES.get(k, k) not in _COMPOUND

    def __getitem__(self, k):
        return "InertExpression"


def parse_type(t, fname, tvars=(), aliases=None):
    """Python representation of a type expression: a native name, ("PackedArray", elem, rank),
    ("ListVector", elem), ("Fn", (args...), ret), ("Literal", v), ("Seq", var), ("Eval", expr) or a TVar."""
    if aliases is None:
        aliases = {}
    if isinstance(t, Expr) and t.head == "TypeSpecifier" and len(t.args) == 1:
        return parse_type(t.args[0], fname, tvars, aliases)
    if isinstance(t, Symbol) and t.name in tvars:
        return TVar(t.name)
    if isinstance(t, str):
        if t in tvars:
            return TVar(t)
        if t in aliases:
            return aliases[t]
        n = _ALIASES.get(t, t)
        if n in NATIVE:
            return n
        if n in _COMPOUND:
            raise ValueError(f"{fname}: the compound type {t!r} needs parameters, as in "
                             f"TypeSpecifier[\"{t}\"][...]")
        if n.startswith("C") and n[1:2].isupper():
            raise NotImplementedError(f"{fname}: C types such as {t!r} are only supported by ForeignFunctionLoad")
        raise ValueError(f"{fname}: unknown type {t!r}")
    if K.is_int(t) and not isinstance(t, bool):
        return ("Literal", int(t))
    if t is True or t is False:
        return ("Literal", t)
    if isinstance(t, Expr):
        if t.head == "Rule" and is_list(t.args[0]):
            return ("Fn", tuple(parse_type(a, fname, tvars, aliases) for a in t.args[0].args),
                    parse_type(t.args[1], fname, tvars, aliases))
        if t.head == "LiteralType" and len(t.args) == 1:
            v = t.args[0]
            if not (isinstance(v, (str, bool)) or K.is_int(v)):
                raise TypeError(f"{fname}: a literal type must be an integer, a string or a Boolean")
            return ("Literal", int(v) if K.is_int(v) and not isinstance(v, bool) else v)
        if t.head == "SequenceType" and len(t.args) == 1:
            return ("Seq", _tvar_name(t.args[0]))
        if t.head == "TypeEvaluate" and len(t.args) == 1:
            return ("Eval", t.args[0])
        h = t.head
        cons = None
        if isinstance(h, Expr) and h.head == "TypeSpecifier" and len(h.args) == 1 and isinstance(h.args[0], str):
            cons = h.args[0]
        elif isinstance(h, str) and h in _COMPOUND:
            cons = h
        if cons is not None:
            if cons not in _COMPOUND:
                raise NotImplementedError(f"{fname}: the compound type {cons!r} is not supported "
                                          f"(PackedArray, NumericArray, ListVector)")
            if len(t.args) != _COMPOUND[cons]:
                raise ValueError(f"{fname}: {cons} takes {_COMPOUND[cons]} type parameter(s)")
            ps = [parse_type(a, fname, tvars, aliases) for a in t.args]
            if cons in ("PackedArray", "NumericArray"):
                r = ps[1]
                if not (isinstance(r, TVar) or (isinstance(r, tuple) and r[0] in ("Literal", "Eval"))):
                    raise ValueError(f"{fname}: the rank of a {cons} must be an integer")
                return (cons, ps[0], r)
            return (cons, ps[0])
        if isinstance(t, Expr) and t.head == "ForAllType":
            raise ValueError(f"{fname}: ForAllType is only allowed as the type of a declared function")
    if isinstance(t, Symbol):
        raise ValueError(f"{fname}: {t.name} is not a type (types are strings such as \"Integer64\")")
    raise ValueError(f"{fname}: {_show(t)} is not a valid type")


def type_expr(t):
    """Wolfram form of a parsed type."""
    if isinstance(t, str):
        return t
    if isinstance(t, TVar):
        return Symbol(t.name)
    k = t[0]
    if k == "Literal":
        return t[1] if not isinstance(t[1], str) else Expr("LiteralType", t[1])
    if k == "Fn":
        return Expr("Rule", L(*(type_expr(a) for a in t[1])), type_expr(t[2]))
    if k in _COMPOUND:
        return Expr(Expr("TypeSpecifier", k), *(type_expr(a) for a in t[1:]))
    if k == "Seq":
        return Expr("SequenceType", Symbol(t[1]))
    if k == "Eval":
        return Expr("TypeEvaluate", t[1])
    if k in TYPE_EXPRS:
        return TYPE_EXPRS[k](t)
    raise ValueError(t)


def _typespec(t):
    e = type_expr(t)
    if isinstance(e, Expr) and isinstance(e.head, Expr) and e.head.head == "TypeSpecifier":
        return e
    return Expr("TypeSpecifier", e)


def _value_type(v):
    """The natural type of a concrete value (for binding type variables)."""
    if v is True or v is False:
        return "Boolean"
    if isinstance(v, str):
        return "String"
    if K.is_int(v):
        return "Integer64"
    if isinstance(v, float):
        return "Real64"
    if isinstance(v, complex):
        return "ComplexReal64"
    if is_list(v):
        if not v.args:
            return None
        ts = {_value_type(a) for a in v.args}
        if len(ts) == 1:
            t = ts.pop()
            if isinstance(t, tuple) and t[0] == "PackedArray":
                return ("PackedArray", t[1], ("Literal", t[2][1] + 1))
            if t in ("Integer64", "Real64", "ComplexReal64"):
                return ("PackedArray", t, ("Literal", 1))
            if t is not None:
                return ("ListVector", t)
    return None


def _num(v):
    """Python number of an exact or inexact numeric expression, or None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float, complex)):
        return v
    if K.is_int(v):
        return int(v)
    try:
        from ..math_ops import to_sympy
        s = to_sympy(v)
    except Exception:
        return None
    if isinstance(s, sp.Basic) and s.is_number and not s.free_symbols:
        c = complex(sp.N(s))
        return c.real if c.imag == 0 and s.is_real else c
    return None


def _eval_literal(expr, b, fname):
    e = expr
    for name, val in b.items():
        if isinstance(val, tuple) and val[0] == "Literal":
            e = K.substitute(e, {name: val[1]})
    r = ev(e)
    if K.is_int(r) and not isinstance(r, bool):
        return ("Literal", int(r))
    if isinstance(r, (str, bool)):
        return ("Literal", r)
    if isinstance(r, Expr) and r.head == "TypeSpecifier":
        return parse_type(r, fname)
    raise ValueError(f"{fname}: TypeEvaluate gave {_show(r)}, not a type")


def conform(v, t, b, fname):
    """(ok, coerced value); b holds type-variable bindings and is updated."""
    if isinstance(t, TVar):
        if t.name in b:
            return conform(v, b[t.name], b, fname)
        vt = _value_type(v)
        if vt is None:
            return False, v
        b[t.name] = vt
        return True, v
    if isinstance(t, tuple) and t[0] == "Eval":
        return conform(v, _eval_literal(t[1], b, fname), b, fname)
    if isinstance(t, tuple) and t[0] in TYPE_CONFORMERS:
        return TYPE_CONFORMERS[t[0]](v, t, b, fname)
    if isinstance(t, str):
        if t in _INT_RANGES:
            if K.is_int(v) and not isinstance(v, bool):
                lo, hi = _INT_RANGES[t]
                return lo <= int(v) <= hi, int(v)
            return False, v
        if t in ("Real64", "Real32"):
            if isinstance(v, float):
                return True, v
            if K.is_int(v) and not isinstance(v, bool):
                return True, float(int(v))
            return False, v
        if t == "ComplexReal64":
            n = _num(v)
            if n is None or isinstance(v, bool):
                return False, v
            if K.is_int(v):
                return True, float(int(v))
            if isinstance(v, (float, complex)) or ev(Expr("MachineNumberQ", v)) is True:
                return True, v
            return False, v
        if t == "Boolean":
            return (v is True or v is False), v
        if t == "String":
            return isinstance(v, str), v
        if t == "InertExpression":
            return True, v
        return False, v
    k = t[0]
    if k == "Literal":
        return (v == t[1] and type(v) is type(t[1])) or (K.is_int(v) and isinstance(t[1], int)
                                                         and not isinstance(t[1], bool) and int(v) == t[1]), v
    if k == "Fn":
        ok = isinstance(v, Expr) and (v.head == "Function" or v.head == "CompiledCodeFunction")
        return ok, v
    if k in ("PackedArray", "NumericArray"):
        r = t[2]
        if isinstance(r, TVar):
            depth = _depth(v)
            if r.name in b:
                rv = b[r.name]
                if not (isinstance(rv, tuple) and rv[0] == "Literal" and rv[1] == depth):
                    return False, v
            else:
                b[r.name] = ("Literal", depth)
            rank = depth
        else:
            if r[0] == "Eval":
                r = _eval_literal(r[1], b, fname)
            rank = r[1]
        return _conform_array(v, t[1], rank, b, fname)
    if k == "ListVector":
        if not is_list(v):
            return False, v
        out = []
        for a in v.args:
            ok, c = conform(a, t[1], b, fname)
            if not ok:
                return False, v
            out.append(c)
        return True, L(*out)
    return False, v


def _depth(v):
    d = 0
    while is_list(v) and v.args:
        d += 1
        v = v.args[0]
    return d


def _conform_array(v, elem, rank, b, fname):
    if rank == 0:
        return conform(v, elem, b, fname)
    if not is_list(v) or not v.args:
        return False, v
    out = []
    shape = None
    for a in v.args:
        ok, c = _conform_array(a, elem, rank - 1, b, fname)
        if not ok:
            return False, v
        s = len(a.args) if is_list(a) else None
        if shape is not None and s != shape:
            return False, v
        shape = s
        out.append(c)
    return True, L(*out)


def machine(x):
    """Machine-number form of a compiled result: integers and Booleans stay, other exact numbers become reals or
    complexes, lists are converted elementwise, anything else is left as evaluated."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, complex):
        return x.real if x.imag == 0 else ev(Expr("Plus", x.real, Expr("Times", x.imag, Symbol("I"))))
    if K.is_int(x) or isinstance(x, float):
        return x
    if is_list(x):
        return L(*(machine(a) for a in x.args))
    n = _num(x)
    if n is not None:
        r = ev(Expr("N", x))
        return machine(r) if isinstance(r, complex) else r
    return x


# ═══════════════════════════════════════ type inference (TypeOf) ═══════════════════════════════════════

_ORDER = ["Integer8", "UnsignedInteger8", "Integer16", "UnsignedInteger16", "Integer32", "UnsignedInteger32",
          "Integer64", "UnsignedInteger64", "Integer128", "Real32", "Real64", "ComplexReal64"]
_REALFUN = {"Sin", "Cos", "Tan", "Exp", "Log", "Sqrt", "ArcTan", "ArcSin", "ArcCos", "Sinh", "Cosh", "Tanh", "N",
            "Log10", "Log2"}
_BOOLFUN = {"Equal", "Unequal", "Less", "Greater", "LessEqual", "GreaterEqual", "And", "Or", "Not", "SameQ",
            "UnsameQ", "EvenQ", "OddQ", "Xor", "TrueQ", "StringMatchQ", "StringContainsQ", "PrimeQ"}


def _num_join(a, b, fname):
    for t in (a, b):
        if t not in _ORDER:
            raise NotImplementedError(f"{fname}: cannot infer arithmetic on type {_show(type_expr(t))}")
    if a == b:
        return a
    ia, ib = _ORDER.index(a), _ORDER.index(b)
    if a in _INT_RANGES and b in _INT_RANGES and (a.startswith("U") != b.startswith("U")):
        return "Integer64" if max(ia, ib) <= _ORDER.index("Integer64") else _ORDER[max(ia, ib)]
    return _ORDER[max(ia, ib)]


def _floatify(t):
    if t in _INT_RANGES:
        return "Real64"
    return t


def infer(e, env, decls, fname="TypeOf"):
    """Type of the unevaluated expression e; env maps variable names to types."""
    if e is True or e is False:
        return "Boolean"
    if isinstance(e, str):
        return "String"
    if K.is_int(e):
        return "Integer64"
    if isinstance(e, float):
        return "Real64"
    if isinstance(e, complex):
        return "ComplexReal64"
    if isinstance(e, Symbol):
        if e.name in env:
            return env[e.name]
        if e.name in ("Pi", "E", "EulerGamma", "GoldenRatio", "Degree"):
            return "Real64"
        if e.name == "I":
            return "ComplexReal64"
        raise NotImplementedError(f"{fname}: cannot infer the type of the symbol {e.name} (give it with Typed)")
    if not isinstance(e, Expr):
        raise NotImplementedError(f"{fname}: cannot infer the type of {_show(e)}")
    h = e.head
    a = e.args
    if isinstance(h, str):
        if h == "Typed" and len(a) == 2:
            return parse_type(ev(a[1]), fname)
        if h == "TypeHint" and len(a) == 2:
            return parse_type(ev(a[1]), fname)
        if h == "IfCompiled" and len(a) == 2:
            return infer(a[0], env, decls, fname)
        if h == "Function" and len(a) == 2:
            ps = a[0].args if is_list(a[0]) else (a[0],)
            env2 = dict(env)
            ts = []
            for p in ps:
                if not (isinstance(p, Expr) and p.head == "Typed" and len(p.args) == 2
                        and isinstance(p.args[0], Symbol)):
                    raise NotImplementedError(f"{fname}: Function parameters need types (Typed[x, type])")
                t = parse_type(ev(p.args[1]), fname)
                env2[p.args[0].name] = t
                ts.append(t)
            return ("Fn", tuple(ts), infer(a[1], env2, decls, fname))
        if h in ("Plus", "Times", "Subtract", "Minus"):
            t = infer(a[0], env, decls, fname)
            for x in a[1:]:
                t = _num_join(t, infer(x, env, decls, fname), fname)
            return t
        if h == "Divide" and len(a) == 2:
            return _floatify(_num_join(infer(a[0], env, decls, fname), infer(a[1], env, decls, fname), fname))
        if h == "Power" and len(a) == 2:
            tb, te = infer(a[0], env, decls, fname), infer(a[1], env, decls, fname)
            if isinstance(a[1], Expr) and a[1].head in ("Rational",) or (isinstance(a[1], Expr) and a[1].head ==
                                                                         "Times"):
                return _floatify(_num_join(tb, te, fname))
            if K.is_int(a[1]) and int(a[1]) < 0:
                return _floatify(tb)
            return _num_join(tb, te, fname)
        if h in _REALFUN:
            t = infer(a[0], env, decls, fname)
            return "ComplexReal64" if t == "ComplexReal64" else _floatify(_num_join(t, t, fname))
        if h in ("Abs", "Re", "Im") and len(a) == 1:
            t = infer(a[0], env, decls, fname)
            return "Real64" if t == "ComplexReal64" else t
        if h in _BOOLFUN:
            return "Boolean"
        if h in ("StringLength", "Length", "Floor", "Ceiling", "Round", "Quotient", "Mod", "IntegerLength") \
                and a:
            if h == "Mod":
                return _num_join(infer(a[0], env, decls, fname), infer(a[1], env, decls, fname), fname)
            return "Integer64"
        if h in ("StringJoin", "ToString", "StringTake", "StringDrop", "StringReverse", "ToUpperCase",
                 "ToLowerCase", "StringReplace"):
            return "String"
        if h == "If" and len(a) == 3:
            t1, t2 = infer(a[1], env, decls, fname), infer(a[2], env, decls, fname)
            if t1 == t2:
                return t1
            return _num_join(t1, t2, fname)
        if h == "CompoundExpression":
            return infer(a[-1], env, decls, fname) if a else "Void"
        if h in ("Module", "With", "Block") and len(a) == 2 and is_list(a[0]):
            env2 = dict(env)
            for v in a[0].args:
                if isinstance(v, Expr) and v.head == "Set" and isinstance(v.args[0], Symbol):
                    env2[v.args[0].name] = infer(v.args[1], env2, decls, fname)
            return infer(a[1], env2, decls, fname)
        if h == "List":
            if not a:
                raise NotImplementedError(f"{fname}: cannot infer the type of an empty list")
            ts = [infer(x, env, decls, fname) for x in a]
            t = ts[0]
            if isinstance(t, tuple) and t[0] == "PackedArray" and all(x == t for x in ts):
                return ("PackedArray", t[1], ("Literal", t[2][1] + 1))
            if all(x in _ORDER for x in ts):
                el = ts[0]
                for x in ts[1:]:
                    el = _num_join(el, x, fname)
                return ("PackedArray", el, ("Literal", 1))
            if all(x == t for x in ts):
                return ("ListVector", t)
            raise NotImplementedError(f"{fname}: cannot infer the type of a list with mixed element types")
        if h == "Range":
            ts = [infer(x, env, decls, fname) for x in a]
            el = ts[0]
            for x in ts[1:]:
                el = _num_join(el, x, fname)
            return ("PackedArray", el, ("Literal", 1))
        if h == "Table" and len(a) >= 2:
            env2 = dict(env)
            for it in a[1:]:
                if is_list(it) and it.args and isinstance(it.args[0], Symbol):
                    rng = [infer(x, env2, decls, fname) for x in it.args[1:]]
                    t = rng[0] if rng else "Integer64"
                    for x in rng[1:]:
                        t = _num_join(t, x, fname)
                    env2[it.args[0].name] = t
            el = infer(a[0], env2, decls, fname)
            if isinstance(el, tuple) and el[0] == "PackedArray":
                return ("PackedArray", el[1], ("Literal", el[2][1] + len(a) - 1))
            if el in _ORDER:
                return ("PackedArray", el, ("Literal", len(a) - 1))
            raise NotImplementedError(f"{fname}: cannot infer the element type of Table")
        if h == "Dot" and len(a) == 2:
            t1, t2 = infer(a[0], env, decls, fname), infer(a[1], env, decls, fname)
            if all(isinstance(t, tuple) and t[0] == "PackedArray" for t in (t1, t2)):
                r = t1[2][1] + t2[2][1] - 2
                el = _num_join(t1[1], t2[1], fname)
                return el if r == 0 else ("PackedArray", el, ("Literal", r))
        if h in ("Total", "First", "Last") and len(a) == 1:
            t = infer(a[0], env, decls, fname)
            if isinstance(t, tuple) and t[0] == "PackedArray":
                r = t[2][1] - 1
                return t[1] if r == 0 else ("PackedArray", t[1], ("Literal", r))
            if isinstance(t, tuple) and t[0] == "ListVector":
                return t[1]
        if h == "Cast" and len(a) == 2:
            return parse_type(ev(a[1]), fname)
        if h == "TypeOf":
            raise NotImplementedError(f"{fname}: nested TypeOf is not supported")
        if h in decls:
            ts = tuple(infer(x, env, decls, fname) for x in a)
            for ov in decls[h]:
                b = {}
                if len(ov.args) == len(ts) and all(_type_match(p, t, b) for p, t in zip(ov.args, ts)):
                    return _subst_type(ov.ret, b, fname)
            raise ValueError(f"{fname}: no declaration of {h} accepts argument types "
                             f"{[_show(type_expr(t)) for t in ts]}")
    raise NotImplementedError(f"{fname}: cannot infer the type of {_show(e)}")


def _type_match(p, t, b):
    if isinstance(p, TVar):
        if p.name in b:
            return b[p.name] == t
        b[p.name] = t
        return True
    if isinstance(p, tuple) and isinstance(t, tuple) and p[0] == t[0] and len(p) == len(t):
        return all(_type_match(x, y, b) for x, y in zip(p[1:], t[1:]))
    return p == t


def _subst_type(t, b, fname):
    if isinstance(t, TVar):
        return b.get(t.name, t)
    if isinstance(t, tuple):
        if t[0] == "Eval":
            return _eval_literal(t[1], b, fname)
        if t[0] in ("Literal", "Seq"):
            return t
        return tuple([t[0]] + [_subst_type(x, b, fname) for x in t[1:]]) if t[0] != "Fn" else \
            ("Fn", tuple(_subst_type(x, b, fname) for x in t[1]), _subst_type(t[2], b, fname))
    return t


# ═══════════════════════════════════════ declarations ═══════════════════════════════════════

class Overload:
    def __init__(self, name, tvars, cond, args, ret, func):
        self.name, self.tvars, self.cond, self.args, self.ret, self.func = name, tvars, cond, args, ret, func


def _typed_function(tf, fname, aliases):
    """(tvars, cond, argtypes, ret, Function) of Typed[type]@Function[...] / Typed[Function[...], type]."""
    if isinstance(tf, Expr) and isinstance(tf.head, Expr) and tf.head.head == "Typed" and len(tf.head.args) == 1 \
            and len(tf.args) == 1:
        ty, f = tf.head.args[0], tf.args[0]
    elif isinstance(tf, Expr) and tf.head == "Typed" and len(tf.args) == 2:
        f, ty = tf.args
    else:
        raise TypeError(f"{fname}: a declaration needs Typed[type]@Function[...], not {_show(tf)}")
    if not (isinstance(f, Expr) and f.head == "Function" and len(f.args) == 2):
        if isinstance(f, Expr) and f.head in ("KernelFunction", "DownValuesFunction") and len(f.args) == 1:
            pass                                    # X01: evaluated by the kernel (see _run_overload)
        elif isinstance(f, Expr) and f.head == "LibraryFunction":
            raise NotImplementedError(f"{fname}: {f.head} declarations are not supported")
        else:
            raise TypeError(f"{fname}: the declared value must be Function[params, body]")
    tvars, cond = (), None
    if isinstance(ty, Expr) and ty.head == "ForAllType":
        if len(ty.args) not in (2, 3):
            raise ValueError(f"{fname}: ForAllType[vars, type] or ForAllType[vars, cond, type]")
        vs = ty.args[0]
        tvars = tuple(_tvar_name(v) for v in (vs.args if is_list(vs) else (vs,)))
        if any(v is None for v in tvars):
            raise TypeError(f"{fname}: ForAllType variables must be symbols")
        cond = ty.args[1] if len(ty.args) == 3 else None
        ty = ty.args[-1]
    t = parse_type(ty, fname, tvars, aliases)
    if not (isinstance(t, tuple) and t[0] == "Fn"):
        raise ValueError(f"{fname}: the type of a declared function must be {{args...}} -> result")
    return tvars, cond, t[1], t[2], f


def _check_cond(cond, b, fname):
    if cond is None:
        return True
    conds = cond.args if is_list(cond) else (cond,)
    for c in conds:
        if isinstance(c, Expr) and c.head in ("Element", "NotElement") and len(c.args) == 2:
            v = b.get(_tvar_name(c.args[0]))
            dom = c.args[1]
            if is_list(dom):
                inside = any(v == parse_type(x, fname) for x in dom.args)
            elif isinstance(dom, str):
                inside = _abstract_member(v, dom, fname)
            else:
                raise ValueError(f"{fname}: unsupported condition {_show(c)}")
            if inside != (c.head == "Element"):
                return False
        elif isinstance(c, Expr) and c.head == "TypeEvaluate":
            r = _eval_literal(c.args[0], b, fname)
            if r != ("Literal", True):
                return False
        else:
            raise ValueError(f"{fname}: unsupported ForAllType condition {_show(c)}")
    return True


_ABSTRACT = {"Integers": set(_INT_RANGES), "SignedIntegers": {t for t in _INT_RANGES if not t.startswith("U")},
             "UnsignedIntegers": {t for t in _INT_RANGES if t.startswith("U")},
             "Reals": set(_INT_RANGES) | {"Real32", "Real64"}, "RealFloatingPoint": {"Real32", "Real64"},
             "Numbers": set(_INT_RANGES) | {"Real32", "Real64", "ComplexReal64"},
             "Complex": {"ComplexReal64"}, "Ordered": set(_INT_RANGES) | {"Real32", "Real64", "String"},
             "Strings": {"String"}}


def _abstract_member(v, dom, fname):
    if dom not in _ABSTRACT:
        raise NotImplementedError(f"{fname}: the abstract type {dom!r} is not supported "
                                  f"({', '.join(sorted(_ABSTRACT))})")
    return v in _ABSTRACT[dom]


def _decl_list(d, fname):
    if d is None:
        return []
    if is_list(d):
        out = []
        for x in d.args:
            out.extend(_decl_list(x, fname))
        return out
    if isinstance(d, Expr) and (d.head in ("FunctionDeclaration", "TypeDeclaration", "OperationDeclaration")
                                or d.head in DECL_HOOKS):
        return [d]
    if isinstance(d, Expr) and d.head == "CompilerEnvironmentObject":
        return list(_env_decls(d, fname))
    raise TypeError(f"{fname}: {_show(d)} is not a declaration")


def build_env(decls, fname):
    """(overloads by name, type aliases) of a list of declarations."""
    aliases = {}
    for d in decls:
        if d.head == "TypeDeclaration":
            kind = d.args[0] if d.args else None
            if kind in ("Macro", "Alias") and len(d.args) >= 3 and isinstance(d.args[1], str):
                aliases[d.args[1]] = parse_type(d.args[2], fname, (), aliases)
            elif ("TypeDeclaration", kind) in DECL_HOOKS:
                DECL_HOOKS[("TypeDeclaration", kind)](d, aliases, fname)
            else:
                raise NotImplementedError(f"{fname}: TypeDeclaration[{_show(kind)}, ...] (product and abstract "
                                          f"types) is not supported in compilation; \"Macro\" and \"Alias\" are")
        elif d.head == "OperationDeclaration":
            raise NotImplementedError(f"{fname}: OperationDeclaration needs product types, which are not supported")
        elif d.head in DECL_HOOKS:
            DECL_HOOKS[d.head](d, aliases, fname)
    funcs = {}
    for d in decls:
        if d.head == "FunctionDeclaration":
            name = d.args[0]
            if not isinstance(name, Symbol):
                raise TypeError(f"{fname}: a declared function name must be a symbol")
            tvars, cond, args, ret, f = _typed_function(d.args[1], fname, aliases)
            funcs.setdefault(name.name, []).append(Overload(name.name, tvars, cond, args, ret, f))
    return funcs, aliases


# ═══════════════════════════════════════ running compiled code ═══════════════════════════════════════

def _prepare(body, env_types, decls, fname):
    """Compiled view of a body: IfCompiled -> compiled branch, TypeHint/Typed -> value, TypeOf -> inferred type."""
    def walk(x):
        if not isinstance(x, Expr):
            return x
        for hook in PREPARE_HOOKS:
            r = hook(x, walk)
            if r is not None:
                return r
        if x.head == "IfCompiled" and len(x.args) == 2:
            return walk(x.args[0])
        if x.head in ("TypeHint", "Typed") and len(x.args) == 2:
            return walk(x.args[0])
        if x.head == "TypeOf" and len(x.args) >= 1:
            return _typespec(infer(x.args[0], env_types, decls, fname))
        if x.head == "Function":
            return x if len(x.args) != 2 else Expr("Function", x.args[0], walk(x.args[1]))
        h = walk(x.head) if isinstance(x.head, Expr) else x.head
        return Expr(h, *(walk(a) for a in x.args))
    return walk(body)


class _Bound:
    """Temporarily give declared names a value (like Block)."""

    def __init__(self, values):
        self.values = values
        self.saved = {}

    def __enter__(self):
        st = K.state()
        for n, v in self.values.items():
            self.saved[n] = st.own.get(n, _MISSING)
            st.own[n] = v
        return self

    def __exit__(self, *exc):
        st = K.state()
        for n, v in self.saved.items():
            if v is _MISSING:
                st.own.pop(n, None)
            else:
                st.own[n] = v
        return False


_MISSING = object()
_DECL_REGISTRY = {}
_COUNTER = itertools.count(1)


def _decl_object(name, overloads, funcs, aliases):
    key = f"decl-{next(_COUNTER)}"
    _DECL_REGISTRY[key] = (overloads, funcs, aliases)
    return Expr("CompiledCodeFunction", _assoc([("Declaration", name), ("ID", key)]))


def _run_overload(name, overloads, args, funcs, aliases, fname):
    overloads = sorted(overloads, key=lambda o: any(isinstance(a, tuple) and a[0] == "Seq" for a in o.args))
    for ov in overloads:
        b = {}
        fixed = [a for a in ov.args if not (isinstance(a, tuple) and a[0] == "Seq")]
        has_seq = len(fixed) != len(ov.args)
        if (not has_seq and len(args) != len(ov.args)) or (has_seq and len(args) < len(fixed)):
            continue
        coerced, ok = [], True
        for i, a in enumerate(args):
            if i < len(ov.args) and not (isinstance(ov.args[i], tuple) and ov.args[i][0] == "Seq"):
                good, c = conform(a, ov.args[i], b, fname)
                if not good:
                    ok = False
                    break
                coerced.append(c)
            else:
                coerced.append(a)
        if not ok or not _check_cond(ov.cond, b, fname):
            continue
        if ov.func.head in ("KernelFunction", "DownValuesFunction"):
            res = machine(_kernel_call(name, ov.func, coerced))
            good, res2 = conform(res, _subst_type(ov.ret, b, fname), b, fname)
            if not good:
                raise TypeError(f"{fname}: {name} returned {_show(res)}, which does not have its declared type "
                                f"{_show(type_expr(_subst_type(ov.ret, b, fname)))}")
            return res2
        params = ov.func.args[0]
        names = [K._untyped(p) for p in (params.args if is_list(params) else (params,))]
        env_types = {}
        for p, t in zip(names, ov.args):
            if isinstance(p, Symbol):
                env_types[p.name] = _subst_type(t, b, fname)
        if has_seq:
            n_fixed = len(names) - 1
            vals = coerced[:n_fixed] + [Expr("Sequence", *coerced[n_fixed:])]
        else:
            vals = coerced
        body = _prepare(ov.func.args[1], env_types, funcs, fname)
        res = machine(ev(K.substitute(body, {p.name: v for p, v in zip(names, vals)})))
        ret = _subst_type(ov.ret, b, fname)
        good, res2 = conform(res, ret, b, fname)
        if not good:
            raise TypeError(f"{fname}: {name} returned {_show(res)}, which does not have its declared type "
                            f"{_show(type_expr(ret))}")
        return res2
    raise TypeError(f"{fname}: no declaration of {name} accepts the arguments {[_show(a) for a in args]}")


def _kernel_call(name, kf, args):
    """X01: a declaration Typed[type]@KernelFunction[f] evaluates f[args] with the evaluator;
    Typed[type]@DownValuesFunction[sym] applies the definitions of sym (the declared name is unbound meanwhile, so
    its own downvalues are used)."""
    f = kf.args[0]
    if kf.head == "DownValuesFunction":
        if not isinstance(f, Symbol):
            raise TypeError("DownValuesFunction: the argument must be a symbol")
        st = K.state()
        saved = st.own.pop(f.name, _MISSING)
        try:
            return ev(Expr(f.name, *args))
        finally:
            if saved is not _MISSING:
                st.own[f.name] = saved
    return ev(Expr(f.name if isinstance(f, Symbol) else f, *args))


def _decl_values(funcs, aliases):
    return {n: _decl_object(n, ovs, funcs, aliases) for n, ovs in funcs.items()}


def _failure(tag, template, params, cf, args):
    return Expr("Failure", tag, _assoc([("MessageTemplate", Expr("MessageName", Symbol("CompiledCodeFunction"),
                                                                     template)),
                                        ("MessageParameters", L(*params)),
                                        ("Symbol", Symbol("CompiledCodeFunction")), ("Function", cf),
                                        ("Arguments", L(*args))]))


def _ccf_parts(cf, fname="CompiledCodeFunction"):
    a = cf.args[0] if cf.args and KL.is_assoc(cf.args[0]) else None
    if a is None or len(cf.args) != 1:
        raise TypeError(f"{fname}: malformed CompiledCodeFunction")
    return a


def _ccf_hook(head, args, ev_):
    if not head.args or not KL.is_assoc(head.args[0]):
        return K.UNHANDLED
    a = head.args[0]
    did = _assoc_get(a, "ID")
    if isinstance(did, str) and did.startswith("decl-"):
        if did not in _DECL_REGISTRY:
            raise ValueError("CompiledCodeFunction: this declaration is no longer available")
        ovs, funcs, aliases = _DECL_REGISTRY[did]
        with _Bound(_decl_values(funcs, aliases)):
            return _run_overload(_assoc_get(a, "Declaration"), ovs, list(args), funcs, aliases, "CompiledCodeFunction")
    return call_ccf(head, list(args))


def call_ccf(cf, args):
    a = _ccf_parts(cf)
    f = _assoc_get(a, "Input")
    decls = _assoc_get(a, "Declarations", L())
    action = _assoc_get(a, "ErrorAction", Symbol("Automatic"))
    sig = _assoc_get(a, "Signature")
    funcs, aliases = build_env(_decl_list(decls, "CompiledCodeFunction"), "CompiledCodeFunction")
    if isinstance(f, Symbol):
        if f.name not in funcs:
            raise ValueError(f"CompiledCodeFunction: {f.name} has no declaration")
        with _Bound(_decl_values(funcs, aliases)):
            return _run_overload(f.name, funcs[f.name], args, funcs, aliases, "CompiledCodeFunction")
    params = f.args[0]
    plist = list(params.args) if is_list(params) else [params]
    if len(args) != len(plist):
        return _failure("ArgumentCount", "argx", [Symbol("CompiledCodeFunction"), len(args)], cf, args)
    env_types, vals = {}, {}
    for i, (p, v) in enumerate(zip(plist, args)):
        t = parse_type(p.args[1], "CompiledCodeFunction", (), aliases)
        ok, c = conform(v, t, {}, "CompiledCodeFunction")
        if not ok:
            return _failure("ArgumentType", "argtype", [i + 1], cf, args)
        env_types[p.args[0].name] = t
        vals[p.args[0].name] = c
    body = _prepare(f.args[1], env_types, funcs, "CompiledCodeFunction")
    with _Bound(_decl_values(funcs, aliases)):
        res = machine(ev(K.substitute(body, vals)))
    ret = None
    if isinstance(sig, Expr) and sig.head == "TypeSpecifier" and sig.args and _is_rule(sig.args[0]):
        try:
            ret = parse_type(sig.args[0].args[1], "CompiledCodeFunction", (), aliases)
        except (ValueError, NotImplementedError):
            ret = None
    if isinstance(ret, str) and ret in _INT_RANGES and K.is_int(res) and not isinstance(res, bool):
        lo, hi = _INT_RANGES[ret]
        if not lo <= int(res) <= hi:
            fail = _failure("RuntimeError", "runtime", ["IntegerOverflow"], cf, args)
            if is_sym(action, "Automatic") or action == "Evaluate":
                return res                    # rerun in the evaluator: the exact result
            if is_sym(action, "None"):
                return fail
            return apply(action, fail)
    return res


KL.COMPOUND_HEAD_HOOKS["CompiledCodeFunction"] = _ccf_hook

_FC_OPTS = {"CompilerEnvironment", "CompilerOptions", "CompilerRuntimeErrorAction", "ProgressReporting",
            "TargetSystem", "UseEmbeddedLibrary"}


def _env_decls(env, fname):
    if is_sym(env, "Automatic"):
        env = ev(Symbol("$CompilerEnvironment"))
    if isinstance(env, Expr) and env.head == "CompilerEnvironmentObject" and env.args and KL.is_assoc(env.args[0]):
        eid = _assoc_get(env.args[0], "ID")
        if eid not in _ENVS:
            raise ValueError(f"{fname}: unknown compiler environment {_show(env)}")
        return list(_ENVS[eid])
    raise ValueError(f"{fname}: CompilerEnvironment must be Automatic or a CompilerEnvironmentObject, "
                     f"not {_show(env)}")


def _compile_one(f, decls, opts, fname):
    env_decls = _env_decls(opts.get("CompilerEnvironment", Symbol("Automatic")), fname)
    all_decls = env_decls + decls
    funcs, aliases = build_env(all_decls, fname)
    action = opts.get("CompilerRuntimeErrorAction", Symbol("Automatic"))
    if isinstance(f, Symbol):
        if f.name not in funcs:
            raise ValueError(f"{fname}: {f.name} is neither a Function nor a declared function")
        ov = funcs[f.name][0]
        sig = Expr("TypeSpecifier", type_expr(("Fn", ov.args, ov.ret)))
    else:
        if not (isinstance(f, Expr) and f.head == "Function" and len(f.args) == 2):
            if isinstance(f, Expr) and f.head == "Function":
                raise ValueError(f"{fname}: pure functions with slots cannot be compiled; use Function[params, body] "
                                 f"with Typed parameters")
            raise TypeError(f"{fname}: expected a Function, not {_show(f)}")
        params = f.args[0]
        plist = list(params.args) if is_list(params) else [params]
        env_types, ts = {}, []
        newp = []
        for p in plist:
            if not (isinstance(p, Expr) and p.head == "Typed" and len(p.args) == 2 and isinstance(p.args[0], Symbol)):
                raise ValueError(f"{fname}: the parameter {_show(p)} needs a type: use Typed[{_show(p)}, type]")
            ty = ev(p.args[1])
            t = parse_type(ty, fname, (), aliases)
            env_types[p.args[0].name] = t
            ts.append(t)
            newp.append(Expr("Typed", p.args[0], ty))
        f = Expr("Function", L(*newp) if is_list(params) else newp[0], f.args[1])
        try:
            ret = infer(f.args[1], env_types, funcs, fname)
        except (NotImplementedError, ValueError):
            ret = "InertExpression"
        sig = Expr("TypeSpecifier", type_expr(("Fn", tuple(ts), ret)))
    if not (is_sym(action, "Automatic") or action == "Evaluate" or is_sym(action, "None")) and \
            not (isinstance(action, (Symbol, Expr))):
        raise ValueError(f"{fname}: invalid CompilerRuntimeErrorAction {_show(action)}")
    return Expr("CompiledCodeFunction", _assoc([("Signature", sig), ("Input", f), ("Declarations", L(*all_decls)),
                                                ("ErrorAction", action)]))


@fn("FunctionCompile", area=AREA, status="added",
    notes="FunctionCompile[f], FunctionCompile[{f1, ...}], FunctionCompile[<|k -> f, ...|>], FunctionCompile[decls, "
          "f] and FunctionCompile[decls, name] give CompiledCodeFunction objects; f is Function[{Typed[x, type], "
          "...}, body] with types \"Integer8\"...\"Integer128\", \"UnsignedInteger8\"...\"UnsignedInteger64\", "
          "\"MachineInteger\", \"Real32\", \"Real64\", \"ComplexReal64\", \"Boolean\", \"String\", "
          "\"InertExpression\", TypeSpecifier[\"PackedArray\"][t, r], TypeSpecifier[\"ListVector\"][t], function "
          "types {...} -> t; declarations are FunctionDeclaration (with ForAllType, SequenceType, LiteralType, "
          "TypeEvaluate) and TypeDeclaration[\"Macro\"|\"Alias\", ...]. Calling checks each argument against its "
          "type (integers are accepted for reals) and gives the documented Failure[\"ArgumentCount\", ...] or "
          "Failure[\"ArgumentType\", ...] otherwise; the body is evaluated with IfCompiled taking its compiled "
          "branch, TypeHint/Typed giving their value and TypeOf the inferred type; exact non-integer results become "
          "machine numbers. Integer overflow of an inferred integer result type follows CompilerRuntimeErrorAction "
          "(Automatic/\"Evaluate\": the exact result, as Wolfram's rerun gives; None: Failure[\"RuntimeError\", "
          "...]; f: f[failure]). Options CompilerEnvironment, CompilerOptions, ProgressReporting, TargetSystem and "
          "UseEmbeddedLibrary are accepted (no machine code is generated). Product types, KernelFunction, "
          "DownValuesFunction, KernelEvaluate, InertExpression, CompiledExpressionDeclaration and raw pointers "
          "are handled by the X01 extensions (wl_core_extras_compile); abstract types and "
          "OperationDeclaration raise NotImplementedError; untyped "
          "parameters ValueError. Deviation: the \"PackedArray\"::[...] input syntax is not parsed; write "
          "TypeSpecifier[\"PackedArray\"][...]. Check: reference (Wolfram documentation examples) and agreement "
          "with the uncompiled Function in tests.")
def _functioncompile(*args: Held) -> Expr:
    args = list(args)
    pos, opts = _split_opts(args, _FC_OPTS, "FunctionCompile")
    if not pos or len(pos) > 2:
        raise TypeError("FunctionCompile: expected FunctionCompile[f] or FunctionCompile[decls, f]")
    for k in ("CompilerOptions",):
        if k in opts and not (is_sym(opts[k], "Automatic") or KL.is_assoc(opts[k])):
            raise ValueError("FunctionCompile: CompilerOptions must be Automatic or an association")
    decls = []
    if len(pos) == 2:
        decls = _decl_list(ev(pos[0]), "FunctionCompile")
    spec = pos[-1]
    if isinstance(spec, Symbol) and spec.name in K.state().own:
        v = ev(spec)
        if isinstance(v, Expr) and v.head in ("Function", "List", "Association"):
            spec = v
    if is_list(spec):
        return L(*(_compile_one(f, decls, opts, "FunctionCompile") for f in spec.args))
    if isinstance(spec, Expr) and spec.head == "Association":
        return _assoc([(r.args[0], _compile_one(r.args[1], decls, opts, "FunctionCompile")) for r in spec.args])
    return _compile_one(spec, decls, opts, "FunctionCompile")


@fn("CompiledCodeFunction", area=AREA, status="added",
    notes="CompiledCodeFunction[<|\"Signature\" -> TypeSpecifier[{types} -> type], \"Input\" -> f, "
          "\"Declarations\" -> {...}, \"ErrorAction\" -> ...|>] is made by FunctionCompile (\"InertExpression\" "
          "stands for a result type that could not be inferred) and applied like the function: arguments of the "
          "wrong type or number give the documented Failure[\"ArgumentType\"|\"ArgumentCount\", <|...|>]. A "
          "malformed object raises TypeError. Check: reference (Wolfram documentation) and agreement with the "
          "uncompiled function in tests.")
def _ccf(a: Assoc) -> Expr:
    if not KL.is_assoc(a):
        if isinstance(a, Symbol):
            raise Unevaluated()
        raise TypeError("CompiledCodeFunction: expected an association made by FunctionCompile")
    if _assoc_get(a, "Input") is None and _assoc_get(a, "ID") is None:
        raise ValueError("CompiledCodeFunction: missing \"Input\" function")
    raise Unevaluated()


# ═══════════════════════════════════════ Compile (the classic compiler) ═══════════════════════════════════════

_C_TYPES = {"Real": "Real64", "Integer": "Integer64", "Complex": "ComplexReal64"}
_C_OPTS = {"CompilationOptions", "CompilationTarget", "Parallelization", "RuntimeAttributes", "RuntimeOptions"}
_CO_KEYS = {"ExpressionOptimization", "InlineCompiledFunctions", "InlineExternalDefinitions"}


def _blank_type(t):
    if isinstance(t, Expr) and t.head == "Blank" and len(t.args) == 1 and isinstance(t.args[0], Symbol) \
            and t.args[0].name in _C_TYPES:
        return t.args[0].name
    if isinstance(t, Expr) and t.head == "Alternatives" and len(t.args) == 2 and set(map(str, t.args)) == \
            {"True", "False"} or (isinstance(t, Expr) and t.head == "Alternatives" and t.args == (True, False)):
        return "Boolean"
    return None


def _compile_spec(v):
    """(name, type, rank) of a Compile variable specification."""
    if isinstance(v, Symbol):
        return v.name, "Real", 0
    if is_list(v) and v.args and isinstance(v.args[0], Symbol):
        if len(v.args) == 1:
            return v.args[0].name, "Real", 0
        t = _blank_type(v.args[1])
        if t is None:
            raise ValueError(f"Compile: the type {_show(v.args[1])} is not _Integer, _Real, _Complex or True | False")
        rank = 0
        if len(v.args) == 3:
            r = v.args[2]
            if not (K.is_int(r) and not isinstance(r, bool) and int(r) >= 0):
                raise ValueError("Compile: the rank must be a non-negative integer")
            rank = int(r)
        elif len(v.args) > 3:
            raise ValueError(f"Compile: malformed variable specification {_show(v)}")
        return v.args[0].name, t, rank
    raise TypeError(f"Compile: malformed variable specification {_show(v)}")


def _target(opt):
    if opt is None or is_sym(opt, "Automatic"):
        t = ev(Symbol("$CompilationTarget"))
        if t not in ("WVM", "C"):
            raise ValueError(f"Compile: $CompilationTarget must be \"WVM\" or \"C\", not {_show(t)}")
        return t
    if opt in ("WVM", "C"):
        return opt
    raise ValueError(f"Compile: CompilationTarget must be \"WVM\" or \"C\", not {_show(opt)}")


@fn("Compile", area=AREA, status="added",
    notes="Compile[{x1, ...}, expr], Compile[{{x, _Integer|_Real|_Complex|True | False}, ...}, expr] and "
          "Compile[{{x, t, rank}, ...}, expr] (HoldAll) give CompiledFunction[...]; options CompilationOptions "
          "({\"ExpressionOptimization\" | \"InlineCompiledFunctions\" | \"InlineExternalDefinitions\" -> "
          "True|False|Automatic}), CompilationTarget (\"WVM\" or \"C\"; default $CompilationTarget), "
          "Parallelization, RuntimeAttributes ({} or {Listable}) and RuntimeOptions are validated (no machine code "
          "is generated, so \"C\" needs no C compiler and gives the same results). Applying the CompiledFunction "
          "to arguments of the declared types (integers are accepted for reals, full arrays of the declared rank) "
          "evaluates expr with them and gives machine numbers (integers stay integers); other arguments, e.g. "
          "symbolic ones, use ordinary evaluation as Wolfram documents. A wrong argument count raises ValueError; "
          "malformed specifications ValueError/TypeError. Check: reference (Wolfram documentation examples) and "
          "agreement with N of the uncompiled expression in tests.")
def _compile(*args: Held) -> Expr:
    args = list(args)
    if len(args) < 2:
        raise TypeError("Compile: expected Compile[{vars...}, expr, options]")
    vars_, body, rest = args[0], args[1], args[2:]
    _pos, opts = _split_opts(rest, _C_OPTS, "Compile")
    if _pos:
        raise ValueError("Compile: the third argument (external function types) is not supported")
    if not is_list(vars_):
        raise TypeError("Compile: the variables must be given as a list")
    specs = [_compile_spec(v) for v in vars_.args]
    names = [s[0] for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("Compile: duplicate variable names")
    co = opts.get("CompilationOptions", Symbol("Automatic"))
    if not is_sym(co, "Automatic"):
        items = co.args if is_list(co) else (co,)
        for r in items:
            if not (_is_rule(r) and r.args[0] in _CO_KEYS):
                raise ValueError(f"Compile: unknown CompilationOptions setting {_show(r)}")
            if not (r.args[1] is True or r.args[1] is False or is_sym(r.args[1], "Automatic")):
                raise ValueError(f"Compile: {r.args[0]} must be True, False or Automatic")
    target = _target(opts.get("CompilationTarget"))
    ra = opts.get("RuntimeAttributes", L())
    ral = ra.args if is_list(ra) else (ra,)
    for x in ral:
        if not is_sym(x, "Listable"):
            raise ValueError(f"Compile: unsupported RuntimeAttributes {_show(ra)} (only Listable)")
    par = opts.get("Parallelization", Symbol("Automatic"))
    if not (par is True or par is False or is_sym(par, "Automatic") or KL.is_assoc(par)):
        raise ValueError("Compile: Parallelization must be True, False, Automatic or an association")
    if "Evaluate" == getattr(body, "head", None) and len(body.args) == 1:
        body = ev(body.args[0])
    types = L(*(L(Expr("Blank", Symbol(t)) if t != "Boolean" else Expr("Alternatives", True, False), r)
                for _n, t, r in specs))
    meta = _assoc([("CompilationTarget", target), ("RuntimeAttributes", L(*ral))])
    return Expr("CompiledFunction", Expr("HoldComplete", L(*(Symbol(n) for n in names))), types,
                Expr("HoldComplete", body), meta)


def _cf_parts(cf):
    a = cf.args
    if len(a) != 4 or not (isinstance(a[0], Expr) and a[0].head == "HoldComplete" and is_list(a[0].args[0])) or \
            not is_list(a[1]) or not (isinstance(a[2], Expr) and a[2].head == "HoldComplete") or not KL.is_assoc(a[3]):
        raise TypeError("CompiledFunction: malformed CompiledFunction (make one with Compile)")
    names = [s.name for s in a[0].args[0].args]
    specs = []
    for t in a[1].args:
        tt = _blank_type(t.args[0])
        specs.append((tt, int(t.args[1])))
    return names, specs, a[2].args[0], a[3]


def _c_conform(v, t, rank):
    if rank > 0:
        if not is_list(v) or not v.args:
            return False, v
        out = []
        n = None
        for e in v.args:
            ok, c = _c_conform(e, t, rank - 1)
            if not ok:
                return False, v
            m = len(e.args) if is_list(e) else None
            if n is not None and m != n:
                return False, v
            n = m
            out.append(c)
        return True, L(*out)
    if t == "Boolean":
        return (v is True or v is False), v
    if t == "Integer":
        return (K.is_int(v) and not isinstance(v, bool)), v
    nv = _num(v)
    if nv is None:
        return False, v
    if t == "Real":
        if isinstance(nv, complex):
            return False, v
        return True, float(nv)
    return True, (complex(nv) if not isinstance(nv, complex) else nv)


def _cf_hook(head, args, ev_):
    if len(head.args) != 4:
        return K.UNHANDLED
    names, specs, body, meta = _cf_parts(head)
    args = list(args)
    if len(args) != len(names):
        raise ValueError(f"CompiledFunction: called with {len(args)} argument(s); {len(names)} expected")
    listable = any(is_sym(x, "Listable") for x in _assoc_get(meta, "RuntimeAttributes", L()).args)
    if listable:
        for i, (a, (t, r)) in enumerate(zip(args, specs)):
            if is_list(a) and _depth(a) > r:
                return L(*(_cf_hook(head, args[:i] + [e] + args[i + 1:], ev_) for e in a.args))
    vals, compiled = {}, True
    for n, a, (t, r) in zip(names, args, specs):
        ok, c = _c_conform(a, t, r)
        if not ok:
            compiled = False
            vals[n] = a
        else:
            vals[n] = c
    res = ev(K.substitute(body, vals if compiled else {n: a for n, a in zip(names, args)}))
    return machine(res) if compiled else res


KL.COMPOUND_HEAD_HOOKS["CompiledFunction"] = _cf_hook


@fn("CompiledFunction", area=AREA, status="added",
    notes="CompiledFunction[HoldComplete[{vars}], {{type, rank}, ...}, HoldComplete[body], <|...|>] is made by "
          "Compile (the representation differs from Wolfram's register code) and applied to arguments: declared "
          "types are checked and coerced (_Real accepts integers and exact reals, _Complex any number, _Integer "
          "integers only, True | False Booleans, rank-n full arrays), results are machine numbers, and "
          "non-conforming arguments fall back to ordinary evaluation (documented behaviour); RuntimeAttributes -> "
          "Listable threads over extra list levels. A wrong argument count raises ValueError and a malformed "
          "object TypeError. Check: reference (Wolfram documentation examples) and agreement with N of the "
          "uncompiled expression in tests.")
def _compiledfunction(*args: Expr) -> Expr:
    _cf_parts(Expr("CompiledFunction", *args))
    raise Unevaluated()


E.SYSTEM_VARIABLES["$CompilationTarget"] = lambda: "WVM"
record("$CompilationTarget", AREA, "added",
       "$CompilationTarget is \"WVM\" (the documented default) and gives the default CompilationTarget of Compile; "
       "Block[{$CompilationTarget = \"C\"}, ...] changes it (results are identical since no machine code is "
       "generated); other values make Compile raise ValueError. Check: reference (Wolfram documentation) in "
       "tests.")
for _n, _d in {
    "CompilationTarget": "option of Compile: \"WVM\" or \"C\" (default $CompilationTarget); recorded in the "
                         "CompiledFunction, both evaluate the same way (no external C compiler is used); other "
                         "values raise ValueError. Check: reference (Wolfram documentation example values) in tests.",
    "CompilationOptions": "option of Compile: Automatic or a list of \"ExpressionOptimization\", "
                          "\"InlineCompiledFunctions\", \"InlineExternalDefinitions\" -> True|False|Automatic "
                          "(validated; they only affect generated code, so results are unchanged); other settings "
                          "raise ValueError. Check: reference (Wolfram documentation) in tests.",
    "CompilerOptions": "option of FunctionCompile: Automatic or an association of pipeline settings (accepted; "
                       "there is no compilation pipeline, so results are unchanged); other values raise "
                       "ValueError. Check: reference (Wolfram documentation) in tests.",
    "CompilerRuntimeErrorAction": "option of FunctionCompile for low-level runtime errors (integer overflow of the "
                                  "inferred integer result type): Automatic or \"Evaluate\" give the exact "
                                  "evaluated result (Wolfram's rerun), None gives Failure[\"RuntimeError\", <|..., "
                                  "\"MessageParameters\" -> {\"IntegerOverflow\"}|>], f gives f[failure]. Check: "
                                  "reference (Wolfram documentation UnsignedInteger8 example) in tests.",
    "Compiled": "option of numerical functions (NIntegrate, NSum, FindRoot, ... in wl_numcalc) saying whether to "
                "compile the integrand: accepted with True or False; MikoshiLang evaluates directly, so results "
                "are the same either way. Check: reference (Wolfram documentation) and identical results with "
                "True and False in tests.",
    "AssumeDeterministic": "option of BayesianMinimization/BayesianMaximization (wl_linalg_optim): True assumes "
                           "the objective gives the same value for the same configuration (no noise term), False "
                           "(the default) does not. Check: reference (Wolfram documentation) and the optimiser "
                           "result with both settings in tests.",
}.items():
    record(_n, AREA, "added", _d)


# ═══════════════════════════════════════ compiler environments ═══════════════════════════════════════

_ENVS = {"Default": []}
_ENV_COUNTER = itertools.count(1)


def _default_env():
    return Expr("CompilerEnvironmentObject", _assoc([("ID", "Default")]))


E.SYSTEM_VARIABLES["$CompilerEnvironment"] = _default_env
record("$CompilerEnvironment", AREA, "added",
       "$CompilerEnvironment is the CompilerEnvironmentObject FunctionCompile uses by default (initially "
       "CompilerEnvironmentObject[<|\"ID\" -> \"Default\"|>], empty); CompilerEnvironmentAppendTo[decls] adds to "
       "it and $CompilerEnvironment = CreateCompilerEnvironment[] replaces it. The environment lives for the "
       "Python process. Check: reference (Wolfram documentation examples) in tests.")


def _env_obj(env, fname):
    if isinstance(env, Expr) and env.head == "CompilerEnvironmentObject" and len(env.args) == 1 and \
            KL.is_assoc(env.args[0]):
        eid = _assoc_get(env.args[0], "ID")
        if eid in _ENVS:
            return eid
        raise ValueError(f"{fname}: unknown compiler environment {_show(env)}")
    if isinstance(env, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a CompilerEnvironmentObject, not {_show(env)}")


@fn("CreateCompilerEnvironment", area=AREA, status="added",
    notes="CreateCompilerEnvironment[] gives a new, empty CompilerEnvironmentObject[<|\"ID\" -> n|>]; the "
          "TargetSystem option is accepted (Inherited, Automatic, All, a system name or a list) and stored, since "
          "no machine code is generated. Other options raise ValueError. Check: reference (Wolfram documentation) "
          "and environments kept separate in tests.")
def _createenv(*opts: Option) -> Expr:
    _p, o = _split_opts(opts, {"TargetSystem"}, "CreateCompilerEnvironment")
    if _p:
        raise TypeError("CreateCompilerEnvironment: expected CreateCompilerEnvironment[opts]")
    eid = next(_ENV_COUNTER)
    _ENVS[eid] = []
    items = [("ID", eid)]
    if "TargetSystem" in o:
        ts = o["TargetSystem"]
        names = [ts] if isinstance(ts, str) else (list(ts.args) if is_list(ts) else None)
        if names is None and not any(is_sym(ts, n) for n in ("Automatic", "All", "Inherited")):
            raise ValueError("CreateCompilerEnvironment: TargetSystem must be a name, a list, Automatic, All or "
                             "Inherited")
        for n in names or ():
            if n not in _TARGETS:
                raise ValueError(f"CreateCompilerEnvironment: unknown TargetSystem {_show(n)} (see $TargetSystems)")
        items.append(("TargetSystem", ts if not isinstance(ts, str) else L(ts)))
    return Expr("CompilerEnvironmentObject", _assoc(items))


@fn("CompilerEnvironmentObject", area=AREA, status="added",
    notes="CompilerEnvironmentObject[<|\"ID\" -> id, ...|>] is a collection of declarations made by "
          "CreateCompilerEnvironment (or $CompilerEnvironment); env[\"Declarations\"] lists them and "
          "env[TargetSystem] gives the stored target list ({} when none). An unknown id raises ValueError. "
          "Check: exact (registry contents) in tests.")
def _envobj(a: Assoc) -> Expr:
    _env_obj(Expr("CompilerEnvironmentObject", a), "CompilerEnvironmentObject")
    raise Unevaluated()


def _env_hook(head, args, ev_):
    if len(args) != 1 or len(head.args) != 1:
        return K.UNHANDLED
    eid = _env_obj(head, "CompilerEnvironmentObject")
    p = args[0]
    if p == "Declarations":
        return L(*_ENVS[eid])
    if is_sym(p, "TargetSystem"):
        return _assoc_get(head.args[0], "TargetSystem", L())
    raise ValueError(f"CompilerEnvironmentObject: unknown property {_show(p)} (\"Declarations\", TargetSystem)")


KL.COMPOUND_HEAD_HOOKS["CompilerEnvironmentObject"] = _env_hook


@fn("CompilerEnvironmentAppendTo", area=AREA, status="added",
    notes="CompilerEnvironmentAppendTo[decls] appends FunctionDeclaration/TypeDeclaration/OperationDeclaration "
          "objects (one or a list) to $CompilerEnvironment, CompilerEnvironmentAppendTo[env, decls] to env; gives "
          "the environment. Declarations are validated when appended (bad types raise ValueError, unsupported "
          "kinds NotImplementedError when used). Check: reference (Wolfram documentation AddTwo examples) in "
          "tests.")
def _envappend(a: Expr, b: Expr = None) -> Expr:
    if b is None:
        env, decls = ev(Symbol("$CompilerEnvironment")), a
    else:
        env, decls = a, b
    eid = _env_obj(env, "CompilerEnvironmentAppendTo")
    ds = _decl_list(decls, "CompilerEnvironmentAppendTo")
    for d in ds:
        _validate_decl(d, "CompilerEnvironmentAppendTo")
    _ENVS[eid].extend(ds)
    return env


# ═══════════════════════════════════════ declarations and type objects ═══════════════════════════════════════

def _validate_decl(d, fname):
    if d.head == "FunctionDeclaration":
        if len(d.args) < 2 or not isinstance(d.args[0], Symbol):
            raise TypeError(f"{fname}: FunctionDeclaration[name, Typed[type]@Function[...]]")
        for o in d.args[2:]:
            if not (_is_rule(o) and o.args[0] == "Inline" and o.args[1] in ("Hint", "Never", "Always")
                    or (_is_rule(o) and o.args[0] == "Inline" and is_sym(o.args[1], "Automatic"))):
                raise ValueError(f"{fname}: unknown FunctionDeclaration option {_show(o)}")
        _typed_function(d.args[1], fname, {})
    elif d.head == "TypeDeclaration":
        if not d.args or d.args[0] not in ("Product", "Abstract", "Alias", "Macro"):
            raise ValueError(f"{fname}: the kind of a TypeDeclaration must be \"Product\", \"Abstract\", \"Alias\" "
                             f"or \"Macro\"")
        if len(d.args) < 2 or not isinstance(d.args[1], str):
            raise TypeError(f"{fname}: a declared type name must be a string")
        if d.args[0] in ("Macro", "Alias"):
            if len(d.args) < 3:
                raise ValueError(f"{fname}: TypeDeclaration[\"{d.args[0]}\", name, type]")
            parse_type(d.args[2], fname)
        if d.args[0] == "Product":
            if len(d.args) < 3 or not KL.is_assoc(d.args[2]):
                raise ValueError(f"{fname}: TypeDeclaration[\"Product\", name, <|field -> type, ...|>]")
            for r in d.args[2].args:
                if not isinstance(r.args[0], str):
                    raise TypeError(f"{fname}: product field names must be strings")
                try:
                    parse_type(r.args[1], fname, (), {d.args[1]: "InertExpression"})
                except NotImplementedError:
                    pass
    elif d.head == "OperationDeclaration":
        if len(d.args) != 3 or not isinstance(d.args[0], str) or not isinstance(d.args[1], str):
            raise TypeError(f"{fname}: OperationDeclaration[\"type\", \"op\", Typed[...]@Function[...]]")
        _typed_function(d.args[2], fname, {d.args[0]: "InertExpression"})


def _inert_decl(head):
    def impl(*args: Expr) -> Expr:
        _validate_decl(Expr(head, *args), head)
        raise Unevaluated()
    return impl


fn("FunctionDeclaration", area=AREA, status="added",
   notes="FunctionDeclaration[name, Typed[{t1, ...} -> t]@Function[{x1, ...}, body], \"Inline\" -> ...] is a "
         "validated, inert declaration used by FunctionCompile and CompilerEnvironmentAppendTo; several "
         "declarations of one name with different types are overloads, ForAllType makes them polymorphic and "
         "SequenceType variadic; Typed[type]@KernelFunction[f] and Typed[type]@DownValuesFunction[sym] declare "
         "functions evaluated by the kernel (X01). Malformed declarations raise TypeError/ValueError. Check: reference (Wolfram documentation AddTwo, "
         "VarArgsAdd and ForAllType examples) in tests.")(_inert_decl("FunctionDeclaration"))
fn("TypeDeclaration", area=AREA, status="added",
   notes="Partial. TypeDeclaration[kind, name, ...] is a validated, inert declaration (kinds \"Product\", "
         "\"Abstract\", \"Alias\", \"Macro\"); FunctionCompile uses \"Macro\" and \"Alias\" declarations as type "
         "names and \"Product\" types with CreateTypeInstance (X01), and raises NotImplementedError for abstract "
         "types. Malformed declarations raise ValueError/TypeError. Check: reference (Wolfram documentation "
         "fooInteger example) in tests.")(_inert_decl("TypeDeclaration"))
fn("OperationDeclaration", area=AREA, status="added",
   notes="Partial. OperationDeclaration[\"type\", \"op\", Typed[...]@Function[...]] is a validated, inert "
         "declaration; using it in FunctionCompile raises NotImplementedError because product types are not "
         "supported. Malformed declarations raise TypeError. Check: reference (Wolfram documentation form) in "
         "tests.")(_inert_decl("OperationDeclaration"))


def _typed_hook(head, args, ev_):
    if len(head.args) == 1 and len(args) == 1:          # Typed[type][f] -> Typed[f, type]
        return Expr("Typed", args[0], head.args[0])
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["Typed"] = _typed_hook


@fn("Typed", area=AREA, status="added",
    notes="Typed[x, type] annotates x with a compiler type and stays as it is when evaluated (Function[{Typed[x, "
          "t]}, body] binds x like a plain parameter, as documented; FunctionCompile uses the types); Typed[type] "
          "is an operator form: Typed[type][f] gives Typed[f, type]. The type is validated structurally (a string "
          "name, which may come from a TypeDeclaration, a function type, a compound or literal type); anything else "
          "raises ValueError. Check: reference (Wolfram documentation: Function[{Typed[x, \"Real64\"]}, x + 1][\"two\"] "
          "gives 1 + \"two\") in tests.")
def _typed(x: Expr, t: Expr = None) -> Expr:
    ty = x if t is None else t
    if not (isinstance(ty, Expr) and ty.head == "ForAllType"):
        parse_type(ty, "Typed", _free_tvars(ty), _AnyName())
    raise Unevaluated()


def _free_tvars(t):
    """Symbols used as types inside t (allowed as type variables in declarations)."""
    out = set()

    def walk(x):
        if isinstance(x, Symbol):
            out.add(x.name)
        elif isinstance(x, Expr):
            for a in x.args:
                walk(a)
    walk(t)
    return tuple(out)


@fn("TypeSpecifier", area=AREA, status="added",
    notes="TypeSpecifier[type] represents a type (as TypeOf returns it) and TypeSpecifier[cons][t1, ...] a compound "
          "type (\"PackedArray\", \"NumericArray\", \"ListVector\" here; Tabular types such as "
          "TypeSpecifier[\"Date\"][...] belong to the Tabular functions and are left alone); both stay as they "
          "are. A malformed native type inside TypeSpecifier[type] raises ValueError. Check: reference (Wolfram "
          "documentation) and FunctionCompile argument checks in tests.")
def _typespecifier(t: Expr) -> Expr:
    if isinstance(t, str) and t not in _COMPOUND and not t[:1].isupper():
        raise ValueError(f"TypeSpecifier: {t!r} is not a type name")
    if isinstance(t, (Expr, str)) and not (isinstance(t, str) and (t in _COMPOUND or t not in NATIVE)):
        try:
            parse_type(t, "TypeSpecifier", _free_tvars(t), _AnyName())
        except NotImplementedError:
            pass
    raise Unevaluated()


@fn("TypeHint", area=AREA, status="added",
    notes="TypeHint[expr, type] gives expr when evaluated (the type, validated, only matters in compiled code, "
          "where FunctionCompile also gives expr); an unknown type raises ValueError. Check: reference (Wolfram "
          "documentation example TypeHint[1, \"UnsignedInteger8\"]) in tests.")
def _typehint(x: Expr, t: Expr) -> Expr:
    parse_type(t, "TypeHint")
    return x


@fn("IfCompiled", area=AREA, status="added",
    notes="IfCompiled[comp, uncomp] (HoldAll) evaluates uncomp when evaluated and comp inside FunctionCompile "
          "code, never both. Check: reference (Wolfram documentation example: \"uncompiled\" evaluated, "
          "\"compiled\" from FunctionCompile) in tests.")
def _ifcompiled(comp: Held, uncomp: Held) -> Expr:
    return ev(uncomp)


@fn("TypeOf", area=AREA, status="added",
    notes="TypeOf[expr], TypeOf[expr, Typed[x, t]], TypeOf[expr, {Typed[x, t], ...}] and TypeOf[expr, decls] "
          "(HoldFirst) infer the compiler type of expr without evaluating it and give TypeSpecifier[type]: "
          "literals (integers Integer64, reals Real64, complexes ComplexReal64, strings, Booleans), typed "
          "variables, arithmetic with the usual promotion (integer division and elementary functions give "
          "Real64), comparisons and logic (Boolean), StringLength/Length (Integer64), If, Module/With, lists, "
          "Range and Table (PackedArray), Dot, Total, Function with typed parameters ({...} -> t) and calls to "
          "declared functions. Expressions outside this subset raise NotImplementedError; bad types ValueError. "
          "Check: reference (Wolfram documentation examples) and the types of FunctionCompile results in tests.")
def _typeof(e: Held, decls: Held = None) -> Expr:
    env, funcs = {}, {}
    if decls is not None:
        d = ev(decls)
        items = d.args if is_list(d) else (d,)
        fdecls = []
        for x in items:
            if isinstance(x, Expr) and x.head == "Typed" and len(x.args) == 2 and isinstance(x.args[0], Symbol):
                env[x.args[0].name] = parse_type(x.args[1], "TypeOf")
            elif isinstance(x, Expr) and x.head in ("FunctionDeclaration", "TypeDeclaration"):
                fdecls.append(x)
            else:
                raise TypeError(f"TypeOf: {_show(x)} is not Typed[x, type] or a declaration")
        funcs, _aliases = build_env(fdecls, "TypeOf")
    return _typespec(infer(e, env, funcs, "TypeOf"))


def _inert_type(head, check):
    def impl(*args: Expr) -> Expr:
        check(args)
        raise Unevaluated()
    return impl


def _chk_literal(args):
    if len(args) != 1:
        raise TypeError("LiteralType: expected LiteralType[x]")
    v = args[0]
    if not (isinstance(v, (str, bool)) or (K.is_int(v))):
        if isinstance(v, Symbol):
            raise Unevaluated()
        raise TypeError("LiteralType: the literal must be an integer, a string or a Boolean")


def _chk_forall(args):
    if len(args) not in (2, 3):
        raise TypeError("ForAllType: expected ForAllType[x, type] or ForAllType[x, cond, type]")
    vs = args[0].args if is_list(args[0]) else (args[0],)
    if not all(_tvar_name(v) for v in vs):
        raise TypeError("ForAllType: the type variables must be symbols")
    parse_type(args[-1], "ForAllType", tuple(_tvar_name(v) for v in vs))


def _chk_seq(args):
    if len(args) != 1 or _tvar_name(args[0]) is None:
        raise TypeError("SequenceType: expected SequenceType[var] with a symbol var")


def _chk_eval(args):
    if len(args) != 1:
        raise TypeError("TypeEvaluate: expected TypeEvaluate[expr]")


fn("LiteralType", area=AREA, status="added",
   notes="LiteralType[x] (x an integer, string or Boolean) is an inert literal type, e.g. the rank in "
         "TypeSpecifier[\"PackedArray\"][\"MachineInteger\", LiteralType[2]] or a string argument type in a "
         "declaration; other x raise TypeError, a symbol stays unevaluated. Check: reference (Wolfram documentation "
         "examples) in tests.")(_inert_type("LiteralType", _chk_literal))
fn("ForAllType", area=AREA, status="added",
   notes="ForAllType[x, type], ForAllType[x, cond, type] and ForAllType[{x1, ...}, cond, type] are inert "
         "polymorphic types for declarations: FunctionCompile binds the variables from the argument types and "
         "checks cond (Element/NotElement with a list of types or the abstract types \"Integers\", "
         "\"SignedIntegers\", \"UnsignedIntegers\", \"Reals\", \"RealFloatingPoint\", \"Numbers\", \"Complex\", "
         "\"Ordered\", \"Strings\", or TypeEvaluate[...]). Malformed forms raise TypeError/ValueError. Check: "
         "reference (Wolfram documentation polyFunc and addtwo examples) in tests.")(_inert_type("ForAllType", _chk_forall))
fn("SequenceType", area=AREA, status="added",
   notes="SequenceType[var] is an inert type for zero or more trailing arguments in a declaration (FunctionCompile "
         "passes them on as a Sequence, as in Wolfram's VarArgsAdd example); var must be a symbol (TypeError "
         "otherwise). Check: reference (Wolfram documentation VarArgsAdd example) in tests.")(_inert_type("SequenceType", _chk_seq))
fn("TypeEvaluate", area=AREA, status="added",
   notes="TypeEvaluate[expr] is an inert type computed by evaluating expr with the type variables of the "
         "enclosing ForAllType replaced by their literal values (e.g. a rank r + 1), also usable as a ForAllType "
         "condition; a result that is not a literal or TypeSpecifier raises ValueError. Check: reference (Wolfram "
         "documentation addRank example) in tests.")(_inert_type("TypeEvaluate", _chk_eval))


K.BUILTIN_ATTRIBUTES["Compile"] = {"HoldAll", "Protected"}
K.BUILTIN_ATTRIBUTES["IfCompiled"] = {"HoldAll", "Protected"}
K.BUILTIN_ATTRIBUTES["TypeOf"] = {"HoldFirst", "Protected"}


_TARGETS = ("Linux-ARM", "Linux-x86-64", "MacOSX-ARM64", "MacOSX-x86-64", "Windows-x86-64")
E.SYSTEM_VARIABLES["$TargetSystems"] = lambda: L(*_TARGETS)
record("$TargetSystems", AREA, "added",
       "$TargetSystems lists the documented TargetSystem names {\"Linux-ARM\", \"Linux-x86-64\", "
       "\"MacOSX-ARM64\", \"MacOSX-x86-64\", \"Windows-x86-64\"}; they are accepted by "
       "CreateCompilerEnvironment and FunctionCompile (no machine code is generated for any of them). Check: "
       "reference (Wolfram documentation output) in tests.")
record("TargetSystem", AREA, "added",
       "option of CreateCompilerEnvironment and FunctionCompile naming the machine architectures to target: "
       "Inherited/Automatic/All, a name from $TargetSystems or a list of them; CreateCompilerEnvironment stores it "
       "(env[TargetSystem] gives the list) and other strings raise ValueError. No code is generated, so results do "
       "not depend on it. Check: reference (Wolfram documentation: env[TargetSystem] gives {\"Windows-x86-64\"}) "
       "in tests.")
record("CompilerEnvironment", AREA, "added",
       "option of FunctionCompile: Automatic (use $CompilerEnvironment) or a CompilerEnvironmentObject whose "
       "declarations are available to the compiled function; other values raise ValueError. Check: reference "
       "(Wolfram documentation AddTwo example) in tests.")
