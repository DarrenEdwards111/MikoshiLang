"""Wolfram batch X01 (part 5): compile constructs for the interpreted FunctionCompile of wl_external_compile.

FunctionCompile runs the body of a compiled function with the evaluator after type checking the arguments (see
wl_external_compile); the constructs here extend that compiled view through its PREPARE_HOOKS (rewrites applied to
the body when it is run), DECL_HOOKS (declaration kinds) and TYPE_CONFORMERS (type kinds):

* KernelFunction[f], DownValuesFunction[sym] and KernelEvaluate[...]: evaluation by the kernel from compiled code;
* InertExpression and InertEvaluate: "InertExpression" values, held unevaluated by the internal head
  Compile`InertValue, with Head, Length, Part, Construct and Cast inspecting them without evaluation;
* TypeDeclaration["Product", ...] with CreateTypeInstance, and CompiledExpressionDeclaration;
* ToRawPointer and FromRawPointer on memory allocated through the FFI functions of wl_external_ffi (freed when the
  outermost compiled call returns, as stack memory is in Wolfram's compiler).
Outside compiled code these constructs stay unevaluated or raise, as their reference pages say.
"""
from __future__ import annotations

import contextvars

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, L, is_list, Unevaluated
from .wl_common import ev, is_sym

AREA = "core extras (X01)"
INERT = "Compile`InertValue"
_KF = "Compile`KernelFunction"
_KE = "Compile`KernelEvaluate"
_IC = "Compile`InertConstruct"
_IEV = "Compile`InertEvaluate"
_IOP = "Compile`InertOperation"
_CTI = "Compile`CreateTypeInstance"
_INST = "Compile`ProductInstance"
_CEX = "Compile`ConstructExpression"
_TRP = "Compile`ToRawPointer"
_FRP = "Compile`FromRawPointer"
_PTR = "Compile`Pointer"

_DEPTH = contextvars.ContextVar("x01_compile_depth", default=0)
_ALLOCS = contextvars.ContextVar("x01_compile_allocs", default=None)
_PRODUCTS = {}          # product type name -> [(field, parsed type)]
_CEXPRS = {}            # head name -> declaration record


def _WC():
    from . import wl_external_compile as WC
    return WC


def _show(x):
    try:
        from .. import formatting as F
        return F.input_form(x)
    except Exception:  # noqa: BLE001 - display only
        return repr(x)


def _inert(x):
    return Expr(INERT, x)


def _is_inert(x):
    return isinstance(x, Expr) and x.head == INERT and len(x.args) == 1


def _to_inert_part(v):
    """The expression an inert construction stores for a compiled value."""
    return v.args[0] if _is_inert(v) else v


for _h in (INERT, _KE, _IC, _IEV, _IOP, _CTI, _TRP, _FRP, _CEX, "KernelEvaluate", "InertExpression",
           "InertEvaluate", "CompiledExpressionDeclaration"):
    K.BUILTIN_ATTRIBUTES[_h] = {"HoldAllComplete" if _h == INERT else "HoldAll", "Protected"}


def _mk(h, *args):
    """h[args] with a symbol head given by name (Expr heads of symbols are strings)."""
    return Expr(h.name if isinstance(h, Symbol) else h, *args)


# ═══════════════════════════════════════ the compiled view (PREPARE_HOOKS) ═══════════════════════════════════════

def _fn_type(t):
    return isinstance(t, Expr) and t.head == "Rule" and is_list(t.args[0])


def _prepare_hook(x, walk):
    h = x.head
    # Typed[KernelFunction[f], type] / TypeHint[...]: keep the declared function type for checking
    if h in ("Typed", "TypeHint") and len(x.args) == 2 and isinstance(x.args[0], Expr) and \
            x.args[0].head in ("KernelFunction", "DownValuesFunction") and len(x.args[0].args) == 1:
        return Expr(_KF, x.args[0].head, x.args[0].args[0], x.args[1])
    if h in ("KernelFunction", "DownValuesFunction") and len(x.args) == 1:
        return Expr(_KF, h, x.args[0], Symbol("None"))
    if h in ("Typed", "TypeHint") and len(x.args) == 2 and isinstance(x.args[0], Expr) and \
            x.args[0].head == "KernelEvaluate":
        return Expr(_KE, x.args[1], *x.args[0].args)
    if h == "KernelEvaluate":
        return Expr(_KE, Symbol("None"), *x.args)
    if isinstance(h, Expr) and h.head == "InertExpression" and len(h.args) == 1:
        return Expr(_IC, h.args[0], *(walk(a) for a in x.args))
    if h == "InertExpression":
        if len(x.args) != 1:
            raise TypeError("InertExpression: InertExpression[expr] takes one argument")
        return _inert(x.args[0])
    if h == "InertEvaluate":
        if len(x.args) != 1:
            raise TypeError("InertEvaluate: InertEvaluate[inert] takes one argument")
        return Expr(_IEV, walk(x.args[0]))
    if h in ("Head", "Length", "Part", "Construct", "Cast") and x.args:
        return Expr(_IOP, h, *(walk(a) for a in x.args))
    if h == "CreateTypeInstance":
        return Expr(_CTI, *(walk(a) for a in x.args))
    if h == "ToRawPointer":
        return Expr(_TRP, Symbol("None"), *(walk(a) for a in x.args))
    if h in ("Typed", "TypeHint") and len(x.args) == 2 and isinstance(x.args[0], Expr) and \
            x.args[0].head == "ToRawPointer":
        return Expr(_TRP, x.args[1], *(walk(a) for a in x.args[0].args))
    if h == "FromRawPointer":
        return Expr(_FRP, *(walk(a) for a in x.args))
    if isinstance(h, str) and h in _CEXPRS and _CEXPRS[h]["types"] is not None:
        return Expr(_CEX, h, *(walk(a) for a in x.args))
    return None


def _install():
    WC = _WC()
    if _prepare_hook not in WC.PREPARE_HOOKS:
        WC.PREPARE_HOOKS.append(_prepare_hook)
    WC.DECL_HOOKS[("TypeDeclaration", "Product")] = _product_decl
    WC.DECL_HOOKS["CompiledExpressionDeclaration"] = _cexpr_decl
    WC.TYPE_CONFORMERS["Product"] = _product_conform
    WC.TYPE_CONFORMERS["CExpr"] = _cexpr_conform
    WC.TYPE_EXPRS["Product"] = lambda t: t[1]
    WC.TYPE_EXPRS["CExpr"] = lambda t: t[1]
    orig = KL.COMPOUND_HEAD_HOOKS.get("CompiledCodeFunction")
    if orig is not None and not getattr(orig, "_x01", False):
        def wrapped(head, args, ev_):
            depth = _DEPTH.get()
            tok = _DEPTH.set(depth + 1)
            atok = _ALLOCS.set([]) if depth == 0 else None
            try:
                r = orig(head, args, ev_)
                if depth == 0 and r is not K.UNHANDLED:
                    r = _returned(r)
                return r
            finally:
                if atok is not None:
                    _free_all()
                    _ALLOCS.reset(atok)
                _DEPTH.reset(tok)
        wrapped._x01 = True
        KL.COMPOUND_HEAD_HOOKS["CompiledCodeFunction"] = wrapped


def _in_compiled():
    return _DEPTH.get() > 0


def _returned(r):
    """A value leaving compiled code: inert expressions are converted and evaluated, as documented."""
    if _contains(r, (_INST,)):
        raise NotImplementedError("CompiledCodeFunction: returning an instance of a product type from compiled code "
                                  "is not supported (return its fields)")
    if _contains(r, (_PTR,)):
        raise NotImplementedError("CompiledCodeFunction: returning a raw pointer from compiled code is not "
                                  "supported (the memory is freed when the function returns)")
    if _contains(r, (INERT,)):
        return ev(_strip_inert(r))
    return r


def _contains(x, heads):
    if isinstance(x, Expr):
        if x.head in heads:
            return True
        if isinstance(x.head, Expr) and _contains(x.head, heads):
            return True
        return any(_contains(a, heads) for a in x.args)
    return False


def _strip_inert(x):
    if _is_inert(x):
        return _strip_inert(x.args[0])
    if isinstance(x, Expr):
        h = _strip_inert(x.head) if isinstance(x.head, Expr) else x.head
        return Expr(h, *(_strip_inert(a) for a in x.args))
    return x


# ═══════════════════════════════════════ KernelFunction / DownValuesFunction / KernelEvaluate ════════════════════

def _parse(t, fname):
    return _WC().parse_type(ev(t), fname)


def _check_value(v, t, fname, what):
    WC = _WC()
    ok, c = WC.conform(v, t, {}, fname)
    if not ok:
        raise TypeError(f"{fname}: {what} {_show(v)} does not have the declared type {_show(WC.type_expr(t))}")
    return c


def _kf_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == _KF and len(head.args) == 3):
        return K.UNHANDLED
    kind, f, ty = head.args
    fname = kind
    sig = None
    if not is_sym(ty, "None"):
        sig = _parse(ty, fname)
        if not (isinstance(sig, tuple) and sig[0] == "Fn"):
            raise TypeError(f"{fname}: the type of {kind}[{_show(f)}] must be a function type {{args...}} -> result")
        if len(sig[1]) != len(args):
            raise TypeError(f"{fname}: {kind}[{_show(f)}] is declared with {len(sig[1])} arguments, called with "
                            f"{len(args)}")
        args = [_check_value(a, t, fname, "the argument") for a, t in zip(args, sig[1])]
    args = [_strip_inert(a) for a in args]
    if kind == "DownValuesFunction":
        if not isinstance(f, Symbol):
            raise TypeError("DownValuesFunction: the argument must be a symbol")
        r = _WC()._kernel_call(f.name, Expr("DownValuesFunction", f), args)
    else:
        r = ev(_mk(f, *args))
    r = _WC().machine(r)
    if sig is not None:
        r = _check_value(r, sig[2], fname, "the result")
    return r


KL.COMPOUND_HEAD_HOOKS[_KF] = _kf_hook


def _vars_of(spec):
    """Captured variables still present as symbols (function parameters are already replaced by their values)."""
    items = spec.args if is_list(spec) else (spec,)
    return [v for v in items if isinstance(v, Symbol)]


@fn(_KE, area=AREA, status="added", hold=True,
    notes="internal form of KernelEvaluate in compiled code (type, [vars,] expr); Check: exact in tests.")
def _ke_impl(ty: Held, *rest: Held) -> Expr:
    if len(rest) == 1:
        body, cap = rest[0], {}
    elif len(rest) == 2:
        cap = {v.name: ev(v) for v in _vars_of(rest[0])}
        body = rest[1]
    else:
        raise TypeError("KernelEvaluate: KernelEvaluate[expr], KernelEvaluate[x, expr] or KernelEvaluate[{x, ...}, "
                        "expr]")
    if cap:
        body = K.substitute(body, {k: _strip_inert(v) for k, v in cap.items()})
    r = _WC().machine(ev(body))
    if not is_sym(ty, "None"):
        r = _check_value(r, _parse(ty, "KernelEvaluate"), "KernelEvaluate", "the result")
    return r


@fn("KernelFunction", area=AREA, status="added",
    notes="KernelFunction[f] in compiled code (FunctionCompile) is a function evaluated by the evaluator: "
          "KernelFunction[f][args] gives f[args]; with Typed[KernelFunction[f], {t1, ...} -> t] or TypeHint the "
          "arguments and the result are checked against the types (TypeError otherwise), and "
          "FunctionDeclaration[name, Typed[type]@KernelFunction[f]] declares name as such a function. It can be "
          "stored in compiled variables (Module[{g = TypeHint[KernelFunction[...], ...]}, g[x]]). Outside compiled "
          "code KernelFunction[f] and KernelFunction[f][args] stay unevaluated. Check: reference (the documentation "
          "examples: AiryAi from compiled code, f[x_] := x^2 gives 100 for 10, # + 10 & gives 20.2) in tests.")
def _kernelfunction(f: Expr) -> Expr:
    raise Unevaluated()


@fn("DownValuesFunction", area=AREA, status="added",
    notes="DownValuesFunction[sym] in compiled code is a function given by the definitions of sym: "
          "Typed[DownValuesFunction[sym], {t...} -> t][args] and FunctionDeclaration[name, "
          "Typed[type]@DownValuesFunction[sym]] apply them with the evaluator after checking the argument types, "
          "and the result is checked against the declared result type (TypeError otherwise); a recursive "
          "definition such as factorial calls itself through its definitions. Outside compiled code it stays "
          "unevaluated. The definitions are not translated to machine code, so every pattern construct works. "
          "Check: reference (the documentation examples square[100] = 10000 and factorialC[10] = 3628800) in "
          "tests.")
def _downvaluesfunction(s: Expr) -> Expr:
    raise Unevaluated()


@fn("KernelEvaluate", area=AREA, status="added", hold=True,
    notes="KernelEvaluate[expr], KernelEvaluate[x, expr] and KernelEvaluate[{x1, ...}, expr] in compiled code "
          "evaluate expr with the evaluator; the captured compiled variables are substituted by their values "
          "(bound lexically, so expr cannot change them); TypeHint[KernelEvaluate[...], type] checks the result "
          "type (TypeError otherwise). Outside compiled code it stays unevaluated (HoldAll). Check: reference (the "
          "documentation examples: AiryAi[1.1] = 0.120049, f[10] = 100, a Module variable tmp = 2 arg) in tests.")
def _kernelevaluate(*args: Held) -> Expr:
    raise Unevaluated()


# ═══════════════════════════════════════ InertExpression / InertEvaluate ═══════════════════════════════════════

@fn(_IC, area=AREA, status="added", hold=True,
    notes="internal form of InertExpression[h][e1, ...] in compiled code; Check: exact in tests.")
def _ic_impl(h: Held, *args: Held) -> Expr:
    vals = [_to_inert_part(ev(a)) for a in args]
    return _inert(_mk(h, *vals))


@fn(_IEV, area=AREA, status="added", hold=True,
    notes="internal form of InertEvaluate in compiled code; Check: exact in tests.")
def _iev_impl(x: Held) -> Expr:
    v = ev(x)
    if not _is_inert(v):
        raise TypeError(f"InertEvaluate: {_show(v)} is not an \"InertExpression\" value")
    return _inert(ev(v.args[0]))


def _held_part(e, idx):
    for i in idx:
        if not (K.is_int(i) and not isinstance(i, bool)):
            raise NotImplementedError("Part: only integer positions are supported for \"InertExpression\" values")
        i = int(i)
        if i == 0:
            e = e.head if isinstance(e, Expr) else Symbol(K.head_name(e)) if hasattr(K, "head_name") else e
            continue
        if not isinstance(e, Expr) or not 1 <= abs(i) <= len(e.args):
            raise ValueError(f"Part: part {i} of {_show(e)} does not exist")
        e = e.args[i - 1 if i > 0 else i]
    return e


@fn(_IOP, area=AREA, status="added", hold=True,
    notes="internal form of Head, Length, Part, Construct and Cast in compiled code (inert-aware); Check: exact in "
          "tests.")
def _iop_impl(op: Held, *args: Held) -> Expr:
    vals = [ev(a) for a in args]
    if not any(_is_inert(v) for v in vals) and not (op == "Cast" and len(vals) >= 2 and vals[1] == "InertExpression"):
        if op == "Cast":
            return _cast(vals)
        return ev(Expr(op, *vals))
    if op == "Head":
        e = vals[0].args[0] if _is_inert(vals[0]) else vals[0]
        return _inert(_head_of(e))
    if op == "Length":
        e = vals[0].args[0] if _is_inert(vals[0]) else vals[0]
        return len(e.args) if isinstance(e, Expr) else 0
    if op == "Part":
        if not _is_inert(vals[0]):
            return ev(Expr("Part", *vals))
        return _inert(_held_part(vals[0].args[0], vals[1:]))
    if op == "Construct":
        return _inert(_mk(_to_inert_part(vals[0]), *(_to_inert_part(v) for v in vals[1:])))
    if op == "Cast":
        return _cast(vals)
    raise ValueError(op)


def _head_of(e):
    if isinstance(e, Expr):
        return e.head if isinstance(e.head, Expr) else Symbol(e.head)
    r = ev(Expr("Head", e))
    return r


def _cast(vals):
    WC = _WC()
    if len(vals) < 2:
        raise TypeError("Cast: Cast[value, type] or Cast[value, type, method]")
    v, t = vals[0], vals[1]
    if t == "InertExpression":
        return v if _is_inert(v) else _inert(v)
    if _is_inert(v):
        v = v.args[0]
    try:
        ty = WC.parse_type(t, "Cast")
    except (ValueError, NotImplementedError):
        return ev(Expr("Cast", v, *vals[1:]))
    ok, c = WC.conform(v, ty, {}, "Cast")
    if ok:
        return c
    if isinstance(ty, str) and ty in WC._INT_RANGES and isinstance(v, float) and len(vals) > 2:
        return ev(Expr("Cast", v, *vals[1:]))
    raise TypeError(f"Cast: {_show(v)} cannot be converted to the type {_show(t)}")


def _only_compiled(name):
    raise ValueError(f"{name}: {name} is only available in compiled code (FunctionCompile)")


@fn("InertExpression", area=AREA, status="added", hold=True,
    notes="In compiled code (FunctionCompile) InertExpression[expr] is the unevaluated expr as an "
          "\"InertExpression\" value and InertExpression[h][e1, ...] builds h[e1, ...] from compiled values (inert "
          "values are inserted as they are, other values as expressions), without evaluating it. Head, Length, "
          "Part (integer positions), Construct and Cast (to and from \"InertExpression\") inspect inert values "
          "without evaluating them; InertEvaluate evaluates. An inert value returned from compiled code is converted "
          "to an expression and evaluated, as documented. The value is represented by the internal head "
          "Compile`InertValue (HoldAllComplete). Outside compiled code InertExpression raises ValueError "
          "(\"only available in compiled code\"). Check: reference (the documentation examples: foo, f[x], f[10], "
          "Length[InertExpression[2 + 2]] = 2, Hold[Total[{1, 2, 3}]], Echo not evaluated) in tests.")
def _inertexpression(*args: Held) -> Expr:
    _only_compiled("InertExpression")


@fn("InertEvaluate", area=AREA, status="added", hold=True,
    notes="In compiled code InertEvaluate[inert] evaluates the expression of an \"InertExpression\" value once and "
          "gives the result as a new inert value (it is not evaluated further: InertEvaluate[InertExpression[y + 2 + "
          "3]] stays a Plus after y = 3). A value that is not inert raises TypeError; outside compiled code "
          "InertEvaluate raises ValueError. Check: reference (the documentation examples: Head[...] = Integer, "
          "Cast[InertEvaluate[Construct[InertExpression[f], Cast[arg, \"InertExpression\"]]], \"MachineInteger\"] = "
          "100, Head = Plus) in tests.")
def _inertevaluate(*args: Held) -> Expr:
    _only_compiled("InertEvaluate")


# ═══════════════════════════════════════ product types and CreateTypeInstance ═══════════════════════════════════

def _product_decl(d, aliases, fname):
    WC = _WC()
    if len(d.args) < 3 or not isinstance(d.args[1], str) or not KL.is_assoc(d.args[2]):
        raise ValueError(f"{fname}: TypeDeclaration[\"Product\", name, <|field -> type, ...|>]")
    name = d.args[1]
    fields = []
    for r in d.args[2].args:
        if not isinstance(r.args[0], str):
            raise TypeError(f"{fname}: product field names must be strings")
        fields.append((r.args[0], WC.parse_type(r.args[1], fname, (), aliases)))
    _PRODUCTS[name] = fields
    aliases[name] = ("Product", name)


def _product_conform(v, t, b, fname):
    if isinstance(v, Expr) and v.head == _INST and len(v.args) == 2 and v.args[0] == t[1]:
        return True, v
    return False, v


def _type_name(t):
    if isinstance(t, Expr) and t.head == "TypeSpecifier" and len(t.args) == 1:
        t = t.args[0]
    return t


@fn(_CTI, area=AREA, status="added", hold=True,
    notes="internal form of CreateTypeInstance in compiled code; Check: exact in tests.")
def _cti_impl(*args: Held) -> Expr:
    vals = [ev(a) for a in args]
    if not vals:
        raise TypeError("CreateTypeInstance: CreateTypeInstance[type, args...]")
    t = _type_name(vals[0])
    if isinstance(t, str) and t in _PRODUCTS:
        fields = _PRODUCTS[t]
        if len(vals) != 2 or not KL.is_assoc(vals[1]):
            raise TypeError(f"CreateTypeInstance: the product type {t!r} is created with CreateTypeInstance[{t!r}, "
                            "<|field -> value, ...|>]")
        given = {r.args[0]: r.args[1] for r in vals[1].args}
        unknown = set(given) - {f for f, _ in fields}
        if unknown:
            raise ValueError(f"CreateTypeInstance: {sorted(unknown)[0]!r} is not a field of the product type {t!r}")
        out = []
        for f, ft in fields:
            if f not in given:
                raise ValueError(f"CreateTypeInstance: the field {f!r} of {t!r} is not initialised")
            out.append(Expr("Rule", f, _check_value(given[f], ft, "CreateTypeInstance", f"the field {f!r} value")))
        return Expr(_INST, t, Expr("Association", *out))
    ca = _carray_type(t)
    if ca is not None:
        if len(vals) != 2 or not (K.is_int(vals[1]) and not isinstance(vals[1], bool)) or int(vals[1]) < 1:
            raise TypeError("CreateTypeInstance: a \"CArray\" is created with CreateTypeInstance[type, n], n a "
                            "positive integer")
        return _new_pointer(ca, int(vals[1]))
    raise NotImplementedError(f"CreateTypeInstance: the type {_show(t)} is not supported (declared product types, "
                              "\"CArray\" and \"Managed\"[\"CArray\"] are)")


def _inst_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == _INST and len(head.args) == 2 and len(args) == 1):
        return K.UNHANDLED
    k = args[0]
    for r in head.args[1].args:
        if r.args[0] == k:
            return r.args[1]
    raise ValueError(f"CreateTypeInstance: the product type {head.args[0]!r} has no field {_show(k)}")


KL.COMPOUND_HEAD_HOOKS[_INST] = _inst_hook


@fn("CreateTypeInstance", area=AREA, status="added",
    notes="Partial. In compiled code CreateTypeInstance[productType, <|field -> x, ...|>] creates an instance of a "
          "type declared with TypeDeclaration[\"Product\", name, <|field -> type, ...|>] (every field must be "
          "given and is checked against its type; TypeError/ValueError otherwise) and inst[\"field\"] reads a field; "
          "CreateTypeInstance[TypeSpecifier[\"CArray\"][t], n] and TypeSpecifier[\"Managed\"][...] allocate n zeroed "
          "elements (memory of RawMemoryAllocate, freed when the compiled function returns) for ToRawPointer and "
          "FromRawPointer. Returning an instance from compiled code, custom constructors declared with "
          "FunctionDeclaration[CreateTypeInstance, ...] and other types (DataStructure, ...) raise "
          "NotImplementedError; outside compiled code CreateTypeInstance raises ValueError. Check: reference (the "
          "documentation example prod[\"field1\"] = 12 for 12, and the CArray offset example 8) in tests.")
def _createtypeinstance(*args: Expr) -> Expr:
    _only_compiled("CreateTypeInstance")


# ═══════════════════════════════════════ CompiledExpressionDeclaration ═══════════════════════════════════════

def _cexpr_parse(d, aliases, fname):
    WC = _WC()
    args = list(d.args)
    if len(args) < 2:
        raise TypeError(f"{fname}: CompiledExpressionDeclaration[h, n | {{types}} | test]")
    hspec = args[0]
    tvars = ()
    name = None
    if isinstance(hspec, Expr) and hspec.head == "Rule" and len(hspec.args) == 2:
        h, nm = hspec.args
        if isinstance(nm, str):
            name = nm
        elif isinstance(nm, Expr) and isinstance(nm.head, Expr) and nm.head.head == "TypeSpecifier" and \
                len(nm.head.args) == 1 and isinstance(nm.head.args[0], str):
            name = nm.head.args[0]
            tvars = tuple(WC._tvar_name(v) for v in nm.args)
            if any(v is None for v in tvars):
                raise TypeError(f"{fname}: the type parameters must be symbols")
        else:
            raise TypeError(f"{fname}: the type name must be a string or TypeSpecifier[\"name\"][vars...]")
    else:
        h = hspec
    if not isinstance(h, Symbol):
        raise TypeError(f"{fname}: the head of a compiled expression type must be a symbol")
    if name is None:
        name = h.name.split("`")[-1]
    rest = args[1:]
    cond = None
    if len(rest) == 2 and tvars:
        cond, rest = rest[0], rest[1:]
    test = None
    if len(rest) == 2 and isinstance(rest[1], Expr) and rest[1].head == "Function":
        rest, test = rest[:1], rest[1]
    if len(rest) != 1:
        raise TypeError(f"{fname}: CompiledExpressionDeclaration[h, n | {{types}} | test]")
    spec = rest[0]
    rec = {"head": h.name, "name": name, "tvars": tvars, "cond": cond, "types": None, "n": None, "test": test}
    if K.is_int(spec) and not isinstance(spec, bool) and int(spec) >= 0:
        rec["n"] = int(spec)
    elif is_list(spec):
        rec["types"] = [WC.parse_type(t, fname, tvars, aliases) for t in spec.args]
    elif isinstance(spec, Expr) and spec.head == "Function":
        rec["test"] = spec
    else:
        raise TypeError(f"{fname}: the second argument must be an arity, a list of types or a test function")
    return rec


def _cexpr_decl(d, aliases, fname):
    rec = _cexpr_parse(d, aliases, fname)
    _CEXPRS[rec["head"]] = rec
    aliases[rec["name"]] = ("CExpr", rec["name"], rec["head"])


def _cexpr_check(v, rec, b, fname):
    WC = _WC()
    if not (isinstance(v, Expr) and v.head == rec["head"]):
        return False, v
    if rec["n"] is not None and len(v.args) != rec["n"]:
        return False, v
    if rec["types"] is not None:
        if len(v.args) != len(rec["types"]):
            return False, v
        out = []
        for a, t in zip(v.args, rec["types"]):
            ok, c = WC.conform(a, t, b, fname)
            if not ok:
                return False, v
            out.append(c)
        if not WC._check_cond(rec["cond"], b, fname):
            return False, v
        v = Expr(v.head, *out)
    if rec["test"] is not None:
        r = ev(Expr(rec["test"], v, rec["name"]))
        if r is not True:
            return False, v
    return True, v


def _cexpr_conform(v, t, b, fname):
    rec = _CEXPRS.get(t[2])
    if rec is None:
        return False, v
    return _cexpr_check(v, rec, b, fname)


@fn(_CEX, area=AREA, status="added", hold=True,
    notes="internal constructor of a typed compiled expression; Check: exact in tests.")
def _cex_impl(h: Held, *args: Held) -> Expr:
    rec = _CEXPRS[h]
    vals = [_WC().machine(ev(a)) for a in args]
    ok, v = _cexpr_check(Expr(h, *vals), rec, {}, "CompiledExpressionDeclaration")
    if not ok:
        raise TypeError(f"CompiledExpressionDeclaration: {_show(Expr(h, *vals))} does not have the declared type "
                        f"{rec['name']!r}")
    return v


@fn("CompiledExpressionDeclaration", area=AREA, status="added", hold=True,
    notes="CompiledExpressionDeclaration[h, n], [h, {t1, ...}], [h -> \"name\", ...], [h -> "
          "TypeSpecifier[\"name\"][v1, ...], cond, {t1, ...}] (type parameters, cond as in ForAllType, e.g. "
          "Element[v, \"Numbers\"]) and [..., test] is an inert declaration (HoldAll, so that the "
          "TypeSpecifier[\"name\"][v, ...] parameter form is kept as written) for FunctionCompile and "
          "CompilerEnvironmentAppendTo; the type name (SymbolName[h] by default) can be used as a parameter or "
          "result type: a value has the type when its head is h and it has n arguments, arguments of the given "
          "types, or test[expr, name] gives True. In compiled code h[x1, ...] with declared argument types checks "
          "and converts the arguments (TypeError otherwise) and gives the expression h[x1, ...]; Part, Head and "
          "the other functions work on it as on any expression. Deviation: the \"name\"::[v1, ...] input syntax "
          "is not parsed; write TypeSpecifier[\"name\"][v1, ...]. Check: reference (the documentation examples "
          "foo[3, 5.6], the Length test accepting foo[1, 2] and rejecting bar[1, 2, 3] with "
          "Failure[\"ArgumentType\", ...]) in tests.")
def _compiledexpressiondeclaration(*args: Held) -> Expr:
    _cexpr_parse(Expr("CompiledExpressionDeclaration", *args), {}, "CompiledExpressionDeclaration")
    raise Unevaluated()


# ═══════════════════════════════════════ ToRawPointer / FromRawPointer ═══════════════════════════════════════

_CTYPES = {"MachineInteger": "Integer64", "Integer64": "Integer64", "Integer32": "Integer32",
           "Integer16": "Integer16", "Integer8": "Integer8", "UnsignedInteger8": "UnsignedInteger8",
           "UnsignedInteger16": "UnsignedInteger16", "UnsignedInteger32": "UnsignedInteger32",
           "UnsignedInteger64": "UnsignedInteger64", "Real64": "Real64", "MachineReal": "Real64",
           "Real32": "Real32"}


def _elem_ctype(t):
    if isinstance(t, str):
        return _CTYPES.get(t, t if t.startswith("C") else None)
    return None


def _carray_type(t):
    """Element type of TypeSpecifier["CArray"][t] or TypeSpecifier["Managed"][TypeSpecifier["CArray"][t]]."""
    if isinstance(t, Expr) and isinstance(t.head, Expr) and t.head.head == "TypeSpecifier" and \
            len(t.head.args) == 1 and len(t.args) == 1:
        k = t.head.args[0]
        if k == "Managed":
            return _carray_type(_type_name(t.args[0]))
        if k == "CArray":
            return _elem_ctype(_type_name(t.args[0]))
    return None


def _pointer_type(t):
    if isinstance(t, Expr) and isinstance(t.head, Expr) and t.head.head == "TypeSpecifier" and \
            len(t.head.args) == 1 and t.head.args[0] == "RawPointer" and len(t.args) == 1:
        return _elem_ctype(_type_name(t.args[0]))
    return None


def _new_pointer(ctype, n):
    allocs = _ALLOCS.get()
    if allocs is None:
        _only_compiled("ToRawPointer")
    p = ev(Expr("RawMemoryAllocate", ctype, n))
    allocs.append(p)
    return Expr(_PTR, len(allocs) - 1, ctype, n)


def _free_all():
    for p in _ALLOCS.get() or []:
        try:
            ev(Expr("RawMemoryFree", p))
        except Exception:  # noqa: BLE001 - already freed
            pass


def _raw(ptr, fname):
    if not (isinstance(ptr, Expr) and ptr.head == _PTR and len(ptr.args) == 3):
        raise TypeError(f"{fname}: {_show(ptr)} is not a pointer or \"CArray\" made in compiled code")
    allocs = _ALLOCS.get() or []
    i = ptr.args[0]
    if not (isinstance(i, int) and 0 <= i < len(allocs)):
        raise ValueError(f"{fname}: the pointer is no longer valid (the function that made it has returned)")
    return allocs[i], ptr.args[1], ptr.args[2]


def _value_ctype(v):
    if K.is_int(v) and not isinstance(v, bool):
        return "Integer64"
    if isinstance(v, float):
        return "Real64"
    raise NotImplementedError(f"ToRawPointer: values such as {_show(v)} are not supported (machine integers and "
                              "reals are)")


def _offset(o, n, fname):
    if not (K.is_int(o) and not isinstance(o, bool)):
        raise TypeError(f"{fname}: the offset must be an integer")
    if not 0 <= int(o) < n:
        raise ValueError(f"{fname}: the offset {int(o)} is outside the array of length {n}")
    return int(o)


@fn(_TRP, area=AREA, status="added", hold=True,
    notes="internal form of ToRawPointer in compiled code; Check: exact in tests.")
def _trp_impl(ty: Held, *args: Held) -> Expr:
    vals = [ev(a) for a in args]
    if not vals:
        ct = _pointer_type(ev(ty)) if not is_sym(ty, "None") else None
        if not is_sym(ty, "None") and ct is None:
            raise NotImplementedError(f"ToRawPointer: the pointer type {_show(ev(ty))} is not supported")
        if ct is None:
            return Expr(_PTR, -1, Symbol("None"), 1)        # typed on its first store
        return _new_pointer(ct, 1)
    if len(vals) == 1:
        p = _new_pointer(_value_ctype(vals[0]), 1)
        _store(p, 0, vals[0])
        return p
    if len(vals) == 2:
        p = vals[0]
        if isinstance(p, Expr) and p.head == _PTR and p.args[0] == -1:
            raise NotImplementedError("ToRawPointer: store through the untyped pointer by writing "
                                      "With[{p = ToRawPointer[val]}, ...] or give its type with "
                                      "Typed[ToRawPointer[], TypeSpecifier[\"RawPointer\"][t]]")
        _store(p, 0, vals[1])
        return p
    if len(vals) == 3:
        _store(vals[0], vals[1], vals[2])
        return vals[0]
    raise TypeError("ToRawPointer: ToRawPointer[], ToRawPointer[val], ToRawPointer[p, val] or "
                    "ToRawPointer[array, offset, val]")


def _store(p, off, val):
    raw, ct, n = _raw(p, "ToRawPointer")
    o = _offset(off, n, "ToRawPointer")
    ev(Expr("RawMemoryWrite", raw, _strip_inert(val), o))


@fn(_FRP, area=AREA, status="added", hold=True,
    notes="internal form of FromRawPointer in compiled code; Check: exact in tests.")
def _frp_impl(*args: Held) -> Expr:
    vals = [ev(a) for a in args]
    if len(vals) not in (1, 2):
        raise TypeError("FromRawPointer: FromRawPointer[p] or FromRawPointer[array, offset]")
    raw, ct, n = _raw(vals[0], "FromRawPointer")
    o = _offset(vals[1] if len(vals) == 2 else 0, n, "FromRawPointer")
    return ev(Expr("RawMemoryRead", raw, o))


@fn("ToRawPointer", area=AREA, status="added",
    notes="Partial. In compiled code ToRawPointer[val] gives a pointer to a new copy of val (machine integers as "
          "\"Integer64\", reals as \"Real64\"), Typed[ToRawPointer[], TypeSpecifier[\"RawPointer\"][t]] a pointer to "
          "a new zeroed element, ToRawPointer[p, val] stores val through p and ToRawPointer[array, offset, val] "
          "stores into a \"CArray\" from CreateTypeInstance (bounds checked: ValueError). The memory comes from "
          "RawMemoryAllocate and is freed when the outermost compiled function returns (Wolfram allocates on the "
          "stack, with the same lifetime); returning a pointer raises NotImplementedError. An untyped "
          "ToRawPointer[] followed by ToRawPointer[p, val], other value types and LibraryFunction calls raise "
          "NotImplementedError; outside compiled code ToRawPointer raises ValueError. Check: reference (the "
          "documentation examples: cf[4] = 4, the CArray example 8) in tests.")
def _torawpointer(*args: Expr) -> Expr:
    _only_compiled("ToRawPointer")


@fn("FromRawPointer", area=AREA, status="added",
    notes="In compiled code FromRawPointer[p] reads the value a pointer from ToRawPointer refers to and "
          "FromRawPointer[array, offset] an element of a \"CArray\" from CreateTypeInstance (offsets from 0, "
          "bounds checked: ValueError); a value that is not such a pointer raises TypeError; outside compiled code "
          "FromRawPointer raises ValueError. Check: reference (the documentation examples: cf[4] = 4 and "
          "FromRawPointer[arr, 0] + FromRawPointer[arr, 0] = 8) in tests.")
def _fromrawpointer(*args: Expr) -> Expr:
    _only_compiled("FromRawPointer")


_install()
