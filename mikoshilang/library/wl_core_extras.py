"""Wolfram batch X01 (part 1): core and system extras.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). This module covers option symbols that other batches already honour (registered here so they
are System` symbols with notes), Symmetrize, NCache, MessageList, the TraceOriginal option of Trace, $MaxExtraPrecision,
TargetFunctions (ComplexExpand[expr, vars, TargetFunctions -> ...]) and AbsoluteOptions.
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated, L, is_list, REGISTRY
from .wl_common import ev, is_sym
from .wl_graphics_color import inert_option

AREA = "core extras (X01)"


def _show(x):
    try:
        from .. import formatting as F
        return F.input_form(x)
    except Exception:  # noqa: BLE001 - display only
        return repr(x)


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def _opt_key(x):
    k = x.args[0]
    if isinstance(k, Symbol):
        return k.name
    if isinstance(k, str):
        return k
    return None


def _split(args, names, fname):
    """(positional, {name: value}) with trailing Name -> value options; unknown option names raise ValueError."""
    args = list(args)
    opts = {}
    while args and _is_rule(args[-1]) and _opt_key(args[-1]) is not None and (
            isinstance(args[-1].args[0], Symbol) or _opt_key(args[-1]) in names):
        k = _opt_key(args[-1])
        if k not in names:
            raise ValueError(f"{fname}: {k} is not a known option (expected one of {', '.join(sorted(names))})")
        opts[k] = args.pop().args[1]
    return args, opts


# ═══════════════════════════════════ option symbols honoured elsewhere ═══════════════════════════════════

_OPTION_NOTES = {
    "FailureAction": "FailureAction: option of Query (Dataset) giving what happens when a part of the query issues a "
                     "message or fails: None, \"Abort\", \"Drop\", \"Encapsulate\", \"Replace\" or {act, f}; "
                     "honoured by Query in wl_dataset (W12). Inert as an expression.",
    "MissingBehavior": "MissingBehavior: option of Query (Dataset): None (ordinary Missing behaviour) or Automatic "
                       "(Missing elements dropped by Total/Mean/..., empty aggregates give "
                       "Missing[\"Indeterminate\"]); honoured by Query in wl_dataset (W12). Inert as an expression.",
    "IncludeDirectories": "IncludeDirectories: option of FileSystemMap, FileSystemScan and FileSystemTree giving "
                          "whether directories are included (default False); honoured by wl_files (W27) and "
                          "FileSystemTree (X01). Inert as an expression.",
    "Resampling": "Resampling: option of ArrayResample (wl_structural) and the image resampling functions "
                  "(wl_images) naming the interpolation (\"Nearest\", \"Linear\", \"Cubic\", ...); each function "
                  "validates the settings it supports and raises NotImplementedError for others. Inert as an "
                  "expression.",
    "Parallelization": "Parallelization: option of Compile (True, False or Automatic) validated by Compile in "
                       "wl_external_compile (W30); the compiled function runs single-threaded, which gives the same "
                       "results. Inert as an expression.",
    "RuntimeAttributes": "RuntimeAttributes: option of Compile; {} or {Listable} (threading over higher-rank "
                         "arguments) as handled by Compile in wl_external_compile (W30); other attributes raise "
                         "ValueError. Inert as an expression.",
    "RuntimeOptions": "RuntimeOptions: option of Compile (\"Speed\", \"Quality\" or nested settings) accepted by "
                      "Compile in wl_external_compile (W30). Inert as an expression.",
}
for _n, _d in _OPTION_NOTES.items():
    inert_option(_n, _d + " Check: reference (Wolfram documentation settings) in tests.", area=AREA)


# ═══════════════════════════════════════════════ Symmetrize ═══════════════════════════════════════════════

def _st():
    from . import wl_structural as S
    return S


def _array_dims(x):
    S = _st()
    d = S._full_dims(x)
    return d


def _symmetrize_symbolic(t, sym, fname):
    """(1/|G|) Sum phase^-1 TensorTranspose[T, perm] for a symbolic tensor, when the symmetry names its slots."""
    S = _st()
    if isinstance(sym, Expr) and sym.head in ("Symmetric", "Antisymmetric") and len(sym.args) == 1 and \
            is_list(sym.args[0]) and sym.args[0].args and all(K.is_int(v) for v in sym.args[0].args):
        r = max(int(v) for v in sym.args[0].args)
    elif is_list(sym) and sym.args and all(is_list(g) and len(g.args) == 2 and is_list(g.args[0])
                                           for g in sym.args):
        r = max(len(g.args[0].args) for g in sym.args)
    else:
        raise Unevaluated()            # Symmetric[All] etc. need the rank of the symbolic tensor
    group = S._group(S._sym_generators(sym, r, fname), r, fname)
    terms = []
    for p, ph in sorted(group.items()):
        q = list(p)
        while q and q[-1] == len(q):
            q.pop()
        base = t if not q else Expr("TensorTranspose", t, L(*q))
        terms.append(Expr("Times", K.from_sp(1 / ph), base))
    return ev(Expr("Times", Expr("Power", len(group), -1), Expr("Plus", *terms)))


@fn("Symmetrize", area=AREA, status="added",
    notes="Symmetrize[array, sym] gives the symmetrization of a full array under sym (Symmetric[slots], "
          "Antisymmetric[slots] or a generator list {{perm, phase}, ...}) as a SymmetrizedArray: each independent "
          "component is the phase-weighted average over the symmetry group (1/|G|) Sum phase^-1 "
          "A[Permute[i, g]]; Symmetrize[array] uses Symmetric[All]. A symbolic tensor with a symmetry naming its "
          "slots gives (1/|G|) Sum phase^-1 TensorTranspose[T, g] (trailing fixed points dropped, as documented); "
          "Symmetric[All] on a symbolic tensor stays unevaluated (its rank is unknown). Slots outside the rank, "
          "inconsistent phases or dimensions raise ValueError. Check: self-check (the result is verified to be "
          "invariant under every generator with its phase; ArithmeticError otherwise) and reference (Wolfram "
          "documentation examples) in tests.")
def _symmetrize(t: Expr, sym: Expr = None) -> Expr:
    S = _st()
    if sym is None:
        sym = Expr("Symmetric", Symbol("All"))
    if isinstance(t, (str, bool)) or (isinstance(t, (int, float)) and not isinstance(t, bool)):
        raise TypeError(f"Symmetrize: expected an array or a symbolic tensor, not {_show(t)}")
    if isinstance(t, Expr) and t.head == "SymmetrizedArray":
        t = ev(Expr("Normal", t))
    if not is_list(t):
        if isinstance(t, (Symbol, Expr)):
            return _symmetrize_symbolic(t, sym, "Symmetrize")
        raise TypeError(f"Symmetrize: expected an array, not {_show(t)}")
    dims = _array_dims(t)
    if not dims or 0 in dims:
        raise ValueError("Symmetrize: a non-empty full (rectangular) array is expected")
    info = S._SymInfo(dims, sym, "Symmetrize")

    def get(idx):
        v = t
        for i in idx:
            v = v.args[i - 1]
        return v
    n = len(info.group)
    vals = {}
    for idx in info.independent():
        terms = [Expr("Times", K.from_sp(1 / ph), get(S._permute(idx, p))) for p, ph in info.group.items()]
        vals[idx] = ev(Expr("Together", Expr("Times", Expr("Power", n, -1), Expr("Plus", *terms))))
        vals[idx] = ev(Expr("Simplify", vals[idx])) if not K.is_int(vals[idx]) else vals[idx]
    out = S._sa_make(dims, vals, sym)
    # self-check: the full array is invariant (with phases) under each generator
    full = S._sa_normal(dims, {k: v for k, v in vals.items()}, sym, "Symmetrize")
    gens = S._sym_generators(sym, len(dims), "Symmetrize")
    import itertools
    for p, ph in gens:
        for idx in itertools.product(*(range(1, d + 1) for d in dims)):
            a = full
            for i in idx:
                a = a.args[i - 1]
            b = full
            for i in S._permute(idx, p):
                b = b.args[i - 1]
            diff = ev(Expr("Simplify", Expr("Subtract", b, Expr("Times", K.from_sp(ph), a))))
            if not (K.is_int(diff) and int(diff) == 0) and not (isinstance(diff, float) and abs(diff) < 1e-9):
                raise ArithmeticError("Symmetrize: result failed verification")
    return out


# ═══════════════════════════════════════════════ NCache ═══════════════════════════════════════════════

def _numeric_like(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, (int, float, complex)):
        return True
    if is_list(x):
        return all(_numeric_like(a) for a in x.args)
    return ev(Expr("NumericQ", x)) is True


@fn("NCache", area=AREA, status="added",
    notes="NCache[x, xn] is an inert numeric cache object pairing an exact quantity x with its approximate value xn "
          "(as Wolfram's Graphics formatting produces); it stays as it is. Both parts must be numeric (or lists of "
          "numerics of the same shape): otherwise TypeError/ValueError. Deviation: Graphics formatting (ToBoxes) "
          "does not generate NCache objects. Check: exact (xn is verified to agree with N[x] to 1e-6 relative; "
          "ArithmeticError otherwise) in tests.")
def _ncache(x: Expr, xn: Expr) -> Expr:
    if not _numeric_like(x):
        if isinstance(x, (str, bool)):
            raise TypeError(f"NCache: the exact value must be numeric, not {_show(x)}")
        raise Unevaluated()
    if not _numeric_like(xn):
        raise TypeError(f"NCache: the approximate value must be numeric, not {_show(xn)}")
    if is_list(x) != is_list(xn) or (is_list(x) and _array_dims(x) != _array_dims(xn)):
        raise ValueError("NCache: the exact and approximate values must have the same shape")
    nx = ev(Expr("N", x))
    flat_a = ev(Expr("Flatten", L(nx))).args
    flat_b = ev(Expr("Flatten", L(xn))).args
    for a, b in zip(flat_a, flat_b):
        a, b = complex(K.to_sp(a)), complex(K.to_sp(b))
        if abs(a - b) > 1e-6 * max(1.0, abs(a)):
            raise ArithmeticError("NCache: the approximate value does not agree with the exact value")
    raise Unevaluated()


# ═══════════════════════════════════════════════ MessageList ═══════════════════════════════════════════════

def _hist():
    from . import wl_evaluation as WE
    return WE._hist()


def _record_messages(n, h):
    """Main-loop hook (end of line n): MessageList[n] = $MessageList; $MessageList is reset to {}."""
    st = K.state()
    cur = st.own.get("$MessageList")
    if cur is None:
        from .wl_numerics import SYSTEM_CONSTANTS
        cur = SYSTEM_CONSTANTS.get("$MessageList", L())
    if not is_list(cur):
        cur = L()
    h.setdefault(n, {})["messages"] = cur
    st.own["$MessageList"] = L()


@fn("MessageList", area=AREA, status="added",
    notes="MessageList[n] is the list of the names (HoldForm[s::t]) of the messages output while the main loop "
          "(Session.evaluate_input, REPL, Jupyter) processed input line n; MessageList[-k] counts back from the "
          "current line, as Out does. At the end of each line the main loop stores $MessageList in "
          "MessageList[n] and resets $MessageList to {} (documented for $MessageList). Lines that are no longer "
          "in the history ($HistoryLength) or have not been processed stay unevaluated; a non-integer argument "
          "raises TypeError. Check: exact (messages issued with Message in earlier lines are listed, quiet lines "
          "give {}) in tests.")
def _messagelist(n: Integer) -> ListOf[Expr]:
    if isinstance(n, (bool, str, float)):
        raise TypeError(f"MessageList: the line number must be an integer, not {_show(n)}")
    if not K.is_int(n):
        raise Unevaluated()
    from . import wl_evaluation as WE
    k = int(n)
    i = WE._line() + k if k <= 0 else k
    rec = K.state().ext.get("\x00history", {}).get(i)
    if rec is None or "messages" not in rec:
        raise Unevaluated()
    return rec["messages"]


def _install_line_hook():
    from . import wl_evaluation as WE
    hooks = getattr(WE, "LINE_END_HOOKS", None)
    if hooks is not None and _record_messages not in hooks:
        hooks.append(_record_messages)


_install_line_hook()


# ═══════════════════════════════════════════════ TraceOriginal ═══════════════════════════════════════════════

record("TraceOriginal", AREA, "added",
       "TraceOriginal: option of Trace, TraceScan and TracePrint. False (default): forms are recorded after their "
       "head and arguments are evaluated and chains of expressions that do not change are omitted; True: the form "
       "before evaluation of the head and arguments starts every chain, the head and every argument get their own "
       "chain (also when they do not change, e.g. {f}, {1}) and the form after argument evaluation is kept even "
       "when equal to the original; Automatic: the starting form of each chain is shown but unchanged chains are "
       "dropped. Other settings raise ValueError. Check: reference (the Wolfram documentation example "
       "Trace[f[1, 2 + 3], TraceOriginal -> True]) in tests.")


# ═══════════════════════════════════════════════ $MaxExtraPrecision ═══════════════════════════════════════════════

def _mep_getter():
    return 50


def _assign_mep(lhs, rhs, ev_):
    if isinstance(lhs, Expr):
        raise ValueError(f"Set: cannot assign to {_show(lhs)}")
    v = ev_(rhs)
    if not (is_sym(v, "Infinity") or (not isinstance(v, bool) and (K.is_int(v) or isinstance(v, float))
                                       and float(K.to_sp(v)) >= 0)):
        raise ValueError("$MaxExtraPrecision: the value must be a non-negative number or Infinity")
    K.state().own["$MaxExtraPrecision"] = v


def _install_mep():
    import importlib
    E = importlib.import_module("mikoshilang.evaluate")
    E.SYSTEM_VARIABLES.setdefault("$MaxExtraPrecision", _mep_getter)
    K.ASSIGN_HOOKS["$MaxExtraPrecision"] = _assign_mep


_install_mep()
record("$MaxExtraPrecision", AREA, "added",
       "$MaxExtraPrecision: the maximum number of extra digits Wolfram's N and exact numeric comparisons may use "
       "(default 50; assignable and Block-able). MikoshiLang's exact numerics (SymPy adaptive evalf, used by N, "
       "Equal/Less on exact numbers, Round, Floor, Sign) are not capped: they always raise the working precision "
       "until the requested precision is reached, so results agree with Wolfram whenever Wolfram's result is "
       "within its cap and are more accurate (never less) when Wolfram would hit it (N::meprec). The value is "
       "therefore informational. Check: reference (default 50 from the Wolfram documentation; N[Sin[10^100]] "
       "agrees with mpmath at 50 digits) in tests.")


# ═══════════════════════════════════════════════ TargetFunctions ═══════════════════════════════════════════════

record("TargetFunctions", AREA, "added",
       "TargetFunctions: option of ComplexExpand[expr, vars, TargetFunctions -> spec] giving the functions of the "
       "complex variables in the result: {Re, Im} (default), {Abs, Arg} or Conjugate ({Conjugate}); other "
       "settings raise NotImplementedError. (FindDistribution is not available.) Check: numeric cross-check (the "
       "rewritten result agrees with the input at sample points) in tests.")


def _cx_vars(v, fname):
    if isinstance(v, Symbol):
        return [v]
    if is_list(v) and all(isinstance(x, (Symbol, Expr)) and not isinstance(x, str) for x in v.args):
        out = []
        for x in v.args:
            if isinstance(x, Expr) and x.head == "Blank":
                raise NotImplementedError(f"{fname}: patterns as variables are not supported")
            out.append(x)
        return out
    raise TypeError(f"{fname}: the variables must be a symbol or a list of symbols, not {_show(v)}")


def _target(spec, fname):
    if spec is None:
        return "ReIm"
    names = set()
    if isinstance(spec, Symbol):
        names = {spec.name}
    elif is_list(spec) and all(isinstance(x, Symbol) for x in spec.args):
        names = {x.name for x in spec.args}
    else:
        raise ValueError(f"{fname}: TargetFunctions must be a function name or a list of them")
    if names <= {"Re", "Im"} and names:
        return "ReIm"
    if names == {"Abs", "Arg"}:
        return "AbsArg"
    if names == {"Conjugate"}:
        return "Conj"
    raise NotImplementedError(f"{fname}: TargetFunctions -> {_show(spec)} is not supported ({{Re, Im}}, "
                              "{Abs, Arg} or Conjugate)")


@fn("ComplexExpand", area=AREA, status="fixed",
    notes="ComplexExpand[expr, {z1, ...}] (also a single variable) expands assuming the zi are complex and every "
          "other variable real, giving the result in terms of Re[zi], Im[zi] (default), or with TargetFunctions -> "
          "{Abs, Arg} in terms of Abs[zi], Arg[zi], or with TargetFunctions -> Conjugate in terms of zi and "
          "Conjugate[zi] (SymPy expand_complex on z = x + I y with real x, y, then x, y rewritten). The one-argument "
          "form (all variables real) is unchanged. Check: numeric cross-check (the result is compared with the "
          "input at a complex sample point; ArithmeticError on mismatch) in tests.")
def _complexexpand_vars(e: Expr, v: Var | ListOf[Var], *opts: Option) -> Expr:
    if _is_rule(v) and _opt_key(v) == "TargetFunctions":
        opts = (v,) + tuple(opts)
        v = L()
    _, o = _split(opts, {"TargetFunctions"}, "ComplexExpand")
    mode = _target(ev(o["TargetFunctions"]) if "TargetFunctions" in o else None, "ComplexExpand")
    cvars = _cx_vars(v, "ComplexExpand")
    se = K.to_sp(e)
    cs = [K.to_sp(x) for x in cvars]
    subs, back = {}, []
    reals = {}
    for s in se.free_symbols:
        if s not in cs:
            reals[s] = sp.Symbol(s.name + "_x01r", real=True)
    for c in cs:
        if not isinstance(c, sp.Symbol):
            raise NotImplementedError("ComplexExpand: only symbols can be declared complex")
        x = sp.Symbol(c.name + "_x01x", real=True)
        y = sp.Symbol(c.name + "_x01y", real=True)
        subs[c] = x + sp.I * y
        back.append((c, x, y))
    work = se.xreplace(reals).xreplace(subs)
    out = sp.expand_complex(work)
    out = sp.simplify(out) if out.count_ops() > 4 * work.count_ops() + 40 else out
    rep = {}
    for c, x, y in back:
        if mode == "ReIm":
            rep[x], rep[y] = sp.re(sp.Symbol(c.name + "_x01c")), sp.im(sp.Symbol(c.name + "_x01c"))
        elif mode == "AbsArg":
            a, t = sp.Abs(sp.Symbol(c.name + "_x01c")), sp.arg(sp.Symbol(c.name + "_x01c"))
            rep[x], rep[y] = a * sp.cos(t), a * sp.sin(t)
        else:
            z = sp.Symbol(c.name + "_x01c")
            rep[x], rep[y] = (z + sp.conjugate(z)) / 2, (z - sp.conjugate(z)) / (2 * sp.I)
    res = out.xreplace(rep)
    res = res.xreplace({sp.Symbol(c.name + "_x01c"): c for c, _, _ in back})
    res = res.xreplace({r: s for s, r in reals.items()})
    if mode == "Conj":
        res = sp.expand(res)
    # numeric cross-check at a sample point
    pt = {}
    for k, c in enumerate(cs):
        pt[c] = sp.Rational(3, 7) + k * sp.Rational(1, 5) + sp.I * (sp.Rational(2, 9) + k * sp.Rational(1, 11))
    for k, s in enumerate(sorted(reals, key=lambda q: q.name)):
        pt[s] = sp.Rational(5, 13) + sp.Rational(k, 17)
    try:
        a = complex(sp.N(se.xreplace(pt), 30))
        b = complex(sp.N(res.xreplace(pt), 30))
        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
            raise ArithmeticError("ComplexExpand: result failed verification")
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return K.from_sp(res)


# ═══════════════════════════════════════════════ AbsoluteOptions ═══════════════════════════════════════════════

_SYMBOLIC_SETTINGS = {"Automatic", "All", "Full", "Inherited"}


def _coords(x, dim, out, fname):
    """Collect the coordinates of 2-D/3-D graphics primitives (for PlotRange -> All)."""
    if is_list(x):
        if x.args and all(_is_point(p, dim) for p in x.args):
            out.extend(_pt(p) for p in x.args)
            return
        for a in x.args:
            _coords(a, dim, out, fname)
        return
    if x is None or isinstance(x, (Symbol, str, int, float)):
        return
    if not isinstance(x, Expr):
        return
    h = x.head
    if h in ("Point", "Line", "Polygon", "Arrow", "BezierCurve", "BSplineCurve", "Triangle", "Simplex",
             "Tetrahedron", "Cuboid", "Rectangle", "Parallelogram", "InfiniteLine") and x.args:
        if h == "Rectangle" and len(x.args) == 1 and _is_point(x.args[0], 2):
            p = _pt(x.args[0])
            out.extend([p, (p[0] + 1, p[1] + 1)])
            return
        if h == "Cuboid" and len(x.args) == 1 and _is_point(x.args[0], 3):
            p = _pt(x.args[0])
            out.extend([p, tuple(c + 1 for c in p)])
            return
        if h in ("InfiniteLine", "Parallelogram"):
            raise NotImplementedError(f"{fname}: the extent of {h} is not supported")
        for a in (x.args if h in ("Rectangle", "Cuboid") else x.args[:1]):
            if _is_point(a, dim):
                out.append(_pt(a))
            else:
                _coords(a, dim, out, fname)
        return
    if h in ("Disk", "Circle", "Sphere", "Ball") and len(x.args) <= 2:
        c = x.args[0] if x.args else L(*([0] * dim))
        r = x.args[1] if len(x.args) > 1 else 1
        if not _is_point(c, dim):
            raise NotImplementedError(f"{fname}: the extent of {_show(x)} is not supported")
        p = _pt(c)
        rs = [float(K.to_sp(v)) for v in r.args] if is_list(r) else [float(K.to_sp(ev(Expr("N", r))))] * dim
        out.append(tuple(a - b for a, b in zip(p, rs)))
        out.append(tuple(a + b for a, b in zip(p, rs)))
        return
    if h == "Text" and len(x.args) >= 2 and _is_point(x.args[1], dim):
        out.append(_pt(x.args[1]))
        return
    if h in ("Style", "Annotation", "Tooltip", "Labeled", "Callout", "Hyperlink", "Mouseover", "Legended") and x.args:
        _coords(x.args[0], dim, out, fname)
        return
    if h == "GraphicsComplex" and len(x.args) >= 2:
        pts = [_pt(p) for p in x.args[0].args]
        out.extend(pts)
        return
    if h in ("Directive", "RGBColor", "GrayLevel", "Hue", "CMYKColor", "Opacity", "Thickness", "AbsoluteThickness",
             "PointSize", "AbsolutePointSize", "Dashing", "AbsoluteDashing", "EdgeForm", "FaceForm", "Arrowheads",
             "JoinForm", "CapForm", "LABColor", "XYZColor", "LCHColor", "LUVColor", "Specularity", "Glow"):
        return
    raise NotImplementedError(f"{fname}: the extent of {h} primitives is not supported")


def _is_point(p, dim):
    return is_list(p) and len(p.args) == dim and all(_real(v) is not None for v in p.args)


def _real(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = K.to_sp(v)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(s, sp.Basic) and s.is_real and s.is_number:
        return float(s)
    return None


def _pt(p):
    return tuple(_real(v) for v in p.args)


def _plot_range(g, od, fname):
    dim = 3 if g.head == "Graphics3D" else 2
    pr = od.get("PlotRange", Symbol("All"))
    pts = []
    _coords(g.args[0] if g.args else L(), dim, pts, fname)

    def bbox(i):
        if not pts:
            return (-1.0, 1.0)
        lo, hi = min(p[i] for p in pts), max(p[i] for p in pts)
        if lo == hi:
            half = max(abs(lo), 1.0) * 0.5 if dim == 2 else 0.5
            return (lo - half, hi + half)
        return (lo, hi)

    def axis(spec, i):
        if is_sym(spec, "All"):
            return bbox(i)
        if is_list(spec) and len(spec.args) == 2:
            lo, hi = bbox(i)
            a = lo if is_sym(spec.args[0], "All") else _real(spec.args[0])
            b = hi if is_sym(spec.args[1], "All") else _real(spec.args[1])
            if a is None or b is None:
                raise NotImplementedError(f"{fname}: PlotRange {_show(pr)} (Automatic/Full ends with outlier "
                                          "clipping) is not supported")
            return (a, b)
        if isinstance(spec, Symbol) and spec.name in ("Automatic", "Full"):
            raise NotImplementedError(f"{fname}: PlotRange -> {spec.name} (outlier clipping) is not reproduced; "
                                      "use All or explicit ranges")
        v = _real(spec)
        if v is not None:
            return (-abs(v), abs(v))
        raise ValueError(f"{fname}: invalid PlotRange {_show(spec)}")
    if is_list(pr) and len(pr.args) == dim and all(is_list(a) or isinstance(a, Symbol) for a in pr.args) \
            and not (dim == 2 and all(_real(a) is not None for a in pr.args)):
        ranges = [axis(pr.args[i], i) for i in range(dim)]
    elif is_list(pr) and len(pr.args) == 2 and all(_real(a) is not None or is_sym(a, "All") for a in pr.args):
        ranges = [bbox(i) for i in range(dim - 1)] + [axis(pr, dim - 1)]
    else:
        ranges = [axis(pr, i) for i in range(dim)] if not is_list(pr) else None
        if ranges is None:
            raise ValueError(f"{fname}: invalid PlotRange {_show(pr)}")
    return [tuple(float(c) for c in r) for r in ranges]


def _absolute_value(g, od, name, fname):
    is_g = isinstance(g, Expr) and g.head in ("Graphics", "Graphics3D")
    if is_g and name == "PlotRange":
        return L(*(L(*r) for r in _plot_range(g, od, fname)))
    if is_g and g.head == "Graphics" and name == "AspectRatio":
        ar = od.get("AspectRatio", Symbol("Automatic"))
        if is_sym(ar, "Automatic"):
            (x0, x1), (y0, y1) = _plot_range(g, od, fname)
            return (y1 - y0) / (x1 - x0)
        v = _real(ar)
        if v is None or v <= 0:
            raise NotImplementedError(f"{fname}: AspectRatio -> {_show(ar)} is not resolved")
        return v
    if name in od:
        v = od[name]
        if isinstance(v, Symbol) and v.name in _SYMBOLIC_SETTINGS:
            raise NotImplementedError(f"{fname}: the absolute value of {name} -> {v.name} is decided by the "
                                      "renderer/front end and is not available")
        return v
    raise NotImplementedError(f"{fname}: the default of {name} for {g.head if isinstance(g, Expr) else g} is not "
                              "catalogued (it is decided by the front end)")


@fn("AbsoluteOptions", area=AREA, status="added",
    notes="AbsoluteOptions[obj, name] and AbsoluteOptions[obj, {name1, ...}] give {name -> value} with the "
          "absolute settings: for Graphics/Graphics3D, PlotRange (missing or All: the bounding box of the "
          "primitives' coordinates, a single point padded by half its magnitude; explicit numeric ranges as given) "
          "and, for Graphics, AspectRatio (Automatic: height/width of that plot range, as documented); any option "
          "set explicitly to a concrete value (Graph, Graphics, ...) gives that value. Settings decided by the front "
          "end (unset options, Automatic/Full values other than the above, Lighting, ImageSize, ...), "
          "PlotRange -> Automatic (outlier clipping) and the one-argument form AbsoluteOptions[obj] (every option "
          "with its default) raise NotImplementedError; a non-option name raises TypeError. Check: exact (the "
          "bounding box of the listed coordinates) and reference (Plot ranges from the W16 Plot) in tests.")
def _absoluteoptions(obj: Expr, names: Expr = None) -> ListOf[Rule]:
    if names is None:
        if isinstance(obj, Symbol):
            raise Unevaluated()
        raise NotImplementedError("AbsoluteOptions: the one-argument form needs the complete catalogue of option "
                                  "defaults, which is decided by the Wolfram front end; give the option names")
    if not isinstance(obj, Expr) or isinstance(obj, Symbol):
        if isinstance(obj, Symbol):
            raise Unevaluated()
        raise TypeError(f"AbsoluteOptions: expected an object with options, not {_show(obj)}")
    single = not is_list(names)
    ns = [names] if single else list(names.args)
    keys = []
    for n in ns:
        if isinstance(n, Symbol):
            keys.append(n.name)
        elif isinstance(n, str):
            keys.append(n)
        else:
            raise TypeError(f"AbsoluteOptions: option names must be symbols, not {_show(n)}")
    od = {}
    for a in obj.args[1:]:
        if _is_rule(a) and _opt_key(a) is not None:
            od.setdefault(_opt_key(a), a.args[1])
    return L(*(Expr("Rule", Symbol(k), _absolute_value(obj, od, k, "AbsoluteOptions")) for k in keys))
