"""Wolfram batch W08: linear algebra extras, distance measures and optimisation.

* distance and dissimilarity measures (EuclideanDistance ... YuleDissimilarity, DistanceMatrix, dynamic time
  warping);
* structured-matrix constructors (CompanionMatrix, HadamardMatrix, JordanMatrix, PermutationMatrix,
  BlockDiagonalMatrix, ...).  Wolfram returns structured-array objects for several of these; MikoshiLang has no
  structured arrays, so they return the equivalent ordinary (Normal) matrix, as SparseArray does here;
* further decompositions (LDL, Bunch-Kaufman, polar, rank, Smith, Frobenius, Jordan form, core-nilpotent,
  Drazin inverse, Popov, ordered Schur, Karhunen-Loeve, Pfaffian);
* optimisation: exact linear programming (rational simplex, branch and bound for integer variables), quadratic,
  linear-fractional, convex / conic / semidefinite / geometric problems (SciPy), FindArgMin and friends,
  KnapsackSolve, FindShortestTour, FindCurvePath, Bayesian optimisation;
* matrix (normal-form) games and tree-game equilibria, Turing machines and rule plots.

Exact input is handled exactly where the problem allows it; inexact input gives machine (or input-precision)
results.  Anything that cannot be computed correctly raises a clear error rather than returning an
approximation silently.
"""
from __future__ import annotations

import itertools
import math
from fractions import Fraction
from functools import reduce

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import (Unevaluated, fn, record, S, R, L, rule, is_list, int_arg, pos_int, nonneg_int, list_arg,
                   has_float, numeric_out, mat_out)
from .wl_common import split_options, ev, apply, is_sym, is_rule, is_assoc

AREA = "linear algebra & optimisation (W08)"
ADDED = dict(area=AREA, status="added")


def _added(name, notes):
    record(name, AREA, "added", notes)


def _sym(name):
    return Symbol(name)


def _true(x):
    return x is True


def _str_opt(v):
    return v if isinstance(v, str) else None


def _need_scipy(fname):
    try:
        import scipy  # noqa: F401
        import scipy.optimize  # noqa: F401
        return scipy
    except ImportError:
        raise NotImplementedError(f"{fname}: this computation needs SciPy (pip install mikoshilang[signal])")


def _np():
    import numpy as np
    return np


def _is_num(v):
    return isinstance(v, sp.Basic) and v.is_number


def _items(x, fname, what="argument"):
    if is_list(x):
        return list(x.args)
    if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
        raise Unevaluated()
    raise ValueError(f"{fname}: {what} {x!r} must be a list")


def _dims(x):
    """Dimensions of a full (rectangular) array; scalars -> ()."""
    if not is_list(x):
        return ()
    if not x.args:
        return (0,)
    subs = [_dims(a) for a in x.args]
    if any(s != subs[0] for s in subs):
        return (len(x.args),)
    return (len(x.args),) + subs[0]


def _matrix(x, fname, square=False, allow_empty=False):
    """SymPy Matrix for a matrix argument (a symbol stays unevaluated)."""
    if not is_list(x):
        raise Unevaluated()
    d = _dims(x)
    if len(d) < 2 or d[1] == 0:
        if allow_empty and d == (0,):
            return sp.zeros(0, 0)
        raise ValueError(f"{fname}: {x} is not a rectangular matrix")
    if len(d) != 2:
        raise ValueError(f"{fname}: {x} is not a matrix (it has rank {len(d)})")
    M = sp.Matrix([[S(v) for v in r.args] for r in x.args])
    if square and M.rows != M.cols:
        raise ValueError(f"{fname}: {x} is not a square matrix")
    return M


def _mout(M):
    if M.rows == 0:
        return L()
    return L(*(L(*(R(M[i, j]) for j in range(M.cols))) for i in range(M.rows)))


def _inexact(M):
    return any(isinstance(a, sp.Float) for v in M for a in sp.preorder_traversal(v))


def _all_numeric(M):
    return all(v.is_number for v in M)


def _npmat(M):
    np = _np()
    vals = [[complex(sp.N(M[i, j])) for j in range(M.cols)] for i in range(M.rows)]
    a = np.array(vals, dtype=complex)
    if np.all(np.abs(a.imag) == 0):
        return a.real.astype(float)
    return a


def _from_np(a, tol=0.0):
    np = _np()
    a = np.asarray(a)
    if np.iscomplexobj(a):
        if np.all(np.abs(a.imag) <= tol * max(1.0, float(np.max(np.abs(a))) if a.size else 1.0)):
            a = a.real
    return R(a.tolist())


def _check_opts(opts, fname, allowed_values=None):
    """Options whose only supported value is Automatic (or listed values) raise otherwise."""
    for k, v in opts.items():
        ok = (allowed_values or {}).get(k)
        if is_sym(v, "Automatic") or (ok is not None and ok(v)):
            continue
        if ok is None and k in (allowed_values or {}):
            continue
        raise NotImplementedError(f"{fname}: the option {k} -> {v} is not supported")


def _target_structure(opts, fname):
    ts = opts.pop("TargetStructure", None)
    if ts is None or is_sym(ts, "Automatic") or ts in ("Dense", "Structured", "Sparse"):
        return ts
    if isinstance(ts, str):
        return ts
    raise ValueError(f"{fname}: invalid TargetStructure {ts}")


# ═══════════════════════════════ distances and dissimilarities ═══════════════════════════════

def _dist_args(u, v, fname, allow_scalar=True):
    """Two vectors (lists) of the same length, or two scalars -> (list_u, list_v, is_vector)."""
    if is_list(u) and is_list(v):
        if len(u.args) != len(v.args):
            raise ValueError(f"{fname}: {u} and {v} must be vectors of the same length")
        if any(is_list(a) for a in u.args + v.args):
            raise ValueError(f"{fname}: the arguments must be vectors (lists of scalars)")
        return [S(a) for a in u.args], [S(b) for b in v.args], True
    if is_list(u) or is_list(v):
        raise ValueError(f"{fname}: {u} and {v} must both be vectors or both be scalars")
    if isinstance(u, str) or isinstance(v, str):
        raise ValueError(f"{fname}: the arguments must be numeric or symbolic vectors")
    if not allow_scalar:
        raise ValueError(f"{fname}: the arguments must be vectors")
    return [S(u)], [S(v)], False


def _dist_out(value, u, v):
    return numeric_out(value, u, v) if (has_float(u) or has_float(v)) else R(value)


def _dist(name, f, notes, scalar=True):
    @fn(name, **ADDED, notes=notes)
    def impl(u, v):
        a, b, _vec = _dist_args(u, v, name, scalar)
        return _dist_out(f(a, b), u, v)
    impl.__name__ = "_" + name
    return impl


def _sqeuclid(a, b):
    return sp.Add(*[sp.Abs(x - y) ** 2 for x, y in zip(a, b)])


_dist("EuclideanDistance", lambda a, b: sp.sqrt(_sqeuclid(a, b)),
      "EuclideanDistance[u, v] = Sqrt[Total[Abs[u - v]^2]] for numeric or symbolic vectors (Abs[u - v] for scalars); "
      "results keep the precision of the input")
_dist("SquaredEuclideanDistance", _sqeuclid, "SquaredEuclideanDistance[u, v] = Total[Abs[u - v]^2]")
_dist("ManhattanDistance", lambda a, b: sp.Add(*[sp.Abs(x - y) for x, y in zip(a, b)]),
      "ManhattanDistance[u, v] = Total[Abs[u - v]]")
_dist("ChessboardDistance", lambda a, b: sp.Max(*[sp.Abs(x - y) for x, y in zip(a, b)]),
      "ChessboardDistance[u, v] = Max[Abs[u - v]] (Chebyshev distance)")


def _braycurtis(a, b):
    num = sp.Add(*[sp.Abs(x - y) for x, y in zip(a, b)])
    den = sp.Add(*[sp.Abs(x + y) for x, y in zip(a, b)])
    if den == 0:
        if num == 0:
            return sp.Integer(0)
        raise ValueError("BrayCurtisDistance: Total[Abs[u + v]] is zero")
    return num / den


def _canberra(a, b):
    tot = sp.Integer(0)
    for x, y in zip(a, b):
        num, den = sp.Abs(x - y), sp.Abs(x) + sp.Abs(y)
        if den == 0:
            continue                      # 0/0 terms (both entries zero) contribute nothing
        tot += num / den
    return tot


_dist("BrayCurtisDistance", _braycurtis, "BrayCurtisDistance[u, v] = Total[Abs[u - v]]/Total[Abs[u + v]]")
_dist("CanberraDistance", _canberra,
      "CanberraDistance[u, v] = Total[Abs[u - v]/(Abs[u] + Abs[v])] (terms with u_i = v_i = 0 are 0)")


def _norm2(a):
    return sp.sqrt(sp.Add(*[sp.Abs(x) ** 2 for x in a]))


def _cosine(a, b, fname="CosineDistance"):
    na, nb = _norm2(a), _norm2(b)
    if (na.is_number and na == 0) or (nb.is_number and nb == 0):
        raise ValueError(f"{fname}: the distance is undefined for a zero vector")
    return 1 - sp.Add(*[x * sp.conjugate(y) for x, y in zip(a, b)]) / (na * nb)


def _correlation(a, b):
    n = len(a)
    if n < 2:
        raise ValueError("CorrelationDistance: the vectors need at least two elements")
    ma, mb = sp.Add(*a) / n, sp.Add(*b) / n
    return _cosine([x - ma for x in a], [y - mb for y in b], "CorrelationDistance")


def _nsqeuclid(a, b):
    n = len(a)
    if n < 2:
        raise ValueError("NormalizedSquaredEuclideanDistance: the vectors need at least two elements")
    ma, mb = sp.Add(*a) / n, sp.Add(*b) / n
    da, db = [x - ma for x in a], [y - mb for y in b]
    den = sp.Add(*[sp.Abs(x) ** 2 for x in da]) + sp.Add(*[sp.Abs(y) ** 2 for y in db])
    if den.is_number and den == 0:
        raise ValueError("NormalizedSquaredEuclideanDistance: both vectors are constant")
    return sp.Rational(1, 2) * sp.Add(*[sp.Abs(x - y) ** 2 for x, y in zip(da, db)]) / den


_dist("CosineDistance", _cosine, "CosineDistance[u, v] = 1 - u.Conjugate[v]/(Norm[u] Norm[v]); a zero vector raises",
      scalar=False)
_dist("CorrelationDistance", _correlation,
      "CorrelationDistance[u, v] = 1 - (u - Mean[u]).Conjugate[v - Mean[v]]/(Norm[u - Mean[u]] Norm[v - Mean[v]])",
      scalar=False)
_dist("NormalizedSquaredEuclideanDistance", _nsqeuclid,
      "NormalizedSquaredEuclideanDistance[u, v] = Norm[(u - Mean[u]) - (v - Mean[v])]^2/(2 (Norm[u - Mean[u]]^2 + "
      "Norm[v - Mean[v]]^2))", scalar=False)


@fn("BinaryDistance", **ADDED,
    notes="BinaryDistance[u, v] = Boole[!(u == v)]: 0 for equal expressions (1 == 1. counts as equal), 1 for "
          "different ones; Boole[u != v] when equality cannot be decided")
def _binarydistance(u, v):
    r = ev(Expr("Equal", u, v))
    if r is True:
        return 0
    if r is False:
        return 1
    if is_list(u) and is_list(v) and len(u.args) != len(v.args):
        return 1
    return Expr("Boole", Expr("Unequal", u, v))


def _bool_counts(u, v, fname):
    if not (is_list(u) and is_list(v)):
        if isinstance(u, Symbol) or isinstance(v, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: the arguments must be Boolean vectors")
    if len(u.args) != len(v.args):
        raise ValueError(f"{fname}: the vectors must have the same length")

    def b(x):
        if x is True or x is False:
            return x
        if K.is_int(x) and int(x) in (0, 1) and not isinstance(x, bool):
            return int(x) == 1
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not is_list(x)):
            raise Unevaluated()
        raise ValueError(f"{fname}: {x} is not a Boolean value (True/False or 0/1)")
    n = {(1, 1): 0, (1, 0): 0, (0, 1): 0, (0, 0): 0}
    for x, y in zip(u.args, v.args):
        n[(int(b(x)), int(b(y)))] += 1
    return n[(1, 1)], n[(1, 0)], n[(0, 1)], n[(0, 0)]


def _ratio(num, den):
    if den == 0:
        return 0
    return R(sp.Rational(num, den))


def _booldist(name, f, formula):
    @fn(name, **ADDED, notes=f"{name}[u, v] = {formula} for Boolean vectors (True/False or 0/1), where nij counts "
                             "positions with u = i, v = j; 0/0 gives 0")
    def impl(u, v):
        n11, n10, n01, n00 = _bool_counts(u, v, name)
        return f(n11, n10, n01, n00)
    impl.__name__ = "_" + name
    return impl


_booldist("DiceDissimilarity", lambda a, b, c, d: _ratio(b + c, 2 * a + b + c), "(n10 + n01)/(2 n11 + n10 + n01)")
_booldist("JaccardDissimilarity", lambda a, b, c, d: _ratio(b + c, a + b + c), "(n10 + n01)/(n11 + n10 + n01)")
_booldist("MatchingDissimilarity", lambda a, b, c, d: _ratio(b + c, a + b + c + d), "(n10 + n01)/Length[u]")
_booldist("RogersTanimotoDissimilarity", lambda a, b, c, d: _ratio(2 * (b + c), a + d + 2 * (b + c)),
          "2 (n10 + n01)/(n11 + n00 + 2 (n10 + n01))")
_booldist("RussellRaoDissimilarity", lambda a, b, c, d: _ratio(b + c + d, a + b + c + d), "(n10 + n01 + n00)/Length[u]")
_booldist("SokalSneathDissimilarity", lambda a, b, c, d: _ratio(2 * (b + c), a + 2 * (b + c)),
          "2 (n10 + n01)/(n11 + 2 (n10 + n01))")
_booldist("YuleDissimilarity", lambda a, b, c, d: _ratio(2 * b * c, a * d + b * c), "2 n10 n01/(n11 n00 + n10 n01)")


def _is_boolish(x):
    return x is True or x is False


def _default_distance(a, b):
    """Default element distance (DistanceMatrix): Euclidean for numbers, EditDistance for strings, Jaccard for
    Boolean vectors."""
    if isinstance(a, str) and isinstance(b, str):
        return _sym("EditDistance")
    if is_list(a) and a.args and all(_is_boolish(t) for t in a.args):
        return _sym("JaccardDissimilarity")
    if _is_boolish(a) or _is_boolish(b):
        return Expr("Function", Expr("Boole", Expr("UnsameQ", Expr("Slot", 1), Expr("Slot", 2))))
    return _sym("EuclideanDistance")


_DM_OPTS = {"DistanceFunction", "PerformanceGoal", "WorkingPrecision", "FeatureExtractor", "RandomSeeding"}


def _dist_fn(opts, fname):
    d = opts.get("DistanceFunction")
    for k in ("FeatureExtractor",):
        if k in opts and not (is_sym(opts[k], "Identity") or is_sym(opts[k], "Automatic")):
            raise NotImplementedError(f"{fname}: the option {k} is not supported")
    return None if d is None or is_sym(d, "Automatic") else d


def _elements(x, fname):
    if is_assoc(x):
        return KL.assoc_values(x, ev)
    return _items(x, fname, "data")


@fn("DistanceMatrix", **ADDED,
    notes="DistanceMatrix[{u1, ...}] and DistanceMatrix[{u1, ...}, {v1, ...}] matrix of distances (Euclidean for "
          "numeric data, EditDistance for strings, JaccardDissimilarity for Boolean vectors) or DistanceFunction -> f; "
          "associations use their values; PerformanceGoal is accepted (results are always computed exactly or at "
          "input precision)")
def _distancematrix(us, *rest):
    pos, opts = split_options(rest, _DM_OPTS, "DistanceMatrix")
    if len(pos) > 1:
        raise ValueError("DistanceMatrix called with too many arguments")
    a = _elements(us, "DistanceMatrix")
    b = _elements(pos[0], "DistanceMatrix") if pos else a
    df = _dist_fn(opts, "DistanceMatrix")
    rows = []
    for x in a:
        row = []
        for y in b:
            f = df if df is not None else _default_distance(x, y)
            row.append(apply(f, x, y))
        rows.append(L(*row))
    return L(*rows)


record("DistanceFunction", AREA, "added",
       "option name DistanceFunction -> f (Automatic or a two-argument distance) for Nearest, DistanceMatrix, "
       "WarpingDistance, FeatureDistance, ...; an inert symbol")


# ─────────────────────────── dynamic time warping ───────────────────────────

_DTW_OPTS = {"DistanceFunction", "Method"}


def _seq(x, fname):
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise ValueError(f"{fname}: {x} is not a sequence (a list of numbers, Booleans or vectors)")
    if not x.args:
        raise ValueError(f"{fname}: the sequences must not be empty")
    return list(x.args)


def _dtw_window(win, n, m, fname):
    """Predicate (i, j) -> bool (0-based) for the search window."""
    if win is None or is_sym(win, "Automatic"):
        return lambda i, j: True
    kind, r = "SlantedBand", win
    if is_list(win) and len(win.args) == 2 and isinstance(win.args[0], str):
        kind, r = win.args
    try:
        rv = float(sp.N(S(r)))
    except Exception:
        raise ValueError(f"{fname}: invalid window specification {win}")
    if not rv >= 0:
        raise ValueError(f"{fname}: the window radius must be non-negative")
    if max(n, m) <= 2 * rv + 1:
        return lambda i, j: True
    if kind == "Band":
        return lambda i, j: abs(i - j) <= rv
    if kind == "SlantedBand":
        if n == 1 or m == 1:
            return lambda i, j: True
        return lambda i, j: abs(i * (m - 1) / (n - 1) - j) <= rv
    if kind == "Parallelogram":
        raise NotImplementedError(f"{fname}: the \"Parallelogram\" window is not supported")
    raise ValueError(f"{fname}: unknown window {kind!r}")


def _dtw(s1, s2, rest, fname):
    pos, opts = split_options(rest, _DTW_OPTS, fname)
    if len(pos) > 1:
        raise ValueError(f"{fname} called with too many arguments")
    a, b = _seq(s1, fname), _seq(s2, fname)
    meth = opts.get("Method")
    if meth is not None and not is_sym(meth, "Automatic"):
        raise NotImplementedError(f"{fname}: Method -> {meth} is not supported (only the full-match DTW)")
    df = _dist_fn(opts, fname)
    if df is None:
        df = _sym("MatchingDissimilarity") if all(_is_boolish(t) for t in a + b) or all(
            is_list(t) and all(_is_boolish(q) for q in t.args) for t in a + b) else _sym("EuclideanDistance")
        if df.name == "MatchingDissimilarity" and all(_is_boolish(t) for t in a + b):
            df = _sym("BinaryDistance")
    n, m = len(a), len(b)
    inwin = _dtw_window(pos[0] if pos else None, n, m, fname)
    cache = {}

    def cost(i, j):
        key = (i, j)
        if key not in cache:
            v = apply(df, a[i], b[j])
            try:
                c = complex(sp.N(S(v)))
            except Exception:
                raise ValueError(f"{fname}: the distance between {a[i]} and {b[j]} is not numeric: {v}")
            if abs(c.imag) > 1e-12 * max(1.0, abs(c.real)) or c.real < 0:
                raise ValueError(f"{fname}: the distance {v} is not a non-negative real number")
            cache[key] = c.real
        return cache[key]
    INF = float("inf")
    D = [[INF] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if not inwin(i, j):
                continue
            c = cost(i, j)
            if i == 0 and j == 0:
                D[i][j] = c
                continue
            best = min(D[i - 1][j - 1] if i and j else INF, D[i - 1][j] if i else INF, D[i][j - 1] if j else INF)
            D[i][j] = c + best
    if D[n - 1][m - 1] == INF:
        raise ValueError(f"{fname}: the search window admits no warping path")
    # backtrack: prefer the diagonal step, then a step in s1, then a step in s2
    i, j, path = n - 1, m - 1, [(n - 1, m - 1)]
    while i or j:
        cands = []
        if i and j:
            cands.append((D[i - 1][j - 1], 0, i - 1, j - 1))
        if i:
            cands.append((D[i - 1][j], 1, i - 1, j))
        if j:
            cands.append((D[i][j - 1], 2, i, j - 1))
        _, _, i, j = min(cands)
        path.append((i, j))
    path.reverse()
    return D[n - 1][m - 1], path


@fn("WarpingDistance", **ADDED,
    notes="WarpingDistance[s1, s2] dynamic time warping distance (sum of element distances along the optimal "
          "warping path; EuclideanDistance for numbers and vectors, BinaryDistance/MatchingDissimilarity for "
          "Booleans, or DistanceFunction -> f); WarpingDistance[s1, s2, win] with win = r | {\"SlantedBand\", r} | "
          "{\"Band\", r} | Automatic; the result is a machine real; the Parallelogram window and Method settings "
          "raise")
def _warpingdistance(s1, s2, *rest):
    d, _ = _dtw(s1, s2, rest, "WarpingDistance")
    return float(d)


@fn("WarpingCorrespondence", **ADDED,
    notes="WarpingCorrespondence[s1, s2] = {n, m}: the index lists of the optimal dynamic time warping path (same "
          "windows and DistanceFunction as WarpingDistance; ties prefer the diagonal step)")
def _warpingcorrespondence(s1, s2, *rest):
    _, path = _dtw(s1, s2, rest, "WarpingCorrespondence")
    return L(L(*(p[0] + 1 for p in path)), L(*(p[1] + 1 for p in path)))


@fn("FeatureDistance", **ADDED,
    notes="FeatureDistance[e1, e2, extractor] distance between extractor[e1] and extractor[e2] (EuclideanDistance "
          "for numeric features, or DistanceFunction -> f); extractor is any function (e.g. a pure Function or "
          "Identity); FeatureDistance[extractor][e1, e2] operator form; FeatureExtractorFunction objects (machine-"
          "learned extractors from FeatureExtraction) are not available and raise")
def _featuredistance(*args):
    pos, opts = split_options(args, {"DistanceFunction"}, "FeatureDistance")
    if len(pos) == 1:
        raise Unevaluated()
    if len(pos) != 3:
        raise ValueError("FeatureDistance takes two examples and a feature extractor")
    e1, e2, ext = pos
    if isinstance(ext, Expr) and ext.head == "FeatureExtractorFunction" or isinstance(ext, str):
        raise NotImplementedError("FeatureDistance: FeatureExtractorFunction / named extractors need the "
                                  "machine-learning feature extractors, which are not available")
    f1, f2 = apply(ext, e1), apply(ext, e2)
    df = _dist_fn(opts, "FeatureDistance") or _default_distance(f1, f2)
    return apply(df, f1, f2)


def _featuredistance_op(head, args, ev_):
    if len(head.args) == 1 and len(args) == 2:
        return ev_(Expr("FeatureDistance", args[0], args[1], head.args[0]))
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["FeatureDistance"] = _featuredistance_op


# ═══════════════════════════════ Band (SparseArray band specifications) ═══════════════════════════════

def _depth(x):
    d = 0
    while is_list(x):
        d += 1
        if not x.args:
            break
        x = x.args[0]
    return d


def _band_parts(band, rank, shape, fname="SparseArray"):
    a = band.args
    if not 1 <= len(a) <= 3:
        raise ValueError(f"{fname}: Band takes a start position, an optional end and an optional step")

    def vec(x, what):
        if K.is_int(x):
            return [int(x)] * rank
        if is_list(x) and len(x.args) == rank and all(K.is_int(i) for i in x.args):
            return [int(i) for i in x.args]
        raise ValueError(f"{fname}: Band {what} {x} must be a list of {rank} integers")

    def resolve(p):
        out = []
        for i, d in zip(p, shape):
            if d is None:
                out.append(i)
            else:
                out.append(d + 1 + i if i < 0 else i)
        return out
    start = resolve(vec(a[0], "start"))
    end = None if len(a) < 2 or is_sym(a[1], "Automatic") else resolve(vec(a[1], "end"))
    step = vec(a[2], "step") if len(a) == 3 else [1] * rank
    if any(s <= 0 for s in start):
        raise ValueError(f"{fname}: Band start {list(start)} must be a valid position")
    return start, end, step, len(a) >= 2 and not is_sym(a[1], "Automatic")


def band_rank(band, rhs):
    """Rank of the array a Band rule belongs to (the length of its start position)."""
    s = band.args[0] if band.args else None
    return len(s.args) if is_list(s) else None


def band_entries(band, rhs, rank, shape):
    """{position tuple: value} for a rule Band[...] -> rhs in an array of the given rank and shape (shape entries may
    be None while the dimensions are being inferred)."""
    start, end, step, has_end = _band_parts(band, rank, shape)
    limit = end if end is not None else [d if d is not None else 10 ** 9 for d in shape]
    out = {}
    d = _depth(rhs)
    blocks = None
    if d == rank and rank > 0 and is_list(rhs):
        blocks, cyclic = [rhs], False
    elif d == rank + 1 and rank > 0 and is_list(rhs) and all(_depth(b) == rank for b in rhs.args):
        blocks, cyclic = list(rhs.args), has_end
    if blocks is not None:
        pos, k = list(start), 0
        while all(p <= lim for p, lim in zip(pos, limit)):
            blk = blocks[k % len(blocks)]
            bd = _dims(blk)
            if len(bd) != rank:
                raise ValueError(f"SparseArray: Band block {blk} is not a full array of rank {rank}")
            for idx in itertools.product(*(range(n) for n in bd)):
                p = tuple(pos[i] + idx[i] for i in range(rank))
                if all(p[i] <= limit[i] for i in range(rank)):
                    v = blk
                    for i in idx:
                        v = v.args[i]
                    out.setdefault(p, v)
            pos = [pos[i] + bd[i] for i in range(rank)]
            k += 1
            if not cyclic and k >= len(blocks):
                break
        return out
    if is_list(rhs) and d == 1:
        vals, cyclic = list(rhs.args), has_end
    elif not is_list(rhs):
        vals, cyclic = [rhs], True
    else:
        raise ValueError(f"SparseArray: invalid Band values {rhs}")
    if not vals:
        return out
    pos, k = list(start), 0
    while all(p <= lim and p >= 1 for p, lim in zip(pos, limit)):
        if not cyclic and k >= len(vals):
            break
        out.setdefault(tuple(pos), vals[k % len(vals)])
        pos = [pos[i] + step[i] for i in range(rank)]
        k += 1
        if all(s == 0 for s in step):
            break
    return out


@fn("Band", **ADDED,
    notes="Band[start], Band[start, end], Band[start, end | Automatic, step] in SparseArray rules: Band[...] -> v "
          "repeats v along the diagonal to the edge (or end); -> {v1, ...} fills successive positions (cycling when "
          "an end is given); -> array inserts a block; -> {a1, ...} places blocks along the diagonal (cyclically up "
          "to an explicit end, clipped at the edge); negative positions count from the end; Band itself is inert")
def _band(*args):
    raise Unevaluated()


# ═══════════════════════════════ structured-matrix constructors ═══════════════════════════════

_STRUCT_NOTE = ("; MikoshiLang has no structured arrays, so the result is the ordinary matrix (Wolfram's Normal[...] "
                "or TargetStructure -> \"Dense\" form) and structured-array properties are not available")
_TS = {"TargetStructure", "WorkingPrecision"}


def _wp(x, opts, fname):
    wp = opts.get("WorkingPrecision")
    if wp is None or is_sym(wp, "Infinity") or is_sym(wp, "Automatic"):
        return x
    return ev(Expr("N", x, wp) if not is_sym(wp, "MachinePrecision") else Expr("N", x))


def _zero_eq(v):
    if isinstance(v, sp.Basic) and v.is_number and _inexact(sp.Matrix([v])):
        return abs(complex(sp.N(v))) < 1e-10
    try:
        return sp.simplify(v) == 0
    except Exception:
        return False


def _companion_coeffs(args, fname):
    if len(args) == 1 and is_list(args[0]) and len(args[0].args) == 2 and not all(
            isinstance(a, (int, float)) for a in args[0].args) and isinstance(args[0].args[1], Symbol) \
            and not K.is_number(args[0].args[0]) and not is_list(args[0].args[0]):
        args = (args[0].args[0], args[0].args[1]) + tuple(args[1:])
    if len(args) >= 2 and isinstance(args[1], Symbol) and args[1].name not in ("Right", "Left", "Top", "Bottom"):
        poly, x = S(args[0]), sp.Symbol(args[1].name)
        try:
            P = sp.Poly(sp.expand(poly), x)
        except sp.PolynomialError:
            raise ValueError(f"{fname}: {args[0]} is not a polynomial in {args[1]}")
        if P.degree() < 1:
            raise ValueError(f"{fname}: the polynomial must have positive degree in {args[1]}")
        cs = P.all_coeffs()[::-1]           # c0, c1, ..., cn
        lead = cs[-1]
        return [sp.cancel(c / lead) for c in cs[:-1]], args[2:]
    if not is_list(args[0]):
        raise Unevaluated()
    if not args[0].args:
        raise ValueError(f"{fname}: the coefficient vector must not be empty")
    return [S(c) for c in args[0].args], args[1:]


@fn("CompanionMatrix", **ADDED,
    notes="CompanionMatrix[{c0, ..., c(n-1)}] (implicit monic x^n term) and CompanionMatrix[poly, x] (normalised by "
          "the leading coefficient): ones on the subdiagonal and -c_i down the last column; CompanionMatrix[..., dir] "
          "with dir = Right (default), Bottom (transpose), Top (-c(n-1), ..., -c0 along the first row) or Left "
          "(its transpose); TargetStructure -> \"Dense\" | \"Sparse\" accepted (dense result)")
def _companionmatrix(*args):
    pos, opts = split_options(args, {"TargetStructure"}, "CompanionMatrix")
    if not pos:
        raise ValueError("CompanionMatrix needs a coefficient list or a polynomial and variable")
    cs, rest = _companion_coeffs(tuple(pos), "CompanionMatrix")
    direction = "Right"
    if rest:
        if len(rest) > 1 or not isinstance(rest[0], Symbol) or rest[0].name not in ("Right", "Left", "Top", "Bottom"):
            raise ValueError("CompanionMatrix: the direction must be Right, Left, Top or Bottom")
        direction = rest[0].name
    n = len(cs)
    M = sp.zeros(n, n)
    for i in range(1, n):
        M[i, i - 1] = 1
    for i in range(n):
        M[i, n - 1] = -cs[i]
    if direction in ("Top", "Left"):
        T = sp.zeros(n, n)
        for i in range(1, n):
            T[i, i - 1] = 1
        for j in range(n):
            T[0, j] = -cs[n - 1 - j]
        M = T
    if direction in ("Bottom", "Left"):
        M = M.T
    return _mout(M)


def _sylvester(n):
    H = [[1]]
    while len(H) < n:
        H = [r + r for r in H] + [r + [-v for v in r] for r in H]
    return H


def _sequency(row):
    return sum(1 for a, b in zip(row, row[1:]) if a != b)


@fn("HadamardMatrix", **ADDED,
    notes="HadamardMatrix[n] (n a power of 2): the normalised (orthogonal, symmetric) Walsh-Hadamard matrix with "
          "entries +-1/Sqrt[n], rows in sequency order (Method -> \"Sequency\", default), Sylvester order "
          "(\"BitComplement\") or Paley/dyadic order (\"GrayCode\"); WorkingPrecision -> p gives numeric entries"
          + _STRUCT_NOTE)
def _hadamardmatrix(n, *rest):
    pos, opts = split_options(rest, _TS | {"Method"}, "HadamardMatrix")
    if pos:
        raise ValueError("HadamardMatrix takes one size argument")
    k = pos_int(n, "HadamardMatrix", "size")
    if k & (k - 1):
        raise ValueError(f"HadamardMatrix: the size {k} must be a power of 2")
    meth = opts.get("Method", Symbol("Automatic"))
    meth = "Sequency" if is_sym(meth, "Automatic") else meth
    if meth not in ("Sequency", "BitComplement", "GrayCode"):
        raise ValueError(f"HadamardMatrix: unknown Method {meth}")
    H = _sylvester(k)
    if meth != "BitComplement":
        by_seq = {_sequency(r): r for r in H}
        if meth == "Sequency":
            H = [by_seq[s] for s in range(k)]
        else:
            def inv_gray(g):
                b = 0
                while g:
                    b ^= g
                    g >>= 1
                return b
            H = [by_seq[inv_gray(i)] for i in range(k)]
    c = 1 / sp.sqrt(k)
    return _wp(_mout(sp.Matrix(H) * c), opts, "HadamardMatrix")


@fn("JordanMatrix", **ADDED,
    notes="JordanMatrix[l, n] n x n Jordan block (l on the diagonal, 1 on the superdiagonal); JordanMatrix[l, n, "
          "Right] puts the ones below the diagonal (the transpose); l must be a scalar" + _STRUCT_NOTE)
def _jordanmatrix(lam, n, *rest):
    pos, opts = split_options(rest, {"TargetStructure"}, "JordanMatrix")
    if is_list(lam):
        raise ValueError("JordanMatrix: the diagonal value must be a scalar")
    k = pos_int(n, "JordanMatrix", "dimension")
    d = "Left"
    if pos:
        if len(pos) > 1 or not (is_sym(pos[0], "Left") or is_sym(pos[0], "Right")):
            raise ValueError("JordanMatrix: the direction must be Left or Right")
        d = pos[0].name
    rows = []
    for i in range(k):
        row = []
        for j in range(k):
            if i == j:
                row.append(lam)
            elif (d == "Left" and j == i + 1) or (d == "Right" and i == j + 1):
                row.append(1)
            else:
                row.append(0)
        rows.append(L(*row))
    return L(*rows)


def _perm_list(p, fname):
    if isinstance(p, Expr) and p.head == "TwoWayRule" and len(p.args) == 2:
        p = Expr("Cycles", L(L(*p.args)))
    if isinstance(p, Expr) and p.head == "Cycles":
        if len(p.args) != 1 or not is_list(p.args[0]):
            raise ValueError(f"{fname}: invalid Cycles object {p}")
        cyc = [[int_arg(i, fname, "cycle element") for i in c.args] for c in p.args[0].args]
        n = max([max(c) for c in cyc if c] + [0])
        perm = list(range(1, n + 1))
        for c in cyc:
            for a, b in zip(c, c[1:] + c[:1]):
                perm[a - 1] = b
        return perm
    if is_list(p) and all(K.is_int(i) for i in p.args):
        perm = [int(i) for i in p.args]
        if sorted(perm) != list(range(1, len(perm) + 1)):
            raise ValueError(f"{fname}: {p} is not a permutation list")
        return perm
    if is_list(p) and p.args and all(is_list(r) for r in p.args):
        M = _matrix(p, fname, square=True)
        perm = []
        for i in range(M.rows):
            ones = [j for j in range(M.cols) if M[i, j] == 1]
            if len(ones) != 1 or any(M[i, j] != 0 for j in range(M.cols) if j != ones[0]):
                raise ValueError(f"{fname}: {p} is not a permutation matrix")
            perm.append(ones[0] + 1)
        if sorted(perm) != list(range(1, len(perm) + 1)):
            raise ValueError(f"{fname}: {p} is not a permutation matrix")
        return perm
    if isinstance(p, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {p} is not a permutation list, Cycles object or permutation matrix")


@fn("PermutationMatrix", **ADDED,
    notes="PermutationMatrix[permv] has a 1 in column permv[[i]] of row i; also from Cycles[...] (size = largest "
          "moved point), i <-> j, or a permutation matrix (validated); PermutationMatrix[perm, n] pads to n x n; "
          "Det is Signature[permv]" + _STRUCT_NOTE)
def _permutationmatrix(p, *rest):
    pos, opts = split_options(rest, _TS, "PermutationMatrix")
    perm = _perm_list(p, "PermutationMatrix")
    n = len(perm)
    if pos:
        n = pos_int(pos[0], "PermutationMatrix", "size")
        if n < len(perm):
            raise ValueError(f"PermutationMatrix: the size {n} is smaller than the permutation")
        perm = perm + list(range(len(perm) + 1, n + 1))
    if n == 0:
        raise ValueError("PermutationMatrix: the permutation is empty")
    M = L(*(L(*(1 if j + 1 == perm[i] else 0 for j in range(n))) for i in range(n)))
    return _wp(M, opts, "PermutationMatrix")


def _square_blocks(blocks, fname):
    out = []
    for b in blocks:
        if is_list(b) and not b.args:
            continue
        M = _matrix(b, fname)
        if M.rows != M.cols:
            raise ValueError(f"{fname}: the diagonal block {b} is not square")
        out.append(M)
    return out


def _is_blockdiag_list(x):
    return is_list(x) and x.args and all(is_list(b) and (not b.args or (_dims(b) and len(_dims(b)) == 2))
                                         for b in x.args) and len(_dims(x)) != 2


@fn("BlockDiagonalMatrix", **ADDED,
    notes="BlockDiagonalMatrix[{d1, d2, ...}] the block diagonal matrix with square blocks di ({} blocks are 0 x 0); "
          "BlockDiagonalMatrix[mat] accepts a (block diagonal) square matrix" + _STRUCT_NOTE)
def _blockdiagonalmatrix(x, *rest):
    pos, opts = split_options(rest, {"TargetStructure"}, "BlockDiagonalMatrix")
    if pos:
        raise ValueError("BlockDiagonalMatrix takes one argument")
    if not is_list(x):
        raise Unevaluated()
    d = _dims(x)
    if len(d) == 2:
        _matrix(x, "BlockDiagonalMatrix", square=True)
        return x
    if len(d) == 3 or _is_blockdiag_list(x):
        blocks = _square_blocks(x.args, "BlockDiagonalMatrix")
        if not blocks:
            return L()
        return _mout(sp.diag(*blocks))
    raise ValueError(f"BlockDiagonalMatrix: {x} is neither a list of square matrices nor a matrix")


def _triangular_check(x, fname, lower):
    M = _matrix(x, fname, square=True)
    for i in range(M.rows):
        for j in range(M.cols):
            if (j > i if lower else j < i) and not _zero_eq(M[i, j]):
                kind = "lower" if lower else "upper"
                raise ValueError(f"{fname}: {x} is not a {kind} triangular matrix (use "
                                 f"{fname}[{'Lower' if lower else 'Upper'}Triangularize[m]] to keep only the "
                                 f"{kind} triangular entries)")
    return x


def _ltm(x, *rest):
    split_options(rest, {"TargetStructure"}, "LowerTriangularMatrix")
    return _triangular_check(x, "LowerTriangularMatrix", True)


def _utm(x, *rest):
    split_options(rest, {"TargetStructure"}, "UpperTriangularMatrix")
    return _triangular_check(x, "UpperTriangularMatrix", False)


fn("LowerTriangularMatrix", **ADDED,
   notes="LowerTriangularMatrix[m] validates a square lower triangular matrix (entries above the diagonal must be "
         "zero, else a clear error)" + _STRUCT_NOTE)(_ltm)
fn("UpperTriangularMatrix", **ADDED,
   notes="UpperTriangularMatrix[m] validates a square upper triangular matrix (entries below the diagonal must be "
         "zero, else a clear error)" + _STRUCT_NOTE)(_utm)


def _block_triangular(x, fname, lower):
    M = _matrix(x, fname, square=True)
    n = M.rows
    # finest block partition: a cut after k (1 <= k < n) is allowed when the off block is zero
    cuts = [0]
    for k in range(1, n):
        ok = all(_zero_eq(M[i, j]) for i in range(k) for j in range(k, n)) if lower else \
            all(_zero_eq(M[i, j]) for i in range(k, n) for j in range(k))
        if ok:
            cuts.append(k)
    return x


fn("BlockLowerTriangularMatrix", **ADDED,
   notes="BlockLowerTriangularMatrix[lmat] for a square matrix (any square matrix is block lower triangular with "
         "a single block, as in Wolfram)" + _STRUCT_NOTE)(
    lambda x, *rest: (split_options(rest, {"TargetStructure"}, "BlockLowerTriangularMatrix"),
                      _block_triangular(x, "BlockLowerTriangularMatrix", True))[1])
fn("BlockUpperTriangularMatrix", **ADDED,
   notes="BlockUpperTriangularMatrix[umat] for a square matrix (any square matrix is block upper triangular with "
         "a single block, as in Wolfram)" + _STRUCT_NOTE)(
    lambda x, *rest: (split_options(rest, {"TargetStructure"}, "BlockUpperTriangularMatrix"),
                      _block_triangular(x, "BlockUpperTriangularMatrix", False))[1])


def _triangle_rows(x, fname, conj):
    """Full matrix from a symmetric/Hermitian matrix or a triangular row specification."""
    rows = _items(x, fname, "matrix")
    if not rows or not all(is_list(r) for r in rows):
        raise ValueError(f"{fname}: {x} is not a matrix")
    n = len(rows)
    lens = [len(r.args) for r in rows]
    f = (lambda v: sp.conjugate(v)) if conj else (lambda v: v)
    if all(k == n for k in lens):
        M = sp.Matrix([[S(v) for v in r.args] for r in rows])
        for i in range(n):
            for j in range(i + 1, n):
                if not _zero_eq(M[i, j] - f(M[j, i])):
                    raise ValueError(f"{fname}: {x} is not {'Hermitian' if conj else 'symmetric'}")
            if conj and not _zero_eq(sp.im(M[i, i])) and M[i, i].is_number:
                raise ValueError(f"{fname}: {x} is not Hermitian (complex diagonal entry)")
        return _mout(M)
    M = sp.zeros(n, n)
    if lens == list(range(n, 0, -1)):          # upper rows {{h11, h12, ...}, {h22, ...}, ...}
        for i, r in enumerate(rows):
            for k, v in enumerate(r.args):
                j = i + k
                M[i, j] = S(v)
                M[j, i] = f(S(v)) if i != j else S(v)
    elif lens == list(range(1, n + 1)):        # lower rows {{h11}, {h21, h22}, ...}
        for i, r in enumerate(rows):
            for j, v in enumerate(r.args):
                M[i, j] = S(v)
                M[j, i] = f(S(v)) if i != j else S(v)
    else:
        raise ValueError(f"{fname}: {x} is neither a square matrix nor a triangular list of rows")
    return _mout(M)


fn("SymmetricMatrix", **ADDED,
   notes="SymmetricMatrix[smat] validates a symmetric matrix; SymmetricMatrix[{{s11, s12, ...}, {s22, ...}, ...}] "
         "or the lower-triangular rows {{s11}, {s21, s22}, ...} fill in the other triangle" + _STRUCT_NOTE)(
    lambda x, *rest: (split_options(rest, {"TargetStructure"}, "SymmetricMatrix"),
                      _triangle_rows(x, "SymmetricMatrix", False))[1])
fn("HermitianMatrix", **ADDED,
   notes="HermitianMatrix[hmat] validates a Hermitian matrix; HermitianMatrix[{{h11, h12, ...}, {h22, ...}, ...}] or "
         "the lower rows {{h11}, {h21, h22}, ...} fill in the conjugate entries" + _STRUCT_NOTE)(
    lambda x, *rest: (split_options(rest, {"TargetStructure"}, "HermitianMatrix"),
                      _triangle_rows(x, "HermitianMatrix", True))[1])


def _orth_check(x, fname, conj):
    M = _matrix(x, fname)
    A = M.H if conj else M.T
    G = A * M if M.rows >= M.cols else M * A
    I = sp.eye(G.rows)
    if not all(_zero_eq(sp.expand(v)) for v in (G - I)):
        what = "unitary" if conj else "orthogonal"
        raise ValueError(f"{fname}: {x} is not an {what} matrix" if not conj else f"{fname}: {x} is not a unitary "
                         "matrix")
    return x


fn("OrthogonalMatrix", **ADDED,
   notes="OrthogonalMatrix[omat] validates a (square, tall or wide) orthogonal matrix: Transpose[o].o (or "
         "o.Transpose[o]) is the identity" + _STRUCT_NOTE)(
    lambda x, *rest: (split_options(rest, {"TargetStructure"}, "OrthogonalMatrix"),
                      _orth_check(x, "OrthogonalMatrix", False))[1])
fn("UnitaryMatrix", **ADDED,
   notes="UnitaryMatrix[umat] validates a unitary matrix: ConjugateTranspose[u].u (or u.ConjugateTranspose[u]) is "
         "the identity" + _STRUCT_NOTE)(
    lambda x, *rest: (split_options(rest, {"TargetStructure"}, "UnitaryMatrix"),
                      _orth_check(x, "UnitaryMatrix", True))[1])


def _objarray(x, fname):
    np = _np()
    d = _dims(x)
    if not is_list(x):
        raise Unevaluated()
    if 0 in d:
        raise ValueError(f"{fname}: empty arrays are not supported")

    def conv(e, level):
        if level == len(d):
            if is_list(e):
                raise ValueError(f"{fname}: {x} is not a full array")
            return S(e)
        return [conv(a, level + 1) for a in e.args]
    a = np.empty(d, dtype=object)
    flat = conv(x, 0)
    for idx in itertools.product(*(range(n) for n in d)):
        v = flat
        for i in idx:
            v = v[i]
        a[idx] = v
    return a


@fn("ArrayDot", **ADDED,
    notes="ArrayDot[a, b, k] contracts the last k indices of a with the first k of b; ArrayDot[a, b, {{s1, t1}, "
          "...}] contracts index si of a with index ti of b (remaining indices of a, then of b); exact, symbolic and "
          "numeric arrays")
def _arraydot(a, b, spec):
    np = _np()
    A, B = _objarray(a, "ArrayDot"), _objarray(b, "ArrayDot")
    if K.is_int(spec):
        k = nonneg_int(spec, "ArrayDot", "number of indices")
        if k > A.ndim or k > B.ndim:
            raise ValueError("ArrayDot: cannot contract more indices than the arrays have")
        sa, tb = list(range(A.ndim - k, A.ndim)), list(range(k))
    elif is_list(spec) and all(is_list(p) and len(p.args) == 2 for p in spec.args):
        sa = [pos_int(p.args[0], "ArrayDot", "index") - 1 for p in spec.args]
        tb = [pos_int(p.args[1], "ArrayDot", "index") - 1 for p in spec.args]
        if len(set(sa)) != len(sa) or len(set(tb)) != len(tb) or max(sa + [-1]) >= A.ndim or max(tb + [-1]) >= B.ndim:
            raise ValueError("ArrayDot: invalid index pairs")
    else:
        raise ValueError("ArrayDot: the third argument must be k or {{s1, t1}, ...}")
    for s, t in zip(sa, tb):
        if A.shape[s] != B.shape[t]:
            raise ValueError(f"ArrayDot: index {s + 1} of the first array (length {A.shape[s]}) and index {t + 1} of "
                             f"the second (length {B.shape[t]}) have different lengths")
    C = np.tensordot(A, B, axes=(sa, tb)) if sa else np.multiply.outer(A, B)
    if not isinstance(C, np.ndarray) or C.ndim == 0:
        v = C.item() if isinstance(C, np.ndarray) else C
        return R(sp.expand(v) if isinstance(v, sp.Basic) and v.free_symbols else v)

    def out(c):
        if not isinstance(c, np.ndarray):
            return R(c)
        if c.ndim == 0:
            return R(c.item())
        return L(*(out(c[i]) for i in range(c.shape[0])))
    return out(C)
