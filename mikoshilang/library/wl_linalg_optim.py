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


# ═══════════════════════════════ further matrix decompositions ═══════════════════════════════

_DENSE_NOTE = ("; TargetStructure is accepted, but MikoshiLang has no structured arrays, so the factors are always "
               "ordinary (dense) matrices")


def _decomp(m, rest, fname, square=True, extra=()):
    """Matrix and options for a decomposition: TargetStructure is accepted (always dense)."""
    args, opts = split_options(rest, {"TargetStructure", *extra}, fname)
    _target_structure(opts, fname)
    M = _matrix(m, fname, square=square)
    return M, args, opts


def _kind(M):
    """'exact', 'machine' or 'bignum' for a SymPy matrix."""
    floats = [a for v in M for a in sp.preorder_traversal(v) if isinstance(a, sp.Float)]
    if not floats:
        return "exact"
    if all(f._prec <= 53 for f in floats):
        return "machine"
    return "bignum"


def _numeric_kind(M, fname):
    k = _kind(M)
    if k == "bignum":
        raise NotImplementedError(f"{fname}: arbitrary-precision matrices are not supported; use exact or machine "
                                  "numbers")
    return k


def _simp(v):
    if not isinstance(v, sp.Basic) or v.is_Number:
        return v
    if v.free_symbols:
        return sp.factor(sp.cancel(v)) if v.is_rational_function() else sp.simplify(v)
    return sp.simplify(v)


def _msimp(M):
    return M.applyfunc(_simp)


def _ms(M):
    """Output a SymPy matrix with simplified entries."""
    return _mout(_msimp(M))


def _iszero(v):
    if isinstance(v, sp.Float):
        return v == 0
    if not isinstance(v, sp.Basic):
        return v == 0
    if v.is_number:
        return sp.simplify(v) == 0
    return sp.cancel(sp.together(v)) == 0


def _num_abs(v):
    return float(abs(complex(sp.N(v, 30))))


def _no_symbols(M, fname):
    if M.free_symbols:
        raise NotImplementedError(f"{fname}: symbolic matrices are not supported")


def _hermitian_kind(M, fname):
    """'hermitian' if M == ConjugateTranspose[M], else 'symmetric' if M == Transpose[M]."""
    n = M.rows
    if all(_iszero(M[i, j] - sp.conjugate(M[j, i])) for i in range(n) for j in range(i, n)):
        return "hermitian"
    if all(_iszero(M[i, j] - M[j, i]) for i in range(n) for j in range(i + 1, n)):
        return "symmetric"
    raise ValueError(f"{fname}: the matrix is not Hermitian or symmetric")


def _np_real_if(a, tol=1e-13):
    np = _np()
    a = np.asarray(a)
    if np.iscomplexobj(a) and (a.size == 0 or np.max(np.abs(a.imag)) <= tol * max(1.0, np.max(np.abs(a)))):
        return a.real
    return a


def _mach(X, M):
    """Simplified output; for machine-precision input numeric entries become machine reals."""
    X = _msimp(X)
    if _kind(M) == "machine":
        X = X.applyfunc(lambda v: sp.Float(v) if isinstance(v, sp.Basic) and v.is_number and v.is_real
                        else (sp.N(v, 15) if isinstance(v, sp.Basic) and v.is_number else v))
    return _mout(X)


def _float_out(a):
    np = _np()
    a = np.asarray(a)
    if a.ndim == 2 and a.shape[1] == 0:
        return L(*[L() for _ in range(a.shape[0])])
    return _from_np(a)


@fn("LDLDecomposition", **ADDED,
    notes="{l, d} with m == l.d.ConjugateTranspose[l], l unit lower triangular and d diagonal, for a Hermitian (or "
          "real symmetric) matrix, exact, symbolic or machine; a zero pivot (no LDL factorisation without pivoting) "
          "raises (use BunchKaufmanDecomposition)" + _DENSE_NOTE)
def _ldl(m, *rest):
    M, args, _ = _decomp(m, rest, "LDLDecomposition")
    if args:
        raise ValueError("LDLDecomposition called with too many arguments")
    _numeric_kind(M, "LDLDecomposition")
    if _hermitian_kind(M, "LDLDecomposition") != "hermitian":
        raise ValueError("LDLDecomposition: the matrix is not Hermitian")
    n = M.rows
    Lm, d = sp.eye(n), [sp.Integer(0)] * n
    for j in range(n):
        dj = _simp(M[j, j] - sum(Lm[j, k] * sp.conjugate(Lm[j, k]) * d[k] for k in range(j)))
        d[j] = dj
        for i in range(j + 1, n):
            num = _simp(M[i, j] - sum(Lm[i, k] * sp.conjugate(Lm[j, k]) * d[k] for k in range(j)))
            if _iszero(dj):
                if not _iszero(num):
                    raise ValueError("LDLDecomposition: zero pivot encountered; the matrix has no LDL "
                                     "factorisation without pivoting (use BunchKaufmanDecomposition)")
                Lm[i, j] = 0
            else:
                Lm[i, j] = _simp(num / dj)
    return L(_mach(Lm, M), _mach(sp.diag(*d), M))


_BK_ALPHA = (1 + math.sqrt(17)) / 8


@fn("BunchKaufmanDecomposition", **ADDED,
    notes="{l, b, p} with p.m.Transpose[p] == l.b.ConjugateTranspose[l] (Hermitian m) or l.b.Transpose[l] "
          "(complex symmetric m): Bunch-Kaufman diagonal pivoting (alpha = (1 + Sqrt[17])/8), l unit lower "
          "triangular, b block diagonal with 1x1 and 2x2 blocks, p a permutation matrix; exact or machine numeric "
          "matrices (symbolic matrices raise); pivot choices can differ from Wolfram's on ties" + _DENSE_NOTE)
def _bunchkaufman(m, *rest):
    M, args, _ = _decomp(m, rest, "BunchKaufmanDecomposition")
    if args:
        raise ValueError("BunchKaufmanDecomposition called with too many arguments")
    _numeric_kind(M, "BunchKaufmanDecomposition")
    _no_symbols(M, "BunchKaufmanDecomposition")
    herm = _hermitian_kind(M, "BunchKaufmanDecomposition") == "hermitian"
    cj = sp.conjugate if herm else (lambda v: v)
    n = M.rows
    A = M.copy()
    Lm = sp.eye(n)
    B = sp.zeros(n, n)
    perm = list(range(n))

    def swap(i, j):
        if i == j:
            return
        A.row_swap(i, j)
        A.col_swap(i, j)
        for c in range(k):
            Lm[i, c], Lm[j, c] = Lm[j, c], Lm[i, c]
        perm[i], perm[j] = perm[j], perm[i]

    k = 0
    while k < n:
        lam, r = 0.0, k
        for i in range(k + 1, n):
            a = _num_abs(A[i, k])
            if a > lam:
                lam, r = a, i
        akk = _num_abs(A[k, k])
        s = 1
        if lam == 0 or akk >= _BK_ALPHA * lam:
            pass
        else:
            sigma = max(_num_abs(A[r, j]) for j in range(k, n) if j != r)
            if akk * sigma >= _BK_ALPHA * lam * lam:
                pass
            elif _num_abs(A[r, r]) >= _BK_ALPHA * sigma:
                swap(k, r)
            else:
                swap(k + 1, r)
                s = 2
        if s == 1:
            d = A[k, k]
            B[k, k] = d
            if lam != 0 or any(not _iszero(A[i, k]) for i in range(k + 1, n)):
                for i in range(k + 1, n):
                    Lm[i, k] = _simp(A[i, k] / d)
                for i in range(k + 1, n):
                    for j in range(k + 1, n):
                        A[i, j] = _simp(A[i, j] - A[i, k] * cj(A[j, k]) / d)
            k += 1
        else:
            E = A[k:k + 2, k:k + 2]
            Ei = E.inv()
            B[k:k + 2, k:k + 2] = E
            C = A[k + 2:, k:k + 2]
            if C.rows:
                Lb = (C * Ei).applyfunc(_simp)
                Lm[k + 2:, k:k + 2] = Lb
                Ch = C.T.applyfunc(cj)
                A[k + 2:, k + 2:] = (A[k + 2:, k + 2:] - C * Ei * Ch).applyfunc(_simp)
            k += 2
    P = sp.zeros(n, n)
    for i, j in enumerate(perm):
        P[i, j] = 1
    return L(_mach(Lm, M), _mach(B, M), _mout(P))


@fn("RankDecomposition", **ADDED,
    notes="{c, f} with m == c.f and c, f of full rank MatrixRank[m]: exact and symbolic matrices give c = the pivot "
          "columns of m and f = the non-zero rows of RowReduce[m] (as Wolfram); machine matrices give c with "
          "orthonormal columns (SVD-based, the factorisation is not unique); the zero matrix raises" + _DENSE_NOTE)
def _rankdecomposition(m, *rest):
    M, args, _ = _decomp(m, rest, "RankDecomposition", square=False, extra=("Tolerance",))
    if args:
        raise ValueError("RankDecomposition called with too many arguments")
    kind = _numeric_kind(M, "RankDecomposition")
    if kind == "machine":
        np = _np()
        A = _npmat(M)
        U, s, Vh = np.linalg.svd(A, full_matrices=False)
        tol = max(A.shape) * np.finfo(float).eps * (s[0] if s.size else 0)
        r = int(np.sum(s > tol))
        if r == 0:
            raise ValueError("RankDecomposition: the matrix has rank 0")
        C, F = U[:, :r], np.diag(s[:r]) @ Vh[:r, :]
        for j in range(r):          # first non-negligible entry of each column of c positive
            col = C[:, j]
            k = int(np.argmax(np.abs(col) > 1e-12))
            if col[k].real < 0:
                C[:, j], F[j, :] = -C[:, j], -F[j, :]
        return L(_float_out(C), _float_out(F))
    rref, piv = M.rref(iszerofunc=_iszero, simplify=True)
    if not piv:
        raise ValueError("RankDecomposition: the matrix has rank 0")
    C = M[:, list(piv)]
    F = rref[:len(piv), :]
    return L(_ms(C), _ms(F))


def _int_scale(vec):
    """Scale a vector of rationals to coprime integers (keeping the sign of the original)."""
    if not all(isinstance(v, sp.Basic) and v.is_Rational for v in vec):
        return vec
    dens = [v.q for v in vec]
    lcm = reduce(lambda a, b: a * b // math.gcd(a, b), dens, 1)
    ints = [int(v * lcm) for v in vec]
    g = reduce(math.gcd, ints, 0) or 1
    return [sp.Integer(i // g) for i in ints]


@fn("EigenvalueDecomposition", **ADDED,
    notes="{s, d} with m == s.d.Inverse[s]: s has the eigenvectors (Eigenvectors[m]) as columns and d the "
          "eigenvalues on the diagonal, ordered as Eigensystem; rational eigenvectors are scaled to coprime "
          "integers as Wolfram does; a non-diagonalizable matrix raises" + _DENSE_NOTE)
def _eigenvaluedecomposition(m, *rest):
    M, args, _ = _decomp(m, rest, "EigenvalueDecomposition")
    if args:
        raise ValueError("EigenvalueDecomposition called with too many arguments")
    kind = _numeric_kind(M, "EigenvalueDecomposition")
    res = ev(Expr("Eigensystem", m))
    if not (is_list(res) and len(res.args) == 2 and is_list(res.args[1])):
        raise ValueError("EigenvalueDecomposition: eigensystem could not be computed")
    vals = [S(v) for v in res.args[0].args]
    vecs = [[S(x) for x in v.args] for v in res.args[1].args]
    n = M.rows
    if kind == "machine":
        np = _np()
        V = np.array([[complex(x) for x in v] for v in vecs]).T
        if np.linalg.matrix_rank(V) < n or np.linalg.cond(V) > 1e10:
            raise ValueError("EigenvalueDecomposition: the matrix is not diagonalizable")
        return L(_float_out(_np_real_if(V)), _float_out(_np_real_if(np.diag([complex(v) for v in vals]))))
    vecs = [_int_scale(v) for v in vecs]
    Sm = sp.Matrix(vecs).T
    if any(all(_iszero(x) for x in v) for v in vecs) or Sm.rank(iszerofunc=_iszero, simplify=True) < n:
        raise ValueError("EigenvalueDecomposition: the matrix is not diagonalizable")
    return L(_mout(Sm), _mout(sp.diag(*vals)))


# ── Smith normal form over the integers ──

def _smith_int(A):
    """(U, D, V) with U.A.V == D in Smith normal form; A a list of lists of Python ints."""
    m, n = len(A), len(A[0])
    A = [row[:] for row in A]
    U = [[int(i == j) for j in range(m)] for i in range(m)]
    V = [[int(i == j) for j in range(n)] for i in range(n)]

    def rswap(i, j):
        A[i], A[j] = A[j], A[i]
        U[i], U[j] = U[j], U[i]

    def cswap(i, j):
        for row in A:
            row[i], row[j] = row[j], row[i]
        for row in V:
            row[i], row[j] = row[j], row[i]

    def radd(dst, src, q):          # row_dst += q row_src
        A[dst] = [a + q * b for a, b in zip(A[dst], A[src])]
        U[dst] = [a + q * b for a, b in zip(U[dst], U[src])]

    def cadd(dst, src, q):          # col_dst += q col_src
        for row in A:
            row[dst] += q * row[src]
        for row in V:
            row[dst] += q * row[src]

    for t in range(min(m, n)):
        while True:
            best = None
            for i in range(t, m):
                for j in range(t, n):
                    if A[i][j] and (best is None or abs(A[i][j]) < abs(A[best[0]][best[1]])):
                        best = (i, j)
            if best is None:
                break
            rswap(t, best[0])
            cswap(t, best[1])
            p = A[t][t]
            done = True
            for i in range(t + 1, m):
                if A[i][t]:
                    q = A[i][t] // p
                    radd(i, t, -q)
                    if A[i][t]:
                        done = False
            for j in range(t + 1, n):
                if A[t][j]:
                    q = A[t][j] // p
                    cadd(j, t, -q)
                    if A[t][j]:
                        done = False
            if not done:
                continue
            bad = next(((i, j) for i in range(t + 1, m) for j in range(t + 1, n) if A[i][j] % p), None)
            if bad is None:
                break
            radd(t, bad[0], 1)
        if t < m and t < n and A[t][t] < 0:
            A[t] = [-a for a in A[t]]
            U[t] = [-a for a in U[t]]
    return U, A, V


def _smith(m, rest, fname):
    M, args, _ = _decomp(m, rest, fname, square=False)
    if args:
        raise ValueError(f"{fname} called with too many arguments")
    if _kind(M) != "exact":
        raise ValueError(f"{fname}: the matrix must be exact (integers or rationals)")
    if not all(v.is_Rational for v in M):
        if all(v.is_number for v in M):
            raise NotImplementedError(f"{fname}: only integer and rational matrices are supported (Gaussian "
                                      "integers and algebraic entries are not)")
        raise NotImplementedError(f"{fname}: symbolic matrices are not supported (see PolynomialSmithReduce)")
    den = reduce(lambda a, b: a * b // math.gcd(a, b), [int(v.q) for v in M], 1)
    A = [[int(M[i, j] * den) for j in range(M.cols)] for i in range(M.rows)]
    U, D, V = _smith_int(A)
    Dm = sp.Matrix(D) / den
    return sp.Matrix(U), Dm, sp.Matrix(V)


@fn("SmithDecomposition", **ADDED,
    notes="{u, r, v} with u.m.v == r, u and v unimodular and r the Smith normal form (diagonal, non-negative, "
          "each entry dividing the next) for integer matrices; rational matrices are handled by clearing "
          "denominators (as Wolfram); the transformation matrices are not unique and can differ from Wolfram's; "
          "inexact, Gaussian-integer and symbolic matrices raise" + _DENSE_NOTE)
def _smithdecomposition(m, *rest):
    U, D, V = _smith(m, rest, "SmithDecomposition")
    return L(_mout(U), _mout(D), _mout(V))


@fn("SmithReduce", **ADDED,
    notes="the Smith normal form r of an integer or rational matrix (SmithDecomposition[m][[2]], unique)"
          + _DENSE_NOTE)
def _smithreduce(m, *rest):
    return _mout(_smith(m, rest, "SmithReduce")[1])


# ── Frobenius (rational canonical) form ──

def _krylov(A, v):
    """Krylov basis [v, Av, ...] up to the first dependency, and c with A^d v == sum c_i A^i v."""
    basis = [v]
    while True:
        w = (A * basis[-1]).applyfunc(_simp)
        K = sp.Matrix.hstack(*basis)
        if len(basis) == A.rows:
            Kc = sp.Matrix.hstack(K, w)
        else:
            Kc = sp.Matrix.hstack(K, w)
            if Kc.rank(iszerofunc=_iszero, simplify=True) > len(basis):
                basis.append(w)
                continue
        sol, params = K.gauss_jordan_solve(w)
        if params.shape[0]:
            sol = sol.subs({p: 0 for p in params})
        return basis, [_simp(c) for c in sol]


def _vec_candidates(n, seed=20260):
    import random
    rnd = random.Random(seed)
    for i in range(n):
        yield sp.Matrix([int(j == i) for j in range(n)])
    for _ in range(60):
        yield sp.Matrix([rnd.randint(-3, 3) for _ in range(n)])


def _frobenius_blocks(A):
    """List of (coefficients c, basis columns) with invariant factors in increasing (dividing) order."""
    n = A.rows
    if n == 0:
        return []
    # the minimal polynomial degree is the largest local degree; random vectors reach it generically
    cands = list(_vec_candidates(n))
    best = None
    for v in cands:
        b, c = _krylov(A, v)
        if best is None or len(b) > len(best[0]):
            best = (b, c, v)
    # make sure no unit vector has a larger local polynomial: the lcm of local polys has degree <= len(best)
    d = len(best[0])
    if d == n:
        return [(best[1], best[0])]
    At = A.T
    for u in cands:
        bu, cu = _krylov(A, u)
        if len(bu) != d:
            continue
        for w in cands:
            bw, _ = _krylov(At, w)
            if len(bw) != d:
                continue
            Kw, Ku = sp.Matrix.hstack(*bw), sp.Matrix.hstack(*bu)
            G = (Kw.T * Ku).applyfunc(_simp)
            if _iszero(_simp(G.det())):
                continue
            comp = Kw.T.nullspace(iszerofunc=_iszero, simplify=True)
            Bm = sp.Matrix.hstack(*comp)
            _, piv = Bm.T.rref(iszerofunc=_iszero, simplify=True)
            rows = list(piv)
            A2 = (Bm[rows, :].inv() * (A * Bm)[rows, :]).applyfunc(_simp)
            sub = _frobenius_blocks(A2)
            out = [(c, [(Bm * b).applyfunc(_simp) for b in basis]) for c, basis in sub]
            return out + [(cu, bu)]
    raise NotImplementedError("FrobeniusDecomposition: could not find a cyclic decomposition (degenerate case)")


def _frobenius(m, rest, fname):
    M, args, opts = _decomp(m, rest, fname, extra=("Modulus",))
    if args:
        raise ValueError(f"{fname} called with too many arguments")
    mod = opts.get("Modulus")
    if mod is not None and not (isinstance(mod, int) and mod == 0):
        raise NotImplementedError(f"{fname}: the Modulus option is not supported")
    if _kind(M) != "exact":
        raise NotImplementedError(f"{fname}: inexact matrices are not supported (the rational canonical form is "
                                  "not numerically stable); use exact input")
    blocks = _frobenius_blocks(M)
    n = M.rows
    C = sp.zeros(n, n)
    cols = []
    pos = 0
    for c, basis in blocks:
        d = len(basis)
        for i in range(1, d):
            C[pos + i, pos + i - 1] = 1
        for i in range(d):
            C[pos + i, pos + d - 1] = c[i]
        cols.extend(basis)
        pos += d
    Sm = sp.Matrix.hstack(*cols)
    return Sm, C


@fn("FrobeniusDecomposition", **ADDED,
    notes="{s, c} with m == s.c.Inverse[s] and c the Frobenius (rational canonical) form: block diagonal with "
          "companion matrices (ones below the diagonal, coefficients in the last column) of the invariant factors, "
          "each dividing the next; exact numeric and symbolic matrices (generic case) via cyclic Krylov "
          "decomposition; s is not unique and can differ from Wolfram's; Modulus and inexact input raise"
          + _DENSE_NOTE)
def _frobeniusdecomposition(m, *rest):
    Sm, C = _frobenius(m, rest, "FrobeniusDecomposition")
    return L(_ms(Sm), _ms(C))


@fn("FrobeniusReduce", **ADDED,
    notes="the Frobenius (rational canonical) form of a square matrix (unique; FrobeniusDecomposition[m][[2]])"
          + _DENSE_NOTE)
def _frobeniusreduce(m, *rest):
    return _ms(_frobenius(m, rest, "FrobeniusReduce")[1])


@fn("JordanReduce", **ADDED,
    notes="the Jordan normal form j of a square matrix (exact via SymPy, blocks ordered by eigenvalue real part "
          "then imaginary part); machine matrices must be diagonalizable and give the diagonal matrix of "
          "Eigenvalues[m] (decreasing absolute value); Wolfram's block order can differ for exact matrices"
          + _DENSE_NOTE)
def _jordanreduce(m, *rest):
    M, args, _ = _decomp(m, rest, "JordanReduce")
    if args:
        raise ValueError("JordanReduce called with too many arguments")
    kind = _numeric_kind(M, "JordanReduce")
    if kind == "machine":
        np = _np()
        vals, vecs = np.linalg.eig(_npmat(M))
        if np.linalg.cond(vecs) > 1e8:
            raise NotImplementedError("JordanReduce: this machine matrix is not numerically diagonalizable; use "
                                      "exact input")
        ev_ = ev(Expr("Eigenvalues", m))
        return L(*(L(*(ev_.args[i] if i == j else 0 for j in range(M.rows))) for i in range(M.rows)))
    _, J = M.jordan_form()
    n = J.rows
    blocks, i = [], 0
    while i < n:
        j = i
        while j + 1 < n and J[j, j + 1] == 1 and _iszero(J[j + 1, j + 1] - J[i, i]):
            j += 1
        blocks.append((J[i, i], j - i + 1))
        i = j + 1

    def key(b):
        try:
            z = complex(sp.N(b[0], 30))
            return (round(z.real, 12), round(z.imag, 12), -b[1])
        except TypeError:
            return (0, 0, -b[1])
    blocks.sort(key=key)
    out = sp.zeros(n, n)
    pos = 0
    for lam, size in blocks:
        for k in range(size):
            out[pos + k, pos + k] = _simp(lam)
            if k + 1 < size:
                out[pos + k, pos + k + 1] = 1
        pos += size
    return _mout(out)


# ── core-nilpotent decomposition and Drazin inverse ──

def _core_nilpotent(M, fname):
    """(T, C, N) with M == T.BlockDiagonal[C, N].Inverse[T] (C, N SymPy matrices, possibly 0x0)."""
    n = M.rows
    kind = _numeric_kind(M, fname)
    if kind == "machine":
        np = _np()
        A = _npmat(M)
        ranks = [n]
        P = np.eye(n)
        while True:
            P = P @ A
            r = int(np.linalg.matrix_rank(P))
            if r == ranks[-1]:
                break
            ranks.append(r)
        k = len(ranks) - 1
        Ak = np.linalg.matrix_power(A, k)
        r = ranks[-1]
        if r == n or r == 0:
            return "machine", np.eye(n), (A if r == n else np.zeros((0, 0))), (A if r == 0 else np.zeros((0, 0)))
        U, s, Vh = np.linalg.svd(Ak)
        T = np.hstack([U[:, :r], Vh[r:, :].conj().T])
        B = np.linalg.solve(T, A @ T)
        B[np.abs(B) < 1e-12 * max(1.0, float(np.max(np.abs(A))))] = 0
        return "machine", T, B[:r, :r], B[r:, r:]
    ranks = [n]
    P = sp.eye(n)
    while True:
        P = (P * M).applyfunc(_simp)
        r = P.rank(iszerofunc=_iszero, simplify=True)
        if r == ranks[-1]:
            break
        ranks.append(r)
    k = len(ranks) - 1
    r = ranks[-1]
    if r == n:
        return "exact", sp.eye(n), M, sp.zeros(0, 0)
    if r == 0:
        return "exact", sp.eye(n), sp.zeros(0, 0), M
    Ak = (M ** k).applyfunc(_simp)
    _, piv = Ak.rref(iszerofunc=_iszero, simplify=True)
    rng = [Ak[:, j] for j in piv]
    nul = [sp.Matrix(_int_scale(list(v))) for v in reversed(Ak.nullspace(iszerofunc=_iszero, simplify=True))]
    T = sp.Matrix.hstack(*rng, *nul)
    B = (T.inv() * M * T).applyfunc(_simp)
    return "exact", T, B[:r, :r], B[r:, r:]


def _cn_out(kind, X):
    if kind == "machine":
        return _float_out(_np_real_if(X)) if X.size else L()
    return _ms(X) if X.rows else L()


@fn("CoreNilpotentDecomposition", **ADDED,
    notes="{t, c, n} with m == t.BlockDiagonalMatrix[{c, n}].Inverse[t], c invertible and n nilpotent; exact "
          "matrices use t = {pivot columns of m^k, integer null-space basis of m^k} (k the index of m), as "
          "Wolfram; invertible or nilpotent m give t = IdentityMatrix and {} for the empty part; machine matrices "
          "use an SVD basis (t is not unique)" + _DENSE_NOTE)
def _corenilpotent(m, *rest):
    M, args, _ = _decomp(m, rest, "CoreNilpotentDecomposition")
    if args:
        raise ValueError("CoreNilpotentDecomposition called with too many arguments")
    kind, T, C, N = _core_nilpotent(M, "CoreNilpotentDecomposition")
    return L(_cn_out(kind, T), _cn_out(kind, C), _cn_out(kind, N))


@fn("DrazinInverse", **ADDED,
    notes="the Drazin inverse t.BlockDiagonalMatrix[{Inverse[c], 0}].Inverse[t] from the core-nilpotent "
          "decomposition (the inverse for invertible m, zero for nilpotent m); exact, symbolic (generic) or machine "
          "square matrices" + _DENSE_NOTE)
def _drazininverse(m, *rest):
    M, args, _ = _decomp(m, rest, "DrazinInverse")
    if args:
        raise ValueError("DrazinInverse called with too many arguments")
    kind, T, C, N = _core_nilpotent(M, "DrazinInverse")
    n = M.rows
    if kind == "machine":
        np = _np()
        D = np.zeros((n, n), dtype=complex)
        r = C.shape[0]
        if r:
            D[:r, :r] = np.linalg.inv(C)
        return _float_out(_np_real_if(T @ D @ np.linalg.inv(T)))
    r = C.rows
    D = sp.zeros(n, n)
    if r:
        D[:r, :r] = C.inv()
    return _ms(T * D * T.inv())


# ── polar decomposition ──

@fn("PolarDecomposition", **ADDED,
    notes="{u, p} with m == u.p, u with orthonormal columns (or rows) and p = MatrixPower[ConjugateTranspose[m]"
          ".m, 1/2] positive semidefinite, via the SVD for machine matrices; exact and arbitrary-precision matrices "
          "raise (use N[m])" + _DENSE_NOTE)
def _polardecomposition(m, *rest):
    M, args, _ = _decomp(m, rest, "PolarDecomposition", square=False)
    if args:
        raise ValueError("PolarDecomposition called with too many arguments")
    _no_symbols(M, "PolarDecomposition")
    if _kind(M) != "machine":
        raise NotImplementedError("PolarDecomposition: only machine-precision matrices are supported; use N[m]")
    np = _np()
    A = _npmat(M)
    W, s, Vh = np.linalg.svd(A, full_matrices=False)
    U = W @ Vh
    Pm = Vh.conj().T @ np.diag(s) @ Vh
    return L(_float_out(_np_real_if(U)), _float_out(_np_real_if(Pm)))


# ── ordered Schur decomposition ──

_SCHUR_ORDERS = {
    "DecreasingAbs": lambda z: -abs(z), "IncreasingAbs": lambda z: abs(z),
    "DecreasingRe": lambda z: -z.real, "IncreasingRe": lambda z: z.real,
    "DecreasingAbsRe": lambda z: -abs(z.real), "IncreasingAbsRe": lambda z: abs(z.real),
    "DecreasingAbsIm": lambda z: -abs(z.imag), "IncreasingAbsIm": lambda z: abs(z.imag),
}


def _schur_blocks(T, real):
    n = T.shape[0]
    out, i = [], 0
    while i < n:
        if real and i + 1 < n and T[i + 1, i] != 0:
            a, b, c, d = T[i, i], T[i, i + 1], T[i + 1, i], T[i + 1, i + 1]
            tr, det = a + d, a * d - b * c
            disc = complex(tr * tr / 4 - det) ** 0.5
            out.append((i, 2, tr / 2 + disc))
            i += 2
        else:
            out.append((i, 1, complex(T[i, i])))
            i += 1
    return out


@fn("OrderedSchurDecomposition", **ADDED,
    notes="{q, t} with m == q.t.ConjugateTranspose[q], t (block) upper triangular with eigenvalues ordered by "
          "ord (default \"DecreasingAbs\"; also IncreasingAbs, Decreasing/IncreasingRe, ...AbsRe, ...AbsIm); "
          "RealBlockDiagonalForm -> True (default) keeps 2x2 real blocks for complex pairs of a real matrix; "
          "{m, a} gives the generalized decomposition {q, s, p, t} with m == q.s.ConjugateTranspose[p], a == "
          "q.t.ConjugateTranspose[p] ordered by the generalized eigenvalues; LAPACK reordering (trexc/tgexc) of "
          "SciPy's Schur/QZ forms; numeric matrices only (exact input is evaluated numerically)" + _DENSE_NOTE)
def _orderedschur(m, *rest):
    fname = "OrderedSchurDecomposition"
    args, opts = split_options(rest, {"TargetStructure", "RealBlockDiagonalForm"}, fname)
    _target_structure(opts, fname)
    rbd = opts.get("RealBlockDiagonalForm", True)
    if rbd not in (True, False):
        raise ValueError(f"{fname}: RealBlockDiagonalForm must be True or False")
    if len(args) > 1:
        raise ValueError(f"{fname} called with too many arguments")
    order = args[0] if args else "DecreasingAbs"
    if order not in _SCHUR_ORDERS:
        raise ValueError(f"{fname}: unknown ordering {order}; use one of {', '.join(_SCHUR_ORDERS)}")
    key = _SCHUR_ORDERS[order]
    _need_scipy(fname)
    import scipy.linalg as sla
    from scipy.linalg import lapack
    np = _np()
    gen = is_list(m) and len(m.args) == 2 and len(_dims(m)) == 3
    if gen:
        Ms = [_matrix(x, fname, square=True) for x in m.args]
        for X in Ms:
            _no_symbols(X, fname)
            _numeric_kind(X, fname)
        A, Bm = _npmat(Ms[0]), _npmat(Ms[1])
        if A.shape != Bm.shape:
            raise ValueError(f"{fname}: the matrices must have the same dimensions")
        real = not (np.iscomplexobj(A) or np.iscomplexobj(Bm)) and rbd
        out = "real" if real else "complex"
        if not real:
            A, Bm = A.astype(complex), Bm.astype(complex)
        AA, BB, Q, Z = sla.qz(A, Bm, output=out)
        n = A.shape[0]
        pos = 0
        while pos < n:
            blocks = []
            i = 0
            while i < n:
                size = 2 if real and i + 1 < n and AA[i + 1, i] != 0 else 1
                if size == 2:
                    w = sla.eigvals(AA[i:i + 2, i:i + 2], BB[i:i + 2, i:i + 2])[0]
                else:
                    w = AA[i, i] / BB[i, i] if BB[i, i] != 0 else complex("inf")
                blocks.append((i, size, complex(w)))
                i += size
            rem = [b for b in blocks if b[0] >= pos]
            best = min(rem, key=lambda b: (key(b[2]) if np.isfinite(b[2]) else float("inf"), b[0]))
            if best[0] != pos:
                fun = lapack.dtgexc if real else lapack.ztgexc
                res = fun(AA, BB, Q, Z, best[0] + 1, pos + 1)
                AA, BB, Q, Z, info = res[0], res[1], res[2], res[3], res[-1]
                if info != 0:
                    raise ValueError(f"{fname}: LAPACK reordering failed (info={info})")
            pos += best[1]
        return L(_float_out(Q), _float_out(AA), _float_out(Z), _float_out(BB))
    M = _matrix(m, fname, square=True)
    _no_symbols(M, fname)
    _numeric_kind(M, fname)
    A = _npmat(M)
    real = not np.iscomplexobj(A) and rbd
    T, Q = sla.schur(A if real else A.astype(complex), output="real" if real else "complex")
    n = A.shape[0]
    pos = 0
    while pos < n:
        rem = [b for b in _schur_blocks(T, real) if b[0] >= pos]
        best = min(rem, key=lambda b: (key(b[2]), b[0]))
        if best[0] != pos:
            fun = lapack.dtrexc if real else lapack.ztrexc
            T, Q, info = fun(T, Q, best[0] + 1, pos + 1)
            if info != 0:
                raise ValueError(f"{fname}: LAPACK reordering failed (info={info})")
        pos += best[1]
    if real:
        T = np.triu(T, -1)
    else:
        T = np.triu(T)
    return L(_float_out(Q), _float_out(T))


record("RealBlockDiagonalForm", AREA, "added",
       "option of OrderedSchurDecomposition (and Wolfram's SchurDecomposition): True keeps real 2x2 blocks for "
       "complex-conjugate eigenvalue pairs of a real matrix, False gives a complex upper triangular form")


# ── Karhunen-Loeve decomposition and principal components ──

def _flat_numeric(x, fname):
    if is_list(x):
        out = []
        for a in x.args:
            out.extend(_flat_numeric(a, fname))
        return out
    v = S(x)
    if not (isinstance(v, sp.Basic) and v.is_number):
        raise ValueError(f"{fname}: {x} is not a number")
    return [complex(sp.N(v))]


def _sign_fix(vecs):
    """Make the largest-magnitude component of each eigenvector (rows) positive."""
    np = _np()
    out = vecs.copy()
    for i in range(out.shape[0]):
        j = int(np.argmax(np.abs(out[i])))
        if out[i, j].real < 0:
            out[i] = -out[i]
    return out


def _reshape_like(flat, x):
    if not is_list(x):
        return R(flat.pop(0))
    return L(*(_reshape_like(flat, a) for a in x.args))


@fn("KarhunenLoeveDecomposition", **ADDED,
    notes="KarhunenLoeveDecomposition[{a1, ..., an}] gives {{b1, ..., bn}, m}: the rows of m are the unit "
          "eigenvectors of the covariance matrix of the (flattened) arrays ai in order of decreasing eigenvalue "
          "(sign: largest component positive) and {b1, ...} == m.{a1, ...}; Standardized -> True first shifts each "
          "ai to zero mean; KarhunenLoeveDecomposition[{b1, ...}, m] inverts the transform (missing b's are 0); "
          "machine-precision results")
def _kld(arrays, *rest):
    fname = "KarhunenLoeveDecomposition"
    args, opts = split_options(rest, {"Standardized"}, fname)
    std = opts.get("Standardized", False)
    if std not in (True, False):
        raise ValueError(f"{fname}: Standardized must be True or False")
    items = _items(arrays, fname)
    if not items:
        raise ValueError(f"{fname}: the list of arrays is empty")
    shapes = {_dims(a) for a in items}
    if len(shapes) != 1:
        raise ValueError(f"{fname}: the arrays must all have the same dimensions")
    np = _np()
    data = np.array([_flat_numeric(a, fname) for a in items])
    data = _np_real_if(data)
    if len(args) > 1:
        raise ValueError(f"{fname} called with too many arguments")
    if args:
        Mt = _matrix(args[0], fname, square=True)
        Mn = _npmat(Mt)
        k = Mn.shape[0]
        if len(items) > k:
            raise ValueError(f"{fname}: more arrays than the transformation matrix has rows")
        full = np.zeros((k, data.shape[1]), dtype=data.dtype)
        full[:len(items)] = data
        res = np.linalg.solve(Mn, full)
        res = _np_real_if(res)
        flat = [list(r) for r in res]
        return L(L(*(_reshape_like([float(v) if not np.iscomplexobj(res) else complex(v) for v in f], items[0])
                     for f in flat)), args[0])
    if std:
        data = data - data.mean(axis=1, keepdims=True)
    if data.shape[1] < 2:
        raise ValueError(f"{fname}: the arrays need at least two elements")
    cov = np.cov(data)
    cov = np.atleast_2d(cov)
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(-w, kind="stable")
    Mn = _sign_fix(V[:, idx].T)
    Mn = _np_real_if(Mn)
    B = Mn @ data
    out = [_reshape_like([float(v) if not np.iscomplexobj(B) else complex(v) for v in row], items[0]) for row in B]
    return L(L(*out), _float_out(Mn))


record("Standardized", AREA, "added",
       "option of KarhunenLoeveDecomposition: True shifts each array to zero mean before the transform")


@fn("PrincipalComponents", **ADDED,
    notes="PrincipalComponents[matrix] transforms the rows (observations; columns are variables) to unscaled "
          "principal components: the centred data times the unit eigenvectors of the covariance matrix in order "
          "of decreasing variance; Method -> \"Correlation\" standardises the columns first; machine numeric "
          "data (exact numbers are evaluated numerically); the sign of each component is normalised so that the "
          "largest eigenvector component is positive, which can differ from Wolfram's sign choice; symbolic and "
          "arbitrary-precision data raise")
def _principalcomponents(m, *rest):
    fname = "PrincipalComponents"
    args, opts = split_options(rest, {"Method"}, fname)
    if args:
        raise ValueError(f"{fname} called with too many arguments")
    method = opts.get("Method", "Covariance")
    if is_sym(method, "Automatic"):
        method = "Covariance"
    if method not in ("Covariance", "Correlation"):
        raise ValueError(f"{fname}: Method must be \"Covariance\" or \"Correlation\"")
    M = _matrix(m, fname)
    _no_symbols(M, fname)
    if _kind(M) == "bignum":
        raise NotImplementedError(f"{fname}: arbitrary-precision data are not supported")
    if _kind(M) == "exact" and not all(v.is_Rational for v in M):
        raise NotImplementedError(f"{fname}: exact irrational data are not supported; use N[...]")
    if _kind(M) == "exact":
        raise NotImplementedError(f"{fname}: exact data give exact (radical) components in Wolfram, which are not "
                                  "supported; use N[...] for machine-precision components")
    np = _np()
    X = _npmat(M)
    if X.shape[0] < 2:
        raise ValueError(f"{fname}: at least two observations are needed")
    Xc = X - X.mean(axis=0)
    if method == "Correlation":
        sd = Xc.std(axis=0, ddof=1)
        if np.any(sd == 0):
            raise ValueError(f"{fname}: a column has zero variance")
        Xc = Xc / sd
    cov = np.atleast_2d(np.cov(Xc, rowvar=False))
    w, V = np.linalg.eigh(cov)
    idx = np.argsort(-w, kind="stable")
    Vs = _sign_fix(V[:, idx].T).T
    return _float_out(_np_real_if(Xc @ Vs))


# ── Pfaffian ──

def _pfaffian_expand(A):
    n = A.rows
    if n == 0:
        return sp.Integer(1)
    total = sp.Integer(0)
    for j in range(1, n):
        if _iszero(A[0, j]):
            continue
        keep = [i for i in range(n) if i not in (0, j)]
        total += (-1) ** (j - 1) * A[0, j] * _pfaffian_expand(A[keep, keep])
    return total


def _pfaffian_elim(A):
    A = A.copy()
    n = A.rows
    pf = sp.Integer(1)
    for k in range(0, n - 1, 2):
        piv = None
        best = -1.0
        for j in range(k + 1, n):
            if not _iszero(A[k, j]):
                a = _num_abs(A[k, j])
                if a > best:
                    piv, best = j, a
        if piv is None:
            return sp.Integer(0)
        if piv != k + 1:
            A.row_swap(k + 1, piv)
            A.col_swap(k + 1, piv)
            pf = -pf
        akk = A[k, k + 1]
        pf = pf * akk
        if k + 2 < n:
            u = A[k + 2:, k]          # column k below (entries -A[k, i])
            v = A[k + 2:, k + 1]
            # A <- A - (v u^T - u v^T)/akk on the trailing block (skew-symmetric update)
            upd = (v * u.T - u * v.T) / akk
            A[k + 2:, k + 2:] = (A[k + 2:, k + 2:] + upd).applyfunc(lambda e: e if _kind_float(e) else sp.cancel(e))
    return pf


def _kind_float(e):
    return isinstance(e, sp.Float) or (isinstance(e, sp.Basic) and e.has(sp.Float))


@fn("PfaffianDet", **ADDED,
    notes="the Pfaffian of an antisymmetric matrix (Pf[m]^2 == Det[m], 0 for odd dimension): exact and machine "
          "numeric matrices by skew-symmetric (Parlett-Reid style) elimination with pivoting, symbolic matrices by "
          "expansion along the first row; Method -> Automatic, \"ParlettReid\", \"Householder\", \"Hessenberg\" "
          "and \"Pauli\" give the same value, \"Det\" gives Sqrt[Det[m]] (sign not conformed, as Wolfram); a "
          "non-antisymmetric matrix raises")
def _pfaffiandet(m, *rest):
    fname = "PfaffianDet"
    args, opts = split_options(rest, {"Method"}, fname)
    if args:
        raise ValueError(f"{fname} called with too many arguments")
    method = opts.get("Method", "Automatic")
    if is_sym(method, "Automatic"):
        method = "Automatic"
    if method not in ("Automatic", "ParlettReid", "Householder", "Hessenberg", "Pauli", "Det"):
        raise ValueError(f"{fname}: unknown Method {method}")
    M = _matrix(m, fname, square=True)
    n = M.rows
    if not all(_iszero(M[i, j] + M[j, i]) for i in range(n) for j in range(i, n)):
        raise ValueError(f"{fname}: the matrix is not antisymmetric")
    if n % 2:
        return sp.Integer(0) if _kind(M) == "exact" else 0.0
    if method == "Det":
        return R(_simp(sp.sqrt(M.det())))
    if M.free_symbols and n <= 10:
        return R(sp.expand(_pfaffian_expand(M)))
    if M.free_symbols:
        return R(sp.factor(sp.cancel(_pfaffian_elim(M))))
    pf = _pfaffian_elim(M)
    if _kind(M) == "exact":
        return R(_simp(pf))
    z = complex(sp.N(pf))
    return R(sp.Float(z.real) if z.imag == 0 else sp.Float(z.real) + sp.I * sp.Float(z.imag))


# ── LinearSolveFunction and TransformationMatrix ──

def _linearsolve_one(m):
    if not is_list(m):
        raise Unevaluated()
    M = _matrix(m, "LinearSolve")
    if M.rows == M.cols and all(v.is_number for v in M):
        if _kind(M) == "exact":
            if M.det() == 0:
                raise ValueError("LinearSolve: the matrix is singular")
        else:
            np = _np()
            if np.linalg.cond(_npmat(M)) > 1e15:
                raise ValueError("LinearSolve: the matrix is singular to machine precision")
    return Expr("LinearSolveFunction", L(M.rows, M.cols), m)


from .core import REGISTRY as _REGISTRY  # noqa: E402
_REGISTRY["LinearSolve"].impls.append((_linearsolve_one, 1, 1))


@fn("LinearSolveFunction", **ADDED,
    notes="LinearSolve[m] gives LinearSolveFunction[{r, c}, m] (the stored data is the matrix itself, not "
          "Wolfram's factorisation data); f[b] solves m.x == b for a vector or matrix b like LinearSolve[m, b]")
def _linearsolvefunction(*args):
    raise Unevaluated()


def _lsf_call(head, args, ev_):
    if len(head.args) == 2 and len(args) == 1:
        return ev_(Expr("LinearSolve", head.args[1], args[0]))
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["LinearSolveFunction"] = _lsf_call


@fn("TransformationMatrix", **ADDED,
    notes="TransformationMatrix[TransformationFunction[m]] gives the homogeneous matrix m; anything else stays "
          "unevaluated")
def _transformationmatrix(t):
    tf = isinstance(t, Expr) and t.head == "TransformationFunction"
    if tf and len(t.args) == 1 and is_list(t.args[0]):
        return t.args[0]
    if tf:
        raise ValueError("TransformationMatrix: invalid TransformationFunction")
    raise Unevaluated()
