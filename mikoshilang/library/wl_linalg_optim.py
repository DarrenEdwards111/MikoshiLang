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


# ═══════════════════════════════ vector inequalities ═══════════════════════════════

_VEC_OPS = {"VectorLess": ("<", True), "VectorLessEqual": ("<", False),
            "VectorGreater": (">", True), "VectorGreaterEqual": (">", False)}


def _np_array_of(x):
    """NumPy array of a numeric scalar / array, or None when it is not explicitly numeric."""
    np = _np()
    if is_list(x):
        try:
            parts = [_np_array_of(a) for a in x.args]
        except ValueError:
            return None
        if any(p is None for p in parts):
            return None
        try:
            return np.array(parts)
        except ValueError:
            return None
    try:
        v = S(x) if not isinstance(x, (bool, str)) else None
    except Exception:
        return None
    if isinstance(v, sp.Basic) and v.is_number:
        z = complex(sp.N(v))
        if z.imag != 0:
            return None
        return np.array(z.real)
    return None


def _cone_spec(c, fname):
    if isinstance(c, str):
        return c, None
    if is_list(c) and len(c.args) == 2 and isinstance(c.args[0], str):
        return c.args[0], c.args[1]
    raise ValueError(f"{fname}: invalid cone specification {c}")


def _in_cone(d, cone, param, strict):
    """Whether the numeric array d lies in (the interior of, when strict) the cone."""
    np = _np()
    tol = 0.0
    if cone == "NonNegativeCone":
        return bool(np.all(d > tol) if strict else np.all(d >= -tol))
    if cone == "NormCone":
        v = d.ravel()
        r = float(np.linalg.norm(v[:-1]))
        return bool(v[-1] > r if strict else v[-1] >= r)
    if cone == "SemidefiniteCone":
        if d.ndim != 2 or d.shape[0] != d.shape[1]:
            raise ValueError("SemidefiniteCone: a square matrix is required")
        if not np.allclose(d, d.T):
            return False
        w = np.linalg.eigvalsh(d)
        return bool(np.all(w > 0) if strict else np.all(w >= -1e-14 * max(1.0, float(np.max(np.abs(w))))))
    if cone == "ExponentialCone":
        x, y, z = d.ravel()
        if y > 0:
            return bool(y * math.exp(x / y) < z if strict else y * math.exp(x / y) <= z)
        return (not strict) and y == 0 and x <= 0 and z >= 0
    if cone == "DualExponentialCone":
        u, v, w = d.ravel()
        if u < 0:
            return bool(-u * math.exp(v / u) < math.e * w if strict else -u * math.exp(v / u) <= math.e * w)
        return (not strict) and u == 0 and v >= 0 and w >= 0
    if cone == "PowerCone":
        a = float(param if param is not None else 0.5)
        x, y, z = d.ravel()
        if x < 0 or y < 0:
            return False
        lhs = x ** a * y ** (1 - a)
        return bool(lhs > abs(z) if strict else lhs >= abs(z))
    raise NotImplementedError(f"cone {cone} is not supported")


def _vec_ineq(name, args):
    op, strict = _VEC_OPS[name]
    if len(args) not in (1, 2):
        raise ValueError(f"{name} called with {len(args)} arguments; expected 1 or 2")
    if not is_list(args[0]) or len(args[0].args) < 2:
        raise Unevaluated()
    cone, param = ("NonNegativeCone", None)
    if len(args) == 2:
        cone, param = _cone_spec(args[1], name)
        if cone not in ("NonNegativeCone", "NormCone", "SemidefiniteCone", "ExponentialCone",
                        "DualExponentialCone", "PowerCone"):
            raise ValueError(f"{name}: unknown cone {cone}")
    arrs = [_np_array_of(a) for a in args[0].args]
    if any(a is None for a in arrs):
        raise Unevaluated()
    np = _np()
    for a, b in zip(arrs, arrs[1:]):
        if a.ndim and b.ndim and a.shape != b.shape:
            raise ValueError(f"{name}: the arrays {a.tolist()} and {b.tolist()} have incompatible dimensions")
        d = (b - a) if op == "<" else (a - b)
        if cone == "NonNegativeCone":
            d = np.broadcast_to(d, np.broadcast_shapes(a.shape, b.shape))
        elif not d.ndim:
            raise ValueError(f"{name}: the cone {cone} needs vector or matrix arguments")
        if not _in_cone(np.asarray(d, dtype=float), cone, param, strict):
            return False
    return True


for _vn, (_vop, _vstrict) in _VEC_OPS.items():
    def _mk(vn):
        def impl(*args):
            return _vec_ineq(vn, args)
        return impl
    fn(_vn, **ADDED,
       notes=f"{_vn}[{{x, y, ...}}] (infix \\[{_vn}]): {'strict ' if _vstrict else ''}elementwise "
             f"{'<' if _vop == '<' else '>'}{'' if _vstrict else '='} comparison of numeric scalars/arrays, a "
             "scalar is broadcast against an array; "
             f"{_vn}[{{x, y}}, cone] uses the partial order of {{\"NonNegativeCone\", n}}, {{\"NormCone\", n}}, "
             "{\"SemidefiniteCone\", n}, \"ExponentialCone\", \"DualExponentialCone\" or {\"PowerCone\", a}; "
             "symbolic comparisons stay unevaluated (they are understood as constraints by the optimisers)")(_mk(_vn))


# ═══════════════════════════════ optimisation engine ═══════════════════════════════

class _OptError(ValueError):
    pass


class _Prob:
    """A finite-dimensional optimisation problem in flat scalar variables."""

    def __init__(self):
        self.syms = []          # real SymPy symbols
        self.ints = set()       # indices of integer variables
        self.template = []      # output layout: (name, None) scalar or (name, [indices]) vector
        self.starts = {}        # index -> starting value
        self.obj = sp.Integer(0)
        self.cons = []          # (expr, op) meaning expr op 0, op in "<=", "==" (>= normalised to <=)
        self.cones = []         # (list of exprs or matrix, cone, param)
        self.inexact = False
        self.bare = False       # a single bare variable (NArgMin returns a scalar)


def _fresh(name, i=None):
    return sp.Symbol(f"{name}" if i is None else f"{name}${i + 1}", real=True)


def _infer_dim(e, name):
    """Length of the vector variable `name`, from a Dot with an explicit list or matrix."""
    if isinstance(e, Expr):
        if e.head == "Dot":
            for k, a in enumerate(e.args):
                if isinstance(a, Symbol) and a.name == name:
                    nb = e.args[k - 1] if k > 0 else None
                    if nb is not None and is_list(nb):
                        d = _dims(nb)
                        return d[-1]
                    na = e.args[k + 1] if k + 1 < len(e.args) else None
                    if na is not None and is_list(na):
                        return _dims(na)[0]
        for a in e.args:
            r = _infer_dim(a, name)
            if r:
                return r
        if isinstance(e.head, Expr):
            return _infer_dim(e.head, name)
    return None


def _subst(e, mapping):
    """Replace symbols (by name) in a MikoshiLang expression and re-evaluate."""
    if not mapping:
        return e
    rules = L(*(Expr("Rule", Symbol(k) if isinstance(k, str) else k, v) for k, v in mapping.items()))
    return ev(Expr("ReplaceAll", e, rules))


def _parse_vars(vspec, fname, prob, exprs, allow_start=False, domains=None):
    """Fill prob.syms / template from a variable specification; returns the name -> replacement mapping."""
    specs = list(vspec.args) if is_list(vspec) else [vspec]
    if allow_start and is_list(vspec) and len(vspec.args) in (2, 4) and isinstance(vspec.args[0], Symbol) and \
            not isinstance(vspec.args[1], Symbol) and not is_list(vspec.args[1]):
        specs = [vspec]
    prob.bare = not is_list(vspec) and isinstance(vspec, Symbol)
    mapping = {}
    for k, s in enumerate(specs):
        integer = False
        dim = None
        start = lo = hi = None
        if domains is not None and k < len(domains):
            integer = is_sym(domains[k], "Integers")
        if isinstance(s, Expr) and s.head == "Element" and len(s.args) == 2:
            v, dom = s.args
            if is_sym(dom, "Integers"):
                integer = True
            elif is_sym(dom, "Reals"):
                pass
            elif isinstance(dom, Expr) and dom.head == "Vectors":
                dim = int_arg(dom.args[0], fname)
                if len(dom.args) > 1:
                    integer = is_sym(dom.args[1], "Integers")
                    if not integer and not is_sym(dom.args[1], "Reals"):
                        raise NotImplementedError(f"{fname}: vector domain {dom} is not supported")
            else:
                raise NotImplementedError(f"{fname}: variable domain {dom} is not supported")
            if is_list(v):
                for w in v.args:
                    mapping.update(_parse_vars(Expr("Element", w, dom), fname, prob, exprs, False))
                continue
            s = v
        elif is_list(s) and allow_start and len(s.args) in (2, 4) and isinstance(s.args[0], Symbol):
            start = S(s.args[1])
            if len(s.args) == 4:
                lo, hi = S(s.args[2]), S(s.args[3])
            s = s.args[0]
        if not isinstance(s, (Symbol, Expr)) or (isinstance(s, Expr) and s.head == "List"):
            raise ValueError(f"{fname}: {s} is not a valid variable")
        name = str(s) if isinstance(s, Expr) else s.name
        if isinstance(s, Symbol) and dim is None:
            for e in exprs:
                dim = dim or _infer_dim(e, s.name)
        if dim:
            idx = []
            comps = []
            for i in range(dim):
                sym = _fresh(name, i)
                idx.append(len(prob.syms))
                if integer:
                    prob.ints.add(len(prob.syms))
                prob.syms.append(sym)
                comps.append(Symbol(sym.name))
            prob.template.append((s, idx))
            mapping[s] = L(*comps)
        else:
            sym = _fresh(name)
            i = len(prob.syms)
            if integer:
                prob.ints.add(i)
            prob.syms.append(sym)
            prob.template.append((s, i))
            if isinstance(s, Expr) or s.name != sym.name:
                mapping[s] = Symbol(sym.name)
            if start is not None:
                prob.starts[i] = float(sp.N(start))
            if lo is not None:
                prob.cons.append(("bound", sym, lo, hi))
    return mapping


def _to_sym(prob, e, fname):
    v = S(e)
    if isinstance(v, list):
        raise ValueError(f"{fname}: expected a scalar expression, got a list {e}")
    if not isinstance(v, sp.Basic):
        v = sp.sympify(v)
    rep = {sp.Symbol(s.name): s for s in prob.syms}
    v = v.xreplace(rep)
    if any(isinstance(a, sp.Float) for a in sp.preorder_traversal(v)):
        prob.inexact = True
    return v


def _region_cons(pts, region, fname):
    """Inequalities (as MikoshiLang expressions) for {x, ...} \\[Element] region."""
    if not isinstance(region, Expr):
        raise NotImplementedError(f"{fname}: region {region} is not supported")
    h, a = region.head, region.args
    n = len(pts)
    if h in ("Disk", "Ball"):
        c = a[0] if a else L(*([0] * n))
        r = a[1] if len(a) > 1 else 1
        if not is_list(c) or len(c.args) != n or is_list(r):
            raise NotImplementedError(f"{fname}: region {region} is not supported")
        return [Expr("LessEqual", Expr("Plus", *(Expr("Power", Expr("Plus", p, Expr("Times", -1, ci)), 2)
                                                  for p, ci in zip(pts, c.args))), Expr("Power", r, 2))]
    if h in ("Rectangle", "Cuboid"):
        lo = a[0] if a else L(*([0] * n))
        hi = a[1] if len(a) > 1 else Expr("Plus", lo, L(*([1] * n)))
        hi = ev(hi)
        out = []
        for p, l_, u_ in zip(pts, lo.args, hi.args):
            out += [Expr("GreaterEqual", p, l_), Expr("LessEqual", p, u_)]
        return out
    raise NotImplementedError(f"{fname}: region {h} is not supported (Disk, Ball, Rectangle and Cuboid are)")


def _add_rel(prob, lhs, rhs, op, fname):
    a, b = _to_sym(prob, lhs, fname), _to_sym(prob, rhs, fname)
    if op in ("<=", "<"):
        prob.cons.append((a - b, "<="))
    elif op in (">=", ">"):
        prob.cons.append((b - a, "<="))
    else:
        prob.cons.append((a - b, "=="))


_REL = {"LessEqual": "<=", "Less": "<", "GreaterEqual": ">=", "Greater": ">", "Equal": "=="}


def _parse_cons(prob, c, fname, mapping):
    if c is True or c is None:
        return
    if c is False:
        raise _OptError(f"{fname}: the constraints are infeasible")
    if is_list(c) or (isinstance(c, Expr) and c.head == "And"):
        for a in c.args:
            _parse_cons(prob, a, fname, mapping)
        return
    if not isinstance(c, Expr):
        raise ValueError(f"{fname}: invalid constraint {c}")
    h = c.head
    if h == "Element" and len(c.args) == 2:
        v, dom = c.args
        vs = list(v.args) if is_list(v) else [v]
        if is_sym(dom, "Integers") or is_sym(dom, "Reals"):
            for w in vs:
                w = _subst(w, mapping)
                ws = list(w.args) if is_list(w) else [w]
                for x in ws:
                    sym = _to_sym(prob, x, fname)
                    if sym not in prob.syms:
                        raise ValueError(f"{fname}: {x} is not a variable")
                    if is_sym(dom, "Integers"):
                        prob.ints.add(prob.syms.index(sym))
            return
        pts = list(_subst(v, mapping).args) if is_list(_subst(v, mapping)) else [_subst(v, mapping)]
        for r in _region_cons(pts, dom, fname):
            _parse_cons(prob, ev(r), fname, mapping)
        return
    if h in _VEC_OPS:
        op, strict = _VEC_OPS[h]
        items = c.args[0]
        if not is_list(items):
            raise ValueError(f"{fname}: invalid vector inequality {c}")
        cone = "NonNegativeCone"
        param = None
        if len(c.args) == 2:
            cone, param = _cone_spec(c.args[1], fname)
        parts = [ev(p) for p in items.args]
        for a, b in zip(parts, parts[1:]):
            d = ev(Expr("Plus", b, Expr("Times", -1, a))) if op == "<" else ev(Expr("Plus", a, Expr("Times", -1, b)))
            if cone == "NonNegativeCone":
                flat = _flat_expr(d)
                for e in flat:
                    prob.cons.append((-_to_sym(prob, e, fname), "<="))
            else:
                prob.cones.append((d, cone, param))
        return
    if h == "Inequality":
        a = c.args
        for i in range(0, len(a) - 2, 2):
            _add_rel(prob, a[i], a[i + 2], _REL[a[i + 1].name], fname)
        return
    if h in _REL and len(c.args) >= 2:
        args = c.args
        for x, y in zip(args, args[1:]):
            if is_list(x) or is_list(y):
                xs, ys = _flat_expr(x), _flat_expr(y)
                if len(xs) == 1:
                    xs = xs * len(ys)
                if len(ys) == 1:
                    ys = ys * len(xs)
                if len(xs) != len(ys):
                    raise ValueError(f"{fname}: incompatible dimensions in {c}")
                for p, q in zip(xs, ys):
                    _add_rel(prob, p, q, _REL[h], fname)
            else:
                _add_rel(prob, x, y, _REL[h], fname)
        return
    if h == "Unequal":
        raise NotImplementedError(f"{fname}: Unequal constraints are not supported")
    raise ValueError(f"{fname}: invalid constraint {c}")


def _flat_expr(x):
    if is_list(x):
        out = []
        for a in x.args:
            out.extend(_flat_expr(a))
        return out
    return [x]


def _build(f, cons, vspec, fname, allow_start=False, domains=None):
    prob = _Prob()
    exprs = [f, cons]
    mapping = _parse_vars(vspec, fname, prob, exprs, allow_start, domains)
    bounds = [c for c in prob.cons if c and c[0] == "bound"]
    prob.cons = [c for c in prob.cons if not (c and c[0] == "bound")]
    for _, sym, lo, hi in bounds:
        prob.cons.append((lo - sym, "<="))
        prob.cons.append((sym - hi, "<="))
    f2 = _subst(f, mapping)
    if is_list(f2):
        raise ValueError(f"{fname}: the objective must be a scalar, got {f2}")
    prob.obj = _to_sym(prob, f2, fname)
    _parse_cons(prob, _subst(cons, mapping) if cons is not None else None, fname, mapping)
    allowed = set(prob.syms)
    for e in [prob.obj] + [c[0] for c in prob.cons]:
        extra = e.free_symbols - allowed
        if extra:
            raise ValueError(f"{fname}: the problem contains parameters {sorted(map(str, extra))} that are not "
                             "variables")
    return prob


def _split_problem(args, fname):
    """(objective, constraints) from f or {f, cons...}."""
    f = args
    if is_list(f) and f.args:
        return f.args[0], L(*f.args[1:]) if len(f.args) > 1 else None
    return f, None


# ── exact simplex (Fractions) ──

def _frac(v):
    if isinstance(v, Fraction):
        return v
    v = sp.nsimplify(v) if isinstance(v, sp.Float) and False else v
    if isinstance(v, sp.Float):
        return Fraction(float(v))
    if isinstance(v, sp.Rational):
        return Fraction(int(v.p), int(v.q))
    if isinstance(v, (int,)):
        return Fraction(v)
    raise TypeError("not rational")


def _linear_parts(e, syms):
    """(coefficient list, constant) of a linear expression, or None if not linear."""
    e = sp.expand(e)
    try:
        p = sp.Poly(e, *syms)
    except sp.PolynomialError:
        return None
    if p.total_degree() > 1:
        return None
    coeffs = []
    for s in syms:
        coeffs.append(p.coeff_monomial(s))
    const = p.coeff_monomial(1)
    vals = coeffs + [const]
    if not all(isinstance(v, sp.Basic) and v.is_number for v in vals):
        return None
    return coeffs, const


def _simplex(c, A, b, kinds, n):
    """Minimise c.x over free x with rows A_i.x (kind) b_i, kind in '<=', '=='.  Exact Fractions.

    Returns ('optimal', x, value) | ('infeasible',) | ('unbounded',)."""
    # variables: x = p - q (2n), slack per '<=' row, artificial per row
    rows = []
    rhs = []
    m = len(A)
    nslack = sum(1 for k in kinds if k == "<=")
    ncol = 2 * n + nslack
    si = 0
    for i in range(m):
        row = [Fraction(0)] * ncol
        for j in range(n):
            row[j] = A[i][j]
            row[n + j] = -A[i][j]
        if kinds[i] == "<=":
            row[2 * n + si] = Fraction(1)
            si += 1
        bi = b[i]
        if bi < 0:
            row = [-v for v in row]
            bi = -bi
        rows.append(row)
        rhs.append(bi)
    # phase 1 tableau with artificials
    total = ncol + m
    T = [rows[i] + [Fraction(int(i == k)) for k in range(m)] + [rhs[i]] for i in range(m)]
    basis = [ncol + i for i in range(m)]

    def pivot(r, col):
        pv = T[r][col]
        T[r] = [v / pv for v in T[r]]
        for i in range(len(T)):
            if i != r and T[i][col] != 0:
                f = T[i][col]
                T[i] = [a - f * bb for a, bb in zip(T[i], T[r])]
        basis[r] = col

    def run(cost, allowed):
        while True:
            # reduced costs
            cb = [cost[basis[i]] for i in range(m)]
            enter = None
            for j in range(total):
                if not allowed(j) or j in basis:
                    continue
                rc = cost[j] - sum(cb[i] * T[i][j] for i in range(m))
                if rc < 0:
                    enter = j          # Bland: smallest index
                    break
            if enter is None:
                return "optimal"
            best = None
            for i in range(m):
                if T[i][enter] > 0:
                    ratio = T[i][-1] / T[i][enter]
                    if best is None or ratio < best[0] or (ratio == best[0] and basis[i] < basis[best[1]]):
                        best = (ratio, i)
            if best is None:
                return "unbounded"
            pivot(best[1], enter)

    cost1 = [Fraction(0)] * ncol + [Fraction(1)] * m
    run(cost1, lambda j: True)
    if sum(T[i][-1] for i in range(m) if basis[i] >= ncol) != 0:
        return ("infeasible",)
    # drive artificials out of the basis
    for i in range(m):
        if basis[i] >= ncol:
            for j in range(ncol):
                if T[i][j] != 0:
                    pivot(i, j)
                    break
    cost2 = [Fraction(0)] * total
    for j in range(n):
        cost2[j] = c[j]
        cost2[n + j] = -c[j]
    st = run(cost2, lambda j: j < ncol)
    if st == "unbounded":
        return ("unbounded",)
    z = [Fraction(0)] * total
    for i in range(m):
        z[basis[i]] = T[i][-1]
    x = [z[j] - z[n + j] for j in range(n)]
    return ("optimal", x, sum(ci * xi for ci, xi in zip(c, x)))


def _lp_data(prob, fname):
    """Exact (Fraction) LP data from a linear problem, or None if it is not linear."""
    if prob.cones:
        return None
    lin = _linear_parts(prob.obj, prob.syms)
    if lin is None:
        return None
    rows = []
    for e, op in prob.cons:
        lp = _linear_parts(e, prob.syms)
        if lp is None:
            return None
        rows.append((lp, op))
    try:
        c = [_frac(v) for v in lin[0]]
        c0 = _frac(lin[1])
        A = [[_frac(v) for v in lp[0]] for lp, _ in rows]
        b = [-_frac(lp[1]) for lp, _ in rows]
    except TypeError:
        # irrational coefficients: use floats exactly converted
        c = [Fraction(float(v)) for v in lin[0]]
        c0 = Fraction(float(lin[1]))
        A = [[Fraction(float(v)) for v in lp[0]] for lp, _ in rows]
        b = [-Fraction(float(lp[1])) for lp, _ in rows]
        prob.inexact = True
    kinds = [op for _, op in rows]
    return c, c0, A, b, kinds


def _bnb(solve, ints, extra=(), limit=20000):
    """Branch and bound: solve(extra_bounds) -> ('optimal', x, val) | ...; ints indices integer."""
    best = None
    stack = [list(extra)]
    nodes = 0
    status_any = None
    while stack:
        nodes += 1
        if nodes > limit:
            raise NotImplementedError("branch and bound exceeded its node limit")
        bounds = stack.pop()
        res = solve(bounds)
        if res[0] == "unbounded":
            if best is None:
                return res
            continue
        if res[0] != "optimal":
            continue
        status_any = "optimal"
        x, val = res[1], res[2]
        if best is not None and val >= best[2] - (1e-9 if isinstance(val, float) else 0):
            continue
        frac_i = None
        for i in sorted(ints):
            xi = x[i]
            if abs(xi - round(xi)) > (1e-7 if isinstance(xi, float) else 0):
                frac_i = i
                break
        if frac_i is None:
            best = res
            continue
        xi = x[frac_i]
        fl = math.floor(xi)
        stack.append(bounds + [(frac_i, "<=", fl)])
        stack.append(bounds + [(frac_i, ">=", fl + 1)])
    if best is None:
        return ("infeasible",) if status_any is None else ("infeasible",)
    return best


def _solve_lp(prob, fname):
    data = _lp_data(prob, fname)
    if data is None:
        return None
    c, c0, A, b, kinds = data
    n = len(prob.syms)

    def solve(bounds):
        A2, b2, k2 = [r[:] for r in A], b[:], kinds[:]
        for i, op, v in bounds:
            row = [Fraction(0)] * n
            row[i] = Fraction(1) if op == "<=" else Fraction(-1)
            A2.append(row)
            b2.append(Fraction(v) if op == "<=" else Fraction(-v))
            k2.append("<=")
        return _simplex(c, A2, b2, k2, n)

    res = _bnb(solve, prob.ints) if prob.ints else solve([])
    if res[0] == "infeasible":
        raise _OptError(f"{fname}: the constraints are infeasible")
    if res[0] == "unbounded":
        raise _OptError(f"{fname}: the problem is unbounded")
    return res[1], res[2] + c0


def _result_values(prob, x, exact):
    """MikoshiLang values for the flat solution x (Fractions or floats)."""
    out = []
    for i, v in enumerate(x):
        if i in prob.ints:
            out.append(sp.Integer(int(round(v))))
        elif exact and isinstance(v, Fraction) and not prob.inexact and not prob.ints:
            out.append(sp.Rational(v.numerator, v.denominator))
        else:
            out.append(sp.Float(float(v)))
    return out


def _rules(prob, vals):
    out = []
    for var, idx in prob.template:
        if isinstance(idx, int):
            out.append(Expr("Rule", var, R(vals[idx])))
        else:
            out.append(Expr("Rule", var, L(*(R(vals[i]) for i in idx))))
    return L(*out)


def _vector(prob, vals):
    out = []
    for var, idx in prob.template:
        if isinstance(idx, int):
            out.append(R(vals[idx]))
        else:
            out.append(L(*(R(vals[i]) for i in idx)))
    return L(*out)


# ── exact convex quadratic programming (active-set enumeration) ──

def _frac_solve(M, rhs):
    """One solution of M x == rhs (free variables 0) over Fractions, or None if inconsistent."""
    m = len(M)
    n = len(M[0]) if M else 0
    A = [row[:] + [r] for row, r in zip(M, rhs)]
    piv_cols = []
    r = 0
    for c in range(n):
        p = next((i for i in range(r, m) if A[i][c] != 0), None)
        if p is None:
            continue
        A[r], A[p] = A[p], A[r]
        pv = A[r][c]
        A[r] = [v / pv for v in A[r]]
        for i in range(m):
            if i != r and A[i][c] != 0:
                f = A[i][c]
                A[i] = [a - f * b for a, b in zip(A[i], A[r])]
        piv_cols.append(c)
        r += 1
        if r == m:
            break
    for i in range(r, m):
        if A[i][-1] != 0:
            return None
    x = [Fraction(0)] * n
    for i, c in enumerate(piv_cols):
        x[c] = A[i][-1]
    return x


def _qp_data(prob):
    """(Q, c, c0) Fractions for a quadratic objective, or None."""
    e = sp.expand(prob.obj)
    try:
        p = sp.Poly(e, *prob.syms)
    except sp.PolynomialError:
        return None
    if p.total_degree() > 2:
        return None
    n = len(prob.syms)
    try:
        Q = [[Fraction(0)] * n for _ in range(n)]
        c = [Fraction(0)] * n
        c0 = Fraction(0)
        for mon, coef in p.terms():
            if not coef.is_number:
                return None
            v = _frac(coef) if coef.is_Rational or isinstance(coef, sp.Float) else Fraction(float(coef))
            idx = [i for i, k in enumerate(mon) for _ in range(k)]
            if len(idx) == 0:
                c0 += v
            elif len(idx) == 1:
                c[idx[0]] += v
            else:
                i, j = idx
                if i == j:
                    Q[i][i] += 2 * v
                else:
                    Q[i][j] += v
                    Q[j][i] += v
    except TypeError:
        return None
    return Q, c, c0


def _psd(Q):
    M = sp.Matrix([[sp.Rational(v.numerator, v.denominator) for v in row] for row in Q])
    return bool(M.is_positive_semidefinite)


def _solve_qp_exact(prob, fname, Q, c, c0, A, b, kinds, extra=()):
    n = len(prob.syms)
    ineq = [(A[i], b[i]) for i in range(len(A)) if kinds[i] == "<="]
    eq = [(A[i], b[i]) for i in range(len(A)) if kinds[i] == "=="]
    for i, op, v in extra:
        row = [Fraction(0)] * n
        row[i] = Fraction(1) if op == "<=" else Fraction(-1)
        ineq.append((row, Fraction(v) if op == "<=" else Fraction(-v)))
    m = len(ineq)
    if m > 12:
        return None
    for size in range(0, min(m, n + len(eq) + m) + 1):
        for W in itertools.combinations(range(m), size):
            rows_c = [ineq[i][0] for i in W] + [r for r, _ in eq]
            k = len(rows_c)
            M = []
            for i in range(n):
                M.append(Q[i][:] + [rows_c[j][i] for j in range(k)])
            for j in range(k):
                M.append(rows_c[j][:] + [Fraction(0)] * k)
            rhs = [-v for v in c] + [ineq[i][1] for i in W] + [v for _, v in eq]
            sol = _frac_solve(M, rhs)
            if sol is None:
                continue
            x, lam = sol[:n], sol[n:n + len(W)]
            if any(l_ < 0 for l_ in lam):       # Q x + c + A_W^T lambda == 0 needs lambda >= 0
                continue
            if all(sum(a * xi for a, xi in zip(r, x)) <= bb for r, bb in ineq):
                val = sum(Q[i][j] * x[i] * x[j] for i in range(n) for j in range(n)) / 2 + \
                    sum(ci * xi for ci, xi in zip(c, x)) + c0
                return ("optimal", x, val)
    return ("infeasible",)


def _solve_qp(prob, fname):
    if prob.cones:
        return None
    qd = _qp_data(prob)
    if qd is None:
        return None
    Q, c, c0 = qd
    lin = []
    for e, op in prob.cons:
        lp = _linear_parts(e, prob.syms)
        if lp is None:
            return None
        lin.append((lp, op))
    if not _psd(Q):
        raise _OptError(f"{fname}: the objective is not convex (its Hessian is not positive semidefinite)")
    try:
        A = [[_frac(v) for v in lp[0]] for lp, _ in lin]
        b = [-_frac(lp[1]) for lp, _ in lin]
    except TypeError:
        A = [[Fraction(float(v)) for v in lp[0]] for lp, _ in lin]
        b = [-Fraction(float(lp[1])) for lp, _ in lin]
    kinds = [op for _, op in lin]
    if sum(1 for k in kinds if k == "<=") + 2 * len(prob.ints) > 12:
        return None

    def solve(bounds):
        r = _solve_qp_exact(prob, fname, Q, c, c0, A, b, kinds, bounds)
        if r is None:
            raise NotImplementedError("too many constraints")
        return r
    try:
        res = _bnb(solve, prob.ints) if prob.ints else solve([])
    except NotImplementedError:
        return None
    if res[0] == "infeasible":
        # an unbounded convex QP has no KKT point either; distinguish with the LP of the constraints
        raise _OptError(f"{fname}: the problem is infeasible or unbounded")
    return res[1], res[2]


# ── general nonlinear problems (SciPy SLSQP, multi-start) ──

def _lam(e, syms):
    np = _np()
    f = sp.lambdify(syms, e, modules=["numpy", {"Abs": np.abs}])
    return lambda x: f(*x)


def _cone_funcs(prob, cone, d, param, fname):
    """Inequality functions (>= 0) for a cone constraint d \\[Element] cone."""
    np = _np()
    syms = prob.syms
    if is_list(d) and is_list(d.args[0]) if is_list(d) and d.args else False:
        mat = [[_to_sym(prob, v, fname) for v in r.args] for r in d.args]
        fs = [[_lam(v, syms) for v in r] for r in mat]
        if cone != "SemidefiniteCone":
            raise NotImplementedError(f"{fname}: matrix arguments need the SemidefiniteCone")

        def sdp(x):
            Mx = np.array([[float(f(x)) for f in r] for r in fs])
            Mx = (Mx + Mx.T) / 2
            return float(np.linalg.eigvalsh(Mx)[0])
        return [sdp]
    vec = [_to_sym(prob, v, fname) for v in _flat_expr(d)]
    fs = [_lam(v, syms) for v in vec]
    if cone == "NormCone":
        return [lambda x: float(fs[-1](x)) - float(np.sqrt(sum(float(f(x)) ** 2 for f in fs[:-1])))]
    if cone == "ExponentialCone":
        def ec(x):
            a, b_, c_ = (float(f(x)) for f in fs)
            return c_ - b_ * math.exp(min(a / b_, 700)) if b_ > 1e-12 else -abs(b_) - max(a, 0)
        return [ec, lambda x: float(fs[1](x))]
    if cone == "PowerCone":
        a = float(sp.N(S(param))) if param is not None else 0.5

        def pc(x):
            u, v, w = (float(f(x)) for f in fs)
            return max(u, 0) ** a * max(v, 0) ** (1 - a) - abs(w)
        return [pc, lambda x: float(fs[0](x)), lambda x: float(fs[1](x))]
    if cone == "NonNegativeCone":
        return [(lambda f: (lambda x: float(f(x))))(f) for f in fs]
    raise NotImplementedError(f"{fname}: the cone {cone} is not supported")


def _nlp(prob, fname, local, extra_bounds=()):
    _need_scipy(fname)
    from scipy.optimize import minimize
    np = _np()
    syms = prob.syms
    n = len(syms)
    fobj = _lam(prob.obj, syms)
    try:
        grad = [_lam(sp.diff(prob.obj, s), syms) for s in syms]
        jac = lambda x: np.array([float(g(x)) for g in grad])  # noqa: E731
    except Exception:
        jac = None
    cons = []
    viol = []
    for e, op in prob.cons:
        fe = _lam(e, syms)
        try:
            ge = [_lam(sp.diff(e, s), syms) for s in syms]
            je = (lambda ge_: (lambda x: -np.array([float(g(x)) for g in ge_])))(ge)
        except Exception:
            je = None
        if op == "<=":
            fun = (lambda f: (lambda x: -float(f(x))))(fe)
            cons.append({"type": "ineq", "fun": fun, **({"jac": je} if je else {})})
            viol.append((lambda f: (lambda x: max(0.0, float(f(x)))))(fe))
        else:
            fun = (lambda f: (lambda x: float(f(x))))(fe)
            cons.append({"type": "eq", "fun": fun, **({"jac": (lambda j: (lambda x: -j(x)))(je)} if je else {})})
            viol.append((lambda f: (lambda x: abs(float(f(x)))))(fe))
    for d, cone, param in prob.cones:
        for g in _cone_funcs(prob, cone, d, param, fname):
            cons.append({"type": "ineq", "fun": g})
            viol.append((lambda g_: (lambda x: max(0.0, -g_(x))))(g))
    for i, op, v in extra_bounds:
        g = (lambda i_, v_: (lambda x: v_ - x[i_]))(i, v) if op == "<=" else (lambda i_, v_: (lambda x: x[i_] - v_))(i, v)
        cons.append({"type": "ineq", "fun": g})
        viol.append((lambda g_: (lambda x: max(0.0, -g_(x))))(g))
    rng = np.random.default_rng(1234)
    base = np.array([prob.starts.get(i, 1.0 if local else 0.0) for i in range(n)], dtype=float)
    starts = [base]
    if not local:
        starts += [np.ones(n), -np.ones(n), np.full(n, 0.5)]
        starts += [rng.normal(0, 3, n) for _ in range(16)]
    best = None
    for x0 in starts:
        try:
            with np.errstate(all="ignore"):
                r = minimize(fobj, x0, jac=jac, method="SLSQP", constraints=cons,
                             options={"maxiter": 1000, "ftol": 1e-12})
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
            continue
        x = r.x
        try:
            v = float(fobj(x))
        except Exception:
            continue
        if not np.isfinite(v):
            continue
        mv = max([f(x) for f in viol], default=0.0)
        if mv > 1e-6 * max(1.0, float(np.max(np.abs(x))) if n else 1.0):
            continue
        if np.max(np.abs(x)) > 1e10 or v < -1e14:
            raise _OptError(f"{fname}: the problem appears to be unbounded")
        if best is None or v < best[1] - 1e-12:
            best = (x, v)
    if best is None:
        return ("infeasible",)
    return ("optimal", [float(t) for t in best[0]], best[1])


def _solve_nlp(prob, fname, local):
    if prob.ints:
        res = _bnb(lambda bounds: _nlp(prob, fname, local, bounds), prob.ints, limit=400)
    else:
        res = _nlp(prob, fname, local)
    if res[0] != "optimal":
        raise _OptError(f"{fname}: no feasible point was found (the constraints may be infeasible)")
    return res[1], res[2]


def _solve(prob, fname, local=False, convex=False):
    """(x list, value, exact?) minimising prob.obj."""
    r = _solve_lp(prob, fname)
    if r is not None:
        return r[0], r[1], True
    r = _solve_qp(prob, fname)
    if r is not None:
        return r[0], r[1], False
    if convex:
        _check_convex(prob, fname)
    x, v = _solve_nlp(prob, fname, local)
    return x, v, False


def _check_convex(prob, fname):
    """Reject problems whose objective or inequality constraints are visibly non-convex (sampled Hessians)."""
    np = _np()
    syms = prob.syms
    rng = np.random.default_rng(7)
    pts = [rng.normal(0, 2, len(syms)) for _ in range(12)]

    def convex_at(e):
        try:
            H = sp.hessian(e, syms)
        except Exception:
            return True
        if all(_iszero(h) for h in H):
            return True
        Hf = sp.lambdify(syms, H, modules=["numpy"])
        for p in pts:
            try:
                with np.errstate(all="ignore"):
                    Hn = np.array(Hf(*p), dtype=float)
            except Exception:
                continue
            if not np.all(np.isfinite(Hn)):
                continue
            if np.linalg.eigvalsh((Hn + Hn.T) / 2)[0] < -1e-8 * max(1.0, float(np.max(np.abs(Hn)))):
                return False
        return True
    if not convex_at(prob.obj):
        raise _OptError(f"{fname}: the objective is not convex")
    for e, op in prob.cons:
        if op == "==" and _linear_parts(e, syms) is None:
            raise _OptError(f"{fname}: equality constraints must be affine for a convex problem")
        if op == "<=" and not convex_at(e):
            raise _OptError(f"{fname}: the constraint {e} <= 0 is not convex")


_OPT_OPTS = {"Method", "WorkingPrecision", "AccuracyGoal", "PrecisionGoal", "MaxIterations", "StepMonitor",
             "EvaluationMonitor", "Tolerance", "PerformanceGoal", "RandomSeed"}


def _opt_options(opts, fname):
    wp = opts.get("WorkingPrecision")
    if wp is not None and not is_sym(wp, "MachinePrecision") and not (S(wp).is_number and float(S(wp)) <= 16):
        raise NotImplementedError(f"{fname}: WorkingPrecision above machine precision is not supported")
    for k in ("StepMonitor", "EvaluationMonitor"):
        if k in opts:
            raise NotImplementedError(f"{fname}: the option {k} is not supported")


def _argopt(fname, sense, what, local):
    def impl(f, *rest):
        args, opts = split_options(rest, _OPT_OPTS, fname)
        _opt_options(opts, fname)
        if len(args) == 0 and not local:
            raise ValueError(f"{fname}: the variables must be given")
        obj, cons = _split_problem(f, fname)
        if len(args) == 2 and not local:
            # NArgMin[f, cons, vars] is not a Wolfram form; NArgMin[f, x \[Element] reg] is
            raise ValueError(f"{fname} called with too many arguments")
        if len(args) != 1:
            raise ValueError(f"{fname}: expected the variables as the second argument")
        vspec = args[0]
        if not local and isinstance(vspec, Expr) and vspec.head == "Element" and not \
                (is_sym(vspec.args[1], "Integers") or is_sym(vspec.args[1], "Reals") or
                 (isinstance(vspec.args[1], Expr) and vspec.args[1].head == "Vectors")):
            cons = L(*(cons.args if cons is not None else ()), vspec)
            vspec = vspec.args[0]
        prob = _build(obj, cons, vspec, fname, allow_start=local)
        if sense == "max":
            prob.obj = -prob.obj
        x, v, exact = _solve(prob, fname, local=local)
        if sense == "max":
            v = -v
        if what == "value":
            return R(sp.Float(float(v)))
        vals = [sp.Integer(int(round(t))) if i in prob.ints else sp.Float(float(t)) for i, t in enumerate(x)]
        out = _vector(prob, vals)
        if not local and prob.bare and len(out.args) == 1:
            return out.args[0]
        return out
    return impl


_OPT_NOTE = ("; linear problems are solved exactly (rational simplex, branch and bound for integer variables), "
             "convex quadratic ones by exact active-set enumeration, others with SciPy SLSQP ({}); constraints may "
             "be equations, inequalities, vector inequalities (VectorGreaterEqual, ...), x \\[Element] Integers and "
             "{{x, ...}} \\[Element] Disk/Ball/Rectangle/Cuboid; infeasible or unbounded problems raise")

for _nm, _sense, _what, _local in [
        ("FindArgMin", "min", "arg", True), ("FindArgMax", "max", "arg", True),
        ("FindMinValue", "min", "value", True), ("FindMaxValue", "max", "value", True),
        ("NArgMin", "min", "arg", False), ("NArgMax", "max", "arg", False),
        ("NMinValue", "min", "value", False), ("NMaxValue", "max", "value", False)]:
    _desc = ("the {} {} of f (optionally {{f, cons}}) ".format(
        "local" if _local else "global", ("minimiser" if _sense == "min" else "maximiser") if _what == "arg"
        else ("minimum value" if _sense == "min" else "maximum value")))
    _desc += ("as a list of values for vars ({x, x0} starting points and {x, x0, xmin, xmax} accepted)"
              if _what == "arg" and _local else
              ("(a single variable gives a number, a list of variables a list)" if _what == "arg" else
               "as a machine number"))
    fn(_nm, **ADDED, notes=_desc + _OPT_NOTE.format(
        "a local search from the starting point (default 1)" if _local else
        "multi-start, so a global optimum is likely but not guaranteed for non-convex problems"))(
        _argopt(_nm, _sense, _what, _local))


# ── LinearOptimization and the convex-optimisation family ──

_PROPS = ("PrimalMinimizer", "PrimalMinimizerRules", "PrimalMinimizerVector", "PrimalMinimumValue")


def _is_prop(x):
    return isinstance(x, str) or (is_list(x) and x.args and all(isinstance(a, str) for a in x.args))


def _numeric_list(x):
    return is_list(x) and all(isinstance(a, (int, float)) or (not is_list(a) and _is_num(S(a)))
                              for a in x.args)


def _matrix_form_prob(c_vec, ineq, eq, domains, fname, quad=None, frac=None):
    """Problem for the matrix forms: minimise c.x (or 1/2 x.q.x + c.x, or (a.x+b)/(g.x+d)) subject to
    a.x + b >= 0 and aeq.x + beq == 0."""
    n = len(c_vec.args) if c_vec is not None else None
    if n is None:
        raise ValueError(f"{fname}: invalid objective specification")
    names = [Symbol(f"x$${i + 1}") for i in range(n)]
    xs = L(*names)
    if quad is not None:
        q = quad
        f = Expr("Plus", Expr("Times", sp.Rational(1, 2) if False else Expr("Rational", 1, 2),
                              Expr("Dot", xs, q, xs)), Expr("Dot", c_vec, xs))
    elif frac is not None:
        a1, b1, g1, d1 = frac
        f = Expr("Times", Expr("Plus", Expr("Dot", a1, xs), b1),
                 Expr("Power", Expr("Plus", Expr("Dot", g1, xs), d1), -1))
    else:
        f = Expr("Dot", c_vec, xs)
    cons = []
    for spec, op in ((ineq, "GreaterEqual"), (eq, "Equal")):
        if spec is None:
            continue
        if not (is_list(spec) and len(spec.args) == 2):
            raise ValueError(f"{fname}: constraints must be given as {{a, b}}")
        a, b = spec.args
        lhs = ev(Expr("Plus", Expr("Dot", a, xs), b))
        for e in _flat_expr(lhs):
            cons.append(Expr(op, e, 0))
    dom = list(domains.args) if domains is not None and is_list(domains) else \
        ([domains] * n if domains is not None else None)
    prob = _build(ev(f), L(*cons), xs, fname, domains=dom)
    return prob


def _opt_props(prob, x, v, exact, props, default_rules, fname):
    vals = _result_values(prob, x, exact)
    value = (sp.Rational(v.numerator, v.denominator) if exact and isinstance(v, Fraction) and not prob.inexact
             and not prob.ints else sp.Float(float(v)))

    def one(p):
        if p in ("PrimalMinimizer", "PrimalMinimizerVector"):
            return _vector(prob, vals) if p == "PrimalMinimizerVector" or not default_rules else \
                _vector(prob, vals)
        if p == "PrimalMinimizerRules":
            return _rules(prob, vals)
        if p == "PrimalMinimumValue":
            return R(value)
        raise NotImplementedError(f"{fname}: the property {p!r} is not supported (supported: "
                                  f"{', '.join(_PROPS)})")
    if props is None:
        return _rules(prob, vals) if default_rules else _vector(prob, vals)
    if isinstance(props, str):
        return one(props)
    return L(*(one(p) for p in props.args))


def _family(fname, kind):
    """Implementation for LinearOptimization-like functions: kind in lp, qp, lfp, convex."""
    def impl(*rest):
        args, opts = split_options(rest, _OPT_OPTS, fname)
        _opt_options(opts, fname)
        props = None
        if args and _is_prop(args[-1]) and len(args) >= 2:
            props, args = args[-1], args[:-1]
        if len(args) < 2:
            raise ValueError(f"{fname}: expected an objective, constraints and variables")
        matrix_form = False
        if kind == "lp" and _numeric_list(args[0]):
            matrix_form = True
        if kind == "qp" and is_list(args[0]) and len(args[0].args) == 2 and is_list(args[0].args[1]) and \
                _numeric_list(args[0].args[1]):
            matrix_form = True
        if kind == "lfp" and is_list(args[0]) and len(args[0].args) == 4 and is_list(args[0].args[0]):
            matrix_form = True
        if matrix_form:
            rest_args = list(args[1:])
            domains = None
            if rest_args and (is_sym(rest_args[-1], "Integers") or is_sym(rest_args[-1], "Reals") or
                              (is_list(rest_args[-1]) and rest_args[-1].args and
                               all(is_sym(a, "Integers") or is_sym(a, "Reals") for a in rest_args[-1].args))):
                domains = rest_args.pop()
            ineq = rest_args[0] if rest_args else None
            eq = rest_args[1] if len(rest_args) > 1 else None
            if ineq is not None and is_sym(ineq, "None"):
                ineq = None
            if kind == "lp":
                prob = _matrix_form_prob(args[0], ineq, eq, domains, fname)
            elif kind == "qp":
                prob = _matrix_form_prob(args[0].args[1], ineq, eq, domains, fname, quad=args[0].args[0])
            else:
                a1, b1, g1, d1 = args[0].args
                prob = _matrix_form_prob(a1, ineq, eq, domains, fname, frac=(a1, b1, g1, d1))
            default_rules = False
        else:
            if len(args) != 3:
                raise ValueError(f"{fname}: expected {fname}[f, cons, vars]")
            prob = _build(args[0], args[1], args[2], fname)
            default_rules = True
        if kind == "lfp":
            x, v, exact = _solve_lfp(prob, fname)
        elif kind == "lp":
            r = _solve_lp(prob, fname)
            if r is None:
                raise ValueError(f"{fname}: the objective and constraints must be linear in the variables")
            x, v, exact = r[0], r[1], True
        elif kind == "qp":
            if _qp_data(prob) is None:
                raise ValueError(f"{fname}: the objective must be quadratic in the variables")
            r = _solve_lp(prob, fname)
            if r is None:
                r = _solve_qp(prob, fname)
            if r is None:
                if any(_linear_parts(e, prob.syms) is None for e, _ in prob.cons):
                    raise ValueError(f"{fname}: the constraints must be linear in the variables")
                if not _psd(_qp_data(prob)[0]):
                    raise _OptError(f"{fname}: the objective is not convex")
                r = _solve_nlp(prob, fname, False)
            x, v, exact = r[0], r[1], False
        elif kind == "geometric":
            x, v, exact = _solve_geometric(prob, fname)
        else:
            x, v, exact = _solve(prob, fname, convex=True)
            exact = False
        return _opt_props(prob, x, v, exact, props, default_rules, fname)
    return impl


def _solve_lfp(prob, fname):
    """Linear-fractional programming by the Charnes-Cooper transformation (exact)."""
    num, den = sp.fraction(sp.together(prob.obj))
    ln, ld = _linear_parts(num, prob.syms), _linear_parts(den, prob.syms)
    if ln is None or ld is None:
        raise ValueError(f"{fname}: the objective must be a ratio of linear (affine) functions")
    rows = []
    for e, op in prob.cons:
        lp = _linear_parts(e, prob.syms)
        if lp is None:
            raise ValueError(f"{fname}: the constraints must be linear")
        rows.append((lp, op))
    n = len(prob.syms)
    try:
        a = [_frac(v) for v in ln[0]]
        b = _frac(ln[1])
        g = [_frac(v) for v in ld[0]]
        d = _frac(ld[1])
        A = [[_frac(v) for v in lp[0]] for lp, _ in rows]
        bb = [-_frac(lp[1]) for lp, _ in rows]
    except TypeError:
        raise NotImplementedError(f"{fname}: irrational coefficients are not supported")
    kinds = [op for _, op in rows]

    def solve(bounds):
        # variables y (n) and t: minimise a.y + b t s.t. A y - bb t (op) 0, g.y + d t == 1, -t <= 0
        A2, b2, k2 = [], [], []
        for row, rhs, k in zip(A, bb, kinds):
            A2.append(row + [-rhs])
            b2.append(Fraction(0))
            k2.append(k)
        for i, op, v in bounds:
            row = [Fraction(0)] * (n + 1)
            row[i] = Fraction(1) if op == "<=" else Fraction(-1)
            row[n] = Fraction(-v) if op == "<=" else Fraction(v)
            A2.append(row)
            b2.append(Fraction(0))
            k2.append("<=")
        A2.append(g + [d])
        b2.append(Fraction(1))
        k2.append("==")
        row = [Fraction(0)] * (n + 1)
        row[n] = Fraction(-1)
        A2.append(row)
        b2.append(Fraction(0))
        k2.append("<=")
        res = _simplex(a + [b], A2, b2, k2, n + 1)
        if res[0] != "optimal":
            return res
        y, t = res[1][:n], res[1][n]
        if t <= 0:
            return ("unbounded",)
        return ("optimal", [yi / t for yi in y], res[2])
    def run(sign):
        nonlocal a, b, g, d
        a, b, g, d = [sign * v for v in a0], sign * b0, [sign * v for v in g0], sign * d0
        return _bnb(solve, prob.ints) if prob.ints else solve([])
    a0, b0, g0, d0 = a, b, g, d
    rp, rn = run(1), run(-1)
    if rp[0] == "optimal" and rn[0] == "optimal":
        raise _OptError(f"{fname}: the denominator changes sign on the feasible set")
    res = rp if rp[0] == "optimal" else (rn if rn[0] == "optimal" else rp)
    if res[0] == "infeasible":
        raise _OptError(f"{fname}: the constraints are infeasible")
    if res[0] == "unbounded":
        raise _OptError(f"{fname}: the problem is unbounded or its minimum is not attained")
    return res[1], res[2], True


def _solve_geometric(prob, fname):
    """Geometric programming: positive variables x = Exp[u]; minimise Log[f] with Log[g/h] <= 0."""
    us = [sp.Symbol(f"u${i}", real=True) for i in range(len(prob.syms))]
    sub = {s: sp.exp(u) for s, u in zip(prob.syms, us)}
    gp = _Prob()
    gp.syms = us
    gp.ints = set(prob.ints)
    if prob.ints:
        raise NotImplementedError(f"{fname}: integer variables are not supported")
    obj = prob.obj.xreplace(sub)
    if not _is_posy(prob.obj, prob.syms):
        raise ValueError(f"{fname}: the objective must be a posynomial (or a generalized posynomial)")
    gp.obj = sp.log(obj)
    for e, op in prob.cons:
        # e = lhs - rhs; recover lhs/rhs as positive and negative parts
        pos = sp.Add(*[t for t in sp.Add.make_args(sp.expand(e)) if not t.could_extract_minus_sign()])
        neg = -sp.Add(*[t for t in sp.Add.make_args(sp.expand(e)) if t.could_extract_minus_sign()])
        if neg == 0:
            raise _OptError(f"{fname}: the constraint {e} <= 0 cannot be satisfied by positive variables")
        if pos == 0:
            continue
        expr = sp.log(pos.xreplace(sub)) - sp.log(neg.xreplace(sub))
        gp.cons.append((sp.simplify(expr), op))
    x, v = _solve_nlp(gp, fname, False)
    xs = [math.exp(t) for t in x]
    return xs, float(sp.N(prob.obj.subs(dict(zip(prob.syms, xs))))), False


def _is_posy(e, syms):
    try:
        val = e.subs({s: sp.Rational(3, 2) for s in syms})
        return bool(val.is_positive)
    except Exception:
        return False


_FAMILY_NOTE = ("; constraints as in NMinimize (equations, inequalities, vector inequalities, x \\[Element] "
                "Integers); properties \"PrimalMinimizer\", \"PrimalMinimizerRules\", \"PrimalMinimizerVector\" and "
                "\"PrimalMinimumValue\" (dual properties raise); infeasible or unbounded problems raise")

fn("LinearOptimization", **ADDED,
   notes="LinearOptimization[f, cons, vars] gives the minimiser rules; LinearOptimization[c, {a, b}] (a.x + b >= 0) "
         "and [c, {a, b}, {aeq, beq}] (aeq.x + beq == 0) give the vector; domains {Integers, Reals, ...}; exact "
         "rational simplex (Bland's rule) so exact input gives exact results, branch and bound for integer "
         "variables (integer values exact, the others machine numbers as Wolfram); vertex choice among multiple "
         "optima can differ from Wolfram's" + _FAMILY_NOTE)(_family("LinearOptimization", "lp"))
fn("QuadraticOptimization", **ADDED,
   notes="minimise a convex quadratic objective subject to linear constraints: QuadraticOptimization[f, cons, "
         "vars] or [{q, c}, {a, b}, {aeq, beq}] (1/2 x.q.x + c.x); exact active-set enumeration (up to 12 "
         "inequalities, results as machine numbers like Wolfram), SciPy beyond that; a non-convex objective "
         "raises" + _FAMILY_NOTE)(_family("QuadraticOptimization", "qp"))
fn("LinearFractionalOptimization", **ADDED,
   notes="minimise (a.x + b)/(c.x + d) subject to linear constraints (the denominator must be positive on the "
         "feasible set): exact Charnes-Cooper transformation to a linear program, branch and bound for integer "
         "variables; matrix form [{a, b, c, d}, {a1, b1}, ...]" + _FAMILY_NOTE)(
    _family("LinearFractionalOptimization", "lfp"))
for _nm, _what in [("ConvexOptimization", "convex objective and constraints"),
                   ("ConicOptimization", "linear objective with conic constraints (NormCone, SemidefiniteCone, "
                                         "ExponentialCone, PowerCone, quadratic inequalities)"),
                   ("SecondOrderConeOptimization", "linear objective with second-order-cone (Norm[...] <= ...) "
                                                   "constraints"),
                   ("SemidefiniteOptimization", "linear objective with linear matrix inequalities "
                                                "(VectorGreaterEqual[{m, 0}, {\"SemidefiniteCone\", n}])")]:
    fn(_nm, **ADDED,
       notes=f"{_nm}[f, cons, vars] for a {_what}: linear problems are solved exactly (results as machine "
             "numbers), convex quadratic ones by active-set enumeration, the rest with SciPy SLSQP after a sampled "
             "convexity check (a visibly non-convex objective or constraint raises); matrix forms are not "
             "supported" + _FAMILY_NOTE)(_family(_nm, "convex"))
fn("GeometricOptimization", **ADDED,
   notes="GeometricOptimization[f, cons, vars] for a posynomial objective with posynomial <= monomial and monomial "
         "== monomial constraints in positive variables: solved as the convex problem in u = Log[x] (SciPy SLSQP); "
         "matrix forms are not supported" + _FAMILY_NOTE)(_family("GeometricOptimization", "geometric"))


# ── parametric and robust convex optimisation ──

@fn("ParametricConvexOptimization", **ADDED,
    notes="ParametricConvexOptimization[f, cons, vars, pars] gives ParametricFunction[...]; applying it to "
          "parameter values solves ConvexOptimization[f, cons, vars] with the parameters substituted (each call "
          "re-solves the problem; Wolfram's precompiled solver data are not reproduced)" + _FAMILY_NOTE)
def _parametricconvex(f, cons, vars_, pars, *rest):
    args, opts = split_options(rest, _OPT_OPTS, "ParametricConvexOptimization")
    props = args[0] if args else None
    ps = list(pars.args) if is_list(pars) else [pars]
    for p in ps:
        if not isinstance(p, Symbol):
            raise ValueError(f"ParametricConvexOptimization: {p} is not a valid parameter")
    return Expr("ParametricFunction", "ConvexOptimization", f, cons, vars_, L(*ps),
                props if props is not None else Symbol("Automatic"))


_PREV_PF_HOOK = KL.COMPOUND_HEAD_HOOKS.get("ParametricFunction")


def _parametric_call(head, args, ev_):
    if len(head.args) == 6 and head.args[0] == "ConvexOptimization":
        _, f, cons, vars_, pars, props = head.args
        if len(args) != len(pars.args):
            raise ValueError(f"ParametricFunction: expected {len(pars.args)} parameter value(s)")
        mapping = {p: a for p, a in zip(pars.args, args)}
        call = [_subst(f, mapping), _subst(cons, mapping), vars_]
        if not is_sym(props, "Automatic"):
            call.append(props)
        return ev_(Expr("ConvexOptimization", *call))
    if _PREV_PF_HOOK is not None:
        return _PREV_PF_HOOK(head, args, ev_)
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["ParametricFunction"] = _parametric_call


def _param_box(cond, pars, fname):
    """{par: (lo, hi)} from a conjunction of bounds on single parameters (or Interval memberships)."""
    box = {}
    names = {p.name for p in pars}

    def walk(c):
        if c is True:
            return
        if is_list(c) or (isinstance(c, Expr) and c.head == "And"):
            for a in c.args:
                walk(a)
            return
        if isinstance(c, Expr) and c.head == "Element" and isinstance(c.args[0], Symbol) and \
                isinstance(c.args[1], Expr) and c.args[1].head == "Interval":
            lo, hi = c.args[1].args[0].args
            box[c.args[0].name] = (S(lo), S(hi))
            return
        if isinstance(c, Expr) and c.head in ("LessEqual", "Less", "GreaterEqual", "Greater") and len(c.args) == 2:
            a, b = c.args
            lt = c.head in ("LessEqual", "Less")
            if isinstance(a, Symbol) and a.name in names and not (isinstance(b, Symbol) and b.name in names):
                lo, hi = box.get(a.name, (None, None))
                box[a.name] = (lo, S(b)) if lt else (S(b), hi)
                return
            if isinstance(b, Symbol) and b.name in names:
                lo, hi = box.get(b.name, (None, None))
                box[b.name] = (S(a), hi) if lt else (lo, S(a))
                return
        raise NotImplementedError(f"{fname}: only box (interval) uncertainty sets are supported, not {c}")
    walk(cond)
    for p in pars:
        lo, hi = box.get(p.name, (None, None))
        if lo is None or hi is None or not (_is_num(lo) and _is_num(hi)):
            raise NotImplementedError(f"{fname}: the parameter {p} needs finite numeric bounds")
    return box


def _multiaffine(e, pars):
    """Affine in each parameter, or (sampled) convex in the parameters jointly: either way the maximum over a box
    is attained at a vertex."""
    real = {s_: sp.Symbol(s_.name, real=True) for s_ in e.free_symbols}
    e = sp.expand(e.xreplace(real))
    pars = [real.get(p, sp.Symbol(p.name, real=True)) for p in pars]
    try:
        if all(sp.degree(e, p) <= 1 if e.has(p) else True for p in pars):
            return True
    except sp.PolynomialError:
        pass
    np = _np()
    others = sorted(e.free_symbols - set(pars), key=str)
    try:
        H = sp.lambdify(others + list(pars), sp.hessian(e, pars), modules=["numpy"])
    except Exception:
        return False
    rng = np.random.default_rng(3)
    for _ in range(20):
        pt = rng.normal(0, 1.5, len(others) + len(pars))
        try:
            with np.errstate(all="ignore"):
                Hn = np.array(H(*pt), dtype=float)
        except Exception:
            continue
        if np.all(np.isfinite(Hn)) and np.linalg.eigvalsh((Hn + Hn.T) / 2)[0] < -1e-9:
            return False
    return True


@fn("RobustConvexOptimization", **ADDED, hold=True,
    notes="RobustConvexOptimization[f, ForAll[pars, cond, cons], vars] and [f, cons, vars, pars] (parameter bounds "
          "among cons): the constraints (and objective, through an epigraph variable) must hold for every "
          "parameter value in a box given by bounds or Interval memberships; when the objective and constraint "
          "functions are affine in each parameter or convex in the parameters (sampled Hessian check) the worst case "
          "is attained at the box vertices, which are all enforced; other uncertainty sets or parameter dependence "
          "raise" + _FAMILY_NOTE)
def _robustconvex(f, spec, vars_, *rest):
    fname = "RobustConvexOptimization"
    f, spec, vars_ = ev(f), spec, ev(vars_)
    rest = [ev(r) for r in rest]
    args, opts = split_options(rest, _OPT_OPTS, fname)
    if isinstance(spec, Expr) and spec.head == "ForAll" and len(spec.args) == 3:
        pv, cond, cons = spec.args
        pars = list(pv.args) if is_list(pv) else [pv]
        cons = ev(cons)
        cond = ev(cond)
        props = args[0] if args else None
    else:
        cons = ev(spec)
        if not args:
            raise ValueError(f"{fname}: expected ForAll[pars, cond, cons] or a parameter list")
        pv = args[0]
        pars = list(pv.args) if is_list(pv) else [pv]
        props = args[1] if len(args) > 1 else None
        cond_parts, keep = [], []
        names = {p.name for p in pars}
        for c in (cons.args if is_list(cons) or (isinstance(cons, Expr) and cons.head == "And") else [cons]):
            syms = {s for s in _symbols_in(c)}
            (cond_parts if syms and syms <= names else keep).append(c)
        cond, cons = L(*cond_parts), L(*keep)
    box = _param_box(cond, pars, fname)
    psyms = [sp.Symbol(p.name) for p in pars]
    t = Symbol("t$robust")
    obj_has = bool(_symbols_in(f) & {p.name for p in pars})
    vertices = list(itertools.product(*[box[p.name] for p in pars]))
    all_cons = []
    for vert in vertices:
        mapping = {p: R(v) for p, v in zip(pars, vert)}
        all_cons.append(_subst(cons, mapping))
        if obj_has:
            all_cons.append(Expr("LessEqual", _subst(f, mapping), t))
    # affine-in-parameter check on the constraint and objective expressions
    for e in [f] + _flat_cons(cons):
        if isinstance(e, Expr) and e.head in _REL:
            pairs = list(zip(e.args, e.args[1:]))
        else:
            pairs = [(e, 0)]
        for a_, b_ in pairs:
            try:
                x_ = S(b_) - S(a_) if isinstance(e, Expr) and e.head in ("GreaterEqual", "Greater") else \
                    S(a_) - S(b_)
            except Exception:
                continue
            if isinstance(x_, sp.Expr) and not _multiaffine(x_, psyms):
                raise NotImplementedError(f"{fname}: the problem must depend affinely on each parameter")
    vlist = list(vars_.args) if is_list(vars_) else [vars_]
    if obj_has:
        prob = _build(t, L(*all_cons), L(*vlist, t), fname)
        x, v, _ = _solve(prob, fname, convex=True)
        prob.template = prob.template[:-1]
    else:
        prob = _build(f, L(*all_cons), vars_, fname)
        x, v, _ = _solve(prob, fname, convex=True)
    return _opt_props(prob, x, v, False, props, True, fname)


def _symbols_in(e):
    out = set()
    if isinstance(e, Symbol):
        out.add(e.name)
    elif isinstance(e, Expr):
        for a in e.args:
            out |= _symbols_in(a)
    return out


def _flat_cons(c):
    if is_list(c) or (isinstance(c, Expr) and c.head == "And"):
        out = []
        for a in c.args:
            out.extend(_flat_cons(a))
        return out
    return [c]


# ── KnapsackSolve ──

def _knap_item(it, fname):
    """(payoff, cost, maxcount or None) of one item specification."""
    if is_assoc(it) or (is_list(it) and it.args and all(is_rule(r) for r in it.args)):
        rules = {}
        items = it.args if is_list(it) else ev(Expr("Normal", it)).args
        for r in items:
            rules[r.args[0]] = r.args[1]
        if "Cost" not in rules:
            raise ValueError(f"{fname}: an item must specify a \"Cost\"")
        return (rules.get("Payoff", 1), rules["Cost"], rules.get("MaxCount"))
    if is_list(it):
        a = it.args
        if len(a) == 2:
            return (a[0], a[1], None)
        if len(a) == 3:
            return (a[0], a[1], a[2])
        raise ValueError(f"{fname}: invalid item {it}")
    return (1, it, None)


@fn("KnapsackSolve", **ADDED,
    notes="KnapsackSolve[{cost1, ...}, capacity] maximises the number of items, KnapsackSolve[{{payoff, cost}, "
          "...}, capacity] and {payoff, cost, maxcount} items (or \"Payoff\"/\"Cost\"/\"MaxCount\" rules or "
          "associations) maximise the total payoff with total cost <= capacity, giving the item counts (an "
          "association of items gives an association); exact integer programming (branch and bound on the exact "
          "simplex), ties broken by the smaller total cost; several capacities (vector costs) raise")
def _knapsack(items, capacity, *rest):
    fname = "KnapsackSolve"
    if rest:
        raise ValueError(f"{fname} called with too many arguments")
    keys = None
    if is_assoc(items):
        norm = ev(Expr("Normal", items))
        keys = [r.args[0] for r in norm.args]
        specs = [r.args[1] for r in norm.args]
    elif is_list(items) and items.args and all(is_rule(r) for r in items.args) and \
            not all(isinstance(r.args[0], str) and r.args[0] in ("Payoff", "Cost", "MaxCount") for r in items.args):
        keys = [r.args[0] for r in items.args]
        specs = [r.args[1] for r in items.args]
    else:
        specs = _items(items, fname, "item list")
    if is_list(capacity):
        raise NotImplementedError(f"{fname}: several capacities (vector costs) are not supported")
    data = [_knap_item(it, fname) for it in specs]
    n = len(data)
    names = [Symbol(f"k$${i + 1}") for i in range(n)]
    pay = Expr("Plus", *(Expr("Times", p, v) for (p, _, _), v in zip(data, names))) if n else 0
    cost = Expr("Plus", *(Expr("Times", c, v) for (_, c, _), v in zip(data, names))) if n else 0
    cons = [Expr("LessEqual", cost, capacity)]
    for (p, c, mc), v in zip(data, names):
        cons.append(Expr("GreaterEqual", v, 0))
        if mc is not None and not is_sym(mc, "Infinity"):
            cons.append(Expr("LessEqual", v, mc))
        if not _is_num(S(c)) or float(sp.N(S(c))) <= 0:
            if mc is None:
                raise ValueError(f"{fname}: item costs must be positive (or the item needs a MaxCount)")
    dom = [Symbol("Integers")] * n
    prob = _build(ev(Expr("Times", -1, pay)), ev(L(*cons)), L(*names), fname, domains=dom)
    r = _solve_lp(prob, fname)
    best = r[1]
    # tie-break: least total cost among optimal payoffs
    prob2 = _build(ev(cost), ev(L(*cons, Expr("LessEqual", Expr("Times", -1, pay),
                                               R(sp.Rational(best.numerator, best.denominator))))),
                   L(*names), fname, domains=dom)
    r2 = _solve_lp(prob2, fname)
    counts = [sp.Integer(int(round(v))) for v in r2[0]]
    if keys is not None:
        return Expr("Association", *(Expr("Rule", k, R(c)) for k, c in zip(keys, counts)))
    return L(*(R(c) for c in counts))


# ── FindShortestTour ──

def _tour_dist(opts, pts, fname):
    df = opts.get("DistanceFunction")
    if df is not None and not is_sym(df, "Automatic"):
        return lambda a, b: S(ev(K.call(df, a, b)))
    if all(isinstance(p, str) for p in pts):
        return lambda a, b: S(ev(Expr("EditDistance", a, b)))
    return lambda a, b: S(ev(Expr("EuclideanDistance", a, b)))


@fn("FindShortestTour", **ADDED,
    notes="FindShortestTour[{p1, ...}] gives {length, tour} for the shortest closed tour (starting and ending at "
          "1); FindShortestTour[pts, j, k] the shortest path from pj to pk through all points; points are vectors "
          "(EuclideanDistance, exact lengths for exact points) or strings (EditDistance), or use DistanceFunction "
          "-> f; exact (Held-Karp dynamic programming) for up to 12 points, a 2-opt/Or-opt local search beyond "
          "that (not guaranteed optimal, like Wolfram's heuristics); the direction of an optimal tour and the choice "
          "among equally short tours can differ from Wolfram's; graphs and GeoPositions raise")
def _findshortesttour(pts, *rest):
    fname = "FindShortestTour"
    args, opts = split_options(rest, {"DistanceFunction", "Method"}, fname)
    if isinstance(pts, Expr) and pts.head == "Graph":
        raise NotImplementedError(f"{fname}: graph input is not supported")
    P = _items(pts, fname, "point list")
    if any(isinstance(p, Expr) and p.head == "GeoPosition" for p in P):
        raise NotImplementedError(f"{fname}: GeoPosition points are not supported")
    n = len(P)
    if n == 0:
        return L(0, L())
    dist = _tour_dist(opts, P, fname)
    D = [[sp.Integer(0)] * n for _ in range(n)]
    Df = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = dist(P[i], P[j])
            D[i][j] = D[j][i] = d
            Df[i][j] = Df[j][i] = float(sp.N(d))
    if args:
        if len(args) != 2:
            raise ValueError(f"{fname}: expected FindShortestTour[pts, j, k]")
        j0 = int_arg(args[0], fname) - 1
        k0 = int_arg(args[1], fname) - 1
        if not (0 <= j0 < n and 0 <= k0 < n):
            raise ValueError(f"{fname}: point index out of range")
        path = _shortest_path_tsp(Df, j0, k0)
        length = sum((D[a][b] for a, b in zip(path, path[1:])), sp.Integer(0))
        return L(R(sp.nsimplify(length) if False else sp.simplify(length)), L(*(i + 1 for i in path)))
    if n == 1:
        return L(0, L(1, 1))
    tour = _tsp(Df)
    length = sum((D[a][b] for a, b in zip(tour, tour[1:] + tour[:1])), sp.Integer(0))
    return L(R(sp.simplify(length)), L(*([i + 1 for i in tour] + [1])))


def _held_karp(Df, start, end=None):
    n = len(Df)
    others = [i for i in range(n) if i != start and i != end]
    idx = {v: k for k, v in enumerate(others)}
    m = len(others)
    INF = float("inf")
    dp = {}
    for v in others:
        dp[(1 << idx[v], v)] = (Df[start][v], start)
    for size in range(2, m + 1):
        for subset in itertools.combinations(others, size):
            mask = 0
            for v in subset:
                mask |= 1 << idx[v]
            for v in subset:
                pm = mask & ~(1 << idx[v])
                best = (INF, None)
                for u in subset:
                    if u == v:
                        continue
                    c = dp[(pm, u)][0] + Df[u][v]
                    if c < best[0] - 1e-12:
                        best = (c, u)
                dp[(mask, v)] = best
    full = (1 << m) - 1
    target = end if end is not None else start
    if m == 0:
        return [start] + ([end] if end is not None else [])
    best = (INF, None)
    for v in others:
        c = dp[(full, v)][0] + Df[v][target]
        if c < best[0] - 1e-12:
            best = (c, v)
    path = []
    mask, v = full, best[1]
    while v != start:
        path.append(v)
        pv = dp[(mask, v)][1]
        mask &= ~(1 << idx[v])
        v = pv
    path.reverse()
    return [start] + path + ([end] if end is not None else [])


def _two_opt(tour, Df, closed=True):
    n = len(tour)
    improved = True
    while improved:
        improved = False
        rng_ = range(1, n - 1) if closed else range(1, n - 2)
        for i in rng_:
            for j in range(i + 1, n if closed else n - 1):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n] if closed else tour[j + 1]
                if Df[a][c] + Df[b][d] < Df[a][b] + Df[c][d] - 1e-12:
                    tour[i:j + 1] = reversed(tour[i:j + 1])
                    improved = True
    return tour


def _tsp(Df):
    n = len(Df)
    if n <= 12:
        return _held_karp(Df, 0)
    best = None
    for s0 in range(min(n, 8)):
        tour = [s0]
        left = set(range(n)) - {s0}
        while left:
            nxt = min(left, key=lambda v: Df[tour[-1]][v])
            tour.append(nxt)
            left.remove(nxt)
        k = tour.index(0)
        tour = tour[k:] + tour[:k]
        tour = _two_opt(tour, Df)
        L_ = sum(Df[a][b] for a, b in zip(tour, tour[1:] + tour[:1]))
        if best is None or L_ < best[0] - 1e-12:
            best = (L_, tour)
    return best[1]


def _shortest_path_tsp(Df, j0, k0):
    n = len(Df)
    if j0 == k0:
        t = _tsp(Df)
        k = t.index(j0)
        return t[k:] + t[:k] + [j0]
    if n <= 12:
        return _held_karp(Df, j0, k0)
    tour = [j0] + [v for v in range(n) if v not in (j0, k0)] + [k0]
    return _two_opt(tour, Df, closed=False)


# ── Bayesian optimisation (Gaussian-process expected improvement) ──

def _bo_domain(spec, fname):
    """('box', lows, highs, scalar?) | ('list', [configs]) | ('sampler', f)."""
    if is_list(spec) and spec.args:
        return ("list", list(spec.args))
    if isinstance(spec, Expr):
        if spec.head == "Interval" and len(spec.args) == 1 and is_list(spec.args[0]):
            lo, hi = spec.args[0].args
            return ("box", [float(sp.N(S(lo)))], [float(sp.N(S(hi)))], True)
        if spec.head in ("Rectangle", "Cuboid"):
            lo = spec.args[0] if spec.args else None
            hi = spec.args[1] if len(spec.args) > 1 else None
            if lo is None:
                raise ValueError(f"{fname}: invalid region {spec}")
            los = [float(sp.N(S(v))) for v in lo.args]
            his = [float(sp.N(S(v))) for v in hi.args] if hi is not None else [v + 1 for v in los]
            return ("box", los, his, False)
        if spec.head in ("Function",) or isinstance(spec.head, Expr):
            return ("sampler", spec)
    if isinstance(spec, Symbol):
        return ("sampler", spec)
    raise NotImplementedError(f"{fname}: the domain {spec} is not supported (use Interval, Rectangle, Cuboid, a "
                              "list of configurations or a sampler function)")


def _gp_fit_predict(X, y, Xs, noise):
    np = _np()
    ym, ys = y.mean(), y.std() or 1.0
    yn = (y - ym) / ys
    best = None
    span = np.ptp(X, axis=0)
    span[span == 0] = 1.0
    for ls in (0.05, 0.1, 0.2, 0.4, 0.8):
        ell = ls * span

        def k(A, B):
            d = ((A[:, None, :] - B[None, :, :]) / ell) ** 2
            return np.exp(-0.5 * d.sum(-1))
        Kxx = k(X, X) + (noise + 1e-8) * np.eye(len(X))
        try:
            Lc = np.linalg.cholesky(Kxx)
        except np.linalg.LinAlgError:
            continue
        alpha = np.linalg.solve(Lc.T, np.linalg.solve(Lc, yn))
        ll = -0.5 * yn @ alpha - np.log(np.diag(Lc)).sum()
        if best is None or ll > best[0]:
            best = (ll, k, Lc, alpha)
    _, k, Lc, alpha = best
    Ks = k(Xs, X)
    mu = Ks @ alpha
    v = np.linalg.solve(Lc, Ks.T)
    var = np.clip(1.0 - (v ** 2).sum(0), 1e-12, None)
    return mu * ys + ym, np.sqrt(var) * ys


def _bayes(fname, sense):
    def impl(f, spec, *rest):
        args, opts = split_options(rest, {"AssumeDeterministic", "InitialEvaluationHistory", "MaxIterations",
                                          "Method", "ProgressReporting", "RandomSeeding"}, fname)
        if args:
            raise ValueError(f"{fname} called with too many arguments")
        np = _np()
        from math import erf, sqrt, pi, exp
        method = opts.get("Method", "MaxExpectedImprovement")
        if is_sym(method, "Automatic"):
            method = "MaxExpectedImprovement"
        if method not in ("MaxExpectedImprovement", "MaxImprovementProbability"):
            raise ValueError(f"{fname}: unknown Method {method}")
        iters = opts.get("MaxIterations", 100)
        iters = int_arg(iters, fname)
        seed = opts.get("RandomSeeding", 1234)
        seed = seed if isinstance(seed, int) else 1234
        det = opts.get("AssumeDeterministic", False) is True
        rng = np.random.default_rng(seed)
        dom = _bo_domain(spec, fname)
        sign = 1.0 if sense == "min" else -1.0

        def to_vec(c):
            if dom[0] == "box":
                return np.atleast_1d(np.array(c, dtype=float))
            v = _np_array_of(c)
            if v is None:
                raise NotImplementedError(f"{fname}: configurations must be numeric for the Gaussian-process model")
            return np.atleast_1d(v.astype(float)).ravel()

        def to_conf(v):
            if dom[0] == "box":
                return R(sp.Float(float(v[0]))) if dom[3] else L(*(sp.Float(float(t)) for t in v))
            return v

        def evaluate_f(conf):
            r = ev(K.call(f, conf))
            val = S(r)
            if not (isinstance(val, sp.Basic) and val.is_number and val.is_real):
                raise ValueError(f"{fname}: the function must return a real number, got {r}")
            return float(sp.N(val))

        def sample():
            if dom[0] == "box":
                lo, hi = np.array(dom[1]), np.array(dom[2])
                return lo + rng.random(len(lo)) * (hi - lo)
            if dom[0] == "sampler":
                return ev(K.call(dom[1]))
            return dom[1][int(rng.integers(len(dom[1])))]

        hist = []          # (conf (Mikoshi), vector, value)
        init = opts.get("InitialEvaluationHistory")
        if init is not None and not is_sym(init, "None"):
            for r in _items(init, fname, "evaluation history"):
                if not is_rule(r):
                    raise ValueError(f"{fname}: the evaluation history must be a list of rules conf -> value")
                c = r.args[0]
                hist.append((c, to_vec(c if dom[0] != "box" else [float(sp.N(S(t))) for t in _flat_expr(c)]),
                             float(sp.N(S(r.args[1])))))
        candidates = None
        if dom[0] == "list":
            candidates = list(dom[1])
        n_init = min(5, iters) if not hist else 0
        total = max(iters, n_init)
        nxt = None
        for it in range(total + 1):
            if it < n_init:
                if candidates is not None:
                    left = [c for c in candidates if all(c != h[0] for h in hist)]
                    if not left:
                        break
                    c = left[int(rng.integers(len(left)))]
                    v = to_vec(c)
                else:
                    raw = sample()
                    c, v = (to_conf(raw), raw) if dom[0] == "box" else (raw, to_vec(raw))
            else:
                X = np.array([h[1] for h in hist])
                y = np.array([sign * h[2] for h in hist])
                if candidates is not None:
                    pool = [c for c in candidates if det is False or all(c != h[0] for h in hist)]
                    if not pool:
                        break
                    Xs = np.array([to_vec(c) for c in pool])
                else:
                    raws = [sample() for _ in range(1500 if dom[0] == "box" else 60)]
                    Xs = np.array([r if dom[0] == "box" else to_vec(r) for r in raws])
                mu, sd = _gp_fit_predict(X, y, Xs, 1e-6 if det else 1e-3)
                ybest = y.min()
                z = (ybest - mu) / sd
                cdf = np.array([0.5 * (1 + erf(t / sqrt(2))) for t in z])
                pdf = np.exp(-0.5 * z ** 2) / sqrt(2 * pi)
                score = (ybest - mu) * cdf + sd * pdf if method == "MaxExpectedImprovement" else cdf
                k = int(np.argmax(score))
                if candidates is not None:
                    c, v = pool[k], Xs[k]
                else:
                    c, v = (to_conf(Xs[k]), Xs[k]) if dom[0] == "box" else (raws[k], Xs[k])
            if it == total:
                nxt = c
                break
            hist.append((c, v, evaluate_f(c)))
        if not hist:
            raise ValueError(f"{fname}: no configuration could be evaluated")
        vals = [h[2] for h in hist]
        k = int(np.argmin([sign * t for t in vals]))
        head = "BayesianMinimizationObject" if sense == "min" else "BayesianMaximizationObject"
        key = "Minimum" if sense == "min" else "Maximum"
        data = [Expr("Rule", f"{key}Configuration", hist[k][0]),
                Expr("Rule", f"{key}Value", R(sp.Float(vals[k]))),
                Expr("Rule", "EvaluationHistory", L(*(Expr("Rule", h[0], R(sp.Float(h[2]))) for h in hist))),
                Expr("Rule", "ObjectiveFunction", f),
                Expr("Rule", "Method", method),
                Expr("Rule", "NextConfiguration", nxt if nxt is not None else Symbol("None"))]
        return Expr(head, Expr("Association", *data))
    return impl


def _bo_props(head, args, ev_):
    if len(head.args) != 1 or not is_assoc(head.args[0]) or len(args) != 1:
        return K.UNHANDLED
    data = {r.args[0]: r.args[1] for r in head.args[0].args}
    props = list(data) + ["Properties"]

    def one(p):
        if p == "Properties":
            return L(*props, "PredictorFunction")
        if p == "PredictorFunction":
            raise NotImplementedError(f"{head.head}: the PredictorFunction property is not supported")
        if p not in data:
            raise ValueError(f"{head.head}: unknown property {p!r}")
        return data[p]
    q = args[0]
    if is_list(q):
        return L(*(one(p) for p in q.args))
    return one(q)


for _h in ("BayesianMinimizationObject", "BayesianMaximizationObject"):
    KL.COMPOUND_HEAD_HOOKS[_h] = _bo_props

_BO_NOTE = ("Gaussian-process Bayesian optimisation (RBF kernel with the length scale chosen by marginal likelihood, "
            "expected-improvement or improvement-probability acquisition over random candidates) over an Interval, "
            "Rectangle/Cuboid, a list of numeric configurations or a sampler function; options MaxIterations "
            "(default 100), AssumeDeterministic, InitialEvaluationHistory, Method, RandomSeeding (default 1234, "
            "deterministic); results are stochastic like Wolfram's and the GP model differs, so configurations "
            "differ in detail; the \"PredictorFunction\" property raises")
fn("BayesianMinimization", **ADDED, notes="BayesianMinimization[f, spec] gives a BayesianMinimizationObject: "
   + _BO_NOTE)(_bayes("BayesianMinimization", "min"))
fn("BayesianMaximization", **ADDED, notes="BayesianMaximization[f, spec] gives a BayesianMaximizationObject: "
   + _BO_NOTE)(_bayes("BayesianMaximization", "max"))


@fn("BayesianMinimizationObject", **ADDED,
    notes="result of BayesianMinimization; obj[\"MinimumConfiguration\"], [\"MinimumValue\"], "
          "[\"EvaluationHistory\"], [\"ObjectiveFunction\"], [\"Method\"], [\"NextConfiguration\"], "
          "[\"Properties\"] and lists of properties")
def _bmo(*args):
    raise Unevaluated()


@fn("BayesianMaximizationObject", **ADDED,
    notes="result of BayesianMaximization; obj[\"MaximumConfiguration\"], [\"MaximumValue\"], "
          "[\"EvaluationHistory\"], [\"ObjectiveFunction\"], [\"Method\"], [\"NextConfiguration\"], "
          "[\"Properties\"] and lists of properties")
def _bmxo(*args):
    raise Unevaluated()


# ═══════════════════════════════ matrix games ═══════════════════════════════

def _game_array(p, fname):
    """(payoff array as nested Python lists of SymPy numbers, dims of actions, number of players)."""
    d = _dims(p)
    if len(d) == 2:                                     # zero-sum matrix
        rows = [[S(v) for v in r.args] for r in p.args]
        arr = [[[v, -v] for v in r] for r in rows]
        return arr, [d[0], d[1]], 2
    if len(d) >= 3 and d[-1] == len(d) - 1:
        def conv(x):
            return [conv(a) for a in x.args] if is_list(x) else S(x)
        return conv(p), list(d[:-1]), d[-1]
    raise ValueError(f"{fname}: {p} is not a payoff matrix or an array of payoff vectors with dimensions "
                     "{m1, ..., mn, n}")


def _game_parts(g, fname):
    if not (isinstance(g, Expr) and g.head == "MatrixGame" and g.args):
        if isinstance(g, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: {g} is not a MatrixGame")
    arr, dims, n = _game_array(g.args[0], fname)
    opts = {}
    for r in g.args[1:]:
        if is_rule(r):
            k = r.args[0]
            opts[k.name if isinstance(k, Symbol) else k] = r.args[1]
    for v in _flat_py(arr):
        if not (isinstance(v, sp.Basic) and v.is_number and v.is_real):
            raise ValueError(f"{fname}: payoffs must be real numbers")
    return arr, dims, n, opts


def _flat_py(x):
    if isinstance(x, list):
        out = []
        for a in x:
            out.extend(_flat_py(a))
        return out
    return [x]


def _payoff(arr, idx):
    for i in idx:
        arr = arr[i]
    return arr


def _py_to_ml(x):
    if isinstance(x, list):
        return L(*(_py_to_ml(a) for a in x))
    return R(x)


@fn("MatrixGame", **ADDED,
    notes="MatrixGame[{{a11, ...}, ...}] (two-player zero-sum) or MatrixGame[p] with p of dimensions {m1, ..., mn, "
          "n} gives MatrixGame[payoff array, options..., \"Validity\" -> {True}]; options GameActionLabels and "
          "GamePlayerLabels; properties \"PayoffArray\", \"ComponentPayoffArray\", \"GameActionLabels\", "
          "\"GamePlayerLabels\", \"Dataset\" (Dataset of payoff vectors by action labels) and \"Plot\" "
          "(MatrixGamePlot); \"TreeGame\" and SparseArray/rule specifications raise")
def _matrixgame(p, *rest):
    fname = "MatrixGame"
    if any(isinstance(r, Expr) and is_rule(r) and r.args[0] == "Validity" for r in rest):
        raise Unevaluated()
    args, opts = split_options(rest, {"GameActionLabels", "GamePlayerLabels"}, fname)
    if args:
        raise ValueError(f"{fname} called with too many arguments")
    if is_list(p) and p.args and all(is_rule(r) for r in p.args):
        raise NotImplementedError(f"{fname}: rule (SparseArray) specifications are not supported")
    arr, dims, n = _game_array(p, fname)
    for v in _flat_py(arr):
        if not (isinstance(v, sp.Basic) and v.is_number and v.is_real):
            raise ValueError(f"{fname}: payoffs must be real numbers")
    out = [_py_to_ml(arr)]
    for k in ("GameActionLabels", "GamePlayerLabels"):
        if k in opts and not is_sym(opts[k], "Automatic"):
            v = opts[k]
            if k == "GamePlayerLabels" and (not is_list(v) or len(v.args) != n):
                raise ValueError(f"{fname}: GamePlayerLabels must be a list of {n} labels")
            if k == "GameActionLabels" and (not is_list(v) or len(v.args) != n or
                                            any(not is_list(a) or len(a.args) != m for a, m in zip(v.args, dims))):
                raise ValueError(f"{fname}: GameActionLabels must give one list of labels per player matching the "
                                 "numbers of actions")
            out.append(Expr("Rule", Symbol(k), v))
    out.append(Expr("Rule", "Validity", L(True)))
    return Expr("MatrixGame", *out)


def _labels(dims, n, opts):
    al = opts.get("GameActionLabels")
    pl = opts.get("GamePlayerLabels")
    al = [list(a.args) for a in al.args] if al is not None else [list(range(1, m + 1)) for m in dims]
    pl = list(pl.args) if pl is not None else list(range(1, n + 1))
    return al, pl


def _matrixgame_props(head, args, ev_):
    if len(args) != 1 or not isinstance(args[0], str):
        return K.UNHANDLED
    arr, dims, n, opts = _game_parts(head, "MatrixGame")
    prop = args[0]
    al, pl = _labels(dims, n, opts)
    if prop == "PayoffArray":
        return _py_to_ml(arr)
    if prop == "ComponentPayoffArray":
        def comp(k):
            def rec(x, depth):
                return [rec(a, depth + 1) for a in x] if depth < len(dims) else x[k]
            return rec(arr, 0)
        return L(*(_py_to_ml(comp(k)) for k in range(n)))
    if prop == "GameActionLabels":
        return L(*(L(*a) for a in al))
    if prop == "GamePlayerLabels":
        return L(*pl)
    if prop == "Dataset":
        def key(v):
            return str(v) if isinstance(v, int) else v

        def rec(x, depth):
            if depth == len(dims):
                return _py_to_ml(x)
            return Expr("Association", *(Expr("Rule", key(al[depth][i]), rec(x[i], depth + 1))
                                         for i in range(dims[depth])))
        return Expr("Dataset", rec(arr, 0))
    if prop == "Plot":
        return ev_(Expr("MatrixGamePlot", head))
    if prop == "Properties":
        return L("ComponentPayoffArray", "Dataset", "GameActionLabels", "GamePlayerLabels", "PayoffArray", "Plot",
                 "TreeGame")
    if prop == "TreeGame":
        raise NotImplementedError("MatrixGame: the \"TreeGame\" property is not supported")
    raise ValueError(f"MatrixGame: unknown property {prop!r}")


KL.COMPOUND_HEAD_HOOKS["MatrixGame"] = _matrixgame_props

record("GameActionLabels", AREA, "added",
       "option of MatrixGame and MatrixGamePlot: one list of action labels per player")
record("GamePlayerLabels", AREA, "added",
       "option of MatrixGame and MatrixGamePlot: a list of player labels")


def _profile(strat, dims, n, pl, fname):
    """List of n probability vectors (SymPy) from a strategy profile specification."""
    if is_assoc(strat) or (is_list(strat) and strat.args and all(is_rule(r) for r in strat.args)):
        rules = strat.args
        table = {}
        for r in rules:
            k, v = r.args
            if is_list(k):
                raise NotImplementedError(f"{fname}: joint strategies for groups of players are not supported")
            if k in pl:
                table[pl.index(k)] = v
            elif isinstance(k, int) and 1 <= k <= n:
                table[k - 1] = v
            else:
                raise ValueError(f"{fname}: unknown player {k}")
        if len(table) != n:
            raise ValueError(f"{fname}: a strategy is needed for every player")
        vecs = [table[i] for i in range(n)]
    else:
        vecs = _items(strat, fname, "strategy profile")
    if len(vecs) != n:
        raise ValueError(f"{fname}: the profile must give {n} strategies")
    out = []
    for v, m in zip(vecs, dims):
        if not is_list(v) or len(v.args) != m:
            raise ValueError(f"{fname}: a strategy must be a probability vector of length {m}")
        vals = [S(x) for x in v.args]
        out.append(vals)
    return out


def _is_profile_list(strat, n):
    return is_list(strat) and strat.args and all(is_list(s) and len(s.args) == n and
                                                 all(is_list(t) for t in s.args) for s in strat.args)


def _expected(arr, dims, n, prof, power=1):
    res = [sp.Integer(0)] * n
    for idx in itertools.product(*[range(m) for m in dims]):
        pr = sp.Integer(1)
        for j, a in enumerate(idx):
            pr *= prof[j][a]
        if pr == 0:
            continue
        pv = _payoff(arr, idx)
        for i in range(n):
            res[i] += pr * pv[i] ** power
    return [sp.nsimplify(r) if False else sp.expand(r) for r in res]


@fn("MatrixGamePayoff", **ADDED,
    notes="MatrixGamePayoff[game, {s1, ..., sn}] gives the expected payoff of each player for mixed strategies si "
          "(probability vectors; also {1 -> s1, ...} rules or <|label -> s|>, and lists of profiles); "
          "MatrixGamePayoff[game, profile, \"Expectation\" | \"Variance\"]; exact for exact strategies; joint "
          "group strategies and the distribution/simulation properties raise")
def _matrixgamepayoff(game, strat, *rest):
    fname = "MatrixGamePayoff"
    arr, dims, n, opts = _game_parts(game, fname)
    _, pl = _labels(dims, n, opts)
    prop = rest[0] if rest else "Expectation"
    if len(rest) > 1:
        raise ValueError(f"{fname} called with too many arguments")
    if prop not in ("Expectation", "Variance"):
        raise NotImplementedError(f"{fname}: the property {prop} is not supported (\"Expectation\", \"Variance\")")
    if _is_profile_list(strat, n):
        return L(*(_matrixgamepayoff(game, s, *rest) for s in strat.args))
    prof = _profile(strat, dims, n, pl, fname)
    for v in prof:
        tot = sum(v)
        if any(x.is_number and x < 0 for x in v) or (tot.is_number and abs(complex(sp.N(tot)) - 1) > 1e-9):
            raise ValueError(f"{fname}: strategies must be probability vectors")
    e1 = _expected(arr, dims, n, prof)
    if prop == "Variance":
        e2 = _expected(arr, dims, n, prof, 2)
        e1 = [sp.expand(b - a ** 2) for a, b in zip(e1, e2)]
    return L(*(R(v) for v in e1))


def _nash_check(arr, dims, n, prof, tol=0):
    vals = _expected(arr, dims, n, prof)
    for i in range(n):
        for a in range(dims[i]):
            dev = [list(p) for p in prof]
            dev[i] = [sp.Integer(int(k == a)) for k in range(dims[i])]
            v = _expected(arr, dims, n, dev)[i]
            diff = v - vals[i]
            if diff.is_number:
                if float(sp.N(diff)) > tol:
                    return False
            else:
                raise NotImplementedError("VerifyMatrixGameStrategy: symbolic strategies are not supported")
    return True


@fn("VerifyMatrixGameStrategy", **ADDED,
    notes="True when the strategy profile {s1, ..., sn} is a Nash equilibrium of the matrix game (no player gains "
          "by deviating to any pure action; machine-number strategies use a 10^-9 tolerance); a list of profiles "
          "gives a list; symbolic strategies raise")
def _verifymatrixgame(game, strat, *rest):
    fname = "VerifyMatrixGameStrategy"
    if rest:
        raise ValueError(f"{fname} called with too many arguments")
    arr, dims, n, opts = _game_parts(game, fname)
    _, pl = _labels(dims, n, opts)
    if _is_profile_list(strat, n):
        return L(*(_verifymatrixgame(game, s) for s in strat.args))
    prof = _profile(strat, dims, n, pl, fname)
    inexact = any(isinstance(x, sp.Float) for v in prof for x in v)
    return _nash_check(arr, dims, n, prof, 1e-9 if inexact else 0)


def _frac_of(v):
    return _frac(v) if (v.is_Rational or isinstance(v, sp.Float)) else Fraction(float(v))


@fn("FindMatrixGameStrategies", **ADDED,
    notes="Nash equilibria of a matrix game as a list of strategy profiles: pure equilibria first, then mixed ones, "
          "each sorted; two-player games use exact support enumeration (all equilibria of a non-degenerate game; "
          "for degenerate games the extreme equilibria with equal-size supports), n-player games give the pure "
          "equilibria only with \"Pure\" (mixed n-player equilibria raise); FindMatrixGameStrategies[game, "
          "\"Pure\" | \"Mixed\"] restricts the kind; the Method option is not supported")
def _findmatrixgame(game, *rest):
    fname = "FindMatrixGameStrategies"
    args, opts = split_options(rest, {"Method"}, fname)
    if "Method" in opts and not is_sym(opts["Method"], "Automatic") and opts["Method"] != "SolveValues":
        raise NotImplementedError(f"{fname}: the Method option is not supported")
    spec = args[0] if args else "All"
    if spec not in ("All", "Pure", "Mixed"):
        raise ValueError(f"{fname}: the specification must be \"Pure\" or \"Mixed\"")
    arr, dims, n, opts2 = _game_parts(game, fname)
    pure = []
    for idx in itertools.product(*[range(m) for m in dims]):
        prof = [[sp.Integer(int(k == a)) for k in range(m)] for a, m in zip(idx, dims)]
        if _nash_check(arr, dims, n, prof):
            pure.append(prof)
    mixed = []
    if spec != "Pure":
        if n != 2:
            raise NotImplementedError(f"{fname}: mixed equilibria of games with more than two players are not "
                                      "supported (use \"Pure\")")
        m1, m2 = dims
        A = [[_frac_of(_payoff(arr, (i, j))[0]) for j in range(m2)] for i in range(m1)]
        B = [[_frac_of(_payoff(arr, (i, j))[1]) for j in range(m2)] for i in range(m1)]
        seen = set()
        for k in range(2, min(m1, m2) + 1):
            for I in itertools.combinations(range(m1), k):
                for J in itertools.combinations(range(m2), k):
                    # y on J makes player 1 indifferent over I; x on I makes player 2 indifferent over J
                    My = [[A[i][j] for j in J] + [Fraction(-1)] for i in I] + [[Fraction(1)] * k + [Fraction(0)]]
                    y = _frac_solve(My, [Fraction(0)] * k + [Fraction(1)])
                    Mx = [[B[i][j] for i in I] + [Fraction(-1)] for j in J] + [[Fraction(1)] * k + [Fraction(0)]]
                    x = _frac_solve(Mx, [Fraction(0)] * k + [Fraction(1)])
                    if x is None or y is None:
                        continue
                    yv, v = y[:k], y[k]
                    xv, u = x[:k], x[k]
                    if any(t <= 0 for t in yv) or any(t <= 0 for t in xv):
                        continue
                    yfull = [Fraction(0)] * m2
                    for j, t in zip(J, yv):
                        yfull[j] = t
                    xfull = [Fraction(0)] * m1
                    for i, t in zip(I, xv):
                        xfull[i] = t
                    if any(sum(A[i][j] * yfull[j] for j in range(m2)) > v for i in range(m1)):
                        continue
                    if any(sum(B[i][j] * xfull[i] for i in range(m1)) > u for j in range(m2)):
                        continue
                    key = (tuple(xfull), tuple(yfull))
                    if key in seen:
                        continue
                    seen.add(key)
                    inexact = any(isinstance(v_, sp.Float) for v_ in _flat_py(arr))
                    conv = (lambda t: sp.Float(float(t))) if inexact else \
                        (lambda t: sp.Rational(t.numerator, t.denominator))
                    mixed.append([[conv(t) for t in xfull], [conv(t) for t in yfull]])

    def sort_key(prof):
        return tuple(tuple(float(t) for t in v) for v in prof)
    pure.sort(key=sort_key)
    mixed.sort(key=sort_key)
    res = (pure if spec in ("All", "Pure") else []) + (mixed if spec in ("All", "Mixed") else [])
    return L(*(L(*(L(*(R(t) for t in v)) for v in prof)) for prof in res))


def _game_plot_prims(arr, dims, n, al, pl, strategy=None):
    """Graphics primitives: a grid of payoff cells, each split into one coloured strip per player."""
    colors = [Expr("RGBColor", 0.368417, 0.506779, 0.709798), Expr("RGBColor", 0.880722, 0.611041, 0.142051),
              Expr("RGBColor", 0.560181, 0.691569, 0.194885), Expr("RGBColor", 0.922526, 0.385626, 0.209179),
              Expr("RGBColor", 0.528488, 0.470624, 0.701351)]
    rp = list(range((n + 1) // 2))
    cp = list(range((n + 1) // 2, n))
    rows = list(itertools.product(*[range(dims[i]) for i in rp]))
    cols = list(itertools.product(*[range(dims[i]) for i in cp])) or [()]
    prims = []
    H = len(rows)
    for r, ra in enumerate(rows):
        for c, ca in enumerate(cols):
            idx = [0] * n
            for i, a in zip(rp, ra):
                idx[i] = a
            for i, a in zip(cp, ca):
                idx[i] = a
            pv = _payoff(arr, idx)
            x0, y0 = c, H - 1 - r
            w = sp.Rational(1, n)
            for k in range(n):
                prims.append(L(Expr("Opacity", 0.6), colors[k % len(colors)],
                               Expr("Rectangle", L(R(x0 + k * w), y0), L(R(x0 + (k + 1) * w), y0 + 1))))
                prims.append(Expr("Text", R(pv[k]), L(R(x0 + (k + sp.Rational(1, 2)) * w), R(y0 + sp.Rational(1, 2)))))
            prims.append(L(Expr("EdgeForm", Symbol("Black")), Expr("FaceForm", Symbol("None")),
                           Expr("Rectangle", L(x0, y0), L(x0 + 1, y0 + 1))))
    for r, ra in enumerate(rows):
        lab = ", ".join(str(al[i][a]) for i, a in zip(rp, ra))
        prims.append(Expr("Text", lab, L(R(sp.Rational(-1, 2)), R(H - 1 - r + sp.Rational(1, 2)))))
    for c, ca in enumerate(cols):
        lab = ", ".join(str(al[i][a]) for i, a in zip(cp, ca))
        prims.append(Expr("Text", lab, L(R(c + sp.Rational(1, 2)), R(H + sp.Rational(1, 3)))))
    prims.append(Expr("Text", ", ".join(str(pl[i]) for i in rp) + " \\ " + ", ".join(str(pl[i]) for i in cp),
                      L(R(sp.Rational(-1, 2)), R(H + sp.Rational(1, 3)))))
    return prims


@fn("MatrixGamePlot", **ADDED,
    notes="MatrixGamePlot[game] gives Graphics[...]: a grid of payoff cells (rows: the first Ceiling[n/2] players' "
          "action combinations, columns: the rest), each split into one coloured strip per player with the payoff "
          "as Text, and action/player labels (GameActionLabels, GamePlayerLabels options); Graphics options are "
          "passed through; PlotLayout, ColorFunction and strategy thickness are not supported (raise)")
def _matrixgameplot(game, *rest):
    fname = "MatrixGamePlot"
    args, opts = split_options(rest, None, fname) if False else ([r for r in rest if not is_rule(r)],
                                                                 [r for r in rest if is_rule(r)])
    if args:
        raise NotImplementedError(f"{fname}: strategy arguments are not supported")
    arr, dims, n, gopts = _game_parts(game, fname)
    gopts = dict(gopts)
    gfx_opts = []
    for r in opts:
        k = r.args[0].name if isinstance(r.args[0], Symbol) else r.args[0]
        if k in ("GameActionLabels", "GamePlayerLabels"):
            gopts[k] = r.args[1]
        elif k in ("PlotLayout", "ColorFunction", "ColorFunctionScaling", "PlotLegends"):
            raise NotImplementedError(f"{fname}: the option {k} is not supported")
        else:
            gfx_opts.append(r)
    al, pl = _labels(dims, n, gopts)
    return Expr("Graphics", L(*_game_plot_prims(arr, dims, n, al, pl)), *gfx_opts)


# ═══════════════════════════════ tree-game strategies ═══════════════════════════════

def _tree_nodes(root):
    """Preorder list of (path, node) for action nodes, and players in order of first appearance."""
    out, players = [], []

    def walk(n, path):
        if n.payoffs is not None:
            return
        if n.player not in players:
            players.append(n.player)
        out.append((path, n))
        for i, c in enumerate(n.children):
            walk(c, path + (i + 1,))
    walk(root, ())
    return out, players


def _is_chance(n):
    return n.children and n.children[0].prob is not None


def _tree_checks(root, fname):
    nodes, players = _tree_nodes(root)
    infos = {}
    for path, n in nodes:
        if n.infoset is not None:
            infos.setdefault((n.player, n.infoset), []).append(path)
    # like Wolfram, information sets are ignored: every action node is solved as its own subgame
    return nodes, players


def _player_index(players, root):
    # payoffs are listed per player in order of first appearance of the non-chance players
    return {p: i for i, p in enumerate([q for q in players if not _chance_player(q, root)])}


def _chance_player(p, root):
    nodes, _ = _tree_nodes(root)
    return all(_is_chance(n) for _, n in nodes if n.player == p)


def _ksp(v):
    return K.to_sp(v) if not isinstance(v, sp.Basic) else v


@fn("FindTreeGameStrategies", **ADDED,
    notes="pure-strategy solutions of a TreeGame by backward induction: at each action node every action "
          "that is optimal for the moving player under some tie-breaking below it is kept, and all combinations "
          "are listed (this reproduces Wolfram's results, including its handling of ties), each as <|player -> <|node -> probabilities|>|> with "
          "nodes as child-index paths ({} the root), players in order of appearance (chance players get <||>), "
          "sorted; payoff vectors are matched to the non-chance players in order of appearance; as in Wolfram, "
          "information sets are ignored (each node solved separately)")
def _findtreegame(game, *rest):
    fname = "FindTreeGameStrategies"
    if rest:
        raise ValueError(f"{fname} called with too many arguments")
    from .wl_trees import _game_root
    root = _game_root(game, fname)
    nodes, players = _tree_checks(root, fname)
    pidx = _player_index(players, root)

    acc = {}

    def solve(n, path):
        """Set of payoff vectors reachable from n under some tie-breaking; records each node's acceptable actions."""
        if n.payoffs is not None:
            return {tuple(_ksp(v) for v in n.payoffs)}
        subs = [solve(c, path + (i + 1,)) for i, c in enumerate(n.children)]
        if _is_chance(n):
            out = set()
            for combo in itertools.product(*subs):
                out.add(tuple(sp.nsimplify(0) + sum(_ksp(c.prob) * v[k] for c, v in zip(n.children, combo))
                              for k in range(len(combo[0]))))
                if len(out) > 10 ** 4:
                    raise NotImplementedError(f"{fname}: too many tie-breaking alternatives")
            return out
        if n.player not in pidx:
            raise ValueError(f"{fname}: player {n.player} has no payoff component")
        k = pidx[n.player]
        f = lambda v: float(sp.N(v[k]))
        lows = [min(f(v) for v in V) for V in subs]
        ok, out = [], set()
        for a, V in enumerate(subs):
            floor_ = max([lows[b] for b in range(len(subs)) if b != a], default=-math.inf)
            good = [v for v in V if f(v) >= floor_ - 1e-12]
            if good:
                ok.append(a)
                out.update(good)
        acc[path] = ok
        return out
    solve(root, ())
    keys = [path for path, n in nodes if path in acc]
    total = 1
    for kk in keys:
        total *= len(acc[kk])
    if total > 10 ** 5:
        raise NotImplementedError(f"{fname}: too many equilibria ({total})")
    results = [dict(zip(keys, combo)) for combo in itertools.product(*(acc[kk] for kk in keys))]

    def as_assoc(choices):
        per = []
        for p in players:
            entries = []
            for path, n in nodes:
                if n.player == p and not _is_chance(n):
                    vec = [int(i == choices[path]) for i in range(len(n.children))]
                    entries.append((path, vec))
            per.append((p, entries))
        return per

    order = sorted(keys, key=lambda pth: (len(pth), pth))      # Wolfram sorts by node in canonical order
    results.sort(key=lambda c: tuple(-c[pth] for pth in order))
    profs = [as_assoc(c) for c in results]
    return L(*(Expr("Association", *(Expr("Rule", p, Expr("Association", *(Expr("Rule", L(*path), L(*vec))
                                                                           for path, vec in ent)))
                                     for p, ent in pr)) for pr in profs))


@fn("VerifyTreeGameStrategy", **ADDED,
    notes="True when the strategy profile <|player -> <|node -> probabilities|>|> is a subgame-perfect equilibrium "
          "of a perfect-information TreeGame (at every action node no action gives the moving player a higher "
          "expected payoff than the strategy, given the strategy below it; a 10^-9 tolerance for machine numbers); "
          "every non-chance action node must be specified; information sets are ignored; symbolic strategies raise")
def _verifytreegame(game, strat, *rest):
    fname = "VerifyTreeGameStrategy"
    if rest:
        raise ValueError(f"{fname} called with too many arguments")
    from .wl_trees import _game_root, _strategy_table
    root = _game_root(game, fname)
    nodes, players = _tree_checks(root, fname)
    pidx = _player_index(players, root)
    table = _strategy_table(strat, fname)
    ok = [True]

    def value(n, path):
        if n.payoffs is not None:
            return [_ksp(v) for v in n.payoffs]
        vals = [value(c, path + (i + 1,)) for i, c in enumerate(n.children)]
        if _is_chance(n) and path not in table:
            ps = [_ksp(c.prob) for c in n.children]
        else:
            if path not in table:
                raise ValueError(f"{fname}: the strategy does not specify node {list(path)}")
            ps = [_ksp(v) for v in table[path]]
            if len(ps) != len(n.children):
                raise ValueError(f"{fname}: node {list(path)} has {len(n.children)} actions")
        res = [sum(p * v[k] for p, v in zip(ps, vals)) for k in range(len(vals[0]))]
        if not _is_chance(n):
            k = pidx[n.player]
            for v in vals:
                d = v[k] - res[k]
                if not d.is_number:
                    raise NotImplementedError(f"{fname}: symbolic strategies are not supported")
                if float(sp.N(d)) > 1e-9:
                    ok[0] = False
        return res
    value(root, ())
    return ok[0]


# ═══════════════════════════════ Turing machines ═══════════════════════════════

def _tm_rule(rule, fname):
    """(lookup(s, a) -> (s', a', off) | None, s, k)."""
    spec = list(rule.args) if is_list(rule) and rule.args and not is_rule(rule.args[0]) else [rule]
    n = spec[0]
    if is_list(n) or (isinstance(n, Expr) and is_rule(n)) or (is_list(rule) and rule.args and is_rule(rule.args[0])):
        rules = rule if (is_list(rule) and rule.args and is_rule(rule.args[0])) else n

        def look(s, a):
            r = ev(Expr("Replace", L(s, a), rules))
            if r == L(s, a):
                return None
            if not (is_list(r) and len(r.args) == 3):
                raise ValueError(f"{fname}: a rule must give {{state, value, offset}}")
            return r.args[0], r.args[1], int_arg(r.args[2], fname)
        return look
    num = int_arg(n, fname)
    s = int_arg(spec[1], fname) if len(spec) > 1 else 2
    k = int_arg(spec[2], fname) if len(spec) > 2 else 2
    if len(spec) > 3 and not (isinstance(spec[3], int) and spec[3] == 1):
        raise NotImplementedError(f"{fname}: offset ranges other than 1 and multidimensional machines are not "
                                  "supported")
    sk = s * k
    if num < 0 or num >= (2 * sk) ** sk:
        raise ValueError(f"{fname}: rule number {num} is out of range for {s} states and {k} colours")
    digits = []
    x = num
    for _ in range(sk):
        digits.append(x % (2 * sk))
        x //= 2 * sk
    digits.reverse()
    table = {}
    for i, d in enumerate(digits):
        st, col = 1 + i // k, k - 1 - i % k
        table[(st, col)] = (1 + d // (2 * k), (d // 2) % k, 2 * (d % 2) - 1)

    def look(st, a):
        key = (int(st) if isinstance(st, int) else st, int(a) if isinstance(a, int) else a)
        return table.get(key)
    return look


def _tm_parse_init(init, fname):
    if not (is_list(init) and len(init.args) == 2):
        raise ValueError(f"{fname}: the initial condition must be {{state, tape}}")
    head, tape = init.args
    if is_list(head):
        s = head.args[0]
        x = int_arg(head.args[1], fname) if len(head.args) > 1 else 1
        dx = int_arg(head.args[2], fname) if len(head.args) > 2 else 0
        if len(head.args) > 1 and is_list(head.args[1]):
            raise NotImplementedError(f"{fname}: multidimensional machines are not supported")
    else:
        s, x, dx = head, 1, 0
    if is_list(tape) and len(tape.args) == 2 and is_list(tape.args[0]) and not is_list(tape.args[1]):
        cells = list(tape.args[0].args)
        if any(is_list(c) for c in cells):
            raise NotImplementedError(f"{fname}: multidimensional machines are not supported")
        return s, x, dx, ("inf", cells, tape.args[1])
    if is_list(tape) and len(tape.args) == 2 and is_list(tape.args[0]) and is_list(tape.args[1]):
        raise NotImplementedError(f"{fname}: repetitive backgrounds are not supported")
    if is_list(tape) and not any(is_list(c) for c in tape.args):
        if not tape.args:
            raise ValueError(f"{fname}: a finite tape must not be empty")
        return s, x, dx, ("fin", list(tape.args), None)
    raise ValueError(f"{fname}: invalid tape {tape}")


def _tm_run(look, init, steps, fname):
    s, x, dx, (kind, cells, bg) = _tm_parse_init(init, fname)
    hist = []
    if kind == "fin":
        tape = list(cells)
        n = len(tape)
        if not 1 <= x <= n:
            raise ValueError(f"{fname}: the head position {x} is outside the tape")
        pos = x - 1
        hist.append((s, pos, dx, list(tape)))
        for _ in range(steps):
            r = look(s, tape[pos])
            if r is not None:
                s, tape[pos], off = r
                pos = (pos + off) % n
                dx += off
            hist.append((s, pos, dx, list(tape)))
        return kind, hist, None
    tape = {i - (x - 1): v for i, v in enumerate(cells)}
    off0 = 0
    region = set(tape) | {0}
    head = off0
    hist.append((s, head, dx, dict(tape)))
    for _ in range(steps):
        a = tape.get(head, bg)
        r = look(s, a)
        if r is not None:
            s, tape[head], off = r
            head += off
            dx += off
        region.add(head)
        hist.append((s, head, dx, dict(tape)))
    return kind, hist, (bg, region)


def _tm_format(kind, hist, extra, region_override=None):
    out = []
    if kind == "fin":
        for s, pos, dx, tape in hist:
            out.append(L(L(s, pos + 1, dx), L(*tape)))
        return out
    bg, region = extra
    lo, hi = min(region), max(region)
    for s, head, dx, tape in hist:
        out.append(L(L(s, head - lo + 1, dx), L(*(tape.get(i, bg) for i in range(lo, hi + 1)))))
    return out


@fn("TuringMachine", **ADDED,
    notes="one-dimensional Turing machines: rule n, {n, s} or {n, s, k} (Wolfram numbering) or explicit rules "
          "{s, a} -> {s', a', offset} (patterns and conditions allowed; no matching rule leaves the configuration "
          "unchanged); initial conditions {s, {{a1, ...}, b}} (infinite tape with background b), {{s, x}, ...}, "
          "{{s, x, dx}, ...} and finite cyclic tapes {s, {a1, ...}}; TuringMachine[rule, init, t] gives the t + 1 "
          "steps {{s, x, dx}, tape} (infinite tapes padded to the region visited), TuringMachine[rule, init] and "
          "TuringMachine[rule][init] one step; multidimensional machines, offset ranges and repetitive "
          "backgrounds raise")
def _turingmachine(rule, *rest):
    fname = "TuringMachine"
    if not rest:
        _tm_rule(rule, fname)
        raise Unevaluated()
    init = rest[0]
    look = _tm_rule(rule, fname)
    if len(rest) == 1:
        kind, hist, extra = _tm_run(look, init, 1, fname)
        if kind == "fin":
            return _tm_format(kind, hist[1:], extra)[0]
        s, head, dx, tape = hist[1]
        _, _, _, (_, cells, bg) = _tm_parse_init(init, fname)
        s0, h0, _, t0 = hist[0]
        keys = set(t0) | set(tape) | {h0, head}
        lo, hi = min(keys), max(keys)
        return L(L(s, head - lo + 1, dx), L(L(*(tape.get(i, bg) for i in range(lo, hi + 1))), bg))
    if len(rest) > 2:
        raise ValueError(f"{fname} called with too many arguments")
    t = rest[1]
    if is_list(t):
        raise NotImplementedError(f"{fname}: step specifications other than an integer t are not supported")
    steps = nonneg_int(t, fname, "number of steps") if True else 0
    kind, hist, extra = _tm_run(look, init, steps, fname)
    return L(*_tm_format(kind, hist, extra))


def _tm_call(head, args, ev_):
    if len(head.args) == 1 and len(args) == 1:
        return ev_(Expr("TuringMachine", head.args[0], args[0]))
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["TuringMachine"] = _tm_call


# ── RulePlot ──

def _gray(v, k):
    return Expr("GrayLevel", R(sp.Rational(k - 1 - v, k - 1)) if k > 1 else 1)


def _cell(x, y, color):
    return L(color, Expr("EdgeForm", Expr("GrayLevel", 0.5)), Expr("Rectangle", L(x, y), L(x + 1, y + 1)))


@fn("RulePlot", **ADDED, hold=True,
    notes="RulePlot[CellularAutomaton[n | {n, k} | {n, k, r}]] gives Graphics of the rule icons (each neighbourhood, "
          "in decreasing order, above the new centre cell; colour v drawn as GrayLevel[1 - v/(k-1)]) and "
          "RulePlot[TuringMachine[n | {n, s, k}]] the icons of each {state, colour} case (head state as a number above "
          "the cell, new colour, state and move below); other systems (and 2D rules) raise; the drawing style differs "
          "from Wolfram's")
def _ruleplot(system, *rest):
    fname = "RulePlot"
    gfx = [ev(r) for r in rest if is_rule(r)]
    if not isinstance(system, Expr) or system.head not in ("CellularAutomaton", "TuringMachine") or \
            len(system.args) != 1:
        raise NotImplementedError(f"{fname}: only CellularAutomaton[rule] and TuringMachine[rule] are supported")
    spec = ev(system.args[0])
    prims = []
    if system.head == "CellularAutomaton":
        parts = list(spec.args) if is_list(spec) else [spec]
        if any(is_list(p) or is_assoc(p) for p in parts[1:]) or is_assoc(spec):
            raise NotImplementedError(f"{fname}: only 1D numbered cellular automaton rules are supported")
        num = int_arg(parts[0], fname)
        k = int_arg(parts[1], fname) if len(parts) > 1 else 2
        r = parts[2] if len(parts) > 2 else 1
        r = int(S(r) * 2) / 2 if not isinstance(r, int) else r
        if not float(r).is_integer():
            raise NotImplementedError(f"{fname}: half-integer ranges are not supported")
        r = int(r)
        w = 2 * r + 1
        ncase = k ** w
        if num < 0 or num >= k ** ncase:
            raise ValueError(f"{fname}: rule number out of range")
        outs = []
        x = num
        for _ in range(ncase):
            outs.append(x % k)
            x //= k
        for idx, case in enumerate(range(ncase - 1, -1, -1)):
            cells = []
            c = case
            for _ in range(w):
                cells.append(c % k)
                c //= k
            cells.reverse()
            x0 = idx * (w + 1)
            for j, v in enumerate(cells):
                prims.append(_cell(x0 + j, 1, _gray(v, k)))
            prims.append(_cell(x0 + r, 0, _gray(outs[case], k)))
        return Expr("Graphics", L(*prims), *gfx)
    parts = list(spec.args) if is_list(spec) else [spec]
    num = int_arg(parts[0], fname)
    s = int_arg(parts[1], fname) if len(parts) > 1 else 2
    k = int_arg(parts[2], fname) if len(parts) > 2 else 2
    look = _tm_rule(spec if is_list(spec) else L(num, s, k), fname)
    idx = 0
    for st in range(1, s + 1):
        for col in range(k - 1, -1, -1):
            ns, nc, off = look(st, col)
            x0 = idx * 4
            prims.append(_cell(x0 + 1, 1, _gray(col, k)))
            prims.append(Expr("Text", st, L(R(x0 + sp.Rational(3, 2)), R(sp.Rational(5, 2)))))
            prims.append(_cell(x0 + 1, 0, _gray(nc, k)))
            prims.append(Expr("Text", ns, L(R(x0 + sp.Rational(3, 2) + off), R(sp.Rational(-1, 2)))))
            prims.append(Expr("Arrow", L(L(R(x0 + sp.Rational(3, 2)), R(sp.Rational(-1, 5))),
                                         L(R(x0 + sp.Rational(3, 2) + off), R(sp.Rational(-1, 5))))))
            idx += 1
    return Expr("Graphics", L(*prims), *gfx)


K.BUILTIN_ATTRIBUTES["RulePlot"] = {"HoldAll", "Protected"}   # CellularAutomaton[rule] alone does not evaluate here


# ═══════════════════════════════ option names ═══════════════════════════════

record("Normalized", AREA, "added",
       "option of OrthogonalMatrixQ and UnitaryMatrixQ (Normalized -> False only requires orthogonal columns/rows)")
record("NormFunction", AREA, "added",
       "option name (Wolfram: the norm minimised by FindFit/Fit); MikoshiLang's FindFit and Fit do not accept it and "
       "raise an unknown-option error")
record("FitRegularization", AREA, "added",
       "option name (Wolfram: {\"Tikhonov\", l}, {\"LASSO\", l}, ... regularisation for Fit/FindFit); MikoshiLang's "
       "Fit and FindFit do not accept it and raise an unknown-option error")
record("Regularization", AREA, "added",
       "option name (Wolfram: regularisation of divergent sums, \"Abel\", \"Borel\", \"Cesaro\", \"Dirichlet\", "
       "\"Euler\"); Sum raises when it is given")
record("LossFunction", AREA, "added",
       "option name (Wolfram: the loss for NetTrain/Predict/Classify); MikoshiLang's machine-learning functions do not "
       "accept it")
record("UnconstrainedParameters", AREA, "added",
       "option name (Wolfram: parameters left free in GeometricScene); GeometricScene does not accept it here")
