"""Combinatorics, sequences and set-like list functions."""
import itertools
import math

import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, pos_int, list_arg, has_float, is_numeric_value)

AREA = "combinatorics"


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


def _binom_int(n, k):
    if n >= 0:
        return math.comb(n, k) if 0 <= k <= n else 0
    if k >= 0:
        return (-1) ** k * math.comb(k - n - 1, k)
    if k <= n:
        return (-1) ** (n - k) * math.comb(-k - 1, n - k)
    return 0


@fn("Binomial", area=AREA, status="fixed", listable=True,
    notes="legacy truncated arguments with int() (Binomial[5/2, 1] = 2) and failed on symbols; now exact/symbolic "
          "with Wolfram's conventions for negative integers (Binomial[-1, -1] = 1)")
def _binomial(n, k):
    if _intq(n) and _intq(k):
        return _binom_int(int(n), int(k))
    sn, sk = sym(n, "Binomial"), sym(k, "Binomial")
    return numeric_out(sp.binomial(sn, sk), n, k)


@fn("Factorial", area=AREA, status="fixed", listable=True,
    notes="non-integers use Gamma[n + 1] (Factorial[1/2] = Sqrt[Pi]/2); negative integers give ComplexInfinity; "
          "symbols stay unevaluated; legacy truncated with int()")
def _factorial(n):
    if _intq(n):
        k = int(n)
        return R(sp.zoo if k < 0 else sp.factorial(k))
    if not is_numeric_value(n):
        raise Unevaluated()
    return numeric_out(sp.gamma(S(n) + 1), n)


def _fact2(n, fname):
    if _intq(n):
        k = int(n)
        if k >= -1:
            return R(sp.factorial2(k))
        if k % 2 == 0:
            return R(sp.zoo)
        # (-2m-1)!! = (-1)^m / (2m-1)!!
        m = (-k - 1) // 2
        return R(sp.Integer(-1) ** m / sp.factorial2(2 * m - 1))
    if not is_numeric_value(n):
        raise Unevaluated()
    raise NotImplementedError(f"{fname}: non-integer arguments are not supported")


@fn("DoubleFactorial", area=AREA, status="fixed", listable=True,
    notes="negative odd integers supported ((-5)!! = 1/3); legacy truncated reals")
def _doublefact(n):
    return _fact2(n, "DoubleFactorial")


@fn("Factorial2", area=AREA, status="promoted", listable=True, notes="Wolfram name for the double factorial")
def _factorial2(n):
    return _fact2(n, "Factorial2")


@fn("HarmonicNumber", area=AREA, status="fixed", listable=True,
    notes="HarmonicNumber[n] and HarmonicNumber[n, r]; exact; symbolic n stays symbolic; legacy truncated reals")
def _harmonic(n, r=1):
    if not is_numeric_value(n):
        raise Unevaluated()
    sn, sr = S(n), sym(r, "HarmonicNumber")
    if sn.is_Integer and sn < 0:
        raise ValueError("HarmonicNumber: negative integers are poles")
    return numeric_out(sp.harmonic(sn, sr), n, r)


def _subfact(n, fname):
    return R(sp.subfactorial(nonneg_int(n, fname)))


@fn("Derangements", area=AREA, status="verified", listable=True,
    notes="number of derangements !n (MikoshiLang; same as Subfactorial)")
def _derangements(n):
    return _subfact(n, "Derangements")


@fn("Subfactorial", area=AREA, status="promoted", listable=True, notes="!n")
def _subfactorial(n):
    return _subfact(n, "Subfactorial")


def _partitions_p(n, fname):
    from sympy.functions.combinatorial.numbers import partition
    k = int_arg(n, fname)
    return 0 if k < 0 else int(partition(k))


@fn("PartitionCount", area=AREA, status="verified", listable=True, notes="MikoshiLang alias of PartitionsP")
def _partcount(n):
    return _partitions_p(n, "PartitionCount")


@fn("PartitionsP", area=AREA, status="promoted", listable=True, notes="number of unrestricted partitions")
def _partitionsp(n):
    return _partitions_p(n, "PartitionsP")


@fn("PartitionsQ", area=AREA, status="promoted", listable=True, notes="number of partitions into distinct parts")
def _partitionsq(n):
    k = int_arg(n, "PartitionsQ")
    if k < 0:
        return 0
    q = [1] + [0] * k
    for part in range(1, k + 1):
        for s in range(k, part - 1, -1):
            q[s] += q[s - part]
    return q[k]


def _partitions(n, maxpart=None, lo=0, hi=None, allowed=None):
    """Partitions of n in Wolfram order (reverse lexicographic)."""
    out = []

    def rec(rem, maxp, acc):
        if rem == 0:
            if lo <= len(acc) and (hi is None or len(acc) <= hi):
                out.append(list(acc))
            return
        if hi is not None and len(acc) >= hi:
            return
        for p in range(min(rem, maxp), 0, -1):
            if allowed is not None and p not in allowed:
                continue
            acc.append(p)
            rec(rem - p, p, acc)
            acc.pop()

    if n == 0:
        return [[]] if lo == 0 else []
    rec(n, n if maxpart is None else maxpart, [])
    return out


def _kspec(spec, fname):
    if _intq(spec):
        return 0, int(spec)
    if is_list(spec) and len(spec.args) == 1:
        k = nonneg_int(spec.args[0], fname, "length")
        return k, k
    if is_list(spec) and len(spec.args) == 2:
        return nonneg_int(spec.args[0], fname, "length"), nonneg_int(spec.args[1], fname, "length")
    if isinstance(spec, Symbol) and spec.name in ("All", "Infinity"):
        return 0, None
    raise ValueError(f"{fname}: invalid length specification")


@fn("IntegerPartitions", area=AREA, status="fixed",
    notes="legacy returned the number of partitions; Wolfram IntegerPartitions lists them in reverse "
          "lexicographic order; supports [n], [n, k], [n, {k}], [n, {kmin, kmax}], [n, kspec, {s1, ...}]")
def _intpartitions(n, kspec=None, elems=None):
    k = int_arg(n, "IntegerPartitions")
    if k < 0:
        return L()
    lo, hi = (0, None) if kspec is None else _kspec(kspec, "IntegerPartitions")
    allowed = None
    if elems is not None:
        allowed = {pos_int(e, "IntegerPartitions", "part") for e in list_arg(elems, "IntegerPartitions")}
    return L(*(L(*p) for p in _partitions(k, lo=lo, hi=hi, allowed=allowed)))


@fn("PartitionsList", area=AREA, status="fixed",
    notes="legacy reused one mutable dict (every entry was the last partition); now the partitions of n "
          "in Wolfram IntegerPartitions order")
def _partitionslist(n):
    k = nonneg_int(n, "PartitionsList")
    return L(*(L(*p) for p in _partitions(k)))


@fn("Compositions", area=AREA, status="fixed",
    notes="legacy returned k-subsets of {1..n-1}; now Compositions[n, k] lists all ordered k-tuples of "
          "non-negative integers summing to n (Combinatorica convention), lexicographic")
def _compositions(n, k):
    nn, kk = nonneg_int(n, "Compositions"), nonneg_int(k, "Compositions", "number of parts")
    if kk == 0:
        return L(L()) if nn == 0 else L()
    if math.comb(nn + kk - 1, kk - 1) > 100000:
        raise ValueError("Compositions: too many compositions to list")
    out = []
    for bars in itertools.combinations(range(nn + kk - 1), kk - 1):
        parts, prev = [], -1
        for b in bars + (nn + kk - 1,):
            parts.append(b - prev - 1)
            prev = b
        out.append(L(*parts))
    return L(*out)


@fn("Multinomial", area=AREA, status="fixed",
    notes="legacy Multinomial[n, {k...}] was not Wolfram's form; now Multinomial[n1, n2, ...] = "
          "(n1 + n2 + ...)!/(n1! n2! ...), exact for any exact arguments")
def _multinomial(*args):
    if any(is_list(a) for a in args):
        raise ValueError("Multinomial takes the counts as separate arguments: Multinomial[n1, n2, ...]")
    if not all(is_numeric_value(a) for a in args):
        raise Unevaluated()
    vals = [S(a) for a in args]
    if all(v.is_Integer and v >= 0 for v in vals):
        out = math.factorial(int(sum(vals)))
        for v in vals:
            out //= math.factorial(int(v))
        return out
    res = sp.gamma(sum(vals) + 1)
    for v in vals:
        res = res / sp.gamma(v + 1)
    return numeric_out(sp.simplify(res), *args)


def _perm_list(items, lo, hi):
    out = []
    seen = set()
    for r in range(lo, (len(items) if hi is None else min(hi, len(items))) + 1):
        for p in itertools.permutations(range(len(items)), r):
            key = tuple(str(items[i]) for i in p)
            if key in seen:
                continue
            seen.add(key)
            out.append(L(*(items[i] for i in p)))
    return out


@fn("Permutations", area=AREA, status="fixed",
    notes="legacy Permutations[n, k] returned the count n!/(n-k)!; Wolfram lists permutations of a list "
          "(duplicates removed); Permutations[list, n] and [list, {n}] supported; integer arguments raise")
def _permutations(lst, spec=None):
    if not is_list(lst):
        if _intq(lst):
            raise ValueError("Permutations takes a list; for the number of k-permutations of n use "
                             "Factorial[n]/Factorial[n - k]")
        raise Unevaluated()
    items = list(lst.args)
    lo, hi = (len(items), len(items)) if spec is None else _kspec(spec, "Permutations")
    if math.factorial(len(items)) > 200000 and (hi is None or hi >= 8):
        raise ValueError("Permutations: too many permutations to list")
    return L(*_perm_list(items, lo, hi))


def _subsets(items, lo, hi):
    out = []
    top = len(items) if hi is None else min(hi, len(items))
    for r in range(lo, top + 1):
        for c in itertools.combinations(items, r):
            out.append(L(*c))
    return out


@fn("Subsets", area=AREA, status="fixed",
    notes="Subsets[list], [list, n], [list, {n}], [list, {min, max}] in Wolfram order (by size, then position)")
def _subsets_fn(lst, spec=None):
    items = list_arg(lst, "Subsets")
    lo, hi = (0, None) if spec is None else _kspec(spec, "Subsets")
    if 2 ** len(items) > 1 << 20 and spec is None:
        raise ValueError("Subsets: too many subsets to list")
    return L(*_subsets(items, lo, hi))


@fn("PowerSet", area=AREA, status="verified", listable=False, notes="MikoshiLang alias of Subsets[list]")
def _powerset(lst):
    items = list_arg(lst, "PowerSet")
    if len(items) > 20:
        raise ValueError("PowerSet: too many subsets to list")
    return L(*_subsets(items, 0, None))


@fn("CartesianProduct", area=AREA, status="verified",
    notes="pairs {a, b} for a in the first list and b in the second (= Tuples[{l1, l2}]); now variadic")
def _cartesian(*lists):
    if not lists:
        raise ValueError("CartesianProduct needs at least one list")
    seqs = [list_arg(l, "CartesianProduct") for l in lists]
    return L(*(L(*t) for t in itertools.product(*seqs)))


@fn("Tuples", area=AREA, status="promoted", notes="Tuples[list, n] and Tuples[{l1, l2, ...}]")
def _tuples(lst, n=None):
    if n is None:
        seqs = [list_arg(l, "Tuples") for l in list_arg(lst, "Tuples")]
    else:
        k = nonneg_int(n, "Tuples", "length")
        seqs = [list_arg(lst, "Tuples")] * k
    return L(*(L(*t) for t in itertools.product(*seqs)))


@fn("HookLengthFormula", area=AREA, status="fixed",
    notes="legacy subtracted the row index from each hook (wrong counts / ZeroDivisionError); now the number of "
          "standard Young tableaux of the shape")
def _hook(shape):
    parts = [pos_int(p, "HookLengthFormula", "part") for p in list_arg(shape, "HookLengthFormula")]
    if any(parts[i] < parts[i + 1] for i in range(len(parts) - 1)):
        raise ValueError("HookLengthFormula: shape must be a weakly decreasing list of positive integers")
    n = sum(parts)
    prod = 1
    for i, row in enumerate(parts):
        for j in range(row):
            arm = row - j - 1
            leg = sum(1 for r in parts[i + 1:] if r > j)
            prod *= arm + leg + 1
    return math.factorial(n) // prod


@fn("ArithmeticSequence", area=AREA, status="fixed",
    notes="first n terms a, a + d, ...; exact for exact input (legacy converted to floats and rounded to 10 digits)")
def _arith(a, d, n):
    k = nonneg_int(n, "ArithmeticSequence", "length")
    sa, sd = sym(a, "ArithmeticSequence"), sym(d, "ArithmeticSequence")
    return L(*(numeric_out(sa + i * sd, a, d) for i in range(k)))


@fn("GeometricSequence", area=AREA, status="fixed",
    notes="first n terms a, a r, ...; exact for exact input (legacy converted to floats and rounded)")
def _geom(a, r, n):
    k = nonneg_int(n, "GeometricSequence", "length")
    sa, sr = sym(a, "GeometricSequence"), sym(r, "GeometricSequence")
    return L(*(numeric_out(sp.expand(sa * sr ** i), a, r) for i in range(k)))


@fn("FibonacciSequence", area=AREA, status="verified", notes="first n Fibonacci numbers F0, F1, ... (MikoshiLang)")
def _fibseq(n):
    k = nonneg_int(n, "FibonacciSequence", "length")
    a, b, out = 0, 1, []
    for _ in range(k):
        out.append(a)
        a, b = b, a + b
    return L(*out)


record("YoungTableaux", AREA, "quarantined",
       "legacy built a single StandardTableau and returned its size (not the tableaux, not their count); "
       "quarantined: use HookLengthFormula[shape] for the number of standard tableaux")
