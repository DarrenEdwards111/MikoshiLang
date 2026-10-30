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
    if (isinstance(n, float) or isinstance(k, float)) and sn.is_real and sk.is_real and sn.is_number \
            and sk.is_number and not any(isinstance(v, sp.Float) and v._prec > 53 for v in (sn, sk)):
        # batch 6: machine-precision arguments are evaluated with guard digits (Binomial[10, 3.] = 120.,
        # previously 120.00000000000001)
        import mpmath
        with mpmath.workprec(113):
            v = mpmath.binomial(mpmath.mpf(sp.Float(sn, 30)), mpmath.mpf(sp.Float(sk, 30)))
        if isinstance(v, mpmath.mpf) and mpmath.isfinite(v):
            return float(v)
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
          "lexicographic order; supports [n], [n, k], [n, {k}], [n, {kmin, kmax}], [n, kspec, {s1, ...} or All] and "
          "[n, kspec, sspec, m] (first m partitions, batch 6)")
def _intpartitions(n, kspec=None, elems=None, count=None):
    k = int_arg(n, "IntegerPartitions")
    if k < 0:
        return L()
    lo, hi = (0, None) if kspec is None else _kspec(kspec, "IntegerPartitions")
    allowed = None
    if elems is not None and not (isinstance(elems, Symbol) and elems.name == "All"):
        allowed = {pos_int(e, "IntegerPartitions", "part") for e in list_arg(elems, "IntegerPartitions")}
    parts = _partitions(k, lo=lo, hi=hi, allowed=allowed)
    if count is not None:
        # batch 6: IntegerPartitions[n, kspec, sspec, m] gives at most m partitions (the first m)
        m = nonneg_int(count, "IntegerPartitions", "number of partitions")
        parts = parts[:m]
    return L(*(L(*p) for p in parts))


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


def _subset_sizes(spec, n, fname):
    """Sizes requested by a Subsets length spec (batch 6 adds {min, max, step})."""
    if is_list(spec) and len(spec.args) == 3:
        a = nonneg_int(spec.args[0], fname, "length")
        b = nonneg_int(spec.args[1], fname, "length")
        d = int_arg(spec.args[2], fname, "step")
        if d == 0:
            raise ValueError(f"{fname}: the length step must be non-zero")
        return [r for r in range(a, b + (1 if d > 0 else -1), d) if r <= n]
    lo, hi = (0, None) if spec is None else _kspec(spec, fname)
    return list(range(lo, (n if hi is None else min(hi, n)) + 1))


def _unrank_subset(items, sizes, r):
    """The r-th (0-based) subset in Wolfram order for the given sizes, or None when out of range."""
    n = len(items)
    for size in sizes:
        c = math.comb(n, size)
        if r < c:
            out, start = [], 0
            for left in range(size, 0, -1):
                for i in range(start, n):
                    cnt = math.comb(n - i - 1, left - 1)
                    if r < cnt:
                        out.append(items[i])
                        start = i + 1
                        break
                    r -= cnt
            return out
        r -= c
    return None


@fn("Subsets", area=AREA, status="fixed",
    notes="Subsets[list], [list, n], [list, {n}], [list, {min, max}], [list, {min, max, step}] in Wolfram order "
          "(by size, then position); batch 6 adds Subsets[list, nspec, s] (first s), [list, nspec, {s}] (the s-th, "
          "computed by unranking) and [list, nspec, {m, n(, step)}], and any head (Subsets[f[a, b]])")
def _subsets_fn(lst, spec=None, take=None):
    head = "List"
    if is_list(lst):
        items = list(lst.args)
    elif isinstance(lst, Expr) and not isinstance(lst, Symbol) and isinstance(lst.head, str) and lst.args:
        head, items = lst.head, list(lst.args)
    else:
        items = list_arg(lst, "Subsets")
    if spec is not None and isinstance(spec, Symbol) and spec.name == "All":
        spec = None
    sizes = _subset_sizes(spec, len(items), "Subsets")
    total = sum(math.comb(len(items), r) for r in sizes)
    mk = lambda c: Expr(head, *c)
    if take is None:
        if total > 1 << 20:
            raise ValueError("Subsets: too many subsets to list")
        return L(*(mk(c) for r in sizes for c in itertools.combinations(items, r)))
    if _intq(take):
        k = int(take)
        idx = list(range(min(k, total))) if k >= 0 else list(range(max(total + k, 0), total))
    elif is_list(take) and len(take.args) in (1, 2, 3) and all(_intq(v) for v in take.args):
        vals = [int(v) for v in take.args]
        if len(vals) == 1:
            i = vals[0]
            i = i - 1 if i > 0 else total + i
            if i < 0 or i >= total or vals[0] == 0:
                raise ValueError(f"Subsets: subset {vals[0]} does not exist (there are {total})")
            return mk(_unrank_subset(items, sizes, i))
        a, b = vals[0], vals[1]
        step = vals[2] if len(vals) == 3 else 1
        if step == 0 or a == 0 or b == 0:
            raise ValueError("Subsets: invalid subset range")
        a = a - 1 if a > 0 else total + a
        b = b - 1 if b > 0 else total + b
        idx = [i for i in range(a, b + (1 if step > 0 else -1), step) if 0 <= i < total]
    else:
        raise ValueError(f"Subsets: invalid subset specification {take!r}")
    if len(idx) > 1 << 20:
        raise ValueError("Subsets: too many subsets to list")
    return L(*(mk(_unrank_subset(items, sizes, i)) for i in idx))


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


def _reshape(flat, dims):
    if len(dims) == 1:
        return L(*flat)
    step = len(flat) // dims[0]
    return L(*(_reshape(flat[i * step:(i + 1) * step], dims[1:]) for i in range(dims[0])))


@fn("Tuples", area=AREA, status="promoted",
    notes="Tuples[list, n] and Tuples[{l1, l2, ...}]; batch 6 adds Tuples[list, {n1, n2, ...}] (all n1 x n2 x ... "
          "arrays) and any head (Tuples[f[a, b], 2] = {f[a, a], f[a, b], f[b, a], f[b, b]})")
def _tuples(lst, n=None):
    head = "List"
    if n is None:
        seqs = [list_arg(l, "Tuples") for l in list_arg(lst, "Tuples")]
        dims = None
    else:
        if is_list(lst):
            items = list(lst.args)
        elif isinstance(lst, Expr) and not isinstance(lst, Symbol) and isinstance(lst.head, str) and lst.args:
            head, items = lst.head, list(lst.args)
        else:
            items = list_arg(lst, "Tuples")
        if is_list(n):
            dims = [nonneg_int(d, "Tuples", "dimension") for d in n.args]
            if not dims:
                raise ValueError("Tuples: the dimension list must not be empty")
            k = math.prod(dims)
            head = "List"
        else:
            dims = None
            k = nonneg_int(n, "Tuples", "length")
        seqs = [items] * k
    total = math.prod(len(q) for q in seqs)
    if total > 1 << 20:
        raise ValueError("Tuples: too many tuples to list")
    if dims is not None:
        return L(*(_reshape(list(t), dims) for t in itertools.product(*seqs)))
    return L(*(Expr(head, *t) for t in itertools.product(*seqs)))


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


_SYT_LIMIT = 100000


def _syt_partitions(n):
    def rec(n, mx):
        if n == 0:
            yield []
            return
        for k in range(min(n, mx), 0, -1):
            for rest in rec(n - k, k):
                yield [k] + rest
    return list(rec(n, n))


def _syt(shape):
    """All standard Young tableaux of the given shape, in lexicographic order of their row words."""
    n = sum(shape)
    rows = [[] for _ in shape]
    out = []

    def place(k):
        if k > n:
            out.append([list(r) for r in rows])
            return
        for i, r in enumerate(rows):
            if len(r) < shape[i] and (i == 0 or len(rows[i - 1]) > len(r)):
                r.append(k)
                place(k + 1)
                r.pop()
    place(1)
    out.sort(key=lambda t: [x for r in t for x in r])
    return out


@fn("YoungTableaux", area=AREA, status="promoted",
    notes="YoungTableaux[shape]: all standard Young tableaux of the partition shape (rows strictly increase left "
          "to right and down columns), sorted by row word, as Combinatorica Tableaux[shape]; their number equals "
          "HookLengthFormula[shape]. YoungTableaux[n]: all standard tableaux with n cells (over all partitions in "
          "reverse-lexicographic order), whose number is the number of involutions of n. Refuses more than "
          f"{_SYT_LIMIT} tableaux. Legacy built one tableau and returned its size")
def _young(arg):
    if is_list(arg):
        shape = [pos_int(p, "YoungTableaux", "part") for p in list_arg(arg, "YoungTableaux")]
        if any(shape[i] < shape[i + 1] for i in range(len(shape) - 1)):
            raise ValueError("YoungTableaux: shape must be a weakly decreasing list of positive integers")
        shapes = [shape]
    else:
        n = nonneg_int(arg, "YoungTableaux", "n")
        shapes = _syt_partitions(n)
    total = sum(_hook(L(*sh)) for sh in shapes) if shapes and shapes[0] else 1
    if total > _SYT_LIMIT:
        raise ValueError(f"YoungTableaux: {total} tableaux is more than the limit {_SYT_LIMIT}; use "
                         "HookLengthFormula[shape] to count them")
    out = []
    for sh in shapes:
        out.extend(_syt(sh))
    return L(*(L(*(L(*r) for r in t)) for t in out))
