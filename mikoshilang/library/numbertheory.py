"""Number theory.  Integer arguments are required exactly (no silent int()
truncation: legacy Prime[2.5] returned 3); symbolic arguments leave the call
unevaluated as in the Wolfram Language."""
import math
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, pos_int, list_arg, has_float, is_numeric_value)

AREA = "number theory"
NO_TRUNC = "legacy truncated non-integer arguments with int(); now exact integers only"


def _intq(x):
    return isinstance(x, int) and not isinstance(x, bool) or isinstance(x, sp.Integer)


@fn("IsPrime", area=AREA, status="verified", listable=True,
    notes="MikoshiLang alias of PrimeQ (same kernel implementation)"
          ". Check: reference (differential cases in tests/library).")
def _isprime(n):
    return Expr("PrimeQ", n)


@fn("NextPrime", area=AREA, status="fixed", listable=True,
    notes="legacy truncated reals and ignored negative primes; now NextPrime[x] is the smallest prime > x "
          "(primes include -2, -3, ... as in Wolfram) and NextPrime[x, k] the k-th"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _nextprime(x, k=1):
    v = number(x, "NextPrime")
    kk = int_arg(k, "NextPrime", "count")
    if kk == 0:
        raise ValueError("NextPrime: k must be non-zero")
    cur = v
    for _ in range(abs(kk)):
        cur = _next_after(cur) if kk > 0 else _prev_before(cur)
    return R(cur)


def _next_after(v):
    f = sp.floor(v)            # smallest prime > v
    if v >= 2:
        return sp.nextprime(int(f))
    if v >= -2:
        return sp.Integer(2)
    # negative primes: -p > v  <=>  p < -v ; take the largest such p
    bound = -v
    return -sp.prevprime(int(sp.ceiling(bound)))


def _prev_before(v):
    return -_next_after(-v)


@fn("PrimePi", area=AREA, status="fixed", listable=True,
    notes="PrimePi[x] for real x counts primes <= x (legacy truncated with int(), wrong for negative floats)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _primepi(x):
    v = number(x, "PrimePi")
    if v in (sp.oo, -sp.oo):
        raise ValueError("PrimePi: argument must be finite")
    return R(sp.primepi(int(sp.floor(v))) if v >= 2 else 0)


@fn("Prime", area=AREA, status="fixed", listable=True, notes="Prime[n] for positive integers only; " + NO_TRUNC)
def _prime(n):
    return R(sp.prime(pos_int(n, "Prime", "index")))


def _factor(n, fname):
    if _intq(n):
        n = int(n)
        if n == 0:
            return L(L(0, 1))
        if n in (1,):
            return L(L(1, 1))
        f = sp.factorint(n)
    else:
        v = number(n, fname)
        if not v.is_Rational:
            raise ValueError(f"{fname}: argument must be an integer or a rational number")
        f = sp.factorrat(v)
    items = sorted(f.items(), key=lambda kv: kv[0])
    return L(*(L(int(p), int(e)) for p, e in items))


@fn("FactorInteger", area=AREA, status="fixed", listable=True,
    notes="FactorInteger[1] returned {} (Wolfram {{1,1}}), 0 and rationals failed; now {{-1,1},...} for negatives, "
          "negative exponents for rationals")
def _factorint(n):
    return _factor(n, "FactorInteger")


@fn("PrimeFactorization", area=AREA, status="fixed", listable=True, notes="MikoshiLang alias of FactorInteger"
                                                                          ". Check: exact (exact input gives exact "
                                                                          "results), reference (differential cases "
                                                                          "in tests/library).")
def _primefactorization(n):
    return _factor(n, "PrimeFactorization")


def _nonzero(n, fname):
    k = int_arg(n, fname)
    if k == 0:
        raise ValueError(f"{fname}: argument must be a non-zero integer")
    return abs(k)


@fn("Divisors", area=AREA, status="fixed", listable=True, notes="Divisors[-n] = Divisors[n]; 0 raises; " + NO_TRUNC)
def _divisors(n):
    return R(sp.divisors(_nonzero(n, "Divisors")))


@fn("DivisorCount", area=AREA, status="fixed", listable=True, notes="number of divisors; " + NO_TRUNC)
def _divisorcount(n):
    return R(sp.divisor_count(_nonzero(n, "DivisorCount")))


@fn("DivisorSigma", area=AREA, status="fixed", listable=True,
    notes="DivisorSigma[k, n] = sum of d^k over divisors of n; exact for any exact k (legacy truncated k)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _divisorsigma(k, n):
    m = _nonzero(n, "DivisorSigma")
    kk = number(k, "DivisorSigma", "power", real=False)
    return numeric_out(sp.Add(*[sp.Integer(d) ** kk for d in sp.divisors(m)]), k)


@fn("EulerPhi", area=AREA, status="fixed", listable=True,
    notes="EulerPhi[0] = 0 and EulerPhi[-n] = EulerPhi[n] (legacy raised / gave wrong values)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _eulerphi(n):
    k = int_arg(n, "EulerPhi")
    return R(0 if k == 0 else sp.totient(abs(k)))


@fn("MoebiusMu", area=AREA, status="fixed", listable=True, notes="MoebiusMu[-n] = MoebiusMu[n]; 0 raises"
                                                                 ". Check: exact (exact input gives exact results), "
                                                                 "reference (differential cases in tests/library).")
def _mobius(n):
    return R(sp.mobius(_nonzero(n, "MoebiusMu")))


@fn("PowerMod", area=AREA, status="fixed", listable=True,
    notes="negative exponents use the modular inverse and raise when it does not exist; rational exponents raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _powermod(a, b, m):
    aa, bb, mm = int_arg(a, "PowerMod"), int_arg(b, "PowerMod", "exponent"), int_arg(m, "PowerMod", "modulus")
    if mm == 0:
        raise ValueError("PowerMod: modulus must be non-zero")
    if bb < 0:
        if math.gcd(aa, mm) != 1:
            raise ValueError(f"PowerMod: {aa} is not invertible modulo {mm}")
        aa = pow(aa % abs(mm), -1, abs(mm))
        bb = -bb
    r = pow(aa, bb, abs(mm))
    if mm < 0 and r:
        r += mm
    return R(r)


@fn("ModularInverse", area=AREA, status="fixed", listable=True,
    notes="raises a clear error when no inverse exists (legacy leaked a Python ValueError)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _modinv(a, m):
    aa, mm = int_arg(a, "ModularInverse"), int_arg(m, "ModularInverse", "modulus")
    if mm <= 0:
        raise ValueError("ModularInverse: modulus must be a positive integer")
    if math.gcd(aa, mm) != 1:
        raise ValueError(f"ModularInverse: {aa} is not invertible modulo {mm}")
    return R(pow(aa, -1, mm) if mm > 1 else 0)


@fn("ChineseRemainder", area=AREA, status="fixed",
    notes="smallest non-negative solution; non-coprime moduli handled; no solution leaves the call "
          "unevaluated (legacy returned an Error expression)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _crt(rs, ms):
    from sympy.ntheory.modular import solve_congruence
    r = [int_arg(v, "ChineseRemainder") for v in list_arg(rs, "ChineseRemainder")]
    m = [int_arg(v, "ChineseRemainder", "modulus") for v in list_arg(ms, "ChineseRemainder")]
    if len(r) != len(m) or not r:
        raise ValueError("ChineseRemainder: lists of remainders and moduli must have the same non-zero length")
    if any(v <= 0 for v in m):
        raise ValueError("ChineseRemainder: moduli must be positive")
    res = solve_congruence(*zip(r, m))
    if res is None:
        raise Unevaluated()
    return R(res[0])


def _rationals(args, fname):
    out = []
    for a in args:
        if _intq(a):
            out.append(sp.Integer(a))
            continue
        if not is_numeric_value(a):
            raise Unevaluated()
        v = S(a)
        if not v.is_Rational:
            raise ValueError(f"{fname}: arguments must be integers or rational numbers")
        out.append(v)
    return out


@fn("GCD", area=AREA, status="fixed", listable=True,
    notes="now variadic, exact for rationals, GCD[] = 0; legacy truncated reals (GCD[2.5, 5] = 1)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _gcd(*args):
    vals = _rationals(args, "GCD")
    if not vals:
        return 0
    g = vals[0]
    for v in vals[1:]:
        g = sp.gcd(g, v)
    return R(abs(g))


@fn("LCM", area=AREA, status="fixed", listable=True,
    notes="now variadic, exact for rationals, LCM[] = 1; legacy truncated reals"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _lcm(*args):
    vals = _rationals(args, "LCM")
    if not vals:
        return 1
    g = vals[0]
    for v in vals[1:]:
        g = sp.lcm(g, v)
    return R(abs(g))


@fn("ExtendedGCD", area=AREA, status="fixed",
    notes="returned {x, y, g}; Wolfram order is {g, {x, y}} with g = a x + b y"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _egcd(a, b):
    aa, bb = int_arg(a, "ExtendedGCD"), int_arg(b, "ExtendedGCD")
    # iterative extended Euclid (same minimal Bezout pair as SymPy igcdex)
    x0, y0, x1, y1, r0, r1 = 1, 0, 0, 1, aa, bb
    while r1:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    x, y, g = x0, y0, r0
    if g == 0:
        return L(0, L(0, 0))          # ExtendedGCD[0, 0] = {0, {0, 0}} (Wolfram)
    if g < 0:
        x, y, g = -x, -y, -g
    return L(int(g), L(int(x), int(y)))


def _fib(n):
    if n >= 0:
        return sp.fibonacci(n)
    return (-1) ** (-n + 1) * sp.fibonacci(-n)


def _fibonacci(n, x=None):
    if x is not None:
        k = nonneg_int(n, "Fibonacci", "index")
        return R(sp.expand(sp.fibonacci(k, sym(x, "Fibonacci"))))
    if _intq(n):
        return R(_fib(int(n)))
    if is_numeric_value(n):
        z = S(n)
        phi = sp.GoldenRatio
        return numeric_out((phi ** z - sp.cos(sp.pi * z) * phi ** (-z)) / sp.sqrt(5), n)
    raise Unevaluated()


fn("Fibonacci", area=AREA, status="fixed", listable=True,
   notes="symbolic n raised TypeError (now unevaluated); negative n, real n and Fibonacci[n, x] added"
         ". Check: exact (exact input gives exact results), reference (differential cases in "
         "tests/library).")(_fibonacci)
fn("FibonacciNumber", area=AREA, status="fixed", listable=True, notes="MikoshiLang alias of Fibonacci"
                                                                      ". Check: exact (exact input gives exact "
                                                                      "results), reference (differential cases in "
                                                                      "tests/library).")(_fibonacci)


def _lucas(n):
    if _intq(n):
        k = int(n)
        return R(sp.lucas(k) if k >= 0 else (-1) ** (-k) * sp.lucas(-k))
    if is_numeric_value(n):
        z = S(n)
        phi = sp.GoldenRatio
        return numeric_out(phi ** z + sp.cos(sp.pi * z) * phi ** (-z), n)
    raise Unevaluated()


fn("Lucas", area=AREA, status="fixed", listable=True,
   notes="MikoshiLang alias of LucasL; negative and real n added; " + NO_TRUNC)(_lucas)
fn("LucasL", area=AREA, status="promoted", listable=True, notes="Lucas numbers L_n")(_lucas)


def _bernoulli(n, x=None):
    k = nonneg_int(n, "BernoulliB", "index")
    if x is not None:
        return R(sp.expand(sp.bernoulli(k, sym(x, "BernoulliB"))))
    return R(sp.Rational(-1, 2) if k == 1 else sp.bernoulli(k))


B1 = "Bernoulli[1] returned +1/2 (SymPy >= 1.12 convention); Wolfram BernoulliB[1] = -1/2"
fn("Bernoulli", area=AREA, status="fixed", listable=True, notes=B1)(_bernoulli)
fn("BernoulliNumber", area=AREA, status="fixed", listable=True, notes=B1)(_bernoulli)
fn("BernoulliB", area=AREA, status="promoted", listable=True, notes="BernoulliB[n] and BernoulliB[n, x]"
                                                                    ". Check: exact (exact input gives exact "
                                                                    "results), reference (differential cases in "
                                                                    "tests/library).")(_bernoulli)


def _euler(n, x=None):
    k = nonneg_int(n, "EulerE", "index")
    if x is not None:
        return R(sp.expand(sp.euler(k, sym(x, "EulerE"))))
    return R(sp.euler(k))


fn("EulerNumber", area=AREA, status="verified", listable=True, notes="Euler numbers E_n (E_1 = 0, E_4 = 5)"
                                                                     ". Check: exact (exact input gives exact "
                                                                     "results), reference (differential cases in "
                                                                     "tests/library).")(_euler)
fn("EulerE", area=AREA, status="promoted", listable=True, notes="EulerE[n] and EulerE[n, x]"
                                                                ". Check: exact (exact input gives exact results), "
                                                                "reference (differential cases in tests/library).")(_euler)

@fn("Catalan", area=AREA, status="fixed",
    notes="Wolfram constant Catalan = 0.915965594177219... (SymPy Catalan; N[Catalan, n] to any precision, "
          "Sum[(-1)^k/(2k+1)^2, {k, 0, Infinity}] -> Catalan); Catalan[...] used as a function raises ValueError "
          "pointing to CatalanNumber; legacy Catalan[n] gave Catalan numbers")
def _catalan_call(*args):
    raise ValueError("Catalan is Catalan's constant (0.915965...), not a function; use CatalanNumber[n] for "
                     "Catalan numbers")


@fn("CatalanNumber", area=AREA, status="fixed", listable=True,
    notes="exact for any exact argument, symbolic n stays unevaluated; " + NO_TRUNC)
def _catalan(n):
    if not is_numeric_value(n):
        raise Unevaluated()
    return numeric_out(sp.catalan(S(n)), n)


def _bell(n, x=None):
    k = nonneg_int(n, "BellB", "index")
    if x is not None:
        return R(sp.expand(sp.bell(k, sym(x, "BellB"))))
    return R(sp.bell(k))


fn("Bell", area=AREA, status="fixed", listable=True, notes="alias of BellB; " + NO_TRUNC)(_bell)
fn("BellNumber", area=AREA, status="fixed", listable=True, notes="alias of BellB; " + NO_TRUNC)(_bell)
fn("BellB", area=AREA, status="promoted", listable=True, notes="BellB[n] and Bell polynomials BellB[n, x]"
                                                               ". Check: exact (exact input gives exact results), "
                                                               "reference (differential cases in tests/library).")(_bell)


def _stirling(n, k, kind):
    from sympy.functions.combinatorial.numbers import stirling
    nn = nonneg_int(n, "Stirling")
    kk = nonneg_int(k, "Stirling")
    return R(stirling(nn, kk, kind=kind, signed=(kind == 1)))


@fn("StirlingS1", area=AREA, status="fixed", listable=True,
    notes="returned unsigned Stirling numbers (StirlingS1[3, 2] = 3); Wolfram uses signed numbers (-3)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _s1(n, k):
    return _stirling(n, k, 1)


@fn("StirlingS2", area=AREA, status="fixed", listable=True, notes="Stirling numbers of the second kind; " + NO_TRUNC)
def _s2(n, k):
    return _stirling(n, k, 2)


@fn("Stirling", area=AREA, status="fixed",
    notes="Stirling[n, k] = StirlingS2[n, k]; Stirling[n, k, 1] now gives the signed numbers like StirlingS1"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _stirling_any(n, k, kind=2):
    t = int_arg(kind, "Stirling", "kind")
    if t not in (1, 2):
        raise ValueError("Stirling: kind must be 1 or 2")
    return _stirling(n, k, t)


@fn("DigitSum", area=AREA, status="fixed", listable=True,
    notes="DigitSum[n] and DigitSum[n, b] (sum of base-b digits of |n|); " + NO_TRUNC)
def _digitsum(n, b=10):
    k, base = abs(int_arg(n, "DigitSum")), int_arg(b, "DigitSum", "base")
    if base < 2:
        raise ValueError("DigitSum: base must be an integer >= 2")
    s = 0
    while k:
        s += k % base
        k //= base
    return s


def _digits(n, base):
    n = abs(n)
    if n == 0:
        return [0]
    out = []
    while n:
        out.append(n % base)
        n //= base
    return out[::-1]


def _base(b, fname):
    base = int_arg(b, fname, "base")
    if base < 2:
        raise ValueError(f"{fname}: base must be an integer >= 2")
    return base


@fn("DigitCount", area=AREA, status="fixed", listable=True,
    notes="legacy returned the number of digits; Wolfram DigitCount[n] is the list of counts of digits "
          "1, 2, ..., 9, 0; DigitCount[n, b] and DigitCount[n, b, d] also supported"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _digitcount(n, b=10, d=None):
    k = int_arg(n, "DigitCount")
    base = _base(b, "DigitCount")
    ds = _digits(k, base)
    if d is not None:
        dd = int_arg(d, "DigitCount", "digit")
        return ds.count(dd)
    return L(*([ds.count(i) for i in range(1, base)] + [ds.count(0)]))


@fn("IntegerDigits", area=AREA, status="fixed", listable=True,
    notes="legacy silently used base 10 for a non-integer base; now IntegerDigits[n], [n, b], [n, b, len] "
          "(digits of |n|, padded/truncated on the left)")
def _intdigits(n, b=10, length=None):
    k = int_arg(n, "IntegerDigits")
    base = _base(b, "IntegerDigits")
    ds = _digits(k, base)
    if length is not None:
        ln = nonneg_int(length, "IntegerDigits", "length")
        ds = ([0] * ln + ds)[-ln:] if ln else []
    return L(*ds)


def _letter_value(ch, fname):
    if ch.isdigit():
        return int(ch)
    if ch.isalpha() and ch.isascii():
        return ord(ch.lower()) - ord("a") + 10
    raise ValueError(f"{fname}: invalid digit character {ch!r}")


@fn("FromDigits", area=AREA, status="fixed",
    notes="now accepts digit strings (letters a-z = 10-35), symbolic digits and digits >= base as Wolfram; "
          "legacy silently used base 10 for invalid bases")
def _fromdigits(ds, b=10):
    if isinstance(b, (Symbol, Expr)) and not isinstance(b, str):
        base = sym(b, "FromDigits")
    else:
        base = sp.Integer(_base(b, "FromDigits"))
    if isinstance(ds, str):
        vals = [_letter_value(c, "FromDigits") for c in ds]
        if base.is_Integer and any(v >= base for v in vals):
            raise ValueError(f"FromDigits: digit out of range for base {base}")
        vals = [sp.Integer(v) for v in vals]
    else:
        items = list_arg(ds, "FromDigits")
        if len(items) == 2 and is_list(items[0]) and not is_list(items[1]):
            return _fromdigits_real(items[0], items[1], base)          # FromDigits[{digits, exponent}, b]
        vals = [sym(v, "FromDigits") for v in items]
    acc = sp.Integer(0)
    for v in vals:
        acc = acc * base + v
    return R(sp.expand(acc))


def _fromdigits_real(digs, e, base):
    """Inverse of RealDigits: {{d1, d2, ..., {r1, ..., rk}}, e} (a trailing nested list repeats forever)."""
    items = list(digs.args)
    rep = []
    if items and is_list(items[-1]):
        rep = [sym(v, "FromDigits") for v in items[-1].args]
        items = items[:-1]
    head = [sym(v, "FromDigits") for v in items]
    ex = sym(e, "FromDigits")
    val = sp.Integer(0)
    for i, d in enumerate(head, 1):
        val += d * base ** (-i)
    if rep:
        k, n0 = len(rep), len(head)
        block = sum(d * base ** (k - j) for j, d in enumerate(rep, 1))
        val += block / (base ** k - 1) * base ** (-n0)
    return R(sp.nsimplify(sp.expand(val * base ** ex)) if not has_float(digs) else sp.expand(val * base ** ex))


@fn("BaseConvert", area=AREA, status="fixed",
    notes="BaseConvert[n, from, to] reads the decimal digits of n as base-'from' digits; legacy accepted "
          "invalid digits (BaseConvert[19, 2, 10] = 3); now raises. Returns an integer for to = 10, else a digit list"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _baseconvert(n, b1, b2):
    k = nonneg_int(n, "BaseConvert")
    fb, tb = _base(b1, "BaseConvert"), _base(b2, "BaseConvert")
    if fb > 10:
        raise ValueError("BaseConvert: source bases above 10 cannot be written with decimal digits; "
                         "use FromDigits[\"string\", base]")
    dec = 0
    for ch in str(k):
        d = int(ch)
        if d >= fb:
            raise ValueError(f"BaseConvert: digit {d} is not valid in base {fb}")
        dec = dec * fb + d
    if tb == 10:
        return dec
    return L(*_digits(dec, tb))


@fn("CarmichaelLambda", area=AREA, status="fixed", listable=True,
    notes="CarmichaelLambda[-n] = CarmichaelLambda[n]; 0 raises; " + NO_TRUNC)
def _carmichael(n):
    m = _nonzero(n, "CarmichaelLambda")
    lam = 1
    for p, e in sp.factorint(m).items():
        if p == 2 and e >= 3:
            t = 2 ** (e - 2)
        else:
            t = (p - 1) * p ** (e - 1)
        lam = lam * t // math.gcd(lam, t)
    return lam


@fn("Radical", area=AREA, status="fixed", listable=True,
    notes="product of the distinct primes dividing n; Radical[0] raises; " + NO_TRUNC)
def _radical(n):
    m = _nonzero(n, "Radical")
    return R(math.prod(sp.factorint(m).keys()) if m > 1 else 1)


@fn("JordanTotient", area=AREA, status="fixed", listable=True,
    notes="legacy called a SymPy function that does not exist (AttributeError); now J_k(n) = n^k prod(1 - p^-k), "
          "JordanTotient[n] = EulerPhi[n]"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _jordan(n, k=1):
    m = pos_int(n, "JordanTotient")
    kk = pos_int(k, "JordanTotient", "order")
    out = sp.Integer(m) ** kk
    for p in sp.factorint(m):
        out = out * (1 - sp.Integer(p) ** (-kk))
    return R(sp.Integer(out))


@fn("LegendreSymbol", area=AREA, status="fixed", listable=True,
    notes="p must be an odd prime (clear error otherwise); " + NO_TRUNC)
def _legendre(a, p):
    aa, pp = int_arg(a, "LegendreSymbol"), int_arg(p, "LegendreSymbol")
    if pp < 3 or not sp.isprime(pp):
        raise ValueError("LegendreSymbol: second argument must be an odd prime")
    return R(sp.legendre_symbol(aa % pp, pp))


@fn("JacobiSymbol", area=AREA, status="fixed", listable=True,
    notes="m must be a positive odd integer (clear error otherwise); " + NO_TRUNC)
def _jacobi(n, m):
    nn, mm = int_arg(n, "JacobiSymbol"), int_arg(m, "JacobiSymbol")
    if mm <= 0 or mm % 2 == 0:
        raise ValueError("JacobiSymbol: second argument must be a positive odd integer")
    return R(sp.jacobi_symbol(nn % mm, mm))


def _kron(a, n):
    if n == 0:
        return 1 if abs(a) == 1 else 0
    res = 1
    if n < 0:
        n = -n
        if a < 0:
            res = -res
    v = 0
    while n % 2 == 0:
        n //= 2
        v += 1
    if v:
        if a % 2 == 0:
            return 0
        if v % 2 and a % 8 in (3, 5):
            res = -res
    if n == 1:
        return res
    return res * int(sp.jacobi_symbol(a % n, n))


@fn("KroneckerSymbol", area=AREA, status="fixed", listable=True,
    notes="legacy called a SymPy function that does not exist (AttributeError); now the full Kronecker symbol "
          "(n = 0, even and negative n)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _kronecker(a, n):
    return _kron(int_arg(a, "KroneckerSymbol"), int_arg(n, "KroneckerSymbol"))


@fn("QuadraticResidues", area=AREA, status="verified", listable=True,
    notes="sorted residues x^2 mod n including 0 (MikoshiLang function); n must be positive"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _qr(n):
    m = pos_int(n, "QuadraticResidues", "modulus")
    return L(*sorted({(i * i) % m for i in range(m)}))


@fn("PrimitiveRoot", area=AREA, status="fixed", listable=True,
    notes="smallest primitive root; legacy returned None (displayed as nothing) when none exists, now unevaluated "
          "like Wolfram"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _primroot(n):
    m = pos_int(n, "PrimitiveRoot", "modulus")
    if m == 1:
        return 0
    r = sp.primitive_root(m)
    if r is None:
        raise Unevaluated()
    return R(r)


@fn("PrimitiveRootQ", area=AREA, status="fixed", listable=True,
    notes="PrimitiveRootQ[a, n]: True when a generates the units mod n; False when gcd(a, n) != 1 "
          "(legacy raised a SymPy ValueError)"
          ". Check: reference (differential cases in tests/library).")
def _primrootq(a, n):
    aa, m = int_arg(a, "PrimitiveRootQ"), pos_int(n, "PrimitiveRootQ", "modulus")
    if math.gcd(aa, m) != 1:
        return False
    return bool(sp.is_primitive_root(aa % m, m)) if m > 1 else True


@fn("MultiplicativeOrder", area=AREA, status="fixed", listable=True,
    notes="raises a clear error when gcd(k, n) != 1; " + NO_TRUNC)
def _order(k, n):
    kk, m = int_arg(k, "MultiplicativeOrder"), pos_int(n, "MultiplicativeOrder", "modulus")
    if math.gcd(kk, m) != 1:
        raise ValueError(f"MultiplicativeOrder: {kk} is not a unit modulo {m}")
    return R(sp.n_order(kk % m, m) if m > 1 else 1)


@fn("PrimePowerQ", area=AREA, status="fixed", listable=True,
    notes="True for p^k with k >= 1; False for 0, 1 and non-integers; negative integers raise (Gaussian/negative "
          "conventions not supported); legacy truncated reals"
          ". Check: reference (differential cases in tests/library).")
def _primepowerq(n):
    if not _intq(n):
        if is_numeric_value(n):
            return False
        raise Unevaluated()
    k = int(n)
    if k < 0:
        raise NotImplementedError("PrimePowerQ: negative integers are not supported")
    return k > 1 and len(sp.factorint(k)) == 1


@fn("ContinuedFractionConvergents", area=AREA, status="fixed",
    notes="convergents p/q of [a0; a1, ...] as {p, q} pairs (MikoshiLang); legacy raised for integer convergents"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _convergents(cf):
    terms = [int_arg(t, "ContinuedFractionConvergents", "term") for t in list_arg(cf, "ContinuedFractionConvergents", min_len=1)]
    out = []
    for c in sp.continued_fraction_convergents(terms):
        c = sp.Rational(c)
        out.append(L(int(c.p), int(c.q)))
    return L(*out)


@fn("DiophantineSolve", area=AREA, status="fixed",
    notes="legacy called sympy.diophantine.diophantine (AttributeError for every input); now the general "
          "integer solution via SymPy diophantine, parameters t0, t1, ... in sorted order"
          ". Check: reference (differential cases in tests/library).")
def _diophantine(eq, vars_):
    from sympy.solvers.diophantine.diophantine import diophantine
    e = sym(eq, "DiophantineSolve")
    if isinstance(e, sp.Equality):
        e = e.lhs - e.rhs
    vs = [sp.Symbol(v.name) if isinstance(v, Symbol) else None for v in list_arg(vars_, "DiophantineSolve")]
    if None in vs or not vs:
        raise ValueError("DiophantineSolve: second argument must be a list of variables")
    if not e.is_polynomial(*vs):
        raise ValueError("DiophantineSolve: equation must be polynomial in the variables")
    sols = diophantine(sp.expand(e), syms=vs)
    return L(*sorted((L(*(R(s) for s in sol)) for sol in sols), key=str))


@fn("RationalReconstruct", area=AREA, status="fixed",
    notes="legacy used nsimplify, so RationalReconstruct[1.4142135623730951] returned Sqrt[2] (not a rational); "
          "now the exact rational with the same decimal digits, simplest within machine precision"
          ". Check: reference (differential cases in tests/library).")
def _ratrec(x):
    v = number(x, "RationalReconstruct")
    if v.is_Rational:
        return R(v)
    if not v.is_Float:
        raise ValueError("RationalReconstruct: argument must be a rational or decimal number")
    return R(sp.nsimplify(v, rational=True))


@fn("Ackermann", area=AREA, status="fixed",
    notes="exact closed forms for m <= 3; larger m raise instead of recursing until RecursionError"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _ackermann(m, n):
    mm, nn = nonneg_int(m, "Ackermann"), nonneg_int(n, "Ackermann")
    if mm == 0:
        return nn + 1
    if mm == 1:
        return nn + 2
    if mm == 2:
        return 2 * nn + 3
    if mm == 3:
        if nn > 10000:
            raise ValueError("Ackermann: result too large")
        return 2 ** (nn + 3) - 3
    if mm == 4 and nn == 0:
        return 13
    if mm == 4 and nn == 1:
        return 65533
    raise ValueError("Ackermann: values for m >= 4 (beyond A(4, 1)) are too large to compute")


def _collatz_seq(n, fname):
    k = pos_int(n, fname)
    seq = [k]
    while k != 1:
        k = k // 2 if k % 2 == 0 else 3 * k + 1
        seq.append(k)
    return seq


@fn("CollatzSequence", area=AREA, status="fixed", listable=True,
    notes="legacy looped forever for n <= 0; now n must be a positive integer"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _collatz(n):
    return L(*_collatz_seq(n, "CollatzSequence"))


@fn("CollatzLength", area=AREA, status="fixed", listable=True,
    notes="number of terms of CollatzSequence[n] (including n and 1); legacy looped forever for n <= 0"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _collatzlen(n):
    return len(_collatz_seq(n, "CollatzLength"))


# ───────────────────────────── coverage batch 2 additions and fixes ─────────────────────────────

B2 = "number theory (batch 2)"


@fn("LucasL", area=AREA, status="fixed", listable=True,
    notes="LucasL[n, x] Lucas polynomials L_0 = 2, L_1 = x, L_n = x L_(n-1) + L_(n-2) (LucasL[3, x] = x^3 + 3 x); "
          "the 2-argument form was missing"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _lucas_poly(n, x):
    k = nonneg_int(n, "LucasL", "index")
    xv = sym(x, "LucasL")
    a, b = sp.Integer(2), xv
    if k == 0:
        return 2
    for _ in range(k - 1):
        a, b = b, sp.expand(xv * b + a)
    return numeric_out(sp.expand(b), x)


@fn("HarmonicNumber", area=AREA, status="fixed", listable=True,
    notes="HarmonicNumber[n] for rational n in closed form (HarmonicNumber[1/2] = 2 - 2 Log[2], legacy left it "
          "unevaluated); HarmonicNumber[n, r]"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _harmonic_b2(n, r=1):
    if not is_numeric_value(n):
        raise Unevaluated()
    sn, sr = S(n), sym(r, "HarmonicNumber")
    if sn.is_Integer and sn < 0:
        raise ValueError("HarmonicNumber: negative integers are poles")
    v = sp.harmonic(sn, sr)
    if isinstance(v, sp.harmonic) and sn.is_Rational and sr == 1:
        v = sp.expand_func(v)
    return numeric_out(v, n, r)




def _replace_first(name, newf):
    from .core import REGISTRY
    e = REGISTRY[name]
    e.impls = [i for i in e.impls if i[0] is newf] + [i for i in e.impls if i[0] is not newf]


_replace_first("HarmonicNumber", _harmonic_b2)


@fn("CatalanNumber", area=AREA, status="fixed", listable=True,
    notes="CatalanNumber[n] for any exact n via 4^n Gamma[n + 1/2]/(Sqrt[Pi] Gamma[n + 2]) "
          "(CatalanNumber[1/2] = 8/(3 Pi), legacy left it unevaluated); symbolic n stays unevaluated")
def _catalan_b2(n):
    if not is_numeric_value(n):
        raise Unevaluated()
    v = sp.catalan(S(n))
    if isinstance(v, sp.catalan):
        v = sp.simplify(v.rewrite(sp.gamma))
    return numeric_out(v, n)


_replace_first("CatalanNumber", _catalan_b2)


@fn("PowerMod", area=AREA, status="fixed", listable=True,
    notes="PowerMod[a, 1/r, m] is the smallest modular r-th root of a (PowerMod[3, 1/2, 11] = 5), unevaluated "
          "when none exists; other rational exponents p/r use the root then the power")
def _powermod_root(a, b, m):
    bb = sym(b, "PowerMod")
    if bb.is_Integer:
        return _powermod(a, b, m)
    if not bb.is_Rational:
        raise ValueError("PowerMod: exponent must be an integer or a rational number")
    aa, mm = int_arg(a, "PowerMod"), int_arg(m, "PowerMod", "modulus")
    if mm == 0:
        raise ValueError("PowerMod: modulus must be non-zero")
    p, q = int(bb.p), int(bb.q)
    from sympy.ntheory.residue_ntheory import nthroot_mod
    roots = nthroot_mod(aa % abs(mm), q, abs(mm), all_roots=True) if abs(mm) > 1 else [0]
    if not roots:
        raise Unevaluated()
    r = min(roots)
    return _powermod(r, p, m)


_replace_first("PowerMod", _powermod_root)


def _unlist(name):
    """Make a listable head thread only over its leading arguments (so a list-valued option argument works)."""
    from .core import REGISTRY
    e = REGISTRY[name]
    e.listable = False
    old = [i for i in e.impls]

    def threaded(*args):
        if any(is_list(a) for a in args):
            n = {len(a.args) for a in args if is_list(a)}
            if len(n) != 1:
                raise ValueError(f"{name}: cannot thread over lists of unequal length")
            k = n.pop()
            return L(*(Expr(name, *(a.args[i] if is_list(a) else a for a in args)) for i in range(k)))
        return old[0][0](*args)
    e.impls = [(threaded, old[0][1], old[0][2])] + old[1:]


_unlist("MultiplicativeOrder")


@fn("MultiplicativeOrder", area=AREA, status="fixed",
    notes="MultiplicativeOrder[k, n, {r1, r2, ...}] is the smallest positive m with k^m = ri (mod n) for some ri; "
          "unevaluated when there is none (the 3-argument form was missing)")
def _order3(k, n, rs):
    kk, m = int_arg(k, "MultiplicativeOrder"), pos_int(n, "MultiplicativeOrder", "modulus")
    targets = {int_arg(r, "MultiplicativeOrder", "residue") % m for r in list_arg(rs, "MultiplicativeOrder")}
    if math.gcd(kk, m) != 1:
        raise ValueError(f"MultiplicativeOrder: {kk} is not a unit modulo {m}")
    order = int(sp.n_order(kk % m, m)) if m > 1 else 1
    x = 1
    for e in range(1, order + 1):
        x = (x * kk) % m
        if x in targets:
            return e
    raise Unevaluated()


@fn("PrimitiveRoot", area=AREA, status="fixed",
    notes="PrimitiveRoot[n, k] is the smallest primitive root of n that is >= k (the 2-argument form was missing)")
def _primroot2(n, k):
    m = pos_int(n, "PrimitiveRoot", "modulus")
    start = int_arg(k, "PrimitiveRoot", "lower bound")
    if sp.primitive_root(m) is None:
        raise Unevaluated()
    for a in range(max(start, 1), start + 10 * m + 2):
        if math.gcd(a, m) == 1 and sp.is_primitive_root(a % m, m):
            return a
    raise Unevaluated()


@fn("MangoldtLambda", area=B2, status="added", listable=True,
    notes="von Mangoldt function: Log[p] if n is a power of the prime p, else 0; MangoldtLambda[1] = 0"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _mangoldt(n):
    k = int_arg(n, "MangoldtLambda")
    if k < 1:
        raise ValueError("MangoldtLambda: argument must be a positive integer")
    f = sp.factorint(k)
    if len(f) == 1:
        return R(sp.log(next(iter(f))))
    return 0


@fn("LiouvilleLambda", area=B2, status="added", listable=True,
    notes="Liouville function (-1)^Omega(n), Omega counting prime factors with multiplicity; LiouvilleLambda[1] = 1"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _liouville(n):
    k = int_arg(n, "LiouvilleLambda")
    if k == 0:
        raise ValueError("LiouvilleLambda: argument must be a non-zero integer")
    return (-1) ** sum(sp.factorint(abs(k)).values())


@fn("DivisorSum", area=B2, status="added",
    notes="DivisorSum[n, form] = Sum of form[d] over the positive divisors d of n; DivisorSum[n, form, cond] keeps "
          "the divisors for which cond[d] is True"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _divisorsum(n, form, cond=None):
    from ..evaluate import evaluate
    from .wl_common import apply
    k = int_arg(n, "DivisorSum")
    if k <= 0:
        raise ValueError("DivisorSum: n must be a positive integer")
    total = []
    for d in sp.divisors(k):
        if cond is not None and apply(cond, int(d)) is not True:
            continue
        total.append(apply(form, int(d)))
    return evaluate(Expr("Plus", *total)) if total else 0


@fn("RandomPrime", area=B2, status="added",
    notes="RandomPrime[n] is a uniformly random prime in [2, n]; RandomPrime[{a, b}]; RandomPrime[range, k] a "
          "list of k primes (independent draws); raises when the range contains no prime")
def _randomprime(rng, count=None):
    import random
    if is_list(rng):
        lo, hi = (int_arg(v, "RandomPrime", "range") for v in list_arg(rng, "RandomPrime"))
    else:
        lo, hi = 2, int_arg(rng, "RandomPrime", "bound")
    lo = max(lo, 2)
    if hi < lo or sp.nextprime(lo - 1) > hi:
        raise ValueError(f"RandomPrime: there are no primes in [{lo}, {hi}]")

    def one():
        # uniform over the primes in [lo, hi]: draw uniform integers until one is prime
        while True:
            c = random.randint(lo, hi)
            if sp.isprime(c):
                return c
    if count is None:
        return one()
    if is_list(count):
        dims = [pos_int(v, "RandomPrime", "dimension") for v in count.args]
    else:
        dims = [pos_int(count, "RandomPrime", "count")]

    def build(ds):
        if len(ds) == 1:
            return L(*(one() for _ in range(ds[0])))
        return L(*(build(ds[1:]) for _ in range(ds[0])))
    return build(dims)
