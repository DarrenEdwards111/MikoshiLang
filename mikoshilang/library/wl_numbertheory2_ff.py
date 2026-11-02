"""Wolfram batch W09 -- finite fields: FiniteField, FiniteFieldElement and the functions on them.

Representation
--------------
* A field is the inert expression ``FiniteField[p, d]`` (default irreducible, "Polynomial" representation),
  ``FiniteField[p, d, "Exponential"]`` or ``FiniteField[p, f, rep]`` with ``f`` a pure function (the field
  irreducible, coefficients reduced modulo p).  ``FiniteField[q]`` for a prime power q is ``FiniteField[p, d]``.
* The default irreducible of ``FiniteField[p, d]`` is the Conway polynomial C(p, d) (the polynomial the Wolfram
  Language documentation shows for FiniteField[29, 4], 2 + 15 #1 + 2 #1^2 + #1^4 &, is C(29, 4)).  It is computed by
  the standard search (least primitive polynomial in Conway order compatible with the Conway polynomials of all
  subfields); when that search is too large a clear error asks for an explicit irreducible.  It is only computed
  when needed (d > 1, the "Exponential" representation, or an explicit "FieldIrreducible" query).
* An element is ``FiniteFieldElement[ff, {c0, c1, ...}]`` (coefficients in the polynomial basis 1, t, t^2, ...,
  trailing zeros dropped; the zero element has ``{}``) for both representations.  Indices follow the Wolfram
  conventions: for "Polynomial" the index is sum c_i p^i; for "Exponential" the element t^k has index k + 1 and
  zero has index 0.

Arithmetic (Plus, Times, Power including rational powers with denominator 2 or p^k, Equal, Unequal) combines
elements of the same field and integers/rationals (interpreted modulo p).  Elements of different fields do not
combine (an error).  Polynomial algebra, linear algebra and equation solving over finite fields are not
implemented: those heads raise a clear error when given finite field elements instead of treating them as
opaque symbols.
"""
from __future__ import annotations

import math
import time

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated, REGISTRY, S, R, L, is_list
from .wl_common import ev, is_sym
from . import wl_numbertheory2 as NT

AREA = "finite fields (W09)"
ADDED = dict(area=AREA, status="added")
UNH = K.UNHANDLED
FFE = "FiniteFieldElement"
NT.SPECIAL_HEADS.add(FFE)

EXP_LIMIT = 2 ** 24          # largest field for the "Exponential" representation (discrete logs)
CONWAY_BUDGET = 2000000      # candidate polynomials tried in the Conway search
CONWAY_SECONDS = 20.0        # time limit of the Conway search


def _intq(x):
    return (isinstance(x, int) and not isinstance(x, bool)) or isinstance(x, sp.Integer)


# ─────────────────────────── polynomial arithmetic over GF(p) (lists low -> high) ───────────────────────────

def _trim(c):
    c = list(c)
    while c and c[-1] == 0:
        c.pop()
    return c


def _padd(a, b, p):
    n = max(len(a), len(b))
    return _trim([((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0)) % p for i in range(n)])


def _pscale(a, s, p):
    return _trim([(x * s) % p for x in a])


def _pmul(a, b, p):
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] = (r[i + j] + x * y) % p
    return _trim(r)


def _pmod(a, m, p):
    """a mod m (m monic, trimmed)."""
    a = list(a)
    n = len(m) - 1
    while len(a) > n:
        c = a[-1] % p
        if c:
            sh = len(a) - 1 - n
            for i in range(n + 1):
                a[sh + i] = (a[sh + i] - c * m[i]) % p
        a.pop()
    return _trim([x % p for x in a])


def _pdivmod(a, b, p):
    a = _trim([x % p for x in a])
    b = _trim(b)
    inv = pow(b[-1], -1, p)
    q = [0] * max(len(a) - len(b) + 1, 0)
    while len(a) >= len(b) and a:
        c = a[-1] * inv % p
        sh = len(a) - len(b)
        q[sh] = c
        for i in range(len(b)):
            a[sh + i] = (a[sh + i] - c * b[i]) % p
        a = _trim(a)
    return _trim(q), a


def _pinv(a, m, p):
    """Inverse of a modulo the irreducible m (extended Euclid)."""
    r0, r1 = list(m), _trim(a)
    s0, s1 = [], [1]
    if not r1:
        raise ZeroDivisionError
    while r1:
        q, r = _pdivmod(r0, r1, p)
        r0, r1 = r1, r
        s0, s1 = s1, _padd(s0, _pscale(_pmul(q, s1, p), p - 1, p), p)
    # r0 is a nonzero constant
    c = pow(r0[0], -1, p)
    return _pmod(_pscale(s0, c, p), m, p)


def _ppow(a, e, m, p):
    res, b = [1], _pmod(a, m, p)
    while e > 0:
        if e & 1:
            res = _pmod(_pmul(res, b, p), m, p)
        b = _pmod(_pmul(b, b, p), m, p)
        e >>= 1
    return res


def _peval(poly, x, m, p):
    """poly (list over GF(p)) evaluated at the field element x (list), modulo m."""
    r = []
    for c in reversed(poly):
        r = _padd(_pmod(_pmul(r, x, p), m, p), [c % p] if c % p else [], p)
    return r


# ─────────────────────────── Conway polynomials ───────────────────────────

_CONWAY = {}


def _divisors(n):
    return [k for k in range(1, n + 1) if n % k == 0]


def _primitive_monic(m, p, n, qfac):
    """Is the monic m (low -> high, degree n) primitive over GF(p)?"""
    if m[0] % p == 0:
        return False
    N = p ** n - 1
    x = [0, 1] if n > 1 else [(-m[0]) % p]
    if _ppow(x, N, m, p) != [1]:
        return False
    return all(_ppow(x, N // r, m, p) != [1] for r in qfac)


def _no_root(m, p):
    if p > 64:
        return True
    for x in range(p):
        v = 0
        for c in reversed(m):
            v = (v * x + c) % p
        if v == 0:
            return False
    return True


def conway(p, n):
    """The Conway polynomial C(p, n) as a monic coefficient list (low -> high)."""
    key = (p, n)
    if key in _CONWAY:
        return _CONWAY[key]
    if p ** n > 10 ** 40:
        raise NotImplementedError(f"FiniteField: the default (Conway) irreducible for FiniteField[{p}, {n}] is out "
                                  f"of reach; give an explicit irreducible, FiniteField[p, f]")
    qfac = list(sp.factorint(p ** n - 1))
    subs = [(m, conway(p, m)) for m in _divisors(n) if m < n]
    tried = 0
    # c_i in lexicographic order; coefficient of x^(n-i) is (-1)^i c_i.  The norm of a root, (-1)^n f(0) = c_n,
    # must be the root of C(p, 1) (the least primitive root), which fixes c_n for n > 1.
    digits = [0] * n
    fixed_last = None
    if n > 1:
        fixed_last = (-conway(p, 1)[0]) % p
        digits[n - 1] = fixed_last
    deadline = time.monotonic() + CONWAY_SECONDS
    while True:
        tried += 1
        if tried > CONWAY_BUDGET or (tried % 64 == 0 and time.monotonic() > deadline):
            raise NotImplementedError(f"FiniteField: the Conway polynomial search for FiniteField[{p}, {n}] is too "
                                      f"large; give an explicit irreducible, FiniteField[p, f]")
        coeffs = [0] * (n + 1)
        coeffs[n] = 1
        for i in range(1, n + 1):
            coeffs[n - i] = ((-1) ** i * digits[i - 1]) % p
        if coeffs[0] and (n == 1 or _no_root(coeffs, p)) and _primitive_monic(coeffs, p, n, qfac):
            ok = True
            N = p ** n - 1
            x = [0, 1] if n > 1 else [(-coeffs[0]) % p]
            for m, cm in subs:
                g = _ppow(x, N // (p ** m - 1), coeffs, p)
                if _peval(cm, g, coeffs, p):
                    ok = False
                    break
            if ok:
                _CONWAY[key] = coeffs
                return coeffs
        # next tuple
        i = n - 1 if fixed_last is None else n - 2
        while i >= 0:
            digits[i] += 1
            if digits[i] < p:
                break
            digits[i] = 0
            i -= 1
        if i < 0:
            raise ValueError(f"FiniteField: no Conway polynomial found for ({p}, {n})")


# ─────────────────────────── fields ───────────────────────────

class Field:
    def __init__(self, p, d, f, rep, expr):
        self.p, self.d, self._f, self.rep, self.expr = p, d, f, rep, expr
        self.default = f is None
        self.q = p ** d
        self._logs = None

    @property
    def f(self):
        """The field irreducible as given (coefficients mod p, low -> high)."""
        if self._f is None:
            self._f = conway(self.p, self.d)
        return self._f

    @property
    def m(self):
        """Monic version of f used for reduction (for d = 1 with the default irreducible: x)."""
        f = self.f
        inv = pow(f[-1], -1, self.p)
        return [(c * inv) % self.p for c in f]

    def key(self):
        if self.default and self.d == 1:
            return (self.p, 1)
        return (self.p, tuple(self.m))

    # elements as coefficient lists
    def reduce(self, cs):
        if self.d > 1:
            return _pmod([c % self.p for c in cs], self.m, self.p)
        return _trim([_eval1(cs, self)])

    def gen(self):
        return self.reduce([0, 1])

    def from_index(self, k):
        if not (0 <= k < self.q):
            raise ValueError(f"FiniteFieldElement: the index {k} is not in the range 0 to {self.q - 1}")
        if self.rep == "Exponential":
            if k == 0:
                return []
            return self.pow(self.gen(), k - 1)
        cs = []
        for _ in range(self.d):
            cs.append(k % self.p)
            k //= self.p
        return _trim(cs)

    def index(self, cs):
        if self.rep == "Exponential":
            if not cs:
                return 0
            return self.log(cs) + 1
        return sum(c * self.p ** i for i, c in enumerate(cs))

    def log(self, cs):
        if self._logs is None:
            g = self.gen()
            tbl, cur = {}, [1]
            for k in range(self.q - 1):
                tbl[tuple(cur)] = k
                cur = self.mul(cur, g)
            self._logs = tbl
        return self._logs[tuple(cs)]

    def mul(self, a, b):
        if self.d == 1:
            return _trim([((a[0] if a else 0) * (b[0] if b else 0)) % self.p])
        return _pmod(_pmul(a, b, self.p), self.m, self.p)

    def inv(self, a):
        if not a:
            raise ZeroDivisionError("FiniteFieldElement: division by zero")
        if self.d == 1:
            return [pow(a[0], -1, self.p)]
        return _pinv(a, self.m, self.p)

    def pow(self, a, e):
        if e < 0:
            a, e = self.inv(a), -e
        if e == 0:
            return [1]
        if not a:
            return []
        if self.d == 1:
            return _trim([pow(a[0], e, self.p)])
        return _ppow(a, e, self.m, self.p)


def _eval1(cs, F):
    """Reduce a coefficient list in degree 1: the generator t is the root -f0/f1 of f (0 for the default)."""
    p = F.p
    cs = _trim([c % p for c in cs])
    if len(cs) <= 1:
        return cs[0] if cs else 0
    m = F.m
    t = (-m[0]) % p
    v = 0
    for c in reversed(cs):
        v = (v * t + c) % p
    return v


_FIELDS = {}


def _poly_of_function(f, p, fname="FiniteField"):
    if not (isinstance(f, Expr) and f.head == "Function"):
        raise ValueError(f"{fname}: the irreducible must be a pure function such as #^2 + 1 &")
    x = Symbol("FiniteField`x$")
    body = ev(K.call(f, x))
    try:
        P = sp.Poly(S(body), S(x))
    except (sp.PolynomialError, sp.GeneratorsNeeded):
        raise ValueError(f"{fname}: the irreducible must be a univariate polynomial function")
    cs = [sp.Rational(c) for c in reversed(P.all_coeffs())]
    out = []
    for c in cs:
        if c.q % p == 0:
            raise ValueError(f"{fname}: the coefficients of the irreducible must be valid modulo {p}")
        out.append(int(c.p * pow(int(c.q), -1, p)) % p)
    return _trim(out)


def _function_of(cs):
    s = Expr("Slot", 1)
    terms = []
    for i, c in enumerate(cs):
        if c:
            m = s if i == 1 else Expr("Power", s, i)
            terms.append(c if i == 0 else (m if c == 1 else Expr("Times", c, m)))
    body = ev(Expr("Plus", *terms)) if terms else 0
    return Expr("Function", body)


def _irreducible(cs, p):
    P = sp.Poly(list(reversed(cs)), sp.Symbol("x"), modulus=p)
    return P.degree() >= 1 and P.is_irreducible


def field_of(ff):
    """Field object for a canonical FiniteField[...] expression (validating it), or None."""
    if not (isinstance(ff, Expr) and ff.head == "FiniteField"):
        return None
    try:
        return _FIELDS[ff]
    except KeyError:
        pass
    except TypeError:
        return None
    a = ff.args
    p = a[0] if a else None
    if not (_intq(p) and sp.isprime(int(p))) or len(a) not in (2, 3):
        return None
    p = int(p)
    rep = "Polynomial"
    if len(a) == 3:
        if a[2] not in ("Polynomial", "Exponential"):
            return None
        rep = a[2]
    if _intq(a[1]):
        d, f = int(a[1]), None
        if d < 1:
            return None
    else:
        f = _poly_of_function(a[1], p)
        d = len(f) - 1
    F = Field(p, d, f, rep, ff)
    if rep == "Exponential":
        if F.q > EXP_LIMIT:
            raise NotImplementedError(f"FiniteField: the \"Exponential\" representation is limited to fields with at "
                                      f"most {EXP_LIMIT} elements; use \"Polynomial\"")
        mm = F.m
        if not _primitive_monic(mm, p, d, list(sp.factorint(F.q - 1))):
            raise ValueError("FiniteField: the \"Exponential\" representation needs a primitive irreducible")
    _FIELDS[ff] = F
    return F


@fn("FiniteField", **ADDED, notes="FiniteField[p, d], FiniteField[q] (q = p^d), FiniteField[p, f] (f an "
    "irreducible pure function), FiniteField[p, d|f, \"Polynomial\"|\"Exponential\"]: the finite field GF(p^d); "
    "ff[k] / ff[{c0, c1, ...}] give elements; Information[ff, prop] gives Characteristic, ExtensionDegree, "
    "FieldSize, FieldIrreducible, ElementRepresentation. The default irreducible is the Conway polynomial"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _finitefield(*a: Expr) -> Expr:
    if not a or len(a) > 3:
        raise ValueError("FiniteField takes 1 to 3 arguments")
    p = a[0]
    if not _intq(p) or int(p) < 2:
        raise ValueError("FiniteField: the first argument must be a prime or a prime power")
    p = int(p)
    rest = list(a[1:])
    rep = None
    if rest and isinstance(rest[-1], str):
        rep = rest.pop()
        if rep not in ("Polynomial", "Exponential"):
            raise ValueError("FiniteField: the representation must be \"Polynomial\" or \"Exponential\"")
    if not sp.isprime(p):
        fac = sp.factorint(p)
        if len(fac) != 1 or rest:
            raise ValueError(f"FiniteField: {p} is not a prime" + ("" if rest else " power"))
        (pp, d), = fac.items()
        return ev(Expr("FiniteField", int(pp), int(d), *([rep] if rep == "Exponential" else [])))
    if not rest:
        rest = [1]
    if len(rest) != 1:
        raise ValueError("FiniteField: invalid arguments")
    spec = rest[0]
    if _intq(spec):
        d = int(spec)
        if d < 1:
            raise ValueError("FiniteField: the extension degree must be a positive integer")
        new = Expr("FiniteField", p, d) if rep in (None, "Polynomial") else Expr("FiniteField", p, d, rep)
    else:
        cs = _poly_of_function(spec, p)
        if len(cs) < 2 or not _irreducible(cs, p):
            raise ValueError(f"FiniteField: the polynomial is not irreducible modulo {p}")
        new = Expr("FiniteField", p, _function_of(cs), rep or "Polynomial")
    if K.same(new, Expr("FiniteField", *a)):
        field_of(new)
        raise Unevaluated()
    return new


def make(F, cs):
    return Expr(FFE, F.expr, L(*cs))


def is_ffe(x):
    return isinstance(x, Expr) and x.head == FFE and len(x.args) == 2 and field_of(x.args[0]) is not None \
        and is_list(x.args[1])


def data(x):
    return field_of(x.args[0]), [int(c) for c in x.args[1].args]


def _coerce_rational(v, p):
    """Integer/rational v modulo p, or None for non-rationals; raises for denominators divisible by p."""
    if isinstance(v, bool):
        return None
    if _intq(v):
        return int(v) % p
    if not isinstance(v, Expr) or v.head not in ("Rational", "Times", "Power"):
        return None
    n = NT.number_of(v)
    if not isinstance(n, sp.Rational):
        return None
    num, den = int(n.p), int(n.q)
    if den % p == 0:
        raise ValueError(f"FiniteFieldElement: the rational number {num}/{den} is not valid modulo {p}")
    return num * pow(den, -1, p) % p


def _element_from(F, spec):
    if _intq(spec):
        return make(F, F.from_index(int(spec)))
    if is_list(spec):
        cs = []
        for c in spec.args:
            v = _coerce_rational(c, F.p)
            if v is None:
                raise ValueError("FiniteFieldElement: the coefficients must be integers")
            cs.append(v)
        return make(F, F.reduce(cs))
    raise ValueError("FiniteFieldElement: the second argument must be an index or a list of coefficients")


@fn(FFE, **ADDED, notes="FiniteFieldElement[ff, ind] (index) or FiniteFieldElement[ff, {c0, c1, ...}] (c0 + c1 t + "
    "...); a[prop] gives Field, Index, Coefficients, Characteristic, ExtensionDegree, FieldSize, FieldIrreducible, "
    "ElementRepresentation; elements of one field combine under +, *, ^ (also with rationals mod p)")
def _ffe(ff: Expr, spec: Expr) -> Expr:
    F = field_of(ev(ff))
    if F is None:
        raise ValueError("FiniteFieldElement: the first argument must be a FiniteField")
    if is_list(spec):
        cs = []
        for c in spec.args:
            v = _coerce_rational(c, F.p)
            if v is None:
                raise ValueError("FiniteFieldElement: the coefficients must be integers")
            cs.append(v)
        red = F.reduce(cs)
        if red == list(int(c) for c in spec.args) and F.expr is ff:
            raise Unevaluated()
        new = make(F, red)
        if K.same(new, Expr(FFE, ff, spec)):
            raise Unevaluated()
        return new
    if not _intq(spec):
        raise Unevaluated()
    return _element_from(F, spec)


def _field_props(F):
    return {"Characteristic": F.p, "ExtensionDegree": F.d, "FieldSize": F.q,
            "FieldIrreducible": lambda: _function_of(F.f), "ElementRepresentation": F.rep}


def _prop(F, name, cs=None):
    if cs is not None:
        if name == "Field":
            return F.expr
        if name == "Index":
            return F.index(cs)
        if name == "Coefficients":
            return L(*(cs + [0] * (F.d - len(cs))))
    props = _field_props(F)
    if name not in props:
        raise ValueError(f"FiniteField: unknown property {name!r}")
    v = props[name]
    return v() if callable(v) else v


def _props(F, spec, cs=None):
    if is_list(spec):
        return L(*(_prop(F, s, cs) for s in spec.args))
    if isinstance(spec, str):
        return _prop(F, spec, cs)
    raise ValueError("FiniteField: the property must be a string")


def _ff_call(head, args, ev_):
    F = field_of(head)
    if F is None:
        return UNH
    if len(args) != 1:
        raise ValueError("ff[k] takes one argument")
    a = args[0]
    if isinstance(a, str) or (is_list(a) and a.args and all(isinstance(s, str) for s in a.args)):
        return _props(F, a)
    if not (_intq(a) or is_list(a)):
        return UNH
    return _element_from(F, a)


def _ffe_call(head, args, ev_):
    if not is_ffe(head) or len(args) != 1:
        return UNH
    F, cs = data(head)
    return _props(F, args[0], cs)


KL.COMPOUND_HEAD_HOOKS["FiniteField"] = _ff_call
KL.COMPOUND_HEAD_HOOKS[FFE] = _ffe_call

_orig_info = K._HELD.get("Information")


def _info(a, ev_, expr):
    if a and len(a) == 2 and isinstance(a[0], (Symbol, Expr)):
        v = ev_(a[0])
        prop = ev_(a[1])
        if isinstance(v, Expr) and v.head == "FiniteField" and field_of(v) is not None:
            return _props(field_of(v), prop)
        if is_ffe(v):
            F, cs = data(v)
            return _props(F, prop, cs)
    return _orig_info(a, ev_, expr)


if _orig_info is not None:
    K._HELD["Information"] = _info


# ─────────────────────────── arithmetic ───────────────────────────

def _same_field(F, G):
    if F is G or F.expr == G.expr:
        return True
    return F.p == G.p and F.d == G.d and F.key() == G.key()


def _sqrt(F, a):
    """A square root of a in F, or None."""
    if not a:
        return []
    q = F.q
    if F.p == 2:
        return F.pow(a, q // 2)
    if F.pow(a, (q - 1) // 2) != [1]:
        return None
    # Tonelli-Shanks
    s, t = 0, q - 1
    while t % 2 == 0:
        s, t = s + 1, t // 2
    z = None
    for k in range(1, q):
        c = F.from_index(k) if F.rep == "Polynomial" else F.from_index(k + 1)
        if c and F.pow(c, (q - 1) // 2) != [1]:
            z = c
            break
    M, c, T, Rr = s, F.pow(z, t), F.pow(a, t), F.pow(a, (t + 1) // 2)
    while T != [1]:
        i, tt = 0, T
        while tt != [1]:
            tt, i = F.mul(tt, tt), i + 1
        b = F.pow(c, 2 ** (M - i - 1))
        M, c, T, Rr = i, F.mul(b, b), F.mul(T, F.mul(b, b)), F.mul(Rr, b)
    # canonical choice: the root with the smaller index
    other = _padd([], _pscale(Rr, F.p - 1, F.p), F.p)
    return min(Rr, other, key=F.index)


def _root_p(F, a, k):
    """The unique p^k-th root of a: a^(p^(k (d-1)))... i.e. the inverse Frobenius power."""
    d = F.d
    e = (d - k % d) % d
    return F.pow(a, F.p ** e) if e else a


def _rat_power(F, a, n, dd):
    """a^(n/dd) for dd = 2^e p^k (e <= 1); None if not available."""
    k = 0
    while dd % F.p == 0:
        dd //= F.p
        k += 1
    b = _root_p(F, a, k) if k else a
    if dd == 2:
        b = _sqrt(F, b)
        if b is None:
            return None
    elif dd != 1:
        return None
    return F.pow(b, n)


def _ff_arith(expr, ev_):
    h = expr.head
    args = list(expr.args)
    ffs = [x for x in args if is_ffe(x)]
    if not ffs and h == "Plus":
        ffs = [f for t in args if isinstance(t, Expr) and t.head == "Times" for f in t.args if is_ffe(f)]
    if not ffs:
        return UNH
    F = field_of(ffs[0].args[0])
    for x in ffs[1:]:
        if not _same_field(F, field_of(x.args[0])):
            raise ValueError("FiniteFieldElement: elements of different finite fields cannot be combined")
    if h == "Power":
        if len(args) != 2 or not is_ffe(args[0]):
            return UNH
        _, cs = data(args[0])
        e = args[1]
        if _intq(e):
            return make(F, F.pow(cs, int(e)))
        en = NT.number_of(e) if isinstance(e, Expr) else (e if isinstance(e, sp.Rational) else None)
        if isinstance(en, sp.Rational):
            r = _rat_power(F, cs, int(en.p), int(en.q))
            return UNH if r is None else make(F, r)
        return UNH
    if h == "Sqrt":
        if len(args) != 1:
            return UNH
        r = _sqrt(F, data(args[0])[1])
        return UNH if r is None else make(F, r)
    if h == "Plus":
        r = _plus_terms(F, args)
        if r is not UNH:
            return r
    if h in ("Plus", "Times"):
        acc = [] if h == "Plus" else [1]
        rest, used = [], 0
        for x in args:
            if is_ffe(x):
                cs = data(x)[1]
            else:
                v = _coerce_rational(x, F.p)
                if v is None:
                    rest.append(x)
                    continue
                cs = _trim([v])
            used += 1
            acc = _padd(acc, cs, F.p) if h == "Plus" else F.mul(acc, cs)
        if used < 2:
            return UNH
        el = make(F, acc)
        if not rest:
            return el
        if h == "Plus" and not acc:
            return Expr("Plus", *rest) if len(rest) > 1 else rest[0]
        if h == "Times" and not acc:
            return el
        if h == "Times" and acc == [1] and F.d >= 1:
            return Expr("Times", el, *rest)
        return Expr(h, el, *rest)
    if h in ("Equal", "Unequal") and len(args) >= 2:
        vals = []
        for x in args:
            if is_ffe(x):
                vals.append(data(x)[1])
            else:
                v = _coerce_rational(x, F.p)
                if v is None:
                    return UNH
                vals.append(_trim([v]))
        eq = all(v == vals[0] for v in vals)
        if h == "Equal":
            return eq
        return len({tuple(v) for v in vals}) == len(vals)
    return UNH


def _split_term(F, t):
    """(coefficient list, rest factors tuple) of a Plus term, or None when it has no field coefficient form."""
    if is_ffe(t):
        return data(t)[1], ()
    v = _coerce_rational(t, F.p)
    if v is not None:
        return _trim([v]), ()
    if isinstance(t, Expr) and t.head == "Times":
        coef, rest = [1], []
        for f in t.args:
            if is_ffe(f):
                coef = F.mul(coef, data(f)[1])
            else:
                v = _coerce_rational(f, F.p)
                if v is not None:
                    coef = F.mul(coef, _trim([v]))
                else:
                    rest.append(f)
        return coef, tuple(rest)
    return [1], (t,)


def _plus_terms(F, args):
    """Collect like terms c1 m + c2 m (field coefficients) in a sum; UNH when nothing combines."""
    groups, order = {}, []
    for t in args:
        c, rest = _split_term(F, t)
        k = tuple(rest)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(c)
    if len(order) == len(args) and not any(isinstance(t, Expr) and t.head == "Times" and
                                           sum(1 for f in t.args if is_ffe(f) or _coerce_rational(f, F.p) is not None)
                                           > 1 for t in args):
        return UNH
    terms = []
    for k in order:
        acc = []
        for c in groups[k]:
            acc = _padd(acc, c, F.p)
        if not k:
            if acc or len(order) == 1:
                terms.append(make(F, acc))
            continue
        if not acc:
            continue
        rest = list(k)
        terms.append(Expr("Times", make(F, acc), *rest))
    if not terms:
        return make(F, [])
    return terms[0] if len(terms) == 1 else Expr("Plus", *terms)


NT.ARITH_HANDLERS.append(_ff_arith)


def _power_builtin(a, ev_, expr):
    if len(a) == 2 and is_ffe(a[0]):
        return Expr("Power", *a)        # a power the field arithmetic cannot do stays unevaluated
    raise NT.NOT_MINE()


NT.wrap_builtin("Power", _power_builtin)


def _sqrt_pre(args):
    if len(args) == 1 and is_ffe(args[0]):
        F, cs = data(args[0])
        r = _sqrt(F, cs)
        if r is None:
            raise Unevaluated()
        return make(F, r)
    raise NT.NOT_MINE()


NT.wrap_registry("Sqrt", _sqrt_pre)


# ─────────────────────────── Expand over a finite field ───────────────────────────

def _poly_of(F, e):
    """Polynomial as {monomial: coeffs}, monomial a sorted tuple of (factor, exponent)."""
    if is_ffe(e):
        c = data(e)[1]
        return {(): c} if c else {}
    v = _coerce_rational(e, F.p)
    if v is not None:
        return {(): [v]} if v else {}
    if isinstance(e, Expr) and e.head == "Plus":
        r = {}
        for t in e.args:
            r = _poly_add(F, r, _poly_of(F, t))
        return r
    if isinstance(e, Expr) and e.head == "Times":
        r = {(): [1]}
        for t in e.args:
            r = _poly_mul(F, r, _poly_of(F, t))
        return r
    if isinstance(e, Expr) and e.head == "Power" and len(e.args) == 2 and _intq(e.args[1]) and int(e.args[1]) > 0 \
            and _contains_ffe(e.args[0]) or (isinstance(e, Expr) and e.head == "Power" and len(e.args) == 2 and
                                             _intq(e.args[1]) and int(e.args[1]) > 0 and
                                             isinstance(e.args[0], Expr) and e.args[0].head == "Plus"):
        b = _poly_of(F, e.args[0])
        r = {(): [1]}
        for _ in range(int(e.args[1])):
            r = _poly_mul(F, r, b)
        return r
    if isinstance(e, Expr) and e.head == "Power" and len(e.args) == 2 and _intq(e.args[1]):
        return {((e.args[0], int(e.args[1])),): [1]}
    if _contains_ffe(e):
        raise NotImplementedError("Expand: only polynomials with finite field coefficients can be expanded")
    return {((e, 1),): [1]}


def _mono_mul(m1, m2):
    d = {}
    keys = []
    for b, k in m1 + m2:
        for kk in keys:
            if K.same(kk, b):
                d[id(kk)] = (kk, d[id(kk)][1] + k)
                break
        else:
            keys.append(b)
            d[id(b)] = (b, k)
    out = [v for v in d.values() if v[1] != 0]
    out.sort(key=lambda v: (K.order_key(v[0]), v[1]))
    return tuple(out)


def _poly_add(F, a, b):
    r = dict(a)
    for m, c in b.items():
        k = next((x for x in r if _mono_eq(x, m)), m)
        r[k] = _padd(r.get(k, []), c, F.p)
        if not r[k]:
            del r[k]
    return r


def _mono_eq(a, b):
    return len(a) == len(b) and all(K.same(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))


def _poly_mul(F, a, b):
    r = {}
    for m1, c1 in a.items():
        for m2, c2 in b.items():
            r = _poly_add(F, r, {_mono_mul(m1, m2): F.mul(c1, c2)})
    return r


def _first_field(e, depth=0):
    if is_ffe(e):
        return field_of(e.args[0])
    if isinstance(e, Expr) and depth < 60:
        for a in e.args:
            F = _first_field(a, depth + 1)
            if F is not None:
                return F
    return None


def _expand_pre(args):
    if len(args) != 1 or not _contains_ffe(args[0]):
        raise NT.NOT_MINE()
    F = _first_field(args[0])
    P = _poly_of(F, args[0])
    terms = []
    for m, c in P.items():
        fs = [b if k == 1 else Expr("Power", b, k) for b, k in m]
        terms.append(Expr("Times", make(F, c), *fs) if fs else make(F, c))
    if not terms:
        return make(F, [])
    return ev(Expr("Plus", *terms))


NT.wrap_registry("Expand", _expand_pre)

_GUARDED = ["Factor", "ExpandAll", "FactorList", "PolynomialGCD", "PolynomialLCM", "PolynomialQuotient", "PolynomialRemainder",
            "PolynomialQuotientRemainder", "PolynomialMod", "Cancel", "Together", "Apart", "Resultant",
            "Discriminant", "Det", "Inverse", "RowReduce", "NullSpace", "MatrixRank", "LinearSolve", "MatrixPower",
            "LUDecomposition", "CharacteristicPolynomial", "Eigenvalues", "Eigenvectors", "Solve", "NSolve",
            "Reduce", "FindInstance", "Resolve", "Simplify", "FullSimplify", "MinimalPolynomial",
            "MultiplicativeOrder", "N", "Roots", "Integrate", "D", "Series", "Limit", "Sum", "Product"]


def _contains_ffe(x, depth=0):
    if is_ffe(x):
        return True
    if isinstance(x, Expr) and depth < 40:
        return any(_contains_ffe(a, depth + 1) for a in x.args)
    return False


def _make_guard(name, prev):
    def g(expr):
        if any(_contains_ffe(a) for a in expr.args):
            raise NotImplementedError(f"{name}: computations over finite fields (FiniteFieldElement arguments) are "
                                      f"not implemented in MikoshiLang")
        if prev is not None:
            prev(expr)
    return g


for _n in _GUARDED:
    NT.GUARDS[_n] = _make_guard(_n, NT.GUARDS.get(_n))


# ─────────────────────────── functions of elements ───────────────────────────

def _need_ffe(a, fname):
    if not is_ffe(a):
        raise ValueError(f"{fname}: the argument must be a FiniteFieldElement")
    return data(a)


def _frob(F, cs, k):
    k %= F.d
    return F.pow(cs, F.p ** k) if k else cs


@fn("FrobeniusAutomorphism", **ADDED, notes="FrobeniusAutomorphism[a] = a^p; FrobeniusAutomorphism[a, k] is the k-th "
    "functional power a^(p^k) (k taken modulo the extension degree, negative k allowed)")
def _frobenius(a: Expr, k: Expr = 1) -> Expr:
    F, cs = _need_ffe(a, "FrobeniusAutomorphism")
    if not _intq(k):
        if isinstance(k, (Symbol, Expr)) and not isinstance(k, bool):
            raise Unevaluated()
        raise ValueError("FrobeniusAutomorphism: the power must be an integer")
    return make(F, _frob(F, cs, int(k)))


def _sub_degree(F, k, fname):
    if is_ffe_embedding(k):
        raise UnevaluatedEmb()
    if not _intq(k) or int(k) < 1 or F.d % int(k):
        raise ValueError(f"{fname}: the subfield degree must be a positive divisor of {F.d}")
    return int(k)


class UnevaluatedEmb(Exception):
    pass


def _norm_cs(F, cs, k):
    n = F.d // k
    r = [1]
    for i in range(n):
        r = F.mul(r, _frob(F, cs, i * k))
    return r


def _trace_cs(F, cs, k):
    n = F.d // k
    r = []
    for i in range(n):
        r = _padd(r, _frob(F, cs, i * k), F.p)
    return r


def _norm_trace(a, k, fname, op):
    F, cs = _need_ffe(a, fname)
    if k is None:
        r = op(F, cs, 1)
        return r[0] if r else 0
    if is_ffe_embedding(k):
        E = emb_of(k)
        if not _same_field(E.F2, F):
            raise ValueError(f"{fname}: the element must belong to the target field of the embedding")
        r = op(F, cs, E.F1.d)
        return make(E.F1, E.project(r))
    kk = _sub_degree(F, k, fname)
    r = op(F, cs, kk)
    sub = field_of(ev(Expr("FiniteField", F.p, kk)))
    E = Embedding(sub, F)
    return make(sub, E.project(r))


@fn("FiniteFieldElementNorm", **ADDED, notes="FiniteFieldElementNorm[a] is the absolute norm (an integer 0..p-1, "
    "the product of the conjugates of a); [a, k] the norm relative to the p^k-element subfield (an element of "
    "FiniteField[p, k], identified through the standard embedding); [a, emb] relative to an embedding"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _ffnorm(a: Expr, k: Expr = None) -> Expr:
    return _norm_trace(a, k, "FiniteFieldElementNorm", _norm_cs)


@fn("FiniteFieldElementTrace", **ADDED, notes="FiniteFieldElementTrace[a] is the absolute trace (an integer 0..p-1, "
    "the sum of the conjugates); [a, k] relative to the p^k-element subfield; [a, emb] relative to an embedding"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fftrace(a: Expr, k: Expr = None) -> Expr:
    return _norm_trace(a, k, "FiniteFieldElementTrace", _trace_cs)


def _mult_order(F, cs):
    N = F.q - 1
    o = N
    for r, e in sp.factorint(N).items():
        for _ in range(e):
            if F.pow(cs, o // r) == [1]:
                o //= r
            else:
                break
    return o


@fn("FiniteFieldElementPrimitiveQ", **ADDED, notes="True if the FiniteFieldElement a generates the multiplicative "
    "group of its field (order q - 1); False for anything else"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _ffprimq(a: Expr) -> Boolean:
    if not is_ffe(a):
        return False
    F, cs = data(a)
    if not cs:
        return False
    return _mult_order(F, cs) == F.q - 1


@fn("FiniteFieldIndex", **ADDED, listable=True, notes="FiniteFieldIndex[u] = u[\"Index\"] (Listable): sum c_i p^i "
    "for the polynomial representation; k + 1 for t^k in the exponential representation; 0 for zero"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _ffindex(u: Expr) -> Expr:
    F, cs = _need_ffe(u, "FiniteFieldIndex")
    return F.index(cs)


@fn("FromFiniteFieldIndex", **ADDED, notes="FromFiniteFieldIndex[ind, ff] = ff[ind]; listable in ind (vectors and "
    "matrices of indices)"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fromffindex(ind: Expr, ff: Expr) -> Expr:
    F = field_of(ff)
    if F is None:
        raise ValueError("FromFiniteFieldIndex: the second argument must be a FiniteField")

    def go(x):
        if is_list(x):
            return L(*(go(y) for y in x.args))
        if not _intq(x):
            raise ValueError("FromFiniteFieldIndex: the indices must be integers")
        return make(F, F.from_index(int(x)))
    return go(ind)


def _map_coeffs(expr, conv, fname, depth=0):
    """Rebuild expr through List/Plus/Times/integer Power, converting leaves with conv (None: keep)."""
    r = conv(expr)
    if r is not None:
        return r
    if isinstance(expr, Expr) and expr.head in ("List", "Plus", "Times"):
        return Expr(expr.head, *(_map_coeffs(a, conv, fname, depth + 1) for a in expr.args))
    if isinstance(expr, Expr) and expr.head == "Power" and len(expr.args) == 2 and _intq(expr.args[1]):
        return Expr("Power", _map_coeffs(expr.args[0], conv, fname, depth + 1), expr.args[1])
    return expr


@fn("ToFiniteField", **ADDED, notes="ToFiniteField[k, ff]: integers and rationals (mod p) become prime-field "
    "elements ff[{k}]; ToFiniteField[expr, ff] maps the rational coefficients inside List, Plus, Times and integer "
    "Power; ToFiniteField[expr, ff, t] also replaces t by the field generator ff[{0, 1}]"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _toff(expr: Expr, ff: Expr, t: Expr = None) -> Expr:
    F = field_of(ff)
    if F is None:
        raise ValueError("ToFiniteField: the second argument must be a FiniteField")

    def conv(x):
        v = _coerce_rational(x, F.p)
        if v is not None:
            return make(F, _trim([v]))
        if t is not None and K.same(x, t):
            return make(F, F.gen())
        return None
    return ev(_map_coeffs(expr, conv, "ToFiniteField"))


@fn("FromFiniteField", **ADDED, notes="FromFiniteField[a, ff] turns prime-subfield elements into integers 0..p-1 "
    "(inside List, Plus, Times, integer Power); elements outside the prime field stay; FromFiniteField[expr, ff, t] "
    "writes every element as c0 + c1 t + ... in the generator t"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fromff(expr: Expr, ff: Expr, t: Expr = None) -> Expr:
    F = field_of(ff)
    if F is None:
        raise ValueError("FromFiniteField: the second argument must be a FiniteField")

    def conv(x):
        if is_ffe(x):
            G, cs = data(x)
            if not _same_field(F, G):
                return None
            if t is None:
                if len(cs) <= 1:
                    return cs[0] if cs else 0
                return None
            terms = [c if i == 0 else Expr("Times", c, Expr("Power", t, i)) for i, c in enumerate(cs) if c]
            return Expr("Plus", *terms) if len(terms) > 1 else (terms[0] if terms else 0)
        return None
    return ev(_map_coeffs(expr, conv, "FromFiniteField"))


# ─────────────────────────── embeddings ───────────────────────────

def _solve_mod(M, b, p):
    """Solve M x = b over GF(p) (M square invertible), lists of lists."""
    n = len(M)
    A = [row[:] + [b[i]] for i, row in enumerate(M)]
    for c in range(n):
        piv = next(r for r in range(c, n) if A[r][c] % p)
        A[c], A[piv] = A[piv], A[c]
        inv = pow(A[c][c], -1, p)
        A[c] = [x * inv % p for x in A[c]]
        for r in range(n):
            if r != c and A[r][c]:
                f = A[r][c]
                A[r] = [(x - f * y) % p for x, y in zip(A[r], A[c])]
    return [A[i][n] for i in range(n)]


def _vec(cs, n):
    return [cs[i] if i < len(cs) else 0 for i in range(n)]


class Embedding:
    """Embedding of F1 in F2 mapping the element e1 (coefficients) to e2."""

    def __init__(self, F1, F2, e1=None, e2=None):
        if F1.p != F2.p or F2.d % F1.d:
            raise ValueError("FiniteFieldEmbedding: the first field must have the same characteristic as the second "
                             "and an extension degree dividing it")
        self.F1, self.F2 = F1, F2
        if e1 is None:
            if F1.d == 1:
                e1 = e2 = [1]
            else:
                e1 = F1.gen()
                e2 = self._default_image(e1)
        self.e1, self.e2 = e1, e2
        d1 = F1.d
        # basis e1^i of F1 over GF(p) and its images e2^i
        self.pows1 = [F1.pow(e1, i) for i in range(d1)]
        self.pows2 = [F2.pow(e2, i) for i in range(d1)]
        cols = [_vec(v, d1) for v in self.pows1]
        self.M1 = [[cols[j][i] for j in range(d1)] for i in range(d1)]
        # complement for the projection: images extended greedily by standard basis vectors
        d2 = F2.d
        basis = [_vec(v, d2) for v in self.pows2]
        for i in range(d2):
            if len(basis) == d2:
                break
            cand = basis + [[1 if j == i else 0 for j in range(d2)]]
            if _rank(cand, F2.p) == len(cand):
                basis = cand
        self.B2 = [[basis[j][i] for j in range(d2)] for i in range(d2)]

    def _default_image(self, e1):
        F1, F2 = self.F1, self.F2
        mp = _minpoly(F1, e1)
        if F1.default and F2.default:
            # Conway compatibility: theta2^((q2-1)/(q1-1)) is a root of C(p, d1), the minimal polynomial of theta1
            g = F2.pow(F2.gen(), (F2.q - 1) // (F1.q - 1))
            if not _peval(mp, g, F2.m, F2.p):
                return g
        # the root of the minimal polynomial with the smallest index
        roots = _roots(F2, mp)
        if not roots:
            raise ValueError("FiniteFieldEmbedding: no embedding exists")
        return min(roots, key=F2.index)

    def map(self, cs):
        d1 = self.F1.d
        x = _solve_mod(self.M1, _vec(cs, d1), self.F1.p)
        r = []
        for c, v in zip(x, self.pows2):
            r = _padd(r, _pscale(v, c, self.F2.p), self.F2.p)
        return r

    def project(self, cs):
        d1, d2 = self.F1.d, self.F2.d
        x = _solve_mod(self.B2, _vec(cs, d2), self.F2.p)[:d1]
        r = []
        for c, v in zip(x, self.pows1):
            r = _padd(r, _pscale(v, c, self.F1.p), self.F1.p)
        return r

    def expr(self):
        return Expr("FiniteFieldEmbedding", Expr("Rule", make(self.F1, self.e1), make(self.F2, self.e2)))


def _rank(rows, p):
    A = [r[:] for r in rows]
    rk, cols = 0, len(A[0]) if A else 0
    for c in range(cols):
        piv = next((r for r in range(rk, len(A)) if A[r][c] % p), None)
        if piv is None:
            continue
        A[rk], A[piv] = A[piv], A[rk]
        inv = pow(A[rk][c], -1, p)
        A[rk] = [x * inv % p for x in A[rk]]
        for r in range(len(A)):
            if r != rk and A[r][c]:
                f = A[r][c]
                A[r] = [(x - f * y) % p for x, y in zip(A[r], A[rk])]
        rk += 1
    return rk


def _minpoly(F, cs):
    """Minimal polynomial over GF(p) (monic, low -> high) of the element cs."""
    conj = [cs]
    while True:
        nxt = F.pow(conj[-1], F.p)
        if nxt == cs:
            break
        conj.append(nxt)
    poly = [[1]]                           # coefficients are field elements
    for c in conj:
        neg = _pscale(c, F.p - 1, F.p)
        new = [[] for _ in range(len(poly) + 1)]
        for i, a in enumerate(poly):
            new[i + 1] = _padd(new[i + 1], a, F.p)
            new[i] = _padd(new[i], F.mul(a, neg), F.p)
        poly = new
    out = []
    for a in poly:
        if len(a) > 1:
            raise ArithmeticError("minimal polynomial has non-prime-field coefficients")
        out.append(a[0] if a else 0)
    return out


def _roots(F, poly):
    """Roots in F of a polynomial over GF(p) (brute force for small fields, else via SymPy factoring)."""
    if F.q <= 200000:
        out = []
        for k in range(F.q):
            x = F.from_index(k)
            v = []
            for c in reversed(poly):
                v = _padd(F.mul(v, x), _trim([c]), F.p)
            if not v:
                out.append(x)
        return out
    raise NotImplementedError("FiniteFieldEmbedding: the target field is too large for the root search; give the "
                              "embedding as FiniteFieldEmbedding[e1 -> e2]")


_EMBS = {}


def is_ffe_embedding(x):
    return isinstance(x, Expr) and x.head == "FiniteFieldEmbedding" and len(x.args) == 1 and \
        isinstance(x.args[0], Expr) and x.args[0].head == "Rule" and all(is_ffe(y) for y in x.args[0].args)


def emb_of(x):
    try:
        return _EMBS[x]
    except KeyError:
        pass
    e1, e2 = x.args[0].args
    F1, c1 = data(e1)
    F2, c2 = data(e2)
    E = Embedding(F1, F2, c1, c2)
    _EMBS[x] = E
    return E


@fn("FiniteFieldEmbedding", **ADDED, notes="FiniteFieldEmbedding[ff1, ff2] gives an embedding of ff1 in ff2 "
    "(as FiniteFieldEmbedding[e1 -> e2], e1 the generator of ff1; for two default (Conway) fields the standard "
    "compatible embedding, otherwise the root of the minimal polynomial with the smallest index); "
    "FiniteFieldEmbedding[e1 -> e2] needs e1 to generate ff1 and e2 to share its minimal polynomial; emb[a] maps "
    "elements, emb[\"Projection\"] is a GF(p)-linear left inverse"
    ". Check: reference (differential cases in tests/library).")
def _ffemb(*a: Expr) -> Expr:
    if len(a) == 2:
        F1, F2 = field_of(a[0]), field_of(a[1])
        if F1 is None or F2 is None:
            raise ValueError("FiniteFieldEmbedding: the arguments must be finite fields")
        E = Embedding(F1, F2)
        x = E.expr()
        _EMBS[x] = E
        return x
    if len(a) == 1 and isinstance(a[0], Expr) and a[0].head == "Rule" and all(is_ffe(y) for y in a[0].args):
        e1, e2 = a[0].args
        F1, c1 = data(e1)
        F2, c2 = data(e2)
        if F1.p != F2.p or F2.d % F1.d:
            raise ValueError("FiniteFieldEmbedding: the fields are not compatible")
        mp = _minpoly(F1, c1)
        if len(mp) - 1 != F1.d:
            raise ValueError("FiniteFieldEmbedding: the first element does not generate its field")
        v = []
        for c in reversed(mp):
            v = _padd(F2.mul(v, c2), _trim([c]), F2.p)
        if v:
            raise ValueError("FiniteFieldEmbedding: the elements do not have the same minimal polynomial")
        emb_of(Expr("FiniteFieldEmbedding", *a))
        raise Unevaluated()
    raise ValueError("FiniteFieldEmbedding takes two finite fields or a rule e1 -> e2")


def _emb_call(head, args, ev_):
    if not is_ffe_embedding(head) or len(args) != 1:
        return UNH
    E = emb_of(head)
    a = args[0]
    if a == "Projection":
        return Expr("FiniteFieldEmbedding", head.args[0], "Projection")
    if isinstance(a, str):
        props = {"DomainField": E.F1.expr, "CodomainField": E.F2.expr}
        if a in props:
            return props[a]
        raise ValueError(f"FiniteFieldEmbedding: unknown property {a!r}")
    return _map_through(a, lambda x: _emb_apply(E, x))


def _emb_apply(E, x):
    if is_ffe(x):
        F, cs = data(x)
        if not _same_field(F, E.F1):
            raise ValueError("FiniteFieldEmbedding: the element does not belong to the domain field")
        return make(E.F2, E.map(cs))
    v = _coerce_rational(x, E.F1.p)
    if v is not None:
        return make(E.F2, _trim([v]))
    return None


def _map_through(a, f):
    r = f(a)
    if r is not None:
        return r
    if is_list(a):
        return L(*(_map_through(y, f) for y in a.args))
    raise ValueError("FiniteFieldEmbedding: the argument must be a finite field element (or a list of them)")


def _proj_call(head, args, ev_):
    # FiniteFieldEmbedding[e1 -> e2, "Projection"][x]
    if not (isinstance(head, Expr) and len(head.args) == 2 and head.args[1] == "Projection"):
        return UNH
    inner = Expr("FiniteFieldEmbedding", head.args[0])
    if not is_ffe_embedding(inner) or len(args) != 1:
        return UNH
    E = emb_of(inner)

    def f(x):
        if is_ffe(x):
            F, cs = data(x)
            if not _same_field(F, E.F2):
                raise ValueError("FiniteFieldEmbedding: the element does not belong to the codomain field")
            return make(E.F1, E.project(cs))
        v = _coerce_rational(x, E.F1.p)
        return make(E.F1, _trim([v])) if v is not None else None
    return _map_through(args[0], f)


def _emb_hook(head, args, ev_):
    r = _proj_call(head, args, ev_)
    if r is not UNH:
        return r
    return _emb_call(head, args, ev_)


KL.COMPOUND_HEAD_HOOKS["FiniteFieldEmbedding"] = _emb_hook


from .. import math_ops as _MO                      # noqa: E402
_MO.NON_ALGEBRAIC_HEADS = frozenset(_MO.NON_ALGEBRAIC_HEADS | {FFE, "FiniteField", "FiniteFieldEmbedding"})
