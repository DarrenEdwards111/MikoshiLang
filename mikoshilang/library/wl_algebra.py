"""Coverage batch 4: polynomial algebra, simplification, solving and reduction (Wolfram semantics).

Every function either returns a result checked against its definition (by construction with SymPy's exact
polynomial algorithms, or by substituting back / numerically comparing), raises a clear error, or stays
unevaluated.  Known representational limit: the kernel (like SymPy) distributes a rational number over a sum,
so 2 (x^2 - 2) displays as -4 + 2 x^2; FactorTermsList keeps the factors apart.
"""

from __future__ import annotations
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import (fn, record, R, L, rule, sym, variable, is_list, Unevaluated, numeric_out, nonneg_int, int_arg,
                   list_arg)
from .wl_symbolic import AREA_ALG as AREA, kernel_override, replace, out, split_opts, is_true, is_false
from . import algebra as _AL


def _ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def _vars_opt(v, fname):
    if v is None:
        return None
    if is_list(v):
        return [variable(x, fname) for x in v.args]
    return [variable(v, fname)]


def _gens(p, vs):
    """Polynomial generators: given variables, or SymPy's generators of the expression."""
    if vs:
        return vs
    try:
        return list(sp.Poly(p).gens)
    except Exception:
        return sorted(p.free_symbols, key=sp.default_sort_key)


# ───────────────────────────── expansion ─────────────────────────────

def _expand_top(e, x=None):
    """Wolfram Expand: distribute products and positive integer powers over sums at the top level (through
    nested sums, products and powers) but not inside function arguments or denominators. With x, only parts
    containing x are expanded (factors free of x are kept together)."""
    if isinstance(e, sp.Add):
        return sp.Add(*(_expand_top(a, x) for a in e.args))
    if isinstance(e, sp.Mul):
        num, den = [], []
        for f in e.args:
            if isinstance(f, sp.Pow) and f.exp.is_negative:
                den.append(f)
            else:
                num.append(f)
        if x is not None:
            free = [f for f in num if not f.has(x)]
            dep = [f for f in num if f.has(x)]
            prod = sp.Integer(1)
            for f in dep:
                prod = sp.expand_mul(prod * _expand_top(f, x), deep=False)
            block = sp.Mul(*free) * sp.Mul(*den)
            return sp.Add(*(block * t for t in sp.Add.make_args(sp.expand_mul(prod, deep=False))))
        prod = sp.Integer(1)
        for f in num:
            prod = sp.expand_mul(prod * _expand_top(f, x), deep=False)
        dd = sp.Mul(*den)
        return sp.Add(*(t * dd for t in sp.Add.make_args(prod)))
    if isinstance(e, sp.Pow) and e.exp.is_Integer and e.exp > 1 and (x is None or e.base.has(x)):
        b = _expand_top(e.base, x)
        return sp.expand_multinomial(b ** e.exp, deep=False) if isinstance(b, sp.Add) else \
            sp.expand_mul(b ** e.exp, deep=False)
    return e


def _expand_all(e, x=None):
    if isinstance(e, sp.Basic) and e.args and not isinstance(e, (sp.Number,)):
        e = e.func(*(_expand_all(a, x) for a in e.args)) if not isinstance(e, sp.Integral) else e
    if isinstance(e, sp.Pow) and e.exp.is_negative:
        return _expand_top(e.base, x) ** e.exp
    return _expand_top(e, x)


def _thread(e, f):
    if is_list(e):
        return L(*(f(x) for x in e.args))
    if isinstance(e, Expr) and e.head in ("Equal", "Unequal", "Less", "Greater", "LessEqual", "GreaterEqual"):
        return Expr(e.head, *(f(x) for x in e.args))
    return None


def _patt_var(patt, fname):
    if len(patt) > 1:
        raise ValueError(f"{fname} takes at most one pattern argument")
    if not patt:
        return None
    if isinstance(patt[0], Symbol):
        return sp.Symbol(patt[0].name)
    raise NotImplementedError(f"{fname}: only a symbol is supported as the pattern argument")


@replace("Expand", AREA,
         "Wolfram Expand: products and positive integer powers are distributed through nested sums and products "
         "((x + 1)^2 (y + 1) -> 1 + 2 x + x^2 + y + 2 x y + x^2 y; legacy left (1 + x)^2 unexpanded inside "
         "products), not inside function arguments or denominators; Expand[e, x] expands only the parts "
         "containing x; threads over lists and equations; Modulus/Trig options raise")
def _expand(e, *patt):
    patt, _ = split_opts(patt, set(), "Expand")
    th = _thread(e, lambda y: _expand(y, *patt))
    if th is not None:
        return th
    return out(_expand_top(sym(e, "Expand"), _patt_var(patt, "Expand")))


@fn("ExpandAll", area=AREA, status="added",
    notes="ExpandAll[e]: Expand applied to every subexpression, including function arguments and denominators "
          "(ExpandAll[Sin[(x + 1)^2]] -> Sin[1 + 2 x + x^2]); ExpandAll[e, x] only parts containing x"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _expandall(e: Expr, *patt: Expr) -> Expr:
    th = _thread(e, lambda y: _expandall(y, *patt))
    if th is not None:
        return th
    return out(_expand_all(sym(e, "ExpandAll"), _patt_var(patt, "ExpandAll")))


@fn("ExpandNumerator", area=AREA, status="added",
    notes="expands the numerator of a rational expression (Numerator/Denominator as Wolfram), denominator kept"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _expandnum(e: Expr) -> Expr:
    th = _thread(e, _expandnum)
    if th is not None:
        return th
    n, d = sp.fraction(sym(e, "ExpandNumerator"))
    return out(_expand_top(n) / d)


@fn("ExpandDenominator", area=AREA, status="added",
    notes="expands the denominator of a rational expression (ExpandDenominator[1/((x + 1) (x - 1))] -> "
          "1/(-1 + x^2)), numerator kept"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _expandden(e: Expr) -> Expr:
    th = _thread(e, _expandden)
    if th is not None:
        return th
    n, d = sp.fraction(sym(e, "ExpandDenominator"))
    dd = _expand_top(d)
    return out(sp.Mul(n, sp.Pow(dd, -1, evaluate=False), evaluate=False) if dd != 1 else n)


# ───────────────────────────── structure ─────────────────────────────

@fn("PolynomialQ", area=AREA, status="added",
    notes="PolynomialQ[e, x] / PolynomialQ[e, {x, ...}]: True if e is a polynomial in the variables (coefficients "
          "may be anything free of them; nonnegative integer powers only); PolynomialQ[e] in all its variables"
          ". Check: reference (differential cases in tests/library).")
def _polyq(e: Expr, v: Var | ListOf[Var] = None) -> Boolean:
    try:
        p = sym(e, "PolynomialQ")
    except Exception:
        return False
    vs = _vars_opt(v, "PolynomialQ")
    if vs is None:
        if p.has(sp.Function) and not p.is_number:
            gens = list(p.atoms(sp.Function))
            try:
                return bool(sp.Poly(p) is not None) and all(
                    isinstance(g, (sp.Symbol, sp.Function)) or (isinstance(g, sp.Pow) and g.exp.is_Integer)
                    for g in sp.Poly(p).gens) and not any(isinstance(g, sp.Pow) for g in sp.Poly(p).gens)
            except Exception:
                return False
        return bool(p.is_polynomial(*sorted(p.free_symbols, key=str))) if p.free_symbols else True
    return bool(p.is_polynomial(*vs))


def _variables(p):
    n, d = sp.fraction(sp.together(p))
    gens = set()
    for q in (n, d):
        if q.is_number:
            continue
        try:
            for g in sp.Poly(q).gens:
                if isinstance(g, sp.Pow) and g.exp.is_Rational:
                    g = g.base
                gens.add(g)
        except Exception:
            gens |= q.free_symbols
    return sorted(gens, key=sp.default_sort_key)


@fn("Variables", area=AREA, status="added",
    notes="Variables[poly]: the independent variables (symbols and non-polynomial subexpressions such as "
          "Sin[z]) of a polynomial or rational expression, sorted (Variables[x^2 + y Sin[z]] -> {x, y, Sin[z]})"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _variables_fn(e: Expr) -> Expr:
    if is_list(e):
        vs = set()
        for x in e.args:
            vs |= set(_variables(sym(x, "Variables")))
        return out(sorted(vs, key=sp.default_sort_key))
    return out(_variables(sym(e, "Variables")))


_ORDERS = {"Lexicographic": "lex", "DegreeLexicographic": "grlex", "DegreeReverseLexicographic": "grevlex"}


def _order(o, fname):
    if o is None:
        return "lex"
    name = o if isinstance(o, str) else (o.name if isinstance(o, Symbol) else None)
    if name not in _ORDERS:
        raise NotImplementedError(f"{fname}: monomial order {o!r} is not supported "
                                  "(Lexicographic, DegreeLexicographic, DegreeReverseLexicographic)")
    return _ORDERS[name]


def _poly(e, v, fname):
    p = sp.expand(sym(e, fname))
    gens = _gens(p, _vars_opt(v, fname))
    if not p.is_polynomial(*gens):
        raise ValueError(f"{fname}: expression is not a polynomial in {', '.join(map(str, gens))}")
    return sp.Poly(p, *gens), gens


@fn("CoefficientRules", area=AREA, status="added",
    notes="CoefficientRules[p, {x1, ...}] -> {{e1, ...} -> c, ...} for the nonzero terms, in lexicographic order "
          "(third argument: \"Lexicographic\", \"DegreeLexicographic\", \"DegreeReverseLexicographic\"); "
          "CoefficientRules[p] uses Variables[p]; CoefficientRules[0, vars] -> {}"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _coeffrules(e: Expr, v: Var | ListOf[Var] = None, order: Expr = None) -> ListOf[Expr]:
    P, gens = _poly(e, v, "CoefficientRules")
    if P.is_zero:
        return L()
    terms = P.terms(order=_order(order, "CoefficientRules"))
    return L(*(rule(L(*(int(k) for k in m)), out(c)) for m, c in terms))


@fn("MonomialList", area=AREA, status="added",
    notes="MonomialList[p, {x1, ...}] -> the terms (coefficient times monomial) in lexicographic order (third "
          "argument: monomial order); MonomialList[p] uses Variables[p]"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _monolist(e: Expr, v: Var | ListOf[Var] = None, order: Expr = None) -> ListOf[Expr]:
    P, gens = _poly(e, v, "MonomialList")
    if P.is_zero:
        return L()
    terms = P.terms(order=_order(order, "MonomialList"))
    return L(*(out(c * sp.Mul(*(g ** k for g, k in zip(gens, m)))) for m, c in terms))


@replace("Exponent", AREA,
         "Exponent[p, x]: highest power of x (Exponent[0, x] = -Infinity), also for sums of non-integer powers; "
         "batch 4: Exponent[p, x, h] applies h to the list of distinct exponents (in increasing order, 0 for a "
         "constant term): Exponent[x^3 + x, x, Min] -> 1, Exponent[1 + x^2, x, List] -> {0, 2}; threads over "
         "lists")
def _exponent(e: Expr, x: Expr, h: Expr = None) -> Expr:
    if is_list(e):
        return L(*(_exponent(y, x, h) for y in e.args))
    if h is None:
        return _AL._exponent(e, x)
    p = sp.expand(sym(e, "Exponent"))
    v = variable(x, "Exponent")
    if p == 0:
        return _ev(Expr(h.name if isinstance(h, Symbol) else h, R(-sp.oo))) if not (isinstance(h, Symbol) and h.name == "List") else L(R(-sp.oo))
    pows = set()
    for t in sp.Add.make_args(p):
        c, k = t.as_coeff_exponent(v)
        if c.has(v):
            raise ValueError("Exponent: expression is not a sum of powers of the variable")
        pows.add(k)
    ks = sorted(pows, key=lambda k: (float(k) if k.is_number else 0, sp.default_sort_key(k)))
    return _ev(Expr(h.name if isinstance(h, Symbol) else h, *(out(k) for k in ks)))


# ───────────────────────────── polynomial arithmetic ─────────────────────────────

def _poly_list(args, fname, min_n=1):
    ps = [sym(a, fname) for a in args]
    if len(ps) < min_n:
        raise ValueError(f"{fname} needs at least {min_n} polynomials")
    for p in ps:
        if not sp.together(p).is_rational_function():
            raise ValueError(f"{fname}: arguments must be polynomials")
    return ps


def _modulus(opts, fname):
    m = opts.get("Modulus")
    if m is None:
        return None
    mv = sym(m, fname)
    if not (mv.is_Integer and mv >= 0):
        raise ValueError(f"{fname}: Modulus must be a nonnegative integer")
    if mv != 0 and not sp.isprime(int(mv)):
        raise NotImplementedError(f"{fname}: Modulus must be a prime")
    return int(mv) or None


def _mod_positive(p, m, gens=None):
    """Coefficients of p reduced into 0..m-1 (Wolfram's representation; SymPy uses symmetric residues)."""
    if m is None:
        return p
    if p.is_number:
        return sp.Integer(int(p) % m)
    gens = gens or sorted(p.free_symbols, key=str)
    P = sp.Poly(p, *gens)
    return sp.Add(*(sp.Integer(int(c) % m) * sp.Mul(*(g ** k for g, k in zip(gens, mon))) for mon, c in P.terms()))


@fn("PolynomialGCD", area=AREA, status="added",
    notes="greatest common divisor of polynomials with rational coefficients (rational functions: gcd of "
          "numerators over lcm of denominators, as Wolfram); Modulus -> p; PolynomialGCD[p] = p"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pgcd(*args: Expr) -> Expr:
    args, opts = split_opts(args, {"Modulus"}, "PolynomialGCD", extra_known=("Extension",))
    m = _modulus(opts, "PolynomialGCD")
    ps = _poly_list(args, "PolynomialGCD")
    g = ps[0]
    for q in ps[1:]:
        g = sp.gcd(g, q, modulus=m) if m else sp.gcd(g, q)
    if m:
        g = _mod_positive(sp.expand(g), m)
    return out(sp.factor(g) if not m else g)


@fn("PolynomialLCM", area=AREA, status="added",
    notes="least common multiple of polynomials (factored, as Wolfram); Modulus -> p"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _plcm(*args: Expr) -> Expr:
    args, opts = split_opts(args, {"Modulus"}, "PolynomialLCM", extra_known=("Extension",))
    m = _modulus(opts, "PolynomialLCM")
    ps = _poly_list(args, "PolynomialLCM")
    g = ps[0]
    for q in ps[1:]:
        g = sp.lcm(g, q, modulus=m) if m else sp.lcm(g, q)
    if m:
        return out(_mod_positive(sp.expand(g), m))
    return out(sp.factor(g))


@fn("PolynomialQuotientRemainder", area=AREA, status="added",
    notes="PolynomialQuotientRemainder[p, q, x] -> {quotient, remainder} with p = quotient q + remainder and "
          "deg remainder < deg q (rational coefficients); a zero divisor raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pqr(p: Expr, q: Expr, x: Expr) -> ListOf[Expr]:
    quo, rem = _AL._polydiv(p, q, x, "PolynomialQuotientRemainder")
    return L(out(quo), out(rem))


@fn("PolynomialReduce", area=AREA, status="added",
    notes="PolynomialReduce[p, {p1, ...}, {x1, ...}] -> {{a1, ...}, b} with p = Σ ai pi + b and b reduced with "
          "respect to the leading terms (lexicographic order, SymPy reduced; for a Gröbner basis the remainder "
          "is unique)"
          ". Check: exact, self-check (the result is verified before it is returned), reference (differential cases "
          "in tests/library).")
def _preduce(p: Expr, qs: Expr, vs: Var | ListOf[Var], *rest: Expr) -> ListOf[Expr]:
    rest, opts = split_opts(rest, {"MonomialOrder"}, "PolynomialReduce")
    gens = _vars_opt(vs, "PolynomialReduce")
    P = sp.expand(sym(p, "PolynomialReduce"))
    Q = [sp.expand(sym(q, "PolynomialReduce")) for q in (qs.args if is_list(qs) else [qs])]
    order = _order(opts.get("MonomialOrder"), "PolynomialReduce")
    coeffs, r = sp.reduced(P, Q, *gens, order=order)
    if sp.expand(sp.Add(*(c * q for c, q in zip(coeffs, Q))) + r - P) != 0:
        raise ArithmeticError("PolynomialReduce: verification failed")
    return L(L(*(out(c) for c in coeffs)), out(r))


@fn("PolynomialMod", area=AREA, status="added",
    notes="PolynomialMod[p, m]: integer m reduces every coefficient into 0..m-1 (rational coefficients need m "
          "coprime to the denominators); a polynomial m in one variable with a unit leading coefficient gives "
          "the remainder of p modulo m (no division by the leading coefficient otherwise: raises); lists of "
          "moduli apply in turn"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _pmod(p: Expr, m: Expr) -> Expr:
    P = sp.expand(sym(p, "PolynomialMod"))
    mods = list(m.args) if is_list(m) else [m]
    for mm in mods:
        mv = sp.expand(sym(mm, "PolynomialMod"))
        if mv.is_Integer:
            mi = int(mv)
            if mi == 0:
                continue
            gens = sorted(P.free_symbols, key=str)
            if not gens:
                P = sp.Integer(P % mi) if P.is_Integer else sp.Integer(int(sp.Rational(P).p * sp.mod_inverse(
                    sp.Rational(P).q, mi)) % mi)
                continue
            Pp = sp.Poly(P, *gens)
            terms = []
            for mon, c in Pp.terms():
                c = sp.Rational(c)
                if sp.gcd(c.q, mi) != 1:
                    raise ValueError("PolynomialMod: a denominator is not invertible modulo m")
                terms.append(sp.Integer((c.p * sp.mod_inverse(c.q, mi)) % mi) * sp.Mul(*(g ** k for g, k in
                                                                                         zip(gens, mon))))
            P = sp.Add(*terms)
        else:
            gens = sorted(mv.free_symbols, key=str)
            if len(gens) != 1:
                raise NotImplementedError("PolynomialMod: polynomial moduli in one variable only")
            x = gens[0]
            lc = sp.Poly(mv, x).LC()
            if lc not in (1, -1):
                raise NotImplementedError("PolynomialMod: the modulus polynomial must have leading coefficient ±1")
            P = sp.expand(sp.rem(P, mv, x))
    return out(P)


@fn("GroebnerBasis", area=AREA, status="added",
    notes="GroebnerBasis[{p1, ...}, {x1, ...}] (lexicographic, the default Wolfram MonomialOrder; also "
          "DegreeLexicographic, DegreeReverseLexicographic): reduced basis, each polynomial made primitive with "
          "integer coefficients and a positive leading coefficient, listed from the smallest leading monomial "
          "(Wolfram's order); GroebnerBasis[polys, vars, elims] eliminates elims; equations are moved to one "
          "side; Modulus -> p"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _groebner(polys: Expr, vs: Var | ListOf[Var], *rest: Expr) -> ListOf[Expr]:
    rest, opts = split_opts(rest, {"MonomialOrder", "Modulus"}, "GroebnerBasis",
                            extra_known=("CoefficientDomain", "Sort", "Method"))
    elims = _vars_opt(rest[0], "GroebnerBasis") if rest else []
    if len(rest) > 1:
        raise ValueError("GroebnerBasis: too many arguments")
    gens = _vars_opt(vs, "GroebnerBasis")
    ps = []
    for p in (polys.args if is_list(polys) else [polys]):
        if isinstance(p, Expr) and p.head == "Equal" and len(p.args) == 2:
            ps.append(sp.expand(sym(p.args[0], "GroebnerBasis") - sym(p.args[1], "GroebnerBasis")))
        else:
            ps.append(sp.expand(sym(p, "GroebnerBasis")))
    order = _order(opts.get("MonomialOrder"), "GroebnerBasis")
    m = _modulus(opts, "GroebnerBasis")
    allg = list(elims) + [g for g in gens if g not in elims]
    if elims:
        order = "lex"
    extra = sorted(set().union(*(p.free_symbols for p in ps)) - set(allg), key=str) if ps else []
    kw = {"modulus": m} if m else {}
    if extra:
        kw["domain"] = sp.FractionField(sp.QQ, *extra) if not m else None
        if m:
            raise NotImplementedError("GroebnerBasis: parameters with Modulus are not supported")
    G = sp.groebner(ps, *allg, order=order, **kw)
    res = []
    for g in G.exprs:
        if elims and g.has(*elims):
            continue
        if m:
            res.append(_mod_positive(sp.expand(g), m, allg))
            continue
        num, den = sp.fraction(sp.together(g))
        q = sp.Poly(num, *allg)
        if not extra:
            q = q.primitive()[1]
        e = q.as_expr()
        lc = sp.Poly(e, *allg).terms(order=order)[0][1]
        if (lc.is_number and lc < 0) or (not lc.is_number and sp.Poly(lc).LC() < 0 if lc.free_symbols else False):
            e = -e
        res.append(sp.expand(e))
    return L(*(out(g) for g in reversed(res)))


# ───────────────────────────── factoring ─────────────────────────────

def _factor_kw(opts, fname):
    kw = {}
    if "Extension" in opts:
        ext = opts["Extension"]
        if isinstance(ext, Symbol) and ext.name in ("None",):
            pass
        elif isinstance(ext, Symbol) and ext.name == "Automatic":
            raise NotImplementedError(f"{fname}: Extension -> Automatic is not supported")
        elif isinstance(ext, Symbol) and ext.name == "All":
            raise NotImplementedError(f"{fname}: Extension -> All is not supported")
        else:
            kw["extension"] = [sym(a, fname) for a in ext.args] if is_list(ext) else sym(ext, fname)
    if "GaussianIntegers" in opts and is_true(opts["GaussianIntegers"]):
        kw["gaussian"] = True
    m = _modulus(opts, fname)
    if m:
        kw["modulus"] = m
    return kw, m


def _fix_mod_factors(fl, m, gens):
    c, facs = fl
    return sp.Integer(int(c) % m), [(_mod_positive(sp.expand(f), m, gens), k) for f, k in facs]


@replace("Factor", AREA,
         "factors over the rationals (SymPy factor); batch 4: Extension -> a or {a, ...} (algebraic extension), "
         "GaussianIntegers -> True, Modulus -> p (prime; coefficients shown in 0..p-1 as Wolfram); threads over "
         "lists and equations; note the kernel distributes a numeric content over a single sum (2 (x^2 - 2) "
         "displays expanded)")
def _factor(e, *rest):
    rest, opts = split_opts(rest, {"Extension", "GaussianIntegers", "Modulus"}, "Factor")
    if rest:
        raise ValueError("Factor takes one expression and options")
    th = _thread(e, lambda y: _factor(y, *[rule(Symbol(k), v) for k, v in opts.items()]))
    if th is not None:
        return th
    p = sym(e, "Factor")
    kw, m = _factor_kw(opts, "Factor")
    if m:
        gens = sorted(p.free_symbols, key=str)
        if not gens:
            return out(sp.Integer(int(p) % m))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            c, facs = _fix_mod_factors(sp.factor_list(p, *gens, modulus=m), m, gens)
        return out(sp.Mul(c, *(f ** k for f, k in facs)))
    return out(sp.factor(p, **kw))


@fn("FactorList", area=AREA, status="added",
    notes="FactorList[p] -> {{c, 1}, {f1, e1}, ...}: the numeric content first (always present, 1 when trivial), "
          "then the irreducible factors over the rationals with exponents; Extension, GaussianIntegers, Modulus"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _factorlist(e: Expr, *rest: Expr) -> ListOf[Expr]:
    rest, opts = split_opts(rest, {"Extension", "GaussianIntegers", "Modulus"}, "FactorList")
    p = sym(e, "FactorList")
    kw, m = _factor_kw(opts, "FactorList")
    gens = sorted(p.free_symbols, key=str)
    if not gens:
        return L(L(out(p), 1))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fl = sp.factor_list(p, *gens, **kw)
    if m:
        fl = _fix_mod_factors(fl, m, gens)
    c, facs = fl
    return L(L(out(c), 1), *(L(out(f), int(k)) for f, k in facs))


def _content_split(p, vs):
    """(numeric content, factor free of vs, remaining polynomial)."""
    num, den = sp.fraction(sp.together(p))
    gens = sorted(num.free_symbols, key=str)
    if not gens:
        return p, sp.Integer(1), sp.Integer(1)
    P = sp.Poly(num, *gens)
    c, prim = P.primitive()
    if prim.LC() < 0 if prim.LC().is_number else False:
        c, prim = -c, -prim
    c = c / den if den.is_number else c
    rest = prim.as_expr() if den.is_number else prim.as_expr() / den
    if not vs:
        return c, sp.Integer(1), rest
    others = [g for g in gens if g not in vs]
    if not others or not any(g in vs for g in gens):
        return (c, rest, sp.Integer(1)) if not any(g in vs for g in gens) else (c, sp.Integer(1), rest)
    Q = sp.Poly(prim.as_expr(), *vs)
    g = sp.Integer(0)
    for cf in Q.coeffs():
        g = sp.gcd(g, cf)
    g = sp.factor(g)
    rem = sp.factor_terms(sp.cancel(prim.as_expr() / g))
    rem = sp.expand(rem)
    if den.is_number:
        return c, g, rem
    return c, g, rem / den


@fn("FactorTerms", area=AREA, status="added",
    notes="FactorTerms[p] pulls out the overall numeric factor (FactorTerms[2 x^2 - 4] = 2 (-2 + x^2), which the "
          "kernel displays distributed); FactorTerms[p, x] also pulls out factors free of x; the leading "
          "coefficient of the remaining polynomial is made positive"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _factorterms(e: Expr, v: Var | ListOf[Var] = None) -> Expr:
    c, g, r = _content_split(sym(e, "FactorTerms"), _vars_opt(v, "FactorTerms"))
    return out(sp.Mul(c, g, r, evaluate=False) if (g != 1 or c != 1) else r)


@fn("FactorTermsList", area=AREA, status="added",
    notes="FactorTermsList[p] -> {numeric factor, polynomial}; FactorTermsList[p, {x, ...}] -> {numeric factor, "
          "factor free of the variables, remaining polynomial}"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _factortermslist(e: Expr, v: Var | ListOf[Var] = None) -> ListOf[Expr]:
    vs = _vars_opt(v, "FactorTermsList")
    c, g, r = _content_split(sym(e, "FactorTermsList"), vs)
    if vs:
        return L(out(c), out(g), out(r))
    return L(out(c), out(r))


@fn("FactorSquareFree", area=AREA, status="added",
    notes="square-free factorisation (SymPy sqf): FactorSquareFree[x^3 - x^2 - x + 1] -> (-1 + x)^2 (1 + x)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sqf(e: Expr, *rest: Expr) -> Expr:
    rest, opts = split_opts(rest, {"Modulus"}, "FactorSquareFree")
    m = _modulus(opts, "FactorSquareFree")
    p = sym(e, "FactorSquareFree")
    return out(sp.sqf(p, modulus=m) if m else sp.sqf(p))


@fn("FactorSquareFreeList", area=AREA, status="added",
    notes="FactorSquareFreeList[p] -> {{c, 1}, {f1, 1}, {f2, 2}, ...} (numeric content first, then square-free "
          "factors with their multiplicities)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sqflist(e: Expr) -> ListOf[Expr]:
    p = sym(e, "FactorSquareFreeList")
    gens = sorted(p.free_symbols, key=str)
    if not gens:
        return L(L(out(p), 1))
    c, facs = sp.sqf_list(p, *gens)
    return L(L(out(c), 1), *(L(out(f), int(k)) for f, k in facs))


@fn("Decompose", area=AREA, status="added",
    notes="Decompose[p, x] -> {p1, ..., pn} with p = p1[p2[...pn[x]]] (functional decomposition, SymPy); "
          "indecomposable p gives {p}; checked by composing back"
          ". Check: exact, self-check (the result is verified before it is returned), reference (differential cases "
          "in tests/library).")
def _decompose(e: Expr, x: Var) -> ListOf[Expr]:
    p = sp.expand(sym(e, "Decompose"))
    v = variable(x, "Decompose")
    if not p.is_polynomial(v):
        raise ValueError("Decompose: expression is not a polynomial in the variable")
    parts = sp.decompose(p, v) if sp.degree(p, v) > 1 else [p]
    comp = v
    for q in reversed(parts):
        comp = q.subs(v, comp)
    if sp.expand(comp - p) != 0:
        raise ArithmeticError("Decompose: verification failed")
    return L(*(out(q) for q in parts))


@fn("HornerForm", area=AREA, status="added",
    notes="HornerForm[p] / HornerForm[p, x] / HornerForm[p, {x, y}]: nested Horner form (SymPy horner); rational "
          "functions: HornerForm of numerator and denominator"
          ". Check: reference (differential cases in tests/library).")
def _horner(e: Expr, v: Var | ListOf[Var] = None) -> Expr:
    p = sym(e, "HornerForm")
    vs = _vars_opt(v, "HornerForm") or []
    n, d = sp.fraction(sp.together(p))

    def h(q):
        q = sp.expand(q)
        if q.is_number:
            return q
        return sp.horner(q, *vs) if vs else sp.horner(q)
    return out(h(n) / h(d) if d != 1 else h(n))


@fn("MinimalPolynomial", area=AREA, status="added",
    notes="MinimalPolynomial[s, x]: the minimal polynomial of the algebraic number s over the rationals, primitive "
          "with integer coefficients and positive leading coefficient (Wolfram); MinimalPolynomial[s] gives a "
          "pure function; transcendental input raises"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _minpoly(s: Expr, x: Expr = None, *rest: Expr) -> Expr:
    v = sym(s, "MinimalPolynomial")
    if x is None:
        t = sp.Symbol("x")
    else:
        t = variable(x, "MinimalPolynomial")
    try:
        p = sp.minimal_polynomial(v, t)
    except (sp.polys.polyerrors.NotAlgebraic, NotImplementedError) as exc:
        raise ValueError(f"MinimalPolynomial: {v} is not an algebraic number") from exc
    P = sp.Poly(p, t)
    P = P.primitive()[1]
    if P.LC() < 0:
        P = -P
    p = P.as_expr()
    if x is None:
        return Expr("Function", _slotify(out(p), t.name))
    return out(p)


def _slotify(e, name):
    if isinstance(e, Symbol) and e.name == name:
        return Expr("Slot", 1)
    if isinstance(e, Expr):
        return Expr(e.head if not isinstance(e.head, Expr) else _slotify(e.head, name),
                    *(_slotify(a, name) for a in e.args))
    return e


@fn("IrreduciblePolynomialQ", area=AREA, status="added",
    notes="True if the polynomial cannot be factored over the rationals (numeric content ignored, as units of Q); "
          "constants give False; Extension, GaussianIntegers, Modulus options"
          ". Check: reference (differential cases in tests/library).")
def _irredq(e: Expr, *rest: Expr) -> Boolean:
    rest, opts = split_opts(rest, {"Extension", "GaussianIntegers", "Modulus"}, "IrreduciblePolynomialQ")
    try:
        p = sp.expand(sym(e, "IrreduciblePolynomialQ"))
    except Exception:
        return False
    gens = sorted(p.free_symbols, key=str)
    if not gens or not p.is_polynomial(*gens):
        return False
    kw, m = _factor_kw(opts, "IrreduciblePolynomialQ")
    c, facs = sp.factor_list(p, *gens, **kw)
    return len(facs) == 1 and facs[0][1] == 1


@fn("SymmetricPolynomial", area=AREA, status="added",
    notes="SymmetricPolynomial[k, {x1, ..., xn}]: the k-th elementary symmetric polynomial (0 for k > n, 1 for "
          "k = 0)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sympoly(k: NonNegInteger, vs: Var | ListOf[Var]) -> Expr:
    kk = nonneg_int(k, "SymmetricPolynomial", "degree")
    xs = _vars_opt(vs, "SymmetricPolynomial")
    if kk > len(xs):
        return 0
    from itertools import combinations
    return out(sp.Add(*(sp.Mul(*c) for c in combinations(xs, kk))))


@fn("SymmetricReduction", area=AREA, status="added",
    notes="SymmetricReduction[f, {x1, ..., xn}] -> {p, q}: p the symmetric part written in the elementary "
          "symmetric polynomials of the xi, q the remainder, f = p + q; SymmetricReduction[f, xs, {s1, ..., sn}] "
          "writes p in the si; checked by expansion"
          ". Check: exact, self-check (the result is verified before it is returned), reference (differential cases "
          "in tests/library).")
def _symred(e: Expr, vs: Var | ListOf[Var], names: Expr = None) -> ListOf[Expr]:
    from sympy.polys.polyfuncs import symmetrize
    f = sp.expand(sym(e, "SymmetricReduction"))
    xs = _vars_opt(vs, "SymmetricReduction")
    symp, rem, defs = symmetrize(f, *xs, formal=True)
    if names is not None:
        ss = [sym(a, "SymmetricReduction") for a in list_arg(names, "SymmetricReduction", "names")]
        if len(ss) != len(xs):
            raise ValueError("SymmetricReduction: give one name per variable")
        sub = {d[0]: ss[i] for i, d in enumerate(defs)}     # defs are listed s1, s2, ... in order
        p = symp.xreplace(sub)
    else:
        p = symp.xreplace({d[0]: d[1] for d in defs})
    full = symp.xreplace({d[0]: d[1] for d in defs})
    if sp.expand(full + rem - f) != 0:
        raise ArithmeticError("SymmetricReduction: verification failed")
    return L(out(p), out(rem))


def _root_radicals(r):
    """Radical form of a CRootOf if SymPy can find one (matched numerically to the root)."""
    poly = r.poly
    x = poly.gen
    try:
        rts = sp.roots(poly, x, cubics=True, quartics=True)
    except Exception:
        return r
    if sum(rts.values()) != poly.degree():
        return r
    target = complex(sp.N(r, 30))
    best = None
    for c in rts:
        if c.has(sp.CRootOf):
            continue
        if abs(complex(sp.N(c, 30)) - target) < 1e-20:
            best = c
            break
    return best if best is not None else r


@fn("ToRadicals", area=AREA, status="added",
    notes="ToRadicals[e] rewrites Root objects of degree <= 4 (or solvable by radicals in SymPy) as radicals, "
          "matched to the root numerically; other roots stay Root[...]"
          ". Check: reference (differential cases in tests/library).")
def _toradicals(e: Expr) -> Expr:
    v = sym(e, "ToRadicals")
    return out(v.replace(lambda u: isinstance(u, sp.CRootOf), _root_radicals))


# ───────────────────────────── simplification ─────────────────────────────

def _numeric_equal(a, b, n=6, tol=1e-20):
    """a == b numerically at random complex-free sample points (all free symbols real in 0.3..2.3)."""
    import random
    rnd = random.Random(7)
    free = sorted((a.free_symbols | b.free_symbols), key=str)
    ok = 0
    for _ in range(n):
        vals = {s_: sp.Rational(rnd.randint(3, 23), 10) for s_ in free}
        try:
            d = complex(sp.N((a - b).subs(vals), 40))
            ref = abs(complex(sp.N(a.subs(vals), 40)))
        except Exception:
            continue
        if d != d:          # nan
            continue
        if abs(d) > tol * max(1.0, ref):
            return False
        ok += 1
    return ok >= 3


def _size(e):
    return (sp.count_ops(e, visual=False), len(str(e)))


def _full_simplify_candidates(v):
    import importlib
    fu = importlib.import_module("sympy.simplify.fu")
    tries = [lambda e: sp.simplify(e), sp.factor, sp.cancel, sp.together, lambda e: sp.expand(e),
             sp.trigsimp, lambda e: fu.fu(e), sp.powsimp, sp.radsimp, sp.sqrtdenest, sp.gammasimp,
             lambda e: sp.simplify(sp.expand_func(e)), lambda e: sp.simplify(e.rewrite(sp.exp)),
             lambda e: sp.simplify(sp.logcombine(e)), lambda e: sp.factor(sp.simplify(e)),
             lambda e: sp.simplify(sp.expand_log(e)), lambda e: sp.nsimplify(e) if e.is_number and
             not e.has(sp.Float) and False else e]
    cands = [v]
    for t in tries:
        try:
            c = t(v)
        except Exception:
            continue
        if isinstance(c, sp.Basic):
            cands.append(c)
    return cands


@fn("FullSimplify", area=AREA, status="added",
    notes="FullSimplify[e] / FullSimplify[e, assum]: the smallest (operation count, then length) of a wider set "
          "of equivalence-preserving rewrites than Simplify (simplify, factor, cancel, together, expand, trigsimp, "
          "fu, powsimp, radsimp, sqrtdenest, gammasimp, expand_func, exp rewriting, logcombine), applied after "
          "and before refining with the assumptions; every candidate is checked numerically against the input; "
          "threads over lists and equations")
def _fullsimplify(e, *assum):
    from .assume import split_option, effective, to_sym, from_sym
    th = _thread(e, lambda y: _fullsimplify(y, *assum)) if is_list(e) else None
    if th is not None:
        return th
    rest, opt = split_option(assum, "FullSimplify")
    if len(rest) > 1:
        raise ValueError("FullSimplify[expr, assum] takes one assumption argument")
    A = effective(rest, opt, "FullSimplify")
    v = A.refine(to_sym(e, "FullSimplify"))
    if not isinstance(v, sp.Basic) or isinstance(v, sp.logic.boolalg.BooleanAtom):
        return from_sym(A.unapply(v))
    if isinstance(v, sp.Rel) or isinstance(v, sp.logic.boolalg.Boolean):
        return from_sym(A.unapply(A.refine(sp.simplify(v))))
    best = v
    for c in _full_simplify_candidates(v):
        c = A.refine(c)
        if not isinstance(c, sp.Basic) or c.has(sp.nan, sp.zoo) and not v.has(sp.nan, sp.zoo):
            continue
        if _size(c) < _size(best):
            if c.free_symbols - v.free_symbols:
                continue
            if A.assumptions_free() if hasattr(A, "assumptions_free") else True:
                if not (sp.simplify(c - v) == 0 or _numeric_equal(c, v)):
                    continue
            best = c
    return from_sym(A.unapply(best))


@fn("TrigFactor", area=AREA, status="added",
    notes="TrigFactor[e]: sums of sines and cosines turned into products (sum-to-product, Fu's TR9/TR10i), then "
          "factored (TrigFactor[Sin[x] + Sin[y]] -> 2 Cos[x/2 - y/2] Sin[x/2 + y/2]); the result is checked "
          "numerically; threads over lists"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _trigfactor(e: Expr) -> Expr:
    import importlib
    fu = importlib.import_module("sympy.simplify.fu")
    if is_list(e):
        return L(*(_trigfactor(y) for y in e.args))
    v = sym(e, "TrigFactor")
    cands = []
    for t in (lambda u: sp.factor(fu.TR9(u)), lambda u: sp.factor(fu.TR10i(u)),
              lambda u: sp.factor(fu.TR9(fu.TR10i(u))), lambda u: sp.factor(u),
              lambda u: sp.factor(sp.trigsimp(u))):
        try:
            c = t(v)
        except Exception:
            continue
        if _numeric_equal(c, v):
            cands.append(c)
    if not cands:
        return out(v)
    # prefer products: fewest additive terms at the top level, then size
    best = min(cands, key=lambda c: (len(sp.Add.make_args(c)), _size(c)))
    return out(best)


@fn("FunctionExpand", area=AREA, status="added",
    notes="FunctionExpand[e]: special functions rewritten in simpler functions (SymPy expand_func: Gamma[x + 3] -> "
          "x (1 + x) (2 + x) Gamma[x], Binomial[n, 2] -> (n - 1) n/2, Beta -> Gamma ratio, PolyGamma and Zeta at "
          "special arguments), checked numerically; FunctionExpand[e, assum] refines with assumptions"
          ". Check: exact, self-check (the result is verified before it is returned), reference (differential cases "
          "in tests/library).")
def _functionexpand(e: Expr, *assum: Expr) -> Expr:
    from .assume import split_option, effective, to_sym, from_sym
    if is_list(e):
        return L(*(_functionexpand(y, *assum) for y in e.args))
    rest, opt = split_option(assum, "FunctionExpand")
    A = effective(rest, opt, "FunctionExpand")
    v = A.refine(to_sym(e, "FunctionExpand"))
    r = sp.expand_func(v)
    r = r.replace(lambda u: isinstance(u, sp.beta), lambda u: sp.gamma(u.args[0]) * sp.gamma(u.args[1])
                  / sp.gamma(u.args[0] + u.args[1]))
    r = sp.expand_func(r)
    if r != v and not _numeric_equal(r, v):
        raise ArithmeticError("FunctionExpand: verification failed")
    return from_sym(A.unapply(A.refine(r)))


@fn("PossibleZeroQ", area=AREA, status="added",
    notes="PossibleZeroQ[e]: True if e simplifies to 0 or is numerically 0 (40 digits) at several sample points of "
          "its variables, False otherwise (a heuristic, as in Wolfram)"
          ". Check: reference (differential cases in tests/library).")
def _possiblezeroq(e: Expr) -> Boolean:
    try:
        v = sym(e, "PossibleZeroQ")
    except Exception:
        return False
    if not isinstance(v, sp.Expr):
        return False
    if v == 0:
        return True
    try:
        if sp.simplify(v) == 0:
            return True
    except Exception:
        pass
    return bool(_numeric_equal(v, sp.Integer(0), n=8, tol=1e-25))


# ───────────────────────────── domains and Element ─────────────────────────────

_DOMAINS = {"Reals": "is_real", "Integers": "is_integer", "Rationals": "is_rational", "Complexes": "is_complex",
            "Algebraics": "is_algebraic", "Primes": "is_prime"}


def _element1(v, dom):
    if dom == "Booleans":
        if isinstance(v, sp.logic.boolalg.BooleanAtom):
            return True
        if v.is_number:
            return False
        return None
    if dom == "Primes":
        if v.is_Integer:
            return bool(sp.isprime(int(v)))
        if v.is_number and v.is_integer is False:
            return False
        return None
    if dom == "Complexes":
        if v.is_number:
            fin = v.is_finite
            return True if fin else (False if fin is False else None)
        return None
    if not v.is_number:
        return None
    val = getattr(v, _DOMAINS[dom])
    if val is None and dom in ("Rationals", "Algebraics", "Integers", "Reals"):
        try:
            if dom == "Reals" and abs(complex(sp.N(v, 30)).imag) > 1e-20:
                return False
        except Exception:
            pass
    return val


@fn("Element", area=AREA, status="added",
    notes="Element[x, dom] for dom in Reals, Integers, Rationals, Algebraics, Complexes, Primes, Booleans: True or "
          "False when decidable for explicit numbers (Element[Sqrt[2], Rationals] -> False, Element[Pi, "
          "Algebraics] -> False), unevaluated for symbols (used in assumptions); Element[{x1, ...}, dom] tests "
          "every element (False if any is False)")
def _element(x, dom):
    if not isinstance(dom, Symbol) or dom.name not in set(_DOMAINS) | {"Booleans"}:
        raise Unevaluated()
    items = list(x.args) if is_list(x) else [x]
    res = []
    for it in items:
        if it is True or it is False:
            res.append(dom.name == "Booleans")
            continue
        try:
            v = sym(it, "Element")
        except Exception:
            raise Unevaluated()
        res.append(_element1(v, dom.name))
    if any(r is False for r in res):
        return False
    if all(r is True for r in res):
        return True
    raise Unevaluated()


# ───────────────────────────── solving ─────────────────────────────

C_INT = None


def _c(i):
    return sp.Symbol(f"C{i}")


def _wl_order_key(v):
    try:
        z = complex(sp.N(v, 30))
    except Exception:
        return (2, 0, 0, 0, sp.default_sort_key(v))
    real = abs(z.imag) < 1e-25
    return (0 if real else 1, z.real, abs(z.imag), z.imag, sp.default_sort_key(v))


def _poly_roots(num, x, cubics=True, quartics=True):
    """All roots of a polynomial in x, with multiplicity, in Wolfram order (radicals for degree <= 2 and binomial
    factors, and for degree 3/4 when cubics/quartics; Root objects otherwise). Parameters allowed."""
    P = sp.Poly(num, x)
    if P.degree() <= 0:
        return None if P.is_zero else []
    numeric = not (P.free_symbols - {x})
    res = []
    for fac, mult in sp.factor_list(num, x)[1]:
        F = sp.Poly(fac, x)
        d = F.degree()
        if d == 0:
            continue
        binomial = len(F.terms()) == 2 and F.terms()[1][0] == (0,)
        if d <= 2 or binomial or (d == 3 and cubics) or (d == 4 and quartics):
            rts = sp.roots(F, x, cubics=cubics, quartics=quartics)
            if sum(rts.values()) != d:
                if not numeric:
                    raise NotImplementedError("Solve: cannot express the roots of a parametric polynomial")
                rts = {sp.CRootOf(F.as_expr(), x, i): 1 for i in range(d)}
        else:
            if not numeric:
                raise NotImplementedError("Solve: cannot express the roots of a parametric polynomial of degree > 4")
            rts = {sp.CRootOf(fac, x, i): 1 for i in range(d)}
        for r, k in rts.items():
            res.extend([r] * (k * mult))
    if numeric:
        res.sort(key=_wl_order_key)
    return res


def _imageset_cond(img, k):
    """ImageSet(Lambda(n, f(n)), Integers) -> (f(C[k]) as SymPy, True)."""
    lam = img.lamda
    n = lam.variables[0]
    return lam.expr.xreplace({n: _c(k)})


def _verify_root(f, x, r, eqs_extra=()):
    try:
        val = f.subs(x, r)
    except Exception:
        return False
    if val.has(sp.CRootOf) or r.has(sp.CRootOf):
        try:
            return abs(complex(sp.N(val, 40))) < 1e-25
        except Exception:
            return False
    try:
        z = sp.simplify(val)
    except Exception:
        z = val
    if z == 0:
        return True
    if z.free_symbols:
        return _numeric_equal(z, sp.Integer(0))
    try:
        return abs(complex(sp.N(z, 40))) < 1e-25
    except Exception:
        return False


def _univariate(f, x, dom, cubics=True, quartics=True):
    """Solutions of f(x) == 0 as a list of SymPy values (possibly containing C1 for periodic families: then the
    entries are (value, 'C') tuples). Raises if the solution set cannot be determined."""
    f = sp.together(f)
    num, den = sp.fraction(f)
    num = sp.expand(num)
    if num.is_polynomial(x) and den.is_polynomial(x) and not den.has(x) is None:
        pass
    if num.is_polynomial(x) and (den.is_polynomial(x) or not den.has(x)):
        if num.is_zero:
            raise NotImplementedError("Solve: the equation holds identically")
        rts = _poly_roots(num, x, cubics, quartics)
        if rts is None:
            raise NotImplementedError("Solve: the equation holds identically")
        rts = [r for r in rts if not (den.has(x) and sp.simplify(den.subs(x, r)) == 0)]
        return _filter_dom(rts, dom)
    domain = sp.S.Reals if dom == "Reals" else sp.S.Complexes
    try:
        S = sp.solveset(f, x, domain)
    except Exception:
        S = None
    if S is None or isinstance(S, (sp.ConditionSet,)) or S.has(sp.ConditionSet):
        raise NotImplementedError("Solve: cannot solve this equation")
    vals = []
    parts = S.args if isinstance(S, sp.Union) else (S,)
    for part in parts:
        if isinstance(part, sp.FiniteSet):
            for r in part.args:
                if not _verify_root(f, x, r):
                    raise ArithmeticError("Solve: a solution failed verification")
                vals.append(r)
        elif isinstance(part, sp.ImageSet) and part.base_sets == (sp.S.Integers,) or (
                isinstance(part, sp.ImageSet) and len(part.base_sets) == 1 and part.base_sets[0] == sp.S.Integers):
            v = _imageset_cond(part, 1)
            for k in (0, 1, -1, 2):
                if not _verify_root(f, x, v.subs(_c(1), k)):
                    raise ArithmeticError("Solve: a solution family failed verification")
            vals.append((v, "C"))
        elif part is sp.S.EmptySet:
            continue
        else:
            raise NotImplementedError("Solve: the solution set is not a finite set or a simple periodic family")
    if dom == "Integers":
        vals = [v for v in vals if not isinstance(v, tuple)]
        return _filter_dom(vals, dom)
    return vals


def _filter_dom(rts, dom):
    if dom is None or dom == "Complexes":
        return rts
    keep = []
    for r in rts:
        if isinstance(r, tuple):
            raise NotImplementedError(f"Solve: periodic solution families over {dom} are not supported")
        if dom == "Reals":
            ok = r.is_real
            if ok is None:
                try:
                    if r.free_symbols:
                        raise NotImplementedError("Solve: cannot decide which parametric solutions are real")
                    z = complex(sp.N(r, 40))
                    ok = abs(z.imag) < 1e-25
                except NotImplementedError:
                    raise
                except Exception:
                    raise NotImplementedError("Solve: cannot decide whether a solution is real")
        elif dom == "Integers":
            ok = r.is_integer
            if ok is None:
                if r.free_symbols:
                    raise NotImplementedError("Solve: cannot decide which parametric solutions are integers")
                try:
                    z = complex(sp.N(r, 40))
                    ok = abs(z.imag) < 1e-25 and abs(z.real - round(z.real)) < 1e-25
                    if ok:
                        r = sp.Integer(round(z.real))
                except Exception:
                    raise NotImplementedError("Solve: cannot decide whether a solution is an integer")
        elif dom == "Rationals":
            ok = r.is_rational
            if ok is None:
                raise NotImplementedError("Solve: cannot decide whether a solution is rational")
        else:
            raise NotImplementedError(f"Solve: domain {dom} is not supported")
        if ok:
            keep.append(sp.nsimplify(r) if isinstance(r, sp.Float) else r)
    return keep


def _cond_rule(var, v):
    """Rule var -> v, with C1, C2 families wrapped as ConditionalExpression[v, C[1] ∈ Integers]."""
    if isinstance(v, tuple):
        v = v[0]
    cs = sorted((c for c in v.free_symbols if c.name[:1] == "C" and c.name[1:].isdigit()), key=str)
    val = K._constants_to_c(out(v))
    if cs:
        conds = [Expr("Element", Expr("C", int(c.name[1:])), Symbol("Integers")) for c in cs]
        cond = conds[0] if len(conds) == 1 else Expr("And", *conds)
        val = Expr("ConditionalExpression", val, cond)
    return rule(var, val)


_REL_SP = {"Less": sp.Lt, "Greater": sp.Gt, "LessEqual": sp.Le, "GreaterEqual": sp.Ge, "Unequal": sp.Ne}


def _rel_from(e, fname):
    """Less[a, b, c] etc. -> SymPy relational (chains become And)."""
    args = [sym(a, fname) for a in e.args]
    if len(args) < 2:
        raise ValueError(f"{fname}: malformed inequality")
    op = _REL_SP[e.head]
    if e.head == "Unequal" and len(args) > 2:
        raise NotImplementedError(f"{fname}: chained Unequal is not supported")
    rels = [op(a, b) for a, b in zip(args, args[1:])]
    return rels[0] if len(rels) == 1 else sp.And(*rels)


def _rel_parts(e, fname):
    """MikoshiLang equation/inequality -> ('eq', SymPy f) meaning f == 0, or ('rel', SymPy relational)."""
    if e is True or e is False:
        return ("bool", e)
    if isinstance(e, Expr) and e.head == "Equal":
        if len(e.args) != 2:
            raise NotImplementedError(f"{fname}: chained equations are not supported")
        return ("eq", sym(e.args[0], fname) - sym(e.args[1], fname))
    if isinstance(e, Expr) and e.head in _REL_SP:
        return ("rel", _rel_from(e, fname))
    if isinstance(e, Expr) and e.head in ("Or", "Not", "Implies", "Xor"):
        return ("logic", sym(e, fname))
    if isinstance(e, Expr) and e.head == "Element":
        return ("elem", e)
    return ("eq", sym(e, fname))


def _flatten_and(e):
    if is_list(e) or (isinstance(e, Expr) and e.head == "And"):
        return [x for y in e.args for x in _flatten_and(y)]
    return [e]


def _dom_name(d, fname):
    if d is None:
        return None
    if isinstance(d, Symbol) and d.name in ("Reals", "Integers", "Complexes", "Rationals"):
        return d.name
    raise NotImplementedError(f"{fname}: domain {d!r} is not supported")


def _solve_core(eqs_expr, vars_expr, dom, fname, opts):
    cubics = not is_false(opts.get("Cubics", True))
    quartics = not is_false(opts.get("Quartics", True))
    parts = [_rel_parts(e, fname) for e in _flatten_and(eqs_expr)]
    if any(k == "bool" and v is False for k, v in parts):
        return None, []
    parts = [(k, v) for k, v in parts if not (k == "bool" and v is True)]
    if any(k in ("rel", "logic", "elem") for k, _ in parts):
        raise NotImplementedError(f"{fname}: inequalities and logical combinations are not supported; use Reduce")
    eqs = [v for _, v in parts]
    if vars_expr is None:
        syms = set()
        for e in eqs:
            syms |= e.free_symbols
        xs = sorted(syms, key=str)
    else:
        xs = [variable(v, fname) for v in (vars_expr.args if is_list(vars_expr) else [vars_expr])]
    if not xs:
        raise ValueError(f"{fname}: no variables to solve for")
    if not eqs:
        return xs, [{}]
    if len(eqs) == 1 and len(xs) == 1:
        sols = _univariate(eqs[0], xs[0], dom, cubics, quartics)
        return xs, [{xs[0]: s_} for s_ in sols]
    if dom == "Integers":
        return xs, _diophantine(eqs, xs, fname)
    # systems: rational equations only (complete polynomial solving via SymPy/Gröbner)
    nums = []
    for e in eqs:
        n_, d_ = sp.fraction(sp.together(e))
        if not (n_.is_polynomial(*xs) and d_.is_polynomial(*xs)):
            raise NotImplementedError(f"{fname}: only polynomial and rational systems are supported")
        nums.append((sp.expand(n_), d_))
    try:
        sols = sp.solve([n_ for n_, _ in nums], list(reversed(xs)), dict=True)
    except NotImplementedError as exc:
        raise NotImplementedError(f"{fname}: SymPy cannot solve this system ({exc})") from exc
    good = []
    for s_ in sols:
        for n_, d_ in nums:
            val = n_.subs(s_)
            if not (sp.simplify(val) == 0 or (not val.free_symbols and _verify_root(val, xs[0], xs[0]))):
                if val.free_symbols and _numeric_equal(val, sp.Integer(0)):
                    continue
                raise ArithmeticError(f"{fname}: a solution failed verification")
            if d_.has(*xs) and sp.simplify(d_.subs(s_)) == 0:
                break
        else:
            good.append(s_)
    if dom in ("Reals", "Rationals"):
        filt = []
        for s_ in good:
            vals = list(s_.values())
            kept = _filter_dom(vals, dom)
            if len(kept) == len(vals):
                filt.append(s_)
        good = filt
    good.sort(key=lambda s_: tuple(_wl_order_key(s_[x]) if x in s_ else (3,) for x in xs))
    return xs, good


def _diophantine(eqs, xs, fname):
    from sympy.solvers.diophantine.diophantine import diophantine
    if len(eqs) != 1:
        raise NotImplementedError(f"{fname}: systems over the integers are not supported")
    f = sp.expand(eqs[0])
    if not f.is_polynomial(*xs):
        raise NotImplementedError(f"{fname}: only polynomial equations over the integers are supported")
    sols = diophantine(f, syms=xs, permute=True)
    res = []
    params = set()
    for tup in sorted(sols, key=lambda t: tuple(sp.default_sort_key(a) for a in t)):
        ps = set().union(*(a.free_symbols for a in tup)) - set(xs)
        params |= ps
        res.append(dict(zip(xs, tup)))
    if params:
        if sp.Poly(f, *xs).total_degree() != 1:
            raise NotImplementedError(f"{fname}: parametric solutions of nonlinear Diophantine equations are not "
                                      "supported")
        pl = sorted(params, key=str)
        ren = {p: _c(i + 1) for i, p in enumerate(pl)}
        res = [{k: v.xreplace(ren) for k, v in s_.items()} for s_ in res]
        res = [{k: (v, "C") for k, v in s_.items()} for s_ in res]
    for s_ in res:
        sub = {k: (v[0] if isinstance(v, tuple) else v) for k, v in s_.items()}
        if sp.expand(f.subs(sub)) != 0:
            raise ArithmeticError(f"{fname}: a solution failed verification")
    res.sort(key=lambda s_: tuple(_wl_order_key(s_[x][0] if isinstance(s_[x], tuple) else s_[x]) for x in xs))
    return res


def _solve_rules(xs_given, xs, sols):
    rows = []
    for s_ in sols:
        rr = []
        for x in xs:
            if x in s_:
                rr.append(_cond_rule(Symbol(x.name), s_[x]))
        rows.append(L(*rr))
    return L(*rows)


def _solve_args(a, fname):
    args, opts = split_opts(a, {"Cubics", "Quartics", "VerifySolutions", "InverseFunctions", "Method"}, fname,
                            extra_known=("MaxExtraConditions", "Modulus", "GeneratedParameters"))
    if "Method" in opts:
        raise NotImplementedError(f"{fname}: the option Method is not supported")
    if not args or len(args) > 3:
        raise ValueError(f"{fname} expects {fname}[eqns, vars] or {fname}[eqns, vars, dom]")
    eqs = args[0]
    vs = args[1] if len(args) >= 2 else None
    dom = _dom_name(args[2], fname) if len(args) == 3 else None
    return eqs, vs, dom, opts


@kernel_override("Solve", AREA,
                 "Solve[eqns, vars], Solve[eqns, vars, dom] (Reals, Integers, Complexes): polynomial roots with "
                 "multiplicity in Wolfram order (real ascending, then complex), radicals up to degree 4 (Cubics/"
                 "Quartics -> False give Root objects) and Root[...] for higher degree (legacy returned {} for "
                 "x^5 - x + 1 == 0), periodic families as ConditionalExpression[..., C[1] ∈ Integers] (legacy "
                 "returned only {0, Pi} for Sin[x] == 0), denominators' zeros excluded, polynomial/rational systems "
                 "(underdetermined systems solved for the later variables, as Wolfram), linear and finite quadratic "
                 "Diophantine equations over the Integers; every solution is substituted back; inequalities raise "
                 "(use Reduce); transcendental equations SymPy cannot solve completely raise instead of returning "
                 "a partial list")
def _b_solve(a, ev, expr):
    eqs, vs, dom, opts = _solve_args(a, "Solve")
    xs, sols = _solve_core(eqs, vs, dom, "Solve", opts)
    if xs is None:
        return L()
    return _solve_rules(vs, xs, sols)


@fn("SolveValues", area=AREA, status="added",
    notes="SolveValues[eqns, x] -> {v1, v2, ...}; SolveValues[eqns, {x, y}] -> {{x1, y1}, ...}; same solver and "
          "checks as Solve (domains, multiplicity, periodic families)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _solvevalues(*a: Expr) -> ListOf[Expr]:
    eqs, vs, dom, opts = _solve_args(a, "SolveValues")
    xs, sols = _solve_core(eqs, vs, dom, "SolveValues", opts)
    if xs is None:
        return L()
    rules = _solve_rules(vs, xs, sols)
    if vs is not None and not is_list(vs):
        return L(*(r.args[0].args[1] for r in rules.args if r.args))
    return L(*(L(*(x.args[1] for x in r.args)) for r in rules.args))


@fn("Roots", area=AREA, status="added",
    notes="Roots[lhs == rhs, x] -> x == r1 || x == r2 || ... for polynomial equations (repeated roots repeated, as "
          "Wolfram; radicals up to degree 4 unless Cubics/Quartics -> False, Root objects beyond); a single root "
          "gives x == r; no roots gives False"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _roots(eq: Expr, x: Var, *rest: Expr) -> Expr:
    rest, opts = split_opts(rest, {"Cubics", "Quartics"}, "Roots", extra_known=("Modulus", "Multiplicity"))
    kind, f = _rel_parts(eq, "Roots")
    if kind != "eq":
        raise ValueError("Roots: expected an equation")
    v = variable(x, "Roots")
    num, den = sp.fraction(sp.together(f))
    if not (sp.expand(num).is_polynomial(v) and not den.has(v)):
        raise ValueError("Roots: the equation must be polynomial in the variable")
    rts = _poly_roots(sp.expand(num), v, not is_false(opts.get("Cubics", True)),
                      not is_false(opts.get("Quartics", True)))
    if rts is None:
        return True
    if not rts:
        return False
    terms = [Expr("Equal", Symbol(v.name), out(r)) for r in rts]
    return terms[0] if len(terms) == 1 else Expr("Or", *terms)


@fn("ToRules", area=AREA, status="added",
    notes="ToRules[eqns]: x == a && y == b -> {x -> a, y -> b}; a disjunction gives one rule list per case as a "
          "Sequence (so {ToRules[...]} is a Solve-style list); False gives nothing"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _torules(e: Expr) -> Expr:
    if e is False:
        return Expr("Sequence")
    if e is True:
        return Expr("Sequence", L())
    cases = list(e.args) if isinstance(e, Expr) and e.head == "Or" else [e]
    res = []
    for c in cases:
        rr = []
        for p in _flatten_and(c):
            if isinstance(p, Expr) and p.head == "Equal" and len(p.args) == 2:
                rr.append(rule(p.args[0], p.args[1]))
            elif isinstance(p, Expr) and p.head == "Element":
                continue
            else:
                raise ValueError("ToRules: expected equations combined with And/Or")
        res.append(L(*rr))
    return res[0] if len(res) == 1 else Expr("Sequence", *res)


@fn("Eliminate", area=AREA, status="added",
    notes="Eliminate[eqns, vars]: polynomial equations with vars eliminated (lexicographic Gröbner basis "
          "elimination), returned as p == 0 relations joined with And (Wolfram may arrange the two sides "
          "differently); True when nothing remains"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _eliminate(eqs: Expr, vs: Var | ListOf[Var]) -> Expr:
    parts = [_rel_parts(e, "Eliminate") for e in _flatten_and(eqs)]
    if any(k != "eq" for k, _ in parts):
        raise ValueError("Eliminate: expected equations")
    ps = [sp.expand(sp.fraction(sp.together(v))[0]) for _, v in parts]
    elim = _vars_opt(vs, "Eliminate")
    others = sorted(set().union(*(p.free_symbols for p in ps)) - set(elim), key=str)
    if not all(p.is_polynomial(*(elim + others)) for p in ps):
        raise NotImplementedError("Eliminate: only polynomial equations are supported")
    G = sp.groebner(ps, *(elim + others), order="lex")
    keep = []
    for g in G.exprs:
        if g.has(*elim):
            continue
        P = sp.Poly(g, *others) if others else None
        if P is not None:
            g = P.primitive()[1].as_expr()
            if sp.Poly(g, *others).LC() < 0:
                g = -g
        keep.append(sp.expand(g))
    if any(g.is_number and g != 0 for g in keep):
        return False
    rels = [Expr("Equal", out(g), 0) for g in reversed(keep)]
    if not rels:
        return True
    return rels[0] if len(rels) == 1 else Expr("And", *rels)


@fn("CountRoots", area=AREA, status="added",
    notes="CountRoots[p, x]: number of real roots counted with multiplicity; CountRoots[p, {x, a, b}] in the "
          "closed interval [a, b] (real a, b) or the closed rectangle with corners a, b (complex a, b); exact "
          "(Sturm sequences, SymPy)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _countroots(p: Expr, spec: Expr) -> Expr:
    if is_list(spec):
        if len(spec.args) != 3:
            raise ValueError("CountRoots: give {x, a, b}")
        x = variable(spec.args[0], "CountRoots")
        a, b = sym(spec.args[1], "CountRoots"), sym(spec.args[2], "CountRoots")
    else:
        x, a, b = variable(spec, "CountRoots"), None, None
    P = sp.expand(sym(p, "CountRoots"))
    if not P.is_polynomial(x) or P.free_symbols - {x}:
        raise ValueError("CountRoots: expected a polynomial with numeric coefficients")
    if P.has(sp.Float):
        P = sp.nsimplify(P, rational=True)
    if P == 0:
        raise ValueError("CountRoots: the zero polynomial has infinitely many roots")
    total = 0
    for f, k in sp.sqf_list(P, x)[1]:
        Fp = sp.Poly(f, x)
        if a is None:
            c = Fp.count_roots()
        elif a.is_real and b.is_real:
            c = Fp.count_roots(a, b)
        else:
            c = Fp.count_roots(a, b)
        total += int(c) * int(k)
    return total


# ───────────────────────────── Reduce ─────────────────────────────

def _set_to_rel(S, x, integers=False):
    """SymPy subset of the reals -> MikoshiLang relation in x (Or of intervals / equations)."""
    X = Symbol(x.name)
    if S is sp.S.EmptySet:
        return False
    if S == sp.S.Reals:
        return True if not integers else Expr("Element", X, Symbol("Integers"))
    parts = S.args if isinstance(S, sp.Union) else (S,)
    rels = []
    for P in parts:
        if isinstance(P, sp.FiniteSet):
            for v in sorted(P.args, key=_wl_order_key):
                rels.append(Expr("Equal", X, out(v)))
        elif isinstance(P, sp.Interval):
            if integers:
                lo = sp.ceiling(P.start) if P.start.is_finite else None
                hi = sp.floor(P.end) if P.end.is_finite else None
                if lo is not None and P.left_open and lo == P.start:
                    lo += 1
                if hi is not None and P.right_open and hi == P.end:
                    hi -= 1
                if lo is not None and hi is not None:
                    if hi - lo <= 200:
                        rels.extend(Expr("Equal", X, out(sp.Integer(k))) for k in range(int(lo), int(hi) + 1))
                        continue
                cs = [Expr("Element", X, Symbol("Integers"))]
                if lo is not None:
                    cs.append(Expr("GreaterEqual", X, out(lo)))
                if hi is not None:
                    cs.append(Expr("LessEqual", X, out(hi)))
                rels.append(Expr("And", *cs))
                continue
            cs = []
            if P.start.is_finite and not P.end.is_finite:
                cs.append(Expr("Greater" if P.left_open else "GreaterEqual", X, out(P.start)))
            elif P.start.is_finite:
                cs.append(Expr("Less" if P.left_open else "LessEqual", out(P.start), X))
            if P.end.is_finite:
                cs.append(Expr("Less" if P.right_open else "LessEqual", X, out(P.end)))
            if P.start == P.end:
                rels.append(Expr("Equal", X, out(P.start)))
            elif not cs:
                rels.append(True)
            else:
                rels.append(cs[0] if len(cs) == 1 else Expr("And", *cs))
        else:
            raise NotImplementedError("Reduce: the solution set has an unsupported form")
    if not rels:
        return False
    return rels[0] if len(rels) == 1 else Expr("Or", *rels)


def _check_real_set(cond, x, S):
    """Sample check that S is exactly the solution set of cond on the reals (points inside and outside, including
    boundary points)."""
    pts = set(sp.Rational(k, 4) for k in range(-60, 61)) | {sp.Integer(k) for k in (-1000, -100, 100, 1000)}
    for b in S.boundary if not isinstance(S, sp.FiniteSet) else S:
        if b.is_real and b.is_finite:
            for d in (0, sp.Rational(1, 1000), -sp.Rational(1, 1000)):
                pts.add(b + d)
    for p in pts:
        try:
            truth = cond.subs(x, p)
            truth = sp.simplify(truth) if not isinstance(truth, sp.logic.boolalg.BooleanAtom) else truth
        except Exception:
            continue
        if truth not in (sp.true, sp.false):
            try:
                truth = sp.true if bool(sp.N(cond.subs(x, p), 30)) else sp.false
            except Exception:
                continue
        inside = S.contains(p)
        if inside not in (sp.true, sp.false):
            continue
        if (truth is sp.true) != (inside is sp.true):
            raise ArithmeticError("Reduce: the solution set failed verification")


def _rel_sym(e, fname):
    """MikoshiLang logical formula of (in)equations -> SymPy boolean."""
    if e is True:
        return sp.true
    if e is False:
        return sp.false
    if isinstance(e, Expr) and e.head in ("And", "Or", "Not"):
        args = [_rel_sym(a, fname) for a in e.args]
        return {"And": sp.And, "Or": sp.Or, "Not": sp.Not}[e.head](*args)
    if is_list(e):
        return sp.And(*(_rel_sym(a, fname) for a in e.args))
    if isinstance(e, Expr) and e.head == "Equal":
        if len(e.args) != 2:
            raise NotImplementedError(f"{fname}: chained equations are not supported")
        return sp.Eq(sym(e.args[0], fname), sym(e.args[1], fname))
    if isinstance(e, Expr) and e.head in _REL_SP:
        return _rel_from(e, fname)
    if isinstance(e, Expr) and e.head in ("Exists", "ForAll"):
        raise NotImplementedError(f"{fname}: quantifiers are not supported")
    raise NotImplementedError(f"{fname}: unsupported condition {e!r}")


def _bool_set(c, x):
    """Solution set in the reals of a boolean combination of (in)equations in x (x real)."""
    if c is sp.true:
        return sp.S.Reals
    if c is sp.false:
        return sp.S.EmptySet
    if isinstance(c, sp.And):
        S = sp.S.Reals
        for a in c.args:
            S = sp.Intersection(S, _bool_set(a, x))
        return S
    if isinstance(c, sp.Or):
        return sp.Union(*(_bool_set(a, x) for a in c.args))
    if isinstance(c, sp.Not):
        return sp.Complement(sp.S.Reals, _bool_set(c.args[0], x))
    if isinstance(c, sp.Eq):
        S = sp.solveset(c.lhs - c.rhs, x, sp.S.Reals)
    else:
        S = sp.solveset(c, x, sp.S.Reals)
    if isinstance(S, sp.ConditionSet) or S.has(sp.ConditionSet):
        raise NotImplementedError("cannot solve the condition")
    return S


def _rational_in(cond, x):
    for r in cond.atoms(sp.Rel):
        f = sp.together(r.lhs - r.rhs)
        n_, d_ = sp.fraction(f)
        if not (sp.expand(n_).is_polynomial(x) and sp.expand(d_).is_polynomial(x)):
            if not (r.has(sp.Abs) and all(sp.fraction(sp.together(a.args[0]))[0].is_polynomial(x)
                                          for a in r.atoms(sp.Abs))):
                return False
    return True


def _reduce_univariate(cond, x, dom, fname):
    rels = list(cond.atoms(sp.Rel))
    eq_only = all(isinstance(r, sp.Eq) for r in rels) and isinstance(cond, (sp.Eq, sp.And)) and \
        not isinstance(cond, sp.Or)
    if isinstance(cond, sp.Eq) and dom in (None, "Complexes"):
        f = cond.lhs - cond.rhs
        sols = _univariate(f, x, dom)
        if any(isinstance(s_, tuple) for s_ in sols):
            fam = [s_ for s_ in sols if isinstance(s_, tuple)]
            fin = [s_ for s_ in sols if not isinstance(s_, tuple)]
            eqs = [Expr("Equal", Symbol(x.name), K._constants_to_c(out(v))) for v in fin] + \
                  [Expr("Equal", Symbol(x.name), K._constants_to_c(out(v[0]))) for v in fam]
            body = eqs[0] if len(eqs) == 1 else Expr("Or", *eqs)
            return Expr("And", Expr("Element", Expr("C", 1), Symbol("Integers")), body)
        uniq = []
        for s_ in sols:
            if not any(sp.simplify(s_ - u) == 0 for u in uniq):
                uniq.append(s_)
        if not uniq:
            return False
        eqs = [Expr("Equal", Symbol(x.name), out(v)) for v in uniq]
        return eqs[0] if len(eqs) == 1 else Expr("Or", *eqs)
    if dom in (None, "Complexes") and not eq_only:
        dom = "Reals"           # inequalities imply real variables (Wolfram)
    if dom in ("Reals", "Integers"):
        if not _rational_in(cond, x):
            raise NotImplementedError(f"{fname}: only polynomial, rational and Abs inequalities are supported")
        xr = sp.Symbol(x.name, real=True)
        c = cond.xreplace({x: xr})
        try:
            S = _bool_set(c, xr)
        except Exception as exc:
            raise NotImplementedError(f"{fname}: cannot reduce this condition ({exc})") from exc
        if isinstance(S, sp.ConditionSet) or S.has(sp.ConditionSet):
            raise NotImplementedError(f"{fname}: cannot reduce this condition")
        S = S.xreplace({xr: x})
        _check_real_set(cond, x, S)
        return _set_to_rel(S, x, integers=(dom == "Integers"))
    raise NotImplementedError(f"{fname}: domain {dom} is not supported")


def _linear_parametric(f, x):
    """a x + b == 0 with symbolic a, b -> (b == 0 && a == 0) || (a != 0 && x == -b/a)."""
    P = sp.Poly(sp.expand(f), x)
    if P.degree() != 1:
        return None
    a, b = P.all_coeffs()
    if a.is_number:
        return None
    X = Symbol(x.name)
    first = Expr("And", Expr("Equal", out(b), 0), Expr("Equal", out(a), 0)) if b != 0 else \
        Expr("Equal", out(a), 0)
    second = Expr("And", Expr("Unequal", out(a), 0), Expr("Equal", X, out(sp.simplify(-b / a))))
    return Expr("Or", first, second)


@fn("Reduce", area=AREA, status="added",
    notes="Reduce[cond, x] / Reduce[cond, x, dom] in one variable: polynomial equations over the complexes "
          "(x == r1 || ..., distinct roots), periodic equations (C[1] ∈ Integers && (x == ...)), polynomial, "
          "rational and Abs inequalities with And/Or over the reals (inequalities imply real x, as Wolfram) as "
          "interval conditions, and over the integers (finite ranges listed, else x ∈ Integers with bounds); "
          "linear equations with symbolic coefficients split into cases ((b == 0 && a == 0) || (a != 0 && x == "
          "-b/a)); polynomial equation systems in several variables (Or of And of equations). Results of "
          "inequalities are checked by sampling; quantifiers and multivariate inequalities raise"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _reduce(cond: Expr, vs: Expr = None, dom: Expr = None, *rest: Expr) -> Expr:
    d = _dom_name(dom, "Reduce")
    c = _rel_sym(cond, "Reduce")
    if c in (sp.true, sp.false):
        return c is sp.true
    if vs is None:
        xs = sorted(c.free_symbols, key=str)
    else:
        xs = [variable(v, "Reduce") for v in (vs.args if is_list(vs) else [vs])]
    if len(xs) == 1:
        x = xs[0]
        if isinstance(c, sp.Eq) and d in (None, "Complexes"):
            f = c.lhs - c.rhs
            n_, d_ = sp.fraction(sp.together(f))
            if sp.expand(n_).is_polynomial(x) and not d_.has(x):
                lp = _linear_parametric(n_, x)
                if lp is not None:
                    return lp
                if sp.expand(n_).free_symbols - {x}:
                    raise NotImplementedError("Reduce: parametric polynomial equations of degree > 1 are not "
                                              "supported")
        return _reduce_univariate(c, x, d, "Reduce")
    # several variables: equation systems only
    atoms = [a for a in sp.preorder_traversal(c) if isinstance(a, sp.Rel)]
    if not all(isinstance(a, sp.Eq) for a in atoms) or c.has(sp.Or) or c.has(sp.Not):
        raise NotImplementedError("Reduce: inequalities and disjunctions in several variables are not supported")
    eq_exprs = L(*(Expr("Equal", out(a.lhs), out(a.rhs)) for a in atoms))
    xs_, sols = _solve_core(eq_exprs, L(*(Symbol(x.name) for x in xs)), d, "Reduce", {})
    if xs_ is None or not sols:
        return False
    cases = []
    for s_ in sols:
        if len(s_) != len(xs):
            raise NotImplementedError("Reduce: underdetermined systems are not supported")
        if any(isinstance(v, tuple) for v in s_.values()):
            raise NotImplementedError("Reduce: parametric integer solutions are not supported")
        eqs = [Expr("Equal", Symbol(x.name), out(s_[x])) for x in xs]
        cases.append(Expr("And", *eqs))
    return cases[0] if len(cases) == 1 else Expr("Or", *cases)


# ───────────────────────────── optimisation and function properties ─────────────────────────────

def _feasible_set(cons, x, fname):
    c = _rel_sym(cons, fname)
    xr = sp.Symbol(x.name, real=True)
    cc = c.xreplace({x: xr})
    if not _rational_in(c, x):
        raise NotImplementedError(f"{fname}: only polynomial, rational and Abs constraints are supported")
    try:
        S = _bool_set(cc, xr)
    except Exception:
        raise NotImplementedError(f"{fname}: cannot solve the constraints")
    S = S.xreplace({xr: x})
    _check_real_set(c, x, S)
    return S


def _extremum_1d(f, x, S, kind, fname):
    """(value, argument or None if not attained) of the inf/sup of f over S (a union of intervals/points)."""
    sign = 1 if kind == "min" else -1
    g = sign * f
    parts = S.args if isinstance(S, sp.Union) else (S,)
    best = None            # (value, arg or None)
    for P in parts:
        cands = []
        if isinstance(P, sp.FiniteSet):
            cands = [(g.subs(x, p), p) for p in P.args]
        elif isinstance(P, sp.Interval):
            try:
                crit = sp.solveset(sp.diff(g, x), x, P)
            except Exception:
                crit = None
            if not isinstance(crit, sp.FiniteSet):
                if crit is sp.S.EmptySet:
                    crit = sp.FiniteSet()
                else:
                    raise NotImplementedError(f"{fname}: cannot find the critical points")
            sing = sp.solveset(sp.fraction(sp.together(g))[1], x, P)
            if isinstance(sing, sp.FiniteSet) and sing.args:
                raise NotImplementedError(f"{fname}: the objective has singularities in the feasible region")
            cands = [(g.subs(x, p), p) for p in crit.args]
            for end, opn in ((P.start, P.left_open), (P.end, P.right_open)):
                if end.is_finite and not opn:
                    cands.append((g.subs(x, end), end))
                else:
                    lim = sp.limit(g, x, end, "+" if end == P.start else "-") if end.is_finite else \
                        sp.limit(g, x, end)
                    if lim.has(sp.AccumBounds) or lim is sp.nan:
                        raise NotImplementedError(f"{fname}: the objective has no limit at a boundary")
                    cands.append((lim, ("limit", end)))
        for val, arg in cands:
            val = sp.simplify(val)
            if best is None or bool(sp.N(val - best[0], 30) < -1e-25) if (val.is_finite and best is not None and
                                                                       best[0].is_finite) else \
                    (best is None or val == -sp.oo or (best[0] == sp.oo)):
                best = (val, arg)
            elif best is not None and not isinstance(arg, tuple) and isinstance(best[1], tuple) and \
                    val.is_finite and best[0].is_finite and abs(complex(sp.N(val - best[0], 30))) < 1e-25:
                best = (val, arg)
    if best is None:
        raise NotImplementedError(f"{fname}: empty feasible region")
    return sign * best[0], best[1]


def _opt_args(obj, vs, fname):
    if is_list(obj):
        if len(obj.args) < 1:
            raise ValueError(f"{fname}: empty objective")
        f = obj.args[0]
        cons = Expr("And", *obj.args[1:]) if len(obj.args) > 2 else (obj.args[1] if len(obj.args) == 2 else True)
    else:
        f, cons = obj, True
    xs = [variable(v, fname) for v in (vs.args if is_list(vs) else [vs])]
    return f, cons, xs


def _optimize(obj, vs, kind, fname, dom=None):
    f, cons, xs = _opt_args(obj, vs, fname)
    if dom is not None:
        raise NotImplementedError(f"{fname}: a domain argument is not supported")
    fv = sym(f, fname)
    if cons is True:
        return None             # unconstrained: the reviewed implementation (unchanged)
    if len(xs) != 1:
        raise NotImplementedError(f"{fname}: constrained optimisation in several variables is not supported")
    x = xs[0]
    S = sp.S.Reals if cons is True else _feasible_set(cons, x, fname)
    if S is sp.S.EmptySet:
        return (sp.oo if kind == "min" else -sp.oo), None, x
    val, arg = _extremum_1d(fv, x, S, kind, fname)
    return val, arg, x


_orig_minimize = None
_orig_maximize = None


def _opt_entry(kind):
    from .core import REGISTRY
    name = "Minimize" if kind == "min" else "Maximize"
    impls = list(REGISTRY[name].impls)

    def call(*args):
        for f_, lo, hi in impls:
            if len(args) >= lo and (hi is None or len(args) <= hi):
                return f_(*args)
        raise ValueError(f"{name} called with {len(args)} argument(s)")
    return call


_MIN_ORIG = _opt_entry("min")
_MAX_ORIG = _opt_entry("max")


def _opt_result(obj, vs, kind, fname):
    res = _optimize(obj, vs, kind, fname)
    if res is None:
        r = (_MIN_ORIG if kind == "min" else _MAX_ORIG)(obj, vs)
        return r
    val, arg, x = res
    X = Symbol(x.name)
    if arg is None:
        return L(out(val), L(rule(X, Symbol("Indeterminate"))))
    if isinstance(arg, tuple):          # not attained: Wolfram returns the boundary point (with a warning)
        arg = arg[1]
    return L(out(val), L(rule(X, out(arg))))


@replace("Minimize", AREA,
         "Minimize[f, x] / Minimize[f, {x, ...}] (unconstrained, as before) and batch 4: Minimize[{f, cons}, x] "
         "for one variable with polynomial/rational/Abs constraints: global minimum over the feasible set from "
         "critical points, closed endpoints and limits at open ends and ±Infinity -> {min, {x -> x0}}; an "
         "infimum that is not attained gives the boundary point ({0, {x -> 0}} for {x, x > 0}; {-Infinity, {x -> "
         "-Infinity}} when unbounded), as Wolfram (which also warns); constraints in several "
         "variables raise")
def _minimize(obj: Expr, vs: Expr, *rest: Expr) -> Expr:
    return _opt_result(obj, vs, "min", "Minimize")


@replace("Maximize", AREA,
         "Maximize[f, x] (unconstrained, as before) and batch 4: Maximize[{f, cons}, x] for one variable, as "
         "Minimize")
def _maximize(obj: Expr, vs: Expr, *rest: Expr) -> Expr:
    return _opt_result(obj, vs, "max", "Maximize")


def _value_of(kind, obj, vs, fname, arg=False):
    r = _opt_result(obj, vs, kind, fname)
    if not (is_list(r) and len(r.args) == 2):
        raise Unevaluated()
    if not arg:
        return r.args[0]
    rules = r.args[1]
    vals = [x.args[1] for x in rules.args]
    return vals[0] if not is_list(vs) else L(*vals)


@fn("MinValue", area=AREA, status="added", notes="MinValue[f, x] / MinValue[{f, cons}, x]: the minimum (or "
    "infimum) value, as Minimize"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _minvalue(obj: Expr, vs: Expr, *rest: Expr) -> Expr:
    return _value_of("min", obj, vs, "MinValue")


@fn("MaxValue", area=AREA, status="added", notes="MaxValue[f, x] / MaxValue[{f, cons}, x]: the maximum (or "
    "supremum) value, as Maximize"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _maxvalue(obj: Expr, vs: Expr, *rest: Expr) -> Expr:
    return _value_of("max", obj, vs, "MaxValue")


@fn("ArgMin", area=AREA, status="added", notes="ArgMin[f, x] / ArgMin[{f, cons}, x]: a minimising point "
    "(the boundary point if the infimum is not attained), as Minimize; {x, y} gives a list"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _argmin(obj: Expr, vs: Expr, *rest: Expr) -> Expr:
    return _value_of("min", obj, vs, "ArgMin", arg=True)


@fn("ArgMax", area=AREA, status="added", notes="ArgMax[f, x] / ArgMax[{f, cons}, x]: a maximising point, as "
    "Maximize"
    ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _argmax(obj: Expr, vs: Expr, *rest: Expr) -> Expr:
    return _value_of("max", obj, vs, "ArgMax", arg=True)


_PIECEWISE_HEADS = (sp.floor, sp.ceiling, sp.sign, sp.Heaviside, sp.Piecewise, sp.frac, sp.Max, sp.Min)


@fn("FunctionDomain", area=AREA, status="added",
    notes="FunctionDomain[f, x]: the largest real domain of f as a condition on x (FunctionDomain[1/x, x] -> x < 0 "
          "|| x > 0, FunctionDomain[Log[x^2 - 1], x] -> x < -1 || x > 1) for elementary f (SymPy continuous "
          "domain; functions with jumps such as Floor raise); checked by sampling; other domains raise"
          ". Check: exact, self-check (the result is verified before it is returned), reference (differential cases "
          "in tests/library).")
def _functiondomain(f: Expr, x: Var, dom: Expr = None) -> Expr:
    from sympy.calculus.util import continuous_domain
    if dom is not None and not (isinstance(dom, Symbol) and dom.name == "Reals"):
        raise NotImplementedError("FunctionDomain: only the real domain is supported")
    xv = variable(x, "FunctionDomain")
    v = sym(f, "FunctionDomain")
    if v.has(*_PIECEWISE_HEADS):
        raise NotImplementedError("FunctionDomain: functions with jumps are not supported")
    if v.free_symbols - {xv}:
        raise NotImplementedError("FunctionDomain: parameters are not supported")
    xr = sp.Symbol(xv.name, real=True)
    S = continuous_domain(v.xreplace({xv: xr}), xr, sp.S.Reals).xreplace({xr: xv})
    # sample check: inside points give a finite real value, outside points do not
    for k in range(-40, 41):
        p = sp.Rational(k, 4) + sp.Rational(1, 997)
        val = v.subs(xv, p)
        try:
            z = complex(sp.N(val, 30))
            real_ok = abs(z.imag) < 1e-20 and z == z and abs(z) != float("inf")
        except Exception:
            real_ok = False
        if bool(S.contains(p) == sp.true) != real_ok:
            raise ArithmeticError("FunctionDomain: the domain failed verification")
    return _set_to_rel(S, xv)


@fn("FunctionRange", area=AREA, status="added",
    notes="FunctionRange[f, x, y]: the range of the real function f as a condition on y (FunctionRange[x^2, x, y] "
          "-> y >= 0, FunctionRange[Sin[x], x, y] -> -1 <= y <= 1) via SymPy function_range over the domain; "
          "elementary f in one variable without parameters"
          ". Check: exact, self-check (the result is verified before it is returned), reference (differential cases "
          "in tests/library).")
def _functionrange(f: Expr, x: Var, y: Var, *rest: Expr) -> Expr:
    from sympy.calculus.util import function_range, continuous_domain
    xv, yv = variable(x, "FunctionRange"), variable(y, "FunctionRange")
    v = sym(f, "FunctionRange")
    if v.free_symbols - {xv} or v.has(*_PIECEWISE_HEADS):
        raise NotImplementedError("FunctionRange: parameters and functions with jumps are not supported")
    xr = sp.Symbol(xv.name, real=True)
    vr = v.xreplace({xv: xr})
    D = continuous_domain(vr, xr, sp.S.Reals)
    try:
        Rg = function_range(vr, xr, D)
    except Exception as exc:
        raise NotImplementedError(f"FunctionRange: cannot determine the range ({exc})") from exc
    if not isinstance(Rg, (sp.Interval, sp.Union, sp.FiniteSet)) or Rg.has(sp.ConditionSet):
        raise NotImplementedError("FunctionRange: cannot determine the range")
    # sanity check: sampled values lie in the range
    for k in range(-30, 31):
        p = sp.Rational(k, 3) + sp.Rational(1, 991)
        if D.contains(p) != sp.true:
            continue
        val = sp.N(vr.subs(xr, p), 30)
        if val.is_real and Rg.contains(sp.nsimplify(val, rational=True)) == sp.false:
            try:
                if not any(abs(float(val - b)) < 1e-12 for b in Rg.boundary):
                    raise ArithmeticError("FunctionRange: the range failed verification")
            except TypeError:
                raise ArithmeticError("FunctionRange: the range failed verification")
    return _set_to_rel(Rg, yv)


@fn("FunctionPeriod", area=AREA, status="added",
    notes="FunctionPeriod[f, x]: the fundamental period (FunctionPeriod[Sin[x]^2, x] -> Pi; SymPy's periodicity "
          "can return a multiple, so T/k is tried for k = 2..24 and verified symbolically/numerically); "
          "non-periodic polynomials and rational functions give 0 (Wolfram); undecided cases stay unevaluated"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _functionperiod(f: Expr, x: Var) -> Expr:
    from sympy.calculus.util import periodicity
    xv = variable(x, "FunctionPeriod")
    v = sym(f, "FunctionPeriod")
    if not v.has(xv):
        return 0
    if sp.together(v).is_rational_function(xv):
        return 0
    try:
        T = periodicity(v, xv)
    except Exception:
        T = None
    if T is None or T == 0:
        raise Unevaluated()

    def is_period(t):
        d = v.subs(xv, xv + t) - v
        try:
            if sp.simplify(d) == 0:
                return True
        except Exception:
            pass
        return _numeric_equal(v.subs(xv, xv + t), v, n=8, tol=1e-25)
    if not is_period(T):
        raise Unevaluated()
    changed = True
    while changed:
        changed = False
        for k in range(24, 1, -1):
            if is_period(T / k):
                T = T / k
                changed = True
                break
    return out(sp.simplify(T))
