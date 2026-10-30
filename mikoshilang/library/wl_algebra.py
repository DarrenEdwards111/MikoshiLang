"""Coverage batch 4: polynomial algebra, simplification, solving and reduction (Wolfram semantics).

Every function either returns a result checked against its definition (by construction with SymPy's exact
polynomial algorithms, or by substituting back / numerically comparing), raises a clear error, or stays
unevaluated.  Known representational limit: the kernel (like SymPy) distributes a rational number over a sum,
so 2 (x^2 - 2) displays as -4 + 2 x^2; FactorTermsList keeps the factors apart.
"""
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
          "(ExpandAll[Sin[(x + 1)^2]] -> Sin[1 + 2 x + x^2]); ExpandAll[e, x] only parts containing x")
def _expandall(e, *patt):
    th = _thread(e, lambda y: _expandall(y, *patt))
    if th is not None:
        return th
    return out(_expand_all(sym(e, "ExpandAll"), _patt_var(patt, "ExpandAll")))


@fn("ExpandNumerator", area=AREA, status="added",
    notes="expands the numerator of a rational expression (Numerator/Denominator as Wolfram), denominator kept")
def _expandnum(e):
    th = _thread(e, _expandnum)
    if th is not None:
        return th
    n, d = sp.fraction(sym(e, "ExpandNumerator"))
    return out(_expand_top(n) / d)


@fn("ExpandDenominator", area=AREA, status="added",
    notes="expands the denominator of a rational expression (ExpandDenominator[1/((x + 1) (x - 1))] -> "
          "1/(-1 + x^2)), numerator kept")
def _expandden(e):
    th = _thread(e, _expandden)
    if th is not None:
        return th
    n, d = sp.fraction(sym(e, "ExpandDenominator"))
    dd = _expand_top(d)
    return out(sp.Mul(n, sp.Pow(dd, -1, evaluate=False), evaluate=False) if dd != 1 else n)


# ───────────────────────────── structure ─────────────────────────────

@fn("PolynomialQ", area=AREA, status="added",
    notes="PolynomialQ[e, x] / PolynomialQ[e, {x, ...}]: True if e is a polynomial in the variables (coefficients "
          "may be anything free of them; nonnegative integer powers only); PolynomialQ[e] in all its variables")
def _polyq(e, v=None):
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
          "Sin[z]) of a polynomial or rational expression, sorted (Variables[x^2 + y Sin[z]] -> {x, y, Sin[z]})")
def _variables_fn(e):
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
          "CoefficientRules[p] uses Variables[p]; CoefficientRules[0, vars] -> {}")
def _coeffrules(e, v=None, order=None):
    P, gens = _poly(e, v, "CoefficientRules")
    if P.is_zero:
        return L()
    terms = P.terms(order=_order(order, "CoefficientRules"))
    return L(*(rule(L(*(int(k) for k in m)), out(c)) for m, c in terms))


@fn("MonomialList", area=AREA, status="added",
    notes="MonomialList[p, {x1, ...}] -> the terms (coefficient times monomial) in lexicographic order (third "
          "argument: monomial order); MonomialList[p] uses Variables[p]")
def _monolist(e, v=None, order=None):
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
def _exponent(e, x, h=None):
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
          "numerators over lcm of denominators, as Wolfram); Modulus -> p; PolynomialGCD[p] = p")
def _pgcd(*args):
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
    notes="least common multiple of polynomials (factored, as Wolfram); Modulus -> p")
def _plcm(*args):
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
          "deg remainder < deg q (rational coefficients); a zero divisor raises")
def _pqr(p, q, x):
    quo, rem = _AL._polydiv(p, q, x, "PolynomialQuotientRemainder")
    return L(out(quo), out(rem))


@fn("PolynomialReduce", area=AREA, status="added",
    notes="PolynomialReduce[p, {p1, ...}, {x1, ...}] -> {{a1, ...}, b} with p = Σ ai pi + b and b reduced with "
          "respect to the leading terms (lexicographic order, SymPy reduced; for a Gröbner basis the remainder "
          "is unique)")
def _preduce(p, qs, vs, *rest):
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
          "moduli apply in turn")
def _pmod(p, m):
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
          "side; Modulus -> p")
def _groebner(polys, vs, *rest):
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
          "then the irreducible factors over the rationals with exponents; Extension, GaussianIntegers, Modulus")
def _factorlist(e, *rest):
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
          "coefficient of the remaining polynomial is made positive")
def _factorterms(e, v=None):
    c, g, r = _content_split(sym(e, "FactorTerms"), _vars_opt(v, "FactorTerms"))
    return out(sp.Mul(c, g, r, evaluate=False) if (g != 1 or c != 1) else r)


@fn("FactorTermsList", area=AREA, status="added",
    notes="FactorTermsList[p] -> {numeric factor, polynomial}; FactorTermsList[p, {x, ...}] -> {numeric factor, "
          "factor free of the variables, remaining polynomial}")
def _factortermslist(e, v=None):
    vs = _vars_opt(v, "FactorTermsList")
    c, g, r = _content_split(sym(e, "FactorTermsList"), vs)
    if vs:
        return L(out(c), out(g), out(r))
    return L(out(c), out(r))


@fn("FactorSquareFree", area=AREA, status="added",
    notes="square-free factorisation (SymPy sqf): FactorSquareFree[x^3 - x^2 - x + 1] -> (-1 + x)^2 (1 + x)")
def _sqf(e, *rest):
    rest, opts = split_opts(rest, {"Modulus"}, "FactorSquareFree")
    m = _modulus(opts, "FactorSquareFree")
    p = sym(e, "FactorSquareFree")
    return out(sp.sqf(p, modulus=m) if m else sp.sqf(p))


@fn("FactorSquareFreeList", area=AREA, status="added",
    notes="FactorSquareFreeList[p] -> {{c, 1}, {f1, 1}, {f2, 2}, ...} (numeric content first, then square-free "
          "factors with their multiplicities)")
def _sqflist(e):
    p = sym(e, "FactorSquareFreeList")
    gens = sorted(p.free_symbols, key=str)
    if not gens:
        return L(L(out(p), 1))
    c, facs = sp.sqf_list(p, *gens)
    return L(L(out(c), 1), *(L(out(f), int(k)) for f, k in facs))


@fn("Decompose", area=AREA, status="added",
    notes="Decompose[p, x] -> {p1, ..., pn} with p = p1[p2[...pn[x]]] (functional decomposition, SymPy); "
          "indecomposable p gives {p}; checked by composing back")
def _decompose(e, x):
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
          "functions: HornerForm of numerator and denominator")
def _horner(e, v=None):
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
          "pure function; transcendental input raises")
def _minpoly(s, x=None, *rest):
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
          "constants give False; Extension, GaussianIntegers, Modulus options")
def _irredq(e, *rest):
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
          "k = 0)")
def _sympoly(k, vs):
    kk = nonneg_int(k, "SymmetricPolynomial", "degree")
    xs = _vars_opt(vs, "SymmetricPolynomial")
    if kk > len(xs):
        return 0
    from itertools import combinations
    return out(sp.Add(*(sp.Mul(*c) for c in combinations(xs, kk))))


@fn("SymmetricReduction", area=AREA, status="added",
    notes="SymmetricReduction[f, {x1, ..., xn}] -> {p, q}: p the symmetric part written in the elementary "
          "symmetric polynomials of the xi, q the remainder, f = p + q; SymmetricReduction[f, xs, {s1, ..., sn}] "
          "writes p in the si; checked by expansion")
def _symred(e, vs, names=None):
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
          "matched to the root numerically; other roots stay Root[...]")
def _toradicals(e):
    v = sym(e, "ToRadicals")
    return out(v.replace(lambda u: isinstance(u, sp.CRootOf), _root_radicals))
