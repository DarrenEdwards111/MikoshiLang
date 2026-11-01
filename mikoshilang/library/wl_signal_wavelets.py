"""Wolfram batch W23 (signal processing), wavelet part.

Wavelet family objects (HaarWavelet, DaubechiesWavelet, ShannonWavelet, MorletWavelet, MexicanHatWavelet,
GaborWavelet, PaulWavelet, DGaussianWavelet), their filter coefficients (WaveletFilterCoefficients) and closed-form
scaling/wavelet functions (WaveletPhi, WaveletPsi).  Wavelet transforms (DiscreteWaveletTransform, ...) are not part
of this module.  See CONVENTIONS.md: results are correct, or raise, or stay unevaluated for symbolic input.
"""
from __future__ import annotations

import math

import mpmath
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, R, S, is_list
from .wl_common import L
from .wl_signal import _opts, _wp_opt, _sym_var

AREA = "signal processing (W23)"

# name -> (default parameter or None for no parameter, kind of parameter)
_FAMILIES = {
    "HaarWavelet": (None, None),
    "MorletWavelet": (None, None),
    "DaubechiesWavelet": (2, "posint"),
    "ShannonWavelet": (10, "posint"),
    "MexicanHatWavelet": (1, "posreal"),
    "GaborWavelet": (6, "real"),
    "PaulWavelet": (4, "posint"),
    "DGaussianWavelet": (2, "posint"),
}


def _family_param(name, args):
    """Validated parameter (SymPy number or None) of a wavelet family object; symbolic -> Unevaluated."""
    default, kind = _FAMILIES[name]
    if kind is None:
        if args:
            raise TypeError(f"{name}: takes no arguments")
        return None
    if len(args) > 1:
        raise TypeError(f"{name}: takes at most one argument")
    if not args:
        return sp.Integer(default)
    v = args[0]
    if isinstance(v, (str, bool)) or is_list(v):
        raise TypeError(f"{name}: the parameter must be a number, not {v!r}")
    x = sp.sympify(S(v)) if not isinstance(v, (int, float)) else sp.sympify(v)
    if x.free_symbols:
        raise Unevaluated()
    if kind == "posint":
        if not (x.is_Integer and x > 0):
            raise ValueError(f"{name}: the parameter must be a positive integer, not {v!r}")
    elif kind == "posreal":
        if not (x.is_real and x > 0):
            raise ValueError(f"{name}: the parameter must be a positive real number, not {v!r}")
    elif not x.is_real:
        raise ValueError(f"{name}: the parameter must be a real number, not {v!r}")
    return x


def _family_ctor(name):
    def impl(*args: Expr) -> Expr:
        p = _family_param(name, args)
        return Expr(name) if p is None else Expr(name, R(p))
    return impl


_FAMILY_NOTES = {
    "HaarWavelet": "HaarWavelet[]: the Haar wavelet object (φ = 1 on [0, 1), ψ = 1 on [0, 1/2) and -1 on [1/2, 1))",
    "MorletWavelet": "MorletWavelet[]: the Morlet wavelet object, ψ = π^(-1/4) e^(-x²/2) cos(π √(2/log 2) x), "
                     "no scaling function",
    "DaubechiesWavelet": "DaubechiesWavelet[n] (default 2): the Daubechies wavelet object with n vanishing moments "
                         "(filter coefficients via WaveletFilterCoefficients)",
    "ShannonWavelet": "ShannonWavelet[n] (default 10): the Shannon wavelet object, φ = Sinc[π x], filters "
                      "truncated to |k| ≤ n",
    "MexicanHatWavelet": "MexicanHatWavelet[σ] (default 1): ψ = 2/(√(3σ) π^(1/4)) (1 - x²/σ²) e^(-x²/(2σ²))",
    "GaborWavelet": "GaborWavelet[w] (default 6): ψ = π^(-1/4) e^(i w x - x²/2)",
    "PaulWavelet": "PaulWavelet[n] (default 4): ψ = 2^n i^n n!/√(π (2n)!) (1 - i x)^(-(n+1))",
    "DGaussianWavelet": "DGaussianWavelet[n] (default 2): ψ = (-1)^(n+1) D^n e^(-x²/2)/√Γ(n + 1/2)",
}

for _n in _FAMILIES:
    fn(_n, area=AREA, status="added",
       notes=_FAMILY_NOTES[_n] + "; the default parameter is filled in, invalid parameters raise ValueError and "
             "symbolic ones stay unevaluated. Check: exact (validated constructor); reference (Wolfram "
             "documentation) in tests.")(_family_ctor(_n))


def _wavelet(w, fname):
    """(family name, parameter) of a wavelet object argument."""
    if isinstance(w, Expr) and w.head in _FAMILIES:
        return w.head, _family_param(w.head, w.args)
    if isinstance(w, Symbol):
        raise Unevaluated()
    if isinstance(w, Expr) and w.head.endswith("Wavelet"):
        raise NotImplementedError(f"{fname}: {w.head} is not supported")
    raise TypeError(f"{fname}: {w!r} is not a wavelet object")


# ───────────────────────────────────────────────────────────── filter coefficients ──
def _daubechies_exact(n):
    y = sp.Symbol("y")
    P = sum(sp.binomial(n - 1 + k, k) * y ** k for k in range(n))
    w = sp.Symbol("w")
    H = (1 + w) ** n
    for yr in sp.roots(sp.Poly(P, y), multiple=True):
        zs = sp.solve(sp.Symbol("z") ** 2 - (2 - 4 * yr) * sp.Symbol("z") + 1, sp.Symbol("z"))
        z = min(zs, key=lambda t: abs(complex(sp.N(t))))
        H = H * (1 - z * w)
    coeffs = sp.Poly(sp.expand(H), w).all_coeffs()[::-1]
    tot = sum(coeffs)
    return [sp.radsimp(sp.simplify(c / tot)) for c in coeffs]


def _daubechies_mp(n, dps):
    with mpmath.workdps(dps + 4 * n + 20):
        P = [mpmath.binomial(n - 1 + k, k) for k in range(n)]
        roots = mpmath.polyroots(P[::-1], maxsteps=400, extraprec=8 * n + 50) if n > 1 else []
        coeffs = [mpmath.mpc(1)]
        for _ in range(n):
            coeffs = [a + b for a, b in zip(coeffs + [0], [0] + coeffs)]
        for yr in roots:
            b = 2 - 4 * yr
            disc = mpmath.sqrt(b * b - 4)
            z1, z2 = (b + disc) / 2, (b - disc) / 2
            z = z1 if abs(z1) < abs(z2) else z2
            coeffs = [a - z * c for a, c in zip(coeffs + [0], [0] + coeffs)]
        tot = sum(coeffs)
        return [mpmath.re(c / tot) for c in coeffs]


def _lowpass(name, p, exact, dps):
    """{k: h_k} primal lowpass filter (Σ h_k = 1)."""
    if name == "HaarWavelet":
        return {0: sp.Rational(1, 2), 1: sp.Rational(1, 2)} if exact else {0: mpmath.mpf(1) / 2,
                                                                           1: mpmath.mpf(1) / 2}
    if name == "DaubechiesWavelet":
        n = int(p)
        if exact:
            if n > 2:
                raise NotImplementedError("WaveletFilterCoefficients: exact (WorkingPrecision -> Infinity) "
                                          "Daubechies filters are supported for n <= 2")
            return dict(enumerate(_daubechies_exact(n)))
        if n > 40:
            raise NotImplementedError("WaveletFilterCoefficients: DaubechiesWavelet[n] is supported for n <= 40")
        return dict(enumerate(_daubechies_mp(n, dps)))
    if name == "ShannonWavelet":
        n = int(p)
        if n > 100000:
            raise ValueError("WaveletFilterCoefficients: the Shannon filter length is too large")
        out = {}
        for k in range(-n, n + 1):
            if exact:
                out[k] = sp.Rational(1, 2) if k == 0 else sp.sin(sp.pi * k / 2) / (sp.pi * k)
            else:
                out[k] = mpmath.mpf(1) / 2 if k == 0 else (mpmath.sinpi(mpmath.mpf(k) / 2) / (mpmath.pi * k)
                                                           if k % 2 else mpmath.mpf(0))
        return out
    raise ValueError(f"WaveletFilterCoefficients: {name} is a continuous wavelet without filter coefficients")


def _highpass(name, h):
    if name == "ShannonWavelet":
        return {k: (1 if (k - 1) % 2 == 0 else -1) * h[1 - k] for k in sorted(1 - j for j in h)}
    return {k: (1 if k % 2 == 0 else -1) * h[1 - k] for k in sorted(1 - j for j in h)}


def _check_orthonormal(h, exact):
    """Σ h = 1 and Σ h_k h_(k+2m) = δ_m / 2 (orthonormal two-scale relation, Σh = 1 normalisation)."""
    ks = sorted(h)
    tol = 0 if exact else mpmath.mpf(10) ** (-(mpmath.mp.dps - 8))

    def bad(v):
        return (sp.simplify(v) != 0) if exact else abs(v) > tol
    if bad(sum(h.values()) - 1):
        raise ArithmeticError("WaveletFilterCoefficients: result failed verification")
    for m in range(0, (ks[-1] - ks[0]) // 2 + 1):
        s = sum(h[k] * h[k + 2 * m] for k in ks if k + 2 * m in h)
        if bad(s - (sp.Rational(1, 2) if exact else mpmath.mpf(1) / 2) * (1 if m == 0 else 0)):
            raise ArithmeticError("WaveletFilterCoefficients: result failed verification")


_FILTERS = ("PrimalLowpass", "PrimalHighpass")


def _wfc(w: Expr, *rest: Expr) -> ListOf[ListOf[Real]]:
    fname = "WaveletFilterCoefficients"
    pos, opts = _opts(rest, {"WorkingPrecision"}, fname)
    name, p = _wavelet(w, fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: called with {len(pos) + 1} positional arguments; 1 or 2 are expected")
    filt = pos[0] if pos else "PrimalLowpass"
    if filt == "Properties":
        return L(*_FILTERS)
    many = is_list(filt)
    filts = list(filt.args) if many else [filt]
    for f in filts:
        if not isinstance(f, str):
            raise TypeError(f"{fname}: the filter must be a string such as \"PrimalLowpass\", not {f!r}")
        if f not in _FILTERS:
            raise NotImplementedError(f"{fname}: the filter {f!r} is not supported (\"PrimalLowpass\" or "
                                      "\"PrimalHighpass\")")
    wp = _wp_opt(opts["WorkingPrecision"], fname) if "WorkingPrecision" in opts else "m"
    exact = wp == "exact"
    dps = 20 if wp in (None, "m", "exact") else int(mpmath.libmp.prec_to_dps(wp)) + 5
    with mpmath.workdps(dps + 10):
        h = _lowpass(name, p, exact, dps + 10)
        if name != "ShannonWavelet":
            _check_orthonormal(h, exact)

        def conv(v):
            if exact:
                return R(sp.simplify(v))
            if wp in (None, "m"):
                return float(v)
            return sp.Float(v, int(mpmath.libmp.prec_to_dps(wp)))
        outs = []
        for f in filts:
            c = h if f == "PrimalLowpass" else _highpass(name, h)
            outs.append(L(*[L(k, conv(c[k])) for k in sorted(c)]))
    return L(*outs) if many else outs[0]


fn("WaveletFilterCoefficients", area=AREA, status="added",
   notes="WaveletFilterCoefficients[wave, filt]: {k, h_k} pairs of the \"PrimalLowpass\" (default, normalised "
         "Σ h_k = 1) or \"PrimalHighpass\" filter (g_k = (-1)^k h_(1-k); Shannon (-1)^(k-1) h_(1-k)) for "
         "HaarWavelet, DaubechiesWavelet[n] (minimum-phase spectral factorisation, n <= 40) and ShannonWavelet[n] "
         "(sinc filter on |k| <= n); a list of filters, \"Properties\"; WorkingPrecision (machine default, "
         "digits, Infinity exact for Haar, Shannon and Daubechies n <= 2); lifting and dual filters and other "
         "families raise NotImplementedError, continuous wavelets raise ValueError. Check: self-check "
         "(orthonormality Σ h_k h_(k+2m) = δ_m/2 and Σ h_k = 1 verified, ArithmeticError on failure); reference "
         "(Wolfram documentation examples) in tests.")(_wfc)


# ───────────────────────────────────────────────────────────── φ and ψ ──
def _psi_expr(name, p, x):
    if name == "HaarWavelet":
        half = sp.Rational(1, 2)
        return sp.Piecewise((1, (x >= 0) & (x < half)), (-1, (x >= half) & (x < 1)), (0, True))
    if name == "ShannonWavelet":
        return 2 * (sp.sin(2 * sp.pi * x) - sp.cos(sp.pi * x)) / (sp.pi - 2 * sp.pi * x)
    if name == "MorletWavelet":
        return sp.exp(-x ** 2 / 2) * sp.cos(sp.pi * sp.sqrt(2 / sp.log(2)) * x) / sp.pi ** sp.Rational(1, 4)
    if name == "MexicanHatWavelet":
        s = p
        return (2 / (sp.sqrt(3 * s) * sp.pi ** sp.Rational(1, 4)) * (1 - x ** 2 / s ** 2)
                * sp.exp(-x ** 2 / (2 * s ** 2)))
    if name == "GaborWavelet":
        return sp.exp(sp.I * p * x - x ** 2 / 2) / sp.pi ** sp.Rational(1, 4)
    if name == "PaulWavelet":
        n = int(p)
        return (2 ** n * sp.I ** n * sp.factorial(n) / sp.sqrt(sp.pi * sp.factorial(2 * n))
                * (1 - sp.I * x) ** (-(n + 1)))
    if name == "DGaussianWavelet":
        n = int(p)
        t = sp.Dummy("t")
        d = sp.diff(sp.exp(-t ** 2 / 2), t, n).subs(t, x)
        return sp.simplify((-1) ** (n + 1) * d / sp.sqrt(sp.gamma(n + sp.Rational(1, 2))))
    raise NotImplementedError(f"WaveletPsi: {name} has no closed form here (the numerical cascade "
                              "InterpolatingFunction is not supported)")


def _phi_expr(name, p, x):
    if name == "HaarWavelet":
        return sp.Piecewise((1, (x >= 0) & (x < 1)), (0, True))
    if name == "ShannonWavelet":
        return sp.sinc(sp.pi * x)
    if name == "DaubechiesWavelet":
        raise NotImplementedError("WaveletPhi: Daubechies scaling functions have no closed form here (the "
                                  "numerical cascade InterpolatingFunction is not supported)")
    raise ValueError(f"WaveletPhi: {name} has no scaling function")


def _r(v):
    """SymPy -> MikoshiLang, with sinc written as Wolfram's Sinc."""
    if v.has(sp.sinc):
        holders = {}

        def rep(*a):
            d = sp.Symbol(f"sincholder{len(holders)}")
            holders[d.name] = a[0]
            return d
        v2 = v.replace(sp.sinc, rep)
        out = R(v2)
        return _replace_syms(out, {k: Expr("Sinc", R(a)) for k, a in holders.items()})
    return R(v)


def _replace_syms(e, table):
    if isinstance(e, Symbol) and e.name in table:
        return table[e.name]
    if isinstance(e, Expr):
        return Expr(e.head, *[_replace_syms(a, table) for a in e.args])
    return e


def _to_slot(e, name):
    return _replace_syms(e, {name: Expr("Slot", 1)})


def _phipsi(fname, builder):
    def impl(w: Expr, *rest: Expr) -> Expr:
        pos, opts = _opts(rest, {"WorkingPrecision"}, fname)
        name, p = _wavelet(w, fname)
        if len(pos) > 1:
            raise TypeError(f"{fname}: called with {len(pos) + 2} arguments; 1 or 2 are expected")
        wp = _wp_opt(opts["WorkingPrecision"], fname) if "WorkingPrecision" in opts else None
        if not pos:
            slot = sp.Symbol("wslot", real=True)
            return Expr("Function", _to_slot(_r(builder(name, p, slot)), "wslot"))
        x = pos[0]
        if isinstance(x, Symbol):
            xr = sp.Symbol(x.name, real=True)
            return _r(builder(name, p, xr).subs(xr, sp.Symbol(x.name)))
        if isinstance(x, (str, bool)) or is_list(x):
            raise TypeError(f"{fname}: the argument must be a number or a symbol, not {x!r}")
        xv = sp.sympify(S(x)) if not isinstance(x, (int, float)) else sp.sympify(x)
        if xv.free_symbols:
            return _r(builder(name, p, xv))
        if not xv.is_real:
            raise ValueError(f"{fname}: the argument must be real")
        v = builder(name, p, xv)
        if wp == "exact":
            return R(sp.simplify(v))
        if wp not in (None, "m"):
            return R(sp.N(v, int(mpmath.libmp.prec_to_dps(wp))))
        if any(isinstance(a, sp.Float) for a in sp.preorder_traversal(xv)) or wp == "m":
            return R(sp.N(v))
        return R(sp.simplify(v))
    return impl


fn("WaveletPsi", area=AREA, status="added",
   notes="WaveletPsi[wave, x] / WaveletPsi[wave] (pure function): closed-form wavelet functions of HaarWavelet "
         "(Piecewise), ShannonWavelet, MorletWavelet, MexicanHatWavelet[σ], GaborWavelet[w], PaulWavelet[n] and "
         "DGaussianWavelet[n]; numeric x evaluates (WorkingPrecision honoured); Daubechies (numerical cascade) "
         "raises NotImplementedError. Check: exact (closed forms); reference (Wolfram documentation formulas) "
         "and the zero-mean property ∫ψ = 0 in tests.")(_phipsi("WaveletPsi", _psi_expr))
fn("WaveletPhi", area=AREA, status="added",
   notes="WaveletPhi[wave, x] / WaveletPhi[wave]: scaling functions of HaarWavelet (1 on [0, 1)) and "
         "ShannonWavelet (Sinc[π x]); continuous wavelets without a scaling function raise ValueError, Daubechies "
         "(numerical cascade) raises NotImplementedError. Check: exact (closed forms); reference (Wolfram "
         "documentation) and ∫φ = 1 in tests.")(_phipsi("WaveletPhi", _phi_expr))


# ──────────────────────────────────────────── further wavelet families (objects only) ──
def _posint_arg(v, name, what):
    if isinstance(v, (str, bool)) or is_list(v):
        raise TypeError(f"{name}: the {what} must be a positive integer, not {v!r}")
    x = sp.sympify(S(v)) if not isinstance(v, (int, float)) else sp.sympify(v)
    if x.free_symbols:
        raise Unevaluated()
    if not (x.is_Integer and x > 0):
        raise ValueError(f"{name}: the {what} must be a positive integer, not {v!r}")
    return x


def _real_arg(v, name, what):
    if isinstance(v, (str, bool)) or is_list(v):
        raise TypeError(f"{name}: the {what} must be a positive real number, not {v!r}")
    x = sp.sympify(S(v)) if not isinstance(v, (int, float)) else sp.sympify(v)
    if x.free_symbols:
        raise Unevaluated()
    if not (x.is_real and x > 0):
        raise ValueError(f"{name}: the {what} must be a positive real number, not {v!r}")
    return x


def _order_ctor(name, default, lo=1, hi=None):
    """Family[] -> Family[default]; Family[n] for integers lo <= n (<= hi)."""
    def impl(*args: PosInteger) -> Expr:
        if len(args) > 1:
            raise TypeError(f"{name}: takes at most one argument")
        if not args:
            return Expr(name, default)
        n = _posint_arg(args[0], name, "order")
        if n < lo or (hi is not None and n > hi):
            raise ValueError(f"{name}: the order must be an integer between {lo} and {hi}, not {args[0]!r}")
        return Expr(name, R(n))
    return impl


def _spline_ctor(name):
    """BiorthogonalSplineWavelet[] -> [4, 2]; [n, m] with positive integers and n + m even."""
    def impl(*args: PosInteger) -> Expr:
        if not args:
            return Expr(name, 4, 2)
        if len(args) != 2:
            raise TypeError(f"{name}: takes no arguments or two orders n, m")
        n, m = (_posint_arg(a, name, "order") for a in args)
        if (n + m) % 2:
            raise ValueError(f"{name}: the orders n and m must have an even sum, not {n} + {m}")
        return Expr(name, R(n), R(m))
    return impl


def _lim_ctor(name):
    """BattleLemarieWavelet / MeyerWavelet: [] -> [3]; [n]; [n, lim] with lim > 0."""
    def impl(*args: PosInteger | Real) -> Expr:
        if not args:
            return Expr(name, 3)
        if len(args) > 2:
            raise TypeError(f"{name}: takes at most two arguments")
        n = _posint_arg(args[0], name, "order")
        if len(args) == 1:
            return Expr(name, R(n))
        lim = _real_arg(args[1], name, "limit")
        return Expr(name, R(n), args[1] if isinstance(args[1], float) else R(lim))
    return impl


def _cdf_ctor(*args: String) -> Expr:
    name = "CDFWavelet"
    if not args:
        return Expr(name, "9/7")
    if len(args) > 1:
        raise TypeError(f"{name}: takes at most one argument")
    t = args[0]
    if isinstance(t, Symbol):
        raise Unevaluated()
    if not isinstance(t, str):
        raise TypeError(f"{name}: the type must be the string \"5/3\" or \"9/7\", not {t!r}")
    if t not in ("5/3", "9/7"):
        raise ValueError(f"{name}: the type must be \"5/3\" or \"9/7\", not {t!r}")
    return Expr(name, t)


_OBJ_NOTE = ("; invalid parameters raise ValueError (wrong kind TypeError), symbolic ones stay unevaluated. Filter "
             "coefficients, φ/ψ and the wavelet transforms of this family are not available yet: "
             "WaveletFilterCoefficients, WaveletPhi and WaveletPsi raise NotImplementedError for it. "
             "Check: exact (validated constructor); reference (Wolfram documentation defaults) in tests.")

fn("SymletWavelet", area=AREA, status="added",
   notes="SymletWavelet[] = SymletWavelet[4], SymletWavelet[n] for positive integers n (least-asymmetric orthogonal "
         "wavelet with n vanishing moments)" + _OBJ_NOTE)(_order_ctor("SymletWavelet", 4))
fn("CoifletWavelet", area=AREA, status="added",
   notes="CoifletWavelet[] = CoifletWavelet[2], CoifletWavelet[n] for integers 1 <= n <= 5 (Wolfram's documented "
         "range)" + _OBJ_NOTE)(_order_ctor("CoifletWavelet", 2, 1, 5))
fn("BiorthogonalSplineWavelet", area=AREA, status="added",
   notes="BiorthogonalSplineWavelet[] = BiorthogonalSplineWavelet[4, 2], BiorthogonalSplineWavelet[n, m] for "
         "positive integers with n + m even" + _OBJ_NOTE)(_spline_ctor("BiorthogonalSplineWavelet"))
fn("ReverseBiorthogonalSplineWavelet", area=AREA, status="added",
   notes="ReverseBiorthogonalSplineWavelet[] = ReverseBiorthogonalSplineWavelet[4, 2], [n, m] for positive "
         "integers with n + m even" + _OBJ_NOTE)(_spline_ctor("ReverseBiorthogonalSplineWavelet"))
fn("BattleLemarieWavelet", area=AREA, status="added",
   notes="BattleLemarieWavelet[] = BattleLemarieWavelet[3], BattleLemarieWavelet[n] (n a positive integer; "
         "equivalent to [n, 10]) and BattleLemarieWavelet[n, lim] with lim > 0" + _OBJ_NOTE)(
    _lim_ctor("BattleLemarieWavelet"))
fn("MeyerWavelet", area=AREA, status="added",
   notes="MeyerWavelet[] = MeyerWavelet[3], MeyerWavelet[n] (n a positive integer) and MeyerWavelet[n, lim] with "
         "lim > 0" + _OBJ_NOTE)(_lim_ctor("MeyerWavelet"))
fn("CDFWavelet", area=AREA, status="added",
   notes="CDFWavelet[] = CDFWavelet[\"9/7\"], CDFWavelet[\"5/3\"] (Cohen-Daubechies-Feauveau biorthogonal "
         "wavelets of JPEG2000)" + _OBJ_NOTE)(_cdf_ctor)
