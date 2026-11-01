"""Differential tests for Wolfram batch X03 (Signal II: wavelets, spectral estimation, inverse transforms and control
plots; modules wl_signal2 and wl_signal2_control).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): documented examples and identities,
  e.g. ``Map[InverseFourier[#, FourierParameters -> {1, -1}] &, SpectrogramArray[list, 3, 3]] == Partition[list,
  3]``, ``KalmanFilter[ARProcess[{1/3}, 1], {v, x, y, z}] == {0, v/3, x/3, y/3}``, ``PowerSpectralDensity[
  ARProcess[{a}, σ^2], ω] == σ^2/(1 + a^2 - 2 a Cos[ω])``, ``InverseHilbertTransform[Cos[t], t, s] == -Sin[s]``,
  ``InverseHilbertTransform[1/(π t), t, s] == DiracDelta[s]``, ``InverseHilbertTransform[1/(1 + t^2), t, s] ==
  -s/(1 + s^2)``, the Haar DWT of {1, 2, 3, 4} ({0} -> {3/√2, 7/√2}, {1} -> {-1/√2, -1/√2}) and the default
  partition sizes 2^Round[Log2[Sqrt[m]] + 1], Round[n/3].
* Independent references computed here with NumPy / mpmath / SymPy from the textbook definitions: Mallat's
  periodized Haar/Daubechies filter bank (D4 taps (1 ± √3)/(4√2), (3 ± √3)/(4√2)), the discrete STFT as
  numpy.fft.fft of windowed partitions, the biased sample autocovariance (Brockwell & Davis §1.4), the Durbin–Levinson
  recursion, the local Wiener filter of Lim (1990) / MATLAB wiener2, SNIP clipping (Ryan et al. 1988), Griffin–Lim
  phase retrieval (1984), canonical time warping (Zhou & De la Torre 2009), principal-value quadrature with mpmath
  (``mpmath.quad`` with the singularity subtracted), bilateral Laplace integrals by ``mpmath.quad`` and bilateral
  Z series summed directly, and C (sI - A)^-1 B + D by hand.

Symbols assigned in cases are prefixed ``x03``.
"""
import math

import mpmath
import numpy as np
import pytest

from _harness import Pred, Raises, check, ev, ids
from mikoshilang.expr import Expr


# ───────────────────────────────────────────────────────────────── helpers
def _np(x):
    """MikoshiLang (nested) list -> numpy array."""
    def rec(z):
        if isinstance(z, Expr) and z.head == "List":
            return [rec(a) for a in z.args]
        from _harness import to_complex
        return to_complex(z)
    return np.array(rec(x))


def near(ref, tol=1e-9):
    ref = np.asarray(ref, dtype=complex)

    def f(got):
        g = _np(got)
        return g.shape == ref.shape and np.allclose(g, ref, atol=tol * max(1.0, float(np.max(np.abs(ref))) if ref.size
                                                                         else 1.0), rtol=0)
    return Pred(f, f"≈ {np.round(ref, 6).tolist()}")


def head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"head {h}")


def less(v):
    return Pred(lambda g: float(_np(g).real) < v, f"< {v}")


def is_true():
    return Pred(lambda g: g is True, "True")


S2 = math.sqrt(2)
X03_SIG = [math.sin(k / 3) + 0.3 * math.cos(k) for k in range(64)]
X03_SIG_S = "{" + ",".join(repr(v) for v in X03_SIG) + "}"
X03_SIG2 = [math.sin(k / 2) + math.sin(k / 5) for k in range(64)]
X03_SIG2_S = "{" + ",".join(repr(v) for v in X03_SIG2) + "}"


def _haar_level(x):
    x = np.asarray(x, float)
    return (x[0::2] + x[1::2]) / S2, (x[0::2] - x[1::2]) / S2


def _d4():
    r3 = math.sqrt(3)
    h = np.array([1 + r3, 3 + r3, 3 - r3, 1 - r3]) / (4 * S2)
    return h


def _d4_level(x):
    """Periodized D4 analysis (Mallat): a[n] = Σ h[k] x[2n + k], d[n] = Σ g[k] x[2n + k], g[k] = (-1)^k h[3 - k]."""
    h = _d4()
    g = np.array([h[3], -h[2], h[1], -h[0]])
    x = np.asarray(x, float)
    L = len(x)
    a = np.array([sum(h[k] * x[(2 * n + k) % L] for k in range(4)) for n in range(L // 2)])
    d = np.array([sum(g[k] * x[(2 * n + k) % L] for k in range(4)) for n in range(L // 2)])
    return a, d


def _stft_ref(x, n, d, w=None):
    x = np.asarray(x, float)
    m = len(x)
    k = 1 if m <= n else math.ceil((m - n) / d) + 1
    total = (k - 1) * d + n
    xp = np.concatenate([x, np.full(total - m, x[-1])])
    w = np.ones(n) if w is None else w
    return np.array([np.fft.fft(xp[i * d:i * d + n] * w) for i in range(k)])


def _hann(n):
    t = np.linspace(-0.5, 0.5, n)
    return (1 + np.cos(2 * np.pi * t)) / 2


def _acov(x, h):
    x = np.asarray(x, float)
    mu = x.mean()
    return float(np.sum((x[:len(x) - h] - mu) * (x[h:] - mu)) / len(x))


def _wiener2(x, r, ns=None):
    x = np.asarray(x, float)
    xp = np.pad(x, r, mode="edge")
    n = len(x)
    mu = np.array([xp[i:i + 2 * r + 1].mean() for i in range(n)])
    var = np.array([(xp[i:i + 2 * r + 1] ** 2).mean() for i in range(n)]) - mu ** 2
    ns = var.mean() if ns is None else ns
    return mu + np.maximum(var - ns, 0) / np.maximum(var, ns) * (x - mu)


def _snip(y, sig):
    b = np.asarray(y, float).copy()
    n = len(b)
    for p in range(1, sig + 1):
        bp = np.pad(b, p, mode="symmetric")
        b = np.minimum(b, (bp[:n] + bp[2 * p:2 * p + n]) / 2)
    return b


def _pv_iht(f, s, brk=()):
    """-(1/π) PV ∫ f(t)/(s - t) dt by mpmath (singularity subtracted; brk = kinks of f)."""
    fs = f(s)
    g = lambda t: (f(t) - fs) / (s - t)
    pts = sorted({s - 1, s, s + 1, *brk})
    v = mpmath.quad(g, [-mpmath.inf] + pts + [mpmath.inf])
    return float(-v / mpmath.pi)


X03_IHT_RAT = _pv_iht(lambda t: 1 / (1 + t * t), 0.7)            # = -0.7/(1 + 0.49)
X03_IHT_BOX = float(-(1 / math.pi) * math.log(abs((0.3 + 0.5) / (0.3 - 0.5))))


def _ctw_ref_ok(got):
    v = float(_np(got).real)
    return 0 <= v < 1e-8


# ───────────────────────────────────────────────────────────────── discrete wavelets
DWT = [
    ("DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][{0}, \"Values\"]",
     near([_haar_level([1, 2, 3, 4])[0]])),
    ("DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][{1}, \"Values\"]",
     near([_haar_level([1, 2, 3, 4])[1]])),
    ("DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 2, WorkingPrecision -> Infinity][{0, 0}, \"Values\"]",
     "{{5}}"),
    ("DiscreteWaveletTransform[{x03a, x03b}, HaarWavelet[], 1, WorkingPrecision -> Infinity][{1}, \"Values\"]",
     "{{(x03a - x03b)/Sqrt[2]}}"),
    ("DiscreteWaveletTransform[Range[8.], DaubechiesWavelet[2], 1][{1}, \"Values\"]",
     near([_d4_level(range(1, 9))[1]])),
    ("DiscreteWaveletTransform[Range[8.], DaubechiesWavelet[2], 1][{0}, \"Values\"]",
     near([_d4_level(range(1, 9))[0]])),
    ("DiscreteWaveletTransform[Range[16.]][\"Refinement\"]", "4"),
    ("DiscreteWaveletTransform[Table[i + 2 j, {i, 4}, {j, 4}], HaarWavelet[], 1][\"WaveletIndex\"]",
     "{{0}, {1}, {2}, {3}}"),
    ("DiscreteWaveletTransform[x03u]", "DiscreteWaveletTransform[x03u]"),
    ("DiscreteWaveletTransform[\"abc\"]", Raises(TypeError)),
    ("DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 0]", Raises(ValueError)),
    ("DiscreteWaveletTransform[{1, 2, 3, 4}, CDFWavelet[]]", Raises(NotImplementedError)),
    # Check: self-check (perfect reconstruction) and Parseval for an orthonormal filter bank
    ("With[{x03x = Range[16.]}, Max[Abs[InverseWaveletTransform[DiscreteWaveletTransform[x03x, DaubechiesWavelet[3], "
     "3]] - x03x]]]", less(1e-9)),
    ("With[{x03w = DiscreteWaveletTransform[Range[16.], DaubechiesWavelet[2], 2]}, Total[Map[Total[#^2] &, "
     "x03w[Automatic, \"Values\"]]]]", float(sum(k * k for k in range(1, 17))), 1e-9),
]

PACKET = [
    ("DiscreteWaveletPacketTransform[{1, 2, 3, 4}, HaarWavelet[], 2, WorkingPrecision -> Infinity][\"WaveletIndex\"]",
     "{{0}, {1}, {0, 0}, {0, 1}, {1, 0}, {1, 1}}"),
    ("DiscreteWaveletPacketTransform[{1., 2, 3, 4}, HaarWavelet[], 2][{1, 0}, \"Values\"]",
     near([[-1.0]])),
    ("DiscreteWaveletPacketTransform[Range[32.]][\"Refinement\"]", "4"),
    ("DiscreteWaveletPacketTransform[x03u, HaarWavelet[]]", "DiscreteWaveletPacketTransform[x03u, HaarWavelet[]]"),
    ("DiscreteWaveletPacketTransform[{1, 2}, HaarWavelet[], -1]", Raises(ValueError)),
    ("With[{x03x = Range[8.]}, Max[Abs[InverseWaveletTransform[DiscreteWaveletPacketTransform[x03x, HaarWavelet[], "
     "3], Automatic, {{1}, {0, 1}, {0, 0, 0}, {0, 0, 1}}] - x03x]]]", less(1e-12)),
]

STATIONARY = [
    ("StationaryWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1, WorkingPrecision -> Infinity][{1}, \"Values\"]",
     "{{-1/Sqrt[2], -1/Sqrt[2], -1/Sqrt[2], 3/Sqrt[2]}}"),
    ("StationaryWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][{0}, \"Values\"]",
     near([[3 / S2, 5 / S2, 7 / S2, 5 / S2]])),
    ("StationaryWaveletTransform[Range[8.], HaarWavelet[], 2][\"Dimensions\"]",
     "{{0} -> {8}, {1} -> {8}, {0, 0} -> {8}, {0, 1} -> {8}}"),
    ("StationaryWaveletTransform[x03u]", "StationaryWaveletTransform[x03u]"),
    ("StationaryWaveletTransform[{{1, 2}, {3}}]", Raises(ValueError)),
    ("With[{x03x = N[Range[16]^2]}, Max[Abs[InverseWaveletTransform[StationaryWaveletTransform[x03x, "
     "DaubechiesWavelet[2], 3]] - x03x]]]", less(1e-9)),
    ("StationaryWaveletPacketTransform[{1, 2, 3, 4}, HaarWavelet[], 2][\"Dimensions\"]",
     "{{0} -> {4}, {1} -> {4}, {0, 0} -> {4}, {0, 1} -> {4}, {1, 0} -> {4}, {1, 1} -> {4}}"),
    ("StationaryWaveletPacketTransform[{1, 2, 3, 4}, HaarWavelet[], 1, WorkingPrecision -> Infinity][{1}, "
     "\"Values\"]", "{{-1/Sqrt[2], -1/Sqrt[2], -1/Sqrt[2], 3/Sqrt[2]}}"),
    ("StationaryWaveletPacketTransform[x03u]", "StationaryWaveletPacketTransform[x03u]"),
    ("StationaryWaveletPacketTransform[{1, 2, 3, 4}, MexicanHatWavelet[]]", Raises(NotImplementedError)),
    ("With[{x03x = Range[8.]}, Max[Abs[InverseWaveletTransform[StationaryWaveletPacketTransform[x03x, HaarWavelet[], "
     "2]] - x03x]]]", less(1e-12)),
]

LIFTING = [
    ("LiftingWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1, WorkingPrecision -> Infinity][All, \"Values\"]",
     "{{3/Sqrt[2], 7/Sqrt[2]}, {-1/Sqrt[2], -1/Sqrt[2]}}"),
    ("LiftingWaveletTransform[Range[8.], HaarWavelet[], 2][{0, 0}, \"Values\"]",
     near([[10 / 2, 26 / 2]])),
    ("LiftingWaveletTransform[{1, 2, 3, 4, 5, 6, 7, 8}, CDFWavelet[], 1, WorkingPrecision -> Infinity]",
     Raises(NotImplementedError)),
    ("LiftingWaveletTransform[x03u]", "LiftingWaveletTransform[x03u]"),
    ("LiftingWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1.5]", Raises(TypeError)),
    ("With[{x03x = N[Range[16]^2]}, Max[Abs[InverseWaveletTransform[LiftingWaveletTransform[x03x, CDFWavelet[], "
     "3]] - x03x]]]", less(1e-9)),
]

DWDATA = [
    ("DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][\"BasisIndex\"]", "{{0}, {1}}"),
    ("DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][\"Wavelet\"]", "HaarWavelet[]"),
    ("DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][\"Transform\"]", "\"DiscreteWaveletTransform\""),
    ("DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1, WorkingPrecision -> Infinity][\"EnergyFraction\"]",
     "{{0} -> 29/30, {1} -> 1/30}"),
    ("Normal[DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 1, WorkingPrecision -> Infinity]]",
     "{{0} -> {3/Sqrt[2], 7/Sqrt[2]}, {1} -> {-1/Sqrt[2], -1/Sqrt[2]}}"),
    ("InverseWaveletTransform[DiscreteWaveletData[{{0} -> {3/Sqrt[2]}, {1} -> {-1/Sqrt[2]}}, HaarWavelet[], "
     "\"DiscreteWaveletTransform\"]]", near([1, 2])),
    ("DiscreteWaveletData[x03r, HaarWavelet[], \"DiscreteWaveletTransform\"]",
     "DiscreteWaveletData[x03r, HaarWavelet[], \"DiscreteWaveletTransform\"]"),
    ("DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][\"Nonsense\"]", Raises(ValueError)),
    ("DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1][\"TreeView\"]", Raises(NotImplementedError)),
    ("DiscreteWaveletData[{{0} -> {1.}, {1} -> {1.}}, 7, \"DiscreteWaveletTransform\"]", Raises(TypeError)),
    ("DiscreteWaveletData[{{0} -> {1.}}, HaarWavelet[], \"DiscreteWaveletTransform\"]", Raises(ValueError)),
]

IWT = [
    ("InverseWaveletTransform[DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1], Automatic, {0}]",
     near([1.5, 1.5, 3.5, 3.5])),
    ("InverseWaveletTransform[DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1], Automatic, {1}]",
     near([-0.5, 0.5, -0.5, 0.5])),
    ("InverseWaveletTransform[DiscreteWaveletTransform[{1, 2, 3, 4}, HaarWavelet[], 2, WorkingPrecision -> "
     "Infinity]]", "{1, 2, 3, 4}"),
    ("InverseWaveletTransform[x03u]", "InverseWaveletTransform[x03u]"),
    ("InverseWaveletTransform[{1, 2, 3}]", Raises(TypeError)),
    ("InverseWaveletTransform[DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1], Automatic, {7}]",
     Raises(ValueError)),
]

BESTBASIS = [
    ("WaveletBestBasis[DiscreteWaveletPacketTransform[{1, 1, 1, 1, 1, 1, 1, 1}, HaarWavelet[], 3]][\"BasisIndex\"]",
     "{{1}, {0, 1}, {0, 0, 0}, {0, 0, 1}}"),
    ("WaveletBestBasis[DiscreteWaveletPacketTransform[{1., -1, 1, -1, 1, -1, 1, -1}, HaarWavelet[], 2]]"
     "[\"BasisIndex\"]", Pred(lambda g: {(1, 0), (1, 1)} <= {tuple(int(v) for v in w.args) for w in g.args}, "contains {1,0},{1,1}")),
    ("Length[WaveletBestBasis[DiscreteWaveletPacketTransform[Range[8.], HaarWavelet[], 3], \"LogEnergy\"]"
     "[\"BasisIndex\"]] > 0", is_true()),
    ("WaveletBestBasis[x03u]", "WaveletBestBasis[x03u]"),
    ("WaveletBestBasis[DiscreteWaveletPacketTransform[Range[8.], HaarWavelet[], 2], \"NoSuchCost\"]",
     Raises(ValueError)),
    ("With[{x03x = Range[8.]}, Max[Abs[InverseWaveletTransform[WaveletBestBasis[DiscreteWaveletPacketTransform[x03x, "
     "HaarWavelet[], 3]]] - x03x]]]", less(1e-12)),
]

THRESH = [
    ("WaveletThreshold[DiscreteWaveletTransform[{1., 2, 3, 4, 5, 6, 7, 8}, HaarWavelet[], 1], {\"Hard\", 1.}]"
     "[{1}, \"Values\"]", near([[0, 0, 0, 0]])),
    ("WaveletThreshold[DiscreteWaveletTransform[{1., 5, 3, 4}, HaarWavelet[], 1], {\"Soft\", 1.}][{1}, \"Values\"]",
     near([[-(4 / S2 - 1), 0]])),
    ("WaveletThreshold[DiscreteWaveletTransform[{1., 5, 3, 4}, HaarWavelet[], 1], {\"Hard\", 1.}][{0}, \"Values\"]",
     near([[6 / S2, 7 / S2]])),
    ("WaveletThreshold[x03u]", "WaveletThreshold[x03u]"),
    ("WaveletThreshold[DiscreteWaveletTransform[{1., 2, 3, 4}], {\"NoSuchRule\", 1.}]", Raises(ValueError)),
    ("WaveletThreshold[{1, 2}]", Raises(TypeError)),
]

WMAP = [
    ("WaveletMapIndexed[#*0 &, DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1], {1}][{1}, \"Values\"]",
     near([[0, 0]])),
    ("WaveletMapIndexed[2 # &, DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1], {0}][{0}, \"Values\"]",
     near([[6 / S2, 14 / S2]])),
    ("InverseWaveletTransform[WaveletMapIndexed[#*0 &, DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1], "
     "{1}]]", near([1.5, 1.5, 3.5, 3.5])),
    ("WaveletMapIndexed[f, x03u]", "WaveletMapIndexed[f, x03u]"),
    ("WaveletMapIndexed[f, {1, 2}]", Raises(TypeError)),
]

WPLOTS = [
    ("WaveletMatrixPlot[DiscreteWaveletTransform[{1., 2, 3, 4}, HaarWavelet[], 1]]", head("Graphics")),
    ("WaveletMatrixPlot[DiscreteWaveletTransform[Table[i + j, {i, 8}, {j, 8}], HaarWavelet[], 2]]", head("Graphics")),
    ("WaveletMatrixPlot[x03u]", "WaveletMatrixPlot[x03u]"),
    ("WaveletMatrixPlot[{1, 2}]", Raises(TypeError)),
    ("WaveletImagePlot[DiscreteWaveletTransform[Table[i + j, {i, 8}, {j, 8}], HaarWavelet[], 1]]", head("Image")),
    ("ImageDimensions[WaveletImagePlot[DiscreteWaveletTransform[Table[i + j, {i, 8}, {j, 8}], HaarWavelet[], 2]]]",
     "{8, 8}"),
    ("WaveletImagePlot[x03u]", "WaveletImagePlot[x03u]"),
    ("WaveletImagePlot[DiscreteWaveletTransform[Range[8.]]]", Raises(NotImplementedError)),
    ("WaveletListPlot[DiscreteWaveletTransform[Range[16.], HaarWavelet[], 2]]", head("GraphicsColumn")),
    ("Length[First[WaveletListPlot[DiscreteWaveletTransform[Range[16.], HaarWavelet[], 2]]]]", "3"),
    ("WaveletListPlot[x03u]", "WaveletListPlot[x03u]"),
    ("WaveletListPlot[DiscreteWaveletTransform[Table[i + j, {i, 4}, {j, 4}]]]", Raises(NotImplementedError)),
    ("WaveletScalogram[DiscreteWaveletTransform[Range[16.], HaarWavelet[], 2]]", head("Graphics")),
    ("WaveletScalogram[ContinuousWaveletTransform[Range[32.]]]", head("Graphics")),
    ("WaveletScalogram[x03u]", "WaveletScalogram[x03u]"),
    ("WaveletScalogram[{1, 2, 3}]", Raises(TypeError)),
]

# ───────────────────────────────────────────────────────────────── continuous wavelets
CWT = [
    ("ContinuousWaveletTransform[Range[64.]][\"Octaves\"]", "5"),
    ("ContinuousWaveletTransform[Range[64.]][\"Voices\"]", "4"),
    ("Length[ContinuousWaveletTransform[Range[64.], MexicanHatWavelet[], {3, 2}][\"WaveletIndex\"]]", "6"),
    ("ContinuousWaveletTransform[x03u]", "ContinuousWaveletTransform[x03u]"),
    ("ContinuousWaveletTransform[\"abc\"]", Raises(TypeError)),
    ("ContinuousWaveletTransform[Range[64.], DaubechiesWavelet[2]]", Raises(NotImplementedError)),
    ("ContinuousWaveletTransform[Range[64.], MexicanHatWavelet[], {0, 2}]", Raises(ValueError)),
    ("Normal[ContinuousWaveletTransform[Range[16.], MexicanHatWavelet[], {1, 1}]][[1, 1]]", "{1, 1}"),
]

ICWT = [
    ("With[{x03x = " + X03_SIG_S + "}, Max[Abs[InverseContinuousWaveletTransform[ContinuousWaveletTransform[x03x, "
     "GaborWavelet[6], {4, 6}]] - x03x]]]", less(1e-6)),
    ("With[{x03x = " + X03_SIG_S + "}, Max[Abs[InverseContinuousWaveletTransform[ContinuousWaveletTransform[x03x]] - "
     "x03x]]]", less(1e-6)),
    ("With[{x03c = ContinuousWaveletTransform[" + X03_SIG_S + ", MorletWavelet[]]}, Max[Abs["
     "InverseContinuousWaveletTransform[x03c, Automatic, {1, _}] + InverseContinuousWaveletTransform[x03c, "
     "Automatic, Except[{1, _}]] + x03c[\"DataMean\"] - " + X03_SIG_S + "]]]", less(1e-6)),
    ("InverseContinuousWaveletTransform[x03u]", "InverseContinuousWaveletTransform[x03u]"),
    ("InverseContinuousWaveletTransform[{1, 2}]", Raises(TypeError)),
    ("InverseContinuousWaveletTransform[ContinuousWaveletTransform[Range[16.]], PaulWavelet[]]",
     Raises(NotImplementedError)),
    ("InverseContinuousWaveletTransform[ContinuousWaveletTransform[Range[16.]], Automatic, {9, 9}]",
     Raises(ValueError)),
]

CWDATA = [
    ("ContinuousWaveletTransform[Range[16.], MexicanHatWavelet[], {2, 2}][\"Wavelet\"]", "MexicanHatWavelet[]"),
    ("ContinuousWaveletTransform[Range[16.], MexicanHatWavelet[], {2, 2}][\"WaveletIndex\"]",
     "{{1, 1}, {1, 2}, {2, 1}, {2, 2}}"),
    ("ContinuousWaveletTransform[{1., 2, 3, 6}, MexicanHatWavelet[], {1, 1}][\"DataMean\"]", 3.0),
    ("ContinuousWaveletData[x03r, MexicanHatWavelet[]]", "ContinuousWaveletData[x03r, MexicanHatWavelet[]]"),
    ("ContinuousWaveletTransform[Range[16.], MexicanHatWavelet[], {2, 2}][\"Nonsense\"]", Raises(ValueError)),
    ("ContinuousWaveletTransform[Range[16.], MexicanHatWavelet[], {2, 2}][\"LogScalogramFunction\"]",
     Raises(NotImplementedError)),
]

# ───────────────────────────────────────────────────────────────── short-time Fourier
X03_L9 = [0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.8, 0.7, 0.6]
X03_L9_S = "{0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.8, 0.7, 0.6}"
STFT = [
    ("ShortTimeFourier[" + X03_SIG_S + "][\"PartitionSize\"]", "16"),
    ("ShortTimeFourier[" + X03_SIG_S + "][\"PartitionOffset\"]", "5"),
    ("ShortTimeFourier[" + X03_SIG_S + ", 8, 3][\"Data\"]", near(_stft_ref(X03_SIG, 8, 3))),
    ("ShortTimeFourier[" + X03_SIG_S + ", 8, 2, HannWindow][\"Data\"]", near(_stft_ref(X03_SIG, 8, 2, _hann(8)))),
    ("ShortTimeFourier[x03u]", "ShortTimeFourier[x03u]"),
    ("ShortTimeFourier[\"abc\"]", Raises(TypeError)),
    ("ShortTimeFourier[{1., 2, 3}, 0]", Raises(ValueError)),
    ("ShortTimeFourier[Audio[x03u]]", Raises(NotImplementedError)),
    ("ShortTimeFourierData[<|\"Data\" -> {{1, 2}}, \"PartitionSize\" -> 2, \"PartitionOffset\" -> 1|>][\"DataType\"]",
     "\"Complex\""),
    ("ShortTimeFourier[{1., 2, 3, 4}, 2, 2][\"Properties\"]",
     "{\"Data\", \"DataType\", \"FourierParameters\", \"PartitionOffset\", \"PartitionPadding\", \"PartitionSize\", "
     "\"SampleRate\", \"SmoothingWindow\"}"),
    ("With[{x03s = ShortTimeFourier[{1., 2, 3, 4}, 2, 2]}, Normal[x03s] == x03s[\"Data\"]]", is_true()),
    ("ShortTimeFourierData[x03r]", "ShortTimeFourierData[x03r]"),
    ("ShortTimeFourierData[{1, 2}]", Raises(TypeError)),
    ("ShortTimeFourierData[<|\"Data\" -> {{1, 2}}, \"PartitionSize\" -> 4, \"PartitionOffset\" -> 1|>]",
     Raises(ValueError)),
    ("ShortTimeFourier[{1., 2, 3, 4}, 2, 2][\"NoSuchProperty\"]", Raises(ValueError)),
]

SARRAY = [
    ("Max[Abs[Flatten[Map[InverseFourier[#, FourierParameters -> {1, -1}] &, SpectrogramArray[" + X03_L9_S +
     ", 3, 3]] - Partition[" + X03_L9_S + ", 3]]]]", less(1e-12)),
    ("SpectrogramArray[" + X03_SIG_S + ", 16, 4]", near(_stft_ref(X03_SIG, 16, 4))),
    ("Dimensions[SpectrogramArray[" + X03_SIG_S + ", 16, 4, DirichletWindow, 32]]", "{13, 32}"),
    ("SpectrogramArray[x03u]", "SpectrogramArray[x03u]"),
    ("SpectrogramArray[{1., 2, 3}, 4, 1, DirichletWindow, 2]", Raises(ValueError)),
    ("SpectrogramArray[{{1., 2}, {3, 4}}]", Raises(NotImplementedError)),
]

ISTFT = [
    ("With[{x03x = " + X03_SIG_S + "}, Max[Abs[InverseShortTimeFourier[ShortTimeFourier[x03x]] - x03x]]]",
     less(1e-12)),
    ("With[{x03x = " + X03_SIG_S + "}, Max[Abs[Take[InverseShortTimeFourier[ShortTimeFourier[x03x, 8, 2, "
     "HannWindow]], {2, -2}] - Take[x03x, {2, -2}]]]]", less(1e-12)),
    ("InverseShortTimeFourier[SpectrogramArray[" + X03_L9_S + ", 3, 3], 3, 3]", near(X03_L9)),
    ("InverseShortTimeFourier[x03u]", "InverseShortTimeFourier[x03u]"),
    ("InverseShortTimeFourier[{{1, 2}}, 4]", Raises(ValueError)),
    ("InverseShortTimeFourier[{1, 2}]", Raises(ValueError)),
]


def _isg_err(src_mag, n, d, win, method, x):
    def f(got):
        y = np.real(_np(got))
        w = _hann(n) if win == "HannWindow" else np.ones(n)
        S = np.abs(_stft_ref(x, n, d, w))
        S2_ = np.abs(_stft_ref(y, n, d, w))[:len(S)]
        return y.shape[0] >= len(x) - 0 and np.linalg.norm(S2_ - S) / np.linalg.norm(S) < 0.2
    return Pred(f, "relative spectrogram error < 0.2")


X03_MAG = np.abs(_stft_ref(X03_SIG2, 32, 4, _hann(32)))[:, :17]
X03_MAG_S = "{" + ",".join("{" + ",".join(repr(float(v)) for v in row) + "}" for row in X03_MAG) + "}"
ISPEC = [
    ("InverseSpectrogram[" + X03_MAG_S + ", 32, 4, HannWindow]", _isg_err(None, 32, 4, "HannWindow", "Hybrid",
                                                                         X03_SIG2)),
    ("Length[InverseSpectrogram[" + X03_MAG_S + ", 32, 4, HannWindow, Method -> \"SPSI\"]]", str(len(X03_SIG2))),
    ("Abs[InverseSpectrogram[{{4., 0, 0}}, 4, 1]]", near([1, 1, 1, 1])),
    ("InverseSpectrogram[x03u]", "InverseSpectrogram[x03u]"),
    ("InverseSpectrogram[" + X03_MAG_S + ", 32, 16]", Raises(ValueError)),
    ("InverseSpectrogram[{1, 2}]", Raises(ValueError)),
    ("InverseSpectrogram[Image[{{0.5}}]]", Raises(NotImplementedError)),
]

SPLOTS = [
    ("Spectrogram[" + X03_SIG_S + "]", head("Graphics")),
    ("Spectrogram[ShortTimeFourier[" + X03_SIG_S + ", 8, 2]]", head("Graphics")),
    ("Spectrogram[" + X03_SIG_S + ", 8, 2, HannWindow, 16, SampleRate -> 100]", head("Graphics")),
    ("Spectrogram[x03u]", "Spectrogram[x03u]"),
    ("Spectrogram[" + X03_SIG_S + ", Method -> \"MelFrequency\"]", Raises(NotImplementedError)),
    ("Spectrogram[\"abc\"]", Raises(TypeError)),
    ("Periodogram[" + X03_SIG_S + "]", head("Graphics")),
    ("Periodogram[{" + X03_SIG_S + ", " + X03_SIG2_S + "}, 16]", head("Graphics")),
    ("Periodogram[" + X03_SIG_S + ", ScalingFunctions -> \"Absolute\", SampleRate -> 8]", head("Graphics")),
    ("Periodogram[x03u]", "Periodogram[x03u]"),
    ("Periodogram[" + X03_SIG_S + ", ScalingFunctions -> \"Log10\"]", Raises(NotImplementedError)),
    ("Periodogram[\"abc\"]", Raises(TypeError)),
    ("Cepstrogram[" + X03_SIG_S + "]", head("Graphics")),
    ("Cepstrogram[" + X03_SIG_S + ", 16, 4]", head("Graphics")),
    ("Cepstrogram[" + X03_SIG_S + ", 16, 4, DirichletWindow, 32, SampleRate -> 10]", head("Graphics")),
    ("Cepstrogram[x03u]", "Cepstrogram[x03u]"),
    ("Cepstrogram[" + X03_SIG_S + ", 16, 4, HannWindow]", Raises(NotImplementedError)),
    ("Cepstrogram[\"abc\"]", Raises(TypeError)),
]

# ───────────────────────────────────────────────────────────────── spectral density
X03_R10 = [_acov(range(1, 11), h) for h in range(10)]
PSD = [
    ("PowerSpectralDensity[ARProcess[{x03a}, x03s^2], x03w]", "x03s^2/(1 + x03a^2 - 2 x03a Cos[x03w])"),
    ("PowerSpectralDensity[MAProcess[{1/2}, 1], x03w]", "5/4 + Cos[x03w]"),
    ("PowerSpectralDensity[ARMAProcess[{1/2}, {1/3}, 2], 0]", "2 (1 + 1/3)^2/(1 - 1/2)^2"),
    ("PowerSpectralDensity[Range[10], 0.7]",
     X03_R10[0] + 2 * sum(X03_R10[h] * math.cos(0.7 * h) for h in range(1, 10))),
    ("PowerSpectralDensity[Range[10], 0.7, 5]",
     X03_R10[0] + 2 * sum(X03_R10[h] * math.cos(0.7 * h) for h in range(1, 6))),
    ("PowerSpectralDensity[Range[10], 0.7, {5, BartlettWindow}]",
     X03_R10[0] + 2 * sum(X03_R10[h] * (1 - 2 * abs(h / 10)) * math.cos(0.7 * h) for h in range(1, 6))),
    ("PowerSpectralDensity[Range[10], 0.7, FourierParameters -> {-1, 1}]",
     (X03_R10[0] + 2 * sum(X03_R10[h] * math.cos(0.7 * h) for h in range(1, 10))) / (2 * math.pi)),
    ("PowerSpectralDensity[x03u, x03w]", "PowerSpectralDensity[x03u, x03w]"),
    ("PowerSpectralDensity[Range[10], x03w, 20]", Raises(ValueError)),
    ("PowerSpectralDensity[{1, 2, 3}, x03w, {2, BartlettWindow, 3}]", Raises(ValueError)),
    ("PowerSpectralDensity[ARProcess[{{{1/2, 0}, {0, 1/3}}}, {{1, 0}, {0, 1}}], x03w]", Raises(NotImplementedError)),
    ("PowerSpectralDensity[\"abc\", x03w]", Raises(TypeError)),
    # Check: the process spectrum equals the covariance series Σ r(h) e^{-ihω} (truncated, AR(1) a = 1/2)
    ("N[PowerSpectralDensity[ARProcess[{1/2}, 1], 0.4]]",
     sum((4 / 3) * 0.5 ** abs(h) * math.cos(0.4 * h) for h in range(-200, 201)), 1e-9),
]

# ───────────────────────────────────────────────────────────────── filters
FILTERS = [
    ("WienerFilter[{1, 2, 3, 2, 1}, 1]", near(_wiener2([1, 2, 3, 2, 1], 1))),
    ("WienerFilter[" + X03_SIG_S + ", 2, 0.01]", near(_wiener2(X03_SIG, 2, 0.01))),
    ("WienerFilter[{3., 3, 3, 3}, 1]", near([3, 3, 3, 3])),
    ("WienerFilter[{1., 5, 2, 8}, 1, 0]", near([1, 5, 2, 8])),
    ("Dimensions[WienerFilter[Table[N[i j], {i, 5}, {j, 6}], 1]]", "{5, 6}"),
    ("Length[WienerFilter[{1., 5, 2, 8, 3}, 1, Padding -> None]]", "3"),
    ("WienerFilter[x03u, 1]", "WienerFilter[x03u, 1]"),
    ("WienerFilter[{1., 2}, -1]", Raises(ValueError)),
    ("WienerFilter[{1., 2}, 1.5]", Raises(TypeError)),
    ("WienerFilter[Image[{{0.5}}], 1]", Raises(NotImplementedError)),
    ("EstimatedBackground[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}, 2]",
     near(_snip([2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1], 2))),
    ("EstimatedBackground[{2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}, 0]",
     near([2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1])),
    ("EstimatedBackground[{1., 1, 9, 1, 1}, 1, Method -> \"MovingAverage\"]", near([1, 1, 11 / 3, 1, 1])),
    ("With[{x03y = {2, 1, 3, 5, 6, 6, 4, 3, 2, 4, 7, 3, 2, 4, 2, 2, 1}}, And @@ Thread[EstimatedBackground[x03y] <= "
     "x03y]]", is_true()),
    ("EstimatedBackground[x03u]", "EstimatedBackground[x03u]"),
    ("EstimatedBackground[{1., 2}, Method -> \"Nope\"]", Raises(ValueError)),
    ("EstimatedBackground[{1., 2}, Method -> {\"SNIP\", 4}]", Raises(NotImplementedError)),
    ("EstimatedBackground[{1., 2}, 1.5]", Raises(TypeError)),
    ("ListDeconvolve[{1/4, 1/2, 1/4}, ListConvolve[{1/4, 1/2, 1/4}, {0, 0, 1., 3, 1, 0, 0, 0, 2, 0}, {2, 2}, 0], "
     "Padding -> 0, MaxIterations -> 50]", near([0, 0, 1, 3, 1, 0, 0, 0, 2, 0], 1e-8)),
    ("ListDeconvolve[{1/2, 1/2}, {1., 1, 1, 1}, Padding -> \"Fixed\"]", near([1, 1, 1, 1])),
    ("With[{x03k = {0.2, 0.6, 0.2}, x03t = {0, 1., 4, 1, 0, 0, 2, 0}}, Max[Abs[ListDeconvolve[x03k, "
     "ListConvolve[x03k, x03t, {2, 2}, 0], Padding -> 0, Method -> \"RichardsonLucy\", MaxIterations -> 2000] - "
     "x03t]]]", less(1e-2)),
    ("ListDeconvolve[x03u, {1, 2}]", "ListDeconvolve[x03u, {1, 2}]"),
    ("ListDeconvolve[{1, 2, 3}, {1, 2}]", Raises(ValueError)),
    ("ListDeconvolve[{1, 2}, {1, 2, 3}, Method -> \"Tikhonov\"]", Raises(NotImplementedError)),
    ("ListDeconvolve[{1, 2}, {1, 2, 3}, MaxIterations -> 1.5]", Raises(TypeError)),
]

# ───────────────────────────────────────────────────────────────── systems
SYSTEMS = [
    ("TransferFunctionModel[1/(x03s + 1), x03s]",
     "TransferFunctionModel[{{{1}}, 1 + x03s}, x03s, SamplingPeriod -> None, SystemsModelLabels -> None]"),
    ("TransferFunctionModel[{{1/(x03s + 1), 1/(x03s + 2)}}, x03s][1]", "{{1/2, 1/3}}"),
    ("TransferFunctionModel[(x03s + 3)/(x03s^2 + 3 x03s + 2), x03s][\"Poles\"]", "{-2, -1}"),
    ("TransferFunctionModel[(x03s + 3)/(x03s^2 + 3 x03s + 2), x03s][\"Zeros\"]", "{-3}"),
    ("TransferFunctionModel[{{{1, x03s}}, x03s^2 + 1}, x03s][I/2]", "{{4/3, 2 I/3}}"),
    ("TransferFunctionModel[StateSpaceModel[{{{0, 1}, {-2, -3}}, {{0}, {1}}, {{1, 0}}}]][x03q]",
     "{{1/(x03q^2 + 3 x03q + 2)}}"),
    ("TransferFunctionModel[x03h, x03s]", "TransferFunctionModel[x03h, x03s]"),
    ("TransferFunctionModel[Exp[-x03s]/(x03s + 1), x03s]", Raises(NotImplementedError)),
    ("TransferFunctionModel[1/(x03s + 1), 3]", Raises(TypeError)),
    ("TransferFunctionModel[1/(x03s + 1), x03s][-1]", Raises(ValueError)),
    ("StateSpaceModel[{{{0, 1}, {-2, -3}}, {{0}, {1}}, {{1, 0}}}]",
     "StateSpaceModel[{{{0, 1}, {-2, -3}}, {{0}, {1}}, {{1, 0}}, {{0}}}, SamplingPeriod -> None, "
     "SystemsModelLabels -> None]"),
    ("StateSpaceModel[TransferFunctionModel[(x03s + 3)/(x03s^2 + 3 x03s + 2), x03s]]",
     "StateSpaceModel[{{{0, 1}, {-2, -3}}, {{0}, {1}}, {{3, 1}}, {{0}}}, SamplingPeriod -> None, "
     "SystemsModelLabels -> None]"),
    ("TransferFunctionModel[StateSpaceModel[TransferFunctionModel[{{1/(x03s + 1), x03s/(x03s + 2)}}, x03s]], x03s][2]",
     "{{1/3, 1/2}}"),
    ("StateSpaceModel[x03h]", "StateSpaceModel[x03h]"),
    ("StateSpaceModel[{{{0, 1}}, {{0}, {1}}, {{1, 0}}}]", Raises(ValueError)),
    ("StateSpaceModel[TransferFunctionModel[x03s^2/(x03s + 1), x03s]]", Raises(NotImplementedError)),
    ("StateSpaceModel[7]", Raises(TypeError)),
]

X03_TF1 = "TransferFunctionModel[1/(x03s + 1), x03s]"


def _bode_ok(got):
    if not (isinstance(got, Expr) and got.head == "GraphicsColumn"):
        return False
    return True


PLOTS = [
    ("BodePlot[" + X03_TF1 + "]", Pred(_bode_ok, "GraphicsColumn")),
    ("BodePlot[" + X03_TF1 + ", {0.1, 10}]", head("GraphicsColumn")),
    ("BodePlot[{" + X03_TF1 + ", TransferFunctionModel[1/(x03s + 2), x03s]}]", head("GraphicsColumn")),
    ("BodePlot[TransferFunctionModel[1/(x03s + x03a), x03s]]",
     "BodePlot[TransferFunctionModel[1/(x03s + x03a), x03s]]"),
    ("BodePlot[" + X03_TF1 + ", {10, 1}]", Raises(ValueError)),
    ("BodePlot[1/(x03s + 1)]", Raises(TypeError)),
    ("NyquistPlot[" + X03_TF1 + "]", head("Graphics")),
    ("NyquistPlot[StateSpaceModel[{{{-1}}, {{1}}, {{1}}}]]", head("Graphics")),
    ("NyquistPlot[" + X03_TF1 + ", {0.01, 100}]", head("Graphics")),
    ("NyquistPlot[x03u]", "NyquistPlot[x03u]"),
    ("NyquistPlot[{1, 2}]", Raises(TypeError)),
    ("NicholsPlot[" + X03_TF1 + "]", head("Graphics")),
    ("NicholsPlot[TransferFunctionModel[10/(x03s (x03s + 1)), x03s]]", head("Graphics")),
    ("NicholsPlot[" + X03_TF1 + ", {1, 2}]", head("Graphics")),
    ("NicholsPlot[x03u]", "NicholsPlot[x03u]"),
    ("NicholsPlot[" + X03_TF1 + ", {-1, 2}]", Raises(ValueError)),
    ("SingularValuePlot[TransferFunctionModel[{{1/(x03s + 1), 1/(x03s + 2)}, {0, 1/(x03s + 3)}}, x03s]]",
     head("Graphics")),
    ("SingularValuePlot[" + X03_TF1 + "]", head("Graphics")),
    ("SingularValuePlot[TransferFunctionModel[1/(x03s + 1), x03s, SamplingPeriod -> 1/10]]", head("Graphics")),
    ("SingularValuePlot[x03u]", "SingularValuePlot[x03u]"),
    ("SingularValuePlot[" + X03_TF1 + ", {1}]", Raises(ValueError)),
]

KALMAN = [
    ("KalmanFilter[ARProcess[{1/3}, 1], {x03v, x03x, x03y, x03z}]", "{0, x03v/3, x03x/3, x03y/3}"),
    ("KalmanFilter[MAProcess[{1/2}, 1], {1, 2, 3}]", "{0, 2/5, 16/21}"),
    ("KalmanFilter[ARMAProcess[{0.4}, {0.8}, 1], {1., 2, 3}]",
     Pred(lambda g: abs(float(_np(g)[0].real)) < 1e-15 and abs(float(_np(g)[1].real) -
                                                                _arma_pred([1, 2, 3])[1]) < 1e-12 and
          abs(float(_np(g)[2].real) - _arma_pred([1, 2, 3])[2]) < 1e-12, "Durbin–Levinson")),
    ("KalmanFilter[ARProcess[{x03a}, 1], {1, 2}]", "KalmanFilter[ARProcess[{x03a}, 1], {1, 2}]"),
    ("KalmanFilter[ARProcess[1, {1/3}, 1], {1, 2}]", Raises(NotImplementedError)),
    ("KalmanFilter[7, {1, 2}]", Raises(TypeError)),
]


def _arma_pred(x):
    phi, th = 0.4, 0.8
    r = [(1 + 2 * th * phi + th * th) / (1 - phi * phi)]
    r.append((1 + phi * th) * (phi + th) / (1 - phi * phi))
    r.append(phi * r[1])
    out = [0.0, r[1] / r[0] * x[0]]
    k1 = r[1] / r[0]
    v1 = r[0] * (1 - k1 * k1)
    k2 = (r[2] - k1 * r[1]) / v1
    p1 = k1 - k2 * k1
    out.append(p1 * x[1] + k2 * x[0])
    return out


X03_W1 = [1, 2, 3, 4, 5, 6, 7, 8]
X03_W2 = [3 * v + 1 for v in [1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 8]]
CTW = [
    ("CanonicalWarpingDistance[{1, 2, 3, 4, 5, 6, 7, 8}, {4, 4, 7, 10, 13, 16, 16, 19, 22, 25, 25}]",
     Pred(_ctw_ref_ok, "≈ 0 (affine copy with a time warp)")),
    ("CanonicalWarpingDistance[{{0, 0}, {1, 0}, {1, 1}}, {{-1, -2}, {1, 2}, {0, 0}}] >= 0", is_true()),
    ("CanonicalWarpingDistance[{1., 2, 3, 4, 5}, {-10., -20, -30, -40, -50}]",
     Pred(_ctw_ref_ok, "≈ 0 (negated, scaled copy)")),
    ("CanonicalWarpingDistance[x03u, {1, 2}]", "CanonicalWarpingDistance[x03u, {1, 2}]"),
    ("CanonicalWarpingDistance[{1, 1, 1}, {1, 2}]", Raises(ValueError)),
    ("CanonicalWarpingDistance[{1, 2, 3}, {1, 2}, Automatic, {\"Parallelogram\", 2}]", Raises(NotImplementedError)),
    ("CanonicalWarpingDistance[7, {1, 2}]", Raises(TypeError)),
    ("CanonicalWarpingCorrespondence[{1, 2, 3, 4, 5, 6, 7, 8}, {4, 4, 7, 10, 13, 16, 16, 19, 22, 25, 25}]",
     "{{1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 8}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}}"),
    ("First /@ CanonicalWarpingCorrespondence[{1., 3, 2, 5, 4}, {2., 1, 7, 3}]", "{1, 1}"),
    ("Last /@ CanonicalWarpingCorrespondence[{1., 3, 2, 5, 4}, {2., 1, 7, 3}]", "{5, 4}"),
    ("CanonicalWarpingCorrespondence[x03u, {1, 2}]", "CanonicalWarpingCorrespondence[x03u, {1, 2}]"),
    ("CanonicalWarpingCorrespondence[{1, 2}, {1, 2}, {{1}, {1, 2}}]", Raises(ValueError)),
    ("CanonicalWarpingCorrespondence[{1, 2}, {1, 2}, DistanceFunction -> ManhattanDistance]",
     Raises(NotImplementedError)),
]

# ───────────────────────────────────────────────────────────────── inverse transforms
INVERSE = [
    ("InverseHilbertTransform[Cos[x03t], x03t, x03q]", "-Sin[x03q]"),
    ("InverseHilbertTransform[Sin[x03t], x03t, x03q]", "Cos[x03q]"),
    ("InverseHilbertTransform[1/(Pi x03t), x03t, x03q]", "DiracDelta[x03q]"),
    ("InverseHilbertTransform[1/(1 + x03t^2), x03t, x03q]", "-x03q/(1 + x03q^2)"),
    ("InverseHilbertTransform[1/(1 + x03t^2), x03t, 0.7]", X03_IHT_RAT, 1e-8),
    ("InverseHilbertTransform[UnitBox[x03t], x03t, 0.3]", X03_IHT_BOX, 1e-12),
    ("InverseHilbertTransform[Cos[x03t] - Sin[x03t], x03t, 1.3]", -math.sin(1.3) - math.cos(1.3)),
    ("InverseHilbertTransform[I Exp[2 I x03t], x03t, x03q]", "-Exp[2 I x03q]"),
    ("InverseHilbertTransform[Sin[x03a x03t], x03t, x03q, Assumptions -> x03a > 0]", "Cos[x03a x03q]"),
    ("InverseHilbertTransform[Piecewise[{{1 - x03t^2, Abs[x03t] <= 1}}, 0], x03t, 0.4]",
     _pv_iht(lambda t: (1 - t * t) if abs(t) <= 1 else 0.0, 0.4, (-1, 1)), 1e-8),
    ("InverseHilbertTransform[x03f[x03t], x03t, x03q]", "InverseHilbertTransform[x03f[x03t], x03t, x03q]"),
    ("InverseHilbertTransform[Cos[x03t], 3, x03q]", Raises(TypeError)),
    ("InverseHilbertTransform[Cos[x03t], x03t, x03q, 4]", Raises(ValueError)),
    ("InverseBilateralLaplaceTransform[ConditionalExpression[-3/((1 + x03s) (x03s - 2)), -1 < Re[x03s] < 2], x03s, "
     "x03t]", "HeavisideTheta[x03t] Exp[-x03t] + HeavisideTheta[-x03t] Exp[2 x03t]"),
    ("InverseBilateralLaplaceTransform[ConditionalExpression[1/(x03s^2 + 2 x03s + 2), Re[x03s] > -1], x03s, x03t]",
     "HeavisideTheta[x03t] Exp[-x03t] Sin[x03t]"),
    ("InverseBilateralLaplaceTransform[ConditionalExpression[1/(x03s + 1)^2, Re[x03s] < -1], x03s, x03t]",
     "-x03t Exp[-x03t] HeavisideTheta[-x03t]"),
    ("InverseBilateralLaplaceTransform[ConditionalExpression[Exp[-2 x03s]/(x03s + 1), Re[x03s] > -1], x03s, x03t]",
     "HeavisideTheta[x03t - 2] Exp[2 - x03t]"),
    ("InverseBilateralLaplaceTransform[x03s^2 + 1, x03s, x03t]", "DiracDelta[x03t] + DiracDelta''[x03t]"),
    ("InverseBilateralLaplaceTransform[ConditionalExpression[1/(x03s^2 - 1), -1 < Re[x03s] < 1], x03s, 0.5]",
     -0.5 * math.exp(-0.5)),
    ("BilateralLaplaceTransform[InverseBilateralLaplaceTransform[ConditionalExpression[1/((x03s + 1) (x03s + 3)), "
     "Re[x03s] > -1], x03s, x03t], x03t, x03s]",
     "ConditionalExpression[1/((x03s + 1) (x03s + 3)), Re[x03s] > -1]"),
    ("InverseBilateralLaplaceTransform[x03F[x03s], x03s, x03t]", "InverseBilateralLaplaceTransform[x03F[x03s], x03s, "
     "x03t]"),
    ("InverseBilateralLaplaceTransform[1/(x03s + 1), x03s, x03t]", Raises(ValueError)),
    ("InverseBilateralLaplaceTransform[ConditionalExpression[1/(x03s + 1), -3 < Re[x03s] < 2], x03s, x03t]",
     Raises(ValueError)),
    ("InverseBilateralLaplaceTransform[1/(x03s + 1), 3, x03t]", Raises(TypeError)),
    ("InverseBilateralLaplaceTransform[x03s, x03s, {x03t, x03y}]", Raises(NotImplementedError)),
    ("InverseBilateralZTransform[ConditionalExpression[x03z (4 x03z - 7)/((2 x03z - 1) (x03z - 3)), 1/2 < Abs[x03z] "
     "< 3], x03z, x03n]", "UnitStep[x03n]/2^x03n - 3^x03n UnitStep[-x03n - 1]"),
    ("InverseBilateralZTransform[ConditionalExpression[x03z/(x03z - 2)^2, Abs[x03z] < 2], x03z, x03n]",
     "-x03n 2^(x03n - 1) UnitStep[-x03n - 1]"),
    ("InverseBilateralZTransform[ConditionalExpression[x03z/(x03z^2 + 1/4), Abs[x03z] > 1/2], x03z, x03n]",
     "2^(1 - x03n) UnitStep[x03n] Sin[Pi x03n/2]"),
    ("InverseBilateralZTransform[ConditionalExpression[x03z/(x03z - 1/2), Abs[x03z] > 1/2], x03z, 3]", "1/8"),
    ("InverseBilateralZTransform[1 + 1/x03z^2, x03z, x03n]", "DiscreteDelta[x03n] + DiscreteDelta[x03n - 2]"),
    ("BilateralZTransform[InverseBilateralZTransform[ConditionalExpression[x03z/((x03z - 1/2) (x03z - 3)), 1/2 < "
     "Abs[x03z] < 3], x03z, x03n], x03n, x03z]",
     Pred(lambda g: isinstance(g, Expr) and g.head == "ConditionalExpression", "round trip")),
    ("InverseBilateralZTransform[x03F[x03z], x03z, x03n]", "InverseBilateralZTransform[x03F[x03z], x03z, x03n]"),
    ("InverseBilateralZTransform[x03z/(x03z - 2), x03z, x03n]", Raises(ValueError)),
    ("InverseBilateralZTransform[ConditionalExpression[x03z/(x03z - 2), 1 < Abs[x03z] < 3], x03z, x03n]",
     Raises(ValueError)),
    ("InverseBilateralZTransform[x03z, x03z, 1/2]", Raises(TypeError)),
    ("InverseBilateralZTransform[x03z, x03z, {x03n, x03m}]", Raises(NotImplementedError)),
]

CASES = (DWT + PACKET + STATIONARY + LIFTING + DWDATA + IWT + BESTBASIS + THRESH + WMAP + WPLOTS + CWT + ICWT
         + CWDATA + STFT + SARRAY + ISTFT + ISPEC + SPLOTS + PSD + FILTERS + SYSTEMS + PLOTS + KALMAN + CTW + INVERSE)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_x03(case):
    check(case)


def test_bode_values():
    """Reference: the Bode magnitude/phase of 1/(s + 1) are -10 Log10(1 + ω²) dB and -ArcTan[ω] (degrees)."""
    g = ev("BodePlot[TransferFunctionModel[1/(x03s + 1), x03s], {0.1, 10}]")
    pts = []

    def walk(x):
        if isinstance(x, Expr):
            if x.head == "Line":
                pts.append(_np(x.args[0]))
            for a in x.args:
                walk(a)
    walk(g)
    assert len(pts) >= 2
    mag, ph = pts[0], pts[1]
    w = np.exp(mag[:, 0]) if mag[:, 0].min() < 0 else mag[:, 0]
    w_log = mag[:, 0].real
    w = np.exp(w_log) if np.all(w_log < 3) else w_log
    assert np.allclose(mag[:, 1].real, -10 * np.log10(1 + w.real ** 2), atol=1e-6)
    assert np.allclose(ph[:, 1].real, -np.degrees(np.arctan(w.real)), atol=1e-6)
