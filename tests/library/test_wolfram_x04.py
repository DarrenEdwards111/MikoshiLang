"""Differential tests for Wolfram batch X04 (images III: filters, restoration, multiscale pyramids, composition and
image creation; modules wl_imageproc2, wl_imageproc2_create, wl_barcode, wl_radon, wl_cie_data).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): documented definitions and examples
  (LaplacianFilter kernels, the flat-field formula of BrightnessEqualize, the Poisson-noise scale of ImageEffect,
  the Radon transforms of a Gaussian and of the unit disk, ...);
* published references for the stated methods: Fornberg (1988) finite-difference weights; Unser (1999) cubic
  B-spline derivative responses 3 sin ω/(2 + cos ω) and 6 (1 - cos ω)/(2 + cos ω); Burt & Adelson (1983) pyramids
  (perfect reconstruction of Laplacian pyramids); Chambolle–Pock (2011) total variation; Perona & Malik (1990);
  Buades, Coll & Morel (IPOL 2011); Mertens, Kautz & Van Reeth (2007); Durand & Dorsey (2002); Machado, Oliveira &
  Fernandes (2009) colour-vision matrices; ISO/IEC 18004 (QR: the "HELLO WORLD" 1-M codewords from Thonky's QR
  tutorial), GS1 (EAN-13 4006381333931), ISO/IEC 15417 (Code 128 "Wikipedia" from the Code 128 article); the CIE
  1931 2° colour-matching tables (CIE 018:2019);
* closed forms and independent computations written in the tests (exact ramps, constant images, quadratures).

Symbol names used as symbolic placeholders are prefixed ``x04`` so they cannot collide with other test modules.
"""
import math

import numpy as np
import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr


def _num(x):
    from mikoshilang.library.wl_graphics_color import real_value
    return float(real_value(x, "test", "x"))


def _nest(e):
    if isinstance(e, Expr) and e.head == "List":
        return [_nest(a) for a in e.args]
    if isinstance(e, Expr) and e.head in ("Image", "Image3D"):
        return _nest(e.args[0])
    return _num(e)


def _close(g, w, tol):
    if isinstance(w, (list, tuple)):
        return isinstance(g, list) and len(g) == len(w) and all(_close(a, b, tol) for a, b in zip(g, w))
    return not isinstance(g, list) and abs(g - w) <= tol * max(1.0, abs(w))


def N(want, tol=1e-6):
    """Pred: a (nested) list of machine numbers, or an image whose raw data is, close to want."""
    def ok(got):
        try:
            return _close(_nest(got), want, tol)
        except Exception:
            return False
    return Pred(ok, f"~{want}")


def Small(tol):
    """Pred: a real number with absolute value at most tol."""
    def ok(got):
        try:
            return abs(_num(got)) <= tol
        except Exception:
            return False
    return Pred(ok, f"|x| <= {tol}")


def Head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"head {h}")


def ImgType(t):
    return Pred(lambda g: isinstance(g, Expr) and g.head in ("Image", "Image3D") and g.args[1] == t, f"type {t}")


R = "Image[Table[N[(i + 2 j)/40], {i, 10}, {j, 12}]]"                  # linear ramp (gray)
C = "Image[Table[{i/10., j/12., 0.5}, {i, 10}, {j, 12}]]"               # smooth colour image
K0 = "Image[ConstantArray[0.3, {6, 6}]]"                                 # constant
G = "Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}]"
M = "Table[Boole[3 <= i <= 6 && 4 <= j <= 8], {i, 10}, {j, 12}]"         # inpainting mask
RAW = "Image[Table[Which[EvenQ[i] && EvenQ[j], 0.2, OddQ[i] && OddQ[j], 0.8, True, 0.5], {i, 0, 7}, {j, 0, 7}]]"
ID3 = "{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}"
SINE = "N[Table[Sin[2 Pi k/16], {k, 0, 15}]]"
W1 = math.pi / 8
W2 = math.pi / 4

LINEAR = [
    # LaplacianFilter (reference: LaplacianFilter page — r = 1 kernel {1, -2, 1} and the 2D 5-point stencil)
    ("LaplacianFilter[{0, 0, 1, 1, 1, 0, 0}, 1, WorkingPrecision -> Infinity]", "{0, 1, -1, 0, -1, 1, 0}"),
    ("LaplacianFilter[{{0, 0, 0}, {0, 1, 0}, {0, 0, 0}}, 1, WorkingPrecision -> Infinity]",
     "{{0, 1, 0}, {1, -4, 1}, {0, 1, 0}}"),
    ("LaplacianFilter[{0, 0, 1, 1, 1, 0, 0}, 1]", N([0, 1, -1, 0, -1, 1, 0])),
    ("LaplacianFilter[{{1, 2}, {3, 4}}, 1, Padding -> \"Periodic\", WorkingPrecision -> Infinity]",
     "{{6, 2}, {-2, -6}}"),                                                          # hand-computed wrap-around
    ("LaplacianFilter[Range[7]^2, 2, WorkingPrecision -> Infinity][[3 ;; 5]]", "{2, 2, 2}"),   # Check: exact x^2''
    ("LaplacianFilter[{1, 2, 3, 2, 1, 0, 0, 0}, 1, Padding -> None]", N([0, -2, 0, 0, 1, 0])),
    ("LaplacianFilter[x04x, 1]", "LaplacianFilter[x04x, 1]"),
    ("LaplacianFilter[{1, x04x}, 1]", "LaplacianFilter[{1, x04x}, 1]"),
    ("LaplacianFilter[{1, 2}, -1]", Raises(ValueError, "^LaplacianFilter: ")),
    ("LaplacianFilter[{1, 2}, \"a\"]", Raises(TypeError, "^LaplacianFilter: ")),
    # DerivativeFilter (Unser 1999: the cubic spline derivative of a sampled sine has response 3 sin ω/(2 + cos ω))
    (f"DerivativeFilter[{SINE}, {{1}}, Padding -> \"Periodic\"][[1 ;; 3]]",
     N([3 * math.sin(W1) / (2 + math.cos(W1)) * math.cos(W1 * k) for k in range(3)], 1e-9)),
    ("DerivativeFilter[N[Range[40]^2], {1}][[18 ;; 22]]", N([36, 38, 40, 42, 44], 1e-8)),   # Check: interior 2x
    ("DerivativeFilter[Table[N[i^2 + 3 j], {i, 6}, {j, 6}], {0, 1}, Padding -> \"Periodic\"][[1 ;; 2, 1]]",
     N([-8.4, -8.4], 1e-9)),                         # periodic spline of 3 j: constant except at the wrap
    ("DerivativeFilter[x04x, {1}]", "DerivativeFilter[x04x, {1}]"),
    ("DerivativeFilter[{1., 2., 3.}, {1}, Padding -> None]", Raises(NotImplementedError, "^DerivativeFilter: ")),
    ("DerivativeFilter[{1., 2., 3.}, {-1}]", Raises(ValueError, "^DerivativeFilter: ")),
    ("DerivativeFilter[{1., 2., 3.}, \"a\"]", Raises(TypeError, "^DerivativeFilter: ")),
    # RidgeFilter (Hessian; the spline second derivative response 6 (1 - cos ω)/(2 + cos ω))
    ("RidgeFilter[N[Table[Cos[2 Pi j/8], {i, 8}, {j, 8}]], 0, Padding -> \"Periodic\"][[1, {1, 8}]]",
     N([6 * (1 - math.cos(W2)) / (2 + math.cos(W2)) * math.cos(W2),
        6 * (1 - math.cos(W2)) / (2 + math.cos(W2))], 1e-9)),
    ("Max[RidgeFilter[N[Table[Cos[2 Pi j/8], {i, 8}, {j, 8}]], 0, Padding -> \"Periodic\"][[1, 3 ;; 5]]] < 10^-12",
     "True"),                                                                        # valleys give 0
    ("Dimensions[RidgeFilter[N[Table[Sin[i] Cos[j], {i, 9}, {j, 9}]]]]", "{9, 9}"),
    ("RidgeFilter[x04x]", "RidgeFilter[x04x]"),
    ("RidgeFilter[{1., 2., 3.}]", Raises(NotImplementedError, "^RidgeFilter: ")),
    ("RidgeFilter[{{1., 2.}, {3., 4.}}, -1]", Raises(ValueError, "^RidgeFilter: ")),
    # GaborFilter (convolution with GaborMatrix, documented relation)
    ("GaborFilter[{0, 0, 0, 1, 0, 0, 0}, 1, 1.][[3 ;; 5]] == GaborMatrix[1, 1.]", "True"),
    ("GaborFilter[{0, 0, 0, 1, 0, 0, 0}, 1, 1.]", N([0, 0, 0.057545966301093515, 0.7869860421615985,
                                                      0.057545966301093515, 0, 0], 1e-9)),
    ("Dimensions[GaborFilter[N[Table[Sin[i + j], {i, 8}, {j, 8}]], 2, {1., 0.5}]]", "{8, 8}"),
    ("GaborFilter[x04x, 1, 1.]", "GaborFilter[x04x, 1, 1.]"),
    ("GaborFilter[{1., 2.}, 1, 1., WorkingPrecision -> 30]", Raises(NotImplementedError, "^GaborFilter: ")),
    ("GaborFilter[{1., 2.}, -1, 1.]", Raises(ValueError, "^GaborFilter: ")),
    # ImagePeriodogram (|FFT|^2 peaks at the image frequency; Center alignment)
    ("Position[ImageData[ImagePeriodogram[Image[Table[Cos[2 Pi 2 j/8.], {i, 8}, {j, 8}]]]], 1.]",
     "{{5, 3}, {5, 7}}"),
    ("ImageDimensions[ImagePeriodogram[Image[Table[Cos[2 Pi 2 j/8.], {i, 8}, {j, 8}]]]]", "{8, 8}"),
    ("Max[ImageData[ImagePeriodogram[Image[Table[Cos[2 Pi 2 j/8.], {i, 8}, {j, 8}]]]]]", N(1.0)),
    ("ImagePeriodogram[x04x]", "ImagePeriodogram[x04x]"),
    ("ImagePeriodogram[{1, 2}]", Raises(TypeError, "^ImagePeriodogram: ")),
]

PYR = f"ImagePyramid[Image[Table[N[Sin[i/3] Cos[j/4]]^2, {{i, 13}}, {{j, 9}}]]"
PYRAMIDS = [
    # ImagePyramid / InverseImagePyramid / ImagePyramidApply (Burt & Adelson 1983)
    (f"{PYR}][\"LevelCount\"]", "5"),                                    # halving down to 1 x 1
    (f"ImageDimensions /@ {PYR}][All]", "{{9, 13}, {5, 7}, {3, 4}, {2, 2}, {1, 1}}"),
    (f"{PYR}, \"Laplacian\"][\"Type\"]", "\"Bandpass\""),
    (f"With[{{x04q = {PYR}, \"Laplacian\"]}}, Max[Abs[ImageData[InverseImagePyramid[x04q]] - "
     "ImageData[Image[Table[N[Sin[i/3] Cos[j/4]]^2, {i, 13}, {j, 9}]]]]]]", Small(1e-6)),   # Check: perfect reconstruction
    (f"With[{{x04q = {PYR}]}}, InverseImagePyramid[x04q] === x04q[1]]", "True"),
    (f"ImageDimensions[InverseImagePyramid[{PYR}, \"Laplacian\"], \"Laplacian\", 2]]", "{5, 7}"),
    (f"ImagePyramidApply[ImageMultiply[#, 2] &, {PYR}, \"Laplacian\"]][\"LevelCount\"]", "5"),
    (f"With[{{x04q = {PYR}]}}, ImagePyramidApply[ImageAdd, {{x04q, x04q}}][1] == ImageMultiply[x04q[1], 2]]", "True"),
    (f"Dimensions[ImageData[{PYR}, {{\"Lowpass\", {{1, 2, 1}}/4}}, 2][2]]]", "{7, 5}"),
    ("ImagePyramid[x04x]", "ImagePyramid[x04x]"),
    ("InverseImagePyramid[x04x]", "InverseImagePyramid[x04x]"),
    ("ImagePyramidApply[x04f, x04x]", "ImagePyramidApply[x04f, x04x]"),
    ("ImagePyramid[3]", Raises(TypeError, "^ImagePyramid: ")),
    (f"{PYR}, \"Gaussian\", 2, 3]", Raises(NotImplementedError, "^ImagePyramid: ")),
    (f"{PYR}, \"Foo\"]", Raises(ValueError, "^ImagePyramid: ")),
    ("InverseImagePyramid[3]", Raises(TypeError, "^InverseImagePyramid: ")),
    ("ImagePyramidApply[x04f, 3]", Raises(TypeError, "^ImagePyramidApply: ")),
]

NOISY = "Image[Table[If[j > 6, 0.8, 0.2] + 0.05 Sin[7 i + 3 j], {i, 12}, {j, 12}]]"   # deterministic "noise"
STEP = "Image[Table[If[j > 6, 0.8, 0.2], {i, 12}, {j, 12}]]"
BLUR = f"ImageConvolve[{NOISY}, {{{{0, 0.1, 0}}, {{0.1, 0.6, 0.1}}, {{0, 0.1, 0}}}}, Padding -> \"Periodic\"]"
KER = "{{0, 0.1, 0}, {0.1, 0.6, 0.1}, {0, 0.1, 0}}"


def _std_cols(src, cols="1 ;; 5"):
    return f"StandardDeviation[Flatten[ImageData[{src}][[All, {cols}]]]]"


RESTORATION = [
    # TotalVariationFilter (Chambolle–Pock 2011; constant data are a fixed point)
    ("TotalVariationFilter[{1., 1., 1.}]", N([1, 1, 1])),
    ("TotalVariationFilter[{1, 2, 3, 10, 11, 12}, 0.5]", N([1.5, 2, 3, 10, 11, 11.5], 1e-4)),   # Check: TV denoising
    (f"{_std_cols(f'TotalVariationFilter[{NOISY}, 0.1]')} < {_std_cols(NOISY)}", "True"),
    ("TotalVariationFilter[{0, 0, 1, 0, 0}, 0.2, Method -> \"Laplacian\"] // Length", "5"),
    ("TotalVariationFilter[x04x, 0.1]", "TotalVariationFilter[x04x, 0.1]"),
    ("TotalVariationFilter[{1, 2}, -1]", Raises(ValueError, "^TotalVariationFilter: ")),
    ("TotalVariationFilter[{1, 2}, \"a\"]", Raises(TypeError, "^TotalVariationFilter: ")),
    # PeronaMalikFilter (Perona & Malik 1990; constants unchanged, mean preserved by the Neumann scheme)
    (f"Max[Abs[ImageData[PeronaMalikFilter[{K0}]] - 0.3]]", Small(1e-6)),
    (f"Abs[Mean[Flatten[ImageData[PeronaMalikFilter[{NOISY}, 2]]]] - Mean[Flatten[ImageData[{NOISY}]]]]",
     Small(1e-6)),                                                                    # Check: conservation
    (f"{_std_cols(f'PeronaMalikFilter[{NOISY}, 2]')} < {_std_cols(NOISY)}", "True"),
    ("PeronaMalikFilter[x04x]", "PeronaMalikFilter[x04x]"),
    (f"PeronaMalikFilter[{R}, -1]", Raises(ValueError, "^PeronaMalikFilter: ")),
    ("PeronaMalikFilter[{1., 2., 3.}]", Raises(TypeError, "^PeronaMalikFilter: ")),
    # CurvatureFlowFilter (straight edges have zero curvature: a vertical step is a fixed point)
    (f"Max[Abs[ImageData[CurvatureFlowFilter[{K0}, 2]] - 0.3]]", Small(1e-6)),
    (f"Max[Abs[ImageData[CurvatureFlowFilter[{STEP}, 2]] - ImageData[{STEP}]]]", Small(1e-6)),   # Check
    (f"ImageDimensions[CurvatureFlowFilter[{NOISY}]]", "{12, 12}"),
    ("CurvatureFlowFilter[x04x]", "CurvatureFlowFilter[x04x]"),
    (f"CurvatureFlowFilter[{R}, -1]", Raises(ValueError, "^CurvatureFlowFilter: ")),
    ("CurvatureFlowFilter[{1., 2., 3.}]", Raises(TypeError, "^CurvatureFlowFilter: ")),
    # NonlocalMeansFilter (Buades, Coll & Morel, IPOL 2011)
    (f"Max[Abs[ImageData[NonlocalMeansFilter[{K0}, 1]] - 0.3]]", Small(1e-6)),       # Check: weighted mean
    (f"{_std_cols(f'NonlocalMeansFilter[{NOISY}, 1]')} < {_std_cols(NOISY)}", "True"),
    (f"ImageType[NonlocalMeansFilter[Image[{NOISY}, \"Byte\"], 1]]", "\"Byte\""),
    ("NonlocalMeansFilter[x04x, 1]", "NonlocalMeansFilter[x04x, 1]"),
    (f"NonlocalMeansFilter[{R}, -1]", Raises(ValueError, "^NonlocalMeansFilter: ")),
    (f"NonlocalMeansFilter[{R}, 1.5]", Raises(TypeError, "^NonlocalMeansFilter: ")),
    # FourierDCTFilter (t = 0 keeps every coefficient: the DCT round trip is exact up to rounding)
    (f"Max[Abs[ImageData[FourierDCTFilter[{NOISY}, 0]] - ImageData[{NOISY}]]]", Small(1e-6)),   # Check
    (f"{_std_cols(f'FourierDCTFilter[{NOISY}, 0.1]')} < {_std_cols(NOISY)}", "True"),
    (f"ImageDimensions[FourierDCTFilter[{C}, 0.05]]", "{12, 10}"),
    ("FourierDCTFilter[x04x, 0.1]", "FourierDCTFilter[x04x, 0.1]"),
    (f"FourierDCTFilter[{R}, -0.1]", Raises(ValueError, "^FourierDCTFilter: ")),
    ("FourierDCTFilter[{1., 2.}, 0.1]", Raises(TypeError, "^FourierDCTFilter: ")),
    # ImageDeconvolve (spectral methods invert a periodic blur exactly when regularization -> 0)
    (f"Max[Abs[ImageData[ImageDeconvolve[{BLUR}, {KER}, Padding -> \"Periodic\", Method -> {{\"Tikhonov\", 10^-9}}]]"
     f" - ImageData[{NOISY}]]]", Small(1e-4)),                                         # Check: inverse filter
    (f"Max[Abs[ImageData[ImageDeconvolve[{BLUR}, {KER}, Padding -> \"Periodic\", Method -> {{\"TSVD\", 10^-6}}]]"
     f" - ImageData[{NOISY}]]]", Small(1e-4)),
    (f"Max[Abs[ImageData[ImageDeconvolve[{R}, {ID3}]] - ImageData[{R}]]]", Small(1e-6)),   # identity kernel
    (f"Max[Abs[ImageData[ImageDeconvolve[{BLUR}, {KER}, Method -> \"RichardsonLucy\", MaxIterations -> 50, "
     f"Padding -> \"Periodic\"]] - ImageData[{NOISY}]]] < Max[Abs[ImageData[{BLUR}] - ImageData[{NOISY}]]]", "True"),
    ("ImageDeconvolve[x04x, {{1}}]", "ImageDeconvolve[x04x, {{1}}]"),
    (f"ImageDeconvolve[{R}, {ID3}, Method -> \"Hybrid\"]", Raises(NotImplementedError, "^ImageDeconvolve: ")),
    (f"ImageDeconvolve[{R}, {ID3}, Method -> \"Foo\"]", Raises(ValueError, "^ImageDeconvolve: ")),
    ("ImageDeconvolve[{1., 2.}, {{1}}]", Raises(TypeError, "^ImageDeconvolve: ")),
    # Inpaint (an exact linear ramp is harmonic, so diffusion and TV recover it)
    (f"Max[Abs[ImageData[Inpaint[{R}, {M}, Method -> \"Diffusion\"]] - ImageData[{R}]]]", Small(1e-6)),   # Check
    (f"Max[Abs[ImageData[Inpaint[{R}, {M}, Method -> \"FastMarching\"]] - ImageData[{R}]]]", Small(0.02)),
    (f"Max[Abs[ImageData[Inpaint[{R}, Rectangle[{{3, 3}}, {{6, 6}}], Method -> \"Diffusion\"]] - ImageData[{R}]]]",
     Small(1e-6)),
    (f"ImageDimensions[Inpaint[{R}, {M}, Method -> \"TotalVariation\"]]", "{12, 10}"),
    (f"ImageDimensions[Inpaint[{R}, {M}]]", "{12, 10}"),
    (f"Inpaint[{R}, x04x]", f"Inpaint[{R}, x04x]"),
    (f"Inpaint[{R}, Disk[{{5, 5}}, 2], Method -> \"NavierStokes\"]", Raises(NotImplementedError, "^Inpaint: ")),
    (f"Inpaint[{R}, {{{{1, 1}}}}]", Raises(NotImplementedError, "^Inpaint: ")),
    (f"Inpaint[{R}, 3]", Raises(TypeError, "^Inpaint: ")),
    (f"Inpaint[{R}, {{{{1, 0, 1}}}}]", Raises(ValueError, "^Inpaint: ")),
    # ImageDemosaic (bilinear demosaicing of an RGGB mosaic of constant colour planes)
    (f"Union[Flatten[ImageData[ImageDemosaic[{RAW}, \"RGGB\"]], 1]]", N([[0.2, 0.5, 0.8]])),   # Check: planes
    (f"ImageDemosaic[{RAW}, {{{{Red, Green}}, {{Green, Blue}}}}] === ImageDemosaic[{RAW}, \"Bayer\"]", "True"),
    (f"ImageDemosaic[{RAW}, {{\"BGGR\", {{2, 2}}}}] === ImageDemosaic[{RAW}, \"RGGB\"]", "True"),
    (f"ImageColorSpace[ImageDemosaic[{RAW}, \"GRBG\"]]", "\"RGB\""),
    ("ImageDemosaic[x04x, \"RGGB\"]", "ImageDemosaic[x04x, \"RGGB\"]"),
    (f"ImageDemosaic[{RAW}, \"CYGM\"]", Raises(NotImplementedError, "^ImageDemosaic: ")),
    (f"ImageDemosaic[{RAW}, \"XYZW\"]", Raises(ValueError, "^ImageDemosaic: ")),
    (f"ImageDemosaic[{RAW}, 3]", Raises(TypeError, "^ImageDemosaic: ")),
]

ILLUM = "Image[Table[0.5 + j/40., {i, 10}, {j, 14}]]"
OBJ = "Image[Table[0.3 + 0.2 Sin[i/2.], {i, 10}, {j, 14}]]"
ENHANCE = [
    # BrightnessEqualize (documented flat-field formula (image - D) Mean[F - D]/(F - D) on LAB lightness)
    (f"Max[Abs[ImageData[BrightnessEqualize[Image[ConstantArray[{{0.4, 0.5, 0.6}}, {{6, 7}}]]]] - "
     "ImageData[Image[ConstantArray[{0.4, 0.5, 0.6}, {6, 7}]]]]]", Small(1e-9)),
    (f"Max[Abs[ImageData[BrightnessEqualize[Image[Table[0.6, {{i, 8}}, {{j, 9}}]]]] - 0.6]]", Small(1e-9)),
    (f"ImageDimensions[BrightnessEqualize[{C}]]", "{12, 10}"),
    (f"Max[Abs[ImageData[BrightnessEqualize[{R}, 0.7]] - ImageData[{R}]]]", Small(1e-9)),   # Check: F constant
    ("BrightnessEqualize[x04x]", "BrightnessEqualize[x04x]"),
    (f"BrightnessEqualize[{R}, 0.5, 0.6]", Raises(ValueError, "^BrightnessEqualize: ")),
    (f"BrightnessEqualize[{R}, {{x04x^2, {{x04a}}, {{x04x}}}}]", Raises(NotImplementedError, "^BrightnessEqualize: ")),
    (f"BrightnessEqualize[{R}, \"a\"]", Raises(TypeError, "^BrightnessEqualize: ")),
    # ColorToneMapping (c = 0 with s = 1 keeps colours, scaled to unit luminance; Schlick/Mantiuk colour rule)
    (f"Max[Abs[ImageData[ColorToneMapping[{C}, 0, 1, Method -> \"LogLuminance\"]] - Clip[ImageData[{C}]/"
     f"Max[ImageData[{C}] . {{0.2126, 0.7152, 0.0722}}], {{0, 1}}]]]", Small(1e-6)),     # Check
    (f"With[{{x04t = ImageData[ColorToneMapping[{R}, 1, Method -> \"LogLuminance\"]]}}, Max[x04t] - Min[x04t]]",
     Small(1e-6)),                                                                   # c = 1: uniform luminance
    (f"ImageType[ColorToneMapping[{C}]]", "\"Real32\""),
    (f"Max[ImageData[ColorToneMapping[{C}, 0.5, Method -> \"RGB\"]]]", N(1.0)),
    ("ColorToneMapping[x04x]", "ColorToneMapping[x04x]"),
    (f"ColorToneMapping[{R}, 2]", Raises(ValueError, "^ColorToneMapping: ")),
    (f"ColorToneMapping[{R}, 0.5, Method -> \"DodgingAndBurning\"]", Raises(NotImplementedError, "^ColorToneMapping: ")),
    (f"ColorToneMapping[{R}, {{All, \"a\"}}]", Raises(TypeError, "^ColorToneMapping: ")),
    # ImageExposureCombine / ImageFocusCombine (pyramid fusion; identical inputs reproduce the image)
    (f"Max[Abs[ImageData[ImageExposureCombine[{{{R}, {R}}}]] - ImageData[{R}]]]", Small(1e-6)),   # Check
    (f"ImageDimensions[ImageExposureCombine[{{{C}, ImageMultiply[{C}, 0.5]}}]]", "{12, 10}"),
    (f"ImageColorSpace[ImageExposureCombine[{{{C}, {C}}}]]", "\"RGB\""),
    ("ImageExposureCombine[x04x]", "ImageExposureCombine[x04x]"),
    (f"ImageExposureCombine[{{{R}, {R}}}, \"HDR\"]", Raises(NotImplementedError, "^ImageExposureCombine: ")),
    ("ImageExposureCombine[{}]", Raises(ValueError, "^ImageExposureCombine: ")),
    ("ImageExposureCombine[3]", Raises(TypeError, "^ImageExposureCombine: ")),
    (f"Max[Abs[ImageData[ImageFocusCombine[{{{R}, {R}}}]] - ImageData[{R}]]]", Small(1e-6)),      # Check
    (f"With[{{x04s = ImageData[{NOISY}]}}, Max[Abs[ImageData[ImageFocusCombine[{{{NOISY}, "
     f"GaussianFilter[{NOISY}, 2]}}]] - x04s][[All, 3 ;; 10]]] < Max[Abs[ImageData[GaussianFilter[{NOISY}, 2]] - "
     "x04s][[All, 3 ;; 10]]]]", "True"),                                             # the sharp image wins
    (f"ImageDimensions[ImageFocusCombine[{{{C}, {C}}}]]", "{12, 10}"),
    ("ImageFocusCombine[x04x]", "ImageFocusCombine[x04x]"),
    ("ImageFocusCombine[{}]", Raises(ValueError, "^ImageFocusCombine: ")),
    ("ImageFocusCombine[3]", Raises(TypeError, "^ImageFocusCombine: ")),
]

BARCODES = [
    # BarcodeImage (ISO/IEC 18004, GS1, ISO/IEC 15417; every symbol is decoded back as a self-check)
    ("ImageDimensions[BarcodeImage[\"HELLO WORLD\", \"QR\"]]", "{116, 116}"),     # version 1: (21 + 8) x 4 pixels
    ("ImageType[BarcodeImage[\"HELLO WORLD\", {\"QR\", \"H\"}]]", "\"Bit\""),
    ("ImageData[BarcodeImage[\"HELLO WORLD\", \"QR\"]][[17, 17 ;; 44 ;; 4]]", "{0, 0, 0, 0, 0, 0, 0}"),  # finder edge
    ("ImageData[BarcodeImage[\"4006381333931\", \"EAN13\"]][[1, 21 ;; 26]]", "{0, 0, 1, 1, 0, 0}"),   # guard 101
    ("ImageData[BarcodeImage[\"400638133393\", \"EAN13\"]] == ImageData[BarcodeImage[\"4006381333931\", \"EAN13\"]]",
     "True"),                                                                         # Check: GS1 check digit
    ("ImageDimensions[BarcodeImage[\"Wikipedia\", \"Code128\"]]", "{308, 60}"),      # 134 + 20 modules x 2 pixels
    ("ImageDimensions[BarcodeImage[\"03600029145\", \"UPC\"]]", "{230, 60}"),
    ("ImageDimensions[BarcodeImage[\"9638507\", \"EAN8\"]]", "{174, 60}"),
    ("ImageDimensions[BarcodeImage[\"HELLO-123\", \"Code39\", {400, 30}]]", "{390, 30}"),
    ("ImageDimensions[BarcodeImage[\"12345678\", \"ITF\"]]", "{202, 60}"),
    ("BarcodeImage[x04x, \"QR\"]", "BarcodeImage[x04x, \"QR\"]"),
    ("BarcodeImage[\"x\", \"Aztec\"]", Raises(NotImplementedError, "^BarcodeImage: ")),
    ("BarcodeImage[\"x\", {\"QR\", 3}]", Raises(NotImplementedError, "^BarcodeImage: ")),
    ("BarcodeImage[\"4006381333932\", \"EAN13\"]", Raises(ValueError, "^BarcodeImage: ")),
    ("BarcodeImage[\"abc\", \"Code39\"]", Raises(ValueError, "^BarcodeImage: ")),
    ("BarcodeImage[3, \"QR\"]", Raises(TypeError, "^BarcodeImage: ")),
]

CREATE = [
    # RegionImage (coverage rasterisation; disk area Pi r^2)
    ("With[{x04d = RegionImage[Disk[{0, 0}, 1], RasterSize -> 40]}, Total[1 - Flatten[ImageData[x04d]]] (2/40)^2]",
     N(math.pi, 2e-3)),                                                              # Check: area
    ("ImageData[RegionImage[Rectangle[{0, 0}, {2, 1}], {{-1, 3}, {-1, 2}}, RasterSize -> {4, 3}]]",
     N([[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]])),
    ("ImageDimensions[RegionImage[Line[{{0, 0}, {1, 1}}], RasterSize -> 20]]", "{20, 20}"),
    ("ImageDimensions[RegionImage[Ball[{0, 0, 0}, 1], RasterSize -> 6]]", "{6, 6, 6}"),
    ("RegionImage[x04x]", "RegionImage[x04x]"),
    ("RegionImage[Disk[], {{0, 1}}]", Raises(ValueError, "^RegionImage: ")),
    ("RegionImage[3]", Raises(TypeError, "^RegionImage: ")),
    # ReliefImage (Lambert shading: a plane gives a uniform shade n . l)
    ("Union[Flatten[ImageData[ReliefImage[Table[N[x + 2 y], {y, 5}, {x, 6}]], \"Byte\"][[2 ;; 4, 2 ;; 5]]]]",
     "{201}"),                                                                       # Check: hand-computed n . l
    ("ImageType[ReliefImage[Table[N[x + 2 y], {y, 5}, {x, 6}], ColorFunction -> (Hue[#] &)]]", "\"Byte\""),
    (f"ImageDimensions[ReliefImage[{C}]]", "{12, 10}"),
    ("ReliefImage[x04x]", "ReliefImage[x04x]"),
    ("ReliefImage[{{1., 2.}}, Method -> \"AspectBasedShading\"]", Raises(NotImplementedError, "^ReliefImage: ")),
    ("ReliefImage[{1., 2.}]", Raises(ValueError, "^ReliefImage: ")),
    ("ReliefImage[\"a\"]", Raises(TypeError, "^ReliefImage: ")),
    # HighlightImage (blend (1 - a) pixel + a colour, a = 0.5, Red; Darken halves the background)
    (f"HighlightImage[{C}, Rectangle[{{0, 0}}, {{6, 5}}]][[1, 1, 1, 1, 1]]",
     N([1.0, 0.5 * (1 / 12), 0.25], 1e-5)),                                          # Check: bottom-left blend
    (f"HighlightImage[{C}, Rectangle[{{0, 0}}, {{6, 5}}], \"Darken\"][[1, 1, 1, 10, 12]]",
     N([0.05, 0.5, 0.25], 1e-5)),
    (f"Head[HighlightImage[{C}, {{{{1, 1}}, {{3, 4}}}}]]", "Graphics"),
    (f"Dimensions[HighlightImage[{C}, Disk[{{6, 5}}, 3], \"Remove\"][[1, 1, 1]]]", "{10, 12, 4}"),
    (f"HighlightImage[{C}, x04x]", f"HighlightImage[{C}, x04x]"),
    (f"HighlightImage[{C}, Tooltip[Disk[{{6, 5}}, 3], \"a\"]]", Raises(NotImplementedError, "^HighlightImage: ")),
    (f"HighlightImage[{C}, Disk[{{6, 5}}, 3], \"Foo\"]", Raises(ValueError, "^HighlightImage: ")),
    (f"HighlightImage[{C}, \"a\"]", Raises(TypeError, "^HighlightImage: ")),
    # ImageCollage (justified rows keep aspect ratios; two equal images sit side by side)
    (f"With[{{x04cc = ImageCollage[{{{C}, {C}}}, Automatic, 24]}}, Max[Abs[ImageData[x04cc][[All, 13 ;; 24]] - "
     f"ImageData[{C}]]]]", Small(1e-6)),                                              # Check
    (f"ImageDimensions[ImageCollage[{{{C}, {C}}}]]", "{24, 10}"),
    (f"ImageDimensions[ImageCollage[{{{C}, {R}, {C}}}, \"Fit\", 60, Method -> \"Grid\"]]", "{60, 50}"),
    (f"ImageDimensions[ImageCollage[{{2 -> {C}, 1 -> {R}}}, Method -> \"Columns\"]]",
     Pred(lambda g: isinstance(g, Expr) and g.head == "List" and len(g.args) == 2, "dimensions")),
    ("ImageCollage[x04x]", "ImageCollage[x04x]"),
    (f"ImageCollage[{{{C}}}, Method -> \"ClosestPacking\"]", Raises(NotImplementedError, "^ImageCollage: ")),
    ("ImageCollage[{}]", Raises(ValueError, "^ImageCollage: ")),
    ("ImageCollage[3]", Raises(TypeError, "^ImageCollage: ")),
    # ImageEffect (sepia matrix, Machado 2009 matrices, documented Poisson scale, noise statistics)
    ("ImageData[ImageEffect[Image[{{{0.5, 0.2, 0.1}}}], \"Sepia\"]]",
     N([[[0.393 * 0.5 + 0.769 * 0.2 + 0.189 * 0.1, 0.349 * 0.5 + 0.686 * 0.2 + 0.168 * 0.1,
          0.272 * 0.5 + 0.534 * 0.2 + 0.131 * 0.1]]], 1e-6)),
    ("ImageData[ImageEffect[Image[{{{1., 0, 0}}}], \"Deuteranopia\"]]",
     N([[[1.055 * 0.367322 ** (1 / 2.4) - 0.055, 1.055 * 0.280085 ** (1 / 2.4) - 0.055, 0.0]]], 1e-6)),
    ("With[{x04n = ImageEffect[Image[ConstantArray[0.5, {100, 100}]], {\"GaussianNoise\", 0.1}, RandomSeeding -> 3]},"
     " {Mean[Flatten[ImageData[x04n]]], StandardDeviation[Flatten[ImageData[x04n]]]}]", N([0.5, 0.1], 0.02)),  # Check
    ("ImageData[ImageEffect[Image[{{0.1, 0.4, 0.6, 0.9}}], {\"Posterization\", 2}]]", N([[0, 0, 1, 1]])),
    ("ImageData[ImageEffect[Image[{{0.2, 0.7}}], {\"Solarization\", 0.5}]]", N([[0.2, 0.3]])),
    ("Length[ImageEffect[]]", "15"),
    ("ImageEffect[x04x, \"Sepia\"]", "ImageEffect[x04x, \"Sepia\"]"),
    (f"ImageEffect[{C}, \"Charcoal\"]", Raises(NotImplementedError, "^ImageEffect: ")),
    (f"ImageEffect[{C}, \"Foo\"]", Raises(ValueError, "^ImageEffect: ")),
    (f"ImageEffect[{C}, 3]", Raises(TypeError, "^ImageEffect: ")),
]

CHROMA = [
    # ImageWaveformPlot (per-column level counts)
    (f"Dimensions[ImageWaveformPlot[{C}][[1, 1, 1]]]", "{128, 12, 3}"),
    (f"Head[ImageWaveformPlot[{C}, \"L\"]]", "Graphics"),
    ("With[{x04w = ImageWaveformPlot[Image[{{0.}, {0.}, {1.}}]][[1, 1, 1]]}, {x04w[[1, 1]], x04w[[128, 1]]}]",
     N([[1, 1, 1], [0.5, 0.5, 0.5]])),                                               # Check: counts 1 and 2
    ("ImageWaveformPlot[x04x]", "ImageWaveformPlot[x04x]"),
    (f"ImageWaveformPlot[{C}, PlotLayout -> \"Row\"]", Raises(NotImplementedError, "^ImageWaveformPlot: ")),
    (f"ImageWaveformPlot[{C}, \"Foo\"]", Raises(ValueError, "^ImageWaveformPlot: ")),
    ("ImageWaveformPlot[{1, 2}]", Raises(TypeError, "^ImageWaveformPlot: ")),
    # ChromaticityPlot (CIE 1931 2° locus; sRGB primaries Bradford-adapted to D50 = ColorConvert's XYZ)
    ("Cases[ChromaticityPlot[Red], Point[x04p_] :> x04p, Infinity][[1, 1]]",
     N([0.648427, 0.330856], 1e-4)),
    ("Cases[ChromaticityPlot[\"sRGB\"], Line[x04p_] :> x04p, Infinity][[2, 1]]",
     N([0.648434, 0.330856], 1e-4)),                                                 # Check: equals Red above
    ("MemberQ[Cases[ChromaticityPlot[\"sRGB\"], Polygon[x04p_] :> x04p, Infinity][[1]], {0.074302, 0.833803}]", "True"),
    (f"Head[ChromaticityPlot[{C}, \"CIE76\"]]", "Graphics"),
    ("ChromaticityPlot[x04x]", "ChromaticityPlot[x04x]"),
    ("ChromaticityPlot[\"LAB\"]", Raises(NotImplementedError, "^ChromaticityPlot: ")),
    ("ChromaticityPlot[\"Foo\"]", Raises(ValueError, "^ChromaticityPlot: ")),
    ("ChromaticityPlot[Red, 3]", Raises(TypeError, "^ChromaticityPlot: ")),
    # ChromaticityPlot3D
    ("Count[ChromaticityPlot3D[\"RGB\", \"LAB\", PlotPoints -> 3], _Polygon, Infinity]", "24"),
    ("Cases[ChromaticityPlot3D[Red, \"XYZ\"], Point[x04p_] :> x04p, Infinity][[1, 1]]",
     N([0.4360747, 0.2225045, 0.0139322], 1e-5)),                                   # Check: ColorConvert[Red, "XYZ"]
    (f"Head[ChromaticityPlot3D[{{{C}, Red}}]]", "Graphics3D"),
    ("ChromaticityPlot3D[x04x]", "ChromaticityPlot3D[x04x]"),
    ("ChromaticityPlot3D[\"AdobeRGB\"]", Raises(NotImplementedError, "^ChromaticityPlot3D: ")),
    ("ChromaticityPlot3D[Red, \"Foo\"]", Raises(ValueError, "^ChromaticityPlot3D: ")),
    ("ChromaticityPlot3D[Red, 3]", Raises(TypeError, "^ChromaticityPlot3D: ")),
]

RADON = [
    # RadonTransform (reference: RadonTransform page — Gaussian and unit-disk examples)
    ("RadonTransform[Exp[-x04x^2 - x04y^2], {x04x, x04y}, {x04p, x04t}]", "Sqrt[Pi] Exp[-x04p^2]"),
    ("RadonTransform[Boole[x04x^2 + x04y^2 < 1], {x04x, x04y}, {x04p, x04t}] /. x04p -> 1/2", "Sqrt[3]"),
    ("RadonTransform[x04x Exp[-x04x^2 - x04y^2], {x04x, x04y}, {x04p, x04t}]",
     "Sqrt[Pi] x04p Cos[x04t] Exp[-x04p^2]"),
    ("RadonTransform[1/(1 + x04x^2 + x04y^2)^2, {x04x, x04y}, {x04p, x04t}]",
     "Pi/(2 (1 + x04p^2)^(3/2))"),                                                   # Check: numeric quadrature
    ("RadonTransform[x04f[x04x, x04y], {x04x, x04y}, {x04p, x04t}]",
     "RadonTransform[x04f[x04x, x04y], {x04x, x04y}, {x04p, x04t}]"),
    ("RadonTransform[Exp[-x04x^2], x04x, {x04p, x04t}]", Raises(TypeError, "^RadonTransform: ")),
    ("RadonTransform[Exp[-x04x^2], {x04x, x04x}, {x04p, x04t}]", Raises(ValueError, "^RadonTransform: ")),
    ("RadonTransform[Exp[-x04x^2], {x04x, x04y}, {x04p, x04t}, GenerateConditions -> True]",
     Raises(NotImplementedError, "^RadonTransform: ")),
    # InverseRadonTransform (Abel inversion of φ-independent transforms)
    ("InverseRadonTransform[Sqrt[Pi] Exp[-x04p^2], {x04p, x04t}, {x04x, x04y}]", "Exp[-x04x^2 - x04y^2]"),
    ("InverseRadonTransform[Pi/(2 (1 + x04p^2)^(3/2)), {x04p, x04t}, {x04x, x04y}]",
     "1/(1 + x04x^2 + x04y^2)^2"),                                                   # Check: round trip
    ("InverseRadonTransform[Sqrt[Pi/2] Exp[-x04p^2/2], {x04p, x04t}, {x04x, x04y}]",
     "Exp[-(x04x^2 + x04y^2)/2]/2"),
    ("InverseRadonTransform[x04g[x04p], {x04p, x04t}, {x04x, x04y}]",
     "InverseRadonTransform[x04g[x04p], {x04p, x04t}, {x04x, x04y}]"),
    ("InverseRadonTransform[x04p Cos[x04t] Exp[-x04p^2], {x04p, x04t}, {x04x, x04y}]",
     Raises(NotImplementedError, "^InverseRadonTransform: ")),
    ("InverseRadonTransform[x04p Exp[-x04p^2], {x04p, x04t}, {x04x, x04y}]",
     Raises(ValueError, "^InverseRadonTransform: ")),
    ("InverseRadonTransform[Exp[-x04p^2], x04p, {x04x, x04y}]", Raises(TypeError, "^InverseRadonTransform: ")),
]

CASES = LINEAR + PYRAMIDS + RESTORATION + ENHANCE + BARCODES + CREATE + CHROMA + RADON


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_x04(case):
    check(case)


# ───────────────────────────── module-level references ─────────────────────────────

def test_qr_hello_world_codewords():
    """Reference: Thonky's QR tutorial, "HELLO WORLD" at 1-M (data and error-correction codewords)."""
    from mikoshilang.library import wl_barcode as B
    ver, mode, data = B.qr_codewords("HELLO WORLD", "M")
    assert (ver, mode) == (1, "Alphanumeric")
    assert data == [32, 91, 11, 120, 209, 114, 220, 77, 67, 64, 236, 17, 236, 17, 236, 17]
    assert B.qr_interleave(data, ver, "M")[16:] == [196, 35, 39, 119, 235, 215, 231, 226, 93, 23]


def test_qr_format_and_capacity_tables():
    """ISO/IEC 18004 Annex C: format information for M with mask 0 is 101010000010010; Table 7 capacities."""
    from mikoshilang.library import wl_barcode as B
    assert format(B.qr_format_bits("M", 0), "015b") == "101010000010010"
    assert format(B.qr_format_bits("L", 4), "015b") == "110011000101111"
    assert [B.qr_data_codewords(v, "L") for v in (1, 2, 10, 40)] == [19, 34, 274, 2956]
    assert B.qr_data_codewords(40, "H") == 1276


def test_qr_round_trip_all_levels():
    """Check: every generated symbol decodes back (format BCH, Reed–Solomon syndromes, segment data)."""
    from mikoshilang.library import wl_barcode as B
    for text in ("01234567", "HELLO WORLD", "https://example.org/x?q=1", "Grüße", "A" * 300):
        for lev in "LMQH":
            m, _ver, _mask = B.qr_matrix(text, lev)
            assert B.qr_decode(m) == text


def test_qr_matches_segno_when_available():
    """Independent encoder (segno) with the same version and mask gives the same module matrix."""
    segno = pytest.importorskip("segno")
    from mikoshilang.library import wl_barcode as B
    for text in ("HELLO WORLD", "01234567", "9" * 200, "AB" * 60):
        for lev in "LMQH":
            m, ver, mask = B.qr_matrix(text, lev)
            q = segno.make_qr(text, error=lev.lower(), version=ver, mask=mask, boost_error=False)
            ref = np.array([[1 if c else 0 for c in row] for row in q.matrix], dtype=np.int8)
            assert (ref == m).all(), (text, lev)


def test_linear_barcodes_reference_patterns():
    """GS1 and ISO/IEC 15417 reference symbols: EAN-13 4006381333931 and Code 128 "Wikipedia" (start B, checksum)."""
    from mikoshilang.library import wl_barcode as B
    d = B._gs1_digits("400638133393", 13, "EAN13")
    assert d == "4006381333931"
    mods = B.ean13_modules(d)
    assert len(mods) == 95 and mods[:3] == "101" and mods[45:50] == "01010"
    assert B.ean_decode(mods) == d
    vals = B.code128_values("Wikipedia")
    assert vals[0] == 104 and vals[1:-1] == [ord(c) - 32 for c in "Wikipedia"]
    assert vals[-1] == (104 + sum(i * v for i, v in enumerate(vals[1:-1], 1))) % 103
    assert B.code128_decode(B.code128_modules("RI476394652CH")) == "RI476394652CH"
    assert B.itf_decode(B.itf_modules("9081726354")) == "9081726354"
    assert B.code39_decode(B.code39_modules("HELLO-123")) == "HELLO-123"


def test_cie_tables():
    """CIE 1931 2° table: ybar(555) = 1 and the 520 nm chromaticity (0.0743, 0.8338)."""
    from mikoshilang.library import wl_cie_data as CIE
    t = {r[0]: r[1:] for r in CIE.CIE1931_2DEG}
    assert len(CIE.CIE1931_2DEG) == len(CIE.CIE1964_10DEG) == 81
    assert t[555][1] == 1.0
    x, y, z = t[520]
    assert abs(x / (x + y + z) - 0.0743) < 1e-4 and abs(y / (x + y + z) - 0.8338) < 1e-4


def test_pyramid_api_for_later_batches():
    """Python API used by X05: Laplacian levels collapse back to the input."""
    from mikoshilang.library import wl_imageproc2 as P
    a = np.random.default_rng(0).random((13, 9, 1))
    n = P.pyramid_level_count(a.shape[:2])
    lev = P.laplacian_levels(a, 2, n)
    assert np.abs(P.collapse_laplacian(lev, 2) - a).max() < 1e-9
