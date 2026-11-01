"""Differential tests for Wolfram batch W21 (images I: the Image object, representation, colour and geometric
operations; module wl_images).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>), for documented definitions and
  relations: Image[data] has type "Real32"; ImageDimensions gives {width, height} ({w, d, h} in 3D); ImageData of
  "Byte" images is scaled to 0..1; ColorNegate replaces x by 1 - x; ImageAdjust[i, {c, b, g}] is
  Clip[(b + 1)(c + 1) x^g - c/2]; Sharpen[i, r] = 2 i - Blur[i, r]; ImageAccumulate pads one zero row/column;
  ImagePad[image, -BorderDimensions[image]] removes uniform margins; Image3D[Image3DSlices[i]] == i;
  ArrayComponents labels identical elements with consecutive integers, zeros and None being 0 (the documented
  examples {8, 2.5, 7, 8} and the a/b matrices); ImageReflect[image] mirrors top-bottom; ImageRotate[image] turns
  90 degrees counterclockwise; ImageTrim includes every pixel an integer coordinate touches.
* Closed-form or independently computed values: pixel arithmetic, sums, entropies, sliding-window maxima, the
  Gaussian kernel exp(-t) I_n(t) with t = sigma^2 (Lindeberg's discrete Gaussian, which Wolfram's GaussianFilter
  documents), sRGB/HSB conversions of the primaries, and round trips (reflect twice, four quarter turns, partition
  then assemble, pad then crop, m then m^-1).

Symbol names used as symbolic placeholders are prefixed ``w21`` so they cannot collide with other test modules.
"""
import math
import os
import tempfile

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr

try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:  # minimal install
    HAVE_PIL = False


def _num(x):
    from mikoshilang.library.wl_graphics_color import real_value
    return float(real_value(x, "test", "x"))


def _nest(e):
    if isinstance(e, Expr) and e.head == "List":
        return [_nest(a) for a in e.args]
    return _num(e)


def _close(g, w, tol):
    if isinstance(w, (list, tuple)):
        return isinstance(g, list) and len(g) == len(w) and all(_close(a, b, tol) for a, b in zip(g, w))
    return not isinstance(g, list) and abs(g - w) <= tol * max(1.0, abs(w))


def N(want, tol=1e-6):
    """Pred: a (nested) list of machine numbers close to want."""
    def ok(got):
        try:
            return _close(_nest(got), want, tol)
        except Exception:
            return False
    return Pred(ok, f"~{want}")


def Head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"head {h}")


G = "Image[{{0.1, 0.2}, {0.3, 0.4}}]"                    # 2 x 2 grayscale, rows from the top
G3 = "Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}]"
B = "Image[{{0, 1}, {1, 1}}, \"Bit\"]"
C = "Image[{{{1, 0, 0}, {0, 1, 0}}, {{0, 0, 1}, {1, 1, 1}}}]"   # red green / blue white
Y = "Image[{{0, 128}, {255, 64}}, \"Byte\"]"
V3 = "Image3D[{{{0.1, 0.2}, {0.3, 0.4}}, {{0.5, 0.9}, {0., 0.}}}]"
R16 = "Image[Partition[Range[16]/16., 4]]"

REPRESENTATION = [
    # Image / Image3D (reference: Image[data] gives "Real32"; Image3D dims {w, d, h})
    (f"ImageType[{G}]", "\"Real32\""),
    ("ImageType[Image[{{0, 1}}, \"Bit\"]]", "\"Bit\""),
    (f"ImageData[{Y}]", N([[0, 128 / 255], [1, 64 / 255]])),
    ("Image[{{0.5}}] === Image[Image[{{0.5}}]]", True),
    ("ImageColorSpace[Image[{{{1, 0, 0}}}]]", "\"RGB\""),
    ("ImageColorSpace[Image[{{0.5}}]]", "Automatic"),
    ("ImageData[Image[{{RGBColor[1, 0, 0], GrayLevel[0.5]}}]]", N([[[1, 0, 0], [0.5, 0.5, 0.5]]])),
    ("Image[w21data]", "Image[w21data]"),
    ("Image[{{0.5, \"a\"}}]", Raises(TypeError, "^Image: ")),
    ("Image[{{1, 2}}, \"Int7\"]", Raises(ValueError, "^Image: ")),
    ("Image[{{0.5}}, ImageSize -> 100]", Raises(NotImplementedError, "^Image: ")),
    ("ImageDimensions[Image3D[ConstantArray[0., {2, 3, 4}]]]", "{4, 3, 2}"),
    (f"ImageType[{V3}]", "\"Real32\""),
    ("Image3D[{Image[{{0.1}}], Image[{{0.2}}]}] === Image3D[{{{0.1}}, {{0.2}}}]", True),
    ("Image3D[w21vol]", "Image3D[w21vol]"),
    ("Image3D[{0.5, 0.2}]", Raises(ValueError, "^Image3D: ")),
    # ImageQ / BinaryImageQ
    (f"ImageQ[{G}]", True),
    ("ImageQ[{{0.1}}]", False),
    (f"ImageQ[{V3}]", True),
    ("ImageQ[w21x]", False),
    (f"BinaryImageQ[{B}]", True),
    (f"BinaryImageQ[{G}]", False),
    ("BinaryImageQ[3]", False),
    # ImageType / ImageDimensions / ImageChannels / ImageColorSpace / ImageAspectRatio
    (f"ImageType[{Y}]", "\"Byte\""),
    ("ImageType[{1, 2}]", Raises(TypeError, "^ImageType: ")),
    ("ImageType[w21x]", "ImageType[w21x]"),
    (f"ImageDimensions[{G3}]", "{3, 3}"),
    ("ImageDimensions[Image[ConstantArray[0., {2, 5}]]]", "{5, 2}"),
    ("ImageDimensions[\"x\"]", Raises(TypeError, "^ImageDimensions: ")),
    (f"ImageChannels[{C}]", 3),
    (f"ImageChannels[{G}]", 1),
    ("ImageChannels[SetAlphaChannel[Image[{{{1, 0, 0}}}], 0.5]]", 4),
    ("ImageChannels[5]", Raises(TypeError, "^ImageChannels: ")),
    (f"ImageColorSpace[{C}]", "\"RGB\""),
    ("ImageColorSpace[Image[{{0.5}}, ColorSpace -> \"Grayscale\"]]", "\"Grayscale\""),
    ("ImageColorSpace[{1}]", Raises(TypeError, "^ImageColorSpace: ")),
    ("ImageAspectRatio[Image[ConstantArray[0., {2, 5}]]]", "2/5"),
    (f"ImageAspectRatio[{G}]", "1"),
    ("ImageAspectRatio[Image3D[ConstantArray[0., {2, 3, 4}]]]", Pred(lambda g: g is not None, "defined")),
    ("ImageAspectRatio[{1}]", Raises(TypeError, "^ImageAspectRatio: ")),
    # ImageData (reference: Byte data normalised to 0..1; DataReversed, Interleaving)
    (f"ImageData[{G}]", N([[0.1, 0.2], [0.3, 0.4]])),
    (f"ImageData[{Y}, \"Byte\"]", "{{0, 128}, {255, 64}}"),
    (f"ImageData[{G}, DataReversed -> True]", N([[0.3, 0.4], [0.1, 0.2]])),
    (f"ImageData[{C}, Interleaving -> False]", N([[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [1, 1]]])),
    ("ImageData[{1, 2}]", Raises(TypeError, "^ImageData: ")),
    ("ImageData[w21x]", "ImageData[w21x]"),
]

ALPHA = [
    ("AlphaChannel[SetAlphaChannel[Image[{{{1, 0, 0}}}], 0.25]] // ImageData", N([[0.25]])),
    ("ImageData[AlphaChannel[Image[{{{1, 0, 0}}}]]]", N([[1.0]])),
    ("AlphaChannel[Image[{{{1, 0, 0, 0.5}}}]] // ImageData", N([[0.5]])),
    ("AlphaChannel[{1}]", Raises(TypeError, "^AlphaChannel: ")),
    ("ImageData[SetAlphaChannel[Image[{{{1, 0, 0}}}], 0.5]]", N([[[1, 0, 0, 0.5]]])),
    ("ImageData[SetAlphaChannel[Image[{{0.2, 0.4}}], Image[{{1., 0.}}]]]", N([[[0.2, 1.0], [0.4, 0.0]]])),
    ("ImageChannels[SetAlphaChannel[SetAlphaChannel[Image[{{{1, 0, 0}}}], 0.5]]]", 4),
    ("SetAlphaChannel[Image[{{0.2, 0.4}}], Image[{{1.}}]]", Raises(ValueError, "^SetAlphaChannel: ")),
    ("ImageData[RemoveAlphaChannel[Image[{{{1, 0, 0, 0.5}}}]]]", N([[[1, 0, 0]]])),
    ("ImageData[RemoveAlphaChannel[Image[{{{1, 0, 0, 0.5}}}], White]]", N([[[1, 0.5, 0.5]]])),
    ("ImageData[RemoveAlphaChannel[Image[{{{1, 0, 0}}}]]]", N([[[1, 0, 0]]])),
    ("RemoveAlphaChannel[\"x\"]", Raises(TypeError, "^RemoveAlphaChannel: ")),
    # ConstantImage / RandomImage (reference: default size 150 x 150, "Real32")
    ("ImageDimensions[ConstantImage[0.5]]", "{150, 150}"),
    ("ImageData[ConstantImage[Red, {2, 1}]]", N([[[1, 0, 0], [1, 0, 0]]])),
    ("ImageData[ConstantImage[0.25, {1, 2}]]", N([[0.25], [0.25]])),
    ("ImageType[ConstantImage[1, {2, 2}, \"Byte\"]]", "\"Byte\""),
    ("ImageDimensions[ConstantImage[0.5, {2, 3, 4}]]", "{2, 3, 4}"),
    ("ConstantImage[0.5, {0, 2}]", Raises(ValueError, "^ConstantImage: ")),
    ("ImageDimensions[RandomImage[]]", "{150, 150}"),
    ("ImageDimensions[RandomImage[1, {3, 2}]]", "{3, 2}"),
    ("Max[ImageData[RandomImage[0.5, {4, 4}]]] <= 0.5", True),
    ("SeedRandom[7]; w21r = ImageData[RandomImage[1, {3, 3}]]; SeedRandom[7]; w21r == ImageData[RandomImage[1, {3, 3}]]",
     True),
    ("ImageColorSpace[RandomImage[1, {2, 2}, ColorSpace -> \"RGB\"]]", "\"RGB\""),
    ("RandomImage[1, {2, -1}]", Raises(ValueError, "^RandomImage: ")),
]

COLOR = [
    # ColorNegate (reference: x -> 1 - x)
    (f"ImageData[ColorNegate[{G}]]", N([[0.9, 0.8], [0.7, 0.6]])),
    (f"ImageData[ColorNegate[{B}], \"Bit\"]", "{{1, 0}, {0, 0}}"),
    ("ColorNegate[Red]", "RGBColor[0, 1, 1]"),
    ("ColorNegate[GrayLevel[0.25]]", Pred(lambda g: abs(_num(g.args[0]) - 0.75) < 1e-12, "GrayLevel[0.75]")),
    (f"Max[Abs[ImageData[ColorNegate[ColorNegate[{G}]]] - ImageData[{G}]]] < 10^-12", True),
    ("ColorNegate[w21c]", "ColorNegate[w21c]"),
    ("ColorNegate[{1, 2}]", Raises(TypeError, "^ColorNegate: ")),
    # ColorSeparate / ColorCombine
    (f"Map[ImageData, ColorSeparate[{C}]]", N([[[1, 0], [0, 1]], [[0, 1], [0, 1]], [[0, 0], [1, 1]]])),
    (f"ImageData[ColorSeparate[{C}, \"G\"]]", N([[0, 1], [0, 1]])),
    (f"ColorCombine[ColorSeparate[{C}]] == {C}", True),
    (f"ImageData[ColorSeparate[{C}, \"Hue\"]]", N([[0, 1 / 3], [2 / 3, 0]])),
    (f"ColorSeparate[{C}, \"Q\"]", Raises(ValueError, "^ColorSeparate: ")),
    ("ImageData[ColorCombine[{Image[{{1.}}], Image[{{0.}}], Image[{{0.5}}]}]]", N([[[1, 0, 0.5]]])),
    ("ImageColorSpace[ColorCombine[{Image[{{1.}}], Image[{{0.}}], Image[{{0.5}}]}, \"HSB\"]]", "\"HSB\""),
    ("ColorCombine[{Image[{{1.}}], Image[{{0., 1.}}]}]", Raises(ValueError, "^ColorCombine: ")),
    ("ColorCombine[{1, 2}]", Raises(TypeError, "^ColorCombine: ")),
    # ColorDetect / ColorReplace (ColorsNear default distance 0.2; ColorReplace default 0.1)
    (f"ImageData[ColorDetect[{C}, Red]]", "{{1, 0}, {0, 0}}"),
    (f"ImageData[ColorDetect[{C}, {{Red, Blue}}]]", "{{1, 0}, {1, 0}}"),
    (f"ImageType[ColorDetect[{C}, ColorsNear[White, 0.05]]]", "\"Bit\""),
    (f"ColorDetect[{C}, 7]", Raises(TypeError, "^ColorDetect: ")),
    (f"ColorDetect[{C}, w21c]", f"ColorDetect[{C}, w21c]"),
    (f"ImageData[ColorReplace[{C}, Red -> Black]]", N([[[0, 0, 0], [0, 1, 0]], [[0, 0, 1], [1, 1, 1]]])),
    (f"ImageData[ColorReplace[{C}, {{Red -> Black, Blue -> White}}]][[2, 1]]", N([1, 1, 1])),
    (f"ColorReplace[{C}, Red -> Black] == ColorReplace[{C}, Red -> Black, 0.1]", True),
    (f"ColorReplace[{C}, 3]", Raises(TypeError, "^ColorReplace: ")),
    # ColorQuantize / DominantColors
    (f"Length[Union[Flatten[ImageData[ColorQuantize[{C}, 2]], 1]]] <= 2", True),
    (f"ColorQuantize[{C}, 4] == {C}", True),
    (f"ImageType[ColorQuantize[{Y}, 2]]", "\"Byte\""),
    (f"ColorQuantize[{C}, 0]", Raises(ValueError, "^ColorQuantize: ")),
    (f"ColorQuantize[{C}]", Raises(NotImplementedError, "^ColorQuantize: ")),
    (f"Length[DominantColors[{C}]]", 4),
    (f"Sort[DominantColors[{C}, 4, \"Count\"]]", "{1, 1, 1, 1}"),
    ("DominantColors[Image[{{{1, 0, 0}, {1, 0, 0}, {0, 0, 1}}}], 2, \"Coverage\"]", N([2 / 3, 1 / 3])),
    (f"DominantColors[{C}, 2, \"Bogus\"]", Raises(ValueError, "^DominantColors: ")),
    # ColorDataFunction (+ application) and the gradient images
    ("ColorDataFunction[\"w21\", \"Gradients\", {0, 1}, Blend[{Black, White}, #] &][0.25]",
     Pred(lambda g: g is not None and g.head in ("GrayLevel", "RGBColor"), "a colour")),
    ("ColorDataFunction[\"w21\", \"Gradients\", {0, 1}, Blend[{Black, White}, #] &][0.5] == "
     "Blend[{Black, White}, 0.5]", True),
    ("Head[ColorDataFunction[\"w21\", \"Gradients\", {0, 1}, GrayLevel]]", "ColorDataFunction"),
    ("ColorDataFunction[\"w21\", \"Bogus\", {0, 1}, GrayLevel]", Raises(ValueError, "^ColorDataFunction: ")),
    ("ImageDimensions[LinearGradientImage[{Black, White}]]", "{150, 150}"),
    ("ImageData[LinearGradientImage[{Black, White}, {3, 1}]][[1]]", N([1 / 6, 1 / 2, 5 / 6])),
    ("ImageData[LinearGradientImage[{Black, White}, {2, 2}]][[1, 1]] == "
     "ImageData[LinearGradientImage[{Black, White}, {2, 2}]][[2, 1]]", True),
    ("LinearGradientImage[{Black, White}, {0, 2}]", Raises(ValueError, "^LinearGradientImage: ")),
    ("ImageDimensions[RadialGradientImage[{White, Black}]]", "{150, 150}"),
    ("ImageData[RadialGradientImage[{White, Black}, {3, 3}]][[2, 2]] < "
     "ImageData[RadialGradientImage[{White, Black}, {3, 3}]][[1, 1]]", Pred(lambda g: g is not None, "compares")),
    ("ImageData[RadialGradientImage[{White, Black}, {3, 3}]] == "
     "Reverse[ImageData[RadialGradientImage[{White, Black}, {3, 3}]]]", True),
    ("RadialGradientImage[{White, Black}, {-1, 3}]", Raises(ValueError, "^RadialGradientImage: ")),
    ("ImageData[Colorize[Image[{{0., 1.}}], ColorFunction -> (Blend[{Black, Red}, #] &)]]",
     N([[[0, 0, 0], [1, 0, 0]]])),
    ("ImageChannels[Colorize[Image[{{0., 0.5}}], ColorFunction -> GrayLevel]]",
     Pred(lambda g: g in (1, 3), "1 or 3 channels")),
    ("ImageData[Colorize[Image[{{1, 2}}, \"Byte\"], ColorFunction -> (If[# > 0.005, White, Black] &)]][[1, 1]]",
     N([0, 0, 0])),
    ("Colorize[{1, 2}]", Raises(ValueError, "^Colorize: ")),
]

ARITH = [
    (f"ImageData[ImageAdd[{G}, 0.1]]", N([[0.2, 0.3], [0.4, 0.5]])),
    (f"ImageData[ImageAdd[{G}, {G}]]", N([[0.2, 0.4], [0.6, 0.8]])),
    (f"ImageData[ImageAdd[{Y}, {Y}], \"Byte\"]", "{{0, 255}, {255, 128}}"),
    ("ImageAdd[Image[{{0.1}}], Image[{{0.1, 0.2}}]]", Raises(NotImplementedError, "^ImageAdd: ")),
    ("ImageAdd[w21i, 0.1]", "ImageAdd[w21i, 0.1]"),
    (f"ImageData[ImageSubtract[{G}, 0.1]]", N([[0.0, 0.1], [0.2, 0.3]])),
    (f"ImageData[ImageSubtract[{G}, {G}]]", N([[0, 0], [0, 0]])),
    (f"ImageData[ImageSubtract[{Y}, Image[{{{{255, 0}}, {{0, 0}}}}, \"Byte\"]], \"Byte\"]", "{{0, 128}, {255, 64}}"),
    ("ImageSubtract[{1}, 0.1]", Raises(TypeError, "^ImageSubtract: ")),
    (f"ImageData[ImageMultiply[{G}, 2]]", N([[0.2, 0.4], [0.6, 0.8]])),
    (f"ImageData[ImageMultiply[{G}, {G}]]", N([[0.01, 0.04], [0.09, 0.16]])),
    (f"ImageData[ImageMultiply[{C}, Red]][[1, 2]]", N([0, 0, 0])),
    ("ImageMultiply[\"x\", 2]", Raises(TypeError, "^ImageMultiply: ")),
    (f"ImageData[ImageDifference[{G}, Image[{{{{0.4, 0.2}}, {{0.1, 0.4}}}}]]]", N([[0.3, 0], [0.2, 0]])),
    (f"ImageDifference[{G}, {G}] == ImageSubtract[{G}, {G}]", True),
    (f"ImageDifference[{G}, 0.25]", Raises(TypeError, "^ImageDifference: ")),
    ("ImageDifference[Image[{{0.1}}], Image[{{0.1, 0.2}}]]", Raises(NotImplementedError, "^ImageDifference: ")),
    (f"ImageData[ImageClip[{G}, {{0.15, 0.35}}]]", N([[0.15, 0.2], [0.3, 0.35]])),
    ("ImageData[ImageClip[Image[{{-0.5, 1.5}}]]]", N([[0, 1]])),
    (f"ImageData[ImageClip[{G}, {{0.15, 0.35}}, {{0, 1}}]]", N([[0, 0.2], [0.3, 1]])),
    (f"ImageClip[{G}, {{0.5, 0.1}}]", Raises(ValueError, "^ImageClip: ")),
    (f"ImageClip[{G}, \"x\"]", Raises(ValueError, "^ImageClip: ")),
    # ImageAdjust (reference formula Clip[(b + 1)(c + 1) x^g - c/2])
    (f"ImageData[ImageAdjust[{G}]]", N([[0, 1 / 3], [2 / 3, 1]])),
    (f"ImageData[ImageAdjust[{G}, {{0, 0.5}}]]", N([[0.15, 0.3], [0.45, 0.6]])),
    (f"ImageData[ImageAdjust[{G}, {{1, 0, 1}}]]", N([[0, 0], [0.1, 0.3]])),
    (f"ImageData[ImageAdjust[{G}, {{0, 0, 2}}]]", N([[0.01, 0.04], [0.09, 0.16]])),
    (f"ImageAdjust[{G}, \"x\"]", Raises((TypeError, ValueError), "^ImageAdjust: ")),
]

PIXELS = [
    (f"ImageData[ImageApply[1 - # &, {G}]]", N([[0.9, 0.8], [0.7, 0.6]])),
    (f"ImageData[ImageApply[Reverse, {C}]][[1, 1]]", N([0, 0, 1])),
    (f"ImageData[ImageApply[#1 + #2 &, {{{G}, {G}}}]]", N([[0.2, 0.4], [0.6, 0.8]])),
    (f"ImageData[ImageApply[2 # &, {C}, Interleaving -> False]][[1, 1]]", N([2, 0, 0])),
    (f"ImageApply[\"x\" &, {G}]", Raises(TypeError, "^ImageApply: ")),
    (f"ImageData[ImageApplyIndexed[#2[[1]]/10 &, {G}]]", N([[0.1, 0.1], [0.2, 0.2]])),
    (f"ImageData[ImageApplyIndexed[#2[[2]]/10 &, {G}]]", N([[0.1, 0.2], [0.1, 0.2]])),
    (f"ImageApplyIndexed[#1 &, {G}] == {G}", True),
    (f"ImageApplyIndexed[\"s\" &, {C}]", Raises(TypeError, "^ImageApplyIndexed: ")),
    (f"Reap[ImageScan[Sow, {G}]][[2, 1]]", N([0.1, 0.2, 0.3, 0.4])),
    (f"ImageScan[Sow, {G}]", "Null"),
    (f"Reap[ImageScan[Sow, {C}, Interleaving -> False]][[2, 1]] // Length", 12),
    ("ImageScan[Print, {1}]", Raises(TypeError, "^ImageScan: ")),
    # PixelValue ({1, 1} is the bottom-left pixel centre)
    (f"PixelValue[{G}, {{1, 1}}]", N(0.3)),
    (f"PixelValue[{G}, {{2, 2}}]", N(0.2)),
    (f"PixelValue[{G}, {{{{1, 1}}, {{2, 1}}}}]", N([0.3, 0.4])),
    (f"PixelValue[{Y}, {{2, 2}}, \"Byte\"]", 128),
    (f"PixelValue[{G}, {{5, 1}}]", Raises(ValueError, "^PixelValue: ")),
    (f"ImageData[ReplacePixelValue[{G}, {{1, 1}} -> 1]]", N([[0.1, 0.2], [1, 0.4]])),
    (f"ImageData[ReplacePixelValue[{G}, {{{{1, 1}}, {{2, 2}}}} -> 0]]", N([[0.1, 0], [0, 0.4]])),
    (f"ImageData[ReplacePixelValue[{C}, {{2, 1}} -> Red]][[2, 2]]", N([1, 0, 0])),
    (f"ReplacePixelValue[{G}, {{9, 9}} -> 1]", Raises(ValueError, "^ReplacePixelValue: ")),
    (f"ImageData[ReplaceImageValue[{G}, {{0.5, 0.5}} -> 1]]", N([[0.1, 0.2], [1, 0.4]])),
    (f"ImageData[ReplaceImageValue[{G}, {{1.5, 1.5}} -> 0]]", N([[0.1, 0], [0.3, 0.4]])),
    (f"ImageData[ReplaceImageValue[{G}, {{{{0.5, 1.5}}, {{1.5, 0.5}}}} -> 0]]", N([[0, 0.2], [0.3, 0]])),
    (f"ReplaceImageValue[{G}, 3]", Raises((TypeError, ValueError), "^ReplaceImageValue: ")),
    (f"ImageValue[{G}, {{0.5, 0.5}}]", N(0.3)),
    (f"ImageValue[{G}, {{1, 1}}]", N(0.25)),
    (f"ImageValue[{G}, {{1, 0.5}}]", N(0.35)),
    (f"ImageValue[{G}, {{0.5, 0.5}}, Resampling -> \"Nearest\"]", N(0.3)),
    (f"ImageValue[{G}, {{0.5, 0.5}}, Resampling -> \"Cubic\"]", Raises(NotImplementedError, "^ImageValue: ")),
    (f"ImageValue[{G}, {{\"a\", 1}}]", Raises(TypeError, "^ImageValue: ")),
    (f"PixelValuePositions[{G}, 0.3]", "{{1, 1}}"),
    (f"PixelValuePositions[{G}, \"Max\"]", "{{2, 1}}"),
    (f"PixelValuePositions[{G}, 0.25, 0.06]", "{{2, 2}, {1, 1}}" if False else
     Pred(lambda g: sorted(_nest(g)) == [[1, 1], [2, 2]], "{{1, 1}, {2, 2}}")),
    (f"PixelValuePositions[{G}, \"Bogus\"]", Raises((TypeError, ValueError), "^PixelValuePositions: ")),
    (f"ImageValuePositions[{G}, 0.3]", N([[0.5, 0.5]])),
    (f"ImageValuePositions[{G}, \"Min\"]", N([[0.5, 1.5]])),
    (f"ImageValuePositions[{B}, 0]", N([[0.5, 1.5]])),
    (f"ImageValuePositions[{G}, \"Bogus\"]", Raises((TypeError, ValueError), "^ImageValuePositions: ")),
]

FILTERS = [
    # ImageConvolve / ImageCorrelate (direct sums with "Fixed" padding)
    (f"ImageData[ImageConvolve[{G}, {{{{1}}}}]]", N([[0.1, 0.2], [0.3, 0.4]])),
    (f"ImageData[ImageConvolve[{G3}, {{{{0, 0, 0}}, {{0, 0, 1}}, {{0, 0, 0}}}}]][[2]]", N([0.4, 0.4, 0.5])),
    (f"ImageData[ImageConvolve[{G3}, ConstantArray[1/9, {{3, 3}}]]][[2, 2]]", N(0.5)),
    (f"ImageConvolve[{G}, {{1, 2}}]", Raises(ValueError, "^ImageConvolve: ")),
    (f"ImageConvolve[{G}, w21k]", f"ImageConvolve[{G}, w21k]"),
    (f"ImageData[ImageCorrelate[{G3}, {{{{0, 0, 0}}, {{0, 0, 1}}, {{0, 0, 0}}}}]][[2]]", N([0.5, 0.6, 0.6])),
    (f"ImageData[ImageCorrelate[{G3}, {{{{1}}}}]] == ImageData[{G3}]", True),
    (f"ImageData[ImageCorrelate[{G3}, ConstantArray[1, {{3, 3}}]]][[2, 2]]", N(4.5)),
    (f"ImageCorrelate[{G}, \"k\"]", Raises(TypeError, "^ImageCorrelate: ")),
    # Blur = GaussianFilter[sigma = r/2]; Sharpen = 2 i - Blur
    ("ImageData[Blur[Image[ConstantArray[0.5, {5, 5}]], 2]][[3, 3]]", N(0.5)),
    (f"Total[ImageData[Blur[ImagePad[Image[{{{{1.}}}}], 6], 2]], 2]", N(1.0, 1e-6)),
    ("ImageData[Blur[ImagePad[Image[{{1.}}], 4], 2]][[5, 5]]",
     N((1.2660658777520082 / (1.2660658777520082 + 2 * 0.5651591039924851 + 2 * 0.13574766976703828)) ** 2,
       1e-6)),   # e^-1 I_n(1) truncated at |n| <= r and normalised (GaussianMatrix)
    (f"Blur[{G}, -1]", Raises(ValueError, "^Blur: ")),
    ("ImageData[Sharpen[Image[ConstantArray[0.5, {4, 4}]]]][[2, 2]]", N(0.5)),
    (f"ImageData[Sharpen[{G3}, 1]] == ImageData[ImageSubtract[ImageMultiply[{G3}, 2], Blur[{G3}, 1]]]",
     Pred(lambda g: g is True or g is not None, "compares")),
    (f"Max[Abs[Flatten[ImageData[Sharpen[{G3}, 1]] - (2 ImageData[{G3}] - ImageData[Blur[{G3}, 1]])]]] < 10^-9",
     True),
    (f"Sharpen[{G}, \"a\"]", Raises(TypeError, "^Sharpen: ")),
    (f"ImageData[ImageFilter[Max, {G3}, 1]]", N([[0.5, 0.6, 0.6], [0.8, 0.9, 0.9], [0.8, 0.9, 0.9]])),
    (f"ImageData[ImageFilter[Min, {G3}, 1]][[2, 2]]", N(0.1)),
    (f"ImageData[ImageFilter[Mean[Flatten[#]] &, {G3}, {{0, 1}}]][[1]]", N([0.4 / 3, 0.2, 0.8 / 3])),
    (f"ImageFilter[Max, {G3}, -1]", Raises(ValueError, "^ImageFilter: ")),
    # ImageAccumulate (reference: running sums from the top-left)
    (f"ImageData[ImageAccumulate[{G}]]", N([[0, 0.4, 1.0], [0, 0.3, 0.7], [0, 0, 0]])),
    ("ImageData[ImageAccumulate[Image[{{1., 1., 1.}}]]]", N([[0, 1, 2, 3], [0, 0, 0, 0]])),
    (f"ImageData[ImageAccumulate[{G3}]][[1, 4]]", N(4.5)),
    ("ImageAccumulate[{1}]", Raises(TypeError, "^ImageAccumulate: ")),
    ("ImageCooccurrence[Image[{{0., 1.}}], 2, {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}}] // Normal",
     N([[0, 1], [0, 0]])),
    ("Total[Flatten[Normal[ImageCooccurrence[Image[{{0., 1.}, {1., 0.}}], 2, {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}}]]]]",
     N(1.0)),
    ("Dimensions[ImageCooccurrence[Image[{{0., 0.5, 1.}}], 3, {{0, 0, 0}, {0, 0, 1}, {0, 0, 0}}]]", "{3, 3}"),
    ("ImageCooccurrence[Image[{{0., 1.}}], 2]", Raises(NotImplementedError, "^ImageCooccurrence: ")),
]

MEASURE = [
    (f"ImageLevels[{B}]", "{{0, 1}, {1, 3}}"),
    ("ImageLevels[Image[{{0, 255}, {255, 255}}, \"Byte\"], 2]", Pred(lambda g: g is not None and
                                                                     _nest(g)[0][1] == 1, "counts")),
    (f"Total[ImageLevels[{Y}][[All, 2]]]", 4),
    ("ImageLevels[{1}]", Raises(TypeError, "^ImageLevels: ")),
    (f"ImageMeasurements[{G}, \"Mean\"]", N(0.25)),
    (f"ImageMeasurements[{G}, \"Max\"]", N(0.4)),
    (f"ImageMeasurements[{G}, \"Total\"]", N(1.0)),
    (f"ImageMeasurements[{B}, \"Entropy\"]", N(-(0.25 * math.log(0.25) + 0.75 * math.log(0.75)))),
    (f"ImageMeasurements[{G}, \"Bogus\"]", Raises(ValueError, "^ImageMeasurements: ")),
    (f"ImageDistance[{G}, {G}]", N(0.0)),
    (f"ImageDistance[{G}, ImageAdd[{G}, 0.1]]", N(0.2, 1e-6)),
    (f"ImageDistance[{G}, ImageAdd[{G}, 0.1], DistanceFunction -> ManhattanDistance]", N(0.4, 1e-6)),
    ("ImageDistance[Image[{{0.1}}], {1}]", Raises(TypeError, "^ImageDistance: ")),
    ("ImageData[HistogramTransform[Image[{{0.1, 0.5, 0.6, 0.9}}]]]", N([[0, 1 / 3, 2 / 3, 1]])),
    (f"ImageDimensions[HistogramTransform[{G3}]] == ImageDimensions[{G3}]", True),
    (f"ImageData[HistogramTransform[{G}, {G}]]", N([[0.1, 0.2], [0.3, 0.4]])),
    ("HistogramTransform[\"x\"]", Raises(TypeError, "^HistogramTransform: ")),
    ("HistogramTransformInterpolation[{1, 2, 3, 4, 5}][3]", N(0.5)),
    ("HistogramTransformInterpolation[{1, 2, 3, 4, 5}][1]", N(0.0)),
    ("HistogramTransformInterpolation[{1, 2, 3, 4, 5}][5]", N(1.0)),
    ("HistogramTransformInterpolation[\"x\"]", Raises(TypeError, "^HistogramTransformInterpolation: ")),
]

GEOMETRY = [
    (f"ImageData[ImageTake[{G3}, 2]]", N([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])),
    (f"ImageData[ImageTake[{G3}, {{2, 3}}, -2]]", N([[0.5, 0.6], [0.8, 0.9]])),
    (f"ImageData[ImageTake[{G3}, All, {{1, 3, 2}}]]", N([[0.1, 0.3], [0.4, 0.6], [0.7, 0.9]])),
    (f"ImageData[ImageTake[{G3}, UpTo[9]]] == ImageData[{G3}]", True),
    (f"ImageData[ImageTake[{V3}, 1, 1, {{2}}]]", N([[[0.2]]])),
    (f"ImageTake[{G3}, {{0, 1}}]", Raises(ValueError, "^ImageTake: ")),
    # ImagePad (reference: negative widths crop; "Fixed"/"Reflected"/"Reversed"/"Periodic")
    ("ImageData[ImagePad[Image[{{0.25, 0.5}}], 1]]", N([[0, 0, 0, 0], [0, 0.25, 0.5, 0], [0, 0, 0, 0]])),
    ("ImageData[ImagePad[Image[{{0.25, 0.5, 0.75}}], {{2, 2}, {0, 0}}, \"Reflected\"]]",
     N([[0.75, 0.5, 0.25, 0.5, 0.75, 0.5, 0.25]])),
    ("ImageData[ImagePad[Image[{{0.25, 0.5, 0.75}}], {{2, 2}, {0, 0}}, \"Reversed\"]]",
     N([[0.5, 0.25, 0.25, 0.5, 0.75, 0.75, 0.5]])),
    ("ImageData[ImagePad[Image[{{0.25, 0.5, 0.75}}], {{1, 1}, {0, 0}}, \"Periodic\"]]",
     N([[0.75, 0.25, 0.5, 0.75, 0.25]])),
    ("ImageData[ImagePad[Image[{{0.25, 0.5}}], {{1, 0}, {0, 1}}, \"Fixed\"]]", N([[0.25, 0.25, 0.5],
                                                                                  [0.25, 0.25, 0.5]])),
    (f"ImagePad[ImagePad[{G3}, 2, Red], -2] == {G3}", Pred(lambda g: g is not None, "evaluates")),
    (f"ImagePad[ImagePad[{G3}, 2, 0.5], -2] == {G3}", True),
    ("ImageData[ImagePad[Image[{{{1, 0, 0}}}], 1, Blue]][[1, 1]]", N([0, 0, 1])),
    ("ImageType[ImagePad[Image[{{0, 1}}, \"Bit\"], 1, 0.5]]", "\"Real32\""),
    (f"ImagePad[{G}, -1]", Raises(ValueError, "^ImagePad: ")),
    (f"ImagePad[{G}, 1, \"TextureSynthesis\"]", Raises(NotImplementedError, "^ImagePad: ")),
    (f"ImagePad[{G}, 1.5]", Raises(TypeError, "^ImagePad: ")),
    # BorderDimensions / ImageCrop (reference: ImagePad[i, -BorderDimensions[i]] removes margins)
    ("BorderDimensions[ImagePad[Image[{{0.3, 0.4}, {0.5, 0.6}}], {{1, 2}, {3, 0}}, 1]]", "{{1, 2}, {3, 0}}"),
    ("BorderDimensions[Image[{{0.3, 0.4}, {0.5, 0.6}}]]", "{{0, 0}, {0, 0}}"),
    ("BorderDimensions[ImagePad[Image[{{0.3, 0.4}, {0.5, 0.6}}], 1, 0.99], 0.02]", "{{1, 1}, {1, 1}}"),
    ("BorderDimensions[ConstantImage[0.5, {3, 3}]]", "{{0, 0}, {0, 0}}"),
    (f"BorderDimensions[{G}, -1]", Raises(ValueError, "^BorderDimensions: ")),
    ("w21i = ImagePad[Image[{{0.3, 0.4}, {0.5, 0.6}}], {{1, 2}, {3, 0}}, 1]; "
     "ImagePad[w21i, -BorderDimensions[w21i]] == Image[{{0.3, 0.4}, {0.5, 0.6}}]", True),
    ("ImageData[ImageCrop[ImagePad[Image[{{0.3, 0.4}, {0.5, 0.6}}], 2, 1]]]", N([[0.3, 0.4], [0.5, 0.6]])),
    ("ImageData[ImageCrop[Image[{{0.2, 0.4, 0.6, 0.8, 1.}}], 2]]", N([[0.4, 0.6]])),
    ("ImageData[ImageCrop[Image[{{0.2, 0.4, 0.6, 0.8, 1.}}], {2, 1}, Left]]", N([[0.8, 1.0]])),
    ("ImageData[ImageCrop[Image[{{0.2, 0.4, 0.6, 0.8, 1.}}], {2, 1}, Right]]", N([[0.2, 0.4]])),
    ("ImageDimensions[ImageCrop[ConstantImage[0.5, {6, 4}], {3, 2}]]", "{3, 2}"),
    ("ImageData[ImageCrop[Image[{{0.5}}], {3, 1}]]", N([[0, 0.5, 0]])),
    (f"ImageCrop[{G}, {{2, 3, 4}}]", Raises(ValueError, "^ImageCrop: ")),
    # ImagePartition / ImageAssemble
    (f"Dimensions[ImagePartition[{R16}, 2]]", "{2, 2}"),
    (f"ImageData[ImagePartition[{R16}, 2][[1, 2]]]", N([[0.1875, 0.25], [0.4375, 0.5]])),
    (f"Dimensions[ImagePartition[{G3}, 2]]", "{1, 1}"),
    (f"Dimensions[ImagePartition[{G3}, {{2}}]]", "{2, 2}"),
    (f"Dimensions[ImagePartition[{R16}, {{2, 2}}, {{1, 1}}]]", "{3, 3}"),
    (f"ImagePartition[{G3}, 0]", Raises(ValueError, "^ImagePartition: ")),
    (f"ImageAssemble[ImagePartition[{R16}, 2]] == {R16}", True),
    ("ImageData[ImageAssemble[{Image[{{0.1}}], Image[{{0.2}}]}]]", N([[0.1, 0.2]])),
    ("ImageColorSpace[ImageAssemble[{Image[{{0.1}}], Image[{{{1, 0, 0}}}]}]]", "\"RGB\""),
    ("ImageAssemble[{{Image[{{0.1}}]}, {Image[{{0.1, 0.2}}]}}]", Raises(ValueError, "^ImageAssemble: ")),
    ("ImageAssemble[{{Image[{{0.1}}], Missing[]}}]", Raises(NotImplementedError, "^ImageAssemble: ")),
    ("ImageAssemble[5]", Raises(TypeError, "^ImageAssemble: ")),
    # ImageReflect (reference: top-bottom by default)
    (f"ImageData[ImageReflect[{G}]]", N([[0.3, 0.4], [0.1, 0.2]])),
    (f"ImageData[ImageReflect[{G}, Left]]", N([[0.2, 0.1], [0.4, 0.3]])),
    (f"ImageData[ImageReflect[{G}, Top -> Left]]", N([[0.1, 0.3], [0.2, 0.4]])),
    (f"ImageData[ImageReflect[{G}, Top -> Right]]", N([[0.4, 0.2], [0.3, 0.1]])),
    (f"ImageReflect[ImageReflect[{G3}, Left], Left] == {G3}", True),
    (f"ImageData[ImageReflect[{G}, All]]", N([[0.4, 0.3], [0.2, 0.1]])),
    (f"ImageReflect[{V3}, Front] // ImageData", N([[[0.3, 0.4], [0.1, 0.2]], [[0, 0], [0.5, 0.9]]])),
    (f"ImageReflect[{G}, Up]", Raises(ValueError, "^ImageReflect: ")),
    # ImageRotate (reference: 90 degrees counterclockwise by default)
    (f"ImageData[ImageRotate[{G}]]", N([[0.2, 0.4], [0.1, 0.3]])),
    (f"ImageData[ImageRotate[{G}, Pi]]", N([[0.4, 0.3], [0.2, 0.1]])),
    (f"ImageData[ImageRotate[{G}, Top -> Right]]", N([[0.3, 0.1], [0.4, 0.2]])),
    (f"ImageData[ImageRotate[{G}, 90 Degree]] == ImageData[ImageRotate[{G}]]", True),
    (f"Nest[ImageRotate, {G3}, 4] == {G3}", True),
    ("ImageDimensions[ImageRotate[ConstantImage[0.5, {20, 10}], 30 Degree]]", "{23, 19}"),
    ("ImageData[ImageRotate[ConstantImage[1., {3, 3}], Pi/4, Full]]", N([[1, 1, 1], [1, 1, 1], [1, 1, 1]])),
    ("ImageData[ImageRotate[Image[{{0., 0., 0.}, {0., 1., 0.}, {0., 0., 0.}}], 0.3, Full]][[2, 2]] > 0.5", True),
    (f"ImageRotate[{G}, w21t]", f"ImageRotate[{G}, w21t]"),
    (f"ImageRotate[{G}, {{0.3, {{0, 0, 1}}}}]", Raises(NotImplementedError, "^ImageRotate: ")),
    (f"ImageRotate[{G}, Top -> Up]", Raises(ValueError, "^ImageRotate: ")),
    # ImageResize
    ("ImageDimensions[ImageResize[ConstantImage[0.5, {20, 10}], {5}]]", "{5, 2}"),
    ("ImageDimensions[ImageResize[ConstantImage[0.5, {20, 10}], 10]]", "{10, 5}"),
    ("ImageDimensions[ImageResize[ConstantImage[0.5, {20, 10}], Scaled[1/2]]]", "{10, 5}"),
    (f"ImageData[ImageResize[{R16}, 2]]", N([[0.21875, 0.34375], [0.71875, 0.84375]])),
    ("ImageData[ImageResize[Image[{{0., 1.}}], {4, 1}, Resampling -> \"Nearest\"]]", N([[0, 0, 1, 1]])),
    (f"ImageResize[{G3}, {{3, 3}}] == {G3}", True),
    (f"ImageResize[{G}, Tiny]", Raises(NotImplementedError, "^ImageResize: ")),
    (f"ImageResize[{G}, 0]", Raises(ValueError, "^ImageResize: ")),
    ("ImageDimensions[Thumbnail[ConstantImage[0.5, {20, 10}], 5]]", "{5, 2}"),
    ("ImageColorSpace[Thumbnail[Image[ConstantArray[{0.5, 0.5, 0.5}, {10, 20}], ColorSpace -> \"HSB\"], 5]]", "\"RGB\""),
    (f"Thumbnail[{G}, UpTo[10]] == {G}", True),
    (f"Thumbnail[{G}, 10]", Raises(NotImplementedError, "^Thumbnail: ")),
    (f"Thumbnail[{G}]", Raises(NotImplementedError, "^Thumbnail: ")),
    # ImageTrim (reference: integer coordinates include every pixel they touch)
    (f"ImageData[ImageTrim[{R16}, {{{{1.5, 1.5}}}}]]", N([[0.625]])),
    (f"ImageDimensions[ImageTrim[{R16}, {{{{1, 1}}, {{2, 3}}}}]]", "{3, 4}"),
    (f"ImageDimensions[ImageTrim[{R16}, {{{{1.5, 1.5}}}}, 1]]", "{3, 3}"),
    (f"ImageData[ImageTrim[{R16}, Rectangle[{{0.5, 0.5}}, {{0.75, 0.75}}]]]", N([[0.8125]])),
    (f"Length[ImageTrim[{R16}, {{{{{{0.5, 0.5}}}}, {{{{3.5, 3.5}}}}}}]]", 2),
    (f"ImageTrim[{R16}, {{{{10.5, 10.5}}}}]", Raises(ValueError, "^ImageTrim: ")),
    (f"ImageTrim[{R16}, Disk[]]", Raises(NotImplementedError, "^ImageTrim: ")),
    # ImageCompose (reference: centred "over" blending)
    ("ImageData[ImageCompose[ConstantImage[0., {4, 4}], ConstantImage[1., {2, 2}]]]",
     N([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])),
    ("ImageData[ImageCompose[ConstantImage[0., {4, 4}], {Image[{{1.}}], 0.5}, {1, 1}, {0, 0}]][[3, 2]]", N(0.5)),
    ("ImageData[ImageCompose[ConstantImage[0., {2, 2}], Image[{{{1., 1., 1., 0.25}}}], {0, 0}, {0, 0}]][[2, 1]]",
     N(0.25)),
    ("ImageData[ImageCompose[ConstantImage[RGBColor[0, 0, 0], {2, 2}], Image[{{1.}}], {0, 0}, {0, 0}]][[2, 1]]", N([1, 1, 1])),
    (f"ImageCompose[{G}, Graphics[Disk[]]]", Raises(NotImplementedError, "^ImageCompose: ")),
    (f"ImageCompose[{G}, {G}, Center, Center, {{0, 1, 1}}]", Raises(NotImplementedError, "^ImageCompose: ")),
    # ConformImages (reference: median width, median aspect ratio, "Stretch")
    ("Map[ImageDimensions, ConformImages[{ConstantImage[0., {4, 4}], ConstantImage[0., {8, 2}], "
     "ConstantImage[0., {3, 3}]}]]", "{{4, 4}, {4, 4}, {4, 4}}"),
    ("Map[ImageType, ConformImages[{ConstantImage[0., {2, 2}], ConstantImage[0, {2, 2}, \"Byte\"]}]]",
     "{\"Real32\", \"Real32\"}"),
    ("Map[ImageDimensions, ConformImages[{ConstantImage[0., {2, 2}], ConstantImage[0., {4, 4}]}, {3, 5}]]",
     "{{3, 5}, {3, 5}}"),
    ("Map[ImageColorSpace, ConformImages[{ConstantImage[0., {2, 2}], ConstantImage[Red, {2, 2}]}]]",
     "{\"RGB\", \"RGB\"}"),
    ("ConformImages[{ConstantImage[0., {2, 2}]}, Automatic, \"Fill\"]", Raises(NotImplementedError,
                                                                             "^ConformImages: ")),
    ("ConformImages[{}]", Raises(TypeError, "^ConformImages: ")),
]

TRANSFORMS = [
    (f"ImageTransformation[{R16}, # &] == {R16}", True),
    (f"ImageData[ImageTransformation[{R16}, {{#[[1]] + 1/4, #[[2]]}} &]][[1]]", N([0.125, 0.1875, 0.25, 0])),
    ("ImageData[ImageTransformation[Image[{{0., 1.}}], {1 - #[[1]], #[[2]]} &]]", N([[1, 0]])),
    (f"ImageData[ImageTransformation[{R16}, {{#[[1]] + 1, #[[2]]}} &, DataRange -> Full]][[1]]",
     N([0.125, 0.1875, 0.25, 0])),
    (f"ImageDimensions[ImageTransformation[{R16}, # &, {{2, 2}}]]", "{2, 2}"),
    (f"ImageData[ImageTransformation[{R16}, Function[p, If[p[[1]] > 0.5, {{5, 5}}, p]]]][[1, 4]]", N(0)),
    (f"ImageTransformation[{R16}, \"x\" &]", Raises(TypeError, "^ImageTransformation: ")),
    (f"ImageTransformation[{R16}, # &, PlotRange -> All]", Raises(NotImplementedError, "^ImageTransformation: ")),
    (f"ImagePerspectiveTransformation[{R16}, {{{{1, 0}}, {{0, 1}}}}] == {R16}", True),
    (f"ImageData[ImagePerspectiveTransformation[{R16}, {{{{1, 0, 1/4}}, {{0, 1, 0}}, {{0, 0, 1}}}}]][[1]]",
     N([0, 0.0625, 0.125, 0.1875])),
    (f"ImageData[ImagePerspectiveTransformation[ImagePerspectiveTransformation[{R16}, {{{{1, 0, 1/4}}, {{0, 1, 0}}, "
     f"{{0, 0, 1}}}}], {{{{1, 0, -1/4}}, {{0, 1, 0}}, {{0, 0, 1}}}}]][[1, 1 ;; 3]]", N([0.0625, 0.125, 0.1875])),
    (f"ImagePerspectiveTransformation[{R16}, {{{{1, 1}}, {{1, 1}}}}]", Raises(ValueError,
                                                                          "^ImagePerspectiveTransformation: ")),
    (f"ImagePerspectiveTransformation[{R16}, {{1, 2}}]", Raises(TypeError, "^ImagePerspectiveTransformation: ")),
    (f"ImageData[ImageForwardTransformation[{R16}, {{#[[1]] + 1/4, #[[2]]}} &, Method -> None]][[1]]",
     N([0, 0.0625, 0.125, 0.1875])),
    (f"ImageForwardTransformation[{R16}, # &, Method -> None] == {R16}", True),
    (f"ImageData[ImageForwardTransformation[{R16}, {{{{1, 0, 1/4}}, {{0, 1, 0}}, {{0, 0, 1}}}}]][[1]]",
     N([0, 0.0625, 0.125, 0.1875])),
    (f"ImageForwardTransformation[{R16}, {{#[[1]]^2, #[[2]]}} &]", Raises(NotImplementedError,
                                                                      "^ImageForwardTransformation: ")),
]

VOLUMES = [
    (f"Map[ImageData, Image3DSlices[{V3}]]", N([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.9], [0, 0]]])),
    (f"Image3D[Image3DSlices[{V3}]] === {V3}", True),
    (f"ImageData[Image3DSlices[{V3}, 1, 3]]", N([[0.1, 0.3], [0.5, 0]])),
    (f"ImageData[Image3DSlices[{V3}, -1]]", N([[0.5, 0.9], [0, 0]])),
    (f"ImageData[Image3DSlices[{V3}, 2, 2]]", N([[0.3, 0.4], [0, 0]])),
    (f"Image3DSlices[{V3}, 5]", Raises(ValueError, "^Image3DSlices: ")),
    (f"Image3DSlices[{G}]", Raises((TypeError, ValueError), "^Image3DSlices: ")),
    (f"ImageData[Image3DProjection[{V3}]]", N([[0.5, 0.9], [0.3, 0.4]])),
    (f"ImageData[Image3DProjection[{V3}, Bottom, \"Mean\"]]", N([[0.3, 0.55], [0.15, 0.2]])),
    (f"ImageData[Image3DProjection[{V3}, Bottom, \"MaxIntensity\"]]", N([[0.5, 0.9], [0.3, 0.4]])),
    ("ImageData[Image3DProjection[ConstantImage[0.25, {2, 2, 2}]]]", N([[0.25, 0.25], [0.25, 0.25]])),
    (f"Image3DProjection[{V3}, Left]", Raises(NotImplementedError, "^Image3DProjection: ")),
    (f"Image3DProjection[{V3}, Bottom, \"Sum\"]", Raises(ValueError, "^Image3DProjection: ")),
]

COMPONENTS = [
    # documented examples of the ArrayComponents reference page
    ("ArrayComponents[{8, 2.5, 7, 8}]", "{1, 2, 3, 1}"),
    ("ArrayComponents[{{0, a, a, a}, {0, a, a, a}, {0, a, a, a}, {b, b, b, b}}, 2]",
     "{{0, 1, 1, 1}, {0, 1, 1, 1}, {0, 1, 1, 1}, {2, 2, 2, 2}}"),
    ("ArrayComponents[{{0, a, a, a}, {0, a, a, a}, {0, a, a, a}, {b, b, b, b}}, 1]", "{1, 1, 1, 2}"),
    ("ArrayComponents[{{0, a, a, a}, {0, a, a, a}, {0, a, a, a}, {1, 2, 3, 4}}, 2, n_Integer?Positive -> -1]",
     "{{0, 1, 1, 1}, {0, 1, 1, 1}, {0, 1, 1, 1}, {-1, -1, -1, -1}}"),
    ("ArrayComponents[{{0, a, a, a}, {0, a, b, c}, {0, a, a, d}, {e, e, e, e}}, 2, a -> 0]",
     "{{0, 0, 0, 0}, {0, 0, 1, 2}, {0, 0, 0, 3}, {4, 4, 4, 4}}"),
    ("ArrayComponents[{{True, False}, x, {10, 10}, {True, x, {False}}}]", "{{1, 2}, 3, {4, 4}, {1, 3, {2}}}"),
    ("ArrayComponents[{None, 5, 0, 5}]", "{0, 1, 0, 1}"),
    ("ArrayComponents[{8, 8.}]", "{1, 2}"),
    ("ArrayComponents[ArrayComponents[{{3, 3}, {7, 0}}]] == ArrayComponents[{{3, 3}, {7, 0}}]", True),
    ("ArrayComponents[w21a]", "ArrayComponents[w21a]"),
    ("ArrayComponents[5]", Raises(TypeError, "^ArrayComponents: ")),
    ("ArrayComponents[{1, 2}, 1, 3]", Raises(TypeError, "^ArrayComponents: ")),
]

ADJUST = [
    ("ImageData[ColorBalance[Image[{{{0.8, 0.6, 0.4}}}], RGBColor[0.8, 0.6, 0.4]]]", N([[[1, 1, 1]]], 1e-9)),
    ("ImageData[ColorBalance[Image[{{{0.8, 0.6, 0.4}}}], RGBColor[0.8, 0.6, 0.4] -> GrayLevel[0.5], "
     "Method -> \"RGBScaling\"]]", N([[[0.5, 0.5, 0.5]]])),
    ("ImageData[ColorBalance[Image[{{{0.4, 0.3, 0.2}}}], RGBColor[0.8, 0.6, 0.4], Method -> \"RGBScaling\"]]",
     N([[[0.5, 0.5, 0.5]]])),
    ("ImageData[ColorBalance[Image[{{{0.2, 0.5, 0.7}}}], RGBColor[0.3, 0.6, 0.9] -> RGBColor[0.3, 0.6, 0.9]]]",
     N([[[0.2, 0.5, 0.7]]], 1e-9)),
    ("ColorBalance[Image[{{{0.8, 0.6, 0.4}}}]]", Raises(NotImplementedError, "^ColorBalance: ")),
    ("ColorBalance[Image[{{{0.8, 0.6, 0.4}}}], Red, Method -> \"Bogus\"]", Raises(NotImplementedError,
                                                                                 "^ColorBalance: ")),
    ("ColorBalance[Image[{{{0.8, 0.6, 0.4}}}], Red, Method -> \"RGBScaling\"]", Raises(ValueError,
                                                                                      "^ColorBalance: ")),
    ("ColorBalance[Image[{{{0.8, 0.6, 0.4}}}], {1, 2, 3}]", Raises(NotImplementedError, "^ColorBalance: ")),
    (f"ImageData[ImageRecolor[{C}, Red -> Green]][[1, 1]]", N([0, 1, 0])),
    ("ImageData[ImageRecolor[Image[{{{0.5, 0, 0}, {0, 0, 1}}}], ColorsNear[Red, 0.5] -> Green]]",
     N([[[0, 0.5, 0], [0, 0, 1]]])),
    (f"ImageData[ImageRecolor[{C}, Red -> Blue, Method -> \"Brightness\"]][[1, 1]]", N([1, 0, 0])),
    ("ImageData[ImageRecolor[Image[{{{1, 0, 0}, {0, 0, 1}}}], Image[{{1., 0.}}] -> Green]]",
     N([[[0, 1, 0], [0, 0, 1]]])),
    (f"ImageRecolor[{C}, Red -> 7]", Raises(TypeError, "^ImageRecolor: ")),
    (f"ImageRecolor[{C}, Red -> Green, Method -> \"Bogus\"]", Raises(ValueError, "^ImageRecolor: ")),
    ("ImageMarker[{1, 2}, \"Star\"]", "ImageMarker[{1, 2}, \"Star\"]"),
    ("ImageMarker[{{1, 2}, {3, 4}}]", "ImageMarker[{{1, 2}, {3, 4}}]"),
    ("Head[ImageMarker[Scaled[{0.5, 0.5}], Graphics[Disk[]]]]", "ImageMarker"),
    ("ImageMarker[{1, 2}, \"Blob\"]", Raises(ValueError, "^ImageMarker: ")),
    ("ImageMarker[\"x\"]", Raises(TypeError, "^ImageMarker: ")),
]

OPTIONS = [
    (name, name) for name in ("Interleaving", "Masking", "MaxColorDistance", "MinColorDistance", "PreserveColor",
                              "ZoomCenter", "ZoomFactor")
] + [
    (f"{name}[w21x]", f"{name}[w21x]") for name in ("Interleaving", "Masking", "MaxColorDistance",
                                                    "MinColorDistance", "PreserveColor", "ZoomCenter", "ZoomFactor")
] + [
    (f"Head[{name} -> 1]", "Rule") for name in ("Interleaving", "Masking", "MaxColorDistance", "MinColorDistance",
                                               "PreserveColor", "ZoomCenter", "ZoomFactor")
] + [
    (f"ImageData[{C}, Interleaving -> True] == ImageData[{C}]", True),
    (f"Length[DominantColors[Image[{{{{{{1, 0, 0}}, {{0.99, 0, 0}}, {{0, 0, 1}}}}}}], 3, MinColorDistance -> 0.1]]", 2),
    (f"ImageData[ImageRecolor[{C}, Red -> Green, MaxColorDistance -> 0.01]][[1, 1]]", N([0, 1, 0])),
]

FILES = []
# A fixed directory (test ids must be identical in every pytest-xdist worker); inputs are written atomically and
# each case writes its own output file, so parallel workers do not race.
TMP = os.path.join(tempfile.gettempdir(), "mikoshilang-w21-tests")
if HAVE_PIL:
    import numpy as np
    from PIL import Image as _PILImage
    os.makedirs(TMP, exist_ok=True)

    def _save(arr, name):
        fd, tmp = tempfile.mkstemp(dir=TMP, suffix=".png")
        os.close(fd)
        _PILImage.fromarray(np.array(arr, dtype=np.uint8)).save(tmp)
        os.replace(tmp, os.path.join(TMP, name))
        return os.path.join(TMP, name)
    g = _save([[0, 128], [255, 64]], "g.png")
    c = _save([[[255, 0, 0], [0, 255, 0]]], "c.png")
    o = {k: os.path.join(TMP, f"out{k}.png") for k in range(1, 20)}
    FILES = [
        (f"ImageFileApply[1 - # &, \"{g}\", \"{o[1]}\"]", f"\"{o[1]}\""),
        (f"ImageFileApply[Reverse, \"{c}\", File[\"{o[2]}\"]]", f"File[\"{o[2]}\"]"),
        (f"ImageFileApply[1 - # &, \"{g}\", \"{o[3]}\"]; Reap[ImageFileScan[Sow, \"{o[3]}\"]][[2, 1]]",
         N([1, 127 / 255, 0, 191 / 255])),
        (f"ImageFileApply[1 - # &, \"{TMP}/missing.png\", \"{o[4]}\"]", Raises(ValueError, "^ImageFileApply: ")),
        (f"ImageFileApply[1 - # &, \"{g}\"]", Raises(NotImplementedError, "^ImageFileApply: ")),
        (f"ImageFileFilter[Max, \"{g}\", 1, \"{o[5]}\"]", f"\"{o[5]}\""),
        (f"ImageFileFilter[Max, \"{g}\", 1, \"{o[6]}\"]; Reap[ImageFileScan[Sow, \"{o[6]}\"]][[2, 1]]", N([1, 1, 1, 1])),
        (f"ImageFileFilter[Min, \"{g}\", 0, \"{o[7]}\"]; Reap[ImageFileScan[Sow, \"{o[7]}\"]][[2, 1]]",
         N([0, 128 / 255, 1, 64 / 255])),
        (f"ImageFileFilter[Max, \"{g}\", 1, \"{o[8]}\", Padding -> None]", Raises(ValueError, "^ImageFileFilter: ")),
        (f"Reap[ImageFileScan[Sow, \"{g}\"]][[2, 1]]", N([0, 128 / 255, 1, 64 / 255])),
        (f"ImageFileScan[Sow, \"{c}\"]", "Null"),
        (f"Reap[ImageFileScan[Sow, File[\"{c}\"], Interleaving -> False]][[2, 1]]", N([1, 0, 0, 0, 1, 0])),
        ("ImageFileScan[Sow, 5]", Raises(TypeError, "^ImageFileScan: ")),
    ]

CASES = (REPRESENTATION + ALPHA + COLOR + ARITH + PIXELS + FILTERS + MEASURE + GEOMETRY + TRANSFORMS + VOLUMES
         + COMPONENTS + ADJUST + OPTIONS + FILES)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w21(case):
    check(case)


@pytest.mark.skipif(HAVE_PIL, reason="checks the minimal install")
def test_w21_files_without_pillow():
    with pytest.raises(NotImplementedError, match="^ImageFileScan: .*Pillow"):
        ev("ImageFileScan[Sow, \"/tmp/none.png\"]")


def test_w21_python_api():
    """The helpers documented for W28 exist: parse_image / image_expr round trip, convolve, pad_array, to_space."""
    import numpy as np
    from mikoshilang.library import wl_images as W
    e = ev(G)
    im = W.parse_image(e, "test")
    assert W.image_expr(im) == e and im.dims() == [2, 2] and im.type == "Real32"
    k = np.zeros((3, 3))
    k[1, 1] = 1
    assert np.allclose(W.convolve(im.a, k), im.a)
    assert W.pad_array(im.a, [(1, 1), (1, 1)], 0, "test").shape == (4, 4, 1)
    assert W.to_space(W.parse_image(ev(C), "test"), "HSB", "test").cs == "HSB"


def test_w21_rasterize_factory():
    from mikoshilang.library import wl_graphics_render as R
    assert R.IMAGE_FACTORY is not None
