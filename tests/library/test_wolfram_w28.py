"""Differential tests for Wolfram batch W28 (images II: filtering, morphology, segmentation and feature
detection; module wl_imageproc).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): the documented examples on numeric
  arrays (quoted with the page name next to each case) and the documented definitions (smaller neighbourhoods at
  the boundaries, sample standard deviation, natural-log entropy, the Kuwahara block rule, ...);
* closed-form or independently computed values (direct window computations written in the tests, SciPy where the
  implementation does not use it, hand-worked morphology on small binary matrices).

Symbol names used as symbolic placeholders are prefixed ``w28`` so they cannot collide with other test modules.
Cases for heads that need SciPy are skipped on a minimal install when the head reports "requires SciPy".
"""
import math
import statistics

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:  # minimal install
    HAVE_SCIPY = False


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


def Head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"head {h}")


def ImgType(t):
    return Pred(lambda g: isinstance(g, Expr) and g.head in ("Image", "Image3D") and g.args[1] == t, f"type {t}")


G = "Image[{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}, {0.7, 0.8, 0.9}}]"
C = "Image[{{{1, 0, 0}, {0, 1, 0}}, {{0, 0, 1}, {1, 1, 1}}}]"     # red green / blue white
Y = "Image[{{0, 128}, {255, 64}}, \"Byte\"]"

NEIGHBOURHOOD = [
    # MeanFilter (reference: MeanFilter page examples)
    ("MeanFilter[{1, 2, 3, 2, 1}, 1]", "{3/2, 2, 7/3, 2, 3/2}"),
    ("MeanFilter[{0, 3, 8, 2}, 1]", "{3/2, 11/3, 13/3, 5}"),
    ("MeanFilter[{{1, 2}, {3, 4}}, {0, 1}]", "{{3/2, 3/2}, {7/2, 7/2}}"),          # direct row means
    ("MeanFilter[{1., 2., 4.}, 1]", N([1.5, 7 / 3, 3.0])),
    (f"MeanFilter[{G}, 1]", N([[0.3, 0.35, 0.4], [0.45, 0.5, 0.55], [0.6, 0.65, 0.7]])),   # direct block means
    ("MeanFilter[{1, w28x}, 1]", "MeanFilter[{1, w28x}, 1]"),
    ("MeanFilter[w28x, 1]", "MeanFilter[w28x, 1]"),
    ("MeanFilter[{1, 2}, -1]", Raises(ValueError, "^MeanFilter: ")),
    ("MeanFilter[{1, 2}, 1.5]", Raises(NotImplementedError, "^MeanFilter: ")),
    ("MeanFilter[{\"a\", 2}, 1]", Raises(TypeError, "^MeanFilter: ")),
    ("MeanFilter[{{1, 2}, {3}}, 1]", Raises(ValueError, "^MeanFilter: ")),
    ("Total[MeanFilter[Range[20], 0]] == 210", "True"),                              # r = 0 is the identity
    # MaxFilter / MinFilter (reference: MaxFilter / MinFilter page examples)
    ("MaxFilter[{1, 2, 3, 2, 1}, 1]", "{2, 3, 3, 3, 2}"),
    ("MinFilter[{1, 2, 3, 2, 1}, 1]", "{1, 1, 2, 1, 1}"),
    ("MinFilter[{0, 3, 8, 2}, 1]", "{0, 0, 2, 2}"),
    ("MaxFilter[{{1, 5, 2}, {0, 0, 7}}, 1]", "{{5, 7, 7}, {5, 7, 7}}"),
    (f"MaxFilter[{G}, 1]", N([[0.5, 0.6, 0.6], [0.8, 0.9, 0.9], [0.8, 0.9, 0.9]])),
    (f"MaxFilter[{C}, {{0, 1}}]", N([[[1, 0, 0], [1, 0, 0]], [[1, 1, 1], [1, 1, 1]]])),   # brightest pixel vector
    (f"ImageType[MaxFilter[{Y}, 1]]", "\"Byte\""),
    ("MaxFilter[{w28a, 1}, 1]", "MaxFilter[{w28a, 1}, 1]"),
    ("MaxFilter[{1, 2}, {1, 1}]", Raises(ValueError, "^MaxFilter: ")),
    ("MaxFilter[{1, 2}, \"r\"]", Raises(TypeError, "^MaxFilter: ")),
    ("MinFilter[{2, 7, 1, 8}, 1] == -MaxFilter[-{2, 7, 1, 8}, 1]", "True"),              # Check: duality
    ("MinFilter[Image[{{{0.9, 0, 0}, {0, 0.6, 0}}}], {0, 1}]", N([[[0, 0.6, 0], [0, 0.6, 0]]])),   # darkest vector
    ("MinFilter[{1, 2}, 1.5]", Raises(NotImplementedError, "^MinFilter: ")),
    # RangeFilter (reference: RangeFilter page examples)
    ("RangeFilter[{1, 2, -3, 4, 5, 4, 3, 2, 1}, 1]", "{1, 5, 7, 8, 1, 2, 2, 2, 1}"),
    ("RangeFilter[{1, -1, 0, 1, 1, 0}, 1]", "{2, 2, 2, 1, 1, 1}"),
    ("With[{w28d = {3, 1, 4, 1, 5, 9, 2, 6}}, RangeFilter[w28d, 2] == MaxFilter[w28d, 2] - MinFilter[w28d, 2]]",
     "True"),
    ("RangeFilter[{1, w28x}, 1]", "RangeFilter[{1, w28x}, 1]"),
    ("RangeFilter[{1, 2}, -2]", Raises(ValueError, "^RangeFilter: ")),
    # MedianFilter (reference: MedianFilter page examples)
    ("MedianFilter[{1, 2, 3, 2, 1}, 1]", "{3/2, 2, 2, 2, 3/2}"),
    ("MedianFilter[{0, 3, 8, 2}, 1]", "{3/2, 3, 3, 5}"),
    ("MedianFilter[{{9, 1, 1}, {1, 1, 1}, {1, 1, 9}}, 1]", "{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}"),   # salt removed
    (f"MedianFilter[{G}, {{0, 1}}]", N([[0.15, 0.2, 0.25], [0.45, 0.5, 0.55], [0.75, 0.8, 0.85]])),
    ("MedianFilter[{w28x, 1}, 1]", "MedianFilter[{w28x, 1}, 1]"),
    ("MedianFilter[{1, 2}, 1/2]", Raises(NotImplementedError, "^MedianFilter: ")),
    # StandardDeviationFilter (reference: StandardDeviationFilter page example)
    ("StandardDeviationFilter[{0, 0, 1, 1, 1, 1, 1, 0, 0, 0}, 1]",
     "{0, 1/Sqrt[3], 1/Sqrt[3], 0, 0, 0, 1/Sqrt[3], 1/Sqrt[3], 0, 0}"),
    ("StandardDeviationFilter[{0, 1, 3, 8, 2, 6}, 1] // N",                                   # statistics.stdev
     N([statistics.stdev(w) for w in ([0, 1], [0, 1, 3], [1, 3, 8], [3, 8, 2], [8, 2, 6], [2, 6])])),
    ("StandardDeviationFilter[{1., 3.}, 1]", N([math.sqrt(2), math.sqrt(2)])),
    (f"ImageType[StandardDeviationFilter[{Y}, 1]]", "\"Real32\""),
    ("StandardDeviationFilter[{1, w28x}, 1]", "StandardDeviationFilter[{1, w28x}, 1]"),
    ("StandardDeviationFilter[{1, 2}, {1, 2}]", Raises(ValueError, "^StandardDeviationFilter: ")),
    # GeometricMeanFilter / HarmonicMeanFilter (reference: their page examples)
    ("GeometricMeanFilter[{1, 2, 4, 8, 16}, 1]", "{Sqrt[2], 2, 4, 8, 8 Sqrt[2]}"),
    ("GeometricMeanFilter[{1, 2, 3, 2, 1}, 1]", "{Sqrt[2], 6^(1/3), 2^(2/3) 3^(1/3), 6^(1/3), Sqrt[2]}"),
    ("GeometricMeanFilter[{1., 4.}, 1]", N([2.0, 2.0])),
    ("GeometricMeanFilter[{1, -2}, 1]", Raises(ValueError, "^GeometricMeanFilter: ")),
    ("GeometricMeanFilter[{1, w28x}, 1]", "GeometricMeanFilter[{1, w28x}, 1]"),
    ("HarmonicMeanFilter[{1, 2, 30, 2, 1}, 1]", "{4/3, 45/23, 90/31, 45/23, 4/3}"),
    ("HarmonicMeanFilter[{1, 2, 3, 4, 5}, 1]", "{4/3, 18/11, 36/13, 180/47, 40/9}"),
    ("HarmonicMeanFilter[{0, 2, 2}, 1]", "{0, 0, 2}"),                                   # HarmonicMean[{0, 2}] == 0
    ("HarmonicMeanFilter[{1, -1}, 1]", Raises(ValueError, "^HarmonicMeanFilter: ")),
    ("HarmonicMeanFilter[{1, w28x}, 1]", "HarmonicMeanFilter[{1, w28x}, 1]"),
    # EntropyFilter (reference: EntropyFilter page examples)
    ("EntropyFilter[{0, 0, 0, 1, 1, 1, 1, 0, 0, 0}, 1] // N",
     N([0, 0, 0.636514, 0.636514, 0, 0, 0.636514, 0.636514, 0, 0])),
    ("EntropyFilter[{1, 2, 3, 3, 3, 3, 3}, 1]", "{Log[2], Log[3], 2/3 Log[3/2] + Log[3]/3, 0, 0, 0, 0}"),
    ("EntropyFilter[{{1, 2}, {3, 4}}, 1]", "{{Log[4], Log[4]}, {Log[4], Log[4]}}"),
    (f"ImageType[EntropyFilter[{G}, 1]]", "\"Real32\""),
    ("EntropyFilter[{1, w28x}, 1]", "EntropyFilter[{1, w28x}, 1]"),
    ("EntropyFilter[{1, 2}, -1]", Raises(ValueError, "^EntropyFilter: ")),
    # CommonestFilter (reference: CommonestFilter page examples, including symbolic data)
    ("CommonestFilter[{0, 1, -1, 1, 2, 1, 3, 3, 3, 1}, 1]", "{0, 1, 1, 1, 1, 1, 3, 3, 3, 1}"),
    ("CommonestFilter[{w28a, w28b, w28b, w28c, w28c, w28c, w28d, w28d, w28e}, 2]",
     "{w28b, w28b, w28b, w28c, w28c, w28c, w28d, w28d, w28d}"),
    ("CommonestFilter[{{1, 1, 2}, {1, 2, 2}}, 1]", "{{1, 1, 2}, {1, 2, 2}}"),            # centre kept on ties
    (f"CommonestFilter[{C}, 1]", Head("Image")),
    ("CommonestFilter[{1, 2}, 0.5]", Raises(NotImplementedError, "^CommonestFilter: ")),
    ("CommonestFilter[{1, 2}, -1]", Raises(ValueError, "^CommonestFilter: ")),
    # KuwaharaFilter (reference: KuwaharaFilter page example)
    ("KuwaharaFilter[{1, 2, 1, -1, 0, 2}, 1]", "{1, 3/2, 3/2, -1/2, -1/2, 2}"),
    ("KuwaharaFilter[{0, 0, 0, 10, 10, 10}, 2]", "{0, 0, 0, 10, 10, 10}"),               # edges preserved
    ("KuwaharaFilter[{1., 2., 1., -1., 0., 2.}, 1]", N([1, 1.5, 1.5, -0.5, -0.5, 2])),
    (f"KuwaharaFilter[{G}, 1]", Head("Image")),
    ("KuwaharaFilter[{1, w28x}, 1]", "KuwaharaFilter[{1, w28x}, 1]"),
    ("KuwaharaFilter[{1, 2}, -1]", Raises(ValueError, "^KuwaharaFilter: ")),
    # MeanShiftFilter (reference: MeanShiftFilter page examples)
    ("MeanShiftFilter[{1, 0, 1, 4, 4, 5, 2, 1}, 2, 1]", N([2 / 3, 2 / 3, 2 / 3, 13 / 3, 13 / 3, 13 / 3, 1.5, 1.5])),
    ("MeanShiftFilter[{0, 1, 10, 11}, 1, 1]", N([0.5, 0.5, 10.5, 10.5])),
    ("MeanShiftFilter[{0, 1, 10, 11}, 1, 1, DistanceFunction -> ManhattanDistance]", N([0.5, 0.5, 10.5, 10.5])),
    ("MeanShiftFilter[{0, 1, 10, 11}, 1, 1/2]", N([0, 1, 10, 11])),                      # nothing within 1/2
    ("MeanShiftFilter[{0, 2, 4}, 1, 3, MaxIterations -> 2]", N([1.0, 2.0, 3.0])),        # same fixed point
    ("MeanShiftFilter[{1, 2}, 1, w28d]", "MeanShiftFilter[{1, 2}, 1, w28d]"),
    ("MeanShiftFilter[{1, 2}, 1, -1]", Raises(ValueError, "^MeanShiftFilter: ")),
    ("MeanShiftFilter[{1, 2}, 1, 1, DistanceFunction -> CosineDistance]", Raises(NotImplementedError,
                                                                                 "^MeanShiftFilter: ")),
]

B5 = "{{0,0,0,0,0,0},{0,0,0,1,1,0},{0,1,1,1,1,0},{0,0,1,1,0,0},{0,0,0,1,1,0},{0,0,0,0,0,0}}"
K3 = "{{0,0,0},{0,1,1},{0,0,1}}"
SQ = "{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,1,0},{0,1,1,1,0},{0,0,0,0,0}}"
BI = 'Image[{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}}, "Bit"]'

MORPHOLOGY = [
    # Dilation (reference: Dilation page examples)
    (f"Dilation[{B5}, {K3}]",
     "{{0,0,1,1,0,0},{1,1,1,1,1,0},{1,1,1,1,1,0},{0,1,1,1,0,0},{0,0,1,1,1,0},{0,0,0,0,0,0}}"),
    ("Dilation[{{2,3,2,5,5},{1,4,1,2,5},{3,2,1,4,5},{2,1,2,3,1},{1,1,2,4,2}}, 1]",
     "{{4,4,5,5,5},{4,4,5,5,5},{4,4,4,5,5},{3,3,4,5,5},{2,2,4,4,4}}"),
    ("Dilation[{0, 0, 1, 0, 0}, 1]", "{0, 1, 1, 1, 0}"),                                   # hand-worked
    ("Dilation[{1/2, 0, 0}, 1]", "{1/2, 1/2, 0}"),
    ("Image[{{0., 1.}, {0., 0.}}] // Dilation[#, 1] & // ImageData", N([[1, 1], [1, 1]])),
    ("Dilation[{1, w28x}, 1]", "Dilation[{1, w28x}, 1]"),
    ("Dilation[{1, 0}, -1]", Raises(ValueError, "^Dilation: ")),
    ("Dilation[{{1, 0}}, {{1, 1}, {1, 1}}]", Raises(NotImplementedError, "^Dilation: ")),
    ("Dilation[\"w28s\", 1]", Raises(TypeError, "^Dilation: ")),
    # Erosion (reference: Erosion page examples)
    (f"Erosion[{B5}, {K3}]",
     "{{0,0,0,0,0,0},{0,0,0,1,0,0},{0,1,1,0,0,0},{0,0,1,0,0,0},{0,0,0,0,0,0},{0,0,0,0,0,0}}"),
    ("Erosion[{{2,3,2,5,5},{1,4,2,3,5},{3,2,2,4,5},{2,1,2,3,1},{1,1,2,4,2}}, 1]",
     "{{1,1,2,2,3},{1,1,2,2,3},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}}"),
    ("Erosion[{1, 1, 1, 0, 1}, 1]", "{1, 1, 0, 0, 0}"),                                   # hand-worked
    ("Image[{{1., 1.}, {0., 1.}}] // Erosion[#, 1] & // ImageData", N([[0, 0], [0, 0]])),
    ("Erosion[{1, w28x}, 1]", "Erosion[{1, w28x}, 1]"),
    ("Erosion[{1, 0}, -1]", Raises(ValueError, "^Erosion: ")),
    ("Erosion[{{1, 0}}, {{1, 1}, {1, 1}}]", Raises(NotImplementedError, "^Erosion: ")),
    # Opening / Closing (reference: Opening and Closing page examples)
    ("Opening[{1,1,1,0,0,1,1,0,0,0,1}, {1,1,1}]", "{1,1,1,0,0,0,0,0,0,0,0}"),
    ("Opening[{0, 3, 3, 3, 0, 5}, 1]", "{0, 3, 3, 3, 0, 0}"),                                    # hand-worked
    (f"Opening[{SQ}, 1]", SQ),                                                           # 3x3 block survives
    ("Opening[{1, w28x}, 1]", "Opening[{1, w28x}, 1]"),
    ("Opening[{1, 0}, -1]", Raises(ValueError, "^Opening: ")),
    ("Closing[{1,1,1,0,0,1,1,0,0,0,1}, {1,1,1}]", "{1,1,1,1,1,1,1,0,0,0,1}"),
    ("Closing[{5, 0, 5}, 1]", "{5, 5, 5}"),                                                # hand-worked
    ("Closing[{1, 0, 0, 0, 1}, 1]", "{1, 0, 0, 0, 1}"),
    ("Closing[{1, w28x}, 1]", "Closing[{1, w28x}, 1]"),
    ("Closing[{1, 0}, -1]", Raises(ValueError, "^Closing: ")),
    # Top / bottom hat (definition: f - Opening[f], Closing[f] - f)
    ("TopHatTransform[{1/2, 3, 1, 0}, 1]", "{0, 5/2, 1/2, 0}"),
    ("TopHatTransform[{0, 3, 3, 3, 0, 5}, 1]", "{0, 0, 0, 0, 0, 5}"),
    ("TopHatTransform[{1, w28x}, 1]", "TopHatTransform[{1, w28x}, 1]"),
    ("TopHatTransform[{1, 0}, -1]", Raises(ValueError, "^TopHatTransform: ")),
    ("BottomHatTransform[{1/2, 3, 1, 0}, 1]", "{5/2, 0, 0, 1}"),
    ("BottomHatTransform[{5, 0, 5}, 1]", "{0, 5, 0}"),
    ("BottomHatTransform[{1, w28x}, 1]", "BottomHatTransform[{1, w28x}, 1]"),
    ("BottomHatTransform[{1, 0}, -1]", Raises(ValueError, "^BottomHatTransform: ")),
    # Geodesic transforms (reference: GeodesicDilation / GeodesicErosion / GeodesicOpening / GeodesicClosing pages)
    ("GeodesicDilation[{0,0,1,1,0}, {w28a,w28b,3,4,0}]", "{Min[0, w28a], Min[1, w28b], 1, 1, 0}"),
    ("GeodesicDilation[{0,0,1,0,0}, {1,1,1,1,0}]", "{1, 1, 1, 1, 0}"),                    # hand-worked
    ("GeodesicDilation[{0,0,1,0,0}, {1,0,1,1,1}]", "{0, 0, 1, 1, 1}"),
    ("GeodesicDilation[{0, 1}, {1}]", Raises(ValueError, "^GeodesicDilation: ")),
    ("GeodesicErosion[{0,0,1,1,0}, {w28a,w28b,3,4,0}]", "{Max[0, w28a], Max[0, w28b], 3, 4, 0}"),
    ("GeodesicErosion[{0,0,1,0,0}, {1,0,3,4,0}]", "{1, 0, 3, 4, 0}"),
    ("GeodesicErosion[{1,1,0,1,1}, {1,0,3,4,0}]", "{1, 0, 3, 4, 1}"),
    ("GeodesicErosion[{0, 1}, {1}]", Raises(ValueError, "^GeodesicErosion: ")),
    ("GeodesicOpening[{0,1,0,1,1,1}, 1]", "{0, 0, 0, 1, 1, 1}"),
    ("GeodesicOpening[{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,0,0},{0,0,1,1,0},{0,0,0,0,0}}, 1]",
     "{{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0},{0,0,0,0,0}}"),
    (f"GeodesicOpening[{SQ}, 1]", SQ),
    ("GeodesicOpening[{1, w28x}, 1]", "GeodesicOpening[{1, w28x}, 1]"),
    ("GeodesicOpening[{1, 0}, -1]", Raises(ValueError, "^GeodesicOpening: ")),
    ("GeodesicClosing[{{1,1,1,1,1},{1,0,0,0,1},{1,0,0,1,1},{1,1,0,0,1},{1,1,1,1,1}}, 1]",
     "{{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1},{1,1,1,1,1}}"),
    ("GeodesicClosing[{1, 0, 1}, 1]", "{1, 1, 1}"),                                         # hand-worked
    ("GeodesicClosing[{1, 0, 0, 0, 1}, 1]", "{1, 0, 0, 0, 1}"),
    ("GeodesicClosing[{1, w28x}, 1]", "GeodesicClosing[{1, w28x}, 1]"),
    ("GeodesicClosing[{1, 0}, -1]", Raises(ValueError, "^GeodesicClosing: ")),
]

SEGMENTATION = [
    # MaxDetect / MinDetect (reference: MaxDetect and MinDetect page examples)
    ("MaxDetect[{1, 2, 1, 2, 3, 2}]", "{0, 1, 0, 0, 1, 0}"),
    ("MaxDetect[{{7,8,2,2},{8,4,7,4},{7,1,3,3},{1,1,9,3}}]", "{{0,1,0,0},{1,0,0,0},{0,0,0,0},{0,0,1,0}}"),
    ("MaxDetect[{0, 3, 1, 5, 4}, 2]", "{0, 0, 0, 1, 1}"),
    ("MaxDetect[{2, 2, 2}]", "{1, 1, 1}"),                                                   # one plateau
    ("MaxDetect[{1, w28x}]", "MaxDetect[{1, w28x}]"),
    ("MaxDetect[{1, 2}, -1]", Raises(ValueError, "^MaxDetect: ")),
    ("MinDetect[{1, 2, 1, 3, 2, 3}]", "{1, 0, 1, 0, 1, 0}"),
    ("MinDetect[{{7,8,2,2},{8,4,7,4},{7,1,3,3},{1,1,9,3}}]", "{{0,0,1,1},{0,0,0,0},{0,1,0,0},{1,1,0,0}}"),
    ("MinDetect[{3, 0, 1}]", "{0, 1, 0}"),
    ("MinDetect[{1, w28x}]", "MinDetect[{1, w28x}]"),
    ("MinDetect[{1, 2}, -1]", Raises(ValueError, "^MinDetect: ")),
    # FillingTransform (definition: reconstruction by erosion from the border)
    ("FillingTransform[{{1,1,1,1},{1,0,0,1},{1,1,1,1}}]", "{{1,1,1,1},{1,1,1,1},{1,1,1,1}}"),
    ("FillingTransform[{{1,1,1,1},{1,0,0,1},{1,1,0,1}}]", "{{1,1,1,1},{1,0,0,1},{1,1,0,1}}"),   # hole open
    ("FillingTransform[{{5,5,5},{5,1,5},{5,5,5}}, 2]", "{{5,5,5},{5,3,5},{5,5,5}}"),          # h-fill
    ("FillingTransform[{{5,5,5,5,5},{5,1,5,0,5},{5,5,5,5,5}}, {{0,0,0,0,0},{0,1,0,0,0},{0,0,0,0,0}}]",
     "{{5,5,5,5,5},{5,5,5,0,5},{5,5,5,5,5}}"),                                               # marker
    ("FillingTransform[{{0.5,0.5,0.5},{0.5,0.1,0.5},{0.5,0.5,0.5}}]", N([[0.5] * 3] * 3)),
    (f"FillingTransform[{BI}] // ImageData", "{{1,1,1},{1,1,1},{1,1,1}}"),
    ("FillingTransform[{1, w28x}]", "FillingTransform[{1, w28x}]"),
    ("FillingTransform[{1, 0}, -1]", Raises(ValueError, "^FillingTransform: ")),
    ("FillingTransform[{1, 0}, Padding -> \"Periodic\"]", Raises(NotImplementedError, "^FillingTransform: ")),
    # HitMissTransform (reference: hand-worked; documented 1 / -1 / 0 kernel convention)
    ("HitMissTransform[{{0,0,0},{0,1,0},{0,1,0}}, {{-1,-1,-1},{-1,1,-1},{0,0,0}}]", "{{0,0,0},{0,1,0},{0,0,0}}"),
    ("HitMissTransform[{0, 1, 0, 1, 1}, {-1, 1, -1}]", "{0, 1, 0, 0, 0}"),                    # isolated ones
    ("HitMissTransform[{1, 1, 1}, {1, 1, 1}]", "{1, 1, 1}"),                                  # Padding 1
    ("HitMissTransform[{1, 0}, {1, 1}]", Raises(NotImplementedError, "^HitMissTransform: ")),
    ("HitMissTransform[{1, 0}, {2, 1, 0}]", Raises(ValueError, "^HitMissTransform: ")),
    # MorphologicalComponents (reference: MorphologicalComponents page example)
    ("MorphologicalComponents[{{0,0,1,2},{0,0,3,4},{5,0,0,0},{6,7,0,0}}]", "{{0,0,1,1},{0,0,1,1},{2,0,0,0},{2,2,0,0}}"),
    ("MorphologicalComponents[{{1,0,1},{0,1,0}}, CornerNeighbors -> False]", "{{1,0,2},{0,3,0}}"),
    ("MorphologicalComponents[{{1,0,1},{0,1,0}}]", "{{1,0,1},{0,1,0}}"),
    ("MorphologicalComponents[{0, 0}]", "{0, 0}"),
    ("MorphologicalComponents[{1, w28x}]", "MorphologicalComponents[{1, w28x}]"),
    ("MorphologicalComponents[{1, 0}, Method -> \"Nearest\"]", Raises(NotImplementedError, "^MorphologicalComponents: ")),
    # DeleteBorderComponents (reference: DeleteBorderComponents page example)
    ("DeleteBorderComponents[{{0,0,0,0},{0,1,0,0},{0,0,0,0},{0,0,0,2}}]", "{{0,0,0,0},{0,1,0,0},{0,0,0,0},{0,0,0,0}}"),
    ("DeleteBorderComponents[{1, 0, 1}]", "{0, 0, 0}"),
    ("DeleteBorderComponents[{0, 1, 0}]", "{0, 1, 0}"),
    ("DeleteBorderComponents[{1, w28x}]", "DeleteBorderComponents[{1, w28x}]"),
    ("DeleteBorderComponents[\"w28s\"]", Raises(TypeError, "^DeleteBorderComponents: ")),
    # FindThreshold (reference: FindThreshold page examples; Otsu over 256 bins)
    ("FindThreshold[{1, 2, 3, 10, 11}]", N(2.9921875)),
    ("FindThreshold[{0, 0, 1, 1}]", Pred(lambda g: 0 <= _num(g) < 1, "between the clusters")),
    ("FindThreshold[{1, 2, 3, 10, 11}, Method -> \"Mean\"]", N(5.4)),
    ("FindThreshold[{1, 2, 3, 10, 11}, Method -> \"Median\"]", N(3)),
    ("FindThreshold[Image[{{0.1, 0.9}, {0.2, 0.8}}]]", Pred(lambda g: 0.2 <= _num(g) < 0.8, "between")),
    ("FindThreshold[w28x]", "FindThreshold[w28x]"),
    ("FindThreshold[{1, 2}, Method -> \"Entropy\"]", Raises(NotImplementedError, "^FindThreshold: ")),
    ("FindThreshold[{1, 2}, Method -> \"w28bad\"]", Raises(ValueError, "^FindThreshold: ")),
    # Binarize (definition: pixels above the threshold become 1)
    ("Binarize[Image[{{0.1, 0.9}, {0.2, 0.8}}]] // ImageData", "{{0, 1}, {0, 1}}"),
    ("Binarize[Image[{{0.1, 0.9}, {0.2, 0.8}}], 0.5] // ImageData", "{{0, 1}, {0, 1}}"),
    ("Binarize[Image[{{0.1, 0.9}, {0.2, 0.8}}], {0.15, 0.85}] // ImageData", "{{0, 0}, {1, 1}}"),
    ("Binarize[Image[{{0.1, 0.9}, {0.2, 0.8}}], # > 0.15 &] // ImageData", "{{0, 1}, {1, 1}}"),
    ("Binarize[Image[{{0.1, 0.9}}]]", ImgType("Bit")),
    ("Binarize[w28x]", "Binarize[w28x]"),
    ("Binarize[{1, 2}]", Raises(TypeError, "^Binarize: ")),
    # LocalAdaptiveBinarize (definition: value > mean of the neighbourhood)
    ("LocalAdaptiveBinarize[Image[{{0.1, 0.5, 0.9}}], 1] // ImageData", "{{0, 0, 1}}"),
    ("LocalAdaptiveBinarize[Image[{{0.5, 0.5, 0.5}}], 1] // ImageData", "{{0, 0, 0}}"),
    ("LocalAdaptiveBinarize[Image[{{0.1, 0.5, 0.9}}], 1]", ImgType("Bit")),
    ("LocalAdaptiveBinarize[w28x, 1]", "LocalAdaptiveBinarize[w28x, 1]"),
    ("LocalAdaptiveBinarize[Image[{{0.1}}], -1]", Raises(ValueError, "^LocalAdaptiveBinarize: ")),
    # MorphologicalBinarize (definition: values > t2 plus the connected values > t1)
    ("MorphologicalBinarize[Image[{{0.9, 0.5, 0.1, 0.5}}], {0.3, 0.8}] // ImageData", "{{1, 1, 0, 0}}"),
    ("MorphologicalBinarize[Image[{{0.5, 0.5}}], {0.3, 0.8}] // ImageData", "{{0, 0}}"),
    ("MorphologicalBinarize[Image[{{0.9, 0.5}}], {0.3, 0.8}]", ImgType("Bit")),
    ("MorphologicalBinarize[w28x, {0.3, 0.8}]", "MorphologicalBinarize[w28x, {0.3, 0.8}]"),
    ("MorphologicalBinarize[Image[{{0.5}}]]", Raises(NotImplementedError, "^MorphologicalBinarize: ")),
    # Threshold (reference: Threshold page examples)
    ("Threshold[{1, 10^-1, 10^-2, 10^-8, 10^-11}]", "{1, 1/10, 1/100, 1/100000000, 0}"),
    ("Threshold[{-3,1,-2,0,2,-1,0,1,-3,3,2}, 3/2]", "{-3,0,-2,0,2,0,0,0,-3,3,2}"),
    ("Threshold[{-3,1,-2,0,2,-1,0,1,-3,3,2}, {\"LargestValues\", 3}]", "{-3,0,0,0,0,0,0,0,-3,3,0}"),
    ("Threshold[{1., 1., 8.12619*10^-17, 1.18674*10^-16, 0.}]", N([1, 1, 0, 0, 0])),
    ("Threshold[{-3, 1, 2}, {\"Soft\", 1}]", "{-2, 0, 1}"),
    ("Threshold[{1, w28x}, 1/2]", "Threshold[{1, w28x}, 1/2]"),
    ("Threshold[{1 + I, 2}]", Raises(NotImplementedError, "^Threshold: ")),
    ("Threshold[{1, 2}, -1]", Raises(ValueError, "^Threshold: ")),
]

DISTANCE = [
    # DistanceTransform (reference: hand-worked distances)
    (f"DistanceTransform[Image[{SQ}]] // ImageData", N([[0] * 5, [0, 1, 1, 1, 0], [0, 1, 2, 1, 0], [0, 1, 1, 1, 0], [0] * 5])),
    ("DistanceTransform[Image[{{1,1,1},{1,1,1},{1,1,1}}], Padding -> 0] // ImageData", N([[1, 1, 1], [1, 2, 1], [1, 1, 1]])),
    ("DistanceTransform[Image[{{0,1,1,1}}], DistanceFunction -> ManhattanDistance] // ImageData", N([[0, 1, 2, 3]])),
    ("DistanceTransform[Image[{{0,0,0},{0,0,0},{0,0,1}}], Padding -> 1] // ImageData",
     N([[0, 0, 0], [0, 0, 0], [0, 0, 1]])),
    ("DistanceTransform[Image[{{0,1,1},{1,1,1},{1,1,1}}]] // ImageData",
     N([[0, 1, 2], [1, math.sqrt(2), math.sqrt(5)], [2, math.sqrt(5), math.sqrt(8)]])),     # brute force
    ("DistanceTransform[Image[{{0,1,1},{1,1,1},{1,1,1}}], DistanceFunction -> ChessboardDistance] // ImageData",
     N([[0, 1, 2], [1, 1, 2], [2, 2, 2]])),
    ("DistanceTransform[Image[{{0, 1}}]]", ImgType("Real32")),
    ("DistanceTransform[w28x]", "DistanceTransform[w28x]"),
    ("DistanceTransform[{{0, 1}}]", Raises(TypeError, "^DistanceTransform: ")),
    ("DistanceTransform[Image[{{1, 1}}]]", Raises(ValueError, "^DistanceTransform: ")),
    ("DistanceTransform[Image[{{0, 1}}], DistanceFunction -> CosineDistance]",
     Raises(NotImplementedError, "^DistanceTransform: ")),
    # InverseDistanceTransform (reference: documented round trip with DistanceTransform)
    ("InverseDistanceTransform[DistanceTransform[Image[{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,1,0},{0,1,1,0,0},{0,0,0,0,0}}]]]"
     " // ImageData", "{{0,0,0,0,0},{0,1,1,1,0},{0,1,1,1,0},{0,1,1,0,0},{0,0,0,0,0}}"),
    ("InverseDistanceTransform[Image[{{0,0,0},{0,1.2,0},{0,0,0}}]] // ImageData", "{{0,1,0},{1,1,1},{0,1,0}}"),
    ("InverseDistanceTransform[Image[{{0,0,0},{0,2,0},{0,0,0}}]] // ImageData", "{{1,1,1},{1,1,1},{1,1,1}}"),
    ("InverseDistanceTransform[Image[{{0,0,0},{0,1.2,0},{0,0,0}}], DistanceFunction -> ChessboardDistance] // ImageData",
     "{{1,1,1},{1,1,1},{1,1,1}}"),
    ("InverseDistanceTransform[Image[{{0, 1}}]]", ImgType("Bit")),
    ("InverseDistanceTransform[w28x]", "InverseDistanceTransform[w28x]"),
    ("InverseDistanceTransform[{{0, 1}}]", Raises(TypeError, "^InverseDistanceTransform: ")),
    ("InverseDistanceTransform[Image[{{0, 1}}], DistanceFunction -> CosineDistance]",
     Raises(NotImplementedError, "^InverseDistanceTransform: ")),
    # SkeletonTransform (reference: centres of maximal disks, verified by the documented round trip)
    ("SkeletonTransform[Image[{{0,0,0,0,0,0,0},{0,1,1,1,1,1,0},{0,1,1,1,1,1,0},{0,1,1,1,1,1,0},{0,0,0,0,0,0,0}}]]"
     " // ImageData", N([[0] * 7, [0] * 7, [0, 0, 2, 2, 2, 0, 0], [0] * 7, [0] * 7])),
    (f"SkeletonTransform[Image[{SQ}]] // ImageData", N([[0] * 5, [0] * 5, [0, 0, 2, 0, 0], [0] * 5, [0] * 5])),
    (f"InverseDistanceTransform[SkeletonTransform[Image[{SQ}]]] // ImageData", SQ),
    ("SkeletonTransform[Image[{{0, 1}}]]", ImgType("Real32")),
    ("SkeletonTransform[w28x]", "SkeletonTransform[w28x]"),
    ("SkeletonTransform[{{0, 1}}]", Raises(TypeError, "^SkeletonTransform: ")),
    ("SkeletonTransform[Image[{{0, 1}}], Method -> \"ReducedMedialAxis\"]",
     Raises(NotImplementedError, "^SkeletonTransform: ")),
]

COMPONENTS = [
    # DeleteSmallComponents (definition: Otsu threshold of the component sizes)
    ("DeleteSmallComponents[{{1,1,1,0},{1,1,1,0},{0,0,0,0},{3,0,0,2}}]", "{{1,1,1,0},{1,1,1,0},{0,0,0,0},{0,0,0,0}}"),
    ("DeleteSmallComponents[{{1,1,1,0},{1,1,1,0},{1,1,1,0},{0,2,2,2}}]", "{{1,1,1,0},{1,1,1,0},{1,1,1,0},{0,0,0,0}}"),
    ("DeleteSmallComponents[Image[{{1, 1, 0, 1, 0, 1, 1, 1}}, \"Bit\"], 2] // ImageData", "{{0, 0, 0, 0, 0, 1, 1, 1}}"),
    ("DeleteSmallComponents[{1, 1, 0, 0, 0, 2, 2, 2}, 2]", "{0, 0, 0, 0, 0, 2, 2, 2}"),      # labels, not pieces,
    ("DeleteSmallComponents[{1, 0, 1}]", "{1, 0, 1}"),                                     # one size only
    ("DeleteSmallComponents[{1, w28x}]", "DeleteSmallComponents[{1, w28x}]"),
    ("DeleteSmallComponents[{1, 0}, -1]", Raises(ValueError, "^DeleteSmallComponents: ")),
    # ComponentMeasurements (reference: ComponentMeasurements page Count example; hand-worked geometry)
    ("ComponentMeasurements[Image[{{0,0,0,0,0,0,0},{0,0,0,1,1,1,0},{0,0,0,1,0,1,0},{0,0,0,1,1,1,0},{0,0,0,0,0,0,0},"
     "{0,1,0,1,1,1,0},{0,0,0,0,0,0,0}}, \"Bit\"], \"Count\"]", "{1 -> 8, 2 -> 1, 3 -> 3}"),
    ("ComponentMeasurements[{{1,1,0},{0,0,0},{0,0,2}}, {\"Centroid\", \"BoundingBox\"}]",
     "{1 -> {{1., 2.5}, {{0, 2}, {2, 3}}}, 2 -> {{2.5, 0.5}, {{2, 0}, {3, 1}}}}"),
    ("ComponentMeasurements[{{1,1,0},{0,0,0},{0,0,2}}, \"Count\", #Count > 1 &]", "{1 -> 2}"),
    ("ComponentMeasurements[{Image[{{0.2, 0.4, 0}}], {{1, 1, 0}}}, \"Mean\"]",
     Pred(lambda g: abs(_num(g.args[0].args[1]) - 0.3) < 1e-9, "1 -> 0.3")),
    ("ComponentMeasurements[{{1,0,2}}, \"LabelCount\"]", "2"),
    ("ComponentMeasurements[{{1,1,2}}, \"Neighbors\"]", "{1 -> {2}, 2 -> {1}}"),
    ("ComponentMeasurements[{{1,1,0},{0,0,2}}, \"Count\", \"PropertyAssociation\"]", "<|\"Count\" -> {2, 1}|>"),
    ("ComponentMeasurements[w28x, \"Count\"]", "ComponentMeasurements[w28x, \"Count\"]"),
    ("ComponentMeasurements[{{1, 0}}, \"Eccentricity\"]", Raises(NotImplementedError, "^ComponentMeasurements: ")),
    ("ComponentMeasurements[{{1, 0}}, \"Mean\"]", Raises(NotImplementedError, "^ComponentMeasurements: ")),
    ("ComponentMeasurements[{{1, 0}}, 5]", Raises(TypeError, "^ComponentMeasurements: ")),
    # SelectComponents (reference: hand-worked, consistent with ComponentMeasurements)
    ("SelectComponents[{{1,1,0},{0,0,0},{0,0,2}}, #Count > 1 &]", "{{1,1,0},{0,0,0},{0,0,0}}"),
    ("SelectComponents[{{1,1,0},{0,0,0},{0,0,2}}, \"Count\", 1]", "{{0,0,0},{0,0,0},{0,0,2}}"),
    ("SelectComponents[{{1,1,0},{0,0,0},{0,0,2}}, \"Count\", -1]", "{{1,1,0},{0,0,0},{0,0,0}}"),
    ("SelectComponents[{{1,1,0},{0,0,0},{0,0,2}}, \"Large\"]", "{{1,1,0},{0,0,0},{0,0,0}}"),
    ("SelectComponents[w28x, \"Large\"]", "SelectComponents[w28x, \"Large\"]"),
    ("SelectComponents[{{1, 0}}, \"w28bad\"]", Raises(ValueError, "^SelectComponents: ")),
    ("SelectComponents[{{1, 0}}, \"Eccentricity\", 1]", Raises(NotImplementedError, "^SelectComponents: ")),
    # MorphologicalPerimeter (definition: foreground with a background neighbour)
    ("MorphologicalPerimeter[{{1,1,1},{1,1,1},{1,1,1}}]", "{{1,1,1},{1,0,1},{1,1,1}}"),
    (f"MorphologicalPerimeter[{SQ}]", "{{0,0,0,0,0},{0,1,1,1,0},{0,1,0,1,0},{0,1,1,1,0},{0,0,0,0,0}}"),
    ("MorphologicalPerimeter[{{1,1,1},{1,1,1},{1,1,1}}, Padding -> 1]", "{{0,0,0},{0,0,0},{0,0,0}}"),
    ("MorphologicalPerimeter[{1, w28x}]", "MorphologicalPerimeter[{1, w28x}]"),
    ("MorphologicalPerimeter[{1, 0}, Padding -> 2]", Raises(NotImplementedError, "^MorphologicalPerimeter: ")),
    # MorphologicalEulerNumber (definition: regions minus holes)
    ("MorphologicalEulerNumber[Image[{{1,1,1,0,0},{1,0,1,0,1},{1,1,1,0,0}}]]", "1"),
    ("MorphologicalEulerNumber[{{1,0,1},{0,0,0},{1,0,1}}]", "4"),
    ("MorphologicalEulerNumber[{{1,1,1},{1,0,1},{1,1,1}}]", "0"),
    ("MorphologicalEulerNumber[{{1,0},{0,1}}, CornerNeighbors -> False]", "2"),
    ("MorphologicalEulerNumber[w28x]", "MorphologicalEulerNumber[w28x]"),
    ("MorphologicalEulerNumber[{{{1}}}]", Raises(NotImplementedError, "^MorphologicalEulerNumber: ")),
    # Pruning (reference: Pruning page example)
    ("Pruning[{{0,0,1,0,0},{0,0,1,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}]",
     "{{0,0,0,0,0},{0,0,0,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}"),
    ("Pruning[{{0,0,1,0,0},{0,0,1,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}, {1}]",
     "{{0,0,0,0,0},{0,0,1,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}"),
    ("Pruning[{{0,0,1,0,0},{0,0,1,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}, 2]",
     "{{0,0,0,0,0},{0,0,0,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}"),
    ("Pruning[{{0,0,1,0,0},{0,0,1,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}, 1]",
     "{{0,0,1,0,0},{0,0,1,0,0},{1,1,1,1,1},{1,0,0,0,1},{1,1,1,1,1}}"),                 # branch of 2 kept
    ("Pruning[{1, w28x}]", "Pruning[{1, w28x}]"),
    ("Pruning[{{1, 0}}, -1]", Raises(ValueError, "^Pruning: ")),
    ("Pruning[{{1, 0}}, Padding -> 1]", Raises(NotImplementedError, "^Pruning: ")),
]

BOX7 = "{{0,0,0,0,0,0,0},{0,0,0,0,0,0,0},{0,0,1,1,1,0,0},{0,0,1,1,1,0,0},{0,0,1,1,1,0,0},{0,0,0,0,0,0,0},{0,0,0,0,0,0,0}}"
_c, _e, _m = 0.759989, 1.08185, 1.84184
LOG7 = [[0] * 7, [0, _c, _e, _m, _e, _c, 0], [0, _e, -2.92369, -_m, -2.92369, _e, 0], [0, _m, -_m, 0, -_m, _m, 0],
        [0, _e, -2.92369, -_m, -2.92369, _e, 0], [0, _c, _e, _m, _e, _c, 0], [0] * 7]
STEP = "{0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0}"

LINEAR = [
    # GaussianFilter (reference: GaussianFilter page examples)
    ("GaussianFilter[{1, 1, 2, 1, 4, 3, 4, 3, 2, 4, 2}, 2]",
     N([1.05088, 1.21184, 1.62721, 1.94912, 3.05088, 3.32191, 3.47456, 3.05088, 2.73728, 3., 2.42368], 1e-5)),
    (f"GaussianFilter[{STEP}, 2]",
     N([0, 0, 0.0508822, 0.262721, 0.737279, 0.949118, 1, 0.949118, 0.737279, 0.262721, 0.0508822, 0, 0], 1e-5)),
    (f"GaussianFilter[{STEP}, 2, 1]",
     N([0, 0, 0.122499, 0.377501, 0.377501, 0.122499, 0, -0.122499, -0.377501, -0.377501, -0.122499, 0, 0], 1e-5)),
    ("GaussianFilter[{0, 1, 2, 3, 4, 5}, 2, 1, Padding -> None]", N([1, 1])),               # ramp slope 1
    ("GaussianFilter[{{2, 2}, {2, 2}}, 1]", N([[2, 2], [2, 2]])),                            # constants kept
    ("GaussianFilter[Image[{{0.5, 0.5}, {0.5, 0.5}}], 1] // ImageData", N([[0.5, 0.5], [0.5, 0.5]])),
    ("GaussianFilter[{1, w28x}, 1]", "GaussianFilter[{1, w28x}, 1]"),
    ("GaussianFilter[{1, 2}, w28r]", "GaussianFilter[{1, 2}, w28r]"),
    ("GaussianFilter[{1, 2}, -1]", Raises(ValueError, "^GaussianFilter: ")),
    ("GaussianFilter[{1, 2}, 1, Method -> \"w28bad\"]", Raises(NotImplementedError, "^GaussianFilter: ")),
    ("GaussianFilter[{{1, 2}}, 1, {2, 2}]", Raises(NotImplementedError, "^GaussianFilter: ")),
    # LaplacianGaussianFilter (reference: LaplacianGaussianFilter page examples)
    ("LaplacianGaussianFilter[{0,0,0,0,0,1,1,1,1,1,0,0,0,0,0}, 1]", N([0, 0, 0, 0, 1, -1, 0, 0, 0, -1, 1, 0, 0, 0, 0])),
    (f"LaplacianGaussianFilter[{BOX7}, 1]", N(LOG7, 1e-5)),
    ("LaplacianGaussianFilter[{3, 3, 3, 3}, 1]", N([0, 0, 0, 0])),
    ("LaplacianGaussianFilter[{0, 1, 4, 9, 16}, 1, Padding -> None]", N([2, 2, 2])),          # (x²)'' = 2
    ("LaplacianGaussianFilter[{1, w28x}, 1]", "LaplacianGaussianFilter[{1, w28x}, 1]"),
    ("LaplacianGaussianFilter[{1, 2}, -1]", Raises(ValueError, "^LaplacianGaussianFilter: ")),
    ("LaplacianGaussianFilter[{1, 2}, 1, Standardized -> False]",
     Raises(NotImplementedError, "^LaplacianGaussianFilter: ")),
    # GradientFilter (reference: GradientFilter page examples)
    ("GradientFilter[{1, 2, 3, 4, 5, 1, 5, 4, 3, 2, 1}, 1]", N([0.5, 1, 1, 1, 1.5, 0, 1.5, 1, 1, 1, 0.5])),
    ("GradientFilter[{0, 0, 0, 1, 4, 6, 4, 1, 0, 0, 0}, 1]", N([0, 0, 0.5, 2, 2.5, 0, 2.5, 2, 0.5, 0, 0])),
    ("GradientFilter[{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, 1, Padding -> None]", N([[1]])),       # unit slope
    ("GradientFilter[{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}, 1, Padding -> None]", N([[1]])),
    ("GradientFilter[Image[{{0., 0.5, 1.}}], 1]", ImgType("Real32")),
    ("GradientFilter[{1, w28x}, 1]", "GradientFilter[{1, w28x}, 1]"),
    ("GradientFilter[{1, 2}, -1]", Raises(ValueError, "^GradientFilter: ")),
    ("GradientFilter[{1, 2}, 1, Method -> {\"NonMaxSuppression\" -> True}]",
     Raises(NotImplementedError, "^GradientFilter: ")),
    # GradientOrientationFilter (reference: GradientOrientationFilter page example; documented angle convention)
    ("GradientOrientationFilter[{1, 1, 0, 1, 0, 1, 1}, 1]",
     N([[-math.pi / 2], [0], [-math.pi / 2], [-math.pi / 2], [-math.pi / 2], [0], [-math.pi / 2]])),
    ("GradientOrientationFilter[{{1, 2}, {1, 2}}, 1]", N([[[math.pi / 2]] * 2] * 2)),         # doc: ArcTan[0, |a-b|]
    ("GradientOrientationFilter[{{0, 0, 0}, {1, 1, 1}, {2, 2, 2}}, 1, Padding -> None]", N([[[0]]])),
    ("GradientOrientationFilter[{{0, 1, 2}, {1, 2, 3}, {2, 3, 4}}, 1, Padding -> None]", N([[[math.pi / 4]]])),
    ("GradientOrientationFilter[{1, w28x}, 1]", "GradientOrientationFilter[{1, w28x}, 1]"),
    ("GradientOrientationFilter[{1, 2}, -1]", Raises(ValueError, "^GradientOrientationFilter: ")),
    ("GradientOrientationFilter[{{{1}}}, 1]", Raises(NotImplementedError, "^GradientOrientationFilter: ")),
    # BilateralFilter (reference: BilateralFilter page example)
    ("BilateralFilter[{1, 0, 1, 4, 4, 5, 2, 1}, 3, 1]",
     N([0.819466, 0.573802, 0.908264, 4.10055, 4.11161, 4.47202, 1.73819, 1.26565], 1e-5)),
    ("BilateralFilter[{2, 2, 2}, 1, 1]", N([2, 2, 2])),
    ("BilateralFilter[{0, 0, 10, 10}, 1, 1/100]", N([0, 0, 10, 10])),                        # edges kept
    ("BilateralFilter[Image[{{0.2, 0.4}}], 1, 1]", ImgType("Real32")),
    ("BilateralFilter[{1, w28x}, 1, 1]", "BilateralFilter[{1, w28x}, 1, 1]"),
    ("BilateralFilter[{1, 2}, 1, w28m]", "BilateralFilter[{1, 2}, 1, w28m]"),
    ("BilateralFilter[{1, 2}, -1, 1]", Raises(ValueError, "^BilateralFilter: ")),
    # GuidedFilter (reference: closed forms of the guided filter)
    ("GuidedFilter[{1, 2, 3, 4}, 1, 0]", N([1, 2, 3, 4])),                                    # ϵ = 0: identity
    ("GuidedFilter[{1, 2, 3, 4}, 1, 10^12]", N([1.75, 13 / 6, 17 / 6, 3.25], 1e-6)),          # box mean of means
    ("GuidedFilter[{5, 5, 5}, {1, 2, 3}, 1, 0.1]", N([5, 5, 5])),
    ("GuidedFilter[{1, 2, 3, 4}, {1, 2, 3, 4}, 1, 0]", N([1, 2, 3, 4])),
    ("GuidedFilter[Image[{{0.2, 0.4}}], 1, 0.1]", ImgType("Real32")),
    ("GuidedFilter[{1, 2}, 1, w28e]", "GuidedFilter[{1, 2}, 1, w28e]"),
    ("GuidedFilter[{1, 2}, 1, -1]", Raises(ValueError, "^GuidedFilter: ")),
    ("GuidedFilter[{1, 2}]", Raises(TypeError, "^GuidedFilter: ")),
]

DETECT = [
    # CrossingDetect / ContourDetect (reference: CrossingDetect and ContourDetect page examples)
    ("CrossingDetect[{4, 0, 1, -2, 1, -2, -3, -1, 3}]", "{0, 0, 1, 0, 1, 0, 0, 0, 1}"),
    ("CrossingDetect[{1, -.2, -1, 1}, CornerNeighbors -> None]", "{1, 0, 1}"),
    ("CrossingDetect[{1, -.2, -1, 1}, 0.5]", "{0, 0, 0, 1}"),
    ("CrossingDetect[{{1, 0}, {0, -1}}]", "{{1, 0}, {0, 0}}"),
    ("CrossingDetect[{{1, 0}, {0, -1}}, CornerNeighbors -> False]", "{{0, 0}, {0, 0}}"),
    ("CrossingDetect[Image[{{0.5}}]]", ImgType("Bit")),
    ("CrossingDetect[w28x]", "CrossingDetect[w28x]"),
    ("CrossingDetect[{1, -1}, -1]", Raises(ValueError, "^CrossingDetect: ")),
    ("ContourDetect[{4, 0, 1, -2, 1, -2, -3, -1, 3}]", "{0, 1, 1, 0, 1, 0, 0, 0, 1}"),
    ("ContourDetect[{1, -1, 1, 1, 0}, CornerNeighbors -> None]", "{1, 1, 0, 1}"),
    ("ContourDetect[{1, 0.1, -1}, 0.5]", "{0, 1, 0}"),
    ("ContourDetect[w28x]", "ContourDetect[w28x]"),
    ("ContourDetect[{1, -1}, -1]", Raises(ValueError, "^ContourDetect: ")),
    ("ContourDetect[{1}, CornerNeighbors -> None]", Raises(ValueError, "^ContourDetect: ")),
    # MorphologicalTransform (reference: hand-worked neighbourhood rules; Game of Life blinker)
    ("MorphologicalTransform[{{0,0,0,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,0,0,0}}, \"Life\"]",
     "{{0,0,0,0,0},{0,0,0,0,0},{0,1,1,1,0},{0,0,0,0,0},{0,0,0,0,0}}"),
    ("MorphologicalTransform[{{0,0,0,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,0,0,0}}, \"Life\", 2]",
     "{{0,0,0,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,1,0,0},{0,0,0,0,0}}"),
    ("MorphologicalTransform[{{0,0,0},{0,1,0},{0,0,0}}, \"Max\"]", "{{1,1,1},{1,1,1},{1,1,1}}"),
    ("MorphologicalTransform[{{0,0,0},{0,1,0},{0,0,0}}, \"Top\"]", "{{0,1,0},{0,0,0},{0,0,0}}"),
    ("MorphologicalTransform[{{0,0,0},{0,1,0},{0,0,0}}, \"Clean\"]", "{{0,0,0},{0,0,0},{0,0,0}}"),
    ("MorphologicalTransform[{{0,0,0},{0,1,0},{0,0,0}}, Total[Flatten[#]] > 0 &]", "{{1,1,1},{1,1,1},{1,1,1}}"),
    ("MorphologicalTransform[{{1,0,0},{0,0,0},{0,0,0}}, \"Right\", 2]", "{{0,0,1},{0,0,0},{0,0,0}}"),
    ("MorphologicalTransform[{{1,0,0},{0,0,0},{0,0,0}}, {\"Right\", \"Bottom\"}]", "{{0,0,0},{0,1,0},{0,0,0}}"),
    ("MorphologicalTransform[{{1,1,1},{1,1,1},{1,1,1}}, \"Remove\", Padding -> 1]", "{{0,0,0},{0,0,0},{0,0,0}}"),
    ("MorphologicalTransform[w28x, \"Max\"]", "MorphologicalTransform[w28x, \"Max\"]"),
    ("MorphologicalTransform[{{1}}, 30]", Raises(NotImplementedError, "^MorphologicalTransform: ")),
    ("MorphologicalTransform[{{1}}, \"Bridge\"]", Raises(NotImplementedError, "^MorphologicalTransform: ")),
    ("MorphologicalTransform[{{1}}, 2 &]", Raises(ValueError, "^MorphologicalTransform: ")),
    # RegionBinarize (reference: hand-worked region growing)
    ("RegionBinarize[Image[{{0.1,0.2,0.9},{0.15,0.8,0.9}}], {{1,0,0},{0,0,0}}, 0.1] // ImageData", "{{1,1,0},{1,0,0}}"),
    ("RegionBinarize[Image[{{0.1,0.2,0.9},{0.15,0.8,0.9}}], {{0.5, 1.5}}, 0.1] // ImageData", "{{1,1,0},{1,0,0}}"),
    ("RegionBinarize[Image[{{0.1,0.2,0.9},{0.15,0.8,0.9}}], {{0,0,1},{0,0,0}}, 0.15] // ImageData", "{{0,0,1},{0,1,1}}"),
    ("RegionBinarize[Image[{{0.1, 0.5, 0.1}}], {{1, 0, 0}}, 0.1] // ImageData", "{{1, 0, 0}}"),   # not connected
    ("RegionBinarize[Image[{{0.1}}], {{1}}, w28d]", "RegionBinarize[Image[{{0.1}}], {{1}}, w28d]"),
    ("RegionBinarize[Image[{{0.1}}], {{1}}, -1]", Raises(ValueError, "^RegionBinarize: ")),
    ("RegionBinarize[{{0.1}}, {{1}}, 1]", Raises(TypeError, "^RegionBinarize: ")),
    ("RegionBinarize[Image[{{0.1}}], {{1}}, 1, Method -> \"Mahalanobis\"]",
     Raises(NotImplementedError, "^RegionBinarize: ")),
    # option symbols (inert names)
    ("CornerNeighbors", "CornerNeighbors"),
    ("Head[CornerNeighbors]", "Symbol"),
    ("CornerNeighbors -> False", "CornerNeighbors -> False"),
    ("MaxFeatures", "MaxFeatures"),
    ("Head[MaxFeatures]", "Symbol"),
    ("MaxFeatures -> 10", "MaxFeatures -> 10"),
    ("MaxFeatureDisplacement", "MaxFeatureDisplacement"),
    ("Head[MaxFeatureDisplacement]", "Symbol"),
    ("MaxFeatureDisplacement -> 5", "MaxFeatureDisplacement -> 5"),
    ("KeypointStrength", "KeypointStrength"),
    ("Head[KeypointStrength]", "Symbol"),
    ("KeypointStrength -> 0.1", "KeypointStrength -> 0.1"),
]

CASES = NEIGHBOURHOOD + MORPHOLOGY + SEGMENTATION + DISTANCE + COMPONENTS + LINEAR + DETECT

SCIPY_HEADS = set()


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w28(case):
    if not HAVE_SCIPY:
        try:
            check(case)
        except NotImplementedError as exc:
            if "requires SciPy" in str(exc):
                pytest.skip("needs SciPy")
            raise
        return
    check(case)
