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

CASES = NEIGHBOURHOOD

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
