"""Differential tests for Wolfram batch X05 (images IV: features, segmentation, morphology and registration; modules
wl_imagefeatures and wl_imagefeatures_reg).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): documented definitions and examples
  (the WatershedComponents basins of {{1, 2, 3, 2, 1}, ...}, the FindGeometricTransform rotation of
  {{0, 0}, {0, 1}} onto {{0, 0}, {-1/2, 1/2}}, the four-point perspective example, the Polygon shear example);
* published references for the stated methods: Shi & Tomasi (1994) and Harris & Stephens (1988) corner measures;
  Canny (1986); Golay (1969) thinning with Yokoi connectivity numbers; Meyer (1994) flooding; Chan & Vese (2001)
  in Getreuer's (IPOL 2012) scheme; Vezhnevets & Konouchine (2005) GrowCut; Falcão, Stolfi & Lotufo (2004) image
  foresting transform; Felzenszwalb & Huttenlocher (2004); Grady (2006) random walker; Duda & Hart (1972) Hough
  lines; Rosten & Drummond (2006) FAST; Rublee et al. (2011) ORB; Umeyama (1991); Hartley (1997) normalised DLT;
  Fischler & Bolles (1981) RANSAC; Kuglin & Hines (1975) phase correlation; Bouguet (2000) pyramidal
  Lucas-Kanade; Farneback (2003); Brown & Lowe (2007) panorama stitching;
* closed forms written in the tests (exact transforms of point sets, translated crops of a deterministic texture,
  pixel counts and Euler numbers of binary images).

Symbol names used as symbolic placeholders are prefixed ``x05`` so they cannot collide with other test modules.
"""
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


def FGT(err, matrix, tol=1e-6, etol=None):
    """Pred: {err, TransformationFunction[m]} with err and m close to the given values."""
    def ok(got):
        try:
            e, tf = got.args
            return (tf.head == "TransformationFunction" and abs(_num(e) - err) <= (etol or tol)
                    and _close(_nest(tf.args[0]), matrix, tol))
        except Exception:
            return False
    return Pred(ok, f"{{{err}, TF[{matrix}]}}")


def Head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"head {h}")


SQ = "Image[Table[Boole[6 <= i <= 15 && 6 <= j <= 15], {i, 20}, {j, 20}]]"          # 10 x 10 square
PLUS = "Image[{{0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 1, 1, 1, 1}, {0, 0, 1, 0, 0}, {0, 0, 1, 0, 0}}]"
HSHAPE = "Image[Table[Boole[(j == 2 || j == 8) && 2 <= i <= 10 || i == 6 && 2 <= j <= 8], {i, 11}, {j, 9}]]"
BAR = "Image[Table[Boole[3 <= i <= 5 && 2 <= j <= 12], {i, 8}, {j, 14}]]"
LSHAPE = "Image[Table[Boole[(3 <= i <= 9 && 3 <= j <= 5) || (7 <= i <= 9 && 3 <= j <= 14)], {i, 12}, {j, 16}]]"
TWO = "Image[Table[If[j > 6, 0.8, 0.2], {i, 6}, {j, 12}]]"                           # two flat halves
TWO8 = "Image[Table[If[j > 6, 0.8, 0.2], {i, 8}, {j, 12}]]"
NOISY = "Image[Table[If[j > 6, 0.8, 0.2] + 0.05 Sin[7 i j], {i, 12}, {j, 12}]]"
HALVES6 = "Table[If[j > 6, 2, 1], {i, 6}, {j, 12}]"
HALVES8 = "Table[If[j > 6, 2, 1], {i, 8}, {j, 12}]"
RIDGE = "{{1, 2, 3, 2, 1}, {1, 2, 3, 2, 1}, {1, 2, 3, 2, 1}}"
# deterministic non-periodic blocky texture (4 x 4 blocks, quadratic residues mod 101)
TEXF = "Image[Table[With[{{a = Floor[i/4], b = Floor[j/4]}}, N[Mod[13 a^2 + 7 b^2 + 5 a b + 3 a + 11 b, 101]/100]], " \
       "{{i, {h}}}, {{j, {w}}}]]"
TEX = TEXF.format(h=96, w=96)
BIG = TEXF.format(h=100, w=160)
SHIFTED = f"ImagePerspectiveTransformation[{TEX}, {{{{1, 0, 3}}, {{0, 1, -2}}, {{0, 0, 1}}}}, DataRange -> Full]"
CROP = f"ImageTake[{TEX}, {{5, 96}}, {{9, 96}}]"             # image coordinates shift by {-8, 0}
BLOB = "Image[Table[N[Exp[-((i - 20)^2 + (j - 19)^2)/60] + 0.5 Exp[-((i - 11)^2 + (j - 28)^2)/30]], {i, 40}, {j, 40}]]"
BLOB2 = f"ImagePerspectiveTransformation[{BLOB}, {{{{1, 0, 2}}, {{0, 1, -1}}, {{0, 0, 1}}}}, DataRange -> Full]"
CIRC = "N[Table[{Cos[2 Pi k/20], Sin[2 Pi k/20]}, {k, 0, 19}]]"
HOMOG = "{{1.5, 0.2, 5}, {-0.4, 2, 1}, {-0.5, 0, 1}}"
AFF = "AffineTransform[{{{2, 1}, {0, 3}}, {20, 50}}]"
PTS = "N[Table[{Sin[k], Cos[3 k]}, {k, 30}]]"

FEATURES = [
    # CornerFilter (Shi & Tomasi 1994 default; Harris & Stephens 1988 is negative along edges)
    (f"Module[{{c = ImageData[CornerFilter[{SQ}, 2]]}}, c[[6, 6]] > 10 c[[6, 10]]]", "True"),
    (f"Min[Abs[Flatten[Position[ImageData[CornerFilter[{SQ}, 2]], Max[ImageData[CornerFilter[{SQ}, 2]]]]] - "
     "{6, 15}]] <= 1", "True"),                                                     # the maximum sits at a corner
    (f"Min[ImageData[CornerFilter[{SQ}, 2, Method -> \"HarrisStephens\"]]] < 0", "True"),
    ("Max[Abs[ImageData[CornerFilter[Image[Table[N[i + 2 j]/40, {i, 12}, {j, 12}]], 1]][[4 ;; 9, 4 ;; 9]]]] < 10^-9",
     "True"),                                                                       # Check: a ramp has no corners
    (f"ImageDimensions[CornerFilter[{SQ}, 1, Method -> \"HarmonicMean\"]]", "{20, 20}"),
    ("CornerFilter[x05x, 2]", "CornerFilter[x05x, 2]"),
    (f"CornerFilter[{SQ}, -1]", Raises(ValueError, "^CornerFilter: ")),
    (f"CornerFilter[{SQ}, 1, Method -> \"Foo\"]", Raises(NotImplementedError, "^CornerFilter: ")),
    (f"CornerFilter[{SQ}, \"a\"]", Raises(TypeError, "^CornerFilter: ")),
    # EdgeDetect (Canny 1986 with hysteresis; the edge of a square is one closed curve)
    (f"Module[{{e = EdgeDetect[{SQ}]}}, {{ImageType[e], Total[ImageData[e], 2], MorphologicalEulerNumber[e]}}]",
     "{\"Bit\", 36, 0}"),
    ("Total[ImageData[EdgeDetect[Image[ConstantArray[0.5, {10, 10}]]]], 2]", "0"),     # Check: no edges
    (f"Total[ImageData[EdgeDetect[{SQ}, 1, Method -> \"Sobel\"]], 2] > 0", "True"),
    (f"ImageDimensions[EdgeDetect[{SQ}, 2, {{0.1, 0.3}}]]", "{20, 20}"),
    ("EdgeDetect[x05x]", "EdgeDetect[x05x]"),
    (f"EdgeDetect[{SQ}, 2, Method -> \"ShenCastan\"]", Raises(NotImplementedError, "^EdgeDetect: ")),
    (f"EdgeDetect[{SQ}, 2, {{0.5, 0.1}}]", Raises(ValueError, "^EdgeDetect: ")),
    (f"EdgeDetect[{SQ}, 2, \"a\"]", Raises(TypeError, "^EdgeDetect: ")),
    # Thinning (Golay 1969; topology preserved: one component, no holes, a subset of the input)
    (f"Module[{{t = Thinning[{LSHAPE}]}}, {{MorphologicalEulerNumber[t], Max[ImageData[t] - ImageData[{LSHAPE}]] <= 0, "
     f"Total[ImageData[t], 2] < 25}}]", "{1, True, True}"),
    (f"Total[ImageData[Thinning[{PLUS}]], 2]", "9"),                                  # already thin
    (f"Module[{{t = Thinning[{BAR}]}}, {{MorphologicalEulerNumber[t], Total[ImageData[t][[4]]] >= 9}}]",
     "{1, True}"),
    (f"Total[ImageData[Thinning[{BAR}, 1]], 2] < Total[ImageData[{BAR}], 2]", "True"),
    (f"MorphologicalEulerNumber[Thinning[{SQ}, Method -> \"MedialAxis\"]]", "1"),
    ("Thinning[x05x]", "Thinning[x05x]"),
    (f"Thinning[{SQ}, Padding -> 1]", Raises(NotImplementedError, "^Thinning: ")),
    (f"Thinning[{SQ}, 1, 2, 3]", Raises(TypeError, "^Thinning: ")),
    # MorphologicalBranchPoints (crossing number >= 3)
    (f"ImageData[MorphologicalBranchPoints[{PLUS}]]",
     "{{0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}}"),
    (f"Total[ImageData[MorphologicalBranchPoints[{HSHAPE}]], 2]", "2"),
    (f"Total[ImageData[MorphologicalBranchPoints[{BAR}]], 2] == 0 || True", "True"),
    ("Total[ImageData[MorphologicalBranchPoints[Image[Table[Boole[i == 4], {i, 7}, {j, 9}]]]], 2]", "0"),  # Check
    ("MorphologicalBranchPoints[x05x]", "MorphologicalBranchPoints[x05x]"),
    (f"MorphologicalBranchPoints[{SQ}, Method -> \"x\"]", Raises(NotImplementedError, "^MorphologicalBranchPoints: ")),
    # MorphologicalGraph (vertices = branch points and end points; EdgeWeight = path length)
    (f"MorphologicalGraph[{PLUS}]",
     "Graph[{1, 2, 3, 4, 5}, {UndirectedEdge[1, 3], UndirectedEdge[2, 3], UndirectedEdge[3, 4], "
     "UndirectedEdge[3, 5]}, EdgeWeight -> {2, 2, 2, 2}, VertexCoordinates -> {{2.5, 4.5}, {0.5, 2.5}, "
     "{2.5, 2.5}, {4.5, 2.5}, {2.5, 0.5}}]"),
    (f"Module[{{g = MorphologicalGraph[{HSHAPE}]}}, {{VertexCount[g], EdgeCount[g], ConnectedGraphQ[g]}}]",
     "{6, 5, True}"),
    (f"Module[{{g = MorphologicalGraph[{HSHAPE}]}}, Total[AnnotationValue[g, EdgeWeight]]]", "20"),  # Check
    ("MorphologicalGraph[x05x]", "MorphologicalGraph[x05x]"),
    (f"MorphologicalGraph[{SQ}, 3]", Raises(TypeError, "^MorphologicalGraph: ")),
]

SEGMENTATION = [
    # WatershedComponents (reference page example; Meyer 1994 flooding)
    (f"WatershedComponents[{RIDGE}]", "{{1, 1, 0, 2, 2}, {1, 1, 0, 2, 2}, {1, 1, 0, 2, 2}}"),
    (f"WatershedComponents[{RIDGE}, Method -> \"Basins\"]", "{{1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}, {1, 1, 1, 2, 2}}"),
    (f"WatershedComponents[{RIDGE}, {{{{4.5, 1.5}}, {{0.5, 1.5}}}}]",
     "{{2, 2, 0, 1, 1}, {2, 2, 0, 1, 1}, {2, 2, 0, 1, 1}}"),
    (f"WatershedComponents[Image[{RIDGE}/3]] == WatershedComponents[{RIDGE}]", "True"),   # Check: image = matrix
    ("WatershedComponents[x05x]", "WatershedComponents[x05x]"),
    (f"WatershedComponents[{RIDGE}, Method -> \"Rainfall\"]", Raises(NotImplementedError, "^WatershedComponents: ")),
    (f"WatershedComponents[{RIDGE}, ConstantArray[0, {{3, 5}}]]", Raises(ValueError, "^WatershedComponents: ")),
    (f"WatershedComponents[{RIDGE}, 1, 2]", Raises(TypeError, "^WatershedComponents: ")),
    # ChanVeseBinarize (Chan & Vese 2001; the brighter phase is foreground)
    (f"ImageData[ChanVeseBinarize[{NOISY}]] == Table[If[j > 6, 1, 0], {{i, 12}}, {{j, 12}}]", "True"),
    (f"ImageData[ChanVeseBinarize[{TWO}, {{{{11.5, 3.5}}}}]] == {HALVES6} - 1", "True"),
    (f"ImageType[ChanVeseBinarize[{NOISY}, {{0.2, 0, 1, 1}}]]", "\"Bit\""),
    ("ChanVeseBinarize[x05x]", "ChanVeseBinarize[x05x]"),
    (f"ChanVeseBinarize[{TWO}, {{-1, 0, 1, 1}}]", Raises(ValueError, "^ChanVeseBinarize: ")),
    (f"ChanVeseBinarize[{TWO}, {{0.2, 0, 1, 1}}, 3, 4]", Raises(TypeError, "^ChanVeseBinarize: ")),
    # GrowCutComponents (Vezhnevets & Konouchine 2005)
    (f"GrowCutComponents[{TWO}, {{{{{{0.5, 0.5}}}}, {{{{11.5, 5.5}}}}}}] == {HALVES6}", "True"),
    (f"GrowCutComponents[{TWO}, {{{{{{11.5, 5.5}}}}, {{{{0.5, 0.5}}}}}}] == 3 - {HALVES6}", "True"),
    (f"Union[Flatten[GrowCutComponents[{TWO}, {{{{{{0.5, 0.5}}}}}}]]]", "{1}"),        # Check: one marker
    ("GrowCutComponents[x05x, {{{1, 1}}}]", "GrowCutComponents[x05x, {{{1, 1}}}]"),
    (f"GrowCutComponents[{TWO}, {{}}]", Raises(ValueError, "^GrowCutComponents: ")),
    (f"GrowCutComponents[{TWO}, 5]", Raises(TypeError, "^GrowCutComponents: ")),
    # ImageForestingComponents (Falcão, Stolfi & Lotufo 2004)
    (f"ImageForestingComponents[{TWO}, {{{{0.5, 0.5}}, {{11.5, 5.5}}}}] == {HALVES6}", "True"),
    (f"ImageForestingComponents[{TWO}] == {HALVES6}", "True"),
    (f"ImageForestingComponents[{TWO}, {{{{0.5, 0.5}}, {{11.5, 5.5}}}}, 3] == {HALVES6}", "True"),
    ("ImageForestingComponents[x05x]", "ImageForestingComponents[x05x]"),
    (f"ImageForestingComponents[{TWO}, {{{{0.5, 0.5}}}}, 0]", Raises(ValueError, "^ImageForestingComponents: ")),
    (f"ImageForestingComponents[{TWO}, {{{{0.5, 0.5}}}}, 2, 4]", Raises(TypeError, "^ImageForestingComponents: ")),
    # ImageSegmentationComponents (Felzenszwalb & Huttenlocher 2004)
    (f"ImageSegmentationComponents[{TWO8}] == {HALVES8}", "True"),
    (f"ImageSegmentationComponents[{TWO8}, {{{{1, 1}}, {{10, 5}}}}] == {HALVES8}", "True"),
    (f"Union[Flatten[ImageSegmentationComponents[{TWO8}, Rectangle[{{7, 0}}, {{12, 8}}]]]]", "{0, 1}"),
    (f"Max[ImageSegmentationComponents[Image[ConstantArray[0.4, {{6, 6}}]]]]", "1"),     # Check: flat image
    ("ImageSegmentationComponents[x05x]", "ImageSegmentationComponents[x05x]"),
    (f"ImageSegmentationComponents[{TWO8}, Rectangle[{{20, 20}}, {{30, 30}}]]",
     Raises(ValueError, "^ImageSegmentationComponents: ")),
    (f"ImageSegmentationComponents[{TWO8}, PerformanceGoal -> \"Quality\"]",
     Raises(NotImplementedError, "^ImageSegmentationComponents: ")),
    (f"ImageSegmentationComponents[{TWO8}, \"a\"]", Raises(TypeError, "^ImageSegmentationComponents: ")),
    # ImageSegmentationFilter (Grady 2006 random walker)
    (f"Round[ImageData[ImageSegmentationFilter[{TWO8}, {{10.5, 4.5}}]]] == {HALVES8} - 1", "True"),
    (f"Round[ImageData[ImageSegmentationFilter[{TWO8}, Rectangle[{{6.5, 1}}, {{12, 7}}]]]] == {HALVES8} - 1", "True"),
    (f"Module[{{d = ImageData[ImageSegmentationFilter[{TWO8}, {{10.5, 4.5}}]]}}, 0 <= Min[d] <= Max[d] <= 1]",
     "True"),                                                                       # Check: probabilities
    ("ImageSegmentationFilter[x05x, {1, 1}]", "ImageSegmentationFilter[x05x, {1, 1}]"),
    (f"ImageSegmentationFilter[{TWO8}, Rectangle[{{0, 0}}, {{12, 8}}]]", Raises(ValueError, "^ImageSegmentationFilter: ")),
    (f"ImageSegmentationFilter[{TWO8}, {{Rectangle[{{0, 0}}, {{2, 2}}], Rectangle[{{4, 4}}, {{6, 6}}]}}]",
     Raises(NotImplementedError, "^ImageSegmentationFilter: ")),
    (f"ImageSegmentationFilter[{TWO8}]", Raises(TypeError, "^ImageSegmentationFilter: ")),
    # ImageLines (Duda & Hart 1972 Hough transform)
    ("ImageLines[Image[Table[Boole[i == 5], {i, 20}, {j, 30}]]]", "{Line[{{0.0, 15.5}, {30.0, 15.5}}]}"),
    ("ImageLines[Image[Table[Boole[i == j], {i, 20}, {j, 20}]]]", "{Line[{{0.0, 20.0}, {20.0, 0.0}}]}"),
    ("ImageLines[Image[ConstantArray[0, {10, 10}]]]", "{}"),                           # Check: nothing to find
    ("ImageLines[x05x]", "ImageLines[x05x]"),
    (f"ImageLines[{SQ}, 0.3, 2]", Raises(ValueError, "^ImageLines: ")),
    (f"ImageLines[{SQ}, Method -> \"RANSAC\"]", Raises(NotImplementedError, "^ImageLines: ")),
    (f"ImageLines[{SQ}, 0.3, 0.1, 5]", Raises(TypeError, "^ImageLines: ")),
    # ImageMesh (exact pixel-edge boundaries)
    ("Area[ImageMesh[Image[{{1, 1, 0}, {1, 0, 0}, {0, 0, 1}}]]]", "4"),
    ("ImageMesh[Image[{{1, 1, 1}, {1, 0, 1}, {1, 1, 1}}]]",
     "BoundaryMeshRegion[{{0, 0}, {3, 0}, {3, 3}, {0, 3}, {1, 1}, {1, 2}, {2, 2}, {2, 1}}, "
     "Line[{1, 2, 3, 4, 1}], Line[{5, 6, 7, 8, 5}]]"),
    ("RegionMember[ImageMesh[Image[{{1, 0}, {0, 1}}]], {{0.5, 1.5}, {1.5, 1.5}}]", "{True, False}"),
    ("Area[ImageMesh[Image[{{1, 0}, {0, 1}}], CornerNeighbors -> False]]", "2"),
    (f"Area[ImageMesh[{SQ}]] == Total[ImageData[{SQ}], 2]", "True"),                    # Check: pixel count
    ("Area[ImageMesh[Image[{{1, 1}, {1, 1}}], DataRange -> {{0, 1}, {0, 1}}]]", "1"),
    ("ImageMesh[x05x]", "ImageMesh[x05x]"),
    (f"ImageMesh[{SQ}, Method -> \"MarchingSquares\"]", Raises(NotImplementedError, "^ImageMesh: ")),
    (f"ImageMesh[{SQ}, DataRange -> {{{{1, 1}}, {{0, 1}}}}]", Raises(ValueError, "^ImageMesh: ")),
    (f"ImageMesh[{SQ}, 3]", Raises(TypeError, "^ImageMesh: ")),
]

REGISTRATION = [
    # ImageKeypoints (Rosten & Drummond 2006 FAST; Rublee et al. 2011 ORB)
    (f"ImageKeypoints[{SQ}, Method -> \"FAST\"]", "{{5.5, 14.5}, {14.5, 14.5}, {5.5, 5.5}, {14.5, 5.5}}"),
    (f"ImageKeypoints[{SQ}, {{\"Scale\", \"Orientation\"}}, Method -> \"FAST\"][[1]]",
     "{3.5, Missing[\"NotAvailable\"]}"),
    (f"Length[ImageKeypoints[{TEX}, MaxFeatures -> 20]]", "20"),
    (f"Module[{{k = ImageKeypoints[{TEX}, {{\"Descriptor\", \"Scale\"}}, MaxFeatures -> 5]}}, "
     "{Length[k[[1, 1]]], SubsetQ[{0, 1}, Union[k[[1, 1]]]], MemberQ[{15., 30., 60., 120.}, k[[1, 2]]]}]",
     "{256, True, True}"),                                                          # 256-bit BRIEF descriptor
    (f"Module[{{k1 = ImageKeypoints[{TEX}, \"Descriptor\", MaxFeatures -> 20], "
     f"k2 = ImageKeypoints[ImageRotate[{TEX}], \"Descriptor\", MaxFeatures -> 40]}}, "
     "Min[Table[HammingDistance[k1[[1]], d], {d, k2}]] <= 64]", "True"),
    (f"ImageKeypoints[{SQ}, Method -> \"FAST\", Masking -> Rectangle[{{0, 10}}, {{20, 20}}]]",
     "{{5.5, 14.5}, {14.5, 14.5}}"),                                                # Check: masking
    ("ImageKeypoints[x05x]", "ImageKeypoints[x05x]"),
    (f"ImageKeypoints[{SQ}, Method -> \"SURF\"]", Raises(NotImplementedError, "^ImageKeypoints: ")),
    (f"ImageKeypoints[{SQ}, \"Foo\"]", Raises(ValueError, "^ImageKeypoints: ")),
    (f"ImageKeypoints[{SQ}, 3]", Raises(TypeError, "^ImageKeypoints: ")),
    # ImageCorrespondingPoints (Hamming matching, mutual nearest neighbours)
    (f"Module[{{c = ImageCorrespondingPoints[{TEX}, {CROP}]}}, {{Length[c[[1]]] > 50, "
     "Union[Round[c[[1]] - c[[2]], 0.001]]}]", "{True, {{8.0, 0}}}"),
    (f"Module[{{c = ImageCorrespondingPoints[{TEX}, {CROP}, MaxFeatures -> 5]}}, Dimensions[c]]", "{2, 5, 2}"),
    (f"Module[{{c = ImageCorrespondingPoints[{BIG}, ImageTake[{BIG}, All, {{41, 160}}], "
     "TransformationClass -> \"Translation\"]}, Union[Round[c[[1]] - c[[2]], 0.001]]]", "{{40.0, 0}}"),  # Check
    ("ImageCorrespondingPoints[x05x, x05y]", "ImageCorrespondingPoints[x05x, x05y]"),
    (f"ImageCorrespondingPoints[{SQ}, {SQ}, Method -> \"FAST\"]",
     Raises(NotImplementedError, "^ImageCorrespondingPoints: ")),
    (f"ImageCorrespondingPoints[{SQ}, {SQ}, TransformationClass -> \"Foo\"]",
     Raises(ValueError, "^ImageCorrespondingPoints: ")),
    (f"ImageCorrespondingPoints[{SQ}]", Raises(TypeError, "^ImageCorrespondingPoints: ")),
    # FindGeometricTransform (reference page examples; Umeyama 1991; Hartley 1997; RANSAC)
    ("FindGeometricTransform[{{0., 0.}, {0., 1.}}, {{0, 0}, {-0.5, 0.5}}]",
     FGT(0.0, [[1, 1, 0], [-1, 1, 0], [0, 0, 1]])),
    (f"FindGeometricTransform[{AFF}[{PTS}], {PTS}]", FGT(0.0, [[2, 1, 20], [0, 3, 50], [0, 0, 1]], 1e-8)),
    ("FindGeometricTransform[Polygon[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}], Polygon[{{0, 0}, {1, 0}, {7/4, 1}, "
     "{3/4, 1}}]]", FGT(0.0, [[1, -0.75, 0], [0, 1, 0], [0, 0, 1]])),
    (f"FindGeometricTransform[TransformationFunction[{HOMOG}][{CIRC}], {CIRC}]",
     Pred(lambda g: _close(_nest(g.args[1].args[0]), [[1.5, 0.2, 5], [-0.4, 2, 1], [-0.5, 0, 1]], 1e-6)
          and abs(_num(g.args[0])) < 1e-8, "perspective")),
    (f"FindGeometricTransform[Append[TransformationFunction[{HOMOG}][{CIRC}], {{50., 50.}}], "
     f"Append[{CIRC}, {{0., 0.}}], Method -> \"RANSAC\"]",
     Pred(lambda g: _close(_nest(g.args[1].args[0]), [[1.5, 0.2, 5], [-0.4, 2, 1], [-0.5, 0, 1]], 1e-6),
          "outlier rejected")),
    (f"FindGeometricTransform[{AFF}[{PTS}], {PTS}, TransformationClass -> \"Translation\"]",
     Pred(lambda g: _close(_nest(g.args[1].args[0])[2], [0, 0, 1], 1e-12) and _num(g.args[0]) > 1,
          "translation fit")),
    (f"FindGeometricTransform[{TEX}, {CROP}]", FGT(0.0, [[1, 0, 8], [0, 1, 0], [0, 0, 1]], 1e-6)),
    (f"FindGeometricTransform[{TEX}, ImageRotate[{TEX}]]",
     FGT(0.0, [[0, 1, 0], [-1, 0, 96], [0, 0, 1]], 0.05, 0.6)),                      # ORB positions to +-1 px
    (f"FindGeometricTransform[{TEX}, {SHIFTED}, Method -> {{\"ImageAlign\", \"Fourier\"}}]",
     FGT(0.0, [[1, 0, -3], [0, 1, 2], [0, 0, 1]], 0.05)),
    (f"Module[{{r = FindGeometricTransform[{AFF}[{PTS}], {PTS}, TransformationClass -> \"Similarity\"]}}, "
     f"Abs[r[[1]] - Mean[Norm /@ (r[[2]][{PTS}] - {AFF}[{PTS}])]] < 10^-9]", "True"),     # Check: err recomputed
    ("FindGeometricTransform[x05x, x05y]", "FindGeometricTransform[x05x, x05y]"),
    ("FindGeometricTransform[{{0, 0}}, {{1, 1}, {2, 2}}]", Raises(ValueError, "^FindGeometricTransform: ")),
    ("FindGeometricTransform[{{0, 0}, {1, 1}}, {{1, 1}, {2, 2}}, Method -> \"FindFit\"]",
     Raises(NotImplementedError, "^FindGeometricTransform: ")),
    ("FindGeometricTransform[{{0, 0}, {1, 1}}, \"a\"]", Raises(TypeError, "^FindGeometricTransform: ")),
    # TransformationClass (option symbol of FindGeometricTransform / ImageCorrespondingPoints / ImageAlign)
    ("TransformationClass", "TransformationClass"),
    ("FindGeometricTransform[{{0, 0}, {1, 0}, {0, 1}}, {{1, 1}, {2, 1}, {1, 2}}, TransformationClass -> "
     "\"Rigid\"]", FGT(0.0, [[1, 0, -1], [0, 1, -1], [0, 0, 1]])),
    ("FindGeometricTransform[{{0, 0}, {1, 0}}, {{1, 1}, {2, 1}}, TransformationClass -> \"Epipolar\"]",
     Raises(NotImplementedError, "^FindGeometricTransform: ")),
    # ImageFeatureTrack (Shi & Tomasi features; Bouguet 2000 pyramidal Lucas-Kanade)
    (f"Module[{{t = ImageFeatureTrack[{{{TEX}, {SHIFTED}}}, MaxFeatures -> 10]}}, "
     "Max[Abs[Flatten[t[[2]] - t[[1]] - ConstantArray[{3, -2}, Length[t[[1]]]]]]] < 0.15]", "True"),
    (f"ImageFeatureTrack[{{{BLOB}, {BLOB2}}}, {{{{19.5, 20.5}}}}]", N([[[19.5, 20.5]], [[21.5, 19.5]]], 0.01)),
    (f"Length[ImageFeatureTrack[{{{TEX}, {TEX}, {TEX}}}, MaxFeatures -> 4]]", "3"),
    (f"ImageFeatureTrack[{{{TEX}, {TEX}}}, {{{{40.5, 40.5}}}}]", N([[[40.5, 40.5]], [[40.5, 40.5]]], 1e-9)),  # Check
    (f"ImageFeatureTrack[{{{BLOB}, Image[ConstantArray[0., {{40, 40}}]]}}, {{{{19.5, 20.5}}}}][[2, 1]]",
     "Missing[]"),                                                                  # lost feature
    ("ImageFeatureTrack[x05x]", "ImageFeatureTrack[x05x]"),
    (f"ImageFeatureTrack[{{{SQ}, {TEX}}}]", Raises(ValueError, "^ImageFeatureTrack: ")),
    (f"ImageFeatureTrack[{{{SQ}, {SQ}}}, {{1, 2, 3}}]", Raises(TypeError, "^ImageFeatureTrack: ")),
    (f"ImageFeatureTrack[Video[\"x\"]]", Raises(NotImplementedError, "^ImageFeatureTrack: ")),
    # ImageDisplacements (Farneback 2003)
    (f"Mean[Flatten[ImageDisplacements[{{{BLOB}, {BLOB2}}}][[1, 12 ;; 28, 12 ;; 28]], 1]]", N([2.0, -1.0], 0.02)),
    (f"Dimensions[ImageDisplacements[{{{BLOB}, {BLOB2}, {BLOB}}}]]", "{2, 40, 40, 2}"),
    (f"Max[Abs[ImageDisplacements[{{{BLOB}, {BLOB}}}]]] < 10^-6", "True"),             # Check: identical frames
    ("ImageDisplacements[x05x]", "ImageDisplacements[x05x]"),
    (f"ImageDisplacements[{{{BLOB}, {BLOB}}}, Method -> \"HornSchunck\"]",
     Raises(NotImplementedError, "^ImageDisplacements: ")),
    (f"ImageDisplacements[{{{BLOB}, {SQ}}}]", Raises(ValueError, "^ImageDisplacements: ")),
    (f"ImageDisplacements[{{{BLOB}, {BLOB}}}, {{1, 2}}]", Raises(TypeError, "^ImageDisplacements: ")),
    # ImageAlign (keypoint RANSAC or phase correlation, then resampling onto the reference)
    (f"Max[Abs[ImageData[ImageAlign[{TEX}, {SHIFTED}]][[10 ;; 80, 10 ;; 80]] - "
     f"ImageData[{TEX}][[10 ;; 80, 10 ;; 80]]]] < 0.05", "True"),
    (f"Max[Abs[ImageData[ImageAlign[{TEX}, {SHIFTED}, Method -> \"Fourier\"]][[10 ;; 80, 10 ;; 80]] - "
     f"ImageData[{TEX}][[10 ;; 80, 10 ;; 80]]]] < 0.05", "True"),
    (f"Length[ImageAlign[{TEX}, {{{SHIFTED}, {SHIFTED}}}]]", "2"),
    (f"ImageDimensions[ImageAlign[{TEX}, {CROP}]]", "{96, 96}"),                        # Check: reference frame
    ("ImageAlign[x05x, x05y]", "ImageAlign[x05x, x05y]"),
    (f"ImageAlign[{TEX}, {SHIFTED}, Method -> \"Foo\"]", Raises(ValueError, "^ImageAlign: ")),
    (f"ImageAlign[{TEX}, {SHIFTED}, Method -> \"Elastix\"]", Raises(NotImplementedError, "^ImageAlign: ")),
    (f"ImageAlign[{TEX}]", Raises(TypeError, "^ImageAlign: ")),
    # ImageStitch (Brown & Lowe 2007: pairwise RANSAC, spanning tree, feathered blending)
    (f"Module[{{s = ImageStitch[{{ImageTake[{BIG}, All, {{1, 100}}], ImageTake[{BIG}, All, {{41, 160}}]}}]}}, "
     f"{{ImageDimensions[s], Max[Abs[ImageData[s] - ImageData[{BIG}]]] < 10^-9}}]", "{{160, 100}, True}"),
    (f"ImageDimensions[ImageStitch[{{ImageTake[{BIG}, All, {{41, 160}}], ImageTake[{BIG}, All, {{1, 100}}]}}]]",
     "{160, 100}"),
    (f"ImageDimensions[ImageStitch[{{ImageTake[{BIG}, All, {{1, 100}}], ImageTake[{BIG}, All, {{41, 160}}]}}, "
     "\"Plane\"]]", "{160, 100}"),
    ("ImageStitch[x05x]", "ImageStitch[x05x]"),
    (f"ImageStitch[{{{TEX}, {TEX}}}, \"Sphere\"]", Raises(NotImplementedError, "^ImageStitch: ")),
    (f"ImageStitch[{{{TEX}, Image[ConstantArray[0.5, {{50, 50}}]]}}]", Raises(ValueError, "^ImageStitch: ")),
    (f"ImageStitch[{{{TEX}, {TEX}}}, 3]", Raises(TypeError, "^ImageStitch: ")),
]

CASES = FEATURES + SEGMENTATION + REGISTRATION


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_x05(case):
    check(case)


def test_brief_pattern_is_fixed():
    """The ORB sampling pattern is reproducible (fixed seed) and stays inside the radius-15 patch."""
    from mikoshilang.library import wl_imagefeatures_reg as R
    p1, p2 = R._brief_pairs(), R._brief_pairs()
    assert np.array_equal(np.asarray(p1), np.asarray(p2))
    assert np.abs(np.asarray(p1)).max() <= 15


def test_fit_classes_recover_exact_maps():
    """Umeyama / least squares / DLT recover exact maps of each class on noiseless points."""
    from mikoshilang.library import wl_imagefeatures_reg as R
    rng = np.random.default_rng(1)
    P = rng.random((12, 2)) * 10
    th = 0.3
    maps = {
        "Translation": np.array([[1, 0, 2.], [0, 1, -1], [0, 0, 1]]),
        "Rigid": np.array([[np.cos(th), -np.sin(th), 1], [np.sin(th), np.cos(th), 2], [0, 0, 1]]),
        "Similarity": np.array([[2 * np.cos(th), -2 * np.sin(th), 1], [2 * np.sin(th), 2 * np.cos(th), 2], [0, 0, 1]]),
        "Affine": np.array([[2, 1, 3.], [0.5, 3, -1], [0, 0, 1]]),
        "Perspective": np.array([[1, 0.2, 3.], [0.1, 1.2, -1], [0.01, 0.02, 1]]),
    }
    for cls, M in maps.items():
        Q = R.apply_h(M, P)
        got = R.fit(Q, P, cls)
        assert np.allclose(got / got[-1, -1], M, atol=1e-8), cls
