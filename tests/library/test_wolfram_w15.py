"""Differential tests for Wolfram batch W15 (plot, chart, axes, frame, tick, legend and mesh options; module
wl_plot_options).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>), "Details" tables: the documented
  settings of every option (e.g. PlotRange: All, Full, Automatic, max, {min, max}, {{xmin, xmax}, {ymin, ymax}};
  BarOrigin: Bottom, Top, Left, Right; LegendLayout: "Row", "Column", "ReversedRow", "ReversedColumn"); the good
  values below are taken from those tables, the bad values are outside them.
* FindDivisions reference page, Basic Examples (outputs read off the page): FindDivisions[{0, 1}, 5] ==
  {0, 1/5, 2/5, 3/5, 4/5, 1}; FindDivisions[{0.1, 0.9}, 5] gives the same exact list; FindDivisions[{0, 10, Pi}, 5]
  == {0, Pi, 2 Pi, 3 Pi, 4 Pi}; the two-level forms {5, 5}; base 2 {0, 1/4, 1/2, 3/4, 1} and base 3
  {0, 1/3, 2/3, 1}.
* $PerformanceGoal ("Quality" outside Manipulate), $ImageResolution (typical default Automatic), $PlotTheme
  (Automatic) and $DisplayFunction (Identity) reference pages.
* Rendering: pixel read-back of what must be drawn (a frame line, an axis through the origin, a grid line at a given
  coordinate, Epilog above / Prolog below the content).

Symbol names used as symbolic placeholders are prefixed ``w15`` so they cannot collide with other test modules.
"""
import math

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr

try:
    import matplotlib  # noqa: F401
    import numpy  # noqa: F401
    HAVE_MPL = True
except ImportError:  # minimal install
    HAVE_MPL = False

# ───────────── option settings: (documented good settings, invalid settings) ─────────────
OPTS = {
    "PlotRange": (["All", "Full", "Automatic", "3", "{0, 1}", "{{0, 1}, {-1, 2}}", "{{0, 1}, Automatic}",
                   "{0, Automatic}", "{{0, 1}, {0, 1}, {0, 2}}"], ["{1, 0}", "-2", "\"x\"", "{{1, 1}, {0, 1}}"]),
    "ImageSize": (["300", "{300, 200}", "Small", "Large", "UpTo[400]", "{UpTo[300], UpTo[200]}", "Automatic", "Full",
                   "{{100, 300}, {50, 200}}", "{Automatic, 200}"], ["-5", "0", "\"big\"", "{1, 2, 3}"]),
    "ContentSize": (["300", "{400, 300}", "UpTo[400]", "{UpTo[400], Automatic}"], ["-1", "\"x\""]),
    "AspectRatio": (["1", "1/GoldenRatio", "Automatic", "Full", "0.5"], ["0", "-1", "\"x\"", "{1, 2}"]),
    "DataRange": (["Automatic", "All", "{0, 10}", "{{0, 1}, {0, 2}}"], ["{3, 1}", "5", "\"x\""]),
    "PlotRangePadding": (["None", "Automatic", "0.1", "Scaled[0.05]", "{Scaled[0.1], None}", "{{1, 2}, {0, 0}}"],
                         ["\"x\"", "{1, 2, 3, 4}"]),
    "PlotRegion": (["Automatic", "{{0, 1}, {0, 1}}", "{{0.1, 0.9}, {0.2, 0.8}}"], ["{{1, 0}, {0, 1}}", "3"]),
    "PlotRangeClipping": (["True", "False"], ["3", "Automatic"]),
    "ClipRange": (["None", "{{0, 1}, {0, 1}, {0, 0.5}}"], ["{{1, 0}, {0, 1}, {0, 1}}", "3"]),
    "ImageMargins": (["0", "10", "{{1, 2}, {3, 4}}", "Automatic"], ["-1", "\"x\""]),
    "ImagePadding": (["None", "All", "Automatic", "10", "{{10, 5}, {20, 5}}"], ["-3", "\"x\""]),
    "FrameMargins": (["Automatic", "None", "5", "{{1, 2}, {3, 4}}"], ["-5", "\"x\""]),
    "LegendMargins": (["Automatic", "5", "{{1, 2}, {3, 4}}"], ["-5", "\"x\""]),
    "ImageResolution": (["Automatic", "72", "300"], ["0", "-72", "\"x\""]),
    "ImageSizeAction": (["\"Clip\"", "\"ResizeToFit\"", "\"Scrollable\"", "\"ShrinkToFit\""], ["\"Grow\"", "3"]),
    "ImageSizeMultipliers": (["{0.5, 0.25}", "1"], ["{0, 1}", "-1"]),
    "Axes": (["True", "False", "{True, False}", "{False, True, True}"], ["3", "Automatic", "{1, 2}"]),
    "AxesOrigin": (["Automatic", "{0, 0}", "{1, 2, 3}"], ["3", "{1}", "\"x\""]),
    "AxesEdge": (["Automatic", "None", "{{-1, -1}, {1, -1}, None}", "{Automatic, {1, 1}, {-1, 1}}"],
                 ["{{2, 1}, {1, 1}, {1, 1}}", "3"]),
    "AxesLabel": (["None", "\"y\"", "{\"x\", \"y\"}", "{\"x\", \"y\", \"z\"}", "Automatic"],
                  ["{1, 2, 3, 4}", "True"]),
    "AxesStyle": (["Red", "Directive[Red, Thick]", "{Red, Blue}", "{}", "14"], ["True", "{True}"]),
    "Ticks": (["None", "Automatic", "{Automatic, None}", "{{0, 1, 2}, {{0, \"a\"}, {1, \"b\"}}}",
               "{{{0, \"a\", 0.02}}, Automatic}", "{{0, 1}, Automatic, None}"], ["\"x\"", "3"]),
    "TicksStyle": (["Red", "{Red, Blue}", "Directive[Bold, FontSize -> 12]"], ["True"]),
    "TickDirection": (["Automatic", "\"Inward\"", "\"Outward\"", "Left", "{Up, Down}"], ["\"Sideways\"", "3"]),
    "TickLabelOrientation": (["\"Horizontal\"", "\"Vertical\"", "Pi/4", "{\"Parallel\", \"Perpendicular\"}", "None"],
                             ["\"Diagonal\""]),
    "TickLabelPositioning": (["\"Base\"", "\"Tip\"", "Before", "{After, \"Base\"}"], ["\"Middle\"", "3"]),
    "TickLabels": (["Automatic", "All", "None", "{{\"a\", \"b\"}, None}"], ["3", "\"x\""]),
    "TickLengths": (["Automatic", "0.1", "{0.1, 0.05}"], ["-0.1", "\"x\""]),
    "TickPositions": (["Automatic", "5", "{1/2}", "{0, 10, 2}", "{{{0, 1, 5}}}", "{5, {1}}"], ["-2", "\"x\""]),
    "AxisLabel": (["None", "\"x\"", "Placed[\"x\", \"End\"]"], ["True"]),
    "AxisStyle": (["Red", "Directive[Thick, Blue]"], ["3", "\"x\""]),
    "PolarAxes": (["True", "False", "{True, False}", "Automatic"], ["3", "{1, 2}"]),
    "PolarAxesOrigin": (["Automatic", "{Pi/2, 1}", "{{Left, Up}, 2}"], ["3", "{{\"x\", Up}, 2}"]),
    "PolarGridLines": (["None", "Automatic", "{Automatic, {1, 2}}", "{{0, Pi/2}, None}"], ["3", "\"x\""]),
    "PolarTicks": (["None", "Automatic", "{\"Degrees\", Automatic}", "{{0, Pi}, {1, 2}}"], ["3", "{\"Gradians\", None}"]),
    "MultiaxisArrangement": (["None", "All", "{{1, 2}, 3}", "{Right -> {1, 2}, Left -> 3}"], ["3", "{\"a\"}"]),
    "Frame": (["True", "False", "None", "All", "{{True, False}, {True, False}}", "{All, False}", "{-1 -> True}"],
              ["3", "\"x\"", "{1, 2}"]),
    "FrameLabel": (["None", "\"t\"", "{\"x\", \"y\"}", "{{\"l\", \"r\"}, {\"b\", \"t\"}}"],
                   ["{1, 2, 3, 4, 5}", "True"]),
    "FrameStyle": (["Red", "Directive[Thick, Gray]", "{{Red, Blue}, {Green, Black}}"], ["True"]),
    "FrameTicks": (["None", "Automatic", "True", "All", "{{Automatic, None}, {All, None}}",
                    "{{{0, 1}, None}, {{{0, \"a\"}}, None}}"], ["\"x\"", "3", "{{\"a\", None}, {None, None}}"]),
    "FrameTicksStyle": (["Red", "{{Red, Blue}, {Green, Black}}", "Directive[FontSize -> 12]"], ["True"]),
    "RotateLabel": (["True", "False", "\"Vertical\"", "Pi/2"], ["\"Diagonal\"", "{1}"]),
    "GridLines": (["None", "Automatic", "{{1, 2}, None}", "{Automatic, {{0.5, Red}}}"], ["3", "\"x\"", "{1, 2, 3}"]),
    "GridLinesStyle": (["Red", "Directive[Dashed, Gray]", "{Red, Blue}"], ["True"]),
    "FaceGrids": (["None", "All", "{{0, 0, -1}, {-1, 0, 0}}", "{{{0, 0, -1}, {{0.5}, {0.5}}}}"],
                  ["{{0, 1, 1}}", "3", "{{0, 0, 2}}"]),
    "FaceGridsStyle": (["Red", "{Red, Blue, Green}"], ["True"]),
    "PlotLabel": (["None", "\"Title\"", "Sin[x]", "Style[\"T\", 14]"], []),
    "LabelStyle": (["Red", "14", "{Bold, 12}", "\"Text\"", "{FontSize -> 12, FontFamily -> \"Times\"}"],
                   ["-3", "True"]),
    "BaseStyle": (["Red", "{FontSize -> 12}", "\"Graphics\""], ["True", "0"]),
    "DefaultBaseStyle": (["\"Graphics\"", "{}"], ["True"]),
    "DefaultAxesStyle": (["\"GraphicsAxes\"", "{}", "Red"], ["True"]),
    "DefaultFaceGridsStyle": (["\"Graphics3DFaceGrids\"", "{}"], ["True"]),
    "DefaultFrameStyle": (["\"GraphicsFrame\"", "{}"], ["True"]),
    "DefaultFrameTicksStyle": (["\"GraphicsFrameTicks\"", "{}"], ["True"]),
    "DefaultGridLinesStyle": (["\"GraphicsGridLines\"", "{}"], ["True"]),
    "DefaultLabelStyle": (["\"TooltipLabel\"", "{}"], ["True"]),
    "DefaultTicksStyle": (["\"GraphicsTicks\"", "{}"], ["True"]),
    "Epilog": (["{}", "{Red, Disk[]}", "Text[\"x\", {0, 0}]"], ["3", "\"x\""]),
    "Prolog": (["{}", "{Blue, Rectangle[]}"], ["3", "True"]),
    "EpilogFunction": (["None", "(Print[#] &)", "File[\"x.m\"]"], ["3", "\"x\"", "{1}"]),
    "PlotStyle": (["Automatic", "None", "Red", "{Red, Blue}", "Directive[Thick, Dashed]", "{{Red, Thick}, Blue}"],
                  ["3", "\"red\"", "True"]),
    "PlotTheme": (["Automatic", "\"Scientific\"", "{\"Business\", \"Grid\"}", "\"Detailed\"", "None",
                   "{\"BackgroundColor\", Black}"], ["\"Fancy\"", "3", "{\"Web\", \"Nope\"}"]),
    "ColorFunction": (["Automatic", "None", "\"Rainbow\"", "Hue", "(RGBColor[#, 0, 0] &)", "ColorData[\"Rainbow\"]"],
                      ["3", "{Red}", "True"]),
    "ColorFunctionScaling": (["True", "False", "{True, False}"], ["3", "Automatic"]),
    "ColorFunctionBinning": (["None", "Automatic", "5", "\"KMeans\"", "{\"Quantile\", 4}", "{0.5}", "{0, 1, 0.1}",
                              "{{1, 10, 100}}"], ["0", "\"Bins\"", "{-1}"]),
    "ColorRules": (["Automatic", "{1 -> Red, 0 -> White}", "{_?Positive -> Blue}"], ["3", "{Red}"]),
    "ColorCoverage": (["0.5", "{0.2, 0.8}", "1"], ["1.5", "{0.8, 0.2}", "-0.1"]),
    "Filling": (["None", "Axis", "Bottom", "Top", "0.5", "{1 -> {2}}", "{1 -> Axis, 2 -> {Top, Red}}", "1 -> 0",
                 "{1 -> {Axis, {Red, Blue}}}"], ["\"x\"", "{0 -> Axis}", "True"]),
    "FillingStyle": (["Automatic", "Red", "Opacity[0.3]", "{Red, Blue}"], ["3", "True"]),
    "Joined": (["True", "False", "{True, False}"], ["3", "Automatic"]),
    "Mesh": (["None", "Automatic", "All", "Full", "5", "{3, 4}", "{{0.5, 1.5}}", "{{{0.5, Red}}}"],
             ["-1", "\"x\"", "1.5"]),
    "MeshFunctions": (["Automatic", "{#1 &, #2 &}", "{Sin[#1] &}"], ["3", "\"x\""]),
    "MeshStyle": (["Automatic", "None", "Red", "{Red, Blue}", "Directive[Thick]"], ["3", "True"]),
    "MeshShading": (["None", "Automatic", "{Red, Blue}", "{{Red, None}, {None, Blue}}"], ["3", "\"x\""]),
    "BoundaryStyle": (["Automatic", "None", "Red", "Directive[Thick, Black]"], ["3", "True"]),
    "ClippingStyle": (["Automatic", "None", "Red", "{Red, Blue}"], ["3", "True"]),
    "ExclusionsStyle": (["None", "Automatic", "Red", "{Red, Blue}"], ["3", "True"]),
    "Exclusions": (["None", "All", "Automatic", "\"Discontinuities\"", "x == 0", "{x == 0, x == 1}",
                    "{{Sin[x] == 0, x > 0}}", "{0, 1}"], ["\"Holes\"", "True"]),
    "RegionFunction": (["(True &)", "Function[{x, y}, x^2 + y^2 < 1]"], ["3", "\"x\"", "True"]),
    "RegionBoundaryStyle": (["Automatic", "None", "Red"], ["3", "True"]),
    "RegionFillingStyle": (["Automatic", "None", "LightBlue"], ["3", "True"]),
    "ReferenceLineStyle": (["None", "Red", "{Red, Dashed}", "Directive[Gray, Dashed]"], ["3", "True"]),
    "Contours": (["Automatic", "10", "{Automatic, 5}", "{0, 0.5, 1}", "{{0.5, Red}}", "(Range[#1, #2, 0.1] &)", "0"],
                 ["-3", "\"x\"", "True"]),
    "ContourStyle": (["Automatic", "None", "Red", "{Red, Blue}"], ["3", "True"]),
    "ContourLabels": (["None", "Automatic", "True", "All", "(Text[#3, {#1, #2}] &)", "{(#3 &), (#3 &)}"],
                      ["3", "\"x\""]),
    "PerformanceGoal": (["\"Speed\"", "\"Quality\"", "Automatic", "$PerformanceGoal"], ["\"Fast\"", "3"]),
    "PlotPoints": (["Automatic", "50", "{30, 20}"], ["0", "-1", "1.5", "\"x\""]),
    "MaxRecursion": (["Automatic", "0", "5", "Infinity"], ["-1", "1.5"]),
    "MaxPlotPoints": (["Automatic", "Infinity", "1000"], ["0", "-5"]),
    "PlotMarkers": (["None", "Automatic", "\"OpenMarkers\"", "{Automatic, Small}", "{\"*\", 10}",
                     "Graphics[{Disk[]}]"], ["3", "True"]),
    "ScalingFunctions": (["None", "\"Log\"", "\"Log10\"", "{\"Reverse\", \"Log\"}", "{\"SignedLog\", 2}",
                          "{Log, Exp}", "{None, \"Reciprocal\"}"], ["\"Cubic\"", "3", "{\"Log\", \"Bogus\"}"]),
    "ReImLabels": (["Automatic", "\"Text\"", "\"Gothic\"", "{\"re\", \"im\"}"], ["\"Roman\"", "3"]),
    "ReImStyle": (["Automatic", "{Red, Dashed}", "{{Red, Blue}, {Green, Black}}"], ["3", "{Red}"]),
    "TernaryPlotCorners": (["Automatic", "\"Clockwise\"", "{{0, 0}, {1, 0}, {0, 1}}"], ["3", "{{0, 0}, {1, 0}}"]),
    "PlotLegends": (["None", "Automatic", "\"Expressions\"", "{\"a\", \"b\"}", "SwatchLegend[{Red}, {\"a\"}]",
                     "Placed[{\"a\"}, Below]", "\"Custom\""], ["True", "False"]),
    "ChartLegends": (["None", "Automatic", "{\"a\", \"b\"}", "Placed[{\"a\"}, Above]"], ["True"]),
    "ImageLegends": (["None", "Automatic", "\"Expressions\"", "{\"a\"}"], ["True"]),
    "PlotLabels": (["None", "Automatic", "\"Expressions\"", "{\"a\", \"b\"}", "Placed[{\"a\"}, Above]"],
                   ["3", "True", "\"x\""]),
    "ImageLabels": (["None", "Automatic", "{\"a\"}", "Placed[{\"a\"}, Below]"], ["3", "True"]),
    "ChartLabels": (["None", "Automatic", "{\"a\", \"b\"}", "Placed[{\"a\", \"b\"}, Above]"], ["3", "\"x\""]),
    "EventLabels": (["None", "Automatic", "{\"e1\" -> \"x\"}", "Placed[{\"a\"}, Above]"], ["3", "True"]),
    "ResidueLabels": (["None", "Automatic", "{\"ChainNames\", \"ResidueIndices\"}", "Placed[{\"ChainNames\"}, Tooltip]"],
                      ["{\"Colour\"}", "3"]),
    "LabelingFunction": (["None", "Automatic", "Top", "Tooltip", "(Placed[#1, Above] &)"], ["3", "{1}"]),
    "LabelingSize": (["Automatic", "All", "Full", "50", "{100, 30}"], ["0", "-1", "\"x\""]),
    "LabelingTarget": (["Automatic", "All", "None", "0.5", "\"Dense\"", "<|\"LabelingFraction\" -> 0.5|>"],
                       ["2", "\"Crowded\""]),
    "LabelVisibility": (["All", "None", "Automatic", "Infinity", "-Infinity", "3"], ["\"x\"", "{1}"]),
    "LeaderSize": (["Automatic", "10", "{10, Pi/4, 2}", "{{10, Pi/4, 2}, {5, 0}}"], ["-1", "\"x\""]),
    "LegendAppearance": (["Automatic", "\"Row\"", "\"Column\""], ["3", "True"]),
    "LegendFunction": (["Identity", "None", "\"Frame\"", "\"Panel\"", "(Framed[#] &)"], ["3", "True"]),
    "LegendLabel": (["None", "\"Legend\"", "Placed[\"L\", Top]"], ["True"]),
    "LegendLayout": (["Automatic", "\"Row\"", "\"Column\"", "\"ReversedRow\"", "\"ReversedColumn\"", "{\"Column\", 2}",
                      "(Column[#] &)"], ["\"Grid\"", "3", "{\"Row\", 0}"]),
    "LegendMarkerSize": (["Automatic", "10", "{20, 10}", "{10, 20, 30}"], ["0", "-1", "\"x\""]),
    "LegendMarkers": (["None", "Automatic", "\"Bubble\"", "{Graphics[{Disk[]}]}", "Graphics[{Disk[]}]"], ["3", "True"]),
    "ChartStyle": (["Red", "{Red, Blue}", "24", "\"Pastel\"", "Directive[EdgeForm[None], Opacity[0.5]]"],
                   ["-1", "True"]),
    "ChartBaseStyle": (["Red", "EdgeForm[None]"], ["3", "True"]),
    "ChartElementFunction": (["Automatic", "\"GlassRectangle\"", "(Rectangle @@ Transpose[#1] &)"], ["3", "True"]),
    "ChartElements": (["None", "Automatic", "Graphics[{Disk[]}]", "{Graphics[{Disk[]}], All}"], ["3", "\"x\""]),
    "ChartLayout": (["Automatic", "\"Stacked\"", "\"Percentile\"", "{\"Column\", 2}", "{\"Row\", UpTo[3]}"],
                    ["3", "True"]),
    "BarOrigin": (["Bottom", "Top", "Left", "Right"], ["Center", "3", "\"Bottom\""]),
    "BarSpacing": (["Automatic", "None", "Small", "0.5", "{0.1, 1}", "{Small, Large}"], ["-1", "\"x\""]),
    "BubbleScale": (["\"Area\"", "\"Diameter\"", "\"Volume\""], ["\"Radius\"", "3"]),
    "BubbleSizes": (["Automatic", "Large", "{0.05, 0.1}", "{Automatic, 0.2}"], ["{0.2, 0.1}", "{-1, 0.1}", "3"]),
    "SectorOrigin": (["Automatic", "Top", "Pi/2", "{Pi/2, \"Clockwise\"}", "{{Pi/2, -1}, 0.5}"],
                     ["\"x\"", "{Pi/2, 2}"]),
    "SectorSpacing": (["Automatic", "None", "Small", "0.2", "{0.1, 0.2}"], ["-1", "\"x\""]),
    "ExtentElementFunction": (["Automatic", "\"Rectangle\"", "(Rectangle @@ Transpose[#1] &)"], ["3", "True"]),
    "ExtentMarkers": (["None", "\"Filled\"", "\"Empty\"", "{\"Filled\", \"Empty\"}", "Graphics[{Disk[]}]"],
                      ["\"Square\"", "3"]),
    "ExtentSize": (["None", "Full", "All", "Left", "Right", "0.5", "Scaled[0.3]", "{0.1, 0.2}"], ["-1", "\"x\""]),
    "LayerSizeFunction": (["(#^2 &)", "Sqrt"], ["3", "\"x\"", "True"]),
    "RulerUnits": (["\"Inches\"", "\"Centimeters\"", "\"Points\""], ["\"Feet\"", "3"]),
    "MaxOverlapFraction": (["0.5", "0", "1", "Automatic"], ["1.5", "-0.5"]),
    "ScaleDivisions": (["None", "Automatic", "10", "{5, 2}"], ["0", "-3", "{5, 0}"]),
    "ScaleOrigin": (["Automatic", "0.5", "Left", "{0, 1}", "{Left, 0.5}"], ["\"x\"", "{1, 2, 3}"]),
    "ScalePadding": (["Automatic", "None", "Full", "All", "0.1", "Scaled[0.1]", "{0.1, 0.2}", "{{0.1, 0.2}, None}"],
                     ["\"x\"", "True"]),
    "ScaleRanges": (["None", "All", "{0, 10}", "{{0, 5} -> Red, {5, 10} -> Blue}"], ["3", "\"x\""]),
    "ScaleRangeStyle": (["None", "Automatic", "Red", "{Red, Blue}", "3", "\"Rainbow\""], ["-1", "True"]),
    "VectorAspectRatio": (["Automatic", "0.3"], ["0", "-1", "\"x\""]),
    "VectorColorFunction": (["Automatic", "None", "\"Rainbow\"", "Hue"], ["3", "True"]),
    "VectorColorFunctionScaling": (["True", "False", "{True, False}"], ["3"]),
    "VectorMarkers": (["None", "Automatic", "\"Arrow\"", "\"Segment\"", "\"Start\"", "{\"Arrow\", \"Dart\"}",
                       "Placed[\"Arrow\", \"Middle\"]"], ["3", "True"]),
    "VectorPoints": (["None", "Automatic", "Coarse", "Fine", "10", "{10, 20}", "\"Hexagonal\"", "{\"Regular\", 5}",
                      "{{0, 0}, {1, 1}}"], ["0", "-1", "\"Spiral\""]),
    "VectorRange": (["Automatic", "All", "{0, 10}"], ["{10, 0}", "{-1, 1}", "3"]),
    "VectorScaling": (["Automatic", "None", "\"Linear\"", "\"Log\"", "\"Sqrt\""], ["\"Cubic\"", "3"]),
    "VectorSizes": (["Automatic", "Small", "0.5", "{0.2, 0.8}", "{Automatic, 0.5}"], ["0", "{0.8, 0.2}", "\"x\""]),
    "VectorStyle": (["Red", "{Red, Blue}", "Arrowheads[0.02]"], ["3", "True"]),
    "StreamColorFunction": (["Automatic", "None", "\"Rainbow\"", "Hue"], ["3", "True"]),
    "StreamColorFunctionScaling": (["True", "False", "{False, True}"], ["3"]),
    "StreamMarkers": (["None", "Automatic", "\"Arrow\"", "\"Toothpick\"", "{\"Arrow\", \"Dot\"}"], ["3", "True"]),
    "StreamPoints": (["None", "Automatic", "Coarse", "Fine", "10", "{{0, 0}, {1, 1}}", "{Automatic, 0.1}"],
                     ["0", "-3", "\"x\""]),
    "StreamScale": (["Automatic", "None", "Full", "Large", "0.1", "{0.1, 3}", "{Full, All, 0.05}"], ["-1", "\"x\""]),
    "StreamStyle": (["Red", "{Red, Blue}", "Arrowheads[0.02]"], ["3", "True"]),
    "LineIntegralConvolutionScale": (["Automatic", "Small", "0.5", "Scaled[0.1]"], ["0", "-1", "\"x\""]),
    "DateTicksFormat": (["Automatic", "{\"MonthNameShort\", \" \", \"Year\"}", "\"Year\""], ["3", "True"]),
}

GRAPHICS_2D = {"AspectRatio", "Axes", "AxesLabel", "AxesOrigin", "AxesStyle", "BaseStyle", "Epilog", "Frame",
               "FrameLabel", "FrameStyle", "FrameTicks", "FrameTicksStyle", "GridLines", "GridLinesStyle",
               "ImageMargins", "ImagePadding", "ImageSize", "LabelStyle", "PlotLabel", "PlotRange", "PlotRangeClipping",
               "PlotRangePadding", "PlotRegion", "Prolog", "RotateLabel", "Ticks", "TicksStyle"}
GRAPHICS_3D_ONLY = {"AxesEdge", "FaceGrids", "FaceGridsStyle"}

OPTION_CASES = []
for _name, (_good, _bad) in OPTS.items():
    OPTION_CASES += [
        (_name, _name),                                                   # inert symbol (ref/<Name>)
        (f"{_name} -> {_good[0]}", f"{_name} -> {_good[0]}"),               # inert rule
        (f"{{{_name} -> {_good[-1]}}}", f"{{{_name} -> {_good[-1]}}}"),
        (f"{_name}[w15arg]", f"{_name}[w15arg]"),                            # Name[...] stays unevaluated
    ]
    if _name in GRAPHICS_2D or _name in GRAPHICS_3D_ONLY:
        _box = "Graphics3D" if _name in GRAPHICS_3D_ONLY else "Graphics"
        for _g in _good[:3]:
            OPTION_CASES.append((f"{_box}[{{}}, {_name} -> {_g}]", f"{_box}[{{}}, {_name} -> {_g}]"))
        for _b in _bad:
            OPTION_CASES.append((f"{_box}[{{}}, {_name} -> {_b}]", Raises(ValueError)))
        # a setting containing a user symbol stays unevaluated
        if _name not in ("PlotLabel", "AxesLabel", "FrameLabel", "Epilog", "Prolog", "Ticks", "FrameTicks",
                         "GridLines", "BaseStyle", "LabelStyle", "AxesStyle", "TicksStyle", "FrameStyle",
                         "FrameTicksStyle", "GridLinesStyle", "FaceGridsStyle"):
            OPTION_CASES.append((f"{_box}[{{}}, {_name} -> w15sym]", f"{_box}[{{}}, {_name} -> w15sym]"))

# options used by other heads (Raster3D ClipRange, Callout LeaderSize/LabelVisibility, GraphicsGrid Frame)
OPTION_CASES += [
    ("Callout[Point[{0, 0}], \"x\", LeaderSize -> 10]", "Callout[Point[{0, 0}], \"x\", LeaderSize -> 10]"),
    ("Callout[Point[{0, 0}], \"x\", LeaderSize -> -10]", Raises(ValueError)),
    ("Callout[Point[{0, 0}], \"x\", LabelVisibility -> All]", "Callout[Point[{0, 0}], \"x\", LabelVisibility -> All]"),
    ("Callout[Point[{0, 0}], \"x\", LabelVisibility -> \"x\"]", Raises(ValueError)),
    ("GraphicsGrid[{{Graphics[{}]}}, Frame -> All]", "GraphicsGrid[{{Graphics[{}]}}, Frame -> All]"),
    ("GraphicsGrid[{{Graphics[{}]}}, Frame -> {All, False}]", "GraphicsGrid[{{Graphics[{}]}}, Frame -> {All, False}]"),
    ("GraphicsGrid[{{Graphics[{}]}}, Frame -> 3]", Raises(ValueError)),
    ("Graphics[{}, Epilog -> {3}]", Raises(TypeError)),                    # a number is not a primitive
    ("Graphics[{}, ImageResolution -> 72]", Raises(ValueError)),          # not a Graphics option
    ("Graphics[{}, PlotTheme -> \"Scientific\"]", Raises(ValueError)),     # plot option, not a Graphics option
]

# ───────────── system variables and Axis ─────────────
SYSTEM = [
    ("$PlotTheme", "Automatic"),                               # ref/$PlotTheme
    ("$PerformanceGoal", "\"Quality\""),                       # ref/$PerformanceGoal (outside Manipulate)
    ("$DisplayFunction", "Identity"),                          # ref/$DisplayFunction
    ("$ImageResolution", "Automatic"),                         # ref/$ImageResolution "typical default"
    ("Block[{$PlotTheme = \"Web\"}, $PlotTheme]", "\"Web\""),
    ("Block[{$PerformanceGoal = \"Speed\"}, $PerformanceGoal]", "\"Speed\""),
    ("Block[{$ImageResolution = 50}, $ImageResolution]", "50"),
    ("Block[{$DisplayFunction = Print}, $DisplayFunction]", "Print"),
    ("PerformanceGoal -> $PerformanceGoal", "PerformanceGoal -> \"Quality\""),
    ("PlotTheme -> $PlotTheme", "PlotTheme -> Automatic"),
    ("{$DisplayFunction, $ImageResolution}", "{Identity, Automatic}"),
    ("ImageResolution -> $ImageResolution", "ImageResolution -> Automatic"),
    ("Axis", "Axis"), ("Filling -> Axis", "Filling -> Axis"), ("{Axis, Top}", "{Axis, Top}"),
    ("Axis[1]", "Axis[1]"),
]

# ───────────── legends ─────────────
LEGENDS = [
    ("SwatchLegend[{Red, Blue}, {\"a\", \"b\"}]", "SwatchLegend[{Red, Blue}, {\"a\", \"b\"}]"),
    ("SwatchLegend[{Red, Blue}, Automatic]", "SwatchLegend[{Red, Blue}, Automatic]"),
    ("SwatchLegend[{\"a\", \"b\"}]", "SwatchLegend[{\"a\", \"b\"}]"),
    ("SwatchLegend[\"Rainbow\", {\"a\", \"b\"}]", "SwatchLegend[\"Rainbow\", {\"a\", \"b\"}]"),
    ("SwatchLegend[{Red}, {\"a\"}, LegendLayout -> \"Row\", LegendLabel -> \"L\"]",
     "SwatchLegend[{Red}, {\"a\"}, LegendLayout -> \"Row\", LegendLabel -> \"L\"]"),
    ("SwatchLegend[{Red, Blue}, {\"a\"}]", Raises(ValueError)),              # one label per style
    ("SwatchLegend[{Red, 3}, {\"a\", \"b\"}]", Raises(TypeError)),           # 3 is not a colour
    ("SwatchLegend[{Red}, {\"a\"}, LegendLayout -> \"Grid\"]", Raises(ValueError)),
    ("SwatchLegend[{Red}, {\"a\"}, Joined -> True]", Raises(ValueError)),    # not a SwatchLegend option
    ("SwatchLegend[\"a\"]", Raises(TypeError)),
    ("SwatchLegend[w15cols, {\"a\"}]", "SwatchLegend[w15cols, {\"a\"}]"),    # symbolic: unevaluated
    ("LineLegend[{Red, Directive[Blue, Dashed]}, {\"sin\", \"cos\"}]",
     "LineLegend[{Red, Directive[Blue, Dashed]}, {\"sin\", \"cos\"}]"),
    ("LineLegend[{\"a\", \"b\"}, Joined -> False]", "LineLegend[{\"a\", \"b\"}, Joined -> False]"),
    ("LineLegend[Automatic, {\"a\", \"b\"}, LegendMarkers -> Automatic]",
     "LineLegend[Automatic, {\"a\", \"b\"}, LegendMarkers -> Automatic]"),
    ("LineLegend[{Red}, {\"a\"}, Joined -> 3]", Raises(ValueError)),
    ("LineLegend[{Red}, {\"a\"}, LegendMarkerSize -> -1]", Raises(ValueError)),
    ("LineLegend[{Red}, 3]", Raises(TypeError)),
    ("LineLegend[w15lbl]", "LineLegend[w15lbl]"),
    ("PointLegend[{Red, Blue}, {\"a\", \"b\"}, LegendMarkerSize -> 10]",
     "PointLegend[{Red, Blue}, {\"a\", \"b\"}, LegendMarkerSize -> 10]"),
    ("PointLegend[{Red, Blue}, {\"a\", \"b\"}, Joined -> True, LegendFunction -> \"Frame\"]",
     "PointLegend[{Red, Blue}, {\"a\", \"b\"}, Joined -> True, LegendFunction -> \"Frame\"]"),
    ("PointLegend[{\"a\"}]", "PointLegend[{\"a\"}]"),
    ("PointLegend[{Red}, {\"a\", \"b\"}]", Raises(ValueError)),
    ("PointLegend[{Red}, {\"a\"}, LegendMargins -> -2]", Raises(ValueError)),
    ("PointLegend[{Red}, {\"a\"}, {\"b\"}]", Raises(ValueError)),
    ("BarLegend[\"Rainbow\"]", "BarLegend[\"Rainbow\"]"),
    ("BarLegend[{\"Rainbow\", {0, 1}}]", "BarLegend[{\"Rainbow\", {0, 1}}]"),
    ("BarLegend[{Hue, {0, 1}}, 10]", "BarLegend[{Hue, {0, 1}}, 10]"),
    ("BarLegend[Automatic, {0, 0.5, 1}, LegendLabel -> \"z\"]", "BarLegend[Automatic, {0, 0.5, 1}, LegendLabel -> \"z\"]"),
    ("BarLegend[{\"Rainbow\", {1, 0}}]", Raises(ValueError)),                 # min < max
    ("BarLegend[3]", Raises(TypeError)),
    ("BarLegend[\"Rainbow\", -2]", Raises(ValueError)),
    ("BarLegend[\"Rainbow\", LegendLayout -> \"Diagonal\"]", Raises(ValueError)),
    ("BarLegend[w15cf]", "BarLegend[w15cf]"),
    ("Legended[Graphics[{}], SwatchLegend[{Red}, {\"a\"}]]", "Legended[Graphics[{}], SwatchLegend[{Red}, {\"a\"}]]"),
    ("PlotLegends -> LineLegend[{Red}, {\"a\"}]", "PlotLegends -> LineLegend[{Red}, {\"a\"}]"),
    ("Placed[PointLegend[{Red}, {\"a\"}], Below]", "Placed[PointLegend[{Red}, {\"a\"}], Below]"),
]

# ───────────── AxisObject ─────────────
AXISOBJ = [
    ("AxisObject[{{0, 0}, {1, 0}}]", "AxisObject[{{0, 0}, {1, 0}}]"),
    ("AxisObject[{\"Horizontal\", 0}, {0, 10}]", "AxisObject[{\"Horizontal\", 0}, {0, 10}]"),
    ("AxisObject[{\"Vertical\", 1, {0, 2}}]", "AxisObject[{\"Vertical\", 1, {0, 2}}]"),
    ("AxisObject[Line[{{-2, -3}, {2, -3}}], {0, 10}, TickDirection -> \"Inward\"]",
     "AxisObject[Line[{{-2, -3}, {2, -3}}], {0, 10}, TickDirection -> \"Inward\"]"),
    ("AxisObject[Line[{{0, 0}, {1, 1}}], {0 -> 1, 1 -> 2}, AxisLabel -> \"t\", TickPositions -> 4]",
     "AxisObject[Line[{{0, 0}, {1, 1}}], {0 -> 1, 1 -> 2}, AxisLabel -> \"t\", TickPositions -> 4]"),
    ("Graphics[{AxisObject[{\"Horizontal\", 0}, {0, 10}]}]", "Graphics[{AxisObject[{\"Horizontal\", 0}, {0, 10}]}]"),
    ("AxisObject[{\"Diagonal\", 0}]", Raises(ValueError)),
    ("AxisObject[{{0, 0}, {0, 0}}]", Raises(ValueError)),                   # the two points must differ
    ("AxisObject[3]", Raises(TypeError)),
    ("AxisObject[{{0, 0}, {1, 0}}, {1, 1}]", Raises(ValueError)),           # scale values must differ
    ("AxisObject[{{0, 0}, {1, 0}}, TickDirection -> \"Sideways\"]", Raises(ValueError)),
    ("AxisObject[{{0, 0}, {1, 0}}, PlotRange -> All]", Raises(ValueError)),  # not an AxisObject option
    ("Graphics3D[{AxisObject[{{0, 0}, {1, 0}}]}]", Raises(ValueError)),       # 2D primitive
    ("AxisObject[w15path]", "AxisObject[w15path]"),
]
if HAVE_MPL:
    AXISOBJ.append(("Rasterize[Graphics[{AxisObject[{\"Horizontal\", 0}, {0, 10}]}], \"Data\"]",
                    Raises(NotImplementedError)))


# ───────────── FindDivisions (ref/FindDivisions, Basic Examples) ─────────────
def _divisions_ok(g):
    """Check: equally spaced, nice step (1, 2 or 5 times a power of ten)."""
    vals = [float(ev("N[%s]" % _k(a))) for a in g.args]
    steps = [b - a for a, b in zip(vals, vals[1:])]
    s = steps[0]
    m = s / 10 ** math.floor(math.log10(s))
    return all(abs(t - s) < 1e-9 * s for t in steps) and any(abs(m - c) < 1e-9 for c in (1, 2, 5))


def _k(e):
    from mikoshilang.library.wl_graphics_color import K_input
    return K_input(e)


FINDDIV = [
    ("FindDivisions[{0, 1}, 5]", "{0, 1/5, 2/5, 3/5, 4/5, 1}"),
    ("FindDivisions[{0.1, 0.9}, 5]", "{0, 1/5, 2/5, 3/5, 4/5, 1}"),
    ("FindDivisions[{0, 1}, {5, 5}]",
     "{{0, 1/5, 2/5, 3/5, 4/5, 1}, {{0, 1/20, 1/10, 3/20, 1/5}, {1/5, 1/4, 3/10, 7/20, 2/5}, "
     "{2/5, 9/20, 1/2, 11/20, 3/5}, {3/5, 13/20, 7/10, 3/4, 4/5}, {4/5, 17/20, 9/10, 19/20, 1}}}"),
    ("FindDivisions[{0, 10, Pi}, 5]", "{0, Pi, 2 Pi, 3 Pi, 4 Pi}"),
    ("N[FindDivisions[{0, 10, Pi}, 5]]", Pred(lambda g: all(abs(float(x) - k * math.pi) < 1e-12 for k, x in
                                                      enumerate(g.args)) and len(g.args) == 5, "k Pi, k = 0..4")),
    ("FindDivisions[{0, 10, {Pi, Pi/4}}, {5, 5}]",
     "{{0, Pi, 2 Pi, 3 Pi, 4 Pi}, {{0, Pi/4, Pi/2, 3 Pi/4, Pi}, {Pi, 5 Pi/4, 3 Pi/2, 7 Pi/4, 2 Pi}, "
     "{2 Pi, 9 Pi/4, 5 Pi/2, 11 Pi/4, 3 Pi}, {3 Pi, 13 Pi/4, 7 Pi/2, 15 Pi/4, 4 Pi}}}"),
    ("FindDivisions[{0, 1}, 5, 2]", "{0, 1/4, 1/2, 3/4, 1}"),
    ("FindDivisions[{0, 1}, 5, 3]", "{0, 1/3, 2/3, 1}"),
    ("N[FindDivisions[{0, 1}, 5, 10]]", "{0., 0.2, 0.4, 0.6, 0.8, 1.}"),
    # nice-number cases checkable by hand: [0, 10] in 5 parts of 2; [0, 100] in 10 parts of 10
    ("FindDivisions[{0, 10}, 5]", "{0, 2, 4, 6, 8, 10}"),
    ("FindDivisions[{0, 100}, 10]", "Range[0, 100, 10]"),
    ("FindDivisions[{-1, 1}, 4]", "{-1, -1/2, 0, 1/2, 1}"),
    # Check: equally spaced nice steps that enclose the interval
    ("FindDivisions[{-3.7, 12.2}, 6]", Pred(_divisions_ok, "nice equal steps")),
    ("FindDivisions[{1234, 5678}, 7]", Pred(_divisions_ok, "nice equal steps")),
    ("With[{d = FindDivisions[{0.37, 2.91}, 8]}, First[d] <= 0.37 && Last[d] >= 2.91]", "True"),
    ("With[{d = FindDivisions[{-17, 3}, 3]}, First[d] <= -17 && Last[d] >= 3]", "True"),
    # symbolic and errors
    ("FindDivisions[{w15a, 1}, 5]", "FindDivisions[{w15a, 1}, 5]"),
    ("FindDivisions[{0, 1}, w15n]", "FindDivisions[{0, 1}, w15n]"),
    ("FindDivisions[{1, 0}, 5]", Raises(NotImplementedError)),
    ("FindDivisions[{0, 1}, 0]", Raises(ValueError)),
    ("FindDivisions[{0, 1, -1}, 5]", Raises(ValueError)),
    ("FindDivisions[{0, 1}, 5, 1]", Raises(ValueError)),
    ("FindDivisions[{0, 1, {1, 2}}, {5}]", Raises(ValueError)),
    ("FindDivisions[{0, 1}, 2.5]", Raises(TypeError)),
    ("FindDivisions[3, 5]", Raises(TypeError)),
    ("FindDivisions[{\"a\", 1}, 5]", Raises(TypeError)),
]


# ───────────── rendering (pixel read-back) ─────────────
def _px(src):
    from mikoshilang.library.wl_graphics_render import raster_array
    return raster_array(ev(src), raster_size=(100, 100))


def _dark(p):
    return int(p[0]) + int(p[1]) + int(p[2]) < 300


RASTER = []
if HAVE_MPL:
    def _arr(g):
        import numpy as np
        return np.array([[[int(c) for c in px.args] for px in r.args] for r in g.args])

    def _dark_cols(g):
        """Columns in which more than half of the pixels are dark (a vertical line across the image)."""
        import numpy as np
        a = _arr(g)
        return set(np.nonzero((a.sum(2) < 384).sum(0) > a.shape[0] // 2)[0].tolist())

    def _dark_rows(g):
        import numpy as np
        a = _arr(g)
        return set(np.nonzero((a.sum(2) < 384).sum(1) > a.shape[1] // 2)[0].tolist())

    def _frame_drawn(g):
        cols, rows = _dark_cols(g), _dark_rows(g)
        n = len(g.args)
        return any(c < n // 4 for c in cols) and any(c > 3 * n // 4 for c in cols) and \
            any(r < n // 4 for r in rows) and any(r > 3 * n // 4 for r in rows)

    def _left_edge_only(g):
        cols, rows = _dark_cols(g), _dark_rows(g)
        n = len(g.args)
        return any(c < n // 4 for c in cols) and not any(c > 3 * n // 4 for c in cols) and not rows

    def _center_col_dark(g):
        n = len(g.args)
        return any(abs(c - n / 2) <= 3 for c in _dark_cols(g))

    def _all_white(g):
        return bool((_arr(g) > 250).all())

    def _top_rows_have_ink(g):
        return bool((_arr(g)[:40] < 200).any())

    def _rgb_near(want, tol=40):
        return Pred(lambda g: all(abs(int(c) - w) <= tol for c, w in zip(g.args, want)), f"~{want}")

    D = "Rasterize[Graphics[%s], \"Data\"]"
    RASTER += [
        # no decoration: blank; with Frame: a dark line near each edge
        (D % "{}, PlotRange -> {{0, 1}, {0, 1}}", Pred(_all_white, "all white")),
        (D % "{}, PlotRange -> {{0, 1}, {0, 1}}, Frame -> True, FrameTicks -> None", Pred(_frame_drawn, "frame")),
        (D % "{}, PlotRange -> {{0, 1}, {0, 1}}, Frame -> True", Pred(_frame_drawn, "frame with ticks")),
        (D % "{}, PlotRange -> {{0, 1}, {0, 1}}, Frame -> {{True, False}, {False, False}}, FrameTicks -> None",
         Pred(_left_edge_only, "left frame edge only")),
        # Axes through the origin: the centre column is dark (x = 0 in the middle of {-1, 1})
        (D % "{}, PlotRange -> {{-1, 1}, {-1, 1}}, Axes -> True, Ticks -> None, AxesStyle -> Black",
         Pred(_center_col_dark, "y axis at x = 0")),
        (D % "{}, PlotRange -> {{-1, 1}, {-1, 1}}, Axes -> {False, True}, AxesOrigin -> {0, 0}, Ticks -> None, "
             "AxesStyle -> Black", Pred(_center_col_dark, "y axis only")),
        (D % "{}, PlotRange -> {{-1, 1}, {-1, 1}}, Axes -> {True, False}, Ticks -> None",
         Pred(lambda g: not _center_col_dark(g), "x axis only")),
        # GridLines at x = 0.5 of [0, 1] (no padding): a dark line in the centre column, or a red one
        (D % "{}, PlotRange -> {{0, 1}, {0, 1}}, PlotRangePadding -> None, GridLines -> {{0.5}, None}, "
             "GridLinesStyle -> Directive[Black, AbsoluteThickness[3]]", Pred(_center_col_dark, "grid line")),
        ("Rasterize[Graphics[{}, PlotRange -> {{0, 1}, {0, 1}}, PlotRangePadding -> None, GridLines -> {{{0.5, "
         "Directive[Red, AbsoluteThickness[4]]}}, None}], \"Data\"][[90, 180]]", "{255, 0, 0}"),
        # Epilog drawn above the content, Prolog below it
        ("Rasterize[Graphics[{Blue, Rectangle[{-1, -1}, {1, 1}]}, Epilog -> {Red, Disk[{0, 0}, 0.5]}], \"Data\", "
         "RasterSize -> 100][[50, 50]]", "{255, 0, 0}"),
        ("Rasterize[Graphics[{Blue, Disk[{0, 0}, 0.5]}, Prolog -> {Red, Rectangle[{-1, -1}, {1, 1}]}], \"Data\", "
         "RasterSize -> 100][[50, 50]]", "{0, 0, 255}"),
        ("Rasterize[Graphics[{Blue, Disk[{0, 0}, 0.5]}, Prolog -> {Red, Rectangle[{-1, -1}, {1, 1}]}], \"Data\", "
         "RasterSize -> 100][[5, 5]]", "{255, 0, 0}"),
        ("Rasterize[Graphics[{Blue, Disk[]}, Prolog -> {Red, Disk[]}, Epilog -> {Green, Disk[{0, 0}, 0.2]}], "
         "\"Data\", RasterSize -> 100][[50, 50]]", "{0, 255, 0}"),
        # labels add ink in the margins; the raster size is unchanged
        (D % "{}, PlotRange -> {{0, 1}, {0, 1}}, PlotLabel -> \"Title\"", Pred(_top_rows_have_ink, "plot label")),
        (D % "{}, PlotRange -> {{0, 1}, {0, 1}}, Frame -> True, FrameTicks -> None, PlotLabel -> \"Title\"",
         Pred(_frame_drawn, "frame below the label")),
        ("Rasterize[Graphics[{Disk[]}, Frame -> True, FrameLabel -> {\"x\", \"y\"}, PlotLabel -> \"t\", "
         "GridLines -> Automatic], \"RasterSize\", RasterSize -> {120, 80}]", "{120, 80}"),
        ("Rasterize[Graphics[{Disk[]}, Axes -> True, AxesLabel -> {\"x\", \"y\"}, LabelStyle -> {Bold, 8}], "
         "\"RasterSize\", RasterSize -> 90]", "{90, 90}"),
        ("Rasterize[Graphics[{Disk[]}, Axes -> True, Ticks -> {{-1, {0.5, \"half\"}}, Automatic}], \"RasterSize\", "
         "RasterSize -> 60]", "{60, 60}"),
        ("Rasterize[Graphics[{Disk[]}, Frame -> True, FrameTicks -> {{{0, 1}, None}, {All, None}}, "
         "FrameTicksStyle -> Red, RotateLabel -> False, FrameLabel -> {{\"l\", None}, {\"b\", None}}], "
         "\"RasterSize\", RasterSize -> 80]", "{80, 80}"),
        # unsupported settings raise rather than being left out
        ("Rasterize[Graphics[{Disk[]}, Axes -> True, Ticks -> ({#1} &)], \"Data\"]", Raises(NotImplementedError)),
        ("Rasterize[Graphics[{Disk[]}, Frame -> {All, False}], \"Data\"]", Raises(NotImplementedError)),
        ("Rasterize[Graphics[{Disk[]}, Frame -> True, LabelStyle -> \"Text\"], \"Data\"]",
         Raises(NotImplementedError)),
        ("Rasterize[Graphics[{Disk[]}, Frame -> True, LabelStyle -> {FontFamily -> \"Times\"}], \"Data\"]",
         Raises(NotImplementedError)),
        ("Rasterize[Graphics[{Disk[]}, GridLines -> {Range, None}], \"Data\"]", Raises(NotImplementedError)),
        ("Rasterize[Graphics[{Disk[]}, Frame -> True, FrameTicks -> {{{0, 1}, {0.5}}, {None, None}}, "
         "Frame -> True], \"Data\"]", Raises(NotImplementedError)),
        ("Rasterize[Graphics[{Disk[]}, AxesLabel -> Automatic, Axes -> True], \"Data\"]", Raises(NotImplementedError)),
    ]


CASES = OPTION_CASES + SYSTEM + LEGENDS + AXISOBJ + FINDDIV + RASTER


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w15(case):
    check(case)


# ───────────── validators (every documented setting accepted, invalid ones raise ValueError) ─────────────
_GOOD = [(n, g) for n, (gs, _b) in OPTS.items() for g in gs]
_BAD = [(n, b) for n, (_g, bs) in OPTS.items() for b in bs]


@pytest.mark.parametrize("name,good", _GOOD, ids=[f"{n}->{g}" for n, g in _GOOD])
def test_w15_validator_accepts(name, good):
    from mikoshilang.library.wl_graphics_core import OPTION_VALIDATORS
    OPTION_VALIDATORS[name](ev(good), "Test")


@pytest.mark.parametrize("name,bad", _BAD, ids=[f"{n}->{b}" for n, b in _BAD])
def test_w15_validator_rejects(name, bad):
    from mikoshilang.library.wl_graphics_core import OPTION_VALIDATORS
    with pytest.raises(ValueError, match=f"^Test: {name}"):
        OPTION_VALIDATORS[name](ev(bad), "Test")


_SYMBOLIC = ["PlotRange", "ImageSize", "AspectRatio", "Axes", "Mesh", "PlotPoints", "BarOrigin", "Joined",
             "VectorRange", "ScaleDivisions"]


@pytest.mark.parametrize("name", _SYMBOLIC)
def test_w15_validator_symbolic(name):
    """A user symbol where a concrete setting is needed: Unevaluated (control flow), not an error."""
    from mikoshilang.library.core import Unevaluated
    from mikoshilang.library.wl_graphics_core import OPTION_VALIDATORS
    with pytest.raises(Unevaluated):
        OPTION_VALIDATORS[name](ev("w15sym"), "Test")


def test_w15_every_batch_symbol_registered():
    """Every W15 symbol except the documented skip has an audit status."""
    from mikoshilang.library import STATUS
    names = list(OPTS) + ["Axis", "BarLegend", "LineLegend", "PointLegend", "SwatchLegend", "AxisObject",
                          "FindDivisions", "$DisplayFunction", "$ImageResolution", "$PerformanceGoal", "$PlotTheme"]
    assert len(set(names)) == 163
    missing = [n for n in names if n not in STATUS]
    assert not missing, missing
    # HighlightRegion (skipped in W15: needs the region batches) is implemented in batch X06


def test_w15_python_api_resolve_options():
    from mikoshilang.library.wl_plot_options import resolve_options
    from mikoshilang.expr import Symbol
    opts = [ev("PlotStyle -> Red"), ev("PlotStyle -> Blue"), ev("Mesh -> 5")]
    out = resolve_options("Plot", opts, {"PlotStyle": Symbol("Automatic"), "Mesh": Symbol("None"),
                                         "Axes": True})
    assert out["Axes"] is True and out["Mesh"] == 5                       # default kept, given value used
    assert out["PlotStyle"] == ev("Red")                                  # first setting wins (OptionValue)
    with pytest.raises(ValueError, match="^Plot: Frame is not a known option"):
        resolve_options("Plot", [ev("Frame -> True")], {"Axes": True})
    with pytest.raises(ValueError, match="^Plot: Mesh"):
        resolve_options("Plot", [ev("Mesh -> -1")], {"Mesh": Symbol("None")})


@pytest.mark.skipif(not HAVE_MPL, reason="numpy is optional")
def test_w15_python_api_scaling_functions():
    """Check: each named scale composed with its inverse is the identity."""
    from mikoshilang.library.wl_plot_options import scaling_function
    for name in ("Log", "Log10", "Log2", "Reverse", "Reciprocal"):
        f, g = scaling_function(name)
        for x in (0.5, 2.0, 10.0):
            assert abs(float(g(f(x))) - x) < 1e-12 * x
    f, _ = scaling_function("Log10")
    assert abs(float(f(1000.0)) - 3.0) < 1e-12
    f, _ = scaling_function("Reverse")
    assert float(f(2.0)) == -2.0
    with pytest.raises(NotImplementedError):
        scaling_function("SignedLog")
    with pytest.raises(ValueError):
        scaling_function("Bogus")


def test_w15_python_api_tick_positions():
    from mikoshilang.library.wl_plot_options import tick_positions, nice_step
    import sympy as sp
    assert tick_positions(0.0, 1.0, 5) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert tick_positions(-1.04, 1.04, 6) == [-1.0, -0.5, 0.0, 0.5, 1.0]
    assert tick_positions(1.0, 1.0) == []
    s, a, b = nice_step(sp.Integer(0), sp.Integer(1), 5)
    assert (s, a, b) == (sp.Rational(1, 5), 0, 5)


@pytest.mark.skipif(not HAVE_MPL, reason="matplotlib is optional")
def test_w15_render_hooks_registered():
    from mikoshilang.library.wl_graphics_render import RENDER_OPTIONS
    from mikoshilang.library.wl_plot_options import RENDERED_OPTIONS
    assert all(callable(RENDER_OPTIONS[n]) for n in RENDERED_OPTIONS)
