"""Differential tests for Wolfram batch W14 (graphics core: colours, directives, Graphics/Graphics3D containers,
transforms, Show, Rasterize; modules wl_graphics_color, wl_graphics_core, wl_graphics_render).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): the documented values of named
  colours (e.g. ``Red`` is ``RGBColor[1, 0, 0]``, ``LightRed`` is ``RGBColor[1, 0.85, 0.85]``, ``StandardRed`` is
  ``RGBColor[0.93, 0.27, 0.27]``), ``Thick == Thickness[Large]``, ``Dashed == Dashing[{Small, Small}]``,
  ``Blend[{Red, Blue}] == RGBColor[0.5, 0, 0.5]`` and the inert behaviour of directives and options.
* ColorDistance reference values from the ColorDistance page and the Mathics3 documentation (which reproduces
  WL): ``ColorDistance[Magenta, Green] = 2.2507`` and the CMC values ``{1.0495, 1.27455}``.
* CIEDE2000 test data of G. Sharma, W. Wu, E. N. Dalal, "The CIEDE2000 color-difference formula", Color Res. Appl.
  30 (2005), Table 1 (pairs 1, 7 and 17), scaled by 1/100 as WL LABColor channels are.
* Bruce Lindbloom's tables (brucelindbloom.com): the D50 reference white (0.96422, 1, 0.82521) and sRGB red in
  D50 L*a*b* (54.29, 80.81, 69.89).
* Independent re-implementations written here (CIE94, HSB/CMYK formulas) and exact geometric identities
  (rotation/translation/scaling of coordinates, round trips between colour spaces).

Symbol names used as symbolic placeholders are prefixed ``w14`` so they cannot collide with other test modules.
"""
import math

import pytest

from _harness import Pred, Raises, check, ids, ev
from mikoshilang.expr import Expr

try:
    import matplotlib  # noqa: F401
    HAVE_MPL = True
except ImportError:  # minimal install
    HAVE_MPL = False


def _floats(e):
    from mikoshilang.library.wl_graphics_color import real_value
    return [float(real_value(a, "test", "x")) for a in e.args]


def Col(head, *vals, tol=1e-6):
    """Pred: a colour directive ``head[vals...]`` with machine-close channels."""
    def ok(got):
        if not (isinstance(got, Expr) and got.head == head and len(got.args) == len(vals)):
            return False
        return all(abs(g - v) <= tol * max(1.0, abs(v)) for g, v in zip(_floats(got), vals))
    return Pred(ok, f"~{head}{list(vals)}")


def Num(v, tol=1e-6):
    def ok(got):
        try:
            return abs(float(got) - v) <= tol * max(1.0, abs(v))
        except Exception:
            return False
    return Pred(ok, f"~{v}")


def NumList(vs, tol=1e-6):
    def ok(got):
        return isinstance(got, Expr) and got.head == "List" and len(got.args) == len(vs) and \
            all(abs(float(g) - v) <= tol * max(1.0, abs(v)) for g, v in zip(got.args, vs))
    return Pred(ok, f"~{vs}")


def is_head(head, n=None):
    return Pred(lambda g: isinstance(g, Expr) and g.head == head and (n is None or len(g.args) == n), f"{head}[...]")


# --- named colours: documented values (ref/<Name>) ---------------------------------------------------------
NAMED = {
    "Red": "RGBColor[1, 0, 0]", "Green": "RGBColor[0, 1, 0]", "Blue": "RGBColor[0, 0, 1]",
    "Black": "GrayLevel[0]", "White": "GrayLevel[1]", "Gray": "GrayLevel[0.5]",
    "Cyan": "RGBColor[0, 1, 1]", "Magenta": "RGBColor[1, 0, 1]", "Yellow": "RGBColor[1, 1, 0]",
    "Brown": "RGBColor[0.6, 0.4, 0.2]", "Orange": "RGBColor[1, 0.5, 0]", "Pink": "RGBColor[1, 0.5, 0.5]",
    "Purple": "RGBColor[0.5, 0, 0.5]", "Transparent": "GrayLevel[0, 0]",
    "LightRed": "RGBColor[1, 0.85, 0.85]", "LightBlue": "RGBColor[0.87, 0.94, 1]",
    "LightGreen": "RGBColor[0.88, 1, 0.88]", "LightGray": "GrayLevel[0.85]",
    "LightCyan": "RGBColor[0.9, 1, 1]", "LightMagenta": "RGBColor[1, 0.9, 1]",
    "LightYellow": "RGBColor[1, 1, 0.85]", "LightBrown": "RGBColor[0.94, 0.91, 0.88]",
    "LightOrange": "RGBColor[1, 0.9, 0.8]", "LightPink": "RGBColor[1, 0.925, 0.925]",
    "LightPurple": "RGBColor[0.94, 0.88, 0.94]",
    "DarkRed": "RGBColor[0.48, 0.06, 0.06]", "DarkBlue": "RGBColor[0.07, 0.25, 0.6]", "DarkGreen": "RGBColor[0, 0.35, 0]",
    "DarkGray": "GrayLevel[0.28]", "DarkCyan": "RGBColor[0, 0.44, 0.44]", "DarkMagenta": "RGBColor[0.49, 0.11, 0.49]",
    "DarkYellow": "RGBColor[0.6, 0.45, 0]", "DarkBrown": "RGBColor[0.32, 0.23, 0.15]",
    "DarkOrange": "RGBColor[0.5, 0.27, 0.05]", "DarkPink": "RGBColor[0.5, 0.25, 0.25]",
    "DarkPurple": "RGBColor[0.42, 0.03, 0.42]",
    "StandardRed": "RGBColor[0.93, 0.27, 0.27]", "StandardBlue": "RGBColor[0.4, 0.6, 1]",
    "StandardGreen": "RGBColor[0.14, 0.8, 0.14]", "StandardGray": "GrayLevel[0.62]",
    "StandardCyan": "RGBColor[0, 0.74, 0.74]", "StandardMagenta": "RGBColor[0.95, 0.43, 0.96]",
    "StandardYellow": "RGBColor[1, 0.75, 0]", "StandardBrown": "RGBColor[0.67, 0.54, 0.42]",
    "StandardOrange": "RGBColor[0.98, 0.56, 0.17]", "StandardPink": "RGBColor[1, 0.51, 0.51]",
    "StandardPurple": "RGBColor[0.8, 0.3, 0.8]",
}

# documented "LightX is equivalent to Lighter[X, f]" (ref/LightRed etc.); values are rounded to 2-3 digits there
LIGHT_OF = {"LightRed": ("Red", 0.85), "LightGreen": ("Green", 0.88), "LightGray": ("Gray", 0.7),
            "LightCyan": ("Cyan", 0.9), "LightMagenta": ("Magenta", 0.9), "LightYellow": ("Yellow", 0.85),
            "LightBrown": ("Brown", 0.85), "LightOrange": ("Orange", 0.8), "LightPink": ("Pink", 0.85),
            "LightPurple": ("Purple", 0.88)}

NAMED_CASES = []
for _n, _v in NAMED.items():
    NAMED_CASES += [
        (_n, _v),                                                    # reference: documented value
        (f"ColorQ[{_n}]", True),
        (f"Head[{_n}]", _v.split("[")[0]),
        (f"Graphics[{{{_n}, Disk[]}}]", f"Graphics[{{{_v}, Disk[]}}]"),
    ]
for _n, (_b, _f) in LIGHT_OF.items():                                 # Check: LightX ~ Lighter[X, f]
    NAMED_CASES.append((f"ColorDistance[{_n}, Lighter[{_b}, {_f}], DistanceFunction -> \"DeltaL\"] < 0.01", True))


def cie94(l1, l2):
    """Independent CIE94 (graphic arts, kL = kC = kH = 1, K1 = 0.045, K2 = 0.015) on WL-scaled Lab."""
    L1, a1, b1 = (100 * v for v in l1)
    L2, a2, b2 = (100 * v for v in l2)
    C1, C2 = math.hypot(a1, b1), math.hypot(a2, b2)
    dL, dC = L1 - L2, C1 - C2
    dH2 = max((a1 - a2) ** 2 + (b1 - b2) ** 2 - dC ** 2, 0.0)
    return math.sqrt(dL ** 2 + (dC / (1 + 0.045 * C1)) ** 2 + dH2 / (1 + 0.015 * C1) ** 2) / 100


COLOR_HEADS = [
    # RGBColor
    ("RGBColor[1, 0, 0]", "RGBColor[1, 0, 0]"),
    ("RGBColor[{1, 0.5, 0}]", "RGBColor[1, 0.5, 0]"),
    ("RGBColor[1, 0, 0, 0.5]", "RGBColor[1, 0, 0, 0.5]"),
    ("RGBColor[\"#FF8000\"]", Col("RGBColor", 1, 128 / 255, 0)),
    ("RGBColor[\"#F00\"]", Col("RGBColor", 1, 0, 0)),
    ("RGBColor[\"red\"]", Col("RGBColor", 1, 0, 0)),
    ("RGBColor[Hue[1/3]]", Col("RGBColor", 0, 1, 0)),
    ("RGBColor[w14x, 0, 0]", "RGBColor[w14x, 0, 0]"),                    # symbolic
    ("RGBColor[\"notacolor\"]", Raises(ValueError)),
    ("RGBColor[1, 2]", Raises(ValueError)),
    ("RGBColor[\"a\", 0, 0]", Raises(TypeError)),
    # GrayLevel
    ("GrayLevel[0.5]", "GrayLevel[0.5]"),
    ("GrayLevel[{0.3, 0.5}]", "GrayLevel[0.3, 0.5]"),
    ("GrayLevel[Red]", Col("GrayLevel", 0.299)),                       # Rec. 601 luma
    ("GrayLevel[RGBColor[0, 1, 0]]", Col("GrayLevel", 0.587)),
    ("GrayLevel[w14g]", "GrayLevel[w14g]"),
    ("GrayLevel[1, 2, 3]", Raises(ValueError)),
    ("GrayLevel[{\"a\"}]", Raises(TypeError)),
    # Hue
    ("Hue[0.5]", "Hue[0.5]"),
    ("Hue[RGBColor[0, 0, 1]]", Col("Hue", 2 / 3, 1, 1)),
    ("RGBColor[Hue[1/6, 1, 1]]", Col("RGBColor", 1, 1, 0)),
    ("RGBColor[Hue[0, 0.5, 0.8]]", Col("RGBColor", 0.8, 0.4, 0.4)),     # HSB: v (1 - s) = 0.4
    ("Hue[w14h, 1, 1]", "Hue[w14h, 1, 1]"),
    ("Hue[0.1, 0.2]", Raises(ValueError)),
    ("Hue[\"x\", 1, 1]", Raises(TypeError)),
    # CMYKColor
    ("CMYKColor[0, 1, 1, 0]", "CMYKColor[0, 1, 1, 0]"),
    ("RGBColor[CMYKColor[0, 1, 1, 0]]", Col("RGBColor", 1, 0, 0)),
    ("CMYKColor[RGBColor[0.5, 0.25, 0]]", Col("CMYKColor", 0, 0.5, 1, 0.5)),
    ("RGBColor[CMYKColor[0.5, 0, 0, 0.5]]", Col("RGBColor", 0.25, 0.5, 0.5)),   # r = (1 - c)(1 - k)
    ("CMYKColor[w14c, 0, 0, 0]", "CMYKColor[w14c, 0, 0, 0]"),
    ("CMYKColor[1, 2]", Raises(ValueError)),
    ("CMYKColor[\"a\", 0, 0, 0]", Raises(TypeError)),
    # XYZColor (D50 white, Lindbloom)
    ("XYZColor[RGBColor[1, 1, 1]]", Col("XYZColor", 0.96422, 1, 0.82521, tol=1e-4)),
    ("XYZColor[0.5, 0.5, 0.5]", "XYZColor[0.5, 0.5, 0.5]"),
    ("XYZColor[Black]", Col("XYZColor", 0, 0, 0)),
    ("XYZColor[w14z, 1, 1]", "XYZColor[w14z, 1, 1]"),
    ("XYZColor[\"a\", 1, 1]", Raises(TypeError)),
    ("XYZColor[1, 1]", Raises(ValueError)),
    # LABColor (Lindbloom: sRGB red -> L*a*b*(D50) = 54.29, 80.81, 69.89)
    ("LABColor[Red]", Col("LABColor", 0.5429, 0.8081, 0.6989, tol=2e-4)),
    ("LABColor[White]", Col("LABColor", 1, 0, 0, tol=1e-6)),
    ("LABColor[0.5, 0.1, -0.1]", "LABColor[0.5, 0.1, -0.1]"),
    ("LABColor[w14l, 0, 0]", "LABColor[w14l, 0, 0]"),
    ("LABColor[1, 2]", Raises(ValueError)),
    ("LABColor[{\"a\", 0, 0}]", Raises(TypeError)),
    # LCHColor (geometric: C = hypot(a, b), h = atan2(b, a) / 2 Pi)
    ("LCHColor[LABColor[0.5, 0, 0.3]]", Col("LCHColor", 0.5, 0.3, 0.25)),
    ("LCHColor[LABColor[0.5, 0.3, 0.4]]", Col("LCHColor", 0.5, 0.5, math.atan2(0.4, 0.3) / (2 * math.pi))),
    ("LABColor[LCHColor[0.6, 0.2, 0.5]]", Col("LABColor", 0.6, -0.2, 0, tol=1e-9)),
    ("LCHColor[w14lc, 0, 0]", "LCHColor[w14lc, 0, 0]"),
    ("LCHColor[1, 2]", Raises(ValueError)),
    ("LCHColor[\"a\", 0, 0]", Raises(TypeError)),
    # LUVColor
    ("LUVColor[White]", Col("LUVColor", 1, 0, 0)),
    ("LUVColor[Black]", Col("LUVColor", 0, 0, 0)),
    ("LUVColor[Red][[1]] == LABColor[Red][[1]]", True),                # same L* in both spaces
    ("LUVColor[w14lu, 0, 0]", "LUVColor[w14lu, 0, 0]"),
    ("LUVColor[1, 2]", Raises(ValueError)),
    ("LUVColor[\"a\", 0, 0]", Raises(TypeError)),
    # Check: round trips through every space
    ("ColorConvert[ColorConvert[RGBColor[0.2, 0.4, 0.6], \"LAB\"], \"RGB\"]", Col("RGBColor", 0.2, 0.4, 0.6)),
    ("ColorConvert[ColorConvert[RGBColor[0.2, 0.4, 0.6], \"LUV\"], \"RGB\"]", Col("RGBColor", 0.2, 0.4, 0.6)),
    ("ColorConvert[ColorConvert[RGBColor[0.2, 0.4, 0.6], \"LCH\"], \"RGB\"]", Col("RGBColor", 0.2, 0.4, 0.6)),
    ("ColorConvert[ColorConvert[RGBColor[0.2, 0.4, 0.6], \"XYZ\"], \"RGB\"]", Col("RGBColor", 0.2, 0.4, 0.6)),
    ("ColorConvert[ColorConvert[RGBColor[0.2, 0.4, 0.6], \"HSB\"], \"RGB\"]", Col("RGBColor", 0.2, 0.4, 0.6)),
    ("ColorConvert[ColorConvert[RGBColor[0.2, 0.4, 0.6], \"CMYK\"], \"RGB\"]", Col("RGBColor", 0.2, 0.4, 0.6)),
]

COLOR_FUNCS = [
    # Opacity
    ("Opacity[0.5]", "Opacity[0.5]"),
    ("Opacity[0.5, Red]", "Opacity[0.5, RGBColor[1, 0, 0]]"),
    ("Graphics[{Opacity[0.3], Disk[]}]", "Graphics[{Opacity[0.3], Disk[]}]"),
    ("Opacity[w14a]", "Opacity[w14a]"),
    ("Opacity[\"a\"]", Raises(TypeError)),
    ("Opacity[0.5, 3]", Raises(TypeError)),
    ("Opacity[1, 2, 3]", Raises(ValueError)),
    # ColorQ
    ("ColorQ[Red]", True), ("ColorQ[Hue[0.3]]", True), ("ColorQ[LABColor[0.5, 0, 0]]", True),
    ("ColorQ[3]", False), ("ColorQ[RGBColor[w14q, 0, 0]]", False), ("ColorQ[{Red}]", False),
    # ColorConvert
    ("ColorConvert[Red, \"CMYK\"]", Col("CMYKColor", 0, 1, 1, 0)),
    ("ColorConvert[Red, \"Grayscale\"]", Col("GrayLevel", 0.299)),
    ("ColorConvert[Blue, \"HSB\"]", Col("Hue", 2 / 3, 1, 1)),
    ("ColorConvert[RGBColor[0, 1, 1], \"HSB\" -> \"RGB\"]", Col("RGBColor", 1, 0, 0)),   # read as Hue[0, 1, 1]
    ("ColorConvert[White, \"XYZ\"]", Col("XYZColor", 0.96422, 1, 0.82521, tol=1e-4)),
    ("ColorConvert[{Red, Blue}, \"Grayscale\"]", Pred(lambda g: g.head == "List" and len(g.args) == 2, "list of 2")),
    ("ColorConvert[Red, \"LAB\", WhitePoint -> \"D50\"]", Col("LABColor", 0.5429, 0.8081, 0.6989, tol=2e-4)),
    ("ColorConvert[w14cc, \"LAB\"]", "ColorConvert[w14cc, \"LAB\"]"),
    ("ColorConvert[Red, \"Bogus\"]", Raises(ValueError)),
    ("ColorConvert[3, \"RGB\"]", Raises(TypeError)),
    ("ColorConvert[Red, \"LAB\", WhitePoint -> \"D65\"]", Raises(NotImplementedError)),
    ("ColorConvert[{1, 0, 0}, \"LAB\"]", Raises(NotImplementedError)),
    # Blend (ref/Blend; Mathics doc: Blend[{Red, Blue}] = RGBColor[0.5, 0., 0.5])
    ("Blend[{Red, Blue}]", Col("RGBColor", 0.5, 0, 0.5)),
    ("Blend[{Red, Blue}, 0.3]", Col("RGBColor", 0.7, 0, 0.3)),
    ("Blend[{Red, Blue, Green}, 0.75]", Col("RGBColor", 0, 0.5, 0.5)),
    ("Blend[{{0, Red}, {10, Blue}}, 5]", Col("RGBColor", 0.5, 0, 0.5)),
    ("Blend[{Red, Blue}, {1, 3}]", Col("RGBColor", 0.25, 0, 0.75)),
    ("Blend[{Black, White}, 0.25]", Col("GrayLevel", 0.25)),
    ("Blend[{Red, Blue}, 0] == RGBColor[1., 0., 0.]", True),
    ("Blend[{Red, Blue}, w14t]", "Blend[{RGBColor[1, 0, 0], RGBColor[0, 0, 1]}, w14t]"),
    ("Blend[{Red, 3}, 0.5]", Raises(TypeError)),
    ("Blend[{Red, Blue}, {-1, 1}]", Raises(ValueError)),
    ("Blend[{}]", Raises(ValueError)),
    # Lighter / Darker (ref: Lighter[c, f] = Blend[{c, White}, f], Darker[c, f] = Blend[{c, Black}, f])
    ("Lighter[Red]", Col("RGBColor", 1, 1 / 3, 1 / 3)),
    ("Lighter[Red, 1]", Col("RGBColor", 1, 1, 1)),
    ("Lighter[{Red, Blue}, 0.5]", Pred(lambda g: g.head == "List" and len(g.args) == 2, "list of 2")),
    ("Lighter[RGBColor[1, 0, 0, 0.5], 0.5]", Col("RGBColor", 1, 0.5, 0.5, 0.5)),
    ("ColorDistance[Lighter[Hue[0.3], 0.4], Blend[{Hue[0.3], White}, 0.4]] < 10^-12", True),   # Check: identity
    ("Lighter[w14li]", "Lighter[w14li]"),
    ("Lighter[3]", Raises(TypeError)),
    ("Darker[Red]", Col("RGBColor", 2 / 3, 0, 0)),
    ("Darker[GrayLevel[0.5], 0.5]", Col("RGBColor", 0.25, 0.25, 0.25)),
    ("Darker[Hue[0.5]]", Col("RGBColor", 0, 2 / 3, 2 / 3)),
    ("Darker[Red, 1]", Col("RGBColor", 0, 0, 0)),
    ("ColorDistance[Darker[Orange, 0.3], Blend[{Orange, Black}, 0.3]] < 10^-12", True),      # Check: identity
    ("Darker[w14dk]", "Darker[w14dk]"),
    ("Darker[\"x\"]", Raises(TypeError)),
    # ColorDistance
    ("ColorDistance[Magenta, Green]", Num(2.2507, tol=2e-5)),          # ref/ColorDistance, Mathics doc
    ("ColorDistance[{Magenta, Red}, Green, DistanceFunction -> \"CMC\"][[2]]", Num(1.0495, tol=5e-5)),
    ("ColorDistance[Red, Red]", Num(0.0)),
    ("ColorDistance[LABColor[0.5, 0.026772, -0.797751], LABColor[0.5, 0, -0.827485], DistanceFunction -> \"CIE2000\"]",
     Num(0.020425, tol=1e-5)),                                          # Sharma pair 1
    ("ColorDistance[LABColor[0.5, 0, 0], LABColor[0.5, -0.01, 0.02], DistanceFunction -> \"CIE2000\"]",
     Num(0.023669, tol=1e-5)),                                          # Sharma pair 7
    ("ColorDistance[LABColor[0.5, 0.025, 0], LABColor[0.73, 0.25, -0.18], DistanceFunction -> \"CIE2000\"]",
     Num(0.271492, tol=1e-5)),                                          # Sharma pair 17
    ("ColorDistance[LABColor[0.5, 0.3, 0.4], LABColor[0.6, 0.1, 0.2], DistanceFunction -> \"CIE94\"]",
     Num(cie94((0.5, 0.3, 0.4), (0.6, 0.1, 0.2)))),
    ("ColorDistance[LABColor[0.5, 0, 0], LABColor[0.7, 0, 0], DistanceFunction -> \"DeltaL\"]", Num(0.2)),
    ("ColorDistance[LABColor[0.5, 0.3, 0.4], LABColor[0.5, 0, 0.2], DistanceFunction -> \"DeltaC\"]", Num(0.3)),
    ("ColorDistance[LABColor[0.5, 0.3, 0.4], LABColor[0.5, 0, 0], DistanceFunction -> \"CIE76\"]", Num(0.5)),
    ("ColorDistance[Red, Blue, DistanceFunction -> Function[{w14u, w14v}, 1]]", 1),
    ("ColorDistance[w14d, Red]", "ColorDistance[w14d, RGBColor[1, 0, 0]]"),
    ("ColorDistance[Red, Blue, DistanceFunction -> \"DeltaH\"]", Raises(NotImplementedError)),
    ("ColorDistance[{Red}, {Red, Blue}]", Raises(ValueError)),
    ("ColorDistance[Red, 3]", Raises(TypeError)),
    # RandomColor (Check: channel ranges / ColorQ on results)
    ("SeedRandom[14]; ColorQ[RandomColor[]]", True),
    ("Length[RandomColor[5]]", 5),
    ("Dimensions[RandomColor[Hue[_, 1, 1], {2, 3}]]", "{2, 3}"),
    ("Head[RandomColor[ColorSpace -> \"HSB\"]]", "Hue"),
    ("RandomColor[RGBColor[0.5, _, 0]][[1]]", 0.5),
    ("SeedRandom[3]; AllTrue[Table[RandomColor[GrayLevel[UniformDistribution[{0.2, 0.3}]]][[1]], {20}], "
     "0.2 <= # <= 0.3 &]", True),
    ("SeedRandom[4]; AllTrue[Flatten[List @@@ RandomColor[10]], 0 <= # <= 1 &]", True),
    ("RandomColor[w14n]", "RandomColor[w14n]"),
    ("RandomColor[LABColor[_, _, _]]", Raises(NotImplementedError)),
    ("RandomColor[-1]", Raises(ValueError)),
    ("RandomColor[Hue[_, 1]]", Raises(ValueError)),
    ("RandomColor[\"x\"]", Raises(TypeError)),
    # ColorsNear (inert region specification, validated)
    ("ColorsNear[Red]", "ColorsNear[RGBColor[1, 0, 0]]"),
    ("ColorsNear[Red, 0.1]", "ColorsNear[RGBColor[1, 0, 0], 0.1]"),
    ("ColorsNear[Red, 0.1, \"CIE2000\"]", "ColorsNear[RGBColor[1, 0, 0], 0.1, \"CIE2000\"]"),
    ("ColorsNear[w14cn]", "ColorsNear[w14cn]"),
    ("ColorsNear[Red, -1]", Raises(ValueError)),
    ("ColorsNear[3]", Raises(TypeError)),
    ("ColorsNear[Red, 0.1, \"Bogus\"]", Raises(ValueError)),
    # ColorSpace / WhitePoint options
    ("ColorSpace", "ColorSpace"),
    ("RandomColor[ColorSpace -> \"Grayscale\"]", is_head("GrayLevel", 1)),
    ("RandomColor[ColorSpace -> \"Bogus\"]", Raises(ValueError)),
    ("WhitePoint", "WhitePoint"),
    ("ColorConvert[Blue, \"XYZ\", WhitePoint -> Automatic]", is_head("XYZColor", 3)),
    ("ColorConvert[Blue, \"XYZ\", WhitePoint -> \"D65\"]", Raises(NotImplementedError)),
]

CONTAINERS = [
    # Graphics / Graphics3D (ref/Graphics: inert symbolic representation; options validated)
    ("Graphics[{Red, Disk[]}, ImageSize -> 100]", "Graphics[{RGBColor[1, 0, 0], Disk[]}, ImageSize -> 100]"),
    ("Head[Graphics[{Line[{{0, 0}, {1, 1}}]}]]", "Graphics"),
    ("Graphics[{}, PlotRange -> {{0, 1}, {0, 1}}][[2]]", "PlotRange -> {{0, 1}, {0, 1}}"),
    ("Graphics[w14g]", "Graphics[w14g]"),
    ("Graphics[{Red, Disk[]}, Bogus -> 1]", Raises(ValueError)),
    ("Graphics[3]", Raises(TypeError)),
    ("Graphics[{Raster3D[{{{0}}}]}]", Raises(ValueError)),
    ("Graphics3D[{Sphere[]}, Boxed -> False]", "Graphics3D[{Sphere[]}, Boxed -> False]"),
    ("Graphics3D[{Red, Cuboid[]}]", "Graphics3D[{RGBColor[1, 0, 0], Cuboid[]}]"),
    ("Head[Graphics3D[{}]]", "Graphics3D"),
    ("Graphics3D[w14g3]", "Graphics3D[w14g3]"),
    ("Graphics3D[{Raster[{{0, 1}}]}]", Raises(ValueError)),
    ("Graphics3D[3]", Raises(TypeError)),
    # GraphicsGroup
    ("GraphicsGroup[{Red, Disk[]}]", "GraphicsGroup[{RGBColor[1, 0, 0], Disk[]}]"),
    ("Graphics[GraphicsGroup[{Disk[]}]]", "Graphics[GraphicsGroup[{Disk[]}]]"),
    ("Normal[GraphicsGroup[{GraphicsComplex[{{0, 0}, {1, 1}}, Line[{1, 2}]]}]]",
     "GraphicsGroup[{Line[{{0, 0}, {1, 1}}]}]"),
    ("GraphicsGroup[w14gg]", "GraphicsGroup[w14gg]"),
    ("GraphicsGroup[1, 2]", Raises(ValueError)),
    ("GraphicsGroup[{3}]", Raises(TypeError)),
    # GraphicsComplex (Check: Normal substitutes the coordinates)
    ("GraphicsComplex[{{0, 0}, {1, 0}, {0, 1}}, Polygon[{1, 2, 3}]]",
     "GraphicsComplex[{{0, 0}, {1, 0}, {0, 1}}, Polygon[{1, 2, 3}]]"),
    ("Normal[GraphicsComplex[{{0, 0}, {1, 0}, {0, 1}}, Polygon[{1, 2, 3}]]]", "Polygon[{{0, 0}, {1, 0}, {0, 1}}]"),
    ("Normal[GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 2}], VertexColors -> {Red, Blue}]]",
     "Line[{{0, 0}, {1, 0}}, VertexColors -> {RGBColor[1, 0, 0], RGBColor[0, 0, 1]}]"),
    ("Normal[GraphicsComplex[{{0, 0}, {3, 4}}, {Red, Point[{1, 2}]}]]", "{RGBColor[1, 0, 0], Point[{{0, 0}, {3, 4}}]}"),
    ("GraphicsComplex[w14pts, Line[{1, 2}]]", "GraphicsComplex[w14pts, Line[{1, 2}]]"),
    ("GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 5}]]", Raises(ValueError)),
    ("GraphicsComplex[3, Line[{1}]]", Raises(TypeError)),
    # Inset / Raster / Raster3D
    ("Inset[\"abc\", {0, 0}]", "Inset[\"abc\", {0, 0}]"),
    ("Inset[Graphics[{Disk[]}], {0, 0}, Center, 1]", "Inset[Graphics[{Disk[]}], {0, 0}, Center, 1]"),
    ("Graphics[{Inset[\"t\", {1, 1}]}]", "Graphics[{Inset[\"t\", {1, 1}]}]"),
    ("Inset[\"abc\", w14ip]", "Inset[\"abc\", w14ip]"),
    ("Inset[Graphics[{Disk[]}], \"x\"]", Raises(TypeError)),
    ("Raster[{{0, 1}, {1, 0}}]", "Raster[{{0, 1}, {1, 0}}]"),
    ("Raster[{{{1, 0, 0}, {0, 1, 0}}}]", "Raster[{{{1, 0, 0}, {0, 1, 0}}}]"),
    ("Graphics[Raster[{{0, 1}}]]", "Graphics[Raster[{{0, 1}}]]"),
    ("Raster[w14ra]", "Raster[w14ra]"),
    ("Raster[3]", Raises(TypeError)),
    ("Raster[{{0, 1}, {1}}]", Raises(ValueError)),
    ("Raster3D[{{{0, 1}}}]", "Raster3D[{{{0, 1}}}]"),
    ("Graphics3D[Raster3D[{{{0, 1}}}]]", "Graphics3D[Raster3D[{{{0, 1}}}]]"),
    ("Raster3D[{{{0, 1}}}, OpacityFunction -> 0.5]", "Raster3D[{{{0, 1}}}, OpacityFunction -> 0.5]"),
    ("Raster3D[w14r3]", "Raster3D[w14r3]"),
    ("Raster3D[3]", Raises(TypeError)),
    ("Raster3D[{{{0, 1}}}, OpacityFunction -> 2]", Raises(ValueError)),
]

TRANSFORMS = [
    # Rotate (ref/Rotate: about the centre of the bounding box by default; Check: exact Normal round trips)
    ("Rotate[Line[{{0, 0}, {1, 0}}], Pi/2]", "Rotate[Line[{{0, 0}, {1, 0}}], Pi/2]"),
    ("Normal[Rotate[Line[{{0, 0}, {2, 0}}], Pi/2]]", "Line[{{1, -1}, {1, 1}}]"),
    ("Normal[Rotate[Point[{1, 0}], Pi/2, {0, 0}]]", "Point[{0, 1}]"),
    ("Normal[Rotate[Point[{1, 0, 0}], Pi/2, {0, 0, 1}]]", "Point[{0, 1, 0}]"),
    ("Normal[Rotate[Point[{1, 0}], {{1, 0}, {0, 1}}]]", "Point[{0, 1}]"),
    ("Normal[Rotate[Rotate[Line[{{0, 0}, {1, 0}}], Pi/3, {0, 0}], -Pi/3, {0, 0}]]", "Line[{{0, 0}, {1, 0}}]"),
    ("Normal[Rotate[Point[{3, 4}], 2 Pi, {0, 0}]]", "Point[{3, 4}]"),
    ("Rotate[Disk[], w14r]", "Rotate[Disk[], w14r]"),
    ("Rotate[Disk[], \"a\"]", Raises(TypeError)),
    # Translate
    ("Normal[Translate[Point[{0, 0}], {1, 2}]]", "Point[{1, 2}]"),
    ("Normal[Translate[Line[{{0, 0}, {1, 1}}], {{1, 0}, {0, 1}}]]", "{Line[{{1, 0}, {2, 1}}], Line[{{0, 1}, {1, 2}}]}"),
    ("Normal[Translate[Translate[Point[{1, 2}], {3, 4}], {-3, -4}]]", "Point[{1, 2}]"),
    ("Translate[Disk[], {1, 2}]", "Translate[Disk[], {1, 2}]"),
    ("Translate[w14tr, {1, 2}]", "Translate[w14tr, {1, 2}]"),
    ("Translate[Point[{0, 0}], 3]", Raises(TypeError)),
    # Scale
    ("Normal[Scale[Polygon[{{0, 0}, {2, 0}, {2, 2}}], 2]]", "Polygon[{{-1, -1}, {3, -1}, {3, 3}}]"),
    ("Normal[Scale[Point[{1, 1}], {2, 3}, {0, 0}]]", "Point[{2, 3}]"),
    ("Normal[Scale[Scale[Point[{1, 1}], 2, {0, 0}], 1/2, {0, 0}]]", "Point[{1, 1}]"),
    ("Scale[Disk[], -1]", "Scale[Disk[], -1]"),
    ("Scale[Disk[], w14sc]", "Scale[Disk[], w14sc]"),
    ("Scale[Point[{0, 0}], \"a\"]", Raises(TypeError)),
    # Show (ref/Show: combines graphics, options of the first win, explicit options override)
    ("Show[Graphics[{Red, Disk[]}], Graphics[{Blue, Circle[]}]]",
     "Graphics[{{RGBColor[1, 0, 0], Disk[]}, {RGBColor[0, 0, 1], Circle[]}}]"),
    ("Show[Graphics[{Disk[]}, ImageSize -> 100], ImageSize -> 200]", "Graphics[{Disk[]}, ImageSize -> 200]"),
    ("Show[Graphics[{Disk[]}, ImageSize -> 100], Graphics[{Circle[]}, ImageSize -> 300]]",
     "Graphics[{{Disk[]}, {Circle[]}}, ImageSize -> 100]"),
    ("Show[{Graphics[{Disk[]}], Graphics[{Circle[]}]}]", "Graphics[{{Disk[]}, {Circle[]}}]"),
    ("Show[Graphics3D[{Sphere[]}], Graphics3D[{Cuboid[]}]]", "Graphics3D[{{Sphere[]}, {Cuboid[]}}]"),
    ("Show[w14s]", "Show[w14s]"),
    ("Show[Graphics[{}], Graphics3D[{}]]", Raises(ValueError)),
    ("Show[3]", Raises(TypeError)),
    # GraphicsGrid / GraphicsRow / GraphicsColumn (inert layouts; deviation: not expanded into Insets)
    ("GraphicsGrid[{{Graphics[{Disk[]}]}}]", "GraphicsGrid[{{Graphics[{Disk[]}]}}]"),
    ("GraphicsGrid[{{Graphics[{Disk[]}], Graphics[{Circle[]}]}}, Frame -> All]",
     "GraphicsGrid[{{Graphics[{Disk[]}], Graphics[{Circle[]}]}}, Frame -> All]"),
    ("GraphicsGrid[w14gr]", "GraphicsGrid[w14gr]"),
    ("GraphicsGrid[3]", Raises(TypeError)),
    ("GraphicsRow[{Graphics[{Disk[]}]}]", "GraphicsRow[{Graphics[{Disk[]}]}]"),
    ("GraphicsRow[{Graphics[{Disk[]}], Graphics[{Circle[]}]}, Spacings -> 5]",
     "GraphicsRow[{Graphics[{Disk[]}], Graphics[{Circle[]}]}, Spacings -> 5]"),
    ("GraphicsRow[w14row]", "GraphicsRow[w14row]"),
    ("GraphicsRow[3]", Raises(TypeError)),
    ("GraphicsColumn[{Graphics[{Disk[]}]}]", "GraphicsColumn[{Graphics[{Disk[]}]}]"),
    ("GraphicsColumn[{Graphics[{Disk[]}]}, Spacings -> 10]", "GraphicsColumn[{Graphics[{Disk[]}]}, Spacings -> 10]"),
    ("GraphicsColumn[w14col]", "GraphicsColumn[w14col]"),
    ("GraphicsColumn[3]", Raises(TypeError)),
    # Annotation / Callout / Legended / Highlighted
    ("Annotation[Disk[], \"x\"]", "Annotation[Disk[], \"x\"]"),
    ("Graphics[{Annotation[Disk[], \"d\", \"Mouse\"]}]", "Graphics[{Annotation[Disk[], \"d\", \"Mouse\"]}]"),
    ("Annotation[w14an, \"x\"]", "Annotation[w14an, \"x\"]"),
    ("Annotation[Disk[]]", Raises(ValueError)),
    ("Callout[Point[{0, 0}], \"x\"]", "Callout[Point[{0, 0}], \"x\"]"),
    ("Callout[Point[{0, 0}], \"x\", Above]", "Callout[Point[{0, 0}], \"x\", Above]"),
    ("Callout[w14co, \"x\"]", "Callout[w14co, \"x\"]"),
    ("Callout[Point[{0, 0}], \"x\", Bogus -> 1]", Raises(ValueError)),
    ("Legended[Disk[], \"x\"]", "Legended[Disk[], \"x\"]"),
    ("Legended[Graphics[{Disk[]}], \"x\"]", "Legended[Graphics[{Disk[]}], \"x\"]"),
    ("Legended[w14lg, \"x\"]", "Legended[w14lg, \"x\"]"),
    ("Legended[Disk[]]", Raises(ValueError)),
    ("Highlighted[w14hl]", "Highlighted[w14hl]"),
    ("Highlighted[42, Background -> Yellow]", "Highlighted[42, Background -> RGBColor[1, 1, 0]]"),
    ("Head[Highlighted[\"t\"]]", "Highlighted"),
    ("Highlighted[]", Raises(ValueError)),
]

DIRECTIVES = [
    # sizes (ref/Thickness etc.; Thick = Thickness[Large], Dashed = Dashing[{Small, Small}] ...)
    ("Thickness[0.01]", "Thickness[0.01]"), ("Thickness[Large]", "Thickness[Large]"),
    ("Graphics[{Thickness[0.02], Line[{{0, 0}, {1, 1}}]}]", "Graphics[{Thickness[0.02], Line[{{0, 0}, {1, 1}}]}]"),
    ("Thickness[w14th]", "Thickness[w14th]"),
    ("Thickness[\"a\"]", Raises(TypeError)), ("Thickness[-1]", Raises(ValueError)),
    ("AbsoluteThickness[2]", "AbsoluteThickness[2]"), ("AbsoluteThickness[0]", "AbsoluteThickness[0]"),
    ("AbsoluteThickness[Small]", "AbsoluteThickness[Small]"), ("AbsoluteThickness[w14at]", "AbsoluteThickness[w14at]"),
    ("AbsoluteThickness[\"a\"]", Raises(TypeError)), ("AbsoluteThickness[-2]", Raises(ValueError)),
    ("PointSize[Medium]", "PointSize[Medium]"), ("PointSize[0.05]", "PointSize[0.05]"),
    ("Graphics[{PointSize[Large], Point[{0, 0}]}]", "Graphics[{PointSize[Large], Point[{0, 0}]}]"),
    ("PointSize[w14ps]", "PointSize[w14ps]"),
    ("PointSize[\"x\"]", Raises(TypeError)), ("PointSize[-1]", Raises(ValueError)),
    ("AbsolutePointSize[5]", "AbsolutePointSize[5]"), ("AbsolutePointSize[Large]", "AbsolutePointSize[Large]"),
    ("AbsolutePointSize[0.5]", "AbsolutePointSize[0.5]"), ("AbsolutePointSize[w14aps]", "AbsolutePointSize[w14aps]"),
    ("AbsolutePointSize[\"x\"]", Raises(TypeError)), ("AbsolutePointSize[-1]", Raises(ValueError)),
    ("Dashing[{0.1, 0.2}]", "Dashing[{0.1, 0.2}]"), ("Dashing[Small]", "Dashing[Small]"),
    ("Dashing[{}]", "Dashing[{}]"), ("Dashing[w14da]", "Dashing[w14da]"),
    ("Dashing[\"x\"]", Raises(TypeError)), ("Dashing[{-1}]", Raises(ValueError)),
    ("AbsoluteDashing[{1, 2}]", "AbsoluteDashing[{1, 2}]"), ("AbsoluteDashing[3]", "AbsoluteDashing[3]"),
    ("AbsoluteDashing[{Small, Large}]", "AbsoluteDashing[{Small, Large}]"),
    ("AbsoluteDashing[w14ad]", "AbsoluteDashing[w14ad]"),
    ("AbsoluteDashing[\"x\"]", Raises(TypeError)), ("AbsoluteDashing[{-1}]", Raises(ValueError)),
    ("Thick", "Thickness[Large]"), ("Directive[Thick, Red]", "Directive[Thickness[Large], RGBColor[1, 0, 0]]"),
    ("Graphics[{Thick, Line[{{0, 0}, {1, 1}}]}][[1, 1]]", "Thickness[Large]"),
    ("Thin", "Thickness[Tiny]"), ("Directive[Thin]", "Directive[Thickness[Tiny]]"),
    ("Graphics[{Thin, Circle[]}][[1, 1]]", "Thickness[Tiny]"),
    ("Dashed", "Dashing[{Small, Small}]"), ("Directive[Dashed]", "Directive[Dashing[{Small, Small}]]"),
    ("Graphics[{Dashed, Circle[]}][[1, 1]]", "Dashing[{Small, Small}]"),
    ("Dotted", "Dashing[{0, Small}]"), ("Directive[Dotted]", "Directive[Dashing[{0, Small}]]"),
    ("Graphics[{Dotted, Circle[]}][[1, 1]]", "Dashing[{0, Small}]"),
    ("DotDashed", "Dashing[{0, Small, Small, Small}]"),
    ("Directive[DotDashed]", "Directive[Dashing[{0, Small, Small, Small}]]"),
    ("Graphics[{DotDashed, Circle[]}][[1, 1]]", "Dashing[{0, Small, Small, Small}]"),
    ("Tiny", "Tiny"), ("Thickness[Tiny]", "Thickness[Tiny]"), ("PointSize[Tiny]", "PointSize[Tiny]"),
    ("Graphics[{}, ImageSize -> Tiny]", "Graphics[{}, ImageSize -> Tiny]"),
    ("Small", "Small"), ("PointSize[Small]", "PointSize[Small]"), ("Dashing[{Small, Tiny}]", "Dashing[{Small, Tiny}]"),
    ("Medium", "Medium"), ("Thickness[Medium]", "Thickness[Medium]"), ("PointSize[Medium] === PointSize[Medium]", True),
    ("Large", "Large"), ("Thickness[Large]", "Thickness[Large]"), ("AbsoluteDashing[Large]", "AbsoluteDashing[Large]"),
    # CapForm / JoinForm / Arrowheads
    ("CapForm[\"Round\"]", "CapForm[\"Round\"]"), ("CapForm[None]", "CapForm[None]"),
    ("CapForm[\"Square\"]", "CapForm[\"Square\"]"), ("CapForm[w14cf]", "CapForm[w14cf]"),
    ("CapForm[\"Bogus\"]", Raises(ValueError)), ("CapForm[3]", Raises(TypeError)),
    ("JoinForm[\"Bevel\"]", "JoinForm[\"Bevel\"]"), ("JoinForm[{\"Miter\", 3}]", "JoinForm[{\"Miter\", 3}]"),
    ("JoinForm[\"Round\"]", "JoinForm[\"Round\"]"), ("JoinForm[w14jf]", "JoinForm[w14jf]"),
    ("JoinForm[\"Bogus\"]", Raises(ValueError)), ("JoinForm[3]", Raises(TypeError)),
    ("Arrowheads[0.05]", "Arrowheads[0.05]"), ("Arrowheads[{-0.05, 0.05}]", "Arrowheads[{-0.05, 0.05}]"),
    ("Arrowheads[{{0.05, 1}}]", "Arrowheads[{{0.05, 1}}]"), ("Arrowheads[w14ah]", "Arrowheads[w14ah]"),
    ("Arrowheads[\"x\"]", Raises(TypeError)),
    # EdgeForm / FaceForm / Directive
    ("EdgeForm[Red]", "EdgeForm[RGBColor[1, 0, 0]]"), ("EdgeForm[{Red, Thick}]", "EdgeForm[{RGBColor[1, 0, 0], Thickness[Large]}]"),
    ("EdgeForm[]", "EdgeForm[]"), ("EdgeForm[w14ef]", "EdgeForm[w14ef]"), ("EdgeForm[3]", Raises(TypeError)),
    ("FaceForm[Red, Blue]", "FaceForm[RGBColor[1, 0, 0], RGBColor[0, 0, 1]]"), ("FaceForm[]", "FaceForm[]"),
    ("FaceForm[Opacity[0.5]]", "FaceForm[Opacity[0.5]]"), ("FaceForm[w14ff]", "FaceForm[w14ff]"),
    ("FaceForm[3]", Raises(TypeError)),
    ("Directive[Red, Thick]", "Directive[RGBColor[1, 0, 0], Thickness[Large]]"),
    ("Directive[{Red, Thick}]", "Directive[RGBColor[1, 0, 0], Thickness[Large]]"),       # list form flattened
    ("Directive[]", "Directive[]"), ("Directive[w14dir]", "Directive[w14dir]"), ("Directive[3]", Raises(TypeError)),
    # 3D surface directives
    ("Glow[Red]", "Glow[RGBColor[1, 0, 0]]"), ("Glow[]", "Glow[]"),
    ("Graphics3D[{Glow[Blue], Sphere[]}]", "Graphics3D[{Glow[RGBColor[0, 0, 1]], Sphere[]}]"),
    ("Glow[w14gl]", "Glow[w14gl]"), ("Glow[3]", Raises(TypeError)),
    ("Specularity[White, 20]", "Specularity[GrayLevel[1], 20]"), ("Specularity[0.5]", "Specularity[0.5]"),
    ("Specularity[Red]", "Specularity[RGBColor[1, 0, 0]]"), ("Specularity[w14sp]", "Specularity[w14sp]"),
    ("Specularity[Red, -1]", Raises(ValueError)), ("Specularity[\"x\"]", Raises(TypeError)),
    ("Texture[{{0, 1}, {1, 0}}]", "Texture[{{0, 1}, {1, 0}}]"),
    ("Texture[{Red, Blue}]", "Texture[{RGBColor[1, 0, 0], RGBColor[0, 0, 1]}]"),
    ("Texture[Graphics[{Disk[]}]]", "Texture[Graphics[{Disk[]}]]"),
    ("Texture[w14tx]", "Texture[w14tx]"), ("Texture[3]", Raises(TypeError)),
    # lights
    ("AmbientLight[Red]", "AmbientLight[RGBColor[1, 0, 0]]"), ("AmbientLight[Red, 3]", "AmbientLight[RGBColor[1, 0, 0], 3]"),
    ("Graphics3D[{}, Lighting -> {AmbientLight[Red]}]", "Graphics3D[{}, Lighting -> {AmbientLight[RGBColor[1, 0, 0]]}]"),
    ("AmbientLight[w14al]", "AmbientLight[w14al]"), ("AmbientLight[3]", Raises(TypeError)),
    ("DirectionalLight[Red, {0, 0, 1}]", "DirectionalLight[RGBColor[1, 0, 0], {0, 0, 1}]"),
    ("DirectionalLight[White, {{0, 0, 1}, {0, 0, 0}}]", "DirectionalLight[GrayLevel[1], {{0, 0, 1}, {0, 0, 0}}]"),
    ("Graphics3D[{}, Lighting -> {DirectionalLight[Red, {1, 0, 0}]}][[2, 2, 1, 2]]", "{1, 0, 0}"),
    ("DirectionalLight[w14dl, {0, 0, 1}]", "DirectionalLight[w14dl, {0, 0, 1}]"),
    ("DirectionalLight[Red, {0, 1}]", Raises(ValueError)), ("DirectionalLight[3, {0, 0, 1}]", Raises(TypeError)),
    ("PointLight[Red, {0, 0, 1}]", "PointLight[RGBColor[1, 0, 0], {0, 0, 1}]"),
    ("PointLight[Blue, {1, 2, 3}]", "PointLight[RGBColor[0, 0, 1], {1, 2, 3}]"),
    ("Graphics3D[{PointLight[Red, {0, 0, 2}], Sphere[]}]", "Graphics3D[{PointLight[RGBColor[1, 0, 0], {0, 0, 2}], Sphere[]}]"),
    ("PointLight[w14pl, {0, 0, 1}]", "PointLight[w14pl, {0, 0, 1}]"),
    ("PointLight[Red, {0, 1}]", Raises(ValueError)), ("PointLight[3, {0, 0, 1}]", Raises(TypeError)),
    ("SpotLight[Red, {0, 0, 1}, Pi/4]", "SpotLight[RGBColor[1, 0, 0], {0, 0, 1}, Pi/4]"),
    ("SpotLight[Red, {{0, 0, 1}, {0, 0, 0}}, Pi/6]", "SpotLight[RGBColor[1, 0, 0], {{0, 0, 1}, {0, 0, 0}}, Pi/6]"),
    ("SpotLight[White, {0, 0, 1}, 1]", "SpotLight[GrayLevel[1], {0, 0, 1}, 1]"),
    ("SpotLight[w14sl, {0, 0, 1}, 1]", "SpotLight[w14sl, {0, 0, 1}, 1]"),
    ("SpotLight[Red, {0, 0, 1}, 4]", Raises(ValueError)), ("SpotLight[3, {0, 0, 1}, 1]", Raises(TypeError)),
    # shadings (ref/FlatShading etc.: X[], X[d] attenuation, list forms)
    ("FlatShading", "FlatShading"), ("FlatShading[]", "FlatShading[]"), ("FlatShading[0.5]", "FlatShading[0.5]"),
    ("FlatShading[w14fs]", "FlatShading[w14fs]"), ("FlatShading[\"x\"]", Raises(TypeError)),
    ("GouraudShading[]", "GouraudShading[]"), ("GouraudShading[{0.5, 0.3}]", "GouraudShading[{0.5, 0.3}]"),
    ("GouraudShading[Red]", "GouraudShading[RGBColor[1, 0, 0]]"), ("GouraudShading[w14gs]", "GouraudShading[w14gs]"),
    ("GouraudShading[\"x\"]", Raises(TypeError)),
    ("PhongShading[]", "PhongShading[]"), ("PhongShading[{0.5, 0.3, 0.2}]", "PhongShading[{0.5, 0.3, 0.2}]"),
    ("PhongShading[0.7]", "PhongShading[0.7]"), ("PhongShading[w14phs]", "PhongShading[w14phs]"),
    ("PhongShading[\"x\"]", Raises(TypeError)),
    ("GoochShading[]", "GoochShading[]"), ("GoochShading[Red]", "GoochShading[RGBColor[1, 0, 0]]"),
    ("Graphics3D[{GoochShading[], Sphere[]}]", "Graphics3D[{GoochShading[], Sphere[]}]"),
    ("GoochShading[w14go]", "GoochShading[w14go]"),
    ("GoochShading[3]", Raises(TypeError)), ("GoochShading[Red, Blue]", Raises(ValueError)),
    ("ToonShading[]", "ToonShading[]"), ("ToonShading[Red]", "ToonShading[RGBColor[1, 0, 0]]"),
    ("Graphics3D[{ToonShading[], Sphere[]}]", "Graphics3D[{ToonShading[], Sphere[]}]"),
    ("ToonShading[w14to]", "ToonShading[w14to]"), ("ToonShading[3]", Raises(TypeError)),
    ("HatchShading[]", "HatchShading[]"), ("HatchShading[0.5]", "HatchShading[0.5]"),
    ("HatchShading[0.5, Red]", "HatchShading[0.5, RGBColor[1, 0, 0]]"), ("HatchShading[w14hs]", "HatchShading[w14hs]"),
    ("HatchShading[3]", Raises(ValueError)),
    ("StippleShading[]", "StippleShading[]"), ("StippleShading[0.5]", "StippleShading[0.5]"),
    ("StippleShading[0.5, Red]", "StippleShading[0.5, RGBColor[1, 0, 0]]"),
    ("StippleShading[w14st]", "StippleShading[w14st]"), ("StippleShading[3]", Raises(ValueError)),
    ("HalftoneShading[]", "HalftoneShading[]"), ("HalftoneShading[0.5]", "HalftoneShading[0.5]"),
    ("HalftoneShading[0.5, Red]", "HalftoneShading[0.5, RGBColor[1, 0, 0]]"),
    ("HalftoneShading[w14ht]", "HalftoneShading[w14ht]"), ("HalftoneShading[3]", Raises(ValueError)),
    ("MaterialShading[\"Gold\"]", "MaterialShading[\"Gold\"]"), ("MaterialShading[]", "MaterialShading[]"),
    ("MaterialShading[Red]", "MaterialShading[RGBColor[1, 0, 0]]"), ("MaterialShading[w14m]", "MaterialShading[w14m]"),
    ("MaterialShading[3]", Raises(TypeError)),
    # fillings and 2D effects
    ("LinearGradientFilling[{Red, Blue}]", "LinearGradientFilling[{RGBColor[1, 0, 0], RGBColor[0, 0, 1]}]"),
    ("LinearGradientFilling[]", "LinearGradientFilling[]"),
    ("LinearGradientFilling[{Red, Blue}, Top]", "LinearGradientFilling[{RGBColor[1, 0, 0], RGBColor[0, 0, 1]}, Top]"),
    ("LinearGradientFilling[w14lgf]", "LinearGradientFilling[w14lgf]"),
    ("LinearGradientFilling[3]", Raises(TypeError)), ("LinearGradientFilling[{Red, 3}]", Raises(TypeError)),
    ("RadialGradientFilling[{Red, Blue}]", "RadialGradientFilling[{RGBColor[1, 0, 0], RGBColor[0, 0, 1]}]"),
    ("RadialGradientFilling[]", "RadialGradientFilling[]"),
    ("Head[RadialGradientFilling[{Red, Green, Blue}]]", "RadialGradientFilling"),
    ("RadialGradientFilling[w14rgf]", "RadialGradientFilling[w14rgf]"), ("RadialGradientFilling[3]", Raises(TypeError)),
    ("ConicGradientFilling[{Red, Blue}]", "ConicGradientFilling[{RGBColor[1, 0, 0], RGBColor[0, 0, 1]}]"),
    ("ConicGradientFilling[]", "ConicGradientFilling[]"),
    ("Head[ConicGradientFilling[{Red, Green, Blue}]]", "ConicGradientFilling"),
    ("ConicGradientFilling[w14cgf]", "ConicGradientFilling[w14cgf]"), ("ConicGradientFilling[3]", Raises(TypeError)),
    ("HatchFilling[]", "HatchFilling[]"), ("HatchFilling[Pi/4, 1, 10]", "HatchFilling[Pi/4, 1, 10]"),
    ("HatchFilling[\"Diagonal\"]", "HatchFilling[\"Diagonal\"]"), ("HatchFilling[w14hf]", "HatchFilling[w14hf]"),
    ("HatchFilling[{1, 2}]", Raises((TypeError, ValueError))),
    ("PatternFilling[\"Checkerboard\"]", "PatternFilling[\"Checkerboard\"]"),
    ("PatternFilling[Graphics[{Disk[]}]]", "PatternFilling[Graphics[{Disk[]}]]"),
    ("PatternFilling[{{0, 1}, {1, 0}}]", "PatternFilling[{{0, 1}, {1, 0}}]"),
    ("PatternFilling[w14pf]", "PatternFilling[w14pf]"), ("PatternFilling[3]", Raises(TypeError)),
    ("Blurring[3]", "Blurring[3]"), ("Blurring[]", "Blurring[]"), ("Blurring[{1, 2}]", "Blurring[{1, 2}]"),
    ("Blurring[w14bl]", "Blurring[w14bl]"), ("Blurring[-1]", Raises(ValueError)), ("Blurring[\"x\"]", Raises(TypeError)),
    ("DropShadowing[]", "DropShadowing[]"), ("DropShadowing[{1, -1}]", "DropShadowing[{1, -1}]"),
    ("DropShadowing[{1, -1}, 2, Red]", "DropShadowing[{1, -1}, 2, RGBColor[1, 0, 0]]"),
    ("DropShadowing[{1, -1}, w14ds]", "DropShadowing[{1, -1}, w14ds]"),
    ("DropShadowing[{1, -1, 2}]", Raises(ValueError)), ("DropShadowing[{\"a\", 1}]", Raises(TypeError)),
    ("Haloing[Red]", "Haloing[RGBColor[1, 0, 0]]"), ("Haloing[]", "Haloing[]"),
    ("Haloing[White, 2, 1]", "Haloing[GrayLevel[1], 2, 1]"), ("Haloing[w14ha]", "Haloing[w14ha]"),
    ("Haloing[3]", Raises(TypeError)), ("Haloing[Red, 2, -1]", Raises(ValueError)),
    # coordinate forms
    ("Offset[{1, 2}]", "Offset[{1, 2}]"), ("Offset[{1, 2}, {0, 0}]", "Offset[{1, 2}, {0, 0}]"),
    ("Graphics[{Point[Offset[{1, 2}, {0, 0}]]}]", "Graphics[{Point[Offset[{1, 2}, {0, 0}]]}]"),
    ("Offset[w14of]", "Offset[w14of]"), ("Offset[\"x\"]", Raises(TypeError)),
    ("Scaled[{0.5, 0.5}]", "Scaled[{0.5, 0.5}]"), ("Scaled[{0.5, 0.5}, {1, 1}]", "Scaled[{0.5, 0.5}, {1, 1}]"),
    ("Scaled[0.1]", "Scaled[0.1]"), ("Scaled[w14scd]", "Scaled[w14scd]"), ("Scaled[\"x\"]", Raises(TypeError)),
    ("ImageScaled[{0.5, 0.5}]", "ImageScaled[{0.5, 0.5}]"), ("ImageScaled[0.2]", "ImageScaled[0.2]"),
    ("Inset[\"t\", ImageScaled[{0.5, 0.5}]]", "Inset[\"t\", ImageScaled[{0.5, 0.5}]]"),
    ("ImageScaled[w14is]", "ImageScaled[w14is]"), ("ImageScaled[\"x\"]", Raises(TypeError)),
]

# options: inert symbols, accepted by their owners, invalid settings raise (ref/<Name> "Details")
_G3 = "Graphics3D[{Sphere[]}, OPT]"
OPTIONS = []
for _name, _ok, _bad, _exc in [
    ("Boxed", "False", "3", ValueError),
    ("BoxRatios", "{1, 1, 2}", "{1, -1, 2}", ValueError),
    ("BoxStyle", "Red", "3", TypeError),
    ("ClipPlanes", "{{0, 0, 1, 0}}", "3", ValueError),
    ("ClipPlanesStyle", "Opacity[0.3]", "3", TypeError),
    ("Lighting", "\"Neutral\"", "\"Bogus\"", ValueError),
    ("ViewPoint", "{1.3, -2.4, 2}", "3", ValueError),
    ("ViewVector", "{1, 2, 3}", "3", ValueError),
    ("ViewCenter", "{0.5, 0.5, 0.5}", "3", ValueError),
    ("ViewVertical", "{0, 1, 0}", "{0, 0, 0}", ValueError),
    ("ViewAngle", "Pi/4", "4", ValueError),
    ("ViewRange", "{1, 10}", "{10, 1}", ValueError),
    ("ViewMatrix", "Automatic", "3", ValueError),
    ("ViewProjection", "\"Perspective\"", "\"Bogus\"", ValueError),
    ("RotationAction", "\"Clip\"", "\"Bogus\"", ValueError),
    ("RenderingOptions", "{\"DepthPeelingLayers\" -> 4}", "3", ValueError),
    ("Antialiasing", "False", "3", ValueError),
]:
    OPTIONS += [
        (_name, _name),                                                         # inert symbol
        (_G3.replace("OPT", f"{_name} -> {_ok}"), _G3.replace("OPT", f"{_name} -> {_ok}")),
        (_G3.replace("OPT", f"{_name} -> {_bad}"), Raises(_exc)),
        (f"{_name} -> w14opt", f"{_name} -> w14opt"),
    ]
OPTIONS += [
    ("Graphics[{}, Antialiasing -> True]", "Graphics[{}, Antialiasing -> True]"),
    ("Graphics3D[{}, ViewPoint -> Front]", "Graphics3D[{}, ViewPoint -> Front]"),
    ("Graphics3D[{}, ViewPoint -> {Left, Front}]", "Graphics3D[{}, ViewPoint -> {Left, Front}]"),
    ("Graphics3D[{}, ViewPoint -> Back]", "Graphics3D[{}, ViewPoint -> Back]"),
    ("Front", "Front"), ("Back", "Back"), ("{Front, Back}", "{Front, Back}"),
    ("Graphics3D[{}, Lighting -> {AmbientLight[Red]}]", "Graphics3D[{}, Lighting -> {AmbientLight[RGBColor[1, 0, 0]]}]"),
    ("Graphics3D[{}, Lighting -> 3]", Raises(ValueError)),
    # vertex options (validated by GraphicsComplex)
    ("VertexColors", "VertexColors"),
    ("GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 2}], VertexColors -> {Red, Blue}]",
     "GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 2}], VertexColors -> {RGBColor[1, 0, 0], RGBColor[0, 0, 1]}]"),
    ("GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 2}], VertexColors -> {Red}]", Raises(ValueError)),
    ("VertexNormals", "VertexNormals"),
    ("GraphicsComplex[{{0, 0, 0}, {1, 0, 0}}, Line[{1, 2}], VertexNormals -> {{0, 0, 1}, {0, 0, 1}}]",
     "GraphicsComplex[{{0, 0, 0}, {1, 0, 0}}, Line[{1, 2}], VertexNormals -> {{0, 0, 1}, {0, 0, 1}}]"),
    ("GraphicsComplex[{{0, 0, 0}, {1, 0, 0}}, Line[{1, 2}], VertexNormals -> 3]", Raises(ValueError)),
    ("VertexTextureCoordinates", "VertexTextureCoordinates"),
    ("GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 2}], VertexTextureCoordinates -> {{0, 0}, {1, 0}}]",
     "GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 2}], VertexTextureCoordinates -> {{0, 0}, {1, 0}}]"),
    ("GraphicsComplex[{{0, 0}, {1, 0}}, Line[{1, 2}], VertexTextureCoordinates -> 3]", Raises(ValueError)),
    # Raster3D options
    ("VertexDataCoordinates", "VertexDataCoordinates"),
    ("Raster3D[{{{0, 1}}}, VertexDataCoordinates -> {{0, 0, 0}, {2, 2, 2}}]",
     "Raster3D[{{{0, 1}}}, VertexDataCoordinates -> {{0, 0, 0}, {2, 2, 2}}]"),
    ("Raster3D[{{{0, 1}}}, VertexDataCoordinates -> 3]", Raises(ValueError)),
    ("OpacityFunction", "OpacityFunction"),
    ("Raster3D[{{{0, 1}}}, OpacityFunction -> None]", "Raster3D[{{{0, 1}}}, OpacityFunction -> None]"),
    ("Raster3D[{{{0, 1}}}, OpacityFunction -> -1]", Raises(ValueError)),
    ("OpacityFunctionScaling", "OpacityFunctionScaling"),
    ("Raster3D[{{{0, 1}}}, OpacityFunctionScaling -> False]", "Raster3D[{{{0, 1}}}, OpacityFunctionScaling -> False]"),
    ("Raster3D[{{{0, 1}}}, OpacityFunctionScaling -> 3]", Raises(ValueError)),
    # Callout options
    ("CalloutMarker", "CalloutMarker"),
    ("Callout[Point[{0, 0}], \"x\", CalloutMarker -> \"Circle\"]",
     "Callout[Point[{0, 0}], \"x\", CalloutMarker -> \"Circle\"]"),
    ("Callout[Point[{0, 0}], \"x\", CalloutMarker -> 3]", Raises(ValueError)),
    ("CalloutStyle", "CalloutStyle"),
    ("Callout[Point[{0, 0}], \"x\", CalloutStyle -> Red]", "Callout[Point[{0, 0}], \"x\", CalloutStyle -> RGBColor[1, 0, 0]]"),
    ("Callout[Point[{0, 0}], \"x\", CalloutStyle -> 3]", Raises(TypeError)),
    # options consumed by later batches (plots W20, images W21): inert here, validators tested below
    ("LightingAngle", "LightingAngle"), ("LightingAngle -> {0, Pi/4}", "LightingAngle -> {0, Pi/4}"),
    ("ContourShading", "ContourShading"), ("ContourShading -> {Red, Blue}", "ContourShading -> {RGBColor[1, 0, 0], RGBColor[0, 0, 1]}"),
    ("TextureCoordinateFunction", "TextureCoordinateFunction"),
    ("TextureCoordinateFunction -> Automatic", "TextureCoordinateFunction -> Automatic"),
    ("TextureCoordinateScaling", "TextureCoordinateScaling"),
    ("TextureCoordinateScaling -> False", "TextureCoordinateScaling -> False"),
    ("TextureMapping", "TextureMapping"), ("TextureMapping -> \"Spherical\"", "TextureMapping -> \"Spherical\""),
    ("Dithering", "Dithering"), ("Dithering -> \"FloydSteinberg\"", "Dithering -> \"FloydSteinberg\""),
    ("{LightingAngle -> None}", "{LightingAngle -> None}"), ("{ContourShading -> None}", "{ContourShading -> None}"),
    ("{TextureCoordinateFunction -> (#1 &)}", "{TextureCoordinateFunction -> (#1 &)}"),
    ("{TextureCoordinateScaling -> True}", "{TextureCoordinateScaling -> True}"),
    ("{TextureMapping -> Automatic}", "{TextureMapping -> Automatic}"), ("{Dithering -> False}", "{Dithering -> False}"),
]

# (option, valid value, invalid value) for validators whose owning heads arrive in later batches
LATER_OPTIONS = [
    ("LightingAngle", "{0, Pi/4}", "{1, 2, 3}"),
    ("ContourShading", "{Red, None}", "3"),
    ("TextureCoordinateFunction", "Automatic", "3"),
    ("TextureCoordinateScaling", "True", "3"),
    ("TextureMapping", "\"Spherical\"", "3"),
    ("Dithering", "\"FloydSteinberg\"", "3"),
    ("OpacityFunction", "0.5", "{1}"),
    ("VertexColors", "{Red, Blue}", "3"),
]

# Rasterize (ref/Rasterize: "Data" is the pixel array, "RasterSize" its {width, height}); needs matplotlib
RASTER = [
    ("Rasterize[w14rz]", "Rasterize[w14rz]"),
    ("Rasterize[w14rz, \"Data\"]", "Rasterize[w14rz, \"Data\"]"),
    ("Rasterize[Graphics[{Disk[]}], \"Bogus\"]", Raises(ValueError)),
    ("Rasterize[Graphics[{Disk[]}], 3]", Raises(TypeError)),
    ("Rasterize[Graphics[{Disk[]}], \"Image\"]", Raises(NotImplementedError)),
]
if HAVE_MPL:
    RASTER += [
        ("Rasterize[Graphics[{Red, Rectangle[]}], \"RasterSize\", RasterSize -> 50]", "{50, 50}"),
        ("Dimensions[Rasterize[Graphics[{Red, Rectangle[]}], \"Data\", RasterSize -> 20]]", "{20, 20, 3}"),
        ("Rasterize[Graphics[{Red, Rectangle[]}], \"Data\", RasterSize -> 20][[10, 10]]", "{255, 0, 0}"),
        # AspectRatio Automatic follows the data ranges (2 x 1 rectangle; rotated: 1 x 2)
        ("Rasterize[Graphics[{Blue, Rectangle[{0, 0}, {2, 1}]}], \"RasterSize\", RasterSize -> 100]", "{100, 50}"),
        ("Rasterize[Graphics[{Rotate[Rectangle[{0, 0}, {2, 1}], Pi/2]}], \"RasterSize\", ImageSize -> 100]",
         "{100, 200}"),
        ("Rasterize[Graphics[{Disk[]}], {\"RasterSize\", \"Data\"}, RasterSize -> 10][[1]]", "{10, 10}"),
        ("Rasterize[Graphics[{Disk[]}], \"Data\", RasterSize -> 10, Background -> Red][[1, 1]]", "{255, 0, 0}"),
        ("Rasterize[Graphics[{Opacity[0.5], Black, Rectangle[]}], \"Data\", RasterSize -> 10][[5, 5]]",
         Pred(lambda g: all(abs(int(v) - 127) <= 1 for v in g.args), "~{127, 127, 127}")),
        ("Rasterize[Graphics[{Disk[]}], \"Data\", RasterSize -> -1]", Raises(ValueError)),
        ("Rasterize[Graphics3D[{Sphere[]}], \"Data\"]", Raises(NotImplementedError)),
        # axes are drawn since W15; an option still without a renderer raises
        ("Rasterize[Graphics[{Disk[]}, ImagePadding -> 20], \"Data\"]", Raises(NotImplementedError)),
        ("Rasterize[Graphics[{LinearGradientFilling[{Red, Blue}], Rectangle[]}], \"Data\"]", Raises(NotImplementedError)),
    ]
else:
    RASTER += [("Rasterize[Graphics[{Red, Rectangle[]}], \"Data\"]", Raises(NotImplementedError))]


# FindMatchingColor (colour-list form; image forms belong to W21)
_FMC_DOC = ("{RGBColor[0.9596025905785178, 0.3984044792339911, 0.41392466016763657], "
            "RGBColor[0.28324476057075976, 0.26097834139620046, 0.27130742463122076], "
            "RGBColor[0.05826288145041437, 0.053591610965247304, 0.06190272775838334], "
            "RGBColor[0.4757177856709867, 0.47140005577751737, 0.4837801597898334], "
            "RGBColor[0.606719111118272, 0.20611757291882038, 0.20631494426934727], "
            "RGBColor[0.5710576308076775, 0.6028166494876642, 0.7367693265717367]}")


def _fmc_is_nearest(g):
    # Check: the returned colour minimises ColorDistance over the candidates (re-measured through ColorDistance)
    cands = ["Red", "Blue", "GrayLevel[0.5]", "RGBColor[0.2, 0.8, 0.3]", "Hue[0.8]"]
    ds = [float(ev("ColorDistance[%s, Pink]" % c)) for c in cands]
    dg = float(ev("ColorDistance[%s, Pink]" % _k_input(g)))
    return abs(dg - min(ds)) < 1e-12


def _k_input(e):
    from mikoshilang.library.wl_graphics_color import K_input
    return K_input(e)


FIND_MATCHING = [
    # Wolfram documentation (Basic Examples): the pinkish entry of the list is the match for Pink
    ("FindMatchingColor[%s, Pink, MaxColorDistance -> Infinity]" % _FMC_DOC,
     Col("RGBColor", 0.9596025905785178, 0.3984044792339911, 0.41392466016763657)),
    # exact member of the list: distance 0
    ("FindMatchingColor[{Blue, Green, RGBColor[0.3, 0.2, 0.1]}, RGBColor[0.3, 0.2, 0.1], MaxColorDistance -> 0.01]",
     Col("RGBColor", 0.3, 0.2, 0.1)),
    # obvious nearest (dark red to Red), and a list of targets
    ("FindMatchingColor[{Blue, RGBColor[0.9, 0, 0], Green}, Red, MaxColorDistance -> Infinity]", Col("RGBColor", 0.9, 0, 0)),
    ("FindMatchingColor[{Black, White}, {GrayLevel[0.1], GrayLevel[0.9]}, MaxColorDistance -> Infinity]",
     "{GrayLevel[0], GrayLevel[1]}"),
    # no candidate within d (Red/Blue are far from Green: CIE76 > 1 on the 1/100 scale)
    ("FindMatchingColor[{Red, Blue}, Green, MaxColorDistance -> 0.3]", 'Missing["NotFound"]'),
    # DistanceFunction: "DeltaL" compares lightness only (L*: Black 0, GrayLevel[0.6] ~ 0.65, White 1; target L ~ 0.5)
    ("FindMatchingColor[{Black, White, GrayLevel[0.6]}, GrayLevel[0.47], MaxColorDistance -> Infinity, "
     "DistanceFunction -> \"DeltaL\"]", "GrayLevel[0.6]"),
    # Check: the result minimises ColorDistance
    ("FindMatchingColor[{Red, Blue, GrayLevel[0.5], RGBColor[0.2, 0.8, 0.3], Hue[0.8]}, Pink, "
     "MaxColorDistance -> Infinity]", Pred(_fmc_is_nearest, "nearest by ColorDistance")),
    # symbolic colour: unevaluated
    ("FindMatchingColor[{Red, Blue}, w14fmc, MaxColorDistance -> Infinity]",
     "FindMatchingColor[{RGBColor[1, 0, 0], RGBColor[0, 0, 1]}, w14fmc, MaxColorDistance -> Infinity]"),
    # errors
    ("FindMatchingColor[{Red, Blue}, Pink]", Raises(NotImplementedError)),        # undocumented default threshold
    ("FindMatchingColor[{Red}, Pink, MaxColorDistance -> 1, ColorCoverage -> 0.2]", Raises(NotImplementedError)),
    ("FindMatchingColor[Image[{{0}}], Pink, MaxColorDistance -> 1]", Raises(NotImplementedError)),
    ("FindMatchingColor[Red, Pink, MaxColorDistance -> 1]", Raises(TypeError)),
    ("FindMatchingColor[{Red, 3}, Pink, MaxColorDistance -> 1]", Raises(TypeError)),
    ("FindMatchingColor[{}, Pink, MaxColorDistance -> 1]", Raises(ValueError)),
    ("FindMatchingColor[{Red}, Pink, MaxColorDistance -> -1]", Raises(ValueError)),
    ("FindMatchingColor[{Red}, Pink, MaxColorDistance -> \"x\"]", Raises(ValueError)),
]


CASES = (NAMED_CASES + COLOR_HEADS + COLOR_FUNCS + CONTAINERS + TRANSFORMS + DIRECTIVES + OPTIONS + RASTER
         + FIND_MATCHING)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w14(case):
    check(case)


@pytest.mark.parametrize("name,good,bad", LATER_OPTIONS, ids=[o[0] for o in LATER_OPTIONS])
def test_w14_option_validators(name, good, bad):
    """Validators of options owned by W14 but consumed by later batches (plots, images)."""
    from mikoshilang.library.wl_graphics_core import OPTION_VALIDATORS
    v = OPTION_VALIDATORS[name]
    v(ev(good), "Test")                      # accepted
    with pytest.raises(ValueError, match=f"^Test: {name}"):
        v(ev(bad), "Test")


def test_w14_registries_for_later_batches():
    """The extension points documented for W15/W16/W19/W20/W21 exist and behave as documented."""
    from mikoshilang.library import wl_graphics_core as core
    from mikoshilang.library import wl_graphics_render as render
    assert "Raster" in core.PRIMITIVES and "Raster3D" in core.PRIMITIVES
    assert "ImageSize" in core.GRAPHICS_OPTIONS[2] and "ViewPoint" in core.GRAPHICS_OPTIONS[3]
    assert "Thickness" in core.DIRECTIVE_HEADS and "EdgeForm" in core.DIRECTIVE_HEADS
    assert callable(render.RENDER_PRIMITIVES["Disk"]) and isinstance(render.RENDER_OPTIONS, dict)
    assert render.IMAGE_FACTORY is None or callable(render.IMAGE_FACTORY)


@pytest.mark.skipif(not HAVE_MPL, reason="matplotlib is optional")
def test_w14_render_api_raster_array():
    """Check: raster_array returns an (h, w, 3) uint8 array whose size matches RasterSize."""
    from mikoshilang.library.wl_graphics_render import raster_array, render_png, render_svg
    g = ev("Graphics[{Red, Disk[]}]")
    a = raster_array(g, raster_size=(40, 30))
    assert a.shape == (30, 40, 3) and a.dtype.name == "uint8"
    assert render_png(g)[:8] == b"\x89PNG\r\n\x1a\n"
    svg = render_svg(g)
    assert "<svg" in (svg.decode() if isinstance(svg, bytes) else svg)


@pytest.mark.skipif(HAVE_MPL, reason="checks the minimal install")
def test_w14_rasterize_without_matplotlib():
    with pytest.raises(NotImplementedError, match="^Rasterize: "):
        ev("Rasterize[Graphics[{Disk[]}], \"Data\"]")
