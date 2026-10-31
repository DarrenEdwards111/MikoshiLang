"""Wolfram batch W14 (part 1 of 3): colours.

Colour directives (``RGBColor``, ``GrayLevel``, ``Hue``, ``CMYKColor``, ``XYZColor``, ``LABColor``,
``LCHColor``, ``LUVColor``), ``Opacity``, the named colour symbols (``Red``, ``LightBlue``, ``DarkRed``,
``StandardRed``, ...), and the colour functions ``ColorQ``, ``ColorConvert``, ``Blend``, ``Lighter``,
``Darker``, ``ColorDistance``, ``RandomColor``, ``ColorsNear`` and ``FindMatchingColor``.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

Python API for later batches (plots W15/W16/W20, images W21/W28): ``to_rgba(expr)`` gives the
``(r, g, b, a)`` floats in [0, 1] of any colour directive (``None`` if ``expr`` is not a colour);
``is_color(expr)``; ``color_space(expr)``; ``convert(expr, space)`` builds the colour in another space;
``channels(expr)`` gives the raw channel floats; ``lab(expr)`` the CIELAB value used by ``ColorDistance``;
``color_distance(c1, c2, spec)``.

Colour science. sRGB (IEC 61966-2-1) with a D65 white point for "RGB", "HSB", "CMYK" and "Grayscale";
CIE XYZ, L*a*b*, L*u*v* and LCh with a D50 white point (as Wolfram's ColorConvert documents), using the
Bradford-adapted sRGB->XYZ(D50) matrix (Lindbloom). LABColor/LUVColor/LCHColor use Wolfram's scaling:
l = L*/100, a = a*/100, b = b*/100 (likewise u, v, chroma), and LCH hue h in [0, 1) as a fraction of a turn.
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from ..symbol_values import SYMBOL_VALUES
from .core import fn, record, Unevaluated, L

AREA = "graphics: colours (W14)"

SPACES = {"RGBColor": "RGB", "GrayLevel": "Grayscale", "Hue": "HSB", "CMYKColor": "CMYK", "XYZColor": "XYZ",
          "LABColor": "LAB", "LCHColor": "LCH", "LUVColor": "LUV"}
HEADS = {v: k for k, v in SPACES.items()}
NCHANNELS = {"RGB": 3, "Grayscale": 1, "HSB": 3, "CMYK": 4, "XYZ": 3, "LAB": 3, "LCH": 3, "LUV": 3}


# ───────────────────────────── numbers ─────────────────────────────

def real_value(x, fname, what="argument"):
    """Float of a numeric real; Unevaluated for symbolic input; TypeError for strings, lists, booleans."""
    if isinstance(x, bool) or isinstance(x, str):
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    if isinstance(x, int):
        return float(x)
    if isinstance(x, float):
        if x != x:
            raise ValueError(f"{fname}: {what} is not a number (NaN)")
        return x
    if isinstance(x, Expr) and x.head in ("List", "Rule", "RuleDelayed", "Association"):
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    try:
        v = K.to_sp(x)
    except Exception:
        raise Unevaluated()
    if not isinstance(v, sp.Basic):
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    if v.free_symbols or not v.is_number:
        raise Unevaluated()
    v = sp.N(v, 17)
    if v.is_real is False or (v.is_real is None and abs(sp.im(v)) > 1e-14):
        raise ValueError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    if v in (sp.oo, -sp.oo, sp.zoo, sp.nan):
        raise ValueError(f"{fname}: {what} must be finite")
    return float(sp.re(v))


def is_real_value(x):
    try:
        real_value(x, "")
        return True
    except (Unevaluated, TypeError, ValueError):
        return False


def K_input(x):
    try:
        from .wl_forms import input_text
        return input_text(x)
    except Exception:  # pragma: no cover - formatting only
        return repr(x)


# ───────────────────────────── conversions (floats) ─────────────────────────────

def _clip(t):
    return 0.0 if t < 0 else (1.0 if t > 1 else t)


def _rgb_to_gray(r, g, b):
    return (0.299 * r + 0.587 * g + 0.114 * b,)      # Rec. 601 luma (Wolfram's "Grayscale")


def _rgb_to_hsb(r, g, b):
    r, g, b = _clip(r), _clip(g), _clip(b)
    mx, mn = max(r, g, b), min(r, g, b)
    c = mx - mn
    if c <= 0:
        h = 0.0
    elif mx == r:
        h = ((g - b) / c) % 6.0
    elif mx == g:
        h = (b - r) / c + 2.0
    else:
        h = (r - g) / c + 4.0
    h = (h / 6.0) % 1.0
    s = 0.0 if mx <= 0 else c / mx
    return (h, s, mx)


def _hsb_to_rgb(h, s, v):
    h = h % 1.0
    s, v = _clip(s), _clip(v)
    i = math.floor(6 * h)
    f = 6 * h - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][int(i) % 6]


def _cmyk_to_rgb(c, m, y, k):
    c, m, y, k = (_clip(t) for t in (c, m, y, k))
    return ((1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k))


def _rgb_to_cmyk(r, g, b):
    r, g, b = _clip(r), _clip(g), _clip(b)
    k = 1 - max(r, g, b)
    if k >= 1:
        return (0.0, 0.0, 0.0, 1.0)
    return ((1 - r - k) / (1 - k), (1 - g - k) / (1 - k), (1 - b - k) / (1 - k), k)


# sRGB -> XYZ(D50), Bradford-adapted (http://www.brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html)
_M = ((0.4360747, 0.3850649, 0.1430804),
      (0.2225045, 0.7168786, 0.0606169),
      (0.0139322, 0.0971045, 0.7141733))


def _inv3(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return ((( e * i - f * h) / det, -(b * i - c * h) / det, (b * f - c * e) / det),
            (-(d * i - f * g) / det, (a * i - c * g) / det, -(a * f - c * d) / det),
            (( d * h - e * g) / det, -(a * h - b * g) / det, (a * e - b * d) / det))


_MI = _inv3(_M)
WHITE_D50 = tuple(sum(row) for row in _M)          # XYZ of RGB white (0.96422, 1, 0.82521)
_EPS, _KAPPA = 216 / 24389, 24389 / 27


def _lin(t):
    return t / 12.92 if t <= 0.04045 else ((t + 0.055) / 1.055) ** 2.4


def _gam(t):
    return 12.92 * t if t <= 0.0031308 else 1.055 * t ** (1 / 2.4) - 0.055


def _rgb_to_xyz(r, g, b):
    v = [_lin(_clip(t)) for t in (r, g, b)]
    return tuple(sum(_M[i][j] * v[j] for j in range(3)) for i in range(3))


def _xyz_to_rgb(x, y, z):
    v = [sum(_MI[i][j] * w for j, w in enumerate((x, y, z))) for i in range(3)]
    return tuple(_clip(_gam(_clip(t))) for t in v)


def _f_lab(t):
    return t ** (1 / 3) if t > _EPS else (_KAPPA * t + 16) / 116


def _xyz_to_lab(x, y, z, white=WHITE_D50):
    fx, fy, fz = (_f_lab(c / w) for c, w in zip((x, y, z), white))
    return ((116 * fy - 16) / 100, 500 * (fx - fy) / 100, 200 * (fy - fz) / 100)


def _lab_to_xyz(l, a, b, white=WHITE_D50):
    fy = (100 * l + 16) / 116
    fx, fz = fy + 100 * a / 500, fy - 100 * b / 200
    xr = fx ** 3 if fx ** 3 > _EPS else (116 * fx - 16) / _KAPPA
    yr = ((100 * l + 16) / 116) ** 3 if 100 * l > _KAPPA * _EPS else 100 * l / _KAPPA
    zr = fz ** 3 if fz ** 3 > _EPS else (116 * fz - 16) / _KAPPA
    return (xr * white[0], yr * white[1], zr * white[2])


def _uv(x, y, z):
    d = x + 15 * y + 3 * z
    return (0.0, 0.0) if d == 0 else (4 * x / d, 9 * y / d)


def _xyz_to_luv(x, y, z, white=WHITE_D50):
    yr = y / white[1]
    lum = 116 * yr ** (1 / 3) - 16 if yr > _EPS else _KAPPA * yr
    if lum == 0:
        return (0.0, 0.0, 0.0)
    u, v = _uv(x, y, z)
    ur, vr = _uv(*white)
    return (lum / 100, 13 * lum * (u - ur) / 100, 13 * lum * (v - vr) / 100)


def _luv_to_xyz(l, u, v, white=WHITE_D50):
    lum = 100 * l
    if lum <= 0:
        return (0.0, 0.0, 0.0)
    y = ((lum + 16) / 116) ** 3 if lum > _KAPPA * _EPS else lum / _KAPPA
    ur, vr = _uv(*white)
    up, vp = 100 * u / (13 * lum) + ur, 100 * v / (13 * lum) + vr
    if vp == 0:
        return (0.0, y, 0.0)
    x = y * 9 * up / (4 * vp)
    z = y * (12 - 3 * up - 20 * vp) / (4 * vp)
    return (x, y, z)


def _lab_to_lch(l, a, b):
    return (l, math.hypot(a, b), (math.atan2(b, a) / (2 * math.pi)) % 1.0)


def _lch_to_lab(l, c, h):
    return (l, c * math.cos(2 * math.pi * h), c * math.sin(2 * math.pi * h))


def _to_xyz(space, ch):
    if space == "XYZ":
        return tuple(ch)
    if space == "LAB":
        return _lab_to_xyz(*ch)
    if space == "LCH":
        return _lab_to_xyz(*_lch_to_lab(*ch))
    if space == "LUV":
        return _luv_to_xyz(*ch)
    return _rgb_to_xyz(*_to_rgb(space, ch))


def _to_rgb(space, ch):
    if space == "RGB":
        return tuple(ch)
    if space == "Grayscale":
        return (ch[0], ch[0], ch[0])
    if space == "HSB":
        return _hsb_to_rgb(*ch)
    if space == "CMYK":
        return _cmyk_to_rgb(*ch)
    return _xyz_to_rgb(*_to_xyz(space, ch))


def convert_channels(space, ch, target):
    """Channel floats of a colour in ``space`` converted to ``target`` (alpha not included)."""
    if space == target:
        return tuple(ch)
    if target in ("XYZ", "LAB", "LCH", "LUV"):
        xyz = _to_xyz(space, ch)
        if target == "XYZ":
            return xyz
        if target == "LAB":
            return _xyz_to_lab(*xyz)
        if target == "LCH":
            return _lab_to_lch(*(_xyz_to_lab(*xyz) if space != "LAB" else ch))
        return _xyz_to_luv(*xyz)
    rgb = _to_rgb(space, ch)
    if target == "RGB":
        return tuple(_clip(t) for t in rgb)
    if target == "Grayscale":
        return (_clip(_rgb_to_gray(*rgb)[0]),)
    if target == "HSB":
        return _rgb_to_hsb(*rgb)
    return _rgb_to_cmyk(*rgb)
HTML_COLORS = {
    "aliceblue": "f0f8ff", "antiquewhite": "faebd7", "aqua": "00ffff", "aquamarine": "7fffd4", "azure": "f0ffff",
    "beige": "f5f5dc", "bisque": "ffe4c4", "black": "000000", "blanchedalmond": "ffebcd", "blue": "0000ff",
    "blueviolet": "8a2be2", "brown": "a52a2a", "burlywood": "deb887", "cadetblue": "5f9ea0", "chartreuse": "7fff00",
    "chocolate": "d2691e", "coral": "ff7f50", "cornflowerblue": "6495ed", "cornsilk": "fff8dc", "crimson": "dc143c",
    "cyan": "00ffff", "darkblue": "00008b", "darkcyan": "008b8b", "darkgoldenrod": "b8860b", "darkgray": "a9a9a9",
    "darkgreen": "006400", "darkgrey": "a9a9a9", "darkkhaki": "bdb76b", "darkmagenta": "8b008b",
    "darkolivegreen": "556b2f", "darkorange": "ff8c00", "darkorchid": "9932cc", "darkred": "8b0000",
    "darksalmon": "e9967a", "darkseagreen": "8fbc8f", "darkslateblue": "483d8b", "darkslategray": "2f4f4f",
    "darkslategrey": "2f4f4f", "darkturquoise": "00ced1", "darkviolet": "9400d3", "deeppink": "ff1493",
    "deepskyblue": "00bfff", "dimgray": "696969", "dimgrey": "696969", "dodgerblue": "1e90ff", "firebrick": "b22222",
    "floralwhite": "fffaf0", "forestgreen": "228b22", "fuchsia": "ff00ff", "gainsboro": "dcdcdc",
    "ghostwhite": "f8f8ff", "gold": "ffd700", "goldenrod": "daa520", "gray": "808080", "green": "008000",
    "greenyellow": "adff2f", "grey": "808080", "honeydew": "f0fff0", "hotpink": "ff69b4", "indianred": "cd5c5c",
    "indigo": "4b0082", "ivory": "fffff0", "khaki": "f0e68c", "lavender": "e6e6fa", "lavenderblush": "fff0f5",
    "lawngreen": "7cfc00", "lemonchiffon": "fffacd", "lightblue": "add8e6", "lightcoral": "f08080",
    "lightcyan": "e0ffff", "lightgoldenrodyellow": "fafad2", "lightgray": "d3d3d3", "lightgreen": "90ee90",
    "lightgrey": "d3d3d3", "lightpink": "ffb6c1", "lightsalmon": "ffa07a", "lightseagreen": "20b2aa",
    "lightskyblue": "87cefa", "lightslategray": "778899", "lightslategrey": "778899", "lightsteelblue": "b0c4de",
    "lightyellow": "ffffe0", "lime": "00ff00", "limegreen": "32cd32", "linen": "faf0e6", "magenta": "ff00ff",
    "maroon": "800000", "mediumaquamarine": "66cdaa", "mediumblue": "0000cd", "mediumorchid": "ba55d3",
    "mediumpurple": "9370db", "mediumseagreen": "3cb371", "mediumslateblue": "7b68ee", "mediumspringgreen": "00fa9a",
    "mediumturquoise": "48d1cc", "mediumvioletred": "c71585", "midnightblue": "191970", "mintcream": "f5fffa",
    "mistyrose": "ffe4e1", "moccasin": "ffe4b5", "navajowhite": "ffdead", "navy": "000080", "oldlace": "fdf5e6",
    "olive": "808000", "olivedrab": "6b8e23", "orange": "ffa500", "orangered": "ff4500", "orchid": "da70d6",
    "palegoldenrod": "eee8aa", "palegreen": "98fb98", "paleturquoise": "afeeee", "palevioletred": "db7093",
    "papayawhip": "ffefd5", "peachpuff": "ffdab9", "peru": "cd853f", "pink": "ffc0cb", "plum": "dda0dd",
    "powderblue": "b0e0e6", "purple": "800080", "rebeccapurple": "663399", "red": "ff0000", "rosybrown": "bc8f8f",
    "royalblue": "4169e1", "saddlebrown": "8b4513", "salmon": "fa8072", "sandybrown": "f4a460", "seagreen": "2e8b57",
    "seashell": "fff5ee", "sienna": "a0522d", "silver": "c0c0c0", "skyblue": "87ceeb", "slateblue": "6a5acd",
    "slategray": "708090", "slategrey": "708090", "snow": "fffafa", "springgreen": "00ff7f", "steelblue": "4682b4",
    "tan": "d2b48c", "teal": "008080", "thistle": "d8bfd8", "tomato": "ff6347", "turquoise": "40e0d0",
    "violet": "ee82ee", "wheat": "f5deb3", "white": "ffffff", "whitesmoke": "f5f5f5", "yellow": "ffff00",
    "yellowgreen": "9acd32",
}


# ───────────────────────────── parsing colour expressions ─────────────────────────────

_COUNTS = {"RGBColor": (3, 4), "GrayLevel": (1, 2), "Hue": (1, 3, 4), "CMYKColor": (3, 4, 5), "XYZColor": (3, 4),
           "LABColor": (3, 4), "LCHColor": (3, 4), "LUVColor": (3, 4)}


def parse_color(x):
    """``(space, channels, alpha)`` for a colour directive with numeric channels, else ``None``.

    ``channels`` are floats in the space's own scale (Hue[h] means Hue[h, 1, 1]; CMYKColor[c, m, y] has k = 0);
    ``alpha`` is a float or ``None`` when no opacity was given. ``Opacity[a, color]`` gives color with alpha a."""
    if not isinstance(x, Expr) or not isinstance(x.head, str):
        return None
    h = x.head
    if h == "Opacity" and len(x.args) == 2:
        inner = parse_color(x.args[1])
        if inner is None or not is_real_value(x.args[0]):
            return None
        return inner[0], inner[1], real_value(x.args[0], "Opacity")
    if h not in _COUNTS or len(x.args) not in _COUNTS[h]:
        return None
    if not all(is_real_value(a) for a in x.args):
        return None
    vals = [real_value(a, h) for a in x.args]
    space = SPACES[h]
    n = NCHANNELS[space]
    if h == "Hue" and len(vals) == 1:
        return space, (vals[0], 1.0, 1.0), None
    if h == "CMYKColor" and len(vals) == 3:
        return space, tuple(vals) + (0.0,), None
    if len(vals) > n:
        return space, tuple(vals[:n]), vals[n]
    return space, tuple(vals), None


def is_color(x) -> bool:
    return parse_color(x) is not None


def color_space(x):
    p = parse_color(x)
    return None if p is None else p[0]


def channels(x):
    p = parse_color(x)
    return None if p is None else p[1]


def to_rgba(x):
    """(r, g, b, a) floats in [0, 1] for a colour directive (``None`` if x is not a colour)."""
    p = parse_color(x)
    if p is None:
        return None
    r, g, b = (_clip(t) for t in convert_channels(p[0], p[1], "RGB"))
    return (r, g, b, 1.0 if p[2] is None else _clip(p[2]))


def make_color(space, ch, alpha=None):
    vals = list(ch) + ([] if alpha is None else [alpha])
    return Expr(HEADS[space], *[float(v) for v in vals])


def convert(x, target):
    """The colour x as a directive of space ``target`` ("RGB", "HSB", ...), machine reals, alpha kept."""
    p = parse_color(x)
    if p is None:
        raise TypeError(f"ColorConvert: {K_input(x)} is not a colour")
    space, ch, alpha = p
    if space == target and not (isinstance(x, Expr) and x.head == "Opacity"):
        return x
    return make_color(target, convert_channels(space, ch, target), alpha)


def lab(x):
    p = parse_color(x)
    return convert_channels(p[0], p[1], "LAB")


def _hex_color(s, fname):
    t = s.strip()
    if t.startswith("#"):
        d = t[1:]
        if not all(c in "0123456789abcdefABCDEF" for c in d) or len(d) not in (3, 4, 6, 8):
            raise ValueError(f"{fname}: {s!r} is not a hex colour (#rgb, #rrggbb or #rrggbbaa)")
        if len(d) in (3, 4):
            d = "".join(c * 2 for c in d)
        vals = [int(d[i:i + 2], 16) / 255 for i in range(0, len(d), 2)]
        return ("RGB", tuple(vals[:3]), vals[3] if len(vals) == 4 else None)
    key = t.lower().replace(" ", "")
    if key in HTML_COLORS:
        d = HTML_COLORS[key]
        return ("RGB", tuple(int(d[i:i + 2], 16) / 255 for i in (0, 2, 4)), None)
    raise ValueError(f"{fname}: {s!r} is not a hex colour or an HTML colour name")


def _color_head(head):
    counts = _COUNTS[head]
    target = SPACES[head]

    def impl(*args):
        if len(args) == 1:
            a = args[0]
            if isinstance(a, str):
                sp_, ch, alpha = _hex_color(a, head)
                return make_color(target, convert_channels(sp_, ch, target), alpha)
            if K.is_list(a):
                items = list(a.args)
                if len(items) not in counts:
                    raise ValueError(f"{head}: {K_input(a)} must be a list of {_counts_text(counts)} numbers")
                if any(isinstance(i, (str, bool)) or K.is_list(i) for i in items):
                    raise TypeError(f"{head}: {K_input(a)} must be a list of numbers")
                for i in items:
                    real_value(i, head, "component")
                return Expr(head, *items)
            if isinstance(a, Expr) and parse_color(a) is not None:
                return convert(a, target)
            if isinstance(a, Expr) and a.head == "Image":
                raise NotImplementedError(f"{head}: images are not supported (Image objects arrive with W21)")
            if isinstance(a, (Symbol, Expr)) and not is_real_value(a):
                raise Unevaluated()                 # symbolic argument (or a colour with symbolic channels)
        if len(args) not in counts:
            raise ValueError(f"{head}: expected {_counts_text(counts)} components, got {len(args)}")
        for a in args:
            real_value(a, head, "component")       # TypeError / ValueError; symbolic -> Unevaluated
        raise Unevaluated()                         # a valid colour directive is inert
    impl.__annotations__ = {"args": "Real | ListOf[Real] | String | Expr", "return": "Expr"}
    return impl


def _counts_text(counts):
    c = [str(n) for n in counts]
    return ", ".join(c[:-1]) + " or " + c[-1] if len(c) > 1 else c[0]


_COLOR_NOTES = {
    "RGBColor": ("RGBColor[r, g, b], RGBColor[r, g, b, a] (opacity a) and RGBColor[{r, g, b(, a)}] (normalised to "
                 "RGBColor[r, g, b(, a)]) are inert colour directives (components outside [0, 1] are clipped when "
                 "drawn); RGBColor[\"#rrggbb\" | \"#rgb\" | \"#rrggbbaa\" | \"htmlname\"] gives machine-real "
                 "components (CSS colour names, case-insensitive); RGBColor[color] converts any ColorQ colour "
                 "(sRGB, D65). Symbolic components stay unevaluated; strings or lists as components raise "
                 "TypeError, complex components or a wrong count ValueError, an unknown colour name ValueError."),
    "GrayLevel": ("GrayLevel[g], GrayLevel[g, a] and GrayLevel[{g(, a)}] inert gray-level directives; "
                  "GrayLevel[\"name\"] and GrayLevel[color] convert (luma 0.299 r + 0.587 g + 0.114 b of sRGB). "
                  "Symbolic stays unevaluated; wrong kinds raise TypeError, wrong counts ValueError."),
    "Hue": ("Hue[h] (= Hue[h, 1, 1]), Hue[h, s, b], Hue[h, s, b, a] and Hue[{h, s, b(, a)}] inert HSB directives "
            "(h is cyclic, s, b, a clipped when drawn); Hue[\"name\"] and Hue[color] convert to HSB. Symbolic stays "
            "unevaluated; wrong kinds raise TypeError, wrong counts ValueError."),
    "CMYKColor": ("CMYKColor[c, m, y] (k = 0), CMYKColor[c, m, y, k], CMYKColor[c, m, y, k, a] and the list form "
                  "are inert; CMYKColor[color] converts with the device-independent formulas k = 1 - max(r, g, b), "
                  "c = (1 - r - k)/(1 - k) (and back r = (1 - c)(1 - k)); no ICC profile is applied (Wolfram may use "
                  "a profile in some versions). Symbolic stays unevaluated; wrong kinds TypeError, counts ValueError."),
    "XYZColor": ("XYZColor[x, y, z(, a)] and the list form are inert CIE 1931 XYZ directives (D50 white point, as "
                 "Wolfram's ColorConvert); XYZColor[color] converts (sRGB -> XYZ(D50) via the Bradford-adapted "
                 "matrix, not clipped). Symbolic stays unevaluated; wrong kinds TypeError, counts ValueError."),
    "LABColor": ("LABColor[l, a, b(, alpha)] and the list form are inert CIE 1976 L*a*b* directives with Wolfram's "
                 "scaling l = L*/100, a = a*/100, b = b*/100 (D50); LABColor[color] converts (not clipped). "
                 "Symbolic stays unevaluated; wrong kinds TypeError, counts ValueError."),
    "LCHColor": ("LCHColor[l, c, h(, alpha)] and the list form are inert CIE LCh(ab) directives (l = L*/100, "
                 "c = C*/100, h = hue angle / 2 Pi in [0, 1)); LCHColor[color] converts. Symbolic stays "
                 "unevaluated; wrong kinds TypeError, counts ValueError."),
    "LUVColor": ("LUVColor[l, u, v(, alpha)] and the list form are inert CIE 1976 L*u*v* directives (scaled by "
                 "1/100, D50); LUVColor[color] converts. Symbolic stays unevaluated; wrong kinds TypeError, counts "
                 "ValueError."),
}

for _h in SPACES:
    fn(_h, area=AREA, status="added",
       notes=_COLOR_NOTES[_h] + " Check: reference (Wolfram documentation values, sRGB/CIE formulas) and round "
                                "trips through ColorConvert in tests.")(_color_head(_h))


# ───────────────────────────── named colours ─────────────────────────────

def _rgb(r, g, b):
    return Expr("RGBColor", r, g, b)


NAMED_COLORS = {
    # Wolfram documentation, "X is equivalent to ..." (pure colours)
    "Red": _rgb(1, 0, 0), "Green": _rgb(0, 1, 0), "Blue": _rgb(0, 0, 1), "Black": Expr("GrayLevel", 0),
    "White": Expr("GrayLevel", 1), "Gray": Expr("GrayLevel", 0.5), "Cyan": _rgb(0, 1, 1),
    "Magenta": _rgb(1, 0, 1), "Yellow": _rgb(1, 1, 0), "Brown": _rgb(0.6, 0.4, 0.2), "Orange": _rgb(1, 0.5, 0),
    "Pink": _rgb(1, 0.5, 0.5), "Purple": _rgb(0.5, 0, 0.5),
    # Light*: Lighter[base, f] with the documented f (LightBlue is given explicitly)
    "LightRed": _rgb(1, 0.85, 0.85), "LightGreen": _rgb(0.88, 1, 0.88), "LightBlue": _rgb(0.87, 0.94, 1),
    "LightGray": Expr("GrayLevel", 0.85), "LightCyan": _rgb(0.9, 1, 1), "LightMagenta": _rgb(1, 0.9, 1),
    "LightYellow": _rgb(1, 1, 0.85), "LightBrown": _rgb(0.94, 0.91, 0.88), "LightOrange": _rgb(1, 0.9, 0.8),
    "LightPink": _rgb(1, 0.925, 0.925), "LightPurple": _rgb(0.94, 0.88, 0.94),
    # Dark* (Wolfram 14.2 documentation)
    "DarkBlue": _rgb(0.07, 0.25, 0.6), "DarkBrown": _rgb(0.32, 0.23, 0.15), "DarkCyan": _rgb(0, 0.44, 0.44),
    "DarkGray": Expr("GrayLevel", 0.28), "DarkGreen": _rgb(0, 0.35, 0), "DarkMagenta": _rgb(0.49, 0.11, 0.49),
    "DarkOrange": _rgb(0.5, 0.27, 0.05), "DarkPink": _rgb(0.5, 0.25, 0.25), "DarkPurple": _rgb(0.42, 0.03, 0.42),
    "DarkRed": _rgb(0.48, 0.06, 0.06), "DarkYellow": _rgb(0.6, 0.45, 0),
    # Standard* ("currently equivalent to", Wolfram 14.2 documentation)
    "StandardBlue": _rgb(0.4, 0.6, 1), "StandardBrown": _rgb(0.67, 0.54, 0.42), "StandardCyan": _rgb(0, 0.74, 0.74),
    "StandardGray": Expr("GrayLevel", 0.62), "StandardGreen": _rgb(0.14, 0.8, 0.14),
    "StandardMagenta": _rgb(0.95, 0.43, 0.96), "StandardOrange": _rgb(0.98, 0.56, 0.17),
    "StandardPink": _rgb(1, 0.51, 0.51), "StandardPurple": _rgb(0.8, 0.3, 0.8), "StandardRed": _rgb(0.93, 0.27, 0.27),
    "StandardYellow": _rgb(1, 0.75, 0),
    "Transparent": Expr("GrayLevel", 0, 0),
}
_LIGHT_OF = {"LightRed": ("Red", "0.85"), "LightGreen": ("Green", "0.88"), "LightGray": ("Gray", "0.7"),
             "LightCyan": ("Cyan", "0.9"), "LightMagenta": ("Magenta", "0.9"), "LightYellow": ("Yellow", "0.85"),
             "LightBrown": ("Brown", "0.85"), "LightOrange": ("Orange", "0.8"), "LightPink": ("Pink", "0.85"),
             "LightPurple": ("Purple", "0.88")}

for _n, _v in NAMED_COLORS.items():
    SYMBOL_VALUES[_n] = _v
    _extra = (f" (documented as Lighter[{_LIGHT_OF[_n][0]}, {_LIGHT_OF[_n][1]}])" if _n in _LIGHT_OF else "")
    record(_n, AREA, "added",
           f"{_n} evaluates to {K_input(_v)}{_extra}, as documented by Wolfram; an assignment by the user takes "
           f"precedence. Check: reference (Wolfram documentation value) in tests.")


# ───────────────────────────── Opacity, ColorQ ─────────────────────────────

@fn("Opacity", area=AREA, status="added",
    notes="Opacity[a] and Opacity[a, color] inert directives (a in [0, 1], clipped when drawn; Opacity[a, color] "
          "is that colour with opacity a for to_rgba / drawing). Symbolic a or color stays unevaluated; a string or "
          "list as a raises TypeError, a complex a ValueError, a non-colour second argument TypeError. "
          "Check: reference (Wolfram documentation forms) in tests.")
def _opacity(a: Real, *rest: Expr) -> Expr:
    if len(rest) > 1:
        raise ValueError(f"Opacity: expected 1 or 2 arguments, got {1 + len(rest)}")
    real_value(a, "Opacity", "opacity")
    if rest:
        c = rest[0]
        if parse_color(c) is None:
            if isinstance(c, (str, bool, int, float)) or K.is_list(c):
                raise TypeError(f"Opacity: {K_input(c)} is not a colour")
            raise Unevaluated()
    raise Unevaluated()


@fn("ColorQ", area=AREA, status="added",
    notes="ColorQ[expr] is True for a colour directive with numeric channels in any of the 8 colour spaces "
          "(RGBColor, GrayLevel, Hue, CMYKColor, XYZColor, LABColor, LCHColor, LUVColor, with or without opacity; "
          "named colours such as Red evaluate to these), False otherwise (including symbolic channels and "
          "Opacity[a, color]). Check: exact (structural test) in tests.")
def _colorq(x: Expr) -> Boolean:
    return isinstance(x, Expr) and x.head != "Opacity" and parse_color(x) is not None


# ───────────────────────────── ColorConvert ─────────────────────────────

SPACE_NAMES = ("Grayscale", "RGB", "CMYK", "HSB", "XYZ", "LAB", "LCH", "LUV")


def _space_arg(x, fname):
    if isinstance(x, str):
        if x in SPACE_NAMES:
            return x
        raise ValueError(f"{fname}: {x!r} is not a colour space (one of {', '.join(SPACE_NAMES)})")
    if isinstance(x, Expr) and x.head == "ColorProfileData":
        raise NotImplementedError(f"{fname}: ICC colour profiles (ColorProfileData) are not supported")
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: colour space must be a string such as \"RGB\", not {K_input(x)}")


def _white_point(opts, fname):
    w = opts.get("WhitePoint")
    if w is None or (isinstance(w, Symbol) and w.name == "Automatic") or w == "D50":
        return
    if isinstance(w, str) or (isinstance(w, Expr) and parse_color(w) is not None):
        raise NotImplementedError(f"{fname}: WhitePoint -> {K_input(w)} is not supported (only Automatic = \"D50\")")
    raise ValueError(f"{fname}: WhitePoint must be Automatic, a named white point or a colour")


def _opts(args, names, fname):
    from .wl_common import split_options
    return split_options(args, names, fname)


@fn("ColorConvert", area=AREA, status="added",
    notes="ColorConvert[color, space] for space \"Grayscale\", \"RGB\", \"CMYK\", \"HSB\", \"XYZ\", \"LAB\", \"LCH\", "
          "\"LUV\" (machine-real channels; opacity kept; a colour already in space is returned unchanged); "
          "ColorConvert[{c1, c2, ...}, space] maps; ColorConvert[color, src -> dst] reinterprets the channels of "
          "color in src. Conversions: sRGB with D65 for Grayscale/RGB/CMYK/HSB, CIE spaces with D50 (Bradford "
          "matrix), channels clipped to [0, 1] only for Grayscale/RGB/CMYK/HSB targets, as documented. "
          "WhitePoint -> Automatic | \"D50\" only (other white points raise NotImplementedError), images and "
          "ColorProfileData raise NotImplementedError, raw channel lists raise NotImplementedError, an unknown "
          "space ValueError, a non-colour TypeError; symbolic input stays unevaluated. "
          "Check: reference (sRGB/CIE formulas, Wolfram documentation values) and round trips in tests.")
def _colorconvert(c: Expr, space: String | Rule, *opts: Option) -> Expr:
    pos, o = _opts(list(opts), {"WhitePoint"}, "ColorConvert")
    if pos:
        raise ValueError("ColorConvert: expected ColorConvert[color, space, options]")
    _white_point(o, "ColorConvert")
    src = None
    if isinstance(space, Expr) and space.head == "Rule" and len(space.args) == 2:
        src, space = _space_arg(space.args[0], "ColorConvert"), space.args[1]
    target = _space_arg(space, "ColorConvert")

    def one(x):
        if isinstance(x, Expr) and x.head in ("Image", "Image3D"):
            raise NotImplementedError("ColorConvert: images are not supported (Image objects arrive with W21)")
        p = parse_color(x)
        if p is None:
            if isinstance(x, (str, bool, int, float)) or K.is_list(x):
                raise TypeError(f"ColorConvert: {K_input(x)} is not a colour")
            raise Unevaluated()
        space_, ch, alpha = p
        if src is not None:
            n = NCHANNELS[src]
            raw = list(ch)
            if space_ == "HSB" and len(x.args) == 1:
                raw = [real_value(x.args[0], "ColorConvert")]
            if len(raw) != n:
                raise ValueError(f"ColorConvert: {K_input(x)} has {len(raw)} channels, {src} needs {n}")
            return make_color(target, convert_channels(src, raw, target), alpha)
        if space_ == target and x.head != "Opacity":
            return x
        return make_color(target, convert_channels(space_, ch, target), alpha)

    if K.is_list(c):
        if c.args and all(is_real_value(v) for v in c.args):
            raise NotImplementedError("ColorConvert: raw channel lists are not supported; give a colour directive")
        return L(*[one(x) for x in c.args])
    return one(c)


# ───────────────────────────── Blend, Lighter, Darker ─────────────────────────────

def _color_or_raise(x, fname):
    p = parse_color(x)
    if p is not None:
        return p
    if isinstance(x, Expr) and x.head in ("Image", "Image3D"):
        raise NotImplementedError(f"{fname}: images are not supported (Image objects arrive with W21)")
    if isinstance(x, (str, bool, int, float)) or K.is_list(x):
        raise TypeError(f"{fname}: {K_input(x)} is not a colour")
    raise Unevaluated()


def mix(parsed, weights):
    """Additive mix of parsed colours with the given (normalised) weights: same-space RGB or Grayscale colours mix
    in that space, anything else in RGB."""
    spaces = {p[0] for p in parsed}
    space = spaces.pop() if len(spaces) == 1 and spaces <= {"RGB", "Grayscale"} else "RGB"
    chans = [convert_channels(p[0], p[1], space) if p[0] != space else p[1] for p in parsed]
    n = len(chans[0])
    out = [sum(w * c[i] for w, c in zip(weights, chans)) for i in range(n)]
    alpha = None
    if any(p[2] is not None for p in parsed):
        alpha = sum(w * (1.0 if p[2] is None else p[2]) for w, p in zip(weights, parsed))
    return make_color(space, out, alpha)


@fn("Blend", area=AREA, status="added",
    notes="Blend[{c1, c2, ...}] (equal parts), Blend[{c1, ..., cn}, x] (piecewise-linear from c1 at 0 to cn at 1; "
          "x clipped to [0, 1]), Blend[{{x1, c1}, {x2, c2}, ...}, x] (ci at xi, clipped to [x1, xn]) and "
          "Blend[{c1, ...}, {u1, ...}] (fractions normalised to total 1). Additive mixing: colours all in RGBColor "
          "or all in GrayLevel mix in that space, any other mixture is converted to RGBColor; opacities mix when "
          "any colour has one. Machine-real result. Symbolic x or colours stay unevaluated; non-colours raise "
          "TypeError, negative or all-zero fractions ValueError, images NotImplementedError. "
          "Check: reference (Wolfram documentation examples, e.g. Blend[{Red, Blue}] = RGBColor[0.5, 0., 0.5]) "
          "and the identity Lighter[c, f] = Blend[{c, White}, f] in tests.")
def _blend(cols: ListOf[Expr], x: Real | ListOf[Real] = None) -> Expr:
    if not K.is_list(cols):
        if isinstance(cols, (Symbol,)) or (isinstance(cols, Expr) and parse_color(cols) is None):
            raise Unevaluated()
        raise TypeError(f"Blend: the first argument must be a list of colours, not {K_input(cols)}")
    items = list(cols.args)
    if not items:
        raise ValueError("Blend: the list of colours is empty")
    pairs = all(K.is_list(i) and len(i.args) == 2 and parse_color(i.args[1]) is not None for i in items)
    if pairs:
        if x is None:
            raise ValueError("Blend: Blend[{{x1, c1}, ...}] needs a blending parameter x")
        xs = [real_value(i.args[0], "Blend", "position") for i in items]
        ps = [parse_color(i.args[1]) for i in items]
        t = real_value(x, "Blend", "blending parameter")
        order = sorted(range(len(xs)), key=lambda k: xs[k])
        xs, ps = [xs[k] for k in order], [ps[k] for k in order]
        if t <= xs[0]:
            return mix([ps[0]], [1.0])
        if t >= xs[-1]:
            return mix([ps[-1]], [1.0])
        for k in range(len(xs) - 1):
            if xs[k] <= t <= xs[k + 1]:
                span = xs[k + 1] - xs[k]
                f = 0.0 if span == 0 else (t - xs[k]) / span
                return mix([ps[k], ps[k + 1]], [1 - f, f])
    ps = [_color_or_raise(i, "Blend") for i in items]
    if x is None:
        return mix(ps, [1.0 / len(ps)] * len(ps))
    if K.is_list(x):
        us = [real_value(u, "Blend", "fraction") for u in x.args]
        if len(us) != len(ps):
            raise ValueError("Blend: the fractions must have the same length as the colours")
        if any(u < 0 for u in us) or sum(us) <= 0:
            raise ValueError("Blend: fractions must be non-negative and not all zero")
        s = sum(us)
        return mix(ps, [u / s for u in us])
    t = _clip(real_value(x, "Blend", "blending parameter"))
    if len(ps) == 1:
        return mix(ps, [1.0])
    pos = t * (len(ps) - 1)
    k = min(int(math.floor(pos)), len(ps) - 2)
    f = pos - k
    return mix([ps[k], ps[k + 1]], [1 - f, f])


def _shade(fname, toward):
    def impl(c, f=None):
        frac = 1 / 3 if f is None else _clip(real_value(f, fname, "fraction"))
        if K.is_list(c):
            return L(*[impl(x, f) for x in c.args])
        space, ch, alpha = _color_or_raise(c, fname)
        rgb = convert_channels(space, ch, "RGB")
        out = [(1 - frac) * v + frac * toward for v in rgb]
        return make_color("RGB", out, alpha)
    impl.__annotations__ = {"c": "Expr | ListOf[Expr]", "f": "Real", "return": "Expr"}
    return impl


fn("Lighter", area=AREA, status="added",
   notes="Lighter[color, f] = Blend[{color, White}, f] in RGB (always an RGBColor, opacity unchanged, f clipped to "
         "[0, 1]); Lighter[color] uses f = 1/3; Lighter[{c1, ...}, f] maps. Symbolic input stays unevaluated; "
         "non-colours raise TypeError, images NotImplementedError. Check: reference (LightRed = Lighter[Red, 0.85] "
         "etc. from the Wolfram documentation) in tests.")(_shade("Lighter", 1.0))
fn("Darker", area=AREA, status="added",
   notes="Darker[color, f] = Blend[{color, Black}, f] in RGB (always an RGBColor, opacity unchanged, f clipped to "
         "[0, 1]); Darker[color] uses f = 1/3; Darker[{c1, ...}, f] maps. Symbolic input stays unevaluated; "
         "non-colours raise TypeError, images NotImplementedError. Check: reference (Darker[c, 1] is black, "
         "Darker[c, 0] is c) and the identity with Blend in tests.")(_shade("Darker", 0.0))


# ───────────────────────────── ColorDistance ─────────────────────────────

def _de76(l1, l2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(l1, l2)))


def _de94(l1, l2, kl=1.0, k1=0.045, k2=0.015):
    L1, a1, b1 = (100 * v for v in l1)
    L2, a2, b2 = (100 * v for v in l2)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    dl, dc = L1 - L2, c1 - c2
    dh2 = max((a1 - a2) ** 2 + (b1 - b2) ** 2 - dc ** 2, 0.0)
    sc, sh = 1 + k1 * c1, 1 + k2 * c1
    return math.sqrt((dl / kl) ** 2 + (dc / sc) ** 2 + dh2 / sh ** 2) / 100


def _de2000(l1, l2):
    """CIEDE2000 (Sharma, Wu, Dalal 2005), on the 0-100 scale, returned / 100."""
    L1, a1, b1 = (100 * v for v in l1)
    L2, a2, b2 = (100 * v for v in l2)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    cb = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(cb ** 7 / (cb ** 7 + 25 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360 if (a1p or b1) else 0.0
    h2p = math.degrees(math.atan2(b2, a2p)) % 360 if (a2p or b2) else 0.0
    dLp, dCp = L2 - L1, c2p - c1p
    if c1p * c2p == 0:
        dhp = 0.0
    elif abs(h2p - h1p) <= 180:
        dhp = h2p - h1p
    elif h2p - h1p > 180:
        dhp = h2p - h1p - 360
    else:
        dhp = h2p - h1p + 360
    dHp = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        hbp = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        hbp = (h1p + h2p) / 2
    elif h1p + h2p < 360:
        hbp = (h1p + h2p + 360) / 2
    else:
        hbp = (h1p + h2p - 360) / 2
    t = (1 - 0.17 * math.cos(math.radians(hbp - 30)) + 0.24 * math.cos(math.radians(2 * hbp))
         + 0.32 * math.cos(math.radians(3 * hbp + 6)) - 0.20 * math.cos(math.radians(4 * hbp - 63)))
    dtheta = 30 * math.exp(-(((hbp - 275) / 25) ** 2))
    rc = 2 * math.sqrt(Cbp ** 7 / (Cbp ** 7 + 25 ** 7))
    sl = 1 + 0.015 * (Lbp - 50) ** 2 / math.sqrt(20 + (Lbp - 50) ** 2)
    sc, sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * t
    rt = -math.sin(math.radians(2 * dtheta)) * rc
    return math.sqrt((dLp / sl) ** 2 + (dCp / sc) ** 2 + (dHp / sh) ** 2 + rt * (dCp / sc) * (dHp / sh)) / 100


def _decmc(l1, l2, lw, cw):
    """CMC l:c (1984), on the 0-100 scale, returned / 100."""
    L1, a1, b1 = (100 * v for v in l1)
    L2, a2, b2 = (100 * v for v in l2)
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    dl, dc = L1 - L2, c1 - c2
    dh2 = max((a1 - a2) ** 2 + (b1 - b2) ** 2 - dc ** 2, 0.0)
    h1 = math.degrees(math.atan2(b1, a1)) % 360
    sl = 0.511 if L1 < 16 else 0.040975 * L1 / (1 + 0.01765 * L1)
    sc = 0.0638 * c1 / (1 + 0.0131 * c1) + 0.638
    f = math.sqrt(c1 ** 4 / (c1 ** 4 + 1900))
    t = (0.56 + abs(0.2 * math.cos(math.radians(h1 + 168))) if 164 <= h1 <= 345
         else 0.36 + abs(0.4 * math.cos(math.radians(h1 + 35))))
    sh = sc * (f * t + 1 - f)
    return math.sqrt((dl / (lw * sl)) ** 2 + (dc / (cw * sc)) ** 2 + dh2 / sh ** 2) / 100


def distance_function(spec, fname="ColorDistance"):
    """A Python function (lab1, lab2) -> float for a DistanceFunction setting."""
    if spec is None or (isinstance(spec, Symbol) and spec.name == "Automatic") or spec == "CIE76":
        return _de76
    if spec == "CIE94":
        return _de94
    if spec in ("CIE2000", "CIEDE2000"):
        return _de2000
    if spec == "DeltaL":
        return lambda u, v: abs(u[0] - v[0])
    if spec == "DeltaC":
        return lambda u, v: abs(math.hypot(u[1], u[2]) - math.hypot(v[1], v[2]))
    if spec == "DeltaH":
        raise NotImplementedError(f"{fname}: DistanceFunction -> \"DeltaH\" is not supported (its scaling is not "
                                  "documented)")
    if spec == "CMC":
        return lambda u, v: _decmc(u, v, 1, 1)
    if K.is_list(spec) and len(spec.args) == 2 and spec.args[0] == "CMC":
        w = spec.args[1]
        if w == "Perceptibility":
            return lambda u, v: _decmc(u, v, 1, 1)
        if w == "Acceptability":
            return lambda u, v: _decmc(u, v, 2, 1)
        if K.is_list(w) and len(w.args) == 2:
            lw, cw = (real_value(t, fname, "CMC weight") for t in w.args)
            if lw <= 0 or cw <= 0:
                raise ValueError(f"{fname}: CMC weights must be positive")
            return lambda u, v: _decmc(u, v, lw, cw)
        raise ValueError(f"{fname}: {K_input(spec)} is not a valid CMC specification")
    if isinstance(spec, str):
        raise ValueError(f"{fname}: {spec!r} is not a known colour distance")
    if isinstance(spec, (Symbol, Expr)) and not isinstance(spec, bool):
        from .wl_common import apply

        def user(u, v):
            r = apply(spec, L(*u), L(*v))
            return real_value(r, fname, "distance")
        return user
    raise ValueError(f"{fname}: {K_input(spec)} is not a valid DistanceFunction")


def color_distance(c1, c2, spec=None):
    f = distance_function(spec)
    return f(lab(c1), lab(c2))


@fn("ColorDistance", area=AREA, status="added",
    notes="ColorDistance[c1, c2] Euclidean distance of the LABColor values (CIE76 on Wolfram's 1/100 scale, opacity "
          "ignored); ColorDistance[list, c], [c, list] and [list1, list2] (equal lengths) thread. DistanceFunction -> "
          "\"CIE76\" (default), \"CIE94\" (graphic arts, kL = 1), \"CIE2000\" (Sharma et al. 2005), {\"CMC\", {l, c}}, "
          "{\"CMC\", \"Perceptibility\" | \"Acceptability\"} (1:1, 2:1), \"DeltaL\", \"DeltaC\", or a function f "
          "applied to the two {l, a, b} lists; \"DeltaH\" raises NotImplementedError (undocumented scaling); "
          "the ΔE formulas run on the 0-100 scale and are divided by 100. Images raise NotImplementedError, "
          "non-colours TypeError, unequal list lengths ValueError; symbolic colours stay unevaluated. "
          "Check: reference (Wolfram/Mathics documentation values, Sharma's CIEDE2000 test data) in tests.")
def _colordistance(c1: Expr | ListOf[Expr], c2: Expr | ListOf[Expr], *opts: Option) -> Real | ListOf[Real]:
    pos, o = _opts(list(opts), {"DistanceFunction"}, "ColorDistance")
    if pos:
        raise ValueError("ColorDistance: expected ColorDistance[c1, c2, options]")
    f = distance_function(o.get("DistanceFunction"))

    def d(a, b):
        pa, pb = _color_or_raise(a, "ColorDistance"), _color_or_raise(b, "ColorDistance")
        return float(f(convert_channels(pa[0], pa[1], "LAB"), convert_channels(pb[0], pb[1], "LAB")))

    if K.is_list(c1) and K.is_list(c2):
        if len(c1.args) != len(c2.args):
            raise ValueError("ColorDistance: the two lists of colours must have the same length")
        return L(*[d(a, b) for a, b in zip(c1.args, c2.args)])
    if K.is_list(c1):
        return L(*[d(a, c2) for a in c1.args])
    if K.is_list(c2):
        return L(*[d(c1, b) for b in c2.args])
    return d(c1, c2)


# ───────────────────────────── RandomColor ─────────────────────────────

_BOUNDED = {"RGB", "Grayscale", "HSB", "CMYK"}


def _random_model(model, fname):
    """(head, [sampler per channel]) for a colour model RGBColor[r, g, b], Hue[_, 1, 1], ..."""
    from .statistics import _RNG
    if isinstance(model, Symbol) and model.name in SPACES:
        model = Expr(model.name, *([Expr("Blank")] * {"Hue": 3, "GrayLevel": 1, "CMYKColor": 4}.get(model.name, 3)))
    if isinstance(model, Symbol) or (isinstance(model, Expr) and model.head not in SPACES and not K.is_list(model)):
        raise Unevaluated()                       # symbolic model
    if not (isinstance(model, Expr) and model.head in SPACES):
        raise TypeError(f"{fname}: {K_input(model)} is not a colour model such as RGBColor[_, _, _]")
    if len(model.args) not in _COUNTS[model.head]:
        raise ValueError(f"{fname}: {K_input(model)} has the wrong number of channels for a colour model")
    space = SPACES[model.head]
    samplers = []
    for a in model.args:
        if isinstance(a, Expr) and a.head == "Blank" and not a.args:
            if space not in _BOUNDED:
                raise NotImplementedError(f"{fname}: uniform channels (_) in the unbounded space {space} are not "
                                          "supported (Wolfram's visual-range limits are not documented); give "
                                          "constants or distributions")
            samplers.append(lambda: _RNG.random())
        elif is_real_value(a):
            v = real_value(a, fname)
            samplers.append(lambda v=v: v)
        elif isinstance(a, Expr) and a.head.endswith("Distribution"):
            from .wl_common import ev
            d = a

            def draw(d=d):
                return real_value(ev(Expr("RandomVariate", d)), fname, "random channel")
            samplers.append(draw)
        else:
            raise ValueError(f"{fname}: channel {K_input(a)} must be a number, _ or a distribution")
    return model.head, samplers


@fn("RandomColor", area=AREA, status="added",
    notes="RandomColor[] a uniform RGBColor[r, g, b]; RandomColor[n] n of them; RandomColor[model], "
          "RandomColor[model, n] and RandomColor[model, {n1, n2, ...}] for a model such as Hue[_, 1, 1] or "
          "RGBColor[_, _, _, 0.5] whose channels are constants, _ (uniform on [0, 1] for RGB, Grayscale, HSB, CMYK) "
          "or a distribution (sampled with RandomVariate); ColorSpace -> space converts the result. Uniform _ "
          "channels in XYZ/LAB/LCH/LUV raise NotImplementedError (visual-range limits undocumented). Reproducible "
          "after SeedRandom (the stream differs from Wolfram's). Symbolic models or counts stay unevaluated; a "
          "non-model (string, number) raises TypeError, wrong channel counts or negative counts ValueError. "
          "Check: self-check of channel ranges and ColorQ on results in tests.")
def _randomcolor(*args: Expr) -> Expr | ListOf[Expr]:
    pos, o = _opts(list(args), {"ColorSpace"}, "RandomColor")
    cs = o.get("ColorSpace")
    if cs is not None and not (isinstance(cs, Symbol) and cs.name == "Automatic"):
        cs = _space_arg(cs, "RandomColor")
    else:
        cs = None
    model, count = Expr("RGBColor", Expr("Blank"), Expr("Blank"), Expr("Blank")), None
    if len(pos) == 1:
        if K.is_int(pos[0]) or K.is_list(pos[0]):
            count = pos[0]
        else:
            model = pos[0]
    elif len(pos) == 2:
        model, count = pos
    elif len(pos) > 2:
        raise ValueError(f"RandomColor: expected at most 2 arguments, got {len(pos)}")
    head, samplers = _random_model(model, "RandomColor")

    def one():
        c = Expr(head, *[float(s()) for s in samplers])
        return convert(c, cs) if cs is not None else c

    def dims(spec):
        if K.is_int(spec):
            n = int(spec)
            if n < 0:
                raise ValueError("RandomColor: the number of colours must be non-negative")
            return [n]
        if K.is_list(spec) and spec.args and all(K.is_int(v) and int(v) >= 0 for v in spec.args):
            return [int(v) for v in spec.args]
        if isinstance(spec, (Symbol,)) or (isinstance(spec, Expr) and not K.is_list(spec)):
            raise Unevaluated()
        raise ValueError(f"RandomColor: {K_input(spec)} is not a non-negative integer or list of them")

    if count is None:
        return one()

    def build(ds):
        if len(ds) == 1:
            return L(*[one() for _ in range(ds[0])])
        return L(*[build(ds[1:]) for _ in range(ds[0])])
    return build(dims(count))


# ───────────────────────────── ColorsNear, options ─────────────────────────────

@fn("ColorsNear", area=AREA, status="added",
    notes="ColorsNear[color], ColorsNear[color, d] and ColorsNear[color, d, dfun] inert colour regions (d > 0 a "
          "maximum distance, dfun any ColorDistance DistanceFunction setting; default CIE2000, d = 0.2 as "
          "documented); the arguments are validated. Matching against it (in ColorReplace, ImageValuePositions) "
          "belongs to the image batches. Symbolic input stays unevaluated; a non-colour raises TypeError, d <= 0 or "
          "an unknown distance ValueError. Check: reference (Wolfram documentation forms) in tests.")
def _colorsnear(c: Expr, *rest: Real | Expr) -> Expr:
    if len(rest) > 2:
        raise ValueError(f"ColorsNear: expected 1 to 3 arguments, got {1 + len(rest)}")
    _color_or_raise(c, "ColorsNear")
    if rest:
        d = real_value(rest[0], "ColorsNear", "distance")
        if d <= 0:
            raise ValueError("ColorsNear: the distance must be positive")
    if len(rest) == 2:
        distance_function(rest[1], "ColorsNear")
    raise Unevaluated()


def inert_option(name, notes, area=AREA):
    """Register an option name as an inert symbol (Name[...] also stays unevaluated)."""
    @fn(name, area=area, status="added", notes=notes)
    def _opt(*args: Expr) -> Expr:
        raise Unevaluated()
    return _opt


inert_option("ColorSpace", "option of RandomColor (and Image, ColorConvert-related functions) naming the colour "
                           "space: \"Grayscale\", \"RGB\", \"CMYK\", \"HSB\", \"XYZ\", \"LAB\", \"LCH\", \"LUV\" or "
                           "Automatic; RandomColor validates it (unknown names raise ValueError). Inert as an "
                           "expression. Check: reference (Wolfram documentation settings) in tests.")
inert_option("WhitePoint", "option of ColorConvert (and ChromaticityPlot) giving the white point; ColorConvert "
                           "supports Automatic and \"D50\" (others raise NotImplementedError, invalid values "
                           "ValueError). Inert as an expression. Check: reference in tests.")
