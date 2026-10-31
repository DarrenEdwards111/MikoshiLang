"""Wolfram batch W15: options for axes, frames, ticks, legends, plots, charts and meshes.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

Most symbols of this batch are *option names* consumed by graphics containers (``Graphics``, ``Graphics3D``) and by
the plotting/charting functions of later batches (W16 2-D plots, W18 statistical charts, W20 3-D plots, ...). As in
Wolfram they are inert symbols (``PlotStyle`` and ``PlotStyle -> Red`` evaluate to themselves). What this module adds:

* a **validator** for every option (``OPTION_VALIDATORS[name](value, fname)``): a documented setting passes, a
  setting of the wrong kind or with an invalid value raises ``ValueError("fname: Name ...")`` (``TypeError`` for
  non-primitives in ``Epilog`` / ``Prolog``), and a setting containing user symbols where a concrete value is
  needed raises ``Unevaluated``. ``Graphics`` / ``Graphics3D`` call these validators for their options;
* **option resolution** for the plotting batches: ``resolve_options(fname, opts, names)`` -> {name: value} with
  Wolfram's documented defaults (``PLOT_DEFAULTS``), unknown names raising ``ValueError``;
  ``scaling_function(spec)`` -> (f, inverse) Python callables for the named ``ScalingFunctions``;
  ``tick_positions(lo, hi, n)`` -> FindDivisions-style nice tick positions;
* **rendering hooks** (``wl_graphics_render.RENDER_OPTIONS``) so that ``Rasterize`` draws ``Axes``, ``AxesOrigin``,
  ``AxesLabel``, ``AxesStyle``, ``Ticks``, ``TicksStyle``, ``Frame``, ``FrameLabel``, ``FrameStyle``,
  ``FrameTicks``, ``FrameTicksStyle``, ``GridLines``, ``GridLinesStyle``, ``PlotLabel``, ``RotateLabel``,
  ``Epilog`` and ``Prolog`` (unsupported settings of these raise ``NotImplementedError``);
* the legend constructors ``BarLegend``, ``LineLegend``, ``PointLegend``, ``SwatchLegend`` (validated inert
  expressions, as used with ``Legended`` and ``PlotLegends``), the ``AxisObject`` graphics primitive, the
  ``Axis`` symbol, ``FindDivisions``, and the system variables ``$DisplayFunction``, ``$ImageResolution``,
  ``$PerformanceGoal`` and ``$PlotTheme``.
"""
from __future__ import annotations

import math

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from ..symbol_values import SYMBOL_VALUES
from .core import fn, record, Unevaluated, L, REGISTRY, STATUS
from .wl_graphics_color import real_value, is_real_value, parse_color, K_input, inert_option
from . import wl_graphics_core as G
from .wl_graphics_core import is_sym, is_rule, opt_name, split_trailing_options, options_dict

AREA = "graphics: plot options (W15)"

SIZES = ("Tiny", "Small", "Medium", "Large")

# Symbols that are documented *settings* somewhere in this batch: a bare symbol outside this set (and not a
# built-in head) is treated as symbolic input (Unevaluated) rather than as an invalid setting.
_SETTING_SYMBOLS = {"Automatic", "All", "None", "Full", "Inherited", "Left", "Right", "Top", "Bottom", "Center",
                    "Axis", "Above", "Below", "Before", "After", "Up", "Down", "Front", "Back", "Tiny", "Small",
                    "Medium", "Large", "Infinity", "Identity", "Coarse", "Fine", "Baseline", "Tooltip",
                    "StatusArea", "Null", "Indeterminate", "ComplexInfinity"}


# ───────────────────────────── predicates (never raise) ─────────────────────────────

def _is_bool(x):
    return x is True or x is False


def _num(x):
    """Float value of a numeric real, else None (symbolic, strings, lists, booleans -> None)."""
    if isinstance(x, (bool, str)):
        return None
    try:
        return real_value(x, "", "x")
    except (Unevaluated, TypeError, ValueError):
        return None


def _real(lo=None, hi=None, lo_open=False, hi_open=False):
    def p(x):
        v = _num(x)
        if v is None or not math.isfinite(v):
            return False
        if lo is not None and (v < lo or (lo_open and v == lo)):
            return False
        if hi is not None and (v > hi or (hi_open and v == hi)):
            return False
        return True
    return p


def _int(lo=None):
    def p(x):
        if not K.is_int(x):
            return False
        return lo is None or int(x) >= lo
    return p


def _sym(*names):
    return lambda x: is_sym(x, *names)


def _string(*choices):
    return lambda x: isinstance(x, str) and (not choices or x in choices)


def _head(*heads):
    return lambda x: isinstance(x, Expr) and x.head in heads


def _list(pred=None, lens=None, minlen=0):
    def p(x):
        if not K.is_list(x):
            return False
        if lens is not None and len(x.args) not in lens:
            return False
        if len(x.args) < minlen:
            return False
        return pred is None or all(pred(a) for a in x.args)
    return p


def _tuple(*preds):
    def p(x):
        return K.is_list(x) and len(x.args) == len(preds) and all(q(a) for q, a in zip(preds, x.args))
    return p


def _rule(lp=None, rp=None):
    def p(x):
        return is_rule(x) and (lp is None or lp(x.args[0])) and (rp is None or rp(x.args[1]))
    return p


def _any_of(*preds):
    return lambda x: any(q(x) for q in preds)


def _anything(x):
    return True


def _not_bool(x):
    return not _is_bool(x)


def _is_number_like(x):
    return isinstance(x, (int, float)) or (not isinstance(x, (str, Symbol)) and _num(x) is not None)


def _func(x):
    """Something applicable: a symbol, a pure function, or another non-numeric, non-list expression."""
    if _is_bool(x) or isinstance(x, str) or x is None:
        return False
    if isinstance(x, Symbol):
        return x.name not in _SETTING_SYMBOLS
    if isinstance(x, Expr):
        if x.head in ("List", "Rule", "RuleDelayed", "Association"):
            return False
        return _num(x) is None
    return False


def _style(allow_str=False, allow_num=False, allow_rules=False):
    """A graphics directive / style specification (colours, Thickness[..], Directive[..], lists of them, ...)."""
    def p(x):
        if K.is_list(x):
            return all(p(a) for a in x.args)
        if _is_bool(x):
            return False
        if isinstance(x, str):
            return allow_str
        if is_rule(x):
            return allow_rules
        if isinstance(x, Symbol):
            return True
        if _is_number_like(x):
            return allow_num and (_num(x) or 0) > 0
        return isinstance(x, Expr)
    return p


STYLE = _style()
TEXT_STYLE = _style(allow_str=True, allow_num=True, allow_rules=True)


def _user_symbols(x):
    """True when x contains a symbol that is neither a documented setting nor a known built-in."""
    if isinstance(x, Symbol):
        n = x.name
        return not (n in _SETTING_SYMBOLS or n in REGISTRY or n in STATUS or n in SYMBOL_VALUES
                    or n in ("Pi", "E", "Degree", "GoldenRatio", "EulerGamma", "I", "True", "False"))
    if isinstance(x, Expr):
        return any(_user_symbols(a) for a in x.args) or (isinstance(x.head, Expr) and _user_symbols(x.head))
    return False


def make_validator(name, msg, *preds, check=None):
    """Validator for option ``name``: accepted when any predicate holds (then ``check(x, fname)`` may raise
    ValueError for a value-level problem); symbolic settings raise Unevaluated; anything else ValueError."""
    def v(x, fname):
        if x is None:
            x = Symbol("None")
        for q in preds:
            if q(x):
                if check is not None:
                    check(x, fname)
                return
        if _user_symbols(x):
            raise Unevaluated()
        raise ValueError(f"{fname}: {name} must be {msg}, not {K_input(x)}")
    v.__name__ = f"validate_{name}"
    v.msg = msg
    return v


# ───────────────────────────── shared value spaces ─────────────────────────────

POS = _real(0, lo_open=True)
NONNEG = _real(0)
REAL = _real()
UNIT = _real(0, 1)
POSINT = _int(1)
NONNEGINT = _int(0)
AUTO = _sym("Automatic")
NONE = _sym("None")
AUTO_NONE = _sym("Automatic", "None")
SIZE_SYM = _sym(*SIZES)
SCALED = _head("Scaled")
LABEL = _not_bool          # any expression can be a label (strings, graphics, formulas)


def _pair_ordered(name):
    """{min, max} with min < max when both are numbers."""
    def chk(x, fname):
        a, b = _num(x.args[0]), _num(x.args[1])
        if a is not None and b is not None and not a < b:
            raise ValueError(f"{fname}: {name} range {K_input(x)} must have min < max")
    return chk


_RANGE_END = _any_of(REAL, _sym("Automatic", "All", "Full"))
_RANGE_PAIR = _tuple(_RANGE_END, _RANGE_END)


def _check_ranges(name):
    def chk(x, fname):
        items = []
        if K.is_list(x) and len(x.args) == 2 and all(_RANGE_END(a) for a in x.args):
            items = [x]
        elif K.is_list(x):
            items = [a for a in x.args if K.is_list(a)]
        for it in items:
            _pair_ordered(name)(it, fname)
        if _num(x) is not None and _num(x) <= 0:
            raise ValueError(f"{fname}: {name} -> m needs m > 0, not {K_input(x)}")
    return chk


# ImageSize-like size specifications
_DIM = _any_of(POS, _sym("Automatic", "Full", "All", *SIZES), _head("Scaled", "UpTo"))
_DIM_RANGE = _tuple(_any_of(NONNEG, AUTO), _any_of(POS, _sym("Automatic", "Infinity")))
SIZE_SPEC = _any_of(_DIM, _tuple(_DIM, _DIM), _tuple(_DIM_RANGE, _DIM_RANGE))

# per-side margins: m or {{left, right}, {bottom, top}}
_SIDE = _any_of(NONNEG, AUTO)
SIDES = _any_of(NONNEG, _tuple(_tuple(_SIDE, _SIDE), _tuple(_SIDE, _SIDE)))

# tick and grid specifications for one axis
_TICK_ITEM = _any_of(REAL, _list(None, minlen=1))


def _tick_item_ok(t):
    if REAL(t):
        return True
    if K.is_list(t) and 1 <= len(t.args) <= 4 and REAL(t.args[0]):
        if len(t.args) >= 3:
            ln = t.args[2]
            if not (REAL(ln) or _tuple(REAL, REAL)(ln)):
                return False
        return True
    return False


TICK_AXIS = _any_of(_sym("None", "Automatic", "All"), _is_bool, _list(_tick_item_ok), _func)
_GRID_ITEM = _any_of(REAL, _tuple(REAL, STYLE))
GRID_AXIS = _any_of(_sym("None", "Automatic"), _list(_GRID_ITEM), _func)

PLOT_THEMES = ("Default", "Business", "Detailed", "Marketing", "Minimal", "Monochrome", "Scientific", "Web",
               "Classic", "NoAxes", "MinimalAxes", "Grid", "SingleAxis", "FullAxes", "Frame", "HeightGrid",
               "FullAxesGrid", "FrameGrid", "BlackBackground", "BackgroundColor", "BoldColor", "CoolColor",
               "DarkColor", "GrayColor", "NeonColor", "PastelColor", "RoyalColor", "VibrantColor", "WarmColor",
               "BoldLabels", "ItalicLabels", "LargeLabels", "SmallLabels", "SansLabels", "SerifLabels",
               "Sparkline", "Icon", "Thumbnail", "Square", "Tall", "Wide", "SizeScale", "OpenMarkers",
               "OpenMarkersThick", "PlotMarkers", "ThinLines", "MediumLines", "ThickLines", "DashedLines",
               "DarkMesh", "GrayMesh", "LightMesh", "ZMesh", "BoldScheme", "FadeScheme", "ThickSurface",
               "FilledSurface")
_THEME = _any_of(_string(*PLOT_THEMES), _tuple(_string("BackgroundColor"), lambda c: parse_color(c) is not None))

SCALING_NAMES = ("Log", "Log2", "Log10", "SignedLog", "Date", "Time", "Reverse", "Reciprocal", "Infinite")
_SCALING_ONE = _any_of(_sym("None", "Automatic"), _string(*SCALING_NAMES),
                       _tuple(_string("SignedLog"), POS), _tuple(_string("Infinite"), _anything),
                       _tuple(_func, _func), _head("NominalScale", "OrdinalScale"))
LEGEND_LAYOUTS = ("Row", "Column", "ReversedRow", "ReversedColumn")
_LEGEND_HEADS = ("BarLegend", "LineLegend", "PointLegend", "SwatchLegend", "Placed", "Legended")

_ORIENT = _any_of(_string("Horizontal", "Vertical", "Parallel", "Perpendicular"), REAL)
_TICKDIR = _any_of(_sym("Automatic", "Left", "Right", "Up", "Down"), _string("Inward", "Outward"))
_TICKPOS = _any_of(_sym("Automatic", "None"), _string("Base", "Tip"), _sym("Before", "After"))
_TICKPOSITIONS = _any_of(AUTO, POSINT, _tuple(POS), _tuple(REAL, REAL, POS), _tuple(_list(REAL)))
_TICKLEN = _any_of(AUTO, NONNEG)
_TICKLABELS = _any_of(_sym("Automatic", "All", "None"), _list(_anything))


def _face(x):
    if not (K.is_list(x) and len(x.args) == 3 and all(K.is_int(c) for c in x.args)):
        return False
    vals = sorted(int(c) for c in x.args)
    return vals in ([-1, 0, 0], [0, 0, 1])


def _edge_dirs(x):
    return _any_of(_sym("Automatic", "None"), _tuple(_any_of(_int(), AUTO), _any_of(_int(), AUTO)))(x) and \
        (not K.is_list(x) or all(not K.is_int(c) or int(c) in (-1, 1) for c in x.args))


def _frame_edges(x):
    b = _any_of(_is_bool, _sym("None", "All", "Automatic"))
    return _tuple(_tuple(b, b), _tuple(b, b))(x) or _list(b, lens=(2, 4))(x)


def _content_check(name):
    def chk(x, fname):
        G.check_content(x, 2, fname)
    return chk


def _content(x):
    return not _is_bool(x) and not isinstance(x, str) and not _is_number_like(x)


def _check_real_pairs(name, lo=None):
    def chk(x, fname):
        for p in _iter_pairs(x):
            a, b = _num(p.args[0]), _num(p.args[1])
            if a is not None and b is not None:
                if a > b:
                    raise ValueError(f"{fname}: {name} {K_input(p)} must have min <= max")
                if lo is not None and a < lo:
                    raise ValueError(f"{fname}: {name} values must be >= {lo}")
    return chk


def _iter_pairs(x):
    if K.is_list(x) and len(x.args) == 2 and all(_num(a) is not None for a in x.args):
        yield x
    elif K.is_list(x):
        for a in x.args:
            yield from _iter_pairs(a)


def _filling_item(x):
    return _any_of(_sym("None", "Axis", "Top", "Bottom", "Automatic"), REAL, _tuple(POSINT))(x)


def _filling_target(x):
    return _filling_item(x) or (K.is_list(x) and len(x.args) == 2 and _filling_item(x.args[0])
                                and (STYLE(x.args[1]) or _tuple(STYLE, STYLE)(x.args[1])))


_FILL_RULE = _rule(POSINT, _filling_target)


def _mesh_one(x):
    return _any_of(_sym("None", "Automatic", "All", "Full"), NONNEGINT,
                   _list(_any_of(REAL, _tuple(REAL, STYLE))))(x)


def _contours(x):
    return _any_of(AUTO_NONE, NONNEGINT, _tuple(AUTO, POSINT), _list(_any_of(REAL, _tuple(REAL, STYLE))),
                   _func)(x)


def _frame_spec(x):
    """Frame settings: True/False/None/All/Automatic, {{left, right}, {bottom, top}}, or Grid-style lists and
    rules ({All, False}, {-1 -> True}, {False, {{True}}}, ...)."""
    if _is_bool(x) or is_sym(x, "None", "All", "Automatic"):
        return True
    if K.is_list(x):
        return all(_frame_spec(a) for a in x.args)
    if is_rule(x):
        return _frame_spec(x.args[1])
    return False


_COLORFN = _any_of(AUTO_NONE, _string(), _func)
_LEGENDS = _any_of(_sym("None", "Automatic"), _string(), _list(_anything), _head(*_LEGEND_HEADS),
                   lambda x: _func(x) and not _is_bool(x))
_LABELS = _any_of(_sym("None", "Automatic"), _string("Expressions"), _list(_anything), _head("Placed", "Callout"))
_MARKER_SIZE = _any_of(SIZE_SYM, POS, SCALED, _head("Offset"))

# (name, accepted forms (for messages), predicates, value check or None, notes)
_SPECS = [
    # ── ranges, sizes, shape ──
    ("PlotRange", "All, Full, Automatic, m > 0, {min, max} or {{xmin, xmax}, {ymin, ymax}(, {zmin, zmax})}",
     [_sym("All", "Full", "Automatic"), POS, _RANGE_PAIR, _list(_any_of(_RANGE_PAIR, _sym("All", "Full", "Automatic")),
                                                                 lens=(1, 2, 3))],
     _check_ranges("PlotRange"),
     "option of Graphics, Graphics3D and plotting functions: All, Full, Automatic, max, {min, max}, "
     "{{xmin, xmax}, {ymin, ymax}} or with {zmin, zmax}; any limit may be Automatic/All/Full; min >= max raises."),
    ("ImageSize", "w, {w, h}, UpTo[s], {{wmin, wmax}, {hmin, hmax}}, Automatic, Full, Tiny, Small, Medium or Large",
     [SIZE_SPEC], None,
     "option of graphics and layout constructs: w, {w, h}, UpTo[s], {UpTo[w], UpTo[h]}, {{wmin, wmax}, {hmin, "
     "hmax}}, Automatic, Full, All, Scaled[s] or Tiny/Small/Medium/Large; sizes in printer's points must be > 0."),
    ("ContentSize", "w, {w, h}, UpTo[s], {{wmin, wmax}, {hmin, hmax}}, Automatic, Full or Scaled[s]",
     [SIZE_SPEC], None,
     "option of Manipulate, Labeled and similar: the content area size (w, {w, h}, UpTo[s], ranges, Automatic, "
     "Full, Scaled[s]); sizes must be > 0."),
    ("AspectRatio", "Automatic, Full or a positive number", [_sym("Automatic", "Full"), POS], None,
     "option of graphics: height-to-width ratio, Automatic (from the coordinates) or Full; non-positive ratios "
     "raise."),
    ("DataRange", "Automatic, All, {min, max} or {{xmin, xmax}, {ymin, ymax}, ...}",
     [_sym("Automatic", "All"), _tuple(REAL, REAL), _list(_tuple(REAL, REAL), minlen=1)],
     _check_real_pairs("DataRange"),
     "option of list plots and ArrayPlot: the coordinate range the data spans, {xmin, xmax} or one pair per "
     "dimension, Automatic or All; min > max raises."),
    ("PlotRangePadding", "Automatic, None, s, Scaled[s], {px, py, ...} or {{pxL, pxR}, ...}",
     [_any_of(_sym("Automatic", "None", "All"), REAL, SCALED),
      _list(_any_of(_sym("Automatic", "None", "All"), REAL, SCALED,
                    _list(_any_of(_sym("Automatic", "None", "All"), REAL, SCALED), lens=(2,))), lens=(1, 2, 3))],
     None, "option of graphics: padding added to the plot range, Automatic, None, s, Scaled[s], one per direction "
           "or {left, right} pairs."),
    ("PlotRegion", "Automatic or {{sxmin, sxmax}, {symin, symax}}",
     [AUTO, _tuple(_tuple(REAL, REAL), _tuple(REAL, REAL))], _check_real_pairs("PlotRegion"),
     "option of graphics: the scaled region {{sxmin, sxmax}, {symin, symax}} of the display area the plot fills "
     "(Automatic = {{0, 1}, {0, 1}})."),
    ("PlotRangeClipping", "True or False", [_is_bool], None,
     "option of graphics: whether objects are clipped at the plot range (False for Graphics, True for Plot)."),
    ("ClipRange", "None, All or {{xmin, xmax}, {ymin, ymax}, {zmin, zmax}}",
     [_sym("None", "All", "Automatic"), _tuple(_tuple(REAL, REAL), _tuple(REAL, REAL), _tuple(REAL, REAL))],
     _check_real_pairs("ClipRange"),
     "option of Raster3D (and Image3D): the box {{xmin, xmax}, {ymin, ymax}, {zmin, zmax}} cut away from the view "
     "(default None)."),
    ("ImageMargins", "m >= 0, {{left, right}, {bottom, top}} or Automatic", [_any_of(SIDES, AUTO)], None,
     "option of graphics and boxes: margins outside ImageSize in printer's points, m or {{l, r}, {b, t}}."),
    ("ImagePadding", "None, All, Automatic, m >= 0 or {{left, right}, {bottom, top}}",
     [_sym("None", "All", "Automatic"), SIDES,
      _tuple(_tuple(_any_of(_SIDE, _sym("All")), _any_of(_SIDE, _sym("All"))),
             _tuple(_any_of(_SIDE, _sym("All")), _any_of(_SIDE, _sym("All"))))], None,
     "option of graphics: padding inside ImageSize around the plot range, None, All, Automatic, m or "
     "{{l, r}, {b, t}} in printer's points."),
    ("FrameMargins", "Automatic, None, m >= 0 or {{left, right}, {bottom, top}}", [_sym("Automatic", "None"), SIDES],
     None, "option of Framed, Panel, legends and similar: margins inside the frame, m or {{l, r}, {b, t}}."),
    ("LegendMargins", "Automatic, m >= 0 or {{left, right}, {bottom, top}}", [AUTO, SIDES], None,
     "option of the legend constructors: space between legend content and its frame, m or {{l, r}, {b, t}}."),
    ("ImageResolution", "Automatic, $ImageResolution or a positive dpi value", [AUTO, POS], None,
     "option of Rasterize, Export and images: bitmap resolution in dots per inch (Automatic: 72 dpi)."),
    ("ImageSizeAction", "\"Clip\", \"ResizeToFit\", \"Scrollable\", \"ShrinkToFit\" or Automatic",
     [AUTO, _string("Clip", "ResizeToFit", "Scrollable", "ShrinkToFit")], None,
     "option of panes and graphics: what to do when contents do not fit ImageSize."),
    ("ImageSizeMultipliers", "a positive number or {u, v}", [POS, _tuple(POS, POS)], None,
     "option of graphics: scale factors {u, v} for Automatic-sized graphics inside lists and inside function-like "
     "constructs (typical default {0.5, 0.25})."),
    # ── axes ──
    ("Axes", "True, False or a list of True/False per axis", [_is_bool, _list(_is_bool, lens=(1, 2, 3))], None,
     "option of graphics: whether to draw axes, True, False or {xaxis, yaxis(, zaxis)}."),
    ("AxesOrigin", "Automatic or a point {x, y} / {x, y, z}", [AUTO, _list(_any_of(REAL, AUTO), lens=(2, 3))], None,
     "option of graphics: the point where the axes cross (2D) or the axes are placed (3D), or Automatic."),
    ("AxesEdge", "Automatic, None or {{diry, dirz}, {dirx, dirz}, {dirx, diry}} with dir = +1 or -1",
     [_sym("Automatic", "None"), _list(_edge_dirs, lens=(3,))], None,
     "option of Graphics3D: on which edges of the bounding box the three axes are drawn; each pair may be "
     "Automatic or None."),
    ("AxesLabel", "None, Automatic, a label or {xlabel, ylabel(, zlabel)}",
     [_sym("None", "Automatic"), _list(_anything, lens=(1, 2, 3)), lambda x: not K.is_list(x) and LABEL(x)], None,
     "option of graphics: axis labels, None, Automatic, a label (for y in 2D, z in 3D) or {xlabel, ylabel(, "
     "zlabel)}."),
    ("AxesStyle", "a graphics directive, a list of directives per axis, or a style name",
     [TEXT_STYLE], None, "option of graphics: style of the axes (and default style of their ticks and labels)."),
    ("Ticks", "None, Automatic, a tick specification or {xticks, yticks(, zticks)}",
     [_sym("None", "Automatic", "All"), _is_bool, _list(TICK_AXIS, lens=(1, 2, 3)), _func], None,
     "option of graphics: tick marks on the axes, None, Automatic or per axis None/Automatic/All/{x1, ...}/"
     "{{x1, label1(, len1(, style1))}, ...}/function."),
    ("TicksStyle", "a graphics directive, a list per axis, or a style name", [TEXT_STYLE], None,
     "option of graphics: style of tick marks and tick labels."),
    ("TickDirection", "Automatic, \"Inward\", \"Outward\", Left, Right, Up, Down or a list of these",
     [_TICKDIR, _list(_TICKDIR)], None,
     "option of AxisObject: on which side of the axis ticks are drawn (one setting or one per tick collection)."),
    ("TickLabelOrientation", "\"Horizontal\", \"Vertical\", \"Parallel\", \"Perpendicular\", an angle or a list of "
                             "these", [_ORIENT, AUTO_NONE, _list(_any_of(_ORIENT, AUTO_NONE))], None,
     "option of AxisObject: rotation of tick labels (default None)."),
    ("TickLabelPositioning", "\"Base\", \"Tip\", Before, After or a list of these", [_TICKPOS, _list(_TICKPOS)], None,
     "option of AxisObject: where tick labels sit relative to the ticks."),
    ("TickLabels", "Automatic, All, None or a list of per-collection specs", [_TICKLABELS], None,
     "option of AxisObject: labels of each collection of ticks (Automatic, All, None or explicit label lists)."),
    ("TickLengths", "Automatic, a scaled length s >= 0 or a list of these", [_TICKLEN, _list(_TICKLEN)], None,
     "option of AxisObject: scaled tick lengths, one setting or one per tick collection."),
    ("TickPositions", "Automatic, n, {dx}, {min, max, dx}, {{t1, t2, ...}} or a list of these",
     [_TICKPOSITIONS, _list(_TICKPOSITIONS)], None,
     "option of AxisObject: where ticks are placed in each tick collection."),
    ("AxisLabel", "None or a label (Placed[label, pos] for a position)", [_any_of(NONE, LABEL)], None,
     "option of AxisObject: the axis label, any expression or Placed[label, \"Start\"|\"End\"|Center|...]."),
    ("AxisStyle", "a graphics directive", [STYLE], None,
     "option of AxisObject: style of the axis path only (ticks and labels unaffected)."),
    ("PolarAxes", "True, False, Automatic or {angular, radial}", [_is_bool, AUTO, _tuple(_is_bool, _is_bool)], None,
     "option of polar plots: whether to draw the angular and radial axes."),
    ("PolarAxesOrigin", "Automatic, {θ, r} or {{pos, dir}, r}",
     [AUTO, _tuple(REAL, REAL),
      _tuple(_tuple(_sym("Left", "Right", "Top", "Bottom"), _sym("Up", "Down", "Left", "Right")), REAL)], None,
     "option of polar plots: angle of the radial axis and radius of the angular axis."),
    ("PolarGridLines", "None, Automatic or {radial, angular}", [_sym("None", "Automatic"), _tuple(GRID_AXIS, GRID_AXIS)],
     None, "option of polar plots: radial (angles in radians) and angular grid lines, each None/Automatic/positions/"
           "{{t, style}, ...}/function."),
    ("PolarTicks", "None, Automatic or {θticks, rticks}",
     [_sym("None", "Automatic"),
      _tuple(_any_of(TICK_AXIS, _string("Degrees", "Radians", "Percent")), TICK_AXIS)], None,
     "option of polar plots: angular (Automatic, \"Degrees\", \"Radians\", \"Percent\", explicit) and radial ticks."),
    ("MultiaxisArrangement", "None, All or a list of axis groups (optionally Left -> ..., Right -> ...)",
     [_sym("None", "All"),
      _list(_any_of(POSINT, _list(POSINT), _rule(_sym("Left", "Right"), _any_of(POSINT, _list(_any_of(POSINT, _list(POSINT)))))))],
     None, "option of plotting functions: which plotted items get their own vertical axes."),
    # ── frames ──
    ("Frame", "True, False, None, All, {{left, right}, {bottom, top}} or a Grid-style frame specification",
     [_frame_spec], None,
     "option of graphics and Grid-like constructs: True, False/None, All, {{left, right}, {bottom, top}} "
     "(graphics) or Grid-style specifications ({All, False}, {-1 -> True}, ...)."),
    ("FrameLabel", "None, a label, {bottom, left} or {{left, right}, {bottom, top}}",
     [NONE, _list(_anything, lens=(1, 2, 3, 4)), lambda x: not K.is_list(x) and LABEL(x)], None,
     "option of graphics: frame labels, None, label (bottom), {bottom, left}, {bottom, left, top, right} or "
     "{{left, right}, {bottom, top}}."),
    ("FrameStyle", "a graphics directive or {{left, right}, {bottom, top}} of directives", [TEXT_STYLE], None,
     "option of graphics and boxes: style of the frame edges."),
    ("FrameTicks", "None, Automatic, True, All or {{left, right}, {bottom, top}} of tick specifications",
     [_sym("None", "Automatic", "All"), _is_bool, _tuple(_tuple(TICK_AXIS, TICK_AXIS), _tuple(TICK_AXIS, TICK_AXIS)),
      _list(TICK_AXIS, lens=(2, 4)), _func], None,
     "option of graphics: frame ticks, None, Automatic, True (bottom and left), All, or per edge "
     "{{left, right}, {bottom, top}} with None/Automatic/All/positions/{{x, label, ...}, ...}/function."),
    ("FrameTicksStyle", "a graphics directive or a list per edge", [TEXT_STYLE], None,
     "option of graphics: style of frame ticks and frame tick labels."),
    ("RotateLabel", "True, False or a rotation (\"Horizontal\", \"Vertical\", \"Parallel\", \"Perpendicular\", θ)",
     [_is_bool, _ORIENT], None,
     "option of graphics (and AxisObject): whether vertical frame labels are rotated."),
    # ── grid lines, labels, styles ──
    ("GridLines", "None, Automatic or {xgrid, ygrid}", [_sym("None", "Automatic", "Full"), _tuple(GRID_AXIS, GRID_AXIS)],
     None, "option of graphics: grid lines, None, Automatic or {xgrid, ygrid} with each None, Automatic, "
           "{x1, ...}, {{x1, style1}, ...} or a function."),
    ("GridLinesStyle", "a graphics directive or {xstyle, ystyle}", [TEXT_STYLE], None,
     "option of graphics: style of grid lines."),
    ("FaceGrids", "None, All or a list of faces {dirx, diry, dirz} (optionally {face, {xgrid, ygrid}})",
     [_sym("None", "All"), _list(_any_of(_face, _tuple(_face, _tuple(GRID_AXIS, GRID_AXIS))))], None,
     "option of Graphics3D: grid lines on faces of the bounding box, a face being {0, 0, -1} etc."),
    ("FaceGridsStyle", "a graphics directive or {xgrid, ygrid, zgrid}", [TEXT_STYLE], None,
     "option of Graphics3D: style of face grids."),
    ("PlotLabel", "None or a label", [_anything], None,
     "option of graphics: overall label (any expression; strings are shown as text)."),
    ("LabelStyle", "a style specification (directives, a style name, a font size or option rules)", [TEXT_STYLE],
     None, "option of graphics, legends and other constructs: style of label-like elements; negative font sizes "
           "and booleans raise."),
    ("BaseStyle", "a style specification", [TEXT_STYLE], None,
     "option of constructs: base style applied to the whole object (like Style[obj, spec])."),
    ("DefaultBaseStyle", "a style specification", [TEXT_STYLE], None,
     "option of constructs: default style applied before BaseStyle (e.g. \"Graphics\")."),
    ("DefaultAxesStyle", "a style specification", [TEXT_STYLE], None,
     "option of graphics: default style of axes, applied before AxesStyle."),
    ("DefaultFaceGridsStyle", "a style specification", [TEXT_STYLE], None,
     "option of Graphics3D: default style of face grids, applied before FaceGridsStyle."),
    ("DefaultFrameStyle", "a style specification", [TEXT_STYLE], None,
     "option of graphics and boxes: default frame style, applied before FrameStyle."),
    ("DefaultFrameTicksStyle", "a style specification", [TEXT_STYLE], None,
     "option of graphics: default style of frame ticks, applied before FrameTicksStyle."),
    ("DefaultGridLinesStyle", "a style specification", [TEXT_STYLE], None,
     "option of graphics: default style of grid lines, applied before GridLinesStyle."),
    ("DefaultLabelStyle", "a style specification", [TEXT_STYLE], None,
     "option of constructs: default label style, applied before LabelStyle."),
    ("DefaultTicksStyle", "a style specification", [TEXT_STYLE], None,
     "option of graphics: default style of ticks, applied before TicksStyle."),
    ("Epilog", "graphics primitives and directives", [_content], _content_check("Epilog"),
     "option of graphics: 2D primitives drawn after the main content (in 3D in a {0, 1} coordinate system); "
     "numbers or strings as primitives raise TypeError."),
    ("Prolog", "graphics primitives and directives", [_content], _content_check("Prolog"),
     "option of graphics: 2D primitives drawn before the main content (after axes and frames); numbers or strings "
     "as primitives raise TypeError."),
    ("EpilogFunction", "None, a function, File[...] or a notebook/CloudObject",
     [NONE, _func], None,
     "option of DocumentGenerator: code run after a document is generated (function, notebook, CloudObject or "
     "File)."),
    # ── plot styling ──
    ("PlotStyle", "Automatic, None, a graphics directive or a list of directives", [STYLE], None,
     "option of plotting functions: directive(s) for the main objects, used cyclically for multiple objects."),
    ("PlotTheme", "Automatic, None, a theme name such as \"Scientific\" or a list of theme names",
     [AUTO_NONE, _THEME, _list(_THEME)], None,
     "option of visualisation functions: a base theme (\"Business\", \"Detailed\", \"Marketing\", \"Minimal\", "
     "\"Monochrome\", \"Scientific\", \"Web\", \"Classic\") and/or feature themes (\"Frame\", \"Grid\", "
     "\"BoldColor\", \"ThickLines\", \"OpenMarkers\", ...); unknown theme names raise ValueError (Wolfram warns)."),
    ("ColorFunction", "Automatic, None, a gradient name or a function", [_COLORFN], None,
     "option of plots and Raster: function giving the colour of each element, a ColorData gradient name, "
     "Automatic or None."),
    ("ColorFunctionScaling", "True, False or a list of True/False", [_is_bool, _list(_is_bool)], None,
     "option of plots and Raster: whether arguments of ColorFunction are scaled to [0, 1]."),
    ("ColorFunctionBinning", "None, Automatic, n, \"KMeans\", \"Quantile\", {d}, {min, max, d}, {{b1, b2, ...}}, "
                             "{\"name\", n} or a function",
     [_sym("None", "Automatic"), POSINT, _string("KMeans", "Quantile"), _tuple(POS), _tuple(REAL, REAL, POS),
      _tuple(_list(REAL)), _tuple(_string("KMeans", "Quantile"), POSINT), _func], None,
     "option of plotting functions: divides values into bins for colouring."),
    ("ColorRules", "Automatic, None or a list of rules value -> colour", [AUTO_NONE, _list(is_rule), is_rule], None,
     "option of ArrayPlot and similar: rules (patterns allowed) giving the colour of cell values."),
    ("ColorCoverage", "a fraction f in [0, 1], {fmin, fmax} or Automatic",
     [AUTO, UNIT, _tuple(UNIT, UNIT)], _check_real_pairs("ColorCoverage"),
     "option of DominantColors and FindMatchingColor: fraction of the input to be covered."),
    ("Filling", "None, Axis, Top, Bottom, a value, {m}, or rules {i -> spec, ...}",
     [_filling_item, _list(_any_of(_FILL_RULE, POSINT)), _FILL_RULE], None,
     "option of plotting functions: filling under curves (None, Axis, Bottom, Top, a value v, {m}, i -> p, "
     "i -> {p, style}, i -> {p, {below, above}})."),
    ("FillingStyle", "a graphics directive or {below, above}", [STYLE, _tuple(STYLE, STYLE)], None,
     "option of plotting functions: style of filling."),
    ("Joined", "True, False or a list of True/False", [_is_bool, _list(_is_bool)], None,
     "option of list plots and LineLegend/PointLegend: whether points are joined by lines."),
    ("Mesh", "None, Automatic, All, Full, n or a list of mesh specifications",
     [_mesh_one, _list(_mesh_one)], None,
     "option of plotting functions: mesh divisions, None, n (>= 0), All, Full, {a1, ...}, {{a1, g1}, ...}, or one "
     "specification per mesh function."),
    ("MeshFunctions", "Automatic or a list of functions", [AUTO, _list(_func), _func], None,
     "option of plotting functions: functions whose level sets give the mesh."),
    ("MeshStyle", "a graphics directive or a list of directives", [STYLE], None,
     "option of plotting functions: style of mesh lines or points."),
    ("MeshShading", "None, Automatic or a (nested) list of directives", [AUTO_NONE, _list(STYLE)], None,
     "option of plotting functions: directives used cyclically for regions between mesh lines."),
    ("BoundaryStyle", "Automatic, None or a graphics directive", [STYLE], None,
     "option of plotting functions: style of the boundary of regions defined by RegionFunction."),
    ("ClippingStyle", "Automatic, None, a directive or {bottom, top}", [STYLE], None,
     "option of plotting functions: how clipped parts of curves and surfaces are shown."),
    ("ExclusionsStyle", "None, Automatic, a style or {style, boundary style}", [STYLE], None,
     "option of plotting functions: style of excluded regions and their boundaries."),
    ("Exclusions", "None, All, Automatic, \"Discontinuities\", \"Singularities\", an equation or a list of these",
     [_sym("None", "All", "Automatic"), _string("Discontinuities", "Singularities"), _head("Equal"), REAL,
      _list(_any_of(_string("Discontinuities", "Singularities"), _head("Equal"), REAL,
                    _tuple(_head("Equal"), _anything)))], None,
     "option of plotting functions: points, curves or named classes of points to exclude."),
    ("RegionFunction", "a function returning True for points to include", [_func], None,
     "option of plotting functions: predicate selecting the plotted region (default (True &))."),
    ("RegionBoundaryStyle", "Automatic, None or a graphics directive", [STYLE], None,
     "option of discrete plots: style of the region boundary."),
    ("RegionFillingStyle", "Automatic, None or a graphics directive", [STYLE], None,
     "option of discrete plots: style of the filled region."),
    ("ReferenceLineStyle", "None or a graphics directive (or a list)", [STYLE], None,
     "option of plots with reference lines (e.g. QuantilePlot): style of the reference line(s)."),
    ("Contours", "Automatic, n, {Automatic, n}, a list of levels or a function", [_contours], None,
     "option of contour plots: number or levels of contours (n >= 0)."),
    ("ContourStyle", "Automatic, None, a directive or a list of directives", [STYLE], None,
     "option of contour plots: style of contour lines or surfaces."),
    ("ContourLabels", "None, Automatic, True, All, a function or {f, g}",
     [_sym("None", "Automatic", "All"), _is_bool, _func, _tuple(_func, _func)], None,
     "option of contour plots: labelling of contour lines."),
    ("PerformanceGoal", "\"Speed\", \"Quality\", Automatic or $PerformanceGoal",
     [AUTO, _string("Speed", "Quality"), _head("ControlActive")], None,
     "option of graphics and algorithms: what to optimise for."),
    ("PlotPoints", "Automatic, n >= 1 or {n1, n2, ...}", [AUTO, POSINT, _list(_any_of(POSINT, AUTO))], None,
     "option of plotting functions: initial number of sample points."),
    ("MaxRecursion", "Automatic, Infinity or n >= 0", [_sym("Automatic", "Infinity"), NONNEGINT], None,
     "option of plotting functions: maximum levels of adaptive recursive subdivision."),
    ("MaxPlotPoints", "Automatic, Infinity or a positive integer", [_sym("Automatic", "Infinity"), POSINT], None,
     "option of ArrayPlot, ListPlot3D and similar: maximum number of points kept."),
    ("PlotMarkers", "None, Automatic, a marker g, {g, s}, {g1, g2, ...} or a named marker set",
     [_sym("None", "Automatic"), _string(), _list(_anything), lambda x: isinstance(x, Expr) and not _is_number_like(x)],
     None, "option of list plots: markers for points, None, Automatic, {Automatic, s}, g, {g, s}, {g1, ...}, "
           "\"OpenMarkers\" ...; sizes s are Tiny..Large, Scaled[s] or Offset[d]."),
    ("ScalingFunctions", "None, a scale name (\"Log\", \"Log10\", \"Reverse\", ...), {f, finv} or a list per direction",
     [_SCALING_ONE, _list(_SCALING_ONE)], None,
     "option of plotting functions: coordinate scales; named scales \"Log\", \"Log2\", \"Log10\", \"SignedLog\", "
     "{\"SignedLog\", α}, \"Date\", \"Time\", \"Reverse\", \"Reciprocal\", \"Infinite\", NominalScale/OrdinalScale, "
     "or {f, inverse}; unknown names raise ValueError."),
    ("ReImLabels", "Automatic, \"Text\", \"Gothic\" or {re, im}", [AUTO, _string("Text", "Gothic"), _tuple(_anything, _anything)],
     None, "option of ReImPlot and similar: decoration of labels for real and imaginary parts."),
    ("ReImStyle", "Automatic, {re, im} or {{re1, im1}, ...}", [AUTO, _tuple(STYLE, STYLE), _list(_tuple(STYLE, STYLE))],
     None, "option of ReImPlot and similar: styles of the real and imaginary parts."),
    ("TernaryPlotCorners", "Automatic, \"Clockwise\" or three points {pt1, pt2, pt3}",
     [AUTO, _string("Clockwise", "Counterclockwise"), _list(_tuple(REAL, REAL), lens=(3,))], None,
     "option of TernaryListPlot: placement and orientation of the triangle corners."),
    # ── legends and labelling ──
    ("PlotLegends", "None, Automatic, \"Expressions\", a list of labels, a legend construct or Placed[leg, pos]",
     [_LEGENDS], None,
     "option of plotting functions: legends (SwatchLegend, LineLegend, PointLegend, BarLegend, Placed[...], "
     "label lists, any expression)."),
    ("ChartLegends", "None, Automatic, a list of labels, a legend construct or Placed[leg, pos]", [_LEGENDS], None,
     "option of charting functions: legend labels for chart elements."),
    ("ImageLegends", "None, Automatic, \"Expressions\", a list of labels or Placed[...]", [_LEGENDS], None,
     "option of image visualisation functions: legends for the inputs."),
    ("PlotLabels", "None, Automatic, \"Expressions\", {lbl1, ...} or Placed[labels, pos]", [_LABELS], None,
     "option of plotting functions: labels attached to each curve or dataset."),
    ("ImageLabels", "None, Automatic, \"Expressions\", {lbl1, ...} or Placed[labels, pos]", [_LABELS], None,
     "option of image visualisation functions: labels for each data source."),
    ("ChartLabels", "None, Automatic, {lbl1, ...} or Placed[labels, pos]",
     [_sym("None", "Automatic"), _list(_anything), _head("Placed")], None,
     "option of charting functions: labels of chart elements."),
    ("EventLabels", "None, Automatic, {event -> label, ...} or Placed[...]",
     [_sym("None", "Automatic"), _list(_anything), _rule(), _head("Placed")], None,
     "option of TimelinePlot and EventSeries: labels for events."),
    ("ResidueLabels", "None, Automatic, a list of residue properties or Placed[spec, Tooltip|StatusArea]",
     [_sym("None", "Automatic"),
      _list(_string("ChainNames", "ResidueAbbreviations", "ResidueIndices", "ResidueIDs", "AuthorChainNames",
                    "AuthorResidueIDs")), _head("Placed")], None,
     "option of biomolecule plots: residue properties shown as tooltips."),
    ("LabelingFunction", "None, Automatic, a position (Top, Bottom, ...), a function or Placed[...]",
     [_sym("None", "Automatic", "Top", "Bottom", "Left", "Right", "Center", "Above", "Below", "Before", "After",
           "Tooltip", "StatusArea"), _func], None,
     "option of charts and plots: how each element is labelled."),
    ("LabelingSize", "Automatic, All, Full, s > 0 or {w, h}", [_sym("Automatic", "All", "Full"), POS, _tuple(POS, POS)],
     None, "option of plots with Callout/Labeled: maximum label size in printer's points."),
    ("LabelingTarget", "Automatic, All, None, a fraction in [0, 1], \"Dense\", \"Sparse\" or an association",
     [_sym("Automatic", "All", "None"), UNIT, _string("Dense", "Sparse"), _head("Association")], None,
     "option of plots with callouts: how many labels are placed and how densely."),
    ("LabelVisibility", "All, None, Automatic, Infinity, -Infinity or a priority n",
     [_sym("All", "None", "Automatic", "Infinity"), REAL, lambda x: K_input(x) == "-Infinity"], None,
     "option of Callout and labels: priority of a label in automatic placement."),
    ("LeaderSize", "Automatic, a length, {lr, lθ, s} or {{lr, lθ, s}, {nr, nθ}}",
     [AUTO, NONNEG, _tuple(NONNEG, REAL, NONNEG), _tuple(_tuple(NONNEG, REAL, NONNEG), _tuple(NONNEG, REAL))], None,
     "option of Callout: leader and neck lengths/angles and gap to the anchor."),
    ("LegendAppearance", "Automatic or a named appearance such as \"Row\" or \"Column\"", [AUTO, _string()], None,
     "option of charting functions: appearance of generated legends."),
    ("LegendFunction", "Identity, None, Automatic, \"Frame\", \"Panel\" or a function",
     [_sym("None", "Automatic", "Identity"), _string(), _func], None,
     "option of the legend constructors: wrapper applied to the whole legend."),
    ("LegendLabel", "None or a label (Placed[label, pos] for a position)", [_any_of(NONE, LABEL)], None,
     "option of the legend constructors: overall legend label."),
    ("LegendLayout", "Automatic, \"Row\", \"Column\", \"ReversedRow\", \"ReversedColumn\", {\"Row\"|\"Column\", k} or a "
                     "function",
     [AUTO, _string(*LEGEND_LAYOUTS), _tuple(_string(*LEGEND_LAYOUTS), POSINT), _func], None,
     "option of the legend constructors: arrangement of legend entries."),
    ("LegendMarkerSize", "Automatic, s > 0, {w, h} or a list of sizes",
     [AUTO, POS, _tuple(POS, POS), _list(_any_of(POS, _tuple(POS, POS)))], None,
     "option of the legend constructors: size of legend markers in printer's points."),
    ("LegendMarkers", "None, Automatic, a marker g or {g1, g2, ...}",
     [_sym("None", "Automatic"), _list(_anything), _string(), lambda x: isinstance(x, Expr) and not _is_number_like(x)],
     None, "option of the legend constructors: markers shown for each entry."),
    # ── charts ──
    ("ChartStyle", "a directive, a list of directives, a colour scheme name or index, or a colour function",
     [STYLE, NONNEGINT, _string()], None,
     "option of charting functions: styles of chart elements (directives, ColorData index or scheme name)."),
    ("ChartBaseStyle", "a graphics directive", [STYLE], None,
     "option of charting functions: base style of all chart elements."),
    ("ChartElementFunction", "Automatic, a named element function or a function", [AUTO, _string(), _func], None,
     "option of charting functions: how chart elements are drawn (e.g. \"GlassRectangle\")."),
    ("ChartElements", "None, Automatic, a graphic or image, {g, All}, {g, {w, h}} or a list",
     [_sym("None", "Automatic"), _list(_anything), lambda x: isinstance(x, Expr) and not _is_number_like(x)], None,
     "option of charting functions: graphics replicated to fill chart elements."),
    ("ChartLayout", "Automatic, a layout name or {\"Column\"|\"Row\", k}",
     [AUTO, _string(), _tuple(_string(), _any_of(POSINT, _head("UpTo")))], None,
     "option of charting functions: \"Grouped\", \"Stacked\", \"Percentile\", \"Overlapped\", \"Column\", "
     "\"Row\", ..."),
    ("BarOrigin", "Bottom, Top, Left, Right, Front or Back", [_sym("Bottom", "Top", "Left", "Right", "Front", "Back")],
     None, "option of bar charts: from which edge bars grow."),
    ("BarSpacing", "Automatic, None, Tiny ... Large, s >= 0 or {sb, sg}",
     [_sym("Automatic", "None", *SIZES), NONNEG, _tuple(_any_of(NONNEG, _sym("Automatic", "None", *SIZES)),
                                                       _any_of(NONNEG, _sym("Automatic", "None", *SIZES)))], None,
     "option of bar charts: spacing between bars and groups."),
    ("BubbleScale", "\"Diameter\", \"Area\" or \"Volume\"", [AUTO, _string("Diameter", "Area", "Volume")], None,
     "option of bubble charts: which bubble measure is proportional to the data."),
    ("BubbleSizes", "Automatic, Tiny ... Large or {smin, smax}",
     [_sym("Automatic", *SIZES), _tuple(_any_of(POS, AUTO), _any_of(POS, AUTO))], _check_real_pairs("BubbleSizes"),
     "option of bubble charts: bubble size range as fractions of the chart diagonal."),
    ("SectorOrigin", "Automatic, Left, Right, Top, Bottom, θ, {pos, sense} or {{pos, sense}, r}",
     [_sym("Automatic", "Left", "Right", "Top", "Bottom"), REAL,
      _tuple(_any_of(_sym("Automatic", "Left", "Right", "Top", "Bottom"), REAL),
             _any_of(_string("Clockwise", "Counterclockwise"), lambda x: K.is_int(x) and int(x) in (-1, 1))),
      _tuple(_list(_anything, lens=(2,)), NONNEG)], None,
     "option of pie and sector charts: angle and direction of the first sector (and inner radius)."),
    ("SectorSpacing", "Automatic, None, Tiny ... Large, s >= 0 or {sp, sc}",
     [_sym("Automatic", "None", *SIZES), NONNEG, _tuple(_any_of(NONNEG, _sym("Automatic", "None", *SIZES)),
                                                       _any_of(NONNEG, _sym("Automatic", "None", *SIZES)))], None,
     "option of pie and sector charts: popout and concentric spacing."),
    ("ExtentElementFunction", "Automatic, a named function or a function", [AUTO, _string(), _func], None,
     "option of DiscretePlot: how extent elements are drawn."),
    ("ExtentMarkers", "None, \"Empty\", \"Filled\", \"Line\", \"Tube\", {ml, mr}, {specx, specy} or a graphic",
     [_sym("None", "Automatic"), _string("Empty", "Filled", "Line", "Tube"), _list(_anything),
      lambda x: isinstance(x, Expr) and not _is_number_like(x)], None,
     "option of DiscretePlot: markers at the edges of extents."),
    ("ExtentSize", "None, Full, All, Left, Right, s, Scaled[s], {sl, sr} or {specx, specy}",
     [_sym("None", "Full", "All", "Left", "Right"), NONNEG, SCALED,
      _list(_any_of(_sym("None", "Full", "All", "Left", "Right"), NONNEG, SCALED, _list(_any_of(NONNEG, SCALED))),
            lens=(2,))], None, "option of DiscretePlot: extents of points to their neighbours."),
    ("LayerSizeFunction", "a function", [_func], None,
     "option of TreePlot and layered graph layouts: relative height of each layer."),
    ("RulerUnits", "\"Inches\", \"DecimalInches\", \"Points\", \"Picas\", \"Centimeters\" or \"Millimeters\"",
     [_string("Inches", "DecimalInches", "Points", "Picas", "Centimeters", "Millimeters")], None,
     "option of notebooks: units of the ruler."),
    ("MaxOverlapFraction", "a fraction in [0, 1] or Automatic", [AUTO, UNIT], None,
     "option of object detection functions: maximum intersection-over-union between detections."),
    # ── gauges ──
    ("ScaleDivisions", "None, Automatic, n >= 1 or {nmajor, nminor}", [_sym("None", "Automatic"), POSINT,
                                                                        _tuple(POSINT, POSINT)], None,
     "option of gauges: number of major and minor scale divisions."),
    ("ScaleOrigin", "Automatic, r, Left, Right, Bottom, Top or a pair", [_sym("Automatic", "Left", "Right", "Bottom", "Top"),
                                                                       REAL, _tuple(_any_of(REAL, _sym("Left", "Right", "Bottom", "Top", "Automatic")),
                                                                                    _any_of(REAL, _sym("Automatic")))], None,
     "option of gauges: position (or angular range) of the scale."),
    ("ScalePadding", "Automatic, None, Full, All, s, Scaled[s] or lists of these",
     [_sym("Automatic", "None", "Full", "All"), REAL, SCALED,
      _list(_any_of(_sym("Automatic", "None"), REAL, SCALED, _list(_any_of(_sym("Automatic", "None"), REAL, SCALED))))],
     None, "option of gauges: background padding around the scale."),
    ("ScaleRanges", "None, All, {vmin, vmax}, {{vmin, vmax}, r} or a list of ranges (range -> style)",
     [_sym("None", "All"), _list(_anything, minlen=1)], None,
     "option of gauges: value ranges of the scale to emphasise."),
    ("ScaleRangeStyle", "None, Automatic, a style, {back, front}, n, a scheme name or a colour function",
     [STYLE, NONNEGINT, _string()], None, "option of gauges: style of the scale ranges."),
    # ── vector and stream plots ──
    ("VectorAspectRatio", "Automatic or a positive number", [AUTO, POS], None,
     "option of vector plots: width-to-length ratio of vector markers."),
    ("VectorColorFunction", "Automatic, None, a gradient name or a function", [_COLORFN], None,
     "option of vector plots: colour of each vector."),
    ("VectorColorFunctionScaling", "True, False or a list of True/False", [_is_bool, _list(_is_bool)], None,
     "option of vector plots: whether VectorColorFunction arguments are scaled to [0, 1]."),
    ("VectorMarkers", "None, Automatic, a marker name, a position, {m1, m2, ...} or Placed[marker, pos]",
     [_sym("None", "Automatic"), _string(), _list(_anything), _head("Placed", "Graphics", "Graphics3D")], None,
     "option of vector plots: shape of vectors (\"Arrow\", \"Dart\", \"Segment\", ... or \"Start\"/\"Middle\"/"
     "\"End\")."),
    ("VectorPoints", "None, Automatic, Coarse, Fine, n, {m, n(, k)}, a named pattern, {\"name\", n} or a list of points",
     [_sym("None", "Automatic", "Coarse", "Fine", "All"), POSINT, _list(POSINT, lens=(2, 3)),
      _string("Regular", "Hexagonal", "Mesh", "FaceCenteredCubic"),
      _tuple(_string("Regular", "Hexagonal", "Mesh", "FaceCenteredCubic"), POSINT),
      _list(_any_of(_list(REAL, lens=(2, 3)), _tuple(_list(REAL, lens=(2, 3)), STYLE)), minlen=1)], None,
     "option of vector plots: where vectors are drawn."),
    ("VectorRange", "Automatic, All or {min, max} magnitudes", [_sym("Automatic", "All"), _tuple(NONNEG, NONNEG)],
     _check_real_pairs("VectorRange"), "option of vector plots: range of vector magnitudes shown."),
    ("VectorScaling", "Automatic, None, \"Linear\", \"Log\", \"Sqrt\" or a function",
     [AUTO_NONE, _string("Linear", "Log", "Sqrt"), _func], None,
     "option of vector plots: how magnitudes map to arrow lengths."),
    ("VectorSizes", "Automatic, Tiny ... Large, s > 0 or {smin, smax}",
     [_sym("Automatic", *SIZES), POS, _tuple(_any_of(POS, AUTO), _any_of(POS, AUTO))], _check_real_pairs("VectorSizes"),
     "option of vector plots: arrow sizes as fractions of the bounding circle."),
    ("VectorStyle", "a graphics directive or a list of directives", [STYLE], None,
     "option of vector plots: style of vectors."),
    ("StreamColorFunction", "Automatic, None, a gradient name or a function", [_COLORFN], None,
     "option of stream plots: colour along streamlines."),
    ("StreamColorFunctionScaling", "True, False or a list of True/False", [_is_bool, _list(_is_bool)], None,
     "option of stream plots: whether StreamColorFunction arguments are scaled to [0, 1]."),
    ("StreamMarkers", "None, Automatic, a marker name or a list of markers",
     [_sym("None", "Automatic"), _string(), _list(_anything), _head("Graphics")], None,
     "option of stream plots: marker shapes (\"Arrow\", \"Dart\", \"Segment\", ...)."),
    ("StreamPoints", "None, Automatic, Coarse, Fine, n, a list of points, {spec, d} or {spec, d, len}",
     [_sym("None", "Automatic", "Coarse", "Fine"), POSINT,
      _list(_anything, minlen=1)], None,
     "option of stream plots: number or seed points of streamlines."),
    ("StreamScale", "Automatic, None, Full, a segment length or {seg, npts, aratio, sfun}",
     [_sym("Automatic", "None", "Full", *SIZES), POS, _list(_anything, lens=(1, 2, 3, 4))], None,
     "option of stream plots: how streamlines are segmented."),
    ("StreamStyle", "a graphics directive or a list of directives", [STYLE], None,
     "option of stream plots: style of streamlines."),
    ("LineIntegralConvolutionScale", "Automatic, Tiny ... Large, s > 0 or Scaled[s]",
     [_sym("Automatic", *SIZES), POS, SCALED], None,
     "option of line-integral-convolution plots: convolution length."),
]

_CHECK_NOTE = (" Invalid settings raise ValueError; settings containing user symbols where a value is needed stay "
               "unevaluated. Inert as an expression (Name and Name -> value evaluate to themselves). "
               "Check: reference (Wolfram documentation settings) in tests.")

OPTION_VALIDATORS = G.OPTION_VALIDATORS
for _name, _msg, _preds, _chk, _note in _SPECS:
    OPTION_VALIDATORS[_name] = make_validator(_name, _msg, *_preds, check=_chk)
    inert_option(_name, _note + _CHECK_NOTE, area=AREA)


# ───────────────────────────── system variables, Axis ─────────────────────────────

_SYSTEM_VALUES = {
    "$PlotTheme": (Symbol("Automatic"), "the default PlotTheme of visualisation functions; evaluates to Automatic "
                                        "(an assignment or Block by the user takes precedence)."),
    "$PerformanceGoal": ("Quality", "the default setting of PerformanceGoal; evaluates to \"Quality\" (Wolfram's "
                                    "value outside interactive Manipulate controls, where it is \"Speed\")."),
    "$DisplayFunction": (Symbol("Identity"), "the default DisplayFunction of graphics; evaluates to Identity "
                                             "(graphics are returned as expressions)."),
    "$ImageResolution": (Symbol("Automatic"), "the default ImageResolution for rasterizing; evaluates to Automatic, "
                                              "which Rasterize and Export treat as 72 dpi."),
}
for _n, (_v, _d) in _SYSTEM_VALUES.items():
    SYMBOL_VALUES[_n] = _v
    record(_n, AREA, "added", f"{_n} is {_d} Check: reference (Wolfram documentation) in tests.")

@fn("Axis", area=AREA, status="added",
    notes="Axis is an inert symbol used as a setting (Filling -> Axis, alignment and placement positions); "
          "Axis[...] also stays unevaluated. Check: reference (Wolfram documentation forms) in tests.")
def _axis(*args: Expr) -> Expr:
    raise Unevaluated()


# ───────────────────────────── legends ─────────────────────────────

_LEGEND_OPTION_NAMES = ("LabelStyle", "LegendFunction", "LegendLabel", "LegendLayout", "LegendMargins",
                        "LegendMarkers", "LegendMarkerSize")
_LEGEND_OPTIONS = {
    "SwatchLegend": _LEGEND_OPTION_NAMES,
    "LineLegend": _LEGEND_OPTION_NAMES + ("Joined",),
    "PointLegend": _LEGEND_OPTION_NAMES + ("Joined",),
    "BarLegend": _LEGEND_OPTION_NAMES,
}


def _check_named_options(opts, allowed, fname):
    for r in opts:
        n = opt_name(r)
        if n not in allowed:
            raise ValueError(f"{fname}: {n} is not an option of {fname} (allowed: {', '.join(allowed)})")
        if r.head == "Rule":
            OPTION_VALIDATORS[n](r.args[1], fname)


def _legend_styles(x, fname):
    """Styles argument of Swatch/Line/PointLegend: a list of directives, Automatic, a colour scheme name or index,
    or a colour function."""
    if K.is_list(x):
        for i, c in enumerate(x.args):
            if not (STYLE(c) or is_sym(c, "Automatic", "None")):
                if _user_symbols(c):
                    raise Unevaluated()
                raise TypeError(f"{fname}: entry {i + 1} of the styles, {K_input(c)}, is not a colour or graphics "
                                f"directive")
        return len(x.args)
    if is_sym(x, "Automatic") or isinstance(x, str) or NONNEGINT(x):
        return None
    if _func(x) and parse_color(x) is None:
        if isinstance(x, Symbol) and _user_symbols(x):
            raise Unevaluated()
        return None
    raise TypeError(f"{fname}: the first argument must be a list of colours or directives, Automatic, a colour "
                    f"scheme or a colour function, not {K_input(x)}")


def _make_legend(head):
    def impl(*args: ListOf[Expr] | Expr | Option) -> Expr:
        pos, opts = split_trailing_options(args)
        if not 1 <= len(pos) <= 2:
            raise ValueError(f"{head}: expected {head}[{{lbl1, ...}}] or {head}[{{col1, ...}}, {{lbl1, ...}}], "
                             f"got {len(pos)} positional argument(s)")
        _check_named_options(opts, _LEGEND_OPTIONS[head], head)
        if len(pos) == 1:
            if not K.is_list(pos[0]):
                if isinstance(pos[0], Symbol) and _user_symbols(pos[0]):
                    raise Unevaluated()
                raise TypeError(f"{head}: the labels must be a list {{lbl1, ...}}, not {K_input(pos[0])}")
            raise Unevaluated()
        n = _legend_styles(pos[0], head)
        lbls = pos[1]
        if K.is_list(lbls):
            if n is not None and n != len(lbls.args):
                raise ValueError(f"{head}: {n} styles but {len(lbls.args)} labels; give one label per style")
        elif not is_sym(lbls, "Automatic"):
            if isinstance(lbls, Symbol) and _user_symbols(lbls):
                raise Unevaluated()
            raise TypeError(f"{head}: the labels must be a list or Automatic, not {K_input(lbls)}")
        raise Unevaluated()
    impl.__name__ = f"_{head.lower()}"
    return impl


_LEGEND_DOC = {
    "SwatchLegend": "swatches of colours",
    "LineLegend": "lines (and markers) of the given styles",
    "PointLegend": "points of the given colours",
}
for _h, _what in _LEGEND_DOC.items():
    fn(_h, area=AREA, status="added",
       notes=f"{_h}[{{col1, ...}}, {{lbl1, ...}}] legend associating {_what} with labels; {_h}[{{col1, ...}}, "
             f"Automatic] placeholder labels; {_h}[{{lbl1, ...}}] labels with styles inherited from the "
             f"visualisation; the styles may also be Automatic, a colour scheme name/index or a colour function. "
             f"Options {', '.join(_LEGEND_OPTIONS[_h])} are validated; unknown options raise ValueError, styles "
             f"that are not directives raise TypeError, and a styles list whose length differs from the labels "
             f"raises ValueError (Wolfram pairs them silently). Inert, displayed by the front end / plotting batches. "
             f"Check: reference (Wolfram documentation forms) in tests.")(_make_legend(_h))


def _barlegend_cf(x, fname):
    """cf or {cf, {min, max}}."""
    if K.is_list(x):
        if len(x.args) != 2:
            raise ValueError(f"{fname}: expected {{cf, {{min, max}}}}, not {K_input(x)}")
        _barlegend_cf(x.args[0], fname)
        r = x.args[1]
        if is_sym(r, "Automatic", "All"):
            return
        if not (K.is_list(r) and len(r.args) == 2):
            raise ValueError(f"{fname}: the range must be {{min, max}}, not {K_input(r)}")
        a, b = (real_value(v, fname, "range limit") for v in r.args)
        if not a < b:
            raise ValueError(f"{fname}: the range {{min, max}} needs min < max")
        return
    if is_sym(x, "Automatic") or isinstance(x, str):
        return
    if _is_bool(x) or _is_number_like(x):
        raise TypeError(f"{fname}: the colour function must be a function, a gradient name or Automatic, not "
                        f"{K_input(x)}")
    if isinstance(x, Symbol) and _user_symbols(x):
        raise Unevaluated()


@fn("BarLegend", area=AREA, status="added",
    notes="BarLegend[cf], BarLegend[{cf, {min, max}}], BarLegend[cf, contours] and BarLegend[{cf, {min, max}}, "
          "contours]: colour-bar legend for colour function cf (a function, a gradient name such as \"Rainbow\" or "
          "Automatic); contours n, Automatic, All, None, {Automatic, n}, {f1, ...}, {{f1, g1}, ...} or a function. "
          "min >= max raises ValueError, a number or boolean as cf TypeError, invalid contours and unknown options "
          "ValueError. Inert, displayed by the front end / plotting batches. "
          "Check: reference (Wolfram documentation forms) in tests.")
def _barlegend(cf: Expr, *rest: Expr | Option) -> Expr:
    pos, opts = split_trailing_options(rest)
    G.nargs(pos, 0, 1, "BarLegend")
    _check_named_options(opts, _LEGEND_OPTIONS["BarLegend"], "BarLegend")
    _barlegend_cf(cf, "BarLegend")
    if pos:
        c = pos[0]
        if not (_contours(c) or is_sym(c, "All")):
            if _user_symbols(c):
                raise Unevaluated()
            raise ValueError(f"BarLegend: contours must be n, Automatic, All, None, {{Automatic, n}}, a list of "
                             f"levels or a function, not {K_input(c)}")
    raise Unevaluated()


# ───────────────────────────── AxisObject ─────────────────────────────

_AXIS_OPTIONS = ("AxisLabel", "AxisStyle", "BaseStyle", "LabelStyle", "RotateLabel", "TickDirection",
                 "TickLabelOrientation", "TickLabelPositioning", "TickLabels", "TickLengths", "TickPositions",
                 "Ticks", "TicksStyle")
_PATH_HEADS = ("Line", "InfiniteLine", "HalfLine", "BSplineCurve", "BezierCurve", "Circle")


def _axis_path(p, fname):
    if K.is_list(p) and p.args and isinstance(p.args[0], str):
        kind = p.args[0]
        if kind not in ("Horizontal", "Vertical"):
            raise ValueError(f"{fname}: the path kind must be \"Horizontal\" or \"Vertical\", not {kind!r}")
        if len(p.args) not in (2, 3):
            raise ValueError(f"{fname}: expected {{\"{kind}\", c}} or {{\"{kind}\", c, {{min, max}}}}")
        real_value(p.args[1], fname, "axis position")
        if len(p.args) == 3:
            r = p.args[2]
            if not (K.is_list(r) and len(r.args) == 2):
                raise ValueError(f"{fname}: the extent must be {{min, max}}, not {K_input(r)}")
            a, b = (real_value(v, fname, "axis extent") for v in r.args)
            if a == b:
                raise ValueError(f"{fname}: the axis extent {{min, max}} must have min != max")
        return
    if K.is_list(p):
        if len(p.args) != 2:
            raise ValueError(f"{fname}: the path {{pt1, pt2}} needs exactly two points")
        for q in p.args:
            G.coords(q, fname, "axis point", dims=(2,), scaled_ok=False)
        if all(K.is_list(q) for q in p.args) and \
                [real_value(c, fname, "axis point") for c in p.args[0].args] == \
                [real_value(c, fname, "axis point") for c in p.args[1].args]:
            raise ValueError(f"{fname}: the two points of the axis path must differ")
        return
    if isinstance(p, Expr) and p.head in _PATH_HEADS:
        return
    if isinstance(p, Symbol) and _user_symbols(p):
        raise Unevaluated()
    raise TypeError(f"{fname}: the axis path must be {{pt1, pt2}}, {{\"Horizontal\", y}}, {{\"Vertical\", x}}, or a "
                    f"Line, InfiniteLine, HalfLine, BezierCurve, BSplineCurve or Circle, not {K_input(p)}")


def _axis_scale(s, fname):
    if is_sym(s, "Automatic"):
        return
    if K.is_list(s) and len(s.args) == 2:
        if all(is_rule(r) for r in s.args):
            vals = [real_value(r.args[1], fname, "scale value") for r in s.args]
        else:
            vals = [real_value(v, fname, "scale value") for v in s.args]
        if vals[0] == vals[1]:
            raise ValueError(f"{fname}: the scale values must differ")
        return
    if isinstance(s, Symbol) and _user_symbols(s):
        raise Unevaluated()
    raise ValueError(f"{fname}: the scale must be {{val1, val2}} or {{loc1 -> val1, loc2 -> val2}}, not {K_input(s)}")


def _validate_axisobject(e, dim=2):
    pos, opts = split_trailing_options(e.args)
    G.nargs(pos, 1, 2, "AxisObject")
    _axis_path(pos[0], "AxisObject")
    if len(pos) == 2:
        _axis_scale(pos[1], "AxisObject")
    _check_named_options(opts, _AXIS_OPTIONS, "AxisObject")


G.register_primitive("AxisObject", dims=(2,), validate=_validate_axisobject)


@fn("AxisObject", area=AREA, status="added",
    notes="AxisObject[path] and AxisObject[path, scale]: graphics primitive for an axis along path ({pt1, pt2}, "
          "{\"Horizontal\", y0(, {xmin, xmax})}, {\"Vertical\", x0(, {ymin, ymax})}, Line, InfiniteLine, HalfLine, "
          "BezierCurve, BSplineCurve, Circle) with scale {val1, val2} or {loc1 -> val1, loc2 -> val2}; options "
          "AxisLabel, AxisStyle, BaseStyle, LabelStyle, RotateLabel, TickDirection, TickLabelOrientation, "
          "TickLabelPositioning, TickLabels, TickLengths, TickPositions, Ticks, TicksStyle are validated (unknown "
          "options and invalid settings raise ValueError, a malformed path TypeError/ValueError). Accepted inside "
          "Graphics; drawing it with Rasterize is not supported yet (NotImplementedError). "
          "Check: reference (Wolfram documentation forms) in tests.")
def _axisobject(path: Expr, *rest: Expr | Option) -> Expr:
    _validate_axisobject(Expr("AxisObject", path, *rest))
    raise Unevaluated()


# ───────────────────────────── FindDivisions ─────────────────────────────

def _exact_real(x, fname, what):
    """Exact SymPy value of a finite real; floats are read as the decimal they print as (0.1 -> 1/10)."""
    if isinstance(x, bool) or isinstance(x, str) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be a real number, not {K_input(x)}")
    v = real_value(x, fname, what)
    if not math.isfinite(v):
        raise ValueError(f"{fname}: {what} must be finite, not {K_input(x)}")
    if isinstance(x, float):
        return sp.Rational(repr(x))
    e = K.to_sp(x)
    if e.has(sp.Float):
        e = e.xreplace({f: sp.Rational(repr(float(f))) for f in e.atoms(sp.Float)})
    return sp.nsimplify(e) if not e.is_Rational and e.is_number and not e.free_symbols and e.has(sp.Float) else e


def _mantissas(base):
    return [m for m in range(1, base) if base % m == 0]


def nice_step(lo, hi, n, base=10, dx=None):
    """(step, first index, last index) of the nice division of [lo, hi] (exact SymPy numbers) into about n parts:
    steps m * base^e (m a proper divisor of base) or, with dx, (m * base^e) * dx for e >= 0; the parts count
    ceil(hi/s) - floor(lo/s) closest to n wins (ties: the coarser division)."""
    width = float(hi - lo)
    raw = width / n
    cands = []
    if dx is None:
        e0 = int(math.floor(math.log(raw, base)))
        for e in range(e0 - 1, e0 + 2):
            for m in _mantissas(base):
                cands.append(sp.Integer(m) * sp.Integer(base) ** e)
    else:
        r = raw / float(dx)
        e0 = max(0, int(math.floor(math.log(r, base))) if r >= 1 else 0)
        for e in range(0, e0 + 2):
            for m in _mantissas(base):
                cands.append(sp.Integer(m) * sp.Integer(base) ** e * dx)
    best = None
    for s in cands:
        a = sp.floor(sp.nsimplify(lo / s) if not (lo / s).is_Rational else lo / s)
        b = sp.ceiling(sp.nsimplify(hi / s) if not (hi / s).is_Rational else hi / s)
        parts = int(b - a)
        if parts < 1:
            continue
        key = (abs(parts - n), parts)
        if best is None or key < best[0]:
            best = (key, s, int(a), int(b))
    return best[1], best[2], best[3]


def _divide(lo, hi, n, base, dx):
    s, a, b = nice_step(lo, hi, n, base, dx)
    pts = [sp.simplify(k * s) if not s.is_Rational else k * s for k in range(a, b + 1)]
    # Check (self-check): equally spaced, covering [lo, hi], steps multiples of dx
    if not (sp.simplify(pts[0] - lo) <= 0 and sp.simplify(pts[-1] - hi) >= 0):
        raise ArithmeticError("FindDivisions: result failed verification (does not cover the interval)")
    if dx is not None and not sp.simplify(s / dx).is_Integer:
        raise ArithmeticError("FindDivisions: result failed verification (step is not a multiple of dx)")
    return pts


def _pos_int(x, fname, what):
    if isinstance(x, bool) or isinstance(x, str) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be a positive integer, not {K_input(x)}")
    if K.is_int(x):
        if int(x) < 1:
            raise ValueError(f"{fname}: {what} must be a positive integer, not {x}")
        return int(x)
    if _num(x) is not None:
        raise TypeError(f"{fname}: {what} must be a positive integer, not {K_input(x)}")
    raise Unevaluated()


@fn("FindDivisions", area=AREA, status="added",
    notes="FindDivisions[{xmin, xmax}, n] about n+1 equally spaced \"nice\" numbers (steps 1, 2 or 5 times a power "
          "of 10) whose first and last elements enclose [xmin, xmax] (first <= xmin, last >= xmax); "
          "FindDivisions[{xmin, xmax, dx}, n] steps that are nice integer multiples of dx (dx may be exact, e.g. "
          "Pi/2); FindDivisions[{xmin, xmax}, {n1, n2, ...}] successive levels, level k subdividing each interval "
          "of level k - 1 (as nested lists, as in the Wolfram documentation example); "
          "FindDivisions[{xmin, xmax, {dx1, dx2, ...}}, {n1, n2, ...}] per-level dx; FindDivisions[{xmin, xmax}, n, "
          "k] nice numbers in base k (steps d k^e with d a proper divisor of k). Results are exact (machine inputs "
          "are read as the decimals they print as). Among candidate steps the part count closest to n wins (ties: "
          "the coarser one). Symbolic limits stay unevaluated; xmin >= xmax raises NotImplementedError; n < 1, "
          "dx <= 0, base < 2 or mismatched dx/n lists ValueError; non-integer n TypeError. "
          "Check: self-check (equal spacing covering the interval, steps multiples of dx) and reference values "
          "(Wolfram documentation examples) in tests.")
def _find_divisions(spec: ListOf[Real], n: PosInteger | ListOf[PosInteger], *rest: PosInteger) -> ListOf[Rational]:
    fname = "FindDivisions"
    G.nargs(rest, 0, 1, fname)
    if not K.is_list(spec):
        if isinstance(spec, Symbol) and _user_symbols(spec):
            raise Unevaluated()
        raise TypeError(f"{fname}: the first argument must be {{xmin, xmax}} or {{xmin, xmax, dx}}, not "
                        f"{K_input(spec)}")
    if len(spec.args) not in (2, 3):
        raise ValueError(f"{fname}: the first argument must be {{xmin, xmax}} or {{xmin, xmax, dx}}")
    lo = _exact_real(spec.args[0], fname, "xmin")
    hi = _exact_real(spec.args[1], fname, "xmax")
    if not bool(lo < hi):
        raise NotImplementedError(f"{fname}: only intervals with xmin < xmax are supported")
    levels = [_pos_int(v, fname, "the number of divisions") for v in n.args] if K.is_list(n) else \
        [_pos_int(n, fname, "the number of divisions")]
    if K.is_list(n) and not levels:
        raise ValueError(f"{fname}: the list of division counts must not be empty")
    base = 10
    if rest:
        base = _pos_int(rest[0], fname, "the base")
        if base < 2:
            raise ValueError(f"{fname}: the base must be an integer >= 2")
    dxs = [None] * len(levels)
    if len(spec.args) == 3:
        d = spec.args[2]
        ds = list(d.args) if K.is_list(d) else [d] * len(levels)
        if K.is_list(d) and len(ds) != len(levels):
            raise ValueError(f"{fname}: {len(ds)} spacings dx for {len(levels)} levels")
        dxs = []
        for v in ds:
            e = _exact_real(v, fname, "dx")
            if not bool(e > 0):
                raise ValueError(f"{fname}: dx must be positive, not {K_input(v)}")
            dxs.append(e)
    first = _divide(lo, hi, levels[0], base, dxs[0])
    out = [first]
    prev = first
    for k in range(1, len(levels)):
        def sub(node, k=k):
            if node and isinstance(node[0], list):
                return [sub(c) for c in node]
            return [_divide(a, b, levels[k], base, dxs[k]) for a, b in zip(node, node[1:])]
        prev = sub(prev)
        out.append(prev)

    def conv(node):
        if isinstance(node, list):
            return L(*[conv(c) for c in node])
        return K.from_sp(node)
    return conv(out) if K.is_list(n) else conv(first)


def tick_positions(lo, hi, n=5, base=10):
    """Python API: float positions of nice ticks inside [lo, hi] (about n divisions), via FindDivisions' rule."""
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        return []
    a, b = sp.Rational(repr(float(lo))), sp.Rational(repr(float(hi)))
    pts = _divide(a, b, n, base, None)
    eps = 1e-9 * (hi - lo)
    return [float(p) for p in pts if lo - eps <= float(p) <= hi + eps]


# ───────────────────────────── option resolution (Python API) ─────────────────────────────

def resolve_options(fname, opts, defaults):
    """Merge option rules ``opts`` over ``defaults`` ({name: value}); every name must be in ``defaults`` (else
    ValueError, as for Graphics), each given value is validated with its OPTION_VALIDATORS entry, and the first
    setting of a name wins (Wolfram's OptionValue)."""
    out = dict(defaults)
    seen = set()
    for r in opts:
        name = opt_name(r)
        if name is None:
            raise ValueError(f"{fname}: {K_input(r)} is not an option rule Name -> value")
        if name not in defaults:
            raise ValueError(f"{fname}: {name} is not a known option of {fname}")
        if name in seen:
            continue
        seen.add(name)
        v = OPTION_VALIDATORS.get(name)
        if v is not None and r.head == "Rule":
            v(r.args[1], fname)
        out[name] = r.args[1]
    return out


def scaling_function(spec):
    """(f, inverse) Python callables (floats or numpy arrays) for a ScalingFunctions entry: None, \"Log\",
    \"Log2\", \"Log10\", \"Reverse\", \"Reciprocal\". Other documented scales (\"SignedLog\", \"Date\", \"Time\",
    \"Infinite\", NominalScale, OrdinalScale, {f, finv}) raise NotImplementedError until a plotting batch needs
    them."""
    import numpy as np
    if spec is None or is_sym(spec, "None", "Automatic"):
        return (lambda x: x), (lambda y: y)
    table = {
        "Log": (np.log, np.exp),
        "Log10": (np.log10, lambda y: np.power(10.0, y)),
        "Log2": (np.log2, lambda y: np.power(2.0, y)),
        "Reverse": (lambda x: -np.asarray(x, dtype=float), lambda y: -np.asarray(y, dtype=float)),
        "Reciprocal": (lambda x: 1.0 / np.asarray(x, dtype=float), lambda y: 1.0 / np.asarray(y, dtype=float)),
    }
    if isinstance(spec, str) and spec in table:
        return table[spec]
    OPTION_VALIDATORS["ScalingFunctions"](spec, "ScalingFunctions")
    raise NotImplementedError(f"ScalingFunctions: the scale {K_input(spec)} is not supported yet")
