"""Wolfram batch W14 (part 2 of 3): graphics containers, directives, options, transforms and layout.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

Graphics objects are inert symbolic expressions, as in Wolfram: ``Graphics[prims, opts]`` evaluates to itself
after its contents and options have been validated (a malformed directive or option value raises instead of being
reported only when drawn). Drawing happens in ``wl_graphics_render`` (``Rasterize``; matplotlib, optional).

Extension points for later batches
----------------------------------
* ``register_primitive(head, dims=(2, 3), validate=None)``: a graphics primitive (W19 registers ``Point``,
  ``Line``, ``Polygon``, ``Disk``, ``Sphere``, ...). ``dims`` says whether it may appear in ``Graphics`` (2) or
  ``Graphics3D`` (3); ``validate(expr, dim)`` may raise for malformed input inside a container.
* ``register_graphics_option(name, validator=None, dims=(2, 3))``: a container option (W15 adds validators for
  ``PlotRange``, ``Axes``, ``Ticks``, ...; the standard Graphics/Graphics3D option names are pre-registered with
  no validator). ``validator(value, fname)`` raises ``ValueError`` / ``TypeError`` for a bad setting.
* ``OPTION_VALIDATORS[name]``: the validator of an option symbol owned by this batch (``ViewPoint``,
  ``BoxRatios``, ``Lighting``, ``VertexColors``, ...), usable by any function that accepts that option.
* ``DIRECTIVE_HEADS``: heads that are graphics directives (colours, ``Thickness``, ``EdgeForm``, ...).
* ``graphics_parts(g)`` -> ``(dim, content, options)``; ``options_dict(opts)``; ``is_graphics(x)``.
* ``normal_graphics(expr)``: the ``Normal`` expansion of ``GraphicsComplex`` / ``Translate`` / ``Rotate`` /
  ``Scale`` (also reachable as ``Normal[...]`` in the language).
"""
from __future__ import annotations

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from ..symbol_values import SYMBOL_VALUES
from .core import fn, record, Unevaluated, L, R
from .wl_graphics_color import (real_value, is_real_value, parse_color, K_input, inert_option, distance_function,
                                SPACES)

AREA = "graphics: containers & directives (W14)"

SIZE_SYMBOLS = ("Tiny", "Small", "Medium", "Large")


# ───────────────────────────── small helpers ─────────────────────────────

def is_sym(x, *names):
    return isinstance(x, Symbol) and (not names or x.name in names)


def is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def opt_name(x):
    """Name of an option rule ``Name -> v`` (Symbol or string lhs), else None."""
    if is_rule(x):
        lhs = x.args[0]
        if isinstance(lhs, Symbol):
            return lhs.name
        if isinstance(lhs, str):
            return lhs
    return None


def split_trailing_options(args):
    args = list(args)
    opts = []
    while args and opt_name(args[-1]) is not None:
        opts.insert(0, args.pop())
    return args, opts


def options_dict(opts):
    """{name: value} of a list of option rules; the first setting of a name wins (Wolfram's OptionValue)."""
    d = {}
    for r in opts:
        n = opt_name(r)
        if n is not None and n not in d:
            d[n] = r.args[1]
    return d


def _symbolic(x):
    return isinstance(x, Symbol) or (isinstance(x, Expr) and not K.is_list(x) and x.head not in
                                     ("Rule", "RuleDelayed", "Association") and not is_real_value(x)
                                     and parse_color(x) is None)


def _bad_kind(x):
    return isinstance(x, (str, bool)) or (isinstance(x, (int, float)))


def real_arg(x, fname, what, lo=None, strict=False):
    v = real_value(x, fname, what)
    if lo is not None and (v < lo or (strict and v == lo)):
        raise ValueError(f"{fname}: {what} must be {'>' if strict else '>='} {lo}, not {K_input(x)}")
    return v


def coords(x, fname, what="coordinate", dims=(2, 3), scaled_ok=True):
    """Validate a point {x, y(, z)} (or Scaled/ImageScaled/Offset form); returns its dimension.
    Symbolic coordinates -> Unevaluated."""
    if scaled_ok and isinstance(x, Expr) and x.head in ("Scaled", "ImageScaled", "Offset"):
        return None
    if not K.is_list(x):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not _bad_kind(x)):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a list of numbers {{x, y}}, not {K_input(x)}")
    if len(x.args) not in dims:
        raise ValueError(f"{fname}: {what} {K_input(x)} must have {' or '.join(map(str, dims))} components")
    for c in x.args:
        real_value(c, fname, what)
    return len(x.args)


def size_value(x, fname, what, allow=SIZE_SYMBOLS, lo=0):
    """A non-negative size: a real or one of Tiny/Small/Medium/Large (Scaled[s] / Offset[..] also accepted)."""
    if isinstance(x, Symbol):
        if x.name in allow:
            return
        raise Unevaluated()
    if isinstance(x, Expr) and x.head in ("Scaled", "Offset", "ImageScaled"):
        return
    real_arg(x, fname, what, lo)


def directive_like(x, fname, what="directive"):
    """Something usable as a graphics directive (a colour, a directive head, a list of them, a style rule)."""
    if K.is_list(x):
        for i in x.args:
            directive_like(i, fname, what)
        return
    if x is None or is_sym(x, "None", "Automatic", "Inherited"):
        return
    if isinstance(x, (str, bool, int, float)) or (is_real_value(x) and not isinstance(x, Symbol)):
        raise TypeError(f"{fname}: {K_input(x)} is not a graphics directive")


def nargs(args, lo, hi, fname):
    if not lo <= len(args) <= hi:
        want = str(lo) if lo == hi else f"{lo} to {hi}"
        raise ValueError(f"{fname}: expected {want} argument(s), got {len(args)}")


def color_arg(x, fname, what="colour", none_ok=False):
    if none_ok and is_sym(x, "None"):
        return
    if parse_color(x) is not None:
        return
    if isinstance(x, (str, bool, int, float)) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be a colour, not {K_input(x)}")
    if isinstance(x, Expr) and x.head in SPACES:
        raise Unevaluated()
    raise Unevaluated()


def _string_choice(x, choices, fname, what):
    if isinstance(x, str):
        if x in choices:
            return
        raise ValueError(f"{fname}: {what} must be one of {', '.join(repr(c) for c in choices)}, not {x!r}")
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be a string such as {choices[0]!r}, not {K_input(x)}")


# ───────────────────────────── registries ─────────────────────────────

PRIMITIVES = {}          # head -> {"dims": set, "validate": f or None}
DIRECTIVE_HEADS = set(SPACES) | {"Opacity"}
GRAPHICS_OPTIONS = {2: {}, 3: {}}     # dim -> {name: validator or None}
OPTION_VALIDATORS = {}


def register_primitive(head, dims=(2, 3), validate=None):
    PRIMITIVES[head] = {"dims": set(dims), "validate": validate}


def register_graphics_option(name, validator=None, dims=(2, 3)):
    for d in dims:
        GRAPHICS_OPTIONS[d][name] = validator


_STD_2D = ("AlignmentPoint", "AspectRatio", "Axes", "AxesLabel", "AxesOrigin", "AxesStyle", "Background",
           "BaselinePosition", "BaseStyle", "ContentSelectable", "CoordinatesToolOptions", "DisplayFunction",
           "Epilog", "FormatType", "Frame", "FrameLabel", "FrameStyle", "FrameTicks", "FrameTicksStyle", "GridLines",
           "GridLinesStyle", "ImageMargins", "ImagePadding", "ImageSize", "ImageSizeRaw", "LabelStyle", "Method",
           "PlotLabel", "PlotRange", "PlotRangeClipping", "PlotRangePadding", "PlotRegion", "PreserveImageOptions",
           "Prolog", "RotateLabel", "Ticks", "TicksStyle", "LightDark", "Antialiasing")
_STD_3D = ("AlignmentPoint", "AspectRatio", "AutomaticImageSize", "Axes", "AxesEdge", "AxesLabel", "AxesOrigin",
           "AxesStyle", "Background", "BaselinePosition", "BaseStyle", "Boxed", "BoxRatios", "BoxStyle",
           "ClipPlanes", "ClipPlanesStyle", "ColorOutput", "ContentSelectable", "ControllerLinking",
           "ControllerMethod", "ControllerPath", "CoordinatesToolOptions", "DisplayFunction", "Epilog", "FaceGrids",
           "FaceGridsStyle", "FormatType", "ImageMargins", "ImagePadding", "ImageSize", "ImageSizeRaw", "LabelStyle",
           "Lighting", "Method", "PlotLabel", "PlotRange", "PlotRangePadding", "PlotRegion", "PreserveImageOptions",
           "Prolog", "RotationAction", "SphericalRegion", "Ticks", "TicksStyle", "TouchscreenAutoZoom", "ViewAngle",
           "ViewCenter", "ViewMatrix", "ViewPoint", "ViewProjection", "ViewRange", "ViewVector", "ViewVertical",
           "RenderingOptions", "LightDark", "Antialiasing")
for _o in _STD_2D:
    register_graphics_option(_o, None, (2,))
for _o in _STD_3D:
    register_graphics_option(_o, None, (3,))


def check_options(opts, dim, fname, table=None):
    """Validate option rules for a container: known names, and each value with its validator."""
    known = GRAPHICS_OPTIONS[dim] if table is None else table
    for r in opts:
        n = opt_name(r)
        if n is None:
            raise ValueError(f"{fname}: {K_input(r)} is not an option rule Name -> value")
        if n not in known:
            raise ValueError(f"{fname}: {n} is not a known option of {fname}")
        v = known.get(n) or OPTION_VALIDATORS.get(n)
        if v is not None and r.head == "Rule":
            v(r.args[1], fname)


def is_graphics(x):
    return isinstance(x, Expr) and x.head in ("Graphics", "Graphics3D")


def graphics_parts(g):
    """(dim, content, options) of Graphics[content, opts] / Graphics3D[...]."""
    dim = 3 if g.head == "Graphics3D" else 2
    content = g.args[0] if g.args else L()
    return dim, content, list(g.args[1:])


# ───────────────────────────── content validation ─────────────────────────────

_WRAPPERS = {"Style", "Annotation", "Tooltip", "Mouseover", "Hyperlink", "Button", "EventHandler", "Legended",
             "Callout", "Highlighted", "Labeled", "StatusArea", "PopupWindow", "Dynamic", "Placed"}


def check_content(x, dim, fname, depth=0):
    """Walk graphics content: lists, primitives, directives, wrappers, transforms and style rules."""
    if depth > 200:
        raise ValueError(f"{fname}: graphics content is nested too deeply")
    if K.is_list(x):
        for i in x.args:
            check_content(i, dim, fname, depth + 1)
        return
    if x is None or isinstance(x, Symbol):
        return
    if isinstance(x, (str, bool, int, float)) or (not isinstance(x, Expr)):
        raise TypeError(f"{fname}: {K_input(x)} is not a graphics primitive or directive")
    if is_rule(x):
        return                                           # style option inside the content (Antialiasing -> False)
    h = x.head
    if not isinstance(h, str):
        return
    if h in ("Graphics", "Graphics3D"):
        raise ValueError(f"{fname}: a {h} object cannot appear inside graphics content; use Inset[{h}[...]]")
    if is_real_value(x):
        raise TypeError(f"{fname}: {K_input(x)} is not a graphics primitive or directive")
    spec = PRIMITIVES.get(h)
    if spec is not None:
        if dim not in spec["dims"]:
            kind = "two" if 2 in spec["dims"] else "three"
            raise ValueError(f"{fname}: {h} is a {kind}-dimensional primitive and cannot be used in {fname}")
        if spec["validate"] is not None:
            spec["validate"](x, dim)
        return
    if h in ("GraphicsGroup",) and x.args:
        check_content(x.args[0], dim, fname, depth + 1)
    elif h == "GraphicsComplex" and x.args:
        pts = x.args[0]
        if K.is_list(pts) and pts.args and K.is_list(pts.args[0]) and len(pts.args[0].args) != dim:
            raise ValueError(f"{fname}: GraphicsComplex points must be {dim}-dimensional in {fname}")
        if len(x.args) > 1:
            check_content(x.args[1], dim, fname, depth + 1)
    elif h in ("Translate", "Rotate", "Scale", "GeometricTransformation") and x.args:
        check_content(x.args[0], dim, fname, depth + 1)
    elif h in _WRAPPERS and x.args:
        check_content(x.args[0], dim, fname, depth + 1)


# ───────────────────────────── Graphics, Graphics3D ─────────────────────────────

def _container(head, dim):
    def impl(*args):
        if not args:
            raise Unevaluated()
        content, opts = args[0], list(args[1:])
        if opt_name(content) is not None and not K.is_list(content):
            raise ValueError(f"{head}: the first argument must be graphics content, not the option "
                             f"{K_input(content)}")
        for r in opts:
            if opt_name(r) is None:
                raise ValueError(f"{head}: expected {head}[primitives, options]; {K_input(r)} is not an option")
        check_content(content, dim, head)
        check_options(opts, dim, head)
        raise Unevaluated()
    impl.__annotations__ = {"args": "Expr | Option", "return": "Expr"}
    return impl


fn("Graphics", area=AREA, status="added",
   notes="Graphics[primitives, options] inert 2D graphics object (displayed by Rasterize / the notebook front end). "
         "Contents are checked: numbers or strings as primitives raise TypeError, a Graphics inside Graphics or a "
         "3D-only primitive ValueError; options must be known Graphics options (AspectRatio, PlotRange, Axes, "
         "Background, ImageSize, ...; unknown names raise ValueError, deviating from Wolfram's message-only "
         "behaviour) and known settings are validated. Check: reference (Wolfram documentation forms) in tests."
   )(_container("Graphics", 2))
fn("Graphics3D", area=AREA, status="added",
   notes="Graphics3D[primitives, options] inert 3D graphics object; contents checked as for Graphics (2D-only "
         "primitives such as Raster raise ValueError); the 3D options Boxed, BoxRatios, BoxStyle, ClipPlanes, "
         "Lighting, ViewPoint, ViewVector, ViewAngle, ViewCenter, ViewVertical, ViewRange, ViewMatrix, "
         "ViewProjection, RotationAction, RenderingOptions, ... are validated; unknown options raise ValueError. "
         "Rendering of 3D graphics is not available yet (Rasterize raises NotImplementedError). "
         "Check: reference (Wolfram documentation forms) in tests.")(_container("Graphics3D", 3))


@fn("GraphicsGroup", area=AREA, status="added",
    notes="GraphicsGroup[{g1, g2, ...}, opts] inert group of graphics objects (selection unit in a notebook); valid "
          "in Graphics and Graphics3D; the contents are checked like Graphics contents. Check: reference in tests.")
def _graphicsgroup(g: Expr, *opts: Option) -> Expr:
    for r in opts:
        if opt_name(r) is None:
            raise ValueError("GraphicsGroup: expected GraphicsGroup[{g1, ...}, options]")
    check_content(g, 2 if not _contains_3d(g) else 3, "GraphicsGroup")
    raise Unevaluated()


def _contains_3d(x):
    if K.is_list(x):
        return any(_contains_3d(i) for i in x.args)
    if isinstance(x, Expr) and isinstance(x.head, str):
        spec = PRIMITIVES.get(x.head)
        return spec is not None and spec["dims"] == {3}
    return False


# ───────────────────────────── GraphicsComplex ─────────────────────────────

_POINT_HEADS = ("Point", "Line", "Polygon", "Arrow", "BezierCurve", "BSplineCurve", "Triangle", "Simplex",
                "Tetrahedron", "Hexahedron", "Prism", "Pyramid", "Tube", "JoinedCurve", "FilledCurve")
_CENTER_HEADS = ("Disk", "Circle", "Sphere", "Ball", "Text", "Inset", "Rectangle", "Cuboid", "Cylinder", "Cone")


def _indices(x):
    """Integer vertex indices used as coordinates in a GraphicsComplex data expression."""
    out = []
    if K.is_list(x):
        for i in x.args:
            out += _indices(i)
        return out
    if not isinstance(x, Expr) or not isinstance(x.head, str):
        return out
    h = x.head
    if h in _POINT_HEADS and x.args:
        out += _int_leaves(x.args[0])
    elif h in _CENTER_HEADS and x.args:
        k = 1 if h in ("Text", "Inset") else 0
        if len(x.args) > k:
            for a in x.args[k:k + (2 if h in ("Rectangle", "Cuboid") else 1)]:
                out += _int_leaves(a) if (K.is_int(a) or K.is_list(a)) else []
    elif h in ("GraphicsGroup", "Style", "Annotation", "Tooltip", "Translate", "Rotate", "Scale") and x.args:
        out += _indices(x.args[0])
    return out


def _int_leaves(x):
    if K.is_int(x) and not isinstance(x, bool):
        return [int(x)]
    if K.is_list(x) and all(K.is_int(v) or K.is_list(v) for v in x.args):
        out = []
        for v in x.args:
            out += _int_leaves(v)
        return out
    return []


_VERTEX_OPTS = ("VertexColors", "VertexNormals", "VertexTextureCoordinates", "ContentSelectable")


@fn("GraphicsComplex", area=AREA, status="added",
    notes="GraphicsComplex[{pt1, pt2, ...}, data, opts] inert: integers i used as coordinates in primitives of data "
          "stand for pti. The points must form a numeric n x 2 or n x 3 array (strings or ragged lists raise "
          "TypeError/ValueError), indices in point-list primitives must lie in 1..n (else ValueError), and "
          "VertexColors / VertexNormals / VertexTextureCoordinates must have one entry per point. "
          "Normal[GraphicsComplex[...]] substitutes the coordinates (and per-vertex options) into the primitives. "
          "Check: exact (Normal substitution) and reference values in tests.")
def _graphicscomplex(pts: MatrixOf[Real], data: Expr, *opts: Option) -> Expr:
    if not K.is_list(pts):
        if _bad_kind(pts):
            raise TypeError(f"GraphicsComplex: the points must be a list of coordinates, not {K_input(pts)}")
        raise Unevaluated()
    dims = set()
    for p in pts.args:
        d = coords(p, "GraphicsComplex", "point", scaled_ok=False)
        dims.add(d)
    if len(dims) > 1:
        raise ValueError("GraphicsComplex: all points must have the same dimension")
    n = len(pts.args)
    bad = [i for i in _indices(data) if not 1 <= i <= n]
    if bad:
        raise ValueError(f"GraphicsComplex: vertex index {bad[0]} is out of range 1..{n}")
    for r in opts:
        nm = opt_name(r)
        if nm is None or nm not in _VERTEX_OPTS:
            raise ValueError(f"GraphicsComplex: {K_input(r)} is not a GraphicsComplex option ({', '.join(_VERTEX_OPTS)})")
        v = r.args[1]
        if nm != "ContentSelectable" and K.is_list(v) and len(v.args) != n:
            raise ValueError(f"GraphicsComplex: {nm} needs one entry per point ({n}), got {len(v.args)}")
        if nm in OPTION_VALIDATORS:
            OPTION_VALIDATORS[nm](v, "GraphicsComplex")
    raise Unevaluated()


# ───────────────────────────── directives ─────────────────────────────

def _directive(name, check, notes, lo=0, hi=8):
    """Register an inert directive head: check(args) validates (raising TypeError/ValueError, or Unevaluated
    for symbolic arguments), then the expression stays as it is."""
    DIRECTIVE_HEADS.add(name)

    def impl(*args):
        nargs(args, lo, hi, name)
        check(*args)
        raise Unevaluated()
    impl.__annotations__ = {"args": "Expr", "return": "Expr"}
    fn(name, area=AREA, status="added", notes=notes)(impl)


def _thickness(fname):
    def check(r):
        size_value(r, fname, "thickness")
    return check


def _dash_list(x, fname):
    if isinstance(x, Symbol):
        if x.name in SIZE_SYMBOLS:
            return
        raise Unevaluated()
    if K.is_list(x):
        for d in x.args:
            size_value(d, fname, "dash length")
        return
    size_value(x, fname, "dash length")


def _dashing(fname):
    def check(spec, offset=None, cap=None):
        _dash_list(spec, fname)
        if offset is not None:
            real_value(offset, fname, "offset")
        if cap is not None and not is_sym(cap, "None"):
            _string_choice(cap, ("Butt", "Round", "Square"), fname, "cap form")
    return check


_SIZES = "a non-negative number or Tiny, Small, Medium, Large"

_directive("Thickness", _thickness("Thickness"),
           "Thickness[r] inert directive: lines drawn with thickness r as a fraction of the plot width; Tiny, Small, "
           "Medium, Large are absolute sizes. r must be " + _SIZES + " (TypeError for strings, ValueError for "
           "negative r); symbolic r stays unevaluated. Check: reference in tests.", 1, 1)
_directive("AbsoluteThickness", _thickness("AbsoluteThickness"),
           "AbsoluteThickness[d] inert directive: line thickness d in printer's points; d must be " + _SIZES +
           "; symbolic d stays unevaluated. Check: reference in tests.", 1, 1)
_directive("PointSize", _thickness("PointSize"),
           "PointSize[d] inert directive: point diameter d as a fraction of the plot width (Tiny ... Large "
           "absolute); d must be " + _SIZES + ". Check: reference in tests.", 1, 1)
_directive("AbsolutePointSize", _thickness("AbsolutePointSize"),
           "AbsolutePointSize[d] inert directive: point diameter d in printer's points; d must be " + _SIZES +
           ". Check: reference in tests.", 1, 1)
_directive("Dashing", _dashing("Dashing"),
           "Dashing[r], Dashing[{r1, r2, ...}], Dashing[{...}, offset] and Dashing[{...}, offset, capform] inert "
           "directive (lengths as fractions of the plot width, Tiny ... Large absolute, Dashing[{}] solid; capform "
           "\"Butt\" | \"Round\" | \"Square\" | None). Bad lengths raise ValueError/TypeError; symbolic stays "
           "unevaluated. Check: reference (Dashed = Dashing[{Small, Small}] etc.) in tests.", 1, 3)
_directive("AbsoluteDashing", _dashing("AbsoluteDashing"),
           "AbsoluteDashing[d], AbsoluteDashing[{d1, d2, ...}], [{...}, offset], [{...}, offset, capform] inert "
           "directive with lengths in printer's points; validated like Dashing. Check: reference in tests.", 1, 3)


def _capform(t):
    if not is_sym(t, "None"):
        _string_choice(t, ("Butt", "Round", "Square"), "CapForm", "cap type")


def _joinform(t):
    if is_sym(t, "None"):
        return
    if K.is_list(t) and len(t.args) == 2 and t.args[0] == "Miter":
        real_arg(t.args[1], "JoinForm", "miter limit", 0)
        return
    _string_choice(t, ("Bevel", "Miter", "Round"), "JoinForm", "join type")


_directive("CapForm", _capform,
           "CapForm[\"Butt\" | \"Round\" | \"Square\" | None] inert directive for line ends (None = \"Butt\"); other "
           "strings raise ValueError, non-strings TypeError. Check: reference in tests.", 1, 1)
_directive("JoinForm", _joinform,
           "JoinForm[\"Bevel\" | \"Miter\" | {\"Miter\", d} | \"Round\" | None] inert directive for line joins "
           "(None = \"Bevel\"); other strings raise ValueError, a negative miter limit ValueError. "
           "Check: reference in tests.", 1, 1)


def _arrowheads(spec, *opts):
    for r in opts:
        if opt_name(r) != "Appearance":
            raise ValueError(f"Arrowheads: {K_input(r)} is not an option of Arrowheads (Appearance)")
        if not (is_sym(r.args[1], "Automatic") or r.args[1] in ("Flat", "Projected")):
            raise ValueError("Arrowheads: Appearance must be Automatic, \"Flat\" or \"Projected\"")
    if is_sym(spec, "Automatic", "None"):
        return
    if K.is_list(spec):
        for e in spec.args:
            if K.is_list(e):
                if not 2 <= len(e.args) <= 3:
                    raise ValueError(f"Arrowheads: element {K_input(e)} must be {{s, pos}} or {{s, pos, g}}")
                if not is_sym(e.args[0], "Automatic"):
                    size_value(e.args[0], "Arrowheads", "arrowhead size", lo=None)
                real_value(e.args[1], "Arrowheads", "arrowhead position")
            else:
                size_value(e, "Arrowheads", "arrowhead size", lo=None)
        return
    size_value(spec, "Arrowheads", "arrowhead size", lo=None)


_directive("Arrowheads", _arrowheads,
           "Arrowheads[s], Arrowheads[{s1, s2, ...}] (sizes at evenly spaced positions, negative = reversed), "
           "Arrowheads[{{s, pos}, {s, pos, g}, ...}], Arrowheads[Automatic] inert directive; Appearance -> "
           "\"Flat\" | \"Projected\" | Automatic. Malformed elements raise ValueError/TypeError. "
           "Check: reference in tests.", 1, 2)


def _edgeform(*g):
    for x in g:
        directive_like(x, "EdgeForm")


def _faceform(*g):
    for x in g:
        directive_like(x, "FaceForm")


_directive("EdgeForm", _edgeform,
           "EdgeForm[] (no edges), EdgeForm[g] / EdgeForm[{g1, ...}] inert directive giving the style of edges of "
           "polygons, disks and rectangles (EdgeForm[None] also draws none); numbers or strings as directives raise "
           "TypeError. Check: reference in tests.", 0, 1)
_directive("FaceForm", _faceform,
           "FaceForm[] (no faces), FaceForm[g], FaceForm[g, gback] (front and back faces in 3D) inert directive; "
           "numbers or strings as directives raise TypeError. Check: reference in tests.", 0, 2)


@fn("Directive", area=AREA, status="added",
    notes="Directive[g1, g2, ...] a single compound graphics directive (inert); Directive[{g1, g2, ...}] is "
          "normalised to Directive[g1, g2, ...] as documented; numbers or strings as directives raise TypeError. "
          "Check: reference in tests.")
def _directive_head(*gs: Expr) -> Expr:
    if len(gs) == 1 and K.is_list(gs[0]):
        return Expr("Directive", *gs[0].args)
    for g in gs:
        if not is_rule(g):
            directive_like(g, "Directive")
    raise Unevaluated()


DIRECTIVE_HEADS.add("Directive")


def _glow(*c):
    if c:
        color_arg(c[0], "Glow", none_ok=True)


def _specularity(*a):
    if not a:
        return
    s = a[0]
    if parse_color(s) is None:
        if K.is_list(s) or isinstance(s, (str, bool)):
            raise TypeError(f"Specularity: {K_input(s)} must be a number or a colour")
        if is_real_value(s):
            real_arg(s, "Specularity", "specularity", 0)
        else:
            raise Unevaluated()
    if len(a) == 2:
        real_arg(a[1], "Specularity", "specular exponent", 0, strict=True)


def _texture(obj, *rest):
    if isinstance(obj, (bool, int, float)) or (isinstance(obj, str) and not obj):
        raise TypeError(f"Texture: {K_input(obj)} is not a texture (an image, graphics, array or colour list)")
    for r in rest:
        if opt_name(r) is not None and opt_name(r) not in ("TextureMapping",):
            raise ValueError(f"Texture: {opt_name(r)} is not an option of Texture")


_directive("Glow", _glow,
           "Glow[] (= Glow[Black], no glow) and Glow[col] inert 3D directive for colour emitted independently of "
           "lighting; a non-colour raises TypeError, symbolic stays unevaluated. Check: reference in tests.", 0, 1)
_directive("Specularity", _specularity,
           "Specularity[s] (s >= 0 or a colour; Specularity[s] = Specularity[GrayLevel[s]]), Specularity[s, n] "
           "(specular exponent n > 0) and Specularity[] (none) inert 3D directive; invalid values raise "
           "ValueError/TypeError. Check: reference in tests.", 0, 2)
_directive("Texture", _texture,
           "Texture[obj] and Texture[obj, map] inert directive using obj (image, graphics, array, colour list) as a "
           "texture on faces; numbers or booleans raise TypeError. Check: reference in tests.", 1, 3)


# lights
def _light_color(c, fname):
    if K.is_list(c) and len(c.args) == 2 and isinstance(c.args[1], Expr) and c.args[1].head == "Specularity":
        color_arg(c.args[0], fname)
        return
    color_arg(c, fname, "light colour")


def _light_point(p, fname, what="position"):
    if isinstance(p, Expr) and p.head in ("Scaled", "ImageScaled"):
        return
    coords(p, fname, what, dims=(3,), scaled_ok=False)


def _ambient(c, *opts):
    _light_color(c, "AmbientLight")


def _directional(c, p, *opts):
    _light_color(c, "DirectionalLight")
    if K.is_list(p) and len(p.args) == 2 and all(K.is_list(q) or isinstance(q, Expr) for q in p.args) \
            and not all(is_real_value(q) for q in p.args):
        for q in p.args:
            _light_point(q, "DirectionalLight")
    else:
        _light_point(p, "DirectionalLight")


def _attenuation(a, fname):
    if K.is_list(a):
        if len(a.args) != 3:
            raise ValueError(f"{fname}: attenuation must be a number or {{a, b, c}}")
        for v in a.args:
            real_arg(v, fname, "attenuation coefficient", 0)
    else:
        real_arg(a, fname, "attenuation", 0)


def _pointlight(c, p, att=None, *opts):
    _light_color(c, "PointLight")
    _light_point(p, "PointLight")
    if att is not None and opt_name(att) is None:
        _attenuation(att, "PointLight")


def _spotlight(c, p, ang, att=None, *opts):
    _light_color(c, "SpotLight")
    if K.is_list(p) and len(p.args) == 2 and not all(is_real_value(q) for q in p.args):
        for q in p.args:
            _light_point(q, "SpotLight")
    else:
        _light_point(p, "SpotLight")
    if K.is_list(ang):
        if len(ang.args) != 2:
            raise ValueError("SpotLight: the angle must be α or {α, s}")
        a, s = (real_value(v, "SpotLight", "angle") for v in ang.args)
        if s < 0:
            raise ValueError("SpotLight: the spot exponent must be non-negative")
    else:
        a = real_value(ang, "SpotLight", "half-angle")
    if not 0 < a <= 3.141592653589793:
        raise ValueError("SpotLight: the half-angle must lie in (0, Pi]")
    if att is not None and opt_name(att) is None:
        _attenuation(att, "SpotLight")


_LIGHT_NOTE = (" The light colour may be any colour or {dcol, Specularity[scol]}; positions are {x, y, z}, "
               "Scaled[{x, y, z}] or ImageScaled[{x, y, z}]. Invalid values raise TypeError/ValueError; symbolic "
               "values stay unevaluated. Check: reference in tests.")
_directive("AmbientLight", _ambient, "AmbientLight[col] inert 3D light source: uniform ambient light of colour col "
           "(a Lighting setting or directive)." + _LIGHT_NOTE, 1, 2)
_directive("DirectionalLight", _directional, "DirectionalLight[col, pt] (light from pt towards the centre) and "
           "DirectionalLight[col, {pt1, pt2}] (along pt1 -> pt2) inert 3D light source." + _LIGHT_NOTE, 2, 3)
_directive("PointLight", _pointlight, "PointLight[col, pt] and PointLight[col, pt, att] (att >= 0 or {a, b, c} "
           "for 1/(a + b d + c d^2)) inert 3D light source." + _LIGHT_NOTE, 2, 4)
_directive("SpotLight", _spotlight, "SpotLight[col, pt | {pt, target}, α | {α, s}] and SpotLight[..., att] inert "
           "3D spotlight with half-angle α in (0, Pi] and spot exponent s >= 0." + _LIGHT_NOTE, 3, 5)


# shadings and fillings
def _color_list(spec, fname, allow_positions=True):
    """{c1, c2, ...}, {p1, ...} -> {c1, ...}, {p1 -> c1, ...}, {{p1, c1}, ...} or a "scheme" name."""
    if isinstance(spec, str):
        return
    if is_rule(spec) and allow_positions:
        ps, cs = spec.args
        if not (K.is_list(ps) and K.is_list(cs)):
            raise ValueError(f"{fname}: expected {{p1, p2, ...}} -> {{c1, c2, ...}}")
        if len(ps.args) != len(cs.args):
            raise ValueError(f"{fname}: positions and colours must have the same length")
        for p in ps.args:
            if not (isinstance(p, Expr) and p.head == "ImageScaled"):
                real_value(p, fname, "position")
        for c in cs.args:
            color_arg(c, fname)
        return
    if not K.is_list(spec):
        if isinstance(spec, (bool, int, float)):
            raise TypeError(f"{fname}: {K_input(spec)} is not a list of colours")
        color_arg(spec, fname)
        return
    for e in spec.args:
        if is_rule(e) and allow_positions:
            real_value(e.args[0], fname, "position")
            color_arg(e.args[1], fname)
        elif K.is_list(e) and len(e.args) == 2 and allow_positions and parse_color(e.args[1]) is not None:
            real_value(e.args[0], fname, "position")
        else:
            color_arg(e, fname)


def _density_color(fname, shapes=None):
    def check(*a):
        rest = list(a)
        if rest and is_real_value(rest[0]) and not isinstance(rest[0], Symbol):
            d = real_value(rest.pop(0), fname, "density")
            if not 0 <= d <= 1:
                raise ValueError(f"{fname}: the density must lie between 0 and 1")
        if rest and not isinstance(rest[0], str):
            color_arg(rest.pop(0), fname)
        if rest and shapes is not None:
            _string_choice(rest.pop(0), shapes, fname, "shape")
        if rest:
            raise ValueError(f"{fname}: unexpected argument {K_input(rest[0])}")
    return check


def _attenuation_shading(fname, n):
    def check(*a):
        if not a:
            return
        d = a[0]
        if K.is_list(d):
            if not 1 <= len(d.args) <= n:
                raise ValueError(f"{fname}: expected at most {n} attenuation factors")
            for v in d.args:
                real_arg(v, fname, "attenuation factor", 0)
        else:
            real_arg(d, fname, "attenuation factor", 0)
    return check


def _weighted_colors(fname, k):
    def check(*a):
        if not a:
            return
        s = a[0]
        if isinstance(s, str) or parse_color(s) is not None:
            return
        if is_rule(s):
            ws, cs = s.args
            if not (K.is_list(ws) and K.is_list(cs)) or len(ws.args) != len(cs.args):
                raise ValueError(f"{fname}: expected {{w1, ...}} -> {{c1, ...}} with equal lengths")
            for w in ws.args:
                real_arg(w, fname, "weight", 0)
            for c in cs.args:
                color_arg(c, fname)
            return
        if K.is_list(s):
            for c in s.args:
                color_arg(c, fname)
            return
        color_arg(s, fname)
    return check


_SHADE = (" Inert 3D surface-shading directive; invalid values raise TypeError/ValueError, symbolic values stay "
          "unevaluated. Check: reference (Wolfram documentation forms) in tests.")
_directive("FlatShading", _attenuation_shading("FlatShading", 2),
           "FlatShading[] (= FlatShading[{1, 1}]), FlatShading[d], FlatShading[{d, a}] (diffuse and ambient "
           "attenuation factors >= 0)." + _SHADE, 0, 1)
_directive("GouraudShading", _attenuation_shading("GouraudShading", 3),
           "GouraudShading[] (= {1, 1, 1}), GouraudShading[d], [{d, s}], [{d, s, a}] (attenuation factors >= 0)."
           + _SHADE, 0, 1)
_directive("PhongShading", _attenuation_shading("PhongShading", 3),
           "PhongShading[] (= {1, 1, 1}), PhongShading[d], [{d, s}], [{d, s, a}] (attenuation factors >= 0)."
           + _SHADE, 0, 1)
_directive("GoochShading", _weighted_colors("GoochShading", 2),
           "GoochShading[], GoochShading[col], [{ccol, wcol}], [{w1, w2} -> {ccol, wcol}], [\"scheme\"] (cool-to-warm "
           "shading)." + _SHADE, 0, 1)
_directive("ToonShading", _weighted_colors("ToonShading", 3),
           "ToonShading[] (= {GrayLevel[0.2], GrayLevel[0.7], GrayLevel[1.]}), ToonShading[col], [{dcol, bcol, hcol}], "
           "[{w1, w2, w3} -> {cols}], [\"scheme\"] (cel shading)." + _SHADE, 0, 1)
_directive("HatchShading", _density_color("HatchShading"),
           "HatchShading[] (= HatchShading[0.5, Black]), HatchShading[d], [col], [d, col] with density d in [0, 1]."
           + _SHADE, 0, 2)
_directive("StippleShading", _density_color("StippleShading"),
           "StippleShading[] (= StippleShading[0.5, Black]), StippleShading[d], [col], [d, col] with density d in "
           "[0, 1]." + _SHADE, 0, 2)
_directive("HalftoneShading", _density_color("HalftoneShading", ("Disk", "Hexagon", "Line", "Square", "Triangle")),
           "HalftoneShading[] (= HalftoneShading[0.5, Black, \"Disk\"]), [d], [col], [\"shape\"], [d, col, \"shape\"] "
           "with d in [0, 1] and shape \"Disk\", \"Hexagon\", \"Line\", \"Square\" or \"Triangle\"." + _SHADE, 0, 3)


def _material(*a):
    if not a:
        return
    m = a[0]
    if isinstance(m, str):
        return
    if K.is_list(m) and len(m.args) == 2 and isinstance(m.args[0], str):
        color_arg(m.args[1], "MaterialShading")
        return
    if isinstance(m, Expr) and m.head == "Association" or is_rule(m):
        return
    if isinstance(m, (bool, int, float)):
        raise TypeError(f"MaterialShading: {K_input(m)} is not a material specification")
    raise Unevaluated()


_directive("MaterialShading", _material,
           "MaterialShading[] (= <|\"BaseColor\" -> White|>), MaterialShading[\"material\"], [{\"material\", col}], "
           "[<|param -> value, ...|>] or [param -> value, ...] physically based shading." + _SHADE, 0, 8)


def _gradient(fname, second):
    def check(*a):
        if not a:
            return
        _color_list(a[0], fname)
        if len(a) > 1:
            second(a[1])
        if len(a) > 2 and not is_sym(a[2], "None", "Automatic"):
            _string_choice(a[2], ("Fixed", "Stretched", "Reflect", "Repeat", "Pad", "None"), fname, "padding")
    return check


def _lin_dir(d):
    if is_sym(d, "Left", "Right", "Top", "Bottom", "Up", "Down", "Center"):
        return
    if K.is_list(d) and all(is_sym(v) or is_real_value(v) for v in d.args) and len(d.args) == 2:
        if all(is_real_value(v) for v in d.args):
            return
        return
    real_value(d, "LinearGradientFilling", "direction angle")


def _center(c, fname="RadialGradientFilling"):
    if K.is_list(c) and len(c.args) == 2 and K.is_list(c.args[0]):
        coords(c.args[0], fname, "centre", dims=(2,))
        if K.is_list(c.args[1]):
            for v in c.args[1].args:
                real_arg(v, fname, "semi-axis", 0)
        else:
            real_arg(c.args[1], fname, "radius", 0)
        return
    if is_sym(c, "Center"):
        return
    coords(c, fname, "centre", dims=(2,))


_FILL = (" Inert 2D filling directive for polygons, disks and other filled objects; colours may be given as "
         "{c1, ...}, {p1, ...} -> {c1, ...}, {p1 -> c1, ...}, {{p1, c1}, ...} or a \"scheme\" name. Invalid values "
         "raise TypeError/ValueError; symbolic values stay unevaluated. Check: reference in tests.")
_directive("LinearGradientFilling", _gradient("LinearGradientFilling", _lin_dir),
           "LinearGradientFilling[] (= [{Black, White}]), [colors], [colors, dir] (Left, Right, Top, Bottom, "
           "{x, y} or an angle θ), [colors, dir, padding]." + _FILL, 0, 3)
_directive("RadialGradientFilling", _gradient("RadialGradientFilling", _center),
           "RadialGradientFilling[] (= [{Black, White}]), [colors], [colors, {x, y}], [colors, {{x, y}, {r1, r2}}], "
           "[colors, centre, padding]." + _FILL, 0, 3)
_directive("ConicGradientFilling", _gradient("ConicGradientFilling", lambda c: _center(c, "ConicGradientFilling")),
           "ConicGradientFilling[] (= [{Black, White}]), [colors] (around the centre, angles as positions), "
           "[colors, {x, y}], [colors, centre, padding]." + _FILL, 0, 3)


def _hatch(*a):
    if not a:
        return
    t = a[0]
    if isinstance(t, str):
        _string_choice(t, ("Diagonal", "Horizontal", "Vertical", "DiagonalCrossed", "Crossed"), "HatchFilling",
                       "hatching name")
    else:
        real_value(t, "HatchFilling", "angle")
    for v, what in zip(a[1:], ("line thickness", "gap")):
        real_arg(v, "HatchFilling", what, 0)


def _pattern(obj, *a):
    if isinstance(obj, (bool, int, float)):
        raise TypeError(f"PatternFilling: {K_input(obj)} is not a pattern (a graphic, an image or a name)")
    if a:
        s = a[0]
        if not (isinstance(s, Expr) and s.head == "ImageScaled"):
            if K.is_list(s):
                for v in s.args:
                    real_arg(v, "PatternFilling", "size", 0, strict=True)
            else:
                real_arg(s, "PatternFilling", "size", 0, strict=True)
    if len(a) > 1:
        coords(a[1], "PatternFilling", "offset", dims=(2,))


_directive("HatchFilling", _hatch,
           "HatchFilling[] (= HatchFilling[Pi/4]), HatchFilling[\"Diagonal\" | \"Horizontal\" | \"Vertical\"], "
           "HatchFilling[θ], [θ, r] (line thickness), [θ, r, d] (gap); thicknesses must be >= 0." + _FILL, 0, 3)
_directive("PatternFilling", _pattern,
           "PatternFilling[obj] (= [obj, ImageScaled[0.2]]), PatternFilling[\"name\"], [obj, size | {w, h}], "
           "[obj, size, {dx, dy}] tiling fill with a graphic or image; sizes must be positive." + _FILL, 1, 3)


def _blur_r(r, fname):
    if K.is_list(r):
        if len(r.args) != 2:
            raise ValueError(f"{fname}: expected r or {{r1, r2}}")
        for v in r.args:
            real_arg(v, fname, "blur radius", 0)
    else:
        real_arg(r, fname, "blur radius", 0)


def _blurring(*a):
    if a:
        _blur_r(a[0], "Blurring")


def _dropshadow(*a):
    rest = list(a)
    if rest and K.is_list(rest[0]):
        coords(rest.pop(0), "DropShadowing", "offset", dims=(2,), scaled_ok=False)
    if rest and is_real_value(rest[0]) and not isinstance(rest[0], Symbol):
        real_arg(rest.pop(0), "DropShadowing", "blur radius", 0)
    if rest:
        color_arg(rest.pop(0), "DropShadowing")
    if rest:
        raise ValueError(f"DropShadowing: unexpected argument {K_input(rest[0])}")


def _haloing(*a):
    if a:
        color_arg(a[0], "Haloing")
    if len(a) > 1:
        real_value(a[1], "Haloing", "halo width")
    if len(a) > 2:
        real_arg(a[2], "Haloing", "blur radius", 0)


_EFFECT = " Invalid values raise TypeError/ValueError; symbolic values stay unevaluated. Check: reference in tests."
_directive("Blurring", _blurring, "Blurring[] (= Blurring[3]), Blurring[r], Blurring[{r1, r2}] inert 2D directive "
           "drawing objects blurred with radius r >= 0 (printer's points)." + _EFFECT, 0, 1)
_directive("DropShadowing", _dropshadow,
           "DropShadowing[] (= DropShadowing[{-3, -3}, 2, GrayLevel[0, 1/3]]), DropShadowing[{dx, dy}], "
           "[{dx, dy}, r], [{dx, dy}, r, col], [col], [r] inert 2D drop-shadow directive." + _EFFECT, 0, 3)
_directive("Haloing", _haloing, "Haloing[] (= Haloing[White, 2, 0]), Haloing[col], [col, w] (negative w draws "
           "inward), [col, w, r] (blur radius r >= 0) inert 2D halo directive." + _EFFECT, 0, 3)


# coordinate wrappers
def _offset(d, *pos):
    if K.is_list(d):
        coords(d, "Offset", "offset", dims=(2, 3), scaled_ok=False)
    else:
        real_value(d, "Offset", "offset")
    if pos:
        coords(pos[0], "Offset", "position")


def _scaled(p, *base):
    if K.is_list(p):
        coords(p, "Scaled", "scaled coordinate", scaled_ok=False)
    else:
        real_value(p, "Scaled", "scaled length")
    if base:
        coords(base[0], "Scaled", "base point", scaled_ok=False)


def _imagescaled(p):
    if K.is_list(p):
        coords(p, "ImageScaled", "scaled coordinate", scaled_ok=False)
    else:
        real_value(p, "ImageScaled", "scaled length")


_COORD = " Non-numeric coordinates raise TypeError, wrong lengths ValueError; symbolic values stay unevaluated. " \
         "Check: reference in tests."
_directive("Offset", _offset, "Offset[{dx, dy}, pos] position pos ({x, y}, Scaled[...] or ImageScaled[...]) moved "
           "by {dx, dy} printer's points; Offset[{dx, dy}] / Offset[d] an absolute size (inert)." + _COORD, 1, 2)
_directive("Scaled", _scaled, "Scaled[{x, y(, z)}] coordinates scaled 0..1 over the plot range, Scaled[{dx, ...}, "
           "{x0, ...}] a scaled offset from an ordinary point, Scaled[r] a scaled length (inert; also an option "
           "setting for sizes)." + _COORD, 1, 2)
_directive("ImageScaled", _imagescaled, "ImageScaled[{x, y}] coordinates scaled 0..1 across the whole image, "
           "ImageScaled[r] a scaled length (inert)." + _COORD, 1, 1)

for _h in ("Offset", "Scaled", "ImageScaled"):
    DIRECTIVE_HEADS.discard(_h)


# ───────────────────────────── named directive symbols ─────────────────────────────

_SYMBOL_DIRECTIVES = {
    "Thick": Expr("Thickness", Symbol("Large")), "Thin": Expr("Thickness", Symbol("Tiny")),
    "Dashed": Expr("Dashing", L(Symbol("Small"), Symbol("Small"))), "Dotted": Expr("Dashing", L(0, Symbol("Small"))),
    "DotDashed": Expr("Dashing", L(0, Symbol("Small"), Symbol("Small"), Symbol("Small"))),
}
for _n, _v in _SYMBOL_DIRECTIVES.items():
    SYMBOL_VALUES[_n] = _v
    record(_n, AREA, "added", f"{_n} evaluates to {K_input(_v)}, as documented by Wolfram (an assignment by the user "
                              f"takes precedence). Check: reference (Wolfram documentation) in tests.")
for _n in SIZE_SYMBOLS:
    record(_n, AREA, "added", f"{_n} is an inert symbolic size used by Thickness, PointSize, Dashing, Arrowheads, "
                              f"ImageSize and font sizes (absolute, independent of the plot size). "
                              f"Check: reference in tests.")
for _n, _d in (("Front", "front (in 3D the minimum depth)"), ("Back", "back (in 3D the maximum depth)")):
    record(_n, AREA, "added", f"{_n} is an inert symbol for the {_d} of a graphic, used in placement and alignment "
                              f"specifications and as a ViewPoint setting. Check: reference in tests.")


# ───────────────────────────── option validators ─────────────────────────────

def _v_bool(name, extra=()):
    def v(x, fname):
        if x is True or x is False or is_sym(x, *extra):
            return
        raise ValueError(f"{fname}: {name} must be True or False{' or ' + ', '.join(extra) if extra else ''}, "
                         f"not {K_input(x)}")
    return v


def _v_boxratios(x, fname):
    if is_sym(x, "Automatic"):
        return
    if K.is_list(x) and len(x.args) == 3:
        for c in x.args:
            if real_value(c, fname, "box ratio") <= 0:
                raise ValueError(f"{fname}: BoxRatios must be positive")
        return
    raise ValueError(f"{fname}: BoxRatios must be Automatic or {{sx, sy, sz}}, not {K_input(x)}")


def _v_directive(name):
    def v(x, fname):
        directive_like(x, fname, f"{name} setting")
    return v


def _v_clipplanes(x, fname):
    if is_sym(x, "None"):
        return
    items = list(x.args) if K.is_list(x) and (not x.args or not all(is_real_value(v) for v in x.args)) else [x]
    for p in items:
        if isinstance(p, Expr) and p.head == "InfinitePlane":
            continue
        if K.is_list(p) and len(p.args) == 4:
            for c in p.args:
                real_value(c, fname, "clip plane coefficient")
            continue
        raise ValueError(f"{fname}: ClipPlanes must be None, {{a, b, c, d}}, InfinitePlane[...] or a list of them")


_LIGHT_NAMES = ("Standard", "Neutral", "Accent", "ThreePoint", "Classic")
_LIGHT_HEADS = ("AmbientLight", "DirectionalLight", "PointLight", "SpotLight")


def _v_lighting(x, fname):
    if is_sym(x, "Automatic", "None"):
        return
    if isinstance(x, str):
        _string_choice(x, _LIGHT_NAMES, fname, "Lighting")
        return
    if parse_color(x) is not None:
        return
    if K.is_list(x):
        for i, s in enumerate(x.args):
            if i == 0 and is_sym(s, "None"):
                continue
            if isinstance(s, Expr) and s.head in _LIGHT_HEADS:
                continue
            if K.is_list(s) and s.args and s.args[0] in ("Ambient", "Directional", "Point", "Spot"):
                continue
            raise ValueError(f"{fname}: {K_input(s)} is not a light source (AmbientLight, DirectionalLight, "
                             f"PointLight, SpotLight)")
        return
    raise ValueError(f"{fname}: Lighting must be Automatic, None, a named lighting, a colour or a list of lights")


_VP_SYMBOLS = ("Above", "Below", "Front", "Back", "Left", "Right", "Top", "Bottom", "Automatic")


def _vec3(x, fname, what, allow_inf=False):
    if not (K.is_list(x) and len(x.args) == 3):
        raise ValueError(f"{fname}: {what} must be a list {{x, y, z}}, not {K_input(x)}")
    for c in x.args:
        if allow_inf and (is_sym(c, "Infinity") or (isinstance(c, Expr) and c.head in ("DirectedInfinity",)) or
                          (isinstance(c, Expr) and c.head == "Times" and any(is_sym(a, "Infinity") for a in c.args))):
            continue
        real_value(c, fname, what)


def _v_viewpoint(x, fname):
    if is_sym(x, *_VP_SYMBOLS):
        return
    if K.is_list(x) and len(x.args) in (2, 3) and all(is_sym(v, *_VP_SYMBOLS) for v in x.args):
        return
    _vec3(x, fname, "ViewPoint", allow_inf=True)


def _v_viewvector(x, fname):
    if is_sym(x, "Automatic"):
        return
    if K.is_list(x) and len(x.args) == 2 and all(K.is_list(v) for v in x.args):
        for v in x.args:
            _vec3(v, fname, "ViewVector point")
        return
    _vec3(x, fname, "ViewVector")


def _v_viewcenter(x, fname):
    if is_sym(x, "Automatic"):
        return
    if K.is_list(x) and len(x.args) == 2 and K.is_list(x.args[0]):
        _vec3(x.args[0], fname, "ViewCenter")
        coords(x.args[1], fname, "ViewCenter image position", dims=(2,), scaled_ok=False)
        return
    _vec3(x, fname, "ViewCenter")


def _v_viewvertical(x, fname):
    if is_sym(x, "Automatic"):
        return
    _vec3(x, fname, "ViewVertical")
    if all(real_value(c, fname, "ViewVertical") == 0 for c in x.args):
        raise ValueError(f"{fname}: ViewVertical must not be the zero vector")


def _v_viewangle(x, fname):
    if is_sym(x, "Automatic", "All"):
        return
    a = real_value(x, fname, "ViewAngle")
    if not 0 < a < 3.141592653589793:
        raise ValueError(f"{fname}: ViewAngle must lie strictly between 0 and Pi radians")


def _v_viewrange(x, fname):
    if is_sym(x, "All", "Automatic"):
        return
    if K.is_list(x) and len(x.args) == 2:
        a, b = (real_value(v, fname, "ViewRange") for v in x.args)
        if not 0 <= a < b:
            raise ValueError(f"{fname}: ViewRange must be {{min, max}} with 0 <= min < max")
        return
    raise ValueError(f"{fname}: ViewRange must be All or {{min, max}}")


def _v_viewmatrix(x, fname):
    if is_sym(x, "Automatic"):
        return
    if K.is_list(x) and len(x.args) == 2:
        for m in x.args:
            if not (K.is_list(m) and len(m.args) == 4 and all(K.is_list(r) and len(r.args) == 4 for r in m.args)):
                raise ValueError(f"{fname}: ViewMatrix must be a pair of 4 x 4 matrices {{t, p}}")
            for r in m.args:
                for c in r.args:
                    real_value(c, fname, "ViewMatrix entry")
        return
    raise ValueError(f"{fname}: ViewMatrix must be Automatic or {{t, p}} with 4 x 4 matrices")


def _v_choice(name, choices, syms=("Automatic",)):
    def v(x, fname):
        if is_sym(x, *syms):
            return
        if isinstance(x, str) and x in choices:
            return
        pre = f"{', '.join(syms)} or " if syms else ""
        raise ValueError(f"{fname}: {name} must be {pre}one of "
                         f"{', '.join(repr(c) for c in choices)}, not {K_input(x)}")
    return v


def _v_rules(name):
    def v(x, fname):
        if K.is_list(x) and all(is_rule(r) for r in x.args):
            return
        if is_rule(x) or is_sym(x, "Automatic", "Inherited"):
            return
        raise ValueError(f"{fname}: {name} must be a list of rules \"name\" -> value")
    return v


def _v_vertex(name, kind):
    def v(x, fname):
        if is_sym(x, "None", "Automatic"):
            return
        if not K.is_list(x):
            raise ValueError(f"{fname}: {name} must be None, Automatic or a list with one entry per vertex")
        for e in x.args:
            if kind == "color":
                if parse_color(e) is None and not is_sym(e, "None", "Automatic"):
                    if isinstance(e, (str, bool, int, float)) or K.is_list(e):
                        raise ValueError(f"{fname}: {name} entries must be colours, not {K_input(e)}")
            elif kind in ("vector", "texture"):
                if K.is_list(e):
                    for c in e.args:
                        real_value(c, fname, f"{name} component")
                elif not (K.is_list(e) or (isinstance(e, Expr) and not _bad_kind(e))):
                    raise ValueError(f"{fname}: {name} entries must be vectors, not {K_input(e)}")
    return v


def _v_datacoords(x, fname):
    if K.is_list(x) and len(x.args) == 2:
        for p in x.args:
            _vec3(p, fname, "VertexDataCoordinates point")
        return
    raise ValueError(f"{fname}: VertexDataCoordinates must be {{{{u1, v1, w1}}, {{u2, v2, w2}}}}")


def _v_opacityfn(x, fname):
    if is_sym(x, "None", "Automatic"):
        return
    if isinstance(x, (bool, str)) or (K.is_list(x)):
        raise ValueError(f"{fname}: OpacityFunction must be a function, a number, None or Automatic")
    if is_real_value(x):
        if not 0 <= real_value(x, fname, "opacity") <= 1:
            raise ValueError(f"{fname}: a constant OpacityFunction must lie in [0, 1]")


def _v_lightingangle(x, fname):
    if is_sym(x, "None", "Automatic"):
        return
    if K.is_list(x):
        if len(x.args) != 2:
            raise ValueError(f"{fname}: LightingAngle must be θ or {{θ, φ}}")
        for v in x.args:
            real_value(v, fname, "LightingAngle")
        return
    real_value(x, fname, "LightingAngle")


def _v_contourshading(x, fname):
    if x is True or x is False or is_sym(x, "None", "Automatic"):
        return
    if K.is_list(x):
        for c in x.args:
            if parse_color(c) is None and not is_sym(c, "None", "Automatic"):
                raise ValueError(f"{fname}: ContourShading entries must be colours, None or Automatic")
        return
    if parse_color(x) is not None:
        return
    raise ValueError(f"{fname}: ContourShading must be Automatic, None, True, False, a colour or a list of colours")


def _v_tcfun(x, fname):
    if isinstance(x, (str, bool, int, float)) or K.is_list(x):
        if not is_sym(x, "Automatic"):
            raise ValueError(f"{fname}: TextureCoordinateFunction must be Automatic or a function")


def _v_texturemapping(x, fname):
    if is_sym(x, "Automatic", "None"):
        return
    if isinstance(x, str) or (isinstance(x, Expr) and not _bad_kind(x)):
        return
    raise ValueError(f"{fname}: TextureMapping must be Automatic, a projection name or a function")


def _v_calloutmarker(x, fname):
    if is_sym(x, "None", "Automatic") or isinstance(x, str) or (isinstance(x, Expr) and not _bad_kind(x)):
        return
    raise ValueError(f"{fname}: CalloutMarker must be None, Automatic, a marker name or a graphic")


def _v_dithering(x, fname):
    if x is True or x is False or is_sym(x, "Automatic", "None"):
        return
    if isinstance(x, str):
        return
    raise ValueError(f"{fname}: Dithering must be True, False, Automatic or a method name")


_OPTION_SPECS = [
    ("Boxed", _v_bool("Boxed"), "option of Graphics3D: whether the edges of the bounding box are drawn (default "
                                "True); Graphics3D validates True/False (else ValueError)."),
    ("BoxRatios", _v_boxratios, "option of Graphics3D: side-length ratios {sx, sy, sz} of the bounding box or "
                                "Automatic (actual coordinate ratios); Graphics3D validates positive ratios."),
    ("BoxStyle", _v_directive("BoxStyle"), "option of Graphics3D: directives for the bounding box edges; numbers or "
                                           "strings raise TypeError."),
    ("ClipPlanes", _v_clipplanes, "option of Graphics3D: None (default), a plane {a, b, c, d} (a x + b y + c z + d "
                                  "> 0 kept), InfinitePlane[...] or a list of them; validated."),
    ("ClipPlanesStyle", _v_directive("ClipPlanesStyle"), "option of Graphics3D: style of the planes given by "
                                                         "ClipPlanes (directives; numbers raise TypeError)."),
    ("Lighting", _v_lighting, "option of Graphics3D (and a style option): Automatic, None, \"Standard\", "
                              "\"Neutral\", \"Accent\", \"ThreePoint\", a colour, or a list of AmbientLight / "
                              "DirectionalLight / PointLight / SpotLight sources (optionally starting with None); "
                              "validated."),
    ("ViewPoint", _v_viewpoint, "option of Graphics3D: view point {x, y, z} relative to the box centre (default "
                                "{1.3, -2.4, 2}; infinite coordinates for orthographic views) or Above, Below, Front, "
                                "Back, Left, Right, {Left, Top}, ...; validated."),
    ("ViewVector", _v_viewvector, "option of Graphics3D: camera position {x, y, z} or {{x, y, z}, {tx, ty, tz}}, "
                                  "or Automatic; validated."),
    ("ViewCenter", _v_viewcenter, "option of Graphics3D: scaled point {x, y, z} (or {{x, y, z}, {u, v}}) shown at "
                                  "the image centre, Automatic = {1/2, 1/2, 1/2}; validated."),
    ("ViewVertical", _v_viewvertical, "option of Graphics3D: direction {x, y, z} (scaled coordinates, default "
                                      "{0, 0, 1}) that is vertical in the image; the zero vector raises ValueError."),
    ("ViewAngle", _v_viewangle, "option of Graphics3D: camera opening angle in (0, Pi), Automatic or All; "
                                "validated."),
    ("ViewRange", _v_viewrange, "option of Graphics3D: All or {min, max} distances from the camera (0 <= min < "
                                "max); validated."),
    ("ViewMatrix", _v_viewmatrix, "option of Graphics3D: Automatic or {t, p} 4 x 4 transformation and projection "
                                  "matrices; validated."),
    ("ViewProjection", _v_choice("ViewProjection", ("Orthographic", "Perspective")),
     "option of Graphics3D: Automatic, \"Orthographic\" or \"Perspective\"; validated."),
    ("RotationAction", _v_choice("RotationAction", ("Fit", "Clip", "KeepCenter"), ()),
     "option of Graphics3D: how the image is sized when rotated interactively, \"Fit\" (default) or \"Clip\"; "
     "validated."),
    ("RenderingOptions", _v_rules("RenderingOptions"), "option of Graphics3D, Style and cells: a list of "
                                                       "\"name\" -> value rendering settings; validated as rules."),
    ("Antialiasing", _v_bool("Antialiasing", ("Automatic",)), "option and style setting (also Antialiasing -> v "
                                                              "inside graphics content): True, False or Automatic."),
    ("VertexColors", _v_vertex("VertexColors", "color"), "option of GraphicsComplex and primitives: one colour per "
                                                         "vertex (or None/Automatic); GraphicsComplex checks the "
                                                         "length."),
    ("VertexNormals", _v_vertex("VertexNormals", "vector"), "option of GraphicsComplex and 3D primitives: one "
                                                            "normal vector per vertex (or None/Automatic)."),
    ("VertexTextureCoordinates", _v_vertex("VertexTextureCoordinates", "texture"),
     "option of GraphicsComplex and primitives: one texture coordinate {u, v(, w)} per vertex (or None)."),
    ("VertexDataCoordinates", _v_datacoords, "option of Raster3D: {{u1, v1, w1}, {u2, v2, w2}} mapping the data to "
                                             "the displayed range (default {{0, 0, 0}, {1, 1, 1}}); validated."),
    ("OpacityFunction", _v_opacityfn, "option of Raster3D (and Image3D): None, Automatic, a constant in [0, 1] or a "
                                      "function giving the opacity of each value; validated by Raster3D."),
    ("OpacityFunctionScaling", _v_bool("OpacityFunctionScaling"), "option of Raster3D: whether values are scaled "
                                                                  "to [0, 1] before OpacityFunction (True/False)."),
    ("LightingAngle", _v_lightingangle, "option of ReliefPlot / ReliefImage: angle θ or {θ, φ} of the simulated "
                                        "light, None or Automatic (consumed by the plotting/image batches)."),
    ("ContourShading", _v_contourshading, "option of ContourPlot / ListContourPlot: Automatic, None, True, False, "
                                          "a colour or a list of colours for the regions between contours "
                                          "(consumed by the plotting batch W20)."),
    ("TextureCoordinateFunction", _v_tcfun, "option of Plot3D and similar: Automatic or a function giving texture "
                                            "coordinates (consumed by the plotting batch W20)."),
    ("TextureCoordinateScaling", _v_bool("TextureCoordinateScaling"),
     "option of Plot3D and similar: whether texture-coordinate function arguments are scaled to [0, 1] "
     "(True/False; consumed by the plotting batch W20)."),
    ("TextureMapping", _v_texturemapping, "option of graphics primitives / Texture: the texture projection "
                                          "(Automatic, a name or a function)."),
    ("CalloutMarker", _v_calloutmarker, "option of Callout: marker at the anchor (None by default, Automatic, a "
                                        "marker name or a graphic); Callout validates it."),
    ("CalloutStyle", _v_directive("CalloutStyle"), "option of Callout: style of the leader and knockout strokes "
                                                   "(directives; numbers or strings raise TypeError)."),
    ("Dithering", _v_dithering, "option of ColorQuantize: True, False, Automatic or a method such as "
                                "\"FloydSteinberg\" (consumed by the image batches)."),
]
for _n, _v, _note in _OPTION_SPECS:
    OPTION_VALIDATORS[_n] = _v
    inert_option(_n, _note + " Inert as an expression. Check: reference (Wolfram documentation settings) in tests.",
                 area=AREA)


# ───────────────────────────── Inset, Raster, Raster3D ─────────────────────────────

_POS_SYMBOLS = ("Center", "Left", "Right", "Top", "Bottom", "Baseline", "Axis", "Automatic", "Above", "Below",
                "Before", "After")


def _position(p, fname, what="position"):
    if is_sym(p, *_POS_SYMBOLS):
        return
    if K.is_list(p) and p.args and all(is_sym(v, *_POS_SYMBOLS) or is_real_value(v) for v in p.args) \
            and not all(is_real_value(v) for v in p.args):
        if len(p.args) not in (2, 3):
            raise ValueError(f"{fname}: {what} must have 2 or 3 components")
        return
    coords(p, fname, what)


@fn("Inset", area=AREA, status="added",
    notes="Inset[obj], Inset[obj, pos], Inset[obj, pos, opos], Inset[obj, pos, opos, size] and "
          "Inset[obj, pos, opos, size, dirs] inert graphics object placing obj (graphic, string or any expression) "
          "at pos ({x, y}, {x, y, z}, Scaled/ImageScaled/Offset, Center, {Left, Top}, ...); size Automatic, w, "
          "{w, h}, Scaled[...] or Offset[...]. Malformed positions raise TypeError/ValueError. Inset[obj] means "
          "Inset[obj, Center]. Check: reference in tests.")
def _inset(obj: Expr, *rest: Expr) -> Expr:
    pos, opts = split_trailing_options(rest)
    nargs([obj] + pos, 1, 5, "Inset")
    if pos:
        _position(pos[0], "Inset")
    if len(pos) > 1:
        _position(pos[1], "Inset", "object position")
    if len(pos) > 2:
        s = pos[2]
        if not (is_sym(s, "Automatic") or (isinstance(s, Expr) and s.head in ("Scaled", "Offset", "ImageScaled"))):
            if K.is_list(s):
                for v in s.args:
                    if not is_sym(v, "Automatic"):
                        real_arg(v, "Inset", "size", 0)
            else:
                real_arg(s, "Inset", "size", 0)
    if len(pos) > 3 and not is_sym(pos[3], "Automatic"):
        coords(pos[3], "Inset", "direction", scaled_ok=False)
    raise Unevaluated()


def _numeric_array(a, fname, depth_options):
    """Dimensions of a rectangular numeric array (all leaves real); symbolic -> Unevaluated."""
    if isinstance(a, Expr) and a.head == "SparseArray":
        raise Unevaluated()
    if not K.is_list(a):
        if isinstance(a, (str, bool, int, float)):
            raise TypeError(f"{fname}: the data must be a rectangular array of numbers, not {K_input(a)}")
        raise Unevaluated()

    def dims(x):
        if K.is_list(x):
            if not x.args:
                return [0]
            ds = [dims(v) for v in x.args]
            if any(d != ds[0] for d in ds):
                raise ValueError(f"{fname}: the data array is not rectangular")
            return [len(x.args)] + ds[0]
        real_value(x, fname, "array value")
        return []
    d = dims(a)
    if len(d) not in depth_options:
        raise ValueError(f"{fname}: the data must be an array of depth {' or '.join(map(str, depth_options))}")
    return d


def _raster_rect(rect, fname, n):
    if is_sym(rect, "Automatic"):
        return
    if K.is_list(rect) and len(rect.args) == 2:
        a, b = rect.args
        da, db = coords(a, fname, "corner", dims=(n,)), coords(b, fname, "corner", dims=(n,))
        if da and db:
            lo, hi = [real_value(v, fname) for v in a.args], [real_value(v, fname) for v in b.args]
            if any(x >= y for x, y in zip(lo, hi)):
                raise ValueError(f"{fname}: the rectangle must have min < max in every direction")
        return
    raise ValueError(f"{fname}: the region must be {{min corner, max corner}} or Automatic, not {K_input(rect)}")


def _raster_range(r, fname):
    if is_sym(r, "Automatic"):
        return
    if K.is_list(r) and len(r.args) == 2:
        a, b = (real_value(v, fname, "range bound") for v in r.args)
        if not a < b:
            raise ValueError(f"{fname}: the value range {{amin, amax}} must have amin < amax")
        return
    raise ValueError(f"{fname}: the value range must be {{amin, amax}} or Automatic")


_RASTER_OPTS = {"ColorFunction": None, "ColorFunctionScaling": None, "ColorSpace": None, "Interleaving": None,
                "Method": None, "ImageResolution": None}
_RASTER3D_OPTS = dict(_RASTER_OPTS, OpacityFunction=_v_opacityfn, OpacityFunctionScaling=_v_bool("OpacityFunctionScaling"),
                      VertexDataCoordinates=_v_datacoords)


@fn("Raster", area=AREA, status="added",
    notes="Raster[array], Raster[array, {{xmin, ymin}, {xmax, ymax}}], Raster[array, rect, {amin, amax}] inert 2D "
          "primitive: an n x m array of gray values, or n x m x k cells with k = 2 (gray, opacity), 3 (RGB) or 4 "
          "(RGBA); Raster[array] occupies {{0, 0}, {m, n}} with the first row at the bottom; options ColorFunction, "
          "ColorFunctionScaling, ColorSpace. Non-numeric or ragged data raise TypeError/ValueError, a bad rectangle "
          "or range ValueError. Rendered by Rasterize. Check: reference and pixel read-back via Rasterize in tests.")
def _raster(a: Expr, *rest: Expr) -> Expr:
    pos, opts = split_trailing_options(rest)
    nargs(pos, 0, 2, "Raster")
    d = _numeric_array(a, "Raster", (2, 3))
    if len(d) == 3 and d[2] not in (2, 3, 4):
        raise ValueError("Raster: colour cells must have 2 (gray, opacity), 3 (RGB) or 4 (RGBA) components")
    if pos:
        _raster_rect(pos[0], "Raster", 2)
    if len(pos) > 1:
        _raster_range(pos[1], "Raster")
    check_options(opts, 2, "Raster", _RASTER_OPTS)
    raise Unevaluated()


@fn("Raster3D", area=AREA, status="added",
    notes="Raster3D[array], Raster3D[array, {{xmin, ymin, zmin}, {xmax, ymax, zmax}}], Raster3D[array, coords, "
          "{amin, amax}] inert 3D primitive for a cubical array of gray values or of cells with 2, 3 or 4 "
          "components; options ColorFunction, ColorFunctionScaling, OpacityFunction, OpacityFunctionScaling, "
          "VertexDataCoordinates (validated). Non-numeric or ragged data raise TypeError/ValueError. "
          "Check: reference in tests.")
def _raster3d(a: Expr, *rest: Expr) -> Expr:
    pos, opts = split_trailing_options(rest)
    nargs(pos, 0, 2, "Raster3D")
    d = _numeric_array(a, "Raster3D", (3, 4))
    if len(d) == 4 and d[3] not in (2, 3, 4):
        raise ValueError("Raster3D: colour cells must have 2, 3 or 4 components")
    if pos:
        _raster_rect(pos[0], "Raster3D", 3)
    if len(pos) > 1:
        _raster_range(pos[1], "Raster3D")
    check_options(opts, 3, "Raster3D", _RASTER3D_OPTS)
    raise Unevaluated()


register_primitive("Raster", (2,))
register_primitive("Raster3D", (3,))


# ───────────────────────────── geometric transforms ─────────────────────────────

_BBOX_SYMBOLS = ("Left", "Center", "Right", "Top", "Bottom")


def _fixed_point(p, fname):
    if K.is_list(p) and p.args and all(is_sym(v, *_BBOX_SYMBOLS) for v in p.args):
        return
    if is_sym(p, "Center"):
        return
    coords(p, fname, "fixed point")


@fn("Rotate", area=AREA, status="added",
    notes="Rotate[g, θ] (about the bounding-box centre), Rotate[g, θ, {x, y}] or {Left, Bottom}-style points, "
          "Rotate[g, {u, v}] (u -> v about the origin), Rotate[g, θ, w] (3D, axis w), Rotate[g, θ, w, p] inert "
          "wrapper for graphics primitives (or any expression outside graphics); θ in radians (θ Degree for "
          "degrees). Normal[...] rotates the coordinates of point-list primitives (Point, Line, Polygon, Arrow, "
          "...), disks and circles explicitly (exact for exact input). Non-numeric angles that are not symbolic "
          "raise TypeError, malformed points ValueError. Check: exact (Normal) and the identity of rotation by 2 Pi "
          "in tests.")
def _rotate(g: Expr, a: Real | ListOf[ListOf[Real]], *rest: ListOf[Real]) -> Expr:
    nargs(rest, 0, 2, "Rotate")
    if K.is_list(a):
        if len(a.args) != 2 or not all(K.is_list(v) for v in a.args):
            raise ValueError("Rotate: Rotate[g, {u, v}] needs two vectors u and v")
        for v in a.args:
            coords(v, "Rotate", "vector", scaled_ok=False)
        if rest:
            coords(rest[0], "Rotate", "fixed point", scaled_ok=False)
        raise Unevaluated()
    real_value(a, "Rotate", "angle")
    if len(rest) == 1:
        if K.is_list(rest[0]) and len(rest[0].args) == 3 and all(is_real_value(v) for v in rest[0].args):
            pass
        else:
            _fixed_point(rest[0], "Rotate")
    elif len(rest) == 2:
        coords(rest[0], "Rotate", "axis", dims=(3,), scaled_ok=False)
        coords(rest[1], "Rotate", "fixed point", dims=(3,), scaled_ok=False)
    raise Unevaluated()


@fn("Translate", area=AREA, status="added",
    notes="Translate[g, {x, y(, z)}] graphics g translated by a vector, Translate[g, {v1, v2, ...}] several copies "
          "(inert wrapper). Normal[...] adds the vector to the coordinates of the primitives (every copy for a list "
          "of vectors). Non-numeric vectors raise TypeError, ragged lists ValueError; symbolic stays unevaluated. "
          "Check: exact (Normal) in tests.")
def _translate(g: Expr, v: ListOf[Real] | MatrixOf[Real]) -> Expr:
    if K.is_list(v) and v.args and all(K.is_list(w) for w in v.args):
        ds = {coords(w, "Translate", "translation vector", scaled_ok=False) for w in v.args}
        if len(ds) > 1:
            raise ValueError("Translate: all translation vectors must have the same dimension")
    else:
        coords(v, "Translate", "translation vector", scaled_ok=False)
    raise Unevaluated()


@fn("Scale", area=AREA, status="added",
    notes="Scale[g, s] (about the bounding-box centre), Scale[g, {sx, sy(, sz)}], Scale[g, s, {x, y}] (fixed "
          "point, or {Left, Bottom}-style) inert wrapper. Normal[...] scales the coordinates of point-list primitives "
          "(and disks/circles for a uniform factor) explicitly. Non-numeric factors raise TypeError; symbolic stays "
          "unevaluated. Check: exact (Normal) in tests.")
def _scale(g: Expr, s: Real | ListOf[Real], *rest: ListOf[Real]) -> Expr:
    nargs(rest, 0, 1, "Scale")
    if K.is_list(s):
        coords(s, "Scale", "scale factors", scaled_ok=False)
    else:
        real_value(s, "Scale", "scale factor")
    if rest:
        _fixed_point(rest[0], "Scale")
    raise Unevaluated()



# ───────────────────────────── Show ─────────────────────────────

@fn("Show", area=AREA, status="added",
    notes="Show[g, opts] returns g with the options added (explicit options override those in g); Show[g1, g2, ...] "
          "and Show[{g1, g2, ...}, opts] combine graphics of the same dimension into Graphics[{prims1, prims2, "
          "...}, opts] (or Graphics3D), with the explicit options first and then the options of g1, g2, ... (a name "
          "set twice keeps its first value, which is the one Wolfram uses). Mixing 2D and 3D raises ValueError, "
          "non-graphics TypeError, GraphicsGrid/Row/Column and images NotImplementedError; symbolic arguments stay "
          "unevaluated. Check: exact (structural) and reference (Wolfram documentation rules) in tests.")
def _show(*args: Expr | Option) -> Expr:
    pos, opts = split_trailing_options(args)
    if not pos:
        raise ValueError("Show: expected at least one graphics object")
    gs = []
    for a in pos:
        items = list(a.args) if K.is_list(a) else [a]
        for g in items:
            if is_graphics(g):
                gs.append(g)
            elif isinstance(g, Expr) and g.head in ("GraphicsGrid", "GraphicsRow", "GraphicsColumn", "Image",
                                                    "Image3D", "Legended"):
                raise NotImplementedError(f"Show: combining {g.head} objects is not supported")
            elif isinstance(g, (str, bool, int, float)) or K.is_list(g) or is_real_value(g):
                raise TypeError(f"Show: {K_input(g)} is not a graphics object")
            else:
                raise Unevaluated()
    dims = {graphics_parts(g)[0] for g in gs}
    if len(dims) > 1:
        raise ValueError("Show: cannot combine two-dimensional and three-dimensional graphics")
    dim = dims.pop()
    head = "Graphics3D" if dim == 3 else "Graphics"
    check_options(opts, dim, "Show")
    merged, seen = [], set()
    for r in list(opts) + [r for g in gs for r in graphics_parts(g)[2]]:
        n = opt_name(r)
        if n is not None and n not in seen:
            seen.add(n)
            merged.append(r)
    if len(gs) == 1:
        content = graphics_parts(gs[0])[1]
    else:
        content = L(*[graphics_parts(g)[1] for g in gs])
    return Expr(head, content, *merged)


# ───────────────────────────── layouts ─────────────────────────────

_LAYOUT_OPTS = {n: None for n in ("Alignment", "AspectRatio", "Background", "BaselinePosition", "BaseStyle",
                                  "Dividers", "Frame", "FrameStyle", "ImageSize", "ItemAspectRatio", "Spacings",
                                  "ImagePadding", "ImageMargins", "PlotRangePadding", "Epilog", "Prolog",
                                  "ContentSelectable", "ItemSize", "AlignmentPoint", "PlotLabel", "LabelStyle")}


def _layout_items(x, fname):
    if not K.is_list(x):
        if isinstance(x, (str, bool, int, float)):
            raise TypeError(f"{fname}: expected a list of graphics, not {K_input(x)}")
        raise Unevaluated()
    return list(x.args)


@fn("GraphicsGrid", area=AREA, status="added",
    notes="GraphicsGrid[{{g11, g12, ...}, ...}, opts] inert layout of graphics in a 2D grid (rows may differ in "
          "length; SpanFromLeft / SpanFromAbove / SpanFromBoth for spans); options Alignment, AspectRatio, "
          "Background, Dividers, Frame, FrameStyle, ImageSize, Spacings, ... (unknown options raise ValueError). "
          "Deviation: Wolfram expands the layout into a Graphics of Insets; here it stays a GraphicsGrid object. "
          "A non-matrix first argument raises TypeError/ValueError. Check: reference in tests.")
def _graphicsgrid(rows: MatrixOf[Expr], *opts: Option) -> Expr:
    for r in _layout_items(rows, "GraphicsGrid"):
        if not K.is_list(r):
            raise ValueError(f"GraphicsGrid: each row must be a list, not {K_input(r)}")
    check_options(opts, 2, "GraphicsGrid", _LAYOUT_OPTS)
    raise Unevaluated()


@fn("GraphicsRow", area=AREA, status="added",
    notes="GraphicsRow[{g1, g2, ...}], GraphicsRow[list, spacing] and options (as GraphicsGrid) inert layout of "
          "graphics in a row (Wolfram expands it into a Graphics of Insets; here it stays a GraphicsRow object). "
          "A non-list raises TypeError, a negative spacing ValueError. Check: reference in tests.")
def _graphicsrow(items: ListOf[Expr], *rest: Expr) -> Expr:
    pos, opts = split_trailing_options(rest)
    _layout_items(items, "GraphicsRow")
    nargs(pos, 0, 1, "GraphicsRow")
    if pos and not (is_sym(pos[0], "Automatic") or isinstance(pos[0], Expr) and pos[0].head in ("Scaled", "Offset")):
        real_arg(pos[0], "GraphicsRow", "spacing", 0)
    check_options(opts, 2, "GraphicsRow", _LAYOUT_OPTS)
    raise Unevaluated()


@fn("GraphicsColumn", area=AREA, status="added",
    notes="GraphicsColumn[{g1, g2, ...}], GraphicsColumn[list, alignment], GraphicsColumn[list, alignment, spacing] "
          "and options inert layout of graphics in a column (alignment Left, Center, Right or Automatic; Wolfram "
          "expands it into Graphics with Insets; here it stays a GraphicsColumn object). Bad alignments or "
          "spacings raise ValueError. Check: reference in tests.")
def _graphicscolumn(items: ListOf[Expr], *rest: Expr) -> Expr:
    pos, opts = split_trailing_options(rest)
    _layout_items(items, "GraphicsColumn")
    nargs(pos, 0, 2, "GraphicsColumn")
    if pos and not is_sym(pos[0], "Left", "Center", "Right", "Automatic"):
        raise ValueError(f"GraphicsColumn: alignment must be Left, Center, Right or Automatic, not {K_input(pos[0])}")
    if len(pos) > 1 and not is_sym(pos[1], "Automatic"):
        real_arg(pos[1], "GraphicsColumn", "spacing", 0)
    check_options(opts, 2, "GraphicsColumn", _LAYOUT_OPTS)
    raise Unevaluated()


# ───────────────────────────── annotation wrappers ─────────────────────────────

@fn("Annotation", area=AREA, status="added",
    notes="Annotation[expr, data] and Annotation[expr, data, \"type\"] inert wrapper attaching data to expr (e.g. "
          "\"Mouse\" or \"Hyperlink\" types; displays as expr); Annotation[items, key -> value] for Graph and mesh "
          "properties. A non-string type raises TypeError. Check: reference in tests.")
def _annotation(expr: Expr, data: Expr, *rest: String) -> Expr:
    nargs(rest, 0, 1, "Annotation")
    if rest and not isinstance(rest[0], str):
        if isinstance(rest[0], (bool, int, float)) or K.is_list(rest[0]):
            raise TypeError(f"Annotation: the annotation type must be a string, not {K_input(rest[0])}")
        raise Unevaluated()
    raise Unevaluated()


_CALLOUT_OPTS = {n: None for n in ("Appearance", "Background", "Frame", "FrameMargins", "LabelStyle",
                                   "LabelVisibility", "LeaderSize", "RoundingRadius", "FrameStyle")}
_CALLOUT_OPTS["CalloutMarker"] = _v_calloutmarker
_CALLOUT_OPTS["CalloutStyle"] = _v_directive("CalloutStyle")


@fn("Callout", area=AREA, status="added",
    notes="Callout[data, expr], Callout[data, expr, pos], Callout[data, expr, pos, apos] and Callout[expr] inert "
          "wrapper labelling data with a callout in visualisation functions (the plotting batches draw it); "
          "options Appearance, Background, CalloutMarker, CalloutStyle, Frame, FrameMargins, LabelStyle, "
          "LabelVisibility, LeaderSize, RoundingRadius are validated (unknown options raise ValueError). "
          "Check: reference in tests.")
def _callout(*args: Expr) -> Expr:
    pos, opts = split_trailing_options(args)
    nargs(pos, 1, 4, "Callout")
    check_options(opts, 2, "Callout", _CALLOUT_OPTS)
    raise Unevaluated()


@fn("Legended", area=AREA, status="added",
    notes="Legended[expr, leg] inert wrapper displaying expr with legend leg (leg may be any expression or "
          "Placed[leg, pos]); used by the plotting functions to attach legends. Exactly two arguments. "
          "Check: reference in tests.")
def _legended(expr: Expr, leg: Expr) -> Expr:
    raise Unevaluated()


@fn("Highlighted", area=AREA, status="added",
    notes="Highlighted[expr] and Highlighted[expr, effect] (Automatic, None, \"name\", Placed[effect, pos], a list "
          "of effects) inert wrapper; options are accepted as rules (other non-rule extra arguments raise "
          "ValueError). Check: reference in tests.")
def _highlighted(expr: Expr, *rest: Expr) -> Expr:
    pos, opts = split_trailing_options(rest)
    nargs(pos, 0, 1, "Highlighted")
    if pos and isinstance(pos[0], (bool, int, float)):
        raise TypeError(f"Highlighted: {K_input(pos[0])} is not a highlighting effect")
    raise Unevaluated()


# ───────────────────────────── Normal: explicit coordinates ─────────────────────────────

class _Unresolved(Exception):
    """A transform cannot be applied explicitly (Normal keeps the wrapper)."""


def _S(x):
    return K.to_sp(x)


def _is_point(x, dim=None):
    return K.is_list(x) and len(x.args) in ((2, 3) if dim is None else (dim,)) and all(is_real_value(c) for c in x.args)


def _map_points(x, f):
    """Apply f to every numeric point in a nested coordinate structure."""
    if _is_point(x):
        return f(x)
    if K.is_list(x) and x.args:
        return L(*[_map_points(v, f) for v in x.args])
    raise _Unresolved()


def _collect_points(x, out):
    if _is_point(x):
        out.append(x)
    elif K.is_list(x) and x.args:
        for v in x.args:
            _collect_points(v, out)
    else:
        raise _Unresolved()


def _bbox_points(g):
    """Corner-ish points determining the bounding box of g (for the default centre of Rotate/Scale)."""
    pts = []
    if K.is_list(g):
        for v in g.args:
            pts += _bbox_points(v)
        return pts
    if not isinstance(g, Expr) or g.head in DIRECTIVE_HEADS or is_rule(g):
        return pts
    h = g.head
    if h in _POINT_HEADS and g.args:
        _collect_points(g.args[0], pts)
        return pts
    if h in ("Disk", "Circle") and g.args:
        c = g.args[0]
        r = g.args[1] if len(g.args) > 1 else 1
        if not _is_point(c, 2):
            raise _Unresolved()
        rx, ry = (r.args if K.is_list(r) else (r, r))
        cx, cy = c.args
        pts += [L(Expr("Plus", cx, Expr("Times", -1, rx)), Expr("Plus", cy, Expr("Times", -1, ry))),
                L(Expr("Plus", cx, rx), Expr("Plus", cy, ry))]
        return pts
    if h == "Rectangle" and g.args:
        a = g.args[0]
        b = g.args[1] if len(g.args) > 1 else _map_points(a, lambda p: L(*[_ev_plus(c, 1) for c in p.args]))
        _collect_points(L(a, b), pts)
        return pts
    raise _Unresolved()


def _ev_plus(a, b):
    from .wl_common import ev
    return ev(Expr("Plus", a, b))


def _center_of(g):
    pts = _bbox_points(g)
    if not pts:
        raise _Unresolved()
    from .wl_common import ev
    dim = len(pts[0].args)
    pts = [[ev(c) for c in p.args] for p in pts]
    lo = [min((_S(p[i]) for p in pts), key=lambda v: float(v)) for i in range(dim)]
    hi = [max((_S(p[i]) for p in pts), key=lambda v: float(v)) for i in range(dim)]
    return [(a + b) / 2 for a, b in zip(lo, hi)]


def _apply_affine(g, mat, shift, kind):
    """Transform the coordinates of g by x -> mat.x + shift (sympy Matrix / list). kind: 'translate', 'rotate'
    (orthogonal), 'scale' (diagonal)."""
    def f(p):
        v = sp.Matrix([_S(c) for c in p.args])
        if v.rows != mat.rows:
            raise _Unresolved()
        w = mat * v + sp.Matrix(shift)
        return L(*[R(sp.nsimplify(c) if False else sp.simplify(c)) for c in w])

    if K.is_list(g):
        return L(*[_apply_affine(v, mat, shift, kind) for v in g.args])
    if g is None or isinstance(g, Symbol) or not isinstance(g, Expr):
        return g
    if g.head in DIRECTIVE_HEADS or is_rule(g):
        return g
    h = g.head
    if h in _POINT_HEADS and g.args:
        return Expr(h, _map_points(g.args[0], f), *g.args[1:])
    if h in ("Disk", "Circle") and g.args and _is_point(g.args[0], 2):
        c = f(g.args[0])
        r = g.args[1] if len(g.args) > 1 else 1
        if kind == "translate":
            return Expr(h, c, *g.args[1:]) if len(g.args) > 1 else Expr(h, c)
        if kind == "rotate":
            if K.is_list(r):
                raise _Unresolved()
            if len(g.args) > 2:
                ang = sp.atan2(mat[1, 0], mat[0, 0])
                a = g.args[2]
                if not K.is_list(a) or len(a.args) != 2:
                    raise _Unresolved()
                return Expr(h, c, r, L(*[R(sp.simplify(_S(t) + ang)) for t in a.args]))
            return Expr(h, c, r)
        sx, sy = mat[0, 0], mat[1, 1]
        if len(g.args) > 2:
            raise _Unresolved()
        if K.is_list(r):
            rx, ry = (_S(v) for v in r.args)
        else:
            rx = ry = _S(r)
        nx, ny = sp.simplify(abs(sx) * rx), sp.simplify(abs(sy) * ry)
        if sp.simplify(nx - ny) == 0 and not K.is_list(r):
            return Expr(h, c, R(nx))
        return Expr(h, c, L(R(nx), R(ny)))
    if h == "Rectangle" and g.args and kind in ("translate", "scale"):
        a = g.args[0]
        b = g.args[1] if len(g.args) > 1 else L(*[_ev_plus(c, 1) for c in a.args])
        if not (_is_point(a, 2) and _is_point(b, 2)):
            raise _Unresolved()
        pa, pb = f(a), f(b)
        lo = L(*[R(sp.Min(_S(u), _S(v))) for u, v in zip(pa.args, pb.args)])
        hi = L(*[R(sp.Max(_S(u), _S(v))) for u, v in zip(pa.args, pb.args)])
        return Expr("Rectangle", lo, hi, *g.args[2:])
    if h == "Text" and len(g.args) >= 2 and _is_point(g.args[1]) and kind == "translate":
        return Expr("Text", g.args[0], f(g.args[1]), *g.args[2:])
    if h in ("GraphicsGroup", "Style", "Annotation", "Tooltip") and g.args:
        return Expr(h, _apply_affine(g.args[0], mat, shift, kind), *g.args[1:])
    raise _Unresolved()


def _rotation_matrix(theta, axis=None):
    t = _S(theta)
    if axis is None:
        return sp.Matrix([[sp.cos(t), -sp.sin(t)], [sp.sin(t), sp.cos(t)]])
    w = sp.Matrix([_S(c) for c in axis.args])
    n = sp.sqrt(w.dot(w))
    if n == 0:
        raise ValueError("Rotate: the rotation axis must not be the zero vector")
    ux, uy, uz = [c / n for c in w]
    c, s = sp.cos(t), sp.sin(t)
    return sp.Matrix([[c + ux ** 2 * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
                      [uy * ux * (1 - c) + uz * s, c + uy ** 2 * (1 - c), uy * uz * (1 - c) - ux * s],
                      [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz ** 2 * (1 - c)]])


def _fixed(g, p):
    if p is None or is_sym(p, "Center") or (K.is_list(p) and all(is_sym(v, "Center") for v in p.args)):
        return _center_of(g)
    if K.is_list(p) and all(is_sym(v, *_BBOX_SYMBOLS) for v in p.args):
        pts = _bbox_points(g)
        if not pts:
            raise _Unresolved()
        out = []
        for i, v in enumerate(p.args):
            vals = sorted((_S(q.args[i]) for q in pts), key=float)
            out.append({"Left": vals[0], "Bottom": vals[0], "Right": vals[-1], "Top": vals[-1]}.get(
                v.name, (vals[0] + vals[-1]) / 2))
        return out
    if _is_point(p):
        return [_S(c) for c in p.args]
    raise _Unresolved()


def _about(mat, c):
    """shift for x -> mat (x - c) + c."""
    cv = sp.Matrix(c)
    return list(cv - mat * cv)


def normal_graphics(x):
    """Normal of graphics: GraphicsComplex substituted, Translate/Rotate/Scale applied where possible."""
    if K.is_list(x):
        return L(*[normal_graphics(v) for v in x.args])
    if not isinstance(x, Expr) or not isinstance(x.head, str):
        return x
    h = x.head
    if h in ("Graphics", "Graphics3D") and x.args:
        return Expr(h, normal_graphics(x.args[0]), *x.args[1:])
    if h == "GraphicsComplex" and len(x.args) >= 2:
        return normal_graphics(_gc_substitute(x))
    if h == "Translate" and len(x.args) == 2:
        g, v = normal_graphics(x.args[0]), x.args[1]
        try:
            if K.is_list(v) and v.args and all(K.is_list(w) for w in v.args):
                return L(*[_apply_affine(g, sp.eye(len(w.args)), [_S(c) for c in w.args], "translate")
                           for w in v.args])
            return _apply_affine(g, sp.eye(len(v.args)), [_S(c) for c in v.args], "translate")
        except (_Unresolved, Unevaluated, TypeError, ValueError, AttributeError):
            return Expr(h, g, *x.args[1:])
    if h == "Rotate" and len(x.args) >= 2:
        g = normal_graphics(x.args[0])
        a, rest = x.args[1], list(x.args[2:])
        try:
            if K.is_list(a):
                u, v = ([_S(c) for c in w.args] for w in a.args)
                if len(u) != 2:
                    raise _Unresolved()
                theta = sp.atan2(v[1], v[0]) - sp.atan2(u[1], u[0])
                mat = _rotation_matrix(theta)
                c = [_S(t) for t in rest[0].args] if rest else [0, 0]
            elif len(rest) == 2 or (len(rest) == 1 and _is_point(rest[0], 3) and not _is_2d(g)):
                mat = _rotation_matrix(a, rest[0])
                c = [_S(t) for t in rest[1].args] if len(rest) == 2 else [0, 0, 0]
            else:
                mat = _rotation_matrix(a)
                c = _fixed(g, rest[0] if rest else None)
            return _apply_affine(g, mat, _about(mat, c), "rotate")
        except (_Unresolved, Unevaluated, TypeError, ValueError, AttributeError):
            return Expr(h, g, *x.args[1:])
    if h == "Scale" and len(x.args) >= 2:
        g, s = normal_graphics(x.args[0]), x.args[1]
        try:
            c = _fixed(g, x.args[2] if len(x.args) > 2 else None)
            fs = [_S(v) for v in s.args] if K.is_list(s) else [_S(s)] * len(c)
            if len(fs) != len(c):
                raise _Unresolved()
            mat = sp.diag(*fs)
            return _apply_affine(g, mat, _about(mat, c), "scale")
        except (_Unresolved, Unevaluated, TypeError, ValueError, AttributeError):
            return Expr(h, g, *x.args[1:])
    if h in ("GraphicsGroup",) and x.args:
        return Expr(h, normal_graphics(x.args[0]), *x.args[1:])
    return x


def _is_2d(g):
    try:
        pts = _bbox_points(g)
    except _Unresolved:
        return True
    return bool(pts) and len(pts[0].args) == 2


def _gc_substitute(gc):
    pts, data = gc.args[0], gc.args[1]
    vopts = options_dict(gc.args[2:])
    P = list(pts.args)

    def sub_coords(c):
        if K.is_int(c) and not isinstance(c, bool):
            return P[int(c) - 1]
        if K.is_list(c):
            return L(*[sub_coords(v) for v in c.args])
        return c

    def vertex_opts(idx_struct):
        out = []
        for name in ("VertexColors", "VertexNormals", "VertexTextureCoordinates"):
            v = vopts.get(name)
            if K.is_list(v):
                vals = list(v.args)

                def pick(c):
                    if K.is_int(c) and not isinstance(c, bool):
                        return vals[int(c) - 1]
                    return L(*[pick(u) for u in c.args])
                out.append(Expr("Rule", Symbol(name), pick(idx_struct)))
        return out

    def walk(x):
        if K.is_list(x):
            return L(*[walk(v) for v in x.args])
        if not isinstance(x, Expr) or not isinstance(x.head, str) or x.head in DIRECTIVE_HEADS or is_rule(x):
            return x
        h = x.head
        if h in _POINT_HEADS and x.args:
            first = x.args[0]
            extra = vertex_opts(first) if _int_leaves(first) else []
            return Expr(h, sub_coords(first), *x.args[1:], *extra)
        if h in _CENTER_HEADS and x.args:
            k = 1 if h in ("Text", "Inset") else 0
            n = 2 if h in ("Rectangle", "Cuboid") else 1
            new = list(x.args)
            for i in range(k, min(k + n, len(new))):
                new[i] = sub_coords(new[i])
            return Expr(h, *new)
        if h in ("GraphicsGroup", "Style", "Annotation", "Tooltip", "Translate", "Rotate", "Scale") and x.args:
            return Expr(h, walk(x.args[0]), *x.args[1:])
        return x
    return walk(data)


def _contains_graphics_forms(x, depth=0):
    if depth > 50:
        return False
    if isinstance(x, Expr) and isinstance(x.head, str):
        if x.head in ("GraphicsComplex", "Translate", "Rotate", "Scale"):
            return True
        if x.head in ("Graphics", "Graphics3D", "List", "GraphicsGroup"):
            return any(_contains_graphics_forms(a, depth + 1) for a in x.args)
    return False


_prev_normal = K._BUILTINS.get("Normal")


def _normal_w14(a, ev_, expr):
    if len(a) == 1 and _contains_graphics_forms(a[0]):
        return normal_graphics(a[0])
    return _prev_normal(a, ev_, expr)


K._BUILTINS["Normal"] = _normal_w14
