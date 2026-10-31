"""Wolfram batch W14 (part 3 of 3): drawing 2D graphics (matplotlib, optional) and Rasterize.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). matplotlib and numpy are optional: without them Rasterize raises NotImplementedError and the
rest of the graphics layer still works.

Python API for later batches:

* ``render_figure(g, image_size=None, raster_size=None, background=None)`` -> a matplotlib ``Figure`` of a
  ``Graphics`` object (Agg canvas, nothing shown);
* ``render_png(g, ...)`` -> PNG bytes, ``render_svg(g, ...)`` -> SVG text (for Export, W13);
* ``raster_array(g, ...)`` -> uint8 numpy array (height x width x 3/4, top row first);
* ``RENDER_PRIMITIVES[head] = f(ctx, expr)`` draws a primitive (W19 can add BezierCurve, ...); ``ctx`` gives
  ``ctx.ax``, ``ctx.state`` (current style), ``ctx.xy(points)`` (transformed numpy coordinates), ``ctx.extend``
  (bounds), ``ctx.line_width()``, ``ctx.face()``, ``ctx.edge()``;
* ``RENDER_OPTIONS[name] = f(ctx, value)`` supports a Graphics option when drawing (W15 adds Axes, Frame, Ticks,
  ...). An option without a handler whose value is not one of the no-effect defaults raises NotImplementedError,
  so a picture is never drawn silently without a requested element;
* ``IMAGE_FACTORY``: set by the image batch (W21) to turn an RGB(A) uint8 array into an ``Image`` expression, which
  ``Rasterize[g]`` / ``Rasterize[g, "Image"]`` then return.

Sizes follow Wolfram's conventions where they are documented (ImageSize in printer's points, default 360 wide;
Thickness and PointSize as fractions of the width; PlotRangePadding Scaled[0.02]); the symbolic sizes Tiny ... Large
and default line widths approximate the notebook front end, so pixels are not identical to Wolfram's renderer.
"""
from __future__ import annotations

import math

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L
from .wl_graphics_color import parse_color, to_rgba, real_value, is_real_value, K_input
from . import wl_graphics_core as G

AREA = "graphics: rendering (W14)"

RENDER_PRIMITIVES = {}
RENDER_OPTIONS = {}
IMAGE_FACTORY = None

_IMAGE_SIZES = {"Tiny": 100, "Small": 180, "Medium": 360, "Large": 576}
_THICK = {"Tiny": 0.25, "Small": 0.5, "Medium": 1.0, "Large": 2.0}
_POINT = {"Tiny": 2.0, "Small": 3.0, "Medium": 4.5, "Large": 6.5}
_DASH = {"Tiny": 1.0, "Small": 2.0, "Medium": 4.0, "Large": 8.0}


def _need():
    try:
        import numpy  # noqa: F401
        import matplotlib
        matplotlib.use("Agg", force=False)
    except ImportError as exc:
        raise NotImplementedError(f"Rasterize: drawing graphics requires matplotlib and numpy "
                                  f"(pip install mikoshilang[plotting]): {exc}")


def _f(x, what="coordinate"):
    return real_value(x, "Rasterize", what)


class State:
    def __init__(self):
        self.color = (0.0, 0.0, 0.0, 1.0)
        self.opacity = None
        self.thickness = ("abs", 1.0)
        self.pointsize = ("rel", 0.008)
        self.dashing = None
        self.face = None            # None: follow colour; False: no faces; rgba
        self.edge = False           # False: no edges; dict(color, thickness, dashing)
        self.cap = "projecting"
        self.join = "miter"
        self.arrow = ("rel", 0.04)
        self.font = {}

    def copy(self):
        s = State.__new__(State)
        s.__dict__.update(self.__dict__)
        s.edge = dict(self.edge) if isinstance(self.edge, dict) else self.edge
        s.font = dict(self.font)
        return s


class Ctx:
    def __init__(self, ax, width_pts):
        import numpy as np
        self.np = np
        self.ax = ax
        self.width = width_pts
        self.state = State()
        self.T = np.eye(3)
        self.bounds = [math.inf, -math.inf, math.inf, -math.inf]
        self.draw = []              # deferred drawing callables (need final bounds for relative sizes)

    # coordinates
    def xy(self, pts):
        np = self.np
        a = np.asarray(pts, dtype=float).reshape(-1, 2)
        h = np.hstack([a, np.ones((len(a), 1))]) @ self.T.T
        return h[:, :2]

    def extend(self, xy):
        if len(xy):
            self.bounds[0] = min(self.bounds[0], float(xy[:, 0].min()))
            self.bounds[1] = max(self.bounds[1], float(xy[:, 0].max()))
            self.bounds[2] = min(self.bounds[2], float(xy[:, 1].min()))
            self.bounds[3] = max(self.bounds[3], float(xy[:, 1].max()))

    def _rgba(self, c):
        r, g, b, a = c
        if self.state.opacity is not None:
            a = self.state.opacity
        return (r, g, b, a)

    def line_width(self, spec=None):
        kind, v = spec or self.state.thickness
        return v * self.width if kind == "rel" else v

    def color(self):
        return self._rgba(self.state.color)

    def face(self):
        f = self.state.face
        if f is False:
            return "none"
        return self._rgba(self.state.color if f is None else f)

    def edge(self):
        e = self.state.edge
        if e is False:
            return None
        return e


def _points(x, fname):
    """Nested coordinate structure -> list of point lists (each a list of (x, y))."""
    def pt(p):
        if not (K.is_list(p) and len(p.args) == 2):
            raise NotImplementedError(f"Rasterize: {fname} coordinates {K_input(p)} are not 2D numeric points "
                                      "(Scaled/Offset coordinates are not supported yet)")
        return (_f(p.args[0]), _f(p.args[1]))
    if K.is_list(x) and x.args and K.is_list(x.args[0]) and x.args[0].args and K.is_list(x.args[0].args[0]):
        return [[pt(p) for p in seg.args] for seg in x.args]
    if K.is_list(x) and x.args and K.is_list(x.args[0]):
        return [[pt(p) for p in x.args]]
    if K.is_list(x) and len(x.args) == 2:
        return [[pt(x)]]
    raise NotImplementedError(f"Rasterize: cannot read the coordinates of {fname}: {K_input(x)}")


_ALPHA_ARITY = {"RGBColor": 4, "GrayLevel": 2, "Hue": 4, "CMYKColor": 5, "XYZColor": 4, "LABColor": 4,
                "LCHColor": 4, "LUVColor": 4}


def _apply_directive(ctx, d):
    s = ctx.state
    rgba = to_rgba(d) if isinstance(d, Expr) and d.head != "Opacity" else None
    if rgba is not None:
        s.color = rgba
        # a colour with its own opacity channel replaces Opacity[..]; a plain colour keeps it (WL semantics)
        if len(d.args) == _ALPHA_ARITY.get(d.head, -1):
            s.opacity = None
        return True
    h = d.head
    a = d.args
    if h == "Opacity":
        if len(a) == 2:
            c = to_rgba(a[1])
            s.color = (c[0], c[1], c[2], 1.0)
        s.opacity = min(max(_f(a[0], "opacity"), 0.0), 1.0)
        return True
    if h in ("Thickness", "AbsoluteThickness"):
        v = a[0]
        if isinstance(v, Symbol):
            s.thickness = ("abs", _THICK[v.name])
        else:
            s.thickness = ("rel" if h == "Thickness" else "abs", _f(v, "thickness"))
        return True
    if h in ("PointSize", "AbsolutePointSize"):
        v = a[0]
        if isinstance(v, Symbol):
            s.pointsize = ("abs", _POINT[v.name])
        else:
            s.pointsize = ("rel" if h == "PointSize" else "abs", _f(v, "point size"))
        return True
    if h in ("Dashing", "AbsoluteDashing"):
        spec = a[0]
        vals = list(spec.args) if K.is_list(spec) else [spec, spec]
        out = []
        for v in vals:
            if isinstance(v, Symbol):
                out.append(_DASH[v.name])
            else:
                x = _f(v, "dash length")
                out.append(x * ctx.width if h == "Dashing" else x)
        if len(a) > 1 and _f(a[1], "offset") != 0:
            raise NotImplementedError("Rasterize: dash offsets are not supported yet")
        s.dashing = None if not out else (0, [max(v, 0.6) if v == 0 else v for v in out])
        return True
    if h == "CapForm":
        s.cap = {"Butt": "butt", "Round": "round", "Square": "projecting"}.get(a[0], "butt")
        return True
    if h == "JoinForm":
        j = a[0]
        s.join = {"Bevel": "bevel", "Round": "round", "Miter": "miter"}.get(j if isinstance(j, str) else "Miter",
                                                                            "bevel")
        return True
    if h == "Arrowheads":
        v = a[0]
        if isinstance(v, Symbol) and v.name in _THICK:
            s.arrow = ("abs", {"Tiny": 5.0, "Small": 8.0, "Medium": 12.0, "Large": 18.0}[v.name])
        elif is_real_value(v):
            s.arrow = ("rel", _f(v))
        elif not (isinstance(v, Symbol) and v.name == "Automatic"):
            raise NotImplementedError("Rasterize: only Arrowheads[s] and Arrowheads[Automatic] are drawn so far")
        return True
    if h == "EdgeForm":
        if not a or (isinstance(a[0], Symbol) and a[0].name == "None"):
            s.edge = False
            return True
        sub = ctx.state.copy()
        sub.color, sub.opacity, sub.thickness, sub.dashing = (0, 0, 0, 1.0), None, ("abs", 1.0), None
        saved = ctx.state
        ctx.state = sub
        _apply_style(ctx, a[0])
        ctx.state = saved
        s.edge = {"color": ctx._rgba(sub.color) if sub.opacity is None else sub.color[:3] + (sub.opacity,),
                  "width": sub.thickness, "dashing": sub.dashing}
        return True
    if h == "FaceForm":
        if not a or (isinstance(a[0], Symbol) and a[0].name == "None"):
            s.face = False
            return True
        sub = ctx.state.copy()
        saved = ctx.state
        ctx.state = sub
        _apply_style(ctx, a[0])
        ctx.state = saved
        s.face = sub.color[:3] + ((sub.opacity if sub.opacity is not None else sub.color[3]),)
        return True
    if h == "Directive":
        for x in a:
            _apply_style(ctx, x)
        return True
    if h in ("Specularity", "Glow", "AmbientLight", "DirectionalLight", "PointLight", "SpotLight", "FlatShading",
             "GouraudShading", "PhongShading", "GoochShading", "ToonShading", "HatchShading", "StippleShading",
             "HalftoneShading", "MaterialShading", "Texture"):
        return True                 # 3D surface directives have no effect in 2D graphics (as in Wolfram)
    if h in ("LinearGradientFilling", "RadialGradientFilling", "ConicGradientFilling", "HatchFilling",
             "PatternFilling", "DropShadowing", "Haloing", "Blurring"):
        raise NotImplementedError(f"Rasterize: drawing with {h} is not supported yet")
    return False


def _apply_style(ctx, x):
    if K.is_list(x):
        for i in x.args:
            _apply_style(ctx, i)
        return
    if isinstance(x, Expr) and _apply_directive(ctx, x):
        return
    if isinstance(x, Symbol) and x.name in ("None", "Automatic", "Inherited"):
        return
    if isinstance(x, str) or (G.is_rule(x)):
        return
    if isinstance(x, (int, float)):
        ctx.state.font["size"] = float(x)
        return
    raise NotImplementedError(f"Rasterize: {K_input(x)} is not a supported style directive")


def _mpl_dash(d):
    return None if d is None else d


def _draw_lines(ctx, segs, closed=False):
    from matplotlib.lines import Line2D
    st = ctx.state.copy()
    for seg in segs:
        xy = ctx.xy(seg)
        ctx.extend(xy)

        def go(xy=xy, st=st):
            ln = Line2D(xy[:, 0], xy[:, 1], color=_rgba_of(st), linewidth=_lw(ctx, st.thickness),
                        solid_capstyle=st.cap, dash_capstyle="butt", solid_joinstyle=st.join)
            if st.dashing:
                ln.set_dashes([max(v, 0.01) for v in st.dashing[1]])
            ctx.ax.add_line(ln)
        ctx.draw.append(go)


def _rgba_of(st):
    r, g, b, a = st.color
    return (r, g, b, st.opacity if st.opacity is not None else a)


def _lw(ctx, spec):
    kind, v = spec
    return v * ctx.width if kind == "rel" else v


def _draw_polys(ctx, polys):
    from matplotlib.patches import Polygon as MPoly
    st = ctx.state.copy()
    for poly in polys:
        xy = ctx.xy(poly)
        ctx.extend(xy)

        def go(xy=xy, st=st):
            f = "none" if st.face is False else (_rgba_of(st) if st.face is None else st.face)
            e = st.edge
            kw = dict(facecolor=f, closed=True)
            if e:
                kw.update(edgecolor=e["color"], linewidth=_lw(ctx, e["width"]), joinstyle=st.join)
                if e["dashing"]:
                    kw["linestyle"] = (0, e["dashing"][1])
            else:
                kw.update(edgecolor="none", linewidth=0)
            ctx.ax.add_patch(MPoly(xy, **kw))
        ctx.draw.append(go)


def _ellipse_pts(c, rx, ry, t0=0.0, t1=2 * math.pi, sector=False, n=256):
    pts = [(c[0] + rx * math.cos(t0 + (t1 - t0) * k / n), c[1] + ry * math.sin(t0 + (t1 - t0) * k / n))
           for k in range(n + 1)]
    if sector:
        pts = [tuple(c)] + pts
    return pts


def _disk_parts(e):
    a = e.args
    c = (0.0, 0.0) if not a else _points(a[0], e.head)[0][0]
    if len(a) > 1:
        r = a[1]
        rx, ry = (_f(r.args[0], "radius"), _f(r.args[1], "radius")) if K.is_list(r) else (_f(r, "radius"),) * 2
    else:
        rx = ry = 1.0
    t0, t1 = 0.0, 2 * math.pi
    if len(a) > 2:
        if not (K.is_list(a[2]) and len(a[2].args) == 2):
            raise NotImplementedError(f"Rasterize: {e.head} angle range {K_input(a[2])} is not supported")
        t0, t1 = _f(a[2].args[0], "angle"), _f(a[2].args[1], "angle")
    if len(a) > 3:
        raise NotImplementedError(f"Rasterize: {K_input(e)} is not supported")
    return c, rx, ry, t0, t1, len(a) > 2


def _r_point(ctx, e):
    st = ctx.state.copy()
    pts = [p for seg in _points(e.args[0], "Point") for p in seg]
    xy = ctx.xy(pts)
    ctx.extend(xy)

    def go(xy=xy, st=st):
        d = _lw(ctx, st.pointsize)
        ctx.ax.scatter(xy[:, 0], xy[:, 1], s=d * d, c=[_rgba_of(st)], marker="o", linewidths=0)
    ctx.draw.append(go)


def _r_line(ctx, e):
    _draw_lines(ctx, _points(e.args[0], "Line"))


def _r_polygon(ctx, e):
    if len(e.args) > 1:
        raise NotImplementedError("Rasterize: Polygon options (VertexColors, ...) are not drawn yet")
    _draw_polys(ctx, _points(e.args[0], "Polygon"))


def _r_rectangle(ctx, e):
    a = e.args
    p = (0.0, 0.0) if not a else _points(a[0], "Rectangle")[0][0]
    q = (p[0] + 1, p[1] + 1) if len(a) < 2 else _points(a[1], "Rectangle")[0][0]
    if len(a) > 2:
        raise NotImplementedError("Rasterize: Rectangle options (RoundingRadius) are not drawn yet")
    _draw_polys(ctx, [[(p[0], p[1]), (q[0], p[1]), (q[0], q[1]), (p[0], q[1])]])


def _r_disk(ctx, e):
    c, rx, ry, t0, t1, sector = _disk_parts(e)
    _draw_polys(ctx, [_ellipse_pts(c, rx, ry, t0, t1, sector)])


def _r_circle(ctx, e):
    c, rx, ry, t0, t1, arc = _disk_parts(e)
    _draw_lines(ctx, [_ellipse_pts(c, rx, ry, t0, t1)])


def _text_of(x):
    if isinstance(x, str):
        return x
    try:
        from .wl_forms import output_text
        return output_text(x)
    except Exception:
        return K_input(x)


def _r_text(ctx, e):
    a = e.args
    pos = (0.0, 0.0) if len(a) < 2 else _points(a[1], "Text")[0][0]
    ha, va = "center", "center"
    if len(a) > 2:
        off = a[2]
        if not (K.is_list(off) and len(off.args) == 2):
            raise NotImplementedError("Rasterize: this Text offset is not supported")
        ox, oy = _f(off.args[0]), _f(off.args[1])
        ha = {-1.0: "left", 0.0: "center", 1.0: "right"}.get(ox)
        va = {-1.0: "bottom", 0.0: "center", 1.0: "top"}.get(oy)
        if ha is None or va is None:
            raise NotImplementedError("Rasterize: Text offsets other than -1, 0, 1 are not supported yet")
    if len(a) > 3:
        raise NotImplementedError("Rasterize: Text direction arguments are not supported yet")
    xy = ctx.xy([pos])
    ctx.extend(xy)
    st = ctx.state.copy()
    T = ctx.T.copy()
    ang = math.degrees(math.atan2(T[1, 0], T[0, 0]))
    txt = _text_of(a[0]) if a else ""

    def go():
        ctx.ax.text(xy[0, 0], xy[0, 1], txt, ha=ha, va=va, color=_rgba_of(st), rotation=ang,
                    fontsize=st.font.get("size", 10.0), rotation_mode="anchor")
    ctx.draw.append(go)


def _r_arrow(ctx, e):
    segs = _points(e.args[0], "Arrow")
    if len(e.args) > 1:
        raise NotImplementedError("Rasterize: Arrow setbacks are not drawn yet")
    st = ctx.state.copy()
    _draw_lines(ctx, segs)
    for seg in segs:
        if len(seg) < 2:
            continue
        xy = ctx.xy(seg[-2:])

        def go(xy=xy, st=st):
            kind, v = st.arrow
            length = v * ctx.width if kind == "rel" else v
            ctx.ax.annotate("", xy=tuple(xy[1]), xytext=tuple(xy[0]),
                            arrowprops=dict(arrowstyle=f"-|>,head_length={length / 10:.3f},"
                                                       f"head_width={length / 20:.3f}",
                                            color=_rgba_of(st), linewidth=0, shrinkA=0, shrinkB=0))
        ctx.draw.append(go)


def _r_raster(ctx, e):
    np = ctx.np
    a = e.args
    pos, opts = G.split_trailing_options(a[1:])
    if opts:
        raise NotImplementedError("Rasterize: Raster options (ColorFunction, ...) are not drawn yet")
    data = np.asarray(_tolist(a[0]), dtype=float)
    n, m = data.shape[:2]
    if pos and not (isinstance(pos[0], Symbol) and pos[0].name == "Automatic"):
        (x0, y0), (x1, y1) = _points(pos[0], "Raster")[0]
    else:
        x0, y0, x1, y1 = 0.0, 0.0, float(m), float(n)
    if len(pos) > 1 and not (isinstance(pos[1], Symbol) and pos[1].name == "Automatic"):
        lo, hi = _f(pos[1].args[0]), _f(pos[1].args[1])
        data = (data - lo) / (hi - lo)
    data = np.clip(data, 0, 1)
    if data.ndim == 3 and data.shape[2] == 2:
        g, al = data[..., 0], data[..., 1]
        data = np.stack([g, g, g, al], axis=-1)
    if not np.allclose(ctx.T, np.eye(3)):
        raise NotImplementedError("Rasterize: transformed rasters are not drawn yet")
    ctx.extend(np.array([[x0, y0], [x1, y1]]))

    def go():
        kw = dict(cmap="gray", vmin=0, vmax=1) if data.ndim == 2 else {}
        ctx.ax.imshow(data, origin="lower", extent=(x0, x1, y0, y1), interpolation="nearest", aspect="auto", **kw)
    ctx.draw.append(go)


def _tolist(x):
    if K.is_list(x):
        return [_tolist(v) for v in x.args]
    return _f(x, "raster value")


RENDER_PRIMITIVES.update({"Point": _r_point, "Line": _r_line, "Polygon": _r_polygon, "Rectangle": _r_rectangle,
                          "Disk": _r_disk, "Circle": _r_circle, "Text": _r_text, "Arrow": _r_arrow,
                          "Raster": _r_raster})


def _matrix_of(ctx, e):
    """3x3 homogeneous matrix for Translate/Rotate/Scale wrappers (list of them for several translations)."""
    np = ctx.np
    h, a = e.head, e.args
    if h == "Translate":
        v = a[1]
        vecs = list(v.args) if K.is_list(v) and v.args and K.is_list(v.args[0]) else [v]
        out = []
        for w in vecs:
            if len(w.args) != 2:
                raise NotImplementedError("Rasterize: only 2D translations are drawn")
            M = np.eye(3)
            M[0, 2], M[1, 2] = _f(w.args[0]), _f(w.args[1])
            out.append(M)
        return out
    if h == "Rotate":
        if K.is_list(a[1]):
            u, v = ([_f(c) for c in w.args] for w in a[1].args)
            th = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
            c = [_f(t) for t in a[2].args] if len(a) > 2 else [0.0, 0.0]
        else:
            th = _f(a[1], "angle")
            c = _center(ctx, a[0], a[2] if len(a) > 2 else None)
        R = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1]])
        return [_about(np, R, c)]
    if h == "Scale":
        s = a[1]
        sx, sy = ((_f(s.args[0]), _f(s.args[1])) if K.is_list(s) else (_f(s),) * 2)
        c = _center(ctx, a[0], a[2] if len(a) > 2 else None)
        S = np.diag([sx, sy, 1.0])
        return [_about(np, S, c)]
    raise NotImplementedError(f"Rasterize: cannot draw {h}")


def _about(np, M, c):
    Tm = np.eye(3)
    Tm[0, 2], Tm[1, 2] = c[0], c[1]
    Ti = np.eye(3)
    Ti[0, 2], Ti[1, 2] = -c[0], -c[1]
    return Tm @ M @ Ti


def _center(ctx, g, p):
    if p is not None and K.is_list(p) and all(is_real_value(v) for v in p.args):
        return [_f(v) for v in p.args]
    try:
        c = G._fixed(g, p)
    except (G._Unresolved, Unevaluated, TypeError, ValueError):
        raise NotImplementedError("Rasterize: cannot find the bounding-box centre for this Rotate/Scale; give the "
                                  "fixed point explicitly")
    return [float(v) for v in c]


def _draw(ctx, x):
    if K.is_list(x):
        saved = ctx.state
        ctx.state = saved.copy()
        for i in x.args:
            _draw(ctx, i)
        ctx.state = saved
        return
    if x is None:
        return
    if G.is_rule(x):
        if G.opt_name(x) in ("Antialiasing",):
            return
        raise NotImplementedError(f"Rasterize: the style rule {K_input(x)} is not supported in graphics yet")
    if isinstance(x, Symbol):
        raise NotImplementedError(f"Rasterize: cannot draw the symbol {x.name}")
    if not isinstance(x, Expr) or not isinstance(x.head, str):
        raise NotImplementedError(f"Rasterize: cannot draw {K_input(x)}")
    h = x.head
    if _apply_directive(ctx, x):
        return
    f = RENDER_PRIMITIVES.get(h)
    if f is not None:
        f(ctx, x)
        return
    if h == "GraphicsComplex":
        _draw(ctx, G._gc_substitute(x))
        return
    if h in ("GraphicsGroup", "Annotation", "Tooltip", "Hyperlink", "Mouseover", "EventHandler", "Button") and x.args:
        _draw(ctx, x.args[0])
        return
    if h == "Style" and x.args:
        saved = ctx.state
        ctx.state = saved.copy()
        for s in x.args[1:]:
            if G.opt_name(s) == "FontSize":
                ctx.state.font["size"] = _f(s.args[1], "font size")
            elif G.is_rule(s):
                continue
            else:
                _apply_style(ctx, s)
        _draw(ctx, x.args[0])
        ctx.state = saved
        return
    if h in ("Translate", "Rotate", "Scale") and len(x.args) >= 2:
        for M in _matrix_of(ctx, x):
            saved_T, saved = ctx.T, ctx.state
            ctx.T = saved_T @ M
            ctx.state = saved.copy()
            _draw(ctx, x.args[0])
            ctx.T, ctx.state = saved_T, saved
        return
    raise NotImplementedError(f"Rasterize: drawing {h} is not supported yet")


_NO_EFFECT = {"Axes": (False,), "Frame": (False,), "Ticks": ("Automatic",), "FrameTicks": ("Automatic",),
              "GridLines": ("None",), "PlotLabel": ("None",), "AxesLabel": ("None",), "FrameLabel": ("None",),
              "Epilog": ("{}",), "Prolog": ("{}",), "AxesOrigin": ("Automatic",), "AxesStyle": ("{}",),
              "FrameStyle": ("{}",), "TicksStyle": ("{}",), "FrameTicksStyle": ("{}",), "GridLinesStyle": ("{}",),
              "PlotRangeClipping": (False, True), "ImagePadding": ("All", "Automatic", "None"),
              "ImageMargins": (0, "Automatic"), "ContentSelectable": ("Automatic", True, False),
              "Method": ("{}", "Automatic"), "DisplayFunction": ("Identity", "$DisplayFunction"),
              "FormatType": ("TraditionalForm", "StandardForm"), "BaseStyle": ("{}",), "LabelStyle": ("{}",),
              "AlignmentPoint": ("Center",), "BaselinePosition": ("Automatic",), "RotateLabel": (True,),
              "CoordinatesToolOptions": ("Automatic",), "PreserveImageOptions": ("Automatic", True),
              "ImageSizeRaw": ("Automatic",), "PlotRegion": ("Automatic",), "Antialiasing": (True, False, "Automatic"),
              "LightDark": ("Automatic", "Inherited")}


def _no_effect(name, v):
    allowed = _NO_EFFECT.get(name)
    if allowed is None:
        return False
    for a in allowed:
        if a is v or (isinstance(a, bool) and v is a):
            return True
        if isinstance(a, str) and not isinstance(v, bool):
            if a == "{}" and K.is_list(v) and not v.args:
                return True
            if isinstance(v, Symbol) and v.name == a:
                return True
        if isinstance(a, int) and not isinstance(a, bool) and v == a and not isinstance(v, bool):
            return True
    return False


def _plot_range(v, b):
    if isinstance(v, Symbol) and v.name in ("Automatic", "All", "Full"):
        return b
    if K.is_list(v) and len(v.args) == 2 and all(K.is_list(p) for p in v.args):
        out = []
        for p, (lo, hi) in zip(v.args, ((b[0], b[1]), (b[2], b[3]))):
            if isinstance(p, Symbol) and p.name in ("Automatic", "All", "Full"):
                out += [lo, hi]
            elif K.is_list(p) and len(p.args) == 2:
                out += [_f(p.args[0], "plot range"), _f(p.args[1], "plot range")]
            else:
                raise NotImplementedError(f"Rasterize: PlotRange {K_input(v)} is not supported yet")
        return out
    raise NotImplementedError(f"Rasterize: PlotRange {K_input(v)} is not supported yet")


def _image_size(v):
    if v is None or (isinstance(v, Symbol) and v.name == "Automatic"):
        return None
    if isinstance(v, Symbol) and v.name in _IMAGE_SIZES:
        return (float(_IMAGE_SIZES[v.name]), None)
    if K.is_list(v) and len(v.args) == 2:
        w = None if isinstance(v.args[0], Symbol) else _f(v.args[0], "image size")
        h = None if isinstance(v.args[1], Symbol) else _f(v.args[1], "image size")
        return (w, h)
    w = _f(v, "image size")
    if w <= 0:
        raise ValueError("Rasterize: ImageSize must be positive")
    return (w, None)


def render_figure(g, image_size=None, raster_size=None, background=None, resolution=72.0):
    """matplotlib Figure for a Graphics object (see module docstring)."""
    _need()
    if not (isinstance(g, Expr) and g.head == "Graphics"):
        if isinstance(g, Expr) and g.head == "Graphics3D":
            raise NotImplementedError("Rasterize: drawing 3D graphics is not supported yet")
        if isinstance(g, Expr) and g.head in ("GraphicsGrid", "GraphicsRow", "GraphicsColumn"):
            raise NotImplementedError(f"Rasterize: drawing {g.head} layouts is not supported yet")
        raise NotImplementedError(f"Rasterize: only Graphics objects can be rasterized so far, not {K_input(g)}")
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    dim, content, opts = G.graphics_parts(g)
    od = G.options_dict(opts)
    size = _image_size(image_size if image_size is not None else od.get("ImageSize"))
    width = (size[0] if size and size[0] else 360.0)
    ctx = Ctx(None, width)
    _draw(ctx, content)
    b = ctx.bounds
    if not math.isfinite(b[0]):
        b = [-1.0, 1.0, -1.0, 1.0]
    pr = od.get("PlotRange")
    b = list(_plot_range(pr, b)) if pr is not None else list(b)
    for i in (0, 2):
        if b[i + 1] - b[i] <= 0:
            c = b[i]
            half = max(abs(c), 1.0) * 0.5
            b[i], b[i + 1] = c - half, c + half
    pad = od.get("PlotRangePadding")
    if pad is None or (isinstance(pad, Symbol) and pad.name == "Automatic"):
        fx = fy = 0.02
    elif isinstance(pad, Symbol) and pad.name == "None":
        fx = fy = 0.0
    elif isinstance(pad, Expr) and pad.head == "Scaled":
        fx = fy = _f(pad.args[0], "padding")
    else:
        raise NotImplementedError(f"Rasterize: PlotRangePadding {K_input(pad)} is not supported yet")
    dx, dy = (b[1] - b[0]) * fx, (b[3] - b[2]) * fy
    xlim, ylim = (b[0] - dx, b[1] + dx), (b[2] - dy, b[3] + dy)
    ar = od.get("AspectRatio")
    if ar is None or (isinstance(ar, Symbol) and ar.name == "Automatic"):
        aspect = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
    elif isinstance(ar, Symbol) and ar.name == "Full":
        raise NotImplementedError("Rasterize: AspectRatio -> Full is not supported yet")
    else:
        aspect = _f(ar, "aspect ratio")
        if aspect <= 0:
            raise ValueError("Rasterize: AspectRatio must be positive")
    if size and size[1] and not size[0]:
        width = size[1] / aspect
    height = size[1] if size and size[1] and size[0] else width * aspect
    ctx.width = width
    for name, v in od.items():
        if name in ("ImageSize", "PlotRange", "PlotRangePadding", "AspectRatio", "Background"):
            continue
        h = RENDER_OPTIONS.get(name)
        if h is not None:
            continue
        if not _no_effect(name, v):
            raise NotImplementedError(f"Rasterize: rendering the option {name} -> {K_input(v)} is not supported "
                                      "yet")
    dpi = resolution
    if raster_size is not None:
        rw, rh = raster_size
        if rw is None:
            rw = rh / aspect
        dpi = rw / (width / 72.0)
        if rh is not None:
            height = rh / dpi * 72.0
    # Snap to whole pixels (nudged up by 1e-3 px) so the raster size is the same whether matplotlib truncates
    # (older versions) or rounds figsize * dpi; otherwise 199.9999 px could become 199 px.
    px_w, px_h = max(1, int(round(width / 72.0 * dpi))), max(1, int(round(height / 72.0 * dpi)))
    fig = Figure(figsize=((px_w + 1e-3) / dpi, (px_h + 1e-3) / dpi), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    bg = background if background is not None else od.get("Background")
    if bg is None or (isinstance(bg, Symbol) and bg.name == "Automatic"):
        fig.patch.set_facecolor("white")
    elif isinstance(bg, Symbol) and bg.name == "None":
        fig.patch.set_alpha(0.0)
    else:
        c = to_rgba(bg)
        if c is None:
            raise ValueError(f"Rasterize: Background must be a colour or None, not {K_input(bg)}")
        fig.patch.set_facecolor(c)
    ctx.ax = ax
    for name, v in od.items():
        h = RENDER_OPTIONS.get(name)
        if h is not None:
            h(ctx, v)
    for d in ctx.draw:
        d()
    fig._mikoshi_has_alpha = isinstance(bg, Symbol) and bg.name == "None"
    return fig


def raster_array(g, **kw):
    import numpy as np
    fig = render_figure(g, **kw)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba()).copy()
    return buf if getattr(fig, "_mikoshi_has_alpha", False) else buf[..., :3]


def render_png(g, **kw):
    import io
    fig = render_figure(g, **kw)
    out = io.BytesIO()
    fig.savefig(out, format="png", dpi=fig.dpi, facecolor=fig.get_facecolor())
    return out.getvalue()


def render_svg(g, **kw):
    import io
    fig = render_figure(g, **kw)
    out = io.StringIO()
    fig.savefig(out, format="svg", facecolor=fig.get_facecolor())
    return out.getvalue()


def _raster_size(v):
    if v is None or (isinstance(v, Symbol) and v.name == "Automatic"):
        return None
    if K.is_list(v) and len(v.args) == 2:
        w, h = (None if isinstance(t, Symbol) else int(round(_f(t, "raster size"))) for t in v.args)
        return (w, h)
    n = _f(v, "raster size")
    if n <= 0:
        raise ValueError("Rasterize: RasterSize must be positive")
    return (int(round(n)), None)


_ELEMENTS = ("Image", "Data", "RasterSize", "Graphics", "BoundingBox", "Regions")


@fn("Rasterize", area=AREA, status="added",
    notes="Rasterize[g, \"Data\"] the pixel array (integers 0-255, top row first, {r, g, b} or {r, g, b, a} with "
          "Background -> None) of a 2D Graphics drawn with matplotlib; Rasterize[g, \"RasterSize\"] its {width, "
          "height}; Rasterize[g, {elem1, ...}] a list. Options ImageSize (points, default 360 wide; Tiny/Small/"
          "Medium/Large), RasterSize (pixels), ImageResolution (dpi, default 72) and Background. Drawn: Point, "
          "Line, Polygon, Rectangle, Disk, Circle (and sectors), Text, Arrow, Raster, GraphicsComplex, "
          "GraphicsGroup, Style, Annotation, Translate/Rotate/Scale, colours, Opacity, Thickness, PointSize, "
          "Dashing, EdgeForm, FaceForm, CapForm, JoinForm, Arrowheads[s], Directive; PlotRange, PlotRangePadding, "
          "AspectRatio; and, through the W15 hooks (wl_plot_options), Axes, AxesOrigin, AxesLabel, AxesStyle, Ticks, "
          "TicksStyle, Frame, FrameLabel, FrameStyle, FrameTicks, FrameTicksStyle, GridLines, GridLinesStyle, "
          "PlotLabel, RotateLabel, LabelStyle, Epilog and Prolog. Anything else (Graphics3D, gradient fillings, "
          "insets, unsupported option settings, ...) raises "
          "NotImplementedError rather than being left out; Rasterize[g] / \"Image\" needs the Image object (image "
          "batch W21) and raises NotImplementedError until then. Pixels approximate, not reproduce, Wolfram's "
          "renderer. Requires matplotlib and numpy (NotImplementedError otherwise); symbolic input stays "
          "unevaluated. Check: self-check of the raster shape against RasterSize, and pixel read-back of filled "
          "regions in tests.")
def _rasterize(g: Expr, *rest: String | ListOf[String] | Option) -> Expr:
    pos, opts = G.split_trailing_options(rest)
    G.nargs(pos, 0, 1, "Rasterize")
    od = G.options_dict(opts)
    for n in od:
        if n not in ("ImageSize", "RasterSize", "ImageResolution", "Background"):
            raise ValueError(f"Rasterize: {n} is not a supported option of Rasterize")
    if isinstance(g, Symbol) or (isinstance(g, Expr) and g.head not in ("Graphics", "Graphics3D", "GraphicsGrid",
                                                                         "GraphicsRow", "GraphicsColumn")
                                 and not K.is_list(g) and not is_real_value(g)):
        if isinstance(g, Symbol):
            raise Unevaluated()
    elem = pos[0] if pos else "Image"
    elems = list(elem.args) if K.is_list(elem) else [elem]
    for e in elems:
        if not isinstance(e, str):
            raise TypeError(f"Rasterize: the element must be a string such as \"Data\", not {K_input(e)}")
        if e not in _ELEMENTS:
            raise ValueError(f"Rasterize: {e!r} is not a Rasterize element ({', '.join(_ELEMENTS)})")
        if e in ("Graphics", "BoundingBox", "Regions"):
            raise NotImplementedError(f"Rasterize: the element {e!r} is not supported yet")
        if e == "Image" and IMAGE_FACTORY is None:
            raise NotImplementedError("Rasterize: Image results need the Image object (image batch W21); use "
                                      "Rasterize[g, \"Data\"] for the pixel array")
    res = od.get("ImageResolution")
    dpi = 72.0 if res is None or (isinstance(res, Symbol) and res.name == "Automatic") else \
        G.real_arg(res, "Rasterize", "ImageResolution", 0, strict=True)
    kw = dict(image_size=od.get("ImageSize"), raster_size=_raster_size(od.get("RasterSize")),
              background=od.get("Background"), resolution=dpi)
    arr = raster_array(g, **kw)
    if kw["raster_size"] is not None:
        want_w, want_h = kw["raster_size"]
        if (want_w is not None and abs(arr.shape[1] - want_w) > 1) or (want_h is not None and
                                                                       abs(arr.shape[0] - want_h) > 1):
            raise ArithmeticError("Rasterize: raster size failed verification")
    out = []
    for e in elems:
        if e == "Data":
            out.append(L(*[L(*[L(*[int(v) for v in px]) for px in row]) for row in arr]))
        elif e == "RasterSize":
            out.append(L(int(arr.shape[1]), int(arr.shape[0])))
        else:
            out.append(IMAGE_FACTORY(arr))
    return L(*out) if K.is_list(elem) else out[0]
