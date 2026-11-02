"""Wolfram batch X04 (part 3): image creation, composition and colour visualisation.

RegionImage, ReliefImage, HighlightImage, ImageCollage, ImageEffect, ImageWaveformPlot, ChromaticityPlot and
ChromaticityPlot3D. Each function states its method and defaults in its notes; nothing claims to reproduce
undocumented Wolfram rendering choices. CIE colour-matching data: ``wl_cie_data`` (official CIE tables).
"""
from __future__ import annotations

import math

import numpy as np

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, S
from . import wl_images as WI
from .wl_images import parse_image, image_expr, opts_split, IMAGE_HEADS, _short, _is_sym, quantize
from .wl_imageproc2 import _num, _image, _opt_check, _ev, region_mask, gaussian_smooth
from . import wl_cie_data as CIE

AREA = "image processing III (X04)"
AUTO = Symbol("Automatic")


def fl(v):
    return float(round(float(v), 6))


def pt(*v):
    return L(*[fl(x) for x in v])


def raster(a, x0=0.0, y0=0.0, x1=None, y1=None):
    """Raster of an (h, w, c) array on 0..1 (rows from the top), placed on {{x0, y0}, {x1, y1}} (default {w, h})."""
    h, w = a.shape[:2]
    x1 = w if x1 is None else x1
    y1 = h if y1 is None else y1
    rows = a[::-1]
    if a.shape[2] == 1:
        data = L(*[L(*[fl(v) for v in row[:, 0]]) for row in rows])
    else:
        data = L(*[L(*[L(*[fl(v) for v in px]) for px in row]) for row in rows])
    return Expr("Raster", data, L(pt(x0, y0), pt(x1, y1)), L(0, 1))


def _rgb_image(x, fname):
    """(Img, RGB array (h, w, 3), alpha or None, was single channel)."""
    im = _image(x, fname, allow3d=False)
    src = WI.source_space(im, fname)
    col = im.color()
    single = col.shape[-1] == 1
    rgb = WI.convert_array(col, src, "RGB") if src != "RGB" else col
    return im, np.clip(rgb, 0, 1), (im.a[..., -1:] if im.alpha else None), single


def _split_any(args):
    """(positional, {name: value}) with every trailing Name -> value rule taken as an option."""
    args = list(args)
    out = {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") \
            and len(args[-1].args) == 2 and isinstance(args[-1].args[0], Symbol):
        r = args.pop()
        out = {r.args[0].name: r.args[1], **out}
    return args, out


# ───────────────────────────── RegionImage ─────────────────────────────

REGION_RASTER = 100          # default pixels along the larger side
REGION_SUPER = 4             # supersamples per pixel and axis (2D)


def _region_dim(reg):
    v = _ev(Expr("RegionDimension", reg))
    return v if isinstance(v, int) else None


@fn("RegionImage", area=AREA, status="added",
    notes="RegionImage[reg] and RegionImage[reg, {{xmin, xmax}, {ymin, ymax}}] (or the 3D range) rasterise a "
          "constant 2D or 3D region into a grayscale Real32 Image (Image3D in 3D) with the region dark: each pixel "
          "value is 1 - coverage: pixels whose centre and all neighbours' centres agree are 0 or 1 (RegionMember at "
          "the centre), and the boundary band (mixed neighbourhoods) uses the fraction of 4 x 4 (2 x 2 x 2 in 3D) "
          "regularly spaced samples per pixel inside the region, so features thinner than a pixel that miss every "
          "centre are not drawn. Lower-dimensional regions (points, curves, "
          "surfaces) are drawn with thickness t: samples within RegionDistance t/2 count as inside; t defaults to "
          "one pixel and Method -> {\"Thickness\" -> t} sets it in region units. The default range is RegionBounds "
          "(padded by t/2 for lower-dimensional regions); RasterSize -> n gives n pixels along the larger side "
          "(default 100 in 2D, 50 in 3D) and {nx, ny} (or {nx, ny, nz}) sets every side. Pixel {i, j} covers the standard image "
          "coordinates, the top row at ymax. ImageResolution, ImageSize and MetaInformation are accepted and do "
          "not change the pixels. Check: reference (the dark area of a Disk image equals Pi r^2 within the "
          "sampling error; a Rectangle aligned with pixels gives exact 0/1 values) in tests.")
def _regionimage(reg: Expr, *rest: ListOf[ListOf[Real]] | Option) -> Expr:
    fname = "RegionImage"
    pos, o = opts_split(rest, {"ImageResolution", "ImageSize", "MetaInformation", "Method", "RasterSize"}, fname)
    _opt_check(pos, fname, 1)
    if isinstance(reg, Symbol):
        raise Unevaluated()
    from .wl_regions import region_of
    r = region_of(reg, fname)
    emb = _ev(Expr("RegionEmbeddingDimension", reg))
    if emb not in (2, 3):
        raise ValueError(f"{fname}: the region must be embedded in 2D or 3D, not {_short(emb)}")
    rdim = _region_dim(reg)
    if pos:
        rng = WI._numeric_array(pos[0], fname)
        if rng is None or rng.shape != (emb, 2) or not (rng[:, 1] > rng[:, 0]).all():
            raise ValueError(f"{fname}: the range must be {{{{min, max}}, ...}} for each of the {emb} coordinates "
                             "with min < max")
    else:
        b = r.bounds()
        rng = np.array([[float(lo), float(hi)] for lo, hi in b], dtype=float)
        if not np.isfinite(rng).all():
            raise ValueError(f"{fname}: the region is unbounded; give an explicit range")
    rs = o.get("RasterSize", AUTO)
    if _is_sym(rs, "Automatic"):
        span = rng[:, 1] - rng[:, 0]
        span = np.where(span > 0, span, max(span.max(), 1.0))
        n = REGION_RASTER if emb == 2 else 50
        sizes = np.maximum(1, np.round(n * span / span.max())).astype(int)
    elif isinstance(rs, int) and not isinstance(rs, bool) and rs > 0:
        span = rng[:, 1] - rng[:, 0]
        span = np.where(span > 0, span, max(span.max(), 1.0))
        sizes = np.maximum(1, np.round(rs * span / span.max())).astype(int)
    else:
        arr = WI._numeric_array(rs, fname) if K.is_list(rs) else None
        if arr is None or arr.shape != (emb,) or (arr < 1).any() or (arr != np.round(arr)).any():
            raise ValueError(f"{fname}: RasterSize must be Automatic, a positive integer or a list of {emb} of them")
        sizes = arr.astype(int)
    thick = None
    m = o.get("Method", AUTO)
    if not _is_sym(m, "Automatic"):
        rules = m.args if K.is_list(m) else (m,)
        for ru in rules:
            if isinstance(ru, Expr) and ru.head == "Rule" and ru.args[0] == "Thickness":
                thick = _num(ru.args[1], fname, "the thickness", lo=0)
            else:
                raise ValueError(f"{fname}: unknown Method setting {_short(ru)}")
    pix = (rng[:, 1] - rng[:, 0]) / sizes
    lowdim = rdim is not None and rdim < emb
    if lowdim:
        t = float(pix.min()) if thick is None else thick
        if not pos:
            rng = rng + np.array([[-t / 2, t / 2]] * emb)
            pix = (rng[:, 1] - rng[:, 0]) / sizes
    ss = REGION_SUPER if emb == 2 else 2

    def inside(p):
        return float(r.dist(tuple(p))) <= t / 2 if lowdim else bool(r.member(tuple(p)))
    centres = [rng[k, 0] + (np.arange(sizes[k]) + 0.5) * pix[k] for k in range(emb)]
    cov = np.zeros(tuple(sizes))
    for idx in np.ndindex(*sizes):
        cov[idx] = inside([centres[k][idx[k]] for k in range(emb)])
    # supersample the pixels whose neighbourhood is mixed (the boundary band)
    mixed = np.zeros(cov.shape, dtype=bool)
    padded = np.pad(cov, 1, mode="edge")
    for off in np.ndindex(*([3] * emb)):
        sl = tuple(slice(o, o + sizes[k]) for k, o in enumerate(off))
        mixed |= padded[sl] != cov
    sub = (np.arange(ss) + 0.5) / ss
    for idx in zip(*np.nonzero(mixed)):
        acc = 0
        for so in np.ndindex(*([ss] * emb)):
            acc += inside([rng[k, 0] + (idx[k] + sub[so[k]]) * pix[k] for k in range(emb)])
        cov[idx] = acc / ss ** emb
    if emb == 2:
        img = (1 - cov).T[::-1][..., None]                    # (x, y) -> rows from the top
        return image_expr(WI.Img(quantize(img, "Real32"), "Real32", "Automatic", False))
    img = (1 - cov).transpose(2, 1, 0)[:, ::-1][..., None]    # (x, y, z) -> (slices z, rows y from the top, x)
    return image_expr(WI.Img(quantize(img, "Real32"), "Real32", "Automatic", True))


# ───────────────────────────── ReliefImage ─────────────────────────────

RELIEF_AZIMUTH, RELIEF_ELEVATION = 3 * math.pi / 4, math.pi / 4      # light from the upper left, 45° high
RELIEF_EXAGGERATION = 1 / 8                                          # height range = 1/8 of the larger side


@fn("ReliefImage", area=AREA, status="added",
    notes="ReliefImage[array] and ReliefImage[image] (the first channel's gray level, or the mean of the colour "
          "channels) render a height field as an 8-bit RGB image by Lambertian hillshading: heights are scaled so "
          "that the PlotRange (All: the data range; {min, max} clips) spans 1/8 of the larger dimension in pixels, "
          "normals come from central differences (one-sided at the edges), and the shade is max(0, n·l) for a "
          "light at azimuth 3Pi/4 (from the upper left) and elevation Pi/4; LightingAngle -> θ or {θ, φ} sets the "
          "azimuth (counterclockwise from the +x axis) and elevation. The colour is shade x ColorFunction[h] with "
          "h the height scaled to 0..1 (ColorFunctionScaling -> True; False passes raw heights); ColorFunction -> "
          "Automatic is white, so the default image is the gray shading. With an explicit {min, max} PlotRange the "
          "clipped parts take ClippingStyle colours {below, above} (default {Black, White}); ClippingStyle -> None "
          "shades them like other parts. Method \"DiffuseReflection\" (the default) is the Lambert model; "
          "\"AspectBasedShading\" raises NotImplementedError. Check: reference (a plane facing the light gives the "
          "uniform shade cos of the angle between normal and light) in tests.")
def _reliefimage(x: MatrixOf[Real] | Expr, *opts: Option) -> Expr:
    fname = "ReliefImage"
    pos, o = opts_split(opts, {"ClippingStyle", "ColorFunction", "ColorFunctionScaling", "ImageSize", "LightingAngle",
                               "Method", "PlotRange"}, fname)
    _opt_check(pos, fname, 0)
    if isinstance(x, Expr) and x.head in IMAGE_HEADS:
        im = _image(x, fname, allow3d=False)
        H = im.color().mean(-1)
    else:
        arr = WI._numeric_array(x, fname) if K.is_list(x) else None
        if arr is None:
            if isinstance(x, Symbol) or (isinstance(x, Expr) and not K.is_list(x)):
                raise Unevaluated()
            raise TypeError(f"{fname}: expected a matrix of real numbers or an image")
        if arr.ndim != 2:
            raise ValueError(f"{fname}: the array must be a matrix of real numbers")
        H = arr.astype(float)
    if not np.isfinite(H).all():
        raise ValueError(f"{fname}: the heights must be finite")
    meth = o.get("Method", AUTO)
    if meth == "AspectBasedShading":
        raise NotImplementedError(f"{fname}: Method \"AspectBasedShading\" is not supported")
    if not (_is_sym(meth, "Automatic") or meth == "DiffuseReflection"):
        raise ValueError(f"{fname}: unknown Method {_short(meth)}")
    pr = o.get("PlotRange", Symbol("All"))
    explicit = False
    if _is_sym(pr, "All") or _is_sym(pr, "Automatic"):
        lo, hi = float(H.min()), float(H.max())
    else:
        a = WI._numeric_array(pr, fname) if K.is_list(pr) else None
        if a is None or a.shape != (2,) or not a[1] > a[0]:
            raise ValueError(f"{fname}: PlotRange must be All or {{min, max}} with min < max")
        lo, hi = float(a[0]), float(a[1])
        explicit = True
    span = hi - lo
    Hc = np.clip(H, lo, hi)
    h, w = H.shape
    z = (Hc - lo) / span * RELIEF_EXAGGERATION * max(h, w) if span > 0 else np.zeros_like(H)
    gy, gx = np.gradient(z) if min(h, w) > 1 else (np.zeros_like(z), np.zeros_like(z))
    # image rows go down: dz/dy (up) = -dz/drow
    nx, ny, nz = -gx, gy, np.ones_like(z)
    nn = np.sqrt(nx * nx + ny * ny + nz * nz)
    la = o.get("LightingAngle", AUTO)
    az, el = RELIEF_AZIMUTH, RELIEF_ELEVATION
    if not _is_sym(la, "Automatic"):
        if K.is_list(la) and len(la.args) == 2:
            az = _num(la.args[0], fname, "the light azimuth")
            el = _num(la.args[1], fname, "the light elevation")
        else:
            az = _num(la, fname, "the light azimuth")
    lv = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    shade = np.maximum(0.0, (nx * lv[0] + ny * lv[1] + nz * lv[2]) / nn)
    cf = o.get("ColorFunction", AUTO)
    scaling = o.get("ColorFunctionScaling", True)
    if scaling not in (True, False):
        raise ValueError(f"{fname}: ColorFunctionScaling must be True or False")
    if _is_sym(cf, "Automatic"):
        col = np.ones(H.shape + (3,))
    else:
        from .wl_graphics_color import to_rgba
        hs = (Hc - lo) / span if (scaling and span > 0) else (np.zeros_like(Hc) if scaling else Hc)
        uniq = {}
        col = np.zeros(H.shape + (3,))
        for idx in np.ndindex(H.shape):
            v = float(round(hs[idx], 12))
            if v not in uniq:
                c = _ev(Expr(cf, v) if not isinstance(cf, str) else Expr("ColorData", cf, v))
                rgba = to_rgba(c)
                if rgba is None:
                    raise ValueError(f"{fname}: the ColorFunction must return colours, not {_short(c)}")
                uniq[v] = rgba[:3]
            col[idx] = uniq[v]
    out = col * shade[..., None]
    cs = o.get("ClippingStyle", L(Symbol("Black"), Symbol("White")))
    if explicit and not _is_sym(cs, "None"):
        from .wl_graphics_color import to_rgba
        cl = _ev(cs)
        pair = cl.args if K.is_list(cl) and len(cl.args) == 2 else (cl, cl)
        rg = [to_rgba(c) for c in pair]
        if None in rg:
            raise ValueError(f"{fname}: ClippingStyle must be None, a colour or {{below, above}} colours")
        out[H < lo] = rg[0][:3]
        out[H > hi] = rg[1][:3]
    return image_expr(WI.Img(quantize(np.clip(out, 0, 1), "Byte"), "Byte", "RGB", False))


# ───────────────────────────── ImageWaveformPlot ─────────────────────────────

_WAVE_CHANNELS = {"R": ("RGB", 0), "G": ("RGB", 1), "B": ("RGB", 2), "C": ("CMYK", 0), "M": ("CMYK", 1),
                  "Y": ("CMYK", 2), "K": ("CMYK", 3), "H": ("HSB", 0), "S": ("HSB", 1), "V": ("HSB", 2),
                  "L": ("LAB", 0), "I": ("Grayscale", 0)}
_WAVE_COLORS = {"RGB": ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "CMYK": ((0, 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, 1)),
                "Grayscale": ((1, 1, 1),)}
WAVE_LEVELS = 128


@fn("ImageWaveformPlot", area=AREA, status="added",
    notes="ImageWaveformPlot[image], [image, colorspace] and [image, channel]: for every image column and channel, "
          "the counts of pixel levels in 128 equal bins over 0..1 (levels outside are clipped into the end bins; "
          "with an alpha channel each pixel counts with its alpha weight), drawn as a Raster on {{0, 0}, {w, 1}} "
          "(x the column, y the level) on a black background: each channel adds its colour (red/green/blue; "
          "cyan/magenta/yellow/white; white for other spaces and single channels) times count/max count, clipped "
          "to 1 (PlotLayout \"Overlaid\", the default). colorspace: \"RGB\" (default; grayscale images use their "
          "gray channel), \"CMYK\", \"HSB\", \"XYZ\", \"LAB\", \"LCH\", \"LUV\" or Automatic (the image's own "
          "channels); channel: \"I\", \"L\", \"R\", \"G\", \"B\", \"C\", \"M\", \"Y\", \"K\", \"H\", \"S\", \"V\", "
          "\"Average\" or \"A\". Channels on other scales (LAB a and b, ...) are rescaled from their space's range "
          "as in ImageData. Frame True, AspectRatio 1/GoldenRatio. PlotLayout \"Row\"/\"Column\", Masking, "
          "MaxPlotPoints and ScalingFunctions other than Automatic raise NotImplementedError. Check: exact (column "
          "counts compared with BinCounts in tests).")
def _imagewaveformplot(x: Expr, *rest: String | Expr | Option) -> Expr:
    fname = "ImageWaveformPlot"
    pos, o = _split_any(rest)
    _opt_check(pos, fname, 1)
    im = _image(x, fname, allow3d=False)
    gopts = []
    for k, v in o.items():
        if k in ("PlotLayout",):
            if v != "Overlaid" and not _is_sym(v, "Automatic"):
                raise NotImplementedError(f"{fname}: PlotLayout {_short(v)} is not supported")
        elif k in ("Masking", "MaxPlotPoints", "ScalingFunctions"):
            if not (_is_sym(v, "Automatic") or (k == "Masking" and _is_sym(v, "All"))):
                raise NotImplementedError(f"{fname}: the option {k} is not supported")
        else:
            gopts.append(Expr("Rule", Symbol(k), v))
    src = WI.source_space(im, fname)
    col = im.color()
    wts = im.a[..., -1] if im.alpha else np.ones(col.shape[:2])
    spec = pos[0] if pos else "RGB"
    if _is_sym(spec, "Automatic"):
        chans, cols = col, _WAVE_COLORS.get(src, ((1, 1, 1),) * col.shape[-1])
    elif spec in _WAVE_CHANNELS:
        sp, k = _WAVE_CHANNELS[spec]
        chans = (WI.convert_array(col, src, sp) if src != sp else col)[..., k:k + 1]
        cols = ((1, 1, 1),)
    elif spec == "Average":
        chans, cols = col.mean(-1, keepdims=True), ((1, 1, 1),)
    elif spec == "A":
        if not im.alpha:
            raise ValueError(f"{fname}: the image has no alpha channel")
        chans, cols = im.a[..., -1:], ((1, 1, 1),)
    elif spec in ("RGB", "CMYK", "HSB", "XYZ", "LAB", "LCH", "LUV"):
        if spec == "RGB" and src == "Grayscale":
            chans, cols = col, ((1, 1, 1),)
        else:
            chans = WI.convert_array(col, src, spec) if src != spec else col
            cols = _WAVE_COLORS.get(spec, ((1, 1, 1),) * chans.shape[-1])
    elif isinstance(spec, Symbol):
        raise Unevaluated()
    else:
        raise ValueError(f"{fname}: unknown colour space or channel {_short(spec)}")
    h, w = chans.shape[:2]
    acc = np.zeros((WAVE_LEVELS, w, 3))
    for c in range(chans.shape[-1]):
        idx = np.clip((chans[..., c] * WAVE_LEVELS).astype(int), 0, WAVE_LEVELS - 1)
        cnt = np.zeros((WAVE_LEVELS, w))
        for j in range(w):
            cnt[:, j] = np.bincount(idx[:, j], weights=wts[:, j], minlength=WAVE_LEVELS)
        mx = cnt.max()
        if mx > 0:
            acc += (cnt / mx)[..., None] * np.array(cols[c], dtype=float)
    acc = np.clip(acc, 0, 1)[::-1]          # top row = highest level
    from .wl_plots2d import finish_graphics
    return finish_graphics(L(raster(acc, 0, 0, w, 1)), gopts,
                           [("Frame", True), ("AspectRatio", Expr("Power", Symbol("GoldenRatio"), -1)),
                            ("Background", Symbol("Black")), ("PlotRange", L(L(0, w), L(0, 1)))])


# ───────────────────────────── chromaticity ─────────────────────────────

# primaries (x, y) and white points of RGB spaces (Lindbloom's RGB working space table; IEC 61966-2-1 for sRGB)
D65 = (0.3127, 0.3290)
D50 = (0.3457, 0.3585)
RGB_SPACES = {
    "sRGB": (((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), D65),
    "AdobeRGB": (((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), D65),
    "AppleRGB": (((0.625, 0.34), (0.28, 0.595), (0.155, 0.07)), D65),
    "CIERGB": (((0.7347, 0.2653), (0.2738, 0.7174), (0.1666, 0.0089)), (1 / 3, 1 / 3)),
    "ProPhotoRGB": (((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001)), D50),
    "WideGamutRGB": (((0.7350, 0.2650), (0.1150, 0.8260), (0.1570, 0.0180)), D50),
}
BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])
_SRGB_FAMILY = ("RGB", "sRGB", "HSB", "CMYK")
_SPACE_COLORS = ((0.8, 0.1, 0.1), (0.1, 0.5, 0.9), (0.1, 0.6, 0.2), (0.6, 0.2, 0.7), (0.9, 0.5, 0.0), (0.3, 0.3, 0.3))


def _xyz_of_xy(x, y, Y=1.0):
    return np.array([x / y * Y, Y, (1 - x - y) / y * Y])


def adapted_primaries(name):
    """xy of an RGB space's primaries after Bradford adaptation from its white to the D50 white used by
    ColorConvert's XYZ (ICC profile connection space), so that "sRGB" matches ColorConvert[RGBColor[...], "XYZ"]."""
    from .wl_graphics_color import WHITE_D50
    prim, wp = RGB_SPACES[name]
    P = np.stack([_xyz_of_xy(*p) for p in prim], axis=1)
    W = _xyz_of_xy(*wp)
    Sc = np.linalg.solve(P, W)
    M = P * Sc
    src = BRADFORD @ W
    dst = BRADFORD @ np.array(WHITE_D50)
    A = np.linalg.inv(BRADFORD) @ np.diag(dst / src) @ BRADFORD
    XYZ = A @ M
    return [(XYZ[0, k] / XYZ[:, k].sum(), XYZ[1, k] / XYZ[:, k].sum()) for k in range(3)]


def spectral_locus(observer=2):
    tab = CIE.CIE1931_2DEG if observer == 2 else CIE.CIE1964_10DEG
    out = []
    for _, X, Y, Z in tab:
        s = X + Y + Z
        if s > 0:
            out.append((X / s, Y / s))
    return out


def _uv(x, y):
    d = -2 * x + 12 * y + 3
    return 4 * x / d, 9 * y / d


def _to_xyz(space, ch):
    from .wl_graphics_color import convert_channels
    return convert_channels(space, tuple(float(c) for c in ch), "XYZ")


def _colour_points(obj, fname, maxpts):
    """XYZ triples of a colour, an image's pixels (subsampled to maxpts) or None if obj is neither."""
    from .wl_graphics_color import parse_color
    p = parse_color(obj)
    if p is not None:
        return [_to_xyz(p[0], p[1])]
    if isinstance(obj, Expr) and obj.head in IMAGE_HEADS:
        im = _image(obj, fname, allow3d=False)
        src = WI.source_space(im, fname)
        px = im.color().reshape(-1, im.color().shape[-1])
        step = max(1, int(math.ceil(len(px) / maxpts)))
        sp = {"Grayscale": "Grayscale"}.get(src, src)
        return [_to_xyz(sp, c) for c in px[::step]]
    return None


def _chrom_parse(objs, fname, maxpts):
    spaces, points = [], []
    for ob in objs:
        if isinstance(ob, str):
            if ob in RGB_SPACES or ob in _SRGB_FAMILY or ob == "Grayscale":
                spaces.append(ob)
            elif ob in ("XYZ", "LAB", "LCH", "LUV"):
                raise NotImplementedError(f"{fname}: the gamut of {ob} is not bounded by the spectral locus "
                                          "rendering; give colours or images in it instead")
            else:
                raise ValueError(f"{fname}: unknown colour space {ob!r}")
            continue
        pts = _colour_points(ob, fname, maxpts)
        if pts is None:
            if isinstance(ob, Expr) and ob.head in ("ColorProfileData", "ColorsNear"):
                raise NotImplementedError(f"{fname}: {ob.head} is not supported")
            if isinstance(ob, Symbol) or (isinstance(ob, Expr) and not K.is_list(ob)):
                raise Unevaluated()
            raise TypeError(f"{fname}: {_short(ob)} is not a colour space, colour or image")
        points += pts
    return spaces, points


def _space_xy(name):
    if name in _SRGB_FAMILY:
        name = "sRGB"
    if name == "Grayscale":
        from .wl_graphics_color import WHITE_D50
        s = sum(WHITE_D50)
        return [(WHITE_D50[0] / s, WHITE_D50[1] / s)]
    return adapted_primaries(name)


def _objects(arg):
    return list(arg.args) if K.is_list(arg) and not (arg.args and all(isinstance(a, (int, float)) for a in arg.args)) \
        else [arg]


@fn("ChromaticityPlot", area=AREA, status="added",
    notes="ChromaticityPlot[obj] and ChromaticityPlot[obj, refspace] (obj: a colour space name, a colour, an "
          "image or a list of them) draw the CIE spectral locus (the 1931 2° standard observer, 380-780 nm at 5 nm, "
          "from the official CIE tables; Method -> {\"Observer\" -> 10} uses the 1964 10° observer) closed by the "
          "purple line, as a light gray Polygon with a black boundary, in the CIE 1931 xy (\"CIE31\", default) or "
          "CIE 1976 u'v' (\"CIE76\"; u' = 4x/(-2x + 12y + 3), v' = 9y/(-2x + 12y + 3)) diagram. RGB colour spaces "
          "(\"RGB\"/\"sRGB\", \"HSB\" and \"CMYK\" share the sRGB gamut; \"AdobeRGB\", \"AppleRGB\", \"CIERGB\", "
          "\"ProPhotoRGB\", \"WideGamutRGB\") are triangles of their primaries (Lindbloom's working space table), "
          "Bradford-adapted to the D50 white of ColorConvert's XYZ so that they agree with converted colours; "
          "\"Grayscale\" is the white point. Colours and image pixels (at most MaxPlotPoints, default 2000, by a "
          "regular stride) are black Points at their xy (or u'v') chromaticity from ColorConvert's XYZ; colours "
          "with X + Y + Z = 0 are omitted. {ref, l} is accepted (the chromaticity does not depend on the luminance "
          "slice). XYZ, LAB, LCH and LUV gamuts, ColorProfileData and ColorsNear raise NotImplementedError. Frame "
          "True, PlotRange {{0, 0.8}, {0, 0.9}} (CIE31) or {{0, 0.65}, {0, 0.6}} (CIE76). Check: reference (sRGB "
          "red gives the same point as the adapted triangle vertex; the 520 nm locus point x = 0.0743, y = 0.8338) "
          "in tests.")
def _chromaticityplot(obj: Expr, *rest: String | ListOf[Expr] | Option) -> Expr:
    fname = "ChromaticityPlot"
    pos, o = _split_any(rest)
    _opt_check(pos, fname, 1)
    ref, observer, maxpts, gopts = _chrom_options(pos, o, fname, ("CIE31", "CIE76"), "CIE31")
    spaces, points = _chrom_parse(_objects(obj), fname, maxpts)
    tf = (lambda x, y: (x, y)) if ref == "CIE31" else _uv
    locus = [tf(*p) for p in spectral_locus(observer)]
    content = [Expr("GrayLevel", 0.9), Expr("Polygon", L(*[pt(*p) for p in locus])),
               Expr("GrayLevel", 0), Expr("Line", L(*[pt(*p) for p in locus + [locus[0]]]))]
    for k, sp in enumerate(spaces):
        xy = [tf(*p) for p in _space_xy(sp)]
        c = _SPACE_COLORS[k % len(_SPACE_COLORS)]
        if len(xy) == 1:
            content += [Expr("RGBColor", *c), Expr("PointSize", 0.02), Expr("Point", pt(*xy[0]))]
        else:
            content += [Expr("RGBColor", *c), Expr("Line", L(*[pt(*p) for p in xy + [xy[0]]]))]
    xs = []
    for X, Y, Z in points:
        s = X + Y + Z
        if s > 0:
            xs.append(pt(*tf(X / s, Y / s)))
    if xs:
        content += [Expr("GrayLevel", 0), Expr("PointSize", 0.012), Expr("Point", L(*xs))]
    prange = L(L(0, 0.8), L(0, 0.9)) if ref == "CIE31" else L(L(0, 0.65), L(0, 0.6))
    from .wl_plots2d import finish_graphics
    labels = L("x", "y") if ref == "CIE31" else L("u'", "v'")
    return finish_graphics(L(*content), gopts, [("Frame", True), ("FrameLabel", labels), ("PlotRange", prange),
                                                ("AspectRatio", Symbol("Automatic"))])


def _chrom_options(pos, o, fname, refs, default):
    ref = pos[0] if pos else default
    if K.is_list(ref) and len(ref.args) == 2 and isinstance(ref.args[0], str):
        _num(ref.args[1], fname, "the luminance", lo=0)
        ref = ref.args[0]
    if not isinstance(ref, str):
        if isinstance(ref, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the reference colour space must be a string")
    if ref not in refs:
        raise ValueError(f"{fname}: the reference colour space must be one of {', '.join(refs)}")
    observer, maxpts, gopts = 2, 2000, []
    for k, v in o.items():
        if k == "Method":
            if _is_sym(v, "Automatic"):
                continue
            rules = v.args if K.is_list(v) else (v,)
            for ru in rules:
                if isinstance(ru, Expr) and ru.head == "Rule" and ru.args[0] == "Observer" and ru.args[1] in (2, 10):
                    observer = ru.args[1]
                else:
                    raise ValueError(f"{fname}: Method must be Automatic or {{\"Observer\" -> 2 | 10}}")
        elif k == "MaxPlotPoints":
            if not (isinstance(v, int) and v > 0):
                raise ValueError(f"{fname}: MaxPlotPoints must be a positive integer")
            maxpts = v
        elif k in ("Appearance", "BoundaryStyle", "FillingStyle", "PlotLegends", "PlotPoints", "PlotStyle",
                   "PlotTheme", "WhitePoint", "Mesh", "MeshStyle"):
            if not (_is_sym(v, "Automatic") or _is_sym(v, "None")):
                raise NotImplementedError(f"{fname}: the option {k} is not supported")
        else:
            gopts.append(Expr("Rule", Symbol(k), v))
    return ref, observer, maxpts, gopts


def _ref_coords(xyz, ref):
    """Coordinates of an XYZ colour in a 3D reference space."""
    from .wl_graphics_color import _xyz_to_lab, _xyz_to_luv, _lab_to_lch, _xyz_to_rgb, _rgb_to_hsb
    X, Y, Z = xyz
    if ref in ("XYZ",):
        return (X, Y, Z)
    if ref == "CIE31":
        s = X + Y + Z
        return (X / s, Y / s, Y) if s > 0 else None
    if ref == "CIE76":
        s = X + Y + Z
        if s <= 0:
            return None
        u, v = _uv(X / s, Y / s)
        return (u, v, Y)
    if ref == "LAB":
        return _xyz_to_lab(X, Y, Z)
    if ref == "LCH":
        return _lab_to_lch(*_xyz_to_lab(X, Y, Z))
    if ref == "LUV":
        return _xyz_to_luv(X, Y, Z)
    rgb = _xyz_to_rgb(X, Y, Z)
    return rgb if ref == "RGB" else _rgb_to_hsb(*rgb)


GAMUT_POINTS = 9


@fn("ChromaticityPlot3D", area=AREA, status="added",
    notes="ChromaticityPlot3D[obj] and ChromaticityPlot3D[obj, refspace] draw colour spaces, colours and images in "
          "a 3D reference space: \"CIE31\" (default; coordinates {x, y, Y}), \"CIE76\" ({u', v', Y}), \"XYZ\", "
          "\"LAB\", \"LCH\", \"LUV\", \"RGB\" or \"HSB\", using ColorConvert's conversions (XYZ relative to D50). "
          "The sRGB-gamut spaces \"RGB\", \"sRGB\", \"HSB\" and \"CMYK\" are drawn as the surface of the RGB cube "
          "(9 x 9 samples per face, PlotPoints -> n sets n) mapped into the reference space as Polygons with "
          "VertexColors; colours and image pixels (at most MaxPlotPoints, default 2000) are black Points. Other "
          "named RGB spaces, XYZ/LAB/LCH/LUV gamuts, ColorProfileData and ColorsNear raise NotImplementedError. "
          "Axes True, BoxRatios {1, 1, 1}. Check: reference (the cube corner for red maps to ColorConvert[Red, "
          "ref]) in tests.")
def _chromaticityplot3d(obj: Expr, *rest: String | ListOf[Expr] | Option) -> Expr:
    fname = "ChromaticityPlot3D"
    pos, o = _split_any(rest)
    _opt_check(pos, fname, 1)
    n = GAMUT_POINTS
    if "PlotPoints" in o and not _is_sym(o["PlotPoints"], "Automatic"):
        n = o.pop("PlotPoints")
        if not (isinstance(n, int) and n >= 2):
            raise ValueError(f"{fname}: PlotPoints must be an integer of at least 2")
    ref, _obs, maxpts, gopts = _chrom_options(pos, o, fname, ("CIE31", "CIE76", "XYZ", "LAB", "LCH", "LUV", "RGB",
                                                             "HSB"), "CIE31")
    objs = _objects(obj)
    for ob in objs:
        if isinstance(ob, str) and ob in RGB_SPACES and ob != "sRGB":
            raise NotImplementedError(f"{fname}: the {ob} gamut is not supported in 3D")
    spaces, points = _chrom_parse(objs, fname, maxpts)
    content = []
    t = np.linspace(0, 1, n)
    for sp in spaces:
        if sp == "Grayscale":
            line = [_ref_coords(_to_xyz("RGB", (v, v, v)), ref) for v in t]
            content += [Expr("GrayLevel", 0.3), Expr("Line", L(*[pt(*p) for p in line if p is not None]))]
            continue
        polys = []
        for ax in range(3):
            for side in (0.0, 1.0):
                for i in range(n - 1):
                    for j in range(n - 1):
                        quad = []
                        for a, b in ((t[i], t[j]), (t[i + 1], t[j]), (t[i + 1], t[j + 1]), (t[i], t[j + 1])):
                            c = [a, b]
                            c.insert(ax, side)
                            quad.append(tuple(c))
                        crd = [_ref_coords(_to_xyz("RGB", c), ref) for c in quad]
                        if any(p is None for p in crd):
                            continue
                        polys.append(Expr("Polygon", L(*[pt(*p) for p in crd]),
                                          Expr("Rule", Symbol("VertexColors"),
                                               L(*[Expr("RGBColor", *[fl(v) for v in c]) for c in quad]))))
        content += [Expr("EdgeForm"), *polys]
    pts = [_ref_coords(p, ref) for p in points]
    pts = [pt(*p) for p in pts if p is not None]
    if pts:
        content += [Expr("GrayLevel", 0), Expr("PointSize", 0.012), Expr("Point", L(*pts))]
    from .wl_plots2d import finish_graphics
    return finish_graphics(L(*content), gopts, [("Axes", True), ("BoxRatios", L(1, 1, 1))], dim=3)


# ───────────────────────────── HighlightImage ─────────────────────────────

HIGHLIGHT_COLOR = (1.0, 0.0, 0.0)      # default foreground colour (Red)
HIGHLIGHT_OPACITY = 0.5                # default foreground opacity
_VECTOR_ROI = ("Point", "Line", "Circle")
_WRAPPERS = ("Annotation", "Button", "EventHandler", "Hyperlink", "ImageMarker", "Labeled", "Legended", "PopupWindow",
             "StatusArea", "Tooltip")


def _directive_color(x):
    from .wl_graphics_color import to_rgba
    return to_rgba(x)


def _roi_items(spec, fname, im, style, out):
    """Flatten a foreground spec into [(kind, payload, style)]; kind 'mask' (h x w bool) or 'prim' (primitive)."""
    h, w = im.a.shape[:2]
    if K.is_list(spec):
        arr = WI._numeric_array(spec, fname) if spec.args and all(K.is_list(a) for a in spec.args) else None
        if arr is not None and arr.ndim == 2 and arr.shape == (h, w):
            out.append(("mask", arr != 0, style))
            return
        if arr is not None and arr.ndim == 2 and arr.shape[1] == 2:
            out.append(("prim", Expr("Point", spec), style))
            return
        st = dict(style)
        for item in spec.args:
            if _directive_color(item) is not None:
                st["color"] = _directive_color(item)
            elif isinstance(item, Expr) and item.head == "Opacity" and len(item.args) == 1:
                st["opacity"] = _num(item.args[0], fname, "the opacity", lo=0)
            elif isinstance(item, Expr) and item.head in ("PointSize", "Thickness", "AbsoluteThickness",
                                                          "AbsolutePointSize", "Dashing", "EdgeForm", "FaceForm",
                                                          "Directive"):
                st.setdefault("directives", [])
                st["directives"] = st["directives"] + [item]
            elif _is_sym(item, "Automatic"):
                st = {}
            elif item == "Boundary" or (K.is_list(item) and item.args and item.args[0] == "Boundary"):
                st["boundary"] = 1 if item == "Boundary" else int(_num(item.args[1], fname, "the thickness", lo=1))
            elif item == "Blur" or (K.is_list(item) and item.args and item.args[0] == "Blur"):
                st["blur"] = 2.0 if item == "Blur" else _num(item.args[1], fname, "the blur range", lo=0)
            elif isinstance(item, Expr) and item.head == "Transparent" or _is_sym(item, "Transparent"):
                st["opacity"] = 0.0
            else:
                _roi_items(item, fname, im, st, out)
        return
    if isinstance(spec, Expr) and spec.head in IMAGE_HEADS:
        m = parse_image(spec, fname)
        if m.d3 or m.a.shape[:2] != (h, w):
            raise ValueError(f"{fname}: a mask image must have the dimensions of the image")
        out.append(("mask", np.abs(m.color()).max(-1) > 0, style))
        return
    if isinstance(spec, Expr) and spec.head in _WRAPPERS + ("Style", "Association", "Graphics"):
        raise NotImplementedError(f"{fname}: {spec.head} regions of interest are not supported")
    if isinstance(spec, Expr) and spec.head in _VECTOR_ROI:
        out.append(("prim", spec, style))
        return
    if isinstance(spec, Expr) and isinstance(spec.head, str) and spec.head[:1].isupper() and spec.args:
        try:
            out.append(("mask", region_mask(spec, w, h, fname), style))
            return
        except TypeError:
            pass
    if isinstance(spec, Symbol) or (isinstance(spec, Expr) and not K.is_list(spec)):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(spec)} is not a region of interest")


def _boundary(mask, t):
    """Pixels of mask within t pixels (city-block erosion steps) of its complement."""
    inner = mask.copy()
    for _ in range(t):
        p = np.pad(inner, 1, mode="constant", constant_values=False)
        inner = p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]
    return mask & ~inner


def _blur(rgb, sigma):
    return gaussian_smooth(rgb, 2, sigma, ["Fixed", "Fixed"], "HighlightImage") if sigma > 0 else rgb


@fn("HighlightImage", area=AREA, status="added",
    notes="HighlightImage[image, roi], [image, {spec, ...}] and [image, fg, bgstyle] return Graphics[{Raster[...], "
          "...}] with the image on {{0, 0}, {w, h}} (standard image coordinates) and PlotRange {{0, w}, {0, h}}. "
          "roi: a mask image or matrix of the image size (nonzero pixels; label matrices give one region for all "
          "labels), a list of positions {{x, y}, ...} (drawn as Point), Point/Line/Circle primitives (drawn as "
          "primitives) or another 2D region (pixel centres inside it). Styles in a list apply to the rest of that "
          "list: a colour (default Red), Opacity[a] (default 0.5), directives (PointSize, Thickness, ...), "
          "\"Boundary\"/{\"Boundary\", t} (only the t-pixel inner boundary of mask regions, default 1), "
          "\"Blur\"/{\"Blur\", r} (Gaussian σ = r, default 2, inside the region instead of colouring) and Automatic "
          "(reset). Mask regions are blended into the raster: pixel (1 - a) value + a colour. bgstyle acts on the "
          "complement of all mask regions and sets the default foreground opacity to 0: \"Darken\"/{\"Darken\", a} "
          "(blend with black, default 0.5), \"Lighten\"/{\"Lighten\", a} (with white, 0.5), \"Desaturate\"/"
          "{\"Desaturate\", s} (blend with the gray level by s, default 1), \"Blur\"/{\"Blur\", r}, \"Remove\" "
          "(alpha 0) and {\"Remove\", colour}. Wrappers (Annotation, Labeled, Tooltip, ...), associations, Graphics "
          "and function rois, function bgstyles and ImageLabels/ImageLegends raise NotImplementedError. Check: "
          "exact (blended pixel values compared with the formula in tests).")
def _highlightimage(x: Expr, fg: Expr | ListOf[Expr], *rest: String | ListOf[Expr] | Option) -> Expr:
    fname = "HighlightImage"
    pos, o = _split_any(rest)
    _opt_check(pos, fname, 1)
    gopts = []
    for k, v in o.items():
        if k in ("ImageLabels", "ImageLegends"):
            if not (_is_sym(v, "None") or _is_sym(v, "Automatic")):
                raise NotImplementedError(f"{fname}: the option {k} is not supported")
        elif k == "DataRange":
            if not _is_sym(v, "Full"):
                raise NotImplementedError(f"{fname}: DataRange other than Full is not supported")
        else:
            gopts.append(Expr("Rule", Symbol(k), v))
    im, rgb, alpha, _single = _rgb_image(x, fname)
    h, w = rgb.shape[:2]
    bg = pos[0] if pos else None
    default_op = HIGHLIGHT_OPACITY if bg is None else 0.0
    items = []
    _roi_items(fg if K.is_list(fg) else L(fg), fname, im, {}, items)
    out = rgb.copy()
    alpha_out = np.ones((h, w)) if alpha is None else alpha[..., 0].copy()
    union = np.zeros((h, w), dtype=bool)
    prims = []
    for kind, payload, st in items:
        color = st.get("color", HIGHLIGHT_COLOR)[:3]
        op = st.get("opacity", default_op)
        if kind == "prim":
            dirs = st.get("directives", [])
            prims.append(L(Expr("RGBColor", *[fl(c) for c in color]), Expr("Opacity", fl(max(op, 0.5) if bg is None
                                                                                          else 1.0)),
                           *dirs, payload))
            continue
        mask = payload
        union |= mask
        if "boundary" in st:
            mask = _boundary(mask, st["boundary"])
        if "blur" in st:
            bl = _blur(rgb, st["blur"])
            out[mask] = bl[mask]
        elif op > 0:
            out[mask] = (1 - op) * out[mask] + op * np.array(color)
    if bg is not None:
        bgm = ~union
        name, par = (bg, None) if isinstance(bg, str) else \
            ((bg.args[0], bg.args[1]) if K.is_list(bg) and len(bg.args) == 2 and isinstance(bg.args[0], str)
             else (None, None))
        if name == "Darken":
            a = 0.5 if par is None else _num(par, fname, "the darkening amount", lo=0)
            out[bgm] = (1 - a) * out[bgm]
        elif name == "Lighten":
            a = 0.5 if par is None else _num(par, fname, "the lightening amount", lo=0)
            out[bgm] = (1 - a) * out[bgm] + a
        elif name == "Desaturate":
            s = 1.0 if par is None else _num(par, fname, "the desaturation", lo=0)
            g = WI.convert_array(out, "RGB", "Grayscale")
            out[bgm] = (1 - s) * out[bgm] + s * np.repeat(g, 3, axis=-1)[bgm]
        elif name == "Blur":
            bl = _blur(rgb, 2.0 if par is None else _num(par, fname, "the blur range", lo=0))
            out[bgm] = bl[bgm]
        elif name == "Remove":
            if par is None:
                alpha_out[bgm] = 0.0
            else:
                c = _directive_color(par)
                if c is None:
                    raise ValueError(f"{fname}: {{\"Remove\", colour}} needs a colour")
                out[bgm] = np.array(c[:3])
        elif name is None and not isinstance(bg, str) and not K.is_list(bg):
            raise NotImplementedError(f"{fname}: function background styles are not supported")
        else:
            raise ValueError(f"{fname}: unknown background style {_short(bg)}")
    data = np.clip(out, 0, 1)
    if alpha is not None or (alpha_out < 1).any():
        data = np.concatenate([data, alpha_out[..., None]], axis=-1)
    from .wl_plots2d import finish_graphics
    return finish_graphics(L(raster(data), *prims), gopts, [("PlotRange", L(L(0, w), L(0, h))),
                                                           ("ImageSize", w)])


# ───────────────────────────── ImageCollage ─────────────────────────────

COLLAGE_BG = (0.2, 0.2, 0.2)


def _collage_items(spec, fname):
    """[(weight or ('Scaled', s), Img)]."""
    if isinstance(spec, Expr) and spec.head == "Rule" and K.is_list(spec.args[1]):
        wgt = spec.args[0]
        return [(_weight(wgt, fname), _image(v, fname, allow3d=False)) for v in spec.args[1].args]
    if isinstance(spec, Expr) and spec.head == "Association":
        raise NotImplementedError(f"{fname}: associations are not supported; give {{w1 -> image1, ...}}")
    if not K.is_list(spec):
        if isinstance(spec, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list of images")
    if not spec.args:
        raise ValueError(f"{fname}: the list of images is empty")
    out = []
    for v in spec.args:
        if isinstance(v, Expr) and v.head == "Rule":
            out.append((_weight(v.args[0], fname), _image(v.args[1], fname, allow3d=False)))
        else:
            out.append((1.0, _image(v, fname, allow3d=False)))
    return out


def _weight(wgt, fname):
    if isinstance(wgt, Expr) and wgt.head == "Scaled" and len(wgt.args) == 1:
        return ("Scaled", _num(wgt.args[0], fname, "the weight", lo=0, strict=True))
    return _num(wgt, fname, "the weight", lo=0, strict=True)


def _fit(im, rgb, alpha, W, H, fitting, bg):
    """(H, W, 4) RGBA tile of an image fitted into a W x H cell."""
    h, w = rgb.shape[:2]
    a = np.concatenate([rgb, alpha if alpha is not None else np.ones((h, w, 1))], axis=-1)
    tile = np.zeros((H, W, 4))
    tile[..., :3] = bg
    if fitting == "Stretch":
        return WI.resize_array(a, [H, W], "Bilinear")
    s = (max if fitting == "Fill" else min)(W / w, H / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = WI.resize_array(a, [nh, nw], "Bilinear")
    if fitting == "Fill":
        y0, x0 = (nh - H) // 2, (nw - W) // 2
        return r[y0:y0 + H, x0:x0 + W]
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    tile[y0:y0 + nh, x0:x0 + nw] = r
    return tile


def _justified_rows(aspects, areas, nrows):
    """Split images (in order) into nrows rows of about equal nominal width, where an image of nominal area A and
    aspect ratio a (width/height) has nominal width Sqrt[A a]; returns the lists of indices per row."""
    nominal = [math.sqrt(A * a) for A, a in zip(areas, aspects)]
    total = sum(nominal)
    rows, cur, acc = [], [], 0.0
    for i in range(len(aspects)):
        cur.append(i)
        acc += nominal[i]
        if len(rows) < nrows - 1 and acc >= total * (len(rows) + 1) / nrows - 1e-9:
            rows.append(cur)
            cur = []
    if cur:
        rows.append(cur)
    return rows


@fn("ImageCollage", area=AREA, status="added",
    notes="ImageCollage[{image1, ...}], [{w1 -> image1, ...}], [w -> {image1, ...}], [..., fitting] and [..., "
          "fitting, size] compose 2D images into one RGB image (alpha kept when an input has alpha) on the "
          "Background (default GrayLevel[0.2]). Method \"Rows\" (Automatic; justified rows): images in order are "
          "split into Round[Sqrt[n]] rows of about equal width and each row is scaled to the collage width, so "
          "aspect ratios are kept exactly. Weights w (default 1; Scaled[s]: s times the pixel area) set nominal "
          "areas that decide the split into rows (nominal width Sqrt[area x aspect]); within a justified row every "
          "image has the row height, so final areas are not proportional to the weights (a partial "
          "implementation of the documented weighting; \"Grid\" ignores weights). \"Columns\" is the "
          "transposed layout; \"Grid\": Ceiling[Sqrt[n]] columns of equal cells of the largest image's size, each "
          "image fitted by fitting (\"Fit\" default: keep aspect and pad; \"Fill\": cover and crop the centre; "
          "\"Stretch\": resample). size: width w or {w, h} (h only for \"Grid\"); the default width is "
          "Ceiling[Sqrt[n]] times the median image width. ImagePadding -> m puts m background pixels around each "
          "image (default None). Resampling is bilinear. \"ClosestPacking\" raises NotImplementedError. Check: "
          "reference (two equal images give side-by-side copies; pixel values of a one-image collage equal the "
          "rescaled image) in tests.")
def _imagecollage(spec: ListOf[Expr] | Expr, *rest: String | PosInteger | ListOf[PosInteger] | Option) -> Expr:
    fname = "ImageCollage"
    pos, o = opts_split(rest, {"Background", "ImagePadding", "ImageSize", "Method", "Padding", "Resampling"}, fname)
    _opt_check(pos, fname, 2)
    items = _collage_items(spec, fname)
    fitting = pos[0] if pos else "Fit"
    if _is_sym(fitting, "Automatic"):
        fitting = "Fit"
    if fitting not in ("Fit", "Fill", "Stretch"):
        raise ValueError(f"{fname}: fitting must be \"Fit\", \"Fill\" or \"Stretch\"")
    meth = o.get("Method", AUTO)
    meth = "Rows" if _is_sym(meth, "Automatic") else meth
    if meth == "ClosestPacking":
        raise NotImplementedError(f"{fname}: Method \"ClosestPacking\" is not supported")
    if meth not in ("Rows", "Columns", "Grid"):
        raise ValueError(f"{fname}: Method must be \"Rows\", \"Columns\", \"Grid\" or \"ClosestPacking\"")
    from .wl_graphics_color import to_rgba
    bgc = to_rgba(_ev(o.get("Background", Expr("GrayLevel", 0.2))))
    if bgc is None:
        raise ValueError(f"{fname}: Background must be a colour")
    bg = np.array(bgc[:3])
    pad = o.get("ImagePadding", Symbol("None"))
    if _is_sym(pad, "None"):
        m = 0
    elif isinstance(pad, int) and not isinstance(pad, bool) and pad >= 0:
        m = pad
    else:
        raise NotImplementedError(f"{fname}: ImagePadding must be None or a non-negative integer here")
    parts = []
    for wgt, im in items:
        src = WI.source_space(im, fname)
        col = im.color()
        rgb = np.clip(WI.convert_array(col, src, "RGB") if src != "RGB" else col, 0, 1)
        parts.append((wgt, im, rgb, im.a[..., -1:] if im.alpha else None))
    has_alpha = any(p[3] is not None for p in parts)
    n = len(parts)
    size = pos[1] if len(pos) > 1 else AUTO
    Wt = Ht = None
    if K.is_list(size) and len(size.args) == 2:
        Wt = int(_num(size.args[0], fname, "the width", lo=1))
        Ht = int(_num(size.args[1], fname, "the height", lo=1))
    elif not _is_sym(size, "Automatic"):
        Wt = int(_num(size, fname, "the width", lo=1))
    ncol = int(math.ceil(math.sqrt(n)))
    if Wt is None:
        Wt = int(ncol * np.median([p[2].shape[1] for p in parts]))
    if meth == "Grid":
        cw = max(p[2].shape[1] for p in parts)
        ch = max(p[2].shape[0] for p in parts)
        nrow = int(math.ceil(n / ncol))
        cellw = max(1, Wt // ncol - 2 * m)
        cellh = max(1, (Ht // nrow - 2 * m) if Ht else int(round(cellw * ch / cw)))
        canvas = np.zeros((nrow * (cellh + 2 * m), ncol * (cellw + 2 * m), 4))
        canvas[..., :3] = bg
        canvas[..., 3] = 1
        for k, (_, im, rgb, al) in enumerate(parts):
            r, c = divmod(k, ncol)
            y0, x0 = r * (cellh + 2 * m) + m, c * (cellw + 2 * m) + m
            canvas[y0:y0 + cellh, x0:x0 + cellw] = _fit(im, rgb, al, cellw, cellh, fitting, bg)
    else:
        transpose = meth == "Columns"
        if transpose:
            parts = [(wg, im, rgb.transpose(1, 0, 2), None if al is None else al.transpose(1, 0, 2))
                     for wg, im, rgb, al in parts]
        aspects = [p[2].shape[1] / p[2].shape[0] for p in parts]
        areas = []
        for (wg, _, rgb, _), a in zip(parts, aspects):
            areas.append(wg[1] * rgb.shape[0] * rgb.shape[1] if isinstance(wg, tuple) else wg)
        if not any(isinstance(p[0], tuple) for p in parts):
            mean_area = float(np.mean([p[2].shape[0] * p[2].shape[1] for p in parts]))
            areas = [a * mean_area for a in areas]
        nrows = max(1, int(round(math.sqrt(n))))
        rows = _justified_rows(aspects, areas, nrows)
        rows_px = []
        for idx in rows:
            avail = Wt - 2 * m * len(idx)
            hh = max(1, int(round(avail / sum(aspects[i] for i in idx))))
            rows_px.append((idx, hh, avail))
        Htot = sum(hh + 2 * m for _, hh, _ in rows_px)
        canvas = np.zeros((Htot, Wt, 4))
        canvas[..., :3] = bg
        canvas[..., 3] = 1
        y = 0
        for idx, hh, avail in rows_px:
            x = 0
            widths = [max(1, int(round(aspects[i] * hh))) for i in idx]
            widths[-1] = max(1, avail - sum(widths[:-1]))
            for i, ww in zip(idx, widths):
                _, im, rgb, al = parts[i]
                tile = _fit(im, rgb, al, ww, hh, "Stretch", bg)
                canvas[y + m:y + m + hh, x + m:x + m + ww] = tile[:, :min(ww, Wt - x - m)]
                x += ww + 2 * m
            y += hh + 2 * m
        if transpose:
            canvas = canvas.transpose(1, 0, 2)
    if not has_alpha:
        canvas = canvas[..., :3]
    typ = WI.larger_type(*[im.type for _, im in items])
    typ = "Real32" if typ == "Bit" else typ
    return image_expr(WI.Img(quantize(np.clip(canvas, 0, 1), typ), typ, "RGB", False))


# ───────────────────────────── ImageEffect ─────────────────────────────

SEPIA = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]])   # Microsoft sepia matrix
# Machado, Oliveira & Fernandes (2009), severity 1.0, applied to linear RGB
MACHADO = {
    "Protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "Deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "Tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}
EFFECTS = ("ColorTones", "Deuteranopia", "Embossing", "GaussianNoise", "Jitter", "MotionBlur", "Noise",
           "PoissonNoise", "Posterization", "Protanopia", "SaltPepperNoise", "Sepia", "Solarization", "Tritanopia",
           "Vignette")
UNSUPPORTED_EFFECTS = ("ColorBoosting", "Decolorization", "DetailEnhancing", "Charcoal", "Comics", "EdgeStylization",
                       "OilPainting", "Stippling", "FadedFrame", "TornFrame", "Frame", "NestedComposition")


def _srgb_lin(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_gam(v):
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055)


def _rng_of(o, fname):
    seed = o.get("RandomSeeding", AUTO)
    if _is_sym(seed, "Automatic"):
        return WI._rng()
    if isinstance(seed, int) and not isinstance(seed, bool):
        return np.random.default_rng(seed)
    if isinstance(seed, str):
        return np.random.default_rng(abs(hash(seed)) % (2 ** 63))
    raise ValueError(f"{fname}: RandomSeeding must be Automatic, an integer or a string")


def _line_kernel(d, alpha):
    """Normalised line kernel of length d pixels at angle alpha (counterclockwise from +x), 8 samples per pixel."""
    r = int(math.ceil(d / 2))
    k = np.zeros((2 * r + 1, 2 * r + 1))
    for s in np.linspace(-d / 2, d / 2, max(2, int(8 * d) + 1)):
        x, y = s * math.cos(alpha), s * math.sin(alpha)
        k[int(round(r - y)), int(round(r + x))] += 1
    return k / k.sum()


def _conv2(a, k):
    """Correlate the spatial axes of (h, w, c) with k, repeating edge pixels."""
    from .wl_imageproc import correlate_nd
    return correlate_nd(a, k, "Fixed")


@fn("ImageEffect", area=AREA, status="added",
    notes="ImageEffect[image, \"effect\"] and ImageEffect[image, {\"effect\", params...}] (2D images; results are "
          "RGB, or grayscale for single-channel images, in the image's type, Real32 for Bit images, clipped to "
          "0..1; alpha kept); ImageEffect[] lists the supported effects. Effects and defaults: \"Sepia\" (the "
          "Microsoft sepia matrix), {\"Posterization\", n} (n levels per channel, Round[v (n - 1)]/(n - 1), default "
          "4), {\"Solarization\", t} (v > t -> 1 - v, default 0.5), {\"ColorTones\", {c1, c2, ...}} (gray level "
          "mapped piecewise linearly through the colours), \"Protanopia\"/\"Deuteranopia\"/\"Tritanopia\" (Machado, "
          "Oliveira & Fernandes 2009 severity-1 matrices on linear sRGB), {\"MotionBlur\", d, α} (normalised line "
          "kernel of length d pixels at angle α, defaults 10 and 0), {\"Embossing\", x, α} (0.5 + x times the "
          "difference of the pixel and its neighbour at angle α + Pi, defaults 1 and Pi/4), {\"Jitter\", r} (each "
          "pixel from a uniformly random position within range r, default 2), {\"Vignette\", r, colour} (blend "
          "toward colour with weight ((ρ - r)/(1 - r))^2 for ρ > r, ρ the distance from the centre over the half "
          "diagonal; defaults 0.3 and Black), {\"Noise\", a} (uniform on [-a, a], default 0.1), "
          "{\"GaussianNoise\", σ} (default 0.1), {\"PoissonNoise\", a} (as documented: σ = 100 (1 - a)/a, p -> "
          "PoissonDistribution[p σ]/σ, default 0.5) and {\"SaltPepperNoise\", f | {fs, fp}} (default 0.05: fraction "
          "f of the pixels set to white or black with equal probability). RandomSeeding -> Automatic (the session "
          "generator), an integer or a string. ColorBoosting, Decolorization, DetailEnhancing, Charcoal, Comics, "
          "EdgeStylization, OilPainting, Stippling, the frame effects and NestedComposition raise "
          "NotImplementedError. Check: reference (sepia and Machado matrices applied to pure colours; noise means "
          "and variances against their distributions) in tests.")
def _imageeffect(*args: Expr | String | ListOf[Expr] | Option) -> Expr | ListOf[String]:
    fname = "ImageEffect"
    pos, o = opts_split(args, {"RandomSeeding"}, fname)
    if not pos:
        return L(*EFFECTS)
    if len(pos) != 2:
        raise TypeError(f"{fname}: expected ImageEffect[image, effect]")
    x, eff = pos
    im, rgb, alpha, single = _rgb_image(x, fname)
    if isinstance(eff, str):
        name, ps = eff, []
    elif K.is_list(eff) and eff.args and isinstance(eff.args[0], str):
        name, ps = eff.args[0], list(eff.args[1:])
    elif isinstance(eff, Symbol):
        raise Unevaluated()
    else:
        raise TypeError(f"{fname}: the effect must be \"name\" or {{\"name\", params...}}")
    if name in UNSUPPORTED_EFFECTS:
        raise NotImplementedError(f"{fname}: the effect {name!r} is not supported")
    if name not in EFFECTS:
        raise ValueError(f"{fname}: unknown effect {name!r}")

    def par(i, default, what, lo=None):
        return default if len(ps) <= i or _is_sym(ps[i], "Automatic") else _num(ps[i], fname, what, lo=lo)
    h, w = rgb.shape[:2]
    out = rgb
    if name == "Sepia":
        out = rgb @ SEPIA.T
    elif name == "Posterization":
        n = par(0, 4, "the number of levels", lo=2)
        if n != int(n):
            raise ValueError(f"{fname}: the number of levels must be an integer of at least 2")
        out = np.round(rgb * (n - 1)) / (n - 1)
    elif name == "Solarization":
        t = par(0, 0.5, "the threshold")
        out = np.where(rgb > t, 1 - rgb, rgb)
    elif name == "ColorTones":
        if not ps or not K.is_list(ps[0]) or len(ps[0].args) < 2:
            raise ValueError(f"{fname}: ColorTones needs a list of at least two colours")
        cols = []
        for c in ps[0].args:
            v = _directive_color(_ev(c))
            if v is None:
                raise ValueError(f"{fname}: {_short(c)} is not a colour")
            cols.append(v[:3])
        cols = np.array(cols)
        g = WI.convert_array(rgb, "RGB", "Grayscale")[..., 0] * (len(cols) - 1)
        i0 = np.clip(np.floor(g).astype(int), 0, len(cols) - 2)
        f = (g - i0)[..., None]
        out = (1 - f) * cols[i0] + f * cols[i0 + 1]
    elif name in MACHADO:
        out = _srgb_gam(_srgb_lin(rgb) @ MACHADO[name].T)
    elif name == "MotionBlur":
        d = par(0, 10, "the blur size", lo=0)
        a = par(1, 0.0, "the angle")
        out = _conv2(rgb, _line_kernel(d, a)[::-1, ::-1]) if d > 0 else rgb
    elif name == "Embossing":
        x_ = par(0, 1.0, "the amount")
        a = par(1, math.pi / 4, "the angle")
        di, dj = -int(round(math.sin(a + math.pi))), int(round(math.cos(a + math.pi)))
        p = np.pad(rgb, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nb = p[1 + di:1 + di + h, 1 + dj:1 + dj + w]
        g = (rgb - nb).mean(-1, keepdims=True)
        out = np.repeat(0.5 + x_ * g, 3, axis=-1)
    elif name == "Jitter":
        r = int(par(0, 2, "the range", lo=0))
        rng = _rng_of(o, fname)
        ii = np.clip(np.arange(h)[:, None] + rng.integers(-r, r + 1, (h, w)), 0, h - 1)
        jj = np.clip(np.arange(w)[None, :] + rng.integers(-r, r + 1, (h, w)), 0, w - 1)
        out = rgb[ii, jj]
    elif name == "Vignette":
        r = par(0, 0.3, "the radius", lo=0)
        c = np.zeros(3)
        if len(ps) > 1:
            cc = _directive_color(_ev(ps[1]))
            if cc is None:
                raise ValueError(f"{fname}: {_short(ps[1])} is not a colour")
            c = np.array(cc[:3])
        yy, xx = np.mgrid[0:h, 0:w]
        rho = np.hypot(yy + 0.5 - h / 2, xx + 0.5 - w / 2) / math.hypot(h / 2, w / 2)
        wt = np.clip((rho - r) / (1 - r), 0, 1) ** 2 if r < 1 else np.zeros_like(rho)
        out = (1 - wt[..., None]) * rgb + wt[..., None] * c
    else:
        rng = _rng_of(o, fname)
        base = rgb[..., :1] if single else rgb
        if name == "Noise":
            a = par(0, 0.1, "the amplitude", lo=0)
            out = base + rng.uniform(-a, a, base.shape)
        elif name == "GaussianNoise":
            s = par(0, 0.1, "the standard deviation", lo=0)
            out = base + rng.normal(0, s, base.shape)
        elif name == "PoissonNoise":
            a = par(0, 0.5, "the amount", lo=0)
            if not 0 < a <= 1:
                raise ValueError(f"{fname}: the Poisson noise amount must be in (0, 1]")
            sig = 100 * (1 - a) / a
            out = rng.poisson(base * sig) / sig if sig > 0 else base
        else:
            if ps and K.is_list(ps[0]) and len(ps[0].args) == 2:
                fs = _num(ps[0].args[0], fname, "the salt fraction", lo=0)
                fp = _num(ps[0].args[1], fname, "the pepper fraction", lo=0)
            else:
                f = par(0, 0.05, "the fraction", lo=0)
                fs = fp = f / 2
            if fs + fp > 1:
                raise ValueError(f"{fname}: the noise fractions must add up to at most 1")
            u = rng.random((h, w))
            out = base.copy()
            out[u < fs] = 1.0
            out[(u >= fs) & (u < fs + fp)] = 0.0
        if single:
            return _effect_out(im, np.clip(out, 0, 1), alpha, True)
    if single:
        out = WI.convert_array(np.clip(out, 0, 1), "RGB", "Grayscale")
    return _effect_out(im, np.clip(out, 0, 1), alpha, single)


def _effect_out(im, col, alpha, single):
    a = col if alpha is None else np.concatenate([col, alpha], axis=-1)
    typ = "Real32" if im.type == "Bit" else im.type
    cs = (im.cs if im.cs in ("Grayscale", "Automatic") else "Grayscale") if single else "RGB"
    return image_expr(WI.Img(quantize(a, typ), typ, cs, False))
