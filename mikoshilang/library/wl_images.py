"""Wolfram batch W21: images I (the Image object, representation, colour and geometric operations).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). numpy (a core dependency) does the pixel arithmetic; Pillow is optional and only needed for
reading and writing image files (``ImageFileApply``, ``ImageFileFilter``, ``ImageFileScan``): without it those raise
NotImplementedError and everything else works.

Representation. An image is the ordinary expression ``Image[data, type, ColorSpace -> cs, Interleaving -> il]``
(``Image3D[...]`` for volumes), the canonical form that ``Image[...]`` evaluates to:

* ``data`` is the pixel array, top row first: a matrix for a single-channel image, else rows x columns x channels
  (slices x rows x columns [x channels] for Image3D), holding the raw values of ``type``: integers 0/1 for "Bit",
  0..255 for "Byte", 0..65535 for "Bit16", machine reals for "Real32"/"Real64" (stored at double precision);
* ``type`` is "Bit", "Byte", "Bit16", "Real32" or "Real64" (``Image[data]`` gives "Real32", as in Wolfram);
* ``cs`` is a colour space ("Grayscale", "RGB", "CMYK", "HSB", "XYZ", "LAB", "LCH", "LUV") or ``Automatic``; an
  image has an alpha channel when it has one channel more than its colour space needs;
* ``il`` is ``True`` for multichannel images and ``None`` for single-channel ones (as Wolfram reports it).

Because the image is a normal expression, SameQ, Part and pattern matching work structurally. Channel values seen by
user functions (ImageApply, PixelValue, ...) are on the 0..1 scale whatever the type.

Coordinates follow Wolfram: pixel positions ``{x, y}`` (PixelValue, ReplacePixelValue, PixelValuePositions) are
1-based column and row-from-the-bottom indices with integer values at pixel centres; image coordinates (ImageValue,
ImageValuePositions, ImageTrim, ImageCompose) run from ``{0, 0}`` at the bottom-left corner to ``{w, h}``; matrix
positions ``{row, column}`` (ImageTake, ImageApplyIndexed) count rows from the top.

Python API for later batches (W28 image processing): ``parse_image(expr, fname)`` -> ``Img`` (float array on the 0..1
scale, type, colour space), ``image_expr(img)`` -> the canonical expression, ``Img.new(array, like=img)``,
``convolve(arr, kernel, padding)``, ``pad_array(arr, ...)``, ``to_space(img, cs)``.
"""
from __future__ import annotations

import math

import numpy as np

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L
from .wl_graphics_color import (parse_color, to_rgba, real_value, is_real_value, K_input, convert_channels,
                                make_color, SPACES, HEADS, NCHANNELS, inert_option, color_distance, lab,
                                distance_function)

AREA = "images (W21)"

TYPES = ("Bit", "Byte", "Bit16", "Real32", "Real64")
TYPE_RANK = {t: i for i, t in enumerate(TYPES)}
MAXV = {"Bit": 1, "Byte": 255, "Bit16": 65535}
SPACE_LIST = ("Grayscale", "RGB", "CMYK", "HSB", "XYZ", "LAB", "LCH", "LUV")
AUTO = Symbol("Automatic")
NONE = Symbol("None")
IMAGE_HEADS = ("Image", "Image3D")


def _is_sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def _str(x):
    return K_input(x)


# ───────────────────────────── the Img value ─────────────────────────────

class Img:
    """Pixel data as float64 on the 0..1 scale for integer types (raw values for real types), shape (h, w, c) or
    (d, h, w, c) for 3D, plus the type, the colour space ("Automatic" or a name) and the 3D flag."""
    __slots__ = ("a", "type", "cs", "d3")

    def __init__(self, a, typ="Real32", cs="Automatic", d3=False):
        a = np.asarray(a, dtype=float)
        if a.ndim == (3 if d3 else 2):
            a = a[..., None]
        self.a, self.type, self.cs, self.d3 = a, typ, cs, d3

    def new(self, a, typ=None, cs=None, d3=None):
        return Img(a, self.type if typ is None else typ, self.cs if cs is None else cs,
                   self.d3 if d3 is None else d3)

    @property
    def c(self):
        return self.a.shape[-1]

    @property
    def h(self):
        return self.a.shape[-3]

    @property
    def w(self):
        return self.a.shape[-2]

    @property
    def alpha(self):
        return self.cs != "Automatic" and self.c == NCHANNELS[self.cs] + 1

    def color(self):
        return self.a[..., :-1] if self.alpha else self.a

    def alpha_array(self):
        return self.a[..., -1:] if self.alpha else np.ones(self.a.shape[:-1] + (1,))

    def dims(self):
        s = self.a.shape
        return [s[2], s[1], s[0]] if self.d3 else [s[1], s[0]]


def default_space(c):
    """Colour space Image[data] assumes for c channels: RGB for 3 or 4 (with alpha), Automatic otherwise."""
    return "RGB" if c in (3, 4) else "Automatic"


def _lst(x):
    if isinstance(x, list):
        return Expr("List", *[_lst(v) for v in x])
    return x


def stored(a, typ):
    """Array on the 0..1 scale -> raw stored values of typ (rounded and clipped for integer types)."""
    if typ in MAXV:
        return np.rint(np.clip(a, 0.0, 1.0) * MAXV[typ]).astype(np.int64)
    return np.asarray(a, dtype=float)


def quantize(a, typ):
    """Round an array on the 0..1 scale to what typ can hold (clipping for integer types)."""
    if typ in MAXV:
        return np.rint(np.clip(a, 0.0, 1.0) * MAXV[typ]) / MAXV[typ]
    return np.asarray(a, dtype=float)


def image_expr(img):
    """The canonical Image / Image3D expression of an Img."""
    if not np.all(np.isfinite(img.a)):
        raise ValueError(("Image3D" if img.d3 else "Image") + ": pixel values must be finite")
    v = stored(img.a, img.type)
    if img.c == 1:
        v = v[..., 0]
    data = _lst(v.tolist())
    cs = AUTO if img.cs == "Automatic" else img.cs
    il = NONE if img.c == 1 else True
    return Expr("Image3D" if img.d3 else "Image", data, img.type, Expr("Rule", Symbol("ColorSpace"), cs),
                Expr("Rule", Symbol("Interleaving"), il))


_CACHE = []          # [(expr, Img)]: the last images parsed (the expression is kept, so `is` identifies it)


def parse_image(x, fname, allow3d=True, allow2d=True):
    """Img of a canonical Image/Image3D expression; TypeError for other values, Unevaluated if symbolic."""
    for e, im in _CACHE:
        if e is x:
            _check_dim(im, fname, allow3d, allow2d)
            return im
    if not (isinstance(x, Expr) and x.head in IMAGE_HEADS):
        if isinstance(x, (Symbol,)) or (isinstance(x, Expr) and not K.is_list(x) and x.head not in
                                         ("Rule", "Association", "Graphics", "Graphics3D") and parse_color(x) is None
                                         and not is_real_value(x)):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_short(x)} is not an image")
    if len(x.args) != 4 or x.args[1] not in TYPES:
        raise Unevaluated()                          # an Image[...] that did not evaluate (symbolic data)
    d3 = x.head == "Image3D"
    a = _numeric_array(x.args[0], x.head)
    if a is None:
        raise Unevaluated()
    typ = x.args[1]
    cs = x.args[2].args[1]
    cs = "Automatic" if _is_sym(cs, "Automatic") else cs
    if typ in MAXV:
        a = a / MAXV[typ]
    im = Img(a, typ, cs, d3)
    im.a.setflags(write=False)
    _CACHE.append((x, im))
    if len(_CACHE) > 16:
        del _CACHE[0]
    _check_dim(im, fname, allow3d, allow2d)
    return im


def _check_dim(im, fname, allow3d, allow2d):
    if im.d3 and not allow3d:
        raise NotImplementedError(f"{fname}: Image3D objects are not supported")
    if not im.d3 and not allow2d:
        raise TypeError(f"{fname}: expected an Image3D object, not a 2D image")


def is_image(x):
    try:
        parse_image(x, "ImageQ")
        return True
    except (TypeError, ValueError, NotImplementedError, Unevaluated):
        return False


def _short(x):
    t = _str(x)
    return t if len(t) <= 60 else t[:57] + "..."


def _nested(x):
    """Nested Python lists of the leaves of a List expression (leaves untouched)."""
    if K.is_list(x):
        return [_nested(v) for v in x.args]
    return x


def _leaf_float(v, fname):
    if isinstance(v, bool) or isinstance(v, str):
        raise TypeError(f"{fname}: pixel values must be real numbers, not {_short(v)}")
    if isinstance(v, (int, float)):
        return float(v)
    return real_value(v, fname, "pixel value")


def _numeric_array(x, fname):
    """float ndarray of a rectangular List array of reals (None if it is not a List); raises for bad leaves."""
    if not K.is_list(x):
        return None
    nest = _nested(x)
    try:
        a = np.array(nest, dtype=float)
    except (TypeError, ValueError):
        a = None
    if a is None or a.dtype != float:
        flat = []

        def walk(v):
            if isinstance(v, list):
                return [walk(t) for t in v]
            return _leaf_float(v, fname)
        try:
            a = np.array(walk(nest), dtype=float)
        except ValueError:
            raise ValueError(f"{fname}: the data must be a rectangular array")
        del flat
    else:
        # np.array accepted the leaves: still reject booleans (True -> 1.0)
        def has_bool(v):
            if isinstance(v, list):
                return any(has_bool(t) for t in v)
            return isinstance(v, bool)
        if has_bool(nest):
            raise TypeError(f"{fname}: pixel values must be real numbers, not booleans")
    if a.dtype == object:
        raise ValueError(f"{fname}: the data must be a rectangular array")
    return a


# ───────────────────────────── options and small argument helpers ─────────────────────────────

KNOWN_OPTIONS = {"ImageSize", "ImageResolution", "Magnification", "MetaInformation", "AlignmentPoint",
                 "BaselinePosition", "ColorSpace", "Interleaving", "Padding", "Resampling", "Background", "DataRange",
                 "Masking", "Method", "PlotRange", "DistanceFunction", "MinColorDistance", "MaxColorDistance",
                 "ColorCoverage", "Dithering", "ColorFunction", "ColorFunctionScaling", "ColorRules",
                 "DataReversed", "PerformanceGoal", "Spacings", "PaddingSize", "IncludeMetaInformation",
                 "PreserveColor", "ZoomCenter", "ZoomFactor", "WorkingPrecision", "Standardized", "ImageMargins",
                 "ImagePadding", "BoxRatios", "ViewPoint", "Axes", "Boxed", "ClipRange", "Lighting", "Opacity",
                 "FillingStyle", "Antialiasing", "ColorRules"}


def opts_split(args, allowed, fname):
    """(positional, {name: value}) with trailing Name -> value options; an option that is not in `allowed` but is a
    known Wolfram option raises NotImplementedError."""
    args = list(args)
    out = {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") \
            and len(args[-1].args) == 2 and isinstance(args[-1].args[0], (Symbol, str)):
        lhs = args[-1].args[0]
        name = lhs.name if isinstance(lhs, Symbol) else lhs
        if name in allowed:
            out[name] = args.pop().args[1]
        elif name in KNOWN_OPTIONS:
            raise NotImplementedError(f"{fname}: option {name} is not supported")
        else:
            break
    return args, out


def int_arg(x, fname, what="argument", lo=None):
    if isinstance(x, bool) or isinstance(x, str) or K.is_list(x):
        raise TypeError(f"{fname}: {what} must be an integer, not {_short(x)}")
    if K.is_int(x):
        v = int(x)
    elif isinstance(x, float) or is_real_value(x):
        raise TypeError(f"{fname}: {what} must be an integer, not {_short(x)}")
    else:
        raise Unevaluated()
    if lo is not None and v < lo:
        raise ValueError(f"{fname}: {what} must be at least {lo}, not {v}")
    return v


def real_arg(x, fname, what="argument"):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if x != x or x in (math.inf, -math.inf):
            raise ValueError(f"{fname}: {what} must be finite")
        return float(x)
    if K.is_list(x) or isinstance(x, (str, bool)):
        raise TypeError(f"{fname}: {what} must be a real number, not {_short(x)}")
    return real_value(x, fname, what)


def type_arg(x, fname):
    if isinstance(x, str):
        if x in TYPES:
            return x
        raise ValueError(f"{fname}: {x!r} is not an image type (one of {', '.join(TYPES)})")
    if _is_sym(x, "Automatic"):
        return None
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: the image type must be a string such as \"Byte\", not {_short(x)}")


def space_arg(x, fname, allow_auto=True):
    if isinstance(x, str):
        if x in SPACE_LIST:
            return x
        raise ValueError(f"{fname}: {x!r} is not a colour space (one of {', '.join(SPACE_LIST)})")
    if _is_sym(x, "Automatic") and allow_auto:
        return "Automatic"
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: ColorSpace must be a string such as \"RGB\", not {_short(x)}")


def larger_type(*types):
    return max(types, key=lambda t: TYPE_RANK[t])


def convert_type(img, typ):
    if typ is None or typ == img.type:
        return img
    return img.new(quantize(img.a, typ), typ=typ)


def size_arg(x, fname, allow3d=True, default=(150, 150)):
    """[w, h] or [w, d, h] of a size specification `side`, {w, h} or {w, d, h}."""
    if x is None:
        return list(default)
    if K.is_list(x):
        if len(x.args) not in ((2, 3) if allow3d else (2,)):
            raise ValueError(f"{fname}: the size must be {{width, height}}" + (" or {width, depth, height}"
                                                                               if allow3d else ""))
        return [int_arg(v, fname, "size", 1) for v in x.args]
    n = int_arg(x, fname, "size", 1)
    return [n, n]


def _rng():
    from .statistics import _RNG
    return np.random.default_rng(_RNG.getrandbits(64))


def color_values(x, fname, nchan=None):
    """(channels on the space's scale, space or None, alpha or None) of a pixel value: a number, a colour or a list of
    channel values."""
    p = parse_color(x)
    if p is not None:
        return list(p[1]), p[0], p[2]
    if K.is_list(x):
        return [real_arg(v, fname, "channel value") for v in x.args], None, None
    return [real_arg(x, fname, "pixel value")], None, None


def value_for(img, x, fname):
    """Channel vector (length img.c) for a pixel value x given as a number, channel list or colour, in img's space."""
    ch, space, alpha = color_values(x, fname)
    if space is not None:
        target = img.cs if img.cs != "Automatic" else ("Grayscale" if img.color().shape[-1] == 1 else "RGB")
        if target != space:
            ch = list(convert_channels(space, ch, target))
        if len(ch) != img.color().shape[-1]:
            raise ValueError(f"{fname}: the colour {_short(x)} does not match the {img.c}-channel image")
        if img.alpha:
            ch = ch + [1.0 if alpha is None else alpha]
        return np.array(ch)
    if len(ch) == 1:
        return np.full(img.c, ch[0])
    if len(ch) != img.c:
        raise ValueError(f"{fname}: {len(ch)} channel values given for a {img.c}-channel image")
    return np.array(ch)


# ───────────────────────────── Image and Image3D ─────────────────────────────

_NUMERIC_ARRAY_TYPES = {"UnsignedInteger8": "Byte", "UnsignedInteger16": "Bit16", "Real32": "Real32",
                        "Real64": "Real64"}


def _color_leaves(nest):
    """If the leaves at depth 2 (3 for volumes) are colour directives, an RGB(A) float array; else None."""
    found = []

    def walk(v):
        if isinstance(v, list):
            return [walk(t) for t in v]
        if isinstance(v, Expr) and parse_color(v) is not None:
            found.append(True)
            return list(to_rgba(v))
        return v
    out = walk(nest)
    if not found:
        return None
    def fill(v):
        if isinstance(v, list) and v and not isinstance(v[0], (list,)) and not all(isinstance(t, float) for t in v):
            return [fill(t) for t in v]
        return v

    def leaf_to_rgba(v):
        if isinstance(v, list) and len(v) == 4 and all(isinstance(t, float) for t in v):
            return v
        if isinstance(v, list):
            return [leaf_to_rgba(t) for t in v]
        g = _leaf_float(v, "Image")
        return [g, g, g, 1.0]
    try:
        a = np.array(leaf_to_rgba(out), dtype=float)
    except ValueError:
        raise ValueError("Image: the data must be a rectangular array of values or colours")
    if np.all(a[..., 3] == 1.0):
        a = a[..., :3]
    return a


def build_image(data, typ, opts, d3, fname):
    """Img from Image/Image3D arguments (data already evaluated)."""
    cs = space_arg(opts["ColorSpace"], fname) if "ColorSpace" in opts else None
    il = opts.get("Interleaving", AUTO)
    if not (il is True or il is False or _is_sym(il, "Automatic") or _is_sym(il, "None")):
        raise ValueError(f"{fname}: Interleaving must be True, False, None or Automatic")
    if isinstance(data, Expr) and data.head == "NumericArray" and len(data.args) == 2:
        t = _NUMERIC_ARRAY_TYPES.get(data.args[1])
        if t is None:
            raise NotImplementedError(f"{fname}: NumericArray type {_short(data.args[1])} is not supported")
        a = _numeric_array(data.args[0], fname)
        if t in MAXV:
            a = a / MAXV[t]
        typ = typ or t
        data_type = t
    elif K.is_list(data):
        nest = _nested(data)
        ca = _color_leaves(nest)
        if ca is not None:
            a, data_type = ca, "Real64"
            if cs is None:
                cs = "RGB"
        else:
            a = _numeric_array(data, fname)
            data_type = None
    else:
        if isinstance(data, Symbol) or (isinstance(data, Expr) and data.head not in ("Rule", "Association")
                                        and parse_color(data) is None):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_short(data)} is not an array of pixel values")
    base = 3 if d3 else 2
    if a.size == 0 or a.ndim not in (base, base + 1):
        raise ValueError(f"{fname}: the data must be a non-empty {'3D' if d3 else '2D'} array of values or of "
                         f"channel lists, not an array of dimensions {list(a.shape)}")
    if a.ndim == base + 1 and il is False:
        a = np.moveaxis(a, 0, -1)
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{fname}: pixel values must be finite")
    typ = typ or "Real32"
    if typ in MAXV and data_type is None:
        raw = a
        if np.any(raw != np.rint(raw)) or np.any(raw < 0) or np.any(raw > MAXV[typ]):
            raise ValueError(f"{fname}: \"{typ}\" data must be integers from 0 to {MAXV[typ]}")
        a = raw / MAXV[typ]
    elif typ in MAXV:
        a = quantize(a, typ)
    img = Img(a, typ, "Automatic", d3)
    if cs is None:
        cs = default_space(img.c)
    if cs != "Automatic" and img.c not in (NCHANNELS[cs], NCHANNELS[cs] + 1):
        raise ValueError(f"{fname}: ColorSpace \"{cs}\" needs {NCHANNELS[cs]} or {NCHANNELS[cs] + 1} channels, the "
                         f"data has {img.c}")
    img.cs = cs
    return img


_IMAGE_OPTS = {"ColorSpace", "Interleaving"}


def _image_impl(head, d3):
    def impl(data, *rest):
        pos, o = opts_split(rest, _IMAGE_OPTS, head)
        if len(pos) > 1:
            raise ValueError(f"{head}: expected {head}[data, type, options]")
        typ = type_arg(pos[0], head) if pos else None
        if isinstance(data, Expr) and data.head in IMAGE_HEADS:
            src = parse_image(data, head)
            if src.d3 != d3:
                raise TypeError(f"{head}: {data.head} objects cannot be converted to {head}"
                                + (" (use Image3DSlices or Image3DProjection)" if not d3 else ""))
            img = convert_type(src, typ)
            if "ColorSpace" in o:
                cs = space_arg(o["ColorSpace"], head)
                if cs != "Automatic" and img.c not in (NCHANNELS[cs], NCHANNELS[cs] + 1):
                    raise ValueError(f"{head}: ColorSpace \"{cs}\" needs {NCHANNELS[cs]} or {NCHANNELS[cs] + 1} "
                                     f"channels, the image has {img.c}")
                img = img.new(img.a, cs=cs)
            return image_expr(img)
        if not d3 and isinstance(data, Expr) and data.head in ("Graphics", "Graphics3D"):
            if o or pos:
                raise NotImplementedError("Image: options and types are not supported for Image[graphics]")
            return _rasterize(data)
        if not d3 and isinstance(data, Expr) and data.head == "Raster":
            return image_expr(_from_raster(data, typ, o))
        if d3 and K.is_list(data) and data.args and all(isinstance(v, Expr) and v.head == "Image" for v in data.args):
            return image_expr(_stack(data.args, typ, o))
        img = build_image(data, typ, o, d3, head)
        out = image_expr(img)
        if out == Expr(head, data, *rest):
            raise Unevaluated()                     # already canonical
        return out
    impl.__annotations__ = {"data": "MatrixOf[Real] | ListOf[Expr] | Expr", "rest": "String | Option",
                            "return": "Expr"}
    return impl


def _rasterize(g):
    from ..evaluate import evaluate
    return evaluate(Expr("Rasterize", g, "Image"))


def _from_raster(r, typ, o):
    if len(r.args) not in (1, 2) or not K.is_list(r.args[0]):
        raise NotImplementedError("Image: only Raster[array] and Raster[array, {{x0, y0}, {x1, y1}}] are supported")
    img = build_image(r.args[0], typ, o, False, "Image")
    return img.new(img.a[::-1].copy())             # Raster rows run bottom to top


def _stack(images, typ, o):
    ims = [parse_image(v, "Image3D", allow3d=False) for v in images]
    if len({(i.w, i.h, i.c) for i in ims}) != 1:
        raise ValueError("Image3D: the 2D images must have the same dimensions and number of channels")
    t = typ or larger_type(*[i.type for i in ims])
    cs = ims[0].cs if len({i.cs for i in ims}) == 1 else "Automatic"
    if "ColorSpace" in o:
        cs = space_arg(o["ColorSpace"], "Image3D")
    return Img(quantize(np.stack([i.a for i in ims]), t), t, cs, True)


fn("Image", area=AREA, status="added",
   notes="Image[data] for a matrix of reals (one channel), an array of channel lists or a matrix of colour directives "
         "(RGB, with alpha when any colour has opacity); Image[data, type] with type \"Bit\", \"Byte\", \"Bit16\" "
         "(data must then be integers in the type's range, else ValueError), \"Real32\" (default) or \"Real64\"; "
         "options ColorSpace (Automatic gives \"RGB\" for 3 or 4 channels, Automatic otherwise; a space must match "
         "the channel count, with one extra channel taken as alpha) and Interleaving (False: channels first). "
         "Image[image, type] converts the type (rounding and clipping to integer types), Image[image, ColorSpace -> cs] "
         "relabels; Image[NumericArray[...]] takes the type from the array; Image[Raster[array]] reverses the rows "
         "(Raster rows run bottom up); Image[graphics] is Rasterize[graphics, \"Image\"] (needs matplotlib). "
         "The canonical form Image[data, type, ColorSpace -> cs, Interleaving -> True | None] is inert. Display "
         "options (ImageSize, ImageResolution, MetaInformation, ...) raise NotImplementedError; real images are "
         "stored at double precision also for \"Real32\". Symbolic data stays unevaluated; ragged or non-finite "
         "data raises ValueError, non-numeric data TypeError. Check: exact (round trips through ImageData; "
         "reference values from the Image/ImageType documentation) in tests.")(_image_impl("Image", False))

fn("Image3D", area=AREA, status="added",
   notes="Image3D[data] for a 3D array of reals (slices top to bottom, rows back to front, columns left to right) or "
         "of channel lists; Image3D[{image1, image2, ...}] stacks 2D images of equal size (largest type); types, "
         "ColorSpace and Interleaving as for Image. The canonical form Image3D[data, type, ColorSpace -> cs, "
         "Interleaving -> ...] is inert. Rendering options (ViewPoint, BoxRatios, ...) raise NotImplementedError. "
         "Check: exact (round trips through ImageData and Image3DSlices) in tests.")(_image_impl("Image3D", True))


def _from_uint8(arr):
    """IMAGE_FACTORY for Rasterize: an RGB(A) uint8 array -> Image expression of type "Byte"."""
    a = np.asarray(arr, dtype=float) / 255.0
    return image_expr(Img(a, "Byte", "RGB"))


try:
    from . import wl_graphics_render as _R
    _R.IMAGE_FACTORY = _from_uint8
except ImportError:  # pragma: no cover
    pass


# ───────────────────────────── properties ─────────────────────────────

@fn("ImageQ", area=AREA, status="added",
    notes="ImageQ[expr] is True for a valid Image or Image3D object (the canonical form Image[...] evaluates to) and "
          "False for anything else, including unevaluated Image[...] calls with symbolic data. "
          "Check: exact (structural test) in tests.")
def _imageq(x: Expr) -> Boolean:
    return is_image(x)


@fn("BinaryImageQ", area=AREA, status="added",
    notes="BinaryImageQ[image] is True for a single-channel image of type \"Bit\" (2D or 3D), False otherwise (also "
          "for non-images). Check: exact in tests.")
def _binaryimageq(x: Expr) -> Boolean:
    if not is_image(x):
        return False
    im = parse_image(x, "BinaryImageQ")
    return im.type == "Bit" and im.c == 1


def _img1(x, fname, **kw):
    return parse_image(x, fname, **kw)


@fn("ImageType", area=AREA, status="added",
    notes="ImageType[image] gives \"Bit\", \"Byte\", \"Bit16\", \"Real32\" or \"Real64\". Non-images raise TypeError; "
          "symbols stay unevaluated. Check: exact in tests.")
def _imagetype(x: Expr) -> String:
    return _img1(x, "ImageType").type


@fn("ImageDimensions", area=AREA, status="added",
    notes="ImageDimensions[image] gives {width, height} ({width, depth, height} for Image3D). Non-images raise "
          "TypeError. Check: exact in tests.")
def _imagedimensions(x: Expr) -> ListOf[PosInteger]:
    return L(*_img1(x, "ImageDimensions").dims())


@fn("ImageChannels", area=AREA, status="added",
    notes="ImageChannels[image] gives the number of channels including alpha (1 grayscale, 2 grayscale + alpha, 3 "
          "RGB, 4 RGBA or CMYK, ...). Check: exact in tests.")
def _imagechannels(x: Expr) -> PosInteger:
    return _img1(x, "ImageChannels").c


@fn("ImageColorSpace", area=AREA, status="added",
    notes="ImageColorSpace[image] gives the colour space name or Automatic (no colour space; Image[data] of 1, 2 or "
          "5+ channels). Check: exact in tests.")
def _imagecolorspace(x: Expr) -> String | Expr:
    cs = _img1(x, "ImageColorSpace").cs
    return AUTO if cs == "Automatic" else cs


@fn("ImageAspectRatio", area=AREA, status="added",
    notes="ImageAspectRatio[image] gives height/width as an exact rational (of the slices for Image3D). "
          "Check: exact in tests.")
def _imageaspectratio(x: Expr) -> Rational:
    im = _img1(x, "ImageAspectRatio")
    return K.from_sp(__import__("sympy").Rational(im.h, im.w))


def data_expr(a, typ, interleave=True, reverse=False, d3=False):
    if typ in MAXV:
        v = stored(a, typ)
    else:
        v = np.asarray(a, dtype=float)
    if reverse:
        v = v[:, ::-1] if d3 else v[::-1]
    if v.shape[-1] == 1:
        v = v[..., 0]
    elif not interleave:
        v = np.moveaxis(v, -1, 0)
    return _lst(v.tolist())


@fn("ImageData", area=AREA, status="added",
    notes="ImageData[image] gives the pixel array, top row first (channel lists for multichannel images, plain values "
          "for one channel): integers 0/1 for \"Bit\", values normalised to 0..1 for \"Byte\" and \"Bit16\", the raw "
          "values for real types. ImageData[image, type] converts (\"Byte\" gives Round[255 v] clipped to 0..255, "
          "real types the 0..1 scale); Automatic gives the stored values. Options Interleaving -> False (channels "
          "first) and DataReversed -> True (bottom row first). Wolfram files ImageData as needing curated data; it "
          "is implemented here because the image batches need it. Check: exact (round trips Image <-> ImageData) "
          "in tests.")
def _imagedata(x: Expr, *rest: String | Option) -> Expr:
    pos, o = opts_split(rest, {"Interleaving", "DataReversed"}, "ImageData")
    if len(pos) > 1:
        raise ValueError("ImageData: expected ImageData[image, type, options]")
    im = _img1(x, "ImageData")
    il = o.get("Interleaving", True)
    if il not in (True, False) and not _is_sym(il, "Automatic"):
        raise ValueError("ImageData: Interleaving must be True, False or Automatic")
    rev = o.get("DataReversed", False)
    if rev not in (True, False):
        raise ValueError("ImageData: DataReversed must be True or False")
    if pos:
        t = type_arg(pos[0], "ImageData") or im.type
    else:
        t = "Bit" if im.type == "Bit" else "Real64"
    return data_expr(im.a, t, il is not False, rev, im.d3)


# ───────────────────────────── colour spaces of pixel arrays ─────────────────────────────

def source_space(img, fname):
    """Colour space to interpret img's colour channels in (Automatic: 1 channel grayscale, 3 RGB)."""
    if img.cs != "Automatic":
        return img.cs
    n = img.color().shape[-1]
    if n == 1:
        return "Grayscale"
    if n == 3:
        return "RGB"
    raise ValueError(f"{fname}: an image with {img.c} channels and ColorSpace Automatic has no colour "
                     f"interpretation; set ColorSpace first")


def convert_array(a, src, target):
    """(..., n_src) float array of colours in src -> (..., n_target) in target, via the W14 colour formulas."""
    if src == target:
        return np.array(a, dtype=float)
    if src == "Grayscale" and target == "RGB":
        return np.repeat(a, 3, axis=-1)
    if src == "RGB" and target == "Grayscale":
        return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])[..., None]
    flat = a.reshape(-1, a.shape[-1])
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    conv = np.array([convert_channels(src, tuple(float(t) for t in u), target) for u in uniq], dtype=float)
    return conv[np.asarray(inv).reshape(-1)].reshape(a.shape[:-1] + (NCHANNELS[target],))


def to_space(img, target, fname):
    """Img converted to colour space `target` (alpha kept); type kept for Grayscale/RGB/CMYK/HSB, else Real32."""
    src = source_space(img, fname)
    col = convert_array(img.color(), src, target)
    a = np.concatenate([col, img.alpha_array()], axis=-1) if img.alpha else col
    typ = img.type if target in ("Grayscale", "RGB", "CMYK", "HSB") else larger_type(img.type, "Real32")
    if typ == "Bit" and not np.all((a == 0) | (a == 1)):
        typ = "Real32"
    return Img(quantize(a, typ), typ, target, img.d3)


# ───────────────────────────── alpha channels ─────────────────────────────

def _color_with_alpha(c, alpha, fname):
    p = parse_color(c)
    space, ch, _ = p
    if space == "HSB" and len(c.args) == 1:
        ch = (real_value(c.args[0], fname), 1.0, 1.0)
    return make_color(space, ch, alpha)


@fn("AlphaChannel", area=AREA, status="added",
    notes="AlphaChannel[color] gives the opacity (1 when none is given); AlphaChannel[image] gives the alpha channel "
          "as a single-channel \"Grayscale\" image of the same type (all 1 when the image has no alpha channel). "
          "Non-colours raise TypeError. Check: exact in tests.")
def _alphachannel(x: Expr) -> Real | Expr:
    p = parse_color(x)
    if p is not None:
        return 1 if p[2] is None else p[2]
    im = _img1(x, "AlphaChannel")
    return image_expr(Img(im.alpha_array(), im.type, "Grayscale", im.d3))


def _alpha_space(im, fname):
    if im.cs != "Automatic":
        return im.cs
    if im.c == 1:
        return "Grayscale"
    raise ValueError(f"{fname}: the image needs an explicit ColorSpace (it has {im.c} channels and ColorSpace "
                     f"Automatic)")


@fn("SetAlphaChannel", area=AREA, status="added",
    notes="SetAlphaChannel[color] and SetAlphaChannel[color, a] give the colour with opacity 1 or a (an existing "
          "opacity is replaced); SetAlphaChannel[image] = SetAlphaChannel[image, 1]; SetAlphaChannel[image, a] with a "
          "number or a single-channel image of the same dimensions (values clipped to 0..1). An Automatic single-"
          "channel image becomes \"Grayscale\"; other Automatic images raise ValueError, as Wolfram requires an explicit "
          "colour space. Result type: the larger of the two types (a \"Bit\" image with fractional opacity becomes "
          "\"Real32\"). Check: exact (AlphaChannel round trip) in tests.")
def _setalphachannel(x: Expr, a: Real | Expr = 1) -> Expr:
    if parse_color(x) is not None:
        return _color_with_alpha(x, min(max(real_arg(a, "SetAlphaChannel", "opacity"), 0.0), 1.0), "SetAlphaChannel")
    im = _img1(x, "SetAlphaChannel")
    cs = _alpha_space(im, "SetAlphaChannel")
    typ = im.type
    if isinstance(a, Expr) and a.head in IMAGE_HEADS:
        am = parse_image(a, "SetAlphaChannel")
        if am.c != 1:
            raise ValueError("SetAlphaChannel: the alpha image must have a single channel")
        if am.a.shape[:-1] != im.a.shape[:-1]:
            raise ValueError("SetAlphaChannel: the alpha image must have the same dimensions as the image")
        al = np.clip(am.a, 0, 1)
        typ = larger_type(typ, am.type)
    else:
        v = min(max(real_arg(a, "SetAlphaChannel", "opacity"), 0.0), 1.0)
        al = np.full(im.a.shape[:-1] + (1,), v)
    if typ == "Bit" and not np.all((al == 0) | (al == 1)):
        typ = "Real32"
    out = np.concatenate([im.color(), al], axis=-1)
    return image_expr(Img(quantize(out, typ), typ, cs, im.d3))


def _bg_rgb(bg, fname):
    p = parse_color(bg)
    if p is None:
        if K.is_list(bg) or isinstance(bg, str):
            raise TypeError(f"{fname}: the background must be a colour or a gray level, not {_short(bg)}")
        g = real_arg(bg, fname, "background")
        return "Grayscale", [g]
    if p[0] == "Grayscale":
        return "Grayscale", [p[1][0]]
    return "RGB", list(convert_channels(p[0], p[1], "RGB"))


@fn("RemoveAlphaChannel", area=AREA, status="added",
    notes="RemoveAlphaChannel[color] drops the opacity; RemoveAlphaChannel[color, bg] blends c a + bg (1 - a) and "
          "gives GrayLevel (both gray) or RGBColor; RemoveAlphaChannel[image] drops the alpha channel (the image itself "
          "when it has none) and RemoveAlphaChannel[image, bg] blends with the background colour, giving a Grayscale "
          "or RGB image. Check: exact (blend formula) in tests.")
def _removealphachannel(x: Expr, bg: Expr = None) -> Expr:
    fname = "RemoveAlphaChannel"
    p = parse_color(x)
    if p is not None:
        space, ch, alpha = p
        if space == "HSB" and len(x.args) == 1:
            ch = (real_value(x.args[0], fname), 1.0, 1.0)
        if bg is None:
            return make_color(space, ch) if alpha is not None or x.head == "Opacity" else x
        a = 1.0 if alpha is None else alpha
        bspace, bch = _bg_rgb(bg, fname)
        if space == "Grayscale" and bspace == "Grayscale":
            return Expr("GrayLevel", ch[0] * a + bch[0] * (1 - a))
        c = convert_channels(space, ch, "RGB")
        b = bch * 3 if bspace == "Grayscale" else bch
        return Expr("RGBColor", *[float(c[i] * a + b[i] * (1 - a)) for i in range(3)])
    im = _img1(x, fname)
    if bg is None:
        return x if not im.alpha else image_expr(im.new(im.color().copy()))
    if not im.alpha:
        return x
    bspace, bch = _bg_rgb(bg, fname)
    src = source_space(im, fname)
    a = im.alpha_array()
    if src == "Grayscale" and bspace == "Grayscale":
        out, cs = im.color() * a + bch[0] * (1 - a), "Grayscale"
    else:
        col = convert_array(im.color(), src, "RGB")
        b = np.array(bch * 3 if bspace == "Grayscale" else bch)
        out, cs = col * a + b * (1 - a), "RGB"
    typ = im.type if im.type != "Bit" or np.all((out == 0) | (out == 1)) else "Real32"
    return image_expr(Img(quantize(out, typ), typ, cs, im.d3))


# ───────────────────────────── constructors ─────────────────────────────

def _value_image(val, dims, fname):
    """(array (h, w, c) or (d, h, w, c), colour space) of a constant value spec."""
    d3 = len(dims) == 3
    shape = (dims[2], dims[1], dims[0]) if d3 else (dims[1], dims[0])
    p = parse_color(val)
    if p is not None:
        space, ch, alpha = p
        if space == "HSB" and len(val.args) == 1:
            ch = (real_value(val.args[0], fname), 1.0, 1.0)
        vec = list(ch) + ([] if alpha is None else [alpha])
        return np.broadcast_to(np.array(vec, dtype=float), shape + (len(vec),)).copy(), space
    if K.is_list(val):
        vec = [real_arg(v, fname, "channel value") for v in val.args]
        if not vec:
            raise ValueError(f"{fname}: the channel list is empty")
        return np.broadcast_to(np.array(vec), shape + (len(vec),)).copy(), default_space(len(vec))
    v = real_arg(val, fname, "value")
    return np.full(shape + (1,), v), "Automatic"


@fn("ConstantImage", area=AREA, status="added",
    notes="ConstantImage[val, size, type] with val a gray level, a list of channel values or a colour directive (the "
          "colour's space, opacity as alpha), size side, {w, h} or {w, d, h} (Image3D), default {150, 150}, type "
          "default \"Real32\"; option ColorSpace relabels. Check: exact in tests.")
def _constantimage(val: Real | ListOf[Real] | Expr, *rest: Integer | ListOf[PosInteger] | String | Option) -> Expr:
    pos, o = opts_split(rest, {"ColorSpace"}, "ConstantImage")
    if len(pos) > 2:
        raise ValueError("ConstantImage: expected ConstantImage[val, size, type]")
    dims = size_arg(pos[0] if pos else None, "ConstantImage")
    typ = type_arg(pos[1], "ConstantImage") if len(pos) > 1 else None
    a, cs = _value_image(val, dims, "ConstantImage")
    if "ColorSpace" in o:
        cs = space_arg(o["ColorSpace"], "ConstantImage")
        if cs != "Automatic" and a.shape[-1] not in (NCHANNELS[cs], NCHANNELS[cs] + 1):
            raise ValueError(f"ConstantImage: ColorSpace \"{cs}\" does not match {a.shape[-1]} channel values")
    typ = typ or "Real32"
    return image_expr(Img(quantize(a, typ), typ, cs, len(dims) == 3))


@fn("RandomImage", area=AREA, status="added",
    notes="RandomImage[] = RandomImage[1, {150, 150}]; RandomImage[max], RandomImage[{min, max}] (uniform) and "
          "RandomImage[dist] (values from RandomVariate[dist]) with size side, {w, h} or {w, d, h} and type (default "
          "\"Real32\"; integer types round the 0..1 values); ColorSpace -> cs gives that many channels per pixel. "
          "Reproducible after SeedRandom (the stream differs from Wolfram's). Check: reference (range, dimensions, "
          "type and sample statistics) in tests.")
def _randomimage(*args: Real | ListOf[Real] | Expr | Option) -> Expr:
    fname = "RandomImage"
    pos, o = opts_split(args, {"ColorSpace", "Interleaving"}, fname)
    if len(pos) > 3:
        raise ValueError("RandomImage: expected RandomImage[range, size, type]")
    spec = pos[0] if pos else 1
    dims = size_arg(pos[1] if len(pos) > 1 else None, fname)
    typ = type_arg(pos[2], fname) if len(pos) > 2 else None
    cs = space_arg(o["ColorSpace"], fname) if "ColorSpace" in o else "Automatic"
    nch = NCHANNELS[cs] if cs != "Automatic" else 1
    d3 = len(dims) == 3
    shape = ((dims[2], dims[1], dims[0]) if d3 else (dims[1], dims[0])) + (nch,)
    n = int(np.prod(shape))
    if K.is_list(spec):
        if len(spec.args) != 2:
            raise ValueError("RandomImage: the range must be {min, max}")
        lo, hi = (real_arg(v, fname, "range") for v in spec.args)
        a = _rng().uniform(lo, hi, size=shape)
    elif isinstance(spec, Expr) and spec.head.endswith("Distribution"):
        from ..evaluate import evaluate
        vals = evaluate(Expr("RandomVariate", spec, n))
        if not K.is_list(vals) or len(vals.args) != n:
            raise Unevaluated()
        a = np.array([_leaf_float(v, fname) for v in vals.args]).reshape(shape)
    else:
        if isinstance(spec, str) or isinstance(spec, bool):
            raise TypeError(f"RandomImage: {_short(spec)} is not a range or a distribution")
        hi = real_arg(spec, fname, "maximum")
        a = _rng().uniform(0.0, hi, size=shape)
    typ = typ or "Real32"
    return image_expr(Img(quantize(a, typ), typ, cs, d3))


for _name, _note in [
        ("Interleaving", "option of Image, ImageData, ImageApply, ImageScan, ImageFilter... giving whether channel "
                         "values are interleaved (True) or planar (False); Image/ImageData honour it and the pixel "
                         "functions pass whole channel lists or single values accordingly. Inert as an expression."),
        ("Masking", "option naming a region of interest for image functions; the W21 functions support only the "
                    "default All (other settings raise NotImplementedError). Inert as an expression."),
        ("MaxColorDistance", "option giving the largest colour distance allowed (FindMatchingColor, ColorDetect "
                             "forms); inert as an expression."),
        ("MinColorDistance", "option of DominantColors: clusters whose representative colours are closer than d (by "
                             "ColorDistance) are merged. Inert as an expression."),
        ("PreserveColor", "option of ImageRestyle and related functions (not part of W21); inert as an expression."),
        ("ZoomCenter", "option of DynamicImage (a front-end display function); inert as an expression."),
        ("ZoomFactor", "option of DynamicImage (a front-end display function); inert as an expression.")]:
    inert_option(_name, _note + " Check: reference (Wolfram documentation settings) in tests.", area=AREA)


# ───────────────────────────── colour operations ─────────────────────────────

def _one_minus(v):
    if isinstance(v, float):
        return 1.0 - v
    if isinstance(v, int) and not isinstance(v, bool):
        return 1 - v
    return K.from_sp(1 - K.to_sp(v))


def _negate_color(c, fname="ColorNegate"):
    space, ch, alpha = parse_color(c)
    if space == "HSB" and len(c.args) == 1:
        ch = (real_value(c.args[0], fname), 1.0, 1.0)
    if c.head in ("RGBColor", "GrayLevel") and len(c.args) in (NCHANNELS[space], NCHANNELS[space] + 1) \
            and not K.is_list(c.args[0]):
        n = NCHANNELS[space]
        return Expr(c.head, *[_one_minus(v) for v in c.args[:n]], *c.args[n:])
    if space in ("RGB", "Grayscale"):
        return make_color(space, [1 - v for v in ch], alpha)
    rgb = convert_channels(space, ch, "RGB")
    return make_color(space, convert_channels("RGB", [1 - v for v in rgb], space), alpha)


def negate_img(im):
    col = im.color()
    if im.cs in ("Automatic", "Grayscale", "RGB"):
        neg = 1.0 - col
    else:
        neg = convert_array(1.0 - convert_array(col, im.cs, "RGB"), "RGB", im.cs)
    a = np.concatenate([neg, im.alpha_array()], axis=-1) if im.alpha else neg
    return im.new(quantize(a, im.type))


@fn("ColorNegate", area=AREA, status="added",
    notes="ColorNegate[color] maps each RGB (or gray) channel v to 1 - v, keeping opacity and the colour's space (other "
          "spaces are negated through RGB, as documented); ColorNegate[image] does the same per pixel keeping type, "
          "colour space and alpha (images with ColorSpace Automatic negate every channel); ColorNegate[{...}] maps. "
          "Non-colours raise TypeError, symbols stay unevaluated. Check: exact (involution ColorNegate[ColorNegate[x]] "
          "== x and the documented 1 - v) in tests.")
def _colornegate(x: Expr) -> Expr:
    if K.is_list(x):
        return L(*[_colornegate(v) for v in x.args])
    if parse_color(x) is not None:
        return _negate_color(x)
    return image_expr(negate_img(_img1(x, "ColorNegate")))


_CHANNEL_NAMES = {"Red": ("RGB", 0), "R": ("RGB", 0), "Green": ("RGB", 1), "G": ("RGB", 1), "Blue": ("RGB", 2),
                  "B": ("RGB", 2), "Cyan": ("CMYK", 0), "C": ("CMYK", 0), "Magenta": ("CMYK", 1), "M": ("CMYK", 1),
                  "Yellow": ("CMYK", 2), "Y": ("CMYK", 2), "Black": ("CMYK", 3), "K": ("CMYK", 3),
                  "Hue": ("HSB", 0), "H": ("HSB", 0), "Saturation": ("HSB", 1), "S": ("HSB", 1),
                  "Brightness": ("HSB", 2), "V": ("HSB", 2), "Intensity": ("Grayscale", 0), "Gray": ("Grayscale", 0),
                  "I": ("Grayscale", 0), "Luminance": ("LAB", 0), "L": ("LAB", 0)}


def _single(im, a, typ=None):
    return Img(a, typ or im.type, "Automatic", im.d3)


def _separate(im, spec, fname="ColorSeparate"):
    """List of single-channel Img for a ColorSeparate specification (a str, Automatic or None)."""
    if spec is None or _is_sym(spec, "Automatic"):
        return [_single(im, im.a[..., i:i + 1]) for i in range(im.color().shape[-1] if im.cs != "Automatic"
                                                                 else im.c)]
    if not isinstance(spec, str):
        raise TypeError(f"{fname}: {_short(spec)} is not a colour space or channel name")
    if spec in ("Alpha", "A"):
        return [_single(im, im.alpha_array())]
    if spec == "Average":
        return [_single(im, im.color().mean(axis=-1, keepdims=True), larger_type(im.type, "Real32")
                        if im.type == "Bit" else im.type)]
    if spec in ("RGBA",):
        conv = to_space(im, "RGB", fname)
        return [_single(conv, conv.a[..., i:i + 1]) for i in range(3)] + [_single(conv, im.alpha_array())]
    if spec in SPACE_LIST:
        conv = to_space(im, spec, fname)
        return [_single(conv, conv.a[..., i:i + 1]) for i in range(NCHANNELS[spec])]
    if spec in _CHANNEL_NAMES:
        space, i = _CHANNEL_NAMES[spec]
        conv = to_space(im, space, fname)
        return [_single(conv, conv.a[..., i:i + 1])]
    raise ValueError(f"{fname}: {spec!r} is not a colour space or channel name")


@fn("ColorSeparate", area=AREA, status="added",
    notes="ColorSeparate[image] gives one single-channel image per colour channel (alpha excluded when the image has a "
          "colour space; every channel for ColorSpace Automatic); ColorSeparate[image, cs] converts to cs (\"RGB\", "
          "\"RGBA\", \"CMYK\", \"HSB\", \"XYZ\", \"LAB\", \"LCH\", \"LUV\", \"Grayscale\") first; channel names "
          "\"R\"/\"Red\", \"G\", \"B\", \"C\", \"M\", \"Y\", \"K\", \"H\", \"S\", \"V\"/\"Brightness\", \"I\"/"
          "\"Intensity\"/\"Gray\" (Rec. 601 luma), \"L\"/\"Luminance\" (L of LAB), \"A\"/\"Alpha\", \"Average\" "
          "give one image; a list of names gives a list. Types are kept except for the CIE spaces (\"Real32\"). "
          "Conversions use the ColorConvert formulas. Check: exact (ColorCombine[ColorSeparate[i]] == i) in tests.")
def _colorseparate(x: Expr, spec: String | ListOf[String] | Expr = None) -> ListOf[Expr] | Expr:
    im = _img1(x, "ColorSeparate")
    if K.is_list(spec):
        out = []
        for s in spec.args:
            out.extend(_separate(im, s))
        return L(*[image_expr(i) for i in out])
    parts = _separate(im, spec)
    if isinstance(spec, str) and (spec in _CHANNEL_NAMES or spec in ("Alpha", "A", "Average")):
        return image_expr(parts[0])
    return L(*[image_expr(i) for i in parts])


@fn("ColorCombine", area=AREA, status="added",
    notes="ColorCombine[{image1, image2, ...}] concatenates the channels of images of equal dimensions (largest "
          "type); single-channel inputs give an \"RGB\" image for 3 or 4 channels (the 4th as alpha) and Automatic "
          "otherwise, and a colour image followed by one single-channel image gives that colour space with alpha; "
          "ColorCombine[{...}, cs] labels the result with cs (n or n + 1 channels, else ValueError). Check: exact "
          "(round trip with ColorSeparate) in tests.")
def _colorcombine(imgs: ListOf[Expr], cs: String = None) -> Expr:
    if not K.is_list(imgs):
        if isinstance(imgs, Symbol):
            raise Unevaluated()
        raise TypeError("ColorCombine: the first argument must be a list of images")
    if not imgs.args:
        raise ValueError("ColorCombine: the list of images is empty")
    ims = [parse_image(v, "ColorCombine") for v in imgs.args]
    if len({i.a.shape[:-1] for i in ims}) != 1 or len({i.d3 for i in ims}) != 1:
        raise ValueError("ColorCombine: the images must have the same dimensions")
    a = np.concatenate([i.a for i in ims], axis=-1)
    typ = larger_type(*[i.type for i in ims])
    if cs is not None:
        space = space_arg(cs, "ColorCombine")
        if space != "Automatic" and a.shape[-1] not in (NCHANNELS[space], NCHANNELS[space] + 1):
            raise ValueError(f"ColorCombine: \"{space}\" needs {NCHANNELS[space]} or {NCHANNELS[space] + 1} "
                             f"channels, the images have {a.shape[-1]}")
    elif len(ims) == 2 and ims[0].cs != "Automatic" and not ims[0].alpha and ims[1].c == 1:
        space = ims[0].cs
    elif all(i.c == 1 for i in ims):
        space = default_space(a.shape[-1])
    else:
        space = "Automatic"
    return image_expr(Img(quantize(a, typ), typ, space, ims[0].d3))


# ───────────────────────────── colour matching ─────────────────────────────

def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


def _call(f, *args):
    return _ev(Expr(f, *args)) if isinstance(f, str) else _ev(K.call(f, *args))


def pixel_lab(im, fname):
    """(LAB array (..., 3), source space) of the colour channels of im."""
    src = source_space(im, fname)
    return convert_array(np.clip(im.color(), 0, 1) if src in ("RGB", "Grayscale", "CMYK", "HSB") else im.color(),
                         src, "LAB"), src


def distance_to(labs, target_lab, dfun):
    """Distances (...,) of an array of LAB colours to one LAB colour with a W14 distance function."""
    flat = labs.reshape(-1, 3)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    d = np.array([dfun(tuple(float(t) for t in u), tuple(target_lab)) for u in uniq], dtype=float)
    return d[np.asarray(inv).reshape(-1)].reshape(labs.shape[:-1])


def _colorsnear(spec, fname):
    """(LAB of the colour, distance, distance function) of a colour or ColorsNear[color, d, dfun]."""
    if isinstance(spec, Expr) and spec.head == "ColorsNear" and 1 <= len(spec.args) <= 3:
        c = spec.args[0]
        if parse_color(c) is None:
            raise TypeError(f"{fname}: {_short(c)} is not a colour")
        d = real_arg(spec.args[1], fname, "distance") if len(spec.args) > 1 else 0.2
        f = distance_function(spec.args[2], fname) if len(spec.args) > 2 else distance_function("CIE2000", fname)
        return lab(c), d, f
    if parse_color(spec) is not None:
        return lab(spec), 0.2, distance_function("CIE2000", fname)
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(spec)} is not a colour or a ColorsNear specification")


@fn("ColorDetect", area=AREA, status="added",
    notes="ColorDetect[image, cspec] gives a \"Bit\" mask image that is 1 where the pixel colour lies in cspec: a colour "
          "(same as ColorsNear[color], i.e. CIEDE2000 distance at most 0.2 as ColorsNear documents), ColorsNear[color, "
          "d] or ColorsNear[color, d, dfun] (any ColorDistance DistanceFunction), or a list of these (union). Images "
          "with ColorSpace Automatic and 2 or 4+ channels raise ValueError. Check: reference (ColorDistance of the "
          "detected pixels, re-measured in tests).")
def _colordetect(x: Expr, spec: Expr) -> Expr:
    im = _img1(x, "ColorDetect")
    labs, _ = pixel_lab(im, "ColorDetect")
    specs = list(spec.args) if K.is_list(spec) else [spec]
    mask = np.zeros(labs.shape[:-1], dtype=bool)
    for s in specs:
        tl, d, f = _colorsnear(s, "ColorDetect")
        mask |= distance_to(labs, tl, f) <= d
    return image_expr(Img(mask[..., None].astype(float), "Bit", "Automatic", im.d3))


@fn("ColorReplace", area=AREA, status="added",
    notes="ColorReplace[image, color] makes pixels within ColorDistance 0.1 of color transparent (adds an alpha "
          "channel); ColorReplace[image, color -> new], [image, color -> new, d] and [image, {c1 -> n1, ...}, d | {d1, "
          "...}] replace pixels within distance d (default 0.1) of ci by ni, the first matching rule winning; a "
          "replacement with opacity adds an alpha channel. Distances are ColorDistance's default (CIE76 on the LAB "
          "1/100 scale): Wolfram does not document the metric, so borderline pixels may differ. Type and colour "
          "space are kept. Check: reference (ColorDistance of replaced pixels, re-measured in tests).")
def _colorreplace(x: Expr, rules: Expr, d: Real | ListOf[Real] = None) -> Expr:
    fname = "ColorReplace"
    im = _img1(x, fname)
    rl = list(rules.args) if K.is_list(rules) else [rules]
    if not rl:
        raise ValueError(f"{fname}: no colours given")
    if d is None:
        ds = [0.1] * len(rl)
    elif K.is_list(d):
        ds = [real_arg(v, fname, "distance") for v in d.args]
        if len(ds) != len(rl):
            raise ValueError(f"{fname}: {len(ds)} distances given for {len(rl)} colours")
    else:
        ds = [real_arg(d, fname, "distance")] * len(rl)
    if any(v < 0 for v in ds):
        raise ValueError(f"{fname}: distances must be non-negative")
    labs, src = pixel_lab(im, fname)
    space = im.cs if im.cs != "Automatic" else src
    dfun = distance_function(None, fname)
    out = np.array(im.color(), dtype=float)
    alpha = np.array(im.alpha_array(), dtype=float)
    need_alpha = im.alpha
    done = np.zeros(labs.shape[:-1], dtype=bool)
    for r, dd in zip(rl, ds):
        if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2:
            c, new = r.args
        else:
            c, new = r, None
        if parse_color(c) is None:
            if isinstance(c, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: {_short(c)} is not a colour")
        m = (distance_to(labs, lab(c), dfun) <= dd) & ~done
        done |= m
        if new is None:
            alpha[m] = 0.0
            need_alpha = True
            continue
        p = parse_color(new)
        if p is None:
            if isinstance(new, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: the replacement {_short(new)} is not a colour")
        ch = p[1]
        if p[0] == "HSB" and len(new.args) == 1:
            ch = (real_value(new.args[0], fname), 1.0, 1.0)
        out[m] = np.array(convert_channels(p[0], ch, space))
        if p[2] is not None:
            alpha[m] = p[2]
            need_alpha = need_alpha or p[2] != 1
    a = np.concatenate([out, alpha], axis=-1) if need_alpha else out
    typ = im.type if im.type != "Bit" or np.all((a == 0) | (a == 1)) else "Real32"
    return image_expr(Img(quantize(a, typ), typ, space, im.d3))


def _median_cut(X, n):
    boxes = [X]
    while len(boxes) < n:
        best, bi, bc = -1.0, None, None
        for i, b in enumerate(boxes):
            if len(np.unique(b, axis=0)) < 2:
                continue
            rng = b.max(axis=0) - b.min(axis=0)
            c = int(np.argmax(rng))
            if rng[c] > best:
                best, bi, bc = rng[c], i, c
        if bi is None:
            break
        b = boxes.pop(bi)
        b = b[np.argsort(b[:, bc], kind="stable")]
        mid = len(b) // 2
        boxes += [b[:mid], b[mid:]]
    return np.array([b.mean(axis=0) for b in boxes])


def _nearest(X, P):
    d = ((X[:, None, :] - P[None, :, :]) ** 2).sum(axis=-1)
    return np.argmin(d, axis=1)


def _dither(img2, P):
    """Floyd-Steinberg error diffusion of an (h, w, n) array onto palette P (raster order)."""
    a = np.array(img2, dtype=float)
    h, w, _ = a.shape
    out = np.zeros_like(a)
    for y in range(h):
        for x in range(w):
            old = a[y, x]
            k = int(np.argmin(((P - old) ** 2).sum(axis=1)))
            out[y, x] = P[k]
            e = old - P[k]
            if x + 1 < w:
                a[y, x + 1] += e * 7 / 16
            if y + 1 < h:
                if x > 0:
                    a[y + 1, x - 1] += e * 3 / 16
                a[y + 1, x] += e * 5 / 16
                if x + 1 < w:
                    a[y + 1, x + 1] += e * 1 / 16
    return out


@fn("ColorQuantize", area=AREA, status="added",
    notes="ColorQuantize[image, n] maps every pixel to the nearest (Euclidean in the image's colour channels) of at most "
          "n colours chosen by median cut (Method -> Automatic | \"MedianCut\"; \"MinVariance\" and \"Octree\" raise "
          "NotImplementedError); ColorQuantize[image, {col1, ...}] uses the given colours. Dithering -> True (default) "
          "is Floyd-Steinberg error diffusion in raster order, Dithering -> False plain nearest-colour mapping (Wolfram "
          "does not document its dithering, so dithered pixels can differ). ColorSpace Automatic images are "
          "quantized per channel value set. ColorQuantize[image] without n raises NotImplementedError (undocumented "
          "default). Alpha, type and colour space are kept. 2D only. Check: self-check (every output pixel is one of "
          "the palette colours, else ArithmeticError) and reference cases in tests.")
def _colorquantize(x: Expr, *rest: PosInteger | ListOf[Expr] | Option) -> Expr:
    fname = "ColorQuantize"
    pos, o = opts_split(rest, {"Dithering", "Method"}, fname)
    im = _img1(x, fname, allow3d=False)
    if not pos:
        raise NotImplementedError("ColorQuantize: give the number of colours n (the automatic default is not "
                                  "documented)")
    if len(pos) > 1:
        raise ValueError("ColorQuantize: expected ColorQuantize[image, n]")
    meth = o.get("Method", AUTO)
    if not (_is_sym(meth, "Automatic") or meth == "MedianCut"):
        if meth in ("MinVariance", "Octree"):
            raise NotImplementedError(f"ColorQuantize: Method {meth!r} is not supported")
        raise ValueError(f"ColorQuantize: unknown Method {_short(meth)}")
    dith = o.get("Dithering", True)
    if dith not in (True, False):
        raise ValueError("ColorQuantize: Dithering must be True or False")
    col = im.color()
    X = col.reshape(-1, col.shape[-1])
    if K.is_list(pos[0]):
        space = im.cs if im.cs != "Automatic" else source_space(im, fname)
        P = []
        for c in pos[0].args:
            p = parse_color(c)
            if p is None:
                if K.is_list(c) or is_real_value(c):
                    P.append(color_values(c, fname)[0])
                    continue
                if isinstance(c, Symbol):
                    raise Unevaluated()
                raise TypeError(f"ColorQuantize: {_short(c)} is not a colour")
            ch = p[1] if not (p[0] == "HSB" and len(c.args) == 1) else (real_value(c.args[0], fname), 1.0, 1.0)
            P.append(list(convert_channels(p[0], ch, space)))
        if not P:
            raise ValueError("ColorQuantize: the list of colours is empty")
        P = np.array(P, dtype=float)
        if P.shape[1] != X.shape[1]:
            raise ValueError("ColorQuantize: the colours do not match the image channels")
    else:
        n = int_arg(pos[0], fname, "number of colours", 1)
        P = _median_cut(X, n)
    P = quantize(P, im.type)
    if dith:
        out = _dither(col, P).reshape(-1, X.shape[1])
    else:
        out = P[_nearest(X, P)]
    if not all(any(np.allclose(r, p) for p in P) for r in np.unique(out, axis=0)):
        raise ArithmeticError("ColorQuantize: result failed verification")
    a = out.reshape(col.shape)
    if im.alpha:
        a = np.concatenate([a, im.alpha_array()], axis=-1)
    return image_expr(im.new(quantize(a, im.type)))


def _kmeans(X, k, seed=0, iters=100):
    """Lloyd's k-means with k-means++ seeding from a fixed seed; (labels, centres)."""
    uniq = np.unique(X, axis=0)
    if len(uniq) <= k:
        cents = uniq
        return _nearest(X, cents), cents
    rng = np.random.default_rng(seed)
    cents = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d = np.min(((X[:, None, :] - np.array(cents)[None]) ** 2).sum(-1), axis=1)
        if d.sum() == 0:
            break
        cents.append(X[rng.choice(len(X), p=d / d.sum())])
    cents = np.array(cents)
    lab_ = _nearest(X, cents)
    for _ in range(iters):
        new = np.array([X[lab_ == j].mean(axis=0) if np.any(lab_ == j) else cents[j] for j in range(len(cents))])
        nl = _nearest(X, new)
        cents = new
        if np.array_equal(nl, lab_):
            break
        lab_ = nl
    return lab_, cents


def _hex(rgb):
    return "#" + "".join(f"{int(round(min(max(v, 0), 1) * 255)):02x}" for v in rgb)


_DC_PROPS = ("Color", "LABColor", "HexRGBColor", "Count", "Coverage", "CoverageImage")


@fn("DominantColors", area=AREA, status="added",
    notes="DominantColors[image, n] clusters the pixel colours in LAB (k-means, k-means++ seeded deterministically; "
          "images with at most n distinct colours give exactly those) and returns the cluster colours (mean colour; "
          "RGBColor, or GrayLevel for grayscale images) ordered by size; DominantColors[image] is supported only for "
          "images with at most 12 distinct colours (the automatic cluster count is undocumented, otherwise "
          "NotImplementedError). Properties \"Color\", \"LABColor\", \"HexRGBColor\", \"Count\", \"Coverage\", "
          "\"CoverageImage\" (a \"Bit\" mask) or a list of them; MinColorDistance -> d merges clusters closer than d "
          "(ColorDistance, the larger keeping its colour); DistanceFunction for that merge. Method, ColorCoverage, "
          "Masking and output formats other than Automatic/\"List\" raise NotImplementedError. On natural images the "
          "clusters can differ from Wolfram's (undocumented method). Check: reference (images built from known "
          "colours) in tests.")
def _dominantcolors(x: Expr, *rest: PosInteger | String | ListOf[String] | Option) -> ListOf[Expr]:
    fname = "DominantColors"
    pos, o = opts_split(rest, {"MinColorDistance", "DistanceFunction"}, fname)
    if K.is_list(x):
        return L(*[_dominantcolors(v, *rest) for v in x.args])
    im = _img1(x, fname)
    labs, src = pixel_lab(im, fname)
    if len(pos) > 3:
        raise ValueError("DominantColors: expected DominantColors[image, n, prop, format]")
    if len(pos) == 3 and not (_is_sym(pos[2], "Automatic") or pos[2] == "List"):
        raise NotImplementedError(f"DominantColors: output format {_short(pos[2])} is not supported")
    X = labs.reshape(-1, 3)
    ncol = len(np.unique(X, axis=0))
    if not pos or _is_sym(pos[0], "Automatic"):
        if ncol > 12:
            raise NotImplementedError("DominantColors: give the number of colours n (the automatic choice is not "
                                      "documented)")
        k = ncol
    else:
        k = int_arg(pos[0], fname, "number of colours", 1)
    labels, _ = _kmeans(X, k)
    colflat = im.color().reshape(-1, im.color().shape[-1])
    groups = [np.flatnonzero(labels == j) for j in range(labels.max() + 1)]
    groups = [g for g in groups if len(g)]
    groups.sort(key=lambda g: (-len(g), tuple(X[g].mean(axis=0))))
    if "MinColorDistance" in o:
        dmin = real_arg(o["MinColorDistance"], fname, "MinColorDistance")
        dfun = distance_function(o.get("DistanceFunction"), fname)
        kept = []
        for g in groups:
            c = X[g].mean(axis=0)
            for kg in kept:
                if dfun(tuple(X[kg[1]].mean(axis=0)), tuple(c)) < dmin:
                    kg[0] = np.concatenate([kg[0], g])
                    break
            else:
                kept.append([g, g])
        # a merged cluster keeps the colour of its larger part (the one kept first)
        groups = [kg[0] for kg in kept]
        first = [kg[1] for kg in kept]
    else:
        first = groups
    prop = pos[1] if len(pos) > 1 else "Color"
    total = X.shape[0]
    gray = src == "Grayscale"

    def one(g, g0, p):
        if p == "Color":
            c = convert_array(colflat[g0].mean(axis=0)[None], src, "Grayscale" if gray else "RGB")[0] \
                if src not in ("Grayscale", "RGB") else colflat[g0].mean(axis=0)
            return Expr("GrayLevel", float(c[0])) if gray else Expr("RGBColor", *[float(v) for v in c[:3]])
        if p == "LABColor":
            return Expr("LABColor", *[float(v) for v in X[g0].mean(axis=0)])
        if p == "HexRGBColor":
            c = convert_array(colflat[g0].mean(axis=0)[None], src, "RGB")[0]
            return _hex(c)
        if p == "Count":
            return int(len(g))
        if p == "Coverage":
            return len(g) / total
        if p == "CoverageImage":
            m = np.zeros(total)
            m[g] = 1.0
            return image_expr(Img(m.reshape(im.a.shape[:-1] + (1,)), "Bit", "Automatic", im.d3))
        if isinstance(p, str):
            raise ValueError(f"DominantColors: unknown property {p!r} (one of {', '.join(_DC_PROPS)})")
        raise TypeError(f"DominantColors: {_short(p)} is not a property name")
    if prop == "Properties":
        return L(*_DC_PROPS)
    if K.is_list(prop):
        return L(*[L(*[one(g, g0, p) for p in prop.args]) for g, g0 in zip(groups, first)])
    return L(*[one(g, g0, prop) for g, g0 in zip(groups, first)])


# ───────────────────────────── colour functions, Colorize, gradients ─────────────────────────────

_CDF_TYPES = ("Gradients", "Indexed", "Named", "Physical")


def _cdf_parts(e, fname="ColorDataFunction"):
    if len(e.args) != 4:
        raise ValueError(f"{fname}: expected ColorDataFunction[name, type, range, f]")
    name, typ, rng, f = e.args
    if not isinstance(name, str):
        raise TypeError(f"{fname}: the scheme name must be a string")
    if typ not in _CDF_TYPES:
        raise ValueError(f"{fname}: the scheme type must be one of {', '.join(_CDF_TYPES)}")
    if not K.is_list(rng) or len(rng.args) not in (2, 3):
        raise ValueError(f"{fname}: the range must be {{min, max}} or {{min, max, step}}")
    return name, typ, [real_arg(v, fname, "range") for v in rng.args], f


@fn("ColorDataFunction", area=AREA, status="added",
    notes="ColorDataFunction[name, type, range, f] is the inert colour-scheme object (type \"Gradients\", \"Indexed\", "
          "\"Named\" or \"Physical\"); ColorDataFunction[...][x] evaluates f[x] with x clipped to the range for "
          "gradients (as ColorData gradients do) and must be an integer in range for indexed schemes (else ValueError). "
          "ColorData itself (the curated scheme table) is not available, so schemes are built explicitly, e.g. "
          "ColorDataFunction[\"RedBlue\", \"Gradients\", {0, 1}, Blend[{Red, Blue}, #] &]. Symbolic x stays "
          "unevaluated. Check: delegated (the stored function f) in tests.")
def _colordatafunction(*args: String | ListOf[Real] | Fn) -> Expr:
    _cdf_parts(Expr("ColorDataFunction", *args))
    raise Unevaluated()


def _cdf_apply(head, args, ev):
    try:
        name, typ, rng, f = _cdf_parts(head)
    except Unevaluated:
        return K.UNHANDLED
    if len(args) != 1:
        raise ValueError("ColorDataFunction: a colour scheme takes one argument")
    x = args[0]
    if not is_real_value(x) or isinstance(x, bool):
        if isinstance(x, (str, bool)) or K.is_list(x):
            raise TypeError(f"ColorDataFunction: {_short(x)} is not a number")
        return K.UNHANDLED
    v = real_value(x, "ColorDataFunction")
    if typ == "Indexed":
        if not K.is_int(x) or not (rng[0] <= int(x) <= rng[1]):
            raise ValueError(f"ColorDataFunction: index {_short(x)} is outside {rng[0]:g}..{rng[1]:g}")
        return ev(K.call(f, x))
    if v < rng[0] or v > rng[1]:
        x = rng[0] if v < rng[0] else rng[1]
    return ev(K.call(f, x))


try:
    from .. import kernel_lists as _KL
    _KL.COMPOUND_HEAD_HOOKS["ColorDataFunction"] = _cdf_apply
except (ImportError, AttributeError):  # pragma: no cover
    pass


def _gradient_colors(cols, fname):
    """Python function t (ndarray in [0, 1]) -> (..., 3|4) RGB(A) array and the output space, for a colour list."""
    items = list(cols.args)
    if len(items) < 1:
        raise ValueError(f"{fname}: the colour list is empty")
    pairs = all(K.is_list(i) and len(i.args) == 2 for i in items)
    if pairs:
        xs = [real_arg(i.args[0], fname, "position") for i in items]
        cs = [i.args[1] for i in items]
    else:
        cs = items
        xs = list(np.linspace(0, 1, len(items))) if len(items) > 1 else [0.0]
    rgba = []
    for c in cs:
        v = to_rgba(c)
        if v is None:
            if isinstance(c, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: {_short(c)} is not a colour")
        rgba.append(v)
    gray = all(parse_color(c)[0] == "Grayscale" for c in cs)
    has_alpha = any(parse_color(c)[2] is not None for c in cs)
    R = np.array(rgba)
    order = np.argsort(xs, kind="stable")
    xs = np.array(xs)[order]
    R = R[order]

    def f(t):
        t = np.clip(t, xs[0], xs[-1])
        chans = [np.interp(t, xs, R[:, i]) for i in range(4)]
        out = np.stack(chans if has_alpha else chans[:3], axis=-1)
        if gray:
            out = np.concatenate([out[..., :1], out[..., 3:]], axis=-1)
        return out
    return f, ("Grayscale" if gray else "RGB")


def _fn_colors(fspec, fname):
    """t-array -> colour array via a language function returning colours (evaluated per distinct t)."""
    def f(t):
        flat = np.asarray(t, dtype=float).reshape(-1)
        uniq, inv = np.unique(flat, return_inverse=True)
        vals = []
        for u in uniq:
            c = _ev(K.call(fspec, float(u)))
            v = to_rgba(c)
            if v is None:
                if is_real_value(c):
                    g = real_value(c, fname)
                    v = (g, g, g, 1.0)
                else:
                    raise TypeError(f"{fname}: the colour function returned {_short(c)}, not a colour")
            vals.append(v)
        V = np.array(vals)
        if np.all(V[:, 3] == 1):
            V = V[:, :3]
        return V[np.asarray(inv).reshape(-1)].reshape(np.shape(t) + (V.shape[1],))
    return f, "RGB"


def gradient_fn(gcol, fname):
    if K.is_list(gcol):
        return _gradient_colors(gcol, fname)
    if isinstance(gcol, str):
        raise NotImplementedError(f"{fname}: named colour schemes (ColorData[\"{gcol}\"]) are not available; give a "
                                  f"list of colours or a function")
    if isinstance(gcol, Symbol) and gcol.name in ("GrayLevel", "Hue"):
        return _fn_colors(gcol, fname)
    if isinstance(gcol, (Expr, Symbol)) and not is_real_value(gcol):
        return _fn_colors(gcol, fname)
    raise TypeError(f"{fname}: {_short(gcol)} is not a colour gradient specification")


_SIDE = {"Left": (0.0, None), "Right": (1.0, None), "Bottom": (None, 0.0), "Top": (None, 1.0), "Center": (0.5, 0.5)}


def _position(p, w, h, fname):
    """Image coordinates of a position {x, y} or a named side/corner ({Left, Top}, Center, ...)."""
    if isinstance(p, Symbol) and p.name in _SIDE:
        fx, fy = _SIDE[p.name]
        return np.array([(0.5 if fx is None else fx) * w, (0.5 if fy is None else fy) * h])
    if K.is_list(p) and len(p.args) == 2:
        out = []
        for i, v in enumerate(p.args):
            if isinstance(v, Symbol) and v.name in _SIDE:
                f = _SIDE[v.name][0 if v.name in ("Left", "Right") else 1] if v.name != "Center" else 0.5
                if f is None:
                    raise ValueError(f"{fname}: {v.name} cannot be used for the {'xy'[i]} coordinate")
                out.append(f * (w if i == 0 else h))
            else:
                out.append(real_arg(v, fname, "position"))
        return np.array(out)
    if isinstance(p, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_short(p)} is not a position")


def _gradient(args, fname, radial):
    pos, o = opts_split(args, set(), fname)
    if len(pos) > 3:
        raise ValueError(f"{fname}: expected {fname}[gcol, size, type]")
    spec = pos[0] if pos else L(Expr("GrayLevel", 0), Expr("GrayLevel", 1))
    dims = size_arg(pos[1] if len(pos) > 1 else None, fname, allow3d=False)
    typ = type_arg(pos[2], fname) if len(pos) > 2 else None
    w, h = dims
    pts = None
    if isinstance(spec, Expr) and spec.head == "Rule" and len(spec.args) == 2:
        pts, spec = spec.args
        if not (K.is_list(pts) and len(pts.args) == 2):
            raise ValueError(f"{fname}: the positions must be {{pos1, pos2}}")
    f, space = gradient_fn(spec, fname)
    if pts is not None:
        p1, p2 = (_position(p, w, h, fname) for p in pts.args)
    elif radial:
        p1, p2 = np.array([w / 2, h / 2]), np.array([float(w), float(h)])
    else:
        p1, p2 = np.array([0.0, h / 2]), np.array([float(w), h / 2])
    xs = np.arange(w) + 0.5
    ys = h - (np.arange(h) + 0.5)                      # row 0 is the top
    X, Y = np.meshgrid(xs, ys)
    v = p2 - p1
    n2 = float(v @ v)
    if n2 == 0:
        raise ValueError(f"{fname}: the two positions must differ")
    if radial:
        t = np.hypot(X - p1[0], Y - p1[1]) / math.sqrt(n2)
    else:
        t = ((X - p1[0]) * v[0] + (Y - p1[1]) * v[1]) / n2
    a = f(np.clip(t, 0.0, 1.0))
    typ = typ or "Real32"
    return image_expr(Img(quantize(a, typ), typ, space if a.shape[-1] in (1, 2, 3, 4) else "Automatic"))


@fn("LinearGradientImage", area=AREA, status="added",
    notes="LinearGradientImage[gcol, size, type] (default {Black, White}, {150, 150}, \"Real32\") colours each pixel "
          "centre by t = the normalised distance from the line through pos1 perpendicular to pos2 - pos1 (default "
          "pos1 = {Left, Center}, pos2 = {Right, Center}), clipped to [0, 1]; LinearGradientImage[{pos1, pos2} -> "
          "gcol, ...] with positions {x, y} in image coordinates or named sides. gcol: a colour list (piecewise "
          "linear in RGB, or in gray when all colours are GrayLevel, like Blend), {{x1, c1}, ...}, or a function "
          "returning colours (GrayLevel, Hue, pure functions, ColorDataFunction). Named ColorData schemes and 3D "
          "sizes raise NotImplementedError. Pixels are sampled at their centres, so values can differ from "
          "Wolfram's in the last digits. Check: exact (the formula for t, endpoint colours) in tests.")
def _lineargradientimage(*args: Expr | ListOf[Expr] | Integer | ListOf[PosInteger] | String) -> Expr:
    return _gradient(args, "LinearGradientImage", False)


@fn("RadialGradientImage", area=AREA, status="added",
    notes="RadialGradientImage[gcol, size, type] colours each pixel centre by t = |p - pos1| / |pos2 - pos1| (default "
          "pos1 = Center, pos2 = the top-right corner), clipped to [0, 1]; {pos1, pos2} -> gcol, gcol forms, sizes and "
          "types as for LinearGradientImage. Check: exact (the formula for t, symmetry) in tests.")
def _radialgradientimage(*args: Expr | ListOf[Expr] | Integer | ListOf[PosInteger] | String) -> Expr:
    return _gradient(args, "RadialGradientImage", True)


def _palette(k):
    """Distinct colour for label k >= 1 (golden-ratio hue steps)."""
    from .wl_graphics_color import _hsb_to_rgb
    return _hsb_to_rgb((0.13 + 0.618033988749895 * (k - 1)) % 1.0, 0.75, 0.95)


@fn("Colorize", area=AREA, status="added",
    notes="Colorize[m] for an integer matrix (or 3D array) gives a \"Byte\" RGB image with 0 and negative entries black "
          "and each positive integer a distinct colour (golden-ratio hue steps; Wolfram's own palette is not "
          "documented, so the colours differ); Colorize[image, ColorFunction -> f] maps intensities (multichannel "
          "images via Rec. 601 luma, clipped to [0, 1]) through f (GrayLevel, Hue, a pure function or "
          "ColorDataFunction; with ColorFunctionScaling -> False the raw values); ColorRules -> {v -> col, ...} "
          "overrides. Colorize[image] without ColorFunction needs ColorData[\"DarkRainbow\"] and raises "
          "NotImplementedError. Check: exact (distinct labels give distinct colours, rules honoured) in tests.")
def _colorize(x: Expr, *opts: Option) -> Expr:
    fname = "Colorize"
    _, o = opts_split(opts, {"ColorFunction", "ColorFunctionScaling", "ColorRules"}, fname)
    if _ and len(_) > 0:
        raise ValueError("Colorize: expected Colorize[m] or Colorize[image] with options")
    rules = {}
    if "ColorRules" in o:
        rl = o["ColorRules"]
        for r in (rl.args if K.is_list(rl) else [rl]):
            if not (isinstance(r, Expr) and r.head == "Rule"):
                raise ValueError("Colorize: ColorRules must be a list of value -> colour rules")
            c = to_rgba(r.args[1])
            if c is None:
                raise TypeError(f"Colorize: {_short(r.args[1])} is not a colour")
            rules[real_arg(r.args[0], fname, "rule value")] = c
    scaling = o.get("ColorFunctionScaling", True)
    if scaling not in (True, False):
        raise ValueError("Colorize: ColorFunctionScaling must be True or False")
    if isinstance(x, Expr) and x.head in IMAGE_HEADS:
        im = _img1(x, fname)
        g = im.color() if im.color().shape[-1] == 1 else convert_array(im.color(), source_space(im, fname),
                                                                       "Grayscale")
        g = g[..., 0]
        if "ColorFunction" not in o and not rules:
            raise NotImplementedError("Colorize: the default colour function ColorData[\"DarkRainbow\"] is not "
                                      "available; give ColorFunction -> f")
        vals = np.clip(g, 0, 1) if scaling else g
        d3 = im.d3
    else:
        a = _numeric_array(x, fname)
        if a is None:
            if isinstance(x, Symbol):
                raise Unevaluated()
            raise TypeError(f"Colorize: {_short(x)} is not an integer array or an image")
        if a.ndim not in (2, 3) or np.any(a != np.rint(a)):
            raise ValueError("Colorize: expected a matrix (or 3D array) of integers")
        g = a
        d3 = a.ndim == 3
        vals = None
    shape = g.shape
    out = np.zeros(shape + (4,))
    out[..., 3] = 1.0
    if vals is None:
        if "ColorFunction" in o:
            mn, mx = float(g.min()), float(g.max())
            t = (g - mn) / (mx - mn) if (scaling and mx > mn) else (g * 0.0 if scaling else g)
            f, _s = gradient_fn(o["ColorFunction"], fname)
            c = f(t)
            out[..., :c.shape[-1] if c.shape[-1] >= 3 else 3] = c if c.shape[-1] >= 3 else np.repeat(c, 3, -1)
        else:
            for k in np.unique(g):
                if k > 0:
                    out[g == k, :3] = _palette(int(k))
    else:
        f, _s = gradient_fn(o["ColorFunction"], fname) if "ColorFunction" in o else (None, None)
        if f is not None:
            c = f(vals)
            if c.shape[-1] in (1, 2):
                c = np.concatenate([np.repeat(c[..., :1], 3, -1), c[..., 1:]], axis=-1)
            out[..., :c.shape[-1]] = c
    for v, c in rules.items():
        out[g == v] = c
    if np.all(out[..., 3] == 1):
        out = out[..., :3]
    return image_expr(Img(quantize(out, "Byte"), "Byte", "RGB", d3))


# ───────────────────────────── image arithmetic ─────────────────────────────

def _operand(x, ref, fname):
    """(array broadcastable against ref's colour channels, Img or None) for an image, number, colour or list."""
    if isinstance(x, Expr) and x.head in IMAGE_HEADS:
        im = parse_image(x, fname)
        if im.d3 != ref.d3 or im.a.shape[:-1] != ref.a.shape[:-1]:
            raise NotImplementedError(f"{fname}: images of different dimensions are not supported (Wolfram centres "
                                      f"the second image)")
        return im.color(), im
    v = value_for(ref.new(ref.color(), cs=ref.cs if not ref.alpha else ref.cs), x, fname)
    n = ref.color().shape[-1]
    return v[:n], None


def _arith(fname, op, args, clip_real=False):
    items = list(args[0].args) if len(args) == 1 and K.is_list(args[0]) else list(args)
    if len(items) < 2:
        raise ValueError(f"{fname}: expected an image and at least one more operand")
    im = parse_image(items[0], fname)
    col = np.array(im.color(), dtype=float)
    typ, cs, alpha = im.type, im.cs, (im.alpha_array() if im.alpha else None)
    for x in items[1:]:
        b, other = _operand(x, Img(col, typ, cs if alpha is None else "Automatic", im.d3) if alpha is None
                            else Img(np.concatenate([col, alpha], -1), typ, cs, im.d3), fname)
        if other is not None:
            typ = larger_type(typ, other.type)
            if col.shape[-1] == 1 and b.shape[-1] > 1:
                cs = other.cs
                if other.alpha and alpha is None:
                    alpha = other.alpha_array()
            elif b.shape[-1] != 1 and b.shape[-1] != col.shape[-1]:
                raise ValueError(f"{fname}: the images have different numbers of channels")
        col = op(col, b)
    if typ in MAXV:
        col = np.clip(col, 0, 1)
    out = np.concatenate([col, alpha], -1) if alpha is not None else col
    return image_expr(Img(quantize(out, typ), typ, cs, im.d3))


_ARITH_NOTES = ("Operands: images of the same dimensions (a single-channel image is applied to every channel of the "
                "other; different channel counts raise ValueError; different dimensions NotImplementedError), "
                "numbers, colours (converted to the image's colour space) or channel lists; folds over several "
                "operands and accepts a single list. Alpha channels are kept from the first image. The result has "
                "the largest type of the images; integer types are clipped to 0..1, real types are not.")


@fn("ImageAdd", area=AREA, status="added",
    notes="ImageAdd[image, x1, x2, ...] adds the operands to each channel value. " + _ARITH_NOTES +
          " For \"Bit\" images it is the logical OR. Check: exact (pixelwise sums) in tests.")
def _imageadd(*args: Expr | Real | ListOf[Real]) -> Expr:
    return _arith("ImageAdd", lambda a, b: a + b, args)


@fn("ImageSubtract", area=AREA, status="added",
    notes="ImageSubtract[image, x1, x2, ...] subtracts the operands from each channel value. " + _ARITH_NOTES +
          " Check: exact (pixelwise differences; ImageAdd[ImageSubtract[i, j], j] == i for real images) in tests.")
def _imagesubtract(*args: Expr | Real | ListOf[Real]) -> Expr:
    return _arith("ImageSubtract", lambda a, b: a - b, args)


@fn("ImageMultiply", area=AREA, status="added",
    notes="ImageMultiply[image, x1, x2, ...] multiplies each channel value by the operands. " + _ARITH_NOTES +
          " For \"Bit\" images it is the logical AND. Check: exact (pixelwise products) in tests.")
def _imagemultiply(*args: Expr | Real | ListOf[Real]) -> Expr:
    return _arith("ImageMultiply", lambda a, b: a * b, args)


@fn("ImageDifference", area=AREA, status="added",
    notes="ImageDifference[image1, image2] gives |image1 - image2| per channel for images of equal dimensions (a "
          "single-channel image is compared with every channel of the other); largest type; for \"Bit\" images the "
          "logical XOR. Check: exact (symmetry and |a - b|) in tests.")
def _imagedifference(a: Expr, b: Expr) -> Expr:
    for v in (a, b):
        if not (isinstance(v, Expr) and v.head in IMAGE_HEADS):
            _img1(v, "ImageDifference")
    return _arith("ImageDifference", lambda u, v: np.abs(u - v), (a, b))


def _per_channel(x, n, fname, what, allow_auto=False):
    """List of n floats (or None for Automatic) from a scalar or a per-channel list."""
    if K.is_list(x):
        vals = list(x.args)
        if len(vals) != n:
            raise ValueError(f"{fname}: {what} lists must have one value per channel ({n})")
    else:
        vals = [x] * n
    out = []
    for v in vals:
        if allow_auto and _is_sym(v, "Automatic"):
            out.append(None)
        else:
            out.append(real_arg(v, fname, what))
    return out


def _range_spec(spec, n, fname, what):
    """[(lo_i, hi_i)] (None for Automatic) of {min, max} with scalar or per-channel entries."""
    if not (K.is_list(spec) and len(spec.args) == 2):
        raise ValueError(f"{fname}: {what} must be {{min, max}}")
    lo = _per_channel(spec.args[0], n, fname, what, True)
    hi = _per_channel(spec.args[1], n, fname, what, True)
    return list(zip(lo, hi))


@fn("ImageClip", area=AREA, status="added",
    notes="ImageClip[image] clips channel values to [0, 1]; ImageClip[image, {min, max}] to [min, max] (per-channel "
          "lists allowed); ImageClip[image, {min, max}, {vmin, vmax}] gives vmin below min and vmax above max. Type "
          "kept. Check: exact in tests.")
def _imageclip(x: Expr, rng: ListOf[Real] = None, vals: ListOf[Real] = None) -> Expr:
    im = _img1(x, "ImageClip")
    n = im.c
    r = [(0.0, 1.0)] * n if rng is None else _range_spec(rng, n, "ImageClip", "the range")
    v = r if vals is None else _range_spec(vals, n, "ImageClip", "the values")
    a = np.array(im.a, dtype=float)
    for i in range(n):
        lo, hi = r[i]
        if lo is None or hi is None:
            raise ValueError("ImageClip: the clipping range must be numeric")
        vlo, vhi = v[i]
        ch = a[..., i]
        below, above = ch < lo, ch > hi
        ch[below] = vlo
        ch[above] = vhi
    return image_expr(im.new(quantize(a, im.type)))


@fn("ImageAdjust", area=AREA, status="added",
    notes="ImageAdjust[image] rescales each colour channel from its own min..max to 0..1; ImageAdjust[image, c], "
          "{c, b} and {c, b, gamma} apply Clip[(b + 1)(c + 1) x^gamma - c/2, {0, 1}] (the documented formula; zero "
          "adjustment {0, 0, 1}); ImageAdjust[image, corr, {inmin, inmax}] first rescales inmin..inmax to 0..1 and "
          "ImageAdjust[image, corr, {inmin, inmax}, {outmin, outmax}] to outmin..outmax (clipped), Automatic meaning "
          "the channel minimum or maximum, per-channel lists allowed for every parameter. Colour images in spaces "
          "other than Grayscale/RGB are converted to RGB first; alpha is kept; type kept. A constant channel "
          "rescales to 0. Check: exact (ImageAdjust[i, -2] == ColorNegate[i], documented formula) in tests.")
def _imageadjust(x: Expr, corr: Real | ListOf[Real] = None, inr: ListOf[Real] = None,
                 outr: ListOf[Real] = None) -> Expr:
    fname = "ImageAdjust"
    im = _img1(x, fname)
    if im.cs not in ("Automatic", "Grayscale", "RGB"):
        im = to_space(im, "RGB", fname)
    col = np.array(im.color(), dtype=float)
    n = col.shape[-1]
    flat = col.reshape(-1, n)
    mins, maxs = flat.min(axis=0), flat.max(axis=0)
    if inr is None and outr is None and corr is None:
        inl = list(zip(mins, maxs))
    elif inr is not None:
        inl = [(mins[i] if lo is None else lo, maxs[i] if hi is None else hi)
               for i, (lo, hi) in enumerate(_range_spec(inr, n, fname, "the input range"))]
    else:
        inl = None
    outl = [(0.0, 1.0)] * n if outr is None else [
        (0.0 if lo is None else lo, 1.0 if hi is None else hi)
        for lo, hi in _range_spec(outr, n, fname, "the output range")]
    if inl is not None:
        for i in range(n):
            lo, hi = inl[i]
            olo, ohi = outl[i]
            t = (col[..., i] - lo) / (hi - lo) if hi != lo else np.zeros_like(col[..., i])
            col[..., i] = np.clip(olo + t * (ohi - olo), min(olo, ohi), max(olo, ohi))
    if corr is not None:
        if K.is_list(corr):
            parts = list(corr.args)
            if not 1 <= len(parts) <= 3:
                raise ValueError("ImageAdjust: the correction must be c, {c, b} or {c, b, gamma}")
        else:
            parts = [corr]
        defaults = [0, 0, 1]
        vals = [_per_channel(parts[k], n, fname, "correction") if k < len(parts) else [defaults[k]] * n
                for k in range(3)]
        for i in range(n):
            c, b, g = vals[0][i], vals[1][i], vals[2][i]
            if g <= 0:
                raise ValueError("ImageAdjust: gamma must be positive")
            col[..., i] = np.clip((b + 1) * (c + 1) * np.power(np.clip(col[..., i], 0, None), g) - c / 2, 0, 1)
    out = np.concatenate([col, im.alpha_array()], -1) if im.alpha else col
    return image_expr(im.new(quantize(out, im.type)))


# ───────────────────────────── per-pixel functions ─────────────────────────────

def _px_expr(v):
    """A pixel's channel vector as a value for user functions: a number for one channel, else a list."""
    v = [float(t) for t in v]
    return v[0] if len(v) == 1 else L(*v)


def _result_vec(r, fname):
    if K.is_list(r):
        return [real_arg(t, fname, "result value") for t in r.args]
    if isinstance(r, Expr) and parse_color(r) is not None:
        return list(to_rgba(r)[:3])
    return [real_arg(r, fname, "result value")]


def _apply_pixels(fname, f, ims, interleave, indexed=False):
    """Array (h, w, n) of f applied to the pixels of the images (cached on the pixel values unless indexed)."""
    ref = ims[0]
    shape = ref.a.shape[:-1]
    cache = {}
    rows = []
    if not interleave:
        if len(ims) != 1:
            raise NotImplementedError(f"{fname}: Interleaving -> False is supported for one image only")
        out = np.zeros(ref.a.shape)
        it = np.ndindex(*ref.a.shape)
        for idx in it:
            v = float(ref.a[idx])
            args = [v] + ([L(*[i + 1 for i in idx[:-1]])] if indexed else [])
            key = None if indexed else v
            if key is not None and key in cache:
                r = cache[key]
            else:
                r = _result_vec(_ev(K.call(f, *args)), fname)
                if len(r) != 1:
                    raise ValueError(f"{fname}: with Interleaving -> False the function must return a number")
                if key is not None:
                    cache[key] = r
            out[idx] = r[0]
        return out
    n_out = None
    for idx in np.ndindex(*shape):
        vals = [tuple(im.a[idx]) for im in ims]
        key = None if indexed else tuple(vals)
        if key is not None and key in cache:
            r = cache[key]
        else:
            args = [_px_expr(v) for v in vals] + ([L(*[i + 1 for i in idx])] if indexed else [])
            r = _result_vec(_ev(K.call(f, *args)), fname)
            if key is not None:
                cache[key] = r
        if n_out is None:
            n_out = len(r)
        elif len(r) != n_out:
            raise ValueError(f"{fname}: the function returned lists of different lengths")
        rows.append(r)
    return np.array(rows, dtype=float).reshape(shape + (n_out,))


def _image_list(x, fname):
    if K.is_list(x):
        ims = [parse_image(v, fname) for v in x.args]
        if not ims:
            raise ValueError(f"{fname}: the list of images is empty")
        if len({(i.a.shape[:-1], i.d3) for i in ims}) != 1:
            raise ValueError(f"{fname}: the images must have the same dimensions")
        return ims, True
    return [parse_image(x, fname)], False


def _pixel_result(ims, a):
    ref = ims[0]
    typ = larger_type(*[i.type for i in ims])
    cs = ref.cs if a.shape[-1] == ref.c else default_space(a.shape[-1])
    if typ in MAXV:
        a = np.clip(a, 0, 1)
    return image_expr(Img(quantize(a, typ), typ, cs, ref.d3))


def _interleave_opt(o, fname, default=True):
    v = o.get("Interleaving", default)
    if v not in (True, False):
        raise ValueError(f"{fname}: Interleaving must be True or False")
    return v


@fn("ImageApply", area=AREA, status="added",
    notes="ImageApply[f, image] applies f to each pixel's channel list (a number for single-channel images) on the 0..1 "
          "scale and builds an image from the results (numbers or equal-length lists; colours give their RGB "
          "values); ImageApply[f, {image1, image2, ...}] passes the corresponding pixels as a sequence (largest "
          "type). Interleaving -> False applies f to each channel value. Evaluations are cached on the pixel value, "
          "as Wolfram allows. Integer types are clipped. Masking raises NotImplementedError. Check: exact (identity "
          "and closed-form maps) in tests.")
def _imageapply(f: Fn, x: Expr | ListOf[Expr], *opts: Option) -> Expr:
    _, o = opts_split(opts, {"Interleaving"}, "ImageApply")
    if _:
        raise ValueError("ImageApply: expected ImageApply[f, image, options]")
    ims, _many = _image_list(x, "ImageApply")
    return _pixel_result(ims, _apply_pixels("ImageApply", f, ims, _interleave_opt(o, "ImageApply")))


@fn("ImageApplyIndexed", area=AREA, status="added",
    notes="ImageApplyIndexed[f, image] applies f[pixel, {row, column}] (row 1 at the top; {slice, row, column} for "
          "Image3D) and builds an image from the results; ImageApplyIndexed[f, {image1, ...}] gives f[p1, p2, ..., "
          "{row, column}]. Interleaving -> False passes single channel values with {row, column, channel}. "
          "Check: exact in tests.")
def _imageapplyindexed(f: Fn, x: Expr | ListOf[Expr], *opts: Option) -> Expr:
    _, o = opts_split(opts, {"Interleaving"}, "ImageApplyIndexed")
    if _:
        raise ValueError("ImageApplyIndexed: expected ImageApplyIndexed[f, image, options]")
    ims, _many = _image_list(x, "ImageApplyIndexed")
    return _pixel_result(ims, _apply_pixels("ImageApplyIndexed", f, ims,
                                            _interleave_opt(o, "ImageApplyIndexed"), indexed=True))


@fn("ImageScan", area=AREA, status="added",
    notes="ImageScan[f, image] evaluates f on each pixel (channel list, or a number for one channel, on the 0..1 "
          "scale) row by row from the top left and returns Null; Interleaving -> False scans single channel values. "
          "Check: exact (side effects collected with Sow/Reap or assignments) in tests.")
def _imagescan(f: Fn, x: Expr, *opts: Option) -> Expr:
    _, o = opts_split(opts, {"Interleaving"}, "ImageScan")
    if _:
        raise ValueError("ImageScan: expected ImageScan[f, image, options]")
    im = _img1(x, "ImageScan")
    il = _interleave_opt(o, "ImageScan")
    for idx in np.ndindex(*im.a.shape[:-1]):
        if il:
            _ev(K.call(f, _px_expr(im.a[idx])))
        else:
            for v in im.a[idx]:
                _ev(K.call(f, float(v)))
    return None


def _axis_indices(v, n, fname):
    """0-based pixel indices along one axis for a pixel coordinate (1-based centres; .5 lies between pixels)."""
    t = real_arg(v, fname, "position")
    f = math.floor(t)
    if abs(t - f - 0.5) < 1e-12:
        idx = [f, f + 1]                              # on the boundary between pixels f and f + 1
    else:
        idx = [int(math.floor(t + 0.5))]
    out = [i - 1 for i in idx]
    if any(i < 0 or i >= n for i in out):
        return None
    return out


def _pixel_positions(spec, im, fname):
    """List of lists of (row, col) 0-based array indices for a PixelValue position spec (one group per position)."""
    def one(p):
        if not (K.is_list(p) and len(p.args) == 2):
            raise TypeError(f"{fname}: {_short(p)} is not a pixel position {{x, y}}")
        if any(isinstance(v, Expr) and v.head == "Span" for v in p.args):
            raise NotImplementedError(f"{fname}: span positions are not supported")
        xs = _axis_indices(p.args[0], im.w, fname)
        ys = _axis_indices(p.args[1], im.h, fname)
        if xs is None or ys is None:
            raise ValueError(f"{fname}: position {_short(p)} lies outside the {im.w} x {im.h} image")
        return [(im.h - 1 - y, x) for y in ys for x in xs]
    if K.is_list(spec) and spec.args and all(K.is_list(p) for p in spec.args):
        return [one(p) for p in spec.args], True
    if isinstance(spec, Expr) and spec.head in IMAGE_HEADS:
        raise NotImplementedError(f"{fname}: marker images are not supported")
    if isinstance(spec, Symbol):
        raise Unevaluated()
    return [one(spec)], False


def _value_out(v, typ):
    if typ is None:
        vals = [float(t) for t in v]
    elif typ in MAXV:
        vals = [int(t) for t in stored(np.array(v), typ)]
    else:
        vals = [float(t) for t in v]
    return vals[0] if len(vals) == 1 else L(*vals)


@fn("PixelValue", area=AREA, status="added",
    notes="PixelValue[image, {x, y}] gives the channel values (0..1 scale; a number for one channel) of the pixel in "
          "column x and row y counted from the bottom (integer coordinates at pixel centres, {1, 1} the bottom-left "
          "pixel); a non-integer position gives the nearest pixel and a position on a boundary the average of the "
          "adjacent pixels; a list of positions gives a list; PixelValue[image, pos, type] converts (\"Byte\" gives "
          "integers). Positions outside the image raise ValueError (no Padding); spans and marker images raise "
          "NotImplementedError; 2D only. Check: exact (agreement with ImageData) in tests.")
def _pixelvalue(x: Expr, pos: ListOf[Real] | ListOf[ListOf[Real]], typ: String = None) -> Real | ListOf[Real]:
    im = _img1(x, "PixelValue", allow3d=False)
    t = type_arg(typ, "PixelValue") if typ is not None else None
    groups, many = _pixel_positions(pos, im, "PixelValue")
    vals = [np.mean([im.a[r, c] for r, c in g], axis=0) for g in groups]
    out = [_value_out(v, t) for v in vals]
    return L(*out) if many else out[0]


def _replace_positions(img, groups, val, typ, fname):
    a = np.array(img.a, dtype=float)
    v = _replacement_value(img, val, typ, fname)
    for g in groups:
        for r, c in g:
            a[r, c] = v
    return image_expr(img.new(quantize(a, img.type)))


def _replacement_value(img, val, typ, fname):
    if typ is not None and typ in MAXV and parse_color(val) is None:
        ch = [real_arg(t, fname, "value") / MAXV[typ] for t in (val.args if K.is_list(val) else [val])]
        v = np.array(ch if len(ch) > 1 else ch * img.c)
        if len(v) != img.c:
            raise ValueError(f"{fname}: {len(ch)} channel values given for a {img.c}-channel image")
        return v
    return value_for(img, val, fname)


def _rules(spec, fname):
    rl = list(spec.args) if K.is_list(spec) and all(isinstance(r, Expr) and r.head == "Rule" for r in spec.args) \
        else [spec]
    for r in rl:
        if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2):
            if isinstance(r, Symbol):
                raise Unevaluated()
            raise TypeError(f"{fname}: expected position -> value rules, not {_short(r)}")
    return rl


def _span_groups(p, im, fname, coords):
    """Groups of (row, col) for a position with Span coordinates (xmin ;; xmax, ...); coords 'pixel' or 'image'."""
    out = []
    axes = []
    for i, v in enumerate(p.args):
        n = im.w if i == 0 else im.h
        if isinstance(v, Expr) and v.head == "Span":
            lo = real_arg(v.args[0], fname, "position") if not _is_sym(v.args[0], "All") else (1 if coords == "pixel"
                                                                                                else 0)
            hi = real_arg(v.args[1], fname, "position") if len(v.args) > 1 and not _is_sym(v.args[1], "All") \
                else (n if coords == "pixel" else n)
            if coords == "pixel":
                idx = [k for k in range(n) if lo - 1e-12 <= k + 1 <= hi + 1e-12]
            else:
                idx = [k for k in range(n) if k + 1 > lo and k < hi]
        else:
            idx = _axis_indices(v, n, fname) if coords == "pixel" else _image_axis(v, n, fname)
            if idx is None:
                raise ValueError(f"{fname}: position {_short(p)} lies outside the image")
        axes.append(idx)
    out.append([(im.h - 1 - y, x) for y in axes[1] for x in axes[0]])
    return out


def _image_axis(v, n, fname):
    """0-based pixel indices along an axis containing the image coordinate v (both pixels on a boundary)."""
    t = real_arg(v, fname, "position")
    if t < 0 or t > n:
        return None
    f = math.floor(t)
    if abs(t - round(t)) < 1e-12:
        k = int(round(t))
        return [i for i in (k - 1, k) if 0 <= i < n]
    return [int(f)] if f < n else [n - 1]


@fn("ReplacePixelValue", area=AREA, status="added",
    notes="ReplacePixelValue[image, pos -> val] sets the pixel at pixel position pos ({x, y} with {1, 1} the bottom-left "
          "pixel centre; a position on a boundary changes all adjacent pixels; spans xmin ;; xmax set a block) to val "
          "(a number, a channel list or a colour converted to the image's colour space); a list of rules applies them "
          "in turn; ReplacePixelValue[image, pos -> val, type] reads val in that type's range (\"Byte\": 0..255). Type "
          "kept (values clipped and rounded for integer types). Positions outside the image raise ValueError. "
          "Check: exact (read back with PixelValue) in tests.")
def _replacepixelvalue(x: Expr, spec: Rule | ListOf[Rule], typ: String = None) -> Expr:
    fname = "ReplacePixelValue"
    im = _img1(x, fname, allow3d=False)
    t = type_arg(typ, fname) if typ is not None else None
    for r in _rules(spec, fname):
        p, val = r.args
        if K.is_list(p) and len(p.args) == 2 and any(isinstance(v, Expr) and v.head == "Span" for v in p.args):
            groups = _span_groups(p, im, fname, "pixel")
        else:
            groups, _m = _pixel_positions(p, im, fname)
        im = parse_image(_replace_positions(im, groups, val, t, fname), fname)
    return image_expr(im)


@fn("ReplaceImageValue", area=AREA, status="added",
    notes="ReplaceImageValue[image, pos -> val] sets the pixel containing the image-coordinate position pos ({0, 0} the "
          "bottom-left corner, {w, h} the top-right; a position on a pixel boundary changes all adjacent pixels; "
          "spans set the pixels inside) to val (number, channel list or colour); lists of rules and a type argument "
          "as for ReplacePixelValue. DataRange other than Full raises NotImplementedError; positions outside the "
          "image raise ValueError. Check: exact (agreement with ReplacePixelValue at pos + 1/2) in tests.")
def _replaceimagevalue(x: Expr, spec: Rule | ListOf[Rule], typ: String = None) -> Expr:
    fname = "ReplaceImageValue"
    im = _img1(x, fname, allow3d=False)
    t = type_arg(typ, fname) if typ is not None else None
    for r in _rules(spec, fname):
        p, val = r.args
        pts = list(p.args) if K.is_list(p) and p.args and all(K.is_list(q) for q in p.args) else [p]
        groups = []
        for q in pts:
            if not (K.is_list(q) and len(q.args) == 2):
                raise TypeError(f"{fname}: {_short(q)} is not a position {{x, y}}")
            groups += _span_groups(q, im, fname, "image")
        im = parse_image(_replace_positions(im, groups, val, t, fname), fname)
    return image_expr(im)


def _data_range(o, im, fname):
    dr = o.get("DataRange", Symbol("Full"))
    if _is_sym(dr, "Full"):
        return (0.0, float(im.w)), (0.0, float(im.h))
    if _is_sym(dr, "Automatic"):
        return (0.0, 1.0), (0.0, im.h / im.w)
    if K.is_list(dr) and len(dr.args) == 2 and all(K.is_list(v) and len(v.args) == 2 for v in dr.args):
        (a, b), (c, d) = [[real_arg(t, fname, "DataRange") for t in v.args] for v in dr.args]
        if a == b or c == d:
            raise ValueError(f"{fname}: DataRange must have non-empty ranges")
        return (a, b), (c, d)
    raise ValueError(f"{fname}: DataRange must be Full, Automatic or {{{{left, right}}, {{bottom, top}}}}")


def _padded(im, pad, fname):
    """(h + 2, w + 2, c) array with one pixel of padding (a value/colour, or "Fixed")."""
    a = im.a
    if pad == "Fixed":
        return np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
    v = _replacement_value(im, pad, None, fname) if not _is_sym(pad, "None") else None
    if v is None:
        raise NotImplementedError(f"{fname}: Padding -> None is not supported")
    out = np.empty((a.shape[0] + 2, a.shape[1] + 2, a.shape[2]))
    out[...] = v
    out[1:-1, 1:-1] = a
    return out


def image_value(im, xs, ys, pad=0, method="Linear", fname="ImageValue"):
    """Values (n, c) at image coordinates (pixel units, origin bottom-left) with padding outside the image."""
    P = _padded(im, pad, fname)
    out = []
    for x, y in zip(xs, ys):
        # continuous pixel index (0-based, centres at k + 0.5) in the padded array; rows from the top
        u = x - 0.5 + 1
        v = (im.h - y) - 0.5 + 1
        if method == "Nearest":
            cols = _nearest_axis(u, P.shape[1])
            rows = _nearest_axis(v, P.shape[0])
            out.append(np.mean([P[r, c] for r in rows for c in cols], axis=0))
            continue
        u = min(max(u, 0.0), P.shape[1] - 1.0)
        v = min(max(v, 0.0), P.shape[0] - 1.0)
        c0, r0 = int(math.floor(u)), int(math.floor(v))
        c1, r1 = min(c0 + 1, P.shape[1] - 1), min(r0 + 1, P.shape[0] - 1)
        fu, fv = u - c0, v - r0
        out.append((P[r0, c0] * (1 - fu) * (1 - fv) + P[r0, c1] * fu * (1 - fv) + P[r1, c0] * (1 - fu) * fv
                    + P[r1, c1] * fu * fv))
    return np.array(out)


def _nearest_axis(u, n):
    f = math.floor(u)
    if abs(u - f - 0.5) < 1e-12:
        idx = [f, f + 1]
    else:
        idx = [int(math.floor(u + 0.5))]
    return [min(max(i, 0), n - 1) for i in idx]


@fn("ImageValue", area=AREA, status="added",
    notes="ImageValue[image, {x, y}] gives the channel values at image coordinates {x, y} ({0, 0} the bottom-left "
          "corner, pixel centres at half-integers) by bilinear interpolation between pixel centres (Resampling -> "
          "\"Linear\", default) or the nearest pixel (\"Nearest\"; on a boundary the average of the adjacent pixels), "
          "with Padding (default 0, a colour or \"Fixed\") one pixel beyond the edges, so positions near or outside "
          "the border mix in the padding as the documented defaults imply; a list of positions gives a list; type "
          "argument as for PixelValue; DataRange Full (default), Automatic or explicit ranges. Other resampling "
          "methods and spans raise NotImplementedError. Check: exact (equals PixelValue at pixel centres, linear "
          "between them) in tests.")
def _imagevalue(x: Expr, pos: ListOf[Real] | ListOf[ListOf[Real]], *rest: String | Option) -> Real | ListOf[Real]:
    fname = "ImageValue"
    p_, o = opts_split(rest, {"DataRange", "Padding", "Resampling"}, fname)
    if len(p_) > 1:
        raise ValueError("ImageValue: expected ImageValue[image, pos, type, options]")
    t = type_arg(p_[0], fname) if p_ else None
    im = _img1(x, fname, allow3d=False)
    method = o.get("Resampling", "Linear")
    if method not in ("Linear", "Nearest", "Bilinear"):
        if isinstance(method, str):
            raise NotImplementedError(f"ImageValue: Resampling {method!r} is not supported")
        raise ValueError("ImageValue: Resampling must be a method name")
    method = "Linear" if method == "Bilinear" else method
    pad = o.get("Padding", 0)
    (x0, x1), (y0, y1) = _data_range(o, im, fname)
    many = K.is_list(pos) and pos.args and all(K.is_list(p) for p in pos.args)
    pts = list(pos.args) if many else [pos]
    xs, ys = [], []
    for p in pts:
        if not (K.is_list(p) and len(p.args) == 2):
            if isinstance(p, Symbol):
                raise Unevaluated()
            raise TypeError(f"ImageValue: {_short(p)} is not a position {{x, y}}")
        if any(isinstance(v, Expr) and v.head == "Span" for v in p.args):
            raise NotImplementedError("ImageValue: span positions are not supported")
        px, py = (real_arg(v, fname, "position") for v in p.args)
        xs.append((px - x0) / (x1 - x0) * im.w)
        ys.append((py - y0) / (y1 - y0) * im.h)
    vals = image_value(im, xs, ys, pad, method, fname)
    out = [_value_out(v, t) for v in vals]
    return L(*out) if many else out[0]


def _intensity(im, fname):
    col = im.color()
    if col.shape[-1] == 1:
        return col[..., 0]
    if im.cs == "Automatic":
        return col.mean(axis=-1)
    return convert_array(col, source_space(im, fname), "Grayscale")[..., 0]


def _value_positions(x, val, d, fname, shift):
    im = _img1(x, fname, allow3d=False)
    if val in ("Max", "Min"):
        it = _intensity(im, fname)
        target = it.max() if val == "Max" else it.min()
        mask = it == target
    else:
        v = value_for(Img(im.color(), im.type, im.cs if not im.alpha else im.cs, im.d3)
                      if not im.alpha else im, val, fname)
        if im.alpha and len(v) == im.c and parse_color(val) is None and K.is_list(val):
            pass
        col = im.a if (len(v) == im.c) else im.color()
        v = v[:col.shape[-1]]
        dist = np.sqrt(((col - v) ** 2).sum(axis=-1))
        tol = 0.0 if d is None else real_arg(d, fname, "distance")
        if tol < 0:
            raise ValueError(f"{fname}: the distance must be non-negative")
        mask = dist <= tol if d is not None else np.all(col == v, axis=-1)
    rows, cols = np.nonzero(mask)
    pts = sorted((int(c) + 1, im.h - int(r)) for r, c in zip(rows, cols))
    if shift:
        return L(*[L(px - 0.5, py - 0.5) for px, py in pts])
    return L(*[L(px, py) for px, py in pts])


_POS_NOTES = ("val is a number, a channel list or a colour (converted to the image's space), or \"Max\" / \"Min\" "
              "for the pixels of largest / smallest intensity (luma for colour images, the channel mean for "
              "ColorSpace Automatic); with d, pixels whose channel vector lies within Euclidean distance d of val "
              "(0..1 scale) match, otherwise values must be equal. Positions are sorted by x then y. 2D only.")


@fn("PixelValuePositions", area=AREA, status="added",
    notes="PixelValuePositions[image, val] and PixelValuePositions[image, val, d] give the pixel positions {x, y} "
          "({1, 1} the bottom-left pixel) of the matching pixels. " + _POS_NOTES +
          " Check: exact (PixelValue at every returned position matches) in tests.")
def _pixelvaluepositions(x: Expr, val: Real | ListOf[Real] | Expr | String, d: Real = None) -> ListOf[ListOf[Integer]]:
    return _value_positions(x, val, d, "PixelValuePositions", False)


@fn("ImageValuePositions", area=AREA, status="added",
    notes="ImageValuePositions[image, val] and ImageValuePositions[image, val, d] give the image coordinates of the "
          "centres of the matching pixels (PixelValuePositions - 1/2, as documented). " + _POS_NOTES +
          " Check: exact (the documented relation to PixelValuePositions) in tests.")
def _imagevaluepositions(x: Expr, val: Real | ListOf[Real] | Expr | String, d: Real = None) -> ListOf[ListOf[Real]]:
    return _value_positions(x, val, d, "ImageValuePositions", True)
