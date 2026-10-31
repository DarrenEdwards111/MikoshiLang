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
    def depth(self):
        return self.a.shape[0] if self.d3 else 1

    @property
    def alpha(self):
        return self.cs != "Automatic" and self.c == NCHANNELS[self.cs] + 1

    def color(self):
        return self.a[..., :-1] if self.alpha else self.a

    def alpha_array(self):
        return self.a[..., -1:] if self.alpha else np.ones(self.a.shape[:-1] + (1,))

    def dims(self):
        return [self.w, self.depth, self.h] if self.d3 else [self.w, self.h]


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
