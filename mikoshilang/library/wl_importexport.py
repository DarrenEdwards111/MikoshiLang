"""Wolfram batch W13: Import/Export formats, serialisation, compression and archives (Wolfram semantics).

* Serialisation: ``BinarySerialize``/``BinaryDeserialize`` write and read WXF exactly as the Wolfram Language
  documents it (tutorial/WXFFormatDescription): header ``8:`` (``8C:`` when zlib-compressed), varint lengths,
  machine integers in the smallest of C/j/i/L, machine reals as little-endian IEEE doubles, big integers and
  arbitrary-precision reals as InputForm text, ``A``/``-``/``:`` for associations, ``B`` for ByteArray, packed and
  numeric arrays when reading.  Symbols are written with their full context except System` symbols.  Lists are
  written as ``f``/``List`` (MikoshiLang has no packed arrays; Wolfram reads both forms identically).
* ``Compress``/``Uncompress`` use Wolfram's string format: ``"1:"`` followed by the Base64 encoding of the zlib
  deflation of the ``!boR`` expression serialisation (f/s/S/i/I/r/R parts, 32-bit little-endian counts).
* ``Import``/``Export``/``ImportString``/``ExportString``/``ImportByteArray``/``ExportByteArray`` support the
  formats listed in ``$ImportFormats``/``$ExportFormats``: Text, String, Lines, List, Table, CSV, TSV, JSON,
  RawJSON, ExpressionJSON, XML (symbolic XML with XMLObject/XMLElement), Package/WL/M (InputForm text), WXF, MX
  (MikoshiLang's own binary dump, like Wolfram's version-specific MX), Base64, GZIP, BZIP2, ZIP, TAR, PEM and the
  raw binary formats Byte, Character8, Integer8..Integer64, UnsignedInteger8..64, Real32, Real64.  CSV/TSV follow
  the classic ("Backend" -> "Table") semantics: ragged rows are kept, fields that look like integers or reals
  (C or Fortran exponent notation) import as numbers, and Export quotes a field only when it needs quoting.
* ``Put``/``PutAppend`` (``>>``/``>>>``) write InputForm text; ``Save`` appends the full definitions of symbols;
  ``DumpSave`` writes a binary dump that ``Get`` reads back.
* ``CreateArchive``/``ExtractArchive`` (zip, tar, tar.gz, tar.bz2), ``FileFormat``, ``FileConvert``, ``CreateUUID``,
  file format registries and the local cache/persistence names.

Cryptography lives in :mod:`wl_crypto` (same batch).
"""
from __future__ import annotations

import base64
import bz2
import csv
import gzip
import io
import json
import math
import os
import re
import struct
import tarfile
import uuid
import zipfile
import zlib
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
import importlib
E = importlib.import_module("mikoshilang.evaluate")
from .. import formatting as F
from .core import Unevaluated, fn, record, L, is_list
from .wl_common import ev, is_sym

AREA = "import/export & serialisation (W13)"
NULL = None


# ═══════════════════════════════════════ small helpers ═══════════════════════════════════════

def _sym(n):
    return Symbol(n)


def _assoc(pairs):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in pairs))


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def _opt_key(x):
    if _is_rule(x):
        k = x.args[0]
        if isinstance(k, Symbol):
            return k.name
        if isinstance(k, str):
            return k
    return None


def split_opts(args, allowed, fname):
    """Trailing options (Name -> v or "Name" -> v) -> dict; unknown names raise a clear error."""
    args = list(args)
    opts = {}
    while args and _opt_key(args[-1]) is not None and (isinstance(args[-1].args[0], Symbol)
                                                       or args[-1].args[0] in allowed):
        r = args.pop()
        k = _opt_key(r)
        if k not in allowed:
            raise ValueError(f"{fname}: unknown option {k}")
        opts.setdefault(k, r.args[1])
    return args, opts


def assoc_get(a, key, default=None):
    if not KL.is_assoc(a):
        return default
    for r in a.args:
        if _is_rule(r) and K.same(r.args[0], key):
            return r.args[1]
    return default


def assoc_dict(a, fname):
    if not KL.is_assoc(a):
        raise TypeError(f"{fname}: {F.input_form(a)} is not an association")
    out = {}
    for r in a.args:
        k = r.args[0]
        if not isinstance(k, str):
            raise TypeError(f"{fname}: association keys must be strings")
        out[k] = r.args[1]
    return out


def file_arg(x, fname):
    """A file name: "name", File["name"] (the user's home directory is expanded)."""
    if isinstance(x, Expr) and x.head == "File" and len(x.args) == 1 and isinstance(x.args[0], str):
        x = x.args[0]
    if not isinstance(x, str):
        raise TypeError(f"{fname}: {F.input_form(x)} is not a valid file specification")
    return os.path.expanduser(x)


# ── ByteArray (the representation of batch W11: ByteArray["base64"]) ──

def byte_array(data):
    return Expr("ByteArray", base64.b64encode(bytes(data)).decode("ascii"))


def is_byte_array(x):
    if not (isinstance(x, Expr) and x.head == "ByteArray" and len(x.args) == 1):
        return False
    a = x.args[0]
    return isinstance(a, str) or (is_list(a) and all(K.is_int(b) and 0 <= b <= 255 for b in a.args))


def ba_bytes(x, fname="ByteArray"):
    if not is_byte_array(x):
        raise TypeError(f"{fname}: {F.input_form(x)} is not a ByteArray")
    a = x.args[0]
    if isinstance(a, str):
        try:
            return base64.b64decode(a, validate=True)
        except Exception:
            raise ValueError(f"{fname}: invalid Base64 data in ByteArray")
    return bytes(a.args)


def bytes_arg(x, fname, what="data"):
    """bytes from a ByteArray, a list of byte values or {}."""
    if is_byte_array(x):
        return ba_bytes(x, fname)
    if is_list(x) and all(K.is_int(b) and not isinstance(b, bool) and 0 <= b <= 255 for b in x.args):
        return bytes(x.args)
    raise TypeError(f"{fname}: {F.input_form(x)} is not a ByteArray or a list of bytes ({what})")


# ═══════════════════════════════════════ canonical atoms ═══════════════════════════════════════

def _is_sympy_complex(x):
    return isinstance(x, sp.Basic) and not isinstance(x, sp.Number) and x.is_number and not x.free_symbols \
        and x.atoms(sp.NumberSymbol) == set() and all(t.is_Number for t in x.as_real_imag())


def atom_kind(x):
    """Classify an evaluated expression into Wolfram's atomic kinds; ('expr', head, args) otherwise."""
    if x is None:
        return ("sym", "Null")
    if isinstance(x, bool):
        return ("sym", "True" if x else "False")
    if isinstance(x, int):
        return ("int", x)
    if isinstance(x, float):
        return ("real", x)
    if isinstance(x, str):
        return ("str", x)
    if isinstance(x, Symbol):
        if x.name == "I":
            return ("complex", 0, 1)
        return ("sym", x.name)
    if isinstance(x, complex):
        return ("complex", x.real, x.imag)
    if isinstance(x, sp.Integer):
        return ("int", int(x))
    if isinstance(x, sp.Rational):
        return ("rational", int(x.p), int(x.q))
    if isinstance(x, sp.Float):
        return ("bigreal", x) if x._prec > 53 else ("real", float(x))
    if isinstance(x, sp.Basic):
        if _is_sympy_complex(x):
            re_, im_ = x.as_real_imag()
            return ("complex", _sp_num(re_), _sp_num(im_))
        return atom_kind(K.from_sp(x))
    if isinstance(x, Expr):
        q = F.rational_value(x)
        if q is not None:
            if q.denominator == 1:
                return ("int", q.numerator)
            return ("rational", q.numerator, q.denominator)
        if x.head in ("Plus", "Times") and K.is_number(x):
            v = K.to_sp(x)
            re_, im_ = v.as_real_imag()
            return ("complex", _sp_num(re_), _sp_num(im_))
        return ("expr", x.head, x.args)
    raise ValueError(f"cannot serialise the Python object {x!r}")


def _sp_num(v):
    """A SymPy real number as a MikoshiLang number (machine floats stay floats)."""
    if isinstance(v, sp.Integer):
        return int(v)
    if isinstance(v, sp.Rational):
        return K.from_sp(v)
    if isinstance(v, sp.Float):
        return float(v) if v._prec <= 53 else v
    return K.from_sp(v)


def make_complex(re_, im_):
    return ev(Expr("Plus", re_, Expr("Times", im_, Symbol("I"))))


def make_rational(p, q):
    if q == 0:
        raise ValueError("Rational with a zero denominator")
    return K.from_sp(sp.Rational(p, q))


# ── symbol names ──

def wl_symbol_name(name):
    """Fully qualified name, except for System` symbols (WXF convention)."""
    if "`" in name:
        return name
    from .wl_evaluation import is_system
    if is_system(name) or name in ("List", "Rule", "RuleDelayed", "Association", "Complex", "Rational"):
        return name
    return "Global`" + name


def internal_symbol(full):
    if full.startswith("System`"):
        return full[7:]
    if full.startswith("Global`"):
        return full[7:]
    return full


# ── InputForm text with arbitrary-precision numbers ──

def bigreal_text(f):
    """InputForm of an arbitrary-precision real: digits`prec*^exp (Wolfram InputForm); the precision mark reads
    back to the same binary precision."""
    from mpmath import mp, mpf
    prec_bits = f._prec
    dps = max(1, int(round(prec_bits / 3.321928094887362 - 1)))
    with mp.workprec(prec_bits):
        s = mp.nstr(mpf(f._mpf_), dps, min_fixed=-5, max_fixed=dps + 1, strip_zeros=False)
    exp = ""
    if "e" in s:
        s, e = s.split("e")
        exp = "*^" + str(int(e))
    if "." not in s:
        s += "."
    return f"{s}`{dps}.{exp}"


def _has_bignum(x):
    if isinstance(x, sp.Float):
        return x._prec > 53
    if isinstance(x, sp.Basic):
        return any(isinstance(a, sp.Float) and a._prec > 53 for a in sp.preorder_traversal(x))
    if isinstance(x, Expr):
        return _has_bignum(x.head) or any(_has_bignum(a) for a in x.args)
    return False


def _real_text(v):
    if v != v:
        return "Indeterminate"
    if v in (float("inf"), float("-inf")):
        return "DirectedInfinity[1]" if v > 0 else "DirectedInfinity[-1]"
    r = repr(float(v))
    if "e" in r:
        m, e = r.split("e")
        if "." not in m:
            m += "."
        return f"{m}*^{int(e)}"
    if r.endswith(".0"):
        r = r[:-1]
    return r


def full_text(x):
    """FullForm-style text that reads back to x exactly (exact numbers, machine and arbitrary-precision reals)."""
    k = atom_kind(x)
    t = k[0]
    if t == "int":
        return str(k[1])
    if t == "real":
        return _real_text(k[1])
    if t == "bigreal":
        return bigreal_text(k[1])
    if t == "rational":
        return f"Rational[{k[1]}, {k[2]}]"
    if t == "complex":
        return f"Complex[{full_text(k[1])}, {full_text(k[2])}]"
    if t == "str":
        return string_literal(k[1])
    if t == "sym":
        return k[1]
    head, args = k[1], k[2]
    h = head if isinstance(head, str) else full_text(head)
    return h + "[" + ", ".join(full_text(a) for a in args) + "]"


def string_literal(s):
    out = ['"']
    for ch in s:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 32:
            out.append("\\:%04x" % ord(ch))
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def input_text(x):
    """InputForm text of x as written by Put/Save; falls back to exact FullForm text where InputForm would lose
    information (arbitrary-precision numbers)."""
    if x is None:
        return "Null"
    if _has_bignum(x):
        return full_text(x)
    if isinstance(x, str):
        return string_literal(x)
    return F.input_form(x)


def parse_text(text, fname):
    from ..parser import parse
    try:
        return parse(text)
    except Exception as e:
        raise ValueError(f"{fname}: the text is not valid Wolfram Language input ({e})")


# ═══════════════════════════════════════ WXF (BinarySerialize) ═══════════════════════════════════════

def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _wxf_text(token, s):
    b = s.encode("utf-8")
    return token + _varint(len(b)) + b


def _wxf_int(n):
    if -128 <= n <= 127:
        return b"C" + struct.pack("<b", n)
    if -32768 <= n <= 32767:
        return b"j" + struct.pack("<h", n)
    if -2 ** 31 <= n < 2 ** 31:
        return b"i" + struct.pack("<i", n)
    if -2 ** 63 <= n < 2 ** 63:
        return b"L" + struct.pack("<q", n)
    return _wxf_text(b"I", str(n))


def _wxf_symbol(name):
    return _wxf_text(b"s", wl_symbol_name(name))


def wxf_encode(x, out):
    if is_byte_array(x):
        data = ba_bytes(x, "BinarySerialize")
        out.append(b"B" + _varint(len(data)) + data)
        return
    k = atom_kind(x)
    t = k[0]
    if t == "int":
        out.append(_wxf_int(k[1]))
    elif t == "real":
        v = k[1]
        if v != v or v in (float("inf"), float("-inf")):
            raise ValueError("BinarySerialize: machine overflow/underflow values cannot be serialised")
        out.append(b"r" + struct.pack("<d", v))
    elif t == "bigreal":
        out.append(_wxf_text(b"R", bigreal_text(k[1])))
    elif t == "str":
        out.append(_wxf_text(b"S", k[1]))
    elif t == "sym":
        out.append(_wxf_symbol(k[1]))
    elif t == "rational":
        out.append(b"f" + _varint(2) + _wxf_symbol("Rational"))
        out.append(_wxf_int(k[1]))
        out.append(_wxf_int(k[2]))
    elif t == "complex":
        out.append(b"f" + _varint(2) + _wxf_symbol("Complex"))
        wxf_encode(k[1], out)
        wxf_encode(k[2], out)
    else:
        head, args = k[1], k[2]
        if head == "Association" and all(_is_rule(r) for r in args):
            out.append(b"A" + _varint(len(args)))
            for r in args:
                out.append(b"-" if r.head == "Rule" else b":")
                wxf_encode(r.args[0], out)
                wxf_encode(r.args[1], out)
            return
        out.append(b"f" + _varint(len(args)))
        if isinstance(head, str):
            out.append(_wxf_symbol(head))
        else:
            wxf_encode(head, out)
        for a in args:
            wxf_encode(a, out)


def binary_serialize(x, compress=False):
    out = []
    wxf_encode(x, out)
    body = b"".join(out)
    if compress:
        return b"8C:" + zlib.compress(body)
    return b"8:" + body


class _Reader:
    def __init__(self, data, fname):
        self.d, self.i, self.fname = data, 0, fname

    def take(self, n):
        if self.i + n > len(self.d):
            raise ValueError(f"{self.fname}: the data ends unexpectedly (corrupt serialisation)")
        b = self.d[self.i:self.i + n]
        self.i += n
        return b

    def byte(self):
        return self.take(1)[0]

    def varint(self):
        n, shift = 0, 0
        while True:
            b = self.byte()
            n |= (b & 0x7F) << shift
            if not b & 0x80:
                return n
            shift += 7
            if shift > 63:
                raise ValueError(f"{self.fname}: invalid varint")


_PACKED_FMT = {0: ("b", 1), 1: ("h", 2), 2: ("i", 4), 3: ("q", 8), 16: ("B", 1), 17: ("H", 2), 18: ("I", 4),
               19: ("Q", 8), 34: ("f", 4), 35: ("d", 8), 51: ("cf", 8), 52: ("cd", 16)}
_NUMERIC_TYPE = {0: "Integer8", 1: "Integer16", 2: "Integer32", 3: "Integer64", 16: "UnsignedInteger8",
                 17: "UnsignedInteger16", 18: "UnsignedInteger32", 19: "UnsignedInteger64", 34: "Real32",
                 35: "Real64", 51: "ComplexReal32", 52: "ComplexReal64"}


def _number_from_text(s, fname):
    v = ev(parse_text(s, fname))
    if not K.is_number(v):
        raise ValueError(f"{fname}: {s!r} is not a number")
    return v


def _wxf_decode(r):
    t = r.byte()
    c = chr(t)
    if c == "f":
        n = r.varint()
        head = _wxf_decode(r)
        args = [_wxf_decode(r) for _ in range(n)]
        if isinstance(head, Symbol):
            if head.name == "Rational" and n == 2 and all(K.is_int(a) for a in args):
                return make_rational(args[0], args[1])
            if head.name == "Complex" and n == 2 and all(K.is_number(a) for a in args):
                return make_complex(args[0], args[1])
            return Expr(head.name, *args)
        if isinstance(head, (Expr,)):
            return Expr(head, *args)
        return Expr(F.input_form(head) if not isinstance(head, str) else head, *args)
    if c == "C":
        return struct.unpack("<b", r.take(1))[0]
    if c == "j":
        return struct.unpack("<h", r.take(2))[0]
    if c == "i":
        return struct.unpack("<i", r.take(4))[0]
    if c == "L":
        return struct.unpack("<q", r.take(8))[0]
    if c == "r":
        return struct.unpack("<d", r.take(8))[0]
    if c in "SsIRB":
        n = r.varint()
        raw = r.take(n)
        if c == "B":
            return byte_array(raw)
        s = raw.decode("utf-8")
        if c == "S":
            return s
        if c == "s":
            n2 = internal_symbol(s)
            if n2 == "True":
                return True
            if n2 == "False":
                return False
            if n2 == "Null":
                return None
            return Symbol(n2)
        if c == "I":
            if not re.fullmatch(r"-?\d+", s):
                raise ValueError(f"{r.fname}: invalid big integer {s!r}")
            return int(s)
        return _number_from_text(s, r.fname)
    if c in "A":
        n = r.varint()
        rules = []
        for _ in range(n):
            k = r.byte()
            if k not in (45, 58):
                raise ValueError(f"{r.fname}: invalid association rule token")
            key = _wxf_decode(r)
            val = _wxf_decode(r)
            rules.append(Expr("Rule" if k == 45 else "RuleDelayed", key, val))
        return Expr("Association", *rules)
    if t in (0xC1, 0xC2):
        typ = r.byte()
        if typ not in _PACKED_FMT:
            raise ValueError(f"{r.fname}: unknown array element type {typ}")
        rank = r.varint()
        dims = [r.varint() for _ in range(rank)]
        code, size = _PACKED_FMT[typ]
        count = 1
        for d in dims:
            count *= d
        raw = r.take(count * size)
        if code in ("cf", "cd"):
            fl = struct.unpack("<" + ("f" if code == "cf" else "d") * (2 * count), raw)
            flat = [make_complex(fl[2 * i], fl[2 * i + 1]) for i in range(count)]
        else:
            flat = list(struct.unpack("<" + code * count, raw))
            if code in ("f", "d"):
                flat = [float(v) for v in flat]

        def nest(items, ds):
            if len(ds) <= 1:
                return L(*items)
            step = len(items) // ds[0] if ds[0] else 0
            return L(*(nest(items[i * step:(i + 1) * step], ds[1:]) for i in range(ds[0])))
        arr = nest(flat, dims) if dims else L()
        if t == 0xC2:
            return Expr("NumericArray", arr, _NUMERIC_TYPE[typ])
        return arr
    raise ValueError(f"{r.fname}: unknown WXF token {t}")


def binary_deserialize(data, fname="BinaryDeserialize"):
    if data[:2] == b"8:":
        body = data[2:]
    elif data[:3] == b"8C:":
        try:
            body = zlib.decompress(data[3:])
        except zlib.error:
            raise ValueError(f"{fname}: corrupt compressed WXF data")
    else:
        raise ValueError(f"{fname}: the data is not in the WXF format (it must start with \"8:\")")
    r = _Reader(body, fname)
    x = _wxf_decode(r)
    if r.i != len(body):
        raise ValueError(f"{fname}: trailing data after the serialised expression")
    return x


# ═══════════════════════════════════════ Compress (!boR) ═══════════════════════════════════════

def _bor_string(s):
    out = []
    for ch in s:
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif o < 128:
            out.append(ch)
        elif o < 256:
            out.append("\\%03o" % o)
        elif o < 0x10000:
            out.append("\\:%04x" % o)
        else:
            v = o - 0x10000
            out.append("\\:%04x\\:%04x" % (0xD800 + (v >> 10), 0xDC00 + (v & 0x3FF)))
    return "".join(out).encode("ascii")


def _bor_unstring(b):
    s = b.decode("latin-1")
    out, i, n = [], 0, len(s)
    units = []

    def flush():
        if units:
            out.append(b"".join(struct.pack("<H", u) for u in units).decode("utf-16-le", errors="strict"))
            units.clear()
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nx = s[i + 1]
            if nx == ":" and i + 6 <= n:
                units.append(int(s[i + 2:i + 6], 16))
                i += 6
                continue
            flush()
            if nx in "01234567" and i + 4 <= n:
                out.append(chr(int(s[i + 1:i + 4], 8)))
                i += 4
                continue
            if nx == "\\":
                out.append("\\")
                i += 2
                continue
            out.append(ch)
            i += 1
            continue
        flush()
        out.append(ch)
        i += 1
    flush()
    return "".join(out)


def _bor_text(token, s):
    b = _bor_string(s) if token in (b"S", b"s") else s.encode("ascii")
    return token + struct.pack("<i", len(b)) + b


def bor_encode(x, out):
    if is_byte_array(x):
        out.append(b"f" + struct.pack("<i", 1) + _bor_text(b"s", "ByteArray"))
        out.append(_bor_text(b"S", base64.b64encode(ba_bytes(x, "Compress")).decode("ascii")))
        return
    k = atom_kind(x)
    t = k[0]
    if t == "int":
        n = k[1]
        if -2 ** 31 <= n < 2 ** 31:
            out.append(b"i" + struct.pack("<i", n))
        else:
            out.append(_bor_text(b"I", str(n)))
    elif t == "real":
        out.append(b"r" + struct.pack("<d", k[1]))
    elif t == "bigreal":
        out.append(_bor_text(b"R", bigreal_text(k[1])))
    elif t == "str":
        out.append(_bor_text(b"S", k[1]))
    elif t == "sym":
        out.append(_bor_text(b"s", wl_symbol_name(k[1])))
    elif t in ("rational", "complex"):
        out.append(b"f" + struct.pack("<i", 2) + _bor_text(b"s", "Rational" if t == "rational" else "Complex"))
        if t == "rational":
            bor_encode(k[1], out)
            bor_encode(k[2], out)
        else:
            bor_encode(k[1], out)
            bor_encode(k[2], out)
    else:
        head, args = k[1], k[2]
        out.append(b"f" + struct.pack("<i", len(args)))
        if isinstance(head, str):
            out.append(_bor_text(b"s", wl_symbol_name(head)))
        else:
            bor_encode(head, out)
        for a in args:
            bor_encode(a, out)


def _bor_decode(r):
    c = chr(r.byte())
    if c == "f":
        n = struct.unpack("<i", r.take(4))[0]
        head = _bor_decode(r)
        args = [_bor_decode(r) for _ in range(n)]
        if isinstance(head, Symbol):
            if head.name == "Rational" and n == 2 and all(K.is_int(a) for a in args):
                return make_rational(args[0], args[1])
            if head.name == "Complex" and n == 2 and all(K.is_number(a) for a in args):
                return make_complex(args[0], args[1])
            return Expr(head.name, *args)
        if isinstance(head, Expr):
            return Expr(head, *args)
        raise ValueError(f"{r.fname}: invalid expression head")
    if c == "i":
        return struct.unpack("<i", r.take(4))[0]
    if c == "r":
        return struct.unpack("<d", r.take(8))[0]
    if c in "SsIR":
        n = struct.unpack("<i", r.take(4))[0]
        raw = r.take(n)
        if c == "S":
            return _bor_unstring(raw)
        if c == "s":
            nm = internal_symbol(_bor_unstring(raw))
            return {"True": True, "False": False, "Null": None}.get(nm, Symbol(nm)) \
                if nm in ("True", "False", "Null") else Symbol(nm)
        s = raw.decode("ascii")
        if c == "I":
            if not re.fullmatch(r"-?\d+", s):
                raise ValueError(f"{r.fname}: invalid big integer {s!r}")
            return int(s)
        return _number_from_text(s, r.fname)
    if c == "e":
        rank = struct.unpack("<i", r.take(4))[0]
        dims = [struct.unpack("<i", r.take(4))[0] for _ in range(rank)]
        count = 1
        for d in dims:
            count *= d
        flat = [float(v) for v in struct.unpack("<" + "d" * count, r.take(8 * count))]

        def nest(items, ds):
            if len(ds) <= 1:
                return L(*items)
            step = len(items) // ds[0] if ds[0] else 0
            return L(*(nest(items[i * step:(i + 1) * step], ds[1:]) for i in range(ds[0])))
        return nest(flat, dims)
    raise ValueError(f"{r.fname}: unknown token {c!r} in compressed data")


def compress_string(x):
    out = [b"!boR"]
    bor_encode(x, out)
    return "1:" + base64.b64encode(zlib.compress(b"".join(out))).decode("ascii")


def uncompress_string(s, fname="Uncompress"):
    if not isinstance(s, str) or not s.startswith("1:"):
        raise ValueError(f"{fname}: the string is not compressed data (it must start with \"1:\")")
    try:
        raw = zlib.decompress(base64.b64decode(re.sub(r"\s+", "", s[2:]), validate=True))
    except Exception:
        raise ValueError(f"{fname}: corrupt compressed data")
    if raw[:4] != b"!boR":
        raise ValueError(f"{fname}: unknown compressed data version")
    r = _Reader(raw[4:], fname)
    x = _bor_decode(r)
    if r.i != len(raw) - 4:
        raise ValueError(f"{fname}: trailing data after the compressed expression")
    return x


@fn("BinarySerialize", area=AREA, status="added",
    notes="BinarySerialize[expr] gives a ByteArray with the WXF serialisation of expr (header 8:, documented "
          "tokens and varint lengths; System` symbols unqualified, others with their context); "
          "PerformanceGoal -> \"Size\" zlib-compresses (header 8C:); \"Speed\"/Automatic do not. Check: reference (documented WXF bytes) and the round trip BinaryDeserialize[BinarySerialize[e]] === e in tests.")
def _binaryserialize(x: Expr, *opts: Option) -> Expr:
    rest, o = split_opts(opts, ("PerformanceGoal", "Method"), "BinarySerialize")
    if rest:
        raise ValueError("BinarySerialize: called with too many arguments")
    pg = o.get("PerformanceGoal", _sym("Automatic"))
    if is_sym(pg, "Automatic") or pg == "Speed":
        comp = False
    elif pg == "Size":
        comp = True
    else:
        raise ValueError("BinarySerialize: PerformanceGoal must be \"Speed\", \"Size\" or Automatic")
    if "Method" in o:
        raise NotImplementedError("BinarySerialize: the Method option is not supported")
    return byte_array(binary_serialize(x, comp))


@fn("BinaryDeserialize", area=AREA, status="added",
    notes="BinaryDeserialize[ba] reads WXF data (compressed or not, including packed and numeric arrays) and "
          "evaluates the result; BinaryDeserialize[ba, h] wraps it in h before evaluation; data that is not WXF "
          "raises an error. Check: reference (documented WXF bytes) and serialise/deserialise round trips in tests.")
def _binarydeserialize(b: Expr, h: Fn = None) -> Expr:
    data = bytes_arg(b, "BinaryDeserialize")
    x = binary_deserialize(data)
    if h is not None:
        return Expr(h if not isinstance(h, Symbol) else h.name, x)
    return x


@fn("Compress", area=AREA, status="added",
    notes="Compress[expr] gives Wolfram's compressed string: \"1:\" <> Base64 of the zlib deflation of the "
          "!boR serialisation of expr (i/I integers, r/R reals, S strings with \\ooo and \\:xxxx escapes, "
          "s symbols, f expressions); Uncompress inverts it. Check: exact; the round trip Uncompress[Compress[e]] === e in tests.")
def _compress(x: Expr, *opts: Option) -> String:
    if opts:
        raise NotImplementedError("Compress: options (Method) are not supported")
    return compress_string(x)


@fn("Uncompress", area=AREA, status="added",
    notes="Uncompress[\"1:...\"] recovers and evaluates the expression from a Compress string (also reads "
          "packed real arrays 'e'); Uncompress[c, h] wraps it in h before evaluation; invalid data raises. Check: exact; Compress round trips in tests.")
def _uncompress(s: String, h: Fn = None) -> Expr:
    x = uncompress_string(s)
    if isinstance(x, Symbol):
        x = ev(x)
    if h is not None:
        return Expr(h if not isinstance(h, Symbol) else h.name, x)
    return x


# ═══════════════════════════════════════ text helpers ═══════════════════════════════════════

_ENCODINGS = {"UTF8": "utf-8", "UTF-8": "utf-8", "ISOLatin1": "latin-1", "ISO8859-1": "latin-1", "ASCII": "ascii",
              "PrintableASCII": "ascii", "UTF-16": "utf-16", "UTF-16LE": "utf-16-le", "UTF-16BE": "utf-16-be",
              "Unicode": "utf-16", "UTF-32": "utf-32", "WindowsANSI": "cp1252", "CP1252": "cp1252",
              "ShiftJIS": "shift_jis", "EUC-JP": "euc_jp", "ISOLatin2": "iso8859-2", "ISOLatin9": "iso8859-15",
              "MacintoshRoman": "mac_roman", "KOI8-R": "koi8_r"}


def _codec(enc, fname):
    if enc is None or is_sym(enc, "Automatic") or enc == "UTF8ISOLatin1":
        return None
    if not isinstance(enc, str) or enc not in _ENCODINGS:
        raise ValueError(f"{fname}: {F.input_form(enc)} is not a supported character encoding")
    return _ENCODINGS[enc]


def decode_text(data, opts, fname):
    c = _codec(opts.get("CharacterEncoding"), fname)
    if c is None:
        try:
            s = data.decode("utf-8")
        except UnicodeDecodeError:
            s = data.decode("latin-1")
    else:
        try:
            s = data.decode(c)
        except UnicodeDecodeError:
            raise ValueError(f"{fname}: the data is not valid {opts.get('CharacterEncoding')} text")
    if s.startswith("\ufeff"):
        s = s[1:]
    return s.replace("\r\n", "\n").replace("\r", "\n")


def encode_text(s, opts, fname):
    c = _codec(opts.get("CharacterEncoding"), fname) or "utf-8"
    try:
        return s.encode(c)
    except UnicodeEncodeError:
        raise ValueError(f"{fname}: the text contains characters that cannot be encoded in "
                         f"{opts.get('CharacterEncoding')}")


_INT_RE = re.compile(r"[+-]?\d+")
_REAL_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][+-]?\d+)?")
_WLREAL_RE = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)\*\^[+-]?\d+")


def interpret_field(s, numeric=True):
    """A text field as Wolfram's table importers read it: integers, reals (C, Fortran or *^ exponents), else the
    string itself."""
    if not numeric:
        return s
    t = s.strip()
    if not t:
        return s
    if _INT_RE.fullmatch(t):
        return int(t)
    if _REAL_RE.fullmatch(t):
        return float(t.replace("d", "e").replace("D", "e"))
    if _WLREAL_RE.fullmatch(t):
        m, e = t.split("*^")
        return float(m) * 10.0 ** int(e) if abs(int(e)) < 300 else float(m + "e" + e)
    return s


def field_text(x):
    """Text of one exported table field: strings as themselves, everything else in InputForm."""
    if isinstance(x, str):
        return x
    if isinstance(x, float):
        return _real_text(x)
    return input_text(x)


def _lines(s):
    if s.endswith("\n"):
        s = s[:-1]
    return s.split("\n") if s != "" else []


# ═══════════════════════════════════════ element specs ═══════════════════════════════════════

def _part_index(n, i, fname):
    if not K.is_int(i) or i == 0 or abs(i) > n:
        raise ValueError(f"{fname}: part {F.input_form(i)} does not exist")
    return i - 1 if i > 0 else n + i


def select_parts(items, spec, fname):
    """Apply a row/column specification (n, -n, All, {n1,...}, n;;m;;s) to a list of items."""
    n = len(items)
    if is_sym(spec, "All"):
        return list(items), True
    if K.is_int(spec):
        return items[_part_index(n, spec, fname)], False
    if is_list(spec):
        return [items[_part_index(n, i, fname)] for i in spec.args], True
    if isinstance(spec, Expr) and spec.head == "Span":
        a = list(spec.args) + [1] * (3 - len(spec.args)) if len(spec.args) < 3 else list(spec.args)
        lo, hi, st = a[0], a[1], a[2]
        lo = 1 if is_sym(lo, "All") else lo
        hi = n if is_sym(hi, "All") else hi
        if not (K.is_int(lo) and K.is_int(hi) and K.is_int(st) and st != 0):
            raise ValueError(f"{fname}: invalid span")
        lo = lo if lo > 0 else n + lo + 1
        hi = hi if hi > 0 else n + hi + 1
        idx = list(range(lo, hi + 1, st)) if st > 0 else list(range(lo, hi - 1, st))
        return [items[_part_index(n, i, fname)] for i in idx], True
    raise ValueError(f"{fname}: {F.input_form(spec)} is not a valid part specification")


def apply_subelements(data, subs, fname):
    """{elem, rows, cols, ...}: successive part specifications applied level by level."""
    if not subs:
        return data
    if not is_list(data):
        raise ValueError(f"{fname}: the element has no parts")
    sel, many = select_parts(list(data.args), subs[0], fname)
    if many:
        return L(*(apply_subelements(x, subs[1:], fname) for x in sel))
    return apply_subelements(sel, subs[1:], fname)


# ═══════════════════════════════════════ formats ═══════════════════════════════════════

class Format:
    def __init__(self, name, exts=(), mimes=(), elements=("Data",), default="Data", imp=None, exp=None,
                 binary=False, opts=()):
        self.name, self.exts, self.mimes = name, list(exts), list(mimes)
        self.elements, self.default = list(elements), default
        self.imp, self.exp, self.binary, self.opts = imp, exp, binary, set(opts)


FORMATS = {}
ALIASES = {"WL": "Package", "M": "Package", "Wolfram": "Package", "TXT": "Text", "JSN": "JSON", "Gzip": "GZIP",
           "GZ": "GZIP", "BZ2": "BZIP2", "CSV": "CSV", "TSV": "TSV", "TAB": "TSV", "XHTML": None}


def register_format(f):
    FORMATS[f.name] = f
    return f


def format_of(name, fname):
    if not isinstance(name, str):
        raise TypeError(f"{fname}: {F.input_form(name)} is not a format name")
    if name in FORMATS:
        return FORMATS[name]
    for k in FORMATS:
        if k.lower() == name.lower():
            return FORMATS[k]
    a = ALIASES.get(name) or ALIASES.get(name.upper())
    if a:
        return FORMATS[a]
    raise ValueError(f"{fname}::fmtnosup: {name} is not a supported format (see $ImportFormats and "
                     f"$ExportFormats)")


TEXT_OPTS = ("CharacterEncoding",)


# ── Text, String, Lines, List, Table ──

def _imp_text(data, elem, subs, opts):
    s = decode_text(data, opts, "Import")
    if elem in ("Plaintext", "Text"):
        return s[:-1] if s.endswith("\n") else s
    if elem == "String":
        return s
    if elem == "Lines":
        return apply_subelements(L(*_lines(s)), subs, "Import")
    if elem == "Words":
        return apply_subelements(L(*s.split()), subs, "Import")
    if elem == "Data":
        return apply_subelements(L(*_lines(s)), subs, "Import")
    raise ValueError(f"Import: the element {elem!r} is not available for this format")


def _exp_text(x, opts):
    if isinstance(x, str):
        s = x
    elif is_list(x) and all(isinstance(t, str) for t in x.args):
        s = "\n".join(x.args)
    else:
        s = ev(Expr("ToString", x))
        if not isinstance(s, str):
            s = input_text(x)
    return encode_text(s, opts, "Export")


def _exp_string(x, opts):
    if not isinstance(x, str):
        raise ValueError("Export: the \"String\" format requires a string")
    return encode_text(x, opts, "Export")


def _exp_lines(x, opts):
    if not is_list(x):
        raise ValueError("Export: the \"Lines\" format requires a list of lines")
    return encode_text("\n".join(field_text(t) for t in x.args), opts, "Export")


def _imp_list(data, elem, subs, opts):
    if elem not in ("Data",):
        raise ValueError(f"Import: the element {elem!r} is not available for the List format")
    s = decode_text(data, opts, "Import")
    items = [interpret_field(t) for t in _lines(s) if t.strip() != ""]
    return apply_subelements(L(*items), subs, "Import")


def _exp_list(x, opts):
    if not is_list(x):
        x = L(x)
    return encode_text("\n".join(field_text(t) for t in x.args), opts, "Export")


def _imp_table(data, elem, subs, opts):
    if elem not in ("Data", "Grid"):
        raise ValueError(f"Import: the element {elem!r} is not available for the Table format")
    s = decode_text(data, opts, "Import")
    sep = opts.get("FieldSeparators")
    rows = []
    for line in _lines(s):
        if not line.strip():
            continue
        if isinstance(sep, str):
            parts = [p for p in line.split(sep)]
        else:
            parts = line.split()
        rows.append(L(*(interpret_field(p) for p in parts)))
    d = apply_subelements(L(*rows), subs, "Import")
    return Expr("Grid", d) if elem == "Grid" else d


def _rows_of(x, fname):
    if is_list(x) and x.args and all(is_list(r) for r in x.args):
        return [list(r.args) for r in x.args]
    if is_list(x):
        return [[v] for v in x.args]
    if isinstance(x, Expr) and x.head in ("Grid", "TableForm", "MatrixForm") and x.args:
        return _rows_of(x.args[0], fname)
    if isinstance(x, Expr) and x.head == "SparseArray":
        return _rows_of(ev(Expr("Normal", x)), fname)
    return [[x]]


def _exp_table(x, opts):
    sep = opts.get("FieldSeparators", "\t")
    if not isinstance(sep, str):
        raise TypeError("Export: FieldSeparators must be a string")
    return encode_text("\n".join(sep.join(field_text(v) for v in r) for r in _rows_of(x, "Export")), opts,
                       "Export")


# ── CSV / TSV ──

CSV_OPTS = ("CharacterEncoding", "EmptyField", "FieldSeparator", "FillRows", "HeaderLines", "IgnoreEmptyLines",
            "Numeric", "QuotingCharacter", "SkipLines", "IncludeQuotingCharacter", "TableHeadings", "Backend",
            "NumberPoint")


def _csv_rows(s, sep, quote):
    rows, row, field, i, n = [], [], [], 0, len(s)
    in_q, quoted, line_has = False, False, False
    while i < n:
        ch = s[i]
        if in_q:
            if ch == quote:
                if i + 1 < n and s[i + 1] == quote:
                    field.append(quote)
                    i += 2
                    continue
                in_q = False
                i += 1
                continue
            field.append(ch)
            i += 1
            continue
        if quote and ch == quote and not field:
            in_q, quoted, line_has = True, True, True
            i += 1
            continue
        if s.startswith(sep, i):
            row.append(("".join(field), quoted))
            field, quoted, line_has = [], False, True
            i += len(sep)
            continue
        if ch == "\n":
            if line_has or field:
                row.append(("".join(field), quoted))
            rows.append(row)
            row, field, quoted, line_has = [], [], False, False
            i += 1
            continue
        field.append(ch)
        line_has = True
        i += 1
    if in_q:
        raise ValueError("Import: unterminated quoted field in CSV data")
    if line_has or field:
        row.append(("".join(field), quoted))
        rows.append(row)
    return rows


def _imp_csv_generic(data, elem, subs, opts, sep_default):
    s = decode_text(data, opts, "Import")
    sep = opts.get("FieldSeparator", sep_default)
    quote = opts.get("QuotingCharacter", "\"")
    if not isinstance(sep, str) or not sep:
        raise ValueError("Import: \"FieldSeparator\" must be a nonempty string")
    if not isinstance(quote, str) or len(quote) > 1:
        raise ValueError("Import: \"QuotingCharacter\" must be a single character or \"\"")
    rows = _csv_rows(s, sep, quote)
    skip = opts.get("SkipLines", 0)
    if is_sym(skip, "Automatic"):
        skip = 0
    if not K.is_int(skip) or skip < 0:
        raise ValueError("Import: \"SkipLines\" must be a nonnegative integer")
    rows = rows[skip:]
    if opts.get("IgnoreEmptyLines", True) is not False:
        rows = [r for r in rows if r]
    raw = elem == "RawData"
    numeric = opts.get("Numeric", not raw if not raw else False)
    if is_sym(numeric, "Automatic"):
        numeric = not raw
    empty = opts.get("EmptyField", "")
    fill = opts.get("FillRows", raw)
    if is_sym(fill, "Automatic"):
        fill = raw
    out = []
    for r in rows:
        vals = []
        for text, q in r:
            if text == "" and not q:
                vals.append(empty)
            elif q:
                vals.append(text)
            else:
                vals.append(interpret_field(text, numeric is True))
        out.append(vals)
    if fill is True and out:
        w = max(len(r) for r in out)
        out = [r + [empty] * (w - len(r)) for r in out]
    if elem in ("RowCount",):
        return len(out)
    if elem == "ColumnCount":
        return max((len(r) for r in out), default=0)
    if elem == "Dimensions":
        return L(len(out), max((len(r) for r in out), default=0))
    d = L(*(L(*r) for r in out))
    if elem in ("Data", "RawData"):
        return apply_subelements(d, subs, "Import")
    if elem == "Grid":
        return Expr("Grid", apply_subelements(d, subs, "Import"))
    raise ValueError(f"Import: the element {elem!r} is not available for this format")


def _csv_field(v, sep, quote, mode, empty):
    t = field_text(v)
    if isinstance(v, str) and v == "":
        t = empty if isinstance(empty, str) else field_text(empty)
    need = any(c in t for c in (sep, "\n", "\r")) or (quote and quote in t) or (t != t.strip() and t != "")
    if quote and (is_sym(mode, "All") and isinstance(v, str) or (is_sym(mode, "Automatic") and need)):
        return quote + t.replace(quote, quote * 2) + quote
    if need and is_sym(mode, "None") and (sep in t or "\n" in t):
        raise ValueError("Export: a field contains the separator but \"IncludeQuotingCharacter\" -> None")
    return t


def _exp_csv_generic(x, opts, sep_default):
    sep = opts.get("FieldSeparator", sep_default)
    quote = opts.get("QuotingCharacter", "\"")
    mode = opts.get("IncludeQuotingCharacter", _sym("Automatic"))
    empty = opts.get("EmptyField", "")
    if not any(is_sym(mode, m) for m in ("Automatic", "All", "None")):
        raise ValueError("Export: \"IncludeQuotingCharacter\" must be Automatic, All or None")
    rows = _rows_of(x, "Export")
    heads = opts.get("TableHeadings", _sym("Automatic"))
    if is_list(heads) and heads.args and not any(is_list(h) for h in heads.args):
        rows = [list(heads.args)] + rows
    elif is_list(heads) and len(heads.args) == 2 and all(is_list(h) or is_sym(h, "None") for h in heads.args):
        rh, ch = heads.args
        if is_list(ch):
            rows = [([""] if is_list(rh) else []) + list(ch.args)] + [
                ([rh.args[i] if i < len(rh.args) else ""] if is_list(rh) else []) + r for i, r in enumerate(rows)]
        elif is_list(rh):
            rows = [[rh.args[i] if i < len(rh.args) else ""] + r for i, r in enumerate(rows)]
    elif not (is_sym(heads, "Automatic") or is_sym(heads, "None")):
        raise ValueError("Export: invalid \"TableHeadings\" setting")
    if opts.get("FillRows") is True and rows:
        w = max(len(r) for r in rows)
        rows = [r + [""] * (w - len(r)) for r in rows]
    text = "\n".join(sep.join(_csv_field(v, sep, quote, mode, empty) for v in r) for r in rows)
    return encode_text(text, opts, "Export")


CSV_ELEMENTS = ("ColumnCount", "Data", "Dimensions", "Grid", "RawData", "RowCount")

register_format(Format("Text", ["txt", "text"], ["text/plain"], ["Data", "Lines", "Plaintext", "String", "Words"],
                       "Plaintext", _imp_text, _exp_text, opts=TEXT_OPTS))
register_format(Format("String", [], [], ["String"], "String",
                       lambda d, e, s, o: decode_text(d, o, "Import"), _exp_string, opts=TEXT_OPTS))
register_format(Format("Lines", [], [], ["Data"], "Data",
                       lambda d, e, s, o: apply_subelements(L(*_lines(decode_text(d, o, "Import"))), s, "Import"),
                       _exp_lines, opts=TEXT_OPTS))
register_format(Format("List", [], [], ["Data"], "Data", _imp_list, _exp_list, opts=TEXT_OPTS))
register_format(Format("Table", ["dat", "table"], ["application/vnd.wolfram.table"], ["Data", "Grid"], "Data",
                       _imp_table, _exp_table, opts=TEXT_OPTS + ("FieldSeparators",)))
register_format(Format("CSV", ["csv"], ["text/csv", "text/comma-separated-values"], CSV_ELEMENTS, "Data",
                       lambda d, e, s, o: _imp_csv_generic(d, e, s, o, ","),
                       lambda x, o: _exp_csv_generic(x, o, ","), opts=CSV_OPTS))
register_format(Format("TSV", ["tsv", "tab"], ["text/tab-separated-values"], CSV_ELEMENTS, "Data",
                       lambda d, e, s, o: _imp_csv_generic(d, e, s, o, "\t"),
                       lambda x, o: _exp_csv_generic(x, o, "\t"), opts=CSV_OPTS))


# ── JSON / RawJSON / ExpressionJSON ──

JSON_OPTS = ("CharacterEncoding", "Compact")


def _from_json(v, raw):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return L(*(_from_json(t, raw) for t in v))
    if isinstance(v, dict):
        rules = [Expr("Rule", k, _from_json(t, raw)) for k, t in v.items()]
        return KL.make_assoc(rules) if raw else L(*rules)
    raise ValueError("Import: unsupported JSON value")


def _json_pairs_hook(pairs):
    d = {}
    for k, v in pairs:
        d[k] = v
    return d


def _imp_json_generic(data, elem, subs, opts, raw):
    if elem not in ("Data",):
        raise ValueError(f"Import: the element {elem!r} is not available for this format")
    s = decode_text(data, opts, "Import")
    try:
        v = json.loads(s, object_pairs_hook=_json_pairs_hook)
    except ValueError as e:
        raise ValueError(f"Import::jsonhintposandchar: the data is not valid JSON ({e})")
    return apply_subelements(_from_json(v, raw), subs, "Import")


def _to_json(x, fname):
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if x != x or x in (float("inf"), float("-inf")):
            raise ValueError(f"{fname}::jsonstrictencoding: {x} cannot be encoded as JSON")
        return x
    if isinstance(x, sp.Float):
        return float(x)
    if isinstance(x, str):
        return x
    if isinstance(x, Symbol) and x.name == "Null":
        return None
    if KL.is_assoc(x) or (is_list(x) and x.args and all(_is_rule(r) for r in x.args)):
        d = {}
        for r in x.args:
            k = r.args[0]
            if not isinstance(k, str):
                if isinstance(k, Symbol):
                    k = k.name
                else:
                    raise TypeError(f"{fname}::jsonstrictencoding: the key {F.input_form(k)} is not a string")
            d[k] = _to_json(r.args[1], fname)
        return d
    if is_list(x):
        return [_to_json(t, fname) for t in x.args]
    q = F.rational_value(x) if isinstance(x, Expr) else None
    if q is not None:
        return q.numerator / q.denominator
    raise ValueError(f"{fname}::jsonstrictencoding: {F.input_form(x)} cannot be encoded as JSON")


def _dump_json(v, compact):
    if compact is True:
        return json.dumps(v, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(v, ensure_ascii=False, indent="\t", separators=(",", ": "))


def _exp_json(x, opts):
    return encode_text(_dump_json(_to_json(x, "Export"), opts.get("Compact", False)), opts, "Export")


def _expr_to_ejson(x):
    if is_byte_array(x):
        return ["ByteArray", "'" + base64.b64encode(ba_bytes(x)).decode("ascii") + "'"]
    k = atom_kind(x)
    t = k[0]
    if t == "int":
        return k[1]
    if t == "real":
        return k[1]
    if t == "bigreal":
        return float(k[1])
    if t == "str":
        return "'" + k[1] + "'"
    if t == "sym":
        return k[1]
    if t == "rational":
        return ["Rational", k[1], k[2]]
    if t == "complex":
        return ["Complex", _expr_to_ejson(k[1]), _expr_to_ejson(k[2])]
    head, args = k[1], k[2]
    return [head if isinstance(head, str) else _expr_to_ejson(head)] + [_expr_to_ejson(a) for a in args]


def _ejson_to_expr(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    if v is None:
        return None
    if isinstance(v, str):
        if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
            return v[1:-1]
        if v in ("True", "False", "Null"):
            return {"True": True, "False": False, "Null": None}[v]
        if not re.fullmatch(r"[A-Za-z$][A-Za-z0-9$`]*", v):
            raise TypeError(f"Import: {v!r} is not a symbol name or a quoted string")
        return Symbol(internal_symbol(v))
    if isinstance(v, list):
        if not v:
            raise ValueError("Import: an empty JSON array is not a valid ExpressionJSON expression")
        h = _ejson_to_expr(v[0])
        args = [_ejson_to_expr(t) for t in v[1:]]
        if isinstance(h, Symbol):
            if h.name == "Rational" and len(args) == 2 and all(K.is_int(a) for a in args):
                return make_rational(*args)
            if h.name == "Complex" and len(args) == 2:
                return make_complex(*args)
            return Expr(h.name, *args)
        if isinstance(h, Expr):
            return Expr(h, *args)
        raise ValueError("Import: invalid ExpressionJSON head")
    raise ValueError("Import: JSON objects are not valid in ExpressionJSON")


def _imp_ejson(data, elem, subs, opts):
    s = decode_text(data, opts, "Import")
    try:
        v = json.loads(s)
    except ValueError as e:
        raise ValueError(f"Import: the data is not valid JSON ({e})")
    return _ejson_to_expr(v)


def _exp_ejson(x, opts):
    return encode_text(_dump_json(_expr_to_ejson(x), opts.get("Compact", False)), opts, "Export")


register_format(Format("JSON", ["json"], ["application/json"], ["Data"], "Data",
                       lambda d, e, s, o: _imp_json_generic(d, e, s, o, False), _exp_json, opts=JSON_OPTS))
register_format(Format("RawJSON", ["rawjson"], [], ["Data"], "Data",
                       lambda d, e, s, o: _imp_json_generic(d, e, s, o, True), _exp_json, opts=JSON_OPTS))
register_format(Format("ExpressionJSON", ["exprjson"], [], ["Expression"], "Expression", _imp_ejson, _exp_ejson,
                       opts=JSON_OPTS))


# ── XML (symbolic XML) ──

def _xml_name(n, nsmap):
    if " " in n:          # expat namespace mode: "uri local"
        uri, local = n.split(" ", 1)
        return L(uri, local)
    return n


def _imp_xml(data, elem, subs, opts):
    import xml.parsers.expat as expat
    p = expat.ParserCreate(namespace_separator=" ")
    p.ordered_attributes = True
    root_children, stack, prolog, epilog = [], [], [], []
    decl = []
    state = {"root": None}

    def add(node):
        if stack:
            stack[-1][2].append(node)
        elif state["root"] is None:
            prolog.append(node)
        else:
            epilog.append(node)

    def xmldecl(version, encoding, standalone):
        rules = [Expr("Rule", "Version", version or "1.0")]
        if encoding:
            rules.append(Expr("Rule", "Encoding", encoding))
        if standalone != -1:
            rules.append(Expr("Rule", "Standalone", "yes" if standalone else "no"))
        decl.append(Expr(Expr("XMLObject", "Declaration"), *rules))

    def start(name, attrs):
        al = []
        for i in range(0, len(attrs), 2):
            al.append(Expr("Rule", _xml_name(attrs[i], None), attrs[i + 1]))
        stack.append([_xml_name(name, None), al, []])

    def end(name):
        tag, al, ch = stack.pop()
        merged = []
        for c in ch:
            if isinstance(c, str) and merged and isinstance(merged[-1], str):
                merged[-1] += c
            else:
                merged.append(c)
        kids = []
        for c in merged:
            if isinstance(c, str):
                t = c.strip()
                if t:
                    kids.append(t)
            else:
                kids.append(c)
        node = Expr("XMLElement", tag, L(*al), L(*kids))
        if stack:
            stack[-1][2].append(node)
        else:
            state["root"] = node

    def chars(t):
        if stack:
            stack[-1][2].append(t)

    def comment(t):
        add(Expr(Expr("XMLObject", "Comment"), t))

    def pi(target, d):
        add(Expr(Expr("XMLObject", "ProcessingInstruction"), target, d))

    p.XmlDeclHandler = xmldecl
    p.StartElementHandler = start
    p.EndElementHandler = end
    p.CharacterDataHandler = chars
    p.CommentHandler = comment
    p.ProcessingInstructionHandler = pi
    try:
        p.Parse(data, True)
    except expat.ExpatError as e:
        raise ValueError(f"Import::fmterr: the data is not well-formed XML ({e})")
    root = state["root"]
    if root is None:
        raise ValueError("Import: the XML data has no root element")
    if elem == "XMLElement":
        return root
    if elem == "Tags":
        tags = []

        def walk(n):
            t = n.args[0]
            t = t if isinstance(t, str) else t.args[1]
            if t not in tags:
                tags.append(t)
            for c in n.args[2].args:
                if isinstance(c, Expr) and c.head == "XMLElement":
                    walk(c)
        walk(root)
        return L(*sorted(tags))
    if elem == "Plaintext":
        parts = []

        def walk2(n):
            for c in n.args[2].args:
                if isinstance(c, str):
                    parts.append(c)
                elif isinstance(c, Expr) and c.head == "XMLElement":
                    walk2(c)
        walk2(root)
        return "\n".join(parts)
    if elem != "XMLObject":
        raise ValueError(f"Import: the element {elem!r} is not available for the XML format")
    return Expr(Expr("XMLObject", "Document"), L(*(decl + prolog)), root, L(*epilog))


def _xml_escape(s, attr=False):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    if attr:
        s = s.replace('"', "&quot;")
    return s


def _xml_tag(t):
    if isinstance(t, str):
        if not re.fullmatch(r"[A-Za-z_][\w.\-:]*", t):
            raise ValueError(f"Export: {t!r} is not a valid XML name")
        return t
    if is_list(t) and len(t.args) == 2 and all(isinstance(v, str) for v in t.args):
        return _xml_tag(t.args[1])
    raise ValueError(f"Export: {F.input_form(t)} is not a valid XML tag")


def xml_text(x):
    if isinstance(x, str):
        return _xml_escape(x)
    if isinstance(x, Expr) and x.head == "XMLElement" and len(x.args) == 3:
        tag, attrs, kids = x.args
        if not (is_list(attrs) and is_list(kids)):
            raise ValueError("Export: XMLElement[tag, {attr -> val, ...}, {data, ...}] expected")
        nsdecl = []
        if is_list(tag):
            nsdecl.append(f' xmlns="{_xml_escape(tag.args[0], True)}"')
        at = []
        for r in attrs.args:
            if not (_is_rule(r) and isinstance(r.args[1], str)):
                raise ValueError("Export: XML attributes must be rules \"name\" -> \"value\"")
            at.append(f' {_xml_tag(r.args[0])}="{_xml_escape(r.args[1], True)}"')
        t = _xml_tag(tag)
        if not kids.args:
            return f"<{t}{''.join(nsdecl)}{''.join(at)}/>"
        return f"<{t}{''.join(nsdecl)}{''.join(at)}>" + "".join(xml_text(k) for k in kids.args) + f"</{t}>"
    if isinstance(x, Expr) and isinstance(x.head, Expr) and x.head.head == "XMLObject" and len(x.head.args) == 1:
        kind = x.head.args[0]
        if kind == "Comment" and len(x.args) == 1 and isinstance(x.args[0], str):
            return f"<!--{x.args[0]}-->"
        if kind == "ProcessingInstruction" and len(x.args) == 2:
            return f"<?{x.args[0]} {x.args[1]}?>"
        if kind == "Declaration":
            d = {r.args[0]: r.args[1] for r in x.args if _is_rule(r)}
            s = f'<?xml version="{d.get("Version", "1.0")}"'
            if "Encoding" in d:
                s += f' encoding="{d["Encoding"]}"'
            if "Standalone" in d:
                s += f' standalone="{d["Standalone"]}"'
            return s + "?>"
        if kind == "Document" and len(x.args) >= 2:
            pro = x.args[0].args if is_list(x.args[0]) else ()
            epi = x.args[2].args if len(x.args) > 2 and is_list(x.args[2]) else ()
            parts = [xml_text(p) for p in pro] + [xml_text(x.args[1])] + [xml_text(p) for p in epi]
            return "\n".join(parts)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return _xml_escape(field_text(x))
    raise ValueError(f"Export: {F.input_form(x)} is not symbolic XML (XMLObject or XMLElement)")


def _exp_xml(x, opts):
    return encode_text(xml_text(x), opts, "Export")


register_format(Format("XML", ["xml"], ["text/xml", "application/xml"], ["Plaintext", "Tags", "XMLElement",
                                                                         "XMLObject"],
                       "XMLObject", _imp_xml, _exp_xml, opts=TEXT_OPTS))


# ── Package / WL / M ──

def _imp_package(data, elem, subs, opts):
    s = decode_text(data, opts, "Import")
    from .wl_evaluation import split_statements, run_source
    if elem == "HeldExpressions":
        out = []
        for chunk in split_statements(s):
            e = parse_text(chunk, "Import")
            if e is not None:
                out.append(Expr("HoldComplete", e))
        return L(*out)
    if elem == "ExpressionList":
        return L(*(ev(parse_text(c, "Import")) for c in split_statements(s) if parse_text(c, "Import") is not None))
    if elem != "Expression":
        raise ValueError(f"Import: the element {elem!r} is not available for the Package format")
    return run_source(s)


def _exp_package(x, opts):
    return encode_text(input_text(x) + "\n", opts, "Export")


register_format(Format("Package", ["m", "wl", "wls"], ["application/vnd.wolfram.mathematica.package"],
                       ["Expression", "ExpressionList", "HeldExpressions"], "Expression", _imp_package,
                       _exp_package, opts=TEXT_OPTS))


# ── WXF, MX ──

MX_HEADER = b"MikoshiLang MX 1\n"

register_format(Format("WXF", ["wxf"], ["application/vnd.wolfram.wxf"], ["Expression"], "Expression",
                       lambda d, e, s, o: binary_deserialize(d, "Import"),
                       lambda x, o: binary_serialize(x, o.get("PerformanceGoal") == "Size"), binary=True,
                       opts=("PerformanceGoal",)))


def _imp_mx(data, elem, subs, opts):
    if not data.startswith(MX_HEADER):
        raise ValueError("Import: the file is not a MikoshiLang MX file (Wolfram MX files are version- and "
                         "platform-specific binary dumps and cannot be read)")
    x = binary_deserialize(data[len(MX_HEADER):], "Import")
    if isinstance(x, Expr) and x.head == "DumpSave`Definitions":
        return restore_dump(x)
    return x


register_format(Format("MX", ["mx"], [], ["Expression"], "Expression", _imp_mx,
                       lambda x, o: MX_HEADER + binary_serialize(x, True), binary=True))


# ── Base64, GZIP, BZIP2 ──

def _exp_base64(x, opts):
    if isinstance(x, str):
        b = encode_text(x, opts, "Export")
    elif is_byte_array(x):
        b = ba_bytes(x, "Export")
    else:
        b = encode_text(input_text(x), opts, "Export")
    return base64.b64encode(b)


def _imp_base64(data, elem, subs, opts):
    try:
        raw = base64.b64decode(re.sub(rb"\s+", b"", data), validate=True)
    except Exception:
        raise ValueError("Import: the data is not valid Base64")
    if elem == "ByteArray":
        return byte_array(raw)
    return decode_text(raw, opts, "Import")


register_format(Format("Base64", ["b64"], [], ["ByteArray", "String"], "String", _imp_base64, _exp_base64,
                       opts=TEXT_OPTS))


def _inner_fmt(opts, default="Text"):
    f = opts.get("\x00inner")
    return format_of(f, "Import") if f else FORMATS[default]


def _decompressing(name, decomp):
    def imp(data, elem, subs, opts):
        try:
            raw = decomp(data)
        except Exception:
            raise ValueError(f"Import: the data is not valid {name} data")
        inner = _inner_fmt(opts)
        e = elem if elem != "\x00default" and elem in inner.elements else inner.default
        return inner.imp(raw, e, subs, opts)
    return imp


def _compressing(comp):
    def exp(x, opts):
        inner = opts.get("\x00inner")
        if inner:
            f = format_of(inner, "Export")
        elif is_byte_array(x):
            return comp(ba_bytes(x, "Export"))
        else:
            f = FORMATS["Text"]
        return comp(f.exp(x, opts))
    return exp


register_format(Format("GZIP", ["gz", "gzip"], ["application/gzip", "application/x-gzip"], ["Data"], "Data",
                       _decompressing("GZIP", gzip.decompress),
                       _compressing(lambda b: gzip.compress(b, mtime=0)), binary=True, opts=CSV_OPTS + JSON_OPTS))
register_format(Format("BZIP2", ["bz2", "bzip2"], ["application/x-bzip2"], ["Data"], "Data",
                       _decompressing("BZIP2", bz2.decompress), _compressing(bz2.compress), binary=True,
                       opts=CSV_OPTS + JSON_OPTS))


# ── ZIP, TAR ──

def _member_import(name, raw, subs, opts):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    f = ext_format(ext) or FORMATS["Text"]
    return f.imp(raw, f.default, subs, opts)


def _imp_zip(data, elem, subs, opts):
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise ValueError("Import: the data is not a ZIP archive")
    names = [n for n in z.namelist() if not n.endswith("/")]
    if elem in ("FileNames", "\x00default"):
        return L(*names)
    if elem in names:
        return _member_import(elem, z.read(elem), subs, opts)
    raise ValueError(f"Import: the archive has no member {elem!r}")


def _imp_tar(data, elem, subs, opts):
    try:
        t = tarfile.open(fileobj=io.BytesIO(data), mode="r:*")
    except tarfile.TarError:
        raise ValueError("Import: the data is not a TAR archive")
    members = [m for m in t.getmembers() if m.isfile()]
    names = [m.name for m in members]
    if elem in ("FileNames", "\x00default"):
        return L(*names)
    if elem in names:
        return _member_import(elem, t.extractfile(members[names.index(elem)]).read(), subs, opts)
    raise ValueError(f"Import: the archive has no member {elem!r}")


def _archive_items(x, fname):
    if is_list(x) and all(_is_rule(r) and isinstance(r.args[0], str) for r in x.args):
        items = []
        for r in x.args:
            name = r.args[0]
            if name.startswith("/") or ".." in name.replace("\\", "/").split("/"):
                raise ValueError(f"{fname}: unsafe archive member name {name!r}")
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
            v = r.args[1]
            if is_byte_array(v):
                b = ba_bytes(v, fname)
            else:
                f = ext_format(ext) or FORMATS["Text"]
                if f.exp is None:
                    raise ValueError(f"{fname}: cannot export {name!r}")
                b = f.exp(v, {})
            items.append((name, b))
        return items
    raise ValueError(f"{fname}: archives are exported from a list {{\"name.ext\" -> expr, ...}}")


def _exp_zip(x, opts):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, b in _archive_items(x, "Export"):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, b)
    return buf.getvalue()


def _exp_tar(x, opts):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, b in _archive_items(x, "Export"):
            info = tarfile.TarInfo(name)
            info.size = len(b)
            t.addfile(info, io.BytesIO(b))
    return buf.getvalue()


register_format(Format("ZIP", ["zip"], ["application/zip"], ["FileNames"], "FileNames", _imp_zip, _exp_zip,
                       binary=True))
register_format(Format("TAR", ["tar"], ["application/x-tar"], ["FileNames"], "FileNames", _imp_tar, _exp_tar,
                       binary=True))


# ── raw binary formats ──

_BIN = {"Byte": ("B", 1), "UnsignedInteger8": ("B", 1), "Integer8": ("b", 1), "Integer16": ("h", 2),
        "Integer32": ("i", 4), "Integer64": ("q", 8), "UnsignedInteger16": ("H", 2),
        "UnsignedInteger32": ("I", 4), "UnsignedInteger64": ("Q", 8), "Real32": ("f", 4), "Real64": ("d", 8)}


def _byte_order(opts, fname):
    bo = opts.get("ByteOrdering", -1)
    if is_sym(bo, "Automatic"):
        bo = -1
    if bo not in (1, -1):
        raise ValueError(f"{fname}: ByteOrdering must be 1 (big-endian) or -1 (little-endian)")
    return ">" if bo == 1 else "<"


def _binfmt(name):
    code, size = _BIN[name]

    def imp(data, elem, subs, opts):
        bo = _byte_order(opts, "Import")
        n = len(data) // size
        vals = list(struct.unpack(bo + code * n, data[:n * size]))
        if code in "fd":
            vals = [float(v) for v in vals]
        return apply_subelements(L(*vals), subs, "Import")

    def exp(x, opts):
        bo = _byte_order(opts, "Export")
        vals = list(x.args) if is_list(x) else [x]
        out = []
        for v in vals:
            if code in "fd":
                if not (isinstance(v, (int, float, sp.Float, sp.Integer)) and not isinstance(v, bool)):
                    q = F.rational_value(v) if isinstance(v, Expr) else None
                    if q is None:
                        if isinstance(v, (Symbol, Expr)) and not K.is_number(v):
                            raise Unevaluated()
                        raise TypeError(f"Export: {F.input_form(v)} is not a real number ({name})")
                    v = q.numerator / q.denominator
                v = float(v)
                if code == "f" and math.isfinite(v) and abs(v) > 3.4028234663852886e38:
                    raise ValueError(f"Export: {v} is out of range for {name}")
            else:
                if not K.is_int(v) or isinstance(v, bool):
                    if isinstance(v, (Symbol, Expr)) and not K.is_number(v):
                        raise Unevaluated()
                    raise TypeError(f"Export: {F.input_form(v)} is not an integer ({name})")
                v = int(v)
            try:
                out.append(struct.pack(bo + code, v))
            except struct.error:
                raise ValueError(f"Export: {v} is out of range for {name}")
        return b"".join(out)
    return imp, exp


for _n in _BIN:
    _i, _e = _binfmt(_n)
    register_format(Format(_n, [], [], ["Data"], "Data", _i, _e, binary=True, opts=("ByteOrdering",)))


def _imp_char8(data, elem, subs, opts):
    return apply_subelements(L(*[chr(b) for b in data]), subs, "Import")


def _exp_char8(x, opts):
    if isinstance(x, str):
        s = x
    elif is_list(x) and all(isinstance(c, str) for c in x.args):
        s = "".join(x.args)
    else:
        raise ValueError("Export: \"Character8\" requires a string or a list of characters")
    try:
        return s.encode("latin-1")
    except UnicodeEncodeError:
        raise ValueError("Export: \"Character8\" can only store characters with codes 0..255")


register_format(Format("Character8", [], [], ["Data"], "Data", _imp_char8, _exp_char8, binary=True))


# ═══════════════════════════════════════ format detection ═══════════════════════════════════════

def ext_format(ext):
    ext = ext.lower().lstrip(".")
    if not ext:
        return None
    for f in FORMATS.values():
        if ext in f.exts:
            return f
    return None


def _split_ext(path):
    base = os.path.basename(path)
    parts = base.split(".")
    return parts[1:] if len(parts) > 1 else []


def sniff_format(data):
    if data[:2] == b"\x1f\x8b":
        return "GZIP"
    if data[:3] == b"BZh":
        return "BZIP2"
    if data[:4] == b"PK\x03\x04" or data[:4] == b"PK\x05\x06":
        return "ZIP"
    if len(data) > 262 and data[257:262] == b"ustar":
        return "TAR"
    if data.startswith(MX_HEADER):
        return "MX"
    if data[:2] == b"8:" or data[:3] == b"8C:":
        return "WXF"
    head = data[:4096].lstrip()
    if head.startswith(b"-----BEGIN ") and "PEM" in FORMATS:
        return "PEM"
    if head.startswith(b"<?xml"):
        return "XML"
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    t = text.strip()
    if t[:1] in ("{", "["):
        try:
            json.loads(t)
            return "JSON"
        except ValueError:
            pass
    return "Text"


def file_format_name(path, data=None):
    exts = _split_ext(path)
    if exts:
        f = ext_format(exts[-1])
        if f is not None:
            return f.name
    if data is None:
        if not os.path.isfile(path):
            raise ValueError(f"FileFormat::nffil: File {path} not found")
        with open(path, "rb") as fh:
            data = fh.read(1 << 16)
    s = sniff_format(data)
    if s is None:
        raise ValueError(f"FileFormat: the format of {path} could not be determined")
    return s


def _inner_from_path(path):
    exts = _split_ext(path)
    if len(exts) >= 2 and exts[-1].lower() in ("gz", "gzip", "bz2", "bzip2"):
        f = ext_format(exts[-2])
        if f is not None:
            return f.name
    return None


# ═══════════════════════════════════════ sources ═══════════════════════════════════════

_URL_RE = re.compile(r"(?i)^(https?|ftp|file)://")


def _url_string(x):
    if isinstance(x, Expr) and x.head in ("URL", "RemoteFile", "LocalObject") and x.args:
        a = x.args[0]
        if isinstance(a, Expr) and a.head == "URL" and a.args:
            a = a.args[0]
        if isinstance(a, str):
            return a
    if isinstance(x, str) and _URL_RE.match(x):
        return x
    return None


def read_source(src, fname):
    """(bytes, name used for format inference) of a file, File[...], URL[...], RemoteFile[...] or URL string."""
    u = _url_string(src)
    if u is not None:
        if u.lower().startswith("file://"):
            from urllib.parse import urlparse, unquote
            path = unquote(urlparse(u).path)
            return _read_file(path, fname), path
        if isinstance(src, Expr) and src.head == "LocalObject":
            raise ValueError(f"{fname}: only file:// local objects are supported")
        import requests
        try:
            r = requests.get(u, timeout=60)
            r.raise_for_status()
        except Exception as e:
            raise ValueError(f"{fname}::nffil: cannot read {u} ({e})")
        return r.content, u.split("?")[0]
    path = file_arg(src, fname)
    return _read_file(path, fname), path


def _read_file(path, fname):
    try:
        with open(path, "rb") as fh:
            return fh.read()
    except FileNotFoundError:
        raise ValueError(f"{fname}::nffil: File {path} not found during {fname}.")
    except IsADirectoryError:
        raise ValueError(f"{fname}: {path} is a directory")
    except OSError as e:
        raise ValueError(f"{fname}: cannot read {path} ({e.strerror})")


# ═══════════════════════════════════════ Import / Export ═══════════════════════════════════════

GENERAL_IMPORT_OPTS = ("CharacterEncoding", "ByteOrdering")


def _is_format_name(x):
    if not isinstance(x, str):
        return False
    try:
        format_of(x, "Import")
        return True
    except (ValueError, TypeError):
        return False


def parse_import_spec(spec, path_hint, data):
    """-> (Format, element or default marker, subelements, inner format name)."""
    fmt, elem, subs, inner = None, "\x00default", [], None
    items = list(spec.args) if is_list(spec) else ([spec] if spec is not None else [])
    if items and _is_format_name(items[0]):
        fmt = format_of(items[0], "Import")
        items = items[1:]
        if fmt.name in ("GZIP", "BZIP2") and items and _is_format_name(items[0]):
            inner = format_of(items[0], "Import").name
            items = items[1:]
    if items:
        if not isinstance(items[0], str):
            raise ValueError(f"Import: {F.input_form(items[0])} is not a valid element")
        elem = items[0]
        subs = items[1:]
    if fmt is None:
        if path_hint is not None:
            fmt = format_of(file_format_name(path_hint, data), "Import")
        else:
            s = sniff_format(data)
            fmt = format_of(s or "Text", "Import")
    if fmt.name in ("GZIP", "BZIP2") and inner is None and path_hint:
        inner = _inner_from_path(path_hint)
    return fmt, elem, subs, inner


def do_import(data, fmt, elem, subs, inner, opts, fname="Import"):
    if fmt.imp is None:
        raise ValueError(f"{fname}::fmtnosup: {fmt.name} is not a supported import format")
    allowed = set(fmt.opts) | set(GENERAL_IMPORT_OPTS)
    for k in opts:
        if k not in allowed:
            raise ValueError(f"{fname}: unknown option {k} for the {fmt.name} format")
    o = dict(opts)
    if inner:
        o["\x00inner"] = inner
    if elem == "Elements":
        return L(*sorted(fmt.elements))
    if elem == "Rules":
        return L(*(Expr("Rule", e, do_import(data, fmt, e, [], inner, opts, fname)) for e in sorted(fmt.elements)))
    if elem == "\x00default":
        if fmt.name in ("ZIP", "TAR", "GZIP", "BZIP2"):
            return fmt.imp(data, elem, subs, o)
        elem = PROPS_OVERRIDE.get(fmt.name, {}).get("DefaultImportElement", fmt.default)
    elif fmt.name not in ("ZIP", "TAR", "GZIP", "BZIP2") and elem not in fmt.elements:
        raise ValueError(f"{fname}::noelem: the element \"{elem}\" is not present when importing as {fmt.name}")
    return fmt.imp(data, elem, subs, o)


@fn("Import", area=AREA, status="added",
    notes="Import[src], Import[src, fmt], Import[src, elem], Import[src, {fmt, elem, sub...}] for files, File[], "
          "URL[], RemoteFile[] and http(s)/file URLs; format from the extension or the content; elements "
          "\"Elements\", \"Rules\" and each format's elements with row/column subelements; formats as in "
          "$ImportFormats (Package files are evaluated, as Get does). Check: reference (documented results per format) and Export/Import round trips in tests.")
def _import(src: String | Expr, *rest: Expr | Option) -> Expr:
    rest, opts = split_opts(rest, set().union(*(f.opts for f in FORMATS.values())) | set(GENERAL_IMPORT_OPTS),
                            "Import")
    if len(rest) > 1:
        raise ValueError("Import: called with too many arguments")
    data, name = read_source(src, "Import")
    fmt, elem, subs, inner = parse_import_spec(rest[0] if rest else None, name, data)
    return do_import(data, fmt, elem, subs, inner, opts)


def _string_bytes(s, fmt, fname):
    if fmt is not None and fmt.binary:
        try:
            return s.encode("latin-1")
        except UnicodeEncodeError:
            raise TypeError(f"{fname}: binary data must be a string of characters with codes 0..255")
    return s.encode("utf-8")


@fn("ImportString", area=AREA, status="added",
    notes="ImportString[\"data\", fmt or {fmt, elem, ...}] imports from a string (binary formats read the "
          "characters as bytes 0..255); without a format the content is sniffed (WXF, JSON, XML, compressed "
          "data, else CSV/TSV when every line has commas/tabs, else Text). Check: reference (documented ImportString results) and ExportString round trips in tests.")
def _importstring(s: String, *rest: Expr | Option) -> Expr:
    if not isinstance(s, str):
        raise TypeError("ImportString: the first argument must be a string")
    rest, opts = split_opts(rest, set().union(*(f.opts for f in FORMATS.values())) | set(GENERAL_IMPORT_OPTS),
                            "ImportString")
    if len(rest) > 1:
        raise ValueError("ImportString: called with too many arguments")
    spec = rest[0] if rest else None
    first = (spec.args[0] if is_list(spec) and spec.args else spec)
    fmt0 = format_of(first, "ImportString") if _is_format_name(first) else None
    data = _string_bytes(s, fmt0, "ImportString")
    if fmt0 is None:
        guess = sniff_format(data if all(ord(c) < 256 for c in s) else s.encode("utf-8"))
        if guess == "Text":
            lines = [x for x in s.split("\n") if x.strip()]
            if lines and all("," in x for x in lines):
                guess = "CSV"
            elif lines and all("\t" in x for x in lines):
                guess = "TSV"
        if guess in ("GZIP", "BZIP2", "ZIP", "TAR", "WXF", "MX"):
            data = _string_bytes(s, FORMATS[guess], "ImportString")
        items = list(spec.args) if is_list(spec) else ([spec] if spec is not None else [])
        spec = L(guess or "Text", *items)
    fmt, elem, subs, inner = parse_import_spec(spec, None, data)
    return do_import(data, fmt, elem, subs, inner, opts, "ImportString")


@fn("ImportByteArray", area=AREA, status="added",
    notes="ImportByteArray[ba, fmt or {fmt, elem, ...}] imports the bytes of a ByteArray in the given format "
          "(format sniffed from the content when omitted). Check: reference and ExportByteArray round trips in tests.")
def _importbytearray(b: Expr, *rest: Expr | Option) -> Expr:
    data = bytes_arg(b, "ImportByteArray")
    rest, opts = split_opts(rest, set().union(*(f.opts for f in FORMATS.values())) | set(GENERAL_IMPORT_OPTS),
                            "ImportByteArray")
    if len(rest) > 1:
        raise ValueError("ImportByteArray: called with too many arguments")
    spec = rest[0] if rest else None
    first = (spec.args[0] if is_list(spec) and spec.args else spec)
    if not _is_format_name(first):
        items = list(spec.args) if is_list(spec) else ([spec] if spec is not None else [])
        spec = L(sniff_format(data) or "Text", *items)
    fmt, elem, subs, inner = parse_import_spec(spec, None, data)
    return do_import(data, fmt, elem, subs, inner, opts, "ImportByteArray")


GENERAL_EXPORT_OPTS = ("CharacterEncoding", "ByteOrdering", "MetaInformation")


def parse_export_spec(spec, path_hint, fname):
    fmt, inner = None, None
    items = list(spec.args) if is_list(spec) else ([spec] if spec is not None else [])
    if items:
        if not isinstance(items[0], str):
            raise TypeError(f"{fname}: {F.input_form(items[0])} is not a format name")
        if not _is_format_name(items[0]):
            raise ValueError(f"{fname}: {F.input_form(items[0])} is not a supported export format")
        fmt = format_of(items[0], fname)
        if fmt.name in ("GZIP", "BZIP2") and len(items) > 1 and _is_format_name(items[1]):
            inner = format_of(items[1], fname).name
        elif len(items) > 1:
            el = items[1]
            if fmt.name == "Package" and el in ("Expression",):
                pass
            elif not (isinstance(el, str) and (el in fmt.elements or el == fmt.default)):
                raise ValueError(f"{fname}: the element {F.input_form(el)} cannot be exported as {fmt.name}")
    elif path_hint is not None:
        exts = _split_ext(path_hint)
        f = ext_format(exts[-1]) if exts else None
        if f is None:
            raise ValueError(f"{fname}::infer: Cannot infer format of file {os.path.basename(path_hint)}.")
        fmt = f
        if fmt.name in ("GZIP", "BZIP2"):
            inner = _inner_from_path(path_hint)
    else:
        raise ValueError(f"{fname}: a format must be specified")
    return fmt, inner


def do_export(x, fmt, inner, opts, fname):
    if fmt.exp is None:
        raise ValueError(f"{fname}::fmtnosup: {fmt.name} is not a supported export format")
    if "MetaInformation" in opts:
        raise NotImplementedError(f"{fname}: MetaInformation cannot be stored in the {fmt.name} format")
    allowed = set(fmt.opts) | {"CharacterEncoding", "ByteOrdering"}
    for k in opts:
        if k not in allowed:
            raise ValueError(f"{fname}: unknown option {k} for the {fmt.name} format")
    o = dict(opts)
    if inner:
        o["\x00inner"] = inner
    return fmt.exp(x, o)


def _all_opts():
    return set().union(*(f.opts for f in FORMATS.values())) | set(GENERAL_EXPORT_OPTS)


@fn("Export", area=AREA, status="added",
    notes="Export[\"file\", expr], Export[\"file\", expr, fmt], Export[\"file\", expr, {fmt, elem}] writes expr "
          "in the format given or implied by the extension (file.csv.gz: GZIP of CSV) and returns the file name; "
          "an unknown extension raises Export::infer. Check: reference (documented text per format) and Import round trips in tests.")
def _export(dest: String | Expr, x: Expr, *rest: Expr | Option) -> String:
    rest, opts = split_opts(rest, _all_opts(), "Export")
    if len(rest) > 1:
        raise ValueError("Export: called with too many arguments")
    path = file_arg(dest, "Export")
    fmt, inner = parse_export_spec(rest[0] if rest else None, path, "Export")
    data = do_export(x, fmt, inner, opts, "Export")
    try:
        with open(path, "wb") as fh:
            fh.write(data)
    except OSError as e:
        raise ValueError(f"Export: cannot write {path} ({e.strerror})")
    return dest


@fn("ExportString", area=AREA, status="added",
    notes="ExportString[expr, fmt] gives the text of expr in the format (binary formats as a string of "
          "characters with codes 0..255, as Wolfram does); CSV/TSV quote fields only when needed, JSON is "
          "tab-indented unless \"Compact\" -> True. Check: reference (documented ExportString text) and ImportString round trips in tests.")
def _exportstring(x: Expr, spec: String | ListOf[String], *rest: Option) -> String:
    rest, opts = split_opts(rest, _all_opts(), "ExportString")
    if rest:
        raise ValueError("ExportString: called with too many arguments")
    fmt, inner = parse_export_spec(spec, None, "ExportString")
    data = do_export(x, fmt, inner, opts, "ExportString")
    if fmt.binary or fmt.name in ("Base64",):
        return data.decode("latin-1")
    return data.decode(_codec(opts.get("CharacterEncoding"), "ExportString") or "utf-8")


@fn("ExportByteArray", area=AREA, status="added",
    notes="ExportByteArray[expr, fmt] gives a ByteArray with the bytes of expr exported in the format, e.g. "
          "ExportByteArray[-1, \"Integer8\"] == ByteArray[{255}], ExportByteArray[4., \"Real64\"]. Check: reference (explicit byte layouts) and ImportByteArray round trips in tests.")
def _exportbytearray(x: Expr, spec: String | ListOf[String], *rest: Option) -> Expr:
    rest, opts = split_opts(rest, _all_opts(), "ExportByteArray")
    if rest:
        raise ValueError("ExportByteArray: called with too many arguments")
    fmt, inner = parse_export_spec(spec, None, "ExportByteArray")
    return byte_array(do_export(x, fmt, inner, opts, "ExportByteArray"))


# ═══════════════════════════════════════ format registries ═══════════════════════════════════════

PROPS_OVERRIDE = {}


def _sysvar(name, getter):
    E.SYSTEM_VARIABLES[name] = getter


_sysvar("$ImportFormats", lambda: L(*sorted(n for n, f in FORMATS.items() if f.imp is not None)))
_sysvar("$ExportFormats", lambda: L(*sorted(n for n, f in FORMATS.items() if f.exp is not None)))
record("$ImportFormats", AREA, "added", "list of the formats Import supports. Check: exact (registry contents).")
record("$ExportFormats", AREA, "added", "list of the formats Export supports. Check: exact (registry contents).")


def _fmt_props(f):
    o = PROPS_OVERRIDE.get(f.name, {})
    pats = o.get("FileNamePatterns", L(*("*." + e for e in f.exts)))
    mimes = o.get("MIMETypes", L(*f.mimes))
    if f.imp is None:
        ie, de = None, None
    else:
        ie, de = L(*sorted(f.elements)), o.get("DefaultImportElement", f.default)
    return [("FileNamePatterns", pats), ("MIMETypes", mimes), ("ImportElements", ie),
            ("DefaultImportElement", de)]


_PROP_NAMES = ("FileNamePatterns", "MIMETypes", "ImportElements", "DefaultImportElement")


@fn("FileFormatProperties", area=AREA, status="added",
    notes="FileFormatProperties[fmt] gives <|\"FileNamePatterns\", \"MIMETypes\", \"ImportElements\", "
          "\"DefaultImportElement\"|>; FileFormatProperties[fmt, prop | {props}]; lists of formats and All give "
          "associations keyed by format; FileFormatProperties[] gives every format. Check: exact (registry lookup); reference values in tests.")
def _fileformatproperties(f: String = None, props: String | ListOf[String] = None) -> Assoc | Expr:
    if f is None or is_sym(f, "All"):
        names = sorted(FORMATS)
    elif is_list(f):
        names = [format_of(n, "FileFormatProperties").name for n in f.args]
    else:
        names = None
    def one(name):
        pr = dict(_fmt_props(FORMATS[name]))
        if props is None:
            return _assoc(_fmt_props(FORMATS[name]))
        if isinstance(props, str):
            if props not in pr:
                raise ValueError(f"FileFormatProperties: unknown property {props!r}")
            return pr[props]
        if is_list(props):
            for p in props.args:
                if p not in pr:
                    raise ValueError(f"FileFormatProperties: unknown property {F.input_form(p)}")
            return _assoc([(p, pr[p]) for p in props.args])
        raise TypeError("FileFormatProperties: the property must be a string or a list of strings")
    if names is None:
        return one(format_of(f, "FileFormatProperties").name)
    return _assoc([(n, one(n)) for n in names])


@fn("SetFileFormatProperties", area=AREA, status="added",
    notes="SetFileFormatProperties[fmt, prop -> val] or {rules} (or an association) changes \"FileNamePatterns\", "
          "\"MIMETypes\" or \"DefaultImportElement\" (which must be an import element) for the session; Import "
          "then uses the new default element. Check: exact; the change is read back with FileFormatProperties in tests.")
def _setfileformatproperties(f: String, spec: Rule | ListOf[Rule] | Assoc) -> Expr:
    fmt = format_of(f, "SetFileFormatProperties")
    rules = list(spec.args) if (is_list(spec) or KL.is_assoc(spec)) else [spec]
    o = PROPS_OVERRIDE.setdefault(fmt.name, {})
    new = {}
    for r in rules:
        if not _is_rule(r) or not isinstance(r.args[0], str):
            raise ValueError("SetFileFormatProperties: properties must be given as \"prop\" -> value")
        k, v = r.args
        if k == "DefaultImportElement":
            if v not in fmt.elements:
                raise ValueError(f"SetFileFormatProperties: {F.input_form(v)} is not an import element of "
                                 f"{fmt.name}")
        elif k in ("FileNamePatterns", "MIMETypes"):
            if not (is_list(v) and all(isinstance(t, str) for t in v.args)):
                raise TypeError(f"SetFileFormatProperties: {k} must be a list of strings")
        elif k == "ImportElements":
            raise ValueError("SetFileFormatProperties: \"ImportElements\" cannot be set")
        else:
            raise ValueError(f"SetFileFormatProperties: unknown property {k!r}")
        new[k] = v
    o.update(new)
    return NULL


def _pattern_formats(fname_):
    import fnmatch
    out = []
    base = os.path.basename(fname_).lower()
    for n in sorted(FORMATS):
        pats = PROPS_OVERRIDE.get(n, {}).get("FileNamePatterns")
        pats = [p.lower() for p in pats.args] if pats is not None else ["*." + e for e in FORMATS[n].exts]
        if any(fnmatch.fnmatchcase(base, p) for p in pats):
            out.append(n)
    return out


@fn("FileNameToFormatList", area=AREA, status="added",
    notes="FileNameToFormatList[\"name.ext\"] gives the formats whose file name patterns match (e.g. "
          "{\"CSV\"} for *.csv); {} when none does. Check: exact (registry lookup); reference values in tests.")
def _filenametoformatlist(name: String) -> ListOf[String]:
    return L(*_pattern_formats(file_arg(name, "FileNameToFormatList")))


@fn("MIMETypeToFormatList", area=AREA, status="added",
    notes="MIMETypeToFormatList[\"type/subtype\"] gives the formats with that MIME type (case-insensitive, "
          "parameters such as ;charset= ignored); {} when none. Check: exact (registry lookup); reference values in tests.")
def _mimetypetoformatlist(m: String) -> ListOf[String]:
    if not isinstance(m, str):
        raise TypeError("MIMETypeToFormatList: the MIME type must be a string")
    m = m.split(";")[0].strip().lower()
    out = []
    for n in sorted(FORMATS):
        ms = PROPS_OVERRIDE.get(n, {}).get("MIMETypes")
        ms = [t.lower() for t in ms.args] if ms is not None else [t.lower() for t in FORMATS[n].mimes]
        if m in ms:
            out.append(n)
    return L(*out)


@fn("FileFormat", area=AREA, status="added",
    notes="FileFormat[\"file\"] gives the Import format of a file: from its extension when known, otherwise "
          "from its content (GZIP, BZIP2, ZIP, TAR, WXF, MX, PEM, XML, JSON, Text); a missing file without a "
          "known extension raises FileFormat::nffil. Check: exact (extension, then content sniffing); reference files in tests.")
def _fileformat(f: String | Expr) -> String:
    u = _url_string(f)
    if u is not None:
        exts = _split_ext(u.split("?")[0])
        fm = ext_format(exts[-1]) if exts else None
        if fm is None:
            raise ValueError("FileFormat: cannot determine the format of the URL from its name")
        return fm.name
    return file_format_name(file_arg(f, "FileFormat"))


@fn("FileFormatQ", area=AREA, status="added",
    notes="FileFormatQ[file, fmt] is True when the file has format fmt (its FileFormat, or content that parses "
          "as fmt for JSON/XML/WXF/archives), False otherwise; an unknown format name raises. Check: exact; reference files in tests.")
def _fileformatq(f: String | Expr, fmt: String) -> Boolean:
    target = format_of(fmt, "FileFormatQ").name
    path = file_arg(f, "FileFormatQ")
    if not os.path.isfile(path):
        return False
    try:
        if file_format_name(path) == target:
            return True
    except (ValueError, TypeError):
        pass
    with open(path, "rb") as fh:
        data = fh.read()
    if target in ("Text", "String", "Lines", "Table", "List", "CSV", "TSV"):
        try:
            data.decode("utf-8")
            return target in ("Text", "String", "Lines") or sniff_format(data) == "Text"
        except UnicodeDecodeError:
            return False
    try:
        FORMATS[target].imp(data, FORMATS[target].default if target not in ("ZIP", "TAR") else "\x00default",
                            [], {})
        return True
    except Exception:
        return False


@fn("FileConvert", area=AREA, status="added",
    notes="FileConvert[\"src\" -> \"fmt\"] converts a file to the format, writing src with the format's "
          "extension; FileConvert[\"src\" -> \"dest\"] infers the format from dest; returns the new file name. Check: exact by construction (Import then Export); reference: tests re-import the converted file.")
def _fileconvert(spec: Rule, *opts: Option) -> String:
    if not _is_rule(spec):
        raise TypeError("FileConvert: FileConvert[source -> format] or FileConvert[source -> destination] "
                         "expected")
    src, target = spec.args
    path = file_arg(src, "FileConvert")
    x = _import(src)
    if isinstance(target, str) and _is_format_name(target):
        f = format_of(target, "FileConvert")
        if not f.exts:
            raise ValueError(f"FileConvert: the {f.name} format has no file extension; give a destination")
        root = path
        exts = _split_ext(path)
        if exts:
            root = path[: -(len(exts[-1]) + 1)]
        dest = root + "." + f.exts[0]
        if os.path.abspath(dest) == os.path.abspath(path):
            raise ValueError("FileConvert: the file already has that format")
        _export(dest, x, f.name, *opts)
        return dest
    dest = file_arg(target, "FileConvert")
    _export(dest, x, *opts)
    return dest


# ═══════════════════════════════════════ Put, PutAppend, Save, DumpSave ═══════════════════════════════════════

def _write_exprs(args, mode, fname):
    if not args:
        raise ValueError(f"{fname}: a file name is required")
    *exprs, dest = args
    path = file_arg(dest, fname)
    text = "".join(input_text(x) + "\n" for x in exprs)
    try:
        with open(path, mode, encoding="utf-8") as fh:
            fh.write(text)
    except OSError as e:
        raise ValueError(f"{fname}::noopen: Cannot open {path} ({e.strerror}).")
    return NULL


@fn("Put", area=AREA, status="added",
    notes="Put[e1, e2, ..., \"file\"] (expr >> file) writes each expression in InputForm on its own line, "
          "replacing the file (arbitrary-precision numbers keep their precision); Put[\"file\"] empties it; "
          "Get reads the file back. Check: exact; Put/Get round trips in tests.")
def _put(*args: Expr) -> Expr:
    return _write_exprs(args, "w", "Put")


@fn("PutAppend", area=AREA, status="added",
    notes="PutAppend[e1, ..., \"file\"] (expr >>> file) appends the expressions in InputForm, one per line. Check: exact; PutAppend/Get round trips in tests.")
def _putappend(*args: Expr) -> Expr:
    return _write_exprs(args, "a", "PutAppend")


def _held(x):
    while isinstance(x, Expr) and x.head in ("Hold", "HoldComplete", "Unevaluated") and len(x.args) == 1:
        x = x.args[0]
    return x


def _symbols_spec(spec, fname):
    """Names to save: a symbol, "name", "pattern*" (Names), "ctx`", or a list of these."""
    spec = _held(spec)
    from .wl_evaluation import _context_of_full  # noqa: F401
    if isinstance(spec, Symbol):
        return [spec.name]
    if isinstance(spec, str):
        if spec.endswith("`"):
            names = ev(Expr("Names", spec + "*"))
        elif "*" in spec or "@" in spec:
            names = ev(Expr("Names", spec))
        else:
            return [spec]
        return [internal_symbol(n) for n in names.args] if is_list(names) else []
    if isinstance(spec, Expr) and spec.head == "List":
        out = []
        for s in spec.args:
            for n in _symbols_spec(s, fname):
                if n not in out:
                    out.append(n)
        return out
    raise TypeError(f"{fname}: {F.input_form(spec)} is not a symbol, a name, a pattern or a list of these")


def _definition_text(names, full=True):
    parts = []
    for n in names:
        d = ev(Expr("ToString", Expr(Expr("FullDefinition" if full else "Definition", Symbol(n)), ),
                    Symbol("InputForm")) if False else
               Expr("ToString", Expr("FullDefinition" if full else "Definition", Symbol(n)), Symbol("InputForm")))
        if isinstance(d, str) and d.strip() and d.strip() != "Null":
            parts.append(d.strip())
    return "\n \n".join(parts)


K.BUILTIN_ATTRIBUTES["Save"] = {"HoldRest", "Protected"}
K.BUILTIN_ATTRIBUTES["DumpSave"] = {"HoldRest", "Protected"}


@fn("Save", area=AREA, status="added",
    notes="Save[\"file\", s] appends the full definitions of s and the symbols it depends on (FullDefinition: "
          "values, attributes, options, messages) in InputForm; s may be a name, a \"pattern*\", a context "
          "\"ctx`\" or a list; Get restores them; HoldRest. Check: exact; definitions restored with Get after Clear in tests.")
def _save(dest: String, spec: Held) -> Expr:
    path = file_arg(dest, "Save")
    names = _symbols_spec(spec, "Save")
    text = _definition_text(names)
    try:
        with open(path, "a", encoding="utf-8") as fh:
            if text:
                fh.write(text + "\n \n")
    except OSError as e:
        raise ValueError(f"Save::noopen: Cannot open {path} ({e.strerror}).")
    return NULL


def restore_dump(x):
    if not (len(x.args) == 2 and is_list(x.args[0]) and isinstance(x.args[1], str)):
        raise ValueError("Get: corrupt DumpSave file")
    from .wl_evaluation import run_source
    run_source(x.args[1])
    return NULL


@fn("DumpSave", area=AREA, status="added",
    notes="DumpSave[\"file.mx\", s] writes the full definitions of s (a symbol, name, context \"ctx`\" or list) "
          "to a binary MikoshiLang MX file and returns the list of symbols saved; Get[\"file.mx\"] restores them "
          "(like Wolfram MX files, the dump is specific to the system that wrote it); HoldRest. Check: exact; definitions restored with Get after Clear in tests.")
def _dumpsave(dest: String, spec: Held) -> ListOf[Expr]:
    path = file_arg(dest, "DumpSave")
    names = _symbols_spec(spec, "DumpSave")
    text = _definition_text(names)
    payload = Expr("DumpSave`Definitions", L(*names), text)
    try:
        with open(path, "wb") as fh:
            fh.write(MX_HEADER + binary_serialize(payload, True))
    except OSError as e:
        raise ValueError(f"DumpSave::noopen: Cannot open {path} ({e.strerror}).")
    return L(*(Symbol(n) for n in names))


def _hook_get():
    from . import REGISTRY
    e = REGISTRY.get("Get")
    if e is None or getattr(e, "_w13", False):
        return
    f0, lo, hi = e.impls[0]

    def get_mx(name: Expr, *opts: Option) -> Expr:
        path = None
        if isinstance(name, str) and not name.endswith("`"):
            path = os.path.expanduser(name)
        elif isinstance(name, Expr) and name.head == "File" and name.args and isinstance(name.args[0], str):
            path = os.path.expanduser(name.args[0])
        elif isinstance(name, Expr) and name.head == "LocalObject":
            u = _url_string(name)
            if u and u.lower().startswith("file://"):
                data, _ = read_source(name, "Get")
                if data.startswith(MX_HEADER):
                    return _imp_mx(data, "Expression", [], {})
                from .wl_evaluation import run_source
                return run_source(decode_text(data, {}, "Get"))
        if path and os.path.isfile(path):
            with open(path, "rb") as fh:
                head = fh.read(len(MX_HEADER))
            if head == MX_HEADER:
                if opts:
                    raise NotImplementedError("Get: options are not supported")
                return _imp_mx(_read_file(path, "Get"), "Expression", [], {})
        return f0(name, *opts)
    e.impls[0] = (get_mx, lo, hi)
    e._w13 = True


# ═══════════════════════════════════════ archives ═══════════════════════════════════════

_ARCHIVE_EXTS = (".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar", ".zip", ".gz")


def _archive_kind(path):
    p = path.lower()
    for e in _ARCHIVE_EXTS:
        if p.endswith(e):
            return e
    return None


def _truthy_opt(opts, name, default, fname):
    v = opts.get(name, default)
    if v not in (True, False):
        raise TypeError(f"{fname}: {name} must be True or False")
    return v


@fn("CreateArchive", area=AREA, status="added",
    notes="CreateArchive[src] compresses a file or directory (or a list of them) into src.zip; "
          "CreateArchive[src, dest] writes dest (.zip, .tar, .tar.gz/.tgz, .tar.bz2, or a directory to put "
          "src.zip in); returns the full archive path; OverwriteTarget -> False (default) refuses to replace. Check: exact (zipfile/tarfile); CreateArchive/ExtractArchive round trips in tests.")
def _createarchive(src: String | ListOf[String], *rest: String | Option) -> String:
    rest, opts = split_opts(rest, ("OverwriteTarget", "CreateIntermediateDirectories"), "CreateArchive")
    if len(rest) > 1:
        raise ValueError("CreateArchive: called with too many arguments")
    sources = [file_arg(s, "CreateArchive") for s in (src.args if is_list(src) else [src])]
    if not sources:
        raise ValueError("CreateArchive: nothing to archive")
    for s in sources:
        if not os.path.exists(s):
            raise ValueError(f"CreateArchive::nffil: {s} does not exist")
    if rest:
        dest = file_arg(rest[0], "CreateArchive")
        if os.path.isdir(dest):
            dest = os.path.join(dest, os.path.basename(os.path.normpath(sources[0])) + ".zip")
    else:
        dest = os.path.normpath(sources[0]) + ".zip"
    dest = os.path.abspath(dest)
    kind = _archive_kind(dest)
    if kind is None:
        raise ValueError("CreateArchive: the destination must end in .zip, .tar, .tar.gz, .tgz, .tar.bz2 or .gz")
    if os.path.exists(dest) and not _truthy_opt(opts, "OverwriteTarget", False, "CreateArchive"):
        raise ValueError(f"CreateArchive::filex: {dest} already exists")
    d = os.path.dirname(dest)
    if d and not os.path.isdir(d):
        if _truthy_opt(opts, "CreateIntermediateDirectories", True, "CreateArchive"):
            os.makedirs(d, exist_ok=True)
        else:
            raise ValueError(f"CreateArchive: the directory {d} does not exist")
    members = []
    for s in sources:
        s = os.path.normpath(s)
        top = os.path.basename(s)
        if os.path.isdir(s):
            for root, dirs, files in os.walk(s):
                dirs.sort()
                rel = os.path.relpath(root, os.path.dirname(s))
                members.append((root, rel + "/", True))
                for fn_ in sorted(files):
                    members.append((os.path.join(root, fn_), os.path.join(rel, fn_), False))
        else:
            members.append((s, top, False))
    if kind == ".zip":
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as z:
            for p, arc, isdir in members:
                if isdir:
                    z.writestr(arc.replace(os.sep, "/"), b"")
                else:
                    z.write(p, arc.replace(os.sep, "/"))
    elif kind == ".gz":
        if len(sources) != 1 or os.path.isdir(sources[0]):
            raise ValueError("CreateArchive: a .gz destination compresses exactly one file")
        with open(sources[0], "rb") as fh, gzip.open(dest, "wb") as out:
            out.write(fh.read())
    else:
        mode = {"tar": "w", ".tar": "w", ".tar.gz": "w:gz", ".tgz": "w:gz", ".tar.bz2": "w:bz2",
                ".tbz2": "w:bz2"}[kind]
        with tarfile.open(dest, mode) as t:
            for p, arc, isdir in members:
                if not isdir:
                    t.add(p, arcname=arc.replace(os.sep, "/"), recursive=False)
    return dest


def _safe_member(name):
    n = name.replace("\\", "/")
    return not (n.startswith("/") or re.match(r"^[A-Za-z]:", n) or ".." in n.split("/"))


@fn("ExtractArchive", area=AREA, status="added",
    notes="ExtractArchive[\"file.zip\"] extracts a zip/tar/tar.gz/tar.bz2/gz archive into the current directory; "
          "ExtractArchive[src, dir] into dir; ExtractArchive[src, dir, patt] only members matching a string or "
          "\"*\" wildcard pattern; returns the list of extracted files (full paths); members with absolute or .. "
          "paths are refused; OverwriteTarget -> False (default) refuses to replace existing files. Check: exact (zipfile/tarfile); round trips with CreateArchive in tests.")
def _extractarchive(src: String, *rest: String | ListOf[String] | Option) -> ListOf[String]:
    rest, opts = split_opts(rest, ("OverwriteTarget", "CreateIntermediateDirectories"), "ExtractArchive")
    if len(rest) > 2:
        raise ValueError("ExtractArchive: called with too many arguments")
    path = file_arg(src, "ExtractArchive")
    if not os.path.isfile(path):
        raise ValueError(f"ExtractArchive::nffil: {path} does not exist")
    dest = file_arg(rest[0], "ExtractArchive") if rest and not is_sym(rest[0], "Automatic") else os.getcwd()
    patt = rest[1] if len(rest) > 1 else None
    import fnmatch

    def wanted(n):
        if patt is None:
            return True
        ps = patt.args if is_list(patt) else [patt]
        for p in ps:
            if not isinstance(p, str):
                raise TypeError("ExtractArchive: the pattern must be a string or a list of strings")
            if fnmatch.fnmatchcase(n, p) or fnmatch.fnmatchcase(os.path.basename(n.rstrip("/")), p):
                return True
        return False
    overwrite = _truthy_opt(opts, "OverwriteTarget", False, "ExtractArchive")
    os.makedirs(dest, exist_ok=True)
    out = []
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                n = info.filename
                if not _safe_member(n):
                    raise ValueError(f"ExtractArchive: unsafe member path {n!r}")
                if n.endswith("/") or not wanted(n):
                    continue
                target = os.path.join(dest, n)
                if os.path.exists(target) and not overwrite:
                    raise ValueError(f"ExtractArchive::filex: {target} already exists")
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(info) as s, open(target, "wb") as o:
                    o.write(s.read())
                out.append(os.path.abspath(target))
        return L(*out)
    if magic[:2] == b"\x1f\x8b" and not tarfile.is_tarfile(path):
        base = os.path.basename(path)
        n = base[:-3] if base.lower().endswith(".gz") else base + ".out"
        target = os.path.join(dest, n)
        if os.path.exists(target) and not overwrite:
            raise ValueError(f"ExtractArchive::filex: {target} already exists")
        with gzip.open(path, "rb") as s, open(target, "wb") as o:
            o.write(s.read())
        return L(os.path.abspath(target))
    try:
        t = tarfile.open(path, "r:*")
    except tarfile.TarError:
        raise ValueError(f"ExtractArchive: {path} is not a supported archive")
    with t:
        for m in t.getmembers():
            if not _safe_member(m.name) or m.issym() or m.islnk() or m.isdev():
                raise ValueError(f"ExtractArchive: unsafe member {m.name!r}")
            if not m.isfile() or not wanted(m.name):
                continue
            target = os.path.join(dest, m.name)
            if os.path.exists(target) and not overwrite:
                raise ValueError(f"ExtractArchive::filex: {target} already exists")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with t.extractfile(m) as s, open(target, "wb") as o:
                o.write(s.read())
            out.append(os.path.abspath(target))
    return L(*out)


# ═══════════════════════════════════════ UUIDs, URLs, caches, persistence ═══════════════════════════════════════

@fn("CreateUUID", area=AREA, status="added",
    notes="CreateUUID[] gives a new random (version 4) UUID string xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx; "
          "CreateUUID[\"base\"] prefixes it with base. Check: exact (uuid4); format and uniqueness in tests.")
def _createuuid(base: String = None) -> String:
    u = str(uuid.uuid4())
    if base is None:
        return u
    if not isinstance(base, str):
        raise TypeError("CreateUUID: the base must be a string")
    return base + u


@fn("URL", area=AREA, status="added",
    notes="URL[\"url\"] is an inert symbolic URL accepted by Import, FileFormat and LocalCache (file:// URLs "
          "read local files); URL[URL[u]] is URL[u]; a non-string argument raises. Check: exact (inert wrapper with validation).")
def _url(u: String) -> Expr:
    if isinstance(u, Expr) and u.head == "URL" and len(u.args) == 1:
        return u
    if not isinstance(u, str):
        raise TypeError(f"URL: {F.input_form(u)} is not a URL string")
    raise Unevaluated()


@fn("RemoteFile", area=AREA, status="added",
    notes="RemoteFile[URL[\"url\"]] (or RemoteFile[\"url\"], normalised to URL form) is an inert reference to a "
          "remote file that Import reads. Check: exact (inert wrapper); Import through it in tests.")
def _remotefile(u: String | Expr, *opts: Option) -> Expr:
    if isinstance(u, str):
        return Expr("RemoteFile", Expr("URL", u), *opts)
    if isinstance(u, Expr) and u.head == "URL" and len(u.args) == 1 and isinstance(u.args[0], str):
        raise Unevaluated()
    raise TypeError(f"RemoteFile: {F.input_form(u)} is not a URL")


def _home_base():
    return os.path.join(os.path.expanduser("~"), ".mikoshilang")


def cache_base():
    return os.environ.get("MIKOSHILANG_CACHE_BASE") or os.path.join(_home_base(), "Caches")


def default_local_base():
    return os.environ.get("MIKOSHILANG_LOCAL_BASE") or os.path.join(_home_base(), "Objects")


_sysvar("$CacheBaseDirectory", cache_base)
_sysvar("$DefaultLocalBase", default_local_base)
_sysvar("$LocalBase", default_local_base)


def _localsymbol_base():
    from .wl_evaluation import local_symbol_base
    return local_symbol_base()


_sysvar("$LocalSymbolBase", _localsymbol_base)
record("$CacheBaseDirectory", AREA, "added",
       "directory where LocalCache stores cached copies (~/.mikoshilang/Caches or $MIKOSHILANG_CACHE_BASE). Check: exact (environment/home lookup).")
record("$DefaultLocalBase", AREA, "added",
       "default base directory for local object storage (~/.mikoshilang/Objects or $MIKOSHILANG_LOCAL_BASE). Check: exact (environment/home lookup).")
record("$LocalBase", AREA, "added", "base directory for local objects; initially $DefaultLocalBase. Check: exact.")
record("$LocalSymbolBase", AREA, "added",
       "directory where LocalSymbol values are stored (~/.mikoshilang/LocalSymbols or "
       "$MIKOSHILANG_LOCALSYMBOL_BASE). Check: exact (shared with LocalSymbol).")


def _file_url(path):
    from urllib.parse import quote
    return "file://" + quote(os.path.abspath(path))


@fn("LocalCache", area=AREA, status="added",
    notes="LocalCache[URL[url]] downloads url (file:// or http(s)) into $CacheBaseDirectory and returns "
          "LocalObject[\"file:///...\"], which Import and Get read; LocalCache[obj, LocalObject[\"name\" or "
          "\"file:///path\"]] chooses the cache; re-evaluation refreshes the copy; CloudObject caches are not "
          "available (no Wolfram Cloud). Check: exact; the cached copy is re-imported in tests.")
def _localcache(obj: Expr, loc: Expr = None, *opts: Option) -> Expr:
    if isinstance(obj, Expr) and obj.head == "CloudObject":
        raise NotImplementedError("LocalCache: Wolfram Cloud objects are not available in MikoshiLang")
    u = _url_string(obj)
    if u is None:
        raise TypeError(f"LocalCache: {F.input_form(obj)} is not a URL[...] to cache")
    import hashlib
    if loc is None:
        tail = os.path.basename(u.split("?")[0]) or "index"
        name = hashlib.sha1(u.encode()).hexdigest()[:16] + "-" + re.sub(r"[^A-Za-z0-9_.-]", "_", tail)[:60]
        path = os.path.join(cache_base(), name)
    elif isinstance(loc, Expr) and loc.head == "LocalObject" and loc.args and isinstance(loc.args[0], str):
        a = loc.args[0]
        if a.lower().startswith("file://"):
            from urllib.parse import urlparse, unquote
            path = unquote(urlparse(a).path)
        else:
            if not _safe_member(a):
                raise ValueError("LocalCache: invalid local object name")
            path = os.path.join(default_local_base(), a)
    else:
        raise ValueError("LocalCache: the cache location must be LocalObject[...]")
    data, _ = read_source(Expr("URL", u), "LocalCache")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)
    return Expr("LocalObject", _file_url(path))


_PERSIST_TYPES = ("KernelSession", "FrontEndSession", "Notebook", "Local", "LocalShared", "Cloud", "Installation")


@fn("PersistenceLocation", area=AREA, status="added",
    notes="PersistenceLocation[\"type\"] for the documented types (KernelSession, FrontEndSession, Notebook, Local, "
          "LocalShared, Cloud, Installation); \"Local\" gets its default base LocalObject[file URL of "
          "$DefaultLocalBase/Persistence]; PersistenceLocation[\"type\" -> base] is PersistenceLocation[\"type\", "
          "base]; unknown types raise. Check: exact (inert wrapper with validation).")
def _persistencelocation(t: String, base: Expr = None) -> Expr:
    if _is_rule(t) and base is None:
        return Expr("PersistenceLocation", t.args[0], t.args[1])
    if not isinstance(t, str):
        raise TypeError(f"PersistenceLocation: {F.input_form(t)} is not a persistence location type name")
    if t not in _PERSIST_TYPES:
        raise ValueError(f"PersistenceLocation: {F.input_form(t)} is not a persistence location type "
                         f"({', '.join(_PERSIST_TYPES)})")
    if base is None:
        if t == "Local":
            return Expr("PersistenceLocation", t,
                        Expr("LocalObject", _file_url(os.path.join(default_local_base(), "Persistence"))))
        raise Unevaluated()
    if isinstance(base, str) and t in ("Local", "LocalShared") and not base.lower().startswith("file://"):
        return Expr("PersistenceLocation", t, Expr("LocalObject", _file_url(base)))
    raise Unevaluated()


@fn("ToMemory", area=AREA, status="added",
    notes="ToMemory[obj] gives the in-memory version of an out-of-core object; MikoshiLang keeps every object "
          "in memory, so it returns obj unchanged. Check: exact (identity).")
def _tomemory(x: Expr) -> Expr:
    return x


def _option_symbol(name, notes):
    from .wl_evaluation import INERT
    INERT[name] = notes
    record(name, AREA, "added", notes.rstrip(". ") + ". Check: exact (inert symbol; tests check it stays "
                                                       "unevaluated and is used as documented).")


_option_symbol("CachePersistence", "option for URL fetching functions: how long downloaded content may be cached "
                                   "(Automatic, Infinity, a Quantity or None); inert symbol")
_option_symbol("PersistenceTime", "option for persistent values: how long a value is kept (Infinity, a time "
                                  "in seconds or a DateObject); inert symbol")
_option_symbol("ConversionRules", "option for Import/Export of some formats giving rules applied while "
                                  "converting; inert symbol (no MikoshiLang format uses it, so Import/Export "
                                  "reject it)")
_option_symbol("DefaultElement", "option for importers naming the element imported by default; see "
                                 "FileFormatProperties[fmt, \"DefaultImportElement\"]; inert symbol")
_option_symbol("ImportOptions", "element/option name used in format registrations to describe import "
                                "options; inert symbol")
_option_symbol("IncludeMetaInformation", "option for Import/Export of some formats controlling whether meta "
                                         "information is included; inert symbol (the supported formats carry "
                                         "none, so Import/Export reject it)")
_option_symbol("MetaInformation", "option for Export of formats that carry metadata; Export raises a clear error "
                                  "for every supported format since none can store it")
_option_symbol("VerifySecurityCertificates", "option for URL functions: whether to verify the server's TLS "
                                             "certificate (True by default); inert symbol")
_option_symbol("Boxes", "element name for the box (typesetting) form of an expression in Import/Export; "
                        "inert symbol")
_option_symbol("XMLObject", "XMLObject[\"type\"][...] represents a symbolic XML object (Document, Declaration, "
                            "Comment, ProcessingInstruction); produced by Import of XML and accepted by Export")
_option_symbol("XMLElement", "XMLElement[tag, {attr -> val, ...}, {data, ...}] represents a symbolic XML element; "
                             "produced by Import of XML and accepted by Export")
_option_symbol("ImportedObject", "ImportedObject[assoc] holds imported data with no Wolfram Language "
                                 "representation (e.g. unsupported PEM entries); obj[\"prop\"] gives its "
                                 "properties, including \"OriginalFormat\" and \"RawContent\"")


def _imported_object_hook(head, args, ev_):
    if len(head.args) == 1 and KL.is_assoc(head.args[0]) and len(args) == 1 and isinstance(args[0], str):
        a = head.args[0]
        if args[0] == "Properties":
            return L(*(r.args[0] for r in a.args))
        v = assoc_get(a, args[0], K.UNHANDLED)
        if v is K.UNHANDLED:
            return Expr("Missing", "NotAvailable")
        return v
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["ImportedObject"] = _imported_object_hook
_hook_get()
