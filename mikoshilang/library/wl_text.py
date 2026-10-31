"""Wolfram batch W11: strings and text.

Templates (StringTemplate, TemplateObject, TemplateApply, XMLTemplate, FileTemplate, ...), text statistics
(CharacterCounts, LetterCounts, WordCounts, WordStem, CharacterName, CharacterNormalize), byte arrays, base
encodings and hashing (ByteArray, BaseEncode, Hash, FileHash), syntax tests (SyntaxQ, SyntaxLength), sequence
alignment and similarity (SequenceAlignment, NeedlemanWunschSimilarity, SmithWatermanSimilarity,
DamerauLevenshteinDistance), repeats and ordered/common sequences (FindRepeat, FindTransientRepeat,
LongestOrderedSequence, LongestCommon*Positions), three-way diffs (Diff3, DiffObject, DiffApply),
biomolecular sequences (BioSequence and the BioSequence* functions), text search over a local in-memory index
(CreateSearchIndex, TextSearch, ContentObject, SearchResultObject, Snippet) and line search in files and
string streams (Find, FindList, StringToStream).

Quality bar: a correct result or a clear error.  Where Wolfram's output depends on undocumented internals
(tie-breaking among equally good alignments, search ranking scores) the notes in STATUS say so.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import html as _html
import itertools
import math
import os
import re
import struct
import unicodedata
import zlib

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated
from .wl_common import L, ev, apply, is_sym, is_list, is_rule, is_assoc, split_options

AREA = "strings & text (W11)"


def _str(x, fname, what="argument"):
    if not isinstance(x, str):
        if isinstance(x, Symbol) or (isinstance(x, Expr) and not is_list(x)):
            raise Unevaluated()
        raise ValueError(f"{fname}: {what} {x!r} must be a string")
    return x


def _opt_name(r):
    if is_rule(r):
        k = r.args[0]
        if isinstance(k, Symbol):
            return k.name
        if isinstance(k, str):
            return k
    return None


def _opts(args, allowed, fname):
    """Split trailing options (symbol or string names) from positional arguments; unknown options raise."""
    args = list(args)
    opts = {}
    while args and _opt_name(args[-1]) is not None:
        r = args.pop()
        n = _opt_name(r)
        if n not in allowed:
            raise ValueError(f"{fname}: unknown option {n}")
        opts[n] = r.args[1]
    return args, opts


def _bool_opt(opts, name, fname, default=False):
    v = opts.get(name, default)
    if v is True or v is False:
        return v
    raise ValueError(f"{fname}: {name} must be True or False")


def _assoc(pairs):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in pairs))


def _missing(*a):
    return Expr("Missing", *a)


def _int(x, fname, what="argument", lo=None):
    if isinstance(x, bool) or not isinstance(x, int):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: {what} {x!r} must be an integer")
    if lo is not None and x < lo:
        raise ValueError(f"{fname}: {what} must be at least {lo}")
    return x


def _hold(*names):
    from ..evaluate import set_hold_all
    for n in names:
        set_hold_all(n)


def _hold_first(*names):
    from ..evaluate import set_hold_first
    for n in names:
        set_hold_first(n)


def _inert(name, notes):
    record(name, AREA, "added", "inert symbol: " + notes)


# ═════════════════════════════════════════════════════════════════ byte arrays ══
def _ba(data: bytes):
    return Expr("ByteArray", base64.b64encode(bytes(data)).decode("ascii"))


def is_bytearray(x):
    return isinstance(x, Expr) and x.head == "ByteArray" and len(x.args) == 1 and isinstance(x.args[0], str)


def ba_bytes(x, fname="ByteArray"):
    if not is_bytearray(x):
        raise ValueError(f"{fname}: {x!r} is not a ByteArray")
    try:
        return base64.b64decode(x.args[0], validate=True)
    except (binascii.Error, ValueError):
        raise ValueError(f"{fname}: invalid Base64 data in ByteArray")


@fn("ByteArray", area=AREA, status="added",
    notes="ByteArray[{b1, ...}] (bytes 0..255) and ByteArray[\"base64\"]; the evaluated form is "
          "ByteArray[\"base64\"] (Wolfram's InputForm); Normal and Length work on it")
def _bytearray(x):
    if isinstance(x, str):
        if x == "":
            raise Unevaluated()
        try:
            data = base64.b64decode(x, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("ByteArray: the string is not valid Base64 data")
        if base64.b64encode(data).decode("ascii") == x:
            raise Unevaluated()
        return _ba(data)
    if is_list(x):
        out = []
        for b in x.args:
            if isinstance(b, bool) or not isinstance(b, int) or not 0 <= b <= 255:
                raise ValueError(f"ByteArray: {b!r} is not a byte value (an integer 0..255)")
            out.append(b)
        return _ba(bytes(out))
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError("ByteArray: the argument must be a list of bytes or a Base64 string")


@fn("ByteArrayQ", area=AREA, status="added", notes="True for a valid ByteArray object, False otherwise")
def _bytearrayq(x):
    if not is_bytearray(x):
        return False
    try:
        base64.b64decode(x.args[0], validate=True)
        return True
    except (binascii.Error, ValueError):
        return False


def _bytes_of(x, fname):
    """Bytes of a ByteArray, or of {} (accepted as an empty byte array)."""
    if is_list(x) and not x.args:
        return b""
    if is_bytearray(x):
        return ba_bytes(x, fname)
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not a ByteArray")


_ENCODINGS = {
    "UTF-8": "utf-8", "UTF8": "utf-8", "ASCII": "ascii", "PrintableASCII": "ascii",
    "ISOLatin1": "latin-1", "ISO8859-1": "latin-1", "ISOLatin2": "iso8859-2", "ISO8859-2": "iso8859-2",
    "ISOLatin3": "iso8859-3", "ISO8859-3": "iso8859-3", "ISOLatin4": "iso8859-4", "ISO8859-4": "iso8859-4",
    "ISOLatinCyrillic": "iso8859-5", "ISO8859-5": "iso8859-5", "ISO8859-6": "iso8859-6",
    "ISO8859-7": "iso8859-7", "ISO8859-8": "iso8859-8", "ISO8859-9": "iso8859-9", "ISO8859-10": "iso8859-10",
    "ISO8859-13": "iso8859-13", "ISO8859-14": "iso8859-14", "ISO8859-15": "iso8859-15",
    "ISO8859-16": "iso8859-16", "UTF-16": "utf-16", "UTF-16BE": "utf-16-be", "UTF-16LE": "utf-16-le",
    "UTF-32": "utf-32", "UTF-32BE": "utf-32-be", "UTF-32LE": "utf-32-le", "WindowsANSI": "cp1252",
    "CP1252": "cp1252", "WindowsCyrillic": "cp1251", "CP1251": "cp1251", "WindowsGreek": "cp1253",
    "CP1250": "cp1250", "WindowsEastEurope": "cp1250", "KOI8-R": "koi8-r", "ShiftJIS": "shift_jis",
    "EUC-JP": "euc_jp", "GB2312": "gb2312", "Big5": "big5", "MacintoshRoman": "mac_roman",
    "IBM-850": "cp850", "CP850": "cp850", "CP437": "cp437",
}


def _codec(enc, fname):
    if enc is None:
        return "utf-8"
    if not isinstance(enc, str):
        raise ValueError(f"{fname}: the encoding must be a string")
    c = _ENCODINGS.get(enc)
    if c is None:
        raise ValueError(f"{fname}: unknown or unsupported character encoding {enc!r}")
    return c


@fn("StringToByteArray", area=AREA, status="added",
    notes="bytes of a string in UTF-8 (default) or a named $CharacterEncodings encoding; a character the "
          "encoding cannot represent raises")
def _str2ba(s, enc=None):
    s = _str(s, "StringToByteArray")
    c = _codec(enc, "StringToByteArray")
    try:
        return _ba(s.encode(c))
    except UnicodeEncodeError as e:
        raise ValueError(f"StringToByteArray: {s[e.start]!r} has no code in the encoding {enc}")


@fn("ByteArrayToString", area=AREA, status="added",
    notes="decodes a ByteArray (or {}) as UTF-8 (default) or a named encoding; invalid data raises")
def _ba2str(b, enc=None):
    data = _bytes_of(b, "ByteArrayToString")
    c = _codec(enc, "ByteArrayToString")
    try:
        return data.decode(c)
    except UnicodeDecodeError:
        raise ValueError(f"ByteArrayToString: the bytes are not valid {enc or 'UTF-8'} data")


# Normal[ByteArray[...]] and Length[ByteArray[...]] (chained onto the existing kernel definitions)
_prev_normal = K._BUILTINS.get("Normal")
_prev_length = K._BUILTINS.get("Length")


def _normal_w11(a, ev_, expr):
    if len(a) == 1 and is_bytearray(a[0]):
        return L(*ba_bytes(a[0], "Normal"))
    return _prev_normal(a, ev_, expr)


def _length_w11(a, ev_, expr):
    if len(a) == 1 and is_bytearray(a[0]):
        return len(ba_bytes(a[0], "Length"))
    return _prev_length(a, ev_, expr)


K._BUILTINS["Normal"] = _normal_w11
K._BUILTINS["Length"] = _length_w11


# ── base encodings ──
_B85_ENC = {"Base85ASCII", "Ascii85", "Base85"}


def _a85_decode(s):
    s = "".join(c for c in s if c == "z" or "!" <= c <= "u")
    out = bytearray()
    group = []
    for c in s:
        if c == "z":
            if group:
                raise ValueError("BaseDecode: 'z' inside an Ascii85 group")
            out += b"\0\0\0\0"
            continue
        group.append(ord(c) - 33)
        if len(group) == 5:
            v = 0
            for d in group:
                v = v * 85 + d
            if v > 0xFFFFFFFF:
                raise ValueError("BaseDecode: invalid Ascii85 group")
            out += struct.pack(">I", v)
            group = []
    if group:
        if len(group) == 1:
            raise ValueError("BaseDecode: a final Ascii85 group needs at least two characters")
        n = len(group)
        group += [84] * (5 - n)
        v = 0
        for d in group:
            v = v * 85 + d
        out += struct.pack(">I", v & 0xFFFFFFFF)[:n - 1]
    return bytes(out)


def _a85_encode(b):
    out = []
    for i in range(0, len(b), 4):
        chunk = b[i:i + 4]
        n = len(chunk)
        if n == 4 and chunk == b"\0\0\0\0":
            out.append("z")
            continue
        v = struct.unpack(">I", chunk + b"\0" * (4 - n))[0]
        digits = []
        for _ in range(5):
            digits.append(chr(v % 85 + 33))
            v //= 85
        out.append("".join(reversed(digits))[:n + 1])
    return "".join(out)


@fn("BaseEncode", area=AREA, status="added",
    notes="\"Base64\" (RFC 4648, padded; default), \"Base16\" (upper-case hex) and "
          "\"Base85ASCII\" (Ascii85 with the z shortcut, no padding) encodings of a ByteArray")
def _baseencode(b, enc="Base64"):
    data = _bytes_of(b, "BaseEncode")
    if enc == "Base64":
        return base64.b64encode(data).decode("ascii")
    if enc == "Base16":
        return data.hex().upper()
    if enc in _B85_ENC:
        return _a85_encode(data)
    raise ValueError(f"BaseEncode: unknown encoding {enc!r}")


@fn("BaseDecode", area=AREA, status="added",
    notes="inverse of BaseEncode, giving a ByteArray; characters outside the alphabet are dropped; Base64 input "
          "must be padded to a multiple of 4 (otherwise an error)")
def _basedecode(s, enc="Base64"):
    s = _str(s, "BaseDecode")
    try:
        if enc == "Base64":
            t = re.sub("[^A-Za-z0-9+/=]", "", s)
            if len(t) % 4:
                raise ValueError("BaseDecode: Base64 input must be padded to a multiple of 4 characters")
            data = base64.b64decode(t, validate=True)
        elif enc == "Base16":
            t = re.sub("[^0-9A-Fa-f]", "", s)
            if len(t) % 2:
                raise ValueError("BaseDecode: Base16 input must have an even number of digits")
            data = bytes.fromhex(t)
        elif enc in _B85_ENC:
            data = _a85_decode(s)
        else:
            raise ValueError(f"BaseDecode: unknown encoding {enc!r}")
    except binascii.Error as e:
        raise ValueError(f"BaseDecode: invalid input ({e})")
    return _ba(data)


# ═════════════════════════════════════════════════════════════════ hashing ══
def _md4(data: bytes) -> bytes:
    def lrot(x, n):
        x &= 0xFFFFFFFF
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF
    msg = bytearray(data)
    ml = (8 * len(data)) & 0xFFFFFFFFFFFFFFFF
    msg.append(0x80)
    while len(msg) % 64 != 56:
        msg.append(0)
    msg += struct.pack("<Q", ml)
    h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476]
    F = lambda x, y, z: (x & y) | (~x & z)  # noqa: E731
    G = lambda x, y, z: (x & y) | (x & z) | (y & z)  # noqa: E731
    H = lambda x, y, z: x ^ y ^ z  # noqa: E731
    for i in range(0, len(msg), 64):
        X = struct.unpack("<16I", bytes(msg[i:i + 64]))
        a, b, c, d = h
        for k in range(16):
            s = (3, 7, 11, 19)[k % 4]
            a, b, c, d = d, lrot(a + F(b, c, d) + X[k], s), b, c
        for k in range(16):
            s = (3, 5, 9, 13)[k % 4]
            idx = (k % 4) * 4 + k // 4
            a, b, c, d = d, lrot(a + G(b, c, d) + X[idx] + 0x5A827999, s), b, c
        for k in range(16):
            s = (3, 9, 11, 15)[k % 4]
            idx = (0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15)[k]
            a, b, c, d = d, lrot(a + H(b, c, d) + X[idx] + 0x6ED9EBA1, s), b, c
        h = [(x + y) & 0xFFFFFFFF for x, y in zip(h, (a, b, c, d))]
    return struct.pack("<4I", *h)


def _ripemd160_py(data: bytes) -> bytes:
    def rol(x, n):
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF
    fs = [lambda x, y, z: x ^ y ^ z, lambda x, y, z: (x & y) | (~x & z), lambda x, y, z: (x | ~y) ^ z,
          lambda x, y, z: (x & z) | (y & ~z), lambda x, y, z: x ^ (y | ~z)]
    KL_ = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E]
    KR = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000]
    RL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
          3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
          4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13]
    RR = [5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
          15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
          12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11]
    SL = [11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13, 12,
          11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6, 5, 12,
          9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6]
    SR = [8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13, 11,
          9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5, 15, 8,
          8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11]
    msg = bytearray(data)
    ml = (8 * len(data)) & 0xFFFFFFFFFFFFFFFF
    msg.append(0x80)
    while len(msg) % 64 != 56:
        msg.append(0)
    msg += struct.pack("<Q", ml)
    h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
    for i in range(0, len(msg), 64):
        X = struct.unpack("<16I", bytes(msg[i:i + 64]))
        al, bl, cl, dl, el = h
        ar, br, cr, dr, er = h
        for j in range(80):
            r = j // 16
            t = (rol((al + (fs[r](bl, cl, dl) & 0xFFFFFFFF) + X[RL[j]] + KL_[r]) & 0xFFFFFFFF, SL[j]) + el) & 0xFFFFFFFF
            al, el, dl, cl, bl = el, dl, rol(cl, 10), bl, t
            t = (rol((ar + (fs[4 - r](br, cr, dr) & 0xFFFFFFFF) + X[RR[j]] + KR[r]) & 0xFFFFFFFF, SR[j]) + er) \
                & 0xFFFFFFFF
            ar, er, dr, cr, br = er, dr, rol(cr, 10), br, t
        t = (h[1] + cl + dr) & 0xFFFFFFFF
        h[1] = (h[2] + dl + er) & 0xFFFFFFFF
        h[2] = (h[3] + el + ar) & 0xFFFFFFFF
        h[3] = (h[4] + al + br) & 0xFFFFFFFF
        h[4] = (h[0] + bl + cr) & 0xFFFFFFFF
        h[0] = t
    return struct.pack("<5I", *h)


def _ripemd160(data: bytes) -> bytes:
    try:
        return hashlib.new("ripemd160", data).digest()
    except (ValueError, TypeError):
        return _ripemd160_py(data)


_KECCAK_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
              0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
              0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
              0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
              0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_KECCAK_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56],
               [27, 20, 39, 8, 14]]
_M64 = 0xFFFFFFFFFFFFFFFF


def _keccak_f(A):
    for rc in _KECCAK_RC:
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ (((C[(x + 1) % 5] << 1) | (C[(x + 1) % 5] >> 63)) & _M64) for x in range(5)]
        A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                r = _KECCAK_ROT[x][y]
                v = A[x][y]
                B[y][(2 * x + 3 * y) % 5] = ((v << r) | (v >> (64 - r))) & _M64 if r else v
        A = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]) for y in range(5)] for x in range(5)]
        A[0][0] ^= rc
    return A


def _keccak(data: bytes, bits: int, pad=0x01) -> bytes:
    rate = 200 - 2 * (bits // 8)
    msg = bytearray(data)
    msg.append(pad)
    while len(msg) % rate:
        msg.append(0)
    msg[-1] |= 0x80
    A = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        block = msg[off:off + rate]
        for i in range(rate // 8):
            x, y = i % 5, i // 5
            A[x][y] ^= struct.unpack("<Q", bytes(block[8 * i:8 * i + 8]))[0]
        A = _keccak_f(A)
    out = b""
    for i in range(rate // 8):
        x, y = i % 5, i // 5
        out += struct.pack("<Q", A[x][y])
    return out[:bits // 8]


def _hash_function(t, fname):
    """(function bytes -> digest bytes, bits) for a Wolfram hash type."""
    def lib(name, **kw):
        return lambda d: hashlib.new(name, d, **kw).digest()
    simple = {
        "MD5": (lib("md5"), 128), "SHA": (lib("sha1"), 160), "SHA1": (lib("sha1"), 160),
        "SHA224": (lib("sha224"), 224), "SHA256": (lib("sha256"), 256), "SHA384": (lib("sha384"), 384),
        "SHA512": (lib("sha512"), 512), "SHA3-224": (lib("sha3_224"), 224), "SHA3-256": (lib("sha3_256"), 256),
        "SHA3-384": (lib("sha3_384"), 384), "SHA3-512": (lib("sha3_512"), 512),
        "SHA256SHA256": (lambda d: hashlib.sha256(hashlib.sha256(d).digest()).digest(), 256),
        "RIPEMD160": (_ripemd160, 160),
        "RIPEMD160SHA256": (lambda d: _ripemd160(hashlib.sha256(d).digest()), 160),
        "MD4": (_md4, 128),
        "Keccak224": (lambda d: _keccak(d, 224), 224), "Keccak256": (lambda d: _keccak(d, 256), 256),
        "Keccak384": (lambda d: _keccak(d, 384), 384), "Keccak512": (lambda d: _keccak(d, 512), 512),
        "BLAKE2b-512": (lambda d: hashlib.blake2b(d, digest_size=64).digest(), 512),
        "BLAKE2s-256": (lambda d: hashlib.blake2s(d, digest_size=32).digest(), 256),
        "Adler32": (lambda d: struct.pack(">I", zlib.adler32(d) & 0xFFFFFFFF), 32),
        "CRC32": (lambda d: struct.pack(">I", zlib.crc32(d) & 0xFFFFFFFF), 32),
    }
    if isinstance(t, str):
        if t in simple:
            return simple[t]
        raise ValueError(f"{fname}: unknown or unsupported hash type {t!r}")
    if is_list(t) and len(t.args) == 2 and isinstance(t.args[0], str) and isinstance(t.args[1], int):
        name, b = t.args
        if name == "SHA" and b in (224, 256, 384, 512):
            return simple[f"SHA{b}"]
        if name == "SHA3" and b in (224, 256, 384, 512):
            return simple[f"SHA3-{b}"]
        if name == "Keccak" and b in (224, 256, 384, 512):
            return simple[f"Keccak{b}"]
        if name in ("BLAKE2b", "BLAKE2s") and b > 0 and b % 8 == 0 and b <= (512 if name == "BLAKE2b" else 256):
            f = hashlib.blake2b if name == "BLAKE2b" else hashlib.blake2s
            return (lambda d: f(d, digest_size=b // 8).digest()), b
        raise ValueError(f"{fname}: unsupported hash type {t!r}")
    raise ValueError(f"{fname}: unknown hash type {t!r}")


_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def _to_base36(n):
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(_B36[r])
    return "".join(reversed(out))


def _hash_format(digest: bytes, bits, fmt, fname):
    n = int.from_bytes(digest, "big")
    if fmt == "Integer":
        return n
    if fmt == "DecimalString":
        return str(n).rjust(len(str(2 ** bits - 1)), "0")
    if fmt == "HexString":
        return digest.hex()
    if fmt == "HexStringLittleEndian":
        return digest[::-1].hex()
    if fmt == "Base36String":
        return _to_base36(n).rjust(len(_to_base36(2 ** bits - 1)), "0")
    if fmt == "Base64Encoding":
        return base64.b64encode(digest).decode("ascii")
    if fmt == "ByteArray":
        return _ba(digest)
    raise ValueError(f"{fname}: unknown hash format {fmt!r}")


def _fullform_safe(x):
    """True if ToString[FullForm[x]] here is known to equal Wolfram's text (integers, rationals, plain symbols,
    printable-ASCII strings and compound expressions of them)."""
    if isinstance(x, bool):
        return True
    if isinstance(x, int):
        return True
    if isinstance(x, str):
        return all(32 <= ord(c) < 127 and c not in '"\\' for c in x)
    if isinstance(x, Symbol):
        return x.name not in ("I",) and re.fullmatch(r"\$?[A-Za-z][A-Za-z0-9`$]*", x.name) is not None
    if isinstance(x, Expr):
        if x.head in ("Complex", "Association", "ByteArray"):
            return False
        h = x.head
        if not (isinstance(h, str) or _fullform_safe(h)):
            return False
        return all(_fullform_safe(a) for a in x.args)
    return False


def _hash_bytes(x, fname):
    if isinstance(x, str):
        return x.encode("utf-8")
    if is_bytearray(x):
        return ba_bytes(x, fname)
    if not _fullform_safe(x):
        raise NotImplementedError(f"{fname}: hashing this expression needs Wolfram's exact FullForm text, which is "
                                  "only reproduced for integers, rationals, symbols, ASCII strings and expressions "
                                  "built from them")
    return ev(Expr("ToString", Expr("FullForm", x))).encode("utf-8")


def _expression_hash(x):
    text = ev(Expr("ToString", Expr("FullForm", x)))
    if not isinstance(text, str):
        text = repr(x)
    d = hashlib.sha256(b"MikoshiLang-Expression\0" + text.encode("utf-8")).digest()[:8]
    return bytes([d[0] & 0x7F]) + d[1:]


@fn("Hash", area=AREA, status="added",
    notes="Hash[expr, type, format]: Adler32, CRC32, MD4, MD5, RIPEMD160, RIPEMD160SHA256, SHA (SHA-1), SHA224/256/"
          "384/512, SHA256SHA256, SHA3-*, Keccak*, BLAKE2b/BLAKE2s ({type, bits}); strings hash their UTF-8 bytes, "
          "ByteArrays their bytes, other expressions the bytes of ToString[FullForm[expr]] (only where that text "
          "matches Wolfram's; otherwise an error). The default \"Expression\" type is a deterministic 63-bit hash "
          "of the expression (Wolfram documents its value as implementation-dependent). Formats Integer, "
          "DecimalString, HexString, HexStringLittleEndian, Base36String, Base64Encoding, ByteArray. MD2 and "
          "the BLAKE2 parallel/XOF variants raise.")
def _hash(x, t="Expression", fmt="Integer"):
    if t == "Expression":
        return _hash_format(_expression_hash(x), 64, fmt, "Hash")
    f, bits = _hash_function(t, "Hash")
    return _hash_format(f(_hash_bytes(x, "Hash")), bits, fmt, "Hash")


def _file_path(f, fname):
    if isinstance(f, Expr) and f.head == "File" and len(f.args) == 1:
        f = f.args[0]
    if not isinstance(f, str):
        if isinstance(f, Symbol):
            raise Unevaluated()
        raise ValueError(f"{fname}: {f!r} is not a file name")
    p = os.path.expanduser(f)
    if not os.path.isfile(p):
        raise ValueError(f"{fname}: cannot open {f}")
    return p


def _byte_range(data, r, fname):
    if is_sym(r, "All"):
        return data
    if isinstance(r, int) and not isinstance(r, bool):
        if r == 0:
            return b""
        return data[:r] if r > 0 else data[r:]
    if is_list(r) and len(r.args) == 1 and isinstance(r.args[0], int):
        k = r.args[0]
        if not 1 <= abs(k) <= len(data):
            raise ValueError(f"{fname}: byte {k} is out of range")
        i = k - 1 if k > 0 else len(data) + k
        return data[i:i + 1]
    if is_list(r) and len(r.args) == 2 and all(isinstance(v, int) for v in r.args):
        m, n = r.args
        n_ = len(data)
        m0 = m - 1 if m > 0 else n_ + m
        n0 = n if n > 0 else n_ + n + 1
        if not (0 <= m0 <= n_ and 0 <= n0 <= n_):
            raise ValueError(f"{fname}: byte range {m};;{n} is out of range")
        return data[m0:n0]
    raise ValueError(f"{fname}: invalid byte range {r!r}")


_HASH_FORMATS = {"Integer", "DecimalString", "HexString", "HexStringLittleEndian", "Base36String",
                 "Base64Encoding", "ByteArray"}


@fn("FileHash", area=AREA, status="added",
    notes="hash of the raw bytes of a file (default MD5, format Integer); FileHash[{file, range}, ...] with "
          "n, -n, {n}, {m, n}, 0 or All byte ranges; the legacy FileHash[file, type, range, format] is accepted")
def _filehash(f, t="MD5", *rest):
    rng = Symbol("All")
    if is_list(f) and len(f.args) == 2:
        f, rng = f.args
    rest = list(rest)
    fmt = "Integer"
    if len(rest) == 1:
        if isinstance(rest[0], str) and rest[0] in _HASH_FORMATS:
            fmt = rest[0]
        else:
            rng = rest[0]
    elif len(rest) == 2:
        rng, fmt = rest
    elif len(rest) > 2:
        raise ValueError("FileHash called with too many arguments")
    p = _file_path(f, "FileHash")
    with open(p, "rb") as fh:
        data = fh.read()
    data = _byte_range(data, rng, "FileHash")
    if t == "Expression":
        raise ValueError("FileHash: the \"Expression\" hash type applies to expressions, not files")
    fun, bits = _hash_function(t, "FileHash")
    return _hash_format(fun(data), bits, fmt, "FileHash")


# ═════════════════════════════════════════════════════════════════ text statistics ══
def _bio_string(x):
    if isinstance(x, Expr) and x.head == "BioSequence" and len(x.args) >= 2 and isinstance(x.args[1], str):
        return x.args[1]
    return None


def _text_arg(x, fname):
    b = _bio_string(x)
    if b is not None:
        return b
    return _str(x, fname)


def _ranked(counts_in_order):
    """Wolfram's order for *Counts: by decreasing count, ties in reverse order of first appearance."""
    items = list(reversed(counts_in_order))
    items.sort(key=lambda kv: -kv[1])
    return _assoc(items)


def _count(keys, keyf=lambda k: k):
    order, counts = [], {}
    for k in keys:
        t = keyf(k)
        if t not in counts:
            counts[t] = [k, 0]
            order.append(t)
        counts[t][1] += 1
    return [(counts[t][0], counts[t][1]) for t in order]


def _thread_strings(x, f):
    if is_list(x):
        return L(*(f(e) for e in x.args))
    return f(x)


def _ngram_n(n, fname):
    return _int(n, fname, "n-gram length", 1)


@fn("CharacterCounts", area=AREA, status="added",
    notes="association of character (or n-character) counts, by decreasing count with ties in reverse order of "
          "first appearance; IgnoreCase; threads over lists of strings; BioSequence letters")
def _charcounts(s, *rest):
    args, o = _opts(rest, {"IgnoreCase"}, "CharacterCounts")
    ic = _bool_opt(o, "IgnoreCase", "CharacterCounts")
    if len(args) > 1:
        raise ValueError("CharacterCounts called with too many arguments")
    n = _ngram_n(args[0], "CharacterCounts") if args else 1

    def one(x):
        t = _text_arg(x, "CharacterCounts")
        if ic:
            t = t.lower()
        grams = [t[i:i + n] for i in range(len(t) - n + 1)]
        return _ranked(_count(grams))
    return _thread_strings(s, one)


def _is_letter(c):
    return c.isalpha()


@fn("LetterCounts", area=AREA, status="added",
    notes="association of letter (LetterQ) or letter n-gram counts, optionally allowing extra characters; "
          "Wolfram's *Counts ordering; IgnoreCase; threads over lists")
def _lettercounts(s, *rest):
    args, o = _opts(rest, {"IgnoreCase"}, "LetterCounts")
    ic = _bool_opt(o, "IgnoreCase", "LetterCounts")
    if len(args) > 2:
        raise ValueError("LetterCounts called with too many arguments")
    n = _ngram_n(args[0], "LetterCounts") if args else 1
    extra = set()
    if len(args) == 2:
        if not is_list(args[1]) or not all(isinstance(c, str) and len(c) == 1 for c in args[1].args):
            raise ValueError("LetterCounts: the third argument must be a list of single characters")
        extra = set(args[1].args)

    def one(x):
        t = _text_arg(x, "LetterCounts")
        if ic:
            t = t.lower()
        ok = [(_is_letter(c) or c in extra) for c in t]
        grams = [t[i:i + n] for i in range(len(t) - n + 1) if all(ok[i:i + n])]
        return _ranked(_count(grams))
    return _thread_strings(s, one)


_WORD_RX = re.compile(r"[^\W_]+(?:['\u2019\-][^\W_]+)*")


def _word_spans(t):
    return [(m.start(), m.end()) for m in _WORD_RX.finditer(t)]


@fn("WordCount", area=AREA, status="added",
    notes="number of words, identified as by TextWords (hyphenated and apostrophe words count once)")
def _wordcount(s):
    return len(_word_spans(_str(s, "WordCount")))


@fn("WordCounts", area=AREA, status="added",
    notes="association of word counts (words as in TextWords); WordCounts[s, n] counts n-grams of consecutive "
          "words separated only by whitespace (keys are lists of words); Wolfram's *Counts ordering; IgnoreCase")
def _wordcounts(s, *rest):
    args, o = _opts(rest, {"IgnoreCase"}, "WordCounts")
    ic = _bool_opt(o, "IgnoreCase", "WordCounts")
    if len(args) > 1:
        raise ValueError("WordCounts called with too many arguments")
    n = _ngram_n(args[0], "WordCounts") if args else 1

    def one(x):
        t = _str(x, "WordCounts")
        if ic:
            t = t.lower()
        spans = _word_spans(t)
        words = [t[a:b] for a, b in spans]
        if n == 1:
            return _ranked(_count(words))
        grams = []
        for i in range(len(spans) - n + 1):
            if all(t[spans[k][1]:spans[k + 1][0]].isspace() for k in range(i, i + n - 1)):
                grams.append(tuple(words[i:i + n]))
        return _ranked([(L(*g), c) for g, c in _count(grams)])
    return _thread_strings(s, one)


# ── Porter stemmer (M. F. Porter, 1980, with the reference implementation's departures) ──
_STEP2 = [("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"), ("izer", "ize"),
          ("bli", "ble"), ("alli", "al"), ("entli", "ent"), ("eli", "e"), ("ousli", "ous"), ("ization", "ize"),
          ("ation", "ate"), ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
          ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log")]
_STEP3 = [("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"), ("ful", ""),
          ("ness", "")]
_STEP4 = ["al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent", "ion", "ou", "ism",
          "ate", "iti", "ous", "ive", "ize"]


def _cons(w, i):
    c = w[i]
    if c in "aeiou":
        return False
    if c == "y":
        return i == 0 or not _cons(w, i - 1)
    return True


def _measure(w):
    n, i, L_ = 0, 0, len(w)
    while i < L_ and _cons(w, i):
        i += 1
    while i < L_:
        while i < L_ and not _cons(w, i):
            i += 1
        if i >= L_:
            break
        while i < L_ and _cons(w, i):
            i += 1
        n += 1
    return n


def _vowel_in(w):
    return any(not _cons(w, i) for i in range(len(w)))


def _doublec(w):
    return len(w) >= 2 and w[-1] == w[-2] and _cons(w, len(w) - 1)


def _cvc(w):
    i = len(w) - 1
    return i >= 2 and _cons(w, i) and not _cons(w, i - 1) and _cons(w, i - 2) and w[i] not in "wxy"


def porter_stem(w):
    if len(w) <= 2:
        return w
    # step 1a
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    # step 1b
    if w.endswith("eed"):
        if _measure(w[:-3]) > 0:
            w = w[:-1]
    else:
        for suf in ("ed", "ing"):
            if w.endswith(suf) and _vowel_in(w[:-len(suf)]):
                w = w[:-len(suf)]
                if w.endswith("at") or w.endswith("bl") or w.endswith("iz"):
                    w += "e"
                elif _doublec(w) and w[-1] not in "lsz":
                    w = w[:-1]
                elif _measure(w) == 1 and _cvc(w):
                    w += "e"
                break
    # step 1c
    if w.endswith("y") and _vowel_in(w[:-1]):
        w = w[:-1] + "i"
    # steps 2 and 3 (first matching suffix only)
    for table in (_STEP2, _STEP3):
        for suf, rep in table:
            if w.endswith(suf):
                if _measure(w[:-len(suf)]) > 0:
                    w = w[:-len(suf)] + rep
                break
    # step 4
    for suf in _STEP4:
        if w.endswith(suf):
            stem = w[:-len(suf)]
            if suf == "ion":
                if not (stem and stem[-1] in "st"):
                    continue
            if _measure(stem) > 1:
                w = stem
            break
    # step 5
    if w.endswith("e"):
        a = _measure(w[:-1])
        if a > 1 or (a == 1 and not _cvc(w[:-1])):
            w = w[:-1]
    if w.endswith("l") and _doublec(w) and _measure(w) > 1:
        w = w[:-1]
    return w


def _restore_case(orig, stem):
    out = []
    for i, c in enumerate(stem):
        if i < len(orig) and orig[i].lower() == c:
            out.append(orig[i])
        elif i < len(orig) and orig[i].isupper():
            out.append(c.upper())
        else:
            out.append(c.upper() if orig.isupper() else c)
    return "".join(out)


def _stem_word(word):
    m = re.fullmatch(r"([^A-Za-z]*)([A-Za-z]+(?:[^A-Za-z]+[A-Za-z]+)*)([^A-Za-z]*)", word)
    if not m:
        return word
    pre, core, post = m.groups()
    if not core.isalpha():
        return word
    return pre + _restore_case(core, porter_stem(core.lower())) + post


@fn("WordStem", area=AREA, status="added", listable=True,
    notes="Porter stemming algorithm (reference implementation: bli->ble, logi->log), preserving the word's "
          "case and surrounding punctuation; listable")
def _wordstem(w):
    w = _str(w, "WordStem")
    if not w:
        return w
    return " ".join(_stem_word(p) for p in w.split(" ")) if " " in w else _stem_word(w)


# ── character names ──
_GREEK = ["Alpha", "Beta", "Gamma", "Delta", None, "Zeta", "Eta", "Theta", "Iota", "Kappa", "Lambda", "Mu", "Nu",
          "Xi", "Omicron", "Pi", "Rho", "FinalSigma", "Sigma", "Tau", "Upsilon", None, "Chi", "Psi", "Omega"]
_NAMED = {}
for _i, _n in enumerate(_GREEK):
    if _n is None:
        continue
    _NAMED[chr(0x3B1 + _i)] = _n
    if _n != "FinalSigma":
        _NAMED[chr(0x391 + _i)] = "Capital" + _n
_NAMED.update({
    "\u03d1": "CurlyTheta", "\u03f0": "CurlyKappa", "\u03f1": "CurlyRho", "\u03d6": "CurlyPi",
    "\u03d2": "CurlyCapitalUpsilon", "\u03dd": "Digamma", "\u03dc": "CapitalDigamma", "\u03df": "Koppa",
    "\u03de": "CapitalKoppa", "\u03db": "Stigma", "\u03da": "CapitalStigma", "\u03e1": "Sampi",
    "\u03e0": "CapitalSampi",
    "\u221e": "Infinity", "\u00b0": "Degree", "\u2208": "Element", "\u2209": "NotElement", "\u2200": "ForAll",
    "\u2203": "Exists", "\u2202": "PartialD", "\u2207": "Del", "\u2211": "Sum", "\u220f": "Product",
    "\u222b": "Integral", "\u00d7": "Times", "\u00f7": "Divide", "\u00b1": "PlusMinus", "\u2213": "MinusPlus",
    "\u2264": "LessEqual", "\u2265": "GreaterEqual", "\u2260": "NotEqual", "\u2192": "RightArrow",
    "\u2190": "LeftArrow", "\u2135": "Aleph", "\u210f": "HBar", "\u20ac": "Euro", "\u00a9": "Copyright",
    "\u2022": "Bullet", "\u2020": "Dagger", "\u2021": "DoubleDagger", "\u2026": "Ellipsis", "\u00a7": "Section",
    "\u00b7": "CenterDot", "\u00b5": "Micro", "\u212b": "Angstrom", "\u00a2": "Cent", "\u00a3": "Sterling",
    "\u00a5": "Yen", "\u00df": "SZ", "\u00fe": "Thorn", "\u00de": "CapitalThorn", "\u00f0": "Eth",
    "\u00d0": "CapitalEth", "\u0131": "DotlessI", "\u00f8": "OSlash", "\u00d8": "CapitalOSlash",
    "\u0142": "LSlash", "\u0141": "CapitalLSlash",
})
_MARKS = {"ACUTE": "Acute", "GRAVE": "Grave", "CIRCUMFLEX": "Hat", "DIAERESIS": "DoubleDot", "TILDE": "Tilde",
          "RING ABOVE": "Ring", "CEDILLA": "Cedilla", "CARON": "Hacek", "BREVE": "Cup", "MACRON": "Bar",
          "DOUBLE ACUTE": "DoubleAcute"}
for _cp in range(0xC0, 0x180):
    _c = chr(_cp)
    try:
        _u = unicodedata.name(_c)
    except ValueError:
        continue
    _m = re.fullmatch(r"LATIN (SMALL|CAPITAL) LETTER ([A-Z]) WITH (.+)", _u)
    if _m and _m.group(3) in _MARKS and _c not in _NAMED:
        _NAMED[_c] = ("Capital" if _m.group(1) == "CAPITAL" else "") + _m.group(2) + _MARKS[_m.group(3)]


@fn("CharacterName", area=AREA, status="added", listable=True,
    notes="Wolfram standard names (\\[name]) for Greek letters, Latin letters with diacritics and common "
          "mathematical/typographic characters, and \"UnicodeName\" for any character; CharacterName[n] by code; "
          "a string gives a list; types without a reproducible table (LongName, CommonName, TeXName, HTMLName), "
          "and characters outside the table, raise. Greek epsilon/phi forms are omitted (Wolfram's naming of the "
          "lunate/curly variants differs from Unicode's)")
def _charname(c, typ=None):
    if isinstance(c, int) and not isinstance(c, bool):
        if not 0 <= c <= 0x10FFFF:
            raise ValueError("CharacterName: character code out of range")
        c = chr(c)
    c = _str(c, "CharacterName")
    if len(c) != 1:
        return L(*(_charname(ch, typ) for ch in c))
    if typ is None or typ == "StandardName":
        n = _NAMED.get(c)
        if n is not None:
            return n
        if typ == "StandardName":
            return _missing("NotAvailable")
        raise NotImplementedError(f"CharacterName: {c!r} has no Wolfram standard name in the table here, and the "
                                  "long canonical name form is not reproduced")
    if typ == "UnicodeName":
        try:
            return unicodedata.name(c)
        except ValueError:
            return _missing("NotAvailable")
    raise NotImplementedError(f"CharacterName: the name type {typ!r} is not supported")


@fn("CharacterNormalize", area=AREA, status="added",
    notes="Unicode normalization forms NFC, NFD, NFKC, NFKD; threads over lists of strings")
def _charnormalize(s, form):
    if form not in ("NFC", "NFD", "NFKC", "NFKD"):
        raise ValueError(f"CharacterNormalize: unknown normalization form {form!r}")
    return _thread_strings(s, lambda t: unicodedata.normalize(form, _str(t, "CharacterNormalize")))


# ── StringApply ──
def _join_inner(r):
    if isinstance(r, str):
        return r
    if is_list(r):
        if all(isinstance(e, str) for e in r.args):
            return "".join(r.args)
        if all(is_list(e) or isinstance(e, str) for e in r.args):
            return L(*(_join_inner(e) for e in r.args))
    raise ValueError(f"StringApply: the function must return a string or (nested) lists of strings, not {r!r}")


def _stringapply(f, s):
    if is_list(s):
        return L(*(_stringapply(f, x) for x in s.args))
    s = _str(s, "StringApply")
    return _join_inner(apply(f, L(*s)))


@fn("StringApply", area=AREA, status="added",
    notes="applies f to Characters[s] and joins the result (nested lists of strings keep their structure, "
          "joined at the innermost level); StringApply[f] operator form; threads over lists")
def _stringapply2(f, s):
    return _stringapply(f, s)


@fn("StringApply", area=AREA, status="added")
def _stringapply1(f):
    raise Unevaluated()


def _stringapply_hook(head, args, ev_):
    if len(head.args) == 1 and len(args) == 1:
        return _stringapply(head.args[0], args[0])
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["StringApply"] = _stringapply_hook


# ── syntax ──
def _syntax_state(s):
    """'ok', 'incomplete' or 'error' for a string of Wolfram input."""
    from ..parser import parse
    if not s.strip():
        return "incomplete"
    try:
        parse(s)
        return "ok"
    except Exception as e:  # noqa: BLE001 - the parser signals every failure with an exception
        msg = str(e)
        if "EOF" in msg or "end of input" in msg:
            return "incomplete"
        return "error"


@fn("SyntaxQ", area=AREA, status="added",
    notes="True if the string parses as one Wolfram Language expression (InputForm); other forms raise")
def _syntaxq(s, form=None):
    s = _str(s, "SyntaxQ")
    if form is not None and not is_sym(form, "InputForm"):
        raise NotImplementedError("SyntaxQ: only InputForm syntax is supported")
    return _syntax_state(s) == "ok"


@fn("SyntaxLength", area=AREA, status="added",
    notes="length of the longest prefix that is syntactically valid so far (the position of the first syntax "
          "error); StringLength + 1 when the whole string is valid but incomplete")
def _syntaxlength(s):
    s = _str(s, "SyntaxLength")
    st = _syntax_state(s)
    if st == "ok":
        return len(s)
    if st == "incomplete":
        return len(s) + 1
    for k in range(len(s) - 1, -1, -1):
        if _syntax_state(s[:k]) != "error":
            return k
    return 0
