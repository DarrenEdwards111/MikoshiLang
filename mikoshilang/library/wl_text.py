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


# ═════════════════════════════════════════════════════════════════ sequences & alignment ══
def _seq(x, fname):
    """(elements, kind) of a string ("str"), BioSequence ("str") or list ("list")."""
    b = _bio_string(x)
    if b is not None:
        return list(b), "str"
    if isinstance(x, str):
        return list(x), "str"
    if is_list(x):
        return list(x.args), "list"
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not a string, list or BioSequence")


def _seq_pair(a, b, fname, ignore_case=False):
    x, ka = _seq(a, fname)
    y, kb = _seq(b, fname)
    if ka != kb:
        raise ValueError(f"{fname}: both arguments must be strings (or BioSequences) or both lists")
    if ignore_case and ka == "str":
        x = [c.lower() for c in x]
        y = [c.lower() for c in y]
    return x, y, ka


def _key(e):
    return e if isinstance(e, str) else ("\0tag", K.tag(e))


def _wrap(items, kind):
    return "".join(items) if kind == "str" else L(*items)


class _Scorer:
    """Similarity scores from SimilarityRules (Automatic: +1 match, -1 mismatch/insertion/deletion)."""

    def __init__(self, rules, fname):
        self.rules = []
        self.cache = {}
        if rules is None or is_sym(rules, "Automatic"):
            return
        if isinstance(rules, str):
            raise NotImplementedError(f"{fname}: named similarity matrices ({rules!r}) are not supported")
        if not is_list(rules):
            raise ValueError(f"{fname}: SimilarityRules must be Automatic or a list of rules {{e1, e2}} -> v")
        for r in rules.args:
            if not (is_rule(r) and is_list(r.args[0]) and len(r.args[0].args) == 2):
                raise ValueError(f"{fname}: invalid similarity rule {r!r}")
            v = r.args[1]
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                try:
                    v = float(ev(Expr("N", v)))
                except Exception:  # noqa: BLE001
                    raise ValueError(f"{fname}: the similarity value in {r!r} must be a real number")
            self.rules.append((r.args[0], v))

    def __call__(self, x, y):
        k = (_key(x), _key(y))
        v = self.cache.get(k)
        if v is None:
            v = self._compute(x, y)
            self.cache[k] = v
        return v

    def _compute(self, x, y):
        pair = L(x, y)
        for lhs, v in self.rules:
            a, b = lhs.args
            if _plain(a) and _plain(b):
                if _key(a) == _key(x) and _key(b) == _key(y):
                    return v
            elif ev(Expr("MatchQ", pair, lhs)) is True:
                return v
        if isinstance(x, str) and x == "" or isinstance(y, str) and y == "":
            return -1
        return 1 if _key(x) == _key(y) else -1


def _plain(e):
    if isinstance(e, Expr):
        if e.head in ("Blank", "BlankSequence", "BlankNullSequence", "Pattern", "Alternatives", "Condition",
                      "PatternTest", "Except", "Optional"):
            return False
        return all(_plain(a) for a in e.args)
    return True


_NEG = float("-inf")


def _align(x, y, score, gap, local=False):
    """Gotoh alignment maximising the similarity; returns (score, ops) with ops ('S'|'D'|'I', i, j) 0-based."""
    n, m = len(x), len(y)
    dl = [score(c, "") for c in x]
    ins = [score("", c) for c in y]
    sub = [[score(x[i], y[j]) for j in range(m)] for i in range(n)]
    M = [[_NEG] * (m + 1) for _ in range(n + 1)]
    X = [[_NEG] * (m + 1) for _ in range(n + 1)]
    Y = [[_NEG] * (m + 1) for _ in range(n + 1)]
    M[0][0] = 0
    if not local:
        acc = -gap
        for i in range(1, n + 1):
            acc += dl[i - 1]
            X[i][0] = acc
        acc = -gap
        for j in range(1, m + 1):
            acc += ins[j - 1]
            Y[0][j] = acc
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1])
            if local:
                best = max(best, 0)
            M[i][j] = best + sub[i - 1][j - 1]
            X[i][j] = max(M[i - 1][j] - gap, X[i - 1][j], Y[i - 1][j] - gap) + dl[i - 1]
            Y[i][j] = max(M[i][j - 1] - gap, Y[i][j - 1], X[i][j - 1] - gap) + ins[j - 1]
    tables = {"M": M, "X": X, "Y": Y}
    if local:
        best, bi, bj = 0, 0, 0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if M[i][j] > best:
                    best, bi, bj = M[i][j], i, j
        if best <= 0:
            return 0, [], (0, 0, 0, 0)
        i, j, st = bi, bj, "M"
    else:
        cands = [(M[n][m], "M"), (X[n][m], "X"), (Y[n][m], "Y")] if (n or m) else [(0, "M")]
        best = max(c[0] for c in cands)
        st = next(s for v, s in cands if v == best)
        i, j = n, m
    ops = []
    eps = 1e-9
    while i > 0 or j > 0:
        if st == "M":
            if local and abs(M[i][j] - sub[i - 1][j - 1]) < eps and max(M[i - 1][j - 1], X[i - 1][j - 1],
                                                                        Y[i - 1][j - 1]) <= 0:
                ops.append(("S", i - 1, j - 1))
                i, j = i - 1, j - 1
                break
            ops.append(("S", i - 1, j - 1))
            target = M[i][j] - sub[i - 1][j - 1]
            i, j = i - 1, j - 1
            if i == 0 and j == 0:
                break
            st = next(s for s in ("M", "X", "Y") if abs(tables[s][i][j] - target) < eps)
        elif st == "X":
            ops.append(("D", i - 1, j))
            v = X[i][j] - dl[i - 1]
            i -= 1
            if i == 0 and j == 0 or (not local and j == 0):
                st = "X"
                if i == 0:
                    break
                continue
            opts_ = [("M", M[i][j] - gap), ("X", X[i][j]), ("Y", Y[i][j] - gap)]
            st = next(s for s, w in opts_ if abs(w - v) < eps)
        else:
            ops.append(("I", i, j - 1))
            v = Y[i][j] - ins[j - 1]
            j -= 1
            if i == 0 and j == 0 or (not local and i == 0):
                st = "Y"
                if j == 0:
                    break
                continue
            opts_ = [("M", M[i][j] - gap), ("Y", Y[i][j]), ("X", X[i][j] - gap)]
            st = next(s for s, w in opts_ if abs(w - v) < eps)
    ops.reverse()
    if local:
        i0 = min((o[1] for o in ops if o[0] != "I"), default=0)
        j0 = min((o[2] for o in ops if o[0] != "D"), default=0)
        return best, ops, (i0, bi, j0, bj)
    return best, ops, (0, n, 0, m)


def _alignment_opts(rest, fname, extra=()):
    allowed = {"GapPenalty", "IgnoreCase", "SimilarityRules"} | set(extra)
    args, o = _opts(rest, allowed, fname)
    if args:
        raise ValueError(f"{fname} called with too many arguments")
    gap = o.get("GapPenalty", 0)
    if isinstance(gap, bool) or not isinstance(gap, (int, float)):
        try:
            gap = float(ev(Expr("N", gap)))
        except Exception:  # noqa: BLE001
            raise ValueError(f"{fname}: GapPenalty must be a real number")
    return o, gap, _bool_opt(o, "IgnoreCase", fname), _Scorer(o.get("SimilarityRules"), fname)


@fn("NeedlemanWunschSimilarity", area=AREA, status="added",
    notes="score of an optimal global (Needleman-Wunsch/Gotoh) alignment as a machine real: +1 per match, -1 "
          "per mismatch/insertion/deletion (SimilarityRules overrides, patterns allowed), minus GapPenalty per "
          "gap run; IgnoreCase; strings, lists or BioSequences. Named matrices (BLOSUM62, ...) raise")
def _nws(a, b, *rest):
    o, gap, ic, sc = _alignment_opts(rest, "NeedlemanWunschSimilarity")
    x, y, _ = _seq_pair(a, b, "NeedlemanWunschSimilarity", ic)
    return float(_align(x, y, sc, gap)[0])


@fn("SmithWatermanSimilarity", area=AREA, status="added",
    notes="score of an optimal local (Smith-Waterman/Gotoh) alignment as a machine real (0. if nothing "
          "matches); options as NeedlemanWunschSimilarity")
def _sws(a, b, *rest):
    o, gap, ic, sc = _alignment_opts(rest, "SmithWatermanSimilarity")
    x, y, _ = _seq_pair(a, b, "SmithWatermanSimilarity", ic)
    return float(_align(x, y, sc, gap, local=True)[0])


def _segments(x, y, ops, kind, merge, span):
    """Wolfram's SequenceAlignment segments from alignment operations."""
    i0, i1, j0, j1 = span
    out = []          # items: ("C", list) common; ("D", kind, list_a, list_b)

    def add_diff(k, ua, ub):
        if out and out[-1][0] == "D" and (merge or out[-1][1] == k):
            out[-1][2].extend(ua)
            out[-1][3].extend(ub)
        else:
            out.append(["D", k, list(ua), list(ub)])
    if i0 > 0 or j0 > 0:
        add_diff("U", x[:i0], y[:j0])
    for op, i, j in ops:
        if op == "S":
            if _key(x[i]) == _key(y[j]):
                if out and out[-1][0] == "C":
                    out[-1][1].append(x[i])
                else:
                    out.append(["C", [x[i]]])
            else:
                add_diff("S", [x[i]], [y[j]])
        elif op == "D":
            add_diff("D", [x[i]], [])
        else:
            add_diff("I", [], [y[j]])
    if i1 < len(x) or j1 < len(y):
        add_diff("U2", x[i1:], y[j1:])
    res = []
    for seg in out:
        if seg[0] == "C":
            res.append(_wrap(seg[1], kind))
        else:
            res.append(L(_wrap(seg[2], kind), _wrap(seg[3], kind)))
    return L(*res)


@fn("SequenceAlignment", area=AREA, status="added",
    notes="optimal alignment as a list of common segments and {u1, u2} differences; Method \"Global\" "
          "(Needleman-Wunsch, default) or \"Local\" (Smith-Waterman, unaligned ends shown as differences); "
          "GapPenalty, IgnoreCase (lower-cases both strings), MergeDifferences, SimilarityRules. Among equally "
          "good alignments the choice follows Wolfram's documented examples; other methods and named "
          "similarity matrices raise")
def _seqalign(a, b, *rest):
    o, gap, ic, sc = _alignment_opts(rest, "SequenceAlignment", ("MergeDifferences", "Method"))
    merge = _bool_opt(o, "MergeDifferences", "SequenceAlignment", True)
    method = o.get("Method", "Global")
    if method not in ("Global", "Local"):
        raise NotImplementedError(f"SequenceAlignment: Method -> {method!r} is not supported")
    x, y, kind = _seq_pair(a, b, "SequenceAlignment", ic)
    _, ops, span = _align(x, y, sc, gap, local=(method == "Local"))
    if method == "Local" and not ops:
        return L(L(_wrap(x, kind), _wrap(y, kind))) if (x or y) else L()
    return _segments(x, y, ops, kind, merge, span)


@fn("DamerauLevenshteinDistance", area=AREA, status="added",
    notes="minimum number of insertions, deletions, substitutions and adjacent transpositions (unrestricted "
          "Damerau-Levenshtein distance, a metric); strings, lists or BioSequences; IgnoreCase")
def _dldist(a, b, *rest):
    args, o = _opts(rest, {"IgnoreCase"}, "DamerauLevenshteinDistance")
    if args:
        raise ValueError("DamerauLevenshteinDistance called with too many arguments")
    x, y, _ = _seq_pair(a, b, "DamerauLevenshteinDistance", _bool_opt(o, "IgnoreCase", "DamerauLevenshteinDistance"))
    x = [_key(c) for c in x]
    y = [_key(c) for c in y]
    n, m = len(x), len(y)
    big = n + m
    da = {}
    d = [[0] * (m + 2) for _ in range(n + 2)]
    d[0][0] = big
    for i in range(n + 1):
        d[i + 1][0] = big
        d[i + 1][1] = i
    for j in range(m + 1):
        d[0][j + 1] = big
        d[1][j + 1] = j
    for i in range(1, n + 1):
        db = 0
        for j in range(1, m + 1):
            k = da.get(y[j - 1], 0)
            l_ = db
            cost = 0 if x[i - 1] == y[j - 1] else 1
            if cost == 0:
                db = j
            d[i + 1][j + 1] = min(d[i][j] + cost, d[i + 1][j] + 1, d[i][j + 1] + 1,
                                  d[k][l_] + (i - k - 1) + 1 + (j - l_ - 1))
        da[x[i - 1]] = i
    return d[n + 1][m + 1]


def _runs(ps):
    out = []
    for p in ps:
        if out and out[-1][1] == p - 1:
            out[-1][1] = p
        else:
            out.append([p, p])
    return L(*(L(*r) for r in out))


@fn("LongestCommonSequencePositions", area=AREA, status="added",
    notes="positions {pos1, pos2} of a longest common (not necessarily contiguous) sequence, each a list of "
          "{start, end} runs; {} if there is none; choice among equally long sequences as in Wolfram's "
          "examples; IgnoreCase")
def _lcseqpos(a, b, *rest):
    args, o = _opts(rest, {"IgnoreCase"}, "LongestCommonSequencePositions")
    if args:
        raise ValueError("LongestCommonSequencePositions called with too many arguments")
    x, y, _ = _seq_pair(a, b, "LongestCommonSequencePositions",
                        _bool_opt(o, "IgnoreCase", "LongestCommonSequencePositions"))
    x = [_key(c) for c in x]
    y = [_key(c) for c in y]
    n, m = len(x), len(y)
    P = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            P[i][j] = P[i - 1][j - 1] + 1 if x[i - 1] == y[j - 1] else max(P[i - 1][j], P[i][j - 1])
    i, j, pa, pb = n, m, [], []
    while i > 0 and j > 0:
        if x[i - 1] == y[j - 1] and P[i][j] == P[i - 1][j - 1] + 1:
            pa.append(i)
            pb.append(j)
            i, j = i - 1, j - 1
        elif P[i - 1][j] >= P[i][j - 1]:
            i -= 1
        else:
            j -= 1
    if not pa:
        return L()
    return L(_runs(pa[::-1]), _runs(pb[::-1]))


@fn("LongestCommonSubsequencePositions", area=AREA, status="added",
    notes="positions {{s1, e1}, {s2, e2}} of a longest contiguous common run (the earliest in s1); {} if none; "
          "IgnoreCase")
def _lcsubpos(a, b, *rest):
    args, o = _opts(rest, {"IgnoreCase"}, "LongestCommonSubsequencePositions")
    if args:
        raise ValueError("LongestCommonSubsequencePositions called with too many arguments")
    x, y, _ = _seq_pair(a, b, "LongestCommonSubsequencePositions",
                        _bool_opt(o, "IgnoreCase", "LongestCommonSubsequencePositions"))
    x = [_key(c) for c in x]
    y = [_key(c) for c in y]
    best, bi, bj = 0, 0, 0
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best or (cur[j] == best and i - cur[j] < bi - best):
                    best, bi, bj = cur[j], i, j
        prev = cur
    if best == 0:
        return L()
    return L(L(bi - best + 1, bi), L(bj - best + 1, bj))


def _order_le(a, b):
    """Wolfram canonical order a <= b (OrderedQ[{a, b}])."""
    return ev(Expr("OrderedQ", L(a, b))) is True


@fn("LongestOrderedSequence", area=AREA, status="added",
    notes="longest subsequence whose successive elements satisfy the ordering function (default: canonical "
          "order for lists, character codes for strings), by patience sorting; IgnoreCase lower-cases strings")
def _longestordered(s, *rest):
    args, o = _opts(rest, {"IgnoreCase"}, "LongestOrderedSequence")
    if len(args) > 1:
        raise ValueError("LongestOrderedSequence called with too many arguments")
    seq, kind = _seq(s, "LongestOrderedSequence")
    if kind == "str" and _bool_opt(o, "IgnoreCase", "LongestOrderedSequence"):
        seq = [c.lower() for c in seq]
    if args:
        p = args[0]
        le = lambda u, v: apply(p, u, v) is True  # noqa: E731
    elif kind == "str":
        le = lambda u, v: ord(u) <= ord(v)  # noqa: E731
    else:
        le = _order_le
    tails, prev = [], [None] * len(seq)
    for i, e in enumerate(seq):
        lo, hi = 0, len(tails)
        while lo < hi:                      # number of piles whose tail t satisfies le(t, e)
            mid = (lo + hi) // 2
            if le(seq[tails[mid]], e):
                lo = mid + 1
            else:
                hi = mid
        if lo > 0:
            prev[i] = tails[lo - 1]
        if lo == len(tails):
            tails.append(i)
        else:
            tails[lo] = i
    out = []
    i = tails[-1] if tails else None
    while i is not None:
        out.append(seq[i])
        i = prev[i]
    return _wrap(out[::-1], kind)


# ── repeats ──
def _same_fn(test):
    if test is None or is_sym(test, "SameQ") or is_sym(test, "Automatic"):
        return lambda u, v: K.tag(u) == K.tag(v) if not (isinstance(u, str) and isinstance(v, str)) else u == v
    return lambda u, v: apply(test, u, v) is True


def _period(items, same, n_min):
    """Smallest p such that items[i] ~ items[i mod p] for all i and len(items) // p >= n_min; None if none."""
    N = len(items)
    for p in range(1, N + 1):
        if N // p < n_min:
            return None
        if all(same(items[i % p], items[i]) for i in range(p, N)):
            return p
    return None


def _dims(x):
    d = []
    while is_list(x):
        d.append(len(x.args))
        if not x.args:
            break
        x = x.args[0]
    return d


def _subarray_level(x, axis):
    """Slices of an array along axis (0-based): list of sub-expressions indexed by that axis."""
    if axis == 0:
        return list(x.args)
    rows = [_subarray_level(r, axis - 1) for r in x.args]
    return [L(*(row[k] for row in rows)) for k in range(len(rows[0]))]


def _take_axis(x, axis, p):
    if axis == 0:
        return L(*x.args[:p])
    return L(*(_take_axis(r, axis - 1, p) for r in x.args))


@fn("FindRepeat", area=AREA, status="added",
    notes="minimal block that repeats (with an incomplete final repeat allowed) to give a list, array, string or "
          "association; FindRepeat[x, n] needs n complete repeats ({} or \"\" if none); FindRepeat[x, {n1, ...}] "
          "per array level; SameTest")
def _findrepeat(x, *rest):
    args, o = _opts(rest, {"SameTest"}, "FindRepeat")
    if len(args) > 1:
        raise ValueError("FindRepeat called with too many arguments")
    same = _same_fn(o.get("SameTest"))
    spec = args[0] if args else None
    if isinstance(x, str):
        n = 1 if spec is None else _int(spec, "FindRepeat", "repeat count", 1)
        p = _period(list(x), same, n)
        return x[:p] if p else ""
    if is_assoc(x):
        vals = KL.assoc_values(x, ev)
        n = 1 if spec is None else _int(spec, "FindRepeat", "repeat count", 1)
        p = _period(vals, same, n)
        return Expr("Association", *x.args[:p]) if p else Expr("Association")
    if not is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("FindRepeat: the first argument must be a list, array, string or association")
    dims = _dims(x)
    if spec is None:
        depth = 1
        while depth < len(dims) and _is_array(x, depth + 1):
            depth += 1
        counts = [1] * depth
    elif is_list(spec):
        counts = [_int(c, "FindRepeat", "repeat count", 1) for c in spec.args]
        if len(counts) > len(dims):
            raise ValueError("FindRepeat: the repeat specification is deeper than the array")
    else:
        counts = [_int(spec, "FindRepeat", "repeat count", 1)]
    k = len(counts)
    if k > 1 and not _is_array(x, k):
        raise ValueError("FindRepeat: the list is not a rectangular array to the requested depth")
    cur = x
    for axis, n in enumerate(counts):
        slices = _subarray_level(cur, axis)
        # a slice along any axis holds the level-k elements k-1 list levels down; SameTest compares those
        p = _period(slices, lambda u, v: _deep_same(u, v, same, k - 1), n)
        if p is None:
            return L()
        cur = _take_axis(cur, axis, p)
    return cur


def _is_array(x, depth):
    d = _dims(x)
    if len(d) < depth:
        return False

    def check(e, lvl):
        if lvl == depth:
            return True
        return is_list(e) and len(e.args) == d[lvl] and all(check(a, lvl + 1) for a in e.args)
    return check(x, 0)


def _deep_same(u, v, same, depth):
    if depth <= 0:
        return same(u, v)
    if not (is_list(u) and is_list(v)) or len(u.args) != len(v.args):
        return False
    return all(_deep_same(a, b, same, depth - 1) for a, b in zip(u.args, v.args))


@fn("FindTransientRepeat", area=AREA, status="added",
    notes="{transient, repeat} with the shortest transient after which a minimal block repeats at least n "
          "complete times (an incomplete final repeat allowed); {list, {}} if none; strings give strings; "
          "SameTest")
def _findtransient(x, n, *rest):
    args, o = _opts(rest, {"SameTest"}, "FindTransientRepeat")
    if args:
        raise ValueError("FindTransientRepeat called with too many arguments")
    n = _int(n, "FindTransientRepeat", "repeat count", 1)
    same = _same_fn(o.get("SameTest"))
    if isinstance(x, str):
        items, kind = list(x), "str"
    elif is_list(x):
        items, kind = list(x.args), "list"
    elif isinstance(x, Symbol):
        raise Unevaluated()
    else:
        raise ValueError("FindTransientRepeat: the first argument must be a list or a string")
    for t in range(len(items) + 1):
        rest_ = items[t:]
        if not rest_:
            break
        p = _period(rest_, same, n)
        if p is not None:
            return L(_wrap(items[:t], kind), _wrap(rest_[:p], kind))
    return L(_wrap(items, kind), _wrap([], kind))


# ═════════════════════════════════════════════════════════════════ templates ══
_TEMPLATE_HEADS = ("TemplateSlot", "TemplateExpression", "TemplateIf", "TemplateSequence", "TemplateWith",
                   "TemplateObject")
_hold("TemplateIf", "TemplateWith")
_hold_first("TemplateExpression")


class _TCtx:
    def __init__(self, positional, named, string_mode=False):
        self.positional = positional      # list of expressions
        self.named = named                # list of associations / dicts, searched in order
        self.string_mode = string_mode

    def lookup(self, key):
        if isinstance(key, int) and not isinstance(key, bool):
            if 1 <= key <= len(self.positional):
                return True, self.positional[key - 1]
            return False, None
        for a in self.named:
            if isinstance(a, dict):
                if key in a:
                    return True, a[key]
                continue
            for r in a.args:
                if is_rule(r) and K.tag(r.args[0]) == K.tag(key):
                    return True, r.args[1]
        return False, None


def _t_opts(args, fname, allowed=("InsertionFunction", "CombinerFunction", "DefaultValue", "Delimiters",
                                   "\0slot", "\0index")):
    pos, opts = [], {}
    for a in args:
        if is_rule(a) and (isinstance(a.args[0], str) or isinstance(a.args[0], Symbol)):
            k = a.args[0] if isinstance(a.args[0], str) else a.args[0].name
            if k in allowed:
                opts[k] = a.args[1]
                continue
        pos.append(a)
    return pos, opts


def _insert(ins, v):
    if ins is None or is_sym(ins, "Identity") or is_sym(ins, "Inherited") or is_sym(ins, "Automatic"):
        return v
    if isinstance(ins, str):
        name = ins.lower()
        if name in ("htmlfragment", "html"):
            v = ev(v)
            if isinstance(v, str) or not isinstance(v, Expr):
                return _html.escape(v if isinstance(v, str) else _text_string(v), quote=False)
            return _html.escape(_text_string(v), quote=False)
        if name in ("text", "string"):
            return Expr("TextString", v)
        if name == "json":
            return _to_json(ev(v))
        raise NotImplementedError(f"InsertionFunction -> {ins!r} is not supported")
    return Expr(K.as_head(ins), v)


def _text_string(v):
    r = ev(Expr("TextString", v))
    return r if isinstance(r, str) else str(r)


def _slot_absent(key, ctx):
    return Expr("Missing", "TemplateSlotAbsent", key)


def _twalk(e, ctx, ins):
    """Substitute template constructs in e (not evaluated)."""
    if isinstance(e, Expr):
        h = e.head
        if h == "TemplateSlot" and e.args:
            pos, o = _t_opts(e.args[1:], "TemplateSlot")
            key = e.args[0]
            if is_sym(key, "Automatic") or isinstance(key, Expr):
                raise ValueError(f"TemplateSlot: invalid slot {key!r}")
            found, v = ctx.lookup(key)
            if not found:
                if "DefaultValue" in o:
                    return o["DefaultValue"]
                return _insert(o.get("InsertionFunction", ins), _slot_absent(key, ctx))
            return _insert(o.get("InsertionFunction", ins), v)
        if h == "TemplateExpression" and e.args:
            pos, o = _t_opts(e.args[1:], "TemplateExpression")
            return _insert(o.get("InsertionFunction", ins), _twalk(e.args[0], ctx, None))
        if h == "TemplateIf" and len(e.args) >= 2:
            pos, o = _t_opts(e.args, "TemplateIf")
            if len(pos) not in (2, 3):
                raise ValueError("TemplateIf takes a condition and one or two branches")
            c = ev(_twalk(pos[0], ctx, None))
            sub_ins = o.get("InsertionFunction", ins)
            if c is True:
                return _twalk(pos[1], ctx, sub_ins)
            if c is False:
                return _twalk(pos[2], ctx, sub_ins) if len(pos) == 3 else Expr("Sequence")
            raise ValueError(f"TemplateIf: the condition evaluated to {c!r}, not True or False")
        if h == "TemplateWith" and len(e.args) == 2:
            spec, body = e.args
            rules = spec.args if is_list(spec) else [spec]
            bind = {}
            for r in rules:
                if not is_rule(r):
                    raise ValueError(f"TemplateWith: {r!r} is not a rule")
                bind[r.args[0]] = ev(_twalk(r.args[1], ctx, None))
            assoc = Expr("Association", *(Expr("Rule", k, v) for k, v in bind.items()))
            return _twalk(body, _TCtx(ctx.positional, [assoc] + ctx.named, ctx.string_mode), ins)
        if h == "TemplateSequence" and e.args:
            pos, o = _t_opts(e.args, "TemplateSequence")
            if len(pos) != 2:
                raise ValueError("TemplateSequence takes a body and a list of values")
            body, vals = pos
            vals = ev(_twalk(vals, ctx, None))
            if isinstance(vals, int) and not isinstance(vals, bool):
                vals = L(*range(1, vals + 1))
            if not is_list(vals):
                raise ValueError(f"TemplateSequence: the values {vals!r} must be a list")
            sub_ins = o.get("InsertionFunction", ins)
            out = []
            for k, v in enumerate(vals.args):
                if k and "Delimiters" in o:
                    out.append(o["Delimiters"])
                p = list(v.args) if is_list(v) else [v]
                extra = {"SequenceIndex": k + 1}
                if "\0slot" in o:
                    extra[o["\0slot"]] = v
                if "\0index" in o:
                    extra[o["\0index"]] = k + 1
                named = [extra] + ([v] if is_assoc(v) else []) + ctx.named
                out.append(_twalk(body, _TCtx(p, named, ctx.string_mode), sub_ins))
            if not out and "DefaultValue" in o:
                return _twalk(o["DefaultValue"], ctx, sub_ins)
            return Expr("Sequence", *out)
        if h in ("StringTemplate", "XMLTemplate", "FileTemplate"):
            e2 = ev(e)
            if isinstance(e2, Expr) and e2.head == "TemplateObject":
                return _twalk(e2, ctx, ins)
            return e2
        if h == "TemplateObject" and e.args:
            return _template_apply(e, ctx, ins)
        head = _twalk(e.head, ctx, ins) if isinstance(e.head, Expr) else e.head
        return Expr(head, *(_twalk(a, ctx, ins) for a in e.args))
    return e


def _tobj_parts(t):
    """(body, bound data list, insertion, combiner) of a TemplateObject."""
    pos, o = _t_opts(t.args[1:], "TemplateObject")
    data = [p for p in pos if is_assoc(p)]
    if len(data) != len(pos):
        raise ValueError(f"TemplateObject: unexpected arguments {pos!r}")
    return t.args[0], data, o.get("InsertionFunction"), o.get("CombinerFunction")


def _template_apply(t, ctx, inherited=None):
    body, data, ins, comb = _tobj_parts(t)
    if ins is None or is_sym(ins, "Inherited"):
        ins = inherited
    string_mode = is_sym(comb, "StringJoin") or ctx.string_mode and comb is None
    inner = _TCtx(ctx.positional, ctx.named + data, string_mode)
    r = _twalk(body, inner, ins)
    if comb is not None and not is_sym(comb, "Identity") and not is_sym(comb, "Inherited"):
        r = Expr(K.as_head(comb), r)
    r = ev(r)
    if isinstance(r, str) and is_sym(comb, "StringJoin") and _XML_BODIES and K.tag(body) in _XML_BODIES:
        r = re.sub(r"\n[ \t]*(?=\n)", "", r)
    return r


def _as_template(t):
    if isinstance(t, str):
        return ev(Expr("StringTemplate", t))
    if isinstance(t, Expr) and t.head == "File":
        return ev(Expr("FileTemplate", t))
    if isinstance(t, Expr) and t.head == "TemplateObject":
        return t
    return Expr("TemplateObject", t)


def _template_ctx(args):
    if not args:
        return _TCtx([], [])
    if len(args) > 1:
        raise ValueError("TemplateApply: too many arguments")
    a = args[0]
    pos = list(a.args) if is_list(a) else [a]
    return _TCtx(pos, [p for p in pos if is_assoc(p)])


@fn("TemplateApply", area=AREA, status="added",
    notes="applies a template (TemplateObject, string, File or any expression containing TemplateSlot, "
          "TemplateExpression, TemplateIf, TemplateSequence, TemplateWith or nested TemplateObject) to "
          "positional arguments (a list, or a single non-list value) and named values from associations; "
          "the template's own bound association is searched first; :> values are inserted unevaluated; a "
          "missing slot without \"DefaultValue\" inserts Missing[\"TemplateSlotAbsent\", key] (\"\" in string "
          "templates)")
def _templateapply(t, *args):
    return _template_apply(_as_template(t), _template_ctx(args))


def _template_obj_hook(head, args, ev_):
    if head.head == "TemplateObject" and head.args:
        return _template_apply(head, _template_ctx([L(*args)] if len(args) != 1 else [args[0]]))
    return K.UNHANDLED


def _template_obj_hook_multi(head, args, ev_):
    if not head.args:
        return K.UNHANDLED
    ctx = _TCtx(list(args), [a for a in args if is_assoc(a)])
    return _template_apply(head, ctx)


KL.COMPOUND_HEAD_HOOKS["TemplateObject"] = _template_obj_hook_multi


def _parse_string_template(s, fname="StringTemplate"):
    from ..parser import parse
    pieces, buf, auto, i, n = [], [], 0, 0, len(s)

    def flush():
        if buf:
            pieces.append("".join(buf))
            buf.clear()
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n and s[i + 1] in "`<*\\":
            buf.append(s[i + 1])
            i += 2
            continue
        if c == "`":
            j = s.find("`", i + 1)
            if j < 0:
                raise ValueError(f"{fname}: unmatched ` in template string")
            name = s[i + 1:j]
            flush()
            if name == "":
                auto += 1
                pieces.append(Expr("TemplateSlot", auto))
            elif name.isdigit():
                pieces.append(Expr("TemplateSlot", int(name)))
            else:
                pieces.append(Expr("TemplateSlot", name))
            i = j + 1
            continue
        if s.startswith("<*", i):
            j = s.find("*>", i + 2)
            if j < 0:
                raise ValueError(f"{fname}: unmatched <* in template string")
            code = s[i + 2:j]
            try:
                ex = parse(code)
            except Exception as err:  # noqa: BLE001
                raise ValueError(f"{fname}: cannot parse template expression {code!r}: {err}")
            flush()
            pieces.append(Expr("TemplateExpression", _slots_to_template(ex)))
            i = j + 2
            continue
        buf.append(c)
        i += 1
    flush()
    return pieces


def _slots_to_template(e):
    if isinstance(e, Expr):
        if e.head == "Slot" and len(e.args) == 1:
            return Expr("TemplateSlot", e.args[0])
        if e.head == "Function":
            return e
        head = _slots_to_template(e.head) if isinstance(e.head, Expr) else e.head
        return Expr(head, *(_slots_to_template(a) for a in e.args))
    return e


def _template_source(src, fname):
    if isinstance(src, str):
        return src
    if isinstance(src, Expr) and src.head == "File" and src.args and isinstance(src.args[0], str):
        p = _find_template_file(src.args[0])
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    if isinstance(src, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: the template source must be a string or File[...]")


def _find_template_file(name):
    p = os.path.expanduser(name)
    if os.path.isabs(p) or os.path.exists(p):
        if not os.path.exists(p):
            raise ValueError(f"template file {name!r} not found")
        return p
    paths = ev(Symbol("$TemplatePath"))
    for d in (paths.args if is_list(paths) else []):
        if isinstance(d, str) and os.path.exists(os.path.join(d, p)):
            return os.path.join(d, p)
    raise ValueError(f"template file {name!r} not found (searched the working directory and $TemplatePath)")


@fn("StringTemplate", area=AREA, status="added",
    notes="TemplateObject[{pieces}, InsertionFunction -> TextString, CombinerFunction -> StringJoin] from a "
          "string or File: `` sequential slots, `n` positional, `name` named, <* expr *> template expressions "
          "with #name/#n slots; \\` escapes; an optional association of default values")
def _stringtemplate(src, *rest):
    pos, o = _t_opts(rest, "StringTemplate", ("InsertionFunction", "CombinerFunction"))
    if any(not is_assoc(p) for p in pos) or len(pos) > 1:
        raise ValueError("StringTemplate: the optional second argument must be an association")
    text = _template_source(src, "StringTemplate")
    return Expr("TemplateObject", L(*_parse_string_template(text)), *pos,
                Expr("Rule", Symbol("InsertionFunction"), o.get("InsertionFunction", Symbol("TextString"))),
                Expr("Rule", Symbol("CombinerFunction"), o.get("CombinerFunction", Symbol("StringJoin"))))


_PATH_DEFAULT = None


def _template_path_default():
    return L(os.path.join(os.path.expanduser("~"), ".mikoshilang", "Templates"), os.getcwd())


import importlib as _il  # noqa: E402
_EV = _il.import_module("mikoshilang.evaluate")
_EV.SYSTEM_VARIABLES["$TemplatePath"] = _template_path_default
record("$TemplatePath", AREA, "added",
       "directories searched (after the working directory) by FileTemplate/StringTemplate[File[...]]: "
       "{~/.mikoshilang/Templates, Directory[]} by default; assignable")
record("TemplateObject", AREA, "added",
       "TemplateObject[body, (assoc), InsertionFunction -> f, CombinerFunction -> g] is applied with "
       "TemplateApply or t[args]; defaults Identity")
for _n, _note in (("TemplateSlot", "slot n or \"name\" with InsertionFunction and \"DefaultValue\" options"),
                  ("TemplateExpression", "HoldFirst; the expression is filled in and evaluated on application"),
                  ("TemplateIf", "HoldAll; TemplateIf[cond, t, f] chooses a branch at application time"),
                  ("TemplateSequence", "TemplateSequence[body, values] splices one copy of body per value, "
                                       "with Delimiters, InsertionFunction and \"DefaultValue\""),
                  ("TemplateWith", "HoldAll; TemplateWith[\"name\" -> value or list, body] binds named slots"),
                  ("CombinerFunction", "template option: function applied to the whole filled-in body"),
                  ("InsertionFunction", "template option: function applied to each inserted value")):
    record(_n, AREA, "added", "template construct: " + _note)


# ── JSON insertion ──
def _json_value(e):
    if e is True or e is False:
        return e
    if is_sym(e, "Null"):
        return None
    if isinstance(e, (int, float, str)):
        return e
    if is_assoc(e):
        return {_json_key(r.args[0]): _json_value(r.args[1]) for r in e.args}
    if is_list(e):
        if e.args and all(is_rule(r) for r in e.args):
            return {_json_key(r.args[0]): _json_value(r.args[1]) for r in e.args}
        return [_json_value(a) for a in e.args]
    from fractions import Fraction  # noqa: F401
    try:
        v = ev(Expr("N", e))
        if isinstance(v, float):
            return v
    except Exception:  # noqa: BLE001
        pass
    raise ValueError(f"cannot export {e!r} as JSON")


def _json_key(k):
    if isinstance(k, str):
        return k
    raise ValueError(f"JSON object keys must be strings, not {k!r}")


def _to_json(e):
    import json
    return json.dumps(_json_value(e), indent=4)


# ── XMLTemplate ──
_XML_TAG = re.compile(r"<(/?)wolfram:([A-Za-z]+)((?:\s+[\w:.-]+\s*=\s*(?:'[^']*'|\"[^\"]*\"|[^\s/>]+))*)\s*(/?)>")
_XML_ATTR = re.compile(r"([\w:.-]+)\s*=\s*(?:'([^']*)'|\"([^\"]*)\"|([^\s/>]+))")
_XML_BODIES = set()    # bodies of XML templates with block elements: blank lines are dropped from their output
_XML_BLOCK = {"sequence", "with", "template", "comment", "which", "if", "else", "get", "verbatim"}


def _xml_tokens(s):
    """Tokens: ('text', str) | ('open', name, attrs, selfclosing) | ('close', name)."""
    toks, pos = [], 0
    for m in _XML_TAG.finditer(s):
        a, b = m.start(), m.end()
        name = m.group(2)
        if a > pos:
            toks.append(("text", s[pos:a]))
        attrs = {}
        for am in _XML_ATTR.finditer(m.group(3) or ""):
            attrs[am.group(1)] = next(g for g in am.groups()[1:] if g is not None)
        if m.group(1):
            toks.append(("close", name))
        else:
            toks.append(("open", name, attrs, bool(m.group(4))))
        pos = b
    if pos < len(s):
        toks.append(("text", s[pos:]))
    return toks


def _xml_code(code, what):
    from ..parser import parse
    try:
        return _slots_to_template(parse(_html.unescape(code).strip()))
    except Exception as err:  # noqa: BLE001
        raise ValueError(f"XMLTemplate: cannot parse {what} {code!r}: {err}")


def _xml_fn(v):
    if v is None:
        return None
    if re.fullmatch(r"\"[^\"]*\"", v.strip()):
        return v.strip()[1:-1]
    return ev(_xml_code(v, "function"))


def _xml_parse(toks, i, end_name=None):
    """Parse tokens from i until </wolfram:end_name>; returns (pieces, i_after)."""
    pieces = []
    while i < len(toks):
        t = toks[i]
        if t[0] == "text":
            pieces.extend(_parse_string_template(t[1], "XMLTemplate"))
            i += 1
            continue
        if t[0] == "close":
            if t[1] != end_name:
                raise ValueError(f"XMLTemplate: unexpected </wolfram:{t[1]}>")
            return pieces, i + 1
        _, name, attrs, selfc = t
        i += 1
        content, raw = [], ""
        if not selfc:
            if name in ("verbatim", "comment"):
                j, depth = i, 0
                buf = []
                while j < len(toks):
                    tj = toks[j]
                    if tj[0] == "close" and tj[1] == name and depth == 0:
                        break
                    if tj[0] == "open" and tj[1] == name and not tj[3]:
                        depth += 1
                    if tj[0] == "close" and tj[1] == name:
                        depth -= 1
                    buf.append(_xml_untok(tj))
                    j += 1
                else:
                    raise ValueError(f"XMLTemplate: missing </wolfram:{name}>")
                raw = "".join(buf)
                i = j + 1
            elif name == "expr":
                buf = []
                while i < len(toks) and not (toks[i][0] == "close" and toks[i][1] == "expr"):
                    buf.append(_xml_untok(toks[i]))
                    i += 1
                if i >= len(toks):
                    raise ValueError("XMLTemplate: missing </wolfram:expr>")
                raw = "".join(buf)
                i += 1
            else:
                content, i = _xml_parse(toks, i, name)
        pieces.extend(_xml_element(name, attrs, content, raw, selfc))
    if end_name is not None:
        raise ValueError(f"XMLTemplate: missing </wolfram:{end_name}>")
    return pieces, i


def _xml_untok(t):
    if t[0] == "text":
        return t[1]
    if t[0] == "close":
        return f"</wolfram:{t[1]}>"
    attrs = "".join(f" {k}='{v}'" for k, v in t[2].items())
    return f"<wolfram:{t[1]}{attrs}{'/' if t[3] else ''}>"


def _rule(k, v):
    return Expr("Rule", Symbol(k) if k[0].isupper() and not k.startswith("\0") else k, v)


def _xml_element(name, attrs, content, raw, selfc):
    ins = _xml_fn(attrs.get("inserter"))
    insopt = [_rule("InsertionFunction", ins)] if ins is not None else []
    if name == "slot":
        sid = attrs.get("id", "1")
        key = int(sid) if sid.isdigit() else sid
        opts = list(insopt)
        if content:
            d = [c.strip() if isinstance(c, str) else c for c in content]
            d = [c for c in d if c != ""]
            opts.append(Expr("Rule", "DefaultValue", d[0] if len(d) == 1 else L(*d)))
        return [Expr("TemplateSlot", key, *opts)]
    if name == "expr":
        return [Expr("TemplateExpression", _xml_code(raw, "expression"), *insopt)]
    if name == "comment":
        return []
    if name == "verbatim":
        return [raw]
    if name == "if":
        if "test" not in attrs:
            raise ValueError("XMLTemplate: <wolfram:if> needs a test attribute")
        return [Expr("TemplateIf", Expr("TrueQ", _xml_code(attrs["test"], "test")), L(*content), "", *insopt)]
    if name == "else":
        return [Expr("\0else", L(*content))]
    if name == "which":
        branches = [c for c in content if isinstance(c, Expr) and c.head in ("TemplateIf", "\0else")]
        res = ""
        for b in reversed(branches):
            if b.head == "\0else":
                res = b.args[0]
            else:
                res = Expr("TemplateIf", b.args[0], b.args[1], res, *b.args[3:])
        return [res]
    if name == "sequence":
        if "values" not in attrs:
            raise ValueError("XMLTemplate: <wolfram:sequence> needs a values attribute")
        opts = list(insopt)
        if "delimiters" in attrs:
            opts.append(Expr("Rule", Symbol("Delimiters"), attrs["delimiters"]))
        if "default" in attrs:
            d = _parse_string_template(attrs["default"], "XMLTemplate")
            opts.append(Expr("Rule", "DefaultValue", d[0] if len(d) == 1 else L(*d)))
        if "slot" in attrs:
            opts.append(Expr("Rule", "\0slot", attrs["slot"]))
        if "index" in attrs:
            opts.append(Expr("Rule", "\0index", attrs["index"]))
        return [Expr("TemplateSequence", L(*content), Expr("TemplateExpression", _xml_code(attrs["values"], "values")),
                     *opts)]
    if name == "with":
        rules = [Expr("Rule", k, Expr("TemplateExpression", _xml_code(v, "value"))) for k, v in attrs.items()]
        return [Expr("TemplateWith", L(*rules), L(*content))]
    if name == "template":
        comb = _xml_fn(attrs.get("combiner"))
        opts = list(insopt) + ([_rule("CombinerFunction", comb)] if comb is not None else [])
        return [Expr("TemplateObject", L(*content), *opts)]
    if name == "get":
        if "path" not in attrs:
            raise ValueError("XMLTemplate: <wolfram:get> needs a path attribute")
        p = _find_template_file(attrs["path"])
        with open(p, encoding="utf-8") as fh:
            sub = _xml_parse(_xml_tokens(fh.read()), 0)[0]
        comb = _xml_fn(attrs.get("combiner"))
        opts = list(insopt) + ([_rule("CombinerFunction", comb)] if comb is not None else [])
        defaults = [c for c in content if isinstance(c, Expr) and c.head == "TemplateSlot"]
        if defaults:
            bind = [Expr("Rule", d.args[0], next((r.args[1] for r in d.args[1:] if is_rule(r)
                                                  and r.args[0] == "DefaultValue"), "")) for d in defaults]
            return [Expr("TemplateWith", L(*bind), Expr("TemplateObject", L(*sub), *opts))]
        return [Expr("TemplateObject", L(*sub), *opts)]
    raise ValueError(f"XMLTemplate: unknown element <wolfram:{name}>")


def _xml_template(text, rest, fname="XMLTemplate"):
    pos, o = _t_opts(rest, fname, ("InsertionFunction", "CombinerFunction"))
    if any(not is_assoc(p) for p in pos) or len(pos) > 1:
        raise ValueError(f"{fname}: the optional second argument must be an association")
    toks = _xml_tokens(text)
    pieces, _ = _xml_parse(toks, 0)
    body = L(*pieces)
    if any(t[0] != "text" and t[1] in _XML_BLOCK for t in toks):
        _XML_BODIES.add(K.tag(body))
    return Expr("TemplateObject", body, *pos,
                Expr("Rule", Symbol("InsertionFunction"), o.get("InsertionFunction", "HTMLFragment")),
                Expr("Rule", Symbol("CombinerFunction"), o.get("CombinerFunction", Symbol("StringJoin"))))


@fn("XMLTemplate", area=AREA, status="added",
    notes="TemplateObject from XML/HTML text or File with <wolfram:slot id= inserter=>default</wolfram:slot>, "
          "expr, if test=, which/else, sequence (values, slot, index, delimiters, default, inserter), with, "
          "template (inserter, combiner), get path=, comment and verbatim, plus `slot` and <* expr *>; "
          "InsertionFunction \"HTMLFragment\" (escapes & < >) by default, \"JSON\" supported; block tags alone on "
          "a line drop that line")
def _xmltemplate(src, *rest):
    return _xml_template(_template_source(src, "XMLTemplate"), rest)


_XML_EXT = {".html", ".htm", ".xml", ".xhtml", ".svg"}


@fn("FileTemplate", area=AREA, status="added",
    notes="template from a file (string path or File, searched in the working directory then $TemplatePath): "
          ".html/.htm/.xml/.xhtml/.svg files are XMLTemplates, anything else a StringTemplate")
def _filetemplate(src, *rest):
    if isinstance(src, Expr) and src.head == "File" and src.args:
        path = src.args[0]
    elif isinstance(src, str):
        path = src
    elif isinstance(src, Symbol):
        raise Unevaluated()
    else:
        raise ValueError("FileTemplate: the source must be a file name or File[...]")
    p = _find_template_file(path)
    with open(p, encoding="utf-8") as fh:
        text = fh.read()
    if os.path.splitext(p)[1].lower() in _XML_EXT:
        return _xml_template(text, rest, "FileTemplate")
    return ev(Expr("StringTemplate", text, *rest))


@fn("FileTemplateApply", area=AREA, status="added",
    notes="applies a template (TemplateObject, string or File) and writes the result (strings as is, other "
          "results as TextString) to the given file, or a new temporary .txt file; returns the file name")
def _filetemplateapply(t, *rest):
    if len(rest) > 2:
        raise ValueError("FileTemplateApply called with too many arguments")
    args = list(rest[:1])
    out = rest[1] if len(rest) == 2 else None
    tmpl = ev(Expr("FileTemplate", t)) if isinstance(t, Expr) and t.head == "File" else _as_template(t)
    r = _template_apply(tmpl, _template_ctx(args))
    text = r if isinstance(r, str) else _text_string(r)
    if out is None:
        import tempfile
        fd, path = tempfile.mkstemp(prefix="template_", suffix=".txt")
        os.close(fd)
    elif isinstance(out, str):
        path = out
    elif isinstance(out, Expr) and out.head == "File" and out.args and isinstance(out.args[0], str):
        path = out.args[0]
    else:
        raise ValueError("FileTemplateApply: the output must be a file name or File[...]")
    with open(os.path.expanduser(path), "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


# ═════════════════════════════════════════════════════════════════ biological sequences ══
_NUC_DEGEN = {"R": "AG", "Y": "CT", "S": "CG", "W": "AT", "K": "GT", "M": "AC", "B": "CGT", "D": "AGT",
              "H": "ACT", "V": "ACG", "N": "ACGT", "X": "ACGT"}
_DNA_FULL = set("ACGT")
_RNA_FULL = set("ACGU")
_DNA_ALPHA = _DNA_FULL | set(_NUC_DEGEN)
_RNA_ALPHA = _RNA_FULL | set(_NUC_DEGEN)
_AA_STD = "ACDEFGHIKLMNPQRSTVWY"
_PEP_DEGEN = {"B": "DN", "Z": "EQ", "J": "IL", "X": _AA_STD}
_PEP_FULL = set(_AA_STD) | set("UO.")
_PEP_ALPHA = _PEP_FULL | set(_PEP_DEGEN)
_BIO_TYPES = ("DNA", "RNA", "Peptide", "CircularDNA", "CircularRNA", "CircularPeptide")
_AA3 = {"Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q", "Glu": "E", "Gly": "G",
        "His": "H", "Ile": "I", "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S",
        "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V", "Sec": "U", "Pyl": "O", "Asx": "B", "Glx": "Z",
        "Xle": "J", "Xaa": "X"}
_AA3_INV = {v: k for k, v in _AA3.items()}
_COMPLEMENT = {"A": "T", "T": "A", "U": "A", "C": "G", "G": "C", "R": "Y", "Y": "R", "K": "M", "M": "K",
               "S": "S", "W": "W", "B": "V", "V": "B", "D": "H", "H": "D", "N": "N", "X": "X"}


def _base_type(t):
    return t[8:] if isinstance(t, str) and t.startswith("Circular") else t


def _alphabet(t):
    b = _base_type(t)
    return {"DNA": _DNA_ALPHA, "RNA": _RNA_ALPHA, "Peptide": _PEP_ALPHA}.get(b)


def _full_alphabet(t):
    b = _base_type(t)
    return {"DNA": _DNA_FULL, "RNA": _RNA_FULL, "Peptide": _PEP_FULL}.get(b)


def _infer_bio_type(s):
    for t in ("DNA", "RNA", "Peptide"):
        if s and set(s) <= _alphabet(t):
            return t
    return None


def _aa3_convert(s):
    if len(s) >= 3 and len(s) % 3 == 0 and re.fullmatch("(?:%s)+" % "|".join(_AA3), s):
        return "".join(_AA3[s[i:i + 3]] for i in range(0, len(s), 3))
    return None


def _pattern_letters(p, t):
    """Letter string from a StringExpression/Alternatives pattern of single letters."""
    def letter(x):
        if isinstance(x, str) and len(x) >= 1:
            return x
        if isinstance(x, Expr) and x.head == "Alternatives" and all(isinstance(a, str) and len(a) == 1
                                                                     for a in x.args):
            return _degenerate_code(set(x.args), t)
        raise ValueError(f"BioSequence: cannot make a sequence letter from {x!r}")
    parts = p.args if isinstance(p, Expr) and p.head == "StringExpression" else [p]
    return "".join(letter(x) for x in parts)


def _degenerate_code(letters, t):
    if len(letters) == 1:
        return next(iter(letters))
    b = _base_type(t)
    table = _PEP_DEGEN if b == "Peptide" else _NUC_DEGEN
    if b == "RNA":
        letters = {"T" if c == "U" else c for c in letters}
    for code, members in table.items():
        if code != "X" and set(members) == letters:
            return code
    raise ValueError(f"BioSequence: no single letter stands for {sorted(letters)}")


def _dot_bracket(s, n):
    pairs = {")": "(", ">": "<", "]": "[", "}": "{"}
    stacks = {o: [] for o in pairs.values()}
    bonds = []
    if len(s) != n:
        raise ValueError("BioSequence: the dot-bracket structure must have one character per letter")
    for i, c in enumerate(s, 1):
        if c in stacks:
            stacks[c].append(i)
        elif c in pairs:
            if not stacks[pairs[c]]:
                raise ValueError("BioSequence: unbalanced dot-bracket structure")
            bonds.append(Expr("Bond", L(stacks[pairs[c]].pop(), i)))
        elif c != ".":
            raise ValueError(f"BioSequence: invalid dot-bracket character {c!r}")
    if any(stacks.values()):
        raise ValueError("BioSequence: unbalanced dot-bracket structure")
    return L(*bonds)


def _bonds_arg(b, fname="BioSequence"):
    if isinstance(b, Expr) and b.head == "Bond":
        return L(b)
    if is_list(b) and all(isinstance(x, Expr) and x.head == "Bond" for x in b.args):
        return b
    raise ValueError(f"{fname}: bonds must be Bond[...] or a list of them")


def _strand(x):
    """Normalise a strand inside HybridStrand/collections to BioSequence[type, s, bonds]."""
    if isinstance(x, str):
        return Expr("BioSequence", _infer_bio_type(x) or Symbol("None"), x, L())
    x = ev(x) if isinstance(x, Expr) and x.head == "InputForm" else x
    if isinstance(x, Expr) and x.head == "InputForm" and x.args:
        x = x.args[0]
    if isinstance(x, Expr) and x.head == "BioSequence":
        if len(x.args) == 2 and isinstance(x.args[1], str):
            return Expr("BioSequence", x.args[0], x.args[1], L())
        return x
    raise ValueError(f"BioSequence: {x!r} is not a sequence")


@fn("BioSequence", area=AREA, status="added",
    notes="BioSequence[type, \"letters\" (, bonds)] for DNA, RNA, Peptide, their Circular forms and None; "
          "BioSequence[\"letters\"] infers DNA, then RNA, then Peptide (else None); three-letter amino-acid "
          "codes, letter patterns (\"A\" | \"G\" ~~ ...), dot-bracket structures and Bond lists; HybridStrand "
          "and sequence collections; properties SequenceString, SequenceLength, SequenceType, SequenceBondList, "
          "SequenceBondCount, SequencePattern, AbbreviationSequence, ChemicalList, ChemicalPatternList, HELM. "
          "Masses, molecules and entity-backed sequences are not available")
def _biosequence(*args):
    if len(args) == 1:
        a = args[0]
        if isinstance(a, str):
            conv = _aa3_convert(a)
            if conv is not None:
                return Expr("BioSequence", "Peptide", conv, L())
            t = _infer_bio_type(a)
            return Expr("BioSequence", t if t else Symbol("None"), a, *([] if t else [L()]))
        if is_list(a):
            return Expr("BioSequence", L(*(_strand(x) for x in a.args)), L())
        raise Unevaluated()
    t, s = args[0], args[1]
    rest = args[2:]
    if len(rest) > 1:
        raise ValueError("BioSequence called with too many arguments")
    if is_list(t):
        if len(args) != 2:
            raise ValueError("BioSequence: a collection takes a list of sequences and a list of bonds")
        strands = L(*(_strand(x) for x in t.args))
        bonds = _bonds_arg(s)
        if K.tag(strands) == K.tag(t) and K.tag(bonds) == K.tag(s):
            raise Unevaluated()
        return Expr("BioSequence", strands, bonds)
    if isinstance(t, Expr) and t.head == "Entity" and len(t.args) == 2 and t.args[0] == "BioSequenceType":
        t = t.args[1]
    if t == "HybridStrand":
        if not is_list(s):
            raise ValueError("BioSequence: a HybridStrand takes a list of strands")
        strands = L(*(_strand(x) for x in s.args))
        bonds = _bonds_arg(rest[0]) if rest else L()
        new = Expr("BioSequence", "HybridStrand", strands, bonds)
        if rest and K.tag(strands) == K.tag(s) and K.tag(bonds) == K.tag(rest[0]):
            raise Unevaluated()
        return new
    if not (isinstance(t, str) or is_sym(t, "None")):
        raise Unevaluated()
    if isinstance(s, Expr) and s.head in ("StringExpression", "Alternatives"):
        return Expr("BioSequence", t, _pattern_letters(s, t), *rest)
    if not isinstance(s, str):
        raise Unevaluated()
    if _base_type(t) == "Peptide" and not set(s) <= _PEP_ALPHA:
        conv = _aa3_convert(s)
        if conv is not None:
            return Expr("BioSequence", t, conv, *(rest or [L()]))
    if rest:
        b = rest[0]
        if isinstance(b, str):
            return Expr("BioSequence", t, s, _dot_bracket(b, len(s)))
        nb = _bonds_arg(b)
        if nb is not b:
            return Expr("BioSequence", t, s, nb)
    raise Unevaluated()


def _bio_parts(x):
    """(type, string, bonds or None) of a simple BioSequence, else None."""
    if isinstance(x, Expr) and x.head == "BioSequence" and len(x.args) in (2, 3) and isinstance(x.args[1], str) \
            and (isinstance(x.args[0], str) or is_sym(x.args[0], "None")) and x.args[0] != "HybridStrand":
        return x.args[0], x.args[1], (x.args[2] if len(x.args) == 3 else None)
    return None


def _bio_valid(x):
    p = _bio_parts(x)
    if p is not None:
        t, s, b = p
        if is_sym(t, "None"):
            return True
        a = _alphabet(t)
        return a is not None and set(s) <= a and (b is None or is_list(b))
    if isinstance(x, Expr) and x.head == "BioSequence":
        if len(x.args) == 3 and x.args[0] == "HybridStrand" and is_list(x.args[1]):
            return all(_bio_valid(y) and _bio_parts(y) and not str(_bio_parts(y)[0]).startswith("Circular")
                       for y in x.args[1].args)
        if len(x.args) == 2 and is_list(x.args[0]) and is_list(x.args[1]):
            return all(_bio_valid(y) for y in x.args[0].args)
    return False


def _bio_kind(x):
    p = _bio_parts(x)
    if p is not None:
        return p[0] if isinstance(p[0], str) else "None"
    if len(x.args) == 3 and x.args[0] == "HybridStrand":
        return "HybridStrand"
    return "SequenceCollection"


def _fully_specified(x):
    p = _bio_parts(x)
    if p is not None:
        t, s, _ = p
        f = _full_alphabet(t) if isinstance(t, str) else None
        return f is not None and set(s) <= f
    strands = x.args[1].args if _bio_kind(x) == "HybridStrand" else x.args[0].args
    return all(_fully_specified(y) for y in strands)


@fn("BioSequenceQ", area=AREA, status="added",
    notes="True for a valid BioSequence (letters in the type's alphabet); BioSequenceQ[x, type] also checks "
          "the type (a string pattern such as \"DNA\" | \"RNA\", \"HybridStrand\", \"SequenceCollection\"), "
          "\"FullySpecified\" or \"FullySpecifiedtype\" (no degenerate letters)")
def _biosequenceq(x, *rest):
    if len(rest) > 1:
        raise ValueError("BioSequenceQ called with too many arguments")
    if not _bio_valid(x):
        return False
    if not rest:
        return True
    spec = rest[0]
    kind = _bio_kind(x)
    if isinstance(spec, str) and spec.startswith("FullySpecified"):
        sub = spec[len("FullySpecified"):]
        if sub and sub != kind:
            return False
        return _fully_specified(x)
    return ev(Expr("MatchQ", kind, spec)) is True


def _bio_rebuild(x, t, s, bonds):
    if bonds is None:
        return Expr("BioSequence", t, s)
    return Expr("BioSequence", t, s, bonds)


def _nucleic(x, fname):
    p = _bio_parts(x)
    if p is None or not _bio_valid(x) or _base_type(p[0]) not in ("DNA", "RNA"):
        raise Unevaluated()
    return p


def _complement_str(s, t):
    rna = _base_type(t) == "RNA"
    return "".join(("U" if rna and _COMPLEMENT[c] == "T" else _COMPLEMENT[c]) for c in s)


@fn("BioSequenceComplement", area=AREA, status="added",
    notes="complementary DNA/RNA strand letter by letter (degenerate letters too: R<->Y, K<->M, B<->V, D<->H); "
          "unevaluated for peptides and invalid sequences")
def _biocomp(x):
    if isinstance(x, str):
        x = ev(Expr("BioSequence", x))
    t, s, b = _nucleic(x, "BioSequenceComplement")
    return _bio_rebuild(x, t, _complement_str(s, t), b)


@fn("BioSequenceReverseComplement", area=AREA, status="added",
    notes="reverse of the complementary DNA/RNA strand (bond positions mirrored); unevaluated for peptides and "
          "invalid sequences")
def _biorevcomp(x):
    if isinstance(x, str):
        x = ev(Expr("BioSequence", x))
    t, s, b = _nucleic(x, "BioSequenceReverseComplement")
    n = len(s)
    if b is not None and b.args:
        b = L(*(_mirror_bond(bd, n) for bd in b.args))
    return _bio_rebuild(x, t, _complement_str(s, t)[::-1], b)


def _mirror_bond(bd, n):
    pos = bd.args[0]
    if is_list(pos) and all(isinstance(i, int) for i in pos.args):
        return Expr("Bond", L(*sorted(n + 1 - i for i in pos.args)), *bd.args[1:])
    raise ValueError("BioSequenceReverseComplement: cannot mirror bond " + repr(bd))


@fn("BioSequenceTranscribe", area=AREA, status="added",
    notes="DNA -> RNA (T -> U) and RNA -> DNA (U -> T), circular forms kept; unevaluated for peptides and "
          "invalid sequences")
def _biotranscribe(x):
    if isinstance(x, str):
        x = ev(Expr("BioSequence", x))
    t, s, b = _nucleic(x, "BioSequenceTranscribe")
    circ = "Circular" if t.startswith("Circular") else ""
    if _base_type(t) == "DNA":
        return _bio_rebuild(x, circ + "RNA", s.replace("T", "U"), b)
    return _bio_rebuild(x, circ + "DNA", s.replace("U", "T"), b)


_GENETIC_CODES = {
    1: ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG", {"TTG", "CTG", "ATG"}),
    2: ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIMMTTTTNNKKSS**VVVVAAAADDEEGGGG",
        {"ATT", "ATC", "ATA", "ATG", "GTG"}),
    3: ("FFLLSSSSYY**CCWWTTTTPPPPHHQQRRRRIIMMTTTTNNKKSSRRVVVVAAAADDEEGGGG", None),
    4: ("FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG", None),
    5: ("FFLLSSSSYY**CCWWLLLLPPPPHHQQRRRRIIMMTTTTNNKKSSSSVVVVAAAADDEEGGGG", None),
    11: ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG",
         {"TTG", "CTG", "ATT", "ATC", "ATA", "ATG", "GTG"}),
}
_CODE_NAMES = {"Standard": 1, "VertebrateMitochondrial": 2, "YeastMitochondrial": 3,
               "MoldMitochondrial": 4, "InvertebrateMitochondrial": 5, "Bacterial": 11,
               "BacterialArchaealAndPlantPlastid": 11}
_TCAG = "TCAG"
_CODONS = [a + b + c for a in _TCAG for b in _TCAG for c in _TCAG]


def _code_table(spec, fname):
    """(dict codon -> letter, start codon set or None)."""
    if spec is None or is_sym(spec, "Automatic"):
        spec = 1
    if isinstance(spec, Expr) and spec.head == "Entity" and len(spec.args) == 2 \
            and spec.args[0] == "GeneticTranslationTable":
        spec = spec.args[1]
    if isinstance(spec, str) and spec in _CODE_NAMES:
        spec = _CODE_NAMES[spec]
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec not in _GENETIC_CODES:
            raise NotImplementedError(f"{fname}: genetic code {spec} is not available "
                                      f"(available: {sorted(_GENETIC_CODES)})")
        aa, starts = _GENETIC_CODES[spec]
        return dict(zip(_CODONS, aa)), starts
    if isinstance(spec, str) and len(spec) == 64:
        return dict(zip(_CODONS, spec)), None
    if is_assoc(spec) or (is_list(spec) and all(is_rule(r) for r in spec.args)):
        tab = {}
        for r in spec.args:
            k, v = r.args[0], ev(r.args[1])
            if not (isinstance(k, str) and isinstance(v, str) and len(k) == 3 and len(v) == 1):
                raise ValueError(f"{fname}: invalid codon rule {r!r}")
            tab[k.upper().replace("U", "T")] = v
        return tab, None
    raise ValueError(f"{fname}: invalid translation table {spec!r}")


def _translate_codon(c, tab):
    choices = set()
    for inst in itertools.product(*[_NUC_DEGEN.get(ch, ch) for ch in c]):
        k = "".join(inst)
        if k not in tab:
            return None
        choices.add(tab[k])
    if len(choices) == 1:
        v = choices.pop()
        return "." if v == "*" else v
    return "X"


@fn("BioSequenceTranslate", area=AREA, status="added",
    notes="translates DNA/RNA codons to a Peptide (stop codons give \".\", incomplete final codons dropped, a "
          "degenerate codon gives its amino acid if all its instances agree, else X); genetic codes 1-5 and 11 "
          "by number or name, a 64-letter TCAG-ordered string or codon rules; third argument True (start codons "
          "of codes 1, 2, 11 give M), a table, or rules for the first codon")
def _biotranslate(x, table=None, start=None):
    if isinstance(x, str):
        x = ev(Expr("BioSequence", x))
    t, s, _ = _nucleic(x, "BioSequenceTranslate")
    s = s.replace("U", "T")
    tab, starts = _code_table(table, "BioSequenceTranslate")
    codons = [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]
    out = []
    for k, c in enumerate(codons):
        use = tab
        if k == 0 and start is not None and start is not False:
            if start is True:
                if starts is None:
                    raise NotImplementedError("BioSequenceTranslate: start codons are only known for genetic "
                                              "codes 1, 2 and 11")
                if c in starts:
                    out.append("M")
                    continue
            else:
                st, _ = _code_table(start, "BioSequenceTranslate")
                use = dict(tab)
                use.update(st)
        r = _translate_codon(c, use)
        if r is None:
            raise ValueError(f"BioSequenceTranslate: no translation for codon {c!r}")
        out.append(r)
    circ = "Circular" if t.startswith("Circular") else ""
    return Expr("BioSequence", circ + "Peptide", "".join(out))


def _instances_of(s, t):
    b = _base_type(t)
    choices = []
    for c in s:
        if b == "Peptide":
            opts = _PEP_DEGEN.get(c, c)
        else:
            opts = _NUC_DEGEN.get(c, c)
            if b == "RNA":
                opts = opts.replace("T", "U")
        choices.append(sorted(opts))
    return choices


@fn("BioSequenceInstances", area=AREA, status="added",
    notes="all fully specified sequences matching a degenerate BioSequence, sorted; BioSequenceInstances[seq, n] "
          "gives at most n of them (the first n in sorted order)")
def _bioinstances(x, n=None):
    p = _bio_parts(x)
    if p is None or not _bio_valid(x) or not isinstance(p[0], str):
        raise Unevaluated()
    t, s, b = p
    if n is not None:
        n = _int(n, "BioSequenceInstances", "count", 0)
    choices = _instances_of(s, t)
    out = []
    for combo in itertools.product(*choices):
        if n is not None and len(out) >= n:
            break
        out.append(_bio_rebuild(x, t, "".join(combo), b))
    return L(*out)


def _bonds_of(x):
    if _bio_parts(x) is not None:
        b = x.args[2] if len(x.args) == 3 else L()
        return b
    return x.args[2] if _bio_kind(x) == "HybridStrand" else x.args[1]


def _set_bonds(x, bonds):
    if _bio_parts(x) is not None:
        return Expr("BioSequence", x.args[0], x.args[1], bonds)
    if _bio_kind(x) == "HybridStrand":
        return Expr("BioSequence", "HybridStrand", x.args[1], bonds)
    return Expr("BioSequence", x.args[0], bonds)


@fn("BioSequenceModify", area=AREA, status="added",
    notes="\"MakeCircular\", \"MakeLinear\", \"DropIncompleteCodons\", \"DropToStartCodon\" (standard start "
          "codons TTG/CTG/ATG), \"DropFromStopLetter\", {\"DeleteBond\", bond} and {\"AddToCollection\", seqs}; "
          "\"AddBond\" (which validates base pairing chemistry) and the bond-representation conversions are not "
          "supported")
def _biomodify(x, op):
    if not _bio_valid(x):
        raise Unevaluated()
    p = _bio_parts(x)
    if isinstance(op, str):
        if p is None:
            raise NotImplementedError(f"BioSequenceModify: {op!r} on strands/collections is not supported")
        t, s, b = p
        if op in ("MakeCircular", "MakeLinear"):
            if not isinstance(t, str):
                raise ValueError("BioSequenceModify: the sequence has no type")
            base = _base_type(t)
            return _bio_rebuild(x, ("Circular" + base) if op == "MakeCircular" else base, s, b)
        if op == "DropIncompleteCodons":
            if _base_type(t) not in ("DNA", "RNA"):
                raise Unevaluated()
            return _bio_rebuild(x, t, s[:len(s) - len(s) % 3], b)
        if op == "DropToStartCodon":
            if _base_type(t) not in ("DNA", "RNA"):
                raise Unevaluated()
            dna = s.replace("U", "T")
            idx = [dna.find(c) for c in ("TTG", "CTG", "ATG")]
            idx = [i for i in idx if i >= 0]
            if not idx:
                raise ValueError("BioSequenceModify: the sequence has no start codon")
            return _bio_rebuild(x, t, s[min(idx):], None if b is None else L())
        if op == "DropFromStopLetter":
            if _base_type(t) != "Peptide":
                raise Unevaluated()
            k = s.find(".")
            return x if k < 0 else _bio_rebuild(x, t, s[:k], None if b is None else L())
        raise NotImplementedError(f"BioSequenceModify: modification {op!r} is not supported")
    if is_list(op) and len(op.args) == 2 and isinstance(op.args[0], str):
        name, arg = op.args
        if name == "DeleteBond":
            bonds = _bonds_of(x)
            gone = {K.tag(bd) for bd in _bonds_arg(arg, "BioSequenceModify").args}
            return _set_bonds(x, L(*(bd for bd in bonds.args if K.tag(bd) not in gone)))
        if name == "AddToCollection":
            new = [_strand(y) for y in (arg.args if is_list(arg) else [arg])]
            if _bio_kind(x) == "SequenceCollection":
                return Expr("BioSequence", L(*x.args[0].args, *new), x.args[1])
            return Expr("BioSequence", L(_strand(x), *new), L())
        raise NotImplementedError(f"BioSequenceModify: modification {name!r} is not supported")
    raise ValueError(f"BioSequenceModify: invalid modification {op!r}")


_CHEM = {"A": "Adenine", "C": "Cytosine", "G": "Guanine", "T": "Thymine", "U": "Uracil"}
_BIO_PROPS = ("AbbreviationSequence", "ChemicalList", "ChemicalPatternList", "HELM", "MetaInformation",
              "MolarMass", "MolecularMass", "Properties", "SequenceBondCount", "SequenceBondList",
              "SequenceLength", "SequencePattern", "SequenceString", "SequenceType")


def _letter_options(c, t):
    return _instances_of(c, t)[0]


def _bio_property(x, prop):
    if not isinstance(prop, str):
        raise Unevaluated()
    if prop == "Properties":
        return L(*_BIO_PROPS)
    p = _bio_parts(x)
    if p is None:
        kind = _bio_kind(x)
        strands = x.args[1].args if kind == "HybridStrand" else x.args[0].args
        if prop in ("SequenceType", "SequenceString", "SequenceLength"):
            vals = [_bio_property(y, prop) for y in strands]
            if kind == "HybridStrand":
                return L(*vals)
            return L(*(v if is_list(v) else L(v) for v in vals))
        if prop == "SequenceBondList":
            return _bonds_of(x)
        if prop == "SequenceBondCount":
            return len(_bonds_of(x).args)
        if prop in ("MolecularMass", "MolarMass", "MetaInformation"):
            raise NotImplementedError(f"BioSequence: property {prop!r} is not available")
        raise NotImplementedError(f"BioSequence: property {prop!r} is not available for strands and collections")
    t, s, b = p
    if prop == "SequenceString":
        return s
    if prop == "SequenceLength":
        return len(s)
    if prop == "SequenceType":
        return Expr("Entity", "BioSequenceType", t) if isinstance(t, str) else Symbol("None")
    if prop == "SequenceBondList":
        return b if b is not None else L()
    if prop == "SequenceBondCount":
        return len(b.args) if b is not None else 0
    if not _bio_valid(x) or not isinstance(t, str):
        raise ValueError(f"BioSequence: property {prop!r} needs a valid typed sequence")
    base = _base_type(t)
    if prop == "SequencePattern":
        parts, buf = [], []
        for c in s:
            opts = _letter_options(c, t)
            if len(opts) == 1:
                buf.append(opts[0])
            else:
                if buf:
                    parts.append("".join(buf))
                    buf = []
                parts.append(Expr("Alternatives", *opts))
        if buf:
            parts.append("".join(buf))
        return parts[0] if len(parts) == 1 else Expr("StringExpression", *parts)
    if prop == "AbbreviationSequence":
        if base != "Peptide":
            return Expr("Missing", "NotApplicable")
        if "." in s:
            raise ValueError("BioSequence: AbbreviationSequence of a sequence with stop letters")
        return "".join(_AA3_INV[c] for c in s)
    if prop in ("ChemicalList", "ChemicalPatternList"):
        if base == "Peptide":
            raise NotImplementedError("BioSequence: chemical entities for amino acids are not available")
        out = []
        for c in s:
            opts = _letter_options(c, t)
            ents = [Expr("Entity", "Chemical", _CHEM[o]) for o in opts]
            if len(ents) > 1 and prop == "ChemicalList":
                raise ValueError("BioSequence: ChemicalList needs a fully specified sequence; use "
                                 "ChemicalPatternList")
            out.append(ents[0] if len(ents) == 1 else Expr("Alternatives", *ents))
        return L(*out)
    if prop == "HELM":
        if t.startswith("Circular") or not _fully_specified(x) or (b is not None and b.args):
            raise NotImplementedError("BioSequence: HELM is only available for linear, fully specified "
                                      "sequences without bonds")
        if base == "Peptide":
            if "." in s:
                raise ValueError("BioSequence: HELM of a peptide with stop letters")
            return "PEPTIDE1{" + ".".join(s) + "}$$$$V2.0"
        sugar = "D" if base == "DNA" else "R"
        return "RNA1{" + ".".join(f"{sugar}({c})" + ("P" if i < len(s) - 1 else "") for i, c in enumerate(s)) \
            + "}$$$$V2.0"
    if prop in ("MolecularMass", "MolarMass", "MetaInformation"):
        raise NotImplementedError(f"BioSequence: property {prop!r} is not available")
    raise ValueError(f"BioSequence: unknown property {prop!r}")


def _bio_hook(head, args, ev_):
    if head.head == "BioSequence" and len(args) == 1 and isinstance(args[0], str) and \
            (_bio_parts(head) is not None or _bio_valid(head)):
        return _bio_property(head, args[0])
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["BioSequence"] = _bio_hook
