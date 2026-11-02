"""Wolfram batch X04: BarcodeImage — barcode encoders with decoding self-checks.

QR codes follow ISO/IEC 18004:2015 (model 2: numeric, alphanumeric or byte mode, Reed–Solomon error correction
over GF(256) with the field polynomial 0x11D, the eight data masks and the standard penalty rules); EAN-13,
UPC-A and EAN-8 follow ISO/IEC 15420 (GS1 General Specifications §5.2); Code 128 follows ISO/IEC 15417; Code 39
follows ISO/IEC 16388; ITF (interleaved 2 of 5) follows ISO/IEC 16390. Every symbol is decoded again from the
generated module pattern (independently of the encoder's internal state) and compared with the input; a
mismatch raises ArithmeticError.

Pure Python and NumPy; nothing here needs Pillow.
"""
from __future__ import annotations

import numpy as np

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, S
from . import wl_images as WI
from .wl_images import image_expr, opts_split, _short, _is_sym

AREA = "image processing III (X04)"

# ───────────────────────────── Reed–Solomon over GF(256), poly 0x11D ─────────────────────────────

_EXP = [0] * 512
_LOG = [0] * 256
_v = 1
for _i in range(255):
    _EXP[_i] = _v
    _LOG[_v] = _i
    _v <<= 1
    if _v & 0x100:
        _v ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]


def gf_mul(a, b):
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]


def rs_generator(degree):
    """Coefficients (highest first, monic omitted) of Π_{i<degree} (x - α^i)."""
    g = [1]
    for i in range(degree):
        ng = [0] * (len(g) + 1)
        for j, c in enumerate(g):
            ng[j] ^= c
            ng[j + 1] ^= gf_mul(c, _EXP[i])
        g = ng
    return g[1:]


def rs_remainder(data, degree):
    gen = rs_generator(degree)
    rem = [0] * degree
    for b in data:
        f = b ^ rem[0]
        rem = rem[1:] + [0]
        for j in range(degree):
            rem[j] ^= gf_mul(gen[j], f)
    return rem


def rs_syndromes_zero(block, degree):
    """True when the codeword polynomial vanishes at α^0..α^(degree-1)."""
    for i in range(degree):
        s = 0
        for c in block:
            s = gf_mul(s, _EXP[i]) ^ c
        if s:
            return False
    return True


# ───────────────────────────── QR tables (ISO/IEC 18004 Table 9) ─────────────────────────────

QR_ECC_PER_BLOCK = {
    "L": (-1, 7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28,
          30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
    "M": (-1, 10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28,
          28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28),
    "Q": (-1, 13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28,
          30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30),
    "H": (-1, 17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30,
          30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30)}
QR_NUM_BLOCKS = {
    "L": (-1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17,
          18, 19, 19, 20, 21, 22, 24, 25),
    "M": (-1, 1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28,
          29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49),
    "Q": (-1, 1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38,
          40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68),
    "H": (-1, 1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45,
          48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81)}
QR_FORMAT_BITS = {"L": 1, "M": 0, "Q": 3, "H": 2}
QR_ALNUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
QR_MASKS = (lambda x, y: (x + y) % 2, lambda x, y: y % 2, lambda x, y: x % 3, lambda x, y: (x + y) % 3,
            lambda x, y: (x // 3 + y // 2) % 2, lambda x, y: x * y % 2 + x * y % 3,
            lambda x, y: (x * y % 2 + x * y % 3) % 2, lambda x, y: ((x + y) % 2 + x * y % 3) % 2)


def qr_raw_modules(v):
    """Number of data + error-correction modules of version v (all modules minus function patterns)."""
    r = (16 * v + 128) * v + 64
    if v >= 2:
        na = v // 7 + 2
        r -= (25 * na - 10) * na - 55
        if v >= 7:
            r -= 36
    return r


def qr_data_codewords(v, ecl):
    return qr_raw_modules(v) // 8 - QR_ECC_PER_BLOCK[ecl][v] * QR_NUM_BLOCKS[ecl][v]


def qr_alignment_positions(v):
    if v == 1:
        return []
    na = v // 7 + 2
    size = 4 * v + 17
    step = 26 if v == 32 else (v * 4 + na * 2 + 1) // (na * 2 - 2) * 2
    return [6] + sorted(size - 7 - i * step for i in range(na - 1))


def _qr_mode(text):
    if text and all(c in "0123456789" for c in text):
        return "Numeric"
    if all(c in QR_ALNUM for c in text):
        return "Alphanumeric"
    return "Byte"


def _qr_count_bits(mode, v):
    k = 0 if v <= 9 else (1 if v <= 26 else 2)
    return {"Numeric": (10, 12, 14), "Alphanumeric": (9, 11, 13), "Byte": (8, 16, 16)}[mode][k]


def _qr_segment_bits(text, mode, v):
    bits = []

    def put(val, n):
        bits.extend((val >> (n - 1 - i)) & 1 for i in range(n))
    put({"Numeric": 1, "Alphanumeric": 2, "Byte": 4}[mode], 4)
    if mode == "Byte":
        data = text.encode("utf-8")
        put(len(data), _qr_count_bits(mode, v))
        for b in data:
            put(b, 8)
    elif mode == "Numeric":
        put(len(text), _qr_count_bits(mode, v))
        for i in range(0, len(text), 3):
            g = text[i:i + 3]
            put(int(g), 3 * len(g) + 1)
    else:
        put(len(text), _qr_count_bits(mode, v))
        for i in range(0, len(text) - 1, 2):
            put(QR_ALNUM.index(text[i]) * 45 + QR_ALNUM.index(text[i + 1]), 11)
        if len(text) % 2:
            put(QR_ALNUM.index(text[-1]), 6)
    return bits


def qr_codewords(text, ecl, v=None):
    """(version, mode, data codewords) for the smallest version that holds text at level ecl."""
    mode = _qr_mode(text)
    versions = range(1, 41) if v is None else [v]
    for ver in versions:
        bits = _qr_segment_bits(text, mode, ver)
        cap = qr_data_codewords(ver, ecl) * 8
        if len(bits) <= cap:
            break
    else:
        raise ValueError(f"BarcodeImage: the data is too long for a QR code at error correction level {ecl}")
    bits += [0] * min(4, cap - len(bits))
    bits += [0] * (-len(bits) % 8)
    data = [int("".join(map(str, bits[i:i + 8])), 2) for i in range(0, len(bits), 8)]
    pad = 0xEC
    while len(data) < cap // 8:
        data.append(pad)
        pad ^= 0xEC ^ 0x11
    return ver, mode, data


def qr_interleave(data, ver, ecl):
    nb = QR_NUM_BLOCKS[ecl][ver]
    ecc = QR_ECC_PER_BLOCK[ecl][ver]
    raw = qr_raw_modules(ver) // 8
    nshort = nb - raw % nb
    short_len = raw // nb
    blocks, k = [], 0
    for i in range(nb):
        dl = short_len - ecc + (0 if i < nshort else 1)
        d = data[k:k + dl]
        k += dl
        blocks.append((d, rs_remainder(d, ecc)))
    out = []
    for i in range(short_len - ecc + 1):
        for j, (d, _) in enumerate(blocks):
            if i < len(d):
                out.append(d[i])
    for i in range(ecc):
        for _, e in blocks:
            out.append(e[i])
    return out


def _qr_function_matrix(ver):
    size = 4 * ver + 17
    m = np.zeros((size, size), dtype=np.int8)      # [y, x]: 1 dark
    fn_ = np.zeros((size, size), dtype=bool)

    def setm(x, y, dark):
        m[y, x] = 1 if dark else 0
        fn_[y, x] = True
    for i in range(size):                          # timing patterns
        setm(6, i, i % 2 == 0)
        setm(i, 6, i % 2 == 0)
    for cx, cy in ((3, 3), (size - 4, 3), (3, size - 4)):     # finders with separators
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                x, y = cx + dx, cy + dy
                if 0 <= x < size and 0 <= y < size:
                    d = max(abs(dx), abs(dy))
                    setm(x, y, d not in (2, 4))
    pos = qr_alignment_positions(ver)
    n = len(pos)
    for i in range(n):
        for j in range(n):
            if (i == 0 and j == 0) or (i == 0 and j == n - 1) or (i == n - 1 and j == 0):
                continue
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    setm(pos[i] + dx, pos[j] + dy, max(abs(dx), abs(dy)) != 1)
    _qr_format(m, fn_, size, "M", 0, setm)        # reserve the format areas (overwritten later)
    if ver >= 7:
        rem = ver
        for _ in range(12):
            rem = (rem << 1) ^ ((rem >> 11) * 0x1F25)
        bits = ver << 12 | rem
        for i in range(18):
            bit = (bits >> i) & 1
            a, b = size - 11 + i % 3, i // 3
            setm(a, b, bit)
            setm(b, a, bit)
    return m, fn_, setm


def qr_format_bits(ecl, mask):
    data = QR_FORMAT_BITS[ecl] << 3 | mask
    rem = data
    for _ in range(10):
        rem = (rem << 1) ^ ((rem >> 9) * 0x537)
    return (data << 10 | rem) ^ 0x5412


def _qr_format(m, fn_, size, ecl, mask, setm):
    bits = qr_format_bits(ecl, mask)
    for i in range(6):
        setm(8, i, (bits >> i) & 1)
    setm(8, 7, (bits >> 6) & 1)
    setm(8, 8, (bits >> 7) & 1)
    setm(7, 8, (bits >> 8) & 1)
    for i in range(9, 15):
        setm(14 - i, 8, (bits >> i) & 1)
    for i in range(8):
        setm(size - 1 - i, 8, (bits >> i) & 1)
    for i in range(8, 15):
        setm(8, size - 15 + i, (bits >> i) & 1)
    setm(8, size - 8, 1)                           # the dark module


def qr_codeword_order(fn_):
    """(x, y) of the data modules in placement order (two-column zigzag from the bottom right)."""
    size = fn_.shape[0]
    out = []
    right = size - 1
    while right >= 1:
        if right == 6:
            right = 5
        for vert in range(size):
            for j in range(2):
                x = right - j
                upward = ((right + 1) & 2) == 0
                y = size - 1 - vert if upward else vert
                if not fn_[y, x]:
                    out.append((x, y))
        right -= 2
    return out


def qr_penalty(m):
    """ISO/IEC 18004 §7.8.3 penalty score (N1 = 3, N2 = 3, N3 = 40, N4 = 10)."""
    size = m.shape[0]
    score = 0
    for grid in (m, m.T):
        for row in grid:
            run, prev = 0, -1
            for v in row:
                if v == prev:
                    run += 1
                else:
                    if run >= 5:
                        score += 3 + run - 5
                    run, prev = 1, v
            if run >= 5:
                score += 3 + run - 5
            padded = np.concatenate([np.zeros(4, np.int8), row, np.zeros(4, np.int8)])
            s = "".join(map(str, padded.tolist()))
            for pat in ("10111010000", "00001011101"):
                start = s.find(pat)
                while start != -1:
                    score += 40
                    start = s.find(pat, start + 1)
    blk = m[:-1, :-1] + m[1:, :-1] + m[:-1, 1:] + m[1:, 1:]
    score += 3 * int(((blk == 0) | (blk == 4)).sum())
    dark = int(m.sum())
    total = size * size
    k = (abs(dark * 20 - total * 10) + total - 1) // total - 1
    score += 10 * k
    return score


def qr_matrix(text, ecl="M", mask=None, version=None):
    """(module matrix [y, x] with 1 = dark, version, mask) of the QR code of text."""
    ver, mode, data = qr_codewords(text, ecl, version)
    cw = qr_interleave(data, ver, ecl)
    base, fn_, _ = _qr_function_matrix(ver)
    order = qr_codeword_order(fn_)
    bits = [(cw[i >> 3] >> (7 - (i & 7))) & 1 for i in range(len(cw) * 8)]
    best = None
    for mk in (range(8) if mask is None else [mask]):
        m = base.copy()
        for i, (x, y) in enumerate(order):
            b = bits[i] if i < len(bits) else 0
            if QR_MASKS[mk](x, y) == 0:
                b ^= 1
            m[y, x] = b

        def setm(x, y, dark, m=m):
            m[y, x] = 1 if dark else 0
        _qr_format(m, fn_, m.shape[0], ecl, mk, setm)
        p = qr_penalty(m)
        if best is None or p < best[0]:
            best = (p, m, mk)
    return best[1], ver, best[2]


def qr_decode(m):
    """Read a module matrix back: format information (checked against all 32 valid codes), unmasking, the
    codeword sequence, de-interleaving, Reed–Solomon syndromes and the data segment. Returns the text."""
    size = m.shape[0]
    ver = (size - 17) // 4
    if size != 4 * ver + 17 or not 1 <= ver <= 40:
        raise ArithmeticError("BarcodeImage: self-check failed (bad QR size)")
    bits = 0
    for i in range(6):
        bits |= int(m[i, 8]) << i
    bits |= int(m[7, 8]) << 6 | int(m[8, 8]) << 7 | int(m[8, 7]) << 8
    for i in range(9, 15):
        bits |= int(m[8, 14 - i]) << i
    found = [(e, k) for e in "LMQH" for k in range(8) if qr_format_bits(e, k) == bits]
    if not found:
        raise ArithmeticError("BarcodeImage: self-check failed (format information)")
    ecl, mk = found[0]
    _, fn_, _ = _qr_function_matrix(ver)
    order = qr_codeword_order(fn_)
    stream = [int(m[y, x]) ^ (1 if QR_MASKS[mk](x, y) == 0 else 0) for x, y in order]
    raw = qr_raw_modules(ver) // 8
    cw = [int("".join(map(str, stream[i * 8:i * 8 + 8])), 2) for i in range(raw)]
    nb, ecc = QR_NUM_BLOCKS[ecl][ver], QR_ECC_PER_BLOCK[ecl][ver]
    nshort = nb - raw % nb
    short_len = raw // nb
    dls = [short_len - ecc + (0 if i < nshort else 1) for i in range(nb)]
    blocks = [[] for _ in range(nb)]
    k = 0
    for i in range(max(dls)):
        for j in range(nb):
            if i < dls[j]:
                blocks[j].append(cw[k])
                k += 1
    eccs = [[] for _ in range(nb)]
    for i in range(ecc):
        for j in range(nb):
            eccs[j].append(cw[k])
            k += 1
    data = []
    for d, e in zip(blocks, eccs):
        if not rs_syndromes_zero(d + e, ecc):
            raise ArithmeticError("BarcodeImage: self-check failed (Reed–Solomon syndromes)")
        data += d
    bitstr = "".join(f"{b:08b}" for b in data)
    pos = 0

    def take(n):
        nonlocal pos
        v = int(bitstr[pos:pos + n], 2)
        pos += n
        return v
    mode = {1: "Numeric", 2: "Alphanumeric", 4: "Byte"}.get(take(4))
    if mode is None:
        raise ArithmeticError("BarcodeImage: self-check failed (mode indicator)")
    n = take(_qr_count_bits(mode, ver))
    if mode == "Byte":
        return bytes(take(8) for _ in range(n)).decode("utf-8")
    if mode == "Numeric":
        out = ""
        while len(out) < n:
            g = min(3, n - len(out))
            out += str(take(3 * g + 1)).zfill(g)
        return out
    out = ""
    while len(out) + 1 < n:
        v = take(11)
        out += QR_ALNUM[v // 45] + QR_ALNUM[v % 45]
    if len(out) < n:
        out += QR_ALNUM[take(6)]
    return out


# ───────────────────────────── EAN / UPC (GS1 §5.2) ─────────────────────────────

EAN_L = ("0001101", "0011001", "0010011", "0111101", "0100011", "0110001", "0101111", "0111011", "0110111", "0001011")
EAN_R = tuple("".join("1" if c == "0" else "0" for c in s) for s in EAN_L)
EAN_G = tuple(s[::-1] for s in EAN_R)
EAN_PARITY = ("LLLLLL", "LLGLGG", "LLGGLG", "LLGGGL", "LGLLGG", "LGGLLG", "LGGGLL", "LGLGLG", "LGLGGL", "LGGLGL")


def gs1_check_digit(digits):
    """GS1 modulo-10 check digit: weights 3, 1, 3, ... from the rightmost data digit."""
    s = sum(int(d) * (3 if i % 2 == 0 else 1) for i, d in enumerate(reversed(digits)))
    return str((10 - s % 10) % 10)


def _gs1_digits(text, n, fmt):
    if not text.isdigit() or not text.isascii():
        raise ValueError(f"BarcodeImage: {fmt} data must consist of digits")
    if len(text) == n - 1:
        return text + gs1_check_digit(text)
    if len(text) != n:
        raise ValueError(f"BarcodeImage: {fmt} needs {n - 1} or {n} digits, not {len(text)}")
    if gs1_check_digit(text[:-1]) != text[-1]:
        raise ValueError(f"BarcodeImage: {text} has an invalid {fmt} check digit (expected "
                         f"{gs1_check_digit(text[:-1])})")
    return text


def ean13_modules(d):
    par = EAN_PARITY[int(d[0])]
    s = "101"
    for c, p in zip(d[1:7], par):
        s += (EAN_L if p == "L" else EAN_G)[int(c)]
    s += "01010"
    for c in d[7:]:
        s += EAN_R[int(c)]
    return s + "101"


def ean8_modules(d):
    return "101" + "".join(EAN_L[int(c)] for c in d[:4]) + "01010" + "".join(EAN_R[int(c)] for c in d[4:]) + "101"


def ean_decode(mods):
    """Digits of an EAN-13 (95 modules) or EAN-8 (67 modules) pattern, with the check digit verified."""
    def guard(s, g):
        if s != g:
            raise ArithmeticError("BarcodeImage: self-check failed (guard pattern)")
    if len(mods) == 95:
        guard(mods[:3], "101"), guard(mods[45:50], "01010"), guard(mods[92:], "101")
        left = [mods[3 + 7 * i:10 + 7 * i] for i in range(6)]
        right = [mods[50 + 7 * i:57 + 7 * i] for i in range(6)]
        digs, par = "", ""
        for s in left:
            if s in EAN_L:
                digs += str(EAN_L.index(s))
                par += "L"
            elif s in EAN_G:
                digs += str(EAN_G.index(s))
                par += "G"
            else:
                raise ArithmeticError("BarcodeImage: self-check failed (left digit)")
        if par not in EAN_PARITY:
            raise ArithmeticError("BarcodeImage: self-check failed (parity pattern)")
        digs = str(EAN_PARITY.index(par)) + digs
    elif len(mods) == 67:
        guard(mods[:3], "101"), guard(mods[31:36], "01010"), guard(mods[64:], "101")
        left = [mods[3 + 7 * i:10 + 7 * i] for i in range(4)]
        right = [mods[36 + 7 * i:43 + 7 * i] for i in range(4)]
        if any(s not in EAN_L for s in left):
            raise ArithmeticError("BarcodeImage: self-check failed (left digit)")
        digs = "".join(str(EAN_L.index(s)) for s in left)
    else:
        raise ArithmeticError("BarcodeImage: self-check failed (length)")
    if any(s not in EAN_R for s in right):
        raise ArithmeticError("BarcodeImage: self-check failed (right digit)")
    digs += "".join(str(EAN_R.index(s)) for s in right)
    if gs1_check_digit(digs[:-1]) != digs[-1]:
        raise ArithmeticError("BarcodeImage: self-check failed (check digit)")
    return digs


# ───────────────────────────── Code 128 (ISO/IEC 15417) ─────────────────────────────

CODE128_WIDTHS = (
    "212222 222122 222221 121223 121322 131222 122213 122312 132212 221213 221312 231212 112232 122132 122231 113222 "
    "123122 123221 223211 221132 221231 213212 223112 312131 311222 321122 321221 312212 322112 322211 212123 212321 "
    "232121 111323 131123 131321 112313 132113 132311 211313 231113 231311 112133 112331 132131 113123 113321 133121 "
    "313121 211331 231131 213113 213311 213131 311123 311321 331121 312113 312311 332111 314111 221411 431111 111224 "
    "111422 121124 121421 141122 141221 112214 112412 122114 122411 142112 142211 241211 221114 413111 241112 134111 "
    "111242 121142 121241 114212 124112 124211 411212 421112 421211 212141 214121 412121 111143 111341 131141 114113 "
    "114311 411113 411311 113141 114131 311141 411131 211412 211214 211232 2331112").split()
C128_START = {"A": 103, "B": 104, "C": 105}
C128_SWITCH = {"A": 101, "B": 100, "C": 99}      # code value that switches to the set (from the other sets)


def _w2m(widths):
    return "".join(("1" if i % 2 == 0 else "0") * int(w) for i, w in enumerate(widths))


def _c128_value(ch, cset):
    o = ord(ch)
    if cset == "B":
        return o - 32 if 32 <= o < 128 else None
    return o + 64 if o < 32 else (o - 32 if o < 96 else None)


def code128_values(text):
    """Symbol values (start, data, checksum) with a simple optimal-for-digits code set choice: set C for runs of
    at least 4 digits (an even number of them), set A for control characters, set B otherwise."""
    if not text:
        raise ValueError("BarcodeImage: Code128 data must not be empty")
    if any(ord(c) > 127 for c in text):
        raise ValueError("BarcodeImage: Code128 encodes ASCII characters only")
    vals = []
    i, cset = 0, None
    n = len(text)

    def digit_run(j):
        k = j
        while k < n and text[k].isdigit():
            k += 1
        return k - j
    while i < n:
        run = digit_run(i)
        if run >= 4 or (run >= 2 and run == n - i and run % 2 == 0 and (cset == "C" or i == 0)):
            use = run - run % 2
            if cset != "C":
                vals.append(C128_START["C"] if cset is None else C128_SWITCH["C"])
                cset = "C"
            for k in range(i, i + use, 2):
                vals.append(int(text[k:k + 2]))
            i += use
            continue
        ch = text[i]
        want = cset if cset in ("A", "B") and _c128_value(ch, cset) is not None else \
            ("A" if ord(ch) < 32 else "B")
        if cset != want:
            vals.append(C128_START[want] if cset is None else C128_SWITCH[want])
            cset = want
        vals.append(_c128_value(ch, cset))
        i += 1
    chk = (vals[0] + sum(k * v for k, v in enumerate(vals[1:], 1))) % 103
    return vals + [chk]


def code128_modules(text):
    vals = code128_values(text)
    return "".join(_w2m(CODE128_WIDTHS[v]) for v in vals) + _w2m(CODE128_WIDTHS[106])


def code128_decode(mods):
    if not mods.endswith(_w2m(CODE128_WIDTHS[106])):
        raise ArithmeticError("BarcodeImage: self-check failed (stop pattern)")
    body = mods[:-13]
    if len(body) % 11:
        raise ArithmeticError("BarcodeImage: self-check failed (length)")
    table = {_w2m(w): k for k, w in enumerate(CODE128_WIDTHS[:106])}
    vals = []
    for i in range(0, len(body), 11):
        v = table.get(body[i:i + 11])
        if v is None:
            raise ArithmeticError("BarcodeImage: self-check failed (symbol)")
        vals.append(v)
    if (vals[0] + sum(k * v for k, v in enumerate(vals[1:-1], 1))) % 103 != vals[-1]:
        raise ArithmeticError("BarcodeImage: self-check failed (checksum)")
    cset = {103: "A", 104: "B", 105: "C"}.get(vals[0])
    if cset is None:
        raise ArithmeticError("BarcodeImage: self-check failed (start symbol)")
    out = ""
    for v in vals[1:-1]:
        if cset == "C" and v < 100:
            out += f"{v:02d}"
        elif v in (99, 100, 101) and not (v == 101 and cset == "A") and not (v == 100 and cset == "B"):
            cset = {99: "C", 100: "B", 101: "A"}[v]
        elif cset == "B":
            out += chr(v + 32)
        elif cset == "A":
            out += chr(v + 32) if v < 64 else chr(v - 64)
        else:
            raise ArithmeticError("BarcodeImage: self-check failed (function code)")
    return out


# ───────────────────────────── Code 39 (ISO/IEC 16388) and ITF (ISO/IEC 16390) ─────────────────────────────

CODE39_CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-. $/+%*"
CODE39_WIDTHS = (
    "111331311 311311113 113311113 313311111 111331113 311331111 113331111 111311313 311311311 113311311 311113113 "
    "113113113 313113111 111133113 311133111 113133111 111113313 311113311 113113311 111133311 311111133 113111133 "
    "313111131 111131133 311131131 113131131 111111333 311111331 113111331 111131331 331111113 133111113 333111111 "
    "131131113 331131111 133131111 131111313 331111311 133111311 131313111 131311131 131113131 111313131 "
    "131131311").split()


def code39_modules(text):
    if not text or any(c not in CODE39_CHARS[:-1] for c in text):
        raise ValueError("BarcodeImage: Code39 data must be uppercase letters, digits or the characters - . $ / + % "
                         "and space")
    return "0".join(_w2m(CODE39_WIDTHS[CODE39_CHARS.index(c)]) for c in "*" + text + "*")


def code39_decode(mods):
    table = {_w2m(w): CODE39_CHARS[k] for k, w in enumerate(CODE39_WIDTHS)}
    parts = []
    i = 0
    while i < len(mods):
        parts.append(table.get(mods[i:i + 15]))
        i += 16
    if None in parts or parts[0] != "*" or parts[-1] != "*":
        raise ArithmeticError("BarcodeImage: self-check failed (Code39 symbol)")
    return "".join(parts[1:-1])


ITF_DIGITS = ("NNWWN", "WNNNW", "NWNNW", "WWNNN", "NNWNW", "WNWNN", "NWWNN", "NNNWW", "WNNWN", "NWNWN")
ITF_WIDE = 3


def itf_modules(text):
    if not text.isdigit() or not text.isascii() or len(text) % 2:
        raise ValueError("BarcodeImage: ITF data must be an even number of digits")
    w = {"N": "1", "W": str(ITF_WIDE)}
    widths = "1111"
    for k in range(0, len(text), 2):
        b, s = ITF_DIGITS[int(text[k])], ITF_DIGITS[int(text[k + 1])]
        widths += "".join(w[x] + w[y] for x, y in zip(b, s))
    widths += str(ITF_WIDE) + "11"
    return _w2m(widths)


def itf_decode(mods):
    import itertools
    widths = [len(list(g)) for _, g in itertools.groupby(mods)]
    if widths[:4] != [1, 1, 1, 1] or widths[-3:] != [ITF_WIDE, 1, 1]:
        raise ArithmeticError("BarcodeImage: self-check failed (ITF start/stop)")
    body = widths[4:-3]
    nw = {1: "N", ITF_WIDE: "W"}
    out = ""
    for k in range(0, len(body), 10):
        grp = body[k:k + 10]
        b = "".join(nw[x] for x in grp[0::2])
        s = "".join(nw[x] for x in grp[1::2])
        if b not in ITF_DIGITS or s not in ITF_DIGITS:
            raise ArithmeticError("BarcodeImage: self-check failed (ITF digit)")
        out += str(ITF_DIGITS.index(b)) + str(ITF_DIGITS.index(s))
    return out


# ───────────────────────────── BarcodeImage ─────────────────────────────

_NOT_SUPPORTED = ("UPCE", "Code93", "Codabar", "Aztec", "DataMatrix", "PDF417")
QR_QUIET, QR_PX = 4, 4                     # quiet zone (modules) and pixels per module by default
LIN_QUIET, LIN_PX, LIN_HEIGHT = 10, 2, 60  # 1D: quiet zone (modules), pixels per module, bar height (pixels)


def _size(spec, fname):
    if spec is None or _is_sym(spec, "Automatic"):
        return None, None
    if K.is_list(spec) and len(spec.args) == 2:
        w = spec.args[0]
        h = spec.args[1]
        return (None if _is_sym(w, "Automatic") else _pos(w, fname)), \
            (None if _is_sym(h, "Automatic") else _pos(h, fname))
    return _pos(spec, fname), None


def _pos(x, fname):
    if isinstance(x, bool) or not isinstance(x, (int, float)) and not (isinstance(x, Expr) and x.head == "Rational"):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"BarcodeImage: the size must be a positive number, not {_short(x)}")
    v = float(S(x)) if not isinstance(x, (int, float)) else float(x)
    if not v > 0:
        raise ValueError("BarcodeImage: the size must be positive")
    return v


def _bit_image(mods2d):
    """Bit image (1 = white) from a 0/1 dark-module array."""
    a = (1 - mods2d).astype(float)[..., None]
    return image_expr(WI.Img(a, "Bit", "Automatic", False))


def barcode_array(text, fmt, level="M", size=(None, None)):
    """(dark-module array in pixels, decoded text) for a barcode."""
    w, h = size
    if fmt == "QR":
        m, ver, mk = qr_matrix(text, level)
        dec = qr_decode(m)
        full = np.pad(m, QR_QUIET)
        n = full.shape[0]
        px = QR_PX if w is None and h is None else max(1, int(round(min(v for v in (w, h) if v is not None) / n)))
        return np.kron(full, np.ones((px, px), dtype=np.int8)), dec
    if fmt in ("EAN13", "UPC"):
        d = _gs1_digits(text, 13 if fmt == "EAN13" else 12, fmt)
        mods = ean13_modules(d if fmt == "EAN13" else "0" + d)
        dec = ean_decode(mods)
        dec = dec if fmt == "EAN13" else dec[1:]
        if dec != d:
            raise ArithmeticError("BarcodeImage: self-check failed (decoded data differs)")
        text = d
    elif fmt == "EAN8":
        d = _gs1_digits(text, 8, fmt)
        mods = ean8_modules(d)
        dec = ean_decode(mods)
        text = d
    elif fmt == "Code128":
        mods = code128_modules(text)
        dec = code128_decode(mods)
    elif fmt == "Code39":
        mods = code39_modules(text)
        dec = code39_decode(mods)
    else:
        mods = itf_modules(text)
        dec = itf_decode(mods)
    row = np.array([int(c) for c in "0" * LIN_QUIET + mods + "0" * LIN_QUIET], dtype=np.int8)
    px = LIN_PX if w is None else max(1, int(round(w / row.size)))
    hh = LIN_HEIGHT if h is None else max(1, int(round(h)))
    return np.tile(np.repeat(row, px), (hh, 1)), dec


@fn("BarcodeImage", area=AREA, status="added",
    notes="BarcodeImage[\"string\", format] and BarcodeImage[\"string\", format, size] give a Bit image (bars and "
          "dark modules 0, background 1). Formats: \"QR\" and {\"QR\", lev} (ISO/IEC 18004 model 2; lev \"L\", "
          "\"M\", \"Q\" or \"H\", default \"M\"; the smallest version that holds the data, a single numeric, "
          "alphanumeric or UTF-8 byte segment, the data mask with the lowest ISO penalty score; 4 pixels per module "
          "and a 4-module quiet zone by default), \"EAN13\" (12 digits plus the computed check digit, or 13 digits "
          "with a valid one), \"UPC\" (UPC-A: 11 or 12 digits), \"EAN8\" (7 or 8 digits), \"Code128\" (ASCII; code "
          "set C for runs of at least 4 digits, A for control characters, B otherwise), \"Code39\" (no check "
          "character) and \"ITF\" (an even number of digits, wide:narrow 3:1). 1D codes use 2 pixels per module, a "
          "10-module quiet zone on both sides and 60 pixel bars by default. size w or {w, h} sets the pixels per "
          "module to Round[w/modules] (at least 1) and, for 1D codes, the bar height h; sizes too small for one "
          "pixel per module give a larger image (as documented). Integer error-correction levels (0-8) and the "
          "formats UPCE, Code93, Codabar, Aztec, DataMatrix and PDF417 raise NotImplementedError. Check: self-check "
          "(every symbol is decoded back from its module pattern — QR format BCH code, Reed–Solomon syndromes and "
          "segment data; EAN guards, parity and check digit; Code128 checksum — and must equal the input, else "
          "ArithmeticError), plus reference codewords in tests.")
def _barcodeimage(s: String, fmt: String | ListOf[Expr], *rest: Real | ListOf[Real] | Option) -> Expr:
    fname = "BarcodeImage"
    pos, o = opts_split(rest, set(), fname)
    if len(pos) > 1:
        raise TypeError(f"{fname}: too many arguments")
    if not isinstance(s, str):
        if isinstance(s, Symbol) or (isinstance(s, Expr) and not K.is_list(s)):
            raise Unevaluated()
        raise TypeError(f"{fname}: the data must be a string, not {_short(s)}")
    level = "M"
    if K.is_list(fmt) and len(fmt.args) == 2 and isinstance(fmt.args[0], str):
        f0, lev = fmt.args
        if f0 in ("Aztec", "PDF417"):
            raise NotImplementedError(f"{fname}: the {f0} format is not supported")
        if f0 != "QR":
            raise ValueError(f"{fname}: error correction levels are given only for QR, Aztec and PDF417")
        if isinstance(lev, int) and not isinstance(lev, bool):
            raise NotImplementedError(f"{fname}: integer error-correction levels are not supported; give \"L\", "
                                      "\"M\", \"Q\" or \"H\"")
        if lev not in ("L", "M", "Q", "H"):
            raise ValueError(f"{fname}: the QR error-correction level must be \"L\", \"M\", \"Q\" or \"H\"")
        level, fmt = lev, "QR"
    if not isinstance(fmt, str):
        if isinstance(fmt, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the format must be a string, not {_short(fmt)}")
    if fmt in _NOT_SUPPORTED:
        raise NotImplementedError(f"{fname}: the {fmt} format is not supported")
    if fmt not in ("QR", "EAN13", "UPC", "EAN8", "Code128", "Code39", "ITF"):
        raise ValueError(f"{fname}: unknown barcode format {fmt!r}")
    size = _size(pos[0] if pos else None, fname)
    arr, dec = barcode_array(s, fmt, level, size)
    want = s if fmt not in ("EAN13", "UPC", "EAN8") else None
    if want is not None and dec != want:
        raise ArithmeticError(f"{fname}: self-check failed (decoded {dec!r}, expected {want!r})")
    return _bit_image(arr)
