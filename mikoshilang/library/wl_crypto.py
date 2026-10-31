"""Wolfram batch W13 (part 2): cryptography with Wolfram semantics, in pure Python (standard library only).

* Symmetric: ``GenerateSymmetricKey``/``SymmetricKey``, ``Encrypt``/``Decrypt``/``EncryptedObject``,
  ``EncryptFile``/``DecryptFile``.  Ciphers AES128/AES192/AES256 (FIPS-197), Blowfish (S-boxes from the hex digits
  of pi, as specified by Schneier) and RC4; block modes ECB, CBC, CFB (CFB-128/64), OFB and CTR (NIST SP 800-38A);
  padding Automatic (PKCS#7 for ECB/CBC) or None.  DES, CAST5 and IDEA are not provided (clear error).
  Strings are encrypted as UTF-8, ByteArrays literally and other expressions as their Compress form.
* Key derivation: ``GenerateDerivedKey``/``DerivedKey``/``VerifyDerivedKey`` with scrypt (RFC 7914, hashlib) and
  the PHC string form; Argon2 is not available (clear error).
* Public key: ``GenerateAsymmetricKeyPair``/``PrivateKey``/``PublicKey`` for RSA (Miller-Rabin key generation,
  PKCS#1 v1.5 and OAEP-SHA1 encryption, PKCS#1 v1.5 signatures), elliptic curves (secp256k1, secp256r1 =
  prime256v1, secp384r1, secp521r1, secp224r1, secp192r1 = prime192v1; ECDSA per FIPS 186-4 with RFC 6979
  deterministic nonces, BIP-340 Schnorr on secp256k1) and Edwards curves (ed25519 per RFC 8032).
* ``GenerateDigitalSignature``/``DigitalSignature``/``VerifyDigitalSignature``, ``GenerateFileSignature``/
  ``VerifyFileSignature``; ``$CryptographicEllipticCurveNames``.
* PEM import/export of keys (PKCS#1, PKCS#8, SPKI, SEC1) and X.509 certificates (``SecurityCertificate``,
  RFC 5280), with ``ImportedObject`` for other PEM entries.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
import struct

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .. import formatting as F
from .core import Unevaluated, fn, record, L, is_list
from .wl_common import ev, is_sym
from . import wl_importexport as IE
from .wl_importexport import (byte_array, is_byte_array, ba_bytes, bytes_arg, assoc_get, split_opts, file_arg,
                              _is_rule, _assoc)

AREA = "cryptography (W13)"
NULL = None
A = _assoc


def _ba(b):
    return byte_array(b)


# ═══════════════════════════════════════ hashing ═══════════════════════════════════════

_KECCAK_RC = [0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000, 0x000000000000808B,
              0x0000000080000001, 0x8000000080008081, 0x8000000000008009, 0x000000000000008A, 0x0000000000000088,
              0x0000000080008009, 0x000000008000000A, 0x000000008000808B, 0x800000000000008B, 0x8000000000008089,
              0x8000000000008003, 0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
              0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008]
_KECCAK_ROT = [[0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61], [28, 55, 25, 21, 56],
               [27, 20, 39, 8, 14]]
_M64 = (1 << 64) - 1


def _keccak_f(st):
    for rnd in range(24):
        c = [st[x][0] ^ st[x][1] ^ st[x][2] ^ st[x][3] ^ st[x][4] for x in range(5)]
        d = [c[(x - 1) % 5] ^ (((c[(x + 1) % 5] << 1) | (c[(x + 1) % 5] >> 63)) & _M64) for x in range(5)]
        for x in range(5):
            for y in range(5):
                st[x][y] ^= d[x]
        b = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                r = _KECCAK_ROT[x][y]
                v = st[x][y]
                b[y][(2 * x + 3 * y) % 5] = ((v << r) | (v >> (64 - r))) & _M64 if r else v
        for x in range(5):
            for y in range(5):
                st[x][y] = b[x][y] ^ ((~b[(x + 1) % 5][y]) & b[(x + 2) % 5][y])
        st[0][0] ^= _KECCAK_RC[rnd]


def keccak(data, bits):
    """Original Keccak (padding 0x01, as used by Ethereum), not SHA-3."""
    rate = 200 - 2 * (bits // 8)
    msg = bytearray(data) + b"\x01"
    while len(msg) % rate:
        msg.append(0)
    msg[-1] |= 0x80
    st = [[0] * 5 for _ in range(5)]
    for off in range(0, len(msg), rate):
        block = msg[off:off + rate]
        for i in range(rate // 8):
            x, y = i % 5, i // 5
            st[x][y] ^= int.from_bytes(block[8 * i:8 * i + 8], "little")
        _keccak_f(st)
    out = bytearray()
    while len(out) < bits // 8:
        for i in range(rate // 8):
            x, y = i % 5, i // 5
            out += st[x][y].to_bytes(8, "little")
        if len(out) < bits // 8:
            _keccak_f(st)
    return bytes(out[:bits // 8])


def _ripemd160_py(data):
    # RIPEMD-160 (Dobbertin, Bosselaers, Preneel), used when hashlib lacks it
    def rol(x, n):
        return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF
    fs = [lambda x, y, z: x ^ y ^ z, lambda x, y, z: (x & y) | (~x & z), lambda x, y, z: (x | ~y) ^ z,
          lambda x, y, z: (x & z) | (y & ~z), lambda x, y, z: x ^ (y | ~z)]
    KL_ = [0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E]
    KR_ = [0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000]
    RL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 7, 4, 13, 1, 10, 6, 15, 3, 12, 0, 9, 5, 2, 14, 11, 8,
          3, 10, 14, 4, 9, 15, 8, 1, 2, 7, 0, 6, 13, 11, 5, 12, 1, 9, 11, 10, 0, 8, 12, 4, 13, 3, 7, 15, 14, 5, 6, 2,
          4, 0, 5, 9, 7, 12, 2, 10, 14, 1, 3, 8, 11, 6, 15, 13]
    RR = [5, 14, 7, 0, 9, 2, 11, 4, 13, 6, 15, 8, 1, 10, 3, 12, 6, 11, 3, 7, 0, 13, 5, 10, 14, 15, 8, 12, 4, 9, 1, 2,
          15, 5, 1, 3, 7, 14, 6, 9, 11, 8, 12, 2, 10, 0, 4, 13, 8, 6, 4, 1, 3, 11, 15, 0, 5, 12, 2, 13, 9, 7, 10, 14,
          12, 15, 10, 4, 1, 5, 8, 7, 6, 2, 13, 14, 0, 3, 9, 11]
    SL = [11, 14, 15, 12, 5, 8, 7, 9, 11, 13, 14, 15, 6, 7, 9, 8, 7, 6, 8, 13, 11, 9, 7, 15, 7, 12, 15, 9, 11, 7, 13,
          12, 11, 13, 6, 7, 14, 9, 13, 15, 14, 8, 13, 6, 5, 12, 7, 5, 11, 12, 14, 15, 14, 15, 9, 8, 9, 14, 5, 6, 8, 6,
          5, 12, 9, 15, 5, 11, 6, 8, 13, 12, 5, 12, 13, 14, 11, 8, 5, 6]
    SR = [8, 9, 9, 11, 13, 15, 15, 5, 7, 7, 8, 11, 14, 14, 12, 6, 9, 13, 15, 7, 12, 8, 9, 11, 7, 7, 12, 7, 6, 15, 13,
          11, 9, 7, 15, 11, 8, 6, 6, 14, 12, 13, 5, 14, 13, 13, 7, 5, 15, 5, 8, 11, 14, 14, 6, 14, 6, 9, 12, 9, 12, 5,
          15, 8, 8, 5, 12, 9, 12, 5, 14, 6, 8, 13, 6, 5, 15, 13, 11, 11]
    h = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0]
    msg = bytearray(data) + b"\x80"
    while len(msg) % 64 != 56:
        msg.append(0)
    msg += struct.pack("<Q", (8 * len(data)) & 0xFFFFFFFFFFFFFFFF)
    for off in range(0, len(msg), 64):
        X = struct.unpack("<16I", msg[off:off + 64])
        al, bl, cl, dl, el = h
        ar, br, cr, dr, er = h
        for j in range(80):
            r = j // 16
            t = (rol((al + (fs[r](bl, cl, dl) & 0xFFFFFFFF) + X[RL[j]] + KL_[r]) & 0xFFFFFFFF, SL[j]) + el) \
                & 0xFFFFFFFF
            al, el, dl, cl, bl = el, dl, rol(cl, 10), bl, t
            t = (rol((ar + (fs[4 - r](br, cr, dr) & 0xFFFFFFFF) + X[RR[j]] + KR_[r]) & 0xFFFFFFFF, SR[j]) + er) \
                & 0xFFFFFFFF
            ar, er, dr, cr, br = er, dr, rol(cr, 10), br, t
        t = (h[1] + cl + dr) & 0xFFFFFFFF
        h[1] = (h[2] + dl + er) & 0xFFFFFFFF
        h[2] = (h[3] + el + ar) & 0xFFFFFFFF
        h[3] = (h[4] + al + br) & 0xFFFFFFFF
        h[4] = (h[0] + bl + cr) & 0xFFFFFFFF
        h[0] = t
    return struct.pack("<5I", *h)


_HASHLIB = {"MD5": "md5", "SHA1": "sha1", "SHA": "sha1", "SHA224": "sha224", "SHA256": "sha256",
            "SHA384": "sha384", "SHA512": "sha512", "SHA3-224": "sha3_224", "SHA3-256": "sha3_256",
            "SHA3-384": "sha3_384", "SHA3-512": "sha3_512", "BLAKE2b": "blake2b", "BLAKE2s": "blake2s"}


def digest(method, data):
    if method in _HASHLIB:
        return hashlib.new(_HASHLIB[method], data).digest()
    if method == "SHA256SHA256":
        return hashlib.sha256(hashlib.sha256(data).digest()).digest()
    if method == "RIPEMD160":
        try:
            return hashlib.new("ripemd160", data).digest()
        except ValueError:
            return _ripemd160_py(data)
    if method == "RIPEMD160SHA256":
        return digest("RIPEMD160", hashlib.sha256(data).digest())
    if method in ("Keccak224", "Keccak256", "Keccak384", "Keccak512"):
        return keccak(data, int(method[6:]))
    raise ValueError(f"unsupported hashing method {method!r}")


HASH_METHODS = sorted(list(_HASHLIB) + ["SHA256SHA256", "RIPEMD160", "RIPEMD160SHA256", "Keccak224", "Keccak256",
                                        "Keccak384", "Keccak512"])


def expr_bytes(x, fname):
    """The bytes Hash[x, ...] hashes: strings as UTF-8, ByteArrays literally, other expressions as the UTF-8 text
    of their FullForm."""
    if isinstance(x, str):
        return x.encode("utf-8")
    if is_byte_array(x):
        return ba_bytes(x, fname)
    return F.full_form(x).encode("utf-8")


def message_digest(x, method, fname):
    if method is None or is_sym(method, "None"):
        if is_byte_array(x):
            return ba_bytes(x, fname)
        if isinstance(x, str) and re.fullmatch(r"(?:[0-9A-Fa-f]{2})+", x):
            return bytes.fromhex(x)
        raise ValueError(f"{fname}: with \"HashingMethod\" -> None the message must be a ByteArray or a hex string")
    if not isinstance(method, str) or method not in HASH_METHODS:
        raise ValueError(f"{fname}: {F.input_form(method)} is not a supported hashing method")
    return digest(method, expr_bytes(x, fname))


# ═══════════════════════════════════════ block ciphers ═══════════════════════════════════════

def _aes_tables():
    sbox = [0] * 256
    p = q = 1
    while True:
        p = p ^ ((p << 1) & 0xFF) ^ (0x1B if p & 0x80 else 0)
        q ^= q << 1
        q ^= q << 2
        q ^= q << 4
        q &= 0xFF
        if q & 0x80:
            q ^= 0x09
        x = q ^ ((q << 1) | (q >> 7)) ^ ((q << 2) | (q >> 6)) ^ ((q << 3) | (q >> 5)) ^ ((q << 4) | (q >> 4))
        sbox[p] = (x ^ 0x63) & 0xFF
        if p == 1:
            break
    sbox[0] = 0x63
    inv = [0] * 256
    for i, v in enumerate(sbox):
        inv[v] = i
    return sbox, inv


_SBOX, _INV_SBOX = _aes_tables()


def _xt(a):
    return ((a << 1) ^ 0x1B) & 0xFF if a & 0x80 else a << 1


def _gmul(a, b):
    r = 0
    while b:
        if b & 1:
            r ^= a
        a = _xt(a)
        b >>= 1
    return r


_MUL = {c: [_gmul(i, c) for i in range(256)] for c in (2, 3, 9, 11, 13, 14)}


class AES:
    block = 16

    def __init__(self, key):
        if len(key) not in (16, 24, 32):
            raise ValueError("AES keys must be 16, 24 or 32 bytes")
        nk = len(key) // 4
        self.nr = nk + 6
        w = [list(key[4 * i:4 * i + 4]) for i in range(nk)]
        rcon = 1
        for i in range(nk, 4 * (self.nr + 1)):
            t = list(w[i - 1])
            if i % nk == 0:
                t = t[1:] + t[:1]
                t = [_SBOX[b] for b in t]
                t[0] ^= rcon
                rcon = _xt(rcon)
            elif nk > 6 and i % nk == 4:
                t = [_SBOX[b] for b in t]
            w.append([w[i - nk][j] ^ t[j] for j in range(4)])
        self.rk = [sum((w[4 * r + c] for c in range(4)), []) for r in range(self.nr + 1)]

    def encrypt_block(self, b):
        s = [b[i] ^ self.rk[0][i] for i in range(16)]
        m2, m3 = _MUL[2], _MUL[3]
        for r in range(1, self.nr + 1):
            s = [_SBOX[v] for v in s]
            s = [s[(i + 4 * (i % 4)) % 16] for i in range(16)]       # ShiftRows (column-major state)
            if r != self.nr:
                t = []
                for c in range(4):
                    a0, a1, a2, a3 = s[4 * c:4 * c + 4]
                    t += [m2[a0] ^ m3[a1] ^ a2 ^ a3, a0 ^ m2[a1] ^ m3[a2] ^ a3, a0 ^ a1 ^ m2[a2] ^ m3[a3],
                          m3[a0] ^ a1 ^ a2 ^ m2[a3]]
                s = t
            s = [s[i] ^ self.rk[r][i] for i in range(16)]
        return bytes(s)

    def decrypt_block(self, b):
        s = [b[i] ^ self.rk[self.nr][i] for i in range(16)]
        m9, m11, m13, m14 = _MUL[9], _MUL[11], _MUL[13], _MUL[14]
        for r in range(self.nr - 1, -1, -1):
            s = [s[(i - 4 * (i % 4)) % 16] for i in range(16)]       # InvShiftRows
            s = [_INV_SBOX[v] for v in s]
            s = [s[i] ^ self.rk[r][i] for i in range(16)]
            if r != 0:
                t = []
                for c in range(4):
                    a0, a1, a2, a3 = s[4 * c:4 * c + 4]
                    t += [m14[a0] ^ m11[a1] ^ m13[a2] ^ m9[a3], m9[a0] ^ m14[a1] ^ m11[a2] ^ m13[a3],
                          m13[a0] ^ m9[a1] ^ m14[a2] ^ m11[a3], m11[a0] ^ m13[a1] ^ m9[a2] ^ m14[a3]]
                s = t
        return bytes(s)


_BF_CACHE = []


def _blowfish_constants():
    """P-array and S-boxes: the fractional hex digits of pi (18 + 4*256 32-bit words), computed with mpmath."""
    if not _BF_CACHE:
        import mpmath
        nwords = 18 + 1024
        with mpmath.workprec(32 * nwords + 64):
            frac = mpmath.pi - 3
            v = int(mpmath.floor(frac * mpmath.mpf(2) ** (32 * nwords)))
        words = [(v >> (32 * (nwords - 1 - i))) & 0xFFFFFFFF for i in range(nwords)]
        _BF_CACHE.append((words[:18], [words[18 + 256 * k:18 + 256 * (k + 1)] for k in range(4)]))
    return _BF_CACHE[0]


class Blowfish:
    block = 8

    def __init__(self, key):
        if not 4 <= len(key) <= 56:
            raise ValueError("Blowfish keys must be 32 to 448 bits")
        P0, S0 = _blowfish_constants()
        self.P = list(P0)
        self.S = [list(s) for s in S0]
        kl = len(key)
        for i in range(18):
            k = 0
            for j in range(4):
                k = (k << 8) | key[(4 * i + j) % kl]
            self.P[i] ^= k
        l = r = 0
        for i in range(0, 18, 2):
            l, r = self._enc(l, r)
            self.P[i], self.P[i + 1] = l, r
        for s in range(4):
            for i in range(0, 256, 2):
                l, r = self._enc(l, r)
                self.S[s][i], self.S[s][i + 1] = l, r

    def _f(self, x):
        S = self.S
        return ((((S[0][x >> 24] + S[1][(x >> 16) & 0xFF]) & 0xFFFFFFFF) ^ S[2][(x >> 8) & 0xFF]) + S[3][x & 0xFF]) \
            & 0xFFFFFFFF

    def _enc(self, l, r):
        P = self.P
        for i in range(16):
            l ^= P[i]
            r ^= self._f(l)
            l, r = r, l
        l, r = r, l
        r ^= P[16]
        l ^= P[17]
        return l, r

    def _dec(self, l, r):
        P = self.P
        for i in range(17, 1, -1):
            l ^= P[i]
            r ^= self._f(l)
            l, r = r, l
        l, r = r, l
        r ^= P[1]
        l ^= P[0]
        return l, r

    def encrypt_block(self, b):
        l, r = struct.unpack(">II", b)
        return struct.pack(">II", *self._enc(l, r))

    def decrypt_block(self, b):
        l, r = struct.unpack(">II", b)
        return struct.pack(">II", *self._dec(l, r))


def rc4(key, data):
    S = list(range(256))
    j = 0
    for i in range(256):
        j = (j + S[i] + key[i % len(key)]) & 0xFF
        S[i], S[j] = S[j], S[i]
    out = bytearray()
    i = j = 0
    for c in data:
        i = (i + 1) & 0xFF
        j = (j + S[i]) & 0xFF
        S[i], S[j] = S[j], S[i]
        out.append(c ^ S[(S[i] + S[j]) & 0xFF])
    return bytes(out)


CIPHERS = {"AES128": (16, 16, False), "AES192": (24, 16, False), "AES256": (32, 16, False),
           "Blowfish": (32, 8, True), "RC4": (32, 0, True)}
UNSUPPORTED_CIPHERS = ("DES", "CAST5", "IDEA")


def _block_cipher(cipher, key):
    if cipher.startswith("AES"):
        return AES(key)
    if cipher == "Blowfish":
        return Blowfish(key)
    raise ValueError(cipher)


def _xor(a, b):
    return bytes(x ^ y for x, y in zip(a, b))


def sym_encrypt(cipher, key, data, mode, iv, padding):
    if cipher == "RC4":
        return rc4(key, data)
    c = _block_cipher(cipher, key)
    bs = c.block
    if mode in ("ECB", "CBC"):
        if padding:
            n = bs - len(data) % bs
            data = data + bytes([n]) * n
        elif len(data) % bs:
            raise ValueError(f"Encrypt: with \"Padding\" -> None the data must be a multiple of {bs} bytes")
        out, prev = [], iv
        for i in range(0, len(data), bs):
            blk = data[i:i + bs]
            if mode == "CBC":
                blk = _xor(blk, prev)
            e = c.encrypt_block(blk)
            out.append(e)
            prev = e
        return b"".join(out)
    if mode in ("CFB", "OFB"):
        out, reg = [], iv
        for i in range(0, len(data), bs):
            ks = c.encrypt_block(reg)
            blk = data[i:i + bs]
            e = _xor(blk, ks)
            out.append(e)
            reg = (e + ks[len(e):]) if mode == "CFB" else ks
        return b"".join(out)
    if mode == "CTR":
        out, ctr = [], int.from_bytes(iv, "big")
        for i in range(0, len(data), bs):
            ks = c.encrypt_block((ctr % (1 << (8 * bs))).to_bytes(bs, "big"))
            out.append(_xor(data[i:i + bs], ks))
            ctr += 1
        return b"".join(out)
    raise ValueError(f"unknown block mode {mode}")


def sym_decrypt(cipher, key, data, mode, iv, padding):
    if cipher == "RC4":
        return rc4(key, data)
    c = _block_cipher(cipher, key)
    bs = c.block
    if mode in ("ECB", "CBC"):
        if len(data) % bs:
            raise ValueError("Decrypt: the encrypted data has an invalid length (wrong key or corrupt data)")
        out, prev = [], iv
        for i in range(0, len(data), bs):
            blk = data[i:i + bs]
            d = c.decrypt_block(blk)
            if mode == "CBC":
                d = _xor(d, prev)
            out.append(d)
            prev = blk
        res = b"".join(out)
        if padding:
            n = res[-1] if res else 0
            if not 1 <= n <= bs or res[-n:] != bytes([n]) * n:
                raise ValueError("Decrypt::failed: the data could not be decrypted (wrong key or corrupt data)")
            res = res[:-n]
        return res
    if mode in ("CFB", "OFB"):
        out, reg = [], iv
        for i in range(0, len(data), bs):
            ks = c.encrypt_block(reg)
            blk = data[i:i + bs]
            out.append(_xor(blk, ks))
            reg = (blk + ks[len(blk):]) if mode == "CFB" else ks
        return b"".join(out)
    return sym_encrypt(cipher, key, data, mode, iv, False)     # CTR is symmetric
