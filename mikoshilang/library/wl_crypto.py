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


# ═══════════════════════════════════════ RSA ═══════════════════════════════════════

_SMALL_PRIMES = [p for p in range(3, 2000, 2) if all(p % q for q in range(3, int(p ** 0.5) + 1, 2))]


def _probable_prime(n, rounds=40):
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def _random_prime(bits, e):
    while True:
        c = secrets.randbits(bits) | (1 << (bits - 1)) | (1 << (bits - 2)) | 1
        if _probable_prime(c) and (c - 1) % e != 0 and __import__("math").gcd(e, c - 1) == 1:
            return c


def rsa_generate(bits, e):
    import math
    if bits < 512:
        raise ValueError("GenerateAsymmetricKeyPair: RSA keys must have at least 512 bits")
    if e < 3 or e % 2 == 0:
        raise ValueError("GenerateAsymmetricKeyPair: the RSA public exponent must be an odd integer >= 3")
    while True:
        p = _random_prime(bits // 2, e)
        q = _random_prime(bits - bits // 2, e)
        n = p * q
        if p != q and n.bit_length() == bits:
            phi = (p - 1) * (q - 1)
            if math.gcd(e, phi) == 1:
                d = pow(e, -1, (p - 1) * (q - 1) // math.gcd(p - 1, q - 1))
                return n, e, d


def _i2osp(x, k):
    if x >= 1 << (8 * k):
        raise ValueError("integer too large")
    return x.to_bytes(k, "big")


def _modlen(n):
    return (n.bit_length() + 7) // 8


def _mgf1(seed, length, h=hashlib.sha1):
    out, c = b"", 0
    while len(out) < length:
        out += h(seed + c.to_bytes(4, "big")).digest()
        c += 1
    return out[:length]


def rsa_pad(data, k, padding, block_type, fname):
    if padding == "PKCS1":
        if len(data) > k - 11:
            raise ValueError(f"{fname}: the data is too long for RSA with PKCS1 padding (at most {k - 11} bytes)")
        if block_type == 2:
            ps = b""
            while len(ps) < k - 3 - len(data):
                b = secrets.token_bytes(1)
                if b != b"\x00":
                    ps += b
        else:
            ps = b"\xff" * (k - 3 - len(data))
        return b"\x00" + bytes([block_type]) + ps + b"\x00" + data
    if padding == "OAEP":
        hl = 20
        if len(data) > k - 2 * hl - 2:
            raise ValueError(f"{fname}: the data is too long for RSA with OAEP padding (at most {k - 2 * hl - 2} "
                             f"bytes)")
        lh = hashlib.sha1(b"").digest()
        db = lh + b"\x00" * (k - len(data) - 2 * hl - 2) + b"\x01" + data
        seed = secrets.token_bytes(hl)
        mdb = _xor(db, _mgf1(seed, k - hl - 1))
        ms = _xor(seed, _mgf1(mdb, hl))
        return b"\x00" + ms + mdb
    if padding is None:
        if len(data) != k:
            raise ValueError(f"{fname}: with \"Padding\" -> None the data must be exactly {k} bytes")
        return data
    raise ValueError(f"{fname}: unknown padding {padding!r}")


def rsa_unpad(em, k, padding, block_type, fname):
    bad = ValueError(f"{fname}::failed: the data could not be decrypted (wrong key or corrupt data)")
    if padding == "PKCS1":
        if len(em) != k or em[0] != 0 or em[1] != block_type:
            raise bad
        i = em.find(b"\x00", 2)
        if i < 10:
            raise bad
        return em[i + 1:]
    if padding == "OAEP":
        hl = 20
        if em[0] != 0:
            raise bad
        ms, mdb = em[1:1 + hl], em[1 + hl:]
        seed = _xor(ms, _mgf1(mdb, hl))
        db = _xor(mdb, _mgf1(seed, k - hl - 1))
        if db[:hl] != hashlib.sha1(b"").digest():
            raise bad
        rest = db[hl:].lstrip(b"\x00")
        if not rest or rest[0] != 1:
            raise bad
        return rest[1:]
    return em


_DIGEST_INFO = {"MD5": "3020300c06082a864886f70d020505000410", "SHA1": "3021300906052b0e03021a05000414",
                "SHA224": "302d300d06096086480165030402040500041c", "SHA256": "3031300d060960864801650304020105000420",
                "SHA384": "3041300d060960864801650304020205000430", "SHA512": "3051300d060960864801650304020305000440",
                "SHA3-224": "302d300d06096086480165030402070500041c",
                "SHA3-256": "3031300d060960864801650304020805000420",
                "SHA3-384": "3041300d060960864801650304020905000430",
                "SHA3-512": "3051300d060960864801650304020a05000440",
                "RIPEMD160": "3021300906052b2403020105000414"}


def rsa_sign(n, d, h, hashname, fname):
    if hashname not in _DIGEST_INFO:
        raise ValueError(f"{fname}: {hashname} is not supported for RSA signatures")
    k = _modlen(n)
    t = bytes.fromhex(_DIGEST_INFO[hashname]) + h
    if k < len(t) + 11:
        raise ValueError(f"{fname}: the RSA key is too small for this hash")
    em = b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t
    return _i2osp(pow(int.from_bytes(em, "big"), d, n), k)


def rsa_verify(n, e, h, hashname, sig):
    k = _modlen(n)
    if len(sig) != k or hashname not in _DIGEST_INFO:
        return False
    s = int.from_bytes(sig, "big")
    if s >= n:
        return False
    em = _i2osp(pow(s, e, n), k)
    t = bytes.fromhex(_DIGEST_INFO[hashname]) + h
    return hmac.compare_digest(em, b"\x00\x01" + b"\xff" * (k - len(t) - 3) + b"\x00" + t)


# ═══════════════════════════════════════ elliptic curves ═══════════════════════════════════════

class Curve:
    def __init__(self, name, p, a, b, gx, gy, n, oid):
        self.name, self.p, self.a, self.b, self.G, self.n, self.oid = name, p, a, b, (gx, gy), n, oid
        self.size = (p.bit_length() + 7) // 8

    def on_curve(self, P):
        if P is None:
            return True
        x, y = P
        return 0 <= x < self.p and 0 <= y < self.p and (y * y - x * x * x - self.a * x - self.b) % self.p == 0

    # Jacobian coordinates
    def _dbl(self, P):
        X, Y, Z = P
        if Y == 0:
            return (0, 1, 0)
        p = self.p
        S = 4 * X * Y * Y % p
        M = (3 * X * X + self.a * pow(Z, 4, p)) % p
        X2 = (M * M - 2 * S) % p
        return (X2, (M * (S - X2) - 8 * pow(Y, 4, p)) % p, 2 * Y * Z % p)

    def _add(self, P, Q):
        if P[2] == 0:
            return Q
        if Q[2] == 0:
            return P
        p = self.p
        Z1Z1, Z2Z2 = P[2] * P[2] % p, Q[2] * Q[2] % p
        U1, U2 = P[0] * Z2Z2 % p, Q[0] * Z1Z1 % p
        S1, S2 = P[1] * Q[2] * Z2Z2 % p, Q[1] * P[2] * Z1Z1 % p
        if U1 == U2:
            return self._dbl(P) if S1 == S2 else (0, 1, 0)
        H, R = (U2 - U1) % p, (S2 - S1) % p
        H2 = H * H % p
        H3 = H * H2 % p
        X3 = (R * R - H3 - 2 * U1 * H2) % p
        return (X3, (R * (U1 * H2 - X3) - S1 * H3) % p, H * P[2] * Q[2] % p)

    def mul(self, k, P=None):
        P = self.G if P is None else P
        if P is None or k % self.n == 0:
            return None
        R, Q = (0, 1, 0), (P[0], P[1], 1)
        for bit in bin(k)[2:]:
            R = self._dbl(R)
            if bit == "1":
                R = self._add(R, Q)
        return self._affine(R)

    def add(self, P, Q):
        if P is None:
            return Q
        if Q is None:
            return P
        return self._affine(self._add((P[0], P[1], 1), (Q[0], Q[1], 1)))

    def _affine(self, R):
        if R[2] == 0:
            return None
        zi = pow(R[2], -1, self.p)
        return (R[0] * zi * zi % self.p, R[1] * zi * zi * zi % self.p)

    def lift_x(self, x, odd):
        p = self.p
        y2 = (x * x * x + self.a * x + self.b) % p
        if p % 4 == 3:
            y = pow(y2, (p + 1) // 4, p)
        else:
            y = _tonelli(y2, p)
        if y is None or y * y % p != y2:
            raise ValueError("the point is not on the curve")
        return (x, y if (y & 1) == odd else p - y)

    def encode(self, P, compressed):
        x, y = P
        if compressed:
            return bytes([2 + (y & 1)]) + x.to_bytes(self.size, "big")
        return b"\x04" + x.to_bytes(self.size, "big") + y.to_bytes(self.size, "big")

    def decode(self, b, fname="PublicKey"):
        if len(b) == 1 + 2 * self.size and b[0] == 4:
            P = (int.from_bytes(b[1:1 + self.size], "big"), int.from_bytes(b[1 + self.size:], "big"))
        elif len(b) == 1 + self.size and b[0] in (2, 3):
            try:
                P = self.lift_x(int.from_bytes(b[1:], "big"), b[0] - 2)
            except ValueError:
                raise ValueError(f"{fname}: the encoded point is not on {self.name}")
        else:
            raise ValueError(f"{fname}: invalid encoded {self.name} curve point")
        if not self.on_curve(P):
            raise ValueError(f"{fname}: the encoded point is not on {self.name}")
        return P


def _tonelli(n, p):
    if pow(n, (p - 1) // 2, p) != 1:
        return None if n % p else 0
    q, s = p - 1, 0
    while q % 2 == 0:
        q //= 2
        s += 1
    z = 2
    while pow(z, (p - 1) // 2, p) != p - 1:
        z += 1
    m, c, t, r = s, pow(z, q, p), pow(n, q, p), pow(n, (q + 1) // 2, p)
    while t != 1:
        i, t2 = 0, t
        while t2 != 1:
            t2 = t2 * t2 % p
            i += 1
        b = pow(c, 1 << (m - i - 1), p)
        m, c, t, r = i, b * b % p, t * b * b % p, r * b % p
    return r


def _h(s):
    return int(s.replace(" ", ""), 16)


_P256 = _h("FFFFFFFF 00000001 00000000 00000000 00000000 FFFFFFFF FFFFFFFF FFFFFFFF")
_P384 = 2 ** 384 - 2 ** 128 - 2 ** 96 + 2 ** 32 - 1
_P521 = 2 ** 521 - 1
_P224 = 2 ** 224 - 2 ** 96 + 1
_P192 = 2 ** 192 - 2 ** 64 - 1
CURVES = {
    "secp256k1": Curve("secp256k1", _h("FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2F"), 0, 7,
                       _h("79BE667E F9DCBBAC 55A06295 CE870B07 029BFCDB 2DCE28D9 59F2815B 16F81798"),
                       _h("483ADA77 26A3C465 5DA4FBFC 0E1108A8 FD17B448 A6855419 9C47D08F FB10D4B8"),
                       _h("FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141"),
                       "1.3.132.0.10"),
    "secp256r1": Curve("secp256r1", _P256, _P256 - 3,
                       _h("5AC635D8 AA3A93E7 B3EBBD55 769886BC 651D06B0 CC53B0F6 3BCE3C3E 27D2604B"),
                       _h("6B17D1F2 E12C4247 F8BCE6E5 63A440F2 77037D81 2DEB33A0 F4A13945 D898C296"),
                       _h("4FE342E2 FE1A7F9B 8EE7EB4A 7C0F9E16 2BCE3357 6B315ECE CBB64068 37BF51F5"),
                       _h("FFFFFFFF 00000000 FFFFFFFF FFFFFFFF BCE6FAAD A7179E84 F3B9CAC2 FC632551"),
                       "1.2.840.10045.3.1.7"),
    "secp384r1": Curve("secp384r1", _P384, _P384 - 3,
                       _h("B3312FA7 E23EE7E4 988E056B E3F82D19 181D9C6E FE814112 0314088F 5013875A C656398D 8A2ED19D "
                          "2A85C8ED D3EC2AEF"),
                       _h("AA87CA22 BE8B0537 8EB1C71E F320AD74 6E1D3B62 8BA79B98 59F741E0 82542A38 5502F25D BF55296C "
                          "3A545E38 72760AB7"),
                       _h("3617DE4A 96262C6F 5D9E98BF 9292DC29 F8F41DBD 289A147C E9DA3113 B5F0B8C0 0A60B1CE 1D7E819D "
                          "7A431D7C 90EA0E5F"),
                       _h("FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF C7634D81 F4372DDF 581A0DB2 48B0A77A "
                          "ECEC196A CCC52973"), "1.3.132.0.34"),
    "secp521r1": Curve("secp521r1", _P521, _P521 - 3,
                       _h("0051 953EB961 8E1C9A1F 929A21A0 B68540EE A2DA725B 99B315F3 B8B48991 8EF109E1 56193951 "
                          "EC7E937B 1652C0BD 3BB1BF07 3573DF88 3D2C34F1 EF451FD4 6B503F00"),
                       _h("00C6 858E06B7 0404E9CD 9E3ECB66 2395B442 9C648139 053FB521 F828AF60 6B4D3DBA A14B5E77 "
                          "EFE75928 FE1DC127 A2FFA8DE 3348B3C1 856A429B F97E7E31 C2E5BD66"),
                       _h("0118 39296A78 9A3BC004 5C8A5FB4 2C7D1BD9 98F54449 579B4468 17AFBD17 273E662C 97EE7299 "
                          "5EF42640 C550B901 3FAD0761 353C7086 A272C240 88BE9476 9FD16650"),
                       _h("01FF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFA 51868783 "
                          "BF2F966B 7FCC0148 F709A5D0 3BB5C9B8 899C47AE BB6FB71E 91386409"), "1.3.132.0.35"),
    "secp224r1": Curve("secp224r1", _P224, _P224 - 3,
                       _h("B4050A85 0C04B3AB F5413256 5044B0B7 D7BFD8BA 270B3943 2355FFB4"),
                       _h("B70E0CBD 6BB4BF7F 321390B9 4A03C1D3 56C21122 343280D6 115C1D21"),
                       _h("BD376388 B5F723FB 4C22DFE6 CD4375A0 5A074764 44D58199 85007E34"),
                       _h("FFFFFFFF FFFFFFFF FFFFFFFF FFFF16A2 E0B8F03E 13DD2945 5C5C2A3D"), "1.3.132.0.33"),
    "secp192r1": Curve("secp192r1", _P192, _P192 - 3,
                       _h("64210519 E59C80E7 0FA7E9AB 72243049 FEB8DEEC C146B9B1"),
                       _h("188DA80E B03090F6 7CBF20EB 43A18800 F4FF0AFD 82FF1012"),
                       _h("07192B95 FFC8DA78 631011ED 6B24CDD5 73F977A1 1E794811"),
                       _h("FFFFFFFF FFFFFFFF FFFFFFFF 99DEF836 146BC9B1 B4D22831"), "1.2.840.10045.3.1.1"),
}
CURVE_ALIASES = {"prime256v1": "secp256r1", "P-256": "secp256r1", "prime192v1": "secp192r1", "P-192": "secp192r1",
                 "P-224": "secp224r1", "P-384": "secp384r1", "P-521": "secp521r1", "Bitcoin": "secp256k1",
                 "Ethereum": "secp256k1"}
EDWARDS_CURVES = ("ed25519",)


def curve_of(name, fname):
    if not isinstance(name, str):
        raise TypeError(f"{fname}: {F.input_form(name)} is not an elliptic curve name")
    n = CURVE_ALIASES.get(name, name)
    if n not in CURVES:
        if name in ("ed448",):
            raise NotImplementedError(f"{fname}: the Edwards curve ed448 is not available")
        raise ValueError(f"{fname}: {name} is not a supported elliptic curve (see "
                         f"$CryptographicEllipticCurveNames)")
    return CURVES[n]


def _bits2int(b, n):
    v = int.from_bytes(b, "big")
    blen = len(b) * 8
    qlen = n.bit_length()
    return v >> (blen - qlen) if blen > qlen else v


def rfc6979_k(x, h1, curve, hashname):
    """Deterministic ECDSA nonce (RFC 6979, section 3.2) with HMAC over the given hash (SHA-256 fallback)."""
    hn = _HASHLIB.get(hashname, "sha256") if hashname in _HASHLIB else "sha256"
    n = curve.n
    qlen = (n.bit_length() + 7) // 8
    z = _bits2int(h1, n) % n
    bx = x.to_bytes(qlen, "big") + z.to_bytes(qlen, "big")
    hl = hashlib.new(hn).digest_size
    V, Kk = b"\x01" * hl, b"\x00" * hl
    Kk = hmac.new(Kk, V + b"\x00" + bx, hn).digest()
    V = hmac.new(Kk, V, hn).digest()
    Kk = hmac.new(Kk, V + b"\x01" + bx, hn).digest()
    V = hmac.new(Kk, V, hn).digest()
    while True:
        T = b""
        while len(T) < qlen:
            V = hmac.new(Kk, V, hn).digest()
            T += V
        k = _bits2int(T[:qlen], n)
        if 1 <= k < n:
            return k
        Kk = hmac.new(Kk, V + b"\x00", hn).digest()
        V = hmac.new(Kk, V, hn).digest()


def ecdsa_sign(curve, x, h, k, low_s=False):
    n = curve.n
    z = _bits2int(h, n)
    while True:
        R = curve.mul(k)
        r = R[0] % n
        if r == 0:
            k = secrets.randbelow(n - 1) + 1
            continue
        s = pow(k, -1, n) * (z + r * x) % n
        if s == 0:
            k = secrets.randbelow(n - 1) + 1
            continue
        rec = (R[1] & 1) | (2 if R[0] >= n else 0)
        if low_s and s > n // 2:
            s = n - s
            rec ^= 1
        return r, s, rec


def ecdsa_verify(curve, Q, h, r, s):
    n = curve.n
    if not (1 <= r < n and 1 <= s < n) or Q is None or not curve.on_curve(Q):
        return False
    z = _bits2int(h, n)
    w = pow(s, -1, n)
    P = curve.add(curve.mul(z * w % n), curve.mul(r * w % n, Q))
    return P is not None and P[0] % n == r


def _tagged(tag, data):
    t = hashlib.sha256(tag.encode()).digest()
    return hashlib.sha256(t + t + data).digest()


def schnorr_sign(x, msg, aux):
    """BIP-340 Schnorr signature over secp256k1 (32-byte message)."""
    c = CURVES["secp256k1"]
    n = c.n
    P = c.mul(x)
    d = x if P[1] % 2 == 0 else n - x
    t = (d ^ int.from_bytes(_tagged("BIP0340/aux", aux), "big")).to_bytes(32, "big")
    px = P[0].to_bytes(32, "big")
    k0 = int.from_bytes(_tagged("BIP0340/nonce", t + px + msg), "big") % n
    if k0 == 0:
        raise ValueError("GenerateDigitalSignature: Schnorr nonce failure")
    R = c.mul(k0)
    k = k0 if R[1] % 2 == 0 else n - k0
    rx = R[0].to_bytes(32, "big")
    e = int.from_bytes(_tagged("BIP0340/challenge", rx + px + msg), "big") % n
    return rx, ((k + e * d) % n).to_bytes(32, "big")


def schnorr_verify(px, msg, rx, s):
    c = CURVES["secp256k1"]
    try:
        P = c.lift_x(px, 0)
    except ValueError:
        return False
    r = int.from_bytes(rx, "big")
    sv = int.from_bytes(s, "big")
    if r >= c.p or sv >= c.n:
        return False
    e = int.from_bytes(_tagged("BIP0340/challenge", rx + px.to_bytes(32, "big") + msg), "big") % c.n
    R = c.add(c.mul(sv), c.mul(c.n - e, P))
    return R is not None and R[1] % 2 == 0 and R[0] == r


# ── Ed25519 (RFC 8032) ──

_ED_P = 2 ** 255 - 19
_ED_L = 2 ** 252 + 27742317777372353535851937790883648493
_ED_D = -121665 * pow(121666, -1, _ED_P) % _ED_P
_ED_I = pow(2, (_ED_P - 1) // 4, _ED_P)


def _ed_add(P, Q):
    X1, Y1, Z1, T1 = P
    X2, Y2, Z2, T2 = Q
    p = _ED_P
    A_ = (Y1 - X1) * (Y2 - X2) % p
    B_ = (Y1 + X1) * (Y2 + X2) % p
    C_ = 2 * T1 * T2 * _ED_D % p
    D_ = 2 * Z1 * Z2 % p
    E_, F_, G_, H_ = B_ - A_, D_ - C_, D_ + C_, B_ + A_
    return (E_ * F_ % p, G_ * H_ % p, F_ * G_ % p, E_ * H_ % p)


def _ed_mul(s, P):
    Q = (0, 1, 1, 0)
    while s > 0:
        if s & 1:
            Q = _ed_add(Q, P)
        P = _ed_add(P, P)
        s >>= 1
    return Q


def _ed_recover_x(y, sign):
    p = _ED_P
    if y >= p:
        return None
    x2 = (y * y - 1) * pow(_ED_D * y * y + 1, -1, p) % p
    if x2 == 0:
        return None if sign else 0
    x = pow(x2, (p + 3) // 8, p)
    if (x * x - x2) % p != 0:
        x = x * _ED_I % p
    if (x * x - x2) % p != 0:
        return None
    if (x & 1) != sign:
        x = p - x
    return x


_ED_GY = 4 * pow(5, -1, _ED_P) % _ED_P
_ED_GX = _ed_recover_x(_ED_GY, 0)
_ED_G = (_ED_GX, _ED_GY, 1, _ED_GX * _ED_GY % _ED_P)


def _ed_compress(P):
    zi = pow(P[2], -1, _ED_P)
    x, y = P[0] * zi % _ED_P, P[1] * zi % _ED_P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def _ed_decompress(s):
    if len(s) != 32:
        return None
    y = int.from_bytes(s, "little")
    sign = y >> 255
    y &= (1 << 255) - 1
    x = _ed_recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, x * y % _ED_P)


def _ed_secret(seed):
    h = hashlib.sha512(seed).digest()
    a = int.from_bytes(h[:32], "little")
    a &= (1 << 254) - 8
    a |= 1 << 254
    return a, h[32:]


def ed25519_public(seed):
    a, _ = _ed_secret(seed)
    return _ed_compress(_ed_mul(a, _ED_G))


def ed25519_sign(seed, msg):
    a, prefix = _ed_secret(seed)
    A_ = _ed_compress(_ed_mul(a, _ED_G))
    r = int.from_bytes(hashlib.sha512(prefix + msg).digest(), "little") % _ED_L
    Rs = _ed_compress(_ed_mul(r, _ED_G))
    h = int.from_bytes(hashlib.sha512(Rs + A_ + msg).digest(), "little") % _ED_L
    return Rs, int.to_bytes((r + h * a) % _ED_L, 32, "little")


def ed25519_verify(pub, msg, Rs, Ss):
    A_ = _ed_decompress(pub)
    R = _ed_decompress(Rs)
    if A_ is None or R is None or len(Ss) != 32:
        return False
    s = int.from_bytes(Ss, "little")
    if s >= _ED_L:
        return False
    h = int.from_bytes(hashlib.sha512(Rs + pub + msg).digest(), "little") % _ED_L
    sB = _ed_mul(s, _ED_G)
    hA = _ed_mul(h, A_)
    lhs, rhs = sB, _ed_add(R, hA)
    p = _ED_P
    return (lhs[0] * rhs[2] - rhs[0] * lhs[2]) % p == 0 and (lhs[1] * rhs[2] - rhs[1] * lhs[2]) % p == 0


# ═══════════════════════════════════════ scrypt / DerivedKey ═══════════════════════════════════════

def _b64n(b):
    return base64.b64encode(b).decode("ascii").rstrip("=")


def _unb64n(s):
    return base64.b64decode(s + "=" * (-len(s) % 4))


def _secret_bytes(x, fname, what="password"):
    if isinstance(x, str):
        return x.encode("utf-8")
    if is_byte_array(x):
        return ba_bytes(x, fname)
    raise TypeError(f"{fname}: the {what} must be a string or a ByteArray")


def scrypt(pw, salt, n, r, p, size, fname):
    if n < 2 or n & (n - 1):
        raise ValueError(f"{fname}: the scrypt parameter N must be a power of 2 greater than 1")
    if r < 1 or p < 1 or size < 1:
        raise ValueError(f"{fname}: the scrypt parameters r, p and KeySize must be positive integers")
    try:
        return hashlib.scrypt(pw, salt=salt, n=n, r=r, p=p, dklen=size, maxmem=128 * r * (n + p + 2) + (1 << 24))
    except (ValueError, MemoryError) as e:
        raise ValueError(f"{fname}: scrypt failed ({e})")


def _symbolic(v):
    return isinstance(v, (Symbol, Expr)) and not isinstance(v, bool) and not K.is_number(v) \
        and not (isinstance(v, Expr) and v.head in ("List", "Association", "Rule", "ByteArray"))


def _pint(v, fname, what):
    if K.is_int(v) and not isinstance(v, bool) and v > 0:
        return int(v)
    if _symbolic(v):
        raise Unevaluated()
    if isinstance(v, (str, bool)) or not K.is_number(v):
        raise TypeError(f"{fname}: {what} must be a positive integer, not {F.input_form(v)}")
    raise ValueError(f"{fname}: {what} must be a positive integer, not {F.input_form(v)}")


def derived_key(pw, salt, method, fname):
    params = {"KeySize": 64, "N": 2 ** 15, "r": 8, "p": 1}
    func = "scrypt"
    if method is not None and not is_sym(method, "Automatic"):
        if isinstance(method, str):
            func = method
        elif KL.is_assoc(method):
            m = IE.assoc_dict(method, fname)
            func = m.pop("Function", "scrypt")
            fp = m.pop("FunctionParameters", None)
            if fp is not None:
                m.update(IE.assoc_dict(fp, fname))
            for k, v in m.items():
                if k not in params:
                    raise ValueError(f"{fname}: unknown Method key \"{k}\"")
                params[k] = _pint(v, fname, k)
        else:
            raise ValueError(f"{fname}: invalid Method {F.input_form(method)}")
    if func in ("Argon2d", "Argon2i", "Argon2id"):
        raise NotImplementedError(f"{fname}: the {func} key derivation function is not available "
                                  f"(use \"scrypt\")")
    if func != "scrypt":
        raise ValueError(f"{fname}: {F.input_form(func)} is not a supported key derivation function")
    key = scrypt(pw, salt, params["N"], params["r"], params["p"], params["KeySize"], fname)
    return Expr("DerivedKey", A([("Function", "scrypt"), ("DerivedKey", _ba(key)), ("Salt", _ba(salt)),
                                 ("FunctionParameters", A([(k, params[k]) for k in ("KeySize", "N", "r", "p")]))]))


def _dk_parts(dk, fname):
    if not (isinstance(dk, Expr) and dk.head == "DerivedKey" and len(dk.args) == 1 and KL.is_assoc(dk.args[0])):
        raise TypeError(f"{fname}: {F.input_form(dk)} is not a DerivedKey")
    a = IE.assoc_dict(dk.args[0], fname)
    fp = IE.assoc_dict(a["FunctionParameters"], fname)
    return a["Function"], ba_bytes(a["DerivedKey"], fname), ba_bytes(a["Salt"], fname), fp


def phc_string(dk):
    func, key, salt, fp = _dk_parts(dk, "DerivedKey")
    ln = int(fp["N"]).bit_length() - 1
    return f"${func}$ln={ln},r={fp['r']},p={fp['p']}${_b64n(salt)}${_b64n(key)}"


_PHC_RE = re.compile(r"\$scrypt\$ln=(\d+),r=(\d+),p=(\d+)\$([A-Za-z0-9+/]*)\$([A-Za-z0-9+/]+)")


@fn("GenerateDerivedKey", area=AREA, status="added",
    notes="GenerateDerivedKey[password] / [password, salt] derives a key with scrypt (RFC 7914; defaults KeySize 64, "
          "N 2^15, r 8, p 1, random 16-byte salt) giving DerivedKey[<|\"Function\", \"DerivedKey\", \"Salt\", "
          "\"FunctionParameters\"|>]; Method -> <|\"Function\" -> \"scrypt\", \"KeySize\", \"N\", \"r\", \"p\"|>. "
          "Argon2 functions raise a clear error. Check: reference (RFC 7914 scrypt vector) and VerifyDerivedKey in tests.")
def _generatederivedkey(pw: String | Expr, *rest: String | Expr | Option) -> Expr:
    rest, o = split_opts(rest, ("Method",), "GenerateDerivedKey")
    if len(rest) > 1:
        raise ValueError("GenerateDerivedKey: called with too many arguments")
    pwb = _secret_bytes(pw, "GenerateDerivedKey")
    salt = _secret_bytes(rest[0], "GenerateDerivedKey", "salt") if rest else secrets.token_bytes(16)
    return derived_key(pwb, salt, o.get("Method"), "GenerateDerivedKey")


@fn("DerivedKey", area=AREA, status="added",
    notes="DerivedKey[<|...|>] is the result of GenerateDerivedKey (properties \"Function\", \"DerivedKey\", "
          "\"Salt\", \"FunctionParameters\", \"PHCString\", \"Properties\"); DerivedKey[\"$scrypt$ln=..$salt$hash\"] "
          "rebuilds one from its PHC string. Check: exact; PHC string round trips in tests.")
def _derivedkey(x: Assoc | String) -> Expr:
    if isinstance(x, str):
        m = _PHC_RE.fullmatch(x.strip())
        if not m:
            raise ValueError("DerivedKey: the string is not a valid scrypt PHC string")
        try:
            salt, key = _unb64n(m.group(4)), _unb64n(m.group(5))
        except Exception:
            raise ValueError("DerivedKey: invalid Base64 in the PHC string")
        return Expr("DerivedKey", A([("Function", "scrypt"), ("DerivedKey", _ba(key)), ("Salt", _ba(salt)),
                                     ("FunctionParameters", A([("KeySize", len(key)), ("N", 2 ** int(m.group(1))),
                                                               ("r", int(m.group(2))), ("p", int(m.group(3)))]))]))
    if KL.is_assoc(x):
        _dk_parts(Expr("DerivedKey", x), "DerivedKey")
        raise Unevaluated
    raise TypeError(f"DerivedKey: {F.input_form(x)} is not an association or a PHC string")


@fn("VerifyDerivedKey", area=AREA, status="added",
    notes="VerifyDerivedKey[key, password] gives True when password derives the DerivedKey key (same function, "
          "salt and parameters; constant-time comparison), False otherwise. Check: delegated (hashlib.scrypt); RFC 7914 reference in tests.")
def _verifyderivedkey(dk: Expr, pw: String | Expr) -> Boolean:
    func, key, salt, fp = _dk_parts(dk, "VerifyDerivedKey")
    got = scrypt(_secret_bytes(pw, "VerifyDerivedKey"), salt, int(fp["N"]), int(fp["r"]), int(fp["p"]), len(key),
                 "VerifyDerivedKey")
    return hmac.compare_digest(got, key)


# ═══════════════════════════════════════ symmetric keys & encryption ═══════════════════════════════════════

def _is_obj(x, head):
    return isinstance(x, Expr) and x.head == head and len(x.args) == 1 and KL.is_assoc(x.args[0])


def _cipher_spec(method, fname):
    cipher, mode, iv, size = "AES256", None, None, None
    if method is None or is_sym(method, "Automatic"):
        pass
    elif isinstance(method, str):
        cipher = method
    elif KL.is_assoc(method):
        m = IE.assoc_dict(method, fname)
        cipher = m.pop("Cipher", cipher)
        mode = m.pop("BlockMode", None)
        iv = m.pop("InitializationVector", None)
        size = m.pop("KeySize", None)
        if m:
            raise ValueError(f"{fname}: unknown Method key \"{next(iter(m))}\"")
    else:
        raise ValueError(f"{fname}: invalid Method {F.input_form(method)}")
    if cipher in UNSUPPORTED_CIPHERS:
        raise NotImplementedError(f"{fname}: the {cipher} cipher is not available (use AES128, AES192, AES256, "
                                  f"Blowfish or RC4)")
    if cipher not in CIPHERS:
        raise ValueError(f"{fname}: {F.input_form(cipher)} is not a supported cipher")
    ksize, bsize, _ = CIPHERS[cipher]
    if size is not None and not is_sym(size, "Automatic"):
        size = _pint(size, fname, "KeySize")
        if cipher.startswith("AES") and size != ksize * 8:
            raise ValueError(f"{fname}: {cipher} keys have {ksize * 8} bits")
        if cipher == "Blowfish" and not (32 <= size <= 448 and size % 8 == 0):
            raise ValueError(f"{fname}: Blowfish keys have 32 to 448 bits")
        if cipher == "RC4" and not (40 <= size <= 2048 and size % 8 == 0):
            raise ValueError(f"{fname}: RC4 keys have 40 to 2048 bits")
        ksize = size // 8
    if cipher == "RC4":
        if mode not in (None, "None") and not is_sym(mode, "None") and not is_sym(mode, "Automatic"):
            raise ValueError(f"{fname}: RC4 is a stream cipher and has no block mode")
        mode = None
    else:
        mode = _block_mode(mode, fname)
    iv = _iv_arg(iv, cipher, fname)
    return cipher, ksize, mode, iv


def _block_mode(mode, fname):
    if mode is None or is_sym(mode, "Automatic"):
        return "CBC"
    if mode in ("ECB", "CBC", "CFB", "OFB", "CTR"):
        return mode
    raise ValueError(f"{fname}: {F.input_form(mode)} is not a supported block mode (ECB, CBC, CFB, OFB, CTR)")


def _iv_arg(iv, cipher, fname):
    if iv is None or is_sym(iv, "None") or is_sym(iv, "Automatic"):
        return None
    b = bytes_arg(iv, fname, "initialization vector")
    if cipher == "RC4" or len(b) != CIPHERS[cipher][1]:
        raise ValueError(f"{fname}: the initialization vector for {cipher} must have {CIPHERS[cipher][1]} bytes")
    return b


def _sym_key(cipher, mode, key, iv):
    return Expr("SymmetricKey", A([("Cipher", cipher), ("BlockMode", mode if mode else None), ("Key", _ba(key)),
                                   ("InitializationVector", _ba(iv) if iv else None)]))


def password_key(pw, size):
    """The key GenerateSymmetricKey["password"] / Encrypt["password", ...] derive: scrypt(password, empty salt,
    N 2^14, r 8, p 1) - deterministic, so the same password always gives the same key."""
    return scrypt(pw, b"", 2 ** 14, 8, 1, size, "GenerateSymmetricKey")


@fn("GenerateSymmetricKey", area=AREA, status="added",
    notes="GenerateSymmetricKey[] (random AES256/CBC key), [\"password\"] (deterministic scrypt derivation), "
          "[ByteArray] (the given key bytes), [DerivedKey[...]]; Method -> cipher | <|\"Cipher\", \"BlockMode\", "
          "\"KeySize\", \"InitializationVector\"|>; ciphers AES128/192/256, Blowfish, RC4 (DES, CAST5, IDEA give a "
          "clear error); gives SymmetricKey[<|\"Cipher\", \"BlockMode\", \"Key\", \"InitializationVector\"|>]. Check: exact; Encrypt/Decrypt round trips and published cipher vectors in tests.")
def _generatesymmetrickey(*args: String | Expr | Option) -> Expr:
    rest, o = split_opts(args, ("Method",), "GenerateSymmetricKey")
    if len(rest) > 1:
        raise ValueError("GenerateSymmetricKey: called with too many arguments")
    cipher, ksize, mode, iv = _cipher_spec(o.get("Method"), "GenerateSymmetricKey")
    if not rest:
        key = secrets.token_bytes(ksize)
    else:
        src = rest[0]
        if isinstance(src, str):
            key = password_key(src.encode("utf-8"), ksize)
        elif _is_obj(src, "DerivedKey"):
            dk = _dk_parts(src, "GenerateSymmetricKey")[1]
            if len(dk) < ksize:
                raise ValueError(f"GenerateSymmetricKey: the derived key is shorter than the {ksize}-byte key")
            key = dk[:ksize]
        else:
            key = bytes_arg(src, "GenerateSymmetricKey", "key")
            if len(key) != ksize and not (cipher in ("Blowfish", "RC4") and "KeySize" not in str(o.get("Method"))
                                          and (4 <= len(key) <= 56 if cipher == "Blowfish" else 5 <= len(key) <= 256)):
                raise ValueError(f"GenerateSymmetricKey: a {cipher} key must have {ksize} bytes, not {len(key)}")
    return _sym_key(cipher, mode, key, iv)


def sym_key_parts(k, fname):
    a = IE.assoc_dict(k.args[0], fname)
    cipher = a.get("Cipher", "AES256")
    if cipher in UNSUPPORTED_CIPHERS:
        raise NotImplementedError(f"{fname}: the {cipher} cipher is not available")
    if cipher not in CIPHERS:
        raise ValueError(f"{fname}: {F.input_form(cipher)} is not a supported cipher")
    if "Key" not in a:
        raise ValueError(f"{fname}: the SymmetricKey has no \"Key\"")
    key = ba_bytes(a["Key"], fname)
    mode = None if cipher == "RC4" else _block_mode(a.get("BlockMode"), fname)
    iv = _iv_arg(a.get("InitializationVector"), cipher, fname)
    return cipher, mode, key, iv


@fn("SymmetricKey", area=AREA, status="added",
    notes="SymmetricKey[<|\"Cipher\", \"BlockMode\", \"Key\", \"InitializationVector\"|>] represents a symmetric "
          "key (properties by name and \"Properties\"); validated, otherwise inert. Check: exact (validated wrapper).")
def _symmetrickey(a: Assoc) -> Expr:
    if not KL.is_assoc(a):
        raise TypeError(f"SymmetricKey: {F.input_form(a)} is not an association")
    sym_key_parts(Expr("SymmetricKey", a), "SymmetricKey")
    raise Unevaluated


def _payload(x, fname):
    if isinstance(x, str):
        return x.encode("utf-8"), "String"
    if is_byte_array(x):
        return ba_bytes(x, fname), "ByteArray"
    return IE.compress_string(x).encode("ascii"), "Expression"


def _unpayload(b, form, fname):
    if form == "String":
        try:
            return b.decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError(f"{fname}::failed: the data could not be decrypted (wrong key or corrupt data)")
    if form == "ByteArray":
        return _ba(b)
    try:
        return ev(IE.uncompress_string(b.decode("ascii"), fname))
    except (UnicodeDecodeError, ValueError):
        raise ValueError(f"{fname}::failed: the data could not be decrypted (wrong key or corrupt data)")


def _padding_opt(v, fname, allowed):
    if v is None or is_sym(v, "Automatic"):
        return "Automatic"
    if is_sym(v, "None") or v == "None":
        return None
    if v in allowed:
        return v
    if v == "X931":
        raise NotImplementedError(f"{fname}: X9.31 padding is not available")
    raise ValueError(f"{fname}: {F.input_form(v)} is not a supported padding")


def _method_assoc(o, fname, allowed):
    m = o.get("Method")
    if m is None or is_sym(m, "Automatic"):
        return {}
    d = IE.assoc_dict(m, fname)
    for k in d:
        if k not in allowed:
            raise ValueError(f"{fname}: unknown Method key \"{k}\"")
    return d


def _key_from_password(pw, fname):
    return _sym_key("AES256", "CBC", password_key(pw.encode("utf-8"), 32), None)


def encrypt(key, x, o, fname="Encrypt"):
    if isinstance(key, str):
        key = _key_from_password(key, fname)
    data, form = _payload(x, fname)
    if _is_obj(key, "SymmetricKey"):
        m = _method_assoc(o, fname, ("Padding", "InitializationVector", "BlockMode"))
        cipher, mode, k, iv = sym_key_parts(key, fname)
        if "BlockMode" in m and cipher != "RC4":
            mode = _block_mode(m["BlockMode"], fname)
        if "InitializationVector" in m:
            iv = _iv_arg(m["InitializationVector"], cipher, fname)
        pad = _padding_opt(m.get("Padding"), fname, ("PKCS7",))
        pad = mode in ("ECB", "CBC") if pad in ("Automatic", "PKCS7") else False
        bs = CIPHERS[cipher][1]
        if cipher != "RC4" and mode != "ECB" and iv is None:
            iv = secrets.token_bytes(bs)
        ct = sym_encrypt(cipher, k, data, mode, iv, pad)
        items = [("Data", _ba(ct)), ("InitializationVector", _ba(iv) if iv and mode != "ECB" else None),
                 ("OriginalForm", Symbol(form))]
        if not pad and mode in ("ECB", "CBC"):
            items.append(("Padding", None))
        if cipher != "RC4" and mode != sym_key_parts(key, fname)[1]:
            items.append(("BlockMode", mode))
        return Expr("EncryptedObject", A(items))
    if _is_obj(key, "PrivateKey") or _is_obj(key, "PublicKey"):
        kd = asym_parts(key, fname)
        if kd["Type"] != "RSA":
            raise ValueError(f"{fname}: {kd['Type']} keys can only sign, not encrypt (use an RSA or a symmetric key)")
        m = _method_assoc(o, fname, ("Padding",))
        private = key.head == "PrivateKey"
        pad = _padding_opt(m.get("Padding", kd.get("Padding")), fname, ("PKCS1", "OAEP"))
        if pad == "Automatic":
            pad = "PKCS1"
        if private and pad == "OAEP":
            raise ValueError(f"{fname}: OAEP padding needs a public key")
        n = kd["PublicModulus"]
        k = _modlen(n)
        em = rsa_pad(data, k, pad, 1 if private else 2, fname)
        mi = int.from_bytes(em, "big")
        if mi >= n:
            raise ValueError(f"{fname}: the data is too large for this RSA key")
        c = pow(mi, kd["PrivateExponent"] if private else kd["PublicExponent"], n)
        items = [("Data", _ba(_i2osp(c, k))), ("OriginalForm", Symbol(form))]
        if pad != "PKCS1":
            items.append(("Padding", pad))
        return Expr("EncryptedObject", A(items))
    raise TypeError(f"{fname}: {F.input_form(key)} is not a SymmetricKey, a PrivateKey, a PublicKey or a "
                     f"password string")


def decrypt(key, obj, o, fname="Decrypt"):
    if not _is_obj(obj, "EncryptedObject"):
        raise TypeError(f"{fname}: {F.input_form(obj)} is not an EncryptedObject")
    a = IE.assoc_dict(obj.args[0], fname)
    if "Data" not in a:
        raise ValueError(f"{fname}: the EncryptedObject has no \"Data\"")
    data = ba_bytes(a["Data"], fname)
    form = a.get("OriginalForm", Symbol("Expression"))
    form = form.name if isinstance(form, Symbol) else form
    if form not in ("String", "ByteArray", "Expression"):
        raise ValueError(f"{fname}: unknown OriginalForm {F.input_form(form)}")
    if isinstance(key, str):
        key = _key_from_password(key, fname)
    if _is_obj(key, "SymmetricKey"):
        m = _method_assoc(o, fname, ("Padding", "InitializationVector", "BlockMode"))
        cipher, mode, k, iv = sym_key_parts(key, fname)
        mode = _block_mode(m.get("BlockMode", a.get("BlockMode", mode)), fname) if cipher != "RC4" else None
        ivv = m.get("InitializationVector", a.get("InitializationVector"))
        if ivv is not None:
            iv = _iv_arg(ivv, cipher, fname)
        if cipher != "RC4" and mode != "ECB" and iv is None:
            raise ValueError(f"{fname}: the initialization vector is missing")
        pad = _padding_opt(m.get("Padding", a.get("Padding", Symbol("Automatic"))), fname, ("PKCS7",))
        pad = mode in ("ECB", "CBC") if pad in ("Automatic", "PKCS7") else False
        return _unpayload(sym_decrypt(cipher, k, data, mode, iv, pad), form, fname)
    if _is_obj(key, "PrivateKey") or _is_obj(key, "PublicKey"):
        kd = asym_parts(key, fname)
        if kd["Type"] != "RSA":
            raise ValueError(f"{fname}: {kd['Type']} keys cannot decrypt")
        m = _method_assoc(o, fname, ("Padding",))
        pad = _padding_opt(m.get("Padding", a.get("Padding", kd.get("Padding"))), fname, ("PKCS1", "OAEP"))
        if pad == "Automatic":
            pad = "PKCS1"
        private = key.head == "PrivateKey"
        n = kd["PublicModulus"]
        k = _modlen(n)
        c = int.from_bytes(data, "big")
        if len(data) != k or c >= n:
            raise ValueError(f"{fname}::failed: the data could not be decrypted (wrong key or corrupt data)")
        em = _i2osp(pow(c, kd["PrivateExponent"] if private else kd["PublicExponent"], n), k)
        return _unpayload(rsa_unpad(em, k, pad, 2 if private else 1, fname), form, fname)
    raise TypeError(f"{fname}: {F.input_form(key)} is not a SymmetricKey, a PrivateKey, a PublicKey or a "
                     f"password string")


@fn("Encrypt", area=AREA, status="added",
    notes="Encrypt[key, expr] with a SymmetricKey (random IV per call for CBC/CFB/OFB/CTR), a \"password\" "
          "(AES256 key derived with scrypt), an RSA PublicKey (PKCS1 or OAEP) or an RSA PrivateKey (PKCS1 type 1, "
          "decrypted with the public key); strings as UTF-8, ByteArrays literally, other expressions via Compress; "
          "Method -> <|\"Padding\", \"InitializationVector\", \"BlockMode\"|>; operator form Encrypt[key]; gives "
          "EncryptedObject[<|\"Data\", \"InitializationVector\", \"OriginalForm\"|>]. Elliptic-curve keys give a "
          "clear error. Check: reference (FIPS-197, SP 800-38A, Blowfish and RC4 vectors) and Decrypt round trips in tests.")
def _encrypt(key: Expr | String, *rest: Expr | Option) -> Expr:
    rest, o = split_opts(rest, ("Method",), "Encrypt")
    if not rest:
        if o:
            raise ValueError("Encrypt: the operator form takes no options")
        raise Unevaluated
    if len(rest) != 1:
        raise ValueError("Encrypt: called with too many arguments")
    return encrypt(key, rest[0], o)


@fn("Decrypt", area=AREA, status="added",
    notes="Decrypt[key, EncryptedObject[...]] recovers the original string, ByteArray or expression; key is the "
          "SymmetricKey or password used, the RSA PrivateKey for public-key encryption or the PublicKey for "
          "private-key encryption; a wrong key raises Decrypt::failed rather than giving garbage when padding or "
          "decoding detects it; operator form Decrypt[key]. Check: exact (padding validation detects most wrong keys); Encrypt round trips in tests.")
def _decrypt(key: Expr | String, *rest: Expr | Option) -> Expr:
    rest, o = split_opts(rest, ("Method",), "Decrypt")
    if not rest:
        raise Unevaluated
    if len(rest) != 1:
        raise ValueError("Decrypt: called with too many arguments")
    return decrypt(key, rest[0], o)


def _operator_hook(fn_name):
    def hook(head, args, ev_):
        if len(head.args) == 1 and len(args) == 1:
            return ev_(Expr(fn_name, head.args[0], args[0]))
        return K.UNHANDLED
    return hook


for _n in ("Encrypt", "Decrypt"):
    KL.COMPOUND_HEAD_HOOKS[_n] = _operator_hook(_n)


# ═══════════════════════════════════════ asymmetric keys ═══════════════════════════════════════

def _quantity_bits(n):
    return Expr("Quantity", n, "Bits")


def _int_of(v, fname, what):
    if K.is_int(v) and not isinstance(v, bool) and v >= 0:
        return int(v)
    if _symbolic(v):
        raise Unevaluated()
    if isinstance(v, (str, bool)) or not K.is_number(v):
        raise TypeError(f"{fname}: {what} must be a non-negative integer, not {F.input_form(v)}")
    raise ValueError(f"{fname}: {what} must be a non-negative integer, not {F.input_form(v)}")


def _hex_bytes(v, fname, what):
    if isinstance(v, str) and re.fullmatch(r"(?:[0-9A-Fa-f]{2})+", v):
        return bytes.fromhex(v)
    raise TypeError(f"{fname}: {what} must be a hexadecimal string")


_COMMON_E = (65537, 3, 5, 17, 257, 37, 7, 11, 13)


def _recover_e(n, d, fname):
    for e in _COMMON_E:
        if pow(pow(2, d, n), e, n) == 2 and pow(pow(3, d, n), e, n) == 3:
            return e
    raise ValueError(f"{fname}: the RSA key needs a \"PublicExponent\" (it is not one of the common exponents)")


def rsa_private_expr(n, e, d, pad="PKCS1"):
    return Expr("PrivateKey", A([("Type", "RSA"), ("Cipher", "RSA"), ("Padding", pad), ("PublicExponent", e),
                                 ("PublicModulus", n), ("PrivateExponent", d)]))


def rsa_public_expr(n, e, pad="PKCS1"):
    return Expr("PublicKey", A([("Type", "RSA"), ("Cipher", "RSA"), ("Padding", pad), ("PublicExponent", e),
                                ("PublicModulus", n)]))


def ec_private_expr(curve, x, compressed=False):
    P = curve.mul(x)
    return Expr("PrivateKey", A([("Type", "EllipticCurve"), ("CurveName", curve.name), ("PrivateMultiplier", x),
                                 ("PublicCurvePoint", L(*P)), ("Compressed", compressed)]))


def ec_public_expr(curve, P, compressed=False):
    return Expr("PublicKey", A([("Type", "EllipticCurve"), ("CurveName", curve.name), ("PublicCurvePoint", L(*P)),
                                ("Compressed", compressed)]))


def ed_private_expr(seed):
    return Expr("PrivateKey", A([("Type", "EdwardsCurve"), ("CurveName", "ed25519"), ("PrivateByteArray", _ba(seed)),
                                 ("PublicByteArray", _ba(ed25519_public(seed)))]))


def ed_public_expr(pub):
    return Expr("PublicKey", A([("Type", "EdwardsCurve"), ("CurveName", "ed25519"), ("PublicByteArray", _ba(pub))]))


_KEY_FIELDS = {"Type", "Cipher", "Padding", "PublicExponent", "PublicModulus", "PrivateExponent", "CurveName",
               "PrivateMultiplier", "PublicCurvePoint", "Compressed", "PrivateByteArray", "PublicByteArray",
               "PrivateHexString", "PublicHexString", "PrivateKeySize", "PublicKeySize"}


def asym_parts(key, fname):
    """Validate a PrivateKey/PublicKey and return a dict with Python values (filling derivable fields)."""
    private = key.head == "PrivateKey"
    a = IE.assoc_dict(key.args[0], fname)
    for k in a:
        if k not in _KEY_FIELDS:
            raise ValueError(f"{fname}: unknown key field \"{k}\"")
    t = a.get("Type")
    if t is None:
        t = "RSA" if "PublicModulus" in a else ("EdwardsCurve" if a.get("CurveName") in EDWARDS_CURVES
                                                 else "EllipticCurve")
    if t in ("Bitcoin", "Ethereum"):
        a.setdefault("CurveName", "secp256k1")
        a.setdefault("Compressed", t == "Bitcoin")
        t = "EllipticCurve"
    out = {"Type": t, "Private": private}
    if t == "RSA":
        if "PublicModulus" not in a:
            raise ValueError(f"{fname}: an RSA key needs a \"PublicModulus\"")
        n = _int_of(a["PublicModulus"], fname, "PublicModulus")
        if n < 3:
            raise ValueError(f"{fname}: invalid RSA modulus")
        out["PublicModulus"] = n
        pad = a.get("Padding", "PKCS1")
        out["Padding"] = None if is_sym(pad, "None") else pad
        if private:
            if "PrivateExponent" not in a:
                raise ValueError(f"{fname}: an RSA private key needs a \"PrivateExponent\"")
            d = _int_of(a["PrivateExponent"], fname, "PrivateExponent")
            out["PrivateExponent"] = d
            out["PublicExponent"] = _int_of(a["PublicExponent"], fname, "PublicExponent") \
                if "PublicExponent" in a else _recover_e(n, d, fname)
        else:
            out["PublicExponent"] = _int_of(a.get("PublicExponent", 65537), fname, "PublicExponent")
        return out
    if t == "EllipticCurve":
        curve = curve_of(a.get("CurveName", "secp256k1"), fname)
        comp = a.get("Compressed", False)
        if not isinstance(comp, bool):
            raise TypeError(f"{fname}: \"Compressed\" must be True or False")
        out.update(Curve=curve, CurveName=curve.name, Compressed=comp)
        x = None
        if private:
            if "PrivateMultiplier" in a:
                x = _int_of(a["PrivateMultiplier"], fname, "PrivateMultiplier")
            elif "PrivateHexString" in a:
                x = int.from_bytes(_hex_bytes(a["PrivateHexString"], fname, "PrivateHexString"), "big")
            elif "PrivateByteArray" in a:
                x = int.from_bytes(ba_bytes(a["PrivateByteArray"], fname), "big")
            else:
                raise ValueError(f"{fname}: an elliptic-curve private key needs a \"PrivateMultiplier\", "
                                 f"\"PrivateHexString\" or \"PrivateByteArray\"")
            if not 1 <= x < curve.n:
                raise ValueError(f"{fname}: the private multiplier is out of range for {curve.name}")
            out["PrivateMultiplier"] = x
            P = curve.mul(x)
            if "PublicCurvePoint" in a and _point_of(a["PublicCurvePoint"], curve, fname) != P:
                raise ValueError(f"{fname}: the public curve point does not match the private multiplier")
        elif "PublicCurvePoint" in a:
            P = _point_of(a["PublicCurvePoint"], curve, fname)
        elif "PublicByteArray" in a:
            P = curve.decode(ba_bytes(a["PublicByteArray"], fname), fname)
        elif "PublicHexString" in a:
            P = curve.decode(_hex_bytes(a["PublicHexString"], fname, "PublicHexString"), fname)
        else:
            raise ValueError(f"{fname}: an elliptic-curve public key needs a \"PublicCurvePoint\", "
                             f"\"PublicByteArray\" or \"PublicHexString\"")
        out["PublicCurvePoint"] = P
        return out
    if t == "EdwardsCurve":
        cn = a.get("CurveName", "ed25519")
        if cn == "ed448":
            raise NotImplementedError(f"{fname}: the Edwards curve ed448 is not available")
        if cn != "ed25519":
            raise ValueError(f"{fname}: {F.input_form(cn)} is not a supported Edwards curve")
        out["CurveName"] = cn
        if private:
            if "PrivateByteArray" in a:
                seed = ba_bytes(a["PrivateByteArray"], fname)
            elif "PrivateHexString" in a:
                seed = _hex_bytes(a["PrivateHexString"], fname, "PrivateHexString")
            else:
                raise ValueError(f"{fname}: an ed25519 private key needs a \"PrivateByteArray\" or "
                                 f"\"PrivateHexString\"")
            if len(seed) != 32:
                raise ValueError(f"{fname}: ed25519 private keys have 32 bytes")
            out["Seed"] = seed
            pub = ed25519_public(seed)
            if "PublicByteArray" in a and ba_bytes(a["PublicByteArray"], fname) != pub:
                raise ValueError(f"{fname}: the public key does not match the private key")
        elif "PublicByteArray" in a:
            pub = ba_bytes(a["PublicByteArray"], fname)
        elif "PublicHexString" in a:
            pub = _hex_bytes(a["PublicHexString"], fname, "PublicHexString")
        else:
            raise ValueError(f"{fname}: an ed25519 public key needs a \"PublicByteArray\" or \"PublicHexString\"")
        if len(pub) != 32 or _ed_decompress(pub) is None:
            raise ValueError(f"{fname}: invalid ed25519 public key")
        out["Public"] = pub
        return out
    raise ValueError(f"{fname}: {F.input_form(t)} is not a supported key type (RSA, EllipticCurve, EdwardsCurve)")


def _point_of(v, curve, fname):
    if is_list(v) and len(v.args) == 2 and all(K.is_int(c) for c in v.args):
        P = (int(v.args[0]), int(v.args[1]))
        if curve.on_curve(P):
            return P
        raise ValueError(f"{fname}: the public curve point is not on {curve.name}")
    raise TypeError(f"{fname}: \"PublicCurvePoint\" must be a pair of integers")


def key_expr(d, private):
    t = d["Type"]
    if t == "RSA":
        if private:
            return rsa_private_expr(d["PublicModulus"], d["PublicExponent"], d["PrivateExponent"], d["Padding"])
        return rsa_public_expr(d["PublicModulus"], d["PublicExponent"], d["Padding"])
    if t == "EllipticCurve":
        if private:
            return ec_private_expr(d["Curve"], d["PrivateMultiplier"], d["Compressed"])
        return ec_public_expr(d["Curve"], d["PublicCurvePoint"], d["Compressed"])
    return ed_private_expr(d["Seed"]) if private else ed_public_expr(d["Public"])


def _key_props(d):
    """Computed (derived) key properties."""
    t, priv = d["Type"], d["Private"]
    p = {}
    if t == "RSA":
        n = d["PublicModulus"]
        pub = _i2osp(n, _modlen(n))
        p["PublicKeySize"] = _quantity_bits(n.bit_length())
        if priv:
            p["PrivateByteArray"] = _ba(_i2osp(d["PrivateExponent"], _modlen(n)))
            p["PrivateKeySize"] = _quantity_bits(n.bit_length())
    elif t == "EllipticCurve":
        c = d["Curve"]
        pub = c.encode(d["PublicCurvePoint"], d["Compressed"])
        p["PublicKeySize"] = _quantity_bits(c.n.bit_length())
        if priv:
            p["PrivateByteArray"] = _ba(d["PrivateMultiplier"].to_bytes(c.size, "big"))
            p["PrivateKeySize"] = _quantity_bits(c.n.bit_length())
    else:
        pub = d["Public"]
        p["PublicKeySize"] = _quantity_bits(256)
        if priv:
            p["PrivateByteArray"] = _ba(d["Seed"])
            p["PrivateKeySize"] = _quantity_bits(256)
    p["PublicByteArray"] = _ba(pub)
    p["PublicHexString"] = pub.hex()
    if priv:
        p["PrivateHexString"] = ba_bytes(p["PrivateByteArray"]).hex()
    return p


def _key_ctor(head):
    def f(a: Assoc) -> Expr:
        fname = head
        if not KL.is_assoc(a):
            raise TypeError(f"{fname}: {F.input_form(a)} is not an association")
        key = Expr(head, a)
        canon = key_expr(asym_parts(key, fname), head == "PrivateKey")
        if K.same(canon, key):
            raise Unevaluated
        return canon
    return f


fn("PrivateKey", area=AREA, status="added",
   notes="PrivateKey[<|...|>] is a private key (RSA: PublicModulus/PublicExponent/PrivateExponent; EllipticCurve: "
         "CurveName and PrivateMultiplier | PrivateHexString | PrivateByteArray; EdwardsCurve ed25519: "
         "PrivateByteArray | PrivateHexString); partial specifications are completed (public point, public key, "
         "RSA exponent); properties: stored fields plus PrivateByteArray, PublicByteArray, Private/PublicHexString, "
         "Private/PublicKeySize, \"Properties\". Check: exact (curve membership and private/public consistency are "
         "validated); RFC 6979 and RFC 8032 keys in tests.")(_key_ctor("PrivateKey"))
fn("PublicKey", area=AREA, status="added",
   notes="PublicKey[<|...|>] is a public key (RSA: PublicModulus, PublicExponent; EllipticCurve: CurveName and "
         "PublicCurvePoint | PublicByteArray | PublicHexString (compressed or not); EdwardsCurve: PublicByteArray | "
         "PublicHexString); validated (points must lie on the curve) and normalised; properties as for "
         "PrivateKey. Check: exact (points are validated on the curve); reference keys in tests.")(_key_ctor("PublicKey"))


def public_of(key, fname):
    """The PublicKey for a PrivateKey or PublicKey (or a SecurityCertificate's key)."""
    if _is_obj(key, "SecurityCertificate"):
        key = assoc_get(key.args[0], "PublicKey")
    if _is_obj(key, "PrivateKey") or _is_obj(key, "PublicKey"):
        return asym_parts(key, fname)
    raise TypeError(f"{fname}: {F.input_form(key)} is not a PublicKey, a PrivateKey or a SecurityCertificate")


@fn("GenerateAsymmetricKeyPair", area=AREA, status="added",
    notes="GenerateAsymmetricKeyPair[] (RSA 2048, exponent 65537), [type] or Method -> type | <|\"Type\", "
          "\"KeySize\", \"PublicExponent\", \"CurveName\", \"Compressed\"|> with types \"RSA\", \"EllipticCurve\" "
          "(default secp256k1), \"EdwardsCurve\" (ed25519), \"Bitcoin\" (secp256k1, compressed) and \"Ethereum\" "
          "(secp256k1); gives <|\"PrivateKey\" -> ..., \"PublicKey\" -> ...|>. Check: exact; sign/verify and encrypt/decrypt round trips in tests.")
def _generateasymmetrickeypair(*args: String | Assoc | Option) -> Assoc:
    rest, o = split_opts(args, ("Method",), "GenerateAsymmetricKeyPair")
    if len(rest) > 1:
        raise ValueError("GenerateAsymmetricKeyPair: called with too many arguments")
    spec = rest[0] if rest else o.get("Method", Symbol("Automatic"))
    fname = "GenerateAsymmetricKeyPair"
    m = {}
    if is_sym(spec, "Automatic"):
        t = "RSA"
    elif isinstance(spec, str):
        t = spec
    elif KL.is_assoc(spec):
        m = IE.assoc_dict(spec, fname)
        t = m.pop("Type", "RSA")
    else:
        raise ValueError(f"{fname}: invalid key specification {F.input_form(spec)}")
    if t in CURVES or t in CURVE_ALIASES and t not in ("Bitcoin", "Ethereum"):
        m.setdefault("CurveName", t)
        t = "EllipticCurve"
    if t == "ed25519":
        t = "EdwardsCurve"
    allowed = {"RSA": ("KeySize", "PublicExponent", "Padding"), "EllipticCurve": ("CurveName", "Compressed"),
               "Bitcoin": ("Compressed",), "Ethereum": ("Compressed",), "EdwardsCurve": ("CurveName",)}
    if t not in allowed:
        raise ValueError(f"{fname}: {F.input_form(t)} is not a supported key type (RSA, EllipticCurve, "
                         f"EdwardsCurve, Bitcoin, Ethereum)")
    for k in m:
        if k not in allowed[t]:
            raise ValueError(f"{fname}: unknown Method key \"{k}\" for {t} keys")
    if t == "RSA":
        bits = _pint(m.get("KeySize", 2048), fname, "KeySize")
        e = _pint(m.get("PublicExponent", 65537), fname, "PublicExponent")
        pad = m.get("Padding", "PKCS1")
        if pad not in ("PKCS1", "OAEP") and not is_sym(pad, "None"):
            raise ValueError(f"{fname}: {F.input_form(pad)} is not a supported RSA padding")
        if bits > 8192:
            raise ValueError(f"{fname}: RSA keys of more than 8192 bits are not supported")
        n, e, d = rsa_generate(bits, e)
        pad = None if is_sym(pad, "None") else pad
        priv, pub = rsa_private_expr(n, e, d, pad), rsa_public_expr(n, e, pad)
    elif t == "EdwardsCurve":
        cn = m.get("CurveName", "ed25519")
        if cn == "ed448":
            raise NotImplementedError(f"{fname}: the Edwards curve ed448 is not available")
        if cn != "ed25519":
            raise ValueError(f"{fname}: {F.input_form(cn)} is not a supported Edwards curve")
        seed = secrets.token_bytes(32)
        priv, pub = ed_private_expr(seed), ed_public_expr(ed25519_public(seed))
    else:
        curve = curve_of(m.get("CurveName", "secp256k1"), fname)
        comp = m.get("Compressed", t == "Bitcoin")
        if not isinstance(comp, bool):
            raise TypeError(f"{fname}: \"Compressed\" must be True or False")
        x = secrets.randbelow(curve.n - 1) + 1
        priv, pub = ec_private_expr(curve, x, comp), ec_public_expr(curve, curve.mul(x), comp)
    return A([("PrivateKey", priv), ("PublicKey", pub)])


# ═══════════════════════════════════════ digital signatures ═══════════════════════════════════════

def _der_sig(r, s):
    return der_seq(der_int(r), der_int(s))


_SIG_KEYS = ("Type", "CurveName", "HashingMethod", "SignatureType", "Padding", "Signature", "R", "S",
             "RecoveryParameter")


def _sig_method(o, kd, fname):
    m = o.get("Method")
    md = {}
    if m is not None and not is_sym(m, "Automatic"):
        md = IE.assoc_dict(m, fname)
    for k in md:
        if k not in ("Type", "HashingMethod", "SignatureType", "CurveName", "Padding", "HashName"):
            raise ValueError(f"{fname}: unknown Method key \"{k}\"")
    if "Type" in md and md["Type"] != kd["Type"] and not (md["Type"] in ("Bitcoin", "Ethereum")
                                                         and kd["Type"] == "EllipticCurve"):
        raise ValueError(f"{fname}: the Method type {md['Type']} does not match the {kd['Type']} key")
    if "CurveName" in md and kd["Type"] != "RSA" and CURVE_ALIASES.get(md["CurveName"], md["CurveName"]) \
            != kd["CurveName"]:
        raise ValueError(f"{fname}: the Method curve does not match the key's curve")
    hm = md.get("HashingMethod", md.get("HashName"))
    if "HashName" in md and "HashingMethod" in md and md["HashName"] != md["HashingMethod"]:
        raise ValueError(f"{fname}: \"HashName\" and \"HashingMethod\" differ")
    return md, hm


def sign(x, key, o, fname):
    if not _is_obj(key, "PrivateKey"):
        raise TypeError(f"{fname}: {F.input_form(key)} is not a PrivateKey")
    kd = asym_parts(key, fname)
    md, hm = _sig_method(o, kd, fname)
    t = kd["Type"]
    if t == "RSA":
        pad = md.get("Padding", "PKCS1")
        if pad == "X931":
            raise NotImplementedError(f"{fname}: X9.31 signature padding is not available")
        if pad != "PKCS1":
            raise ValueError(f"{fname}: {F.input_form(pad)} is not a supported signature padding")
        hm = "SHA256" if hm is None or is_sym(hm, "Automatic") else hm
        if is_sym(hm, "None"):
            raise ValueError(f"{fname}: RSA signatures need a HashingMethod")
        h = message_digest(x, hm, fname)
        sig = rsa_sign(kd["PublicModulus"], kd["PrivateExponent"], h, hm, fname)
        return Expr("DigitalSignature", A([("Type", "RSA"), ("HashingMethod", hm), ("Padding", "PKCS1"),
                                           ("Signature", _ba(sig))]))
    if t == "EdwardsCurve":
        if hm is not None and not is_sym(hm, "Automatic") and hm != "SHA512":
            raise ValueError(f"{fname}: ed25519 signatures always use SHA512")
        st = md.get("SignatureType", Symbol("Automatic"))
        if not (is_sym(st, "Automatic") or st == "Deterministic"):
            raise ValueError(f"{fname}: ed25519 signatures are deterministic")
        R, S = ed25519_sign(kd["Seed"], expr_bytes(x, fname))
        return Expr("DigitalSignature", A([("Type", "EdwardsCurve"), ("CurveName", "ed25519"),
                                           ("HashingMethod", "SHA512"), ("SignatureType", "Deterministic"),
                                           ("R", _ba(R)), ("S", _ba(S))]))
    curve = kd["Curve"]
    hm = "SHA256" if hm is None or is_sym(hm, "Automatic") else hm
    h = message_digest(x, None if is_sym(hm, "None") else hm, fname)
    hmv = None if is_sym(hm, "None") else hm
    st = md.get("SignatureType", Symbol("Automatic"))
    st = "Deterministic" if is_sym(st, "Automatic") else st
    xk = kd["PrivateMultiplier"]
    if st == "Schnorr":
        if curve.name != "secp256k1" or len(h) != 32:
            raise ValueError(f"{fname}: Schnorr (BIP-340) signatures need secp256k1 and a 32-byte digest")
        R, S = schnorr_sign(xk, h, bytes(32))
        return Expr("DigitalSignature", A([("Type", "EllipticCurve"), ("CurveName", curve.name),
                                           ("HashingMethod", hmv), ("SignatureType", "Schnorr"),
                                           ("R", _ba(R)), ("S", _ba(S))]))
    if st == "Deterministic":
        k = rfc6979_k(xk, h, curve, hmv or "SHA256")
    elif st == "NonDeterministic":
        k = secrets.randbelow(curve.n - 1) + 1
    else:
        raise ValueError(f"{fname}: {F.input_form(st)} is not a signature type (Deterministic, NonDeterministic, "
                         f"Schnorr)")
    r, s, rec = ecdsa_sign(curve, xk, h, k, low_s=curve.name == "secp256k1")
    items = [("Type", "EllipticCurve"), ("CurveName", curve.name), ("HashingMethod", hmv), ("SignatureType", st),
             ("R", _ba(r.to_bytes(curve.size, "big"))), ("S", _ba(s.to_bytes(curve.size, "big")))]
    if curve.name == "secp256k1":
        items.append(("RecoveryParameter", rec))
    return Expr("DigitalSignature", A(items))


def _sig_dict(sig, fname):
    if not _is_obj(sig, "DigitalSignature"):
        raise TypeError(f"{fname}: {F.input_form(sig)} is not a DigitalSignature")
    a = IE.assoc_dict(sig.args[0], fname)
    for k in a:
        if k not in _SIG_KEYS:
            raise ValueError(f"{fname}: unknown DigitalSignature field \"{k}\"")
    return a


def verify(x, sig, key, fname):
    kd = public_of(key, fname)
    a = _sig_dict(sig, fname)
    t = a.get("Type", "RSA" if "Signature" in a else "EllipticCurve")
    if t != kd["Type"]:
        return False
    hm = a.get("HashingMethod")
    if t == "RSA":
        h = message_digest(x, hm, fname)
        return rsa_verify(kd["PublicModulus"], kd["PublicExponent"], h, hm, ba_bytes(a["Signature"], fname))
    R, S = ba_bytes(a["R"], fname), ba_bytes(a["S"], fname)
    if t == "EdwardsCurve":
        return ed25519_verify(kd["Public"], expr_bytes(x, fname), R, S)
    curve = kd["Curve"]
    if CURVE_ALIASES.get(a.get("CurveName", curve.name), a.get("CurveName")) != curve.name:
        return False
    h = message_digest(x, Symbol("None") if hm is None or is_sym(hm, "None") else hm, fname)
    if a.get("SignatureType") == "Schnorr":
        return curve.name == "secp256k1" and len(h) == 32 and schnorr_verify(kd["PublicCurvePoint"][0], h, R, S)
    return ecdsa_verify(curve, kd["PublicCurvePoint"], h, int.from_bytes(R, "big"), int.from_bytes(S, "big"))


@fn("GenerateDigitalSignature", area=AREA, status="added",
    notes="GenerateDigitalSignature[expr, privatekey] signs expr (strings as UTF-8, ByteArrays literally, other "
          "expressions as their FullForm text): RSA PKCS#1 v1.5, ECDSA with RFC 6979 deterministic nonces (or "
          "\"NonDeterministic\"; low-s on secp256k1) or BIP-340 \"Schnorr\", ed25519 (RFC 8032); Method -> "
          "<|\"HashingMethod\" (default SHA256; None for a precomputed ByteArray/hex digest), \"SignatureType\", "
          "\"Padding\" (PKCS1; X931 gives a clear error)|>; operator form GenerateDigitalSignature[key]. Check: reference (RFC 6979, RFC 8032, BIP-340 vectors) and VerifyDigitalSignature in tests.")
def _generatedigitalsignature(*args: Expr | Option) -> Expr:
    rest, o = split_opts(args, ("Method",), "GenerateDigitalSignature")
    if len(rest) == 1:
        if o:
            raise ValueError("GenerateDigitalSignature: the operator form takes no options")
        raise Unevaluated
    if len(rest) != 2:
        raise ValueError("GenerateDigitalSignature: called with the wrong number of arguments")
    return sign(rest[0], rest[1], o, "GenerateDigitalSignature")


@fn("DigitalSignature", area=AREA, status="added",
    notes="DigitalSignature[<|...|>] is a signature from GenerateDigitalSignature (RSA: Type, HashingMethod, "
          "Padding, Signature; elliptic/Edwards curves: Type, CurveName, HashingMethod, SignatureType, R, S and "
          "RecoveryParameter on secp256k1); properties by name plus \"DER\" (ECDSA) and \"Properties\". Check: exact (validated wrapper).")
def _digitalsignature(a: Assoc) -> Expr:
    if not KL.is_assoc(a):
        raise TypeError(f"DigitalSignature: {F.input_form(a)} is not an association")
    _sig_dict(Expr("DigitalSignature", a), "DigitalSignature")
    raise Unevaluated


def _pairs(spec, fname):
    if is_list(spec) and len(spec.args) == 2 and _is_obj(spec.args[1], "DigitalSignature"):
        return [spec.args], False
    if is_list(spec) and spec.args and all(is_list(p) and len(p.args) == 2 for p in spec.args):
        return [p.args for p in spec.args], True
    raise ValueError(f"{fname}: expected {{expr, DigitalSignature[...]}} or a list of such pairs")


@fn("VerifyDigitalSignature", area=AREA, status="added",
    notes="VerifyDigitalSignature[{expr, sig}, key] gives True when sig is a valid signature of expr for the "
          "public key (a PrivateKey or a SecurityCertificate also supplies it), False otherwise; a list of pairs "
          "gives a list; operator form VerifyDigitalSignature[key]. Check: reference (OpenSSL signatures and certificates) in tests.")
def _verifydigitalsignature(*args: Expr) -> Boolean | ListOf[Boolean]:
    if len(args) == 1:
        raise Unevaluated
    if len(args) != 2:
        raise ValueError("VerifyDigitalSignature: called with the wrong number of arguments")
    pairs, many = _pairs(args[0], "VerifyDigitalSignature")
    res = [verify(x, s, args[1], "VerifyDigitalSignature") for x, s in pairs]
    return L(*res) if many else res[0]


def _file_bytes(f, fname):
    path = file_arg(f, fname)
    if not os.path.isfile(path):
        raise ValueError(f"{fname}::nffil: file {path} not found")
    with open(path, "rb") as fh:
        return fh.read()


@fn("GenerateFileSignature", area=AREA, status="added",
    notes="GenerateFileSignature[file, privatekey] gives the DigitalSignature of the bytes of file (same methods "
          "and Method option as GenerateDigitalSignature); operator form GenerateFileSignature[key]. Check: reference; verified by VerifyFileSignature and an OpenSSL signature in tests.")
def _generatefilesignature(*args: String | Expr | Option) -> Expr:
    rest, o = split_opts(args, ("Method",), "GenerateFileSignature")
    if len(rest) == 1:
        raise Unevaluated
    if len(rest) != 2:
        raise ValueError("GenerateFileSignature: called with the wrong number of arguments")
    return sign(_ba(_file_bytes(rest[0], "GenerateFileSignature")), rest[1], o, "GenerateFileSignature")


@fn("VerifyFileSignature", area=AREA, status="added",
    notes="VerifyFileSignature[{file, sig}, key] verifies a GenerateFileSignature signature against the current "
          "bytes of file; a list of pairs gives a list; operator form VerifyFileSignature[key]. Check: reference (an OpenSSL ECDSA file signature) in tests.")
def _verifyfilesignature(*args: Expr) -> Boolean | ListOf[Boolean]:
    if len(args) == 1:
        raise Unevaluated
    if len(args) != 2:
        raise ValueError("VerifyFileSignature: called with the wrong number of arguments")
    spec = args[0]
    if is_list(spec) and len(spec.args) == 2 and _is_obj(spec.args[1], "DigitalSignature"):
        pairs, many = [spec.args], False
    else:
        pairs, many = _pairs(spec, "VerifyFileSignature")
    res = [verify(_ba(_file_bytes(f, "VerifyFileSignature")), s, args[1], "VerifyFileSignature") for f, s in pairs]
    return L(*res) if many else res[0]


def _swap_operator_hook(fn_name):
    def hook(head, args, ev_):
        if len(head.args) == 1 and len(args) == 1:
            return ev_(Expr(fn_name, args[0], head.args[0]))
        return K.UNHANDLED
    return hook


for _n in ("GenerateDigitalSignature", "VerifyDigitalSignature", "GenerateFileSignature", "VerifyFileSignature"):
    KL.COMPOUND_HEAD_HOOKS[_n] = _swap_operator_hook(_n)


# ═══════════════════════════════════════ DER / PEM ═══════════════════════════════════════

def _der_len(n):
    if n < 128:
        return bytes([n])
    b = n.to_bytes((n.bit_length() + 7) // 8, "big")
    return bytes([0x80 | len(b)]) + b


def der_tlv(tag, content):
    return bytes([tag]) + _der_len(len(content)) + content


def der_int(v):
    b = v.to_bytes(max(1, (v.bit_length() + 8) // 8), "big", signed=True) if v >= 0 else \
        v.to_bytes((v.bit_length() + 8) // 8, "big", signed=True)
    return der_tlv(0x02, b)


def der_seq(*items):
    return der_tlv(0x30, b"".join(items))


def der_oid(dotted):
    parts = [int(p) for p in dotted.split(".")]
    body = [40 * parts[0] + parts[1]] + parts[2:]
    out = bytearray()
    for v in body:
        enc = [v & 0x7F]
        v >>= 7
        while v:
            enc.append(0x80 | (v & 0x7F))
            v >>= 7
        out += bytes(reversed(enc))
    return der_tlv(0x06, bytes(out))


def der_octet(b):
    return der_tlv(0x04, b)


def der_bits(b):
    return der_tlv(0x03, b"\x00" + b)


DER_NULL = b"\x05\x00"


class DerError(ValueError):
    pass


def der_read(b, i=0):
    """(tag, content, raw_tlv, next index)."""
    if i + 2 > len(b):
        raise DerError("truncated DER data")
    tag = b[i]
    ln = b[i + 1]
    j = i + 2
    if ln & 0x80:
        nb = ln & 0x7F
        if nb == 0 or nb > 4 or j + nb > len(b):
            raise DerError("invalid DER length")
        ln = int.from_bytes(b[j:j + nb], "big")
        j += nb
    if j + ln > len(b):
        raise DerError("truncated DER data")
    return tag, b[j:j + ln], b[i:j + ln], j + ln


def der_items(content):
    out, i = [], 0
    while i < len(content):
        t, c, raw, i = der_read(content, i)
        out.append((t, c, raw))
    return out


def der_to_int(c):
    return int.from_bytes(c, "big", signed=True)


def oid_text(c):
    vals, v = [], 0
    for byte in c:
        v = (v << 7) | (byte & 0x7F)
        if not byte & 0x80:
            vals.append(v)
            v = 0
    if not vals:
        raise DerError("empty OID")
    first = vals[0]
    head = [0, first] if first < 40 else ([1, first - 40] if first < 80 else [2, first - 80])
    return ".".join(str(x) for x in head + vals[1:])


OID_RSA, OID_EC, OID_ED25519 = "1.2.840.113549.1.1.1", "1.2.840.10045.2.1", "1.3.101.112"
_CURVE_OID = {c.oid: c for c in CURVES.values()}


def _rsa_factor(n, e, d):
    import math
    k = d * e - 1
    t = 0
    while k % 2 == 0:
        k //= 2
        t += 1
    for g in range(2, 1000):
        x = pow(g, k, n)
        for _ in range(t):
            y = pow(x, 2, n)
            if y == 1 and x not in (1, n - 1):
                p = math.gcd(x - 1, n)
                return p, n // p
            x = y
    raise ValueError("Export: cannot factor the RSA modulus from the key")


def _rsa_pkcs1_private(d):
    n, e, dd = d["PublicModulus"], d["PublicExponent"], d["PrivateExponent"]
    p, q = _rsa_factor(n, e, dd)
    if p < q:
        p, q = q, p
    return der_seq(der_int(0), der_int(n), der_int(e), der_int(dd), der_int(p), der_int(q), der_int(dd % (p - 1)),
                   der_int(dd % (q - 1)), der_int(pow(q, -1, p)))


def key_to_der(d):
    """(PEM label, DER) for a key dict (PKCS#8 for private keys, SubjectPublicKeyInfo for public keys)."""
    t = d["Type"]
    if d["Private"]:
        if t == "RSA":
            alg, inner = der_seq(der_oid(OID_RSA), DER_NULL), _rsa_pkcs1_private(d)
        elif t == "EllipticCurve":
            c = d["Curve"]
            alg = der_seq(der_oid(OID_EC), der_oid(c.oid))
            inner = der_seq(der_int(1), der_octet(d["PrivateMultiplier"].to_bytes(c.size, "big")),
                            der_tlv(0xA1, der_bits(c.encode(d["PublicCurvePoint"], False))))
        else:
            alg, inner = der_seq(der_oid(OID_ED25519)), der_octet(d["Seed"])
        return "PRIVATE KEY", der_seq(der_int(0), alg, der_octet(inner))
    if t == "RSA":
        alg, pub = der_seq(der_oid(OID_RSA), DER_NULL), der_seq(der_int(d["PublicModulus"]),
                                                                 der_int(d["PublicExponent"]))
    elif t == "EllipticCurve":
        c = d["Curve"]
        alg, pub = der_seq(der_oid(OID_EC), der_oid(c.oid)), c.encode(d["PublicCurvePoint"], d["Compressed"])
    else:
        alg, pub = der_seq(der_oid(OID_ED25519)), d["Public"]
    return "PUBLIC KEY", der_seq(alg, der_bits(pub))


def _curve_from_oid(oid, fname):
    c = _CURVE_OID.get(oid)
    if c is None:
        raise ValueError(f"{fname}: unsupported elliptic curve (OID {oid})")
    return c


def _rsa_priv_from_pkcs1(der):
    it = der_items(der_read(der)[1])
    v = [der_to_int(c) for t, c, _ in it[:4]]
    return rsa_private_expr(v[1], v[2], v[3])


def _ec_priv_from_sec1(der, curve, fname):
    it = der_items(der_read(der)[1])
    x = int.from_bytes(it[1][1], "big")
    for t, c, _ in it[2:]:
        if t == 0xA0:
            curve = _curve_from_oid(oid_text(der_read(c)[1]), fname)
    if curve is None:
        raise ValueError(f"{fname}: the EC private key does not name its curve")
    return ec_private_expr(curve, x)


def _spki_to_key(der, fname):
    alg, bits = der_items(der_read(der)[1])[:2]
    algi = der_items(alg[1])
    oid = oid_text(algi[0][1])
    pub = bits[1][1:]
    if oid == OID_RSA:
        n, e = (der_to_int(c) for t, c, _ in der_items(der_read(pub)[1])[:2])
        return rsa_public_expr(n, e)
    if oid == OID_EC:
        c = _curve_from_oid(oid_text(algi[1][1]), fname)
        return ec_public_expr(c, c.decode(pub, fname), pub[0] in (2, 3))
    if oid == OID_ED25519:
        return ed_public_expr(pub)
    raise ValueError(f"{fname}: unsupported public key algorithm (OID {oid})")


def _pkcs8_to_key(der, fname):
    it = der_items(der_read(der)[1])
    algi = der_items(it[1][1])
    oid = oid_text(algi[0][1])
    inner = it[2][1]
    if oid == OID_RSA:
        return _rsa_priv_from_pkcs1(inner)
    if oid == OID_EC:
        return _ec_priv_from_sec1(inner, _curve_from_oid(oid_text(algi[1][1]), fname), fname)
    if oid == OID_ED25519:
        return ed_private_expr(der_read(inner)[1])
    raise ValueError(f"{fname}: unsupported private key algorithm (OID {oid})")


# ── X.509 certificates ──

_SIG_ALGS = {"1.2.840.113549.1.1.4": ("md5WithRSAEncryption", "RSA", "MD5"),
             "1.2.840.113549.1.1.5": ("sha1WithRSAEncryption", "RSA", "SHA1"),
             "1.2.840.113549.1.1.11": ("sha256WithRSAEncryption", "RSA", "SHA256"),
             "1.2.840.113549.1.1.12": ("sha384WithRSAEncryption", "RSA", "SHA384"),
             "1.2.840.113549.1.1.13": ("sha512WithRSAEncryption", "RSA", "SHA512"),
             "1.2.840.113549.1.1.14": ("sha224WithRSAEncryption", "RSA", "SHA224"),
             "1.2.840.10045.4.1": ("ecdsa-with-SHA1", "EllipticCurve", "SHA1"),
             "1.2.840.10045.4.3.1": ("ecdsa-with-SHA224", "EllipticCurve", "SHA224"),
             "1.2.840.10045.4.3.2": ("ecdsa-with-SHA256", "EllipticCurve", "SHA256"),
             "1.2.840.10045.4.3.3": ("ecdsa-with-SHA384", "EllipticCurve", "SHA384"),
             "1.2.840.10045.4.3.4": ("ecdsa-with-SHA512", "EllipticCurve", "SHA512"),
             "1.3.101.112": ("ED25519", "EdwardsCurve", "SHA512")}
_NAME_ATTRS = {"2.5.4.3": "CommonName", "2.5.4.6": "CountryName", "2.5.4.7": "LocalityName",
               "2.5.4.8": "StateOrProvinceName", "2.5.4.10": "OrganizationName",
               "2.5.4.11": "OrganizationalUnitName", "2.5.4.5": "SerialNumber", "2.5.4.9": "StreetAddress",
               "2.5.4.12": "Title", "2.5.4.42": "GivenName", "2.5.4.4": "Surname",
               "1.2.840.113549.1.9.1": "EmailAddress", "0.9.2342.19200300.100.1.25": "DomainComponent"}
_EXTENSIONS = {"2.5.29.14": "SubjectKeyIdentifier", "2.5.29.35": "AuthorityKeyIdentifier",
               "2.5.29.19": "BasicConstraints", "2.5.29.15": "KeyUsage", "2.5.29.17": "SubjectAlternativeName",
               "2.5.29.37": "ExtendedKeyUsage", "2.5.29.31": "CRLDistributionPoints",
               "2.5.29.32": "CertificatePolicies", "1.3.6.1.5.5.7.1.1": "AuthorityInformationAccess"}


def _der_string(t, c):
    if t == 0x1E:
        return c.decode("utf-16-be", "replace")
    if t == 0x1C:
        return c.decode("utf-32-be", "replace")
    return c.decode("utf-8" if t == 0x0C else "latin-1", "replace")


def _name(c):
    out = []
    for _, rdn, _ in der_items(c):
        for _, atv, _ in der_items(rdn):
            (ot, oc, _), (vt, vc, _) = der_items(atv)[:2]
            o = oid_text(oc)
            out.append((_NAME_ATTRS.get(o, o), _der_string(vt, vc)))
    return A(out)


def _der_time(t, c):
    s = c.decode("ascii")
    m = re.fullmatch((r"(\d\d)" if t == 0x17 else r"(\d{4})") + r"(\d\d)(\d\d)(\d\d)(\d\d)(\d\d)?(?:\.\d+)?Z", s)
    if not m:
        raise DerError(f"unsupported certificate time {s!r}")
    y = int(m.group(1))
    if t == 0x17:
        y += 2000 if y < 50 else 1900
    vals = [y, int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6) or 0)]
    return ev(Expr("DateObject", L(*vals), Expr("Rule", Symbol("TimeZone"), 0)))


def _ecdsa_sig_from_der(b, curve, hm, fname):
    r, s = (int.from_bytes(c, "big") for t, c, _ in der_items(der_read(b)[1])[:2])
    size = curve.size if curve else max((r.bit_length() + 7) // 8, (s.bit_length() + 7) // 8)
    return [("Type", "EllipticCurve"), ("CurveName", curve.name if curve else None), ("HashingMethod", hm),
            ("SignatureType", "Deterministic"), ("R", _ba(r.to_bytes(size, "big"))), ("S", _ba(s.to_bytes(size,
                                                                                                         "big")))]


def certificate_from_der(der, fname):
    try:
        tbs, alg, sigbits = der_items(der_read(der)[1])[:3]
        f = der_items(tbs[1])
        i = 0
        version = 1
        if f[0][0] == 0xA0:
            version = der_to_int(der_read(f[0][1])[1]) + 1
            i = 1
        serial = der_to_int(f[i][1])
        issuer, validity, subject, spki = f[i + 2], f[i + 3], f[i + 4], f[i + 5]
        exts = []
        for t, c, _ in f[i + 6:]:
            if t == 0xA3:
                for _, ext, _ in der_items(der_read(c)[1]):
                    parts = der_items(ext)
                    o = oid_text(parts[0][1])
                    crit = len(parts) == 3 and parts[1][1] != b"\x00"
                    exts.append((_EXTENSIONS.get(o, o), A([("Critical", crit), ("Value", _ba(parts[-1][1]))])))
        sig_oid = oid_text(der_items(alg[1])[0][1])
        sname, stype, hm = _SIG_ALGS.get(sig_oid, (sig_oid, None, None))
        start, end = (_der_time(t, c) for t, c, _ in der_items(validity[1])[:2])
        pub = _spki_to_key(spki[2], fname)
        sig = sigbits[1][1:]
    except (DerError, IndexError, UnicodeDecodeError) as e:
        raise ValueError(f"{fname}: invalid X.509 certificate ({e})")
    if stype == "RSA":
        dsig = Expr("DigitalSignature", A([("Type", "RSA"), ("HashingMethod", hm), ("Padding", "PKCS1"),
                                           ("Signature", _ba(sig))]))
    elif stype == "EllipticCurve":
        curve = None
        try:
            curve = _CURVE_OID.get(oid_text(der_items(der_items(der_read(spki[2])[1])[0][1])[1][1]))
        except (DerError, IndexError):
            pass
        dsig = Expr("DigitalSignature", A(_ecdsa_sig_from_der(sig, curve, hm, fname)))
    elif stype == "EdwardsCurve":
        dsig = Expr("DigitalSignature", A([("Type", "EdwardsCurve"), ("CurveName", "ed25519"),
                                           ("HashingMethod", "SHA512"), ("SignatureType", "Deterministic"),
                                           ("R", _ba(sig[:32])), ("S", _ba(sig[32:]))]))
    else:
        dsig = Expr("Missing", "NotAvailable")
    return Expr("SecurityCertificate", A([
        ("Version", version), ("SerialNumber", serial), ("SignatureAlgorithm", sname), ("Issuer", _name(issuer[1])),
        ("ValidityStartDate", start), ("ValidityEndDate", end), ("Subject", _name(subject[1])), ("PublicKey", pub),
        ("Extensions", A(exts)), ("DigitalSignature", dsig), ("SignedData", _ba(tbs[2])),
        ("Certificate", _ba(der))]))


_PEM_RE = re.compile(rb"-----BEGIN ([A-Z0-9 ]+)-----\s*(.*?)\s*-----END \1-----", re.S)


def pem_entries(data, fname):
    out = []
    for m in _PEM_RE.finditer(data):
        label = m.group(1).decode("ascii")
        body = m.group(2)
        if b":" in body.split(b"\n", 1)[0]:
            out.append(Expr("ImportedObject", A([("OriginalFormat", "PEM"), ("Type", label),
                                                 ("RawContent", m.group(0).decode("ascii", "replace"))])))
            continue
        try:
            der = base64.b64decode(re.sub(rb"\s+", b"", body), validate=True)
        except Exception:
            raise ValueError(f"{fname}: invalid Base64 in the PEM entry {label}")
        try:
            if label == "CERTIFICATE":
                out.append(certificate_from_der(der, fname))
            elif label == "PUBLIC KEY":
                out.append(_spki_to_key(der, fname))
            elif label == "RSA PUBLIC KEY":
                n, e = (der_to_int(c) for t, c, _ in der_items(der_read(der)[1])[:2])
                out.append(rsa_public_expr(n, e))
            elif label == "PRIVATE KEY":
                out.append(_pkcs8_to_key(der, fname))
            elif label == "RSA PRIVATE KEY":
                out.append(_rsa_priv_from_pkcs1(der))
            elif label == "EC PRIVATE KEY":
                out.append(_ec_priv_from_sec1(der, None, fname))
            else:
                out.append(Expr("ImportedObject", A([("OriginalFormat", "PEM"), ("Type", label),
                                                     ("RawContent", _ba(der))])))
        except (DerError, IndexError) as e:
            raise ValueError(f"{fname}: invalid {label} PEM entry ({e})")
    if not out:
        raise ValueError(f"{fname}: no PEM entries found")
    return out


def _pem_block(label, der):
    b = base64.b64encode(der).decode("ascii")
    return f"-----BEGIN {label}-----\n" + "\n".join(b[i:i + 64] for i in range(0, len(b), 64)) + \
        f"\n-----END {label}-----\n"


def to_pem(x, fname):
    items = list(x.args) if is_list(x) else [x]
    out = []
    for it in items:
        if _is_obj(it, "PrivateKey") or _is_obj(it, "PublicKey"):
            out.append(_pem_block(*key_to_der(asym_parts(it, fname))))
        elif _is_obj(it, "SecurityCertificate"):
            raw = assoc_get(it.args[0], "Certificate")
            if raw is None:
                raise ValueError(f"{fname}: the SecurityCertificate has no encoded certificate")
            out.append(_pem_block("CERTIFICATE", ba_bytes(raw, fname)))
        elif _is_obj(it, "ImportedObject") and assoc_get(it.args[0], "OriginalFormat") == "PEM":
            raw = assoc_get(it.args[0], "RawContent")
            out.append(_pem_block(assoc_get(it.args[0], "Type"), ba_bytes(raw, fname)) if is_byte_array(raw)
                       else raw.rstrip("\n") + "\n")
        else:
            raise ValueError(f"{fname}: PEM export needs PrivateKey, PublicKey or SecurityCertificate objects, not "
                             f"{F.input_form(it)}")
    return "".join(out).encode("ascii")


_PEM_ELEMENTS = {"PrivateKey": "PrivateKey", "PublicKey": "PublicKey", "SecurityCertificate": "SecurityCertificate"}


def _imp_pem(data, elem, subs, opts):
    ents = pem_entries(data, "Import")
    if elem in _PEM_ELEMENTS:
        ents = [e for e in ents if e.head == elem]
        if not ents:
            return Expr("Missing", "NotAvailable")
    return ents[0] if len(ents) == 1 else L(*ents)


IE.register_format(IE.Format("PEM", ["pem", "cer", "crt", "key"], ["application/x-pem-file"],
                             ["Data", "PrivateKey", "PublicKey", "SecurityCertificate"], "Data", _imp_pem,
                             lambda x, opts: to_pem(x, "Export")))


@fn("SecurityCertificate", area=AREA, status="added",
    notes="SecurityCertificate[<|...|>] represents an X.509 certificate imported from PEM (properties Version, "
          "SerialNumber, SignatureAlgorithm, Issuer, ValidityStartDate, ValidityEndDate, Subject, PublicKey, "
          "Extensions, DigitalSignature, SignedData, Certificate, Properties); SecurityCertificate[\"PEM text\"] "
          "or SecurityCertificate[ByteArray (DER)] parses one; VerifyDigitalSignature[{cert[\"SignedData\"], "
          "cert[\"DigitalSignature\"]}, issuerKey] checks its signature. Check: reference (OpenSSL certificates; their signatures verify) in tests.")
def _securitycertificate(x: String | Expr | Assoc) -> Expr:
    if isinstance(x, str):
        certs = [e for e in pem_entries(x.encode("latin-1", "replace"), "SecurityCertificate")
                 if e.head == "SecurityCertificate"]
        if len(certs) != 1:
            raise ValueError("SecurityCertificate: the text must contain exactly one PEM certificate")
        return certs[0]
    if is_byte_array(x):
        return certificate_from_der(ba_bytes(x), "SecurityCertificate")
    if KL.is_assoc(x):
        raise Unevaluated
    raise TypeError(f"SecurityCertificate: {F.input_form(x)} is not a PEM string, a DER ByteArray or an "
                     f"association")


# ═══════════════════════════════════════ files, objects, properties ═══════════════════════════════════════

@fn("EncryptedObject", area=AREA, status="added",
    notes="EncryptedObject[<|\"Data\", \"InitializationVector\", \"OriginalForm\", ...|>] is the result of "
          "Encrypt (properties by name and \"Properties\"); validated, otherwise inert. Check: exact (validated wrapper).")
def _encryptedobject(a: Assoc) -> Expr:
    if not KL.is_assoc(a) or assoc_get(a, "Data") is None or not is_byte_array(assoc_get(a, "Data")):
        raise ValueError("EncryptedObject: expected an association with a ByteArray \"Data\"")
    raise Unevaluated


def _write_new(path, data, overwrite, fname):
    if os.path.exists(path) and not overwrite:
        raise ValueError(f"{fname}::exists: the file {path} already exists (use OverwriteTarget -> True)")
    with open(path, "wb") as fh:
        fh.write(data)
    return os.path.abspath(path)


def _overwrite(o, fname):
    v = o.get("OverwriteTarget", False)
    if not isinstance(v, bool):
        raise TypeError(f"{fname}: OverwriteTarget must be True or False")
    return v


@fn("EncryptFile", area=AREA, status="added",
    notes="EncryptFile[key, file] encrypts the bytes of file into file.mx (Import gives the EncryptedObject) and "
          "EncryptFile[key, source, target] into target; key as for Encrypt; OverwriteTarget (default False); "
          "returns the absolute output path. Check: exact; DecryptFile round trips in tests.")
def _encryptfile(key: Expr | String, src: String | Expr, *rest: String | Option) -> String:
    rest, o = split_opts(rest, ("OverwriteTarget", "Method"), "EncryptFile")
    if len(rest) > 1:
        raise ValueError("EncryptFile: called with too many arguments")
    path = file_arg(src, "EncryptFile")
    obj = encrypt(key, _ba(_file_bytes(path, "EncryptFile")), o, "EncryptFile")
    target = file_arg(rest[0], "EncryptFile") if rest else path + ".mx"
    return _write_new(target, IE.FORMATS["MX"].exp(obj, {}), _overwrite(o, "EncryptFile"), "EncryptFile")


@fn("DecryptFile", area=AREA, status="added",
    notes="DecryptFile[key, \"file.mx\"] decrypts an EncryptFile result into file and DecryptFile[key, source, "
          "target] into target; OverwriteTarget (default False); returns the absolute output path. Check: exact; EncryptFile round trips in tests.")
def _decryptfile(key: Expr | String, src: String | Expr, *rest: String | Option) -> String:
    rest, o = split_opts(rest, ("OverwriteTarget", "Method"), "DecryptFile")
    if len(rest) > 1:
        raise ValueError("DecryptFile: called with too many arguments")
    path = file_arg(src, "DecryptFile")
    data = _file_bytes(path, "DecryptFile")
    if not data.startswith(IE.MX_HEADER):
        raise ValueError("DecryptFile: the file is not an EncryptFile result")
    obj = IE.do_import(data, IE.FORMATS["MX"], "\x00default", [], None, {}, "DecryptFile")
    if rest:
        target = file_arg(rest[0], "DecryptFile")
    elif path.endswith(".mx"):
        target = path[:-3]
    else:
        raise ValueError("DecryptFile: give a target file (the source does not end in .mx)")
    res = decrypt(key, obj, o, "DecryptFile")
    if not is_byte_array(res):
        raise ValueError("DecryptFile: the encrypted data is not a file")
    return _write_new(target, ba_bytes(res), _overwrite(o, "DecryptFile"), "DecryptFile")


def _props_hook(computed=None):
    def hook(head, args, ev_):
        if not (len(head.args) == 1 and KL.is_assoc(head.args[0]) and len(args) == 1 and isinstance(args[0], str)):
            return K.UNHANDLED
        a = head.args[0]
        extra = {}
        if computed is not None:
            try:
                extra = computed(head)
            except (ValueError, TypeError, NotImplementedError):
                extra = {}
        stored = [r.args[0] for r in a.args]
        if args[0] == "Properties":
            return L(*sorted(set(stored) | set(extra)))
        v = assoc_get(a, args[0], K.UNHANDLED)
        if v is K.UNHANDLED:
            v = extra.get(args[0], K.UNHANDLED)
        return Expr("Missing", "NotAvailable") if v is K.UNHANDLED else v
    return hook


def _key_computed(k):
    return _key_props(asym_parts(k, k.head))


def _sig_computed(s):
    a = IE.assoc_dict(s.args[0], "DigitalSignature")
    out = {}
    if a.get("Type") == "EllipticCurve" and a.get("SignatureType") != "Schnorr":
        r, s_ = (int.from_bytes(ba_bytes(a[k]), "big") for k in ("R", "S"))
        out["DER"] = _ba(_der_sig(r, s_))
    return out


def _dk_computed(k):
    return {"PHCString": phc_string(k)}


for _h, _c in (("PrivateKey", _key_computed), ("PublicKey", _key_computed), ("DigitalSignature", _sig_computed),
               ("DerivedKey", _dk_computed), ("SymmetricKey", None), ("EncryptedObject", None),
               ("SecurityCertificate", None)):
    KL.COMPOUND_HEAD_HOOKS[_h] = _props_hook(_c)

CURVE_NAMES = sorted(set(CURVES) | {"prime256v1", "prime192v1"} | set(EDWARDS_CURVES))
IE._sysvar("$CryptographicEllipticCurveNames", lambda: L(*CURVE_NAMES))
record("$CryptographicEllipticCurveNames", AREA, "added",
       "list of the elliptic and Edwards curves the cryptographic functions support. Check: exact; "
       "tests verify each curve's parameters (G on the curve, n*G = O, n and p prime).")
