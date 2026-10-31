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

    def decode(self, b):
        if len(b) == 1 + 2 * self.size and b[0] == 4:
            P = (int.from_bytes(b[1:1 + self.size], "big"), int.from_bytes(b[1 + self.size:], "big"))
        elif len(b) == 1 + self.size and b[0] in (2, 3):
            P = self.lift_x(int.from_bytes(b[1:], "big"), b[0] - 2)
        else:
            raise ValueError("invalid encoded curve point")
        if not self.on_curve(P):
            raise ValueError("the point is not on the curve")
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
        raise ValueError(f"{fname}: {F.input_form(name)} is not an elliptic curve name")
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
