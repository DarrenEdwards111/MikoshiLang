"""Strings, encodings, hashes and dates.  Positions are 1-based as in the
Wolfram Language (legacy StringPart/StringTake/StringInsert were 0-based)."""
import base64
import calendar
import datetime as _dt
import hashlib

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, R, L, is_list, int_arg, nonneg_int, pos_int, list_arg,
                   string_arg)

AREA = "strings and dates"


def _s(x, fname, what="argument"):
    return string_arg(x, fname, what)


def _span(n, length, fname):
    """Wolfram take/drop spec -> python slice indices (0-based, end exclusive)."""
    if is_list(n):
        a = [int_arg(v, fname, "position") for v in n.args]
        if len(a) == 1:
            i = a[0]
            i = i if i > 0 else length + i + 1
            if not 1 <= i <= length:
                raise ValueError(f"{fname}: position {a[0]} is out of range for a string of length {length}")
            return i - 1, i
        if len(a) == 2:
            i, j = (v if v > 0 else length + v + 1 for v in a)
            if not (1 <= i <= length + 1 and 0 <= j <= length) or j < i - 1:
                raise ValueError(f"{fname}: range {a} is out of range for a string of length {length}")
            return i - 1, j
        raise ValueError(f"{fname}: invalid position specification")
    k = int_arg(n, fname, "count")
    if abs(k) > length:
        raise ValueError(f"{fname}: cannot take {abs(k)} characters from a string of length {length}")
    return (0, k) if k >= 0 else (length + k, length)


@fn("StringTake", area=AREA, status="fixed",
    notes="StringTake[s, n], [s, -n], [s, {n}], [s, {m, n}] (legacy: only the first n, and Python slicing "
          "silently accepted negative/out-of-range n)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _stake(s, n):
    st = _s(s, "StringTake")
    i, j = _span(n, len(st), "StringTake")
    return st[i:j]


@fn("StringDrop", area=AREA, status="fixed", notes="StringDrop[s, n], [s, -n], [s, {n}], [s, {m, n}]"
                                                   ". Check: exact (exact input gives exact results), reference "
                                                   "(differential cases in tests/library).")
def _sdrop(s, n):
    st = _s(s, "StringDrop")
    i, j = _span(n, len(st), "StringDrop")
    return st[:i] + st[j:]


@fn("StringPart", area=AREA, status="fixed",
    notes="1-based (legacy was 0-based: StringPart[\"abc\", 1] gave \"b\"); negative positions count from the end; "
          "a list of positions gives a list"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _spart(s, n):
    st = _s(s, "StringPart")
    if is_list(n):
        return L(*(_spart(s, v) for v in n.args))
    i, j = _span(L(n), len(st), "StringPart")
    return st[i:j]


@fn("StringInsert", area=AREA, status="fixed",
    notes="Wolfram argument order StringInsert[s, new, n]: new starts at position n (negative n counts from the "
          "end, -1 appends); legacy was StringInsert[s, i, new] with 0-based i"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _sinsert(s, new, n):
    st = _s(s, "StringInsert")
    if isinstance(new, int) and isinstance(n, str):
        raise ValueError("StringInsert: the argument order is StringInsert[s, new, n]")
    nw = _s(new, "StringInsert", "inserted string")
    k = int_arg(n, "StringInsert", "position")
    pos = k - 1 if k > 0 else len(st) + k + 1
    if k == 0 or not 0 <= pos <= len(st):
        raise ValueError(f"StringInsert: position {k} is out of range")
    return st[:pos] + nw + st[pos:]


def _pad_char(c, fname):
    ch = _s(c, fname, "padding")
    if len(ch) != 1:
        raise NotImplementedError(f"{fname}: only single-character padding is supported")
    return ch


@fn("StringPadLeft", area=AREA, status="fixed",
    notes="pads on the left or truncates to the rightmost n characters, as Wolfram (legacy never truncated); "
          "single-character padding"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _padl(s, n, c=" "):
    st, k, ch = _s(s, "StringPadLeft"), nonneg_int(n, "StringPadLeft", "length"), _pad_char(c, "StringPadLeft")
    return st[len(st) - k:] if len(st) >= k else ch * (k - len(st)) + st


@fn("StringPadRight", area=AREA, status="fixed",
    notes="pads on the right or truncates to the leftmost n characters, as Wolfram; single-character padding"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _padr(s, n, c=" "):
    st, k, ch = _s(s, "StringPadRight"), nonneg_int(n, "StringPadRight", "length"), _pad_char(c, "StringPadRight")
    return st[:k] if len(st) >= k else st + ch * (k - len(st))


@fn("StringCenter", area=AREA, status="fixed",
    notes="centres s in width n (extra padding on the right, MikoshiLang); single-character fill, width checked"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _center(s, n, c=" "):
    st, k, ch = _s(s, "StringCenter"), nonneg_int(n, "StringCenter", "width"), _pad_char(c, "StringCenter")
    if len(st) >= k:
        return st
    total = k - len(st)
    left = total // 2
    return ch * left + st + ch * (total - left)


@fn("StringRepeat", area=AREA, status="fixed", notes="StringRepeat[s, n] for n >= 1; legacy truncated non-integers"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _srepeat(s, n):
    return _s(s, "StringRepeat") * pos_int(n, "StringRepeat", "count")


@fn("StringReverse", area=AREA, status="verified", notes="reverses the characters"
                                                         ". Check: exact (exact input gives exact results), "
                                                         "reference (differential cases in tests/library).")
def _srev(s):
    return _s(s, "StringReverse")[::-1]


@fn("StringToUpperCase", area=AREA, status="verified", notes="alias of ToUpperCase"
                                                             ". Check: exact (exact input gives exact results), "
                                                             "reference (differential cases in tests/library).")
def _upper(s):
    return _s(s, "StringToUpperCase").upper()


@fn("StringToLowerCase", area=AREA, status="verified", notes="alias of ToLowerCase"
                                                             ". Check: exact (exact input gives exact results), "
                                                             "reference (differential cases in tests/library).")
def _lower(s):
    return _s(s, "StringToLowerCase").lower()


@fn("StringCapitalize", area=AREA, status="fixed",
    notes="upper-cases the first character only (Wolfram Capitalize); legacy also lower-cased the rest "
          "(\"iPhone\" -> \"Iphone\")"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _capitalize(s):
    st = _s(s, "StringCapitalize")
    return st[:1].upper() + st[1:]


@fn("StringToTitleCase", area=AREA, status="fixed",
    notes="capitalises the first letter of each whitespace-separated word and leaves the rest unchanged; legacy "
          "str.title() mangled words (\"don't\" -> \"Don'T\", \"NASA\" -> \"Nasa\")"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _title(s):
    st = _s(s, "StringToTitleCase")
    out, prev_space = [], True
    for ch in st:
        out.append(ch.upper() if prev_space else ch)
        prev_space = ch.isspace()
    return "".join(out)


@fn("Characters", area=AREA, status="verified", notes="list of single-character strings"
                                                      ". Check: exact (exact input gives exact results), reference "
                                                      "(differential cases in tests/library).")
def _chars(s):
    return L(*list(_s(s, "Characters")))


@fn("ToCharacterCode", area=AREA, status="fixed",
    notes="list of all character codes (legacy returned only the first code); lists of strings thread"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _tocc(s):
    if is_list(s):
        return L(*(_tocc(x) for x in s.args))
    return L(*(ord(c) for c in _s(s, "ToCharacterCode")))


@fn("FromCharacterCode", area=AREA, status="fixed",
    notes="FromCharacterCode[n] and FromCharacterCode[{n1, n2, ...}] (legacy accepted only one code and "
          "truncated reals)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fromcc(n):
    if is_list(n):
        codes = [nonneg_int(v, "FromCharacterCode", "code") for v in n.args]
    else:
        codes = [nonneg_int(n, "FromCharacterCode", "code")]
    if any(c > 0x10FFFF for c in codes):
        raise ValueError("FromCharacterCode: code out of range")
    return "".join(chr(c) for c in codes)


@fn("AlphabetPosition", area=AREA, status="fixed",
    notes="like Wolfram LetterNumber: position in the English alphabet, 0 for non-letters; multi-character "
          "strings give a list (legacy returned 0)"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _alphapos(s):
    st = _s(s, "AlphabetPosition")
    vals = [ord(c.lower()) - 96 if c.isascii() and c.isalpha() else 0 for c in st]
    return vals[0] if len(vals) == 1 else L(*vals)


@fn("Base64Encode", area=AREA, status="verified", notes="Base64 of the UTF-8 bytes (MikoshiLang)"
                                                        ". Check: exact (exact input gives exact results), reference "
                                                        "(differential cases in tests/library).")
def _b64e(s):
    return base64.b64encode(_s(s, "Base64Encode").encode("utf-8")).decode("ascii")


@fn("Base64Decode", area=AREA, status="fixed", notes="invalid Base64 or non-UTF-8 data now raises"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _b64d(s):
    try:
        return base64.b64decode(_s(s, "Base64Decode").encode("ascii"), validate=True).decode("utf-8")
    except (ValueError, UnicodeError):
        raise ValueError("Base64Decode: invalid Base64 input")


@fn("HexEncode", area=AREA, status="verified", notes="hex of the UTF-8 bytes (MikoshiLang)"
                                                     ". Check: exact (exact input gives exact results), reference "
                                                     "(differential cases in tests/library).")
def _hexe(s):
    return _s(s, "HexEncode").encode("utf-8").hex()


@fn("HexDecode", area=AREA, status="fixed", notes="invalid hex now raises (legacy returned an Error expression)"
                                                  ". Check: exact (exact input gives exact results), reference "
                                                  "(differential cases in tests/library).")
def _hexd(s):
    try:
        return bytes.fromhex(_s(s, "HexDecode")).decode("utf-8")
    except (ValueError, UnicodeError):
        raise ValueError("HexDecode: invalid hex input")


def _hash(name, algo):
    @fn(name, area=AREA, status="verified", notes=f"hex digest of the UTF-8 bytes (= Hash[s, \"{name}\", \"HexString\"])")
    def impl(s):
        return hashlib.new(algo, _s(s, name).encode("utf-8")).hexdigest()
    return impl


for _n, _a in [("MD5", "md5"), ("SHA1", "sha1"), ("SHA256", "sha256"), ("SHA512", "sha512")]:
    _hash(_n, _a)


def _rot(st, k):
    out = []
    for c in st:
        if "a" <= c <= "z":
            out.append(chr((ord(c) - 97 + k) % 26 + 97))
        elif "A" <= c <= "Z":
            out.append(chr((ord(c) - 65 + k) % 26 + 65))
        else:
            out.append(c)
    return "".join(out)


@fn("ROT13", area=AREA, status="verified", notes="ROT13 on ASCII letters"
                                                 ". Check: exact (exact input gives exact results), reference "
                                                 "(differential cases in tests/library).")
def _rot13(s):
    return _rot(_s(s, "ROT13"), 13)


@fn("CaesarCipher", area=AREA, status="verified", notes="shift ASCII letters by k (k may be negative); integer k only"
                                                        ". Check: exact (exact input gives exact results), reference "
                                                        "(differential cases in tests/library).")
def _caesar(s, k):
    return _rot(_s(s, "CaesarCipher"), int_arg(k, "CaesarCipher", "shift"))


@fn("XORCipher", area=AREA, status="fixed", notes="hex string of UTF-8 bytes XOR the repeated key; empty key raises"
                                                  ". Check: exact (exact input gives exact results), reference "
                                                  "(differential cases in tests/library).")
def _xor(s, key):
    b, k = _s(s, "XORCipher").encode("utf-8"), _s(key, "XORCipher", "key").encode("utf-8")
    if not k:
        raise ValueError("XORCipher: key must be non-empty")
    return bytes(x ^ k[i % len(k)] for i, x in enumerate(b)).hex()


def _date(y, m, d, fname):
    try:
        return _dt.date(int_arg(y, fname, "year"), int_arg(m, fname, "month"), int_arg(d, fname, "day"))
    except ValueError as exc:
        raise ValueError(f"{fname}: invalid date ({exc})")


@fn("DayOfWeek", area=AREA, status="fixed",
    notes="day of the week of a Gregorian date as a symbol (Friday), DayOfWeek[y, m, d] or DayOfWeek[{y, m, d}] "
          "(Wolfram DayName); legacy returned a string and truncated reals"
          ". Check: reference (differential cases in tests/library).")
def _dow(y, m=None, d=None):
    if m is None:
        parts = list_arg(y, "DayOfWeek", "date {y, m, d}")
        if len(parts) < 3:
            raise ValueError("DayOfWeek: date must be {y, m, d}")
        y, m, d = parts[:3]
    elif d is None:
        raise ValueError("DayOfWeek takes {y, m, d} or y, m, d")
    return Symbol(_date(y, m, d, "DayOfWeek").strftime("%A"))


@fn("DaysInMonth", area=AREA, status="fixed", notes="days in month m of year y; invalid months raise"
                                                    ". Check: exact (exact input gives exact results), reference "
                                                    "(differential cases in tests/library).")
def _dim(y, m):
    yy, mm = int_arg(y, "DaysInMonth", "year"), int_arg(m, "DaysInMonth", "month")
    if not 1 <= mm <= 12 or not 1 <= yy <= 9999:
        raise ValueError("DaysInMonth: invalid year or month")
    return calendar.monthrange(yy, mm)[1]


@fn("LeapYearQ", area=AREA, status="fixed", notes="Gregorian leap year test; LeapYearQ[{y, ...}] accepted"
                                                  ". Check: reference (differential cases in tests/library).")
def _leap(y):
    if is_list(y):
        y = list_arg(y, "LeapYearQ", min_len=1)[0]
    yy = int_arg(y, "LeapYearQ", "year")
    return yy % 4 == 0 and (yy % 100 != 0 or yy % 400 == 0)


def _now():
    return _dt.datetime.now()


def _datelist(d, fname):
    parts = [int_arg(v, fname, "date element") for v in list_arg(d, fname, min_len=1)]
    parts += [1, 1, 0, 0, 0][len(parts) - 1:] if len(parts) < 6 else []
    try:
        return _dt.datetime(*parts[:6])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fname}: invalid date list ({exc})")


@fn("DateString", area=AREA, status="fixed",
    notes="DateString[] (now) and DateString[{y, m, d, h, min, s}] in Wolfram's default format "
          "\"Fri 30 Oct 2026 14:05:09\"; legacy registered only a 1-argument form that ignored its argument")
def _datestring(d=None):
    t = _now() if d is None else _datelist(d, "DateString")
    return f"{t.strftime('%a')} {t.day} {t.strftime('%b %Y %H:%M:%S')}"


@fn("TimeString", area=AREA, status="fixed",
    notes="TimeString[] -> \"HH:MM:SS\" (MikoshiLang); legacy only had a 1-argument form that ignored its argument"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _timestring(d=None):
    t = _now() if d is None else _datelist(d, "TimeString")
    return t.strftime("%H:%M:%S")


@fn("DateTimeString", area=AREA, status="fixed",
    notes="ISO 8601 \"YYYY-MM-DDTHH:MM:SS\" for now or a date list (MikoshiLang); legacy ignored its argument"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _dtstring(d=None):
    t = _now() if d is None else _datelist(d, "DateTimeString")
    return t.strftime("%Y-%m-%dT%H:%M:%S")


@fn("UnixTime", area=AREA, status="fixed",
    notes="UnixTime[] = whole seconds since 1970-01-01 00:00 UTC; UnixTime[{y, m, d, ...}] interprets the date "
          "in UTC; legacy only had a 1-argument form that ignored its argument"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _unixtime(d=None):
    if d is None:
        return int(_dt.datetime.now(_dt.timezone.utc).timestamp())
    t = _datelist(d, "UnixTime").replace(tzinfo=_dt.timezone.utc)
    return int(t.timestamp())

# Check: sentences (CONVENTIONS §4) for heads registered in a loop or by a factory, whose notes are
# built in code. Each names only methods that pass in scripts/verify_library.py.
from .core import add_check as _add_check  # noqa: E402

_add_check("exact (exact input gives exact results), reference (differential cases in tests/library).",
          "MD5", "SHA1", "SHA256", "SHA512")
