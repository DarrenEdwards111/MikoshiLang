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
          "silently accepted negative/out-of-range n)")
def _stake(s, n):
    st = _s(s, "StringTake")
    i, j = _span(n, len(st), "StringTake")
    return st[i:j]


@fn("StringDrop", area=AREA, status="fixed", notes="StringDrop[s, n], [s, -n], [s, {n}], [s, {m, n}]")
def _sdrop(s, n):
    st = _s(s, "StringDrop")
    i, j = _span(n, len(st), "StringDrop")
    return st[:i] + st[j:]


@fn("StringPart", area=AREA, status="fixed",
    notes="1-based (legacy was 0-based: StringPart[\"abc\", 1] gave \"b\"); negative positions count from the end; "
          "a list of positions gives a list")
def _spart(s, n):
    st = _s(s, "StringPart")
    if is_list(n):
        return L(*(_spart(s, v) for v in n.args))
    i, j = _span(L(n), len(st), "StringPart")
    return st[i:j]


@fn("StringInsert", area=AREA, status="fixed",
    notes="Wolfram argument order StringInsert[s, new, n]: new starts at position n (negative n counts from the "
          "end, -1 appends); legacy was StringInsert[s, i, new] with 0-based i")
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


def _patterns(p, fname):
    if isinstance(p, str):
        return [p]
    if is_list(p) and all(isinstance(x, str) for x in p.args):
        return list(p.args)
    raise NotImplementedError(f"{fname}: only literal string patterns (or lists of them) are supported")


@fn("StringDelete", area=AREA, status="fixed",
    notes="Wolfram StringDelete[s, \"sub\"] / [s, {\"a\", \"b\"}] deletes every occurrence; the legacy slice form "
          "StringDelete[s, i, j] now raises (use StringDrop[s, {i, j}])")
def _sdelete(s, p, *rest):
    if rest:
        raise ValueError("StringDelete[s, i, j] is not Wolfram syntax; use StringDrop[s, {i, j}]")
    st = _s(s, "StringDelete")
    return _replace_all(st, [(x, "") for x in _patterns(p, "StringDelete")], None)


def _replace_all(st, rules, limit):
    if any(a == "" for a, _ in rules):
        raise ValueError("cannot replace the empty string")
    out, i, n = [], 0, 0
    while i < len(st):
        if limit is None or n < limit:
            for a, b in rules:
                if st.startswith(a, i):
                    out.append(b)
                    i += len(a)
                    n += 1
                    break
            else:
                out.append(st[i])
                i += 1
        else:
            out.append(st[i:])
            break
    return "".join(out)


def _str_rules(r, fname):
    rs = list(r.args) if is_list(r) else [r]
    out = []
    for x in rs:
        if isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2 \
                and isinstance(x.args[0], str) and isinstance(x.args[1], str):
            out.append((x.args[0], x.args[1]))
        else:
            raise NotImplementedError(f"{fname}: only literal string rules \"a\" -> \"b\" are supported")
    return out


@fn("StringReplace", area=AREA, status="fixed",
    notes="Wolfram StringReplace[s, \"a\" -> \"b\"], [s, {rules}], [s, rules, n] (left-to-right, rules tried in "
          "order); the legacy form StringReplace[s, old, new] now raises")
def _sreplace(s, r, n=None):
    st = _s(s, "StringReplace")
    if isinstance(r, str) and isinstance(n, str):
        raise ValueError("StringReplace[s, old, new] is not Wolfram syntax; use StringReplace[s, old -> new]")
    lim = None if n is None else nonneg_int(n, "StringReplace", "count")
    return _replace_all(st, _str_rules(r, "StringReplace"), lim)


def _pad_char(c, fname):
    ch = _s(c, fname, "padding")
    if len(ch) != 1:
        raise NotImplementedError(f"{fname}: only single-character padding is supported")
    return ch


@fn("StringPadLeft", area=AREA, status="fixed",
    notes="pads on the left or truncates to the rightmost n characters, as Wolfram (legacy never truncated); "
          "single-character padding")
def _padl(s, n, c=" "):
    st, k, ch = _s(s, "StringPadLeft"), nonneg_int(n, "StringPadLeft", "length"), _pad_char(c, "StringPadLeft")
    return st[len(st) - k:] if len(st) >= k else ch * (k - len(st)) + st


@fn("StringPadRight", area=AREA, status="fixed",
    notes="pads on the right or truncates to the leftmost n characters, as Wolfram; single-character padding")
def _padr(s, n, c=" "):
    st, k, ch = _s(s, "StringPadRight"), nonneg_int(n, "StringPadRight", "length"), _pad_char(c, "StringPadRight")
    return st[:k] if len(st) >= k else st + ch * (k - len(st))


@fn("StringCenter", area=AREA, status="fixed",
    notes="centres s in width n (extra padding on the right, MikoshiLang); single-character fill, width checked")
def _center(s, n, c=" "):
    st, k, ch = _s(s, "StringCenter"), nonneg_int(n, "StringCenter", "width"), _pad_char(c, "StringCenter")
    if len(st) >= k:
        return st
    total = k - len(st)
    left = total // 2
    return ch * left + st + ch * (total - left)


@fn("StringRepeat", area=AREA, status="fixed", notes="StringRepeat[s, n] for n >= 1; legacy truncated non-integers")
def _srepeat(s, n):
    return _s(s, "StringRepeat") * pos_int(n, "StringRepeat", "count")


@fn("StringReverse", area=AREA, status="verified", notes="reverses the characters")
def _srev(s):
    return _s(s, "StringReverse")[::-1]


@fn("StringToUpperCase", area=AREA, status="verified", notes="alias of ToUpperCase")
def _upper(s):
    return _s(s, "StringToUpperCase").upper()


@fn("StringToLowerCase", area=AREA, status="verified", notes="alias of ToLowerCase")
def _lower(s):
    return _s(s, "StringToLowerCase").lower()


@fn("StringCapitalize", area=AREA, status="fixed",
    notes="upper-cases the first character only (Wolfram Capitalize); legacy also lower-cased the rest "
          "(\"iPhone\" -> \"Iphone\")")
def _capitalize(s):
    st = _s(s, "StringCapitalize")
    return st[:1].upper() + st[1:]


@fn("StringToTitleCase", area=AREA, status="fixed",
    notes="capitalises the first letter of each whitespace-separated word and leaves the rest unchanged; legacy "
          "str.title() mangled words (\"don't\" -> \"Don'T\", \"NASA\" -> \"Nasa\")")
def _title(s):
    st = _s(s, "StringToTitleCase")
    out, prev_space = [], True
    for ch in st:
        out.append(ch.upper() if prev_space else ch)
        prev_space = ch.isspace()
    return "".join(out)


@fn("StringTrim", area=AREA, status="verified", notes="removes leading and trailing whitespace")
def _strim(s):
    return _s(s, "StringTrim").strip()


@fn("StringCount", area=AREA, status="fixed",
    notes="non-overlapping occurrences of a literal substring; the empty substring now raises")
def _scount(s, sub):
    st, p = _s(s, "StringCount"), _s(sub, "StringCount", "substring")
    if p == "":
        raise ValueError("StringCount: substring must be non-empty")
    return st.count(p)


@fn("StringContainsQ", area=AREA, status="verified", notes="literal substring test")
def _scontains(s, sub):
    return _s(sub, "StringContainsQ", "substring") in _s(s, "StringContainsQ")


@fn("StringStartsQ", area=AREA, status="verified", notes="literal prefix test")
def _sstarts(s, p):
    return _s(s, "StringStartsQ").startswith(_s(p, "StringStartsQ", "prefix"))


@fn("StringEndsQ", area=AREA, status="verified", notes="literal suffix test")
def _sends(s, p):
    return _s(s, "StringEndsQ").endswith(_s(p, "StringEndsQ", "suffix"))


@fn("Characters", area=AREA, status="verified", notes="list of single-character strings")
def _chars(s):
    return L(*list(_s(s, "Characters")))


@fn("ToCharacterCode", area=AREA, status="fixed",
    notes="list of all character codes (legacy returned only the first code); lists of strings thread")
def _tocc(s):
    if is_list(s):
        return L(*(_tocc(x) for x in s.args))
    return L(*(ord(c) for c in _s(s, "ToCharacterCode")))


@fn("FromCharacterCode", area=AREA, status="fixed",
    notes="FromCharacterCode[n] and FromCharacterCode[{n1, n2, ...}] (legacy accepted only one code and "
          "truncated reals)")
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
          "strings give a list (legacy returned 0)")
def _alphapos(s):
    st = _s(s, "AlphabetPosition")
    vals = [ord(c.lower()) - 96 if c.isascii() and c.isalpha() else 0 for c in st]
    return vals[0] if len(vals) == 1 else L(*vals)


@fn("Base64Encode", area=AREA, status="verified", notes="Base64 of the UTF-8 bytes (MikoshiLang)")
def _b64e(s):
    return base64.b64encode(_s(s, "Base64Encode").encode("utf-8")).decode("ascii")


@fn("Base64Decode", area=AREA, status="fixed", notes="invalid Base64 or non-UTF-8 data now raises")
def _b64d(s):
    try:
        return base64.b64decode(_s(s, "Base64Decode").encode("ascii"), validate=True).decode("utf-8")
    except (ValueError, UnicodeError):
        raise ValueError("Base64Decode: invalid Base64 input")


@fn("HexEncode", area=AREA, status="verified", notes="hex of the UTF-8 bytes (MikoshiLang)")
def _hexe(s):
    return _s(s, "HexEncode").encode("utf-8").hex()


@fn("HexDecode", area=AREA, status="fixed", notes="invalid hex now raises (legacy returned an Error expression)")
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


@fn("ROT13", area=AREA, status="verified", notes="ROT13 on ASCII letters")
def _rot13(s):
    return _rot(_s(s, "ROT13"), 13)


@fn("CaesarCipher", area=AREA, status="verified", notes="shift ASCII letters by k (k may be negative); integer k only")
def _caesar(s, k):
    return _rot(_s(s, "CaesarCipher"), int_arg(k, "CaesarCipher", "shift"))


@fn("XORCipher", area=AREA, status="fixed", notes="hex string of UTF-8 bytes XOR the repeated key; empty key raises")
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
          "(Wolfram DayName); legacy returned a string and truncated reals")
def _dow(y, m=None, d=None):
    if m is None:
        parts = list_arg(y, "DayOfWeek", "date {y, m, d}")
        if len(parts) < 3:
            raise ValueError("DayOfWeek: date must be {y, m, d}")
        y, m, d = parts[:3]
    elif d is None:
        raise ValueError("DayOfWeek takes {y, m, d} or y, m, d")
    return Symbol(_date(y, m, d, "DayOfWeek").strftime("%A"))


@fn("DaysInMonth", area=AREA, status="fixed", notes="days in month m of year y; invalid months raise")
def _dim(y, m):
    yy, mm = int_arg(y, "DaysInMonth", "year"), int_arg(m, "DaysInMonth", "month")
    if not 1 <= mm <= 12 or not 1 <= yy <= 9999:
        raise ValueError("DaysInMonth: invalid year or month")
    return calendar.monthrange(yy, mm)[1]


@fn("LeapYearQ", area=AREA, status="fixed", notes="Gregorian leap year test; LeapYearQ[{y, ...}] accepted")
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
    notes="TimeString[] -> \"HH:MM:SS\" (MikoshiLang); legacy only had a 1-argument form that ignored its argument")
def _timestring(d=None):
    t = _now() if d is None else _datelist(d, "TimeString")
    return t.strftime("%H:%M:%S")


@fn("DateTimeString", area=AREA, status="fixed",
    notes="ISO 8601 \"YYYY-MM-DDTHH:MM:SS\" for now or a date list (MikoshiLang); legacy ignored its argument")
def _dtstring(d=None):
    t = _now() if d is None else _datelist(d, "DateTimeString")
    return t.strftime("%Y-%m-%dT%H:%M:%S")


@fn("UnixTime", area=AREA, status="fixed",
    notes="UnixTime[] = whole seconds since 1970-01-01 00:00 UTC; UnixTime[{y, m, d, ...}] interprets the date "
          "in UTC; legacy only had a 1-argument form that ignored its argument")
def _unixtime(d=None):
    if d is None:
        return int(_dt.datetime.now(_dt.timezone.utc).timestamp())
    t = _datelist(d, "UnixTime").replace(tzinfo=_dt.timezone.utc)
    return int(t.timestamp())
