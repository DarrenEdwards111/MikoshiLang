"""Wolfram batch W12: dates and times, Dataset/Query, Tabular data and relational databases.

Dates and times
    ``DateObject[list, gran]`` (dates of granularity "Day" and coarser without an explicit time zone are *floating*,
    exactly Wolfram's two-argument canonical form) and ``DateObject[list, gran, cal, tz]`` (finer granularities, or
    an explicit ``TimeZone``), with an optional fifth time-system argument.  The proleptic Gregorian calendar has no
    year 0 (year -1 is 1 BCE); the Julian calendar is supported through ``CalendarType -> "Julian"``.  Absolute times
    count seconds from 1900-01-01 00:00 in ``$TimeZone`` (default: the system time zone).  Exact inputs give exact
    results.  Time-zone names ("Europe/London") use the IANA database through :mod:`zoneinfo`.

Dataset and Query
    ``Dataset[data]`` wraps lists/associations; ``Query`` implements Wolfram's descending/ascending operator
    semantics with the ``FailureAction``, ``MissingBehavior`` and ``PartBehavior`` options.

Tabular
    ``Tabular[rows, TabularSchema[<|...|>]]`` is the canonical form: rows are plain lists and the schema records the
    column keys, element types and key columns.  ``TabularColumn[{...}, type]`` and ``TabularRow[...]`` are its
    columns and rows.

Relational databases
    ``DatabaseReference``/``RelationalDatabase``/``DatabaseConnect`` use SQLite through :mod:`sqlite3` (the only
    backend available without external servers); other backends raise a clear error.
"""
from __future__ import annotations

import calendar as _calendar
import datetime as _dt
import math
import random as _random
import re
import time
from fractions import Fraction

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, record, L, R, REGISTRY

AREA = "dates, tabular data & databases (W12)"
ADDED = dict(area=AREA, status="added")

NONE = Symbol("None")
AUTOMATIC = Symbol("Automatic")
ALL = Symbol("All")


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def is_sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def is_rule(x):
    return KL.is_rule(x)


def is_assoc(x):
    return KL.is_assoc(x)


def is_list(x):
    return K.is_list(x)


def assoc(*pairs):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in pairs))


def _added(name, notes):
    record(name, AREA, "added", notes)


def _opts(args, names, fname):
    """Split trailing Name -> value options; unknown option names raise."""
    args = list(args)
    opts = {}
    while args and is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol) and \
            args[-1].args[0].name[:1].isupper() and not isinstance(args[-1].args[0], str):
        r = args[-1]
        nm = r.args[0].name
        if nm not in names:
            if nm in _ALL_OPTION_NAMES:
                raise ValueError(f"{fname}: {nm} is not a known option for {fname}")
            break
        args.pop()
        opts[nm] = r.args[1]
    return args, opts


_ALL_OPTION_NAMES = {"CalendarType", "DateFormat", "DateGranularity", "TimeSystem", "TimeZone", "DayCountConvention",
                     "Method", "FailureAction", "MissingBehavior", "PartBehavior", "IncludeRelatedTables",
                     "DatasetTheme", "TimeFormat", "DateFunction", "DateReduction", "AllowedDimensions",
                     "MaxItems", "KeyCollisionFunction", "ExpirationDate", "DateTicksFormat"}


# ─────────────────────────── numbers ───────────────────────────

def _num(x, fname, what="date element"):
    """int | Fraction | float for a real number argument; symbolic -> Unevaluated."""
    if isinstance(x, bool):
        raise ValueError(f"{fname}: {what} {x!r} must be a real number")
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f"{fname}: {what} must be finite")
        return x
    try:
        v = K.to_sp(x)
    except Exception:
        v = None
    if isinstance(v, sp.Integer):
        return int(v)
    if isinstance(v, sp.Rational):
        return Fraction(int(v.p), int(v.q))
    if isinstance(v, sp.Float):
        return float(v)
    if isinstance(v, sp.Basic) and v.is_number and v.is_real:
        return float(v)
    if isinstance(x, (Symbol, Expr)) and not isinstance(x, str) and isinstance(v, sp.Basic) and v.free_symbols:
        raise Unevaluated()
    raise ValueError(f"{fname}: {what} {x!r} must be a real number")


def _out(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, Fraction):
        return v.numerator if v.denominator == 1 else R(sp.Rational(v.numerator, v.denominator))
    if isinstance(v, float):
        return v
    return R(v)


def _is_exact(v):
    return isinstance(v, (int, Fraction))


def _floor(v):
    return math.floor(v)


# ─────────────────────────── calendar arithmetic ───────────────────────────

_EPOCH_JDN = 2415021          # JDN of 1900-01-01 (Gregorian)


def _astro(y):
    """Wolfram year (no year 0; -1 is 1 BCE) -> astronomical year."""
    return y + 1 if y < 0 else y


def _wl_year(ya):
    return ya - 1 if ya <= 0 else ya


def _jdn(ya, m, d, cal="Gregorian"):
    a = (14 - m) // 12
    yy = ya + 4800 - a
    mm = m + 12 * a - 3
    if cal == "Julian":
        return d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - 32083
    return d + (153 * mm + 2) // 5 + 365 * yy + yy // 4 - yy // 100 + yy // 400 - 32045


def _from_jdn(j, cal="Gregorian"):
    if cal == "Julian":
        b, c = 0, j + 32082
    else:
        a = j + 32044
        b = (4 * a + 3) // 146097
        c = a - 146097 * b // 4
    d = (4 * c + 3) // 1461
    e = c - 1461 * d // 4
    m = (5 * e + 2) // 153
    day = e - (153 * m + 2) // 5 + 1
    month = m + 3 - 12 * (m // 10)
    year = 100 * b + d - 4800 + m // 10
    return year, month, day


def _days_in_month(y, m, cal="Gregorian"):
    ya = _astro(y)
    nm = (ya + 1, 1) if m == 12 else (ya, m + 1)
    return _jdn(nm[0], nm[1], 1, cal) - _jdn(ya, m, 1, cal)


def _leap(y, cal="Gregorian"):
    return _days_in_month(y, 2, cal) == 29


def _norm_ym(y, m):
    """(astronomical year, month 1..12) with month overflow carried into the year."""
    ya = _astro(y)
    m0 = m - 1
    return ya + m0 // 12, m0 % 12 + 1


def _list_to_abs(dl, cal="Gregorian", fname="DateObject"):
    """Wall-clock seconds since 1900-01-01 00:00 for a (possibly unnormalised) date list."""
    if not 1 <= len(dl) <= 6:
        raise ValueError(f"{fname}: a date list has 1 to 6 elements {{y, m, d, h, m, s}}")
    vals = list(dl) + [1, 1, 0, 0, 0][len(dl) - 1:] if len(dl) < 6 else list(dl)
    y, m, d, h, mi, s = vals
    if not isinstance(y, int):
        raise ValueError(f"{fname}: the year must be an integer")
    if y == 0:
        raise ValueError(f"{fname}: there is no year 0 (year -1 is 1 BCE)")
    if not isinstance(m, int):
        raise ValueError(f"{fname}: the month must be an integer")
    ya, mm = _norm_ym(y, m)
    days = _jdn(ya, mm, 1, cal) - _EPOCH_JDN
    return (days + (d - 1)) * 86400 + h * 3600 + mi * 60 + s


def _abs_to_list(t, cal="Gregorian"):
    """[y, m, d, h, mi, s] (Wolfram years) for wall-clock seconds since 1900."""
    days = _floor(t / 86400) if not isinstance(t, int) else t // 86400
    rem = t - days * 86400
    if isinstance(rem, float) and rem >= 86400:
        days, rem = days + 1, rem - 86400
    ya, m, d = _from_jdn(days + _EPOCH_JDN, cal)
    h = int(_floor(rem / 3600)) if not isinstance(rem, int) else rem // 3600
    rem2 = rem - h * 3600
    mi = int(_floor(rem2 / 60)) if not isinstance(rem2, int) else rem2 // 60
    s = rem2 - mi * 60
    if isinstance(s, float) and s < 0:
        s = 0.0
    return [_wl_year(ya), m, d, h, mi, s]


def _weekday(t):
    """0 = Monday ... 6 = Sunday for wall-clock seconds t."""
    days = _floor(t / 86400) if not isinstance(t, int) else t // 86400
    return (days + _EPOCH_JDN) % 7


# ─────────────────────────── granularities ───────────────────────────

_GRAN_LEN = {"Eternity": 0, "Millennium": 1, "MillenniumBeginning01": 1, "Century": 1, "CenturyBeginning01": 1,
             "Decade": 1, "Year": 1, "Quarter": 2, "Month": 2, "Week": 3, "WeekBeginningSunday": 3, "Day": 3,
             "Hour": 4, "Minute": 5, "Second": 6, "Decisecond": 6, "Centisecond": 6, "Millisecond": 6,
             "Microsecond": 6, "Nanosecond": 6, "Instant": 6}
_SUBSEC = {"Second": 0, "Decisecond": 1, "Centisecond": 2, "Millisecond": 3, "Microsecond": 6, "Nanosecond": 9}
_SUBSEC_NAME = {v: k for k, v in _SUBSEC.items()}
_BY_LEN = {1: "Year", 2: "Month", 3: "Day", 4: "Hour", 5: "Minute", 6: "Instant"}
_FLOATING = {"Millennium", "MillenniumBeginning01", "Century", "CenturyBeginning01", "Decade", "Year", "Quarter",
             "Month", "Week", "WeekBeginningSunday", "Day"}
# finest first, for "the finest granularity of the dates"
_FINENESS = ["Instant", "Nanosecond", "Microsecond", "Millisecond", "Centisecond", "Decisecond", "Second", "Minute",
             "Hour", "Day", "WeekBeginningSunday", "Week", "Month", "Quarter", "Year", "Decade", "CenturyBeginning01",
             "Century", "MillenniumBeginning01", "Millennium"]


def _gran(g, fname):
    """Canonical granularity: a name or an integer 0..9 (10^-n seconds; named when Wolfram has a name)."""
    if isinstance(g, str):
        if g in _GRAN_LEN and g != "Eternity":
            return g
        raise ValueError(f"{fname}: {g!r} is not a valid calendar granularity")
    if K.is_int(g) and not isinstance(g, bool):
        n = int(g)
        if 0 <= n <= 9:
            return _SUBSEC_NAME.get(n, n)
        raise ValueError(f"{fname}: the precision n in DateObject[date, n] must be an integer from 0 to 9")
    if isinstance(g, (Symbol, Expr)) and not isinstance(g, str) and not K.is_number(g):
        raise Unevaluated()
    raise ValueError(f"{fname}: {g!r} is not a valid calendar granularity")


def _glen(g):
    return 6 if isinstance(g, int) else _GRAN_LEN[g]


def _digits(g):
    return g if isinstance(g, int) else _SUBSEC.get(g)


def _canon_start(t, g, cal):
    """(canonical date list, period start, period end) for wall-clock seconds t and granularity g."""
    dl = _abs_to_list(t, cal)
    y = dl[0]
    if g in ("Year", "Decade", "Century", "CenturyBeginning01", "Millennium", "MillenniumBeginning01"):
        if g == "Decade":
            y = y - y % 10
        elif g == "Century":
            y = y - y % 100
        elif g == "Millennium":
            y = y - y % 1000
        elif g == "CenturyBeginning01":
            y = (y - 1) - (y - 1) % 100 + 1
        elif g == "MillenniumBeginning01":
            y = (y - 1) - (y - 1) % 1000 + 1
        if y == 0:
            y = -1 if dl[0] < 0 else 1
        span = {"Year": 1, "Decade": 10, "Century": 100, "CenturyBeginning01": 100, "Millennium": 1000,
                "MillenniumBeginning01": 1000}[g]
        ya = _astro(y)
        start = _list_to_abs([y], cal)
        end = _list_to_abs([_wl_year(ya + span)], cal)
        return [y], start, end
    if g in ("Quarter", "Month"):
        m = dl[1]
        if g == "Quarter":
            m = 3 * ((m - 1) // 3) + 1
        step = 3 if g == "Quarter" else 1
        start = _list_to_abs([y, m], cal)
        end = _list_to_abs([y, m + step], cal)
        return [y, m], start, end
    day0 = _list_to_abs(dl[:3], cal)
    if g in ("Week", "WeekBeginningSunday"):
        wd = _weekday(day0)
        back = wd if g == "Week" else (wd + 1) % 7
        start = day0 - back * 86400
        return _abs_to_list(start, cal)[:3], start, start + 7 * 86400
    if g == "Day":
        return dl[:3], day0, day0 + 86400
    if g == "Hour":
        start = day0 + dl[3] * 3600
        return dl[:4], start, start + 3600
    if g == "Minute":
        start = day0 + dl[3] * 3600 + dl[4] * 60
        return dl[:5], start, start + 60
    if g == "Instant":
        return dl, t, t
    n = _digits(g)
    s = dl[5]
    if n == 0:
        si = int(_floor(s))
        start = t - (s - si)
        return dl[:5] + [si], start, start + 1
    q = Fraction(1, 10 ** n)
    sf = Fraction(s) if not isinstance(s, float) else Fraction(s).limit_denominator(10 ** (n + 3))
    k = math.floor(sf / q)
    sv = float(k * q)
    start = t - (s - sv) if isinstance(t, float) else t - (sf - k * q)
    return dl[:5] + [sv], start, start + (float(q) if isinstance(start, float) else q)


# ─────────────────────────── time zones ───────────────────────────

def _system_tz():
    return time.localtime().tm_gmtoff / 3600.0


def current_tz():
    v = ev(Symbol("$TimeZone"))
    return _tz(v, "$TimeZone", allow_none=False)


def _tz(z, fname, allow_none=True):
    """Normalised time zone: float hours, an IANA name, or None (floating)."""
    if is_sym(z, "None"):
        if allow_none:
            return None
        raise ValueError(f"{fname}: a time zone is required")
    if is_sym(z, "Automatic"):
        return current_tz()
    if isinstance(z, str):
        if z in ("UTC", "GMT", "Zulu", "Z"):
            return 0.0
        try:
            from zoneinfo import ZoneInfo
            ZoneInfo(z)
        except Exception:
            raise ValueError(f"{fname}: {z!r} is not a known time zone")
        return z
    v = _num(z, fname, "time zone")
    if not -24 <= v <= 24:
        raise ValueError(f"{fname}: time zone offset {z!r} must lie between -24 and 24 hours")
    return float(v)


def _tz_offset(z, wall=None, utc=None):
    """Offset in hours of zone z at a wall-clock time (or at a UTC time), both as seconds since 1900."""
    if z is None:
        z = current_tz()
    if isinstance(z, float):
        return z
    from zoneinfo import ZoneInfo
    zi = ZoneInfo(z)
    base = _dt.datetime(1900, 1, 1)
    if utc is not None:
        secs = float(utc)
        secs = min(max(secs, -1.5e11), 2.5e11)
        try:
            u = (base + _dt.timedelta(seconds=secs)).replace(tzinfo=_dt.timezone.utc)
            return u.astimezone(zi).utcoffset().total_seconds() / 3600.0
        except (OverflowError, ValueError):
            return 0.0
    try:
        w = base + _dt.timedelta(seconds=float(wall))
        return w.replace(tzinfo=zi).utcoffset().total_seconds() / 3600.0
    except (OverflowError, ValueError):
        return 0.0


def _tz_out(z):
    return NONE if z is None else z


def _off_secs(h):
    """Offset in hours -> seconds, exact when it is a whole number of minutes."""
    f = Fraction(h).limit_denominator(3600)
    s = f * 3600
    return int(s) if s.denominator == 1 else float(h) * 3600


# ─────────────────────────── the date record ───────────────────────────

class D:
    """A parsed DateObject: canonical list, granularity, calendar, zone (None = floating), time system, format."""
    __slots__ = ("dl", "gran", "cal", "tz", "tsys", "fmt", "tz_explicit")

    def __init__(self, dl, gran, cal="Gregorian", tz=None, tsys=None, fmt=None, tz_explicit=False):
        self.dl, self.gran, self.cal, self.tz, self.tsys, self.fmt = dl, gran, cal, tz, tsys, fmt
        self.tz_explicit = tz_explicit

    # wall-clock seconds of the period start/end in the date's own zone
    def wall(self):
        return _list_to_abs(self.dl, self.cal)

    def bounds_wall(self):
        _, s, e = _canon_start(self.wall(), self.gran, self.cal)
        return s, e

    def zone(self):
        return self.tz if self.tz is not None else current_tz()

    def to_utc(self, wall):
        return wall - _off_secs(_tz_offset(self.zone(), wall=wall))

    def bounds_utc(self):
        s, e = self.bounds_wall()
        return self.to_utc(s), self.to_utc(e)

    def expr(self):
        out = [L(*(_out(v) for v in self.dl)), self.gran if isinstance(self.gran, str) else self.gran]
        floating = self.gran in _FLOATING and self.tz is None and not self.tz_explicit
        if not floating or self.cal != "Gregorian" or self.tsys is not None:
            out += [self.cal, _tz_out(self.tz)]
        if self.tsys is not None:
            out.append(self.tsys)
        if self.fmt is not None:
            out.append(Expr("Rule", Symbol("DateFormat"), self.fmt))
        return Expr("DateObject", *out)


def _from_wall(t, gran, cal="Gregorian", tz=None, tsys=None, fmt=None, tz_explicit=False):
    dl, _, _ = _canon_start(t, gran, cal)
    return D(dl, gran, cal, tz, tsys, fmt, tz_explicit)


def _nums(lst, fname):
    return [_num(v, fname) for v in lst]


def parse_dobj(e):
    """D for a canonical DateObject expression, else None."""
    if not (isinstance(e, Expr) and e.head == "DateObject" and 2 <= len(e.args) <= 6):
        return None
    args = list(e.args)
    fmt = None
    if is_rule(args[-1]) and is_sym(args[-1].args[0], "DateFormat"):
        fmt = args.pop().args[1]
    if len(args) not in (2, 4, 5) or not is_list(args[0]):
        return None
    try:
        g = _gran(args[1], "DateObject")
        dl = _nums(args[0].args, "DateObject")
    except (ValueError, Unevaluated):
        return None
    if len(dl) != _glen(g) or not all(isinstance(v, int) for v in dl[:5]):
        return None
    cal, tz, tsys, explicit = "Gregorian", None, None, False
    if len(args) >= 4:
        if args[2] not in ("Gregorian", "Julian"):
            return None
        cal = args[2]
        try:
            tz = _tz(args[3], "DateObject")
        except (ValueError, Unevaluated):
            return None
        explicit = tz is not None or g in _FLOATING
        if len(args) == 5:
            if not isinstance(args[4], str):
                return None
            tsys = args[4]
    elif g not in _FLOATING:
        return None
    d = D(dl, g, cal, tz, tsys, fmt, explicit)
    try:
        canon, _, _ = _canon_start(d.wall(), g, cal)
    except (ValueError, OverflowError):
        return None
    if [_out(v) for v in canon] != [_out(v) for v in dl]:
        if not (g == "Instant" and all(K.same(_out(a), _out(b)) or a == b for a, b in zip(canon, dl))):
            return None
    return d


# ─────────────────────────── date strings ───────────────────────────

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
_MONTH_OF = {}
for _i, _m in enumerate(_MONTHS):
    _MONTH_OF[_m.lower()] = _i + 1
    _MONTH_OF[_m[:3].lower()] = _i + 1
_MONTH_OF["sept"] = 9
_DAY_OF = {}
for _i, _d in enumerate(_DAYS):
    _DAY_OF[_d.lower()] = _i
    _DAY_OF[_d[:3].lower()] = _i

_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?$")
_ISO_RE = re.compile(r"^\s*([+-]?\d{4,})(?:-(\d{1,2})(?:-(\d{1,2})(?:[T ](\d{1,2})(?::(\d{2})(?::(\d{2}(?:\.\d+)?))?)?"
                     r"\s*(Z|[+-]\d{2}:?\d{2})?)?)?)?\s*$")


def _sec(text):
    return float(text) if "." in text else int(text)


def _iso_zone(z):
    if not z:
        return None
    if z == "Z":
        return 0.0
    sign = -1 if z[0] == "-" else 1
    digits = z[1:].replace(":", "")
    return sign * (int(digits[:2]) + int(digits[2:]) / 60.0)


def parse_date_string(s, fname="DateObject"):
    """(date list, granularity, zone or None) for a date string."""
    m = _ISO_RE.match(s)
    if m:
        parts = [g for g in m.groups()[:6]]
        dl = []
        for i, p in enumerate(parts):
            if p is None:
                break
            dl.append(_sec(p) if i == 5 else int(p))
        if len(dl) == 4:
            dl.append(0)
        gran = _BY_LEN[len(dl)]
        if len(dl) == 6 and isinstance(dl[5], int):
            gran = "Second"
        _check_fields(dl, fname, s)
        return dl, gran, _iso_zone(m.group(7))
    toks = re.findall(r"\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?|[A-Za-z]+\.?|\d+(?:st|nd|rd|th)?|[+-]\d{2}:?\d{2}", s)
    if not toks or re.sub(r"[\s,./\-:]", "", s) == "":
        raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
    year = month = day = None
    hms = None
    ampm = None
    nums = []
    for t in toks:
        tl = t.lower().rstrip(".")
        if _TIME_RE.match(t):
            if hms is not None:
                raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
            mm = _TIME_RE.match(t)
            hms = [int(mm.group(1)), int(mm.group(2))] + ([_sec(mm.group(3))] if mm.group(3) else [])
        elif tl in ("am", "pm"):
            ampm = tl
        elif tl in _MONTH_OF:
            if month is not None:
                raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
            month = _MONTH_OF[tl]
        elif tl in _DAY_OF:
            continue
        elif re.match(r"^\d+(st|nd|rd|th)$", tl):
            day = int(re.match(r"\d+", tl).group())
        elif tl.isdigit():
            nums.append(t)
        else:
            raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
    rest = []
    for t in nums:
        if len(t) >= 3 and year is None:
            year = int(t)
        else:
            rest.append(int(t))
    if year is None and rest and month is not None and day is not None:
        year = rest.pop()
    if year is None and len(rest) == 3:
        a, b, c = rest
        if a > 31:
            year, rest = a, [b, c]
        else:
            year, rest = c, [a, b]
    if month is None:
        if len(rest) >= 2:
            a, b = rest[0], rest[1]
            first_year = bool(nums) and len(nums[0]) >= 3
            if first_year or a <= 12:
                month, day = (a, b) if (first_year or a <= 12) else (b, a)
            else:
                month, day = b, a
            rest = rest[2:]
        elif len(rest) == 1 and year is not None:
            month, rest = rest[0], []
    elif day is None and rest:
        day = rest.pop(0)
    if rest:
        if year is None and len(rest) == 1:
            year = rest[0]
        else:
            raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
    if year is None:
        if month is None and hms is None:
            raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
        year = _dt.date.today().year
    dl = [year]
    if month is not None:
        dl.append(month)
        if day is not None:
            dl.append(day)
    elif day is not None:
        raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
    if hms is not None:
        if len(dl) < 3:
            dl = (dl + [1, 1])[:3]
        h = hms[0]
        if ampm == "pm" and h < 12:
            h += 12
        elif ampm == "am" and h == 12:
            h = 0
        dl += [h] + hms[1:]
    gran = _BY_LEN[len(dl)]
    if len(dl) == 6 and isinstance(dl[5], int):
        gran = "Second"
    _check_fields(dl, fname, s)
    return dl, gran, None


def _check_fields(dl, fname, s):
    lim = [None, (1, 12), (1, 31), (0, 24), (0, 59), (0, 61)]
    for i, v in enumerate(dl[1:], 1):
        lo, hi = lim[i]
        if not lo <= v <= hi:
            raise ValueError(f"{fname}: cannot interpret {s!r} as a date")
    if len(dl) >= 3 and dl[2] > _days_in_month(dl[0], dl[1]):
        raise ValueError(f"{fname}: {s!r} is not a valid date")


_ELEM_RX = {
    "Year": r"(?P<Year>[+-]?\d{1,4})", "YearShort": r"(?P<YearShort>\d{2})", "Month": r"(?P<Month>\d{1,2})",
    "MonthShort": r"(?P<MonthShort>\d{1,2})", "Day": r"(?P<Day>\d{1,2})", "DayShort": r"(?P<DayShort>\d{1,2})",
    "Hour": r"(?P<Hour>\d{1,2})", "Hour12": r"(?P<Hour12>\d{1,2})", "Hour24": r"(?P<Hour24>\d{1,2})",
    "Minute": r"(?P<Minute>\d{1,2})", "Second": r"(?P<Second>\d{1,2}(?:\.\d+)?)",
    "Millisecond": r"(?P<Millisecond>\d{1,3})", "Quarter": r"(?P<Quarter>[1-4])",
    "MonthName": r"(?P<MonthName>[A-Za-z]+\.?)", "MonthNameShort": r"(?P<MonthNameShort>[A-Za-z]+\.?)",
    "DayName": r"(?P<DayName>[A-Za-z]+)", "DayNameShort": r"(?P<DayNameShort>[A-Za-z]+)",
    "AMPM": r"(?P<AMPM>[AaPp][Mm])",
}


def parse_with_elements(s, elems, fname="DateList"):
    """DateList[{"string", {"e1", "sep", "e2", ...}}]: (date list, granularity)."""
    elems = list(elems)
    for e in elems:
        if not isinstance(e, str):
            raise ValueError(f"{fname}: date elements must be strings")
    explicit = any(e not in _ELEM_RX for e in elems)
    parts = []
    for i, e in enumerate(elems):
        if e in _ELEM_RX:
            if parts and not explicit and elems[i - 1] in _ELEM_RX:
                parts.append(r"[^A-Za-z0-9]*")
            parts.append(_ELEM_RX[e])
        else:
            parts.append(re.escape(e))
    m = re.fullmatch(r"\s*" + "".join(parts) + r"\s*", s)
    if not m:
        raise ValueError(f"{fname}: the string {s!r} does not match the date elements {elems}")
    g = m.groupdict()
    year = int(g["Year"]) if g.get("Year") else (2000 + int(g["YearShort"]) if g.get("YearShort") else None)
    month = None
    if g.get("Month") or g.get("MonthShort"):
        month = int(g.get("Month") or g.get("MonthShort"))
    for k in ("MonthName", "MonthNameShort"):
        if g.get(k):
            nm = g[k].lower().rstrip(".")
            if nm not in _MONTH_OF:
                raise ValueError(f"{fname}: {g[k]!r} is not a month name")
            month = _MONTH_OF[nm]
    if g.get("Quarter") and month is None:
        month = 3 * (int(g["Quarter"]) - 1) + 1
    day = int(g.get("Day") or g.get("DayShort")) if (g.get("Day") or g.get("DayShort")) else None
    hour = None
    for k in ("Hour", "Hour24", "Hour12"):
        if g.get(k):
            hour = int(g[k])
    if g.get("AMPM") and hour is not None:
        if g["AMPM"].lower() == "pm" and hour < 12:
            hour += 12
        elif g["AMPM"].lower() == "am" and hour == 12:
            hour = 0
    minute = int(g["Minute"]) if g.get("Minute") else None
    second = _sec(g["Second"]) if g.get("Second") else None
    if g.get("Millisecond"):
        second = (second or 0) + int(g["Millisecond"]) / 1000.0
    vals = [year if year is not None else _dt.date.today().year, month, day, hour, minute, second]
    last = max([i for i, v in enumerate(vals) if v is not None] + [0])
    defaults = [None, 1, 1, 0, 0, 0]
    dl = [v if v is not None else defaults[i] for i, v in enumerate(vals[:last + 1])]
    gran = _BY_LEN[len(dl)]
    if len(dl) == 6 and isinstance(dl[5], int):
        gran = "Second"
    return dl, gran


# ─────────────────────────── formatting (DateString elements) ───────────────────────────

_NAMED_FORMATS = {
    "Date": ["DayName", " ", "DayShort", " ", "MonthName", " ", "Year"],
    "DateShort": ["DayNameShort", " ", "DayShort", " ", "MonthNameShort", " ", "Year"],
    "Time": ["Hour", ":", "Minute", ":", "Second"],
    "DateTime": ["DayName", " ", "DayShort", " ", "MonthName", " ", "Year", " ", "Hour", ":", "Minute", ":", "Second"],
    "DateTimeShort": ["DayNameShort", " ", "DayShort", " ", "MonthNameShort", " ", "Year", " ", "Hour", ":",
                      "Minute", ":", "Second"],
    "ISODate": ["Year", "-", "Month", "-", "Day"],
    "ISODateTime": ["Year", "-", "Month", "-", "Day", "T", "Hour", ":", "Minute", ":", "Second"],
    "ISOWeekDate": ["ISOYear", "-W", "Week", "-", "ISOWeekDay"],
    "ISOOrdinalDate": ["Year", "-", "DayOfYear"],
    "ISODateTimeMillisecond": ["Year", "-", "Month", "-", "Day", "T", "Hour", ":", "Minute", ":", "Second", ".",
                               "Millisecond"],
}
_ELEMENTS = {"Year", "YearShort", "ISOYear", "Quarter", "QuarterName", "QuarterNameShort", "Month", "MonthShort",
             "MonthName", "MonthNameShort", "MonthNameInitial", "Day", "DayShort", "DayName", "DayNameShort",
             "DayNameInitial", "DayOfYear", "Week", "WeekShort", "ISOWeekDay", "Hour", "Hour12", "Hour24",
             "HourShort", "Hour12Short", "Hour24Short", "AMPM", "AMPMLowerCase", "Minute", "MinuteShort", "Second",
             "SecondShort", "SecondExact", "Millisecond", "MillisecondShort", "TimeZone", "TimeZoneName",
             "TimeZoneNameShort"}


def is_format_spec(x):
    if isinstance(x, str):
        return x in _NAMED_FORMATS or x in _ELEMENTS
    if is_list(x):
        return any(isinstance(e, str) and (e in _ELEMENTS or e in _NAMED_FORMATS) for e in x.args) and \
            all(isinstance(e, str) for e in x.args)
    return False


def _iso_week(dl):
    d = _dt.date(dl[0], dl[1], dl[2]) if 1 <= dl[0] <= 9999 else None
    if d is None:
        raise ValueError("DateString: ISO week elements need a year between 1 and 9999")
    return d.isocalendar()


def _elem(e, dl, tz=None):
    y, m, d, h, mi, s = (list(dl) + [1, 1, 0, 0, 0][len(dl) - 1:])[:6] if len(dl) < 6 else dl
    if e == "Year":
        return str(y) if y > 0 else str(y)
    if e == "YearShort":
        return f"{abs(y) % 100:02d}"
    if e == "ISOYear":
        return str(_iso_week([y, m, d])[0])
    if e == "Quarter":
        return str((m - 1) // 3 + 1)
    if e in ("QuarterName", "QuarterNameShort"):
        q = (m - 1) // 3 + 1
        return f"Quarter {q}" if e == "QuarterName" else f"Q{q}"
    if e == "Month":
        return f"{m:02d}"
    if e == "MonthShort":
        return str(m)
    if e == "MonthName":
        return _MONTHS[m - 1]
    if e == "MonthNameShort":
        return _MONTHS[m - 1][:3]
    if e == "MonthNameInitial":
        return _MONTHS[m - 1][0]
    if e == "Day":
        return f"{d:02d}"
    if e == "DayShort":
        return str(d)
    if e in ("DayName", "DayNameShort", "DayNameInitial"):
        wd = (_jdn(_astro(y), m, d)) % 7
        nm = _DAYS[wd]
        return nm if e == "DayName" else nm[:3] if e == "DayNameShort" else nm[0]
    if e == "DayOfYear":
        return f"{_jdn(_astro(y), m, d) - _jdn(_astro(y), 1, 1) + 1:03d}"
    if e in ("Week", "WeekShort"):
        w = _iso_week([y, m, d])[1]
        return f"{w:02d}" if e == "Week" else str(w)
    if e == "ISOWeekDay":
        return str(_iso_week([y, m, d])[2])
    if e in ("Hour", "Hour24"):
        return f"{h:02d}"
    if e in ("HourShort", "Hour24Short"):
        return str(h)
    if e in ("Hour12", "Hour12Short"):
        hh = h % 12 or 12
        return f"{hh:02d}" if e == "Hour12" else str(hh)
    if e in ("AMPM", "AMPMLowerCase"):
        v = "AM" if h < 12 else "PM"
        return v if e == "AMPM" else v.lower()
    if e == "Minute":
        return f"{mi:02d}"
    if e == "MinuteShort":
        return str(mi)
    if e == "Second":
        return f"{int(_floor(s)):02d}"
    if e == "SecondShort":
        return str(int(_floor(s)))
    if e == "SecondExact":
        return f"{float(s):06.3f}"
    if e in ("Millisecond", "MillisecondShort"):
        ms = int(_floor((s - _floor(s)) * 1000 + 1e-6))
        return f"{ms:03d}" if e == "Millisecond" else str(ms)
    if e in ("TimeZone", "TimeZoneName", "TimeZoneNameShort"):
        if tz is None:
            return ""
        if isinstance(tz, str):
            return tz
        sign = "-" if tz < 0 else "+"
        a = abs(tz)
        return f"GMT{sign}{int(a)}" + (f":{int(round((a - int(a)) * 60)):02d}" if a != int(a) else "")
    return e


def format_date(dl, fmt, tz=None):
    if isinstance(fmt, str):
        if fmt in _NAMED_FORMATS:
            return "".join(_elem(e, dl, tz) for e in _NAMED_FORMATS[fmt])
        return _elem(fmt, dl, tz)
    if is_list(fmt):
        out = []
        for e in fmt.args:
            if not isinstance(e, str):
                raise ValueError("DateString: format elements must be strings")
            out.append(format_date(dl, e, tz) if e in _NAMED_FORMATS else _elem(e, dl, tz))
        return "".join(out)
    raise ValueError(f"DateString: {fmt!r} is not a date format specification")


# ─────────────────────────── date specifications ───────────────────────────

_TSYS = {"UT", "UTC", "SmearedUTC", "TAI", "TT", "TDB", "TCG", "TCB", "UT1", "GPS"}


def _now_utc():
    return Fraction(time.time_ns(), 10 ** 9) + 2208988800


def _now_wall(z):
    u = _now_utc()
    return float(u + _off_secs(_tz_offset(z, utc=u)))


def _instant_from_wall(t, z, cal="Gregorian", explicit=False):
    return _from_wall(t, "Instant", cal, z, tz_explicit=explicit)


def _list_spec(e, fname):
    vals = _nums(e.args, fname)
    if not 1 <= len(vals) <= 6:
        raise ValueError(f"{fname}: a date list has 1 to 6 elements {{y, m, d, h, m, s}}")
    exact_int = all(isinstance(v, int) for v in vals)
    gran = _BY_LEN[len(vals)] if exact_int else "Instant"
    if not all(isinstance(v, int) for v in vals[:2]):
        raise ValueError(f"{fname}: the year and month of a date list must be integers")
    return vals, gran


def to_date(x, fname, zone=None):
    """D for a date specification: DateObject, date list, absolute time, date string or {string, elements}.
    Non-DateObject specifications are interpreted in ``zone`` (default $TimeZone, floating for coarse dates)."""
    d = parse_dobj(x)
    if d is not None:
        return d
    if isinstance(x, Expr) and x.head == "DateObject":
        raise ValueError(f"{fname}: {x!r} is not a valid DateObject")
    if is_list(x):
        if len(x.args) == 2 and isinstance(x.args[0], str) and (is_list(x.args[1]) or isinstance(x.args[1], str)):
            el = x.args[1].args if is_list(x.args[1]) else [x.args[1]]
            dl, gran = parse_with_elements(x.args[0], el, fname)
            t = _list_to_abs(dl)
            return _from_wall(t, gran, tz=None if gran in _FLOATING else (zone or current_tz()))
        vals, gran = _list_spec(x, fname)
        t = _list_to_abs(vals, "Gregorian", fname)
        return _from_wall(t, gran, tz=None if gran in _FLOATING else (zone or current_tz()))
    if isinstance(x, str):
        dl, gran, z = parse_date_string(x, fname)
        t = _list_to_abs(dl)
        if z is not None:
            return _from_wall(t, gran, tz=z, tz_explicit=True)
        return _from_wall(t, gran, tz=None if gran in _FLOATING else (zone or current_tz()))
    if K.is_number(x) and not isinstance(x, bool):
        t = _num(x, fname, "absolute time")
        return _instant_from_wall(t, zone or current_tz())
    if isinstance(x, Expr) and x.head == "TimeObject":
        tt = parse_tobj(x)
        if tt is None:
            raise ValueError(f"{fname}: {x!r} is not a valid TimeObject")
        z = tt.tz if tt.tz is not None else (zone or current_tz())
        today = _abs_to_list(_now_wall(z))[:3]
        t = _list_to_abs(today) + _list_to_abs([1900, 1, 1] + tt.tl)
        return _from_wall(t, tt.gran, tz=z)
    if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not a date specification")


def _with_gran(d, g):
    if g == "MidpointInstant":
        s, e = d.bounds_wall()
        mid = s + (e - s) / 2 if not _is_exact(s) else s + Fraction(e - s, 2)
        if isinstance(mid, Fraction) and mid.denominator == 1:
            mid = int(mid)
        return _from_wall(mid, "Instant", d.cal, d.tz if d.tz is not None else
                          (None if False else current_tz()), d.tsys, d.fmt, True)
    g = _gran(g, "DateObject")
    s, _ = d.bounds_wall()
    tz = d.tz
    explicit = d.tz_explicit
    if tz is None and g not in _FLOATING:
        tz = current_tz()
    return _from_wall(s if g != "Instant" else d.wall(), g, d.cal, tz, d.tsys, d.fmt, explicit)


def _convert_tz(d, z):
    if d.tz is None and d.gran in _FLOATING:
        return D(d.dl, d.gran, d.cal, z, d.tsys, d.fmt, True)
    if d.tz == z:
        return D(d.dl, d.gran, d.cal, z, d.tsys, d.fmt, True)
    w = d.wall()
    u = d.to_utc(w)
    nw = u + _off_secs(_tz_offset(z, utc=u))
    return _from_wall(nw, d.gran, d.cal, z, d.tsys, d.fmt, True)


def _apply_date_opts(d, o, fname):
    if "DateGranularity" in o and not is_sym(o["DateGranularity"], "Automatic"):
        d = _with_gran(d, o["DateGranularity"])
    if "TimeZone" in o:
        z = _tz(o["TimeZone"], fname)
        d = _convert_tz(d, z)
    if "CalendarType" in o and not is_sym(o["CalendarType"], "Automatic"):
        c = o["CalendarType"]
        if c not in ("Gregorian", "Julian"):
            raise NotImplementedError(f"{fname}: CalendarType {c!r} is not supported (Gregorian and Julian only)")
        if c != d.cal:
            d = _from_wall(d.wall(), d.gran, c, d.tz, d.tsys, d.fmt, d.tz_explicit)
    if "TimeSystem" in o and not is_sym(o["TimeSystem"], "Automatic"):
        ts = o["TimeSystem"]
        if ts not in _TSYS:
            raise ValueError(f"{fname}: {ts!r} is not a known time system")
        d = D(d.dl, d.gran, d.cal, d.tz if d.tz is not None else current_tz(), ts, d.fmt, True)
    if "DateFormat" in o and not is_sym(o["DateFormat"], "Automatic"):
        f = o["DateFormat"]
        if not (is_format_spec(f) or isinstance(f, str) or is_assoc(f)):
            raise ValueError(f"{fname}: {f!r} is not a DateFormat specification")
        d = D(d.dl, d.gran, d.cal, d.tz, d.tsys, f, d.tz_explicit)
    return d


_DATE_OPTS = {"CalendarType", "DateFormat", "DateGranularity", "TimeSystem", "TimeZone"}


@fn("DateObject", **ADDED,
    notes="DateObject[] (now, \"Instant\" in $TimeZone), DateObject[date] for a date list, absolute time, date "
          "string, {\"string\", {e1, ...}} or DateObject, DateObject[date, gran] (all 20 calendar granularities, "
          "integer 0..9 precision, \"MidpointInstant\"), DateObject[date, gran, cal, tz, tsys] and the options "
          "CalendarType (\"Gregorian\" or \"Julian\"), DateFormat, DateGranularity, TimeSystem, TimeZone; out-of-range "
          "elements are reduced ({2022, 2, 31} -> {2022, 3, 3}); canonical forms are DateObject[list, gran] for "
          "floating day-and-coarser dates and DateObject[list, gran, cal, tz] otherwise; d[\"Date\"], "
          "d[\"Granularity\"], d[\"TimeZone\"], d[\"CalendarType\"], d[\"DateList\"] and date elements such as "
          "d[\"Year\"], d[\"DayName\"]")
def _dateobject(*args):
    raw = args
    args, o = _opts(args, _DATE_OPTS, "DateObject")
    d0 = parse_dobj(Expr("DateObject", *raw))
    if d0 is not None:
        return d0.expr()
    if not args:
        z = current_tz()
        if "TimeZone" in o:
            z = _tz(o["TimeZone"], "DateObject", allow_none=False)
            o = dict(o)
            del o["TimeZone"]
        d = _instant_from_wall(_now_wall(z), z)
        return _apply_date_opts(d, o, "DateObject").expr()
    if len(args) == 2 and isinstance(args[1], Expr) and args[1].head == "TimeObject":
        dd = to_date(args[0], "DateObject")
        tt = parse_tobj(args[1])
        if tt is None:
            raise ValueError("DateObject: the second argument is not a valid TimeObject")
        day = _with_gran(dd, "Day")
        t = day.wall() + _list_to_abs([1900, 1, 1] + tt.tl)
        z = tt.tz if tt.tz is not None else current_tz()
        d = _from_wall(t, tt.gran, tz=z)
        return _apply_date_opts(d, o, "DateObject").expr()
    if len(args) > 5:
        raise ValueError("DateObject: too many arguments")
    zone = None
    if len(args) >= 4:
        zone = _tz(args[3], "DateObject")
    elif "TimeZone" in o and parse_dobj(args[0]) is None:
        # a date list, string or absolute time is given in the zone of the TimeZone option
        zone = _tz(o["TimeZone"], "DateObject")
        o = dict(o)
        del o["TimeZone"]
        d = to_date(args[0], "DateObject", zone)
        d = _convert_tz(d, zone) if d.tz_explicit else D(d.dl, d.gran, d.cal, zone, d.tsys, d.fmt, True)
        args = (d.expr(),) + tuple(args[1:])
        zone = None
    d = to_date(args[0], "DateObject", zone)
    if len(args) >= 2:
        d = _with_gran(d, args[1])
    if len(args) >= 3:
        if args[2] not in ("Gregorian", "Julian"):
            raise NotImplementedError(f"DateObject: calendar {args[2]!r} is not supported (Gregorian and Julian only)")
        if args[2] != d.cal:
            d = _from_wall(d.wall(), d.gran, args[2], d.tz, d.tsys, d.fmt, d.tz_explicit)
    if len(args) >= 4:
        d = _convert_tz(d, zone) if parse_dobj(args[0]) is not None else \
            D(d.dl, d.gran, d.cal, zone, d.tsys, d.fmt, True)
    if len(args) == 5:
        if args[4] not in _TSYS:
            raise ValueError(f"DateObject: {args[4]!r} is not a known time system")
        d.tsys = args[4]
    return _apply_date_opts(d, o, "DateObject").expr()


def _date_property(d, prop):
    if prop == "Date":
        return date_string(d)
    if prop == "Granularity":
        return d.gran
    if prop == "TimeZone":
        return _tz_out(d.tz)
    if prop == "CalendarType":
        return d.cal
    if prop == "TimeSystem":
        return d.tsys or "UT"
    if prop == "DateFormat":
        return d.fmt if d.fmt is not None else AUTOMATIC
    if prop == "DateList":
        return _datelist_expr(d)
    if prop == "AbsoluteTime":
        return _out(_abs_time(d))
    if prop in ("Year", "Month", "Day", "Hour", "Minute", "Second"):
        i = ["Year", "Month", "Day", "Hour", "Minute", "Second"].index(prop)
        full = _pad(d)
        return _out(full[i]) if i < 5 else _out(full[5])
    if prop in ("DayName", "MonthName"):
        return Symbol(_elem(prop, _pad(d)))
    if prop in ("Week", "Quarter", "DayOfYear", "ISOWeekDay"):
        return int(_elem(prop, _pad(d)))
    if prop == "Properties":
        return L("AbsoluteTime", "CalendarType", "Date", "DateFormat", "DateList", "Day", "DayName", "DayOfYear",
                 "Granularity", "Hour", "ISOWeekDay", "Minute", "Month", "MonthName", "Quarter", "Second",
                 "TimeSystem", "TimeZone", "Week", "Year")
    raise ValueError(f"DateObject: unknown property {prop!r}")


def _date_hook(head, args, ev_):
    d = parse_dobj(head)
    if d is None or len(args) != 1 or not isinstance(args[0], str):
        return K.UNHANDLED
    return _date_property(d, args[0])


KL.COMPOUND_HEAD_HOOKS["DateObject"] = _date_hook


def _pad(d):
    dl = list(d.dl)
    return dl + [1, 1, 0, 0, 0][len(dl) - 1:] if len(dl) < 6 else dl


def _abs_time(d, zone=None):
    """Absolute time of the start of d expressed in zone (default $TimeZone)."""
    s = d.wall() if d.gran == "Instant" else d.bounds_wall()[0]
    if d.tz is None:
        return s
    z = zone if zone is not None else current_tz()
    if z == d.tz:
        return s
    u = d.to_utc(s)
    return u + _off_secs(_tz_offset(z, utc=u))


def _float_secs(dl):
    out = list(dl)
    out[5] = float(out[5])
    return out


def _datelist_expr(d):
    return L(*(_out(v) for v in _float_secs(_pad(d))))


def date_string(d, fmt=None):
    if fmt is None:
        fmt = d.fmt
    if fmt is None:
        fmt = ev(Symbol("$DateStringFormat"))
        if is_list(fmt) and len(fmt.args) == 1 and isinstance(fmt.args[0], str):
            fmt = fmt.args[0]
        if fmt == "DateTimeShort" and d.gran in _FLOATING:
            fmt = {1: "Year", 2: L("MonthNameShort", " ", "Year"), 3: "DateShort"}[_glen(d.gran)]
    return format_date(_pad(d), fmt, d.tz)


@fn("DateObjectQ", **ADDED, notes="True for a DateObject with a valid canonical specification, else False")
def _dateobjectq(x):
    return parse_dobj(x) is not None


@fn("DateList", **ADDED,
    notes="DateList[] (now in $TimeZone), DateList[date] for DateObject, date list (out-of-range values reduced, "
          "fractional d/h/m/s allowed), absolute time (seconds since 1900 in $TimeZone), date string or "
          "{\"string\", {e1, sep, e2, ...}}; always {y, m, d, h, m, s} with real seconds; DateList[TimeZone -> z]")
def _datelist(*args):
    args, o = _opts(args, {"TimeZone"}, "DateList")
    if len(args) > 1:
        raise ValueError("DateList: expected at most one date specification")
    if not args:
        z = _tz(o.get("TimeZone", AUTOMATIC), "DateList", allow_none=False)
        return L(*(_out(v) for v in _float_secs(_abs_to_list(_now_wall(z)))))
    x = args[0]
    d = parse_dobj(x)
    if d is None and (is_list(x) and not (len(x.args) == 2 and isinstance(x.args[0], str))):
        vals, _ = _list_spec(x, "DateList")
        return L(*(_out(v) for v in _float_secs(_abs_to_list(_list_to_abs(vals, "Gregorian", "DateList")))))
    if d is None and K.is_number(x) and not isinstance(x, bool):
        return L(*(_out(v) for v in _float_secs(_abs_to_list(_num(x, "DateList", "absolute time")))))
    if d is None:
        d = to_date(x, "DateList")
    if "TimeZone" in o and d.tz is not None:
        d = _convert_tz(d, _tz(o["TimeZone"], "DateList", allow_none=False))
    return _datelist_expr(d) if d.gran == "Instant" else L(*(_out(v) for v in _float_secs(
        _abs_to_list(d.bounds_wall()[0], d.cal))))


@fn("AbsoluteTime", **ADDED,
    notes="seconds since 1900-01-01 00:00 in $TimeZone: AbsoluteTime[] (machine real), AbsoluteTime[date] for "
          "DateObject (converted from its zone), date list (exact for exact input: AbsoluteTime[{2000, 1, 1}] = "
          "3155673600), date string or absolute time; TimeZone -> z gives the value in zone z; leap seconds are not "
          "counted")
def _absolutetime(*args):
    args, o = _opts(args, {"TimeZone"}, "AbsoluteTime")
    if len(args) > 1:
        raise ValueError("AbsoluteTime: expected at most one date specification")
    z = _tz(o["TimeZone"], "AbsoluteTime", allow_none=False) if "TimeZone" in o else None
    if not args:
        return _now_wall(z if z is not None else current_tz())
    x = args[0]
    if K.is_number(x) and not isinstance(x, bool):
        t = _num(x, "AbsoluteTime", "absolute time")
        if z is None:
            return _out(t)
        u = t - _off_secs(_tz_offset(current_tz(), wall=t))
        return _out(u + _off_secs(_tz_offset(z, utc=u)))
    d = parse_dobj(x)
    if d is None and is_list(x) and not (len(x.args) == 2 and isinstance(x.args[0], str)):
        vals, _ = _list_spec(x, "AbsoluteTime")
        t = _list_to_abs(vals, "Gregorian", "AbsoluteTime")
        if z is None:
            return _out(t)
        d = _instant_from_wall(t, current_tz())
    if d is None:
        d = to_date(x, "AbsoluteTime")
    if d.tz is None and z is not None:
        d = D(d.dl, d.gran, d.cal, current_tz(), d.tsys, d.fmt, True)
    return _out(_abs_time(d, z))


# ─────────────────────────── DateString (extended to date objects and formats) ───────────────────────────

def _wrap_datestring():
    e = REGISTRY.get("DateString")
    old = list(e.impls) if e is not None else []

    def old_call(*a):
        for f_, lo, hi in old:
            if lo <= len(a) and (hi is None or len(a) <= hi):
                return f_(*a)
        raise ValueError("DateString: wrong number of arguments")

    def ds(*args):
        args, o = _opts(args, {"TimeZone"}, "DateString")
        if len(args) > 2:
            raise ValueError("DateString: expected DateString[], DateString[date], DateString[fmt] or "
                             "DateString[date, fmt]")
        if not args:
            fmt = None
            d = None
        elif len(args) == 1 and is_format_spec(args[0]):
            fmt, d = args[0], None
        else:
            fmt = args[1] if len(args) == 2 else None
            if fmt is not None and not (is_format_spec(fmt) or (is_list(fmt) and all(isinstance(v, str)
                                                                                      for v in fmt.args))):
                raise ValueError(f"DateString: {fmt!r} is not a date format specification")
            x = args[0]
            if is_list(x) and not (len(x.args) == 2 and isinstance(x.args[0], str)):
                # a date list is a full {y, m, d, h, m, s} date (DateString[{2020, 1, 2}] includes 00:00:00)
                vals, _ = _list_spec(x, "DateString")
                d = _instant_from_wall(_list_to_abs(vals, "Gregorian", "DateString"), current_tz())
                if fmt is None:
                    fmt = _default_format()
            else:
                d = to_date(x, "DateString")
        if d is None:
            z = _tz(o.get("TimeZone", AUTOMATIC), "DateString", allow_none=False)
            d = _instant_from_wall(_now_wall(z), z)
            if fmt is None:
                return format_date(_pad(d), _default_format())
        elif "TimeZone" in o:
            d = _convert_tz(d, _tz(o["TimeZone"], "DateString", allow_none=False))
        return date_string(d, fmt)

    REGISTRY["DateString"].impls[:] = [(ds, 0, None)]
    record("DateString", AREA, "fixed",
           "DateString[], DateString[date] (DateObject, date list, absolute time, date string), DateString[fmt] and "
           "DateString[date, fmt] with named formats (\"DateTimeShort\", \"ISODate\", \"ISODateTime\", "
           "\"ISOWeekDate\", ...) and element lists ({\"Day\", \"/\", \"Month\"}); default $DateStringFormat; W12 "
           "extended the strings-batch version (date lists keep \"Thu 2 Jan 2020 00:00:00\")")


def _default_format():
    f = ev(Symbol("$DateStringFormat"))
    if is_list(f) and len(f.args) == 1 and isinstance(f.args[0], str):
        return f.args[0]
    return f


_wrap_datestring()


# ─────────────────────────── TimeObject ───────────────────────────

_TGRANS = {"Hour": 1, "Minute": 2, "Second": 3, "Decisecond": 3, "Centisecond": 3, "Millisecond": 3,
           "Microsecond": 3, "Nanosecond": 3, "Instant": 3}


class T:
    __slots__ = ("tl", "gran", "tz")

    def __init__(self, tl, gran, tz=None):
        self.tl, self.gran, self.tz = tl, gran, tz

    def secs(self):
        tl = self.tl + [0, 0][: 3 - len(self.tl)]
        return tl[0] * 3600 + tl[1] * 60 + tl[2]

    def bounds(self):
        _, s, e = _canon_start(_list_to_abs([1900, 1, 1]) + self.secs(), self.gran, "Gregorian")
        return s, e

    def expr(self):
        out = [L(*(_out(v) for v in self.tl)), self.gran]
        if self.tz is not None:
            out.append(Expr("Rule", Symbol("TimeZone"), self.tz))
        return Expr("TimeObject", *out)


def _tgran(g, fname):
    g = _gran(g, fname)
    if isinstance(g, str) and g not in _TGRANS:
        raise ValueError(f"{fname}: {g!r} is not a time granularity")
    return g


def _time_from_secs(secs, gran, tz=None):
    secs = secs % 86400
    dl, _, _ = _canon_start(secs, gran, "Gregorian")
    n = 1 if gran == "Hour" else 2 if gran == "Minute" else 3
    return T(dl[3:3 + n], gran, tz)


def parse_tobj(e):
    if not (isinstance(e, Expr) and e.head == "TimeObject" and len(e.args) in (2, 3) and is_list(e.args[0])):
        return None
    tz = None
    if len(e.args) == 3:
        r = e.args[2]
        if not (is_rule(r) and is_sym(r.args[0], "TimeZone")):
            return None
        try:
            tz = _tz(r.args[1], "TimeObject")
        except (ValueError, Unevaluated):
            return None
    try:
        g = _tgran(e.args[1], "TimeObject")
        tl = _nums(e.args[0].args, "TimeObject")
    except (ValueError, Unevaluated):
        return None
    n = 1 if g == "Hour" else 2 if g == "Minute" else 3
    if len(tl) != n or not all(isinstance(v, int) for v in tl[:2]):
        return None
    t = T(tl, g, tz)
    c = _time_from_secs(t.secs(), g, tz)
    if [_out(v) for v in c.tl] != [_out(v) for v in tl]:
        return None
    return t


@fn("TimeObject", **ADDED,
    notes="TimeObject[] (now, \"Instant\", TimeZone -> $TimeZone), TimeObject[{h}], {h, m}, {h, m, s} (granularity "
          "\"Hour\", \"Minute\", \"Second\"; real seconds -> \"Instant\"), TimeObject[n] (= {n}), TimeObject[date] "
          "(time part of a DateObject of granularity Hour or finer), TimeObject[t, gran] (Hour ... Nanosecond, "
          "Instant, 0..9); options DateGranularity and TimeZone (None by default); times are reduced modulo 24 h; "
          "t[\"Time\"], t[\"Granularity\"], t[\"TimeZone\"], t[\"Hour\"], t[\"Minute\"], t[\"Second\"]")
def _timeobject(*args):
    raw = args
    args, o = _opts(args, {"DateGranularity", "TimeZone"}, "TimeObject")
    t0 = parse_tobj(Expr("TimeObject", *raw))
    if t0 is not None:
        return t0.expr()
    if len(args) > 2:
        raise ValueError("TimeObject: expected TimeObject[time] or TimeObject[time, gran]")
    tz = _tz(o["TimeZone"], "TimeObject") if "TimeZone" in o else None
    if not args:
        z = tz if tz is not None else current_tz()
        w = _now_wall(z)
        t = _time_from_secs(w - _list_to_abs(_abs_to_list(w)[:3]), "Instant", z)
    else:
        x = args[0]
        if K.is_number(x) and not isinstance(x, bool):
            x = L(x)
        t = None
        if is_list(x):
            vals = _nums(x.args, "TimeObject")
            if not 1 <= len(vals) <= 3:
                raise ValueError("TimeObject: a time list is {h}, {h, m} or {h, m, s}")
            exact = all(isinstance(v, int) for v in vals)
            g = ("Hour", "Minute", "Second")[len(vals) - 1] if exact else "Instant"
            secs = sum(v * k for v, k in zip(vals, (3600, 60, 1)))
            t = _time_from_secs(secs, g, tz)
        else:
            tt = parse_tobj(x)
            if tt is not None:
                t = tt
            else:
                d = to_date(x, "TimeObject")
                if d.gran in _FLOATING:
                    raise ValueError("TimeObject: the date has no time component (granularity coarser than Hour)")
                w = d.wall()
                t = _time_from_secs(w - _list_to_abs(_pad(d)[:3]), d.gran, d.tz)
        if tz is not None and t.tz is not None and tz != t.tz:
            off = _off_secs(_tz_offset(tz, utc=0)) - _off_secs(_tz_offset(t.tz, utc=0))
            t = _time_from_secs(t.secs() + off, t.gran, tz)
        elif tz is not None:
            t = T(t.tl, t.gran, tz)
    g = args[1] if len(args) == 2 else o.get("DateGranularity", AUTOMATIC)
    if not is_sym(g, "Automatic"):
        g = _tgran(g, "TimeObject")
        s, _ = t.bounds()
        t = _time_from_secs((t.secs() if g == "Instant" else s - _list_to_abs([1900, 1, 1])), g, t.tz)
    return t.expr()


def _time_hook(head, args, ev_):
    t = parse_tobj(head)
    if t is None or len(args) != 1 or not isinstance(args[0], str):
        return K.UNHANDLED
    p = args[0]
    tl = t.tl + [0, 0][: 3 - len(t.tl)]
    if p == "Time":
        return format_date([1900, 1, 1] + tl, "Time")
    if p == "Granularity":
        return t.gran
    if p == "TimeZone":
        return _tz_out(t.tz)
    if p in ("Hour", "Minute", "Second"):
        return _out(tl[["Hour", "Minute", "Second"].index(p)])
    if p == "Properties":
        return L("Granularity", "Hour", "Minute", "Second", "Time", "TimeZone")
    raise ValueError(f"TimeObject: unknown property {p!r}")


KL.COMPOUND_HEAD_HOOKS["TimeObject"] = _time_hook


@fn("TimeObjectQ", **ADDED, notes="True for a TimeObject with a valid canonical specification, else False")
def _timeobjectq(x):
    return parse_tobj(x) is not None


# ─────────────────────────── system variables ───────────────────────────

def _install_vars():
    from ..evaluate import SYSTEM_VARIABLES
    SYSTEM_VARIABLES["$TimeZone"] = _system_tz
    SYSTEM_VARIABLES["$DateStringFormat"] = lambda: "DateTimeShort"

    def _assign_tz(lhs, rhs, ev_):
        v = ev_(rhs)
        _tz(v, "$TimeZone", allow_none=False)
        K.state().own["$TimeZone"] = v
        return v

    def _assign_fmt(lhs, rhs, ev_):
        v = ev_(rhs)
        if not (is_format_spec(v) or isinstance(v, str) or (is_list(v) and all(isinstance(e, str) for e in v.args))):
            raise ValueError(f"$DateStringFormat: {v!r} is not a date format specification")
        K.state().own["$DateStringFormat"] = v
        return v
    K.ASSIGN_HOOKS["$TimeZone"] = _assign_tz
    K.ASSIGN_HOOKS["$DateStringFormat"] = _assign_fmt


_install_vars()
_added("$TimeZone", "time zone assumed by date functions: hours east of GMT (default the system zone, a real "
                    "number) or an IANA zone name; assignable (validated) and Block-able")
_added("$DateStringFormat", "default DateString format (\"DateTimeShort\"); assignable to any named format or "
                            "element list, and Block-able")


# ─────────────────────────── date intervals, bounds and comparisons ───────────────────────────

def _finest(grans):
    best = None
    for g in grans:
        k = _FINENESS.index(_gran_name(g))
        if best is None or k < best[0]:
            best = (k, g)
    return best[1]


def _gran_name(g):
    if isinstance(g, int):
        return "Instant" if g > 3 else _SUBSEC_NAME.get(g, "Instant")
    return g


class DI:
    """A DateInterval: [(start D, end D)] at one granularity (granular ends are inclusive dates)."""
    __slots__ = ("parts", "gran", "cal", "tz", "explicit")

    def __init__(self, parts, gran, cal="Gregorian", tz=None, explicit=False):
        self.parts, self.gran, self.cal, self.tz, self.explicit = parts, gran, cal, tz, explicit

    def utc_parts(self):
        out = []
        for s, e in self.parts:
            us = s.bounds_utc()[0]
            ue = e.to_utc(e.wall()) if self.gran == "Instant" else e.bounds_utc()[1]
            out.append((us, ue))
        return out

    def expr(self):
        lst = L(*(L(L(*(_out(v) for v in s.dl)), L(*(_out(v) for v in e.dl))) for s, e in self.parts))
        out = [lst, self.gran]
        floating = self.gran in _FLOATING and self.tz is None and not self.explicit
        if not floating or self.cal != "Gregorian":
            out += [self.cal, _tz_out(self.tz)]
        return Expr("DateInterval", *out)


def _date_at(d, g, end=False):
    """d re-expressed at granularity g: the period containing its start (or, for end=True, its last instant)."""
    s, e = d.bounds_wall()
    if g == "Instant":
        t = (e if end and d.gran != "Instant" else (d.wall() if d.gran == "Instant" else s))
        return _from_wall(t, "Instant", d.cal, d.tz if d.tz is not None else current_tz(), d.tsys, None, True)
    if end and d.gran != "Instant":
        _, gs, ge = _canon_start(s, g, d.cal)
        if ge < e:
            # the last g-period that starts before the end of d
            last = e - (Fraction(1, 10 ** 9) if _is_exact(e) else 1e-6)
            return _from_wall(last, g, d.cal, d.tz, d.tsys, None, d.tz_explicit)
    return _from_wall(s if d.gran != "Instant" else d.wall(), g, d.cal, d.tz, d.tsys, None, d.tz_explicit)


def parse_interval(e):
    if not (isinstance(e, Expr) and e.head == "DateInterval" and len(e.args) in (2, 4) and is_list(e.args[0])
            and e.args[0].args):
        return None
    try:
        g = _gran(e.args[1], "DateInterval")
    except (ValueError, Unevaluated):
        return None
    cal, tz, explicit = "Gregorian", None, False
    if len(e.args) == 4:
        if e.args[2] not in ("Gregorian", "Julian"):
            return None
        cal = e.args[2]
        try:
            tz = _tz(e.args[3], "DateInterval")
        except (ValueError, Unevaluated):
            return None
        explicit = True
    elif g not in _FLOATING:
        return None
    parts = []
    for p in e.args[0].args:
        if not (is_list(p) and len(p.args) == 2):
            return None
        ds = []
        for x in p.args:
            args = [x, g] + ([cal, _tz_out(tz)] if len(e.args) == 4 else [])
            d = parse_dobj(Expr("DateObject", *args))
            if d is None:
                return None
            ds.append(d)
        parts.append(tuple(ds))
    return DI(parts, g, cal, tz, explicit)


def _interval_from(pairs, gran, fname, o):
    """Normalise (start D, end D) pairs at a common granularity, sorted and merged."""
    ds = [d for p in pairs for d in p]
    g = _gran(gran, fname) if gran is not None else _finest([d.gran for d in ds])
    zones = {d.tz for d in ds if d.tz is not None}
    tz = None
    if "TimeZone" in o:
        tz = _tz(o["TimeZone"], fname)
    elif len(zones) == 1:
        tz = zones.pop()
    elif zones:
        tz = current_tz()
    if g not in _FLOATING and tz is None:
        tz = current_tz()
    conv = []
    for s, e in pairs:
        if tz is not None and s.tz is not None and s.tz != tz:
            s = _convert_tz(s, tz)
        if tz is not None and e.tz is not None and e.tz != tz:
            e = _convert_tz(e, tz)
        s2, e2 = _date_at(s, g), _date_at(e, g, end=True)
        s2 = D(s2.dl, g, s2.cal, tz, None, None, tz is not None)
        e2 = D(e2.dl, g, e2.cal, tz, None, None, tz is not None)
        if _wall_key(e2, g, True) < _wall_key(s2, g, False):
            raise ValueError(f"{fname}: the end date precedes the start date")
        conv.append((s2, e2))
    conv.sort(key=lambda p: _wall_key(p[0], g, False))
    merged = []
    for s, e in conv:
        if merged and _wall_key(s, g, False) <= _wall_key(merged[-1][1], g, True):
            if _wall_key(e, g, True) > _wall_key(merged[-1][1], g, True):
                merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return DI(merged, g, "Gregorian", tz, tz is not None)


def _wall_key(d, g, end):
    if g == "Instant":
        return d.wall()
    s, e = d.bounds_wall()
    return e if end else s


@fn("DateInterval", **ADDED,
    notes="DateInterval[{start, end}], DateInterval[{start, end}, gran], DateInterval[{{s1, e1}, ...}] (union, "
          "sorted and merged) and DateInterval[gdate]; dates may be DateObjects, date lists, absolute times or "
          "strings; the finest granularity of the endpoints is used by default, the end date's period is included "
          "(its final instant excluded); canonical form DateInterval[{{startlist, endlist}, ...}, gran] (floating) or "
          "DateInterval[..., gran, cal, tz]; properties \"DateBounds\", \"Granularity\", \"TimeZone\", "
          "\"CalendarType\"")
def _dateinterval(*args):
    raw = Expr("DateInterval", *args)
    di = parse_interval(raw)
    if di is not None:
        return di.expr()
    args, o = _opts(args, _DATE_OPTS, "DateInterval")
    if not 1 <= len(args) <= 2:
        raise ValueError("DateInterval: expected DateInterval[{start, end}] or DateInterval[{start, end}, gran]")
    x = args[0]
    gran = args[1] if len(args) == 2 else o.get("DateGranularity")
    if gran is not None and is_sym(gran, "Automatic"):
        gran = None
    d = parse_dobj(x)
    if d is not None:
        pairs = [(d, d)]
    elif is_list(x) and x.args and all(is_list(p) and len(p.args) == 2 and not isinstance(p.args[0], str)
                                       and not K.is_number(p.args[0]) for p in x.args):
        pairs = [(to_date(p.args[0], "DateInterval"), to_date(p.args[1], "DateInterval")) for p in x.args]
    elif is_list(x) and len(x.args) == 2:
        pairs = [(to_date(x.args[0], "DateInterval"), to_date(x.args[1], "DateInterval"))]
    elif isinstance(x, (Symbol,)) or (isinstance(x, Expr) and not is_list(x) and x.head != "DateObject"):
        raise Unevaluated()
    else:
        raise ValueError("DateInterval: expected a {start, end} pair of dates")
    return _interval_from(pairs, gran, "DateInterval", o).expr()


def _interval_hook(head, args, ev_):
    di = parse_interval(head)
    if di is None or len(args) != 1 or not isinstance(args[0], str):
        return K.UNHANDLED
    p = args[0]
    if p == "DateBounds":
        return L(_min_of_interval(di), _max_of_interval(di))
    if p == "Granularity":
        return di.gran
    if p == "TimeZone":
        return _tz_out(di.tz)
    if p == "CalendarType":
        return di.cal
    if p == "Properties":
        return L("CalendarType", "DateBounds", "Granularity", "TimeZone")
    raise ValueError(f"DateInterval: unknown property {p!r}")


KL.COMPOUND_HEAD_HOOKS["DateInterval"] = _interval_hook


def _min_of_interval(di, g=None):
    s = di.parts[0][0]
    return (s if g is None else _with_gran(s, g)).expr()


def _max_of_interval(di, g=None):
    e = di.parts[-1][1]
    if g is None:
        return e.expr()
    return _date_at(e, _gran(g, "MaxDate"), end=True).expr()


def _utc_span(x, fname):
    """(utc start, utc end) of a date, granular date or interval specification."""
    di = parse_interval(x)
    if di is not None:
        ps = di.utc_parts()
        return ps[0][0], ps[-1][1]
    if K.is_number(x) and not isinstance(x, bool):
        t = _num(x, fname, "absolute time")
        u = t - _off_secs(_tz_offset(current_tz(), wall=t))
        return u, u
    d = to_date(x, fname)
    if d.gran == "Instant":
        u = d.to_utc(d.wall())
        return u, u
    return d.bounds_utc()


def _date_items(x, fname):
    if not is_list(x) or not x.args:
        raise ValueError(f"{fname}: expected a non-empty list of dates")
    return list(x.args)


def _extreme(x, fname, pick):
    items = _date_items(x, fname)
    keyed = [(_utc_span(v, fname), i) for i, v in enumerate(items)]
    best = keyed[0]
    for k in keyed[1:]:
        if pick(k[0], best[0]):
            best = k
    return items[best[1]]


def _earlier(a, b):
    return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])


def _later(a, b):
    return a[1] > b[1] or (a[1] == b[1] and a[0] > b[0])


@fn("MinDate", **ADDED,
    notes="MinDate[{d1, d2, ...}] gives the earliest date in the form given (DateObject, date list, absolute time, "
          "string, DateInterval); MinDate[interval] the start of a DateInterval, MinDate[interval, gran] at "
          "granularity gran")
def _mindate(x, g=None):
    di = parse_interval(x)
    if di is not None:
        return _min_of_interval(di, g)
    if g is not None:
        raise ValueError("MinDate: a granularity can only be given for a DateInterval")
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise ValueError("MinDate: expected a list of dates or a DateInterval")
    return _extreme(x, "MinDate", _earlier)


@fn("MaxDate", **ADDED,
    notes="MaxDate[{d1, d2, ...}] gives the latest date (latest end of its period) in the form given; "
          "MaxDate[interval] the end date of a DateInterval, MaxDate[interval, gran] at granularity gran")
def _maxdate(x, g=None):
    di = parse_interval(x)
    if di is not None:
        return _max_of_interval(di, g)
    if g is not None:
        raise ValueError("MaxDate: a granularity can only be given for a DateInterval")
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise ValueError("MaxDate: expected a list of dates or a DateInterval")
    return _extreme(x, "MaxDate", _later)


@fn("DateBounds", **ADDED,
    notes="DateBounds[{d1, d2, ...}] = {earliest, latest} in the form given; DateBounds[interval] the first and last "
          "dates of a DateInterval; DateBounds[gdate] the first and final instants of a granular DateObject (the "
          "final instant is the start of the next period); DateBounds[obj, gran] at granularity gran")
def _datebounds(x, g=None):
    di = parse_interval(x)
    if di is not None:
        return L(_min_of_interval(di, g), _max_of_interval(di, g))
    d = parse_dobj(x)
    if d is not None:
        s, e = d.bounds_wall()
        z = d.tz if d.tz is not None else current_tz()
        a = _from_wall(s, "Instant", d.cal, z, d.tsys, None, True)
        b = _from_wall(e, "Instant", d.cal, z, d.tsys, None, True)
        if g is not None:
            a, b = _with_gran(a, g), _date_at(b, _gran(g, "DateBounds"), end=False)
        return L(a.expr(), b.expr())
    if g is not None:
        raise ValueError("DateBounds: a granularity can only be given for a DateObject or DateInterval")
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise ValueError("DateBounds: expected a list of dates, a DateObject or a DateInterval")
    return L(_extreme(x, "DateBounds", _earlier), _extreme(x, "DateBounds", _later))


def _outer2(f, a, b, fname):
    if is_list(a) and parse_dobj(a) is None and not all(K.is_number(v) for v in a.args):
        return L(*(_outer2(f, x, b, fname) for x in a.args))
    if is_list(b) and parse_dobj(b) is None and not all(K.is_number(v) for v in b.args):
        return L(*(_outer2(f, a, y, fname) for y in b.args))
    for v in (a, b):
        if not (parse_dobj(v) is not None or parse_interval(v) is not None):
            if isinstance(v, (Symbol, Expr)) and not isinstance(v, str) and not is_list(v):
                raise Unevaluated()
            raise ValueError(f"{fname}: {v!r} is not a DateObject or DateInterval")
    return f(_utc_span(a, fname), _utc_span(b, fname), a, b)


@fn("DateWithinQ", **ADDED,
    notes="DateWithinQ[d1, d2] is True if the period of DateObject (or DateInterval) d2 lies entirely within d1; "
          "lists in either argument thread like Outer")
def _datewithinq(a, b):
    def f(s1, s2, x, y):
        di = parse_interval(x)
        if di is not None:
            return any(p[0] <= s2[0] and s2[1] <= p[1] for p in di.utc_parts())
        return s1[0] <= s2[0] and s2[1] <= s1[1]
    return _outer2(f, a, b, "DateWithinQ")


@fn("DateOverlapsQ", **ADDED,
    notes="DateOverlapsQ[d1, d2] is True if the periods of two DateObjects (or DateIntervals) intersect; instants "
          "overlap a period they lie in; lists thread like Outer")
def _dateoverlapsq(a, b):
    def f(s1, s2, x, y):
        p1 = parse_interval(x).utc_parts() if parse_interval(x) is not None else [s1]
        p2 = parse_interval(y).utc_parts() if parse_interval(y) is not None else [s2]
        for a0, a1 in p1:
            for b0, b1 in p2:
                if a0 == a1 and b0 == b1:
                    if a0 == b0:
                        return True
                elif a0 == a1:
                    if b0 <= a0 < b1:
                        return True
                elif b0 == b1:
                    if a0 <= b0 < a1:
                        return True
                elif a0 < b1 and b0 < a1:
                    return True
        return False
    return _outer2(f, a, b, "DateOverlapsQ")


# ─────────────────────────── DateDifference ───────────────────────────

_UNIT_SECS = {"Week": 604800, "Day": 86400, "Hour": 3600, "Minute": 60, "Second": 1}
_UNIT_PLURAL = {"Year": "Years", "Quarter": "Quarters", "Month": "Months", "Week": "Weeks", "Day": "Days",
                "Hour": "Hours", "Minute": "Minutes", "Second": "Seconds"}
_UNIT_ALIASES = {k: k for k in _UNIT_PLURAL}
_UNIT_ALIASES.update({v: k for k, v in _UNIT_PLURAL.items()})
_CONVENTIONS = {"ActualActual", "Thirty360US", "Thirty360", "ThirtyE360", "ThirtyE360ISDA", "ThirtyEPlus360",
                "Actual365", "Actual360", "Actual365L", "ActualActualISDA", "ActualActualICMA"}


def _quantity_value(v):
    if isinstance(v, Fraction):
        if v.denominator == 1:
            return int(v)
        return float(v)
    if isinstance(v, float) and v == int(v) and abs(v) < 2 ** 53:
        return v
    return v


def _add_months(dl, k, cal):
    """Calendar addition of k months to a full date list (day clamped to the month's length: RollBackward)."""
    y, m = dl[0], dl[1]
    ya, mm = _norm_ym(y, m + k)
    y2 = _wl_year(ya)
    d = min(dl[2], _days_in_month(y2, mm, cal))
    return [y2, mm, d] + list(dl[3:])


def _cal_steps(w1, w2, months, cal):
    """(whole steps, fractional remainder) of a calendar unit of `months` months from wall time w1 to w2 >= w1."""
    dl1 = _abs_to_list(w1, cal)
    k = 0
    # jump close to the answer, then step
    approx = (_abs_to_list(w2, cal)[0] - dl1[0]) * 12 + (_abs_to_list(w2, cal)[1] - dl1[1])
    k = max(0, approx // months - 1)
    while _list_to_abs(_add_months(dl1, (k + 1) * months, cal), cal) <= w2:
        k += 1
    a = _list_to_abs(_add_months(dl1, k * months, cal), cal)
    b = _list_to_abs(_add_months(dl1, (k + 1) * months, cal), cal)
    frac = (w2 - a) / (b - a) if not (_is_exact(w2) and _is_exact(a)) else Fraction(w2 - a, b - a)
    return k, frac, a


def _thirty360(d1, d2, conv):
    y1, m1, dd1 = d1[:3]
    y2, m2, dd2 = d2[:3]
    last1 = dd1 == _days_in_month(y1, m1)
    last2 = dd2 == _days_in_month(y2, m2)
    if conv == "Thirty360US":
        if m1 == 2 and last1:
            dd1 = 30
        if m2 == 2 and last2:
            dd2 = 30
        if dd2 == 31 and dd1 >= 30:
            dd2 = 30
        if dd1 == 31:
            dd1 = 30
    elif conv == "Thirty360":
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31 and dd1 >= 30:
            dd2 = 30
    elif conv == "ThirtyE360":
        dd1, dd2 = min(dd1, 30), min(dd2, 30)
    elif conv == "ThirtyE360ISDA":
        if last1:
            dd1 = 30
        if last2 and not (m2 == 2):
            dd2 = 30
        elif m2 == 2 and last2:
            dd2 = 30
    elif conv == "ThirtyEPlus360":
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31:
            ya, mm = _norm_ym(y2, m2 + 1)
            y2, m2, dd2 = _wl_year(ya), mm, 1
    return 360 * (_astro(y2) - _astro(y1)) + 30 * (m2 - m1) + (dd2 - dd1)


def _convention_diff(w1, w2, unit, conv):
    d1, d2 = _abs_to_list(w1), _abs_to_list(w2)
    if conv.startswith("Thirty"):
        days = _thirty360(d1, d2, conv)
        per = {"Day": 1, "Week": 7, "Month": 30, "Quarter": 90, "Year": 360}
        if unit not in per:
            raise ValueError(f"DateDifference: unit {unit!r} cannot be used with DayCountConvention -> {conv!r}")
        return Fraction(days, per[unit])
    days = Fraction(w2 - w1, 86400) if _is_exact(w1) and _is_exact(w2) else (w2 - w1) / 86400
    if unit == "Day":
        return days
    if unit == "Week":
        return days / 7
    if unit != "Year":
        raise NotImplementedError(f"DateDifference: unit {unit!r} is not supported with DayCountConvention -> {conv!r}")
    if conv == "Actual365":
        return days / 365
    if conv == "Actual360":
        return days / 360
    if conv == "ActualActualISDA":
        total = Fraction(0)
        a = w1
        while a < w2:
            y = _abs_to_list(a)[0]
            ny = _list_to_abs([_wl_year(_astro(y) + 1)])
            b = min(ny, w2)
            total += Fraction(b - a, 86400) / (366 if _leap(y) else 365) if _is_exact(a) and _is_exact(b) \
                else (b - a) / 86400 / (366 if _leap(y) else 365)
            a = b
        return total
    raise NotImplementedError(f"DateDifference: DayCountConvention -> {conv!r} is not supported")


def _diff_one(w1, w2, unit, cal):
    """Difference in one unit (fractional) from w1 to w2 (any order)."""
    if unit in _UNIT_SECS:
        if _is_exact(w1) and _is_exact(w2):
            return Fraction(w2 - w1, _UNIT_SECS[unit])
        return (w2 - w1) / _UNIT_SECS[unit]
    months = {"Month": 1, "Quarter": 3, "Year": 12}[unit]
    if w2 < w1:
        k, f, _ = _cal_steps(w2, w1, months, cal)
        return -(k + f)
    k, f, _ = _cal_steps(w1, w2, months, cal)
    return k + f


def _diff_mixed(w1, w2, units, cal):
    sign = 1
    if w2 < w1:
        w1, w2, sign = w2, w1, -1
    out = []
    a = w1
    for i, u in enumerate(units):
        if i == len(units) - 1:
            out.append(sign * _diff_one(a, w2, u, cal))
            break
        if u in _UNIT_SECS:
            n = int((w2 - a) // _UNIT_SECS[u])
            a = a + n * _UNIT_SECS[u]
        else:
            months = {"Month": 1, "Quarter": 3, "Year": 12}[u]
            n, _, a = _cal_steps(a, w2, months, cal)
        out.append(sign * n)
    return out


def _unit(u, fname):
    if isinstance(u, str) and u in _UNIT_ALIASES:
        return _UNIT_ALIASES[u]
    raise ValueError(f"{fname}: {u!r} is not a date unit (Year, Quarter, Month, Week, Day, Hour, Minute, Second)")


def _walls_in_common_zone(a, b, fname):
    d1 = to_date(a, fname) if not (K.is_number(a) and not isinstance(a, bool)) else \
        _instant_from_wall(_num(a, fname, "absolute time"), current_tz())
    d2 = to_date(b, fname) if not (K.is_number(b) and not isinstance(b, bool)) else \
        _instant_from_wall(_num(b, fname, "absolute time"), current_tz())
    w1 = d1.wall() if d1.gran == "Instant" else d1.bounds_wall()[0]
    w2 = d2.wall() if d2.gran == "Instant" else d2.bounds_wall()[0]
    if d1.tz is not None or d2.tz is not None:
        z = d1.zone()
        w2 = d2.to_utc(w2) + _off_secs(_tz_offset(z, utc=d2.to_utc(w2)))
    return w1, w2, d1.cal


@fn("DateDifference", **ADDED,
    notes="DateDifference[d1, d2] = Quantity[n, \"Days\"]; DateDifference[d1, d2, unit] for Year, Quarter, Month "
          "(calendar steps plus the fraction of the next step), Week, Day, Hour, Minute, Second; a unit list gives "
          "Quantity[MixedMagnitude[...], MixedUnit[...]]; DateDifference[d2] from now; DayCountConvention "
          "(ActualActual, Thirty360, Thirty360US, ThirtyE360, ThirtyE360ISDA, ThirtyEPlus360, Actual365, Actual360, "
          "ActualActualISDA; Actual365L and ActualActualICMA raise); integer results are exact, others machine "
          "reals (as Wolfram displays them)")
def _datedifference(*args):
    args, o = _opts(args, {"DayCountConvention", "Method"}, "DateDifference")
    if len(args) == 1:
        z = current_tz()
        args = [_instant_from_wall(_now_wall(z), z).expr(), args[0]]
    if len(args) not in (2, 3):
        raise ValueError("DateDifference: expected DateDifference[d1, d2] or DateDifference[d1, d2, unit]")
    conv = o.get("DayCountConvention", "ActualActual")
    if conv not in _CONVENTIONS:
        raise ValueError(f"DateDifference: {conv!r} is not a DayCountConvention")
    meth = o.get("Method", AUTOMATIC)
    if not (is_sym(meth, "Automatic") or meth in ("RollBackward",)):
        raise NotImplementedError(f"DateDifference: Method -> {meth!r} is not supported (RollBackward only)")
    w1, w2, cal = _walls_in_common_zone(args[0], args[1], "DateDifference")
    spec = args[2] if len(args) == 3 else "Day"
    if is_list(spec):
        units = [_unit(u, "DateDifference") for u in spec.args]
        if not units:
            raise ValueError("DateDifference: empty unit list")
        if conv != "ActualActual":
            raise NotImplementedError("DateDifference: mixed units need DayCountConvention -> \"ActualActual\"")
        vals = _diff_mixed(w1, w2, units, cal)
        mags = [_out(_quantity_value(v)) for v in vals]
        if len(units) == 1:
            return Expr("Quantity", mags[0], _UNIT_PLURAL[units[0]])
        return Expr("Quantity", Expr("MixedMagnitude", L(*mags)),
                    Expr("MixedUnit", L(*(_UNIT_PLURAL[u] for u in units))))
    u = _unit(spec, "DateDifference")
    v = _diff_one(w1, w2, u, cal) if conv == "ActualActual" else _convention_diff(min(w1, w2), max(w1, w2), u, conv) \
        * (1 if w2 >= w1 else -1)
    return Expr("Quantity", _out(_quantity_value(v)), _UNIT_PLURAL[u])


# ─────────────────────────── time systems, Julian dates ───────────────────────────

# (UTC date at which TAI - UTC became n seconds, n)
_LEAPS = [((1972, 1, 1), 10), ((1972, 7, 1), 11), ((1973, 1, 1), 12), ((1974, 1, 1), 13), ((1975, 1, 1), 14),
          ((1976, 1, 1), 15), ((1977, 1, 1), 16), ((1978, 1, 1), 17), ((1979, 1, 1), 18), ((1980, 1, 1), 19),
          ((1981, 7, 1), 20), ((1982, 7, 1), 21), ((1983, 7, 1), 22), ((1985, 7, 1), 23), ((1988, 1, 1), 24),
          ((1990, 1, 1), 25), ((1991, 1, 1), 26), ((1992, 7, 1), 27), ((1993, 7, 1), 28), ((1994, 7, 1), 29),
          ((1996, 1, 1), 30), ((1997, 7, 1), 31), ((1999, 1, 1), 32), ((2006, 1, 1), 33), ((2009, 1, 1), 34),
          ((2012, 7, 1), 35), ((2015, 7, 1), 36), ((2017, 1, 1), 37)]
_LEAP_SECS = [(_list_to_abs(list(d)), n) for d, n in _LEAPS]
_TT_TAI = Fraction(32184, 1000)
_JD_EPOCH = Fraction(4830041, 2)        # Julian date of 1900-01-01 00:00 = 2415020.5


def _dat(u):
    """TAI - UTC at UTC seconds u (since 1900)."""
    if u < _LEAP_SECS[0][0]:
        raise NotImplementedError("TimeSystemConvert: TAI - UTC before 1972 (the rubber-second era) is not supported")
    n = 10
    for b, k in _LEAP_SECS:
        if u >= b:
            n = k
    return n


def _smear(u):
    """TAI - UT for the smeared UTC time scale ("UT"): a leap second is spread over the preceding UTC day."""
    base = _dat(u)
    for b, k in _LEAP_SECS[1:]:
        if b - 86400 <= u < b:
            frac = Fraction(u - (b - 86400), 86400) if _is_exact(u) else (u - (b - 86400)) / 86400
            return base + frac * (k - base)
    return base


def _tdb_minus_tt(tt):
    jd = _JD_EPOCH + Fraction(tt) / 86400 if _is_exact(tt) else float(_JD_EPOCH) + tt / 86400
    g = math.radians(357.53 + 0.9856003 * (float(jd) - 2451545.0))
    return 0.001657 * math.sin(g) + 0.000014 * math.sin(2 * g)


_LG, _LB, _T0JD, _TDB0 = 6.969290134e-10, 1.550519768e-8, 2443144.5003725, -6.55e-5


def _to_tai(wall, z, tsys, utc_day_secs=None):
    """TAI seconds for a wall time (seconds since 1900 in zone z) of time system tsys."""
    off = _off_secs(_tz_offset(z, wall=wall)) if z is not None else 0
    t = wall - off
    if tsys in (None, "UT", "SmearedUTC"):
        return t + _smear(t)
    if tsys == "UTC":
        day0 = utc_day_secs if utc_day_secs is not None else t
        return t + _dat(day0)
    if tsys == "TAI":
        return t
    if tsys == "GPS":
        return t + 19
    if tsys == "TT":
        return t - _TT_TAI
    if tsys == "TDB":
        return t - _tdb_minus_tt(t) - _TT_TAI
    if tsys == "TCG":
        jd = float(_JD_EPOCH) + float(t) / 86400
        return t - _LG * (jd - _T0JD) * 86400 / (1 + _LG) - _TT_TAI
    if tsys == "TCB":
        jd = float(_JD_EPOCH) + float(t) / 86400
        tdb = t - _LB * (jd - _T0JD) * 86400 + _TDB0
        return tdb - _tdb_minus_tt(tdb) - _TT_TAI
    if tsys == "UT1":
        raise NotImplementedError("TimeSystemConvert: UT1 needs IERS Earth-orientation data, which is not available")
    raise ValueError(f"TimeSystemConvert: unknown time system {tsys!r}")


def _from_tai(tai, z, tsys, cal="Gregorian"):
    """Date list for TAI seconds in time system tsys and zone z (UTC lists may show second 60)."""
    if tsys in (None, "UT", "SmearedUTC"):
        u = tai - _smear(tai - _dat(tai - 37 if tai > _LEAP_SECS[-1][0] else tai - 10))
        for _ in range(3):
            u = tai - _smear(u)
        t = u
    elif tsys == "UTC":
        for b, k in _LEAP_SECS[1:]:
            if b + k - 1 <= tai < b + k:
                w = b - 1 + _off_secs(_tz_offset(z, utc=b - 1)) if z is not None else b - 1
                dl = _abs_to_list(w, cal)
                dl[5] = dl[5] + 1 + (tai - (b + k - 1))
                return dl
        u = tai - 10
        for _ in range(2):
            u = tai - _dat(u)
        t = u
    elif tsys == "TAI":
        t = tai
    elif tsys == "GPS":
        t = tai - 19
    elif tsys == "TT":
        t = tai + _TT_TAI
    elif tsys == "TDB":
        tt = tai + _TT_TAI
        t = tt + _tdb_minus_tt(tt)
    elif tsys == "TCG":
        tt = tai + _TT_TAI
        jd = float(_JD_EPOCH) + float(tt) / 86400
        t = tt + _LG * (jd - _T0JD) * 86400 / (1 - _LG)
    elif tsys == "TCB":
        tt = tai + _TT_TAI
        tdb = tt + _tdb_minus_tt(tt)
        jd = float(_JD_EPOCH) + float(tdb) / 86400
        t = tdb + _LB * (jd - _T0JD) * 86400 - _TDB0
    elif tsys == "UT1":
        raise NotImplementedError("TimeSystemConvert: UT1 needs IERS Earth-orientation data, which is not available")
    else:
        raise ValueError(f"TimeSystemConvert: unknown time system {tsys!r}")
    w = t + _off_secs(_tz_offset(z, utc=t)) if z is not None else t
    return _abs_to_list(w, cal)


def _date_tai(d):
    w = d.wall() if d.gran == "Instant" else d.bounds_wall()[0]
    day0 = None
    if d.tsys == "UTC":
        full = _pad(d)
        if full[5] >= 60:
            w = _list_to_abs(full[:5] + [59], d.cal) + (full[5] - 59)
        z = d.zone()
        day_wall = _list_to_abs(full[:3], d.cal)
        day0 = day_wall - _off_secs(_tz_offset(z, wall=day_wall))
    return _to_tai(w, d.zone(), d.tsys, day0)


@fn("TimeSystemConvert", **ADDED,
    notes="TimeSystemConvert[date, tsys] and TimeSystemConvert[{d1, ...}, tsys] between \"UT\" (smeared UTC, the "
          "default), \"UTC\" (leap seconds shown as second 60), \"TAI\", \"TT\" (TAI + 32.184 s), \"GPS\", \"TDB\" "
          "(Fairhead-Bretagnon leading terms, ~10 us), \"TCG\" and \"TCB\" (IAU 2006 linear relations); leap seconds "
          "from 1972 to 2017 are tabulated; \"UT1\" (needs IERS data) and dates before 1972 raise")
def _timesystemconvert(x, tsys=None):
    if is_list(x) and parse_dobj(x) is None:
        return L(*(_timesystemconvert(v, tsys) for v in x.args))
    d = parse_dobj(x)
    if d is None:
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise ValueError("TimeSystemConvert: expected a DateObject")
    target = None if tsys is None or is_sym(tsys, "Automatic") else tsys
    if target is not None and target not in _TSYS:
        raise ValueError(f"TimeSystemConvert: {tsys!r} is not a known time system")
    tai = _date_tai(d)
    z = d.zone()
    dl = _from_tai(tai, z, target, d.cal)
    dl[5] = float(dl[5])
    if target in (None, "UT"):
        return D(dl, "Instant", d.cal, z, None, d.fmt, True).expr()
    return Expr("DateObject", L(*(_out(v) for v in dl)), "Instant", d.cal, _tz_out(z), target)


_JD_VARIANTS = {"Full": Fraction(0), "Reduced": Fraction(2400000), "Modified": Fraction(4800001, 2),
                "Truncated": Fraction(4880001, 2), "Dublin": Fraction(2415020)}


def _jd_of(d, tsys=None):
    tai = _date_tai(d)
    if tsys in (None, "UT", "UTC", "SmearedUTC"):
        u = tai - _smear(tai - (37 if tai > _LEAP_SECS[-1][0] else 10)) if tai >= _LEAP_SECS[0][0] + 10 else None
        if u is None:
            w = d.wall() if d.gran == "Instant" else d.bounds_wall()[0]
            u = d.to_utc(w)
        else:
            for _ in range(3):
                u = tai - _smear(u)
    else:
        lst = _from_tai(tai, None, tsys)
        u = _list_to_abs(lst)
    return (_JD_EPOCH + Fraction(u) / 86400) if _is_exact(u) else float(_JD_EPOCH) + u / 86400


def _jd_date(x, fname):
    if K.is_number(x) and not isinstance(x, bool):
        return _instant_from_wall(_num(x, fname, "absolute time"), current_tz())
    if is_list(x) and not (len(x.args) == 2 and isinstance(x.args[0], str)):
        vals, _ = _list_spec(x, fname)
        return _instant_from_wall(_list_to_abs(vals, "Gregorian", fname), current_tz())
    return to_date(x, fname)


@fn("JulianDate", **ADDED,
    notes="days since noon GMT on 24 Nov -4714 (proleptic Gregorian): JulianDate[], JulianDate[date] (DateObject, "
          "date list or string in $TimeZone, absolute time) and the variants \"Full\", \"Reduced\" (JD - 2400000), "
          "\"Modified\" (JD - 2400000.5), \"Truncated\" (JD - 2440000.5), \"Dublin\" (JD - 2415020); exact for exact "
          "input; TimeSystem option (UT/UTC, TAI, TT, TDB, ...)")
def _juliandate(*args):
    args, o = _opts(args, {"TimeSystem"}, "JulianDate")
    variant = "Full"
    if args and isinstance(args[0], str) and args[0] in _JD_VARIANTS:
        variant, args = args[0], args[1:]
    if len(args) > 1:
        raise ValueError("JulianDate: expected JulianDate[date] or JulianDate[\"type\", date]")
    ts = o.get("TimeSystem", AUTOMATIC)
    ts = None if is_sym(ts, "Automatic") else ts
    if ts is not None and ts not in _TSYS:
        raise ValueError(f"JulianDate: {ts!r} is not a known time system")
    if not args:
        z = current_tz()
        d = _instant_from_wall(_now_wall(z), z)
    else:
        d = _jd_date(args[0], "JulianDate")
    jd = _jd_of(d, ts)
    jd = jd - _JD_VARIANTS[variant] if _is_exact(jd) else jd - float(_JD_VARIANTS[variant])
    return float(jd)          # Wolfram gives Julian dates as reals (JulianDate["Dublin", ...] = 44245.)


@fn("FromJulianDate", **ADDED,
    notes="FromJulianDate[jd] and FromJulianDate[\"type\", jd] give the DateObject (\"Instant\", in $TimeZone or the "
          "TimeZone option) of a Julian date (variants Full, Reduced, Modified, Truncated, Dublin); exact for exact "
          "input (FromJulianDate[2457754, TimeZone -> 0] = 2016-12-31 12:00:00); options CalendarType, DateFormat, "
          "DateGranularity, TimeSystem, TimeZone")
def _fromjuliandate(*args):
    args, o = _opts(args, _DATE_OPTS, "FromJulianDate")
    variant = "Full"
    if len(args) == 2:
        if args[0] not in _JD_VARIANTS:
            raise ValueError(f"FromJulianDate: {args[0]!r} is not a Julian date variant")
        variant, args = args[0], args[1:]
    if len(args) != 1:
        raise ValueError("FromJulianDate: expected FromJulianDate[jd] or FromJulianDate[\"type\", jd]")
    jd = _num(args[0], "FromJulianDate", "Julian date")
    off = _JD_VARIANTS[variant]
    jd = jd + off if _is_exact(jd) else jd + float(off)
    u = (jd - _JD_EPOCH) * 86400 if _is_exact(jd) else (jd - float(_JD_EPOCH)) * 86400
    if isinstance(u, Fraction) and u.denominator == 1:
        u = int(u)
    z = _tz(o["TimeZone"], "FromJulianDate", allow_none=False) if "TimeZone" in o else current_tz()
    ts = o.get("TimeSystem", AUTOMATIC)
    ts = None if is_sym(ts, "Automatic") or ts in ("SmearedUTC", "UT") else ts
    if ts is not None and ts not in _TSYS:
        raise ValueError(f"FromJulianDate: {ts!r} is not a known time system")
    if ts is None:
        w = u + _off_secs(_tz_offset(z, utc=u))
        d = _instant_from_wall(w, z, "Gregorian", True)
    else:
        # the Julian date counts days of the given time system
        tai = _to_tai(u, None, "UT" if ts == "UTC" else ts)
        dl = _from_tai(tai, z, ts)
        d = D(dl, "Instant", "Gregorian", z, ts, None, True)
        return _apply_date_opts(d, {k: v for k, v in o.items() if k not in ("TimeZone", "TimeSystem")},
                                "FromJulianDate").expr()
    return _apply_date_opts(d, {k: v for k, v in o.items() if k != "TimeZone"}, "FromJulianDate").expr()


# ─────────────────────────── lunations and sidereal time ───────────────────────────

# Meeus, Astronomical Algorithms (2nd ed.), table 47.A: (D, M, M', F, coefficient of sin in 1e-6 degrees)
_MOON_LON = [
    (0, 0, 1, 0, 6288774), (2, 0, -1, 0, 1274027), (2, 0, 0, 0, 658314), (0, 0, 2, 0, 213618), (0, 1, 0, 0, 
    -185116), (0, 0, 0, 2, -114332), (2, 0, -2, 0, 58793), (2, -1, -1, 0, 57066), (2, 0, 1, 0, 53322), (2, -1, 0, 
    0, 45758), (0, 1, -1, 0, -40923), (1, 0, 0, 0, -34720), (0, 1, 1, 0, -30383), (2, 0, 0, -2, 15327), (0, 0, 1, 
    2, -12528), (0, 0, 1, -2, 10980), (4, 0, -1, 0, 10675), (0, 0, 3, 0, 10034), (4, 0, -2, 0, 8548), (2, 1, -1, 
    0, -7888), (2, 1, 0, 0, -6766), (1, 0, -1, 0, -5163), (1, 1, 0, 0, 4987), (2, -1, 1, 0, 4036), (2, 0, 2, 0, 
    3994), (4, 0, 0, 0, 3861), (2, 0, -3, 0, 3665), (0, 1, -2, 0, -2689), (2, 0, -1, 2, -2602), (2, -1, -2, 0, 
    2390), (1, 0, 1, 0, -2348), (2, -2, 0, 0, 2236), (0, 1, 2, 0, -2120), (0, 2, 0, 0, -2069), (2, -2, -1, 0, 
    2048), (2, 0, 1, -2, -1773), (2, 0, 0, 2, -1595), (4, -1, -1, 0, 1215), (0, 0, 2, 2, -1110), (3, 0, -1, 0, 
    -892), (2, 1, 1, 0, -810), (4, -1, -2, 0, 759), (0, 2, -1, 0, -713), (2, 2, -1, 0, -700), (2, 1, -2, 0, 691), 
    (2, -1, 0, -2, 596), (4, 0, 1, 0, 549), (0, 0, 4, 0, 537), (4, -1, 0, 0, 520), (1, 0, -2, 0, -487), (2, 1, 0, 
    -2, -399), (0, 0, 2, -2, -381), (1, 1, 1, 0, 351), (3, 0, -2, 0, -340), (4, 0, -3, 0, 330), (2, -1, 2, 0, 
    327), (0, 2, 1, 0, -323), (1, 1, -1, 0, 299), (2, 0, 3, 0, 294), (2, 0, -1, -2, 0)
]
_LUNATION_OFFSETS = {"BrownLunationNumber": 953, "GoldstineLunationNumber": 37105, "HebrewLunationNumber": 71234,
                     "IslamicLunationNumber": 17038, "ThaiLunationNumber": 16843}
_SYNODIC = 29.530588861
_LN0_JD = 2451550.09766


def _delta_t(year):
    """TT - UT in seconds (Espenak & Meeus polynomial approximations)."""
    y = year
    if 2005 <= y < 2050:
        t = y - 2000
        return 62.92 + 0.32217 * t + 0.005589 * t * t
    if 1986 <= y < 2005:
        t = y - 2000
        return 63.86 + 0.3345 * t - 0.060374 * t ** 2 + 0.0017275 * t ** 3 + 0.000651814 * t ** 4 + \
            0.00002373599 * t ** 5
    if 1961 <= y < 1986:
        t = y - 1975
        return 45.45 + 1.067 * t - t ** 2 / 260 - t ** 3 / 718
    if 1941 <= y < 1961:
        t = y - 1950
        return 29.07 + 0.407 * t - t ** 2 / 233 + t ** 3 / 2547
    if 1920 <= y < 1941:
        t = y - 1920
        return 21.20 + 0.84493 * t - 0.076100 * t ** 2 + 0.0020936 * t ** 3
    if 1900 <= y < 1920:
        t = y - 1900
        return -2.79 + 1.494119 * t - 0.0598939 * t ** 2 + 0.0061966 * t ** 3 - 0.000197 * t ** 4
    if 2050 <= y < 2150:
        return -20 + 32 * ((y - 1820) / 100) ** 2 - 0.5628 * (2150 - y)
    u = (y - 1820) / 100
    return -20 + 32 * u * u


def _elongation(jd_ut):
    """Apparent geocentric elongation of the Moon from the Sun (degrees, 0..360) at a UT Julian date."""
    year = 2000 + (jd_ut - 2451545.0) / 365.25
    jde = jd_ut + _delta_t(year) / 86400.0
    T = (jde - 2451545.0) / 36525.0
    Lp = 218.3164477 + 481267.88123421 * T - 0.0015786 * T ** 2 + T ** 3 / 538841 - T ** 4 / 65194000
    D = 297.8501921 + 445267.1114034 * T - 0.0018819 * T ** 2 + T ** 3 / 545868 - T ** 4 / 113065000
    M = 357.5291092 + 35999.0502909 * T - 0.0001536 * T ** 2 + T ** 3 / 24490000
    Mp = 134.9633964 + 477198.8675055 * T + 0.0087414 * T ** 2 + T ** 3 / 69699 - T ** 4 / 14712000
    F = 93.2720950 + 483202.0175233 * T - 0.0036539 * T ** 2 - T ** 3 / 3526000 + T ** 4 / 863310000
    A1 = 119.75 + 131.849 * T
    A2 = 53.09 + 479264.290 * T
    E = 1 - 0.002516 * T - 0.0000074 * T ** 2
    r = math.radians
    sl = 0.0
    for d, m, mp, f, c in _MOON_LON:
        term = c * math.sin(r(d * D + m * M + mp * Mp + f * F))
        if m:
            term *= E ** abs(m)
        sl += term
    sl += 3958 * math.sin(r(A1)) + 1962 * math.sin(r(Lp - F)) + 318 * math.sin(r(A2))
    moon = Lp + sl / 1e6
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T ** 2
    C = (1.914602 - 0.004817 * T - 0.000014 * T ** 2) * math.sin(r(M)) + (0.019993 - 0.000101 * T) * \
        math.sin(r(2 * M)) + 0.000289 * math.sin(r(3 * M))
    sun = L0 + C - 0.00569                       # aberration; nutation cancels in the difference
    return (moon - sun) % 360.0


def _lunation_at(jd):
    e = _elongation(jd) / 360.0
    k = (jd - _LN0_JD) / _SYNODIC
    return round(k - e) + e


def _scheme(s, fname):
    if s is None or is_sym(s, "Automatic"):
        return 0
    if s in _LUNATION_OFFSETS:
        return _LUNATION_OFFSETS[s]
    raise ValueError(f"{fname}: {s!r} is not a lunation counting scheme")


def _jd_ut(d):
    w = d.wall() if d.gran == "Instant" else d.bounds_wall()[0]
    u = d.to_utc(w)
    return float(_JD_EPOCH) + float(u) / 86400.0


@fn("LunationNumber", **ADDED,
    notes="lunations since the new moon of 6 Jan 2000 (fractional part = Moon-Sun elongation / 360: .5 full moon); "
          "LunationNumber[], LunationNumber[date], LunationNumber[scheme, date] for BrownLunationNumber (+953), "
          "GoldstineLunationNumber (+37105), HebrewLunationNumber (+71234), IslamicLunationNumber (+17038), "
          "ThaiLunationNumber (+16843); Moon from Meeus ch. 47 (60 terms), Sun from Meeus ch. 25, TT - UT from the "
          "Espenak-Meeus fits: about 1e-5 lunation (~30 s) accuracy")
def _lunationnumber(*args):
    scheme = None
    if len(args) == 2:
        scheme, args = args[0], args[1:]
    if len(args) > 1:
        raise ValueError("LunationNumber: expected LunationNumber[date] or LunationNumber[scheme, date]")
    off = _scheme(scheme, "LunationNumber")
    if not args:
        z = current_tz()
        d = _instant_from_wall(_now_wall(z), z)
    else:
        d = _jd_date(args[0], "LunationNumber")
    return _lunation_at(_jd_ut(d)) + off


@fn("FromLunationNumber", **ADDED,
    notes="FromLunationNumber[ln] and FromLunationNumber[scheme, ln]: the instant (DateObject in  or the "
          "TimeZone option) with the given lunation number, found by Newton iteration on LunationNumber; "
          "FromLunationNumber[0] is the new moon of 2000-01-06 18:14 UTC; options CalendarType, DateFormat, "
          "DateGranularity, TimeSystem, TimeZone")
def _fromlunationnumber(*args):
    args, o = _opts(args, _DATE_OPTS, "FromLunationNumber")
    scheme = None
    if len(args) == 2:
        scheme, args = args[0], args[1:]
    if len(args) != 1:
        raise ValueError("FromLunationNumber: expected FromLunationNumber[ln] or FromLunationNumber[scheme, ln]")
    ln = float(_num(args[0], "FromLunationNumber", "lunation number")) - _scheme(scheme, "FromLunationNumber")
    jd = _LN0_JD + _SYNODIC * ln
    for _ in range(30):
        cur = _lunation_at(jd)
        step = (ln - cur) * _SYNODIC
        jd += step
        if abs(step) < 1e-9:
            break
    u = (jd - float(_JD_EPOCH)) * 86400.0
    z = _tz(o["TimeZone"], "FromLunationNumber", allow_none=False) if "TimeZone" in o else current_tz()
    d = _instant_from_wall(u + _off_secs(_tz_offset(z, utc=u)), z, "Gregorian", True)
    return _apply_date_opts(d, {k: v for k, v in o.items() if k != "TimeZone"}, "FromLunationNumber").expr()


def _location(x, fname):
    """(latitude, longitude) in degrees from {lat, lon} or GeoPosition[{lat, lon, ...}]."""
    if isinstance(x, Expr) and x.head == "GeoPosition" and x.args:
        x = x.args[0]
    if is_list(x) and len(x.args) in (2, 3) and all(K.is_number(v) for v in x.args):
        lat, lon = float(_num(x.args[0], fname, "latitude")), float(_num(x.args[1], fname, "longitude"))
        if not -90 <= lat <= 90:
            raise ValueError(f"{fname}: latitude {lat} is out of range")
        return lat, lon
    return None


def _sidereal_hours(jd, lon, apparent):
    T = (jd - 2451545.0) / 36525.0
    gmst = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T * T - T ** 3 / 38710000.0
    if apparent:
        r = math.radians
        om = 125.04452 - 1934.136261 * T
        Ls = 280.4665 + 36000.7698 * T
        Lm = 218.3165 + 481267.8813 * T
        dpsi = (-17.20 * math.sin(r(om)) - 1.32 * math.sin(r(2 * Ls)) - 0.23 * math.sin(r(2 * Lm))
                + 0.21 * math.sin(r(2 * om))) / 3600.0
        deps = (9.20 * math.cos(r(om)) + 0.57 * math.cos(r(2 * Ls)) + 0.10 * math.cos(r(2 * Lm))
                - 0.09 * math.cos(r(2 * om))) / 3600.0
        eps = 23.4392911111 - 0.0130041667 * T - 1.6389e-7 * T * T + 5.0361e-7 * T ** 3 + deps
        gmst += dpsi * math.cos(r(eps))
    return ((gmst + lon) % 360.0) / 15.0


def _hms_quantity(h):
    hh = int(h)
    mm = int((h - hh) * 60)
    ss = (h - hh - mm / 60.0) * 3600.0
    if ss >= 60 - 1e-9:
        ss, mm = 0.0, mm + 1
    if mm >= 60:
        mm, hh = 0, (hh + 1) % 24
    return Expr("Quantity", Expr("MixedMagnitude", L(hh, mm, ss)),
                Expr("MixedUnit", L("HoursOfRightAscension", "MinutesOfRightAscension", "SecondsOfRightAscension")))


@fn("SiderealTime", **ADDED,
    notes="SiderealTime[loc, date] (apparent: mean sidereal time plus the equation of the equinoxes) and "
          "SiderealTime[\"MeanTime\" | \"ApparentTime\", loc, date] for loc = {lat, lon} or GeoPosition[{lat, lon}] "
          "(east longitude positive) and a DateObject, date list or string; SiderealTime[{{loc1, date1}, ...}]; "
          "result Quantity[MixedMagnitude[{h, m, s}], MixedUnit[{\"HoursOfRightAscension\", ...}]] (IAU 1982 GMST, "
          "UT1 taken as UTC: ~1 s accuracy); forms needing  (SiderealTime[], SiderealTime[date]) raise")
def _siderealtime(*args):
    kind = "ApparentTime"
    if args and args[0] in ("MeanTime", "ApparentTime"):
        kind, args = args[0], args[1:]
    if len(args) == 1 and is_list(args[0]) and args[0].args and all(
            is_list(p) and len(p.args) == 2 and _location(p.args[0], "SiderealTime") for p in args[0].args):
        return L(*(_siderealtime(kind, p.args[0], p.args[1]) for p in args[0].args))
    if len(args) == 2:
        loc = _location(args[0], "SiderealTime")
        if loc is None:
            if isinstance(args[0], Expr) and args[0].head in ("Entity", "GeoPosition"):
                raise NotImplementedError("SiderealTime: entity locations need curated data; give {lat, lon}")
            raise ValueError("SiderealTime: the location must be {lat, lon} or GeoPosition[{lat, lon}]")
        if is_list(args[1]) and args[1].args and all(parse_dobj(v) is not None for v in args[1].args):
            return L(*(_siderealtime(kind, args[0], v) for v in args[1].args))
        d = _jd_date(args[1], "SiderealTime")
        return _hms_quantity(_sidereal_hours(_jd_ut(d), loc[1], kind == "ApparentTime"))
    if len(args) <= 1:
        raise NotImplementedError("SiderealTime: the location defaults to $GeoLocation, which is not available; "
                                  "use SiderealTime[{lat, lon}, date]")
    raise ValueError("SiderealTime: expected SiderealTime[loc, date] or SiderealTime[type, loc, date]")


# ─────────────────────────── random dates and times ───────────────────────────

def _rng():
    from .statistics import _RNG
    return _RNG


def _uniform(a, b):
    r = _rng()
    return a + (b - a) * r.random()


def _count_spec(n, fname):
    if K.is_int(n) and int(n) >= 0:
        return [int(n)]
    if is_list(n) and n.args and all(K.is_int(v) and int(v) >= 0 for v in n.args):
        return [int(v) for v in n.args]
    raise ValueError(f"{fname}: the number of values must be a non-negative integer or a list of them")


def _array(dims, gen):
    if not dims:
        return gen()
    return L(*(_array(dims[1:], gen) for _ in range(dims[0])))


_QUNITS = {"Seconds": 1, "Minutes": 60, "Hours": 3600, "Days": 86400, "Weeks": 604800}


def _quantity_secs(q, fname):
    if isinstance(q, Expr) and q.head == "Quantity" and len(q.args) == 2 and isinstance(q.args[1], str):
        u = q.args[1]
        v = float(_num(q.args[0], fname, "quantity"))
        if u in _QUNITS:
            return v * _QUNITS[u], None
        if u in ("Months", "Years"):
            return v, u
        raise ValueError(f"{fname}: {u!r} is not a time unit")
    return None


@fn("RandomDate", **ADDED,
    notes="RandomDate[] (in the current year), RandomDate[{dmin, dmax}] (uniform between the start of dmin and the "
          "end of dmax), RandomDate[gdate] (within a granular DateObject or DateInterval), RandomDate[Quantity[q, "
          "unit]] and RandomDate[{q1, q2}] (relative to now), RandomDate[range, n | {n1, n2, ...}]; options "
          "DateGranularity (default \"Instant\"), TimeZone, CalendarType, DateFormat; uses the session generator "
          "(SeedRandom)")
def _randomdate(*args):
    args, o = _opts(args, _DATE_OPTS, "RandomDate")
    if len(args) > 2:
        raise ValueError("RandomDate: expected RandomDate[range] or RandomDate[range, n]")
    dims = _count_spec(args[1], "RandomDate") if len(args) == 2 else []
    z = _tz(o["TimeZone"], "RandomDate", allow_none=False) if "TimeZone" in o else current_tz()
    lo, hi = _random_range(args[0] if args else None, z, "RandomDate")
    gran = o.get("DateGranularity", AUTOMATIC)
    gran = "Instant" if is_sym(gran, "Automatic") else _gran(gran, "RandomDate")
    rest = {k: v for k, v in o.items() if k not in ("TimeZone", "DateGranularity")}

    def gen():
        u = _uniform(float(lo), float(hi))
        w = u + _off_secs(_tz_offset(z, utc=u))
        d = _from_wall(w, gran, "Gregorian", None if gran in _FLOATING else z, None, None, False)
        return _apply_date_opts(d, rest, "RandomDate").expr()
    return _array(dims, gen)


def _random_range(spec, z, fname):
    """(utc low, utc high) for a RandomDate/RandomTime range specification."""
    if spec is None:
        now = _now_wall(z)
        y = _abs_to_list(now)[0]
        d = D([y], "Year", tz=z, tz_explicit=True)
        return d.bounds_utc()
    q = _quantity_secs(spec, fname)
    if q is not None:
        now = _now_utc()
        secs, cal_unit = q
        if cal_unit is not None:
            w = _now_wall(z)
            months = int(round(secs * (12 if cal_unit == "Years" else 1)))
            other = _list_to_abs(_add_months(_abs_to_list(w), months, "Gregorian"))
            other_u = other - _off_secs(_tz_offset(z, wall=other))
            return (min(now, other_u), max(now, other_u))
        return (min(now, now + Fraction(secs)), max(now, now + Fraction(secs)))
    if is_list(spec) and len(spec.args) == 2 and parse_dobj(spec) is None:
        qa, qb = _quantity_secs(spec.args[0], fname), _quantity_secs(spec.args[1], fname)
        if qa is not None and qb is not None:
            if qa[1] or qb[1]:
                raise NotImplementedError(f"{fname}: month and year offsets are only supported alone")
            now = _now_utc()
            return now + Fraction(qa[0]), now + Fraction(qb[0])
        a = _utc_span(spec.args[0], fname) if parse_tobj(spec.args[0]) is None else _time_span(spec.args[0], z)
        b = _utc_span(spec.args[1], fname) if parse_tobj(spec.args[1]) is None else _time_span(spec.args[1], z)
        if b[1] < a[0]:
            raise ValueError(f"{fname}: the end of the range precedes its start")
        return a[0], max(b[1], b[0])
    if parse_tobj(spec) is not None:
        return _time_span(spec, z)
    a = _utc_span(spec, fname)
    if a[0] == a[1]:
        raise ValueError(f"{fname}: an instant is not a range of dates")
    return a


def _time_span(t, z):
    tt = parse_tobj(t)
    s, e = tt.bounds()
    w = _now_wall(tt.tz if tt.tz is not None else z)
    day = _list_to_abs(_abs_to_list(w)[:3])
    base = day - _list_to_abs([1900, 1, 1])
    zz = tt.tz if tt.tz is not None else z
    off = _off_secs(_tz_offset(zz, wall=base + s))
    return base + s - off, base + (e if e > s else s) - off


@fn("RandomTime", **ADDED,
    notes="RandomTime[] (a time of day), RandomTime[{tmin, tmax}], RandomTime[time] (within a granular TimeObject "
          "or DateObject), RandomTime[Quantity[...]] relative to now, RandomTime[range, n | {n1, ...}]; options "
          "DateGranularity (default \"Instant\") and TimeZone; gives TimeObjects")
def _randomtime(*args):
    args, o = _opts(args, {"DateGranularity", "TimeZone"}, "RandomTime")
    if len(args) > 2:
        raise ValueError("RandomTime: expected RandomTime[range] or RandomTime[range, n]")
    dims = _count_spec(args[1], "RandomTime") if len(args) == 2 else []
    z = _tz(o["TimeZone"], "RandomTime", allow_none=False) if "TimeZone" in o else current_tz()
    gran = o.get("DateGranularity", AUTOMATIC)
    gran = "Instant" if is_sym(gran, "Automatic") else _tgran(gran, "RandomTime")
    if args:
        lo, hi = _random_range(args[0], z, "RandomTime")
    else:
        w = _now_wall(z)
        day = _list_to_abs(_abs_to_list(w)[:3])
        lo = day - _off_secs(_tz_offset(z, wall=day))
        hi = lo + 86400

    def gen():
        u = _uniform(float(lo), float(hi))
        w = u + _off_secs(_tz_offset(z, utc=u))
        t = _time_from_secs(w - _list_to_abs(_abs_to_list(w)[:3]), gran, z if "TimeZone" in o else None)
        return t.expr()
    return _array(dims, gen)


# ─────────────────────────── Dated, Clock, DateScale, DatePattern ───────────────────────────

@fn("Dated", **ADDED,
    notes="Dated[obj, year], Dated[obj, date] and Dated[obj, All]: an inert wrapper associating obj (an entity, "
          "entity property or property name) with a date; the date must be an integer year, a DateObject or "
          "DateInterval, Interval of dates or All (resolving entity data needs curated data and is not available)")
def _dated(obj, date):
    if K.is_int(date) or is_sym(date, "All") or parse_dobj(date) is not None or parse_interval(date) is not None:
        raise Unevaluated()
    if isinstance(date, Expr) and date.head == "Interval":
        raise Unevaluated()
    if isinstance(date, Symbol) or (isinstance(date, Expr) and not is_list(date)):
        raise Unevaluated()
    raise ValueError(f"Dated: {date!r} is not a year, date or All")


@fn("Clock", **ADDED,
    notes="Clock[...] outside Dynamic evaluates to its starting value (documented 'Possible Issues'): Clock[] -> 0, "
          "Clock[vmax] -> 0, Clock[{vmin, vmax}] and Clock[{vmin, vmax, dv}] -> vmin, with the optional cycle time "
          "t and count n validated; Clock[Infinity] -> 0")
def _clock(spec=None, t=None, n=None):
    if t is not None:
        tv = _num(t, "Clock", "cycle time")
        if tv <= 0:
            raise ValueError("Clock: the cycle time must be positive")
    if n is not None and not (K.is_int(n) and int(n) >= 0) and not is_sym(n, "Infinity"):
        raise ValueError("Clock: the number of cycles must be a non-negative integer")
    if spec is None:
        return 0
    if is_sym(spec, "Infinity") or (K.is_number(spec) and not isinstance(spec, bool)):
        if not is_sym(spec, "Infinity"):
            _num(spec, "Clock", "maximum value")
        return 0
    if is_list(spec) and len(spec.args) in (2, 3):
        vmin = spec.args[0]
        _num(vmin, "Clock", "minimum value")
        if not is_sym(spec.args[1], "Infinity"):
            _num(spec.args[1], "Clock", "maximum value")
        if len(spec.args) == 3 and _num(spec.args[2], "Clock", "step") <= 0:
            raise ValueError("Clock: the step must be positive")
        return vmin
    if isinstance(spec, Symbol):
        raise Unevaluated()
    raise ValueError("Clock: expected Clock[], Clock[vmax], Clock[{vmin, vmax}] or Clock[{vmin, vmax, dv}]")


@fn("DateScale", **ADDED,
    notes="DateScale[] and DateScale[DateFunction -> f, DateTicksFormat -> fmt]: the inert date scale for "
          "ScalingFunctions; options are validated (unknown options raise)")
def _datescale(*args):
    for a in args:
        if not (is_rule(a) and isinstance(a.args[0], Symbol) and a.args[0].name in ("DateFunction",
                                                                                    "DateTicksFormat")):
            raise ValueError("DateScale: only the options DateFunction and DateTicksFormat can be given")
    raise Unevaluated()


_DP_RX = {"Year": r"\d{4}", "Quarter": "[1-4]", "Month": r"(?:1[0-2]|0?[1-9])",
          "MonthName": "(?:" + "|".join(sorted([m for m in _MONTHS] + [m[:3] for m in _MONTHS], key=len,
                                                reverse=True)) + ")",
          "Day": r"(?:3[01]|[12]\d|0?[1-9])",
          "DayName": "(?:" + "|".join(sorted(_DAYS + [d[:3] for d in _DAYS], key=len, reverse=True)) + ")",
          "Hour": r"(?:2[0-3]|[01]?\d)", "AMPM": "(?:[AaPp][Mm])", "Minute": r"[0-5]\d", "Second": r"[0-5]\d"}


def _datepattern_rx(e, compile_):
    if not (1 <= len(e.args) <= 2 and is_list(e.args[0]) and e.args[0].args):
        raise ValueError("DatePattern: expected DatePattern[{\"e1\", \"e2\", ...}] or DatePattern[{...}, sep]")
    sep = r"[/\-:.]" if len(e.args) == 1 else "(?:" + compile_(e.args[1]) + ")"
    out = []
    prev_elem = False
    for x in e.args[0].args:
        if not isinstance(x, str):
            raise ValueError("DatePattern: date elements must be strings")
        if x in _DP_RX:
            if prev_elem:
                out.append(sep)
            out.append(_DP_RX[x])
            prev_elem = True
        else:
            out.append(re.escape(x))
            prev_elem = False
    return "(?:" + "".join(out) + ")"


def _hook_string_patterns():
    from . import wl_strings as WS
    prev = WS._compile

    def comp(p, ctx, lazy=False):
        if isinstance(p, Expr) and p.head == "DatePattern":
            return _datepattern_rx(p, lambda q: comp(q, ctx, lazy))
        return prev(p, ctx, lazy)
    WS._compile = comp


_hook_string_patterns()


@fn("DatePattern", **ADDED,
    notes="DatePattern[{\"e1\", ...}] and DatePattern[{...}, sep] as string patterns (StringMatchQ, StringCases, "
          "StringReplace, ...): elements Year (4 digits), Quarter, Month, MonthName (full or 3-letter), Day, "
          "DayName, Hour, AMPM, Minute, Second; consecutive elements are separated by one of / - : . (or by sep), "
          "literal strings in the list are matched exactly; inert as an expression")
def _datepattern(*args):
    _datepattern_rx(Expr("DatePattern", *args), lambda q: re.escape(q) if isinstance(q, str) else "(?:.+?)")
    raise Unevaluated()


# ─────────────────────────── option symbols ───────────────────────────

for _n, _note in [
        ("DateFormat", "option of DateObject, DateInterval, FromJulianDate, ... giving the display format: a named "
                       "format, an element list or Automatic; DateString of the date uses it"),
        ("DateFunction", "option of TimeSeries, DateListPlot, DateScale, ... converting inputs to dates; an inert "
                         "option name validated by DateScale"),
        ("DateGranularity", "option of DateObject, RandomDate, RandomTime, TimeObject, FromJulianDate, ... giving the "
                            "calendar granularity of generated dates (Automatic or a granularity)"),
        ("DateReduction", "option of DateHistogram giving a cyclic period (None, \"Week\", \"Year\", ...); an inert "
                          "option name"),
        ("TimeFormat", "option giving the TextString format of a TimeObject (Automatic = 24-hour); an inert option "
                       "name"),
        ("DayCountConvention", "option of DateDifference: ActualActual (default), Thirty360, Thirty360US, ThirtyE360, "
                               "ThirtyE360ISDA, ThirtyEPlus360, Actual365, Actual360, ActualActualISDA"),
        ("ExpirationDate", "option of PersistentSymbol and InitializationValue (when a persistent value expires); an "
                           "inert option name"),
        ("PartitionGranularity", "option of audio analysis functions (partition length, offset, window); an inert "
                                 "option name"),
        ("MaxExtraBandwidths", "option of SmoothKernelDistribution (how far the estimate extends beyond the data); "
                               "an inert option name"),
        ("MaxMixtureKernels", "option of SmoothKernelDistribution and related functions (number and placement of "
                              "kernels); an inert option name"),
        ("AllowedDimensions", "option of Grid and Tabular restricting the displayed number of rows and columns; an "
                              "inert option name"),
        ("ValueDimensions", "option of TemporalData giving the dimension of the value space; an inert option name"),
        ("DatasetTheme", "option of Dataset giving its visual theme; stored by Dataset, otherwise inert"),
        ("IncludeRelatedTables", "option of RelationalDatabase: include tables referenced by foreign keys "
                                 "(default False)")]:
    _added(_n, _note)


# ─────────────────────────── shared wrappers for existing functions ───────────────────────────

_PASS = object()


def wrap_registry(name, pre):
    """Let ``pre(args)`` handle a call of the library function ``name`` first (return _PASS to fall through)."""
    e = REGISTRY.get(name)
    if e is None:
        return
    old = list(e.impls)

    def call(*a):
        r = pre(a)
        if r is not _PASS:
            return r
        n = len(a)
        for f_, lo, hi in old:
            if n >= lo and (hi is None or n <= hi):
                return f_(*a)
        raise ValueError(f"{name} called with {n} argument(s)")
    e.impls[:] = [(call, 0, None)]


def wrap_builtin(name, pre):
    """Same for a kernel builtin (signature (args, ev, expr))."""
    prev = K._BUILTINS.get(name)
    if prev is None:
        return

    def call(a, ev_, expr):
        r = pre(tuple(a))
        if r is not _PASS:
            return r
        return prev(a, ev_, expr)
    K._BUILTINS[name] = call


# ─────────────────────────── categorical values: Nominal, Ordinal, scales ───────────────────────────

class Cat:
    """A categorical scale (kind 'Nominal' or 'Ordinal') with categories, labels and scores."""

    def __init__(self, kind, cats, labels=None, scores=None):
        self.kind = kind
        self.cats = list(cats)
        self.labels = list(labels) if labels is not None else list(cats)
        self.scores = list(scores) if scores is not None else list(range(1, len(cats) + 1))
        if not (len(self.cats) == len(self.labels) == len(self.scores)):
            raise ValueError(f"{kind}: the category, label and score lists must have the same length")
        if len(set(map(repr, self.cats))) != len(self.cats):
            raise ValueError(f"{kind}: the categories must be distinct")
        for s in self.scores:
            if not (K.is_number(s) and not isinstance(s, bool) and not isinstance(s, complex)):
                raise ValueError(f"{kind}: scores must be real numbers")

    def assoc(self):
        return assoc(("CategoryList", L(*self.cats)), ("LabelList", L(*self.labels)),
                     ("ScoreList", L(*self.scores)))

    def expr(self, i=None):
        return Expr(self.kind, self.assoc()) if i is None else Expr(self.kind, self.assoc(), i + 1)

    def index(self, c):
        for i, x in enumerate(self.cats):
            if x == c:
                return i
        raise ValueError(f"{self.kind}: {c!r} is not one of the categories {L(*self.cats)!r}")


def _assoc_get(a, k):
    for r in a.args:
        if r.args[0] == k:
            return r.args[1]
    return None


def parse_scale(x, kind, fname=None):
    """A Cat from a category list, association, NominalScale/OrdinalScale or a Nominal/Ordinal expression."""
    fname = fname or kind
    if is_list(x):
        return Cat(kind, x.args)
    if is_assoc(x):
        cl = _assoc_get(x, "CategoryList")
        if cl is None or not is_list(cl):
            raise ValueError(f"{fname}: a scale association needs a \"CategoryList\" list")
        ll, sl = _assoc_get(x, "LabelList"), _assoc_get(x, "ScoreList")
        if ll is not None and not is_list(ll) or sl is not None and not is_list(sl):
            raise ValueError(f"{fname}: \"LabelList\" and \"ScoreList\" must be lists")
        return Cat(kind, cl.args, ll.args if ll is not None else None, sl.args if sl is not None else None)
    if isinstance(x, Expr) and x.head in ("NominalScale", "OrdinalScale"):
        s = _scale_from_scaleexpr(x)
        return Cat(kind, s.cats, s.labels, s.scores)
    if isinstance(x, Expr) and x.head in ("Nominal", "Ordinal") and len(x.args) >= 1:
        s = parse_scale(x.args[0], x.head, fname)
        return Cat(kind, s.cats, s.labels, s.scores)
    raise ValueError(f"{fname}: {x!r} is not a list of categories or a scale specification")


def _category(x):
    """(Cat, index) for a canonical single category Nominal[assoc, i] / Ordinal[assoc, i], else None."""
    if isinstance(x, Expr) and x.head in ("Nominal", "Ordinal") and len(x.args) == 2 and is_assoc(x.args[0]) \
            and K.is_int(x.args[1]):
        try:
            s = parse_scale(x.args[0], x.head)
        except ValueError:
            return None
        i = int(x.args[1]) - 1
        if 0 <= i < len(s.cats):
            return s, i
    return None


def _scale_of(x):
    if isinstance(x, Expr) and x.head in ("Nominal", "Ordinal") and len(x.args) == 1 and is_assoc(x.args[0]):
        try:
            return parse_scale(x.args[0], x.head)
        except ValueError:
            return None
    return None


def _make_categorical(kind):
    def impl(scale, *c):
        if len(c) > 1:
            raise ValueError(f"{kind}: expected {kind}[scale] or {kind}[scale, category]")
        if not c and isinstance(scale, Expr) and scale.head == kind and len(scale.args) == 1 \
                and is_assoc(scale.args[0]) and _scale_of(scale) is not None:
            if scale.args[0] == _scale_of(scale).assoc():
                raise Unevaluated()
        if c and K.is_int(c[0]) and is_assoc(scale) and _category(Expr(kind, scale, c[0])) is not None:
            if scale == parse_scale(scale, kind).assoc():
                raise Unevaluated()
        s = parse_scale(scale, kind)
        return s.expr(s.index(c[0])) if c else s.expr()
    return impl


_CATNOTE = ("{kind}[{{c1, ...}}] gives the scale {kind}[<|\"CategoryList\" -> ..., \"LabelList\" -> ..., \"ScoreList\" "
            "-> {{1, 2, ...}}|>] and {kind}[scale, c] (or scale[c]) the category {kind}[<|...|>, i]; scales may be "
            "lists, associations, {kind}Scale or {kind} expressions; Normal gives the underlying expression(s); "
            "CategoricalValue reads properties")
fn("Nominal", **ADDED, notes=_CATNOTE.format(kind="Nominal") + "; Sort orders Nominal categories canonically by "
   "their expressions; Min/Max/Less stay unevaluated")(_make_categorical("Nominal"))
fn("Ordinal", **ADDED, notes=_CATNOTE.format(kind="Ordinal") + "; Sort, Min, Max, Less, Greater, LessEqual and "
   "GreaterEqual use the \"Score\" values")(_make_categorical("Ordinal"))


def _cat_hook(head, args, ev_):
    s = _scale_of(head)
    if s is None or len(args) != 1:
        return K.UNHANDLED
    return s.expr(s.index(args[0]))


KL.COMPOUND_HEAD_HOOKS["Nominal"] = _cat_hook
KL.COMPOUND_HEAD_HOOKS["Ordinal"] = _cat_hook

_CAT_PROPS = ("Expression", "Category", "Index", "Label", "Scale", "Score")


def _catvalue(obj, prop="Expression"):
    if prop not in _CAT_PROPS:
        raise ValueError(f"CategoricalValue: {prop!r} is not a property; use one of {', '.join(_CAT_PROPS)}")
    if is_list(obj):
        return L(*(_catvalue(o, prop) for o in obj.args))
    c = _category(obj)
    if c is not None:
        s, i = c
        return {"Expression": s.cats[i], "Category": s.expr(i), "Index": i + 1, "Label": s.labels[i],
                "Scale": s.expr(), "Score": s.scores[i]}[prop]
    s = _scale_of(obj)
    if s is None and isinstance(obj, Expr) and obj.head in ("NominalScale", "OrdinalScale"):
        sc = _scale_from_scaleexpr(obj)
        s = Cat("Ordinal" if obj.head == "OrdinalScale" else "Nominal", sc.cats, sc.labels, sc.scores)
        if prop == "Scale":
            return obj
    if s is not None:
        n = range(len(s.cats))
        return {"Expression": L(*s.cats), "Category": L(*(s.expr(i) for i in n)), "Index": L(*(i + 1 for i in n)),
                "Label": L(*s.labels), "Scale": s.expr(), "Score": L(*s.scores)}[prop]
    raise ValueError(f"CategoricalValue: {obj!r} is not a Nominal or Ordinal category or scale")


fn("CategoricalValue", **ADDED,
   notes="CategoricalValue[obj] and CategoricalValue[obj, prop] for Nominal/Ordinal categories and scales and "
         "NominalScale/OrdinalScale (threads over lists): \"Expression\" (default), \"Category\", \"Index\", "
         "\"Label\", \"Scale\", \"Score\"")(_catvalue)


class _Scale:
    def __init__(self, cats, labels, scores):
        self.cats, self.labels, self.scores = cats, labels, scores


def _labels_from(spec, cats, fname):
    if spec is None or is_sym(spec, "Automatic"):
        return list(cats)
    if is_list(spec):
        if len(spec.args) != len(cats):
            raise ValueError(f"{fname}: the label list must have one label per category")
        return list(spec.args)
    if is_assoc(spec):
        labs = list(cats)
        for r in spec.args:
            k = r.args[0]
            if k not in cats:
                raise ValueError(f"{fname}: {k!r} is not a category")
            labs[cats.index(k)] = r.args[1]
        return labs
    raise ValueError(f"{fname}: labels must be a list or an association")


def _scale_from_scaleexpr(x):
    fname, a = x.head, list(x.args)
    if fname == "NominalScale":
        if not a or len(a) > 2:
            raise ValueError("NominalScale: expected NominalScale[cats] or NominalScale[cats, labels]")
        if is_sym(a[0], "Automatic"):
            raise ValueError("NominalScale[Automatic] has no categories until it is used in a plot")
        if is_assoc(a[0]):
            if len(a) > 1:
                raise ValueError("NominalScale: labels are given twice")
            cats = [r.args[0] for r in a[0].args]
            return _Scale(cats, [r.args[1] for r in a[0].args], list(range(1, len(cats) + 1)))
        if not is_list(a[0]):
            raise ValueError("NominalScale: the categories must be a list")
        cats = list(a[0].args)
        return _Scale(cats, _labels_from(a[1] if len(a) > 1 else None, cats, fname), list(range(1, len(cats) + 1)))
    if not a or len(a) > 3:
        raise ValueError("OrdinalScale: expected OrdinalScale[cats], OrdinalScale[cats, vals] or "
                         "OrdinalScale[cats, vals, labels]")
    if is_sym(a[0], "Automatic"):
        raise ValueError("OrdinalScale[Automatic] has no categories until it is used in a plot")
    if is_assoc(a[0]):
        cats = [r.args[0] for r in a[0].args]
        vals = [r.args[1] for r in a[0].args]
        labs = _labels_from(a[1] if len(a) > 1 else None, cats, fname)
        if len(a) > 2:
            raise ValueError("OrdinalScale: too many arguments after an association")
    else:
        if not is_list(a[0]):
            raise ValueError("OrdinalScale: the categories must be a list")
        cats = list(a[0].args)
        v = a[1] if len(a) > 1 else AUTOMATIC
        if is_sym(v, "Automatic"):
            vals = list(range(1, len(cats) + 1))
        elif is_list(v) and len(v.args) == len(cats):
            vals = list(v.args)
        else:
            raise ValueError("OrdinalScale: the values must be Automatic or a list with one value per category")
        labs = _labels_from(a[2] if len(a) > 2 else None, cats, fname)
    for v in vals:
        if not (K.is_number(v) and not isinstance(v, (bool, complex))):
            raise ValueError("OrdinalScale: category values must be real numbers")
    return _Scale(cats, labs, vals)


def _scale_validate(*a, head):
    _scale_from_scaleexpr(Expr(head, *a))
    raise Unevaluated()


fn("NominalScale", **ADDED,
   notes="NominalScale[{c...}], NominalScale[<|c -> lab, ...|>], NominalScale[cats, labels]: inert scale for "
         "ScalingFunctions and Nominal, validated; scale[prop] and scale[prop, arg] give \"CategoryList\", "
         "\"LabelAssociation\", \"LabelList\", \"Label\", \"Labels\", \"Properties\"")(
    lambda *a: _scale_validate(*a, head="NominalScale"))
fn("OrdinalScale", **ADDED,
   notes="OrdinalScale[{c...}], OrdinalScale[cats, vals], OrdinalScale[<|c -> v|>], OrdinalScale[..., labels]: "
         "inert, validated; scale[prop] / scale[prop, arg] give \"CategoryAssociation\" (<|v -> c|>), "
         "\"CategoryList\", \"LabelAssociation\", \"LabelList\", \"ValueAssociation\", \"ValueList\", \"Category\", "
         "\"Categories\", \"Label\", \"Labels\", \"Value\", \"Values\", \"Properties\"")(
    lambda *a: _scale_validate(*a, head="OrdinalScale"))

_NS_PROPS = ["CategoryList", "LabelAssociation", "LabelList", "Label", "Labels", "Properties"]
_OS_PROPS = ["CategoryAssociation", "CategoryList", "LabelAssociation", "LabelList", "ValueAssociation", "ValueList",
             "Category", "Categories", "Label", "Labels", "Value", "Values", "Properties"]


def _scale_hook(head, args, ev_):
    if not isinstance(head, Expr) or not args or not isinstance(args[0], str):
        return K.UNHANDLED
    s = _scale_from_scaleexpr(head)
    props = _NS_PROPS if head.head == "NominalScale" else _OS_PROPS
    p = args[0]
    if p not in props:
        raise ValueError(f"{head.head}: {p!r} is not a property; use one of {', '.join(props)}")
    if len(args) > 2:
        raise ValueError(f"{head.head}: expected scale[prop] or scale[prop, arg]")

    def lookup(keys, vals, x, what):
        for k, v in zip(keys, vals):
            if k == x:
                return v
        raise ValueError(f"{head.head}: {x!r} is not a {what} of the scale")

    plain = {"CategoryList": lambda: L(*s.cats), "LabelList": lambda: L(*s.labels),
             "ValueList": lambda: L(*s.scores), "Properties": lambda: L(*props),
             "LabelAssociation": lambda: assoc(*zip(s.cats, s.labels)),
             "ValueAssociation": lambda: assoc(*zip(s.cats, s.scores)),
             "CategoryAssociation": lambda: assoc(*zip(s.scores, s.cats))}
    one = {"Category": lambda x: lookup(s.scores, s.cats, x, "value"),
           "Label": lambda x: lookup(s.cats, s.labels, x, "category"),
           "Value": lambda x: lookup(s.cats, s.scores, x, "category")}
    many = {"Categories": "Category", "Labels": "Label", "Values": "Value"}
    if p in plain:
        if len(args) != 1:
            raise ValueError(f"{head.head}: the property {p!r} takes no argument")
        return plain[p]()
    if len(args) == 1:
        return Expr("Function", Expr(head, p, Expr("Slot", 1)))
    x = args[1]
    if p in many:
        if not is_list(x):
            raise ValueError(f"{head.head}: the property {p!r} needs a list")
        return L(*(one[many[p]](v) for v in x.args))
    return one[p](x)


KL.COMPOUND_HEAD_HOOKS["NominalScale"] = _scale_hook
KL.COMPOUND_HEAD_HOOKS["OrdinalScale"] = _scale_hook


def _ord_scores(args):
    """Scores of Ordinal categories of one common scale, or None."""
    cs = [_category(a) for a in args]
    if not cs or any(c is None or c[0].kind != "Ordinal" for c in cs):
        return None
    if any(c[0].assoc() != cs[0][0].assoc() for c in cs):
        raise ValueError("Ordinal categories from different scales cannot be compared")
    return [float(_num(c[0].scores[c[1]], "Ordinal", "score")) for c in cs]


def _cat_sort(a):
    if len(a) != 1 or not is_list(a[0]) or not a[0].args:
        return _PASS
    items = a[0].args
    cs = [_category(x) for x in items]
    if any(c is None for c in cs):
        return _PASS
    if all(c[0].kind == "Ordinal" for c in cs):
        sc = _ord_scores(items)
        order = sorted(range(len(items)), key=lambda i: (sc[i], cs[i][1]))
    else:
        order = sorted(range(len(items)), key=lambda i: K.order_key(cs[i][0].cats[cs[i][1]]))
    return L(*(items[i] for i in order))


def _flat_args(a):
    out = []
    for x in a:
        if is_list(x):
            out.extend(_flat_args(x.args))
        else:
            out.append(x)
    return out


def _cat_extreme(pick):
    def pre(a):
        xs = _flat_args(a)
        sc = _ord_scores(xs) if xs else None
        if sc is None:
            if xs and all(_category(x) is not None for x in xs):
                raise Unevaluated()          # Nominal categories have no order
            return _PASS
        return xs[pick(range(len(xs)), key=lambda i: sc[i])]
    return pre


def _cat_compare(op):
    def pre(a):
        sc = _ord_scores(a) if len(a) >= 2 else None
        if sc is None:
            return _PASS
        return all(op(x, y) for x, y in zip(sc, sc[1:]))
    return pre


def _cat_normal(a):
    if len(a) == 1:
        c = _category(a[0])
        if c is not None:
            return c[0].cats[c[1]]
        s = _scale_of(a[0])
        if s is not None:
            return L(*s.cats)
    return _PASS


wrap_builtin("Sort", _cat_sort)
wrap_registry("Min", _cat_extreme(min))
wrap_registry("Max", _cat_extreme(max))
wrap_builtin("Less", _cat_compare(lambda x, y: x < y))
wrap_builtin("Greater", _cat_compare(lambda x, y: x > y))
wrap_builtin("LessEqual", _cat_compare(lambda x, y: x <= y))
wrap_builtin("GreaterEqual", _cat_compare(lambda x, y: x >= y))
wrap_builtin("Normal", _cat_normal)


def _cat_randomchoice(a):
    if a and isinstance(a[0], Expr) and a[0].head in ("Nominal", "Ordinal"):
        s = _scale_of(a[0])
        if s is not None:
            from .core import dispatch
            idx = dispatch("RandomChoice", (L(*range(len(s.cats))),) + tuple(a[1:]))
            if K.is_int(idx):
                return s.expr(int(idx))
            return Expr("Map", Expr("Function", s.expr(Expr("Slot", 1))), idx) if False else \
                _map_idx(s, idx)
    return _PASS


def _map_idx(s, x):
    if is_list(x):
        return L(*(_map_idx(s, v) for v in x.args))
    return s.expr(int(x))


wrap_registry("RandomChoice", _cat_randomchoice)


# ─────────────────────────── Cast ───────────────────────────

import struct as _struct  # noqa: E402

_INT_TYPES = {f"{u}Integer{b}": (b, u == "") for u in ("", "Unsigned") for b in (8, 16, 32, 64)}
_REAL_TYPES = {"Real32": 32, "Real64": 64}
_CPLX_TYPES = {"ComplexReal32": 64, "ComplexReal64": 128}


def _type_size(t):
    if t in _INT_TYPES:
        return _INT_TYPES[t][0]
    if t in _REAL_TYPES:
        return _REAL_TYPES[t]
    if t in _CPLX_TYPES:
        return _CPLX_TYPES[t]
    if t == "Boolean":
        return 8
    return None


def _wrap_int(v, bits, signed):
    v %= 1 << bits
    if signed and v >= 1 << (bits - 1):
        v -= 1 << bits
    return v


def _cast_scalar(v, t, method):
    if t in ("InertExpression", "Expression"):
        return v
    if t == "String":
        if isinstance(v, str):
            return v
        raise ValueError(f"Cast: cannot convert {v!r} to \"String\"")
    if t == "Boolean":
        if isinstance(v, bool):
            return v
        raise ValueError(f"Cast: cannot convert {v!r} to \"Boolean\"")
    if t not in _INT_TYPES and t not in _REAL_TYPES and t not in _CPLX_TYPES:
        raise ValueError(f"Cast: {t!r} is not a supported type")
    if isinstance(v, bool) or not K.is_number(v):
        raise ValueError(f"Cast: {v!r} is not a number and cannot be converted to {t!r}")
    if t in _INT_TYPES:
        bits, signed = _INT_TYPES[t]
        if K.is_int(v):
            iv = int(v)
        elif method == "CCast" and not isinstance(v, complex):
            iv = int(float(_num(v, "Cast", "value")))       # C truncation
        else:
            raise ValueError(f"Cast: {v!r} cannot be converted to the integer type {t!r}")
        if method == "CCast":
            return _wrap_int(iv, bits, signed)
        lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        if not lo <= iv <= hi:
            raise ValueError(f"Cast: {iv} is out of the range of {t!r}")
        return iv
    if t in _REAL_TYPES:
        if isinstance(v, complex):
            raise ValueError(f"Cast: the complex number {v!r} cannot be converted to {t!r}")
        f = float(_num(v, "Cast", "value"))
        if t == "Real32":
            f = _struct.unpack("<f", _struct.pack("<f", f))[0]
        return f
    c = complex(v) if isinstance(v, complex) else complex(float(_num(v, "Cast", "value")))
    return c


_PACK = {"Integer8": "b", "Integer16": "h", "Integer32": "i", "Integer64": "q", "UnsignedInteger8": "B",
         "UnsignedInteger16": "H", "UnsignedInteger32": "I", "UnsignedInteger64": "Q", "Real32": "f", "Real64": "d"}


def _bitcast(v, src, t):
    if src not in _PACK or t not in _PACK:
        raise ValueError("Cast: \"BitCast\" supports the fixed-size integer and real types")
    if _type_size(src) != _type_size(t):
        raise ValueError(f"Cast: \"BitCast\" needs types of the same size ({src} has {_type_size(src)} bits, "
                         f"{t} has {_type_size(t)})")
    return _struct.unpack("<" + _PACK[t], _struct.pack("<" + _PACK[src], v))[0]


def _type_name(t):
    if isinstance(t, str):
        return t, None
    # TypeSpecifier["PackedArray"][elem, rank]  (the parsed form of "PackedArray"::[elem, rank])
    if isinstance(t, Expr) and isinstance(t.head, Expr) and t.head.head == "TypeSpecifier" \
            and list(t.head.args) == ["PackedArray"]:
        if len(t.args) != 2 or not isinstance(t.args[0], str) or not K.is_int(t.args[1]) or int(t.args[1]) < 1:
            raise ValueError("Cast: a packed array type is TypeSpecifier[\"PackedArray\"][elem, rank]")
        return "PackedArray", (t.args[0], int(t.args[1]))
    raise ValueError(f"Cast: {t!r} is not a type specification")


def _cast_array(v, elem, rank, method):
    if rank == 0:
        if is_list(v):
            raise ValueError("Cast: the array is deeper than the packed array rank")
        return _cast_scalar(v, elem, method)
    if not is_list(v):
        raise ValueError(f"Cast: {v!r} cannot be converted to a packed array")
    out = [_cast_array(x, elem, rank - 1, method) for x in v.args]
    if rank > 1 and len({len(x.args) for x in out}) > 1:
        raise ValueError("Cast: a packed array must be rectangular")
    return L(*out)


@fn("Cast", **ADDED,
    notes="Cast[val, type] and Cast[val, type, method] performed directly (there is no compiler, so this is what "
          "compiled code would compute): \"ConversionCast\" (default; checks ranges and refuses lossy real -> "
          "integer conversions), \"CCast\" (C semantics: wraps integers, truncates reals; numeric types only), "
          "\"BitCast\" (reinterprets the bits between same-size integer/real types; the source type is inferred "
          "as Integer64 / Real64, or give {val, srcType}); types Integer8-64, UnsignedInteger8-64, Real32, Real64, "
          "ComplexReal32/64, Boolean, String, InertExpression and TypeSpecifier[\"PackedArray\"][elem, rank]")
def _cast(val, typ, method="ConversionCast"):
    if method not in ("ConversionCast", "CCast", "BitCast"):
        raise ValueError(f"Cast: {method!r} is not a casting method (ConversionCast, CCast or BitCast)")
    name, pa = _type_name(typ)
    if pa is not None:
        if method == "BitCast":
            raise ValueError("Cast: \"BitCast\" cannot produce a packed array")
        return _cast_array(val, pa[0], pa[1], method)
    if method == "CCast" and _type_size(name) is None:
        raise ValueError("Cast: \"CCast\" only supports numeric types")
    if method == "BitCast":
        if is_list(val) and len(val.args) == 2 and isinstance(val.args[1], str):
            v, src = val.args
        else:
            v = val
            src = "Integer64" if K.is_int(val) else "Real64"
        v = _cast_scalar(v, src, "ConversionCast")
        return _bitcast(v, src, name)
    if is_list(val):
        raise ValueError("Cast: use a \"PackedArray\" type to convert a list")
    return _cast_scalar(val, name, method)
