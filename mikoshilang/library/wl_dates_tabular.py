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
