"""Wolfram batch W25: time series, event series and temporal data (numeric timestamps).

``TimeSeries[{{t1, v1}, ...}]`` (sorted, duplicates merged) is the canonical form of a time series; it is inert and
answers ``ts[t]`` (interpolation, Wolfram's default: first order for numeric values, zero order otherwise, the
nearest end value outside the time range) and ``ts["prop"]``.  ``EventSeries`` has the same form without
interpolation.  ``TemporalData[{path1, path2, ...}, opts]`` holds the ensembles produced by RandomFunction.

Deviations from Wolfram 15: timestamps must be real numbers (DateObject timestamps raise NotImplementedError);
"Values" gives the list of values (Wolfram 15 gives a TabularColumn); component keys, calendars, time zones and
MetaInformation are not supported (they raise NotImplementedError); arithmetic on time series (ts^2, ts1 + ts2) is not
threaded (use TimeSeriesMap / TimeSeriesThread).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).
"""
from __future__ import annotations

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, R, S, is_list, has_float, numeric_out, rule
from .wl_graphics_color import inert_option

AREA = "time series (W25)"
AUTO = Symbol("Automatic")
TS_OPTS = {"ResamplingMethod", "TemporalRegularity", "MergingFunction", "InterpolationOrder",
           "MissingDataMethod", "TimeObservationWindow", "ComponentKeys", "MetaInformation", "CalendarType",
           "DateFunction", "DateGranularity", "HolidayCalendar", "TimeSystem", "TimeZone", "ValueDimensions"}
KEPT = ("ResamplingMethod", "TemporalRegularity")


def _ev(e):
    from ..evaluate import evaluate
    return evaluate(e)


def call(f, *args):
    """evaluate f[args] for a function given as a symbol, pure function or other head"""
    return _ev(Expr(f.name if isinstance(f, Symbol) else f, *args))


def _is_sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def _is_rule(x):
    return isinstance(x, Expr) and x.head in ("Rule", "RuleDelayed") and len(x.args) == 2


def _missing(x):
    return isinstance(x, Expr) and x.head == "Missing"


def _time(t, fname):
    if isinstance(t, bool) or isinstance(t, str):
        raise TypeError(f"{fname}: timestamps must be real numbers, not {t!r}")
    if isinstance(t, Expr) and t.head == "DateObject":
        raise NotImplementedError(f"{fname}: DateObject timestamps are not supported (use numeric times)")
    if isinstance(t, (int, float)):
        return t
    v = S(t)
    if not isinstance(v, sp.Basic) or v.free_symbols:
        raise Unevaluated()
    if not (v.is_number and v.is_real):
        raise ValueError(f"{fname}: timestamps must be real numbers, not {t}")
    return t


def _key(t):
    return float(sp.N(S(t))) if not isinstance(t, (int, float)) else t


def _split_opts(args, fname):
    args = list(args)
    opts = {}
    while args and _is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol):
        n = args[-1].args[0].name
        if n not in TS_OPTS:
            raise ValueError(f"{fname}: unknown option {n}")
        opts[n] = args.pop().args[1]
    for n in opts:
        if n not in KEPT + ("MergingFunction", "InterpolationOrder"):
            raise NotImplementedError(f"{fname}: the option {n} is not supported")
    return args, opts


def _check_resampling(m, fname):
    """validate a ResamplingMethod setting; returns the interpolation order (0 or 1) or None for Automatic"""
    if _is_sym(m, "Automatic"):
        return None
    if m in ("Interpolation", "LinearInterpolation"):
        return 1
    if m == "Constant":
        return 0
    if is_list(m) and m.args and m.args[0] == "Interpolation":
        order = 1
        for o in m.args[1:]:
            if _is_rule(o) and _is_sym(o.args[0], "InterpolationOrder"):
                order = o.args[1]
            else:
                raise NotImplementedError(f"{fname}: ResamplingMethod sub-option {o} is not supported")
        if order not in (0, 1):
            raise NotImplementedError(f"{fname}: only InterpolationOrder 0 or 1 is supported")
        return order
    if _is_sym(m, "None"):
        return None
    raise ValueError(f"{fname}: invalid ResamplingMethod {m}")


def _merge(pairs, mf, fname):
    """sort by time, merge duplicate timestamps with the merging function (default Mean)"""
    groups = {}
    order = []
    for t, v in pairs:
        k = _key(t)
        if k not in groups:
            groups[k] = (t, [])
            order.append(k)
        groups[k][1].append(v)
    out = []
    for k in sorted(order):
        t, vs = groups[k]
        if len(vs) == 1:
            out.append((t, vs[0]))
        else:
            f = Symbol("Mean") if mf is None or _is_sym(mf, "Automatic") else mf
            if isinstance(f, (int, float, str, bool)) or is_list(f):
                raise TypeError(f"{fname}: MergingFunction must be a function, not {f!r}")
            out.append((t, call(f, L(*vs))))
    return out


def _times_from_spec(spec, n, fname):
    if spec is None or _is_sym(spec, "Automatic"):
        return list(range(n))
    if not is_list(spec):
        raise ValueError(f"{fname}: invalid time specification {spec}")
    a = list(spec.args)
    if len(a) == 1 and is_list(a[0]):
        ts = [_time(t, fname) for t in a[0].args]
        if len(ts) != n:
            raise ValueError(f"{fname}: the number of times ({len(ts)}) differs from the number of values ({n})")
        return ts
    if len(a) == 1:
        t0 = _time(a[0], fname)
        return [_ev(Expr("Plus", t0, k)) for k in range(n)]
    if len(a) == 2:
        t0, t1 = _time(a[0], fname), _time(a[1], fname)
        if n == 1:
            return [t0]
        return [_ev(Expr("Plus", t0, Expr("Times", Expr("Plus", t1, Expr("Times", -1, t0)), R(sp.Rational(k, n - 1)))))
                for k in range(n)]
    if len(a) == 3 and _is_sym(a[1], "Automatic"):
        t0, dt = _time(a[0], fname), _time(a[2], fname)
        return [_ev(Expr("Plus", t0, Expr("Times", k, dt))) for k in range(n)]
    raise ValueError(f"{fname}: invalid time specification {spec}")


def pairs_of(x, fname, allow=("TimeSeries", "EventSeries", "TemporalData")):
    """(pairs, opts) of a time series / event series / single-path TemporalData, or of a list of values / pairs"""
    if isinstance(x, Expr) and x.head in allow and x.head in ("TimeSeries", "EventSeries"):
        return [(p.args[0], p.args[1]) for p in x.args[0].args], dict(_split_opts(x.args[1:], fname)[1])
    if isinstance(x, Expr) and x.head == "TemporalData" and "TemporalData" in allow:
        paths, opts = td_paths(x)
        if len(paths) != 1:
            raise ValueError(f"{fname}: expected a single path, got an ensemble of {len(paths)} paths")
        return paths[0], opts
    if is_list(x):
        vals = list(x.args)
        if vals and all(is_list(v) and len(v.args) == 2 for v in vals):
            return [(_time(v.args[0], fname), v.args[1]) for v in vals], {}
        return [(k, v) for k, v in enumerate(vals)], {}
    if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
        raise Unevaluated()
    raise TypeError(f"{fname}: {x!r} is not a time series or list of data")


def make_ts(pairs, opts=None, head="TimeSeries"):
    extra = [rule(Symbol(k), opts[k]) for k in KEPT if opts and k in opts]
    return Expr(head, L(*(L(t, v) for t, v in pairs)), *extra)


def _canonical(e):
    a = e.args
    if not a or not is_list(a[0]):
        return False
    pts = a[0].args
    if not all(is_list(p) and len(p.args) == 2 for p in pts):
        return False
    try:
        ks = [_key(_time(p.args[0], "TimeSeries")) for p in pts]
    except (Unevaluated, TypeError, ValueError, NotImplementedError):
        return False
    if any(k2 <= k1 for k1, k2 in zip(ks, ks[1:])):
        return False
    if not all(_is_rule(o) and isinstance(o.args[0], Symbol) and o.args[0].name in KEPT for o in a[1:]):
        return False
    for o in a[1:]:
        if o.args[0].name == "ResamplingMethod":
            _check_resampling(o.args[1], e.head)
    return True


def _build(head, args, fname):
    args, opts = _split_opts(args, fname)
    if "ResamplingMethod" in opts:
        _check_resampling(opts["ResamplingMethod"], fname)
    if "InterpolationOrder" in opts:
        o = opts.pop("InterpolationOrder")
        if o not in (0, 1):
            raise NotImplementedError(f"{fname}: only InterpolationOrder 0 or 1 is supported")
        opts["ResamplingMethod"] = L("Interpolation", rule(Symbol("InterpolationOrder"), o))
    if "TemporalRegularity" in opts and opts["TemporalRegularity"] not in (True, False) and \
            not _is_sym(opts["TemporalRegularity"], "Automatic"):
        raise ValueError(f"{fname}: TemporalRegularity must be True, False or Automatic")
    mf = opts.pop("MergingFunction", None)
    if not args or len(args) > 2:
        raise ValueError(f"{fname}: expected {fname}[{{{{t1, v1}}, ...}}], {fname}[values, tspec] or {fname}[ts]")
    x = args[0]
    if isinstance(x, Expr) and x.head in ("TimeSeries", "EventSeries", "TemporalData") and len(args) == 1:
        pairs, o2 = pairs_of(x, fname)
        o2.update(opts)
        return make_ts(pairs, o2, head)
    if _is_sym(x, "None"):
        raise NotImplementedError(f"{fname}: a series without values is not supported")
    if isinstance(x, Expr) and x.head == "Association":
        x = L(*(L(r.args[0], r.args[1]) for r in x.args))
    if is_list(x) and x.args and all(_is_rule(r) for r in x.args):
        x = L(*(L(r.args[0], r.args[1]) for r in x.args))
    if not is_list(x):
        if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
            raise Unevaluated()
        raise TypeError(f"{fname}: expected a list of values or time-value pairs, not {x!r}")
    if not x.args:
        raise ValueError(f"{fname}: a series needs at least one value")
    if len(args) == 2:
        times = _times_from_spec(args[1], len(x.args), fname)
        pairs = list(zip(times, x.args))
    elif all(is_list(p) and len(p.args) == 2 for p in x.args):
        pairs = [(_time(p.args[0], fname), p.args[1]) for p in x.args]
    else:
        pairs = list(zip(range(len(x.args)), x.args))
    return make_ts(_merge(pairs, mf, fname), opts, head)


@fn("TimeSeries", area=AREA, status="added",
    notes="TimeSeries[{{t1, v1}, ...}], TimeSeries[{t1 -> v1, ...}] / <|t1 -> v1, ...|>, TimeSeries[values] (times 0, "
          "1, ...), TimeSeries[values, tspec] with tspec Automatic, {tmin}, {tmin, tmax}, {tmin, Automatic, dt} or "
          "{{t1, ...}}, and TimeSeries[ts] (from an event series or single-path TemporalData); options "
          "ResamplingMethod (Automatic, \"Constant\", {\"Interpolation\", InterpolationOrder -> 0 | 1}), "
          "InterpolationOrder (0, 1), MergingFunction (default Mean for duplicate timestamps) and "
          "TemporalRegularity. The canonical form TimeSeries[{{t1, v1}, ...}] (sorted) is inert; ts[t] interpolates "
          "(first order for numeric values, zero order otherwise, the nearest end value outside the range) and "
          "ts[\"prop\"] gives \"Path\", \"NormalPath\", \"Times\", \"Values\" (a list), \"NormalValues\", \"FirstTime\", "
          "\"LastTime\", \"FirstValue\", \"LastValue\", \"PathLength\", \"ValueList\", \"TimeList\", "
          "\"ResamplingMethod\", \"MinimumTimeIncrement\", \"Properties\"; Normal[ts] is the path. Numeric timestamps "
          "only (DateObject raises NotImplementedError); unsupported options raise NotImplementedError. "
          "Check: exact; reference (Wolfram documentation examples) in tests.")
def _timeseries(*args: Expr) -> Expr:
    e = Expr("TimeSeries", *args)
    if _canonical(e):
        raise Unevaluated()
    return _build("TimeSeries", args, "TimeSeries")


@fn("EventSeries", area=AREA, status="added",
    notes="EventSeries with the same argument forms as TimeSeries (canonical EventSeries[{{t1, v1}, ...}]); es[t] "
          "gives the value at a timestamp and Missing[\"NotAvailable\"] between timestamps (no interpolation); the "
          "same properties as TimeSeries. Check: exact; reference in tests.")
def _eventseries(*args: Expr) -> Expr:
    e = Expr("EventSeries", *args)
    if _canonical(e):
        raise Unevaluated()
    return _build("EventSeries", args, "EventSeries")


@fn("EventSeriesQ", area=AREA, status="added",
    notes="True for an EventSeries object, False for anything else (exact). Check: exact; reference in tests.")
def _esq(x: Expr) -> Boolean:
    return isinstance(x, Expr) and x.head == "EventSeries" and _canonical(x)


@fn("TimeSeriesQ", area=AREA, status="added",
    notes="TimeSeriesQ[x]: True if x is a (valid, constructed) TimeSeries object, False otherwise (EventSeries, "
          "TemporalData, lists ...). Check: exact; reference (Wolfram documentation) in tests.")
def _tsq(x: Expr) -> Boolean:
    return isinstance(x, Expr) and x.head == "TimeSeries" and _canonical(x)


# ─────────────────────────── TemporalData (ensembles) ───────────────────────────

def make_td(paths, opts=None):
    extra = [rule(Symbol(k), v) for k, v in (opts or {}).items()]
    return Expr("TemporalData", L(*(L(*(L(t, v) for t, v in p)) for p in paths)), *extra)


def td_paths(x):
    paths = [[(p.args[0], p.args[1]) for p in path.args] for path in x.args[0].args]
    opts = {}
    for o in x.args[1:]:
        if _is_rule(o) and isinstance(o.args[0], Symbol):
            opts[o.args[0].name] = o.args[1]
    return paths, opts


def _numeric_value(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if is_list(v):
        return all(_numeric_value(x) for x in v.args)
    try:
        s = S(v)
        return isinstance(s, sp.Basic) and s.is_number
    except Exception:
        return False


def _order(pairs, opts):
    m = opts.get("ResamplingMethod")
    o = _check_resampling(m, "TimeSeries") if m is not None else None
    if o is None:
        vals = [v for _, v in pairs]
        dims = {len(v.args) if is_list(v) else -1 for v in vals}
        o = 1 if not any(isinstance(v, str) or _is_rule(v) for v in vals) and len(dims) == 1 else 0
    return o


def value_at(pairs, opts, t, fname, interpolate=True):
    tk = _key(_time(t, fname))
    ks = [_key(p[0]) for p in pairs]
    for k, (tt, v) in zip(ks, pairs):
        if k == tk:
            return v
    if not interpolate:
        return Expr("Missing", "NotAvailable")
    if tk <= ks[0]:
        return pairs[0][1]
    if tk >= ks[-1]:
        return pairs[-1][1]
    i = max(j for j, k in enumerate(ks) if k < tk)
    (t1, v1), (t2, v2) = pairs[i], pairs[i + 1]
    if _order(pairs, opts) == 0:
        return v1
    if _missing(v1) or _missing(v2):
        return Expr("Missing", "NotAvailable")
    w = Expr("Times", Expr("Plus", t, Expr("Times", -1, t1)), Expr("Power", Expr("Plus", t2, Expr("Times", -1, t1)), -1))
    return _ev(Expr("Plus", v1, Expr("Times", w, Expr("Plus", v2, Expr("Times", -1, v1)))))


def _min_increment(pairs):
    ks = [_key(p[0]) for p in pairs]
    if len(ks) < 2:
        return None
    d = [_ev(Expr("Plus", b[0], Expr("Times", -1, a[0]))) for a, b in zip(pairs, pairs[1:])]
    return _ev(Expr("Min", L(*d)))


def _regular(pairs, opts):
    tr = opts.get("TemporalRegularity")
    if tr is True or tr is False:
        return tr
    if len(pairs) < 3:
        return True
    d = [S(_ev(Expr("Plus", b[0], Expr("Times", -1, a[0])))) for a, b in zip(pairs, pairs[1:])]
    if any(isinstance(x, sp.Float) for x in d):
        return all(abs(float(x) - float(d[0])) <= 1e-10 * max(1.0, abs(float(d[0]))) for x in d)
    return all(sp.simplify(x - d[0]) == 0 for x in d)


def series_property(head, pairs, opts, prop, rest, fname):
    times = [t for t, _ in pairs]
    vals = [v for _, v in pairs]
    path = L(*(L(t, v) for t, v in pairs))
    table = {
        "Path": lambda: path, "NormalPath": lambda: path, "Times": lambda: L(*times), "Values": lambda: L(*vals),
        "NormalValues": lambda: L(*vals), "FirstTime": lambda: times[0], "LastTime": lambda: times[-1],
        "FirstValue": lambda: vals[0], "LastValue": lambda: vals[-1], "PathLength": lambda: len(pairs),
        "ValueList": lambda: L(L(*vals)), "TimeList": lambda: L(L(*times)), "Paths": lambda: L(path),
        "PathCount": lambda: 1,
        "ResamplingMethod": lambda: opts.get("ResamplingMethod", AUTO),
        "MinimumTimeIncrement": lambda: _min_increment(pairs),
    }
    if prop == "Properties":
        return L(*sorted(table))
    if prop not in table:
        raise ValueError(f"{fname}: unknown property {prop!r}")
    if rest:
        raise NotImplementedError(f"{fname}: property arguments are not supported for {prop!r}")
    r = table[prop]()
    if r is None:
        return Expr("Missing", "NotApplicable")
    return r


def _ts_apply(head, args, ev):
    if not _canonical(head):
        return _UNH
    pairs, opts = pairs_of(head, head.head)
    if not args:
        return _UNH
    a0 = args[0]
    if isinstance(a0, str):
        return series_property(head.head, pairs, opts, a0, args[1:], head.head)
    if len(args) != 1:
        raise ValueError(f"{head.head}: ts[t] takes one time")
    interp = head.head == "TimeSeries"
    try:
        if is_list(a0):
            return L(*(value_at(pairs, opts, t, head.head, interp) for t in a0.args))
        return value_at(pairs, opts, a0, head.head, interp)
    except Unevaluated:
        return _UNH


def _td_apply(head, args, ev):
    if not (head.args and is_list(head.args[0]) and all(is_list(p) for p in head.args[0].args)):
        return _UNH
    paths, opts = td_paths(head)
    if not args:
        return _UNH
    a0 = args[0]
    fname = "TemporalData"
    if isinstance(a0, str):
        if a0 == "PathCount":
            return len(paths)
        if a0 == "Paths":
            return L(*(L(*(L(t, v) for t, v in p)) for p in paths))
        if a0 == "ValueList":
            return L(*(L(*(v for _, v in p)) for p in paths))
        if a0 == "TimeList":
            return L(*(L(*(t for t, _ in p)) for p in paths))
        if a0 == "SliceData":
            if len(args) != 2:
                raise ValueError("TemporalData: td[\"SliceData\", t] needs a time")
            return L(*(value_at(p, opts, args[1], fname) for p in paths))
        if a0 == "Properties":
            return L("Path", "PathCount", "Paths", "SliceData", "TimeList", "Times", "ValueList", "Values")
        if len(paths) != 1:
            if a0 in ("Path", "Values", "Times", "FirstTime", "LastTime", "FirstValue", "LastValue", "PathLength"):
                raise ValueError(f"TemporalData: the property {a0!r} needs a single path; use \"Paths\", "
                                 "\"ValueList\" or \"TimeList\" for an ensemble")
            raise ValueError(f"TemporalData: unknown property {a0!r}")
        return series_property("TemporalData", paths[0], opts, a0, args[1:], fname)
    if len(args) != 1:
        raise ValueError("TemporalData: td[t] takes one time")
    try:
        vals = [value_at(p, opts, a0, fname) for p in paths]
    except Unevaluated:
        return _UNH
    return vals[0] if len(paths) == 1 else L(*vals)


from .. import kernel as _K  # noqa: E402
_UNH = _K.UNHANDLED
KL.COMPOUND_HEAD_HOOKS["TimeSeries"] = _ts_apply
KL.COMPOUND_HEAD_HOOKS["EventSeries"] = _ts_apply
KL.COMPOUND_HEAD_HOOKS["TemporalData"] = _td_apply


# Normal[ts] -> path (chained with the existing Normal)
_prev_normal = _K._BUILTINS.get("Normal")


def _normal_w25(a, ev_, expr):
    if len(a) == 1 and isinstance(a[0], Expr) and a[0].head in ("TimeSeries", "EventSeries") and _canonical(a[0]):
        return a[0].args[0]
    if len(a) == 1 and isinstance(a[0], Expr) and a[0].head == "TemporalData" and a[0].args and is_list(a[0].args[0]):
        paths = a[0].args[0].args
        return paths[0] if len(paths) == 1 else a[0].args[0]
    return _prev_normal(a, ev_, expr)


_K._BUILTINS["Normal"] = _normal_w25


def series_of(x, fname):
    """(head, pairs, opts) for a TimeSeries / EventSeries / single-path TemporalData argument"""
    if isinstance(x, Expr) and x.head in ("TimeSeries", "EventSeries") and _canonical(x):
        p, o = pairs_of(x, fname)
        return x.head, p, o
    if isinstance(x, Expr) and x.head == "TemporalData" and x.args and is_list(x.args[0]):
        p, o = pairs_of(x, fname)
        return "TimeSeries", p, o
    if isinstance(x, Expr) and x.head in ("TimeSeries", "EventSeries", "TemporalData"):
        raise Unevaluated()
    if is_list(x):
        p, _ = pairs_of(x, fname)
        return "TimeSeries", p, {}
    if isinstance(x, (Symbol, Expr)) and not isinstance(x, str):
        raise Unevaluated()
    raise TypeError(f"{fname}: {x!r} is not a time series")


def _strict_series(x, fname):
    if not (isinstance(x, Expr) and x.head in ("TimeSeries", "EventSeries", "TemporalData")):
        if isinstance(x, Symbol):
            raise Unevaluated()
        if is_list(x):
            return series_of(x, fname)
        raise TypeError(f"{fname}: {x!r} is not a time series or event series")
    return series_of(x, fname)


def _sub(a, b):
    return _ev(Expr("Plus", a, Expr("Times", -1, b)))


@fn("RegularlySampledQ", area=AREA, status="added",
    notes="True if the timestamps of a time series, event series or TemporalData path are equally spaced (exact "
          "comparison for exact times, relative 1e-10 for machine times), or the TemporalRegularity option says so; "
          "False otherwise; non-series arguments raise TypeError. Check: exact; reference in tests.")
def _rsq(x: Expr) -> Boolean:
    if isinstance(x, Expr) and x.head == "TemporalData" and x.args and is_list(x.args[0]):
        paths, opts = td_paths(x)
        return all(_regular(p, opts) for p in paths)
    h, pairs, opts = _strict_series(x, "RegularlySampledQ")
    return _regular(pairs, opts)


@fn("MinimumTimeIncrement", area=AREA, status="added",
    notes="smallest difference between consecutive timestamps of a time series, event series or list of "
          "time-value pairs (exact for exact times); a single timestamp raises ValueError. "
          "Check: exact; reference in tests.")
def _mti(x: Expr) -> Real:
    h, pairs, opts = _strict_series(x, "MinimumTimeIncrement")
    if len(pairs) < 2:
        raise ValueError("MinimumTimeIncrement: needs at least two timestamps")
    return _min_increment(pairs)


@fn("TimeSeriesShift", area=AREA, status="added",
    notes="TimeSeriesShift[ts, dt] shifts every timestamp by dt; TimeSeriesShift[ts] shifts by 1 (Wolfram: the "
          "minimum time increment for regular series; here 1 only for unit-spaced series, else raises "
          "NotImplementedError). Symbolic dt stays unevaluated. Check: exact; reference in tests.")
def _tss(x: Expr, dt: Real = None) -> Expr:
    h, pairs, opts = _strict_series(x, "TimeSeriesShift")
    if dt is None:
        d = _min_increment(pairs)
        if d is None or not _regular(pairs, opts):
            raise NotImplementedError("TimeSeriesShift: the default shift needs a regularly sampled series; give dt")
        dt = d
    _time(dt, "TimeSeriesShift")
    return make_ts([(_ev(Expr("Plus", t, dt)), v) for t, v in pairs], opts, h)


@fn("TimeSeriesRescale", area=AREA, status="added",
    notes="TimeSeriesRescale[ts, {tmin, tmax}] maps the timestamps affinely onto [tmin, tmax] (values unchanged); "
          "tmin >= tmax or a series with a single timestamp raises ValueError. Check: exact; reference in tests.")
def _tsr(x: Expr, spec: ListOf[Real]) -> Expr:
    h, pairs, opts = _strict_series(x, "TimeSeriesRescale")
    if not (is_list(spec) and len(spec.args) == 2):
        raise TypeError("TimeSeriesRescale: the second argument must be {tmin, tmax}")
    a, b = (_time(v, "TimeSeriesRescale") for v in spec.args)
    if not _key(a) < _key(b):
        raise ValueError("TimeSeriesRescale: need tmin < tmax")
    t0, t1 = pairs[0][0], pairs[-1][0]
    if _key(t0) == _key(t1):
        raise ValueError("TimeSeriesRescale: the series has a single timestamp")
    out = []
    for t, v in pairs:
        nt = _ev(Expr("Plus", a, Expr("Times", _sub(b, a), _sub(t, t0), Expr("Power", _sub(t1, t0), -1))))
        out.append((nt, v))
    return make_ts(out, opts, h)


@fn("TimeSeriesWindow", area=AREA, status="added",
    notes="TimeSeriesWindow[ts, {tmin, tmax}] keeps the time-value pairs with tmin <= t <= tmax (All for an open "
          "end); a list of windows {{t1, t2}, ...} keeps their union; an empty window raises ValueError. "
          "Check: exact; reference in tests.")
def _tsw(x: Expr, spec: ListOf[Real] | ListOf[ListOf[Real]]) -> Expr:
    h, pairs, opts = _strict_series(x, "TimeSeriesWindow")
    if not is_list(spec):
        raise TypeError("TimeSeriesWindow: the window must be {tmin, tmax}")
    wins = list(spec.args) if spec.args and all(is_list(w) for w in spec.args) else [spec]
    bounds = []
    for w in wins:
        if not (is_list(w) and len(w.args) == 2):
            raise TypeError("TimeSeriesWindow: each window must be {tmin, tmax}")
        lo, hi = w.args
        lo = -float("inf") if _is_sym(lo, "All") or _is_sym(lo, "Automatic") else _key(_time(lo, "TimeSeriesWindow"))
        hi = float("inf") if _is_sym(hi, "All") or _is_sym(hi, "Automatic") else _key(_time(hi, "TimeSeriesWindow"))
        if lo > hi:
            raise ValueError("TimeSeriesWindow: need tmin <= tmax")
        bounds.append((lo, hi))
    out = [(t, v) for t, v in pairs if any(lo <= _key(t) <= hi for lo, hi in bounds)]
    if not out:
        raise ValueError("TimeSeriesWindow: no data in the window")
    return make_ts(out, opts, h)


@fn("TimeSeriesMap", area=AREA, status="added",
    notes="TimeSeriesMap[f, ts] applies f to every value; TimeSeriesMap[f, ts, {t}]... not supported; "
          "TimeSeriesMap[f, td] maps every path of TemporalData. Check: exact; reference in tests.")
def _tsm(f: Fn, x: Expr) -> Expr:
    if isinstance(x, Expr) and x.head == "TemporalData" and x.args and is_list(x.args[0]):
        paths, opts = td_paths(x)
        return make_td([[(t, call(f, v)) for t, v in p] for p in paths], opts)
    h, pairs, opts = _strict_series(x, "TimeSeriesMap")
    return make_ts([(t, call(f, v)) for t, v in pairs], opts, h)


@fn("TimeSeriesMapThread", area=AREA, status="added",
    notes="TimeSeriesMapThread[f, ts] gives f[t, v] at every timestamp t (Wolfram: f[t, v] is the new value). "
          "Check: exact; reference in tests.")
def _tsmt(f: Fn, x: Expr) -> Expr:
    h, pairs, opts = _strict_series(x, "TimeSeriesMapThread")
    return make_ts([(t, call(f, t, v)) for t, v in pairs], opts, h)


def _all_pairs(x, fname):
    if isinstance(x, Expr) and x.head == "TemporalData" and x.args and is_list(x.args[0]):
        return td_paths(x)
    if is_list(x) and x.args and all(isinstance(s, Expr) and s.head in ("TimeSeries", "EventSeries", "TemporalData")
                                     for s in x.args):
        out, opts = [], {}
        for s in x.args:
            if s.head == "TemporalData":
                ps, o = td_paths(s)
                out.extend(ps)
            else:
                h, p, o = series_of(s, fname)
                out.append(p)
            opts.update(o)
        return out, opts
    if isinstance(x, Symbol) or (is_list(x) and any(isinstance(s, Symbol) for s in x.args)):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a list of time series or a TemporalData ensemble")


@fn("TimeSeriesThread", area=AREA, status="added",
    notes="TimeSeriesThread[f, {ts1, ts2, ...}] or TimeSeriesThread[f, td]: the time series whose value at each "
          "time t of the union of the timestamps (restricted to the common time range) is f[{ts1[t], ts2[t], ...}] "
          "(series are resampled with their own interpolation). Check: exact; reference in tests.")
def _tst(f: Fn, x: Expr) -> Expr:
    paths, opts = _all_pairs(x, "TimeSeriesThread")
    lo = max(_key(p[0][0]) for p in paths)
    hi = min(_key(p[-1][0]) for p in paths)
    if lo > hi:
        raise ValueError("TimeSeriesThread: the series have no common time range")
    times = {}
    for p in paths:
        for t, _ in p:
            k = _key(t)
            if lo <= k <= hi and k not in times:
                times[k] = t
    out = []
    for k in sorted(times):
        t = times[k]
        out.append((t, call(f, L(*(value_at(p, opts, t, "TimeSeriesThread") for p in paths)))))
    return make_ts(out, {})


@fn("TimeSeriesInsert", area=AREA, status="added",
    notes="TimeSeriesInsert[ts, {t, v}] inserts (or replaces) the value v at time t; TimeSeriesInsert[ts, {{t1, v1}, "
          "...}] inserts several; TimeSeriesInsert[ts1, ts2] inserts the path of ts2. Check: exact; reference in "
          "tests.")
def _tsi(x: Expr, new: ListOf[Expr] | Expr) -> Expr:
    h, pairs, opts = _strict_series(x, "TimeSeriesInsert")
    if isinstance(new, Expr) and new.head in ("TimeSeries", "EventSeries", "TemporalData"):
        _, add, _ = series_of(new, "TimeSeriesInsert")
    elif is_list(new) and len(new.args) == 2 and not is_list(new.args[0]):
        add = [(_time(new.args[0], "TimeSeriesInsert"), new.args[1])]
    elif is_list(new) and all(is_list(p) and len(p.args) == 2 for p in new.args):
        add = [(_time(p.args[0], "TimeSeriesInsert"), p.args[1]) for p in new.args]
    else:
        raise TypeError("TimeSeriesInsert: insert {t, v}, a list of pairs or a time series")
    d = {_key(t): (t, v) for t, v in pairs}
    for t, v in add:
        d[_key(t)] = (t, v)
    return make_ts([d[k] for k in sorted(d)], opts, h)


@fn("TimeSeriesAggregate", area=AREA, status="added",
    notes="TimeSeriesAggregate[ts, dt] / TimeSeriesAggregate[ts, dt, f]: splits the time range into consecutive "
          "windows [t0 + k dt, t0 + (k + 1) dt) starting at the first timestamp and applies f (default Mean) to the "
          "values in each non-empty window, placed at the window start (numeric dt only; calendar specifications "
          "such as \"Week\" raise NotImplementedError; Wolfram's alignment options are not supported). "
          "Check: exact; reference in tests.")
def _tsa(x: Expr, dt: Real, f: Fn = None) -> Expr:
    h, pairs, opts = _strict_series(x, "TimeSeriesAggregate")
    if isinstance(dt, str) or (is_list(dt)):
        raise NotImplementedError("TimeSeriesAggregate: calendar or {dt, align} window specifications are not "
                                  "supported; give a numeric window length")
    _time(dt, "TimeSeriesAggregate")
    if not _key(dt) > 0:
        raise ValueError("TimeSeriesAggregate: the window length must be positive")
    f = Symbol("Mean") if f is None else f
    t0 = pairs[0][0]
    groups = {}
    for t, v in pairs:
        k = int(sp.floor(S(_sub(t, t0)) / S(dt))) if not has_float(L(t, t0, dt)) else \
            int((_key(t) - _key(t0)) // _key(dt))
        groups.setdefault(k, []).append(v)
    out = [(_ev(Expr("Plus", t0, Expr("Times", k, dt))), call(f, L(*vs))) for k, vs in sorted(groups.items())]
    return make_ts(out, opts, h)


@fn("TimeSeriesResample", area=AREA, status="added",
    notes="TimeSeriesResample[ts] (regular grid with the minimum time increment), TimeSeriesResample[ts, dt], "
          "TimeSeriesResample[ts, {tmin, tmax, dt}], TimeSeriesResample[ts, {{t1, t2, ...}}] and a trailing "
          "resampling method (\"Constant\" | \"Interpolation\" | {\"Interpolation\", InterpolationOrder -> 0 | 1}); "
          "values come from the path function ts[t] (Wolfram's \"Holiday\"/calendar specifications raise "
          "NotImplementedError). Check: exact; reference in tests.")
def _tsrs(x: Expr, *rest: Expr) -> Expr:
    h, pairs, opts = _strict_series(x, "TimeSeriesResample")
    rest = list(rest)
    if len(rest) == 2:
        _check_resampling(rest[1], "TimeSeriesResample")
        opts = dict(opts)
        opts["ResamplingMethod"] = rest[1]
        rest = rest[:1]
    if len(rest) > 1:
        raise ValueError("TimeSeriesResample: expected TimeSeriesResample[ts, spec, method]")
    t0, t1 = pairs[0][0], pairs[-1][0]
    if not rest or _is_sym(rest[0], "Automatic"):
        spec = L(t0, t1, _min_increment(pairs) if len(pairs) > 1 else 1)
    elif is_list(rest[0]):
        spec = rest[0]
    elif isinstance(rest[0], str):
        if rest[0] in ("Constant", "Interpolation", "LinearInterpolation"):
            opts = dict(opts)
            opts["ResamplingMethod"] = rest[0]
            spec = L(t0, t1, _min_increment(pairs) if len(pairs) > 1 else 1)
        else:
            raise NotImplementedError(f"TimeSeriesResample: the specification {rest[0]!r} is not supported")
    else:
        spec = L(t0, t1, rest[0])
    if len(spec.args) == 1 and is_list(spec.args[0]):
        times = [_time(t, "TimeSeriesResample") for t in spec.args[0].args]
    elif len(spec.args) == 3:
        a, b, d = (_time(v, "TimeSeriesResample") for v in spec.args)
        if not _key(d) > 0:
            raise ValueError("TimeSeriesResample: the step must be positive")
        times, k = [], 0
        while True:
            t = _ev(Expr("Plus", a, Expr("Times", k, d)))
            if _key(t) > _key(b) + 1e-12 * max(1.0, abs(_key(b))):
                break
            times.append(t)
            k += 1
    else:
        raise ValueError("TimeSeriesResample: invalid resampling specification")
    if not times:
        raise ValueError("TimeSeriesResample: no resampling times")
    out = [(t, value_at(pairs, opts, t, "TimeSeriesResample", h == "TimeSeries")) for t in times]
    return make_ts(out, {k: v for k, v in opts.items() if k == "ResamplingMethod"}, h)


for _n, _note in (
        ("ResamplingMethod", "option of TimeSeries, TemporalData and TimeSeriesResample: Automatic, \"Constant\", "
                             "\"Interpolation\" or {\"Interpolation\", InterpolationOrder -> 0 | 1}; other settings "
                             "raise (ValueError or NotImplementedError)."),
        ("TemporalRegularity", "option of TimeSeries / EventSeries: True, False or Automatic; RegularlySampledQ "
                               "reports it when set; other settings raise ValueError."),
        ("MergingFunction", "option of TimeSeries / EventSeries: the function applied to the list of values sharing "
                            "a timestamp (default Mean)."),
):
    inert_option(_n, _note + " Inert as an expression. Check: reference (Wolfram documentation) in tests.", area=AREA)
