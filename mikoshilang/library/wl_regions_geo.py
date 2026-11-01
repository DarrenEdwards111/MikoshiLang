"""Wolfram batch W22 (part): geodesy on the reference ellipsoid, degree-minute-second conversion, and coordinate
bounds / coordinate arrays.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).

Geodesics are computed on the GRS80 ellipsoid of Wolfram's default datum "ITRF00" (a = 6378137 m,
f = 1/298.257222101) with Vincenty's inverse and direct formulae (iterated to 1e-12 rad, sub-millimetre accuracy).
No curated geographic data is used: locations are {lat, lon} pairs in degrees, GeoPosition[{lat, lon}] or
GeoPosition[{lat, lon, h}] (heights ignored, as in Wolfram).  Entities raise NotImplementedError.
"""
from __future__ import annotations

import math
import re

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, Unevaluated, L, R, has_float
from .wl_graphics_color import K_input
from .wl_graphics_core import split_trailing_options, opt_name
from .wl_geomprims import sx, out, num

AREA = "geometry: geodesy & coordinate arrays (W22)"

A_GRS80 = 6378137.0
F_GRS80 = 1 / 298.257222101
B_GRS80 = A_GRS80 * (1 - F_GRS80)

_LEN_UNITS = {"Meters": 1.0, "Kilometers": 1000.0, "Miles": 1609.344, "Feet": 0.3048, "NauticalMiles": 1852.0,
              "Yards": 0.9144, "Centimeters": 0.01}
_ANG_UNITS = {"AngularDegrees": 1.0, "Radians": 180 / math.pi, "ArcMinutes": 1 / 60, "ArcSeconds": 1 / 3600}


# ───────────────────────────── DMS parsing ─────────────────────────────

_DMS_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d*)?)\s*(?:d|°|deg)?\s*(?:(\d+(?:\.\d*)?)\s*(?:m|'|′)?\s*)?"
    r"(?:(\d+(?:\.\d*)?)\s*(?:s|\"|″|'')?\s*)?([NSEW]|North|South|East|West)?\s*$", re.IGNORECASE)


def _num_text(t):
    return sp.Float(t, 15) if "." in t else sp.Integer(t)


def parse_dms(s, f):
    """A DMS string -> SymPy degrees (signed by N/S/E/W or a leading minus)."""
    m = _DMS_RE.match(s)
    if not m or not any(m.group(i) for i in (1,)):
        raise ValueError(f"{f}: {s!r} is not a degrees-minutes-seconds string")
    d = _num_text(m.group(1))
    mi = _num_text(m.group(2)) if m.group(2) else sp.Integer(0)
    se = _num_text(m.group(3)) if m.group(3) else sp.Integer(0)
    if mi >= 60 or se >= 60:
        raise ValueError(f"{f}: minutes and seconds in {s!r} must be less than 60")
    neg = d < 0 or m.group(1).strip().startswith("-")
    v = abs(d) + mi / 60 + se / 3600
    card = (m.group(4) or "").upper()
    if card in ("S", "W", "SOUTH", "WEST"):
        neg = not neg if not neg else neg
        neg = True
    return -v if neg else v


def _split_pair_string(s, f):
    """'lat lon' DMS string pair (e.g. 40°5'49.92\"N 88°14'45.96\"W) -> two strings."""
    m = re.match(r"^(.*?[NS])\s*,?\s*(.*?[EW])\s*$", s.strip(), re.IGNORECASE)
    if not m:
        raise ValueError(f"{f}: {s!r} is not a latitude-longitude DMS string")
    return m.group(1), m.group(2)


# ───────────────────────────── quantities and locations ─────────────────────────────

def _quantity(x):
    return isinstance(x, Expr) and x.head == "Quantity" and len(x.args) == 2 and isinstance(x.args[1], str)


def angle_deg(x, f, what="angle"):
    """Degrees (SymPy) from a number in degrees, a Quantity angle or a DMS string."""
    if isinstance(x, str):
        return parse_dms(x, f)
    if _quantity(x):
        u = x.args[1]
        if u not in _ANG_UNITS:
            raise ValueError(f"{f}: {what} unit {u!r} is not an angle unit")
        v = sx(x.args[0], f, what)
        if u == "Radians":
            return v * 180 / sp.pi
        return v * sp.nsimplify(_ANG_UNITS[u])
    return sx(x, f, what)


def length_m(x, f, what="distance"):
    if _quantity(x):
        u = x.args[1]
        if u not in _LEN_UNITS:
            raise ValueError(f"{f}: {what} unit {u!r} is not a supported length unit ({', '.join(_LEN_UNITS)})")
        return sx(x.args[0], f, what) * sp.nsimplify(_LEN_UNITS[u])
    return sx(x, f, what)


def location(x, f, what="location"):
    """(lat, lon) in degrees (SymPy) of {lat, lon}, GeoPosition[{lat, lon(, h)}] or a 'lat lon' DMS string."""
    if isinstance(x, Expr) and x.head == "GeoPosition":
        if len(x.args) > 1 and not (len(x.args) == 2 and x.args[1] in ("ITRF00", "WGS84")):
            raise NotImplementedError(f"{f}: only the default datum (ITRF00 / WGS84) is supported")
        if not x.args:
            raise ValueError(f"{f}: GeoPosition needs coordinates")
        x = x.args[0]
        if not K.is_list(x):
            if isinstance(x, Symbol):
                raise Unevaluated()
            raise TypeError(f"{f}: GeoPosition coordinates must be {{lat, lon}}")
    if isinstance(x, Expr) and x.head == "Entity":
        raise NotImplementedError(f"{f}: geographic entities need curated data, which is not available")
    if isinstance(x, Expr) and isinstance(x.head, str) and x.head.startswith("GeoPosition"):
        raise NotImplementedError(f"{f}: {x.head} positions are not supported (use GeoPosition)")
    if isinstance(x, str):
        a, b = _split_pair_string(x, f)
        lat, lon = parse_dms(a, f), parse_dms(b, f)
    elif K.is_list(x) and len(x.args) in (2, 3) and not any(K.is_list(c) for c in x.args):
        lat, lon = angle_deg(x.args[0], f, "latitude"), angle_deg(x.args[1], f, "longitude")
    else:
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: {what} must be {{lat, lon}} or GeoPosition[{{lat, lon}}], not {K_input(x)}")
    for v in (lat, lon):
        if v.free_symbols:
            raise Unevaluated()
    if abs(num(lat)) > 90:
        raise ValueError(f"{f}: latitude {lat} must lie in [-90, 90]")
    return lat, lon


def _wrap180(x):
    y = math.fmod(x + 180.0, 360.0)
    if y <= 0:
        y += 360.0
    return y - 180.0


# ───────────────────────────── Vincenty ─────────────────────────────

def vincenty_inverse(lat1, lon1, lat2, lon2, f="GeoDistance"):
    """(distance m, initial azimuth deg, final azimuth deg) on GRS80."""
    a, b, fl = A_GRS80, B_GRS80, F_GRS80
    p1, p2 = math.radians(lat1), math.radians(lat2)
    Lr = math.radians(_wrap180(lon2 - lon1))
    U1 = math.atan((1 - fl) * math.tan(p1)) if abs(lat1) < 90 else math.copysign(math.pi / 2, lat1)
    U2 = math.atan((1 - fl) * math.tan(p2)) if abs(lat2) < 90 else math.copysign(math.pi / 2, lat2)
    sU1, cU1, sU2, cU2 = math.sin(U1), math.cos(U1), math.sin(U2), math.cos(U2)
    lam = Lr
    for _ in range(500):
        sl, cl = math.sin(lam), math.cos(lam)
        ss = math.hypot(cU2 * sl, cU1 * sU2 - sU1 * cU2 * cl)
        if ss == 0:
            return 0.0, 0.0, 0.0
        cs = sU1 * sU2 + cU1 * cU2 * cl
        sig = math.atan2(ss, cs)
        sa = cU1 * cU2 * sl / ss
        c2a = 1 - sa * sa
        c2sm = cs - 2 * sU1 * sU2 / c2a if c2a != 0 else 0.0
        C = fl / 16 * c2a * (4 + fl * (4 - 3 * c2a))
        lp = lam
        lam = Lr + (1 - C) * fl * sa * (sig + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm * c2sm)))
        if abs(lam - lp) < 1e-13:
            break
    else:
        raise NotImplementedError(f"{f}: the geodesic between nearly antipodal points is not supported "
                                  f"(Vincenty's method does not converge)")
    if abs(lam) > math.pi:
        raise NotImplementedError(f"{f}: the geodesic between nearly antipodal points is not supported")
    u2 = c2a * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    ds = B * ss * (c2sm + B / 4 * (cs * (-1 + 2 * c2sm * c2sm) - B / 6 * c2sm * (-3 + 4 * ss * ss) *
                                   (-3 + 4 * c2sm * c2sm)))
    s = b * A * (sig - ds)
    a1 = math.degrees(math.atan2(cU2 * math.sin(lam), cU1 * sU2 - sU1 * cU2 * math.cos(lam)))
    a2 = math.degrees(math.atan2(cU1 * math.sin(lam), -sU1 * cU2 + cU1 * sU2 * math.cos(lam)))
    return s, a1, a2


def vincenty_direct(lat1, lon1, az, s):
    """(lat2, lon2) in degrees after travelling s metres with initial azimuth az (degrees)."""
    a, b, fl = A_GRS80, B_GRS80, F_GRS80
    al1 = math.radians(az)
    sa1, ca1 = math.sin(al1), math.cos(al1)
    tU1 = (1 - fl) * math.tan(math.radians(lat1)) if abs(lat1) < 90 else math.copysign(1e300, lat1)
    cU1 = 1 / math.sqrt(1 + tU1 * tU1)
    sU1 = tU1 * cU1
    s1 = math.atan2(tU1, ca1)
    sa = cU1 * sa1
    c2a = 1 - sa * sa
    u2 = c2a * (a * a - b * b) / (b * b)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    sig = s / (b * A)
    for _ in range(500):
        c2sm = math.cos(2 * s1 + sig)
        ss, cs = math.sin(sig), math.cos(sig)
        ds = B * ss * (c2sm + B / 4 * (cs * (-1 + 2 * c2sm * c2sm) - B / 6 * c2sm * (-3 + 4 * ss * ss) *
                                       (-3 + 4 * c2sm * c2sm)))
        sp_ = sig
        sig = s / (b * A) + ds
        if abs(sig - sp_) < 1e-13:
            break
    ss, cs = math.sin(sig), math.cos(sig)
    c2sm = math.cos(2 * s1 + sig)
    x = sU1 * ss - cU1 * cs * ca1
    lat2 = math.atan2(sU1 * cs + cU1 * ss * ca1, (1 - fl) * math.hypot(sa, x))
    lam = math.atan2(ss * sa1, cU1 * cs - sU1 * ss * ca1)
    C = fl / 16 * c2a * (4 + fl * (4 - 3 * c2a))
    Lr = lam - (1 - C) * fl * sa * (sig + C * ss * (c2sm + C * cs * (-1 + 2 * c2sm * c2sm)))
    return math.degrees(lat2), _wrap180(lon1 + math.degrees(Lr))


def _inverse_checked(p, q, f):
    lat1, lon1 = (num(c) for c in p)
    lat2, lon2 = (num(c) for c in q)
    s, a1, a2 = vincenty_inverse(lat1, lon1, lat2, lon2, f)
    if s > 0:
        la, lo = vincenty_direct(lat1, lon1, a1, s)
        if abs(la - lat2) > 1e-7 or abs(_wrap180(lo - lon2)) * math.cos(math.radians(lat2)) > 1e-7:
            raise ArithmeticError(f"{f}: result failed verification")
    return s, a1, a2


def _path_type(rest, f):
    a, opts = split_trailing_options(list(rest))
    for r in opts:
        n = opt_name(r)
        if n not in ("UnitSystem",):
            raise ValueError(f"{f}: {n} is not a known option of {f}")
    if a:
        if len(a) > 1:
            raise ValueError(f"{f}: too many arguments")
        if a[0] != "Geodesic":
            if isinstance(a[0], str):
                raise NotImplementedError(f"{f}: path type {a[0]!r} is not supported (only \"Geodesic\")")
            raise TypeError(f"{f}: the path type must be a string such as \"Geodesic\"")
    return opts


def _unit_system(opts, f):
    for r in opts:
        if opt_name(r) == "UnitSystem":
            v = r.args[1]
            if v == "Metric":
                return "Kilometers", 1000.0
            if v == "SIBase":
                return "Meters", 1.0
            if v == "Imperial":
                return "Miles", 1609.344
            raise ValueError(f"{f}: UnitSystem -> {K_input(v)} must be \"Metric\", \"SIBase\" or \"Imperial\"")
    return "Kilometers", 1000.0


def _is_loc_list(x):
    return K.is_list(x) and len(x.args) >= 2 and all(
        (K.is_list(p) and len(p.args) in (2, 3) and not any(K.is_list(c) for c in p.args)) or
        (isinstance(p, Expr) and p.head == "GeoPosition") for p in x.args)


@fn("GeoDistance", area=AREA, status="added",
    notes="GeoDistance[loc1, loc2] gives the geodesic distance on the GRS80 reference ellipsoid (default datum "
          "ITRF00) between locations {lat, lon} (degrees, Quantity angles or DMS strings) or GeoPosition[{lat, lon}] "
          "(heights ignored); GeoDistance[{loc1, loc2, ...}] gives the length of the geodesic polyline. The result "
          "is Quantity[d, \"Kilometers\"] (machine precision); option UnitSystem -> \"Metric\" (kilometres, "
          "default), \"SIBase\" (metres) or \"Imperial\" (miles). Only the path type \"Geodesic\" is supported "
          "(\"Rhumb\", \"GreatCircle\", ... raise NotImplementedError), entities raise NotImplementedError (no "
          "curated data), and nearly antipodal pairs, where Vincenty's iteration does not converge, raise "
          "NotImplementedError. Symbolic coordinates stay unevaluated. Deviation: Wolfram's default unit follows "
          "$UnitSystem (miles in the US). Check: self-check (the direct problem is solved from the result and must "
          "return to loc2, else ArithmeticError) and reference values from GeographicLib in tests.")
def _geo_distance(a: Expr, *rest: Expr | Option) -> Expr:
    f = "GeoDistance"
    if _is_loc_list(a) and (not rest or opt_name(rest[0]) is not None or isinstance(rest[0], str)):
        opts = _path_type(rest, f)
        locs = [location(p, f) for p in a.args]
        tot = sum(_inverse_checked(locs[i], locs[i + 1], f)[0] for i in range(len(locs) - 1))
    else:
        if not rest:
            if isinstance(a, Symbol):
                raise Unevaluated()
            raise ValueError(f"{f}: expected GeoDistance[loc1, loc2] or GeoDistance[{{loc1, loc2, ...}}]")
        opts = _path_type(rest[1:], f)
        p, q = location(a, f), location(rest[0], f)
        tot = _inverse_checked(p, q, f)[0]
    unit, scale = _unit_system(opts, f)
    return Expr("Quantity", float(tot / scale), unit)


@fn("GeoDirection", area=AREA, status="added",
    notes="GeoDirection[loc1, loc2] gives the initial azimuth (clockwise from true north) of the geodesic from loc1 "
          "to loc2 on the GRS80 ellipsoid, as Quantity[α, \"AngularDegrees\"] with α in (-180, 180] (machine "
          "precision). Locations as in GeoDistance; coincident points give 0; nearly antipodal points raise "
          "NotImplementedError. Check: self-check (direct problem round trip) and GeographicLib reference values "
          "in tests.")
def _geo_direction(a: Expr, b: Expr) -> Expr:
    f = "GeoDirection"
    s, a1, _ = _inverse_checked(location(a, f), location(b, f), f)
    return Expr("Quantity", float(a1), "AngularDegrees")


_COMPASS = {}
_PTS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
_LONG = {"N": "North", "E": "East", "S": "South", "W": "West"}
for _i, _p in enumerate(_PTS16):
    _COMPASS[_p] = sp.Rational(45, 2) * _i
    if len(_p) == 1:
        _COMPASS[_LONG[_p]] = sp.Rational(45, 2) * _i
    elif len(_p) == 2:
        _COMPASS[_LONG[_p[0]] + _LONG[_p[1]]] = sp.Rational(45, 2) * _i
    else:
        _COMPASS[_LONG[_p[0]] + "-" + _LONG[_p[1]] + _LONG[_p[2]]] = sp.Rational(45, 2) * _i


def bearing(x, f):
    if isinstance(x, str):
        if x in _COMPASS:
            return _COMPASS[x]
        if re.search(r"\d", x):
            return parse_dms(x, f)
        raise NotImplementedError(f"{f}: compass point {x!r} is not supported (16-point names only)")
    return angle_deg(x, f, "bearing")


def _disp_parts(spec, f):
    """{dist, α} (lists allowed) -> (list of metres, list of degrees, shape)."""
    if not (K.is_list(spec) and len(spec.args) == 2):
        if isinstance(spec, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: a displacement must be {{dist, α}}, not {K_input(spec)}")
    d, al = spec.args

    def expand_len(v):
        if _quantity(v) and K.is_list(v.args[0]):
            return [length_m(Expr("Quantity", c, v.args[1]), f) for c in v.args[0].args]
        if K.is_list(v):
            return [length_m(c, f) for c in v.args]
        return length_m(v, f)

    def expand_ang(v):
        if _quantity(v) and K.is_list(v.args[0]):
            return [bearing(Expr("Quantity", c, v.args[1]), f) for c in v.args[0].args]
        if K.is_list(v):
            return [bearing(c, f) for c in v.args]
        return bearing(v, f)
    D, A = expand_len(d), expand_ang(al)
    for v in (D if isinstance(D, list) else [D]) + (A if isinstance(A, list) else [A]):
        if v.free_symbols:
            raise Unevaluated()
    for v in (D if isinstance(D, list) else [D]):
        if num(v) < 0:
            raise ValueError(f"{f}: distances must be non-negative")
    return D, A


def _qty_out(v, unit, flt):
    if isinstance(v, list):
        return Expr("Quantity", L(*[float(num(c)) if flt else R(c) for c in v]), unit)
    return Expr("Quantity", float(num(v)) if flt else R(v), unit)


@fn("GeoDisplacement", area=AREA, status="added",
    notes="GeoDisplacement[{dist, α}] represents a geodesic displacement (dist a number of metres or a Quantity "
          "length, α a bearing in degrees, a Quantity angle, a DMS string or one of the 16 compass points \"N\", "
          "\"NNE\", ..., \"NorthEast\", \"North-NorthEast\", ...; lists of distances or bearings allowed); it "
          "evaluates to the normal form GeoDisplacement[{Quantity[d, \"Meters\"], Quantity[α, \"AngularDegrees\"]}]. "
          "GeoDisplacement[loc1, loc2] gives the displacement from loc1 to loc2 along the GRS80 geodesic (machine "
          "precision). Path types other than \"Geodesic\" and the 32-point / traditional wind names raise "
          "NotImplementedError; negative distances ValueError. Check: self-check (direct problem round trip) and "
          "GeographicLib reference values in tests.")
def _geo_displacement(a: Expr, *rest: Expr) -> Expr:
    f = "GeoDisplacement"
    if K.is_list(a) and len(a.args) == 2 and not rest or (rest and len(rest) == 1 and isinstance(rest[0], str) and
                                                            K.is_list(a) and len(a.args) == 2 and
                                                            not _looks_loc(a)):
        if rest:
            _path_type(rest, f)
        if _looks_loc(a) and not rest and _plain_numbers(a) is False:
            pass
        D, A = _disp_parts(a, f)
        flt = has_float(a)
        is_normal = (_quantity(a.args[0]) and a.args[0].args[1] == "Meters" and _quantity(a.args[1]) and
                     a.args[1].args[1] == "AngularDegrees")
        if is_normal:
            raise Unevaluated()
        return Expr("GeoDisplacement", L(_qty_out(D, "Meters", flt), _qty_out(A, "AngularDegrees", flt)))
    if not rest or len(rest) > 2:
        raise ValueError(f"{f}: expected GeoDisplacement[{{dist, α}}] or GeoDisplacement[loc1, loc2]")
    _path_type(rest[1:], f)
    s, a1, _ = _inverse_checked(location(a, f), location(rest[0], f), f)
    return Expr("GeoDisplacement", L(Expr("Quantity", float(s), "Meters"), Expr("Quantity", float(a1),
                                                                                  "AngularDegrees")))


def _looks_loc(a):
    return False


def _plain_numbers(a):
    return None


@fn("GeoDestination", area=AREA, status="added",
    notes="GeoDestination[loc, {dist, α}] and GeoDestination[loc, GeoDisplacement[{dist, α}]] give the GeoPosition "
          "reached by travelling dist metres (or a Quantity length) along the geodesic with initial bearing α "
          "(degrees, Quantity angle, DMS string or compass point) on the GRS80 ellipsoid (Vincenty's direct "
          "formula, machine precision). Lists of distances or bearings give GeoPosition[{{lat1, lon1}, ...}] "
          "(a list for both gives the grid, distances outermost). Longitudes are reduced to [-180, 180). Path types "
          "other than \"Geodesic\" raise NotImplementedError. Check: self-check (the inverse problem from the result "
          "must give back the distance) and GeographicLib reference values in tests.")
def _geo_destination(loc: Expr, disp: Expr) -> Expr:
    f = "GeoDestination"
    p = location(loc, f)
    if isinstance(disp, Expr) and disp.head == "GeoDisplacement":
        if len(disp.args) == 2:
            _path_type(disp.args[1:], f)
        if not disp.args:
            raise ValueError(f"{f}: empty GeoDisplacement")
        disp = disp.args[0]
    D, A = _disp_parts(disp, f)
    lat1, lon1 = num(p[0]), num(p[1])

    def one(d, a):
        d, a = num(d), num(a)
        la, lo = vincenty_direct(lat1, lon1, a, d)
        if d > 0:
            s2, _, _ = vincenty_inverse(lat1, lon1, la, lo, f)
            if abs(s2 - d) > 1e-6 * max(1.0, d) and d < 19e6:
                raise ArithmeticError(f"{f}: result failed verification")
        if lo >= 180:
            lo -= 360
        return L(float(la), float(lo))
    if isinstance(D, list) and isinstance(A, list):
        res = L(*[L(*[one(d, a) for a in A]) for d in D])
    elif isinstance(D, list):
        res = L(*[one(d, A) for d in D])
    elif isinstance(A, list):
        res = L(*[one(D, a) for a in A])
    else:
        res = one(D, A)
    return Expr("GeoPosition", res)


# ───────────────────────────── latitude / longitude ─────────────────────────────

def _deg_q(v, flt):
    return Expr("Quantity", float(num(v)) if flt else R(v), "AngularDegrees")


@fn("Latitude", area=AREA, status="added",
    notes="Latitude[pos] gives the latitude of GeoPosition[{lat, lon}] (or {lat, lon}, or a 'lat lon' DMS string) "
          "as Quantity[lat, \"AngularDegrees\"]; exact input stays exact. Latitudes outside [-90, 90] raise "
          "ValueError; entities NotImplementedError; symbolic positions stay unevaluated. Check: exact (coordinate "
          "extraction).")
def _latitude(pos: Expr) -> Expr:
    lat, _ = location(pos, "Latitude")
    return _deg_q(lat, has_float(pos))


@fn("Longitude", area=AREA, status="added",
    notes="Longitude[pos] gives the longitude of GeoPosition[{lat, lon}] (or {lat, lon}, or a 'lat lon' DMS string) "
          "as Quantity[lon, \"AngularDegrees\"], as given (not reduced). Check: exact (coordinate extraction).")
def _longitude(pos: Expr) -> Expr:
    _, lon = location(pos, "Longitude")
    return _deg_q(lon, has_float(pos))


@fn("LatitudeLongitude", area=AREA, status="added",
    notes="LatitudeLongitude[pos] gives {Quantity[lat, \"AngularDegrees\"], Quantity[lon, \"AngularDegrees\"]} for "
          "GeoPosition[{lat, lon}], {lat, lon} or a 'lat lon' DMS string. Check: exact (coordinate extraction).")
def _latitude_longitude(pos: Expr) -> ListOf[Expr]:
    lat, lon = location(pos, "LatitudeLongitude")
    flt = has_float(pos)
    return L(_deg_q(lat, flt), _deg_q(lon, flt))


# ───────────────────────────── GeoDisk / GeoPath ─────────────────────────────

def _check_loc(x, f):
    location(x, f)


@fn("GeoDisk", area=AREA, status="added",
    notes="GeoDisk[loc, r] represents the geodesic disk of radius r (metres or a Quantity length) around loc "
          "({lat, lon} or GeoPosition); GeoDisk[loc, r, {α1, α2}] a sector. Inert geographic primitive (for "
          "GeoGraphics): arguments are validated (negative radius ValueError, non-length unit ValueError, bad "
          "location TypeError). GeoDisk[] and GeoDisk[loc] need $GeoLocation / a default radius and raise "
          "NotImplementedError. Check: reference (Wolfram documentation forms) in tests.")
def _geo_disk(*args: Expr) -> Expr:
    f = "GeoDisk"
    if len(args) < 2:
        raise NotImplementedError(f"{f}: GeoDisk[] and GeoDisk[loc] need a default location or radius; give "
                                  f"GeoDisk[loc, r]")
    if len(args) > 3:
        raise ValueError(f"{f}: expected GeoDisk[loc, r] or GeoDisk[loc, r, {{α1, α2}}]")
    try:
        _check_loc(args[0], f)
    except Unevaluated:
        pass
    r = length_m(args[1], f, "radius")
    if r.is_number and num(r) < 0:
        raise ValueError(f"{f}: the radius must be non-negative")
    if len(args) == 3:
        if not (K.is_list(args[2]) and len(args[2].args) == 2):
            raise TypeError(f"{f}: the sector must be {{α1, α2}}")
        for c in args[2].args:
            angle_deg(c, f)
    raise Unevaluated()


@fn("GeoPath", area=AREA, status="added",
    notes="GeoPath[{loc1, loc2, ...}] and GeoPath[{loc1, ...}, pathtype] (pathtype a string such as \"Geodesic\", "
          "\"Rhumb\", \"GreatCircle\") represent geographic paths; inert geographic primitive. Locations are "
          "validated ({lat, lon} or GeoPosition; GeoDisplacement steps allowed after the first location); fewer "
          "than two entries raise ValueError. Check: reference (Wolfram documentation forms) in tests.")
def _geo_path(locs: ListOf[Expr], *ptype: String | Option) -> Expr:
    f = "GeoPath"
    a, opts = split_trailing_options(list(ptype))
    if len(a) > 1:
        raise ValueError(f"{f}: expected GeoPath[{{loc1, ...}}] or GeoPath[{{loc1, ...}}, pathtype]")
    if a and not isinstance(a[0], str):
        raise TypeError(f"{f}: the path type must be a string")
    if not K.is_list(locs):
        if isinstance(locs, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: the path must be a list of locations")
    if len(locs.args) < 2:
        raise ValueError(f"{f}: a path needs at least two locations")
    for i, p in enumerate(locs.args):
        if i and isinstance(p, Expr) and p.head == "GeoDisplacement":
            continue
        try:
            _check_loc(p, f)
        except Unevaluated:
            pass
    raise Unevaluated()


# ───────────────────────────── DMS ─────────────────────────────

def _dms_list(v, f, flt):
    """Signed degrees -> [d, m, s] (all with the sign of v)."""
    neg = num(v) < 0
    a = -v if neg else v
    if flt:
        a = float(num(a))
        d = int(math.floor(a + 1e-12))
        rem = (a - d) * 60
        m = int(math.floor(rem + 1e-10))
        s = (rem - m) * 60
        if s < 0:
            s = 0.0
        if s >= 60 - 1e-9:
            s, m = 0.0, m + 1
        if m >= 60:
            m, d = m - 60, d + 1
        vals = [sp.Integer(d), sp.Integer(m), sp.Float(s)]
    else:
        a = sp.nsimplify(a) if not a.is_Rational else a
        if not a.is_Rational:
            a = sp.sympify(a)
            d = sp.floor(a)
            rem = (a - d) * 60
            m = sp.floor(rem)
            s = sp.simplify((rem - m) * 60)
        else:
            d = sp.floor(a)
            rem = (a - d) * 60
            m = sp.floor(rem)
            s = (rem - m) * 60
        vals = [d, m, s]
    return [-x for x in vals] if neg else vals


def _dms_input(x, f):
    """A DMS spec -> (signed degrees SymPy, inexact flag)."""
    if isinstance(x, str):
        return parse_dms(x, f), "." in x
    if _quantity(x):
        return angle_deg(x, f), has_float(x)
    if K.is_list(x):
        a = list(x.args)
        card = None
        if a and isinstance(a[-1], str):
            card = a.pop()
        if not 1 <= len(a) <= 3:
            raise ValueError(f"{f}: a DMS list must be {{d}}, {{d, m}} or {{d, m, s}}")
        vals = [sx(c, f, "DMS component") for c in a]
        concrete_check(vals)
        v = sum(c / 60 ** i for i, c in enumerate(vals))
        if card is not None:
            if card not in ("N", "S", "E", "W"):
                raise ValueError(f"{f}: the cardinal direction must be \"N\", \"S\", \"E\" or \"W\"")
            v = -abs(v) if card in ("S", "W") else abs(v)
        return v, has_float(x)
    v = sx(x, f, "angle")
    concrete_check([v])
    return v, has_float(x)


def concrete_check(vals):
    for v in vals:
        if sp.sympify(v).free_symbols:
            raise Unevaluated()


@fn("FromDMS", area=AREA, status="added",
    notes="FromDMS[{d, m, s}] gives d + m/60 + s/3600 (also {d, m} and {d}); FromDMS[\"dms string\"] parses strings "
          "such as \"40d26m46s N\", \"40°26'46.302\\\"\" or \"23d11m49.92s South\" (S and W give negative values). "
          "Exact input gives an exact rational; inexact input a machine number. Unparseable strings raise "
          "ValueError, other kinds TypeError, symbolic components stay unevaluated. Deviation: Wolfram returns a "
          "machine number for every string; exact strings give exact values here. Check: exact; tests use the "
          "round trip FromDMS[DMSList[x]] == x.")
def _from_dms(x: ListOf[Real] | String) -> Real:
    f = "FromDMS"
    if not (isinstance(x, str) or K.is_list(x)):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: expected a DMS list {{d, m, s}} or a DMS string, not {K_input(x)}")
    if K.is_list(x) and x.args and all(K.is_list(c) or isinstance(c, str) for c in x.args):
        return L(*[_from_dms(c) for c in x.args])
    v, flt = _dms_input(x, f)
    return float(num(v)) if flt else R(v)


@fn("DMSList", area=AREA, status="added",
    notes="DMSList[θ] converts an angle θ in decimal degrees to {d, m, s} (all components carry the sign of θ; "
          "exact input gives exact components, inexact input integer d, m and machine s); DMSList[\"dms string\"] "
          "parses a string; DMSList[{ϕ, λ}] gives the pair of lists. Symbolic θ stays unevaluated, other kinds "
          "raise TypeError. Check: exact; tests use the round trip FromDMS[DMSList[θ]] == θ.")
def _dms_list_fn(x: Real | String | ListOf[Real]) -> ListOf[Real]:
    f = "DMSList"
    if K.is_list(x):
        if len(x.args) != 2:
            raise ValueError(f"{f}: expected an angle or a {{lat, lon}} pair")
        return L(*[_dms_list_fn(c) for c in x.args])
    if isinstance(x, (bool,)) or (isinstance(x, Expr) and x.head in ("Rule", "Association")):
        raise TypeError(f"{f}: expected an angle in degrees or a DMS string, not {K_input(x)}")
    v, flt = _dms_input(x, f)
    d, m, s = _dms_list(v, f, flt)
    return L(R(d), R(m), float(s) if flt else R(s))


def _fmt_dms(v, flt, digits, card=None):
    d, m, s = _dms_list(v, "DMSString", flt)
    d, m, s = abs(d), abs(m), abs(s)
    if flt or not sp.sympify(s).is_Integer:
        sv = float(num(s))
        txt = f"{sv:.{digits}f}"
        if float(txt) >= 60:
            sv, m = 0.0, m + 1
            if m >= 60:
                m, d = m - 60, d + 1
            txt = f"{sv:.{digits}f}"
        if digits == 0:
            txt = txt.rstrip(".")
    else:
        txt = str(int(s))
    res = f"{int(d)}°{int(m)}'{txt}\""
    if card:
        res += card
    return res


@fn("DMSString", area=AREA, status="added",
    notes="DMSString[θ] gives the degrees-minutes-seconds string d°m's\" of an angle in decimal degrees (sign "
          "ignored); DMSString[θ, \"NS\"] / DMSString[θ, \"EW\"] add the cardinal direction (N/E for non-negative "
          "θ); DMSString[{d, m, s}], DMSString[{d, m, s, \"S\"}], DMSString[\"dms string\"] and Quantity angles are "
          "accepted; DMSString[{ϕ, λ}] gives \"lat N lon E\" style pairs. Inexact input shows seconds with 3 "
          "decimals (DMSString[θ, n] or DMSString[θ, {\"NS\", n}] set n); exact input with integer seconds shows "
          "them exactly, exact input with fractional seconds is shown numerically. A bad direction string raises "
          "ValueError. Check: exact; tests use documented strings and the round trip FromDMS[DMSString[θ]].")
def _dms_string(x: Real | String | ListOf[Expr], *spec: String | Integer | ListOf[Expr]) -> String:
    f = "DMSString"
    card_mode, digits = None, 3
    if len(spec) > 1:
        raise ValueError(f"{f}: expected DMSString[θ] or DMSString[θ, spec]")
    if spec:
        s0 = spec[0]
        if K.is_list(s0) and len(s0.args) == 2:
            card_mode, digits = s0.args
        elif isinstance(s0, str):
            card_mode = s0
        else:
            digits = s0
        if card_mode is not None and card_mode not in ("NS", "EW"):
            raise ValueError(f"{f}: the direction specification must be \"NS\" or \"EW\"")
        if isinstance(digits, bool) or not K.is_int(digits) or int(digits) < 0:
            raise ValueError(f"{f}: the number of digits must be a non-negative integer")
        digits = int(digits)
    if isinstance(x, Expr) and x.head == "GeoPosition":
        x = L(*location(x, f))
    if K.is_list(x) and len(x.args) == 2 and not any(K.is_list(c) or isinstance(c, str) for c in x.args) and \
            not spec_is_dms(x):
        lat, flt1 = _dms_input(x.args[0], f)
        lon, flt2 = _dms_input(x.args[1], f)
        flt = flt1 or flt2
        a = _fmt_dms(lat, flt, digits, "N" if num(lat) >= 0 else "S")
        b = _fmt_dms(lon, flt, digits, "E" if num(lon) >= 0 else "W")
        return f"{a} {b}"
    if isinstance(x, bool):
        raise TypeError(f"{f}: expected an angle, not {x!r}")
    v, flt = _dms_input(x, f)
    card = None
    if card_mode == "NS":
        card = "N" if num(v) >= 0 else "S"
    elif card_mode == "EW":
        card = "E" if num(v) >= 0 else "W"
    return _fmt_dms(v, flt, digits, card)


def spec_is_dms(x):
    """A 2-element list {d, m} is ambiguous with {lat, lon}; Wolfram reads two numbers as a lat-lon pair."""
    return False


# ───────────────────────────── coordinate bounds and arrays ─────────────────────────────

def _coord_leaves(x, f, depth=0):
    """All coordinate vectors (innermost lists of numbers) of an array of coordinates."""
    if not K.is_list(x):
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: expected an array of coordinates, not {K_input(x)}")
    if not x.args:
        raise ValueError(f"{f}: the coordinate array is empty")
    if all(not K.is_list(c) for c in x.args):
        if depth == 0:
            return [[sx(c, f, "coordinate")] for c in x.args]
        return [[sx(c, f, "coordinate") for c in x.args]]
    res = []
    for c in x.args:
        res += _coord_leaves(c, f, depth + 1)
    return res


def _pad_spec(d, n, bounds, f):
    if isinstance(d, Expr) and d.head == "Scaled" and len(d.args) == 1:
        s = d.args[0]
        if K.is_list(s):
            ss = [sx(c, f, "padding") for c in s.args]
            if len(ss) != n:
                raise ValueError(f"{f}: Scaled padding must have {n} components")
        else:
            ss = [sx(s, f, "padding")] * n
        return [((hi - lo) * k, (hi - lo) * k) for (lo, hi), k in zip(bounds, ss)]
    if K.is_list(d):
        if len(d.args) != n:
            raise ValueError(f"{f}: the padding must have {n} components")
        res = []
        for c in d.args:
            if K.is_list(c):
                if len(c.args) != 2:
                    raise ValueError(f"{f}: padding pairs must be {{dmin, dmax}}")
                res.append((sx(c.args[0], f, "padding"), sx(c.args[1], f, "padding")))
            else:
                v = sx(c, f, "padding")
                res.append((v, v))
        return res
    v = sx(d, f, "padding")
    return [(v, v)] * n


@fn("CoordinateBounds", area=AREA, status="added",
    notes="CoordinateBounds[coords] gives {{xmin, xmax}, {ymin, ymax}, ...} for a list (or nested array) of "
          "coordinate vectors (a flat list of numbers is a list of 1-D points); CoordinateBounds[coords, δ] pads by "
          "δ, {δx, ...}, {{δxmin, δxmax}, ...} or Scaled[s] (a fraction of each range). Vectors of different "
          "lengths raise ValueError, non-numeric entries TypeError, symbolic coordinates stay unevaluated. "
          "Check: exact (min / max).")
def _coordinate_bounds(coords: ListOf[Expr], *pad: Expr) -> MatrixOf[Real]:
    f = "CoordinateBounds"
    P = _coord_leaves(coords, f)
    if len({len(p) for p in P}) != 1:
        raise ValueError(f"{f}: all coordinate vectors must have the same length")
    concrete_check([c for p in P for c in p])
    n = len(P[0])
    bounds = []
    for i in range(n):
        col = [p[i] for p in P]
        lo = min(col, key=lambda v: num(v))
        hi = max(col, key=lambda v: num(v))
        bounds.append((lo, hi))
    if len(pad) > 1:
        raise ValueError(f"{f}: expected CoordinateBounds[coords] or CoordinateBounds[coords, δ]")
    if pad:
        pads = _pad_spec(pad[0], n, bounds, f)
        bounds = [(lo - a, hi + b) for (lo, hi), (a, b) in zip(bounds, pads)]
    return out([[lo, hi] for lo, hi in bounds], coords, *pad)


def _steps(d, n, bounds, f):
    if isinstance(d, Expr) and d.head == "Into" and len(d.args) == 1:
        k = d.args[0]
        ks = [c for c in k.args] if K.is_list(k) else [k] * n
        if len(ks) != n:
            raise ValueError(f"{f}: Into needs {n} counts")
        res = []
        for (lo, hi), c in zip(bounds, ks):
            if isinstance(c, bool) or not K.is_int(c) or int(c) < 1:
                raise ValueError(f"{f}: Into[n] needs positive integers")
            res.append((hi - lo) / int(c))
        return res
    if K.is_list(d):
        if len(d.args) != n:
            raise ValueError(f"{f}: the step specification must have {n} components")
        res = [sx(c, f, "step") for c in d.args]
    else:
        res = [sx(d, f, "step")] * n
    concrete_check(res)
    for v in res:
        if num(v) <= 0:
            raise ValueError(f"{f}: steps must be positive")
    return res


def _offsets(o, steps, f):
    n = len(steps)
    if isinstance(o, Symbol) and o.name == "Center":
        return [s / 2 for s in steps]
    if isinstance(o, Expr) and o.head == "Scaled" and len(o.args) == 1:
        k = o.args[0]
        ks = [sx(c, f, "offset") for c in k.args] if K.is_list(k) else [sx(k, f, "offset")] * n
        return [s * c for s, c in zip(steps, ks)]
    if K.is_list(o):
        if len(o.args) != n:
            raise ValueError(f"{f}: the offset must have {n} components")
        return [sx(c, f, "offset") for c in o.args]
    return [sx(o, f, "offset")] * n


def _coord_array(bounds, rest, f, inputs):
    n = len(bounds)
    if len(rest) > 2:
        raise ValueError(f"{f}: too many arguments")
    steps = _steps(rest[0], n, bounds, f) if rest else [sp.Integer(1)] * n
    offs = _offsets(rest[1], steps, f) if len(rest) == 2 else [sp.Integer(0)] * n
    axes = []
    for (lo, hi), s, o in zip(bounds, steps, offs):
        if num(hi) < num(lo):
            raise ValueError(f"{f}: each range must have min <= max")
        start = lo + o
        k = int(math.floor(num((hi - start) / s) + 1e-9)) if num(hi - start) >= 0 else -1
        if k > 10 ** 6:
            raise ValueError(f"{f}: the array would be too large")
        axes.append([start + i * s for i in range(k + 1)])

    def build(i, prefix):
        if i == n:
            return list(prefix)
        return [build(i + 1, prefix + [v]) for v in axes[i]]
    return out(build(0, []), *inputs)


@fn("CoordinateBoundsArray", area=AREA, status="added",
    notes="CoordinateBoundsArray[{{xmin, xmax}, {ymin, ymax}, ...}] gives the array of coordinates {x, y, ...} with "
          "unit steps from the minima (nested as Table[{x, y}, {x, ...}, {y, ...}]); a second argument d, {dx, dy, "
          "...} or Into[n] / Into[{nx, ...}] (n equal parts, n + 1 points) sets the steps, and a third argument "
          "Center, Scaled[s] or an explicit offset shifts the points (Center gives cell centres). Points never go "
          "past the maxima. Non-positive steps or reversed ranges raise ValueError. Check: exact (arithmetic "
          "progressions).")
def _coordinate_bounds_array(bounds: MatrixOf[Real], *rest: Expr) -> Expr:
    f = "CoordinateBoundsArray"
    if not (K.is_list(bounds) and bounds.args and all(K.is_list(r) and len(r.args) == 2 for r in bounds.args)):
        if isinstance(bounds, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: bounds must be {{{{xmin, xmax}}, ...}}, not {K_input(bounds)}")
    b = [(sx(r.args[0], f, "bound"), sx(r.args[1], f, "bound")) for r in bounds.args]
    concrete_check([c for r in b for c in r])
    return _coord_array(b, rest, f, (bounds,) + rest)


@fn("CoordinateBoundingBoxArray", area=AREA, status="added",
    notes="CoordinateBoundingBoxArray[{pmin, pmax}] gives the array of coordinates of the box with corners pmin and "
          "pmax with unit steps; second and third arguments as in CoordinateBoundsArray (steps / Into[n]; offset "
          "Center, Scaled or explicit). Corners of different dimension raise ValueError. Check: exact (same "
          "construction as CoordinateBoundsArray).")
def _coordinate_bbox_array(corners: MatrixOf[Real], *rest: Expr) -> Expr:
    f = "CoordinateBoundingBoxArray"
    if not (K.is_list(corners) and len(corners.args) == 2 and all(K.is_list(p) for p in corners.args)):
        if isinstance(corners, Symbol):
            raise Unevaluated()
        raise TypeError(f"{f}: expected {{pmin, pmax}}, not {K_input(corners)}")
    p, q = corners.args
    if len(p.args) != len(q.args):
        raise ValueError(f"{f}: the corners must have the same dimension")
    b = [(sx(a, f, "corner"), sx(c, f, "corner")) for a, c in zip(p.args, q.args)]
    concrete_check([c for r in b for c in r])
    return _coord_array(b, rest, f, (corners,) + rest)
