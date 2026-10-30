"""Physical constants, physics formulas, elementary geometry and finance.

Formulas are exact for exact input.  Physical constants are CODATA 2018 values
(exact where the 2019 SI defines them) and are returned as machine numbers; a
formula that uses a measured constant therefore returns a machine number.
Angles in the physics/geometry helpers are in degrees (MikoshiLang
convention), converted exactly."""
import sympy as sp

from ..expr import Expr, Symbol
from .core import (Unevaluated, fn, error, record, S, R, L, rule, numeric_out, sym, number, is_list, int_arg,
                   nonneg_int, pos_int, list_arg, has_float)

PHYS, GEO, FIN, CONST = "physics", "geometry", "finance", "constants"

_C = {   # CODATA 2018 / SI 2019 / IAU 2012-2015
    "SpeedOfLight": ("299792458", "exact (SI)"),
    "PlanckConstant": ("6.62607015e-34", "exact (SI)"),
    "ReducedPlanckConstant": ("1.054571817e-34", "h/(2 Pi), CODATA 2018"),
    "GravitationalConstant": ("6.67430e-11", "CODATA 2018"),
    "BoltzmannConstant": ("1.380649e-23", "exact (SI)"),
    "AvogadroNumber": ("6.02214076e23", "exact (SI)"),
    "ElementaryCharge": ("1.602176634e-19", "exact (SI)"),
    "ElectronMass": ("9.1093837015e-31", "CODATA 2018"),
    "ProtonMass": ("1.67262192369e-27", "CODATA 2018"),
    "NeutronMass": ("1.67492749804e-27", "CODATA 2018"),
    "FineStructureConstant": ("7.2973525693e-3", "CODATA 2018"),
    "RydbergConstant": ("10973731.568160", "CODATA 2018 (1/m)"),
    "StefanBoltzmannConstant": ("5.670374419e-8", "CODATA 2018 (exact from SI constants to the digits given)"),
    "GasConstant": ("8.314462618", "N_A k (CODATA 2018)"),
    "VacuumPermittivity": ("8.8541878128e-12", "CODATA 2018"),
    "VacuumPermeability": ("1.25663706212e-6", "CODATA 2018"),
    "CoulombConstant": ("8.9875517923e9", "1/(4 Pi epsilon0), CODATA 2018"),
    "FaradayConstant": ("96485.33212", "N_A e (CODATA 2018)"),
    "AtomicMassUnit": ("1.66053906660e-27", "CODATA 2018"),
    "BohrMagneton": ("9.2740100783e-24", "CODATA 2018"),
    "StandardGravity": ("9.80665", "exact (CGPM)"),
    "StandardAtmosphere": ("101325", "exact (Pa)"),
    "WienDisplacement": ("2.897771955e-3", "Wien displacement constant b (m K), CODATA 2018"),
    "WienDisplacementConstant": ("2.897771955e-3", "CODATA 2018 (m K)"),
    "MolarVolumeIdealGas": ("22.71095464e-3", "at 273.15 K and 100 kPa, CODATA 2018 (m^3/mol)"),
    "HubbleConstant": ("67.4", "Planck 2018 (km/s/Mpc; measured, uncertain)"),
    "SolarMass": ("1.98847e30", "IAU 2015 / CODATA (kg)"),
    "EarthMass": ("5.9722e24", "IAU 2015 nominal (kg)"),
    "EarthRadius": ("6.371e6", "mean radius (m)"),
    "LightYear": ("9460730472580800", "exact: Julian year x c (m)"),
    "Parsec": ("3.0856775814913673e16", "648000/Pi au (IAU 2015) (m)"),
    "BohrRadius": ("5.29177210903e-11", "CODATA 2018 (m)"),
}

_FIXED_CONST = {"LightYear": "was 9.461e15 (rounded to 4 digits)", "Parsec": "was 3.086e16 (rounded to 4 digits)",
                "SolarMass": "was 1.989e30", "EarthMass": "was 5.972e24",
                "WienDisplacementConstant": "only a 1-argument form existed, ignoring its argument",
                "MolarVolumeIdealGas": "only a 1-argument form existed, ignoring its argument"}


def _cval(name):
    return float(_C[name][0])


def _const(name):
    notes = _C[name][1] + "; the legacy 1-argument form, which ignored its argument, now raises an arity error"
    status = "fixed"
    if name in _FIXED_CONST:
        notes = _C[name][1] + "; " + _FIXED_CONST[name]
    @fn(name, area=CONST, status=status, notes=notes)
    def impl():
        return _cval(name)
    return impl


for _n in _C:
    if _n != "BohrRadius":
        _const(_n)


@fn("PhysicalConstant", area=CONST, status="fixed",
    notes="PhysicalConstant[\"name\"] for the constants above; unknown names raise (legacy returned an Error expression)")
def _physconst(name):
    if not isinstance(name, str):
        if isinstance(name, Symbol) and name.name in _C:
            return _cval(name.name)
        raise ValueError("PhysicalConstant: argument must be a constant name string")
    if name not in _C:
        raise ValueError(f"PhysicalConstant: unknown constant {name!r}")
    return _cval(name)


def _x(v, f):
    return sym(v, f)


def _deg(a, f):
    return _x(a, f) * sp.pi / 180


def _num(v, *inputs, measured=False):
    """Exact unless an input is inexact or a measured constant was used."""
    if measured:
        return R(sp.N(v, 15)) if not sp.sympify(v).free_symbols else R(sp.sympify(v).evalf(15))
    return numeric_out(v, *inputs)


def _nonzero(v, f, what):
    if v.is_number and v == 0:
        raise ValueError(f"{f}: {what} must be non-zero")
    return v


def _formula(name, area, nargs, body, notes, defaults=None, measured=False, status="fixed"):
    """Register formula `body(*sympy_args)`; defaults: list of (name_of_constant or value) for optional args."""
    defaults = defaults or []

    @fn(name, area=area, status=status, notes=notes)
    def impl(*args):
        lo, hi = nargs, nargs + len(defaults)
        if not lo <= len(args) <= hi:
            raise ValueError(f"{name} called with {len(args)} argument(s); expected {lo}" +
                             (f"-{hi}" if hi > lo else ""))
        vals = [_x(a, name) for a in args]
        used_measured = measured
        for i in range(len(args), hi):
            d = defaults[i - nargs]
            if isinstance(d, str):
                vals.append(sp.Float(_C[d][0], 15))
                used_measured = True
            else:
                vals.append(sp.sympify(d))
        return _num(body(*vals, fname=name), *args, measured=used_measured)
    return impl


NUM = "exact for exact input (legacy converted to floats)"
CONSTS = "uses CODATA values by default (legacy used rounded defaults such as G = 6.674e-11, c = 3e8, k = 8.99e9, h = 6.626e-34)"
GRAV = "default g = 9.80665 (standard gravity; legacy used 9.81)"


def _div(a, b, f, what="denominator"):
    return a / _nonzero(b, f, what)


_formula("Velocity", PHYS, 2, lambda d, t, fname: _div(d, t, fname, "time"), "displacement/time, " + NUM)
_formula("Acceleration", PHYS, 2, lambda dv, t, fname: _div(dv, t, fname, "time"), "velocity change/time, " + NUM)
_formula("Force", PHYS, 2, lambda m, a, fname: m * a, "m a, " + NUM)
_formula("Momentum", PHYS, 2, lambda m, v, fname: m * v, "m v, " + NUM)
_formula("Work", PHYS, 2, lambda F, d, fname: F * d, "F d, " + NUM)
_formula("MechanicalPower", PHYS, 2, lambda W, t, fname: _div(W, t, fname, "time"), "work/time, " + NUM)
_formula("Impulse", PHYS, 2, lambda F, t, fname: F * t, "F t, " + NUM)
_formula("KineticEnergy", PHYS, 2, lambda m, v, fname: m * v ** 2 / 2, "m v^2/2, " + NUM)
_formula("PotentialEnergy", PHYS, 2, lambda m, h, g, fname: m * g * h, "m g h; " + GRAV, defaults=["StandardGravity"])
_formula("CentripetalForce", PHYS, 3, lambda m, v, r, fname: _div(m * v ** 2, r, fname, "radius"), "m v^2/r, " + NUM)
_formula("EscapeVelocity", PHYS, 2, lambda M, r, G, fname: sp.sqrt(2 * G * M / _nonzero(r, fname, "radius")),
         "Sqrt[2 G M/r]; " + CONSTS, defaults=["GravitationalConstant"])
_formula("OrbitalVelocity", PHYS, 2, lambda M, r, G, fname: sp.sqrt(G * M / _nonzero(r, fname, "radius")),
         "Sqrt[G M/r]; " + CONSTS, defaults=["GravitationalConstant"])
_formula("FreeFallDistance", PHYS, 1, lambda t, g, fname: g * t ** 2 / 2, "g t^2/2; " + GRAV, defaults=["StandardGravity"])
_formula("ProjectileRange", PHYS, 2, lambda v, a, g, fname: v ** 2 * sp.sin(2 * a * sp.pi / 180) / g,
         "v^2 Sin[2 theta]/g, theta in degrees; " + GRAV, defaults=["StandardGravity"])
_formula("ProjectileMaxHeight", PHYS, 2, lambda v, a, g, fname: v ** 2 * sp.sin(a * sp.pi / 180) ** 2 / (2 * g),
         "v^2 Sin[theta]^2/(2 g), theta in degrees; " + GRAV, defaults=["StandardGravity"])
_formula("HeatCapacity", PHYS, 2, lambda Q, dT, fname: _div(Q, dT, fname, "temperature change"), "Q/dT, " + NUM)
_formula("ThermalExpansion", PHYS, 3, lambda L0, a, dT, fname: L0 * (1 + a * dT), "L0 (1 + alpha dT), " + NUM)
_formula("StefanBoltzmann", PHYS, 1, lambda T, eps, s, fname: eps * s * T ** 4,
         "emitted power per area eps sigma T^4; default sigma is CODATA 5.670374419e-8 (legacy 5.67e-8)",
         defaults=[1, "StefanBoltzmannConstant"])
_formula("CarnotEfficiency", PHYS, 2, lambda Th, Tc, fname: 1 - _div(Tc, Th, fname, "hot temperature"),
         "1 - Tc/Th, " + NUM)
_formula("EntropyChange", PHYS, 2, lambda Q, T, fname: _div(Q, T, fname, "temperature"), "Q/T, " + NUM)
_formula("CoulombForce", PHYS, 3, lambda q1, q2, r, k, fname: k * q1 * q2 / _nonzero(r, fname, "distance") ** 2,
         "k q1 q2/r^2; " + CONSTS, defaults=["CoulombConstant"])
_formula("ElectricField", PHYS, 2, lambda Q, r, k, fname: k * Q / _nonzero(r, fname, "distance") ** 2,
         "k Q/r^2; " + CONSTS, defaults=["CoulombConstant"])
_formula("ElectricPotential", PHYS, 2, lambda Q, r, k, fname: k * Q / _nonzero(r, fname, "distance"),
         "k Q/r; " + CONSTS, defaults=["CoulombConstant"])
_formula("Capacitance", PHYS, 2, lambda Q, V, fname: _div(Q, V, fname, "voltage"), "Q/V, " + NUM)
_formula("CapacitorEnergy", PHYS, 2, lambda C, V, fname: C * V ** 2 / 2, "C V^2/2, " + NUM)
_formula("PowerElectric", PHYS, 2, lambda V, I, fname: V * I, "V I, " + NUM)
_formula("MagneticForce", PHYS, 3, lambda q, v, B, fname: q * v * B, "q v B (perpendicular), " + NUM)
_formula("MagneticFieldWire", PHYS, 2, lambda I, r, mu0, fname: mu0 * I / (2 * sp.pi * _nonzero(r, fname, "distance")),
         "mu0 I/(2 Pi r); default mu0 is CODATA 1.25663706212e-6", defaults=["VacuumPermeability"])
_formula("FaradayLaw", PHYS, 2, lambda dPhi, dt, fname: -_div(dPhi, dt, fname, "time"), "-dPhi/dt, " + NUM)
_formula("InductanceEnergy", PHYS, 2, lambda Lh, I, fname: Lh * I ** 2 / 2, "L I^2/2, " + NUM)
_formula("LCFrequency", PHYS, 2, lambda Lh, C, fname: 1 / (2 * sp.pi * sp.sqrt(Lh * C)), "1/(2 Pi Sqrt[L C]), " + NUM)
_formula("Magnification", PHYS, 2, lambda di, do, fname: -_div(di, do, fname, "object distance"), "-di/do, " + NUM)
_formula("PhotonEnergy", PHYS, 1, lambda lam, h, c, fname: h * c / _nonzero(lam, fname, "wavelength"),
         "h c/lambda; " + CONSTS, defaults=["PlanckConstant", "SpeedOfLight"])
_formula("DeBroglieWavelength", PHYS, 1, lambda p, h, fname: h / _nonzero(p, fname, "momentum"),
         "h/p; " + CONSTS, defaults=["PlanckConstant"])
_formula("WaveSpeed", PHYS, 2, lambda f, lam, fname: f * lam, "f lambda, " + NUM)
_formula("DopplerShift", PHYS, 2, lambda f0, vs, vo, c, fname: f0 * (c + vo) / _nonzero(c - vs, fname, "c - v_source"),
         "f0 (c + v_observer)/(c - v_source), source approaching for v_source > 0; default c = 343 m/s (sound in air)",
         defaults=[0, 343])
_formula("BeatFrequency", PHYS, 2, lambda f1, f2, fname: sp.Abs(f1 - f2), "|f1 - f2|, " + NUM)
_formula("ResonanceFrequencySpring", PHYS, 2, lambda k, m, fname: sp.sqrt(_div(k, m, fname, "mass")) / (2 * sp.pi),
         "Sqrt[k/m]/(2 Pi), " + NUM)
_formula("ResonanceFrequencyPendulum", PHYS, 1, lambda Lp, g, fname: sp.sqrt(_div(g, Lp, fname, "length")) / (2 * sp.pi),
         "Sqrt[g/L]/(2 Pi); " + GRAV, defaults=["StandardGravity"])
_formula("ComptonWavelength", PHYS, 2,
         lambda ang, lam, h, me, c, fname: lam + h / (me * c) * (1 - sp.cos(ang * sp.pi / 180)),
         "scattered wavelength lambda + h/(m_e c) (1 - Cos[theta]), theta in degrees; " + CONSTS,
         defaults=["PlanckConstant", "ElectronMass", "SpeedOfLight"])
_formula("RydbergFormula", PHYS, 2, lambda n1, n2, Rr, fname: Rr * (1 / _nonzero(n1, fname, "n1") ** 2 - 1 / _nonzero(n2, fname, "n2") ** 2),
         "1/lambda = R (1/n1^2 - 1/n2^2); default R is CODATA 10973731.568160 (legacy 1.097e7)",
         defaults=["RydbergConstant"])


def _gamma_factor(v, c, fname):
    if v.is_number and c.is_number and not abs(v) < c:
        raise ValueError(f"{fname}: speed must be less than c")
    return 1 / sp.sqrt(1 - (v / c) ** 2)


_formula("TimeDilation", PHYS, 2, lambda t0, v, c, fname: t0 * _gamma_factor(v, c, fname),
         "gamma t0; default c = 299792458 (legacy 3e8); |v| >= c raises", defaults=["SpeedOfLight"])
_formula("LengthContraction", PHYS, 2, lambda L0, v, c, fname: L0 / _gamma_factor(v, c, fname),
         "L0/gamma; default c = 299792458 (legacy 3e8)", defaults=["SpeedOfLight"])
_formula("RelativisticMass", PHYS, 2, lambda m0, v, c, fname: m0 * _gamma_factor(v, c, fname),
         "gamma m0; default c = 299792458 (legacy 3e8)", defaults=["SpeedOfLight"])
_formula("MassEnergyEquivalence", PHYS, 1, lambda m, c, fname: m * c ** 2,
         "m c^2; default c = 299792458 (legacy 3e8)", defaults=["SpeedOfLight"])
_formula("UncertaintyProduct", PHYS, 2, lambda dx, dp, hbar, fname: dx * dp / hbar,
         "dx dp/hbar (dimensionless, >= 1/2 by Heisenberg); default hbar is CODATA (legacy 1.055e-34)",
         defaults=["ReducedPlanckConstant"])


@fn("BohrRadius", area=CONST, status="fixed",
    notes="BohrRadius[] = a0 (CODATA 2018); BohrRadius[n] = n^2 a0 (radius of the n-th Bohr orbit) and "
          "BohrRadius[n, a0]; legacy BohrRadius[n] returned a0 because a constant rule ignoring its argument "
          "shadowed the formula")
def _bohr(n=None, a0=None):
    base = _cval("BohrRadius") if a0 is None else _x(a0, "BohrRadius")
    if n is None:
        return base
    k = _x(n, "BohrRadius")
    return _num(k ** 2 * base, n, measured=a0 is None)


@fn("BraggLaw", area=PHYS, status="fixed",
    notes="BraggLaw[d, theta] and [d, theta, n] give the wavelength 2 d Sin[theta]/n (theta in degrees); "
          "BraggLaw[d, theta, n, lambda] gives the Bragg angle in degrees for lambda (theta ignored); impossible "
          "angles raise; exact for exact input")
def _bragg(d, theta, n=1, lam=None):
    dd, nn = _x(d, "BraggLaw"), _x(n, "BraggLaw")
    _nonzero(nn, "BraggLaw", "order n")
    if lam is None:
        return numeric_out(2 * dd * sp.sin(_deg(theta, "BraggLaw")) / nn, d, theta, n)
    s = nn * _x(lam, "BraggLaw") / (2 * dd)
    if s.is_number and abs(s) > 1:
        raise ValueError("BraggLaw: no diffraction angle (n lambda > 2 d)")
    return numeric_out(sp.asin(s) * 180 / sp.pi, d, n, lam)


@fn("SnellsLaw", area=PHYS, status="fixed",
    notes="refraction angle in degrees ArcSin[n1 Sin[theta1]/n2]; total internal reflection raises (legacy returned an "
          "Error expression); exact for exact input")
def _snell(n1, th, n2):
    s = _x(n1, "SnellsLaw") * sp.sin(_deg(th, "SnellsLaw")) / _nonzero(_x(n2, "SnellsLaw"), "SnellsLaw", "n2")
    if s.is_number and abs(s) > 1:
        raise ValueError("SnellsLaw: total internal reflection (no refracted ray)")
    return numeric_out(sp.asin(s) * 180 / sp.pi, n1, th, n2)


# ─────────────────────────── geometry ───────────────────────────

def _g(name, nargs, body, notes):
    _formula(name, GEO, nargs, body, notes)


_g("CircleArea", 1, lambda r, fname: sp.pi * r ** 2, "Pi r^2 (exact)")
_g("CircleCircumference", 1, lambda r, fname: 2 * sp.pi * r, "2 Pi r (exact)")
_g("SphereVolume", 1, lambda r, fname: sp.Rational(4, 3) * sp.pi * r ** 3, "4/3 Pi r^3 (exact)")
_g("SphereSurfaceArea", 1, lambda r, fname: 4 * sp.pi * r ** 2, "4 Pi r^2 (exact)")
_g("CylinderVolume", 2, lambda r, h, fname: sp.pi * r ** 2 * h, "Pi r^2 h (exact)")
_g("CylinderSurfaceArea", 2, lambda r, h, fname: 2 * sp.pi * r * (r + h), "2 Pi r (r + h), " + NUM)
_g("ConeVolume", 2, lambda r, h, fname: sp.pi * r ** 2 * h / 3, "Pi r^2 h/3 (exact)")
_g("ConeSurfaceArea", 2, lambda r, h, fname: sp.pi * r * (r + sp.sqrt(r ** 2 + h ** 2)), "total area Pi r (r + slant), " + NUM)
_g("CubeVolume", 1, lambda a, fname: a ** 3, "a^3, " + NUM)
_g("CubeSurfaceArea", 1, lambda a, fname: 6 * a ** 2, "6 a^2, " + NUM)
_g("EllipsoidVolume", 3, lambda a, b, c, fname: sp.Rational(4, 3) * sp.pi * a * b * c, "4/3 Pi a b c, " + NUM)
_g("TorusVolume", 2, lambda Rr, r, fname: 2 * sp.pi ** 2 * Rr * r ** 2, "2 Pi^2 R r^2, " + NUM)
_g("TorusSurfaceArea", 2, lambda Rr, r, fname: 4 * sp.pi ** 2 * Rr * r, "4 Pi^2 R r, " + NUM)
_g("PrismVolume", 2, lambda A, h, fname: A * h, "base area x height, " + NUM)
_g("PyramidVolume", 2, lambda A, h, fname: A * h / 3, "base area x height/3, " + NUM)
_g("FrustumVolume", 3, lambda r1, r2, h, fname: sp.pi * h * (r1 ** 2 + r1 * r2 + r2 ** 2) / 3,
   "Pi h (r1^2 + r1 r2 + r2^2)/3, " + NUM)
_g("TetrahedronVolume", 1, lambda a, fname: a ** 3 / (6 * sp.sqrt(2)), "regular tetrahedron a^3/(6 Sqrt[2]), " + NUM)
_g("Hypotenuse", 2, lambda a, b, fname: sp.sqrt(a ** 2 + b ** 2), "Sqrt[a^2 + b^2], exact (legacy rounded floats)")


@fn("TriangleArea", area=GEO, status="fixed",
    notes="Heron's formula, exact; sides violating the triangle inequality raise (legacy returned a complex number "
          "or a math domain error)")
def _triangle(a, b, c):
    x, y, z = (_x(v, "TriangleArea") for v in (a, b, c))
    if all(v.is_number for v in (x, y, z)):
        if min(x, y, z) <= 0 or x + y <= z or x + z <= y or y + z <= x:
            raise ValueError("TriangleArea: sides do not form a triangle")
    s = (x + y + z) / 2
    return numeric_out(sp.sqrt(sp.expand(s * (s - x) * (s - y) * (s - z))), a, b, c)


@fn("RegularPolygonArea", area=GEO, status="fixed",
    notes="n s^2/(4 Tan[Pi/n]), exact; n must be an integer >= 3 (legacy truncated n and rounded)")
def _regpoly(n, s):
    k = int_arg(n, "RegularPolygonArea", "number of sides")
    if k < 3:
        raise ValueError("RegularPolygonArea: n must be at least 3")
    ss = _x(s, "RegularPolygonArea")
    return numeric_out(sp.simplify(k * ss ** 2 / (4 * sp.tan(sp.pi / k))), s)


def _points(x, fname, dim=None):
    pts = list_arg(x, fname, "point")
    v = [_x(c, fname) for c in pts]
    if dim is not None and len(v) != dim:
        raise ValueError(f"{fname}: points must be {dim}-dimensional")
    return v


def _dist(p, q, fname, dim=None):
    a, b = _points(p, fname, dim), _points(q, fname, dim)
    if len(a) != len(b):
        raise ValueError(f"{fname}: points must have the same dimension")
    return numeric_out(sp.sqrt(sp.Add(*((x - y) ** 2 for x, y in zip(a, b)))), p, q)


fn("Distance", area=GEO, status="fixed", notes="Euclidean distance, exact (legacy rounded floats and zip-truncated)")(
    lambda p, q: _dist(p, q, "Distance"))
fn("Distance3D", area=GEO, status="fixed", notes="Euclidean distance of 3D points, exact")(
    lambda p, q: _dist(p, q, "Distance3D", 3))


def _mid(p, q, fname, dim=None):
    a, b = _points(p, fname, dim), _points(q, fname, dim)
    if len(a) != len(b):
        raise ValueError(f"{fname}: points must have the same dimension")
    return L(*(numeric_out((x + y) / 2, p, q) for x, y in zip(a, b)))


fn("Midpoint", area=GEO, status="fixed", notes="exact midpoint (legacy rounded floats)")(lambda p, q: _mid(p, q, "Midpoint"))
fn("Midpoint3D", area=GEO, status="fixed", notes="exact midpoint of 3D points")(lambda p, q: _mid(p, q, "Midpoint3D", 3))


@fn("Slope", area=GEO, status="fixed",
    notes="(y2 - y1)/(x2 - x1), exact; a vertical line now raises (legacy returned +Infinity even for downward lines)")
def _slope(p, q):
    a, b = _points(p, "Slope", 2), _points(q, "Slope", 2)
    if sp.simplify(b[0] - a[0]) == 0:
        raise ValueError("Slope: the line is vertical (slope undefined)")
    return numeric_out((b[1] - a[1]) / (b[0] - a[0]), p, q)


@fn("PolygonArea", area=GEO, status="fixed", notes="shoelace formula, exact (legacy rounded floats)")
def _polyarea(pts):
    ps = [_points(p, "PolygonArea", 2) for p in list_arg(pts, "PolygonArea", min_len=3)]
    s = sp.Add(*(ps[i][0] * ps[(i + 1) % len(ps)][1] - ps[(i + 1) % len(ps)][0] * ps[i][1] for i in range(len(ps))))
    return numeric_out(sp.Abs(s) / 2, pts)


@fn("PlaneEquation", area=GEO, status="fixed",
    notes="n.({x, y, z} - p) == 0 expanded, exact (legacy converted to floats)")
def _plane(p, n):
    a, nn = _points(p, "PlaneEquation", 3), _points(n, "PlaneEquation", 3)
    x, y, z = sp.symbols("x y z")
    lhs = sp.expand(nn[0] * (x - a[0]) + nn[1] * (y - a[1]) + nn[2] * (z - a[2]))
    return Expr("Equal", numeric_out(lhs, p, n), 0)


@fn("PointToPlaneDistance", area=GEO, status="fixed",
    notes="|a x0 + b y0 + c z0 + d|/Sqrt[a^2 + b^2 + c^2] for plane {a, b, c, d}, exact")
def _p2plane(p, plane):
    a = _points(p, "PointToPlaneDistance", 3)
    c = _points(plane, "PointToPlaneDistance", 4)
    den = sp.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)
    if den == 0:
        raise ValueError("PointToPlaneDistance: plane normal must be non-zero")
    return numeric_out(sp.Abs(c[0] * a[0] + c[1] * a[1] + c[2] * a[2] + c[3]) / den, p, plane)


@fn("LineSphereIntersection", area=GEO, status="fixed",
    notes="intersection points of p + t d with the sphere |x - c| = r, exact; tangency gives one point; zero "
          "direction raises (legacy divided by zero)")
def _linesphere(p, d, c, r):
    P, D, C = _points(p, "LineSphereIntersection", 3), _points(d, "LineSphereIntersection", 3), _points(c, "LineSphereIntersection", 3)
    rr = _x(r, "LineSphereIntersection")
    a = sp.Add(*(v ** 2 for v in D))
    if a == 0:
        raise ValueError("LineSphereIntersection: direction must be non-zero")
    b = 2 * sp.Add(*((P[i] - C[i]) * D[i] for i in range(3)))
    cc = sp.Add(*((P[i] - C[i]) ** 2 for i in range(3))) - rr ** 2
    disc = sp.simplify(b ** 2 - 4 * a * cc)
    if disc.is_number and disc < 0:
        return L()
    ts = [(-b + sp.sqrt(disc)) / (2 * a)] if disc == 0 else [(-b + sp.sqrt(disc)) / (2 * a), (-b - sp.sqrt(disc)) / (2 * a)]
    return L(*(L(*(numeric_out(sp.simplify(P[i] + t * D[i]), p, d, c, r) for i in range(3))) for t in ts))


# ─────────────────────────── finance ───────────────────────────

FN = "exact for exact input; legacy rounded to cents/6 digits or used floats"


def _f(name, nargs, body, notes):
    _formula(name, FIN, nargs, body, notes)


def _annuity_factor(r, n):
    return n if r == 0 else (1 - (1 + r) ** -n) / r


_f("CompoundInterest", 3, lambda P, r, t, fname: P * (1 + r) ** t, "P (1 + r)^t (amount); " + FN)
_f("FutureValue", 3, lambda pv, r, n, fname: pv * (1 + r) ** n, "PV (1 + r)^n; " + FN)
_f("PresentValue", 3, lambda fv, r, n, fname: fv / (1 + r) ** n, "FV/(1 + r)^n; " + FN)
_f("ContinuousCompounding", 3, lambda P, r, t, fname: P * sp.exp(r * t), "P E^(r t); " + FN)
_f("AnnuityPV", 3, lambda pmt, r, n, fname: pmt * _annuity_factor(r, n), "PMT (1 - (1 + r)^-n)/r (rate 0 -> PMT n); " + FN)
_f("AnnuityFV", 3, lambda pmt, r, n, fname: pmt * (n if r == 0 else ((1 + r) ** n - 1) / r),
   "PMT ((1 + r)^n - 1)/r (rate 0 -> PMT n); " + FN)
_f("LoanPayment", 3, lambda P, r, n, fname: _div(P, _annuity_factor(r, n), fname, "number of periods"),
   "P r (1 + r)^n/((1 + r)^n - 1) (rate 0 -> P/n; legacy divided by zero); " + FN)
_f("PMT", 3, lambda r, n, pv, fname: _div(pv, _annuity_factor(r, n), fname, "number of periods"),
   "PMT[rate, periods, pv] payment per period; " + FN)
_f("RemainingBalance", 4,
   lambda P, r, n, k, fname: P * (1 - k / n) if r == 0 else P * ((1 + r) ** n - (1 + r) ** k) / ((1 + r) ** n - 1),
   "balance after k payments; " + FN)
_f("PerpetuityPV", 2, lambda pmt, r, fname: _div(pmt, r, fname, "rate"), "PMT/r; " + FN)
_f("GrowingPerpetuityPV", 3, lambda pmt, r, g, fname: _div(pmt, r - g, fname, "r - g"), "PMT/(r - g); " + FN)
_f("GrowingAnnuityPV", 4,
   lambda pmt, r, g, n, fname: pmt * n / (1 + r) if sp.simplify(r - g) == 0 else pmt * (1 - ((1 + g) / (1 + r)) ** n) / (r - g),
   "PMT (1 - ((1 + g)/(1 + r))^n)/(r - g); " + FN)
_f("EffectiveRate", 2, lambda r, m, fname: (1 + r / _nonzero(m, fname, "compounding periods")) ** m - 1,
   "(1 + r/m)^m - 1; " + FN)
_f("APRtoEAR", 2, lambda r, m, fname: (1 + r / _nonzero(m, fname, "compounding periods")) ** m - 1,
   "alias of EffectiveRate; " + FN)
_f("CAGR", 3, lambda a, b, t, fname: (b / _nonzero(a, fname, "start value")) ** (1 / _nonzero(t, fname, "years")) - 1,
   "(end/start)^(1/years) - 1; " + FN)
_f("DoublingTime", 1, lambda r, fname: sp.log(2) / sp.log(1 + _nonzero(r, fname, "rate")), "Log[2]/Log[1 + r]; " + FN)
_f("BondYieldApprox", 4, lambda P, F, c, n, fname: (F * c + (F - P) / _nonzero(n, fname, "years")) / ((F + P) / 2),
   "approximate yield to maturity (C + (F - P)/n)/((F + P)/2); " + FN)


@fn("BondPrice", area=FIN, status="fixed",
    notes="sum of discounted annual coupons plus discounted face value; years must be a positive integer (legacy "
          "truncated with int())")
def _bond(F, c, y, n):
    FF, cc, yy = _x(F, "BondPrice"), _x(c, "BondPrice"), _x(y, "BondPrice")
    k = pos_int(n, "BondPrice", "years")
    C = FF * cc
    v = sp.Add(*(C / (1 + yy) ** t for t in range(1, k + 1))) + FF / (1 + yy) ** k
    return numeric_out(sp.simplify(v), F, c, y)


@fn("NPV", area=FIN, status="fixed",
    notes="NPV[r, {cf0, cf1, ...}] = Sum[cf_t/(1 + r)^t] with cf0 undiscounted at t = 0; " + FN)
def _npv(r, cfs):
    rr = _x(r, "NPV")
    vals = [_x(v, "NPV") for v in list_arg(cfs, "NPV", min_len=1)]
    return numeric_out(sp.Add(*(v / (1 + rr) ** t for t, v in enumerate(vals))), r, cfs)


@fn("IRR", area=FIN, status="fixed",
    notes="the rate r > -1 with NPV = 0, found exactly from the cash-flow polynomial and returned as a machine number; "
          "no root or several roots raise (legacy ran Newton from 10% and returned whatever it reached, rounded)")
def _irr(cfs):
    vals = [number(v, "IRR", "cash flow") for v in list_arg(cfs, "IRR", min_len=2)]
    x = sp.Symbol("x")   # x = 1/(1 + r)
    poly = sp.Poly(sum(v * x ** t for t, v in enumerate(vals)), x)
    roots = [rt for rt in sp.real_roots(poly) if rt > 0] if not any(isinstance(v, sp.Float) for v in vals) else \
        [complex(rt).real for rt in sp.Poly(poly.as_expr().evalf(), x).nroots() if abs(complex(rt).imag) < 1e-12 and complex(rt).real > 0]
    rates = sorted({float(sp.N(1 / rt - 1, 17)) for rt in roots})
    if not rates:
        raise ValueError("IRR: the cash flows have no internal rate of return")
    if len(rates) > 1:
        raise ValueError(f"IRR: the cash flows have several internal rates of return {rates}")
    return rates[0]
