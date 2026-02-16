"""Physics units and quantities for MikoshiLang.

Supports unit arithmetic, conversion, and physical constants.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import re
from .expr import Expr, Symbol


# ── Unit dimension representation ──
# Each unit is stored as a dict of base dimension -> exponent
# Base dimensions: length(m), mass(kg), time(s), current(A), temperature(K), amount(mol), luminosity(cd)

_BASE_DIMS = ("m", "kg", "s", "A", "K", "mol", "cd")


class Unit:
    """A physical unit with magnitude and dimension."""
    __slots__ = ("factor", "dims", "offset")

    def __init__(self, factor: float, dims: Dict[str, int], offset: float = 0.0):
        self.factor = factor  # conversion factor to SI base
        self.dims = {k: v for k, v in dims.items() if v != 0}
        self.offset = offset  # for temperature conversions

    def __eq__(self, other):
        if isinstance(other, Unit):
            return abs(self.factor - other.factor) < 1e-12 and self.dims == other.dims
        return False

    def __repr__(self):
        return f"Unit({self.factor}, {self.dims})"


# ── Unit definitions ──

_UNITS: Dict[str, Unit] = {}


def _def(name: str, factor: float, dims: Dict[str, int], offset: float = 0.0):
    _UNITS[name] = Unit(factor, dims, offset)


# Length
_def("m", 1.0, {"m": 1})
_def("cm", 0.01, {"m": 1})
_def("mm", 0.001, {"m": 1})
_def("km", 1000.0, {"m": 1})
_def("in", 0.0254, {"m": 1})
_def("ft", 0.3048, {"m": 1})
_def("yd", 0.9144, {"m": 1})
_def("mi", 1609.344, {"m": 1})

# Mass
_def("kg", 1.0, {"kg": 1})
_def("g", 0.001, {"kg": 1})
_def("mg", 1e-6, {"kg": 1})
_def("lb", 0.45359237, {"kg": 1})
_def("oz", 0.028349523125, {"kg": 1})

# Time
_def("s", 1.0, {"s": 1})
_def("ms", 0.001, {"s": 1})
_def("min", 60.0, {"s": 1})
_def("h", 3600.0, {"s": 1})
_def("day", 86400.0, {"s": 1})

# Speed
_def("m/s", 1.0, {"m": 1, "s": -1})
_def("km/h", 1000.0 / 3600.0, {"m": 1, "s": -1})
_def("mph", 1609.344 / 3600.0, {"m": 1, "s": -1})

# Acceleration
_def("m/s^2", 1.0, {"m": 1, "s": -2})

# Force
_def("N", 1.0, {"kg": 1, "m": 1, "s": -2})
_def("lbf", 4.4482216152605, {"kg": 1, "m": 1, "s": -2})

# Energy
_def("J", 1.0, {"kg": 1, "m": 2, "s": -2})
_def("kJ", 1000.0, {"kg": 1, "m": 2, "s": -2})
_def("cal", 4.184, {"kg": 1, "m": 2, "s": -2})
_def("kcal", 4184.0, {"kg": 1, "m": 2, "s": -2})
_def("eV", 1.602176634e-19, {"kg": 1, "m": 2, "s": -2})
_def("kWh", 3.6e6, {"kg": 1, "m": 2, "s": -2})

# Power
_def("W", 1.0, {"kg": 1, "m": 2, "s": -3})
_def("kW", 1000.0, {"kg": 1, "m": 2, "s": -3})
_def("hp", 745.69987158227022, {"kg": 1, "m": 2, "s": -3})

# Pressure
_def("Pa", 1.0, {"kg": 1, "m": -1, "s": -2})
_def("kPa", 1000.0, {"kg": 1, "m": -1, "s": -2})
_def("atm", 101325.0, {"kg": 1, "m": -1, "s": -2})
_def("bar", 100000.0, {"kg": 1, "m": -1, "s": -2})
_def("psi", 6894.757293168, {"kg": 1, "m": -1, "s": -2})

# Temperature (special handling for offset)
_def("K", 1.0, {"K": 1})
_def("C", 1.0, {"K": 1}, offset=273.15)
_def("F", 5.0 / 9.0, {"K": 1}, offset=255.3722222222)  # (F+459.67)*5/9

# Electric
_def("A", 1.0, {"A": 1})
_def("V", 1.0, {"kg": 1, "m": 2, "s": -3, "A": -1})
_def("ohm", 1.0, {"kg": 1, "m": 2, "s": -3, "A": -2})
# Farad (capacitance) - use "Farad" to avoid conflict with "F" temperature
_def("Farad", 1.0, {"kg": -1, "m": -2, "s": 4, "A": 2})
_def("H", 1.0, {"kg": 1, "m": 2, "s": -2, "A": -2})
_def("Coulomb", 1.0, {"s": 1, "A": 1})
_def("Hz", 1.0, {"s": -1})


def _get_unit(name: str) -> Unit:
    if name in _UNITS:
        return _UNITS[name]
    raise ValueError(f"Unknown unit: {name}")


def _dims_str(dims: Dict[str, int]) -> str:
    """Convert dimension dict to string representation."""
    parts = []
    for d in _BASE_DIMS:
        exp = dims.get(d, 0)
        if exp == 1:
            parts.append(d)
        elif exp != 0:
            parts.append(f"{d}^{exp}")
    return "*".join(parts) if parts else "dimensionless"


def _find_unit_name(factor: float, dims: Dict[str, int]) -> Optional[str]:
    """Find a named unit matching the given factor and dims."""
    clean_dims = {k: v for k, v in dims.items() if v != 0}
    for name, unit in _UNITS.items():
        if unit.dims == clean_dims and abs(unit.factor - factor) < 1e-10 * max(abs(factor), 1):
            return name
    # Try just matching dims and return the SI base
    for name, unit in _UNITS.items():
        if unit.dims == clean_dims and unit.factor == 1.0:
            return name
    return _dims_str(clean_dims)


# ── Quantity class ──

class Quantity:
    """A physical quantity: value with a unit."""
    __slots__ = ("value", "unit_name", "_unit")

    def __init__(self, value: float, unit_name: str):
        self.value = float(value)
        self.unit_name = unit_name
        self._unit = _get_unit(unit_name)

    def _to_si(self) -> float:
        if self._unit.offset:
            return self.value * self._unit.factor + self._unit.offset
        return self.value * self._unit.factor

    def _si_dims(self) -> Dict[str, int]:
        return dict(self._unit.dims)

    def __repr__(self):
        v = self.value
        if v == int(v) and abs(v) < 1e15:
            return f"Quantity[{int(v)}, \"{self.unit_name}\"]"
        return f"Quantity[{v:g}, \"{self.unit_name}\"]"

    def __eq__(self, other):
        if isinstance(other, Quantity):
            return abs(self._to_si() - other._to_si()) < 1e-10 and self._si_dims() == other._si_dims()
        return False

    def __add__(self, other):
        if isinstance(other, Quantity):
            if self._si_dims() != other._si_dims():
                raise ValueError(f"Cannot add {self.unit_name} and {other.unit_name}: incompatible dimensions")
            si_val = self._to_si() + other._to_si()
            return Quantity(si_val / self._unit.factor, self.unit_name)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Quantity):
            if self._si_dims() != other._si_dims():
                raise ValueError(f"Cannot subtract {self.unit_name} from {other.unit_name}: incompatible dimensions")
            si_val = self._to_si() - other._to_si()
            return Quantity(si_val / self._unit.factor, self.unit_name)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Quantity):
            new_val_si = self._to_si() * other._to_si()
            new_dims = dict(self._si_dims())
            for d, e in other._si_dims().items():
                new_dims[d] = new_dims.get(d, 0) + e
            new_dims = {k: v for k, v in new_dims.items() if v != 0}
            unit_name = _find_unit_name(1.0, new_dims)
            return Quantity(new_val_si, unit_name)
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit_name)
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Quantity(self.value * other, self.unit_name)
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Quantity):
            new_val_si = self._to_si() / other._to_si()
            new_dims = dict(self._si_dims())
            for d, e in other._si_dims().items():
                new_dims[d] = new_dims.get(d, 0) - e
            new_dims = {k: v for k, v in new_dims.items() if v != 0}
            unit_name = _find_unit_name(1.0, new_dims)
            return Quantity(new_val_si, unit_name)
        if isinstance(other, (int, float)):
            return Quantity(self.value / other, self.unit_name)
        return NotImplemented


def UnitConvert(qty: Quantity, target_unit: str) -> Quantity:
    """Convert a Quantity to a different unit."""
    target = _get_unit(target_unit)
    if qty._si_dims() != target.dims:
        raise ValueError(f"Cannot convert {qty.unit_name} to {target_unit}: incompatible dimensions")
    si_val = qty._to_si()
    if target.offset:
        new_val = (si_val - target.offset) / target.factor
    else:
        new_val = si_val / target.factor
    return Quantity(round(new_val, 10), target_unit)


# ── Physical constants ──

CONSTANTS = {
    "SpeedOfLight": Quantity(299792458, "m/s"),
    "GravitationalConstant": Quantity(6.67430e-11, "m^3*kg^-1*s^-2") if False else None,
    "PlanckConstant": Quantity(6.62607015e-34, "J"),  # J*s approximation
    "BoltzmannConstant": Quantity(1.380649e-23, "J"),  # J/K approximation
    "ElementaryCharge": Quantity(1.602176634e-19, "Coulomb"),
}

# Constants that are just numbers
NUMERIC_CONSTANTS = {
    "SpeedOfLight": 299792458.0,
    "GravitationalConstant": 6.67430e-11,
    "PlanckConstant": 6.62607015e-34,
    "BoltzmannConstant": 1.380649e-23,
    "AvogadroNumber": 6.02214076e23,
    "ElementaryCharge": 1.602176634e-19,
}
