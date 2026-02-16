"""Tests for physics units."""

import pytest
from mikoshilang.units import Quantity, UnitConvert, NUMERIC_CONSTANTS


class TestQuantityCreation:
    def test_basic(self):
        q = Quantity(9.8, "m/s^2")
        assert q.value == 9.8
        assert q.unit_name == "m/s^2"

    def test_repr(self):
        q = Quantity(9.8, "m/s^2")
        assert "9.8" in repr(q)

    def test_integer_repr(self):
        q = Quantity(3, "s")
        assert "3" in repr(q)


class TestArithmetic:
    def test_multiply_quantities(self):
        q = Quantity(9.8, "m/s^2")
        t = Quantity(3, "s")
        v = q * t
        assert abs(v.value - 29.4) < 0.01
        assert v.unit_name == "m/s"

    def test_scalar_multiply(self):
        q = Quantity(5, "m")
        r = q * 2
        assert r.value == 10
        assert r.unit_name == "m"

    def test_scalar_rmultiply(self):
        q = Quantity(5, "m")
        r = 3 * q
        assert r.value == 15

    def test_add_same_units(self):
        a = Quantity(5, "m")
        b = Quantity(3, "m")
        c = a + b
        assert c.value == 8

    def test_add_incompatible(self):
        with pytest.raises(ValueError):
            Quantity(5, "m") + Quantity(3, "s")

    def test_subtract(self):
        a = Quantity(10, "m")
        b = Quantity(3, "m")
        c = a - b
        assert c.value == 7

    def test_divide_quantities(self):
        d = Quantity(100, "m")
        t = Quantity(10, "s")
        v = d / t
        assert abs(v.value - 10) < 0.01

    def test_divide_scalar(self):
        q = Quantity(10, "m")
        r = q / 2
        assert r.value == 5


class TestConversion:
    def test_cm_to_m(self):
        q = UnitConvert(Quantity(100, "cm"), "m")
        assert abs(q.value - 1.0) < 1e-10
        assert q.unit_name == "m"

    def test_kg_to_lb(self):
        q = UnitConvert(Quantity(72, "kg"), "lb")
        assert abs(q.value - 158.73) < 0.1

    def test_km_to_m(self):
        q = UnitConvert(Quantity(1, "km"), "m")
        assert abs(q.value - 1000) < 1e-10

    def test_incompatible_conversion(self):
        with pytest.raises(ValueError):
            UnitConvert(Quantity(5, "m"), "s")

    def test_kj_to_j(self):
        q = UnitConvert(Quantity(1, "kJ"), "J")
        assert abs(q.value - 1000) < 1e-10

    def test_hours_to_seconds(self):
        q = UnitConvert(Quantity(1, "h"), "s")
        assert abs(q.value - 3600) < 1e-10


class TestConstants:
    def test_speed_of_light(self):
        assert NUMERIC_CONSTANTS["SpeedOfLight"] == 299792458.0

    def test_avogadro(self):
        assert abs(NUMERIC_CONSTANTS["AvogadroNumber"] - 6.022e23) < 1e20

    def test_planck(self):
        assert abs(NUMERIC_CONSTANTS["PlanckConstant"] - 6.626e-34) < 1e-36

    def test_boltzmann(self):
        assert abs(NUMERIC_CONSTANTS["BoltzmannConstant"] - 1.381e-23) < 1e-25

    def test_elementary_charge(self):
        assert abs(NUMERIC_CONSTANTS["ElementaryCharge"] - 1.602e-19) < 1e-21
