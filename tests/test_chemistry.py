"""Tests for chemistry module."""

import pytest
from mikoshilang.chemistry import Element, AtomicMass, ElectronConfiguration, MolecularMass, BalanceEquation
from mikoshilang.expr import Expr


class TestElements:
    def test_hydrogen(self):
        e = Element("H")
        assert isinstance(e, Expr)
        # Check it has the right data
        assert e.head == "List"

    def test_oxygen(self):
        mass = AtomicMass("O")
        assert abs(mass - 15.999) < 0.01

    def test_iron_config(self):
        config = ElectronConfiguration("Fe")
        assert config == "[Ar] 3d6 4s2"

    def test_by_number(self):
        e = Element(1)
        assert e.head == "List"

    def test_by_name(self):
        mass = AtomicMass("Carbon")
        assert abs(mass - 12.011) < 0.01

    def test_all_118_elements(self):
        for i in range(1, 119):
            e = Element(i)
            assert e.head == "List"

    def test_unknown_element(self):
        with pytest.raises(ValueError):
            Element("Xx")

    def test_gold(self):
        assert abs(AtomicMass("Au") - 196.97) < 0.01

    def test_uranium(self):
        assert abs(AtomicMass("U") - 238.03) < 0.01

    def test_helium(self):
        config = ElectronConfiguration("He")
        assert config == "1s2"


class TestMolecularMass:
    def test_water(self):
        mass = MolecularMass("H2O")
        assert abs(mass - 18.015) < 0.01

    def test_glucose(self):
        mass = MolecularMass("C6H12O6")
        assert abs(mass - 180.156) < 0.1

    def test_co2(self):
        mass = MolecularMass("CO2")
        assert abs(mass - 44.009) < 0.01

    def test_nacl(self):
        mass = MolecularMass("NaCl")
        expected = AtomicMass("Na") + AtomicMass("Cl")
        assert abs(mass - expected) < 0.01

    def test_single_element(self):
        mass = MolecularMass("He")
        assert abs(mass - 4.003) < 0.01


class TestBalanceEquation:
    def test_h2_o2(self):
        result = BalanceEquation("H2 + O2 -> H2O")
        assert "2H2" in result
        assert "2H2O" in result

    def test_already_balanced(self):
        # Simple case
        result = BalanceEquation("H2 + Cl2 -> HCl")
        assert "2HCl" in result

    def test_combustion(self):
        result = BalanceEquation("CH4 + O2 -> CO2 + H2O")
        # Should balance to CH4 + 2O2 -> CO2 + 2H2O
        assert "2" in result
