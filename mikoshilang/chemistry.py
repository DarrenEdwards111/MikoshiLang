"""Chemistry module for MikoshiLang — elements, molecular mass, equation balancing."""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional, Tuple
from .expr import Expr, Symbol


# ── Element data: (number, symbol, name, mass, electron_config, electronegativity, category) ──

_ELEMENTS_DATA = [
    (1, "H", "Hydrogen", 1.008, "1s1", 2.20, "nonmetal"),
    (2, "He", "Helium", 4.003, "1s2", None, "noble gas"),
    (3, "Li", "Lithium", 6.941, "[He] 2s1", 0.98, "alkali metal"),
    (4, "Be", "Beryllium", 9.012, "[He] 2s2", 1.57, "alkaline earth metal"),
    (5, "B", "Boron", 10.81, "[He] 2s2 2p1", 2.04, "metalloid"),
    (6, "C", "Carbon", 12.011, "[He] 2s2 2p2", 2.55, "nonmetal"),
    (7, "N", "Nitrogen", 14.007, "[He] 2s2 2p3", 3.04, "nonmetal"),
    (8, "O", "Oxygen", 15.999, "[He] 2s2 2p4", 3.44, "nonmetal"),
    (9, "F", "Fluorine", 18.998, "[He] 2s2 2p5", 3.98, "halogen"),
    (10, "Ne", "Neon", 20.180, "[He] 2s2 2p6", None, "noble gas"),
    (11, "Na", "Sodium", 22.990, "[Ne] 3s1", 0.93, "alkali metal"),
    (12, "Mg", "Magnesium", 24.305, "[Ne] 3s2", 1.31, "alkaline earth metal"),
    (13, "Al", "Aluminium", 26.982, "[Ne] 3s2 3p1", 1.61, "post-transition metal"),
    (14, "Si", "Silicon", 28.086, "[Ne] 3s2 3p2", 1.90, "metalloid"),
    (15, "P", "Phosphorus", 30.974, "[Ne] 3s2 3p3", 2.19, "nonmetal"),
    (16, "S", "Sulfur", 32.06, "[Ne] 3s2 3p4", 2.58, "nonmetal"),
    (17, "Cl", "Chlorine", 35.45, "[Ne] 3s2 3p5", 3.16, "halogen"),
    (18, "Ar", "Argon", 39.948, "[Ne] 3s2 3p6", None, "noble gas"),
    (19, "K", "Potassium", 39.098, "[Ar] 4s1", 0.82, "alkali metal"),
    (20, "Ca", "Calcium", 40.078, "[Ar] 4s2", 1.00, "alkaline earth metal"),
    (21, "Sc", "Scandium", 44.956, "[Ar] 3d1 4s2", 1.36, "transition metal"),
    (22, "Ti", "Titanium", 47.867, "[Ar] 3d2 4s2", 1.54, "transition metal"),
    (23, "V", "Vanadium", 50.942, "[Ar] 3d3 4s2", 1.63, "transition metal"),
    (24, "Cr", "Chromium", 51.996, "[Ar] 3d5 4s1", 1.66, "transition metal"),
    (25, "Mn", "Manganese", 54.938, "[Ar] 3d5 4s2", 1.55, "transition metal"),
    (26, "Fe", "Iron", 55.845, "[Ar] 3d6 4s2", 1.83, "transition metal"),
    (27, "Co", "Cobalt", 58.933, "[Ar] 3d7 4s2", 1.88, "transition metal"),
    (28, "Ni", "Nickel", 58.693, "[Ar] 3d8 4s2", 1.91, "transition metal"),
    (29, "Cu", "Copper", 63.546, "[Ar] 3d10 4s1", 1.90, "transition metal"),
    (30, "Zn", "Zinc", 65.38, "[Ar] 3d10 4s2", 1.65, "transition metal"),
    (31, "Ga", "Gallium", 69.723, "[Ar] 3d10 4s2 4p1", 1.81, "post-transition metal"),
    (32, "Ge", "Germanium", 72.630, "[Ar] 3d10 4s2 4p2", 2.01, "metalloid"),
    (33, "As", "Arsenic", 74.922, "[Ar] 3d10 4s2 4p3", 2.18, "metalloid"),
    (34, "Se", "Selenium", 78.971, "[Ar] 3d10 4s2 4p4", 2.55, "nonmetal"),
    (35, "Br", "Bromine", 79.904, "[Ar] 3d10 4s2 4p5", 2.96, "halogen"),
    (36, "Kr", "Krypton", 83.798, "[Ar] 3d10 4s2 4p6", 3.00, "noble gas"),
    (37, "Rb", "Rubidium", 85.468, "[Kr] 5s1", 0.82, "alkali metal"),
    (38, "Sr", "Strontium", 87.62, "[Kr] 5s2", 0.95, "alkaline earth metal"),
    (39, "Y", "Yttrium", 88.906, "[Kr] 4d1 5s2", 1.22, "transition metal"),
    (40, "Zr", "Zirconium", 91.224, "[Kr] 4d2 5s2", 1.33, "transition metal"),
    (41, "Nb", "Niobium", 92.906, "[Kr] 4d4 5s1", 1.6, "transition metal"),
    (42, "Mo", "Molybdenum", 95.95, "[Kr] 4d5 5s1", 2.16, "transition metal"),
    (43, "Tc", "Technetium", 98.0, "[Kr] 4d5 5s2", 1.9, "transition metal"),
    (44, "Ru", "Ruthenium", 101.07, "[Kr] 4d7 5s1", 2.2, "transition metal"),
    (45, "Rh", "Rhodium", 102.91, "[Kr] 4d8 5s1", 2.28, "transition metal"),
    (46, "Pd", "Palladium", 106.42, "[Kr] 4d10", 2.20, "transition metal"),
    (47, "Ag", "Silver", 107.87, "[Kr] 4d10 5s1", 1.93, "transition metal"),
    (48, "Cd", "Cadmium", 112.41, "[Kr] 4d10 5s2", 1.69, "transition metal"),
    (49, "In", "Indium", 114.82, "[Kr] 4d10 5s2 5p1", 1.78, "post-transition metal"),
    (50, "Sn", "Tin", 118.71, "[Kr] 4d10 5s2 5p2", 1.96, "post-transition metal"),
    (51, "Sb", "Antimony", 121.76, "[Kr] 4d10 5s2 5p3", 2.05, "metalloid"),
    (52, "Te", "Tellurium", 127.60, "[Kr] 4d10 5s2 5p4", 2.1, "metalloid"),
    (53, "I", "Iodine", 126.90, "[Kr] 4d10 5s2 5p5", 2.66, "halogen"),
    (54, "Xe", "Xenon", 131.29, "[Kr] 4d10 5s2 5p6", 2.60, "noble gas"),
    (55, "Cs", "Caesium", 132.91, "[Xe] 6s1", 0.79, "alkali metal"),
    (56, "Ba", "Barium", 137.33, "[Xe] 6s2", 0.89, "alkaline earth metal"),
    (57, "La", "Lanthanum", 138.91, "[Xe] 5d1 6s2", 1.10, "lanthanide"),
    (58, "Ce", "Cerium", 140.12, "[Xe] 4f1 5d1 6s2", 1.12, "lanthanide"),
    (59, "Pr", "Praseodymium", 140.91, "[Xe] 4f3 6s2", 1.13, "lanthanide"),
    (60, "Nd", "Neodymium", 144.24, "[Xe] 4f4 6s2", 1.14, "lanthanide"),
    (61, "Pm", "Promethium", 145.0, "[Xe] 4f5 6s2", 1.13, "lanthanide"),
    (62, "Sm", "Samarium", 150.36, "[Xe] 4f6 6s2", 1.17, "lanthanide"),
    (63, "Eu", "Europium", 151.96, "[Xe] 4f7 6s2", 1.2, "lanthanide"),
    (64, "Gd", "Gadolinium", 157.25, "[Xe] 4f7 5d1 6s2", 1.20, "lanthanide"),
    (65, "Tb", "Terbium", 158.93, "[Xe] 4f9 6s2", 1.2, "lanthanide"),
    (66, "Dy", "Dysprosium", 162.50, "[Xe] 4f10 6s2", 1.22, "lanthanide"),
    (67, "Ho", "Holmium", 164.93, "[Xe] 4f11 6s2", 1.23, "lanthanide"),
    (68, "Er", "Erbium", 167.26, "[Xe] 4f12 6s2", 1.24, "lanthanide"),
    (69, "Tm", "Thulium", 168.93, "[Xe] 4f13 6s2", 1.25, "lanthanide"),
    (70, "Yb", "Ytterbium", 173.05, "[Xe] 4f14 6s2", 1.1, "lanthanide"),
    (71, "Lu", "Lutetium", 174.97, "[Xe] 4f14 5d1 6s2", 1.27, "lanthanide"),
    (72, "Hf", "Hafnium", 178.49, "[Xe] 4f14 5d2 6s2", 1.3, "transition metal"),
    (73, "Ta", "Tantalum", 180.95, "[Xe] 4f14 5d3 6s2", 1.5, "transition metal"),
    (74, "W", "Tungsten", 183.84, "[Xe] 4f14 5d4 6s2", 2.36, "transition metal"),
    (75, "Re", "Rhenium", 186.21, "[Xe] 4f14 5d5 6s2", 1.9, "transition metal"),
    (76, "Os", "Osmium", 190.23, "[Xe] 4f14 5d6 6s2", 2.2, "transition metal"),
    (77, "Ir", "Iridium", 192.22, "[Xe] 4f14 5d7 6s2", 2.20, "transition metal"),
    (78, "Pt", "Platinum", 195.08, "[Xe] 4f14 5d9 6s1", 2.28, "transition metal"),
    (79, "Au", "Gold", 196.97, "[Xe] 4f14 5d10 6s1", 2.54, "transition metal"),
    (80, "Hg", "Mercury", 200.59, "[Xe] 4f14 5d10 6s2", 2.00, "transition metal"),
    (81, "Tl", "Thallium", 204.38, "[Xe] 4f14 5d10 6s2 6p1", 1.62, "post-transition metal"),
    (82, "Pb", "Lead", 207.2, "[Xe] 4f14 5d10 6s2 6p2", 1.87, "post-transition metal"),
    (83, "Bi", "Bismuth", 208.98, "[Xe] 4f14 5d10 6s2 6p3", 2.02, "post-transition metal"),
    (84, "Po", "Polonium", 209.0, "[Xe] 4f14 5d10 6s2 6p4", 2.0, "post-transition metal"),
    (85, "At", "Astatine", 210.0, "[Xe] 4f14 5d10 6s2 6p5", 2.2, "halogen"),
    (86, "Rn", "Radon", 222.0, "[Xe] 4f14 5d10 6s2 6p6", 2.2, "noble gas"),
    (87, "Fr", "Francium", 223.0, "[Rn] 7s1", 0.7, "alkali metal"),
    (88, "Ra", "Radium", 226.0, "[Rn] 7s2", 0.9, "alkaline earth metal"),
    (89, "Ac", "Actinium", 227.0, "[Rn] 6d1 7s2", 1.1, "actinide"),
    (90, "Th", "Thorium", 232.04, "[Rn] 6d2 7s2", 1.3, "actinide"),
    (91, "Pa", "Protactinium", 231.04, "[Rn] 5f2 6d1 7s2", 1.5, "actinide"),
    (92, "U", "Uranium", 238.03, "[Rn] 5f3 6d1 7s2", 1.38, "actinide"),
    (93, "Np", "Neptunium", 237.0, "[Rn] 5f4 6d1 7s2", 1.36, "actinide"),
    (94, "Pu", "Plutonium", 244.0, "[Rn] 5f6 7s2", 1.28, "actinide"),
    (95, "Am", "Americium", 243.0, "[Rn] 5f7 7s2", 1.13, "actinide"),
    (96, "Cm", "Curium", 247.0, "[Rn] 5f7 6d1 7s2", 1.28, "actinide"),
    (97, "Bk", "Berkelium", 247.0, "[Rn] 5f9 7s2", 1.3, "actinide"),
    (98, "Cf", "Californium", 251.0, "[Rn] 5f10 7s2", 1.3, "actinide"),
    (99, "Es", "Einsteinium", 252.0, "[Rn] 5f11 7s2", 1.3, "actinide"),
    (100, "Fm", "Fermium", 257.0, "[Rn] 5f12 7s2", 1.3, "actinide"),
    (101, "Md", "Mendelevium", 258.0, "[Rn] 5f13 7s2", 1.3, "actinide"),
    (102, "No", "Nobelium", 259.0, "[Rn] 5f14 7s2", 1.3, "actinide"),
    (103, "Lr", "Lawrencium", 266.0, "[Rn] 5f14 7s2 7p1", 1.3, "actinide"),
    (104, "Rf", "Rutherfordium", 267.0, "[Rn] 5f14 6d2 7s2", None, "transition metal"),
    (105, "Db", "Dubnium", 268.0, "[Rn] 5f14 6d3 7s2", None, "transition metal"),
    (106, "Sg", "Seaborgium", 269.0, "[Rn] 5f14 6d4 7s2", None, "transition metal"),
    (107, "Bh", "Bohrium", 270.0, "[Rn] 5f14 6d5 7s2", None, "transition metal"),
    (108, "Hs", "Hassium", 277.0, "[Rn] 5f14 6d6 7s2", None, "transition metal"),
    (109, "Mt", "Meitnerium", 278.0, "[Rn] 5f14 6d7 7s2", None, "transition metal"),
    (110, "Ds", "Darmstadtium", 281.0, "[Rn] 5f14 6d8 7s2", None, "transition metal"),
    (111, "Rg", "Roentgenium", 282.0, "[Rn] 5f14 6d9 7s2", None, "transition metal"),
    (112, "Cn", "Copernicium", 285.0, "[Rn] 5f14 6d10 7s2", None, "transition metal"),
    (113, "Nh", "Nihonium", 286.0, "[Rn] 5f14 6d10 7s2 7p1", None, "post-transition metal"),
    (114, "Fl", "Flerovium", 289.0, "[Rn] 5f14 6d10 7s2 7p2", None, "post-transition metal"),
    (115, "Mc", "Moscovium", 290.0, "[Rn] 5f14 6d10 7s2 7p3", None, "post-transition metal"),
    (116, "Lv", "Livermorium", 293.0, "[Rn] 5f14 6d10 7s2 7p4", None, "post-transition metal"),
    (117, "Ts", "Tennessine", 294.0, "[Rn] 5f14 6d10 7s2 7p5", None, "halogen"),
    (118, "Og", "Oganesson", 294.0, "[Rn] 5f14 6d10 7s2 7p6", None, "noble gas"),
]

# Build lookup dicts
_BY_SYMBOL: Dict[str, Tuple] = {}
_BY_NUMBER: Dict[int, Tuple] = {}
_BY_NAME: Dict[str, Tuple] = {}

for _e in _ELEMENTS_DATA:
    _BY_SYMBOL[_e[1]] = _e
    _BY_NUMBER[_e[0]] = _e
    _BY_NAME[_e[2].lower()] = _e


def _lookup(key) -> Tuple:
    """Look up element by symbol, name, or number."""
    if isinstance(key, int):
        if key in _BY_NUMBER:
            return _BY_NUMBER[key]
        raise ValueError(f"No element with number {key}")
    key_str = str(key)
    if key_str in _BY_SYMBOL:
        return _BY_SYMBOL[key_str]
    if key_str.lower() in _BY_NAME:
        return _BY_NAME[key_str.lower()]
    raise ValueError(f"Unknown element: {key}")


def Element(key) -> Expr:
    """Get element data as an Expr."""
    e = _lookup(key)
    return Expr("List",
                Expr("Rule", "name", e[2]),
                Expr("Rule", "number", e[0]),
                Expr("Rule", "mass", e[3]),
                Expr("Rule", "symbol", e[1]),
                Expr("Rule", "electronConfiguration", e[4]),
                Expr("Rule", "electronegativity", e[5]),
                Expr("Rule", "category", e[6]))


def AtomicMass(key) -> float:
    """Get atomic mass of an element."""
    return _lookup(key)[3]


def ElectronConfiguration(key) -> str:
    """Get electron configuration of an element."""
    return _lookup(key)[4]


# ── Molecular formula parsing ──

def _parse_formula(formula: str) -> Dict[str, int]:
    """Parse a molecular formula like H2O, C6H12O6 into {element: count}."""
    result: Dict[str, int] = {}
    # Handle parentheses
    formula = _expand_parens(formula)
    # Parse element-count pairs
    pattern = re.compile(r'([A-Z][a-z]?)(\d*)')
    for m in pattern.finditer(formula):
        elem = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        if elem:
            result[elem] = result.get(elem, 0) + count
    return result


def _expand_parens(formula: str) -> str:
    """Expand parenthesized groups: Ca(OH)2 -> CaOHOH."""
    while '(' in formula:
        m = re.search(r'\(([^()]+)\)(\d*)', formula)
        if not m:
            break
        group = m.group(1)
        count = int(m.group(2)) if m.group(2) else 1
        expanded = group * count
        formula = formula[:m.start()] + expanded + formula[m.end():]
    return formula


def MolecularMass(formula: str) -> float:
    """Calculate molecular mass from formula string."""
    atoms = _parse_formula(formula)
    total = 0.0
    for elem, count in atoms.items():
        total += AtomicMass(elem) * count
    return round(total, 3)


# ── Equation balancing ──

def BalanceEquation(equation: str) -> str:
    """Balance a chemical equation string like 'H2 + O2 -> H2O'."""
    # Parse equation
    if '->' in equation:
        lhs, rhs = equation.split('->')
    elif '→' in equation:
        lhs, rhs = equation.split('→')
    elif '=' in equation:
        lhs, rhs = equation.split('=')
    else:
        raise ValueError("Equation must contain '->' or '='")

    lhs_compounds = [c.strip() for c in lhs.split('+')]
    rhs_compounds = [c.strip() for c in rhs.split('+')]
    all_compounds = lhs_compounds + rhs_compounds

    # Get all elements
    all_formulas = [_parse_formula(c) for c in all_compounds]
    elements = sorted(set(e for f in all_formulas for e in f))

    n = len(all_compounds)

    # Build matrix: each row is an element, each column is a compound
    # LHS compounds positive, RHS negative
    import numpy as np
    matrix = np.zeros((len(elements), n))
    for j, formula in enumerate(all_formulas):
        sign = 1 if j < len(lhs_compounds) else -1
        for i, elem in enumerate(elements):
            matrix[i, j] = sign * formula.get(elem, 0)

    # Find null space (coefficients)
    from numpy.linalg import svd
    U, S, Vt = svd(matrix)
    # Last row of Vt corresponds to smallest singular value
    null_vec = Vt[-1, :]

    # Make all coefficients positive
    if null_vec[0] < 0:
        null_vec = -null_vec

    # Convert to smallest integers
    null_vec = np.abs(null_vec)
    null_vec = null_vec / null_vec.min()

    # Round to nearest integer and verify
    coeffs = np.round(null_vec).astype(int)

    # Scale to smallest integers
    from math import gcd
    from functools import reduce
    g = reduce(gcd, coeffs)
    coeffs = coeffs // g

    # Format result
    lhs_parts = []
    for i, c in enumerate(lhs_compounds):
        coeff = coeffs[i]
        if coeff == 1:
            lhs_parts.append(c)
        else:
            lhs_parts.append(f"{coeff}{c}")

    rhs_parts = []
    for i, c in enumerate(rhs_compounds):
        coeff = coeffs[len(lhs_compounds) + i]
        if coeff == 1:
            rhs_parts.append(c)
        else:
            rhs_parts.append(f"{coeff}{c}")

    return " + ".join(lhs_parts) + " -> " + " + ".join(rhs_parts)
