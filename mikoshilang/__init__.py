"""MikoshiLang — A symbolic computation language for Python.

Built by Mikoshi Ltd.
"""

__version__ = "3.6.5"

from .expr import Expr, Symbol, symbols
from .pattern import Blank, BlankSequence, BlankNullSequence, Pattern
from .rules import Rule, RuleDelayed, replace, replace_all
from .evaluate import evaluate
from .math_ops import simplify, expand, factor, diff, integrate, solve, limit, series
from .numerical import N, numerical, Pi, E, I, Infinity, GoldenRatio
from .data import List, Table, Map, Select, Sort, Reverse, Take, Drop, Part, Range, Total, Mean, Median, StandardDeviation
from .linalg import Matrix, Dot, Det, Inverse, Eigenvalues, Eigenvectors, Transpose
from .builtins import (
    Sin, Cos, Tan, ArcSin, ArcCos, ArcTan,
    Exp, Log, Log2, Log10,
    Prime, PrimeQ, FactorInteger, GCD, LCM, Mod,
    Factorial, Binomial, Fibonacci,
    And, Or, Not, Xor, Implies,
    Equal, Less, Greater, LessEqual, GreaterEqual,
)
from .parser import parse, parse_file, parse_and_eval, ParseError
from .latex import to_latex
from .units import Quantity, UnitConvert
from .chemistry import Element, AtomicMass, ElectronConfiguration, MolecularMass, BalanceEquation
from .signal import (
    FourierTransform, InverseFourierTransform,
    DFT, IDFT,
    LowPassFilter, HighPassFilter, BandPassFilter,
    Convolve, HammingWindow, HanningWindow, BlackmanWindow,
    Spectrogram,
)

__all__ = [
    "Expr", "Symbol", "symbols",
    "Blank", "BlankSequence", "BlankNullSequence", "Pattern",
    "Rule", "RuleDelayed", "replace", "replace_all",
    "evaluate",
    "simplify", "expand", "factor", "diff", "integrate", "solve", "limit", "series",
    "N", "numerical", "Pi", "E", "I", "Infinity", "GoldenRatio",
    "List", "Table", "Map", "Select", "Sort", "Reverse", "Take", "Drop", "Part",
    "Range", "Total", "Mean", "Median", "StandardDeviation",
    "Matrix", "Dot", "Det", "Inverse", "Eigenvalues", "Eigenvectors", "Transpose",
    "Sin", "Cos", "Tan", "ArcSin", "ArcCos", "ArcTan",
    "Exp", "Log", "Log2", "Log10",
    "Prime", "PrimeQ", "FactorInteger", "GCD", "LCM", "Mod",
    "Factorial", "Binomial", "Fibonacci",
    "And", "Or", "Not", "Xor", "Implies",
    "Equal", "Less", "Greater", "LessEqual", "GreaterEqual",
    "parse", "parse_file", "parse_and_eval", "ParseError",
    "to_latex",
    "Quantity", "UnitConvert",
    "Element", "AtomicMass", "ElectronConfiguration", "MolecularMass", "BalanceEquation",
    "FourierTransform", "InverseFourierTransform",
    "DFT", "IDFT",
    "LowPassFilter", "HighPassFilter", "BandPassFilter",
    "Convolve", "HammingWindow", "HanningWindow", "BlackmanWindow",
    "Spectrogram",
]
