"""Built-in functions — trig, exponential, number theory, combinatorics, logic."""

from __future__ import annotations
import math
import sympy as sp
from .expr import Expr, Symbol
from .math_ops import to_sympy, from_sympy


# ── Expression constructors (symbolic) ──

def Sin(x) -> Expr:
    return Expr("Sin", x)

def Cos(x) -> Expr:
    return Expr("Cos", x)

def Tan(x) -> Expr:
    return Expr("Tan", x)

def ArcSin(x) -> Expr:
    return Expr("ArcSin", x)

def ArcCos(x) -> Expr:
    return Expr("ArcCos", x)

def ArcTan(x) -> Expr:
    return Expr("ArcTan", x)

def Exp(x) -> Expr:
    return Expr("Exp", x)

def Log(x, base=None) -> Expr:
    if base is not None:
        return Expr("Log", base, x)
    return Expr("Log", x)

def Log2(x) -> Expr:
    return Expr("Log", 2, x)

def Log10(x) -> Expr:
    return Expr("Log", 10, x)


# ── Number theory ──

def Prime(n: int) -> int:
    """Return the nth prime number."""
    return int(sp.prime(n))

def PrimeQ(n: int) -> bool:
    """Test if n is prime."""
    return sp.isprime(n)

def FactorInteger(n: int) -> Expr:
    """Factor an integer into prime factors."""
    factors = sp.factorint(n)
    return Expr("List", *(Expr("List", int(p), int(e)) for p, e in sorted(factors.items())))

def GCD(*args: int) -> int:
    """Greatest common divisor."""
    return math.gcd(*args)

def LCM(*args: int) -> int:
    """Least common multiple."""
    return math.lcm(*args)

def Mod(a: int, b: int) -> int:
    """Modular arithmetic."""
    return a % b


# ── Combinatorics ──

def Factorial(n: int) -> int:
    """n!"""
    return math.factorial(n)

def Binomial(n: int, k: int) -> int:
    """Binomial coefficient C(n, k)."""
    return math.comb(n, k)

def Fibonacci(n: int) -> int:
    """nth Fibonacci number."""
    # Validate the index before SymPy's analytic extension can be truncated.
    import operator
    return int(sp.fibonacci(operator.index(n)))


# ── Logic ──

def And(*args: bool):
    return all(args) if all(isinstance(a, bool) for a in args) else Expr("And", *args)

def Or(*args: bool):
    return any(args) if all(isinstance(a, bool) for a in args) else Expr("Or", *args)

def Not(x: bool):
    return not x if isinstance(x, bool) else Expr("Not", x)

def Xor(a: bool, b: bool):
    return a ^ b if isinstance(a, bool) and isinstance(b, bool) else Expr("Xor", a, b)

def Implies(a: bool, b: bool):
    return (not a) or b if isinstance(a, bool) and isinstance(b, bool) else Expr("Implies", a, b)


# ── Comparison ──

def Equal(a, b):
    if isinstance(a, (Expr, Symbol)) or isinstance(b, (Expr, Symbol)):
        return True if a == b else Expr("Equal", a, b)
    return a == b

def Less(a, b):
    if isinstance(a, (Expr, Symbol)) or isinstance(b, (Expr, Symbol)):
        return Expr("Less", a, b)
    return a < b

def Greater(a, b):
    if isinstance(a, (Expr, Symbol)) or isinstance(b, (Expr, Symbol)):
        return Expr("Greater", a, b)
    return a > b

def LessEqual(a, b):
    if isinstance(a, (Expr, Symbol)) or isinstance(b, (Expr, Symbol)):
        return Expr("LessEqual", a, b)
    return a <= b

def GreaterEqual(a, b):
    if isinstance(a, (Expr, Symbol)) or isinstance(b, (Expr, Symbol)):
        return Expr("GreaterEqual", a, b)
    return a >= b


# ── Registry ──

BUILTIN_FUNCTIONS = {
    "Sin": Sin, "Cos": Cos, "Tan": Tan,
    "ArcSin": ArcSin, "ArcCos": ArcCos, "ArcTan": ArcTan,
    "Exp": Exp, "Log": Log, "Log2": Log2, "Log10": Log10,
    "Prime": Prime, "PrimeQ": PrimeQ, "FactorInteger": FactorInteger,
    "GCD": GCD, "LCM": LCM, "Mod": Mod,
    "Factorial": Factorial, "Binomial": Binomial, "Fibonacci": Fibonacci,
    "And": And, "Or": Or, "Not": Not, "Xor": Xor, "Implies": Implies,
    "Equal": Equal, "Less": Less, "Greater": Greater,
    "LessEqual": LessEqual, "GreaterEqual": GreaterEqual,
}
