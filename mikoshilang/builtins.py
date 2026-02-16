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
    result = args[0]
    for a in args[1:]:
        result = math.gcd(result, a)
    return result

def LCM(*args: int) -> int:
    """Least common multiple."""
    result = args[0]
    for a in args[1:]:
        result = result * a // math.gcd(result, a)
    return result

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
    return int(sp.fibonacci(n))


# ── Logic ──

def And(*args: bool) -> bool:
    return all(args)

def Or(*args: bool) -> bool:
    return any(args)

def Not(x: bool) -> bool:
    return not x

def Xor(a: bool, b: bool) -> bool:
    return a ^ b

def Implies(a: bool, b: bool) -> bool:
    return (not a) or b


# ── Comparison ──

def Equal(a, b) -> bool:
    return a == b

def Less(a, b) -> bool:
    return a < b

def Greater(a, b) -> bool:
    return a > b

def LessEqual(a, b) -> bool:
    return a <= b

def GreaterEqual(a, b) -> bool:
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
