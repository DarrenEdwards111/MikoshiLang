"""Symbolic algebra: pattern definitions, simplification, factoring, solving.

Run: python examples/01_symbolic_algebra.py
"""
from mikoshilang import Session

s = Session()  # isolated definitions; network disabled

# Pattern-based definitions, including memoisation and specific-before-general rules
s.evaluate("fib[0] = 0; fib[1] = 1; fib[n_Integer] := fib[n] = fib[n - 1] + fib[n - 2]")
print("fib[60]                 =", s.evaluate("fib[60]"))

# Polynomials
print("Expand[(x + y)^3]       =", s.evaluate("Expand[(x + y)^3]"))
print("Factor[x^4 - 1]         =", s.evaluate("Factor[x^4 - 1]"))
print("Simplify[...]           =", s.evaluate("Simplify[(x^2 - 1)/(x - 1)]"))

# Solve returns Wolfram rule lists
print("Solve[x^2 - 5x + 6 == 0] =", s.evaluate("Solve[x^2 - 5 x + 6 == 0, x]"))
print("Solve[system]           =", s.evaluate("Solve[{x + y == 3, x - y == 1}, {x, y}]"))

# Rules, replacement and associations
print("ReplaceAll              =", s.evaluate("x^2 + y /. {x -> 2, y -> a}"))
print("Association lookup      =", s.evaluate('<|"a" -> 1, "b" -> 2|>["b"]'))

# Exact arithmetic and high precision
print("1/3 + 1/6               =", s.evaluate("1/3 + 1/6"))
print("N[Pi, 30]               =", s.evaluate("N[Pi, 30]"))
