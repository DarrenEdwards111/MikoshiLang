"""Calculus: derivatives, integrals, limits (with Direction), series and ODEs.

Run: python examples/02_calculus.py
"""
from mikoshilang import Session

s = Session()

print("D[Sin[x] x^2, x]               =", s.evaluate("D[Sin[x] x^2, x]"))
print("Integrate[x^2, x]              =", s.evaluate("Integrate[x^2, x]"))
print("Integrate[Exp[-x^2], {x, -Infinity, Infinity}] =",
      s.evaluate("Integrate[Exp[-x^2], {x, -Infinity, Infinity}]"))
print("Limit[Sin[x]/x, x -> 0]        =", s.evaluate("Limit[Sin[x]/x, x -> 0]"))
print("Limit[1/x, x -> 0] (two-sided) =", s.evaluate("Limit[1/x, x -> 0]"))
print('Limit[1/x, x -> 0, Direction -> "FromAbove"] =',
      s.evaluate('Limit[1/x, x -> 0, Direction -> "FromAbove"]'))
print("Series[Exp[x], {x, 0, 4}]      =", s.evaluate("Series[Exp[x], {x, 0, 4}]"))
print("Normal[...]                    =", s.evaluate("Normal[Series[Exp[x], {x, 0, 4}]]"))
print("DSolve[y' == y, y[0] == 1]     =", s.evaluate("DSolve[{y'[x] == y[x], y[0] == 1}, y[x], x]"))
