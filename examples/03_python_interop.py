"""Python interop: Python values out, typed Python functions in, and gated Python access.

Run: python examples/03_python_interop.py
"""
from fractions import Fraction

from mikoshilang import Session

s = Session()

# Results come back as Python data; symbolic results stay expressions
assert s.eval("1/3") == Fraction(1, 3)
print("s.eval('1/3')              ->", repr(s.eval("1/3")))
print("s.eval('Range[5]')         ->", s.eval("Range[5]"))
print("s.eval(association)        ->", s.eval('<|"a" -> 1, "b" -> 2|>'))
print("numpy array                ->", repr(s.eval("{{1., 2.}, {3., 4.}}", arrays="numpy")))

# Pythonic calls through the session namespace
ml = s.ns
x = ml.symbols("x")
print("ml.D(x**3, x)              ->", ml.D(x**3, x))


# A typed Python function becomes a language head (CollatzSteps), listable over lists
@s.function(listable=True)
def collatz_steps(n: int) -> int:
    k = 0
    while n != 1:
        n, k = (n // 2 if n % 2 == 0 else 3 * n + 1), k + 1
    return k


print("CollatzSteps[{1, 6, 27}]   ->", s.eval("CollatzSteps[{1, 6, 27}]"))
print("CollatzSteps[n] (symbolic) ->", s.evaluate("CollatzSteps[n]"))

# Calling Python from MikoshiLang is off unless a module is allow-listed
try:
    s.eval('PythonFunction["math.gcd"][12, 18]')
except PermissionError as e:
    print("without allow_python       -> PermissionError:", str(e)[:60], "...")
t = Session(allow_python={"math"})
print("with allow_python={'math'} ->", t.eval('PythonFunction["math.gcd"][12, 18]'))
