# Python interop

MikoshiLang is a Python library first. This page covers the three ways the two languages meet: calling
MikoshiLang from Python (`Session.eval`, `Session.ns`), calling your Python functions from MikoshiLang
(`@session.function`), and calling library Python functions from MikoshiLang code (`PythonFunction`, gated by
`allow_python`). Packed NumPy-backed arrays (`NumericArray`, which will let `Dot`, `Total` and arithmetic on
machine arrays run in NumPy) are a later phase and are not part of this API yet.

## Calling MikoshiLang from Python

```python
from fractions import Fraction
import numpy as np
from mikoshilang import Session

s = Session()
s.eval("1 + 2")                       # 3                       (int)
s.eval("1/3")                         # Fraction(1, 3)          exact rationals are fractions.Fraction
s.eval("0.5 - 0.5")                   # 0.0                     inexact zero stays inexact
s.eval("N[Pi, 30]")                   # sympy.Float, 30 digits  bignum reals keep their precision
s.eval('<|"a" -> {1, 2.5}|>')         # {'a': [1, 2.5]}
s.eval("Integrate[x^2, x]")           # Expr x^3/3              symbolic results stay expressions
s.eval("Integrate[x^2, x]", symbolic="sympy")   # sympy x**3/3
s.eval("{{1., 2.}, {3., 4.}}", arrays="numpy")  # numpy.ndarray
s.eval(Fraction(1, 3))                # Python values are converted on the way in
```

`Session.eval(source, out="python", symbolic="expr", arrays="list")`

* `source`: MikoshiLang code (a `str`), an expression, or any Python value from the conversion table.
* `out="python"` converts the result with `mikoshilang.interop.to_python`; `out="expr"` returns it raw.
* `symbolic`: what happens to results that are not data: `"expr"` (keep the `Expr`, default), `"sympy"`
  (convert) or `"error"` (raise `TypeError`).
* `arrays`: `"list"` (nested lists, default) or `"numpy"` (a rectangular list of machine numbers becomes an
  `ndarray`).

`Session.evaluate(source)` is unchanged: it returns the raw MikoshiLang result.

### Pythonic calls: `Session.ns`

```python
ml = s.ns
x = ml.symbols("x")
ml.Integrate(x**2, x)                          # Expr x^3/3
ml.Total([1, 2, 3])                            # 6
ml.Mean(np.array([1.0, 2.0, 3.0]))             # 2.0
ml.Keys({"a": 1, "b": 2})                      # ['a', 'b']
ml.NIntegrate(x**2, [x, 0, 1], WorkingPrecision=30)   # keyword arguments become options
s.ns(arrays="numpy").Range(4)                  # array([1, 2, 3, 4])
```

## The conversion table

`mikoshilang.interop.from_python` (Python to MikoshiLang) and `to_python` (back). Every Python type converts in
exactly one documented way or raises `TypeError`: there are no inert foreign atoms. `<->` rows round-trip
exactly (property-tested with hypothesis).

| Python | MikoshiLang | Type | Direction |
|---|---|---|---|
| `bool`, `numpy.bool_` | `True` / `False` | `Boolean` | <-> |
| `int`, `numpy.integer` | Integer | `Integer` | <-> |
| `float`, `numpy.floating` | machine Real | `Real` | <-> |
| `float('inf')`, `-inf` | `Infinity`, `-Infinity` | `Real` | <-> |
| `float('nan')` | none: raises `ValueError` | | |
| `complex` | machine complex `a + b I` | `Number` | <-> |
| `fractions.Fraction` | Rational | `Rational` | <-> |
| `sympy.Float` (more than machine precision) | bignum Real, precision kept | `Real` | <-> |
| `str` | String | `String` | <-> |
| `None` | `Null` | | <-> |
| `list` (`tuple` one way) | List | `ListOf[T]` | <-> |
| `dict` | Association (key order kept) | `Assoc` | <-> |
| `numpy.ndarray` | nested List | `ListOf[Number]` | -> (back with `arrays="numpy"`) |
| `pandas.Series` / `DataFrame` | List or Association / List of Associations | | -> |
| `sympy.Basic` | expression | `Expr` | <-> (back with `symbolic="sympy"`) |
| `sympy.Matrix` | nested List | `MatrixOf[T]` | -> |
| `Expr`, `Symbol` | itself | `Expr`, `Var` | <-> |
| callable, `set`, other objects | none: raises `TypeError` | | |

Exact algebraic numbers (`Sqrt[2]`, `1 + 2 I`) have no exact Python type and stay expressions. Python operators
on symbols use the same table: `x + Fraction(1, 3)` builds `x + 1/3`, and `x + object()` raises `TypeError`.
pandas is never imported by MikoshiLang; it is used only when your program has already imported it.

## Calling your Python functions from MikoshiLang: `@session.function`

```python
s = Session()

@s.function(listable=True, check="reference")
def collatz_steps(n: int) -> int:              # head CollatzSteps, signature {Integer} -> Integer
    if n < 1:
        raise ValueError("n must be positive")
    k = 0
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        k += 1
    return k

s.eval("CollatzSteps[{1, 6, 27}]")           # [0, 8, 111]
s.eval("CollatzSteps[n]")                    # CollatzSteps[n]   symbolic input: unevaluated
s.eval("CollatzSteps[2.5]")                  # TypeError: CollatzSteps: argument 1 must be Integer, not 2.5
s.eval('Check[CollatzSteps[0], "caught"]')   # "caught"
s.eval("Map[CollatzSteps, Range[3]]")        # [0, 1, 7]
```

* **Per session.** Registration is local to the session (other sessions never see it); `Session.clear()`
  removes it, and `Session.functions()` lists what is registered. Library heads are not shadowed unless you pass
  `override=True`; kernel built-ins (`Plus`, `Set`, ...) can never be redefined from Python.
* **Names.** `snake_case` becomes `CamelCase` (`collatz_steps` -> `CollatzSteps`); pass `name=` to choose.
* **Types from hints.** `int` -> Integer, `float` -> Real (exact numbers such as `1/2` or `Pi` are converted),
  `complex` -> Number, `Fraction` -> Rational, `str`, `bool`, `list[int]` -> `ListOf[Integer]`, `dict` ->
  Assoc, `numpy.ndarray`, `sympy.Expr` (the argument is passed as SymPy), `Expr` (raw expression), `Optional`
  and `Union`. CONVENTIONS vocabulary strings work too: `def binom(n: "NonNegInteger", k: "NonNegInteger")`.
  Arguments are checked in the CONVENTIONS order: wrong kind raises `TypeError`, symbolic input leaves the call
  unevaluated, a bad value (a negative `NonNegInteger`) raises `ValueError`. Under Python 3.9 avoid `int | None`
  in hints (use `Optional[int]`).
* **Options.** Keyword-only parameters take trailing options: `def scale(v: list, *, factor: float = 1.0)` is
  called as `Scale[{1, 2}, Factor -> 2]` (`Factor` -> `factor`, `RelTol` -> `rel_tol`).
* **Errors.** `TypeError`, `ValueError`, `NotImplementedError` and `ArithmeticError` raised by your function keep
  their class and get the `"Name: "` prefix; any other exception becomes
  `ValueError("Name: Python raised KeyError: ...")`, chained to the original. All of them are catchable with
  `Check` inside the language.
* **Verification.** `check=` records the CONVENTIONS `Check:` method (`exact`, `self-check`,
  `numeric cross-check`, `reference`, `delegated`) and the function's module and qualified name in
  `fn.mikoshi.notes`.

## Calling Python libraries from MikoshiLang: `PythonFunction` and `allow_python`

```python
s = Session(allow_python={"math", "numpy.linalg"})
s.eval('PythonFunction["math.gcd"][12, 18]')                  # 6
s.eval('PythonFunction["numpy.linalg.det"][{{2, 0}, {0, 3}}]') # 6.0
s.eval('PythonFunction["math.isclose"][1, 1.05, RelTol -> 0.1]')   # True
s.eval('PythonFunction["math.sqrt"][y]')                      # unevaluated: symbolic argument
Session().eval('PythonFunction["math.gcd"][12, 18]')          # PermissionError
```

* `allow_python=False` (default): MikoshiLang code cannot call Python. `allow_python={"math", ...}` (or one
  module name as a string) allows those modules and their submodules; `"numpy.linalg"` allows only that
  submodule. `allow_python=True` allows any importable module.
* Private and dunder names (`os.__dict__`, `math._x`) are always refused.
* `@session.function` needs no permission: the host program chose to expose that function.
* `ExternalEvaluate`, `ExternalFunction` and `StartExternalSession` raise `NotImplementedError`: there are no
  code strings and no `eval`/`exec`.
* Every capability error is a `PermissionError` that names the flag to set, and `Check` catches it.

**A Session is not a sandbox.** An allow-list is a convenience, not a security boundary: `numpy` alone can reach
the file system (`numpy.load`). Untrusted MikoshiLang input must run in an external process sandbox with time
and memory limits, the network off and `allow_python=False`.

## Error mapping

| Situation | Raised at the Python boundary | Inside MikoshiLang |
|---|---|---|
| wrong kind of argument | `TypeError("Name: ...")` | catchable by `Check` |
| bad value or option | `ValueError` | catchable by `Check` |
| unsupported | `NotImplementedError` | catchable by `Check` |
| self-check failed, division by zero | `ArithmeticError` (subclass kept) | catchable by `Check` |
| symbolic input | not an error: unevaluated result | unevaluated |
| capability not granted | `PermissionError` | catchable by `Check` |
| other Python exception | `ValueError("Name: Python raised KeyError: ...")` | catchable by `Check` |
