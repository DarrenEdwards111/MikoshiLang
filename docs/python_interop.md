# Python interop

MikoshiLang is a Python library first. This page covers the ways the two languages meet: calling MikoshiLang
from Python (`Session.eval`, `Session.ns`), calling your Python functions from MikoshiLang (`@session.function`
for one session, `@mikoshilang.function` for library authors), calling Python libraries from MikoshiLang code
(`PythonFunction`, `PythonImport` and `PythonObject` handles, gated by `allow_python`), and packed arrays, which
keep machine-number data in one NumPy array so that `Dot`, `Total`, arithmetic and elementary functions run in
NumPy.

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
| `numpy.ndarray` (int, float, complex) | packed List (one NumPy array, see below) | `ListOf[Number]` | <-> (as `list`, or an ndarray with `arrays="numpy"`) |
| `numpy.ndarray` (bool, str, object, NaN/inf) | nested List | | -> |
| `pandas.Series` / `DataFrame` | List or Association / List of Associations | | -> |
| `sympy.Basic` | expression | `Expr` | <-> (back with `symbolic="sympy"`) |
| `sympy.Matrix` | nested List | `MatrixOf[T]` | -> |
| `Expr`, `Symbol` | itself | `Expr`, `Var` | <-> |
| callable, `set`, other objects | none: raises `TypeError` (results of `PythonFunction` become `PythonObject` handles instead) | | |
| `PythonObject` handle | the Python object it stands for | | <- (`Session.eval`) |

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

### For library authors: `@mikoshilang.function`

```python
import mikoshilang

@mikoshilang.function(check="exact", listable=True, area="combinatorics")
def triangle_number(n: "NonNegInteger") -> "Integer":
    return n * (n + 1) // 2

Session().eval("TriangleNumber[{1, 2, 10}]")      # [1, 3, 55], in every session
mikoshilang.unregister("TriangleNumber")
```

The same typing, error and listable rules as `@session.function`, but the head is global: it is recorded in
`library.STATUS` with status `added` and its `Check:` note, like the built-in library, so library modules and
their tests follow CONVENTIONS.md. Kernel built-ins are never redefined; an existing library head is replaced only
with `override=True`, and `unregister` restores it. A session function with the same name shadows the global one
in that session. In applications prefer `@session.function`: a global head is visible to every session.

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
* `ExternalEvaluate`, `ExternalFunction` and `StartExternalSession` are reserved library heads (status `error`)
  that always raise `NotImplementedError`: there are no code strings and no `eval`/`exec`.
* Trailing `Name -> value` rules are keyword arguments (`RelTol -> 0.1` is `rel_tol=0.1`), so a rule cannot be
  passed as a positional data argument.

### Python objects: `PythonImport`, `PythonObject` handles and `PythonAttribute`

```python
s = Session(allow_python={"numpy"})
s.eval('np = PythonImport["numpy"]')                         # PythonObject[1, "module numpy"]
s.eval('PythonFunction[np, "linalg.det"][{{2, 0}, {0, 3}}]')  # 6.0
s.eval('rng = PythonFunction[np, "random.default_rng"][1]')  # PythonObject[2, "numpy.random._generator.Generator"]
s.eval('PythonFunction[rng, "integers"][10]')                # 0 ... 9
s.eval('PythonAttribute[np, "pi"]')                          # 3.141592653589793
s.eval("rng")                                                # the Generator object itself, in Python
```

* `PythonImport["module"]` needs the module in `allow_python` and returns a handle `PythonObject[id, "description"]`.
* `PythonFunction[handle][args]` calls the object; `PythonFunction[handle, "a.b"][args]` calls an attribute path
  (for a module handle the full dotted name is checked against the allow-list, so `{"numpy.linalg"}` does not
  open the rest of NumPy).
* `PythonAttribute[handle, "a.b"]` reads an attribute.
* A `PythonFunction` result that has no MikoshiLang form (a `Generator`, a file object, ...) becomes a new handle
  instead of an error; handles passed as arguments are given to Python as the objects themselves.
* Handles belong to their session: `Session.clear()` drops them, and another session cannot use them. Every use
  re-checks `allow_python` for the module the object came from. Private and dunder attributes are refused.
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

## Packed arrays

A packed array is a `List` of machine numbers (`int64`, `float64` or `complex128`) stored as one read-only NumPy
array, like Wolfram's packed arrays. It is invisible to the language: `Head` is `List`, `Part`, `Map`, patterns
and printing see ordinary elements, and a packed list equals the unpacked list with the same elements.

Arrays are packed when they come from NumPy (`Session.eval`, `Session.ns`), by `ToPackedArray[list]`
(`Developer\`ToPackedArray`), in `NumericArray[...]` (its values are stored packed; `Normal` gives the packed
list), as results of the fast operations, and by `Dot`, `Total` and `Mean` for plain machine-number lists of at
least 250 elements. `FromPackedArray` unpacks and `PackedArrayQ[expr, type, rank]` tests.

Operations on packed arrays run in NumPy: `Dot`, `Total`, `Mean`, `Max`, `Min`, `Transpose`, `Length`,
`Dimensions`, `N`, `Normal`, `Plus`, `Times`, `Power` (scalar exponent), `Abs` and the machine-real elementary
functions (`Sin`, `Cos`, `Tan`, `Exp`, `Log`, `Sqrt`, `Sinh`, `Cosh`, `Tanh`, `ArcTan`). They fall back to the
ordinary element-by-element path whenever NumPy could give a different answer: integer results that could
overflow `int64`, exact rationals with integer arrays (`{1, 2, 3}/2`), symbolic operands, exact functions of
integers (`Sin[{1, 2}]` stays exact), values outside a real domain (`Sqrt` of a negative, `Log` of a non-positive
number) and NaN or infinite results.

Benchmarks: `python scripts/benchmark_packed.py`.

## Inexact zero

An inexact zero stays inexact, as in Wolfram: `0.5 - 0.5` is `0.`, `0. x` is `0.`, `x + 1.5 - 1.5` is `0. + x`,
and numeric functions of inexact input give inexact results at zero (`Sin[0.]`, `Sqrt[0.]` and `Log[1.]` are
`0.`). Exact zeros are unchanged (`Sin[0]` and `x - x` are `0`; `0 * 1.5` is exact `0`). One known difference
remains: SymPy-based symbolic operations on an expression that contains an inexact zero term may drop it
(`(0. + x)^2` gives `x^2`; Wolfram keeps `(0. + x)^2`).
