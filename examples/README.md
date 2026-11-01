# Examples

Small runnable scripts. Install MikoshiLang first (`pip install mikoshilang`, or `pip install .` from a checkout),
then run any of them from the repository root:

| File | Shows |
|---|---|
| [`01_symbolic_algebra.py`](01_symbolic_algebra.py) | Pattern definitions with memoisation, `Expand`/`Factor`/`Simplify`, `Solve` rule lists, replacement rules, associations, exact and high-precision arithmetic |
| [`02_calculus.py`](02_calculus.py) | `D`, `Integrate`, `Limit` (two-sided and with `Direction`), `Series`/`Normal`, `DSolve` |
| [`03_python_interop.py`](03_python_interop.py) | Python values from `Session.eval`, NumPy arrays, `Session.ns`, typed Python functions as heads, the `allow_python` gate |
| [`04_types.py`](04_types.py) | `TypeSignature`, `Session(check_types=True)`, `TypeCheck`, report mode |
| [`05_verify.py`](05_verify.py) | `Verify` returning `Certified`, `Checked` and `Unchecked` |
| [`06_language.mkl`](06_language.mkl) | A MikoshiLang source file: patterns, `Module`, pure functions, `;;`, optional patterns, `Fold`, `Table` |

```sh
python examples/05_verify.py
python examples/run_mkl.py examples/06_language.mkl
```
