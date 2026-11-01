# Types (pillar 1, v1)

MikoshiLang types are gradual and opt-in. Untyped code works as before. Types come from two places:

* **Library heads** take their signature from the Python annotations of their implementations, written with
  the CONVENTIONS §2 vocabulary (`def _bitlength(n: Integer) -> Integer`). The registry exposes it as
  `REGISTRY[name].signature`, one `types.Signature` per implementation.
* **User functions** declare one with `TypeSignature[f, {t1, t2, ...} -> result]`. Repeat it to declare
  overloads. `RepeatedNull[t]` (last) stands for any number of further arguments.

## Vocabulary

| Type | Accepts |
|---|---|
| `Expr` | anything |
| `Integer`, `PosInteger`, `NonNegInteger` | exact integers (never `True`/`False`), with a sign constraint |
| `Rational` | exact integer or rational |
| `Real` | exact or inexact real number, including `Pi` and `Sqrt[2]` |
| `Number` | a number, complex included (`1 + 2 I`) |
| `Numeric` | anything `N` turns into a number (`Sqrt[2] + Pi`) |
| `String`, `Boolean` | strings; `True`/`False` |
| `Var` | a symbol used as a variable (not `Pi`, `E`, `I`, ...) |
| `ListOf[t]` (or `{t}`), `MatrixOf[t]` | lists whose elements have type `t`; rectangular lists of lists |
| `Assoc`, `Rule`, `Option` | associations; `a -> b`; `Name -> value` options |
| `Fn` | anything applicable (not a number or string) |
| `Held` | an unevaluated argument (held heads) |
| `t1 \| t2` | either |

## Functions

```wolfram
TypeSignature[BitLength]                         (* {{Integer} -> Expr} : from the annotations *)
TypeSignature[area, {Real, Real} -> Real]        (* declare *)
TypeMatchQ[{1, 2}, ListOf[Integer]]              (* True; strict: TypeMatchQ[x, Integer] is False *)
TypeCheck[area[2, "3"]]                          (* <|"Result" -> ..., "Mismatches" -> {<|"Function" -> area,
                                                     "Position" -> 2, "Value" -> "3", "Expected" -> "Real"|>}|> *)
```

From Python:

```python
from mikoshilang import Session
s = Session(check_types=True)        # raise TypeError("f: argument 2 '3' is not of type Real (type check)")
r = Session(check_types="report")    # carry on; mismatches collect in r.type_mismatches
```

## Context

`TypeSignature`, `TypeMatchQ`, `TypeCheck`, `Verify`, `Certified`, `Checked`, `Unchecked` and `Failed` are
MikoshiLang extensions, not Wolfram symbols, so they live in the `Mikoshi`` context:
`Context[Verify]` is `"Mikoshi`"`, and `Mikoshi`Verify[...]` is the full name. `Mikoshi`` is on the default
`$ContextPath` (`{"System`", "Mikoshi`", "Global`"}`), so the short names work everywhere, including inside
`BeginPackage`. They are protected, like `System`` symbols.

## What counts as a mismatch

The checker runs at call boundaries, after the arguments are evaluated and before the implementation runs
(library heads after `Listable` threading, so element types are checked). It also checks results against the
declared or annotated result type. A **mismatch** is a *concrete* value of the wrong kind: a string, a number,
a list, an association, a rule or `True`/`False` where the type excludes it.

* **Symbolic arguments are never mismatches.** `x`, `n + 1` and `f[y]` may become anything, and the function
  may legitimately leave such a call unevaluated (CONVENTIONS §3).
* **Sign constraints are enforced.** `0` is not a `PosInteger` and `-1` is not a `NonNegInteger`, so a
  checking session reports `PolygonalNumber[3, -1]` as a mismatch, as `TypeMatchQ` does. Without checking, the
  function raises its own `ValueError` (CONVENTIONS §2).
* **`TypeCheck` always reports.** Inside a `Session(check_types=True)`, `TypeCheck[expr]` still returns its
  report instead of raising. Its result is final and is not evaluated again under the session's mode.
* Held arguments (`Held`, or a user function with `HoldAll`/`HoldFirst`/`HoldRest`) are not checked. Their
  result is.
* **Kernel built-ins** (`Plus`, `Times`, `Power`, `Greater`, `Range`, `Nest`, `StringJoin`, `Det`, ...) have
  hand-written signatures in `mikoshilang/kernel_types.py`, checked in evaluator step 8. They are deliberately
  lenient: `Plus` takes `Numeric | ListOf[Expr] | Assoc` (symbols and other non-concrete expressions always pass),
  so `Plus["a", 1]`, `Nest[f, x, -1]`, `StringLength[3]`, `Det[{1, 2}]` and `Not["a"]` are mismatches while
  every symbolic call is clean. Held kernel heads (`Block`, `Set`, ...) show `{Held...} -> Expr`.
  `TypeSignature[Plus]` shows the signature.

## Cost

When no checking session is active, the evaluator and the library dispatcher read one module-level integer
(`mikoshilang._typestate.ACTIVE`). `scripts/benchmark_typecheck.py` measures this. On its dispatch-heavy workload (Python 3.13, means of 3
interleaved best-of-5 runs):

* **Checking off:** 1.087 s against 1.083 s on a checkout without the checker. That's +0.4%, within the
  run-to-run noise.
* **`check_types=True`:** 1.196 s (about +10%).

## Coverage of the library annotations

Every registered implementation in the `wl_*` modules is annotated, except in `wl_stats` and `wl_graphs`, which
are being annotated in their own batches. Most parameters are `Expr`: Wolfram functions are polymorphic, and the
specific types were added only where the code enforces them (validation helpers and guards at the top of the
body). Running the whole test suite in report mode is a calibration check: every mismatch it found came from a
test case that expects an error, apart from the threaded string predicates, whose annotations were fixed.

## Not in v1

* Static checking of definitions before they run (v2, a gradual static pass over DownValues).
* Domain types (`Element[x, Reals]`) as types; `library/assume.py` covers those today.
* Inferred types for user definitions; declarations are explicit.
