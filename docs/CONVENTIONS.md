# MikoshiLang function conventions (types, errors, verification, tests)

Status: draft, 31 Oct 2026; updated 1 Nov 2026 for the types and verification follow-ups (§2 checker, §3 `Verify` returns `Failed`, §4 self-check runs an example). Applies to every new or reworked head in `mikoshilang/library/`, starting with the
current Wolfram batches. Checked against `main` @ `253e579` (after the W05/W06/W08 merge `c3bfb03` and W07/W10).
It needs **no change to shared files**: everything here works with today's `core.fn` registry, `STATUS`,
`dispatch`, `wl_errors.py` and the `tests/library/_harness.py` harness. Later registry work (signatures read from
these hints, a `MikoshiError` hierarchy, `Failure` values in `Enclose`) can then pick it up mechanically.

The rule this all serves is the one already in `docs/library_audit.md`: a result is correct, or a clear error, or
the call stays unevaluated because the input is symbolic. It is never silently wrong.

---

## 1. Registration (unchanged)

```python
@fn("Name", "Alias", area=AREA, status="added", listable=False, hold=False, notes="...")
def _name(x: Type, *rest: Option) -> ResultType:
    ...
```

* `status`: new Wolfram symbols are `added`. A reviewed legacy name is `promoted`, a corrected one is `fixed`.
  `error` (via `core.error(...)`) is for a reserved name that only raises.
* `notes` is user-facing and ends up in `docs/library_audit.md`. It must state the supported forms, what stays
  unevaluated, what raises, deliberate deviations from Wolfram, and a final **`Check:`** sentence (§4).
* One module per batch, with `from __future__ import annotations` at the top (every annotated `wl_*` module has it).

## 2. Types: declare them in the Python signature

Annotate **every parameter and the return** of each registered implementation using the vocabulary below.
Because of `from __future__ import annotations`, the annotations are stored as strings and never evaluated, so
**no import is needed** and `core._arity` (`inspect.signature`) is unaffected. That was verified on Python 3.9
and 3.13. These hints are the signature: the planned `Signature[...]` / registry `signature` field will be
derived from them. Don't duplicate them in notes.

| Type | Accepts (after evaluation) |
|---|---|
| `Expr` | anything (use only when truly unconstrained) |
| `Integer`, `PosInteger`, `NonNegInteger` | `int` / `sympy.Integer` (never `bool`), with sign constraint |
| `Rational` | exact integer or rational |
| `Real` | exact or inexact real number (`1/3`, `Pi`, `2.5`, bignum) |
| `Number` | any numeric value incl. complex |
| `Numeric` | an expression that `N` makes a number (`Sqrt[2] + Pi`) |
| `String`, `Boolean` | `str`; `True`/`False` |
| `Var` | a `Symbol` used as a variable (`x` in `D[f, x]`) |
| `ListOf[T]`, `MatrixOf[T]`, `Assoc` | `List` with elements of type T; rectangular list of lists; `Association` |
| `Rule`, `Option` | `lhs -> rhs`; a trailing `Name -> value` option (annotate `*opts: Option`) |
| `Fn` | anything applicable: symbol, `Function`, pure function |
| `Held` | an unevaluated argument (heads registered with `hold=True`) |
| `A \| B` | either (`Integer \| ListOf[Integer]`) |

`Unevaluated` is never a return type: it is control flow (§3), not a value.

Representation types (above) are kept separate from *mathematical domain* claims (`Element[x, Reals]`). If a
function needs a domain assumption, say so in notes. Domain types arrive later with the verification work.

The **body enforces the signature** with the error classes in §3, in this order:

1. wrong *kind* of argument (a string where a number is required, a non-list): `TypeError`;
2. symbolic argument where a concrete value is required (`n`, `n + 1`, `Sqrt[k]`): `raise Unevaluated()`;
3. right kind, bad value (negative where `NonNegInteger`, singular matrix, bad option value): `ValueError`;
4. a valid input we don't handle (a range Wolfram supports but we don't, an option value, a method): `NotImplementedError`.

Wrong argument counts are handled by `core.dispatch`. Keep implementation arity honest (use defaults or `*rest`).

With the opt-in checker on (`Session(check_types=True)`, `TypeCheck`), a concrete argument that doesn't match the
annotation is reported before the body runs, sign constraints included (`0` is not a `PosInteger`). The body
must still raise its own `TypeError`/`ValueError`, because checking is off by default.

## 3. The error model

**One model:** an error is a Python exception at the Python boundary (`Session.evaluate` raises). Inside the
language it is catchable, and the class says what kind of failure it is. Function authors only *raise*; the
conversion to messages and `Failure` values happens centrally.

| Situation | Raise today | Future class (subclass of the left) | Message key |
|---|---|---|---|
| wrong kind of argument | `TypeError` | `ArgumentTypeError` | `f::argtype` |
| invalid value / option value / wrong shape | `ValueError` | `ArgumentError` / `DomainError` | `f::argx` / `f::dom` |
| unimplemented feature, range, option or method | `NotImplementedError` | `Unsupported` | `f::unsupported` |
| a self-check of the result failed | `ArithmeticError` | `VerificationFailed` | `f::verify` |
| `Verify[expr]` found the result wrong | (not raised: returns `Failed[value, <\|"Head" -> f, "Reason" -> ...\|>]`) | none | none |
| symbolic input that a concrete value is needed for | `raise Unevaluated()` (from `.core`) | (not an error) | none |

Rules:

* **Message text starts with the head name**: `raise ValueError("Name: argument 2 must be a square matrix")`.
  `wl_errors.error_key` parses that prefix, so `Check[...]`, `ConfirmQuiet` and `VerificationTest` report it as
  `Name::evalerr` today. (Verified: `Check[AlternatingFactorial[-1], $Failed]` gives `$Failed` and prints
  `AlternatingFactorial::evalerr`.)
* **Unevaluated only for symbolic input.** Never return the call unevaluated to mean "not supported" or "bad
  argument", and never return `$Failed`, `Indeterminate`, `Null` or a string as an error stand-in. Plain wrong
  input that isn't symbolic (`"a"`, `2.5` where an integer is required) raises.
* **Unsupported means it raises.** Partial support is fine, as long as everything outside it raises
  `NotImplementedError("Name: ... is not supported")` and is listed in notes. A symbol you can't do correctly is
  left out of the batch and listed with the reason. Don't register a stub.
* **Wolfram "message + unevaluated" behaviour is a deliberate deviation.** Where Wolfram prints `f::tag` and
  returns the input (e.g. `Part::partw`), we raise. Mention it in notes if users would notice.
* **Don't build `Failure[...]` objects yourself** unless returning one is the documented *result* of the function
  (`Confirm*`, `Enclose`, `Failure`/`Success`, `Import` of a missing file, `Verify`...). Today only `Check`/`ConfirmQuiet`
  catch raised errors inside the language (verified: `Enclose`, `Catch` and `Quiet` let them propagate). Turning
  raised `MikoshiError`s into `Failure["Name::tag", <|...|>]` inside `Enclose`/`Confirm` is roadmap work in
  `wl_errors.py`, not batch work.
* **Messages (`wl_errors.issue`) are for warnings only.** A warning is when the returned value is still correct
  and complete, e.g. precision loss or an option being ignored with a stated effect. If the result might be
  incomplete or approximate, raise instead. Register the text without editing `wl_errors.py`:
  `from .wl_errors import BUILTIN_MESSAGES, issue`, then `BUILTIN_MESSAGES[("Name", "tag")] = "text with `1`"`
  at import, and `issue("Name", "tag", arg)` at the call site.
* **`Verify` is the one checker that returns its failures.** A function's own self-check (§4) still raises
  `ArithmeticError("Name: result failed verification")`, because a wrong value must never flow on. `Verify[expr]`
  exists to *report* on a result, so a failed independent check is its documented result:
  `Failed[value, <|"Head" -> h, "Reason" -> text|>]`, next to `Certified`, `Checked` and `Unchecked`. Errors
  raised while computing `expr` propagate from `Verify` unchanged.
* Don't catch broad `Exception` to hide failures. Re-raise as one of the classes above with a clear message.

## 4. Verification hook: every function says how its result is checked

End `notes` with one `Check:` sentence naming one or more methods. Tests must exercise that method.

| Method | Meaning | Example |
|---|---|---|
| `exact` | exact algorithm by construction (exact SymPy/integer arithmetic), no floating point | `PolygonalNumber` |
| `self-check` | the code verifies its own result before returning (substitute back, multiply out, differentiate), and raises `ArithmeticError("Name: result failed verification")` on mismatch. `scripts/verify_library.py` looks for the step and runs one real (non-error) example of the head | `ApartSquareFree`, `PartialFractionElements` in `wl_polyalg.py`; `DSolve` singular solutions |
| `numeric cross-check` | a symbolic result is compared numerically at sample points in code | `wl_calculus2.py` contour/line integrals |
| `reference` | tests compare against values written independently: Wolfram documentation examples, OEIS, DLMF, mpmath/SciPy/NetworkX | most heads |
| `delegated` | thin wrapper over a trusted library routine; name it (`sympy.ntheory.totient`) | |

Prefer `self-check` wherever a cheap independent check exists (solving, decomposition, inversion, integration,
transforms). It is the hook the verification pillar will build provenance on (`Checked` / `Unchecked`).
Numerical results state their precision rule (machine vs. bignum via `core.numeric_out`).

**Notes built in code.** When a head is registered in a loop or by a factory, so its notes are not one literal
you can edit, state the check with `core.add_check` at module level, after the registrations and in the module
that sets the head's final notes:

```python
from .core import add_check as _add_check

_add_check("reference (differential cases in tests/library).", "JacobiSN", "JacobiCN", "JacobiDN")
```

It appends `Check: <sentence>` to the notes in `STATUS` and on the registry entry. It raises `ValueError` if a
head is not registered or already states a check, so it can't silently claim a method twice. As with literal
notes, name only methods that `scripts/verify_library.py` passes for that head.

**Which head a test case counts towards.** `scripts/verify_library.py` groups the `CASES` in `tests/library`
with `_harness.head_of`:

- `F[...]` counts towards its outermost head `F`.
- A bare symbol or `$`-variable (`"Pi"`, `"$MachinePrecision"`) counts towards that symbol. Use it as a value
  check (a `str` or numeric expected value) or a type check (`Pred(lambda v: isinstance(v, float), "a machine
  real")`).
- A wrapper case counts towards the outer head (`ToString`, `N`, ...) unless it is written as
  `Via("G", "ToString[G[...]]")`. A `Via` case counts towards the inner head `G` only when the expected value
  depends on `G`'s own output: the harness re-runs it with `G` replaced by an inert `ViaProbe` head, and the
  case fails if the expectation still holds (`Via("Sin", "Length[{Sin[1], 2}]")` with expected `2` fails).
- Setup lines with no head (`x = 3`) count towards nothing.

## 5. Required tests (`tests/library/test_wolfram_<id>.py`)

Already enforced by `tests/library/test_coverage.py`: at least **3 differential cases per head**, and a `Raises`
case for every `error`/quarantined head. Add the module to `MODULES` there and in
`scripts/library_audit_report.py`. On top of that, every head needs:

* **reference cases** whose expected values were written independently of the implementation (cite the source in
  a comment: Wolfram doc page, OEIS id, mpmath call). Never paste the implementation's own output as the expectation;
* one **symbolic case** showing it stays unevaluated (if the head has numeric arguments);
* one **`Raises` case per error class** the head documents (`Raises(TypeError)`, `Raises(NotImplementedError)`,
  ...). Name the class, don't use bare `Raises()`;
* a test of the **`Check:` method**: identities or round trips via `Pred`. Self-checks are covered by normal cases
  passing through the verification code;
* for options: one case per supported option value and a `Raises` for an unsupported one;
* the full offline suite green on Python 3.13 and 3.9 (the CI matrix). Avoid 3.10+ syntax at runtime
  (`match`, `X | Y` outside annotations).

## 6. Worked example

The example is `AlternatingFactorial` (it already exists in `wl_combinat.py`; this shows it written to these
conventions, so don't re-register it). The behaviour below was run against main @ `253e579` under a scratch name.

```python
"""Wolfram batch Wxx: combinatorial integer sequences.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md).
"""
from __future__ import annotations

from .core import fn, Unevaluated, is_numeric_value
from .. import kernel as K

AREA = "combinatorics (Wxx)"


def _nonneg_int(x, fname, pos=1):
    """Enforce NonNegInteger: int for a non-negative integer, Unevaluated for symbolic input, else raise."""
    if isinstance(x, (bool, str)):
        raise TypeError(f"{fname}: argument {pos} must be an integer, not {x!r}")
    if K.is_int(x):
        if int(x) < 0:
            raise NotImplementedError(f"{fname}: negative integers are not supported (argument {pos} is {x})")
        return int(x)
    if is_numeric_value(x):
        raise NotImplementedError(f"{fname}: non-integer argument {x!r} is not supported")
    raise Unevaluated()            # n, n + 1, ...: stays unevaluated, as in Wolfram


@fn("AlternatingFactorial", area=AREA, status="added", listable=True,
    notes="AlternatingFactorial[n] = Sum[(-1)^(n - k) k!, {k, n}] for integers n >= 0 (OEIS A005165); symbolic n "
          "stays unevaluated; negative or non-integer n raises (Wolfram also evaluates real n: not supported). "
          "Check: exact; reference values (OEIS A005165) and the identity a(n) + a(n - 1) = n! in tests.")
def _alternating_factorial(n: NonNegInteger) -> Integer:
    k = _nonneg_int(n, "AlternatingFactorial")
    out, f = 0, 1
    for j in range(1, k + 1):
        f *= j
        out = f - out
    return out
```

Tests (`tests/library/test_wolfram_wxx.py`):

```python
import math
import pytest
from _harness import Pred, Raises, check, ids

def _af(n):  # independent definition, not the implementation's recurrence
    return sum((-1) ** (n - k) * math.factorial(k) for k in range(1, n + 1))

CASES = [
    ("AlternatingFactorial[Range[0, 7]]", "{0, 1, 1, 5, 19, 101, 619, 4421}"),   # OEIS A005165
    ("AlternatingFactorial[25]", str(_af(25))),                                   # independent sum
    ("AlternatingFactorial[n]", "AlternatingFactorial[n]"),                        # symbolic: unevaluated
    ("AlternatingFactorial[-1]", Raises(NotImplementedError)),                     # unsupported range
    ("AlternatingFactorial[2.5]", Raises(NotImplementedError)),                    # unsupported (Wolfram: numeric)
    ('AlternatingFactorial["a"]', Raises(TypeError)),                              # wrong kind
    ("Table[AlternatingFactorial[k] + AlternatingFactorial[k - 1] == k!, {k, 1, 30}]",   # Check: identity
     Pred(lambda g: all(v is True for v in g.args), "a(n) + a(n-1) = n!")),
]

@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_wxx(case):
    check(case)
```

For a `self-check` example in real code, see `ApartSquareFree` in `mikoshilang/library/wl_polyalg.py`: it
recombines the decomposition, and `raise ArithmeticError("ApartSquareFree: decomposition failed verification")`
if `simplify(result - input) != 0`.

## 7. Checklist per head

- [ ] Every parameter and the return annotated with §2 types
- [ ] Argument checks in §2 order; errors raised with the §3 classes and a `"Name: ..."` message
- [ ] Unevaluated only for symbolic input; no stubs, no `$Failed`/string error stand-ins
- [ ] notes: forms, unevaluated cases, raised cases, deviations, `Check:` sentence
- [ ] ≥3 independent reference cases, a symbolic case, one `Raises(Class)` per error class, a Check test
- [ ] Module added to `MODULES` (test_coverage.py, library_audit_report.py); suite green on 3.13 and 3.9
