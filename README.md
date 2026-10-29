# MikoshiLang

A small Wolfram-style symbolic language backed by SymPy, built by Mikoshi Ltd.
**Alpha software, not a Mathematica replacement or a proof verifier.**

## Correctness repair (3.8.0 development)

The previous “6,324 working functions” claim was not supported by the implementation.
Generated modules `extended4` through `extended37` are now **quarantined**: they are
retained as historical source but are not registered by the evaluator. This disables
placeholder constants, identity stubs, numbered filler and unreviewed formulas.
Calling a quarantined operation raises `NotImplementedError`, unless an independently
implemented core overload exists (for example SHA256 and graph operations).
The retained `extended`, `extended2`, and `extended3` wrappers are not all formally
verified; consult tests and check results for consequential uses.
Historical manuals and completion reports describe the old version and are not a
current capability specification. This README supersedes their function-count claims.

## Install and run

```sh
python -m pip install .
# Optional notebook, plotting, scientific and graph dependencies:
python -m pip install '.[all]'
mikoshilang
```

```python
from mikoshilang import Session
s = Session()                       # isolated definitions; network disabled
assert s.evaluate('f[x_] := x^2; f[3]') == 9
assert s.evaluate('x^2 /. x -> 3') == 9
print(s.evaluate('Map[#^2 &, {1,2,3}]'))  # {1, 4, 9}
print(s.evaluate('Solve[x^2 == 4,x]'))   # {-2, 2}, values, not Wolfram rule lists
print(s.evaluate('N[Pi,50]'))            # high-precision SymPy number, not float
```

## Language contract

- Strict lexing and end-of-input checks: unknown characters and extra closing
  delimiters are errors. Unsupported derivative-prime notation is rejected, not dropped.
- Arithmetic, comparisons, lists, calls, assignments, delayed definitions,
  named/typed/sequence patterns, replacement rules (`->`, `:>`, `/.`), lazy `If`,
  `Map`, `Apply`, `/@`, `@@`, and slot-based pure functions (`#`, `#2`, `&`).
- Top-level newlines or semicolons separate statements. Newlines inside brackets
  or after an operator permit continuation. This is a subset of Wolfram syntax,
  not a claim of full Wolfram compatibility. `Module`, `Do`, `While`, arbitrary
  conditions, postfix `//`, and derivative-prime syntax are not implemented.
- Exact integer/rational arithmetic and basic canonical algebra use SymPy.
  Decimal literals are machine precision; requesting more digits cannot recover
  information already lost in a decimal input. `N[...,p]` for p > 15 preserves
  SymPy precision. The general bridges do not guess exact rationals from floats.
- Unknown user function heads remain symbolic, not “successfully computed”.
  Definitions belong to a Session; notebooks and REPLs each use their own Session.
  The legacy `evaluate`/`parse_and_eval` API uses a shared default session.
- `Range`/`Table` require integer bounds and nonzero steps; no guessed 1..10 bounds.
  Their output is limited to 100,000 entries. This is not a complete resource sandbox.
- Core numerical/algebra/calculus, linear-algebra, chemistry, units, signal and
  plotting APIs remain available. Optional dependencies may be required.
- `Distance2D` has an explicit two-point implementation. Advanced curvature and
  topology placeholders remain unavailable rather than returning invented answers.

## Knowledge and hosting

Knowledge access is explicit: `Session(allow_network=True)`. It may contact public
APIs and create a local SQLite cache on first use. Imports and ordinary arithmetic
no longer initialise that cache. JHTDB entries are a static catalogue, not simulations.
The direct Python knowledge APIs are still online APIs; the Session flag does not
sandbox arbitrary Python or those direct calls.

**Do not treat this package as a trusted conjecture-to-Lean translator.** There is
no typed quantifier/domain language or verified Lean translation here. Symbolic
outputs are suggestions, not proofs. Any Research Labs integration must freeze and
independently check the intended Lean statement. Untrusted evaluation requires an
external process sandbox and time/memory limits; recursion and iteration limits
alone do not provide that isolation.

## Tests

```sh
python -m pip install '.[all]' pytest build
python -m pytest -q              # offline tests, including audit regressions
python -m build                 # sdist and wheel
# Optional, mutable external-service integration tests (may fail offline):
python -m pytest -m live_service tests/test_knowledge.py tests/test_knowledge_full.py
```

GitHub Actions runs offline tests and package builds on Python 3.9 and 3.12.
New regressions check actual values, precision, malformed-input rejection, session
isolation, lazy branching, quarantine and offline imports. Legacy shape-only tests
are not evidence that every historical library entry works. `test_all_functions.py`
now compares expected answers and labels samples without expectations unverified.
`count_functions.py` reports active registry names, not a verified-function count.

## Licence

Apache-2.0; see [LICENSE](LICENSE).
