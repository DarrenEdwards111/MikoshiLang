# Public Python API robustness audit

## Scope and evidence

This is a **bounded audit of every name in `mikoshilang.__all__`**, based on commit
`8fc1b2c29227d66fa1c2e66c64a8f811849687f2`. The inventory contains **108 exports:
91 functions, 12 classes, and 5 constants**. It is not an audit of every registered
language head, every overload, or every method on the exported classes.

The machine-readable evidence is [`tests/public_api_audit.json`](../tests/public_api_audit.json).
Each export has named test functions, checked behavior, findings and limitations.
[`test_public_api_inventory.py`](../tests/test_public_api_inventory.py) rejects missing,
duplicate or stale records and verifies that the five batch declarations partition
the actual public surface. This guard establishes inventory completeness, **not**
semantic completeness; real value/property assertions are in the batch suites.

| Batch | Exports | Suite |
|---|---:|---|
| Expressions, patterns, parser, sessions and interop | 24 | `tests/test_public_api_core.py` |
| Algebra, calculus, numerics and constants | 15 | `tests/test_public_api_math.py` |
| Lists, statistics and linear algebra | 21 | `tests/test_public_api_data_linalg.py` |
| Elementary functions, number theory, logic and comparisons | 29 | `tests/test_public_api_builtins.py` |
| Units, chemistry and signal processing | 19 | `tests/test_public_api_science.py` |

## Method

- Call the public Python entry points directly; a passing `Session` test is not
  assumed to exercise the same implementation.
- Use known values, exact arithmetic, full Boolean truth tables, polynomial
  arithmetic, complex residuals, transform roundtrips, and bounded generated
  properties. SymPy/backend comparisons are supplemental rather than independent
  proof of the same backend's correctness.
- Check `Session` parity only when the two interfaces promise the same contract.
- Test empty/negative/zero/complex/tiny/huge/malformed inputs where appropriate.
  Use zero absolute tolerance when a tiny nonzero result must not pass as zero.
- Put potentially hanging workloads in killable subprocesses with a 20-second,
  startup-inclusive timeout. This does not provide a general resource sandbox.
- Observe a failing regression before changing a confirmed bug; retain already
  working cases as compatibility coverage without claiming those were failures.

As an additional independent check, 13 selected regression cases were run against
an isolated copy of the unchanged base package. All 13 failed there; all pass in
the repaired combined suite. These cover assumption-bearing calculus variables,
GCD/LCM, packed unsigned overflow, Part zero, chemical balances, complex convolution
and tiny unit conversion. The baseline import path was asserted before execution.

## Confirmed repairs

- Preserve assumption-bearing SymPy variable identity in `diff`, `integrate`,
  `solve`, `limit` and `series` instead of silently using a different symbol.
- Validate requested numerical precision before the `MachinePrecision` special
  case and reject nonfinite precision consistently.
- Correct GCD/LCM signs, zero/empty identities and integer validation; reject
  noninteger Fibonacci indices instead of truncating analytic values.
- Preserve symbolic Boolean/comparison expressions rather than using Python
  truthiness or incorrectly resolving unknown equality.
- Return the list head for `Part[..., 0]`, including empty and nested lists.
- Reject unsigned packed integers that would wrap into negative int64 values.
- Preserve tiny unit conversion values; recognize canonical derived SI units;
  reject ambiguous offset-temperature arithmetic rather than producing a false
  result. Offset conversion remains supported.
- Consume neutral molecular formulas fully and reject unsupported/malformed
  notation. Balance uniquely solvable positive equations with exact rational
  arithmetic rather than rounded SVD vectors.
- Preserve complex components in signal conversion, convolution, filters and
  spectrograms; validate relevant input domains and include the raw power matrix
  in the no-matplotlib spectrogram result.

## Verification commands

```sh
python -m pytest -q tests/test_public_api_*.py tests/test_python_*_regressions.py
python -m pytest -q
python -m build --outdir dist
python -m pip install --force-reinstall --no-deps dist/*.whl
python scripts/check_public_api_wheel.py --work-dir <scratch-directory>
```

Final targeted validation on Windows / Python 3.14:

- **435 new audit cases** (including 5 inventory guards), plus the **118 existing
  direct-Python regressions**: **553 passed**, with warnings treated as errors.
- Separate minimal environment with no SciPy, Hypothesis or matplotlib:
  **483 passed, 70 explicitly skipped**, also with warnings treated as errors.
  Optional skips do not imply those features were tested in that environment.
- Built both sdist and wheel successfully. Source archive inspection confirmed
  that the audit manifest, wheel-check script and report ship alongside the tests.
- The final installed-wheel results are reported with the pull request; the
  wheel runner asserts a `site-packages` import from outside the checkout.

The dedicated `Public Python API robustness` workflow tests source and installed
wheels on Ubuntu and Windows with Python 3.9, 3.12 and 3.13, plus minimal installs
on both operating systems. Declaring that matrix is not itself evidence it passed;
consult the associated GitHub Actions results. The installed-wheel runner copies
only the relevant tests and manifest outside the source checkout, removes
`PYTHONPATH`, and asserts the package is imported from `site-packages`.

Full-suite baseline comparison, final wheel results and CI conclusions are reported
with the pull request; do not infer a globally green Windows suite from this
focused audit. The previous baseline already had platform-specific failures.

## Deliberate limits and compatibility differences

- No licensed Wolfram kernel was used as a differential oracle. These tests do
  not establish full Wolfram Language compatibility.
- Python `solve` returns roots rather than Session rule lists; Python `series`
  removes order terms; direct `limit` retains its right-hand backend default.
- Direct DFT/IDFT retain NumPy's normalization/sign convention, not Session's
  Wolfram Fourier convention.
- Chemistry is limited to supported neutral formulas and uniquely solvable,
  strictly positive balances. Charges, isotope notation and hydrates are not
  silently interpreted as ordinary formulas.
- Units remain floating point; offset temperature arithmetic is unsupported,
  and temperature differences are not modeled.
- Numerical/property cases are bounded. Passing them does not establish broad
  conditioning, high-order filter stability, solver completeness, or correctness
  for all inputs. Class records cover the exercised behaviors, not all methods.
- The thousands of registered library heads remain outside this public-surface
  pass. Their existing library suites are exercised by the full-suite run, but
  that is not a new head-by-head robustness audit.
