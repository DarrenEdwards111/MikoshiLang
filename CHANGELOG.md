# Changelog

All notable changes to MikoshiLang are recorded here. Dates are UK time.

## [3.8.0] - unreleased

3.8 is a rebuild around one rule: a correct result, a clear error, or left unevaluated, never silently wrong.
It replaces 3.7.1 (27 February 2026), whose "6,324 functions" claim has been retracted.

### Correctness repair
- Parser and evaluator repaired: DownValues ordered by specificity, memoisation, conditional definitions,
  capture-avoiding `Module`/`Block`/`With`, pure functions, `Catch`/`Throw`, attributes, strict lexing (#1, #2).
- Bundled library audited head by head: every exposed function was verified, fixed or quarantined (#4). Generated
  modules `extended4` to `extended37` are quarantined and no longer registered; calling a quarantined operation
  raises `NotImplementedError`. The audit and the wrong answers found and fixed are in `docs/library_audit.md`.

### Wolfram Language coverage
- 4,402 of the 6,716 documented Wolfram Language symbols are now implemented with reviewed, value-tested
  semantics (92.9% of the 4,739 judged feasible). `docs/wolfram_coverage_roadmap.md` is generated from the
  registry by `scripts/wolfram_roadmap.py`, and a test fails when it is stale.
- Added in batches 1-6 (#5, #6, #11): core lists, strings and associations; numerics and special functions;
  linear algebra; calculus and algebra; graphs and statistics; discrete maths.
- Added in waves W01-W32 (#16, #23, #24, #27, #31, #37, #44, #50, #57): polynomials, calculus extras,
  optimisation, special functions I and II, number theory, finite fields and groups, text, dates and tabular
  data, Import/Export and cryptography, graphics, plots, geometry, images, distributions, statistics, signal
  processing, graphs II, regions, random processes, PDE-related heads, files, system and notation.
- Language additions include optional and default patterns (`x_.`, `x_:v`), `|` and `..`, `Span` (`;;`), the `.`
  operator, associations, UpValues/TagSet, `Map`/`Apply` at other levels, multivariate `Series`, and `Limit`
  with `Direction`.

### Python interop
- Phase 1 (#34): `Session.eval` returns Python data (`Fraction`, lists, dicts, optional NumPy arrays),
  `Session.ns` gives Pythonic calls, `@session.function` turns typed Python functions into heads, and
  `PythonFunction`/`PythonImport` are gated by an `allow_python` allow-list. There is no `eval`/`exec`.
- Phase 2 (#36): packed NumPy arrays (`Dot`, `Total` and elementwise operations run in NumPy), session-local
  `PythonObject` handles and `@mikoshilang.function` for global heads.

### Types
- Library implementations annotated (#38); signatures read from annotations, `TypeSignature` declarations and
  an opt-in checker, `Session(check_types=True | "report")` (#40). Symbolic arguments are never type errors.
- Follow-ups: the ``Mikoshi` `` context, sign enforcement, `TypeCheck` (#46), kernel built-in signatures (#51),
  and narrower parameter and return types for the algebra and calculus modules (#67).

### Verification
- `Verify[expr]` re-checks a result with an independent method and returns `Certified`, `Checked`, `Unchecked`
  or `Failed[value, info]` (#41, #46). Methods include Pratt certificates or deterministic Miller-Rabin for
  `FactorInteger`, Bezout for `GCD`, substitution for `Solve`, differentiation for `Integrate` and Berkowitz
  for `Det`.
- Every reviewed head carries an executable `Check:` note, run by `scripts/verify_library.py` and by the
  Verification workflow on every push.

### Packaging, docs and CI
- The wheel now ships `mikoshilang.library` and the Jupyter `kernel.json`; CI tests a wheel install from outside
  the source tree and a minimal install without extras, on Python 3.9, 3.12 and 3.13 (#29).
- The offline test suite now collects 27,742 tests, most of them value-level Wolfram-semantics cases.
- Project URLs point to GitHub (the old `mikoshi.co.uk` homepage and "Try It Online" links returned 404).
- New README introduction, an `examples/` folder, this changelog, and a manual updated for 3.8. Pre-3.8
  completion reports moved to `docs/history/`.

### Compatibility notes
- `Solve` returns rule lists (previously bare values).
- `Inverse`/`Transpose` return plain nested lists (previously a `Matrix` head).
- `Log[b, z]` is the base-b logarithm (the old bridge had the arguments reversed).
- `Exp[x]` evaluates to `E^x`; sums and products are n-ary rather than nested binary.
- Network access is off by default: use `Session(allow_network=True)` for the knowledge layer.

## [3.7.1] - 2026-02-27

Last release of the pre-audit series (0.1.0 to 3.7.1, February 2026). Its function-count claims were retracted
in 3.8; see `docs/history/`.
