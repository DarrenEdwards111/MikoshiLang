# October 2026 correctness repair

This change repairs the language core and **reduces the supported library**, rather
than claiming to implement thousands of generated functions. It is a development
version and has not been published to PyPI or deployed to Research Labs.

## Audit disposition

| Finding | Disposition |
| --- | --- |
| Skipped characters/trailing input | Strict lexer, full parse consumption, explicit ParseError |
| Definitions and typed/sequence patterns inert | Session-owned definitions and parsed-pattern compilation |
| Replacement, Map, Apply, If, pure functions missing | Implemented core forms, lazy branches, operator aliases |
| Lists/destructuring broken | Real parsed pattern adapter; tested Distance2D core implementation |
| N precision reduced to float | High-precision SymPy results retained |
| Equal not translated to Eq | Explicit trusted AST conversion; Solve equation regressions |
| Three-factor Times truncated | SymPy Mul over all factors |
| Guessed iteration bounds | Integer bounds required, zero step rejected, explicit output cap |
| Placeholders and numbered filler | Generated modules 4–37 quarantined from dispatch |
| False curvature/topology results | Explicit unavailable errors, not invented values |
| Incorrect generated overloads shadowing real algorithms | Generated modules no longer registered; genuine core SHA256/graph algorithms retained |
| Flat scan over thousands of rules | Head-indexed extension dispatch and top-level rule application |
| Symbolic identity rewrite loops | No self-rewrite recursion; no unbounded evaluator memo |
| Import creates knowledge cache | Cache initialisation deferred to explicit knowledge operations |
| Evaluation invokes network unexpectedly | Knowledge forms require explicit Session opt-in |
| Tests ignore expected values | Sample runner compares expected values; no-expected cases unverified |
| No CI / licence file | Offline pytest/build workflow and Apache-2.0 text added |
| Generated artifacts checked in | Removed from Git index; ignored, not deleted from local disk |

## Deliberate limitations

- This is a subset language, not Wolfram Language compatibility. Unsupported syntax
  is rejected; unknown user heads remain symbolic. Not all special forms, attributes,
  pattern conditions, lexical scoping constructs or differential-equation notation exist.
- Retained legacy wrappers still have uneven testing, optional dependencies and some
  legacy Error-value conventions. The function counter is not a verification certificate.
- Quarantined source is retained for future individual review. Directly importing and
  invoking that historical implementation bypasses the supported dispatcher contract.
- Knowledge integration tests depend on external APIs and mutable data. They are
  explicitly marked live_service and excluded from offline CI. Offline cache and
  opt-in behaviour have deterministic regression tests.
- Iteration/depth bounds are not time/memory isolation. Hosts evaluating untrusted
  expressions must supply an external sandbox. Python APIs are not a security boundary.
- No typed conjecture DSL, Lean translator, quantifier semantics or proof certification
  is added. Research Labs must not use notebook results as proof evidence.

## Compatibility

Use one `Session` per user/notebook. The low-level legacy evaluator retains its shared
default definitions; it should not be shared between tenants. Session knowledge calls
are disabled by default. Generated-library calls previously returning fabricated values
now fail. High-precision results are SymPy numbers instead of Python machine floats.
Top-level newlines separate statements. Consult README for supported syntax.
