# Verification (pillar 3, v1)

Two pieces: `Verify[expr]`, which checks one result independently, and an executable form of the `Check:`
sentence that ends every reviewed head's notes (CONVENTIONS §4).

## Verify

`Verify[expr]` evaluates `expr` and checks the result with a method that doesn't reuse the algorithm that
produced it. There are three outcomes, labelled distinctly so a check is never over-claimed:

| Result | Meaning |
|---|---|
| `Certified[value, info]` | An exact certificate was checked with exact arithmetic. |
| `Checked[value, info]` | The check passed, but it rests on symbolic simplification or on numerical sampling. This is strong evidence, not a proof. `info["Evidence"]` says which. |
| `Unchecked[value, reason]` | No checker applies, or the check was inconclusive (too few points where the expression is defined, symbolic parameters in a definite integral, ...). |

A failed check raises `ArithmeticError("Verify: Head result failed verification (...)")`, which `Check[...]`
catches. Errors raised by the computation itself propagate unchanged.

| Head | Method | Best level |
|---|---|---|
| `Integrate[f, x]` | `D[result, x] - f` is zero (simplification, then sampling at 6 points, 40 digits) | Checked |
| `Integrate[f, {x, a, b}]` | `mpmath.quad` at 30 digits | Checked |
| `Solve`, `NSolve` | every solution substituted into every equation. Completeness is **not** checked | Certified when every residual expands to 0 |
| `DSolve[eq, y[x], x]` | the solution substituted into the equation | Checked |
| `D[f, x]` | `mpmath.diff` at sample points | Checked |
| `Limit[f, x -> a]` | values at `a ± 10^-k` (finite limits) | Checked |
| `Series[f, {x, a, n}]` | the remainder shrinks like `h^(n+1)` | Checked |
| `FactorInteger[n]` | the product of the factors is `n`, and each factor has a primality proof: deterministic Miller–Rabin below 3.3·10^24, a Pratt certificate above | Certified (Checked if a factor is only a BPSW probable prime) |
| `PrimeQ[n]` | `True`: the primality proof above. `False`: a factor or a Miller–Rabin witness | Certified |
| `GCD[a, b]` | a Bezout identity `s a + t b == g` with `g` dividing both | Certified |
| `Factor[p]` | the product expands to the input (irreducibility is not checked) | Certified |
| `Simplify`, `FullSimplify`, `Expand`, `Together`, `Apart`, `TrigExpand` | equivalence with the input | Certified (exact expansion) or Checked |
| `Inverse`, `LinearSolve` | multiplying back gives the identity or `b` | Certified (exact entries) or Checked (floats) |
| `Det` | a second algorithm (Berkowitz), or `numpy.linalg.det` for floats | Certified or Checked |
| `Eigenvalues` | each value is a root of the characteristic polynomial (multiplicities only by count) | Checked |

```wolfram
Verify[FactorInteger[2^64 + 1]]
(* Certified[{{274177, 1}, {67280421310721, 1}}, <|"Method" -> "Product of the factors; primality proof of
   each factor", "Certificate" -> {274177 -> "deterministic Miller-Rabin ...", ...}|>] *)
Verify[Integrate[Sin[x]^2, x]]       (* Checked[x/2 - Cos[x] Sin[x]/2, <|"Method" -> "Differentiation", ...|>] *)
Verify[Sort[{3, 1, 2}]]              (* Unchecked[{1, 2, 3}, "no checker for Sort"] *)
```

## Executable Check: lines

`python scripts/verify_library.py` reads each head's `Check:` sentence and runs the methods it names:

| Stated method | What runs |
|---|---|
| `reference` | the head's differential cases in `tests/library` (each module's cases run in order, because some set up state for later ones) |
| `exact` | every case with exact input gives a result without floats |
| `self-check` | the implementation contains its verification step (`ArithmeticError`) |
| `numeric cross-check` | the exact result evaluated to 30 digits matches the head evaluated on 30-digit input |
| `delegated` | the named routine (`delegated (hashlib.scrypt)`) is importable |

Independent tables are added on top, whether a head states a check or not:

* `IDENTITIES`: round trips and identities, such as `FromDigits[IntegerDigits[n, b], b] == n`,
  `Expand[Factor[p]] == Expand[p]`, `D[Integrate[f, x], x] == f`, `Decrypt[k, Encrypt[k, s]] == s`, Pascal's
  rule and `ImportString[ExportString[...]]`.
* `MPMATH`: special functions (`Gamma`, `Zeta`, `BesselJ`, `Hypergeometric2F1`, `EllipticK`, `ProductLog`,
  ...) at exact arguments, compared at 30 digits with mpmath.
* `NUMPY`: `Det`, `Inverse`, `Eigenvalues`, `Tr`, `MatrixRank`, `Mean`, `Variance`, `Median`, ... on float
  data.

Each function gets one status: `pass`, `fail`, `not exercised` (it states a check but nothing could run) or
`no check stated`. The script exits with status 1 on any failure.

* The default mode runs the heads with a `Check:` line plus the tables, in about 15 seconds. The `Verification`
  workflow runs it on every push and pull request.
* `--all` also runs every audited head's reference cases. On main that is 2669 functions in about 6.5 minutes (2325 pass, 0 fail, 344 with no check), so the workflow
  runs it weekly and on demand (`workflow_dispatch`).

`--markdown` and `--json` write the per-function report.

Caveats:

* `exact` is vacuous for a head whose cases are all inexact or expect errors.
* Test modules whose cases only run inside pytest fixtures (a temporary working directory, local servers,
  per-test isolation: `test_wolfram_w27`, `w29`, `w30`) are left out of the harness; their heads are `not
  exercised` there, and their cases still run in the test suite.
* `self-check` only confirms the verification code exists. Its execution is covered by the head's cases.
* Most heads predate the `Check:` convention, so they are reported as `no check stated` (with their reference
  cases, in `--all` mode) until their notes gain the sentence.

## Not in v1

* Provenance on every result (`Provenance[expr]`) and automatic tagging of `Solve`/`DSolve` results without
  `Verify`.
* Typed conjectures (`ForAll` over domain types) and Lean statement export.
