"""Verification pillar V1: seeded wrong results are caught; the Check: harness parses and runs stated checks."""
from pathlib import Path

import pytest

from mikoshilang import Session
from mikoshilang.expr import Expr, Symbol
from mikoshilang.library import wl_verify as V
from mikoshilang.parser import parse
from mikoshilang import verification as H

TESTS = Path(__file__).resolve().parent / "library"


def p(src):
    return Session().evaluate(src)


# Seeded wrong results: each checker must reject them (VerificationFailure; Verify returns Failed[...]) -------

@pytest.mark.parametrize("head, args, wrong", [
    ("Integrate", ["Sin[x]^2", "x"], "x/2 + Sin[x] Cos[x]/2"),            # sign error
    ("Integrate", ["x Exp[x]", "{x, 0, 1}"], "2"),                          # true value 1
    ("Solve", ["x^2 == 2", "x"], "{{x -> Sqrt[2]}, {x -> 2}}"),
    ("Solve", ["x^2 == 2", "x"], "{{x -> Sqrt[2]}}"),                       # incomplete: root counting
    ("Solve", ["x^3 == x", "x", "Reals"], "{{x -> 0}, {x -> 1}}"),          # incomplete over the reals
    ("Solve", ["x^2 == 4", "x"], "{}"),                                     # no solutions returned
    ("NSolve", ["x^3 - 2 == 0", "x"], "{{x -> 1.2599210498948732}}"),      # real root only
    ("D", ["Sin[x] x^2", "x"], "2 x Sin[x] - x^2 Cos[x]"),
    ("Limit", ["Sin[x]/x", "x -> 0"], "2"),
    ("Series", ["Exp[x]", "{x, 0, 3}"], "1 + x + x^2/2 + x^3/7"),
    ("FactorInteger", ["360"], "{{2, 3}, {3, 2}, {7, 1}}"),
    ("FactorInteger", ["91"], "{{91, 1}}"),                                  # 91 = 7 * 13 is not prime
    ("PrimeQ", ["91"], "True"),
    ("PrimeQ", ["97"], "False"),
    ("GCD", ["12", "18"], "3"),
    ("Factor", ["x^2 - 1"], "(x - 1)^2"),
    ("Simplify", ["Sin[x]^2 + Cos[x]^2"], "2"),
    ("Inverse", ["{{1, 2}, {3, 4}}"], "{{-2, 1}, {3/2, 1/2}}"),
    ("LinearSolve", ["{{1, 2}, {3, 4}}", "{5, 6}"], "{-4, 4}"),
    ("Det", ["{{1, 2}, {3, 4}}"], "2"),
    ("Eigenvalues", ["{{2, 1}, {1, 2}}"], "{3, 2}"),
    ("DSolve", ["y'[x] == y[x]", "y[x]", "x"], "{{y[x] -> C[1] Exp[2 x]}}"),
])
def test_seeded_wrong_results_are_caught(head, args, wrong):
    a = [p(s) for s in args]
    with pytest.raises(V.VerificationFailure, match=rf"^Verify: {head} result failed verification"):
        V.CHECKERS[head](a, p(wrong))


def test_verify_returns_failed_for_a_wrong_result():
    from mikoshilang.library import core
    impls = core.REGISTRY["GCD"].impls
    saved = impls[0]
    impls[0] = (lambda *a: 3,) + tuple(saved[1:])               # a seeded bug in GCD
    try:
        out = p("Verify[GCD[12, 18]]")
    finally:
        impls[0] = saved
    assert out.head == "Failed" and out.args[0] == 3
    assert "Head -> GCD" in str(out).replace('"', "") and "Reason" in str(out)
    assert p("Verify[GCD[12, 18]]").head == "Certified"


def test_root_counting_completeness():
    out = V.CHECKERS["Solve"]([p("x^2 + 1 == 0"), p("x"), p("Reals")], p("{}"))
    assert out.head == "Certified" and "no real roots" in str(out)
    out = V.CHECKERS["Solve"]([p("x^2 == 0"), p("x")], p("{{x -> 0}, {x -> 0}}"))   # multiplicity: 1 distinct root
    assert out.head == "Certified" and "1 distinct complex" in str(out)
    out = V.CHECKERS["Solve"]([p("a x == 1"), p("x")], p("{{x -> 1/a}}"))            # parameter: count unknown
    assert "not checked" in str(out)


def test_right_results_pass_the_same_checkers():
    assert V.CHECKERS["GCD"]([12, 18], 6).head == "Certified"
    assert V.CHECKERS["PrimeQ"]([91], False).head == "Certified"
    assert V.CHECKERS["FactorInteger"]([91], p("{{7, 1}, {13, 1}}")).head == "Certified"


def test_pratt_certificate_is_checkable():
    """An independent check of a Pratt certificate (n, a, {certs of the prime factors of n - 1})."""
    def valid(c):
        if len(c.args) == 1:
            q = c.args[0]
            return V._det_mr(q) and q < V._DET_MR_BOUND
        n, a, subs = c.args
        qs = [s.args[0] for s in subs.args]
        m = n - 1
        for q in qs:
            while m % q == 0:
                m //= q
        return m == 1 and pow(a, n - 1, n) == 1 and all(pow(a, (n - 1) // q, n) != 1 for q in qs) and \
            all(valid(s) for s in subs.args)
    for n in (2 ** 89 - 1, 10 ** 30 + 57, 2 ** 127 - 1):
        c = V._pratt(n)
        assert c is not None and valid(c), n


def test_verify_holds_its_argument():
    s = Session()
    s.evaluate("count = 0; f[x_] := (count++; x^2)")
    s.evaluate("Verify[Integrate[f[y], y]]")
    assert s.evaluate("count") == 1                      # evaluated once, by Verify


# The Check: harness ------------------------------------------------------------------------------------

@pytest.mark.parametrize("notes, methods", [
    ("... Check: exact; reference values (OEIS A005165) in tests.", ["exact", "reference"]),
    ("... Check: delegated (hashlib.scrypt); RFC 7914 reference in tests.", ["reference", "delegated"]),
    ("... Check: self-check; seeded wrong results are caught in tests.", ["self-check"]),
    ("... Check: numeric cross-check against mpmath.", ["numeric cross-check"]),
    ("no check sentence", []),
])
def test_stated(notes, methods):
    assert H.stated(notes)[0] == methods


def test_harness_runs_stated_checks_for_named_heads():
    res = H.run(TESTS, "stated", ["VerifyDerivedKey", "FileFormat", "Gamma", "FactorInteger", "Verify"])
    assert res["VerifyDerivedKey"].checks["delegated"] == (1, 0)
    assert res["VerifyDerivedKey"].checks["reference"][0] >= 3
    assert res["Gamma"].checks["mpmath"][0] >= 4                     # mpmath table
    assert res["FactorInteger"].checks["identity"] == (1, 0)          # identity table
    assert res["Verify"].checks["self-check"] == (1, 0)              # verification step present and an example ran
    assert all(r.status == "pass" for r in res.values()), {k: r.failures for k, r in res.items()}


def test_harness_reports_failures(monkeypatch):
    r = H.Result("X", ["delegated"], "delegated (no.such.module_fn)")
    H._run_delegated(r, [], None)
    assert r.status == "fail" and "not importable" in r.failures[0]
    r = H.Result("Y", ["self-check"], "self-check")
    H._run_selfcheck(r, [], None)                                    # no implementation: no verification step
    assert r.status == "fail"
    harness = H._harness(TESTS)
    r = H.Result("Verify", ["self-check"], "self-check")
    monkeypatch.setattr(H, "_ALL_CASES", {})                         # no nesting examples to fall back on
    H._run_selfcheck(r, [("Verify[GCD[4, 6]]", harness.Raises())], harness)
    assert r.status == "fail" and "no real example" in r.failures[0]       # error cases do not count
    r = H.Result("Verify", ["self-check"], "self-check")
    H._run_selfcheck(r, [("Verify[GCD[4, 6]]", "Certified[3, <||>]")], harness)
    assert r.status == "fail" and "no example ran cleanly" in r.failures[0]  # a wrong example fails
    r = H.Result("Verify", ["self-check"], "self-check")
    H._run_selfcheck(r, [("Verify[GCD[4, 6]][[1]]", "2")], harness)
    assert r.status == "pass"
    assert H.Result("Z", [], "").status == "no check stated"


def test_markdown_report():
    res = H.run(TESTS, "stated", ["Gamma"])
    md = H.markdown(res, 1.0, "stated")
    assert "| `Gamma` |" in md and "pass" in md


def test_reference_runs_in_parallel_match_serial(monkeypatch):
    mods = [mc for mc in H.load_modules(TESTS) if mc[0] in ("test_special", "test_lists_strings")]
    monkeypatch.setattr(H, "load_modules", lambda tests_dir: mods)
    heads = {"BesselJ", "Gamma", "StringTake"}
    serial = H.run_reference(TESTS, heads, jobs=1)
    parallel = H.run_reference(TESTS, heads, jobs=2)
    assert set(serial) == heads and parallel == serial


def test_selfcheck_finds_verification_steps_in_helpers_and_distribution_classes():
    import mikoshilang  # noqa: F401
    # WaveletFilterCoefficients verifies in a helper; TracyWidomDistribution in the class its table entry names
    cases = H.load_cases(TESTS)
    harness = H._harness(TESTS)
    for head in ("WaveletFilterCoefficients", "TracyWidomDistribution"):
        r = H.Result(head, ["self-check"], "self-check")
        H._run_selfcheck(r, cases.get(head, []), harness)       # also runs one real example
        assert r.status == "pass", (head, r.failures)
    assert not any("raise ArithmeticError" in s for s in H._impl_sources("GCD"))


def test_delegated_prose_is_not_a_module_path():
    r = H.Result("X", ["delegated"], "delegated (the stored function f) in tests.")
    H._run_delegated(r, [], None)
    assert r.status == "pass"
    r = H.Result("Y", ["delegated"], "delegated (no.such.module_fn)")
    H._run_delegated(r, [], None)
    assert r.status == "fail"


def test_selfcheck_uses_an_example_that_nests_the_head():
    res = H.run(TESTS, "stated", ["KirchhoffGraph"])
    assert res["KirchhoffGraph"].status == "pass", res["KirchhoffGraph"].failures
