"""Direct public-API audit: bounded values, regressions and isolated global state."""
from fractions import Fraction
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = settings = st = None

from mikoshilang import (
    Expr, Symbol, symbols, Blank, BlankSequence, BlankNullSequence, Pattern,
    Rule, RuleDelayed, replace, replace_all, evaluate, parse, parse_file,
    parse_and_eval, ParseError, to_latex, Session, clear_definitions,
    function, unregister, from_python, to_python, PackedArray,
)

COVERED_EXPORTS = {
    "Expr", "Symbol", "symbols", "Blank", "BlankSequence", "BlankNullSequence",
    "Pattern", "Rule", "RuleDelayed", "replace", "replace_all", "evaluate",
    "parse", "parse_file", "parse_and_eval", "ParseError", "to_latex", "Session",
    "clear_definitions", "function", "unregister", "from_python", "to_python", "PackedArray",
}


def test_packed_array_rejects_unsigned_integer_overflow():
    # Direct construction must not silently reinterpret positive uint64 as negative int64.
    for values in ([2**63], [2**64 - 1], [[0, 1], [2, 2**63]]):
        with pytest.raises(ValueError, match="int64"):
            PackedArray(np.array(values, dtype=np.uint64))
    assert to_python(PackedArray(np.array([2**63 - 1], dtype=np.uint64))) == [2**63 - 1]


def test_expr_structure_hash_and_operator_conversion():
    x = Symbol("auditX")
    e = Expr("AuditF", x, 3)
    assert e.head == "AuditF" and e.args == (x, 3)
    assert e == Expr("AuditF", Symbol("auditX"), 3)
    assert len({e, Expr("AuditF", x, 3)}) == 1
    assert e != Expr("AuditF", x, 4)
    assert e + [1, 2] == Expr("Plus", e, Expr("List", 1, 2))
    assert 2 / e == Expr("Times", 2, Expr("Power", e, -1))
    assert -e == Expr("Times", -1, e)
    assert +e is e
    with pytest.raises(TypeError):
        e + object()


def test_symbol_value_identity_and_arithmetic_trees():
    x = Symbol("auditX")
    assert x == Symbol("auditX") and x != Symbol("auditY")
    assert x != "auditX" and repr(x) == "auditX"
    assert {x: 42}[Symbol("auditX")] == 42
    assert x + Fraction(1, 3) == Expr("Plus", x, from_python(Fraction(1, 3)))
    assert 3 - x == Expr("Plus", 3, Expr("Times", -1, x))
    assert 2**x == Expr("Power", 2, x)
    assert +x is x
    with pytest.raises(TypeError):
        x * object()


def test_symbols_single_multiple_and_whitespace():
    assert symbols("auditX") == Symbol("auditX")
    expected = (Symbol("auditX"), Symbol("auditY"))
    assert symbols("auditX", "auditY") == expected
    assert symbols(" auditX  auditY ") == expected


def test_blank_named_typed_and_repeated_binding():
    rule = Rule(Blank("n", "Integer"), Expr("AuditHit", Symbol("n")))
    assert replace(7, rule) == Expr("AuditHit", 7)
    assert replace(True, rule) is True
    assert replace(2.5, rule) == 2.5
    assert replace(Symbol("auditX"), Rule(Blank(head="Symbol"), 9)) == 9
    repeated = Rule(Pattern("AuditF", Blank("n"), Blank("n")), Symbol("n"))
    assert replace(Expr("AuditF", 3, 3), repeated) == 3
    assert replace(Expr("AuditF", 3, 4), repeated) == Expr("AuditF", 3, 4)


def test_blank_sequence_requires_one_and_splices_captured_arguments():
    rule = Rule(Pattern("AuditF", BlankSequence("xs")), Expr("AuditG", Symbol("xs")))
    assert replace(Expr("AuditF", 1, 2, 3), rule) == Expr("AuditG", 1, 2, 3)
    assert replace(Expr("AuditF"), rule) == Expr("AuditF")
    assert repr(BlankSequence("xs")) == "xs__"


def test_blank_null_sequence_empty_and_backtracking():
    rule = Rule(Pattern("AuditF", BlankNullSequence("xs"), 9), Expr("AuditG", Symbol("xs")))
    assert replace(Expr("AuditF", 9), rule) == Expr("AuditG")
    assert replace(Expr("AuditF", 1, 9, 2, 9), rule) == Expr("AuditG", 1, 9, 2)
    assert replace(Expr("AuditF", 1), rule) == Expr("AuditF", 1)
    assert repr(BlankNullSequence()) == "___"


def test_pattern_nested_head_and_arity_constraints():
    p = Pattern("AuditOuter", Pattern("AuditInner", Blank("v", "String")))
    rule = Rule(p, Symbol("v"))
    assert replace(Expr("AuditOuter", Expr("AuditInner", "yes")), rule) == "yes"
    for value in [Expr("AuditOuter"), Expr("AuditOther", Expr("AuditInner", "yes")),
                  Expr("AuditOuter", Expr("AuditInner", 1))]:
        assert replace(value, rule) == value


def test_rule_template_substitution_and_no_match():
    rule = Rule(Pattern("AuditF", Blank("v")),
                Expr("AuditG", Symbol("v"), Blank("v"), Symbol("unbound")))
    assert rule.apply(Expr("AuditF", 7)) == Expr("AuditG", 7, 7, Symbol("unbound"))
    assert rule.apply(Expr("AuditOther", 7)) is None


def test_rule_delayed_calls_only_after_matching_with_fresh_bindings():
    seen = []
    def replacement(bindings):
        seen.append(bindings.copy())
        return bindings["v"] ** 2
    rule = RuleDelayed(Pattern("AuditF", Blank("v", "Integer")), replacement)
    assert seen == []
    assert rule.apply(Expr("AuditF", "no")) is None
    assert seen == []
    assert rule.apply(Expr("AuditF", 3)) == 9
    assert rule.apply(Expr("AuditF", 4)) == 16
    assert seen == [{"v": 3}, {"v": 4}]


def test_replace_top_level_first_match_and_empty_rules():
    assert replace(1, [Rule(1, 2), Rule(1, 3)]) == 2
    nested = Expr("AuditF", 1)
    assert replace(nested, Rule(1, 2)) == nested
    assert replace(nested, []) is nested


def test_replace_all_recursive_and_repeated_rewrites():
    x, y, z = symbols("auditX auditY auditZ")
    assert replace_all(Expr("AuditF", x, Expr("AuditG", x)), [Rule(x, y), Rule(y, z)]) == Expr("AuditF", z, Expr("AuditG", z))
    assert replace_all(x, Rule(x, x)) == x
    assert replace_all(Expr("AuditF"), []) == Expr("AuditF")


def _run_isolated(source):
    result = subprocess.run([sys.executable, "-c", source], cwd=str(Path(__file__).resolve().parents[1]),
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


def test_replace_all_cyclic_and_growing_rules_terminate_in_subprocess():
    _run_isolated('''from mikoshilang import *
x, y = symbols("auditX auditY")
assert replace_all(x, [Rule(x, y), Rule(y, x)]) in (x, y)
r = replace_all(0, RuleDelayed(Blank("v"), lambda b: Expr("AuditF", b["v"])))
n = 0
while isinstance(r, Expr):
    assert r.head == "AuditF" and len(r.args) == 1
    n += 1
    r = r.args[0]
assert r == 0 and 1 <= n <= 202
''')


def test_evaluate_direct_values_symbolic_and_exact_arithmetic():
    assert evaluate(Expr("Plus", 2, Expr("Times", 3, 4))) == 14
    assert to_python(evaluate(Expr("Power", 3, -1))) == Fraction(1, 3)
    x = Symbol("publicAuditUnbound")
    assert evaluate(Expr("Plus", x, 0)) == x
    assert evaluate(Expr("PublicAuditUnknown", Expr("Plus", 1, 2))) == Expr("PublicAuditUnknown", 3)
    assert evaluate("1 + 2") == "1 + 2"  # Tree evaluator, not source evaluator.


def test_parse_precedence_patterns_strings_and_empty_input():
    x = Symbol("auditX")
    assert parse("1 + 2*3") == Expr("Plus", 1, Expr("Times", 2, 3))
    assert parse("-auditX^2") == Expr("Times", -1, Expr("Power", x, 2))
    assert parse('"a\\n\\\"b"') == 'a\n"b'
    assert parse("{True, False, auditX}") == Expr("List", True, False, x)
    assert parse(" (* comment *) ") is None
    assert parse("AuditF[v_] -> v") == Expr("Rule", Expr("AuditF", Expr("Pattern", Symbol("v"), Expr("Blank"))), Symbol("v"))


@pytest.mark.parametrize("source", ["AuditF[", "{1,2", "(1+2", "1 +", "1 ]", '"unterminated',
                                        '"bad\\q"', "(* unterminated", "☃"])
def test_parse_error_malformed_syntax(source):
    with pytest.raises(ParseError) as caught:
        parse(source)
    assert str(caught.value)


def test_parse_error_preserves_exception_message():
    error = ParseError("public audit diagnostic")
    assert isinstance(error, Exception)
    assert str(error) == "public audit diagnostic"


def test_parse_file_values_empty_malformed_and_missing(tmp_path):
    path = tmp_path / "audit.ml"
    path.write_text("1 + 2\nAuditF[3, 4]\n", encoding="utf-8")
    parsed = parse_file(path)
    # Preserve current compound-expression return contract.
    assert len(parsed) == 1
    assert Session().evaluate(parsed[0]) == Expr("AuditF", 3, 4)
    assert parsed[0].head == "CompoundExpression"
    path.write_text(" (* empty *) ", encoding="utf-8")
    assert parse_file(path) == []
    path.write_text("AuditF[", encoding="utf-8")
    with pytest.raises(ParseError):
        parse_file(path)
    with pytest.raises(FileNotFoundError):
        parse_file(tmp_path / "missing.ml")


def test_parse_and_eval_direct_arithmetic_and_errors():
    assert parse_and_eval("2^5 + 1") == 33
    assert to_python(parse_and_eval("1/7")) == Fraction(1, 7)
    assert parse_and_eval("") is None
    with pytest.raises(ParseError):
        parse_and_eval("1 +")


def test_to_latex_values_precedence_fractions_and_generic_head():
    assert to_latex(Symbol("Pi")) == r"\pi"
    assert to_latex(True) == r"\text{True}"
    assert to_latex(Expr("Times", Symbol("x"), Expr("Plus", Symbol("y"), 1))) == r"x \left(y + 1\right)"
    assert to_latex(Expr("Times", 2, Expr("Power", Symbol("x"), -1))) == r"\frac{2}{x}"
    assert to_latex(Expr("List")) == r"\{\}"
    assert to_latex(Expr("AuditF", 2)) == r"\text{AuditF}\left(2\right)"


def test_session_definitions_and_local_functions_are_isolated():
    a, b = Session(), Session()
    try:
        assert a.eval("publicAuditValue = 41") == 41
        assert a.eval("publicAuditValue + 1") == 42
        assert b.eval("publicAuditValue") == Symbol("publicAuditValue")
        @a.function(name="PublicAuditLocal")
        def increment(n: int) -> int:
            return n + 1
        assert a.eval("PublicAuditLocal[3]") == 4
        assert b.eval("PublicAuditLocal[3]") == Expr("PublicAuditLocal", 3)
        with pytest.raises(TypeError, match="PublicAuditLocal"):
            a.eval('PublicAuditLocal["bad"]')
        assert a.eval("PublicAuditLocal[4]") == 5
        assert b.eval("publicAuditValue") == Symbol("publicAuditValue")
        a.clear()
        assert a.functions() == {}
        assert a.eval("publicAuditValue") == Symbol("publicAuditValue")
    finally:
        a.clear()
        b.clear()


def test_session_modes_and_python_permission_validation():
    s = Session()
    try:
        assert s.eval("1/3") == Fraction(1, 3)
        assert s.eval([1, Fraction(1, 3)]) == [1, Fraction(1, 3)]
        assert s.eval("{1, 2}", out="expr") == Expr("List", 1, 2)
        with pytest.raises(ValueError, match="out"):
            s.eval("1", out="bad")
        with pytest.raises(TypeError, match="symbolic"):
            s.eval("publicAuditUnbound", symbolic="error")
        with pytest.raises(PermissionError):
            s.eval('PythonFunction["math.sqrt"][4]')
        with pytest.raises(ValueError):
            Session(allow_python={"bad-path"})
    finally:
        s.clear()


def test_clear_definitions_default_state_isolated_in_subprocess():
    _run_isolated('''from mikoshilang import *
a = Session()
try:
    clear_definitions()
    assert parse_and_eval("publicAuditDefault = 9") == 9
    assert evaluate(Symbol("publicAuditDefault")) == 9
    assert a.eval("publicAuditDefault = 8") == 8
    clear_definitions()
    assert evaluate(Symbol("publicAuditDefault")) == Symbol("publicAuditDefault")
    assert a.eval("publicAuditDefault") == 8
finally:
    clear_definitions()
    a.clear()
''')


def test_function_global_typed_listable_and_unregister_cleanup():
    name = "PublicAuditTriangle"
    registered = False
    try:
        @function(name=name, listable=True, check="exact")
        def triangle(n: "NonNegInteger") -> int:
            return n * (n + 1) // 2
        registered = True
        assert triangle(4) == 10
        assert triangle.mikoshi.name == name
        assert "Check: exact" in triangle.mikoshi.notes
        assert Session().eval("PublicAuditTriangle[{0, 1, 4, 10}]") == [0, 1, 10, 55]
        assert evaluate(Expr(name, 4)) == 10
        assert Session().eval("PublicAuditTriangle[publicAuditUnbound]") == Expr(name, Symbol("publicAuditUnbound"))
        with pytest.raises(TypeError, match=name):
            Session().eval('PublicAuditTriangle["bad"]')
        with pytest.raises(ValueError, match=name):
            Session().eval("PublicAuditTriangle[-1]")
    finally:
        if registered:
            unregister(name)
    assert Session().eval("PublicAuditTriangle[4]") == Expr(name, 4)
    with pytest.raises(ValueError, match="not registered"):
        unregister(name)


@pytest.mark.parametrize("value", [None, True, False, 0, -(2**150), 2**150, 1.25, "quoted \" text\n",
                                   Fraction(-7, 13), 2.5-3.75j, 0.0+2.5j, [], [1, [False, None]],
                                   {"a": [Fraction(1, 3), 2.0+3.0j], "b": None}])
def test_from_python_to_python_documented_roundtrips(value):
    assert to_python(from_python(value)) == value


def test_from_python_unsupported_nonfinite_and_numpy_scalars():
    with pytest.raises(TypeError, match="cannot convert"):
        from_python(object())
    with pytest.raises(TypeError, match="callables"):
        from_python(lambda: 1)
    for value in [float("nan"), complex(float("nan"), 0), complex(float("inf"), 1)]:
        with pytest.raises(ValueError):
            from_python(value)
    assert to_python(from_python(float("inf"))) == float("inf")
    assert to_python(from_python(float("-inf"))) == float("-inf")
    assert type(from_python(np.int64(3))) is int
    assert type(from_python(np.bool_(True))) is bool
    assert to_python(from_python(np.array([2**63], dtype=np.uint64))) == [2**63]


def test_to_python_symbolic_modes_and_numpy_copy():
    x = Symbol("publicAuditUnbound")
    assert to_python(x) is x
    assert str(to_python(x, symbolic="sympy")) == "publicAuditUnbound"
    with pytest.raises(TypeError, match="symbolic"):
        to_python(x, symbolic="error")
    assert to_python(Symbol("Null")) is None
    packed = PackedArray(np.array([1, 2]))
    arr = to_python(packed, arrays="numpy")
    arr[0] = 99
    assert to_python(packed) == [1, 2]
    np.testing.assert_array_equal(to_python(Expr("List", Expr("List", 1, 2)), arrays="numpy"), [[1, 2]])


@pytest.mark.parametrize("dtype,values,expected_dtype", [
    (np.int16, [1, -2], np.int64), (np.float32, [1.5, -2.25], np.float64),
    (np.complex64, [1+2j, -3+4j], np.complex128),
])
def test_packed_array_dtype_complex_values_readonly_and_equality(dtype, values, expected_dtype):
    original = np.array(values, dtype=dtype)
    packed = PackedArray(original)
    assert packed.head == "List"
    assert packed.array.dtype == expected_dtype
    assert to_python(packed) == values
    assert packed == from_python(values)
    assert from_python(values) == packed
    assert hash(packed) == hash(from_python(values))
    original[0] = 0
    assert to_python(packed) == values
    with pytest.raises(ValueError):
        packed.array[0] = 0
    assert [to_python(v) for v in packed.args] == values


def test_packed_array_multidimensional_and_invalid_inputs():
    packed = PackedArray(np.array([[1+2j, 3-4j], [5+0j, 6+7j]]))
    assert to_python(packed) == [[1+2j, 3-4j], [5+0j, 6+7j]]
    assert to_python(packed.args[0]) == [1+2j, 3-4j]
    for value in [np.array(1), np.array([]), np.array([np.inf]), np.array([np.nan])]:
        with pytest.raises(ValueError):
            PackedArray(value)
    for value in [np.array([True]), np.array(["x"]), np.array([object()])]:
        with pytest.raises(TypeError, match="dtype"):
            PackedArray(value)


@pytest.mark.skipif(given is None, reason="hypothesis optional")
def test_property_integer_fraction_nested_conversion_roundtrip():
    atom = st.one_of(st.none(), st.booleans(), st.integers(-10**30, 10**30),
                     st.text(max_size=12), st.fractions(min_value=-100, max_value=100, max_denominator=100))
    values = st.recursive(atom, lambda children: st.one_of(st.lists(children, max_size=4),
                          st.dictionaries(st.text(max_size=5), children, max_size=4)), max_leaves=12)
    @settings(max_examples=60, deadline=None, derandomize=True)
    @given(values)
    def check(value):
        assert to_python(from_python(value)) == value
    check()


@pytest.mark.skipif(given is None, reason="hypothesis optional")
def test_property_parse_evaluate_bounded_arithmetic():
    @settings(max_examples=50, deadline=None, derandomize=True)
    @given(st.integers(-1000, 1000), st.integers(-1000, 1000), st.integers(-30, 30))
    def check(a, b, c):
        assert parse_and_eval("(%d) + (%d) * (%d)" % (a, b, c)) == a + b*c
        assert evaluate(Expr("Plus", a, Expr("Times", b, c))) == a + b*c
    check()


@pytest.mark.skipif(given is None, reason="hypothesis optional")
def test_property_sequence_capture_preserves_values():
    @settings(max_examples=40, deadline=None, derandomize=True)
    @given(st.lists(st.integers(-100, 100), max_size=8))
    def check(values):
        rule = Rule(Pattern("AuditF", BlankNullSequence("xs")), Expr("AuditG", Symbol("xs")))
        assert replace(Expr("AuditF", *values), rule) == Expr("AuditG", *values)
    check()


@pytest.mark.skipif(given is None, reason="hypothesis optional")
def test_property_machine_complex_conversion_roundtrip():
    @settings(max_examples=60, deadline=None, derandomize=True)
    @given(st.floats(min_value=-1e100, max_value=1e100, allow_nan=False, allow_infinity=False),
           st.floats(min_value=-1e100, max_value=1e100, allow_nan=False, allow_infinity=False))
    def check(real, imag):
        value = complex(real, imag)
        assert to_python(from_python(value)) == value
        packed = PackedArray(np.array([value], dtype=np.complex128))
        assert to_python(packed) == [value]
        assert to_python(packed.args[0]) == value
    check()
