"""Tests for pattern matching."""

import pytest
from mikoshilang.expr import Expr, Symbol
from mikoshilang.pattern import Blank, BlankSequence, BlankNullSequence, Pattern, Condition, match, parse_pattern


class TestBlank:
    def test_match_int(self):
        assert match(Blank(), 42) == {}

    def test_match_symbol(self):
        x = Symbol("x")
        assert match(Blank(), x) == {}

    def test_match_expr(self):
        e = Expr("Plus", 1, 2)
        assert match(Blank(), e) == {}

    def test_named_blank(self):
        r = match(Blank("x"), 42)
        assert r == {"x": 42}

    def test_named_blank_consistent(self):
        # Same name must bind same value
        pat = Expr("Plus", Blank("x"), Blank("x"))
        assert match(pat, Expr("Plus", 3, 3)) == {"x": 3}
        assert match(pat, Expr("Plus", 3, 4)) is None

    def test_head_constraint_integer(self):
        assert match(Blank(head="Integer"), 42) == {}
        assert match(Blank(head="Integer"), 3.14) is None
        assert match(Blank(head="Integer"), Symbol("x")) is None

    def test_head_constraint_real(self):
        assert match(Blank(head="Real"), 3.14) == {}
        assert match(Blank(head="Real"), 42) is None

    def test_head_constraint_string(self):
        assert match(Blank(head="String"), "hello") == {}
        assert match(Blank(head="String"), 42) is None

    def test_head_constraint_symbol(self):
        assert match(Blank(head="Symbol"), Symbol("x")) == {}
        assert match(Blank(head="Symbol"), 42) is None

    def test_head_constraint_expr(self):
        assert match(Blank(head="Plus"), Expr("Plus", 1, 2)) == {}
        assert match(Blank(head="Plus"), Expr("Times", 1, 2)) is None

    def test_named_with_head(self):
        r = match(Blank("x", "Integer"), 42)
        assert r == {"x": 42}
        assert match(Blank("x", "Integer"), 3.14) is None


class TestPatternMatch:
    def test_exact_int(self):
        assert match(42, 42) == {}
        assert match(42, 43) is None

    def test_exact_symbol(self):
        x = Symbol("x")
        assert match(x, x) == {}
        assert match(x, Symbol("y")) is None

    def test_expr_match(self):
        pat = Expr("Plus", Blank("a"), Blank("b"))
        r = match(pat, Expr("Plus", 1, 2))
        assert r == {"a": 1, "b": 2}

    def test_expr_head_mismatch(self):
        pat = Expr("Plus", Blank("a"), Blank("b"))
        assert match(pat, Expr("Times", 1, 2)) is None

    def test_nested_pattern(self):
        pat = Expr("Plus", Expr("Times", Blank("a"), Blank("b")), Blank("c"))
        r = match(pat, Expr("Plus", Expr("Times", 2, 3), 4))
        assert r == {"a": 2, "b": 3, "c": 4}

    def test_pattern_object(self):
        pat = Pattern("Plus", Blank("a"), Blank("b"))
        r = match(pat, Expr("Plus", 1, 2))
        assert r == {"a": 1, "b": 2}


class TestBlankSequence:
    def test_match_one(self):
        pat = Expr("f", BlankSequence("xs"))
        r = match(pat, Expr("f", 1))
        assert r is not None
        assert r["xs"] == (1,)

    def test_match_many(self):
        pat = Expr("f", BlankSequence("xs"))
        r = match(pat, Expr("f", 1, 2, 3))
        assert r is not None
        assert r["xs"] == (1, 2, 3)

    def test_no_match_empty(self):
        pat = Expr("f", BlankSequence("xs"))
        assert match(pat, Expr("f")) is None


class TestBlankNullSequence:
    def test_match_zero(self):
        pat = Expr("f", BlankNullSequence("xs"))
        r = match(pat, Expr("f"))
        assert r is not None
        assert r["xs"] == ()

    def test_match_many(self):
        pat = Expr("f", BlankNullSequence("xs"))
        r = match(pat, Expr("f", 1, 2))
        assert r is not None
        assert r["xs"] == (1, 2)

    def test_with_fixed_arg(self):
        pat = Expr("f", Blank("a"), BlankNullSequence("rest"))
        r = match(pat, Expr("f", 1, 2, 3))
        assert r == {"a": 1, "rest": (2, 3)}

    def test_with_fixed_arg_min(self):
        pat = Expr("f", Blank("a"), BlankNullSequence("rest"))
        r = match(pat, Expr("f", 1))
        assert r == {"a": 1, "rest": ()}


class TestCondition:
    def test_condition_pass(self):
        pat = Condition(Blank("x"), lambda b: isinstance(b["x"], int) and b["x"] > 0)
        assert match(pat, 5) == {"x": 5}

    def test_condition_fail(self):
        pat = Condition(Blank("x"), lambda b: isinstance(b["x"], int) and b["x"] > 0)
        assert match(pat, -1) is None

    def test_condition_fail_type(self):
        pat = Condition(Blank("x"), lambda b: isinstance(b["x"], int) and b["x"] > 0)
        assert match(pat, 3.5) is None


class TestParsePattern:
    def test_blank(self):
        p = parse_pattern("_")
        assert isinstance(p, Blank)
        assert p.name is None

    def test_named_blank(self):
        p = parse_pattern("x_")
        assert isinstance(p, Blank)
        assert p.name == "x"

    def test_typed_blank(self):
        p = parse_pattern("x_Integer")
        assert isinstance(p, Blank)
        assert p.name == "x"
        assert p.head == "Integer"

    def test_blank_sequence(self):
        p = parse_pattern("__")
        assert isinstance(p, BlankSequence)

    def test_named_blank_sequence(self):
        p = parse_pattern("xs__")
        assert isinstance(p, BlankSequence)
        assert p.name == "xs"

    def test_null_sequence(self):
        p = parse_pattern("___")
        assert isinstance(p, BlankNullSequence)

    def test_named_null_sequence(self):
        p = parse_pattern("xs___")
        assert isinstance(p, BlankNullSequence)
        assert p.name == "xs"

    def test_symbol(self):
        p = parse_pattern("x")
        assert isinstance(p, Symbol)
        assert p.name == "x"
