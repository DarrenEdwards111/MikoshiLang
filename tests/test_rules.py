"""Tests for the rule engine."""

import pytest
from mikoshilang.expr import Expr, Symbol
from mikoshilang.pattern import Blank
from mikoshilang.rules import Rule, RuleDelayed, replace, replace_all, SIMPLIFICATION_RULES


x = Symbol("x")
y = Symbol("y")


class TestRule:
    def test_simple_rule(self):
        rule = Rule(Blank("a"), 0)
        assert rule.apply(42) == 0
        assert rule.apply(x) == 0

    def test_structured_rule(self):
        # Replace Plus(x, 0) with x using a rule with template
        rule = Rule(
            Expr("Plus", Blank("a"), 0),
            Blank("a")
        )
        result = rule.apply(Expr("Plus", x, 0))
        assert result == x

    def test_no_match(self):
        rule = Rule(Expr("Plus", Blank("a"), 0), Blank("a"))
        assert rule.apply(Expr("Times", x, 0)) is None


class TestRuleDelayed:
    def test_delayed_rule(self):
        rule = RuleDelayed(
            Expr("Plus", Blank("a"), Blank("b")),
            lambda b: b["a"] if b["b"] == 0 else Expr("Plus", b["a"], b["b"])
        )
        assert rule.apply(Expr("Plus", x, 0)) == x
        result = rule.apply(Expr("Plus", x, 1))
        assert result == Expr("Plus", x, 1)

    def test_delayed_computation(self):
        rule = RuleDelayed(
            Expr("Square", Blank("n", "Integer")),
            lambda b: b["n"] ** 2
        )
        assert rule.apply(Expr("Square", 5)) == 25
        assert rule.apply(Expr("Square", x)) is None


class TestReplace:
    def test_replace_first_match(self):
        r1 = Rule(42, "matched")
        assert replace(42, r1) == "matched"

    def test_replace_no_match(self):
        r1 = Rule(42, "matched")
        assert replace(43, r1) == 43

    def test_replace_list_of_rules(self):
        rules = [
            Rule(1, "one"),
            Rule(2, "two"),
        ]
        assert replace(1, rules) == "one"
        assert replace(2, rules) == "two"
        assert replace(3, rules) == 3


class TestReplaceAll:
    def test_recursive_replace(self):
        rule = Rule(0, 99)
        e = Expr("Plus", 0, Expr("Times", 0, 1))
        result = replace_all(e, rule)
        assert result == Expr("Plus", 99, Expr("Times", 99, 1))

    def test_depth_limit(self):
        # Should not infinite loop
        rule = RuleDelayed(Blank("x"), lambda b: Expr("f", b["x"]))
        # This would normally infinite loop; depth limit prevents it
        result = replace_all(42, rule)
        # Should eventually stop


class TestSimplificationRules:
    def test_plus_zero_right(self):
        result = replace_all(Expr("Plus", x, 0), SIMPLIFICATION_RULES)
        assert result == x

    def test_plus_zero_left(self):
        result = replace_all(Expr("Plus", 0, x), SIMPLIFICATION_RULES)
        assert result == x

    def test_times_one_right(self):
        result = replace_all(Expr("Times", x, 1), SIMPLIFICATION_RULES)
        assert result == x

    def test_times_one_left(self):
        result = replace_all(Expr("Times", 1, x), SIMPLIFICATION_RULES)
        assert result == x

    def test_times_zero_right(self):
        result = replace_all(Expr("Times", x, 0), SIMPLIFICATION_RULES)
        assert result == 0

    def test_times_zero_left(self):
        result = replace_all(Expr("Times", 0, x), SIMPLIFICATION_RULES)
        assert result == 0

    def test_power_one(self):
        result = replace_all(Expr("Power", x, 1), SIMPLIFICATION_RULES)
        assert result == x

    def test_power_zero(self):
        result = replace_all(Expr("Power", x, 0), SIMPLIFICATION_RULES)
        assert result == 1

    def test_numeric_add(self):
        result = replace_all(Expr("Plus", 3, 4), SIMPLIFICATION_RULES)
        assert result == 7

    def test_numeric_mul(self):
        result = replace_all(Expr("Times", 3, 4), SIMPLIFICATION_RULES)
        assert result == 12

    def test_numeric_power(self):
        result = replace_all(Expr("Power", 2, 3), SIMPLIFICATION_RULES)
        assert result == 8

    def test_nested_simplification(self):
        # (x + 0) * 1 -> x
        e = Expr("Times", Expr("Plus", x, 0), 1)
        result = replace_all(e, SIMPLIFICATION_RULES)
        assert result == x
