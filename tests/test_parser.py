"""Tests for the MikoshiLang parser — 50+ tests covering all syntax."""

import pytest
from mikoshilang.parser import parse, parse_file, tokenize, ParseError
from mikoshilang.expr import Expr, Symbol


class TestTokenizer:
    def test_integer(self):
        toks = tokenize("42")
        assert toks[0].type == "INTEGER" and toks[0].value == 42

    def test_float(self):
        toks = tokenize("3.14")
        assert toks[0].type == "FLOAT" and toks[0].value == 3.14

    def test_string(self):
        toks = tokenize('"hello"')
        assert toks[0].type == "STRING" and toks[0].value == "hello"

    def test_comment_stripped(self):
        toks = tokenize("(* comment *) 42")
        assert len(toks) == 1 and toks[0].value == 42

    def test_operators(self):
        toks = tokenize("+ - * / ^")
        types = [t.type for t in toks]
        assert types == ["PLUS", "MINUS", "TIMES", "DIVIDE", "POWER"]

    def test_brackets(self):
        toks = tokenize("[ ] { } ( )")
        types = [t.type for t in toks]
        assert types == ["LSQUARE", "RSQUARE", "LBRACE", "RBRACE", "LPAREN", "RPAREN"]


class TestArithmetic:
    def test_integer_literal(self):
        assert parse("42") == 42

    def test_float_literal(self):
        assert parse("3.14") == 3.14

    def test_negative_integer(self):
        assert parse("-5") == -5

    def test_addition(self):
        assert parse("2 + 3") == Expr("Plus", 2, 3)

    def test_subtraction(self):
        assert parse("5 - 3") == Expr("Plus", 5, Expr("Times", -1, 3))

    def test_multiplication(self):
        assert parse("2 * 3") == Expr("Times", 2, 3)

    def test_division(self):
        assert parse("6 / 2") == Expr("Times", 6, Expr("Power", 2, -1))

    def test_power(self):
        assert parse("x^2") == Expr("Power", Symbol("x"), 2)

    def test_precedence_mul_add(self):
        result = parse("2 + 3 * x")
        assert result == Expr("Plus", 2, Expr("Times", 3, Symbol("x")))

    def test_precedence_power_mul(self):
        result = parse("2 * x^3")
        assert result == Expr("Times", 2, Expr("Power", Symbol("x"), 3))

    def test_parentheses(self):
        result = parse("(2 + 3) * x")
        assert result == Expr("Times", Expr("Plus", 2, 3), Symbol("x"))

    def test_nested_parens(self):
        result = parse("((x))")
        assert result == Symbol("x")

    def test_complex_expression(self):
        result = parse("x^2 - 4")
        assert result == Expr("Plus", Expr("Power", Symbol("x"), 2), Expr("Times", -1, 4))


class TestImplicitMultiplication:
    def test_number_symbol(self):
        result = parse("2x")
        assert result == Expr("Times", 2, Symbol("x"))

    def test_number_function(self):
        result = parse("2Sin[x]")
        assert result == Expr("Times", 2, Expr("Sin", Symbol("x")))

    def test_paren_paren(self):
        result = parse("(x + 1)(x - 1)")
        lhs = Expr("Plus", Symbol("x"), 1)
        rhs = Expr("Plus", Symbol("x"), Expr("Times", -1, 1))
        assert result == Expr("Times", lhs, rhs)

    def test_number_paren(self):
        result = parse("3(x + 1)")
        assert result == Expr("Times", 3, Expr("Plus", Symbol("x"), 1))


class TestFunctionCalls:
    def test_sin(self):
        assert parse("Sin[x]") == Expr("Sin", Symbol("x"))

    def test_nested_function(self):
        assert parse("Sin[Cos[x]]") == Expr("Sin", Expr("Cos", Symbol("x")))

    def test_multiple_args(self):
        assert parse("Diff[x^2, x]") == Expr("Diff", Expr("Power", Symbol("x"), 2), Symbol("x"))

    def test_integrate(self):
        result = parse("Integrate[x^2, x]")
        assert result == Expr("Integrate", Expr("Power", Symbol("x"), 2), Symbol("x"))

    def test_solve(self):
        result = parse("Solve[x^2 - 4 == 0, x]")
        inner = Expr("Equal", Expr("Plus", Expr("Power", Symbol("x"), 2), Expr("Times", -1, 4)), 0)
        assert result == Expr("Solve", inner, Symbol("x"))

    def test_simplify(self):
        result = parse("Simplify[(x^2 - 1)/(x - 1)]")
        assert result.head == "Simplify"

    def test_factor(self):
        result = parse("Factor[x^2 - 4]")
        assert result.head == "Factor"

    def test_expand(self):
        result = parse("Expand[(x + 1)^3]")
        assert result.head == "Expand"

    def test_limit(self):
        result = parse("Limit[Sin[x]/x, x -> 0]")
        assert result.head == "Limit"

    def test_series(self):
        result = parse("Series[Exp[x], {x, 0, 5}]")
        assert result.head == "Series"
        assert isinstance(result.args[1], Expr) and result.args[1].head == "List"


class TestLists:
    def test_simple_list(self):
        result = parse("{1, 2, 3}")
        assert result == Expr("List", 1, 2, 3)

    def test_empty_list(self):
        result = parse("{}")
        assert result == Expr("List")

    def test_nested_list(self):
        result = parse("{{1, 2}, {3, 4}}")
        assert result == Expr("List", Expr("List", 1, 2), Expr("List", 3, 4))

    def test_list_with_exprs(self):
        result = parse("{x, x^2, x^3}")
        assert result.head == "List" and len(result.args) == 3

    def test_range(self):
        result = parse("Range[10]")
        assert result == Expr("Range", 10)

    def test_table(self):
        result = parse("Table[i^2, {i, 1, 10}]")
        assert result.head == "Table"

    def test_map(self):
        result = parse("Map[Sin, {1, 2, 3}]")
        assert result.head == "Map"

    def test_select(self):
        result = parse("Select[{1, -2, 3}, Positive]")
        assert result.head == "Select"


class TestMatrix:
    def test_matrix(self):
        result = parse("{{1, 2}, {3, 4}}")
        assert result == Expr("List", Expr("List", 1, 2), Expr("List", 3, 4))

    def test_det(self):
        result = parse("Det[{{1, 2}, {3, 4}}]")
        assert result.head == "Det"

    def test_inverse(self):
        result = parse("Inverse[{{1, 2}, {3, 4}}]")
        assert result.head == "Inverse"


class TestAssignmentAndRules:
    def test_set(self):
        result = parse("x = 5")
        assert result == Expr("Set", Symbol("x"), 5)

    def test_set_delayed(self):
        result = parse("f[x_] := x^2")
        assert result.head == "SetDelayed"

    def test_rule(self):
        result = parse("x -> 3")
        assert result == Expr("Rule", Symbol("x"), 3)

    def test_replace_all(self):
        result = parse("x /. x -> 3")
        assert result.head == "ReplaceAll"


class TestPatterns:
    def test_blank(self):
        result = parse("_")
        assert result == Expr("Blank")

    def test_named_blank(self):
        result = parse("x_")
        assert result == Expr("Pattern", Symbol("x"), Expr("Blank"))

    def test_typed_blank(self):
        result = parse("x_Integer")
        assert result == Expr("Pattern", Symbol("x"), Expr("Blank", Symbol("Integer")))

    def test_matchq(self):
        result = parse("MatchQ[Sin[x], Sin[_]]")
        assert result.head == "MatchQ"


class TestComparison:
    def test_equal(self):
        assert parse("x == y") == Expr("Equal", Symbol("x"), Symbol("y"))

    def test_unequal(self):
        assert parse("x != y") == Expr("Unequal", Symbol("x"), Symbol("y"))

    def test_greater(self):
        assert parse("x > 0") == Expr("Greater", Symbol("x"), 0)

    def test_less(self):
        assert parse("x < 0") == Expr("Less", Symbol("x"), 0)

    def test_geq(self):
        assert parse("x >= 0") == Expr("GreaterEqual", Symbol("x"), 0)

    def test_leq(self):
        assert parse("x <= 0") == Expr("LessEqual", Symbol("x"), 0)


class TestLogic:
    def test_and(self):
        assert parse("x && y") == Expr("And", Symbol("x"), Symbol("y"))

    def test_or(self):
        assert parse("x || y") == Expr("Or", Symbol("x"), Symbol("y"))

    def test_not(self):
        assert parse("!x") == Expr("Not", Symbol("x"))


class TestConstants:
    def test_pi(self):
        assert parse("Pi") == Symbol("Pi")

    def test_e(self):
        assert parse("E") == Symbol("E")

    def test_i(self):
        assert parse("I") == Symbol("I")

    def test_infinity(self):
        assert parse("Infinity") == Symbol("Infinity")

    def test_true(self):
        assert parse("True") is True

    def test_false(self):
        assert parse("False") is False


class TestEdgeCases:
    def test_empty_input(self):
        assert parse("") is None

    def test_comment_only(self):
        assert parse("(* just a comment *)") is None

    def test_string_literal(self):
        assert parse('"hello"') == "hello"

    def test_scientific_notation(self):
        result = parse("1e5")
        assert result == 1e5

    def test_multiple_operations(self):
        result = parse("1 + 2 + 3")
        assert result == Expr("Plus", Expr("Plus", 1, 2), 3)
