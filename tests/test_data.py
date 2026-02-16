"""Tests for data operations."""

import pytest
from mikoshilang.expr import Expr, Symbol
from mikoshilang.data import (
    List, Table, Map, Select, Sort, Reverse, Take, Drop, Part,
    Range, Total, Mean, Median, StandardDeviation
)


class TestList:
    def test_create(self):
        l = List(1, 2, 3)
        assert l.head == "List"
        assert l.args == (1, 2, 3)

    def test_empty(self):
        l = List()
        assert l.args == ()

    def test_nested(self):
        l = List(List(1, 2), List(3, 4))
        assert l.args[0].head == "List"

    def test_repr(self):
        assert repr(List(1, 2, 3)) == "{1, 2, 3}"


class TestTable:
    def test_basic(self):
        r = Table(lambda i: i ** 2, (1, 5))
        assert r.args == (1, 4, 9, 16, 25)

    def test_with_step(self):
        r = Table(lambda i: i, (0, 10, 2))
        assert r.args == (0, 2, 4, 6, 8, 10)

    def test_single_arg(self):
        r = Table(lambda i: i, 5)
        assert r.args == (1, 2, 3, 4, 5)

    def test_expression_fn(self):
        x = Symbol("x")
        r = Table(lambda i: i + 1, (1, 3))
        assert r.args == (2, 3, 4)


class TestMap:
    def test_basic(self):
        l = List(1, 2, 3, 4)
        r = Map(lambda x: x * 2, l)
        assert r.args == (2, 4, 6, 8)

    def test_with_strings(self):
        l = List("a", "b", "c")
        r = Map(lambda x: x.upper(), l)
        assert r.args == ("A", "B", "C")

    def test_type_error(self):
        with pytest.raises(TypeError):
            Map(lambda x: x, 42)


class TestSelect:
    def test_basic(self):
        l = List(1, 2, 3, 4, 5, 6)
        r = Select(l, lambda x: x % 2 == 0)
        assert r.args == (2, 4, 6)

    def test_none_match(self):
        l = List(1, 3, 5)
        r = Select(l, lambda x: x % 2 == 0)
        assert r.args == ()

    def test_type_error(self):
        with pytest.raises(TypeError):
            Select(42, lambda x: True)


class TestSort:
    def test_basic(self):
        l = List(3, 1, 4, 1, 5)
        r = Sort(l)
        assert r.args == (1, 1, 3, 4, 5)

    def test_with_key(self):
        l = List(-3, 1, -2)
        r = Sort(l, key=abs)
        assert r.args == (1, -2, -3)

    def test_type_error(self):
        with pytest.raises(TypeError):
            Sort(42)


class TestReverse:
    def test_basic(self):
        l = List(1, 2, 3)
        r = Reverse(l)
        assert r.args == (3, 2, 1)

    def test_type_error(self):
        with pytest.raises(TypeError):
            Reverse(42)


class TestTakeDrop:
    def test_take(self):
        l = List(1, 2, 3, 4, 5)
        assert Take(l, 3).args == (1, 2, 3)

    def test_drop(self):
        l = List(1, 2, 3, 4, 5)
        assert Drop(l, 2).args == (3, 4, 5)

    def test_take_zero(self):
        l = List(1, 2, 3)
        assert Take(l, 0).args == ()

    def test_drop_all(self):
        l = List(1, 2, 3)
        assert Drop(l, 3).args == ()


class TestPart:
    def test_first(self):
        l = List(10, 20, 30)
        assert Part(l, 1) == 10

    def test_last(self):
        l = List(10, 20, 30)
        assert Part(l, 3) == 30

    def test_negative(self):
        l = List(10, 20, 30)
        assert Part(l, -1) == 30

    def test_type_error(self):
        with pytest.raises(TypeError):
            Part(42, 1)


class TestRange:
    def test_single(self):
        r = Range(5)
        assert r.args == (1, 2, 3, 4, 5)

    def test_two_args(self):
        r = Range(3, 7)
        assert r.args == (3, 4, 5, 6, 7)

    def test_with_step(self):
        r = Range(0, 10, 2)
        assert r.args == (0, 2, 4, 6, 8, 10)

    def test_repr(self):
        assert repr(Range(3)) == "{1, 2, 3}"


class TestStatistics:
    def test_total(self):
        assert Total(List(1, 2, 3, 4)) == 10

    def test_mean(self):
        assert Mean(List(1, 2, 3, 4)) == 2.5

    def test_median_odd(self):
        assert Median(List(1, 2, 3)) == 2

    def test_median_even(self):
        assert Median(List(1, 2, 3, 4)) == 2.5

    def test_std_dev(self):
        r = StandardDeviation(List(2, 4, 4, 4, 5, 5, 7, 9))
        assert abs(r - 2.138) < 0.01

    def test_total_type_error(self):
        with pytest.raises(TypeError):
            Total(42)
