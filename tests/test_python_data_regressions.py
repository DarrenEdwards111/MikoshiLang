"""Regressions for the direct Python API (not the string evaluator)."""

import subprocess
import sys
from pathlib import Path

import pytest

from mikoshilang import Drop, Expr, List, Range, Table, Take


def assert_list(result, expected):
    assert isinstance(result, Expr)
    assert result.head == "List"
    assert result.args == expected


def run_child(source):
    """Bound runaway helper calls; subprocess.run kills and reaps on timeout."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", source],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("direct Python helper did not terminate within 20 seconds (including startup/import)")
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("count, expected", [(-1, (4,)), (-2, (3, 4)), (-4, (1, 2, 3, 4))])
def test_take_negative_selects_tail(count, expected):
    assert_list(Take(List(1, 2, 3, 4), count), expected)


@pytest.mark.parametrize("count, expected", [(-1, (1, 2, 3)), (-2, (1, 2)), (-4, ())])
def test_drop_negative_removes_tail(count, expected):
    assert_list(Drop(List(1, 2, 3, 4), count), expected)


@pytest.mark.parametrize("args, expected", [((5, 1, -1), (5, 4, 3, 2, 1)), ((5, 1, -2), (5, 3, 1)), ((3, 3, -1), (3,))])
def test_range_descending_includes_endpoint(args, expected):
    assert_list(Range(*args), expected)


@pytest.mark.parametrize("spec, expected", [((3, 1, -1), (3, 2, 1)), ((5, 0, -2), (5, 3, 1)), ((3, 3, -1), (3,))])
def test_table_descending(spec, expected):
    run_child(
        "from mikoshilang import Table, Expr\n"
        f"result = Table(lambda i: i, {spec!r})\n"
        "assert isinstance(result, Expr) and result.head == 'List'\n"
        f"assert result.args == {expected!r}, result\n"
    )


@pytest.mark.parametrize("spec", [(1, 3, -1), (3, 1, 1)])
def test_table_wrong_direction_is_empty_without_calling_body(spec):
    run_child(
        "from mikoshilang import Table\n"
        "calls = []\n"
        f"result = Table(lambda i: calls.append(i), {spec!r})\n"
        "assert result.head == 'List' and result.args == ()\n"
        "assert calls == []\n"
    )


@pytest.mark.parametrize("spec", [(1, 3, 0), (3, 1, 0), (2, 2, 0)])
def test_table_zero_step_raises_clear_error(spec):
    run_child(
        "from mikoshilang import Table\n"
        "try:\n"
        f"    Table(lambda i: i, {spec!r})\n"
        "except ValueError as exc:\n"
        "    assert 'step' in str(exc).lower() and 'zero' in str(exc).lower()\n"
        "else:\n"
        "    raise AssertionError('zero step was accepted')\n"
    )


@pytest.mark.parametrize("step", [
    "float('nan')", "float('inf')", "-float('inf')",
    "Decimal('NaN')", "Decimal('Infinity')", "Decimal('-Infinity')",
    "sympy.nan", "sympy.oo", "-sympy.oo",
])
def test_table_nonfinite_step_is_rejected(step):
    run_child(
        "from decimal import Decimal\n"
        "import sympy\n"
        "from mikoshilang import Table\n"
        "try:\n"
        f"    Table(lambda i: i, (1, 3, {step}))\n"
        "except ValueError as exc:\n"
        "    assert 'finite' in str(exc).lower()\n"
        "else:\n"
        "    raise AssertionError('nonfinite step was accepted')\n"
    )


@pytest.mark.parametrize("step", [
    "Fraction(10**400, 1)", "Decimal('1e400')",
    "sympy.Integer(10**400)", "sympy.Rational(10**400, 3)", "sympy.Float('1e400')",
])
@pytest.mark.parametrize("direction", [1, -1])
def test_table_accepts_finite_steps_beyond_float_range(step, direction):
    run_child(
        "from decimal import Decimal\n"
        "from fractions import Fraction\n"
        "import sympy\n"
        "from mikoshilang import Table\n"
        f"result = Table(lambda i: i, (0, {direction}, {direction} * {step}))\n"
        "assert result.head == 'List' and result.args == (0,), result\n"
    )


@pytest.mark.parametrize("start, end, step", [(1e20, 1e21, 1.0), (-1e20, -1e21, -1.0)])
def test_table_step_must_advance_iterator(start, end, step):
    run_child(
        "from mikoshilang import Table\n"
        "try:\n"
        f"    Table(lambda i: i, ({start!r}, {end!r}, {step!r}))\n"
        "except ValueError as exc:\n"
        "    assert 'step' in str(exc).lower() and 'advance' in str(exc).lower()\n"
        "else:\n"
        "    raise AssertionError('nonadvancing step was accepted')\n"
    )


@pytest.mark.parametrize("count, taken, dropped", [(0, (), (1, 2, 3, 4)), (2, (1, 2), (3, 4)), (4, (1, 2, 3, 4), ())])
def test_take_drop_nonnegative_counts(count, taken, dropped):
    values = List(1, 2, 3, 4)
    assert_list(Take(values, count), taken)
    assert_list(Drop(values, count), dropped)


@pytest.mark.parametrize("helper", [Take, Drop])
def test_take_drop_empty_lists(helper):
    assert_list(helper(List(), 0), ())


@pytest.mark.parametrize("args, expected", [
    ((0,), ()), ((-2,), ()), ((3,), (1, 2, 3)),
    ((1, 3), (1, 2, 3)), ((3, 3), (3,)), ((3, 1), ()),
    ((1, 3, -1), ()), ((3, 1, 1), ()),
    ((0, 5, 2), (0, 2, 4)), ((5, 0, -2), (5, 3, 1)),
    ((1, 2, 5), (1,)), ((2, 1, -5), (2,)),
])
def test_range_boundaries(args, expected):
    assert_list(Range(*args), expected)


@pytest.mark.parametrize("spec, expected", [
    (0, ()), (-2, ()), (3, (1, 2, 3)),
    ((1, 3), (1, 2, 3)), ((3, 3), (3,)), ((3, 1), ()),
    ((0, 5, 2), (0, 2, 4)), ((1, 2, 5), (1,)), ((2, 1, -5), (2,)),
    ((0.0, 1.0, 0.25), (0.0, 0.25, 0.5, 0.75, 1.0)),
    ((1.0, 0.0, -0.25), (1.0, 0.75, 0.5, 0.25, 0.0)),
])
def test_table_boundaries(spec, expected):
    run_child(
        "from mikoshilang import Table, Expr\n"
        f"result = Table(lambda i: i, {spec!r})\n"
        "assert isinstance(result, Expr) and result.head == 'List'\n"
        f"assert result.args == {expected!r}, result\n"
    )


@pytest.mark.parametrize("step", [0, 0.5, float('nan'), float('inf'), None, 'bad', 1j])
def test_range_invalid_steps_fail_safely(step):
    with pytest.raises((TypeError, ValueError)):
        Range(1, 3, step)


@pytest.mark.parametrize("step", ["None", "'bad'", "1j"])
def test_table_nonnumeric_steps_fail_safely(step):
    run_child(
        "from mikoshilang import Table\n"
        "try:\n"
        f"    Table(lambda i: i, (1, 3, {step}))\n"
        "except (TypeError, ValueError):\n"
        "    pass\n"
        "else:\n"
        "    raise AssertionError('invalid step was accepted')\n"
    )


def test_table_preserves_python_callable_state_and_expression_results():
    calls = []
    offset = 10

    class Body:
        def __call__(self, i):
            calls.append(i)
            return List(i, offset + i)

    result = Table(Body(), (3, 1, -1))
    assert result.head == 'List'
    assert [item.args for item in result.args] == [(3, 13), (2, 12), (1, 11)]
    assert calls == [3, 2, 1]


@pytest.mark.parametrize("source, direct", [
    ('Take[{1,2,3,4},-2]', lambda: Take(List(1, 2, 3, 4), -2)),
    ('Drop[{1,2,3,4},-2]', lambda: Drop(List(1, 2, 3, 4), -2)),
    ('Range[5,1,-1]', lambda: Range(5, 1, -1)),
    ('Range[5,0,-2]', lambda: Range(5, 0, -2)),
])
def test_direct_helpers_match_session(source, direct):
    from mikoshilang import Session

    assert_list(direct(), Session().evaluate(source).args)


@pytest.mark.parametrize("spec, source", [
    ((3, 1, -1), 'Table[i,{i,3,1,-1}]'),
    ((1, 3, -1), 'Table[i,{i,1,3,-1}]'),
])
def test_direct_table_matches_session(spec, source):
    run_child(
        "from mikoshilang import Session, Table\n"
        f"direct = Table(lambda i: i, {spec!r})\n"
        f"evaluated = Session().evaluate({source!r})\n"
        "assert direct.head == evaluated.head == 'List'\n"
        "assert direct.args == evaluated.args\n"
    )
