"""Bounded direct-Python audit; Python signatures, not evaluator overloads."""
from fractions import Fraction
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import mikoshilang as ml

COVERED_EXPORTS = {
    'List', 'Table', 'Map', 'Select', 'Sort', 'Reverse', 'Take', 'Drop',
    'Part', 'Range', 'Total', 'Mean', 'Median', 'StandardDeviation', 'Matrix',
    'Dot', 'Det', 'Inverse', 'Eigenvalues', 'Eigenvectors', 'Transpose',
}


def test_part_zero_returns_head_including_empty_and_nested_lists():
    for values in (ml.List(), ml.List(7, 8)):
        assert ml.Part(values, 0) == ml.Symbol('List')
    assert ml.Part(ml.List(ml.List(9)), 1, 0) == ml.Symbol('List')
    assert ml.Part(ml.List(7, 8), 0) == ml.Session().evaluate('Part[{7,8},0]')


def array(matrix):
    assert matrix.head == 'Matrix'
    return np.array([row.args for row in matrix.args], dtype=complex)


def test_list_constructor_preserves_values_without_evaluation():
    x = ml.Symbol('x')
    nested = ml.List(Fraction(1, 3), x)
    assert ml.List().args == ()
    assert ml.List(nested, 'text', 2j).args == (nested, 'text', 2j)
    assert nested.args == (Fraction(1, 3), x)


@pytest.mark.parametrize('name,args', [
    ('Map', (lambda x: x,)), ('Select', (lambda x: True,)),
    ('Sort', ()), ('Reverse', ()), ('Take', (1,)), ('Drop', (1,)),
    ('Part', (1,)), ('Total', ()), ('Mean', ()), ('Median', ()),
    ('StandardDeviation', ()),
])
@pytest.mark.parametrize('bad', [None, [1, 2], ml.Expr('Other', 1)])
def test_list_operations_reject_non_list_expressions(name, args, bad):
    fn = getattr(ml, name)
    with pytest.raises(TypeError):
        fn(args[0], bad) if name == 'Map' else fn(bad, *args)


def test_map_select_sort_reverse_empty_and_callable_contracts():
    calls = []
    def body(x):
        calls.append(x)
        return x * 2
    assert ml.Map(body, ml.List()).args == ()
    assert ml.Select(ml.List(), body).args == ()
    assert calls == []
    assert ml.Map(body, ml.List(3, 1)).args == (6, 2)
    assert calls == [3, 1]
    assert ml.Select(ml.List(3, -1, 2), lambda x: x > 0).args == (3, 2)
    assert ml.Sort(ml.List()).args == ml.Reverse(ml.List()).args == ()
    assert ml.Sort(ml.List(-2, 2, -1, 1), key=abs).args == (-1, 1, -2, 2)
    with pytest.raises(TypeError):
        ml.Sort(ml.List(1, 'a'))
    with pytest.raises(TypeError):
        ml.Map(None, ml.List(1))
    with pytest.raises(TypeError):
        ml.Select(ml.List(1), None)


def test_part_nested_negative_missing_and_invalid_indices():
    values = ml.List(ml.List(1, 2), ml.List(3, 4))
    assert ml.Part(values) is values
    assert ml.Part(values, -1, -2) == 3
    assert ml.Part(values, 1, 2) == 2
    for indices in [(3,), (-3,), (1, 3), (1, 1, 1)]:
        with pytest.raises(IndexError):
            ml.Part(values, *indices)
    with pytest.raises(IndexError):
        ml.Part(ml.List(), 1)
    for idx in [0.0, 0.5, '1', None]:
        with pytest.raises(TypeError):
            ml.Part(values, idx)


@pytest.mark.parametrize('n,taken,dropped', [
    (0, (), (1, 2)), (1, (1,), (2,)), (-1, (2,), (1,)),
    (9, (1, 2), ()), (-9, (1, 2), ()),
])
def test_take_drop_boundaries(n, taken, dropped):
    assert ml.Take(ml.List(1, 2), n).args == taken
    assert ml.Drop(ml.List(1, 2), n).args == dropped
    assert ml.Take(ml.List(), n).args == ml.Drop(ml.List(), n).args == ()


@pytest.mark.parametrize('fn', [ml.Take, ml.Drop])
@pytest.mark.parametrize('count', [0.5, '1', None])
def test_take_drop_reject_noninteger_counts(fn, count):
    with pytest.raises(TypeError):
        fn(ml.List(1, 2), count)


def test_table_rational_symbolic_body_and_python_spec():
    x = ml.Symbol('x')
    assert ml.Table(lambda i: i, (Fraction(1, 3), 1, Fraction(1, 3))).args == (
        Fraction(1, 3), Fraction(2, 3), Fraction(1))
    assert ml.Table(lambda i: ml.List(x, i), 2).args == (ml.List(x, 1), ml.List(x, 2))
    assert ml.Table(lambda i: i, [2, 2]).args == (2,)
    for spec in [(), (1,), (1, 2, 3, 4)]:
        with pytest.raises(ValueError, match='spec'):
            ml.Table(lambda i: i, spec)
    with pytest.raises(TypeError):
        ml.Table(None, (1, 1))


def test_table_termination_guards_in_bounded_child():
    source = '''from mikoshilang import Table
for spec in [(1, 2, 0), (1, 2, float('nan')), (1e20, 2e20, 1.0)]:
    try:
        Table(lambda i: i, spec)
    except ValueError:
        pass
    else:
        raise AssertionError(spec)
calls = []
assert Table(lambda i: calls.append(i), (1, 3, -1)).args == ()
assert calls == []
assert Table(lambda i: i, (3, 1, -1)).args == (3, 2, 1)
'''
    try:
        result = subprocess.run([sys.executable, '-c', source],
            cwd=Path(__file__).resolve().parents[1], capture_output=True,
            text=True, timeout=20, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail('Table child exceeded 20 seconds (killed and reaped)')
    assert result.returncode == 0, result.stdout + result.stderr


def test_range_python_integer_contract_and_arity():
    assert ml.Range(0).args == ()
    assert ml.Range(5, 0, -2).args == (5, 3, 1)
    assert ml.Range(0, 5, 2).args == (0, 2, 4)
    for args in [(), (1, 2, 3, 4), (1, 2, 0)]:
        with pytest.raises(ValueError):
            ml.Range(*args)
    for args in [(2.5,), (1, 2, Fraction(1, 2)), ('3',)]:
        with pytest.raises(TypeError):
            ml.Range(*args)


def test_statistics_empty_singleton_and_rational_contract():
    assert ml.Total(ml.List()) == 0
    values = ml.List(Fraction(1, 3), Fraction(2, 3))
    assert ml.Total(values) == 1
    assert ml.Mean(values) == ml.Median(values) == Fraction(1, 2)
    assert ml.StandardDeviation(values) == pytest.approx(2 ** -0.5 / 3, rel=1e-14, abs=0)
    assert ml.Mean(ml.List(9)) == ml.Median(ml.List(9)) == 9
    assert ml.StandardDeviation(ml.List(4, 4)) == 0
    for fn in [ml.Mean, ml.Median, ml.StandardDeviation]:
        with pytest.raises(statistics.StatisticsError):
            fn(ml.List())
    with pytest.raises(statistics.StatisticsError):
        ml.StandardDeviation(ml.List(1))
    # These are Python statistics adapters, not symbolic Session statistics.
    for fn in [ml.Mean, ml.StandardDeviation]:
        with pytest.raises(TypeError):
            fn(ml.List(ml.Symbol('x'), ml.Symbol('y')))
    x = ml.Symbol('x')
    assert ml.Total(ml.List(x)) == ml.Expr('Plus', 0, x)
    assert ml.Map(lambda v: v * 2, values).args == (Fraction(2, 3), Fraction(4, 3))
    assert ml.Median(ml.List(Fraction(2, 3))) == Fraction(2, 3)


@pytest.mark.parametrize('name', ['Dot', 'Det', 'Inverse', 'Eigenvalues', 'Eigenvectors', 'Transpose'])
def test_symbolic_matrix_entries_are_not_a_numeric_backend(name):
    x = ml.Symbol('x')
    m = ml.Matrix([x])
    assert m.args == (ml.List(x),)
    with pytest.raises((TypeError, ValueError)):
        getattr(ml, name)(m, ml.Matrix([1])) if name == 'Dot' else getattr(ml, name)(m)


def test_matrix_constructor_row_forms_and_empty():
    row = ml.List(3, Fraction(1, 2))
    m = ml.Matrix([1, 2], row, (5, 6))
    assert m.head == 'Matrix'
    assert m.args == (ml.List(1, 2), row, ml.List(5, 6))
    assert ml.Matrix().args == ()
    # Constructor is structural; numerical consumers reject ragged shapes.
    assert ml.Matrix([1, 2], [3]).args == (ml.List(1, 2), ml.List(3))


@pytest.mark.parametrize('name', ['Dot', 'Det', 'Inverse', 'Eigenvalues', 'Eigenvectors', 'Transpose'])
@pytest.mark.parametrize('bad', [None, [[1]], ml.List(1)])
def test_matrix_operations_reject_non_matrix(name, bad):
    fn = getattr(ml, name)
    with pytest.raises(TypeError):
        fn(bad, ml.Matrix([1])) if name == 'Dot' else fn(bad)


@pytest.mark.parametrize('name', ['Dot', 'Det', 'Inverse', 'Eigenvalues', 'Eigenvectors', 'Transpose'])
def test_matrix_operations_reject_ragged_rows(name):
    fn = getattr(ml, name)
    bad = ml.Matrix([1, 2], [3])
    with pytest.raises(ValueError):
        fn(bad, ml.Matrix([1])) if name == 'Dot' else fn(bad)


@pytest.mark.parametrize('name', ['Det', 'Inverse', 'Eigenvalues', 'Eigenvectors'])
@pytest.mark.parametrize('m', [ml.Matrix(), ml.Matrix([1, 2, 3], [4, 5, 6])])
def test_square_operations_reject_empty_and_rectangular(name, m):
    with pytest.raises(np.linalg.LinAlgError):
        getattr(ml, name)(m)


def test_dot_rectangular_shape_errors_and_rational_numeric_inputs():
    a = ml.Matrix([1, 2, 3], [4, 5, 6])
    b = ml.Matrix([1], [2], [3])
    np.testing.assert_array_equal(array(ml.Dot(a, b)), [[14], [32]])
    with pytest.raises(ValueError):
        ml.Dot(a, a)
    assert ml.Dot(ml.Matrix([Fraction(1, 3)]), ml.Matrix([3])).args == (ml.List(1),)


def test_inverse_singular_and_complex_residuals():
    with pytest.raises(np.linalg.LinAlgError):
        ml.Inverse(ml.Matrix([1, 2], [2, 4]))
    m = ml.Matrix([1j, 2], [0, 3 - 1j])
    inv = array(ml.Inverse(m))
    np.testing.assert_allclose(array(m) @ inv, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(inv @ array(m), np.eye(2), atol=1e-14)


def test_det_exact_huge_integer_singular_and_tiny_complex():
    assert ml.Det(ml.Matrix([10**400, 1], [0, 3])) == 3 * 10**400
    assert ml.Det(ml.Matrix([1, 2], [2, 4])) == 0
    assert ml.Det(ml.Matrix([1e-100j, 0], [0, 2])) == pytest.approx(2e-100j, rel=1e-12, abs=0)


def test_eigenvalues_and_eigenvectors_complex_residuals_without_order_assumption():
    m = ml.Matrix([0, -1], [1, 0])
    values = ml.Eigenvalues(m).args
    assert values == pytest.approx((-1j, 1j), abs=0)
    vectors = ml.Eigenvectors(m)
    assert vectors.head == 'List' and len(vectors.args) == 2
    for row in vectors.args:
        v = np.array(row.args, dtype=complex)
        assert np.linalg.norm(v) == pytest.approx(1)
        assert min(np.linalg.norm(array(m) @ v - val * v) for val in values) < 1e-12
    assert np.linalg.matrix_rank([row.args for row in vectors.args]) == 2
    assert ml.Eigenvalues(ml.Matrix([2, 0], [0, 2])).args == (2, 2)


def test_transpose_is_not_conjugate_transpose_and_preserves_tiny_values():
    m = ml.Matrix([1j, 1e-100], [2, 3j])
    np.testing.assert_allclose(array(ml.Transpose(m)), [[1j, 2], [1e-100, 3j]], rtol=1e-14, atol=0)
    np.testing.assert_array_equal(array(ml.Transpose(ml.Transpose(m))), array(m))


@pytest.mark.parametrize('source,direct', [
    ('{1,2,3}', lambda: ml.List(1, 2, 3)),
    ('Reverse[{1,2,3}]', lambda: ml.Reverse(ml.List(1, 2, 3))),
    ('Sort[{3,1,2}]', lambda: ml.Sort(ml.List(3, 1, 2))),
    ('Take[{1,2,3},-1]', lambda: ml.Take(ml.List(1, 2, 3), -1)),
    ('Drop[{1,2,3},1]', lambda: ml.Drop(ml.List(1, 2, 3), 1)),
    ('Range[3,1,-1]', lambda: ml.Range(3, 1, -1)),
    ('Table[i^2,{i,1,3}]', lambda: ml.Table(lambda i: i*i, (1, 3))),
    ('Part[{{1,2},{3,4}},-1,1]', lambda: ml.Part(ml.List(ml.List(1, 2), ml.List(3, 4)), -1, 1)),
    ('Total[{1,2,3}]', lambda: ml.Total(ml.List(1, 2, 3))),
    ('Mean[{1,2,3}]', lambda: ml.Mean(ml.List(1, 2, 3))),
    ('Median[{1,2,3}]', lambda: ml.Median(ml.List(1, 2, 3))),
    ('StandardDeviation[{1,2,3}]', lambda: ml.StandardDeviation(ml.List(1, 2, 3))),
])
def test_session_parity_for_shared_numeric_list_contract(source, direct):
    assert direct() == ml.Session().evaluate(source)


def test_session_parity_for_shared_numeric_matrix_values():
    session = ml.Session()
    m = ml.Matrix([1, 2], [3, 5])
    assert ml.Det(m) == session.evaluate('Det[{{1,2},{3,5}}]') == -1
    for source, direct in [
        ('Dot[{{1,2},{3,5}},{{1,0},{0,1}}]', ml.Dot(m, ml.Matrix([1, 0], [0, 1]))),
        ('Transpose[{{1,2},{3,5}}]', ml.Transpose(m)),
        ('Inverse[{{1,0},{0,1}}]', ml.Inverse(ml.Matrix([1, 0], [0, 1]))),
    ]:
        evaluated = session.evaluate(source)
        # Session uses nested List, direct API deliberately uses Matrix head.
        expected = np.array([row.args for row in evaluated.args], dtype=complex)
        np.testing.assert_allclose(array(direct), expected, rtol=1e-14, atol=0)


def test_list_and_statistics_bounded_properties():
    hypothesis = pytest.importorskip('hypothesis')
    from hypothesis import strategies as st
    @hypothesis.settings(max_examples=35, deadline=None, derandomize=True)
    @hypothesis.given(st.lists(st.integers(-100, 100), max_size=15), st.integers(-20, 20))
    def check(xs, n):
        values = ml.List(*xs)
        assert values.args == tuple(xs)
        assert ml.Map(lambda x: x + 1, values).args == tuple(x + 1 for x in xs)
        assert ml.Select(values, lambda x: x % 2 == 0).args == tuple(x for x in xs if x % 2 == 0)
        assert ml.Sort(values).args == tuple(sorted(xs))
        assert ml.Reverse(ml.Reverse(values)) == values
        assert ml.Take(values, n).args == tuple(xs[:n] if n >= 0 else xs[n:])
        assert ml.Drop(values, n).args == tuple(xs[n:] if n >= 0 else xs[:n])
        assert ml.Total(values) == sum(xs)
        if xs:
            assert ml.Part(values, 1) == xs[0]
            assert ml.Part(values, -1) == xs[-1]
            assert ml.Mean(values) == pytest.approx(sum(xs) / len(xs))
            ordered = sorted(xs)
            k = len(xs) // 2
            expected = ordered[k] if len(xs) % 2 else (ordered[k-1] + ordered[k]) / 2
            assert ml.Median(values) == expected
        if len(xs) > 1:
            mean = sum(xs) / len(xs)
            variance = sum((x - mean)**2 for x in xs) / (len(xs) - 1)
            assert ml.StandardDeviation(values) == pytest.approx(variance**0.5)
    check()


def test_range_table_bounded_properties():
    hypothesis = pytest.importorskip('hypothesis')
    from hypothesis import strategies as st
    @hypothesis.settings(max_examples=30, deadline=None, derandomize=True)
    @hypothesis.given(st.integers(-10, 10), st.integers(-10, 10), st.sampled_from([-3, -2, -1, 1, 2, 3]))
    def check(a, b, step):
        expected = tuple(range(a, b + (1 if step > 0 else -1), step))
        assert ml.Range(a, b, step).args == expected
        assert ml.Table(lambda i: i, (a, b, step)).args == expected
    check()


def test_matrix_bounded_properties():
    hypothesis = pytest.importorskip('hypothesis')
    from hypothesis import strategies as st
    @hypothesis.settings(max_examples=30, deadline=None, derandomize=True)
    @hypothesis.given(st.lists(st.integers(-5, 5), min_size=4, max_size=4))
    def check(xs):
        a, b, c, d = xs
        m = ml.Matrix([a, b], [c, d])
        np.testing.assert_array_equal(array(m), [[a, b], [c, d]])
        np.testing.assert_array_equal(array(ml.Transpose(m)), [[a, c], [b, d]])
        np.testing.assert_array_equal(array(ml.Dot(m, ml.Matrix([1, 0], [0, 1]))), array(m))
        determinant = a*d - b*c
        assert ml.Det(m) == determinant
        if determinant:
            np.testing.assert_allclose(array(m) @ array(ml.Inverse(m)), np.eye(2), atol=1e-12)
        vals = ml.Eigenvalues(m).args
        assert sum(vals) == pytest.approx(a + d, abs=1e-12)
        assert vals[0] * vals[1] == pytest.approx(determinant, abs=1e-12)
        for row in ml.Eigenvectors(m).args:
            v = np.array(row.args, dtype=complex)
            assert np.linalg.norm(v) == pytest.approx(1)
            assert min(np.linalg.norm(array(m) @ v - val*v) for val in vals) < 1e-10
    check()
