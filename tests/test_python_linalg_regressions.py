"""Regressions for the public Python helpers (not Session evaluation)."""

import numpy as np
import pytest

import mikoshilang as ml


def test_eigenvalues_preserve_nonreal_spectrum_of_real_matrix():
    result = ml.Eigenvalues(ml.Matrix([0, -1], [1, 0]))
    assert result.head == "List"
    assert sorted(result.args, key=lambda x: (complex(x).real, complex(x).imag)) == pytest.approx([-1j, 1j])


@pytest.mark.parametrize("value", [1e-6, -1e-6, 1e-100])
def test_determinant_preserves_small_nonzero_values(value):
    result = ml.Det(ml.Matrix([value, 0], [0, abs(value)]))
    assert result != 0
    assert result == pytest.approx(value * abs(value), rel=1e-12, abs=0)


def test_eigenvectors_preserve_complex_components_and_satisfy_residuals():
    array = np.array([[0, -1], [1, 0]])
    result = ml.Eigenvectors(ml.Matrix(*array.tolist()))
    assert result.head == "List"
    assert len(result.args) == 2
    for vector in result.args:
        assert vector.head == "List"
        v = np.array(vector.args, dtype=complex)
        assert np.linalg.norm(v) == pytest.approx(1)
        assert min(np.linalg.norm(array @ v - value * v) for value in (-1j, 1j)) < 1e-12


def test_eigenvalues_accept_complex_matrix_entries():
    result = ml.Eigenvalues(ml.Matrix([1 + 2j, 0], [0, -3j]))
    assert result.head == "List"
    assert result.args == pytest.approx((-3j, 1 + 2j))


def test_matrix_results_do_not_snap_small_values_to_zero():
    result = ml.Dot(ml.Matrix([1e-12, 0], [0, 1]), ml.Matrix([1, 0], [0, 1]))
    assert result.head == "Matrix"
    assert all(row.head == "List" for row in result.args)
    assert result.args[0].args[0] == 1e-12


@pytest.mark.parametrize("operation, expected", [
    (lambda m: ml.Dot(m, ml.Matrix([1, 0], [0, 1])), [[1j, 1e-12j], [0, 2]]),
    (ml.Transpose, [[1j, 0], [1e-12j, 2]]),
    (ml.Inverse, [[-1j, -5e-13], [0, 0.5]]),
])
def test_complex_matrix_results_preserve_components(operation, expected):
    result = operation(ml.Matrix([1j, 1e-12j], [0, 2]))
    assert result.head == "Matrix"
    assert all(row.head == "List" for row in result.args)
    np.testing.assert_allclose([row.args for row in result.args], expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("values", [
    [1e-12, 2.0],
    [1.0 + 5e-11, 2.0],
    [1e-12j, 1.0 + 1e-12j],
    [3.0, -2.0],
])
def test_diagonal_eigenvalues_preserve_small_and_near_integer_components(values):
    result = ml.Eigenvalues(ml.Matrix([values[0], 0], [0, values[1]]))
    expected = sorted(values, key=lambda x: (complex(x).real, complex(x).imag))
    np.testing.assert_allclose(result.args, expected, rtol=1e-14, atol=0)


@pytest.mark.parametrize("rows", [
    [[2, 1], [1, 2]],
    [[3, 0], [0, -2]],
    [[0, -1], [1, 0]],
    [[1 + 2j, 3 - 1j], [0, -2j]],
    [[2, 1j], [-1j, 3]],
    [[1e-12j, 0], [0, 1 + 1e-12j]],
])
def test_eigenvectors_have_small_residuals_for_real_and_complex_matrices(rows):
    array = np.array(rows, dtype=complex)
    result = ml.Eigenvectors(ml.Matrix(*rows))
    assert result.head == "List"
    assert len(result.args) == len(rows)
    vectors = []
    for vector in result.args:
        assert vector.head == "List"
        v = np.array(vector.args, dtype=complex)
        assert v.shape == (len(rows),)
        assert np.linalg.norm(v) == pytest.approx(1)
        value = np.vdot(v, array @ v) / np.vdot(v, v)
        assert np.linalg.norm(array @ v - value * v) <= 1e-12 * np.linalg.norm(array)
        vectors.append(v)
    # These examples all have distinct eigenvalues: returning the same vector
    # repeatedly must not satisfy this regression even if its residual is zero.
    assert np.linalg.matrix_rank(vectors) == len(rows)


@pytest.mark.parametrize("rows, expected", [
    ([[1.0 + 5e-11, 0], [0, 1.0]], 1.0 + 5e-11),
    ([[1e-12j, 0], [0, 2]], 2e-12j),
    ([[1 + 2j, 3], [0, -2j]], 4 - 2j),
    ([[1j, 2j], [2j, 4j]], 0),
])
def test_determinant_complex_and_near_integer_edges(rows, expected):
    result = ml.Det(ml.Matrix(*rows))
    assert type(result) in (int, float, complex)
    assert result == pytest.approx(expected, rel=1e-14, abs=0)


@pytest.mark.parametrize("rows, expected", [
    ([[1, 2], [3, 4]], -2),
    ([[1, 2, 3], [4, 5, 6], [7, 8, 0]], 27),
    ([[1, 2], [2, 4]], 0),
    ([[2**53 + 1, 0], [0, 1]], 2**53 + 1),
    ([[10**400, 0], [0, 1]], 10**400),
])
def test_integer_determinants_are_exact_not_rounded_numerical_results(rows, expected):
    result = ml.Det(ml.Matrix(*rows))
    assert type(result) is int
    assert result == expected


@pytest.mark.parametrize("matrix, error, message", [
    (ml.List(1, 2), TypeError, "Expected a Matrix"),
    ([[1, 2], [3, 4]], TypeError, "Expected a Matrix"),
    (ml.Matrix([1, 2], [3]), ValueError, "inhomogeneous"),
    (ml.Matrix([1, 2, 3], [4, 5, 6]), np.linalg.LinAlgError, "square"),
    (ml.Matrix(), np.linalg.LinAlgError, "dimensional"),
])
def test_determinant_malformed_inputs_keep_clear_errors(matrix, error, message):
    with pytest.raises(error, match=message):
        ml.Det(matrix)


def test_exactly_integral_real_results_keep_python_integer_scalars():
    assert ml.Eigenvalues(ml.Matrix([2, 0], [0, 3])).args == (2, 3)
    result = ml.Dot(ml.Matrix([1, 0], [0, 1]), ml.Matrix([2, 0], [0, 3]))
    assert all(type(value) is int for row in result.args for value in row.args)
