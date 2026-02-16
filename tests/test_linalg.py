"""Tests for linear algebra operations."""

import pytest
from mikoshilang.expr import Expr
from mikoshilang.linalg import Matrix, Dot, Det, Inverse, Eigenvalues, Eigenvectors, Transpose


class TestMatrix:
    def test_create(self):
        m = Matrix([1, 2], [3, 4])
        assert m.head == "Matrix"
        assert len(m.args) == 2

    def test_repr(self):
        m = Matrix([1, 0], [0, 1])
        r = repr(m)
        assert "Matrix" in r

    def test_identity(self):
        m = Matrix([1, 0], [0, 1])
        assert m.args[0].args == (1, 0)
        assert m.args[1].args == (0, 1)


class TestDot:
    def test_identity(self):
        m = Matrix([1, 0], [0, 1])
        v = Matrix([3, 4], [5, 6])
        r = Dot(m, v)
        assert r.args[0].args == (3, 4)
        assert r.args[1].args == (5, 6)

    def test_multiplication(self):
        a = Matrix([1, 2], [3, 4])
        b = Matrix([5, 6], [7, 8])
        r = Dot(a, b)
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19,22],[43,50]]
        assert r.args[0].args == (19, 22)
        assert r.args[1].args == (43, 50)


class TestDet:
    def test_2x2(self):
        m = Matrix([1, 2], [3, 4])
        assert Det(m) == -2

    def test_identity(self):
        m = Matrix([1, 0], [0, 1])
        assert Det(m) == 1

    def test_singular(self):
        m = Matrix([1, 2], [2, 4])
        assert Det(m) == 0

    def test_3x3(self):
        m = Matrix([1, 2, 3], [4, 5, 6], [7, 8, 0])
        d = Det(m)
        assert d == 27


class TestInverse:
    def test_identity(self):
        m = Matrix([1, 0], [0, 1])
        r = Inverse(m)
        assert r.args[0].args == (1, 0)
        assert r.args[1].args == (0, 1)

    def test_2x2(self):
        m = Matrix([4, 7], [2, 6])
        r = Inverse(m)
        # Inverse of [[4,7],[2,6]] = [[0.6,-0.7],[-0.2,0.4]]
        assert abs(r.args[0].args[0] - 0.6) < 1e-10
        assert abs(r.args[0].args[1] - (-0.7)) < 1e-10


class TestEigenvalues:
    def test_diagonal(self):
        m = Matrix([2, 0], [0, 3])
        r = Eigenvalues(m)
        assert r.head == "List"
        assert set(r.args) == {2, 3}

    def test_identity(self):
        m = Matrix([1, 0], [0, 1])
        r = Eigenvalues(m)
        assert r.args == (1, 1)


class TestEigenvectors:
    def test_basic(self):
        m = Matrix([2, 0], [0, 3])
        r = Eigenvectors(m)
        assert r.head == "List"
        assert len(r.args) == 2


class TestTranspose:
    def test_2x2(self):
        m = Matrix([1, 2], [3, 4])
        r = Transpose(m)
        assert r.args[0].args == (1, 3)
        assert r.args[1].args == (2, 4)

    def test_identity(self):
        m = Matrix([1, 0], [0, 1])
        r = Transpose(m)
        assert r.args[0].args == (1, 0)

    def test_rectangular(self):
        m = Matrix([1, 2, 3], [4, 5, 6])
        r = Transpose(m)
        assert len(r.args) == 3
        assert r.args[0].args == (1, 4)
