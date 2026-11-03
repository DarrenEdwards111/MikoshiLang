"""Linear algebra operations."""

from __future__ import annotations
from typing import Any
import numpy as np
from .expr import Expr, Symbol


def Matrix(*rows) -> Expr:
    """Create a matrix from row lists."""
    row_exprs = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            row_exprs.append(Expr("List", *row))
        elif isinstance(row, Expr) and row.head == "List":
            row_exprs.append(row)
        else:
            row_exprs.append(row)
    return Expr("Matrix", *row_exprs)


def _to_numpy(m: Expr) -> np.ndarray:
    """Convert a Matrix Expr to numpy array."""
    if isinstance(m, Expr) and m.head == "Matrix":
        rows = []
        for row in m.args:
            if isinstance(row, Expr) and row.head == "List":
                rows.append([float(x) if isinstance(x, (int, float)) else x for x in row.args])
            else:
                rows.append([float(row) if isinstance(row, (int, float)) else row])
        dtype = complex if any(np.iscomplexobj(x) for row in rows for x in row) else float
        return np.array(rows, dtype=dtype)
    raise TypeError("Expected a Matrix expression")


def _numeric_scalar(x) -> int | float | complex:
    """Use Python scalars without discarding components or rounding near integers."""
    if x.imag != 0:
        return complex(x)
    value = float(x.real)
    return int(value) if value.is_integer() else value


def _from_numpy(arr: np.ndarray) -> Expr:
    """Convert numpy array back to Matrix Expr."""
    if arr.ndim == 1:
        return Expr("List", *(_numeric_scalar(x) for x in arr))
    return Expr("Matrix", *(Expr("List", *(_numeric_scalar(x) for x in row)) for row in arr))


def Dot(a: Expr, b: Expr) -> Expr:
    """Matrix multiplication."""
    na, nb = _to_numpy(a), _to_numpy(b)
    return _from_numpy(na @ nb)


def Det(m: Expr) -> int | float | complex:
    """Matrix determinant; integer input is exact, other input is numerical."""
    if not isinstance(m, Expr) or m.head != "Matrix":
        raise TypeError("Expected a Matrix expression")
    rows = [row.args if isinstance(row, Expr) and row.head == "List" else (row,)
            for row in m.args]
    if rows and all(len(row) == len(rows) for row in rows) and all(
        isinstance(x, (int, np.integer)) for row in rows for x in row
    ):
        # Preserve exact integer results without rounding numerical determinants.
        from sympy import Matrix as SympyMatrix
        return int(SympyMatrix(rows).det())
    result = np.linalg.det(_to_numpy(m))
    return result.item()


def Inverse(m: Expr) -> Expr:
    """Matrix inverse."""
    return _from_numpy(np.linalg.inv(_to_numpy(m)))


def Eigenvalues(m: Expr) -> Expr:
    """Eigenvalues of a matrix."""
    vals = np.linalg.eigvals(_to_numpy(m))
    result = [_numeric_scalar(v) for v in sorted(vals, key=lambda x: (x.real, x.imag))]
    return Expr("List", *result)


def Eigenvectors(m: Expr) -> Expr:
    """Eigenvectors of a matrix."""
    vals, vecs = np.linalg.eig(_to_numpy(m))
    rows = []
    for i in range(vecs.shape[1]):
        col = vecs[:, i]
        rows.append(Expr("List", *(complex(x) if x.imag != 0 else float(x.real) for x in col)))
    return Expr("List", *rows)


def Transpose(m: Expr) -> Expr:
    """Matrix transpose."""
    return _from_numpy(_to_numpy(m).T)
