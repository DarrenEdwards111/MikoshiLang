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
        return np.array(rows, dtype=float)
    raise TypeError("Expected a Matrix expression")


def _from_numpy(arr: np.ndarray) -> Expr:
    """Convert numpy array back to Matrix Expr."""
    if arr.ndim == 1:
        return Expr("List", *(int(x) if x == int(x) else float(x) for x in arr))
    rows = []
    for row in arr:
        vals = []
        for x in row:
            if abs(x - round(x)) < 1e-10:
                vals.append(int(round(x)))
            else:
                vals.append(float(x))
        rows.append(Expr("List", *vals))
    return Expr("Matrix", *rows)


def Dot(a: Expr, b: Expr) -> Expr:
    """Matrix multiplication."""
    na, nb = _to_numpy(a), _to_numpy(b)
    return _from_numpy(na @ nb)


def Det(m: Expr) -> float:
    """Matrix determinant."""
    result = np.linalg.det(_to_numpy(m))
    if abs(result - round(result)) < 1e-10:
        return int(round(result))
    return float(result)


def Inverse(m: Expr) -> Expr:
    """Matrix inverse."""
    return _from_numpy(np.linalg.inv(_to_numpy(m)))


def Eigenvalues(m: Expr) -> Expr:
    """Eigenvalues of a matrix."""
    vals = np.linalg.eigvals(_to_numpy(m))
    result = []
    for v in sorted(vals.real):
        if abs(v - round(v)) < 1e-10:
            result.append(int(round(v)))
        else:
            result.append(float(v))
    return Expr("List", *result)


def Eigenvectors(m: Expr) -> Expr:
    """Eigenvectors of a matrix."""
    vals, vecs = np.linalg.eig(_to_numpy(m))
    rows = []
    for i in range(vecs.shape[1]):
        col = vecs[:, i].real
        rows.append(Expr("List", *(float(x) for x in col)))
    return Expr("List", *rows)


def Transpose(m: Expr) -> Expr:
    """Matrix transpose."""
    return _from_numpy(_to_numpy(m).T)
