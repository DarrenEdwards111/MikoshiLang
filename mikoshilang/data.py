"""Data operations — lists, tables, maps, and statistics."""

from __future__ import annotations
from typing import Any, Callable, Optional
import statistics as _stats
from .expr import Expr, Symbol


def List(*args) -> Expr:
    """Create an ordered list."""
    return Expr("List", *args)


def Table(expr_fn: Callable, spec) -> Expr:
    """Generate a list. spec = (min, max) or (min, max, step)."""
    if isinstance(spec, (list, tuple)):
        if len(spec) == 2:
            start, end = spec
            step = 1
        elif len(spec) == 3:
            start, end, step = spec
        else:
            raise ValueError("Table spec must be (min, max) or (min, max, step)")
    else:
        start, end, step = 1, spec, 1
    results = []
    i = start
    while i <= end:
        results.append(expr_fn(i))
        i += step
    return Expr("List", *results)


def Map(fn: Callable, lst: Expr) -> Expr:
    """Apply fn to each element of a List."""
    if isinstance(lst, Expr) and lst.head == "List":
        return Expr("List", *(fn(a) for a in lst.args))
    raise TypeError("Map expects a List expression")


def Select(lst: Expr, cond: Callable) -> Expr:
    """Filter elements of a List by condition."""
    if isinstance(lst, Expr) and lst.head == "List":
        return Expr("List", *(a for a in lst.args if cond(a)))
    raise TypeError("Select expects a List expression")


def Sort(lst: Expr, key=None) -> Expr:
    """Sort a List."""
    if isinstance(lst, Expr) and lst.head == "List":
        return Expr("List", *sorted(lst.args, key=key))
    raise TypeError("Sort expects a List expression")


def Reverse(lst: Expr) -> Expr:
    """Reverse a List."""
    if isinstance(lst, Expr) and lst.head == "List":
        return Expr("List", *reversed(lst.args))
    raise TypeError("Reverse expects a List expression")


def Take(lst: Expr, n: int) -> Expr:
    """Take first n elements."""
    if isinstance(lst, Expr) and lst.head == "List":
        return Expr("List", *lst.args[:n])
    raise TypeError("Take expects a List expression")


def Drop(lst: Expr, n: int) -> Expr:
    """Drop first n elements."""
    if isinstance(lst, Expr) and lst.head == "List":
        return Expr("List", *lst.args[n:])
    raise TypeError("Drop expects a List expression")


def Part(lst: Expr, *indices: int) -> Any:
    """Access element(s) by index (1-based like Wolfram)."""
    if isinstance(lst, Expr) and lst.head == "List":
        result = lst
        for idx in indices:
            if isinstance(result, Expr) and result.head == "List":
                result = result.args[idx - 1] if idx > 0 else result.args[idx]
            else:
                raise IndexError(f"Cannot index into {result}")
        return result
    raise TypeError("Part expects a List expression")


def Range(*args) -> Expr:
    """Range(n), Range(a, b), or Range(a, b, step)."""
    if len(args) == 1:
        return Expr("List", *builtins_range(1, args[0] + 1))
    elif len(args) == 2:
        return Expr("List", *builtins_range(args[0], args[1] + 1))
    elif len(args) == 3:
        return Expr("List", *builtins_range(args[0], args[1] + 1, args[2]))
    raise ValueError("Range takes 1-3 arguments")


def builtins_range(start, stop, step=1):
    """Python range as list of ints."""
    return list(range(start, stop, step))


def Total(lst: Expr) -> Any:
    """Sum of elements."""
    if isinstance(lst, Expr) and lst.head == "List":
        return sum(lst.args)
    raise TypeError("Total expects a List expression")


def Mean(lst: Expr) -> float:
    """Arithmetic mean."""
    if isinstance(lst, Expr) and lst.head == "List":
        return _stats.mean(lst.args)
    raise TypeError("Mean expects a List expression")


def Median(lst: Expr) -> float:
    """Median value."""
    if isinstance(lst, Expr) and lst.head == "List":
        return _stats.median(lst.args)
    raise TypeError("Median expects a List expression")


def StandardDeviation(lst: Expr) -> float:
    """Standard deviation (sample)."""
    if isinstance(lst, Expr) and lst.head == "List":
        return _stats.stdev(lst.args)
    raise TypeError("StandardDeviation expects a List expression")
