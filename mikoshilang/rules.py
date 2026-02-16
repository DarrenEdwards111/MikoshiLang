"""Rule engine — substitution rules and simplification."""

from __future__ import annotations
from typing import Any, Callable, Dict, List as ListType, Optional, Union
from .expr import Expr, Symbol
from .pattern import match, Blank, Pattern


class Rule:
    """A substitution rule: pattern -> replacement."""
    def __init__(self, pattern, replacement):
        self.pattern = pattern
        self.replacement = replacement

    def apply(self, expr) -> Optional[Any]:
        bindings = match(self.pattern, expr)
        if bindings is not None:
            return _substitute(self.replacement, bindings)
        return None

    def __repr__(self):
        return f"Rule({self.pattern!r}, {self.replacement!r})"


class RuleDelayed:
    """A lazy substitution rule: pattern :> replacement_fn."""
    def __init__(self, pattern, replacement_fn: Callable[[Dict[str, Any]], Any]):
        self.pattern = pattern
        self.replacement_fn = replacement_fn

    def apply(self, expr) -> Optional[Any]:
        bindings = match(self.pattern, expr)
        if bindings is not None:
            return self.replacement_fn(bindings)
        return None

    def __repr__(self):
        return f"RuleDelayed({self.pattern!r}, <fn>)"


def _substitute(template, bindings: Dict[str, Any]):
    """Replace Blank placeholders in template with bound values."""
    if isinstance(template, Blank):
        if template.name and template.name in bindings:
            return bindings[template.name]
        return template
    if isinstance(template, Symbol):
        if template.name in bindings:
            return bindings[template.name]
        return template
    if isinstance(template, Expr):
        new_args = tuple(_substitute(a, bindings) for a in template.args)
        return Expr(template.head, *new_args)
    if isinstance(template, Pattern):
        new_args = tuple(_substitute(a, bindings) for a in template.args)
        return Expr(template.head, *new_args)
    return template


def replace(expr, rules) -> Any:
    """Apply first matching rule to expr (top level only)."""
    if not isinstance(rules, (list, tuple)):
        rules = [rules]
    for rule in rules:
        result = rule.apply(expr)
        if result is not None:
            return result
    return expr


def replace_all(expr, rules, _depth=0) -> Any:
    """Apply rules recursively throughout the expression."""
    if _depth > 200:
        return expr
    if not isinstance(rules, (list, tuple)):
        rules = [rules]

    # First try to match at top level
    for rule in rules:
        result = rule.apply(expr)
        if result is not None:
            return replace_all(result, rules, _depth + 1)

    # Then recurse into subexpressions
    if isinstance(expr, Expr):
        new_args = tuple(replace_all(a, rules, _depth + 1) for a in expr.args)
        new_expr = Expr(expr.head, *new_args)
        # Try matching again after transforming children
        for rule in rules:
            result = rule.apply(new_expr)
            if result is not None:
                return result
        return new_expr

    return expr


# ── Built-in simplification rules ──

def _make_simplification_rules():
    """Create standard algebraic simplification rules."""
    rules = []

    # x + 0 -> x
    rules.append(RuleDelayed(
        Expr("Plus", Blank("x"), 0),
        lambda b: b["x"]
    ))
    rules.append(RuleDelayed(
        Expr("Plus", 0, Blank("x")),
        lambda b: b["x"]
    ))

    # x * 1 -> x
    rules.append(RuleDelayed(
        Expr("Times", Blank("x"), 1),
        lambda b: b["x"]
    ))
    rules.append(RuleDelayed(
        Expr("Times", 1, Blank("x")),
        lambda b: b["x"]
    ))

    # x * 0 -> 0
    rules.append(RuleDelayed(
        Expr("Times", Blank("x"), 0),
        lambda b: 0
    ))
    rules.append(RuleDelayed(
        Expr("Times", 0, Blank("x")),
        lambda b: 0
    ))

    # x ^ 1 -> x
    rules.append(RuleDelayed(
        Expr("Power", Blank("x"), 1),
        lambda b: b["x"]
    ))

    # x ^ 0 -> 1
    rules.append(RuleDelayed(
        Expr("Power", Blank("x"), 0),
        lambda b: 1
    ))

    # Numeric addition
    rules.append(RuleDelayed(
        Expr("Plus", Blank("a", "Integer"), Blank("b", "Integer")),
        lambda b: b["a"] + b["b"]
    ))

    # Numeric multiplication
    rules.append(RuleDelayed(
        Expr("Times", Blank("a", "Integer"), Blank("b", "Integer")),
        lambda b: b["a"] * b["b"]
    ))

    # Numeric power
    rules.append(RuleDelayed(
        Expr("Power", Blank("a", "Integer"), Blank("b", "Integer")),
        lambda b: b["a"] ** b["b"] if b["b"] >= 0 else Expr("Power", b["a"], b["b"])
    ))

    return rules


SIMPLIFICATION_RULES = _make_simplification_rules()
