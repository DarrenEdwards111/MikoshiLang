"""
Helper to convert (pattern_string, lambda) tuples into RuleDelayed objects
for extended6-33 modules
"""
from .expr import Expr, Symbol
from .rules import RuleDelayed
from .pattern import Blank
import re

# Helper functions for extracting Python types from MikoshiLang Expr objects

def extract_list(expr):
    """Extract a Python list from Expr("List", ...) or return as-is."""
    if isinstance(expr, Expr) and expr.head == "List":
        return list(expr.args)
    elif isinstance(expr, (list, tuple)):
        return list(expr)
    return [expr]

def extract_number(expr):
    """Extract a Python number from an Expr."""
    if isinstance(expr, (int, float)):
        return expr
    if isinstance(expr, Symbol):
        # Try to evaluate constants
        if expr.name == 'Pi':
            import math
            return math.pi
        elif expr.name == 'E':
            import math
            return math.e
    return float(expr) if hasattr(expr, '__float__') else expr

def extract_matrix(expr):
    """Extract a nested list (matrix) from Expr."""
    if isinstance(expr, Expr) and expr.head == "List":
        return [extract_list(row) for row in expr.args]
    return expr

def parse_pattern_vars(pattern_str):
    """Validate syntax and collect distinct bindings in traversal order."""
    from .parser import parse
    from .pattern import compile_pattern, BlankSequence, BlankNullSequence
    declaration = re.fullmatch(r"([A-Za-z][A-Za-z0-9]*)\[.*\]", pattern_str, re.DOTALL)
    if declaration is None:
        raise ValueError("Expected a single function-pattern declaration")
    pattern = compile_pattern(parse(pattern_str))
    if not isinstance(pattern, Expr) or pattern.head != declaration.group(1):
        raise ValueError("Invalid function-pattern declaration")
    if not isinstance(pattern, Expr) or not isinstance(pattern.head, str):
        raise ValueError("Rule must have a function head")
    names = []
    def visit(node):
        if isinstance(node, (Blank, BlankSequence, BlankNullSequence)):
            if node.name is None:
                raise ValueError("Generated rules require named arguments")
            if node.name not in names: names.append(node.name)
        elif isinstance(node, Expr):
            for arg in node.args: visit(arg)
    visit(pattern)
    return pattern.head, names


def make_rule(pattern_str, user_func):
    from .parser import parse
    from .pattern import compile_pattern
    from .math_ops import to_sympy, from_sympy
    import sympy as sp
    _, names = parse_pattern_vars(pattern_str)
    pattern = compile_pattern(parse(pattern_str))
    def convert_in(value):
        if isinstance(value, tuple): return [convert_in(x) for x in value]
        if isinstance(value, (Expr, Symbol)): return to_sympy(value)
        return value
    def convert_out(value):
        if isinstance(value, sp.Basic): return from_sympy(value)
        if isinstance(value, (list, tuple)): return Expr('List', *(convert_out(x) for x in value))
        return value
    def apply(bindings):
        return convert_out(user_func(*(convert_in(bindings[name]) for name in names)))
    return RuleDelayed(pattern, apply)


def convert_rules(rules_list):
    """Convert validated rules; invalid declarations raise instead of disappearing."""
    return [make_rule(pattern, fn) for pattern, fn in rules_list]
