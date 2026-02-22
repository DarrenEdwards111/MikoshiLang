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
    """
    Extract function name and variable names from pattern string.
    
    Example:
        'Sin[x_]' -> ('Sin', ['x'])
        'Plus[x_, y_]' -> ('Plus', ['x', 'y'])
        'Func[a_, b_, c_]' -> ('Func', ['a', 'b', 'c'])
    
    Returns:
        (func_name, [var1, var2, ...])
    """
    # Extract function name and arguments
    match = re.match(r'([A-Za-z][A-Za-z0-9]*)\[(.*?)\]', pattern_str)
    if not match:
        return None, []
    
    func_name = match.group(1)
    args_str = match.group(2).strip()
    
    if not args_str:
        # No arguments
        return func_name, []
    
    # Split arguments by comma and extract variable names
    arg_parts = [a.strip() for a in args_str.split(',')]
    
    var_names = []
    for arg in arg_parts:
        if arg.endswith('__'):
            # Sequence pattern
            var_names.append(arg[:-2])
        elif arg.endswith('_'):
            # Single pattern variable
            var_names.append(arg[:-1])
        else:
            # Literal or unrecognized - use as-is
            var_names.append(arg)
    
    return func_name, var_names

def make_rule(pattern_str, user_func):
    """
    Convert (pattern_string, lambda) tuple into RuleDelayed object.
    
    The user_func takes positional arguments matching the pattern variables.
    We create a RuleDelayed that takes a binding dict and calls user_func with those values.
    
    Args:
        pattern_str: Pattern like 'Sin[x_]' or 'Plus[x_, y_]'
        user_func: Lambda function that takes matched variables as positional args
    
    Returns:
        RuleDelayed object
    """
    func_name, var_names = parse_pattern_vars(pattern_str)
    if func_name is None:
        return None
    
    # Create Expr pattern with Blank for each variable
    pattern_args = [Blank(vname) for vname in var_names]
    pattern = Expr(func_name, *pattern_args)
    
    # Create RuleDelayed that extracts bindings and calls user_func
    def rule_lambda(bindings):
        # Extract values in the same order as var_names
        args = [bindings[vname] for vname in var_names]
        return user_func(*args)
    
    return RuleDelayed(pattern, rule_lambda)

def convert_rules(rules_list):
    """
    Convert list of (pattern_string, lambda) tuples into RuleDelayed objects.
    
    Args:
        rules_list: List of (str, callable) tuples
    
    Returns:
        List of RuleDelayed objects
    """
    converted = []
    for pattern_str, func in rules_list:
        rule = make_rule(pattern_str, func)
        if rule is not None:
            converted.append(rule)
    return converted
