"""Parser — Wolfram-style syntax to MikoshiLang expressions.

Supports arithmetic, function calls with square brackets, lists with braces,
implicit multiplication, pattern matching, assignments, and rules.
"""

from __future__ import annotations
from typing import Any, List as ListType, Optional, Tuple
import re
from .expr import Expr, Symbol
from .evaluate import evaluate

# ── Token types ──

class Token:
    __slots__ = ("type", "value", "pos")
    def __init__(self, type: str, value: Any, pos: int = 0):
        self.type = type
        self.value = value
        self.pos = pos
    def __repr__(self):
        return f"Token({self.type}, {self.value!r})"


# ── Lexer ──

_KEYWORDS = {
    "True": True, "False": False,
    "Pi": Symbol("Pi"), "E": Symbol("E"), "I": Symbol("I"),
    "Infinity": Symbol("Infinity"),
}

_TOKEN_SPEC = [
    ("COMMENT",   r'\(\*.*?\*\)'),
    ("FLOAT",     r'\d+\.\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?|\d+[eE][+-]?\d+'),
    ("INTEGER",   r'\d+'),
    ("STRING",    r'"[^"]*"'),
    ("SETDELAYED",r':='),
    ("SET",       r'(?<![=!<>])=(?!=)'),
    ("RULE",      r'->'),
    ("REPLACEALL",r'/\.'),
    ("CONDITION", r'/;'),
    ("EQUAL",     r'=='),
    ("UNEQUAL",   r'!='),
    ("GEQ",       r'>='),
    ("LEQ",       r'<='),
    ("GREATER",   r'>(?!>)'),
    ("LESS",      r'<(?!<)'),
    ("AND",       r'&&'),
    ("OR",        r'\|\|'),
    ("NOT",       r'!(?!=)'),
    ("PLUS",      r'\+'),
    ("MINUS",     r'-'),
    ("TIMES",     r'\*'),
    ("DIVIDE",    r'/(?![.;])'),
    ("POWER",     r'\^'),
    ("LSQUARE",   r'\['),
    ("RSQUARE",   r'\]'),
    ("LBRACE",    r'\{'),
    ("RBRACE",    r'\}'),
    ("LPAREN",    r'\((?!\*)'),
    ("RPAREN",    r'\)'),
    ("COMMA",     r','),
    ("SEMICOLON", r';'),
    ("BLANKNS",   r'___'),
    ("BLANKS",    r'__'),
    ("BLANK",     r'_'),
    ("SYMBOL",    r'[A-Za-z$][A-Za-z0-9$]*'),
    ("WS",        r'[ \t]+'),
    ("NEWLINE",   r'\n'),
]

_TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in _TOKEN_SPEC))


def tokenize(text: str) -> ListType[Token]:
    """Tokenize Wolfram-style input text."""
    tokens = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        value = m.group()
        pos = m.start()
        if kind in ("WS", "NEWLINE", "COMMENT"):
            continue
        if kind == "INTEGER":
            value = int(value)
        elif kind == "FLOAT":
            value = float(value)
        elif kind == "STRING":
            value = value[1:-1]  # strip quotes
        tokens.append(Token(kind, value, pos))
    return tokens


# ── Parser ──

class ParseError(Exception):
    pass


class Parser:
    """Recursive descent parser for Wolfram-style syntax."""

    def __init__(self, tokens: ListType[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def expect(self, type: str) -> Token:
        tok = self.peek()
        if tok is None or tok.type != type:
            expected = type
            got = tok.type if tok else "EOF"
            raise ParseError(f"Expected {expected}, got {got}")
        return self.advance()

    def at(self, *types) -> bool:
        tok = self.peek()
        return tok is not None and tok.type in types

    # ── Entry point ──

    def parse_expression(self) -> Any:
        return self.parse_compound()

    def parse_compound(self) -> Any:
        """Handle semicolon-separated compound expressions."""
        expr = self.parse_assignment()
        if self.at("SEMICOLON"):
            exprs = [expr]
            while self.at("SEMICOLON"):
                self.advance()
                if self.peek() is not None and not self.at("RPAREN", "RSQUARE", "RBRACE"):
                    exprs.append(self.parse_assignment())
            if len(exprs) == 1:
                return exprs[0]
            return Expr("CompoundExpression", *exprs)
        return expr

    def parse_assignment(self) -> Any:
        """Handle = and := assignments."""
        expr = self.parse_replace_all()
        if self.at("SET"):
            self.advance()
            rhs = self.parse_replace_all()
            return Expr("Set", expr, rhs)
        if self.at("SETDELAYED"):
            self.advance()
            rhs = self.parse_replace_all()
            return Expr("SetDelayed", expr, rhs)
        return expr

    def parse_replace_all(self) -> Any:
        """Handle /. (ReplaceAll)."""
        expr = self.parse_condition()
        while self.at("REPLACEALL"):
            self.advance()
            rhs = self.parse_condition()
            expr = Expr("ReplaceAll", expr, rhs)
        return expr

    def parse_condition(self) -> Any:
        """Handle /; (Condition)."""
        expr = self.parse_rule()
        if self.at("CONDITION"):
            self.advance()
            cond = self.parse_rule()
            expr = Expr("Condition", expr, cond)
        return expr

    def parse_rule(self) -> Any:
        """Handle -> (Rule)."""
        expr = self.parse_or()
        if self.at("RULE"):
            self.advance()
            rhs = self.parse_or()
            expr = Expr("Rule", expr, rhs)
        return expr

    def parse_or(self) -> Any:
        expr = self.parse_and()
        while self.at("OR"):
            self.advance()
            rhs = self.parse_and()
            expr = Expr("Or", expr, rhs)
        return expr

    def parse_and(self) -> Any:
        expr = self.parse_not()
        while self.at("AND"):
            self.advance()
            rhs = self.parse_not()
            expr = Expr("And", expr, rhs)
        return expr

    def parse_not(self) -> Any:
        if self.at("NOT"):
            self.advance()
            operand = self.parse_not()
            return Expr("Not", operand)
        return self.parse_comparison()

    def parse_comparison(self) -> Any:
        expr = self.parse_addition()
        comp_ops = {
            "EQUAL": "Equal", "UNEQUAL": "Unequal",
            "LESS": "Less", "GREATER": "Greater",
            "LEQ": "LessEqual", "GEQ": "GreaterEqual",
        }
        tok = self.peek()
        if tok and tok.type in comp_ops:
            op = comp_ops[tok.type]
            self.advance()
            rhs = self.parse_addition()
            return Expr(op, expr, rhs)
        return expr

    def parse_addition(self) -> Any:
        expr = self.parse_multiplication()
        while self.at("PLUS", "MINUS"):
            tok = self.advance()
            rhs = self.parse_multiplication()
            if tok.type == "PLUS":
                expr = Expr("Plus", expr, rhs)
            else:
                expr = Expr("Plus", expr, Expr("Times", -1, rhs))
        return expr

    def parse_multiplication(self) -> Any:
        expr = self.parse_unary()
        while True:
            # Explicit * or /
            if self.at("TIMES"):
                self.advance()
                rhs = self.parse_unary()
                expr = Expr("Times", expr, rhs)
            elif self.at("DIVIDE"):
                self.advance()
                rhs = self.parse_unary()
                expr = Expr("Times", expr, Expr("Power", rhs, -1))
            # Implicit multiplication: number followed by symbol/function/paren/brace
            elif self._can_implicit_multiply(expr):
                rhs = self.parse_unary()
                expr = Expr("Times", expr, rhs)
            else:
                break
        return expr

    def _can_implicit_multiply(self, left) -> bool:
        """Check if current position allows implicit multiplication."""
        tok = self.peek()
        if tok is None:
            return False
        if tok.type in ("SYMBOL", "LPAREN", "LBRACE", "INTEGER", "FLOAT"):
            # Only if the previous expression is a "value" type
            # Avoid implicit mult after operators
            return True
        return False

    def parse_unary(self) -> Any:
        if self.at("MINUS"):
            self.advance()
            operand = self.parse_unary()
            if isinstance(operand, (int, float)):
                return -operand
            return Expr("Times", -1, operand)
        if self.at("PLUS"):
            self.advance()
            return self.parse_unary()
        return self.parse_power()

    def parse_power(self) -> Any:
        base = self.parse_postfix()
        if self.at("POWER"):
            self.advance()
            # Right-associative
            exp = self.parse_unary()
            return Expr("Power", base, exp)
        return base

    def parse_postfix(self) -> Any:
        """Handle function calls f[...] and factorial !"""
        expr = self.parse_atom()
        while True:
            if self.at("LSQUARE"):
                # Function call: head[args]
                if not isinstance(expr, Symbol):
                    break
                head = expr.name
                self.advance()  # consume [
                args = self._parse_arglist("RSQUARE")
                self.expect("RSQUARE")
                expr = Expr(head, *args)
            elif self.at("LPAREN"):
                # Implicit multiplication: expr(...)
                # But only if expr is a value, not an operator result
                # Actually this is handled by implicit mult in parse_multiplication
                break
            else:
                break
        return expr

    def parse_atom(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of input")

        if tok.type == "INTEGER":
            self.advance()
            return tok.value

        if tok.type == "FLOAT":
            self.advance()
            return tok.value

        if tok.type == "STRING":
            self.advance()
            return tok.value

        if tok.type == "SYMBOL":
            self.advance()
            name = tok.value
            # Check for keywords
            if name == "True":
                return True
            if name == "False":
                return False
            # Check for pattern syntax: x_, x_Integer, x__, x___
            if self.at("BLANKNS"):
                self.advance()
                return Expr("Pattern", Symbol(name), Expr("BlankNullSequence"))
            if self.at("BLANKS"):
                self.advance()
                return Expr("Pattern", Symbol(name), Expr("BlankSequence"))
            if self.at("BLANK"):
                self.advance()
                # Check for head constraint: x_Integer
                if self.at("SYMBOL"):
                    head = self.advance().value
                    return Expr("Pattern", Symbol(name), Expr("Blank", Symbol(head)))
                return Expr("Pattern", Symbol(name), Expr("Blank"))
            return Symbol(name)

        if tok.type == "BLANK":
            self.advance()
            if self.at("SYMBOL"):
                head = self.advance().value
                return Expr("Blank", Symbol(head))
            return Expr("Blank")

        if tok.type == "BLANKS":
            self.advance()
            return Expr("BlankSequence")

        if tok.type == "BLANKNS":
            self.advance()
            return Expr("BlankNullSequence")

        if tok.type == "LPAREN":
            self.advance()
            expr = self.parse_expression()
            self.expect("RPAREN")
            return expr

        if tok.type == "LBRACE":
            self.advance()
            args = self._parse_arglist("RBRACE")
            self.expect("RBRACE")
            return Expr("List", *args)

        raise ParseError(f"Unexpected token: {tok}")

    def _parse_arglist(self, end_type: str) -> ListType[Any]:
        """Parse comma-separated arguments until end_type token."""
        args = []
        if self.at(end_type):
            return args
        args.append(self.parse_expression())
        while self.at("COMMA"):
            self.advance()
            args.append(self.parse_expression())
        return args


def parse(text: str) -> Any:
    """Parse a single Wolfram-style expression string into a MikoshiLang Expr."""
    tokens = tokenize(text)
    if not tokens:
        return None
    parser = Parser(tokens)
    result = parser.parse_expression()
    return result


def parse_file(path: str) -> ListType[Any]:
    """Parse a .ml file, returning a list of expressions (one per line/statement)."""
    with open(path, "r") as f:
        text = f.read()
    tokens = tokenize(text)
    if not tokens:
        return []
    parser = Parser(tokens)
    exprs = []
    while parser.pos < len(parser.tokens):
        expr = parser.parse_expression()
        if expr is not None:
            exprs.append(expr)
        # Skip semicolons between top-level expressions
        while parser.at("SEMICOLON"):
            parser.advance()
    return exprs


def parse_and_eval(text: str) -> Any:
    """Parse a Wolfram-style expression and evaluate it."""
    expr = parse(text)
    if expr is None:
        return None
    return evaluate(expr)
