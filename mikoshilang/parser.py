"""Parser — Wolfram-style syntax to MikoshiLang expressions.

Supports arithmetic, function calls with square brackets, lists with braces,
implicit multiplication, pattern matching, assignments, and rules.
"""

from __future__ import annotations
from typing import Any, List as ListType, Optional, Tuple
import re
import json
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
    ("STRING",    r'"(?:\\.|[^"\\])*"'),
    ("SAMEQ",     r'==='),
    ("UNSAMEQ",   r'=!='),
    ("SETDELAYED",r':='),
    ("SET",       r'(?<![=!<>])=(?!=)'),
    ("RULEDELAYED", r':>'),
    ("MAP", r'/@'),
    ("APPLYLEVEL1", r'@@@'),
    ("APPLY", r'@@'),
    ("AT",        r'@'),
    ("RULE",      r'->'),
    ("REPLACEREPEATED", r'//\.'),
    ("POSTFIXAPP", r'//'),
    ("REPLACEALL",r'/\.'),
    ("CONDITION", r'/;'),
    ("DIVIDEBY",  r'/='),
    ("EQUAL",     r'=='),
    ("UNEQUAL",   r'!='),
    ("GEQ",       r'>='),
    ("LEQ",       r'<='),
    ("STRINGJOIN", r'<>'),
    ("GREATER",   r'>(?!>)'),
    ("LESS",      r'<(?!<)'),
    ("AND",       r'&&'),
    ("OR",        r'\|\|'),
    ("NOT",       r'!(?!=)'),
    ("INCREMENT", r'\+\+'),
    ("ADDTO",     r'\+='),
    ("PLUS",      r'\+'),
    ("DECREMENT", r'--'),
    ("SUBTRACTFROM", r'-='),
    ("MINUS",     r'-'),
    ("TIMESBY",   r'\*='),
    ("TIMES",     r'\*'),
    ("DIVIDE",    r'/(?![.;@/=])'),
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
    ("SLOTSEQ",   r'##[0-9]*'),
    ("SLOT",      r'#[0-9]*'),
    ("FUNCTION",  r'&(?!&)'),
    ("PRIME",     r"'+"),
    ("QUESTION",  r'\?'),
    ("WS",        r'[ \t\r]+'),
    ("NEWLINE",   r'\n'),
]

_TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in _TOKEN_SPEC), re.DOTALL)

_STATEMENT_END = ("INTEGER", "FLOAT", "STRING", "SYMBOL", "RPAREN", "RSQUARE", "RBRACE",
                  "FUNCTION", "SLOT", "SLOTSEQ", "INCREMENT", "DECREMENT", "BLANK", "BLANKS", "BLANKNS")


def tokenize(text: str) -> ListType[Token]:
    """Tokenize Wolfram-style input text."""
    tokens = []
    end = 0
    level = 0
    for m in _TOKEN_RE.finditer(text):
        if m.start() != end:
            raise ParseError(f"Unsupported character at position {end}: {text[end]!r}")
        end = m.end()
        kind = m.lastgroup
        value = m.group()
        pos = m.start()
        if kind == "NEWLINE":
            # Newlines separate top-level statements, never implicit products.
            if level == 0 and tokens and tokens[-1].type in _STATEMENT_END:
                tokens.append(Token("SEMICOLON", ";", pos))
            continue
        if kind in ("WS", "COMMENT"):
            continue
        if kind in ("LPAREN", "LSQUARE", "LBRACE"):
            level += 1
        elif kind in ("RPAREN", "RSQUARE", "RBRACE"):
            level -= 1
        if kind == "INTEGER":
            value = int(value)
        elif kind == "FLOAT":
            value = float(value)
        elif kind == "STRING":
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ParseError(f"Invalid string at position {pos}") from exc
        tokens.append(Token(kind, value, pos))
    if end != len(text):
        raise ParseError(f"Unsupported character at position {end}: {text[end]!r}")
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
        """Handle =, :=, +=, -=, *=, /= (right-associative)."""
        expr = self.parse_postfix_application()
        ops = {"SET": "Set", "SETDELAYED": "SetDelayed", "ADDTO": "AddTo",
               "SUBTRACTFROM": "SubtractFrom", "TIMESBY": "TimesBy", "DIVIDEBY": "DivideBy"}
        tok = self.peek()
        if tok is not None and tok.type in ops:
            self.advance()
            rhs = self.parse_assignment()
            return Expr(ops[tok.type], expr, rhs)
        return expr

    def parse_postfix_application(self) -> Any:
        """Handle expr // f  (== f[expr]), lowest precedence above assignment."""
        expr = self.parse_function()
        while self.at("POSTFIXAPP"):
            self.advance()
            f = self.parse_function()
            expr = Expr(_as_head(f), expr)
        return expr

    def parse_function(self) -> Any:
        """Handle postfix & (pure functions) and direct application (body &)[args]."""
        expr = self.parse_replace_all()
        while self.at("FUNCTION"):
            self.advance()
            expr = Expr("Function", expr)
            while self.at("LSQUARE", "MAP", "APPLY", "APPLYLEVEL1", "AT"):
                if self.at("LSQUARE"):
                    self.advance()
                    args = self._parse_arglist("RSQUARE")
                    self.expect("RSQUARE")
                    expr = Expr(expr, *args)
                    continue
                op = self.advance().type
                rhs = self.parse_replace_all()
                if op == "MAP":
                    expr = Expr("Map", expr, rhs)
                elif op == "APPLY":
                    expr = Expr("Apply", expr, rhs)
                elif op == "APPLYLEVEL1":
                    expr = Expr("Apply", expr, rhs, Expr("List", 1))
                else:
                    expr = Expr(expr, rhs)
        return expr

    def parse_replace_all(self) -> Any:
        """Handle /. (ReplaceAll) and //. (ReplaceRepeated), left-associative."""
        expr = self.parse_rule()
        while self.at("REPLACEALL", "REPLACEREPEATED"):
            op = self.advance().type
            rhs = self.parse_rule()
            expr = Expr("ReplaceAll" if op == "REPLACEALL" else "ReplaceRepeated", expr, rhs)
        return expr

    def parse_rule(self) -> Any:
        """Handle -> and :> (right-associative). Condition binds tighter, as in Wolfram."""
        expr = self.parse_condition()
        if self.at("RULE", "RULEDELAYED"):
            op = self.advance().type
            rhs = self.parse_rule()
            expr = Expr("RuleDelayed" if op == "RULEDELAYED" else "Rule", expr, rhs)
        return expr

    def parse_condition(self) -> Any:
        """Handle /; (Condition)."""
        expr = self.parse_or()
        while self.at("CONDITION"):
            self.advance()
            cond = self.parse_or()
            expr = Expr("Condition", expr, cond)
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

    _COMP_OPS = {
        "EQUAL": "Equal", "UNEQUAL": "Unequal",
        "LESS": "Less", "GREATER": "Greater",
        "LEQ": "LessEqual", "GEQ": "GreaterEqual",
        "SAMEQ": "SameQ", "UNSAMEQ": "UnsameQ",
    }

    def parse_comparison(self) -> Any:
        expr = self.parse_stringjoin()
        comparisons = []
        while self.peek() is not None and self.peek().type in self._COMP_OPS:
            op = self._COMP_OPS[self.advance().type]
            rhs = self.parse_stringjoin()
            comparisons.append(Expr(op, expr, rhs))
            expr = rhs
        if not comparisons:
            return expr
        if len(comparisons) == 1:
            return comparisons[0]
        # a < b < c  ==  a < b && b < c
        return Expr("And", *comparisons)

    def parse_stringjoin(self) -> Any:
        expr = self.parse_addition()
        if self.at("STRINGJOIN"):
            parts = [expr]
            while self.at("STRINGJOIN"):
                self.advance()
                parts.append(self.parse_addition())
            return Expr("StringJoin", *parts)
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
            # Implicit multiplication: juxtaposition
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
        return tok.type in ("SYMBOL", "LPAREN", "LBRACE", "INTEGER", "FLOAT", "SLOT", "SLOTSEQ")

    def parse_unary(self) -> Any:
        if self.at("MINUS"):
            self.advance()
            operand = self.parse_unary()
            if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                return -operand
            return Expr("Times", -1, operand)
        if self.at("PLUS"):
            self.advance()
            return self.parse_unary()
        expr = self.parse_power()
        if self.at("MAP", "APPLY", "APPLYLEVEL1"):
            op = self.advance().type
            rhs = self.parse_unary()
            if op == "MAP":
                expr = Expr("Map", expr, rhs)
            elif op == "APPLY":
                expr = Expr("Apply", expr, rhs)
            else:
                expr = Expr("Apply", expr, rhs, Expr("List", 1))
        return expr

    def parse_power(self) -> Any:
        base = self.parse_prefix_application()
        if self.at("POWER"):
            self.advance()
            # Right-associative
            exp = self.parse_unary()
            return Expr("Power", base, exp)
        return base

    def parse_prefix_application(self) -> Any:
        """Handle f @ x  (== f[x]), right-associative."""
        expr = self.parse_postfix()
        if self.at("AT"):
            self.advance()
            arg = self.parse_prefix_application()
            return Expr(_as_head(expr), arg)
        return expr

    def parse_postfix(self) -> Any:
        """Handle calls f[...], Part x[[i]], derivatives f', x++, x--, and factorial n!"""
        expr = self.parse_atom()
        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok.type == "LSQUARE":
                nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
                if nxt is not None and nxt.type == "LSQUARE" and nxt.pos == tok.pos + 1:
                    # Part: expr[[i, j, ...]]
                    self.advance(); self.advance()
                    args = self._parse_arglist("RSQUARE")
                    if not args:
                        raise ParseError("Empty Part specification")
                    self.expect("RSQUARE")
                    close = self.expect("RSQUARE")
                    expr = Expr("Part", expr, *args)
                    continue
                head = _as_head(expr)
                self.advance()  # consume [
                args = self._parse_arglist("RSQUARE")
                self.expect("RSQUARE")
                expr = Expr(head, *args)
            elif tok.type == "PRIME":
                self.advance()
                expr = Expr(Expr("Derivative", len(tok.value)), expr)
            elif tok.type in ("INCREMENT", "DECREMENT"):
                self.advance()
                expr = Expr("Increment" if tok.type == "INCREMENT" else "Decrement", expr)
            elif tok.type == "NOT" and self._factorial_position():
                self.advance()
                expr = Expr("Factorial", expr)
            else:
                break
        return expr

    def _factorial_position(self) -> bool:
        nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
        return nxt is None or nxt.type not in (
            "SYMBOL", "INTEGER", "FLOAT", "LPAREN", "LBRACE", "SLOT", "SLOTSEQ", "STRING")

    def _blank_tail(self, kind: str):
        """After _, __ or ___, read an optional head constraint."""
        head = None
        if self.at("SYMBOL") and self.peek().pos == self.tokens[self.pos - 1].pos + len(
                {"Blank": "_", "BlankSequence": "__", "BlankNullSequence": "___"}[kind]):
            head = self.advance().value
        return Expr(kind, Symbol(head)) if head else Expr(kind)

    def _pattern_test(self, pattern):
        if self.at("QUESTION"):
            self.advance()
            test = self.parse_atom()
            return Expr("PatternTest", pattern, test)
        return pattern

    def parse_atom(self) -> Any:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of input")

        if tok.type == "SLOT":
            self.advance()
            index = int(tok.value[1:] or "1")
            if index < 1:
                raise ParseError("Slots start at #1 (#0 is not supported)")
            return Expr("Slot", index)

        if tok.type == "SLOTSEQ":
            self.advance()
            index = int(tok.value[2:] or "1")
            if index < 1:
                raise ParseError("Slot sequences start at ##1")
            return Expr("SlotSequence", index)

        if tok.type in ("INTEGER", "FLOAT", "STRING"):
            self.advance()
            return tok.value

        _blank_kinds = {"BLANK": "Blank", "BLANKS": "BlankSequence", "BLANKNS": "BlankNullSequence"}

        if tok.type == "SYMBOL":
            self.advance()
            name = tok.value
            if name == "True":
                return True
            if name == "False":
                return False
            nxt = self.peek()
            if nxt is not None and nxt.type in _blank_kinds and nxt.pos == tok.pos + len(name):
                self.advance()
                blank = self._blank_tail(_blank_kinds[nxt.type])
                return self._pattern_test(Expr("Pattern", Symbol(name), blank))
            return Symbol(name)

        if tok.type in _blank_kinds:
            self.advance()
            return self._pattern_test(self._blank_tail(_blank_kinds[tok.type]))

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


def _as_head(expr):
    """Turn a parsed operand into an expression head (symbol name or compound head)."""
    if isinstance(expr, Symbol):
        return expr.name
    if isinstance(expr, Expr):
        return expr
    raise ParseError(f"{expr!r} cannot be used as a function head")


def parse(text: str) -> Any:
    """Parse a single Wolfram-style expression string into a MikoshiLang Expr."""
    tokens = tokenize(text)
    if not tokens:
        return None
    parser = Parser(tokens)
    result = parser.parse_expression()
    if parser.peek() is not None:
        raise ParseError(f"Unexpected trailing token: {parser.peek()}")
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
