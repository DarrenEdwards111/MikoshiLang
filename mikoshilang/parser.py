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
from . import notation_ops as _NOTATION

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
    ("BIGNUM",    r'(?:\d+\.\d*|\.\d+|\d+)(?:``-?(?:\d+\.?\d*|\.\d+)|`(?:\d+\.?\d*|\.\d+)?)(?:\*\^[+-]?\d+)?'),
    ("SCINUM",    r'(?:\d+\.\d*|\.\d+|\d+)\*\^[+-]?\d+'),
    ("FLOAT",     r'\d+\.\d*([eE][+-]?\d+)?|\.\d+([eE][+-]?\d+)?|\d+[eE][+-]?\d+'),
    ("INTEGER",   r'\d+'),
    ("STRING",    r'"(?:\\.|[^"\\])*"'),
    ("SAMEQ",     r'==='),
    ("UNSAMEQ",   r'=!='),
    ("SETDELAYED",r':='),
    ("UPSETDELAYED", r'\^:='),
    ("UPSET",     r'\^=(?!=)'),
    ("UNSET",     r'=\.(?![0-9.])'),
    ("SET",       r'(?<![=!<>])=(?!=)'),
    ("RULEDELAYED", r':>'),
    ("MESSAGENAME", r'::'),
    ("COLON",     r':(?![=>])'),
    ("MAPALL", r'//@'),
    ("MAP", r'/@'),
    ("COMPOSITION", r'@\*'),
    ("APPLYLEVEL1", r'@@@'),
    ("APPLY", r'@@'),
    ("AT",        r'@'),
    ("TWOWAYRULE", r'<->'),
    ("VECTOROP",  r'\\\[Vector(?:LessEqual|GreaterEqual|Less|Greater)\]'),
    ("NAMEDOP",   r'\\\[(?:DirectedEdge|UndirectedEdge|Distributed|TwoWayRule)\]'),
    # Notation operators (W31/W32): \[Name] long names and Wolfram's characters; the value becomes the head name
    # (\[GreaterSlantEqual] and \[LessSlantEqual] become GEQ/LEQ tokens, as in Wolfram).
    ("NOTATION",  '|'.join(re.escape(k) for k in sorted(_NOTATION.input_map(), key=len, reverse=True))),
    ("LANGLE",    r'\\\[LeftAngleBracket\]|\u2329'),
    ("RANGLE",    r'\\\[RightAngleBracket\]|\u232a'),
    ("LDBAR",     r'\\\[LeftDoubleBracketingBar\]|\uf605'),
    ("RDBAR",     r'\\\[RightDoubleBracketingBar\]|\uf606'),
    ("RULE",      r'->'),
    ("REPLACEREPEATED", r'//\.'),
    ("APPLYTO",   r'//=(?!=)'),
    ("POSTFIXAPP", r'//'),
    ("REPLACEALL",r'/\.'),
    ("CONDITION", r'/;'),
    ("TAGSET",    r'/:'),
    ("DIVIDEBY",  r'/='),
    ("EQUAL",     r'=='),
    ("UNEQUAL",   r'!='),
    ("PUTAPPEND", r'>>>[ \t]*(?:"(?:\\.|[^"\\])*"|[^\s;,\])}"]+)'),
    ("PUT",       r'>>[ \t]*(?:"(?:\\.|[^"\\])*"|[^\s;,\])}"]+)'),
    ("GEQ",       r'>='),
    ("LEQ",       r'<='),
    ("STRINGJOIN", r'<>'),
    ("GET",       r'<<[ \t]*(?:"(?:\\.|[^"\\])*"|[^\s;,\])}"]+)'),
    ("OUT",       r'%(?:[0-9]+|%*)'),
    ("LASSOC",    r'<\|'),
    ("RASSOC",    r'\|>'),
    ("STRINGEXPR", r'~~'),
    ("GREATER",   r'>(?!>)'),
    ("LESS",      r'<(?!<)'),
    ("AND",       r'&&'),
    ("OR",        r'\|\|'),
    ("ALTERNATIVES", r'\|'),
    ("NOT",       r'!(?!=)'),
    ("INCREMENT", r'\+\+'),
    ("ADDTO",     r'\+='),
    ("PLUS",      r'\+'),
    ("DECREMENT", r'--'),
    ("SUBTRACTFROM", r'-='),
    ("MINUS",     r'-'),
    ("NCMULT",    r'\*\*'),
    ("TIMESBY",   r'\*='),
    ("TIMES",     r'\*'),
    ("RIGHTCOMPOSITION", r'/\*'),
    ("DIVIDE",    r'/(?![.;@/=*:])'),
    ("POWER",     r'\^'),
    ("LSQUARE",   r'\['),
    ("RSQUARE",   r'\]'),
    ("LBRACE",    r'\{'),
    ("RBRACE",    r'\}'),
    ("LPAREN",    r'\((?!\*)'),
    ("RPAREN",    r'\)'),
    ("COMMA",     r','),
    ("SPAN",      r';;'),
    ("SEMICOLON", r';'),
    ("REPEATEDNULL", r'\.\.\.'),
    ("REPEATED",  r'\.\.'),
    ("DOT",       r'\.(?![0-9])'),
    ("BLANKOPT",  r'_\.(?![.0-9])'),
    ("BLANKNS",   r'___'),
    ("BLANKS",    r'__'),
    ("BLANK",     r'_'),
    ("SYMBOL",    r'`?[A-Za-z$][A-Za-z0-9$]*(?:`[A-Za-z$][A-Za-z0-9$]*)*'),
    ("SLOTSEQ",   r'##[0-9]*'),
    ("SLOTNAME",  r'#[A-Za-z][A-Za-z0-9]*'),
    ("SLOT",      r'#[0-9]*'),
    ("FUNCTION",  r'&(?!&)'),
    ("PRIME",     r"'+"),
    ("QUESTION",  r'\?'),
    ("WS",        r'[ \t\r]+'),
    ("NEWLINE",   r'\n'),
]

_NOTATION_INPUT = _NOTATION.input_map()
_TOKEN_RE = re.compile('|'.join(f'(?P<{name}>{pattern})' for name, pattern in _TOKEN_SPEC), re.DOTALL)

_STATEMENT_END = ("INTEGER", "FLOAT", "BIGNUM", "SCINUM", "STRING", "SYMBOL", "RPAREN", "RSQUARE", "RBRACE", "RASSOC",
                  "RANGLE", "RDBAR",
                  "FUNCTION", "SLOT", "SLOTSEQ", "SLOTNAME", "INCREMENT", "DECREMENT", "BLANK", "BLANKS",
                  "BLANKNS", "BLANKOPT", "REPEATED", "REPEATEDNULL", "OUT", "GET", "UNSET", "PUT", "PUTAPPEND")


def _wl_escapes(lit: str) -> str:
    """Wolfram string escapes \\:xxxx and \\.xx -> JSON \\uxxxx (other escapes are left to json.loads)."""
    out, i, n = [], 0, len(lit)
    while i < n:
        c = lit[i]
        if c == "\\" and i + 1 < n:
            d = lit[i + 1]
            if d == ":" and re.fullmatch(r"[0-9A-Fa-f]{4}", lit[i + 2:i + 6] or ""):
                out.append("\\u" + lit[i + 2:i + 6])
                i += 6
                continue
            if d == "." and re.fullmatch(r"[0-9A-Fa-f]{2}", lit[i + 2:i + 4] or ""):
                out.append("\\u00" + lit[i + 2:i + 4])
                i += 4
                continue
            out.append(lit[i:i + 2])
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _dps_bits(dps: float) -> int:
    """Binary precision for ``dps`` decimal digits (mpmath's dps_to_prec, so N[x, n] and x`n agree)."""
    return max(1, int(round((dps + 1) * 3.321928094887362)))


def _number_literal(text: str):
    """Wolfram number marks: ``1.5`30`` (precision 30), ``1.5``20`` (accuracy 20), ``1.5` `` (machine) and
    ``1.5*^-3`` (scientific notation).  Precision at or below machine precision gives a machine number
    (MikoshiLang keeps no low-precision bignums)."""
    import sympy as sp
    from fractions import Fraction
    exp = 0
    if "*^" in text:
        text, e = text.split("*^")
        exp = int(e)
    mark = None
    if "``" in text:
        text, acc = text.split("``")
        mark = ("acc", float(acc))
    elif "`" in text:
        text, prec = text.split("`")
        mark = ("prec", float(prec)) if prec else ("machine", None)
    if mark is None and "." not in text:          # 2*^3 is the exact integer 2000; 2*^-3 is 1/500
        v = Fraction(int(text)) * Fraction(10) ** exp
        if v.denominator == 1:
            return int(v)
        from .math_ops import from_sympy
        return from_sympy(sp.Rational(v.numerator, v.denominator))
    mant = f"{text}e{exp}" if exp else text
    if mark is None or mark[0] == "machine":
        return float(mant)
    exact = sp.Rational(Fraction(mant if "e" not in mant else Fraction(text) * Fraction(10) ** exp))
    if mark[0] == "acc":
        import math
        prec = mark[1] + (math.log10(abs(exact)) if exact != 0 else 0)
        if exact == 0:
            return sp.Float(0, precision=_dps_bits(mark[1]))
    else:
        prec = mark[1]
    bits = _dps_bits(prec)
    if prec <= 15.954589770191003:
        return float(exact)
    return sp.Float(exact, precision=bits)


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
        if kind in ("LPAREN", "LSQUARE", "LBRACE", "LASSOC", "LANGLE", "LDBAR"):
            level += 1
        elif kind in ("RPAREN", "RSQUARE", "RBRACE", "RASSOC", "RANGLE", "RDBAR"):
            level -= 1
        if kind == "NOTATION":
            value = _NOTATION_INPUT[value]
            if value in ("GreaterEqual", "LessEqual"):
                kind = "GEQ" if value == "GreaterEqual" else "LEQ"
            elif _NOTATION.OPS[value][2] == "prefix":
                kind = "NOTATIONPREFIX"
        if kind == "INTEGER":
            value = int(value)
        elif kind == "FLOAT":
            value = float(value)
        elif kind in ("BIGNUM", "SCINUM"):
            value = _number_literal(value)
            kind = "INTEGER" if isinstance(value, int) else "FLOAT"
        elif kind == "STRING":
            try:
                value = json.loads(_wl_escapes(value) if "\\:" in value or "\\." in value else value)
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
        self._band = []        # per-product flags: a CenterDot..Backslash operator was used (see parse_multiplication)

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
        expr = self.parse_put()
        if self.at("SEMICOLON"):
            exprs = [expr]
            while self.at("SEMICOLON"):
                self.advance()
                if self.peek() is not None and not self.at("RPAREN", "RSQUARE", "RBRACE"):
                    exprs.append(self.parse_put())
            if len(exprs) == 1:
                return exprs[0]
            return Expr("CompoundExpression", *exprs)
        return expr

    def parse_put(self) -> Any:
        """expr >> file (Put) and expr >>> file (PutAppend), binding more loosely than assignment."""
        expr = self.parse_assignment()
        while self.at("PUT", "PUTAPPEND"):
            tok = self.advance()
            name = tok.value[3 if tok.type == "PUTAPPEND" else 2:].strip()
            if name.startswith('"'):
                name = json.loads(name)
            expr = Expr("PutAppend" if tok.type == "PUTAPPEND" else "Put", expr, name)
        return expr

    def parse_assignment(self) -> Any:
        """Handle =, :=, ^=, ^:=, f /: lhs = rhs, =., +=, -=, *=, /=, //= (right-associative)."""
        expr = self._notation(50, self._parse_vertical_separator)
        ops = {"SET": "Set", "SETDELAYED": "SetDelayed",
               "UPSET": "UpSet", "UPSETDELAYED": "UpSetDelayed", "APPLYTO": "ApplyTo"}
        tok = self.peek()
        if tok is not None and tok.type == "UNSET":
            self.advance()
            return Expr("Unset", expr)
        if tok is not None and tok.type == "TAGSET":
            self.advance()
            lhs = self.parse_postfix_application()
            op = self.peek()
            if op is not None and op.type == "UNSET":
                self.advance()
                return Expr("TagUnset", expr, lhs)
            if op is None or op.type not in ("SET", "SETDELAYED"):
                raise ParseError("'/:' must be followed by lhs = rhs, lhs := rhs or lhs =.")
            self.advance()
            rhs = self.parse_assignment()
            return Expr("TagSet" if op.type == "SET" else "TagSetDelayed", expr, lhs, rhs)
        if tok is not None and tok.type in ops:
            self.advance()
            rhs = self.parse_assignment()
            return Expr(ops[tok.type], expr, rhs)
        return expr

    def parse_postfix_application(self) -> Any:
        """Handle expr // f  (== f[expr]), lowest precedence above assignment."""
        expr = self._notation(80, self.parse_function)
        while self.at("POSTFIXAPP"):
            self.advance()
            f = self._notation(80, self.parse_function)
            expr = Expr(_as_head(f), expr)
        return expr

    def parse_function(self) -> Any:
        """Handle postfix & (pure functions) and direct application (body &)[args]."""
        expr = self.parse_add_to()
        while self.at("FUNCTION"):
            self.advance()
            expr = Expr("Function", expr)
            while self.at("LSQUARE", "MAP", "APPLY", "APPLYLEVEL1", "AT", "MAPALL"):
                if self.at("LSQUARE"):
                    self.advance()
                    args = self._parse_arglist("RSQUARE")
                    self.expect("RSQUARE")
                    expr = Expr(expr, *args)
                    continue
                op = self.advance().type
                rhs = self.parse_add_to()
                if op == "MAP":
                    expr = Expr("Map", expr, rhs)
                elif op == "MAPALL":
                    expr = Expr("MapAll", expr, rhs)
                elif op == "APPLY":
                    expr = Expr("Apply", expr, rhs)
                elif op == "APPLYLEVEL1":
                    expr = Expr("Apply", expr, rhs, Expr("List", 1))
                else:
                    expr = Expr(expr, rhs)
        return expr

    _ADD_TO = {"ADDTO": "AddTo", "SUBTRACTFROM": "SubtractFrom", "TIMESBY": "TimesBy", "DIVIDEBY": "DivideBy"}

    def parse_add_to(self) -> Any:
        """Handle x += y, x -= y, x *= y, x /= y (precedence 100, right-grouping: between & (90) and /. (110), so
        ``x += 1 &`` is Function[AddTo[x, 1]] and ``x += y // f`` is f[AddTo[x, y]], as in Wolfram)."""
        expr = self.parse_replace_all()
        tok = self.peek()
        if tok is not None and tok.type in self._ADD_TO:
            self.advance()
            return Expr(self._ADD_TO[tok.type], expr, self.parse_add_to())
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
        """Handle -> and :> (precedence 120, right-associative). TwoWayRule and Condition bind tighter."""
        expr = self.parse_two_way_rule()
        if self.at("RULE", "RULEDELAYED"):
            op = self.advance().type
            rhs = self.parse_rule()
            expr = Expr("RuleDelayed" if op == "RULEDELAYED" else "Rule", expr, rhs)
        return expr

    def _at_named(self, *names) -> bool:
        tok = self.peek()
        return tok is not None and tok.type == "NAMEDOP" and tok.value[2:-1] in names

    def parse_two_way_rule(self) -> Any:
        """Handle a <-> b and a \\[TwoWayRule] b (precedence 125, right-associative: between -> and /;)."""
        expr = self.parse_condition()
        if self.at("TWOWAYRULE") or self._at_named("TwoWayRule"):
            self.advance()
            return Expr("TwoWayRule", expr, self.parse_two_way_rule())
        return expr

    def parse_condition(self) -> Any:
        """Handle /; (Condition)."""
        expr = self.parse_string_expression()
        while self.at("CONDITION"):
            self.advance()
            cond = self.parse_string_expression()
            expr = Expr("Condition", expr, cond)
        return expr

    def parse_string_expression(self) -> Any:
        """Handle a ~~ b ~~ ... (StringExpression, flat; precedence 135)."""
        expr = self.parse_colon()
        if self.at("STRINGEXPR"):
            parts = [expr]
            while self.at("STRINGEXPR"):
                self.advance()
                parts.append(self.parse_colon())
            return Expr("StringExpression", *parts)
        return expr

    def parse_colon(self) -> Any:
        """Handle x : patt (Pattern) and x_ : default (Optional), precedence 140-150."""
        expr = self.parse_alternatives()
        if self.at("COLON"):
            self.advance()
            rhs = self.parse_alternatives()
            if isinstance(expr, Symbol):
                return Expr("Pattern", expr, rhs)
            if isinstance(expr, Expr) and expr.head in ("Pattern", "Blank", "BlankSequence", "BlankNullSequence"):
                return Expr("Optional", expr, rhs)
            raise ParseError("':' must follow a symbol (x:patt) or a pattern (x_:default)")
        return expr

    def parse_alternatives(self) -> Any:
        """Handle a | b | ... (Alternatives, flat; precedence 160)."""
        expr = self.parse_repeated()
        if self.at("ALTERNATIVES"):
            alts = [expr]
            while self.at("ALTERNATIVES"):
                self.advance()
                alts.append(self.parse_repeated())
            return Expr("Alternatives", *alts)
        return expr

    def parse_repeated(self) -> Any:
        """Handle postfix p.. (Repeated) and p... (RepeatedNull), precedence 170."""
        expr = self._notation(190, self.parse_or)
        while self.at("REPEATED", "REPEATEDNULL"):
            expr = Expr("Repeated" if self.advance().type == "REPEATED" else "RepeatedNull", expr)
        return expr

    def parse_or(self) -> Any:
        terms = [self.parse_and()]
        while self.at("OR"):
            self.advance()
            terms.append(self.parse_and())
        return terms[0] if len(terms) == 1 else Expr("Or", *terms)     # a op b op c parses to Or[a, b, c]

    def parse_and(self) -> Any:
        terms = [self.parse_not()]
        while self.at("AND"):
            self.advance()
            terms.append(self.parse_not())
        return terms[0] if len(terms) == 1 else Expr("And", *terms)     # a op b op c parses to And[a, b, c]

    def parse_not(self) -> Any:
        if self.at("NOT"):
            self.advance()
            operand = self.parse_not()
            return Expr("Not", operand)
        return self._parse_element_level()

    def _parse_element_level(self) -> Any:
        """Precedence 250: NotElement, Subset, ... (notation table) and x \\[Distributed] dist (binary)."""
        expr = self._notation(250, self.parse_sameq)
        if self._at_named("Distributed"):
            self.advance()
            expr = Expr("Distributed", expr, self.parse_sameq())
            if self._at_named("Distributed") or self._at_notation(250):
                raise ParseError("Distributed and operators of precedence 250 cannot be chained; use parentheses")
        return expr

    def parse_sameq(self) -> Any:
        """Handle a === b === c (SameQ) and a =!= b =!= c (UnsameQ): precedence 260, below the arrows (270) and
        ==, < (290), so ``a == b === c`` is SameQ[Equal[a, b], c] as in Wolfram.  Chains are flat; SameQ and
        UnsameQ are not mixed without parentheses (ParseError)."""
        expr = self._parse_arrows()
        if not self.at("SAMEQ", "UNSAMEQ"):
            return expr
        kind = self.peek().type
        args = [expr]
        while self.at(kind):
            self.advance()
            args.append(self._parse_arrows())
        if self.at("SAMEQ", "UNSAMEQ"):
            raise ParseError("=== and =!= have the same precedence; use parentheses")
        return Expr("SameQ" if kind == "SAMEQ" else "UnsameQ", *args)

    _COMP_OPS = {
        "EQUAL": "Equal", "UNEQUAL": "Unequal",
        "LESS": "Less", "GREATER": "Greater",
        "LEQ": "LessEqual", "GEQ": "GreaterEqual",
    }

    _EXPR_START = ("INTEGER", "FLOAT", "STRING", "SYMBOL", "LPAREN", "LBRACE", "LASSOC", "MINUS", "PLUS",
                   "SLOT", "SLOTSEQ", "SLOTNAME", "BLANK", "BLANKS", "BLANKNS", "BLANKOPT", "NOT")

    def parse_span(self) -> Any:
        """Handle i;;j, i;;j;;k, ;;j, i;; and ;; (Span, precedence 305)."""
        if self.at("SPAN"):
            start = 1
        else:
            start = self.parse_addition()
            if not self.at("SPAN"):
                return start
        self.advance()
        end = self.parse_addition() if self.at(*self._EXPR_START) else Symbol("All")
        if self.at("SPAN"):
            self.advance()
            step = self.parse_addition()
            return Expr("Span", start, end, step)
        return Expr("Span", start, end)

    def parse_comparison(self) -> Any:
        expr = self._parse_edges()
        if self._at_notation(290):                        # x \[Succeeds] y ... (W31): never mixed with comparisons
            expr = self._notation(290, self._parse_edges, first=expr)
            if self.peek() is not None and (self.peek().type in self._COMP_OPS or self.peek().type == "VECTOROP"):
                raise ParseError(f"{expr.head} and comparison operators have the same precedence; use parentheses")
            return expr
        comparisons = []
        while self.peek() is not None and (self.peek().type in self._COMP_OPS or self.peek().type == "VECTOROP"):
            tok = self.advance()
            if tok.type == "VECTOROP":                    # x \[VectorLess] y  ->  VectorLess[{x, y}]
                op = tok.value[2:-1]
                rhs = self._parse_edges()
                last = comparisons[-1] if comparisons else None
                if last is not None and last.head == op and last.args[0].args[-1] is expr:
                    comparisons[-1] = Expr(op, Expr("List", *last.args[0].args, rhs))
                else:
                    comparisons.append(Expr(op, Expr("List", expr, rhs)))
                expr = rhs
                continue
            op = self._COMP_OPS[tok.type]
            rhs = self._parse_edges()
            comparisons.append(Expr(op, expr, rhs))
            expr = rhs
        if comparisons and self._at_notation(290):
            raise ParseError(f"comparison operators and {self.peek().value} have the same precedence; use parentheses")
        if not comparisons:
            return expr
        if len(comparisons) == 1:
            return comparisons[0]
        # a < b < c  ==  a < b && b < c
        return Expr("And", *comparisons)

    def parse_addition(self) -> Any:
        expr = self._notation(330, self._parse_cup)
        while self.at("PLUS", "MINUS") or self._at_op("MinusPlus") or self._at_op("PlusMinus"):
            tok = self.advance()
            rhs = self._notation(330, self._parse_cup)
            if tok.type == "NOTATION":                    # x \[MinusPlus] y, x \[PlusMinus] y group left, like x - y
                expr = Expr(tok.value, expr, rhs)
            elif tok.type == "PLUS":
                expr = Expr("Plus", expr, rhs)
            elif isinstance(rhs, (int, float)) and not isinstance(rhs, bool):
                expr = Expr("Plus", expr, -rhs)          # Wolfram FullForm: a - 1 is Plus[a, -1]
            else:
                expr = Expr("Plus", expr, Expr("Times", -1, rhs))
        return expr

    def parse_multiplication(self) -> Any:
        self._band.append(False)
        try:
            return self._parse_multiplication()
        finally:
            self._band.pop()

    def _parse_multiplication(self) -> Any:
        expr = self._parse_band()
        divided = False
        while True:
            # Explicit * or /
            if self.at("TIMES"):
                self.advance()
                rhs = self._parse_band()
                expr = Expr("Times", expr, rhs)
            elif self.at("DIVIDE"):
                self.advance()
                divided = True
                rhs = self._parse_band()
                expr = Expr("Times", expr, Expr("Power", rhs, -1))
            # Implicit multiplication: juxtaposition
            elif self._can_implicit_multiply(expr):
                rhs = self._parse_band()
                expr = Expr("Times", expr, rhs)
            else:
                break
            if divided and self._band[-1]:
                raise ParseError("'/' and the operators CenterDot, CircleTimes, Vee, Diamond, Backslash bind "
                                 "differently in Wolfram (Divide binds tighter); use parentheses")
        return expr

    def _can_implicit_multiply(self, left) -> bool:
        """Check if current position allows implicit multiplication."""
        tok = self.peek()
        if tok is None:
            return False
        return tok.type in ("SYMBOL", "LPAREN", "LBRACE", "LASSOC", "INTEGER", "FLOAT", "SLOT", "SLOTSEQ",
                            "SLOTNAME", "LANGLE", "LDBAR", "NOTATIONPREFIX", "STRING")

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
        if self._at_op("MinusPlus") or self._at_op("PlusMinus"):     # \[MinusPlus] x, \[PlusMinus] x: like unary minus
            return Expr(self.advance().value, self.parse_unary())
        return self.parse_dot()

    def _tight_operand(self, level) -> Any:
        """Right operand of a binary operator above unary minus (^, /@, @@, @): a leading - or + (or ± / ∓)
        applies to one operand of that same level only, so ``a^-b c`` is Times[Power[a, -b], c] and
        ``b ^ -c . d`` is Dot[Power[b, -c], d]."""
        if self.at("MINUS"):
            self.advance()
            operand = self._tight_operand(level)
            if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                return -operand
            return Expr("Times", -1, operand)
        if self.at("PLUS"):
            self.advance()
            return self._tight_operand(level)
        if self._at_op("MinusPlus") or self._at_op("PlusMinus"):
            return Expr(self.advance().value, self._tight_operand(level))
        return level()

    def parse_dot(self) -> Any:
        """Handle a . b . c (Dot, precedence 490: between Times and Power/Map), flat."""
        expr = self.parse_ncmult()
        if not self.at("DOT"):
            return expr
        args = [expr]
        while self.at("DOT"):
            self.advance()
            args.append(self.parse_ncmult())
        return Expr("Dot", *args)

    def parse_ncmult(self) -> Any:
        """Handle a ** b ** c (NonCommutativeMultiply, precedence 510: above Dot, below Power), flat."""
        expr = self._notation(520, self._parse_small_circle)
        if not self.at("NCMULT"):
            return expr
        args = [expr]
        while self.at("NCMULT"):
            self.advance()
            args.append(self._notation(520, self._parse_small_circle))
        return Expr("NonCommutativeMultiply", *args)

    def parse_power(self) -> Any:
        """Handle a ^ b (Power, precedence 590, right-associative).  The base and exponent are StringJoin-level
        operands (StringJoin 600, Factorial 610, Map 620, @ 640, ... bind tighter), so ``b ^ c . d`` is
        Dot[Power[b, c], d] and ``a ^ b ^ c`` is Power[a, Power[b, c]]."""
        base = self.parse_stringjoin()
        if self.at("POWER"):
            self.advance()
            return Expr("Power", base, self._tight_operand(self.parse_power))
        return base

    def parse_stringjoin(self) -> Any:
        """Handle a <> b <> c (StringJoin, precedence 600: above Power, below Factorial), flat."""
        expr = self.parse_factorial()
        if self.at("STRINGJOIN"):
            parts = [expr]
            while self.at("STRINGJOIN"):
                self.advance()
                parts.append(self.parse_factorial())
            return Expr("StringJoin", *parts)
        return expr

    def parse_factorial(self) -> Any:
        """Handle postfix n! (Factorial) and n!! (Factorial2), precedence 610: below Map (620) and @ (640), so
        ``f @ x!`` is Factorial[f[x]] as in Wolfram.  ``!!`` written without a space is Factorial2."""
        expr = self.parse_map()
        while self.at("NOT") and self._factorial_position():
            tok = self.advance()
            nxt = self.peek()
            if nxt is not None and nxt.type == "NOT" and nxt.pos == tok.pos + 1 and self._factorial_position():
                self.advance()
                expr = Expr("Factorial2", expr)
            else:
                expr = Expr("Factorial", expr)
        return expr

    def parse_map(self) -> Any:
        """Handle f /@ x, f //@ x, f @@ x, f @@@ x (precedence 620, right-associative)."""
        expr = self.parse_prefix_application()
        if self.at("MAP", "APPLY", "APPLYLEVEL1", "MAPALL"):
            op = self.advance().type
            rhs = self._tight_operand(self.parse_map)
            if op == "MAP":
                expr = Expr("Map", expr, rhs)
            elif op == "MAPALL":
                expr = Expr("MapAll", expr, rhs)
            elif op == "APPLY":
                expr = Expr("Apply", expr, rhs)
            else:
                expr = Expr("Apply", expr, rhs, Expr("List", 1))
        return expr

    def parse_prefix_application(self) -> Any:
        """Handle f @ x  (== f[x]; precedence 640, right-associative)."""
        expr = self.parse_right_composition()
        if self.at("AT"):
            self.advance()
            arg = self._tight_operand(self.parse_prefix_application)
            return Expr(_as_head(expr), arg)
        return expr

    def parse_right_composition(self) -> Any:
        """Handle f /* g /* h (RightComposition, precedence 650, flat); its operands are Composition-level (655),
        so ``f /* g @* h`` is RightComposition[f, Composition[g, h]]."""
        expr = self.parse_composition()
        if not self.at("RIGHTCOMPOSITION"):
            return expr
        args = [expr]
        while self.at("RIGHTCOMPOSITION"):
            self.advance()
            args.append(self.parse_composition())
        return Expr("RightComposition", *args)

    def parse_composition(self) -> Any:
        """Handle f @* g @* h (Composition, precedence 655: above @ and /*), flat."""
        expr = self.parse_postfix()
        if not self.at("COMPOSITION"):
            return expr
        args = [expr]
        while self.at("COMPOSITION"):
            self.advance()
            args.append(self.parse_postfix())
        return Expr("Composition", *args)

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
            else:
                break
        return expr

    def _factorial_position(self) -> bool:
        nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else None
        return nxt is None or nxt.type not in (
            "SYMBOL", "INTEGER", "FLOAT", "LPAREN", "LBRACE", "LASSOC", "SLOT", "SLOTSEQ", "SLOTNAME", "STRING")

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

    # ── Notation operators (W31/W32): levels that only act when their operator tokens are present ──

    def _at_notation(self, prec) -> bool:
        tok = self.peek()
        return tok is not None and tok.type == "NOTATION" and _NOTATION.OPS[tok.value][1] == prec

    def _at_op(self, name) -> bool:
        tok = self.peek()
        return tok is not None and tok.type == "NOTATION" and tok.value == name

    def _notation(self, prec, operand, first=None) -> Any:
        """Operators of one precedence level: nary chains (x op y op z -> op[x, y, z]), left- or right-grouping
        binary operators.  Different operators of one level are not mixed without parentheses (ParseError)."""
        expr = operand() if first is None else first
        if not self._at_notation(prec):
            return expr
        head = self.peek().value
        kind = _NOTATION.OPS[head][2]
        if kind == "nary":
            args = [expr]
            while self._at_op(head):
                self.advance()
                args.append(operand())
            expr = Expr(head, *args)
        elif kind == "left":
            while self._at_notation(prec) and _NOTATION.OPS[self.peek().value][2] == "left":
                h = self.advance().value
                expr = Expr(h, expr, operand())
        else:                                             # right
            items, heads = [expr], []
            while self._at_notation(prec) and _NOTATION.OPS[self.peek().value][2] == "right":
                heads.append(self.advance().value)
                items.append(operand())
            expr = items[-1]
            for h, x in zip(reversed(heads), reversed(items[:-1])):
                expr = Expr(h, x, expr)
        if self._at_notation(prec):
            raise ParseError(f"{head} and {self.peek().value} have the same precedence; use parentheses")
        if 410 <= prec <= 460 and self._band:
            self._band[-1] = True
        return expr

    def _parse_vertical_separator(self) -> Any:
        return self._notation(60, self.parse_postfix_application)

    def _parse_arrows(self) -> Any:
        return self._notation(270, lambda: self._notation(280, self.parse_comparison))

    def _parse_edges(self) -> Any:
        """Precedence 295: x \\[DirectedEdge] y and x \\[UndirectedEdge] y (right-grouping), between == (290) and
        SquareUnion (300), so ``a \\[DirectedEdge] b -> c`` is Rule[DirectedEdge[a, b], c] as in Wolfram."""
        expr = self._parse_square_union()
        if self._at_named("DirectedEdge", "UndirectedEdge"):
            head = self.advance().value[2:-1]
            return Expr(head, expr, self._parse_edges())
        return expr

    def _parse_square_union(self) -> Any:
        return self._notation(300, self._parse_square_intersection)

    def _parse_square_intersection(self) -> Any:
        start = self.pos
        expr = self._notation(305, self.parse_span)
        if isinstance(expr, Expr) and expr.head == "SquareIntersection":
            depth = 0
            for tok in self.tokens[start:self.pos]:
                if tok.type in ("LPAREN", "LSQUARE", "LBRACE", "LASSOC", "LANGLE", "LDBAR"):
                    depth += 1
                elif tok.type in ("RPAREN", "RSQUARE", "RBRACE", "RASSOC", "RANGLE", "RDBAR"):
                    depth -= 1
                elif tok.type == "SPAN" and depth == 0:
                    raise ParseError("SquareIntersection and Span (;;) have the same precedence; use parentheses")
        return expr

    def _parse_cup(self) -> Any:
        """Levels 340 (Cup) .. 390 (Star), between CirclePlus and Times."""
        f = self.parse_multiplication
        for prec in (390, 370, 360, 350):
            f = (lambda g, q: (lambda: self._notation(q, g)))(f, prec)
        return self._notation(340, f)

    def _parse_band(self) -> Any:
        """Times operands: levels 410 (CenterDot) .. 460 (Backslash), between Times and Divide."""
        f = self.parse_unary
        for prec in (460, 450, 430, 420):
            f = (lambda g, q: (lambda: self._notation(q, g)))(f, prec)
        return self._notation(410, f)

    def _parse_small_circle(self) -> Any:
        return self._notation(530, self._parse_prefix_d)

    def _parse_prefix_d(self) -> Any:
        """Prefix notation operators: DifferentialD, CapitalDifferentialD (550), Del (550) and Square (540), all
        right-grouping; no binary operator lies between 540 and 550, so they share this level."""
        if self.at("NOTATIONPREFIX"):
            head = self.advance().value
            return Expr(head, self._parse_prefix_d())
        return self._notation(580, self.parse_power)

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

        if tok.type == "SLOTNAME":
            self.advance()
            return Expr("Slot", tok.value[1:])

        if tok.type == "SLOTSEQ":
            self.advance()
            index = int(tok.value[2:] or "1")
            if index < 1:
                raise ParseError("Slot sequences start at ##1")
            return Expr("SlotSequence", index)

        if tok.type in ("INTEGER", "FLOAT", "STRING"):
            self.advance()
            return tok.value

        if tok.type == "OUT":                        # %, %%, %%%, %n
            self.advance()
            t = tok.value[1:]
            if t.isdigit():
                return Expr("Out", int(t))
            return Expr("Out") if not t else Expr("Out", -(len(t) + 1))

        if tok.type == "GET":                        # <<file or <<"file"
            self.advance()
            name = tok.value[2:].strip()
            if name.startswith('"'):
                name = json.loads(name)
            return Expr("Get", name)

        _blank_kinds = {"BLANK": "Blank", "BLANKS": "BlankSequence", "BLANKNS": "BlankNullSequence"}

        if tok.type == "SYMBOL":
            self.advance()
            name = tok.value
            if name == "True":
                return True
            if name == "False":
                return False
            nxt = self.peek()
            if nxt is not None and nxt.type == "BLANKOPT" and nxt.pos == tok.pos + len(name):
                self.advance()
                return Expr("Optional", Expr("Pattern", Symbol(name), Expr("Blank")))
            if nxt is not None and nxt.type in _blank_kinds and nxt.pos == tok.pos + len(name):
                self.advance()
                blank = self._blank_tail(_blank_kinds[nxt.type])
                return self._pattern_test(Expr("Pattern", Symbol(name), blank))
            if nxt is not None and nxt.type == "MESSAGENAME":
                return self._message_name(Symbol(name))
            return Symbol(name)

        if tok.type == "BLANKOPT":
            self.advance()
            return Expr("Optional", Expr("Blank"))

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

        if tok.type == "LASSOC":
            self.advance()
            args = self._parse_arglist("RASSOC")
            self.expect("RASSOC")
            return Expr("Association", *args)

        if tok.type == "LANGLE":                     # \[LeftAngleBracket] x, ... \[RightAngleBracket] (W31)
            self.advance()
            args = self._parse_arglist("RANGLE")
            self.expect("RANGLE")
            return Expr("AngleBracket", *args)

        if tok.type == "LDBAR":                      # \[LeftDoubleBracketingBar] x, ... \[RightDoubleBracketingBar]
            self.advance()
            args = self._parse_arglist("RDBAR")
            self.expect("RDBAR")
            return Expr("DoubleBracketingBar", *args)

        raise ParseError(f"Unexpected token: {tok}")

    def _message_name(self, sym):
        """symbol::tag and symbol::tag::lang (MessageName; tag and lang may be symbols or strings)."""
        parts = []
        while self.at("MESSAGENAME") and len(parts) < 2:
            self.advance()
            tok = self.peek()
            if tok is None or tok.type not in ("SYMBOL", "STRING"):
                raise ParseError("'::' must be followed by a message tag")
            self.advance()
            parts.append(tok.value)
        return Expr("MessageName", sym, *parts)

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


# Applied to every parsed expression (symbol creation and context resolution; set by wl_evaluation).
POST_PARSE_HOOK = None


def _information_query(text: str):
    """?name and ??name (Information[\"name\", LongForm -> False] and Information[\"name\"])."""
    body = text.strip()
    long_form = body.startswith("??")
    name = body[2:].strip() if long_form else body[1:].strip()
    if not name or any(c.isspace() for c in name):
        raise ParseError("? must be followed by a symbol name or a name pattern")
    if long_form:
        return Expr("Information", name)
    return Expr("Information", name, Expr("Rule", Symbol("LongForm"), False))


def parse(text: str) -> Any:
    """Parse a single Wolfram-style expression string into a MikoshiLang Expr."""
    if text.lstrip().startswith("?"):
        result = _information_query(text)
        return POST_PARSE_HOOK(result) if POST_PARSE_HOOK is not None else result
    tokens = tokenize(text)
    if not tokens:
        return None
    parser = Parser(tokens)
    result = parser.parse_expression()
    if parser.peek() is not None:
        raise ParseError(f"Unexpected trailing token: {parser.peek()}")
    if POST_PARSE_HOOK is not None:
        result = POST_PARSE_HOOK(result)
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
