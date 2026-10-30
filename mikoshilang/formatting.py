"""Text forms of expressions: InputForm (default display) and FullForm.

InputForm output is valid MikoshiLang/Wolfram input: ``x^3/3``, ``Sqrt[Pi]``,
``1/(2*x)``, ``a - b``.  FullForm shows the canonical tree, e.g.
``Times[Rational[1, 3], Power[x, 3]]``.
"""
from __future__ import annotations

import json
from fractions import Fraction

from .expr import Expr, Symbol

# Wolfram precedences (higher binds tighter)
P_COMPOUND, P_SET, P_FUNCTION, P_REPLACE, P_RULE, P_CONDITION = 10, 40, 90, 110, 120, 130
P_OR, P_AND, P_NOT, P_COMPARE, P_SAMEQ = 215, 215, 230, 290, 290
P_PLUS, P_TIMES, P_UMINUS, P_DIVIDE, P_POWER, P_STRINGJOIN = 310, 400, 480, 470, 590, 600
P_CALL, P_ATOM = 1000, 2000

_INFIX = {
    "Equal": ("==", P_COMPARE), "Unequal": ("!=", P_COMPARE), "Less": ("<", P_COMPARE),
    "Greater": (">", P_COMPARE), "LessEqual": ("<=", P_COMPARE), "GreaterEqual": (">=", P_COMPARE),
    "SameQ": ("===", P_SAMEQ), "UnsameQ": ("=!=", P_SAMEQ), "And": ("&&", P_AND), "Or": ("||", P_OR),
    "StringJoin": ("<>", P_STRINGJOIN), "Alternatives": ("|", 160), "StringExpression": ("~~", 135),
    "Composition": ("@*", 625), "RightComposition": ("/*", 624),
}
_BINARY = {
    "Rule": ("->", P_RULE), "RuleDelayed": (":>", P_RULE), "Set": ("=", P_SET),
    "SetDelayed": (":=", P_SET), "ReplaceAll": ("/.", P_REPLACE), "ReplaceRepeated": ("//.", P_REPLACE),
    "Condition": ("/;", P_CONDITION), "AddTo": ("+=", P_SET), "SubtractFrom": ("-=", P_SET),
    "TimesBy": ("*=", P_SET), "DivideBy": ("/=", P_SET),
}


def is_int(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    try:
        import sympy as sp
        return isinstance(x, sp.Integer)
    except Exception:  # pragma: no cover
        return False


def rational_value(e):
    """Return Fraction for an exact rational number representation, else None.

    Rationals are represented as ``Times[p, Power[q, -1]]`` (q > 1) or plain ints.
    """
    if is_int(e):
        return Fraction(int(e))
    if (isinstance(e, Expr) and e.head == "Times" and len(e.args) == 2 and is_int(e.args[0])
            and isinstance(e.args[1], Expr) and e.args[1].head == "Power" and len(e.args[1].args) == 2
            and is_int(e.args[1].args[0]) and int(e.args[1].args[0]) > 1 and is_int(e.args[1].args[1])
            and int(e.args[1].args[1]) == -1):
        return Fraction(int(e.args[0]), int(e.args[1].args[0]))
    try:
        import sympy as sp
        if isinstance(e, sp.Rational):
            return Fraction(int(e.p), int(e.q))
    except Exception:  # pragma: no cover
        pass
    return None


def _is_real_number(e) -> bool:
    if isinstance(e, bool):
        return False
    if isinstance(e, (int, float)):
        return True
    try:
        import sympy as sp
        return isinstance(e, sp.Number)
    except Exception:  # pragma: no cover
        return False


def _number_text(x) -> str:
    if isinstance(x, float):
        if x != x:
            return "Indeterminate"
        if x in (float("inf"), float("-inf")):
            return "Infinity" if x > 0 else "-Infinity"
        return repr(x)
    try:
        import sympy as sp
        if isinstance(x, sp.Float) and x._prec <= 53:
            return repr(float(x))
    except Exception:  # pragma: no cover
        pass
    return str(x)


def _neg(e):
    """If e is syntactically negative, return its absolute counterpart; else None."""
    if _is_real_number(e):
        return -e if e < 0 else None
    q = rational_value(e)
    if q is not None and not is_int(e):
        return _rational_expr(-q) if q < 0 else None
    if isinstance(e, Expr) and e.head == "Times" and e.args:
        first = e.args[0]
        if _is_real_number(first) and first < 0:
            rest = e.args[1:]
            if first == -1:
                return rest[0] if len(rest) == 1 else Expr("Times", *rest)
            return Expr("Times", -first, *rest)
        fq = rational_value(first)
        if fq is not None and fq < 0:
            return Expr("Times", _rational_expr(-fq), *e.args[1:])
    return None


def _rational_expr(q: Fraction):
    if q.denominator == 1:
        return q.numerator
    return Expr("Times", q.numerator, Expr("Power", q.denominator, -1))


def _paren(text: str, prec: int, parent: int) -> str:
    return f"({text})" if prec < parent else text


def _fmt_head(h) -> str:
    if isinstance(h, str):
        return h
    return input_form(h, P_CALL)


def _flatten_times(args):
    out = []
    for a in args:
        if isinstance(a, Expr) and a.head == "Times" and rational_value(a) is None:
            out.extend(_flatten_times(a.args))
        else:
            out.append(a)
    return out


def _format_times(e, parent):
    factors = _flatten_times(e.args)
    if sum(1 for f in factors if _is_real_number(f) or rational_value(f) is not None) > 1:
        return _paren("*".join(input_form(f, P_TIMES + 1) for f in factors), P_TIMES, parent)
    coeff = Fraction(1)
    num, den = [], []
    float_coeff = None
    for f in factors:
        q = rational_value(f)
        if q is not None:
            coeff *= q
            continue
        if isinstance(f, float) or (_is_real_number(f) and not is_int(f)):
            float_coeff = f if float_coeff is None else float_coeff * f
            continue
        if isinstance(f, Expr) and f.head == "Power" and len(f.args) == 2:
            base, ex = f.args
            neg = _neg(ex)
            if neg is not None:
                den.append(base if (is_int(neg) and int(neg) == 1) else Expr("Power", base, neg))
                continue
        num.append(f)
    negative = coeff < 0
    if float_coeff is not None:
        value = float_coeff * (float(coeff) if coeff != 1 else 1)
        negative = value < 0
        num.insert(0, abs(value) if negative else value)
    else:
        if coeff.numerator not in (1, -1):
            num.insert(0, abs(coeff.numerator))
        if coeff.denominator != 1:
            den.insert(0, coeff.denominator)
    if not num:
        num_text = "1"
        num_prec = P_ATOM
    elif len(num) == 1:
        num_text = input_form(num[0], P_TIMES + 1 if den else P_TIMES)
        num_prec = P_TIMES if den else _prec_of(num[0])
    else:
        num_text = "*".join(input_form(f, P_TIMES + 1) for f in num)
        num_prec = P_TIMES
    if den:
        if len(den) == 1:
            den_text = input_form(den[0], P_POWER)
        else:
            den_text = "(" + "*".join(input_form(f, P_TIMES + 1) for f in den) + ")"
        text, prec = f"{num_text}/{den_text}", P_TIMES
    else:
        text, prec = num_text, num_prec
    if negative:
        text, prec = "-" + _paren(text, prec, P_TIMES), P_UMINUS
    return _paren(text, prec, parent)


def _prec_of(e) -> int:
    if isinstance(e, Expr):
        if e.head == "Plus":
            return P_PLUS
        if e.head == "Times":
            return P_TIMES
        if e.head == "Power":
            return P_POWER
        if e.head in _INFIX:
            return _INFIX[e.head][1]
        if e.head in _BINARY:
            return _BINARY[e.head][1]
        return P_CALL
    if _is_real_number(e) and e < 0:
        return P_UMINUS
    return P_ATOM


def input_form(e, parent: int = 0) -> str:
    """Render e as re-parseable Wolfram-style input."""
    if e is None:
        return "Null"
    if isinstance(e, bool):
        return "True" if e else "False"
    if isinstance(e, Symbol):
        return e.name
    if isinstance(e, str):
        return json.dumps(e, ensure_ascii=False)
    if _is_real_number(e):
        text = _number_text(e)
        return _paren(text, P_UMINUS if text.startswith("-") else P_ATOM, parent)
    if isinstance(e, complex):
        return input_form(Expr("Plus", e.real, Expr("Times", e.imag, Symbol("I"))), parent)
    if isinstance(e, tuple):
        return "Sequence[" + ", ".join(input_form(a) for a in e) + "]"
    if not isinstance(e, Expr):
        return str(e)

    h, args = e.head, e.args
    q = rational_value(e)
    if q is not None and not is_int(e):
        text = f"{abs(q.numerator)}/{q.denominator}"
        if q < 0:
            return _paren("-" + text, P_UMINUS, parent)
        return _paren(text, P_TIMES, parent)
    if isinstance(h, str):
        if h == "Plus" and args:
            parts = []
            for i, a in enumerate(args):
                neg = _neg(a)
                if i == 0:
                    parts.append(input_form(a, P_PLUS))
                elif neg is not None:
                    parts.append(" - " + input_form(neg, P_PLUS + 1))
                else:
                    parts.append(" + " + input_form(a, P_PLUS + 1))
            return _paren("".join(parts), P_PLUS, parent)
        if h == "Times" and args:
            return _format_times(e, parent)
        if h == "Power" and len(args) == 2:
            base, ex = args
            qe = rational_value(ex)
            if qe is not None and qe == Fraction(1, 2):
                return f"Sqrt[{input_form(base)}]"
            if qe is not None and qe < 0:
                return _format_times(Expr("Times", e), parent)
            if isinstance(ex, Expr) and _neg(ex) is not None:
                return _format_times(Expr("Times", e), parent)
            b = input_form(base, P_POWER + 1)
            x = input_form(ex, P_POWER + 1)
            return _paren(f"{b}^{x}", P_POWER, parent)
        if h == "List":
            return "{" + ", ".join(input_form(a) for a in args) + "}"
        if h in _INFIX and len(args) >= 2:
            op, prec = _INFIX[h]
            return _paren(f" {op} ".join(input_form(a, prec + 1) for a in args), prec, parent)
        if h in _BINARY and len(args) == 2:
            op, prec = _BINARY[h]
            right_assoc = h in ("Rule", "RuleDelayed", "Set", "SetDelayed")
            left = input_form(args[0], prec + (0 if not right_assoc else 1))
            right = input_form(args[1], prec if right_assoc else prec + 1)
            return _paren(f"{left} {op} {right}", prec, parent)
        if h == "Not" and len(args) == 1:
            return _paren("!" + input_form(args[0], P_NOT), P_NOT, parent)
        if h == "Pattern" and len(args) == 2 and isinstance(args[0], Symbol):
            if isinstance(args[1], Expr) and args[1].head in ("Blank", "BlankSequence", "BlankNullSequence"):
                return args[0].name + input_form(args[1], P_ATOM)
            return _paren(args[0].name + ":" + input_form(args[1], 151), 150, parent)
        if h == "Optional" and len(args) == 1 and isinstance(args[0], Expr) and args[0].head in ("Pattern", "Blank") \
                and input_form(args[0]).endswith("_"):
            return input_form(args[0]) + "."
        if h == "Optional" and len(args) == 2:
            return _paren(input_form(args[0], 141) + ":" + input_form(args[1], 141), 140, parent)
        if h in ("Repeated", "RepeatedNull") and len(args) == 1:
            return _paren(input_form(args[0], 171) + (".." if h == "Repeated" else "..."), 170, parent)
        if h == "Association":
            return "<|" + ", ".join(input_form(a) for a in args) + "|>"
        if h == "Span" and len(args) in (2, 3):
            parts = [input_form(a, 306) for a in args]
            if isinstance(args[1], Symbol) and args[1].name == "All":
                parts[1] = ""
            return _paren(";;".join(parts), 305, parent)
        if h in ("Blank", "BlankSequence", "BlankNullSequence") and len(args) <= 1:
            mark = {"Blank": "_", "BlankSequence": "__", "BlankNullSequence": "___"}[h]
            return mark + (args[0].name if args and isinstance(args[0], Symbol) else "")
        if h == "PatternTest" and len(args) == 2:
            return input_form(args[0], P_ATOM) + "?" + input_form(args[1], P_ATOM)
        if h == "Slot" and len(args) == 1:
            if isinstance(args[0], str):
                return "#" + args[0]
            return "#" if args[0] == 1 else f"#{args[0]}"
        if h == "SlotSequence" and len(args) == 1:
            return "##" if args[0] == 1 else f"##{args[0]}"
        if h == "Function" and len(args) == 1:
            return _paren(input_form(args[0], P_FUNCTION + 1) + " &", P_FUNCTION, parent)
        if h == "CompoundExpression":
            return _paren("; ".join(input_form(a, P_COMPOUND + 1) for a in args), P_COMPOUND, parent)
        if h == "Part" and len(args) >= 2:
            return input_form(args[0], P_CALL) + "[[" + ", ".join(input_form(a) for a in args[1:]) + "]]"
        if h in ("HoldForm",) and len(args) == 1:
            return input_form(args[0], parent)
        if h == "InputForm" and len(args) == 1:
            return input_form(args[0], parent)
        if h == "FullForm" and len(args) == 1:
            return full_form(args[0])
        if h == "Increment" and len(args) == 1:
            return input_form(args[0], P_CALL) + "++"
        if h == "Decrement" and len(args) == 1:
            return input_form(args[0], P_CALL) + "--"
    # Derivative[n][f] -> f'  (only for a single variable)
    if (isinstance(h, Expr) and h.head == "Derivative" and len(h.args) == 1 and is_int(h.args[0])
            and len(args) == 1 and isinstance(args[0], Symbol)):
        return args[0].name + "'" * int(h.args[0])
    return _fmt_head(h) + "[" + ", ".join(input_form(a) for a in args) + "]"


def full_form(e) -> str:
    """Render the canonical expression tree, e.g. Plus[1, Times[2, x]]."""
    if e is None:
        return "Null"
    if isinstance(e, bool):
        return "True" if e else "False"
    if isinstance(e, Symbol):
        return e.name
    if isinstance(e, str):
        return json.dumps(e, ensure_ascii=False)
    if _is_real_number(e):
        return _number_text(e)
    if isinstance(e, complex):
        return f"Complex[{_number_text(e.real)}, {_number_text(e.imag)}]"
    if not isinstance(e, Expr):
        return str(e)
    q = rational_value(e)
    if q is not None and not is_int(e):
        return f"Rational[{q.numerator}, {q.denominator}]"
    head = e.head if isinstance(e.head, str) else full_form(e.head)
    return head + "[" + ", ".join(full_form(a) for a in e.args) + "]"
