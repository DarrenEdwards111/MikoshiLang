"""LaTeX rendering for MikoshiLang expressions."""

from __future__ import annotations
from .expr import Expr, Symbol


_GREEK = {
    "Alpha": r"\alpha", "Beta": r"\beta", "Gamma": r"\gamma", "Delta": r"\delta",
    "Epsilon": r"\epsilon", "Zeta": r"\zeta", "Eta": r"\eta", "Theta": r"\theta",
    "Iota": r"\iota", "Kappa": r"\kappa", "Lambda": r"\lambda", "Mu": r"\mu",
    "Nu": r"\nu", "Xi": r"\xi", "Pi": r"\pi", "Rho": r"\rho",
    "Sigma": r"\sigma", "Tau": r"\tau", "Upsilon": r"\upsilon", "Phi": r"\phi",
    "Chi": r"\chi", "Psi": r"\psi", "Omega": r"\omega",
    "alpha": r"\alpha", "beta": r"\beta", "gamma": r"\gamma", "delta": r"\delta",
    "epsilon": r"\epsilon", "zeta": r"\zeta", "eta": r"\eta", "theta": r"\theta",
    "pi": r"\pi", "sigma": r"\sigma", "tau": r"\tau", "phi": r"\phi",
    "omega": r"\omega",
}

_CONSTANTS = {
    "Pi": r"\pi", "E": r"e", "I": r"i", "Infinity": r"\infty",
    "GoldenRatio": r"\varphi",
}

_TRIG = {
    "Sin": r"\sin", "Cos": r"\cos", "Tan": r"\tan",
    "ArcSin": r"\arcsin", "ArcCos": r"\arccos", "ArcTan": r"\arctan",
    "Sinh": r"\sinh", "Cosh": r"\cosh", "Tanh": r"\tanh",
}


def to_latex(expr, parent_prec: int = 0) -> str:
    """Convert a MikoshiLang expression to a LaTeX string."""
    if isinstance(expr, bool):
        return r"\text{True}" if expr else r"\text{False}"
    if isinstance(expr, int):
        return str(expr)
    if isinstance(expr, float):
        return f"{expr:g}"
    if isinstance(expr, str):
        return r'\text{"' + expr + r'"}'
    if isinstance(expr, Symbol):
        name = expr.name
        if name in _CONSTANTS:
            return _CONSTANTS[name]
        if name in _GREEK:
            return _GREEK[name]
        if len(name) == 1:
            return name
        return r"\text{" + name + "}"
    if not isinstance(expr, Expr):
        return str(expr)

    head = expr.head
    args = expr.args

    if head == "Plus":
        parts = []
        for i, a in enumerate(args):
            is_neg_times = (isinstance(a, Expr) and a.head == "Times" and len(a.args) >= 1
                           and isinstance(a.args[0], (int, float)) and a.args[0] < 0)
            is_neg_atom = isinstance(a, (int, float)) and a < 0
            if i == 0:
                parts.append(to_latex(a, 1))
            else:
                if is_neg_atom:
                    parts.append(f" - {abs(a)}")
                elif is_neg_times:
                    coeff = a.args[0]
                    rest_args = a.args[1:]
                    if coeff == -1 and len(rest_args) == 1:
                        parts.append(f" - {to_latex(rest_args[0], 1)}")
                    else:
                        abs_coeff = abs(coeff)
                        if abs_coeff == 1 and len(rest_args) == 1:
                            pos_expr = rest_args[0]
                        else:
                            pos_expr = Expr("Times", abs_coeff, *rest_args)
                        parts.append(f" - {to_latex(pos_expr, 1)}")
                else:
                    parts.append(f" + {to_latex(a, 1)}")
        result = "".join(parts)
        if parent_prec > 1:
            result = r"\left(" + result + r"\right)"
        return result

    if head == "Times":
        # Check for negative coefficient
        if len(args) == 2 and args[0] == -1:
            inner = to_latex(args[1], 2)
            result = f"-{inner}"
            if parent_prec > 1:
                result = r"\left(" + result + r"\right)"
            return result
        # Check for fraction: a * b^-1
        num_parts = []
        den_parts = []
        for a in args:
            if isinstance(a, Expr) and a.head == "Power" and isinstance(a.args[1], int) and a.args[1] == -1:
                den_parts.append(a.args[0])
            elif isinstance(a, Expr) and a.head == "Power" and isinstance(a.args[1], (int, float)) and a.args[1] < 0:
                den_parts.append(Expr("Power", a.args[0], -a.args[1]))
            else:
                num_parts.append(a)
        if den_parts:
            num = _latex_product(num_parts) if num_parts else "1"
            den = _latex_product(den_parts)
            return r"\frac{" + num + "}{" + den + "}"
        return _latex_product(args)

    if head == "Power":
        base = args[0]
        exp = args[1]
        if isinstance(exp, Expr) and exp.head == "Times" and len(exp.args) == 2 and exp.args[0] == 1 and isinstance(exp.args[1], Expr) and exp.args[1].head == "Power" and exp.args[1].args[1] == -1:
            # x^(1/n) -> \sqrt[n]{x}
            n = exp.args[1].args[0]
            if n == 2:
                return r"\sqrt{" + to_latex(base) + "}"
            return r"\sqrt[" + to_latex(n) + "]{" + to_latex(base) + "}"
        if exp == -1:
            return r"\frac{1}{" + to_latex(base) + "}"
        if isinstance(exp, (int, float)) and exp == 0.5:
            return r"\sqrt{" + to_latex(base) + "}"
        base_s = to_latex(base, 4)
        exp_s = to_latex(exp)
        if _needs_braces(exp):
            return f"{base_s}^{{{exp_s}}}"
        return f"{base_s}^{{{exp_s}}}"

    if head in _TRIG:
        fn = _TRIG[head]
        arg = to_latex(args[0]) if args else ""
        return f"{fn}\\left({arg}\\right)"

    if head == "Exp":
        return r"e^{" + to_latex(args[0]) + "}"

    if head == "Log":
        if len(args) == 1:
            return r"\ln\left(" + to_latex(args[0]) + r"\right)"
        return r"\log_{" + to_latex(args[0]) + r"}\left(" + to_latex(args[1]) + r"\right)"

    if head == "Sqrt":
        return r"\sqrt{" + to_latex(args[0]) + "}"

    if head == "Abs":
        return r"\left|" + to_latex(args[0]) + r"\right|"

    if head == "Factorial":
        return to_latex(args[0], 4) + "!"

    if head == "List":
        inner = ", ".join(to_latex(a) for a in args)
        return r"\{" + inner + r"\}"

    if head == "Matrix":
        rows = []
        for row in args:
            if isinstance(row, Expr) and row.head == "List":
                rows.append(" & ".join(to_latex(x) for x in row.args))
            else:
                rows.append(to_latex(row))
        inner = r" \\ ".join(rows)
        return r"\begin{pmatrix} " + inner + r" \end{pmatrix}"

    if head == "Equal":
        return to_latex(args[0]) + " = " + to_latex(args[1])

    if head == "Unequal":
        return to_latex(args[0]) + r" \neq " + to_latex(args[1])

    if head == "Less":
        return to_latex(args[0]) + " < " + to_latex(args[1])

    if head == "Greater":
        return to_latex(args[0]) + " > " + to_latex(args[1])

    if head == "LessEqual":
        return to_latex(args[0]) + r" \leq " + to_latex(args[1])

    if head == "GreaterEqual":
        return to_latex(args[0]) + r" \geq " + to_latex(args[1])

    if head == "And":
        parts = [to_latex(a) for a in args]
        return r" \land ".join(parts)

    if head == "Or":
        parts = [to_latex(a) for a in args]
        return r" \lor ".join(parts)

    if head == "Not":
        return r"\lnot " + to_latex(args[0], 4)

    if head == "Integrate":
        if len(args) >= 2:
            return r"\int " + to_latex(args[0]) + r"\, d" + to_latex(args[1])
        return r"\int " + to_latex(args[0])

    if head == "Diff":
        if len(args) >= 2:
            return r"\frac{d}{d" + to_latex(args[1]) + "}" + to_latex(args[0])
        return r"\frac{d}{dx}" + to_latex(args[0])

    if head == "Limit":
        if len(args) >= 3:
            return r"\lim_{" + to_latex(args[1]) + r" \to " + to_latex(args[2]) + "} " + to_latex(args[0])
        return r"\lim " + to_latex(args[0])

    if head == "Sum":
        if len(args) >= 4:
            return r"\sum_{" + to_latex(args[1]) + "=" + to_latex(args[2]) + "}^{" + to_latex(args[3]) + "} " + to_latex(args[0])
        return r"\sum " + to_latex(args[0])

    if head == "Rule":
        return to_latex(args[0]) + r" \to " + to_latex(args[1])

    # Generic function
    inner = ", ".join(to_latex(a) for a in args)
    return r"\text{" + head + r"}\left(" + inner + r"\right)"


def _latex_product(parts) -> str:
    """Format a product of parts for LaTeX."""
    if not parts:
        return "1"
    strs = []
    for p in parts:
        s = to_latex(p, 2)
        strs.append(s)
    return " ".join(strs)


def _needs_braces(expr) -> bool:
    """Check if expression needs braces in superscript."""
    if isinstance(expr, (int, float)):
        s = str(expr)
        return len(s) > 1 or expr < 0
    return True
