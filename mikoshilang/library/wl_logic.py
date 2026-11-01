"""Coverage batch 6: Boolean computation (Wolfram semantics).

* Truth tables enumerate assignments in the order ``Tuples[{True, False}, n]`` (BooleanTable reference page);
  ``BooleanFunction[k, n]`` is the function whose table, read as 1/0 digits, is ``IntegerDigits[k, 2, 2^n]``.
* Truth tables are computed exactly with bit-parallel evaluation (one Python integer bit per row).
* Normal forms (BooleanConvert, BooleanFunction[k, vars], Conjunction, Resolve) are always equivalent to the input;
  where Wolfram's exact choice among equivalent forms is not documented a minimal two-level form is returned.
* Functions that decide a property (SatisfiableQ, TautologyQ, SatisfiabilityCount, SatisfiabilityInstances,
  Resolve) only accept purely Boolean expressions: a non-Boolean atom such as ``x > 0`` raises rather than being
  treated as an independent variable.
"""

from __future__ import annotations
import itertools

import sympy as sp
from sympy.logic import boolalg as BA

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import Unevaluated, fn, L, R, is_list, int_arg, nonneg_int, list_arg
from .wl_common import ev

AREA = "logic (batch 6)"
MAX_VARS = 22

_CONNECTIVES = ("And", "Or", "Not", "Xor", "Nand", "Nor", "Implies", "Equivalent", "Xnor", "Majority")


def _bf_head(e):
    """BooleanFunction[k, n][args] / BooleanCountingFunction[spec, n][args] -> (truth predicate over counts/rows)."""
    return isinstance(e, Expr) and isinstance(e.head, Expr) and e.head.head in ("BooleanFunction",
                                                                                 "BooleanCountingFunction")


class Conv:
    """MikoshiLang Boolean expression <-> SymPy Boolean with atoms mapped to fresh symbols."""

    def __init__(self, strict, fname, variables=None):
        self.strict, self.fname = strict, fname
        self.atoms = {}          # tag -> sympy symbol
        self.back = {}           # sympy symbol -> original atom
        self.order = []
        if variables is not None:
            for v in variables:
                self.atom(v)

    def atom(self, e):
        t = K.tag(e)
        s = self.atoms.get(t)
        if s is None:
            if self.strict and not boolean_variable_q(e):
                raise ValueError(f"{self.fname}: {e!r} is not a Boolean variable; only Boolean expressions of "
                                 "symbols are supported")
            s = sp.Symbol(f"_b{len(self.atoms)}")
            self.atoms[t] = s
            self.back[s] = e
            self.order.append(e)
        return s

    def to(self, e):
        if e is True:
            return sp.true
        if e is False:
            return sp.false
        if isinstance(e, Symbol) and e.name in ("True", "False"):
            return sp.true if e.name == "True" else sp.false
        if isinstance(e, Expr) and isinstance(e.head, str) and e.head in _CONNECTIVES:
            a = [self.to(x) for x in e.args]
            h = e.head
            if h == "And":
                return sp.And(*a)
            if h == "Or":
                return sp.Or(*a)
            if h == "Not":
                if len(a) != 1:
                    raise ValueError(f"{self.fname}: Not takes one argument")
                return sp.Not(a[0])
            if h == "Xor":
                return sp.Xor(*a)
            if h == "Xnor":
                return sp.Not(sp.Xor(*a))
            if h == "Nand":
                return sp.Nand(*a)
            if h == "Nor":
                return sp.Nor(*a)
            if h == "Implies":
                if len(a) != 2:
                    raise ValueError(f"{self.fname}: Implies takes two arguments")
                return sp.Implies(a[0], a[1])
            if h == "Equivalent":
                return sp.Equivalent(*a) if len(a) > 1 else sp.true
            if h == "Majority":
                if len(a) % 2 == 0:
                    raise ValueError(f"{self.fname}: Majority needs an odd number of arguments")
                need = len(a) // 2 + 1
                return sp.Or(*(sp.And(*c) for c in itertools.combinations(a, need)))
        if _bf_head(e):
            args = [self.to(x) for x in e.args]
            rows = _bf_rows(e.head, len(args), self.fname)
            n = len(args)
            terms = []
            for r in rows:
                vals = _row_values(r, n)
                terms.append(sp.And(*(x if v else sp.Not(x) for x, v in zip(args, vals))))
            return sp.Or(*terms)
        return self.atom(e)

    def back_to(self, v):
        if v is sp.true or v is True:
            return True
        if v is sp.false or v is False:
            return False
        if isinstance(v, sp.Symbol):
            return self.back[v]
        name = {BA.And: "And", BA.Or: "Or", BA.Not: "Not", BA.Xor: "Xor", BA.Implies: "Implies",
                BA.Equivalent: "Equivalent", BA.Nand: "Nand", BA.Nor: "Nor"}.get(type(v))
        if name is None:
            raise ValueError(f"{self.fname}: unexpected Boolean form {v}")
        args = list(v.args)
        if name in ("And", "Or", "Xor"):
            args.sort(key=self._sort_key)
        return Expr(name, *(self.back_to(a) for a in args))

    def _sort_key(self, a):
        syms = sorted(a.free_symbols, key=lambda s: int(s.name[2:]))
        idx = [int(s.name[2:]) for s in syms]
        return (len(a.args) if isinstance(a, (BA.And, BA.Or)) else 0, idx, str(a))

    def symbols(self):
        return [self.atoms[K.tag(e)] for e in self.order]


def boolean_variable_q(e):
    """A symbol or an indexed variable such as x[1] (not a number, relation or known mathematical function)."""
    if isinstance(e, Symbol):
        return e.name not in ("True", "False", "Pi", "E", "I", "Infinity")
    if isinstance(e, Expr) and isinstance(e.head, str):
        from .core import REGISTRY
        if e.head in K._BUILTINS or e.head in REGISTRY or e.head in K._HELD or e.head in K.STRUCTURAL_HEADS:
            return False
        return e.head not in ("Equal", "Unequal", "Less", "Greater", "LessEqual", "GreaterEqual", "Element",
                              "Inequality", "SameQ", "UnsameQ")
    return False


def _row_values(r, n):
    """Assignment for row r (0-based) of Tuples[{True, False}, n]."""
    return [((r >> (n - 1 - j)) & 1) == 0 for j in range(n)]


def _columns(n):
    """Bitmask column of each variable: bit r set when the variable is True in row r."""
    rows = 1 << n
    cols = []
    for j in range(n):
        block = 1 << (n - 1 - j)          # run length of equal values for variable j
        pattern = 0
        # rows with bit (n-1-j) == 0 are True
        unit = ((1 << block) - 1)
        period = block * 2
        for start in range(0, rows, period):
            pattern |= unit << start
        cols.append(pattern)
    return cols, (1 << rows) - 1


def _mask(v, env, full):
    if v is sp.true:
        return full
    if v is sp.false:
        return 0
    if isinstance(v, sp.Symbol):
        return env[v]
    a = [_mask(x, env, full) for x in v.args]
    t = type(v)
    if t is BA.And:
        out = full
        for x in a:
            out &= x
        return out
    if t is BA.Or:
        out = 0
        for x in a:
            out |= x
        return out
    if t is BA.Not:
        return full & ~a[0]
    if t is BA.Xor:
        out = 0
        for x in a:
            out ^= x
        return out
    if t is BA.Nand:
        out = full
        for x in a:
            out &= x
        return full & ~out
    if t is BA.Nor:
        out = 0
        for x in a:
            out |= x
        return full & ~out
    if t is BA.Implies:
        return (full & ~a[0]) | a[1]
    if t is BA.Equivalent:
        allt, allf = full, full
        for x in a:
            allt &= x
            allf &= full & ~x
        return allt | allf
    if t is BA.ITE:
        return (a[0] & a[1]) | ((full & ~a[0]) & a[2])
    raise ValueError(f"unexpected Boolean form {v}")


def truth_mask(sv, syms):
    n = len(syms)
    if n > MAX_VARS:
        raise ValueError(f"too many Boolean variables ({n}); at most {MAX_VARS} are supported")
    cols, full = _columns(n)
    env = dict(zip(syms, cols))
    extra = sv.free_symbols - set(syms)
    if extra:
        raise ValueError("the expression has variables that are not in the variable list")
    return _mask(sv, env, full), n


def _var_list(x, fname):
    if is_list(x):
        return list(x.args)
    if isinstance(x, Symbol) or isinstance(x, Expr):
        return [x]
    raise ValueError(f"{fname}: variables must be a list of symbols")


def _bf_rows(head, n, fname):
    """Rows (indices into Tuples[{True, False}, n]) where a pure Boolean function is True."""
    if head.head == "BooleanFunction":
        k, m = _bf_spec(head, fname)
        if m != n:
            raise ValueError(f"{fname}: BooleanFunction of {m} variables applied to {n} arguments")
        total = 1 << n
        return [r for r in range(total) if (k >> (total - 1 - r)) & 1]
    counts, m = _bcf_spec(head, fname)
    if m != n:
        raise ValueError(f"{fname}: BooleanCountingFunction of {m} variables applied to {n} arguments")
    return [r for r in range(1 << n) if sum(_row_values(r, n)) in counts]


def _bf_spec(head, fname):
    if len(head.args) != 2 or not K.is_int(head.args[0]) or not K.is_int(head.args[1]):
        raise Unevaluated()
    k, n = int(head.args[0]), int(head.args[1])
    if n < 0 or n > MAX_VARS:
        raise ValueError(f"{fname}: BooleanFunction needs 0 <= n <= {MAX_VARS} variables")
    if not 0 <= k < (1 << (1 << n)):
        raise ValueError(f"BooleanFunction: index {k} is outside 0 .. 2^(2^{n}) - 1")
    return k, n


def _bcf_counts(spec, n, fname):
    if K.is_int(spec):
        return set(range(0, min(int(spec), n) + 1))
    if is_list(spec):
        if len(spec.args) == 1 and is_list(spec.args[0]):
            lo, hi = [int_arg(v, fname, "count") for v in spec.args[0].args]
            return set(range(lo, hi + 1))
        return {int_arg(v, fname, "count") for v in spec.args}
    raise ValueError(f"{fname}: invalid count specification {spec!r}")


def _bcf_spec(head, fname):
    if len(head.args) != 2 or not K.is_int(head.args[1]):
        raise Unevaluated()
    n = int(head.args[1])
    return _bcf_counts(head.args[0], n, fname), n


def _apply_pure(head, args, ev_):
    """BooleanFunction[k, n][args] and BooleanCountingFunction[spec, n][args] for True/False arguments."""
    try:
        if not all(isinstance(a, bool) for a in args):
            return K.UNHANDLED
        rows = _bf_rows(head, len(args), head.head)
    except Unevaluated:
        return K.UNHANDLED
    n = len(args)
    r = sum((0 if a else 1) << (n - 1 - j) for j, a in enumerate(args))
    return r in set(rows)


from ..kernel_lists import COMPOUND_HEAD_HOOKS  # noqa: E402

COMPOUND_HEAD_HOOKS["BooleanFunction"] = _apply_pure
COMPOUND_HEAD_HOOKS["BooleanCountingFunction"] = _apply_pure


def _bool_expr_vars(e, fname, vars_=None, strict=True):
    c = Conv(strict, fname, vars_)
    sv = c.to(e)
    if vars_ is None:
        order = sorted(c.order, key=K.order_key)
        c2 = Conv(strict, fname, order)
        sv = c2.to(e)
        c = c2
    return c, sv


def _minimal(sv, form, syms=None):
    syms = syms if syms is not None else sorted(sv.free_symbols, key=lambda s: int(s.name[2:]))
    if sv in (sp.true, sp.false):
        return sv
    m, n = truth_mask(sv, syms)
    full = (1 << (1 << n)) - 1
    if m == 0:
        return sp.false
    if m == full:
        return sp.true
    minterms = [[1 if v else 0 for v in _row_values(r, n)] for r in range(1 << n) if (m >> r) & 1]
    if n <= 10:
        return sp.SOPform(syms, minterms) if form == "dnf" else sp.POSform(syms, minterms)
    return sp.to_dnf(sv, simplify=False) if form == "dnf" else sp.to_cnf(sv, simplify=False)


# ─────────────────────────────── simple connectives ───────────────────────────────

@fn("Boole", area=AREA, status="added", listable=True, notes="Boole[True] = 1, Boole[False] = 0; other arguments "
    "stay unevaluated")
def _boole(x: Boolean) -> Integer:
    if x is True:
        return 1
    if x is False:
        return 0
    raise Unevaluated()


@fn("Majority", area=AREA, status="added",
    notes="True when more than half of the (odd number of) arguments are True, False when more than half are False; "
          "symbolic arguments stay unevaluated unless the count is already decided; an even number of arguments "
          "raises")
def _majority(*args: Boolean) -> Boolean:
    if len(args) % 2 == 0:
        raise ValueError("Majority needs an odd number of arguments")
    t = sum(1 for a in args if a is True)
    f = sum(1 for a in args if a is False)
    need = len(args) // 2 + 1
    if t >= need:
        return True
    if f >= need:
        return False
    raise Unevaluated()


@fn("BooleanVariables", area=AREA, status="added",
    notes="the distinct variables (non-connective atoms) of a Boolean expression in canonical order")
def _boolvars(e: Expr) -> ListOf[Expr]:
    c = Conv(False, "BooleanVariables")
    c.to(e)
    return L(*sorted(c.order, key=K.order_key))


# ─────────────────────────────── BooleanFunction / BooleanCountingFunction ───────────────────────────────

def _truth_values(x, fname):
    vals = list_arg(x, fname, "truth table")
    out = []
    for v in vals:
        if v is True or v == 1:
            out.append(True)
        elif v is False or v == 0:
            out.append(False)
        else:
            raise ValueError(f"{fname}: truth table entries must be True/False or 1/0")
    n = len(out).bit_length() - 1
    if len(out) != 1 << n:
        raise ValueError(f"{fname}: a truth table needs 2^n entries")
    return out, n


def _expr_from_rows(rows, vars_, fname):
    c = Conv(False, fname, vars_)
    syms = c.symbols()
    n = len(syms)
    if not rows:
        return False
    if len(rows) == 1 << n:
        return True
    minterms = [[1 if v else 0 for v in _row_values(r, n)] for r in rows]
    if n <= 10:
        sv = sp.SOPform(syms, minterms)
    else:
        sv = sp.Or(*(sp.And(*(s if b else sp.Not(s) for s, b in zip(syms, mt))) for mt in minterms))
    return c.back_to(sv)


@fn("BooleanFunction", area=AREA, status="added",
    notes="BooleanFunction[k, n] pure function (truth table IntegerDigits[k, 2, 2^n] in Tuples[{True, False}, n] "
          "order; BooleanFunction[8, 2] is And) applied to True/False arguments; BooleanFunction[k, {vars}] and "
          "BooleanFunction[table, {vars}] give an equivalent minimal disjunctive normal form; "
          "BooleanFunction[table] gives BooleanFunction[k, n]")
def _boolfun(spec: Expr, n: Expr = None) -> Expr:
    if n is None:
        vals, m = _truth_values(spec, "BooleanFunction")
        k = 0
        for v in vals:
            k = 2 * k + (1 if v else 0)
        return Expr("BooleanFunction", k, m)
    if is_list(spec):
        vals, m = _truth_values(spec, "BooleanFunction")
        if not is_list(n):
            raise ValueError("BooleanFunction[table, vars] needs a list of variables")
        vars_ = list(n.args)
        if len(vars_) != m:
            raise ValueError("BooleanFunction: the truth table does not match the number of variables")
        return _expr_from_rows([r for r, v in enumerate(vals) if v], vars_, "BooleanFunction")
    if not K.is_int(spec):
        raise Unevaluated()
    if is_list(n):
        vars_ = list(n.args)
        rows = _bf_rows(Expr("BooleanFunction", spec, len(vars_)), len(vars_), "BooleanFunction")
        return _expr_from_rows(rows, vars_, "BooleanFunction")
    _bf_spec(Expr("BooleanFunction", spec, n), "BooleanFunction")
    raise Unevaluated()


@fn("BooleanCountingFunction", area=AREA, status="added",
    notes="BooleanCountingFunction[kmax, n] (at most kmax True), [{k}, n] (exactly k), [{k1, k2, ...}, n], "
          "[{{kmin, kmax}}, n] pure functions; with a variable list gives an equivalent minimal DNF")
def _bcf(spec: Expr, n: Expr) -> Expr:
    if is_list(n):
        vars_ = list(n.args)
        counts = _bcf_counts(spec, len(vars_), "BooleanCountingFunction")
        rows = [r for r in range(1 << len(vars_)) if sum(_row_values(r, len(vars_))) in counts]
        return _expr_from_rows(rows, vars_, "BooleanCountingFunction")
    if not K.is_int(n):
        raise Unevaluated()
    _bcf_counts(spec, int(n), "BooleanCountingFunction")
    raise Unevaluated()


def _pure_arity(bf, fname):
    if isinstance(bf, Expr) and bf.head == "BooleanFunction":
        return _bf_spec(bf, fname)[1]
    if isinstance(bf, Expr) and bf.head == "BooleanCountingFunction":
        return _bcf_spec(bf, fname)[1]
    if isinstance(bf, Expr) and bf.head == "Function" and len(bf.args) == 1:
        slots = [int(s.args[0]) for s in _walk(bf.args[0]) if isinstance(s, Expr) and s.head == "Slot"
                 and s.args and K.is_int(s.args[0])]
        if not slots:
            raise ValueError(f"{fname}: cannot determine the number of arguments of {bf!r}")
        return max(slots)
    return None


def _walk(e):
    yield e
    if isinstance(e, Expr):
        if isinstance(e.head, Expr):
            yield from _walk(e.head)
        for a in e.args:
            yield from _walk(a)


def _pure_to_expr(bf, fname):
    n = _pure_arity(bf, fname)
    vs = [Symbol(f"$bv{i}") for i in range(1, n + 1)]
    return ev(Expr(bf, *vs)), vs


# ─────────────────────────────── tables, counts, instances ───────────────────────────────

def _subst(e, assignment):
    rules = L(*(Expr("Rule", v, b) for v, b in assignment))
    return ev(Expr("ReplaceAll", e, rules))


def _table(e, groups):
    if not groups:
        return e
    vs, rest = groups[0], groups[1:]
    out = []
    for vals in itertools.product([True, False], repeat=len(vs)):
        out.append(_table(_subst(e, list(zip(vs, vals))), rest))
    return L(*out)


@fn("BooleanTable", area=AREA, status="added",
    notes="truth values over Tuples[{True, False}, n] (BooleanTable[expr] uses BooleanVariables); nested tables for "
          "several variable lists (outermost first); pure BooleanFunction / BooleanCountingFunction / Function; "
          "non-Boolean expressions are substituted (BooleanTable[f[a, b], {a, b}])")
def _booltable(e: Expr, *groups: Expr) -> Expr:
    if _pure_arity(e, "BooleanTable") is not None:
        expr, vs = _pure_to_expr(e, "BooleanTable")
        if not groups:
            return _table(expr, [vs])
        gs = []
        for g in groups:
            idx = [int_arg(i, "BooleanTable", "argument position") for i in list_arg(g, "BooleanTable")]
            gs.append([vs[i - 1] for i in idx])
        return _table(expr, gs)
    if not groups:
        c = Conv(False, "BooleanTable")
        c.to(e)
        return _table(e, [sorted(c.order, key=K.order_key)])
    return _table(e, [_var_list(g, "BooleanTable") for g in groups])


def _decide_setup(e, vars_, fname):
    if _pure_arity(e, fname) is not None:
        expr, vs = _pure_to_expr(e, fname)
        e, vars_ = expr, L(*vs)
    if vars_ is None:
        c, sv = _bool_expr_vars(e, fname, None, strict=True)
    else:
        vl = _var_list(vars_, fname)
        for v in vl:
            if not boolean_variable_q(v):
                raise ValueError(f"{fname}: {v!r} is not a variable")
        c = Conv(True, fname, vl)
        sv = c.to(e)
        if len(c.order) > len(vl):
            raise ValueError(f"{fname}: the expression contains variables that are not in the variable list")
    return c, sv


@fn("SatisfiableQ", area=AREA, status="added", notes="True if some assignment of the variables makes the "
    "expression True (exact truth table up to 22 variables, SymPy SAT solver beyond)")
def _satq(e: Expr, vars_: Var | ListOf[Expr] = None) -> Boolean:
    c, sv = _decide_setup(e, vars_, "SatisfiableQ")
    syms = c.symbols()
    if len(syms) > MAX_VARS:
        return sp.satisfiable(sv) is not False
    m, _ = truth_mask(sv, syms)
    return m != 0


@fn("TautologyQ", area=AREA, status="added", notes="True if every assignment of the variables makes the expression "
    "True")
def _tautq(e: Expr, vars_: Var | ListOf[Expr] = None) -> Boolean:
    c, sv = _decide_setup(e, vars_, "TautologyQ")
    syms = c.symbols()
    if len(syms) > MAX_VARS:
        return sp.satisfiable(sp.Not(sv)) is False
    m, n = truth_mask(sv, syms)
    return m == (1 << (1 << n)) - 1


@fn("SatisfiabilityCount", area=AREA, status="added", notes="number of assignments (of BooleanVariables or the "
    "given variables, so extra variables double the count) that make the expression True")
def _satcount(e: Expr, vars_: Var | ListOf[Expr] = None) -> Integer:
    c, sv = _decide_setup(e, vars_, "SatisfiabilityCount")
    m, _ = truth_mask(sv, c.symbols())
    return bin(m).count("1")


@fn("SatisfiabilityInstances", area=AREA, status="added",
    notes="SatisfiabilityInstances[expr, vars], [expr, vars, m], [expr, vars, All] and pure-function forms: lists of "
          "truth values that make expr True, enumerated with False before True (lexicographic); {} if unsatisfiable. "
          "Wolfram does not document which instances it returns first; every returned instance is valid")
def _satinst(e: Expr, vars_: Var | ListOf[Expr] = None, m: PosInteger = None) -> ListOf[Expr]:
    if _pure_arity(e, "SatisfiabilityInstances") is not None:
        if vars_ is not None and m is None:
            vars_, m = None, vars_
        c, sv = _decide_setup(e, None, "SatisfiabilityInstances")
    else:
        if vars_ is None:
            raise ValueError("SatisfiabilityInstances: give the list of variables, SatisfiabilityInstances[expr, vars]")
        c, sv = _decide_setup(e, vars_, "SatisfiabilityInstances")
    syms = c.symbols()
    if m is None:
        count = 1
    elif isinstance(m, Symbol) and m.name == "All":
        count = None
    else:
        count = nonneg_int(m, "SatisfiabilityInstances", "number of instances")
    mask, n = truth_mask(sv, syms)
    out = []
    for r in range((1 << n) - 1, -1, -1):
        if (mask >> r) & 1:
            out.append(L(*_row_values(r, n)))
            if count is not None and len(out) >= count:
                break
    return L(*out)


# ─────────────────────────────── normal forms ───────────────────────────────

def _anf_terms(sv, syms):
    m, n = truth_mask(sv, syms)
    size = 1 << n
    f = [0] * size                     # index: bit j set <=> variable j True
    for idx in range(size):
        r = (size - 1) - sum(1 << (n - 1 - j) for j in range(n) if (idx >> j) & 1)
        f[idx] = (m >> r) & 1
    for j in range(n):                 # Moebius transform over GF(2)
        bit = 1 << j
        for idx in range(size):
            if idx & bit:
                f[idx] ^= f[idx ^ bit]
    terms = [idx for idx in range(size) if f[idx]]
    terms.sort(key=lambda idx: (bin(idx).count("1"), [j for j in range(n) if (idx >> j) & 1]))
    return terms


def _anf(c, sv):
    syms = c.symbols()
    terms = _anf_terms(sv, syms)
    if not terms:
        return False
    parts = []
    for idx in terms:
        vs = [c.order[j] for j in range(len(syms)) if (idx >> j) & 1]
        parts.append(True if not vs else (vs[0] if len(vs) == 1 else Expr("And", *vs)))
    return parts[0] if len(parts) == 1 else Expr("Xor", *parts)


def _literal_parts(c, sv, kind):
    """Clauses of a DNF (kind 'dnf': products) or CNF (sums) as lists of MikoshiLang literals."""
    outer, inner = (BA.Or, BA.And) if kind == "dnf" else (BA.And, BA.Or)
    clauses = list(sv.args) if isinstance(sv, outer) else [sv]
    out = []
    for cl in clauses:
        lits = list(cl.args) if isinstance(cl, inner) else [cl]
        out.append([c.back_to(l) for l in lits])
    return out


def _two_level(c, sv, form):
    if form in ("NAND", "AND"):
        kind = "dnf"
    else:
        kind = "cnf"
    mn = _minimal(sv, kind, c.symbols())
    if mn in (sp.true, sp.false):
        return mn is sp.true
    clauses = _literal_parts(c, mn, kind)
    if len(clauses) == 1 and len(clauses[0]) == 1:
        return clauses[0][0]
    if form == "NAND":      # Or(t1..) = Nand(Nand(lits t1), ...)
        return Expr("Nand", *(Expr("Nand", *cl) if len(cl) > 1 else Expr("Not", cl[0]) for cl in clauses))
    if form == "NOR":       # And(c1..) = Nor(Nor(lits c1), ...)
        return Expr("Nor", *(Expr("Nor", *cl) if len(cl) > 1 else Expr("Not", cl[0]) for cl in clauses))
    if form == "AND":       # Or(t1..) = Not(And(Not t1, ...)), t = And(lits)
        return Expr("Not", Expr("And", *(Expr("Not", Expr("And", *cl) if len(cl) > 1 else cl[0]) for cl in clauses)))
    # OR: And(c1..) = Not(Or(Not c1, ...))
    return Expr("Not", Expr("Or", *(Expr("Not", Expr("Or", *cl) if len(cl) > 1 else cl[0]) for cl in clauses)))


_FORMS = {"DNF": "DNF", "SOP": "DNF", "CNF": "CNF", "POS": "CNF", "ANF": "ANF", "ESOP": "ANF", "NAND": "NAND",
          "NOR": "NOR", "AND": "AND", "OR": "OR"}


@fn("BooleanConvert", area=AREA, status="added",
    notes="forms \"DNF\"/\"SOP\" (default), \"CNF\"/\"POS\" (minimal two-level forms), \"ANF\" (unique algebraic "
          "normal form, Xor of Ands), \"ESOP\" (given as the ANF, which is an exclusive sum of products), \"NAND\", "
          "\"NOR\", \"AND\", \"OR\" (two-level with Not); BooleanConvert[expr, form, cond] uses cond as a care set "
          "for DNF/CNF; other forms (IMPLIES, ITE, BFF, BDT, binary NAND/NOR) raise")
def _bconvert(e: Expr, form: Expr = None, cond: Expr = None) -> Expr:
    if isinstance(form, Expr) and not isinstance(form, str) and cond is None and not isinstance(form, Symbol):
        form, cond = None, form
    fname = "BooleanConvert"
    f = "DNF" if form is None else form
    if not isinstance(f, str):
        if isinstance(f, Symbol):
            cond, f = f, "DNF"
        else:
            raise ValueError(f"{fname}: form must be a string such as \"DNF\"")
    if f not in _FORMS:
        raise NotImplementedError(f"{fname}: form {f!r} is not supported (supported: DNF, SOP, CNF, POS, ANF, ESOP, "
                                  "NAND, NOR, AND, OR)")
    f = _FORMS[f]
    if _pure_arity(e, fname) is not None:
        e, _ = _pure_to_expr(e, fname)
    c = Conv(False, fname)
    sv = c.to(e)
    if cond is not None:
        sc = c.to(cond)
        if f not in ("DNF", "CNF"):
            raise NotImplementedError(f"{fname}: a condition is only supported for DNF and CNF")
        order = sorted(c.order, key=K.order_key)
        c2 = Conv(False, fname, order)
        sv, sc = c2.to(e), c2.to(cond)
        syms = c2.symbols()
        m, n = truth_mask(sv, syms)
        mc, _ = truth_mask(sc, syms)
        if n > 10:
            raise ValueError(f"{fname}: conditional conversion supports at most 10 variables")
        on = [[1 if v else 0 for v in _row_values(r, n)] for r in range(1 << n) if (m >> r) & 1 and (mc >> r) & 1]
        dc = [[1 if v else 0 for v in _row_values(r, n)] for r in range(1 << n) if not (mc >> r) & 1]
        if not on:
            return False
        out = sp.SOPform(syms, on, dc) if f == "DNF" else sp.POSform(syms, on, dc)
        return c2.back_to(out)
    order = sorted(c.order, key=K.order_key)
    c = Conv(False, fname, order)
    sv = c.to(e)
    if f == "ANF":
        return _anf(c, sv)
    if f in ("DNF", "CNF"):
        return c.back_to(_minimal(sv, f.lower(), c.symbols()))
    return _two_level(c, sv, f)


def _clean_dnf(sv):
    """Drop contradictory products and absorbed products (a || a && b -> a) without changing the meaning."""
    terms = list(sv.args) if isinstance(sv, BA.Or) else [sv]
    sets = []
    for t in terms:
        lits = frozenset(t.args) if isinstance(t, BA.And) else frozenset([t])
        if any(sp.Not(l) in lits for l in lits if not isinstance(l, BA.Not)):
            continue
        if lits == frozenset([sp.true]):
            return sp.true
        sets.append(lits)
    kept = []
    for s in sets:
        if any(o < s for o in sets) or s in kept:
            continue
        kept.append(s)
    if not kept:
        return sp.false
    return sp.Or(*(sp.And(*s) for s in kept))


def _expand_lists(e):
    """{a, b} == {c, d} -> a == c && b == d; Unequal of lists -> Or of Unequal (LogicalExpand)."""
    if isinstance(e, Expr):
        if isinstance(e.head, str) and e.head in ("Equal", "Unequal") and len(e.args) == 2 and \
                all(is_list(a) for a in e.args) and len(e.args[0].args) == len(e.args[1].args):
            parts = [_expand_lists(Expr(e.head, x, y)) for x, y in zip(e.args[0].args, e.args[1].args)]
            return Expr("And" if e.head == "Equal" else "Or", *parts)
        if isinstance(e.head, str) and e.head in _CONNECTIVES:
            return Expr(e.head, *(_expand_lists(a) for a in e.args))
    return e


@fn("LogicalExpand", area=AREA, status="added",
    notes="expands Boolean connectives into a disjunction of conjunctions (distributing And over Or, pushing Not "
          "inwards, expanding Xor/Implies/Equivalent), removing contradictory and absorbed terms; equations between "
          "lists are split elementwise; non-Boolean atoms (equations, inequalities) are kept as they are "
          "(LogicalExpand[x y == 0] is not split into cases)")
def _lexpand(e: Expr) -> Expr:
    e2 = _expand_lists(e)
    c = Conv(False, "LogicalExpand")
    sv = c.to(e2)
    out = _clean_dnf(sp.to_dnf(sv, simplify=False))
    return ev(c.back_to(out))


# ─────────────────────────────── conditional expressions and quantifiers ───────────────────────────────

# ConditionalExpression is implemented once, in wl_calculus.py (it is also returned by Integrate, Sum and
# Solve); that implementation includes the nested-condition rule ConditionalExpression[ConditionalExpression[x, a], b]
# = ConditionalExpression[x, a && b] that batch 6 added here.


def _qvars(x, fname):
    vs = list(x.args) if is_list(x) else [x]
    for v in vs:
        if not isinstance(v, Symbol):
            raise ValueError(f"{fname}: {v!r} is not a valid quantified variable")
    return vs


def _free_of(e, vs):
    names = {v.name for v in vs}
    return not any(isinstance(s, Symbol) and s.name in names for s in _walk(e))


def _quant(name):
    @fn(name, area=AREA, status="added",
        notes=f"{name}[x, expr] and {name}[x, cond, expr] (x a symbol or list of symbols); evaluates when the body does "
              f"not depend on the variables (e.g. {name}[x, True] = True), otherwise stays for Resolve")
    def impl(x: Var | ListOf[Var], a: Expr, b: Expr = None) -> Expr:
        vs = _qvars(x, name)
        body = a if b is None else Expr("Implies" if name == "ForAll" else "And", a, b)
        if _free_of(body, vs):
            return ev(body)
        raise Unevaluated()
    return impl


_quant("ForAll")
_quant("Exists")


def _eliminate(e, fname):
    """Expand ForAll/Exists over Boolean variables (Shannon expansion); returns a quantifier-free expression."""
    if isinstance(e, Expr) and isinstance(e.head, str) and e.head in ("ForAll", "Exists") and len(e.args) in (2, 3):
        vs = _qvars(e.args[0], fname)
        if len(e.args) == 2:
            body = e.args[1]
        else:
            body = Expr("Implies" if e.head == "ForAll" else "And", e.args[1], e.args[2])
        body = _eliminate(body, fname)
        for v in vs:
            t = _subst(body, [(v, True)])
            f = _subst(body, [(v, False)])
            body = ev(Expr("And" if e.head == "ForAll" else "Or", t, f))
        return body
    if isinstance(e, Expr) and isinstance(e.head, str) and e.head in _CONNECTIVES:
        return ev(Expr(e.head, *(_eliminate(a, fname) for a in e.args)))
    if isinstance(e, Expr) and isinstance(e.head, str) and _contains_quantifier(e):
        raise NotImplementedError(f"{fname}: quantifiers are only supported inside Boolean connectives")
    return e


def _contains_quantifier(e):
    return any(isinstance(s, Expr) and s.head in ("ForAll", "Exists") for s in _walk(e))


@fn("Resolve", area=AREA, status="added",
    notes="eliminates ForAll/Exists from purely Boolean formulas (variables over Booleans) and returns an equivalent "
          "minimal DNF; quantifiers over non-Boolean formulas (real/complex/integer quantifier elimination) raise; "
          "quantifier-free input is returned unchanged")
def _resolve(e: Expr, dom: Expr = None) -> Expr:
    if dom is not None and not (isinstance(dom, Symbol) and dom.name == "Booleans"):
        if _contains_quantifier(e):
            raise NotImplementedError(f"Resolve: quantifier elimination over {dom!r} is not supported (only Booleans)")
    if not _contains_quantifier(e):
        return e
    c = Conv(False, "Resolve")
    try:
        c.to(_strip(e))
    except ValueError:
        pass
    if any(not isinstance(a, Symbol) for a in c.order):
        raise NotImplementedError("Resolve: quantifier elimination is only supported for purely Boolean formulas")
    out = _eliminate(e, "Resolve")
    c, sv = _bool_expr_vars(out, "Resolve", None, strict=True)
    return c.back_to(_minimal(sv, "dnf", c.symbols()))


def _strip(e):
    """Replace quantifiers by their bodies (to inspect atoms)."""
    if isinstance(e, Expr) and isinstance(e.head, str) and e.head in ("ForAll", "Exists") and len(e.args) in (2, 3):
        return Expr("And", *(_strip(a) for a in e.args[1:]))
    if isinstance(e, Expr) and isinstance(e.head, str):
        return Expr(e.head, *(_strip(a) for a in e.args))
    return e


def _junction(name, op):
    @fn(name, area=AREA, status="added",
        notes=f"{name}[expr, {{a1, ...}}]: {op} of expr over all True/False values of the a_i, as an equivalent "
              "minimal DNF")
    def impl(e: Expr, vars_: Var | ListOf[Var]) -> Expr:
        vs = _var_list(vars_, name)
        body = e
        for v in vs:
            if not isinstance(v, Symbol):
                raise ValueError(f"{name}: {v!r} is not a variable")
            body = ev(Expr(op, _subst(body, [(v, True)]), _subst(body, [(v, False)])))
        c, sv = _bool_expr_vars(body, name, None, strict=False)
        return c.back_to(_minimal(sv, "dnf", c.symbols()))
    return impl


_junction("Conjunction", "And")
_junction("Disjunction", "Or")


# ─────────────────────────────── corrections: partial evaluation of connectives ───────────────────────────────
# Wolfram simplifies connectives with some True/False arguments (Xor[True, c] = !c, Implies[True, b] = b,
# Nand[True, x] = !x, Equivalent[True, a, b] = a && b); the legacy connectives only evaluated when every argument
# was True/False.

def _not(x):
    if isinstance(x, Expr) and x.head == "Not" and len(x.args) == 1:
        return x.args[0]
    return ev(Expr("Not", x))


def _xor_partial(*args):
    if all(isinstance(a, bool) for a in args):
        return sum(args) % 2 == 1
    rest = [a for a in args if not isinstance(a, bool)]
    flip = sum(1 for a in args if a is True) % 2 == 1
    if len(rest) == len(args):
        raise Unevaluated()
    core_ = rest[0] if len(rest) == 1 else Expr("Xor", *rest)
    return _not(core_) if flip else core_


def _xnor_partial(*args):
    return _not(_xor_partial(*args)) if len(args) else True


def _nand_partial(*args):
    if any(a is False for a in args):
        return True
    rest = [a for a in args if a is not True]
    if not rest:
        return False
    if len(rest) == len(args):
        raise Unevaluated()
    return _not(rest[0]) if len(rest) == 1 else Expr("Nand", *rest)


def _nor_partial(*args):
    if any(a is True for a in args):
        return False
    rest = [a for a in args if a is not False]
    if not rest:
        return True
    if len(rest) == len(args):
        raise Unevaluated()
    return _not(rest[0]) if len(rest) == 1 else Expr("Nor", *rest)


def _implies_partial(p, q):
    if p is False or q is True:
        return True
    if p is True:
        return q
    if q is False:
        return _not(p)
    raise Unevaluated()


def _equiv_partial(*args):
    if all(isinstance(a, bool) for a in args):
        return all(x == args[0] for x in args)
    consts = {a for a in args if isinstance(a, bool)}
    rest = [a for a in args if not isinstance(a, bool)]
    if not consts:
        raise Unevaluated()
    if consts == {True, False}:
        return False
    if True in consts:
        return rest[0] if len(rest) == 1 else Expr("And", *rest)
    return _not(rest[0]) if len(rest) == 1 else Expr("And", *(Expr("Not", r) for r in rest))


def _install(name, impl, min_args, notes):
    from .core import REGISTRY, record

    def f(*args: Boolean) -> Boolean:
        if len(args) < min_args:
            raise ValueError(f"{name} needs at least {min_args} arguments")
        return impl(*args)
    f.__name__ = f"_{name}_b6"
    e = REGISTRY[name]
    e.impls = [(f, 0, None)]
    record(name, e.area, "fixed", notes)


_install("Xor", _xor_partial, 2, "variadic exclusive or; True/False arguments are simplified away "
         "(Xor[True, c] = !c, Xor[False, c] = c; batch 6)")
_install("Xnor", _xnor_partial, 2, "Not[Xor[...]] with the same partial evaluation as Xor (batch 6)")
_install("Nand", _nand_partial, 2, "variadic Not[And[...]]; Nand[False, x] = True, Nand[True, x] = !x (batch 6)")
_install("Nor", _nor_partial, 2, "variadic Not[Or[...]]; Nor[True, x] = False, Nor[False, x] = !x (batch 6)")
_install("Implies", _implies_partial, 2, "Implies[p, q]; Implies[True, q] = q, Implies[False, q] = True, "
         "Implies[p, True] = True, Implies[p, False] = !p (batch 6)")
_install("Equivalent", _equiv_partial, 2, "variadic; True when all arguments are equal booleans; "
         "Equivalent[True, a, b] = a && b, Equivalent[False, a] = !a, mixed True/False = False (batch 6)")
