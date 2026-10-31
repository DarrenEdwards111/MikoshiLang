"""Wolfram batch W12, part 3: Dataset and Query (with the FailureAction, MissingBehavior and PartBehavior options).

Query[op1, op2, ...] applies op_k at level k of the data.  Part specifications and filtering operators (Select,
SelectFirst, KeySelect, MaximalBy, MinimalBy, SortBy, KeySortBy, DeleteDuplicatesBy, DeleteMissing, GroupBy) are
"descending": they act before the deeper operators; everything else (aggregations, functions, subqueries, lists and
associations of operators) is "ascending": it acts on the result of the deeper operators.  Dataset[data] is an inert
wrapper whose calls ds[ops...] run Query[ops...] on the data and re-wrap structured results.
"""
from __future__ import annotations

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L
from .wl_dates_tabular import (ADDED, ev, is_sym, is_rule, is_assoc, is_list, assoc, wrap_builtin, wrap_registry,
                               _PASS)
from . import wl_tabular as WT

NONE = Symbol("None")
AUTOMATIC = Symbol("Automatic")


def apply(f, *args):
    return ev(K.call(f, *args))


def missing(*a):
    return Expr("Missing", *a)


def is_missing(x):
    return isinstance(x, Expr) and x.head == "Missing"


class QueryFailure(Exception):
    def __init__(self, failure):
        super().__init__("query failure")
        self.failure = failure


class _Drop:
    pass


DROP = _Drop()


class Ctx:
    def __init__(self, failure_action="Abort", missing_behavior=AUTOMATIC, part_behavior=AUTOMATIC):
        fa = failure_action
        self.fa_fn = None
        if is_list(fa) and len(fa.args) == 2:
            fa, self.fa_fn = fa.args
        if not (is_sym(fa, "None") or fa in ("Abort", "Drop", "Encapsulate", "Replace")):
            raise ValueError("Query: FailureAction must be None, \"Abort\", \"Drop\", \"Encapsulate\", \"Replace\" or "
                             "{action, f}")
        self.fa = "None" if is_sym(fa, "None") else fa
        if not (is_sym(missing_behavior, "None") or is_sym(missing_behavior, "Automatic")):
            raise ValueError("Query: MissingBehavior must be None or Automatic")
        if not (is_sym(part_behavior, "None") or is_sym(part_behavior, "Automatic")):
            raise ValueError("Query: PartBehavior must be None or Automatic")
        self.mb = is_sym(missing_behavior, "Automatic")
        self.pb = is_sym(part_behavior, "Automatic")


_QOPTS = ("FailureAction", "MissingBehavior", "PartBehavior")


def split_query(args):
    ops, opts = [], {}
    for a in args:
        if is_rule(a) and isinstance(a.args[0], Symbol) and a.args[0].name in _QOPTS:
            opts[a.args[0].name] = a.args[1]
        else:
            ops.append(a)
    return ops, Ctx(opts.get("FailureAction", "Abort"), opts.get("MissingBehavior", AUTOMATIC),
                    opts.get("PartBehavior", AUTOMATIC))


# ─────────────────────────── failure detection ───────────────────────────

def _contains_infinity(x):
    if isinstance(x, Symbol):
        return x.name in ("ComplexInfinity", "Indeterminate")
    if isinstance(x, Expr):
        return any(_contains_infinity(a) for a in x.args) or (isinstance(x.head, Expr) and
                                                                _contains_infinity(x.head))
    return False


def _failure(tag, template, params):
    return Expr("Failure", tag, assoc(("MessageTemplate", template), ("MessageParameters", L(*params))))


def guarded(ctx, f, *inputs):
    """Run f() under the FailureAction policy.  Messages, raised errors and new ComplexInfinity/Indeterminate
    values (where Wolfram issues Power::infy / Infinity::indet) count as failures."""
    from .wl_errors import _Frame, _Collect, is_error, error_key
    if ctx.fa == "None":
        with _Frame(_Collect()):
            return f()
    with _Frame(_Collect()) as col:
        try:
            r = f()
            fail = None
            if col.messages:
                key, call = col.messages[0]
                from .wl_errors import message_text
                txt = message_text(key[0], key[1]) or "-- Message text not found --"
                msg = call.args[0] if isinstance(call, Expr) and call.args else call
                fail = _failure(key[0], txt, list(msg.args[1:]) if isinstance(msg, Expr) else [])
            elif _contains_infinity(r) and not any(_contains_infinity(i) for i in inputs):
                tag = "Infinity" if "Indeterminate" in repr(r) else "Power"
                fail = _failure(tag, "Indeterminate expression encountered." if tag == "Infinity"
                                else "Infinite expression encountered.", [])
        except Exception as exc:  # noqa: BLE001 - only MikoshiLang errors are handled
            if not is_error(exc) or isinstance(exc, QueryFailure):
                raise
            key = error_key(exc)
            fail = _failure(key[0], str(exc), [])
    if fail is None:
        return r
    if ctx.fa_fn is not None:
        fv = apply(ctx.fa_fn, fail)
        if ctx.fa == "Replace":
            return fv
    if ctx.fa == "Abort":
        raise QueryFailure(fail)
    if ctx.fa == "Encapsulate":
        return fail
    if ctx.fa == "Replace":
        return missing("Failed")
    return DROP


def _clean(x):
    """Remove dropped results from lists and associations."""
    if isinstance(x, _Drop):
        return x
    return x


# ─────────────────────────── operator classification ───────────────────────────

_DESC_FILTERS = {"Select", "SelectFirst", "KeySelect", "MaximalBy", "MinimalBy", "SortBy", "KeySortBy",
                 "DeleteDuplicatesBy", "GroupBy", "TakeLargestBy", "TakeSmallestBy", "TakeLargest", "TakeSmallest"}
_STR_AS_KEY = {"GroupBy", "SortBy", "CountsBy", "MaximalBy", "MinimalBy", "DeleteDuplicatesBy", "TakeLargestBy",
               "TakeSmallestBy", "CountDistinctBy"}


def _is_part_spec(op):
    if is_sym(op, "All") or isinstance(op, str):
        return True
    if K.is_int(op) and not isinstance(op, bool):
        return True
    if isinstance(op, Expr) and op.head in ("Key", "Span"):
        return True
    return False


def _is_part_list(op):
    return is_list(op) and all(_is_part_spec(x) and not is_sym(x, "All") and not
                               (isinstance(x, Expr) and x.head == "Span") for x in op.args)


def _keyify(op):
    if isinstance(op, Expr) and op.head in _STR_AS_KEY and op.args and isinstance(op.args[0], str):
        return Expr(op.head, Expr("Key", op.args[0]), *op.args[1:])
    return op


def _elements(x):
    if is_list(x):
        return list(x.args)
    if is_assoc(x):
        return [r.args[1] for r in x.args]
    return None


def _map_rest(rest, x, ctx):
    """Apply the deeper operators to each element of x."""
    if not rest:
        return x
    if is_missing(x):
        return x
    if is_list(x):
        out = [run(rest, e, ctx) for e in x.args]
        return L(*(v for v in out if not isinstance(v, _Drop)))
    if is_assoc(x):
        out = [(r.args[0], run(rest, r.args[1], ctx)) for r in x.args]
        return Expr("Association", *(Expr(r.head if False else "Rule", k, v) for k, v in out
                                     if not isinstance(v, _Drop)))
    return guarded(ctx, lambda: _raise(ValueError(f"Query: the data {x!r} has no parts for the operators "
                                                  f"{L(*rest)!r}")), x)


def _raise(e):
    raise e


def _part(x, op, ctx):
    """One part of x (descending part operator with PartBehavior)."""
    if is_missing(x):
        return x
    if isinstance(op, str) or (isinstance(op, Expr) and op.head == "Key"):
        k = op.args[0] if isinstance(op, Expr) else op
        if is_assoc(x):
            for r in x.args:
                if r.args[0] == k:
                    return r.args[1]
            if ctx.pb:
                return missing("KeyAbsent", k)
            raise ValueError(f"Part: the key {k!r} does not exist")
        tr = WT.parse_row(x)
        if tr is not None and is_assoc(tr):
            return _part(tr, op, ctx)
        raise ValueError(f"Query: cannot take the key {k!r} of {x!r}")
    i = int(op)
    els = _elements(x)
    if els is None:
        raise ValueError(f"Query: {x!r} has no part {i}")
    n = len(els)
    if 1 <= i <= n:
        return els[i - 1]
    if -n <= i <= -1:
        return els[n + i]
    if ctx.pb:
        return missing("PartAbsent", i)
    raise ValueError(f"Part: part {i} of {x!r} does not exist")


# Missing-aware aggregation (MissingBehavior -> Automatic)
_DROP_MISSING = {"Total", "Mean", "Median", "Min", "Max", "StandardDeviation", "Variance", "Quantile", "Quartiles",
                 "RootMeanSquare", "GeometricMean", "HarmonicMean", "MeanDeviation", "MedianDeviation",
                 "InterquartileRange", "Commonest", "Kurtosis", "Skewness"}
_MIN_LEN = {"Mean": 1, "Median": 1, "StandardDeviation": 2, "Variance": 2, "Quantile": 1, "Quartiles": 1,
            "RootMeanSquare": 1, "GeometricMean": 1, "HarmonicMean": 1, "MeanDeviation": 1, "MedianDeviation": 1,
            "InterquartileRange": 1, "Commonest": 1, "Kurtosis": 2, "Skewness": 2}
_STRUCTURAL = {"Counts", "GroupBy", "Map", "Select", "Length", "Keys", "Values", "Sort", "SortBy", "Reverse",
               "Catenate", "Merge", "CountsBy", "CountDistinct", "CountDistinctBy", "DeleteDuplicates", "First",
               "Last", "Take", "Drop", "Union", "Tally"}


def _head_name(f):
    if isinstance(f, Symbol):
        return f.name
    if isinstance(f, Expr) and isinstance(f.head, Symbol):
        return f.head.name
    if isinstance(f, Expr) and isinstance(f.head, str):
        return f.head
    return None


def _apply_fn(f, x, ctx):
    """Apply an ascending operator f to x, with the MissingBehavior special rules."""
    if ctx.mb:
        name = f.name if isinstance(f, Symbol) else None
        if name in _DROP_MISSING and is_list(x):
            xs = [v for v in x.args if not is_missing(v)]
            if len(xs) < _MIN_LEN.get(name, 0):
                return missing("Indeterminate")
            return ev(Expr(name, L(*xs)))
        if isinstance(f, Expr) and f.head == "Apply" and len(f.args) == 1 and is_list(x) and \
                any(is_missing(v) for v in x.args):
            op = _head_name(f.args[0])
            if op == "Plus":
                return ev(Expr("Plus", *(v for v in x.args if not is_missing(v))))
            return missing("Indeterminate")
        if is_missing(x) and _head_name(f) in _STRUCTURAL:
            return x
        if isinstance(f, Expr) and f.head == "Function" and is_missing(x) is False:
            pass
    return apply(f, x)


# ─────────────────────────── the query engine ───────────────────────────

def run(ops, x, ctx):
    if not ops:
        return x
    op, rest = ops[0], ops[1:]
    op = _keyify(op)
    # descending part operators
    if is_sym(op, "All"):
        if is_missing(x):
            return x
        if not (is_list(x) or is_assoc(x)):
            return guarded(ctx, lambda: _raise(ValueError(f"Query: All needs a list or association, not {x!r}")), x)
        return _map_rest(rest, x, ctx)
    if _is_part_spec(op) and not (isinstance(op, Expr) and op.head == "Span"):
        r = guarded(ctx, lambda: _part(x, op, ctx), x)
        if isinstance(r, _Drop) or (isinstance(r, Expr) and r.head == "Failure"):
            return r
        return run(rest, r, ctx)
    if isinstance(op, Expr) and op.head == "Span":
        if is_missing(x):
            return x
        r = guarded(ctx, lambda: ev(Expr("Part", x, op)), x)
        if isinstance(r, _Drop):
            return r
        return _map_rest(rest, r, ctx)
    if _is_part_list(op):
        if is_missing(x):
            return x
        if is_assoc(x) and all(not K.is_int(p) for p in op.args):
            pairs = []
            for p in op.args:
                k = p.args[0] if isinstance(p, Expr) else p
                v = guarded(ctx, lambda p=p: _part(x, p, ctx), x)
                if not isinstance(v, _Drop):
                    v = run(rest, v, ctx)
                    if not isinstance(v, _Drop):
                        pairs.append((k, v))
            return assoc(*pairs)
        vals = []
        for p in op.args:
            v = guarded(ctx, lambda p=p: _part(x, p, ctx), x)
            if not isinstance(v, _Drop):
                v = run(rest, v, ctx)
                if not isinstance(v, _Drop):
                    vals.append(v)
        return L(*vals)
    if is_sym(op, "Keys") or is_sym(op, "Values"):
        if is_missing(x):
            return x
        r = guarded(ctx, lambda: ev(Expr(op.name, x)), x)
        return _map_rest(rest, r, ctx) if not isinstance(r, _Drop) else r
    if is_sym(op, "DeleteMissing"):
        r = guarded(ctx, lambda: ev(Expr("DeleteMissing", x)), x)
        return _map_rest(rest, r, ctx) if not isinstance(r, _Drop) else r
    if isinstance(op, Expr) and op.head in _DESC_FILTERS and isinstance(op.head, str):
        if is_missing(x):
            return x
        r = guarded(ctx, lambda: ev(Expr(op, x)) if False else ev(K.call(op, x)), x)
        if isinstance(r, _Drop) or (isinstance(r, Expr) and r.head == "Failure"):
            return r
        if op.head == "SelectFirst":
            return run(rest, r, ctx)
        if op.head == "GroupBy" and len(op.args) == 1:
            return _map_rest(rest, r, ctx)
        return _map_rest(rest, r, ctx)
    # composition
    if isinstance(op, Expr) and op.head == "RightComposition" and op.args:
        r = run([op.args[0]] + list(rest), x, ctx)
        for g in op.args[1:]:
            if isinstance(r, _Drop):
                return r
            r = _ascend(g, r, ctx)
        return r
    if isinstance(op, Expr) and op.head == "Composition" and op.args:
        return run([Expr("RightComposition", *reversed(op.args))] + list(rest), x, ctx)
    # ascending operators: deeper operators first
    inner = _map_rest(rest, x, ctx) if rest else x
    if isinstance(inner, _Drop):
        return inner
    return _ascend(op, inner, ctx)


def _ascend(op, r, ctx):
    if isinstance(op, Expr) and op.head == "Query":
        ops, sub = split_query(op.args)
        return run(ops, r, ctx)
    if is_list(op) and not _is_part_list(op):
        if op.args and all(is_rule(o) for o in op.args):
            # {key1 -> op1, ...}: apply opi to the given parts
            if is_assoc(r):
                out = []
                for rr in r.args:
                    k, v = rr.args
                    for o in op.args:
                        if o.args[0] == k or (isinstance(o.args[0], Expr) and o.args[0].head == "Key" and
                                              o.args[0].args[0] == k):
                            v = run([o.args[1]], v, ctx)
                    out.append((k, v))
                return assoc(*out)
            if is_list(r):
                vals = list(r.args)
                for o in op.args:
                    if K.is_int(o.args[0]):
                        i = int(o.args[0])
                        vals[i - 1] = run([o.args[1]], vals[i - 1], ctx)
                return L(*vals)
        vals = [run([o], r, ctx) for o in op.args]
        return L(*(v for v in vals if not isinstance(v, _Drop)))
    if is_assoc(op):
        pairs = [(o.args[0], run([o.args[1]], r, ctx)) for o in op.args]
        return assoc(*((k, v) for k, v in pairs if not isinstance(v, _Drop)))
    if is_sym(op, "Identity"):
        return r
    return guarded(ctx, lambda: _apply_fn(op, r, ctx), r)


def run_query(ops, ctx, data):
    ds = isinstance(data, Expr) and data.head == "Dataset" and len(data.args) >= 1
    x = data.args[0] if ds else data
    try:
        r = run(list(ops), x, ctx)
    except QueryFailure as qf:
        return qf.failure
    if isinstance(r, _Drop):
        return missing("Failed") if False else L()
    if ds and (is_list(r) or is_assoc(r)):
        return Expr("Dataset", r)
    return r


# ─────────────────────────── Query and Dataset ───────────────────────────

@fn("Query", **ADDED,
    notes="Query[op1, op2, ...][data] on nested lists/associations, Dataset and Tabular: descending part "
          "operators (All, i, i;;j, \"key\", Key, Keys, Values, {parts}) and filters (Select, SelectFirst, "
          "KeySelect, MaximalBy, MinimalBy, SortBy, KeySortBy, DeleteDuplicatesBy, DeleteMissing, GroupBy; a string "
          "argument means Key) act before deeper operators; aggregations, functions, Query[...], {op...}, "
          "<|k -> op|>, {k -> op} and op1/*op2 act on the results; options FailureAction (\"Abort\" default, None, "
          "\"Drop\", \"Encapsulate\", \"Replace\", {act, f}), MissingBehavior (Automatic: Total/Mean/... skip "
          "Missing, too-short data gives Missing[\"Indeterminate\"], Apply[Times] with Missing gives "
          "Missing[\"Indeterminate\"]; None) and PartBehavior (Automatic: absent parts give Missing[\"PartAbsent\", "
          "i] / Missing[\"KeyAbsent\", k]; None: they fail)")
def _query(*args):
    ops, ctx = split_query(args)       # validates the options
    raise Unevaluated()


def _query_call(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "Query"):
        return K.UNHANDLED
    if len(args) != 1:
        raise ValueError("Query: a query is applied to one expression")
    ops, ctx = split_query(head.args)
    data = args[0]
    if isinstance(data, Expr) and data.head == "Tabular":
        t = WT.parse_tab(data)
        if t is not None:
            return query_on_tabular(t, ops, ctx)
    return run_query(ops, ctx, data)


KL.COMPOUND_HEAD_HOOKS["Query"] = _query_call

_DS_OPTS = {"DatasetTheme", "MaxItems", "HiddenItems", "ItemDisplayFunction", "ItemStyle", "HeaderStyle",
            "HeaderDisplayFunction", "HeaderBackground", "Background", "Alignment", "ItemSize", "HeaderSize",
            "HeaderAlignment", "Alignment", "DatasetDisplayPanel", "MaxItems", "ImageSize"}


@fn("Dataset", **ADDED,
    notes="Dataset[data] and Dataset[data, opts] (display options such as DatasetTheme, MaxItems are kept): an inert "
          "wrapper of nested lists/associations; ds[ops...] = Query[ops...] on the data (structured results stay "
          "Datasets), Normal, Length, Part (ds[[...]] gives a Dataset for structured parts), Keys, Values; Dataset "
          "of a Tabular gives its rows; type and schema inference (TypeSystem) is not reproduced")
def _dataset(*args):
    if not args:
        raise ValueError("Dataset: expected Dataset[data]")
    data, opts = args[0], args[1:]
    for o in opts:
        if not (is_rule(o) and isinstance(o.args[0], Symbol) and o.args[0].name in _DS_OPTS):
            raise ValueError(f"Dataset: {o!r} is not a Dataset option")
    if isinstance(data, Expr) and data.head == "Tabular":
        t = WT.parse_tab(data)
        if t is not None:
            return Expr("Dataset", L(*t.rows()), *opts)
    if isinstance(data, Expr) and data.head == "Dataset" and data.args:
        return Expr("Dataset", data.args[0], *opts)
    raise Unevaluated()


def _dataset_call(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "Dataset" and head.args):
        return K.UNHANDLED
    ops, ctx = split_query(args)
    return run_query(ops, ctx, Expr("Dataset", head.args[0]))


KL.COMPOUND_HEAD_HOOKS["Dataset"] = _dataset_call


def _ds_data(x):
    if isinstance(x, Expr) and x.head == "Dataset" and x.args:
        return x.args[0]
    return None


def _ds_normal(a):
    if len(a) == 1 and _ds_data(a[0]) is not None:
        return _ds_data(a[0])
    return _PASS


def _ds_length(a):
    if len(a) == 1 and _ds_data(a[0]) is not None:
        d = _ds_data(a[0])
        return len(d.args) if (is_list(d) or is_assoc(d)) else 0
    return _PASS


def _ds_part(a):
    if a and _ds_data(a[0]) is not None:
        r = ev(Expr("Part", _ds_data(a[0]), *a[1:]))
        return Expr("Dataset", r) if (is_list(r) or is_assoc(r)) else r
    return _PASS


def _ds_wrapped(name):
    def pre(a):
        if a and _ds_data(a[0]) is not None:
            r = ev(Expr(name, _ds_data(a[0]), *a[1:]))
            return Expr("Dataset", r) if (is_list(r) or is_assoc(r)) and name not in ("Keys",) else r
        return _PASS
    return pre


wrap_builtin("Normal", _ds_normal)
wrap_builtin("Length", _ds_length)
wrap_builtin("Part", _ds_part)
for _n in ("Keys", "Values", "Select", "SortBy", "Take", "Reverse", "Total", "Mean", "Max", "Min", "Counts",
           "GroupBy", "Map", "First", "Last"):
    wrap_builtin(_n, _ds_wrapped(_n))
    wrap_registry(_n, _ds_wrapped(_n))


# ─────────────────────────── queries on Tabular ───────────────────────────

def _rows_result_to_tabular(r, t):
    if isinstance(r, Expr) and r.head == "Failure":
        return r
    if is_list(r) and r.args and all(is_assoc(x) for x in r.args):
        return WT._rows_to_tab(list(r.args), "Tabular").expr()
    if is_list(r) and r.args and all(is_list(x) for x in r.args) and t.keys is None:
        return WT._rows_to_tab(list(r.args), "Tabular").expr()
    if is_list(r) and not r.args:
        return WT.Tab(t.keys, [[] for _ in range(t.ncols)], t.types).expr()
    if is_assoc(r) and t.keys is not None and all(x.args[0] in t.keys for x in r.args):
        return Expr("TabularRow", r)
    if is_list(r) and all(not (is_list(x) or is_assoc(x)) for x in r.args):
        return WT.col_expr(list(r.args), WT.infer_type(list(r.args)))
    return r


def query_on_tabular(t, ops, ctx=None):
    if ctx is None:
        ops, ctx = split_query(ops)
    # tab[rows, cols] with column patterns / keys: Part-like extraction keeping column types
    if len(ops) == 2 and (is_sym(ops[0], "All") or (K.is_int(ops[0]) and not isinstance(ops[0], bool)) or
                          (isinstance(ops[0], Expr) and ops[0].head == "Span")):
        c = ops[1]
        if is_sym(c, "All") or isinstance(c, str) or (isinstance(c, Expr) and c.head in ("Key", "ExtendedKey")) or \
                (is_list(c) and all(isinstance(x, (str, int)) or (isinstance(x, Expr) and x.head in
                                                                   ("Key", "ExtendedKey")) for x in c.args)) or \
                (isinstance(c, Expr) and c.head in ("Pattern", "Blank", "Alternatives")):
            if isinstance(c, (str,)) or (isinstance(c, Expr) and c.head == "Key") or \
                    (isinstance(c, Expr) and c.head == "ExtendedKey" and t.col_index(c, "Tabular", False) is not None):
                return WT.tab_part(t, [ops[0], c], "Tabular")
            js = t.col_indices(c, "Tabular")
            sub = t.take_cols(js)
            return WT.tab_part(sub, [ops[0], Symbol("All")], "Tabular") if not K.is_int(ops[0]) else \
                sub.row_expr(WT._rows_index(ops[0], t.nrows, "Tabular")[0][0])
    rows = L(*t.rows())
    try:
        r = run(list(ops), rows, ctx)
    except QueryFailure as qf:
        return qf.failure
    return _rows_result_to_tabular(r, t)


# ─────────────────────────── relational databases (SQLite through sqlite3) ───────────────────────────

import os as _os
import sqlite3 as _sqlite3
from urllib.parse import urlparse as _urlparse, unquote as _unquote

_BACKENDS = {"sqlite": "SQLite", "sqlite3": "SQLite", "postgres": "PostgreSQL", "postgresql": "PostgreSQL",
             "mysql": "MySQL", "mariadb": "MySQL", "mssql": "MicrosoftSQL", "microsoftsql": "MicrosoftSQL",
             "sqlserver": "MicrosoftSQL", "oracle": "Oracle"}
_REF_KEYS = ["Backend", "Name", "Host", "Port", "Username", "Password"]
_CONNECTIONS = {}


def _backend(b):
    if not isinstance(b, str) or b.lower() not in _BACKENDS:
        raise ValueError(f"DatabaseReference: {b!r} is not a supported backend (SQLite, PostgreSQL, MySQL, "
                         f"MicrosoftSQL, Oracle)")
    return _BACKENDS[b.lower()]


def _ref_from_url(u):
    p = _urlparse(u)
    if not p.scheme:
        return None
    b = _backend(p.scheme)
    out = [("Backend", b)]
    if b == "SQLite":
        path = _unquote(p.netloc + p.path)
        if path:
            out.append(("Name", path))
        return out
    if p.hostname:
        out.append(("Host", p.hostname))
    if p.port:
        out.append(("Port", p.port))
    if p.path and p.path != "/":
        out.append(("Name", _unquote(p.path.lstrip("/"))))
    if p.username:
        out.append(("Username", _unquote(p.username)))
    if p.password:
        out.append(("Password", _unquote(p.password)))
    return out


def parse_ref(x):
    """The normalized association of a DatabaseReference, or None."""
    if not (isinstance(x, Expr) and x.head == "DatabaseReference" and len(x.args) == 1):
        return None
    a = x.args[0]
    if is_assoc(a):
        return a
    return None


def _ref_assoc(spec):
    if is_assoc(spec):
        pairs = []
        for r in spec.args:
            k, v = r.args
            if k not in _REF_KEYS:
                raise ValueError(f"DatabaseReference: {k!r} is not a database property ({', '.join(_REF_KEYS)})")
            pairs.append((k, _backend(v) if k == "Backend" else v))
        if not any(k == "Backend" for k, _ in pairs):
            pairs.insert(0, ("Backend", "SQLite"))
        return assoc(*sorted(pairs, key=lambda kv: _REF_KEYS.index(kv[0])))
    if isinstance(spec, Expr) and spec.head == "File" and len(spec.args) == 1 and isinstance(spec.args[0], str):
        return assoc(("Backend", "SQLite"), ("Name", spec.args[0]))
    if isinstance(spec, Expr) and spec.head == "URL" and len(spec.args) == 1 and isinstance(spec.args[0], str):
        r = _ref_from_url(spec.args[0])
        if r is None:
            raise ValueError("DatabaseReference: the URL needs a scheme such as postgres:// or sqlite://")
        return assoc(*r)
    if isinstance(spec, str):
        r = _ref_from_url(spec) if "://" in spec else None
        if r is not None:
            return assoc(*r)
        return assoc(("Backend", "SQLite"), ("Name", spec))
    raise ValueError("DatabaseReference: expected File[path], URL[url], a path or URL string, or an association")


@fn("DatabaseReference", **ADDED,
    notes="DatabaseReference[File[path]], DatabaseReference[\"path\"], DatabaseReference[URL[url]] / \"backend://"
          "user:pass@host:port/name\" and DatabaseReference[<|\"Backend\" -> ..., \"Name\" -> ..., ...|>] give the "
          "canonical DatabaseReference[<|...|>]; ref[prop] for Backend, Name, Host, Port, Username, Password and "
          "\"Connected\"; only SQLite databases can actually be opened (Python sqlite3)")
def _databasereference(spec):
    if parse_ref(Expr("DatabaseReference", spec)) is not None:
        canon = _ref_assoc(spec)
        if canon == spec:
            raise Unevaluated()
        return Expr("DatabaseReference", canon)
    return Expr("DatabaseReference", _ref_assoc(spec))


def _ref_key(a):
    return repr(a)


def _ref_hook(head, args, ev_):
    a = parse_ref(head)
    if a is None or len(args) != 1:
        return K.UNHANDLED
    p = args[0]
    if p == "Connected":
        return _ref_key(a) in _CONNECTIONS
    if p == "Properties":
        return L(*_REF_KEYS, "Connected")
    if p in _REF_KEYS:
        for r in a.args:
            if r.args[0] == p:
                return r.args[1]
        if p == "Port" and _get(a, "Backend") != "SQLite":
            return {"PostgreSQL": 5432, "MySQL": 3306, "MicrosoftSQL": 1433, "Oracle": 1521}[_get(a, "Backend")]
        return missing("NotAvailable")
    raise ValueError(f"DatabaseReference: {p!r} is not a property")


KL.COMPOUND_HEAD_HOOKS["DatabaseReference"] = _ref_hook


def _get(a, k, d=None):
    for r in a.args:
        if r.args[0] == k:
            return r.args[1]
    return d


def _to_ref(db, fname):
    if isinstance(db, Expr) and db.head == "RelationalDatabase":
        rd = _parse_rdb(db)
        if rd is None:
            raise ValueError(f"{fname}: not a valid RelationalDatabase")
        return rd[1]
    if isinstance(db, Expr) and db.head == "DatabaseReference":
        a = parse_ref(db)
        if a is None:
            a = _ref_assoc(db.args[0])
        return a
    if isinstance(db, (str,)) or (isinstance(db, Expr) and db.head in ("File", "URL")):
        return _ref_assoc(db)
    if isinstance(db, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {db!r} is not a database specification")


def _open(a, fname):
    """An sqlite3 connection for the reference (reusing a DatabaseConnect connection)."""
    key = _ref_key(a)
    if key in _CONNECTIONS:
        return _CONNECTIONS[key], False
    b = _get(a, "Backend")
    if b != "SQLite":
        raise NotImplementedError(f"{fname}: only SQLite databases are supported (no {b} driver is available)")
    name = _get(a, "Name")
    if name is None:
        return _sqlite3.connect(":memory:"), True
    if not isinstance(name, str) or not _os.path.exists(name):
        raise ValueError(f"{fname}: the database file {name!r} does not exist")
    return _sqlite3.connect(f"file:{name}?mode=rw", uri=True), True


def _success(tag, msg):
    return Expr("Success", tag, assoc(("MessageTemplate", msg)))


def _fail(tag, msg):
    return Expr("Failure", tag, assoc(("MessageTemplate", msg)))


@fn("DatabaseConnect", **ADDED,
    notes="DatabaseConnect[db] (DatabaseReference or RelationalDatabase) opens an SQLite connection and gives "
          "Success[\"DatabaseConnect\", ...]; a missing file or another backend gives a Failure; ref[\"Connected\"] "
          "reports the state")
def _databaseconnect(db):
    a = _to_ref(db, "DatabaseConnect")
    if _ref_key(a) in _CONNECTIONS:
        return _success("DatabaseConnect", "The connection is already active.")
    try:
        con, _ = _open(a, "DatabaseConnect")
    except (ValueError, NotImplementedError, _sqlite3.Error) as e:
        return _fail("DatabaseConnect", str(e))
    _CONNECTIONS[_ref_key(a)] = con
    return _success("DatabaseConnect", "Connection established.")


@fn("DatabaseDisconnect", **ADDED,
    notes="DatabaseDisconnect[db] closes the connection opened by DatabaseConnect and gives Success[...]; a Failure "
          "if there is no active connection")
def _databasedisconnect(db):
    a = _to_ref(db, "DatabaseDisconnect")
    con = _CONNECTIONS.pop(_ref_key(a), None)
    if con is None:
        return _fail("DatabaseDisconnect", "There is no active connection to this database.")
    con.close()
    return _success("DatabaseDisconnect", "Connection closed.")


def _quote(name):
    return '"' + name.replace('"', '""') + '"'


def _table_schema(con, table):
    cur = con.execute(f"PRAGMA table_info({_quote(table)})")
    info = cur.fetchall()
    cols = assoc(*((r[1], assoc(("NativeTypeString", r[2] or ""), ("Nullable", not bool(r[3]) and not r[5])))
                   for r in info))
    pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
    fks = {}
    for r in con.execute(f"PRAGMA foreign_key_list({_quote(table)})").fetchall():
        fid, seq, to_table, frm, to = r[0], r[1], r[2], r[3], r[4]
        d = fks.setdefault(fid, [to_table, [], []])
        d[1].append(frm)
        d[2].append(to if to is not None else frm)
    fk_list = [assoc(("ToTable", v[0]), ("FromColumns", L(*v[1])), ("ToColumns", L(*v[2])))
               for _, v in sorted(fks.items())]
    uniq, idxs = [], []
    for r in con.execute(f"PRAGMA index_list({_quote(table)})").fetchall():
        iname, unique, origin = r[1], r[2], (r[3] if len(r) > 3 else "c")
        icols = [c[2] for c in con.execute(f"PRAGMA index_info({_quote(iname)})").fetchall()]
        if origin == "pk":
            continue
        if unique and origin == "u":
            uniq.append(assoc(("Columns", L(*icols)), ("ConstraintName", NONE)))
        else:
            idxs.append(assoc(("Columns", L(*icols)), ("IndexName", iname)))
    return assoc(("Columns", cols),
                 ("PrimaryKey", assoc(("Columns", L(*pk)), ("ConstraintName", NONE)) if pk else NONE),
                 ("ForeignKeys", L(*fk_list)), ("UniquenessConstraints", L(*uniq)), ("Indexes", L(*idxs)))


def _parse_rdb(x):
    if isinstance(x, Expr) and x.head == "RelationalDatabase" and len(x.args) == 2 and is_assoc(x.args[0]) and \
            isinstance(x.args[1], Expr) and x.args[1].head == "DatabaseReference" and parse_ref(x.args[1]):
        return x.args[0], parse_ref(x.args[1])
    return None


@fn("RelationalDatabase", **ADDED,
    notes="RelationalDatabase[db] and RelationalDatabase[{tables}, db] (db a DatabaseReference, File, path or URL) "
          "read the SQLite schema into RelationalDatabase[<|table -> <|\"Columns\", \"PrimaryKey\", "
          "\"ForeignKeys\", \"UniquenessConstraints\", \"Indexes\"|>|>, ref]; option IncludeRelatedTables (follow "
          "foreign keys); rdb[\"Tables\"], rdb[\"Connection\"], rdb[table, prop], rdb[table, \"PrimaryKey\", prop], "
          "rdb[table, col, \"NativeTypeString\" | \"Nullable\" | All]; Tabular[rdb -> table] and Tabular[<|"
          "\"RelationalDatabase\" -> rdb, \"Query\" -> table|>] load a table into memory")
def _relationaldatabase(*args):
    if _parse_rdb(Expr("RelationalDatabase", *args)) is not None:
        raise Unevaluated()
    args = list(args)
    include = False
    while args and is_rule(args[-1]) and is_sym(args[-1].args[0], "IncludeRelatedTables"):
        v = args.pop().args[1]
        if not isinstance(v, bool):
            raise ValueError("RelationalDatabase: IncludeRelatedTables must be True or False")
        include = v
    if len(args) == 1:
        tables, db = None, args[0]
    elif len(args) == 2:
        tables, db = args
        if isinstance(tables, str):
            tables = L(tables)
        if not (is_list(tables) and all(isinstance(t, str) for t in tables.args)):
            raise ValueError("RelationalDatabase: the tables must be a list of names")
    else:
        raise ValueError("RelationalDatabase: expected RelationalDatabase[db] or RelationalDatabase[{tables}, db]")
    a = _to_ref(db, "RelationalDatabase")
    con, own = _open(a, "RelationalDatabase")
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE "
                                           "'sqlite_%' ORDER BY name").fetchall()]
        if tables is None:
            chosen = names
        else:
            chosen = []
            for t in tables.args:
                if t not in names:
                    raise ValueError(f"RelationalDatabase: there is no table {t!r}")
                chosen.append(t)
            if include:
                i = 0
                while i < len(chosen):
                    for r in con.execute(f"PRAGMA foreign_key_list({_quote(chosen[i])})").fetchall():
                        if r[2] not in chosen and r[2] in names:
                            chosen.append(r[2])
                    i += 1
        schema = assoc(*((t, _table_schema(con, t)) for t in chosen))
    finally:
        if own:
            con.close()
    return Expr("RelationalDatabase", schema, Expr("DatabaseReference", a))


def _rdb_hook(head, args, ev_):
    rd = _parse_rdb(head)
    if rd is None or not args:
        return K.UNHANDLED
    schema, a = rd
    if len(args) == 1:
        p = args[0]
        if p == "Tables":
            return L(*(r.args[0] for r in schema.args))
        if p == "Connection":
            return Expr("DatabaseReference", a)
        if p == "Properties":
            return L("Tables", "Connection")
        if isinstance(p, str) and _get(schema, p) is not None:
            raise ValueError("RelationalDatabase: give rdb[table, prop]")
        raise ValueError(f"RelationalDatabase: {p!r} is not a database property (Tables, Connection)")
    t = _get(schema, args[0])
    if t is None:
        raise ValueError(f"RelationalDatabase: there is no table {args[0]!r}")
    if len(args) == 2:
        p = args[1]
        if is_sym(p, "All"):
            return t
        if p in ("Columns",):
            return L(*(r.args[0] for r in _get(t, "Columns").args))
        if p in ("PrimaryKey", "ForeignKeys", "UniquenessConstraints", "Indexes"):
            return _get(t, p)
        if p == "Properties":
            return L("Columns", "PrimaryKey", "ForeignKeys", "UniquenessConstraints", "Indexes")
        raise ValueError(f"RelationalDatabase: {p!r} is not a table property")
    if len(args) == 3:
        x, p = args[1], args[2]
        if x == "PrimaryKey":
            pk = _get(t, "PrimaryKey")
            if is_sym(pk, "None"):
                return missing("NotAvailable")
            if p in ("Columns", "ConstraintName"):
                return _get(pk, p)
            raise ValueError("RelationalDatabase: primary key properties are Columns and ConstraintName")
        col = _get(_get(t, "Columns"), x)
        if col is None:
            raise ValueError(f"RelationalDatabase: the table {args[0]!r} has no column {x!r}")
        if is_sym(p, "All"):
            return col
        if p in ("NativeTypeString", "Nullable"):
            return _get(col, p)
        raise ValueError("RelationalDatabase: column properties are NativeTypeString and Nullable")
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["RelationalDatabase"] = _rdb_hook


def _sql_value(v):
    if v is None:
        return missing("NotAvailable")
    if isinstance(v, bytes):
        return Expr("ByteArray", L(*v))
    return v


def load_table(rdb, table, fname="Tabular"):
    rd = _parse_rdb(rdb)
    if rd is None:
        raise ValueError(f"{fname}: not a valid RelationalDatabase")
    schema, a = rd
    if table is None:
        names = [r.args[0] for r in schema.args]
        if len(names) != 1:
            raise ValueError(f"{fname}: the database has several tables; use Tabular[rdb -> \"table\"]")
        table = names[0]
    t = _get(schema, table)
    if t is None:
        raise ValueError(f"{fname}: there is no table {table!r}")
    cols = [r.args[0] for r in _get(t, "Columns").args]
    con, own = _open(a, fname)
    try:
        rows = con.execute(f"SELECT {', '.join(_quote(c) for c in cols)} FROM {_quote(table)}").fetchall()
    finally:
        if own:
            con.close()
    pk = _get(t, "PrimaryKey")
    kc = list(_get(pk, "Columns").args) if is_assoc(pk) else None
    data = [[_sql_value(r[j]) for r in rows] for j in range(len(cols))]
    return WT.Tab(cols, data, None, kc)


def _tabular_db_pre(a):
    if not a:
        return _PASS
    x = a[0]
    if isinstance(x, Expr) and x.head == "RelationalDatabase":
        return load_table(x, None).expr()
    if is_rule(x) and isinstance(x.args[0], Expr) and x.args[0].head == "RelationalDatabase":
        return load_table(x.args[0], x.args[1]).expr()
    if is_assoc(x) and _get(x, "RelationalDatabase") is not None:
        q = _get(x, "Query")
        if not isinstance(q, str):
            raise NotImplementedError("Tabular: only a table name is supported as \"Query\"")
        return load_table(_get(x, "RelationalDatabase"), q).expr()
    return _PASS


wrap_registry("Tabular", _tabular_db_pre)


@fn("DataConnectionObject", **ADDED,
    notes="DataConnectionObject[<|\"ConnectionName\" -> ..., \"Location\" -> ..., ...|>] (and a second "
          "association, Authentication option): an inert, validated reference; obj[\"ConnectionName\"], "
          "obj[\"Location\"], obj[\"Redownload\"] (default False), obj[\"Properties\"] and other keys; retrieving "
          "remote data (Import) is not available")
def _dataconnectionobject(*args):
    args = list(args)
    while args and is_rule(args[-1]) and is_sym(args[-1].args[0], "Authentication"):
        args.pop()
    if not args or not is_assoc(args[0]) or len(args) > 2 or (len(args) == 2 and not is_assoc(args[1])):
        raise ValueError("DataConnectionObject: expected DataConnectionObject[<|\"ConnectionName\" -> ..., ...|>]")
    if not isinstance(_get(args[0], "ConnectionName"), str):
        raise ValueError("DataConnectionObject: the association needs a \"ConnectionName\" string")
    raise Unevaluated()


def _dco_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "DataConnectionObject" and head.args and is_assoc(head.args[0])
            and len(args) == 1):
        return K.UNHANDLED
    a = head.args[0]
    p = args[0]
    if p == "Properties":
        ks = [r.args[0] for r in a.args]
        for k in ("ConnectionName", "Location", "Redownload"):
            if k not in ks:
                ks.append(k)
        return L(*ks)
    v = _get(a, p)
    if v is not None:
        return v
    if p == "Redownload":
        return False
    return missing("NotAvailable")


KL.COMPOUND_HEAD_HOOKS["DataConnectionObject"] = _dco_hook
