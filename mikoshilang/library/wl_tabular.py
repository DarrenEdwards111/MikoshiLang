"""Wolfram batch W12, part 2: Tabular data (Tabular, TabularColumn, TabularRow and the column/row operations).

A Tabular object is kept as plain Python columns.  Its canonical expression follows the layout of Wolfram's own
InputForm,

    Tabular[<|"RawSchema" -> <|"ColumnProperties" -> <|key -> <|"ElementType" -> type|>, ...|>,
                                "KeyColumns" -> None | {key, ...}, "Backend" -> "WolframKernel"|>,
              "Options" -> {opts}, "BackendData" -> <|"ColumnData" -> {TabularColumn[...], ...}|>|>]

with TabularColumn[<|"Data" -> {{v1, v2, ...}, {}, None}, "ElementType" -> type|>]; anonymous columns use a list of
column properties.  Wolfram's packed forms (DataStructure["ColumnTable", ...] and the {n, offsets, chars} string
encoding) are accepted as input.  Element types are inferred as in Wolfram for the common cases ("Integer64",
"Real64", "ComplexReal64", "String", "Boolean", "Null"); dates, times, Nominal/Ordinal categories and lists get
TypeSpecifier["Date"|"Time"|"Nominal"|"Ordinal"|"ListVector"][...] types; everything else is "InertExpression".
"""
from __future__ import annotations

import math

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, REGISTRY
from .wl_dates_tabular import (AREA, ADDED, ev, is_sym, is_rule, is_assoc, is_list, assoc, _added, wrap_registry,
                               wrap_builtin, _PASS, parse_dobj, parse_tobj, _category, _INT_TYPES)

NONE = Symbol("None")
ALL = Symbol("All")


def apply(f, *args):
    return ev(K.call(f, *args))


def missing(*a):
    return Expr("Missing", *a)


def is_missing(x):
    return isinstance(x, Expr) and x.head == "Missing"


def akeys(a):
    return [r.args[0] for r in a.args]


def avals(a):
    return [r.args[1] for r in a.args]


def aget(a, k, default=None):
    for r in a.args:
        if r.args[0] == k:
            return r.args[1]
    return default


def TS(name, *params):
    return Expr(Expr("TypeSpecifier", name), *params)


def ts_name(t):
    """'Date' for TypeSpecifier["Date"][...], the string for a plain type name."""
    if isinstance(t, str):
        return t
    if isinstance(t, Expr) and isinstance(t.head, Expr) and t.head.head == "TypeSpecifier" and t.head.args:
        return t.head.args[0]
    return None


# ─────────────────────────── element types ───────────────────────────

_ALIASES = {"MachineInteger": "Integer64", "MachineReal": "Real64", "Integer": "Integer64", "Real": "Real64",
            "Expression": "InertExpression", "Complex": "ComplexReal64", "MachineComplex": "ComplexReal64"}
_SIMPLE = set(_INT_TYPES) | {"Real32", "Real64", "ComplexReal32", "ComplexReal64", "String", "Boolean",
                             "InertExpression", "Null"}


def _is_real(x):
    return K.is_number(x) and not isinstance(x, (bool, complex)) and not K.is_int(x)


def infer_type(vals):
    xs = [v for v in vals if not is_missing(v)]
    if not xs:
        return "Null"
    if all(isinstance(v, bool) for v in xs):
        return "Boolean"
    if all(K.is_int(v) and not isinstance(v, bool) for v in xs):
        return "Integer64" if all(-2 ** 63 <= int(v) < 2 ** 63 for v in xs) else "InertExpression"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in xs):
        return "Real64"
    if all((isinstance(v, (int, float, complex)) and not isinstance(v, bool)) for v in xs) and \
            any(isinstance(v, complex) for v in xs) and \
            all(isinstance(v, (float, complex)) or isinstance(v, int) for v in xs):
        return "ComplexReal64"
    if all(isinstance(v, str) for v in xs):
        return "String"
    ds = [parse_dobj(v) if isinstance(v, Expr) and v.head == "DateObject" else None for v in xs]
    if all(d is not None for d in ds):
        grans = {d.gran for d in ds}
        return TS("Date", grans.pop() if len(grans) == 1 else "Instant")
    ts = [parse_tobj(v) if isinstance(v, Expr) and v.head == "TimeObject" else None for v in xs]
    if all(t is not None for t in ts):
        grans = {t.gran for t in ts}
        return TS("Time", grans.pop() if len(grans) == 1 else "Instant")
    cs = [_category(v) for v in xs]
    if all(c is not None for c in cs) and len({repr(c[0].assoc()) for c in cs}) == 1 and \
            len({c[0].kind for c in cs}) == 1:
        return TS(cs[0][0].kind, L(*cs[0][0].cats))
    if all(is_list(v) for v in xs):
        inner = infer_type([y for v in xs for y in v.args])
        return TS("ListVector", inner)
    return "InertExpression"


def norm_type(t, fname):
    if isinstance(t, str):
        t = _ALIASES.get(t, t)
        if t in _SIMPLE:
            return t
        raise ValueError(f"{fname}: {t!r} is not a supported element type")
    n = ts_name(t)
    if n in ("Date", "Time", "Nominal", "Ordinal", "ListVector"):
        return t
    raise ValueError(f"{fname}: {t!r} is not a supported element type")


_ROUND = {"Round": round, "Floor": math.floor, "Ceiling": math.ceil, "Truncate": math.trunc}


def convert(v, t, fname, method=None):
    """Convert one element to the element type t (Missing passes through); a clear error if it cannot be."""
    if is_missing(v):
        return v
    n = ts_name(t)
    if n in _INT_TYPES:
        bits, signed = _INT_TYPES[n]
        if isinstance(v, bool):
            raise ValueError(f"{fname}: {v!r} is not an integer")
        if isinstance(v, str):
            try:
                v = int(v.strip())
            except ValueError:
                raise ValueError(f"{fname}: the string {v!r} is not an integer") from None
        if not K.is_int(v):
            if K.is_number(v) and not isinstance(v, complex):
                x = float(K.numeric_value(v)) if hasattr(K, "numeric_value") else float(ev(Expr("N", v)))
                if method in _ROUND:
                    v = int(_ROUND[method](x))
                elif float(x).is_integer():
                    v = int(x)
                else:
                    raise ValueError(f"{fname}: {v!r} is not an integer; give a conversion method such as "
                                     f"\"Round\"")
            else:
                raise ValueError(f"{fname}: {v!r} is not an integer")
        v = int(v)
        lo, hi = (-(1 << (bits - 1)), (1 << (bits - 1)) - 1) if signed else (0, (1 << bits) - 1)
        if not lo <= v <= hi:
            raise ValueError(f"{fname}: {v} does not fit in {n}")
        return v
    if n in ("Real32", "Real64"):
        if isinstance(v, bool):
            raise ValueError(f"{fname}: {v!r} is not a real number")
        if isinstance(v, str):
            try:
                return float(v.strip())
            except ValueError:
                raise ValueError(f"{fname}: the string {v!r} is not a number") from None
        if K.is_number(v) and not isinstance(v, complex):
            return float(ev(Expr("N", v)))
        raise ValueError(f"{fname}: {v!r} is not a real number")
    if n in ("ComplexReal32", "ComplexReal64"):
        if K.is_number(v) and not isinstance(v, bool):
            r = ev(Expr("N", v))
            return r if isinstance(r, (float, complex)) else complex(r)
        raise ValueError(f"{fname}: {v!r} is not a number")
    if n == "String":
        if isinstance(v, str):
            return v
        return ev(Expr("ToString", v))
    if n == "Boolean":
        if isinstance(v, bool):
            return v
        if v in ("True", "False"):
            return v == "True"
        raise ValueError(f"{fname}: {v!r} is not True or False")
    if n == "Null":
        raise ValueError(f"{fname}: a \"Null\" column can only hold missing values")
    if n == "Date":
        if parse_dobj(v) is None:
            d = ev(Expr("DateObject", v))
            if parse_dobj(d) is None:
                raise ValueError(f"{fname}: {v!r} is not a date")
            return d
        return v
    if n == "Time":
        if parse_tobj(v) is None:
            raise ValueError(f"{fname}: {v!r} is not a time")
        return v
    if n in ("Nominal", "Ordinal"):
        c = _category(v)
        if c is None:
            return ev(Expr(n, t.args[0], v))
        return v
    return v


def convert_column(vals, t, fname, method=None):
    if t is None:
        return list(vals), infer_type(vals)
    t = norm_type(t, fname)
    return [convert(v, t, fname, method) for v in vals], t


# ─────────────────────────── the Tab class ───────────────────────────

class Tab:
    def __init__(self, keys, cols, types=None, keycols=None, options=()):
        self.keys = list(keys) if keys is not None else None
        self.cols = [list(c) for c in cols]
        self.types = list(types) if types is not None else [infer_type(c) for c in self.cols]
        self.keycols = list(keycols) if keycols else None
        self.options = list(options)
        if self.keys is not None and len(self.keys) != len(self.cols):
            raise ValueError("Tabular: the number of column keys does not match the number of columns")
        if len({len(c) for c in self.cols}) > 1:
            raise ValueError("Tabular: all columns must have the same length")
        if self.keys is not None and len(set(map(repr, self.keys))) != len(self.keys):
            raise ValueError("Tabular: column keys must be distinct")
        if self.keycols:
            if self.keys is None:
                raise ValueError("Tabular: key columns need named columns")
            for k in self.keycols:
                if k not in self.keys:
                    raise ValueError(f"Tabular: the key column {k!r} is not a column")
            seen = set()
            for r in range(self.nrows):
                kv = repr([self.cols[self.keys.index(k)][r] for k in self.keycols])
                if kv in seen:
                    raise ValueError("Tabular: the key columns do not identify the rows uniquely")
                seen.add(kv)

    @property
    def nrows(self):
        return len(self.cols[0]) if self.cols else 0

    @property
    def ncols(self):
        return len(self.cols)

    def colexpr(self, j):
        return Expr("TabularColumn", assoc(("Data", L(L(*self.cols[j]), L(), NONE)), ("ElementType", self.types[j])))

    def expr(self):
        if self.keys is None:
            props = L(*(assoc(("ElementType", t)) for t in self.types))
        else:
            props = assoc(*((k, assoc(("ElementType", t))) for k, t in zip(self.keys, self.types)))
        raw = assoc(("ColumnProperties", props), ("KeyColumns", L(*self.keycols) if self.keycols else NONE),
                    ("Backend", "WolframKernel"))
        return Expr("Tabular", assoc(("RawSchema", raw), ("Options", L(*self.options)),
                                     ("BackendData", assoc(("ColumnData", L(*(self.colexpr(j)
                                                                               for j in range(self.ncols))))))))

    def row(self, i):
        return [c[i] for c in self.cols]

    def row_value(self, i):
        """The row as an association (named columns) or a list (anonymous columns)."""
        vals = self.row(i)
        return assoc(*zip(self.keys, vals)) if self.keys is not None else L(*vals)

    def rows(self):
        return [self.row_value(i) for i in range(self.nrows)]

    def row_expr(self, i):
        return Expr("TabularRow", self.row_value(i))

    def take_rows(self, idx):
        return Tab(self.keys, [[c[i] for i in idx] for c in self.cols], self.types,
                   self.keycols if self.keycols and all(k in (self.keys or []) for k in self.keycols) else None,
                   self.options)

    def take_cols(self, js):
        keys = [self.keys[j] for j in js] if self.keys is not None else None
        kc = [k for k in (self.keycols or []) if keys and k in keys] or None
        return Tab(keys, [self.cols[j] for j in js], [self.types[j] for j in js], kc, self.options)

    def col_index(self, c, fname="Tabular", must=True):
        """Index of the column named by c (key, Key[key], ExtendedKey or a position)."""
        if K.is_int(c) and not isinstance(c, bool) and not (self.keys and c in self.keys):
            j = int(c)
            if 1 <= j <= self.ncols:
                return j - 1
            if -self.ncols <= j <= -1:
                return self.ncols + j
            if must:
                raise ValueError(f"{fname}: column {j} does not exist (the table has {self.ncols} columns)")
            return None
        if isinstance(c, Expr) and c.head == "Key" and len(c.args) == 1:
            c = c.args[0]
        if self.keys is not None and c in self.keys:
            return self.keys.index(c)
        if must:
            raise ValueError(f"{fname}: there is no column {c!r}")
        return None

    def col_indices(self, spec, fname="Tabular"):
        if is_sym(spec, "All"):
            return list(range(self.ncols))
        if is_list(spec):
            return [self.col_index(c, fname) for c in spec.args]
        if isinstance(spec, Expr) and spec.head == "Span":
            return _span(spec, self.ncols, fname)
        if isinstance(spec, (str, int)) or (isinstance(spec, Expr) and spec.head in ("Key", "ExtendedKey")) and \
                self.col_index(spec, fname, False) is not None:
            return [self.col_index(spec, fname)]
        if self.keys is not None:      # a pattern
            js = [j for j, k in enumerate(self.keys) if ev(Expr("MatchQ", k, spec)) is True]
            return js
        raise ValueError(f"{fname}: {spec!r} is not a column specification")

    def keyed_row(self, key, fname="Tabular"):
        if not self.keycols:
            raise ValueError(f"{fname}: the table has no key columns")
        if is_assoc(key):
            want = [aget(key, k, missing("KeyAbsent", k)) for k in self.keycols]
            if len(key.args) != len(self.keycols) or any(k not in self.keycols for k in akeys(key)):
                raise ValueError(f"{fname}: the association must give exactly the key columns {self.keycols}")
        else:
            want = list(key.args) if is_list(key) else [key]
        if len(want) != len(self.keycols):
            raise ValueError(f"{fname}: a row key needs {len(self.keycols)} value(s)")
        kj = [self.keys.index(k) for k in self.keycols]
        for i in range(self.nrows):
            if all(self.cols[j][i] == w for j, w in zip(kj, want)):
                return i
        return None


def _span(sp, n, fname):
    a = list(sp.args) + [1] * (3 - len(sp.args))
    a, b, s = a[0], a[1], a[2]
    a = 1 if is_sym(a, "All") else a
    b = n if is_sym(b, "All") else b
    if not (K.is_int(a) and K.is_int(b) and K.is_int(s)) or int(s) == 0:
        raise ValueError(f"{fname}: bad span")
    a, b, s = int(a), int(b), int(s)
    a = a + n + 1 if a < 0 else a
    b = b + n + 1 if b < 0 else b
    if not (1 <= a <= n + 1 and 0 <= b <= n):
        raise ValueError(f"{fname}: the span {sp!r} is out of range")
    return [i - 1 for i in range(a, b + (1 if s > 0 else -1), s)]


def _rows_index(spec, n, fname):
    if is_sym(spec, "All"):
        return list(range(n)), True
    if K.is_int(spec) and not isinstance(spec, bool):
        i = int(spec)
        if 1 <= i <= n:
            return [i - 1], False
        if -n <= i <= -1:
            return [n + i], False
        raise ValueError(f"{fname}: row {i} does not exist (the table has {n} rows)")
    if is_list(spec):
        out = []
        for x in spec.args:
            out.extend(_rows_index(x, n, fname)[0] if K.is_int(x) else
                       (_ for _ in ()).throw(ValueError(f"{fname}: {x!r} is not a row index")))
        return out, True
    if isinstance(spec, Expr) and spec.head == "Span":
        return _span(spec, n, fname), True
    raise ValueError(f"{fname}: {spec!r} is not a row specification")


# ─────────────────────────── parsing ───────────────────────────

def _decode_strings(data):
    n, offs, chars = data.args
    if not (is_list(offs) and isinstance(chars, str)):
        return None
    b = chars.encode("utf-8")
    o = [int(x) for x in offs.args]
    return [b[o[i]:o[i + 1]].decode("utf-8") for i in range(len(o) - 1)]


def decode_column(tc, fname="Tabular"):
    if not (isinstance(tc, Expr) and tc.head == "TabularColumn" and len(tc.args) == 1 and is_assoc(tc.args[0])):
        return None
    a = tc.args[0]
    t = aget(a, "ElementType")
    data = aget(a, "Data")
    cached = aget(a, "CachedOriginalExpression")
    if cached is not None and is_list(cached):
        return list(cached.args), t
    if not (is_list(data) and len(data.args) == 3):
        return None
    first, mask, _ = data.args
    if is_list(mask) and mask.args:
        raise NotImplementedError(f"{fname}: Wolfram's packed encoding of missing values is not supported")
    if is_list(first) and ts_name(t) == "String" and len(first.args) == 3 and K.is_int(first.args[0]) and \
            is_list(first.args[1]) and isinstance(first.args[2], str):
        vals = _decode_strings(first)       # Wolfram's {n, byte offsets, concatenated characters}
        if vals is not None and len(vals) == int(first.args[0]):
            return vals, t
    if is_list(first):
        return list(first.args), t
    raise NotImplementedError(f"{fname}: this packed column encoding ({t!r}) is not supported")


def parse_tab(e):
    """A Tab from a canonical (or Wolfram InputForm) Tabular expression, else None."""
    if not (isinstance(e, Expr) and e.head == "Tabular" and len(e.args) == 1 and is_assoc(e.args[0])):
        return None
    a = e.args[0]
    raw, bd = aget(a, "RawSchema"), aget(a, "BackendData")
    if not (is_assoc(raw) and is_assoc(bd)):
        return None
    cd = aget(bd, "ColumnData")
    if isinstance(cd, Expr) and cd.head == "DataStructure" and len(cd.args) == 2 and cd.args[0] == "ColumnTable":
        inner = cd.args[1]
        cd = inner.args[0] if is_list(inner) and len(inner.args) == 1 and is_list(inner.args[0]) else inner
    if not is_list(cd):
        return None
    cols, types = [], []
    for tc in cd.args:
        r = decode_column(tc)
        if r is None:
            return None
        cols.append(r[0])
        types.append(r[1])
    props = aget(raw, "ColumnProperties")
    keys = akeys(props) if is_assoc(props) else None
    kc = aget(raw, "KeyColumns")
    kc = list(kc.args) if is_list(kc) else None
    opts = aget(a, "Options")
    t = Tab(keys, cols, None, kc, opts.args if is_list(opts) else ())
    t.types = [norm_type(x, "Tabular") if isinstance(x, str) else x for x in types]
    return t


def is_tabular(x):
    return isinstance(x, Expr) and x.head == "Tabular" and parse_tab(x) is not None


def _rows_to_tab(rows, fname):
    """Tab from a list of rows (lists) or associations."""
    if not rows:
        return Tab(None, [])
    if all(is_assoc(r) for r in rows):
        keys = akeys(rows[0])
        ks = set(map(repr, keys))
        for r in rows[1:]:
            for k in akeys(r):
                if repr(k) not in ks:
                    keys.append(k)
                    ks.add(repr(k))
        cols = [[aget(r, k, missing("KeyAbsent", k)) for r in rows] for k in keys]
        return Tab(keys, cols)
    if all(is_list(r) for r in rows):
        n = {len(r.args) for r in rows}
        if len(n) != 1:
            raise ValueError(f"{fname}: the rows must all have the same length")
        n = n.pop()
        return Tab(None, [[r.args[j] for r in rows] for j in range(n)])
    if all(isinstance(r, Expr) and r.head == "TabularRow" for r in rows):
        return _rows_to_tab([_row_value(r) for r in rows], fname)
    raise ValueError(f"{fname}: the data must be a matrix or a list of associations")


def _row_value(r):
    if isinstance(r, Expr) and r.head == "TabularRow":
        if len(r.args) == 2 and is_list(r.args[0]) and is_list(r.args[1]):
            return assoc(*zip(r.args[1].args, r.args[0].args))
        return r.args[0]
    return r


def to_tab(x, fname, rows_ok=True):
    """Tab from a Tabular, Dataset, list of rows/associations or a TabularRow list."""
    t = parse_tab(x) if isinstance(x, Expr) and x.head == "Tabular" else None
    if t is not None:
        return t
    if isinstance(x, Expr) and x.head == "Dataset" and x.args:
        return to_tab(x.args[0], fname)
    if isinstance(x, Expr) and x.head in ("SparseArray", "QuantityArray", "StructuredArray", "NumericArray"):
        return to_tab(ev(Expr("Normal", x)), fname)
    if is_list(x) and rows_ok:
        return _rows_to_tab(list(x.args), fname)
    if is_assoc(x) and rows_ok and all(is_assoc(v) for v in avals(x)):
        # <|k1 -> <|...|>, ...|>: the outer keys become a key column named Automatic -> "Key"
        raise ValueError(f"{fname}: use ToTabular[data, \"Columns\"] or a list of associations")
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not tabular data")


def _schema_assoc(s, fname):
    if isinstance(s, Expr) and s.head == "TabularSchema" and len(s.args) == 1 and is_assoc(s.args[0]):
        return s.args[0]
    if is_assoc(s):
        return s
    return None


def _types_spec(spec, t, fname):
    """Per-column types from a list, association or rule(s)."""
    out = [None] * t.ncols
    if is_list(spec) and all(not is_rule(x) for x in spec.args):
        if len(spec.args) != t.ncols:
            raise ValueError(f"{fname}: give one element type per column")
        return list(spec.args)
    rules = spec.args if (is_list(spec) or is_assoc(spec)) else [spec]
    for r in rules:
        if not is_rule(r):
            if isinstance(r, (str, Expr)) and not is_list(spec):
                return [r] * t.ncols
            raise ValueError(f"{fname}: {r!r} is not a column -> type rule")
        out[t.col_index(r.args[0], fname)] = r.args[1]
    return out


def apply_schema(t, s, fname):
    a = _schema_assoc(s, fname)
    if a is None:
        raise ValueError(f"{fname}: {s!r} is not a schema (an association or TabularSchema)")
    for r in a.args:
        k, v = r.args
        if k == "ColumnKeys":
            if is_sym(v, "None") or is_sym(v, "Automatic"):
                t.keys = None if is_sym(v, "None") else t.keys
                continue
            if not is_list(v) or len(v.args) != t.ncols:
                raise ValueError(f"{fname}: \"ColumnKeys\" must be a list with one key per column")
            t = Tab(list(v.args), t.cols, t.types, None, t.options)
        elif k in ("ElementType", "ColumnTypes", "ElementTypes"):
            ts = _types_spec(v, t, fname)
            cols, types = list(t.cols), list(t.types)
            for j, ty in enumerate(ts):
                if ty is not None and not is_sym(ty, "Automatic"):
                    cols[j], types[j] = convert_column(t.cols[j], ty, fname)
            t = Tab(t.keys, cols, types, t.keycols, t.options)
        elif k == "KeyColumns":
            kc = None if is_sym(v, "None") else (list(v.args) if is_list(v) else [v])
            if kc:
                kc = [t.keys[t.col_index(c, fname)] if t.keys else c for c in kc]
            t = Tab(t.keys, t.cols, t.types, kc, t.options)
        elif k in ("Backend", "DefaultColumnBaseName", "ColumnProperties", "Nullable"):
            if k == "Backend" and v not in ("WolframKernel", "Automatic") and not is_sym(v, "Automatic"):
                raise NotImplementedError(f"{fname}: only the in-memory \"WolframKernel\" backend is available")
        else:
            raise ValueError(f"{fname}: {k!r} is not a schema property")
    return t


_TAB_OPTS = {"AllowedDimensions", "AppearanceElements", "BaselinePosition", "BaseStyle", "ImageMargins", "ImageSize",
             "Scrollbars", "ScrollPosition", "Alignment", "Background", "Dividers", "HeaderAlignment",
             "HeaderBackground", "HeaderSize", "HeaderStyle", "ItemDisplayFunction", "ItemSize", "ItemStyle"}


def _split_tab_opts(args):
    args = list(args)
    opts = []
    while args and is_rule(args[-1]) and isinstance(args[-1].args[0], Symbol) and args[-1].args[0].name in _TAB_OPTS:
        opts.insert(0, args.pop())
    return args, opts


def _columntable_pre(a):
    # DataStructure["ColumnTable", {{TabularColumn[...], ...}}] is the column store inside Wolfram's InputForm of a
    # Tabular object; it stays inert so that such input can be read by Tabular
    if a and a[0] == "ColumnTable":
        raise Unevaluated()
    return _PASS


wrap_registry("DataStructure", _columntable_pre)


@fn("Tabular", **ADDED,
    notes="Tabular[rows], Tabular[{assoc, ...}], Tabular[data, {keys}], Tabular[data, schema] (an association or "
          "TabularSchema with \"ColumnKeys\", \"KeyColumns\", \"ElementType\"/\"ColumnTypes\"), Tabular[Dataset[...]], "
          "Tabular[tab, opts] (display options such as AllowedDimensions are stored); in-memory column store with "
          "inferred types; tab[[i]], tab[[i, j]], tab[[rows, cols]], tab[RowKey[...]], tab[<|key -> v|>], "
          "tab[All, cols], Normal, Length, Dimensions, Select, Take, Drop, Reverse, SortBy, Join, Map; Wolfram's "
          "packed InputForm is accepted; SQL-backed Tabular is not available")
def _tabular(*args):
    if len(args) == 1 and parse_tab(Expr("Tabular", *args)) is not None:
        t = parse_tab(Expr("Tabular", *args))
        if Expr("Tabular", *args) == t.expr():
            raise Unevaluated()
        return t.expr()
    args, opts = _split_tab_opts(args)
    if not args or len(args) > 2:
        raise ValueError("Tabular: expected Tabular[data], Tabular[data, keys] or Tabular[data, schema]")
    data = args[0]
    if isinstance(data, Expr) and data.head == "RelationalDatabase":
        raise NotImplementedError("Tabular: SQL-backed Tabular objects are not available; use "
                                  "DatabaseReference with Normal or the SQL query functions")
    t = to_tab(data, "Tabular")
    if len(args) == 2:
        s = args[1]
        if is_list(s):
            if len(s.args) != t.ncols:
                raise ValueError("Tabular: give one column key per column")
            t = Tab(list(s.args), t.cols, t.types, None, t.options)
        else:
            t = apply_schema(t, s, "Tabular")
    if opts:
        t.options = [o for o in t.options if not any(o.args[0] == p.args[0] for p in opts)] + opts
    return t.expr()


@fn("TabularQ", **ADDED, notes="TabularQ[x]: True for a valid Tabular object"
                               ". Check: reference (differential cases in tests/library).")
def _tabularq(x: Expr) -> Boolean:
    try:
        return is_tabular(x)
    except (ValueError, NotImplementedError):
        return False


# ─────────────────────────── TabularColumn and TabularRow ───────────────────────────

def parse_column(x):
    try:
        r = decode_column(x)
    except NotImplementedError:
        return None
    if r is None:
        return None
    vals, t = r
    return vals, (norm_type(t, "TabularColumn") if isinstance(t, str) else t)


def col_expr(vals, t):
    return Expr("TabularColumn", assoc(("Data", L(L(*vals), L(), NONE)), ("ElementType", t)))


@fn("TabularColumn", **ADDED,
    notes="TabularColumn[{v...}] (type inferred), TabularColumn[{...}, etype], TabularColumn[{...}, etype, "
          "<|\"ConversionMethod\" -> \"Round\" | \"Floor\" | \"Ceiling\" | \"Truncate\"|>]; types Integer8-64, "
          "UnsignedInteger8-64, Real32/64, ComplexReal32/64, MachineInteger, MachineReal, String, Boolean, "
          "InertExpression; col[\"ElementType\"], col[\"Length\"], Normal, Length, Part, Total, Mean, Min, Max"
          ". Check: reference (differential cases in tests/library).")
def _tabularcolumn(vals: Expr, etype: Expr = None, method: Expr = None) -> Expr:
    if etype is None and parse_column(Expr("TabularColumn", vals)) is not None:
        vs, t = parse_column(Expr("TabularColumn", vals))
        if col_expr(vs, t) == Expr("TabularColumn", vals):
            raise Unevaluated()
        return col_expr(vs, t)
    if parse_column(vals) is not None:
        vals = L(*parse_column(vals)[0])
    if not is_list(vals):
        if isinstance(vals, Symbol):
            raise Unevaluated()
        raise ValueError("TabularColumn: the data must be a list")
    m = None
    if method is not None:
        m = aget(method, "ConversionMethod") if is_assoc(method) else method
        if m not in _ROUND and m != "Strict":
            raise ValueError("TabularColumn: the conversion method must be \"Round\", \"Floor\", \"Ceiling\", "
                             "\"Truncate\" or \"Strict\"")
    vs, t = convert_column(list(vals.args), etype, "TabularColumn", m)
    return col_expr(vs, t)


def _type_matches(t, sel):
    n = ts_name(t) or ""
    classes = {
        "Numbers": lambda: n in _INT_TYPES or n in ("Real32", "Real64", "ComplexReal32", "ComplexReal64"),
        "MachineNumbers": lambda: n in _INT_TYPES or n in ("Real32", "Real64", "ComplexReal32", "ComplexReal64"),
        "Reals": lambda: n in _INT_TYPES or n in ("Real32", "Real64"),
        "MachineReals": lambda: n in _INT_TYPES or n in ("Real32", "Real64"),
        "Integers": lambda: n in _INT_TYPES, "MachineIntegers": lambda: n in _INT_TYPES,
        "FloatingPoint": lambda: n in ("Real32", "Real64", "ComplexReal32", "ComplexReal64"),
        "MachineFloatingPoint": lambda: n in ("Real32", "Real64", "ComplexReal32", "ComplexReal64"),
        "FloatingPointReals": lambda: n in ("Real32", "Real64"),
        "MachineFloatingPointReals": lambda: n in ("Real32", "Real64"),
        "FloatingPointComplexes": lambda: n in ("ComplexReal32", "ComplexReal64"),
        "Lists": lambda: n == "ListVector", "Strings": lambda: n == "String"}
    if isinstance(sel, str) and sel in classes:
        return classes[sel]()
    if isinstance(sel, str) or (isinstance(sel, Expr) and sel.head in ("StringExpression", "Alternatives")):
        if isinstance(t, str):
            from .wl_strings import _smatchq
            return _smatchq(t, sel) is True
        return ev(Expr("MatchQ", t, sel)) is True
    return ev(Expr("MatchQ", t, sel)) is True


@fn("TabularColumnQ", **ADDED,
    notes="TabularColumnQ[col] and TabularColumnQ[col, tsel] (type pattern such as \"Integer*\" or a class: "
          "Numbers, Reals, Integers, MachineIntegers, FloatingPoint, Lists, Strings, ...)"
          ". Check: reference (differential cases in tests/library).")
def _tabularcolumnq(x: Expr, sel: Expr = None) -> Boolean:
    c = parse_column(x) if isinstance(x, Expr) and x.head == "TabularColumn" else None
    if c is None:
        return False
    return True if sel is None else _type_matches(c[1], sel)


def parse_row(x):
    if not (isinstance(x, Expr) and x.head == "TabularRow"):
        return None
    if len(x.args) == 1 and (is_assoc(x.args[0]) or is_list(x.args[0])):
        return x.args[0]
    return None


@fn("TabularRow", **ADDED,
    notes="TabularRow[<|k -> v, ...|>], TabularRow[{v...}], TabularRow[{v...}, {k...}] (= the association form); "
          "Normal gives the association or list; ColumnKeys (None when unnamed), ColumnTypes, Part and row[key]")
def _tabularrow(data: Expr, keys: Expr = None) -> Expr:
    if keys is not None:
        if not (is_list(data) and is_list(keys) and len(data.args) == len(keys.args)):
            raise ValueError("TabularRow: give a list of values and a list of the same number of keys")
        return Expr("TabularRow", assoc(*zip(keys.args, data.args)))
    if is_assoc(data) or is_list(data):
        raise Unevaluated()
    if isinstance(data, Symbol):
        raise Unevaluated()
    raise ValueError("TabularRow: the data must be an association or a list")


@fn("TabularRowQ", **ADDED, notes="TabularRowQ[x]: True for a valid TabularRow object"
                                  ". Check: reference (differential cases in tests/library).")
def _tabularrowq(x: Expr) -> Boolean:
    return parse_row(x) is not None


# ─────────────────────────── part extraction and object calls ───────────────────────────

def tab_part(t, specs, fname="Part"):
    if not specs:
        return t.expr()
    rs, multi_r = _rows_index(specs[0], t.nrows, fname)
    if len(specs) == 1:
        sub = t.take_rows(rs)
        return sub.expr() if multi_r else t.row_expr(rs[0])
    cspec = specs[1]
    single_c = not (is_sym(cspec, "All") or is_list(cspec) or (isinstance(cspec, Expr) and cspec.head == "Span"))
    js = t.col_indices(cspec, fname) if not single_c else [t.col_index(cspec, fname)]
    if len(specs) > 2:
        if multi_r or not single_c:
            raise ValueError(f"{fname}: deeper parts need a single row and column")
        return ev(Expr("Part", t.cols[js[0]][rs[0]], *specs[2:]))
    if not multi_r and single_c:
        return t.cols[js[0]][rs[0]]
    if not multi_r:
        sub = t.take_cols(js)
        return sub.row_expr(rs[0])
    if single_c:
        return col_expr([t.cols[js[0]][i] for i in rs], t.types[js[0]])
    return t.take_rows(rs).take_cols(js).expr()


def _part_pre(a):
    if not a:
        return _PASS
    x = a[0]
    if isinstance(x, Expr) and x.head == "Tabular":
        t = parse_tab(x)
        if t is not None:
            return tab_part(t, list(a[1:]))
    if isinstance(x, Expr) and x.head == "TabularColumn" and len(a) >= 2:
        c = parse_column(x)
        if c is not None:
            vals, ty = c
            r = ev(Expr("Part", L(*vals), *a[1:]))
            return col_expr(list(r.args), ty) if is_list(r) and not (K.is_int(a[1])) else r
    if isinstance(x, Expr) and x.head == "TabularRow" and len(a) >= 2:
        rv = parse_row(x)
        if rv is not None:
            r = ev(Expr("Part", rv, *a[1:]))
            if is_list(r) or is_assoc(r):
                return Expr("TabularRow", r)
            return r
    return _PASS


wrap_builtin("Part", _part_pre)


def _tabular_call(head, args, ev_):
    t = parse_tab(head)
    if t is None:
        return K.UNHANDLED
    if len(args) == 1:
        k = args[0]
        if isinstance(k, Expr) and k.head == "RowKey":
            i = t.keyed_row(k.args[0] if len(k.args) == 1 else L(*k.args), "Tabular")
            return t.row_expr(i) if i is not None else missing("KeyAbsent", k)
        if isinstance(k, Expr) and k.head == "Key" and len(k.args) == 1 and t.keycols:
            i = t.keyed_row(k.args[0], "Tabular")
            return t.row_expr(i) if i is not None else missing("KeyAbsent", k)
        if is_assoc(k) and t.keycols:
            i = t.keyed_row(k, "Tabular")
            return t.row_expr(i) if i is not None else missing("KeyAbsent", k)
    try:
        from . import wl_dataset as WD
    except ImportError:
        return K.UNHANDLED
    return WD.query_on_tabular(t, list(args))


KL.COMPOUND_HEAD_HOOKS["Tabular"] = _tabular_call


def _column_call(head, args, ev_):
    c = parse_column(head)
    if c is None or len(args) != 1:
        return K.UNHANDLED
    p = args[0]
    if p == "ElementType":
        return c[1]
    if p == "Length":
        return len(c[0])
    if p == "Properties":
        return L("ElementType", "Length", "Properties")
    raise ValueError(f"TabularColumn: {p!r} is not a property (ElementType, Length)")


KL.COMPOUND_HEAD_HOOKS["TabularColumn"] = _column_call


def _row_call(head, args, ev_):
    rv = parse_row(head)
    if rv is None or len(args) != 1:
        return K.UNHANDLED
    k = args[0]
    if is_assoc(rv):
        return aget(rv, k.args[0] if isinstance(k, Expr) and k.head == "Key" else k, missing("KeyAbsent", k))
    if K.is_int(k):
        return ev(Expr("Part", rv, k))
    raise ValueError(f"TabularRow: {k!r} is not a column of the row")


KL.COMPOUND_HEAD_HOOKS["TabularRow"] = _row_call


# ─────────────────────────── Normal, Length, Dimensions and list functions ───────────────────────────

def _normal_pre(a):
    if len(a) == 1:
        x = a[0]
        if isinstance(x, Expr):
            if x.head == "Tabular":
                t = parse_tab(x)
                if t is not None:
                    return L(*t.rows())
            elif x.head == "TabularColumn":
                c = parse_column(x)
                if c is not None:
                    return L(*c[0])
            elif x.head == "TabularRow":
                rv = parse_row(x)
                if rv is not None:
                    return rv
    return _PASS


def _length_pre(a):
    if len(a) == 1 and isinstance(a[0], Expr):
        x = a[0]
        if x.head == "Tabular":
            t = parse_tab(x)
            if t is not None:
                return t.nrows
        if x.head == "TabularColumn":
            c = parse_column(x)
            if c is not None:
                return len(c[0])
        if x.head == "TabularRow":
            rv = parse_row(x)
            if rv is not None:
                return len(rv.args)
    return _PASS


def _dims_pre(a):
    if len(a) == 1 and isinstance(a[0], Expr) and a[0].head == "Tabular":
        t = parse_tab(a[0])
        if t is not None:
            return L(t.nrows, t.ncols)
    return _PASS


wrap_builtin("Normal", _normal_pre)
wrap_builtin("Length", _length_pre)
wrap_builtin("Dimensions", _dims_pre)
wrap_registry("Dimensions", _dims_pre)


def _col_values_pre(name):
    def pre(a):
        if a and isinstance(a[0], Expr) and a[0].head == "TabularColumn":
            c = parse_column(a[0])
            if c is not None:
                return ev(Expr(name, L(*c[0]), *a[1:]))
        return _PASS
    return pre


for _n in ("Total", "Mean", "Min", "Max", "Median", "Accumulate", "Variance", "StandardDeviation", "Counts",
           "Tally", "Union", "DeleteDuplicates", "StringJoin"):
    wrap_builtin(_n, _col_values_pre(_n))
    wrap_registry(_n, _col_values_pre(_n))


def _tab_rows_fn(name):
    def pre(a):
        if not a or not (isinstance(a[0], Expr) and a[0].head == "Tabular"):
            return _PASS
        t = parse_tab(a[0])
        if t is None:
            return _PASS
        n = t.nrows
        if name == "Select":
            if len(a) not in (2, 3):
                return _PASS
            idx = [i for i in range(n) if apply(a[1], t.row_value(i)) is True]
            if len(a) == 3:
                idx = idx[:int(a[2])]
            return t.take_rows(idx).expr()
        if name in ("Take", "Drop"):
            got = ev(Expr(name, L(*range(n)), *a[1:]))
            if not is_list(got):
                return _PASS
            return t.take_rows([int(i) for i in got.args]).expr()
        if name == "Reverse" and len(a) == 1:
            return t.take_rows(list(range(n))[::-1]).expr()
        if name == "SortBy" and len(a) == 2:
            keys = [apply(a[1], t.row_value(i)) for i in range(n)]
            order = sorted(range(n), key=lambda i: K.order_key(keys[i]))
            return t.take_rows(order).expr()
        if name == "Map" and False:
            return _PASS
        return _PASS
    return pre


for _n in ("Select", "Take", "Drop", "Reverse", "SortBy"):
    wrap_builtin(_n, _tab_rows_fn(_n))
    wrap_registry(_n, _tab_rows_fn(_n))


def _join_pre(a):
    if len(a) >= 2 and all(isinstance(x, Expr) and x.head == "Tabular" for x in a):
        ts = [parse_tab(x) for x in a]
        if any(t is None for t in ts):
            return _PASS
        return join_rows(ts, "Join").expr()
    return _PASS


def join_rows(ts, fname):
    t0 = ts[0]
    if any((t.keys is None) != (t0.keys is None) for t in ts):
        raise ValueError(f"{fname}: cannot join named and anonymous tables")
    if t0.keys is None:
        if any(t.ncols != t0.ncols for t in ts):
            raise ValueError(f"{fname}: the tables have different numbers of columns")
        return Tab(None, [sum((t.cols[j] for t in ts), []) for j in range(t0.ncols)])
    keys = list(t0.keys)
    for t in ts[1:]:
        for k in t.keys:
            if k not in keys:
                keys.append(k)
    cols = []
    for k in keys:
        c = []
        for t in ts:
            c.extend(t.cols[t.keys.index(k)] if k in t.keys else [missing("NotAvailable")] * t.nrows)
        cols.append(c)
    return Tab(keys, cols)


wrap_builtin("Join", _join_pre)
wrap_registry("Join", _join_pre)


# ─────────────────────────── column keys and types ───────────────────────────

def _data_keys(x, fname):
    """(keys or None, types or None) of Tabular, TabularRow, Dataset or a list of associations."""
    if isinstance(x, Expr) and x.head == "Tabular":
        t = parse_tab(x)
        if t is not None:
            return t.keys, t.types
    rv = parse_row(x)
    if rv is not None:
        if is_assoc(rv):
            return akeys(rv), [infer_type([v]) for v in avals(rv)]
        return None, [infer_type([v]) for v in rv.args]
    if isinstance(x, Expr) and x.head == "Dataset" and x.args:
        x = x.args[0]
    if is_list(x) and x.args and all(is_assoc(r) for r in x.args):
        t = _rows_to_tab(list(x.args), fname)
        return t.keys, t.types
    if is_list(x) and x.args and all(is_list(r) for r in x.args):
        t = _rows_to_tab(list(x.args), fname)
        return None, t.types
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not tabular data")


def _key_selected(k, t, sel):
    if isinstance(sel, str) or (isinstance(sel, Expr) and sel.head in ("StringExpression", "Alternatives",
                                                                        "RegularExpression")):
        if not isinstance(k, str):
            return False
        from .wl_strings import _smatchq
        return _smatchq(k, sel) is True
    return apply(sel, assoc(("ColumnKey", k), ("ElementType", t))) is True


@fn("ColumnKeys", **ADDED,
    notes="ColumnKeys[data] (None for anonymous columns), ColumnKeys[data, spatt] (string pattern, * wildcards) and "
          "ColumnKeys[data, crit] (crit receives <|\"ColumnKey\" -> k, \"ElementType\" -> t|>) for Tabular, "
          "TabularRow, Dataset and lists of associations"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _columnkeys(data, sel=None):
    keys, types = _data_keys(data, "ColumnKeys")
    if keys is None:
        return NONE
    if sel is None:
        return L(*keys)
    return L(*(k for k, t in zip(keys, types) if _key_selected(k, t, sel)))


@fn("ColumnKeyExistsQ", **ADDED,
    notes="ColumnKeyExistsQ[data, key] for Tabular, Dataset and lists of associations; ColumnKeyExistsQ[key] is "
          "the operator form"
          ". Check: reference (differential cases in tests/library).")
def _columnkeyexistsq(*a: Expr) -> Boolean:
    if len(a) == 1:
        raise Unevaluated()
    if len(a) != 2:
        raise ValueError("ColumnKeyExistsQ: expected ColumnKeyExistsQ[data, key]")
    keys, _ = _data_keys(a[0], "ColumnKeyExistsQ")
    k = a[1].args[0] if isinstance(a[1], Expr) and a[1].head == "Key" and len(a[1].args) == 1 else a[1]
    return keys is not None and k in keys


def _opcall(name, nargs):
    """Operator forms: Name[spec][data] -> Name[data, spec]."""
    def hook(head, args, ev_):
        if not (isinstance(head, Expr) and head.head == name and len(head.args) in nargs and len(args) == 1):
            return K.UNHANDLED
        return ev(Expr(name, args[0], *head.args))
    KL.COMPOUND_HEAD_HOOKS[name] = hook


_opcall("ColumnKeyExistsQ", (1,))


@fn("ColumnTypes", **ADDED,
    notes="ColumnTypes[tab] (an association for named columns, a list otherwise) and ColumnTypes[tab, tsel] with "
          "a type pattern (\"Real*\") or a class (Numbers, Reals, Integers, FloatingPoint, Strings, Lists, ...); "
          "also TabularRow"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _columntypes(data, sel=None):
    keys, types = _data_keys(data, "ColumnTypes")
    idx = [j for j, t in enumerate(types) if sel is None or _type_matches(t, sel)]
    if keys is None:
        return L(*(types[j] for j in idx))
    return assoc(*((keys[j], types[j]) for j in idx))


# ─────────────────────────── column operations ───────────────────────────

def _tab_or_rows(x, fname):
    """(Tab, kind) where kind tells how to give results back: 'tab', 'dataset' or 'rows' (list of associations)."""
    if isinstance(x, Expr) and x.head == "Tabular":
        t = parse_tab(x)
        if t is not None:
            return t, "tab"
    if isinstance(x, Expr) and x.head == "Dataset" and x.args:
        return to_tab(x.args[0], fname), "dataset"
    if is_list(x) and all(is_assoc(r) for r in x.args):
        return _rows_to_tab(list(x.args), fname), "rows"
    if is_list(x) and all(is_list(r) for r in x.args):
        return _rows_to_tab(list(x.args), fname), "rows"
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise ValueError(f"{fname}: {x!r} is not tabular data")


def _give(t, kind):
    if kind == "tab":
        return t.expr()
    rows = L(*t.rows())
    return Expr("Dataset", rows) if kind == "dataset" else rows


@fn("CastColumns", **ADDED,
    notes="CastColumns[tab, col -> type] and CastColumns[tab, {col -> type, ...}] (also an association or a list of "
          "types, one per column): converts values (integers <-> reals, numbers <-> strings, strings to numbers or "
          "dates) or raises when a value cannot be converted")
def _castcolumns(tab: Expr, spec: Expr) -> Expr:
    t, kind = _tab_or_rows(tab, "CastColumns")
    ts = _types_spec(spec, t, "CastColumns")
    cols, types = list(t.cols), list(t.types)
    for j, ty in enumerate(ts):
        if ty is not None:
            cols[j], types[j] = convert_column(t.cols[j], ty, "CastColumns")
    return _give(Tab(t.keys, cols, types, t.keycols, t.options), kind)


_opcall("CastColumns", (1,))


@fn("DeleteColumns", **ADDED,
    notes="DeleteColumns[tab, col] and DeleteColumns[tab, {cols}] (keys, Key, ExtendedKey, positions, spans or "
          "patterns) for Tabular, Dataset and lists of associations; DeleteColumns[cspec] is the operator form")
def _deletecolumns(*a: Expr) -> Expr:
    if len(a) == 1:
        raise Unevaluated()
    if len(a) != 2:
        raise ValueError("DeleteColumns: expected DeleteColumns[tab, cspec]")
    t, kind = _tab_or_rows(a[0], "DeleteColumns")
    spec = a[1]
    drop = set(t.col_indices(spec, "DeleteColumns") if (is_list(spec) or isinstance(spec, Expr)
                                                        and spec.head == "Span")
               else [t.col_index(spec, "DeleteColumns")] if (isinstance(spec, (str, int)) or
                                                             (isinstance(spec, Expr) and spec.head in ("Key",)) or
                                                             t.col_index(spec, "DeleteColumns", False) is not None)
               else t.col_indices(spec, "DeleteColumns"))
    return _give(t.take_cols([j for j in range(t.ncols) if j not in drop]), kind)


_opcall("DeleteColumns", (1,))


def _column_values(v, n, fname):
    c = parse_column(v) if isinstance(v, Expr) and v.head == "TabularColumn" else None
    if c is not None:
        vals = c[0]
    elif is_list(v):
        vals = list(v.args)
    else:
        return [v] * n                     # a constant column
    if len(vals) != n:
        raise ValueError(f"{fname}: the new column has {len(vals)} values but the table has {n} rows")
    return vals


def _put_column(t, key, vals, after=None, fname="InsertColumns"):
    keys = list(t.keys) if t.keys is not None else None
    cols, types = list(t.cols), list(t.types)
    if keys is None:
        raise ValueError(f"{fname}: anonymous tables take a list of new columns, not rules")
    if key in keys:
        j = keys.index(key)
        cols[j], types[j] = vals, infer_type(vals)
    else:
        pos = len(keys) if after is None else t.col_index(after, fname) + 1
        keys.insert(pos, key)
        cols.insert(pos, vals)
        types.insert(pos, infer_type(vals))
    return Tab(keys, cols, types, t.keycols, t.options)


@fn("InsertColumns", **ADDED,
    notes="InsertColumns[tab, col -> vals], InsertColumns[tab, {col1 -> v1, ...}] (an existing column is replaced, "
          "a new one appended; a non-list value is repeated down the column), InsertColumns[tab, {anchor, col -> "
          "vals}] (insert after the anchor column), InsertColumns[atab, {vals1, ...}] for anonymous tables; operator "
          "form InsertColumns[spec]")
def _insertcolumns(*a: Expr) -> Expr:
    if len(a) == 1:
        raise Unevaluated()
    if len(a) != 2:
        raise ValueError("InsertColumns: expected InsertColumns[tab, spec]")
    t, kind = _tab_or_rows(a[0], "InsertColumns")
    spec = a[1]
    n = t.nrows
    if is_rule(spec):
        spec = L(spec)
    if not is_list(spec):
        raise ValueError("InsertColumns: the columns must be given as rules or lists")
    items = list(spec.args)
    if t.keys is None:
        if any(is_rule(x) for x in items):
            raise ValueError("InsertColumns: anonymous tables take a list of new columns, not rules")
        cols = list(t.cols) + [_column_values(v, n, "InsertColumns") for v in items]
        return _give(Tab(None, cols), kind)
    if len(items) == 2 and not is_rule(items[0]) and is_rule(items[1]):
        r = items[1]
        return _give(_put_column(t, r.args[0], _column_values(r.args[1], n, "InsertColumns"), items[0]), kind)
    for r in items:
        if not is_rule(r):
            raise ValueError(f"InsertColumns: {r!r} is not a column -> values rule")
        t = _put_column(t, r.args[0], _column_values(r.args[1], n, "InsertColumns"))
    return _give(t, kind)


_opcall("InsertColumns", (1,))


@fn("RenameColumns", **ADDED,
    notes="RenameColumns[tab, {old -> new, ...}] (old by key or position), RenameColumns[tab, {new1, new2, ...}] "
          "(renames the first columns; also names anonymous columns) and the operator form RenameColumns[spec]")
def _renamecolumns(*a: Expr) -> Expr:
    if len(a) == 1:
        raise Unevaluated()
    if len(a) != 2:
        raise ValueError("RenameColumns: expected RenameColumns[tab, renamings]")
    t, kind = _tab_or_rows(a[0], "RenameColumns")
    spec = a[1]
    if is_rule(spec):
        spec = L(spec)
    if is_assoc(spec):
        spec = L(*spec.args)
    if not is_list(spec):
        raise ValueError("RenameColumns: give a list of renamings or new names")
    keys = list(t.keys) if t.keys is not None else list(range(1, t.ncols + 1))
    if all(is_rule(r) for r in spec.args):
        for r in spec.args:
            keys[t.col_index(r.args[0], "RenameColumns")] = r.args[1]
    elif not any(is_rule(r) for r in spec.args):
        if len(spec.args) > t.ncols:
            raise ValueError("RenameColumns: more names than columns")
        for j, k in enumerate(spec.args):
            keys[j] = k
        if t.keys is None and len(spec.args) < t.ncols:
            raise ValueError("RenameColumns: give a name for every column of an anonymous table")
    else:
        raise ValueError("RenameColumns: do not mix renaming rules and plain names")
    kc = None
    if t.keycols:
        kc = [keys[t.keys.index(k)] for k in t.keycols]
    return _give(Tab(keys, t.cols, t.types, kc, t.options), kind)


_opcall("RenameColumns", (1,))


# columnwise evaluation inside transformation functions

def _function_parts(f):
    """(params or None, body) of a Function expression, else None."""
    if isinstance(f, Expr) and f.head == "Function":
        if len(f.args) == 1:
            return None, f.args[0]
        if len(f.args) >= 2:
            return f.args[0], f.args[1]
    return None


def _has_head(e, heads):
    if isinstance(e, Expr):
        if e.head in heads:
            return True
        if isinstance(e.head, Expr) and _has_head(e.head, heads):
            return True
        return any(_has_head(a, heads) for a in e.args)
    return False


def _subst_slots_columns(e, colassoc):
    """Replace #key / Slot["key"] / #[...] inside a ColumnwiseThread/ColumnwiseValue body by the whole columns."""
    if isinstance(e, Expr):
        if e.head == "Slot" and len(e.args) == 1 and isinstance(e.args[0], str):
            return aget(colassoc, e.args[0], missing("KeyAbsent", e.args[0]))
        if e.head == "Slot" and e.args in ((), (1,)):
            return colassoc
        if e.head == "Function":
            return e
        return Expr(_subst_slots_columns(e.head, colassoc) if isinstance(e.head, Expr) else e.head,
                    *(_subst_slots_columns(a, colassoc) for a in e.args))
    return e


def _columnwise(f, t):
    """Evaluate ColumnwiseThread/ColumnwiseValue parts of f once on the columns; returns (f', per-row values)."""
    fp = _function_parts(f)
    if fp is None or not _has_head(fp[1], ("ColumnwiseThread", "ColumnwiseValue")):
        return f, None
    params, body = fp
    if params is not None:
        raise ValueError("ColumnwiseThread/ColumnwiseValue need a slot function (#col)")
    colassoc = assoc(*((k, L(*c)) for k, c in zip(t.keys or range(1, t.ncols + 1), t.cols)))
    threaded = []

    def rewrite(e):
        if isinstance(e, Expr):
            if e.head in ("ColumnwiseThread", "ColumnwiseValue") and len(e.args) == 1:
                v = ev(_subst_slots_columns(e.args[0], colassoc))
                if e.head == "ColumnwiseThread":
                    vals = parse_column(v)[0] if parse_column(v) is not None else (list(v.args) if is_list(v)
                                                                                    else None)
                    if vals is None or len(vals) != t.nrows:
                        raise ValueError("ColumnwiseThread: the columnwise result must be a list with one value per "
                                         "row")
                    threaded.append(vals)
                    return Expr("Slot", f"__cw{len(threaded) - 1}")
                return Expr("Unevaluated", v) if False else v
            if e.head == "Function":
                return e
            return Expr(rewrite(e.head) if isinstance(e.head, Expr) else e.head, *(rewrite(a) for a in e.args))
        return e
    new_body = rewrite(body)
    return Expr("Function", new_body), threaded


def _row_fn_values(t, f, fname):
    f2, threaded = _columnwise(f, t)
    out = []
    for i in range(t.nrows):
        rv = t.row_value(i)
        if threaded:
            extra = [Expr("Rule", f"__cw{j}", vals[i]) for j, vals in enumerate(threaded)]
            rv = Expr("Association", *(list(rv.args) if is_assoc(rv) else
                                        [Expr("Rule", j + 1, v) for j, v in enumerate(rv.args)]) + extra)
        out.append(apply(f2, rv))
    return out


@fn("TransformColumns", **ADDED,
    notes="TransformColumns[tab, col -> f] and TransformColumns[tab, {col1 -> f1, ...}]: f receives each row as an "
          "association (#key), existing columns are replaced in place, new ones appended, rules applied in "
          "sequence; ColumnwiseThread/ColumnwiseValue parts of f are evaluated on whole columns; col -> Nothing "
          "deletes a column; operator form TransformColumns[spec]")
def _transformcolumns(*a: Expr) -> Expr:
    if len(a) == 1:
        raise Unevaluated()
    if len(a) != 2:
        raise ValueError("TransformColumns: expected TransformColumns[tab, col -> f]")
    t, kind = _tab_or_rows(a[0], "TransformColumns")
    spec = a[1]
    rules = [spec] if is_rule(spec) else (list(spec.args) if is_list(spec) or is_assoc(spec) else None)
    if rules is None or not all(is_rule(r) for r in rules):
        raise ValueError("TransformColumns: give col -> f or a list of such rules")
    for r in rules:
        k, f = r.args
        if is_sym(f, "Nothing"):
            j = t.col_index(k, "TransformColumns")
            t = t.take_cols([i for i in range(t.ncols) if i != j])
            continue
        vals = _row_fn_values(t, f, "TransformColumns")
        if t.keys is None:
            j = t.col_index(k, "TransformColumns")
            cols, types = list(t.cols), list(t.types)
            cols[j], types[j] = vals, infer_type(vals)
            t = Tab(None, cols, types)
        else:
            t = _put_column(t, k, vals, fname="TransformColumns")
    return _give(t, kind)


_opcall("TransformColumns", (1,))


@fn("ConstructColumns", **ADDED,
    notes="ConstructColumns[tab, {col, new -> f, ...}] keeps only the listed columns (existing columns by key or "
          "position, new ones computed row by row from #key, with ColumnwiseThread/ColumnwiseValue support), in the "
          "order given; ConstructColumns[tab, col] and the operator form ConstructColumns[spec]")
def _constructcolumns(*a: Expr) -> Expr:
    if len(a) == 1:
        raise Unevaluated()
    if len(a) != 2:
        raise ValueError("ConstructColumns: expected ConstructColumns[tab, spec]")
    t, kind = _tab_or_rows(a[0], "ConstructColumns")
    spec = a[1]
    items = list(spec.args) if is_list(spec) else [spec]
    keys, cols, types = [], [], []
    for x in items:
        if is_rule(x):
            k, f = x.args
            if isinstance(f, (str,)) or (isinstance(f, Expr) and f.head == "Key"):
                j = t.col_index(f, "ConstructColumns")
                vals, ty = t.cols[j], t.types[j]
            else:
                vals = _row_fn_values(t, f, "ConstructColumns")
                ty = infer_type(vals)
        else:
            j = t.col_index(x, "ConstructColumns")
            k = t.keys[j] if t.keys is not None else x
            vals, ty = t.cols[j], t.types[j]
        keys.append(k)
        cols.append(vals)
        types.append(ty)
    kc = [k for k in (t.keycols or []) if k in keys] or None
    return _give(Tab(keys, cols, types, kc, t.options), kind)


_opcall("ConstructColumns", (1,))


@fn("ColumnwiseThread", **ADDED,
    notes="ColumnwiseThread[body] inside a Function given to TransformColumns or ConstructColumns: body is evaluated "
          "once with #col standing for the whole column and its elements are used row by row; inert elsewhere")
def _columnwisethread(*a: Expr) -> Expr:
    raise Unevaluated()


@fn("ColumnwiseValue", **ADDED,
    notes="ColumnwiseValue[body] inside a Function given to TransformColumns or ConstructColumns: body is evaluated "
          "once on whole columns (#col) and the value is used in every row; inert elsewhere")
def _columnwisevalue(*a: Expr) -> Expr:
    raise Unevaluated()


# ─────────────────────────── aggregation and pivoting ───────────────────────────

def _group_indices(t, gcols):
    groups = {}
    order = []
    for i in range(t.nrows):
        kv = tuple(t.cols[j][i] for j in gcols)
        rk = repr(kv)
        if rk not in groups:
            groups[rk] = (kv, [])
            order.append(rk)
        groups[rk][1].append(i)
    return [groups[k] for k in order]


def _col_assoc(t, idx):
    keys = t.keys if t.keys is not None else list(range(1, t.ncols + 1))
    return assoc(*((k, L(*(c[i] for i in idx))) for k, c in zip(keys, t.cols)))


def _agg_rules(spec, fname):
    if is_rule(spec):
        return [spec]
    if (is_list(spec) or is_assoc(spec)) and all(is_rule(r) for r in spec.args):
        return list(spec.args)
    raise ValueError(f"{fname}: the aggregations must be key -> f rules")


@fn("AggregateRows", **ADDED,
    notes="AggregateRows[tab, key -> f | {key -> f, ...}] and AggregateRows[tab, fspec, gcols] (groups by the "
          "values of the columns gcols, which become key columns, in order of first appearance); f receives the "
          "association of the (sub)table's columns as lists (#col); lists of associations give lists of "
          "associations; operator forms AggregateRows[fspec] and AggregateRows[fspec, gspec]")
def _aggregaterows(*a: Expr) -> Expr:
    if len(a) == 1 or (len(a) == 2 and not _is_tabular_data(a[0])):
        raise Unevaluated()
    if len(a) not in (2, 3):
        raise ValueError("AggregateRows: expected AggregateRows[tab, fspec] or AggregateRows[tab, fspec, gspec]")
    t, kind = _tab_or_rows(a[0], "AggregateRows")
    rules = _agg_rules(a[1], "AggregateRows")
    if len(a) == 2:
        ca = _col_assoc(t, range(t.nrows))
        vals = [apply(r.args[1], ca) for r in rules]
        res = Tab([r.args[0] for r in rules], [[v] for v in vals])
        return _give(res, kind)
    g = a[2]
    gl = list(g.args) if is_list(g) else [g]
    gcols = [t.col_index(c, "AggregateRows") for c in gl]
    gkeys = [t.keys[j] if t.keys is not None else j + 1 for j in gcols]
    out_cols = [[] for _ in gcols] + [[] for _ in rules]
    for kv, idx in _group_indices(t, gcols):
        ca = _col_assoc(t, idx)
        for j, v in enumerate(kv):
            out_cols[j].append(v)
        for j, r in enumerate(rules):
            out_cols[len(gcols) + j].append(apply(r.args[1], ca))
    res = Tab(gkeys + [r.args[0] for r in rules], out_cols, None, gkeys if kind == "tab" else None)
    return _give(res, kind)


def _is_tabular_data(x):
    return (isinstance(x, Expr) and x.head in ("Tabular", "Dataset")) or \
        (is_list(x) and bool(x.args) and all(is_assoc(r) or is_list(r) for r in x.args))


def _agg_op(head, args, ev_):
    if isinstance(head, Expr) and head.head == "AggregateRows" and len(head.args) in (1, 2) and len(args) == 1 \
            and not _is_tabular_data(head.args[0]):
        return ev(Expr("AggregateRows", args[0], *head.args))
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["AggregateRows"] = _agg_op


def _pivot_to(t, varcol, valcols, fname):
    vj = t.col_index(varcol, fname)
    vjs = [t.col_index(v, fname) for v in valcols]
    idj = [j for j in range(t.ncols) if j != vj and j not in vjs]
    vars_ = []
    for v in t.cols[vj]:
        if v not in vars_:
            vars_.append(v)
    groups = _group_indices(t, idj)
    keys = [t.keys[j] for j in idj]
    cols = [[kv[n] for kv, _ in groups] for n in range(len(idj))]
    for vk in vjs:
        for var in vars_:
            col = []
            for kv, idx in groups:
                hit = [i for i in idx if t.cols[vj][i] == var]
                if len(hit) > 1:
                    raise ValueError(f"{fname}: several rows share the identifying values {list(kv)!r} and "
                                     f"{varcol!r} = {var!r}")
                col.append(t.cols[vk][hit[0]] if hit else missing("NotAvailable"))
            keys.append(Expr("ExtendedKey", t.keys[vk], var))
            cols.append(col)
    return Tab(keys, cols)


@fn("PivotToColumns", **ADDED,
    notes="PivotToColumns[tab, varcol -> valcol], varcol -> {valcol1, ...} and lists of such rules: spreads the "
          "values of valcol into new columns ExtendedKey[valcol, var] for each distinct var (first-appearance "
          "order); the remaining columns identify the rows; absent combinations give Missing[\"NotAvailable\"]")
def _pivottocolumns(tab: Expr, spec: Expr) -> Expr:
    t, kind = _tab_or_rows(tab, "PivotToColumns")
    if t.keys is None:
        raise ValueError("PivotToColumns: the table needs named columns")
    rules = [spec] if is_rule(spec) else (list(spec.args) if is_list(spec) else None)
    if not rules or not all(is_rule(r) for r in rules):
        raise ValueError("PivotToColumns: give varcol -> valcol rules")
    for r in rules:
        vals = list(r.args[1].args) if is_list(r.args[1]) else [r.args[1]]
        t = _pivot_to(t, r.args[0], vals, "PivotToColumns")
    return _give(t, kind)


@fn("PivotFromColumns", **ADDED,
    notes="PivotFromColumns[tab, {c1, c2, ...} -> {var, val}] (and lists of such rules): gathers the columns ci into "
          "a column var of their keys and a column val of their values, row by row; ExtendedKey[k, v] column keys "
          "can be gathered with a key pattern")
def _pivotfromcolumns(tab: Expr, spec: Expr) -> Expr:
    t, kind = _tab_or_rows(tab, "PivotFromColumns")
    if t.keys is None:
        raise ValueError("PivotFromColumns: the table needs named columns")
    rules = [spec] if is_rule(spec) else (list(spec.args) if is_list(spec) else None)
    if not rules or not all(is_rule(r) for r in rules):
        raise ValueError("PivotFromColumns: give {cols} -> {var, val} rules")
    for r in rules:
        cs, target = r.args
        if not (is_list(target) and len(target.args) == 2):
            raise ValueError("PivotFromColumns: the target must be {varcol, valcol}")
        js = t.col_indices(cs, "PivotFromColumns") if is_list(cs) else t.col_indices(cs, "PivotFromColumns")
        rest = [j for j in range(t.ncols) if j not in js]
        keys = [t.keys[j] for j in rest] + list(target.args)
        cols = [[] for _ in keys]
        for i in range(t.nrows):
            for j in js:
                for n, rj in enumerate(rest):
                    cols[n].append(t.cols[rj][i])
                cols[-2].append(t.keys[j])
                cols[-1].append(t.cols[j][i])
        t = Tab(keys, cols)
    return _give(t, kind)


@fn("PivotTable", **ADDED,
    notes="PivotTable[tab, f, rowcols, colcols] and PivotTable[tab, {key -> f, ...}, rowcols, colcols]: aggregates "
          "each (row, column) group with f (which receives the group's columns as lists) and spreads the results "
          "into columns (the colcols value, or ExtendedKey[...] for several colcols or functions); rowcols become "
          "key columns; IncludeGroupAggregates is not supported")
def _pivottable(tab: Expr, f: Expr, rowcols: Expr, colcols: Expr, *opts: Option) -> Expr:
    for o in opts:
        if is_rule(o) and is_sym(o.args[0], "IncludeGroupAggregates") and o.args[1] is False:
            continue
        raise NotImplementedError("PivotTable: the IncludeGroupAggregates option is not supported")
    t, kind = _tab_or_rows(tab, "PivotTable")
    rl = list(rowcols.args) if is_list(rowcols) else [rowcols]
    cl = list(colcols.args) if is_list(colcols) else [colcols]
    multi_f = not (is_rule(f) is False and not is_list(f)) and (is_list(f) and all(is_rule(r) for r in f.args)
                                                                or is_rule(f))
    rules = _agg_rules(f, "PivotTable") if multi_f else [Expr("Rule", "__value", f)]
    rj = [t.col_index(c, "PivotTable") for c in rl]
    cj = [t.col_index(c, "PivotTable") for c in cl]
    rkeys = [t.keys[j] for j in rj]
    rgroups = _group_indices(t, rj)
    cvals = [kv for kv, _ in _group_indices(t, cj)]
    keys = list(rkeys)
    cols = [[kv[n] for kv, _ in rgroups] for n in range(len(rj))]
    for r in rules:
        for cv in cvals:
            col = []
            for kv, idx in rgroups:
                sub = [i for i in idx if tuple(t.cols[j][i] for j in cj) == cv or
                       repr(tuple(t.cols[j][i] for j in cj)) == repr(cv)]
                col.append(apply(r.args[1], _col_assoc(t, sub)) if sub else missing("NotAvailable"))
            if multi_f:
                keys.append(Expr("ExtendedKey", r.args[0], *cv))
            else:
                keys.append(cv[0] if len(cv) == 1 else Expr("ExtendedKey", *cv))
            cols.append(col)
    return _give(Tab(keys, cols, None, rkeys if kind == "tab" else None), kind)


_added("IncludeGroupAggregates", "option of PivotTable; only False (the default) is supported") \
    if "IncludeGroupAggregates" not in REGISTRY else None


# ─────────────────────────── conversion: ToTabular, FromTabular ───────────────────────────

@fn("ToTabular", **ADDED,
    notes="ToTabular[data] and ToTabular[data, \"Rows\"] (matrix or list of associations), ToTabular[data, "
          "\"Columns\"] (a list of columns, an association or list of key -> column rules, or TabularColumns), "
          "ToTabular[data, \"Dataset\"], ToTabular[data, form, <|\"ColumnKeys\" -> ..., \"HeaderRows\" -> n|>]; "
          "TimeSeries, TemporalData and SpatialPointData forms are not supported")
def _totabular(data: Expr, form: Expr = "Rows", spec: Expr = None) -> Expr:
    if form not in ("Rows", "Columns", "Dataset"):
        raise NotImplementedError(f"ToTabular: the form {form!r} is not supported (Rows, Columns, Dataset)")
    keys_override, header = None, 0
    if spec is not None:
        if not is_assoc(spec):
            raise ValueError("ToTabular: the third argument must be an association")
        for r in spec.args:
            k, v = r.args
            if k == "ColumnKeys":
                if not is_list(v):
                    raise ValueError("ToTabular: \"ColumnKeys\" must be a list")
                keys_override = list(v.args)
            elif k == "HeaderRows":
                if not (K.is_int(v) and int(v) in (0, 1)):
                    raise NotImplementedError("ToTabular: \"HeaderRows\" must be 0 or 1")
                header = int(v)
            elif k == "CacheOriginalExpression":
                pass
            else:
                raise NotImplementedError(f"ToTabular: the directive {k!r} is not supported")
    if form == "Columns":
        if is_assoc(data) or (is_list(data) and data.args and all(is_rule(r) for r in data.args)):
            rules = list(data.args)
            keys = [r.args[0] for r in rules]
            cols = [_column_values(r.args[1], None, "ToTabular") if False else
                    (parse_column(r.args[1])[0] if parse_column(r.args[1]) is not None else list(r.args[1].args))
                    for r in rules]
            types = [parse_column(r.args[1])[1] if parse_column(r.args[1]) is not None else None for r in rules]
            t = Tab(keys, cols, [ty if ty is not None else infer_type(c) for ty, c in zip(types, cols)])
        elif is_list(data) and all(is_list(c) or parse_column(c) is not None for c in data.args):
            cols = [parse_column(c)[0] if parse_column(c) is not None else list(c.args) for c in data.args]
            t = Tab(None, cols)
        else:
            raise ValueError("ToTabular: \"Columns\" data must be a list of columns or key -> column rules")
    else:
        if header and is_list(data) and data.args:
            keys_override = list(data.args[0].args)
            data = L(*data.args[1:])
        t = to_tab(data, "ToTabular")
    if keys_override is not None:
        if len(keys_override) != t.ncols:
            raise ValueError("ToTabular: give one column key per column")
        t = Tab(keys_override, t.cols, t.types)
    return t.expr()


@fn("FromTabular", **ADDED,
    notes="FromTabular[tab, \"Rows\"] (list of associations, or lists for anonymous columns), \"Columns\" "
          "(association of TabularColumn objects, or a list), \"Matrix\" (list of rows of values), \"Dataset\"; "
          "TimeSeries/EventSeries/TemporalData forms are not supported"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _fromtabular(tab: Expr, form: Expr = "Rows", spec: Expr = None) -> Expr:
    t = parse_tab(tab) if isinstance(tab, Expr) and tab.head == "Tabular" else None
    if t is None:
        if isinstance(tab, Symbol):
            raise Unevaluated()
        raise ValueError("FromTabular: the first argument must be a Tabular object")
    if form == "Rows":
        return L(*t.rows())
    if form == "Columns":
        cs = [t.colexpr(j) for j in range(t.ncols)]
        return L(*cs) if t.keys is None else assoc(*zip(t.keys, cs))
    if form == "Matrix":
        return L(*(L(*t.row(i)) for i in range(t.nrows)))
    if form == "Dataset":
        return Expr("Dataset", L(*t.rows()))
    raise NotImplementedError(f"FromTabular: the form {form!r} is not supported (Rows, Columns, Matrix, Dataset)")


# ─────────────────────────── schema, structure, summary ───────────────────────────

def _schema_expr(t):
    if t.keys is None:
        props = L(*(assoc(("ElementType", ty)) for ty in t.types))
    else:
        props = assoc(*((k, assoc(("ElementType", ty))) for k, ty in zip(t.keys, t.types)))
    return Expr("TabularSchema", assoc(("ColumnProperties", props),
                                       ("KeyColumns", L(*t.keycols) if t.keycols else NONE),
                                       ("Backend", "WolframKernel")))


def _parse_schema(x):
    """(keys or None, types, keycols, backend) from TabularSchema[<|...|>]."""
    if not (isinstance(x, Expr) and x.head == "TabularSchema" and len(x.args) == 1 and is_assoc(x.args[0])):
        return None
    a = x.args[0]
    props = aget(a, "ColumnProperties")
    if is_assoc(props):
        keys = akeys(props)
        types = [aget(p, "ElementType", "InertExpression") if is_assoc(p) else "InertExpression"
                 for p in avals(props)]
    elif is_list(props):
        keys = None
        types = [aget(p, "ElementType", "InertExpression") if is_assoc(p) else "InertExpression" for p in props.args]
    else:
        return None
    kc = aget(a, "KeyColumns")
    return keys, types, (list(kc.args) if is_list(kc) else None), aget(a, "Backend", "WolframKernel")


def _schema_from(keys, types, kc, backend="WolframKernel"):
    if keys is None:
        props = L(*(assoc(("ElementType", ty)) for ty in types))
    else:
        props = assoc(*((k, assoc(("ElementType", ty))) for k, ty in zip(keys, types)))
    return Expr("TabularSchema", assoc(("ColumnProperties", props), ("KeyColumns", L(*kc) if kc else NONE),
                                       ("Backend", backend)))


_SCHEMA_GLOBAL = ["Columns", "ColumnKeys", "ColumnTypes", "KeyColumns", "Backend", "DefaultColumnBaseName",
                  "ColumnProperties", "Properties"]


@fn("TabularSchema", **ADDED,
    notes="TabularSchema[tab] extracts the schema; TabularSchema[<|\"ColumnProperties\" -> ..., \"KeyColumns\" -> "
          "..., \"Backend\" -> ...|>] is the schema object; TabularSchema[schema, \"prop\" -> val] and "
          "TabularSchema[schema, <|...|>] modify ColumnKeys, KeyColumns, ColumnTypes or \"ElementType\" -> col -> "
          "type; schema[\"prop\"] (Columns, ColumnKeys, ColumnTypes, KeyColumns, Backend, ColumnProperties, "
          "Properties) and schema[col, \"ElementType\" | \"Nullable\" | \"Properties\"]; ColumnKeys/ColumnTypes "
          "accept schemas; Tabular[data, schema] applies one"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _tabularschema(x: Expr, *mods: Expr) -> Expr:
    if isinstance(x, Expr) and x.head == "Tabular":
        t = parse_tab(x)
        if t is None:
            raise ValueError("TabularSchema: not a valid Tabular object")
        if mods:
            raise ValueError("TabularSchema: expected TabularSchema[tab]")
        return _schema_expr(t)
    if is_assoc(x) and not mods:
        s = _parse_schema(Expr("TabularSchema", x))
        if s is None:
            raise ValueError("TabularSchema: the association needs \"ColumnProperties\"")
        canon = _schema_from(*s)
        if canon == Expr("TabularSchema", x):
            raise Unevaluated()
        return canon
    s = _parse_schema(x)
    if s is None:
        if isinstance(x, Symbol):
            raise Unevaluated()
        raise ValueError("TabularSchema: expected a Tabular object, a schema association or a TabularSchema")
    keys, types, kc, backend = s
    rules = []
    for m in mods:
        if is_rule(m):
            rules.append(m)
        elif is_assoc(m):
            rules.extend(m.args)
        else:
            raise ValueError("TabularSchema: modifications must be \"prop\" -> value rules or an association")
    for r in rules:
        k, v = r.args
        if k == "ColumnKeys":
            if not (is_list(v) and len(v.args) == len(types)):
                raise ValueError("TabularSchema: \"ColumnKeys\" needs one key per column")
            if kc and keys:
                kc = [v.args[keys.index(c)] for c in kc]
            keys = list(v.args)
        elif k == "KeyColumns":
            kc = None if is_sym(v, "None") else (list(v.args) if is_list(v) else [v])
        elif k in ("ElementType", "ColumnTypes"):
            items = [v] if is_rule(v) else (list(v.args) if (is_list(v) or is_assoc(v)) else None)
            if items is None:
                raise ValueError("TabularSchema: give col -> type")
            if items and not is_rule(items[0]):
                if len(items) != len(types):
                    raise ValueError("TabularSchema: give one type per column")
                types = [norm_type(ty, "TabularSchema") for ty in items]
            else:
                for it in items:
                    c, ty = it.args
                    j = keys.index(c) if keys and c in keys else (int(c) - 1 if K.is_int(c) else None)
                    if j is None or not 0 <= j < len(types):
                        raise ValueError(f"TabularSchema: there is no column {c!r}")
                    types[j] = norm_type(ty, "TabularSchema")
        elif k == "Backend":
            backend = v
        else:
            raise ValueError(f"TabularSchema: {k!r} cannot be modified")
    return _schema_from(keys, types, kc, backend)


def _schema_hook(head, args, ev_):
    s = _parse_schema(head)
    if s is None:
        return K.UNHANDLED
    keys, types, kc, backend = s
    if len(args) == 1:
        p = args[0]
        if p == "Columns":
            return L(*(keys if keys is not None else range(1, len(types) + 1)))
        if p == "ColumnKeys":
            return L(*keys) if keys is not None else NONE
        if p == "ColumnTypes":
            return assoc(*zip(keys, types)) if keys is not None else L(*types)
        if p == "KeyColumns":
            return L(*kc) if kc else NONE
        if p == "Backend":
            return backend
        if p == "DefaultColumnBaseName":
            return "Column"
        if p == "ColumnProperties":
            return aget(head.args[0], "ColumnProperties")
        if p == "Properties":
            return L(*_SCHEMA_GLOBAL)
        raise ValueError(f"TabularSchema: {p!r} is not a schema property")
    if len(args) == 2:
        c, p = args
        j = keys.index(c) if keys and c in keys else (int(c) - 1 if K.is_int(c) and keys is None else None)
        if j is None or not 0 <= j < len(types):
            raise ValueError(f"TabularSchema: there is no column {c!r}")
        if p == "ElementType":
            return types[j]
        if p == "Nullable":
            return True
        if p == "Properties":
            return L("ElementType", "Nullable", "Properties")
        raise ValueError(f"TabularSchema: {p!r} is not a column property")
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["TabularSchema"] = _schema_hook


def _schema_keys_pre(name):
    def pre(a):
        if a and _parse_schema(a[0]) is not None:
            keys, types, _, _ = _parse_schema(a[0])
            fake = Tab(keys, [[] for _ in types], types)
            return ev(Expr(name, fake.expr(), *a[1:]))
        return _PASS
    return pre


wrap_registry("ColumnKeys", _schema_keys_pre("ColumnKeys"))
wrap_registry("ColumnTypes", _schema_keys_pre("ColumnTypes"))


def _select_columns(t, sel, fname):
    if sel is None or is_sym(sel, "All"):
        return list(range(t.ncols))
    if is_list(sel):
        return [t.col_index(c, fname) for c in sel.args]
    keys = t.keys if t.keys is not None else list(range(1, t.ncols + 1))
    if isinstance(sel, str) and sel in keys:
        return [keys.index(sel)]
    if isinstance(sel, str) and sel in ("Numbers", "Integers", "Reals", "Strings", "Lists", "Booleans", "Dates",
                                        "Times", "Nominals", "Ordinals", "InertExpressions") or \
            (isinstance(sel, str) and (sel in _SIMPLE or sel in _ALIASES)):
        cls = {"Booleans": lambda ty: ty == "Boolean", "Dates": lambda ty: ts_name(ty) == "Date",
               "Times": lambda ty: ts_name(ty) == "Time", "Nominals": lambda ty: ts_name(ty) == "Nominal",
               "Ordinals": lambda ty: ts_name(ty) == "Ordinal",
               "InertExpressions": lambda ty: ty == "InertExpression"}
        if sel in cls:
            return [j for j in range(t.ncols) if cls[sel](t.types[j])]
        if sel in _SIMPLE or sel in _ALIASES:
            return [j for j in range(t.ncols) if t.types[j] == _ALIASES.get(sel, sel)]
        return [j for j in range(t.ncols) if _type_matches(t.types[j], sel)]
    return [j for j in range(t.ncols) if _key_selected(keys[j], t.types[j], sel)]


def _distinct(vals):
    seen = []
    for v in vals:
        if v not in seen:
            seen.append(v)
    return seen


def _struct_prop(t, j, p):
    c = t.cols[j]
    if p == "ColumnType":
        return t.types[j]
    if p == "MissingCount":
        return sum(1 for v in c if is_missing(v))
    if p == "NonMissingCount":
        return sum(1 for v in c if not is_missing(v))
    if p == "ByteCount":
        return ev(Expr("ByteCount", t.colexpr(j)))
    if p == "DuplicateFreeQ":
        return len(_distinct(c)) == len(c)
    if p == "CountDistinct":
        return len(_distinct(c))
    if p == "UniqueCount":
        return len(_distinct([v for v in c if not is_missing(v)]))
    raise ValueError(f"TabularStructure: {p!r} is not a property")


_STRUCT_ALL = ["ColumnType", "MissingCount", "NonMissingCount", "ByteCount", "DuplicateFreeQ", "CountDistinct"]
_STRUCT_AUTO = ["ColumnType", "NonMissingCount", "MissingCount", "ByteCount"]


@fn("TabularStructure", **ADDED,
    notes="TabularStructure[tab], TabularStructure[tab, cols] and TabularStructure[tab, cols, props]: a Tabular with "
          "key column \"ColumnKey\" and the properties ColumnType, NonMissingCount, MissingCount, ByteCount "
          "(Automatic), DuplicateFreeQ, CountDistinct (All); cols as All, a list, a string pattern or a criterion; "
          "ByteCount is this implementation's size of the column")
def _tabularstructure(tab: Expr, cols: Expr = None, props: Expr = None) -> Expr:
    t, _ = _tab_or_rows(tab, "TabularStructure")
    js = _select_columns(t, cols, "TabularStructure")
    if props is None or is_sym(props, "Automatic"):
        ps = _STRUCT_AUTO
    elif is_sym(props, "All"):
        ps = _STRUCT_ALL
    elif isinstance(props, str):
        ps = [props]
    elif is_list(props):
        ps = list(props.args)
    else:
        raise ValueError("TabularStructure: bad property specification")
    for p in ps:
        if p not in _STRUCT_ALL and p != "UniqueCount":
            raise ValueError(f"TabularStructure: {p!r} is not a property")
    keys = t.keys if t.keys is not None else list(range(1, t.ncols + 1))
    out = Tab(["ColumnKey"] + ps, [[keys[j] for j in js]] + [[_struct_prop(t, j, p) for j in js] for p in ps],
              None, ["ColumnKey"])
    return out.expr()


_STAT_PROPS = {"Mean": "Mean", "Min": "Min", "Max": "Max", "Median": "Median", "StandardDeviation":
               "StandardDeviation", "Variance": "Variance", "Total": "Total"}
_QUART = {"LowerQuartile": 1, "FirstQuartile": 1, "Q1": 1, "UpperQuartile": 3, "LastQuartile": 3, "Q3": 3}
_STAT_GROUP = ["Mean", "StandardDeviation", "Min", "Q1", "Median", "Q3", "Max"]
_STRUCT_GROUP = ["ColumnType", "MissingCount", "NonMissingCount", "ByteCount", "DuplicateFreeQ", "CountDistinct",
                 "UniqueCount"]


def _numeric_type(ty):
    n = ts_name(ty)
    return n in _INT_TYPES or n in ("Real32", "Real64")


def _summary_value(t, j, p):
    if p in _STRUCT_GROUP:
        return _struct_prop(t, j, p)
    vals = [v for v in t.cols[j] if not is_missing(v)]
    if not _numeric_type(t.types[j]) or not vals:
        return missing("NotApplicable")
    if p in _STAT_PROPS:
        return ev(Expr(_STAT_PROPS[p], L(*vals)))
    if p in _QUART:
        return ev(Expr("Quantile", L(*vals), Expr("Rational", _QUART[p], 4)))
    raise ValueError(f"TabularSummary: {p!r} is not a supported property")


@fn("TabularSummary", **ADDED,
    notes="TabularSummary[tab], TabularSummary[tab, cols], TabularSummary[tab, cols -> props] and lists of such "
          "rules: a Tabular keyed by \"ColumnKey\" with ExtendedKey[\"Statistical\", p] columns (Mean, "
          "StandardDeviation, Min, Q1, Median, Q3, Max; default Min, Median, Max) and ExtendedKey[\"Structural\", "
          "p] columns; non-numeric columns give Missing[\"NotApplicable\"]; the graphical properties (CategoryPlot, "
          "Histogram, BoxWhiskerChart) are omitted")
def _tabularsummary(tab: Expr, spec: Expr = None) -> Expr:
    t, _ = _tab_or_rows(tab, "TabularSummary")
    pairs = []
    rules = [spec] if is_rule(spec) else (list(spec.args) if is_list(spec) and spec.args and
                                          all(is_rule(r) for r in spec.args) else None)
    if rules is None:
        js = _select_columns(t, spec, "TabularSummary")
        pairs = [(js, ["Min", "Median", "Max"])]
    else:
        for r in rules:
            js = _select_columns(t, r.args[0], "TabularSummary")
            p = r.args[1]
            if p == "Statistical":
                ps = _STAT_GROUP
            elif p == "Structural":
                ps = ["ColumnType", "MissingCount", "NonMissingCount", "CountDistinct"]
            elif isinstance(p, str):
                ps = [p]
            elif is_list(p):
                ps = []
                for q in p.args:
                    ps.extend(_STAT_GROUP if q == "Statistical" else [q])
            else:
                raise ValueError("TabularSummary: bad property specification")
            for q in ps:
                if q in ("CategoryPlot", "Histogram", "BoxWhiskerChart", "Graphical"):
                    raise NotImplementedError("TabularSummary: graphical properties are not supported")
                if q not in _STAT_PROPS and q not in _QUART and q not in _STRUCT_GROUP:
                    raise ValueError(f"TabularSummary: {q!r} is not a property")
            pairs.append((js, ps))
    allj, allp = [], []
    for js, ps in pairs:
        for j in js:
            if j not in allj:
                allj.append(j)
        for p in ps:
            if p not in allp:
                allp.append(p)
    keys = t.keys if t.keys is not None else list(range(1, t.ncols + 1))
    ckeys = ["ColumnKey"] + [Expr("ExtendedKey", "Structural" if p in _STRUCT_GROUP else "Statistical", p)
                             for p in allp]
    cols = [[keys[j] for j in allj]]
    for p in allp:
        cols.append([_summary_value(t, j, p) if any(j in js and p in ps for js, ps in pairs)
                     else missing("NotApplicable") for j in allj])
    return Tab(ckeys, cols, None, ["ColumnKey"]).expr()


# ─────────────────────────── keys ───────────────────────────

@fn("RowKey", **ADDED,
    notes="RowKey[{k1, ...}] (RowKey[k] = RowKey[{k}]) selects the row of a Tabular with key columns whose key "
          "values are ki: tab[RowKey[...]] or RowKey[...][tab]")
def _rowkey(*a: Expr) -> Expr:
    if len(a) == 1 and not is_list(a[0]):
        return Expr("RowKey", L(a[0]))
    if len(a) == 1:
        raise Unevaluated()
    raise ValueError("RowKey: expected RowKey[{key1, key2, ...}]")


def _rowkey_call(head, args, ev_):
    if isinstance(head, Expr) and head.head == "RowKey" and len(args) == 1 and isinstance(args[0], Expr) and \
            args[0].head == "Tabular":
        return ev(Expr(args[0], head))
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["RowKey"] = _rowkey_call


@fn("ExtendedKey", **ADDED,
    notes="ExtendedKey[k1, k2, ...]: a multilevel column key (inert); used by PivotToColumns, TabularSummary, "
          "ColumnwiseCombine and JoinAcross prefixes and matched as a pattern in tab[All, ExtendedKey[k, _]]; "
          "ExtendedKey[] raises")
def _extendedkey(*a: Expr) -> Expr:
    if not a:
        raise ValueError("ExtendedKey: at least one key is needed")
    raise Unevaluated()


# ─────────────────────────── joins ───────────────────────────

def _key_of(k):
    return k.args[0] if isinstance(k, Expr) and k.head == "Key" and len(k.args) == 1 else k


def _joinspec(spec, fname):
    """[(left key, right key), ...] or a function."""
    if is_list(spec):
        out = []
        for s in spec.args:
            out.extend(_joinspec(s, fname))
        return out
    if is_rule(spec):
        return [(_key_of(spec.args[0]), _key_of(spec.args[1]))]
    if isinstance(spec, Expr) and spec.head == "Function":
        return spec
    return [(_key_of(spec), _key_of(spec))]


def _as_rows(x, fname):
    """(prefix or None, list of associations, kind)."""
    prefix = None
    if is_rule(x) and not is_list(x.args[1]) or (is_rule(x) and is_list(x.args[1]) and
                                                  all(is_assoc(r) for r in x.args[1].args)):
        prefix, x = x.args
    t, kind = _tab_or_rows(x, fname)
    if t.keys is None:
        raise ValueError(f"{fname}: the data must have named columns")
    return prefix, [t.row_value(i) for i in range(t.nrows)], kind, t


def join_across(a, b, spec, how, kcf, fname):
    pa, arows, kind_a, ta = _as_rows(a, fname)
    pb, brows, kind_b, tb = _as_rows(b, fname)
    js = _joinspec(spec, fname)
    if how not in ("Inner", "Left", "Right", "Outer"):
        raise ValueError(f"{fname}: the join type must be \"Inner\", \"Left\", \"Right\" or \"Outer\"")
    is_fn = not isinstance(js, list)
    tabular_out = kind_a == "tab" or kind_b == "tab"

    def match(x, y):
        if is_fn:
            return apply(js, x, y) is True
        return all(aget(x, ka, missing("KeyAbsent", ka)) == aget(y, kb, missing("KeyAbsent", kb)) and
                   not is_missing(aget(x, ka, missing("KeyAbsent"))) for ka, kb in js)
    same_keys = set() if is_fn else {repr(ka) for ka, kb in js if ka == kb}
    akeys_all = list(ta.keys)
    bkeys_all = list(tb.keys)

    def pk(p, k):
        return Expr("ExtendedKey", p, *(k.args if isinstance(k, Expr) and k.head == "ExtendedKey" else (k,))) \
            if p is not None else k

    def merge(x, y):
        out = []
        xs = x if x is not None else assoc(*((k, missing("Unmatched")) for k in akeys_all))
        ys = y if y is not None else assoc(*((k, missing("Unmatched")) for k in bkeys_all))
        bk = {repr(k) for k in bkeys_all}
        ak = {repr(k) for k in akeys_all}
        for r in xs.args:
            k, v = r.args
            if repr(k) in same_keys and x is None and y is not None:
                v = aget(y, k)
            if pa is None and pb is None and repr(k) in bk and repr(k) not in same_keys:
                yv = aget(ys, k)
                if is_sym(kcf, "Left") or kcf is None:
                    out.append((k, v))
                elif is_sym(kcf, "Right"):
                    out.append((k, yv))
                elif is_sym(kcf, "None"):
                    pass
                else:
                    nk = apply(kcf, k)
                    if not (is_list(nk) and len(nk.args) == 2):
                        raise ValueError(f"{fname}: KeyCollisionFunction must return a pair of keys")
                    out.append((nk.args[0], v))
                    if not tabular_out:      # lists of associations keep the pair together
                        out.append((nk.args[1], yv))
                continue
            out.append((pk(pa, k), v))
        for r in ys.args:
            k, v = r.args
            if repr(k) in same_keys and pa is None and pb is None:
                continue
            if pa is None and pb is None and repr(k) in ak:
                if tabular_out and not (is_sym(kcf, "Left") or kcf is None or is_sym(kcf, "Right") or
                                        is_sym(kcf, "None")):
                    nk = apply(kcf, k)
                    out.append((nk.args[1], v))
                continue
            out.append((pk(pb, k), v))
        return assoc(*out)

    res = []
    matched_b = set()
    for x in arows:
        hit = False
        for jb, y in enumerate(brows):
            if match(x, y):
                res.append(merge(x, y))
                matched_b.add(jb)
                hit = True
        if not hit and how in ("Left", "Outer"):
            res.append(merge(x, None))
    if how in ("Right", "Outer"):
        for jb, y in enumerate(brows):
            if jb not in matched_b:
                res.append(merge(None, y))
    if kind_a == "tab" or kind_b == "tab":
        return _rows_to_tab(res, fname).expr() if res else Tab(
            [k for k in akeys_all] + [k for k in bkeys_all if repr(k) not in same_keys], [[] for _ in range(
                len(akeys_all) + len([k for k in bkeys_all if repr(k) not in same_keys]))]).expr()
    if kind_a == "dataset" or kind_b == "dataset":
        return Expr("Dataset", L(*res))
    return L(*res)


def _join_args(a, fname, default_how):
    a = list(a)
    kcf = None
    while a and is_rule(a[-1]) and is_sym(a[-1].args[0], "KeyCollisionFunction"):
        kcf = a.pop().args[1]
    if len(a) == 4:
        return a[0], a[1], a[2], a[3], kcf
    if len(a) == 3:
        return a[0], a[1], a[2], default_how, kcf
    raise ValueError(f"{fname}: expected {fname}[a, b, keyspec] or {fname}[a, b, keyspec, joinspec]")


@fn("JoinAcross", **ADDED,
    notes="JoinAcross[a, b, keyspec] and JoinAcross[a, b, keyspec, \"Inner\" | \"Left\" | \"Right\" | \"Outer\"] "
          "for lists of associations, Datasets and Tabular objects (results in the same form); keyspec key, "
          "{k1, ...}, ka -> kb, {ka -> kb, ...} or a function; prefixes p -> obj give ExtendedKey[p, k] keys; "
          "absent values are Missing[\"Unmatched\"]; KeyCollisionFunction -> Left (default), Right, None or f"
          ". Check: exact (exact input gives exact results), reference (differential cases in tests/library).")
def _joinacross(*a: Expr) -> Expr:
    x, y, spec, how, kcf = _join_args(a, "JoinAcross", "Inner")
    return join_across(x, y, spec, how, kcf, "JoinAcross")


@fn("LeftJoinAcross", **ADDED,
    notes="LeftJoinAcross[a, b, keyspec] = JoinAcross[a, b, keyspec, \"Left\"] (unmatched rows of a kept with "
          "Missing[\"Unmatched\"] values); LeftJoinAcross[b, keyspec] is the operator form for a; "
          "KeyCollisionFunction as in JoinAcross")
def _leftjoinacross(*a: Expr) -> Expr:
    a = list(a)
    opts = [x for x in a if is_rule(x) and is_sym(x.args[0], "KeyCollisionFunction")]
    pos = [x for x in a if x not in opts]
    if len(pos) == 2:
        raise Unevaluated()
    if len(pos) != 3:
        raise ValueError("LeftJoinAcross: expected LeftJoinAcross[a, b, keyspec]")
    x, y, spec, how, kcf = _join_args(pos + opts, "LeftJoinAcross", "Left")
    return join_across(x, y, spec, "Left", kcf, "LeftJoinAcross")


def _lj_op(head, args, ev_):
    if isinstance(head, Expr) and head.head == "LeftJoinAcross" and len(head.args) == 2 and len(args) == 1:
        return ev(Expr("LeftJoinAcross", args[0], *head.args))
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["LeftJoinAcross"] = _lj_op


@fn("ColumnwiseCombine", **ADDED,
    notes="ColumnwiseCombine[{tab1, ...} -> ckey | {ckeys}], ColumnwiseCombine[{tab1 -> ckeys1, ...}], prefixed "
          "<|p1 -> tab1, ...|> -> ckeys (other columns become ExtendedKey[p, col]) and <|p -> tab -> ckeys, ...|>, "
          "<|...|> -> crit (crit gets <|p1 -> row1, ...|>; True& = cross join), with required rows {} / \"Inner\", "
          "\"Outer\", \"Left\", \"Right\", i or {i...} (unmatched rows filled with missing values); join keys come "
          "first in the result")
def _columnwisecombine(spec: Expr, required: Expr = "Inner") -> Expr:
    crit = None
    common = None
    if is_rule(spec) and (is_list(spec.args[0]) or is_assoc(spec.args[0])):
        group, ks = spec.args
        if isinstance(ks, Expr) and ks.head == "Function":
            crit = ks
        else:
            common = list(ks.args) if is_list(ks) else [ks]
    else:
        group = spec
    if is_assoc(group):
        items = [(r.args[0], r.args[1]) for r in group.args]
    elif is_list(group):
        items = [(None, x) for x in group.args]
    else:
        raise ValueError("ColumnwiseCombine: give a list or association of Tabular objects")
    tabs, perkeys, prefixes = [], [], []
    for p, x in items:
        ck = None
        if is_rule(x):
            x, ckx = x.args
            ck = list(ckx.args) if is_list(ckx) else [ckx]
        t, _ = _tab_or_rows(x, "ColumnwiseCombine")
        if t.keys is None:
            raise ValueError("ColumnwiseCombine: the tables need named columns")
        tabs.append(t)
        prefixes.append(p)
        perkeys.append(ck if ck is not None else common)
    n = len(tabs)
    if crit is None and any(k is None for k in perkeys):
        raise ValueError("ColumnwiseCombine: give the key column(s) to combine on")
    if crit is None and len({len(k) for k in perkeys}) != 1:
        raise ValueError("ColumnwiseCombine: every table needs the same number of key columns")
    merged = crit is None and common is not None
    # required tables
    if is_sym(required, "None") or required in ("Inner",) or (is_list(required) and not required.args):
        req = []
    elif required == "Outer":
        req = list(range(n))
    elif required == "Left":
        req = [0]
    elif required == "Right":
        req = [n - 1]
    elif K.is_int(required):
        req = [int(required) - 1]
    elif is_list(required):
        req = []
        for r in required.args:
            if K.is_int(r):
                req.append(int(r) - 1)
            elif r in prefixes:
                req.append(prefixes.index(r))
            else:
                raise ValueError(f"ColumnwiseCombine: {r!r} does not identify a table")
    else:
        raise ValueError("ColumnwiseCombine: bad specification of required rows")
    for r in req:
        if not 0 <= r < n:
            raise ValueError("ColumnwiseCombine: a required table index is out of range")
    kidx = [[t.col_index(k, "ColumnwiseCombine") for k in ks] for t, ks in zip(tabs, perkeys)] if crit is None \
        else None
    # result columns
    out_keys = []
    if merged:
        out_keys.extend(common)
    colmap = []          # (table, column index, output key)
    for ti, t in enumerate(tabs):
        for j, k in enumerate(t.keys):
            if merged and j in kidx[ti]:
                continue
            ok = Expr("ExtendedKey", prefixes[ti], k) if prefixes[ti] is not None else k
            if ok in out_keys:
                raise ValueError(f"ColumnwiseCombine: the column {k!r} appears in several tables; use prefixes "
                                 f"<|p1 -> tab1, ...|>")
            out_keys.append(ok)
            colmap.append((ti, j))
    rows = []

    def emit(combo):
        row = []
        if merged:
            first = next(ti for ti in range(n) if combo[ti] is not None)
            row.extend(tabs[first].cols[j][combo[first]] for j in kidx[first])
        for ti, j in colmap:
            row.append(tabs[ti].cols[j][combo[ti]] if combo[ti] is not None else missing("NotAvailable"))
        rows.append(row)

    used = [set() for _ in range(n)]

    def rec(ti, combo):
        if ti == n:
            if crit is not None:
                arg = assoc(*((prefixes[i] if prefixes[i] is not None else i + 1, tabs[i].row_value(combo[i]))
                              for i in range(n)))
                if apply(crit, arg) is not True:
                    return
            emit(combo)
            for i, r in enumerate(combo):
                used[i].add(r)
            return
        for r in range(tabs[ti].nrows):
            if crit is None and ti > 0:
                kv = [tabs[ti].cols[j][r] for j in kidx[ti]]
                kv0 = [tabs[0].cols[j][combo[0]] for j in kidx[0]]
                if kv != kv0:
                    continue
            rec(ti + 1, combo + [r])
    rec(0, [])
    for ti in req:
        for r in range(tabs[ti].nrows):
            if r not in used[ti]:
                emit([r if i == ti else None for i in range(n)])
    return Tab(out_keys, [list(c) for c in zip(*rows)] if rows else [[] for _ in out_keys]).expr()


def _options_tab_pre(a):
    if not a:
        return _PASS
    x, names_ = a[0], a[1:]
    t = parse_tab(x)
    if t is None:
        return _PASS
    rules = list(t.options)
    if names_:
        keys = names_[0].args if is_list(names_[0]) else (names_[0],)
        rules = [r for r in rules if r.args[0] in keys]
    return L(*rules)


wrap_registry("Options", _options_tab_pre)
