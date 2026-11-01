"""Wolfram batch X01 (part 2): the Structured Package Format (SPF).

PackageInitialize, PackageExported, PackageScoped and PackageImport follow the tutorial "Using the Structured
Package Format" (reference.wolfram.com/language/tutorial/UsingTheStructuredPackageFormat): a scanning pass reads
the top-level declarations of every file without evaluating them, then each file is read with $Context set to its
own private context and $ContextPath reset to the package contexts.
"""
from __future__ import annotations

import os
import re

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, L, is_list
from .wl_common import ev, is_sym

AREA = "core extras (X01)"

_DECL_RE = re.compile(r"^\s*(PackageExported|PackageScoped|PackageImport)\s*\[(.*)\]\s*;?\s*$", re.S)
_NAME_RE = re.compile(r"[$A-Za-z][$A-Za-z0-9]*")
_STR_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
_SEG_RE = re.compile(r"[^A-Za-z0-9$]")
_PROPS = ("ExtraPublicContexts", "FileOrderingFunction", "HiddenImports", "IgnoreFiles", "LoadFirstFiles",
          "LoadLastFiles", "SkipLoadingFiles", "SymbolsToProtect")
_LOADING = []          # stack of the SPF file being read: dicts with "ctx", "file", "imports"
_PACKAGES = {}         # ctx -> {"root", "props", "files", "exported", "scoped"} of loaded SPF packages


def _WE():
    from . import wl_evaluation as WE
    return WE


def _input_file():
    from .wl_files import _INPUT_FILE
    return _INPUT_FILE[-1] if _INPUT_FILE and _INPUT_FILE[-1] else None


def _failure_nodir(ctx):
    msg = ("When executed during interactive input, PackageLoad requires a second argument to specify the "
           "directory in which the package's code files reside.")
    return Expr("Failure", "RootDirNotAvailable",
                Expr("Association", Expr("Rule", "MessageTemplate", msg),
                     Expr("Rule", "MessageParameters", L(ctx))))


def _strip_comments(text):
    out, i, depth = [], 0, 0
    while i < len(text):
        if text.startswith("(*", i):
            depth += 1
            i += 2
        elif depth and text.startswith("*)", i):
            depth -= 1
            i += 2
        else:
            if not depth:
                out.append(text[i])
            i += 1
    return "".join(out)


def _decl_args(body):
    """(strings, names) of the literal arguments of a declaration (no evaluation)."""
    strings = _STR_RE.findall(body)
    rest = _STR_RE.sub(" ", body)
    names = _NAME_RE.findall(rest)
    return strings, names


def _scan(path):
    """First pass: the declarations of a file, in order, as (kind, strings, names, chunk_index)."""
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    chunks = _WE().split_statements(text)
    decls = []
    for i, ch in enumerate(chunks):
        m = _DECL_RE.match(_strip_comments(ch))
        if m:
            strings, names = _decl_args(m.group(2))
            decls.append((m.group(1), strings, names, i))
    return chunks, decls


def _private_context(ctx, root, path):
    rel = os.path.relpath(path, root) if root else os.path.basename(path)
    parts = rel.split(os.sep)
    parts[-1] = os.path.splitext(parts[-1])[0]
    segs = []
    for p in parts:
        s = _SEG_RE.sub("", p)
        if s and s[0].isdigit():
            s = "F" + s
        segs.append(s or "File")
    return ctx + "".join(s + "`" for s in segs) + "Private`"


def _ctx_list(v, prop):
    if v is None:
        return []
    if isinstance(v, str):
        v = L(v)
    if not (is_list(v) and all(isinstance(c, str) and _WE()._CTX_RE.match(c) for c in v.args)):
        raise ValueError(f"PackageInitialize: \"{prop}\" must be a context name or a list of context names")
    return list(v.args)


def _file_list(v, prop):
    if v is None:
        return []
    if isinstance(v, str):
        v = L(v)
    if not (is_list(v) and all(isinstance(c, str) for c in v.args)):
        raise ValueError(f"PackageInitialize: \"{prop}\" must be a file name or a list of file names")
    return list(v.args)


def _props(args):
    out = {}
    for a in args:
        rules = a.args if isinstance(a, Expr) and a.head == "Association" else (a,)
        for r in rules:
            if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2):
                raise ValueError(f"PackageInitialize: {r!r} is not a property rule \"name\" -> value")
            k = r.args[0]
            if isinstance(k, Symbol):
                k = k.name
            if k not in _PROPS:
                raise ValueError(f"PackageInitialize: unknown property {k!r}; the properties are "
                                 + ", ".join(_PROPS))
            out[k] = r.args[1]
    return out


def _matches(rel, names):
    base = os.path.basename(rel)
    return any(n in (rel, base) or os.path.normpath(n) == os.path.normpath(rel) for n in names)


_LOADER_RE = re.compile(r"(?m)^\s*PackageInitialize\s*\[")


def _is_loader(path):
    """A file whose text has a top-level PackageInitialize call is a loader file and is not read as package code
    (the documentation: pointing PackageInitialize at a directory bypasses its init.wl loader)."""
    try:
        with open(path, encoding="utf-8") as fh:
            text = _strip_comments(fh.read())
    except OSError:
        return False
    return bool(_LOADER_RE.search(text))


def _collect(root, loader, props):
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        depth = 0 if dirpath == root else os.path.relpath(dirpath, root).count(os.sep) + 1
        if depth >= 3:
            dirnames[:] = []
        dirnames.sort()
        for f in filenames:
            if f.endswith((".wl", ".m")):
                p = os.path.join(dirpath, f)
                if (loader and os.path.abspath(p) == os.path.abspath(loader)) or _is_loader(p):
                    continue
                found.append(os.path.relpath(p, root))
    ignore = _file_list(props.get("IgnoreFiles"), "IgnoreFiles")
    found = [r for r in found if not _matches(r, ignore)]
    order = props.get("FileOrderingFunction")
    if order is None or is_sym(order, "Automatic"):
        found.sort()
    else:
        s = ev(Expr("Sort", L(*found), order))
        if not (is_list(s) and sorted(s.args) == sorted(found)):
            raise ValueError("PackageInitialize: \"FileOrderingFunction\" must be an ordering function for Sort")
        found = list(s.args)
    first = _file_list(props.get("LoadFirstFiles"), "LoadFirstFiles")
    last = _file_list(props.get("LoadLastFiles"), "LoadLastFiles")
    head = [r for n in first for r in found if _matches(r, [n])]
    tail = [r for n in last for r in found if _matches(r, [n]) and r not in head]
    mid = [r for r in found if r not in head and r not in tail]
    return [os.path.join(root, r) for r in head + mid + tail]


def _parse_with(chunk, rename):
    """Parse one statement; names imported by name in this file resolve to their full names."""
    from .. import parser
    hook = parser.POST_PARSE_HOOK
    if not rename:
        return parser.parse(chunk)

    def pre(e):
        WE = _WE()
        e = WE._rename_tree(e, rename)
        return hook(e) if hook is not None else e
    parser.POST_PARSE_HOOK = pre
    try:
        return parser.parse(chunk)
    finally:
        parser.POST_PARSE_HOOK = hook


def _import(ctx_name, names, alias, frame):
    """Load an imported context and make it (or only the named symbols) visible in the current file."""
    WE = _WE()
    WE._needs(Expr("Rule", ctx_name, Symbol("None")))
    if alias is not None:
        WE._check_context(alias, "PackageImport")
        al = WE.aliases()
        al[alias] = ctx_name
        WE._set_own("$ContextAliases", WE._assoc_of(al))
        return
    if names:
        for n in names:
            if not WE._exists_in(ctx_name, n):
                raise ValueError(f"PackageImport::nosym: The symbol {n} was not found in the package being "
                                 f"imported, {ctx_name}.")
            frame["rename"][n] = WE._internal(ctx_name + n)
        return
    p = WE.context_path()
    if ctx_name not in p:
        WE._set_own("$ContextPath", L(*p[:-1], ctx_name, p[-1]) if p and p[-1] == "System`" else L(*p, ctx_name))


def _import_decls(decls):
    """Coalesce PackageImport declarations: ctx -> (first chunk index, names, alias)."""
    out = {}
    for kind, strings, names, i in decls:
        if kind != "PackageImport":
            continue
        if not strings:
            raise ValueError("PackageImport: the first argument must be a literal context name")
        c = strings[0]
        alias = strings[1] if len(strings) > 1 else None
        rec = out.setdefault(c, [i, [], alias])
        rec[1].extend(n for n in names if n not in rec[1])
    return out


def _load_file(ctx, root, path, chunks, decls, hidden, skip=False):
    """Second pass for one file: evaluate its statements in its private context."""
    if skip:
        return
    WE = _WE()
    from .wl_files import _INPUT_FILE
    import importlib
    E = importlib.import_module("mikoshilang.evaluate")
    o = K.state().own
    saved = {k: o.get(k, WE.UNH) for k in ("$Context", "$ContextPath")}
    frame = {"ctx": ctx, "file": path, "rename": {}}
    imports = _import_decls(decls)
    at = {v[0]: (c, v[1], v[2]) for c, v in imports.items()}
    _LOADING.append(frame)
    _INPUT_FILE.append(os.path.abspath(path))
    try:
        WE._set_own("$Context", _private_context(ctx, root, path))
        WE._set_own("$ContextPath", L(ctx, ctx + "PackageScope`", *hidden, "System`"))
        for i, chunk in enumerate(chunks):
            if i in at:
                c, names, alias = at[i]
                _import(c, names, alias, frame)
                continue
            if any(d[3] == i and d[0] == "PackageImport" for d in decls):
                continue                      # coalesced into the first PackageImport of its context
            e = _parse_with(chunk, frame["rename"])
            if e is None:
                continue
            E.evaluate(e)
    finally:
        _INPUT_FILE.pop()
        _LOADING.pop()
        for k, v in saved.items():
            WE._restore_own(k, v)


def _declare(ctx, decls_by_file):
    WE = _WE()
    exported, scoped = [], []
    for decls in decls_by_file:
        for kind, _s, names, _i in decls:
            if kind == "PackageExported":
                exported += [n for n in names if n not in exported]
            elif kind == "PackageScoped":
                scoped += [n for n in names if n not in scoped]
    for n in exported:
        WE._register(ctx + n)
    for n in scoped:
        WE._register(ctx + "PackageScope`" + n)
    return exported, scoped


def _finish(ctx, exported, props, saved_path):
    WE = _WE()
    extra = _ctx_list(props.get("ExtraPublicContexts"), "ExtraPublicContexts")
    for c in extra:
        WE._needs(Expr("Rule", c, Symbol("None")))
    WE._restore_own("$ContextPath", saved_path)
    p = [c for c in WE.context_path() if c != ctx and c not in extra]
    WE._set_own("$ContextPath", L(ctx, *extra, *p))
    if ctx not in WE.packages():
        WE._set_own("$Packages", L(ctx, *WE.packages()))
    prot = props.get("SymbolsToProtect")
    if prot is not None:
        if is_sym(prot, "All"):
            names = list(exported)
        else:
            names = _file_list(prot, "SymbolsToProtect")
        for n in names:
            if WE._exists_in(ctx, n):
                full = ctx + n
            elif WE._exists_in(ctx + "PackageScope`", n):
                full = ctx + "PackageScope`" + n
            else:
                raise ValueError(f"PackageInitialize: \"SymbolsToProtect\": {n} is not a symbol of the package {ctx}")
            ev(Expr("Protect", Symbol(WE._internal(full))))


def _load_package(ctx, root, files, props, loader=None, single=None):
    WE = _WE()
    hidden = _ctx_list(props.get("HiddenImports"), "HiddenImports")
    extra = _ctx_list(props.get("ExtraPublicContexts"), "ExtraPublicContexts")
    for c in hidden + extra:
        WE._needs(Expr("Rule", c, Symbol("None")))
    skip = _file_list(props.get("SkipLoadingFiles"), "SkipLoadingFiles")
    scanned = [(p, *_scan(p)) for p in files]
    if single is not None:
        scanned.append(single)
    exported, scoped = _declare(ctx, [d for _p, _c, d in scanned])
    saved_path = K.state().own.get("$ContextPath", WE.UNH)
    _PACKAGES[ctx] = {"root": root, "props": props, "files": [p for p, _c, _d in scanned],
                      "exported": exported, "scoped": scoped, "hidden": hidden + extra}
    for p, chunks, decls in scanned:
        rel = os.path.relpath(p, root) if root else os.path.basename(p)
        _load_file(ctx, root, p, chunks, decls, hidden + extra, skip=_matches(rel, skip))
    _finish(ctx, exported, props, saved_path)


@fn("PackageInitialize", area=AREA, status="added",
    notes="PackageInitialize[\"ctx`\"] (in a loader file) loads every .wl and .m file in the loader's directory and "
          "its subdirectories up to three levels deep (alphabetical by relative path; loader files, those with a top-level "
          "PackageInitialize call, are excluded) "
          "as the package ctx`; PackageInitialize[\"ctx`\", dir] loads the files of dir (bypassing a loader "
          "file there, as documented); PackageInitialize[\"ctx`\", file] reloads one "
          "file of a package loaded before; PackageInitialize[\"ctx`\", None] makes the rest of the current file a "
          "single-file package. Two passes as in the SPF tutorial: a scan of the top-level PackageExported, "
          "PackageScoped and PackageImport declarations (literal arguments, not evaluated), then each file is read "
          "with $Context = its private context ctx`<subdirs>`<file name>`Private` (non-alphanumeric characters "
          "of the segments removed) and $ContextPath = {ctx`, ctx`PackageScope`, hidden imports, System`}; "
          "afterwards $Context and $ContextPath are restored, ctx` (and \"ExtraPublicContexts\") is prepended to "
          "$ContextPath and ctx` is added to $Packages. Properties (an association, or rules): "
          "ExtraPublicContexts, FileOrderingFunction (ordering function for Sort on the relative file names), "
          "HiddenImports, IgnoreFiles, LoadFirstFiles, LoadLastFiles, SkipLoadingFiles (scanned but not read), "
          "SymbolsToProtect (names or All for the exported symbols; set Protected as in the documented "
          "example). Interactive use without a directory returns the documented "
          "Failure[\"RootDirNotAvailable\", ...]. Check: reference (the documentation's SPFExample package: "
          "Func1[2] = 3, Func2 uses the PackageScoped utilFunc, contexts as the tutorial states) in tests.")
def _packageinitialize(ctx: String, spec: Expr = None, *props: Expr) -> Expr:
    WE = _WE()
    if not (isinstance(ctx, str) and WE._CTX_RE.match(ctx)):
        raise ValueError(f"PackageInitialize: {ctx!r} is not a context name (a string such as \"MyPackage`\")")
    if spec is not None and (isinstance(spec, Expr) and spec.head in ("Association", "Rule", "RuleDelayed")):
        props = (spec,) + props
        spec = None
    pr = _props(props)
    loader = _input_file()
    if isinstance(spec, Expr) and spec.head == "File" and len(spec.args) == 1:
        spec = spec.args[0]
    if spec is None:
        if loader is None:
            return _failure_nodir(ctx)
        root = os.path.dirname(os.path.abspath(loader))
        _load_package(ctx, root, _collect(root, loader, pr), pr, loader)
        return None
    if is_sym(spec, "None"):
        if loader is None:
            raise ValueError("PackageInitialize: PackageInitialize[ctx, None] must be read from a file (Get)")
        with open(loader, encoding="utf-8") as fh:
            chunks = WE.split_statements(fh.read())
        k = next((i for i, c in enumerate(chunks) if re.match(r"\s*PackageInitialize\s*\[", _strip_comments(c))),
                 None)
        if k is None:
            raise ValueError("PackageInitialize: the PackageInitialize call must be a top-level statement of its "
                             "file")
        rest = chunks[k + 1:]
        decls = []
        for i, ch in enumerate(rest):
            m = _DECL_RE.match(_strip_comments(ch))
            if m:
                decls.append((m.group(1), *_decl_args(m.group(2)), i))
        _load_package(ctx, None, [], pr, loader, single=(loader, rest, decls))
        raise WE.StopSource(None)
    if not isinstance(spec, str):
        raise ValueError(f"PackageInitialize: {spec!r} is not a directory, a file name or None")
    path = os.path.expanduser(spec)
    if not os.path.isabs(path) and not os.path.exists(path):
        for d in WE._search_dirs():
            if os.path.exists(os.path.join(d, path)):
                path = os.path.join(d, path)
                break
    if os.path.isdir(path):
        root = os.path.abspath(path)
        _load_package(ctx, root, _collect(root, loader, pr), pr)
        return None
    if os.path.isfile(path):
        info = _PACKAGES.get(ctx)
        if info is None:
            raise ValueError(f"PackageInitialize: the package {ctx} has not been loaded, so {spec} cannot be "
                             "reloaded")
        chunks, decls = _scan(path)
        _declare(ctx, [decls])
        saved_path = K.state().own.get("$ContextPath", WE.UNH)
        _load_file(ctx, info["root"], os.path.abspath(path), chunks, decls, info["hidden"])
        WE._restore_own("$ContextPath", saved_path)
        return None
    raise ValueError(f"PackageInitialize: cannot open {spec}")


def _decl_check(name, args):
    if not _LOADING:
        raise ValueError(f"{name}: {name} can only be used as a top-level declaration in a file of a package loaded "
                         "by PackageInitialize")
    for a in args:
        items = a.args if is_list(a) else (a,)
        for x in items:
            if not (isinstance(x, (Symbol, str))):
                raise TypeError(f"{name}: {x!r} is not a symbol or a list of symbols")
    return None


@fn("PackageExported", area=AREA, status="added", hold=True,
    notes="PackageExported[sym] or PackageExported[{sym1, ...}] (HoldAll), a top-level declaration in an SPF file: "
          "the symbols are created in the package context ctx` before any file is read (scanning pass of "
          "PackageInitialize), so every file and the user see the same symbol. Evaluated in a package file it "
          "returns Null; outside a file read by PackageInitialize it raises ValueError; arguments that are not "
          "symbols raise TypeError. Check: reference (the SPFExample package: Func1 and Func2 are in "
          "SPFExample`) in tests.")
def _packageexported(*syms: Held) -> Expr:
    return _decl_check("PackageExported", syms)


@fn("PackageScoped", area=AREA, status="added", hold=True,
    notes="PackageScoped[sym] or PackageScoped[{sym1, ...}] (HoldAll), a top-level declaration in an SPF file: "
          "the symbols are created in ctx`PackageScope`, which is on $ContextPath while every file of the package "
          "is read but not afterwards. Evaluated in a package file it returns Null; outside a file read by "
          "PackageInitialize it raises ValueError. Check: reference (the SPFExample package: utilFunc is "
          "SPFExample`PackageScope`utilFunc, visible from PackageFile2.wl) in tests.")
def _packagescoped(*syms: Held) -> Expr:
    return _decl_check("PackageScoped", syms)


@fn("PackageImport", area=AREA, status="added", hold=True,
    notes="PackageImport[\"ctx`\"] is Needs[\"ctx`\"] at an interactive prompt or in a non-SPF file; "
          "PackageImport[\"ctx`\" -> \"alias`\"] loads ctx` and defines the context alias (ctx` is not put on "
          "$ContextPath). In an SPF file it is a declaration handled by the scanning pass: the context is loaded at "
          "the first PackageImport for it in the file and made visible in that file only; "
          "PackageImport[\"ctx`\", sym] or {syms} (import by name) makes only the named symbols visible "
          "from that point of the file (a name the package does not have raises PackageImport::nosym as "
          "ValueError). Import by name outside an SPF file raises ValueError (documented: only possible in SPF "
          "packages). Check: reference (documentation examples: Needs behaviour, the alias form, nosym) in tests.")
def _packageimport(c: Held, names: Held = None) -> Expr:
    WE = _WE()
    c = ev(c)
    if names is not None:
        raise ValueError("PackageImport: importing by name is only possible in a package that uses the Structured "
                         "Package Format")
    if isinstance(c, Expr) and c.head == "Rule" and len(c.args) == 2:
        if not isinstance(c.args[1], str):
            raise ValueError("PackageImport: the alias must be a context name")
        return WE._needs(c)
    if not isinstance(c, str):
        raise ValueError(f"PackageImport: {c!r} is not a context name")
    return WE._needs(c)
