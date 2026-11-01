"""Wolfram batch X01 (part 3): FileSystemTree, TextSearchReport and BioSequenceBackTranslateList."""
from __future__ import annotations

import itertools
import os
from functools import lru_cache

from ..expr import Expr, Symbol
from .. import kernel as K
from .core import fn, L, is_list, Unevaluated
from .wl_common import ev, is_sym

AREA = "core extras (X01)"
INF = float("inf")


def _opts(args, names, fname):
    pos, o = [], {}
    for a in args:
        if isinstance(a, Expr) and a.head in ("Rule", "RuleDelayed") and len(a.args) == 2 and \
                isinstance(a.args[0], (Symbol, str)):
            k = a.args[0].name if isinstance(a.args[0], Symbol) else a.args[0]
            if k not in names:
                raise ValueError(f"{fname}: unknown option {k}")
            o[k] = a.args[1]
        else:
            if o:
                raise ValueError(f"{fname}: positional argument after options")
            pos.append(a)
    return pos, o


def _max_items(v, fname, default):
    if v is None:
        return default
    if is_sym(v, "Infinity") or is_sym(v, "All"):
        return INF
    if K.is_int(v) and not isinstance(v, bool) and int(v) >= 0:
        return int(v)
    raise ValueError(f"{fname}: MaxItems must be a non-negative integer or Infinity")


# ═══════════════════════════════════════════════ FileSystemTree ═══════════════════════════════════════════════

_TREE_STYLE = {"TreeElementStyle", "TreeElementLabel", "TreeElementLabelFunction", "TreeElementStyleFunction",
               "ParentEdgeLabel", "ParentEdgeLabelFunction", "ParentEdgeStyle", "ParentEdgeStyleFunction",
               "ParentEdgeShapeFunction", "TreeElementShape", "TreeElementShapeFunction", "TreeElementSize",
               "TreeElementSizeFunction", "TreeLayout", "MaxDisplayedChildren", "ImageSize", "PlotTheme",
               "TreeElementLabelStyle", "ParentEdgeLabelStyle", "TreeElementCoordinates", "Background"}


def _levels(spec, fname):
    def lv(x):
        if is_sym(x, "Infinity"):
            return INF
        if K.is_int(x) and not isinstance(x, bool):
            if int(x) < 0:
                raise NotImplementedError(f"{fname}: negative levels are not supported")
            return int(x)
        raise ValueError(f"{fname}: invalid level specification {x!r}")
    if spec is None or is_sym(spec, "All"):
        return 0, INF
    if is_list(spec) and len(spec.args) == 1:
        n = lv(spec.args[0])
        return n, n
    if is_list(spec) and len(spec.args) == 2:
        return lv(spec.args[0]), lv(spec.args[1])
    return 1, lv(spec)


def _matching(d, form):
    if form is None or is_sym(form, "All"):
        return None
    r = ev(Expr("FileNames", form, L(d)))
    if not is_list(r):
        raise ValueError(f"FileSystemTree: FileNameForms {form!r} is not a valid FileNames form")
    return {os.path.abspath(x) for x in r.args if isinstance(x, str)}


def _fs_tree(path, level, lo, hi, o):
    """Tree node for directory path at level; None when nothing in it is included."""
    try:
        names = sorted(os.listdir(path))
    except OSError as e:
        raise ValueError(f"FileSystemTree: cannot read the directory {path}: {e.strerror}")
    keep = _matching(path, o.get("FileNameForms"))
    children = []
    k = level + 1
    for nm in names:
        full = os.path.join(path, nm)
        if os.path.isdir(full):
            if k > hi:
                continue
            if k == hi:
                children.append((nm, Expr("Tree", full, Symbol("None"))))
                continue
            sub = _fs_tree(full, k, lo, hi, o)
            if sub is not None or k >= lo:
                children.append((nm, sub if sub is not None else Expr("Tree", full, Symbol("None"))))
        else:
            if k < lo or k > hi or (keep is not None and os.path.abspath(full) not in keep):
                continue
            children.append((nm, Expr("Tree", full, Symbol("None"))))
    mx = o.get("\0max", INF)
    if mx != INF:
        children = children[:mx]
    if o.get("IncludeDirectories") is True:
        children.append((Symbol("All"), Expr("Tree", path, Symbol("None"))))
    if not children:
        return None if level > 0 else Expr("Tree", path, Symbol("None"))
    return Expr("Tree", path, Expr("Association", *(Expr("Rule", a, b) for a, b in children)))


@fn("FileSystemTree", area=AREA, status="added",
    notes="FileSystemTree[root] gives Tree[root, <|name -> subtree, ...|>]: a file is the leaf Tree[full name, "
          "None] keyed by its file name, a directory is Tree[full name, <|...|>] (an empty directory, or one at "
          "the deepest included level, is Tree[full name, None]); entries are in alphabetical order. "
          "FileSystemTree[root, levelspec] uses TreeLevel level specifications (n, All, Infinity, {n}, "
          "{n1, n2}; default {0, Infinity}): files at the included levels are kept and directories above the "
          "lowest included level are kept as the path to them. Options: IncludeDirectories -> True adds a leaf "
          "keyed All (data the directory name) to every directory; FileNameForms (any FileNames form) keeps only "
          "matching files; MaxItems -> n keeps the first n entries of each directory. The third argument r (level "
          "combination), Tree styling options and negative levels raise NotImplementedError; a missing directory "
          "raises ValueError. Check: exact (a temporary directory tree built in the tests) in tests.")
def _filesystemtree(root: Expr, *rest: Expr) -> Expr:
    pos, o = _opts(rest, {"IncludeDirectories", "FileNameForms", "MaxItems"} | _TREE_STYLE, "FileSystemTree")
    bad = set(o) & _TREE_STYLE
    if bad:
        raise NotImplementedError(f"FileSystemTree: the Tree styling option {sorted(bad)[0]} is not supported")
    if isinstance(root, Expr) and root.head == "File" and len(root.args) == 1:
        root = root.args[0]
    if not isinstance(root, str):
        raise TypeError(f"FileSystemTree: {root!r} is not a directory name")
    if len(pos) > 2:
        raise ValueError("FileSystemTree: too many arguments")
    if len(pos) == 2:
        raise NotImplementedError("FileSystemTree: the level-combination argument r is not supported")
    lo, hi = _levels(pos[0] if pos else None, "FileSystemTree")
    path = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(path):
        raise ValueError(f"FileSystemTree: {root} is not an existing directory")
    inc = o.get("IncludeDirectories", False)
    if inc not in (True, False):
        raise ValueError("FileSystemTree: IncludeDirectories must be True or False")
    o["IncludeDirectories"] = inc
    o["\0max"] = _max_items(o.get("MaxItems"), "FileSystemTree", INF)
    return _fs_tree(path, 0, lo, hi, o)


# ═══════════════════════════════════════════════ TextSearchReport ═══════════════════════════════════════════════

def _assoc_get(a, k):
    for r in a.args:
        if r.args[0] == k:
            return r.args[1]
    return None


@fn("TextSearchReport", area=AREA, status="added",
    notes="TextSearchReport[source, form] runs TextSearch[source, form] and gives a Dataset with one association "
          "per matching file, in TextSearch's ranking order: \"Title\", \"FileName\", \"ModificationDate\" "
          "(FileDate of the file), \"CreationDate\" (the file's birth time where the operating system records it, "
          "else Missing[\"NotAvailable\"]: Linux stat has no creation time), \"FileByteCount\", \"FileExtension\" "
          "and \"Location\" (File[...]). Results that are not files are omitted. The \"Score\" and "
          "\"MissedQueryParts\" keys of index searches are not given. Check: exact (a temporary directory with "
          "known files) in tests.")
def _textsearchreport(src: Expr, form: Expr) -> Expr:
    if isinstance(src, Symbol):
        raise Unevaluated()
    try:
        r = ev(Expr("TextSearch", src, form, "ContentObject"))
    except (ValueError, TypeError, NotImplementedError) as e:
        msg = str(e)
        if not msg.startswith("TextSearchReport: "):
            msg = "TextSearchReport: " + msg.split(": ", 1)[-1]
        raise type(e)(msg) from e
    if not is_list(r):
        raise ValueError(f"TextSearchReport: TextSearch did not return results for {form!r}")
    rows = []
    for co in r.args:
        if not (isinstance(co, Expr) and co.head == "ContentObject" and co.args
                and isinstance(co.args[0], Expr) and co.args[0].head == "Association"):
            continue
        a = co.args[0]
        loc = _assoc_get(a, "Location")
        if not (isinstance(loc, Expr) and loc.head == "File" and loc.args and isinstance(loc.args[0], str)):
            continue
        path = loc.args[0]
        try:
            stt = os.stat(path)
        except OSError:
            continue
        mod = ev(Expr("FileDate", path, "Modification"))
        bt = getattr(stt, "st_birthtime", None)
        cre = ev(Expr("FromUnixTime", bt)) if bt is not None else Expr("Missing", "NotAvailable")
        base = os.path.basename(path)
        title = _assoc_get(a, "Title")
        rows.append(Expr("Association",
                         Expr("Rule", "Title", title if title is not None else os.path.splitext(base)[0]),
                         Expr("Rule", "FileName", base),
                         Expr("Rule", "ModificationDate", mod),
                         Expr("Rule", "CreationDate", cre),
                         Expr("Rule", "FileByteCount", stt.st_size),
                         Expr("Rule", "FileExtension", os.path.splitext(base)[1].lstrip(".")),
                         Expr("Rule", "Location", Expr("File", path))))
    return Expr("Dataset", L(*rows))


# ═══════════════════════════════════════════ BioSequenceBackTranslateList ═══════════════════════════════════════════

_BASES = "ACGT"
_IUPAC = {"A": "A", "C": "C", "G": "G", "T": "T", "AG": "R", "CT": "Y", "CG": "S", "AT": "W", "GT": "K",
          "AC": "M", "CGT": "B", "AGT": "D", "ACT": "H", "ACG": "V", "ACGT": "N"}


def _letters(bases):
    return _IUPAC["".join(sorted(bases))]


@lru_cache(maxsize=None)
def _cover(codons):
    """Degenerate triplets whose instances are exactly the codon set: the codons are grouped by first base, first
    bases with the same set of (second, third) pairs share one degenerate letter, and within a group the second
    bases with the same set of third bases share one letter. This reproduces every documented example (e.g. the
    covers {TAY, TGB, TYN, VNN} of X and {CTN, TTR} of L); the triplets are sorted alphabetically."""
    by_first = {}
    for c in codons:
        by_first.setdefault(c[0], set()).add(c[1:])
    groups = {}
    for b, pairs in by_first.items():
        groups.setdefault(frozenset(pairs), []).append(b)
    out = []
    for pairs, firsts in groups.items():
        by_second = {}
        for p in pairs:
            by_second.setdefault(p[0], set()).add(p[1])
        sub = {}
        for b, thirds in by_second.items():
            sub.setdefault(frozenset(thirds), []).append(b)
        for thirds, seconds in sub.items():
            out.append(_letters(firsts) + _letters(seconds) + _letters(thirds))
    return tuple(sorted(out))


def _letter_codons(letter, tab, fname):
    from .wl_text import _PEP_DEGEN
    letters = _PEP_DEGEN.get(letter, letter)
    if letter == ".":
        letters = "*"
    out = frozenset(c for c, v in tab.items() if v in letters)
    if not out:
        raise ValueError(f"{fname}: no codon of the translation table codes for {letter!r}")
    return out


@fn("BioSequenceBackTranslateList", area=AREA, status="added",
    notes="BioSequenceBackTranslateList[pep, gtt, startspec] gives DNA BioSequences with IUPAC degenerate letters "
          "whose instances are exactly the codings of the peptide: for each residue its codon set is written as "
          "degenerate triplets by grouping the codons by first base (first bases with equal sets of remaining "
          "pairs share a letter) and then by second base (equal third-base sets share a letter); this is small "
          "but not guaranteed to be the minimum. The triplets are sorted alphabetically and combined as a "
          "product with the first residue varying fastest, as in the documented MaxItems example. gtt: Automatic "
          "(standard code), a name, Entity[\"GeneticTranslationTable\", name], an NCBI number (codes 1-5 and 11), a "
          "64-letter NCBI string (TCAG order) or a codon association. startspec: Automatic and False (no start "
          "translation; BioSequence input is not a complete protein), True (the table's start codons code M for the "
          "first residue), n (the start codons of NCBI code n), a 64-letter start translation string, or an "
          "association of start codon translations overriding the table. Peptide letters B, Z, J, X are unions, "
          "\".\" is a stop; BioSequence[\"Peptide\", n] is n unknown residues X. MaxItems (default 10000, Infinity) "
          "caps the list. Check: reference (every example of the Wolfram documentation page, e.g. I -> ATH, "
          "L -> {CTN, TTR}, M with start True -> HTG) in tests.")
def _backtranslate(x: Expr, gtt: Expr = None, *rest: Expr) -> ListOf[Expr]:
    from .wl_text import _code_table, _GENETIC_CODES, _bio_parts, _PEP_ALPHA
    fname = "BioSequenceBackTranslateList"
    pos, o = _opts(rest, {"MaxItems"}, fname)
    if isinstance(gtt, Expr) and gtt.head in ("Rule", "RuleDelayed") and is_sym(gtt.args[0], "MaxItems"):
        pos, o = [], {"MaxItems": gtt.args[1]}
        gtt = None
    if len(pos) > 1:
        raise ValueError(f"{fname}: too many arguments")
    start = pos[0] if pos else None
    mx = _max_items(o.get("MaxItems"), fname, 10000)
    if isinstance(x, str):
        seq = x
    elif isinstance(x, Expr) and x.head == "BioSequence" and len(x.args) == 2 and x.args[0] in (
            "Peptide", "CircularPeptide") and K.is_int(x.args[1]) and not isinstance(x.args[1], bool):
        seq = "X" * int(x.args[1])
    else:
        p = _bio_parts(x)
        if p is None or p[0] not in ("Peptide", "CircularPeptide"):
            if isinstance(x, (Symbol,)) or (isinstance(x, Expr) and x.head != "BioSequence"):
                raise TypeError(f"{fname}: {x!r} is not a peptide BioSequence")
            raise TypeError(f"{fname}: {x!r} is not a peptide BioSequence")
        seq = p[1]
    seq = seq.upper()
    if not set(seq) <= _PEP_ALPHA:
        raise ValueError(f"{fname}: {seq!r} is not a valid peptide sequence")
    tab, starts = _code_table(gtt, fname)
    first = tab
    if start is None or start is False or is_sym(start, "Automatic"):
        pass
    elif start is True:
        if starts is None:
            raise NotImplementedError(f"{fname}: start codons are only known for genetic codes 1, 2 and 11")
        first = dict(tab)
        first.update({c: "M" for c in starts})
    elif K.is_int(start) and not isinstance(start, bool):
        _, st = _code_table(int(start), fname)
        if st is None:
            raise NotImplementedError(f"{fname}: start codons are only known for genetic codes 1, 2 and 11")
        first = dict(tab)
        first.update({c: "M" for c in st})
    elif isinstance(start, str) and len(start) == 64:
        first, _ = _code_table(start, fname)
    elif isinstance(start, Expr) and start.head == "Association":
        st, _ = _code_table(start, fname)
        first = dict(tab)
        first.update(st)
    else:
        raise ValueError(f"{fname}: invalid start codon specification {start!r}")
    if not seq:
        return L(Expr("BioSequence", "DNA", ""))
    lists = [_cover(_letter_codons(c, first if i == 0 else tab, fname)) for i, c in enumerate(seq)]
    total = 1
    for li in lists:
        total *= len(li)
    n = int(min(total, mx))
    out = []
    # first residue varies fastest: iterate the product of the reversed lists and reverse each tuple
    for combo in itertools.islice(itertools.product(*reversed(lists)), n):
        out.append(Expr("BioSequence", "DNA", "".join(reversed(combo))))
    _ = _GENETIC_CODES
    return L(*out)
