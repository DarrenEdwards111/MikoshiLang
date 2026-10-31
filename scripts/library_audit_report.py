#!/usr/bin/env python3
"""Generate docs/library_audit.md from library.STATUS and the differential test cases."""
import collections
import importlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "library"))

from mikoshilang import library  # noqa: E402
from mikoshilang.runtime import _quarantined  # noqa: E402

MODULES = ["test_elementary", "test_special", "test_numbertheory", "test_combinatorics", "test_algebra",
           "test_calculus", "test_linalg", "test_lists_strings", "test_statistics", "test_applied",
           "test_logic_graphs", "test_numerics_transforms", "test_plots_ml", "test_kernelheads", "test_wolfram_fixes",
           "test_wolfram_batch1", "test_wolfram_batch2", "test_wolfram_batch3", "test_wolfram_batch4", "test_wolfram_batch5", "test_wolfram_batch6", "test_wolfram_w01", "test_wolfram_w02", "test_wolfram_w03", "test_wolfram_w04",
           "test_wolfram_w05", "test_wolfram_w06", "test_wolfram_w08", "test_wolfram_w07", "test_wolfram_w10", "test_wolfram_w09", "test_wolfram_w11", "test_wolfram_w13", "test_wolfram_w12", "test_wolfram_w17"]
IDENT = re.compile(r"(?<![A-Za-z0-9\"$])(\$?[A-Z][A-Za-z0-9]*)")

used = collections.Counter()
total = 0
for m in MODULES:
    cases = importlib.import_module(m).CASES
    total += len(cases)
    for c in cases:
        for n in set(IDENT.findall(c[0])):
            used[n] += 1

inventory = set((ROOT / "tests/library/inventory_heads.txt").read_text().split())
counts = collections.Counter(s for _a, s, _n in library.STATUS.values())
by_area = collections.defaultdict(collections.Counter)
for h, (a, s, _n) in library.STATUS.items():
    by_area[a][s] += 1

STATUSES = ("verified", "fixed", "promoted", "added", "kernel", "error", "quarantined")
out = []
w = out.append
w("# MikoshiLang bundled library audit\n")
w(open(ROOT / "scripts/audit_preamble.md").read())
w("\n## Status counts\n")
w("| Status | Heads | Meaning |\n|---|---|---|")
meaning = {
    "verified": "legacy behaviour confirmed correct against SymPy / mpmath / SciPy / NetworkX / Wolfram references",
    "fixed": "legacy behaviour was wrong, crashed or was non-Wolfram; reimplemented and tested",
    "promoted": "was quarantined; now properly implemented and tested",
    "added": "new Wolfram built-in added in Wolfram batch 1, differentially tested against Wolfram documentation examples",
    "kernel": "implemented by the evaluator kernel before the library is consulted; Wolfram results confirmed",
    "error": "raises NotImplementedError with the Wolfram-correct replacement (legacy meaning conflicted with Wolfram)",
    "quarantined": "raises NotImplementedError (legacy implementation wrong and not worth fixing)",
}
for s in STATUSES:
    w(f"| {s} | {counts.get(s, 0)} | {meaning[s]} |")
w(f"| **total reviewed** | **{len(library.STATUS)}** | {len(inventory)} pre-audit inventory heads, all with a status, "
  f"plus {len(set(library.STATUS) - inventory)} promoted/aliased heads |")
w(f"\nRemaining unreviewed quarantined names: **{len(_quarantined)}** (all raise NotImplementedError; see below).")
w(f"\nDifferential test cases in tests/library: **{total}**.\n")
w("\n## By area\n")
w("| Area | " + " | ".join(STATUSES) + " |\n|---|" + "---|" * len(STATUSES))
for a in sorted(by_area):
    w(f"| {a} | " + " | ".join(str(by_area[a].get(s, 0)) for s in STATUSES) + " |")
w("\n## Every function\n")
w("Cases = number of differential test cases in which the head occurs.\n")
for a in sorted(by_area):
    w(f"\n### {a}\n")
    w("| Function | Status | Cases | Notes |\n|---|---|---|---|")
    for h in sorted(x for x, v in library.STATUS.items() if v[0] == a):
        _a, s, n = library.STATUS[h]
        n = (n or "").replace("|", "\\|").replace("\n", " ")
        w(f"| `{h}` | {s} | {used.get(h, 0)} | {n} |")
w("\n## Quarantine decision for the remaining names\n")
w(open(ROOT / "scripts/audit_quarantine.md").read())
text = "\n".join(out) + "\n"
(ROOT / "docs/library_audit.md").write_text(text)
if len(sys.argv) > 1:
    pathlib.Path(sys.argv[1]).write_text(text)
print(f"wrote {len(text.splitlines())} lines; cases={total}; counts={dict(counts)}")
