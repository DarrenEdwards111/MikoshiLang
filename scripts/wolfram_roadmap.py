#!/usr/bin/env python3
"""Generate docs/wolfram_coverage_roadmap.md (and .csv) -- coverage of the documented Wolfram built-ins.

Inputs (scripts/wolfram_data/, fetched from reference.wolfram.com and wolfram.com on 2026-10-30):

* ``documented.txt`` -- every symbol linked from the Wolfram Language "Alphabetical Listing" index;
* ``guides.json``    -- the 23 top-level areas of the Documentation Center root page, the guide pages under
  each area (two levels deep) and, for every documented symbol, the guide pages that list it;
* ``eiwl_counts.txt`` -- how often each symbol is linked from the 48 chapters of *An Elementary
  Introduction to the Wolfram Language* (3rd ed.), used as a usage-frequency proxy.

Implementation status comes from the live registry: ``library.STATUS`` (verified / fixed / promoted /
kernel / added) plus the evaluator's kernel builtins and structural heads.  Run with ``--check`` to verify the
committed files are up to date.
"""
import collections
import csv
import io
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "scripts" / "wolfram_data"

from mikoshilang import library, kernel  # noqa: E402

DOCUMENTED = [s for s in (DATA / "documented.txt").read_text().split() if s]
G = json.loads((DATA / "guides.json").read_text())
AREAS = G["areas"]
GUIDES = [g for g, _a in G["guides"]]
GUIDE_AREA = dict(G["guides"])
SYMBOL_GUIDES = {s: [GUIDES[i] for i in ix] for s, ix in G["symbols"].items()}
EIWL = {}
for line in (DATA / "eiwl_counts.txt").read_text().splitlines():
    if line and not line.startswith("#"):
        s, c = line.split()
        EIWL[s] = int(c)

OTHER = "Options, system variables & unclassified"

# Areas whose functions need something MikoshiLang cannot provide.
INFEASIBLE_AREAS = {
    "Machine Learning & LLMs": "ML models / LLM services",
    "Sound & Video": "audio / video",
    "Geographic Data & Computation": "curated data",
    "Knowledge Representation & Natural Language": "curated data / natural-language understanding",
    "Scientific and Medical Data & Computation": "curated data",
    "Engineering Data & Computation": "curated data",
    "Financial Data & Computation": "curated data",
    "Social, Cultural & Linguistic Data": "curated data",
    "Notebook Documents & Presentation": "front end / notebooks",
    "User Interface Construction": "front end / notebooks",
    "Cloud & Deployment": "Wolfram Cloud",
}
# Name-based overrides: symbols that need curated data, the front end, audio, ML models or the cloud even
# when they are listed in a feasible area's guide page.
INFEASIBLE_NAMES = [
    (re.compile(r"^(Entity\w*|\w+Data|Interpreter|SemanticInterpretation|SemanticImport\w*|WolframAlpha\w*|"
                r"GeoGraphics|GeoListPlot|GeoPosition\w*|GeoNearest|GeoEntities|GeoElevationData|DictionaryLookup|"
                r"WordList|WordDefinition|WordFrequency\w*|RandomWord|TextSentences|Pluralize|LanguageIdentify|"
                r"IntegerName|SpokenString|CommonName|Alphabet|Transliterate|RandomEntity|FromEntity|ToEntity|"
                r"Quantity\w*|UnitConvert|CurrentDate|TimeZone\w*|Sunrise|Sunset|MoonPhase|Here|\$GeoLocation\w*)$"),
     "curated data"),
    (re.compile(r"^(\w*Box|\w*BoxOptions|Notebook\w*|Cell\w*|FrontEnd\w*|\$FrontEnd\w*|Dynamic\w*|Manipulate\w*|"
                r"Button\w*|Slider\w*|Checkbox\w*|InputField\w*|Panel\w*|Popup\w*|\w*Dialog\w*|Window\w*|"
                r"Selection\w*|Palette\w*|StyleSheet\w*|Docked\w*|Animate\w*|Animator\w*|Locator\w*|Tooltip\w*|"
                r"Mouse\w*|Clipboard\w*|CopyToClipboard|Paste\w*|Print\w*Notebook|CreateDocument|Toggler\w*|"
                r"ActionMenu\w*|Opener\w*|TabView\w*|MenuView\w*|FlipView\w*|SlideView\w*|PopupView\w*|"
                r"Setter\w*|RadioButton\w*|ColorSetter\w*|Control\w*|Trigger\w*|Hyperlink\w*|EventHandler\w*|"
                r"Refresh|ProgressIndicator\w*|ListAnimate)$"), "front end / notebooks"),
    (re.compile(r"^(Audio\w*|\$Audio\w*|Sound\w*|\$Sound\w*|Speak\w*|Speech\w*|Video\w*|\$Video\w*|Play|"
                r"EmitSound|Spectrogram\w*|Music\w*)$"), "audio / video"),
    (re.compile(r"^(Net\w*|\$Net\w*|\w+Layer|Classify|ClassifierFunction|ClassifierMeasurements\w*|Predict|"
                r"PredictorFunction|PredictorMeasurements\w*|LLM\w*|\$LLM\w*|Chat\w*|ImageIdentify|ImageCaptions?|"
                r"ImageRestyle|ImageSynthesize|FeatureExtract\w*|TextRecognize|SpeechRecognize|FindTextualAnswer|"
                r"Classifier\w*|Predictor\w*|SequencePredict\w*|AnomalyDetect\w*|ActiveClassification\w*|"
                r"ActivePrediction\w*|ContentDetectorFunction|FacialFeatures|FindFaces|ImageContents|"
                r"TextContents|TextCases|TextPosition|TextStructure|Vector\w*Database\w*|SemanticSearch\w*|"
                r"\$Speech\w*|FindPerson|ObjectRecognition\w*)$"), "ML models / LLM services"),
    (re.compile(r"^(Cloud\w*|\$Cloud\w*|Permissions\w*|\w*Permissions|Deploy\w*|APIFunction|FormFunction|FormPage|"
                r"FormObject|AskFunction|ScheduledTask\w*|\$Scheduled\w*|Resource\w*|\$Resource\w*|Databin\w*|"
                r"Channel\w*|\$Channel\w*|SendMail|MailReceiver\w*|Service\w*|\$Service\w*|URLDispatcher|"
                r"Wolfram\w*Account|\$Wolfram(ID|UUID)|AutoRefreshed|Delayed|EmbedCode|\$Requester\w*|"
                r"PersistentObject\w*|PersistentSymbol|\$Persistence\w*|LocalObject\w*|"
                r"Blockchain\w*|\$Blockchain\w*|Cryptocurrency\w*|ExternalBundle|ExternalSessions?\w*)$"),
     "Wolfram Cloud / external services"),
]

# Individually reviewed symbols (symbol -> reason) that need curated data, the front end, the cloud, ML models,
# or other Wolfram infrastructure even though their guide pages sit in feasible areas.
INFEASIBLE_SYMBOLS = json.loads((DATA / "infeasible_names.json").read_text())

# Guide pages whose functions are the most-used core of the language (with EIWL usage, priority 1).
CORE_GUIDES = set("""ListManipulation ConstructingLists ElementsOfLists RearrangingAndRestructuringLists
ApplyingFunctionsToLists MathematicalAndCountingOperationsOnLists Associations StringManipulation StringOperations
StringPatterns CharacterOperations ConvertingBetweenExpressionsAndStrings FunctionalProgramming FunctionalIteration
FunctionCompositionAndOperatorForms Patterns Rules PatternMatchingFunctions TestingExpressions ProceduralProgramming
LoopingConstructs Conditionals FlowControl Assignments ScopingConstructs ExpressionStructure PartsOfExpressions
StructuralOperationsOnExpressions EvaluationControl Attributes VariablesAndFunctions MathematicalFunctions
ElementaryFunctions AlgebraicManipulation FormulaManipulation Calculus EquationSolving MatricesAndLinearAlgebra
IntegerFunctions NumberTheory DescriptiveStatistics HandlingArraysOfData""".split())
CORE_AREAS = {"Core Language & Structure", "Data Manipulation & Analysis", "Symbolic & Numeric Computation",
              "Strings & Text", "Higher Mathematical Computation", "Graphs & Networks", "Geometry",
              "Time-Related Computation"}


def implemented_names():
    """Names whose evaluation MikoshiLang implements with reviewed, tested Wolfram semantics."""
    lib = {n for n, (_a, s, _n) in library.STATUS.items() if s in ("verified", "fixed", "promoted", "kernel",
                                                                  "added")}
    return lib | set(kernel._BUILTINS) | set(kernel._HELD) | set(kernel.STRUCTURAL_HEADS)


def primary_area(sym):
    guides = SYMBOL_GUIDES.get(sym) or []
    if not guides:
        return OTHER
    c = collections.Counter(GUIDE_AREA[g] for g in guides)
    return max(c, key=lambda a: (c[a], -AREAS.index(a)))


# Symbols that the name/area rules above would wrongly mark infeasible (they are plain numeric containers or
# models, not curated data or audio): SpatialPointData (spatial point collections, not curated data) is
# implemented in batch X08.
FEASIBLE_NAMES = {"SpatialPointData"}


def feasibility(sym):
    if sym in FEASIBLE_NAMES:
        return ""
    for rx, why in INFEASIBLE_NAMES:
        if rx.match(sym):
            return why
    if sym in INFEASIBLE_SYMBOLS:
        return INFEASIBLE_SYMBOLS[sym]
    guides = SYMBOL_GUIDES.get(sym) or []
    areas = {GUIDE_AREA[g] for g in guides}
    if areas and all(a in INFEASIBLE_AREAS for a in areas):
        return INFEASIBLE_AREAS[primary_area(sym)]
    return ""


def priority(sym, area):
    guides = set(SYMBOL_GUIDES.get(sym) or [])
    uses = EIWL.get(sym, 0)
    if uses >= 2 or (guides & CORE_GUIDES and (uses or area in ("Core Language & Structure", "Strings & Text"))):
        return 1
    if area in CORE_AREAS or uses:
        return 2
    return 3


def build():
    impl = implemented_names()
    rows = []
    for s in DOCUMENTED:
        area = primary_area(s)
        why = feasibility(s)
        done = s in impl
        if done:
            status = "implemented"
        elif why:
            status = "infeasible"
        else:
            status = "missing"
        rows.append(dict(symbol=s, area=area, status=status, priority="" if status == "infeasible" else priority(s, area),
                         reason=why if status == "infeasible" else "", eiwl=EIWL.get(s, 0),
                         guides=" ".join((SYMBOL_GUIDES.get(s) or [])[:4])))
    extra = sorted(n for n in impl if n not in set(DOCUMENTED) and re.match(r"^[A-Z$]", n))
    return rows, extra


def render(rows, extra):
    total = len(rows)
    by = collections.Counter(r["status"] for r in rows)
    feasible = total - by["infeasible"]
    out = []
    w = out.append
    w("# Wolfram Language coverage roadmap\n")
    w("Generated by `scripts/wolfram_roadmap.py` from the Wolfram Documentation Center index "
      "(reference.wolfram.com, fetched 2026-10-30) and MikoshiLang's live function registry. "
      "Do not edit by hand; run `python scripts/wolfram_roadmap.py` after adding functions "
      "(`tests/test_roadmap.py` fails when this file is stale).\n")
    w("## Coverage\n")
    w("| | Symbols |\n|---|---|")
    w(f"| Documented Wolfram Language symbols (Alphabetical Listing) | **{total}** |")
    w(f"| Implemented in MikoshiLang with reviewed, tested Wolfram semantics | **{by['implemented']}** "
      f"({100 * by['implemented'] / total:.1f}%) |")
    w(f"| Feasible (not infeasible by the rules below) | **{feasible}** — implemented {by['implemented']} "
      f"({100 * by['implemented'] / feasible:.1f}% of feasible), missing {by['missing']} |")
    w(f"| Infeasible without curated data, the front end/notebooks, audio/video, ML models or the cloud "
      f"| **{by['infeasible']}** |")
    pr = collections.Counter((r["priority"], r["status"]) for r in rows if r["status"] != "infeasible")
    w("\n| Priority | Implemented | Missing | Total |\n|---|---|---|---|")
    for p in (1, 2, 3):
        i, m = pr[(p, "implemented")], pr[(p, "missing")]
        w(f"| P{p} | {i} | {m} | {i + m} |")
    w(f"\nMikoshiLang also implements {len(extra)} names that are not documented Wolfram symbols "
      "(legacy MikoshiLang extensions and aliases); they are not counted above.\n")
    w("## Method\n")
    w("* **Universe**: every symbol linked from the Wolfram Language *Alphabetical Listing* "
      "(`guide/AlphabeticalListing`), the documented built-ins (functions, options, constants and system "
      "variables).\n"
      "* **Area**: the Documentation Center's 23 top-level areas. Each guide page linked from an area (and the "
      "guides linked from those, two levels deep) is assigned to that area; a symbol's area is the one whose "
      "guides list it most often. Symbols listed on no guide page (mostly options and `$` variables) are "
      f"grouped under *{OTHER}*.\n"
      "* **Priority** (usage frequency proxy): **P1** = linked at least twice from *An Elementary Introduction to "
      "the Wolfram Language* (3rd ed.), or listed on a core-language guide page (lists, strings, associations, "
      "functional programming, patterns, rules, scoping, testing, elementary maths) and either used in that book or "
      "in the Core Language or Strings areas; **P2** = other symbols in the core computational areas, or used at "
      "least once in the book; **P3** = everything else that is feasible.\n"
      "* **Infeasible**: symbols whose every guide page lies in an area that needs curated Wolfram data "
      "(geographic, scientific, engineering, financial, cultural and linguistic data, entities, natural-language "
      "understanding), the front end or notebooks (documents, UI construction), audio or video, trained ML models "
      "or LLM services, or the Wolfram Cloud; plus name-based overrides for such symbols listed in other areas "
      "(`*Data` functions, `Entity*`, `Cloud*`, `Notebook*`, `*Box`, `Audio*`, `Net*`, `LLM*`, ...), and "
      f"{len(INFEASIBLE_SYMBOLS)} individually reviewed symbols listed with their reasons in "
      "`scripts/wolfram_data/infeasible_names.json`. "
      "Infeasible does not mean impossible: a local substitute (e.g. a small bundled dataset) could be added "
      "later, but it would not reproduce Wolfram's results.\n"
      "* **Implemented**: the head has a `verified`, `fixed`, `promoted`, `kernel` or `added` status in "
      "`mikoshilang.library.STATUS` (each enforced by differential tests in `tests/library`), or is evaluated by "
      "the kernel itself (builtins, held special forms and structural heads such as `List`, `Rule`, `Function` "
      "and pattern objects). Quarantined and error heads count as missing.\n")
    w("## By area\n")
    w("| Area | Documented | Implemented | Missing P1 | Missing P2 | Missing P3 | Infeasible |\n"
      "|---|---|---|---|---|---|---|")
    by_area = collections.defaultdict(list)
    for r in rows:
        by_area[r["area"]].append(r)
    order = [a for a in AREAS if a in by_area] + ([OTHER] if OTHER in by_area else [])
    for a in order:
        rs = by_area[a]
        c = collections.Counter(r["status"] for r in rs)
        mp = collections.Counter(r["priority"] for r in rs if r["status"] == "missing")
        w(f"| [{a}](#{_anchor(a)}) | {len(rs)} | {c['implemented']} | {mp[1]} | {mp[2]} | {mp[3]} | {c['infeasible']} |")
    w("")
    for a in order:
        rs = by_area[a]
        w(f"### {a}\n")
        impl = [r["symbol"] for r in rs if r["status"] == "implemented"]
        w(f"**Implemented ({len(impl)})**: " + (", ".join(f"`{s}`" for s in impl) or "none") + "\n")
        for p in (1, 2, 3):
            miss = sorted((r for r in rs if r["status"] == "missing" and r["priority"] == p),
                          key=lambda r: (-r["eiwl"], r["symbol"]))
            if miss:
                w(f"**Missing, P{p} ({len(miss)})**: " + ", ".join(f"`{r['symbol']}`" for r in miss) + "\n")
        inf = collections.defaultdict(list)
        for r in rs:
            if r["status"] == "infeasible":
                inf[r["reason"]].append(r["symbol"])
        for why in sorted(inf):
            w(f"**Infeasible — {why} ({len(inf[why])})**: " + ", ".join(f"`{s}`" for s in inf[why]) + "\n")
    w("## MikoshiLang names that are not documented Wolfram symbols\n")
    w(", ".join(f"`{s}`" for s in extra) + "\n")
    return "\n".join(out)


def _anchor(a):
    return re.sub(r"[^a-z0-9 -]", "", a.lower()).replace(" ", "-")


def render_csv(rows):
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=["symbol", "area", "status", "priority", "reason", "eiwl", "guides"],
                        lineterminator="\n")
    wr.writeheader()
    for r in rows:
        wr.writerow(r)
    return buf.getvalue()


def main(argv):
    rows, extra = build()
    md, cs = render(rows, extra), render_csv(rows)
    md_path, csv_path = ROOT / "docs/wolfram_coverage_roadmap.md", ROOT / "docs/wolfram_coverage_roadmap.csv"
    if "--check" in argv:
        stale = [p.name for p, t in ((md_path, md), (csv_path, cs)) if not p.exists() or p.read_text() != t]
        if stale:
            print("stale: " + ", ".join(stale) + " (run python scripts/wolfram_roadmap.py)")
            return 1
        return 0
    md_path.write_text(md)
    csv_path.write_text(cs)
    by = collections.Counter(r["status"] for r in rows)
    print(f"documented {len(rows)}; implemented {by['implemented']}; missing {by['missing']}; "
          f"infeasible {by['infeasible']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
