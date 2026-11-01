"""Differential tests for Wolfram batch X01 (core, compile constructs and system extras; modules wl_core_extras,
wl_core_extras_pkg, wl_core_extras_sys, wl_core_extras_url, wl_core_extras_compile, wl_core_extras_math).

Sources of the expected values:

* Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>): the documented examples, e.g.
  ``Trace[f[1, 2 + 3], TraceOriginal -> True]``, ``Normal@Symmetrize[{{a, b}, {c, d}}]``, the ComplexExpand
  TargetFunctions forms of ``Re[Tan[z]]``, every BioSequenceBackTranslateList example, the SPF tutorial package
  (tutorial/StructuredPackageFormat: Func1, Func2, utilFunc), the compile-construct examples (AiryAi[1.1] =
  0.120049, square[100] = 10000, factorialC[10] = 3628800, foo[3, 5.6], cf[4] = 4, the CArray example 8),
  ``NumberFieldNormRepresentatives[I, 5]`` and the classes of ``[Sqrt[2], 7]``, the 50 digits of
  ``N[SpheroidalRadialFactor[7, 0, 1/4], 50]``, the NBodySimulation three-body state at t = 4 and the default
  $MaxExtraPrecision = 50.
* Independent computations: mpmath (Sin[2^1000] to 32 digits), the closed-form two-body free-fall time
  (Pi/2) Sqrt[r0^3/(2 G M)] and the harmonic two-body solution, os/stat for the temporary file trees, the bytes
  served by the local test server (``_w30_servers``) and exact Gaussian/quadratic integer arithmetic for the norm
  representatives.

Safety: network cases talk only to the local server started by the module fixture on 127.0.0.1; files are written
only below temporary directories created by the fixture. Symbols assigned in cases are prefixed ``x01``.
"""
import math
import os
import shutil
import tempfile

import mpmath
import pytest
import sympy as sp

from _harness import Pred, Raises, check, ids
from _w30_servers import HTTPServer
from mikoshilang.expr import Expr


def _ev(src):
    from _harness import ev
    return ev(src)


def _all_true(r):
    return isinstance(r, Expr) and r.head == "List" and all(v is True for v in r.args)


def _num(x):
    return complex(sp.N(sp.sympify(str(x).replace("^", "**")))) if not isinstance(x, (int, float)) else x


# --------------------------------------------------------------------------------------------------- core (part 1)

def _sin_2_1000():
    with mpmath.workdps(60):
        return float(mpmath.sin(mpmath.mpf(2) ** 1000))


CORE = [
    # Symmetrize (Wolfram documentation examples)
    ("Normal[Symmetrize[{{a, b}, {c, d}}]]", "{{a, (b + c)/2}, {(b + c)/2, d}}"),
    ("Normal[Symmetrize[{{a, b}, {c, d}}, Antisymmetric[{1, 2}]]]", "{{0, (b - c)/2}, {(-b + c)/2, 0}}"),
    ("Symmetrize[x01T, Symmetric[{1, 2, 3}]]",
     "(x01T + TensorTranspose[x01T, {2, 1}] + TensorTranspose[x01T, {1, 3, 2}] + TensorTranspose[x01T, {2, 3, 1}]"
     " + TensorTranspose[x01T, {3, 1, 2}] + TensorTranspose[x01T, {3, 2, 1}])/6"),
    ("Normal[Symmetrize[Array[x01a, {2, 2, 2}], Symmetric[{1, 2, 3}]]][[1, 1, 2]]",
     "(x01a[1, 1, 2] + x01a[1, 2, 1] + x01a[2, 1, 1])/3"),
    ("Symmetrize[x01T]", "Symmetrize[x01T]"),                                       # symbolic, rank unknown
    ("Symmetrize[{{1, 2}, {3, 4}}, Symmetric[{1, 3}]]", Raises(ValueError)),
    ("x01s = Normal[Symmetrize[{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}, Antisymmetric[{1, 2}]]]; x01s == -Transpose[x01s]",
     True),                                                                           # Check: invariance
    # NCache (inert numeric cache)
    ("NCache[1/2, 0.5]", Pred(lambda r: isinstance(r, Expr) and r.head == "NCache" and len(r.args) == 2,
                              "stays NCache")),
    ("NCache[{Pi, Sqrt[Pi]}, N[{Pi, Sqrt[Pi]}]][[2, 1]]", math.pi),
    ("NCache[1/2, 0.7]", Raises(ArithmeticError)),                                  # Check: value must agree
    ("NCache[x01a, x01b]", "NCache[x01a, x01b]"),
    ('NCache[1/2, "x"]', Raises(TypeError)),
    ("NCache[1/2]", Raises(ValueError)),
    ("Abs[NCache[1/4, 0.25][[2]] - N[NCache[1/4, 0.25][[1]]]] < 10^-12", True),      # Check: value agrees
    # TraceOriginal (Wolfram documentation example)
    ("Trace[x01f[1, 2 + 3]]", Pred(lambda r: str(r) == "{{2 + 3, 5}, x01f[1, 5]}", "documented default trace")),
    ("Trace[x01f[1, 2 + 3], TraceOriginal -> True]",
     Pred(lambda r: str(r) == "{x01f[1, 2 + 3], {x01f}, {1}, {2 + 3, {Plus}, {2}, {3}, 2 + 3, 5}, x01f[1, 5]}",
          "documented TraceOriginal -> True trace")),
    ("Trace[x01f[1, 2 + 3], TraceOriginal -> Automatic]",
     Pred(lambda r: str(r) == "{x01f[1, 2 + 3], {2 + 3, 5}, x01f[1, 5]}", "documented Automatic trace")),
    ("Trace[x01f[1, 2 + 3], TraceOriginal -> False]", Pred(lambda r: str(r) == "{{2 + 3, 5}, x01f[1, 5]}",
                                                             "default trace")),
    ("Trace[x01f[1, 2 + 3], TraceOriginal -> 3]", Raises(ValueError)),
    # $MaxExtraPrecision
    ("$MaxExtraPrecision", "50"),                                                     # documented default
    ("Block[{$MaxExtraPrecision = 1000}, $MaxExtraPrecision]", "1000"),
    ("Block[{$MaxExtraPrecision = Infinity}, $MaxExtraPrecision]", "Infinity"),
    ("Block[{$MaxExtraPrecision = Infinity}, N[Sin[2^1000], 32]]", _sin_2_1000(), 1e-15),   # mpmath
    ("$MaxExtraPrecision = -1", Raises(ValueError)),
    ('$MaxExtraPrecision = "a"', Raises(ValueError)),
    # AbsoluteOptions
    ("AbsoluteOptions[Graphics[{Point[{0, 0}], Point[{2, 1}]}], PlotRange]", "{PlotRange -> {{0., 2.}, {0., 1.}}}"),
    ("AbsoluteOptions[Graphics[{Point[{0, 0}], Point[{2, 1}]}], AspectRatio]", "{AspectRatio -> 0.5}"),
    ("AbsoluteOptions[Graphics[{Point[{0, 0}]}, PlotRange -> {{0, 1}, {0, 2}}], {PlotRange, AspectRatio}]",
     "{PlotRange -> {{0., 1.}, {0., 2.}}, AspectRatio -> 2.}"),
    ("AbsoluteOptions[Graph[{1 -> 2}, Background -> Red], Background]", "{Background -> RGBColor[1, 0, 0]}"),
    ("AbsoluteOptions[x01g, PlotRange]", "AbsoluteOptions[x01g, PlotRange]"),
    ("AbsoluteOptions[Graphics[Point[{0, 0}]], Lighting]", Raises(NotImplementedError)),
    ("AbsoluteOptions[Graphics[Point[{0, 0}]]]", Raises(NotImplementedError)),
    ("AbsoluteOptions[Graphics[Point[{0, 0}]], 5]", Raises(TypeError)),
    ("AbsoluteOptions[Graphics[Line[{{-1, 3}, {4, 5}}]], PlotRange] == {PlotRange -> {{-1., 4.}, {3., 5.}}}",
     True),                                                                           # Check: bounding box
    # TargetFunctions (ComplexExpand, documentation examples)
    ("ComplexExpand[Re[Tan[z]], z]", "Sin[2 Re[z]]/(Cos[2 Re[z]] + Cosh[2 Im[z]])"),
    ("ComplexExpand[Re[Tan[z]], z, TargetFunctions -> {Abs, Arg}]",
     "Sin[2 Abs[z] Cos[Arg[z]]]/(Cos[2 Abs[z] Cos[Arg[z]]] + Cosh[2 Abs[z] Sin[Arg[z]]])"),
    ("ComplexExpand[Re[Tan[z]], z, TargetFunctions -> Conjugate]",
     "Sin[z + Conjugate[z]]/(Cos[z - Conjugate[z]] + Cos[z + Conjugate[z]])"),
    ("ComplexExpand[Abs[z]^2, z, TargetFunctions -> {Re, Im}]", "Re[z]^2 + Im[z]^2"),
    ("ComplexExpand[Re[Tan[z]], z, TargetFunctions -> {Sin}]", Raises(NotImplementedError)),
    ("Abs[(ComplexExpand[Re[Tan[z]], z, TargetFunctions -> {Abs, Arg}] /. z -> 0.3 + 0.7 I) - Re[Tan[0.3 + 0.7 I]]]"
     " < 10^-12", True),                                                              # Check: numeric cross-check
    ("TargetFunctions", "TargetFunctions"),
    # register_only options, exercised through the functions that honour them
    ("FailureAction", "FailureAction"),
    ("Query[All, 1/# &, FailureAction -> \"Drop\"][{1, 0, 2}]", "{1, 1/2}"),
    ("Query[All, 1/# &, FailureAction -> \"Bogus\"][{1, 0, 2}]", Raises(ValueError)),
    ("MissingBehavior", "MissingBehavior"),
    ("Query[All, \"a\", MissingBehavior -> None][{<|\"a\" -> 1|>, <|\"b\" -> 2|>}]", "{1, Missing[\"KeyAbsent\", \"a\"]}"),
    ("Query[All, \"a\", MissingBehavior -> \"Bogus\"][{<|\"a\" -> 1|>}]", Raises(ValueError)),
    ("Resampling", "Resampling"),
    ("ArrayResample[{1, 2, 3}, 4, Resampling -> \"Nearest\"]", "{1, 2, 2, 3}"),           # nearest of 0, 2/3, 4/3, 2
    ("ArrayResample[{1, 2, 3}, 5, Resampling -> \"NearestLeft\"]", "{1, 1, 2, 2, 3}"),
    ("ArrayResample[{1, 2, 3}, 5, Resampling -> \"Bogus\"]", Raises(NotImplementedError)),
    ("Parallelization", "Parallelization"),
    ("Compile[{x}, x^2, Parallelization -> True][3.]", 9.0),
    ("Compile[{x}, x^2, Parallelization -> 7]", Raises(ValueError)),
    ("RuntimeAttributes", "RuntimeAttributes"),
    ("Compile[{{x, _Real}}, x^2, RuntimeAttributes -> {Listable}][{1., 2.}]", "{1., 4.}"),
    ("Compile[{x}, x^2, RuntimeAttributes -> {Orderless}]", Raises(ValueError)),
    ("RuntimeOptions", "RuntimeOptions"),
    ("Compile[{x}, x^2, RuntimeOptions -> \"Speed\"][2.]", 4.0),
    ("Compile[{x}, x^2, RuntimeOptions -> \"Quality\"][3.]", 9.0),
    ("IncludeDirectories", "IncludeDirectories"),
    # MessageList (main-loop behaviour is in test_x01_messagelist below)
    ("MessageList[x01n]", "MessageList[x01n]"),
    ('MessageList["a"]', Raises(TypeError)),
    ("MessageList[1.5]", Raises(TypeError)),
]


def test_x01_messagelist():
    """Wolfram documentation example pattern (there with LinearSolve::nosol, which raises in MikoshiLang): a message
    issued on line 1 is reported by MessageList[-1] on line 2, MessageList[-1] on line 3 is {} and MessageList[-3]
    on line 4 is again the message."""
    from mikoshilang import Session
    s = Session()
    s.evaluate_input("Message[General::argx, x01f, 2]")
    a = s.evaluate_input("MessageList[-1]")
    b = s.evaluate_input("MessageList[-1]")
    c = s.evaluate_input("MessageList[-3]")
    d = s.evaluate_input("Map[MessageList, Range[4]]")
    assert "argx" in str(a) and str(b).endswith("{})]") and "argx" in str(c), (a, b, c)
    assert "argx" in str(d) and str(d).count("argx") == 1, d


# --------------------------------------------------------------------------------------------------- SPF packages

PKG = [
    ('PackageInitialize["X01A`", "{S}/X01SPFExample"]; {X01A`Func1[2], X01A`Func2[3]}', "{3, 9}"),   # SPF tutorial
    ('PackageInitialize["X01B`", "{S}/X01SPFExample"]; Names["X01B`PackageScope`*"]',
     '{"X01B`PackageScope`utilFunc"}'),
    ('PackageInitialize["X01C`", "{S}/X01SPFExample"]; {MemberQ[$ContextPath, "X01C`"], '
     'MemberQ[$ContextPath, "X01C`PackageScope`"], MemberQ[$Packages, "X01C`"]}', "{True, False, True}"),
    ('Get["{S}/X01SPFExample/init.wl"]; X01SPFExample`Func2[4]', "16"),               # loader file
    ('Get["{S}/Single/X01Single.wl"]; {X01Single`Function1[3], Names["X01Single`privateFunc"]}', "{10, {}}"),
    ('PackageInitialize["X01P`", "{S}/X01SPFExample", <|"SymbolsToProtect" -> All|>]; '
     'MemberQ[Attributes[X01P`Func1], Protected]', True),
    ('PackageInitialize["X01Q`", "{S}/X01SPFExample", <|"IgnoreFiles" -> {"PackageFile2.wl"}|>]; Names["X01Q`Func*"]',
     '{"Func1"}'),
    ('PackageInitialize["X01Z`"]', Pred(lambda r: isinstance(r, Expr) and r.head == "Failure"
                                         and r.args[0] == "RootDirNotAvailable", "documented Failure")),
    ('PackageInitialize["X01R`", "{S}/X01SPFExample", <|"Bogus" -> 1|>]', Raises(ValueError)),
    ('PackageInitialize["X01R`", "{S}/nodir"]', Raises(ValueError)),
    ("PackageInitialize[5]", Raises(ValueError)),
    ('PackageInitialize["X01D`", "{S}/X01SPFExample"]; X01D`Func2[5] == X01D`Func1[5]^2 - 2 X01D`Func1[5] + 1', True),
    # PackageImport: import by name (only the named symbol is visible), Needs form, alias, nosym
    ('PackageInitialize["X01I`", "{S}/X01Imp"]; {X01I`useA[1], X01I`useB[1]}',
     Pred(lambda r: r.args[0] == 10 and r.args[1] != 100, "LibA imported, LibB not visible")),
    ('PackageInitialize["X01J`", "{S}/X01Bad"]', Raises(ValueError, "nosym")),
    ('PackageImport["X01Lib`"]; MemberQ[$ContextPath, "X01Lib`"]', True),
    ('PackageImport["X01Lib`" -> "x01l`"]; ToExpression["x01l`LibA[2]"]', "20"),
    ('PackageImport["X01Lib`", x01q]', Raises(ValueError)),
    ('PackageImport["X01NoSuchPackage`"]', Raises(ValueError)),
    ("PackageExported[x01q]", Raises(ValueError)),
    ("PackageExported[{x01q, x01r}]", Raises(ValueError)),
    ('Attributes[PackageExported]', Pred(lambda r: "HoldAll" in str(r), "HoldAll")),
    ('PackageInitialize["X01E`", "{S}/X01SPFExample"]; Context[X01E`Func1]', '"X01E`"'),
    ("PackageScoped[x01q]", Raises(ValueError)),
    ("PackageScoped[{x01q}]", Raises(ValueError)),
    ('Attributes[PackageScoped]', Pred(lambda r: "HoldAll" in str(r), "HoldAll")),
    ('PackageInitialize["X01F`", "{S}/X01SPFExample"]; Context[X01F`PackageScope`utilFunc]', '"X01F`PackageScope`"'),
]

SPF_FILES = {
    "X01SPFExample/PackageFile1.wl": "PackageExported[Func1]\n\nFunc1[x_] := x+1\n\nPackageScoped[utilFunc]\n\n"
                                     "utilFunc[n_] := n^2\n",
    "X01SPFExample/PackageFile2.wl": "PackageExported[Func2]\n\nFunc2[x_] := utilFunc[x]\n",
    "X01SPFExample/init.wl": 'PackageInitialize["X01SPFExample`"]\n',
    "Single/X01Single.wl": 'PackageInitialize["X01Single`", None]\n\nPackageExported[Function1]\n'
                           "Function1[x_] := privateFunc[x] + 1\n\nprivateFunc[x_] := x^2\n",
    "X01Lib.wl": 'BeginPackage["X01Lib`"]\nLibA::usage = "a"\nLibB::usage = "b"\nBegin["`Private`"]\n'
                 "LibA[x_] := 10 x\nLibB[x_] := 100 x\nEnd[]\nEndPackage[]\n",
    "X01Imp/main.wl": 'PackageImport["X01Lib`", LibA]\n\nPackageExported[useA]\n\nuseA[x_] := LibA[x]\n\n'
                      "useB[x_] := LibB[x]\n\nPackageExported[useB]\n",
    "X01Bad/main.wl": 'PackageImport["X01Lib`", DoesntExist]\n',
}

# --------------------------------------------------------------------------------------------------- files


def _tree(d):
    """Expected FileSystemTree of the fixture directory, written out by hand from the files the fixture creates."""
    j = lambda *p: os.path.join(d, *p).replace("\\", "/")  # noqa: E731
    return (f'Tree["{d}", <|"a.txt" -> Tree["{j("a.txt")}", None], "empty" -> Tree["{j("empty")}", None], '
            f'"sub" -> Tree["{j("sub")}", <|"b.wl" -> Tree["{j("sub", "b.wl")}", None], '
            f'"deep" -> Tree["{j("sub", "deep")}", <|"c.txt" -> Tree["{j("sub", "deep", "c.txt")}", None]|>]|>]|>]')


FILES = [
    ('FileSystemTree["{D}"]', "{TREE}"),
    ('FileSystemTree["{D}", 1]', 'Tree["{D}", <|"a.txt" -> Tree["{D}/a.txt", None], "empty" -> Tree["{D}/empty", None],'
                                 ' "sub" -> Tree["{D}/sub", None]|>]'),
    ('FileSystemTree["{D}", {2}]', 'Tree["{D}", <|"sub" -> Tree["{D}/sub", <|"b.wl" -> Tree["{D}/sub/b.wl", None], '
                                   '"deep" -> Tree["{D}/sub/deep", None]|>]|>]'),
    ('FileSystemTree["{D}", MaxItems -> 1]', 'Tree["{D}", <|"a.txt" -> Tree["{D}/a.txt", None]|>]'),
    ('FileSystemTree["{D}", 1, IncludeDirectories -> True]',
     'Tree["{D}", <|"a.txt" -> Tree["{D}/a.txt", None], "empty" -> Tree["{D}/empty", None], '
     '"sub" -> Tree["{D}/sub", None], All -> Tree["{D}", None]|>]'),
    ('FileSystemTree["{D}", 1, IncludeDirectories -> False]',
     'Tree["{D}", <|"a.txt" -> Tree["{D}/a.txt", None], "empty" -> Tree["{D}/empty", None], '
     '"sub" -> Tree["{D}/sub", None]|>]'),
    ('Sort[TreeData /@ TreeLeaves[FileSystemTree["{D}", FileNameForms -> "*.txt"]]]',
     'Sort[{"{D}/a.txt", "{D}/empty", "{D}/sub/deep/c.txt"}]'),
    ('Sort[TreeData /@ TreeLeaves[FileSystemTree["{D}"]]] == Sort[Join[FileNames["*", "{D}", Infinity] /. '
     's_String /; DirectoryQ[s] && FileNames["*", s] =!= {} -> Nothing]]', True),       # Check: same files
    ('FileSystemTree["{D}", 1, 1]', Raises(NotImplementedError)),
    ('FileSystemTree["{D}/nope"]', Raises(ValueError)),
    ("FileSystemTree[5]", Raises(TypeError)),
    # TextSearchReport
    ('Normal[TextSearchReport["{D}", "dog"]][[All, {"Title", "FileName", "FileByteCount", "FileExtension"}]]',
     '{<|"Title" -> "a", "FileName" -> "a.txt", "FileByteCount" -> 13, "FileExtension" -> "txt"|>, '
     '<|"Title" -> "c", "FileName" -> "c.txt", "FileByteCount" -> 15, "FileExtension" -> "txt"|>}'),
    ('Normal[TextSearchReport["{D}", "lorem"]][[All, "Location"]]', '{File["{D}/sub/deep/c.txt"]}'),
    ('Normal[TextSearchReport["{D}", "zebra"]]', "{}"),
    ('Head[TextSearchReport["{D}", "dog"]]', "Dataset"),
    ('Normal[TextSearchReport["{D}", "dog"]][[1, "ModificationDate"]] == FileDate["{D}/a.txt", "Modification"]',
     True),                                                                           # Check: matches FileDate
    ('Normal[TextSearchReport["{D}", "dog"]][[All, "FileByteCount"]] == FileByteCount /@ {"{D}/a.txt", '
     '"{D}/sub/deep/c.txt"}', True),
    ('TextSearchReport[x01src, "dog"]', 'TextSearchReport[x01src, "dog"]'),
    ('TextSearchReport["{D}/nope", "dog"]', Raises(ValueError, "TextSearchReport")),
]

# --------------------------------------------------------------------------------------------------- BioSequence

_STD = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"   # NCBI table 1, TCAG order

BIO = [   # every example of the BioSequenceBackTranslateList documentation page
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "I"]]', '{BioSequence["DNA", "ATH"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "L"]]',
     '{BioSequence["DNA", "CTN"], BioSequence["DNA", "TTR"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "I"], "VertebrateMitochondrial"]',
     '{BioSequence["DNA", "ATY"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "I"], '
     '"GGGGEEDDAAAAVVVVRRSSKKNNTTTTMIIIRRRRQQHHPPPPLLLLW*CC**YYSSSSLLFF"]', '{BioSequence["DNA", "CGV"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "M"], Automatic, True]', '{BioSequence["DNA", "HTG"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "M"], Automatic, False]', '{BioSequence["DNA", "ATG"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "M"], Automatic, <|"CTG" -> "M"|>]',
     '{BioSequence["DNA", "MTG"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "M"], Automatic, '
     '"GGGGEEDDAAAAVVVVRRSSKKNNTTTTMIIIRRRRQQHHPPPPLLLLW*CC**YYSSSSLLFF"]', '{BioSequence["DNA", "CGT"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", 12], MaxItems -> 3]',
     '{BioSequence["DNA", "TAYTAYTAYTAYTAYTAYTAYTAYTAYTAYTAYTAY"], BioSequence["DNA", '
     '"TGBTAYTAYTAYTAYTAYTAYTAYTAYTAYTAYTAY"], BioSequence["DNA", "TYNTAYTAYTAYTAYTAYTAYTAYTAYTAYTAYTAY"]}'),
    ('Length[BioSequenceBackTranslateList[BioSequence["Peptide", 9]]]', "10000"),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "GENETICS"], MaxItems -> 1]',
     '{BioSequence["DNA", "GGNGARAAYGARACNATHTGYAGY"]}'),
    ('BioSequenceTranslate[First[BioSequenceBackTranslateList[BioSequence["Peptide", "GENETICS"], MaxItems -> 1]]]',
     'BioSequence["Peptide", "GENETICS"]'),                                            # Check: round trip
    ('Flatten[BioSequenceInstances /@ BioSequenceBackTranslateList[BioSequence["Peptide", "L"]]]',
     '{BioSequence["DNA", "CTA"], BioSequence["DNA", "CTC"], BioSequence["DNA", "CTG"], BioSequence["DNA", "CTT"], '
     'BioSequence["DNA", "TTA"], BioSequence["DNA", "TTG"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "I"], "' + _STD + '"]', '{BioSequence["DNA", "ATH"]}'),
    ('BioSequenceBackTranslateList[BioSequence["Peptide", "I"], "NoSuchTable"]', Raises(ValueError)),
    ('BioSequenceBackTranslateList[5]', Raises(TypeError)),
    ('BioSequenceBackTranslateList[x01pep]', Raises(TypeError)),
]

# --------------------------------------------------------------------------------------------------- URL tasks

URL = [
    ('x01res = {}; x01t = URLSubmit["http://127.0.0.1:{P}/text", HandlerFunctions -> <|"BodyReceived" -> '
     '(AppendTo[x01res, #] &)|>, HandlerFunctionsKeys -> {"StatusCode", "Body"}]; TaskWait[x01t]; x01res',
     '{<|"StatusCode" -> 200, "Body" -> "hello from w30"|>}'),
    ('x01ch = {}; x01t = URLSubmit["http://127.0.0.1:{P}/big", HandlerFunctions -> <|"BodyChunkReceived" -> '
     '(AppendTo[x01ch, #] &)|>, HandlerFunctionsKeys -> {"ByteCountDownloaded", "FractionComplete"}]; '
     'TaskWait[x01t]; {Length[x01ch], x01ch[[All, "ByteCountDownloaded"]], x01ch[[-1, "FractionComplete"]]}',
     "{3, {8192, 16384, 20000}, 1.}"),                                                 # 20000 bytes in 8192 chunks
    ('x01t = URLSubmit["http://127.0.0.1:{P}/text"]; TaskWait[x01t]["EvaluationResult"]["StatusCode"]', "200"),
    ('x01t = URLSubmit["http://127.0.0.1:{P}/echo", {"a" -> "1"}]; '
     'ImportString[TaskWait[x01t]["EvaluationResult"]["Body"], "RawJSON"]["path"]', '"/echo?a=1"'),
    ('x01cf = {}; x01t = URLSubmit["http://127.0.0.1:{F}/x", HandlerFunctions -> <|"ConnectionFailed" -> '
     '(AppendTo[x01cf, #] &)|>]; TaskWait[x01t]; x01cf[[1, "Failure"]][[1]]', '"ConnectionFailure"'),
    ('x01ck = {}; x01t = URLSubmit["http://127.0.0.1:{P}/cookie", HandlerFunctions -> <|"CookiesReceived" -> '
     '(AppendTo[x01ck, #] &)|>]; TaskWait[x01t]; Length[x01ck[[1, "Cookies"]]]', "2"),
    ('URLSubmit["http://127.0.0.1:{P}/text", HandlerFunctions -> <|"Foo" -> Print|>]', Raises(ValueError)),
    ('URLSubmit["http://127.0.0.1:{P}/text", HandlerFunctionsKeys -> {"Nope"}]', Raises(ValueError)),
    ('URLSubmit[CloudObject["x"]]', Raises(NotImplementedError)),
    # URLDownloadSubmit: the file must hold exactly the 256 bytes served by /bin
    ('x01dl = {}; x01t = URLDownloadSubmit["http://127.0.0.1:{P}/bin", "{T}/x01dl.bin", HandlerFunctions -> '
     '<|"TaskFinished" -> (AppendTo[x01dl, #] &)|>, HandlerFunctionsKeys -> {"StatusCode", "ByteCountDownloaded"}];'
     ' TaskWait[x01t]; {x01dl, FileByteCount["{T}/x01dl.bin"]}',
     '{{<|"StatusCode" -> 200, "ByteCountDownloaded" -> 256|>}, 256}'),
    ('x01t = URLDownloadSubmit["http://127.0.0.1:{P}/bin", "{T}/x01dl2.bin"]; TaskWait[x01t]; '
     'Normal[ReadByteArray["{T}/x01dl2.bin"]] == Range[0, 255]', True),               # Check: byte for byte
    ('x01t = URLDownloadSubmit["http://127.0.0.1:{P}/text", File["{T}/x01dl3.txt"]]; '
     'TaskWait[x01t]["EvaluationResult"]', 'File["{T}/x01dl3.txt"]'),
    ('x01t = URLDownloadSubmit["http://127.0.0.1:{P}/text"]; '
     'FileExistsQ[First[TaskWait[x01t]["EvaluationResult"]]]', True),
    ('URLDownloadSubmit["http://127.0.0.1:{P}/text", "{T}/x01dl4.txt", HandlerFunctions -> <|"BodyChunkReceived" -> '
     'Print|>]', Raises(ValueError)),
    ('URLDownloadSubmit[CloudObject["x"], "{T}/x01dl5.txt"]', Raises(NotImplementedError)),
]

# --------------------------------------------------------------------------------------------------- compile

COMP = [
    # KernelFunction (documentation examples; AiryAi[1.1] = 0.120049 checked against mpmath)
    ('x01cf = FunctionCompile[Function[Typed[arg, "Real64"], Typed[KernelFunction[AiryAi], {"Real64"} -> "Real64"]'
     '[arg]]]; x01cf[1.1]', float(mpmath.airyai(1.1))),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Real64"], TypeHint[KernelFunction[# + 10 &], {"Real64"} -> '
     '"Real64"][arg]]]; x01cf[10.2]', 20.2),
    ('x01sq[x_] := x^2; x01cf = FunctionCompile[Function[Typed[arg, "MachineInteger"], TypeHint[KernelFunction['
     'x01sq], {"MachineInteger"} -> "MachineInteger"][arg]]]; x01cf[10]', "100"),
    ('x01cf = FunctionCompile[Function[Typed[arg, "MachineInteger"], TypeHint[KernelFunction[N], '
     '{"MachineInteger"} -> "MachineInteger"][arg]]]; x01cf[3]', Raises(TypeError)),    # result type checked
    ("KernelFunction[x01k]", "KernelFunction[x01k]"),
    ("KernelFunction[x01k][3]", "KernelFunction[x01k][3]"),
    # KernelEvaluate
    ('x01cf = FunctionCompile[Function[Typed[arg, "Real64"], TypeHint[KernelEvaluate[arg, AiryAi[arg]], "Real64"]]];'
     ' x01cf[1.1]', float(mpmath.airyai(1.1))),
    ('x01sq[x_] := x^2; x01cf = FunctionCompile[Function[Typed[arg, "MachineInteger"], TypeHint[KernelEvaluate[arg, '
     'x01sq[arg]], "MachineInteger"]]]; x01cf[10]', "100"),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Real64"], Module[{tmp}, tmp = 2 arg; TypeHint[KernelEvaluate['
     'tmp, AiryAi[tmp]], "Real64"]]]]; x01cf[0.55]', float(mpmath.airyai(1.1))),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Real64"], TypeHint[KernelEvaluate[arg, ToString[arg]], '
     '"Real64"]]]; x01cf[1.5]', Raises(TypeError)),
    ("KernelEvaluate[1 + 1]", Pred(lambda r: isinstance(r, Expr) and r.head == "KernelEvaluate", "held, unevaluated")),
    # DownValuesFunction
    ('x01square[x_] := x^2; x01fc = FunctionCompile[Function[Typed[arg, "MachineInteger"], Typed[DownValuesFunction['
     'x01square], {"MachineInteger"} -> "MachineInteger"][arg]]]; x01fc[100]', "10000"),
    ('x01fac[0] = x01fac[1] = 1; x01fac[n_] := n x01fac[n - 1]; x01fc = FunctionCompile[FunctionDeclaration[x01fac, '
     'Typed[{"MachineInteger"} -> "MachineInteger"]@DownValuesFunction[x01fac]], Function[Typed[arg, '
     '"MachineInteger"], x01fac[arg]]]; x01fc[10]', str(math.factorial(10))),
    ('x01half[x_] := x/2; x01fc = FunctionCompile[Function[Typed[arg, "MachineInteger"], Typed[DownValuesFunction['
     'x01half], {"MachineInteger"} -> "MachineInteger"][arg]]]; x01fc[3]', Raises(TypeError)),
    ("DownValuesFunction[x01square]", "DownValuesFunction[x01square]"),
    # InertExpression / InertEvaluate
    ('x01cf = FunctionCompile[Function[{}, InertExpression[x01foo]]]; x01cf[]', "x01foo"),
    ('x01cf = FunctionCompile[Function[{Typed[arg, "InertExpression"]}, InertExpression[x01h][arg]]]; x01cf[x01y]',
     "x01h[x01y]"),
    ('x01cf = FunctionCompile[Function[{Typed[arg, "Integer64"]}, InertExpression[x01h][arg]]]; x01cf[10]',
     "x01h[10]"),
    ('x01cf = FunctionCompile[Function[{}, Length[InertExpression[2 + 2]]]]; x01cf[]', "2"),
    ("InertExpression[2 + 2]", Raises(ValueError)),
    ('x01cf = FunctionCompile[Function[{}, Head[InertEvaluate[InertExpression[2 + 2]]]]]; x01cf[]', "Integer"),
    ('x01sq[x_] := x^2; x01cf = FunctionCompile[Function[Typed[arg, "MachineInteger"], Cast[InertEvaluate[Construct['
     'InertExpression[x01sq], Cast[arg, "InertExpression"]]], "MachineInteger"]]]; x01cf[10]', "100"),
    ('x01cf = FunctionCompile[Function[{}, Head[InertExpression[2 + 2]]]]; x01cf[]', "Plus"),
    ("InertEvaluate[x01z]", Raises(ValueError)),
    ('x01cf = FunctionCompile[Function[Typed[arg, "MachineInteger"], InertEvaluate[arg]]]; x01cf[1]',
     Raises(TypeError)),
    # CreateTypeInstance / ToRawPointer / FromRawPointer
    ('x01td = TypeDeclaration["Product", "x01Prod", <|"field1" -> "MachineInteger", "field2" -> "Real64"|>]; '
     'x01cf = FunctionCompile[{x01td}, Function[Typed[f1, "MachineInteger"], With[{prod = CreateTypeInstance['
     '"x01Prod", <|"field1" -> f1, "field2" -> 3.14|>]}, prod["field1"]]]]; x01cf[12]', "12"),
    ('x01td = TypeDeclaration["Product", "x01Prod2", <|"field1" -> "MachineInteger", "field2" -> "Real64"|>]; '
     'x01cf = FunctionCompile[{x01td}, Function[Typed[f1, "MachineInteger"], With[{prod = CreateTypeInstance['
     '"x01Prod2", <|"field1" -> f1, "field2" -> 3.14|>]}, prod["field2"]]]]; x01cf[12]', 3.14),
    ('x01td = TypeDeclaration["Product", "x01Prod3", <|"field1" -> "MachineInteger"|>]; '
     'x01cf = FunctionCompile[{x01td}, Function[Typed[f1, "MachineInteger"], With[{prod = CreateTypeInstance['
     '"x01Prod3", <|"field1" -> f1, "nope" -> 1|>]}, prod["field1"]]]]; x01cf[12]', Raises(ValueError)),
    ('x01td = TypeDeclaration["Product", "x01Prod4", <|"field1" -> "MachineInteger"|>]; '
     'x01cf = FunctionCompile[{x01td}, Function[Typed[f1, "MachineInteger"], CreateTypeInstance["x01Prod4", '
     '<|"field1" -> f1|>]]]; x01cf[12]', Raises(NotImplementedError)),
    ('CreateTypeInstance["x01Prod", <|"field1" -> 1|>]', Raises(ValueError)),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Integer64"], With[{p = ToRawPointer[arg]}, FromRawPointer[p]]]];'
     ' x01cf[4]', "4"),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Real64"], With[{p = ToRawPointer[arg]}, FromRawPointer[p]]]];'
     ' x01cf[2.5]', 2.5),
    ('x01cf = FunctionCompile[Function[{Typed[arg, "Integer64"]}, Module[{arr}, arr = CreateTypeInstance['
     'TypeSpecifier["CArray"]["MachineInteger"], 2]; ToRawPointer[arr, 0, arg]; ToRawPointer[arr, 1, arg]; '
     'FromRawPointer[arr, 0] + FromRawPointer[arr, 0]]]]; x01cf[4]', "8"),
    ('x01cf = FunctionCompile[Function[{Typed[arg, "Integer64"]}, Module[{arr}, arr = CreateTypeInstance['
     'TypeSpecifier["CArray"]["MachineInteger"], 2]; ToRawPointer[arr, 1, arg]; FromRawPointer[arr, 1] + '
     'FromRawPointer[arr, 0]]]]; x01cf[7]', "7"),                                       # zeroed memory
    ('x01cf = FunctionCompile[Function[{Typed[arg, "Integer64"]}, Module[{arr}, arr = CreateTypeInstance['
     'TypeSpecifier["CArray"]["MachineInteger"], 2]; FromRawPointer[arr, 5]]]]; x01cf[4]', Raises(ValueError)),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Integer64"], ToRawPointer[arg]]]; x01cf[4]',
     Raises(NotImplementedError)),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Integer64"], FromRawPointer[arg]]]; x01cf[4]', Raises(TypeError)),
    ("ToRawPointer[4]", Raises(ValueError)),
    ("FromRawPointer[4]", Raises(ValueError)),
    ('x01cf = FunctionCompile[Function[Typed[arg, "Integer64"], With[{p = ToRawPointer[arg]}, '
     'FromRawPointer[p] - arg]]]; x01cf[123456789]', "0"),                            # Check: round trip (integers)
    # CompiledExpressionDeclaration (documentation examples)
    ('x01dec = CompiledExpressionDeclaration[x01foo2, {"Integer64", "Real64"}]; x01cf = FunctionCompile[x01dec, '
     'Function[{}, x01foo2[3, 5.6]]]; x01cf[]', "x01foo2[3, 5.6]"),
    ('x01dec = CompiledExpressionDeclaration[x01foo3, Function[{expr, ty}, Length[expr] === 2]]; x01cf = '
     'FunctionCompile[x01dec, Function[Typed[arg, "x01foo3"], arg]]; {x01cf[x01foo3[1, 2]], Head[x01cf[x01bar[1, 2, '
     '3]]], x01cf[x01bar[1, 2, 3]][[1]]}', '{x01foo3[1, 2], Failure, "ArgumentType"}'),
    ('x01dec = CompiledExpressionDeclaration[x01foo4, 2]; x01cf = FunctionCompile[x01dec, Function[Typed[arg, '
     '"x01foo4"], Length[arg]]]; x01cf[x01foo4[1, 2]]', "2"),
    ('x01dec = CompiledExpressionDeclaration[x01foo5, {"Integer64", "Real64"}]; x01cf = FunctionCompile[x01dec, '
     'Function[{}, x01foo5[3, "a"]]]; x01cf[]', Raises(TypeError)),
    ('CompiledExpressionDeclaration[x01foo6, {"Integer64"}]', "CompiledExpressionDeclaration[x01foo6, {\"Integer64\"}]"),
]

# --------------------------------------------------------------------------------------------------- math


def _qint(D, c0, c1):
    """(X, Y) with c0 + c1 sqrt(D) = (X + Y sqrt(D))/2, both integers when the element is an algebraic integer."""
    X, Y = 2 * sp.Rational(c0), 2 * sp.Rational(c1)
    return X, Y


def _associated(D, a, b):
    """a / b is a unit of the ring of integers of Q(sqrt(D)) (exact arithmetic)."""
    s = sp.sqrt(D)
    q = sp.nsimplify(sp.radsimp((a[0] + a[1] * s) / (b[0] + b[1] * s)))
    q = sp.expand(q)
    c0, c1 = q.as_independent(s)[0], sp.expand(q - q.as_independent(s)[0]) / s
    c0, c1 = sp.Rational(c0), sp.Rational(sp.simplify(c1))
    X, Y = 2 * c0, 2 * c1
    if not (X.is_integer and Y.is_integer):
        return False
    if D % 4 != 1 and not (c0.is_integer and c1.is_integer):
        return False
    if (X - Y) % 2:
        return False
    return abs(c0 ** 2 - D * c1 ** 2) == 1


def _reps_match(D, m, documented):
    def f(r):
        if not (isinstance(r, Expr) and r.head == "List" and len(r.args) == len(documented)):
            return False
        mine = []
        for an in r.args:
            c0, c1 = (sp.sympify(str(c)) for c in an.args[1].args)
            if D == -1:
                mine.append((c0, c1))                                # basis 1, I
            else:
                mine.append((c0, c1))
            if sp.simplify(c0 ** 2 - D * c1 ** 2 - m) != 0:
                return False                                         # norm exactly m (documented sign)
        return all(sum(_associated(D, d, x) for x in mine) == 1 for d in documented)
    return Pred(f, f"one representative of norm {m} per documented class")


MATH = [
    # NumberFieldNormRepresentatives (documentation: classes of -3 - Sqrt[2], -3 + Sqrt[2]; of -5 - 2 Sqrt[5];
    # 1 - 2 I, 1 + 2 I); the representatives may differ from Wolfram's by a unit factor
    ("NumberFieldNormRepresentatives[Sqrt[2], 7]", _reps_match(2, 7, [(-3, -1), (-3, 1)])),
    ("NumberFieldNormRepresentatives[Sqrt[5], 5]", _reps_match(5, 5, [(-5, -2)])),
    ("NumberFieldNormRepresentatives[I, 5]", "{AlgebraicNumber[I, {1, -2}], AlgebraicNumber[I, {1, 2}]}"),
    ("AlgebraicNumberNorm /@ NumberFieldNormRepresentatives[Sqrt[2], 7]", "{7, 7}"),
    ("Length[NumberFieldNormRepresentatives[Sqrt[3], 2]]", "1"),                    # documented: {1 - Sqrt[3]}
    ("NumberFieldNormRepresentatives[Sqrt[3], 2]", _reps_match(3, -2, [(1, -1)])),
    ("NumberFieldNormRepresentatives[Sqrt[3], 7]", "{}"),                           # 3 is a non-residue mod 7
    ("NumberFieldNormRepresentatives[{Sqrt[3], Sqrt[2]}, 7] // Length", "2"),        # Listable
    ("x01g = Expand[Flatten[Outer[Times, {1, -1, I, -I}, RootReduce[NumberFieldNormRepresentatives[I, 5]]]]]; "
     "{Union[Expand[x01g Conjugate[x01g]]], CountDistinct[x01g], "
     "Complement[{-2 - I, -2 + I, -1 - 2 I, -1 + 2 I, 1 - 2 I, 1 + 2 I, 2 - I, 2 + I}, x01g]}",
     "{{5}, 8, {}}"),                                     # Check: the associates are the 8 Gaussian integers of norm 5
    ("NumberFieldNormRepresentatives[x01a, 5]", "NumberFieldNormRepresentatives[x01a, 5]"),
    ("NumberFieldNormRepresentatives[2^(1/3), 2]", Raises(NotImplementedError)),
    ("NumberFieldNormRepresentatives[Sqrt[2], 0]", Raises(ValueError)),
    ("NumberFieldNormRepresentatives[1.5, 5]", Raises(TypeError)),
    ("NumberFieldNormRepresentatives[Sqrt[2], 1/2]", Raises(TypeError)),
    # SpheroidalRadialFactor (order 0)
    ("N[SpheroidalRadialFactor[7, 0, 1/4], 50]",
     Pred(lambda r: mpmath.almosteq(mpmath.mpf(str(r).split("`")[0]),
                                    mpmath.mpf("0.99999869915286302825185488984328477481665817545047"), 1e-48),
          "the 50 documented digits")),
    ("SpheroidalRadialFactor[0, 0, 0.]", 1.0),
    ("SpheroidalRadialFactor[{7, 7}, 0, 0.25]", [0.99999869915286302825, 0.99999869915286302825], 1e-14),
    ("SpheroidalRadialFactor[n, 0, c]", "SpheroidalRadialFactor[n, 0, c]"),
    ("SpheroidalRadialFactor[3, 2, 10.]", Raises(NotImplementedError)),
    ("SpheroidalRadialFactor[1, 0.5, -5]", Raises(NotImplementedError)),
]


# NBodySimulation: references
def _harmonic_x1(t):
    """Masses 1 and 2 at -1/3 and 1/3, at rest, spring k = 1, length 1: r(t) = 1 - cos(w t)/3 with
    w = Sqrt[k/mu], mu = 2/3; centre of mass 1/9; x1 = 1/9 - (2/3) r."""
    w = math.sqrt(1.5)
    return 1 / 9 - (2 / 3) * (1 - math.cos(w * t) / 3)


_FREE_FALL = math.pi / 2 * math.sqrt(math.sqrt(2) ** 3 / (2 * 2))                  # (Pi/2) Sqrt[r0^3/(2 M)]

_THREE = ('NBodySimulation["InverseSquare", {<|"Mass" -> 1, "Position" -> {0, 0}, "Velocity" -> {0, .5}|>, '
          '<|"Mass" -> 1, "Position" -> {1, 1}, "Velocity" -> {0, -.5}|>, <|"Mass" -> 1, "Position" -> {0, 1}, '
          '"Velocity" -> {0, 0}|>}, 4]')
_HARM = ('NBodySimulation["Harmonic", {<|"Mass" -> 1, "Position" -> {-1/3}, "Velocity" -> {0}|>, '
         '<|"Mass" -> 2, "Position" -> {1/3}, "Velocity" -> {0}|>}, 10]')

NBODY = [
    # documentation: data[All, 4] = positions {0.869387, 1.03249}, {-0.694252, -0.384878}, {0.824865, 1.35239}
    # (Wolfram's default NDSolve tolerances; our tighter integration agrees to about 1e-4)
    ("x01d = " + _THREE + '; Flatten[x01d[All, "Position", 4]]',
     [0.869387, 1.03249, -0.694252, -0.384878, 0.824865, 1.35239], 2e-4),
    ("x01d = " + _THREE + '; x01d["SimulationTime"]', 4.0),
    ("x01d = " + _THREE + '; Total[x01d[All, "Momentum", 4]]', [0.0, 0.0], 1e-9),       # momentum conserved
    ("x01d = " + _THREE + '; Abs[x01d["TotalEnergy", 4] - x01d["TotalEnergy", 0]] < 10^-7', True),  # Check: energy
    ("x01d = " + _THREE + '; Keys[x01d[1, 2]]', '{"Mass", "Position", "Velocity", "Momentum"}'),
    # two bodies at rest collide at the free-fall time (documented 1.32088)
    ('x01d = NBodySimulation["InverseSquare", {<|"Mass" -> 1, "Position" -> {0, 0}, "Velocity" -> {0, 0}|>, '
     '<|"Mass" -> 1, "Position" -> {1, 1}, "Velocity" -> {0, 0}|>}, 2]; x01d["SimulationTime"]', _FREE_FALL, 1e-4),
    # harmonic: analytic solution
    ("x01d = " + _HARM + '; x01d[1, "Position", 2.5]', [_harmonic_x1(2.5)], 1e-8),
    ("x01d = " + _HARM + '; x01d[1, "Position", 7.25]', [_harmonic_x1(7.25)], 1e-8),
    ('x01d = NBodySimulation[<|"PairwisePotential" -> Function[{a, b}, (Norm[a["Position"] - b["Position"]] - 1)^2/2]'
     '|>, {<|"Mass" -> 1, "Position" -> {-1/3}, "Velocity" -> {0}|>, <|"Mass" -> 2, "Position" -> {1/3}, '
     '"Velocity" -> {0}|>}, 10]; x01d[1, "Position", 2.5]', [_harmonic_x1(2.5)], 1e-8),   # potential function
    ('x01d = NBodySimulation[{"Harmonic", "SpringConstant" -> 2, "EquilibriumLength" -> 1}, {<|"Mass" -> 1, '
     '"Position" -> {0}, "Velocity" -> {0}|>, <|"Mass" -> 1, "Position" -> {1/2}, "Velocity" -> {0}|>}, 3]; '
     'x01d[2, "Position", 1.]', [0.25 + 0.5 * (1 - 0.5 * math.cos(2 * 1.0))], 1e-8),     # w = Sqrt[k/mu] = 2
    ('x01d = NBodySimulation[{"Harmonic", <|"SpringConstant" -> 2|>}, {<|"Mass" -> 1, "Position" -> {0}, '
     '"Velocity" -> {0}|>, <|"Mass" -> 1, "Position" -> {1/2}, "Velocity" -> {0}|>}, 3]; x01d[2, "Position", 1.]',
     [0.25 + 0.5 * (1 - 0.5 * math.cos(2 * 1.0))], 1e-8),
    ('x01d = NBodySimulation[{"Harmonic", "SpringConstant" -> .5, "EquilibriumLength" -> 1.5}, <|"bill" -> <|"Mass" '
     '-> .25, "Position" -> {0, 1}, "Velocity" -> {0, 0}|>, "sam" -> <|"Mass" -> 1, "Position" -> {-1, 0}, '
     '"Velocity" -> {0, 0}|>, "george" -> <|"Mass" -> 1, "Position" -> {1, 0}, "Velocity" -> {0, 0}|>|>, 25]; '
     '{x01d["Bodies"], First[x01d["bill", "Position", 3]], x01d["sam", "Position", 3][[1]] + '
     'x01d["george", "Position", 3][[1]]}', Pred(lambda r: str(r.args[0]) == '{"bill", "sam", "george"}'
                                                 and abs(r.args[1]) < 1e-9 and abs(r.args[2]) < 1e-9,
                                                 "named bodies, mirror symmetry")),
    # external potential: a single body in V = x^2/2 oscillates as cos t
    ('x01d = NBodySimulation[<|"ExternalPotential" -> Function[s, s["Position"].s["Position"]/2]|>, {<|"Mass" -> 1, '
     '"Position" -> {1}, "Velocity" -> {0}|>}, 4]; x01d[1, "Position", 2.]', [math.cos(2.0)], 1e-8),
    # pairwise force (documentation elastic-collision example): momentum is conserved
    ('x01force[r_, e_] := With[{delta = #1["Position"] - #2["Position"]}, Function[delta e Ramp[2 r - Norm[delta]]^'
     '(3/2)/(2 Norm[delta])]]; x01d = NBodySimulation[<|"PairwiseForce" -> x01force[1., 2.]|>, {<|"Mass" -> 1, '
     '"Position" -> {0, 0, 0}, "Velocity" -> {0, 0, 0}|>, <|"Mass" -> 1, "Position" -> {.1, .1, 1}, "Velocity" -> '
     '{0, 0, -1}|>}, 2]; Total[x01d[All, "Velocity", 2]]', [0.0, 0.0, -1.0], 1e-9),
    ("x01d = " + _HARM + '; x01d[1, "Position", x01t]', Pred(lambda r: isinstance(r, Expr) and isinstance(r.head, Expr)
                                                              and r.head.head == "NBodySimulationData",
                                                              "symbolic time stays unevaluated")),
    ("NBodySimulation[x01law, x01states, 4]", "NBodySimulation[x01law, x01states, 4]"),
    ("x01d = " + _HARM + '; x01d[1, "Position", 11]', Raises(ValueError)),
    ("x01d = " + _HARM + '; x01d["Equations"]', Raises(NotImplementedError)),
    ('NBodySimulation["Newtonian", {<|"Mass" -> 1, "Position" -> {0}, "Velocity" -> {0}|>, <|"Mass" -> 1, '
     '"Position" -> {1}, "Velocity" -> {0}|>}, 1]',
     Raises(NotImplementedError)),
    ('NBodySimulation[<|"PairwisePotential" -> "InverseSquare", "Region" -> Disk[]|>, {<|"Mass" -> 1, "Position" -> '
     '{0, 0}, "Velocity" -> {0, 0}|>}, 1]', Raises(NotImplementedError)),
    ('NBodySimulation["InverseSquare", {<|"Mass" -> Quantity[1, "Kilograms"], "Position" -> {0}, "Velocity" -> {0}|>}'
     ', 1]', Raises(NotImplementedError)),
    ('NBodySimulation["Bogus", {<|"Mass" -> 1, "Position" -> {0}, "Velocity" -> {0}|>, <|"Mass" -> 1, '
     '"Position" -> {1}, "Velocity" -> {0}|>}, 1]', Raises(ValueError)),
    ('NBodySimulation["InverseSquare", {<|"Mass" -> 1, "Position" -> {0}|>}, 1]', Raises(ValueError)),
    ('NBodySimulation["InverseSquare", {5}, 1]', Raises(TypeError)),
]

CASES = CORE + PKG + FILES + BIO + URL + COMP + MATH + NBODY


@pytest.fixture(scope="module")
def x01_env():
    from mikoshilang.library import wl_evaluation as WE
    from _w30_servers import free_port
    root = tempfile.mkdtemp(prefix="x01spf")
    for rel, text in SPF_FILES.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            f.write(text)
    d = tempfile.mkdtemp(prefix="x01fs")
    os.makedirs(os.path.join(d, "empty"))
    os.makedirs(os.path.join(d, "sub", "deep"))
    for rel, text in (("a.txt", "the quick dog"), ("sub/b.wl", "x01b = 1"), ("sub/deep/c.txt", "lorem ipsum dog")):
        with open(os.path.join(d, rel), "w") as f:
            f.write(text)
    t = tempfile.mkdtemp(prefix="x01dl")
    WE.PACKAGE_PATH.append(root)
    http = HTTPServer()
    fwd = lambda p: p.replace("\\", "/")  # noqa: E731
    subs = {"{TREE}": _tree(fwd(d)), "{S}": fwd(root), "{D}": fwd(d), "{T}": fwd(t), "{P}": str(http.port),
            "{F}": str(free_port())}
    yield subs
    http.close()
    WE.PACKAGE_PATH.remove(root)
    for p in (root, d, t):
        shutil.rmtree(p, ignore_errors=True)


def _sub(x, subs):
    if not isinstance(x, str):
        return x
    for k, v in subs.items():
        x = x.replace(k, v)
    return x


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_x01(case, x01_env):
    check(tuple(_sub(c, x01_env) for c in case))
