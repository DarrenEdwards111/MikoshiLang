"""Differential tests for Wolfram batch W01 (evaluation control, attributes, definitions, contexts & packages).

Expected values come from the Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>.html:
the examples and the rules stated in their Details sections).  Every case uses its own symbol names (w1...) because
the harness shares one session across cases; cases that change $Context, $ContextPath or $Packages do so inside
Block.  The main loop (In, Out, %, $Line, $Pre, $Post, ...) is exercised with separate Session objects at the end.
"""
import os
import tempfile

import pytest

from _harness import Pred, Raises, check, ids

from mikoshilang.expr import Expr, Symbol
from mikoshilang.formatting import input_form
from mikoshilang.runtime import Session


def text(want):
    return Pred(lambda r: isinstance(r, str) and r == want, repr(want))


def form(want):
    return Pred(lambda r: input_form(r) == want, want)


_PKG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "w01_packages")
os.environ.setdefault("MIKOSHILANG_LOCALSYMBOL_BASE", tempfile.mkdtemp(prefix="w01ls"))
CLEAN = '{$ContextPath = {"System`", "Global`"}, $Packages = {"System`", "Global`"}, $Context = "Global`", ' \
         '$ContextAliases = <||>}'

from mikoshilang.library import wl_evaluation as _W  # noqa: E402
if _PKG_DIR not in _W.PACKAGE_PATH:
    _W.PACKAGE_PATH.append(_PKG_DIR)

CONTEXTS = [
    ("$Context", text("Global`")),
    ("$ContextPath", '{"System`", "Global`"}'),
    ("Block[{$Context = \"w1ctx`\"}, $Context]", text("w1ctx`")),
    ("$Context = 5", Raises()),
    ("$Context = \"no context\"", Raises()),
    ("Block[{$ContextPath = {\"System`\"}}, $ContextPath]", '{"System`"}'),
    ("Block[{$ContextPath = {\"System`\", \"w1cp`\", \"Global`\"}}, ToExpression[\"w1cp`w1pv = 3\"]; "
     "ToExpression[\"w1pv\"]]", 3),
    ("$ContextPath = {1, 2}", Raises()),
    ("$ContextAliases", "<||>"),
    ("Block[{$ContextAliases = <|\"w1a`\" -> \"w1long`\"|>}, ToExpression[\"w1a`w1q = 9\"]; w1long`w1q]", 9),
    ("Block[{$ContextAliases = <|\"w1b`\" -> \"w1longb`\"|>}, ToExpression[\"w1b`w1zz\"]; Context[\"w1b`w1zz\"]]",
     text("w1longb`")),
    ("$ContextAliases = 3", Raises()),
    ("$Packages", '{"System`", "Global`"}'),
    ("Block[" + CLEAN + ", BeginPackage[\"W1pk`\"]; EndPackage[]; MemberQ[$Packages, \"W1pk`\"]]", True),
    ("$Packages = {3}", Raises()),
    ("Context[Sin]", text("System`")),
    ("w1cx = 1; Context[w1cx]", text("Global`")),
    ("Context[]", text("Global`")),
    ("Context[\"Plus\"]", text("System`")),
    ("Context[w1ctxq`w1sym]", text("w1ctxq`")),
    ("Context[\"w1never`seen`thing\"]", Raises()),
    ("Contexts[\"System*\"]", '{"System`"}'),
    ("Block[" + CLEAN + ", ToExpression[\"w1cts`w1inner`w1x\"]; Contexts[\"w1cts*\"]]",
     '{"w1cts`", "w1cts`w1inner`"}'),
    ("MemberQ[Contexts[], \"Global`\"]", True),
    ("Block[" + CLEAN + ", {Begin[\"w1bg`\"], $Context, End[], $Context}]",
     '{"w1bg`", "w1bg`", "w1bg`", "Global`"}'),
    ("Block[" + CLEAN + ", Begin[\"w1bh`\"]; Begin[\"`sub`\"]; {$Context, End[], End[], $Context}]",
     '{"w1bh`sub`", "w1bh`sub`", "w1bh`", "Global`"}'),
    ("Block[" + CLEAN + ", Begin[\"w1bi`\"]; ToExpression[\"w1bsym\"]; End[]; Context[\"w1bi`w1bsym\"]]",
     text("w1bi`")),
    ("Begin[3]", Raises()),
    ("Block[" + CLEAN + ", End[]]", Raises()),
    ("Block[" + CLEAN + ", BeginPackage[\"W1bp`\"]; {$Context, $ContextPath, EndPackage[], $Context, "
     "First[$ContextPath]}]", '{"W1bp`", {"W1bp`", "System`"}, Null, "Global`", "W1bp`"}'),
    ("Block[" + CLEAN + ", BeginPackage[\"W1bq`\"]; ToExpression[\"w1bqf\"]; Begin[\"`Private`\"]; "
     "ToExpression[\"w1bqf[x_] := w1bqg[x]; w1bqg[x_] := x + 1\"]; End[]; EndPackage[]; ToExpression[\"w1bqf[1]\"]]",
     2),
    ("Block[" + CLEAN + ", BeginPackage[\"W1br`\"]; EndPackage[]; EndPackage[]]", Raises()),
    ("BeginPackage[\"bad context\"]", Raises()),
    ("Block[" + CLEAN + ", BeginPackage[\"W1bs`\"]; EndPackage[]; MemberQ[$ContextPath, \"W1bs`\"]]", True),
    ("Block[" + CLEAN + ", EndPackage[]]", Raises()),
    ("Block[" + CLEAN + ", Get[\"W1Pkg.m\"]; {Context[\"w1pub\"], W1Pkg`w1pub[5]}]",
     '{"W1Pkg`", 10}'),
    ("Get[\"w1script.m\"]", 42),
    ("Get[\"w1script.m\"]; w1scriptval", 41),
    ("Get[\"w1_no_such_file.m\"]", Raises()),
    ("Block[" + CLEAN + ", Get[\"W1Dir`W1Sub`\"]; W1Dir`W1Sub`w1subf[1]]", 101),
    ("Block[" + CLEAN + ", Needs[\"W1Pkg`\"]; {MemberQ[$ContextPath, \"W1Pkg`\"], W1Pkg`w1pub[2]}]",
     '{True, 4}'),
    ("Block[" + CLEAN + ", Needs[\"W1Pkg`\" -> \"w1al`\"]; {MemberQ[$ContextPath, \"W1Pkg`\"], "
     "$ContextAliases[\"w1al`\"]}]", '{False, "W1Pkg`"}'),
    ("Block[" + CLEAN + ", Needs[\"W1Nonexistent`\"]]", Raises()),
    ("Needs[42]", Raises()),
    ("Block[" + CLEAN + ", DeclarePackage[\"W1Stub`\", {\"w1stubf\"}]; {MemberQ[Attributes[\"W1Stub`w1stubf\"], "
     "Stub], First[$ContextPath]}]", '{True, "W1Stub`"}'),
    ("Block[" + CLEAN + ", DeclarePackage[\"W1Stub`\", {\"w1stubf\"}]; ToExpression[\"w1stubf[2]\"]]", 8),
    ("DeclarePackage[\"W1x`\", 5]", Raises()),
    ("Names[\"Plu*\"]", Pred(lambda r: "Plus" in [a for a in r.args], "contains Plus")),
    ("w1namesa = 1; w1namesb = 2; Names[\"w1names*\"]", '{"w1namesa", "w1namesb"}'),
    ("Names[\"w1nm`*\"]", '{}'),
    ("ToExpression[\"w1nn`w1x\"]; Names[\"w1nn`*\"]", '{"w1nn`w1x"}'),
    ("w1nmq = 1; Names[\"w1n@q\"]", '{"w1nmq"}'),
    ("Names[3]", Raises()),
    ("NameQ[\"Sin\"]", True),
    ("NameQ[\"w1definitelyNotASymbolXYZ\"]", False),
    ("w1nameqa = 1; NameQ[\"w1nameq*\"]", True),
    ("w1nmc = 1; Names[\"w1nmc\", ResolveContextAliases -> False]", '{"w1nmc"}'),
    ("Contexts[\"System`\", ResolveContextAliases -> False]", '{"System`"}'),
    ("NameQ[\"Sin\", ResolveContextAliases -> True]", True),
    ("Block[{$NewSymbol = (w1newlog = {##}) &}, ToExpression[\"w1brandnew\"]; w1newlog]",
     '{"w1brandnew", "Global`"}'),
    ("$NewSymbol", "$NewSymbol"),
    ("ValueQ[$NewSymbol]", False),
]

ATTRIBUTES = [
    ("Attributes[Plus]", "{Flat, Listable, NumericFunction, OneIdentity, Orderless, Protected}"),
    ("SetAttributes[w1at, {Listable, HoldFirst}]; Attributes[w1at]", "{HoldFirst, Listable}"),
    ("Attributes[{w1at0, Sin}]", "{{}, {Listable, NumericFunction, Protected}}"),
    ("Attributes[\"Sin\"]", "{Listable, NumericFunction, Protected}"),
    ("Attributes[w1at1] = {Orderless}; Attributes[w1at1]", "{Orderless}"),
    ("Attributes[w1at2] = {Bogus}", Raises()),
    ("SetAttributes[w1hac, HoldAllComplete]; w1hac[1 + 1, Sequence[a, b], Unevaluated[3]]",
     form("w1hac[1 + 1, Sequence[a, b], Unevaluated[3]]")),
    ("SetAttributes[w1hac2, HoldAllComplete]; Length[w1hac2[Sequence[1, 2]]]", 1),
    ("SetAttributes[w1hac3, HoldAll]; Length[w1hac3[Sequence[1, 2]]]", 2),
    ("SetAttributes[w1hac4, HoldAllComplete]; w1hac4[1 + 2] /. w1hac4[x_] :> Hold[x]", "Hold[1 + 2]"),
    ("SetAttributes[w1nha, NHoldAll]; N[w1nha[1/2, 1/4]]", "w1nha[1/2, 1/4]"),
    ("SetAttributes[w1nhf, NHoldFirst]; N[w1nhf[1/2, 1/4]]", "w1nhf[1/2, 0.25]"),
    ("SetAttributes[w1nhr, NHoldRest]; N[w1nhr[1/2, 1/4]]", "w1nhr[0.5, 1/4]"),
    ("SetAttributes[w1nhf2, NHoldFirst]; N[{w1nhf2[1, 1/2], 1/2}]", "{w1nhf2[1, 0.5], 0.5}"),
    ("SetAttributes[w1nha2, NHoldAll]; Attributes[w1nha2]", "{NHoldAll}"),
    ("SetAttributes[w1nhr2, NHoldRest]; N[w1nhr2[Pi, Pi]] // InputForm // ToString",
     text("w1nhr2[3.141592653589793, Pi]")),
    ("Block[" + CLEAN + ", DeclarePackage[\"W1Stub`\", {\"w1stubf\"}]; MemberQ[Attributes[\"W1Stub`w1stubf\"], "
     "Stub]]", True),
    ("SetAttributes[w1stb, Stub]; Attributes[w1stb]", "{Stub}"),
    ("MemberQ[Attributes[Sin], Stub]", False),
    ("Module[{w1tmp}, Attributes[w1tmp]]", "{Temporary}"),
    ("Attributes[Unique[w1tq, {Temporary}]]", "{Temporary}"),
    ("Module[{w1tmp2}, MemberQ[Attributes[w1tmp2], Temporary]]", True),
    ("SetAttributes[w1svh, SubValuesHoldAll]; w1svh[1 + 1][2 + 2]", form("w1svh[2][2 + 2]")),
    ("SetAttributes[w1svh2, SubValuesHoldAll]; w1svh2[a_][b_] := Hold[b]; w1svh2[1][1 + 2]", "Hold[1 + 2]"),
    ("SetAttributes[w1svh3, {SubValuesHoldAll, HoldAll}]; w1svh3[1 + 1][2 + 2]", form("w1svh3[1 + 1][2 + 2]")),
    ("SetAttributes[w1svh4, SubValuesHoldAll]; w1svh4[1][Evaluate[1 + 1]]", form("w1svh4[1][2]")),
    ("SetAttributes[w1nt, NonThreadable]; w1nt + {1, 2}", form("w1nt + {1, 2}")),
    ("SetAttributes[w1nt2, NonThreadable]; {1, 2} + {3, 4} + w1nt2", form("w1nt2 + {4, 6}")),
    ("SetAttributes[w1nt3, NonThreadable]; Length[w1nt3[1] {1, 2, 3}]", 2),
    ("w1nt4 + {1, 2}", "{1 + w1nt4, 2 + w1nt4}"),
    ("SetAttributes[w1lk, {Locked, Protected}]; ClearAttributes[w1lk, Protected]", Raises()),
    ("SetAttributes[w1lk2, Locked]; SetAttributes[w1lk2, Listable]", Raises()),
    ("SetAttributes[w1lk3, Locked]; Attributes[w1lk3]", "{Locked}"),
    ("SetAttributes[w1lk4, Locked]; Unprotect[w1lk4]", Raises()),
    ("SetAttributes[w1lk5, Locked]; w1lk5 = 3; w1lk5", 3),
    ("SetAttributes[w1cn, Constant]; Dt[w1cn x, x]", "w1cn"),
    ("SetAttributes[w1cn2, Constant]; Dt[w1cn2]", 0),
    ("SetAttributes[w1cn3, Constant]; Dt[w1cn3 x^2 + y]", "2 w1cn3 x Dt[x] + Dt[y]"),
    ("SetAttributes[w1rp, ReadProtected]; w1rp[x_] := x^2; w1rp[3]", 9),
    ("SetAttributes[w1rp2, ReadProtected]; w1rp2[x_] := x^2; ToString[Definition[w1rp2]]",
     text("Attributes[w1rp2] = {ReadProtected}")),
    ("SetAttributes[w1rp3, ReadProtected]; w1rp3 = 7; DownValues[w1rp3]", "{}"),
    ("Protect[w1pr]; w1pr = 1", Raises()),
    ("Protect[w1pr2]", '{"w1pr2"}'),
    ("Protect[w1pr3]; Protect[w1pr3]", '{}'),
    ("Protect[w1pr4]; Attributes[w1pr4]", "{Protected}"),
    ("Protect[w1pr5]; Unprotect[w1pr5]", '{"w1pr5"}'),
    ("Protect[w1pr6]; Unprotect[w1pr6]; w1pr6 = 3; w1pr6", 3),
    ("Unprotect[w1pr7]", '{}'),
    ("Sin[w1x] = 3", Raises()),
]

DEFINITIONS = [
    ("w1us[w1ug[x_]] ^:= x^2; w1us[w1ug[3]]", 9),
    ("w1us2[w1ug2[x_]] ^= 5; {UpValues[w1ug2], DownValues[w1us2]}", "{{HoldPattern[w1us2[w1ug2[x_]]] :> 5}, {}}"),
    ("w1ua[w1sq] ^= 4; w1ua[w1sq]", 4),
    ("w1ub[x_w1obj] ^:= 10; w1ub[w1obj[1]]", 10),
    ("w1uc[w1ud, w1ue] ^= 1; {UpValues[w1ud], UpValues[w1ue]}",
     "{{HoldPattern[w1uc[w1ud, w1ue]] :> 1}, {HoldPattern[w1uc[w1ud, w1ue]] :> 1}}"),
    ("w1uf ^= 3", Raises()),
    ("w1uh[1 + 1] ^= 2", Raises()),
    ("w1uset[w1uarg] ^= 1 + 1", 2),
    ("w1usd[w1uargd] ^:= 1 + 1", "Null"),
    ("w1tg /: w1tf[w1tg[x_]] := x + 1; w1tf[w1tg[5]]", 6),
    ("w1tg2 /: w1tf2[w1tg2[x_]] = 7; {UpValues[w1tg2], DownValues[w1tf2]}",
     "{{HoldPattern[w1tf2[w1tg2[x_]]] :> 7}, {}}"),
    ("w1tg3 /: w1tg3[x_] := x^3; w1tg3[2]", 8),
    ("w1tg4 /: w1tf4[w1tz] = 1", Raises()),
    ("w1tg5 /: w1tf5[w1tg5] = 2 + 3", 5),
    ("w1tg6 /: w1tf6[w1tg6] := 2 + 3", "Null"),
    ("w1tg7 /: w1tf7[w1tg7[x_]] := x; w1tg7 /: w1tf7[w1tg7[x_]] =.; w1tf7[w1tg7[1]]", "w1tf7[w1tg7[1]]"),
    ("w1tg8 /: w1tf8[w1tg8] = 1; w1tg8 /: w1tf8[w1tg8] =.; UpValues[w1tg8]", "{}"),
    ("w1tg9 /: w1tf9[w1tg9] =.", Raises()),
    ("w1tga /: w1tga[1] = 3; w1tga /: w1tga[1] =.; w1tga[1]", "w1tga[1]"),
    ("w1ov = 5; OwnValues[w1ov]", "{HoldPattern[w1ov] :> 5}"),
    ("OwnValues[w1ovnone]", "{}"),
    ("OwnValues[w1ov2] = {HoldPattern[w1ov2] :> 3}; w1ov2", 3),
    ("OwnValues[w1ov3] = {HoldPattern[w1other] :> 3}", Raises()),
    ("w1dv[1] = 2; w1dv[x_] := x; DownValues[w1dv]", "{HoldPattern[w1dv[1]] :> 2, HoldPattern[w1dv[x_]] :> x}"),
    ("DownValues[w1dv2] = {HoldPattern[w1dv2[x_]] :> x + 1}; w1dv2[1]", 2),
    ("w1dv3[x_] := x; DownValues[w1dv3] = {}; w1dv3[1]", "w1dv3[1]"),
    ("DownValues[w1dv4] = {HoldPattern[w1other[x_]] :> 1}", Raises()),
    ("w1sv[a_][b_] := a + b; SubValues[w1sv]", "{HoldPattern[w1sv[a_][b_]] :> a + b}"),
    ("w1sv2[a_][b_] := a b; {DownValues[w1sv2], w1sv2[2][3]}", "{{}, 6}"),
    ("SubValues[w1sv3] = {HoldPattern[w1sv3[a_][b_]] :> a - b}; w1sv3[5][1]", 4),
    ("SubValues[w1sv4] = {HoldPattern[w1sv4[a_]] :> a}", Raises()),
    ("w1uv[w1uvt] ^= 1; UpValues[w1uvt]", "{HoldPattern[w1uv[w1uvt]] :> 1}"),
    ("UpValues[w1uvt2] = {HoldPattern[w1uvf[w1uvt2[x_]]] :> x^2}; w1uvf[w1uvt2[4]]", 16),
    ("UpValues[w1uvnone]", "{}"),
    ("UpValues[w1uvt3] = {HoldPattern[w1uvt3[x_]] :> x}", Raises()),
    ("Default[w1df] = 7; w1df[x_.] := x; w1df[]", 7),
    ("Default[w1df2, 2] = 9; w1df2[a_, b_.] := {a, b}; w1df2[1]", "{1, 9}"),
    ("Default[w1df3] = 0; Default[w1df3]", 0),
    ("Default[Plus]", 0),
    ("Default[Times]", 1),
    ("Default[Power, 2]", 1),
    ("Default[w1df4]", "Default[w1df4]"),
    ("Default[3]", Raises()),
    ("Default[w1dfv] = 5; DefaultValues[w1dfv]", "{HoldPattern[Default[w1dfv]] :> 5}"),
    ("Default[w1dfv2, 1] = 2; Default[w1dfv2, 1, 3] = 4; DefaultValues[w1dfv2]",
     "{HoldPattern[Default[w1dfv2, 1]] :> 2, HoldPattern[Default[w1dfv2, 1, 3]] :> 4}"),
    ("DefaultValues[w1dfv3] = {HoldPattern[Default[w1dfv3]] :> 11}; Default[w1dfv3]", 11),
    ("DefaultValues[w1dfnone]", "{}"),
    ("w1vq = 1; ValueQ[w1vq]", True),
    ("ValueQ[w1vqnone]", False),
    ("w1vq2[1] = 2; {ValueQ[w1vq2[1]], ValueQ[w1vq2[2]]}", "{True, False}"),
    ("ValueQ[1 + 1]", True),
    ("w1vq3[x_] := x; ValueQ[w1vq3[1], Method -> \"SymbolDefinitionsPresent\"]", True),
    ("ValueQ[w1vqnone2[1], Method -> \"SymbolDefinitionsPresent\"]", False),
    ("ValueQ[w1x, Method -> \"Bogus\"]", Raises()),
    ("w1rm = 3; Remove[w1rm]; Names[\"w1rm\"]", "{}"),
    ("w1rm2[x_] := x; Remove[w1rm2]; w1rm2[1]", "w1rm2[1]"),
    ("Remove[Sin]", Raises()),
    ("w1rma = 1; w1rmb = 2; Remove[\"w1rm*\"]; Names[\"w1rm*\"]", "{}"),
    ("Unique[] // Context", text("Global`")),
    ("StringMatchQ[SymbolName[Unique[w1uq]], \"w1uq$\" ~~ DigitCharacter ..]", True),
    ("Unique[\"w1uz\"]", "w1uz1"),
    ("Unique[\"w1uy\"]; Unique[\"w1uy\"]", "w1uy2"),
    ("Length[Unique[{w1ua1, w1ua2, w1ua3}]]", 3),
    ("Unique[{w1ub1, w1ub1}] // Apply[UnsameQ]", True),
    ("Unique[3]", Raises()),
    ("Unique[\"bad name\"]", Raises()),
    ("$ModuleNumber = 5000; {Unique[w1mn], $ModuleNumber}", "{w1mn$5000, 5001}"),
    ("$ModuleNumber = 6000; Module[{w1mm}, w1mm]", "w1mm$6000"),
    ("$ModuleNumber > 0", True),
    ("$ModuleNumber = 0", Raises()),
    ("w1def[x_] := x^2; w1def[1] = 5; ToString[Definition[w1def]]", text("w1def[1] = 5\n \nw1def[x_] := x^2")),
    ("w1def2 = 3; ToString[Definition[w1def2]]", text("w1def2 = 3")),
    ("SetAttributes[w1def3, Listable]; Options[w1def3] = {w1o -> 1}; ToString[Definition[w1def3]]",
     text("Attributes[w1def3] = {Listable}\n \nOptions[w1def3] = {w1o -> 1}")),
    ("ToString[Definition[w1defnone]]", text("Null")),
    ("w1def4 := RandomReal[]; ToString[Definition[w1def4]]", text("w1def4 := RandomReal[]")),
    ("w1def5[w1defu] ^= 2; ToString[Definition[w1defu]]", text("w1def5[w1defu] ^= 2")),
    ("w1fd1[x_] := w1fd2[x] + 1; w1fd2[x_] := 2 x; ToString[FullDefinition[w1fd1]]",
     text("w1fd1[x_] := w1fd2[x] + 1\n \nw1fd2[x_] := 2*x")),
    ("w1fd3 = w1fd4; w1fd4 = 1; ToString[FullDefinition[w1fd3]]", text("w1fd3 = w1fd4\n \nw1fd4 = 1")),
    ("w1fd5 := w1fd6; w1fd6 = 2; ToString[FullDefinition[w1fd5]]", text("w1fd5 := w1fd6\n \nw1fd6 = 2")),
    ("w1fd7 := w1fdc`w1fd8; w1fdc`w1fd8 = 2; ToString[FullDefinition[w1fd7, ExcludedContexts -> {\"w1fdc\"}]]",
     text("w1fd7 := w1fdc`w1fd8")),
    ("w1fd9 := w1fdd`w1fda; w1fdd`w1fda = 2; ToString[FullDefinition[w1fd9, ExcludedContexts -> {}]]",
     text("w1fd9 := w1fdd`w1fda\n \nw1fdd`w1fda = 2")),
    ("w1fdb := w1fde`w1fdf + w1fdg; w1fde`w1fdf = 2; w1fdg = 3; "
     "ToString[FullDefinition[w1fdb, IncludedContexts -> {\"w1fde\"}]]",
     text("w1fdb := w1fde`w1fdf + w1fdg\n \nw1fde`w1fdf = 2")),
    ("w1fdh := 1; ToString[FullDefinition[w1fdh, IncludedContexts -> {\"Global\"}]]", text("w1fdh := 1")),
    ("FullDefinition[w1x, Bogus -> 1]", Raises()),
    ("w1in = 3; Information[w1in, \"OwnValues\"]", "{HoldPattern[w1in] :> 3}"),
    ("Information[w1in2, \"FullName\"]", text("Global`w1in2")),
    ("w1in3[x_] := x; Information[w1in3][\"DownValues\"]", "{HoldPattern[w1in3[x_]] :> x}"),
    ("Information[Sin, \"Attributes\"]", "{Listable, NumericFunction, Protected}"),
    ("Information[\"w1notexisting1234\"]", Raises()),
    ("Head[Information[Plus]]", "InformationData"),
    ("Information[w1in4, {\"ObjectType\", \"OwnValues\"}]", '{"Symbol", None}'),
    ("Information[w1in5, LongForm -> False][\"OwnValues\"]", "Missing[\"UnknownProperty\", \"OwnValues\"]"),
    ("InformationData[<|\"a\" -> 1|>, False][\"a\"]", 1),
    ("InformationData[<|\"a\" -> 1|>, False][\"b\"]", "Missing[\"UnknownProperty\", \"b\"]"),
    ("Head[InformationData[<|\"a\" -> 1|>, False]]", "InformationData"),
]

OPTIONS = [
    ("Options[w1op] = {w1oa -> 1, w1ob -> 2}; Options[w1op]", "{w1oa -> 1, w1ob -> 2}"),
    ("Options[w1op2] = {w1oa -> 1, w1ob -> 2}; Options[w1op2, w1ob]", "{w1ob -> 2}"),
    ("Options[w1op3]", "{}"),
    ("Options[w1f[x, w1oa -> 1, w1ob :> 2]]", "{w1oa -> 1, w1ob :> 2}"),
    ("Options[Plot]", Raises(NotImplementedError)),
    ("Options[Sin]", Raises(NotImplementedError)),
    ("Options[w1op4] = {w1oa -> 1}; SetOptions[w1op4, w1oa -> 5]", "{w1oa -> 5}"),
    ("Options[w1op5] = {w1oa -> 1, w1ob -> 2}; SetOptions[w1op5, w1ob -> 7]; Options[w1op5]",
     "{w1oa -> 1, w1ob -> 7}"),
    ("Options[w1op6] = {w1oa -> 1}; SetOptions[w1op6, w1oz -> 5]", Raises()),
    ("SetOptions[Plot, PlotRange -> All]", Raises()),
    ("Options[w1ov1] = {w1oa -> 1, w1ob -> 2}; w1ov1[x_, OptionsPattern[]] := {x, OptionValue[w1oa], "
     "OptionValue[w1ob]}; w1ov1[0, w1ob -> 5]", "{0, 1, 5}"),
    ("Options[w1ovb2] = {w1oa -> 1}; w1ovb2[OptionsPattern[]] := OptionValue[w1oa]; w1ovb2[]", 1),
    ("Options[w1ovb3] = {w1oa -> 1}; w1ovb3[OptionsPattern[]] := OptionValue[w1oa]; w1ovb3[{w1oa -> 3}]", 3),
    ("Options[w1ov4] = {w1oa -> 1}; OptionValue[w1ov4, w1oa]", 1),
    ("Options[w1ov5] = {w1oa -> 1}; OptionValue[w1ov5, {w1oa -> 9}, w1oa]", 9),
    ("Options[w1ov6] = {w1oa -> 1, w1ob -> 2}; OptionValue[w1ov6, {}, {w1oa, w1ob}]", "{1, 2}"),
    ("Options[w1ov7] = {w1oa -> 1}; OptionValue[w1ov7, w1oz]", Raises()),
    ("OptionValue[w1oa]", Raises()),
    ("Options[w1ov8a] = {w1oa -> 1}; Options[w1ov8b] = {w1ob -> 2}; "
     "w1ov8[OptionsPattern[{w1ov8a, w1ov8b}]] := {OptionValue[w1oa], OptionValue[w1ob]}; w1ov8[w1ob -> 3]",
     "{1, 3}"),
    ("Options[w1opp] = {w1oa -> 1}; MatchQ[w1g[1, w1oa -> 2], w1g[_, OptionsPattern[]]]", True),
    ("MatchQ[w1g[1, 2], w1g[_, OptionsPattern[]]]", False),
    ("MatchQ[w1g[1], w1g[_, OptionsPattern[]]]", True),
    ("w1g[1, w1oa -> 2, {w1ob -> 3}] /. w1g[x_, o : OptionsPattern[]] :> {o}", "{w1oa -> 2, {w1ob -> 3}}"),
    ("FilterRules[{w1a -> 1, w1b -> 2, w1c -> 3}, {w1a, w1c}]", "{w1a -> 1, w1c -> 3}"),
    ("FilterRules[{w1a -> 1, w1b -> 2, w1c -> 3}, Except[{w1a, w1c}]]", "{w1b -> 2}"),
    ("Options[w1fr] = {w1b -> 0}; FilterRules[{w1a -> 1, w1b -> 2}, Options[w1fr]]", "{w1b -> 2}"),
    ("FilterRules[{w1a -> 1, w1bb -> 2}, w1a]", "{w1a -> 1}"),
    ("FilterRules[{\"w1s\" -> 1, w1b -> 2}, \"w1s\"]", "{\"w1s\" -> 1}"),
    ("FilterRules[{w1a -> 1, 3}, w1a]", Raises()),
    ("Options[w1ca] = {w1oa -> 1}; CheckArguments[w1ca[1, w1oa -> 0], 1]", True),
    ("Options[w1ca2] = {w1oa -> 1}; CheckArguments[w1ca2[1, 2], 1]", False),
    ("Options[w1ca3] = {w1oa -> 1}; CheckArguments[w1ca3[1, w1oz -> 0], 1]", False),
    ("CheckArguments[w1ca4[1, 2], {1, 3}]", True),
    ("CheckArguments[w1ca5[1, w1oq -> 2], 1, <|\"ExtraOptions\" -> {w1oq -> 0}|>]", True),
    ("CheckArguments[w1ca6[1], \"x\"]", Raises()),
    ("Options[w1ao] = {w1oa -> 1}; ArgumentsOptions[w1ao[1, w1oa -> 0], 1]", "{{1}, {w1oa -> 0}}"),
    ("Options[w1ao2] = {w1oa -> 1}; ArgumentsOptions[w1ao2[1, 2], {1, 2}]", "{{1, 2}, {}}"),
    ("Head[ArgumentsOptions[w1ao3[1, 2], 1]]", "Failure"),
    ("ArgumentsOptions[w1ao4[1, 2], 2, <|\"Head\" -> Hold|>]", "{Hold[1, 2], Hold[]}"),
    ("SyntaxInformation[w1si] = {\"ArgumentsPattern\" -> {_, _}}; SyntaxInformation[w1si]",
     "{\"ArgumentsPattern\" -> {_, _}}"),
    ("SyntaxInformation[w1si2]", "{}"),
    ("SyntaxInformation[w1si3] = {\"Bogus\" -> 1}", Raises()),
    ("SyntaxInformation[Plus]", Raises(NotImplementedError)),
    ("w1ar[x_, opts : OptionsPattern[]] := {x, Length[{opts}]}; w1ar[1, Automatic -> 2]", "{1, 1}"),
    ("Automatic", "Automatic"),
    ("{Full, Manual, Inherited}", "{Full, Manual, Inherited}"),
    ("Full = 3", Raises()),
    ("Manual === Manual", True),
    ("Inherited =!= Automatic", True),
    ("Method -> \"w1m\"", "Method -> \"w1m\""),
    ("Head[Method]", "Symbol"),
    ("Method = 1", Raises()),
    ("Head[General]", "Symbol"),
    ("General = 2", Raises()),
    ("Context[General]", text("System`")),
    ("ActivateResult -> True", "ActivateResult -> True"),
    ("Context[ActivateResult]", text("System`")),
    ("ActivateResult = 1", Raises()),
    ("AllowedHeads -> All", "AllowedHeads -> All"),
    ("Context[AllowedHeads]", text("System`")),
    ("AllowedHeads = 1", Raises()),
    ("Application[w1f, w1g]", "Application[w1f, w1g]"),
    ("Application[Application[w1f, w1g], w1h] // Head", "Application"),
    ("Context[Application]", text("System`")),
    ("Context[Automatic]", text("System`")),
    ("Context[Full]", text("System`")),
    ("Context[Manual]", text("System`")),
    ("Context[Inherited]", text("System`")),
    ("Context[Method]", text("System`")),
]

EVALUATION = [
    ("Length[Unevaluated[1 + 2]]", 2),
    ("Unevaluated[1 + 1]", form("Unevaluated[1 + 1]")),
    ("w1uev[x_] := Hold[x]; w1uev[Unevaluated[1 + 1]]", "Hold[1 + 1]"),
    ("w1uev2[Unevaluated[1 + 1]]", form("w1uev2[Unevaluated[1 + 1]]")),
    ("Head[Unevaluated[1 + 1]]", "Plus"),
    ("Defer[1 + 1]", form("1 + 1")),
    ("ToString[Defer[2 + 3]]", text("2 + 3")),
    ("Defer[1 + 1] /. Defer[x_] :> x", 2),
    ("Head[Defer[1 + 2]]", "Defer"),
    ("HoldCompleteForm[1 + 1]", form("1 + 1")),
    ("Length[HoldCompleteForm[Sequence[1, 2]]]", 1),
    ("HoldCompleteForm[1 + 1] /. HoldCompleteForm[x_] :> x", 2),
    ("Inactive[Plus][1, 2]", form("Inactive[Plus][1, 2]")),
    ("Activate[Inactive[Plus][1, 2]]", 3),
    ("Inactive[3]", 3),
    ("Inactive[Length][{w1a, w1b, w1c}] // Activate", 3),
    ("Inactivate[2 + 2 + 3^2]", form("Inactive[Plus][Inactive[Plus][2, 2], Inactive[Power][3, 2]]")),
    ("Activate[Inactivate[2 + 2 + 3^2]]", 13),
    ("Activate[Inactivate[2 + 2 + 3^2], Plus]", form("4 + Inactive[Power][3, 2]")),
    ("Activate[Inactivate[2 + 2 + 3^2], Power]", form("Inactive[Plus][Inactive[Plus][2, 2], 9]")),
    ("Inactivate[w1f[w1g[w1h[x]], y, w1g[z]], w1g]", form("w1f[Inactive[w1g][w1h[x]], y, Inactive[w1g][z]]")),
    ("Inactivate[w1f[w1g[w1h[x]], y, w1g[z]], w1g | w1h]",
     form("w1f[Inactive[w1g][Inactive[w1h][x]], y, Inactive[w1g][z]]")),
    ("Inactivate[{1 + 1, w1a -> 2 + 2}]", form("{Inactive[Plus][1, 1], w1a -> Inactive[Plus][2, 2]}")),
    ("Inactivate[Sin[Pi/2], Sin] // Activate", 1),
    ("Inactivate[w1q[1][2], Heads -> False]", form("w1q[1][2]")),
    ("Inactivate[w1q[1][2]]", form("Inactive[w1q][1][2]")),
    ("Inactivate[1 + 1, Heads -> 3]", Raises()),
    ("Activate[Inactive[Cos][Pi] + Inactive[Sin][0]]", -1),
    ("Activate[Inactive[Cos][Pi] + Inactive[Sin][0], Except[Sin]]", form("-1 + Inactive[Sin][0]")),
    ("Activate[5]", 5),
    ("Cases[{Sin[w1m], Inactive[Sin][w1p]}, IgnoringInactive[Sin[w1a_]]]", form("{Sin[w1m], Inactive[Sin][w1p]}")),
    ("Cases[{w1sn[w1m], Inactive[w1sn][w1p]}, w1sn[_]]", form("{w1sn[w1m]}")),
    ("MatchQ[Inactive[Sin][w1m], IgnoringInactive[Sin[w1a_]]]", True),
    ("{Sin[w1m], Inactive[Sin][w1p]} /. IgnoringInactive[Sin[w1a_]] :> w1t", "{w1t, w1t}"),
    ("w1un = 1; Until[w1un > 3, w1un++]; w1un", 4),
    ("w1un2 = 10; Until[True, w1un2++]; w1un2", 11),
    ("w1un3 = 0; Until[w1un3 > 100, w1un3++; If[w1un3 == 5, Break[]]]; w1un3", 5),
    ("{w1ga, w1gb} = {27, 6}; Until[w1gb == 0, {w1ga, w1gb} = {w1gb, Mod[w1ga, w1gb]}]; w1ga", 3),
    ("Until[w1un4 = 5; True]; w1un4", 5),
    ("Until[1, 2, 3]", Raises()),
    ("w1gt[a_] := Module[{x = 1., xp}, Label[w1begin]; If[Abs[xp - x] < 10^-8, Goto[w1end]]; xp = x; "
     "x = (x + a/x)/2; Goto[w1begin]; Label[w1end]; x]; w1gt[2]", 1.4142135623730951),
    ("w1gl = 0; (Label[w1top]; w1gl++; If[w1gl < 5, Goto[w1top]]; w1gl)", 5),
    ("(Goto[w1skip]; w1gs = 1; Label[w1skip]; w1gs2 = 2); {ValueQ[w1gs], w1gs2}", "{False, 2}"),
    ("Goto[w1nowhere]", Raises()),
    ("Label[w1lab]", "Null"),
    ("Head[Label[w1lab2]]", "Symbol"),
    ("Update[]", "Null"),
    ("Update[w1upd]", "Null"),
    ("Update[1, 2]", Raises()),
    ("w1apt = 2; w1apt //= w1f", "w1f[2]"),
    ("w1apt2 = {3, 1, 2}; w1apt2 //= Sort; w1apt2", "{1, 2, 3}"),
    ("w1apt3 = 4; ApplyTo[w1apt3, Sqrt]; w1apt3", 2),
    ("w1aptnone //= f", Raises()),
    ("$RecursionLimit", 1024),
    ("Block[{$RecursionLimit = 30}, w1rec[n_] := 1 + w1rec[n + 1]; w1rec[0]]", Raises(RecursionError)),
    ("Block[{$RecursionLimit = 50}, $RecursionLimit]", 50),
    ("$RecursionLimit = 5", Raises()),
    ("$IterationLimit", 4096),
    ("Block[{$IterationLimit = 40}, w1it = w1it2; w1it2 = w1it3; w1it3 = w1it4; w1it4 = 1; w1it]", 1),
    ("Block[{$IterationLimit = 25}, w1ite[0] = 0; w1ite[n_] := w1ite[n - 1]; w1ite[100]]", Raises(RecursionError)),
    ("Block[{$IterationLimit = 200}, w1itf[0] = 0; w1itf[n_] := w1itf[n - 1]; w1itf[100]]", 0),
    ("$IterationLimit = 3", Raises()),
    ("Block[{$IterationLimit = Infinity}, $IterationLimit]", "Infinity"),
]

FUNCTIONAL = [
    ("OperatorApplied[w1f][w1x][w1y]", "w1f[w1y, w1x]"),
    ("OperatorApplied[w1f, 3][w1x][w1y][w1z]", "w1f[w1x, w1y, w1z]"),
    ("OperatorApplied[Integrate, {3, 1, 2}][w1x][w1y][w1x Sin[w1y]]", "Integrate[w1x Sin[w1y], w1x, w1y]"),
    ("OperatorApplied[w1f, 2][w1x, w1y]", "w1f[w1x, w1y]"),
    ("OperatorApplied[w1f, 3 -> {3, 1}][w1a][w1b][w1c]", "w1f[w1c, w1a]"),
    ("OperatorApplied[w1f, 3][w1x][w1y]", form("OperatorApplied[w1f, 3][w1x][w1y]")),
    ("OperatorApplied[w1f, -1]", Raises()),
    ("CurryApplied[w1f, 2][w1x][w1y]", "w1f[w1x, w1y]"),
    ("CurryApplied[2][w1f][w1x][w1y]", "w1f[w1x, w1y]"),
    ("CurryApplied[w1f, {2, 1}][w1x][w1y]", "w1f[w1y, w1x]"),
    ("CurryApplied[w1f, 3][w1x, w1y][w1z]", "w1f[w1x, w1y, w1z]"),
    ("CurryApplied[w1f, {1, 2, w1o -> 1}][w1x][w1y]", "w1f[w1x, w1y, w1o -> 1]"),
    ("CurryApplied[w1f]", Raises()),
    ("Threaded[{1, 2}] + {{10, 20}, {30, 40}}", "{{11, 22}, {31, 42}}"),
    ("{{1, 2}, {3, 4}} + Threaded[{10, 20}, 1]", "{{11, 12}, {23, 24}}"),
    ("1 + Threaded[{1, 2}]", "{2, 3}"),
    ("{{1, 2}, {3, 4}} Threaded[{w1a, w1b}]", "{{w1a, 2 w1b}, {3 w1a, 4 w1b}}"),
    ("Threaded[{1, 2}]", form("Threaded[{1, 2}]")),
    ("{1, 2} + Threaded[{{1, 2}, {3, 4}}]", Raises()),
    ("Distribute[w1f[w1a + w1b, w1c + w1d]]", "w1f[w1a, w1c] + w1f[w1a, w1d] + w1f[w1b, w1c] + w1f[w1b, w1d]"),
    ("Distribute[w1f[w1g[w1a, w1b], w1g[w1c]], w1g]", "w1g[w1f[w1a, w1c], w1f[w1b, w1c]]"),
    ("Distribute[(w1a + w1b) . (w1x + w1y)]", "w1a . w1x + w1a . w1y + w1b . w1x + w1b . w1y"),
    ("Distribute[w1f[w1g[w1a, w1b], w1c], w1g, w1f, w1gp, w1fp]", "w1gp[w1fp[w1a, w1c], w1fp[w1b, w1c]]"),
    ("Distribute[w1h[w1a + w1b], Plus, w1f]", "w1h[w1a + w1b]"),
    ("Distribute[And[Or[w1a, w1b], Or[w1c, w1d]], Or, And]",
     "(w1a && w1c) || (w1a && w1d) || (w1b && w1c) || (w1b && w1d)"),
    ("Distribute[w1f[1, 2], 3]", Raises()),
    ("InverseFunction[Sin]", "ArcSin"),
    ("InverseFunction[Exp]", "Log"),
    ("InverseFunction[ArcTan]", "Tan"),
    ("InverseFunction[2 # + 3 &][w1y]", "(w1y - 3)/2"),
    ("InverseFunction[w1f][w1x]", form("InverseFunction[w1f][w1x]")),
    ("InverseFunction[InverseFunction[w1f]]", "w1f"),
    ("InverseFunction[#1^3 + #2 &, 2, 2][w1a, w1b]", "w1b - w1a^3"),
    ("InverseFunction[w1f, 3, 2]", Raises()),
    ("Indexed[{w1a, w1b}, 1]", "w1a"),
    ("Indexed[{{w1a, w1b, w1c}, {w1d, w1e, w1f}}, {1, 2}]", "w1b"),
    ("Indexed[w1x, 1]", form("Indexed[w1x, 1]")),
    ("Indexed[w1x, {1}]", form("Indexed[w1x, 1]")),
    ("Indexed[{w1a}, 5]", Raises()),
    ("Indexed[w1x, {1, 2}]", form("Indexed[w1x, {1, 2}]")),
    ("Prefix[w1f[w1x]]", form("w1f@w1x")),
    ("Prefix[w1f[w1x], \"#\"]", form("#w1x")),
    ("Prefix[w1f[w1x]] /. Prefix[e_] :> e", "w1f[w1x]"),
    ("Postfix[w1f[w1x]]", form("w1x//w1f")),
    ("Postfix[w1f[w1x], \"!\"]", form("w1x!")),
    ("Head[Postfix[w1f[w1x]]]", "Postfix"),
]

FAC = "w1fac[0] = 1; w1fac[n_] := n w1fac[n - 1]; "

STACK_TRACE = [
    ("w1sf[w1sg[1, w1st = Stack[]; 2]]; w1st", Pred(lambda r: [input_form(a) for a in r.args][-4:] ==
                                                     ["w1sf", "w1sg", "CompoundExpression", "Set"], "... f, g, ;, =")),
    ("Head[Stack[]]", "List"),
    ("w1sf2[w1st2 = Stack[_w1sf2]]; w1st2", Pred(lambda r: len(r.args) == 1 and r.args[0].head == "HoldCompleteForm"
                                                 and r.args[0].args[0].head == "w1sf2", "{HoldCompleteForm[w1sf2[...]]}")),
    ("Stack[_w1nothingonstack]", "{}"),
    ("w1sb[StackBegin[w1sbv = Stack[]]]; w1sbv", "{Set}"),
    ("StackBegin[Stack[]]", "{}"),
    ("StackBegin[1 + 1]", 2),
    ("w1si[StackBegin[w1siv = w1sj[StackInhibit[Stack[]]]]]; w1siv", form("w1sj[{Set, w1sj}]")),
    ("StackInhibit[1 + 2]", 3),
    ("Length[StackBegin[StackInhibit[Stack[]]]]", 0),
    ("w1sc[x_] := w1sd[x^2]; w1sd[x_] := (w1scv = Stack[_]; x + 1); StackBegin[StackComplete[w1sc[2]]]; "
     "MemberQ[w1scv, HoldCompleteForm[w1sc[2]]]", True),
    ("w1se[x_] := w1sh[x^2]; w1sh[x_] := (w1sev = Stack[_]; x + 1); StackBegin[w1se[2]]; "
     "MemberQ[w1sev, HoldCompleteForm[w1se[2]]]", False),
    ("StackComplete[1 + 1]", 2),
    ("Trace[" + "w1fa1[0] = 1; w1fa1[n_] := n w1fa1[n - 1]; 0" + "]", Pred(lambda r: r.head == "List", "list")),
    (FAC + "Trace[w1fac[2], w1fac[_]]", form("{w1fac[2], {w1fac[1], {w1fac[0]}}}")),
    (FAC + "Trace[w1fac[2], TraceDepth -> 1]", form("{w1fac[2], 2*w1fac[2 - 1], 2*1, 2}")),
    (FAC + "Length[Trace[w1fac[3], TraceDepth -> 2]]", Pred(lambda r: r > 2, "> 2")),
    (FAC + "Trace[w1fac[1], TraceDepth -> 0]", Raises()),
    (FAC + "Trace[w1fac[3], w1fac[2], TraceForward -> True]", form("{{w1fac[2], 2*w1fac[2 - 1], 2*1, 2}}")),
    (FAC + "Trace[w1fac[1], w1fac[1], TraceForward -> All] // Flatten // Length", Pred(lambda r: r >= 5, ">= 5")),
    (FAC + "Trace[w1fac[1], w1fac[1], TraceForward -> 3]", Raises()),
    (FAC + "Trace[w1fac[3], 2*1, TraceBackward -> True]", form("{{w1fac[2], 2*w1fac[2 - 1], 2*1}}")),
    (FAC + "Trace[w1fac[1], 1*1, TraceBackward -> All] // Flatten // Length", Pred(lambda r: r >= 4, ">= 4")),
    (FAC + "Trace[w1fac[1], 1*1, TraceBackward -> False]", form("{1*1}")),
    (FAC + "Trace[w1fac[2], w1fac[0], TraceAbove -> True]", form("{w1fac[2], {w1fac[1], {w1fac[0]}, 1}, 2}")),
    (FAC + "Trace[w1fac[1], w1fac[0], TraceAbove -> All] // Flatten // Length", Pred(lambda r: r >= 5, ">= 5")),
    (FAC + "Trace[w1fac[1], w1fac[0], TraceAbove -> False]", form("{{w1fac[0]}}")),
    (FAC + "Trace[w1fac[2], TraceOn -> w1fac[1]]", Pred(lambda r: len(r.args) == 1 and input_form(r.args[0].args[0]) == "w1fac[1]"
                                                 and input_form(r.args[0].args[-1]) == "1", "{{w1fac[1], ..., 1}}")),
    (FAC + "Trace[1 + w1fac[1], TraceOn -> w1fac[_]] // Flatten // Length", Pred(lambda r: r >= 5, ">= 5")),
    ("Trace[1 + 2, TraceOn -> w1nomatch]", "{}"),
    (FAC + "Trace[w1fac[2], TraceOff -> w1fac[1]]",
     form("{w1fac[2], 2*w1fac[2 - 1], {{2 - 1, 1}, w1fac[1], 1}, 2*1, 2}")),
    (FAC + "Trace[w1fac[1], TraceOff -> w1nomatch] === Trace[w1fac[1]]", True),
    ("Trace[1 + 2 3, TraceOff -> _Times]", form("{{2*3, 6}, 1 + 6, 7}")),
    ("Trace[1 + 1, TraceOriginal -> True]", Raises(NotImplementedError)),
    ("Trace[1 + 1, TraceOriginal -> False]", form("{1 + 1, 2}")),
    ("Trace[1 + 1, Bogus -> 1]", Raises()),
    (FAC + "Trace[w1fac[1], w1fac]", form("{w1fac[1], {w1fac[0]}}")),
    ("TracePrint[1 + 2]", 3),
    (FAC + "TracePrint[w1fac[2], w1fac[_]]", 2),
    ("TracePrint[1 + 1, TraceBackward -> True]", Raises()),
    ("w1tsc = {}; TraceScan[AppendTo[w1tsc, #] &, 1 + 2 3]; w1tsc", form("{2*3, 6, 1 + 6, 7}")),
    ("TraceScan[w1nothing &, 2 + 3]", 5),
    (FAC + "w1tsn = 0; TraceScan[w1tsn++ &, w1fac[5], w1fac[_Integer]]; w1tsn", 6),
    ("w1tsp = {}; TraceScan[Null &, 1 + 2 3, _Plus, (AppendTo[w1tsp, {##}]) &]; Length[w1tsp] > 0", True),
]

MAIN_LOOP_CASES = [
    ("Out[1]", Pred(lambda r: True, "Out[n] (no main loop in the harness: unevaluated or a value)")),
    ("Head[Out[-100000]]", "Out"),
    ("Out[w1symbolic]", form("Out[w1symbolic]")),
    ("Head[In[-100000]]", "In"),
    ("In[w1symbolic]", form("In[w1symbolic]")),
    ("Head[InString[-100000]]", "InString"),
    ("InString[w1sym]", form("InString[w1sym]")),
    ("InString[1, 2]", Raises()),
    ("IntegerQ[$Line]", True),
    ("$Line = -1", Raises()),
    ("Block[{$Line = 10}, $Line]", 10),
    ("$HistoryLength", "Infinity"),
    ("$HistoryLength = -2", Raises()),
    ("Block[{$HistoryLength = 3}, $HistoryLength]", 3),
    ("ValueQ[$Pre]", False),
    ("Block[{$Pre = w1f}, $Pre]", "w1f"),
    ("Context[$Pre]", text("System`")),
    ("ValueQ[$Post]", False),
    ("Block[{$Post = w1f}, $Post]", "w1f"),
    ("Context[$Post]", text("System`")),
    ("ValueQ[$PrePrint]", False),
    ("Block[{$PrePrint = w1f}, $PrePrint]", "w1f"),
    ("Context[$PrePrint]", text("System`")),
    ("ValueQ[$PreRead]", False),
    ("Block[{$PreRead = w1f}, $PreRead]", "w1f"),
    ("Context[$PreRead]", text("System`")),
    ("ValueQ[$Epilog]", False),
    ("Block[{$Epilog = w1f}, $Epilog]", "w1f"),
    ("Context[$Epilog]", text("System`")),
]

MISC = [
    ("w1oc = 0; Once[w1oc++]; Once[w1oc++]; w1oc", 1),
    ("Once[w1onc = 5; 2 + 3]", 5),
    ("w1od = 0; {Once[w1od += 1], Once[w1od += 1], Once[w1od += 10]}", "{1, 1, 11}"),
    ("Once[1, \"Local\"]", Raises(NotImplementedError)),
    ("w1oe = 0; Once[w1oe++]; Once[w1oe++] =.; Once[w1oe++]; w1oe", 2),
    ("LocalSymbol[\"w1ls\"] = 10!; LocalSymbol[\"w1ls\"]", 3628800),
    ("LocalSymbol[\"w1ls-undefined-xyz\"]", form("LocalSymbol[\"w1ls-undefined-xyz\"]")),
    ("LocalSymbol[\"w1ls2\"] = {1, \"a\", x^2}; LocalSymbol[\"w1ls2\"]", "{1, \"a\", x^2}"),
    ("LocalSymbol[\"w1ls3\"] = 3; LocalSymbol[\"w1ls3\"] =.; LocalSymbol[\"w1ls3\"]",
     form("LocalSymbol[\"w1ls3\"]")),
    ("ValueQ[LocalSymbol[\"w1ls-undefined-abc\"]]", False),
    ("LocalSymbol[5]", Raises()),
]

EXTRA = [
    ("SetAttributes[w1nha3, NHoldAll]; N[w1nha3[Pi]] // ToString", text("w1nha3[Pi]")),
    ("SetAttributes[w1nhf3, NHoldFirst]; Attributes[w1nhf3]", "{NHoldFirst}"),
    ("SetAttributes[w1nhr3, NHoldRest]; Attributes[w1nhr3]", "{NHoldRest}"),
    ("Module[{w1tm3}, SetAttributes[w1tm3, Listable]; Attributes[w1tm3]]", "{Listable, Temporary}"),
    ("UpSet[w1upf[w1upg], 5]; w1upf[w1upg]", 5),
    ("UpSet[w1upf2[w1upg2[x_]], x]", "x"),
    ("UpSet[w1upf3[3], 1]", Raises()),
    ("UpSet[w1upf4[w1upg4], 1 + 1]", 2),
    ("UpSetDelayed[w1usf[w1usg[x_]], x + 1]; w1usf[w1usg[1]]", 2),
    ("UpSetDelayed[w1usf2[w1usg2], 1]", "Null"),
    ("UpSetDelayed[w1usf3[3], 1]", Raises()),
    ("TagSet[w1tsg, w1tsf[w1tsg], 3]; w1tsf[w1tsg]", 3),
    ("TagSet[w1tsq, w1tsf2[w1tsz], 1]", Raises()),
    ("TagSet[3, w1tsf3[w1tsg3], 1]", Raises()),
    ("TagSetDelayed[w1tdg, w1tdf[w1tdg[x_]], x - 1]; w1tdf[w1tdg[5]]", 4),
    ("TagSetDelayed[w1tdg2, w1tdf2[w1tdg2], 1]", "Null"),
    ("TagSetDelayed[w1tdq, w1tdf3[w1tdz], 1]", Raises()),
    ("TagSet[w1tug, w1tuf[w1tug], 1]; TagUnset[w1tug, w1tuf[w1tug]]; w1tuf[w1tug]", "w1tuf[w1tug]"),
    ("TagUnset[w1tug2, w1tuf2[w1tug2]]", Raises()),
    ("TagUnset[w1tuq, w1tuf3[w1tuz]]", Raises()),
    ("w1fdx := w1fdy; w1fdy = 1; ToString[FullDefinition[w1fdx, ExcludedContexts -> {\"Global\"}]]",
     text("w1fdx := w1fdy")),
    ("w1fdz := w1fdw; w1fdw = 1; ToString[FullDefinition[w1fdz, IncludedContexts -> {}]]", text("w1fdz := w1fdw")),
    ("w1apt5 = {1, 2}; ApplyTo[w1apt5, Reverse]", "{2, 1}"),
    ("ApplyTo[w1aptnone2, w1f]", Raises()),
    ("Dt[w1a w1b, w1b, Constants -> {w1a}]", "w1a"),
    ("D[w1nc, w1x, NonConstants -> {w1nc}]", form("D[w1nc, w1x, NonConstants -> {w1nc}]")),
    ("Head[StackComplete[Stack[]]]", "List"),
    ("In[]", form("In[]")),
    ("MemberQ[Attributes[Unique[w1tq2, {Temporary}]], Temporary]", True),
    ("Dt[w1ca w1cb, Constants -> {w1ca}]", "w1ca Dt[w1cb]"),
    ("InverseFunctions -> True", "InverseFunctions -> True"),
    ("Context[InverseFunctions]", text("System`")),
    ("InverseFunctions = 1", Raises()),
    ("Heads -> True", "Heads -> True"),
    ("Context[Heads]", text("System`")),
    ("Heads = 1", Raises()),
    ("Context[Constants]", text("System`")),
    ("Constants = 1", Raises()),
    ("Context[NonConstants]", text("System`")),
    ("NonConstants = 1", Raises()),
    ("D[w1nc2 w1x2, w1x2, NonConstants -> {w1nc2}]", form("w1nc2 + D[w1nc2, w1x2, NonConstants -> {w1nc2}]*w1x2")),
]

CASES = (CONTEXTS + ATTRIBUTES + DEFINITIONS + OPTIONS + EVALUATION + FUNCTIONAL + STACK_TRACE + MAIN_LOOP_CASES + MISC
         + EXTRA)


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w01(case):
    check(case)


# ─────────────────────────── the main loop (Session.evaluate_input) ───────────────────────────

def run(session, *lines):
    out = []
    for line in lines:
        out.extend(session.evaluate_input(line))
    return out


def shown(v):
    return input_form(v)


def test_main_loop_in_out_percent():
    s = Session()
    r = run(s, "1 + 1", "% + 10", "%% * 2", "Out[1]", "%3")
    assert [n for n, _ in r] == [1, 2, 3, 4, 5]
    assert [v for _, v in r] == [2, 12, 4, 2, 4]


def test_main_loop_in_reevaluates():
    s = Session()
    run(s, "w1x = 1", "w1x + 1", "w1x = 5")
    (_, v), = run(s, "In[2]")
    assert v == 6
    (_, v), = run(s, "InString[2]")
    assert v == "w1x + 1"
    (_, v), = run(s, "In[-1]")       # the previous input, InString[2]
    assert v == "w1x + 1"


def test_main_loop_line_numbers():
    s = Session()
    r = run(s, "a", "b", "$Line")
    assert r[-1] == (3, 3)
    run(s, "$Line = 10")
    (n, v), = run(s, "$Line")
    assert (n, v) == (10, 10)


def test_main_loop_statement_mode():
    s = Session()
    r = s.evaluate_input("1 + 1\n2 + 2\n%", whole=False)
    assert [v for _, v in r] == [2, 4, 4]
    assert [n for n, _ in r] == [1, 2, 3]


def test_main_loop_pre_post():
    s = Session()
    run(s, "$Pre = Hold")
    (_, v), = run(s, "1 + 2")
    assert shown(v) == "Hold[1 + 2]"
    s = Session()
    run(s, "$Post = w1post")
    (_, v), = run(s, "1 + 2")
    assert shown(v) == "w1post[3]"
    (_, v), = run(s, "Out[2]")
    assert shown(v) == "w1post[w1post[3]]"


def test_main_loop_preprint_display_only():
    s = Session()
    run(s, "$PrePrint = w1pp")
    (n, v), = run(s, "7")
    assert v == 7 and shown(s.output_form(v)) == "w1pp[7]"
    (_, v), = run(s, "%")
    assert v == 7


def test_main_loop_preread():
    s = Session()
    run(s, "$PreRead = StringReplace[#, \"w1a\" -> \"w1b\"] &")
    (_, v), = run(s, "w1a + 1")
    assert shown(v) == "1 + w1b"


def test_main_loop_history_length():
    s = Session()
    run(s, "$HistoryLength = 2", "10", "20", "30")
    (_, v), = run(s, "Out[2]")
    assert shown(v) == "Out[2]"
    (_, v), = run(s, "Out[4]")
    assert v == 30


def test_main_loop_epilog():
    s = Session()
    run(s, "$Epilog := (w1done = True)")
    s.end()
    (_, v), = run(s, "w1done")
    assert v is True


def test_main_loop_error_still_counts_line():
    s = Session()
    with pytest.raises(Exception):
        s.evaluate_input("Goto[w1nowhere]")
    (n, _), = run(s, "1")
    assert n == 2


def test_main_loop_stack_begins_fresh():
    s = Session()
    (_, v), = run(s, "Stack[]")
    assert shown(v) == "{}"


def test_tracprint_output(capsys):
    from mikoshilang.parser import parse
    from mikoshilang.evaluate import evaluate
    assert evaluate(parse("TracePrint[1 + 2 3]")) == 7
    out = capsys.readouterr().out.splitlines()
    assert [o.strip() for o in out] == ["2*3", "6", "1 + 6", "7"]
    assert out[0].startswith("   ") and out[2].startswith(" ") and not out[2].startswith("  ")


def test_sessions_are_independent():
    a, b = Session(), Session()
    run(a, "1", "2")
    (n, _), = run(b, "3")
    assert n == 1


def test_skipped_symbols_are_documented():
    from mikoshilang.library.wl_evaluation import SKIPPED
    from mikoshilang import library
    for name, reason in SKIPPED.items():
        assert reason
        assert library.STATUS.get(name, (None, None))[1] != "added", name
