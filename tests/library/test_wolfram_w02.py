"""Differential tests for coverage batch W02 (messages, failures, exceptions, aborts, timing and testing).

Expected values come from the Wolfram Language reference pages (reference.wolfram.com/language/ref/<Name>.html:
the examples and the rules stated in their Details sections).  Timing and memory results are machine dependent, so
they are checked with predicates (types, signs, orders of magnitude) instead of values.  Printed output (messages,
Print, Echo) is checked with pytest's capsys fixture in the functions at the end of the file.

Session state (message texts, switches, registered exception types) persists between cases, so every case uses
its own symbols.
"""
import pytest

from _harness import Pred, Raises, check, ev, ids
from mikoshilang.expr import Expr, Symbol
from mikoshilang.library.wl_errors import AbortSignal


def is_num(x, lo=0.0, hi=float("inf")):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and lo <= x <= hi


def timing_pair(lo=0.0, hi=60.0, value=None):
    def f(x):
        return isinstance(x, Expr) and x.head == "List" and len(x.args) == 2 and is_num(x.args[0], lo, hi) and \
            (value is None or x.args[1] == value)
    return Pred(f, f"{{seconds in [{lo}, {hi}], {value}}}")


def num(lo=0.0, hi=float("inf")):
    return Pred(lambda x: is_num(x, lo, hi), f"a number in [{lo}, {hi}]")


def head(h):
    return Pred(lambda x: isinstance(x, Expr) and x.head == h, f"a {h}[...] expression")


CASES = [
    # ── Message / MessageName ──
    ('Block[{$MessageList = {}}, w02m1::t = "bad `1`"; Message[w02m1::t, 3]; $MessageList]',
     "{HoldForm[w02m1::t]}"),
    ('Block[{$MessageList = {}}, Message[w02m2::argx, w02m2, 2]; Length[$MessageList]]', "1"),
    ('Message[w02m3::nodef]', "Null"),
    ('w02m4::t = "text"; w02m4::t', '"text"'),
    ('w02m5::usage = "w02m5[x] is a test."; MessageName[w02m5, "usage"]', '"w02m5[x] is a test."'),
    ('MessageName[w02m6, "argx"]', 'MessageName[w02m6, "argx"]'),
    ('General::argx', '"`1` called with `2` arguments; 1 argument is expected."'),
    ('Head[Hold[w02m7::t][[1]]]', "MessageName"),
    ('Quiet[Message[Sin, 1]]', "Message[Sin, 1]"),
    # ── MessageList / $MessageList ──
    ('Block[{$MessageList = {}}, $MessageList]', "{}"),
    ('Block[{$MessageList = {}}, w02ml::t = "m"; Quiet[Message[w02ml::t]]; $MessageList]', "{}"),
    ('Block[{$MessageList = {}}, w02ml2::t = "m"; Message[w02ml2::t]; Message[w02ml2::t]; Length[$MessageList]]',
     "2"),
    # ── Off / On ──
    ('w02o1::t = "m"; Off[w02o1::t]; w02r = Check[Message[w02o1::t]; 1, 2]; On[w02o1::t]; w02r', "1"),
    ('w02o2::t = "m"; Off[w02o2::t]; On[w02o2::t]; Check[Message[w02o2::t]; 1, 2]', "2"),
    ('w02o3::t = "m"; Off[General::t]; w02r = Check[Message[w02o3::t]; 1, 2]; On[General::t]; w02r', "1"),
    ('Off[w02o4::a, w02o4::b]; Messages[w02o4]', "{}"),
    ('Quiet[Off[3]]', "Off[3]"),
    ('On[w02o5::t, w02o5::u]', "Null"),
    # ── Quiet ──
    ('w02q1::t = "m"; Quiet[Message[w02q1::t]; 5]', "5"),
    ('w02q2::t = "m"; Check[Quiet[Message[w02q2::t]; 5], err]', "5"),
    ('w02q3::t = "m"; Check[Quiet[Message[w02q3::t]; 5, {w02q3::u}], err]', "err"),
    ('w02q4::t = "m"; Check[Quiet[Message[w02q4::t]; 5, All, {w02q4::t}], err]', "err"),
    ('w02q5::t = "m"; Check[Quiet[Quiet[Message[w02q5::t]; 5, None], {w02q5::t}], err]', "5"),
    ('w02q7::t = "m"; Quiet[Quiet[Check[Message[w02q7::t]; 1, err], None, All], {w02q7::t}]', "err"),
    ('w02q6::t = "m"; Check[Quiet[1 + Quiet[Message[w02q6::t]; 1, {w02q6::u}], {w02q6::t}], err]', "2"),
    ('Quiet[1 + 1, 3]', Raises()),
    # ── Check ──
    ('Check[1^0, err]', "1"),
    ('w02c1::t = "m"; Check[Message[w02c1::t]; 1, err]', "err"),
    ('w02c2::t = "m"; Check[Message[w02c2::t]; 1, err, w02c2::u]', "1"),
    ('w02c3::t = "m"; Check[Message[w02c3::t]; 1, err, General::t]', "err"),
    ('w02c4::t = "m"; Check[Message[w02c4::t]; 1, err, {w02c5::t, w02c4::t}]', "err"),
    ('Check[Range[10^7], "failed"]', '"failed"'),
    ('Check[Range[10^7], "failed", Sin::argx]', Raises()),
    ('Check[1, 2, 3, 4]', Raises()),
    # ── Messages ──
    ('w02s1::b = "B"; w02s1::a = "A"; Messages[w02s1]',
     '{HoldPattern[w02s1::a] :> "A", HoldPattern[w02s1::b] :> "B"}'),
    ('Messages[w02s2]', "{}"),
    ('w02s3::a = "A"; Off[w02s3::a]; Messages[w02s3]', '{HoldPattern[w02s3::a] :> $Off["A"]}'),
    # ── $Messages, $MessagePrePrint, $MessageGroups, $NewMessage ──
    ('$Messages', '{OutputStream["stdout", 1]}'),
    ('Block[{$Messages = {}}, w02mm::t = "m"; Check[Message[w02mm::t]; 1, 2]]', "2"),
    ('Head[$Messages]', "List"),
    ('$MessagePrePrint', "Automatic"),
    ('Block[{$MessagePrePrint = Sow}, Reap[Message[w02pp::argx, 1, 2]][[2]]]', "{{HoldForm[1], HoldForm[2]}}"),
    ('Block[{$MessagePrePrint = Automatic}, w02pp2::t = "x"; Check[Message[w02pp2::t, 1]; 1, 2]]', "2"),
    ('$MessageGroups[[All, 1]]', '{"Compiler", "Graphics", "Obsolete", "Packing", "Spelling", "Symbolics"}'),
    ('Head[$MessageGroups]', "List"),
    ('w02g1::t = "m"; Block[{$MessageGroups = {"Mine" :> {w02g1::t}}}, Check[Quiet[Message[w02g1::t]; 1, "Mine"], 2]]',
     "1"),
    ('Block[{$Messages = {}, $NewMessage = Function[{s, t}, Set[MessageName[s, t], "generated"]]}, '
     'Message[w02nm1::zz]]; w02nm1::zz', '"generated"'),
    ('Block[{$NewMessage = Function[{s, t}, Null]}, MessageName[w02nm2, "zz"]]', 'MessageName[w02nm2, "zz"]'),
    ('$NewMessage', "$NewMessage"),
    # ── $Failed, $Aborted, $TimedOut, $Canceled ──
    ('FailureQ[$Failed]', "True"),
    ('$Failed', "$Failed"),
    ('Head[$Failed]', "Symbol"),
    ('FailureQ[$Aborted]', "True"),
    ('CheckAbort[Abort[], $Aborted]', "$Aborted"),
    ('$Aborted === TimeConstrained[Pause[1], 0.05]', "True"),
    ('$TimedOut', "$TimedOut"),
    ('Head[$TimedOut]', "Symbol"),
    ('FailureQ[$TimedOut]', "False"),
    ('$Canceled', "$Canceled"),
    ('Enclose[Confirm[$Canceled], "Tag"]', '"ConfirmationFailed"'),
    ('Failsafe[f][$Canceled]', "$Canceled"),
    # ── Print / PrintTemporary / Echo family / QuietEcho / Monitor ──
    ('Print["a", 1]', "Null"),
    ('Print[]', "Null"),
    ('Block[{$Output = {}}, Print[x]; 2]', "2"),
    ('PrintTemporary["working"]', "Null"),
    ('PrintTemporary[1, 2]', "Null"),
    ('Block[{$Output = {}}, PrintTemporary[x]; 3]', "3"),
    ('Echo[3]', "3"),
    ('Echo[1 + 1, "lbl"]', "2"),
    ('Echo[{1, 2}, "x", Length]', "{1, 2}"),
    ('EchoLabel["x"][5]', "5"),
    ('Map[EchoLabel["v"], {1, 2}]', "{1, 2}"),
    ('EchoLabel["x"]', 'EchoLabel["x"]'),
    ('EchoFunction[Length][{1, 2, 3}]', "{1, 2, 3}"),
    ('EchoFunction["len", Length][{1, 2}]', "{1, 2}"),
    ('EchoFunction[Length]', "EchoFunction[Length]"),
    ('EchoEvaluation[1 + 1]', "2"),
    ('EchoEvaluation[2 + 3, "lbl"]', "5"),
    ('EchoEvaluation[Echo[4] + 1]', "5"),
    ('QuietEcho[Echo[3] + 1]', "4"),
    ('QuietEcho[EchoEvaluation[2]]', "2"),
    ('QuietEcho[Print[1]; 7]', "7"),
    ('Monitor[1 + 1, x]', "2"),
    ('Monitor[Table[i^2, {i, 3}], i]', "{1, 4, 9}"),
    ('Monitor[3, x, 1]', "3"),
    # ── Assert / $AssertFunction ──
    ('Assert[False]', "Null"),
    ('On[Assert]; w02r = Block[{$AssertFunction = Throw[{##}] &}, Catch[Assert[1 > 2]]]; Off[Assert]; w02r',
     "{HoldComplete[Assert[1 > 2]]}"),
    ('On[Assert]; w02r = Block[{$AssertFunction = Throw[{##}] &}, Catch[Assert[2 > 1]; ok]]; Off[Assert]; w02r', "ok"),
    ('On[Assert]; w02r = Block[{$AssertFunction = Throw[{##}] &}, Catch[Assert[x > 1, "tag"]]]; Off[Assert]; w02r',
     '{HoldComplete[Assert[x > 1, "tag"]]}'),
    ('$AssertFunction', "$AssertFunction"),
    ('Block[{$AssertFunction = Throw[0] &}, Catch[Assert[False]; 1]]', "1"),
    # ── Failure / Success / FailureQ ──
    ('Failure["x", <|"MessageTemplate" -> "a `1`", "MessageParameters" -> {3}|>]["Message"]', '"a 3"'),
    ('Failure["x", <|"MessageTemplate" -> "a `n`", "MessageParameters" -> <|"n" -> 4|>|>]["Message"]', '"a 4"'),
    ('Failure["tag", <|"Data" -> 1|>]["Tag"]', '"tag"'),
    ('Failure["tag", <|"Data" -> 1, "B" -> 2|>]["Data"]', "1"),
    ('Failure["tag", <|"MessageTemplate" -> "m", "B" -> 2|>]["Properties"]', '{"B", "Message", "MessageTemplate"}'),
    ('Failure["tag", 3]', Raises()),
    ('Success["done", <|"MessageTemplate" -> "ok `1`", "MessageParameters" -> {1}|>]["Message"]', '"ok 1"'),
    ('Success["done", <|"A" -> 1, "B" -> 2|>]["Properties"]', '{"A", "B"}'),
    ('Success["done", <|"A" -> 1|>]["A"]', "1"),
    ('FailureQ[Failure["x", <||>]]', "True"),
    ('MatchQ[Enclose[Confirm[$Failed]], Failure[_String, _Association]]', "True"),
    ('MatchQ[Success["a", <||>], Success[_, _Association]]', "True"),
    ('Cases[{TerminatedEvaluation["RecursionLimit"], 1}, TerminatedEvaluation[_String]]',
     '{TerminatedEvaluation["RecursionLimit"]}'),
    ('FailureQ[Success["x", <||>]]', "False"),
    ('FailureQ /@ {$Failed, $Aborted, 1, Missing[]}', "{True, True, False, False}"),
    # ── Failsafe ──
    ('Failsafe[f][x]', "f[x]"),
    ('Failsafe[f] /@ {1, Missing["NotAvailable"], $Failed}', '{f[1], Missing["NotAvailable"], $Failed}'),
    ('Failsafe[f][1, Missing[], 3]', "Missing[]"),
    ('Failsafe[Divide, #2 != 0 &, g][2, 0]', "g[2, 0]"),
    ('Failsafe[Divide, #2 != 0 &, Missing[] &] @@@ {{2, 2}, {4, 5}, {7, 0}}', "{1, 4/5, Missing[]}"),
    ('Failsafe[Sqrt, Positive][-1]["Tag"]', '"FailsafeFailed"'),
    ('Failsafe[f, False &, g][x, y, z]', "g[x, y, z]"),
    # ── Enclose / Confirm* ──
    ('Enclose[Confirm[1] + 1]', "2"),
    ('Enclose[Confirm[$Failed]; 1, "Tag"]', '"ConfirmationFailed"'),
    ('Enclose[Confirm[$Failed], "Message"]', '"$Failed could not be confirmed."'),
    ('Enclose[Confirm[Missing["x"]]; 5, #["ConfirmationType"] &]', '"Confirm"'),
    ('Enclose[Confirm[$Failed, "info"], #["Information"] &]', '"info"'),
    ('Catch[Enclose[Confirm[$Failed, Null, w02tag], f, w02other], _][[1]]', '"ConfirmationFailed"'),
    ('Enclose[Enclose[Confirm[$Failed]; 1, "in" &]; 2]', "2"),
    ('Enclose[ConfirmBy[4, EvenQ]]', "4"),
    ('Enclose[ConfirmBy[3, EvenQ], "Message"]', '"EvenQ[3] did not return True."'),
    ('Enclose[ConfirmBy[3, EvenQ], #["ConfirmationType"] &]', '"ConfirmBy"'),
    ('Enclose[ConfirmMatch[{1}, {_Integer}]]', "{1}"),
    ('Enclose[ConfirmMatch[x, _Integer], "Message"]', '"x does not match _Integer."'),
    ('Enclose[ConfirmMatch[1.5, _Integer], #["ConfirmationType"] &]', '"ConfirmMatch"'),
    ('Enclose[ConfirmAssert[1 < 2]; "ok"]', '"ok"'),
    ('Enclose[ConfirmAssert[1 > 2], "Message"]', '"Assertion failed."'),
    ('Enclose[ConfirmAssert[x > 2], "Tag"]', '"ConfirmationFailed"'),
    ('Enclose[ConfirmQuiet[1 + 1]]', "2"),
    ('w02cq1::t = "m"; Enclose[ConfirmQuiet[Message[w02cq1::t]; 1], #["ConfirmationType"] &]', '"ConfirmQuiet"'),
    ('w02cq2::t = "m"; Enclose[ConfirmQuiet[Message[w02cq2::t]; 1, w02cq2::u]]', "1"),
    ('w02cq3::t = "m"; Enclose[ConfirmQuiet[Message[w02cq3::t]; 1], #["HeldMessageName"] &]', "Hold[w02cq3::t]"),
    ('Quiet[Confirm[$Failed]]["Tag"]', '"confirmnotag"'),
    ('Enclose[1, 2, 3, 4]', Raises()),
    # ── Exception system ──
    ('ExceptionQ[Exception[w02tag]]', "True"),
    ('ExceptionQ[Exception[1, 2, 3]]', "False"),
    ('ExceptionQ[Exception[w02tag2], w02tag2]', "True"),
    ('ExceptionQ[Exception["w02x"], "w02y"]', "False"),
    ('ExceptionQ[5]', "False"),
    ('Exception["w02e1"]["ExceptionTag"]', '"w02e1"'),
    ('Exception["w02e2", <|"Data" -> 42|>]["Data"]', "42"),
    ('Exception["w02e3", a + b]["ExceptionPayload"]', "a + b"),
    ('CatchExceptions[1 + 2 + 3, All]', "6"),
    ('CatchExceptions[ThrowException["Error", 42], All -> handler]',
     'handler[<|"ExceptionPayload" -> 42, "ExceptionAPIVersion" -> "1", "ExceptionTag" -> "Error", '
     '"ExceptionTagList" -> {"Error"}, "ExceptionFailureTag" -> "Error"|>]'),
    ('CatchExceptions[ThrowException["Error", 42], All]["Tag"]', '"Error"'),
    ('CatchExceptions[All][ThrowException["Error", 1]]["Tag"]', '"Error"'),
    ('ExceptionQ[Catch[CatchExceptions[ThrowException["Error", 42], "ArgumentError"], _], "Error"]', "True"),
    ('CatchExceptions[ThrowException["A", 1], {"B" -> f, "A" -> g}][[0]]', "g"),
    ('ThrowException["Uncaught"]', Raises(match="Uncaught")),
    ('Catch[ThrowException["w02t", 3], _]["ExceptionPayload"]', "3"),
    ('CatchExceptions[ThrowException[Exception["w02t2", <|"K" -> 1|>]], "w02t2" -> (#K &)]', "1"),
    ('Head[RegisterExceptionType[W02IOError]]', "Success"),
    ('RegisterExceptionType[W02ArgErr, "W02General"]; {ExceptionTypeRegisteredQ[W02ArgErr], '
     'MemberQ[ExceptionTypes[], W02ArgErr]}', "{True, True}"),
    ('RegisterExceptionType[W02Dup]; Quiet[RegisterExceptionType[W02Dup]]', "$Failed"),
    ('RegisterExceptionType[W02P1, "W02Root"]; RegisterExceptionType[W02P2, W02P1]; ExceptionTypes["W02Root"]',
     "{W02P1, W02P2}"),
    ('RegisterExceptionType[W02Q1, "W02Root2"]; ExceptionTypes[W02Q1]', "{W02Q1}"),
    ('ExceptionTypes[W02Nothing]', "{}"),
    ('ExceptionTypeRegisteredQ[W02NotRegistered]', "False"),
    ('RegisterExceptionType[W02R1, "W02Base"]; ExceptionTypeRegisteredQ /@ {W02R1, "W02Base"}', "{True, False}"),
    ('RegisterExceptionType[W02S1, "W02SBase"]; RegisterExceptionType[W02S2, W02S1]; '
     'ExceptionTypeRegisteredQ[W02S2, W02S1]', "True"),
    ('RegisterExceptionType[W02U1, "W02UBase"]; ExceptionQ[Exception[W02U1, <|"Data" -> 42|>], "W02UBase"]',
     "True"),
    ('RegisterExceptionType[W02V1, "W02VBase"]; CatchExceptions["W02VBase" -> (#ExceptionTagList &)]'
     '[ThrowException[W02V1, 42]]', '{W02V1, "W02VBase"}'),
    # ── Abort / AbortProtect / CheckAbort / PropagateAborts / WithCleanup / Interrupt ──
    ('Abort[]', Raises(exc=AbortSignal)),
    ('CheckAbort[Abort[]; 1, 2]', "2"),
    ('CheckAbort[1 + 1, 2]', "2"),
    ('CheckAbort[3, x]', "3"),
    ('Reap[CheckAbort[AbortProtect[Abort[]; Sow[1]; 3], "c"]]', '{"c", {{1}}}'),
    ('AbortProtect[1 + 1]', "2"),
    ('AbortProtect[CheckAbort[Abort[], "inner"]]', '"inner"'),
    ('CheckAbort[CheckAbort[Abort[], 1, PropagateAborts -> True], 2]', "2"),
    ('CheckAbort[CheckAbort[Abort[], 1, PropagateAborts -> Automatic], 2]', "1"),
    ('TimeConstrained[CheckAbort[Pause[1], "absorbed", PropagateAborts -> False], 0.05, "tc"]', '"absorbed"'),
    ('TimeConstrained[CheckAbort[Pause[1], "x"], 0.05, "tc"]', '"tc"'),
    ('CheckAbort[1, 2, PropagateAborts -> 3]', Raises()),
    ('Reap[CheckAbort[WithCleanup[Abort[], Sow["clean"]], x]]', '{x, {{"clean"}}}'),
    ('WithCleanup[1 + 1, w02wc = 3]', "2"),
    ('Reap[WithCleanup[Sow[1], Sow[2]; 5, Sow[3]]]', "{5, {{1, 2, 3}}}"),
    ('Reap[CheckAbort[WithCleanup[Sow["i"]; Abort[], Sow["body"], Sow["clean"]], "aborted"]]',
     '{"aborted", {{"i", "clean"}}}'),
    ('Catch[WithCleanup[Throw[1], w02wy = 2]]', "1"),
    ('Interrupt[]', Raises(exc=KeyboardInterrupt)),
    ('Hold[Interrupt[]]', "Hold[Interrupt[]]"),
    ('CheckAbort[Interrupt[], 1]', Raises(exc=KeyboardInterrupt)),
    ('Hold[PropagateAborts -> False]', "Hold[PropagateAborts -> False]"),
    # ── TimeConstrained / MemoryConstrained / TimeRemaining ──
    ('TimeConstrained[Pause[2]; 1, 0.1]', "$Aborted"),
    ('TimeConstrained[While[True, 1], 0.2, "slow"]', '"slow"'),
    ('TimeConstrained[1 + 1, 1]', "2"),
    ('TimeConstrained[1, Infinity]', "1"),
    ('Quiet[TimeConstrained[1, -1]]', "TimeConstrained[1, -1]"),
    ('TimeConstrained[Pause[1], Quantity[50, "Milliseconds"], "q"]', '"q"'),
    ('MemoryConstrained[Table[ToString[i], {i, 50000}], 100000, "big"]', '"big"'),
    ('MemoryConstrained[1 + 1, 10^6]', "2"),
    ('MemoryConstrained[Table[ToString[i], {i, 50000}], 100000]', "$Aborted"),
    ('Quiet[MemoryConstrained[1, -5]]', "MemoryConstrained[1, -5]"),
    ('MemoryConstrained[3, Infinity]', "3"),
    ('TimeRemaining[]', "Infinity"),
    ('TimeConstrained[TimeRemaining[], 5]', num(0, 5)),
    ('TimeConstrained[TimeConstrained[TimeRemaining[], 10], 2]', num(0, 2)),
    # ── Timing / AbsoluteTiming / RepeatedTiming / EchoTiming / Pause ──
    ('Timing[1 + 1]', timing_pair(0, 5, 2)),
    ('Timing[Pause[0.1]; 3][[2]]', "3"),
    ('Timing[Pause[0.2]][[1]] < 0.15', "True"),
    ('AbsoluteTiming[Pause[0.1]; 3]', timing_pair(0.09, 5, 3)),
    ('AbsoluteTiming[w02at = 5; w02at]', timing_pair(0, 5, 5)),
    ('Length[AbsoluteTiming[1]]', "2"),
    ('RepeatedTiming[1 + 1, 0.05]', timing_pair(0, 1, 2)),
    ('RepeatedTiming[Pause[0.01], 0.01][[1]] >= 0.009', "True"),
    ('Quiet[RepeatedTiming[1, -1]]', "RepeatedTiming[1, -1]"),
    ('EchoTiming[1 + 1]', "2"),
    ('EchoTiming[2 + 2, "t"]', "4"),
    ('EchoTiming[3, "t", Method -> Timing]', "3"),
    ('Pause[0.01]', "Null"),
    ('AbsoluteTiming[Pause[0.05]][[1]] >= 0.05', "True"),
    ('Quiet[Pause[-1]]', "Pause[-1]"),
    # ── SessionTime / TimeUsed ──
    ('SessionTime[]', num(0)),
    ('w02t1 = SessionTime[]; Pause[0.02]; SessionTime[] - w02t1 >= 0.02', "True"),
    ('SessionTime[] >= TimeUsed[] - 1000', "True"),
    ('TimeUsed[]', num(0)),
    ('w02t2 = TimeUsed[]; Pause[0.1]; TimeUsed[] - w02t2 < 0.08', "True"),
    ('w02t3 = TimeUsed[]; Do[i^2, {i, 20000}]; TimeUsed[] - w02t3 > 0', "True"),
    # ── ByteCount / MemoryInUse / MaxMemoryUsed / MemoryAvailable / ClearSystemCache ──
    ('ByteCount[w02bcSym]', "0"),
    ('ByteCount[{True, False, Null}] == ByteCount[{a, b, c}]', "True"),
    ('ByteCount[Range[100]] > ByteCount[Range[10]] > 0', "True"),
    ('ByteCount["abcdefgh"] > ByteCount["a"]', "True"),
    ('MemoryInUse[]', num(10 ** 6)),
    ('IntegerQ[MemoryInUse[]]', "True"),
    ('MemoryInUse[$FrontEnd]', Raises()),
    ('MaxMemoryUsed[] >= 10^6', "True"),
    ('MaxMemoryUsed[Table[ToString[i], {i, 10000}]] > 10^5', "True"),
    ('MaxMemoryUsed[1] < 10^5', "True"),
    ('MemoryAvailable[]', num(1)),
    ('IntegerQ[MemoryAvailable[]]', "True"),
    ('MemoryAvailable[] > 1024', "True"),
    ('ClearSystemCache[]', "Null"),
    ('ClearSystemCache["Symbolic"]', "Null"),
    ('ClearSystemCache["Numeric"]', "Null"),
    ('Quiet[ClearSystemCache["foo"]]', 'ClearSystemCache["foo"]'),
    # ── TerminatedEvaluation ──
    ('TerminatedEvaluation["RecursionLimit"]', 'TerminatedEvaluation["RecursionLimit"]'),
    ('Head[TerminatedEvaluation["IterationLimit"]]', "TerminatedEvaluation"),
    ('TerminatedEvaluation[3]', Raises()),
    # ── TestCreate / TestObject / TestEvaluate / VerificationTest / IntermediateTest ──
    ('TestCreate[1 + 1, 2]["Outcome"]', '"NotEvaluated"'),
    ('TestCreate[1 + 1, 2, TestID -> "two"]["TestID"]', '"two"'),
    ('TestCreate[1 + 1, 2]["Input"]', "HoldForm[1 + 1]"),
    ('TestCreate[1 + 4, 2, SameTest -> Greater, TestID -> "F"]["TestCreate"]',
     'HoldForm[TestCreate[1 + 4, 2, {}, TestID -> "F", SameTest -> Greater, TimeConstraint -> Infinity, '
     'MemoryConstraint -> Infinity, MetaInformation -> None]]'),
    ('TestCreate[1, 1]["ActualOutput"]', 'Missing["NotEvaluated"]'),
    ('TestCreate[1, 1, TestID -> 3]', Raises()),
    ('TestCreate[1 + 1, 2]["TestID"] === TestCreate[1 + 1, 2]["TestID"]', "True"),
    ('Head[TestCreate[1, 1]]', "TestObject"),
    ('TestObject[3]', Raises()),
    ('MemberQ[TestCreate[1, 1]["Properties"], "ActualOutput"]', "True"),
    ('TestCreate[1, 1][{"Outcome", "FailureType"}]', '<|"Outcome" -> "NotEvaluated", "FailureType" -> None|>'),
    ('TestCreate[1, 1]["NoSuchProperty"]', 'Missing["KeyAbsent", "NoSuchProperty"]'),
    ('TestEvaluate[TestCreate[1 + 1, 2]]["Outcome"]', '"Success"'),
    ('TestEvaluate[TestCreate[1 + 1, 3]]["FailureType"]', '"SameTestFailure"'),
    ('TestEvaluate[<|"Input" :> 1 + 1, "ExpectedOutput" -> 2|>]["Outcome"]', '"Success"'),
    ('TestEvaluate[TestCreate[1 + 1, 2]]["ActualOutput"]', "HoldForm[2]"),
    ('TestEvaluate[3]', Raises()),
    ('VerificationTest[1 + 1, 2]["Outcome"]', '"Success"'),
    ('VerificationTest[1 + 1]["FailureType"]', '"SameTestFailure"'),
    ('VerificationTest[1 == 1]["Outcome"]', '"Success"'),
    ('VerificationTest[1 + 4, 2, SameTest -> Greater]["Outcome"]', '"Success"'),
    ('VerificationTest[1, 2, SameTest -> f]["FailureType"]', '"SameTestUnevaluated"'),
    ('w02v1::t = "m"; VerificationTest[Message[w02v1::t]; 1, 1][{"Outcome", "FailureType"}]',
     '<|"Outcome" -> "MessagesFailure", "FailureType" -> "SameMessagesFailure"|>'),
    ('w02v2::t = "m"; VerificationTest[Message[w02v2::t]; 1, 1, {w02v2::t}]["Outcome"]', '"Success"'),
    ('w02v3::t = "m"; VerificationTest[Message[w02v3::t]; 1, 1, {General::t}]["Outcome"]', '"Success"'),
    ('w02v4::t = "m"; Length[VerificationTest[Message[w02v4::t, 5]; 1, 1]["ActualMessages"]]', "1"),
    ('VerificationTest[Pause[1], Null, TimeConstraint -> 0.05]["FailureType"]', '"TimeConstraintFailure"'),
    ('VerificationTest[Table[ToString[i], {i, 50000}], 1, MemoryConstraint -> 10000]["FailureType"]',
     '"MemoryConstraintFailure"'),
    ('VerificationTest[1, 1, TimeConstraint -> -1]["FailureType"]', '"TimeConstraintInvalid"'),
    ('VerificationTest[1, 1, MemoryConstraint -> w02undefined]["FailureType"]', '"MemoryConstraintInvalid"'),
    ('VerificationTest[Throw[1], 1]["FailureType"]', '"UncaughtThrowFailure"'),
    ('VerificationTest[Abort[], 1]["FailureType"]', '"EvaluationAbortedFailure"'),
    ('VerificationTest[Range[10^7], 1]["Outcome"]', '"Failure"'),
    ('VerificationTest[1, 1, MetaInformation -> <|"a" -> 1|>]["MetaInformation"]', '<|"a" -> 1|>'),
    ('VerificationTest[1, 1, TestID -> "v"]["VerificationTest"][[1, 0]]', "VerificationTest"),
    ('IntermediateTest[1 + 1, 2]', "2"),
    ('IntermediateTest[1 + 1, 3]', "2"),
    ('VerificationTest[IntermediateTest[1, 2]; 3, 3]["FailureType"]', '"IntermediateTestFailure"'),
    ('w02it::t = "m"; VerificationTest[IntermediateTest[Message[w02it::t]; 1, 1]; 3, 3]["FailureType"]',
     '"IntermediateTestMessagesFailure"'),
    ('VerificationTest[IntermediateTest[1, 1]; 3, 3]["Outcome"]', '"Success"'),
    ('Length[VerificationTest[IntermediateTest[1, 1]; IntermediateTest[2, 2]; 3, 3]["IntermediateTests"]]', "2"),
    ('VerificationTest[1, 1, SameTest -> Equal, TestID -> "e"][{"TestID", "SameTest"}]',
     '<|"TestID" -> "e", "SameTest" -> Equal|>'),
    ('VerificationTest[1, 1, TimeConstraint -> 5]["TimeConstraint"]', "5"),
    ('VerificationTest[1, 1, MemoryConstraint -> 10^6]["MemoryConstraint"]', "1000000"),
    # ── TestReport / TestReportObject / options ──
    ('TestReport[{TestCreate[1, 1], TestCreate[1, 2]}]["ReportSucceeded"]', "False"),
    ('TestReport[{TestCreate[1, 1], TestCreate[2, 2]}]["ReportSucceeded"]', "True"),
    ('TestReport[TestCreate[1, 1]]["TestsSucceededCount"]', "1"),
    ('w02r::t = "m"; Length /@ TestReport[{TestCreate[1, 1], TestCreate[1, 2], TestCreate[Message[w02r::t]; 1, 1]}]'
     '["ResultsByOutcome"]', '<|"Success" -> 1, "MessagesFailure" -> 1, "Failure" -> 1, "NotEvaluated" -> 0|>'),
    ('w02rr = TestReport[{TestCreate[1, 2]}]; TestReport[{w02rr, w02rr}]["TestsFailedCount"]', "2"),
    ('Keys[TestReport[{TestCreate[1, 1, TestID -> "a"], TestCreate[2, 2, TestID -> "b"]}]["ResultsByTestID"]]',
     '{"a", "b"}'),
    ('TestReport[{VerificationTest[1, 2]}]["TestsFailedCount"]', "1"),
    ('TestReport[{3}]', Raises()),
    ('TestReport[TestCreate[1, 1]]["Title"]', "Automatic"),
    ('Head[TestReport[TestCreate[1, 1]]]', "TestReportObject"),
    ('TestReport[TestCreate[1, 1]]["Properties"]',
     '{"Title", "Results", "ResultsDataset", "ResultsByOutcome", "ResultsByTestFileName", "ResultsByTestID", '
     '"RuntimeFailures", "ReportSucceeded", "MemoryUsed", "CPUTimeUsed", "AbsoluteTimeUsed"}'),
    ('Length[TestReport[{TestCreate[1, 1], TestCreate[2, 2]}]["Results"]]', "2"),
    ('TestReport[TestCreate[1, 1]]["RuntimeFailures"]', "{}"),
    ('Head[TestReport[TestCreate[1, 1]]["ResultsDataset"]]', "Dataset"),
    ('TestReportObject[1]', Raises()),
    ('Reap[TestReport[TestCreate[1, 1], HandlerFunctions -> <|"TestEvaluated" -> (Sow[#Outcome] &)|>]][[2]]',
     '{{"Success"}}'),
    ('Reap[TestReport[TestCreate[0, 1], HandlerFunctions -> <|"UnhandledEvent" -> (Sow[#EventName] &)|>]][[2]]',
     '{{"ReportStarted", "TestCreated", "TestEvaluated", "ReportCompleted"}}'),
    ('Reap[TestReport[TestCreate[0, 1], HandlerFunctions -> <|"ReportStarted" -> (Sow[Keys[#]] &)|>]][[2]]',
     '{{{"EventName", "EventID"}}}'),
    ('TestReport[TestCreate[1, 1], HandlerFunctions -> <|"NoEvent" -> f|>]', Raises()),
    ('Reap[TestReport[TestCreate[0, 1], HandlerFunctions -> <|"TestEvaluated" -> (Sow[Keys[#]] &)|>, '
     'HandlerFunctionsKeys -> {"EventName", "Outcome"}]][[2]]', '{{{"EventName", "Outcome"}}}'),
    ('Reap[TestReport[TestCreate[0, 1], HandlerFunctions -> <|"TestEvaluated" -> (Sow[#FailureType] &)|>, '
     'HandlerFunctionsKeys -> {"FailureType"}]][[2]]', '{{"SameTestFailure"}}'),
    ('TestReport[TestCreate[1, 1], HandlerFunctionsKeys -> 3]', Raises()),
    ('TestReport[{TestCreate[1, 2]}, TestEvaluationFunction -> (TestEvaluate[TestCreate[1, 1]] &)]'
     '["ReportSucceeded"]', "True"),
    ('Reap[TestReport[TestCreate[1, 1], TestEvaluationFunction -> ((Sow[1]; TestEvaluate[#]) &)]][[2]]', "{{1}}"),
    ('TestReport[TestCreate[1, 1], TestEvaluationFunction -> (3 &)]', Raises()),
    ('TestReport[{TestCreate[1, 2]}, SameTest -> Less]["ReportSucceeded"]', "True"),
    ('TestReport[{TestCreate[Pause[1], Null]}, TimeConstraint -> 0.05]["Results"][[1]]["FailureType"]',
     '"TimeConstraintFailure"'),
    ('TestReport[{TestCreate[Table[ToString[i], {i, 50000}], 1]}, MemoryConstraint -> 10000]["Results"][[1]]'
     '["FailureType"]', '"MemoryConstraintFailure"'),
    ('TestCreate[1, 1, SameTest -> Equal]["SameTest"]', "Equal"),
    ('TestCreate[1, 1, TimeConstraint -> 3]["TimeConstraint"]', "3"),
    ('TestCreate[1, 1, MemoryConstraint -> 100]["MemoryConstraint"]', "100"),
    ('TestCreate[1, 1, TestID -> "abc"]["TestCreate"][[1, 4]]', 'TestID -> "abc"'),
    ('Hold[SameTest -> SameQ]', "Hold[SameTest -> SameQ]"),
    ('Hold[TimeConstraint -> 1, MemoryConstraint -> 2]', "Hold[TimeConstraint -> 1, MemoryConstraint -> 2]"),
    ('Hold[HandlerFunctions -> <||>, HandlerFunctionsKeys -> Automatic, TestEvaluationFunction -> TestEvaluate]',
     "Hold[HandlerFunctions -> <||>, HandlerFunctionsKeys -> Automatic, TestEvaluationFunction -> TestEvaluate]"),
    ('TestID', "TestID"),
    ('$TestFileName', '""'),
    ('TestCreate[1, 1]["TestFileName"]', '""'),
    ('Block[{$TestFileName = "/tmp/x.wlt"}, TestCreate[1, 1]["TestFileName"]]', '"/tmp/x.wlt"'),
    ('Block[{$TestFileName = "/tmp/a.wlt"}, $TestFileName]', '"/tmp/a.wlt"'),
    ('StringQ[$TestFileName]', "True"),
    ('MatchQ[VerificationTest[1, 1], TestObject[_Association]]', "True"),
    ('TestObject[<|"Outcome" -> "Success", "TestID" -> "k"|>]["TestID"]', '"k"'),
    ('TestObject[<|"Outcome" -> "NotEvaluated"|>]["ActualOutput"]', 'Missing["NotEvaluated"]'),
    ('MatchQ[TestReport[TestCreate[1, 1]], TestReportObject[_Association]]', "True"),
    ('TestReportObject[<|"Title" -> "T", "TestResults" -> <||>|>]["Title"]', '"T"'),
    ('TestReportObject[<|"Title" -> "T", "TestResults" -> <||>|>]["ReportSucceeded"]', "True"),
    ('Quiet[TestReport["/nonexistent/w02.wlt"]]', 'TestReport["/nonexistent/w02.wlt"]'),
    ('TestReport["/tmp/w02-tests.nb"]', Raises(exc=NotImplementedError, match="notebooks")),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_case(case):
    check(case)


# ── printed output ──

def test_message_output(capsys):
    ev('w02out1::t = "The argument `1` is bad."; Message[w02out1::t, 7]')
    assert capsys.readouterr().out == "w02out1::t: The argument 7 is bad.\n"


def test_message_general_fallback(capsys):
    ev('Message[w02out2::argx, w02out2, 2]')
    assert capsys.readouterr().out == "w02out2::argx: w02out2 called with 2 arguments; 1 argument is expected.\n"


def test_message_missing_text(capsys):
    ev('Message[w02out3::nodef, 5]')
    assert capsys.readouterr().out == "w02out3::nodef: -- Message text not found -- (5)\n"


def test_quiet_and_off_print_nothing(capsys):
    ev('w02out4::t = "m"; Quiet[Message[w02out4::t]]; Off[w02out4::t]; Message[w02out4::t]; On[w02out4::t]')
    assert capsys.readouterr().out == ""


def test_print_and_echo(capsys):
    ev('Print["a", 1, w02printSym]')
    ev('Echo[3, "lbl"]')
    ev('Echo[{1, 2}, "n", Length]')
    out = capsys.readouterr().out.splitlines()
    assert out == ["a1w02printSym", '>> "lbl"  3', '>> "n"  2']


def test_echo_evaluation_and_quiet_echo(capsys):
    ev('EchoEvaluation[1 + 1]')
    ev('QuietEcho[Echo[5]; EchoEvaluation[2]]')
    assert capsys.readouterr().out.splitlines() == ["<< 1 + 1", ">> 2"]


def test_echo_timing(capsys):
    ev('EchoTiming[1 + 1, "t"]')
    out = capsys.readouterr().out
    assert out.startswith('>> "t"  ') and float(out.split()[-1]) >= 0


def test_assert_message(capsys):
    ev('On[Assert]; Assert[1 > 2]; Off[Assert]; Assert[3 > 4]')
    assert capsys.readouterr().out == "Assert::asrtf: Assertion 1 > 2 failed.\n"


def test_messages_channel(capsys):
    ev('Block[{$Messages = {OutputStream["stderr", 2]}}, w02out5::t = "m"; Message[w02out5::t]]')
    got = capsys.readouterr()
    assert got.out == "" and got.err == "w02out5::t: m\n"


def test_check_issues_evalerr(capsys):
    assert ev('Check[Range[10^7], "failed"]') == "failed"
    assert capsys.readouterr().out.startswith("Range::evalerr:")


def test_report_from_file(tmp_path):
    f = tmp_path / "w02.wlt"
    f.write_text('VerificationTest[1 + 1, 2, TestID -> "add"]\n'
                 'VerificationTest[2 * 3, 7, TestID -> "mul"]\n'
                 'TestCreate[StringLength[$TestFileName] > 0, True, TestID -> "file"]\n')
    r = ev(f'w02fr = TestReport["{f}"]; {{w02fr["Title"], Keys[w02fr["ResultsByTestID"]], w02fr["TestsSucceededCount"], '
           f'Keys[w02fr["ResultsByTestFileName"]]}}')
    assert r == ev(f'{{"Test Report: w02.wlt", {{"add", "mul", "file"}}, 2, {{"{f}"}}}}')
    assert ev("$TestFileName") == ""


def test_report_file_runtime_failure(tmp_path):
    f = tmp_path / "bad.wl"
    f.write_text('VerificationTest[1, 1]\nRange[10^7]\n')
    r = ev(f'w02fr = TestReport["{f}"]; {{Length[w02fr["RuntimeFailures"]], w02fr["ReportSucceeded"]}}')
    assert r == ev("{1, False}")
