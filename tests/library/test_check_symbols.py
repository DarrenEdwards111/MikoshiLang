"""Bare-symbol and $-variable cases (tests/library/_harness.py: a bare symbol counts towards that symbol) for
heads that had no case of their own.

Sources of the expected values (never the implementation's own output):

* Wolfram Language semantics for option names, pattern objects, attributes, domains, string-pattern classes,
  graph/tree/plot options and other inert symbols: they have no value, so each evaluates to itself.
* Python's standard library as an independent reference for system variables: ``os.path.expanduser``,
  ``tempfile.gettempdir``, ``os.sep``, ``sys.byteorder`` (Wolfram's $ByteOrdering is -1 on little-endian),
  ``os.getpid``/``os.getppid``, ``os.cpu_count``, ``getpass.getuser``, ``socket.gethostname``, ``sys.maxsize``.
* Wolfram reference pages for the documented type or default of the other $-variables ($Version is a string,
  $VersionNumber a real, $CreationDate a DateObject, $Pre/$Post/$Epilog have no value by default, $KernelID and
  $KernelCount are 0 in the master kernel before LaunchKernels, $Language is "English", $DistributedContexts
  defaults to "Global`", $CompilationTarget to "WVM", $EvaluationEnvironment to "Session").

$ScriptInputString, $UserAgent, $UserAgentString, $InitializationContexts and $SyntaxHandler are left out: the
implementation's values (None or the bare symbol) differ from Wolfram's documented defaults.
"""
import getpass
import os
import socket
import sys
import tempfile

import pytest

from _harness import Pred, check, ids
from mikoshilang.expr import Expr


def _is(*types, what):
    return Pred(lambda v: isinstance(v, types) and not isinstance(v, bool) or (bool in types and isinstance(v, bool)),
                what)


def _list_of(*types, what):
    return Pred(lambda v: isinstance(v, Expr) and v.head == "List" and all(isinstance(a, types) for a in v.args),
                what)


def _head(h):
    return Pred(lambda v: isinstance(v, Expr) and v.head == h, f"an expression with head {h}")


STR, INT, REAL, BOOL = (_is(str, what="a string"), _is(int, what="an integer"), _is(float, what="a real"),
                        _is(bool, what="True or False"))
STRS = _list_of(str, what="a list of strings")
NONNEG = Pred(lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0, "a non-negative integer")
POS = Pred(lambda v: isinstance(v, int) and not isinstance(v, bool) and v > 0, "a positive integer")


def Eq(x, what=None):
    return Pred(lambda v: v == x, what or f"== {x!r}")


INERT_NAMES = [
    'Except', 'Repeated', 'RepeatedNull', 'StringExpression', 'Shortest', 'Longest', 'RegularExpression',
    'DigitCharacter', 'LetterCharacter', 'WordCharacter', 'WhitespaceCharacter', 'PunctuationCharacter',
    'HexadecimalCharacter', 'StartOfString', 'EndOfString', 'StartOfLine', 'EndOfLine', 'WordBoundary', 'Whitespace',
    'NumberString', 'IgnoreCase', 'Overlaps', 'MachinePrecision', 'HoldAllComplete', 'NHoldAll', 'NHoldFirst',
    'NHoldRest', 'Stub', 'Temporary', 'SubValuesHoldAll', 'NonThreadable', 'Locked', 'Constant', 'ReadProtected',
    'OptionsPattern', 'Definition', 'ExcludedContexts', 'IncludedContexts', 'Heads', 'IgnoringInactive', 'Constants',
    'NonConstants', 'TraceDepth', 'TraceForward', 'TraceBackward', 'TraceAbove', 'TraceOn', 'TraceOff', 'Full',
    'Manual', 'Inherited', 'Method', 'General', 'ActivateResult', 'AllowedHeads', 'ResolveContextAliases',
    'InverseFunctions', 'PatternSequence', 'OrderlessPatternSequence', 'Verbatim', 'KeyValuePattern', 'CaseOrdering',
    'Padding', 'TreeElementCoordinates', 'TreeElementLabel', 'TreeElementLabelFunction', 'TreeElementLabelStyle',
    'TreeElementShape', 'TreeElementShapeFunction', 'TreeElementSize', 'TreeElementSizeFunction', 'TreeElementStyle',
    'TreeElementStyleFunction', 'TreeLayout', 'ParentEdgeLabel', 'ParentEdgeLabelFunction', 'ParentEdgeLabelStyle',
    'ParentEdgeShapeFunction', 'ParentEdgeStyle', 'ParentEdgeStyleFunction', 'DigitBlock', 'NumberMultiplier',
    'NumberSeparator', 'AssociationFormat', 'DistanceFunction', 'RealBlockDiagonalForm', 'Standardized',
    'GameActionLabels', 'GamePlayerLabels', 'Khinchin', 'GoldenAngle', 'Reals', 'Integers', 'Rationals', 'Complexes',
    'Primes', 'Booleans', 'PositiveIntegers', 'PositiveRationals', 'PositiveReals', 'NegativeIntegers',
    'NegativeRationals', 'NegativeReals', 'NonNegativeIntegers', 'NonNegativeRationals', 'NonNegativeReals',
    'NonPositiveIntegers', 'NonPositiveRationals', 'NonPositiveReals', 'GaussianIntegers', 'Modulus', 'Extension',
    'GroupActionBase', 'TemplateSlot', 'TemplateExpression', 'TemplateSequence', 'CombinerFunction',
    'InsertionFunction', 'DiffGranularity', 'AnchoredSearch', 'WordSearch', 'WordSeparators', 'RecordSeparators',
    'SearchIndexObject', 'SearchResultObject', 'SearchQueryString', 'SearchAdjustment', 'MaxWordGap',
    'DocumentWeightingRules', 'FixedOrder', 'CaseSensitive', 'AnyOrder', 'GapPenalty', 'SimilarityRules',
    'DateFormat', 'DateGranularity', 'DayCountConvention', 'AllowedDimensions', 'DatasetTheme',
    'IncludeRelatedTables', 'GraphLayout', 'VertexLabels', 'EdgeLabels', 'VertexStyle', 'EdgeStyle',
    'VertexLabelStyle', 'EdgeLabelStyle', 'VertexShape', 'VertexShapeFunction', 'VertexSize', 'EdgeShapeFunction',
    'GraphHighlight', 'GraphHighlightStyle', 'VertexCoordinates', 'MultiedgeStyle', 'SelfLoopStyle', 'EdgeCost',
    'VertexCapacity', 'DirectedEdges', 'TaggedEdgeStyle', 'File', 'InputStream', 'OutputStream', 'EndOfFile',
    'EndOfBuffer', 'Byte', 'Character', 'Expression', 'Number', 'Record', 'Word', 'BinaryFormat', 'ByteOrdering',
    'CreateIntermediateDirectories', 'OverwriteTarget', 'RecordLists', 'FileNameForms', 'Path', 'DirichletCondition',
    'NeumannValue', 'PeriodicBoundaryCondition', 'AccuracyGoal', 'PrecisionGoal', 'WorkingPrecision',
    'MaxIterations', 'MaxStepFraction', 'MaxStepSize', 'MaxSteps', 'StartingStepSize', 'Tolerance',
    'EvaluationMonitor', 'StepMonitor', 'VerifyConvergence', 'RandomGeneratorState', 'RandomSeeding',
    'ProcessObject', 'ProcessEnvironment', 'ProcessDirectory', 'KernelObject', 'KernelConfiguration',
    'EvaluationObject', 'DistributedContexts', 'ProgressReporting', 'ScheduledTask', 'AutoRemove', 'TaskObject',
    'InitialEvaluationHistory', 'OperatingSystem', 'FollowRedirects', 'ConnectionSettings', 'CookieFunction',
    'Authentication', 'CompilationTarget', 'CompilationOptions', 'CompilerOptions', 'CompilerRuntimeErrorAction',
    'Compiled', 'AssumeDeterministic', 'TargetSystem', 'CompilerEnvironment',
]

INERT = [(n, n) for n in INERT_NAMES]

DOLLAR = [
    # unset by default: evaluates to itself
    ("$Pre", "$Pre"), ("$Post", "$Post"), ("$PrePrint", "$PrePrint"), ("$PreRead", "$PreRead"),
    ("$Epilog", "$Epilog"), ("$Aborted", "$Aborted"), ("$NoValue", "$NoValue"),
    # independent references from the Python standard library
    ("$HomeDirectory", Eq(os.path.expanduser("~"))),
    ("$TemporaryDirectory", Eq(tempfile.gettempdir())),
    ("$RootDirectory", Eq(os.path.abspath(os.sep))),
    ("$PathnameSeparator", Eq(os.sep)),
    ("$ByteOrdering", Eq(-1 if sys.byteorder == "little" else 1)),
    ("$ProcessID", Eq(os.getpid())),
    ("$ParentProcessID", Eq(os.getppid())),
    ("$ProcessorCount", Eq(os.cpu_count())),
    ("$Username", Eq(getpass.getuser())),
    ("$MachineName", Eq(socket.gethostname())),
    ("$SystemWordLength", Eq(64 if sys.maxsize > 2 ** 32 else 32)),
    ("$OperatingSystem", Eq({"win32": "Windows", "darwin": "MacOSX"}.get(sys.platform, "Unix"))),
    # documented defaults
    ("$KernelID", Eq(0)), ("$KernelCount", Eq(0)), ("$Language", Eq("English")),
    ("$DistributedContexts", Eq("Global`")), ("$CompilationTarget", Eq("WVM")),
    ("$EvaluationEnvironment", Eq("Session")), ("$Input", Eq("")), ("$InputFileName", Eq("")),
    # documented types
    ("$Version", STR), ("$System", STR), ("$SystemID", STR), ("$ProcessorType", STR), ("$MachineType", STR),
    ("$MachineID", STR), ("$SystemShell", STR), ("$InitialDirectory", STR), ("$UserBaseDirectory", STR),
    ("$BaseDirectory", STR), ("$InstallationDirectory", STR), ("$UserDocumentsDirectory", STR),
    ("$WolframDocumentsDirectory", STR), ("$DefaultLocalKernel", STR), ("$DefaultRemoteKernel", STR),
    ("$DefaultNetworkInterface", STR),
    ("$VersionNumber", REAL), ("$TimeZone", _is(int, float, what="a number")),
    ("$ModuleNumber", POS), ("$Line", POS), ("$ReleaseNumber", NONNEG), ("$SessionID", POS),
    ("$BatchOutput", BOOL), ("$BatchInput", BOOL), ("$IgnoreEOF", BOOL), ("$ProgressReporting", BOOL),
    ("$AllowInternet", BOOL),
    ("$CreationDate", _head("DateObject")), ("$StandardOutputStream", _head("OutputStream")),
    ("$StandardErrorStream", _head("OutputStream")), ("$DefaultProxyRules", _head("Association")),
    ("$CompilerEnvironment", _head("CompilerEnvironmentObject")),
    ("$Path", STRS), ("$TemplatePath", STRS), ("$CommandLine", STRS), ("$ScriptCommandLine", STRS),
    ("$CharacterEncodings", STRS), ("$FontFamilies", STRS), ("$DataStructures", STRS), ("$MachineDomains", STRS),
    ("$MachineAddresses", STRS), ("$InputStreamMethods", STRS), ("$OutputStreamMethods", STRS),
    ("$TargetSystems", STRS), ("$Cookies", _head("List")), ("$MessageList", _head("List")),
    ("$MessageGroups", _head("List")), ("$Output", _list_of(Expr, what="a list of streams")),
    ("$Urgent", _list_of(Expr, what="a list of streams")), ("$Echo", _head("List")), ("$Display", _head("List")),
    ("$DefaultParallelKernels", _head("List")), ("$ConfiguredKernels", _head("List")),
    ("$DistributedDefinitions", _head("List")), ("$SharedVariables", _head("List")),
    ("$SharedFunctions", _head("List")),
]

CASES = INERT + DOLLAR


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_check_symbols(case):
    check(case)
