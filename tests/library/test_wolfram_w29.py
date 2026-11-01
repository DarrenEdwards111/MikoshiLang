"""Differential tests for Wolfram batch W29 (system & session: $-variables, environment, processes, tasks and
parallel computing; module ``wl_system``).

Isolation (see the fixtures at the end):

* every test runs with ``os.environ``, the working directory, ``sys.stdin``, SystemOptions and the random state
  restored afterwards; settable $-variables are only changed inside Block;
* parallel cases use at most 2 local subkernels (``$DefaultParallelKernels = {2}``); kernels, tasks, shared
  variables/functions, distributed definitions and processes started by a test are closed, removed, unset or
  killed afterwards; no test signals a process it did not start;
* external programs are only ``sys.executable -c`` one-liners (bound to the symbol ``w29py``), so no random path
  appears in a test id; every symbol a test assigns starts with ``w29`` (or is Module-local);
* kernel ids and process ids are never compared with literal values, only with each other or with Python's
  ``os.getpid()`` and friends.

Reference values (written independently of the implementation):

* Wolfram documentation pages on reference.wolfram.com/language/ref/ (cited per case as "doc: Name"): documented
  defaults ($FontFamilies {}, $ScriptCommandLine {} and $ScriptInputString None outside a script, $Display {},
  $Echo {}, $IgnoreEOF False, $Language "English", $DefaultLocalKernel "Local"), the forms of RunProcess,
  ProcessInformation, ParallelTry, ParallelCombine, WaitNext, TaskObject properties, $SyntaxHandler (1-based
  error position) and Exit codes;
* Python's os, sys, platform, socket, getpass, codecs, struct and subprocess modules (cited as "py: ...") for
  system-dependent values and for the results of the child programs;
* the sequential Wolfram functions (Map, Table, Sum, Product, Select, Cases, Array) for every parallel function,
  as the documentation states the parallel versions give the same results.
"""
import codecs
import getpass
import io
import os
import platform
import re
import socket
import struct
import sys
import time

import pytest

import mikoshilang
from _harness import Pred, Raises, check, ev, ids
from mikoshilang.expr import Expr, Symbol

PID = os.getpid()


def _is_list(x, n=None):
    return isinstance(x, Expr) and x.head == "List" and (n is None or len(x.args) == n)


def _is_head(x, h):
    return isinstance(x, Expr) and x.head == h


def _sym(x, name):
    return isinstance(x, Symbol) and x.name == name


def _assoc(x):
    """An Association as a Python dict (keys must be strings)."""
    assert _is_head(x, "Association"), x
    return {r.args[0]: r.args[1] for r in x.args}


def _strlist(x):
    return _is_list(x) and all(isinstance(a, str) for a in x.args)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


# ------------------------------------------------------------------------------------------------ $-variables
SYSVARS = [
    # $Version / $VersionNumber / $ReleaseNumber: MikoshiLang values (py: mikoshilang.__version__)
    ('$Version', Pred(lambda v: isinstance(v, str) and v.startswith(mikoshilang.__version__ + " for "),
                      "starts with mikoshilang.__version__")),
    ('StringQ[$Version] && StringContainsQ[$Version, "(64-bit)" | "(32-bit)"]', 'True'),
    ('StringStartsQ[$Version, ToString[$VersionNumber]]', 'True'),                       # doc: $Version "13.0.0 for ..."
    ('$Version = "x"', Raises(ValueError)),                                              # doc: Protected
    ('$VersionNumber', Pred(lambda v: abs(float(v) - float(".".join(mikoshilang.__version__.split(".")[:2])))
                            < 1e-12, "major.minor of __version__")),
    ('NumberQ[$VersionNumber] && Head[$VersionNumber] === Real', 'True'),                 # doc: a real number
    ('$VersionNumber = 1', Raises(ValueError)),
    ('$ReleaseNumber', Pred(lambda v: v == int(re.match(r"\d+\.\d+\.(\d+)", mikoshilang.__version__ + ".0.0")
                                               .group(1)), "patch number of __version__")),
    ('IntegerQ[$ReleaseNumber] && NonNegative[$ReleaseNumber]', 'True'),
    ('$ReleaseNumber = 2', Raises(ValueError)),
    # $CreationDate: a DateObject (doc: $CreationDate); the package's modification time (py: os.path.getmtime)
    ('Head[$CreationDate]', 'DateObject'),
    ('AbsoluteTime[$CreationDate] <= AbsoluteTime[]', 'True'),
    ('$CreationDate = 1', Raises(ValueError)),
    # $OperatingSystem / $System / $SystemID / $ProcessorType / $MachineType / $SystemWordLength (py: platform)
    ('$OperatingSystem', Pred(lambda v: v == {"win32": "Windows", "darwin": "MacOSX"}.get(sys.platform, "Unix"),
                              "py: sys.platform")),
    ('MemberQ[{"Unix", "MacOSX", "Windows"}, $OperatingSystem]', 'True'),                 # doc: $OperatingSystem
    ('$OperatingSystem = "Windows"', Raises(ValueError)),
    ('$System', Pred(lambda v: isinstance(v, str) and v.startswith({"win32": "Microsoft Windows",
                                                                     "darwin": "Mac OS X"}.get(sys.platform,
                                                                                               "Linux")),
                     "py: sys.platform")),
    ('StringContainsQ[$System, "64-bit"] == ($SystemWordLength == 64)', 'True'),
    ('$System = "x"', Raises(ValueError)),
    ('$SystemID', Pred(lambda v: isinstance(v, str) and v.split("-")[0] == {"win32": "Windows", "darwin": "MacOSX"}
                       .get(sys.platform, "Linux"), "py: sys.platform")),
    ('StringMatchQ[$SystemID, ("Linux" | "MacOSX" | "Windows") ~~ ___]', 'True'),        # doc: $SystemID values
    ('$SystemID = "x"', Raises(ValueError)),
    ('$ProcessorType', Pred(lambda v: v == {"x86_64": "x86-64", "amd64": "x86-64", "aarch64": "ARM64",
                                            "arm64": "ARM64"}.get(platform.machine().lower(), v),
                            "py: platform.machine")),
    ('StringQ[$ProcessorType]', 'True'),
    ('$ProcessorType = "x"', Raises(ValueError)),
    ('$MachineType', '"PC"'),                                                             # doc: "PC" on PCs and Macs
    ('StringQ[$MachineType]', 'True'),
    ('$MachineType = "x"', Raises(ValueError)),
    ('$SystemWordLength', struct.calcsize("P") * 8),                                      # py: struct.calcsize("P")
    ('MemberQ[{32, 64}, $SystemWordLength]', 'True'),
    ('$SystemWordLength = 32', Raises(ValueError)),
    # process / machine identity (py: os, socket, getpass)
    ('$ProcessorCount', os.cpu_count() or 1),                                             # py: os.cpu_count()
    ('IntegerQ[$ProcessorCount] && $ProcessorCount >= 1', 'True'),
    ('$ProcessorCount = 1', Raises(ValueError)),
    ('Unprotect[$ProcessorCount]; $ProcessorCount = 1; {$ProcessorCount, Unset[$ProcessorCount]; '
     'Protect[$ProcessorCount]; $ProcessorCount == %d}' % (os.cpu_count() or 1), '{1, True}'),
    ('$ProcessID', PID),                                                                  # py: os.getpid()
    ('IntegerQ[$ProcessID] && Positive[$ProcessID]', 'True'),
    ('$ProcessID = 1', Raises(ValueError)),
    ('$ParentProcessID', os.getppid()),                                                   # py: os.getppid()
    ('$ParentProcessID != $ProcessID', 'True'),
    ('$ParentProcessID = 1', Raises(ValueError)),
    ('$MachineName', Pred(lambda v: v == socket.gethostname().split(".")[0], "py: socket.gethostname()")),
    ('StringQ[$MachineName] && StringFreeQ[$MachineName, "."]', 'True'),
    ('$MachineName = "x"', Raises(ValueError)),
    ('$MachineDomains', Pred(lambda v: _strlist(v), "a list of strings")),
    ('Length[$MachineDomains] <= 1', 'True'),
    ('$MachineDomains = {}', Raises(ValueError)),
    ('$MachineID', Pred(lambda v: isinstance(v, str) and v != "", "a nonempty string")),
    ('$MachineID === $MachineID', 'True'),                                                 # stable in a session
    ('$MachineID = "1"', Raises(ValueError)),
    ('$SessionID', Pred(lambda v: _int(v) and v > 0, "a positive integer")),
    ('$SessionID === $SessionID', 'True'),                                                 # doc: unique per session
    ('$SessionID = 1', Raises(ValueError)),
    ('$Username', Pred(lambda v: v == getpass.getuser(), "py: getpass.getuser()")),
    ('StringQ[$Username]', 'True'),
    ('$Username = "x"', Raises(ValueError)),
    ('$CommandLine', Pred(lambda v: _strlist(v) and list(v.args) == list(getattr(sys, "orig_argv", None) or
                                                                  [sys.executable] + sys.argv[1:]),
                          "py: sys.orig_argv (3.10+), else sys.executable + sys.argv[1:]")),
    ('ListQ[$CommandLine] && Length[$CommandLine] >= 1', 'True'),
    ('$CommandLine = {}', Raises(ValueError)),
    # script launcher values (doc: $ScriptCommandLine "{} if not run from a script", $ScriptInputString None)
    ('$ScriptCommandLine', '{}'),
    ('ListQ[$ScriptCommandLine]', 'True'),
    ('$ScriptCommandLine = {"a"}', Raises(ValueError)),
    ('$ScriptInputString', 'None'),
    ('$ScriptInputString === None', 'True'),
    ('$ScriptInputString = "x"', Raises(ValueError)),
    # $SystemShell (doc: "/bin/sh" on Unix; py: os.environ COMSPEC on Windows)
    ('$SystemShell', Pred(lambda v: v == (os.environ.get("COMSPEC", "cmd.exe") if os.name == "nt" else "/bin/sh"),
                          "py: os.name")),
    ('StringQ[$SystemShell]', 'True'),
    ('$SystemShell = "x"', Raises(ValueError)),
    # $CharacterEncodings: names with a Python codec (py: codecs.lookup)
    # doc: $CharacterEncodings lists names such as "UTF8", "ASCII", "ISOLatin1", "ShiftJIS"; py: codecs.lookup
    ('Select[{"UTF8", "ASCII", "ISOLatin1", "ShiftJIS", "WindowsANSI"}, MemberQ[$CharacterEncodings, #] &]',
     '{"UTF8", "ASCII", "ISOLatin1", "ShiftJIS", "WindowsANSI"}'),
    ('$CharacterEncodings', Pred(lambda v: _strlist(v) and list(v.args) == sorted(v.args, key=str.lower),
                                 "sorted list of names")),
    ('MemberQ[$CharacterEncodings, "UTF-8"] && MemberQ[$CharacterEncodings, "ASCII"]', 'True'),
    ('Length[$CharacterEncodings] > 10', 'True'),
    ('$CharacterEncodings = {}', Raises(ValueError)),
    ('$FontFamilies', '{}'),                                                               # doc: {} without front end
    ('ListQ[$FontFamilies]', 'True'),
    ('$FontFamilies = {"a"}', Raises(ValueError)),
    ('$BatchOutput', 'False'),
    ('BooleanQ[$BatchOutput]', 'True'),
    ('$BatchOutput = True', Raises(ValueError)),
    ('$BatchInput', Pred(lambda v: v is (not sys.stdin.isatty()), "py: not sys.stdin.isatty()")),
    ('BooleanQ[$BatchInput]', 'True'),
    ('Block[{$BatchInput = True}, $BatchInput]', 'True'),
    ('$IgnoreEOF', 'False'),                                                               # doc: default False
    ('Block[{$IgnoreEOF = True}, $IgnoreEOF]', 'True'),
    ('Block[{$IgnoreEOF = True}, 1]; $IgnoreEOF', 'False'),
    ('$Language', '"English"'),                                                            # doc: default "English"
    ('Block[{$Language = "French"}, $Language]', '"French"'),
    ('Block[{$Language = "French"}, 1]; $Language', '"English"'),
    ('$Display', '{}'),                                                                     # doc: initial {}
    ('Block[{$Display = {"stdout"}}, $Display]', '{"stdout"}'),
    ('ListQ[$Display]', 'True'),
    ('$Echo', '{}'),                                                                        # doc: initial {}
    ('Block[{$Echo = {OutputStream["stdout", 1]}}, Length[$Echo]]', '1'),
    ('ListQ[$Echo]', 'True'),
    ('$Output', '{OutputStream["stdout", 1]}'),                                             # doc: {OutputStream[stdout]}
    ('Block[{$Output = {}}, Print["w29 silent"]; 1]', '1'),
    ('Head[First[$Output]]', 'OutputStream'),
    ('$Urgent', '{OutputStream["stdout", 1]}'),
    ('Head[First[$Urgent]]', 'OutputStream'),
    ('Block[{$Urgent = {}}, Length[$Urgent]]', '0'),
    ('$ProgressReporting', 'False'),
    ('Block[{$ProgressReporting = True}, $ProgressReporting]', 'True'),
    ('Block[{$ProgressReporting = True}, Total[{1, 2, 3}]]', '6'),                         # results unchanged
    ('$EvaluationEnvironment', '"Session"'),                                                # doc: "Session"
    ('StringQ[$EvaluationEnvironment]', 'True'),
    ('$EvaluationEnvironment = "x"', Raises(ValueError)),
    ('$Input', '""'),                                                                        # doc: "" interactively
    ('StringQ[$Input]', 'True'),
    ('Block[{$Input = "w29.m"}, $Input]', '"w29.m"'),
    # $SyntaxHandler: no value until assigned (doc); the handler itself: test_syntax_handler_and_echo
    ('$SyntaxHandler', Pred(lambda v: _sym(v, "$SyntaxHandler"), "no value")),
    ('Block[{$SyntaxHandler = w29sh1}, $SyntaxHandler]', 'w29sh1'),
    ('Block[{$SyntaxHandler = (#1 &)}, 1 + 1]', '2'),                                       # valid input: unused
    # inert symbols (doc: $NoValue, $InitializationContexts evaluate to themselves without notebooks)
    ('$NoValue', Pred(lambda v: _sym(v, "$NoValue"), "inert")),
    ('Head[$NoValue]', 'Symbol'),
    ('{$NoValue} === {$NoValue}', 'True'),
    ('$InitializationContexts', Pred(lambda v: _sym(v, "$InitializationContexts"), "inert")),
    ('Head[$InitializationContexts]', 'Symbol'),
    ('{$InitializationContexts} === {$InitializationContexts}', 'True'),
    # Share (doc: Share[] gives the number of bytes saved; MikoshiLang shares nothing, so 0)
    ('Share[]', '0'),
    ('Share[{1, 2, {1, 2}}]', '0'),
    ('IntegerQ[Share[x]]', 'True'),
    ('Share[1, 2]', Raises(ValueError)),                                                    # 0 or 1 argument
]


# ------------------------------------------------------------------------- environment (py: os.environ)
ENVIRONMENT = [
    ('Environment["PATH"]', Pred(lambda v: v == os.environ.get("PATH", "$Failed") or
                                 (os.environ.get("PATH") is None and _sym(v, "$Failed")), "py: os.environ['PATH']")),
    ('Environment["W29_SURELY_UNSET_VAR"]', '$Failed'),                                  # doc: $Failed when unset
    ('SetEnvironment["W29_A" -> "alpha"]; Environment["W29_A"]', '"alpha"'),
    ('SetEnvironment["W29_A" -> "x"]; SetEnvironment["W29_A" -> None]; Environment["W29_A"]', '$Failed'),
    ('Environment[w29sym]', Pred(lambda v: _is_head(v, "Environment"), "symbolic: unevaluated")),
    ('Environment[1]', Raises(TypeError)),
    ('Environment["A=B"]', Raises(ValueError)),
    ('GetEnvironment["W29_SURELY_UNSET_VAR"]', '"W29_SURELY_UNSET_VAR" -> None'),        # doc: None when unset
    ('SetEnvironment["W29_B" -> "beta"]; GetEnvironment["W29_B"]', '"W29_B" -> "beta"'),
    ('SetEnvironment[{"W29_C" -> "1", "W29_D" -> "2"}]; GetEnvironment[{"W29_C", "W29_D"}]',
     '{"W29_C" -> "1", "W29_D" -> "2"}'),
    ('GetEnvironment[]', Pred(lambda v: _is_list(v) and {r.args[0]: r.args[1] for r in v.args} == dict(os.environ),
                              "py: dict(os.environ)")),
    ('SetEnvironment["W29_E" -> "e"]; MemberQ[GetEnvironment[], "W29_E" -> "e"]', 'True'),
    ('GetEnvironment[w29sym]', Pred(lambda v: _is_head(v, "GetEnvironment"), "symbolic: unevaluated")),
    ('GetEnvironment[1]', Raises(TypeError)),
    ('SetEnvironment["W29_F" -> "f"]', 'Null'),                                           # doc: returns Null
    ('SetEnvironment[<|"W29_G" -> "g", "W29_H" -> 7|>]; {Environment["W29_G"], Environment["W29_H"]}',
     '{"g", "7"}'),
    ('SetEnvironment["W29_I" -> "i"]; RunProcess[{w29py, "-c", "import os; print(os.environ[\'W29_I\'])"}, '
     '"StandardOutput"]', '"i\\n"'),                                                     # inherited by children
    ('SetEnvironment[w29sym]', Pred(lambda v: _is_head(v, "SetEnvironment"), "symbolic: unevaluated")),
    ('SetEnvironment["W29_J" -> True]', Raises(TypeError)),
    ('SetEnvironment["W29_J" -> 1.5]', Raises(TypeError)),
    ('SetEnvironment["=" -> "x"]', Raises(ValueError)),
]

# --------------------------------------------------------------------------------- random generator state
RANDOM = [
    # BlockRandom (doc: "the random numbers generated after BlockRandom are the same as without it")
    ('BlockRandom[SeedRandom[5]; RandomReal[]; w29a = RandomReal[]; SeedRandom[5]; RandomReal[]; '
     'BlockRandom[RandomReal[{0, 1}, 10]]; RandomReal[] == w29a]', 'True'),
    ('BlockRandom[SeedRandom[11]; w29a = BlockRandom[RandomInteger[100, 5]]; w29b = RandomInteger[100, 5]; '
     'w29a == w29b]', 'True'),                                                                # state restored
    ('BlockRandom[RandomInteger[10, 3], RandomSeeding -> 7] == BlockRandom[RandomInteger[10, 3], '
     'RandomSeeding -> 7]', 'True'),                                                          # doc: RandomSeeding
    ('BlockRandom[BlockRandom[RandomReal[], RandomSeeding -> "w29"] == BlockRandom[RandomReal[], '
     'RandomSeeding -> "w29"]]', 'True'),
    ('BlockRandom[1 + 1]', '2'),
    ('BlockRandom[SeedRandom[3]; w29s = $RandomGeneratorState; w29x = RandomReal[{0, 1}, 4]; '
     'BlockRandom[RandomReal[{0, 1}, 4], RandomSeeding -> w29s] == w29x]', 'True'),
    ('BlockRandom[w29q = 1; w29q + 1]', '2'),
    ('BlockRandom[1, RandomSeeding -> 1.5]', Raises(TypeError)),
    ('BlockRandom[1, RandomSeeding -> True]', Raises(TypeError)),
    ('BlockRandom[1, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('BlockRandom[1, RandomSeeding -> RandomGeneratorState[{"ExtendedCA", {}}, <||>]]', Raises(NotImplementedError)),
    ('BlockRandom[1, RandomSeeding -> Automatic]', '1'),
    ('BlockRandom[1, RandomSeeding -> Inherited]', '1'),
    # RandomSeeding (doc: equal seeds give equal sequences)
    ('BlockRandom[RandomChoice[Range[100], 5], RandomSeeding -> 42] === BlockRandom[RandomChoice[Range[100], 5], '
     'RandomSeeding -> 42]', 'True'),
    ('BlockRandom[RandomReal[], RandomSeeding -> 1] != BlockRandom[RandomReal[], RandomSeeding -> 2]', 'True'),
    ('BlockRandom[SeedRandom[42]; RandomReal[]] == BlockRandom[RandomReal[], RandomSeeding -> 42]', 'True'),
    ('BlockRandom[1, RandomSeeding -> {1}]', Raises(TypeError)),
    # RandomGeneratorState
    ('Head[$RandomGeneratorState]', 'RandomGeneratorState'),
    ('$RandomGeneratorState["Method"]', '{"MersenneTwister", {}}'),
    ('$RandomGeneratorState["CrossSessionPersistent"]', 'True'),
    ('$RandomGeneratorState["Properties"]', '{"CrossSessionPersistent", "Method", "StateHash"}'),
    ('IntegerQ[$RandomGeneratorState["StateHash"]]', 'True'),
    ('BlockRandom[SeedRandom[9]; w29h1 = $RandomGeneratorState["StateHash"]; RandomReal[]; '
     'w29h2 = $RandomGeneratorState["StateHash"]; SeedRandom[9]; {w29h1 == w29h2, '
     'w29h1 == $RandomGeneratorState["StateHash"]}]', '{False, True}'),
    ('$RandomGeneratorState["w29nope"]', Raises(ValueError)),
    ('RandomGeneratorState[{"MersenneTwister", {}}, <|"Version" -> 3|>]["StateHash"]', Raises(ValueError)),
    ('RandomGeneratorState[w29x]', Pred(lambda v: _is_head(v, "RandomGeneratorState"), "symbolic: inert")),
    ('BlockRandom[SeedRandom[1]; w29st = $RandomGeneratorState; w29v = RandomInteger[10^6]; '
     'BlockRandom[RandomInteger[10^6], RandomSeeding -> w29st] == w29v]', 'True'),
    ('$RandomGeneratorState = 1', Raises(ValueError)),
    ('BlockRandom[SeedRandom[2]; w29u = $RandomGeneratorState; RandomReal[]; w29u === $RandomGeneratorState]',
     'False'),
]

# ----------------------------------------------------------------------------------------- SystemInformation
SYSINFO = [
    ('SystemInformation["Kernel", "Version"] === $Version', 'True'),
    ('SystemInformation["Kernel", "ProcessID"]', PID),                                       # py: os.getpid()
    ('SystemInformation["Kernel", "OperatingSystemRelease"]', Pred(lambda v: v == (platform.release() or v),
                                                                     "py: platform.release()")),
    ('SystemInformation["Machine", "ProcessorCount"]', os.cpu_count() or 1),
    ('SystemInformation["Components"]', '{"Kernel", "Machine", "Links", "Network", "FrontEnd", "Devices"}'),
    ('SystemInformation["FrontEnd"]', Pred(lambda v: _is_head(v, "Missing"), "no front end: Missing")),
    ('Head[SystemInformation["Machine", "PhysicalTotal"]]', Pred(lambda v: v in (Symbol("Quantity"),
                                                                                  Symbol("Missing"))
                                                                  or str(v) in ("Quantity", "Missing"),
                                                                  "Quantity or Missing")),
    ('Lookup[SystemInformation["Small"], "ProcessID"] == $ProcessID', 'True'),
    ('SubsetQ[Keys[SystemInformation["Kernel"]], {"Version", "ProcessID", "MachineName", "SystemID"}]', 'True'),
    ('SystemInformation["w29nope"]', Raises(ValueError)),
    ('SystemInformation["Kernel", "w29nope"]', Raises(ValueError)),
    ('SystemInformation["Components", "x"]', Raises(TypeError)),
    ('SystemInformation[w29x]', Pred(lambda v: _is_head(v, "SystemInformation"), "symbolic: unevaluated")),
    # SystemOptions / SetSystemOptions
    ('SystemOptions["ProcessOptions"]', '{"ProcessOptions" -> {"KillTimeout" -> 1.}}'),
    ('SystemOptions["ParallelOptions" -> "RecoveryMode"]', '{"ParallelOptions" -> {"RecoveryMode" -> "ReQueue"}}'),
    ('SystemOptions["RecursionOptions", "RecursionLimit"] == {"RecursionOptions" -> {"RecursionLimit" -> '
     '$RecursionLimit}}', 'True'),
    ('Length[SystemOptions[]]', '3'),
    ('SystemOptions["w29nope"]', Raises(ValueError)),
    ('SystemOptions["ProcessOptions" -> "w29nope"]', Raises(ValueError)),
    ('SystemOptions[1]', Raises(TypeError)),
    ('SetSystemOptions["ProcessOptions" -> {"KillTimeout" -> 2}]', '{"ProcessOptions" -> {"KillTimeout" -> 2.}}'),
    ('SetSystemOptions["ParallelOptions" -> {"RecoveryMode" -> "Abandon"}]; '
     'SystemOptions["ParallelOptions" -> "RecoveryMode"]', '{"ParallelOptions" -> {"RecoveryMode" -> "Abandon"}}'),
    ('Block[{$RecursionLimit = $RecursionLimit}, SetSystemOptions["RecursionOptions" -> {"RecursionLimit" -> 500}]; '
     '$RecursionLimit]', '500'),
    ('SetSystemOptions["ProcessOptions" -> {"KillTimeout" -> -1}]', Raises(ValueError)),
    ('SetSystemOptions["w29nope" -> {"a" -> 1}]', Raises(ValueError)),
    ('SetSystemOptions[1]', Raises(TypeError)),
    ('SetSystemOptions[]', Raises(TypeError)),
    ('SetSystemOptions["ParallelOptions" -> {"RecoveryMode" -> "w29bad"}]; 1', Raises(ValueError)),
]

# ---------------------------------------------------------------------------------------------- Exit, Quit
EXIT = [
    ('Exit[]', Raises(SystemExit)),                                                          # doc: exit code 0
    ('Exit[3]', Raises(SystemExit, match="3")),
    ('Exit[256]', Raises(ValueError)),
    ('Exit[-1]', Raises((ValueError, TypeError))),
    ('Exit[w29x]', Pred(lambda v: _is_head(v, "Exit"), "symbolic: unevaluated")),
    ('Quit[]', Raises(SystemExit)),
    ('Quit[7]', Raises(SystemExit, match="7")),
    ('Quit[1000]', Raises(ValueError)),
    ('Quit["a"]', Raises(TypeError)),
    ('Quit[w29x]', Pred(lambda v: _is_head(v, "Quit"), "symbolic: unevaluated")),
]

# ------------------------------------------------------------------------- Input / InputString with no stdin
# (the w29iso fixture replaces sys.stdin by an empty StringIO; the prompt goes to $Urgent, here silenced)
INPUT = [
    ('Block[{$Urgent = {}}, Input[]]', 'EndOfFile'),                                          # doc: EndOfFile
    ('Block[{$Urgent = {}}, Input["w29> "]]', 'EndOfFile'),
    ('Input[1, 2, 3, 4]', Raises(TypeError)),
    ('Block[{$Urgent = {}}, InputString[]]', 'EndOfFile'),
    ('Block[{$Urgent = {}}, InputString["w29> "]]', 'EndOfFile'),
    ('InputString[1, 2, 3, 4]', Raises(TypeError)),
]

# --------------------------------------------- options of other batches recorded here (W08, W27)
OTHER = [
    # InitialEvaluationHistory (doc: BayesianMinimization option; W08)
    ('BlockRandom[BayesianMinimization[(# - 1)^2 &, {-2, 2}, InitialEvaluationHistory -> {0. -> 1., 1. -> 0.}, '
     'MaxIterations -> 2]["MinimumConfiguration"], RandomSeeding -> 1]', Pred(lambda v: abs(float(v) - 1.0) < 0.3,
                                                                              "optimum found near 1")),
    ('BlockRandom[BayesianMinimization[(# - 1)^2 &, {-2, 2}, InitialEvaluationHistory -> {1. -> 0.}, '
     'MaxIterations -> 1]["MinimumValue"], RandomSeeding -> 1]', Pred(lambda v: float(v) <= 1e-12,
                                                                     "the history's optimum 0 is kept")),
    ('BlockRandom[BayesianMinimization[#^2 &, {-1, 1}, InitialEvaluationHistory -> {0.5 -> 0.25}, '
     'MaxIterations -> 3]["MinimumValue"] <= 0.25, RandomSeeding -> 2]', 'True'),
    ('BayesianMinimization[#^2 &, {-1, 1}, InitialEvaluationHistory -> 5]', Raises()),
    # OperatingSystem (doc: FileNameJoin option, W27)
    ('FileNameJoin[{"a", "b"}, OperatingSystem -> "Windows"]', '"a\\\\b"'),
    ('FileNameJoin[{"a", "b"}, OperatingSystem -> "Unix"]', '"a/b"'),
    ('FileNameSplit["a\\\\b", OperatingSystem -> "Windows"]', '{"a", "b"}'),
    ('FileNameJoin[{"a", "b"}, OperatingSystem -> $OperatingSystem] == FileNameJoin[{"a", "b"}]', 'True'),
    ('FileNameJoin[{"a", "b"}, OperatingSystem -> "w29os"]', Raises()),
]


# ------------------------------------------------------------------------------------ external processes
# every program is `sys.executable -c <one-liner>` (w29py), or the same through the shell (w29sh: quoted path);
# reference values: py: subprocess.run([sys.executable, "-c", code]) gives the same exit code and output
PROC = [
    # RunProcess (doc: <|"ExitCode" -> 0, "StandardOutput" -> ..., "StandardError" -> ...|>)
    ('RunProcess[{w29py, "-c", "print(6*7)"}]', '<|"ExitCode" -> 0, "StandardOutput" -> "42\\n", '
                                                 '"StandardError" -> ""|>'),
    ('RunProcess[{w29py, "-c", "import sys; sys.exit(3)"}, "ExitCode"]', '3'),
    ('RunProcess[{w29py, "-c", "import sys; sys.stderr.write(\'e\')"}, "StandardError"]', '"e"'),
    ('RunProcess[{w29py, "-c", "import sys; print(sys.stdin.read().upper())"}, "StandardOutput", "abc"]',
     '"ABC\\n"'),                                                                          # doc: input string
    ('RunProcess[{w29py, "-c", "import sys; print(sys.argv[1:])", "x", 2, "name" -> "v", "n" -> 1}, '
     '"StandardOutput"]', '"[\'x\', \'2\', \'--name=v\', \'-n=1\']\\n"'),                     # doc: "name" -> value
    ('RunProcess[{w29py, "-c", "print(1)"}, All]["ExitCode"]', '0'),
    ('RunProcess[{w29py, "-c", "import os; print(os.environ.get(\'W29_PE\'))"}, "StandardOutput", '
     'ProcessEnvironment -> <|"W29_PE" -> "pe"|>]', '"pe\\n"'),
    ('RunProcess[{w29py, "-c", "import os; print(os.path.basename(os.getcwd()))"}, "StandardOutput", '
     'ProcessDirectory -> w29dir]', Pred(lambda v: isinstance(v, str) and v.strip().startswith("w29"),
                                         "child cwd is the test directory")),
    ('RunProcess[w29x]', Pred(lambda v: _is_head(v, "RunProcess"), "symbolic: unevaluated")),
    ('RunProcess[{"w29-no-such-program-xyz"}]', Raises(ValueError)),
    ('RunProcess[{}]', Raises(ValueError)),
    ('RunProcess[{w29py, "-c", "1"}, "w29nope"]', Raises(ValueError)),
    ('RunProcess[{w29py, True}]', Raises(TypeError)),
    ('RunProcess[{w29py, "-c", "1"}, All, "", w29opt -> 1]', Raises((TypeError, ValueError))),
    # ProcessEnvironment / ProcessDirectory (doc: options of RunProcess and StartProcess)
    ('RunProcess[{w29py, "-c", "import os; print(\'W29_PE2\' in os.environ)"}, "StandardOutput", '
     'ProcessEnvironment -> <|"W29_PE2" -> None|>]', '"False\\n"'),
    ('SetEnvironment["W29_PE3" -> "outer"]; RunProcess[{w29py, "-c", "import os; print(os.environ[\'W29_PE3\'])"}, '
     '"StandardOutput", ProcessEnvironment -> <|"W29_PE4" -> "x"|>]', '"outer\\n"'),       # merged, inherited
    ('RunProcess[{w29py, "-c", "import os; print(os.environ[\'W29_PE5\'])"}, "StandardOutput", '
     'ProcessEnvironment -> {"W29_PE5" -> 5}]', '"5\\n"'),
    ('RunProcess[{w29py, "-c", "1"}, ProcessEnvironment -> <|"W29_PE6" -> 1.5|>]', Raises(TypeError)),
    ('RunProcess[{w29py, "-c", "1"}, ProcessEnvironment -> 5]', Raises((TypeError, ValueError))),
    ('RunProcess[{w29py, "-c", "import os; print(os.path.isdir(\'sub\'))"}, "StandardOutput", '
     'ProcessDirectory -> w29dir]', '"True\\n"'),
    ('RunProcess[{w29py, "-c", "import os; print(os.path.basename(os.getcwd()))"}, "StandardOutput", '
     'ProcessDirectory -> FileNameJoin[{w29dir, "sub"}]]', '"sub\\n"'),
    ('RunProcess[{w29py, "-c", "1"}, ProcessDirectory -> FileNameJoin[{w29dir, "w29-missing"}]]',
     Raises(ValueError)),
    ('RunProcess[{w29py, "-c", "1"}, ProcessDirectory -> 5]', Raises(TypeError)),
    # StartProcess / ProcessObject / ProcessStatus / ProcessInformation / ProcessConnection / KillProcess
    ('w29p = StartProcess[{w29py, "-c", "import sys; print(sys.stdin.readline().upper())"}]; '
     'WriteLine[w29p, "abc"]; ReadLine[w29p]', '"ABC"'),                                     # doc: ReadLine[proc]
    ('w29p = StartProcess[{w29py, "-c", "print(1)"}]; Keys[First[w29p]]',
     '{"ManagedProcess", "UID", "PID", "PPID"}'),
    ('w29p = StartProcess[{w29py, "-c", "print(1)"}]; {w29p["ManagedProcess"], w29p["PPID"] == $ProcessID, '
     'IntegerQ[w29p["PID"]]}', '{True, True, True}'),
    ('w29p = StartProcess[{w29py, "-c", "import os; print(os.getpid())"}]; '
     'ToExpression[ReadLine[w29p]] == w29p["PID"]', 'True'),                                  # py: os.getpid()
    ('StartProcess[{w29py, "-c", "1"}, "extra"]', Pred(lambda v: _is_head(v, "StartProcess"),
                                                       "doc: unevaluated with extra arguments")),
    ('StartProcess[w29x]', Pred(lambda v: _is_head(v, "StartProcess"), "symbolic: unevaluated")),
    ('StartProcess["w29-no-such-program-xyz"]', Raises(ValueError)),
    ('StartProcess[{w29py, "-c", "1"}, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('w29p = StartProcess[{w29py, "-c", "import os; print(os.environ[\'W29_SP\'])"}, '
     'ProcessEnvironment -> <|"W29_SP" -> "sp"|>]; ReadLine[w29p]', '"sp"'),
    ('w29p = StartProcess[{w29py, "-c", "import os; print(os.path.basename(os.getcwd()))"}, '
     'ProcessDirectory -> FileNameJoin[{w29dir, "sub"}]]; ReadLine[w29p]', '"sub"'),
    ('w29p = StartProcess[{w29py, "-c", "input()"}]; w29s1 = ProcessStatus[w29p]; WriteLine[w29p, ""]; '
     'TimeConstrained[While[ProcessStatus[w29p] == "Running", Pause[0.02]], 20]; {w29s1, ProcessStatus[w29p]}',
     '{"Running", "Finished"}'),
    ('w29p = StartProcess[{w29py, "-c", "input()"}]; {ProcessStatus[w29p, "Running"], '
     'ProcessStatus[w29p, "Finished"]}', '{True, False}'),
    ('ProcessStatus[ProcessObject[$ProcessID]]', '"Running"'),                                   # this process
    ('ProcessStatus[w29x]', Pred(lambda v: _is_head(v, "ProcessStatus"), "symbolic: unevaluated")),
    ('ProcessStatus[5]', Raises(TypeError)),
    ('w29p = StartProcess[{w29py, "-c", "1"}]; ProcessStatus[w29p, "w29nope"]', Raises(ValueError)),
    ('ProcessStatus[ProcessObject[<|"ManagedProcess" -> True, "UID" -> -7|>]]', Raises(ValueError)),
    ('w29p = StartProcess[{w29py, "-c", "import sys; sys.exit(5)"}]; '
     'TimeConstrained[While[ProcessStatus[w29p] == "Running", Pause[0.02]], 20]; ProcessInformation[w29p]',
     '<|"ExitCode" -> 5|>'),                                                                  # py: exit code 5
    ('w29p = StartProcess[{w29py, "-c", "input()"}]; ProcessInformation[w29p, "ExitCode"]', 'None'),
    ('w29p = StartProcess[{w29py, "-c", "input()"}]; {ProcessInformation[w29p, "Program"] == w29py, '
     'ProcessInformation[w29p, "Arguments"], ProcessInformation[w29p, "Status"]}',
     '{True, {"-c", "input()"}, "Running"}'),
    ('w29p = StartProcess[{w29py, "-c", "1"}]; ProcessInformation[w29p, "w29nope"]', Raises(ValueError)),
    ('ProcessInformation[ProcessObject[$ProcessID]]', Raises(ValueError)),                      # not started here
    ('ProcessInformation[w29x]', Pred(lambda v: _is_head(v, "ProcessInformation"), "symbolic: unevaluated")),
    ('ProcessInformation["x"]', Raises(TypeError)),
    ('w29p = StartProcess[{w29py, "-c", "import sys; print(sys.stdin.read()[::-1])"}]; '
     'w29i = ProcessConnection[w29p, "StandardInput"]; WriteString[w29i, "abc"]; Close[w29i]; '
     'ReadLine[ProcessConnection[w29p, "StandardOutput"]]', '"cba"'),                       # doc: Close sends EOF
    ('w29p = StartProcess[{w29py, "-c", "import sys; sys.stderr.write(\'err1\\\\n\')"}]; '
     'ReadLine[ProcessConnection[w29p, "StandardError"]]', '"err1"'),
    ('w29p = StartProcess[{w29py, "-c", "1"}]; Head /@ {ProcessConnection[w29p, "StandardInput"], '
     'ProcessConnection[w29p, "StandardOutput"]}', '{OutputStream, InputStream}'),
    ('w29p = StartProcess[{w29py, "-c", "1"}]; ProcessConnection[w29p, "w29nope"]', Raises(ValueError)),
    ('ProcessConnection[w29x, "StandardInput"]', Pred(lambda v: _is_head(v, "ProcessConnection"),
                                                      "symbolic: unevaluated")),
    ('ProcessConnection[3, "StandardInput"]', Raises(TypeError)),
    ('w29p = StartProcess[{w29py, "-c", "import time; time.sleep(60)"}]; KillProcess[w29p]; ProcessStatus[w29p]',
     '"Finished"'),
    ('w29p = StartProcess[{w29py, "-c", "import time; time.sleep(60)"}]; KillProcess[w29p]; '
     'ProcessInformation[w29p, "ExitCode"]', Pred(lambda v: v == (1 if os.name == "nt" else 128 + 15),
                                                 "py: 128 + signal.SIGTERM")),
    ('w29p = StartProcess[{w29py, "-c", "1"}]; TimeConstrained[While[ProcessStatus[w29p] == "Running", '
     'Pause[0.02]], 20]; KillProcess[w29p]', 'Null'),                                         # finished: left alone
    ('KillProcess[ProcessObject[$ProcessID]]', Raises(ValueError)),                             # not this kernel
    ('KillProcess[w29x]', Pred(lambda v: _is_head(v, "KillProcess"), "symbolic: unevaluated")),
    ('KillProcess["x"]', Raises(TypeError)),
    ('w29p = StartProcess[{w29py, "-c", "input()"}]; MemberQ[Processes[], w29p]', 'True'),
    ('w29p = StartProcess[{w29py, "-c", "input()"}]; KillProcess[w29p]; MemberQ[Processes[], w29p]', 'False'),
    ('ListQ[Processes[]]', 'True'),
    ('Processes[1]', Raises(ValueError)),
    # SystemProcesses (py: os.getpid(); the test's own child)
    ('MemberQ[#["PID"] & /@ SystemProcesses[], $ProcessID]', 'True'),
    ('w29p = StartProcess[{w29py, "-c", "input()"}]; Select[SystemProcesses["PID" -> w29p["PID"]], '
     '#["PPID"] == $ProcessID &] // Length', '1'),
    ('First[SystemProcesses["PID" -> $ProcessID]]["ManagedProcess"]', 'False'),
    ('Length[SystemProcesses[All]] >= Length[SystemProcesses[]]', 'True'),
    ('SystemProcesses[{"PID" -> $ProcessID, "PPID" -> $ParentProcessID}] // Length', '1'),
    ('SystemProcesses["w29-no-such-program-name-xyz"]', '{}'),
    ('SystemProcesses[w29x]', Pred(lambda v: _is_head(v, "SystemProcesses"), "symbolic: unevaluated")),
    ('First[SystemProcesses["PID" -> $ProcessID]]["w29nope"]', Raises(ValueError)),
    # Run / RunThrough (through the shell; py: exit codes of the same program)
    ('Run[w29sh <> " -c \\"import sys; sys.exit(3)\\""]', '3'),
    ('Run[w29sh <> " -c \\"pass\\""]', '0'),
    ('Run[w29sh, "-c", "\\"import sys; sys.exit(4)\\""]', '4'),                                # Run[e1, e2, ...]
    ('Run[]', Raises(TypeError)),
    ('Run[w29x]', Pred(lambda v: _is_head(v, "Run"), "symbolic: unevaluated")),
    ('RunThrough[w29sh <> " -c \\"import sys; print(int(sys.stdin.read()) * 2)\\"", 21]', '42'),
    ('RunThrough[w29sh <> " -c \\"import sys; print(sys.stdin.read().strip() + \' + 1\')\\"", Unevaluated[w29y]]',
     'w29y + 1'),
    ('RunThrough[w29sh <> " -c \\"pass\\"", 1]', 'Null'),                                     # no output
    ('RunThrough[w29sh <> " -c \\"import sys; sys.exit(2)\\"", 1]', Raises(ValueError)),
    ('RunThrough[1, 2]', Raises(TypeError)),
    # SystemOpen (the w29iso fixture replaces the desktop launcher by a recorder; see test_systemopen_launch)
    ('SystemOpen["https://example.com/w29"]', 'Null'),
    ('SystemOpen[URL["https://example.com/w29"]]', 'Null'),
    ('SystemOpen[FileNameJoin[{w29dir, "sub"}]]', 'Null'),
    ('SystemOpen[FileNameJoin[{w29dir, "w29-missing.txt"}]]', Raises(ValueError)),
    ('SystemOpen[URL["https://"]]', Raises(ValueError)),
    ('SystemOpen[5]', Raises(TypeError)),
]


# ------------------------------------------------------------------------------------------ parallel kernels
# The w29iso fixture sets $DefaultParallelKernels = {2}: the first parallel command launches 2 local kernels,
# which stay up across tests (closed at the end of the module, or after a test that changed their number).
# Kernel ids are only compared with Kernels[] and process ids with $ProcessID (py: os.getpid()).
KERNELS = [
    ('CloseKernels[]; LaunchKernels[]; $KernelCount', '2'),                                                     # {2} configured
    ('ParallelEvaluate[0]; Sort[ParallelEvaluate[$KernelID]] == Sort[#["KernelID"] & /@ Kernels[]]', 'True'),
    ('ParallelEvaluate[0]; w29k = LaunchKernels[1]; {Length[w29k], $KernelCount, CloseKernels[w29k] == w29k, '
     '$KernelCount}', '{1, 3, True, 2}'),
    ('ParallelEvaluate[0]; Length[Union[ParallelEvaluate[$ProcessID]]] == $KernelCount && '
     'FreeQ[ParallelEvaluate[$ProcessID], $ProcessID]', 'True'),
    ('CloseKernels[]; Length[LaunchKernels[KernelConfiguration["Local", "KernelCount" -> 2]]]', '2'),
    ('ParallelEvaluate[0]; LaunchKernels[]', Raises(ValueError)),                                     # already running
    ('LaunchKernels["ssh://w29host"]', Raises(NotImplementedError)),                              # remote kernels
    ('LaunchKernels[-1]', Raises((ValueError, TypeError))),
    ('LaunchKernels[w29x]', Pred(lambda v: _is_head(v, "LaunchKernels"), "symbolic: unevaluated")),
    ('ParallelEvaluate[0]; CloseKernels[]; $KernelCount', '0'),
    ('ParallelEvaluate[0]; Length[CloseKernels[First[Kernels[]]]]', '1'),
    ('ParallelEvaluate[0]; CloseKernels[First[Kernels[]]["KernelID"]]; $KernelCount', '1'),
    ('CloseKernels[]; CloseKernels[]', '{}'),
    ('CloseKernels["x"]', Raises((TypeError, ValueError))),
    ('CloseKernels[w29x]', Pred(lambda v: _is_head(v, "CloseKernels"), "symbolic: unevaluated")),
    ('ParallelEvaluate[0]; w29ab = AbortKernels[]; {Length[w29ab], ParallelEvaluate[1 + 1]}', '{2, {2, 2}}'),
    ('ParallelEvaluate[0]; w29e = ParallelSubmit[Pause[30]; 1]; AbortKernels[]; WaitAll[w29e]', '$Aborted'),
    ('CloseKernels[]; AbortKernels[]', '{}'),
    ('AbortKernels[1]', Raises(ValueError)),
    ('ParallelEvaluate[0]; w29ko = First[Kernels[]]; w29ko["ProcessID"] == ParallelEvaluate[$ProcessID, w29ko]', 'True'),
    ('ParallelEvaluate[0]; w29ko = First[Kernels[]]; {w29ko["Name"], w29ko["Class"], w29ko["Running"]}',
     '{"Local", "LocalKernels", True}'),
    ('ParallelEvaluate[0]; First[Kernels[]]["Properties"]',
     '{"KernelID", "Name", "Class", "KernelCommand", "ProcessID", "Running", "StartTime"}'),
    ('ParallelEvaluate[0]; Keys[First[Kernels[]][All]] == First[Kernels[]]["Properties"]', 'True'),
    ('ParallelEvaluate[0]; First[Kernels[]]["w29nope"]', Raises(ValueError)),
    ('KernelObject[w29x]', Pred(lambda v: _is_head(v, "KernelObject"), "symbolic: inert")),
    ('KernelObject[999, "Local"]["Running"]', 'False'),                                     # not running
    ('ParallelEvaluate[0]; ParallelKernels[] == Kernels[]', 'True'),
    ('ParallelEvaluate[0]; Length[ParallelKernels["OneKernelPerMachine"]]', '1'),
    ('CloseKernels[]; ParallelKernels[]', '{}'),
    ('ParallelKernels["w29nope"]', Raises(ValueError)),
    ('ParallelEvaluate[0]; Length[Kernels[]] == $KernelCount', 'True'),
    ('CloseKernels[]; Kernels[]', '{}'),
    ('ParallelEvaluate[0]; AllTrue[Kernels[], Head[#] === KernelObject &]', 'True'),
    ('Kernels[1]', Raises(ValueError)),
    ('KernelConfiguration["Local", "KernelCount" -> 3]["KernelCount"]', '3'),
    ('KernelConfiguration[2]["KernelCount"]', '2'),
    ('KernelConfiguration["Local", "KernelCount" -> 3]["Name"]', '"Local"'),
    ('KernelConfiguration["Local", "KernelCount" -> 2]["w29nope"]', Raises(ValueError)),
    ('LaunchKernels[KernelConfiguration["ssh://w29host", "KernelCount" -> 1]]', Raises(NotImplementedError)),
    ('KernelConfiguration[w29x]', Pred(lambda v: _is_head(v, "KernelConfiguration"), "symbolic: inert")),
    ('CloseKernels[]; Length[LaunchKernels[{KernelConfiguration[1], KernelConfiguration[1]}]]', '2'),
    ('$KernelID', '0'),                                                                             # doc: master 0
    ('ParallelEvaluate[0]; Min[ParallelEvaluate[$KernelID]] > 0', 'True'),
    ('$KernelID = 1', Raises(ValueError)),
    ('CloseKernels[]; $KernelCount', '0'),
    ('ParallelEvaluate[0]; ParallelEvaluate[$KernelCount]', '{2, 2}'),                                  # doc: master's
    ('$KernelCount = 3', Raises(ValueError)),
    ('$DefaultParallelKernels', '{2}'),
    ('CloseKernels[]; Block[{$DefaultParallelKernels = {1}}, Length[LaunchKernels[]]]', '1'),
    ('Block[{$DefaultParallelKernels = {3}}, 1]; $DefaultParallelKernels', '{2}'),
    ('CloseKernels[]; Block[{$DefaultParallelKernels = {}}, ParallelEvaluate[$KernelID]]', '{0}'),  # no kernels
    ('$ConfiguredKernels', '{2}'),
    ('Block[{$DefaultParallelKernels = {1}}, $ConfiguredKernels]', '{1}'),
    ('Block[{$ConfiguredKernels = {3}}, $ConfiguredKernels]', '{3}'),
    ('$DefaultLocalKernel', '"Local"'),                                                            # doc: "Local"
    ('Block[{$DefaultLocalKernel = "w29"}, $DefaultLocalKernel]', '"w29"'),
    ('StringQ[$DefaultLocalKernel]', 'True'),
    ('$DefaultRemoteKernel', '"localhost"'),
    ('Block[{$DefaultRemoteKernel = "w29"}, $DefaultRemoteKernel]', '"w29"'),
    ('StringQ[$DefaultRemoteKernel]', 'True'),
    ('$DistributedContexts', '"Global`"'),                                                    # doc: $Context
    ('Block[{$DistributedContexts = None}, w29dx[x_] := x; ParallelMap[Length[DownValues[w29dx]] + 0 # &, {1, 2}]]',
     '{0, 0}'),
    ('Block[{$DistributedContexts = All}, w29dy[x_] := x; ParallelMap[Length[DownValues[w29dy]] + 0 # &, {1, 2}]]',
     '{1, 1}'),
    ('$EvaluationEnvironment', '"Session"'),
    ('ParallelEvaluate[0]; Union[ParallelEvaluate[$EvaluationEnvironment]]', '{"Subkernel"}'),
]

# ------------------------------------------------------------------------- parallel commands (doc: same results)
PARALLEL = [
    ('ParallelEvaluate[1 + 1]', '{2, 2}'),
    ('ParallelEvaluate[0]; ParallelEvaluate[$KernelID, First[Kernels[]]] == First[Kernels[]]["KernelID"]', 'True'),
    ('ParallelEvaluate[0]; Length[ParallelEvaluate[$ProcessID, Kernels[]]]', '2'),
    ('ParallelEvaluate[0]; ParallelEvaluate[1 + 1, Kernels[], Hold]', '{Hold[2], Hold[2]}'),                           # doc: h argument
    ('w29f[x_] := x^2; ParallelEvaluate[w29f[3]]', '{9, 9}'),                                     # auto-distributed
    ('ParallelEvaluate[0]; w29g[x_] := x + 1; ParallelEvaluate[w29g[1], Kernels[], Hold, DistributedContexts -> None]',
     '{Hold[w29g[1]], Hold[w29g[1]]}'),                                                      # not distributed
    ('ParallelEvaluate[1, w29x]', Pred(lambda v: _is_head(v, "ParallelEvaluate"), "symbolic: unevaluated")),
    ('ParallelEvaluate[Exit[]]', Raises(RuntimeError)),
    ('ParallelEvaluate[1, "w29nope"]', Raises((TypeError, ValueError))),
    ('ParallelEvaluate[1, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('ParallelMap[#^2 &, Range[10]] == Map[#^2 &, Range[10]]', 'True'),
    ('ParallelMap[#^2 &, {1, 2, 3}]', '{1, 4, 9}'),                                              # doc example
    ('w29h[x_] := 2 x; ParallelMap[w29h, f[1, 2, 3]]', 'f[2, 4, 6]'),
    ('ParallelMap[#^2 &, <|"a" -> 1, "b" -> 2|>]', '<|"a" -> 1, "b" -> 4|>'),
    ('ParallelMap[w29u, {1, 2}]', '{w29u[1], w29u[2]}'),                                          # symbolic f
    ('ParallelMap[f, {{1, 2}, {3}}, {2}]', '{{f[1], f[2]}, {f[3]}}'),
    ('ParallelMap[#^2 &, Range[7], Method -> "FinestGrained"] == Range[7]^2', 'True'),
    ('ParallelMap[#^2 &, Range[7], Method -> "CoarsestGrained"] == Range[7]^2', 'True'),
    ('ParallelMap[#^2 &, Range[7], Method -> "EvaluationsPerKernel" -> 3] == Range[7]^2', 'True'),
    ('ParallelMap[#^2 &, Range[7], ProgressReporting -> True] == Range[7]^2', 'True'),
    ('ParallelMap[#^2 &, Range[3], Method -> "w29nope"]', Raises(ValueError)),
    ('ParallelMap[f, 5] == Map[f, 5]', 'True'),                                              # atom: unchanged
    ('ParallelTable[i^2, {i, 10}] == Table[i^2, {i, 10}]', 'True'),
    ('ParallelTable[i + j, {i, 2}, {j, 3}]', '{{2, 3, 4}, {3, 4, 5}}'),
    ('ParallelTable[$KernelID > 0, {4}]', '{True, True, True, True}'),
    ('ParallelTable[i, {i, w29n}]', Raises(ValueError)),                                   # as Table: symbolic bound
    ('ParallelTable[i, {i, {a, b, c}}]', '{a, b, c}'),
    ('ParallelTable[i, {i, 1, 10, 3}, Method -> "FinestGrained"]', '{1, 4, 7, 10}'),
    ('ParallelTable[i, {i, 3}, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('ParallelSum[i^2, {i, 100}] == Sum[i^2, {i, 100}]', 'True'),
    ('ParallelSum[i^2, {i, 100}]', '338350'),                                                    # n(n+1)(2n+1)/6
    ('ParallelSum[i j, {i, 3}, {j, 3}]', '36'),
    ('ParallelSum[1/i^2, {i, Infinity}]', 'Pi^2/6'),                                               # doc: falls back
    ('ParallelSum[x^i, {i, 3}]', 'x + x^2 + x^3'),
    ('ParallelSum[i, {i, 3}, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('ParallelProduct[i, {i, 10}] == Product[i, {i, 10}]', 'True'),
    ('ParallelProduct[i, {i, 10}]', '3628800'),                                                   # 10!
    ('ParallelProduct[x + i, {i, 2}]', '(x + 1) (x + 2)'),
    ('ParallelProduct[i, {i, n}]', 'n!'),
    ('ParallelProduct[i, {i, 3}, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('w29acc = {}; SetSharedVariable[w29acc]; ParallelDo[CriticalSection[{w29lk}, AppendTo[w29acc, i]], {i, 6}]; '
     'w29r = Sort[w29acc]; UnsetShared[w29acc]; w29r', '{1, 2, 3, 4, 5, 6}'),
    ('ParallelDo[i, {i, 3}]', 'Null'),
    ('w29n = 0; SetSharedVariable[w29n]; ParallelDo[CriticalSection[{w29lk}, w29n += i], {i, 4}, {j, 2}]; '
     'w29r = w29n; UnsetShared[w29n]; w29r', '20'),
    ('ParallelDo[i, {i, 3}, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('ParallelArray[#^2 &, 5] == Array[#^2 &, 5]', 'True'),
    ('ParallelArray[f, {2, 2}]', '{{f[1, 1], f[1, 2]}, {f[2, 1], f[2, 2]}}'),                   # doc example
    ('ParallelArray[f, 3, 0]', '{f[0], f[1], f[2]}'),
    ('ParallelArray[f, {2, 2}, 1, g]', 'g[g[f[1, 1], f[1, 2]], g[f[2, 1], f[2, 2]]]'),
    ('ParallelArray[f, -1]', Raises((TypeError, ValueError))),
    ('ParallelCases[Range[20], _?PrimeQ] == Cases[Range[20], _?PrimeQ]', 'True'),
    ('ParallelCases[{1, a, 2, b}, _Integer]', '{1, 2}'),
    ('ParallelCases[{1, 2, 3, 4}, x_ /; EvenQ[x] -> x^2]', '{4, 16}'),
    ('ParallelCases[Range[20], _?PrimeQ, {1}, 3]', '{2, 3, 5}'),
    ('ParallelCases[{{1}, {2}}, _Integer, Infinity]', '{1, 2}'),
    ('ParallelCases[5, _Integer] == Cases[5, _Integer]', 'True'),                          # atom: {}
    ('ParallelSelect[Range[20], PrimeQ] == Select[Range[20], PrimeQ]', 'True'),
    ('ParallelSelect[{1, 2, 3, 4}, EvenQ]', '{2, 4}'),
    ('ParallelSelect[Range[20], PrimeQ, 2]', '{2, 3}'),
    ('ParallelSelect[f[1, 2, 3], OddQ]', 'f[1, 3]'),
    ('ParallelSelect[5, EvenQ]', Raises((TypeError, ValueError))),
    ('ParallelCombine[Map[#^2 &, #] &, Range[6]] == Map[#^2 &, Range[6]]', 'True'),               # doc
    ('ParallelCombine[Apply[Plus, #] &, {1, 2, 3, 4}, Plus]', '10'),
    ('ParallelCombine[Map[f, #] &, {1, 2, 3}, Join]', '{f[1], f[2], f[3]}'),
    ('ParallelCombine[Total, Range[6], Plus]', '21'),                                            # comb argument
    ('ParallelCombine[f, 5]', 'f[5]'),                                                       # atom: f[expr]
    ('ParallelTry[If[# == 3, #, $Failed] &, Range[5]]', '3'),                                     # one candidate
    ('ParallelTry[$Failed &, {1, 2}]', '$Failed'),                                                # doc: all fail
    ('Sort[ParallelTry[If[OddQ[#], #, $Failed] &, {1, 2, 3, 4, 5}, 2]] // Length', '2'),
    ('SubsetQ[{1, 3, 5}, ParallelTry[If[OddQ[#], #, $Failed] &, {1, 2, 3, 4, 5}, 3]]', 'True'),
    ('ParallelTry[f, 5]', Raises((TypeError, ValueError))),
    ('Parallelize[Map[#^2 &, Range[5]]]', '{1, 4, 9, 16, 25}'),
    ('Parallelize[Table[i^2, {i, 5}]] == Table[i^2, {i, 5}]', 'True'),
    ('Parallelize[Sum[i, {i, 100}]]', '5050'),
    ('Parallelize[Select[Range[10], EvenQ]]', '{2, 4, 6, 8, 10}'),
    ('Parallelize[Count[Range[10], _?PrimeQ]]', '4'),
    ('Parallelize[1 + 1]', '2'),                                                                  # not parallel
    ('Parallelize[Sin[Range[3]]] == Sin[Range[3]]', 'True'),                                     # Listable
    ('Parallelize[Map[f, {1, 2}], ProgressReporting -> False]', '{f[1], f[2]}'),
    ('Parallelize[Map[f, {1, 2}], w29opt -> 1]', Raises((TypeError, ValueError))),
    ('Parallelize[MapThread[Plus, {{1, 2}, {3, 4}}]]', '{4, 6}'),
    ('Parallelize[Apply[Plus, {{1, 2}, {3, 4}}, {1}]]', '{3, 7}'),
    # ProgressReporting (doc: option; results unchanged)
    ('ParallelTable[i, {i, 3}, ProgressReporting -> True]', '{1, 2, 3}'),
    ('ParallelSum[i, {i, 4}, ProgressReporting -> Automatic]', '10'),
    ('Parallelize[Map[f, {1}], ProgressReporting -> True]', '{f[1]}'),
    # DistributedContexts (doc: option)
    ('w29dc[x_] := -x; ParallelMap[Length[DownValues[w29dc]] + 0 # &, {1, 2}, DistributedContexts -> None]',
     '{0, 0}'),                                                                              # not on the kernels
    ('w29dc1[x_] := -x; ParallelMap[Length[DownValues[w29dc1]] + 0 # &, {1, 2}]', '{1, 1}'),     # default: Global`
    ('w29dc2[x_] := -x; ParallelMap[w29dc2, {1, 2}, DistributedContexts -> All]', '{-1, -2}'),
    ('w29dc3[x_] := -x; ParallelMap[w29dc3, {1, 2}, DistributedContexts -> "Global`"]', '{-1, -2}'),
    ('ParallelMap[f, {1}, DistributedContexts -> 5]', Raises((TypeError, ValueError))),
]


# ------------------------------------------------------------------- ParallelSubmit / WaitAll / WaitNext
SUBMIT = [
    ('WaitAll[{ParallelSubmit[1 + 1], ParallelSubmit[2 + 2]}]', '{2, 4}'),                       # doc example
    ('WaitAll[ParallelSubmit[10] + ParallelSubmit[5]]', '15'),                                     # anywhere in expr
    ('w29v = 3; WaitAll[ParallelSubmit[{w29v}, w29v^2]]', '9'),                                     # doc: {vars}
    ('Head[ParallelSubmit[1 + 1]]', 'EvaluationObject'),
    ('ParallelSubmit[1, 2, 3]', Raises(ValueError)),
    ('WaitAll[ParallelSubmit[$KernelID]] > 0', 'True'),
    ('WaitAll[{}]', '{}'),
    ('WaitAll[w29x]', 'w29x'),                                                                     # nothing to wait
    ('WaitAll[3]', '3'),
    ('w29e = ParallelSubmit[1 + 1]; WaitAll[w29e]; w29e["Result"]', 'HoldComplete[2]'),
    ('w29e = ParallelSubmit[1 + 1]; w29e["Process"]', 'HoldForm[1 + 1]'),
    ('w29e = ParallelSubmit[1 + 1]; WaitAll[w29e]; w29e["State"]', '"finished"'),
    ('w29e = ParallelSubmit[1 + 1]; w29e["Properties"]', Pred(lambda v: _strlist(v) and "Result" in v.args,
                                                               "property names")),
    ('w29e = ParallelSubmit[1]; w29e["w29nope"]', Raises(ValueError)),
    ('Head[EvaluationObject[1, 1 + 1]]', 'EvaluationObject'),                                   # inert
    ('w29e = ParallelSubmit[3]; WaitAll[w29e]; Head[w29e] === EvaluationObject', 'True'),
    ('EvaluationObject[w29x]', Pred(lambda v: _is_head(v, "EvaluationObject"), "symbolic: inert")),
    ('w29es = Table[ParallelSubmit[{i}, i^2], {i, 4}]; w29r = {}; While[w29es =!= {}, {w29a, w29b, w29es} = '
     'WaitNext[w29es]; AppendTo[w29r, w29a]]; Sort[w29r]', '{1, 4, 9, 16}'),                       # doc: loop
    ('w29e = ParallelSubmit[7]; WaitNext[{w29e}]', Pred(lambda v: _is_list(v, 3) and v.args[0] == 7
                                                        and v.args[2] == Expr("List"), "{7, eo, {}}")),
    ('w29e = ParallelSubmit[7]; w29f = ParallelSubmit[8]; Length[Last[WaitNext[{w29e, w29f}]]]', '1'),
    ('WaitNext[{}]', Raises(TypeError)),
    ('WaitNext[5]', Raises(TypeError)),
    ('WaitNext[w29x]', Pred(lambda v: _is_head(v, "WaitNext"), "symbolic: unevaluated")),
]

# --------------------------------------------- definitions, sharing, locks, LocalEvaluate (Check: exact)
SHARING = [
    ('w29d[x_] := x + 100; DistributeDefinitions[w29d]', '{HoldForm[w29d]}'),
    ('w29d2[x_] := x + 100; DistributeDefinitions[w29d2]; ParallelEvaluate[w29d2[1], '
     'DistributedContexts -> None]', '{101, 101}'),
    ('w29d3 = 5; w29d4[x_] := x + w29d3; DistributeDefinitions[w29d4]; MemberQ[$DistributedDefinitions, '
     'HoldForm[w29d3]]', 'True'),                                                                    # dependencies
    ('DistributeDefinitions["w29ctx`"]', '{}'),
    ('w29ctx`y = 7; DistributeDefinitions["w29ctx`"]; ParallelEvaluate[w29ctx`y, '
     'DistributedContexts -> None]', '{7, 7}'),
    ('DistributeDefinitions[1]', Raises(TypeError)),
    ('w29c[x_] := x; DistributeDefinitions[w29c]; ClearDistributedDefinitions[w29c]', '{HoldForm[w29c]}'),
    ('w29c2[x_] := -x; DistributeDefinitions[w29c2]; ClearDistributedDefinitions[]; '
     '{MemberQ[$DistributedDefinitions, HoldForm[w29c2]], ParallelEvaluate[Length[DownValues[w29c2]], '
     'DistributedContexts -> None], w29c2[1]}', '{False, {0, 0}, -1}'),
    ('ClearDistributedDefinitions[]; ClearDistributedDefinitions[]', '{}'),
    ('ClearDistributedDefinitions[1]', Raises(TypeError)),
    ('ClearDistributedDefinitions[]; $DistributedDefinitions', '{}'),
    ('w29dd = 1; DistributeDefinitions[w29dd]; $DistributedDefinitions', '{HoldForm[w29dd]}'),
    ('$DistributedDefinitions = {}', Raises(ValueError)),
    # SetSharedVariable / SetSharedFunction / UnsetShared / $SharedVariables / $SharedFunctions
    ('w29sv = 0; SetSharedVariable[w29sv]; ParallelEvaluate[w29sv = $KernelID]; w29r = MemberQ[#["KernelID"] & /@ '
     'Kernels[], w29sv]; UnsetShared[w29sv]; w29r', 'True'),
    ('SetSharedVariable[w29sv2]', 'Null'),
    ('SetSharedVariable[w29sv3]; ParallelEvaluate[w29sv3]', '{Null, Null}'),                       # doc: no value
    ('SetSharedVariable[1]', Raises(TypeError)),
    ('SetSharedVariable[w29sv4]; $SharedVariables', '{Hold[w29sv4]}'),
    ('SetSharedVariable[w29sv5]; UnsetShared[w29sv5]; $SharedVariables', '{}'),
    ('$SharedVariables', '{}'),
    ('$SharedVariables = {}', Raises(ValueError)),
    ('w29sf[x_] := $ProcessID; SetSharedFunction[w29sf]; w29r = Union[ParallelEvaluate[w29sf[1]]]; '
     'UnsetShared[w29sf]; w29r == {$ProcessID}', 'True'),                                          # master's pid
    ('SetSharedFunction[w29sf2]; ParallelEvaluate[w29sf2[$KernelID] = 1]; w29r = Length[DownValues[w29sf2]]; '
     'UnsetShared[w29sf2]; w29r', '2'),                                                             # stored on master
    ('SetSharedFunction[w29sf3]; ParallelEvaluate[w29sf3[1]]', '{Null, Null}'),                    # doc: no rule
    ('SetSharedFunction["x"]', Raises(TypeError)),
    ('SetSharedFunction[w29sf4]; $SharedFunctions', '{Hold[w29sf4]}'),
    ('SetSharedFunction[w29sf5]; UnsetShared[All]; $SharedFunctions', '{}'),
    ('$SharedFunctions', '{}'),
    ('$SharedFunctions = {}', Raises(ValueError)),
    ('SetSharedVariable[w29us]; UnsetShared[w29us]', '{HoldForm[w29us]}'),
    ('UnsetShared[w29us2]', '{}'),
    ('SetSharedVariable[w29us3]; SetSharedFunction[w29us4]; Length[UnsetShared[All]]', '2'),
    ('UnsetShared[1]', Raises(TypeError)),
    # CriticalSection / WithLock
    ('w29cs = 0; SetSharedVariable[w29cs]; ParallelDo[CriticalSection[{w29lock}, w29cs = w29cs + 1], {20}]; '
     'w29r = w29cs; UnsetShared[w29cs]; w29r', '20'),                                              # no lost update
    ('CriticalSection[{w29lock}, 1 + 1]', '2'),
    ('CriticalSection[w29lock, 3]', '3'),
    ('Catch[CriticalSection[{w29lock}, Throw[1]]]; CriticalSection[{w29lock}, 4]', '4'),           # released
    ('CriticalSection[{1}, 1]', Raises(TypeError)),
    ('WithLock[File[FileNameJoin[{w29dir, "w29.lock"}]], 1 + 1]', '2'),
    ('WithLock[FileNameJoin[{w29dir, "w29.lock"}], FileExistsQ[FileNameJoin[{w29dir, "w29.lock"}]]]', 'True'),
    ('w29wl = {}; SetSharedVariable[w29wl]; ParallelDo[WithLock[{w29lock2}, w29wl = Append[w29wl, i]], {i, 6}]; '
     'w29r = Sort[w29wl]; UnsetShared[w29wl]; w29r', '{1, 2, 3, 4, 5, 6}'),
    ('WithLock[5, 1]', Raises(TypeError)),
    ('WithLock[File[FileNameJoin[{w29dir, "nodir", "x", "w29.lock"}]], 1]', Raises((ValueError, OSError))),
    # ParallelNeeds (doc: Needs on every kernel; Wolfram packages are not available here)
    ('ParallelEvaluate[0]; ParallelNeeds["w29ctxNoSuchPackage`"]', Raises(ValueError)),         # Needs fails
    ('ParallelNeeds["w29NotAContext"]', Raises(ValueError)),
    ('ParallelNeeds[5]', Raises(TypeError)),
    # (packages are found in the working directory: the kernels change to w29dir and back)
    ('ParallelEvaluate[SetDirectory[w29dir]; 0]; ParallelNeeds["W29Pkg`"]; ParallelEvaluate[ResetDirectory[]; 0]; '
     'ParallelEvaluate[MemberQ[$Packages, "W29Pkg`"], DistributedContexts -> None]', '{True, True}'),
    ('ParallelEvaluate[SetDirectory[w29dir]; 0]; w29r = ParallelNeeds["W29Pkg`"]; '
     'ParallelEvaluate[ResetDirectory[]; 0]; w29r', 'Null'),
    # LocalEvaluate (a new local kernel per call)
    ('LocalEvaluate[$ProcessID] != $ProcessID', 'True'),
    ('w29le[x_] := 3 x; LocalEvaluate[w29le[2]]', '6'),                                            # definitions sent
    ('LocalEvaluate["Local", 1 + 1]', '2'),
    ('Length[LocalEvaluate[{"Local", "Local"}, 1]]', '2'),
    ('LocalEvaluate["ssh://w29host", 1]', Raises(NotImplementedError)),
    ('LocalEvaluate[$EvaluationEnvironment]', Pred(lambda v: isinstance(v, str) and v != "Session",
                                                   "not the session")),
]

# ------------------------------------------------------------------------------------------------- tasks
TASKS = [
    ('w29t = SessionSubmit[1 + 1]; TaskWait[w29t]; w29t["EvaluationResult"]', '2'),               # doc
    ('w29t = SessionSubmit[1 + 1]; Head[w29t]', 'TaskObject'),
    ('w29t = SessionSubmit[$CurrentTask["TaskUUID"]]; TaskWait[w29t]; '
     'w29t["EvaluationResult"] == w29t["TaskUUID"]', 'True'),                                      # $CurrentTask
    ('w29n = 0; w29t = SessionSubmit[ScheduledTask[w29n++, {0.01, 3}]]; TaskWait[w29t]; w29n', '3'),
    ('w29t = SessionSubmit[2, HandlerFunctions -> <|"TaskFinished" -> ((w29hd = #EvaluationResult) &)|>, '
     'HandlerFunctionsKeys -> {"EvaluationResult"}]; TaskWait[w29t]; Pause[0.05]; w29hd', '2'),
    ('SessionSubmit[1, HandlerFunctions -> <|"PrintOutputGenerated" -> Print|>]', Raises(NotImplementedError)),
    ('SessionSubmit[ScheduledTask[1, DateObject[{2030, 1, 1}]]]', Raises(NotImplementedError)),
    ('SessionSubmit[1, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('w29t = LocalSubmit[$ProcessID]; TaskWait[w29t]; w29t["EvaluationResult"] != $ProcessID', 'True'),
    ('w29lf[x_] := x + 1; w29t = LocalSubmit[w29lf[41]]; TaskWait[w29t]; w29t["EvaluationResult"]', '42'),
    ('w29t = LocalSubmit[1]; w29t["TaskEnvironment"]', '"Local"'),
    ('LocalSubmit[1, w29opt -> 1]', Raises((TypeError, ValueError))),
    ('Head[ScheduledTask[1, 5]]', 'ScheduledTask'),                                                 # inert
    ('w29t = SessionSubmit[ScheduledTask[1, {0.01, 2}]]; TaskWait[w29t]; w29t["PreviousRunCount"]', '2'),
    ('w29t = SessionSubmit[ScheduledTask[1, None]]; w29t["TaskStatus"]', Pred(
        lambda v: v in ("Waiting", "Running"), "on demand: waiting")),
    ('SessionSubmit[ScheduledTask[1, "w29nope"]]', Raises(NotImplementedError)),            # a cron string
    ('SessionSubmit[ScheduledTask[1, -1]]', Raises(ValueError)),
    ('w29t = SessionSubmit[1, AutoRemove -> True]; TaskWait[w29t]; Pause[0.05]; w29t["TaskStatus"]', '"Removed"'),
    ('w29t = SessionSubmit[1, AutoRemove -> False]; TaskWait[w29t]; w29t["TaskStatus"]', '"Finished"'),
    ('w29t = SessionSubmit[ScheduledTask[1, {0.01, 2}, AutoRemove -> True]]; TaskWait[w29t]; Pause[0.05]; '
     'MemberQ[Tasks[], w29t]', 'False'),
    ('SessionSubmit[1, AutoRemove -> 5]', Raises((TypeError, ValueError))),
    ('w29t = SessionSubmit[Pause[0.2]; 5]; TaskWait[w29t]; w29t["EvaluationResult"]', '5'),
    ('w29t = SessionSubmit[Pause[5]; 5]; w29t0 = AbsoluteTime[]; TaskWait[w29t, TimeConstraint -> 0.1]; '
     'w29r = AbsoluteTime[] - w29t0 < 4; TaskAbort[w29t]; w29r', 'True'),
    ('w29t = SessionSubmit[ScheduledTask[1, None]]; Head[TaskWait[w29t]]', 'TaskObject'),          # returns at once
    ('Length[TaskWait[{SessionSubmit[1], SessionSubmit[2]}]]', '2'),                                # Listable
    ('TaskWait[5]', Raises(TypeError)),
    ('TaskWait["w29-no-such-uuid"]', Raises(ValueError)),
    ('TaskWait[w29x]', Pred(lambda v: _is_head(v, "TaskWait"), "symbolic: unevaluated")),
    ('w29t = SessionSubmit[Pause[30]; 1]; Pause[0.1]; TaskAbort[w29t]; TaskWait[w29t, TimeConstraint -> 10]; '
     'w29t["EvaluationResult"]', '$Aborted'),
    ('w29t = SessionSubmit[1]; TaskWait[w29t]; TaskAbort[w29t]; w29t["EvaluationResult"]', '1'),     # not running
    ('TaskAbort[5]', Raises(TypeError)),
    ('w29t = SessionSubmit[1]; TaskWait[w29t]; TaskRemove[w29t]; MemberQ[Tasks[], w29t]', 'False'),
    ('w29t = SessionSubmit[ScheduledTask[1, 30]]; TaskRemove[w29t]; w29t["TaskStatus"]', '"Removed"'),
    ('w29t = SessionSubmit[1]; Head[TaskRemove[w29t]]', 'TaskObject'),
    ('TaskRemove[5]', Raises(TypeError)),
    ('w29n = 0; w29t = SessionSubmit[ScheduledTask[w29n++, 0.02]]; TaskSuspend[w29t]; Pause[0.1]; w29m = w29n; '
     'Pause[0.15]; {w29t["TaskStatus"], w29n == w29m}', '{"Suspended", True}'),
    ('w29t = SessionSubmit[1]; TaskWait[w29t]; TaskSuspend[w29t]; w29t["TaskStatus"]', '"Finished"'),
    ('w29t = SessionSubmit[ScheduledTask[1, 30]]; Head[TaskSuspend[w29t]]', 'TaskObject'),
    ('TaskSuspend[5]', Raises(TypeError)),
    ('w29t = SessionSubmit[ScheduledTask[1, 30]]; TaskSuspend[w29t]; TaskResume[w29t]; w29t["TaskStatus"]',
     Pred(lambda v: v in ("Waiting", "Running"), "not suspended")),
    ('w29n = 0; w29t = SessionSubmit[ScheduledTask[w29n++, {0.02, 3}]]; TaskSuspend[w29t]; TaskResume[w29t]; '
     'TaskWait[w29t]; w29n', '3'),
    ('w29t = SessionSubmit[1]; TaskWait[w29t]; Head[TaskResume[w29t]]', 'TaskObject'),
    ('TaskResume[5]', Raises(TypeError)),
    ('w29t = SessionSubmit[ScheduledTask[6 * 7, None]]; TaskExecute[w29t]; w29t["EvaluationResult"]', '42'),
    ('w29n = 0; w29t = SessionSubmit[ScheduledTask[PreIncrement[w29n], None]]; TaskExecute[w29t]; TaskExecute[w29t]; w29n', '2'),
    ('w29t = SessionSubmit[ScheduledTask[1, None]]; TaskRemove[w29t]; TaskExecute[w29t]', Raises(ValueError)),
    ('TaskExecute[5]', Raises(TypeError)),
    ('w29t = SessionSubmit[1]; MemberQ[Tasks[], w29t]', 'True'),
    ('Tasks[]', '{}'),
    ('w29t = LocalSubmit[1]; {MemberQ[Tasks["Local"], w29t], MemberQ[Tasks["Session"], w29t]}', '{True, False}'),
    ('Tasks["w29nope"]', Raises(ValueError)),
    ('w29t = SessionSubmit[1]; TaskWait[w29t]; w29t["TaskStatus"]', '"Finished"'),
    ('w29t = SessionSubmit[1 + 1]; w29t["EvaluationExpression"]', 'HoldForm[1 + 1]'),
    ('w29t = SessionSubmit[1]; w29t["TaskType"]', '"Asynchronous"'),
    ('w29t = SessionSubmit[1]; w29t["w29nope"]', Raises(ValueError)),
    ('Head[SessionSubmit[1]] === TaskObject', 'True'),
    ('SessionSubmit[1]; SessionSubmit[2]; AllTrue[Tasks[], Head[#] === TaskObject &]', 'True'),
    ('TaskObject[w29x]', Pred(lambda v: _is_head(v, "TaskObject"), "symbolic: inert")),
    ('$CurrentTask', 'None'),                                                                          # doc: outside
    ('w29t = SessionSubmit[Head[$CurrentTask]]; TaskWait[w29t]; w29t["EvaluationResult"]', 'TaskObject'),
    ('$CurrentTask["TaskUUID"]', Raises(ValueError)),
    ('$CurrentTask = 1', Raises(ValueError)),
]


CASES = (SYSVARS + ENVIRONMENT + RANDOM + SYSINFO + EXIT + INPUT + OTHER + PROC + KERNELS + PARALLEL + SUBMIT
         + SHARING + TASKS)

PKG = ('BeginPackage["W29Pkg`"]\nw29pkgf::usage = "w29pkgf[x] adds 1000"\nBegin["`Private`"]\n'
       'w29pkgf[x_] := x + 1000\nEnd[]\nEndPackage[]\n')


def _wl_string(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _w29_names():
    """Every symbol name with a definition in the default session that a test may have created (w29 prefix)."""
    from mikoshilang import kernel as K
    st = K.state()
    names = set()
    for v in vars(st).values():
        if isinstance(v, dict):
            for k in list(v.keys()):
                if isinstance(k, str):
                    names.add(k)
                elif isinstance(k, tuple) and k and isinstance(k[0], str):
                    names.add(k[0])
            for d in list(v.values()):
                if isinstance(d, dict):
                    names.update(k for k in list(d.keys()) if isinstance(k, str))
    return sorted(n for n in names if n.split("`")[-1].startswith("w29") or n.startswith("w29"))


def _clear_w29():
    for n in _w29_names():
        try:
            ev("ClearAll[%s]" % n)
        except Exception:  # noqa: BLE001 - best-effort cleanup of test symbols
            pass


@pytest.fixture(scope="module", autouse=True)
def w29module():
    """Close the parallel kernels started by this module at its end (they are reused between tests)."""
    from mikoshilang.library import wl_system as W
    before = set(W._KERNELS)
    yield
    ev("Unset[$DefaultParallelKernels]")
    for kid in list(W._KERNELS):
        if kid not in before:
            ev("CloseKernels[%d]" % kid)
    assert set(W._KERNELS) == before, "parallel kernels left running"
    assert not [p for p in W._PROCS.values() if p.running()], "child processes left running"
    assert not [t for t in W._ALL_TASKS.values() if t.thread is not None and t.thread.is_alive()], \
        "task threads left running"


@pytest.fixture(autouse=True)
def w29iso(monkeypatch, tmp_path_factory):
    """Per-test isolation: environment, cwd, stdin, SystemOptions, desktop launcher, tasks, processes, sharing."""
    import copy
    import shutil
    import tempfile
    from mikoshilang.library import wl_system as W
    env = dict(os.environ)
    cwd = os.getcwd()
    sysopts = copy.deepcopy(W._SYS_OPTIONS)
    needs = list(W._NEEDS)
    procs = set(W._PROCS)
    tasks = set(W._ALL_TASKS)
    d = os.path.realpath(tempfile.mkdtemp(prefix="w29-"))
    os.makedirs(os.path.join(d, "sub"))
    with open(os.path.join(d, "W29Pkg.m"), "w") as fh:
        fh.write(PKG)
    launched = []
    monkeypatch.setattr(W, "_LAUNCH", launched.append)                 # SystemOpen never opens anything
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    ev("w29py = %s" % _wl_string(sys.executable))
    ev("w29sh = %s" % _wl_string('"%s"' % sys.executable if os.name == "nt" else
                                 __import__("shlex").quote(sys.executable)))
    ev("w29dir = %s" % _wl_string(d))
    ev("$DefaultParallelKernels = {2}")
    try:
        yield launched
    finally:
        for uuid in list(W._ALL_TASKS):
            if uuid not in tasks:
                t = W._ALL_TASKS[uuid]
                try:
                    t.remove()
                    th = getattr(t, "thread", None)
                    if th is not None:
                        th.join(5)
                except Exception:  # noqa: BLE001 - cleanup
                    pass
        for uid in list(W._PROCS):
            if uid not in procs:
                W._PROCS[uid].kill()                                        # only processes this test started
        if W._QUEUE or W._RUNNING or W._EVALS:
            if W._KERNELS:
                ev("AbortKernels[]")
            W._QUEUE.clear()
            W._EVALS.clear()
        ev("UnsetShared[All]")
        ev("ClearDistributedDefinitions[]")
        W._NEEDS[:] = needs
        if len(W._KERNELS) not in (0, 2) or not all(k.alive for k in W._KERNELS.values()):
            ev("CloseKernels[]")
        _clear_w29()
        ev("Unset[$DefaultParallelKernels]")
        W._SYS_OPTIONS.clear()
        W._SYS_OPTIONS.update(sysopts)
        os.environ.clear()
        os.environ.update(env)
        os.chdir(cwd)
        shutil.rmtree(d, ignore_errors=True)
        assert not W._LOCK_OWNERS, "a lock was left held"
        assert all(t.thread is None or not t.thread.is_alive() for u, t in W._ALL_TASKS.items() if u not in tasks), \
            "a task thread is still running"
        assert len(W._KERNELS) <= 2, "more than 2 parallel kernels left running"


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_w29(case):
    check(case)


# ------------------------------------------------------------------------------------- main-loop and I/O tests

def test_systemopen_launch(w29iso):
    """SystemOpen hands the target to the desktop opener (doc: SystemOpen); the launcher is replaced here."""
    ev('SystemOpen["https://example.com/w29"]')
    ev('SystemOpen[FileNameJoin[{w29dir, "sub"}]]')
    assert len(w29iso) == 2
    assert w29iso[0][-1] == "https://example.com/w29"
    assert os.path.samefile(w29iso[1][-1], os.path.join(ev("w29dir"), "sub"))
    if sys.platform.startswith("linux"):
        assert w29iso[0][0] == "xdg-open"


def test_input_reads_stdin(monkeypatch, capsys):
    """Input continues over lines until the expression is complete and evaluates it; InputString reads one line;
    the prompts go to $Urgent; EndOfFile at end of input (doc: Input, InputString)."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("1 +\n 2\nw29 text here\n\n"))
    assert ev('Input["w29> "]') == 3
    assert ev("InputString[]") == "w29 text here"
    assert ev("InputString[]") == ""
    assert ev("Input[]") == Symbol("EndOfFile")
    assert capsys.readouterr().out == "w29> "


def test_input_syntax_error(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("1 +* 2\n"))
    with pytest.raises(ValueError, match="Input"):
        ev("Input[]")


def test_syntax_handler_and_echo():
    """$SyntaxHandler[text, pos] gets the 1-based position of the offending character and its string result is
    used as the input (doc: $SyntaxHandler); $Failed abandons the input; $Echo copies each input line to a
    stream (doc: $Echo with OpenWrite)."""
    from mikoshilang import Session
    from mikoshilang.parser import ParseError
    s = Session()
    s.evaluate_input('$SyntaxHandler = Function[{t, p}, w29seen = {t, p}; "1 + 1"]')
    assert s.evaluate_input("2 +* 3")[-1][1] == 2
    assert s.evaluate_input("w29seen")[-1][1] == ev('{"2 +* 3", 4}')        # '*' is character 4
    s.evaluate_input("f[1,]]")
    assert s.evaluate_input("w29seen")[-1][1] == ev('{"f[1,]]", 5}')
    s.evaluate_input("$SyntaxHandler = ($Failed &)")
    assert s.evaluate_input("1 +* 2")[-1][1] == Symbol("$Failed")
    s.evaluate_input("Unset[$SyntaxHandler]")
    with pytest.raises(ParseError):
        s.evaluate_input("1 +* 2")
    d = ev("w29dir")
    p = os.path.join(d, "echo.txt")
    s.evaluate_input("w29st = OpenWrite[%s]" % _wl_string(p))
    s.evaluate_input("$Echo = {w29st}")
    assert s.evaluate_input("1 + 5")[-1][1] == 6
    s.evaluate_input("$Echo = {}")
    s.evaluate_input("Close[w29st]")
    with open(p) as fh:
        assert fh.read() == "1 + 5\n$Echo = {}\n"                      # lines read while $Echo was set


def test_input_variable_in_get():
    """$Input is the name of the file being read by Get, and "" afterwards (doc: $Input)."""
    d = ev("w29dir")
    p = os.path.join(d, "w29in.m")
    with open(p, "w") as fh:
        fh.write("$Input\n")
    assert os.path.samefile(ev("Get[%s]" % _wl_string(p)), p) or ev("Get[%s]" % _wl_string(p)).endswith("w29in.m")
    assert ev("$Input") == ""


def test_output_channels(capsys):
    """Print writes to the $Output channels (doc: $Output)."""
    ev('Block[{$Output = {OutputStream["stderr", 2]}}, Print["w29 to stderr"]]')
    ev('Block[{$Output = {}}, Print["w29 nowhere"]]')
    out = capsys.readouterr()
    assert out.err == "w29 to stderr\n" and "w29 nowhere" not in out.out


def test_exit_code_and_epilog(capsys):
    """Exit[n] evaluates $Epilog and ends the session with exit code n (doc: Exit, $Epilog)."""
    from mikoshilang import Session
    s = Session()
    s.evaluate_input('$Epilog := Print["w29 bye"]')
    with pytest.raises(SystemExit) as ex:
        s.evaluate_input("Exit[4]")
    assert ex.value.code == 4
    assert capsys.readouterr().out == "w29 bye\n"
    with pytest.raises(SystemExit) as ex:
        ev("Quit[]")
    assert ex.value.code == 0


class _TTY(io.StringIO):
    def isatty(self):
        return True


def test_repl_ignore_eof(monkeypatch, capsys):
    """With $IgnoreEOF = True an interactive REPL ignores end-of-file (doc: $IgnoreEOF); otherwise it ends."""
    from mikoshilang import repl
    lines = iter(["$IgnoreEOF = True", EOFError, "quit"])

    def fake_input(prompt=""):
        x = next(lines)
        if x is EOFError:
            raise EOFError
        return x
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys, "stdin", _TTY(""))
    repl.main()
    out = capsys.readouterr().out
    assert "end-of-file ignored" in out and out.rstrip().endswith("Goodbye!")
    lines2 = iter([EOFError])
    monkeypatch.setattr("builtins.input", lambda prompt="": (_ for _ in ()).throw(next(lines2)))
    repl.main()
    assert "end-of-file ignored" not in capsys.readouterr().out


def test_parallel_capped_and_cleaned():
    """Parallel commands in this module use at most 2 kernels, and the kernels are this process's children."""
    from mikoshilang.library import wl_system as W
    ev("ParallelEvaluate[1]")
    assert 1 <= len(W._KERNELS) <= 2
    pids = ev("ParallelEvaluate[$ParentProcessID]")
    assert all(p == PID for p in pids.args)


def test_input_hook_installed():
    from mikoshilang.library import wl_evaluation, wl_system
    assert wl_evaluation.INPUT_HOOK is wl_system._input_hook


def test_skipped_symbols_are_not_registered():
    """Symbols left out of W29 (BRIEF step 9) have a reason and no implementation."""
    from mikoshilang.library import wl_system as W
    from mikoshilang.library.core import REGISTRY, STATUS
    assert len(W._SKIPS) == 9          # X01 implemented KernelEvaluate, KernelFunction, Parallelization
    for name, why in W._SKIPS.items():
        assert why and name not in STATUS and name not in REGISTRY, name
