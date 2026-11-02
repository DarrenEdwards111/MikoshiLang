"""Wolfram batch W29: system & session ($-variables, environment, processes, tasks and parallel computing).

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input (see
CONVENTIONS.md).

* System information: ``$Version``/``$VersionNumber``/``$ReleaseNumber`` report MikoshiLang's own version;
  ``$System``/``$SystemID``/``$OperatingSystem``/``$ProcessorType`` come from Python's ``platform``/``sys``;
  process ids, user and machine names from ``os``/``getpass``/``socket``.  Read-only variables are Protected.
* Environment: ``Environment``, ``GetEnvironment``, ``SetEnvironment`` act on ``os.environ`` (inherited by every
  child process).
* Random state: ``BlockRandom`` saves and restores every generator the library uses (the shared Mersenne
  Twister behind SeedRandom/RandomReal/RandomInteger/RandomVariate, Python's global ``random`` and NumPy's
  global state); ``$RandomGeneratorState`` exposes the Mersenne Twister state as a ``RandomGeneratorState``.
* External processes: ``Run``, ``RunThrough``, ``RunProcess``, ``StartProcess`` with ``ProcessObject[pid]``,
  ``ProcessStatus``, ``ProcessInformation``, ``ProcessConnection`` (W27 streams), ``KillProcess``,
  ``Processes``, ``SystemProcesses``, ``SystemOpen``.
* Parallel computing: subkernels are local MikoshiLang processes (``sys.executable``) talking to the master
  over pipes.  Definitions are distributed as in Wolfram (DistributedContexts, DistributeDefinitions),
  shared variables/functions and critical sections are served by the master, ParallelSubmit queues
  evaluations that WaitAll/WaitNext run.
* Tasks: ``SessionSubmit`` runs a task in a background thread of the current session (sharing its
  definitions, like Wolfram's preemptive session tasks); ``LocalSubmit`` runs it in a separate local kernel.
"""
from __future__ import annotations

import atexit
import collections
import getpass
import hashlib
import io
import itertools
import os
import pickle
import platform
import queue
import random as _pyrandom
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid as _uuid
from contextvars import ContextVar, copy_context

from ..expr import Expr, Symbol
import importlib as _il
E = _il.import_module("mikoshilang.evaluate")
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated, REGISTRY, STATUS
from .wl_common import L, ev, is_sym, is_list, is_rule, is_assoc

AREA = "system, processes & parallel computing (W29)"

NULL = None
NONE = Symbol("None")
FAILED = Symbol("$Failed")
ABORTED = Symbol("$Aborted")
TIMEDOUT = Symbol("$TimedOut")
AUTOMATIC = Symbol("Automatic")
INHERITED = Symbol("Inherited")
ALL = Symbol("All")


def _form(x):
    from .wl_errors import input_text
    try:
        return input_text(x)
    except Exception:  # noqa: BLE001 - only used inside error messages
        return repr(x)


def _out_text(x):
    from .wl_errors import output_text
    return output_text(x)


def assoc(pairs):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in pairs))


def _assoc_dict(a, fname, what="association"):
    """Plain dict of an Association (string keys kept as str)."""
    if not is_assoc(a):
        raise TypeError(f"{fname}: {what} must be an association, not {_form(a)}")
    out = {}
    for r in a.args:
        out[r.args[0]] = r.args[1]
    return out


def _rules(pairs):
    return L(*(Expr("Rule", k, v) for k, v in pairs))


def _is_symbolic(x):
    return isinstance(x, Symbol) and not x.name.startswith("$") or (
        isinstance(x, Expr) and not is_list(x) and not is_assoc(x) and x.head not in (
            "Rule", "RuleDelayed", "File", "URL", "ProcessObject", "KernelObject", "TaskObject",
            "EvaluationObject", "KernelConfiguration", "RandomGeneratorState", "Quantity", "LocalSymbol",
            "InputStream", "OutputStream"))


def _opt_name(r):
    k = r.args[0]
    if isinstance(k, Symbol):
        return k.name
    if isinstance(k, str):
        return k
    return None


def _opts(args, allowed, fname, unsupported=()):
    """Split trailing options; returns (positional, {name: value}).  Unknown option names raise ValueError,
    names in ``unsupported`` raise NotImplementedError."""
    args = list(args)
    o = {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") and \
            len(args[-1].args) == 2 and isinstance(args[-1].args[0], Symbol):
        r = args.pop()
        n = r.args[0].name
        if n in unsupported:
            raise NotImplementedError(f"{fname}: the option {n} is not supported")
        if n not in allowed:
            raise ValueError(f"{fname}: {n} is not a known option (known: {', '.join(allowed) or 'none'})")
        v = r.args[1]
        if r.head == "RuleDelayed":
            v = ev(v)
        o.setdefault(n, v)
    return args, o


def _string(x, fname, what="argument"):
    if isinstance(x, str):
        return x
    if _is_symbolic(x):
        raise Unevaluated()
    raise TypeError(f"{fname}: {what} must be a string, not {_form(x)}")


def _nonneg_real(x, fname, what="time"):
    """Seconds from a non-negative real or Quantity[t, unit]; None for Infinity."""
    if isinstance(x, Expr) and x.head == "Quantity" and len(x.args) == 2 and isinstance(x.args[1], str):
        scale = {"Seconds": 1, "Milliseconds": 1e-3, "Microseconds": 1e-6, "Minutes": 60, "Hours": 3600,
                 "Days": 86400}.get(x.args[1])
        if scale is None:
            raise ValueError(f"{fname}: {x.args[1]!r} is not a time unit")
        return _nonneg_real(x.args[0], fname, what) * scale
    if is_sym(x, "Infinity"):
        return None
    if isinstance(x, bool) or isinstance(x, str):
        raise TypeError(f"{fname}: the {what} must be a non-negative number of seconds, not {_form(x)}")
    try:
        v = K.to_sp(x)
        if not v.is_number:
            raise Unevaluated()
        f = float(v)
    except Unevaluated:
        raise
    except Exception:  # noqa: BLE001 - not a number
        raise TypeError(f"{fname}: the {what} must be a non-negative number of seconds, not {_form(x)}") from None
    if f < 0 or f != f:
        raise ValueError(f"{fname}: the {what} must be non-negative, not {_form(x)}")
    return f


def _nonneg_int(x, fname, what="argument"):
    if isinstance(x, bool) or not K.is_int(x):
        if _is_symbolic(x):
            raise Unevaluated()
        raise TypeError(f"{fname}: {what} must be a non-negative integer, not {_form(x)}")
    v = int(x)
    if v < 0:
        raise ValueError(f"{fname}: {what} must be non-negative, not {v}")
    return v


def _pos_int(x, fname, what="argument"):
    v = _nonneg_int(x, fname, what)
    if v == 0:
        raise ValueError(f"{fname}: {what} must be positive, not 0")
    return v


def _hold_sym_name(x, fname):
    """Name of a symbol given unevaluated (HoldAll heads)."""
    if isinstance(x, Symbol):
        return x.name
    raise TypeError(f"{fname}: {_form(x)} is not a symbol")


# ═══════════════════════════════════════════ system $-variables ═══════════════════════════════════════════

_PROTECTED_VARS = set()


def _sysvar(name, getter, notes, protected=True, check="reference (Python platform/os/sys values) in tests"):
    E.SYSTEM_VARIABLES[name] = getter
    record(name, AREA, "added", f"{name}: {notes}." + ("" if protected else " A user assignment (or Block) "
                                                          "takes precedence over the default.") + f" Check: {check}.")
    if protected:
        _PROTECTED_VARS.add(name)
        K.BUILTIN_ATTRIBUTES[name] = {"Protected"}


def _version_string():
    from .. import __version__
    return str(__version__)


def _version_parts():
    import re
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", _version_string())
    if not m:
        return 0, 0, 0
    return int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)


def _os_name():
    if sys.platform.startswith("win"):
        return "Windows"
    if sys.platform == "darwin":
        return "MacOSX"
    return "Unix"


def _machine():
    m = (platform.machine() or "").lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86-64"
    if m in ("i386", "i486", "i586", "i686", "x86"):
        return "x86"
    if m in ("arm64", "aarch64", "armv8l"):
        return "ARM64"
    if m.startswith("arm"):
        return "ARM"
    if m.startswith("ppc") or m.startswith("powerpc"):
        return "POWER"
    return m or "Unknown"


def _word_length():
    return 64 if sys.maxsize > 2 ** 32 else 32


def _system():
    arch = _machine()
    fam = "ARM" if arch.startswith("ARM") else "x86" if arch.startswith("x86") else arch
    bits = f"({_word_length()}-bit)"
    osn = _os_name()
    if osn == "Windows":
        return f"Microsoft Windows {bits}"
    if osn == "MacOSX":
        return f"Mac OS X {fam} {bits}"
    plat = "Linux" if sys.platform.startswith("linux") else platform.system() or "Unix"
    return f"{plat} {fam} {bits}"


def _system_id():
    osn = _os_name()
    arch = _machine()
    head = {"Windows": "Windows", "MacOSX": "MacOSX"}.get(osn, "Linux" if sys.platform.startswith("linux")
                                                          else (platform.system() or "Unix"))
    if arch == "x86":
        return head if head != "Linux" else "Linux"
    return f"{head}-{arch}"


def _creation_time():
    """When this copy of MikoshiLang was created: the modification time of the installed package __init__."""
    from .. import __file__ as pkg_init
    try:
        return os.path.getmtime(pkg_init)
    except OSError:
        return time.time()


def _date_object(t):
    """DateObject (local time zone) for a Unix time."""
    lt = time.localtime(t)
    off = -(time.altzone if lt.tm_isdst > 0 and time.daylight else time.timezone) / 3600.0
    secs = lt.tm_sec + (t - int(t))
    return ev(Expr("DateObject", L(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, float(secs)),
                   "Instant", "Gregorian", float(off)))


def _creation_date():
    return _date_object(_creation_time())


_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


def _version():
    lt = time.localtime(_creation_time())
    return f"{_version_string()} for {_system()} ({_MONTHS[lt.tm_mon - 1]} {lt.tm_mday}, {lt.tm_year})"


def _processor_count():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _machine_name():
    try:
        return socket.gethostname().split(".")[0]
    except OSError:
        return ""


def _machine_domains():
    try:
        fq = socket.getfqdn()
    except OSError:
        return L()
    host = _machine_name()
    if "." in fq and fq.split(".")[0] == host:
        return L(fq.split(".", 1)[1])
    return L()


def _machine_id_raw():
    for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
        try:
            with open(p) as fh:
                s = fh.read().strip()
            if s:
                return s
        except OSError:
            pass
    return f"{_uuid.getnode():012x}"


def _machine_id():
    h = int(hashlib.sha256(_machine_id_raw().encode()).hexdigest(), 16)
    d = str(h)[:14].rjust(14, "0")
    return f"{d[:4]}-{d[4:9]}-{d[9:14]}"


_SESSION_ID = int(hashlib.sha256(f"{_machine_id_raw()}|{os.getpid()}|{time.time_ns()}".encode()).hexdigest()[:15],
                  16)


def _username():
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name available
        return NONE


def _command_line():
    argv = getattr(sys, "orig_argv", None)
    if not argv:
        argv = [sys.executable] + list(sys.argv[1:])
    return L(*(str(a) for a in argv))


def _system_shell():
    if _os_name() == "Windows":
        return os.environ.get("COMSPEC", "cmd.exe")
    return "/bin/sh"


def _isatty(f):
    try:
        return bool(f is not None and f.isatty())
    except (AttributeError, ValueError, OSError):
        return False


# Python codec names for the Wolfram character encoding names this library reads and writes.
CHARACTER_ENCODINGS = {
    "ASCII": "ascii", "PrintableASCII": "ascii", "UTF8": "utf-8", "UTF-8": "utf-8", "UTF-16": "utf-16",
    "UTF-16BE": "utf-16-be", "UTF-16LE": "utf-16-le", "UTF-32": "utf-32", "UTF-32BE": "utf-32-be",
    "UTF-32LE": "utf-32-le", "ISOLatin1": "latin-1", "ISO8859-1": "latin-1", "ISOLatin2": "iso8859-2",
    "ISO8859-2": "iso8859-2", "ISOLatin3": "iso8859-3", "ISO8859-3": "iso8859-3", "ISOLatin4": "iso8859-4",
    "ISO8859-4": "iso8859-4", "ISOLatinCyrillic": "iso8859-5", "ISO8859-5": "iso8859-5", "ISO8859-6": "iso8859-6",
    "ISO8859-7": "iso8859-7", "ISO8859-8": "iso8859-8", "ISO8859-9": "iso8859-9", "ISO8859-10": "iso8859-10",
    "ISO8859-11": "iso8859-11", "ISO8859-13": "iso8859-13", "ISO8859-14": "iso8859-14", "ISO8859-15": "iso8859-15",
    "ISO8859-16": "iso8859-16", "WindowsANSI": "cp1252", "WindowsBaltic": "cp1257", "WindowsCyrillic": "cp1251",
    "WindowsEastEurope": "cp1250", "WindowsGreek": "cp1253", "WindowsThai": "cp874", "WindowsTurkish": "cp1254",
    "CP936": "cp936", "CP949": "cp949", "CP950": "cp950", "EUC-JP": "euc_jp", "EUC": "euc_jp",
    "ShiftJIS": "shift_jis", "IBM-850": "cp850", "KOI8-R": "koi8_r", "MacintoshRoman": "mac_roman",
    "MacintoshCyrillic": "mac_cyrillic", "MacintoshGreek": "mac_greek", "MacintoshTurkish": "mac_turkish",
    "MacintoshIcelandic": "mac_iceland", "MacintoshCroatian": "mac_croatian", "MacintoshRomanian": "mac_romanian",
    "GB2312": "gb2312", "Big5": "big5",
}


def _character_encodings():
    import codecs
    out = []
    for k, c in CHARACTER_ENCODINGS.items():
        try:
            codecs.lookup(c)
            out.append(k)
        except LookupError:
            pass
    return L(*sorted(out, key=str.lower))


_sysvar("$Version", _version, "MikoshiLang's version string: \"<version> for <$System> (<creation date>)\"")
_sysvar("$VersionNumber", lambda: float(f"{_version_parts()[0]}.{_version_parts()[1]}"),
        "MikoshiLang's version number major.minor as a machine real (3.8 for 3.8.0)")
_sysvar("$ReleaseNumber", lambda: _version_parts()[2], "the release (patch) number of the MikoshiLang version, "
        "the last integer of major.minor.release")
_sysvar("$CreationDate", _creation_date, "a DateObject for when this copy of MikoshiLang was created (the "
        "modification time of the installed package; there is no separate build date)")
_sysvar("$OperatingSystem", _os_name, "\"Unix\" (Linux and other Unix systems), \"MacOSX\" or \"Windows\"")
_sysvar("$System", _system, "a description such as \"Linux x86 (64-bit)\", \"Mac OS X ARM (64-bit)\" or "
        "\"Microsoft Windows (64-bit)\"")
_sysvar("$SystemID", _system_id, "a short id such as \"Linux-x86-64\", \"MacOSX-ARM64\" or \"Windows-x86-64\"")
_sysvar("$ProcessorType", _machine, "the processor architecture: \"x86-64\", \"x86\", \"ARM64\", \"ARM\" (from "
        "platform.machine())")
_sysvar("$MachineType", lambda: "PC", "the general machine family; \"PC\" on every current system, as in Wolfram")
_sysvar("$SystemWordLength", _word_length, "the bit length of machine pointers: 64 or 32 (sys.maxsize)")
_sysvar("$ProcessorCount", _processor_count, "the number of processor cores available to this process "
        "(os.sched_getaffinity, else os.cpu_count); Protected, but it may be reset after Unprotect to change the "
        "default number of parallel kernels")
_sysvar("$ProcessID", os.getpid, "the operating-system process id of this kernel (os.getpid())")
_sysvar("$ParentProcessID", os.getppid, "the process id of the process that started this kernel (os.getppid())")
_sysvar("$MachineName", _machine_name, "the host name without its domain (socket.gethostname()); \"\" if none")
_sysvar("$MachineDomains", _machine_domains, "the network domains of the host: the domain part of "
        "socket.getfqdn(), or {} if no domain is defined")
_sysvar("$MachineID", _machine_id, "a stable identification code for this computer in the MathID layout "
        "dddd-ddddd-ddddd, derived from /etc/machine-id (or the network hardware address); it is not a "
        "Wolfram license MathID")
_sysvar("$SessionID", lambda: _SESSION_ID, "a number unique to this session, derived from $MachineID, the process "
        "id and the start time")
_sysvar("$Username", _username, "the login name of the user running the kernel (getpass.getuser()), None if "
        "there is none")
_sysvar("$CommandLine", _command_line, "the command line that started this Python process, as a list of strings "
        "(sys.orig_argv)")
_sysvar("$ScriptCommandLine", lambda: L(), "{}: MikoshiLang is not running from a wolframscript-style script "
        "launcher, so (as documented for that case) the list is empty",
        check='exact (the documented {} outside a script launcher; a child Python process reports no script arguments either) in tests')
_sysvar("$ScriptInputString", lambda: NONE, "None: no script launcher supplied standard input (documented value "
        "outside wolframscript)",
        check='exact (the documented None outside a script launcher) in tests')
_sysvar("$SystemShell", _system_shell, "the system shell: \"/bin/sh\" on Unix and macOS, %COMSPEC% (cmd.exe) "
        "on Windows; Run uses it")
_sysvar("$CharacterEncodings", _character_encodings, "the character encoding names known to this installation "
        "(those with a Python codec), sorted case-insensitively",
        check='reference (every entry has a Python codec: codecs.lookup) in tests')
_sysvar("$FontFamilies", lambda: L(), "{}: there is no notebook front end, and Wolfram documents $FontFamilies as "
        "typically {} without one",
        check='exact (the documented {} without a front end) in tests')
_sysvar("$BatchOutput", lambda: False, "False: MikoshiLang has no batch-output mode (InputForm output without "
        "labels); it cannot be reset during a session",
        check='exact (False; Protected) in tests')
_sysvar("$BatchInput", lambda: not _isatty(sys.stdin), "True when standard input is not an interactive terminal "
        "(input from a file or pipe), False in an interactive terminal", protected=False,
        check='reference (not sys.stdin.isatty()) in tests')
_sysvar("$IgnoreEOF", lambda: False, "False: the REPL ends at end-of-file; set it to True to make an interactive "
        "REPL ignore end-of-file (it is always False for non-terminal input, as documented)", protected=False,
        check='exact (default False; Block assignment) in tests')
_sysvar("$Language", lambda: "English", "\"English\", the language of system messages (only English message "
        "texts exist)", protected=False,
        check='exact (default "English"; Block assignment) in tests')
_sysvar("$Input", lambda: (_INPUT_NAMES[-1] if _INPUT_NAMES else ""), "the name of the stream input is read from: "
        "the file name given to Get (<<) while it reads, and \"\" otherwise", protected=False,
        check='exact (inside Get it is the file name, "" afterwards) in tests')
_sysvar("$Output", lambda: L(Expr("OutputStream", "stdout", 1)), "the channels Print writes to: "
        "{OutputStream[\"stdout\", 1]}; assign {} to silence Print, or a list of stdout/stderr streams",
        protected=False,
        check='exact (Print output reaches the assigned channel; {} silences it) in tests')
_sysvar("$Urgent", lambda: L(Expr("OutputStream", "stdout", 1)), "the channels for urgent output (the prompts "
        "of Input and InputString): {OutputStream[\"stdout\", 1]}", protected=False,
        check='exact (Input/InputString prompts are written to it) in tests')
_sysvar("$Echo", lambda: L(), "the streams every main-loop input (Session.evaluate_input, the REPL) is echoed "
        "to: {} by default; OutputStream[\"stdout\", 1], OutputStream[\"stderr\", 2] or W27 output streams (files) "
        "may be assigned", protected=False,
        check='exact (Session.evaluate_input echoes the input line to a W27 file stream) in tests')
_sysvar("$Display", lambda: L(), "{}: the files and pipes for a user-defined $DisplayFunction (initial setting {} "
        "as documented)", protected=False,
        check='exact (the documented initial {}) in tests')
_sysvar("$ProgressReporting", lambda: False, "False: no progress panels are shown in this text interface; True "
        "or Automatic may be assigned (functions with a ProgressReporting option then warn that progress "
        "reporting is unavailable)", protected=False,
        check='exact (default False; results of functions are unchanged when set) in tests')
_sysvar("$EvaluationEnvironment", lambda: _EVAL_ENV.get(), "\"Session\" in the main session, \"Subkernel\" in a "
        "parallel subkernel",
        check='exact ("Session" on the master, "Subkernel" in ParallelEvaluate) in tests')


_INPUT_NAMES = []
_EVAL_ENV = ContextVar("w29_eval_env", default="Session")

for _n, _d in (("$NoValue", "inert symbol: assigned to InitializationValue[...] to mean \"initialise with no "
                            "value\"; it evaluates to itself"),
               ("$InitializationContexts", "has no value by default (evaluates to itself); MikoshiLang reads no "
                                           "packages at start-up, so an assignment only stores the list")):
    record(_n, AREA, "added", f"{_n}: {_d}. Check: exact (evaluates to itself) in tests.")


def _wrap_get():
    """Keep $Input set to the file name while Get (<<) reads a file (chains any earlier wrapper)."""
    from . import wl_evaluation as WE
    orig = WE._get_file
    if getattr(orig, "_w29", False):
        return

    def _get_file(path):
        _INPUT_NAMES.append(str(path))
        try:
            return orig(path)
        finally:
            _INPUT_NAMES.pop()
    _get_file._w29 = True
    for k, v in vars(orig).items():
        setattr(_get_file, k, v)
    WE._get_file = _get_file


_wrap_get()


# ══════════════════════════════════════ main-loop hooks: $Echo, $SyntaxHandler ══════════════════════════════════════

def _own(name):
    return K.state().own.get(name)


def _echo(text):
    """Write an input line to the $Echo streams (stdout/stderr or W27 output streams)."""
    v = _own("$Echo")
    if v is None:
        return
    for s in (v.args if is_list(v) else (v,)):
        if isinstance(s, Expr) and s.head == "OutputStream" and s.args and s.args[0] in ("stdout", "stderr"):
            f = sys.stdout if s.args[0] == "stdout" else sys.stderr
            f.write(text + "\n")
            f.flush()
        else:
            ev(Expr("WriteString", s, text + "\n"))


def _input_hook(text, parse):
    """Parse one main-loop input: echo it to $Echo, and on a syntax error apply $SyntaxHandler[text, pos]."""
    from ..parser import ParseError
    _echo(text)
    try:
        return parse(text)
    except ParseError as err:
        h = _own("$SyntaxHandler")
        if h is None:
            raise
        pos = _error_offset(text, parse) + 1          # 1-based position of the offending character
        new = ev(Expr(K.as_head(h), text, pos))
        if isinstance(new, str):
            return parse(new)
        if new is FAILED or is_sym(new, "$Failed"):
            return FAILED
        raise ValueError(f"$SyntaxHandler: the handler must return a string or $Failed, not {_form(new)} "
                         f"(syntax error: {err})") from None


def _error_offset(text, parse):
    """Character offset of a syntax error: the length of the longest prefix of text that is still a valid start
    of an input (it parses, or only fails for ending too early)."""
    from ..parser import ParseError
    best = 0
    for i in range(1, len(text) + 1):
        pre = text[:i]
        try:
            parse(pre)
        except ParseError as e:
            msg = str(e)
            if not ("end of input" in msg or "got EOF" in msg or "Unsupported character" in msg and
                    ('"' in pre or "(*" in pre)):
                break
        best = i
    return best


def _install_input_hook():
    from . import wl_evaluation as WE
    prev = getattr(WE, "INPUT_HOOK", None)
    if prev is None:
        WE.INPUT_HOOK = _input_hook


_install_input_hook()
_sysvar("$SyntaxHandler", lambda: Symbol("$SyntaxHandler"),
        "unset by default; when assigned a function f, a main-loop input (Session.evaluate_input / the REPL) "
        "with a syntax error is replaced by f[\"text\", pos] where pos is the (1-based) position of the character "
        "where the error was detected: "
        "a string result is parsed instead, $Failed abandons the input", protected=False,
        check="exact (handler called with the text and offset, replacement parsed) in tests")
E.SYSTEM_VARIABLES.pop("$SyntaxHandler", None)     # no value unless assigned (evaluates to itself)


# ═══════════════════════════════════════════ environment variables ═══════════════════════════════════════════

def _env_name(x, fname):
    s = _string(x, fname, "the variable name")
    if not s or "=" in s or "\0" in s:
        raise ValueError(f"{fname}: {s!r} is not a valid environment variable name")
    return s


@fn("Environment", area=AREA, status="added",
    notes="Environment[\"var\"] gives the value of an operating-system environment variable (os.environ), or "
          "$Failed if it is not set (documented result). Check: reference (os.environ) in tests.")
def _environment(var: String) -> String | Expr:
    s = _env_name(var, "Environment")
    v = os.environ.get(s)
    return FAILED if v is None else v


def _get_env_one(x):
    s = _env_name(x, "GetEnvironment")
    v = os.environ.get(s)
    return Expr("Rule", s, NONE if v is None else v)


@fn("GetEnvironment", area=AREA, status="added",
    notes="GetEnvironment[] gives every environment variable as \"var\" -> \"value\" rules (in os.environ order); "
          "GetEnvironment[\"var\"] gives \"var\" -> value, with None for an unset variable; GetEnvironment[{v1, "
          "...}] gives a list of rules. Check: reference (os.environ) in tests.")
def _get_environment(var: String | ListOf[String] = None) -> Rule | ListOf[Rule]:
    if var is None:
        return L(*(Expr("Rule", k, v) for k, v in os.environ.items()))
    if is_list(var):
        return L(*(_get_env_one(x) for x in var.args))
    return _get_env_one(var)


def _set_env_rule(r, fname):
    if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2):
        if _is_symbolic(r):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_form(r)} is not a rule \"var\" -> \"value\"")
    k = _env_name(r.args[0], fname)
    v = r.args[1]
    if is_sym(v, "None") or v is NONE:
        return k, None
    if isinstance(v, str):
        if "\0" in v:
            raise ValueError(f"{fname}: an environment value cannot contain a NUL character")
        return k, v
    if isinstance(v, bool):
        raise TypeError(f"{fname}: the value of {k} must be a string or None, not {_form(v)}")
    if K.is_int(v):
        return k, str(int(v))
    raise TypeError(f"{fname}: the value of {k} must be a string or None, not {_form(v)}")


@fn("SetEnvironment", area=AREA, status="added",
    notes="SetEnvironment[\"var\" -> \"value\"] sets an environment variable of this process (inherited by Run, "
          "RunProcess and StartProcess); \"var\" -> None unsets it; a list or association sets several. "
          "Integer values are converted to strings. Returns Null. Check: reference (os.environ) in tests.")
def _set_environment(spec: Rule | ListOf[Rule] | Assoc) -> Expr:
    if is_list(spec) or is_assoc(spec):
        pairs = [_set_env_rule(r, "SetEnvironment") for r in spec.args]
    else:
        pairs = [_set_env_rule(spec, "SetEnvironment")]
    for k, v in pairs:
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return NULL


# ═══════════════════════════════════════════ random generator state ═══════════════════════════════════════════

def _rng():
    from .statistics import _RNG
    return _RNG


def _np_random():
    try:
        import numpy
        return numpy.random
    except ImportError:  # pragma: no cover - numpy is a dependency
        return None


def _state_hash(st):
    h = hashlib.sha256(repr(st).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


def _rgs(st=None):
    """RandomGeneratorState[{"MersenneTwister", {}}, <|"Version" -> v, "State" -> {...}, "Gauss" -> g|>]."""
    if st is None:
        st = _rng().getstate()
    ver, ints, gauss = st
    return Expr("RandomGeneratorState", L("MersenneTwister", L()),
                assoc([("Version", ver), ("State", L(*ints)),
                       ("Gauss", NONE if gauss is None else float(gauss))]))


def _rgs_state(x, fname):
    if not (isinstance(x, Expr) and x.head == "RandomGeneratorState" and len(x.args) == 2):
        raise TypeError(f"{fname}: {_form(x)} is not a RandomGeneratorState")
    meth, data = x.args
    if not (is_list(meth) and meth.args and meth.args[0] == "MersenneTwister"):
        raise NotImplementedError(f"{fname}: only \"MersenneTwister\" generator states are supported, not "
                                  f"{_form(meth)}")
    d = _assoc_dict(data, fname, "the generator state")
    try:
        ints = tuple(int(i) for i in d["State"].args)
        g = d.get("Gauss", NONE)
        st = (int(d["Version"]), ints, None if is_sym(g, "None") else float(g))
        _pyrandom.Random().setstate(st)
    except (KeyError, AttributeError, TypeError, ValueError):
        raise ValueError(f"{fname}: the generator state data is malformed") from None
    return st


def _rgs_props_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "RandomGeneratorState" and len(args) == 1
            and isinstance(args[0], str)):
        return K.UNHANDLED
    st = _rgs_state(head, "RandomGeneratorState")
    p = args[0]
    if p == "Method":
        return head.args[0]
    if p == "StateHash":
        return _state_hash(st)
    if p == "CrossSessionPersistent":
        return True
    if p == "Properties":
        return L("CrossSessionPersistent", "Method", "StateHash")
    raise ValueError(f"RandomGeneratorState: unknown property {p!r} (known: Method, StateHash, "
                     "CrossSessionPersistent)")


KL.COMPOUND_HEAD_HOOKS["RandomGeneratorState"] = _rgs_props_hook
record("RandomGeneratorState", AREA, "added",
       "RandomGeneratorState: the state of the library's Mersenne Twister (the generator behind SeedRandom, "
       "RandomReal, RandomInteger, RandomChoice, RandomVariate, ...), as RandomGeneratorState[{\"MersenneTwister\","
       " {}}, <|\"Version\", \"State\", \"Gauss\"|>]; it is a value of $RandomGeneratorState, a RandomSeeding "
       "setting of BlockRandom (restoring the state), and has the properties \"Method\", \"StateHash\" and "
       "\"CrossSessionPersistent\" (True: the state reproduces the same numbers in any session). Check: exact "
       "(restoring a state reproduces the same random numbers) in tests.")
_sysvar("$RandomGeneratorState", lambda: _rgs(), "the current state of the library's pseudorandom generator as a "
        "RandomGeneratorState; it changes whenever random numbers are generated and is reset by SeedRandom",
        check="exact (BlockRandom[..., RandomSeeding -> $RandomGeneratorState] repeats the numbers) in tests")


def _seed_value(s, fname):
    """random.Random seed for a RandomSeeding value; None = keep the inherited state."""
    if is_sym(s, "Inherited"):
        return ("inherit",)
    if is_sym(s, "Automatic"):
        return ("seed", int.from_bytes(os.urandom(16), "big"))
    if isinstance(s, bool):
        raise TypeError(f"{fname}: RandomSeeding must be Automatic, Inherited, an integer, a string or a "
                        f"RandomGeneratorState, not {_form(s)}")
    if K.is_int(s):
        return ("seed", int(s))
    if isinstance(s, str):
        return ("seed", s)
    if isinstance(s, Expr) and s.head == "RandomGeneratorState":
        return ("state", _rgs_state(s, fname))
    raise TypeError(f"{fname}: RandomSeeding must be Automatic, Inherited, an integer, a string or a "
                    f"RandomGeneratorState, not {_form(s)}")


def _apply_seed(spec):
    r = _rng()
    if spec[0] == "seed":
        r.seed(spec[1])
        _pyrandom.seed(spec[1])
        npr = _np_random()
        if npr is not None:
            npr.seed(int(hashlib.sha256(repr(spec[1]).encode()).hexdigest()[:8], 16))
    elif spec[0] == "state":
        r.setstate(spec[1])


@fn("BlockRandom", area=AREA, status="added",
    notes="BlockRandom[expr] evaluates expr with all pseudorandom generators localized: the library's Mersenne "
          "Twister (SeedRandom, RandomReal, RandomInteger, RandomChoice, RandomVariate, ...), Python's global "
          "random and NumPy's global generator are saved before and restored after, even on error or abort, so "
          "random numbers generated afterwards are the same as if expr had not been evaluated. Option "
          "RandomSeeding -> Inherited (default: start from the current state), Automatic (a fresh random seed), "
          "an integer or string seed, or a RandomGeneratorState. HoldFirst. Check: exact (the numbers after "
          "BlockRandom equal those generated without it) in tests.")
def _block_random(expr: Held, *opts: Option) -> Expr:
    _, o = _opts(opts, ("RandomSeeding",), "BlockRandom")
    spec = _seed_value(o.get("RandomSeeding", INHERITED), "BlockRandom")
    r = _rng()
    saved = r.getstate()
    saved_py = _pyrandom.getstate()
    npr = _np_random()
    saved_np = npr.get_state() if npr is not None else None
    try:
        _apply_seed(spec)
        return ev(expr)
    finally:
        r.setstate(saved)
        _pyrandom.setstate(saved_py)
        if npr is not None:
            npr.set_state(saved_np)


K.BUILTIN_ATTRIBUTES["BlockRandom"] = {"HoldFirst", "Protected"}
record("RandomSeeding", AREA, "added",
       "RandomSeeding: option of BlockRandom (and of the W29 random-localizing functions) giving the seed used "
       "inside: Inherited (the current state), Automatic (a new random seed), an integer or string seed, or a "
       "RandomGeneratorState. Check: exact (equal seeds give equal numbers) in tests.")


# ═══════════════════════════════════════════ Input, InputString ═══════════════════════════════════════════

def _prompt(text):
    from .wl_errors import _streams
    for c in _streams("$Urgent", L(Expr("OutputStream", "stdout", 1))):
        s = sys.stdout if c == "stdout" else sys.stderr
        s.write(text)
        s.flush()


def _read_line():
    line = sys.stdin.readline()
    if line == "":
        return None
    return line[:-1] if line.endswith("\n") else line


def _input_args(args, fname):
    pos, _ = _opts(args, (), fname)
    if len(pos) > 3:
        raise TypeError(f"{fname}: called with {len(pos)} arguments; between 0 and 3 are expected")
    prompt = pos[0] if pos else ""
    if not isinstance(prompt, str):
        prompt = _out_text(prompt)
    return prompt


@fn("InputString", area=AREA, status="added",
    notes="InputString[] / InputString[prompt] writes the prompt to $Urgent and reads one line of text from "
          "standard input, returned as a string without its newline; EndOfFile at end of input. The init and "
          "window-option arguments of the notebook dialog are accepted and ignored (there is no front end). "
          "Check: exact (stdin replaced by known text) in tests.")
def _input_string(prompt: Expr = None, init: Expr = None, *opts: Expr) -> String | Expr:
    args = [a for a in (prompt, init) if a is not None] + list(opts)
    _prompt(_input_args(args, "InputString"))
    line = _read_line()
    return Symbol("EndOfFile") if line is None else line


@fn("Input", area=AREA, status="added",
    notes="Input[] / Input[prompt] writes the prompt to $Urgent, reads an expression from standard input "
          "(continuing over further lines while the expression is incomplete) and evaluates it; EndOfFile at "
          "end of input, Null for an empty line. A syntax error raises ValueError. The init and dialog-option "
          "arguments are ignored (no front end). Check: exact (stdin replaced by known text) in tests.")
def _input(prompt: Expr = None, init: Expr = None, *opts: Expr) -> Expr:
    from ..parser import parse, ParseError
    from .wl_files import _expr_complete
    args = [a for a in (prompt, init) if a is not None] + list(opts)
    _prompt(_input_args(args, "Input"))
    buf = _read_line()
    if buf is None:
        return Symbol("EndOfFile")
    while True:
        try:
            done = _expr_complete(buf)
        except ParseError:
            done = True
        if done is None:
            return NULL
        if done:
            break
        more = _read_line()
        if more is None:
            break
        buf += "\n" + more
    try:
        expr = parse(buf)
    except ParseError as e:
        raise ValueError(f"Input: syntax error in {buf!r}: {e}") from None
    return ev(expr)


# ═══════════════════════════════════════════ Exit, Quit ═══════════════════════════════════════════

def _exit_code(n, fname):
    if n is None:
        return 0
    v = _nonneg_int(n, fname, "the exit code")
    if v > 255:
        raise ValueError(f"{fname}: the exit code must be between 0 and 255, not {v}")
    return v


def _terminate(n, fname):
    code = _exit_code(n, fname)
    from .wl_evaluation import epilog
    if _EVAL_ENV.get() == "Session":
        epilog()
    raise SystemExit(code)


@fn("Exit", area=AREA, status="added",
    notes="Exit[] / Exit[n] terminates the MikoshiLang kernel with exit code n (0-255, default 0): $Epilog is "
          "evaluated, then SystemExit(n) is raised, which ends the REPL or script (a host program embedding a "
          "Session can catch SystemExit). Same as Quit. Check: exact (SystemExit code) in tests.")
def _exit(n: NonNegInteger = None) -> Expr:
    _terminate(n, "Exit")


@fn("Quit", area=AREA, status="added",
    notes="Quit[] / Quit[n] terminates the kernel with exit code n (0-255, default 0) after evaluating $Epilog, "
          "by raising SystemExit(n); identical to Exit. Check: exact (SystemExit code) in tests.")
def _quit(n: NonNegInteger = None) -> Expr:
    _terminate(n, "Quit")


# ═══════════════════════════════════════════ Share ═══════════════════════════════════════════

@fn("Share", area=AREA, status="added",
    notes="Share[] / Share[expr] gives the number of bytes saved by sharing common subexpressions. MikoshiLang "
          "expressions are immutable Python objects that are already shared by reference wherever they are "
          "reused, so no further memory can be saved and the result is always 0 (Wolfram's documented result "
          "when nothing can be shared). Check: exact in tests.")
def _share(expr: Expr = None) -> NonNegInteger:
    return 0


# ═══════════════════════════════════════════ SystemInformation ═══════════════════════════════════════════

def _meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                k, _, rest = line.partition(":")
                parts = rest.split()
                if parts and parts[0].isdigit():
                    out[k] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == "kB" else 1)
    except OSError:
        pass
    return out


def _bytes_q(n):
    if n is None:
        return Expr("Missing", "NotAvailable")
    return Expr("Quantity", round(n / 1024 ** 3, 6), "Gibibytes")


def _machine_info():
    mi = _meminfo()
    total = mi.get("MemTotal")
    free = mi.get("MemAvailable", mi.get("MemFree"))
    if total is None:
        try:
            total = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            total = free = None
    used = total - free if total is not None and free is not None else None
    swap_t, swap_f = mi.get("SwapTotal"), mi.get("SwapFree")
    try:
        load = L(*(round(x, 4) for x in os.getloadavg()))
    except (OSError, AttributeError):
        load = Expr("Missing", "NotAvailable")
    try:
        page = os.sysconf("SC_PAGE_SIZE")
    except (ValueError, OSError, AttributeError):
        page = Expr("Missing", "NotAvailable")
    return [("PhysicalTotal", _bytes_q(total)), ("PhysicalFree", _bytes_q(free)), ("PhysicalUsed", _bytes_q(used)),
            ("VirtualTotal", _bytes_q(swap_t)), ("VirtualFree", _bytes_q(swap_f)),
            ("VirtualUsed", _bytes_q(swap_t - swap_f if swap_t is not None and swap_f is not None else None)),
            ("PageSize", page), ("ProcessorCount", _processor_count()), ("LoadAverage", load)]


def _dir_var(name):
    g = E.SYSTEM_VARIABLES.get(name)
    if g is None:
        return Expr("Missing", "NotAvailable")
    try:
        return g()
    except Exception:  # noqa: BLE001 - directory variables are best-effort information
        return Expr("Missing", "NotAvailable")


def _kernel_info():
    import sympy
    return [("Version", _version()), ("ReleaseID", _version_string()), ("CreationDate", _creation_date()),
            ("ProductIDName", "MikoshiLang"), ("ProcessorType", _machine()), ("SystemID", _system_id()),
            ("SystemWordLength", _word_length()), ("OperatingSystem", _os_name()),
            ("OperatingSystemRelease", platform.release() or Expr("Missing", "NotAvailable")),
            ("MachineName", _machine_name()), ("MachineDomains", _machine_domains()), ("MachineID", _machine_id()),
            ("SessionID", _SESSION_ID), ("ProcessID", os.getpid()), ("ParentProcessID", os.getppid()),
            ("Username", _username()), ("CommandLine", _command_line()),
            ("InstallationDirectory", _dir_var("$InstallationDirectory")),
            ("BaseDirectory", _dir_var("$BaseDirectory")), ("UserBaseDirectory", _dir_var("$UserBaseDirectory")),
            ("ProcessorCount", _processor_count()), ("CharacterEncoding", _dir_var("$CharacterEncoding")),
            ("SystemCharacterEncoding", _dir_var("$SystemCharacterEncoding")),
            ("PythonVersion", platform.python_version()), ("PythonExecutable", sys.executable),
            ("SymPyVersion", str(sympy.__version__))]


def _links_info():
    try:
        streams = ev(Expr("Streams"))
    except Exception:  # noqa: BLE001
        streams = L(Expr("OutputStream", "stdout", 1), Expr("OutputStream", "stderr", 2))
    return [("OpenStreams", streams), ("Processes", L(*(p.expr() for p in _live_processes()))),
            ("KernelCount", len(_KERNELS))]


def _network_info():
    try:
        allow = bool(ev(Expr("$AllowInternet"))) if "$AllowInternet" in E.SYSTEM_VARIABLES else True
    except Exception:  # noqa: BLE001
        allow = True
    return [("MachineName", _machine_name()), ("MachineDomains", _machine_domains()),
            ("AllowInternetUse", allow is True)]


_SYSINFO = {
    "Kernel": _kernel_info, "Machine": _machine_info, "Links": _links_info, "Network": _network_info,
    "FrontEnd": None, "Devices": None,
}


def _component(name, fname):
    if name not in _SYSINFO:
        raise ValueError(f"{fname}: unknown component {name!r} (known: {', '.join(_SYSINFO)})")
    g = _SYSINFO[name]
    if g is None:
        return None
    return g()


@fn("SystemInformation", area=AREA, status="added",
    notes="SystemInformation[] gives {comp -> {prop -> value, ...}, ...} for the components \"Kernel\" (version, "
          "creation date, system id, process ids, user, machine, directories, Python/SymPy versions), "
          "\"Machine\" (physical and virtual memory as Quantity[..., \"Gibibytes\"], page size, processors, load "
          "average), \"Links\" (open streams, child processes, parallel kernels) and \"Network\"; \"FrontEnd\" and "
          "\"Devices\" are Missing[\"NotActive\"] (no front end). SystemInformation[comp], [comp, prop], [comp, "
          "\"Properties\"], [\"Components\"] and [\"Small\"] (the main Kernel properties). Unknown names raise "
          "ValueError. Check: reference (Python platform/os values) in tests.")
def _system_information(comp: String | Expr = None, prop: String = None) -> Expr:
    fname = "SystemInformation"
    if comp is None or is_sym(comp, "All"):
        out = []
        for k in _SYSINFO:
            v = _component(k, fname)
            out.append(Expr("Rule", k, Expr("Missing", "NotActive") if v is None else _rules(v)))
        return L(*out)
    c = _string(comp, fname, "the component")
    if c == "Components":
        if prop is not None:
            raise TypeError("SystemInformation: \"Components\" takes no property")
        return L(*_SYSINFO)
    if c == "Small":
        if prop is not None:
            raise TypeError("SystemInformation: \"Small\" takes no property")
        keep = ("Version", "ReleaseID", "SystemID", "OperatingSystem", "ProcessorType", "MachineName", "ProcessID")
        return _rules([kv for kv in _kernel_info() if kv[0] in keep])
    v = _component(c, fname)
    if prop is None:
        return Expr("Missing", "NotActive") if v is None else _rules(v)
    p = _string(prop, fname, "the property")
    if v is None:
        return Expr("Missing", "NotActive")
    if p == "Properties":
        return L(*(k for k, _ in v))
    for k, val in v:
        if k == p:
            return val
    raise ValueError(f"{fname}: {c!r} has no property {p!r} (known: {', '.join(k for k, _ in v)})")


# ═══════════════════════════════════════════ SystemOptions ═══════════════════════════════════════════

_SYS_OPTIONS = {
    "ParallelOptions": {"MathLinkTimeout": 60.0, "RelaunchFailedKernels": False, "RecoveryMode": "ReQueue",
                        "MaxKernelLaunchCount": 64},
    "RecursionOptions": None,         # live view of $RecursionLimit / $IterationLimit
    "ProcessOptions": {"KillTimeout": 1.0},
}


def _recursion_options():
    st = K.state().own
    def get(n):
        v = st.get(n)
        if v is None:
            g = E.SYSTEM_VARIABLES.get(n)
            v = g() if g is not None else NONE
        return v
    return {"RecursionLimit": get("$RecursionLimit"), "IterationLimit": get("$IterationLimit")}


def _group(name):
    if name == "RecursionOptions":
        return _recursion_options()
    return _SYS_OPTIONS[name]


def _sysopt_rules(name):
    return Expr("Rule", name, _rules(list(_group(name).items())))


def _opt_value_ok(group, key, v):
    if group == "ParallelOptions":
        if key == "MathLinkTimeout":
            return _nonneg_real(v, "SetSystemOptions", "MathLinkTimeout") is not None
        if key == "RelaunchFailedKernels":
            return isinstance(v, bool)
        if key == "RecoveryMode":
            return v in ("ReQueue", "Abandon")
        if key == "MaxKernelLaunchCount":
            return K.is_int(v) and not isinstance(v, bool) and int(v) >= 1
    if group == "ProcessOptions" and key == "KillTimeout":
        return _nonneg_real(v, "SetSystemOptions", "KillTimeout") is not None
    if group == "RecursionOptions":
        return (K.is_int(v) and not isinstance(v, bool) and int(v) >= 20) or is_sym(v, "Infinity")
    return False


@fn("SystemOptions", area=AREA, status="added",
    notes="SystemOptions[] gives the internal system options as nested rules; SystemOptions[\"group\"] gives "
          "{\"group\" -> {...}} and SystemOptions[\"group\" -> \"name\"] (or [\"group\", \"name\"]) one setting. Groups: "
          "\"ParallelOptions\" (MathLinkTimeout: seconds to wait for a launched subkernel, default 60; "
          "RelaunchFailedKernels; RecoveryMode \"ReQueue\"/\"Abandon\" for evaluations of a failed kernel; "
          "MaxKernelLaunchCount), \"ProcessOptions\" (KillTimeout: seconds KillProcess waits before SIGKILL) and "
          "\"RecursionOptions\" ($RecursionLimit and $IterationLimit). Unknown names raise ValueError. Check: exact "
          "(round trip with SetSystemOptions) in tests.")
def _system_options(spec: String | Rule = None, sub: String = None) -> ListOf[Rule] | Rule:
    fname = "SystemOptions"
    if spec is None:
        return L(*(_sysopt_rules(g) for g in _SYS_OPTIONS))
    if isinstance(spec, Expr) and spec.head == "Rule" and len(spec.args) == 2 and sub is None:
        spec, sub = spec.args
    g = _string(spec, fname, "the option group")
    if g not in _SYS_OPTIONS:
        raise ValueError(f"{fname}: unknown option group {g!r} (known: {', '.join(_SYS_OPTIONS)})")
    if sub is None:
        return L(_sysopt_rules(g))
    k = _string(sub, fname, "the option name")
    grp = _group(g)
    if k not in grp:
        raise ValueError(f"{fname}: {g} has no option {k!r} (known: {', '.join(grp)})")
    return L(Expr("Rule", g, L(Expr("Rule", k, grp[k]))))


def _set_sysopt(r, group=None):
    fname = "SetSystemOptions"
    if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2):
        raise TypeError(f"{fname}: {_form(r)} is not a rule \"group\" -> {{\"name\" -> value}}")
    k, v = r.args
    k = _string(k, fname, "the option name")
    if group is None:
        if k not in _SYS_OPTIONS:
            raise ValueError(f"{fname}: unknown option group {k!r} (known: {', '.join(_SYS_OPTIONS)})")
        items = v.args if is_list(v) else (v,)
        return [x for item in items for x in _set_sysopt(item, k)]
    grp = _group(group)
    if k not in grp:
        raise ValueError(f"{fname}: {group} has no option {k!r} (known: {', '.join(grp)})")
    if not _opt_value_ok(group, k, v):
        raise ValueError(f"{fname}: {_form(v)} is not a valid value for {group} -> {k}")
    return [(group, k, v)]


@fn("SetSystemOptions", area=AREA, status="added",
    notes="SetSystemOptions[\"group\" -> {\"name\" -> value, ...}, ...] resets internal system options (see "
          "SystemOptions) and returns the new settings of the groups changed; every value is validated first "
          "(nothing changes if one is invalid). RecursionOptions set $RecursionLimit/$IterationLimit. Check: "
          "exact (SystemOptions reads back the value) in tests.")
def _set_system_options(*rules: Rule) -> ListOf[Rule]:
    if not rules:
        raise TypeError("SetSystemOptions: called with 0 arguments; at least one rule is expected")
    changes = []
    for r in rules:
        for item in (r.args if is_list(r) else (r,)):
            changes.extend(_set_sysopt(item))
    for g, k, v in changes:
        if g == "RecursionOptions":
            ev(Expr("Set", Symbol("$" + k), v))
        else:
            _SYS_OPTIONS[g][k] = float(_nonneg_real(v, "SetSystemOptions")) if k in ("MathLinkTimeout",
                                                                                     "KillTimeout") else v
    groups = list(dict.fromkeys(g for g, _, _ in changes))
    return L(*(_sysopt_rules(g) for g in groups))


def _sysopt(group, key):
    return _SYS_OPTIONS[group][key]


# ═══════════════════════════════════════════ external processes ═══════════════════════════════════════════

class _PipeIn:
    """Raw input stream over a child's stdout/stderr pipe: a reader thread drains the pipe into a buffer (so the
    child never blocks on a full pipe); reads block until enough data or end of file, except in available()."""

    def __init__(self, f):
        self.f = f
        self.buf = bytearray()
        self.pos = 0
        self.eof = False
        self.cv = threading.Condition()
        self._nb = threading.local()
        self.t = threading.Thread(target=self._pump, daemon=True, name="w29-pipe")
        self.t.start()

    def _pump(self):
        fd = self.f.fileno()
        try:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    chunk = b""
                with self.cv:
                    if not chunk:
                        self.eof = True
                        self.cv.notify_all()
                        return
                    self.buf += chunk
                    self.cv.notify_all()
        finally:
            with self.cv:
                self.eof = True
                self.cv.notify_all()

    def read(self, n=-1):
        with self.cv:
            nb = getattr(self._nb, "on", False)
            if nb and len(self.buf) == self.pos:
                return b""
            if n is None or n < 0:
                while not self.eof and not nb:
                    self.cv.wait()
                out = bytes(self.buf[self.pos:])
            else:
                while len(self.buf) - self.pos < n and not self.eof:
                    self.cv.wait()
                out = bytes(self.buf[self.pos:self.pos + n])
            self.pos += len(out)
            return out

    def tell(self):
        return self.pos

    def seek(self, p, whence=0):
        with self.cv:
            if whence == 2:
                self.pos = len(self.buf)
            else:
                self.pos = max(0, min(int(p), len(self.buf)))
            return self.pos

    def available(self, thunk, eof_value):
        """thunk() run without blocking (what is buffered now); eof_value if the pipe is exhausted."""
        with self.cv:
            if self.eof and self.pos == len(self.buf):
                return eof_value
        self._nb.on = True
        try:
            return thunk()
        finally:
            self._nb.on = False

    def close(self):
        try:
            self.f.close()
        except OSError:
            pass


class _PipeOut:
    """Raw output stream over a child's stdin pipe."""

    def __init__(self, f):
        self.f = f
        self.n = 0

    def write(self, data):
        try:
            self.f.write(data)
            self.f.flush()
        except (BrokenPipeError, ValueError, OSError):
            raise ValueError("the process is no longer reading its standard input") from None
        self.n += len(data)

    def flush(self):
        pass

    def tell(self):
        return self.n

    def seek(self, *a):
        raise NotImplementedError("SetStreamPosition: a process input stream cannot be repositioned")

    def close(self):
        try:
            self.f.close()
        except (OSError, ValueError):
            pass


_PROCS = {}          # uid -> _Proc
_PROC_UID = itertools.count()
_PROC_LOCK = threading.Lock()


class _Proc:
    def __init__(self, popen, argv):
        self.p = popen
        self.argv = argv
        self.uid = next(_PROC_UID)
        self.started = time.time()
        self.raws = {"out": _PipeIn(popen.stdout), "err": _PipeIn(popen.stderr), "in": _PipeOut(popen.stdin)}
        self.streams = {}
        with _PROC_LOCK:
            _PROCS[self.uid] = self

    def expr(self):
        return Expr("ProcessObject", assoc([("ManagedProcess", True), ("UID", self.uid), ("PID", self.p.pid),
                                             ("PPID", os.getpid())]))

    def stream(self, which):
        from . import wl_files as WF
        s = self.streams.get(which)
        if s is None or s.n not in WF.STREAMS:
            name = f"{which}:{self.uid}"
            s = WF._Stream("out" if which == "in" else "in", name, self.raws[which], binary=False, enc="utf-8",
                           method="Process")
            self.streams[which] = s
        return s

    def running(self):
        return self.p.poll() is None

    def kill(self):
        if self.p.poll() is not None:
            return
        try:
            self.p.terminate()
            self.p.wait(timeout=_sysopt("ProcessOptions", "KillTimeout"))
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        except OSError:
            pass


def _live_processes():
    return [p for p in list(_PROCS.values()) if p.running()]


def _kill_all_at_exit():
    for p in list(_PROCS.values()):
        try:
            p.kill()
        except Exception:  # noqa: BLE001 - best effort at interpreter exit
            pass


atexit.register(_kill_all_at_exit)


def _cmd_arg(a, fname):
    if isinstance(a, str):
        return a
    if isinstance(a, Expr) and a.head == "Rule" and len(a.args) == 2 and isinstance(a.args[0], str):
        k = a.args[0]
        return ("-" if len(k) == 1 else "--") + k + "=" + _cmd_arg(a.args[1], fname)
    if isinstance(a, Expr) and a.head == "File" and len(a.args) == 1 and isinstance(a.args[0], str):
        return os.path.expanduser(a.args[0])
    if isinstance(a, bool) or not (K.is_int(a) or isinstance(a, (float,)) or K.is_number(a)):
        if _is_symbolic(a):
            raise Unevaluated()
        raise TypeError(f"{fname}: the command argument {_form(a)} must be a string, a number or a rule "
                        "\"name\" -> value")
    return _out_text(a)


def _argv(cmd, fname):
    """argv for a command given as "program" or {"program", arg1, ...} (the program is not split)."""
    if is_list(cmd):
        if not cmd.args:
            raise ValueError(f"{fname}: the command list is empty")
        prog = cmd.args[0]
        rest = [_cmd_arg(a, fname) for a in cmd.args[1:]]
    else:
        prog, rest = cmd, []
    if isinstance(prog, Expr) and prog.head == "File" and len(prog.args) == 1:
        prog = prog.args[0]
    prog = _string(prog, fname, "the program")
    if not prog:
        raise ValueError(f"{fname}: the program name is empty")
    return [os.path.expanduser(prog)] + rest


def _proc_env(v, fname):
    if v is None or is_sym(v, "Inherited"):
        return None
    if is_list(v):
        v = Expr("Association", *v.args)
    d = _assoc_dict(v, fname, "ProcessEnvironment")
    env = dict(os.environ)
    for k, val in d.items():
        k = _env_name(k, fname)
        if is_sym(val, "None"):
            env.pop(k, None)
        elif isinstance(val, str):
            env[k] = val
        elif K.is_int(val) and not isinstance(val, bool):
            env[k] = str(int(val))
        else:
            raise TypeError(f"{fname}: the ProcessEnvironment value of {k} must be a string or None, not {_form(val)}")
    return env


def _proc_dir(v, fname):
    if v is None or is_sym(v, "Inherited"):
        return None
    if isinstance(v, Expr) and v.head == "File" and len(v.args) == 1:
        v = v.args[0]
    d = os.path.expanduser(_string(v, fname, "ProcessDirectory"))
    if not os.path.isdir(d):
        raise ValueError(f"{fname}: the ProcessDirectory {d!r} is not an existing directory")
    return d


_PROC_OPTS = ("ProcessDirectory", "ProcessEnvironment")


def _popen(argv, o, fname, **kw):
    try:
        return subprocess.Popen(argv, cwd=_proc_dir(o.get("ProcessDirectory"), fname),
                                env=_proc_env(o.get("ProcessEnvironment"), fname), **kw)
    except FileNotFoundError:
        raise ValueError(f"{fname}: program {argv[0]!r} not found (check Environment[\"PATH\"])") from None
    except PermissionError:
        raise ValueError(f"{fname}: program {argv[0]!r} cannot be executed (permission denied)") from None
    except OSError as e:
        raise ValueError(f"{fname}: cannot start {argv[0]!r}: {e.strerror or e}") from None


def _decode(b):
    return b.decode("utf-8", errors="replace")


def _input_bytes(x):
    if isinstance(x, str):
        return x.encode("utf-8")
    if isinstance(x, Expr) and x.head == "ByteArray":
        from .wl_files import _ba_bytes
        return bytes(_ba_bytes(x, "RunProcess"))
    return _out_text(x).encode("utf-8")


_RP_PROPS = ("ExitCode", "StandardOutput", "StandardError")


@fn("RunProcess", area=AREA, status="added",
    notes="RunProcess[\"program\"] / RunProcess[{\"program\", arg1, ...}] runs an external program to completion "
          "(no shell; the program name is not split) and gives <|\"ExitCode\" -> n, \"StandardOutput\" -> s, "
          "\"StandardError\" -> s|>; RunProcess[cmd, \"prop\"] one of these, RunProcess[cmd, All] the association, "
          "RunProcess[cmd, prop, input] feeds input (a string, ByteArray, or the OutputForm text of an expression) "
          "to the standard input. Arguments \"name\" -> value become --name=value (-n=value for one-letter names); "
          "numbers are converted to text. Options ProcessDirectory and ProcessEnvironment (an association merged "
          "into the inherited environment; None unsets a variable). A program that cannot be started raises "
          "ValueError (Wolfram returns $Failed). Note: this runs arbitrary programs with the user's rights. "
          "Check: reference (Python subprocess results for sys.executable -c programs) in tests.")
def _run_process(cmd: String | ListOf[Expr], prop: String | Expr = None, inp: Expr = None, *opts: Option) -> Expr:
    args, o = _opts([a for a in (prop, inp) if a is not None] + list(opts), _PROC_OPTS, "RunProcess")
    if len(args) > 2:
        raise TypeError("RunProcess: called with too many arguments")
    prop = args[0] if args else ALL
    if not (is_sym(prop, "All") or prop in _RP_PROPS):
        raise ValueError(f"RunProcess: {_form(prop)} is not a property (All, \"ExitCode\", \"StandardOutput\", "
                         "\"StandardError\")")
    data = _input_bytes(args[1]) if len(args) > 1 else b""
    argv = _argv(cmd, "RunProcess")
    p = _popen(argv, o, "RunProcess", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        out, err = p.communicate(data)
    except BaseException:
        p.kill()
        p.wait()
        raise
    res = {"ExitCode": p.returncode if p.returncode >= 0 else 128 - p.returncode,
           "StandardOutput": _decode(out), "StandardError": _decode(err)}
    if is_sym(prop, "All"):
        return assoc([(k, res[k]) for k in _RP_PROPS])
    return res[prop]


@fn("StartProcess", area=AREA, status="added",
    notes="StartProcess[\"program\"] / StartProcess[{\"program\", arg1, ...}] starts an external program in the "
          "background (no shell) and gives ProcessObject[<|\"ManagedProcess\" -> True, \"UID\" -> k, \"PID\" -> pid, "
          "\"PPID\" -> ...|>]; WriteLine/WriteString/BinaryWrite on it write to its standard input, ReadLine/"
          "ReadString/Read on it read its standard output (ReadString[proc, EndOfBuffer] what is available now), "
          "ProcessConnection gives the individual streams. Arguments \"name\" -> value become --name=value. Options "
          "ProcessDirectory and ProcessEnvironment as in RunProcess. Extra positional arguments leave it "
          "unevaluated (as documented); a program that cannot be started raises ValueError. Started processes are "
          "terminated when MikoshiLang exits. Check: reference (Python subprocess) in tests.")
def _start_process(cmd: String | ListOf[Expr], *opts: Option) -> Expr:
    args, o = _opts(opts, _PROC_OPTS, "StartProcess")
    if args:
        raise Unevaluated()
    argv = _argv(cmd, "StartProcess")
    p = _popen(argv, o, "StartProcess", stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return _Proc(p, argv).expr()


def _proc_obj(x, fname, unmanaged=True):
    """(_Proc or None, pid) for a ProcessObject."""
    if not (isinstance(x, Expr) and x.head == "ProcessObject" and len(x.args) == 1):
        if _is_symbolic(x) or isinstance(x, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: {_form(x)} is not a ProcessObject")
    a = x.args[0]
    if is_assoc(a):
        d = _assoc_dict(a, fname, "the process data")
        if d.get("ManagedProcess") is True:
            uid = d.get("UID")
            p = _PROCS.get(int(uid)) if K.is_int(uid) else None
            if p is None:
                raise ValueError(f"{fname}: {_form(x)} is not a process started in this session")
            return p, p.p.pid
        pid = d.get("PID")
        if not K.is_int(pid):
            raise ValueError(f"{fname}: {_form(x)} has no PID")
        pid = int(pid)
    elif K.is_int(a) and not isinstance(a, bool):
        pid = int(a)
    else:
        raise TypeError(f"{fname}: {_form(x)} is not a valid ProcessObject")
    for p in _PROCS.values():
        if p.p.pid == pid:
            return p, pid
    if not unmanaged:
        raise ValueError(f"{fname}: {_form(x)} is not a process started in this session")
    return None, pid


def _pid_alive(pid):
    if os.path.isdir("/proc") and os.path.exists("/proc/self"):
        try:
            with open(f"/proc/{pid}/stat") as fh:
                return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False
    try:
        os.kill(pid, 0)          # signal 0: existence check only, nothing is delivered
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


@fn("ProcessStatus", area=AREA, status="added",
    notes="ProcessStatus[proc] gives \"Running\" or \"Finished\" for a ProcessObject (one started by StartProcess, "
          "or a system process from SystemProcesses / ProcessObject[pid]); ProcessStatus[proc, \"status\"] gives "
          "True if the process has that status. Check: exact (process exits after it is told to) in tests.")
def _process_status(proc: Expr, status: String = None) -> String | Boolean:
    p, pid = _proc_obj(proc, "ProcessStatus")
    s = ("Running" if p.running() else "Finished") if p is not None else (
        "Running" if _pid_alive(pid) else "Finished")
    if status is None:
        return s
    st = _string(status, "ProcessStatus", "the status")
    if st not in ("Running", "Finished"):
        raise ValueError(f"ProcessStatus: {st!r} is not a status (\"Running\" or \"Finished\")")
    return st == s


def _proc_info(p):
    rc = p.p.poll()
    return [("ExitCode", NONE if rc is None else (rc if rc >= 0 else 128 - rc))]


_PI_EXTRA = ("PID", "PPID", "Program", "Arguments", "StartTime", "Status")


@fn("ProcessInformation", area=AREA, status="added",
    notes="ProcessInformation[proc] gives <|\"ExitCode\" -> n|> for a process started with StartProcess (None "
          "while it is running; 128+s if killed by signal s); ProcessInformation[proc, \"prop\"] gives \"ExitCode\" "
          "or one of \"PID\", \"PPID\", \"Program\", \"Arguments\", \"StartTime\", \"Status\". Check: exact (exit "
          "codes of sys.executable -c programs) in tests.")
def _process_information(proc: Expr, prop: String = None) -> Expr:
    p, pid = _proc_obj(proc, "ProcessInformation", unmanaged=False)
    if prop is None:
        return assoc(_proc_info(p))
    k = _string(prop, "ProcessInformation", "the property")
    if k == "ExitCode":
        return _proc_info(p)[0][1]
    if k == "PID":
        return pid
    if k == "PPID":
        return os.getpid()
    if k == "Program":
        return p.argv[0]
    if k == "Arguments":
        return L(*p.argv[1:])
    if k == "StartTime":
        return _date_object(p.started)
    if k == "Status":
        return "Running" if p.running() else "Finished"
    raise ValueError(f"ProcessInformation: unknown property {k!r} (known: ExitCode, {', '.join(_PI_EXTRA)})")


_CONN = {"StandardInput": "in", "StandardOutput": "out", "StandardError": "err"}


@fn("ProcessConnection", area=AREA, status="added",
    notes="ProcessConnection[proc, \"StandardInput\"] gives the OutputStream[\"in:k\", n] writing to a started "
          "process's standard input; \"StandardOutput\" and \"StandardError\" give InputStream[\"out:k\", n] / "
          "[\"err:k\", n] reading its output (W27 stream functions work on them; Close on the input stream sends "
          "end-of-file). Check: exact (round trip through a sys.executable -c echo program) in tests.")
def _process_connection(proc: Expr, which: String) -> Expr:
    p, _ = _proc_obj(proc, "ProcessConnection", unmanaged=False)
    w = _string(which, "ProcessConnection", "the stream name")
    if w not in _CONN:
        raise ValueError(f"ProcessConnection: {w!r} is not a stream (StandardInput, StandardOutput, StandardError)")
    return p.stream(_CONN[w]).expr()


@fn("KillProcess", area=AREA, status="added",
    notes="KillProcess[proc] terminates a process: SIGTERM, then SIGKILL after the ProcessOptions KillTimeout "
          "(default 1 s) for processes started by StartProcess; SIGTERM for other ProcessObjects. Returns Null; a "
          "finished process is left alone. Check: exact (ProcessStatus becomes \"Finished\") in tests.")
def _kill_process(proc: Expr) -> Expr:
    p, pid = _proc_obj(proc, "KillProcess")
    if p is not None:
        p.kill()
        return NULL
    if pid == os.getpid():
        raise ValueError("KillProcess: use Exit[] to end this kernel")
    if not _pid_alive(pid):
        return NULL
    import signal
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        raise ValueError(f"KillProcess: not permitted to kill process {pid}") from None
    return NULL


@fn("Processes", area=AREA, status="added",
    notes="Processes[] lists the ProcessObjects of the external processes started in this session by "
          "StartProcess that are still running, in start order. Check: exact (before/after KillProcess) in tests.")
def _processes() -> ListOf[Expr]:
    return L(*(p.expr() for p in _live_processes()))


def _proc_table(all_users):
    """[{prop: value}] for the system processes (Linux /proc, else ps)."""
    rows = []
    me = os.getuid() if hasattr(os, "getuid") else None
    if os.path.isdir("/proc") and os.path.exists("/proc/self/stat"):
        try:
            clk = os.sysconf("SC_CLK_TCK")
            with open("/proc/stat") as fh:
                btime = next(int(l.split()[1]) for l in fh if l.startswith("btime"))
        except (OSError, ValueError, StopIteration):
            clk, btime = 100, None
        try:
            page = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError):
            page = 4096
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                with open(f"/proc/{d}/stat") as fh:
                    stat = fh.read()
                uid = os.stat(f"/proc/{d}").st_uid
                with open(f"/proc/{d}/statm") as fh:
                    rss = int(fh.read().split()[1]) * page
            except (OSError, ValueError, IndexError):
                continue
            if not all_users and me is not None and uid != me:
                continue
            name = stat[stat.find("(") + 1:stat.rfind(")")]
            f = stat[stat.rfind(")") + 2:].split()
            try:
                path = os.readlink(f"/proc/{d}/exe")
            except OSError:
                path = Expr("Missing", "NotAvailable")
            start = (btime + int(f[19]) / clk) if btime is not None else None
            rows.append({"PID": int(d), "PPID": int(f[1]), "Program": name, "Path": path, "User": _uname(uid),
                         "Memory": Expr("Quantity", rss / 1e6, "Megabytes"), "Threads": int(f[17]),
                         "StartTime": _date_object(start) if start else Expr("Missing", "NotAvailable")})
        return rows
    if _os_name() == "Windows":
        raise NotImplementedError("SystemProcesses: listing system processes is not supported on Windows")
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,uid=,rss=,comm="], capture_output=True, text=True,
                             timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        raise NotImplementedError("SystemProcesses: the ps command is not available") from None
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        pid, ppid, uid, rss, comm = parts
        if not all_users and me is not None and int(uid) != me:
            continue
        rows.append({"PID": int(pid), "PPID": int(ppid), "Program": os.path.basename(comm), "Path": comm,
                     "User": _uname(int(uid)), "Memory": Expr("Quantity", int(rss) * 1024 / 1e6, "Megabytes"),
                     "Threads": Expr("Missing", "NotAvailable"), "StartTime": Expr("Missing", "NotAvailable")})
    return rows


def _uname(uid):
    try:
        import pwd
        return pwd.getpwuid(uid).pw_name
    except (ImportError, KeyError):
        return str(uid)


_SP_PROPS = ("PID", "PPID", "Program", "Path", "User", "Memory", "Threads", "StartTime")


def _sp_expr(row):
    return Expr("ProcessObject", assoc([("ManagedProcess", False)] + [(k, row[k]) for k in _SP_PROPS]))


def _sp_match(row, k, v):
    if k not in _SP_PROPS:
        raise ValueError(f"SystemProcesses: unknown property {_form(k)} (known: {', '.join(_SP_PROPS)})")
    x = row[k]
    if isinstance(v, str) and isinstance(x, str):
        return v.lower() in x.lower()
    if isinstance(v, str) or (isinstance(v, Expr) and v.head in ("StringExpression", "RegularExpression")):
        return isinstance(x, str) and ev(Expr("StringContainsQ", x, v, Expr("Rule", Symbol("IgnoreCase"), True))) \
            is True
    return ev(Expr("MatchQ", x, v)) is True


@fn("SystemProcesses", area=AREA, status="added",
    notes="SystemProcesses[] gives ProcessObject[<|\"ManagedProcess\" -> False, \"PID\", \"PPID\", \"Program\", "
          "\"Path\", \"User\", \"Memory\" (Quantity in megabytes), \"Threads\", \"StartTime\"|>] for the processes "
          "of the current user, SystemProcesses[All] for all users; SystemProcesses[patt] those whose program name "
          "contains the string (pattern), ignoring case; SystemProcesses[prop -> val] / [{prop -> val, ...}] those "
          "whose properties match (a string is a case-insensitive substring of string properties, otherwise a "
          "pattern matched with MatchQ). proc[\"prop\"] reads a property. Linux reads /proc, other Unix systems use "
          "ps; Windows raises NotImplementedError. Check: reference (os.getpid() and the test's own child "
          "processes) in tests.")
def _system_processes(spec: Expr = None) -> ListOf[Expr]:
    if spec is None or is_sym(spec, "All"):
        return L(*(_sp_expr(r) for r in _proc_table(spec is not None)))
    rows = _proc_table(False)
    if is_rule(spec) or (is_list(spec) and spec.args and all(is_rule(r) for r in spec.args)):
        conds = [r.args for r in (spec.args if is_list(spec) else (spec,))]
        return L(*(_sp_expr(r) for r in rows if all(_sp_match(r, k, v) for k, v in conds)))
    if isinstance(spec, Symbol):
        raise Unevaluated()
    return L(*(_sp_expr(r) for r in rows if _sp_match(r, "Program", spec)))


def _proc_props_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "ProcessObject" and len(head.args) == 1 and len(args) == 1
            and isinstance(args[0], str)):
        return K.UNHANDLED
    a = head.args[0]
    k = args[0]
    if is_assoc(a):
        d = _assoc_dict(a, "ProcessObject")
        if d.get("ManagedProcess") is True:
            if k in d:
                return d[k]
            if k == "ExitCode":
                return _process_information(head, "ExitCode")
            if k in _PI_EXTRA:
                return _process_information(head, k)
            raise ValueError(f"ProcessObject: unknown property {k!r}")
        if k == "ManagedProcess":
            return False
        if k in _SP_PROPS:
            pid = d.get("PID")
            if k in ("Memory", "Threads"):
                for r in _proc_table(True):
                    if r["PID"] == pid:
                        return r[k]
                return Expr("Missing", "NotAvailable")
            return d.get(k, Expr("Missing", "NotAvailable"))
        raise ValueError(f"ProcessObject: unknown property {k!r} (known: {', '.join(_SP_PROPS)})")
    return K.UNHANDLED


KL.COMPOUND_HEAD_HOOKS["ProcessObject"] = _proc_props_hook
record("ProcessObject", AREA, "added",
       "ProcessObject[<|\"ManagedProcess\" -> True, \"UID\", \"PID\", \"PPID\"|>] represents a process started by "
       "StartProcess, ProcessObject[<|\"ManagedProcess\" -> False, ...|>] a system process from SystemProcesses, "
       "and ProcessObject[pid] the system process with that id; it is used by ProcessStatus, ProcessInformation, "
       "ProcessConnection, KillProcess and the W27 read/write functions, and proc[\"prop\"] gives a property. "
       "Check: exact (properties of a started process) in tests.")
record("ProcessEnvironment", AREA, "added",
       "ProcessEnvironment: option of RunProcess and StartProcess: Inherited (default, this kernel's environment) "
       "or an association \"var\" -> \"value\" merged into the inherited environment (None unsets a variable). "
       "Check: exact (the child reports the variable) in tests.")
record("ProcessDirectory", AREA, "added",
       "ProcessDirectory: option of RunProcess and StartProcess giving the child's working directory: Inherited "
       "(default, the current Directory[]) or an existing directory (string or File); others raise ValueError. "
       "Check: exact (the child reports its working directory) in tests.")


def _process_streams(x, which):
    if isinstance(x, Expr) and x.head == "ProcessObject":
        p, _ = _proc_obj(x, "process stream", unmanaged=False)
        return p.stream(which)
    return None


def _wrap_streams():
    """Let the W27 stream functions accept a ProcessObject (stdin for writing, stdout for reading)."""
    from . import wl_files as WF
    if getattr(WF._input, "_w29", False):
        return
    oi, oo = WF._input, WF._output

    def _input(x, fname, o=None, auto_close=False, binary=False):
        s = _process_streams(x, "out")
        if s is not None:
            return s, False
        return oi(x, fname, o, auto_close, binary)

    def _output(x, fname):
        s = _process_streams(x, "in")
        if s is not None:
            return s
        return oo(x, fname)

    _input._w29 = _output._w29 = True
    WF._input, WF._output = _input, _output


_wrap_streams()


def _run_args(args, fname):
    parts = []
    for a in args:
        parts.append(a if isinstance(a, str) else _form(a))
    return " ".join(parts)


@fn("Run", area=AREA, status="added",
    notes="Run[\"command\"] runs command through the system shell ($SystemShell; cmd.exe on Windows) and gives its "
          "exit code (128+s if killed by signal s); Run[e1, e2, ...] runs the strings as they are and other "
          "expressions in InputForm, separated by spaces. The command's output goes to the kernel's standard "
          "output and error. Note: this runs arbitrary shell commands. Check: reference (exit codes of "
          "sys.executable -c programs) in tests.")
def _run(*args: Expr) -> Integer:
    if not args:
        raise TypeError("Run: called with 0 arguments; at least 1 argument is expected")
    if any(_is_symbolic(a) for a in args if not isinstance(a, str)) and not isinstance(args[0], str):
        raise Unevaluated()
    cmd = _run_args(args, "Run")
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        rc = subprocess.call(cmd, shell=True)
    except OSError as e:
        raise ValueError(f"Run: cannot run the shell: {e}") from None
    return rc if rc >= 0 else 128 - rc


@fn("RunThrough", area=AREA, status="added",
    notes="RunThrough[\"command\", expr] runs command through the system shell with the InputForm text of expr "
          "(evaluated first; use Unevaluated to send it as is) on its standard input, then parses the command's "
          "standard output as MikoshiLang input and evaluates it (Null if there is no output). A non-zero exit "
          "code with no output raises ValueError. Check: exact (a sys.executable -c program transforming its "
          "input) in tests.")
def _run_through(cmd: String, expr: Expr) -> Expr:
    from ..parser import parse, ParseError
    c = _string(cmd, "RunThrough", "the command")
    text = _form(expr)
    try:
        r = subprocess.run(c, shell=True, input=(text + "\n").encode("utf-8"), capture_output=True)
    except OSError as e:
        raise ValueError(f"RunThrough: cannot run the shell: {e}") from None
    out = _decode(r.stdout).strip()
    if not out:
        if r.returncode != 0:
            raise ValueError(f"RunThrough: the command failed with exit code {r.returncode}: "
                             f"{_decode(r.stderr).strip()}")
        return NULL
    try:
        return ev(parse(out))
    except ParseError as e:
        raise ValueError(f"RunThrough: the output {out!r} is not valid input: {e}") from None


def _open_command(target):
    if _os_name() == "Windows":
        return None
    return ["open", target] if _os_name() == "MacOSX" else ["xdg-open", target]


@fn("SystemOpen", area=AREA, status="added",
    notes="SystemOpen[\"file\"], SystemOpen[File[...]] or SystemOpen[URL[...]] / SystemOpen[\"http...\"] opens the "
          "target with the associated program of the desktop (xdg-open on Linux, open on macOS, os.startfile on "
          "Windows) without waiting, and returns Null; a file that does not exist raises ValueError, a URL "
          "without a scheme://... raises ValueError and a missing opener raises NotImplementedError. Check: exact "
          "(the opener command, with the launcher replaced) in tests.")
def _system_open(target: String | Expr) -> Expr:
    import re
    t = target
    is_url = False
    if isinstance(t, Expr) and t.head in ("URL", "File") and len(t.args) == 1 and isinstance(t.args[0], str):
        is_url = t.head == "URL"
        t = t.args[0]
    t = _string(t, "SystemOpen", "the target")
    if is_url or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", t) or t.startswith("mailto:"):
        if not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:(//[^/\s]\S*|[^/\s]\S*)$", t):
            raise ValueError(f"SystemOpen: {t!r} is not a valid URL")
    else:
        t = os.path.abspath(os.path.expanduser(t))
        if not os.path.exists(t):
            raise ValueError(f"SystemOpen: the file {t!r} does not exist")
    cmd = _open_command(t)
    if cmd is None:
        try:
            os.startfile(t)            # noqa: S606 - Windows only
        except OSError as e:
            raise ValueError(f"SystemOpen: cannot open {t!r}: {e}") from None
        return NULL
    try:
        _LAUNCH(cmd)
    except FileNotFoundError:
        raise NotImplementedError(f"SystemOpen: the desktop opener {cmd[0]!r} is not installed") from None
    return NULL


def _launch(cmd):
    subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


_LAUNCH = _launch


# ═══════════════════════════════════════════ parallel kernels: wire protocol ═══════════════════════════════════════════

def _send_msg(f, lock, msg):
    try:
        data = pickle.dumps(msg, protocol=4)
    except Exception as e:  # noqa: BLE001 - pickling arbitrary user data
        raise TypeError(f"cannot transfer {type(msg[0]).__name__} data to a parallel kernel: {e}") from None
    with lock:
        f.write(struct.pack(">Q", len(data)) + data)
        f.flush()


def _recv_msg(f):
    head = f.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack(">Q", head)
    data = f.read(n)
    if len(data) < n:
        return None
    return pickle.loads(data)


_KERNELS = collections.OrderedDict()    # kernel id -> _Kernel (running parallel kernels, in launch order)
_KID = itertools.count(1)
_INBOX = queue.Queue()
_PAR_LOCK = threading.RLock()
_KERNEL_ID = 0                            # in a subkernel: its $KernelID
_WORKER = None                            # in a subkernel: the _Worker
_SHARED_VARS = []                         # master: names of shared variables (in order)
_SHARED_FNS = []
_DIST = collections.OrderedDict()         # master: distributed symbol names
_NEEDS = []                               # master: contexts loaded with ParallelNeeds
_JOB_ID = itertools.count(1)
_BOOT = "from mikoshilang.library.wl_system import _subkernel_main; _subkernel_main()"


def _pkg_parent():
    from .. import __file__ as f
    return os.path.dirname(os.path.dirname(os.path.abspath(f)))


class _Kernel:
    """A local MikoshiLang subkernel process."""

    def __init__(self, kid, name="Local", inbox=None):
        self.id = kid
        self.name = name
        self.inbox = inbox if inbox is not None else _INBOX
        self.defs = {}
        self.job = None
        self.alive = True
        self.pid = None
        self.started = time.time()
        env = dict(os.environ)
        env["PYTHONPATH"] = _pkg_parent() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.p = subprocess.Popen([sys.executable, "-c", _BOOT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      env=env, cwd=os.getcwd())
        except OSError as e:
            raise ValueError(f"LaunchKernels: cannot start a local kernel: {e}") from None
        self.wlock = threading.Lock()
        self.t = threading.Thread(target=self._reader, daemon=True, name=f"w29-kernel-{kid}")
        self.t.start()

    def _reader(self):
        try:
            while True:
                try:
                    msg = _recv_msg(self.p.stdout)
                except Exception:  # noqa: BLE001 - broken pipe or unpicklable data
                    msg = None
                if msg is None:
                    break
                self.inbox.put((self, msg))
        finally:
            self.inbox.put((self, ("dead",)))

    def send(self, msg):
        try:
            _send_msg(self.p.stdin, self.wlock, msg)
        except (BrokenPipeError, OSError, ValueError):
            self.alive = False
            raise RuntimeError(f"parallel kernel {self.id} is no longer running") from None

    def expr(self):
        return Expr("KernelObject", self.id, self.name)

    def kill(self):
        self.alive = False
        try:
            self.p.kill()
        except OSError:
            pass
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        for f in (self.p.stdin, self.p.stdout):
            try:
                f.close()
            except (OSError, ValueError):
                pass

    def close(self):
        if self.alive:
            try:
                self.send(("quit",))
            except RuntimeError:
                pass
        try:
            self.p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.kill()


def _snap(name):
    """Non-destructive copy of every definition of a symbol (the state tuple used by SessionState.restore)."""
    st = K.state()
    exact = {k: v for k, v in st.exact.items() if K._def_key(v[0]) == name}
    ext = {k: d[name] for k, d in st.ext.items() if not k.startswith("\x00") and name in d}
    down = st.down.get(name)
    attrs = st.attrs.get(name)
    snap = (name in st.own, st.own.get(name), exact, list(down) if down else None,
            set(attrs) if attrs else None, ext)
    try:
        pickle.dumps(snap, protocol=4)
    except Exception:  # noqa: BLE001 - drop unpicklable extra data (e.g. Python callables)
        snap = snap[:5] + ({},)
        try:
            pickle.dumps(snap, protocol=4)
        except Exception as e:  # noqa: BLE001
            raise TypeError(f"DistributeDefinitions: the definitions of {name} cannot be sent to a parallel "
                            f"kernel: {e}") from None
    return snap


_EMPTY_SNAP = (False, None, {}, None, None, {})


def _snap_hash(snap):
    return hashlib.sha256(pickle.dumps(snap, protocol=4)).hexdigest()


def _has_defs(name):
    st = K.state()
    if name in st.own or name in st.down or st.attrs.get(name):
        return True
    if any(K._def_key(v[0]) == name for v in st.exact.values()):
        return True
    return any(name in d for k, d in st.ext.items() if not k.startswith("\x00") and k != "Unprotected")


def _is_user_symbol(name):
    if not name or name.startswith("$") or name in REGISTRY or name in K.BUILTIN_ATTRIBUTES:
        return False
    if name in E.SYMBOL_VALUES or name in ("True", "False", "Null", "None", "All", "Automatic"):
        return False
    return True


def _context_of(name):
    if "`" in name:
        return name.rsplit("`", 1)[0] + "`"
    return "Global`"


def _ctx_pred(setting, fname):
    v = setting
    if v is None or is_sym(v, "$DistributedContexts"):
        v = ev(Symbol("$DistributedContexts"))
    if is_sym(v, "All") or is_sym(v, "Automatic"):
        return lambda n: True
    if is_sym(v, "None"):
        return lambda n: False
    if is_sym(v, "$Context"):
        v = ev(Symbol("$Context"))
    if isinstance(v, str):
        ctxs = [v]
    elif is_list(v) and all(isinstance(c, str) for c in v.args):
        ctxs = list(v.args)
    else:
        raise ValueError(f"{fname}: DistributedContexts must be a context string, a list of contexts, All, None or "
                         f"Automatic, not {_form(v)}")
    for c in ctxs:
        if not c.endswith("`"):
            raise ValueError(f"{fname}: {c!r} is not a context (it must end in `)")
    return lambda n: _context_of(n) in ctxs


def _closure(roots, pred, explicit=()):
    """User symbols with definitions reachable from roots (through their definitions), filtered by pred; the
    names in explicit are always included (and followed)."""
    out = []
    seen = set()
    stack = list(roots)
    explicit = set(explicit)
    shared = set(_SHARED_VARS) | set(_SHARED_FNS)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n in shared or not _is_user_symbol(n):
            continue
        if not (n in explicit or pred(n)):
            continue
        if not _has_defs(n):
            continue
        out.append(n)
        st = K.state()
        refs = set()
        for c in st.definitions_of(n):
            K.symbol_names(c, refs)
        for k, d in st.ext.items():
            if not k.startswith("\x00") and n in d:
                try:
                    K.symbol_names(d[n], refs)
                except Exception:  # noqa: BLE001 - non-expression data
                    pass
        stack.extend(sorted(refs - seen))
    return out


def _names_in(*exprs):
    s = set()
    for e in exprs:
        K.symbol_names(e, s)
    return sorted(s)


def _push_defs(kernels, names):
    """Send the current definitions of names to kernels that do not have them yet."""
    if not names:
        return
    snaps = {n: _snap(n) for n in names}
    hashes = {n: _snap_hash(s) for n, s in snaps.items()}
    for k in kernels:
        upd = [(n, snaps[n]) for n in names if k.defs.get(n) != hashes[n]]
        if upd:
            k.send(("defs", upd))
            for n, _ in upd:
                k.defs[n] = hashes[n]


def _auto_distribute(kernels, exprs, setting, fname):
    pred = _ctx_pred(setting, fname)
    names = _closure(_names_in(*exprs), pred)
    for n in names:
        _DIST[n] = True
    _push_defs(kernels, names)
    _push_defs(kernels, [n for n in _DIST if n not in names and _has_defs(n)])


def _held_syms(names):
    return L(*(Expr("HoldForm", Symbol(n)) for n in names))


# ═══════════════════════════════════════════ parallel kernels: subkernel side ═══════════════════════════════════════════

class _SharedOwn(dict):
    """OwnValues of a subkernel: shared variables are read from and written to the master kernel."""

    def __init__(self, base, worker):
        super().__init__(base)
        self.w = worker

    def __contains__(self, k):
        return k in self.w.shared_vars or dict.__contains__(self, k)

    def __getitem__(self, k):
        if k in self.w.shared_vars:
            return self.w.call(("getvar", k))
        return dict.__getitem__(self, k)

    def get(self, k, d=None):
        if k in self.w.shared_vars:
            return self.w.call(("getvar", k))
        return dict.get(self, k, d)

    def __setitem__(self, k, v):
        if k in self.w.shared_vars:
            self.w.call(("setvar", k, v))
            return
        dict.__setitem__(self, k, v)

    def pop(self, k, *d):
        if k in self.w.shared_vars:
            self.w.call(("unsetvar", k))
            return d[0] if d else None
        return dict.pop(self, k, *d)

    def __delitem__(self, k):
        if k in self.w.shared_vars:
            self.w.call(("unsetvar", k))
            return
        dict.__delitem__(self, k)

    def setdefault(self, k, d=None):
        if k in self.w.shared_vars:
            v = self.w.call(("getvar", k))
            if v is None:
                self.w.call(("setvar", k, d))
                return d
            return v
        return dict.setdefault(self, k, d)


_EXC_TYPES = {c.__name__: c for c in (ValueError, TypeError, NotImplementedError, ArithmeticError, ZeroDivisionError,
                                      OverflowError, RecursionError, KeyError, IndexError, RuntimeError,
                                      MemoryError, AttributeError, LookupError)}


def _exc_info(e):
    return (type(e).__name__, str(e))


def _rebuild_exc(info, prefix=""):
    cls = _EXC_TYPES.get(info[0], RuntimeError)
    return cls(prefix + info[1])


class _Worker:
    def __init__(self, fin, fout):
        self.fin, self.fout = fin, fout
        self.wlock = threading.Lock()
        self.shared_vars = set()
        self.shared_fns = set()
        self.kcount = 0
        self.master_pid = None

    def send(self, msg):
        _send_msg(self.fout, self.wlock, msg)

    def call(self, msg):
        self.send(msg)
        r = _recv_msg(self.fin)
        if r is None:
            raise SystemExit(0)
        if r[0] != "reply":
            raise RuntimeError(f"unexpected message from the master kernel: {r[0]}")
        ok, val = r[1], r[2]
        if not ok:
            raise _rebuild_exc(val)
        return val

    def install(self, defs):
        st = K.state()
        for name, snap in defs:
            st.restore(name, snap)

    def run_item(self, x):
        from .wl_errors import AbortSignal
        out = io.StringIO()
        old = sys.stdout
        sys.stdout = out
        try:
            try:
                v = ev(x)
            finally:
                sys.stdout = old
            try:
                pickle.dumps(v, protocol=4)
            except Exception as e:  # noqa: BLE001
                return ("err", ("TypeError", f"the result cannot be returned to the master kernel: {e}"),
                        out.getvalue())
            return ("ok", v, out.getvalue())
        except AbortSignal:
            return ("abort", None, out.getvalue())
        except (SystemExit, KeyboardInterrupt):
            raise
        except Exception as e:  # noqa: BLE001 - every evaluation error is reported to the master
            return ("err", _exc_info(e), out.getvalue())
        except BaseException as e:  # noqa: BLE001 - Throw/Return signals and the like
            return ("err", ("RuntimeError", f"uncaught {type(e).__name__} in a parallel kernel"), out.getvalue())

    def loop(self):
        global _KERNEL_ID
        while True:
            msg = _recv_msg(self.fin)
            if msg is None:
                return
            kind = msg[0]
            if kind == "init":
                _, kid, mpid, svars, sfns, defs, needs = msg
                _KERNEL_ID = kid
                self.master_pid = mpid
                self.shared_vars, self.shared_fns = set(svars), set(sfns)
                self.install(defs)
                for c in needs:
                    try:
                        ev(Expr("Needs", c))
                    except Exception:  # noqa: BLE001 - reported again when used
                        pass
                self.send(("inited", kid))
            elif kind == "eval":
                _, jid, items, kcount = msg
                self.kcount = kcount
                res = [self.run_item(x) for x in items]
                self.send(("result", jid, res))
            elif kind == "defs":
                self.install(msg[1])
            elif kind == "share":
                self.shared_vars, self.shared_fns = set(msg[1]), set(msg[2])
            elif kind == "quit":
                return


_MUTATING = ()


def _worker_pre_hook(prev):
    def hook(head, args, attrs, ev_):
        w = _WORKER
        if w is not None and isinstance(head, str) and head in w.shared_fns:
            return w.call(("call", Expr(head, *args)))
        if prev is not None:
            return prev(head, args, attrs, ev_)
        return K.UNHANDLED
    return hook


def _worker_assign_hook(name):
    def hook(lhs, rhs, ev_):
        if isinstance(lhs, Expr) and isinstance(lhs.head, str) and not (_attrs_of(lhs.head) & {"HoldAll",
                                                                                               "HoldAllComplete"}):
            lhs = Expr(lhs.head, *(ev_(a) for a in lhs.args))     # f[i] = v defines f[<value of i>]
        _WORKER.call(("assign", lhs, rhs))
    return hook


def _subkernel_main():
    """Entry point of a parallel subkernel process (started by LaunchKernels / LocalEvaluate / LocalSubmit)."""
    global _WORKER
    fin = os.fdopen(os.dup(0), "rb")
    fout = os.fdopen(os.dup(1), "wb")
    nul = os.open(os.devnull, os.O_RDONLY)
    os.dup2(nul, 0)
    os.close(nul)
    os.dup2(2, 1)
    try:
        sys.stdin = open(os.devnull)
    except OSError:
        pass
    import mikoshilang  # noqa: F401 - registers the library
    from .. import library  # noqa: F401
    _EVAL_ENV.set("Subkernel")
    w = _Worker(fin, fout)
    _WORKER = w
    K._default_state.own = _SharedOwn(K._default_state.own, w)
    E.PRE_LISTABLE_HOOK = _worker_pre_hook(E.PRE_LISTABLE_HOOK)

    class _FnHooks(dict):
        def get(self, k, d=None):
            if isinstance(k, str) and k in w.shared_fns:
                return _worker_assign_hook(k)
            return dict.get(self, k, d)

        def __bool__(self):
            return True
    K.ASSIGN_HOOKS = _FnHooks(K.ASSIGN_HOOKS)
    w.send(("ready", os.getpid()))
    try:
        w.loop()
    finally:
        try:
            fout.flush()
        except OSError:
            pass


# ═══════════════════════════════════════════ parallel kernels: master side ═══════════════════════════════════════════

class _Job:
    def __init__(self, items, kernel=None, owner=None):
        self.id = next(_JOB_ID)
        self.items = items
        self.kernel = kernel          # required kernel (ParallelEvaluate) or None
        self.owner = owner
        self.state = "queued"
        self.outcome = None           # list of (status, value, printed)
        self.ran_on = None
        self.requeues = 0
        self.error = None
        self.seq = None


_QUEUE = collections.deque()
_SEQ = itertools.count()
_RUNNING = {}                         # kernel id -> _Job
_LOCK_OWNERS = {}                     # lock name -> owner ("k", id) or ("t", thread id)
_LOCK_WAIT = collections.deque()      # (kernel, names) waiting for locks
_LOCK_CV = threading.Condition()


def _kernel_count():
    if _WORKER is not None:
        return _WORKER.kcount
    return len(_KERNELS)


def _reply(k, ok, val):
    try:
        k.send(("reply", ok, val))
    except RuntimeError:
        pass


def _try_grant():
    with _LOCK_CV:
        for item in list(_LOCK_WAIT):
            k, names = item
            if all(n not in _LOCK_OWNERS for n in names):
                for n in names:
                    _LOCK_OWNERS[n] = ("k", k.id)
                _LOCK_WAIT.remove(item)
                _reply(k, True, None)


def _release_kernel_locks(kid):
    with _LOCK_CV:
        for n in [n for n, o in _LOCK_OWNERS.items() if o == ("k", kid)]:
            del _LOCK_OWNERS[n]
        for item in [i for i in _LOCK_WAIT if i[0].id == kid]:
            _LOCK_WAIT.remove(item)
        _LOCK_CV.notify_all()
    _try_grant()


def _service(k, msg):
    """Serve a request a subkernel makes while it evaluates (shared variables/functions, locks)."""
    kind = msg[0]
    try:
        if kind == "getvar":
            v = ev(Symbol(msg[1]))
            _reply(k, True, None if isinstance(v, Symbol) and v.name == msg[1] else v)
        elif kind == "setvar":
            K.state().own[msg[1]] = msg[2]
            _reply(k, True, None)
        elif kind == "unsetvar":
            K.state().own.pop(msg[1], None)
            _reply(k, True, None)
        elif kind == "call":
            x = msg[1]
            v = ev(x)
            if isinstance(v, Expr) and K.same(v, x):
                v = None
            _reply(k, True, v)
        elif kind == "assign":
            K._assign(msg[1], msg[2], ev)
            _reply(k, True, None)
        elif kind == "lock":
            with _LOCK_CV:
                _LOCK_WAIT.append((k, list(msg[1])))
            _try_grant()
        elif kind == "unlock":
            with _LOCK_CV:
                for n in msg[1]:
                    if _LOCK_OWNERS.get(n) == ("k", k.id):
                        del _LOCK_OWNERS[n]
                _LOCK_CV.notify_all()
            _reply(k, True, None)
            _try_grant()
        else:
            _reply(k, False, ("RuntimeError", f"unknown request {kind!r}"))
    except Exception as e:  # noqa: BLE001 - errors are re-raised in the subkernel
        _reply(k, False, _exc_info(e))


_CALLBACKS = ("getvar", "setvar", "unsetvar", "call", "assign", "lock", "unlock")


def _print_from(k, text):
    if text:
        from .wl_errors import print_line
        for line in text.rstrip("\n").split("\n"):
            print_line(f"(kernel {k.id}) {line}")


def _handle(k, msg):
    """Process one message from a parallel kernel (master thread)."""
    kind = msg[0]
    if kind in _CALLBACKS:
        _service(k, msg)
    elif kind == "result":
        job = _RUNNING.get(k.id)
        if job is not None and job.id == msg[1] and job.ran_on is k:
            del _RUNNING[k.id]
            job.outcome = msg[2]
            job.state = "finished"
            job.seq = next(_SEQ)
            for st, _v, printed in msg[2]:
                _print_from(k, printed)
    elif kind == "dead":
        _kernel_died(k)


def _kernel_died(k):
    k.alive = False
    if _KERNELS.get(k.id) is not k:
        return                        # a kernel already closed or replaced (AbortKernels, CloseKernels)
    _KERNELS.pop(k.id, None)
    _release_kernel_locks(k.id)
    job = _RUNNING.get(k.id)
    if job is not None and job.ran_on is k:
        del _RUNNING[k.id]
        if job.kernel is None and job.requeues < 1 and _sysopt("ParallelOptions", "RecoveryMode") == "ReQueue":
            job.requeues += 1
            job.state = "queued"
            _QUEUE.appendleft(job)
        else:
            job.state = "finished"
            job.error = RuntimeError(f"parallel kernel {k.id} terminated during the evaluation (Exit/Quit or a "
                                     "crash)")
    for j in list(_QUEUE):
        if j.kernel is k:
            _QUEUE.remove(j)
            j.state = "finished"
            j.error = RuntimeError(f"parallel kernel {k.id} is no longer running")
    if _sysopt("ParallelOptions", "RelaunchFailedKernels"):
        try:
            _launch(1, k.name)
        except Exception:  # noqa: BLE001 - a failed relaunch leaves one kernel fewer
            pass


def _schedule():
    for job in list(_QUEUE):
        if job.kernel is not None:
            k = job.kernel
            if not k.alive:
                continue
            if k.id in _RUNNING:
                continue
        else:
            free = [k for k in _KERNELS.values() if k.alive and k.id not in _RUNNING]
            if not free:
                continue
            k = free[0]
        _QUEUE.remove(job)
        _RUNNING[k.id] = job
        job.state = "running"
        job.ran_on = k
        try:
            k.send(("eval", job.id, job.items, len(_KERNELS)))
        except RuntimeError:
            _kernel_died(k)
    if not _KERNELS:
        for job in list(_QUEUE):
            if job.kernel is None:
                _QUEUE.remove(job)
                job.state = "finished"
                job.error = RuntimeError("no parallel kernels are running")


def _wait(done):
    """Run the scheduler until done() is true (master thread; aborts and time limits interrupt it)."""
    while True:
        _schedule()
        if done():
            return
        try:
            k, msg = _INBOX.get(timeout=0.05)
        except queue.Empty:
            continue
        _handle(k, msg)
        while True:
            try:
                k, msg = _INBOX.get_nowait()
            except queue.Empty:
                break
            _handle(k, msg)


def _abort_jobs(jobs):
    """Interrupted: drop queued jobs, restart the kernels running them."""
    for j in jobs:
        if j in _QUEUE:
            _QUEUE.remove(j)
            j.state = "finished"
            j.error = RuntimeError("aborted")
    busy = [kid for kid, j in _RUNNING.items() if j in jobs]
    for kid in busy:
        _restart_kernel(kid)


def _restart_kernel(kid):
    k = _KERNELS.get(kid)
    job = _RUNNING.pop(kid, None)
    if job is not None:
        job.state = "finished"
        job.outcome = [("abort", None, "")] * len(job.items)
    if k is None:
        return
    k.kill()
    _release_kernel_locks(kid)
    try:
        nk = _start_kernels([kid], k.name)[0]
    except Exception:  # noqa: BLE001 - the kernel is lost
        _KERNELS.pop(kid, None)
        return
    _KERNELS[kid] = nk


def _run(jobs):
    """Queue jobs, wait for all of them; returns their outcomes (raising the first error in job order)."""
    for j in jobs:
        _QUEUE.append(j)
    try:
        _wait(lambda: all(j.state == "finished" for j in jobs))
    except BaseException:
        _abort_jobs(jobs)
        raise
    return [_outcome(j) for j in jobs]


def _outcome(job):
    if job.error is not None:
        raise job.error
    vals = []
    for st, v, _p in job.outcome:
        if st == "ok":
            vals.append(v)
        elif st == "abort":
            vals.append(ABORTED)
        else:
            raise _rebuild_exc(v)
    return vals


# ── launching ──

def _start_kernels(ids, name, inbox=None, register=True, init_defs=None):
    timeout = _sysopt("ParallelOptions", "MathLinkTimeout")
    ks = [_Kernel(i, name, inbox) for i in ids]
    box = inbox if inbox is not None else _INBOX
    pending = {k.id: k for k in ks}
    deadline = time.monotonic() + timeout
    stash = []
    try:
        while pending:
            left = deadline - time.monotonic()
            if left <= 0:
                raise ValueError(f"LaunchKernels: a local kernel did not start within {timeout:g} seconds "
                                 "(SystemOptions[\"ParallelOptions\" -> \"MathLinkTimeout\"])")
            try:
                k, msg = box.get(timeout=min(left, 0.25))
            except queue.Empty:
                continue
            if k.id in pending and pending[k.id] is k:
                if msg[0] == "ready":
                    k.pid = msg[1]
                    defs = [(n, _snap(n)) for n in _DIST if _has_defs(n)] if register else (init_defs or [])
                    k.send(("init", k.id, os.getpid(), list(_SHARED_VARS) if register else [],
                            list(_SHARED_FNS) if register else [], defs, list(_NEEDS) if register else []))
                    for n, s in defs:
                        k.defs[n] = _snap_hash(s)
                elif msg[0] == "inited":
                    del pending[k.id]
                elif msg[0] == "dead":
                    raise ValueError("LaunchKernels: a local kernel exited during start-up (see its error output)")
            else:
                stash.append((k, msg))
    except BaseException:
        for k in ks:
            k.kill()
        raise
    finally:
        for item in stash:
            box.put(item)
    return ks


def _launch(n, name="Local"):
    if _WORKER is not None:
        raise ValueError("LaunchKernels: parallel kernels cannot be launched from a parallel subkernel")
    limit = int(_sysopt("ParallelOptions", "MaxKernelLaunchCount"))
    if len(_KERNELS) + n > limit:
        raise ValueError(f"LaunchKernels: launching {n} more kernels would exceed the limit of {limit} "
                         "(SystemOptions[\"ParallelOptions\" -> \"MaxKernelLaunchCount\"])")
    ks = _start_kernels([next(_KID) for _ in range(n)], name)
    for k in ks:
        _KERNELS[k.id] = k
    return ks


_REMOTE_PREFIXES = ("ssh://", "wstp://", "lwg://", "https://", "http://", "file://", "cloud")


def _spec_count(spec, fname):
    """Number of local kernels for a kernel specification."""
    if K.is_int(spec) and not isinstance(spec, bool):
        n = int(spec)
        if n < 0:
            raise ValueError(f"{fname}: the number of kernels must be non-negative, not {n}")
        return n
    if isinstance(spec, str):
        if spec in ("Local", "localhost", "local", "file://"):
            return _processor_count_value()
        if spec.startswith(_REMOTE_PREFIXES) or spec:
            raise NotImplementedError(f"{fname}: only local kernels can be launched (got {spec!r}); remote (ssh, "
                                      "wstp, lwg, cloud) and Wolfram kernels are not supported")
    if isinstance(spec, Expr) and spec.head == "KernelConfiguration":
        return _config_count(spec, fname)
    if isinstance(spec, Expr) and spec.head == "KernelObject":
        return 1
    if _is_symbolic(spec):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(spec)} is not a kernel specification")


def _processor_count_value():
    v = ev(Symbol("$ProcessorCount"))
    if K.is_int(v) and not isinstance(v, bool) and int(v) >= 0:
        return int(v)
    return _processor_count()


def _config_parts(c, fname):
    if not (isinstance(c, Expr) and c.head == "KernelConfiguration" and c.args):
        raise TypeError(f"{fname}: {_form(c)} is not a KernelConfiguration")
    a0 = c.args[0]
    if is_assoc(a0) and len(c.args) == 1:
        d = {k: v for k, v in _assoc_dict(a0, fname).items()}
        spec = d.pop("Name", "Local")
        return spec, d
    spec = a0
    d = {}
    for r in c.args[1:]:
        if not (isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2
                and isinstance(r.args[0], str)):
            raise TypeError(f"{fname}: {_form(r)} is not a setting \"name\" -> value of a KernelConfiguration")
        d[r.args[0]] = r.args[1]
    return spec, d


_KC_KEYS = ("KernelCount", "Name", "Class", "KernelCommand")


def _config_count(c, fname):
    spec, d = _config_parts(c, fname)
    if K.is_int(spec) and not isinstance(spec, bool):
        base = int(spec)
    elif isinstance(spec, str) and spec in ("Local", "localhost", "local", "file://"):
        base = None
    else:
        raise NotImplementedError(f"{fname}: only local kernel configurations can be launched (got "
                                  f"{_form(spec)})")
    for k in d:
        if k not in _KC_KEYS:
            raise NotImplementedError(f"{fname}: the kernel setting {k!r} is not supported (supported: "
                                      f"{', '.join(_KC_KEYS)})")
    cnt = d.get("KernelCount", base if base is not None else AUTOMATIC)
    if is_sym(cnt, "Automatic"):
        return _processor_count_value()
    if not K.is_int(cnt) or isinstance(cnt, bool) or int(cnt) < 0:
        raise ValueError(f"{fname}: \"KernelCount\" must be a non-negative integer or Automatic, not {_form(cnt)}")
    return int(cnt)


def _default_parallel_kernels():
    return L(Expr("KernelConfiguration", "Local", Expr("Rule", "KernelCount", AUTOMATIC)))


def _kernel_list(spec, fname):
    """Running kernels selected by All, a KernelObject or a list of them."""
    if spec is None or is_sym(spec, "All"):
        return list(_KERNELS.values())
    items = spec.args if is_list(spec) else (spec,)
    out = []
    for x in items:
        if isinstance(x, Expr) and x.head == "KernelObject" and x.args and K.is_int(x.args[0]):
            k = _KERNELS.get(int(x.args[0]))
            if k is None:
                raise ValueError(f"{fname}: {_form(x)} is not a running kernel")
            out.append(k)
        elif K.is_int(x) and not isinstance(x, bool):
            k = _KERNELS.get(int(x))
            if k is None:
                raise ValueError(f"{fname}: there is no running kernel with id {int(x)}")
            out.append(k)
        else:
            if _is_symbolic(x):
                raise Unevaluated()
            raise TypeError(f"{fname}: {_form(x)} is not a kernel")
    return out


@fn("LaunchKernels", area=AREA, status="added",
    notes="LaunchKernels[] launches the default parallel kernels ($DefaultParallelKernels; Automatic counts mean "
          "$ProcessorCount) when none are running; LaunchKernels[n] launches n more local kernels; "
          "LaunchKernels[\"Local\"|\"localhost\"], LaunchKernels[KernelConfiguration[\"Local\", \"KernelCount\" -> n]] "
          "and lists of these. Subkernels are separate local MikoshiLang processes (Python sys.executable) that "
          "receive distributed definitions, shared variables/functions and ParallelNeeds contexts; returns the new "
          "KernelObject[id, \"Local\"] list. LaunchKernels[] with kernels already running raises ValueError; remote "
          "specs (ssh://, wstp://, lwg://, file:// Wolfram kernels, cloud) raise NotImplementedError. Check: exact "
          "($KernelCount, distinct $ProcessID values) in tests.")
def _launch_kernels(spec: Expr = None) -> ListOf[Expr]:
    if spec is None:
        if _KERNELS:
            raise ValueError("LaunchKernels: some parallel kernels are already running; use LaunchKernels[n] to "
                             "launch more")
        spec = ev(Symbol("$DefaultParallelKernels"))
        if not is_list(spec):
            spec = L(spec)
    items = spec.args if is_list(spec) else (spec,)
    total = sum(_spec_count(x, "LaunchKernels") for x in items)
    with _PAR_LOCK:
        return L(*(k.expr() for k in _launch(total))) if total else L()


@fn("CloseKernels", area=AREA, status="added",
    notes="CloseKernels[] terminates all running parallel kernels; CloseKernels[k], CloseKernels[{k1, ...}] or "
          "CloseKernels[id] only those; gives the list of KernelObjects closed. Distributed definitions are kept "
          "and sent again to kernels launched later. Check: exact ($KernelCount afterwards) in tests.")
def _close_kernels(spec: Expr = None) -> ListOf[Expr]:
    with _PAR_LOCK:
        ks = _kernel_list(spec, "CloseKernels")
        for k in ks:
            _KERNELS.pop(k.id, None)
            job = _RUNNING.pop(k.id, None)
            if job is not None:
                job.state = "finished"
                job.error = RuntimeError(f"parallel kernel {k.id} was closed")
            k.close()
            _release_kernel_locks(k.id)
        return L(*(k.expr() for k in ks))


@fn("AbortKernels", area=AREA, status="added",
    notes="AbortKernels[] aborts the evaluations running on parallel kernels (their kernels are restarted with the "
          "same ids and the current distributed definitions) and discards the queued ParallelSubmit evaluations "
          "(they give $Aborted); gives the list of running KernelObjects. Check: exact (queued evaluations give "
          "$Aborted, kernels keep working) in tests.")
def _abort_kernels() -> ListOf[Expr]:
    with _PAR_LOCK:
        for j in list(_QUEUE):
            _QUEUE.remove(j)
            j.state = "finished"
            j.outcome = [("abort", None, "")] * len(j.items)
        for kid in list(_RUNNING):
            _restart_kernel(kid)
        for e in _EVALS.values():
            if e["job"] is None:
                e["job"] = _Job([e["expr"]])
                e["job"].state = "finished"
                e["job"].outcome = [("abort", None, "")]
        return L(*(k.expr() for k in _KERNELS.values()))


def _kernel_obj(x, fname):
    if not (isinstance(x, Expr) and x.head == "KernelObject" and x.args and K.is_int(x.args[0])):
        raise TypeError(f"{fname}: {_form(x)} is not a KernelObject")
    return int(x.args[0])


_KO_PROPS = ("KernelID", "Name", "Class", "KernelCommand", "ProcessID", "Running", "StartTime")


def _ko_props_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "KernelObject" and len(args) == 1):
        return K.UNHANDLED
    kid = _kernel_obj(head, "KernelObject")
    k = _KERNELS.get(kid)
    p = args[0]

    def val(name):
        if name == "KernelID":
            return kid
        if name == "Name":
            return head.args[1] if len(head.args) > 1 else "Local"
        if name == "Class":
            return "LocalKernels"
        if name == "KernelCommand":
            return sys.executable
        if name == "ProcessID":
            return k.pid if k is not None else Expr("Missing", "NotAvailable")
        if name == "Running":
            return k is not None and k.alive
        if name == "StartTime":
            return _date_object(k.started) if k is not None else Expr("Missing", "NotAvailable")
        raise ValueError(f"KernelObject: unknown property {name!r} (known: {', '.join(_KO_PROPS)})")
    if is_sym(p, "All"):
        return assoc([(n, val(n)) for n in _KO_PROPS])
    if p == "Properties":
        return L(*_KO_PROPS)
    if not isinstance(p, str):
        return K.UNHANDLED
    return val(p)


KL.COMPOUND_HEAD_HOOKS["KernelObject"] = _ko_props_hook
record("KernelObject", AREA, "added",
       "KernelObject[id, \"Local\"] represents a running parallel subkernel (from LaunchKernels, Kernels[] or "
       "ParallelKernels[]); ko[\"prop\"] gives \"KernelID\", \"Name\", \"Class\", \"KernelCommand\", \"ProcessID\", "
       "\"Running\", \"StartTime\"; ko[All] an association and ko[\"Properties\"] the names. Check: exact "
       "(ProcessID matches ParallelEvaluate[$ProcessID, ko]) in tests.")


@fn("ParallelKernels", area=AREA, status="added",
    notes="ParallelKernels[] gives the KernelObjects of the running parallel kernels in launch order (same as "
          "Kernels[]); ParallelKernels[\"OneKernelPerMachine\"] the first kernel (all kernels are local). Check: "
          "exact (after LaunchKernels/CloseKernels) in tests.")
def _parallel_kernels(sel: String = None) -> ListOf[Expr]:
    ks = list(_KERNELS.values())
    if sel is not None:
        s = _string(sel, "ParallelKernels", "the selector")
        if s != "OneKernelPerMachine":
            raise ValueError(f"ParallelKernels: unknown selector {s!r} (\"OneKernelPerMachine\")")
        ks = ks[:1]
    return L(*(k.expr() for k in ks))


@fn("Kernels", area=AREA, status="added",
    notes="Kernels[] gives the list of running parallel KernelObjects in launch order (supporting head used by "
          "ParallelEvaluate/CloseKernels). Check: exact in tests.")
def _kernels_list() -> ListOf[Expr]:
    return L(*(k.expr() for k in _KERNELS.values()))


record("KernelConfiguration", AREA, "added",
       "KernelConfiguration[\"Local\", \"KernelCount\" -> n] (or KernelConfiguration[n], KernelConfiguration[<|\"Name\" "
       "-> \"Local\", \"KernelCount\" -> n|>]) describes local parallel kernels for LaunchKernels, "
       "$DefaultParallelKernels and $ConfiguredKernels; \"KernelCount\" -> Automatic means $ProcessorCount. Other "
       "settings and remote specifications raise NotImplementedError when launched. kc[\"prop\"] reads a setting. "
       "Check: exact (number of kernels launched) in tests.")


def _kc_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "KernelConfiguration" and len(args) == 1
            and isinstance(args[0], str)):
        return K.UNHANDLED
    spec, d = _config_parts(head, "KernelConfiguration")
    p = args[0]
    if p == "KernelCount":
        return _config_count(head, "KernelConfiguration")
    if p == "Name":
        return spec if isinstance(spec, str) else "Local"
    if p == "Class":
        return "LocalKernels"
    if p in d:
        return d[p]
    raise ValueError(f"KernelConfiguration: unknown setting {p!r}")


KL.COMPOUND_HEAD_HOOKS["KernelConfiguration"] = _kc_hook


def _ensure_kernels():
    """Running kernels for a parallel command: launches the default kernels if none run; [] means evaluate on the
    master (inside a subkernel, or when no kernels are configured)."""
    if _WORKER is not None:
        return []
    if not _KERNELS:
        spec = ev(Symbol("$DefaultParallelKernels"))
        items = spec.args if is_list(spec) else (spec,)
        total = sum(_spec_count(x, "LaunchKernels") for x in items)
        if total:
            _launch(total)
    return list(_KERNELS.values())


_sysvar("$KernelID", lambda: _KERNEL_ID, "the id of the parallel kernel the evaluation runs on: 0 on the master "
        "kernel, 1, 2, ... (in launch order) on subkernels", check="exact (ParallelEvaluate gives the kernel ids) in tests")
_sysvar("$KernelCount", _kernel_count, "the number of running parallel kernels (0 before any are launched); in a "
        "subkernel the master's count, as documented", check="exact (after LaunchKernels/CloseKernels) in tests")
_sysvar("$DefaultParallelKernels", _default_parallel_kernels, "the kernels launched by LaunchKernels[] and "
        "automatically by the first parallel command: {KernelConfiguration[\"Local\", \"KernelCount\" -> Automatic]} "
        "($ProcessorCount local kernels); assign e.g. {2} or {} (no kernels: parallel commands then evaluate on "
        "the master). The value is not remembered between sessions", protected=False,
        check="exact (LaunchKernels[] launches the configured number) in tests")
_sysvar("$ConfiguredKernels", lambda: ev(Symbol("$DefaultParallelKernels")), "the kernel configurations "
        "available: by default the value of $DefaultParallelKernels; assignments are kept for the session only",
        protected=False,
        check='exact (follows $DefaultParallelKernels) in tests')
_sysvar("$DefaultLocalKernel", lambda: "Local", "\"Local\", the kernel used by LocalEvaluate and LocalSubmit (only "
        "local kernels are supported); assignments are kept for the session only", protected=False,
        check='exact (default "Local"; Block assignment) in tests')
_sysvar("$DefaultRemoteKernel", lambda: "localhost", "\"localhost\", the default remote kernel specification; "
        "assignments are kept for the session only (remote evaluation is not supported)", protected=False,
        check='exact (default "localhost"; Block assignment) in tests')
_sysvar("$DistributedContexts", lambda: ev(Symbol("$Context")), "the contexts whose definitions parallel commands "
        "distribute automatically: by default the value of $Context (\"Global`\"); All, None, a context or a list "
        "of contexts may be assigned", protected=False,
        check="exact (ctx` symbols are not distributed by default) in tests")
_sysvar("$DistributedDefinitions", lambda: _held_syms(list(_DIST)), "the symbols whose definitions have been "
        "distributed to parallel kernels, as HoldForm[s] in order of distribution",
        check="exact (after DistributeDefinitions/ClearDistributedDefinitions) in tests")
_sysvar("$SharedVariables", lambda: L(*(Expr("Hold", Symbol(n)) for n in _SHARED_VARS)), "the list Hold[x] of "
        "variables shared with SetSharedVariable", check="exact (SetSharedVariable/UnsetShared) in tests")
_sysvar("$SharedFunctions", lambda: L(*(Expr("Hold", Symbol(n)) for n in _SHARED_FNS)), "the list Hold[f] of "
        "functions shared with SetSharedFunction", check="exact (SetSharedFunction/UnsetShared) in tests")


# ═══════════════════════════════════════════ parallel commands ═══════════════════════════════════════════

_PAR_OPTS = ("Method", "DistributedContexts", "ProgressReporting")


def _chunk_ranges(n, nk, method, fname):
    if n == 0:
        return []
    m = method
    size = None
    if m is None or is_sym(m, "Automatic"):
        c = min(n, 2 * max(nk, 1))
    elif m == "FinestGrained":
        c = n
    elif m == "CoarsestGrained":
        c = min(n, max(nk, 1))
    elif isinstance(m, Expr) and m.head in ("Rule", "RuleDelayed") and m.args[0] == "EvaluationsPerKernel":
        e = _pos_int(m.args[1], fname, "EvaluationsPerKernel")
        c = min(n, e * max(nk, 1))
    elif isinstance(m, Expr) and m.head in ("Rule", "RuleDelayed") and m.args[0] == "ItemsPerEvaluation":
        size = _pos_int(m.args[1], fname, "ItemsPerEvaluation")
        c = None
    elif is_list(m) and len(m.args) == 1:
        return _chunk_ranges(n, nk, m.args[0], fname)
    else:
        raise ValueError(f"{fname}: Method must be Automatic, \"FinestGrained\", \"CoarsestGrained\", "
                         f"\"EvaluationsPerKernel\" -> e or \"ItemsPerEvaluation\" -> m, not {_form(m)}")
    out = []
    if size is not None:
        for a in range(0, n, size):
            out.append((a, min(n, a + size)))
        return out
    a = 0
    for i in range(c):
        s = n // c + (1 if i < n % c else 0)
        out.append((a, a + s))
        a += s
    return out


def _par_eval(items, o, fname, chunked=True, extra=()):
    """Evaluate the expressions items in parallel (in order); on the master if no kernels are available."""
    with _PAR_LOCK:
        ks = _ensure_kernels()
        if not ks:
            return [ev(x) for x in items]
        _auto_distribute(ks, list(items) + list(extra), o.get("DistributedContexts"), fname)
        ranges = _chunk_ranges(len(items), len(ks), o.get("Method") if chunked else "FinestGrained", fname)
        outs = _run([_Job(list(items[a:b])) for a, b in ranges])
    return [v for chunk in outs for v in chunk]


@fn("ParallelEvaluate", area=AREA, status="added",
    notes="ParallelEvaluate[expr] evaluates expr on every parallel kernel and gives the list of results (each "
          "evaluated again on the master, as in Wolfram); ParallelEvaluate[expr, k] on one KernelObject (a single "
          "result), ParallelEvaluate[expr, {k1, ...}] on several, ParallelEvaluate[expr, kernels, h] wraps the "
          "results in h before they are evaluated on the master (h = Hold prevents it). Option "
          "DistributedContexts (default $DistributedContexts). Kernels are launched if none run; inside a subkernel "
          "or with no kernels configured it evaluates once on the master and gives {result}. HoldFirst. Check: exact "
          "($KernelID values, distinct $ProcessID values) in tests.")
def _parallel_evaluate(expr: Held, *rest: Expr | Option) -> Expr:
    args, o = _opts(rest, ("DistributedContexts",), "ParallelEvaluate")
    if len(args) > 2:
        raise TypeError("ParallelEvaluate: called with too many arguments")
    spec = args[0] if args else ALL
    h = args[1] if len(args) > 1 else None
    single = isinstance(spec, Expr) and spec.head == "KernelObject" or (K.is_int(spec) and not isinstance(spec, bool))
    with _PAR_LOCK:
        ks = _ensure_kernels() if (spec is ALL or is_sym(spec, "All")) else _kernel_list(spec, "ParallelEvaluate")
        if not ks:
            vals = [ev(expr)]
        else:
            _auto_distribute(ks, [expr], o.get("DistributedContexts"), "ParallelEvaluate")
            outs = _run([_Job([expr], kernel=k) for k in ks])
            vals = [v[0] for v in outs]
    if h is not None:
        vals = [K.call(h, v) for v in vals]
    if single:
        return ev(vals[0])
    return ev(L(*vals))


K.BUILTIN_ATTRIBUTES["ParallelEvaluate"] = {"HoldFirst", "Protected"}


def _level1(lev):
    return (K.is_int(lev) and not isinstance(lev, bool) and int(lev) == 1) or (
        is_list(lev) and len(lev.args) == 1 and K.is_int(lev.args[0]) and int(lev.args[0]) == 1)


@fn("ParallelMap", area=AREA, status="added",
    notes="ParallelMap[f, expr] applies f to the elements of expr (any head, or the values of an association) in "
          "parallel, chunked by Method (Automatic: two evaluations per kernel, \"FinestGrained\", "
          "\"CoarsestGrained\", \"EvaluationsPerKernel\" -> e, \"ItemsPerEvaluation\" -> m); definitions of f and of "
          "the elements are distributed automatically (DistributedContexts). ParallelMap[f, expr, levelspec] "
          "parallelizes level 1 ({1} or 1) and evaluates other level specifications with Map on the master. "
          "Check: reference (Map gives the same result) in tests.")
def _parallel_map(f: Expr, expr: Expr, *rest: Expr | Option) -> Expr:
    args, o = _opts(rest, _PAR_OPTS, "ParallelMap")
    if len(args) > 1:
        raise TypeError("ParallelMap: called with too many arguments")
    if args and not _level1(args[0]):
        return ev(Expr("Map", f, expr, args[0]))
    if is_assoc(expr):
        keys = [r.args[0] for r in expr.args]
        vals = _par_eval([K.call(f, r.args[1]) for r in expr.args], o, "ParallelMap", extra=[f])
        return ev(Expr("Association", *(Expr(r.head, k, v) for r, k, v in zip(expr.args, keys, vals))))
    if not isinstance(expr, Expr):
        return expr
    vals = _par_eval([K.call(f, a) for a in expr.args], o, "ParallelMap", extra=[f])
    return ev(Expr(expr.head, *vals))


def _split_iter_opts(args, fname):
    args = list(args)
    o = {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") and \
            isinstance(args[-1].args[0], Symbol) and args[-1].args[0].name in _PAR_OPTS:
        r = args.pop()
        o.setdefault(r.args[0].name, ev(r.args[1]))
    return args, o


def _iter_items(expr, iters, fname, wrap="Table"):
    """(items, ok): one held item per value of the outer iterator, each evaluating wrap[expr, inner iterators]."""
    if not iters:
        raise TypeError(f"{fname}: an iterator specification is expected")
    it, rest = iters[0], iters[1:]
    inner = Expr(wrap, expr, *rest) if rest else expr
    if not is_list(it):
        n = ev(it)
        if not K.is_int(n):
            return None
        return [inner] * max(0, int(n))
    if len(it.args) == 1:
        n = ev(it.args[0])
        if not K.is_int(n):
            return None
        return [inner] * max(0, int(n))
    var = it.args[0]
    try:
        vals = ev(Expr("Table", var, it))
    except Exception:  # noqa: BLE001 - symbolic iterator bounds: evaluate sequentially
        return None
    if not is_list(vals):
        return None
    return [Expr("Block", L(Expr("Set", var, v)), inner) for v in vals.args]


@fn("ParallelTable", area=AREA, status="added",
    notes="ParallelTable[expr, iter1, iter2, ...] builds Table[expr, iter1, ...] in parallel: the values of the "
          "outer iterator are computed on the master and the parts are evaluated on the kernels (inside Block with "
          "the iterator variable set), chunked by Method; options Method, DistributedContexts. Iterators with "
          "symbolic bounds are evaluated with Table on the master. HoldAll. Check: reference (Table gives the same "
          "result) in tests.")
def _parallel_table(expr: Held, *iters: Held) -> ListOf[Expr]:
    its, o = _split_iter_opts(iters, "ParallelTable")
    items = _iter_items(expr, its, "ParallelTable")
    if items is None:
        return ev(Expr("Table", expr, *its))
    return ev(L(*_par_eval(items, o, "ParallelTable")))


@fn("ParallelSum", area=AREA, status="added",
    notes="ParallelSum[expr, iter1, ...] computes Sum[expr, iter1, ...] by summing the parts for each value of the "
          "outer iterator in parallel (then Plus on the master); iterators that cannot be enumerated (symbolic "
          "bounds, infinite sums) are given to Sum on the master. HoldAll; options Method, DistributedContexts. "
          "Check: reference (Sum gives the same result) in tests.")
def _parallel_sum(expr: Held, *iters: Held) -> Expr:
    its, o = _split_iter_opts(iters, "ParallelSum")
    items = _iter_items(expr, its, "ParallelSum", "Sum")
    if items is None:
        return ev(Expr("Sum", expr, *its))
    return ev(Expr("Plus", *_par_eval(items, o, "ParallelSum"))) if items else 0


@fn("ParallelProduct", area=AREA, status="added",
    notes="ParallelProduct[expr, iter1, ...] computes Product[expr, iter1, ...] by multiplying the parts for each "
          "value of the outer iterator computed in parallel; non-enumerable iterators use Product on the master. "
          "HoldAll; options Method, DistributedContexts. Check: reference (Product gives the same result) in tests.")
def _parallel_product(expr: Held, *iters: Held) -> Expr:
    its, o = _split_iter_opts(iters, "ParallelProduct")
    items = _iter_items(expr, its, "ParallelProduct", "Product")
    if items is None:
        return ev(Expr("Product", expr, *its))
    return ev(Expr("Times", *_par_eval(items, o, "ParallelProduct"))) if items else 1


@fn("ParallelDo", area=AREA, status="added",
    notes="ParallelDo[expr, iter1, ...] evaluates expr for each iterator value in parallel (for side effects such as "
          "shared variables, shared functions or files) and returns Null; non-enumerable iterators use Do on the "
          "master. HoldAll; options Method, DistributedContexts. Check: exact (a shared variable collects the "
          "values) in tests.")
def _parallel_do(expr: Held, *iters: Held) -> Expr:
    its, o = _split_iter_opts(iters, "ParallelDo")
    items = _iter_items(expr, its, "ParallelDo", "Do")
    if items is None:
        ev(Expr("Do", expr, *its))
        return NULL
    _par_eval([Expr("CompoundExpression", x, NULL) for x in items], o, "ParallelDo")
    return NULL


for _h in ("ParallelTable", "ParallelSum", "ParallelProduct", "ParallelDo"):
    K.BUILTIN_ATTRIBUTES[_h] = {"HoldAll", "Protected"}


def _dims(n, fname):
    if is_list(n):
        return [_nonneg_int(x, fname, "a dimension") for x in n.args]
    return [_nonneg_int(n, fname, "the dimension")]


def _reshape(flat, dims, h):
    if len(dims) == 1:
        return Expr(h, *flat) if isinstance(h, str) else K.call(h, *flat)
    step = len(flat) // dims[0] if dims[0] else 0
    parts = [_reshape(flat[i * step:(i + 1) * step], dims[1:], h) for i in range(dims[0])]
    return Expr(h, *parts) if isinstance(h, str) else K.call(h, *parts)


@fn("ParallelArray", area=AREA, status="added",
    notes="ParallelArray[f, n] gives {f[1], ..., f[n]} computed in parallel; ParallelArray[f, {n1, n2, ...}] an "
          "n1 x n2 x ... array of f[i1, i2, ...]; ParallelArray[f, dims, r] starts the indices at r (or {r1, ...}); "
          "ParallelArray[f, dims, r, h] uses head h at every level. Options Method, DistributedContexts. Check: "
          "reference (Array gives the same result) in tests.")
def _parallel_array(f: Expr, n: PosInteger | ListOf[PosInteger], r: Integer | ListOf[Integer] = None,
                    h: Expr = None, *opts: Option) -> Expr:
    pos = [x for x in (r, h) if x is not None]
    pos, o = _opts(pos + list(opts), _PAR_OPTS, "ParallelArray")
    r = pos[0] if pos else None
    h = pos[1] if len(pos) > 1 else None
    dims = _dims(n, "ParallelArray")
    if r is None:
        origins = [1] * len(dims)
    elif is_list(r):
        origins = [int(x) for x in r.args]
        if len(origins) != len(dims) or not all(K.is_int(x) for x in r.args):
            raise ValueError("ParallelArray: the index origins must be integers, one for each dimension")
    elif K.is_int(r) and not isinstance(r, bool):
        origins = [int(r)] * len(dims)
    else:
        raise TypeError(f"ParallelArray: the index origin must be an integer or a list of integers, not {_form(r)}")
    idx = list(itertools.product(*(range(a, a + d) for a, d in zip(origins, dims))))
    vals = _par_eval([K.call(f, *t) for t in idx], o, "ParallelArray", extra=[f])
    head = "List" if h is None else (h.name if isinstance(h, Symbol) else h)
    if any(d == 0 for d in dims):
        return ev(Expr(head, *[]) if isinstance(head, str) else K.call(head))
    return ev(_reshape(vals, dims, head))


def _chunked_list(lst, o, fname):
    ks = len(_KERNELS) or 1
    return [list(lst[a:b]) for a, b in _chunk_ranges(len(lst), ks, o.get("Method"), fname)]


def _take(vals, n):
    out = []
    for v in vals:
        out.extend(v.args if is_list(v) else (v,))
    return out if n is None else out[:n]


@fn("ParallelCases", area=AREA, status="added",
    notes="ParallelCases[list, patt] gives the elements of list (any head) that match patt (or patt -> rhs "
          "transformed), searching chunks in parallel; ParallelCases[list, patt, levelspec, n] at most n; level "
          "specifications other than {1} use Cases on the master. Options Method, DistributedContexts. Check: "
          "reference (Cases gives the same result) in tests.")
def _parallel_cases(expr: Expr, patt: Expr, *rest: Expr | Option) -> ListOf[Expr]:
    args, o = _opts(rest, _PAR_OPTS, "ParallelCases")
    if len(args) > 2:
        raise TypeError("ParallelCases: called with too many arguments")
    lev = args[0] if args else L(1)
    n = _nonneg_int(args[1], "ParallelCases", "the count") if len(args) > 1 and not is_sym(args[1], "Infinity") \
        else None
    if not _level1(lev):
        return ev(Expr("Cases", expr, patt, lev, *(args[1:] or [])))
    if not isinstance(expr, Expr):
        return L()
    elems = [r.args[1] for r in expr.args] if is_assoc(expr) else list(expr.args)
    with _PAR_LOCK:
        _ensure_kernels()
        chunks = _chunked_list(elems, o, "ParallelCases")
    vals = _par_eval([Expr("Cases", L(*c), patt) for c in chunks], o, "ParallelCases", chunked=False)
    return L(*_take(vals, n))


@fn("ParallelSelect", area=AREA, status="added",
    notes="ParallelSelect[list, crit] gives the elements of list for which crit gives True, testing chunks in "
          "parallel; ParallelSelect[list, crit, n] the first n. Options Method, DistributedContexts. Check: "
          "reference (Select gives the same result) in tests.")
def _parallel_select(lst: Expr, crit: Expr, *rest: Integer | Option) -> Expr:
    args, o = _opts(rest, _PAR_OPTS, "ParallelSelect")
    if len(args) > 1:
        raise TypeError("ParallelSelect: called with too many arguments")
    n = _nonneg_int(args[0], "ParallelSelect", "the count") if args and not is_sym(args[0], "Infinity") else None
    if not isinstance(lst, Expr) or is_assoc(lst):
        return ev(Expr("Select", lst, crit, *args))
    with _PAR_LOCK:
        _ensure_kernels()
        chunks = _chunked_list(list(lst.args), o, "ParallelSelect")
    vals = _par_eval([Expr("Select", L(*c), crit) for c in chunks], o, "ParallelSelect", chunked=False, extra=[crit])
    return ev(Expr(lst.head, *_take(vals, n)))


@fn("ParallelCombine", area=AREA, status="added",
    notes="ParallelCombine[f, h[e1, ..., en]] evaluates f[h[ei, ...]] on chunks of the elements in parallel and "
          "combines the results with h if h is Flat (Plus, Times, ...) or Join otherwise; ParallelCombine[f, "
          "h[...], comb] combines with comb. Chunks follow Method (Automatic: two per kernel). Check: reference "
          "(the sequential f[h[...]]) in tests.")
def _parallel_combine(f: Expr, expr: Expr, comb: Expr = None, *opts: Option) -> Expr:
    pos = [comb] if comb is not None else []
    pos, o = _opts(pos + list(opts), _PAR_OPTS, "ParallelCombine")
    comb = pos[0] if pos else None
    if not isinstance(expr, Expr) or is_assoc(expr):
        return ev(K.call(f, expr))
    h = expr.head
    if comb is None:
        attrs = K.BUILTIN_ATTRIBUTES.get(h, set()) | set(K.state().attrs.get(h, ())) if isinstance(h, str) else set()
        comb = Symbol(h) if "Flat" in attrs else Symbol("Join")
    with _PAR_LOCK:
        _ensure_kernels()
        chunks = _chunked_list(list(expr.args), o, "ParallelCombine")
    vals = _par_eval([K.call(f, Expr(h, *c)) for c in chunks], o, "ParallelCombine", chunked=False, extra=[f])
    return ev(K.call(comb, *vals))


@fn("ParallelTry", area=AREA, status="added",
    notes="ParallelTry[f, {a1, a2, ...}] evaluates f[ai] in parallel and gives the first result (in order of "
          "completion) that is not $Failed, or $Failed if all fail; ParallelTry[f, list, k] gives a list of the first "
          "k such results (shorter if fewer exist). Evaluations still running when enough results are in are "
          "aborted (their kernels restarted). Without kernels the ai are tried in order. Options "
          "DistributedContexts. Check: exact (a single non-$Failed candidate) in tests.")
def _parallel_try(f: Expr, lst: ListOf[Expr], k: PosInteger = None, *opts: Option) -> Expr:
    pos = [k] if k is not None else []
    pos, o = _opts(pos + list(opts), ("DistributedContexts", "Method"), "ParallelTry")
    k = pos[0] if pos else None
    want = 1 if k is None else _pos_int(k, "ParallelTry", "the number of results")
    if not isinstance(lst, Expr):
        raise TypeError(f"ParallelTry: {_form(lst)} is not a list")
    items = [K.call(f, a) for a in lst.args]
    found = []
    with _PAR_LOCK:
        ks = _ensure_kernels()
        if not ks:
            for x in items:
                v = ev(x)
                if not is_sym(v, "$Failed"):
                    found.append(v)
                    if len(found) >= want:
                        break
        else:
            _auto_distribute(ks, items + [f], o.get("DistributedContexts"), "ParallelTry")
            jobs = [_Job([x]) for x in items]
            for j in jobs:
                _QUEUE.append(j)

            def good():
                return [j for j in jobs if j.state == "finished" and j.error is None and j.outcome
                        and j.outcome[0][0] == "ok" and not is_sym(j.outcome[0][1], "$Failed")]
            try:
                _wait(lambda: len(good()) >= want or all(j.state == "finished" for j in jobs))
            finally:
                _abort_jobs([j for j in jobs if j.state != "finished"])
            for j in jobs:
                if j.state == "finished" and j.error is None and j.outcome and j.outcome[0][0] == "err":
                    raise _rebuild_exc(j.outcome[0][1])
            found = [j.outcome[0][1] for j in sorted(good(), key=lambda j: j.seq)][:want]
    if k is None:
        return ev(found[0]) if found else FAILED
    return ev(L(*found))


def _attrs_of(h):
    if not isinstance(h, str):
        return set()
    return set(K.BUILTIN_ATTRIBUTES.get(h, set())) | set(K.state().attrs.get(h, ()))


def _parallelize(x, o):
    """Parallel evaluation of a held expression; falls back to ordinary evaluation."""
    if not isinstance(x, Expr) or not isinstance(x.head, str):
        return ev(x)
    h, a = x.head, x.args
    opts = [Expr("Rule", Symbol(k), v) for k, v in o.items()]
    if h == "CompoundExpression":
        v = NULL
        for part in a:
            v = _parallelize(part, o)
        return v
    if h in ("Set", "SetDelayed") and len(a) == 2 and h == "Set":
        return ev(Expr("Set", a[0], Expr("Parallelize", a[1], *opts)))
    if h == "Map" and len(a) == 2:
        return ev(Expr("ParallelMap", a[0], a[1], *opts))
    if h in ("Table", "Sum", "Product", "Do") and len(a) >= 2:
        return ev(Expr("Parallel" + h, *a, *opts))
    if h == "Array" and 2 <= len(a) <= 4:
        return ev(Expr("ParallelArray", *a, *opts))
    if h == "Select" and 2 <= len(a) <= 3:
        return ev(Expr("ParallelSelect", *a, *opts))
    if h == "Cases" and len(a) == 2:
        return ev(Expr("ParallelCases", *a, *opts))
    if h == "Count" and len(a) == 2:
        lst = ev(a[0])
        return ev(Expr("ParallelCombine", Expr("Function", Expr("Count", Expr("Slot", 1), a[1])), lst,
                       Symbol("Plus"), *opts)) if is_list(lst) and lst.args else ev(Expr("Count", lst, a[1]))
    if h == "MapIndexed" and len(a) == 2:
        lst = ev(a[1])
        if is_list(lst):
            items = [K.call(a[0], e, L(i + 1)) for i, e in enumerate(lst.args)]
            return ev(L(*_par_eval(items, o, "Parallelize", extra=[a[0]])))
        return ev(Expr("MapIndexed", a[0], lst))
    if h == "MapThread" and len(a) == 2:
        lsts = ev(a[1])
        if is_list(lsts) and lsts.args and all(is_list(l) for l in lsts.args) and \
                len({len(l.args) for l in lsts.args}) == 1:
            n = len(lsts.args[0].args)
            items = [K.call(a[0], *(l.args[i] for l in lsts.args)) for i in range(n)]
            return ev(L(*_par_eval(items, o, "Parallelize", extra=[a[0]])))
        return ev(Expr("MapThread", a[0], lsts))
    if h == "Scan" and len(a) == 2:
        lst = ev(a[1])
        if isinstance(lst, Expr) and not is_assoc(lst):
            _par_eval([Expr("CompoundExpression", K.call(a[0], e), NULL) for e in lst.args], o, "Parallelize",
                      extra=[a[0]])
            return NULL
        return ev(Expr("Scan", a[0], lst))
    if h == "Apply" and len(a) == 3 and _level1(a[2]):
        lst = ev(a[1])
        if isinstance(lst, Expr) and not is_assoc(lst):
            items = [Expr("Apply", a[0], e) for e in lst.args]
            return ev(Expr(lst.head, *_par_eval(items, o, "Parallelize", extra=[a[0]])))
        return ev(Expr("Apply", a[0], lst, a[2]))
    if len(a) == 1 and "Listable" in _attrs_of(h) and "HoldAll" not in _attrs_of(h):
        arg = ev(a[0])
        if is_list(arg) and len(arg.args) > 1:
            return ev(Expr("ParallelCombine", Symbol(h), arg, *opts))
        return ev(Expr(h, arg))
    return ev(x)


@fn("Parallelize", area=AREA, status="added",
    notes="Parallelize[expr] evaluates expr using the parallel kernels when its form is recognized: Map, Table, "
          "Sum, Product, Do, Array, Select, Cases, Count, MapIndexed, MapThread, Scan, Apply at level {1}, a "
          "Listable function of one list, compound expressions (each part) and x = expr; any other expression is "
          "evaluated on the master, as Wolfram does for expressions that cannot be parallelized. HoldFirst; options "
          "Method, DistributedContexts, ProgressReporting. Check: reference (the sequential evaluation gives the "
          "same result) in tests.")
def _parallelize_fn(expr: Held, *opts: Option) -> Expr:
    _, o = _opts(opts, _PAR_OPTS, "Parallelize")
    return _parallelize(expr, o)


K.BUILTIN_ATTRIBUTES["Parallelize"] = {"HoldFirst", "Protected"}

# ── queued evaluations ──

_EVALS = {}                    # id -> {"expr", "job", "state"}
_EID = itertools.count(1)


def _substitute(expr, names, values):
    if isinstance(expr, Symbol) and expr.name in names:
        return values[names.index(expr.name)]
    if isinstance(expr, Expr):
        h = _substitute(expr.head, names, values) if isinstance(expr.head, Expr) else expr.head
        if isinstance(h, str) and h in names:
            h = values[names.index(h)]
            h = h.name if isinstance(h, Symbol) else h
        return Expr(h, *(_substitute(x, names, values) for x in expr.args))
    return expr


@fn("ParallelSubmit", area=AREA, status="added",
    notes="ParallelSubmit[expr] queues expr for evaluation on a parallel kernel and gives EvaluationObject[id, "
          "expr] at once; ParallelSubmit[{x, y, ...}, expr] first inserts the current values of x, y, ... into expr "
          "(as With). Queued evaluations are sent to kernels by WaitAll/WaitNext; definitions are not distributed "
          "automatically (use DistributeDefinitions). HoldAllComplete. eo[\"Process\"|\"State\"|\"Result\"|All|"
          "\"Properties\"] inspects an EvaluationObject. Check: exact (WaitAll results) in tests.")
def _parallel_submit(a: Held, b: Held = None) -> Expr:
    if b is None:
        expr = a
    else:
        if not (is_list(a) and all(isinstance(s, Symbol) for s in a.args)):
            raise TypeError(f"ParallelSubmit: {_form(a)} is not a list of variables")
        names = [s.name for s in a.args]
        expr = _substitute(b, names, [ev(s) for s in a.args])
    i = next(_EID)
    _EVALS[i] = {"expr": expr, "job": None, "state": "ready for processing"}
    return Expr("EvaluationObject", i, expr)


K.BUILTIN_ATTRIBUTES["ParallelSubmit"] = {"HoldAllComplete", "Protected"}
K.BUILTIN_ATTRIBUTES["EvaluationObject"] = {"HoldAllComplete", "Protected"}


def _eo_id(x):
    if isinstance(x, Expr) and x.head == "EvaluationObject" and x.args and K.is_int(x.args[0]):
        return int(x.args[0])
    return None


def _find_eos(x, out):
    i = _eo_id(x)
    if i is not None:
        if i not in out:
            out.append(i)
        return out
    if isinstance(x, Expr):
        if isinstance(x.head, Expr):
            _find_eos(x.head, out)
        for a in x.args:
            _find_eos(a, out)
    return out


def _dispatch_submitted():
    ks = list(_KERNELS.values()) if _WORKER is None else []
    if ks:
        _push_defs(ks, [n for n in _DIST if _has_defs(n)])
    for i in sorted(_EVALS):
        e = _EVALS[i]
        if e["job"] is None:
            j = _Job([e["expr"]])
            e["job"] = j
            if ks:
                _QUEUE.append(j)
            else:
                try:
                    j.outcome = [("ok", ev(e["expr"]), "")]
                except Exception as ex:  # noqa: BLE001 - re-raised by WaitAll
                    j.outcome = [("err", _exc_info(ex), "")]
                j.state = "finished"
                j.seq = next(_SEQ)


def _eo_done(i):
    e = _EVALS.get(i)
    return e is None or (e["job"] is not None and e["job"].state == "finished")


def _eo_value(i, fname):
    e = _EVALS.get(i)
    if e is None:
        raise ValueError(f"{fname}: EvaluationObject[{i}, ...] is not known in this session")
    v = _outcome(e["job"])[0]
    e["state"] = "finished"
    return v


def _replace_eos(x, vals):
    i = _eo_id(x)
    if i is not None:
        return vals[i]
    if isinstance(x, Expr):
        h = _replace_eos(x.head, vals) if isinstance(x.head, Expr) else x.head
        return Expr(h, *(_replace_eos(a, vals) for a in x.args))
    return x


@fn("WaitAll", area=AREA, status="added",
    notes="WaitAll[expr] sends all queued ParallelSubmit evaluations to the kernels, waits for the "
          "EvaluationObjects in expr (anywhere in it: a list, a sum, ...) to finish, replaces them by their results "
          "and evaluates the result; aborted evaluations give $Aborted. Without kernels the queued evaluations run "
          "on the master. Check: exact in tests.")
def _wait_all(expr: Expr) -> Expr:
    ids = _find_eos(expr, [])
    if not ids:
        return expr
    with _PAR_LOCK:
        if _EVALS and _WORKER is None and not _KERNELS:
            _ensure_kernels()
        _dispatch_submitted()
        _wait(lambda: all(_eo_done(i) for i in ids))
    vals = {i: _eo_value(i, "WaitAll") for i in ids}
    return ev(_replace_eos(expr, vals))


@fn("WaitNext", area=AREA, status="added",
    notes="WaitNext[{eo1, eo2, ...}] sends the queued evaluations to the kernels, waits until one of the given "
          "EvaluationObjects has finished and gives {result, eo, rest} with the remaining objects (the one that "
          "finished first; an already finished one is taken first). Check: exact (all results are collected by "
          "repeated WaitNext) in tests.")
def _wait_next(eos: ListOf[Expr]) -> ListOf[Expr]:
    if isinstance(eos, Symbol):
        raise Unevaluated()
    if not is_list(eos):
        eos = L(eos)
    ids = [_eo_id(x) for x in eos.args]
    if not ids or any(i is None for i in ids):
        raise TypeError("WaitNext: a non-empty list of EvaluationObjects is expected")
    with _PAR_LOCK:
        if _WORKER is None and not _KERNELS:
            _ensure_kernels()
        _dispatch_submitted()
        _wait(lambda: any(_eo_done(i) for i in ids))
    done = [i for i in ids if _eo_done(i)]
    first = min(done, key=lambda i: (_EVALS[i]["job"].seq if i in _EVALS and _EVALS[i]["job"].seq is not None
                                     else -1, ids.index(i)))
    k = ids.index(first)
    v = _eo_value(first, "WaitNext")
    return L(ev(v), eos.args[k], L(*(x for j, x in enumerate(eos.args) if j != k)))


def _eo_props_hook(head, args, ev_):
    i = _eo_id(head)
    if i is None or len(args) != 1:
        return K.UNHANDLED
    e = _EVALS.get(i)
    if e is None:
        raise ValueError(f"EvaluationObject: {i} is not known in this session")
    j = e["job"]
    if j is None:
        state = "ready for processing"
    elif j.state == "queued":
        state = "queued"
    elif j.state == "running":
        state = "running"
    else:
        state = "finished" if e["state"] == "finished" else "received"
    res = Expr("Missing", "NotAvailable")
    if j is not None and j.state == "finished" and j.error is None and j.outcome and j.outcome[0][0] != "err":
        res = Expr("HoldComplete", j.outcome[0][1] if j.outcome[0][0] == "ok" else ABORTED)
    props = {"Process": Expr("HoldForm", e["expr"]), "State": state, "Result": res}
    p = args[0]
    if is_sym(p, "All"):
        return assoc(list(props.items()))
    if p == "Properties":
        return L(*props)
    if isinstance(p, str) and p in props:
        return props[p]
    raise ValueError(f"EvaluationObject: unknown property {_form(p)} (known: Process, State, Result)")


KL.COMPOUND_HEAD_HOOKS["EvaluationObject"] = _eo_props_hook
record("EvaluationObject", AREA, "added",
       "EvaluationObject[id, expr] (HoldAllComplete, inert) represents an evaluation queued by ParallelSubmit; "
       "WaitAll and WaitNext give its result; eo[\"Process\"] (HoldForm of expr), eo[\"State\"] (\"ready for "
       "processing\", \"queued\", \"running\", \"received\", \"finished\"), eo[\"Result\"] (HoldComplete[value], "
       "Missing[\"NotAvailable\"] before), eo[All] and eo[\"Properties\"]. Check: exact in tests.")

# ── definitions ──


def _def_names(args, fname):
    names = []
    for a in args:
        if isinstance(a, Symbol):
            names.append(a.name)
        elif isinstance(a, str):
            if a.endswith("`"):
                st = K.state()
                cands = set(st.own) | set(st.down) | {K._def_key(v[0]) for v in st.exact.values()} | set(st.attrs)
                names.extend(sorted(n for n in cands if _context_of(n) == a and _is_user_symbol(n)))
            else:
                names.append(a)
        elif is_list(a):
            names.extend(_def_names(a.args, fname))
        else:
            raise TypeError(f"{fname}: {_form(a)} is not a symbol, a symbol name or a context")
    return names


@fn("DistributeDefinitions", area=AREA, status="added",
    notes="DistributeDefinitions[s1, s2, ...] sends the current definitions (values, down/sub/up values, "
          "attributes, options) of the symbols and of the user symbols they depend on to all parallel kernels, also "
          "to kernels launched later, and gives the list of HoldForm[s] distributed; a string \"ctx`\" means every "
          "symbol with definitions in that context. HoldAll. Check: exact (ParallelEvaluate sees the definition) "
          "in tests.")
def _distribute_definitions(*syms: Held) -> ListOf[Expr]:
    names = _def_names(syms, "DistributeDefinitions")
    allnames = _closure(names, lambda n: True, explicit=names)
    for n in allnames:
        _DIST[n] = True
    with _PAR_LOCK:
        if _KERNELS:
            _push_defs(list(_KERNELS.values()), allnames)
    return _held_syms(allnames)


K.BUILTIN_ATTRIBUTES["DistributeDefinitions"] = {"HoldAll", "Protected"}


@fn("ClearDistributedDefinitions", area=AREA, status="added",
    notes="ClearDistributedDefinitions[] removes every distributed definition from the parallel kernels and forgets "
          "them (the master's definitions are kept), giving the list of HoldForm[s] cleared; "
          "ClearDistributedDefinitions[s1, ...] only those symbols. HoldAll. Check: exact ($DistributedDefinitions "
          "afterwards; kernels no longer know the definition) in tests.")
def _clear_distributed(*syms: Held) -> ListOf[Expr]:
    names = list(_DIST) if not syms else [n for n in _def_names(syms, "ClearDistributedDefinitions") if n in _DIST]
    for n in names:
        _DIST.pop(n, None)
    with _PAR_LOCK:
        for k in _KERNELS.values():
            upd = [(n, _EMPTY_SNAP) for n in names if n in k.defs]
            if upd:
                k.send(("defs", upd))
                for n, _ in upd:
                    k.defs.pop(n, None)
    return _held_syms(names)


K.BUILTIN_ATTRIBUTES["ClearDistributedDefinitions"] = {"HoldAll", "Protected"}


def _broadcast_shared():
    for k in _KERNELS.values():
        k.send(("share", list(_SHARED_VARS), list(_SHARED_FNS)))
        upd = [(n, _EMPTY_SNAP) for n in _SHARED_VARS + _SHARED_FNS if n in k.defs]
        if upd:
            k.send(("defs", upd))
            for n, _ in upd:
                k.defs.pop(n, None)


def _share(syms, target, fname):
    for s in syms:
        n = _hold_sym_name(s, fname) if not is_list(s) else None
        group = [n] if n is not None else [_hold_sym_name(x, fname) for x in s.args]
        for n in group:
            if n.startswith("$") or n in REGISTRY:
                raise ValueError(f"{fname}: the system symbol {n} cannot be shared")
            other = _SHARED_FNS if target is _SHARED_VARS else _SHARED_VARS
            if n in other:
                other.remove(n)
            if n not in target:
                target.append(n)
            _DIST.pop(n, None)
    with _PAR_LOCK:
        _broadcast_shared()
    return NULL


@fn("SetSharedVariable", area=AREA, status="added",
    notes="SetSharedVariable[s1, ...] makes the symbols shared variables: every read and assignment of si on a "
          "parallel kernel goes to the master kernel (a variable without a value reads as Null there); updates are "
          "not atomic unless done inside CriticalSection. HoldAll; returns Null. Check: exact (values assigned on "
          "the kernels are seen on the master) in tests.")
def _set_shared_variable(*syms: Held) -> Expr:
    if not syms:
        raise TypeError("SetSharedVariable: called with 0 arguments; at least one symbol is expected")
    return _share(syms, _SHARED_VARS, "SetSharedVariable")


@fn("SetSharedFunction", area=AREA, status="added",
    notes="SetSharedFunction[f1, ...] makes the symbols shared functions: f[args] evaluated on a parallel kernel is "
          "evaluated on the master kernel (Null if the master has no applicable definition), and definitions "
          "f[...] = ... made on a kernel are stored on the master. HoldAll; returns Null. Check: exact (calls are "
          "evaluated on the master: $ProcessID of the master) in tests.")
def _set_shared_function(*syms: Held) -> Expr:
    if not syms:
        raise TypeError("SetSharedFunction: called with 0 arguments; at least one symbol is expected")
    return _share(syms, _SHARED_FNS, "SetSharedFunction")


@fn("UnsetShared", area=AREA, status="added",
    notes="UnsetShared[s1, ...] stops sharing the variables or functions si and gives the list of HoldForm[si] that "
          "were shared; UnsetShared[All] stops all sharing. HoldAll. Check: exact ($SharedVariables afterwards) in "
          "tests.")
def _unset_shared(*syms: Held) -> ListOf[Expr]:
    if len(syms) == 1 and is_sym(syms[0], "All"):
        names = _SHARED_VARS + _SHARED_FNS
    else:
        names = [_hold_sym_name(s, "UnsetShared") for s in syms]
    out = []
    for n in list(names):
        hit = False
        for grp in (_SHARED_VARS, _SHARED_FNS):
            if n in grp:
                grp.remove(n)
                hit = True
        if hit:
            out.append(n)
    with _PAR_LOCK:
        _broadcast_shared()
    return _held_syms(out)


for _h in ("SetSharedVariable", "SetSharedFunction", "UnsetShared"):
    K.BUILTIN_ATTRIBUTES[_h] = {"HoldAll", "Protected"}


@fn("ParallelNeeds", area=AREA, status="added",
    notes="ParallelNeeds[\"ctx`\"] evaluates Needs[\"ctx`\"] on every running parallel kernel and remembers it for "
          "kernels launched later (the master is not affected); errors from Needs are raised. Returns Null. Check: "
          "exact (the context appears in $Packages / $ContextPath of the kernels) in tests.")
def _parallel_needs(ctx: String, *rest: Expr) -> Expr:
    c = _string(ctx, "ParallelNeeds", "the context")
    if not c.endswith("`"):
        raise ValueError(f"ParallelNeeds: {c!r} is not a context (it must end in `)")
    x = Expr("Needs", c, *rest)
    with _PAR_LOCK:
        if _KERNELS:
            _run([_Job([x], kernel=k) for k in _KERNELS.values()])
        if c not in _NEEDS:
            _NEEDS.append(c)
    return NULL


def _lock_names(spec, fname):
    if isinstance(spec, Symbol):
        return [spec.name]
    if is_list(spec) and all(isinstance(s, Symbol) for s in spec.args):
        return sorted({s.name for s in spec.args})
    raise TypeError(f"{fname}: the locks must be a symbol or a list of symbols, not {_form(spec)}")


def _with_var_locks(names, expr):
    if _WORKER is not None:
        _WORKER.call(("lock", names))
        try:
            return ev(expr)
        finally:
            _WORKER.call(("unlock", names))
    me = ("t", threading.get_ident())
    with _LOCK_CV:
        while any(n in _LOCK_OWNERS and _LOCK_OWNERS[n] != me for n in names):
            _LOCK_CV.wait(0.05)
        taken = [n for n in names if n not in _LOCK_OWNERS]
        for n in taken:
            _LOCK_OWNERS[n] = me
    try:
        return ev(expr)
    finally:
        with _LOCK_CV:
            for n in taken:
                _LOCK_OWNERS.pop(n, None)
            _LOCK_CV.notify_all()
        _try_grant()


@fn("CriticalSection", area=AREA, status="added",
    notes="CriticalSection[{l1, l2, ...}, expr] evaluates expr while holding the locks li (symbols), so that at "
          "most one parallel kernel (or master task) at a time runs code guarded by the same lock; the locks are "
          "released afterwards, also on error or abort. The locks are managed by the master kernel. HoldAll. "
          "Check: exact (a guarded shared-variable update is not lost) in tests.")
def _critical_section(locks: Held, expr: Held) -> Expr:
    return _with_var_locks(_lock_names(locks, "CriticalSection"), expr)


K.BUILTIN_ATTRIBUTES["CriticalSection"] = {"HoldAll", "Protected"}


def _file_lock(path, expr):
    path = os.path.abspath(os.path.expanduser(path))
    d = os.path.dirname(path)
    if d and not os.path.isdir(d):
        raise ValueError(f"WithLock: the directory of the lock file {path!r} does not exist")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if os.name == "nt":  # pragma: no cover - Windows
            import msvcrt
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                return ev(expr)
            finally:
                os.lseek(fd, 0, 0)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        import fcntl
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            return ev(expr)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@fn("WithLock", area=AREA, status="added",
    notes="WithLock[File[\"path\"], expr] (or a path string) evaluates expr while holding an exclusive operating-"
          "system lock on the file (created if missing and left in place), so concurrent kernels and processes "
          "using the same lock file run expr one at a time; WithLock[{l1, ...}, expr] with symbols uses the "
          "master's variable locks like CriticalSection. HoldRest. Check: exact (evaluations serialized) in tests.")
def _with_lock(lock: Expr, expr: Held) -> Expr:
    if isinstance(lock, Expr) and lock.head == "File" and len(lock.args) == 1 and isinstance(lock.args[0], str):
        return _file_lock(lock.args[0], expr)
    if isinstance(lock, str):
        return _file_lock(lock, expr)
    if isinstance(lock, Expr) and lock.head == "LocalSymbol":
        raise NotImplementedError("WithLock: LocalSymbol locks are not supported")
    return _with_var_locks(_lock_names(lock, "WithLock"), expr)


K.BUILTIN_ATTRIBUTES["WithLock"] = {"HoldRest", "Protected"}


def _local_spec(spec, fname):
    if spec is None:
        spec = ev(Symbol("$DefaultLocalKernel"))
    if isinstance(spec, str) and spec in ("Local", "localhost", "local", "file://"):
        return "Local"
    if isinstance(spec, Expr) and spec.head == "KernelConfiguration":
        _config_count(spec, fname)
        return "Local"
    if isinstance(spec, str):
        raise NotImplementedError(f"{fname}: only local MikoshiLang kernels are supported (got {spec!r})")
    raise TypeError(f"{fname}: {_form(spec)} is not a kernel specification")


def _one_shot(expr, include, fname, on_print=None, holder=None):
    """Evaluate expr in a new local kernel (discarded afterwards); returns the value."""
    defs = []
    if include:
        defs = [(n, _snap(n)) for n in _closure(_names_in(expr), lambda n: True)]
    box = queue.Queue()
    k = _start_kernels([0], "Local", inbox=box, register=False, init_defs=defs)[0]
    if holder is not None:
        holder.append(k)
    try:
        k.send(("eval", 0, [expr], 0))
        while True:
            try:
                kk, msg = box.get(timeout=0.05)
            except queue.Empty:
                continue
            if msg[0] == "result":
                st, v, printed = msg[2][0]
                if printed:
                    if on_print is not None:
                        on_print(printed)
                    else:
                        from .wl_errors import print_line
                        print_line(printed.rstrip("\n"))
                if st == "ok":
                    return v
                if st == "abort":
                    return ABORTED
                raise _rebuild_exc(v)
            if msg[0] == "dead":
                raise RuntimeError(f"{fname}: the local kernel terminated during the evaluation")
            if msg[0] in _CALLBACKS:
                _reply(kk, False, ("RuntimeError", "local kernels have no shared variables or functions"))
    finally:
        k.close()


@fn("LocalEvaluate", area=AREA, status="added",
    notes="LocalEvaluate[expr] evaluates expr in a new local MikoshiLang kernel (a separate process, discarded "
          "afterwards) and gives the result; LocalEvaluate[\"Local\", expr] names the kernel ($DefaultLocalKernel), "
          "LocalEvaluate[{k1, k2, ...}, expr] uses one new kernel per element and gives a list. With "
          "IncludeDefinitions -> True (default) the definitions expr depends on are sent along. Other kernel specs "
          "(ssh://, wstp://, file:// Wolfram kernels, cloud) raise NotImplementedError. HoldAll. Check: exact (a "
          "different $ProcessID; the definitions are available) in tests.")
def _local_evaluate(a: Held, b: Held = None, *opts: Option) -> Expr:
    pos = [x for x in (a, b) if x is not None]
    pos, o = _opts(pos + list(opts), ("IncludeDefinitions",), "LocalEvaluate")
    inc = o.get("IncludeDefinitions", True)
    if not isinstance(inc, bool):
        raise TypeError("LocalEvaluate: IncludeDefinitions must be True or False")
    if len(pos) == 1:
        _local_spec(None, "LocalEvaluate")
        return ev(_one_shot(pos[0], inc, "LocalEvaluate"))
    if len(pos) != 2:
        raise TypeError("LocalEvaluate: called with the wrong number of arguments")
    spec = ev(pos[0])
    if is_list(spec):
        for s in spec.args:
            _local_spec(s, "LocalEvaluate")
        return ev(L(*(_one_shot(pos[1], inc, "LocalEvaluate") for _ in spec.args)))
    _local_spec(spec, "LocalEvaluate")
    return ev(_one_shot(pos[1], inc, "LocalEvaluate"))


K.BUILTIN_ATTRIBUTES["LocalEvaluate"] = {"HoldAll", "Protected"}

record("DistributedContexts", AREA, "added",
       "DistributedContexts: option of ParallelMap, ParallelTable, ParallelSum, ParallelProduct, ParallelDo, "
       "ParallelArray, ParallelCombine, ParallelCases, ParallelSelect, ParallelTry, Parallelize and ParallelEvaluate "
       "giving the contexts whose symbols' definitions are distributed automatically: $DistributedContexts "
       "(default, i.e. $Context), All or Automatic (every context), None, a context \"ctx`\" or a list of contexts. "
       "Check: exact (None leaves a function undefined on the kernels) in tests.")
record("ProgressReporting", AREA, "added",
       "ProgressReporting: option of Parallelize and the parallel commands (and of other long computations) "
       "saying whether to show a progress panel; accepted and ignored, since this text interface has no progress "
       "panels (default $ProgressReporting). Check: exact (results unchanged) in tests.")


# ═══════════════════════════════════════════ tasks ═══════════════════════════════════════════

_TASKS = collections.OrderedDict()          # uuid -> _Task (not removed)
_ALL_TASKS = {}                             # uuid -> _Task
_CURRENT_TASK = ContextVar("w29_current_task", default=None)

_PERIODS = {"Hourly": 3600.0, "Daily": 86400.0, "Weekly": 604800.0, "Monthly": 2629746.0, "Yearly": 31556952.0}
_EVENTS_SESSION = ("TaskStarted", "ScheduleTriggered", "ResultReceived", "ScheduleExecuted", "TaskStatusChanged",
                   "TaskFinished", "TaskRemoved")
_EVENTS_LOCAL = _EVENTS_SESSION + ("PrintOutputGenerated", "FailureOccurred")
_TASK_KEYS = ("EvaluationExpression", "EvaluationResult", "Task", "TaskStatus", "TaskType", "TaskUUID", "EventName",
              "TaskEnvironment", "HandlerFunctions", "HandlerFunctionsKeys", "Schedule", "TotalRunCount",
              "PreviousRunCount", "RemainingRunCount", "RunInterval", "NextScheduledTime", "PrintOutput", "Failure")
_AUTO_KEYS = ("Task", "TaskUUID", "TaskType", "TaskStatus", "TaskEnvironment", "EvaluationExpression",
              "EvaluationResult", "EventName")


def _period(x, fname):
    if isinstance(x, str):
        if x in _PERIODS:
            return _PERIODS[x]
        raise NotImplementedError(f"{fname}: the schedule {x!r} is not supported (cron specifications are not "
                                  "implemented; use a number of seconds, a time Quantity or Hourly/Daily/...)")
    if isinstance(x, Expr) and x.head in ("DateObject", "TimeObject"):
        raise NotImplementedError(f"{fname}: start/end dates and date patterns in schedules are not supported")
    t = _nonneg_real(x, fname, "schedule interval")
    if t is None:
        raise ValueError(f"{fname}: the schedule interval must be finite")
    return t


def _schedule_spec(spec, fname):
    """(interval, count, first delay); count None = forever, 0 = on demand only."""
    if is_sym(spec, "None"):
        return (None, 0, None)
    if is_sym(spec, "Now"):
        return (0.0, 1, 0.0)
    if is_list(spec):
        if len(spec.args) == 1:
            t = _period(spec.args[0], fname)
            return (t, 1, t)
        if len(spec.args) == 2 and not (isinstance(spec.args[0], Expr) and spec.args[0].head == "DateObject"):
            t = _period(spec.args[0], fname)
            n = spec.args[1]
            if is_sym(n, "Infinity"):
                return (t, None, t)
            return (t, _nonneg_int(n, fname, "the run count"), t)
        raise NotImplementedError(f"{fname}: schedules with start or end dates are not supported")
    t = _period(spec, fname)
    return (t, None, t)


class _Task:
    def __init__(self, expr, env, sched, sched_expr, handlers, keys, autoremove):
        self.uuid = str(_uuid.uuid4())
        self.expr = expr
        self.env = env
        self.sched = sched
        self.sched_expr = sched_expr
        self.type = "Asynchronous" if sched is None else "Scheduled"
        self.handlers = handlers
        self.keys = keys
        self.autoremove = autoremove
        self.result = Expr("Missing", "NotAvailable")
        self.status = "Running" if sched is None or sched[1] != 0 else "Waiting"
        self.prev = 0
        self.event_data = {}
        self.cv = threading.Condition()
        self.suspended = False
        self.removed = False
        self.evaluating = False
        self.thread = None
        self.kernels = []
        self.next_time = None
        self.done = threading.Event()
        _TASKS[self.uuid] = self
        _ALL_TASKS[self.uuid] = self

    # ── representation ──
    def expr_obj(self):
        pairs = [("TaskUUID", self.uuid), ("TaskEnvironment", self.env), ("TaskType", self.type)]
        items = [Expr("Rule", k, v) for k, v in pairs]
        items.append(Expr("RuleDelayed", "EvaluationExpression", self.expr))
        if self.sched is not None:
            items.append(Expr("Rule", "Schedule", self._schedule_assoc()))
        return Expr("TaskObject", Expr("Association", *items))

    def _schedule_assoc(self):
        t, n, _ = self.sched
        return assoc([("Repeated", n is None), ("TotalRunCount", Symbol("Infinity") if n is None else n),
                      ("RunInterval", NONE if t is None else t)])

    def total(self):
        if self.sched is None:
            return 1
        return self.sched[1]

    def prop(self, k):
        if k == "TaskUUID":
            return self.uuid
        if k == "TaskEnvironment":
            return self.env
        if k == "TaskType":
            return self.type
        if k == "TaskStatus":
            return self.status
        if k == "EvaluationExpression":
            return Expr("HoldForm", self.expr)
        if k == "EvaluationResult":
            return self.result
        if k == "EventData":
            return assoc([(key, self.event_data.get(key, Expr("Missing", "NotAvailable"))) for key in self.keys])
        if k == "HandlerFunctions":
            return assoc(list(self.handlers.items()))
        if k == "HandlerFunctionsKeys":
            return L(*self.keys)
        if k == "Task":
            return self.expr_obj()
        if k == "Properties":
            base = ["EvaluationExpression", "EvaluationResult", "EventData", "HandlerFunctions",
                    "HandlerFunctionsKeys", "TaskEnvironment", "TaskStatus", "TaskType", "TaskUUID"]
            if self.sched is not None:
                base += ["NextScheduledTime", "PreviousRunCount", "RemainingRunCount", "RunInterval", "Schedule",
                         "TotalRunCount"]
            return L(*base)
        if self.sched is not None:
            t, n, _ = self.sched
            if k == "Schedule":
                return self.sched_expr
            if k == "RunInterval":
                return NONE if t is None else t
            if k == "PreviousRunCount":
                return self.prev
            if k == "TotalRunCount":
                return Symbol("Infinity") if n is None else n
            if k == "RemainingRunCount":
                return Symbol("Infinity") if n is None else max(0, n - self.prev)
            if k == "NextScheduledTime":
                if self.next_time is None or self.status in ("Finished", "Removed"):
                    return NONE
                return _date_object(self.next_time)
        if k in ("PrintOutput", "Failure", "EventName"):
            return self.event_data.get(k, Expr("Missing", "NotAvailable"))
        raise ValueError(f"TaskObject: unknown property {k!r}")

    # ── events ──
    def fire(self, event, **extra):
        f = self.handlers.get(event)
        if extra:
            self.event_data.update(extra)
        self.event_data["EventName"] = event
        if f is None:
            return
        data = []
        for key in self.keys:
            if key == "EventName":
                v = event
            elif key == "EvaluationExpression":
                v = Expr("Hold", self.expr)
            elif key in extra:
                v = extra[key]
            elif key in ("PrintOutput", "Failure"):
                v = self.event_data.get(key, Expr("Missing", "NotAvailable"))
            else:
                try:
                    v = self.prop(key)
                except ValueError:
                    v = Expr("Missing", "NotAvailable")
            self.event_data[key] = v
            data.append((key, v))
        tok = _CURRENT_TASK.set(self)
        try:
            ev(K.call(f, assoc(data)))
        except Exception as e:  # noqa: BLE001 - a failing handler must not stop the task
            sys.stderr.write(f"TaskObject: the {event} handler failed: {e}\n")
        finally:
            _CURRENT_TASK.reset(tok)

    def set_status(self, s):
        if self.status != s:
            self.status = s
            self.fire("TaskStatusChanged")

    # ── execution ──
    def execute(self):
        """One evaluation of the task expression (in the calling thread)."""
        from .wl_errors import AbortSignal
        tok = _CURRENT_TASK.set(self)
        try:
            with self.cv:
                self.evaluating = True
                self.eval_tid = threading.get_ident()
            try:
                if self.env == "Local":
                    v = _one_shot(self.expr, True, "LocalSubmit", on_print=self._printed, holder=self.kernels)
                    v = ev(v)
                else:
                    v = ev(self.expr)
            finally:
                with self.cv:
                    self.evaluating = False
                    self.kernels.clear()
        except AbortSignal:
            v = ABORTED
        except Exception as e:  # noqa: BLE001 - the error is the task's failure
            sys.stderr.write(f"TaskObject: the task evaluation failed: {e}\n")
            self.event_data["Failure"] = Expr("Failure", "TaskError", assoc([("MessageTemplate", str(e))]))
            if "FailureOccurred" in self.handlers:
                self.fire("FailureOccurred")
            v = FAILED
        finally:
            _CURRENT_TASK.reset(tok)
        self.result = v
        self.event_data["EvaluationResult"] = v
        self.fire("ResultReceived", EvaluationResult=v)
        return v

    def _printed(self, text):
        if "PrintOutputGenerated" in self.handlers:
            self.fire("PrintOutputGenerated", PrintOutput=text.rstrip("\n"))
        else:
            from .wl_errors import print_line
            print_line(text.rstrip("\n"))

    def _sleep(self, secs):
        """Wait secs (while not suspended); False if the task was removed."""
        end = time.monotonic() + secs
        self.next_time = time.time() + secs
        with self.cv:
            while True:
                if self.removed:
                    return False
                if self.suspended:
                    self.cv.wait(0.05)
                    end = max(end, time.monotonic())
                    continue
                left = end - time.monotonic()
                if left <= 0:
                    return True
                self.cv.wait(min(left, 0.05))

    def run(self):
        from .wl_errors import AbortSignal
        try:
            self.fire("TaskStarted")
            if self.sched is None:
                self.execute()
            else:
                t, n, first = self.sched
                if n == 0:
                    return
                delay = first
                while (n is None or self.prev < n) and not self.removed:
                    if not self._sleep(delay):
                        break
                    self.fire("ScheduleTriggered")
                    self.execute()
                    self.prev += 1
                    self.fire("ScheduleExecuted")
                    delay = t
        except AbortSignal:
            pass
        finally:
            if not self.removed:
                self.next_time = None
                self.set_status("Finished")
                self.fire("TaskFinished")
                if self.autoremove:
                    self.remove()
            self.done.set()

    def start(self):
        if self.sched is not None and self.sched[1] == 0:
            self.done.set()
            return
        ctx = copy_context()
        self.thread = threading.Thread(target=ctx.run, args=(self.run,), daemon=True,
                                       name=f"w29-task-{self.uuid[:8]}")
        self.thread.start()

    def abort(self):
        from .wl_errors import AbortSignal
        with self.cv:
            if not self.evaluating:
                return
            if self.env == "Local":
                for k in list(self.kernels):
                    k.kill()
                return
            tid = getattr(self, "eval_tid", None)
            if tid is None or tid == threading.get_ident():
                return
            import ctypes
            ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), ctypes.py_object(AbortSignal))

    def remove(self):
        if self.removed:
            return
        self.removed = True
        self.abort()
        with self.cv:
            self.cv.notify_all()
        _TASKS.pop(self.uuid, None)
        self.status = "Removed"
        self.next_time = None
        self.fire("TaskRemoved")
        if self.sched is not None and self.sched[1] == 0:
            self.done.set()


def _task_opts(opts, fname, env):
    _, o = _opts(opts, ("HandlerFunctions", "HandlerFunctionsKeys", "Method", "AutoRemove"), fname)
    hf = o.get("HandlerFunctions", Expr("Association"))
    if is_list(hf):
        hf = Expr("Association", *hf.args)
    handlers = {}
    allowed = _EVENTS_LOCAL if env == "Local" else _EVENTS_SESSION
    for k, v in _assoc_dict(hf, fname, "HandlerFunctions").items():
        if k in ("PrintOutputGenerated", "MessageGenerated", "FailureOccurred") and k not in allowed:
            raise NotImplementedError(f"{fname}: the {k} event is not supported for session tasks")
        if k == "MessageGenerated":
            raise NotImplementedError(f"{fname}: the MessageGenerated event is not supported")
        if k not in allowed:
            raise ValueError(f"{fname}: {k!r} is not a task event (known: {', '.join(allowed)})")
        handlers[k] = v
    keys = o.get("HandlerFunctionsKeys", AUTOMATIC)
    if is_sym(keys, "Automatic"):
        keys = list(_AUTO_KEYS)
    elif isinstance(keys, str):
        keys = [keys]
    elif is_list(keys) and all(isinstance(x, str) for x in keys.args):
        keys = list(keys.args)
    else:
        raise TypeError(f"{fname}: HandlerFunctionsKeys must be Automatic or a list of strings")
    for k in keys:
        if k not in _TASK_KEYS:
            raise ValueError(f"{fname}: {k!r} is not a handler key (known: {', '.join(_TASK_KEYS)})")
    m = o.get("Method", AUTOMATIC)
    if m == "Idle":
        raise NotImplementedError(f"{fname}: Method -> \"Idle\" is not supported (tasks run at any time)")
    if not is_sym(m, "Automatic"):
        raise ValueError(f"{fname}: Method must be Automatic or \"Idle\", not {_form(m)}")
    ar = o.get("AutoRemove", False)
    if not isinstance(ar, bool):
        raise TypeError(f"{fname}: AutoRemove must be True or False")
    return handlers, keys, ar


def _submit(expr, opts, env, fname):
    sched = None
    sched_expr = NONE
    ar_task = False
    if isinstance(expr, Expr) and expr.head == "ScheduledTask":
        if not expr.args:
            raise TypeError(f"{fname}: ScheduledTask[expr, spec] expects an expression and a schedule")
        inner = expr.args[0]
        rest, so = _opts(expr.args[1:], ("AutoRemove",), "ScheduledTask")
        if len(rest) != 1:
            raise TypeError(f"{fname}: ScheduledTask[expr, spec] expects an expression and a schedule")
        sched_expr = ev(rest[0])
        sched = _schedule_spec(sched_expr, fname)
        ar_task = so.get("AutoRemove", False)
        if not isinstance(ar_task, bool):
            raise TypeError(f"{fname}: AutoRemove must be True or False")
        expr = inner
    handlers, keys, ar = _task_opts(opts, fname, env)
    t = _Task(expr, env, sched, sched_expr, handlers, keys, ar or ar_task)
    t.start()
    return t.expr_obj()


@fn("SessionSubmit", area=AREA, status="added",
    notes="SessionSubmit[expr] starts evaluating expr in the background of the current session (a thread sharing the "
          "session's definitions, running preemptively like Wolfram session tasks) and gives a TaskObject at once; "
          "SessionSubmit[ScheduledTask[expr, spec]] runs it on a schedule (None: on demand only; t seconds or a "
          "time Quantity: every t; {t}: once after t; {t, n}: n times; Now; \"Hourly\", \"Daily\", ...). Options "
          "HandlerFunctions (events TaskStarted, ScheduleTriggered, ResultReceived, ScheduleExecuted, "
          "TaskStatusChanged, TaskFinished, TaskRemoved; PrintOutputGenerated/MessageGenerated raise "
          "NotImplementedError), HandlerFunctionsKeys, Method (Automatic), AutoRemove. Dates/cron schedules raise "
          "NotImplementedError. HoldAll. Check: exact (TaskWait then EvaluationResult) in tests.")
def _session_submit(expr: Held, *opts: Held) -> Expr:
    return _submit(expr, [ev(x) for x in opts], "Session", "SessionSubmit")


@fn("LocalSubmit", area=AREA, status="added",
    notes="LocalSubmit[expr] evaluates expr asynchronously in a new local MikoshiLang kernel (a separate process, "
          "with the definitions expr needs) and gives a TaskObject at once; LocalSubmit[ScheduledTask[expr, spec]] "
          "repeats it on a schedule as SessionSubmit does. Options HandlerFunctions (also PrintOutputGenerated, "
          "FailureOccurred), HandlerFunctionsKeys, AutoRemove. TaskAbort kills the kernel process. HoldAll. Check: "
          "exact (a different $ProcessID; result via TaskWait) in tests.")
def _local_submit(expr: Held, *opts: Held) -> Expr:
    return _submit(expr, [ev(x) for x in opts], "Local", "LocalSubmit")


for _h in ("SessionSubmit", "LocalSubmit"):
    K.BUILTIN_ATTRIBUTES[_h] = {"HoldAll", "Protected"}
K.BUILTIN_ATTRIBUTES["ScheduledTask"] = {"HoldFirst", "Protected"}
record("ScheduledTask", AREA, "added",
       "ScheduledTask[expr, spec] (HoldFirst, inert) describes a task to run on a schedule for SessionSubmit and "
       "LocalSubmit: spec None, Now, t seconds or a time Quantity (repeated), {t} (once), {t, n} (n times) or "
       "\"Hourly\"/\"Daily\"/\"Weekly\"/\"Monthly\"/\"Yearly\"; option AutoRemove. Dates and cron specifications raise "
       "NotImplementedError when submitted. Check: exact (run counts) in tests.")
record("AutoRemove", AREA, "added",
       "AutoRemove: option of SessionSubmit, LocalSubmit and ScheduledTask; True removes the task (as TaskRemove) "
       "after its final run, False (default) keeps it in Tasks[]. Check: exact (TaskStatus \"Removed\" after the "
       "last run) in tests.")


def _task_of(x, fname):
    if isinstance(x, str):
        t = _ALL_TASKS.get(x)
        if t is None:
            raise ValueError(f"{fname}: no task with id {x!r}")
        return t
    if isinstance(x, Expr) and x.head == "TaskObject" and len(x.args) == 1 and is_assoc(x.args[0]):
        u = _assoc_dict(x.args[0], fname).get("TaskUUID")
        t = _ALL_TASKS.get(u) if isinstance(u, str) else None
        if t is None:
            raise ValueError(f"{fname}: {_form(x)} is not a task of this session")
        return t
    if _is_symbolic(x) or isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: {_form(x)} is not a TaskObject or task id")


def _listable(x, f):
    if is_list(x):
        return L(*(f(y) for y in x.args))
    return f(x)


@fn("TaskWait", area=AREA, status="added",
    notes="TaskWait[task] waits until the task (TaskObject or id string) has completely finished (all scheduled "
          "runs) and gives its TaskObject; an on-demand (None) task returns at once. Option TimeConstraint -> t "
          "returns after at most t seconds. Listable. Check: exact (EvaluationResult after waiting) in tests.")
def _task_wait(task: Expr | ListOf[Expr], *opts: Option) -> Expr:
    _, o = _opts(opts, ("TimeConstraint",), "TaskWait")
    tc = o.get("TimeConstraint", Symbol("Infinity"))
    limit = _nonneg_real(tc, "TaskWait", "TimeConstraint")

    def one(x):
        t = _task_of(x, "TaskWait")
        end = None if limit is None else time.monotonic() + limit
        while not t.done.is_set():
            if t.removed and (t.thread is None or not t.thread.is_alive()):
                break
            left = 0.05 if end is None else min(0.05, end - time.monotonic())
            if left <= 0:
                break
            t.done.wait(left)
        return t.expr_obj()
    return _listable(task, one)


@fn("TaskAbort", area=AREA, status="added",
    notes="TaskAbort[task] aborts the evaluation the task is currently running (its result becomes $Aborted; a "
          "scheduled task continues with its next run; for LocalSubmit tasks the kernel process is killed) and "
          "gives the TaskObject at once; it does nothing if the task is not evaluating. Listable. Check: exact "
          "(EvaluationResult $Aborted) in tests.")
def _task_abort(task: Expr | ListOf[Expr]) -> Expr:
    def one(x):
        t = _task_of(x, "TaskAbort")
        t.abort()
        return t.expr_obj()
    return _listable(task, one)


@fn("TaskRemove", area=AREA, status="added",
    notes="TaskRemove[task] stops a task (aborting a running evaluation and cancelling future scheduled runs), "
          "removes it from Tasks[] and gives its TaskObject (its status becomes \"Removed\"). Listable. Check: exact "
          "(Tasks[] afterwards) in tests.")
def _task_remove(task: Expr | ListOf[Expr]) -> Expr:
    def one(x):
        t = _task_of(x, "TaskRemove")
        t.remove()
        return t.expr_obj()
    return _listable(task, one)


@fn("TaskSuspend", area=AREA, status="added",
    notes="TaskSuspend[task] suspends a task: no further scheduled runs start until TaskResume (an evaluation "
          "already running continues); its status is \"Suspended\". Gives the TaskObject; finished or removed tasks "
          "are left alone. Listable. Check: exact (TaskStatus; no runs while suspended) in tests.")
def _task_suspend(task: Expr | ListOf[Expr]) -> Expr:
    def one(x):
        t = _task_of(x, "TaskSuspend")
        if t.status not in ("Finished", "Removed"):
            with t.cv:
                t.suspended = True
            t.set_status("Suspended")
        return t.expr_obj()
    return _listable(task, one)


@fn("TaskResume", area=AREA, status="added",
    notes="TaskResume[task] resumes a suspended task (status \"Running\"; a pending scheduled run waits its full "
          "interval again) and gives the TaskObject. Listable. Check: exact (TaskStatus after resuming) in tests.")
def _task_resume(task: Expr | ListOf[Expr]) -> Expr:
    def one(x):
        t = _task_of(x, "TaskResume")
        if t.suspended and t.status == "Suspended":
            with t.cv:
                t.suspended = False
                t.cv.notify_all()
            t.set_status("Running" if not t.done.is_set() else "Finished")
        return t.expr_obj()
    return _listable(task, one)


@fn("TaskExecute", area=AREA, status="added",
    notes="TaskExecute[task] evaluates the task's expression once now, synchronously in the current session (in "
          "a new local kernel for LocalSubmit tasks), without changing its schedule; the result becomes its "
          "EvaluationResult and the TaskObject is returned. Listable. Check: exact (EvaluationResult) in tests.")
def _task_execute(task: Expr | ListOf[Expr]) -> Expr:
    def one(x):
        t = _task_of(x, "TaskExecute")
        if t.removed:
            raise ValueError("TaskExecute: the task has been removed")
        t.execute()
        return t.expr_obj()
    return _listable(task, one)


@fn("Tasks", area=AREA, status="added",
    notes="Tasks[] gives the TaskObjects of this session's tasks that have not been removed (running, waiting, "
          "suspended or finished), in creation order; Tasks[\"Session\"] / Tasks[\"Local\"] only those of that "
          "environment. Check: exact (after SessionSubmit/TaskRemove) in tests.")
def _tasks(env: String = None) -> ListOf[Expr]:
    ts = list(_TASKS.values())
    if env is not None:
        e = _string(env, "Tasks", "the task environment")
        if e not in ("Session", "Local"):
            raise ValueError(f"Tasks: {e!r} is not a task environment (\"Session\" or \"Local\")")
        ts = [t for t in ts if t.env == e]
    return L(*(t.expr_obj() for t in ts))


def _task_props_hook(head, args, ev_):
    if not (isinstance(head, Expr) and head.head == "TaskObject" and len(args) == 1 and isinstance(args[0], str)):
        return K.UNHANDLED
    return _task_of(head, "TaskObject").prop(args[0])


KL.COMPOUND_HEAD_HOOKS["TaskObject"] = _task_props_hook
record("TaskObject", AREA, "added",
       "TaskObject[<|\"TaskUUID\", \"TaskEnvironment\", \"TaskType\", \"EvaluationExpression\" :> expr, ...|>] "
       "represents a task from SessionSubmit or LocalSubmit; task[\"prop\"] gives \"TaskStatus\" (\"Running\", "
       "\"Waiting\", \"Suspended\", \"Finished\", \"Removed\"), \"EvaluationResult\" (Missing until available), "
       "\"EvaluationExpression\" (HoldForm), \"TaskUUID\", \"TaskType\", \"TaskEnvironment\", \"EventData\", "
       "\"HandlerFunctions\", \"HandlerFunctionsKeys\", \"Properties\" and for scheduled tasks \"Schedule\", "
       "\"RunInterval\", \"PreviousRunCount\", \"RemainingRunCount\", \"TotalRunCount\", \"NextScheduledTime\". "
       "Check: exact in tests.")
_sysvar("$CurrentTask", lambda: (_CURRENT_TASK.get().expr_obj() if _CURRENT_TASK.get() is not None else NONE),
        "the TaskObject of the task being evaluated (also inside its handler functions), None outside tasks",
        check="exact (a task reads its own TaskUUID) in tests")


def _shutdown():
    for t in list(_TASKS.values()):
        try:
            t.removed = True
            with t.cv:
                t.cv.notify_all()
        except Exception:  # noqa: BLE001 - interpreter exit
            pass
    for k in list(_KERNELS.values()):
        try:
            k.kill()
        except Exception:  # noqa: BLE001
            pass


atexit.register(_shutdown)


# ═══════════════════════════════════════════ symbols left out ═══════════════════════════════════════════

_SKIPS = {
    "KernelConfigurationEdit": "opens the notebook front end's kernel-configuration dialog; there is no front end",
    "FindSettings": "settings of the notebook front end's Find interface; there is no front end",
    "EvaluationElements": "an option of NotebookEvaluate (which notebook cells to evaluate); notebooks are not "
                          "supported",
    "InsertResults": "an option of NotebookEvaluate (insert outputs into the notebook); notebooks are not supported",
    "CaptureRunning": "a property of audio/image capture devices (AudioCapture, ImageCapture); capture hardware is "
                      "not supported",
    "MaxDuration": "an option of AudioCapture/AudioRecord limiting recording time; audio capture is not supported",
    "TimeGoal": "an option of Classify/Predict and other machine-learning trainers, which are not available",
    "EvaluationEnvironment": "an option of InitializationValue (persistent initialization values), which is not "
                             "available",
    "RemoteRunProcess": "runs a process on a remote machine over SSH with authentication; remote hosts are not "
                        "supported",
}
SKIPPED = dict(_SKIPS)        # left out on purpose (listed in the batch report); nothing is registered for them
record("InitialEvaluationHistory", AREA, "added",
       "InitialEvaluationHistory: option of BayesianMinimization and BayesianMaximization giving earlier "
       "evaluations {conf1 -> value1, ...} to start the optimization from (handled by the W08 optimizers). Check: "
       "exact (a history containing the optimum is used) in tests.")
record("OperatingSystem", AREA, "added",
       "OperatingSystem: option of FileNameJoin, FileNameSplit and related file-name functions giving the naming "
       "conventions to use: \"Unix\", \"MacOSX\" or \"Windows\" (default $OperatingSystem); handled by the W27 "
       "file-name functions. Check: exact (Windows vs Unix separators) in tests.")


@fn("$CurrentTask", area=AREA, status="added",
    notes="$CurrentTask: the TaskObject of the task being evaluated (also inside its handler functions), None "
          "outside tasks; $CurrentTask[\"prop\"] reads a property of the current task (as documented). Check: exact "
          "(a task reads its own TaskUUID) in tests.")
def _current_task_prop(prop: String) -> Expr:
    t = _CURRENT_TASK.get()
    if t is None:
        raise ValueError("$CurrentTask: not evaluating inside a task")
    return t.prop(_string(prop, "$CurrentTask", "the property"))


@fn("$RandomGeneratorState", area=AREA, status="added",
    notes="$RandomGeneratorState: the current state of the library's pseudorandom generator as a "
          "RandomGeneratorState (it changes whenever random numbers are generated and is reset by SeedRandom); "
          "$RandomGeneratorState[\"prop\"] gives a property of it (Method, StateHash, CrossSessionPersistent). "
          "Check: exact (BlockRandom[..., RandomSeeding -> $RandomGeneratorState] repeats the numbers) in tests.")
def _rgs_prop(prop: String) -> Expr:
    return _rgs_props_hook(_rgs(), [prop], ev)
