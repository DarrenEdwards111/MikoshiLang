"""Coverage batch W02: Wolfram Language messages, exceptions, failures, the testing framework, timing and resource
constraints.

Messages
    ``Message[s::t, e1, ...]`` looks up the text of ``s::t`` (a user assignment ``s::t = "..."``, then the built-in
    text, then ``General::t``, then ``$NewMessage``), formats it like ``StringForm`` (`` `n` `` and ``````) and prints
    ``s::t: text`` on the ``$Messages`` channels (stdout by default).  Printed messages are appended to
    ``$MessageList`` as ``HoldForm[s::t]``.  ``Off``/``On`` switch messages (``General::t`` covers every ``s::t``),
    ``Quiet`` suppresses them (innermost specification wins), and ``Check`` tests only for messages that are not
    switched off and not quieted inside it -- the documented Wolfram rules.

    MikoshiLang functions report most errors by raising a Python error rather than issuing a message.  The functions
    whose job is to detect messages (``Check``, ``ConfirmQuiet``, ``VerificationTest``/``TestEvaluate`` and
    ``IntermediateTest``) treat such an error as the message ``f::evalerr`` (``f`` is the function named at the start
    of the error text, else ``General``): it is issued at that point, and the evaluation counts as having produced a
    message.  ``Quiet`` cannot turn a raised error into a value, so it lets the error propagate.

Exceptions and failures
    ``Failure``/``Success`` objects with their properties; ``Enclose`` with the ``Confirm*`` family (untagged
    confirmations are lexically bound to the surrounding ``Enclose`` by rewriting them with a private tag, exactly the
    mechanism Wolfram documents); the Wolfram 15 ``Exception`` system (``Exception``, ``ThrowException``,
    ``CatchExceptions``, ``RegisterExceptionType``, ...).  ``Confirm`` and ``ThrowException`` use ``Throw`` with a
    tag, so ``Catch[..., _]`` intercepts them as in Wolfram; uncaught ones reach the top level as an
    ``Uncaught Throw`` error.

Aborts and constraints
    ``Abort[]`` raises :class:`AbortSignal` (the top-level result of an aborted computation).  ``TimeConstrained`` and
    ``MemoryConstrained`` interrupt the evaluation from a timer (``SIGALRM`` in the main thread, a trace hook
    elsewhere); time is wall-clock time and memory is the Python heap measured with ``tracemalloc``.
    ``AbortProtect``, ``CheckAbort`` (with ``PropagateAborts``) and ``WithCleanup`` follow the documented rules.

Testing
    ``TestCreate``/``TestEvaluate``/``VerificationTest``/``IntermediateTest`` produce ``TestObject[<|...|>]`` and
    ``TestReport`` a ``TestReportObject[<|...|>]`` with the documented properties, outcomes and failure types.
"""
from __future__ import annotations

import hashlib
import itertools
import os
import re
import signal
import sys
import threading
import time
import uuid
from contextvars import ContextVar

import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, record, L
from .wl_common import is_sym

AREA = "messages, exceptions & testing (W02)"

NULL = None
FAILED = Symbol("$Failed")
ABORTED = Symbol("$Aborted")
CANCELED = Symbol("$Canceled")
TIMEDOUT = Symbol("$TimedOut")
INFINITY = Symbol("Infinity")
ALL = Symbol("All")
NONE = Symbol("None")
AUTOMATIC = Symbol("Automatic")


def ev(x):
    from ..evaluate import evaluate
    return evaluate(x, 1)


def _hold_all(*names):
    from ..evaluate import set_hold_all
    for n in names:
        set_hold_all(n)


def _hold_first(*names):
    from ..evaluate import set_hold_first
    for n in names:
        set_hold_first(n)


def H(head, *args):
    return Expr(head, *args)


def assoc(*pairs):
    return Expr("Association", *(Expr("Rule", k, v) for k, v in pairs))


def assoc_items(a):
    """(key, value, delayed) triples of an association."""
    out = []
    for r in a.args:
        if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2:
            out.append((r.args[0], r.args[1], r.head == "RuleDelayed"))
    return out


def assoc_get(a, key, default=None):
    for k, v, d in assoc_items(a):
        if K.same(k, key):
            return ev(v) if d else v
    return default


def assoc_get_raw(a, key):
    """The stored rule (Rule or RuleDelayed) for key, or None."""
    for r in a.args:
        if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and len(r.args) == 2 and K.same(r.args[0], key):
            return r
    return None


def is_assoc(x):
    return KL.is_assoc(x)


def missing(*args):
    return Expr("Missing", *args)


def _state_data():
    """Per-session message/exception/assertion state, attached to the kernel's SessionState."""
    st = K.state()
    d = getattr(st, "_w02", None)
    if d is None:
        d = {"switch": {}, "assert": False, "exc_types": {}, "trace_off": set()}
        st._w02 = d
    return d


def _own(name, default):
    own = K.state().own
    if name in own:
        return ev(Symbol(name))
    return default


_PATTERN_HEADS = {"Blank", "BlankSequence", "BlankNullSequence", "Pattern", "Alternatives", "Condition",
                  "PatternTest", "Optional", "Except", "HoldPattern", "Verbatim", "Repeated", "RepeatedNull",
                  "OrderlessPatternSequence", "PatternSequence"}


def _has_pattern(*xs):
    """True if an argument is (or contains) a pattern: the expression is a pattern for MatchQ / Cases, not data."""
    for x in xs:
        if isinstance(x, Expr):
            if isinstance(x.head, str) and x.head in _PATTERN_HEADS:
                return True
            if _has_pattern(*x.args):
                return True
    return False


# ─────────────────────────── output ───────────────────────────

def _strings_to_symbols(e):
    """OutputForm rendering helper: strings print without quotes."""
    if isinstance(e, str):
        return Symbol(e)
    if isinstance(e, Expr):
        return Expr(_strings_to_symbols(e.head) if isinstance(e.head, Expr) else e.head,
                    *(_strings_to_symbols(a) for a in e.args))
    return e


def input_text(e):
    from ..formatting import input_form
    if isinstance(e, Expr) and e.head == "MessageName" and len(e.args) >= 2 and isinstance(e.args[0], Symbol) \
            and all(isinstance(t, str) for t in e.args[1:]):
        return "::".join([e.args[0].name] + [t if re.fullmatch(r"[A-Za-z$][A-Za-z0-9$]*", t) else '"%s"' % t
                                             for t in e.args[1:]])
    if isinstance(e, Expr) and e.head == "HoldForm" and len(e.args) == 1:
        return input_text(e.args[0])
    return input_form(e)


def output_text(e):
    if isinstance(e, str):
        return e
    from ..formatting import input_form
    return input_form(_strings_to_symbols(e))


def _streams(name, default):
    v = _own(name, None)
    if v is None:
        v = default
    if not K.is_list(v):
        v = L(v)
    out = []
    for s in v.args:
        if isinstance(s, Expr) and s.head == "OutputStream" and s.args and s.args[0] in ("stdout", "stderr"):
            out.append(s.args[0])
        else:
            raise NotImplementedError(f"{name}: only OutputStream[\"stdout\", 1] and OutputStream[\"stderr\", 2] "
                                      f"are supported as output channels (got {input_text(s)})")
    return out


def _write(channels, text):
    for c in channels:
        stream = sys.stdout if c == "stdout" else sys.stderr
        stream.write(text + "\n")
        stream.flush()


STDOUT = Expr("OutputStream", "stdout", 1)


def print_line(text):
    _write(_streams("$Output", L(STDOUT)), text)


# ─────────────────────────── message texts ───────────────────────────

BUILTIN_MESSAGES = {
    ("General", "argx"): "`1` called with `2` arguments; 1 argument is expected.",
    ("General", "argr"): "`1` called with 1 argument; `2` arguments are expected.",
    ("General", "argrx"): "`1` called with `2` arguments; `3` arguments are expected.",
    ("General", "argt"): "`1` called with `2` arguments; `3` or `4` arguments are expected.",
    ("General", "argb"): "`1` called with `2` arguments; between `3` and `4` arguments are expected.",
    ("General", "stop"): "Further output of `1` will be suppressed during this calculation.",
    ("General", "evalerr"): "`1`",
    ("General", "noopen"): "Cannot open `1`.",
    ("General", "spell"): "Possible spelling error: new symbol name \"`1`\" is similar to existing symbol \"`2`\".",
    ("General", "spell1"): "Possible spelling error: new symbol name \"`1`\" is similar to existing symbol \"`2`\".",
    ("Assert", "asrtf"): "Assertion `1` failed.",
    ("Assert", "asrtfl"): "Assertion `1` at line `2` in `3` failed.",
    ("Exception", "untagged"): "The construction of the untagged exception from general expression `1` is not "
                               "supported. Please provide some exception tag.",
    ("Exception", "invargs"): "Invalid type and / or number of arguments passed to ThrowException or Exception "
                              "constructor. The arguments were `1`.",
    ("Exception", "rsrvd"): "Warning: the user-provided exception data with keys `1` will be ignored / overridden "
                            "by the system.",
    ("ThrowException", "uncaught"): "Uncaught exception of type `1` propagated to the top level",
    ("RegisterExceptionType", "regd"): "`1` is already registered as an exception type.",
    ("RegisterExceptionType", "type"): "`1` is not a valid exception type; exception types are strings or symbols.",
    ("Pause", "numnm"): "Non-negative machine-sized number expected at position 1 in `1`.",
    ("TimeConstrained", "timc"): "Number of seconds `1` is not a non-negative real number or Infinity.",
    ("MemoryConstrained", "ipnfm"): "`1` is not a non-negative integer or Infinity.",
    ("RepeatedTiming", "timc"): "Number of seconds `1` is not a non-negative real number.",
    ("ClearSystemCache", "type"): "`1` is not a valid cache type; use \"Numeric\" or \"Symbolic\".",
    ("Message", "name"): "Message name `1` is not of the form symbol::name or symbol::name::language.",
    ("Off", "name"): "`1` is not a message name, a message group name or a symbol.",
    ("On", "name"): "`1` is not a message name, a message group name or a symbol.",
    ("Quiet", "anmlist"): "Argument `1` at position `2` should be All, None, a message name, a message group name, "
                          "or a list of these.",
    ("Check", "name"): "Argument `1` at position 3 should be a message name, a message group name, or a list of "
                       "these.",
}
for _c in ("Confirm", "ConfirmBy", "ConfirmMatch", "ConfirmQuiet", "ConfirmAssert"):
    BUILTIN_MESSAGES[(_c, "confirmnotag")] = "`1` has no tag or surrounding Enclose."

DEFAULT_GROUPS = [("Compiler", []), ("Graphics", []), ("Obsolete", []), ("Packing", []),
                  ("Spelling", [("General", "spell"), ("General", "spell1")]), ("Symbolics", [])]


def mname(s, t):
    return Expr("MessageName", Symbol(s), t)


def _default_groups():
    return L(*(Expr("RuleDelayed", g, L(*(mname(s, t) for s, t in names))) for g, names in DEFAULT_GROUPS))


def _lookup_text(s, t, lang=None):
    """Text of s::t (or s::t::lang): user value, built-in text, None if undefined."""
    q = Expr("MessageName", Symbol(s), t, lang) if lang else Expr("MessageName", Symbol(s), t)
    v = ev(q)
    if isinstance(v, str):
        return v
    if isinstance(v, Expr) and v.head == "MessageName":
        return None
    return output_text(v)


def message_text(s, t, lang=None):
    if lang is not None:
        return _lookup_text(s, t, lang)
    for attempt in range(2):
        for src in (s, "General"):
            txt = _lookup_text(src, t)
            if txt is not None:
                return txt
        nm = _own("$NewMessage", None)
        if nm is None or attempt:
            return None
        ev(K.call(nm, Symbol(s), t))
    return None


def _render_arg(a):
    pp = _own("$MessagePrePrint", AUTOMATIC)
    if not is_sym(pp, "Automatic"):
        a = ev(K.call(pp, Expr("HoldForm", a)))
    return input_text(a)


def string_form(template, args, render=_render_arg):
    """StringForm: `n` inserts argument n, `` inserts the next argument."""
    out, i, seq = [], 0, 0
    pattern = re.compile(r"`(\d*)`")
    for m in pattern.finditer(template):
        out.append(template[i:m.start()])
        if m.group(1):
            k = int(m.group(1))
            seq = k
        else:
            seq += 1
            k = seq
        out.append(render(args[k - 1]) if 1 <= k <= len(args) else m.group(0))
        i = m.end()
    out.append(template[i:])
    return "".join(out)


def _key_of(e, fname, what="message name"):
    """(symbol, tag) of s::t (HoldForm wrappers and Hold allowed)."""
    while isinstance(e, Expr) and e.head in ("HoldForm", "Hold", "HoldPattern") and len(e.args) == 1:
        e = e.args[0]
    if isinstance(e, Expr) and e.head == "MessageName" and len(e.args) in (2, 3) and isinstance(e.args[0], Symbol) \
            and all(isinstance(t, str) for t in e.args[1:]):
        return (e.args[0].name, e.args[1])
    return None


class MessageSpec:
    """A message specification: All, None, s::t, "group", or a list of these (General::t matches every s::t)."""

    def __init__(self, spec, fname, position=2):
        self.all = False
        self.keys = set()
        self._add(spec, fname, position, top=True)

    def _add(self, spec, fname, position, top=False):
        if is_sym(spec, "All") and top:
            self.all = True
            return
        if is_sym(spec, "None") and top:
            return
        k = _key_of(spec, fname)
        if k is not None:
            self.keys.add(k)
            return
        if isinstance(spec, str):
            groups = _own("$MessageGroups", None)
            groups = _default_groups() if groups is None else groups
            for r in (groups.args if K.is_list(groups) else ()):
                if isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed") and r.args[0] == spec:
                    members = r.args[1]
                    members = members if K.is_list(members) else L(members)
                    for m in members.args:
                        self._add(m, fname, position)
                    return
            raise ValueError(f"{fname}: unknown message group \"{spec}\"; known groups are given by $MessageGroups")
        if K.is_list(spec):
            for x in spec.args:
                if is_sym(x, "All"):
                    self.all = True
                else:
                    self._add(x, fname, position)
            return
        raise ValueError(f"{fname}: argument {input_text(spec)} at position {position} should be All, None, a message "
                         f"name s::t, a message group name, or a list of these")

    def matches(self, key):
        return self.all or key in self.keys or ("General", key[1]) in self.keys


def _held_spec(x):
    """Evaluate a held message specification without evaluating message names (s::t stays a name)."""
    if isinstance(x, Expr) and x.head == "MessageName":
        return x
    if isinstance(x, Expr) and x.head == "List":
        return L(*(_held_spec(a) for a in x.args))
    return ev(x)


# ── frames: Quiet, Check, message collectors (tests), ConfirmQuiet ──

class _Quiet:
    def __init__(self, off, on):
        self.off, self.on = off, on

    def decide(self, key):
        if self.on is not None and self.on.matches(key):
            return False
        if self.off.matches(key):
            return True
        return None


class _Check:
    def __init__(self, spec):
        self.spec = spec
        self.triggered = False


class _Collect:
    def __init__(self):
        self.messages = []          # (key, HoldForm[Message[s::t, args...]])


class _ConfirmQuietFrame:
    def __init__(self, spec):
        self.spec = spec


_FRAMES: ContextVar[tuple] = ContextVar("w02_message_frames", default=())


class _Frame:
    def __init__(self, frame):
        self.frame = frame

    def __enter__(self):
        self.token = _FRAMES.set(_FRAMES.get() + (self.frame,))
        return self.frame

    def __exit__(self, *exc):
        _FRAMES.reset(self.token)
        return False


class ConfirmQuietThrow(Exception):
    def __init__(self, frame, key, call):
        super().__init__("ConfirmQuiet")
        self.frame, self.key, self.call = frame, key, call


def _switched_off(key):
    sw = _state_data()["switch"]
    v = sw.get(key)
    if v is not None:
        return not v
    return sw.get(("General", key[1])) is False


def issue(s, t, *args, lang=None):
    """Issue the message s::t with arguments (Wolfram's Message)."""
    key = (s, t)
    if _switched_off(key):
        return
    call = Expr("HoldForm", Expr("Message", mname(s, t), *(Expr("HoldForm", a) for a in args)))
    suppressed = None
    for f in reversed(_FRAMES.get()):
        if isinstance(f, _Quiet):
            if suppressed is None:
                suppressed = f.decide(key)
        elif isinstance(f, _Check):
            if not suppressed and f.spec.matches(key):
                f.triggered = True
        elif isinstance(f, _ConfirmQuietFrame):
            if not suppressed and f.spec.matches(key):
                raise ConfirmQuietThrow(f, key, call)
        elif isinstance(f, _Collect):
            if not suppressed:
                f.messages.append((key, call))
            suppressed = True
    if suppressed:
        return
    text = message_text(s, t, lang)
    if text is None:
        body = "-- Message text not found --" + "".join(f" ({_render_arg(a)})" for a in args)
    else:
        body = string_form(text, list(args))
    name = f"{s}::{t}" + (f"::{lang}" if lang else "")
    _write(_streams("$Messages", L(STDOUT)), f"{name}: {body}")
    own = K.state().own
    cur = own.get("$MessageList")
    cur = cur if K.is_list(cur) else L()
    own["$MessageList"] = L(*cur.args, Expr("HoldForm", mname(s, t)))


# PermissionError: a capability the Session did not grant (allow_network, allow_python); Check catches it too.
_ERRORS = (ValueError, NotImplementedError, TypeError, ArithmeticError, RecursionError, KeyError, IndexError,
           PermissionError)


def is_error(exc):
    return isinstance(exc, _ERRORS) and not isinstance(exc, K.ThrowSignal)


def error_key(exc):
    """The message name (f, "evalerr") a raised MikoshiLang error stands for (f taken from the error text)."""
    text = str(exc) or type(exc).__name__
    m = re.match(r"\s*([A-Z$][A-Za-z0-9$]*)\s*(?::|\[|\s+called)", text)
    if m:
        return (m.group(1), "evalerr")
    m = re.match(r"\s*([A-Z$][A-Za-z0-9$]*)\s", text)
    if m and _known_function(m.group(1)):
        return (m.group(1), "evalerr")
    return ("General", "evalerr")


def _known_function(name):
    from .core import REGISTRY
    return name in REGISTRY or name in getattr(K, "_BUILTINS", {})


def issue_error(exc):
    """Report a raised MikoshiLang error as the message f::evalerr."""
    key = error_key(exc)
    issue(key[0], "evalerr", str(exc) or type(exc).__name__)
    return key


# ─────────────────────────── Message, MessageName, Off/On, Quiet, Check ───────────────────────────

@fn("Message", area=AREA, status="added",
    notes="Message[s::t, e1, ...] prints s::t: text (StringForm of the message text with the ei) on $Messages unless "
          "switched off or quieted, appends HoldForm[s::t] to $MessageList and returns Null; the text is looked up as "
          "s::t, then General::t, then via $NewMessage, else '-- Message text not found --'")
def _message(name, *args):
    while isinstance(name, Expr) and name.head == "HoldForm" and len(name.args) == 1:
        name = name.args[0]
    if not (isinstance(name, Expr) and name.head == "MessageName" and len(name.args) in (2, 3)
            and isinstance(name.args[0], Symbol) and all(isinstance(t, str) for t in name.args[1:])):
        issue("Message", "name", name)
        raise Unevaluated()
    issue(name.args[0].name, name.args[1], *args,
          lang=name.args[2] if len(name.args) == 3 else None)
    return NULL


@fn("MessageName", area=AREA, status="added",
    notes="s::t (parsed as MessageName[s, \"t\"], HoldFirst) evaluates to the message text assigned with "
          "s::t = \"text\" or to MikoshiLang's built-in text; otherwise it stays unevaluated")
def _message_name(s, t, *lang):
    if isinstance(s, Symbol) and isinstance(t, str) and not lang:
        txt = BUILTIN_MESSAGES.get((s.name, t))
        if txt is not None:
            return txt
    raise Unevaluated()


def _switch_targets(args, fname):
    """Resolve Off/On arguments to (message keys, symbols)."""
    keys, syms = [], []
    for a in args:
        k = _key_of(a, fname)
        if k is not None:
            keys.append(k)
            continue
        if isinstance(a, Symbol):
            syms.append(a.name)
            continue
        v = _held_spec(a)
        if isinstance(v, str) or K.is_list(v):
            spec = MessageSpec(v, fname)
            keys.extend(sorted(spec.keys))
            continue
        issue(fname, "name", a)
        raise Unevaluated()
    return keys, syms


@fn("Off", area=AREA, status="added",
    notes="Off[s::t, ...] switches messages off (Off[General::t] switches off every s::t that has not been switched "
          "on explicitly); Off[\"group\"] uses $MessageGroups; Off[Assert] disables assertions; Off[s] (tracing) and "
          "Off[] are accepted and change nothing because symbol tracing is never on")
def _off(*args):
    keys, syms = _switch_targets(args, "Off")
    d = _state_data()
    for k in keys:
        d["switch"][k] = False
    for s in syms:
        if s == "Assert":
            d["assert"] = False
        else:
            d["trace_off"].add(s)
    return NULL


@fn("On", area=AREA, status="added",
    notes="On[s::t, ...] and On[\"group\"] switch messages back on; On[Assert] enables assertions.  On[s] for symbol "
          "tracing is not supported (the evaluator has no per-symbol trace printing) and raises")
def _on(*args):
    keys, syms = _switch_targets(args, "On")
    if not args:
        raise NotImplementedError("On[]: symbol tracing is not supported in MikoshiLang; use Trace or Echo")
    d = _state_data()
    for s in syms:
        if s != "Assert":
            raise NotImplementedError(f"On[{s}]: symbol tracing is not supported in MikoshiLang; use Trace or Echo")
    for k in keys:
        d["switch"][k] = True
    if "Assert" in syms:
        d["assert"] = True
    return NULL


@fn("Quiet", area=AREA, status="added",
    notes="Quiet[expr] / Quiet[expr, moff] / Quiet[expr, moff, mon] (HoldAll): messages in moff are not printed, "
          "those in mon are; the innermost Quiet specification for a message wins; Check outside is unaffected. "
          "A raised MikoshiLang error is not a message and still propagates")
def _quiet(expr, *specs):
    if len(specs) > 2:
        raise ValueError("Quiet called with more than 3 arguments; 1, 2 or 3 arguments are expected")
    off = MessageSpec(_held_spec(specs[0]), "Quiet", 2) if specs else MessageSpec(ALL, "Quiet")
    on = MessageSpec(_held_spec(specs[1]), "Quiet", 3) if len(specs) == 2 else None
    with _Frame(_Quiet(off, on)):
        return ev(expr)


@fn("Check", area=AREA, status="added",
    notes="Check[expr, failexpr] / Check[expr, failexpr, msgs] (HoldAll): returns failexpr if expr generated a "
          "matching message that is not switched off and not quieted inside the Check; a raised MikoshiLang error "
          "counts as the message f::evalerr")
def _check(expr, failexpr, *spec):
    if len(spec) > 1:
        raise ValueError("Check called with more than 3 arguments; 2 or 3 arguments are expected")
    ms = MessageSpec(_held_spec(spec[0]), "Check", 3) if spec else MessageSpec(ALL, "Check")
    with _Frame(_Check(ms)) as frame:
        try:
            value = ev(expr)
        except Exception as exc:  # noqa: BLE001 - only MikoshiLang errors are handled
            if not is_error(exc):
                raise
            if not ms.matches(error_key(exc)):
                raise
            issue_error(exc)
            return ev(failexpr)
    return ev(failexpr) if frame.triggered else value


@fn("Messages", area=AREA, status="added",
    notes="Messages[s] gives {HoldPattern[s::t] :> text, ...} for the messages assigned to s (sorted by tag); "
          "messages switched off are given as HoldPattern[s::t] :> $Off[text]")
def _messages(s):
    if not isinstance(s, Symbol):
        raise ValueError(f"Messages: {input_text(s)} is not a symbol")
    found = {}
    st = K.state()
    for lhs, rhs in st.exact.values():
        if isinstance(lhs, Expr) and lhs.head == "MessageName" and lhs.args and lhs.args[0] == s:
            found[tuple(lhs.args[1:])] = (lhs, rhs)
    out = []
    for tags in sorted(found):
        lhs, rhs = found[tags]
        if _switched_off((s.name, tags[0])):
            rhs = Expr("$Off", rhs)
        out.append(Expr("RuleDelayed", Expr("HoldPattern", lhs), rhs))
    return L(*out)


_hold_first("Messages", "MessageName", "Message")


def _assign_messagename(lhs, rhs, ev_):
    """s::t = text (and :=) stores a message for s.  As in Wolfram this is allowed for Protected symbols too
    (General::tag, built-in heads): messages are not definitions of the head MessageName or of s."""
    if not isinstance(lhs, Expr):
        raise ValueError("Symbol MessageName is protected")
    if not (len(lhs.args) in (2, 3) and isinstance(lhs.args[0], Symbol)):
        raise ValueError(f"Set: {input_text(lhs)} is not a valid message name")
    K.store_definition(Expr("MessageName", lhs.args[0], *[ev_(a) for a in lhs.args[1:]]), rhs)


K.ASSIGN_HOOKS["MessageName"] = _assign_messagename
_hold_all("Off", "On", "Quiet", "Check")


# ─────────────────────────── system symbols ───────────────────────────

def _install_constants():
    from .wl_numerics import SYSTEM_CONSTANTS
    SYSTEM_CONSTANTS.setdefault("$MessageList", L())
    SYSTEM_CONSTANTS.setdefault("$Messages", L(STDOUT))
    SYSTEM_CONSTANTS.setdefault("$MessagePrePrint", AUTOMATIC)
    SYSTEM_CONSTANTS.setdefault("$MessageGroups", _default_groups())
    SYSTEM_CONSTANTS.setdefault("$TestFileName", "")


_install_constants()

for _n, _note in [
    ("$Failed", "inert symbol returned when an operation cannot be done; FailureQ[$Failed] is True"),
    ("$Aborted", "inert symbol: the result of an aborted computation (MemoryConstrained, TimeConstrained, "
                 "CheckAbort); FailureQ[$Aborted] is True"),
    ("$TimedOut", "inert symbol for an operation that timed out"),
    ("$Canceled", "inert symbol for a canceled input; Confirm treats it as a failure"),
    ("$MessageList", "list of HoldForm[s::t] for messages printed in the session ({} initially; assignable and "
                     "Block-able; Quiet-ed and switched-off messages are not added)"),
    ("$Messages", "output channels for messages: {OutputStream[\"stdout\", 1]} by default; assign {} to silence "
                  "messages or OutputStream[\"stderr\", 2] to redirect them"),
    ("$MessagePrePrint", "Automatic by default; if set to a function f, message arguments are rendered as "
                         "f[HoldForm[arg]]"),
    ("$MessageGroups", "{\"name\" :> {s::t, ...}, ...} named message groups used by On, Off, Quiet and Check "
                       "(assignable; the predefined groups list the messages MikoshiLang can issue)"),
    ("$NewMessage", "if set, $NewMessage[s, \"t\"] is evaluated when the text of s::t is not defined, and the "
                    "lookup is retried"),
    ("$AssertFunction", "if set, $AssertFunction[HoldComplete[Assert[test, ...]]] is evaluated when an enabled "
                        "assertion fails, instead of issuing Assert::asrtf"),
    ("$TestFileName", "\"\" interactively; the absolute file name while TestReport[file] runs the file's tests"),
]:
    record(_n, AREA, "added", _note)


# ─────────────────────────── printing and echo ───────────────────────────

_QUIET_ECHO: ContextVar[bool] = ContextVar("w02_quiet_echo", default=False)
_ECHO_DEPTH: ContextVar[int] = ContextVar("w02_echo_depth", default=0)


@fn("Print", area=AREA, status="added",
    notes="Print[e1, e2, ...] writes the ei concatenated (OutputForm: strings without quotes) and a newline to "
          "$Output (stdout) and returns Null")
def _print(*args):
    print_line("".join(output_text(a) for a in args))
    return NULL


@fn("PrintTemporary", area=AREA, status="added",
    notes="text interface: PrintTemporary[e1, ...] prints like Print (there are no notebook cells to delete "
          "afterwards) and returns Null")
def _print_temporary(*args):
    print_line("".join(output_text(a) for a in args))
    return NULL


def echo(value, label=None, f=None):
    shown = value if f is None else ev(K.call(f, value))
    if not _QUIET_ECHO.get():
        print_line(">> " + (f"{input_text(label)}  " if label is not None else "") + input_text(shown))
    return value


@fn("Echo", area=AREA, status="added",
    notes="Echo[expr] prints >> expr and returns expr; Echo[expr, label] prints >> label  expr; Echo[expr, label, f] "
          "prints f[expr]; output is suppressed inside QuietEcho")
def _echo(x, label=None, f=None):
    return echo(x, label, f)


def _echo_label_hook(head, args, ev_):
    if len(args) != 1:
        return K.UNHANDLED
    if len(head.args) > 1:
        raise ValueError("EchoLabel[label] takes one label")
    return echo(args[0], head.args[0] if head.args else None)


def _echo_function_hook(head, args, ev_):
    if len(args) != 1:
        return K.UNHANDLED
    if len(head.args) == 0:
        return echo(args[0])
    if len(head.args) == 1:
        return echo(args[0], None, head.args[0])
    if len(head.args) == 2:
        return echo(args[0], head.args[0], head.args[1])
    raise ValueError("EchoFunction[f] or EchoFunction[label, f] expected")


@fn("EchoLabel", area=AREA, status="added",
    notes="operator form: EchoLabel[label][expr] prints >> label  expr and returns expr; EchoLabel[][expr] is Echo[expr]")
def _echo_label(*label):
    raise Unevaluated()


@fn("EchoFunction", area=AREA, status="added",
    notes="operator form: EchoFunction[f][expr] prints >> f[expr], EchoFunction[label, f][expr] prints >> label  "
          "f[expr]; both return expr; EchoFunction[][expr] is Echo[expr]")
def _echo_function(*args):
    raise Unevaluated()


def _labels(spec):
    if spec is None:
        return None, None
    if isinstance(spec, Expr) and spec.head == "Rule" and len(spec.args) == 2:
        return spec.args[0], spec.args[1]
    return spec, spec


@fn("EchoEvaluation", area=AREA, status="added",
    notes="EchoEvaluation[expr] (HoldFirst) prints << expr before and >> result after evaluating (nested calls use "
          "<<<< / >>>>); EchoEvaluation[expr, l1 -> l2], EchoEvaluation[expr, labels, f] and "
          "EchoEvaluation[expr, labels, g -> f] as documented; returns the result")
def _echo_evaluation(expr, labels=None, f=None):
    labels = None if labels is None else ev(labels)
    f = None if f is None else ev(f)
    l1, l2 = _labels(labels)
    g = None
    if isinstance(f, Expr) and f.head == "Rule" and len(f.args) == 2:
        g, f = f.args
    depth = _ECHO_DEPTH.get() + 1
    if not _QUIET_ECHO.get():
        before = expr if g is None or is_sym(g, "Unevaluated") else ev(Expr(g, expr))
        print_line("<<" * depth + " " + (f"{input_text(l1)}  " if l1 is not None else "") + input_text(before))
    tok = _ECHO_DEPTH.set(depth)
    try:
        result = ev(expr)
    finally:
        _ECHO_DEPTH.reset(tok)
    if not _QUIET_ECHO.get():
        shown = result if f is None else ev(K.call(f, result))
        print_line(">>" * depth + " " + (f"{input_text(l2)}  " if l2 is not None else "") + input_text(shown))
    return result


@fn("QuietEcho", area=AREA, status="added",
    notes="QuietEcho[expr] (HoldFirst) evaluates expr without Echo, EchoLabel, EchoFunction, EchoEvaluation or "
          "EchoTiming printing anything; messages and Print are not affected")
def _quiet_echo(expr):
    tok = _QUIET_ECHO.set(True)
    try:
        return ev(expr)
    finally:
        _QUIET_ECHO.reset(tok)


@fn("Monitor", area=AREA, status="added",
    notes="Monitor[expr, mon] (HoldAll) evaluates and returns expr; the monitor display is a notebook feature, so in "
          "MikoshiLang's text interface nothing is shown (as in a Wolfram text-based session)")
def _monitor(expr, *rest):
    if len(rest) > 2:
        raise ValueError("Monitor called with more than 3 arguments; 1, 2 or 3 arguments are expected")
    return ev(expr)


_hold_first("EchoEvaluation", "QuietEcho")
_hold_all("Monitor")
KL.COMPOUND_HEAD_HOOKS["EchoLabel"] = _echo_label_hook
KL.COMPOUND_HEAD_HOOKS["EchoFunction"] = _echo_function_hook


# ─────────────────────────── Assert ───────────────────────────

@fn("Assert", area=AREA, status="added",
    notes="Assert[test] / Assert[test, tag] (HoldAll): disabled by default (returns Null without evaluating test); "
          "after On[Assert] a test that is not True issues Assert::asrtf (Assert::asrtfl for a {line, file} tag) or "
          "evaluates $AssertFunction[HoldComplete[Assert[...]]] when that is set; returns Null")
def _assert(test, *tag):
    if len(tag) > 1:
        raise ValueError("Assert called with more than 2 arguments; 1 or 2 arguments are expected")
    if not _state_data()["assert"]:
        return NULL
    if ev(test) is True:
        return NULL
    af = _own("$AssertFunction", None)
    if af is not None:
        ev(K.call(af, Expr("HoldComplete", Expr("Assert", test, *tag))))
        return NULL
    t = ev(tag[0]) if tag else None
    if K.is_list(t) and len(t.args) == 2:
        issue("Assert", "asrtfl", Expr("HoldForm", test), t.args[0], t.args[1])
    else:
        issue("Assert", "asrtf", Expr("HoldForm", test))
    return NULL


_hold_all("Assert")


# ─────────────────────────── Failure, Success, FailureQ, Failsafe ───────────────────────────

def _template_text(tpl):
    """Message template: a string, or s::t (held or evaluated) -> its text."""
    if isinstance(tpl, str):
        return tpl
    k = _key_of(tpl, "Failure")
    if k is not None:
        txt = message_text(*k)
        if txt is None:
            raise ValueError(f"Failure: the message {k[0]}::{k[1]} used as MessageTemplate has no text")
        return txt
    if isinstance(tpl, Expr) and tpl.head == "TemplateObject":
        raise NotImplementedError("Failure: TemplateObject message templates are not supported; use a string template")
    raise ValueError(f"Failure: MessageTemplate {input_text(tpl)} is not a string or a message name")


def template_apply(template, params):
    """TemplateApply for message templates: `` and `n` take positional parameters, `name` named ones."""
    pos = list(params.args) if K.is_list(params) else []
    named = {k: v for k, v, _d in assoc_items(params)} if is_assoc(params) else {}
    out, i, seq = [], 0, 0
    for m in re.finditer(r"`([^`]*)`", template):
        out.append(template[i:m.start()])
        key = m.group(1)
        if key == "" or key.isdigit():
            if key:
                seq = int(key)
                k = seq
            else:
                seq += 1
                k = seq
            if named and not pos:
                v = named.get(k, named.get(str(k)))
                out.append(output_text(v) if v is not None else m.group(0))
            else:
                out.append(output_text(pos[k - 1]) if 1 <= k <= len(pos) else m.group(0))
        else:
            v = named.get(key)
            out.append(output_text(v) if v is not None else m.group(0))
        i = m.end()
    out.append(template[i:])
    return "".join(out)


def _object_message(a):
    raw = assoc_get_raw(a, "MessageTemplate")
    if raw is None:
        return None
    tpl = raw.args[1]
    params = assoc_get(a, "MessageParameters", L())
    if isinstance(params, Expr) and params.head == "HoldForm":
        params = params.args[0] if params.args else L()
    params = ev(params) if not (K.is_list(params) or is_assoc(params)) else params
    return template_apply(_template_text(tpl if raw.head == "Rule" else tpl), params)


def _valid_failure(e, head):
    return isinstance(e, Expr) and e.head == head and len(e.args) == 2 and is_assoc(e.args[1])


def _object_property(obj, prop, head):
    a = obj.args[1]
    if isinstance(prop, Expr) and prop.head == "List":
        return L(*(_object_property(obj, p, head) for p in prop.args))
    if not isinstance(prop, str):
        return K.UNHANDLED
    if prop == "Tag":
        return obj.args[0]
    if prop == "Properties":
        keys = [k for k, _v, _d in assoc_items(a) if isinstance(k, str)]
        if head == "Success":
            return L(*keys)
        props = set(keys)
        raw = assoc_get_raw(a, "MessageTemplate")
        if raw is not None:
            props.add("Message")
            if _key_of(raw.args[1], head) is not None:
                props.update({"MessageName", "HeldMessageTemplate"})
        return L(*sorted(props))
    if prop == "Message":
        explicit = assoc_get(a, "Message")
        if explicit is not None:
            return explicit
        m = _object_message(a)
        return missing("NotAvailable") if m is None else m
    if prop == "MessageName":
        raw = assoc_get_raw(a, "MessageTemplate")
        if raw is not None and _key_of(raw.args[1], head) is not None:
            return raw.args[1]
        return missing("NotAvailable")
    if prop == "HeldMessageTemplate":
        raw = assoc_get_raw(a, "MessageTemplate")
        return Expr("Hold", raw.args[1]) if raw is not None else missing("NotAvailable")
    if prop == "MessageTemplate":
        raw = assoc_get_raw(a, "MessageTemplate")
        if raw is None:
            return missing("KeyAbsent", prop)
        k = _key_of(raw.args[1], head)
        return message_text(*k) if k is not None else raw.args[1]
    v = assoc_get(a, prop)
    return missing("KeyAbsent", prop) if v is None and assoc_get_raw(a, prop) is None else v


def _failure_hook(head, args, ev_):
    if not _valid_failure(head, head.head) or len(args) != 1:
        return K.UNHANDLED
    prop = args[0]
    if is_sym(prop, "All"):
        names = _object_property(head, "Properties", head.head)
        return assoc(*((p, _object_property(head, p, head.head)) for p in names.args))
    return _object_property(head, prop, head.head)


@fn("Failure", area=AREA, status="added",
    notes="Failure[\"tag\", assoc] is inert; failure[\"prop\"] gives assoc values and the derived properties "
          "\"Message\" (template applied to \"MessageParameters\"), \"MessageTemplate\", \"MessageName\", "
          "\"HeldMessageTemplate\", \"Tag\" and \"Properties\" (sorted); failure[{p1, p2}] gives a list")
def _failure(*args):
    if _has_pattern(*args):
        raise Unevaluated()
    if len(args) == 2 and not is_assoc(args[1]):
        raise ValueError(f"Failure: the second argument {input_text(args[1])} must be an association")
    if len(args) not in (1, 2):
        raise ValueError(f"Failure called with {len(args)} arguments; 2 arguments are expected")
    raise Unevaluated()


@fn("Success", area=AREA, status="added",
    notes="Success[\"tag\", assoc] is inert; success[\"prop\"] gives assoc values, \"Message\", \"Tag\" and "
          "\"Properties\" (the association keys in order)")
def _success(*args):
    if _has_pattern(*args):
        raise Unevaluated()
    if len(args) == 2 and not is_assoc(args[1]):
        raise ValueError(f"Success: the second argument {input_text(args[1])} must be an association")
    if len(args) not in (1, 2):
        raise ValueError(f"Success called with {len(args)} arguments; 2 arguments are expected")
    raise Unevaluated()


KL.COMPOUND_HEAD_HOOKS["Failure"] = _failure_hook
KL.COMPOUND_HEAD_HOOKS["Success"] = _failure_hook


def failure_q(x):
    return (isinstance(x, Expr) and x.head == "Failure") or is_sym(x, "$Failed") or is_sym(x, "$Aborted")


def confirm_failure_q(x):
    """What Confirm and Failsafe treat as a failure: FailureQ, MissingQ or $Canceled."""
    return failure_q(x) or (isinstance(x, Expr) and x.head == "Missing") or is_sym(x, "$Canceled")


@fn("FailureQ", area=AREA, status="added",
    notes="FailureQ[expr] is True for Failure[...], $Failed and $Aborted, False otherwise")
def _failure_q(x):
    return failure_q(x)


def _failsafe_hook(head, args, ev_):
    n = len(head.args)
    if n == 0 or n > 3:
        return K.UNHANDLED
    f = head.args[0]
    if n == 1:
        for x in args:
            if confirm_failure_q(x):
                return x
        return ev(K.call(f, *args))
    test = ev(K.call(head.args[1], *args))
    if test is True:
        return ev(K.call(f, *args))
    if n == 3:
        return ev(K.call(head.args[2], *args))
    shown = args[0] if len(args) == 1 else Expr("Sequence", *args)
    return Expr("Failure", "FailsafeFailed", Expr("Association",
                Expr("Rule", "MessageTemplate", "The test failed on the arguments `1`."),
                Expr("RuleDelayed", "MessageParameters", L(Expr("HoldForm", shown))),
                Expr("Rule", "Arguments", Expr("Hold", *args))))


@fn("Failsafe", area=AREA, status="added",
    notes="operator form: Failsafe[f][x1, ...] gives f[x1, ...] unless some xi is a failure (Failure, Missing, "
          "$Failed, $Canceled, $Aborted), then the first such xi; Failsafe[f, test][...] gives "
          "Failure[\"FailsafeFailed\", ...] when test[x1, ...] is not True; Failsafe[f, test, failf] gives failf[...]")
def _failsafe(*args):
    if not 1 <= len(args) <= 3:
        raise ValueError(f"Failsafe called with {len(args)} arguments; 1, 2 or 3 arguments are expected")
    raise Unevaluated()


KL.COMPOUND_HEAD_HOOKS["Failsafe"] = _failsafe_hook


# ─────────────────────────── Enclose and the Confirm family ───────────────────────────

_CONFIRM_ARITY = {"Confirm": (1, 3), "ConfirmBy": (2, 4), "ConfirmMatch": (2, 4), "ConfirmQuiet": (1, 4),
                  "ConfirmAssert": (1, 3)}
_enclose_ids = itertools.count(1)


class ConfirmThrow(K.ThrowSignal):
    """Thrown by a failed confirmation; caught by the matching Enclose (or Catch[..., _])."""

    def __init__(self, failure, tag):
        super().__init__(failure, tag)
        self.failure = failure


def _lexical_tag(e):
    return isinstance(e, Expr) and e.head == "Enclose`LexicalTag"


def _tag_confirms(e, tag):
    """Add tag to untagged confirmation calls lexically inside e (not inside an inner untagged Enclose)."""
    if not isinstance(e, Expr):
        return e
    h = e.head
    if isinstance(h, str):
        if h == "Enclose" and len(e.args) <= 2:
            return e
        if h in _CONFIRM_ARITY:
            full = _CONFIRM_ARITY[h][1]
            args = [_tag_confirms(a, tag) for a in e.args]
            if len(args) < full and len(args) >= _CONFIRM_ARITY[h][0]:
                if h == "ConfirmQuiet" and len(args) == 1:
                    args.append(ALL)
                while len(args) < full - 1:
                    args.append(Symbol("Null"))
                args.append(tag)
            return Expr(h, *args)
    new_head = _tag_confirms(h, tag) if isinstance(h, Expr) else h
    return Expr(new_head, *(_tag_confirms(a, tag) for a in e.args))


def _confirmation_failure(template, params, ctype, fields, info):
    pairs = [Expr("Rule", "MessageTemplate", template), Expr("Rule", "MessageParameters", L(*params)),
             Expr("Rule", "ConfirmationType", ctype)]
    pairs += [Expr("Rule", k, v) for k, v in fields]
    pairs.append(Expr("Rule", "Information", info))
    return Expr("Failure", "ConfirmationFailed", Expr("Association", *pairs))


def _confirm_setup(fname, args):
    lo, hi = _CONFIRM_ARITY[fname]
    if not lo <= len(args) <= hi:
        raise ValueError(f"{fname} called with {len(args)} arguments; between {lo} and {hi} arguments are expected")
    if len(args) < hi:
        return None
    return ev(args[-1])


def _no_tag(fname, args):
    issue(fname, "confirmnotag", Symbol(fname))
    held = Expr(fname, *args)
    return Expr("Failure", "confirmnotag", Expr("Association",
                Expr("RuleDelayed", "MessageTemplate", mname(fname, "confirmnotag")),
                Expr("Rule", "MessageParameters", L(Symbol(fname))),
                Expr("Rule", "HeldInput", Expr("Hold", held))))


def _info(x):
    return ev(x)


def _is_null(x):
    return x is None or is_sym(x, "Null")


@fn("Confirm", area=AREA, status="added",
    notes="Confirm[expr] returns expr unless it is a Failure, Missing, $Failed, $Canceled or $Aborted, in which case "
          "Failure[\"ConfirmationFailed\", <|...|>] is thrown to the lexically enclosing Enclose (info is evaluated "
          "only on failure; a ConfirmationFailed failure with Null info is rethrown unchanged); Confirm[expr, info, "
          "tag] is caught dynamically by Enclose[..., f, tag]; with no tag and no Enclose: message confirmnotag")
def _confirm(*args):
    tag = _confirm_setup("Confirm", args)
    if tag is None:
        return _no_tag("Confirm", args)
    x = ev(args[0])
    if not confirm_failure_q(x):
        return x
    info = _info(args[1])
    if _is_null(info) and isinstance(x, Expr) and x.head == "Failure" and x.args and x.args[0] == "ConfirmationFailed":
        raise ConfirmThrow(x, tag)
    raise ConfirmThrow(_confirmation_failure("`` could not be confirmed.", [x], "Confirm", [("Expression", x)], info),
                       tag)


@fn("ConfirmBy", area=AREA, status="added",
    notes="ConfirmBy[expr, f] returns expr if f[expr] is True, else throws Failure[\"ConfirmationFailed\", ...] "
          "with \"Expression\" and \"Function\" to the enclosing Enclose; info/tag as for Confirm")
def _confirm_by(*args):
    tag = _confirm_setup("ConfirmBy", args)
    if tag is None:
        return _no_tag("ConfirmBy", args)
    x = ev(args[0])
    f = ev(args[1])
    if ev(K.call(f, x)) is True:
        return x
    info = _info(args[2])
    raise ConfirmThrow(_confirmation_failure("``[``] did not return True.", [f, x], "ConfirmBy",
                                             [("Expression", x), ("Function", f)], info), tag)


@fn("ConfirmMatch", area=AREA, status="added",
    notes="ConfirmMatch[expr, form] returns expr if it matches form, else throws Failure[\"ConfirmationFailed\", "
          "...] with \"Expression\" and \"Pattern\" to the enclosing Enclose; info/tag as for Confirm")
def _confirm_match(*args):
    tag = _confirm_setup("ConfirmMatch", args)
    if tag is None:
        return _no_tag("ConfirmMatch", args)
    x = ev(args[0])
    form = ev(args[1])
    if K.first_match(form, x, ev) is not None:
        return x
    info = _info(args[2])
    raise ConfirmThrow(_confirmation_failure("`` does not match ``.", [x, form], "ConfirmMatch",
                                             [("Expression", x), ("Pattern", form)], info), tag)


@fn("ConfirmAssert", area=AREA, status="added",
    notes="ConfirmAssert[test] returns Null if test is True, else throws Failure[\"ConfirmationFailed\", "
          "<|\"MessageTemplate\" -> \"Assertion failed.\", ..., \"Test\" -> value|>] to the enclosing Enclose")
def _confirm_assert(*args):
    tag = _confirm_setup("ConfirmAssert", args)
    if tag is None:
        return _no_tag("ConfirmAssert", args)
    t = ev(args[0])
    if t is True:
        return NULL
    info = _info(args[1])
    raise ConfirmThrow(_confirmation_failure("Assertion failed.", [t], "ConfirmAssert", [("Test", t)], info), tag)


@fn("ConfirmQuiet", area=AREA, status="added",
    notes="ConfirmQuiet[expr, mspec] evaluates expr and returns it unless a matching message is generated (not "
          "switched off and not quieted inside); evaluation stops at the first such message, which is quieted, and "
          "Failure[\"ConfirmationFailed\", ...] with \"HeldExpression\", \"HeldMessageName\" and "
          "\"HeldMessageCall\" is thrown to the enclosing Enclose; a raised MikoshiLang error counts as f::evalerr")
def _confirm_quiet(*args):
    tag = _confirm_setup("ConfirmQuiet", args)
    if tag is None:
        return _no_tag("ConfirmQuiet", args)
    spec = MessageSpec(_held_spec(args[1]), "ConfirmQuiet", 2)
    frame = _ConfirmQuietFrame(spec)
    try:
        with _Frame(frame):
            try:
                return ev(args[0])
            except Exception as exc:  # noqa: BLE001
                if not is_error(exc):
                    raise
                issue_error(exc)
                raise
    except ConfirmQuietThrow as cq:
        if cq.frame is not frame:
            raise
        info = _info(args[2])
        name = mname(*cq.key)
        call = cq.call.args[0]
        raise ConfirmThrow(_confirmation_failure(
            "`` produced a message on evaluation.", [Expr("HoldForm", args[0])], "ConfirmQuiet",
            [("HeldExpression", Expr("Hold", args[0])), ("HeldMessageName", Expr("Hold", name)),
             ("HeldMessageCall", Expr("Hold", call))], info), tag) from None


@fn("Enclose", area=AREA, status="added",
    notes="Enclose[expr] (HoldAll) evaluates expr; an untagged confirmation lexically inside throws its Failure to "
          "it (untagged confirmations are rewritten with a private tag, inner untagged Enclose blocks keep their "
          "own); Enclose[expr, f] gives f[failure] (f a string: failure[f]); Enclose[expr, f, form] catches only "
          "confirmations whose tag matches form, dynamically")
def _enclose(expr, *rest):
    if len(rest) > 2:
        raise ValueError(f"Enclose called with {len(rest) + 1} arguments; 1, 2 or 3 arguments are expected")
    handler = ev(rest[0]) if rest else Symbol("Identity")
    if len(rest) == 2:
        form = ev(rest[1])
        body = expr
        own_tag = None
    else:
        own_tag = Expr("Enclose`LexicalTag", next(_enclose_ids))
        body = _tag_confirms(expr, own_tag)
        form = None
    try:
        return ev(body)
    except ConfirmThrow as t:
        if own_tag is not None:
            if not (isinstance(t.tag, Expr) and K.same(t.tag, own_tag)):
                raise
        elif _lexical_tag(t.tag) or K.first_match(form, t.tag, ev) is None:
            raise
        failure = t.failure
    if isinstance(handler, str):
        return ev(K.call(failure, handler))
    return ev(K.call(handler, failure))


_hold_all("Enclose", "Confirm", "ConfirmBy", "ConfirmMatch", "ConfirmAssert", "ConfirmQuiet")


# ─────────────────────────── Exception system (Wolfram 15) ───────────────────────────

_RESERVED = ("ExceptionTagList", "ExceptionTag", "ExceptionValidated", "ExceptionAPIVersion")
_EXC_TAG = Expr("ThrowException`Tag")


class ExceptionThrow(K.ThrowSignal):
    def __init__(self, exc):
        super().__init__(exc, _EXC_TAG)
        self.exception = exc


def _valid_tag(t):
    return isinstance(t, (str, Symbol)) and not isinstance(t, bool)


def _types():
    return _state_data()["exc_types"]


def _type_key(t):
    return ("s", t) if isinstance(t, str) else ("y", t.name)


def _parents(t):
    entry = _types().get(_type_key(t))
    return entry[1] if entry else []


def _expand(tags):
    out, seen = [], set()
    queue = list(tags)
    while queue:
        t = queue.pop(0)
        k = _type_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
        queue.extend(_parents(t))
    return out


def _ancestors(t):
    return _expand([t])


def is_exception(x):
    if not (isinstance(x, Expr) and x.head == "Exception" and len(x.args) == 2 and K.is_list(x.args[0])
            and x.args[0].args and all(_valid_tag(t) for t in x.args[0].args) and is_assoc(x.args[1])):
        return False
    return assoc_get(x.args[1], "ExceptionValidated") is True


def _exc_parts(x):
    """(tag list, data pairs without system keys, failure tag or None)."""
    tags = list(x.args[0].args)
    data, ftag = [], None
    for k, v, d in assoc_items(x.args[1]):
        if k in ("ExceptionValidated", "ExceptionAPIVersion"):
            continue
        if k == "ExceptionFailureTag":
            ftag = v
            continue
        data.append((k, v, d))
    return tags, data, ftag


def _rule(k, v, delayed=False):
    return Expr("RuleDelayed" if delayed else "Rule", k, v)


def _make_exception(tags, data, ftag=None):
    rules = [_rule(k, v, d) for k, v, d in data]
    if ftag is not None and not (isinstance(ftag, str) and ftag == _tag_string(tags[0])):
        rules.append(_rule("ExceptionFailureTag", ftag))
    rules += [_rule("ExceptionValidated", True), _rule("ExceptionAPIVersion", "1")]
    return Expr("Exception", L(*tags), Expr("Association", *rules))


def _tag_string(t):
    return t if isinstance(t, str) else t.name


def _merge(data, payload, fname):
    """Add a payload (association or arbitrary expression) to exception data."""
    if is_assoc(payload):
        new = assoc_items(payload)
        bad = [k for k, _v, _d in new if k in _RESERVED]
        if bad:
            issue("Exception", "rsrvd", L(*bad))
        new = [(k, v, d) for k, v, d in new if k not in _RESERVED]
        out = list(data)
        for k, v, d in new:
            for i, (k0, _v0, _d0) in enumerate(out):
                if K.same(k0, k):
                    out[i] = (k, v, d)
                    break
            else:
                out.append((k, v, d))
        return out
    return [p for p in data if p[0] != "ExceptionPayload"] + [("ExceptionPayload", payload, False)]


class _BadException(Exception):
    def __init__(self, tag, args):
        super().__init__(tag)
        self.tag, self.args_ = tag, args


def build_exception(args, fname="Exception"):
    """Exception[spec] / Exception[spec, payload] -> validated Exception expression (raises _BadException)."""
    if len(args) not in (1, 2):
        raise _BadException("invargs", L(*args))
    spec = args[0]
    payload = args[1] if len(args) == 2 else None
    ftag = None
    if is_exception(spec):
        tags, data, ftag = _exc_parts(spec)
    elif isinstance(spec, Expr) and spec.head == "Failure" and len(spec.args) == 2 and is_assoc(spec.args[1]) \
            and assoc_get(spec.args[1], "ExceptionTagList") is not None:
        a = spec.args[1]
        tags = list(assoc_get(a, "ExceptionTagList").args)
        ftag = spec.args[0]
        data = [(k, v, d) for k, v, d in assoc_items(a) if k not in _RESERVED and k != "ExceptionFailureTag"]
    elif is_assoc(spec):
        tl = assoc_get(spec, "ExceptionTagList")
        mt = assoc_get(spec, "ExceptionTag")
        if tl is not None and K.is_list(tl) and tl.args and all(_valid_tag(t) for t in tl.args):
            tags = list(tl.args)
        elif mt is not None and _valid_tag(mt):
            tags = _expand([mt])
        else:
            raise _BadException("invargs", L(*args))
        ftag = assoc_get(spec, "ExceptionFailureTag")
        data = [(k, v, d) for k, v, d in assoc_items(spec)
                if k not in _RESERVED and k not in ("ExceptionFailureTag", "ExceptionUncaught")]
    elif _valid_tag(spec):
        tags, data = _expand([spec]), []
    elif K.is_list(spec) and spec.args and all(_valid_tag(t) for t in spec.args):
        tags, data = _expand(spec.args), []
    elif K.is_list(spec) or isinstance(spec, (Expr, int, float)) or spec is None:
        if K.is_list(spec) or len(args) > 2:
            raise _BadException("invargs", L(*args))
        raise _BadException("untagged", spec)
    else:
        raise _BadException("untagged", spec)
    if payload is not None:
        if is_assoc(payload) and assoc_get(payload, "ExceptionFailureTag") is not None:
            ftag = assoc_get(payload, "ExceptionFailureTag")
            payload = Expr("Association", *(r for r in payload.args if not (KL.is_rule(r) and
                                                                             r.args[0] == "ExceptionFailureTag")))
        data = _merge(data, payload, fname)
    return _make_exception(tags, data, ftag)


def _error_handling_exception(bad):
    issue("Exception", bad.tag, bad.args_)
    payload = bad.args_
    return _make_exception(["ErrorHandlingException"], [
        ("FailingFunction", Symbol("Exception"), False), ("FailingPayload", payload, False),
        ("MessageTemplate", mname("Exception", bad.tag), True), ("MessageParameters", L(payload), False)],
        "ErrorHandlingError")


def exception_data(x):
    tags, data, ftag = _exc_parts(x)
    rules = [_rule(k, v, d) for k, v, d in data]
    rules += [_rule("ExceptionAPIVersion", "1"), _rule("ExceptionTag", tags[0]), _rule("ExceptionTagList", L(*tags)),
              _rule("ExceptionFailureTag", ftag if ftag is not None else _tag_string(tags[0]))]
    return Expr("Association", *rules)


def exception_failure(x):
    tags, data, ftag = _exc_parts(x)
    rules = [_rule(k, v, d) for k, v, d in data]
    rules += [_rule("ExceptionAPIVersion", "1"), _rule("ExceptionTagList", L(*tags))]
    return Expr("Failure", ftag if ftag is not None else _tag_string(tags[0]), Expr("Association", *rules))


def _exception_property(x, prop):
    if K.is_list(prop):
        return L(*(_exception_property(x, p) for p in prop.args))
    if not isinstance(prop, str):
        return K.UNHANDLED
    tags, data, ftag = _exc_parts(x)
    if prop == "Properties":
        return L("Message", "ExceptionTag", "ExceptionTagList", "ExceptionData", "ExceptionAPIVersion",
                 "ExceptionFailure", *[k for k, _v, _d in data if isinstance(k, str)], "ExceptionFailureTag")
    if prop == "ExceptionTag":
        return tags[0]
    if prop == "ExceptionTagList":
        return L(*tags)
    if prop == "ExceptionData":
        return exception_data(x)
    if prop == "ExceptionAPIVersion":
        return "1"
    if prop == "ExceptionFailure":
        return exception_failure(x)
    if prop == "ExceptionFailureTag":
        return ftag if ftag is not None else _tag_string(tags[0])
    if prop == "Message":
        m = _object_message(x.args[1])
        return missing("NotAvailable") if m is None else m
    for k, v, d in data:
        if k == prop:
            return ev(v) if d else v
    return missing("KeyAbsent", prop)


def _exception_hook(head, args, ev_):
    if not is_exception(head) or len(args) != 1:
        return K.UNHANDLED
    return _exception_property(head, args[0])


KL.COMPOUND_HEAD_HOOKS["Exception"] = _exception_hook


@fn("Exception", area=AREA, status="added",
    notes="Exception[spec] / Exception[spec, payload] builds the validated object Exception[{tags...}, <|data..., "
          "\"ExceptionValidated\" -> True, \"ExceptionAPIVersion\" -> \"1\"|>] with tag expansion through registered "
          "parent types; spec may be a tag, a list of tags, an Exception, a Failure carrying \"ExceptionTagList\" or "
          "an association; malformed input issues Exception::untagged / Exception::invargs and stays unevaluated; "
          "exc[\"prop\"] gives the documented properties")
def _exception(*args):
    if _has_pattern(*args):
        raise Unevaluated()
    if len(args) == 2 and is_exception(Expr("Exception", *args)):
        raise Unevaluated()
    try:
        return build_exception(args)
    except _BadException as bad:
        issue("Exception", bad.tag, bad.args_)
        raise Unevaluated() from None


@fn("ExceptionQ", area=AREA, status="added",
    notes="ExceptionQ[expr] is True for a validated Exception object; ExceptionQ[expr, tag | {tags}] also requires "
          "one of the tags in its (expanded) tag list")
def _exception_q(x, *tagspec):
    if len(tagspec) > 1:
        raise ValueError("ExceptionQ called with more than 2 arguments; 1 or 2 arguments are expected")
    if not is_exception(x):
        return False
    if not tagspec:
        return True
    return _tags_match(x.args[0].args, tagspec[0], "ExceptionQ")


def _tagspec_list(spec, fname):
    if K.is_list(spec):
        items = list(spec.args)
    else:
        items = [spec]
    for t in items:
        if not _valid_tag(t):
            raise ValueError(f"{fname}: {input_text(t)} is not a valid exception type (a string or a symbol)")
    return items


def _tags_match(tags, spec, fname):
    if is_sym(spec, "All"):
        return True
    wanted = {_type_key(t) for t in _tagspec_list(spec, fname)}
    expanded = {_type_key(a) for t in tags for a in _ancestors(t)}
    return bool(wanted & expanded)


@fn("ThrowException", area=AREA, status="added",
    notes="ThrowException[args] throws Exception[args] (to CatchExceptions; Catch[..., _] also intercepts it); "
          "malformed arguments throw an \"ErrorHandlingException\" exception after an Exception:: message; uncaught "
          "exceptions reach the top level as an Uncaught Throw error")
def _throw_exception(*args):
    try:
        exc = build_exception(args, "ThrowException")
    except _BadException as bad:
        exc = _error_handling_exception(bad)
    raise ExceptionThrow(exc)


def _catch_clauses(spec):
    if KL.is_rule(spec) or (isinstance(spec, Expr) and spec.head == "RuleDelayed"):
        return [(spec.args[0], spec.args[1])]
    if K.is_list(spec) and spec.args and all(isinstance(r, Expr) and r.head in ("Rule", "RuleDelayed")
                                             for r in spec.args):
        return [(r.args[0], r.args[1]) for r in spec.args]
    return [(spec, None)]


def catch_exceptions(expr, spec):
    clauses = _catch_clauses(spec)
    for tagspec, _h in clauses:
        if not is_sym(tagspec, "All"):
            _tagspec_list(tagspec, "CatchExceptions")
    try:
        return ev(expr)
    except ExceptionThrow as t:
        exc = t.exception
        for tagspec, handler in clauses:
            if _tags_match(exc.args[0].args, tagspec, "CatchExceptions"):
                break
        else:
            raise
    if handler is None:
        return exception_failure(exc)
    return ev(K.call(handler, exception_data(exc)))


@fn("CatchExceptions", area=AREA, status="added",
    notes="CatchExceptions[expr, spec] (HoldFirst) returns expr's value, or for a thrown exception matching spec "
          "(All, tag, {tags}, tag -> handler, {tagspec -> handler, ...}; first matching clause; subtypes match "
          "their parents) handler[exceptionData] or, without a handler, the exception's Failure; other exceptions "
          "propagate. The operator form CatchExceptions[spec][expr] holds expr")
def _catch_exceptions(*args):
    if len(args) == 1:
        raise Unevaluated()
    if len(args) != 2:
        raise ValueError(f"CatchExceptions called with {len(args)} arguments; 1 or 2 arguments are expected")
    return catch_exceptions(args[0], ev(args[1]))


def _catch_exceptions_hook(head, args, ev_):
    if len(head.args) != 1 or len(args) != 1:
        return K.UNHANDLED
    return catch_exceptions(args[0], head.args[0])


KL.COMPOUND_HEAD_HOOKS["CatchExceptions"] = _catch_exceptions_hook
_hold_first("CatchExceptions")


def _hold_operator(name):
    """Operator forms whose argument is held (name[spec][expr]); needs the evaluator's operator-hold hook."""
    import importlib
    ops = getattr(importlib.import_module("mikoshilang.evaluate"), "_HOLD_ALL_OPERATORS", None)
    if ops is not None:
        ops.add(name)


_hold_operator("CatchExceptions")


@fn("RegisterExceptionType", area=AREA, status="added",
    notes="RegisterExceptionType[sym] / [sym, parent | {parents}] (HoldFirst) registers the symbol sym as an "
          "exception type in this session and returns Success[\"RegisteredExceptionType\", <|\"ExceptionType\" -> sym, "
          "\"ParentExceptionTypes\" -> {...}, \"OwnerContext\" -> \"Global`\"|>]; registering twice issues "
          "RegisterExceptionType::regd and returns $Failed")
def _register_exception_type(sym, *parents):
    if len(parents) > 1:
        raise ValueError("RegisterExceptionType called with more than 2 arguments; 1 or 2 arguments are expected")
    if not isinstance(sym, Symbol):
        raise ValueError(f"RegisterExceptionType: the exception type {input_text(sym)} must be a symbol")
    ps = []
    if parents:
        p = ev(parents[0])
        for t in (p.args if K.is_list(p) else [p]):
            if not _valid_tag(t):
                issue("RegisterExceptionType", "type", t)
                raise Unevaluated()
            ps.append(t)
    types = _types()
    if _type_key(sym) in types:
        issue("RegisterExceptionType", "regd", sym)
        return FAILED
    types[_type_key(sym)] = (sym, ps)
    return Expr("Success", "RegisteredExceptionType", assoc(("ExceptionType", sym),
                                                             ("ParentExceptionTypes", L(*ps)),
                                                             ("OwnerContext", "Global`")))


_hold_first("RegisterExceptionType")


@fn("ExceptionTypes", area=AREA, status="added",
    notes="ExceptionTypes[] / ExceptionTypes[All] lists the registered exception types in registration order; "
          "ExceptionTypes[tag | {tags}] lists the registered types that are the given types or their descendants")
def _exception_types(*spec):
    if len(spec) > 1:
        raise ValueError("ExceptionTypes called with more than 1 argument")
    regs = [v[0] for v in _types().values()]
    if not spec or is_sym(spec[0], "All"):
        return L(*regs)
    wanted = {_type_key(t) for t in _tagspec_list(spec[0], "ExceptionTypes")}
    return L(*(r for r in regs if {_type_key(a) for a in _ancestors(r)} & wanted))


@fn("ExceptionTypeRegisteredQ", area=AREA, status="added",
    notes="ExceptionTypeRegisteredQ[sym] is True if sym is a registered exception type; with a second argument, "
          "True if it is registered and one of the given types is among its ancestors")
def _exception_type_registered_q(sym, *spec):
    if len(spec) > 1:
        raise ValueError("ExceptionTypeRegisteredQ called with more than 2 arguments")
    if not isinstance(sym, Symbol) or _type_key(sym) not in _types():
        return False
    if not spec:
        return True
    wanted = {_type_key(t) for t in _tagspec_list(spec[0], "ExceptionTypeRegisteredQ")}
    return bool({_type_key(a) for a in _ancestors(sym)[1:]} & wanted)


# ─────────────────────────── aborts ───────────────────────────

class AbortSignal(BaseException):
    """Abort[]: aborts the whole computation (a BaseException so that generic error handlers do not swallow it)."""

    def __init__(self, user=True):
        super().__init__("$Aborted: the computation was aborted by Abort[]")
        self.user = user


class ConstraintAbort(AbortSignal):
    """Abort generated by TimeConstrained / MemoryConstrained / a test constraint."""

    def __init__(self, token, kind):
        BaseException.__init__(self, f"$Aborted: {kind} constraint exceeded")
        self.user = False
        self.token, self.kind = token, kind


_ABORT_FRAMES: ContextVar[tuple] = ContextVar("w02_abort_frames", default=())


class _AbortCtx:
    """An AbortProtect-like ("P") or CheckAbort ("C") frame; P frames hold the abort they postponed."""

    def __init__(self, kind, raise_on_exit=True):
        self.kind = kind
        self.pending = None
        self.raise_on_exit = raise_on_exit

    def __enter__(self):
        self.token = _ABORT_FRAMES.set(_ABORT_FRAMES.get() + (self,))
        return self

    def __exit__(self, et, ev_, tb):
        _ABORT_FRAMES.reset(self.token)
        if self.kind == "P" and self.pending is not None and et is None and self.raise_on_exit:
            _raise_or_defer(self.pending)
        return False


def _raise_or_defer(exc):
    """Raise an abort now, or postpone it to the end of the innermost AbortProtect that no CheckAbort is inside."""
    for f in reversed(_ABORT_FRAMES.get()):
        if f.kind == "C":
            break
        if f.kind == "P":
            if f.pending is None:
                f.pending = exc
            return False
    raise exc


@fn("Abort", area=AREA, status="added",
    notes="Abort[] aborts the computation: it raises AbortSignal ($Aborted), caught by CheckAbort; inside "
          "AbortProtect the abort takes effect when the AbortProtect finishes")
def _abort():
    _raise_or_defer(AbortSignal())
    return ABORTED


@fn("Interrupt", area=AREA, status="added",
    notes="Interrupt[] generates an interactive interrupt: in MikoshiLang (Python) that is KeyboardInterrupt, the "
          "same effect as pressing Ctrl-C at that point")
def _interrupt():
    raise KeyboardInterrupt("Interrupt[]")


@fn("AbortProtect", area=AREA, status="added",
    notes="AbortProtect[expr] (HoldAll) evaluates expr completely, saving any Abort[] or TimeConstrained / "
          "MemoryConstrained abort until it finishes (a CheckAbort inside still catches them)")
def _abort_protect(expr):
    with _AbortCtx("P"):
        return ev(expr)


def _option(args, names, fname):
    args = list(args)
    opts = {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") \
            and isinstance(args[-1].args[0], Symbol) and args[-1].args[0].name in names:
        r = args.pop()
        opts[r.args[0].name] = ev(r.args[1])
    return args, opts


@fn("CheckAbort", area=AREA, status="added",
    notes="CheckAbort[expr, failexpr] (HoldAll) returns failexpr if an abort occurs in expr; option PropagateAborts: "
          "Automatic (default: absorb Abort[], propagate constraint aborts after failexpr), True (propagate all), "
          "False (absorb all)")
def _check_abort(*args):
    args, opts = _option(args, {"PropagateAborts"}, "CheckAbort")
    if len(args) != 2:
        raise ValueError(f"CheckAbort called with {len(args)} arguments; 2 arguments are expected")
    prop = opts.get("PropagateAborts", AUTOMATIC)
    if not (prop is True or prop is False or is_sym(prop, "Automatic")):
        raise ValueError(f"CheckAbort: PropagateAborts -> {input_text(prop)} should be Automatic, True or False")
    try:
        with _AbortCtx("C"):
            return ev(args[0])
    except AbortSignal as a:
        result = ev(args[1])
        if prop is True or (is_sym(prop, "Automatic") and not a.user):
            raise
        if isinstance(a, ConstraintAbort):
            a_token = a.token
            a_token.absorbed = True
        return result


@fn("PropagateAborts", area=AREA, status="added",
    notes="option name for CheckAbort (Automatic, True or False); used as a function it stays unevaluated")
def _propagate_aborts(*args):
    raise Unevaluated()


@fn("WithCleanup", area=AREA, status="added",
    notes="WithCleanup[expr, cleanup] / WithCleanup[init, expr, cleanup] (HoldAll) returns expr's value and always "
          "evaluates cleanup, also after an abort, Throw, Confirm or error in expr; aborts during init and cleanup "
          "are postponed until cleanup is done (an abort in init skips expr)")
def _with_cleanup(*args):
    if len(args) == 2:
        init, body, cleanup = None, args[0], args[1]
    elif len(args) == 3:
        init, body, cleanup = args
    else:
        raise ValueError(f"WithCleanup called with {len(args)} arguments; 2 or 3 arguments are expected")
    pending = None
    if init is not None:
        with _AbortCtx("P", raise_on_exit=False) as p:
            ev(init)
        pending = p.pending
    try:
        if pending is None:
            return ev(body)
    finally:
        with _AbortCtx("P", raise_on_exit=False) as p:
            ev(cleanup)
        pending = pending or p.pending
        if pending is not None and sys.exc_info()[0] is None:
            _raise_or_defer(pending)
    return ABORTED


_hold_all("AbortProtect", "CheckAbort", "WithCleanup")


# ─────────────────────────── time and memory constraints ───────────────────────────

class _Constraint:
    def __init__(self, kind, limit):
        self.kind = kind
        self.limit = limit
        self.fired = False
        self.absorbed = False
        self.deadline = None
        self.baseline = 0
        self.peak = 0


_ACTIVE: list = []          # active constraints of the main thread (outermost first)
_TIMER = {"installed": False, "prev": None}
_TICK = 0.005


def _check_constraints():
    now = time.perf_counter()
    mem = None
    for c in list(_ACTIVE):
        if c.fired:
            continue
        if c.kind == "time" and now >= c.deadline:
            c.fired = True
            _raise_or_defer(ConstraintAbort(c, "time"))
        elif c.kind == "memory":
            if mem is None:
                import tracemalloc
                mem = tracemalloc.get_traced_memory()[0]
            c.peak = max(c.peak, mem - c.baseline)
            if mem - c.baseline > c.limit:
                c.fired = True
                _raise_or_defer(ConstraintAbort(c, "memory"))


def _on_alarm(signum, frame):
    if _ACTIVE:
        _check_constraints()


def _use_signals():
    return hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()


def _start_watch():
    if _use_signals():
        if not _TIMER["installed"]:
            _TIMER["prev"] = signal.signal(signal.SIGALRM, _on_alarm)
            _TIMER["installed"] = True
            signal.setitimer(signal.ITIMER_REAL, _TICK, _TICK)
        return "signal"
    prev = sys.gettrace()
    counter = [0]

    def tracer(frame, event, arg):
        counter[0] += 1
        if counter[0] % 64 == 0 and _ACTIVE:
            _check_constraints()
        return None

    sys.settrace(tracer)
    return ("trace", prev)


def _stop_watch(mode):
    if mode == "signal":
        if not _ACTIVE and _TIMER["installed"]:
            signal.setitimer(signal.ITIMER_REAL, 0, 0)
            signal.signal(signal.SIGALRM, _TIMER["prev"] or signal.SIG_DFL)
            _TIMER["installed"] = False
    else:
        sys.settrace(mode[1])


def constrained(thunk, kind, limit):
    """Run thunk under a time (seconds) or memory (bytes) constraint; returns (value, aborted)."""
    c = _Constraint(kind, limit)
    started_tracing = False
    if kind == "time":
        c.deadline = time.perf_counter() + limit
    else:
        import tracemalloc
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            started_tracing = True
        c.baseline = tracemalloc.get_traced_memory()[0]
    _ACTIVE.append(c)
    mode = _start_watch()
    try:
        try:
            value = thunk()
            if kind == "memory":
                import tracemalloc
                peak = tracemalloc.get_traced_memory()[1] - c.baseline if started_tracing else c.peak
                if peak > limit and not c.absorbed:
                    return None, True
            if kind == "time" and time.perf_counter() > c.deadline and not c.absorbed:
                return None, True
            return value, False
        except ConstraintAbort as a:
            if a.token is not c:
                raise
            return None, True
        except MemoryError:
            if kind != "memory":
                raise
            return None, True
    finally:
        if c in _ACTIVE:
            _ACTIVE.remove(c)
        _stop_watch(mode)
        if started_tracing:
            import tracemalloc
            tracemalloc.stop()


def _seconds(t, fname):
    if isinstance(t, Expr) and t.head == "Quantity" and len(t.args) == 2 and isinstance(t.args[1], str):
        scale = {"Seconds": 1, "Milliseconds": 1e-3, "Microseconds": 1e-6, "Minutes": 60, "Hours": 3600}.get(t.args[1])
        if scale is None:
            raise ValueError(f"{fname}: unsupported time unit {t.args[1]!r}")
        t = ev(Expr("Times", t.args[0], scale))
    if is_sym(t, "Infinity"):
        return float("inf")
    try:
        v = float(K.to_sp(t)) if not isinstance(t, bool) else None
    except Exception:  # noqa: BLE001
        v = None
    if v is None or v < 0 or v != v:
        return None
    return v


def _bytes(b, fname):
    if isinstance(b, Expr) and b.head == "Quantity" and len(b.args) == 2 and isinstance(b.args[1], str):
        scale = {"Bytes": 1, "Kilobytes": 1000, "Megabytes": 10 ** 6, "Gigabytes": 10 ** 9, "Kibibytes": 1024,
                 "Mebibytes": 1024 ** 2, "Gibibytes": 1024 ** 3}.get(b.args[1])
        if scale is None:
            raise ValueError(f"{fname}: unsupported memory unit {b.args[1]!r}")
        b = ev(Expr("Times", b.args[0], scale))
    if is_sym(b, "Infinity"):
        return float("inf")
    if isinstance(b, bool):
        return None
    try:
        v = K.to_sp(b)
    except Exception:  # noqa: BLE001
        return None
    if not (isinstance(v, sp.Basic) and v.is_number and v.is_real) or v < 0:
        return None
    return int(v)


@fn("TimeConstrained", area=AREA, status="added",
    notes="TimeConstrained[expr, t] / TimeConstrained[expr, t, failexpr] (HoldAll) evaluates expr, aborting it after "
          "t seconds of real time and returning $Aborted (or failexpr, evaluated only then); t may be Infinity or a "
          "Quantity in seconds; AbortProtect postpones the abort")
def _time_constrained(expr, t, *fail):
    if len(fail) > 1:
        raise ValueError("TimeConstrained called with more than 3 arguments; 2 or 3 arguments are expected")
    tv = ev(t)
    secs = _seconds(tv, "TimeConstrained")
    if secs is None:
        issue("TimeConstrained", "timc", tv)
        raise Unevaluated()
    if secs == float("inf"):
        return ev(expr)
    value, aborted = constrained(lambda: ev(expr), "time", secs)
    if aborted:
        return ev(fail[0]) if fail else ABORTED
    return value


@fn("MemoryConstrained", area=AREA, status="added",
    notes="MemoryConstrained[expr, b] / MemoryConstrained[expr, b, failexpr] (HoldAll) evaluates expr, aborting it "
          "when the additional Python heap memory it allocates (tracemalloc) exceeds b bytes, and returns $Aborted "
          "(or failexpr); b may be Infinity or a byte Quantity")
def _memory_constrained(expr, b, *fail):
    if len(fail) > 1:
        raise ValueError("MemoryConstrained called with more than 3 arguments; 2 or 3 arguments are expected")
    bv = ev(b)
    n = _bytes(bv, "MemoryConstrained")
    if n is None:
        issue("MemoryConstrained", "ipnfm", bv)
        raise Unevaluated()
    if n == float("inf"):
        return ev(expr)
    value, aborted = constrained(lambda: ev(expr), "memory", n)
    if aborted:
        return ev(fail[0]) if fail else ABORTED
    return value


@fn("TimeRemaining", area=AREA, status="added",
    notes="TimeRemaining[] gives the seconds (real time) left before the earliest enclosing TimeConstrained aborts, "
          "or Infinity outside TimeConstrained")
def _time_remaining():
    deadlines = [c.deadline for c in _ACTIVE if c.kind == "time" and not c.fired]
    if not deadlines:
        return INFINITY
    return max(0.0, min(deadlines) - time.perf_counter())


_hold_all("TimeConstrained", "MemoryConstrained")


# ─────────────────────────── timing ───────────────────────────

def _timed(expr, clock):
    t0 = clock()
    value = ev(expr)
    return clock() - t0, value


@fn("Timing", area=AREA, status="added",
    notes="Timing[expr] (HoldAll) gives {CPU seconds used by this process while evaluating expr, result} "
          "(time.process_time); time spent in Pause is not counted")
def _timing(expr):
    t, v = _timed(expr, time.process_time)
    return L(t, v)


@fn("AbsoluteTiming", area=AREA, status="added",
    notes="AbsoluteTiming[expr] (HoldAll) gives {real seconds elapsed while evaluating expr, result} "
          "(time.perf_counter)")
def _absolute_timing(expr):
    t, v = _timed(expr, time.perf_counter)
    return L(t, v)


def repeated_timing(expr, total=None):
    times, value = [], None
    start = time.perf_counter()
    budget = 1.0 if total is None else total
    while True:
        t0 = time.perf_counter()
        v = ev(expr)
        times.append(time.perf_counter() - t0)
        if len(times) == 1:
            value = v
        if len(times) >= 4 and time.perf_counter() - start >= budget:
            break
        if len(times) >= 10 ** 6:
            break
    times.sort()
    n = len(times)
    q = n // 4
    mid = times[q:n - q] or times
    return sum(mid) / len(mid), value


@fn("RepeatedTiming", area=AREA, status="added",
    notes="RepeatedTiming[expr] / RepeatedTiming[expr, t] (HoldFirst) evaluates expr at least 4 times and for at "
          "least t seconds (default 1) and gives {trimmed mean of the timings without the lower and upper quartiles, "
          "result of the first evaluation}")
def _repeated_timing(expr, *t):
    if len(t) > 1:
        raise ValueError("RepeatedTiming called with more than 2 arguments; 1 or 2 arguments are expected")
    total = None
    if t:
        total = _seconds(t[0], "RepeatedTiming")
        if total is None or total == float("inf"):
            issue("RepeatedTiming", "timc", t[0])
            raise Unevaluated()
    avg, v = repeated_timing(expr, total)
    return L(avg, v)


@fn("EchoTiming", area=AREA, status="added",
    notes="EchoTiming[expr] / EchoTiming[expr, label] (HoldFirst) evaluates expr, prints >> [label  ]seconds and "
          "returns the result; Method -> AbsoluteTiming (default), Timing, RepeatedTiming or a holding function "
          "returning {timing, result}")
def _echo_timing(expr, *rest):
    rest, opts = _option(rest, {"Method"}, "EchoTiming")
    if len(rest) > 1:
        raise ValueError("EchoTiming called with too many arguments; EchoTiming[expr] or EchoTiming[expr, label]")
    label = ev(rest[0]) if rest else None
    method = opts.get("Method", Symbol("AbsoluteTiming"))
    if is_sym(method, "AbsoluteTiming"):
        t, v = _timed(expr, time.perf_counter)
    elif is_sym(method, "Timing"):
        t, v = _timed(expr, time.process_time)
    elif is_sym(method, "RepeatedTiming"):
        t, v = repeated_timing(expr)
    else:
        r = ev(K.call(method, expr))
        if not (K.is_list(r) and len(r.args) == 2):
            if failure_q(r):
                return r
            raise ValueError(f"EchoTiming: Method {input_text(method)} did not return {{timing, result}}")
        t, v = r.args
    if not _QUIET_ECHO.get():
        print_line(">> " + (f"{input_text(label)}  " if label is not None else "") + input_text(t))
    return v


@fn("Pause", area=AREA, status="added",
    notes="Pause[n] waits at least n seconds (n a non-negative real number or a Quantity of time) and returns Null")
def _pause(n):
    secs = _seconds(n, "Pause")
    if secs is None or secs == float("inf"):
        issue("Pause", "numnm", Expr("Pause", n))
        raise Unevaluated()
    end = time.perf_counter() + secs
    while True:
        left = end - time.perf_counter()
        if left <= 0:
            return NULL
        time.sleep(min(left, 0.05))


_hold_all("Timing", "AbsoluteTiming")
_hold_first("RepeatedTiming", "EchoTiming")

_SESSION_START = time.time()


def _process_start():
    try:
        with open("/proc/self/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        start_ticks = int(fields[19])
        with open("/proc/uptime") as f:
            uptime = float(f.read().split()[0])
        hz = os.sysconf("SC_CLK_TCK")
        return time.time() - (uptime - start_ticks / hz)
    except (OSError, ValueError, IndexError, AttributeError):
        return _SESSION_START


_PROCESS_START = _process_start()


@fn("SessionTime", area=AREA, status="added",
    notes="SessionTime[] gives the real seconds since the MikoshiLang (Python) process started (procfs on Linux; "
          "elsewhere since the library was loaded)")
def _session_time():
    return time.time() - _PROCESS_START


@fn("TimeUsed", area=AREA, status="added",
    notes="TimeUsed[] gives the CPU seconds used so far by the MikoshiLang (Python) process (time.process_time); "
          "Pause is not counted")
def _time_used():
    return time.process_time()


# ─────────────────────────── memory ───────────────────────────

def _bytecount(e, seen=None):
    if isinstance(e, (Symbol, bool)) or e is None:
        return 0
    if isinstance(e, Expr):
        n = sys.getsizeof(e) + sys.getsizeof(e.args)
        if isinstance(e.head, Expr):
            n += _bytecount(e.head)
        return n + sum(_bytecount(a) for a in e.args)
    if isinstance(e, (list, tuple)):
        return sys.getsizeof(e) + sum(_bytecount(a) for a in e)
    return sys.getsizeof(e)


@fn("ByteCount", area=AREA, status="added",
    notes="ByteCount[expr] gives the bytes MikoshiLang uses to store expr (Python object sizes, every part counted "
          "separately, no sharing); symbols (including True, False, Null) are shared and count 0. Values differ "
          "from Wolfram's own memory layout, as the documentation allows between systems")
def _byte_count(e):
    return _bytecount(e)


def _rss_now():
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        import psutil  # type: ignore
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        raise NotImplementedError("MemoryInUse: process memory is only available on Linux or with psutil installed")


@fn("MemoryInUse", area=AREA, status="added",
    notes="MemoryInUse[] gives the resident memory (bytes) of the MikoshiLang process (/proc/self/statm on Linux, "
          "psutil elsewhere); MemoryInUse[$FrontEnd] raises because there is no front end")
def _memory_in_use(*fe):
    if fe:
        raise NotImplementedError("MemoryInUse[$FrontEnd]: MikoshiLang has no front end process")
    return _rss_now()


@fn("MaxMemoryUsed", area=AREA, status="added",
    notes="MaxMemoryUsed[] gives the peak resident memory (bytes) of the process (getrusage); MaxMemoryUsed[expr] "
          "(HoldAll) gives the peak additional Python heap bytes (tracemalloc) used while evaluating expr")
def _max_memory_used(*args):
    if len(args) > 1:
        raise ValueError("MaxMemoryUsed called with more than 1 argument")
    if not args:
        try:
            import resource
        except ImportError:
            raise NotImplementedError("MaxMemoryUsed[]: peak process memory is not available on this platform")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak) if sys.platform == "darwin" else int(peak) * 1024
    import tracemalloc
    started = not tracemalloc.is_tracing()
    if started:
        tracemalloc.start()
    try:
        base, outer_peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        ev(args[0])
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        if started:
            tracemalloc.stop()
    return max(0, peak - base)


@fn("MemoryAvailable", area=AREA, status="added",
    notes="MemoryAvailable[] gives the bytes the operating system reports as available (MemAvailable in "
          "/proc/meminfo on Linux, psutil elsewhere)")
def _memory_available():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        import psutil  # type: ignore
        return int(psutil.virtual_memory().available)
    except ImportError:
        raise NotImplementedError("MemoryAvailable: available memory is only reported on Linux or with psutil")


_hold_all("MaxMemoryUsed")


@fn("ClearSystemCache", area=AREA, status="added",
    notes="ClearSystemCache[] clears MikoshiLang's internal caches (SymPy's cache and the evaluator memo table); "
          "ClearSystemCache[\"Symbolic\"] / [\"Numeric\"] clear the symbolic (SymPy) / numeric (mpmath) caches; "
          "returns Null")
def _clear_system_cache(*kind):
    if len(kind) > 1:
        raise ValueError("ClearSystemCache called with more than 1 argument")
    if kind and kind[0] not in ("Numeric", "Symbolic"):
        issue("ClearSystemCache", "type", kind[0])
        raise Unevaluated()
    if not kind or kind[0] == "Symbolic":
        from sympy.core.cache import clear_cache
        clear_cache()
        from ..evaluate import clear_memo
        clear_memo()
    if not kind or kind[0] == "Numeric":
        try:
            import mpmath
            for name in ("_cache", "cache"):
                c = getattr(mpmath.mp, name, None)
                if isinstance(c, dict):
                    c.clear()
        except ImportError:  # pragma: no cover
            pass
    return NULL


@fn("TerminatedEvaluation", area=AREA, status="added",
    notes="TerminatedEvaluation[\"reason\"] is the inert representation of an evaluation stopped by a session limit "
          "(\"IterationLimit\", \"RecursionLimit\", ...); a non-string reason raises")
def _terminated_evaluation(reason):
    if _has_pattern(reason):
        raise Unevaluated()
    if not isinstance(reason, str):
        raise ValueError(f"TerminatedEvaluation: the reason {input_text(reason)} must be a string")
    raise Unevaluated()


# ─────────────────────────── testing framework ───────────────────────────

_TEST_OPTIONS = ("MemoryConstraint", "MetaInformation", "SameTest", "TestID", "TimeConstraint")
_OUTCOMES = ("Success", "MessagesFailure", "Failure", "NotEvaluated")
_ITESTS: ContextVar = ContextVar("w02_intermediate_tests", default=None)
_REPORT_COLLECT: ContextVar = ContextVar("w02_report_collect", default=None)
_ABS_EPOCH = 2208988800.0          # seconds from 1900-01-01 (Wolfram AbsoluteTime) to the Unix epoch


def _split_options(args, names, fname):
    args = list(args)
    opts = {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") \
            and isinstance(args[-1].args[0], Symbol):
        r = args[-1]
        name = r.args[0].name
        if name not in names:
            if name in _TEST_OPTIONS or name in ("HandlerFunctions", "HandlerFunctionsKeys",
                                                 "TestEvaluationFunction", "ProgressReporting", "Method"):
                raise ValueError(f"{fname}: {name} is not a known option for {fname}")
            break
        args.pop()
        opts.setdefault(name, ev(r.args[1]))
    return args, opts


def _base36(n):
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = digits[r] + out
    return out or "0"


def _auto_test_id(*parts):
    text = "\x00".join(input_text(p) for p in parts)
    return _base36(int(hashlib.sha256(text.encode()).hexdigest()[:16], 16))


def _test_key(creation_id):
    return int(hashlib.sha256(str(creation_id).encode()).hexdigest()[:15], 16)


def make_test(input_, expected=True, messages=None, opts=None):
    """A not-yet-evaluated TestObject (TestCreate)."""
    opts = opts or {}
    messages = L() if messages is None else messages
    tid = opts.get("TestID", AUTOMATIC)
    if is_sym(tid, "Automatic"):
        tid = _auto_test_id(input_, expected, messages)
    elif not isinstance(tid, str):
        raise ValueError(f"TestCreate: TestID -> {input_text(tid)} should be a string or Automatic")
    tfn = _own("$TestFileName", "")
    return Expr("TestObject", assoc(
        ("Outcome", "NotEvaluated"), ("FailureType", NONE),
        ("Input", Expr("HoldForm", input_)), ("ExpectedOutput", Expr("HoldForm", expected)),
        ("ExpectedMessages", Expr("HoldForm", messages)),
        ("MetaInformation", opts.get("MetaInformation", NONE)),
        ("AbsoluteTime", time.time() + _ABS_EPOCH),
        ("SameTest", opts.get("SameTest", Symbol("SameQ"))),
        ("SameMessages", Symbol("Testing`MessageMatchQ")),
        ("MemoryConstraint", opts.get("MemoryConstraint", INFINITY)),
        ("TimeConstraint", opts.get("TimeConstraint", INFINITY)),
        ("CreationID", str(uuid.uuid4())), ("TestID", tid),
        ("TestFileName", tfn if isinstance(tfn, str) else "")))


def _test_args(args, fname):
    args, opts = _split_options(args, _TEST_OPTIONS, fname)
    if not 1 <= len(args) <= 3:
        raise ValueError(f"{fname} called with {len(args)} arguments; 1 to 3 arguments (plus options) are expected")
    input_ = args[0]
    expected = args[1] if len(args) > 1 else True
    messages = args[2] if len(args) > 2 else L()
    return make_test(input_, expected, messages, opts)


def is_test(x):
    return isinstance(x, Expr) and x.head == "TestObject" and len(x.args) == 1 and is_assoc(x.args[0])


def _held(v):
    if isinstance(v, Expr) and v.head in ("HoldForm", "Hold", "HoldComplete") and len(v.args) == 1:
        return v.args[0]
    return v


def _expected_keys(msgs):
    msgs = _held(msgs)
    items = msgs.args if K.is_list(msgs) else (msgs,)
    keys = []
    for m in items:
        if isinstance(m, Expr) and m.head in ("Message", "HoldForm") and m.args:
            m = m.args[0]
            if isinstance(m, Expr) and m.head == "Message" and m.args:
                m = m.args[0]
        k = _key_of(m, "TestCreate", "expected message")
        if k is None:
            raise ValueError(f"TestCreate: {input_text(m)} is not a message name s::t")
        keys.append(k)
    return keys


def _messages_match(expected, actual):
    if len(expected) != len(actual):
        return False
    for e, a in zip(expected, actual):
        if e != a and not (e[0] == "General" and e[1] == a[1]):
            return False
    return True


def _constraint_value(v, conv, fname):
    x = conv(v, fname)
    return x


def _run_constrained(thunk, tc, mc):
    """Evaluate thunk under the test's constraints; returns (value, failure type or None)."""
    def with_memory():
        if mc == float("inf"):
            return thunk(), None
        v, ab = constrained(thunk, "memory", mc)
        return (ABORTED, "MemoryConstraintFailure") if ab else (v, None)
    if tc == float("inf"):
        return with_memory()
    r, ab = constrained(with_memory, "time", tc)
    return (ABORTED, "TimeConstraintFailure") if ab else r


def evaluate_test(obj, defaults=None):
    """TestEvaluate: returns (evaluated TestObject, actual value)."""
    a = obj.args[0]
    defaults = defaults or {}
    items = [(k, v, d) for k, v, d in assoc_items(a)
             if k not in ("Outcome", "FailureType", "ActualOutput", "ActualMessages", "AbsoluteTimeUsed",
                          "CPUTimeUsed", "MemoryUsed", "IntermediateTestsTree", "EvaluationID")]
    get = dict((k, v) for k, v, _d in items if isinstance(k, str))
    input_ = _held(get.get("Input", True))
    expected_h = _held(get.get("ExpectedOutput", True))
    same = get.get("SameTest", Symbol("SameQ"))
    if is_sym(same, "SameQ") and "SameTest" in defaults:
        same = defaults["SameTest"]
    tcv = get.get("TimeConstraint", INFINITY)
    if is_sym(tcv, "Infinity") and "TimeConstraint" in defaults:
        tcv = defaults["TimeConstraint"]
    mcv = get.get("MemoryConstraint", INFINITY)
    if is_sym(mcv, "Infinity") and "MemoryConstraint" in defaults:
        mcv = defaults["MemoryConstraint"]
    tc = _seconds(tcv, "TestEvaluate")
    mc = _bytes(mcv, "TestEvaluate")
    expected_keys = _expected_keys(get.get("ExpectedMessages", L()))

    collect = _Collect()
    itests = []
    value, failure, outcome = None, None, None
    t0, c0 = time.perf_counter(), time.process_time()
    if tc is None or tc == 0:
        failure = "TimeConstraintInvalid"
    elif mc is None or mc == 0:
        failure = "MemoryConstraintInvalid"
    tok = _ITESTS.set(itests)
    try:
        with _Frame(collect):
            if failure is None:
                try:
                    value, failure = _run_constrained(lambda: ev(input_), tc, mc)
                except ConstraintAbort:
                    raise
                except AbortSignal:
                    value, failure = ABORTED, "EvaluationAbortedFailure"
                except K.ThrowSignal as t:
                    value, failure = Expr("Throw", t.value, *([] if t.tag is None else [t.tag])), \
                        "UncaughtThrowFailure"
                except Exception as exc:  # noqa: BLE001
                    if not is_error(exc):
                        raise
                    issue_error(exc)
                    value = input_
            expected = None
            if failure is None:
                try:
                    expected = ev(expected_h)
                except Exception as exc:  # noqa: BLE001 - an expected output that raises stays unevaluated
                    if not is_error(exc):
                        raise
                    expected = expected_h
    finally:
        _ITESTS.reset(tok)
    dt, dc = time.perf_counter() - t0, time.process_time() - c0
    if failure is None:
        if is_sym(same, "SameQ"):
            r = K.same(value, expected)
        else:
            try:
                r = ev(K.call(same, value, expected))
            except Exception as exc:  # noqa: BLE001
                if not is_error(exc):
                    raise
                r = None
        if r is False:
            failure = "SameTestFailure"
        elif r is not True:
            failure = "SameTestUnevaluated"
    actual_keys = [k for k, _c in collect.messages]
    if failure is not None:
        outcome = "Failure"
    elif not _messages_match(expected_keys, actual_keys):
        outcome, failure = "MessagesFailure", "SameMessagesFailure"
    else:
        bad = [t for t in itests if assoc_get(t.args[0], "Outcome") != "Success"]
        if bad:
            outcome = "Failure"
            failure = "IntermediateTestMessagesFailure" if assoc_get(bad[0].args[0], "Outcome") == \
                "MessagesFailure" else "IntermediateTestFailure"
        else:
            outcome = "Success"
    shown = value if value is not None else ABORTED
    new = [Expr("RuleDelayed" if d else "Rule", k, v) for k, v, d in items]
    new += [Expr("Rule", k, v) for k, v in (
        ("ActualOutput", Expr("HoldForm", shown)),
        ("ActualMessages", L(*(c for _k, c in collect.messages))),
        ("AbsoluteTimeUsed", dt), ("CPUTimeUsed", dc), ("MemoryUsed", _bytecount(shown)),
        ("IntermediateTestsTree", L(*itests)), ("Outcome", outcome), ("FailureType", failure or NONE))]
    return Expr("TestObject", Expr("Association", *new)), shown


_TEST_PROPERTIES = ("TestID", "Outcome", "FailureType", "Input", "ExpectedOutput", "ActualOutput", "ActualMessages",
                    "ExpectedMessages", "SameTest", "TestFileName", "TimeConstraint", "AbsoluteTimeUsed",
                    "CPUTimeUsed", "MemoryConstraint", "MemoryUsed", "AbsoluteTime", "Created", "CreationID",
                    "MetaInformation", "TestCreate", "IntermediateTests")
_EVALUATED_ONLY = ("ActualOutput", "ActualMessages", "AbsoluteTimeUsed", "CPUTimeUsed", "MemoryUsed",
                   "IntermediateTests")


def _original_form(a, head):
    g = lambda k, d=None: assoc_get(a, k, d)  # noqa: E731
    return Expr("HoldForm", Expr(head, _held(g("Input")), _held(g("ExpectedOutput")), _held(g("ExpectedMessages", L())),
                                 Expr("Rule", Symbol("TestID"), g("TestID")),
                                 Expr("Rule", Symbol("SameTest"), g("SameTest", Symbol("SameQ"))),
                                 Expr("Rule", Symbol("TimeConstraint"), g("TimeConstraint", INFINITY)),
                                 Expr("Rule", Symbol("MemoryConstraint"), g("MemoryConstraint", INFINITY)),
                                 Expr("Rule", Symbol("MetaInformation"), g("MetaInformation", NONE))))


def _created(abs_time):
    t = abs_time - _ABS_EPOCH
    lt = time.localtime(t)
    secs = lt.tm_sec + (t - int(t))
    return Expr("DateObject", L(lt.tm_year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, secs), "Instant",
                "Gregorian", lt.tm_gmtoff / 3600.0)


def test_property(obj, prop):
    a = obj.args[0]
    if K.is_list(prop):
        return assoc(*((p, test_property(obj, p)) for p in prop.args))
    if is_sym(prop, "All"):
        return assoc(*((p, test_property(obj, p)) for p in _TEST_PROPERTIES))
    if not isinstance(prop, str):
        return K.UNHANDLED
    evaluated = assoc_get(a, "Outcome") != "NotEvaluated"
    if prop == "Properties":
        return L(*_TEST_PROPERTIES)
    if prop in _EVALUATED_ONLY and not evaluated:
        return missing("NotEvaluated")
    if prop == "IntermediateTests":
        return assoc_get(a, "IntermediateTestsTree", L())
    if prop == "Created":
        return _created(float(K.to_sp(assoc_get(a, "AbsoluteTime"))))
    if prop in ("TestCreate", "VerificationTest"):
        return _original_form(a, prop)
    if prop == "TestKey":
        return _test_key(assoc_get(a, "CreationID"))
    v = assoc_get(a, prop)
    if v is None and assoc_get_raw(a, prop) is None:
        return missing("KeyAbsent", prop)
    return v


def _test_object_hook(head, args, ev_):
    if not is_test(head) or len(args) != 1:
        return K.UNHANDLED
    return test_property(head, args[0])


KL.COMPOUND_HEAD_HOOKS["TestObject"] = _test_object_hook


@fn("TestObject", area=AREA, status="added",
    notes="TestObject[<|...|>] is the inert test result (from TestCreate / TestEvaluate / VerificationTest); "
          "test[\"prop\"] for \"Outcome\" (Success, Failure, MessagesFailure, NotEvaluated), \"FailureType\", "
          "\"Input\", \"ExpectedOutput\", \"ActualOutput\", \"ActualMessages\", \"TestID\", \"TestKey\", "
          "\"TestCreate\", \"Created\", timings, ...; test[{p1, ...}] gives an association; \"Properties\"")
def _test_object(*args):
    if _has_pattern(*args):
        raise Unevaluated()
    if len(args) != 1 or not is_assoc(args[0]):
        raise ValueError("TestObject: the argument must be an association of test data")
    raise Unevaluated()


@fn("TestCreate", area=AREA, status="added",
    notes="TestCreate[input] / [input, expected] / [input, expected, messages] (HoldAllComplete) creates an "
          "unevaluated TestObject; options MemoryConstraint, MetaInformation, SameTest, TestID (Automatic = hash "
          "of the test), TimeConstraint")
def _test_create(*args):
    return _test_args(args, "TestCreate")


def _test_from_assoc(a):
    def g(key, default):
        raw = assoc_get_raw(a, key)
        if raw is None:
            return default
        return raw.args[1] if raw.head == "RuleDelayed" else _held(raw.args[1])
    opts = {}
    for k in ("SameTest", "TimeConstraint", "MemoryConstraint", "MetaInformation"):
        raw = assoc_get_raw(a, k)
        if raw is not None:
            opts[k] = ev(raw.args[1])
    tid = assoc_get(a, "TestID")
    if tid is not None:
        opts["TestID"] = tid if isinstance(tid, str) else AUTOMATIC
    return make_test(g("Input", True), g("ExpectedOutput", True), g("ExpectedMessages", L()), opts)


@fn("TestEvaluate", area=AREA, status="added",
    notes="TestEvaluate[test] runs a TestObject (messages are recorded, not printed) and gives the evaluated "
          "TestObject with \"ActualOutput\", \"ActualMessages\", timings and the \"Outcome\" / \"FailureType\"; "
          "TestEvaluate[<|\"Input\" :> ..., \"ExpectedOutput\" -> ..., ...|>] builds the test from an association")
def _test_evaluate(x):
    if is_assoc(x):
        x = _test_from_assoc(x)
    if not is_test(x):
        raise ValueError(f"TestEvaluate: {input_text(x)} is not a TestObject or an association of test data")
    return evaluate_test(x)[0]


@fn("VerificationTest", area=AREA, status="added",
    notes="VerificationTest[input, expected, messages, opts] (HoldAllComplete) = TestEvaluate[TestCreate[...]]: "
          "evaluates the test immediately without printing messages and gives the evaluated TestObject")
def _verification_test(*args):
    obj = evaluate_test(_test_args(args, "VerificationTest"))[0]
    coll = _REPORT_COLLECT.get()
    if coll is not None:
        coll.append(obj)
    return obj


@fn("IntermediateTest", area=AREA, status="added",
    notes="IntermediateTest[input, expected, messages, opts] (HoldAllComplete) evaluates a test (without printing "
          "messages) and returns the evaluated input; inside VerificationTest / TestEvaluate a failing intermediate "
          "test makes the enclosing test fail (IntermediateTestFailure / IntermediateTestMessagesFailure)")
def _intermediate_test(*args):
    obj, value = evaluate_test(_test_args(args, "IntermediateTest"))
    parent = _ITESTS.get()
    if parent is not None:
        parent.append(obj)
    return value


_hold_all("TestCreate", "VerificationTest", "IntermediateTest")


# ── reports ──

def _report(title, tests, failures, aborted=False, times=None):
    res = []
    keys = {"NotEvaluated": [], "Success": [], "Failure": [], "MessagesFailure": []}
    for t in tests:
        k = _test_key(assoc_get(t.args[0], "CreationID"))
        res.append(Expr("Rule", k, t))
        keys.setdefault(assoc_get(t.args[0], "Outcome"), []).append(k)
    data = [("Title", title), ("Aborted", aborted), ("TestResults", Expr("Association", *res)),
            ("FailureResults", assoc()), ("TestsNotEvaluatedKeys", L(*keys["NotEvaluated"])),
            ("TestsSucceededKeys", L(*keys["Success"])),
            ("TestsFailedWrongResultsKeys", L(*keys["Failure"])),
            ("TestsFailedWithMessagesKeys", L(*keys["MessagesFailure"])),
            ("RuntimeFailures", L(*failures))]
    if times is not None:
        data += [("CPUTimeUsed", times[0]), ("AbsoluteTimeUsed", times[1])]
    return Expr("TestReportObject", assoc(*data))


def is_report(x):
    return isinstance(x, Expr) and x.head == "TestReportObject" and len(x.args) == 1 and is_assoc(x.args[0])


def _report_tests(r):
    tr = assoc_get(r.args[0], "TestResults", assoc())
    return [v for _k, v, _d in assoc_items(tr)]


_REPORT_PROPERTIES = ("Title", "Results", "ResultsDataset", "ResultsByOutcome", "ResultsByTestFileName",
                      "ResultsByTestID", "RuntimeFailures", "ReportSucceeded", "MemoryUsed", "CPUTimeUsed",
                      "AbsoluteTimeUsed")


def _group(tests, prop, keys=()):
    groups = {k: [] for k in keys}
    order = list(keys)
    for t in tests:
        k = assoc_get(t.args[0], prop)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(t)
    return assoc(*((k, L(*groups[k])) for k in order))


def _dataset_row(t):
    row = []
    for p in _TEST_PROPERTIES:
        v = test_property(t, p)
        if p in ("AbsoluteTimeUsed", "CPUTimeUsed") and not (isinstance(v, Expr) and v.head == "Missing"):
            v = Expr("Quantity", v, "Seconds")
        elif p == "MemoryUsed" and not (isinstance(v, Expr) and v.head == "Missing"):
            v = Expr("Quantity", v, "Bytes")
        row.append((p, v))
    return assoc(*row)


def report_property(r, prop):
    a = r.args[0]
    if K.is_list(prop):
        return assoc(*((p, report_property(r, p)) for p in prop.args))
    if not isinstance(prop, str):
        return K.UNHANDLED
    tests = _report_tests(r)
    if prop == "Properties":
        return L(*_REPORT_PROPERTIES)
    if prop == "Results":
        return L(*tests)
    if prop == "ResultsDataset":
        return Expr("Dataset", L(*(_dataset_row(t) for t in tests)))
    if prop == "ResultsByOutcome":
        return _group(tests, "Outcome", _OUTCOMES)
    if prop == "ResultsByTestFileName":
        return _group(tests, "TestFileName")
    if prop == "ResultsByTestID":
        return _group(tests, "TestID")
    if prop == "ReportSucceeded":
        return all(assoc_get(t.args[0], "Outcome") == "Success" for t in tests) and \
            not assoc_get(a, "RuntimeFailures", L()).args
    if prop == "MemoryUsed":
        return Expr("Quantity", sum(int(assoc_get(t.args[0], "MemoryUsed", 0)) for t in tests), "Bytes")
    if prop in ("CPUTimeUsed", "AbsoluteTimeUsed"):
        v = assoc_get(a, prop)
        if v is None:
            v = sum(float(K.to_sp(assoc_get(t.args[0], prop, 0.0))) for t in tests)
        return Expr("Quantity", v, "Seconds")
    if prop == "TestsSucceededCount":
        return len(assoc_get(a, "TestsSucceededKeys", L()).args)
    if prop == "TestsFailedCount":
        return len(assoc_get(a, "TestsFailedWrongResultsKeys", L()).args) + \
            len(assoc_get(a, "TestsFailedWithMessagesKeys", L()).args)
    v = assoc_get(a, prop)
    if v is None and assoc_get_raw(a, prop) is None:
        return missing("KeyAbsent", prop)
    return v


def _report_hook(head, args, ev_):
    if not is_report(head) or len(args) != 1:
        return K.UNHANDLED
    return report_property(head, args[0])


KL.COMPOUND_HEAD_HOOKS["TestReportObject"] = _report_hook


@fn("TestReportObject", area=AREA, status="added",
    notes="TestReportObject[<|...|>] is the inert result of TestReport; report[\"prop\"] for \"Results\", "
          "\"ResultsByOutcome\", \"ResultsByTestID\", \"ResultsByTestFileName\", \"ResultsDataset\", "
          "\"ReportSucceeded\", \"RuntimeFailures\", \"Title\", timings, \"TestsSucceededCount\", the key lists ...")
def _test_report_object(*args):
    if _has_pattern(*args):
        raise Unevaluated()
    if len(args) != 1 or not is_assoc(args[0]):
        raise ValueError("TestReportObject: the argument must be an association of report data")
    raise Unevaluated()


_EVENTS = ("FileStarted", "FileCompleted", "ReportStarted", "ReportCompleted", "RuntimeFailure", "TestCreated",
           "TestEvaluated")


class _Handlers:
    def __init__(self, spec, keys, title):
        self.spec = spec
        self.keys = keys
        self.title = title
        if not (is_assoc(spec) or (K.is_list(spec) and all(KL.is_rule(r) for r in spec.args))):
            raise ValueError(f"TestReport: HandlerFunctions -> {input_text(spec)} should be an association of "
                             f"event -> function")
        self.map = {}
        for r in spec.args:
            k = r.args[0]
            if not isinstance(k, str) or k not in _EVENTS + ("UnhandledEvent",):
                raise ValueError(f"TestReport: {input_text(k)} is not a TestReport event")
            self.map[k] = r.args[1]
        if not (is_sym(keys, "Automatic") or (K.is_list(keys) and all(isinstance(k, str) for k in keys.args))):
            raise ValueError(f"TestReport: HandlerFunctionsKeys -> {input_text(keys)} should be Automatic or a "
                             f"list of key names")

    def fire(self, event, **data):
        f = self.map.get(event, self.map.get("UnhandledEvent"))
        if f is None:
            return
        full = [("EventName", event), ("EventID", str(uuid.uuid4()))]
        if "test" in data:
            t = data["test"]
            ft = assoc_get(t.args[0], "FailureType")
            out = assoc_get(t.args[0], "Outcome")
            full += [("TestObject", t), ("Outcome", out), ("FailureType", "NotEvaluated" if out == "NotEvaluated"
                                                                    else ft)]
        if "failure" in data:
            full.append(("Failure", data["failure"]))
        if "file" in data:
            full.append(("TestFileName", data["file"]))
        if is_sym(self.keys, "Automatic"):
            chosen = full
        else:
            have = dict(full)
            extra = {"Title": self.title, "TestFileName": data.get("file", ""), "Failure": missing("NotAvailable"),
                     "FailureType": missing("NotAvailable"), "Outcome": missing("NotAvailable"),
                     "TestObject": missing("NotAvailable")}
            chosen = [(k, have[k] if k in have else extra.get(k, missing("KeyAbsent", k))) for k in self.keys.args]
        ev(K.call(f, assoc(*chosen)))


def _run_tests(items, handlers, evalf, defaults):
    out = []
    for t in items:
        if assoc_get(t.args[0], "Outcome") == "NotEvaluated":
            handlers.fire("TestCreated", test=t)
            if is_sym(evalf, "TestEvaluate"):
                t = evaluate_test(t, defaults)[0]
            else:
                t = ev(K.call(evalf, t))
                if not is_test(t):
                    raise ValueError(f"TestReport: TestEvaluationFunction {input_text(evalf)} did not return a "
                                     f"TestObject")
            handlers.fire("TestEvaluated", test=t)
        out.append(t)
    return out


def _report_file(path, handlers, evalf, defaults):
    from ..parser import parse_file
    if not isinstance(path, str):
        raise ValueError("TestReport: the file name must be a string")
    full = os.path.abspath(os.path.expanduser(path))
    if full.endswith(".nb"):
        raise NotImplementedError("TestReport: testing notebooks (.nb) are not supported; use a .wl or .wlt file")
    if not os.path.isfile(full):
        issue("General", "noopen", path)
        return None
    handlers.fire("FileStarted", file=full)
    statements = parse_file(full)
    collected, failures, tests = [], [], []
    own = K.state().own
    saved = own.get("$TestFileName", None)
    own["$TestFileName"] = full
    tok = _REPORT_COLLECT.set(collected)
    try:
        for st in statements:
            try:
                r = ev(st)
            except Exception as exc:  # noqa: BLE001
                if not is_error(exc) and not isinstance(exc, K.ThrowSignal):
                    raise
                f = Expr("Failure", "TestFileFailure", assoc(("MessageTemplate", str(exc)), ("TestFileName", full)))
                failures.append(f)
                handlers.fire("RuntimeFailure", failure=f, file=full)
                continue
            if is_test(r) and assoc_get(r.args[0], "Outcome") == "NotEvaluated":
                collected.extend(_run_tests([r], handlers, evalf, defaults))
            for t in collected:
                if not any(t is u for u in tests):
                    tests.append(t)
            collected.clear()
    finally:
        _REPORT_COLLECT.reset(tok)
        if saved is None:
            own.pop("$TestFileName", None)
        else:
            own["$TestFileName"] = saved
    handlers.fire("FileCompleted", file=full)
    return tests, failures, "Test Report: " + os.path.basename(full)


@fn("TestReport", area=AREA, status="added",
    notes="TestReport[{tests}] / TestReport[test] evaluates the NotEvaluated TestObjects and gives a "
          "TestReportObject; TestReport[{reports}] merges reports; TestReport[\"file.wl\"] evaluates a plain-text "
          "test file with $TestFileName set (notebooks raise). Options HandlerFunctions (events ReportStarted, "
          "TestCreated, TestEvaluated, RuntimeFailure, FileStarted, FileCompleted, ReportCompleted, UnhandledEvent), "
          "HandlerFunctionsKeys, TestEvaluationFunction, SameTest, TimeConstraint, MemoryConstraint")
def _test_report(*args):
    names = {"HandlerFunctions", "HandlerFunctionsKeys", "TestEvaluationFunction", "SameTest", "TimeConstraint",
             "MemoryConstraint", "ProgressReporting"}
    args, opts = _split_options(args, names, "TestReport")
    if len(args) != 1:
        raise ValueError(f"TestReport called with {len(args)} arguments; 1 argument (plus options) is expected")
    x = args[0]
    defaults = {k: opts[k] for k in ("SameTest", "TimeConstraint", "MemoryConstraint") if k in opts}
    evalf = opts.get("TestEvaluationFunction", Symbol("TestEvaluate"))
    title = AUTOMATIC
    handlers = _Handlers(opts.get("HandlerFunctions", assoc()), opts.get("HandlerFunctionsKeys", AUTOMATIC), title)
    t0, c0 = time.perf_counter(), time.process_time()
    handlers.fire("ReportStarted")
    if isinstance(x, Expr) and x.head == "File" and len(x.args) == 1:
        x = x.args[0]
    failures = []
    if isinstance(x, str):
        r = _report_file(x, handlers, evalf, defaults)
        if r is None:
            raise Unevaluated()
        tests, failures, title = r
        handlers.title = title
    else:
        items = x.args if K.is_list(x) else (x,)
        if items and all(is_report(i) for i in items):
            tests = [t for rep in items for t in _report_tests(rep)]
            failures = [f for rep in items for f in assoc_get(rep.args[0], "RuntimeFailures", L()).args]
        else:
            bad = [i for i in items if not is_test(i)]
            if bad:
                raise ValueError(f"TestReport: {input_text(bad[0])} is not a TestObject, TestReportObject or file")
            tests = _run_tests(list(items), handlers, evalf, defaults)
    rep = _report(title, tests, failures, times=(time.process_time() - c0, time.perf_counter() - t0))
    handlers.fire("ReportCompleted")
    return rep


def _inert_option(name, notes):
    @fn(name, area=AREA, status="added", notes=notes)
    def _opt(*args):
        raise Unevaluated()
    return _opt


for _n, _note in (
        ("SameTest", "option of TestCreate / VerificationTest / IntermediateTest / TestReport giving the function "
                     "that compares the actual and expected outputs (default SameQ); inert as an expression"),
        ("TestID", "option of TestCreate / VerificationTest / IntermediateTest: the test's identifier (Automatic = "
                   "a hash of the test); inert as an expression"),
        ("TimeConstraint", "option of the test functions: seconds a test may run (default Infinity; exceeding it "
                           "gives TimeConstraintFailure); inert as an expression"),
        ("MemoryConstraint", "option of the test functions: bytes a test may allocate (default Infinity; exceeding "
                             "it gives MemoryConstraintFailure); inert as an expression"),
        ("HandlerFunctions", "option of TestReport: <|event -> f, ...|> handlers for the report's events; inert as "
                             "an expression"),
        ("HandlerFunctionsKeys", "option of TestReport: the keys of the association passed to the handler "
                                 "functions (Automatic or a list); inert as an expression"),
        ("TestEvaluationFunction", "option of TestReport: the function used to evaluate each created test "
                                   "(default TestEvaluate); inert as an expression")):
    _inert_option(_n, _note)
