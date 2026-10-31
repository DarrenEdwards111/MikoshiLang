"""Explicit, isolated evaluation sessions and gated knowledge access.

No Python eval/exec is used. Unknown symbolic heads remain symbolic;
quarantined legacy operations fail rather than manufacture evidence.
Language semantics (definitions, patterns, scoping, control flow) live in
:mod:`mikoshilang.kernel`.
"""
from contextvars import ContextVar
from pathlib import Path
import json
from .expr import Expr, Symbol
from .kernel import UNHANDLED, SessionState, state, _state, ReturnSignal, BreakSignal, ContinueSignal, ThrowSignal

_network = ContextVar("mikoshilang_network", default=False)
_KNOWLEDGE = {"EntitySearch", "EntityValue", "EntityRelationships", "WikipediaText", "KnowledgeAsOf",
              "InterpretQuery", "PackSearch", "PackValue", "CacheClear", "CacheStats",
              # legacy knowledge-layer heads (mikoshilang.knowledge); previously never wired into the
              # evaluator, so they silently stayed unevaluated
              "EntityProperties", "WikipediaSummary", "Disambiguate", "Entity"}
_LEGACY_KNOWLEDGE = {"EntityProperties", "WikipediaSummary", "Disambiguate", "Entity"}


class _Values:
    def __contains__(self, key): return key in state().own
    def __getitem__(self, key): return state().own[key]


values = _Values()


class Session:
    """Independent definitions for one notebook/user; not an execution sandbox."""

    def __init__(self, allow_network=False, allow_python=False):
        """allow_python: False (default), True (any module) or a set of module names PythonFunction may call."""
        self.state = SessionState()
        self.allow_network = allow_network
        self.allow_python = frozenset(allow_python) if isinstance(allow_python, (set, frozenset, list, tuple)) \
            else bool(allow_python)

    def evaluate(self, source):
        from .parser import parse
        from .evaluate import evaluate
        from .interop import _python
        network_token = _network.set(self.allow_network)
        python_token = _python.set(self.allow_python)
        token = _state.set(self.state)
        try:
            return evaluate(parse(source) if isinstance(source, str) else source)
        finally:
            _state.reset(token)
            _python.reset(python_token)
            _network.reset(network_token)

    # ── Python interop (prototype; see PYTHON_INTEROP.md) ──

    def eval(self, source, out="python", symbolic="expr", arrays="list"):
        """Evaluate source (a string, an Expr or a Python value) and convert the result with to_python.

        out="python" (default): data as Python values, symbolic results as Expr (symbolic="sympy" converts them);
        out="expr": the raw MikoshiLang result.
        """
        from .interop import from_python, to_python
        expr = source if isinstance(source, str) else from_python(source)
        result = self.evaluate(expr)
        return result if out == "expr" else to_python(result, symbolic=symbolic, arrays=arrays)

    @property
    def ns(self):
        """Pythonic calls: ml = session.ns; ml.Integrate(x**2, x)."""
        from .interop import Namespace
        return Namespace(self)

    def function(self, fn=None, *, name=None, listable=False, notes="", check="delegated", override=False):
        """Decorator: register a typed Python function as a head in this session (types from its hints)."""
        from .interop import register
        def deco(f):
            f.mikoshi = register(self.state, f, name=name, listable=listable, notes=notes, check=check,
                                 override=override)
            return f
        return deco(fn) if fn is not None else deco

    def evaluate_input(self, source, whole=True):
        """Evaluate input as the Wolfram main loop: In[n]/Out[n]/InString[n] history, $Line, $Pre, $Post, $PreRead.

        Returns a list of (line number, value) pairs; one line for the whole source (whole=True) or one per
        top-level statement.  Use output_form(value) for the printed form ($PrePrint).
        """
        from .library.wl_evaluation import main_loop
        network_token = _network.set(self.allow_network)
        token = _state.set(self.state)
        try:
            return main_loop(source, whole)
        finally:
            _state.reset(token)
            _network.reset(network_token)

    def output_form(self, value):
        """The value as printed by the main loop ($PrePrint applied)."""
        from .library.wl_evaluation import preprint
        token = _state.set(self.state)
        try:
            return preprint(value)
        finally:
            _state.reset(token)

    def end(self):
        """End the session: evaluates $Epilog."""
        from .library.wl_evaluation import epilog
        token = _state.set(self.state)
        try:
            epilog()
        finally:
            _state.reset(token)

    def clear(self):
        self.state.clear()


def clear_definitions():
    state().clear()


_quarantined = frozenset(json.loads(Path(__file__).with_name('quarantined.json').read_text()))


def reject_quarantined(head):
    if isinstance(head, str) and head in _quarantined:
        raise NotImplementedError(f'{head} belongs to the quarantined, unreviewed library')


def apply_definition(expr):
    """Backward-compatible wrapper: apply the session's DownValues to expr."""
    from .kernel import apply_downvalues
    from .evaluate import evaluate
    return apply_downvalues(expr, lambda x: evaluate(x, 1))


def special(expr, ev):
    """Network-gated knowledge heads (arguments are evaluated only once access is allowed)."""
    h, a = expr.head, expr.args
    if h in _KNOWLEDGE:
        if not _network.get():
            raise PermissionError("Knowledge access requires Session(allow_network=True)")
        from .rules import replace
        if h in _LEGACY_KNOWLEDGE:
            from .knowledge_rules import register
            rules = [r for r in register() if r.pattern.head == h]
        else:
            from .knowledge_full_rules import register_full
            rules = register_full()
        return replace(Expr(h, *(ev(x) for x in a)), rules)
    return UNHANDLED
