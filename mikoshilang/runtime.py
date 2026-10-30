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
              "InterpretQuery", "PackSearch", "PackValue", "CacheClear", "CacheStats"}


class _Values:
    def __contains__(self, key): return key in state().own
    def __getitem__(self, key): return state().own[key]


values = _Values()


class Session:
    """Independent definitions for one notebook/user; not an execution sandbox."""

    def __init__(self, allow_network=False):
        self.state = SessionState()
        self.allow_network = allow_network

    def evaluate(self, source):
        from .parser import parse
        from .evaluate import evaluate
        network_token = _network.set(self.allow_network)
        token = _state.set(self.state)
        try:
            return evaluate(parse(source) if isinstance(source, str) else source)
        finally:
            _state.reset(token)
            _network.reset(network_token)

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
        from .knowledge_full_rules import register_full
        from .rules import replace
        return replace(Expr(h, *(ev(x) for x in a)), register_full())
    return UNHANDLED
