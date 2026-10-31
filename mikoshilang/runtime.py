"""Explicit, isolated evaluation sessions and gated knowledge access.

No Python eval/exec is used. Unknown symbolic heads remain symbolic;
quarantined legacy operations fail rather than manufacture evidence.
Language semantics (definitions, patterns, scoping, control flow) live in
:mod:`mikoshilang.kernel`.
"""
from contextlib import contextmanager
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
    """Independent definitions for one notebook/user; not an execution sandbox.

    allow_network: knowledge heads (EntityValue, ...) may reach the network.
    allow_python:  what ``PythonFunction["module.name"]`` may call from MikoshiLang code: ``False`` (default,
                   nothing), a module allow-list (``{"math", "numpy.linalg"}`` or a single module name) or ``True``
                   (any importable module). Functions registered with ``@session.function`` are always callable,
                   because the host program chose to expose them. An allow-list is a convenience, not a security
                   boundary: run untrusted input in an external process sandbox.
    """

    def __init__(self, allow_network=False, allow_python=False):
        from .interop import python_permission
        self.state = SessionState()
        self.allow_network = allow_network
        self.allow_python = python_permission(allow_python)

    @contextmanager
    def _active(self):
        """Make this session current: its definitions, network flag and Python permission."""
        from .interop import _python
        network_token = _network.set(self.allow_network)
        python_token = _python.set(self.allow_python)
        token = _state.set(self.state)
        try:
            yield
        finally:
            _state.reset(token)
            _python.reset(python_token)
            _network.reset(network_token)

    def evaluate(self, source):
        """Evaluate source code (a str) or an expression and return the raw MikoshiLang result."""
        from .parser import parse
        from .evaluate import evaluate
        with self._active():
            return evaluate(parse(source) if isinstance(source, str) else source)

    # ── Python interop (docs/python_interop.md) ──

    def eval(self, source, out="python", symbolic="expr", arrays="list"):
        """Evaluate and return the result as Python data, keeping symbolic results as expressions.

        source:   MikoshiLang code (a str), an expression, or any Python value from the conversion table
                  (converted with ``interop.from_python``; ``Fraction(1, 3)``, NumPy arrays, dicts, ...).
        out:      "python" (default) converts the result with ``interop.to_python``; "expr" returns it raw.
        symbolic: for results that are not data: "expr" (default, keep the Expr), "sympy" (convert to SymPy)
                  or "error" (raise TypeError).
        arrays:   "list" (default, nested lists) or "numpy" (a rectangular list of machine numbers becomes an
                  ndarray).

        Integers -> int, exact rationals -> fractions.Fraction, machine reals -> float, bignum reals ->
        sympy.Float (precision kept), strings -> str, Null -> None, lists -> list, associations -> dict.
        Errors raise with their Python class (TypeError, ValueError, NotImplementedError, ArithmeticError,
        PermissionError) and a "Name: " message.
        """
        from .interop import from_python, to_python, check_modes
        check_modes(out, symbolic, arrays)
        expr = source if isinstance(source, str) else from_python(source)
        result = self.evaluate(expr)
        return result if out == "expr" else to_python(result, symbolic=symbolic, arrays=arrays)

    @property
    def ns(self):
        """Pythonic calls: ``ml = session.ns; x = ml.symbols("x"); ml.Integrate(x**2, x)``."""
        from .interop import Namespace
        return Namespace(self)

    def function(self, fn=None, *, name=None, listable=False, notes="", check="delegated", override=False):
        """Decorator: register a typed Python function as a head in this session only.

        The head name defaults to the CamelCase form of the function name (``collatz_steps`` -> ``CollatzSteps``).
        Type hints become the signature (``int`` -> Integer, ``float`` -> Real, ``list[int]`` -> ListOf[Integer],
        ...): symbolic arguments leave the call unevaluated, arguments of the wrong kind raise TypeError. Built-in
        and library heads are not shadowed unless ``override=True``. ``check`` is the CONVENTIONS ``Check:``
        method recorded in the notes (``exact``, ``self-check``, ``numeric cross-check``, ``reference`` or
        ``delegated``).
        """
        from .interop import register
        def deco(f):
            f.mikoshi = register(self.state, f, name=name, listable=listable, notes=notes, check=check,
                                 override=override)
            return f
        return deco(fn) if fn is not None else deco

    def functions(self):
        """The Python functions registered in this session: {head: PythonFunctionInfo}."""
        from .interop import registered
        return registered(self.state)

    def evaluate_input(self, source, whole=True):
        """Evaluate input as the Wolfram main loop: In[n]/Out[n]/InString[n] history, $Line, $Pre, $Post, $PreRead.

        Returns a list of (line number, value) pairs; one line for the whole source (whole=True) or one per
        top-level statement.  Use output_form(value) for the printed form ($PrePrint).
        """
        from .library.wl_evaluation import main_loop
        with self._active():
            return main_loop(source, whole)

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
            raise PermissionError(f"{h}: knowledge access requires Session(allow_network=True)")
        from .rules import replace
        if h in _LEGACY_KNOWLEDGE:
            from .knowledge_rules import register
            rules = [r for r in register() if r.pattern.head == h]
        else:
            from .knowledge_full_rules import register_full
            rules = register_full()
        return replace(Expr(h, *(ev(x) for x in a)), rules)
    return UNHANDLED
