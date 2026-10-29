"""Held language forms and explicit, isolated evaluation sessions.

No Python eval/exec is used. Unknown symbolic heads remain symbolic;
quarantined legacy operations fail rather than manufacture evidence.
"""
from contextvars import ContextVar
from pathlib import Path
import json
from .expr import Expr, Symbol
from .rules import Rule, _substitute
from .pattern import match

UNHANDLED = object()
_state = ContextVar('mikoshilang_session', default=None)
_default = ({}, [])
_network = ContextVar("mikoshilang_network", default=False)
_KNOWLEDGE = {"EntitySearch", "EntityValue", "EntityRelationships", "WikipediaText", "KnowledgeAsOf", "InterpretQuery", "PackSearch", "PackValue", "CacheClear", "CacheStats"}

def state():
    return _state.get() or _default

class _Values:
    def __contains__(self, key): return key in state()[0]
    def __getitem__(self, key): return state()[0][key]
values = _Values()

class Session:
    """Independent definitions for one notebook/user; not an execution sandbox."""
    def __init__(self, allow_network=False):
        self.state = ({}, [])
        self.allow_network = allow_network
    def evaluate(self, source):
        from .parser import parse
        from .evaluate import evaluate
        network_token = _network.set(self.allow_network)
        token = _state.set(self.state)
        try: return evaluate(parse(source) if isinstance(source, str) else source)
        finally:
            _state.reset(token)
            _network.reset(network_token)
    def clear(self):
        self.state[0].clear()
        self.state[1].clear()

def clear_definitions():
    state()[0].clear()
    state()[1].clear()

_quarantined = frozenset(json.loads(Path(__file__).with_name('quarantined.json').read_text()))
def reject_quarantined(head):
    if isinstance(head, str) and head in _quarantined:
        raise NotImplementedError(f'{head} belongs to the quarantined, unreviewed library')

def apply_definition(expr):
    for lhs, rhs in reversed(state()[1]):
        bindings = match(lhs, expr)
        if bindings is not None: return _substitute(rhs, bindings)
    return UNHANDLED

def replace_once(expr, rules):
    for rule in rules:
        result = rule.apply(expr)
        if result is not None: return result
    if isinstance(expr, Expr):
        return Expr(expr.head, *(replace_once(a, rules) for a in expr.args))
    return expr

def _slots(expr, args):
    if isinstance(expr, Expr):
        if expr.head == 'Function': return expr  # nested lexical scope
        if expr.head == 'Slot':
            index = expr.args[0]
            if index > len(args): raise ValueError('Missing pure-function argument')
            return args[index - 1]
        return Expr(expr.head, *(_slots(a, args) for a in expr.args))
    return expr

def special(expr, ev):
    h, a = expr.head, expr.args
    if h in _KNOWLEDGE:
        if not _network.get():
            raise PermissionError("Knowledge access requires Session(allow_network=True)")
        from .knowledge_full_rules import register_full
        from .rules import replace
        return replace(Expr(h, *(ev(x) for x in a)), register_full())
    if isinstance(h, Expr) and h.head == 'Function':
        if len(h.args) != 1: raise ValueError('Function expects one slot-based body')
        return ev(_slots(h.args[0], tuple(ev(x) for x in a)))
    if h in ('Function', 'Slot', 'Pattern', 'Blank', 'BlankSequence', 'BlankNullSequence', 'Rule', 'RuleDelayed'):
        return expr
    if h == 'CompoundExpression':
        result = None
        for item in a: result = ev(item)
        return result
    if h in ('Set', 'SetDelayed'):
        if len(a) != 2: raise ValueError('Assignment expects two arguments')
        lhs, rhs = a
        rhs = ev(rhs) if h == 'Set' else rhs
        if isinstance(lhs, Symbol): state()[0][lhs.name] = rhs
        elif isinstance(lhs, Expr) and isinstance(lhs.head, str):
            state()[1][:] = [(p,r) for p,r in state()[1] if p != lhs]
            state()[1].append((lhs, rhs))
        else: raise ValueError('Invalid assignment target')
        return rhs if h == 'Set' else None
    if h in ('And', 'Or', 'Not'):
        if h == 'Not':
            if len(a)!=1: raise ValueError('Not expects one argument')
            value = ev(a[0])
            return not value if isinstance(value,bool) else Expr('Not',value)
        unresolved=[]
        for arg in a:
            value=ev(arg)
            if h=='And' and value is False: return False
            if h=='Or' and value is True: return True
            if not isinstance(value,bool): unresolved.append(value)
        if not unresolved: return h=='And'
        return unresolved[0] if len(unresolved)==1 else Expr(h,*unresolved)
    if h == 'If':
        if len(a) != 3: raise ValueError('If expects condition, true branch, false branch')
        condition = ev(a[0])
        if condition is True: return ev(a[1])
        if condition is False: return ev(a[2])
        return Expr('If', condition, *a[1:])
    if h == 'ReplaceAll':
        if len(a) != 2: raise ValueError('ReplaceAll expects expression and rule(s)')
        spec = a[1].args if isinstance(a[1], Expr) and a[1].head == 'List' else (a[1],)
        if any(not isinstance(r, Expr) or r.head not in ('Rule', 'RuleDelayed') or len(r.args) != 2 for r in spec):
            raise ValueError('Expected Rule or list of Rules')
        return ev(replace_once(ev(a[0]), [Rule(r.args[0], r.args[1] if r.head == "RuleDelayed" else ev(r.args[1])) for r in spec]))
    if h in ('Map', 'Apply'):
        if len(a) != 2: raise ValueError(f'{h} expects two arguments')
        fn, arg = a[0], ev(a[1])
        if not isinstance(arg, Expr): raise ValueError(f'{h} requires a composite expression')
        head = fn.name if isinstance(fn, Symbol) else fn
        if not isinstance(head, (str, Expr)): raise ValueError('Invalid function')
        if h == 'Apply': return ev(Expr(head, *arg.args))
        return Expr(arg.head, *(ev(Expr(head, x)) for x in arg.args))
    if h in ('Plus', 'Times', 'Power'):
        from .math_ops import to_sympy, from_sympy
        args = tuple(ev(x) for x in a)
        if h == 'Power' and len(args) != 2: raise ValueError('Power expects two arguments')
        try:
            return from_sympy(to_sympy(Expr(h, *args)))
        except (ValueError, TypeError):
            # Unknown symbolic heads are legitimate algebraic operands.
            return Expr(h, *args)
    if h == 'N':
        from .numerical import numerical
        if len(a) not in (1,2): raise ValueError('N expects one or two arguments')
        return numerical(ev(a[0]), ev(a[1]) if len(a)==2 else 15)
    if h in ('Range', 'Table'):
        def integer(x):
            x = ev(x)
            if isinstance(x, bool) or not isinstance(x, int):
                raise ValueError('Iteration bounds must be integers')
            return x
        if h == 'Range':
            if not 1 <= len(a) <= 3: raise ValueError('Range expects 1 to 3 arguments')
            bounds = (1, integer(a[0]), 1) if len(a)==1 else (integer(a[0]), integer(a[1]), integer(a[2]) if len(a)==3 else 1)
        else:
            if len(a)!=2 or not isinstance(a[1],Expr) or a[1].head!='List' or len(a[1].args) not in (3,4):
                raise ValueError('Table expects expression and {variable, start, stop[, step]}')
            var, *limits = a[1].args
            if not isinstance(var,Symbol): raise ValueError('Table variable must be a symbol')
            bounds = (integer(limits[0]), integer(limits[1]), integer(limits[2]) if len(limits)==3 else 1)
        lo, hi, step = bounds
        if step == 0: raise ValueError('Iteration step cannot be zero')
        iterator = range(lo, hi + (1 if step > 0 else -1), step)
        if len(iterator) > 100000: raise ValueError('Iteration exceeds 100000 element limit')
        if h == 'Range': return Expr('List', *iterator)
        old = state()[0].pop(var.name, UNHANDLED)
        try:
            return Expr('List', *(ev(_substitute(a[0], {var.name: value})) for value in iterator))
        finally:
            if old is not UNHANDLED: state()[0][var.name] = old
    if h in ('D', 'Diff', 'PartialD') and len(a)==2:
        from .math_ops import diff
        if isinstance(a[1],Expr) and a[1].head=='List':
            if len(a[1].args)!=2: raise ValueError('Derivative expects {variable, order}')
            var, order = a[1].args
            if not isinstance(order,int) or order<0: raise ValueError('Invalid derivative order')
            return diff(a[0],var,order)
        return diff(a[0],a[1])
    if h in ('Dot', 'DotProduct'):
        import sympy as sp
        from .math_ops import to_sympy, from_sympy
        if len(a)!=2: raise ValueError('Dot expects two arguments')
        left,right = [to_sympy(ev(x)) for x in a]
        if not isinstance(left,list) or not isinstance(right,list): raise ValueError('Dot expects lists')
        lm = bool(left) and isinstance(left[0],list)
        rm = bool(right) and isinstance(right[0],list)
        if not lm and not rm:
            if len(left)!=len(right): raise ValueError('Vector dimensions differ')
            return from_sympy(sum(x*y for x,y in zip(left,right)))
        result = (sp.Matrix(left) if lm else sp.Matrix([left])) * sp.Matrix(right)
        if lm and rm: return Expr('List', *(Expr('List', *(from_sympy(x) for x in row)) for row in result.tolist()))
        return Expr('List', *(from_sympy(x) for x in result))
    if h == 'Solve':
        from .math_ops import solve
        if len(a) != 2: raise ValueError('Solve expects expression and variable')
        return solve(a[0], a[1])
    if h == 'Distance2D':
        from .math_ops import to_sympy, from_sympy
        import sympy as sp
        points = [ev(x) for x in a]
        if len(points)!=2 or any(not isinstance(x,Expr) or x.head!='List' or len(x.args)!=2 for x in points):
            raise ValueError('Distance2D expects two two-coordinate lists')
        return from_sympy(sp.sqrt(sum((to_sympy(x)-to_sympy(y))**2 for x,y in zip(points[0].args,points[1].args))))
    if h in ('Equal','Unequal','Less','Greater','LessEqual','GreaterEqual'):
        from .math_ops import to_sympy
        import sympy as sp
        if len(a)!=2: raise ValueError('Comparison expects two arguments')
        args = tuple(ev(x) for x in a)
        result = {'Equal':sp.Eq,'Unequal':sp.Ne,'Less':sp.Lt,'Greater':sp.Gt,'LessEqual':sp.Le,'GreaterEqual':sp.Ge}[h](*(to_sympy(x) for x in args))
        if result is sp.true: return True
        if result is sp.false: return False
        return Expr(h,*args)
    return UNHANDLED
