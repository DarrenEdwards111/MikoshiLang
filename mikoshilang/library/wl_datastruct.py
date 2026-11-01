"""Wolfram coverage batch W03 -- mutable data structures (CreateDataStructure / DataStructure).

A data structure is the expression DataStructure["Type", DataStructureHandle[n]]; its contents live in a Python
object for the session, so operations ds["Op", args...] mutate it in place exactly like Wolfram's (ds["Copy"] gives
an independent structure).  Operations follow the reference pages of each type; operations whose result would be
implementation-specific in Wolfram (hash order, the representative chosen by a disjoint-set "Find", visualizations)
are either documented below or raise a clear error.  SameQ on two DataStructure expressions compares identity.
"""
from __future__ import annotations

import collections
import functools
import itertools

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import fn, record, Unevaluated
from .wl_common import L, ev, apply, is_sym, is_list, is_int, is_assoc, is_rule, int_value

AREA = "data structures (W03)"
NULL = None   # the kernel represents the value Null as Python None
_STORE = {}
_COUNTER = itertools.count(1)


class DSError(ValueError):
    pass


def _missing(*a):
    return Expr("Missing", *a)


def _items(x, fname):
    if is_list(x):
        return list(x.args)
    raise DSError(f"{fname}: {x} is not a list of elements")


class _Base:
    type = None
    ops = {}

    def copy(self):
        import copy
        return copy.deepcopy(self)

    def normal(self):
        return self.op_Elements()

    def length(self):
        return self.op_Length()


def _fold(elems, args, fname):
    if len(args) == 1:
        if not elems:
            raise DSError(f"{fname}: Fold of an empty data structure needs an initial value")
        acc, rest = elems[0], elems[1:]
    elif len(args) == 2:
        acc, rest = args[1], elems
    else:
        raise DSError(f"{fname}: \"Fold\" takes a function and an optional initial value")
    for x in rest:
        acc = apply(args[0], acc, x)
    return acc


class _Seq(_Base):
    """Common list-backed behaviour."""

    def __init__(self, elems=()):
        self.d = collections.deque(elems)

    def op_Elements(self):
        return L(*self.d)

    def op_Length(self):
        return len(self.d)

    def op_EmptyQ(self):
        return len(self.d) == 0

    def op_DropAll(self):
        self.d.clear()
        return NULL

    def op_Fold(self, *args):
        return _fold(list(self.d), args, self.type)

    def _nonempty(self, op):
        if not self.d:
            raise DSError(f"{self.type}: \"{op}\" on an empty data structure")


class Stack(_Seq):
    type = "Stack"

    def op_Push(self, x):
        self.d.append(x)
        return NULL

    def op_PushList(self, xs):
        self.d.extend(_items(xs, "Stack"))
        return NULL

    def op_Pop(self):
        self._nonempty("Pop")
        return self.d.pop()

    def op_Peek(self):
        self._nonempty("Peek")
        return self.d[-1]


class Queue(_Seq):
    type = "Queue"

    def op_Push(self, x):
        self.d.append(x)
        return NULL

    def op_PushList(self, xs):
        self.d.extend(_items(xs, "Queue"))
        return NULL

    def op_Pop(self):
        self._nonempty("Pop")
        return self.d.popleft()

    def op_Peek(self):
        self._nonempty("Peek")
        return self.d[0]


class Deque(_Seq):
    type = "Deque"

    def op_PushBack(self, x):
        self.d.append(x)
        return NULL

    def op_PushFront(self, x):
        self.d.appendleft(x)
        return NULL

    def op_PushBackList(self, xs):
        self.d.extend(_items(xs, "Deque"))
        return NULL

    def op_PushFrontList(self, xs):
        for x in _items(xs, "Deque"):
            self.d.appendleft(x)
        return NULL

    def op_PopBack(self):
        self._nonempty("PopBack")
        return self.d.pop()

    def op_PopFront(self):
        self._nonempty("PopFront")
        return self.d.popleft()

    def op_PeekBack(self):
        self._nonempty("PeekBack")
        return self.d[-1]

    def op_PeekFront(self):
        self._nonempty("PeekFront")
        return self.d[0]


class RingBuffer(Deque):
    type = "RingBuffer"

    def __init__(self, cap):
        super().__init__()
        self.cap = cap

    def op_Capacity(self):
        return self.cap

    def op_PushBack(self, x):
        if self.cap == 0:
            return NULL
        if len(self.d) >= self.cap:
            self.d.popleft()
        self.d.append(x)
        return NULL

    def op_PushFront(self, x):
        if self.cap == 0:
            return NULL
        if len(self.d) >= self.cap:
            self.d.pop()
        self.d.appendleft(x)
        return NULL

    op_PushBackList = None
    op_PushFrontList = None


def _index(n, i, fname, extra=0):
    i = int_value(i, fname, "part", None)
    m = n + extra
    if i == 0 or abs(i) > m:
        raise DSError(f"{fname}: part {i} does not exist (length {n})")
    return i - 1 if i > 0 else m + i


class _Indexed(_Seq):
    def op_Part(self, i):
        return self.d[_index(len(self.d), i, self.type)]

    def op_SetPart(self, i, x):
        self.d[_index(len(self.d), i, self.type)] = x
        return x

    def op_SwapPart(self, i, j):
        a, b = _index(len(self.d), i, self.type), _index(len(self.d), j, self.type)
        self.d[a], self.d[b] = self.d[b], self.d[a]
        return NULL


class DynamicArray(_Indexed):
    type = "DynamicArray"

    def op_Append(self, x):
        self.d.append(x)
        return NULL

    def op_JoinBack(self, xs):
        self.d.extend(_items(xs, self.type))
        return NULL

    def op_Drop(self, i):
        j = _index(len(self.d), i, self.type)
        del self.d[j]
        return NULL

    def op_DropLast(self):
        self._nonempty("DropLast")
        self.d.pop()
        return NULL

    def op_Insert(self, x, i):
        j = _index(len(self.d), i, self.type, extra=1)
        self.d.insert(j, x)
        return NULL


class FixedArray(_Indexed):
    type = "FixedArray"
    op_DropAll = None


class LinkedList(_Indexed):
    type = "LinkedList"

    def op_Append(self, x):
        self.d.append(x)
        return NULL

    def op_Prepend(self, x):
        self.d.appendleft(x)
        return NULL

    def op_JoinBack(self, xs):
        self.d.extend(_items(xs, self.type))
        return NULL

    def op_JoinFront(self, xs):
        for x in reversed(_items(xs, self.type)):
            self.d.appendleft(x)
        return NULL

    def op_DropFirst(self):
        self._nonempty("DropFirst")
        self.d.popleft()
        return NULL


class DoublyLinkedList(LinkedList):
    type = "DoublyLinkedList"

    def op_DropLast(self):
        self._nonempty("DropLast")
        self.d.pop()
        return NULL


def _order_cmp(p):
    """Python comparator from a Wolfram ordering function (Sort semantics: 1/True means a before b)."""
    def cmp(a, b):
        r = apply(p, a, b) if p is not None else K.order_cmp(a, b) if hasattr(K, "order_cmp") else None
        if r is None:
            r = apply(Symbol("Order"), a, b)
        if r is True or r == 1:
            return -1
        if r is False or r == -1:
            return 1
        if r == 0:
            return 0
        raise DSError(f"the ordering function gave {r} (expected True, False, 1, 0 or -1)")
    return cmp


class PriorityQueue(_Base):
    type = "PriorityQueue"

    def __init__(self, elems=(), p=None):
        self.p = p
        self.items = list(elems)

    def _sorted(self):
        return sorted(self.items, key=functools.cmp_to_key(_order_cmp(self.p)))

    def op_Push(self, x):
        self.items.append(x)
        return NULL

    def op_Pop(self):
        if not self.items:
            raise DSError("PriorityQueue: \"Pop\" on an empty data structure")
        top = self._sorted()[-1]
        for i in range(len(self.items) - 1, -1, -1):
            if self.items[i] is top:
                del self.items[i]
                break
        return top

    def op_Peek(self):
        if not self.items:
            raise DSError("PriorityQueue: \"Peek\" on an empty data structure")
        return self._sorted()[-1]

    def op_Length(self):
        return len(self.items)

    def op_EmptyQ(self):
        return not self.items

    def op_DropAll(self):
        self.items = []
        return NULL

    def op_Elements(self):
        # Wolfram lists the heap; this implementation lists the elements from highest to lowest priority
        return L(*reversed(self._sorted()))


class _SetBase(_Base):
    def __init__(self, elems=()):
        self.s = collections.OrderedDict()
        for x in elems:
            self.s.setdefault(K.tag(x), x)

    def op_Elements(self):
        return L(*self.s.values())

    def op_Length(self):
        return len(self.s)

    def op_EmptyQ(self):
        return not self.s

    def op_Insert(self, x):
        t = K.tag(x)
        if t in self.s:
            return False
        self.s[t] = x
        return True

    def op_Delete(self, x):
        return self.s.pop(K.tag(x), None) is not None

    def op_DeleteAll(self):
        self.s.clear()
        return NULL

    def op_MemberQ(self, x):
        return K.tag(x) in self.s

    def op_Pop(self):
        if not self.s:
            raise DSError(f"{self.type}: \"Pop\" on an empty data structure")
        return self.s.popitem(last=True)[1]

    def op_Union(self, xs):
        for x in _items(xs, self.type):
            self.s.setdefault(K.tag(x), x)
        return NULL

    def op_Intersection(self, xs):
        keep = {K.tag(x) for x in _items(xs, self.type)}
        for t in [t for t in self.s if t not in keep]:
            del self.s[t]
        return NULL

    def op_Complement(self, xs):
        for x in _items(xs, self.type):
            self.s.pop(K.tag(x), None)
        return NULL


class HashSet(_SetBase):
    type = "HashSet"

    def op_SubsetQ(self, other):
        o = _get(other, "HashSet")
        if not isinstance(o, _SetBase):
            raise DSError("HashSet: \"SubsetQ\" expects another HashSet")
        return all(t in self.s for t in o.s)


class OrderedHashSet(_SetBase):
    type = "OrderedHashSet"


class _TableBase(_Base):
    def __init__(self, rules=()):
        self.t = collections.OrderedDict()
        for k, v in rules:
            self.t[K.tag(k)] = (k, v)

    def op_Insert(self, r):
        if not is_rule(r):
            raise DSError(f"{self.type}: \"Insert\" expects key -> value")
        self.t[K.tag(r.args[0])] = (r.args[0], r.args[1])
        return True

    def op_KeyDrop(self, k):
        self.t.pop(K.tag(k), None)
        return NULL

    def op_KeyDropAll(self):
        self.t.clear()
        return NULL

    def op_KeyExistsQ(self, k):
        return K.tag(k) in self.t

    def op_Keys(self):
        return L(*(k for k, _ in self.t.values()))

    def op_Values(self):
        return L(*(v for _, v in self.t.values()))

    def op_Length(self):
        return len(self.t)

    def op_EmptyQ(self):
        return not self.t

    def op_Lookup(self, k, default=None):
        hit = self.t.get(K.tag(k))
        if hit is not None:
            return hit[1]
        if default is not None:
            return apply(default, k)
        return _missing("KeyAbsent", k)

    def op_Elements(self):
        return L(*(Expr("Rule", k, v) for k, v in self.t.values()))

    def normal(self):
        return Expr("Association", *(Expr("Rule", k, v) for k, v in self.t.values()))


class HashTable(_TableBase):
    type = "HashTable"


class OrderedHashTable(_TableBase):
    type = "OrderedHashTable"


class LeastRecentlyUsedCache(_TableBase):
    type = "LeastRecentlyUsedCache"

    def __init__(self, cap, rules=(), evict=None):
        super().__init__()
        self.cap = cap
        self.evict = evict
        for k, v in rules:
            self.op_Insert(Expr("Rule", k, v))

    def op_Capacity(self):
        return self.cap

    def op_Insert(self, r):
        if not is_rule(r):
            raise DSError("LeastRecentlyUsedCache: \"Insert\" expects key -> value")
        t = K.tag(r.args[0])
        self.t.pop(t, None)
        self.t[t] = (r.args[0], r.args[1])
        evicted = False
        while len(self.t) > self.cap:
            _, (k, v) = self.t.popitem(last=False)
            evicted = True
            if self.evict is not None:
                apply(self.evict, k, v)
        return evicted

    def op_Lookup(self, k, default=None):
        t = K.tag(k)
        if t in self.t:
            self.t.move_to_end(t)
            return self.t[t][1]
        if default is not None:
            return apply(default, k)
        return _missing("KeyAbsent", k)

    def op_Peek(self, *k):
        if k:
            hit = self.t.get(K.tag(k[0]))
            return hit[1] if hit is not None else _missing("KeyAbsent", k[0])
        if not self.t:
            raise DSError("LeastRecentlyUsedCache: \"Peek\" on an empty cache")
        return next(reversed(self.t.values()))[1]

    def op_RemoveOldest(self):
        if not self.t:
            raise DSError("LeastRecentlyUsedCache: \"RemoveOldest\" on an empty cache")
        self.t.popitem(last=False)
        return NULL

    op_Elements = None


class Value(_Base):
    type = "Value"

    def __init__(self, x):
        self.v = x

    def op_Get(self):
        return self.v

    def op_Set(self, x):
        self.v = x
        return x

    def normal(self):
        return self.v

    def length(self):
        raise DSError("Value: Length is not defined")


class Counter(Value):
    type = "Counter"

    def __init__(self, x):
        super().__init__(int_value(x, "Counter", "initial value", None))

    def _int(self, n):
        return int_value(n, "Counter", "increment", None)

    def op_Set(self, x):
        self.v = self._int(x)
        return self.v

    def op_Increment(self):
        old, self.v = self.v, self.v + 1
        return old

    def op_Decrement(self):
        old, self.v = self.v, self.v - 1
        return old

    def op_PreIncrement(self):
        self.v += 1
        return self.v

    def op_PreDecrement(self):
        self.v -= 1
        return self.v

    def op_AddTo(self, n):
        self.v += self._int(n)
        return self.v

    def op_SubtractFrom(self, n):
        self.v -= self._int(n)
        return self.v


class SortedMultiset(_Base):
    type = "SortedMultiset"

    def __init__(self, elems=(), p=None):
        self.p = p
        self.items = []
        for x in elems:
            self.op_Insert(x)

    def _key(self):
        return functools.cmp_to_key(_order_cmp(self.p))

    def op_Insert(self, x):
        self.items.append(x)
        self.items.sort(key=self._key())
        return NULL

    def op_Elements(self):
        return L(*self.items)

    def op_Length(self):
        return len(self.items)

    def op_EmptyQ(self):
        return not self.items

    def op_DeleteAll(self):
        self.items = []
        return NULL

    def op_MemberQ(self, x):
        return any(K.same(x, y) for y in self.items)

    def op_Delete(self, x):
        for i, y in enumerate(self.items):
            if K.same(x, y):
                del self.items[i]
                return True
        return False

    def op_DeleteCases(self, x):
        n = len(self.items)
        self.items = [y for y in self.items if not K.same(x, y)]
        return len(self.items) != n

    def op_Cases(self, patt):
        return L(*(y for y in self.items if K.first_match(patt, y, ev) is not None))

    def op_Count(self, patt):
        return sum(1 for y in self.items if K.first_match(patt, y, ev) is not None)


class DisjointSet(_Base):
    type = "DisjointSet"

    def __init__(self):
        self.order = collections.OrderedDict()   # tag -> element
        self.parent = {}

    def _find(self, t):
        while self.parent[t] != t:
            self.parent[t] = self.parent[self.parent[t]]
            t = self.parent[t]
        return t

    def _tag(self, x):
        t = K.tag(x)
        if t not in self.order:
            raise DSError(f"DisjointSet: {x} is not an element")
        return t

    def op_Insert(self, x):
        t = K.tag(x)
        if t not in self.order:
            self.order[t] = x
            self.parent[t] = t
        return NULL

    def op_InsertAll(self, xs):
        for x in _items(xs, "DisjointSet"):
            self.op_Insert(x)
        return NULL

    def op_MemberQ(self, x):
        return K.tag(x) in self.order

    def op_Length(self):
        return len(self.order)

    def op_EmptyQ(self):
        return not self.order

    def op_Elements(self):
        return L(*self.order.values())

    def op_Unify(self, x, y):
        a, b = self._find(self._tag(x)), self._find(self._tag(y))
        if a != b:
            self.parent[b] = a
        return NULL

    def op_UnifyAll(self, xs):
        items = _items(xs, "DisjointSet")
        for y in items[1:]:
            self.op_Unify(items[0], y)
        return NULL

    def op_CommonSubsetQ(self, x, y):
        return self._find(self._tag(x)) == self._find(self._tag(y))

    def op_Subsets(self):
        groups = collections.OrderedDict()
        for t, x in self.order.items():
            groups.setdefault(self._find(t), []).append(x)
        return L(*(L(*g) for g in groups.values()))

    def op_Delete(self, x):
        t = K.tag(x)
        if t not in self.order:
            return False
        members = [u for u in self.order if self._find(u) == self._find(t)]
        del self.order[t]
        rest = [u for u in members if u != t]
        for u in members:
            self.parent.pop(u, None)
        for u in rest:
            self.parent[u] = rest[0]
        return True

    def op_Merge(self, other):
        o = _get(other, "DisjointSet")
        if not isinstance(o, DisjointSet):
            raise DSError("DisjointSet: \"Merge\" expects another DisjointSet")
        groups = collections.OrderedDict()
        for t, x in o.order.items():
            groups.setdefault(o._find(t), []).append(x)
        for g in groups.values():
            for x in g:
                self.op_Insert(x)
            for x in g[1:]:
                self.op_Unify(g[0], x)
        return NULL

    def normal(self):
        return self.op_Subsets()


class BitVector(_Base):
    type = "BitVector"

    def __init__(self, bits):
        self.bits = list(bits)

    def _k(self, k):
        k = int_value(k, "BitVector", "bit index", 0)
        if k >= len(self.bits):
            raise DSError(f"BitVector: bit {k} is out of range (length {len(self.bits)})")
        return k

    def op_BitGet(self, k):
        return int(self.bits[self._k(k)])

    def op_BitTest(self, k):
        return bool(self.bits[self._k(k)])

    def op_BitSet(self, k):
        self.bits[self._k(k)] = True
        return NULL

    def op_BitClear(self, k):
        self.bits[self._k(k)] = False
        return NULL

    def op_BitInvert(self, k):
        k = self._k(k)
        self.bits[k] = not self.bits[k]
        return NULL

    def op_BitClearAll(self):
        self.bits = [False] * len(self.bits)
        return NULL

    def op_BitCount(self):
        return sum(1 for b in self.bits if b)

    def op_BitList(self):
        return L(*(i for i, b in enumerate(self.bits) if b))

    def op_OffBitList(self, *other):
        if other:
            raise NotImplementedError("BitVector: \"OffBitList\" with a second vector is not supported")
        return L(*(i for i, b in enumerate(self.bits) if not b))

    def op_Length(self):
        return len(self.bits)

    def op_Capacity(self):
        return len(self.bits)

    def normal(self):
        return L(*self.bits)


TYPES = {c.type: c for c in (Stack, Queue, Deque, RingBuffer, DynamicArray, FixedArray, LinkedList,
                             DoublyLinkedList, PriorityQueue, HashSet, OrderedHashSet, HashTable, OrderedHashTable,
                             LeastRecentlyUsedCache, Value, Counter, SortedMultiset, DisjointSet, BitVector)}


def _create(typ, args):
    fname = "CreateDataStructure"
    n = len(args)

    def nargs(lo, hi):
        if not lo <= n <= hi:
            raise DSError(f"{fname}: \"{typ}\" takes {lo} to {hi} arguments after the type")
    if typ in ("Stack", "Queue", "Deque", "DynamicArray", "LinkedList", "DoublyLinkedList", "HashSet",
               "OrderedHashSet"):
        nargs(0, 1)
        return TYPES[typ](_items(args[0], fname) if n else ())
    if typ == "RingBuffer":
        nargs(1, 1)
        return RingBuffer(int_value(args[0], fname, "capacity", 0))
    if typ == "FixedArray":
        nargs(1, 2)
        if n == 2:
            return FixedArray([args[0]] * int_value(args[1], fname, "length", 0))
        if is_list(args[0]):
            return FixedArray(args[0].args)
        m = int_value(args[0], fname, "length", 0)
        if m:
            raise NotImplementedError(f"{fname}: give an initial value: CreateDataStructure[\"FixedArray\", x, n]")
        return FixedArray()
    if typ in ("PriorityQueue", "SortedMultiset"):
        nargs(0, 2)
        elems = [] if n == 0 or is_sym(args[0], "Null") else _items(args[0], fname)
        return TYPES[typ](elems, args[1] if n == 2 else None)
    if typ in ("HashTable", "OrderedHashTable"):
        nargs(0, 1)
        return TYPES[typ](_rules(args[0]) if n else ())
    if typ == "LeastRecentlyUsedCache":
        nargs(1, 2)
        if is_int(args[0]):
            return LeastRecentlyUsedCache(int_value(args[0], fname, "capacity", 1), (),
                                          args[1] if n == 2 else None)
        rules = _rules(args[0])
        return LeastRecentlyUsedCache(max(len(rules), 1), rules, args[1] if n == 2 else None)
    if typ in ("Value", "Counter"):
        nargs(1, 1)
        return TYPES[typ](args[0])
    if typ == "DisjointSet":
        nargs(0, 1)
        ds = DisjointSet()
        if n:
            for grp in _items(args[0], fname):
                if is_list(grp):
                    ds.op_InsertAll(grp)
                    ds.op_UnifyAll(grp) if grp.args else None
                else:
                    ds.op_Insert(grp)
        return ds
    if typ == "BitVector":
        nargs(1, 1)
        if is_list(args[0]):
            if not all(isinstance(b, bool) for b in args[0].args):
                raise DSError(f"{fname}: a BitVector list must contain True and False")
            return BitVector(args[0].args)
        return BitVector([False] * int_value(args[0], fname, "length", 0))
    raise NotImplementedError(f"{fname}: the data structure type {typ!r} is not supported "
                              f"(supported: {', '.join(sorted(TYPES))})")


def _rules(x):
    if is_assoc(x) or (is_list(x) and all(is_rule(r) for r in x.args)):
        return [(r.args[0], r.args[1]) for r in x.args]
    raise DSError(f"CreateDataStructure: {x} is not an association or a list of rules")


def _wrap(obj):
    h = next(_COUNTER)
    _STORE[h] = obj
    return Expr("DataStructure", obj.type, Expr("DataStructureHandle", h))


def _get(x, fname):
    if isinstance(x, Expr) and x.head == "DataStructure" and len(x.args) == 2 and \
            isinstance(x.args[1], Expr) and x.args[1].head == "DataStructureHandle":
        h = x.args[1].args[0]
        if isinstance(h, int) and h in _STORE:
            return _STORE[h]
    raise DSError(f"{fname}: {x} is not a valid DataStructure")


def is_ds(x):
    try:
        _get(x, "")
        return True
    except DSError:
        return False


@fn("CreateDataStructure", area=AREA, status="added",
    notes="CreateDataStructure[\"type\", args] for Stack, Queue, Deque, RingBuffer, DynamicArray, FixedArray, "
          "LinkedList, DoublyLinkedList, PriorityQueue, HashSet, OrderedHashSet, HashTable, OrderedHashTable, "
          "LeastRecentlyUsedCache, Value, Counter, SortedMultiset, DisjointSet, BitVector; operations ds[\"Op\", "
          "...] mutate in place; empty Pop/Peek raise")
def _createds(typ: String, *args: Expr) -> Expr:
    if not isinstance(typ, str):
        raise DSError(f"CreateDataStructure: the type {typ} must be a string")
    return _wrap(_create(typ, list(args)))


@fn("DataStructure", area=AREA, status="added",
    notes="DataStructure[\"type\", DataStructureHandle[n]] is the (mutable) object returned by "
          "CreateDataStructure; DataStructure[\"type\", data] with explicit data creates one")
def _datastructure(typ, data, *rest):
    if isinstance(data, Expr) and data.head == "DataStructureHandle":
        raise Unevaluated()
    if not isinstance(typ, str):
        raise DSError("DataStructure: the type must be a string")
    return _wrap(_create(typ, [data, *rest]))


@fn("DataStructureQ", area=AREA, status="added", notes="DataStructureQ[x] / DataStructureQ[x, \"type\"]")
def _datastructureq(x: Expr, typ: Expr = None) -> Boolean:
    if not is_ds(x):
        return False
    return typ is None or (isinstance(typ, str) and x.args[0] == typ)


def _dispatch(obj, op, args):
    if op == "Copy":
        if args:
            raise DSError(f"{obj.type}: \"Copy\" takes no arguments")
        return _wrap(obj.copy())
    if op == "Visualization":
        raise NotImplementedError(f"{obj.type}: \"Visualization\" is not available")
    m = getattr(obj, "op_" + op, None) if isinstance(op, str) else None
    if m is None:
        raise DSError(f"{obj.type}: unknown operation {op!r}")
    try:
        return m(*args)
    except TypeError as e:
        raise DSError(f"{obj.type}: wrong number of arguments for \"{op}\"") from e


def _ds_hook(head, args, evf):
    if not is_ds(head):
        return K.UNHANDLED
    if not args:
        raise DSError("DataStructure: give an operation name")
    obj = _get(head, "DataStructure")
    return _dispatch(obj, args[0], list(args[1:]))


KL.COMPOUND_HEAD_HOOKS["DataStructure"] = _ds_hook

_prev_normal = K._BUILTINS.get("Normal")


def _ds_normal(a, ev_, expr):
    if len(a) == 1 and is_ds(a[0]):
        return _get(a[0], "Normal").normal()
    return _prev_normal(a, ev_, expr)


K._BUILTINS["Normal"] = _ds_normal

_prev_length = K._BUILTINS.get("Length")


def _ds_length(a, ev_, expr):
    if len(a) == 1 and is_ds(a[0]):
        return _get(a[0], "Length").length()
    return _prev_length(a, ev_, expr)


K._BUILTINS["Length"] = _ds_length

from .wl_numerics import SYSTEM_CONSTANTS  # noqa: E402

SYSTEM_CONSTANTS["$DataStructures"] = L(*sorted(TYPES))
record("$DataStructures", AREA, "added",
       "the data structure types this implementation of CreateDataStructure supports (a subset of Wolfram's list)")
