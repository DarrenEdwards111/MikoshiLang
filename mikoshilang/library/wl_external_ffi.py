"""Wolfram batch W30 (part 3): foreign functions, raw memory and managed objects through ctypes.

Every function returns a correct result, raises a clear error, or stays unevaluated for symbolic input
(see CONVENTIONS.md). Memory safety: RawMemoryRead/Write/Import only touch memory that MikoshiLang allocated
(RawMemoryAllocate/RawMemoryExport) and that has not been freed; pointers into such blocks (e.g. returned by strchr)
are allowed, any other address raises ValueError instead of crashing the process. ForeignFunctionLoad works with any
C-compatible shared library found by ctypes (the tests use only the C standard library); calling a function with a
wrong signature is, as in Wolfram, the caller's responsibility.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import itertools
import os
import struct
import sys

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, is_list
from .wl_common import ev, is_sym, apply

AREA = "external interfaces: foreign functions (W30)"


def _show(x):
    try:
        from .. import formatting as F
        return F.input_form(x)
    except Exception:
        return repr(x)


def _assoc(pairs):
    return KL.make_assoc([Expr("Rule", k, v) for k, v in pairs])


def _assoc_get(a, k, default=None):
    for r in a.args:
        if r.args[0] == k:
            return r.args[1]
    return default


# ═══════════════════════════════════════ C types ═══════════════════════════════════════

CTYPES = {
    "UnsignedInteger8": ctypes.c_uint8, "Integer8": ctypes.c_int8, "UnsignedInteger16": ctypes.c_uint16,
    "Integer16": ctypes.c_int16, "UnsignedInteger32": ctypes.c_uint32, "Integer32": ctypes.c_int32,
    "UnsignedInteger64": ctypes.c_uint64, "Integer64": ctypes.c_int64, "MachineInteger": ctypes.c_int64,
    "CUnsignedChar": ctypes.c_ubyte, "CSignedChar": ctypes.c_byte, "CUnsignedShort": ctypes.c_ushort,
    "CShort": ctypes.c_short, "CUnsignedInt": ctypes.c_uint, "CInt": ctypes.c_int,
    "CUnsignedLong": ctypes.c_ulong, "CLong": ctypes.c_long, "CSizeT": ctypes.c_size_t,
    "CFloat": ctypes.c_float, "CDouble": ctypes.c_double, "Real32": ctypes.c_float, "Real64": ctypes.c_double,
    "MachineReal": ctypes.c_double,
}
_FLOATS = {"CFloat", "CDouble", "Real32", "Real64", "MachineReal"}


def _is_int(v):
    return K.is_int(v) and not isinstance(v, bool)


def parse_ctype(t, fname, allow_void=False):
    """'name' for scalars, ('Opaque',) for void*, ('Ptr', elem) for typed pointers, 'Void'."""
    if isinstance(t, str):
        if t in CTYPES:
            return t
        if t == "OpaqueRawPointer":
            return ("Opaque",)
        if t == "Void":
            if allow_void:
                return "Void"
            raise ValueError(f"{fname}: \"Void\" is only allowed as a return type")
        if t == "RawPointer":
            raise ValueError(f"{fname}: give the element type of a pointer, as in TypeSpecifier[\"RawPointer\"][\"CInt\"]")
        raise ValueError(f"{fname}: unknown C type {t!r}")
    if isinstance(t, Expr) and t.head == "TypeSpecifier" and len(t.args) == 1:
        return parse_ctype(t.args[0], fname, allow_void)
    if isinstance(t, Expr) and isinstance(t.head, Expr) and t.head.head == "TypeSpecifier" and \
            t.head.args == ("RawPointer",) and len(t.args) == 1:
        el = parse_ctype(t.args[0], fname)
        return ("Ptr", el)
    if is_list(t):
        raise NotImplementedError(f"{fname}: struct types {{ty1, ty2, ...}} are not supported")
    raise ValueError(f"{fname}: {_show(t)} is not a C type")


def ctype_expr(t):
    if isinstance(t, str):
        return t
    if t[0] == "Opaque":
        return "OpaqueRawPointer"
    return Expr(Expr("TypeSpecifier", "RawPointer"), ctype_expr(t[1]))


def _ct(t):
    if t == "Void":
        return None
    if isinstance(t, str):
        return CTYPES[t]
    return ctypes.c_void_p


def _elem_name(t):
    """Element type name of a pointer type used in RawPointer[addr, type]."""
    return ctype_expr(t)


def _sizeof(t):
    if isinstance(t, str):
        return ctypes.sizeof(CTYPES[t])
    return ctypes.sizeof(ctypes.c_void_p)


def _check_int(v, t, fname):
    c = CTYPES[t]
    n = ctypes.sizeof(c) * 8
    signed = c(-1).value == -1
    lo, hi = (-(1 << (n - 1)), (1 << (n - 1)) - 1) if signed else (0, (1 << n) - 1)
    if not lo <= int(v) <= hi:
        raise ValueError(f"{fname}: {int(v)} is out of range for {t} ({lo} to {hi})")
    return int(v)


# ═══════════════════════════════════════ memory blocks ═══════════════════════════════════════

class Block:
    def __init__(self, size):
        self.buf = ctypes.create_string_buffer(max(size, 1))
        self.size = size
        self.addr = ctypes.addressof(self.buf)


BLOCKS = {}


def _find_block(addr):
    for b in BLOCKS.values():
        if b.addr <= addr < b.addr + max(b.size, 1):
            return b
    return None


def _check_range(addr, n, fname, base=None):
    b = _find_block(addr)
    if b is None and base is not None and _find_block(base) is not None:
        b0 = _find_block(base)
        raise ValueError(f"{fname}: access at byte offset {addr - b0.addr} is outside the {b0.size}-byte block")
    if b is None:
        raise ValueError(f"{fname}: the address {addr} is not in memory allocated by RawMemoryAllocate or "
                         f"RawMemoryExport (or it has been freed); MikoshiLang only reads and writes such memory")
    if addr + n > b.addr + b.size:
        raise ValueError(f"{fname}: access of {n} byte(s) at offset {addr - b.addr} is beyond the end of the "
                         f"{b.size}-byte block")
    return b


def _alloc(size):
    b = Block(size)
    BLOCKS[b.addr] = b
    return b


def _pointer_parts(p, fname):
    """(address, element type) of a RawPointer/OpaqueRawPointer/ManagedObject/ForeignCallback."""
    if isinstance(p, Expr):
        if p.head == "ManagedObject" and len(p.args) >= 2:
            return _pointer_parts(p.args[1], fname)
        if p.head == "RawPointer" and len(p.args) == 2 and _is_int(p.args[0]):
            return int(p.args[0]), parse_ctype(p.args[1], fname)
        if p.head == "OpaqueRawPointer" and len(p.args) == 1 and _is_int(p.args[0]):
            return int(p.args[0]), ("Opaque",)
        if p.head == "ForeignCallback" and len(p.args) == 3 and KL.is_assoc(p.args[2]):
            fp = _assoc_get(p.args[2], "FunctionPointer")
            return _pointer_parts(fp, fname)
    raise TypeError(f"{fname}: {_show(p)} is not a pointer (RawPointer, OpaqueRawPointer or ManagedObject)")


def _raw(addr, t):
    return Expr("RawPointer", addr, _elem_name(t))


# ═══════════════════════════════════════ managed objects ═══════════════════════════════════════

_MANAGED = {}
_MID = itertools.count(1)


def _managed(base, value, free):
    mid = next(_MID)
    _MANAGED[mid] = True
    return Expr("ManagedObject", base, value, free, _assoc([("ID", mid)]))


def _mo_id(m, fname):
    if isinstance(m, Expr) and m.head == "ManagedObject" and len(m.args) == 4 and KL.is_assoc(m.args[3]):
        mid = _assoc_get(m.args[3], "ID")
        if mid in _MANAGED:
            return mid
        raise ValueError(f"{fname}: unknown managed object")
    raise TypeError(f"{fname}: {_show(m)} is not a ManagedObject")


@fn("CreateManagedObject", area=AREA, status="added",
    notes="Partial. CreateManagedObject[expr, free] gives ManagedObject[expr, expr, free, <|\"ID\" -> n|>] (the ID "
          "is MikoshiLang's handle; Wolfram shows three arguments) and CreateManagedObject[expr, man] one sharing "
          "man's base value and freeing function. MikoshiLang has no reference counting, so the freeing function "
          "is never called automatically (this is the partial part); UnmanageObject deactivates the object. "
          "Check: reference (Wolfram documentation: man[\"Value\"] is foo) in tests.")
def _createmanaged(expr: Expr, free: Expr) -> Expr:
    if isinstance(free, Expr) and free.head == "ManagedObject":
        _mo_id(free, "CreateManagedObject")
        return _managed(free.args[0], expr, free.args[2])
    return _managed(expr, expr, free)


@fn("ManagedObject", area=AREA, status="added",
    notes="Partial. ManagedObject[base, value, free, <|\"ID\" -> n|>] (made by CreateManagedObject, "
          "RawMemoryAllocate, RawMemoryExport and CreateForeignCallback) has properties \"Value\", \"BaseValue\", "
          "\"FreeingFunction\" and \"Active\"; functions expecting pointers unwrap it. The freeing function is not "
          "run automatically (no reference counting); RawMemoryFree frees managed memory explicitly. An unknown ID "
          "raises ValueError. Check: reference (Wolfram documentation property examples) in tests.")
def _managedobject(*args: Expr) -> Expr:
    _mo_id(Expr("ManagedObject", *args), "ManagedObject")
    raise Unevaluated()


def _mo_hook(head, args, ev_):
    if len(args) != 1 or len(head.args) != 4:
        return K.UNHANDLED
    mid = _mo_id(head, "ManagedObject")
    p = args[0]
    props = {"Value": head.args[1], "BaseValue": head.args[0], "FreeingFunction": head.args[2],
             "Active": _MANAGED[mid]}
    if isinstance(p, str) and p in props:
        return props[p]
    raise ValueError(f"ManagedObject: unknown property {_show(p)} ({', '.join(props)})")


KL.COMPOUND_HEAD_HOOKS["ManagedObject"] = _mo_hook


@fn("UnmanageObject", area=AREA, status="added",
    notes="UnmanageObject[man] deactivates the ManagedObject man (man[\"Active\"] becomes False) and gives its "
          "value; deactivating twice raises ValueError, a non-managed argument TypeError. Check: reference (Wolfram "
          "documentation: UnmanageObject[man] gives foo and man[\"Active\"] becomes False) in tests.")
def _unmanage(m: Expr) -> Expr:
    if isinstance(m, Symbol):
        raise Unevaluated()
    mid = _mo_id(m, "UnmanageObject")
    if not _MANAGED[mid]:
        raise ValueError("UnmanageObject: the object is already unmanaged")
    _MANAGED[mid] = False
    return m.args[1]


# ═══════════════════════════════════════ pointers ═══════════════════════════════════════

def _ptr_hook(head, args, ev_):
    if len(args) != 1:
        return K.UNHANDLED
    addr, t = _pointer_parts(head, head.head)
    p = args[0]
    if p == "Address":
        return addr
    if p == "Type" and head.head == "RawPointer":
        return head.args[1]
    known = '"Address", "Type"' if head.head == "RawPointer" else '"Address"'
    raise ValueError(f"{head.head}: unknown property {_show(p)} ({known})")


KL.COMPOUND_HEAD_HOOKS["RawPointer"] = _ptr_hook
KL.COMPOUND_HEAD_HOOKS["OpaqueRawPointer"] = _ptr_hook


@fn("RawPointer", area=AREA, status="added",
    notes="RawPointer[addr, type] is a typed pointer (made by RawMemoryAllocate, RawMemoryExport, "
          "UnmanageObject or a foreign function returning TypeSpecifier[\"RawPointer\"][type]); properties "
          "\"Address\" and \"Type\"; RawPointer[ptr, type] casts a pointer. The address must be a non-negative "
          "integer and the type a C type (ValueError/TypeError otherwise). Check: reference (Wolfram documentation) "
          "and round trips through RawMemoryWrite/RawMemoryRead in tests.")
def _rawpointer(a: Expr, t: Expr) -> Expr:
    if isinstance(a, Expr) and a.head in ("RawPointer", "OpaqueRawPointer", "ManagedObject"):
        addr, _ = _pointer_parts(a, "RawPointer")
        parse_ctype(t, "RawPointer")
        return Expr("RawPointer", addr, t)
    if isinstance(a, Symbol):
        raise Unevaluated()
    if not _is_int(a) or int(a) < 0:
        raise TypeError("RawPointer: the address must be a non-negative integer")
    parse_ctype(t, "RawPointer")
    raise Unevaluated()


@fn("OpaqueRawPointer", area=AREA, status="added",
    notes="OpaqueRawPointer[addr] is an untyped pointer (void*), as returned by ForeignPointerLookup; "
          "OpaqueRawPointer[ptr] casts a typed pointer; property \"Address\". A negative or non-integer address "
          "raises TypeError. Check: reference (Wolfram documentation) in tests.")
def _opaque(a: Expr) -> Expr:
    if isinstance(a, Expr) and a.head in ("RawPointer", "ManagedObject", "ForeignCallback"):
        return Expr("OpaqueRawPointer", _pointer_parts(a, "OpaqueRawPointer")[0])
    if isinstance(a, Symbol):
        raise Unevaluated()
    if not _is_int(a) or int(a) < 0:
        raise TypeError("OpaqueRawPointer: the address must be a non-negative integer")
    raise Unevaluated()


@fn("NullRawPointerQ", area=AREA, status="added",
    notes="NullRawPointerQ[ptr] is True when the address of a RawPointer, OpaqueRawPointer or managed pointer is 0 "
          "and False otherwise (also False for non-pointers, as a predicate). Check: reference (Wolfram "
          "documentation: NullRawPointerQ[OpaqueRawPointer[0]] is True) in tests.")
def _nullq(p: Expr) -> bool:
    try:
        return _pointer_parts(p, "NullRawPointerQ")[0] == 0
    except TypeError:
        return False


# ═══════════════════════════════════════ raw memory ═══════════════════════════════════════

def _encode_value(v, t, fname):
    if isinstance(t, str):
        if t in _FLOATS:
            from .wl_external_compile import _num
            n = _num(v)
            if n is None or isinstance(n, complex):
                raise TypeError(f"{fname}: expected a real number for {t}, not {_show(v)}")
            return struct.pack("<d" if ctypes.sizeof(CTYPES[t]) == 8 else "<f", float(n)) if \
                sys.byteorder == "little" else bytes(CTYPES[t](float(n)))
        if v is True or v is False:
            v = int(v)
        if not _is_int(v):
            raise TypeError(f"{fname}: expected an integer for {t}, not {_show(v)}")
        return bytes(CTYPES[t](_check_int(v, t, fname)))
    addr, _ = _pointer_parts(v, fname)
    return bytes(ctypes.c_void_p(addr))


def _decode_value(raw, t):
    if isinstance(t, str):
        return CTYPES[t].from_buffer_copy(raw).value
    a = ctypes.c_void_p.from_buffer_copy(raw).value or 0
    return Expr("OpaqueRawPointer", a) if t[0] == "Opaque" else _raw(a, t[1])


def _block_ptr(b, t, managed=True):
    p = _raw(b.addr, t)
    return _managed(p, p, Symbol("RawMemoryFree")) if managed else p


@fn("RawMemoryAllocate", area=AREA, status="added",
    notes="RawMemoryAllocate[type] and RawMemoryAllocate[type, n] allocate zeroed memory for one or n elements of a "
          "C type (sizes from ctypes) and give ManagedObject[RawPointer[addr, type], ...]; n must be a positive "
          "integer (ValueError) and type a known C type (ValueError). Check: reference (Wolfram documentation CInt "
          "and CFloat examples) and round trips in tests.")
def _rawalloc(t: Expr, n: Expr = 1) -> Expr:
    ty = parse_ctype(t, "RawMemoryAllocate")
    if not _is_int(n) or int(n) < 1:
        raise ValueError("RawMemoryAllocate: the number of elements must be a positive integer")
    return _block_ptr(_alloc(_sizeof(ty) * int(n)), ty)


@fn("RawMemoryFree", area=AREA, status="added",
    notes="RawMemoryFree[ptr] frees memory from RawMemoryAllocate/RawMemoryExport (the start address of the block) "
          "and gives Null; later reads raise ValueError. Freeing other or already freed memory raises ValueError. "
          "Check: reference (Wolfram documentation) and the ValueError after freeing in tests.")
def _rawfree(p: Expr) -> Expr:
    addr, _ = _pointer_parts(p, "RawMemoryFree")
    if addr not in BLOCKS:
        raise ValueError("RawMemoryFree: the pointer was not allocated by RawMemoryAllocate/RawMemoryExport, or has "
                         "already been freed")
    del BLOCKS[addr]
    if isinstance(p, Expr) and p.head == "ManagedObject" and len(p.args) == 4:
        try:
            _MANAGED[_mo_id(p, "RawMemoryFree")] = False
        except (TypeError, ValueError):
            pass
    return None


def _typed_ptr(p, fname):
    addr, t = _pointer_parts(p, fname)
    if isinstance(t, tuple) and t[0] == "Opaque":
        raise TypeError(f"{fname}: an OpaqueRawPointer has no element type; cast it with RawPointer[ptr, type]")
    return addr, t


@fn("RawMemoryRead", area=AREA, status="added",
    notes="RawMemoryRead[ptr] and RawMemoryRead[ptr, offset] read the element of the pointer's type at the given "
          "element offset (integers exact, CFloat/CDouble as reals, pointer elements as RawPointer). Only memory "
          "from RawMemoryAllocate/RawMemoryExport may be read: other addresses and out-of-bounds offsets raise "
          "ValueError. Check: reference (Wolfram documentation: CInt -1 round trip, string bytes) in tests.")
def _rawread(p: Expr, off: Expr = 0) -> Expr:
    addr, t = _typed_ptr(p, "RawMemoryRead")
    if not _is_int(off):
        raise TypeError("RawMemoryRead: the offset must be an integer")
    sz = _sizeof(t)
    a = addr + int(off) * sz
    _check_range(a, sz, "RawMemoryRead", addr)
    return _decode_value(ctypes.string_at(a, sz), t)


@fn("RawMemoryWrite", area=AREA, status="added",
    notes="RawMemoryWrite[ptr, val] and RawMemoryWrite[ptr, val, offset] store val as the pointer's element type at "
          "the element offset and give the (unmanaged) RawPointer; integers out of the type's range, wrong value "
          "types (TypeError) and addresses outside MikoshiLang-allocated blocks (ValueError) are rejected. Check: "
          "reference (Wolfram documentation: write 65 to \"a string\" gives \"A string\") in tests.")
def _rawwrite(p: Expr, val: Expr, off: Expr = 0) -> Expr:
    addr, t = _typed_ptr(p, "RawMemoryWrite")
    if not _is_int(off):
        raise TypeError("RawMemoryWrite: the offset must be an integer")
    sz = _sizeof(t)
    a = addr + int(off) * sz
    _check_range(a, sz, "RawMemoryWrite", addr)
    data = _encode_value(val, t, "RawMemoryWrite")
    ctypes.memmove(a, data, sz)
    return _raw(addr, t)


def _encoding(opts, fname):
    enc = opts.get("CharacterEncoding", "UTF-8")
    if not isinstance(enc, str):
        raise ValueError(f"{fname}: CharacterEncoding must be a string")
    from .wl_text import _codec
    return _codec(enc, fname)


def _opts(args, fname):
    args = list(args)
    opts = {}
    while args and isinstance(args[-1], Expr) and args[-1].head in ("Rule", "RuleDelayed") and \
            (args[-1].args[0] == Symbol("CharacterEncoding") or args[-1].args[0] == "CharacterEncoding"):
        opts["CharacterEncoding"] = args.pop().args[1]
    for a in args:
        if isinstance(a, Expr) and a.head == "Rule":
            raise ValueError(f"{fname}: unknown option {_show(a.args[0])} (CharacterEncoding)")
    return args, opts


@fn("RawMemoryExport", area=AREA, status="added",
    notes="RawMemoryExport[expr] copies a string (encoded, NUL-terminated, type \"UnsignedInteger8\") or a "
          "ByteArray into new memory, RawMemoryExport[expr, type] a number or a list of numbers of a C type; gives "
          "ManagedObject[RawPointer[addr, type], ...]. Option CharacterEncoding (default \"UTF-8\"). Values that "
          "do not fit the type raise ValueError/TypeError; NumericArray raises NotImplementedError (not available). "
          "Check: reference (Wolfram documentation: RawMemoryExport[-1, \"CInt\"] reads back -1) in tests.")
def _rawexport(*args: Expr) -> Expr:
    pos, opts = _opts(args, "RawMemoryExport")
    if not 1 <= len(pos) <= 2:
        raise TypeError("RawMemoryExport: expected RawMemoryExport[expr] or RawMemoryExport[expr, type]")
    x = pos[0]
    from .wl_importexport import is_byte_array, ba_bytes
    if len(pos) == 1:
        if isinstance(x, str):
            data = x.encode(_encoding(opts, "RawMemoryExport")) + b"\0"
        elif is_byte_array(x):
            data = ba_bytes(x)
        elif isinstance(x, Expr) and x.head == "NumericArray":
            raise NotImplementedError("RawMemoryExport: NumericArray is not available in MikoshiLang")
        else:
            raise TypeError("RawMemoryExport: give a type for numbers and lists: RawMemoryExport[expr, type]")
        b = _alloc(len(data))
        ctypes.memmove(b.addr, data, len(data))
        return _block_ptr(b, "UnsignedInteger8")
    t = parse_ctype(pos[1], "RawMemoryExport")
    if isinstance(x, str) and t in ("UnsignedInteger8", "CUnsignedChar", "CSignedChar", "Integer8"):
        data = x.encode(_encoding(opts, "RawMemoryExport")) + b"\0"
    else:
        items = x.args if is_list(x) else (x,)
        data = b"".join(_encode_value(v, t, "RawMemoryExport") for v in items)
    b = _alloc(len(data))
    ctypes.memmove(b.addr, data, len(data))
    return _block_ptr(b, t)


@fn("RawMemoryImport", area=AREA, status="added",
    notes="RawMemoryImport[ptr, \"String\"] reads a NUL-terminated string, {\"String\", n} n bytes, "
          "{\"ByteArray\", n} a ByteArray and {\"List\", n} n elements of the pointer's type; option "
          "CharacterEncoding (default \"UTF-8\"). Only MikoshiLang-allocated memory can be read (ValueError "
          "otherwise, including a missing terminator or a length past the block); {\"NumericArray\", n} raises "
          "NotImplementedError. Check: reference (Wolfram documentation: \"A string\" bytes {65, 32, 115, ...}) "
          "in tests.")
def _rawimport(*args: Expr) -> Expr:
    pos, opts = _opts(args, "RawMemoryImport")
    if len(pos) != 2:
        raise TypeError("RawMemoryImport: expected RawMemoryImport[ptr, format]")
    p, f = pos
    addr, t = _pointer_parts(p, "RawMemoryImport")
    if f == "String":
        b = _check_range(addr, 1, "RawMemoryImport")
        raw = ctypes.string_at(addr, b.addr + b.size - addr)
        if b"\0" not in raw:
            raise ValueError("RawMemoryImport: no NUL terminator before the end of the block; use {\"String\", n}")
        return raw.split(b"\0", 1)[0].decode(_encoding(opts, "RawMemoryImport"), errors="strict")
    if is_list(f) and len(f.args) == 2 and isinstance(f.args[0], str):
        kind, n = f.args
        if not _is_int(n) or int(n) < 0:
            raise ValueError("RawMemoryImport: the length must be a non-negative integer")
        n = int(n)
        if kind in ("String", "ByteArray"):
            if n:
                _check_range(addr, n, "RawMemoryImport")
            raw = ctypes.string_at(addr, n)
            if kind == "String":
                return raw.decode(_encoding(opts, "RawMemoryImport"))
            from .wl_importexport import byte_array
            return byte_array(raw)
        if kind == "List":
            if isinstance(t, tuple) and t[0] == "Opaque":
                raise TypeError("RawMemoryImport: an OpaqueRawPointer has no element type")
            sz = _sizeof(t)
            if n:
                _check_range(addr, n * sz, "RawMemoryImport")
            return L(*(_decode_value(ctypes.string_at(addr + i * sz, sz), t) for i in range(n)))
        if kind == "NumericArray":
            raise NotImplementedError("RawMemoryImport: NumericArray is not available in MikoshiLang")
    raise ValueError(f"RawMemoryImport: unknown format {_show(f)} (\"String\", {{\"String\", n}}, "
                     f"{{\"ByteArray\", n}}, {{\"List\", n}})")


# ═══════════════════════════════════════ libraries and foreign functions ═══════════════════════════════════════

_LIBS = {}
_FUNCS = {}        # function address -> (ctypes function object, file, name)


def _lib(spec, fname):
    if not isinstance(spec, str):
        if isinstance(spec, Symbol):
            raise Unevaluated()
        raise TypeError(f"{fname}: the library must be a string (a name such as \"c\" or a file path)")
    path = spec
    if not os.path.sep in spec:
        found = ctypes.util.find_library(spec[3:] if spec.startswith("lib") else spec)
        if found is None and spec in ("c", "libc") and sys.platform.startswith("win"):
            found = "msvcrt"
        path = found or spec
    if path in _LIBS:
        return _LIBS[path], path
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        raise ValueError(f"{fname}: cannot load the library {spec!r} ({e})")
    _LIBS[path] = lib
    return lib, path


def _symbol(lib, name, fname):
    if not isinstance(name, str):
        raise TypeError(f"{fname}: the function name must be a string")
    try:
        return getattr(lib, name)
    except AttributeError:
        raise ValueError(f"{fname}: the library has no symbol {name!r}")


@fn("ForeignPointerLookup", area=AREA, status="added",
    notes="ForeignPointerLookup[lib, \"sym\"] gives OpaqueRawPointer[addr] for a symbol of a C-compatible library "
          "(lib a name resolved with ctypes.util.find_library, e.g. \"c\", or a path), like dlsym. A missing "
          "library or symbol raises ValueError. Check: reference (Wolfram documentation) and the address matching "
          "ForeignFunctionLoad's \"FunctionPointer\" in tests.")
def _ptrlookup(lib: str, name: str) -> Expr:
    l, path = _lib(lib, "ForeignPointerLookup")
    f = _symbol(l, name, "ForeignPointerLookup")
    addr = ctypes.cast(f, ctypes.c_void_p).value
    _FUNCS.setdefault(addr, (f, path, name))
    return Expr("OpaqueRawPointer", addr)


def _signature(ty, fname):
    if not (isinstance(ty, Expr) and ty.head == "Rule" and is_list(ty.args[0])):
        raise TypeError(f"{fname}: the type must be {{argtypes...}} -> returntype")
    args = tuple(parse_ctype(a, fname) for a in ty.args[0].args)
    ret = parse_ctype(ty.args[1], fname, allow_void=True)
    return args, ret


def _type_rule(args, ret):
    return Expr("Rule", L(*(ctype_expr(a) for a in args)), ctype_expr(ret))


@fn("ForeignFunctionLoad", area=AREA, status="added",
    notes="ForeignFunctionLoad[lib, \"fun\", {argtypes...} -> rettype] and ForeignFunctionLoad[ptr, type] (ptr from "
          "ForeignPointerLookup) load a C function with ctypes and give ForeignFunction[file, name, <|\"Type\" -> "
          "..., \"FunctionPointer\" -> OpaqueRawPointer[addr]|>]. C types: \"Integer8\"...\"Integer64\", "
          "\"UnsignedInteger8\"...\"UnsignedInteger64\", \"CUnsignedChar\", \"CSignedChar\", \"CShort\", "
          "\"CUnsignedShort\", \"CInt\", \"CUnsignedInt\", \"CLong\", \"CUnsignedLong\", \"CSizeT\", \"CFloat\", "
          "\"CDouble\", \"OpaqueRawPointer\", TypeSpecifier[\"RawPointer\"][t] and \"Void\" (result only); struct "
          "types raise NotImplementedError. Missing libraries or symbols raise ValueError, unknown pointers "
          "ValueError. Deviation: the \"RawPointer\"::[t] input syntax is not parsed. Check: reference (Wolfram "
          "documentation) and libc results (abs, strlen, atoi, toupper) compared with Python in tests.")
def _ffload(a: Expr, b: Expr, c: Expr = None) -> Expr:
    if c is None:
        addr, _ = _pointer_parts(a, "ForeignFunctionLoad")
        if addr not in _FUNCS:
            raise ValueError("ForeignFunctionLoad: the pointer was not obtained from ForeignPointerLookup")
        f0, path, name = _FUNCS[addr]
        ty = b
        file, nm = Symbol("None"), Symbol("None")
    else:
        lib, path = _lib(a, "ForeignFunctionLoad")
        f0 = _symbol(lib, b, "ForeignFunctionLoad")
        name, ty = b, c
        file, nm = path, name
    args, ret = _signature(ty, "ForeignFunctionLoad")
    addr = ctypes.cast(f0, ctypes.c_void_p).value
    _FUNCS.setdefault(addr, (f0, path, name))
    return Expr("ForeignFunction", file, nm, _assoc([("FunctionPointer", Expr("OpaqueRawPointer", addr)),
                                                     ("Type", _type_rule(args, ret))]))


def _ff_parts(ff, fname="ForeignFunction"):
    if len(ff.args) != 3 or not KL.is_assoc(ff.args[2]):
        raise TypeError(f"{fname}: malformed ForeignFunction (make one with ForeignFunctionLoad)")
    fp = _assoc_get(ff.args[2], "FunctionPointer")
    addr, _ = _pointer_parts(fp, fname)
    if addr not in _FUNCS:
        raise ValueError(f"{fname}: unknown function pointer (load it with ForeignFunctionLoad)")
    args, ret = _signature(_assoc_get(ff.args[2], "Type"), fname)
    return addr, args, ret


def _where(pos):
    return "as the result" if pos == 0 else f"at position {pos}"


def _to_c(v, t, fname, pos):
    if isinstance(t, str):
        if t in _FLOATS:
            from .wl_external_compile import _num
            n = _num(v)
            if n is None or isinstance(n, complex) or isinstance(v, bool):
                raise TypeError(f"{fname}: expected an argument with type {t} but found {_show(v)} {_where(pos)}")
            return CTYPES[t](float(n))
        if not _is_int(v):
            raise TypeError(f"{fname}: expected an argument with type {t} but found {_show(v)} {_where(pos)}")
        return CTYPES[t](_check_int(v, t, fname))
    try:
        addr, _ = _pointer_parts(v, fname)
    except TypeError:
        raise TypeError(f"{fname}: expected a pointer {_where(pos)} but found {_show(v)}")
    return ctypes.c_void_p(addr)


def _from_c(r, t):
    if t == "Void":
        return Symbol("Null")
    if isinstance(t, str):
        return r
    a = r or 0
    return Expr("OpaqueRawPointer", a) if t[0] == "Opaque" else _raw(a, t[1])


_CB_ERRORS = []


def _ff_hook(head, args, ev_):
    if len(head.args) != 3:
        return K.UNHANDLED
    addr, ats, ret = _ff_parts(head)
    if len(args) != len(ats):
        raise ValueError(f"ForeignFunction: called with {len(args)} argument(s); {len(ats)} expected")
    f0 = _FUNCS[addr][0]
    proto = ctypes.CFUNCTYPE(_ct(ret), *(_ct(a) for a in ats))
    cf = proto(addr)
    cargs = [_to_c(v, t, "ForeignFunction", i + 1) for i, (v, t) in enumerate(zip(args, ats))]
    mark = len(_CB_ERRORS)
    r = cf(*cargs)
    if len(_CB_ERRORS) > mark:
        e = _CB_ERRORS[mark]
        del _CB_ERRORS[mark:]
        raise e
    _ = f0
    return _from_c(r, ret)


KL.COMPOUND_HEAD_HOOKS["ForeignFunction"] = _ff_hook


@fn("ForeignFunction", area=AREA, status="added",
    notes="ForeignFunction[file, name, <|\"FunctionPointer\" -> ..., \"Type\" -> ...|>] (made by "
          "ForeignFunctionLoad) calls the C function: arguments are checked against the C types (wrong types raise "
          "TypeError like Wolfram's ForeignFunction::argtype, out-of-range integers ValueError); pointer arguments "
          "accept RawPointer, OpaqueRawPointer, ManagedObject and ForeignCallback; errors raised inside callbacks "
          "are re-raised after the call. A wrong argument count raises ValueError. Check: reference (Wolfram "
          "documentation) and libc results compared with Python (abs, strlen, toupper, qsort) in tests.")
def _foreignfunction(*args: Expr) -> Expr:
    _ff_parts(Expr("ForeignFunction", *args))
    raise Unevaluated()


_CALLBACKS = {}


@fn("CreateForeignCallback", area=AREA, status="added",
    notes="CreateForeignCallback[f, {argtypes...} -> rettype] makes a C function pointer (ctypes CFUNCTYPE) that "
          "evaluates f on the converted arguments and converts the result to rettype; gives ManagedObject["
          "ForeignCallback[f, type, <|\"FunctionPointer\" -> OpaqueRawPointer[addr]|>], ...]. A result of the "
          "wrong type raises TypeError when the callback is called (re-raised by the ForeignFunction call). The "
          "callback stays alive for the session. Check: reference (Wolfram documentation) and libc qsort with a "
          "comparison callback in tests.")
def _createcb(f: Expr, ty: Expr) -> Expr:
    ats, ret = _signature(ty, "CreateForeignCallback")
    proto = ctypes.CFUNCTYPE(_ct(ret), *(_ct(a) for a in ats))

    def call(*cargs):
        try:
            vals = [_from_c(a, t) for a, t in zip(cargs, ats)]
            r = apply(f, *vals)
            if ret == "Void":
                return None
            return _to_c(r, ret, "ForeignCallback", 0).value
        except Exception as e:      # ctypes cannot propagate exceptions through C frames
            _CB_ERRORS.append(e)
            return 0 if ret != "Void" else None
    cb = proto(call)
    addr = ctypes.cast(cb, ctypes.c_void_p).value
    _CALLBACKS[addr] = cb
    obj = Expr("ForeignCallback", f, _type_rule(ats, ret), _assoc([("FunctionPointer",
                                                                     Expr("OpaqueRawPointer", addr))]))
    return _managed(obj, obj, Symbol("Identity"))


def _cb_hook(head, args, ev_):
    if len(head.args) != 3 or len(args) != 1:
        return K.UNHANDLED
    p = args[0]
    if p == "Function":
        return head.args[0]
    if p == "FunctionPointer":
        return _assoc_get(head.args[2], "FunctionPointer")
    if p == "Type":
        return head.args[1]
    raise ValueError(f"ForeignCallback: unknown property {_show(p)} (\"Function\", \"FunctionPointer\", \"Type\")")


KL.COMPOUND_HEAD_HOOKS["ForeignCallback"] = _cb_hook


@fn("ForeignCallback", area=AREA, status="added",
    notes="ForeignCallback[f, type, <|\"FunctionPointer\" -> ...|>] (made by CreateForeignCallback) is passed to "
          "foreign functions as a function pointer; properties \"Function\", \"FunctionPointer\" and \"Type\". An "
          "object whose pointer was not made by CreateForeignCallback raises ValueError. Check: reference (Wolfram "
          "documentation) and qsort in tests.")
def _foreigncb(f: Expr, ty: Expr, a: Assoc) -> Expr:
    if not KL.is_assoc(a):
        raise TypeError("ForeignCallback: malformed object (make one with CreateForeignCallback)")
    addr, _ = _pointer_parts(_assoc_get(a, "FunctionPointer"), "ForeignCallback")
    if addr not in _CALLBACKS:
        raise ValueError("ForeignCallback: unknown callback pointer (make one with CreateForeignCallback)")
    raise Unevaluated()
