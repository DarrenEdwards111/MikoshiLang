"""Differential tests for the interop phase-2 library heads: reserved external-evaluation heads and the
packed-array utilities (ToPackedArray, FromPackedArray, PackedArrayQ; also under their Developer` names).

References: reference.wolfram.com/language/Developer/ref/ToPackedArray.html, FromPackedArray.html and
PackedArrayQ.html (packing never changes the value of a list; lists that cannot be packed are returned
unchanged); ExternalEvaluate is a deliberate deviation (no code strings without a sandbox).
"""
import pytest

from _harness import Pred, Raises, check, ids
from mikoshilang.packed import PackedArray


def _packed(dtype=None, ndim=None):
    def f(v):
        return isinstance(v, PackedArray) and (dtype is None or v.array.dtype.kind == dtype) and \
            (ndim is None or v.array.ndim == ndim)
    return Pred(f, f"packed {dtype or ''}{ndim or ''}")


CASES = [
    # reserved heads: always raise, never run code
    ('ExternalEvaluate["Python", "1 + 1"]', Raises(NotImplementedError, match="^ExternalEvaluate: running")),
    ('ExternalEvaluate["Shell", "ls"]', Raises(NotImplementedError)),
    ('ExternalFunction["Python", "def f(x): return x"]', Raises(NotImplementedError, match="^ExternalFunction")),
    ('StartExternalSession["Python"]', Raises(NotImplementedError, match="^StartExternalSession")),
    ('Check[ExternalEvaluate["Python", "1"], "caught"]', '"caught"'),

    # ToPackedArray: same value, packed storage (Wolfram: packing does not change the list)
    ("ToPackedArray[{1, 2, 3}]", "{1, 2, 3}"),
    ("ToPackedArray[{1, 2, 3}]", _packed("i", 1)),
    ("ToPackedArray[{{1., 2.}, {3., 4.}}]", "{{1., 2.}, {3., 4.}}"),
    ("ToPackedArray[{{1., 2.}, {3., 4.}}]", _packed("f", 2)),
    ("ToPackedArray[{1, 2.5}]", "{1., 2.5}"),                        # mixed machine numbers pack as Real
    ("ToPackedArray[{1, 2}, Real]", "{1., 2.}"),
    ("ToPackedArray[{x, 1}]", "{x, 1}"),                             # symbolic: returned unchanged
    ("ToPackedArray[{1/2, 1}]", "{1/2, 1}"),                         # exact rational: unchanged
    ("ToPackedArray[{{1, 2}, {3}}]", "{{1, 2}, {3}}"),               # ragged: unchanged
    ("ToPackedArray[{2^70}]", "{1180591620717411303424}"),           # beyond int64: unchanged
    ("ToPackedArray[t]", "t"),
    ("ToPackedArray[{1.5}, Integer]", Raises(ValueError, match="^ToPackedArray: the list has non-integer")),
    ("ToPackedArray[{1}, Rational]", Raises(ValueError, match="^ToPackedArray: the type must be")),
    ('ToPackedArray[{1}, "Real"]', Raises(TypeError, match="^ToPackedArray: the type must be the symbol")),
    ("ToPackedArray[{1, 2}, Complex]", Raises(NotImplementedError)),
    ("Developer`ToPackedArray[{3, 4}]", "{3, 4}"),
    # Check: round trip through FromPackedArray is the identity
    ("Table[With[{l = RandomInteger[{-9, 9}, {k, 3}]}, FromPackedArray[ToPackedArray[l]] === l], {k, 1, 5}]",
     "{True, True, True, True, True}"),

    # FromPackedArray
    ("FromPackedArray[ToPackedArray[{1, 2}]]", "{1, 2}"),
    ("FromPackedArray[ToPackedArray[{{1., 2.}}]]", Pred(lambda v: not isinstance(v, PackedArray)
                                                         and not isinstance(v.args[0], PackedArray), "unpacked")),
    ("FromPackedArray[{a, b}]", "{a, b}"),
    ("FromPackedArray[f[x]]", "f[x]"),

    # PackedArrayQ
    ("PackedArrayQ[ToPackedArray[{1, 2, 3}]]", "True"),
    ("PackedArrayQ[{1, 2, 3}]", "False"),
    ("PackedArrayQ[ToPackedArray[{1, 2}], Integer]", "True"),
    ("PackedArrayQ[ToPackedArray[{1, 2}], Real]", "False"),
    ("PackedArrayQ[ToPackedArray[{{1., 2.}}], Real, 2]", "True"),
    ("PackedArrayQ[ToPackedArray[{{1., 2.}}], Real, 1]", "False"),
    ("PackedArrayQ[x]", "False"),
    ("PackedArrayQ[{1}, Integer, r]", "PackedArrayQ[{1}, Integer, r]"),
    ("PackedArrayQ[{1}, Number]", Raises(ValueError)),
    ("PackedArrayQ[{1}, Integer, 1.5]", Raises(TypeError)),
    ("Developer`PackedArrayQ[Developer`ToPackedArray[{1.}]]", "True"),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_wolfram_interop(case):
    check(case)
