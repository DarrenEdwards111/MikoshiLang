"""Heads that the legacy library also defined but which are evaluated by the
kernel builtins before any library rule runs (legacy rules were dead code).
They are covered by the kernel test suite and spot-checked in
tests/library/test_kernel_heads.py."""
from .core import record

KERNEL = """Apart Append Cancel D DSolve Det Diff Dot DotProduct Drop EvenQ First Flatten IntegerQ Inverse Join
Last Length Limit MemberQ Most N Negative NonNegative NumberQ OddQ Part PartialD Plus Position Positive Power
Prepend PrimeQ Range Rest Reverse Series Solve Sort StringJoin StringLength Take Times Together Total
Transpose""".split()

for _h in KERNEL:
    record(_h, "kernel", "kernel", "evaluated by the kernel builtin; legacy library rule was unreachable")
