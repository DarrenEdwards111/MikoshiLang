"""Reviewed MikoshiLang function library.

Every function exposed here was audited against an independent reference
(SymPy, mpmath, SciPy or Wolfram Language reference values); see
``STATUS`` for each name's audit status and notes, and
``docs/library_audit.md`` for the generated table.
"""
from .core import REGISTRY, STATUS, ERRORS, UNHANDLED, Unevaluated, dispatch  # noqa: F401
from . import elementary, special, numbertheory, combinatorics, algebra, calculus, linalg, lists, strings, statistics, applied, logic, graphs, numerics, transforms, plots, ml, kernelheads, assume, wl_lists, wl_strings, wl_assoc, wl_numerics, wl_special, wl_numcalc, wl_interp, wl_linalg  # noqa: F401,E402
from . import wl_symbolic, wl_calculus, wl_algebra  # noqa: F401,E402  (coverage batch 4)
