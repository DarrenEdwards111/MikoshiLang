"""Fast gate for the opt-in type checker (see :mod:`mikoshilang.types`).

``ACTIVE`` counts evaluations currently running with type checking on. The evaluator and the library
dispatcher read only this integer, so when no checking session is active the checker costs one attribute
load per call.
"""
ACTIVE = 0
