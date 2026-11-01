"""Opt-in types: signatures from annotations, TypeSignature, TypeCheck and checking sessions.

Run: python examples/04_types.py
"""
from mikoshilang import Session

s = Session(check_types=True)

# Library heads carry signatures read from their Python annotations
print("TypeSignature[BitLength]  =", s.evaluate("TypeSignature[BitLength]"))

# Declare a signature for your own function
s.evaluate("TypeSignature[area, {Real, Real} -> Real]; area[w_, h_] := w h")
print("area[2, 3]                =", s.evaluate("area[2, 3]"))
print("area[2, h] (symbolic ok)  =", s.evaluate("area[2, h]"))
try:
    s.evaluate('area[2, "3"]')
except TypeError as e:
    print('area[2, "3"]              -> TypeError:', e)

# TypeCheck always reports instead of raising
print('TypeCheck[area[2, "3"]]   =', s.evaluate('TypeCheck[area[2, "3"]]'))

# Report mode: keep going and collect mismatches
r = Session(check_types="report")
r.evaluate("TypeSignature[area, {Real, Real} -> Real]; area[w_, h_] := w h")
r.evaluate('area[2, "3"]')
print("report-mode mismatches    =", len(r.type_mismatches))
