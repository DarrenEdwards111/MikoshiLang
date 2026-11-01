"""Verification: Verify[] re-checks a result with an independent method and labels it.

Certified = an exact certificate was checked; Checked = symbolic or numerical evidence;
Unchecked = no checker applies; Failed[value, info] = the value is wrong.

Run: python examples/05_verify.py
"""
from mikoshilang import Session

s = Session()

for src in [
    "Verify[FactorInteger[2^64 + 1]]",   # primality certificates for each factor
    "Verify[GCD[462, 1071]]",            # Bezout coefficients
    "Verify[Solve[x^2 == 2, x]]",        # substitution of every solution
    "Verify[Integrate[Sin[x]^2, x]]",    # differentiate back
    "Verify[Det[{{1, 2}, {3, 4}}]]",     # independent determinant algorithm
    "Verify[Sort[{3, 1, 2}]]",           # no checker: honestly Unchecked
]:
    result = s.evaluate(src)
    label = str(result).split("[", 1)[0]
    print(f"{src:36} -> {label}")
    print(f"{'':36}    {str(result)[:110]}")
