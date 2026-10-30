import pytest
import sympy as sp
from sympy import totient, mobius, primepi, prime, divisor_sigma, n_order, primitive_root

from _harness import Raises, Pred, check, ids

CASES = [
    ("IsPrime[7]", True), ("IsPrime[1]", False), ("IsPrime[-7]", True), ("IsPrime[2^31 - 1]", True),
    ("NextPrime[10]", "11"), ("NextPrime[13]", "17"), ("NextPrime[-10]", "-7"), ("NextPrime[1]", "2"),
    ("PrimePi[100]", str(primepi(100))), ("PrimePi[1]", "0"), ("PrimePi[10^6]", str(primepi(10**6))), ("PrimePi[-5]", "0"),
    ("Prime[1]", "2"), ("Prime[100]", str(prime(100))), ("Prime[10000]", str(prime(10000))), ("Prime[0]", Raises()),
    ("FactorInteger[360]", "{{2, 3}, {3, 2}, {5, 1}}"), ("FactorInteger[1]", "{{1, 1}}"),
    ("FactorInteger[-12]", "{{-1, 1}, {2, 2}, {3, 1}}"), ("FactorInteger[3/4]", "{{2, -2}, {3, 1}}"),
    ("PrimeFactorization[360]", "{{2, 3}, {3, 2}, {5, 1}}"), ("PrimeFactorization[97]", "{{97, 1}}"),
    ("PrimeFactorization[0]", "{{0, 1}}"),
    ("Divisors[12]", "{1, 2, 3, 4, 6, 12}"), ("Divisors[1]", "{1}"), ("Divisors[-6]", "{1, 2, 3, 6}"), ("Divisors[0]", Raises()),
    ("DivisorCount[12]", "6"), ("DivisorCount[1]", "1"), ("DivisorCount[2^10]", "11"),
    ("DivisorSigma[1, 12]", str(divisor_sigma(12, 1))), ("DivisorSigma[0, 12]", "6"), ("DivisorSigma[2, 10]", str(divisor_sigma(10, 2))),
    ("EulerPhi[10]", str(totient(10))), ("EulerPhi[1]", "1"), ("EulerPhi[0]", "0"), ("EulerPhi[97]", "96"),
    ("MoebiusMu[30]", str(mobius(30))), ("MoebiusMu[12]", "0"), ("MoebiusMu[1]", "1"),
    ("PowerMod[2, 10, 1000]", "24"), ("PowerMod[3, -1, 7]", "5"), ("PowerMod[2, -1, 4]", Raises()), ("PowerMod[5, 0, 3]", "1"),
    ("ModularInverse[3, 7]", "5"), ("ModularInverse[10, 17]", "12"), ("ModularInverse[2, 4]", Raises()),
    ("ChineseRemainder[{2, 3}, {3, 5}]", "8"), ("ChineseRemainder[{1, 2, 3}, {2, 3, 5}]", "23"),
    ("ChineseRemainder[{0, 1}, {2, 4}]", "ChineseRemainder[{0, 1}, {2, 4}]"),
    ("GCD[12, 18]", "6"), ("GCD[12, 18, 8]", "2"), ("GCD[1/2, 1/3]", "1/6"), ("GCD[0, 0]", "0"),
    ("LCM[4, 6]", "12"), ("LCM[2, 3, 4]", "12"), ("LCM[1/2, 1/3]", "1"),
    ("ExtendedGCD[12, 18]", Pred(lambda r: r.args[0] == 6 and 12 * r.args[1].args[0] + 18 * r.args[1].args[1] == 6, "Bezout")),
    ("ExtendedGCD[240, 46]", "{2, {-9, 47}}"), ("ExtendedGCD[5, 0]", "{5, {1, 0}}"),
    ("Fibonacci[10]", "55"), ("Fibonacci[0]", "0"), ("Fibonacci[-5]", "5"), ("Fibonacci[-6]", "-8"),
    ("FibonacciNumber[10]", "55"), ("FibonacciNumber[1]", "1"), ("FibonacciNumber[50]", str(sp.fibonacci(50))),
    ("Lucas[0]", "2"), ("Lucas[10]", "123"), ("Lucas[5]", "11"),
    ("LucasL[10]", "123"), ("LucasL[1]", "1"), ("LucasL[-3]", "-4"),
    ("Bernoulli[0]", "1"), ("Bernoulli[1]", "-1/2"), ("Bernoulli[12]", "-691/2730"),
    ("BernoulliNumber[1]", "-1/2"), ("BernoulliNumber[3]", "0"), ("BernoulliNumber[4]", "-1/30"),
    ("BernoulliB[1]", "-1/2"), ("BernoulliB[6]", "1/42"), ("BernoulliB[2, x]", "x^2 - x + 1/6"),
    ("EulerNumber[4]", "5"), ("EulerNumber[6]", "-61"), ("EulerNumber[3]", "0"),
    ("EulerE[4]", "5"), ("EulerE[10]", "-50521"), ("EulerE[2, x]", "x^2 - x"),
    ("Catalan[5]", Raises(ValueError, "CatalanNumber")),
    ("CatalanNumber[5]", "42"), ("CatalanNumber[0]", "1"), ("CatalanNumber[10]", "16796"),
    ("Bell[5]", "52"), ("Bell[0]", "1"), ("Bell[10]", "115975"),
    ("BellNumber[5]", "52"), ("BellNumber[1]", "1"), ("BellNumber[7]", "877"),
    ("BellB[5]", "52"), ("BellB[3, x]", "x + 3 x^2 + x^3"), ("BellB[0]", "1"),
    ("StirlingS1[5, 2]", "-50"), ("StirlingS1[4, 4]", "1"), ("StirlingS1[5, 3]", "35"), ("StirlingS1[3, 0]", "0"),
    ("StirlingS2[5, 2]", "15"), ("StirlingS2[4, 4]", "1"), ("StirlingS2[7, 3]", "301"),
    ("Stirling[5, 2]", "15"), ("Stirling[6, 3]", "90"), ("Stirling[0, 0]", "1"),
    ("DigitSum[12345]", "15"), ("DigitSum[0]", "0"), ("DigitSum[255, 2]", "8"),
    ("DigitCount[1223]", "{1, 2, 1, 0, 0, 0, 0, 0, 0, 0}"), ("DigitCount[1223, 10, 2]", "2"), ("DigitCount[5, 2, 1]", "2"),
    ("IntegerDigits[1234]", "{1, 2, 3, 4}"), ("IntegerDigits[-56]", "{5, 6}"), ("IntegerDigits[5, 2]", "{1, 0, 1}"),
    ("IntegerDigits[5, 2, 6]", "{0, 0, 0, 1, 0, 1}"), ("IntegerDigits[0]", "{0}"),
    ("FromDigits[{1, 2, 3}]", "123"), ("FromDigits[{1, 0, 1}, 2]", "5"), ("FromDigits[\"ff\", 16]", "255"),
    # MikoshiLang BaseConvert[n, from, to]: decimal digits of n read in base `from`
    ("BaseConvert[777, 8, 10]", "511"), ("BaseConvert[255, 10, 2]", "{1, 1, 1, 1, 1, 1, 1, 1}"),
    ("BaseConvert[17, 8, 10]", "15"), ("BaseConvert[19, 2, 10]", Raises(ValueError, "not valid")),
    ("CarmichaelLambda[8]", "2"), ("CarmichaelLambda[15]", "4"), ("CarmichaelLambda[561]", "80"), ("CarmichaelLambda[1]", "1"),
    ("Radical[12]", "6"), ("Radical[1]", "1"), ("Radical[72]", "6"),
    ("JordanTotient[10]", "4"), ("JordanTotient[6, 2]", "24"), ("JordanTotient[1, 3]", "1"), ("JordanTotient[12, 3]", "1456"),
    ("LegendreSymbol[2, 7]", "1"), ("LegendreSymbol[3, 7]", "-1"), ("LegendreSymbol[7, 7]", "0"), ("LegendreSymbol[2, 9]", Raises()),
    ("JacobiSymbol[2, 15]", "1"), ("JacobiSymbol[7, 15]", "-1"), ("JacobiSymbol[5, 15]", "0"), ("JacobiSymbol[2, 8]", Raises()),
    ("KroneckerSymbol[2, 8]", "0"), ("KroneckerSymbol[3, 8]", "-1"), ("KroneckerSymbol[-1, -1]", "-1"), ("KroneckerSymbol[5, 12]", "-1"),
    ("QuadraticResidues[7]", "{0, 1, 2, 4}"), ("QuadraticResidues[10]", "{0, 1, 4, 5, 6, 9}"), ("QuadraticResidues[2]", "{0, 1}"),
    ("PrimitiveRoot[7]", str(primitive_root(7))), ("PrimitiveRoot[18]", "5"), ("PrimitiveRoot[8]", "PrimitiveRoot[8]"),
    ("PrimitiveRootQ[3, 7]", True), ("PrimitiveRootQ[2, 7]", False), ("PrimitiveRootQ[5, 18]", True),
    ("MultiplicativeOrder[2, 7]", str(n_order(2, 7))), ("MultiplicativeOrder[10, 7]", "6"), ("MultiplicativeOrder[2, 4]", Raises()),
    ("PrimePowerQ[8]", True), ("PrimePowerQ[12]", False), ("PrimePowerQ[1]", False), ("PrimePowerQ[7]", True),
    # {p, q} pairs (MikoshiLang convention; Wolfram Convergents gives p/q)
    ("ContinuedFractionConvergents[{1, 2, 2, 2}]", "{{1, 1}, {3, 2}, {7, 5}, {17, 12}}"),
    ("ContinuedFractionConvergents[{3, 7, 15}]", "{{3, 1}, {22, 7}, {333, 106}}"),
    ("ContinuedFractionConvergents[{2}]", "{{2, 1}}"),
    ("DiophantineSolve[3 x + 5 y == 1, {x, y}]", Pred(lambda r: r.head == "List", "solutions")),
    ("DiophantineSolve[x^2 - y^2 == 0, {x, y}]", Pred(lambda r: r.head == "List", "solutions")),
    ("DiophantineSolve[2 x + 4 y == 1, {x, y}]", "{}"),
    ("RationalReconstruct[0.3333333333333333]", "1/3"), ("RationalReconstruct[0.75]", "3/4"), ("RationalReconstruct[2.5]", "5/2"),
    ("Ackermann[2, 3]", "9"), ("Ackermann[3, 3]", "61"), ("Ackermann[0, 5]", "6"), ("Ackermann[1, 0]", "2"),
    ("CollatzSequence[6]", "{6, 3, 10, 5, 16, 8, 4, 2, 1}"), ("CollatzSequence[1]", "{1}"), ("CollatzSequence[0]", Raises()),
    # number of terms including n and 1
    ("CollatzLength[6]", "9"), ("CollatzLength[1]", "1"), ("CollatzLength[27]", "112"), ("CollatzLength[-3]", Raises()),
]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_numbertheory(case):
    check(case)
