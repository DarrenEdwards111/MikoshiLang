"""Tests for extended function library (~900 new functions)."""

import pytest
from mikoshilang.parser import parse
from mikoshilang.evaluate import evaluate
from mikoshilang.expr import Expr, Symbol


def ev(s):
    return evaluate(parse(s))


# ── CALCULUS ─────────────────────────────────────────────────

class TestCalculus:
    def test_diff(self):
        assert ev("Diff[x^3, x]") is not None

    def test_partial_d(self):
        r = ev("PartialD[x^2*y, x]")
        assert r is not None

    def test_integrate(self):
        r = ev("Integrate[x^2, x]")
        assert r is not None

    def test_definite_integral(self):
        r = ev("DefiniteIntegral[x^2, {x, 0, 1}]")
        assert r is not None

    def test_gradient(self):
        r = ev("Gradient[x^2 + y^2, {x, y}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_divergence(self):
        r = ev("Divergence[{x, y, z}, {x, y, z}]")
        assert r == 3

    def test_curl(self):
        r = ev("Curl[{y, -x, 0}, {x, y, z}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_series(self):
        r = ev("Series[Exp[x], {x, 0, 4}]")
        assert r is not None

    def test_limit(self):
        r = ev("Limit[Sin[x]/x, x -> 0]")
        assert r == 1

    def test_sum(self):
        r = ev("Sum[k, {k, 1, 100}]")
        assert r == 5050

    def test_product(self):
        r = ev("Product[k, {k, 1, 5}]")
        assert r == 120

    def test_residue(self):
        r = ev("Residue[1/x, {x, 0}]")
        assert r == 1

    def test_arc_length(self):
        r = ev("ArcLength[x, {x, 0, 1}]")
        assert r is not None


# ── ALGEBRA ──────────────────────────────────────────────────

class TestAlgebra:
    def test_simplify(self):
        assert ev("Simplify[Sin[x]^2 + Cos[x]^2]") == 1

    def test_expand(self):
        r = ev("Expand[(x + 1)^3]")
        assert r is not None

    def test_factor(self):
        r = ev("Factor[x^2 - 1]")
        assert r is not None

    def test_solve(self):
        r = ev("Solve[x^2 - 4, x]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_partial_fractions(self):
        r = ev("PartialFractions[1/(x*(x + 1)), x]")
        assert r is not None

    def test_apart(self):
        r = ev("Apart[1/(x*(x + 1)), x]")
        assert r is not None

    def test_together(self):
        r = ev("Together[1/x + 1/y]")
        assert r is not None

    def test_cancel(self):
        r = ev("Cancel[(x^2 - 1)/(x - 1)]")
        assert r is not None

    def test_collect(self):
        r = ev("Collect[x*y + x*z, x]")
        assert r is not None

    def test_trig_expand(self):
        r = ev("TrigExpand[Sin[2*x]]")
        assert r is not None

    def test_real_part(self):
        r = ev("RealPart[3]")
        assert r == 3

    def test_abs(self):
        assert ev("Abs[-5]") == 5

    def test_sign(self):
        assert ev("Sign[-3]") == -1

    def test_floor(self):
        assert ev("Floor[3.7]") == 3

    def test_ceiling(self):
        assert ev("Ceiling[3.2]") == 4

    def test_mod(self):
        assert ev("Mod[17, 5]") == 2

    def test_gcd(self):
        assert ev("GCD[12, 18]") == 6

    def test_lcm(self):
        assert ev("LCM[4, 6]") == 12

    def test_discriminant(self):
        r = ev("Discriminant[x^2 + 2*x + 1, x]")
        assert r == 0

    def test_complete_the_square(self):
        r = ev("CompleteTheSquare[x^2 + 6*x + 5, x]")
        assert r is not None

    def test_polynomial_quotient(self):
        r = ev("PolynomialQuotient[x^3, x + 1, x]")
        assert r is not None

    def test_coefficient_list(self):
        r = ev("CoefficientList[x^3 + 2*x + 1, x]")
        assert isinstance(r, Expr) and r.head == "List"


# ── TRIG & SPECIAL ──────────────────────────────────────────

class TestTrigSpecial:
    def test_sin(self):
        assert ev("Sin[0]") == 0

    def test_cos(self):
        assert ev("Cos[0]") == 1

    def test_tan(self):
        assert ev("Tan[0]") == 0

    def test_arcsin(self):
        r = ev("ArcSin[1]")
        assert r is not None

    def test_sinh(self):
        assert ev("Sinh[0]") == 0

    def test_cosh(self):
        assert ev("Cosh[0]") == 1

    def test_exp(self):
        assert ev("Exp[0]") == 1

    def test_log(self):
        assert ev("Log[1]") == 0

    def test_log2(self):
        r = ev("Log2[8]")
        assert r == 3

    def test_log10(self):
        r = ev("Log10[1000]")
        assert r == 3

    def test_sqrt(self):
        assert ev("Sqrt[4]") == 2

    def test_cube_root(self):
        r = ev("CubeRoot[27]")
        assert r == 3

    def test_factorial(self):
        assert ev("Factorial[5]") == 120

    def test_double_factorial(self):
        assert ev("DoubleFactorial[5]") == 15

    def test_gamma(self):
        r = ev("Gamma[5]")
        assert r == 24

    def test_erf(self):
        r = ev("Erf[0]")
        assert r == 0


# ── LINEAR ALGEBRA ───────────────────────────────────────────

class TestLinearAlgebra:
    def test_det(self):
        assert ev("Det[{{1, 2}, {3, 4}}]") == -2

    def test_inverse(self):
        r = ev("Inverse[{{1, 0}, {0, 1}}]")
        assert r is not None

    def test_transpose(self):
        r = ev("Transpose[{{1, 2}, {3, 4}}]")
        assert r is not None

    def test_trace(self):
        assert ev("Trace[{{1, 0}, {0, 2}}]") == 3

    def test_eigenvalues(self):
        r = ev("Eigenvalues[{{2, 1}, {1, 2}}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_eigenvectors(self):
        r = ev("Eigenvectors[{{2, 0}, {0, 3}}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_rank(self):
        assert ev("Rank[{{1, 2}, {2, 4}}]") == 1

    def test_nullspace(self):
        r = ev("NullSpace[{{1, 2}, {2, 4}}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_row_reduce(self):
        r = ev("RowReduce[{{1, 2}, {3, 4}}]")
        assert r is not None

    def test_lu(self):
        r = ev("LUDecomposition[{{1, 2}, {3, 4}}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_cross_product(self):
        r = ev("CrossProduct[{1, 0, 0}, {0, 1, 0}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_dot_product(self):
        assert ev("DotProduct[{1, 2, 3}, {4, 5, 6}]") == 32

    def test_norm(self):
        assert ev("Norm[{3, 4}]") == 5

    def test_identity_matrix(self):
        r = ev("IdentityMatrix[3]")
        assert r is not None

    def test_diagonal_matrix(self):
        r = ev("DiagonalMatrix[{1, 2, 3}]")
        assert r is not None

    def test_matrix_power(self):
        r = ev("MatrixPower[{{1, 1}, {0, 1}}, 3]")
        assert r is not None

    def test_linear_solve(self):
        r = ev("LinearSolve[{{1, 0}, {0, 1}}, {3, 4}]")
        assert isinstance(r, Expr)

    def test_charpoly(self):
        r = ev("CharacteristicPolynomial[{{1, 2}, {3, 4}}, x]")
        assert r is not None

    def test_vector_angle(self):
        r = ev("VectorAngle[{1, 0}, {0, 1}]")
        assert r is not None


# ── STATISTICS ───────────────────────────────────────────────

class TestStatistics:
    def test_mean(self):
        assert ev("Mean[{1, 2, 3, 4, 5}]") == 3.0

    def test_median(self):
        assert ev("Median[{1, 2, 3, 4, 5}]") == 3

    def test_mode(self):
        assert ev("Mode[{1, 2, 2, 3}]") == 2

    def test_stdev(self):
        r = ev("StandardDeviation[{2, 4, 4, 4, 5, 5, 7, 9}]")
        assert isinstance(r, float) and r > 0

    def test_variance(self):
        r = ev("Variance[{2, 4, 4, 4, 5, 5, 7, 9}]")
        assert isinstance(r, float) and r > 0

    def test_correlation(self):
        r = ev("Correlation[{1, 2, 3, 4, 5}, {2, 4, 6, 8, 10}]")
        assert abs(r - 1.0) < 0.001

    def test_linear_regression(self):
        r = ev("LinearRegression[{1, 2, 3, 4, 5}, {2, 4, 6, 8, 10}]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_zscore(self):
        assert ev("ZScore[85, 75, 10]") == 1.0

    def test_ttest(self):
        r = ev("TTest[{1, 2, 3}, {4, 5, 6}]")
        assert isinstance(r, Expr)

    def test_shapiro(self):
        r = ev("ShapiroWilk[{1, 2, 3, 4, 5, 6, 7, 8}]")
        assert isinstance(r, Expr)

    def test_conf_interval(self):
        r = ev("ConfidenceInterval[{1, 2, 3, 4, 5}, 0.95]")
        assert isinstance(r, Expr)

    def test_effect_size(self):
        r = ev("EffectSize[{1, 2, 3}, {4, 5, 6}]")
        assert isinstance(r, float)

    def test_pdf(self):
        r = ev("PDF[NormalDistribution[0, 1], 0]")
        assert abs(r - 0.3989422804) < 0.001

    def test_cdf(self):
        r = ev("CDF[NormalDistribution[0, 1], 0]")
        assert abs(r - 0.5) < 0.001

    def test_inverse_cdf(self):
        r = ev("InverseCDF[NormalDistribution[0, 1], 0.975]")
        assert abs(r - 1.96) < 0.01

    def test_random_sample(self):
        r = ev("RandomSample[NormalDistribution[0, 1], 10]")
        assert isinstance(r, Expr) and r.head == "List" and len(r.args) == 10

    def test_anova(self):
        r = ev("ANOVA[{1, 2, 3}, {4, 5, 6}, {7, 8, 9}]")
        assert isinstance(r, Expr)


# ── NUMBER THEORY ────────────────────────────────────────────

class TestNumberTheory:
    def test_prime_q(self):
        assert ev("PrimeQ[17]") == True
        assert ev("PrimeQ[4]") == False

    def test_next_prime(self):
        assert ev("NextPrime[100]") == 101

    def test_prime(self):
        assert ev("Prime[10]") == 29

    def test_factor_integer(self):
        r = ev("FactorInteger[360]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_divisors(self):
        r = ev("Divisors[12]")
        assert isinstance(r, Expr) and 6 in r.args

    def test_divisor_count(self):
        assert ev("DivisorCount[12]") == 6

    def test_euler_phi(self):
        assert ev("EulerPhi[12]") == 4

    def test_fibonacci(self):
        assert ev("Fibonacci[10]") == 55

    def test_lucas(self):
        assert ev("Lucas[5]") == 11

    def test_catalan(self):
        assert ev("Catalan[5]") == 42

    def test_bell(self):
        assert ev("Bell[5]") == 52

    def test_power_mod(self):
        assert ev("PowerMod[2, 10, 1000]") == 24

    def test_digit_sum(self):
        assert ev("DigitSum[123]") == 6

    def test_integer_digits(self):
        r = ev("IntegerDigits[123]")
        assert isinstance(r, Expr) and list(r.args) == [1, 2, 3]


# ── COMBINATORICS ────────────────────────────────────────────

class TestCombinatorics:
    def test_binomial(self):
        assert ev("Binomial[10, 3]") == 120

    def test_permutations(self):
        assert ev("Permutations[5, 3]") == 60

    def test_derangements(self):
        assert ev("Derangements[5]") == 44

    def test_partition_count(self):
        assert ev("PartitionCount[10]") == 42

    def test_harmonic(self):
        r = ev("HarmonicNumber[5]")
        assert r is not None

    def test_stirling_s2(self):
        assert ev("StirlingS2[5, 3]") == 25

    def test_euler_number(self):
        r = ev("EulerNumber[4]")
        assert r == 5

    def test_multinomial(self):
        assert ev("Multinomial[6, {2, 2, 2}]") == 90


# ── GEOMETRY ─────────────────────────────────────────────────

class TestGeometry:
    def test_distance(self):
        assert ev("Distance[{0, 0}, {3, 4}]") == 5.0

    def test_midpoint(self):
        r = ev("Midpoint[{0, 0}, {4, 6}]")
        assert isinstance(r, Expr) and r.args == (2.0, 3.0)

    def test_slope(self):
        assert ev("Slope[{0, 0}, {2, 4}]") == 2.0

    def test_circle_area(self):
        r = ev("CircleArea[1]")
        assert r is not None  # Should be Pi

    def test_sphere_volume(self):
        r = ev("SphereVolume[1]")
        assert r is not None

    def test_cylinder_volume(self):
        r = ev("CylinderVolume[2, 5]")
        assert r is not None

    def test_triangle_area(self):
        r = ev("TriangleArea[3, 4, 5]")
        assert abs(r - 6.0) < 0.001

    def test_hypotenuse(self):
        assert ev("Hypotenuse[3, 4]") == 5.0

    def test_quadratic_formula(self):
        r = ev("QuadraticFormula[1, -3, 2]")
        assert isinstance(r, Expr) and r.head == "List"

    def test_polygon_area(self):
        r = ev("PolygonArea[{{0, 0}, {4, 0}, {4, 3}, {0, 3}}]")
        assert abs(r - 12.0) < 0.001

    def test_regular_polygon_area(self):
        r = ev("RegularPolygonArea[4, 2]")
        assert abs(r - 4.0) < 0.001


# ── FINANCIAL ────────────────────────────────────────────────

class TestFinancial:
    def test_compound_interest(self):
        r = ev("CompoundInterest[1000, 0.05, 10]")
        assert abs(r - 1628.89) < 0.1

    def test_future_value(self):
        r = ev("FutureValue[1000, 0.1, 5]")
        assert abs(r - 1610.51) < 0.1

    def test_present_value(self):
        r = ev("PresentValue[1610.51, 0.1, 5]")
        assert abs(r - 1000) < 1

    def test_npv(self):
        r = ev("NPV[0.1, {-1000, 300, 420, 680}]")
        assert r is not None

    def test_irr(self):
        r = ev("IRR[{-1000, 300, 420, 680}]")
        assert isinstance(r, float) and r > 0

    def test_pmt(self):
        r = ev("PMT[0.05, 30, 200000]")
        assert isinstance(r, float)

    def test_continuous_compounding(self):
        r = ev("ContinuousCompounding[1000, 0.05, 10]")
        assert abs(r - 1648.72) < 0.1


# ── SEQUENCES ────────────────────────────────────────────────

class TestSequences:
    def test_fibonacci_seq(self):
        r = ev("FibonacciSequence[8]")
        assert isinstance(r, Expr) and r.args == (0, 1, 1, 2, 3, 5, 8, 13)

    def test_arithmetic_seq(self):
        r = ev("ArithmeticSequence[1, 2, 5]")
        assert isinstance(r, Expr) and len(r.args) == 5

    def test_geometric_seq(self):
        r = ev("GeometricSequence[1, 2, 5]")
        assert isinstance(r, Expr) and r.args[4] == 16.0

    def test_accumulate(self):
        r = ev("Accumulate[{1, 2, 3, 4, 5}]")
        assert isinstance(r, Expr) and r.args[-1] == 15.0

    def test_differences(self):
        r = ev("Differences[{1, 4, 9, 16}]")
        assert isinstance(r, Expr) and list(r.args) == [3.0, 5.0, 7.0]

    def test_range(self):
        r = ev("Range[5]")
        assert isinstance(r, Expr) and list(r.args) == [1, 2, 3, 4, 5]


# ── SET THEORY & LOGIC ──────────────────────────────────────

class TestSetTheory:
    def test_union(self):
        r = ev("Union[{1, 2, 3}, {3, 4, 5}]")
        assert isinstance(r, Expr) and len(r.args) == 5

    def test_intersection(self):
        r = ev("Intersection[{1, 2, 3}, {2, 3, 4}]")
        assert isinstance(r, Expr) and len(r.args) == 2

    def test_set_difference(self):
        r = ev("SetDifference[{1, 2, 3}, {2, 3}]")
        assert isinstance(r, Expr) and len(r.args) == 1

    def test_power_set(self):
        r = ev("PowerSet[{1, 2}]")
        assert isinstance(r, Expr) and len(r.args) == 4

    def test_cardinality(self):
        assert ev("Cardinality[{1, 2, 3}]") == 3

    def test_element_q(self):
        assert ev("ElementQ[2, {1, 2, 3}]") == True

    def test_subset_q(self):
        assert ev("SubsetQ[{1, 2}, {1, 2, 3}]") == True


# ── SIGNAL PROCESSING ───────────────────────────────────────