import math

import pytest

from _harness import Raises, Pred, check, ids

E = math.e
NUMERICS = [
    ("Bisection[x^2 - 2, x, 0, 2]", 2 ** 0.5, 1e-6), ("Bisection[Cos[x] - x, x, 0, 1, 10^-12]", 0.7390851332151607, 1e-11),
    ("Bisection[x - 1, x, 0, 2]", 1.0), ("Bisection[x^2 + 1, x, 0, 2]", Raises(ValueError, "opposite signs")),
    ("Bisection[x, x, -1, 1, 0]", Raises(ValueError, "tolerance")),
    ("NewtonRaphson[x^2 - 2, x, 1]", 2 ** 0.5), ("NewtonRaphson[Cos[x] - x, x, 1]", 0.7390851332151607),
    ("NewtonRaphson[x^3 - 8, x, 5]", 2.0), ("NewtonRaphson[x^2 - 1, x, 0]", Raises(ValueError, "derivative vanished")),
    ("NewtonRaphson[x^2 + 1, x, 1/2, 10^-10, 20]", Raises(ValueError)),
    ("SecantMethod[x^2 - 2, x, 1, 2]", 2 ** 0.5), ("SecantMethod[Exp[x] - 2, x, 0, 1]", math.log(2)),
    ("SecantMethod[x - 3, x, 0, 1]", 3.0), ("SecantMethod[x^2 - 1, x, -1/2, 1/2]", Raises(ValueError, "horizontal")),
    ("EulerMethod[y, {x, y}, 0, 1, 1, 10]", 1.1 ** 10), ("EulerMethod[1, {x, y}, 0, 0, 2, 5]", 2.0),
    ("EulerMethod[x, {x, y}, 0, 0, 1, 2]", 0.25), ("EulerMethod[y, {x, y}, 0, 1, 1, 0]", Raises()),
    ("RungeKutta4[y, {x, y}, 0, 1, 1, 10]", 2.718279744135166, 1e-12), ("RungeKutta4[y, {x, y}, 0, 1, 1, 1000]", E, 1e-11),
    ("RungeKutta4[x, {x, y}, 0, 0, 2, 4]", 2.0), ("RungeKutta4[3 x^2, {x, y}, 0, 0, 1, 1]", 1.0),
    ("RungeKutta4[y, {x, y}, 0, 1, 1, -3]", Raises()),
    ("TrapezoidalRule[x^2, x, 0, 1, 4]", 0.34375), ("TrapezoidalRule[x, x, 0, 2, 1]", 2.0),
    ("TrapezoidalRule[Sin[x], x, 0, Pi]", 1.9998355038874436, 1e-12), ("TrapezoidalRule[x, x, 0, 1, 0]", Raises()),
    ("SimpsonsRule[x^2, x, 0, 1, 4]", 1 / 3), ("SimpsonsRule[x^3, x, 0, 2, 2]", 4.0),
    ("SimpsonsRule[Sin[x], x, 0, Pi]", 2.0000000108245044, 1e-12), ("SimpsonsRule[x^2, x, 0, 1, 3]", Raises(ValueError, "even")),
    ("GaussQuadrature[x^2, x, 0, 1, 3]", 1 / 3), ("GaussQuadrature[x^5, x, -1, 1, 3]", 0.0, 1e-15),
    # 2-point Gauss is exact to degree 3 but not 4: integral of x^4 over [-1,1] with n=2 is 2/9
    ("GaussQuadrature[x^4, x, -1, 1, 2]", 2 / 9), ("GaussQuadrature[Exp[x], x, 0, 1]", E - 1),
    ("GaussQuadrature[x, x, 0, 1, 0]", Raises()),
    ("MonteCarloIntegration[x^2, x, 0, 1, 20000]", Pred(lambda r: abs(float(r) - 1 / 3) < 0.02, "≈ 1/3")),
    ("MonteCarloIntegration[1, x, 0, 3, 100]", 3.0), ("MonteCarloIntegration[x, x, 0, 2, 20000]", Pred(lambda r: abs(float(r) - 2) < 0.05, "≈ 2")),
    ("MonteCarloIntegration[x, x, 0, 1, 0]", Raises()),
]

TRANSFORMS = [
    ("FFT[{1, 2, 3, 4}]", "{10, -2 + 2 I, -2, -2 - 2 I}"), ("FFT[{1, 0, 0}]", "{1, 1, 1}"),
    ("FFT[{0, 1, 0, 0}]", "{1, -I, -1, I}"), ("FFT[{1., 0, 0}]", [1.0, 1.0, 1.0]), ("FFT[{}]", Raises()),
    ("DFT[{1, 1}]", "{2, 0}"), ("DFT[{1, 2}]", "{3, -1}"), ("DFT[{1, 0, 0, 0}]", "{1, 1, 1, 1}"),
    ("FourierDFT[{1, 2, 3, 4}]", "{10, -2 + 2 I, -2, -2 - 2 I}"), ("FourierDFT[{5}]", "{5}"), ("FourierDFT[{1, -1}]", "{0, 2}"),
    ("DiscreteFourierTransform[{1, 2}]", "{3, -1}"), ("DiscreteFourierTransform[{2, 2, 2}]", "{6, 0, 0}"), ("DiscreteFourierTransform[{0, 0}]", "{0, 0}"),
    ("InverseFFT[{10, -2 + 2 I, -2, -2 - 2 I}]", "{1, 2, 3, 4}"), ("InverseFFT[{1, -I, -1, I}]", "{0, 1, 0, 0}"),
    ("InverseFFT[{0, 2}]", "{1, -1}"), ("InverseFFT[{1, 0}]", "{1/2, 1/2}"),
    ("DCT[{1, 2, 3}]", "{12, -2 Sqrt[3], 0}"), ("DCT[{1, 1}]", "{4, 0}"), ("DCT[{1.0, 2.0, 3.0}]", [12.0, -3.4641016151377544, 0.0]),
    ("DCT[{x}]", "DCT[{x}]"),
    ("DiscreteCosineTransform[{1, 2, 3}]", "{12, -2 Sqrt[3], 0}"), ("DiscreteCosineTransform[{2}]", "{4}"),
    ("DiscreteCosineTransform[{1, -1}]", "{0, 2 Sqrt[2]}"),
    ("DiscreteSineTransform[{1, 2, 3}]", "{8, -2 Sqrt[3], 4}"), ("DiscreteSineTransform[{2}]", "{4}"),
    ("DiscreteSineTransform[{1, -1}]", "{0, 4}"),
    ("DST[{1, 2, 3}]", "{8, -2 Sqrt[3], 4}"), ("DST[{1}]", "{2}"), ("DST[{1., 1.}]", [2 * 2 ** 0.5, 0.0]),
    ("HilbertTransform[{1, 0, -1, 0}]", [0.0, 1.0, 0.0, -1.0]),
    ("HilbertTransform[{1, 2, 3, 4, 5}]", [1.70130162, -1.37638192, -0.64983939, -1.37638192, 1.70130162], 1e-7),
    ("HilbertTransform[{1, 1, 1, 1}]", [0.0, 0.0, 0.0, 0.0]),
    ("Convolve[{1, 2}, {1, 3}]", "{1, 5, 6}"), ("Convolve[{1}, {4, 5}]", "{4, 5}"), ("Convolve[{1, 1, 1}, {1, -1}]", "{1, 0, 0, -1}"),
    ("Convolution[{1, 2, 3}, {0, 1}]", "{0, 1, 2, 3}"), ("Convolution[{x, y}, {1, 1}]", "Convolution[{x, y}, {1, 1}]"), ("Convolution[{1/2}, {1/3}]", "{1/6}"),
    ("WaveletTransform[{1, 2, 3, 4}]", "{{5}, {-2}, {-Sqrt[2]/2, -Sqrt[2]/2}}"), ("WaveletTransform[{1, 1}]", "{{Sqrt[2]}, {0}}"),
    ("WaveletTransform[{1, 2, 3}]", Raises(ValueError, "power of 2")), ("WaveletTransform[{1, 2}, \"db4\"]", Raises()),
    ("FourierTransform[Exp[-t^2/2], t, w]", "E^(-w^2/2)"),
    ("FourierTransform[1/(1 + t^2), t, w]", "Sqrt[Pi/2] E^(-Abs[w])"),
    ("FourierTransform[Exp[-Abs[t]], t, w]", "Sqrt[2/Pi]/(1 + w^2)"),
    ("FourierTransform[HeavisideTheta[t] Exp[-t], t, w]", "I/(Sqrt[2 Pi] (I + w))"),
    ("FourierTransform[t Exp[-t^2], t, w]", "I w E^(-w^2/4)/(2 Sqrt[2])"),
    ("FourierTransform[Sin[t], t, w]", Raises(NotImplementedError, "distributional")),
    ("InverseFourierTransform[Exp[-w^2/2], w, t]", "E^(-t^2/2)"),
    ("InverseFourierTransform[1/(1 + w^2), w, t]", "Sqrt[Pi/2] E^(-Abs[t])"),
    ("InverseFourierTransform[Sqrt[2/Pi]/(1 + w^2), w, t]", "E^(-Abs[t])"),
    ("LaplaceTransform[Sin[t], t, s]", "1/(s^2 + 1)"), ("LaplaceTransform[1, t, s]", "1/s"),
    ("LaplaceTransform[t^2, t, s]", "2/s^3"), ("LaplaceTransform[Exp[a t], t, s]", "1/(s - a)"),
    ("InverseLaplaceTransform[1/(s^2 + 1), s, t]", "Sin[t]"), ("InverseLaplaceTransform[1/s^2, s, t]", "t"),
    ("InverseLaplaceTransform[1/(s - 2), s, t]", "E^(2 t)"),
    ("ZTransform[1, n, z]", "z/(z - 1)"), ("ZTransform[a^n, n, z]", "z/(z - a)"), ("ZTransform[n, n, z]", "z/(z - 1)^2"),
    ("InverseZTransform[z/(z - 1), z, n]", "1"),
]

CASES = NUMERICS + TRANSFORMS


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_numerics_transforms(case):
    check(case)
