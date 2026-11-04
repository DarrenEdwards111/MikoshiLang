"""Direct Python science API regressions; no Session/Wolfram normalization."""
import pytest
import mikoshilang as api
from mikoshilang import Quantity, UnitConvert
from mikoshilang.expr import Expr, Symbol

COVERED_EXPORTS = {
    "Quantity", "UnitConvert", "Element", "AtomicMass", "ElectronConfiguration",
    "MolecularMass", "BalanceEquation", "FourierTransform", "InverseFourierTransform",
    "DFT", "IDFT", "LowPassFilter", "HighPassFilter", "BandPassFilter", "Convolve",
    "HammingWindow", "HanningWindow", "BlackmanWindow", "Spectrogram",
}


@pytest.mark.parametrize("formula", ["", "garbage", "H2O!", "2H2O", "H0", "H-2", "Ca(OH", "Ca)OH(", "CuSO4.5H2O", "Na+", "()", "Xx2"])
def test_molecularmass_rejects_invalid_formula(formula):
    with pytest.raises(ValueError):
        api.MolecularMass(formula)


@pytest.mark.parametrize("equation,expected", [
    ("C2H6 + O2 -> CO2 + H2O", "2C2H6 + 7O2 -> 4CO2 + 6H2O"),
    ("Fe + O2 -> Fe2O3", "4Fe + 3O2 -> 2Fe2O3"),
])
def test_balanceequation_exact_positive_coefficients(equation, expected):
    assert api.BalanceEquation(equation) == expected


@pytest.mark.parametrize("equation", ["H2 -> O2", "H2 + O2 -> H2", "H2 + O2 -> H2O + H2O2", "Xx -> Xx", "H2 + -> H2", "2H2 -> H2", "H2O! -> H2O"])
def test_balanceequation_rejects_invalid_or_ambiguous(equation):
    with pytest.raises(ValueError):
        api.BalanceEquation(equation)


def test_quantity_generated_dimensions_and_dimensionless():
    area = Quantity(2, "m") * Quantity(3, "m")
    assert area.value == 6
    assert area._si_dims() == {"m": 2}
    ratio = Quantity(2, "m") / Quantity(50, "cm")
    assert ratio.value == 4
    assert ratio._si_dims() == {}


@pytest.mark.parametrize("operation", [lambda q: q + Quantity(1, "C"), lambda q: q - Quantity(1, "K"), lambda q: q * 2, lambda q: 2 * q, lambda q: q / 2, lambda q: q * Quantity(2, "m"), lambda q: q / Quantity(2, "s")])
def test_quantity_offset_arithmetic_clear_error(operation):
    with pytest.raises(ValueError, match="offset"):
        operation(Quantity(20, "C"))


def test_convolve_preserves_complex():
    result = api.Convolve(Expr("List", 1j, 2), Expr("List", 2, 1j))
    assert result.args == pytest.approx((2j, 3, 2j))


@pytest.mark.parametrize("name,args", [("LowPassFilter", (0.2,)), ("HighPassFilter", (0.2,)), ("BandPassFilter", (0.1, 0.3))])
def test_filters_preserve_complex_linearity(name, args):
    pytest.importorskip("scipy.signal")
    import numpy as np
    data = np.sin(np.arange(96) * 0.7)
    function = getattr(api, name)
    real = function(Expr("List", *data), *args, order=2)
    imaginary = function(Expr("List", *(1j * data)), *args, order=2)
    assert imaginary.args == pytest.approx([1j * value for value in real.args], abs=1e-12)


def test_spectrogram_raw_power_and_complex_input(monkeypatch):
    pytest.importorskip("scipy.signal")
    import sys
    import numpy as np
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    data = np.exp(2j * np.pi * np.arange(512) / 16)
    result = api.Spectrogram(Expr("List", *data), sample_rate=256)
    raw = dict(rule.args for rule in result.args)
    frequencies = np.asarray(raw["frequencies"].args)
    power = np.asarray([row.args for row in raw["power"].args])
    assert power.shape == (256, len(raw["times"].args))
    assert frequencies[np.argmax(power[:, 0])] == pytest.approx(16)
    assert np.sum(power[:, 0]) == pytest.approx(1, rel=0.02)


@pytest.mark.parametrize("name", ["HammingWindow", "HanningWindow", "BlackmanWindow"])
@pytest.mark.parametrize("length", [-1, 2.5, True])
def test_windows_reject_invalid_lengths(name, length):
    with pytest.raises((TypeError, ValueError)):
        getattr(api, name)(length)


@pytest.mark.parametrize("name,args,options", [
    (name, args, options)
    for name, args in [("LowPassFilter", (0.2,)), ("HighPassFilter", (0.2,)), ("BandPassFilter", (0.1, 0.3))]
    for options in [{"order": 0}, {"order": True}, {"sample_rate": float("inf")}, {"sample_rate": -1}]
] + [
    (name, (cutoff,), {"order": 2})
    for name in ["LowPassFilter", "HighPassFilter"]
    for cutoff in [float("nan"), -float("inf"), float("inf"), -0.1, 0, 0.5, 0.6]
] + [
    ("BandPassFilter", (low, high), {"order": 2})
    for low, high in [
        (float("nan"), 0.3), (0.1, float("nan")), (float("nan"), float("nan")),
        (-float("inf"), 0.3), (float("inf"), 0.3),
        (0.1, -float("inf")), (0.1, float("inf")),
        (-0.1, 0.3), (0, 0.3), (0.1, 0), (0.1, 0.5), (0.1, 0.6),
        (0.2, 0.2), (0.3, 0.1),
    ]
])
def test_filter_parameter_domains(name, args, options):
    pytest.importorskip("scipy.signal")
    match = None
    if options == {"order": 2}:
        match = "low.*high" if name == "BandPassFilter" else "cutoff"
    with pytest.raises((ValueError, TypeError), match=match):
        getattr(api, name)(Expr("List", *([1] * 96)), *args, **options)


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_spectrogram_invalid_rate(rate):
    pytest.importorskip("scipy.signal")
    with pytest.raises(ValueError):
        api.Spectrogram(Expr("List", *([1] * 256)), sample_rate=rate)


def test_element_known_data_and_lookup_aliases():
    expected = {"name": "Oxygen", "number": 8, "mass": 15.999, "symbol": "O", "electronConfiguration": "[He] 2s2 2p4", "electronegativity": 3.44, "category": "nonmetal"}
    for key in [8, "O", "oxygen", "OXYGEN"]:
        assert dict(rule.args for rule in api.Element(key).args) == expected
    assert dict(rule.args for rule in api.Element(118).args)["symbol"] == "Og"


@pytest.mark.parametrize("name", ["Element", "AtomicMass", "ElectronConfiguration"])
@pytest.mark.parametrize("key", [0, 119, "Xx", "", 1.5])
def test_element_exports_invalid_keys(name, key):
    with pytest.raises(ValueError):
        getattr(api, name)(key)


def test_atomicmass_and_electronconfiguration_known_values():
    assert api.AtomicMass("Hydrogen") == 1.008
    assert api.AtomicMass(79) == 196.97
    assert api.ElectronConfiguration("Fe") == "[Ar] 3d6 4s2"
    assert api.ElectronConfiguration(2) == "1s2"
    assert api.ElectronConfiguration(118) == "[Rn] 5f14 6d10 7s2 7p6"


@pytest.mark.parametrize("formula,expected", [("H2O", 18.015), ("Ca(OH)2", 74.092), ("Al2(SO4)3", 342.132), ("((H)2O)2", 36.03), ("CH3COOH", 60.052)])
def test_molecularmass_known_formulas(formula, expected):
    assert api.MolecularMass(formula) == pytest.approx(expected, abs=0.001)


@pytest.mark.parametrize("arrow", ["->", "=", "→"])
def test_balanceequation_arrow_aliases(arrow):
    assert api.BalanceEquation("H2 + O2 " + arrow + " H2O") == "2H2 + O2 -> 2H2O"


def test_quantity_known_arithmetic_and_errors():
    assert (Quantity(1, "m") + Quantity(20, "cm")).value == pytest.approx(1.2)
    assert (Quantity(1, "m") - Quantity(20, "cm")).value == pytest.approx(0.8)
    assert (Quantity(4, "m") / 2).value == 2
    assert (2 * Quantity(4, "m")).value == 8
    assert Quantity(100, "cm") == Quantity(1, "m")
    assert Quantity(1, "m") != Quantity(1, "s")
    for operation in [lambda: Quantity(1, "unknown"), lambda: UnitConvert(Quantity(1, "m"), "s"), lambda: Quantity(1, "m") + Quantity(1, "s")]:
        with pytest.raises(ValueError):
            operation()
    with pytest.raises(ZeroDivisionError):
        Quantity(1, "m") / 0


@pytest.mark.parametrize("value,source,target,expected", [(0, "C", "K", 273.15), (32, "F", "C", 0), (100, "C", "F", 212), (-40, "F", "C", -40), (72, "km/h", "m/s", 20), (1, "eV", "J", 1.602176634e-19)])
def test_unitconvert_known_values(value, source, target, expected):
    result = UnitConvert(Quantity(value, source), target)
    assert result.unit_name == target
    assert result.value == pytest.approx(expected, rel=1e-12, abs=1e-9 if expected == 0 else 0)


@pytest.mark.parametrize("real_variables", [False, True])
def test_fouriertransform_gaussian_and_inverse(real_variables):
    import sympy as sp
    from mikoshilang.math_ops import to_sympy
    t, w = sp.symbols("t w", real=True) if real_variables else sp.symbols("t w")
    gaussian = sp.exp(-sp.pi * t**2)
    transformed = api.FourierTransform(gaussian, t if real_variables else Symbol("t"), w if real_variables else "w")
    forward = to_sympy(transformed).subs(sp.Symbol("w"), w)
    assert sp.simplify(forward - sp.exp(-sp.pi * w**2)) == 0
    inverse = api.InverseFourierTransform(sp.exp(-sp.pi * w**2), w if real_variables else "w", t if real_variables else Symbol("t"))
    restored = to_sympy(inverse).subs(sp.Symbol("t"), t)
    assert sp.simplify(restored - gaussian) == 0
    assert api.FourierTransform(0, "t", "w") == 0
    assert api.InverseFourierTransform(0, "w", "t") == 0
    with pytest.raises(TypeError):
        api.FourierTransform("not symbolic source", "t", "w")


def test_dft_idft_numpy_convention_and_complex_roundtrip():
    assert api.DFT(Expr("List", 1, 2, 3, 4)).args == pytest.approx((10, -2+2j, -2, -2-2j))
    assert api.IDFT(Expr("List", 4, 0, 0, 0)).args == (1, 1, 1, 1)
    data = Expr("List", 1+2j, -3j, 4, 1e-20j)
    assert api.IDFT(api.DFT(data)).args == pytest.approx(data.args, abs=1e-14)
    assert api.DFT(Expr("List", 1e-20j)).args == pytest.approx((1e-20j,), rel=1e-12, abs=0)
    for function in [api.DFT, api.IDFT]:
        with pytest.raises(ValueError):
            function(Expr("List"))
        with pytest.raises(TypeError):
            function([1, 2])


@pytest.mark.parametrize("name,expected", [("HammingWindow", (0.08, 0.54, 1, 0.54, 0.08)), ("HanningWindow", (0, 0.5, 1, 0.5, 0)), ("BlackmanWindow", (0, 0.34, 1, 0.34, 0))])
def test_windows_known_values_and_boundaries(name, expected):
    function = getattr(api, name)
    assert function(5).args == pytest.approx(expected, abs=1e-14)
    assert function(0).args == ()
    assert function(1).args == (1,)


@pytest.mark.parametrize("name,args,expected", [("LowPassFilter", (0.2,), 1), ("HighPassFilter", (0.2,), 0), ("BandPassFilter", (0.1, 0.3), 0)])
def test_filters_constant_response_and_boundaries(name, args, expected):
    pytest.importorskip("scipy.signal")
    function = getattr(api, name)
    assert function(Expr("List", *([1] * 96)), *args, order=2).args == pytest.approx([expected] * 96, abs=1e-12)
    with pytest.raises(ValueError):
        function(Expr("List", 1, 2), *args)
    bad_args = (0, 0.5) if name == "BandPassFilter" else (0.5,)
    with pytest.raises(ValueError):
        function(Expr("List", *([1] * 64)), *bad_args)
    if name == "BandPassFilter":
        with pytest.raises(ValueError):
            function(Expr("List", *([1] * 64)), 0.3, 0.1)


def test_convolve_known_identity_and_empty():
    assert api.Convolve(Expr("List", 1, 2, 3), Expr("List", 1, -1)).args == (1, 1, 1, -3)
    assert api.Convolve(Expr("List", 2j, 3), Expr("List", 1)).args == (2j, 3)
    with pytest.raises(ValueError):
        api.Convolve(Expr("List"), Expr("List", 1))


def test_bounded_complex_transform_and_convolution_properties():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st
    @hypothesis.settings(max_examples=25, deadline=None, database=None, derandomize=True)
    @hypothesis.given(st.lists(st.complex_numbers(min_magnitude=0, max_magnitude=10, allow_nan=False, allow_infinity=False), min_size=1, max_size=12))
    def check(values):
        data = Expr("List", *values)
        assert api.IDFT(api.DFT(data)).args == pytest.approx(values, abs=1e-12)
        assert api.Convolve(data, Expr("List", 1)).args == pytest.approx(values, abs=1e-12)
    check()


def test_bounded_units_and_molecularmass_properties():
    hypothesis = pytest.importorskip("hypothesis")
    from hypothesis import strategies as st
    @hypothesis.settings(max_examples=25, deadline=None, database=None, derandomize=True)
    @hypothesis.given(st.integers(-10000, 10000), st.integers(1, 30))
    def check(value, count):
        assert UnitConvert(UnitConvert(Quantity(value, "m"), "ft"), "m").value == pytest.approx(value, abs=1e-10)
        assert api.MolecularMass("(H2O)" + str(count)) == pytest.approx(18.015 * count, abs=0.00051)
    check()


@pytest.mark.parametrize("name,args", [("DFT", ()), ("IDFT", ()), ("Convolve", (Expr("List", 1),)), ("LowPassFilter", (0.2,)), ("HighPassFilter", (0.2,)), ("BandPassFilter", (0.1, 0.3)), ("Spectrogram", ())])
def test_signal_rejects_nonfinite_and_empty(name, args):
    if name in {"LowPassFilter", "HighPassFilter", "BandPassFilter", "Spectrogram"}:
        pytest.importorskip("scipy.signal")
    for data in [Expr("List"), Expr("List", *([complex(0, float("nan"))] * 256))]:
        with pytest.raises(ValueError):
            getattr(api, name)(data, *args)


def test_spectrogram_figure_real_and_complex():
    pytest.importorskip("scipy.signal")
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    import numpy as np
    import warnings
    for data in [np.sin(2 * np.pi * np.arange(512) / 16), np.exp(2j * np.pi * np.arange(512) / 16)]:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            figure = api.Spectrogram(Expr("List", *data), sample_rate=256)
        try:
            assert figure.axes[0].get_xlabel() == "Time"
            assert figure.axes[0].get_ylabel() == "Frequency"
            assert np.max(figure.axes[0].collections[0].get_array()) > -10
        finally:
            plt.close(figure)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_quantity_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite"):
        Quantity(value, "m")


def test_unitconvert_rejects_nonquantity():
    with pytest.raises(TypeError, match="Quantity"):
        UnitConvert(3, "m")


def test_dft_preserves_numpy_complex_scalars():
    import numpy as np
    assert api.DFT(Expr("List", np.complex64(2j))).args == (2j,)
    assert api.Convolve(Expr("List", np.complex64(2j)), Expr("List", 3)).args == (6j,)


def test_unitconvert_preserves_tiny_values():
    assert UnitConvert(Quantity(1e-20, "m"), "cm").value == pytest.approx(1e-18, rel=1e-12, abs=0)
