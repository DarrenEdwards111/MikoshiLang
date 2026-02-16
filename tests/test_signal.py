"""Tests for signal processing module."""

import pytest
import numpy as np
from mikoshilang.expr import Expr, Symbol
from mikoshilang.signal import (
    DFT, IDFT, Convolve,
    HammingWindow, HanningWindow, BlackmanWindow,
)


class TestDFT:
    def test_basic(self):
        data = Expr("List", 1, 2, 3, 4)
        result = DFT(data)
        assert result.head == "List"
        assert len(result.args) == 4

    def test_roundtrip(self):
        data = Expr("List", 1, 2, 3, 4)
        freq = DFT(data)
        back = IDFT(freq)
        assert back.head == "List"
        for i, v in enumerate(back.args):
            expected = [1, 2, 3, 4][i]
            assert abs(complex(v).real - expected) < 1e-10

    def test_single_value(self):
        data = Expr("List", 5)
        result = DFT(data)
        assert len(result.args) == 1

    def test_invalid_input(self):
        with pytest.raises(TypeError):
            DFT(42)


class TestIDFT:
    def test_basic(self):
        data = Expr("List", 10, -2, -2, -2)
        result = IDFT(data)
        assert result.head == "List"


class TestConvolve:
    def test_basic(self):
        s1 = Expr("List", 1, 2, 3)
        s2 = Expr("List", 0, 1, 0.5)
        result = Convolve(s1, s2)
        assert result.head == "List"
        assert len(result.args) == 5  # len(s1) + len(s2) - 1

    def test_identity(self):
        s1 = Expr("List", 1, 2, 3)
        s2 = Expr("List", 1)
        result = Convolve(s1, s2)
        assert len(result.args) == 3
        for i in range(3):
            assert abs(float(result.args[i]) - [1, 2, 3][i]) < 1e-10


class TestWindows:
    def test_hamming_length(self):
        w = HammingWindow(10)
        assert w.head == "List"
        assert len(w.args) == 10

    def test_hanning_length(self):
        w = HanningWindow(8)
        assert len(w.args) == 8

    def test_blackman_length(self):
        w = BlackmanWindow(16)
        assert len(w.args) == 16

    def test_hamming_symmetry(self):
        w = HammingWindow(11)
        for i in range(5):
            assert abs(float(w.args[i]) - float(w.args[10 - i])) < 1e-10

    def test_hanning_endpoints(self):
        w = HanningWindow(10)
        # Hanning window starts and ends near zero
        assert abs(float(w.args[0])) < 0.01

    def test_blackman_peak(self):
        w = BlackmanWindow(21)
        # Peak should be at center
        center = float(w.args[10])
        assert center > 0.9


class TestFilters:
    """Filters require scipy - test only if available."""

    @pytest.fixture
    def signal_data(self):
        t = np.linspace(0, 1, 1000)
        # Mix of low and high frequency
        data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)
        return Expr("List", *data.tolist())

    def test_low_pass_import(self):
        pytest.importorskip("scipy")
        from mikoshilang.signal import LowPassFilter

    def test_high_pass_import(self):
        pytest.importorskip("scipy")
        from mikoshilang.signal import HighPassFilter

    def test_band_pass_import(self):
        pytest.importorskip("scipy")
        from mikoshilang.signal import BandPassFilter

    def test_low_pass(self):
        scipy = pytest.importorskip("scipy")
        from mikoshilang.signal import LowPassFilter
        t = np.linspace(0, 1, 1000)
        data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)
        lst = Expr("List", *data.tolist())
        result = LowPassFilter(lst, 0.05, sample_rate=1000)
        assert result.head == "List"
        assert len(result.args) == 1000
