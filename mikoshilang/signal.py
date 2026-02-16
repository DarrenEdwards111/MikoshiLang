"""Signal processing module for MikoshiLang.

Wraps numpy.fft and scipy.signal for numerical operations,
SymPy for symbolic Fourier transforms.
"""

from __future__ import annotations
from typing import Any, List as ListType, Optional
import numpy as np
from .expr import Expr, Symbol


# ── Symbolic Fourier transforms (via SymPy) ──

def FourierTransform(expr, t_var, w_var) -> Any:
    """Symbolic Fourier transform using SymPy."""
    import sympy as sp
    from .math_ops import to_sympy, from_sympy
    sexpr = to_sympy(expr)
    st = to_sympy(t_var) if isinstance(t_var, Symbol) else sp.Symbol(str(t_var))
    sw = to_sympy(w_var) if isinstance(w_var, Symbol) else sp.Symbol(str(w_var))
    result = sp.fourier_transform(sexpr, st, sw)
    return from_sympy(result)


def InverseFourierTransform(expr, w_var, t_var) -> Any:
    """Symbolic inverse Fourier transform using SymPy."""
    import sympy as sp
    from .math_ops import to_sympy, from_sympy
    sexpr = to_sympy(expr)
    sw = to_sympy(w_var) if isinstance(w_var, Symbol) else sp.Symbol(str(w_var))
    st = to_sympy(t_var) if isinstance(t_var, Symbol) else sp.Symbol(str(t_var))
    result = sp.inverse_fourier_transform(sexpr, sw, st)
    return from_sympy(result)


# ── Discrete transforms ──

def _list_to_array(lst: Expr) -> np.ndarray:
    """Convert a List Expr to numpy array."""
    if isinstance(lst, Expr) and lst.head == "List":
        return np.array([complex(x) if isinstance(x, complex) else float(x) for x in lst.args])
    raise TypeError("Expected a List expression")


def _array_to_list(arr: np.ndarray) -> Expr:
    """Convert numpy array to List Expr."""
    result = []
    for x in arr:
        if np.isreal(x):
            v = float(np.real(x))
            if v == int(v):
                result.append(int(v))
            else:
                result.append(v)
        else:
            result.append(complex(x))
    return Expr("List", *result)


def DFT(data: Expr) -> Expr:
    """Discrete Fourier Transform."""
    arr = _list_to_array(data)
    result = np.fft.fft(arr)
    return _array_to_list(result)


def IDFT(data: Expr) -> Expr:
    """Inverse Discrete Fourier Transform."""
    arr = _list_to_array(data)
    result = np.fft.ifft(arr)
    return _array_to_list(result)


# ── Filters ──

def LowPassFilter(data: Expr, cutoff: float, sample_rate: float = 1.0, order: int = 5) -> Expr:
    """Apply a Butterworth low-pass filter."""
    from scipy.signal import butter, filtfilt
    arr = _list_to_array(data).real
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    filtered = filtfilt(b, a, arr)
    return _array_to_list(filtered)


def HighPassFilter(data: Expr, cutoff: float, sample_rate: float = 1.0, order: int = 5) -> Expr:
    """Apply a Butterworth high-pass filter."""
    from scipy.signal import butter, filtfilt
    arr = _list_to_array(data).real
    nyq = 0.5 * sample_rate
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high')
    filtered = filtfilt(b, a, arr)
    return _array_to_list(filtered)


def BandPassFilter(data: Expr, low: float, high: float, sample_rate: float = 1.0, order: int = 5) -> Expr:
    """Apply a Butterworth band-pass filter."""
    from scipy.signal import butter, filtfilt
    arr = _list_to_array(data).real
    nyq = 0.5 * sample_rate
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype='band')
    filtered = filtfilt(b, a, arr)
    return _array_to_list(filtered)


# ── Convolution ──

def Convolve(signal1: Expr, signal2: Expr) -> Expr:
    """Convolve two signals."""
    arr1 = _list_to_array(signal1).real
    arr2 = _list_to_array(signal2).real
    result = np.convolve(arr1, arr2)
    return _array_to_list(result)


# ── Window functions ──

def HammingWindow(n: int) -> Expr:
    """Generate a Hamming window of length n."""
    return _array_to_list(np.hamming(n))


def HanningWindow(n: int) -> Expr:
    """Generate a Hanning window of length n."""
    return _array_to_list(np.hanning(n))


def BlackmanWindow(n: int) -> Expr:
    """Generate a Blackman window of length n."""
    return _array_to_list(np.blackman(n))


# ── Spectrogram ──

def Spectrogram(data: Expr, sample_rate: float = 1.0) -> Any:
    """Compute spectrogram. Returns matplotlib figure if available, else raw data."""
    from scipy.signal import spectrogram as scipy_spectrogram
    arr = _list_to_array(data).real
    f, t, Sxx = scipy_spectrogram(arr, fs=sample_rate)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10))
        ax.set_ylabel("Frequency")
        ax.set_xlabel("Time")
        ax.set_title("Spectrogram")
        fig.tight_layout()
        return fig
    except ImportError:
        return Expr("List",
                     Expr("Rule", "frequencies", _array_to_list(f)),
                     Expr("Rule", "times", _array_to_list(t)))
