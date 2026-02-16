# MikoshiLang

A symbolic computation language for Python — pattern matching, algebraic simplification, and computational intelligence.

Built by **Mikoshi Ltd**.

## Installation

```bash
pip install mikoshilang

# With Jupyter support
pip install mikoshilang[jupyter]

# With signal processing
pip install mikoshilang[signal]

# Everything
pip install mikoshilang[all]
```

## Language Syntax

MikoshiLang uses Wolfram-style syntax. Launch the REPL:

```bash
mikoshilang
```

### Arithmetic

```
In[1]:= 2 + 3 * x
Out[1]= 2 + 3*x

In[2]:= x^2 - 4
Out[2]= x^2 - 4

In[3]:= (x + 1)(x - 1)    (* implicit multiplication *)
Out[3]= (x + 1)*(x - 1)
```

### Function Calls (Square Brackets)

```
In[1]:= Sin[Pi/2]
In[2]:= Diff[x^2, x]
In[3]:= Integrate[x^2, x]
In[4]:= Solve[x^2 - 4 == 0, x]
In[5]:= Simplify[(x^2 - 1)/(x - 1)]
In[6]:= Factor[x^2 - 4]
In[7]:= Expand[(x + 1)^3]
In[8]:= Limit[Sin[x]/x, x -> 0]
In[9]:= Series[Exp[x], {x, 0, 5}]
```

### Lists and Data

```
In[1]:= {1, 2, 3, 4, 5}
In[2]:= Range[10]
In[3]:= Table[i^2, {i, 1, 10}]
In[4]:= Map[Sin, {1, 2, 3}]
In[5]:= Select[{1, -2, 3, -4}, Positive]
```

### Matrices

```
In[1]:= Det[{{1, 2}, {3, 4}}]
Out[1]= -2

In[2]:= Inverse[{{1, 2}, {3, 4}}]
```

### Pattern Matching and Rules

```
In[1]:= x /. x -> 3
In[2]:= f[x_] := x^2
In[3]:= MatchQ[Sin[x], Sin[_]]
In[4]:= ReplaceAll[x + y, {x -> 1, y -> 2}]
```

### Constants

`Pi`, `E`, `I`, `Infinity`, `True`, `False`

### Comments

```
(* This is a comment *)
```

## Jupyter Integration

Install the kernel:

```bash
pip install mikoshilang[jupyter]
python -m mikoshilang.jupyter.install
```

Then open Jupyter Notebook and select the "MikoshiLang" kernel. Features:
- LaTeX rendering of expressions
- Inline matplotlib plots with `Plot[Sin[x], {x, -Pi, Pi}]`
- Rich display of matrices and lists

### Plotting

```
Plot[Sin[x], {x, -Pi, Pi}]
Plot[{Sin[x], Cos[x]}, {x, -Pi, Pi}]
ListPlot[{1, 4, 9, 16, 25}]
ListLinePlot[{1, 4, 9, 16, 25}]
```

## Physics Units

```
In[1]:= q = Quantity[9.8, "m/s^2"]
In[2]:= t = Quantity[3, "s"]
In[3]:= q * t
Out[3]= Quantity[29.4, "m/s"]

In[4]:= UnitConvert[Quantity[100, "cm"], "m"]
Out[4]= Quantity[1, "m"]

In[5]:= UnitConvert[Quantity[72, "kg"], "lb"]
Out[5]= Quantity[158.73, "lb"]
```

### Supported Units

| Category | Units |
|----------|-------|
| Length | m, cm, mm, km, in, ft, yd, mi |
| Mass | kg, g, mg, lb, oz |
| Time | s, ms, min, h, day |
| Speed | m/s, km/h, mph |
| Force | N, lbf |
| Energy | J, kJ, cal, kcal, eV, kWh |
| Power | W, kW, hp |
| Pressure | Pa, kPa, atm, bar, psi |
| Temperature | K, C, F |
| Electric | A, V, ohm, Farad, H, Coulomb, Hz |

### Physical Constants

`SpeedOfLight`, `GravitationalConstant`, `PlanckConstant`, `BoltzmannConstant`, `AvogadroNumber`, `ElementaryCharge`

## Chemistry

```
In[1]:= Element["H"]
Out[1]= {name: "Hydrogen", number: 1, mass: 1.008, symbol: "H"}

In[2]:= AtomicMass["O"]
Out[2]= 15.999

In[3]:= ElectronConfiguration["Fe"]
Out[3]= [Ar] 3d6 4s2

In[4]:= MolecularMass["H2O"]
Out[4]= 18.015

In[5]:= MolecularMass["C6H12O6"]
Out[5]= 180.156

In[6]:= BalanceEquation["H2 + O2 -> H2O"]
Out[6]= 2H2 + O2 -> 2H2O
```

All 118 elements included with atomic number, symbol, name, mass, electron configuration, electronegativity, and category.

## Signal Processing

Requires `pip install mikoshilang[signal]` for filters and spectrogram.

```
In[1]:= DFT[{1, 2, 3, 4}]
In[2]:= IDFT[{10, -2, -2, -2}]

In[3]:= Convolve[{1, 2, 3}, {0, 1, 0.5}]

In[4]:= HammingWindow[256]
In[5]:= HanningWindow[256]
In[6]:= BlackmanWindow[256]

(* Symbolic Fourier transforms via SymPy *)
In[7]:= FourierTransform[Exp[-t^2], t, w]

(* Filters (require scipy) *)
In[8]:= LowPassFilter[data, cutoff]
In[9]:= HighPassFilter[data, cutoff]
In[10]:= BandPassFilter[data, low, high]

In[11]:= Spectrogram[data, sample_rate]
```

## Python API

```python
from mikoshilang import *

# Parse and evaluate Wolfram-style syntax
result = parse_and_eval("Simplify[(x^2 - 1)/(x - 1)]")

# Or use Python constructors directly
x = Symbol("x")
expr = x**2 + 2*x + 1
print(simplify(expr))
print(to_latex(expr))

# Units
q = Quantity(100, "cm")
print(UnitConvert(q, "m"))

# Chemistry
print(MolecularMass("C6H12O6"))
print(BalanceEquation("H2 + O2 -> H2O"))
```

## Feature Comparison with Wolfram

| Feature | Wolfram | MikoshiLang |
|---------|---------|-------------|
| Symbolic algebra | ✅ | ✅ (via SymPy) |
| Pattern matching | ✅ | ✅ |
| Calculus | ✅ | ✅ |
| Linear algebra | ✅ | ✅ (via NumPy) |
| Number theory | ✅ | ✅ |
| Wolfram-style syntax | ✅ | ✅ |
| Jupyter notebooks | ✅ | ✅ |
| LaTeX output | ✅ | ✅ |
| Plotting | ✅ | ✅ (via Matplotlib) |
| Physics units | ✅ | ✅ |
| Chemistry | ✅ | ✅ (118 elements) |
| Signal processing | ✅ | ✅ (via SciPy) |
| Free & open source | ❌ | ✅ |

## License

Apache 2.0 — Mikoshi Ltd.
