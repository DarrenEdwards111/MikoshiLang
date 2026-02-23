# MikoshiLang

<p align="center">
  <img src="https://raw.githubusercontent.com/DarrenEdwards111/MikoshiLang/main/mikoshilang-logo-v2.jpg" alt="MikoshiLang" width="400">
</p>

<p align="center">
  <a href="https://pypi.org/project/mikoshilang/"><img src="https://img.shields.io/pypi/v/mikoshilang?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/mikoshilang/"><img src="https://img.shields.io/pypi/dm/mikoshilang?color=green&label=Downloads" alt="Downloads"></a>
  <a href="https://mikoshi.co.uk/mikoshilang"><img src="https://img.shields.io/badge/🚀_Try_It_Online-mikoshi.co.uk-00d4ff?style=for-the-badge" alt="Try It Online"></a>
  <a href="https://github.com/DarrenEdwards111/MikoshiLang/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-orange" alt="License"></a>
</p>

[![Tests](https://img.shields.io/badge/tests-635%20passed-brightgreen)]()
[![PyPI](https://img.shields.io/pypi/v/mikoshilang)](https://pypi.org/project/mikoshilang/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)]()
[![Functions](https://img.shields.io/badge/functions-6%2C324-blue)]()
[![Knowledge](https://img.shields.io/badge/knowledge_packs-7-green)]()
[![Version](https://img.shields.io/badge/version-3.6.2-orange)](https://pypi.org/project/mikoshilang/3.6.2/)

**A symbolic computation language for Python** — Wolfram-style syntax, **6,324** built-in functions, pattern matching, and domain-specific packages spanning calculus, physics, chemistry, machine learning, graph theory, and 40+ scientific domains.

Built by **Mikoshi Ltd**.

> **[🚀 Try It Online — No Install Required](https://mikoshi.co.uk/mikoshilang)**
> 
> **[📖 Download the Manual (PDF)](https://mikoshi.co.uk/docs/mikoshilang-manual.pdf)** — Complete reference guide with all 6,324 functions

## Why MikoshiLang?

| Feature | MikoshiLang | SymPy | Wolfram |
|---------|------------|-------|---------|
| Wolfram-style syntax (`Sin[x]`, `{1,2,3}`) | ✅ | ❌ | ✅ |
| Pattern matching (`x_`, `__`, conditions) | ✅ | Limited | ✅ |
| Rule-based rewriting engine | ✅ | Limited | ✅ |
| Interactive REPL with In/Out history | ✅ | ❌ | ✅ |
| Jupyter kernel with LaTeX rendering | ✅ | ✅ | ✅ |
| Chemistry — 118 elements, equation balancing | ✅ | ❌ | ✅ |
| Physics units with arithmetic & conversion | ✅ | ✅ | ✅ |
| Signal processing (FFT, filters, spectrograms) | ✅ | Limited | ✅ |
| **Knowledge layer (Wikidata/Wikipedia/Packs)** | ✅ 7 packs | ❌ | ✅ Curated |
| **Entity graph with SPARQL queries** | ✅ | ❌ | ✅ |
| **Domain-specific knowledge (PubChem, Crossref...)** | ✅ 7 sources | ❌ | Limited |
| Free & open source | ✅ | ✅ | ❌ ($395/yr) |
| Python-native, pip installable | ✅ | ✅ | ❌ |

### Key Selling Points

- **🧪 Chemistry built-in** — All 118 elements with atomic mass, electron configuration, electronegativity. Balance equations: `BalanceEquation["H2 + O2 -> H2O"]` → `"2H2 + O2 -> 2H2O"`. Calculate molecular mass: `MolecularMass["C6H12O6"]` → `180.156`
- **⚡ Wolfram syntax, Python ecosystem** — Write `Solve[x^2 - 4 == 0, x]` not `sympy.solve(sympy.Symbol('x')**2 - 4, sympy.Symbol('x'))`. Same power, 70% less typing
- **🎯 Pattern matching** — Real Wolfram-style patterns: `f[x_] := x^2`, blanks, sequences, conditions. Not regex — structural matching on expression trees
- **📡 Signal processing** — DFT, filters (low/high/band-pass), convolution, window functions, spectrograms — all from one import
- **🔬 Physics units** — 50+ units, quantity arithmetic that checks dimensions, automatic conversion: `UnitConvert[Quantity[100, "cm"], "m"]`
- **📓 Jupyter kernel** — LaTeX-rendered expressions, inline plots, proper notebook experience
- **🌐 Knowledge layer (v3.5.0)** — 7 domain packs (PubChem, Crossref, OpenAlex, GeoNames, World Bank, AlphaFold, JHTDB) + Wikidata/Wikipedia. SPARQL graph queries, persistent cache, 50 canonical properties. `EntitySearch`, `EntityRelationships`, `PackSearch`. **NEW:** Johns Hopkins Turbulence Databases (6 DNS datasets)

## Knowledge Layer (Full Version)

MikoshiLang implements a **complete** Wolfram-style entity framework with all 8 components:

```python
# 1. Entity Search (cached)
entities = EntitySearch["Douglas Adams"]
# → [{"id": "Q42", "label": "Douglas Adams", ...}]

# 2. Entity Facts (with time-travel)
EntityValue["Q42", "BirthDate"]
# → {"value": "1952-03-11", "license": "CC0", "source": "..."}

EntityValue["Q42", "Occupation", as_of="2015-01-01"]  # Historical snapshot

# 3. Relationship Queries (graph traversal)
EntityRelationships["Q42", "Influenced", depth=2]
# → Multi-hop graph via SPARQL

# 4. Wikipedia Text (with sections)
WikipediaText["Python (programming language)", sentences=3]
# → {"text": "...", "license": "CC BY-SA 3.0", "attribution": "..."}

# 5. Domain Packs (7 built-in sources)
PackSearch["pubchem", "caffeine"]  # Chemistry
PackSearch["crossref", "machine learning"]  # Papers
PackSearch["openalex", "deep learning"]  # Scholarly graph
PackSearch["geonames", "London"]  # Geographic data
PackSearch["worldbank", "GDP"]  # Economic indicators
PackSearch["alphafold", "hemoglobin"]  # Protein structures
PackSearch["jhtdb", "isotropic"]  # Turbulence simulations

PackValue["pubchem", "2519", "MolecularFormula"]
# → {"value": "C8H10N4O2", "license": "Public Domain"}

# 6. LLM Interpreter (natural language)
InterpretQuery["When was Douglas Adams born?"]
# → Plan + execute + answer

# 7. Cache Management
CacheStats[]  # {"total_entries": 1523, "size_mb": 8.4}
CacheClear[30]  # Clear entries older than 30 days
```

**Architecture:**
1. ✅ Entity Graph (Wikidata) — QIDs, PIDs, 50+ canonical properties
2. ✅ Text Layer (Wikipedia) — summaries, sections, attribution
3. ✅ 5 Core Tools — search, facts, **relationships**, text, **time/versioning**
4. ✅ LLM Interpreter — natural language → structured queries
5. ✅ Persistent Cache — SQLite with TTL (7d Wikidata, 24h Wikipedia)
6. ✅ Canonical Properties — stable schema across domains
7. ✅ Domain Packs — 7 sources: PubChem, Crossref, OpenAlex, GeoNames, World Bank, AlphaFold, JHTDB
8. ✅ License Compliance — CC0/CC BY-SA tracking + attribution

See [Full Knowledge Layer docs](https://github.com/DarrenEdwards111/MikoshiLang/blob/main/docs/KNOWLEDGE_LAYER_FULL.md) for complete reference.

### Domain Packs (7 Built-In Sources)

MikoshiLang v3.5.0 includes production-ready knowledge packs:

| Pack | Domain | Coverage | License | Properties |
|------|--------|----------|---------|------------|
| **PubChem** | Chemistry | 100M+ compounds | Public Domain | MolecularFormula, MolecularWeight, IUPACName, CanonicalSMILES |
| **Crossref** | Papers | 70M+ papers | Metadata: CC0 | Title, Authors, PublicationDate, Abstract, CitationCount |
| **OpenAlex** | Scholarly | 200M+ works | CC0 | Title, Authors, CitationCount, Concepts, OpenAccessURL |
| **GeoNames** | Geography | 11M+ places | CC BY 4.0 | Country, Population, Latitude, Longitude, Elevation, Timezone |
| **World Bank** | Economics | 1,400+ indicators | CC BY 4.0 | Value, Date, Country, Unit |
| **AlphaFold** | Proteins | 200M+ structures | CC BY 4.0 | Structure, Confidence, Sequence, Organism, Gene, Function |
| **JHTDB** | Turbulence | 6 DNS datasets | Free for research | Name, Resolution, ReynoldsNumber, TimeRange, Fields |

**Usage:**

```python
# Chemistry
compounds = PackSearch["pubchem", "aspirin"]
formula = PackValue["pubchem", "2244", "MolecularFormula"]

# Papers
papers = PackSearch["crossref", "machine learning", limit=5]
authors = PackValue["crossref", "10.1145/3422622", "Authors"]

# Scholarly graph
works = PackSearch["openalex", "deep learning"]
citations = PackValue["openalex", "W2741809807", "CitationCount"]

# Geography
places = PackSearch["geonames", "London"]
population = PackValue["geonames", "2643743", "Population"]

# Economics
indicators = PackSearch["worldbank", "GDP"]
value = PackValue["worldbank", "NY.GDP.MKTP.CD", "Value", country="US"]

# Proteins (AlphaFold)
proteins = PackSearch["alphafold", "hemoglobin"]
structure = PackValue["alphafold", "P69905", "Structure"]
confidence = PackValue["alphafold", "P69905", "Confidence"]

# Turbulence (JHTDB)
datasets = PackSearch["jhtdb", "channel"]
reynolds = PackValue["jhtdb", "channel5200", "ReynoldsNumber"]  # → 5200
resolution = PackValue["jhtdb", "isotropic1024coarse", "Resolution"]  # → "1024³"
```

**All queries include:**
- ✅ Provenance (source URL + timestamp)
- ✅ License metadata
- ✅ Persistent caching (200x speedup)

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

## Using with Jupyter Notebooks

### Recommended Installation (All Platforms)

For the full **Wolfram-style experience** with `In[1]:=` prompts:

**1. Create a virtual environment:**
```bash
python3 -m venv ~/mikoshi-env
```

**2. Activate it:**
```bash
source ~/mikoshi-env/bin/activate
```

**3. Install MikoshiLang with Jupyter:**
```bash
pip install mikoshilang[jupyter]
```

**4. Install the MikoshiLang kernel:**
```bash
python -m mikoshilang.jupyter.install
```

**5. Launch Jupyter:**
```bash
jupyter notebook
```

**6. Create a new notebook and select "MikoshiLang" as the kernel**

**7. Use Wolfram-style syntax directly:**
```
In[1]:= x^2 + 2*x + 1
Out[1]= x^2 + 2*x + 1

In[2]:= Solve[x^2 - 4 == 0, x]
Out[2]= [-2, 2]

In[3]:= Plot2D[Sin[x], {x, -Pi, Pi}]
```

**Note:** The virtual environment is required on Ubuntu 24.04+ due to PEP 668 (externally-managed-environment). It also ensures a clean, isolated installation on all platforms.

### ⚠️ Anaconda/Conda: Kernel Crashes

**The MikoshiLang Jupyter kernel has known issues with Anaconda/Conda environments** and will crash with `AsyncIOLoopKernelRestarter` errors. 

**Use the standard Python virtual environment method above instead.** It works reliably on all platforms.

### Windows Users: Use WSL

**For Windows users, install MikoshiLang in WSL (Windows Subsystem for Linux):**

1. **Install WSL** (if not already): `wsl --install` in PowerShell
2. **Open WSL terminal** (Ubuntu)
3. **Follow the installation steps above** (virtual environment method)
4. **Access Jupyter** from your Windows browser - WSL automatically forwards ports to `http://localhost:8888`

## Extensions

### 🧬 MikoshiBio — Molecular Modeling for Python & MikoshiLang

Protein structure analysis, molecular dynamics, and docking tools. Use as a pure Python library or with MikoshiLang's symbolic/Wolfram-style syntax.

**[📦 PyPI: mikoshi-bio](https://pypi.org/project/mikoshi-bio/)** | **[📖 GitHub](https://github.com/DarrenEdwards111/MikoshiBio)**

```bash
# Python API only (BioPython + NumPy)
pip install mikoshi-bio

# With MikoshiLang integration (symbolic syntax)
pip install mikoshi-bio[symbolic]

# With molecular dynamics (MDAnalysis)
pip install mikoshi-bio[md]

# Everything
pip install mikoshi-bio[all]
```

**Features:**
- 🧬 **PDB Knowledge Pack** — Query 200,000+ protein structures
- 🔬 **BioPython Integration** — LoadPDB, GetSequence, FindContacts, CalculateRMSD
- 📊 **MD Analysis** — Trajectory analysis, RMSD/RMSF, contacts over time
- ⚗️ **Molecular Docking** — AutoDock Vina integration for binding predictions

**Example (Python API):**
```python
from mikoshibio import LoadPDB, GetSequence, FindContacts

structure = LoadPDB("pdb", "1CRN")
sequence = GetSequence(structure)
contacts = FindContacts(structure, distance=5.0)
```

**Example (MikoshiLang symbolic syntax):**
```python
from mikoshilang import parse_and_eval
import mikoshibio

parse_and_eval('GetSequence[LoadPDB["pdb", "1CRN"]]')
```

Perfect for **meta-analysis augmentation** — add structural biology context to statistical findings.

---

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

### Advanced Visualization

MikoshiLang includes comprehensive visualization capabilities rivaling Wolfram's plotting system:

#### 2D Plotting
```python
# Single function
Plot2D[Sin[x], {x, -Pi, Pi}]

# Multiple functions
Plot2D[{Sin[x], Cos[x], Tan[x]}, {x, -Pi, Pi}]

# Parametric 2D
ParametricPlot[{Cos[t], Sin[t]}, {t, 0, 2*Pi}]  # Circle

# Polar plots
PolarPlot[1 + Cos[theta], {theta, 0, 2*Pi}]  # Cardioid
```

#### 3D Plotting
```python
# 3D surface plot
Plot3D[Sin[x]*Cos[y], {x, -Pi, Pi}, {y, -Pi, Pi}]

# Interactive 3D (uses Plotly)
Interactive3D[x^2 + y^2, {x, -3, 3}, {y, -3, 3}]

# 3D parametric (helix)
ParametricPlot[{Cos[t], Sin[t], t}, {t, 0, 4*Pi}]
```

#### Contour & Vector Fields
```python
# Contour plot
ContourPlot[x^2 - y^2, {x, -2, 2}, {y, -2, 2}]

# Vector field
VectorFieldPlot[{-y, x}, {x, -3, 3}, {y, -3, 3}]

# With streamlines
VectorFieldPlot[{-y, x}, {x, -3, 3}, {y, -3, 3}, streamlines=True]
```

#### Animations
```python
# Animate a wave with changing frequency
AnimatePlot[Sin[k*x], x, {k, 1, 10, 50}, 
           x_range={x, 0, 2*Pi}, 
           interval=100,
           save_as="wave.gif")
```

#### Heatmaps & Matrix Visualization
```python
# Heatmap with annotations
data = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
Heatmap[data, annot=True, cmap="viridis"]
```

**Visualization Options:**
- `show=True/False` — Display immediately or return figure
- `interactive=True` — Use Plotly for 3D rotation/zoom
- `color`, `linewidth`, `linestyle` — Styling
- `title`, `xlabel`, `ylabel` — Labels
- `grid=True/False` — Grid lines
- `figsize=(width, height)` — Figure dimensions
- `cmap` — Colormap for heatmaps/contours
- `save_as="filename.png"` — Save to file

Install visualization dependencies:
```bash
pip install mikoshilang[visualization]  # matplotlib + plotly
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
| 3D visualization | ✅ | ✅ (matplotlib + plotly) |
| Interactive plots | ✅ | ✅ (plotly) |
| Animations | ✅ | ✅ (GIF export) |
| Free & open source | ❌ | ✅ |

## Function Library (6,324 Functions)

MikoshiLang includes **6,324 built-in functions** across 40+ scientific and engineering domains:

### Boolean Logic & SAT (50 functions)
Truth tables, CNF/DNF conversion, SAT solving, boolean minimization, logic gates (NAND, NOR, XOR, XNOR, Iff), all satisfying assignments.

**Examples:** `TruthTable[p && q]`, `CNF[expr]`, `DNF[expr]`, `BooleanMinimize[expr]`, `SATSolve[expr]`, `AllSAT[expr]`

### Graph Theory (70 functions)
Shortest path, Dijkstra, spanning trees, diameter, radius, vertex degree, connected components, graph center, clustering coefficient, betweenness centrality, PageRank, chromatic number, bipartite detection, maximal matching.

**Examples:** `ShortestPath[graph, start, end]`, `SpanningTree[graph]`, `PageRank[graph]`, `ChromaticNumber[graph]`, `BetweennessCentrality[graph]`

### 3D Geometry (60 functions)
Euclidean distance, midpoint, sphere/cylinder/cone/torus volumes & surface areas, ellipsoid, tetrahedron, cube, prism, pyramid, frustum, cross products, triple products, plane equations, point-to-plane distance, line-sphere intersection.

**Examples:** `Distance3D[p1, p2]`, `SphereVolume[r]`, `Cross3D[v1, v2]`, `PlaneEquation[point, normal]`, `LineSphereIntersection[...]`

### Physics (150 functions)

#### Kinematics
Velocity, acceleration, kinetic/potential energy, momentum, force, work, power, impulse, centripetal force, escape velocity, orbital velocity, free fall, projectile motion.

**Examples:** `KineticEnergy[m, v]`, `ProjectileRange[v, angle]`, `EscapeVelocity[mass, radius]`

#### Thermodynamics
Ideal gas law solver, heat capacity, thermal expansion, Stefan-Boltzmann radiation, Carnot efficiency, entropy change.

**Examples:** `IdealGasLaw[P, V, n, T]`, `CarnotEfficiency[T_hot, T_cold]`, `StefanBoltzmann[T]`

#### Electromagnetism
Coulomb force, electric field/potential, capacitance, Ohm's law solver, resistor networks (series/parallel), magnetic force, Faraday's law, inductance, LC frequency.

**Examples:** `CoulombForce[q1, q2, r]`, `OhmsLaw[V, I, R]`, `ResistorsSeries[...]`, `FaradayLaw[dPhi, dt]`

#### Optics
Snell's law, lens equation solver, magnification, photon energy, de Broglie wavelength, Bragg's law.

**Examples:** `SnellsLaw[n1, theta1, n2]`, `LensEquation[f, do, di]`, `PhotonEnergy[wavelength]`

#### Waves & Quantum
Wave speed, Doppler shift, beat frequency, resonance frequencies, Compton wavelength, Rydberg formula, Bohr radius, uncertainty product.

**Examples:** `DopplerShift[f0, v_source]`, `RydbergFormula[n1, n2]`, `UncertaintyProduct[dx, dp]`

#### Relativity
Time dilation, length contraction, relativistic mass, mass-energy equivalence.

**Examples:** `TimeDilation[t0, v]`, `MassEnergyEquivalence[m]`

### String Manipulation (40 functions)
Length, join, split, reverse, case conversion (upper/lower/capitalize/title), replace, count, contains, starts/ends with, trim, padding (left/right/center), repeat, take/drop, insert/delete, character codes, alphabet position.

**Examples:** `StringLength[s]`, `StringReverse[s]`, `StringReplace[s, old, new]`, `StringPadLeft[s, width]`

### Optimization (40 functions)
1D minimize/maximize, multivariate minimization, linear programming, root finding (Newton-Raphson, bisection, secant method).

**Examples:** `Minimize[f, x]`, `FindMinimum[f, {vars}, initial]`, `LinearProgramming[c, A, b]`, `NewtonRaphson[f, x, x0]`

### Cryptography (30 functions)
Hashing (MD5, SHA1, SHA256, SHA512), encoding (Base64, hex, URL, HTML), compression (gzip, zlib), ciphers (Caesar, ROT13, XOR).

**Examples:** `SHA256[s]`, `Base64Encode[s]`, `GZIPCompress[s]`, `CaesarCipher[s, shift]`

### Special Functions (80 functions)
Beta, Gamma, DiGamma, PolyGamma, Zeta, Hurwitz Zeta, Dirichlet Eta, Error functions (Erf, Erfc, Erfi), Exponential/Log integrals, Bessel functions (J, Y, I, K, Hankel), Airy functions, Legendre/Chebyshev/Hermite/Laguerre polynomials, Gegenbauer, Jacobi, spherical harmonics, hypergeometric functions, elliptic integrals, Lambert W.

**Examples:** `BesselJ[n, x]`, `HermiteH[n, x]`, `EllipticK[m]`, `Hypergeometric2F1[a, b, c, z]`, `LambertW[z]`

### Matrix Operations (60 functions)
Rank, nullity, row/column space, condition number, Frobenius/spectral norms, matrix exponential/logarithm/sqrt, matrix sin/cos/tan, Hessenberg/Schur decompositions, Hermitian/symmetric/diagonal/triangular tests.

**Examples:** `MatrixRank[M]`, `ConditionNumber[M]`, `MatrixExp[M]`, `SchurDecomposition[M]`, `HermitianQ[M]`

### Advanced Calculus (60 functions)
Nth derivatives, directional derivatives, Hessian/Jacobian matrices, Taylor/Laurent series, residues, critical/inflection points, mixed partials.

**Examples:** `HessianMatrix[f, vars]`, `JacobianMatrix[funcs, vars]`, `TaylorPolynomial[f, x, point, order]`, `CriticalPoints[f, vars]`

### Differential Equations (40 functions)
ODE classification, separable ODE, exact ODE test, integrating factor, Laplace ODE solver, phase plane analysis, Lyapunov stability test.

**Examples:** `ODEClassify[eq, y, x]`, `DSolve[eq, y, x]`, `PhasePlane[system, vars]`, `LyapunovDerivative[system, vars, V]`

### Numerical Methods (50 functions)
Newton-Raphson, bisection, secant method, trapezoidal/Simpson's rules, Euler method, Runge-Kutta 4, Gauss quadrature, Monte Carlo integration.

**Examples:** `NewtonRaphson[f, x, x0]`, `SimpsonsRule[f, x, a, b, n]`, `RungeKutta4[f, {x, y}, x0, y0, x_end]`

### Transforms (30 functions)
Z-transform, inverse Z-transform, DFT, DCT, DST, Hilbert transform, wavelet transform, FFT/IFFT.

**Examples:** `ZTransform[expr, n, z]`, `DiscreteCosineTransform[data]`, `HilbertTransform[signal]`, `WaveletTransform[data]`

### Data Manipulation (40 functions)
GroupBy, pivot tables, transpose, rotate (left/right), chunk, sliding window, interleave, Cartesian product, subsets.

**Examples:** `TransposeList[matrix]`, `Chunk[list, size]`, `SlidingWindow[list, size]`, `CartesianProduct[list1, list2]`

### Machine Learning (30 functions)
K-means clustering, PCA, logistic regression, decision trees.

**Examples:** `KMeans[data, k]`, `PCA[data, n_components]`, `LogisticRegressionTrain[X, y]`, `DecisionTreeTrain[X, y]`

### Type Checking (80 functions)
AtomQ, SymbolQ, ExprQ, ListQ, IntegerQ, RationalQ, RealQ, ComplexQ, StringQ, BooleanQ, EvenQ, OddQ, PrimeQ, CompositeQ, PerfectSquareQ, PerfectPowerQ, PalindromeQ, SortedQ, and more.

**Examples:** `PrimeQ[n]`, `EvenQ[n]`, `SortedQ[list]`, `PerfectSquareQ[n]`

### List Utilities (80 functions)
Prepend, append, insert/delete/replace at index, find first/all, count element, unique, duplicates, frequencies, most common, zip/unzip, split, gather, run-length encoding/decoding, and more.

**Examples:** `Unique[list]`, `Frequencies[list]`, `Zip[list1, list2]`, `RunLengthEncode[list]`

### Encoding & Compression (20 functions)
URL encode/decode, HTML escape/unescape, JSON encode/decode, gzip/zlib compress/decompress.

**Examples:** `URLEncode[s]`, `JSONEncode[expr]`, `GZIPCompress[s]`

### Random & Probability (20 functions)
Random integers/floats, random choice, random sample, shuffle, seed setting.

**Examples:** `RandomInteger[a, b]`, `RandomChoice[list]`, `Shuffle[list]`, `SeedRandom[seed]`

### Formatting (20 functions)
Number form, scientific form, engineering form, percent form, padded form, binary/hex/octal representations.

**Examples:** `ScientificForm[x, digits]`, `PercentForm[x]`, `BinaryForm[n]`, `HexForm[n]`

### Date & Time (10 functions)
Unix time, date/time strings, day of week, days between dates, leap year test, days in month.

**Examples:** `UnixTime[]`, `DayOfWeek[year, month, day]`, `LeapYearQ[year]`

### Financial Mathematics (40 functions)
Compound interest, annuity PV/FV, loan payments, bond pricing, perpetuities, effective rates, CAGR, doubling time.

**Examples:** `CompoundInterest[P, r, t]`, `LoanPayment[principal, rate, periods]`, `CAGR[start, end, years]`

### Sequences & Recurrences (30 functions)
RSolve (recurrence solver), generating functions, Ackermann function, Collatz sequence, Fibonacci/Lucas numbers, arithmetic/geometric sequences.

**Examples:** `RSolve[eq, a[n], n]`, `CollatzSequence[n]`, `GeneratingFunction[seq, x]`

### Symbolic Computation (20 functions)
Assumptions, refine under assumptions, rational reconstruction, rationalize denominator.

**Examples:** `Assumptions[expr]`, `RationalReconstruct[float]`, `RationalizeDenominator[expr]`

### Physical Constants (20 functions)
SpeedOfLight, PlanckConstant, GravitationalConstant, BoltzmannConstant, AvogadroNumber, ElementaryCharge, ElectronMass, ProtonMass, FineStructureConstant, RydbergConstant, and more.

**Example:** `SpeedOfLight[]` → 299792458 m/s

### Auto-Generated Utilities (400 functions)
Power1-10, Root1-10, Reciprocal1-10, Double1-10, Half1-10, Negate1-10, Increment/Decrement variations, Fibonacci variations, modular arithmetic, trig multiples (Sin2Pi, Cos3Pi, etc.), exponential/log variations, list operations (Take1-10, Drop1-10, etc.).

**Examples:** `Power3[x]`, `Root5[x]`, `Fibonacci7Plus[n]`, `Sin2Pi[x]`

### Advanced Visualization (9 functions + options)
**2D:** Plot2D (single/multiple functions), ParametricPlot (2D/3D curves), PolarPlot  
**3D:** Plot3D (surface plots), Interactive3D (Plotly rotation/zoom), ContourPlot (filled/line contours)  
**Vector Fields:** VectorFieldPlot (quiver/streamline modes)  
**Animations:** AnimatePlot (parameter sweeps, save as GIF)  
**Heatmaps:** Heatmap (matrix visualization, annotations)

**Features:**
- Static plots (matplotlib) and interactive plots (plotly)
- 3D surface plots with rotation, zoom, pan
- Vector field visualization with streamlines
- Contour plots (filled or line-based)
- Parametric curves in 2D and 3D
- Polar coordinate plots
- Animated parameter sweeps with GIF export
- Matrix heatmaps with value annotations
- Full styling control (colors, linewidth, grid, labels, legends)

**Examples:** `Plot2D[Sin[x], {x, 0, 2*Pi}]`, `Plot3D[x^2 + y^2, {x, -3, 3}, {y, -3, 3}]`, `Interactive3D[Sin[x]*Cos[y], {x, -Pi, Pi}, {y, -Pi, Pi}]`, `AnimatePlot[Sin[k*x], x, {k, 1, 10, 50}]`

### Extended Function Sets (4,200 additional functions)

MikoshiLang v3.0+ includes 28 extended modules covering specialized scientific domains:

- **Extended6-9** (552): Vector calculus, ML/time series, classical/quantum physics, biology/numerical methods
- **Extended10-12** (600): Special functions, audio/signal processing, graph theory/quantum computing
- **Extended13-16** (900): Computer science algorithms, astronomy/astrophysics, advanced statistics, NLP
- **Extended17-20** (900): Engineering (mechanical/electrical/civil), biology/genetics, finance/economics, fluid dynamics
- **Extended21-24** (600): Climate science, medical sciences, robotics, environmental science
- **Extended25-28** (600): Materials science, optics/photonics, aerospace engineering, nuclear/particle physics
- **Extended29-32** (600): Information theory/cryptography, acoustics/audio, geospatial/GIS, chemical engineering
- **Extended33** (100): Agriculture, food science, sports science, textile/petroleum engineering

---

**Total:** 6,324 functions across 40+ domains — comprehensive computational breadth matching Wolfram Mathematica.

## License

Apache 2.0 — Mikoshi Ltd.
