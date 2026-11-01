"""Differential tests for coverage batch W26 (numerics: NDSolve extensions, numerical options and PDE modelling
components).

Reference values (cited per block):

* Wolfram documentation (reference.wolfram.com/language/ref/<Name>.html): the documented outputs of the PDE terms,
  components and boundary heads (e.g. DiffusionPDETerm[{u[x], {x}}, 1] = Inactive[Div][{{-1}} . Inactive[Grad][u[x],
  {x}], {x}], PoissonPDEComponent[{u[x], {x}}, <||>] = -1 + Inactive[Div][{{1}} . Inactive[Grad][u[x], {x}], {x}]),
  the NDEigensystem examples (Laplacian spectra on [0, Pi], the Robin example -Laplacian + NeumannValue[u, x == 1]
  -> 0.740174, the harmonic oscillator h (2 n + 1), the wave equation eigenvalues +-I, +-2 I) and the WhenEvent
  bouncing-ball example;
* textbook closed forms: Sturm-Liouville spectra (n^2 Dirichlet / Neumann / periodic on [0, Pi] or [0, 2 Pi], the
  Bloch spectrum (n + alpha/(2 Pi))^2, (m^2 + n^2) on [0, Pi]^2), Poisson problems (x (1 - x)/2, the square torsion
  value 0.0736713532815 from the series 4 sum 1/(n^3 Pi^3) ...), Hooke's law and the Navier-Cauchy equations
  (Timoshenko & Goodier), incompressible Navier-Stokes, von Mises stress (|s| uniaxial, Sqrt[3] |t| shear), CODATA
  exact constants (mu0 = 4 Pi 10^-7 before 2019 as used by Wolfram, h = 6.62607015 10^-34, Stefan-Boltzmann
  2 Pi^5 k^4/(15 h^3 c^2));
* SciPy (computed in this file, independent of MikoshiLang): solve_bvp for boundary-value problems, closed-form
  exponentials for ParametricNDSolve.

Operators are returned as Inactive forms, so most component checks compare Activate[...] with the written-out
equation.  No test assigns a plain global (w26-prefixed names only).
"""
import importlib.util

import numpy as np
import pytest

from _harness import Pred, Raises, check, ev, ids, to_complex
from mikoshilang.expr import Expr

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def Near(want, tol=1e-9):
    return Pred(lambda g: abs(to_complex(g) - complex(want)) <= tol * max(1, abs(complex(want))), f"~{want}")


def NearList(want, tol=1e-8):
    def ok(g):
        return (getattr(g, "head", None) == "List" and len(g.args) == len(want)
                and all(abs(to_complex(a) - complex(w)) <= tol * max(1, abs(complex(w))) for a, w in zip(g.args, want)))
    return Pred(ok, f"~{want}")


def Head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"head {h}")


V3 = "{{u[x, y, z], v[x, y, z], w[x, y, z]}, {x, y, z}}"
YN = '<|"YoungModulus" -> Y, "PoissonRatio" -> nu|>'
LM = '<|"LameParameter" -> l, "ShearModulus" -> m|>'
FL2 = "{{u[x, y], v[x, y], p[x, y]}, {x, y}}"

CASES = [
    # ── PDE terms (Wolfram documentation outputs) ──
    ("DiffusionPDETerm[{u[x], {x}}, 1]", "Inactive[Div][{{-1}} . Inactive[Grad][u[x], {x}], {x}]"),
    ("DiffusionPDETerm[{u[x, y], {x, y}}, {{1, 2}, {3, 4}}]",
     "Inactive[Div][{{-1, -2}, {-3, -4}} . Inactive[Grad][u[x, y], {x, y}], {x, y}]"),
    ("DiffusionPDETerm[{u[x], {x}}, c]", "Inactive[Div][-c Inactive[Grad][u[x], {x}], {x}]"),
    ("Activate[DiffusionPDETerm[{u[x, y], {x, y}}, k]]", "-k D[u[x, y], {x, 2}] - k D[u[x, y], {y, 2}]"),
    ("DiffusionPDETerm[w26sym, 1]", "DiffusionPDETerm[w26sym, 1]"),
    ("DiffusionPDETerm[{u[x], {x}}, \"a\"]", Raises(TypeError)),
    ("DiffusionPDETerm[{u[x], {x}}, {1, 2}]", Raises(ValueError)),
    ("LaplacianPDETerm[{u[x], {x}}]", "Inactive[Div][{{1}} . Inactive[Grad][u[x], {x}], {x}]"),
    ("Activate[LaplacianPDETerm[{u[x, y], {x, y}}]]", "Laplacian[u[x, y], {x, y}]"),
    ("Activate[LaplacianPDETerm[{u[x, y, z], {x, y, z}}]] - Laplacian[u[x, y, z], {x, y, z}]", "0"),
    ("LaplacianPDETerm[{u[x], {x}}, <|\"Foo\" -> 1|>]", Raises(ValueError)),
    ("ConvectionPDETerm[{u[x], {x}}, {2}]", "{2} . Inactive[Grad][u[x], {x}]"),
    ("ConvectionPDETerm[{u[x, y], {x, y}}, {b1, b2}]", "{b1, b2} . Inactive[Grad][u[x, y], {x, y}]"),
    ("Activate[ConvectionPDETerm[{u[x, y], {x, y}}, {b1, b2}]]", "b1 D[u[x, y], x] + b2 D[u[x, y], y]"),
    ("ConvectionPDETerm[{u[x, y], {x, y}}, {1, 2, 3}]", Raises(ValueError)),
    ("ConservativeConvectionPDETerm[{u[x], {x}}, {2}]", "Inactive[Div][{-2 u[x]}, {x}]"),
    ("Activate[ConservativeConvectionPDETerm[{u[x], {x}}, {a[x]}]]", "-D[a[x] u[x], x]"),
    ("Activate[ConservativeConvectionPDETerm[{u[x, y], {x, y}}, {a1, a2}]]", "-a1 D[u[x, y], x] - a2 D[u[x, y], y]"),
    ("ConservativeConvectionPDETerm[{u[x], {x}}, 2]", Raises(ValueError)),
    ("DerivativePDETerm[{u[x, y], {x, y}}, {g1, g2}]", "Inactive[Div][{g1, g2}, {x, y}]"),
    ("Activate[DerivativePDETerm[{u[x], {x}}, {x^2}]]", "2 x"),
    ("Activate[DerivativePDETerm[{u[x, y], {x, y}}, {x y, y^2}]]", "3 y"),
    ("ReactionPDETerm[{u[x], {x}}, a]", "a u[x]"),
    ("ReactionPDETerm[{u[x, y], {x, y}}, 3]", "3 u[x, y]"),
    ("ReactionPDETerm[{u[t, x], t, {x}}, a]", "a u[t, x]"),
    ("SourcePDETerm[{u[x], {x}}, f]", "f"),
    ("SourcePDETerm[{u[x, y], {x, y}}, Sin[x]]", "Sin[x]"),
    ("SourcePDETerm[{u[x], {x}}, f, <|\"RegionSymmetry\" -> \"Spherical\"|>]", Raises((ValueError, NotImplementedError))),
    # ── PDE components ──
    # (the kernel keeps Plus with Inactive operands in construction order; Wolfram prints -1 + ...)
    ("PoissonPDEComponent[{u[x], {x}}, <||>]", "Inactive[Div][{{1}} . Inactive[Grad][u[x], {x}], {x}] - 1"),
    ("Activate[PoissonPDEComponent[{u[x, y], {x, y}}, <|\"PoissonSourceTerm\" -> f|>]]",
     "D[u[x, y], {x, 2}] + D[u[x, y], {y, 2}] - f"),
    ("PoissonPDEComponent[{u[x], {x}}, <|\"Source\" -> 1|>]", Raises(ValueError)),
    ("Activate[HelmholtzPDEComponent[{u[x], {x}}, <|\"HelmholtzEigenvalue\" -> k|>]]", "u''[x] + k^2 u[x]"),
    ("Activate[HelmholtzPDEComponent[{u[x, y], {x, y}}, <||>]]", "D[u[x, y], {x, 2}] + D[u[x, y], {y, 2}] + u[x, y]"),
    ("HelmholtzPDEComponent[{u[x], {x}}, <|\"Wavenumber\" -> 2|>]", Raises(ValueError)),
    ("Activate[WavePDEComponent[{u[t, x], t, {x}}, <|\"WaveCoefficient\" -> c|>]]",
     "D[u[t, x], {t, 2}] - c^2 D[u[t, x], {x, 2}]"),
    ("Activate[WavePDEComponent[{u[t, x, y], t, {x, y}}, <||>]]",
     "D[u[t, x, y], {t, 2}] - D[u[t, x, y], {x, 2}] - D[u[t, x, y], {y, 2}]"),
    ("WavePDEComponent[{u[x], {x}}, <||>]", Raises(ValueError)),
    ("Activate[HeatTransferPDEComponent[{T[t, x], t, {x}}, <|\"ThermalConductivity\" -> k, \"MassDensity\" -> rho, "
     "\"SpecificHeatCapacity\" -> cp|>]]", "rho cp D[T[t, x], t] - k D[T[t, x], {x, 2}]"),
    ("Activate[HeatTransferPDEComponent[{T[x, y], {x, y}}, <|\"ThermalConductivity\" -> k, \"HeatSource\" -> q|>]]",
     "-k D[T[x, y], {x, 2}] - k D[T[x, y], {y, 2}] - q"),
    ("Activate[HeatTransferPDEComponent[{T[x], {x}}, <|\"ThermalConductivity\" -> 1, \"MassDensity\" -> rho, "
     "\"SpecificHeatCapacity\" -> cp, \"HeatConvectionVelocity\" -> {v}|>]]", "rho cp v T'[x] - T''[x]"),
    ("Activate[HeatTransferPDEComponent[{T[t, x], t, {x}}, <|\"ThermalConductivity\" -> 1|>]]",
     "D[T[t, x], t] - D[T[t, x], {x, 2}]"),
    ("HeatTransferPDEComponent[{T[x], {x}}, <|\"Material\" -> \"Copper\"|>]",
     Raises(NotImplementedError)),
    ("HeatTransferPDEComponent[{T[x], {x}}, <|\"Conductivity\" -> 1|>]", Raises(ValueError)),
    ("Activate[MassTransportPDEComponent[{c[t, x], t, {x}}, <|\"DiffusionCoefficient\" -> d, "
     "\"MassConvectionVelocity\" -> {v}|>]]", "D[c[t, x], t] - d D[c[t, x], {x, 2}] + v D[c[t, x], x]"),
    ("Activate[MassTransportPDEComponent[{c[x], {x}}, <|\"DiffusionCoefficient\" -> d, \"MassReactionRate\" -> a, "
     "\"MassSource\" -> f|>]]", "-d c''[x] + a c[x] - f"),
    ("Activate[MassTransportPDEComponent[{c[x], {x}}, <|\"DiffusionCoefficient\" -> 1, \"MassConvectionVelocity\" -> "
     "{v[x]}, \"ModelForm\" -> \"Conservative\"|>]]", "-c''[x] + D[v[x] c[x], x]"),
    ("MassTransportPDEComponent[{c[x], {x}}, <|\"DiffusionCoefficient\" -> 1, \"ModelForm\" -> \"Foo\"|>]",
     Raises(ValueError)),
    ("Activate[AcousticPDEComponent[{p[t, x], t, {x}}, <|\"SoundSpeed\" -> c, \"MassDensity\" -> rho|>]]",
     "D[p[t, x], {t, 2}]/(c^2 rho) - D[p[t, x], {x, 2}]/rho"),
    ("Activate[AcousticPDEComponent[{p[x], om, {x}}, <|\"SoundSpeed\" -> c, \"MassDensity\" -> rho|>]]",
     "-p''[x]/rho - om^2 p[x]/(c^2 rho)"),
    ("Activate[AcousticPDEComponent[{p[x, y], {x, y}}, <|\"SoundSpeed\" -> 1, \"MassDensity\" -> 1|>]]",
     "-D[p[x, y], {x, 2}] - D[p[x, y], {y, 2}]"),
    ("Activate[AcousticPDEComponent[{p[t, x], t, {x}}, <|\"SoundSpeed\" -> c|>]]",
     "D[p[t, x], {t, 2}]/c^2 - D[p[t, x], {x, 2}]"),
    ("AcousticPDEComponent[{p[t, x], t, {x}}, <|\"Speed\" -> c|>]", Raises(ValueError)),
    ("Activate[ElectrostaticPDEComponent[{V[x], {x}}, <|\"RelativePermittivity\" -> 2, \"VacuumPermittivity\" -> e0|>]]",
     "-2 e0 V''[x]"),
    ("Activate[ElectrostaticPDEComponent[{V[x, y], {x, y}}, <|\"VacuumPermittivity\" -> 1, \"VolumeChargeDensity\" -> "
     "q|>]] ", "-D[V[x, y], {x, 2}] - D[V[x, y], {y, 2}] - q"),
    ("ElectrostaticPDEComponent[{V[x], {x}}, <|\"RelativePermittivity\" -> 2|>] /. V -> Function[x, x^2] // Activate",
     "-4 625000/(22468879468420441 Pi)"),
    ("ElectrostaticPDEComponent[{V[t, x], t, {x}}, <||>]", Raises((ValueError, NotImplementedError))),
    ("ElectricCurrentPDEComponent[{V[x], {x}}, <|\"ElectricalConductivity\" -> s|>]",
     "Inactive[Div][-s Inactive[Grad][V[x], {x}], {x}]"),
    ("Activate[ElectricCurrentPDEComponent[{V[x, y], {x, y}}, <|\"ElectricalConductivity\" -> s, \"CurrentSource\" -> "
     "q|>]]", "-s D[V[x, y], {x, 2}] - s D[V[x, y], {y, 2}] - q"),
    ("Activate[ElectricCurrentPDEComponent[{V[x], {x}}, <|\"ElectricalConductivity\" -> {{s}}|>]]", "-s V''[x]"),
    ("ElectricCurrentPDEComponent[{V[x], {x}}, <|\"Conductivity\" -> 1|>]", Raises(ValueError)),
    ("MagnetostaticPDEComponent[{A[x, y], {x, y}}, <|\"RelativePermeability\" -> 2|>]",
     "Inactive[Div][{{-Pi/1250000, 0}, {0, -Pi/1250000}} . Inactive[Grad][A[x, y], {x, y}], {x, y}]"),
    ("Activate[MagnetostaticPDEComponent[{Vm[x], {x}}, <|\"VacuumPermeability\" -> m0, \"RelativePermeability\" -> "
     "mr|>]]", "-m0 mr Vm''[x]"),
    ("Activate[MagnetostaticPDEComponent[{Vm[x], {x}}, <|\"VacuumPermeability\" -> 1, \"Magnetization\" -> {M[x]}|>]]",
     "-Vm''[x] + M'[x]"),
    ("MagnetostaticPDEComponent[{Vm[t, x], t, {x}}, <||>]", Raises((ValueError, NotImplementedError))),
    ("Activate[MagneticPDEComponent[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, "
     "<|\"VacuumPermeability\" -> 1|>]] == Curl[Curl[{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}], {x, y, z}]",
     "True"),
    ("Activate[MagneticPDEComponent[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, <|\"VacuumPermeability\" -> "
     "1, \"ExternalCurrentSource\" -> {0, 0, J}|>]][[3]] == Curl[Curl[{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, "
     "{x, y, z}], {x, y, z}][[3]] - J", "True"),
    ("Length[MagneticPDEComponent[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, <||>]]", "3"),
    ("MagneticPDEComponent[{{Ax[x, y], Ay[x, y]}, {x, y}}, <||>]", Raises((ValueError, NotImplementedError))),
    ("Activate[SchrodingerPDEComponent[{psi[t, x], t, {x}}, <|\"Mass\" -> m, \"ReducedPlanckConstant\" -> hb|>]]",
     "-I hb D[psi[t, x], t] - hb^2 D[psi[t, x], {x, 2}]/(2 m)"),
    ("Activate[SchrodingerPDEComponent[{psi[x], {x}}, <|\"Mass\" -> 1, \"ReducedPlanckConstant\" -> 1, "
     "\"SchrodingerPotential\" -> x^2|>]]", "-psi''[x]/2 + x^2 psi[x]"),
    ("SchrodingerPDEComponent[{psi[x], {x}}, <|\"Mass\" -> 1|>] /. psi -> Function[x, 1] // Activate", "0"),
    ("SchrodingerPDEComponent[{psi[x], {x}}, <|\"Mass\" -> 1, \"MagneticVectorPotential\" -> {1}|>]",
     Raises(NotImplementedError)),
]
CASES += [
    # ── heat boundary heads (Wolfram documentation forms; positive NeumannValue = flux into the domain) ──
    ("HeatFluxValue[x == 1, {T[x], {x}}, <|\"HeatFlux\" -> q|>]", "NeumannValue[q, x == 1]"),
    ("HeatFluxValue[x == 1, {T[x, y], {x, y}}, <|\"HeatFlux\" -> 5, \"Thickness\" -> 2|>]",
     Raises((ValueError, NotImplementedError))),
    ("HeatFluxValue[x == 1, {T[x], {x}}, <|\"Flux\" -> q|>]", Raises(ValueError)),
    ("HeatFluxValue[x == 1, {T[x], {x}}, <|\"HeatFlux\" -> 2|>, \"w26missing\"]", Raises(ValueError)),
    ("HeatInsulationValue[x == 1, {T[x], {x}}, <||>]", "NeumannValue[0, x == 1]"),
    ("HeatInsulationValue[x == 0 || x == 1, {T[x], {x}}, <||>]", "NeumannValue[0, x == 0 || x == 1]"),
    ("HeatInsulationValue[True, {T[x, y], {x, y}}, <||>]", "NeumannValue[0, True]"),
    ("HeatOutflowValue[x == 1, {T[x], {x}}, <|\"ThermalConductivity\" -> k|>]", "NeumannValue[0, x == 1]"),
    ("HeatOutflowValue[y == 1, {T[x, y], {x, y}}, <||>]", "NeumannValue[0, y == 1]"),
    ("HeatOutflowValue[x == 1, {T[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("HeatSymmetryValue[x == 0, {T[x], {x}}, <||>]", "NeumannValue[0, x == 0]"),
    ("HeatSymmetryValue[x == 0, {T[x, y], {x, y}}, <||>]", "NeumannValue[0, x == 0]"),
    ("HeatSymmetryValue[x == 0, {T[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("HeatTransferValue[x == 1, {T[x], {x}}, <|\"AmbientTemperature\" -> Ta, \"HeatTransferCoefficient\" -> h|>]",
     "NeumannValue[h (Ta - T[x]), x == 1]"),
    ("HeatTransferValue[x == 1, {T[t, x], t, {x}}, <|\"AmbientTemperature\" -> 300, \"HeatTransferCoefficient\" -> "
     "10|>]", "NeumannValue[10 (300 - T[t, x]), x == 1]"),
    ("HeatTransferValue[x == 1, {T[x], {x}}, <|\"AmbientTemperature\" -> 300|>]", Raises(ValueError)),
    # Stefan-Boltzmann constant 2 Pi^5 k^4/(15 h^3 c^2) with exact SI k, h, c (CODATA 2018)
    ("HeatRadiationValue[x == 1, {T[x], {x}}, <|\"AmbientTemperature\" -> Ta, \"Emissivity\" -> e|>] === "
     "NeumannValue[e 2 Pi^5 (1380649 10^-29)^4/(15 (662607015 10^-42)^3 299792458^2) (Ta^4 - T[x]^4), x == 1]",
     "True"),
    ("N[HeatRadiationValue[x == 1, {T[x], {x}}, <|\"AmbientTemperature\" -> 0, \"Emissivity\" -> 1|>][[1]] /. "
     "T[x] -> 1]", -5.670374419e-8, 1e-9),
    ("HeatRadiationValue[x == 1, {T[x], {x}}, <|\"AmbientTemperature\" -> Ta, \"Emissivity\" -> e, "
     "\"BoltzmannConstant\" -> s|>]", "NeumannValue[e s (Ta^4 - T[x]^4), x == 1]"),
    ("HeatRadiationValue[x == 1, {T[x], {x}}, <|\"Emissivity\" -> e, \"StefanBoltzmannConstant\" -> s|>]",
     Raises(ValueError)),
    ("HeatTemperatureCondition[x == 0, {T[x], {x}}, <|\"SurfaceTemperature\" -> 300|>]",
     "DirichletCondition[T[x] == 300, x == 0]"),
    ("HeatTemperatureCondition[x == 0, {T[t, x], t, {x}}, <|\"SurfaceTemperature\" -> 300 + t|>]",
     "DirichletCondition[T[t, x] == 300 + t, x == 0]"),
    ("HeatTemperatureCondition[x == 0, {T[x], {x}}, <|\"BC1\" -> <|\"SurfaceTemperature\" -> 7|>|>, \"BC1\"]",
     "DirichletCondition[T[x] == 7, x == 0]"),
    # ── mass transport boundary heads ──
    ("MassFluxValue[x == 1, {c[x], {x}}, <|\"MassFlux\" -> q|>]", "NeumannValue[q, x == 1]"),
    ("MassFluxValue[x >= 0, {c[t, x], t, {x}}, <|\"MassConvectionVelocity\" -> {v}, \"MassFlux\" -> q[t, x]|>]",
     "NeumannValue[c[t, x] NeumannBoundaryUnitNormal[{x}] . {v} + q[t, x], x >= 0]"),
    ("MassFluxValue[x == 1, {c[x], {x}}, <|\"Flux\" -> 1|>]", Raises(ValueError)),
    ("MassImpermeableBoundaryValue[x == 1, {c[x], {x}}, <||>]", "NeumannValue[0, x == 1]"),
    ("MassImpermeableBoundaryValue[True, {c[x, y], {x, y}}, <||>]", "NeumannValue[0, True]"),
    ("MassImpermeableBoundaryValue[x == 1, {c[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("MassOutflowValue[x == 1, {c[x], {x}}, <|\"DiffusionCoefficient\" -> d|>]", "NeumannValue[0, x == 1]"),
    ("MassOutflowValue[x == 1, {c[x], {x}}, <|\"MassConvectionVelocity\" -> {v}, \"BoundaryUnitNormal\" -> {1}, "
     "\"ModelForm\" -> \"Conservative\"|>]", "NeumannValue[-v c[x], x == 1]"),
    ("MassOutflowValue[x == 1, {c[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("MassSymmetryValue[x == 1, {c[x], {x}}, <||>]", "NeumannValue[0, x == 1]"),
    ("MassSymmetryValue[x == 1, {c[x, y], {x, y}}, <||>]", "NeumannValue[0, x == 1]"),
    ("MassSymmetryValue[x == 1, {c[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("MassTransferValue[x == 1, {c[x], {x}}, <|\"AmbientConcentration\" -> ca, \"MassTransferCoefficient\" -> h|>]",
     "NeumannValue[h (ca - c[x]), x == 1]"),
    ("MassTransferValue[x == 1, {c[x], {x}}, <|\"AmbientConcentration\" -> 1, \"MassTransferCoefficient\" -> 2|>]",
     "NeumannValue[2 (1 - c[x]), x == 1]"),
    # documented defaults: "AmbientConcentration" 0, "MassTransferCoefficient" 1
    ("MassTransferValue[x == 1, {c[x], {x}}, <||>]", "NeumannValue[-c[x], x == 1]"),
    ("MassTransferValue[x == 1, {c[x], {x}}, <|\"TransferCoefficient\" -> 1|>]", Raises(ValueError)),
    ("MassConcentrationCondition[x == 0, {c[x], {x}}, <|\"MassConcentration\" -> 2|>]",
     "DirichletCondition[c[x] == 2, x == 0]"),
    ("MassConcentrationCondition[x == 0, {c[t, x], t, {x}}, <|\"MassConcentration\" -> Exp[-t]|>]",
     "DirichletCondition[c[t, x] == Exp[-t], x == 0]"),
    ("MassConcentrationCondition[x == 0, {c[x], {x}}, <|\"Concentration\" -> 2|>]", Raises(ValueError)),
    # ── acoustic boundary heads ──
    ("AcousticSoundHardValue[x == 1, {p[t, x], t, {x}}, <||>]", "NeumannValue[0, x == 1]"),
    ("AcousticSoundHardValue[x == 1, {p[x], om, {x}}, <||>]", "NeumannValue[0, x == 1]"),
    ("AcousticSoundHardValue[x == 1, {p[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("AcousticSoundSoftCondition[x == 1, {p[t, x], t, {x}}, <||>]", "DirichletCondition[p[t, x] == 0, x == 1]"),
    ("AcousticSoundSoftCondition[True, {p[x, y], om, {x, y}}, <||>]", "DirichletCondition[p[x, y] == 0, True]"),
    ("AcousticSoundSoftCondition[x == 1, {p[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("AcousticPressureCondition[x == 0, {p[t, x], t, {x}}, <|\"AcousticPressure\" -> Sin[t]|>]",
     "DirichletCondition[p[t, x] == Sin[t], x == 0]"),
    ("AcousticPressureCondition[x == 0, {p[x], om, {x}}, <|\"AcousticPressure\" -> 2|>]",
     "DirichletCondition[p[x] == 2, x == 0]"),
    ("AcousticPressureCondition[x == 0, {p[x], {x}}, <|\"Pressure\" -> 2|>]", Raises(ValueError)),
    ("AcousticNormalVelocityValue[x == 0, {p[t, x], t, {x}}, <|\"AcousticSoundParticleVelocity\" -> v0|>]",
     "NeumannValue[0, x == 0]"),
    ("AcousticNormalVelocityValue[x == 0, {p[t, x], t, {x}}, <|\"AcousticSoundParticleVelocity\" -> Sin[t]|>]",
     "NeumannValue[Cos[t], x == 0]"),
    ("AcousticNormalVelocityValue[x == 0, {p[x], om, {x}}, <|\"AcousticSoundParticleVelocity\" -> v0|>]",
     "NeumannValue[I om v0, x == 0]"),
    ("AcousticNormalVelocityValue[x == 0, {p[x], {x}}, <|\"Velocity\" -> 1|>]", Raises(ValueError)),
    ("AcousticImpedanceValue[x == 1, {p[t, x], t, {x}}, <|\"SpecificAcousticImpedance\" -> Z, \"MassDensity\" -> "
     "rho|>]", "NeumannValue[-D[p[t, x], t]/Z, x == 1]"),
    ("AcousticImpedanceValue[x == 1, {p[t, x], t, {x}}, <|\"SpecificAcousticImpedance\" -> 2|>]",
     "NeumannValue[-D[p[t, x], t]/2, x == 1]"),
    ("AcousticImpedanceValue[x == 1, {p[t, x], t, {x}}, <|\"Impedance2\" -> 2|>]", Raises(ValueError)),
    # plane-wave absorbing condition: impedance rho c
    ("AcousticAbsorbingValue[x == 1, {p[t, x], t, {x}}, <|\"SoundSpeed\" -> c, \"MassDensity\" -> rho|>]",
     "NeumannValue[-D[p[t, x], t]/(c rho), x == 1]"),
    ("AcousticAbsorbingValue[x == 1, {p[t, x], t, {x}}, <|\"SoundSpeed\" -> 343, \"MassDensity\" -> 6/5|>]",
     "NeumannValue[-D[p[t, x], t]/(2058/5), x == 1]"),
    ("AcousticAbsorbingValue[x == 1, {p[t, x], t, {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("AcousticRadiationValue[x == 1, {p[t, x], t, {x}}, <|\"SoundSpeed\" -> c, \"MassDensity\" -> rho|>]",
     "NeumannValue[-D[p[t, x], t]/(c rho), x == 1]"),
    ("AcousticRadiationValue[x == 1, {p[t, x], t, {x}}, <|\"SoundSpeed\" -> 1, \"MassDensity\" -> 1|>]",
     "NeumannValue[-D[p[t, x], t], x == 1]"),
    ("AcousticRadiationValue[x == 1, {p[t, x], t, {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    # ── electromagnetic boundary heads ──
    ("ElectricPotentialCondition[x == 0, {V[x], {x}}, <||>, <|\"ElectricPotential\" -> 1|>]",
     "DirichletCondition[V[x] == 1, x == 0]"),
    ("ElectricPotentialCondition[x == 0, {V[x], {x}}, <|\"ElectricPotential\" -> 5|>]",
     "DirichletCondition[V[x] == 5, x == 0]"),
    ("ElectricPotentialCondition[x == 0, {V[x], {x}}, <|\"Potential\" -> 1|>]", Raises(ValueError)),
    ("ElectricSymmetryValue[x == 1, {V[x], {x}}, <||>]", "NeumannValue[0, x == 1]"),
    ("ElectricSymmetryValue[y == 0, {V[x, y], {x, y}}, <||>]", "NeumannValue[0, y == 0]"),
    ("ElectricSymmetryValue[x == 1, {V[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("ElectricFluxDensityValue[x == 1, {V[x], {x}}, <|\"ElectricFluxDensity\" -> {d}|>]",
     "NeumannValue[-NeumannBoundaryUnitNormal[{x}] . {d}, x == 1]"),
    ("ElectricFluxDensityValue[x == 1, {V[x], {x}}, <|\"ElectricFluxDensity\" -> {d}, \"BoundaryUnitNormal\" -> {1}|>]",
     "NeumannValue[-d, x == 1]"),
    ("ElectricFluxDensityValue[x == 1, {V[x], {x}}, <|\"Flux\" -> 1|>]", Raises(ValueError)),
    ("ElectricCurrentDensityValue[x == 1, {V[x], {x}}, <|\"CurrentDensity\" -> {j}|>]",
     "NeumannValue[-NeumannBoundaryUnitNormal[{x}] . {j}, x == 1]"),
    ("ElectricCurrentDensityValue[x == 1, {V[x], {x}}, <|\"CurrentDensity\" -> {j}, \"BoundaryUnitNormal\" -> {1}|>]",
     "NeumannValue[-j, x == 1]"),
    ("ElectricCurrentDensityValue[x == 1, {V[x], {x}}, <|\"Current\" -> 1|>]", Raises(NotImplementedError)),
    ("MagneticFluxDensityValue[x == 1, {Vm[x], {x}}, <|\"MagneticFluxDensity\" -> {b}|>]",
     "NeumannValue[-NeumannBoundaryUnitNormal[{x}] . {b}, x == 1]"),
    ("MagneticFluxDensityValue[x == 1, {Vm[x, y], {x, y}}, <|\"MagneticFluxDensity\" -> {b1, b2}, "
     "\"BoundaryUnitNormal\" -> {1, 0}|>]", "NeumannValue[-b1, x == 1]"),
    ("MagneticFluxDensityValue[x == 1, {Vm[x], {x}}, <|\"Flux\" -> 1|>]", Raises(ValueError)),
    ("MagneticSymmetryValue[x == 1, {Vm[x], {x}}, <||>]", "NeumannValue[0, x == 1]"),
    ("MagneticSymmetryValue[x == 1, {Vm[x, y], {x, y}}, <||>]", "NeumannValue[0, x == 1]"),
    ("MagneticSymmetryValue[x == 1, {Vm[x], {x}}, <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("MagneticPotentialCondition[x == 0, {Vm[x], {x}}, <|\"MagneticPotential\" -> 0|>]",
     "DirichletCondition[Vm[x] == 0, x == 0]"),
    ("MagneticPotentialCondition[x == 0, {{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, "
     "<|\"MagneticPotential\" -> {0, 0, 1}|>]",
     "DirichletCondition[{Ax[x, y, z] == 0, Ay[x, y, z] == 0, Az[x, y, z] == 1}, x == 0]"),
    ("MagneticPotentialCondition[x == 0, {Vm[x], {x}}, <|\"Potential\" -> 0|>]", Raises(ValueError)),
    # NeumannBoundaryUnitNormal: inert outward normal kept symbolic unless "BoundaryUnitNormal" is given
    ("FreeQ[MassFluxValue[x == 1, {c[x, y], {x, y}}, <|\"MassConvectionVelocity\" -> {v1, v2}|>], "
     "NeumannBoundaryUnitNormal]", "False"),
    ("MassFluxValue[x == 1, {c[x, y], {x, y}}, <|\"MassConvectionVelocity\" -> {v1, v2}, \"BoundaryUnitNormal\" -> "
     "{1, 0}|>]", "NeumannValue[v1 c[x, y], x == 1]"),
    ("NeumannBoundaryUnitNormal[{x, y}]", "NeumannBoundaryUnitNormal[{x, y}]"),
    ("ElectricFluxDensityValue[x == 1, {V[x], {x}}, <|\"ElectricFluxDensity\" -> {d}|>] /. "
     "NeumannBoundaryUnitNormal[{x}] -> {1}", "NeumannValue[-d, x == 1]"),
    ("Head[NeumannBoundaryUnitNormal[{x}]]", "NeumannBoundaryUnitNormal"),
    # ── solid mechanics (Hooke's law, Navier-Cauchy; Wolfram documentation torsion example) ──
    ("Map[Simplify, Activate[SolidMechanicsPDEComponent[" + V3 + ", " + LM + "]] + (l + m) Grad[Div[{u[x, y, z], "
     "v[x, y, z], w[x, y, z]}, {x, y, z}], {x, y, z}] + m {Laplacian[u[x, y, z], {x, y, z}], Laplacian[v[x, y, z], "
     "{x, y, z}], Laplacian[w[x, y, z], {x, y, z}]}]", "{0, 0, 0}"),
    # the u-term's coefficient matrix, found by pattern so the case does not depend on the order of the Plus terms
    ("Simplify[First[Cases[SolidMechanicsPDEComponent[" + V3 + ", " + YN + "][[1]], Inactive[Div][Dot[m_, "
     "Inactive[Grad][u[x, y, z], _]], _] :> m[[1, 1]]]] + (Y (1 - nu))/((1 + nu) (1 - 2 nu))]", "0"),
    ("Simplify[Activate[SolidMechanicsPDEComponent[{{u[t, x, y, z], v[t, x, y, z], w[t, x, y, z]}, t, {x, y, z}}, "
     "<|\"LameParameter\" -> l, \"ShearModulus\" -> m, \"MassDensity\" -> rho|>]][[1]] - rho D[u[t, x, y, z], {t, 2}] "
     "+ (l + m) D[Div[{u[t, x, y, z], v[t, x, y, z], w[t, x, y, z]}, {x, y, z}], x] + m Laplacian[u[t, x, y, z], "
     "{x, y, z}]]", "0"),
    ("Activate[SolidMechanicsPDEComponent[" + V3 + ", <|\"LameParameter\" -> 0, \"ShearModulus\" -> 1, "
     "\"BodyLoad\" -> {0, 0, -g}|>]][[3]] /. {w -> Function[{x, y, z}, 0], u -> Function[{x, y, z}, 0], "
     "v -> Function[{x, y, z}, 0]}", "g"),
    ("SolidMechanicsPDEComponent[{{u[x, y], v[x, y]}, {x, y}}, " + YN + "]", Raises(NotImplementedError)),
    ("SolidMechanicsPDEComponent[" + V3 + ", <|\"YoungModulus\" -> Y|>]", Raises(ValueError)),
    ("SolidMechanicsPDEComponent[" + V3 + ", <|\"YoungModulus\" -> Y, \"PoissonRatio\" -> nu, "
     "\"SolidMechanicsMaterialModel\" -> \"Hyperelastic\"|>]", Raises(NotImplementedError)),
    ("Normal[SolidMechanicsStrain[" + V3 + ", <||>, {-a y z, a x z, w[x, y]}]]",
     "{{0, 0, -a y + D[w[x, y], x]}, {0, 0, a x + D[w[x, y], y]}, {-a y + D[w[x, y], x], a x + D[w[x, y], y], 0}}"),
    ("Normal[SolidMechanicsStrain[" + V3 + ", <|\"EngineeringStrain\" -> False|>, {x^2, 0, 0}]]",
     "{{2 x, 0, 0}, {0, 0, 0}, {0, 0, 0}}"),
    ("Normal[SolidMechanicsStrain[" + V3 + ", <|\"EngineeringStrain\" -> False|>, {y, x, 0}]]",
     "{{0, 1, 0}, {1, 0, 0}, {0, 0, 0}}"),
    ("Head[SolidMechanicsStrain[" + V3 + ", <||>, {x, y, z}]]", "SymmetrizedArray"),
    ("SolidMechanicsStrain[" + V3 + ", <||>, {x, y}]", Raises(ValueError)),
    ("Normal[SolidMechanicsStress[" + V3 + ", " + YN + ", {{0, 0, g}, {0, 0, 0}, {g, 0, 0}}]]",
     "{{0, 0, Y g/(2 (1 + nu))}, {0, 0, 0}, {Y g/(2 (1 + nu)), 0, 0}}"),
    ("Map[Simplify, Normal[SolidMechanicsStress[" + V3 + ", " + LM + ", {{e, 0, 0}, {0, e, 0}, {0, 0, e}}]], {2}]",
     "{{(3 l + 2 m) e, 0, 0}, {0, (3 l + 2 m) e, 0}, {0, 0, (3 l + 2 m) e}}"),
    # uniaxial stress state: sigma_xx = E eps_xx when the lateral strains are -nu eps_xx
    ("Simplify[Normal[SolidMechanicsStress[" + V3 + ", " + YN + ", {{e, 0, 0}, {0, -nu e, 0}, {0, 0, -nu e}}]][[1, 1]]]",
     "Y e"),
    ("SolidMechanicsStress[" + V3 + ", " + YN + ", {{1, 2, 0}, {0, 1, 0}, {0, 0, 1}}]", Raises(ValueError)),
    ("SolidMechanicsStress[" + V3 + ", " + YN + ", w26strain]", "SolidMechanicsStress[" + V3 + ", " + YN + ", w26strain]"),
    ("VonMisesStress[" + V3 + ", <||>, {{s, 0, 0}, {0, 0, 0}, {0, 0, 0}}]", "Sqrt[s^2]"),
    ("VonMisesStress[" + V3 + ", <||>, {{0, t}, {t, 0}}]", "Sqrt[3] Sqrt[t^2]"),
    ("VonMisesStress[" + V3 + ", <||>, {{1, 2, 3}, {2, 4, 5}, {3, 5, 6}}]", "Sqrt[(9 + 4 + 25 + 6 (4 + 9 + 25))/2]"),
    ("VonMisesStress[" + V3 + ", <||>, {{p, 0, 0}, {0, p, 0}, {0, 0, p}}]", "0"),
    ("VonMisesStress[" + V3 + ", <||>, {{1, 2}, {3, 4}}]", Raises(ValueError)),
    ("EquivalentStrain[" + V3 + ", <||>, {{e, 0, 0}, {0, 0, 0}, {0, 0, 0}}]", "Sqrt[e^2]"),
    ("EquivalentStrain[" + V3 + ", <||>, SymmetrizedArray[{{0, 1, 0}, {1, 0, 0}, {0, 0, 0}}]]", "Sqrt[3]"),
    ("EquivalentStrain[" + V3 + ", <||>, {{1, 0}, {0, 1}}]", "1"),
    ("EquivalentStrain[" + V3 + ", <||>, {1, 2, 3}]", Raises(ValueError)),
    ("SolidDisplacementCondition[x == 0, " + V3 + ", <|\"Displacement\" -> {0, 0, None}|>]",
     "DirichletCondition[{u[x, y, z] == 0, v[x, y, z] == 0}, x == 0]"),
    ("SolidDisplacementCondition[x == 1, " + V3 + ", <|\"Displacement\" -> {d, 0, 0}|>]",
     "DirichletCondition[{u[x, y, z] == d, v[x, y, z] == 0, w[x, y, z] == 0}, x == 1]"),
    ("SolidDisplacementCondition[x == 1, " + V3 + ", <|\"Displacement\" -> {d, 0}|>]", Raises(ValueError)),
    ("SolidFixedCondition[x == 0, " + V3 + ", <||>]",
     "DirichletCondition[{u[x, y, z] == 0, v[x, y, z] == 0, w[x, y, z] == 0}, x == 0]"),
    ("SolidFixedCondition[x == 0 || x == 1, " + V3 + ", " + YN + "]",
     "DirichletCondition[{u[x, y, z] == 0, v[x, y, z] == 0, w[x, y, z] == 0}, x == 0 || x == 1]"),
    ("SolidFixedCondition[x == 0, " + V3 + ", <|\"Bogus\" -> 1|>]", Raises(ValueError)),
    ("SolidBoundaryLoadValue[x == 1, " + V3 + ", <|\"Pressure\" -> {0, 0, -p}|>]",
     "{NeumannValue[0, x == 1], NeumannValue[0, x == 1], NeumannValue[-p, x == 1]}"),
    ("SolidBoundaryLoadValue[x == 1, " + V3 + ", <|\"SpringConstant\" -> k|>]",
     "{NeumannValue[-k u[x, y, z], x == 1], NeumannValue[-k v[x, y, z], x == 1], NeumannValue[-k w[x, y, z], x == 1]}"),
    ("SolidBoundaryLoadValue[x == 1, {{u[t, x, y, z], v[t, x, y, z], w[t, x, y, z]}, t, {x, y, z}}, "
     "<|\"DampingConstant\" -> c|>][[1]]", "NeumannValue[-c D[u[t, x, y, z], t], x == 1]"),
    ("SolidBoundaryLoadValue[x == 1, " + V3 + ", <|\"Force\" -> {0, 0, 1}|>]", Raises(NotImplementedError)),
    ("SolidBoundaryLoadValue[x == 1, " + V3 + ", <|\"DampingConstant\" -> c|>]", Raises(ValueError)),
    # ── fluid flow (incompressible Navier-Stokes) ──
    ("Map[Simplify, Activate[FluidFlowPDEComponent[" + FL2 + ", <|\"DynamicViscosity\" -> mu, \"MassDensity\" -> rho|>]] - "
     "{rho D[u[x, y]^2, x] + rho D[u[x, y] v[x, y], y] + D[p[x, y], x] - mu (2 D[u[x, y], {x, 2}] + D[u[x, y], {y, 2}] "
     "+ D[v[x, y], x, y]), rho D[u[x, y] v[x, y], x] + rho D[v[x, y]^2, y] + D[p[x, y], y] - mu (D[v[x, y], {x, 2}] "
     "+ 2 D[v[x, y], {y, 2}] + D[u[x, y], x, y]), D[u[x, y], x] + D[v[x, y], y]}]", "{0, 0, 0}"),
    ("Simplify[Activate[FluidFlowPDEComponent[{{u[t, x, y], v[t, x, y], p[x, y]}, t, {x, y}}, <|\"ReynoldsNumber\" -> "
     "re, \"ModelForm\" -> \"NonConservative\"|>]][[1]] - (D[u[t, x, y], t] + u[t, x, y] D[u[t, x, y], x] + "
     "v[t, x, y] D[u[t, x, y], y] + D[p[x, y], x] - (2 D[u[t, x, y], {x, 2}] + D[u[t, x, y], {y, 2}] + "
     "D[v[t, x, y], x, y])/re)]", "0"),
    ("Length[FluidFlowPDEComponent[{{u[x, y, z], v[x, y, z], w[x, y, z], p[x, y, z]}, {x, y, z}}, "
     "<|\"ReynoldsNumber\" -> 10|>]]", "4"),
    ("FluidFlowPDEComponent[" + FL2 + ", <|\"DynamicViscosity\" -> mu|>]", Raises(ValueError)),
    ("FluidFlowPDEComponent[" + FL2 + ", <|\"DynamicViscosity\" -> mu, \"MassDensity\" -> rho[x, y]|>]",
     Raises(NotImplementedError)),
    ("FluidViscosity[" + FL2 + ", <|\"DynamicViscosity\" -> mu|>, {U[x, y], W[x, y]}]", "mu"),
    ("FluidViscosity[" + FL2 + ", <|\"ReynoldsNumber\" -> 100|>, {U[x, y], W[x, y]}]", "1/100"),
    ("FluidViscosity[" + FL2 + ", <|\"DynamicViscosity\" -> mu, \"FluidDynamicsMaterialModel\" -> \"PowerLaw\"|>, "
     "{U[x, y], W[x, y]}]", Raises(NotImplementedError)),
    ("Normal[FluidViscousStress[" + FL2 + ", <|\"DynamicViscosity\" -> mu|>, {U[x, y], W[x, y]}]]",
     "{{2 mu D[U[x, y], x], mu (D[U[x, y], y] + D[W[x, y], x])}, {mu (D[U[x, y], y] + D[W[x, y], x]), "
     "2 mu D[W[x, y], y]}}"),
    ("Normal[FluidViscousStress[" + FL2 + ", <|\"ReynoldsNumber\" -> 2|>, {y, 0}]]", "{{0, 1/2}, {1/2, 0}}"),
    ("FluidViscousStress[" + FL2 + ", <|\"DynamicViscosity\" -> mu|>, {U[x, y]}]", Raises(ValueError)),
    # ── magnetic field quantities ──
    ("MagneticFieldIntensity[{Vm[x, y], {x, y}}, <||>, Vm[x, y]]", "{-D[Vm[x, y], x], -D[Vm[x, y], y]}"),
    ("MagneticFieldIntensity[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, <|\"VacuumPermeability\" -> m0, "
     "\"Magnetization\" -> {0, Mz[x, y, z], 0}|>, {Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}]",
     "Curl[{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}]/m0 - {0, Mz[x, y, z], 0}"),
    ("MagneticFieldIntensity[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, <||>, {-y, x, 0}]",
     "{0, 0, 5000000/Pi}"),
    ("MagneticFieldIntensity[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, <|\"RelativePermeability\" -> 2|>, "
     "{-y, x, 0}]", Raises(NotImplementedError)),
    ("MagneticFieldIntensity[{Vm[x, y], {x, y}}, <||>, {1, 2}]", Raises(ValueError)),
    ("MagneticFluxDensity[{Vm[x, y], {x, y}}, <|\"VacuumPermeability\" -> m0, \"Magnetization\" -> {Mx[x, y], 0}|>, "
     "Vm[x, y]]", "{m0 (Mx[x, y] - D[Vm[x, y], x]), -m0 D[Vm[x, y], y]}"),
    ("MagneticFluxDensity[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, <||>, {-y, x, 0}]", "{0, 0, 2}"),
    ("MagneticFluxDensity[{Vm[x], {x}}, <|\"RelativePermeability\" -> 3|>, -x]", "{3 Pi/2500000}"),
    # remanent flux density model: "RelativePermeability" is the recoil permeability (Wolfram documentation)
    ("Activate[MagnetostaticPDEComponent[{Vm[x], {x}}, <|\"RelativePermeability\" -> mr, \"VacuumPermeability\" -> m0, "
     "\"RemanentMagneticFluxDensity\" -> {Br[x]}|>]]", "-m0 mr Vm''[x] + Br'[x]"),
    ("MagneticFluxDensity[{{Ax[x, y, z], Ay[x, y, z], Az[x, y, z]}, {x, y, z}}, <||>, {-y, x}]", Raises(ValueError)),
]

# ── reference values computed independently of MikoshiLang ──
_k1 = 0.8603335890193798   # first root of k tan k = 1 (mpmath.findroot), Robin spectrum k^2
_k2 = 3.425618459481728    # second root
_odd = np.arange(1, 4002, 2.0)
_sgn = np.where(((_odd - 1) / 2) % 2 == 0, 1.0, -1.0)
W26_TORSION = float(16 / np.pi ** 4 * np.sum(np.outer(_sgn, _sgn) / (np.outer(_odd, _odd) *
                                                                    (_odd[:, None] ** 2 + _odd[None, :] ** 2))))
_t1 = (2 * 5 / 9.81) ** 0.5
_v1 = 0.95 * 9.81 * _t1
W26_BOUNCE = _v1 * (1.5 - _t1) - 9.81 / 2 * (1.5 - _t1) ** 2
W26_BLOCH = sorted((n + 0.5 / (2 * np.pi)) ** 2 for n in range(-3, 4))[:5]

EIG = "NDEigenvalues"
CASES += [
    # ── NDEigenvalues / NDEigensystem (Wolfram documentation examples and Sturm-Liouville closed forms) ──
    ("NDEigenvalues[-Laplacian[u[x], {x}], u[x], {x, 0, Pi}, 4]", NearList([0, 1, 4, 9], 1e-8)),
    ("NDEigenvalues[{-Laplacian[u[x], {x}], DirichletCondition[u[x] == 0, True]}, u[x], {x, 0, Pi}, 4]",
     NearList([1, 4, 9, 16], 1e-8)),
    ("NDEigenvalues[{-u''[x], DirichletCondition[u[x] == 0, x == 0 || x == Pi]}, u, {x, 0, Pi}, 3]",
     NearList([1, 4, 9], 1e-8)),
    ("NDEigenvalues[-Laplacian[u[x], {x}] + NeumannValue[u[x], x == 1], u[x], {x, 0, 1}, 2]",
     NearList([_k1 ** 2, _k2 ** 2], 1e-8)),
    ("NDEigenvalues[{-u''[x], PeriodicBoundaryCondition[u[x], x == 2 Pi, Function[x, x - 2 Pi]]}, u[x], "
     "{x, 0, 2 Pi}, 5]", NearList([0, 1, 1, 4, 4], 1e-8)),
    ("NDEigenvalues[{-u''[x], PeriodicBoundaryCondition[u[x], x == 2 Pi, Function[x, x - 2 Pi]], "
     "PeriodicBoundaryCondition[Exp[I/2] u[x], x == 0, Function[x, x + 2 Pi]]}, u[x], {x, 0, 2 Pi}, 5]",
     Raises(ValueError)),
    # Bloch condition u(2 Pi) = e^(I/2) u(0): spectrum (n + 1/(4 Pi))^2
    ("NDEigenvalues[{-u''[x], PeriodicBoundaryCondition[Exp[I/2] u[x], x == 2 Pi, Function[x, x - 2 Pi]]}, u[x], "
     "{x, 0, 2 Pi}, 3]", NearList(W26_BLOCH[:3], 1e-8)),
    ("NDEigenvalues[{-u''[x], u[0] == u[2 Pi], u'[0] == u'[2 Pi]}, u, {x, 0, 2 Pi}, 5]",
     NearList([0, 1, 1, 4, 4], 1e-8)),
    # Wolfram documentation: a lone u[0] == u[2 Pi] is a periodic condition
    ("NDEigenvalues[{-u''[x], u[0] == u[2 Pi]}, u[x], {x, 0, 2 Pi}, 5]", NearList([0, 1, 1, 4, 4], 1e-8)),
    ("NDEigenvalues[{D[u[t, x], t] == -Laplacian[u[t, x], {x}], DirichletCondition[u[t, x] == 0, True]}, u[t, x], "
     "t, {x, 0, Pi}, 4]", NearList([1, 4, 9, 16], 1e-8)),
    ("NDEigenvalues[-1/4 u''[x] + x^2 u[x], u, {x, -4, 4}, 3]", NearList([0.5, 1.5, 2.5], 1e-5)),
    ("NDEigenvalues[{D[u[t, x], t] - D[u[t, x], {x, 2}], DirichletCondition[u[t, x] == 0, True]}, u, t, "
     "{x, 0, Pi}, 3]", NearList([-1, -4, -9], 1e-8)),
    ("NDEigenvalues[{D[u[t, x], {t, 2}] - D[u[t, x], {x, 2}], DirichletCondition[u[t, x] == 0, True]}, u, t, "
     "{x, 0, Pi}, 4]", NearList([-1j, 1j, -2j, 2j], 1e-8)),
    ("NDEigenvalues[{-Laplacian[u[x, y], {x, y}], DirichletCondition[u[x, y] == 0, True]}, u[x, y], "
     "{x, 0, Pi}, {y, 0, Pi}, 4]", NearList([2, 5, 5, 8], 1e-8)),
    ("NDEigenvalues[-Laplacian[u[x, y], {x, y}], u[x, y], Element[{x, y}, Rectangle[{0, 0}, {Pi, Pi}]], 4]",
     NearList([0, 1, 1, 2], 1e-8)),
    ("NDEigenvalues[-u''[x], u, {x, 0, Pi}, w26n]", "NDEigenvalues[-u''[x], u, {x, 0, Pi}, w26n]"),
    ("NDEigenvalues[-c u''[x], u, {x, 0, Pi}, 2]", Raises(ValueError)),
    ("NDEigenvalues[{-u''[x], DirichletCondition[u[x] == 1, True]}, u, {x, 0, Pi}, 2]", Raises(ValueError)),
    ("NDEigenvalues[-Laplacian[u[x, y], {x, y}], u[x, y], Element[{x, y}, Disk[]], 2]", Raises(NotImplementedError)),
    ("NDEigenvalues[-u''[x], u, {x, 0, Pi}, -1]", Raises(ValueError)),
    ("NDEigenvalues[-u''[x], u, {x, 0, Pi}, \"x\"]", Raises(ValueError)),
    ("First[NDEigensystem[{-u''[x], DirichletCondition[u[x] == 0, True]}, u, {x, 0, Pi}, 3]]",
     NearList([1, 4, 9], 1e-8)),
    # unit L2-normalised eigenfunctions: sqrt(2/Pi) sin(n x)
    ("NDEigensystem[{-u''[x], DirichletCondition[u[x] == 0, True]}, u, {x, 0, Pi}, 1][[2, 1]][Pi/2]",
     Near((2 / np.pi) ** 0.5, 1e-7)),
    ("NDEigensystem[{-u''[x], DirichletCondition[u[x] == 0, True]}, u[x], {x, 0, Pi}, 2][[2, 2]] /. x -> Pi/4",
     Near((2 / np.pi) ** 0.5, 1e-7)),
    # residual of the eigen-equation at an interior point: -u'' - lambda u = 0
    ("Module[{w26es = NDEigensystem[-u''[x] + NeumannValue[u[x], x == 1], u, {x, 0, 1}, 1]}, "
     "-w26es[[2, 1]]''[0.3] - w26es[[1, 1]] w26es[[2, 1]][0.3]]", Near(0, 1e-6)),
    ("NDEigensystem[-u''[x], u, {x, 0, Pi}, w26n]", "NDEigensystem[-u''[x], u, {x, 0, Pi}, w26n]"),
    ("NDEigensystem[{-u''[x], DirichletCondition[u[x] == x, True]}, u, {x, 0, Pi}, 1]", Raises(ValueError)),
    ("NDEigensystem[-Laplacian[u[x, y], {x, y}], u[x, y], Element[{x, y}, Disk[]], 1]", Raises(NotImplementedError)),
    # ── NDSolve / NDSolveValue boundary-value problems (spectral collocation) ──
    ("NDSolveValue[{DiffusionPDETerm[{u[x], {x}}, 1] == 1, DirichletCondition[u[x] == 0, True]}, u, {x, 0, 1}][0.3]",
     Near(0.3 * 0.7 / 2, 1e-10)),
    ("NDSolveValue[{-u''[x] == 1, u[0] == 0, u[1] == 0}, u, {x, 0, 1}][1/2]", Near(0.125, 1e-10)),
    ("NDSolveValue[{u''[x] + u[x] == 0, u[0] == 0, u[Pi/2] == 2}, u, {x, 0, Pi/2}][Pi/6]", Near(1, 1e-10)),
    ("NDSolveValue[{-u''[x] == NeumannValue[1, x == 1], DirichletCondition[u[x] == 0, x == 0]}, u, {x, 0, 1}][1]",
     Near(1, 1e-10)),
    ("NDSolveValue[{-u''[x] + u[x] == Cos[x], u[0] == u[2 Pi], u'[0] == u'[2 Pi]}, u, {x, 0, 2 Pi}][Pi]",
     Near(-0.5, 1e-10)),
    ("NDSolveValue[{-u''[x] + u[x] == Cos[x], PeriodicBoundaryCondition[u[x], x == 2 Pi, Function[x, x - 2 Pi]]}, "
     "u, {x, 0, 2 Pi}][0]", Near(0.5, 1e-10)),
    # Robin end u'(1) = -u(1): closed form u = 3 x/4 - x^2/2, u(1) = 1/4
    ("NDSolveValue[{-u''[x] == 1 + NeumannValue[-u[x], x == 1], DirichletCondition[u[x] == 0, x == 0]}, u, "
     "{x, 0, 1}][1]", Near(0.25, 1e-9)),
    ("NDSolveValue[{-Laplacian[u[x, y], {x, y}] == 1, DirichletCondition[u[x, y] == 0, True]}, u, {x, 0, 1}, "
     "{y, 0, 1}][1/2, 1/2]", Near(W26_TORSION, 1e-9)),
    ("NDSolveValue[{-Laplacian[u[x, y], {x, y}] == 1, DirichletCondition[u[x, y] == 0, True]}, u, "
     "Element[{x, y}, Rectangle[{0, 0}, {1, 1}]]][1/2, 1/2]", Near(W26_TORSION, 1e-9)),
    # u'' = u, u(0) = 1, u'(0) = u'(1): u(1) = cosh 1 + sinh(1)^2/(1 - cosh 1) = -1
    ("NDSolveValue[{-u''[x] + u[x] == 0, u[0] == 1, u'[0] == u'[1]}, u, {x, 0, 1}][1]", Near(-1, 1e-9)),
    ("NDSolveValue[{-u''[x] == 1, u[0] == 0, u[1] == 2 u'[0]}, u, {x, 0, 1}]", Raises(NotImplementedError)),
    ("u[0.5] /. First[NDSolve[{-u''[x] == 1, u[0] == 0, u[1] == 0}, u, {x, 0, 1}]]", Near(0.125, 1e-10)),
    ("NDSolve[{-u''[x] == 0, u'[0] == 0, u'[1] == 0}, u, {x, 0, 1}]", Raises(ValueError)),
    ("NDSolveValue[{-u''[x] + u[x]^2 == 1, u[0] == 0, u[1] == 0}, u, {x, 0, 1}]", Raises(NotImplementedError)),
    ("NDSolveValue[{D[u[t, x], t] == D[u[t, x], {x, 2}], u[0, x] == 0, u[t, 0] == 0, u[t, 1] == 0}, u, {t, 0, 1}, "
     "{x, 0, 1}]", Raises(NotImplementedError)),
    ("NDSolveValue[{-Laplacian[u[x, y], {x, y}] == 1, DirichletCondition[u[x, y] == 0, True]}, u, "
     "Element[{x, y}, Disk[]]]", Raises(NotImplementedError)),
    # ── WhenEvent (Wolfram documentation bouncing ball; StopIntegration; side-effect actions) ──
    ("NDSolveValue[{y''[t] == -9.81, y[0] == 5, y'[0] == 0, WhenEvent[y[t] == 0, \"StopIntegration\"]}, y, "
     "{t, 0, 10}][\"Domain\"][[1, 2]]", Near(_t1, 1e-9)),
    ("NDSolveValue[{y''[t] == -9.81, y[0] == 5, y'[0] == 0, WhenEvent[y[t] == 0, y'[t] -> -0.95 y'[t]]}, y, "
     "{t, 0, 2}][1.5]", Near(W26_BOUNCE, 1e-8)),
    ("Reap[NDSolveValue[{y'[t] == 1, y[0] == 0, WhenEvent[Sin[t] == 0, Sow[t]]}, y, {t, 0, 10}]][[2, 1]]",
     NearList([np.pi, 2 * np.pi, 3 * np.pi], 1e-9)),
    ("NDSolveValue[{y'[t] == 1, y[0] == 0, WhenEvent[y[t] > 1/2, {y[t] -> 0, \"RemoveEvent\"}]}, y, {t, 0, 2}][1.5]",
     Near(1, 1e-9)),
    ("NDSolveValue[{y'[t] == 1, y[0] == 0, WhenEvent[y[t] > 1/2, y[t] -> 0]}, y, {t, 0, 2}][1.7]", Near(0.2, 1e-8)),
    ("NDSolveValue[{y'[t] == 1, y[0] == 0, WhenEvent[y[t] > 1/2, w26d[t] -> 1]}, y, {t, 0, 1}]",
     Raises(NotImplementedError)),
    ("NDSolveValue[{y'[t] == 1, y[0] == 0, WhenEvent[y[t], \"StopIntegration\"]}, y, {t, 0, 1}]",
     Raises(NotImplementedError)),
    ("NDSolveValue[{y'[t] == 1, y[0] == 0, WhenEvent[y[t] == 1/2, \"Bogus\"]}, y, {t, 0, 1}]",
     Raises(NotImplementedError)),
    ("NDSolveValue[{y'[t] == 1, y[0] == 0, WhenEvent[y[t] == w26q, \"StopIntegration\"]}, y, {t, 0, 1}]",
     Raises(ValueError)),
    ("WhenEvent[w26cond, w26act]", "WhenEvent[w26cond, w26act]"),
    # ── step-control options and monitors ──
    ("Max[Differences[Reap[NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, MaxStepSize -> 1/20, "
     "StepMonitor :> Sow[t]]][[2, 1]]]] <= 1/20 + 10^-12", "True"),
    ("Length[Reap[NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, MaxStepSize -> 1/20, "
     "StepMonitor :> Sow[t]]][[2, 1]]] >= 20", "True"),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, MaxStepSize -> 0]", Raises(ValueError)),
    ("Max[Differences[Reap[NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 10}, MaxStepFraction -> 1/40, "
     "StepMonitor :> Sow[t]]][[2, 1]]]] <= 1/4 + 10^-12", "True"),
    ("Length[Reap[NDSolveValue[{y'[t] == 1, y[0] == 0}, y, {t, 0, 10}, StepMonitor :> Sow[t]]][[2, 1]]] >= 10",
     "True"),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, MaxStepFraction -> 2]", Raises(ValueError)),
    ("Reap[NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, StartingStepSize -> 1/1000, "
     "StepMonitor :> Sow[t]]][[2, 1, 1]]", Near(0.001, 1e-12)),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, StartingStepSize -> 1/1000][1]", Near(np.exp(-1))),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, StartingStepSize -> 2]", Raises(ValueError)),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 100}, MaxSteps -> 3]", Raises(ValueError)),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, MaxSteps -> Infinity][1]", Near(np.exp(-1))),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, MaxSteps -> 1/2]", Raises(ValueError)),
    # looser goals take fewer steps; the default goals give ~1e-10 accuracy
    ("Length[Reap[NDSolveValue[{y'[t] == Cos[5 t], y[0] == 1}, y, {t, 0, 10}, MaxStepFraction -> Infinity, "
     "AccuracyGoal -> 3, PrecisionGoal -> 3, StepMonitor :> Sow[t]]][[2, 1]]] < Length[Reap[NDSolveValue[{y'[t] == "
     "Cos[5 t], y[0] == 1}, y, {t, 0, 10}, MaxStepFraction -> Infinity, StepMonitor :> Sow[t]]][[2, 1]]]", "True"),
    ("NDSolveValue[{y'[t] == Cos[5 t], y[0] == 1}, y, {t, 0, 10}, AccuracyGoal -> 12, PrecisionGoal -> 12][10]",
     Near(1 + np.sin(50) / 5, 1e-10)),
    ("NDSolveValue[{y'[t] == Cos[5 t], y[0] == 1}, y, {t, 0, 10}, AccuracyGoal -> Infinity][10]",
     Near(1 + np.sin(50) / 5, 1e-9)),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, PrecisionGoal -> -1]", Raises(ValueError)),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, AccuracyGoal -> \"x\"]", Raises(ValueError)),
    ("NIntegrate[Exp[-x^2], {x, 0, 1}, PrecisionGoal -> 6, AccuracyGoal -> 7]", Near(0.7468241328124271, 1e-8)),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, WorkingPrecision -> MachinePrecision][1]",
     Near(np.exp(-1))),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, WorkingPrecision -> 30]", Raises(NotImplementedError)),
    ("FindRoot[Cos[x] == x, {x, 1}, WorkingPrecision -> 30]",
     Pred(lambda g: abs(float(str(g.args[0].args[1])) - 0.7390851332151607) < 1e-15, "x -> 0.739085133215160641655")),
    ("Length[Reap[NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, EvaluationMonitor :> Sow[t]]][[2, 1]]] "
     "> 10", "True"),
    ("Min[Reap[NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, EvaluationMonitor :> Sow[y[t]]]][[2, 1]]] "
     ">= Exp[-1] - 10^-6", "True"),
    ("FindRoot[Cos[x] == x, {x, 1}, EvaluationMonitor :> Sow[x]]", Raises(NotImplementedError)),
    ("Reap[NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}, StepMonitor :> Sow[y[t] Exp[t]]]][[2, 1, -1]]",
     Near(1, 1e-9)),
    ("NIntegrate[x, {x, 0, 1}, StepMonitor :> Sow[x]]", Raises(NotImplementedError)),
    ("NDSolveValue[{y'[t] == y[t], y[0] == 1}, y, {t, 0, 1}, MaxIterations -> 3]", Raises(ValueError)),
    ("MandelbrotSetIterationCount[0.250001, MaxIterations -> 4000]", "3139"),
    ("MandelbrotSetMemberQ[0.2501, MaxIterations -> 250]", "True"),
    ("SingularValueList[{{1, 0}, {0, 10^-12}}, Tolerance -> 10^-6]", "{1}"),
    ("SingularValueList[{{1., 0}, {0, 10.^-12}}, Tolerance -> 10^-14]", NearList([1, 1e-12], 1e-12)),
    ("SingularValueList[{{3, 0}, {0, 2}}, Tolerance -> 1/2]", "{3, 2}"),
    ("SingularValueList[{{3, 0}, {0, 2}}, Tolerance -> 1]", "{3}"),
    ("NSum[1/n, {n, 1, Infinity}, VerifyConvergence -> True]", Raises(ValueError)),
    ("NSum[1/n^2, {n, 1, Infinity}, VerifyConvergence -> True]", Near(np.pi ** 2 / 6, 1e-10)),
    ("NSum[(-1)^n/n, {n, 1, Infinity}, VerifyConvergence -> False]", Near(-np.log(2), 1e-10)),
    # ── ParametricNDSolveValue / ParametricNDSolve / ParametricFunction ──
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 2}, {a}][1][1]", Near(np.e, 1e-9)),
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 2}, {a}][-1][2]", Near(np.exp(-2), 1e-9)),
    ("ParametricNDSolveValue[{y'[t] == a y[t] + b, y[0] == 1}, y, {t, 0, 1}, {a, b}][1, 1][1]",
     Near(2 * np.e - 1, 1e-9)),
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y[1], {t, 0, 1}, {a}][2]", Near(np.exp(2), 1e-9)),
    ("ParametricNDSolveValue[{-u''[x] == c, u[0] == 0, u[1] == 0}, u, {x, 0, 1}, {c}][8][1/2]", Near(1, 1e-10)),
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {{a, 0, 1}}][2]", Raises(ValueError)),
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {a}][1, 2]", Raises(ValueError)),
    ("Head[ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {a}][w26a]]",
     Pred(lambda g: isinstance(g, Expr) and getattr(g, "head", None) == "ParametricFunction", "ParametricFunction")),
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {a}][\"Parameters\"]", "{a}"),
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, a][1][1]", Near(np.e, 1e-9)),
    ("ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {1}]", Raises(ValueError)),
    ("(y /. ParametricNDSolve[{y'[t] == a y[t] + b, y[0] == 1}, y, {t, 0, 1}, {a, b}])[1, 1][1]",
     Near(2 * np.e - 1, 1e-9)),
    ("(y /. ParametricNDSolve[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {a}])[-1][1]", Near(np.exp(-1), 1e-9)),
    ("Head[First[ParametricNDSolve[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {a}]]]", "Rule"),
    ("ParametricNDSolve[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {{a, 0, 1}}]", Pred(lambda g: True, "accepted")),
    ("ParametricNDSolve[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {}]", Raises(ValueError)),
    ("ParametricFunction[{a}, Hold[NDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}]]][1][1]",
     Near(np.e, 1e-9)),
    ("Head[ParametricNDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}, {a}]] === ParametricFunction", "True"),
    ("ParametricFunction[{a}, Hold[NDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}]]][1, 2]",
     Raises(ValueError)),
    ("ParametricFunction[{a}, Hold[NDSolveValue[{y'[t] == a y[t], y[0] == 1}, y, {t, 0, 1}]]][\"Parameters\"]",
     "{a}"),
    # the ParametricFunction hook chains to ParametricConvexOptimization's ParametricFunction form (W08)
    ("x /. ParametricConvexOptimization[(x - a)^2, {x >= 0}, {x}, {a}][3]", Near(3, 1e-6)),
]

# SciPy-backed heads (solve_ivp, LAPACK eigen/linear solves): filtered out when SciPy is absent (minimal CI job)
NEEDS_SCIPY = ("NDEigen", "NDSolve", "ParametricND", "ParametricFunction[{a}, Hold[NDSolve")
if not HAVE_SCIPY:
    CASES = [c for c in CASES if not any(k in c[0] for k in NEEDS_SCIPY)]


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_w26(case):
    check(case)


@pytest.mark.skipif(not HAVE_SCIPY, reason="needs SciPy")
def test_w26_eigen_self_check(monkeypatch):
    """Check: the NDEigenvalues self-check raises ArithmeticError when the grid sequence cannot converge."""
    from mikoshilang.library import wl_pde
    monkeypatch.setitem(wl_pde._EIG_SEQ, 1, (12, 14))
    with pytest.raises(ArithmeticError, match="^NDEigenvalues: self-check"):
        ev("NDEigenvalues[-u''[x] + 400 x^2 u[x], u, {x, -4, 4}, 1]")


@pytest.mark.skipif(not HAVE_SCIPY, reason="needs SciPy")
def test_w26_bvp_self_check(monkeypatch):
    """Check: the boundary-value self-check raises ArithmeticError for an unresolved solution."""
    from mikoshilang.library import wl_pde
    monkeypatch.setitem(wl_pde._BVP_SEQ, 1, (6, 8))
    with pytest.raises(ArithmeticError, match="^NDSolveValue: "):
        ev("NDSolveValue[{-u''[x] == Exp[20 x], u[0] == 0, u[1] == 0}, u, {x, 0, 1}]")


@pytest.mark.skipif(not HAVE_SCIPY, reason="needs SciPy")
def test_w26_abs_kink_self_check():
    """Check: a non-smooth source does not converge spectrally -> ArithmeticError rather than a wrong answer."""
    with pytest.raises(ArithmeticError):
        ev("NDSolveValue[{-u''[x] == Abs[x - 1/3], u[0] == 0, u[1] == 0}, u, {x, 0, 1}]")


@pytest.mark.parametrize("src, name", [
    ("NDEigenvalues[-u''[x], u, {x, 0, Pi}, 2]", "NDEigenvalues"),
    ("NDEigensystem[-u''[x], u, {x, 0, Pi}, 2]", "NDEigensystem"),
    ("NDSolveValue[{y'[t] == -y[t], y[0] == 1}, y, {t, 0, 1}]", "NDSolveValue"),
])
def test_w26_requires_scipy(monkeypatch, src, name):
    """Without SciPy the spectral solvers raise a clear NotImplementedError."""
    import builtins
    real = builtins.__import__

    def fake(mod, *a, **k):
        if mod == "scipy" or mod.startswith("scipy."):
            raise ImportError("no scipy")
        return real(mod, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(NotImplementedError, match=f"^{name}: .*requires SciPy"):
        ev(src)
