import math

import pytest

from _harness import Raises, Pred, check, ids

# CODATA 2018 / SI reference values (https://physics.nist.gov/cuu/Constants/)
c, h, G, k, NA, e = 299792458, 6.62607015e-34, 6.67430e-11, 1.380649e-23, 6.02214076e23, 1.602176634e-19
me, mp_, mn = 9.1093837015e-31, 1.67262192369e-27, 1.67492749804e-27
alpha, Rinf, sigma = 7.2973525693e-3, 10973731.568160, 5.670374419e-8
eps0, mu0, u, muB, a0 = 8.8541878128e-12, 1.25663706212e-6, 1.66053906660e-27, 9.2740100783e-24, 5.29177210903e-11
hbar, R, F, b = h / (2 * math.pi), NA * k, NA * e, 2.897771955e-3
kC, g = 1 / (4 * math.pi * eps0), 9.80665
deg = math.pi / 180

CONSTANTS = [
    ("SpeedOfLight", c), ("PlanckConstant", h), ("ReducedPlanckConstant", hbar), ("GravitationalConstant", G),
    ("BoltzmannConstant", k), ("AvogadroNumber", NA), ("ElementaryCharge", e), ("ElectronMass", me),
    ("ProtonMass", mp_), ("NeutronMass", mn), ("FineStructureConstant", alpha), ("RydbergConstant", Rinf),
    ("StefanBoltzmannConstant", sigma), ("GasConstant", R), ("VacuumPermittivity", eps0), ("VacuumPermeability", mu0),
    ("CoulombConstant", kC), ("FaradayConstant", F), ("AtomicMassUnit", u), ("BohrMagneton", muB),
    ("StandardGravity", g), ("StandardAtmosphere", 101325), ("WienDisplacement", b), ("WienDisplacementConstant", b),
    ("MolarVolumeIdealGas", R * 273.15 / 100000), ("HubbleConstant", 67.4), ("SolarMass", 1.98847e30),
    ("EarthMass", 5.9722e24), ("EarthRadius", 6.371e6), ("LightYear", 9460730472580800), ("Parsec", 3.0856775814913673e16),
]

CASES = []
for name, val in CONSTANTS:
    CASES += [(f"{name}[]", val, 1e-9), (f"{name}[1]", Raises(ValueError)),
              (f"PhysicalConstant[\"{name}\"]", val, 1e-9) if name not in ("WienDisplacementConstant",) else (f"{name}[]", val, 1e-9)]
CASES += [("PhysicalConstant[\"NoSuchConstant\"]", Raises()), ("PhysicalConstant[3]", Raises()),
          ("BohrRadius[]", a0, 1e-9), ("BohrRadius[2]", 4 * a0, 1e-9), ("BohrRadius[3, 1]", "9"), ("BohrRadius[0]", Raises())]

PHYS = [
    ("Velocity[10, 2]", "5"), ("Velocity[1.5, 3]", 0.5), ("Velocity[1, 0]", Raises()),
    ("Acceleration[10, 4]", "5/2"), ("Acceleration[3., 2]", 1.5), ("Acceleration[1, 0]", Raises()),
    ("Force[2, 3]", "6"), ("Force[1/2, 4]", "2"), ("Force[2., 9.80665]", 2 * g),
    ("Momentum[2, 3]", "6"), ("Momentum[0, 5]", "0"), ("Momentum[1.5, 2]", 3.0),
    ("Work[10, 3]", "30"), ("Work[1/2, 1/2]", "1/4"), ("Work[2.5, 2]", 5.0),
    ("MechanicalPower[100, 4]", "25"), ("MechanicalPower[1, 3]", "1/3"), ("MechanicalPower[1, 0]", Raises()),
    ("Impulse[10, 3]", "30"), ("Impulse[1/2, 4]", "2"), ("Impulse[2., 0.5]", 1.0),
    ("KineticEnergy[2, 3]", "9"), ("KineticEnergy[1, 1]", "1/2"), ("KineticEnergy[1.5, 2]", 3.0),
    ("PotentialEnergy[2, 10]", 2 * g * 10), ("PotentialEnergy[2, 10, 10]", "200"), ("PotentialEnergy[1, 1, 1/2]", "1/2"),
    ("CentripetalForce[2, 3, 1]", "18"), ("CentripetalForce[1, 2, 4]", "1"), ("CentripetalForce[1, 1, 0]", Raises()),
    ("EscapeVelocity[5.972e24, 6.371e6]", math.sqrt(2 * G * 5.972e24 / 6.371e6)), ("EscapeVelocity[1, 1, 2]", "2"),
    ("EscapeVelocity[1, 0]", Raises()),
    ("OrbitalVelocity[5.972e24, 7e6]", math.sqrt(G * 5.972e24 / 7e6)), ("OrbitalVelocity[4, 1, 1]", "2"), ("OrbitalVelocity[1, 0]", Raises()),
    ("FreeFallDistance[2]", g * 2), ("FreeFallDistance[2, 10]", "20"), ("FreeFallDistance[1, 1]", "1/2"),
    ("ProjectileRange[10, 45]", 100 / g), ("ProjectileRange[10, 30, 10]", 10 * math.sin(60 * deg)), ("ProjectileRange[10, 90, 10]", "0"),
    ("ProjectileMaxHeight[10, 90]", 100 / (2 * g)), ("ProjectileMaxHeight[10, 30, 10]", "5/4"), ("ProjectileMaxHeight[0, 45]", "0"),
    ("HeatCapacity[100, 4]", "25"), ("HeatCapacity[1, 3]", "1/3"), ("HeatCapacity[1, 0]", Raises()),
    ("ThermalExpansion[1, 2, 3]", "7"), ("ThermalExpansion[10, 1/100, 5]", "21/2"), ("ThermalExpansion[2., 0, 5]", 2.0),
    ("StefanBoltzmann[300]", sigma * 300 ** 4), ("StefanBoltzmann[0]", "0"),
    ("CarnotEfficiency[400, 300]", "1/4"), ("CarnotEfficiency[100, 0]", "1"), ("CarnotEfficiency[300, 400]", Raises()), ("CarnotEfficiency[0, 0]", Raises()),
    ("EntropyChange[100, 4]", "25"), ("EntropyChange[1, 3]", "1/3"), ("EntropyChange[1, 0]", Raises()),
    ("CoulombForce[1, 1, 1]", kC), ("CoulombForce[1e-6, 2e-6, 0.1]", kC * 2e-12 / 0.01), ("CoulombForce[1, 1, 0]", Raises()),
    ("ElectricField[1, 2]", kC / 4), ("ElectricField[2, 1, 1]", "2"), ("ElectricField[1, 0]", Raises()),
    ("ElectricPotential[1, 2]", kC / 2), ("ElectricPotential[2, 1, 1]", "2"), ("ElectricPotential[1, 0]", Raises()),
    ("Capacitance[10, 5]", "2"), ("Capacitance[1, 3]", "1/3"), ("Capacitance[1, 0]", Raises()),
    ("CapacitorEnergy[2, 3]", "9"), ("CapacitorEnergy[1, 1]", "1/2"), ("CapacitorEnergy[1e-6, 10.]", 5e-5),
    ("PowerElectric[10, 2]", "20"), ("PowerElectric[1/2, 1/2]", "1/4"), ("PowerElectric[2.5, 2]", 5.0),
    ("MagneticForce[1, 2, 3]", "6"), ("MagneticForce[1/2, 2, 1]", "1"), ("MagneticForce[0, 1, 1]", "0"),
    ("MagneticFieldWire[1, 1]", mu0 / (2 * math.pi)), ("MagneticFieldWire[10, 0.5]", mu0 * 10 / (2 * math.pi * 0.5)),
    ("MagneticFieldWire[1, 0]", Raises()),
    ("FaradayLaw[2, 4]", "-1/2"), ("FaradayLaw[-3, 1]", "3"), ("FaradayLaw[1, 0]", Raises()),
    ("InductanceEnergy[2, 3]", "9"), ("InductanceEnergy[1, 1]", "1/2"), ("InductanceEnergy[0.5, 2.]", 1.0),
    ("LCFrequency[1, 1]", "1/(2 Pi)"), ("LCFrequency[1/4, 1]", "1/Pi"), ("LCFrequency[0, 1]", Raises()),
    ("Magnification[10, 5]", "-2"), ("Magnification[3, 6]", "-1/2"), ("Magnification[1, 0]", Raises()),
    ("PhotonEnergy[1]", h * c), ("PhotonEnergy[500e-9]", h * c / 500e-9), ("PhotonEnergy[0]", Raises()),
    ("DeBroglieWavelength[1]", h), ("DeBroglieWavelength[2e-24]", h / 2e-24), ("DeBroglieWavelength[0]", Raises()),
    ("WaveSpeed[10, 2]", "20"), ("WaveSpeed[1/2, 4]", "2"), ("WaveSpeed[440., 0.78]", 440 * 0.78),
    ("DopplerShift[440, 10]", 440 * 343 / 333), ("DopplerShift[440, 0, 10]", "155320/343"), ("DopplerShift[100, 0, 0, 340]", "100"),
    ("BeatFrequency[440, 442]", "2"), ("BeatFrequency[3, 1]", "2"), ("BeatFrequency[1.5, 1]", 0.5),
    ("ResonanceFrequencySpring[4, 1]", "1/Pi"), ("ResonanceFrequencySpring[1, 1]", "1/(2 Pi)"), ("ResonanceFrequencySpring[1, 0]", Raises()),
    ("ResonanceFrequencyPendulum[1]", math.sqrt(g) / (2 * math.pi)), ("ResonanceFrequencyPendulum[1, 4]", "1/Pi"),
    ("ResonanceFrequencyPendulum[0]", Raises()),
    ("ComptonWavelength[90, 1e-12]", 1e-12 + h / (me * c)), ("ComptonWavelength[0, 1e-12]", 1e-12),
    ("ComptonWavelength[180, 2e-12]", 2e-12 + 2 * h / (me * c)),
    ("RydbergFormula[1, 2]", Rinf * 0.75), ("RydbergFormula[2, 3, 36]", "5"), ("RydbergFormula[0, 2]", Raises()),
    ("TimeDilation[1, 3/5, 1]", "5/4"), ("TimeDilation[1, 1, 1]", Raises()),
    ("TimeDilation[1, 0.5 * 299792458]", 1 / math.sqrt(0.75)),
    ("LengthContraction[1, 3/5, 1]", "4/5"), ("LengthContraction[2, 0, 1]", "2"), ("LengthContraction[1, 2, 1]", Raises()),
    ("RelativisticMass[1, 3/5, 1]", "5/4"), ("RelativisticMass[1, 0]", 1.0), ("RelativisticMass[1, 1, 1]", Raises()),
    ("MassEnergyEquivalence[1]", c ** 2), ("MassEnergyEquivalence[2, 3]", "18"), ("MassEnergyEquivalence[0]", "0"),
    ("UncertaintyProduct[1e-10, 1e-24]", 1e-34 / hbar), ("UncertaintyProduct[1, 2, 4]", "1/2"), ("UncertaintyProduct[1, 1, 0]", Raises()),
    ("BraggLaw[1, 30]", "1"), ("BraggLaw[1, 30, 2]", "1/2"), ("BraggLaw[2.82e-10, 15.]", 2 * 2.82e-10 * math.sin(15 * deg)),
    ("SnellsLaw[1, 30, 3/2]", 180 / math.pi * math.asin(1 / 3)), ("SnellsLaw[1, 0, 2]", "0"), ("SnellsLaw[3/2, 60, 1]", Raises(ValueError, "total internal")),
]
PHYS += [("StefanBoltzmann[300, 1/2]", sigma * 0.5 * 300 ** 4), ("StefanBoltzmann[300, 2]", Raises()), ("StefanBoltzmann[-1]", Raises())]

GEOM = [
    ("CircleArea[2]", "4 Pi"), ("CircleArea[1/2]", "Pi/4"), ("CircleArea[-1]", Raises()),
    ("CircleCircumference[2]", "4 Pi"), ("CircleCircumference[1.]", 2 * math.pi), ("CircleCircumference[-1]", Raises()),
    ("SphereVolume[3]", "36 Pi"), ("SphereVolume[1]", "4 Pi/3"), ("SphereVolume[-2]", Raises()),
    ("SphereSurfaceArea[1]", "4 Pi"), ("SphereSurfaceArea[1/2]", "Pi"), ("SphereSurfaceArea[-1]", Raises()),
    ("CylinderVolume[1, 2]", "2 Pi"), ("CylinderVolume[2, 1/4]", "Pi"), ("CylinderVolume[-1, 1]", Raises()),
    ("CylinderSurfaceArea[1, 1]", "4 Pi"), ("CylinderSurfaceArea[2, 3]", "20 Pi"), ("CylinderSurfaceArea[1, -1]", Raises()),
    ("ConeVolume[1, 3]", "Pi"), ("ConeVolume[3, 1]", "3 Pi"), ("ConeVolume[1, -1]", Raises()),
    ("ConeSurfaceArea[3, 4]", "24 Pi"), ("ConeSurfaceArea[1, 0]", "2 Pi"), ("ConeSurfaceArea[-1, 1]", Raises()),
    ("CubeVolume[3]", "27"), ("CubeVolume[1/2]", "1/8"), ("CubeVolume[-1]", Raises()),
    ("CubeSurfaceArea[2]", "24"), ("CubeSurfaceArea[1/2]", "3/2"), ("CubeSurfaceArea[-1]", Raises()),
    ("EllipsoidVolume[1, 2, 3]", "8 Pi"), ("EllipsoidVolume[1, 1, 1]", "4 Pi/3"), ("EllipsoidVolume[1, -1, 1]", Raises()),
    ("TorusVolume[2, 1]", "4 Pi^2"), ("TorusVolume[3, 1/2]", "3 Pi^2/2"), ("TorusVolume[1, -1]", Raises()),
    ("TorusSurfaceArea[2, 1]", "8 Pi^2"), ("TorusSurfaceArea[1, 1/2]", "2 Pi^2"), ("TorusSurfaceArea[-1, 1]", Raises()),
    ("PrismVolume[3, 4]", "12"), ("PrismVolume[1/2, 2]", "1"), ("PrismVolume[-1, 1]", Raises()),
    ("PyramidVolume[3, 4]", "4"), ("PyramidVolume[1, 1]", "1/3"), ("PyramidVolume[1, -1]", Raises()),
    ("FrustumVolume[1, 2, 3]", "7 Pi"), ("FrustumVolume[1, 1, 1]", "Pi"), ("FrustumVolume[1, 2, -1]", Raises()),
    ("TetrahedronVolume[1]", "1/(6 Sqrt[2])"), ("TetrahedronVolume[2]", "8/(6 Sqrt[2])"), ("TetrahedronVolume[-1]", Raises()),
    ("Hypotenuse[3, 4]", "5"), ("Hypotenuse[1, 1]", "Sqrt[2]"), ("Hypotenuse[1., 2.]", math.sqrt(5)),
    ("TriangleArea[3, 4, 5]", "6"), ("TriangleArea[1, 1, 1]", "Sqrt[3]/4"), ("TriangleArea[1, 2, 5]", Raises()),
    ("RegularPolygonArea[4, 2]", "4"), ("RegularPolygonArea[6, 1]", "3 Sqrt[3]/2"), ("RegularPolygonArea[2, 1]", Raises()),
    ("Distance[{0, 0}, {3, 4}]", "5"), ("Distance[{1, 1}, {2, 2}]", "Sqrt[2]"), ("Distance[{1}, {1, 2}]", Raises()),
    ("Distance3D[{0, 0, 0}, {1, 2, 2}]", "3"), ("Distance3D[{1, 1, 1}, {1, 1, 1}]", "0"), ("Distance3D[{0, 0}, {1, 1}]", Raises()),
    ("Midpoint[{0, 0}, {2, 4}]", "{1, 2}"), ("Midpoint[{1, 2}, {2, 3}]", "{3/2, 5/2}"), ("Midpoint[{1}, {1, 2}]", Raises()),
    ("Midpoint3D[{0, 0, 0}, {2, 2, 2}]", "{1, 1, 1}"), ("Midpoint3D[{1, 0, 0}, {0, 0, 1}]", "{1/2, 0, 1/2}"),
    ("Midpoint3D[{0, 0}, {1, 1}]", Raises()),
    ("Slope[{0, 0}, {2, 4}]", "2"), ("Slope[{0, 1}, {3, 0}]", "-1/3"), ("Slope[{1, 0}, {1, 5}]", Raises()),
    ("PolygonArea[{{0, 0}, {1, 0}, {1, 1}, {0, 1}}]", "1"), ("PolygonArea[{{0, 0}, {4, 0}, {0, 3}}]", "6"),
    ("PolygonArea[{{0, 0}, {1, 1}}]", Raises()),
    ("PlaneEquation[{0, 0, 1}, {0, 0, 1}]", "z - 1 == 0"), ("PlaneEquation[{1, 1, 1}, {1, 1, 1}]", "x + y + z - 3 == 0"),
    ("PlaneEquation[{0, 0, 0}, {0, 0, 0}]", Raises()),
    ("PointToPlaneDistance[{0, 0, 5}, {0, 0, 1, -1}]", "4"), ("PointToPlaneDistance[{1, 1, 1}, {1, 1, 1, 0}]", "Sqrt[3]"),
    ("PointToPlaneDistance[{0, 0, 0}, {0, 0, 0, 1}]", Raises()),
    ("LineSphereIntersection[{-2, 0, 0}, {1, 0, 0}, {0, 0, 0}, 1]", "{{1, 0, 0}, {-1, 0, 0}}"),
    ("LineSphereIntersection[{0, 1, -5}, {0, 0, 1}, {0, 0, 0}, 1]", "{{0, 1, 0}}"),
    ("LineSphereIntersection[{0, 5, 0}, {1, 0, 0}, {0, 0, 0}, 1]", "{}"),
    ("LineSphereIntersection[{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, 1]", Raises()),
]

FIN = [
    ("CompoundInterest[100, 1/10, 2]", "121"), ("CompoundInterest[100., 0.05, 10]", 100 * 1.05 ** 10), ("CompoundInterest[100, 0, 5]", "100"),
    ("FutureValue[100, 1/10, 2]", "121"), ("FutureValue[50, 1, 3]", "400"), ("FutureValue[100, 0, 3]", "100"),
    ("PresentValue[121, 1/10, 2]", "100"), ("PresentValue[400, 1, 3]", "50"), ("PresentValue[100, -1, 1]", Raises()),
    ("ContinuousCompounding[100, 1/10, 10]", "100 E"), ("ContinuousCompounding[1, 0, 5]", "1"), ("ContinuousCompounding[100., 0.05, 2]", 100 * math.exp(0.1)),
    ("AnnuityPV[100, 1/10, 2]", "21000/121"), ("AnnuityPV[100, 0, 5]", "500"), ("AnnuityPV[100., 0.05, 10]", 100 * (1 - 1.05 ** -10) / 0.05),
    ("AnnuityFV[100, 1/10, 2]", "210"), ("AnnuityFV[100, 0, 5]", "500"), ("AnnuityFV[100., 0.05, 10]", 100 * (1.05 ** 10 - 1) / 0.05),
    ("LoanPayment[1000, 0, 10]", "100"), ("LoanPayment[1000, 1/10, 2]", "12100/21"),
    ("LoanPayment[200000., 0.004, 360]", 200000 * 0.004 * 1.004 ** 360 / (1.004 ** 360 - 1)),
    ("PMT[1/10, 2, 1000]", "12100/21"), ("PMT[0, 4, 100]", "25"), ("PMT[1/10, 0, 100]", Raises()),
    ("RemainingBalance[1000, 0, 10, 4]", "600"), ("RemainingBalance[1000, 1/10, 2, 2]", "0"), ("RemainingBalance[1000, 1/10, 2, 0]", "1000"),
    ("PerpetuityPV[100, 1/20]", "2000"), ("PerpetuityPV[1, 1/10]", "10"), ("PerpetuityPV[1, 0]", Raises()),
    ("GrowingPerpetuityPV[100, 1/10, 1/20]", "2000"), ("GrowingPerpetuityPV[10, 1/5, 0]", "50"), ("GrowingPerpetuityPV[1, 1/10, 1/10]", Raises()),
    ("GrowingAnnuityPV[100, 1/10, 1/20, 3]", "346750/1331"), ("GrowingAnnuityPV[100, 1/10, 1/10, 2]", "2000/11"),
    ("GrowingAnnuityPV[100, 1/10, 0, 2]", "21000/121"),
    ("EffectiveRate[1/10, 2]", "41/400"), ("EffectiveRate[1/10, 1]", "1/10"), ("EffectiveRate[0.12, 12]", 1.01 ** 12 - 1),
    ("APRtoEAR[1/10, 2]", "41/400"), ("APRtoEAR[0, 12]", "0"), ("APRtoEAR[1/10, 0]", Raises()),
    ("CAGR[100, 121, 2]", "1/10"), ("CAGR[1, 8, 3]", "1"), ("CAGR[0, 1, 1]", Raises()),
    ("DoublingTime[1]", "1"), ("DoublingTime[0.07]", math.log(2) / math.log(1.07)), ("DoublingTime[0]", Raises()),
    ("BondYieldApprox[950, 1000, 1/20, 10]", "11/195"), ("BondYieldApprox[1000, 1000, 1/20, 5]", "1/20"),
    ("BondYieldApprox[950, 1000, 1/20, 0]", Raises()),
    ("BondPrice[1000, 1/20, 1/20, 3]", "1000"), ("BondPrice[1000, 0, 1/10, 1]", "10000/11"), ("BondPrice[1000, 1/20, 1/20, 2.5]", Raises()),
    ("NPV[1/10, {-100, 110}]", "0"), ("NPV[0, {-100, 50, 60}]", "10"), ("NPV[1/10, {100}]", "100"),
    ("IRR[{-100, 110}]", 0.1), ("IRR[{-100, 60, 60}]", 0.13066238629180732), ("IRR[{100, 100}]", Raises()),
]

CASES += PHYS + GEOM + FIN


@pytest.mark.parametrize("case", CASES, ids=ids(CASES))
def test_applied(case):
    check(case)
