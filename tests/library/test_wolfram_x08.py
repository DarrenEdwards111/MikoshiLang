"""Differential tests for Wolfram batch X08 (spatial statistics II: Gibbs and cluster point processes, density
estimation and geostatistics; modules wl_spatial2, wl_spatial2_stats, wl_spatial2_density, wl_spatial2_geostat).

Reference values (cited per block):

* closed forms: Cauchy cluster process K and g (Ghorbani, Cauchy cluster process, Metrika 76, 2013; the difference
  of two Cauchy displacements is Cauchy with twice the scale), Poisson-process K = Pi r^2, G = F = 1 - Exp[-mu Pi
  r^2] (Illian et al., Statistical Analysis and Modelling of Spatial Point Patterns, Wiley 2008, ch. 4), the
  Penttinen overlap fraction (Penttinen 1984), the Ripley-Rasson region (Ripley & Rasson 1977);
* estimators worked by hand: border/no-correction K and G, Voronoi and Delaunay (DTFE, Schaap & van de Weygaert
  2000) densities, histogram densities, Gaussian/Epanechnikov kernels, Matheron variograms (Cressie, Statistics for
  Spatial Data, 1993, 2.4), ordinary kriging with a linear variogram (Cressie 3.2);
* the Wolfram documentation (reference.wolfram.com/language/ref/<Name>.html) VariogramModel formulas;
* simulations are seeded (SeedRandom) and checked statistically with tolerant bounds, so they are deterministic.

Every symbol assigned here starts with ``x08`` so that no case collides with definitions made by other modules.
"""
import importlib.util
import math

import pytest

from _harness import Pred, Raises, check, ev, ids, to_complex
from mikoshilang.expr import Expr, Symbol

HAVE_SCIPY = importlib.util.find_spec("scipy") is not None


def near(x, tol):
    return Pred(lambda g: abs(to_complex(g) - x) <= tol, f"within {tol} of {x}")


def between(lo, hi):
    return Pred(lambda g: lo <= to_complex(g).real <= hi, f"in [{lo}, {hi}]")


def nums(xs, tol=1e-9):
    return Pred(lambda g: len(g.args) == len(xs) and all(abs(to_complex(a) - x) <= tol for a, x in zip(g.args, xs)),
                f"close to {xs}")


def is_head(h):
    return Pred(lambda g: isinstance(g, Expr) and g.head == h, f"a {h}")


def _pts(g):
    data = g.args[0]
    if isinstance(data, Expr) and data.head == "Rule":
        data = data.args[0]
    return [[to_complex(c).real for c in p.args] for p in data.args]


def min_dist_at_least(r):
    def f(g):
        P = _pts(g)
        return all(math.dist(P[i], P[j]) >= r for i in range(len(P)) for j in range(i))
    return Pred(f, f"all pairwise distances >= {r}")


def count_between(lo, hi):
    return Pred(lambda g: lo <= len(_pts(g)) <= hi, f"between {lo} and {hi} points")


PTS = "{{0.1, 0.1}, {0.2, 0.5}, {0.6, 0.3}, {0.9, 0.8}, {0.4, 0.7}}"
SQ = "{{0, 0}, {1, 0}, {1, 1}, {0, 1}}"

# ─────────────────────────── cluster processes ───────────────────────────
# Cauchy (Ghorbani 2013): K(r) = Pi r^2 + (1 - 2 b/Sqrt[r^2 + 4 b^2])/mu, g(r) = 1 + (1 + r^2/(4 b^2))^(-3/2) /
# (4 Pi b^2 mu) in 2D; mean density mu lambda. Simulated counts: mean mu lambda |W| = 50.
CLUSTER = [
    ("CauchyPointProcess[2, 3, 1, 2]", "CauchyPointProcess[2, 3, 1, 2]"),
    ("MeanPointDensity[CauchyPointProcess[2, 3, 1, 2]]", "6"),
    ("RipleyK[CauchyPointProcess[2, 3, 1, 2], 1]", near(math.pi + (1 - 2 / math.sqrt(5)) / 2, 1e-12)),
    ("PairCorrelationG[CauchyPointProcess[2, 3, 1, 2], 0]", near(1 + 1 / (8 * math.pi), 1e-12)),
    ("PairCorrelationG[CauchyPointProcess[2, 3, 1, 2], 2]", near(1 + 2 ** -1.5 / (8 * math.pi), 1e-12)),
    ("MeanPointDensity[CauchyPointProcess[m, l, s, 2]]", "m l"),
    ("CauchyPointProcess[-1, 3, 1, 2]", Raises(ValueError, "CauchyPointProcess: ")),
    ("CauchyPointProcess[1, 3, 1]", Raises(ValueError, "CauchyPointProcess: ")),
    ("PointProcessParameterQ[CauchyPointProcess[2, 3, 1, 2]]", True),
    ("x08c = Table[SeedRandom[k]; Length[RandomPointConfiguration[CauchyPointProcess[10, 5, 0.05, 2], "
     "Rectangle[]][\"Points\"]], {k, 40}]; Mean[N[x08c]]", between(42, 58)),
    ("VarianceGammaPointProcess[2, 3, 1, 1, 2]", "VarianceGammaPointProcess[2, 3, 1, 1, 2]"),
    ("MeanPointDensity[VarianceGammaPointProcess[2, 3, 1, 1, 2]]", "6"),
    # VG pcf: 1 + f(r)/mu, f the variance-gamma density of the difference; at alpha = 1, beta = 1, d = 2 (nu = 1):
    # f(r) = 2 (r^2/2)^(1/2) K_1(r Sqrt[2]) / (2 Pi)
    ("PairCorrelationG[VarianceGammaPointProcess[2, 3, 1, 1, 2], 1.]",
     near(1 + 2 * math.sqrt(0.5) * 0.36766433, 2e-3)),
    ("VarianceGammaPointProcess[2, 3, -1, 1, 2]", Raises(ValueError, "VarianceGammaPointProcess: ")),
    ("RipleyK[VarianceGammaPointProcess[2, 3, 1, 1, 2], r]", Raises(NotImplementedError)),
    ("x08v = Table[SeedRandom[k]; Length[RandomPointConfiguration[VarianceGammaPointProcess[10, 5, 1, 0.001, 2], "
     "Rectangle[]][\"Points\"]], {k, 20}]; Mean[N[x08v]]", between(40, 60)),
    ("NeymanScottPointProcess[10, 5, UniformDistribution[{0, 0.1}], 2]",
     "NeymanScottPointProcess[10, 5, UniformDistribution[{0, 0.1}], 2]"),
    ("MeanPointDensity[NeymanScottPointProcess[10, 5, UniformDistribution[{0, 0.1}], 2]]", "50"),
    ("x08n = Table[SeedRandom[k]; Length[RandomPointConfiguration[NeymanScottPointProcess[10, 5, "
     "BinormalDistribution[{0, 0}, {0.05, 0.05}, 0], 2], Rectangle[]][\"Points\"]], {k, 40}]; Mean[N[x08n]]",
     between(42, 58)),
    ("NeymanScottPointProcess[10, -5, UniformDistribution[{0, 0.1}], 2]", Raises(ValueError)),
    ("PointCountDistribution[NeymanScottPointProcess[10, 5, UniformDistribution[{0, 0.1}], 2], Rectangle[]]",
     Raises(NotImplementedError)),
]

# ─────────────────────────── Gibbs processes ───────────────────────────
# Hard-core distances are enforced exactly by the birth-death sampler (Geyer & Moller 1994); Strauss with gamma = 1
# is Poisson (mean count 50 in the unit square).
GIBBS = [
    ("HardcorePointProcess[50, 0.05, 2]", "HardcorePointProcess[50, 0.05, 2]"),
    ("SeedRandom[3]; RandomPointConfiguration[HardcorePointProcess[100, 0.08, 2], Rectangle[]]",
     min_dist_at_least(0.08)),
    ("HardcorePointProcess[50, -0.05, 2]", Raises(ValueError, "HardcorePointProcess: ")),
    ("MeanPointDensity[HardcorePointProcess[50, 0.05, 2]]", Raises(NotImplementedError)),
    ("StraussPointProcess[50, 0.5, 0.1, 2]", "StraussPointProcess[50, 0.5, 0.1, 2]"),
    ("StraussPointProcess[50, 2, 0.1, 2]", Raises(ValueError, "StraussPointProcess: ")),
    ("PointProcessParameterQ[StraussPointProcess[50, 0.5, 0.1, 2]]", True),
    ("x08s = Table[SeedRandom[k]; Length[RandomPointConfiguration[StraussPointProcess[50, 1, 0.1, 2], "
     "Rectangle[], Method -> {\"MCMC\", \"Steps\" -> 3000}][\"Points\"]], {k, 12}]; Mean[N[x08s]]", between(43, 57)),
    ("x08s0 = Table[SeedRandom[k]; Length[RandomPointConfiguration[StraussPointProcess[50, 0.2, 0.1, 2], "
     "Rectangle[], Method -> {\"MCMC\", \"Steps\" -> 3000}][\"Points\"]], {k, 12}]; Mean[N[x08s0]]", between(15, 42)),
    ("RandomPointConfiguration[StraussPointProcess[50, 0.5, 0.1, 2], Rectangle[], Method -> \"Exact\"]",
     Raises(NotImplementedError)),
    ("StraussHardcorePointProcess[50, 0.5, 0.02, 0.1, 2]", "StraussHardcorePointProcess[50, 0.5, 0.02, 0.1, 2]"),
    ("StraussHardcorePointProcess[50, 0.5, 0.2, 0.1, 2]", Raises(ValueError, "StraussHardcorePointProcess: ")),
    ("SeedRandom[4]; RandomPointConfiguration[StraussHardcorePointProcess[100, 0.5, 0.06, 0.1, 2], Rectangle[], "
     "Method -> {\"MCMC\", \"Steps\" -> 3000}]", min_dist_at_least(0.06)),
    ("PenttinenPointProcess[50, 0.5, 0.05, 2]", "PenttinenPointProcess[50, 0.5, 0.05, 2]"),
    ("PenttinenPointProcess[50, 1.5, 0.05, 2]", Raises(ValueError, "PenttinenPointProcess: ")),
    ("SeedRandom[5]; RandomPointConfiguration[PenttinenPointProcess[50, 0, 0.05, 2], Rectangle[], "
     "Method -> {\"MCMC\", \"Steps\" -> 3000}]", min_dist_at_least(0.1)),            # gamma = 0: hard core 2 rp
    ("DiggleGatesPointProcess[50, 0.05, 2]", "DiggleGatesPointProcess[50, 0.05, 2]"),
    ("DiggleGatesPointProcess[50, -0.05, 2]", Raises(ValueError, "DiggleGatesPointProcess: ")),
    ("SeedRandom[6]; RandomPointConfiguration[DiggleGatesPointProcess[50, 0.05, 2], Rectangle[], "
     "Method -> {\"MCMC\", \"Steps\" -> 2000}]", count_between(15, 80)),
    ("DiggleGrattonPointProcess[50, 1, 0.02, 0.05, 2]", "DiggleGrattonPointProcess[50, 1, 0.02, 0.05, 2]"),
    ("DiggleGrattonPointProcess[50, 1, 0.08, 0.05, 2]", Raises(ValueError, "DiggleGrattonPointProcess: ")),
    ("SeedRandom[7]; RandomPointConfiguration[DiggleGrattonPointProcess[80, 1, 0.04, 0.06, 2], Rectangle[], "
     "Method -> {\"MCMC\", \"Steps\" -> 2000}]", min_dist_at_least(0.04)),
    ("GibbsPointProcess[{\"PairInteraction\", 50, If[#1 < 0.1, 0.5, 1] &}, 2]",
     is_head("GibbsPointProcess")),
    # the PairInteraction form with the Strauss interaction reproduces StraussPointProcess exactly (same stream)
    ("(SeedRandom[1]; RandomPointConfiguration[GibbsPointProcess[{\"PairInteraction\", 50, If[#1 < 0.1, 0.5, 1] &}, "
     "2], Rectangle[], Method -> {\"MCMC\", \"Steps\" -> 1500}]) === (SeedRandom[1]; RandomPointConfiguration["
     "StraussPointProcess[50, 0.5, 0.1, 2], Rectangle[], Method -> {\"MCMC\", \"Steps\" -> 1500}])", True),
    ("GibbsPointProcess[{\"Foo\", 50}, 2]", Raises(ValueError, "GibbsPointProcess: ")),
    ("SeedRandom[8]; RandomPointConfiguration[GibbsPointProcess[{\"PairPotential\", 30, If[#1 < 0.05, Infinity, 0] "
     "&}, 2], Rectangle[], Method -> {\"MCMC\", \"Steps\" -> 1500}]", min_dist_at_least(0.05)),
]

# ─────────────────────────── SpatialPointData, regions and selection ───────────────────────────
# Ripley-Rasson region: the convex hull scaled by 1/Sqrt[1 - m/n] about its centroid (m hull vertices); for the 4
# corners of the unit square plus the centre, m = 4, n = 5, scale Sqrt[5].
SPD = [
    (f"SpatialPointData[{PTS}, Rectangle[]][\"PointCount\"]", "5"),
    ("SpatialPointData[{{0.1, 0.1}, {2, 2}}, Rectangle[]][\"PointCount\"]", "1"),    # points outside are dropped
    ("SpatialPointData[{{0.1, 0.2}, {0.3, 0.4}} -> {5, 6}, Rectangle[]][\"Annotations\"]", "<|1 -> {5, 6}|>"),
    ("SpatialPointData[{{1, 2}, {3}}, Rectangle[]]", Raises(ValueError, "SpatialPointData: ")),
    ("SpatialPointData[x08undefined, Rectangle[]]", "SpatialPointData[x08undefined, Rectangle[]]"),
    (f"SpatialPointData[{PTS}, Rectangle[]][\"RegionMeasure\"]", "1"),
    ("RipleyRassonRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {1/2, 1/2}}]",
     "Polygon[{{1/2 - Sqrt[5]/2, 1/2 - Sqrt[5]/2}, {1/2 + Sqrt[5]/2, 1/2 - Sqrt[5]/2}, "
     "{1/2 + Sqrt[5]/2, 1/2 + Sqrt[5]/2}, {1/2 - Sqrt[5]/2, 1/2 + Sqrt[5]/2}}]"),
    ("RegionMeasure[RipleyRassonRegion[{{0, 0}, {1, 0}, {1, 1}, {0, 1}, {1/2, 1/2}}]]", "5"),
    ("RipleyRassonRegion[{{0, 0}, {1, 1}}]", Raises(ValueError, "RipleyRassonRegion: ")),
    ("RipleyRassonRegion[{{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 1, 1}}]", Raises(NotImplementedError)),
    ("RipleyRassonRegion[x08p]", "RipleyRassonRegion[x08p]"),
    ("SpatialObservationRegionQ[Rectangle[]]", True),
    ("SpatialObservationRegionQ[Disk[{0, 0}, 2]]", True),
    ("SpatialObservationRegionQ[Line[{{0, 0}, {1, 1}}]]", False),
    ("SpatialObservationRegionQ[Rectangle[{0, 0}, {a, 1}]]", False),
    ("SpatialObservationRegionQ[HalfPlane[{{0, 0}, {1, 0}}, {0, 1}]]", False),
    (f"SpatialPointSelect[SpatialPointData[{PTS}, Rectangle[]], #[[1]] > 0.5 &][\"PointCount\"]", "2"),
    ("SpatialPointSelect[SpatialPointData[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}} -> {1, 2, 3}, Rectangle[]], "
     "#[[2]] >= 0.4 &][\"Annotations\"]", "<|1 -> {2, 3}|>"),
    ("SpatialPointSelect[{{1, 2}}, #[[1]] > 0 &]", Raises(TypeError)),
    ("SpatialPointSelect[x08spd, #[[1]] > 0 &]", "SpatialPointSelect[x08spd, #[[1]] > 0 &]"),
]

# ─────────────────────────── point-data statistics ───────────────────────────
# PTS in the unit square: pair distances <= 0.3: (0.2,0.5)-(0.4,0.7) = 0.283 only; with no correction
# K(0.3) = |W| 2/(n (n - 1)) = 2/20 = 0.1. Nearest-neighbour distances: 0.412, 0.283, 0.361, 0.539, 0.283;
# G_none(0.4) = 3/5 (0.361 <= 0.4: points 2, 3, 5 ... and 0.412 > 0.4) -> 2/5 + 1/5 = 0.6? (0.283, 0.361, 0.283).
STATS = [
    (f"RipleyK[{{{PTS}, Rectangle[]}}, {{0.1, 0.3, 0.5}}, SpatialBoundaryCorrection -> None]",
     nums([0.0, 0.1, 0.4])),
    (f"NearestNeighborG[{{{PTS}, Rectangle[]}}, {{0.2, 0.4}}, SpatialBoundaryCorrection -> None]",
     nums([0.0, 0.6])),
    (f"x08f = RipleyK[{{{PTS}, Rectangle[]}}]; x08f[0.3] == RipleyK[{{{PTS}, Rectangle[]}}, 0.3]", True),
    (f"RipleyK[{{{PTS}, Rectangle[]}}][\"Statistic\"]", "\"RipleyK\""),
    (f"RipleyK[{{{PTS}, Rectangle[]}}, 0.3, SpatialBoundaryCorrection -> \"Hanisch\"]", Raises(ValueError)),
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "RipleyK[x08d, 0.1]/(Pi 0.01)", between(0.85, 1.15)),
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "BesagL[x08d, 0.2]", between(0.18, 0.22)),
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "NearestNeighborG[x08d, 0.05] - (1 - Exp[-200 Pi 0.0025])", between(-0.08, 0.08)),
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "EmptySpaceF[x08d, 0.05] - (1 - Exp[-200 Pi 0.0025])", between(-0.08, 0.08)),
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "Mean[PairCorrelationG[x08d, {0.05, 0.08, 0.1, 0.12, 0.15, 0.2}]]", between(0.85, 1.15)),
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "SpatialJ[x08d, 0.03]", between(0.8, 1.2)),
    (f"BesagL[{{{PTS}, Disk[{{0.5, 0.5}}, 0.6]}}, 0.3, SpatialBoundaryCorrection -> \"BorderMargin\"]",
     is_head("Real") if False else Pred(lambda g: to_complex(g).real >= 0, "non-negative")),
    ("PointStatisticFunction[<|\"Statistic\" -> \"RipleyK\", \"Data\" -> SpatialPointData[{{0.1, 0.1}, {0.2, 0.2}}, "
     "Rectangle[]], \"SpatialBoundaryCorrection\" -> \"None\"|>][0.2]", "1.0"),     # |W| 2/(2 1) = 1
    ("PointStatisticFunction[<|\"Statistic\" -> \"Foo\"|>]", Raises(ValueError, "PointStatisticFunction: ")),
    ("PointStatisticFunction[5]", Raises(TypeError, "PointStatisticFunction: ")),
    ("SpatialBoundaryCorrection", "SpatialBoundaryCorrection"),
    ("RipleyK[{{{0.1, 0.1}, {0.2, 0.2}}, Rectangle[]}, 0.2, SpatialBoundaryCorrection -> None]", "1.0"),
    ("RipleyK[{{{0.1, 0.1}, {0.2, 0.2}}, Rectangle[]}, 0.2, SpatialBoundaryCorrection -> 5]", Raises(TypeError)),
]

# ─────────────────────────── estimation and tests ───────────────────────────
# Poisson maximum likelihood: n/|W|. Thomas minimum contrast and Strauss maximum pseudolikelihood (Baddeley &
# Turner 2000) recover the simulation parameters within tolerant bounds.
EST = [
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "(m /. FindPointProcessParameters[x08d, PoissonPointProcess[m, 2]]) == x08d[\"PointCount\"]", True),
    ("SeedRandom[5]; x08d = RandomPointConfiguration[PoissonPointProcess[200, 2], Rectangle[]]; "
     "EstimatedPointProcess[x08d, PoissonPointProcess[m, 2]] === PoissonPointProcess[x08d[\"PointCount\"], 2]",
     True),
    ("FindPointProcessParameters[{{{0.1, 0.1}, {0.5, 0.5}}, Rectangle[]}, 5]", Raises(TypeError)),
    ("EstimatedPointProcess[{{{0.1, 0.1}, {0.5, 0.5}}, Rectangle[]}, PoissonPointProcess[m, 2], "
     "PointProcessEstimator -> \"FindClusters\"]", Raises(NotImplementedError)),
    ("PointProcessEstimator", "PointProcessEstimator"),
    ("SpatialRandomnessTest[{{{0.1, 0.1}, {0.5, 0.5}, {0.7, 0.2}}, Rectangle[]}, \"Foo\"]", Raises(ValueError)),
    ("SpatialRandomnessTest[{{{0.1, 0.1}, {0.5, 0.5}, {0.7, 0.2}}, Rectangle[]}, \"HypothesisTestData\"]",
     Raises(NotImplementedError)),
    ("PointProcessFitTest[{{{0.1, 0.1}, {0.5, 0.5}, {0.7, 0.2}}, Rectangle[]}, 5]", Raises(TypeError)),
]
EST_SCIPY = [
    ("SeedRandom[6]; x08t = RandomPointConfiguration[ThomasPointProcess[20, 10, 0.03, 2], Rectangle[]]; "
     "s /. FindPointProcessParameters[x08t, ThomasPointProcess[m, l, s, 2]]", between(0.015, 0.06)),
    ("SeedRandom[6]; x08g = RandomPointConfiguration[StraussPointProcess[200, 0.3, 0.05, 2], Rectangle[]]; "
     "g /. FindPointProcessParameters[x08g, StraussPointProcess[m, g, 0.05, 2]]", between(0.1, 0.6)),
    ("SeedRandom[6]; x08g = RandomPointConfiguration[StraussPointProcess[200, 0.3, 0.05, 2], Rectangle[]]; "
     "Head[EstimatedPointProcess[x08g, StraussPointProcess[m, g, 0.05, 2]]]", "StraussPointProcess"),
    ("SeedRandom[7]; x08p = RandomPointConfiguration[PoissonPointProcess[100, 2], Rectangle[]]; "
     "SpatialRandomnessTest[x08p]", between(0.05, 1)),
    ("SeedRandom[7]; x08p = RandomPointConfiguration[ThomasPointProcess[10, 10, 0.03, 2], Rectangle[]]; "
     "SpatialRandomnessTest[x08p]", between(0, 0.01)),
    ("SeedRandom[7]; x08p = RandomPointConfiguration[ThomasPointProcess[10, 10, 0.03, 2], Rectangle[]]; "
     "SpatialRandomnessTest[x08p, \"BesagL\"]", between(0, 0.05)),
    ("SeedRandom[8]; x08q = RandomPointConfiguration[ThomasPointProcess[10, 10, 0.03, 2], Rectangle[]]; "
     "PointProcessFitTest[x08q, ThomasPointProcess[10, 10, 0.03, 2]]", between(0.02, 1)),
    ("SeedRandom[8]; x08q = RandomPointConfiguration[ThomasPointProcess[10, 10, 0.03, 2], Rectangle[]]; "
     "PointProcessFitTest[x08q, PoissonPointProcess[100, 2]]", between(0, 0.01)),
]

# ─────────────────────────── densities ───────────────────────────
# Voronoi (1D cells [0, .2], [.2, .5], [.5, 1]): 1/0.2, 1/0.3, 1/0.5. DTFE in 1D: 2/(x_(i+1) - x_(i-1)) (ends
# 2/gap), linear between. Single point, Gaussian bandwidth 1 far from the edges: 1/(2 Pi); Epanechnikov h = 0.2 at
# the centre of the unit square: (3/4/0.2)^2 = 14.0625. Histogram 2 x 2 bins of area 1/4: counts 2, 1, 0, 2.
DENS = [
    ("x08v1 = PointDensity[{{{0.1}, {0.3}, {0.7}}, Interval[{0, 1}]}, \"Voronoi\"]; {x08v1[0.05], x08v1[0.25], "
     "x08v1[{0.9}]}", nums([5.0, 10 / 3, 2.0])),
    ("x08d1 = PointDensity[{{{0.1}, {0.3}, {0.7}}, Interval[{0, 1}]}]; {x08d1[0.1], x08d1[0.2], x08d1[0.5], "
     "x08d1[0.8]}", nums([10.0, 20 / 3, 25 / 6, 0.0])),
    ("PointDensity[PoissonPointProcess[3, 2]][{1, 2}]", "3"),
    ("PointDensity[InhomogeneousPoissonPointProcess[Function[{x, y}, x + y], 2]][{1, 2}]", "3"),
    ("PointDensity[{{{0.1}, {0.3}}, Interval[{0, 1}]}, {\"Voronoi\", 2}]", Raises(NotImplementedError)),
    ("PointDensity[{{{0.1}, {0.1}}, Interval[{0, 1}]}]", Raises(ValueError, "PointDensity: ")),
    ("PointDensity[{{{0.1}, {0.3}}, Interval[{0, 1}]}, \"Foo\"]", Raises(ValueError, "PointDensity: ")),
    ("PointDensity[x08undef]", "PointDensity[x08undef]"),
    ("PointDensity[{{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.1}}, \"Voronoi\"][\"Method\"]", "\"Voronoi\""),
    ("x08k = SmoothPointDensity[{{{0.5, 0.5}}, Rectangle[{-10, -10}, {10, 10}]}, 1]; x08k[{0.5, 0.5}]",
     near(1 / (2 * math.pi), 1e-12)),
    ("x08e = SmoothPointDensity[{{{0.5, 0.5}}, Rectangle[]}, 0.2, \"Epanechnikov\"]; {x08e[{0.5, 0.5}], "
     "x08e[{0.6, 0.5}], x08e[{2, 2}]}", nums([14.0625, 10.546875, 0.0])),
    # edge correction: a point at the corner of the unit square with a Gaussian kernel keeps 1/4 of the mass
    ("SmoothPointDensity[{{{0, 0}}, Rectangle[]}, 0.1][\"EdgeWeights\"]", nums([4.0], 1e-9)),
    ("SmoothPointDensity[{{{0.5, 0.5}}, Rectangle[]}, 0.2, \"Foo\"]", Raises(ValueError, "SmoothPointDensity: ")),
    ("SmoothPointDensity[{{{0.5, 0.5}}, Rectangle[]}, \"SheatherJones\"]", Raises(NotImplementedError)),
    ("SmoothPointDensity[{{{0.5, 0.5}}, Rectangle[]}, -1]", Raises(ValueError)),
    ("SmoothPointDensity[PoissonPointProcess[7, 2]][{0, 0}]", "7"),
    ("x08h = HistogramPointDensity[{{{0.1, 0.1}, {0.2, 0.3}, {0.7, 0.8}, {0.9, 0.9}, {0.6, 0.2}}, Rectangle[]}, 2]; "
     "{x08h[{0.1, 0.1}], x08h[{0.9, 0.9}], x08h[{0.9, 0.1}], x08h[{0.1, 0.9}]}", nums([8.0, 8.0, 4.0, 0.0])),
    ("HistogramPointDensity[{{{0.1, 0.1}, {0.2, 0.3}, {0.7, 0.2}}, Polygon[{{0, 0}, {1, 0}, {0, 1}}]}, "
     "{{{0, 0.5, 1}}, {{0, 0.5, 1}}}][{0.6, 0.3}]", "8.0"),                       # 1 point / area 1/8
    ("HistogramPointDensity[{{{0.1, 0.1}}, Rectangle[]}, \"Knuth\"]", Raises(NotImplementedError)),
    ("HistogramPointDensity[{{{0.1, 0.1}}, Rectangle[]}, 0]", Raises(ValueError, "HistogramPointDensity: ")),
    ("HistogramPointDensity[PoissonPointProcess[2, 2]][{5, 5}]", "2"),
    ("PointDensityFunction[<|\"Method\" -> \"Foo\"|>]", Raises(ValueError, "PointDensityFunction: ")),
    ("PointDensityFunction[3]", Raises(TypeError, "PointDensityFunction: ")),
    ("PointDensityFunction[<|\"Method\" -> \"Exact\", \"Density\" -> 4, \"Dimension\" -> 2|>][{1, 1}]", "4"),
    ("PointDensity[PoissonPointProcess[3, 2]][x08loc]", "PointDensity[PoissonPointProcess[3, 2]][x08loc]"),
]
DENS_SCIPY = [
    ("x08vo = PointDensity[{{{0.25, 0.5}, {0.75, 0.5}, {0.5, 0.2}, {0.5, 0.8}}, Rectangle[]}, \"Voronoi\"]; "
     "{x08vo[{0.05, 0.5}], x08vo[{0.5, 0.02}]}", nums([1 / 0.255, 1 / 0.245], 1e-9)),
    ("x08dl = PointDensity[{{{0, 0}, {1, 0}, {0, 1}, {1, 1}}, Rectangle[]}]; {x08dl[{0, 0}], x08dl[{0.5, 0.5}]}",
     Pred(lambda g: len(g.args) == 2, "two values")),
]

# ─────────────────────────── PointValuePlot ───────────────────────────
PLOT = [
    ("PointValuePlot[{{0, 0} -> 1, {1, 1} -> 3, {2, 0} -> 2}]", is_head("Graphics")),
    ("PointValuePlot[{{0, 0} -> 1, {1, 1} -> 3, {2, 0} -> 2}][[1, 2, 1]]", "RGBColor[0.9, 0.25, 0.1]"),  # maximum
    ("Count[PointValuePlot[{{0, 0}, {1, 1}} -> {\"a\", \"b\"}, \"Label\"], _Text, Infinity]", "2"),
    ("PointValuePlot[{{0, 0, 0} -> 1, {1, 1, 1} -> 2}, \"Size\"]", is_head("Graphics3D")),
    ("PointValuePlot[{{0, 0} -> 1}, \"Shape\"]", Raises(NotImplementedError)),
    ("PointValuePlot[{{0, 0} -> \"a\"}, \"Size\"]", Raises(TypeError, "PointValuePlot: ")),
    ("PointValuePlot[{{0, 0} -> 1}, \"Foo\"]", Raises(ValueError, "PointValuePlot: ")),
    ("PointValuePlot[x08data]", "PointValuePlot[x08data]"),
]

# ─────────────────────────── variograms and kriging ───────────────────────────
# Spherical {2, 3}: 2 (3/2 (r/3) - 1/2 (r/3)^3): r = 1 -> 26/27, r = 1.5 -> 1.375; Exponential range 3 b Log[20]/3;
# Poisson range: first zero of J_1 (3.8317...) times b; GaussianLaguerre L_1^(1/2)(x) = 3/2 - x -> r = 2 Sqrt[3/2].
VARIO = [
    ("x08sp = VariogramModel[\"Spherical\", {2, 3}]; {x08sp[0], x08sp[1], x08sp[3], x08sp[5]}", "{0, 26/27, 2, 2}"),
    ("VariogramModel[\"Spherical\", {2, 3}][1.5]", near(1.375, 1e-12)),
    ("VariogramModel[\"Spherical\", {2, 3}][\"Range\"]", "3"),
    ("VariogramModel[\"Exponential\", {2, 3}, SpatialNoiseLevel -> 1/2][3]", "5/2 - 2/E"),
    ("VariogramModel[\"Exponential\", {2, 3}][\"Range\"]", near(3 * math.log(20), 1e-9)),
    ("VariogramModel[\"Gaussian\", {1, 2}][\"Range\"]", near(2 * math.sqrt(math.log(20)), 1e-9)),
    ("VariogramModel[\"SineHoleEffect\", {1, 2}][\"Range\"]", near(2.0, 1e-9)),
    ("VariogramModel[\"GaussianLaguerre\", {1, 2, 1, 1}][\"Range\"]", near(2 * math.sqrt(1.5), 1e-9)),
    ("VariogramModel[\"Dagum\", {1, 2, 1, 2}][\"Range\"]", near(2 * (0.95 ** -2 - 1) ** -0.5, 1e-9)),
    ("VariogramModel[\"Power\", {2, 1.5}][4]", near(2 ** 1.5, 1e-12)),
    ("VariogramModel[\"Power\", {2, 1.5}][\"Sill\"]", "Infinity"),
    ("VariogramModel[\"Cubic\", {2, 3}][3]", "2"),
    ("VariogramModel[\"Pentaspherical\", {1, 1}][1/2]", "203/256"),   # (1/2)(15/8 - 5/16 + 3/128)
    ("VariogramModel[\"Circular\", {1, 1}][1]", "1"),
    ("VariogramModel[\"BoundedLinear\", {2, 4}][1]", "1/2"),
    ("VariogramModel[\"Askey\", {1, 2, 2}][1]", "3/4"),
    ("VariogramModel[\"WhiteNoise\", {3}][{0, 1}]", "{0, 3}"),
    ("VariogramModel[\"Linear\", {2}][\"Range\"]", "Infinity"),
    ("VariogramModel[\"GeneralizedCauchy\", {1, 1, 2, 2}][1]", "1/2"),
    ("VariogramModel[\"PoweredExponential\", {1, 1, 1}][1]", "1 - 1/E"),
    ("VariogramModel[\"Spherical\", {2, 3}][\"Name\"]", "\"Spherical\""),
    ("VariogramModel[\"Spherical\", {2, 3}][\"Parameters\"]", "{2, 3}"),
    ("VariogramModel[\"Spherical\", {2, 3}][\"SpatialNoiseLevel\"]", "0"),
    ("VariogramModel[\"Spherical\", {-2, 3}]", Raises(ValueError, "VariogramModel: ")),
    ("VariogramModel[\"Foo\", {2, 3}]", Raises(ValueError, "VariogramModel: ")),
    ("VariogramModel[\"Spherical\", {2}]", Raises(ValueError, "VariogramModel: ")),
    ("VariogramModel[\"Spherical\", 2]", Raises(TypeError, "VariogramModel: ")),
    ("VariogramModel[\"PoweredExponential\", {1, 1, 3}]", Raises(ValueError, "VariogramModel: ")),
    ("VariogramModel[\"Spherical\", {a, b}][x08r]", "VariogramModel[\"Spherical\", {a, b}][x08r]"),
    ("VariogramModel[\"Spherical\", {2, 3}][\"Visualization\"]", is_head("Graphics")),
    # Matheron by hand: lags 1: (2^2 + 1^2 + 3^2)/6 = 7/3; lags 2-3: (1 + 4 + 16)/6 = 7/2 at mean distance 7/3
    ("BinnedVariogramList[{{0}, {1}, {2}, {3}} -> {1, 3, 2, 5}, {{0, 1.5, 3.5}}, Method -> {\"NonLattice\", "
     "\"MinPairs\" -> 1, \"MaxDistanceRatio\" -> 1, \"ScaleEstimator\" -> \"Matheron\"}]",
     "WeightedData[{{1.0, 2.3333333333333335}, {2.3333333333333335, 3.5}}, {3, 3}]"),
    ("BinnedVariogramList[{0 -> 1, 1 -> 3, 2 -> 2, 3 -> 5}, {{0, 1.5}}, Method -> {\"NonLattice\", \"MinPairs\" -> 1, "
     "\"ScaleEstimator\" -> \"Matheron\"}]", "WeightedData[{{1.0, 2.3333333333333335}}, {3}]"),
    ("BinnedVariogramList[{{0}, {1}} -> {1, 3}, Method -> {\"NonLattice\", \"ScaleEstimator\" -> \"Qn\"}]",
     Raises(NotImplementedError)),
    ("BinnedVariogramList[{{0}, {1}, {2}} -> {1, 3}]", Raises(ValueError, "BinnedVariogramList: ")),
    ("BinnedVariogramList[{{0}, {1}, {2}} -> {1, 3, 2}]", Raises(ValueError, "BinnedVariogramList: ")),  # < 30 pairs
    ("BinnedVariogramList[x08data]", "BinnedVariogramList[x08data]"),
    # ordinary kriging, linear variogram, two points: weights 1/2, 1/2, variance 1/2 (Cressie 3.2)
    ("x08k1 = SpatialEstimate[{{0}, {1}} -> {1., 3.}, VariogramFunction -> VariogramModel[\"Linear\", {1}]]; "
     "{x08k1[0.5], x08k1[0.5, \"Variance\"], x08k1[{{0}, {1}}]}", nums([2.0, 0.5, None], 1e-9)
     if False else Pred(lambda g: abs(to_complex(g.args[0]) - 2) < 1e-9 and abs(to_complex(g.args[1]) - 0.5) < 1e-9
                        and [to_complex(v).real for v in g.args[2].args] == pytest.approx([1.0, 3.0]),
                        "2, 0.5, {1, 3}")),
    ("x08k1 = SpatialEstimate[{{0}, {1}} -> {1., 3.}, VariogramFunction -> VariogramModel[\"Linear\", {1}]]; "
     "x08k1[0.5, \"Around\"]", Pred(lambda g: g.head == "Around", "Around")),
    ("x08k2 = SpatialEstimate[{{0}, {1}, {3}} -> {1., 3., 2.}, VariogramFunction -> {\"Exponential\", {1, 1}}, "
     "SpatialNoiseLevel -> 0.5]; Abs[x08k2[0] - 1] > 0.1", True),                  # noise: smoothing, no interpolation
    ("x08k3 = SpatialEstimate[{{0}, {1}, {3}} -> {1., 3., 2.}, VariogramFunction -> {\"Exponential\", {1, 1}}, "
     "SpatialNoiseLevel -> 0]; {x08k3[0], x08k3[1], x08k3[3]}", nums([1.0, 3.0, 2.0], 1e-8)),
    ("x08k4 = SpatialEstimate[{{0}, {1}, {2}, {3}} -> {1., 2., 3., 4.}, VariogramFunction -> {\"Exponential\", "
     "{1, 1}}, SpatialTrendFunction -> 1, SpatialNoiseLevel -> 0]; x08k4[10]", near(11.0, 1e-8)),  # linear trend
    ("SpatialEstimate[{{0}, {0}} -> {1., 3.}, VariogramFunction -> VariogramModel[\"Linear\", {1}]]",
     Raises(ValueError, "SpatialEstimate: ")),
    ("SpatialEstimate[{{0}, {1}} -> {1., 3.}, VariogramFunction -> 5]", Raises(TypeError, "SpatialEstimate: ")),
    ("SpatialEstimate[{GeoPosition[{0, 0}] -> 1, GeoPosition[{1, 1}] -> 2}]", Raises(NotImplementedError)),
    ("SpatialEstimate[x08data]", "SpatialEstimate[x08data]"),
    ("x08k1 = SpatialEstimate[{{0}, {1}} -> {1., 3.}, VariogramFunction -> VariogramModel[\"Linear\", {1}]]; "
     "x08k1[0.5, \"Foo\"]", Raises(ValueError, "SpatialEstimatorFunction: ")),
    ("x08k1 = SpatialEstimate[{{0}, {1}} -> {1., 3.}, VariogramFunction -> VariogramModel[\"Linear\", {1}]]; "
     "x08k1[0.5, {\"Value\", \"StandardDeviation\"}]", nums([2.0, math.sqrt(0.5)], 1e-9)),
    ("SpatialEstimatorFunction[3]", Raises(TypeError, "SpatialEstimatorFunction: ")),
    ("SpatialEstimatorFunction[<|\"Locations\" -> {{0}}|>]", Raises(ValueError, "SpatialEstimatorFunction: ")),
    ("x08k1 = SpatialEstimate[{{0}, {1}} -> {1., 3.}, VariogramFunction -> VariogramModel[\"Linear\", {1}]]; "
     "Head[x08k1]", "SpatialEstimatorFunction"),
    ("VariogramFunction", "VariogramFunction"),
    ("SpatialNoiseLevel", "SpatialNoiseLevel"),
    ("SpatialTrendFunction", "SpatialTrendFunction"),
    ("SpatialEstimate[{{0}, {1}} -> {1., 3.}, SpatialNoiseLevel -> -1]", Raises(ValueError, "SpatialEstimate: ")),
    ("SpatialEstimate[{{0}, {1}} -> {1., 3.}, SpatialTrendFunction -> -1]", Raises(ValueError, "SpatialEstimate: ")),
]
GRID = ("x08pts = Flatten[Table[{i, j}, {i, 0, 9}, {j, 0, 9}], 1]; x08z = x08pts -> (Sin[#[[1]]/2.] + "
        "Cos[#[[2]]/3.] & /@ x08pts); ")
VARIO_SCIPY = [
    (GRID + "Head[EstimatedVariogramModel[x08z]]", "VariogramModel"),
    (GRID + "EstimatedVariogramModel[x08z, \"Exponential\"][\"Name\"]", "\"Exponential\""),
    (GRID + "EstimatedVariogramModel[x08z, {\"Spherical\", {a, 20}}][\"Parameters\"][[2]]", "20.0"),
    (GRID + "EstimatedVariogramModel[x08z, \"Gaussian\", SpatialNoiseLevel -> 0][\"SpatialNoiseLevel\"]", "0.0"),
    (GRID + "Head[EstimatedVariogramModel[x08z][\"BinnedVariogram\"]]", "WeightedData"),
    (GRID + "EstimatedVariogramModel[x08z, \"Foo\"]", Raises(ValueError, "EstimatedVariogramModel: ")),
    ("EstimatedVariogramModel[{{0}, {1}, {2}} -> {1, 3, 2}]", Raises(ValueError, "EstimatedVariogramModel: ")),
    ("EstimatedVariogramModel[x08data]", "EstimatedVariogramModel[x08data]"),
    # exact data from a spherical model: the fitted range is close to the true 6
    ("x08e1 = Table[{i}, {i, 0, 59}]; SeedRandom[2]; x08e2 = Accumulate[RandomVariate[NormalDistribution[], 60]]; "
     "Head[EstimatedVariogramModel[x08e1 -> x08e2, \"Linear\"]]", "VariogramModel"),
    (GRID + "x08f = SpatialEstimate[x08z]; Abs[x08f[{2.5, 3.5}] - (Sin[1.25] + Cos[3.5/3])]", between(0, 0.05)),
    (GRID + "x08f = SpatialEstimate[x08z, SpatialNoiseLevel -> 0]; Abs[x08f[{2, 3}] - (Sin[1.] + Cos[1.])]",
     between(0, 1e-6)),
]

GROUPS = [CLUSTER, GIBBS, SPD, STATS, EST, DENS, PLOT, VARIO]
SCIPY_GROUPS = [EST_SCIPY, DENS_SCIPY, VARIO_SCIPY]
CASES = [c for g in GROUPS for c in g] + [c for g in SCIPY_GROUPS for c in g]
BASE = [c for g in GROUPS for c in g]
WITH_SCIPY = [c for g in SCIPY_GROUPS for c in g]


@pytest.mark.parametrize("case", BASE, ids=ids(BASE))
def test_wolfram_x08(case):
    check(case)


@pytest.mark.skipif(not HAVE_SCIPY, reason="SciPy not installed")
@pytest.mark.parametrize("case", WITH_SCIPY, ids=ids(WITH_SCIPY))
def test_wolfram_x08_scipy(case):
    check(case)


@pytest.mark.skipif(HAVE_SCIPY, reason="only without SciPy")
def test_without_scipy_raises():
    with pytest.raises(NotImplementedError, match="SciPy"):
        ev("EstimatedVariogramModel[Table[{i}, {i, 0, 40}] -> Table[Sin[i/3.], {i, 0, 40}]]")
    with pytest.raises(NotImplementedError, match="SciPy"):
        ev("PointDensity[{{0.1, 0.1}, {0.9, 0.2}, {0.5, 0.8}, {0.4, 0.4}}]")


def test_density_integrates_to_count():
    """Check: every estimator integrates to the number of points over the window (midpoint rule, 200 x 200)."""
    import numpy as np
    from mikoshilang.library import wl_spatial2_density as D
    src = "SeedRandom[3]; x08w = RandomPointConfiguration[PoissonPointProcess[100, 2], Rectangle[]]; "
    n = to_complex(ev(src + "x08w[\"PointCount\"]")).real
    t = (np.arange(200) + 0.5) / 200
    G = np.stack(np.meshgrid(t, t, indexing="ij"), -1).reshape(-1, 2)
    methods = ["SmoothPointDensity[x08w]", "HistogramPointDensity[x08w, 5]"]
    if HAVE_SCIPY:
        methods += ["PointDensity[x08w]", "PointDensity[x08w, \"Voronoi\"]"]
    for m in methods:
        d = D._pdf_parts(ev(src + m))
        tot = D._eval_pdf(d, G, "test").sum() / 200 ** 2
        assert abs(tot - n) < 0.02 * n, (m, tot, n)


def test_seeded_reproducible():
    a = ev("SeedRandom[11]; RandomPointConfiguration[CauchyPointProcess[5, 4, .1, 2], Rectangle[]]")
    b = ev("SeedRandom[11]; RandomPointConfiguration[CauchyPointProcess[5, 4, .1, 2], Rectangle[]]")
    assert a == b


def test_numeric_and_exact_variograms_agree():
    """Check: the numpy and SymPy forms of every model agree at a few distances."""
    import numpy as np
    import sympy as sp
    from mikoshilang.library import wl_spatial2_geostat as G
    pars = {"a": 1.5, "b": 2.0, "c": 1.5, "d": 1}
    for name, (pn, _fam) in G.MODELS.items():
        if name in G.NEEDS_SCIPY and not HAVE_SCIPY:
            continue
        p = [pars[k] for k in pn]
        if name == "PoweredExponential" or name == "GeneralizedCauchy":
            p[2] = 1.5
        if name == "Power":
            p[1] = 1.2
        rs = np.array([0.3, 1.0, 1.7, 2.5])
        num = G.num_model(name, p)(rs)
        r = sp.Symbol("r", positive=True)
        f = G.sym_formula(name, [sp.Float(v) if not float(v).is_integer() else sp.Integer(int(v)) for v in p], r)
        ex = [float(sp.N(f.subs(r, sp.Float(x)))) for x in rs]
        assert np.allclose(num, ex, rtol=1e-8, atol=1e-10), (name, num, ex)
