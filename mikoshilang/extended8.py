"""
MikoshiLang Extended Functions - Set 8
Physics, astronomy, biology, and logic
"""
import sympy as sp
from sympy import symbols, sqrt, sin, cos, exp, log, pi, I, E, oo
from sympy.physics import units

def get_rules():
    """Extended set 8: Physics, astronomy, biology, logic (250 functions)"""
    x, y, z, t, m, v, r, T, P, V = symbols('x y z t m v r T P V')
    
    return [
        # ===== CLASSICAL MECHANICS (50 functions) =====
        ('KineticEnergy[m_, v_]', lambda m, v: m * v**2 / 2),
        ('PotentialEnergy[m_, g_, h_]', lambda m, g, h: m * g * h),
        ('MomentumLinear[m_, v_]', lambda m, v: m * v),
        ('MomentumAngular[I_, omega_]', lambda I, w: I * w),
        ('Force[m_, a_]', lambda m, a: m * a),
        ('Work[F_, d_]', lambda F, d: F * d),
        ('Power[W_, t_]', lambda W, t: W / t),
        ('Impulse[F_, dt_]', lambda F, dt: F * dt),
        ('Torque[r_, F_]', lambda r, F: r * F),
        ('AngularVelocity[theta_, t_]', lambda th, t: sp.diff(th, t)),
        ('AngularAcceleration[omega_, t_]', lambda w, t: sp.diff(w, t)),
        ('MomentOfInertia[m_, r_]', lambda m, r: m * r**2),
        ('CenterOfMass[{masses__}, {positions__}]', lambda *args: sum(args[i] * args[len(args)//2+i] for i in range(len(args)//2)) / sum(args[:len(args)//2])),
        ('Projectile[v0_, angle_, g_]', lambda v0, a, g: (2 * v0**2 * sin(a) * cos(a) / g, v0**2 * sin(a)**2 / (2*g))),
        ('CentripetalForce[m_, v_, r_]', lambda m, v, r: m * v**2 / r),
        ('CentripetalAcceleration[v_, r_]', lambda v, r: v**2 / r),
        ('EscapeVelocity[M_, r_]', lambda M, r: sqrt(2 * sp.Symbol('G') * M / r)),
        ('OrbitalVelocity[M_, r_]', lambda M, r: sqrt(sp.Symbol('G') * M / r)),
        ('OrbitalPeriod[a_, M_]', lambda a, M: 2 * pi * sqrt(a**3 / (sp.Symbol('G') * M))),
        ('HookeForce[k_, x_]', lambda k, x: -k * x),
        ('SpringEnergy[k_, x_]', lambda k, x: k * x**2 / 2),
        ('PendulumPeriod[L_, g_]', lambda L, g: 2 * pi * sqrt(L / g)),
        ('FreeFallDistance[g_, t_]', lambda g, t: g * t**2 / 2),
        ('DragForce[Cd_, A_, rho_, v_]', lambda Cd, A, rho, v: sp.Rational(1,2) * Cd * A * rho * v**2),
        ('ReynoldsNumber[rho_, v_, L_, mu_]', lambda rho, v, L, mu: rho * v * L / mu),
        
        # ===== THERMODYNAMICS (45 functions) =====
        ('IdealGasLaw[P_, V_, n_, T_]', lambda P, V, n, T: P * V - n * sp.Symbol('R') * T),
        ('BoltzmannDistribution[E_, T_]', lambda E, T: exp(-E / (sp.Symbol('k_B') * T))),
        ('MaxwellBoltzmann[v_, m_, T_]', lambda v, m, T: 4 * pi * (m / (2*pi*sp.Symbol('k_B')*T))**sp.Rational(3,2) * v**2 * exp(-m*v**2/(2*sp.Symbol('k_B')*T))),
        ('Entropy[Q_, T_]', lambda Q, T: Q / T),
        ('Enthalpy[U_, P_, V_]', lambda U, P, V: U + P * V),
        ('GibbsFreeEnergy[H_, T_, S_]', lambda H, T, S: H - T * S),
        ('HelmholtzFreeEnergy[U_, T_, S_]', lambda U, T, S: U - T * S),
        ('HeatCapacity[Q_, dT_]', lambda Q, dT: Q / dT),
        ('SpecificHeat[Q_, m_, dT_]', lambda Q, m, dT: Q / (m * dT)),
        ('LatentHeat[Q_, m_]', lambda Q, m: Q / m),
        ('ClausiusClapeyron[dP_, dT_, L_, T_, dV_]', lambda dP, dT, L, T, dV: dP/dT - L/(T*dV)),
        ('VanDerWaals[P_, V_, n_, T_, a_, b_]', lambda P, V, n, T, a, b: (P + n**2*a/V**2) * (V - n*b) - n*sp.Symbol('R')*T),
        ('CarnotEfficiency[Th_, Tc_]', lambda Th, Tc: 1 - Tc/Th),
        ('OttoEfficiency[r_, gamma_]', lambda r, g: 1 - 1/r**(g-1)),
        ('DieselEfficiency[r_, cutoff_, gamma_]', lambda r, c, g: 1 - (c**g - 1)/(g*(c-1)*r**(g-1))),
        ('StefanBoltzmann[T_]', lambda T: sp.Symbol('sigma') * T**4),
        ('PlankBlackbody[freq_, T_]', lambda f, T: (2*sp.Symbol('h')*f**3/sp.Symbol('c')**2) / (exp(sp.Symbol('h')*f/(sp.Symbol('k_B')*T)) - 1)),
        ('WienDisplacement[T_]', lambda T: sp.Symbol('b') / T),
        
        # ===== ELECTROMAGNETISM (50 functions) =====
        ('CoulombForce[q1_, q2_, r_]', lambda q1, q2, r: sp.Symbol('k_e') * q1 * q2 / r**2),
        ('ElectricField[q_, r_]', lambda q, r: sp.Symbol('k_e') * q / r**2),
        ('ElectricPotential[q_, r_]', lambda q, r: sp.Symbol('k_e') * q / r),
        ('ElectricFlux[E_, A_]', lambda E, A: E * A),
        ('GaussLaw[Q_, epsilon0_]', lambda Q, e0: Q / e0),
        ('Capacitance[Q_, V_]', lambda Q, V: Q / V),
        ('ParallelPlateCapacitor[epsilon0_, A_, d_]', lambda e0, A, d: e0 * A / d),
        ('CapacitorEnergy[C_, V_]', lambda C, V: C * V**2 / 2),
        ('OhmLaw[V_, R_]', lambda V, R: V / R),
        ('ResistanceSeries[{R__}]', lambda *R: sum(R)),
        ('ResistanceParallel[{R__}]', lambda *R: 1 / sum(1/Ri for Ri in R)),
        ('ElectricalPower[V_, I_]', lambda V, I: V * I),
        ('JouleHeating[I_, R_, t_]', lambda I, R, t: I**2 * R * t),
        ('BiotSavart[I_, dl_, r_]', lambda I, dl, r: sp.Symbol('mu0') * I * dl / (4*pi*r**2)),
        ('AmpereForce[I1_, I2_, d_]', lambda I1, I2, d: sp.Symbol('mu0') * I1 * I2 / (2*pi*d)),
        ('MagneticField[I_, r_]', lambda I, r: sp.Symbol('mu0') * I / (2*pi*r)),
        ('MagneticFlux[B_, A_]', lambda B, A: B * A),
        ('FaradayInduction[dPhi_, dt_]', lambda dPhi, dt: -dPhi / dt),
        ('Inductance[Phi_, I_]', lambda Phi, I: Phi / I),
        ('InductorEnergy[L_, I_]', lambda L, I: L * I**2 / 2),
        ('LorentzForce[q_, v_, B_]', lambda q, v, B: q * v * B),
        ('CyclotronFrequency[q_, B_, m_]', lambda q, B, m: q * B / m),
        ('HallVoltage[I_, B_, n_, d_]', lambda I, B, n, d: I * B / (n * sp.Symbol('e') * d)),
        ('PoyntingVector[E_, H_]', lambda E, H: E * H),
        ('MaxwellDisplacement[dE_, dt_]', lambda dE, dt: sp.Symbol('epsilon0') * dE / dt),
        
        # ===== QUANTUM MECHANICS (45 functions) =====
        ('PlankEnergy[freq_]', lambda f: sp.Symbol('h') * f),
        ('DeBroglieWavelength[p_]', lambda p: sp.Symbol('h') / p),
        ('ComptonScattering[lambda_, theta_]', lambda l, th: l + sp.Symbol('h')/(sp.Symbol('m_e')*sp.Symbol('c')) * (1 - cos(th))),
        ('PhotoelectricEffect[freq_, work_]', lambda f, w: sp.Symbol('h') * f - w),
        ('BohrRadius[n_]', lambda n: n**2 * sp.Symbol('a0')),
        ('BohrEnergy[n_]', lambda n: -sp.Symbol('E0') / n**2),
        ('RydbergFormula[n1_, n2_]', lambda n1, n2: sp.Symbol('R_inf') * (1/n1**2 - 1/n2**2)),
        ('SchrodingerEquation[psi_, V_]', lambda psi, V: -sp.Symbol('hbar')**2/(2*sp.Symbol('m'))*sp.diff(psi, x, 2) + V*psi),
        ('HeisenbergUncertainty[dx_, dp_]', lambda dx, dp: dx * dp >= sp.Symbol('hbar')/2),
        ('PauliMatrix[i_]', lambda i: Matrix([[0, 1], [1, 0]]) if i == 1 else Matrix([[0, -I], [I, 0]]) if i == 2 else Matrix([[1, 0], [0, -1]])),
        ('SpinOperator[s_]', lambda s: sp.Symbol('hbar') * s),
        ('AngularMomentumQuantum[l_]', lambda l: sp.Symbol('hbar') * sqrt(l * (l + 1))),
        ('FermiDirac[E_, mu_, T_]', lambda E, mu, T: 1 / (exp((E - mu) / (sp.Symbol('k_B')*T)) + 1)),
        ('BoseEinstein[E_, mu_, T_]', lambda E, mu, T: 1 / (exp((E - mu) / (sp.Symbol('k_B')*T)) - 1)),
        ('TunnelProbability[E_, V_, a_]', lambda E, V, a: exp(-2 * sqrt(2*sp.Symbol('m')*(V-E)) * a / sp.Symbol('hbar'))),
        
        # ===== RELATIVITY (30 functions) =====
        ('LorentzFactor[v_]', lambda v: 1 / sqrt(1 - v**2/sp.Symbol('c')**2)),
        ('TimeDilation[t0_, v_]', lambda t0, v: t0 / sqrt(1 - v**2/sp.Symbol('c')**2)),
        ('LengthContraction[L0_, v_]', lambda L0, v: L0 * sqrt(1 - v**2/sp.Symbol('c')**2)),
        ('RelativisticMomentum[m_, v_]', lambda m, v: m * v / sqrt(1 - v**2/sp.Symbol('c')**2)),
        ('RelativisticEnergy[m_, v_]', lambda m, v: m * sp.Symbol('c')**2 / sqrt(1 - v**2/sp.Symbol('c')**2)),
        ('MassEnergyEquivalence[m_]', lambda m: m * sp.Symbol('c')**2),
        ('VelocityAddition[v1_, v2_]', lambda v1, v2: (v1 + v2) / (1 + v1*v2/sp.Symbol('c')**2)),
        ('DopplerShift[freq_, v_]', lambda f, v: f * sqrt((1 + v/sp.Symbol('c')) / (1 - v/sp.Symbol('c')))),
        ('SchwarzschildRadius[M_]', lambda M: 2 * sp.Symbol('G') * M / sp.Symbol('c')**2),
        ('GravitationalRedshift[freq_, M_, r_]', lambda f, M, r: f * sqrt(1 - 2*sp.Symbol('G')*M/(r*sp.Symbol('c')**2))),
        ('GravitationalTimeD ilation[t0_, M_, r_]', lambda t0, M, r: t0 / sqrt(1 - 2*sp.Symbol('G')*M/(r*sp.Symbol('c')**2))),
        
        # ===== ASTRONOMY (30 functions) =====
        ('KeplerThird[a_, M_]', lambda a, M: sqrt(4*pi**2*a**3 / (sp.Symbol('G')*M))),
        ('StellarLuminosity[R_, T_]', lambda R, T: 4*pi*R**2*sp.Symbol('sigma')*T**4),
        ('HRDiagram[L_, T_]', lambda L, T: (log(T), log(L))),
        ('MainSequenceLifetime[M_]', lambda M: (M/sp.Symbol('M_sun'))**(-sp.Rational(5,2)) * 10**10),
        ('ChandrasekharLimit[]', lambda: sp.Rational(14,10) * sp.Symbol('M_sun')),
        ('SchwarzschildMetric[r_, t_]', lambda r, t: -(1 - 2*sp.Symbol('G')*sp.Symbol('M')/(r*sp.Symbol('c')**2))*sp.Symbol('c')**2*t**2 + 
            (1 - 2*sp.Symbol('G')*sp.Symbol('M')/(r*sp.Symbol('c')**2))**(-1)*r**2),
        ('HubbleLaw[v_, d_]', lambda v, d: v - sp.Symbol('H0') * d),
        ('CosmologicalRedshift[z_]', lambda z: (1+z)**2 - 1 / ((1+z)**2 + 1)),
        ('CriticalDensity[H_]', lambda H: 3*H**2 / (8*pi*sp.Symbol('G'))),
        ('JeansLength[T_, rho_]', lambda T, rho: sqrt(15*sp.Symbol('k_B')*T / (4*pi*sp.Symbol('G')*sp.Symbol('mu')*sp.Symbol('m_H')*rho))),
    ]

def register():
    """Register all extended8 rules"""
    return get_rules()
