"""
MikoshiLang Extended Functions - Set 25
Materials science, nanotechnology, and polymer science
"""
import sympy as sp
from sympy import symbols, sqrt, exp, log, pi, sin, cos

def get_rules():
    """Extended set 25: Materials and nanoscience (150 functions)"""
    T, E, sigma = symbols('T E sigma')
    
    return [
        # ===== CRYSTALLOGRAPHY (30 functions) =====
        ('BravaisLattice[type_]', lambda t: str(t)),
        ('CubicLatticeParameter[a_]', lambda a: a),
        ('TetragonalLatticeParameter[a_, c_]', lambda a, c: (a, c)),
        ('OrthorhombicLatticeParameter[a_, b_, c_]', lambda a, b, c: (a, b, c)),
        ('HexagonalLatticeParameter[a_, c_]', lambda a, c: (a, c)),
        ('RhombohedralLatticeParameter[a_, alpha_]', lambda a, alpha: (a, alpha)),
        ('MonoclinicLatticeParameter[a_, b_, c_, beta_]', lambda a, b, c, beta: (a, b, c, beta)),
        ('TriclinicLatticeParameter[a_, b_, c_, alpha_, beta_, gamma_]', lambda a, b, c, alpha, beta, gamma: (a, b, c, alpha, beta, gamma)),
        ('ReciprocalLatticeVector[a_, b_, c_]', lambda a, b, c: 2*pi/a),
        ('ZoneAxis[h_, k_, l_]', lambda h, k, l: (h, k, l)),
        ('XRayDiffraction[n_, lambda_, d_, theta_]', lambda n, lam, d, theta: n*lam - 2*d*sin(theta)),
        ('BraggAngle[n_, lambda_, d_]', lambda n, lam, d: sp.asin(n*lam/(2*d))),
        ('StructureFactor[f_, phase_]', lambda f, phi: f*exp(sp.I*phi)),
        ('AtomicScatteringFactor[Z_, sin_theta_lambda_]', lambda Z, stl: Z*exp(-sp.Symbol('B')*stl**2)),
        ('LaueEquation[k_in_, k_out_, G_]', lambda kin, kout, G: kout - kin - G),
        ('EwaldSphere[k_]', lambda k: 2*pi/sp.Symbol('lambda')),
        ('PowderDiffraction[d_, theta_]', lambda d, theta: 2*d*sin(theta)),
        ('TextureCoefficient[I_hkl_, I0_hkl_]', lambda I, I0: I/I0),
        ('CrystalliteSize[K_, lambda_, beta_, theta_]', lambda K, lam, beta, theta: K*lam/(beta*cos(theta))),  # Scherrer
        ('MicrostrainBroadening[epsilon_, theta_]', lambda eps, theta: 4*eps*tan(theta)),
        ('WilliamsonHallPlot[beta_, theta_]', lambda beta, theta: beta*cos(theta)),
        ('DefectDensity[broadening_]', lambda b: b**2),
        ('StickingCoefficient[adsorbed_, incident_]', lambda ads, inc: ads/inc),
        ('SurfaceEnergy[gamma_]', lambda gamma: gamma),
        ('GrainBoundaryEnergy[theta_]', lambda theta: sp.Symbol('gamma_gb')*theta*(1 - log(theta))),
        ('TwinBoundaryEnergy[gamma_]', lambda gamma: gamma/2),
        ('InterfacialEnergy[gamma1_, gamma2_, gamma12_]', lambda g1, g2, g12: g1 + g2 - g12),
        ('WettingAngle[gamma_sv_, gamma_sl_, gamma_lv_]', lambda gsv, gsl, glv: sp.acos((gsv - gsl)/glv)),
        ('YoungDupreEquation[gamma_sv_, gamma_sl_, gamma_lv_]', lambda gsv, gsl, glv: gsv - gsl - glv*cos(sp.Symbol('theta'))),
        ('ContactAngleMeasurement[theta_]', lambda theta: theta),
        
        # ===== PHASE TRANSFORMATIONS (30 functions) =====
        ('NucleationRate[n0_, delta_G_]', lambda n0, dG: n0*exp(-dG/(sp.Symbol('k_B')*sp.Symbol('T')))),
        ('CriticalNucleusRadius[sigma_, delta_G_v_]', lambda sig, dGv: -2*sig/dGv),
        ('NucleationBarrier[sigma_, delta_G_v_]', lambda sig, dGv: 16*pi*sig**3/(3*dGv**2)),
        ('HomogeneousNucleation[I0_, delta_G_star_]', lambda I0, dGstar: I0*exp(-dGstar/(sp.Symbol('k_B')*sp.Symbol('T')))),
        ('HeterogeneousNucleation[f_theta_]', lambda f: f),  # Shape factor
        ('GrowthRate[mobility_, driving_force_]', lambda mu, F: mu*F),
        ('DiffusionControlledGrowth[D_, C_, t_]', lambda D, C, t: sqrt(D*t)),
        ('InterfaceControlledGrowth[v_, delta_G_]', lambda v, dG: v*dG),
        ('JohnsonMehlAvramiEquation[t_, k_, n_]', lambda t, k, n: 1 - exp(-k*t**n)),
        ('AvramiExponent[mechanism_]', lambda m: 3 if str(m) == '3D' else 2),
        ('KolmogorovModel[V_, t_]', lambda V, t: V*(1 - exp(-sp.Symbol('k')*t**sp.Symbol('n')))),
        ('TTTCurve[T_, time_]', lambda T, t: (T, log(t))),
        ('CCTCurve[T_, cooling_rate_]', lambda T, cr: (T, cr)),
        ('MartensiticTransformation[Ms_, Mf_]', lambda Ms, Mf: (Ms, Mf)),
        ('MartensiteStartTemperature[composition_]', lambda comp: 500),  # Simplified
        ('MartensiteFinishTemperature[Ms_]', lambda Ms: Ms - 200),
        ('BainiteTTransformation[T_]', lambda T: T),
        ('PearliteFormation[T_]', lambda T: T),
        ('OstwardRipening[r_, t_]', lambda r, t: r**3 - sp.Symbol('r0')**3 == sp.Symbol('k')*t),  # LSW theory
        ('CoarseningRate[gamma_, D_]', lambda gamma, D: gamma*D),
        ('SpinoidalDecomposition[c_, c_crit_]', lambda c, cc: c != cc),
        ('ChemicalSpinodal[d2G_dc2_]', lambda d2G: d2G < 0),
        ('CoherentSpinodal[elastic_, chemical_]', lambda el, ch: el + ch < 0),
        ('CahnHilliardEquation[c_, t_]', lambda c, t: sp.diff(c, t)),
        ('UpHillDiffusion[flux_]', lambda J: J > 0),  # Against concentration gradient
        ('PhaseFieldModel[phi_, eta_]', lambda phi, eta: (phi, eta)),
        ('OrderParameter[psi_]', lambda psi: psi),
        ('GinzburgLandauEquation[psi_]', lambda psi: sp.diff(sp.Symbol('F'), psi)),
        ('EutecticSolidification[T_]', lambda T: T),
        ('PeritecticSolidification[T_]', lambda T: T),
        
        # ===== MECHANICAL PROPERTIES (30 functions) =====
        ('TruStress[F_, A_]', lambda F, A: F/A),
        ('TrueStrain[l_, l0_]', lambda l, l0: log(l/l0)),
        ('EngineeringStress[F_, A0_]', lambda F, A0: F/A0),
        ('EngineeringStrain[l_, l0_]', lambda l, l0: (l - l0)/l0),
        ('StrainHardeningExponent[n_]', lambda n: n),
        ('HollomanEquation[K_, epsilon_, n_]', lambda K, eps, n: K*eps**n),
        ('ConsidereCriterion[sigma_, epsilon_]', lambda sig, eps: sp.diff(sigma, eps) == sigma/eps),
        ('NeckingInstability[d_sigma_d_epsilon_]', lambda dsigdeps: dsigdeps == sp.Symbol('sigma')/sp.Symbol('epsilon')),
        ('PlasticDeformation[epsilon_p_]', lambda epsp: epsp),
        ('ElasticRecovery[epsilon_e_]', lambda epse: epse),
        ('WorkHardening[d_sigma_d_epsilon_]', lambda dsigdeps: dsigdeps),
        ('RecoveryProcess[rho_]', lambda rho: -sp.diff(rho, sp.Symbol('t'))),
        ('RecrystallizationKinetics[f_, t_]', lambda f, t: 1 - exp(-sp.Symbol('k')*t**sp.Symbol('n'))),
        ('GrainGrowthKinetics[D_, t_]', lambda D, t: D**2 - sp.Symbol('D0')**2 == sp.Symbol('k')*t),
        ('HallPetchSlope[k_HP_]', lambda k: k),
        ('PetchRelation[sigma_y_, sigma_0_, k_, d_]', lambda sigy, sig0, k, d: sig0 + k/sqrt(d)),
        ('DislocationVelocity[stress_, b_]', lambda sig, b: sp.Symbol('v0')*(sig/sp.Symbol('sigma0'))**sp.Symbol('m')),
        ('OrowanimechansimBypass[tau_, G_, b_, r_]', lambda tau, G, b, r: G*b/r),
        ('TaylorHardening[alpha_, G_, b_, rho_]', lambda alpha, G, b, rho: alpha*G*b*sqrt(rho)),
        ('BaileyNortonCreep[sigma_, T_, t_]', lambda sig, T, t: sp.Symbol('A')*sig**sp.Symbol('n')*t**sp.Symbol('m')*exp(-sp.Symbol('Q')/(sp.Symbol('R')*T))),
        ('PowerLawCreep[epsilon_dot_, sigma_]', lambda epsdot, sig: epsdot == sp.Symbol('A')*sig**sp.Symbol('n')*exp(-sp.Symbol('Q')/(sp.Symbol('R')*sp.Symbol('T')))),
        ('DiffusionalCreep[sigma_, d_]', lambda sig, d: sp.Symbol('A')*sig/d**2),  # Nabarro-Herring
        ('GrainBoundarySliding[epsilon_, m_]', lambda eps, m: eps**m),
        ('SuperplasticDeformation[epsilon_dot_, sigma_, d_]', lambda epsdot, sig, d: epsdot == sp.Symbol('A')*sig**2/d**2),
        ('LarsonMillerParameter[T_, t_]', lambda T, t: T*(20 + log(t))),
        ('MansonHaferdParameter[T_, t_]', lambda T, t: (T - sp.Symbol('Ta'))/(log(t) - log(sp.Symbol('ta')))),
        ('MonkmanGrantRelation[epsilon_dot_, t_r_]', lambda epsdot, tr: epsdot*tr == sp.Symbol('C')),
        ('RobinsonLifeFraction[t_, t_r_]', lambda t, tr: sum(ti/tri for ti, tri in zip(t, tr))),
        ('CreepdamageParameter[omega_]', lambda omega: omega),
        ('KachanovDamage[sigma_, omega_]', lambda sig, omega: sig/(1 - omega)),
        
        # ===== POLYMERS (30 functions) =====
        ('DegreeOfPolymerization[n_]', lambda n: n),
        ('NumberAverageMolecularWeight[Mn_]', lambda Mn: Mn),
        ('WeightAverageMolecularWeight[Mw_]', lambda Mw: Mw),
        ('PolydispersityIndex[Mw_, Mn_]', lambda Mw, Mn: Mw/Mn),
        ('EndToEndDistance[N_, l_]', lambda N, l: sqrt(N)*l),
        ('RadiusOfGyration[Rg_]', lambda Rg: Rg),
        ('FloryExponent[nu_]', lambda nu: nu),  # 0.5 ideal, 0.6 good solvent
        ('MarkHouwinkEquation[M_, K_, a_]', lambda M, K, a: K*M**a),  # Intrinsic viscosity
        ('FloryHugginsParameter[chi_]', lambda chi: chi),
        ('ThetaTemperature[T_]', lambda T: T),  # Ideal polymer solution
        ('GlassTransitionTemperature[Tg_]', lambda Tg: Tg),
        ('MeltingPointPolymer[Tm_]', lambda Tm: Tm),
        ('CrystallinityDegree[Xc_]', lambda Xc: Xc),
        ('RubberElasticity[sigma_, lambda_]', lambda sig, lam: sig == sp.Symbol('G')*(lam - 1/lam**2)),
        ('EntropyElasticity[S_, lambda_]', lambda S, lam: -sp.diff(S, lam)),
        ('FloryRehnerEquation[V2_, chi_]', lambda V2, chi: log(1 - V2) + V2 + chi*V2**2),
        ('WilliamsLandelFerryEquation[T_, Tg_]', lambda T, Tg: -sp.Symbol('C1')*(T - Tg)/(sp.Symbol('C2') + T - Tg)),
        ('TimeTemperatureSuperposition[aT_]', lambda aT: aT),
        ('MaxwellModel[G_, tau_]', lambda G, tau: G*exp(-sp.Symbol('t')/tau)),
        ('KelvinVoigtModel[E_, eta_, t_]', lambda E, eta, t: E*(1 - exp(-E*t/eta))),
        ('StandardLinearSolid[E1_, E2_, eta_]', lambda E1, E2, eta: E1 + E2*(1 - exp(-sp.Symbol('t')*E2/eta))),
        ('ViscoelasticCompliance[J_]', lambda J: J),
        ('RelaxationModulus[G_, t_]', lambda G, t: G*exp(-t/sp.Symbol('tau'))),
        ('RetardationTime[tau_]', lambda tau: tau),
        ('LossModulus[G_double_prime_]', lambda Gpp: Gpp),
        ('StorageModulus[G_prime_]', lambda Gp: Gp),
        ('LossTangent[G_double_prime_, G_prime_]', lambda Gpp, Gp: Gpp/Gp),
        ('DynamicMechanicalAnalysis[E_star_]', lambda Estar: (sp.re(Estar), sp.im(Estar))),
        ('ThixotropicBehavior[viscosity_, time_]', lambda eta, t: eta*exp(-sp.Symbol('k')*t)),
        ('RheopexyBehavior[viscosity_, time_]', lambda eta, t: eta*(1 + sp.Symbol('k')*t)),
        
        # ===== NANOMATERIALS (30 functions) =====
        ('QuantumConfinement[E_g_, d_]', lambda Eg, d: Eg + sp.Symbol('C')/d**2),
        ('QuantumDotEnergy[n_, L_]', lambda n, L: n**2*sp.Symbol('h')**2/(8*sp.Symbol('m')*L**2)),
        ('ExcitonBohrRadius[epsilon_, m_]', lambda eps, m: eps*sp.Symbol('a0')*sp.Symbol('me')/m),
        ('SurfaceToVolumeRatio[d_]', lambda d: 6/d),  # For sphere
        ('NanoparticleMeltingPoint[Tm_bulk_, d_]', lambda Tm, d: Tm*(1 - sp.Symbol('alpha')/d)),
        ('GibbsThomsonEffect[r_]', lambda r: sp.Symbol('delta_T') == 2*sp.Symbol('gamma')*sp.Symbol('Tm')/(sp.Symbol('L')*sp.Symbol('rho')*r)),
        ('CarbonNanotubeBandGap[n_, m_]', lambda n, m: 0 if (n - m) % 3 == 0 else sp.Symbol('Eg')),
        ('GrapheneDispersion[k_]', lambda k: sp.Symbol('hbar')*sp.Symbol('v_F')*k),
        ('PlasmonResonance[omega_p_]', lambda omegap: omegap),
        ('LocalizedSurfacePlasmon[epsilon_, R_]', lambda eps, R: sp.Symbol('omega_p')/sqrt(1 + 2*eps)),
        ('MieScatteringNano[r_, lambda_]', lambda r, lam: (2*pi*r/lam)**4),
        ('RayleighScatteringNano[lambda_]', lambda lam: lam**(-4)),
        ('QuantumYield[photons_emitted_, photons_absorbed_]', lambda emit, abs: emit/abs),
        ('StokesShift[lambda_abs_, lambda_em_]', lambda labdabs, lambem: lambem - labdabs),
        ('FluorescenceLifetime[tau_]', lambda tau: tau),
        ('PhotoluminescenceQuantumEfficiency[radiative_, total_]', lambda rad, tot: rad/tot),
        ('SuperparamagnetismThreshold[K_, V_, T_]', lambda K, V, T: K*V/(sp.Symbol('k_B')*T)),
        ('BlockingTemperature[K_, V_]', lambda K, V: K*V/sp.Symbol('k_B')),
        ('CoercivityNanoparticle[K_, Ms_]', lambda K, Ms: 2*K/Ms),
        ('SaturationMagnetization[M_bulk_, d_]', lambda M, d: M*(1 - sp.Symbol('delta')/d)),
        ('NanoporesConductance[d_, l_]', lambda d, l: sp.Symbol('sigma')*pi*d**2/(4*l)),
        ('CapillaryCondensation[r_, gamma_, V_m_]', lambda r, gamma, Vm: sp.Symbol('p')/sp.Symbol('p0') == exp(-2*gamma*Vm/(sp.Symbol('R')*sp.Symbol('T')*r))),
        ('BETSurfaceArea[V_m_, N_A_, a_m_]', lambda Vm, NA, am: Vm*NA*am),
        ('LangmuirIsotherm[theta_, K_, p_]', lambda theta, K, p: K*p/(1 + K*p)),
        ('FreundlichIsotherm[q_, K_, p_, n_]', lambda q, K, p, n: K*p**sp.Rational(1,n)),
        ('DLVOTheory[V_A_, V_R_]', lambda VA, VR: VA + VR),
        ('vanderWaalsAttraction[A_, d_]', lambda A, d: -A/(12*pi*d**2)),
        ('ElectrostaticRepulsion[psi_, kappa_, d_]', lambda psi, kappa, d: 64*sp.Symbol('n')*sp.Symbol('k_B')*sp.Symbol('T')*sp.tanh(sp.Symbol('e')*psi/(4*sp.Symbol('k_B')*sp.Symbol('T')))**2*exp(-kappa*d)),
        ('DebyeLength[epsilon_, T_, c_]', lambda eps, T, c: sqrt(eps*sp.Symbol('k_B')*T/(2*sp.Symbol('e')**2*sp.Symbol('N_A')*c))),
        ('ZetaPotential[mu_, epsilon_, eta_]', lambda mu, eps, eta: mu*eta/eps),
    ]

def register():
    """Register all extended25 rules"""
    return get_rules()
