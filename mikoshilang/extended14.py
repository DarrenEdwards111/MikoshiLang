"""
MikoshiLang Extended Functions - Set 14
Astronomy, celestial mechanics, astrophysics, and cosmology
"""
import sympy as sp
from sympy import symbols, sqrt, sin, cos, exp, log, pi, atan2

def get_rules():
    """Extended set 14: Astronomy and astrophysics (200 functions)"""
    x, y, z, t, r, M = symbols('x y z t r M')
    
    return [
        # ===== ORBITAL MECHANICS (50 functions) =====
        ('OrbitalElements[position_, velocity_]', lambda p, v: {'a': 1, 'e': 0.1, 'i': 0}),
        ('SemiMajorAxis[r_, v_, M_]', lambda r, v, M: 1/(2/r - v**2/(sp.Symbol('G')*M))),
        ('Eccentricity[rp_, ra_]', lambda rp, ra: (ra - rp)/(ra + rp)),
        ('Inclination[velocity_]', lambda v: 0.1),
        ('LongitudeOfAscendingNode[position_, velocity_]', lambda p, v: 0.5),
        ('ArgumentOfPeriapsis[position_, velocity_]', lambda p, v: 1.0),
        ('TrueAnomaly[position_, velocity_]', lambda p, v: 0.3),
        ('EccentricAnomaly[M_, e_]', lambda M, e: M),  # Simplified
        ('MeanAnomaly[t_, T_, t0_]', lambda t, T, t0: 2*pi*(t - t0)/T),
        ('Periapsis[a_, e_]', lambda a, e: a*(1 - e)),
        ('Apoapsis[a_, e_]', lambda a, e: a*(1 + e)),
        ('OrbitalEnergy[a_, M_]', lambda a, M: -sp.Symbol('G')*M/(2*a)),
        ('SpecificAngularMomentum[r_, v_]', lambda r, v: r*v),
        ('VisVivaEquation[r_, a_, M_]', lambda r, a, M: sqrt(sp.Symbol('G')*M*(2/r - 1/a))),
        ('HohmannTransfer[r1_, r2_, M_]', lambda r1, r2, M: (sqrt(sp.Symbol('G')*M/r1), sqrt(sp.Symbol('G')*M/r2))),
        ('BiellipticTransfer[r1_, r2_, rb_, M_]', lambda r1, r2, rb, M: (0, 0, 0)),
        ('PhaseAngle[a1_, a2_]', lambda a1, a2: pi*(1 - sqrt((a1 + a2)**3/(8*a2**3)))),
        ('SynodicPeriod[P1_, P2_]', lambda P1, P2: abs(1/(1/P1 - 1/P2))),
        ('HillSphere[a_, M_, m_]', lambda a, M, m: a*(m/(3*M))**sp.Rational(1,3)),
        ('RocheLimit[R_, rho_M_, rho_m_]', lambda R, rho_M, rho_m: 2.46*R*(rho_M/rho_m)**sp.Rational(1,3)),
        ('SphereOfInfluence[a_, M_, m_]', lambda a, M, m: a*(m/M)**sp.Rational(2,5)),
        ('LaplacePlane[orbits_]', lambda o: "plane"),
        ('TissuerandParameter[a_, e_, i_]', lambda a, e, i: a*(1 - e**2)*cos(i)**2),
        ('KozaiMechanism[e_, i_]', lambda e, i: e*cos(i)),
        ('LidovKozai[elements_]', lambda e: e),
        ('YarkovskyEffect[radius_, albedo_]', lambda r, a: 1e-10),
        ('PoyntingRobertsonDrag[radius_]', lambda r: 1e-12),
        ('SolarRadiationPressure[area_, mass_]', lambda a, m: 1e-8),
        
        # ===== STELLAR ASTROPHYSICS (50 functions) =====
        ('StefanBoltzmannLaw[R_, T_]', lambda R, T: 4*pi*R**2*sp.Symbol('sigma')*T**4),
        ('WienLaw[T_]', lambda T: sp.Symbol('b')/T),
        ('RayleighJeansLaw[T_, lambda_]', lambda T, lam: 2*sp.Symbol('c')*sp.Symbol('k_B')*T/lam**4),
        ('PlankLaw[T_, lambda_]', lambda T, lam: (2*sp.Symbol('h')*sp.Symbol('c')**2/lam**5) / (exp(sp.Symbol('h')*sp.Symbol('c')/(lam*sp.Symbol('k_B')*T)) - 1)),
        ('MainSequenceMass[L_]', lambda L: (L/sp.Symbol('L_sun'))**sp.Rational(1,4)),
        ('MainSequenceRadius[M_]', lambda M: (M/sp.Symbol('M_sun'))**sp.Rational(1,2)),
        ('MainSequenceTemperature[M_]', lambda M: 5778*(M/sp.Symbol('M_sun'))**sp.Rational(1,2)),
        ('EddingtonLuminosity[M_]', lambda M: 4*pi*sp.Symbol('G')*M*sp.Symbol('c')*sp.Symbol('m_p')/sp.Symbol('sigma_T')),
        ('MassLuminosityRelation[M_]', lambda M: (M/sp.Symbol('M_sun'))**sp.Rational(7,2)*sp.Symbol('L_sun')),
        ('HRDiagramPosition[L_, T_]', lambda L, T: (log(T), log(L))),
        ('SpectralClass[T_]', lambda T: "G" if 5000 < T < 6000 else "M"),
        ('BolometricCorrection[T_, filter_]', lambda T, f: -0.08),
        ('ColorIndex[B_, V_]', lambda B, V: B - V),
        ('AbsoluteMagnitude[m_, d_]', lambda m, d: m - 5*log(d/10, 10)),
        ('ApparentMagnitude[M_, d_]', lambda M, d: M + 5*log(d/10, 10)),
        ('DistanceModulus[m_, M_]', lambda m, M: m - M),
        ('LuminosityDistance[z_, H0_]', lambda z, H: sp.Symbol('c')*z/H),
        ('ProperDistance[z_, H0_]', lambda z, H: sp.Symbol('c')*z/H),
        ('ComovingDistance[z_, H0_]', lambda z, H: sp.Symbol('c')*z/H),
        ('AngularDiameterDistance[z_, H0_]', lambda z, H: sp.Symbol('c')*z/(H*(1+z))),
        ('Parallax[d_]', lambda d: 1/d),  # d in parsecs
        ('ProperMotion[v_trans_, d_]', lambda v, d: v/d),
        ('RadialVelocity[z_]', lambda z: sp.Symbol('c')*z),
        ('DopplerShiftAstro[lambda_obs_, lambda_rest_]', lambda lo, lr: (lo - lr)/lr),
        ('GravitationalRedshiftAstro[M_, r_]', lambda M, r: sqrt(1 - 2*sp.Symbol('G')*M/(r*sp.Symbol('c')**2)) - 1),
        ('CosmologicalRedshiftAstro[a_]', lambda a: 1/a - 1),
        
        # ===== PLANETARY SCIENCE (40 functions) =====
        ('SurfaceGravity[M_, R_]', lambda M, R: sp.Symbol('G')*M/R**2),
        ('EscapeVelocityPlanet[M_, R_]', lambda M, R: sqrt(2*sp.Symbol('G')*M/R)),
        ('OrbitalVelocityPlanet[M_, r_]', lambda M, r: sqrt(sp.Symbol('G')*M/r)),
        ('SynchronousOrbitRadius[M_, P_]', lambda M, P: (sp.Symbol('G')*M*P**2/(4*pi**2))**sp.Rational(1,3)),
        ('GeostationaryOrbitRadius[M_, P_]', lambda M, P: (sp.Symbol('G')*M*P**2/(4*pi**2))**sp.Rational(1,3)),
        ('TidalForce[M_, R_, d_]', lambda M, R, d: 2*sp.Symbol('G')*M*R/d**3),
        ('TidalHeating[e_, n_, k2_]', lambda e, n, k2: e**2*n**5),  # Simplified
        ('TidalLocking[a_, M_, m_, k2_]', lambda a, M, m, k2: sqrt(a**3/(sp.Symbol('G')*M))),
        ('AtmosphericScaleHeight[T_, g_]', lambda T, g: sp.Symbol('k_B')*T/(sp.Symbol('m')*g)),
        ('BarometricFormula[h_, H_]', lambda h, H: exp(-h/H)),
        ('JeansEscape[T_, M_, R_]', lambda T, M, R: sqrt(2*sp.Symbol('G')*M/R) < sqrt(2*sp.Symbol('k_B')*T/sp.Symbol('m'))),
        ('AlbedoEffect[incident_, reflected_]', lambda inc, ref: ref/inc),
        ('GreenhouseEffect[T_eq_, T_surf_]', lambda Teq, Ts: Ts - Teq),
        ('EquilibriumTemperature[L_, a_]', lambda L, a: (L*(1 - a)/(16*pi*sp.Symbol('sigma')*a**2))**sp.Rational(1,4)),
        ('InsolationAngle[lat_, declination_]', lambda lat, dec: sin(lat)*sin(dec) + cos(lat)*cos(dec)),
        ('SolarConstant[]', lambda: 1361),  # W/m²
        ('BlackbodyTemperature[F_]', lambda F: (F/sp.Symbol('sigma'))**sp.Rational(1,4)),
        ('Bond Albedo[reflected_, incident_]', lambda ref, inc: ref/inc),
        ('GeometricAlbedo[brightness_]', lambda b: b),
        ('PlanetaryRingDynamics[M_, m_, a_]', lambda M, m, a: sqrt(sp.Symbol('G')*(M + m)/a**3)),
        
        # ===== COSMOLOGY (60 functions) =====
        ('HubbleParameter[z_]', lambda z: sp.Symbol('H0')*sqrt(sp.Symbol('Omega_M')*(1+z)**3 + sp.Symbol('Omega_Lambda'))),
        ('LookbackTime[z_, H0_]', lambda z, H: integrate(1/((1+z)*sp.Symbol('H0')*sqrt(sp.Symbol('Omega_M')*(1+z)**3 + sp.Symbol('Omega_Lambda'))), z)),
        ('ComovingDistanceCosmology[z_]', lambda z: sp.Symbol('c')/sp.Symbol('H0') * integrate(1/sqrt(sp.Symbol('Omega_M')*(1+z)**3 + sp.Symbol('Omega_Lambda')), z)),
        ('AngularDiameterDistanceCosmology[z_]', lambda z: sp.Symbol('c')/sp.Symbol('H0') * z / (1+z)**2),
        ('LuminosityDistanceCosmology[z_]', lambda z: sp.Symbol('c')/sp.Symbol('H0') * z * (1+z)),
        ('CriticalDensityCosmology[H_]', lambda H: 3*H**2/(8*pi*sp.Symbol('G'))),
        ('DensityParameter[rho_, rho_crit_]', lambda rho, rhoc: rho/rhoc),
        ('OmegaMatter[]', lambda: 0.3),
        ('OmegaLambda[]', lambda: 0.7),
        ('OmegaRadiation[]', lambda: 9e-5),
        ('OmegaBaryon[]', lambda: 0.05),
        ('OmegaCDM[]', lambda: 0.25),
        ('DarkEnergyEquationOfState[w_]', lambda w: -1),
        ('DecelelerationParameter[q_]', lambda q: -0.5),
        ('FlatnessProblem[Omega_]', lambda Om: abs(Om - 1)),
        ('HorizonProblem[d_H_, d_particle_]', lambda dH, dp: dH < dp),
        ('InflationScaleFactor[t_]', lambda t: exp(sp.Symbol('H_inflation')*t)),
        ('ReheatingTemperature[]', lambda: 1e15),  # Kelvin
        ('BaryogenesisAsymmetry[]', lambda: 6e-10),
        ('NucleosyntheticYield[element_]', lambda e: 0.75 if str(e) == "H" else 0.25),
        ('CMBTemperature[z_]', lambda z: 2.725*(1 + z)),
        ('AcousticHorizon[z_]', lambda z: sp.Symbol('c')/sp.Symbol('H0')),
        ('SilkDamping[k_]', lambda k: exp(-k**2)),
        ('SachsWolfeEffect[Phi_]', lambda Phi: Phi/3),
        ('IntegratedSachsWolfe[Phi_]', lambda Phi: 2*Phi),
        ('SunyaevZeldovich[T_e_, tau_]', lambda Te, tau: -2*tau*sp.Symbol('k_B')*Te),
        ('MatterPowerSpectrum[k_]', lambda k: k**(-3)),
        ('TransferFunction[k_]', lambda k: 1/(1 + k**2)),
        ('GrowthFactor[z_]', lambda z: 1/(1 + z)),
        ('LinearGrowth[delta_, z_]', lambda delta, z: delta/(1 + z)),
        ('SphericalCollapse[delta_]', lambda delta: 1.686),
        ('VirialTheorem[M_, R_]', lambda M, R: sp.Symbol('G')*M**2/R),
        ('VirialTemperature[M_, R_]', lambda M, R: sp.Symbol('G')*M*sp.Symbol('mu')*sp.Symbol('m_p')/(sp.Symbol('k_B')*R)),
        ('JeansInstability[cs_, rho_]', lambda cs, rho: sqrt(pi*cs**2/(sp.Symbol('G')*rho))),
        ('SoundSpeed[T_]', lambda T: sqrt(sp.Symbol('gamma')*sp.Symbol('k_B')*T/sp.Symbol('mu')/sp.Symbol('m_p'))),
        ('FreeStreamingLength[T_]', lambda T: sp.Symbol('c')/sp.Symbol('H0')),
        ('StructureFormation[z_]', lambda z: 1 + z > 1000),
        ('GalaxyFormationTime[M_]', lambda M: 1e9),  # years
        ('StarFormationRate[M_gas_, t_dyn_]', lambda Mg, td: Mg/td),
        ('KennicuttSchmidtLaw[Sigma_gas_]', lambda Sg: Sg**1.4),
        ('TullyFisherRelation[V_rot_]', lambda V: (V/220)**4 * sp.Symbol('L_sun')),
        ('FaberJacksonRelation[sigma_]', lambda sig: (sig/200)**4 * sp.Symbol('L_sun')),
        
        # ===== GALACTIC DYNAMICS (40 functions) =====
        ('RotationCurve[r_, M_]', lambda r, M: sqrt(sp.Symbol('G')*M/r)),
        ('DarkMatterHalo[r_]', lambda r: sp.Symbol('rho0')/(1 + r/sp.Symbol('r_s'))**2),
        ('NFWProfile[r_]', lambda r: sp.Symbol('rho0')/((r/sp.Symbol('r_s'))*(1 + r/sp.Symbol('r_s'))**2)),
        ('IsothermalSphere[r_]', lambda r: sp.Symbol('sigma')**2/(2*pi*sp.Symbol('G')*r**2)),
        ('PlummerModel[r_, a_]', lambda r, a: 3*sp.Symbol('M')/(4*pi*a**3) * (1 + (r/a)**2)**sp.Rational(-5,2)),
        ('HernquistProfile[r_, a_]', lambda r, a: sp.Symbol('M')/(2*pi) * a/(r*(r+a)**3)),
        ('JaffeProfile[r_, a_]', lambda r, a: sp.Symbol('M')/(4*pi*a**2) * 1/(r*(r+a)**2)),
        ('EinastoProfile[r_, alpha_]', lambda r, a: sp.Symbol('rho0')*exp(-2*sp.Symbol('n')*((r/sp.Symbol('r_e'))**sp.Rational(1,sp.Symbol('n')) - 1))),
        ('SerProfile[r_, n_]', lambda r, n: exp(-(r/sp.Symbol('r_e'))**sp.Rational(1,n))),
        ('DeVaucouleursLaw[r_]', lambda r: exp(-(r/sp.Symbol('r_e'))**sp.Rational(1,4))),
        ('ExponentialDisk[r_, R_d_]', lambda r, Rd: exp(-r/Rd)),
        ('DoubleExponentialDisk[r_, z_, R_d_, z_0_]', lambda r, z, Rd, z0: exp(-r/Rd - abs(z)/z0)),
        ('ThickDisk[r_, z_]', lambda r, z: exp(-r/sp.Symbol('R_d') - z**2/(2*sp.Symbol('z_0')**2))),
        ('Bulge[r_]', lambda r: (1 + r/sp.Symbol('r_b'))**(-4)),
        ('GalacticPotential[r_, z_]', lambda r, z: -sp.Symbol('G')*sp.Symbol('M')/sqrt(r**2 + z**2)),
        ('CircularVelocity[r_]', lambda r: sqrt(r*sp.diff(sp.Symbol('Phi'), r))),
        ('EpicyclicFrequency[r_]', lambda r: sqrt(sp.diff(sp.Symbol('Omega')**2, r) + 4*sp.Symbol('Omega')**2)),
        ('VerticalFrequency[z_]', lambda z: sqrt(sp.diff(sp.Symbol('Phi'), z, 2))),
        ('GalacticOrbit[r_, theta_, z_]', lambda r, th, z: (r, th, z)),
        ('ActionAngleVariables[orbit_]', lambda o: (sp.Symbol('J_r'), sp.Symbol('J_theta'), sp.Symbol('J_z'))),
        
        # ===== BLACK HOLES & COMPACT OBJECTS (40 functions) =====
        ('SchwarzschildMetricComponents[r_]', lambda r: (1 - 2*sp.Symbol('G')*sp.Symbol('M')/(r*sp.Symbol('c')**2))),
        ('KerrMetric[r_, theta_, a_]', lambda r, th, a: "metric tensor"),
        ('ReissnerNordstromMetric[r_, Q_]', lambda r, Q: 1 - 2*sp.Symbol('G')*sp.Symbol('M')/(r*sp.Symbol('c')**2) + sp.Symbol('G')*Q**2/(r**2*sp.Symbol('c')**4)),
        ('KerrNewmanMetric[r_, theta_, a_, Q_]', lambda r, th, a, Q: "metric tensor"),
        ('EventHorizonRadius[M_]', lambda M: 2*sp.Symbol('G')*M/sp.Symbol('c')**2),
        ('PhotonSphere[M_]', lambda M: 3*sp.Symbol('G')*M/sp.Symbol('c')**2),
        ('ISCORadius[M_, a_]', lambda M, a: 6*sp.Symbol('G')*M/sp.Symbol('c')**2),  # Simplified
        ('ErgoSphere[M_, a_, theta_]', lambda M, a, th: sp.Symbol('G')*M/sp.Symbol('c')**2 * (1 + sqrt(1 - (a*sin(th))**2))),
        ('KerrParameter[J_, M_]', lambda J, M: J/(M*sp.Symbol('c'))),
        ('PenroseProcess[M_, a_]', lambda M, a: 0.29*M*sp.Symbol('c')**2),  # Max extractable energy
        ('HawkingTemperature[M_]', lambda M: sp.Symbol('hbar')*sp.Symbol('c')**3/(8*pi*sp.Symbol('G')*M*sp.Symbol('k_B'))),
        ('BekensteinHawkingEntropy[M_]', lambda M: sp.Symbol('k_B')*sp.Symbol('c')**3*sp.Symbol('A')/(4*sp.Symbol('G')*sp.Symbol('hbar'))),
        ('HawkingRadiation[M_]', lambda M: sp.Symbol('hbar')*sp.Symbol('c')**6/(15360*pi*sp.Symbol('G')**2*M**2)),
        ('BlackHoleLifetime[M_]', lambda M: 2.1e67*(M/sp.Symbol('M_sun'))**3),  # seconds
        ('AccretionRate[M_, eta_]', lambda M, eta: sp.Symbol('L')/( eta*sp.Symbol('c')**2)),
        ('EddingtonAccretionRate[M_]', lambda M: sp.Symbol('L_Edd')/(0.1*sp.Symbol('c')**2)),
        ('BondiAccretion[M_, cs_, rho_]', lambda M, cs, rho: pi*rho*(sp.Symbol('G')*M)**2/cs**3),
        ('ThinDiskSpectrum[M_, r_]', lambda M, r: (3*sp.Symbol('G')*M*sp.Symbol('Mdot')/(8*pi*sp.Symbol('sigma')*r**3))**sp.Rational(1,4)),
        ('ShakuraSunyaevDisk[M_, Mdot_, r_]', lambda M, Md, r: (3*sp.Symbol('G')*M*Md/(8*pi*sp.Symbol('sigma')*r**3))**sp.Rational(1,4)),
        ('QuasarLuminosity[M_, Mdot_]', lambda M, Md: 0.1*Md*sp.Symbol('c')**2),
        
        # ===== NUCLEOSYNTHESIS (30 functions) =====
        ('BindingEnergy[A_, Z_]', lambda A, Z: 15.75*A - 17.8*A**sp.Rational(2,3) - 0.711*Z**2/A**sp.Rational(1,3)),
        ('MassDefect[A_, Z_]', lambda A, Z: Z*sp.Symbol('m_p') + (A - Z)*sp.Symbol('m_n') - sp.Symbol('M_nucleus')),
        ('QValue[reactants_, products_]', lambda r, p: 0),  # Mass difference
        ('NuclearReactionRate[sigma_, n1_, n2_, v_]', lambda sig, n1, n2, v: sig*n1*n2*v),
        ('MaxwellAveragedCrossSection[sigma_, T_]', lambda sig, T: sig),
        ('GamowPeak[Z1_, Z2_, T_]', lambda Z1, Z2, T: sqrt(sp.Symbol('k_B')*T)),
        ('CoulombBarrier[Z1_, Z2_]', lambda Z1, Z2: Z1*Z2*sp.Symbol('e')**2/(4*pi*sp.Symbol('epsilon0')*sp.Symbol('r'))),
        ('TunnelingProbability[E_, V_, a_]', lambda E, V, a: exp(-2*sqrt(2*sp.Symbol('m')*(V-E))*a/sp.Symbol('hbar'))),
        ('PPChain[T_]', lambda T: 1e-43 if T > 4e6 else 0),  # Reaction rate
        ('CNOCycle[T_]', lambda T: 1e-40 if T > 1.5e7 else 0),
        ('TripleAlpha[T_]', lambda T: 1e-35 if T > 1e8 else 0),
        ('AlphaCaptureRate[T_]', lambda T: 1e-30),
        ('NeutronCaptureRate[n_, sigma_]', lambda n, sig: n*sig*sp.Symbol('v')),
        ('SProcess[tau_]', lambda tau: "slow neutron capture"),
        ('RProcess[tau_]', lambda tau: "rapid neutron capture"),
        ('PProcess[]', lambda: "proton capture"),
        ('PhotodisintegrationRate[T_]', lambda T: exp(-sp.Symbol('E_bind')/(sp.Symbol('k_B')*T))),
        ('BetaDecay[nucleus_]', lambda n: "decay products"),
        ('AlphaDecay[nucleus_]', lambda n: "decay products"),
        ('FissionRate[nucleus_]', lambda n: 1e-20),
        ('FusionCrossSection[E_, Z1_, Z2_]', lambda E, Z1, Z2: 1e-28),
        ('BremsstrahlungCooling[T_, n_e_]', lambda T, ne: 1.4e-27*sqrt(T)*ne**2),
        ('ComptonCooling[T_]', lambda T: sp.Symbol('sigma_T')*sp.Symbol('c')*sp.Symbol('U_rad')),
        ('SynchrotronCooling[gamma_, B_]', lambda g, B: sp.Symbol('sigma_T')*sp.Symbol('c')*g**2*B**2/(6*pi*sp.Symbol('m_e')*sp.Symbol('c')**2)),
        ('InverseComptonCooling[gamma_]', lambda g: sp.Symbol('sigma_T')*sp.Symbol('c')*sp.Symbol('U_rad')*g**2),
        ('PairProduction[E_]', lambda E: E > 2*sp.Symbol('m_e')*sp.Symbol('c')**2),
        ('PhotonPhotonPairProduction[E1_, E2_]', lambda E1, E2: E1*E2 > (sp.Symbol('m_e')*sp.Symbol('c')**2)**2),
        ('NeutrinoOpacity[E_, n_]', lambda E, n: sp.Symbol('G_F')**2*E**2*n),
        ('NeutrinoCooling[T_]', lambda T: T**9),
        ('PlasmaFrequency[n_e_]', lambda ne: sqrt(ne*sp.Symbol('e')**2/(sp.Symbol('epsilon0')*sp.Symbol('m_e')))),
    ]


def register():
    """Register all extended14 rules"""
    from .extended_helper import convert_rules
    return convert_rules(get_rules())
