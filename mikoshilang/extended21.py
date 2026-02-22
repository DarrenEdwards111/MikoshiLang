"""
MikoshiLang Extended Functions - Set 21
Climate science, meteorology, and atmospheric physics
"""
import sympy as sp
from sympy import symbols, sqrt, exp, log, pi, sin, cos

def get_rules():
    """Extended set 21: Climate and atmospheric science (150 functions)"""
    T, P, rho, z = symbols('T P rho z')
    
    return [
        # ===== ATMOSPHERIC THERMODYNAMICS (40 functions) =====
        ('IdealGasLaw[P_, V_, n_, R_, T_]', lambda P, V, n, R, T: P*V - n*R*T),
        ('BarometricFormula[P0_, M_, g_, R_, T_, h_]', lambda P0, M, g, R, T, h: P0*exp(-M*g*h/(R*T))),
        ('HypsometricEquation[T_, P1_, P2_]', lambda T, P1, P2: sp.Symbol('R')*T/sp.Symbol('g')*log(P1/P2)),
        ('PotentialTemperature[T_, P_, P0_]', lambda T, P, P0: T*(P0/P)**sp.Rational(2,7)),
        ('VirtualTemperature[T_, q_]', lambda T, q: T*(1 + 0.608*q)),
        ('EquivalentTemperature[T_, q_]', lambda T, q: T + sp.Symbol('L_v')*q/sp.Symbol('c_p')),
        ('WetBulbTemperature[T_, RH_]', lambda T, RH: T*atan2(0.151977*sqrt(RH + 8.313659), 1) + atan2(T + RH, 1) - atan2(RH - 1.676331, 1) + 0.00391838*RH**1.5*atan2(0.023101*RH, 1) - 4.686035),
        ('DewPoint[T_, RH_]', lambda T, RH: T - (100 - RH)/5),  # Simplified
        ('VaporPressure[T_]', lambda T: 6.112*exp(17.67*T/(T + 243.5))),  # Magnus formula
        ('SaturationVaporPressure[T_]', lambda T: 6.112*exp(17.67*T/(T + 243.5))),
        ('RelativeHumidity[e_, es_]', lambda e, es: e/es * 100),
        ('SpecificHumidity[e_, P_]', lambda e, P: 0.622*e/(P - 0.378*e)),
        ('MixingRatio[e_, P_]', lambda e, P: 0.622*e/(P - e)),
        ('AbsoluteHumidity[e_, T_]', lambda e, T: e/(sp.Symbol('R_v')*T)),
        ('LapseRateD ry[g_, cp_]', lambda g, cp: g/cp),
        ('LapseRateMoist[gamma_d_, L_, e_s_]', lambda gd, L, es: gd*(1 + L*es/(sp.Symbol('R')*sp.Symbol('T')**2))/(1 + L**2*es/(sp.Symbol('c_p')*sp.Symbol('R_v')*sp.Symbol('T')**2))),
        ('AdiabaticCooling[gamma_, delta_z_]', lambda gamma, dz: gamma*dz),
        ('LiftedCondensationLevel[T_, Td_]', lambda T, Td: 125*(T - Td)),  # Meters
        ('LevelOfFreeConvection[T_parcel_, T_env_]', lambda Tp, Te: Tp > Te),
        ('ConvectiveAvailablePotentialEnergy[T_parcel_, T_env_, z_]', lambda Tp, Te, z: sp.Symbol('g')*sp.integrate((Tp - Te)/Te, z)),
        ('ConvectiveInhibition[T_parcel_, T_env_, z_]', lambda Tp, Te, z: -sp.Symbol('g')*sp.integrate((Tp - Te)/Te, z)),
        ('BruntVaisalaFrequency[g_, T_, dT_dz_]', lambda g, T, dTdz: sqrt(g/T*dTdz)),
        ('RichardsonNumberAtmos[g_, T_, dT_dz_, du_dz_]', lambda g, T, dTdz, dudz: g/T*dTdz/dudz**2),
        ('StaticStability[dtheta_dz_]', lambda dthetadz: dthetadz > 0),
        ('ConditionalInstability[gamma_, gamma_m_]', lambda gamma, gammam: gamma > gammam),
        ('AbsoluteInstability[gamma_, gamma_d_]', lambda gamma, gammad: gamma > gammad),
        ('EquivalentPotentialTemperature[theta_, q_]', lambda theta, q: theta*exp(sp.Symbol('L_v')*q/(sp.Symbol('c_p')*sp.Symbol('T')))),
        ('CloudBase[LCL_]', lambda LCL: LCL),
        ('CloudTop[T_, gamma_]', lambda T, gamma: T/gamma),
        ('PrecipitableWater[q_, P_]', lambda q, P: sp.integrate(q*P/sp.Symbol('g'), P)),
        ('LatentHeatVaporization[T_]', lambda T: 2.5e6 - 2400*T),  # J/kg
        ('LatentHeatFusion[]', lambda: 3.34e5),  # J/kg
        ('LatentHeatSublimation[]', lambda: 2.83e6),  # J/kg
        ('ClaudiusClapeyron[L_, T_]', lambda L, T: L/(sp.Symbol('R_v')*T**2)),
        ('BoltonFormula[T_, RH_]', lambda T, RH: T - (100 - RH)/5),
        ('TetensFormula[T_]', lambda T: 6.112*exp(17.27*T/(T + 237.3))),
        ('GoffGratchEquation[T_]', lambda T: 6.112*exp(17.67*T/(T + 243.5))),
        ('WexlerEquation[T_]', lambda T: exp(-5800.2206/T + 1.3914993 - 0.048640239*T)),
        ('MagnusFormula[T_]', lambda T: 6.112*exp(17.67*T/(T + 243.5))),
        ('AugustRocheMagnus[T_]', lambda T: 6.1094*exp(17.625*T/(T + 243.04))),
        
        # ===== RADIATION & ENERGY BALANCE (30 functions) =====
        ('SolarConstant[]', lambda: 1361),  # W/m²
        ('SolarZenithAngle[lat_, decl_, hour_]', lambda lat, decl, h: sp.acos(sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(h))),
        ('SolarDeclination[day_]', lambda d: 23.45*sin(2*pi*(d + 284)/365)),
        ('DayLength[lat_, decl_]', lambda lat, decl: 24/pi*sp.acos(-tan(lat)*tan(decl))),
        ('Insolation[S0_, theta_]', lambda S0, theta: S0*cos(theta)),
        ('TopOfAtmosphereRadiation[S0_, distance_factor_]', lambda S0, df: S0*df),
        ('AlbedoSurface[reflected_, incident_]', lambda refl, inc: refl/inc),
        ('PlanetaryAlbedo[]', lambda: 0.30),  # Earth average
        ('GreenhouseEffect[T_surface_, T_effective_]', lambda Ts, Te: Ts - Te),
        ('BlackbodyRadiation[T_]', lambda T: sp.Symbol('sigma')*T**4),
        ('WienDisplacementLaw[T_]', lambda T: 2898/T),  # μm
        ('StefanBoltzmann[T_]', lambda T: 5.67e-8*T**4),
        ('RayleighScattering[lambda_]', lambda lam: lam**(-4)),
        ('MieScattering[r_, lambda_]', lambda r, lam: r/lam),
        ('OpticalDepth[tau_]', lambda tau: tau),
        ('TransmittanceAtmospheric[tau_]', lambda tau: exp(-tau)),
        ('BeerLambertLaw[I0_, k_, x_]', lambda I0, k, x: I0*exp(-k*x)),
        ('AbsorptionCoefficient[optical_depth_, path_]', lambda tau, x: tau/x),
        ('ScatteringCoefficient[tau_scat_, path_]', lambda taus, x: taus/x),
        ('SingleScatteringAlbedo[tau_scat_, tau_total_]', lambda taus, taut: taus/taut),
        ('AsymmetryParameter[g_]', lambda g: g),  # -1 to 1
        ('AerosolOpticalDepth[tau_]', lambda tau: tau),
        ('CloudOpticalDepth[LWP_, r_eff_]', lambda LWP, reff: 1.5*LWP/reff),
        ('LongwaveRadiation[T_, epsilon_]', lambda T, eps: eps*sp.Symbol('sigma')*T**4),
        ('NetRadiation[SW_down_, SW_up_, LW_down_, LW_up_]', lambda SWd, SWu, LWd, LWu: SWd - SWu + LWd - LWu),
        ('RadiativeForcing[delta_F_]', lambda dF: dF),
        ('CO2RadiativeForcing[C_, C0_]', lambda C, C0: 5.35*log(C/C0)),
        ('ClimatesensitivityParameter[lambda_]', lambda lam: lam),  # °C/(W/m²)
        ('EquilibriumClimateSensitivity[lambda_, forcing_]', lambda lam, F: lam*F),
        ('TransientClimateResponse[warming_]', lambda w: w),
        
        # ===== ATMOSPHERIC DYNAMICS (30 functions) =====
        ('GeostrophicWind[f_, dp_dy_]', lambda f, dpdy: -dpdy/(sp.Symbol('rho')*f)),
        ('AgeostrophicWind[v_total_, v_geostrophic_]', lambda vt, vg: vt - vg),
        ('ThermalWind[f_, dT_dy_, z_]', lambda f, dTdy, z: -sp.Symbol('g')/f*dTdy/sp.Symbol('T')*z),
        ('CoriolisParameter[lat_]', lambda lat: 2*sp.Symbol('Omega')*sin(lat)),
        ('RossbyNumber[U_, L_, f_]', lambda U, L, f: U/(L*f)),
        ('EkmanLayer[nu_, f_]', lambda nu, f: sqrt(2*nu/f)),
        ('EkmanSpiral[z_, f_, nu_]', lambda z, f, nu: exp(-z*sqrt(f/(2*nu)))),
        ('PressureGradientForce[dp_dx_, rho_]', lambda dpdx, rho: -dpdx/rho),
        ('GradientWind[f_, R_, dp_dr_]', lambda f, R, dpdr: -R*f + sqrt(R**2*f**2 + R*dpdr/sp.Symbol('rho'))),
        ('CyclostrophicWind[R_, dp_dr_]', lambda R, dpdr: sqrt(R*dpdr/sp.Symbol('rho'))),
        ('VorticityRelative[du_dy_, dv_dx_]', lambda dudy, dvdx: dvdx - dudy),
        ('VorticityAbsolute[zeta_rel_, f_]', lambda zetar, f: zetar + f),
        ('VorticityPotential[zeta_abs_, h_]', lambda zetaa, h: (zetaa + sp.Symbol('f'))/h),
        ('Divergence[du_dx_, dv_dy_]', lambda dudx, dvdy: dudx + dvdy),
        ('Deformation[du_dx_, dv_dy_, du_dy_, dv_dx_]', lambda dudx, dvdy, dudy, dvdx: sqrt((dudx - dvdy)**2 + (dudy + dvdx)**2)),
        ('VorticityAdvection[v_, grad_zeta_]', lambda v, gradz: -v*gradz),
        ('TemperatureAdvection[v_, grad_T_]', lambda v, gradT: -v*gradT),
        ('OmegaEquation[omega_]', lambda omega: omega),  # Vertical velocity
        ('QuasiGeostrophicVorticity[f_, omega_]', lambda f, omega: sp.diff(omega, sp.Symbol('p'))),
        ('RossbyWave[beta_, k_, l_]', lambda beta, k, l: -beta*k/(k**2 + l**2)),
        ('BaroclinicInstability[f_, N_, dT_dy_]', lambda f, N, dTdy: dTdy > 0),
        ('BarotropicInstability[du_dy_]', lambda dudy: dudy != 0),
        ('ErtelPotentialVorticity[zeta_, theta_]', lambda zeta, theta: (zeta + sp.Symbol('f'))*sp.diff(theta, sp.Symbol('p'))),
        ('IsentropicPV[theta_, u_, v_]', lambda theta, u, v: sp.Symbol('g')*(sp.Symbol('f') + sp.diff(v, sp.Symbol('x')) - sp.diff(u, sp.Symbol('y')))*sp.diff(theta, sp.Symbol('p'))),
        ('MountainWaves[N_, U_, h_]', lambda N, U, h: N*h/U),
        ('GravityWaveDispersion[N_, k_]', lambda N, k: N/k),
        ('KelvinWave[c_, y_, R_]', lambda c, y, R: exp(-y**2/(2*R**2))),
        ('MixedRossbyGravity[beta_, c_]', lambda beta, c: sqrt(beta*c)),
        ('EquatorialBetaPlane[beta_]', lambda beta: beta),
        ('HadleyCirculation[T_equator_, T_pole_]', lambda Teq, Tpole: Teq - Tpole),
        
        # ===== CLOUD & PRECIPITATION PHYSICS (25 functions) =====
        ('DropletGrowth[r_, S_]', lambda r, S: S*sp.Symbol('D_v')/r),
        ('KohlerCurve[r_, S_]', lambda r, S: S - 1),
        ('CriticalSupersaturation[r_]', lambda r: sqrt(4*sp.Symbol('A')**3/(27*r**3))),
        ('CriticalRadius[S_]', lambda S: sqrt(3*sp.Symbol('A')/S)),
        ('CondensationRate[S_, N_]', lambda S, N: S*N*sp.Symbol('D_v')),
        ('CollisionEfficiency[r1_, r2_]', lambda r1, r2: (r1 + r2)**2/(4*r1*r2)),
        ('CoalescenceEfficiency[E_collision_]', lambda Ec: Ec),
        ('TerminalVelocityDrop[r_]', lambda r: sp.Symbol('g')*r**2/(18*sp.Symbol('mu'))),  # Stokes
        ('RaindropSizeDistribution[N0_, lambda_, D_]', lambda N0, lam, D: N0*exp(-lam*D)),
        ('MarshallPalmer[R_]', lambda R: 8000*exp(-41*sp.Symbol('D')*R**(-0.21))),
        ('ReflectivityFactor[Z_]', lambda Z: Z),  # mm⁶/m³
        ('RadarReflectivity[Z_]', lambda Z: 10*log(Z, 10)),  # dBZ
        ('ZRRelation[Z_]', lambda Z: (Z/200)**sp.Rational(1,1.6)),  # Rainfall rate
        ('IceNucleation[T_]', lambda T: T < 273.15),
        ('HomogeneousFreezing[T_]', lambda T: T < 235),
        ('HeterogeneousFreezing[T_, IN_]', lambda T, IN: T < 273.15 and IN > 0),
        ('DepositionGrowth[S_ice_, D_v_]', lambda Si, Dv: Si*Dv),
        ('RimingRate[LWC_, v_]', lambda LWC, v: LWC*v*sp.Symbol('E')),
        ('AggregationRate[N_, v_]', lambda N, v: N**2*v*sp.Symbol('E')),
        ('SnowflakeMass[D_]', lambda D: sp.Symbol('a')*D**sp.Symbol('b')),
        ('GraupelDensity[rho_]', lambda rho: rho),  # 0.1-0.9 g/cm³
        ('HailGrowth[LWC_, v_, D_]', lambda LWC, v, D: LWC*v*pi*D**2/4),
        ('LiquidWaterContent[rho_L_, r_]', lambda rhoL, r: sp.Rational(4,3)*pi*rhoL*r**3*sp.Symbol('N')),
        ('CloudWaterPath[LWC_, H_]', lambda LWC, H: LWC*H),
        ('IceWaterPath[IWC_, H_]', lambda IWC, H: IWC*H),
        
        # ===== CLIMATE INDICES & PHENOMENA (25 functions) =====
        ('ElNinoIndex[SST_]', lambda SST: SST),  # Niño3.4 anomaly
        ('SouthernOscillationIndex[P_Tahiti_, P_Darwin_]', lambda PT, PD: PT - PD),
        ('NorthAtlanticOscillation[P_Iceland_, P_Azores_]', lambda PI, PA: PI - PA),
        ('ArcticOscillation[SLP_]', lambda SLP: SLP),  # First EOF of SLP
        ('PacificDecadalOscillation[SST_]', lambda SST: SST),  # First PC
        ('AtlanticMultidecadalOscillation[SST_]', lambda SST: SST),
        ('IndianOceanDipole[SST_west_, SST_east_]', lambda SSTw, SSTe: SSTw - SSTe),
        ('MaddenJulianOscillation[RMM1_, RMM2_]', lambda RMM1, RMM2: sqrt(RMM1**2 + RMM2**2)),
        ('QuasiBiennialOscillation[U_]', lambda U: U),  # Zonal wind at 30 hPa
        ('AnnularMode[EOF_]', lambda EOF: EOF),
        ('TeleconnectionPattern[corr_field_]', lambda corr: corr),
        ('WalkerCirculation[SST_gradient_]', lambda dSST: dSST),
        ('MonsoonIndex[precip_]', lambda P: P),
        ('DroughtIndex[P_, PET_]', lambda P, PET: P - PET),
        ('StandardizedPrecipitationIndex[P_]', lambda P: (P - sp.Symbol('mean'))/sp.Symbol('std')),
        ('PalmerDroughtIndex[P_, PET_, soil_]', lambda P, PET, soil: P - PET + soil),
        ('HeatWaveIndex[T_, threshold_, duration_]', lambda T, th, dur: T > th for dur),
        ('FreezingDegreedays[T_]', lambda T: sum(max(0, 0 - Ti) for Ti in T)),
        ('GrowingDegreeDays[T_, T_base_]', lambda T, Tb: sum(max(0, Ti - Tb) for Ti in T)),
        ('VorticityIndex[vort_]', lambda vort: vort),
        ('BlockingIndex[Z500_]', lambda Z: Z),
        ('JetStreamIndex[u_max_]', lambda umax: umax),
        ('StormTrackActivity[variance_]', lambda var: var),
        ('StratosphericSuddenWarming[T_strat_]', lambda Tstrat: Tstrat > sp.Symbol('threshold')),
        ('PolarVortexStrength[u_60N_]', lambda u: u),
    ]

def register():
    """Register all extended21 rules"""
    return get_rules()
