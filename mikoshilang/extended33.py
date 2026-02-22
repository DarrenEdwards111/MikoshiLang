"""
MikoshiLang Extended Functions - Set 33
Agriculture, food science, sports science, and miscellaneous engineering
"""
import sympy as sp
from sympy import symbols, sqrt, exp, log, pi, sin, cos

def get_rules():
    """Extended set 33: Agriculture, food, sports, and specialized engineering (100 functions)"""
    x, y, t = symbols('x y t')
    
    return [
        # ===== AGRICULTURE & SOIL SCIENCE (25 functions) =====
        ('SoilMoisture[theta_]', lambda theta: theta),
        ('FieldCapacity[FC_]', lambda FC: FC),
        ('WiltingPoint[WP_]', lambda WP: WP),
        ('AvailableWaterCapacity[FC_, WP_]', lambda FC, WP: FC - WP),
        ('SoilWaterDeficit[FC_, current_]', lambda FC, curr: FC - curr),
        ('IrrigationRequirement[ET_, P_, Delta_S_]', lambda ET, P, dS: ET - P + dS),
        ('EvapotranspirationPenman[Rn_, T_, u_, e_s_, e_a_]', lambda Rn, T, u, es, ea: sp.Symbol('ET')),
        ('ReferenceET[T_, RH_, wind_, radiation_]', lambda T, RH, w, R: sp.Symbol('ET0')),
        ('CropCoefficient[Kc_]', lambda Kc: Kc),
        ('CropEvapotranspiration[ET0_, Kc_]', lambda ET0, Kc: ET0*Kc),
        ('GrowingDegreeDays[T_max_, T_min_, T_base_]', lambda Tmax, Tmin, Tbase: (Tmax + Tmin)/2 - Tbase),
        ('ChillingHours[T_]', lambda T: 1 if T < 7 else 0),
        ('HeatUnits[T_, T_base_]', lambda T, Tbase: max(0, T - Tbase)),
        ('LeafAreaIndex[leaf_area_, ground_area_]', lambda La, Ga: La/Ga),
        ('PhotosyntheticRate[light_, CO2_, T_]', lambda L, CO2, T: sp.Symbol('P')),
        ('SoilpH[H_concentration_]', lambda H: -log(H, 10)),
        ('CationExchangeCapacity[CEC_]', lambda CEC: CEC),
        ('SoilOrganicMatter[C_content_]', lambda C: C*1.724),  # Van Bemmelen factor
        ('BulkDensity[mass_, volume_]', lambda m, V: m/V),
        ('Porosity[bulk_density_, particle_density_]', lambda rhob, rhop: (1 - rhob/rhop)*100),
        ('HydraulicConductivity[K_]', lambda K: K),
        ('InfiltrationRate[I_]', lambda I: I),
        ('RunoffCoefficient[C_]', lambda C: C),
        ('SoilErosionUSLE[R_, K_, LS_, C_, P_]', lambda R, K, LS, C, P: R*K*LS*C*P),
        ('NutrientUptakeRate[C_, K_m_, V_max_]', lambda C, Km, Vmax: Vmax*C/(Km + C)),
        
        # ===== FOOD SCIENCE (25 functions) =====
        ('WaterActivity[P_, P0_]', lambda P, P0: P/P0),
        ('MoistureContent[wet_basis_]', lambda wb: wb/(1 - wb)*100),
        ('SolidContent[moisture_]', lambda m: 100 - m),
        ('Brix[sucrose_percent_]', lambda s: s),
        ('AcidityTA[acid_meq_, volume_]', lambda meq, V: meq/V),
        ('pHFood[H_]', lambda H: -log(H, 10)),
        ('TitrableAcidity[NaOH_vol_, normality_, sample_]', lambda V, N, s: V*N/s*100),
        ('BrowningIndex[A_420_]', lambda A420: A420*100),
        ('ColorDifference[L1_, a1_, b1_, L2_, a2_, b2_]', lambda L1, a1, b1, L2, a2, b2: sqrt((L2-L1)**2 + (a2-a1)**2 + (b2-b1)**2)),
        ('TextureHardness[force_]', lambda F: F),
        ('Viscosity[shear_stress_, shear_rate_]', lambda tau, gamma: tau/gamma),
        ('PowerLawFluid[K_, n_, gamma_]', lambda K, n, gamma: K*gamma**n),
        ('YieldStress[tau_0_]', lambda tau0: tau0),
        ('RheologyIndex[n_]', lambda n: n),
        ('FreezingPointDepression[molality_]', lambda m: 1.86*m),
        ('DValue[T_]', lambda T: sp.Symbol('D')),  # Decimal reduction time
        ('ZValue[microorganism_]', lambda m: sp.Symbol('z')),
        ('FValue[D_, T_ref_, z_]', lambda D, Tref, z: D*10**((sp.Symbol('T') - Tref)/z)),
        ('ThermalDeathTime[F0_]', lambda F0: F0),
        ('PasteurizationUnit[PU_]', lambda PU: PU),
        ('ShelfLifePrediction[k_, T_]', lambda k, T: log(2)/k),
        ('ArrheniusShelfLife[k0_, Ea_, R_, T_]', lambda k0, Ea, R, T: k0*exp(-Ea/(R*T))),
        ('OxygenPermeability[P_, A_, delta_x_, delta_p_]', lambda P, A, dx, dp: P*A*dp/dx),
        ('WaterVaporTransmissionRate[WVTR_]', lambda WVTR: WVTR),
        ('PackagingBarrierFactor[actual_rate_, ideal_rate_]', lambda actual, ideal: ideal/actual),
        
        # ===== SPORTS SCIENCE (20 functions) =====
        ('VO2Max[ml_min_, kg_]', lambda V, kg: V/kg),
        ('MaxHeartRate[age_]', lambda age: 220 - age),
        ('TargetHeartRateZone[max_HR_, intensity_]', lambda maxHR, i: maxHR*i),
        ('KarvonenFormula[max_HR_, resting_HR_, intensity_]', lambda maxHR, restHR, i: (maxHR - restHR)*i + restHR),
        ('CaloriesBurned[MET_, weight_, duration_]', lambda MET, w, d: MET*w*d/60),
        ('RunningPower[mass_, velocity_, grade_, Cair_]', lambda m, v, gr, Cair: m*9.81*v*gr + Cair*v**3),
        ('CriticalPower[W_max_, AWC_, t_]', lambda Wmax, AWC, t: Wmax - AWC/t),
        ('AnaerobicWorkCapacity[AWC_]', lambda AWC: AWC),
        ('FunctionalThresholdPower[FTP_]', lambda FTP: FTP),
        ('PowerToWeightRatio[watts_, kg_]', lambda W, kg: W/kg),
        ('CyclingEfficiency[power_out_, power_in_]', lambda Pout, Pin: Pout/Pin*100),
        ('SwimmingPace[time_, distance_]', lambda t, d: t/d),
        ('RunningEconomy[VO2_, velocity_]', lambda VO2, v: VO2/v),
        ('StrideLength[distance_, steps_]', lambda d, s: d/s),
        ('StrideFrequency[steps_, time_]', lambda s, t: s/t),
        ('GroundContactTime[t_contact_]', lambda tc: tc),
        ('VerticalOscillation[height_]', lambda h: h),
        ('LactateThreshold[lactate_]', lambda lac: lac),
        ('RespiratoryExchangeRatio[VCO2_, VO2_]', lambda VCO2, VO2: VCO2/VO2),
        ('TrainingLoad[duration_, intensity_]', lambda d, i: d*i),
        
        # ===== TEXTILE ENGINEERING (15 functions) =====
        ('YarnCount[weight_, length_]', lambda w, l: l/w),
        ('Tex[grams_, km_]', lambda g, km: g/km),
        ('Denier[grams_, 9000m_]', lambda g, L: g/L*9000),
        ('FabricGSM[weight_, area_]', lambda w, A: w/A*1000),  # Grams per square meter
        ('ThreadsPerInch[TPI_]', lambda TPI: TPI),
        ('FabricCoverFactor[threads_, yarn_diameter_]', lambda n, d: n*d),
        ('TensilleStrength[force_, area_]', lambda F, A: F/A),
        ('Elongation[delta_L_, L0_]', lambda dL, L0: dL/L0*100),
        ('YoungModulusTextile[stress_, strain_]', lambda sig, eps: sig/eps),
        ('FabricThickness[t_]', lambda t: t),
        ('AirPermeability[Q_, A_, delta_P_]', lambda Q, A, dP: Q/(A*dP)),
        ('MoistureRegain[W_water_, W_dry_]', lambda Ww, Wd: (Ww - Wd)/Wd*100),
        ('WaterAbsorption[W_wet_, W_dry_]', lambda Wwet, Wdry: (Wwet - Wdry)/Wdry*100),
        ('ThermalResistance[thickness_, conductivity_]', lambda t, k: t/k),
        ('ClothingInsulation[Clo_]', lambda Clo: Clo*0.155),  # Clo to m²·K/W
        
        # ===== PETROLEUM ENGINEERING (15 functions) =====
        ('OilGravityAPI[SG_]', lambda SG: 141.5/SG - 131.5),
        ('GasOilRatio[gas_volume_, oil_volume_]', lambda Vg, Vo: Vg/Vo),
        ('FormationVolumeFactor[V_reservoir_, V_surface_]', lambda Vr, Vs: Vr/Vs),
        ('ReservoirPressure[depth_, gradient_]', lambda d, g: d*g),
        ('DarcyLaw[Q_, k_, A_, mu_, L_, delta_P_]', lambda Q, k, A, mu, L, dP: Q - k*A*dP/(mu*L)),
        ('Permeability[Q_, mu_, L_, A_, delta_P_]', lambda Q, mu, L, A, dP: Q*mu*L/(A*dP)),
        ('PorousMediaFlow[k_, phi_]', lambda k, phi: k*phi),
        ('RecoveryFactor[produced_, OOIP_]', lambda prod, OOIP: prod/OOIP*100),
        ('OilInPlace[A_, h_, phi_, Sw_, Bo_]', lambda A, h, phi, Sw, Bo: 7758*A*h*phi*(1 - Sw)/Bo),
        ('GasInPlace[A_, h_, phi_, Sw_, Bg_]', lambda A, h, phi, Sw, Bg: 43560*A*h*phi*(1 - Sw)/Bg),
        ('WaterSaturation[Sw_]', lambda Sw: Sw),
        ('ResidualOilSaturation[Sor_]', lambda Sor: Sor),
        ('CapillaryPressure[sigma_, theta_, r_]', lambda sig, theta, r: 2*sig*cos(theta)/r),
        ('RelativePermeability[k_rel_]', lambda krel: krel),
        ('WellProductivityIndex[Q_, P_res_, P_well_]', lambda Q, Pres, Pwell: Q/(Pres - Pwell)),
    ]


def register():
    """Register all extended33 rules"""
    from .extended_helper import convert_rules
    return convert_rules(get_rules())
