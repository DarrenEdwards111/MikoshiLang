"""
MikoshiLang Extended Functions - Set 22
Medical sciences, pharmacology, and biomechanics
"""
import sympy as sp
from sympy import symbols, sqrt, exp, log, pi

def get_rules():
    """Extended set 22: Medical and pharmacological sciences (150 functions)"""
    t, C, V = symbols('t C V')
    
    return [
        # ===== PHARMACOKINETICS (40 functions) =====
        ('OneCompartmentModel[C0_, k_, t_]', lambda C0, k, t: C0*exp(-k*t)),
        ('TwoCompartmentModel[A_, B_, alpha_, beta_, t_]', lambda A, B, alpha, beta, t: A*exp(-alpha*t) + B*exp(-beta*t)),
        ('ThreeCompartmentModel[C1_, C2_, C3_, k1_, k2_, k3_, t_]', lambda C1, C2, C3, k1, k2, k3, t: C1*exp(-k1*t) + C2*exp(-k2*t) + C3*exp(-k3*t)),
        ('AbsorptionRateConstant[ka_]', lambda ka: ka),
        ('EliminationRateConstant[ke_]', lambda ke: ke),
        ('DistributionRateConstant[k12_]', lambda k12: k12),
        ('CentralCompartment[V1_, C1_]', lambda V1, C1: V1*C1),
        ('PeripheralCompartment[V2_, C2_]', lambda V2, C2: V2*C2),
        ('MassBalance[amount_in_, amount_out_]', lambda ain, aout: ain - aout),
        ('Bioavailability[AUC_po_, AUC_iv_, dose_iv_, dose_po_]', lambda AUCpo, AUCiv, Div, Dpo: (AUCpo/AUCiv)*(Div/Dpo)),
        ('FirstPassMetabolism[F_]', lambda F: 1 - F),
        ('ExtractionRatio[Cl_organ_, Q_]', lambda Cl, Q: Cl/Q),
        ('HepaticClearance[Q_, E_]', lambda Q, E: Q*E),
        ('RenalClearance[U_, P_]', lambda U, P: U/P),  # Urine conc / plasma conc * urine flow
        ('TotalBodyClearance[Cl_hepatic_, Cl_renal_, Cl_other_]', lambda Clh, Clr, Clo: Clh + Clr + Clo),
        ('IntrinsicClearance[Vmax_, Km_]', lambda Vmax, Km: Vmax/Km),
        ('UnboundFraction[fu_]', lambda fu: fu),
        ('ProteinBinding[fu_]', lambda fu: 1 - fu),
        ('VolumeOfDistributionSteadyState[Vd_]', lambda Vd: Vd),
        ('ApparentVolumeOfDistribution[amount_, concentration_]', lambda amt, C: amt/C),
        ('MeanResidenceTime[AUMC_, AUC_]', lambda AUMC, AUC: AUMC/AUC),
        ('MeanAbsorptionTime[MAT_]', lambda MAT: MAT),
        ('TimeToMaxConcentration[ka_, ke_]', lambda ka, ke: log(ka/ke)/(ka - ke)),
        ('MaxConcentration[dose_, F_, V_, ka_, ke_]', lambda D, F, V, ka, ke: D*F/V*ka/(ka - ke)),
        ('AUCInfinity[C0_, ke_]', lambda C0, ke: C0/ke),
        ('AUCTrapezoid[C1_, C2_, delta_t_]', lambda C1, C2, dt: (C1 + C2)/2*dt),
        ('SteadyStateConcentrationAverage[dose_, tau_, Cl_]', lambda D, tau, Cl: D/(tau*Cl)),
        ('SteadyStateConcentrationMax[Css_avg_, ka_, ke_, tau_]', lambda Cssavg, ka, ke, tau: Cssavg*ka/(ka - ke)*(1 - exp(-ke*tau))),
        ('SteadyStateConcentrationMin[Css_max_, ke_, tau_]', lambda Cssmax, ke, tau: Cssmax*exp(-ke*tau)),
        ('FluctuationDegree[Css_max_, Css_min_]', lambda Cmax, Cmin: (Cmax - Cmin)/Cmin*100),
        ('AccumulationIndex[Css_, C1_]', lambda Css, C1: Css/C1),
        ('LoadingDosePK[Vd_, Css_target_]', lambda Vd, Csst: Vd*Csst),
        ('MaintenanceDosePK[Cl_, Css_, tau_]', lambda Cl, Css, tau: Cl*Css*tau),
        ('DoseAdjustmentRenal[dose_normal_, CLcr_, CLcr_normal_]', lambda Dn, CLcr, CLcrn: Dn*CLcr/CLcrn),
        ('CrockcroftGault[age_, weight_, Scr_, sex_]', lambda age, wt, Scr, sex: (140 - age)*wt/(72*Scr)*(0.85 if str(sex) == 'F' else 1)),
        ('MDRDFormula[Scr_, age_, sex_]', lambda Scr, age, sex: 175*Scr**(-1.154)*age**(-0.203)*(0.742 if str(sex) == 'F' else 1)),
        ('ChildPughScore[bili_, alb_, PT_, ascites_, enceph_]', lambda bili, alb, PT, asc, enc: bili + alb + PT + asc + enc),
        ('TherapeuticRange[Cmin_, Cmax_]', lambda Cmin, Cmax: (Cmin, Cmax)),
        ('ToxicConcentration[Ctox_]', lambda Ctox: Ctox),
        ('SubtherapeuticConcentration[Csub_]', lambda Csub: Csub),
        
        # ===== PHARMACODYNAMICS (30 functions) =====
        ('DoseResponseCurve[dose_, response_]', lambda D, R: (D, R)),
        ('EC50[dose_]', lambda D: D),  # Effective concentration for 50% response
        ('ED50[dose_]', lambda D: D),  # Effective dose for 50% response
        ('IC50[concentration_]', lambda C: C),  # Inhibitory concentration 50%
        ('LD50[dose_]', lambda D: D),  # Lethal dose 50%
        ('TherapeuticWindowPD[ED50_, TD50_]', lambda ED50, TD50: TD50/ED50),
        ('EMaxModel[E0_, Emax_, C_, EC50_]', lambda E0, Emax, C, EC50: E0 + Emax*C/(EC50 + C)),
        ('SigmoidEMax[E0_, Emax_, C_, EC50_, gamma_]', lambda E0, Emax, C, EC50, gamma: E0 + Emax*C**gamma/(EC50**gamma + C**gamma)),
        ('LinearModel[E0_, slope_, C_]', lambda E0, m, C: E0 + m*C),
        ('LogLinearModel[E0_, slope_, log_C_]', lambda E0, m, logC: E0 + m*logC),
        ('InhibitoryEMax[E0_, Imax_, C_, IC50_]', lambda E0, Imax, C, IC50: E0 - Imax*C/(IC50 + C)),
        ('AgonistActivity[efficacy_, EC50_]', lambda eff, EC50: eff/EC50),
        ('AntagonismCompetitive[Kd_agonist_, Kantagonist_, antagonist_conc_]', lambda Kda, Ka, Ca: Kda*(1 + Ca/Ka)),
        ('AntagonismNoncompetitive[Emax_, antagonist_]', lambda Emax, antag: Emax/(1 + antag)),
        ('PartialAgonist[E_max_partial_, E_max_full_]', lambda Emaxp, Emaxf: Emaxp/Emaxf),
        ('InverseAgonist[basal_, inhibition_]', lambda basal, inhib: basal - inhib),
        ('Desensitization[response_initial_, response_sustained_]', lambda Ri, Rs: (Ri - Rs)/Ri),
        ('Tachyphylaxis[response_decrease_]', lambda dec: dec),
        ('Tolerance[dose_initial_, dose_required_]', lambda Di, Dr: Dr/Di),
        ('Sensitization[response_increase_]', lambda inc: inc),
        ('Hysteresis[C_, E_]', lambda C, E: E - sp.Symbol('E_predicted')),
        ('EffectCompartmentModel[Ce_, keo_]', lambda Ce, keo: sp.diff(Ce, sp.Symbol('t')) + keo*Ce),
        ('ReceptorOccupancyTheory[Kd_, L_]', lambda Kd, L: L/(Kd + L)),
        ('SparereceptorTheory[occupancy_, response_]', lambda occ, resp: resp > occ),
        ('AllosericModulation[binding_, modulator_]', lambda bind, mod: bind*mod),
        ('Synergism[effect_combined_, effect_individual_]', lambda Ec, Ei: Ec > sum(Ei)),
        ('Antagonism[effect_combined_, effect_individual_]', lambda Ec, Ei: Ec < sum(Ei)),
        ('AdditiveEffect[E1_, E2_]', lambda E1, E2: E1 + E2),
        ('PotentiationEffect[E_enhanced_]', lambda Ee: Ee),
        ('FunctionalAntagonism[effect1_, effect2_]', lambda E1, E2: E1 - E2),
        
        # ===== TOXICOLOGY (25 functions) =====
        ('LethalDose[LD50_]', lambda LD50: LD50),
        ('NoObservedAdverseEffectLevel[NOAEL_]', lambda NOAEL: NOAEL),
        ('LowestObservedAdverseEffectLevel[LOAEL_]', lambda LOAEL: LOAEL),
        ('AcceptableDailyIntake[NOAEL_, safety_factor_]', lambda NOAEL, SF: NOAEL/SF),
        ('ReferenceOose[RfD_]', lambda RfD: RfD),
        ('MarginOfSafety[NOAEL_, exposure_]', lambda NOAEL, exp: NOAEL/exp),
        ('HazardQuotient[exposure_, RfD_]', lambda exp, RfD: exp/RfD),
        ('CumulativeRisk[HQ_]', lambda HQ: sum(HQ)),
        ('BenchmarkDose[BMD_]', lambda BMD: BMD),
        ('PointOfDeparture[POD_]', lambda POD: POD),
        ('DoseAdditionModel[doses_]', lambda doses: sum(doses)),
        ('ResponseAdditionModel[responses_]', lambda resp: 1 - sp.Mul(*[1 - ri for ri in resp])),
        ('InteractionIndex[toxicity_combined_, toxicity_expected_]', lambda Tc, Te: Tc/Te),
        ('CytotoxicityAssay[viable_cells_, total_cells_]', lambda viable, total: (1 - viable/total)*100),
        ('GenotoxicityIndex[damage_]', lambda dam: dam),
        ('MutagenicityTest[mutations_]', lambda mut: mut),
        ('CarcinogenicityRisk[slope_factor_, dose_]', lambda SF, D: SF*D),
        ('ExposureLimit[TLV_]', lambda TLV: TLV),  # Threshold limit value
        ('PermissibleExposureLimit[PEL_]', lambda PEL: PEL),
        ('ImmediateDangerous ToLifeOrHealth[IDLH_]', lambda IDLH: IDLH),
        ('AcuteExposureGuideline[AEGL_]', lambda AEGL: AEGL),
        ('EmergencyResponsePlanningGuideline[ERPG_]', lambda ERPG: ERPG),
        ('BiomonitoringEquivalent[BE_]', lambda BE: BE),
        ('InternalDose[absorbed_, metabolized_]', lambda abs, met: abs - met),
        ('TargetOrganDose[internal_dose_, distribution_]', lambda ID, dist: ID*dist),
        
        # ===== CLINICAL MEASUREMENTS (30 functions) =====
        ('BMI[weight_, height_]', lambda wt, ht: wt/ht**2),  # Alias for BodyMassIndex
        ('BodyMassIndex[weight_, height_]', lambda wt, ht: wt/ht**2),
        ('BodySurfaceArea[height_, weight_]', lambda ht, wt: sqrt(ht*wt/3600)),  # Mosteller
        ('DuBoisFormula[height_, weight_]', lambda ht, wt: 0.007184*wt**0.425*ht**0.725),
        ('IdealBodyWeight[height_, sex_]', lambda ht, sex: 50 + 2.3*(ht - 60) if str(sex) == 'M' else 45.5 + 2.3*(ht - 60)),
        ('LeanBodyMass[weight_, height_, sex_]', lambda wt, ht, sex: 0.407*wt + 0.267*ht - 19.2 if str(sex) == 'M' else 0.252*wt + 0.473*ht - 48.3),
        ('BasalMetabolicRate[weight_, height_, age_, sex_]', lambda wt, ht, age, sex: 10*wt + 6.25*ht - 5*age + 5 if str(sex) == 'M' else 10*wt + 6.25*ht - 5*age - 161),
        ('TotalEnergyExpenditure[BMR_, PAL_]', lambda BMR, PAL: BMR*PAL),  # Physical activity level
        ('CardiacOutput[HR_, SV_]', lambda HR, SV: HR*SV/1000),  # L/min
        ('StrokeVolume[EDV_, ESV_]', lambda EDV, ESV: EDV - ESV),
        ('EjectionFraction[SV_, EDV_]', lambda SV, EDV: SV/EDV*100),
        ('MeanArterialPressure[SBP_, DBP_]', lambda SBP, DBP: DBP + (SBP - DBP)/3),
        ('PulsePressure[SBP_, DBP_]', lambda SBP, DBP: SBP - DBP),
        ('SystemicVascularResistance[MAP_, CVP_, CO_]', lambda MAP, CVP, CO: (MAP - CVP)/CO*80),  # dynes·s/cm⁵
        ('PulmonaryVascularResistance[MPAP_, PCWP_, CO_]', lambda MPAP, PCWP, CO: (MPAP - PCWP)/CO*80),
        ('RespiratoryRate[breaths_per_min_]', lambda RR: RR),
        ('TidalVolume[TV_]', lambda TV: TV),  # mL
        ('MinuteVentilation[TV_, RR_]', lambda TV, RR: TV*RR/1000),  # L/min
        ('AlveolarVentilation[TV_, dead_space_, RR_]', lambda TV, DS, RR: (TV - DS)*RR/1000),
        ('OxygenSaturation[SpO2_]', lambda SpO2: SpO2),
        ('PaO2FiO2Ratio[PaO2_, FiO2_]', lambda PaO2, FiO2: PaO2/FiO2),
        ('AlveolarArterialGradient[PAO2_, PaO2_]', lambda PAO2, PaO2: PAO2 - PaO2),
        ('AlveolarGasEquation[FiO2_, PaCO2_]', lambda FiO2, PaCO2: (FiO2*713) - PaCO2/0.8),
        ('RespiratoryQuotient[VCO2_, VO2_]', lambda VCO2, VO2: VCO2/VO2),
        ('OxygenConsumption[VO2_]', lambda VO2: VO2),  # mL/min
        ('CarbonDioxideProduction[VCO2_]', lambda VCO2: VCO2),
        
        # ===== BIOMECHANICS (25 functions) =====
        ('ForceBodyWeight[mass_, g_]', lambda m, g: m*g),
        ('TorqueJoint[F_, r_]', lambda F, r: F*r),
        ('MomentArm[perpendicular_distance_]', lambda d: d),
        ('MechanicalAdvantageBody[F_out_, F_in_]', lambda Fout, Fin: Fout/Fin),
        ('GroundReactionForce[mass_, acceleration_]', lambda m, a: m*(sp.Symbol('g') + a)),
        ('ImpulseCollision[F_, delta_t_]', lambda F, dt: F*dt),
        ('WorkMechanical[F_, d_]', lambda F, d: F*d),
        ('PowerOutput[work_, time_]', lambda W, t: W/t),
        ('StrainTissue[delta_L_, L0_]', lambda dL, L0: dL/L0),
        ('StressTissue[F_, A_]', lambda F, A: F/A),
        ('YoungsModulusTissue[stress_, strain_]', lambda sig, eps: sig/eps),
        ('ViscoelasticResponse[stress_, time_]', lambda sig, t: sig*exp(-t/sp.Symbol('tau'))),
        ('StressRelaxation[sigma0_, t_, tau_]', lambda sig0, t, tau: sig0*exp(-t/tau)),
        ('CreepCompliance[strain_, time_]', lambda eps, t: eps/sp.Symbol('sigma0')*(1 - exp(-t/sp.Symbol('tau')))),
        ('BoneStrength[density_, geometry_]', lambda rho, geom: rho*geom),
        ('FractureLoad[stress_ultimate_, area_]', lambda sigult, A: sigult*A),
        ('GaitCycle[stance_, swing_]', lambda stance, swing: stance + swing),
        ('CadenceWalking[steps_per_min_]', lambda cadence: cadence),
        ('StrideLength[distance_, steps_]', lambda d, steps: d/steps),
        ('DoubleSupport[time_both_feet_]', lambda t: t),
        ('CenterOfMass[masses_, positions_]', lambda m, x: sum(mi*xi for mi, xi in zip(m, x))/sum(m)),
        ('BalanceIndex[sway_]', lambda sway: sway),
        ('PosturalStability[COP_displacement_]', lambda COP: COP),
        ('JumpHeight[flight_time_]', lambda t: sp.Symbol('g')*t**2/8),
        ('PowerClean[work_, time_]', lambda W, t: W/t),
    ]


def register():
    """Register all extended22 rules"""
    from .extended_helper import convert_rules
    return convert_rules(get_rules())
