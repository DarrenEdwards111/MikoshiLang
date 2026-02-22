"""
MikoshiLang Extended Functions - Set 37
Renewable energy, power systems, and energy storage
"""
import sympy as sp
from sympy import symbols, sqrt, exp, log, pi, sin, cos, atan2

def get_rules():
    """Extended set 37: Renewable energy and power systems (150 functions)"""
    t, T, P = symbols('t T P')
    
    return [
        # ===== SOLAR ENERGY (40 functions) =====
        ('SolarIrradiance[I_]', lambda I: I),  # W/m²
        ('DirectNormalIrradiance[DNI_]', lambda DNI: DNI),
        ('DiffuseHorizontalIrradiance[DHI_]', lambda DHI: DHI),
        ('GlobalHorizontalIrradiance[GHI_]', lambda GHI: GHI),
        ('SolarConstant[]', lambda: 1367),  # W/m²
        ('AirMass[zenith_angle_]', lambda theta: 1/cos(theta)),
        ('SolarZenithAngle[latitude_, declination_, hour_angle_]', lambda lat, decl, ha: sp.acos(sin(lat)*sin(decl) + cos(lat)*cos(decl)*cos(ha))),
        ('SolarAzimuthAngle[zenith_, declination_, latitude_]', lambda z, decl, lat: sp.Symbol('azimuth')),
        ('SolarDeclination[day_of_year_]', lambda d: 23.45*sin(2*pi*(284 + d)/365)),
        ('HourAngle[solar_time_]', lambda t: 15*(t - 12)),
        ('EquationOfTime[day_]', lambda d: sp.Symbol('EoT')),
        ('SolarTime[standard_time_, longitude_, EoT_]', lambda st, lon, eot: st + lon/15 + eot/60),
        ('OptimalTiltAngle[latitude_]', lambda lat: lat),
        ('SolarPanelOutput[irradiance_, area_, efficiency_]', lambda I, A, eta: I*A*eta),
        ('PVCellEfficiency[Voc_, Isc_, FF_, irradiance_, area_]', lambda Voc, Isc, FF, I, A: Voc*Isc*FF/(I*A)),
        ('FillFactor[Vmp_, Imp_, Voc_, Isc_]', lambda Vmp, Imp, Voc, Isc: Vmp*Imp/(Voc*Isc)),
        ('OpenCircuitVoltage[Voc_]', lambda Voc: Voc),
        ('ShortCircuitCurrent[Isc_]', lambda Isc: Isc),
        ('MaximumPowerPoint[Vmp_, Imp_]', lambda Vmp, Imp: Vmp*Imp),
        ('TemperatureCoefficient[dP_dT_]', lambda dPdT: dPdT),
        ('NOCTtemperature[ambient_, irradiance_]', lambda Ta, I: Ta + I/800*20),
        ('CellTemperature[ambient_, irradiance_, NOCT_]', lambda Ta, I, NOCT: Ta + (NOCT - 20)/800*I),
        ('PowerTemperatureDerating[P_STC_, temp_coeff_, T_cell_, T_STC_]', lambda P, tc, Tc, Tstc: P*(1 + tc*(Tc - Tstc))),
        ('PerformanceRatio[actual_yield_, theoretical_yield_]', lambda actual, theor: actual/theor*100),
        ('CapacityFactor[actual_output_, rated_capacity_, time_]', lambda actual, rated, t: actual/(rated*t)*100),
        ('SolarArrayPower[num_panels_, panel_power_]', lambda n, P: n*P),
        ('StringConfiguration[panels_series_, panels_parallel_]', lambda s, p: (s, p)),
        ('InverterEfficiency[AC_output_, DC_input_]', lambda AC, DC: AC/DC*100),
        ('InverterSizing[DC_array_, temperature_factor_]', lambda DC, tf: DC*tf),
        ('MPPTEfficiency[actual_, theoretical_]', lambda actual, theor: actual/theor*100),
        ('DCSideLosses[soiling_, shading_, mismatch_]', lambda soil, shade, mm: soil + shade + mm),
        ('ACSideLosses[inverter_, transformer_, wiring_]', lambda inv, trans, wire: inv + trans + wire),
        ('ShadingLoss[shaded_area_, total_area_]', lambda shaded, total: shaded/total*100),
        ('SoilingLoss[clean_, soiled_]', lambda clean, soiled: (clean - soiled)/clean*100),
        ('BifacialGain[rear_irradiance_, front_irradiance_]', lambda rear, front: rear/front*100),
        ('AlbedoReflection[albedo_]', lambda a: a),
        ('SingleAxisTracking[fixed_output_, tracking_output_]', lambda fixed, track: (track - fixed)/fixed*100),
        ('DualAxisTracking[single_, dual_]', lambda single, dual: (dual - single)/single*100),
        ('ConcentratedSolarPower[DNI_, concentration_ratio_, efficiency_]', lambda DNI, CR, eta: DNI*CR*eta),
        ('SolarThermalEfficiency[Q_out_, Q_in_]', lambda Qout, Qin: Qout/Qin*100),
        
        # ===== WIND ENERGY (35 functions) =====
        ('WindPower[rho_, A_, v_]', lambda rho, A, v: 0.5*rho*A*v**3),
        ('BetzLimit[]', lambda: 16/27),  # Maximum theoretical efficiency
        ('PowerCoefficient[actual_power_, wind_power_]', lambda Pactual, Pwind: Pactual/Pwind),
        ('TipSpeedRatio[blade_tip_speed_, wind_speed_]', lambda omega_R, v: omega_R/v),
        ('RotorSweptArea[radius_]', lambda R: pi*R**2),
        ('AirDensity[P_, T_, R_]', lambda P, T, R: P/(R*T)),
        ('AirDensityAltitude[altitude_]', lambda h: 1.225*exp(-h/10400)),
        ('PowerLaw[v_, h_, href_, alpha_]', lambda v, h, href, alpha: v*(h/href)**alpha),
        ('LogLaw[v_, z_, z0_, u_star_]', lambda v, z, z0, ustar: ustar/0.4*log(z/z0)),
        ('RoughnessLength[terrain_]', lambda t: sp.Symbol('z0')),
        ('ShearExponent[alpha_]', lambda alpha: alpha),
        ('WindSpeedHeight[v_ref_, h_, h_ref_, alpha_]', lambda vref, h, href, alpha: vref*(h/href)**alpha),
        ('WeibullDistribution[v_, k_, c_]', lambda v, k, c: (k/c)*(v/c)**(k - 1)*exp(-(v/c)**k)),
        ('WeibullShapeParameter[k_]', lambda k: k),
        ('WeibullScaleParameter[c_]', lambda c: c),
        ('AverageWindSpeed[weibull_c_, weibull_k_]', lambda c, k: c*sp.gamma(1 + 1/k)),
        ('WindPowerDensity[rho_, v_]', lambda rho, v: 0.5*rho*v**3),
        ('AnnualEnergyProduction[rated_power_, CF_, hours_]', lambda P, CF, hrs: P*CF*hrs),
        ('WindTurbineEfficiency[actual_, theoretical_]', lambda actual, theor: actual/theor*100),
        ('CutInSpeed[v_in_]', lambda vin: vin),
        ('RatedSpeed[v_rated_]', lambda vr: vr),
        ('CutOutSpeed[v_out_]', lambda vout: vout),
        ('TurbineAvailability[uptime_, total_time_]', lambda up, total: up/total*100),
        ('WakeLoss[turbine_spacing_, diameter_]', lambda s, D: sp.Symbol('wake_loss')),
        ('ArrayEfficiency[array_output_, sum_individual_]', lambda array, indiv: array/indiv*100),
        ('ThrustCoefficient[thrust_, dynamic_pressure_area_]', lambda T, qA: T/qA),
        ('TorqueCoefficient[torque_, density_area_radius_velocity_]', lambda Q, rho_A_R_v: Q/(0.5*rho_A_R_v)),
        ('BladeLoading[thrust_, swept_area_]', lambda T, A: T/A),
        ('SolidityRatio[blade_area_, swept_area_]', lambda Ab, A: Ab/A),
        ('GeneratorEfficiency[electrical_output_, mechanical_input_]', lambda Pelec, Pmech: Pelec/Pmech*100),
        ('GearboxEfficiency[output_torque_, input_torque_]', lambda Tout, Tin: Tout/Tin*100),
        ('YawError[actual_direction_, wind_direction_]', lambda actual, wind: abs(actual - wind)),
        ('PitchAngle[theta_]', lambda theta: theta),
        ('CapacityCredit[firm_capacity_, installed_capacity_]', lambda firm, inst: firm/inst*100),
        ('LevelizedCostEnergy[capex_, opex_, AEP_, discount_rate_, lifetime_]', lambda capex, opex, aep, r, n: (capex + sum(opex/(1 + r)**i for i in range(1, int(n) + 1)))/sum(aep/(1 + r)**i for i in range(1, int(n) + 1))),
        
        # ===== ENERGY STORAGE (30 functions) =====
        ('BatteryCapacity[voltage_, Ah_]', lambda V, Ah: V*Ah),  # Wh
        ('EnergyDensity[energy_, volume_]', lambda E, V: E/V),  # Wh/L
        ('SpecificEnergy[energy_, mass_]', lambda E, m: E/m),  # Wh/kg
        ('PowerDensity[power_, volume_]', lambda P, V: P/V),  # W/L
        ('SpecificPower[power_, mass_]', lambda P, m: P/m),  # W/kg
        ('StateOfCharge[current_capacity_, total_capacity_]', lambda Qcurr, Qtot: Qcurr/Qtot*100),
        ('DepthOfDischarge[discharged_, total_]', lambda dis, tot: dis/tot*100),
        ('ChargeEfficiency[energy_out_, energy_in_]', lambda Eout, Ein: Eout/Ein*100),
        ('RoundTripEfficiency[discharge_energy_, charge_energy_]', lambda Edis, Echg: Edis/Echg*100),
        ('CRate[current_, capacity_]', lambda I, C: I/C),
        ('ChargeTime[capacity_, current_]', lambda C, I: C/I),
        ('CycleLife[cycles_]', lambda cycles: cycles),
        ('CalendarLife[years_]', lambda yrs: yrs),
        ('SelfDischargeRate[capacity_loss_, time_]', lambda loss, t: loss/t*100),
        ('BatteryVoltage[cell_voltage_, cells_series_]', lambda Vcell, n: Vcell*n),
        ('BatteryCurrentParallel[cell_current_, cells_parallel_]', lambda Icell, p: Icell*p),
        ('InternalResistance[V_oc_, V_load_, I_]', lambda Voc, Vload, I: (Voc - Vload)/I),
        ('PowerLoss[I_, R_]', lambda I, R: I**2*R),
        ('TheoreticalCapacity[M_, n_, F_]', lambda M, n, F: M/(n*F)*26.8),  # Ah/g
        ('PracticalCapacity[theoretical_, utilization_]', lambda theor, util: theor*util),
        ('EnergyStorageSystem[capacity_, power_, duration_]', lambda C, P, d: (C, P, d)),
        ('PumpedHydroEfficiency[generated_, pumped_]', lambda gen, pump: gen/pump*100),
        ('CompressedAirEnergyStorage[pressure_, volume_]', lambda p, V: p*V),
        ('FlywheelEnergy[I_, omega_]', lambda I, omega: 0.5*I*omega**2),
        ('SupercapacitorEnergy[C_, V_]', lambda C, V: 0.5*C*V**2),
        ('HydrogenStorageDensity[mass_H2_, volume_]', lambda m, V: m/V),
        ('FuelCellEfficiency[electrical_energy_, fuel_energy_]', lambda Eelec, Efuel: Eelec/Efuel*100),
        ('ElectrolyzerEfficiency[H2_energy_, electrical_input_]', lambda EH2, Eelec: EH2/Eelec*100),
        ('ThermalStorageCapacity[mass_, Cp_, delta_T_]', lambda m, Cp, dT: m*Cp*dT),
        ('LatentHeatStorage[mass_, Lf_]', lambda m, Lf: m*Lf),
        
        # ===== POWER SYSTEMS (25 functions) =====
        ('ThreePhaseActivePower[V_, I_, pf_, sqrt3_]', lambda V, I, pf, s3: s3*V*I*pf),
        ('ThreePhaseReactivePower[V_, I_, sin_phi_]', lambda V, I, sinphi: sqrt(3)*V*I*sinphi),
        ('ApparentPower[active_, reactive_]', lambda P, Q: sqrt(P**2 + Q**2)),
        ('PowerFactor[active_, apparent_]', lambda P, S: P/S),
        ('ReactivePowerCompensation[Q_load_, Q_capacitor_]', lambda Qload, Qcap: Qload - Qcap),
        ('CapacitorSizing[Q_, V_]', lambda Q, V: Q/V**2),  # Capacitance
        ('LineResistance[resistivity_, length_, area_]', lambda rho, L, A: rho*L/A),
        ('LineReactance[omega_, L_inductance_]', lambda omega, L: omega*L),
        ('LineImpedance[R_, X_]', lambda R, X: sqrt(R**2 + X**2)),
        ('VoltageDrop[I_, Z_]', lambda I, Z: I*Z),
        ('LineLoadability[V_, X_, delta_]', lambda V, X, delta: V**2/(X)*sin(delta)),
        ('TransmissionLineLoss[I_, R_]', lambda I, R: I**2*R),
        ('PowerTransferCapacity[V_send_, V_recv_, X_, delta_]', lambda Vs, Vr, X, delta: Vs*Vr/X*sin(delta)),
        ('TransformerEfficiency[output_, input_]', lambda Pout, Pin: Pout/Pin*100),
        ('TransformerLosses[core_loss_, copper_loss_]', lambda Pcore, Pcu: Pcore + Pcu),
        ('VoltageRegulation[V_nl_, V_fl_]', lambda Vnl, Vfl: (Vnl - Vfl)/Vfl*100),
        ('PerUnitImpedance[Z_ohm_, Z_base_]', lambda Z, Zb: Z/Zb),
        ('BaseImpedance[V_base_, MVA_base_]', lambda Vb, Sb: Vb**2/Sb),
        ('ShortCircuitCurrent[V_, Z_]', lambda V, Z: V/Z),
        ('FaultLevel[V_, Z_]', lambda V, Z: V**2/Z),
        ('SymmetricalComponents[Va_, Vb_, Vc_]', lambda Va, Vb, Vc: (Va + Vb + Vc)/3),
        ('LoadFlowAnalysis[P_, Q_, V_, delta_]', lambda P, Q, V, delta: (P, Q, V, delta)),
        ('EconomicDispatch[fuel_costs_, power_outputs_]', lambda costs, outputs: sum(ci*Pi for ci, Pi in zip(costs, outputs))),
        ('FrequencyRegulation[delta_f_]', lambda df: df),
        ('VoltageStability[dQ_dV_]', lambda dQdV: dQdV),
        
        # ===== GRID INTEGRATION (20 functions) =====
        ('GridPenetration[renewable_generation_, total_generation_]', lambda Eren, Etot: Eren/Etot*100),
        ('CurtailmentRate[curtailed_, available_]', lambda curt, avail: curt/avail*100),
        ('RampRate[power_change_, time_]', lambda dP, dt: dP/dt),
        ('Frequency Nadir[min_frequency_]', lambda fmin: fmin),
        ('RateOfChangeFrequency[df_dt_]', lambda dfdt: dfdt),
        ('InertiaConstant[H_]', lambda H: H),
        ('SystemInertia[sum_H_MVA_]', lambda H_MVA: sum(H_MVA)),
        ('SynchronousInertia[mechanical_]', lambda mech: mech),
        ('VirtualInertia[emulated_]', lambda emul: emul),
        ('GridFormingInverter[voltage_source_]', lambda vs: vs),
        ('GridFollowingInverter[current_source_]', lambda cs: cs),
        ('FaultRideThroughCapability[duration_]', lambda dur: dur),
        ('VoltageRideThrough[V_min_, duration_]', lambda Vmin, dur: (Vmin, dur)),
        ('FrequencyRideThrough[f_min_, f_max_]', lambda fmin, fmax: (fmin, fmax)),
        ('ActivePowerControl[P_setpoint_]', lambda Pset: Pset),
        ('ReactivePowerControl[Q_setpoint_]', lambda Qset: Qset),
        ('VoltVARControl[V_target_, Q_]', lambda Vtarg, Q: (Vtarg, Q)),
        ('DroopControl[delta_f_, delta_P_]', lambda df, dP: df/dP),
        ('AGCSignal[area_control_error_]', lambda ACE: ACE),
        ('InterconnectionCapacity[tie_line_capacity_]', lambda cap: cap),
    ]

def register():
    """Register all extended37 rules"""
    from .extended_helper import convert_rules
    return convert_rules(get_rules())
