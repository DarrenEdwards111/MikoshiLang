"""
MikoshiLang Extended Functions - Set 30
Acoustics, audio engineering, psychoacoustics, and vibration analysis
"""
import sympy as sp
from sympy import symbols, sqrt, sin, cos, exp, log, pi

def get_rules():
    """Extended set 30: Acoustics and audio (150 functions)"""
    f, t, A = symbols('f t A')
    
    return [
        # ===== ACOUSTICS FUNDAMENTALS (35 functions) =====
        ('SoundPressureLevel[p_, p_ref_]', lambda p, pref: 20*log(p/pref, 10)),
        ('SoundIntensityLevel[I_, I_ref_]', lambda I, Iref: 10*log(I/Iref, 10)),
        ('SoundPowerLevel[W_, W_ref_]', lambda W, Wref: 10*log(W/Wref, 10)),
        ('AcousticIntensity[p_, rho_, c_]', lambda p, rho, c: p**2/(rho*c)),
        ('AcousticImpedanceWave[rho_, c_]', lambda rho, c: rho*c)),
        ('SpeedOfSoundAir[T_]', lambda T: 331.3*sqrt(1 + T/273.15)),
        ('SpeedOfSoundWater[T_, S_, d_]', lambda T, S, d: 1449 + 4.6*T - 0.055*T**2 + 0.00029*T**3 + (1.34 - 0.01*T)*(S - 35) + 0.016*d),
        ('WavelengthAcoustic[c_, f_]', lambda c, f: c/f),
        ('FrequencyAcoustic[c_, lambda_]', lambda c, lam: c/lam),
        ('ParticleVelocity[p_, rho_, c_]', lambda p, rho, c: p/(rho*c)),
        ('ParticleDisplacement[v_, omega_]', lambda v, omega: v/omega),
        ('AcousticEnergy Density[p_, rho_, c_]', lambda p, rho, c: p**2/(rho*c**2)),
        ('RayleighDistance[D_, lambda_]', lambda D, lam: D**2/(4*lam)),
        ('NearFieldDistance[D_, lambda_]', lambda D, lam: D**2/(4*lam)),
        ('FarFieldApproximation[r_, D_, lambda_]', lambda r, D, lam: r > D**2/(4*lam)),
        ('FrequencyRatio[f1_, f2_]', lambda f1, f2: log(f1/f2, 2)),  # Octaves
        ('MusicalInterval[f1_, f2_]', lambda f1, f2: 1200*log(f1/f2, 2)),  # Cents
        ('HarmonicSeries[f0_, n_]', lambda f0, n: n*f0),
        ('Overtone[f0_, n_]', lambda f0, n: (n + 1)*f0),
        ('FundamentalFrequency[harmonics_]', lambda h: sp.gcd(*h) if len(h) > 1 else h[0]),
        ('Inharmonicity[f_partial_, n_, f0_]', lambda fp, n, f0: (fp - n*f0)/(n*f0)),
        ('BeatFrequency[f1_, f2_]', lambda f1, f2: abs(f1 - f2)),
        ('CombinationTone[f1_, f2_]', lambda f1, f2: abs(f1 - f2)),  # Difference tone
        ('SummationTone[f1_, f2_]', lambda f1, f2: f1 + f2),
        ('DopplerShift[f0_, v_source_, v_observer_, c_]', lambda f0, vs, vo, c: f0*(c + vo)/(c - vs)),
        ('SonicBoom[M_]', lambda M: M > 1),
        ('MachWaveAngle[M_]', lambda M: sp.asin(1/M)),
        ('StandingWave[A_, k_, x_, omega_, t_]', lambda A, k, x, omega, t: 2*A*sin(k*x)*cos(omega*t)),
        ('NodePosition[n_, lambda_]', lambda n, lam: n*lam/2),
        ('AntinodePosition[n_, lambda_]', lambda n, lam: (2*n + 1)*lam/4),
        ('ResonanceFrequency[L_, n_, c_]', lambda L, n, c: n*c/(2*L)),
        ('FundamentalMode[L_, c_]', lambda L, c: c/(2*L)),
        ('OpenPipeResonance[L_, n_, c_]', lambda L, n, c: n*c/(2*L)),
        ('ClosedPipeResonance[L_, n_, c_]', lambda L, n, c: (2*n - 1)*c/(4*L)),
        ('HelmholtzResonator[V_, A_, L_]', lambda V, A, L: sp.Symbol('c')/( 2*pi)*sqrt(A/(V*L))),
        
        # ===== ROOM ACOUSTICS (30 functions) =====
        ('SabineReverbTime[V_, A_]', lambda V, A: 0.161*V/A),
        ('EyringReverbTime[V_, S_, alpha_]', lambda V, S, alpha: -0.161*V/(S*log(1 - alpha))),
        ('MilngtonSetteFormula[V_, S_i_, alpha_i_]', lambda V, Si, alphai: -0.161*V/sum(Sii*log(1 - alphai) for Sii, alphai in zip(Si, alphai))),
        ('AbsorptionCoefficient[alpha_]', lambda alpha: alpha),
        ('SabinAbsorption[S_, alpha_]', lambda S, alpha: S*alpha),
        ('TotalRoomAbsorption[materials_]', lambda m: sum(m)),
        ('RoomConstant[S_, alpha_]', lambda S, alpha: S*alpha/(1 - alpha)),
        ('CriticalDistance[R_, Q_]', lambda R, Q: sqrt(R*Q/(16*pi))),
        ('DirectToreverbRatio[r_, R_]', lambda r, R: 10*log(R/(16*pi*r**2) + 1, 10)),
        ('SoundDecayRate[T60_]', lambda T60: 60/T60),  # dB/s
        ('EarlyDecayTime[EDT_]', lambda EDT: EDT),
        ('Clarity[C80_]', lambda C80: C80),
        ('Definition[D50_]', lambda D50: D50),
        ('SpeechTransmissionIndex[STI_]', lambda STI: STI),
        ('ArticulationIndex[AI_]', lambda AI: AI),
        ('ModulationTransferFunction[m_]', lambda m: m),
        ('RapidSpeechTransmissionIndex[RASTI_]', lambda RASTI: RASTI),
        ('SoundStrengthG[I_, I_ref_]', lambda I, Iref: 10*log(I/Iref, 10)),
        ('LateralEnergyFraction[LF_]', lambda LF: LF),
        ('InterauralCrossCorrelation[IACC_]', lambda IACC: IACC)),
        ('CenterTime[ts_]', lambda ts: ts),
        ('RoomGain[direct_, reverb_]', lambda d, r: 10*log((d + r)/d, 10)),
        ('NoiseReductionCoefficient[alpha_250_, alpha_500_, alpha_1000_, alpha_2000_]', lambda a250, a500, a1000, a2000: (a250 + a500 + a1000 + a2000)/4),
        ('SoundAbsorptionAverage[alphas_]', lambda alphas: sum(alphas)/len(alphas)),
        ('RoomModeFrequency[nx_, ny_, nz_, Lx_, Ly_, Lz_, c_]', lambda nx, ny, nz, Lx, Ly, Lz, c: c/2*sqrt((nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2)),
        ('AxialMode[n_, L_, c_]', lambda n, L, c: n*c/(2*L)),
        ('TangentialMode[nx_, ny_, Lx_, Ly_, c_]', lambda nx, ny, Lx, Ly, c: c/2*sqrt((nx/Lx)**2 + (ny/Ly)**2)),
        ('ObliqueMode[nx_, ny_, nz_, Lx_, Ly_, Lz_, c_]', lambda nx, ny, nz, Lx, Ly, Lz, c: c/2*sqrt((nx/Lx)**2 + (ny/Ly)**2 + (nz/Lz)**2)),
        ('ModalDensity[V_, c_, f_]', lambda V, c, f: 4*pi*V*f**2/c**3),
        ('SchroederFrequency[T60_, V_]', lambda T60, V: 2000*sqrt(T60/V)),
        
        # ===== PSYCHOACOUSTICS (35 functions) =====
        ('LoudnessLevel[SPL_, f_]', lambda SPL, f: SPL),  # Phons, simplified
        ('Loudness[phons_]', lambda phons: 2**((phons - 40)/10)),  # Sones
        ('EqualLoudnessContour[f_]', lambda f: sp.Symbol('SPL')),
        ('FletcherMunsonCurve[f_]', lambda f: sp.Symbol('threshold')),
        ('ThresholdOfHearing[f_]', lambda f: 20*log(f/1000, 10) - 10 if f > 1000 else 0),
        ('ThresholdOfPain[]', lambda: 130),  # dB SPL
        ('DynamicRange[threshold_, ceiling_]', lambda th, ceil: ceil - th),
        ('MaskingLevel[masker_, signal_]', lambda m, s: m - s),
        ('SimultaneousMasking[f_masker_, f_signal_]', lambda fm, fs: abs(fm - fs)),
        ('TemporalMasking[pre_post_]', lambda pp: pp),
        ('CriticalBand[f_]', lambda f: 25 + 75*(1 + 1.4*(f/1000)**2)**0.69),  # Hz, Zwicker
        ('BarkScale[f_]', lambda f: 13*sp.atan(0.00076*f) + 3.5*sp.atan((f/7500)**2)),
        ('ERBScale[f_]', lambda f: 21.4*log(1 + 0.00437*f, 10)),  # Equivalent rectangular bandwidth
        ('CochlearFilterBandwidth[f_]', lambda f: 24.7*(4.37*f/1000 + 1)),
        ('PitchPerception[f_]', lambda f: f),
        ('MissingFundamental[harmonics_]', lambda h: sp.gcd(*h)),
        ('VirtualPitch[partials_]', lambda p: sp.gcd(*p)),
        ('PitchShift[semitones_]', lambda s: 2**(s/12)),
        ('MusicalPitch[f_]', lambda f: 69 + 12*log(f/440, 2)),  # MIDI note number
        ('FrequencyFromMIDI[note_]', lambda n: 440*2**((n - 69)/12)),
        ('JustNoticeDifferenceFrequency[f_]', lambda f: 0.003*f),  # Weber's law
        ('JNDIntensity[I_]', lambda I: 0.1*I),
        ('TimbreDifference[spectrum1_, spectrum2_]', lambda s1, s2: sum(abs(s1i - s2i) for s1i, s2i in zip(s1, s2))),
        ('SpectralCentroid[f_, A_]', lambda f, A: sum(fi*Ai for fi, Ai in zip(f, A))/sum(A)),
        ('SpectralSpread[f_, A_, centroid_]', lambda f, A, c: sqrt(sum(Ai*(fi - c)**2 for fi, Ai in zip(f, A))/sum(A))),
        ('SpectralFlatness[spectrum_]', lambda s: sp.exp(sum(log(si) for si in s)/len(s)) / (sum(s)/len(s))),
        ('SpectralRolloff[spectrum_]', lambda s: sp.Symbol('f_rolloff')),
        ('ZeroCrossingRate[signal_]', lambda sig: sum(1 for i in range(len(sig)-1) if sig[i]*sig[i+1] < 0)/len(sig)),
        ('AttackTime[envelope_]', lambda env: sp.Symbol('t_attack')),
        ('DecayTime[envelope_]', lambda env: sp.Symbol('t_decay')),
        ('SustainLevel[envelope_]', lambda env: sp.Symbol('level')),
        ('ReleaseTime[envelope_]', lambda env: sp.Symbol('t_release')),
        ('PrecedenceEffect[delay_]', lambda delay: delay < 0.04),  # Haas effect
        ('CocktailPartyEffect[SNR_]', lambda SNR: SNR > -6),
        ('BinauralBeatFrequency[f_left_, f_right_]', lambda fl, fr: abs(fl - fr)),
        
        # ===== AUDIO SIGNAL PROCESSING (30 functions) =====
        ('TimeStretch[signal_, factor_]', lambda sig, f: sig),  # Placeholder
        ('PitchShiftSignal[signal_, semitones_]', lambda sig, s: sig),
        ('AutoTune[signal_, scale_]', lambda sig, sc: sig),
        ('VocoderAnalysis[signal_, carriers_]', lambda sig, car: car),
        ('PhaseVocoder[STFT_, time_stretch_, pitch_shift_]', lambda S, ts, ps: S),
        ('GranularSynthesis[grains_]', lambda g: sum(g)),
        ('WavetableSynthesis[table_, index_]', lambda tab, idx: tab[int(idx) % len(tab)]),
        ('FMSynthesis[carrier_, modulator_, index_]', lambda fc, fm, idx: sin(2*pi*fc*sp.Symbol('t') + idx*sin(2*pi*fm*sp.Symbol('t')))),
        ('AMSynthesis[carrier_, modulator_]', lambda fc, fm: (1 + fm)*fc),
        ('SubtractiveSynthesis[oscillator_, filter_]', lambda osc, filt: filt*osc),
        ('AdditiveSynthesis[partials_]', lambda p: sum(p)),
        ('SampleAndHold[signal_, rate_]', lambda sig, rate: sig),
        ('DownsamplingAudio[signal_, factor_]', lambda sig, f: sig[::int(f)]),
        ('UpsamplingAudio[signal_, factor_]', lambda sig, f: sig),
        ('ResamplingAudio[signal_, ratio_]', lambda sig, r: sig),
        ('BandlimitedInterpolation[signal_]', lambda sig: sig),
        ('WindowingFunction[type_, N_]', lambda typ, N: [1]*int(N)),  # Placeholder
        ('HannWindow[n_, N_]', lambda n, N: 0.5*(1 - cos(2*pi*n/N))),
        ('HammingWindow[n_, N_]', lambda n, N: 0.54 - 0.46*cos(2*pi*n/N)),
        ('BlackmanWindow[n_, N_]', lambda n, N: 0.42 - 0.5*cos(2*pi*n/N) + 0.08*cos(4*pi*n/N)),
        ('KaiserWindow[n_, N_, beta_]', lambda n, N, beta: sp.Symbol('I0')(beta*sqrt(1 - (2*n/N - 1)**2))/sp.Symbol('I0')(beta)),
        ('SpectralLeakage[window_]', lambda w: w),
        ('ScallopingLoss[bin_offset_]', lambda offset: sin(pi*offset)/(pi*offset) if offset != 0 else 1),
        ('OverlapAddMethod[windows_]', lambda w: sum(w)),
        ('ShortTimeFourierTransform[signal_, window_, hop_]', lambda sig, win, hop: sp.Symbol('STFT')),
        ('MelFilterbank[f_, N_mels_]', lambda f, Nmels: [0]*int(Nmels)),
        ('MFCCCoefficients[signal_]', lambda sig: [0]*13),
        ('ChromaFeatures[signal_]', lambda sig: [0]*12),
        ('TonalCentroid[chroma_]', lambda chr: (sp.Symbol('x'), sp.Symbol('y'))),
        ('SpectralContrast[spectrum_]', lambda s: sp.Symbol('contrast')),
        
        # ===== NOISE & VIBRATION (20 functions) =====
        ('WhiteNoise[duration_]', lambda d: "noise"),
        ('PinkNoise[duration_]', lambda d: "1/f noise"),
        ('BrownNoise[duration_]', lambda d: "1/f^2 noise"),
        ('BandlimitedNoise[f_low_, f_high_]', lambda flow, fhigh: "noise"),
        ('NoiseFloor[signal_]', lambda sig: min(sig)),
        ('SignalToNoiseRatio[signal_power_, noise_power_]', lambda Ps, Pn: 10*log(Ps/Pn, 10)),
        ('DynamicRangeAudio[peak_, noise_floor_]', lambda peak, nf: 20*log(peak/nf, 10)),
        ('CrestFactor[peak_, RMS_]', lambda peak, rms: 20*log(peak/rms, 10)),
        ('TotalHarmonicDistortion[harmonics_, fundamental_]', lambda h, f: sqrt(sum(hi**2 for hi in h))/f*100),
        ('IntermodulationDistortion[f1_, f2_]', lambda f1, f2: abs(f1 - f2)),
        ('VibrationLevel[a_, a_ref_]', lambda a, aref: 20*log(a/aref, 10)),
        ('VibrationDose[a_, t_]', lambda a, t: a*sqrt(t)),
        ('RMSVibration[signal_]', lambda sig: sqrt(sum(si**2 for si in sig)/len(sig))),
        ('PeakVibration[signal_]', lambda sig: max(abs(si) for si in sig)),
        ('VibrationalEnergy[m_, omega_, A_]', lambda m, omega, A: sp.Rational(1,2)*m*omega**2*A**2),
        ('DampingRatioVibration[zeta_]', lambda zeta: zeta),
        ('QFactorVibration[omega_n_, zeta_]', lambda omegan, zeta: 1/(2*zeta)),
        ('TransmissibilityVibration[omega_, omega_n_, zeta_]', lambda omega, omegan, zeta: sqrt((1 + (2*zeta*omega/omegan)**2)/((1 - (omega/omegan)**2)**2 + (2*zeta*omega/omegan)**2))),
        ('VibrationIsolationEfficiency[T_]', lambda T: (1 - T)*100),
        ('ShockResponseSpectrum[acceleration_, time_]', lambda a, t: sp.Symbol('SRS')),
    ]

def register():
    """Register all extended30 rules"""
    return get_rules()
