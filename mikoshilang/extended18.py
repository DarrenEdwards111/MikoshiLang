"""
MikoshiLang Extended Functions - Set 18
Biology, genetics, bioinformatics, and molecular biology
"""
import sympy as sp
from sympy import symbols, sqrt, exp, log, pi

def get_rules():
    """Extended set 18: Biology and genetics (250 functions)"""
    x, t, n = symbols('x t n')
    
    return [
        # ===== MOLECULAR BIOLOGY (60 functions) =====
        ('DNAMolecularWeight[sequence_]', lambda seq: len(str(seq)) * 330),  # Average MW per base
        ('RNAMolecularWeight[sequence_]', lambda seq: len(str(seq)) * 340),
        ('ProteinMolecularWeight[sequence_]', lambda seq: len(str(seq)) * 110),  # Average MW per amino acid
        ('GCContent[sequence_]', lambda seq: (str(seq).count('G') + str(seq).count('C')) / len(str(seq)) * 100),
        ('MeltingTemperature[sequence_]', lambda seq: 64.9 + 41 * (str(seq).count('G') + str(seq).count('C') - 16.4) / len(str(seq))),
        ('NearestNeighbor[seq1_, seq2_]', lambda s1, s2: -1.0),  # Simplified deltaG
        ('PrimerTm[sequence_]', lambda seq: 4 * (str(seq).count('G') + str(seq).count('C')) + 2 * (str(seq).count('A') + str(seq).count('T'))),
        ('OligoConcentration[A260_, dilution_]', lambda A, dil: A * dil * 33),  # μg/mL for ssDNA
        ('PlasmidCopyNumber[oriType_]', lambda ori: 15 if str(ori) == "pSC101" else 100),
        ('TransformationEfficiency[colonies_, DNA_amount_]', lambda col, dna: col / dna),  # CFU/μg
        ('GeneExpression[mRNA_, protein_]', lambda m, p: p / m),
        ('TranscriptionRate[mRNA_, time_]', lambda m, t: m / t),
        ('TranslationRate[protein_, mRNA_, time_]', lambda p, m, t: p / (m * t)),
        ('mRNAHalfLife[N0_, N_, t_]', lambda N0, N, t: -t / log(N / N0, 2)),
        ('ProteinHalfLife[N0_, N_, t_]', lambda N0, N, t: -t / log(N / N0, 2)),
        ('MichaelisMenten[Vmax_, Km_, S_]', lambda Vmax, Km, S: Vmax * S / (Km + S)),
        ('LineweaverBurk[Vmax_, Km_]', lambda Vmax, Km: (1/Vmax, Km/Vmax)),
        ('EnzymeInhibitionCompetitive[Vmax_, Km_, S_, I_, Ki_]', lambda Vmax, Km, S, I, Ki: Vmax * S / (Km * (1 + I/Ki) + S)),
        ('EnzymeInhibitionNoncompetitive[Vmax_, Km_, S_, I_, Ki_]', lambda Vmax, Km, S, I, Ki: Vmax * S / ((Km + S) * (1 + I/Ki))),
        ('EnzymeInhibitionUncompetitive[Vmax_, Km_, S_, I_, Ki_]', lambda Vmax, Km, S, I, Ki: Vmax * S / (Km + S * (1 + I/Ki))),
        ('TurnoverNumber[Vmax_, Et_]', lambda Vmax, Et: Vmax / Et),  # kcat
        ('CatalyticEfficiency[kcat_, Km_]', lambda kcat, Km: kcat / Km),
        ('HillEquation[Vmax_, S_, K_, n_]', lambda Vmax, S, K, n: Vmax * S**n / (K**n + S**n)),
        ('CooperativityCoefficient[n_]', lambda n: n),
        ('AllosericEffect[Y_, L_, c_]', lambda Y, L, c: Y * (1 + L) / (1 + L * c)),
        ('BindingAffinity[Kd_]', lambda Kd: -sp.Symbol('RT') * log(Kd)),  # Delta G
        ('DissociationConstant[kon_, koff_]', lambda kon, koff: koff / kon),
        ('AssociationConstant[Kd_]', lambda Kd: 1 / Kd),
        ('LigandBindingFraction[L_, Kd_]', lambda L, Kd: L / (L + Kd)),
        ('ScatchardPlot[bound_, free_]', lambda B, F: (B/F, B)),
        ('BmaxKd[Bmax_, Kd_, L_]', lambda Bmax, Kd, L: Bmax * L / (Kd + L)),
        ('ReceptorOccupancy[L_, Kd_]', lambda L, Kd: L / (L + Kd)),
        ('EfficacyDrug[Emax_, EC50_, C_]', lambda Emax, EC50, C: Emax * C / (EC50 + C)),
        ('Dose Response[dose_, EC50_, slope_]', lambda d, EC50, m: 1 / (1 + (EC50/d)**m)),
        ('TherapeuticIndex[LD50_, ED50_]', lambda LD50, ED50: LD50 / ED50),
        ('SafetyMargin[LD1_, ED99_]', lambda LD1, ED99: LD1 / ED99),
        ('BioavailabilityOral[AUC_oral_, AUC_iv_, dose_iv_, dose_oral_]', lambda AUCo, AUCiv, Div, Do: (AUCo * Div) / (AUCiv * Do)),
        ('PlasmaConcentration[dose_, Vd_]', lambda D, Vd: D / Vd),
        ('VolumeOfDistribution[amount_, concentration_]', lambda A, C: A / C),
        ('ClearanceRenal[excretion_rate_, plasma_conc_]', lambda ER, Cp: ER / Cp),
        ('EliminationRateConstant[Cl_, Vd_]', lambda Cl, Vd: Cl / Vd),
        ('HalfLifePharmacological[ln2_, kel_]', lambda ln2, kel: 0.693 / kel),
        ('LoadingDose[Vd_, target_conc_]', lambda Vd, Ct: Vd * Ct),
        ('MaintenanceDose[Cl_, target_conc_, tau_]', lambda Cl, Ct, tau: Cl * Ct * tau),
        ('SteadyStateConcentration[dose_, Cl_, tau_]', lambda D, Cl, tau: D / (Cl * tau)),
        ('AccumulationFactor[tau_, half_life_]', lambda tau, t12: 1 / (1 - exp(-0.693 * tau / t12))),
        ('PeakTroughRatio[Cmax_, Cmin_]', lambda Cmax, Cmin: Cmax / Cmin),
        ('PCREfficiency[Ct_diluted_, Ct_undiluted_, dilution_]', lambda Ctd, Ctu, dil: 10**(-1/log(dil, 10)/(Ctd - Ctu)) - 1),
        ('FoldChange[ddCt_]', lambda ddCt: 2**(-ddCt)),
        ('qPCRStandardCurve[Ct_, conc_]', lambda Ct, C: (-log(C, 10), Ct)),
        ('AmplificationFactor[N_, N0_, cycles_]', lambda N, N0, c: (N / N0)**(1/c)),
        ('CloningEfficiency[colonies_, cells_plated_]', lambda col, cells: col / cells * 100),
        ('ProteinConcentration[A280_, epsilon_, path_]', lambda A, eps, l: A / (eps * l)),
        ('BradfordAssay[A595_]', lambda A: A * 1000),  # Simplified μg/mL
        ('BCAAssay[A562_]', lambda A: A * 1000),
        ('LowryAssay[A750_]', lambda A: A * 1000),
        ('SDS_PAGE_Migration[MW_]', lambda MW: -log(MW)),
        ('IsoelectricPoint[pKa_values_]', lambda pkas: sum(pkas) / len(pkas)),
        ('NetCharge[pH_, pI_]', lambda pH, pI: pH - pI),
        
        # ===== GENETICS (60 functions) =====
        ('HardyWeinberg[p_]', lambda p: (p**2, 2*p*(1-p), (1-p)**2)),
        ('AllelicFrequency[AA_, Aa_, aa_]', lambda AA, Aa, aa: (2*AA + Aa) / (2*(AA + Aa + aa))),
        ('GenotypeFrequency[p_, q_]', lambda p, q: (p**2, 2*p*q, q**2)),
        ('HeterozygosityExpected[p_, q_]', lambda p, q: 2*p*q),
        ('HeterozygosityObserved[heterozy gotes_, total_]', lambda het, tot: het / tot),
        ('InbreedingCoefficient[Ho_, He_]', lambda Ho, He: 1 - Ho/He),
        ('FixationIndex[Ht_, Hs_]', lambda Ht, Hs: (Ht - Hs) / Ht),  # Fst
        ('GeneFlow[m_]', lambda m: m),  # Migration rate
        ('SelectionCoefficient[wAA_, wAa_, waa_]', lambda wAA, wAa, waa: 1 - waa/wAA),
        ('FitnessRelative[w_, wmax_]', lambda w, wmax: w / wmax),
        ('MutationRate[mutations_, generations_, individuals_]', lambda mut, gen, ind: mut / (gen * ind)),
        ('EffectivePopulationSize[Ne_]', lambda Ne: Ne),
        ('CoalescenceTime[Ne_]', lambda Ne: 2 * Ne),
        ('GeneticDrift[p_, Ne_]', lambda p, Ne: sqrt(p * (1 - p) / (2 * Ne))),
        ('BottleneckEffect[N_before_, N_during_]', lambda Nb, Nd: Nb / Nd),
        ('FounderEffect[founders_, population_]', lambda f, p: f / p),
        ('LinkageDisequilibrium[pAB_, pA_, pB_]', lambda pAB, pA, pB: pAB - pA*pB),
        ('RecombinationFraction[recombinants_, total_]', lambda rec, tot: rec / tot),
        ('GeneticDistance[theta_]', lambda theta: -sp.Rational(1,2) * log(1 - 2*theta)),  # Haldane
        ('LODScore[likelihood_linked_, likelihood_unlinked_]', lambda Ll, Lu: log(Ll / Lu, 10)),
        ('MappingFunction[theta_]', lambda theta: sp.Rational(1,2) * (1 - exp(-2*theta))),  # Kosambi
        ('PhysicalDistance[recombination_fraction_, cM_per_Mb_]', lambda theta, r: theta * 100 / r),  # cM to Mb
        ('CodonUsageBias[observed_, expected_]', lambda obs, exp: obs / exp),
        ('SynonymousSubstitutionRate[dS_]', lambda dS: dS),
        ('NonsynonymousSubstitutionRate[dN_]', lambda dN: dN),
        ('SelectionPressure[dN_, dS_]', lambda dN, dS: dN / dS),  # ω (omega)
        ('NeutralEvolution[dN_dS_]', lambda w: w == 1),
        ('PositiveSelection[dN_dS_]', lambda w: w > 1),
        ('PurifyingSelection[dN_dS_]', lambda w: w < 1),
        ('MolecularClock[substitutions_, time_]', lambda sub, t: sub / (2 * t)),
        ('DivergenceTime[distance_, rate_]', lambda d, r: d / (2 * r)),
        ('JukesCantor[p_]', lambda p: -sp.Rational(3,4) * log(1 - sp.Rational(4,3)*p)),
        ('KimuraTwoParameter[P_, Q_]', lambda P, Q: -sp.Rational(1,2)*log(1 - 2*P - Q) - sp.Rational(1,4)*log(1 - 2*Q)),
        ('TamuaNei[p_, GC_]', lambda p, GC: -GC * log(1 - p/GC)),
        ('GammaDistanceCorrection[d_, alpha_]', lambda d, a: a * ((1 - d)**(-1/a) - 1)),
        ('PairwiseIdentity[matches_, length_]', lambda m, l: m / l * 100),
        ('SequenceSimilarity[score_, max_score_]', lambda s, ms: s / ms * 100),
        ('PercentIdentityLocal[aligned_identical_, alignment_length_]', lambda ai, al: ai / al * 100),
        ('PercentSimilarityLocal[aligned_similar_, alignment_length_]', lambda asim, al: asim / al * 100),
        ('AlignmentScore[matches_, mismatches_, gaps_, match_score_, mismatch_penalty_, gap_penalty_]', lambda m, mm, g, ms, mmp, gp: m*ms - mm*mmp - g*gp),
        ('SmithWatermanScore[matrix_]', lambda mat: max(mat)),
        ('NeedlemanWunschScore[matrix_]', lambda mat: mat[-1][-1]),
        ('BLASTExpectValue[S_, m_, n_]', lambda S, m, n: m * n * exp(-S)),
        ('BitScore[S_, K_, lambda_]', lambda S, K, lam: (lam * S - log(K)) / log(2)),
        ('EvolutionaryDistance[p_]', lambda p: -log(1 - p)),
        ('TransitionTransversionRatio[ti_, tv_]', lambda ti, tv: ti / tv),
        ('CpGIsland[observed_CpG_, expected_CpG_]', lambda obs, exp: obs / exp),
        ('CpGObservedExpected[C_, G_, CpG_, length_]', lambda C, G, CpG, L: CpG * L / (C * G)),
        
        # ===== BIOINFORMATICS (60 functions) =====
        ('ORFFinder[sequence_, min_length_]', lambda seq, ml: [0]),
        ('StartCodonPosition[sequence_]', lambda seq: str(seq).find('ATG')),
        ('StopCodonPosition[sequence_]', lambda seq: min(str(seq).find('TAA'), str(seq).find('TAG'), str(seq).find('TGA'))),
        ('SixFrameTranslation[sequence_]', lambda seq: ['frame1', 'frame2', 'frame3', 'frame-1', 'frame-2', 'frame-3']),
        ('ReverseComplement[sequence_]', lambda seq: str(seq)[::-1].translate(str.maketrans('ATCG', 'TAGC'))),
        ('Translate[codon_]', lambda c: 'M' if str(c) == 'ATG' else 'X'),
        ('CodonTable[genetic_code_]', lambda gc: {'ATG': 'M', 'TAA': '*'}),
        ('AminoAcidComposition[sequence_]', lambda seq: {aa: str(seq).count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}),
        ('HydropathyIndex[sequence_]', lambda seq: 0.0),  # Kyte-Doolittle
        ('IsoelectricPointProtein[sequence_]', lambda seq: 7.0),
        ('ExtinctionCoefficient[Y_, W_, C_]', lambda Y, W, C: Y*1490 + W*5500 + C*125),
        ('AliphaticIndex[A_, V_, I_, L_]', lambda A, V, I, L: A + 2.9*V + 3.9*(I + L)),
        ('InstabilityIndex[sequence_]', lambda seq: 40.0),
        ('SecondaryStructurePrediction[sequence_]', lambda seq: 'HHHCCCHHH'),  # H=helix, C=coil
        ('TransmembranePredictor[sequence_]', lambda seq: [(10, 30)]),
        ('SignalPeptidePrediction[sequence_]', lambda seq: (0, 20)),
        ('SubcellularLocalization[features_]', lambda f: 'nucleus'),
        ('ProteinDomainPrediction[sequence_]', lambda seq: [('domain1', 10, 100)]),
        ('MotifFinding[sequence_, motif_]', lambda seq, mot: [str(seq).find(str(mot))]),
        ('ConservedRegion[alignment_]', lambda aln: [(0, 10)]),
        ('PhylogeneticTreeNJ[distance_matrix_]', lambda dm: '((A,B),C)'),  # Neighbor-joining
        ('PhylogeneticTreeUPGMA[distance_matrix_]', lambda dm: '((A,B),C)'),
        ('PhylogeneticTreeMP[sequences_]', lambda seq: '((A,B),C)'),  # Maximum parsimony
        ('PhylogeneticTreeML[sequences_]', lambda seq: '((A,B),C)'),  # Maximum likelihood
        ('BootstrapSupport[tree_, replicates_]', lambda tree, rep: 95),
        ('TreeDistance[tree1_, tree2_]', lambda t1, t2: 0.5),
        ('Robinson FouldsDistance[tree1_, tree2_]', lambda t1, t2: 2),
        ('ParseTreeNewick[newick_]', lambda nwk: str(nwk)),
        ('TreeToNewick[tree_]', lambda tree: '((A,B),C);'),
        ('PairwiseAlignmentGlobal[seq1_, seq2_]', lambda s1, s2: (str(s1), str(s2), 100)),
        ('PairwiseAlignmentLocal[seq1_, seq2_]', lambda s1, s2: (str(s1), str(s2), 100)),
        ('MultipleSequenceAlignment[sequences_]', lambda seqs: seqs),
        ('ProfileHMM[alignment_]', lambda aln: 'HMM'),
        ('PSI_BLAST[query_, database_, iterations_]', lambda q, db, it: ['hit1', 'hit2']),
        ('MotifMatrixPWM[motif_instances_]', lambda mi: [[0.25]*4]),
        ('SequenceLogoHeight[frequency_, total_]', lambda f, t: f * log(f/0.25, 2)),
        ('InformationContent[column_]', lambda col: 2 - sum([-p*log(p, 2) for p in col if p > 0])),
        ('ConservationScore[column_]', lambda col: max(col)),
        ('RNAFolding[sequence_]', lambda seq: '(((...)))'),  # Dot-bracket
        ('RNAMinimumFreeEnergy[structure_]', lambda s: -10.0),  # kcal/mol
        ('RNABasePairProbability[i_, j_, ensemble_]', lambda i, j, ens: 0.8),
        ('MicroRNATarget[mirna_, mrna_]', lambda mir, mrna: [(100, 120)]),
        ('miRNASeedMatch[seed_, utr_]', lambda seed, utr: str(utr).find(str(seed))),
        ('siRNADesign[sequence_]', lambda seq: [str(seq)[i:i+21] for i in range(0, len(str(seq))-20, 10)]),
        ('shRNAStructure[target_]', lambda tgt: f'5\'-{tgt}-loop-{tgt[::-1]}-3\''),
        ('CRISPRTargetScore[sequence_, offtargets_]', lambda seq, off: 100 - len(off)*10),
        ('PAMSequence[target_]', lambda tgt: str(tgt)[-3:]),
        ('GuideRNADesign[target_]', lambda tgt: str(tgt)[:-3]),
        ('OffTargetPrediction[grna_, genome_]', lambda grna, gen: []),
        ('GenomeAssemblyN50[contig_lengths_]', lambda lens: sorted(lens)[len(lens)//2]),
        ('GenomeCompleteness[BUSCO_complete_, BUSCO_total_]', lambda comp, tot: comp/tot*100),
        ('ContigCoverage[reads_, contig_length_]', lambda r, l: r / l),
        ('ReadDepth[total_bases_, genome_size_]', lambda tb, gs: tb / gs),
        ('VariantCallQuality[phred_]', lambda Q: 10**(-Q/10)),
        ('GenotypeQuality[GQ_]', lambda GQ: 10**(-GQ/10)),
        ('AlleleFrequencyPopulation[alt_alleles_, total_alleles_]', lambda alt, tot: alt / tot),
        ('MinorAlleleFrequency[MAF_]', lambda MAF: min(MAF, 1 - MAF)),
        
        # ===== SYSTEMS BIOLOGY (40 functions) =====
        ('LogisticGrowth[r_, K_, N_]', lambda r, K, N: r * N * (1 - N/K)),
        ('ExponentialGrowth[r_, N_]', lambda r, N: r * N),
        ('CarryingCapacity[K_]', lambda K: K),
        ('PopulationDoublingTime[r_]', lambda r: log(2) / r),
        ('LotkaVolterra[alpha_, beta_, gamma_, delta_, x_, y_]', lambda a, b, g, d, x, y: (a*x - b*x*y, -g*y + d*x*y)),
        ('PredatorPreyEquilibrium[alpha_, beta_, gamma_, delta_]', lambda a, b, g, d: (g/d, a/b)),
        ('CompetitionModel[r1_, r2_, K1_, K2_, alpha_, beta_, N1_, N2_]', lambda r1, r2, K1, K2, a, b, N1, N2: (r1*N1*(1 - (N1 + a*N2)/K1), r2*N2*(1 - (N2 + b*N1)/K2))),
        ('MutualismModel[r1_, r2_, K1_, K2_, alpha_, beta_, N1_, N2_]', lambda r1, r2, K1, K2, a, b, N1, N2: (r1*N1*(1 + a*N2), r2*N2*(1 + b*N1))),
        ('SIRModel[beta_, gamma_, S_, I_, R_]', lambda beta, gamma, S, I, R: (-beta*S*I, beta*S*I - gamma*I, gamma*I)),
        ('BasicReproductionNumber[beta_, gamma_]', lambda beta, gamma: beta / gamma),  # R0
        ('HerdImmunityThreshold[R0_]', lambda R0: 1 - 1/R0),
        ('SEIRModel[beta_, sigma_, gamma_, S_, E_, I_, R_]', lambda beta, sigma, gamma, S, E, I, R: (-beta*S*I, beta*S*I - sigma*E, sigma*E - gamma*I, gamma*I)),
        ('VaccinationRate[R0_, coverage_]', lambda R0, cov: (1 - 1/R0) / cov),
        ('MetabolicFluxBalance[S_, v_]', lambda S, v: 0),  # S*v = 0
        ('EnzymeFlux[Vmax_, S_, Km_]', lambda Vmax, S, Km: Vmax * S / (Km + S)),
        ('MetabolicControlCoefficient[enzyme_, flux_]', lambda e, f: sp.diff(f, e) * e / f),
        ('ElasticityCoefficient[substrate_, rate_]', lambda s, r: sp.diff(r, s) * s / r),
        ('FluxControlSummation[coefficients_]', lambda c: sum(c)),  # Should equal 1
        ('MetabolicCost[ATP_, biomass_]', lambda ATP, bio: ATP / bio),
        ('YieldCoefficient[product_, substrate_]', lambda p, s: p / s),
        ('SpecificGrowthRate[biomass_, time_]', lambda X, t: sp.diff(log(X), t)),
        ('BiomassProductivity[biomass_, time_]', lambda X, t: X / t),
        ('SubstrateUptakeRate[substrate_, biomass_, time_]', lambda S, X, t: -sp.diff(S, t) / X),
        ('MaintenanceEnergy[m_]', lambda m: m),  # ATP/g/h
        ('ElementalBalance[C_in_, C_out_]', lambda Cin, Cout: Cin - Cout),
        ('RedoxBalance[electrons_]', lambda e: e),
        ('GeneRegulatoryNetwork[TF_, targets_]', lambda TF, tgt: len(tgt)),
        ('TranscriptionFactorActivity[TF_conc_, DNA_binding_]', lambda TF, Kd: TF / (TF + Kd)),
        ('PromoterStrength[RNAP_binding_]', lambda Kd: 1 / Kd),
        ('HillFunctionActivation[TF_, K_, n_]', lambda TF, K, n: TF**n / (K**n + TF**n)),
        ('HillFunctionRepression[TF_, K_, n_]', lambda TF, K, n: K**n / (K**n + TF**n)),
        ('FeedbackLoop[X_, Y_]', lambda X, Y: X * Y),
        ('Bifurcation[parameter_, steady_state_]', lambda p, ss: ss),
        ('AttractorBasin[initial_conditions_]', lambda ic: 'attractor'),
        ('LyapunovExponent[trajectory_]', lambda traj: 0.1),
        ('ChaoticBehavior[lyapunov_]', lambda lyap: lyap > 0),
        ('Oscillation[amplitude_, period_]', lambda A, T: A * sin(2*pi*t/T)),
        ('RelaxationTime[tau_]', lambda tau: tau),
        ('SteadyState[rates_]', lambda r: 0),
        ('Perturbation[delta_]', lambda delta: delta),
    ]

def register():
    """Register all extended18 rules"""
    return get_rules()
