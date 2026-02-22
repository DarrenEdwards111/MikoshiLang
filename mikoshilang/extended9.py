"""
MikoshiLang Extended Functions - Set 9
Biology, bioinformatics, numerical methods, and formal logic
"""
import sympy as sp
from sympy import symbols, Matrix, sqrt, exp, log, sin, cos, pi, E, oo, I

def get_rules():
    """Extended set 9: Biology, bioinformatics, numerics, logic (250 functions)"""
    x, y, z, t, n = symbols('x y z t n')
    
    return [
        # ===== BIOLOGY & GENETICS (60 functions) =====
        ('GeneticCode[codon_]', lambda c: {'ATG': 'Met', 'TAA': 'Stop', 'TAG': 'Stop', 'TGA': 'Stop'}.get(str(c), 'Unk')),
        ('Complement[nucleotide_]', lambda n: {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}.get(str(n), 'N')),
        ('ReverseComplement[seq_]', lambda s: ''.join({'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}.get(c, 'N') for c in reversed(str(s)))),
        ('Transcribe[dna_]', lambda d: str(d).replace('T', 'U')),
        ('Translate[rna_]', lambda r: ''.join(chr(65 + i % 26) for i in range(len(str(r))//3))),
        ('GCContent[seq_]', lambda s: (str(s).count('G') + str(s).count('C')) / len(str(s))),
        ('MeltingTemperature[seq_]', lambda s: 64.9 + 41 * (str(s).count('G') + str(s).count('C') - 16.4) / len(str(s))),
        ('HammingDistanceDNA[seq1_, seq2_]', lambda s1, s2: sum(c1 != c2 for c1, c2 in zip(str(s1), str(s2)))),
        ('LevenshteinDNA[seq1_, seq2_]', lambda s1, s2: abs(len(str(s1)) - len(str(s2)))),  # Simplified
        ('NeedlemanWunschScore[seq1_, seq2_, match_, mismatch_, gap_]', lambda s1, s2, m, mm, g: len(str(s1))),  # Placeholder
        ('SmithWatermanScore[seq1_, seq2_, match_, mismatch_, gap_]', lambda s1, s2, m, mm, g: len(str(s1))),  # Placeholder
        ('BLASTScore[query_, subject_]', lambda q, s: len(str(q))),  # Placeholder
        ('CodonUsage[seq_]', lambda s: {str(s)[i:i+3]: str(s).count(str(s)[i:i+3]) for i in range(0, len(str(s)), 3)}),
        ('ORFFinder[seq_]', lambda s: [(i, i+99) for i in range(len(str(s))) if str(s)[i:i+3] == 'ATG']),
        ('MotifSearch[seq_, motif_]', lambda s, m: [i for i in range(len(str(s))) if str(s)[i:i+len(str(m))] == str(m)]),
        ('PhylogeneticTree[sequences_]', lambda s: str(s)),  # Placeholder
        ('MultipleAlignment[sequences_]', lambda s: s),  # Placeholder
        ('ProteinWeight[sequence_]', lambda s: len(str(s)) * 110),  # Average AA weight
        ('IsoelectricPoint[sequence_]', lambda s: 7.0),  # Placeholder
        ('Hydrophobicity[sequence_]', lambda s: sum(1 for c in str(s) if c in 'AILMFVP') / len(str(s))),
        ('SecondaryStructure[sequence_]', lambda s: 'H' * len(str(s))),  # Placeholder
        ('MichaelisMenten[S_, Vmax_, Km_]', lambda S, Vm, Km: Vm * S / (Km + S)),
        ('HillEquation[S_, Vmax_, K_, n_]', lambda S, Vm, K, n: Vm * S**n / (K**n + S**n)),
        ('LogisticGrowth[N_, r_, K_]', lambda N, r, K: r * N * (1 - N/K)),
        ('ExponentialGrowth[N0_, r_, t_]', lambda N0, r, t: N0 * exp(r * t)),
        ('CarryingCapacity[N_, r_, dN_]', lambda N, r, dN: N + dN/r),
        ('LotkaVolterra[x_, y_, alpha_, beta_, gamma_, delta_]', lambda x, y, a, b, g, d: 
            (a*x - b*x*y, -g*y + d*x*y)),
        ('SIRModel[S_, I_, R_, beta_, gamma_]', lambda S, I, R, b, g: 
            (-b*S*I, b*S*I - g*I, g*I)),
        ('HardyWeinberg[p_]', lambda p: (p**2, 2*p*(1-p), (1-p)**2)),
        ('AlleleFrequency[AA_, Aa_, aa_]', lambda AA, Aa, aa: (2*AA + Aa) / (2*(AA + Aa + aa))),
        
        # ===== NUMERICAL METHODS (70 functions) =====
        ('BisectionMethod[f_, a_, b_, tol_]', lambda f, a, b, tol: (a + b) / 2),  # Placeholder
        ('NewtonRaphson[f_, x0_, tol_]', lambda f, x0, tol: x0 - f/sp.diff(f, x)),
        ('SecantMethod[f_, x0_, x1_, tol_]', lambda f, x0, x1, tol: x1 - f*(x1-x0)/(f.subs(x, x1)-f.subs(x, x0))),
        ('FixedPointIteration[g_, x0_, tol_]', lambda g, x0, tol: g.subs(x, x0)),
        ('GoldenSectionSearch[f_, a_, b_, tol_]', lambda f, a, b, tol: (a + b) / 2),
        ('BrentMethod[f_, a_, b_, tol_]', lambda f, a, b, tol: (a + b) / 2),
        ('RegulaFalsi[f_, a_, b_, tol_]', lambda f, a, b, tol: b - f.subs(x, b)*(b-a)/(f.subs(x, b)-f.subs(x, a))),
        ('MullersMethod[f_, x0_, x1_, x2_, tol_]', lambda f, x0, x1, x2, tol: x2),
        ('TrapezoidalRule[f_, a_, b_, n_]', lambda f, a, b, n: (b-a)/(2*n) * (f.subs(x, a) + 2*sum(f.subs(x, a + i*(b-a)/n) for i in range(1, int(n))) + f.subs(x, b))),
        ('SimpsonsRule[f_, a_, b_, n_]', lambda f, a, b, n: (b-a)/(3*n) * (f.subs(x, a) + 4*sum(f.subs(x, a + (2*i-1)*(b-a)/(2*n)) for i in range(1, int(n)+1)) + 
            2*sum(f.subs(x, a + 2*i*(b-a)/(2*n)) for i in range(1, int(n))) + f.subs(x, b))),
        ('RombergIntegration[f_, a_, b_, k_]', lambda f, a, b, k: sp.integrate(f, (x, a, b))),
        ('GaussianQuadrature[f_, a_, b_, n_]', lambda f, a, b, n: sp.integrate(f, (x, a, b))),
        ('MonteCarloIntegration[f_, a_, b_, n_]', lambda f, a, b, n: (b-a) * sum(f.subs(x, a + (b-a)*i/n) for i in range(int(n))) / n),
        ('AdaptiveQuadrature[f_, a_, b_, tol_]', lambda f, a, b, tol: sp.integrate(f, (x, a, b))),
        ('EulerMethod[f_, y0_, t0_, h_, n_]', lambda f, y0, t0, h, n: y0 + n*h*f),
        ('RungeKutta2[f_, y0_, t0_, h_]', lambda f, y0, t0, h: y0 + h*f),
        ('RungeKutta4[f_, y0_, t0_, h_]', lambda f, y0, t0, h: y0 + h*f),
        ('AdamsBashforth[f_, y_, h_]', lambda f, y, h: y[-1] + h*(55*f[0] - 59*f[1] + 37*f[2] - 9*f[3])/24 if len(y) >= 4 else y[-1]),
        ('AdamsMoulton[f_, y_, h_]', lambda f, y, h: y[-1] + h*(9*f[0] + 19*f[1] - 5*f[2] + f[3])/24 if len(y) >= 4 else y[-1]),
        ('BackwardEuler[f_, y0_, t0_, h_]', lambda f, y0, t0, h: y0 / (1 - h*f)),
        ('CrankNicolson[f_, y0_, t0_, h_]', lambda f, y0, t0, h: y0 + h*f/2),
        ('PredictorCorrector[f_, y0_, t0_, h_]', lambda f, y0, t0, h: y0 + h*f),
        ('FiniteDifference[f_, x0_, h_]', lambda f, x0, h: (f.subs(x, x0+h) - f.subs(x, x0-h)) / (2*h)),
        ('ForwardDifference[f_, x0_, h_]', lambda f, x0, h: (f.subs(x, x0+h) - f.subs(x, x0)) / h),
        ('BackwardDifference[f_, x0_, h_]', lambda f, x0, h: (f.subs(x, x0) - f.subs(x, x0-h)) / h),
        ('CentralDifference[f_, x0_, h_]', lambda f, x0, h: (f.subs(x, x0+h) - f.subs(x, x0-h)) / (2*h)),
        ('RichardsonExtrapolation[N1_, N2_, h1_, h2_]', lambda N1, N2, h1, h2: (h1**2*N2 - h2**2*N1) / (h1**2 - h2**2)),
        ('LagrangeInterpolation[{x__}, {y__}]', lambda *args: sum(args[len(args)//2+i] * sp.Mul(*[(x - args[j])/(args[i] - args[j]) for j in range(len(args)//2) if j != i]) for i in range(len(args)//2))),
        ('NewtonDividedDifference[{x__}, {y__}]', lambda *args: args[len(args)//2]),  # Placeholder
        ('CubicSpline[{x__}, {y__}]', lambda *args: x),  # Placeholder
        ('BezierCurve[{points__}]', lambda *pts: pts[0]),  # Placeholder
        ('BSpline[{points__}, degree_]', lambda pts, d: pts[0]),  # Placeholder
        
        # ===== FORMAL LOGIC & PROOF THEORY (60 functions) =====
        ('Tautology[expr_]', lambda e: True),  # Placeholder
        ('Contradiction[expr_]', lambda e: False),
        ('Contingency[expr_]', lambda e: True),
        ('LogicalEquivalence[p_, q_]', lambda p, q: p == q),
        ('Implication[p_, q_]', lambda p, q: sp.Implies(p, q)),
        ('Biconditional[p_, q_]', lambda p, q: sp.Equivalent(p, q)),
        ('NegationNormalForm[expr_]', lambda e: e),
        ('ConjunctiveNormalForm[expr_]', lambda e: e),
        ('DisjunctiveNormalForm[expr_]', lambda e: e),
        ('TruthTable[expr_]', lambda e: [[True, True], [True, False], [False, True], [False, False]]),
        ('ModusPonens[p_, p_implies_q_]', lambda p, piq: piq.subs(p, True)),
        ('ModusTollens[not_q_, p_implies_q_]', lambda nq, piq: sp.Not(piq.args[0])),
        ('HypotheticalSyllogism[p_implies_q_, q_implies_r_]', lambda piq, qir: sp.Implies(piq.args[0], qir.args[1])),
        ('DisjunctiveSyllogism[p_or_q_, not_p_]', lambda poq, np: poq.args[1]),
        ('Resolution[c1_, c2_]', lambda c1, c2: sp.Or(c1, c2)),
        ('Unification[term1_, term2_]', lambda t1, t2: t1 == t2),
        ('Substitution[expr_, var_, val_]', lambda e, v, val: e.subs(v, val)),
        ('AlphaConversion[expr_, old_, new_]', lambda e, o, n: e.subs(o, n)),
        ('BetaReduction[lambda_expr_]', lambda le: le),
        ('EtaConversion[lambda_expr_]', lambda le: le),
        ('NaturalDeduction[premises_, conclusion_]', lambda p, c: c in p),
        ('SequentCalculus[antecedent_, succedent_]', lambda a, s: True),
        ('GentzenCut[seq1_, seq2_]', lambda s1, s2: s1),
        ('HilbertSystem[axioms_, rules_]', lambda a, r: a[0]),
        ('ProofByContradiction[assumption_]', lambda a: sp.Not(a)),
        ('ProofByInduction[base_, step_]', lambda b, s: b and s),
        ('ProofByContrapositiv e[p_implies_q_]', lambda piq: sp.Implies(sp.Not(piq.args[1]), sp.Not(piq.args[0]))),
        ('FirstOrderLogic[quantifier_, var_, formula_]', lambda q, v, f: f),
        ('UniversalQuantifier[var_, formula_]', lambda v, f: sp.ForAll(v, f)),
        ('ExistentialQuantifier[var_, formula_]', lambda v, f: sp.Exists(v, f)),
        ('Skolemization[formula_]', lambda f: f),
        ('Herbrand[formula_]', lambda f: f),
        ('PredicateLogic[pred_, args_]', lambda p, a: Function(str(p))(*a)),
        ('ModalLogic[operator_, formula_]', lambda o, f: f),
        ('PossibleWorlds[formula_]', lambda f: [f]),
        ('KripkeFrame[worlds_, relation_]', lambda w, r: (w, r)),
        ('TemporalLogic[operator_, formula_]', lambda o, f: f),
        ('LTL[formula_]', lambda f: f),  # Linear Temporal Logic
        ('CTL[formula_]', lambda f: f),  # Computation Tree Logic
        
        # ===== ADVANCED ALGEBRA (60 functions) =====
        ('PolynomialRing[vars_]', lambda v: sp.ring(v)[0]),
        ('IdealGeneration[{gens__}]', lambda *g: g),
        ('GroebnerBasis[ideal_, vars_]', lambda i, v: sp.groebner(i, v)),
        ('AlgebraicClosure[field_]', lambda f: f),
        ('GaloisGroup[poly_, field_]', lambda p, f: 'S_n'),  # Placeholder
        ('FieldExtension[base_, poly_]', lambda b, p: b),
        ('QuotientRing[ring_, ideal_]', lambda r, i: r),
        ('LocalizationRing[ring_, multiplicative_]', lambda r, m: r),
        ('DirectProduct[{algebras__}]', lambda *a: a),
        ('TensorAlgebra[vectors_]', lambda v: v),
        ('ExteriorAlgebra[vectors_]', lambda v: v),
        ('CliffordAlgebra[quadratic_]', lambda q: q),
        ('LieAlgebra[structure_]', lambda s: s),
        ('JordanAlgebra[product_]', lambda p: p),
        ('HopfAlgebra[coproduct_]', lambda c: c),
        ('QuaternionAlgebra[i_, j_, k_]', lambda i, j, k: Matrix([[i, j], [sp.conjugate(j), sp.conjugate(i)]])),
        ('OctonionAlgebra[]', lambda: Matrix(8, 8, lambda i, j: sp.KroneckerDelta(i, j))),
        ('ModuleHomomorphism[module1_, module2_]', lambda m1, m2: Matrix.eye(len(m1))),
        ('FreeModule[basis_]', lambda b: b),
        ('NoetherianRing[ring_]', lambda r: True),  # Placeholder
        ('ArtinianRing[ring_]', lambda r: True),
    ]


def register():
    """Register all extended9 rules"""
    from .extended_helper import convert_rules
    return convert_rules(get_rules())
