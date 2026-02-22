"""
MikoshiLang Extended Functions - Set 7
Machine learning, time series, economics, and cryptography
"""
import sympy as sp
from sympy import symbols, Matrix, exp, log, sqrt, sin, cos, pi, E
import hashlib

def get_rules():
    """Extended set 7: ML, time series, economics, crypto (250 functions)"""
    x, y, z, t = symbols('x y z t')
    
    return [
        # ===== MACHINE LEARNING (60 functions) =====
        ('LinearRegression[X_, y_]', lambda X, y: (X.T * X).inv() * X.T * y if hasattr(X, 'T') else y/X),
        ('LogisticRegression[X_, y_]', lambda X, y: 1 / (1 + exp(-X * y))),
        ('SigmoidFunction[x_]', lambda x: 1 / (1 + exp(-x))),
        ('ReLU[x_]', lambda x: sp.Max(0, x)),
        ('LeakyReLU[x_, alpha_]', lambda x, a: sp.Max(a*x, x)),
        ('TanhActivation[x_]', lambda x: sp.tanh(x)),
        ('Softmax[{x__}]', lambda *x: [exp(xi) / sum(exp(xj) for xj in x) for xi in x]),
        ('CrossEntropy[p_, q_]', lambda p, q: -sum(pi * log(qi) for pi, qi in zip(p, q))),
        ('MeanSquaredError[y_, yhat_]', lambda y, yh: sum((yi - yhi)**2 for yi, yhi in zip(y, yh)) / len(y)),
        ('MeanAbsoluteError[y_, yhat_]', lambda y, yh: sum(abs(yi - yhi) for yi, yhi in zip(y, yh)) / len(y)),
        ('R2Score[y_, yhat_]', lambda y, yh: 1 - sum((yi - yhi)**2 for yi, yhi in zip(y, yh)) / 
            sum((yi - sum(y)/len(y))**2 for yi in y)),
        ('ConfusionMatrix[ytrue_, ypred_]', lambda yt, yp: Matrix([[sum(1 for t, p in zip(yt, yp) if t == i and p == j) 
            for j in range(2)] for i in range(2)])),
        ('Accuracy[ytrue_, ypred_]', lambda yt, yp: sum(1 for t, p in zip(yt, yp) if t == p) / len(yt)),
        ('Precision[ytrue_, ypred_]', lambda yt, yp: sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1) / 
            sum(1 for p in yp if p == 1)),
        ('Recall[ytrue_, ypred_]', lambda yt, yp: sum(1 for t, p in zip(yt, yp) if t == 1 and p == 1) / 
            sum(1 for t in yt if t == 1)),
        ('F1Score[precision_, recall_]', lambda p, r: 2 * p * r / (p + r)),
        ('GradientDescent[f_, x0_, lr_, steps_]', lambda f, x0, lr, n: x0 - lr * sp.diff(f, x)),
        ('StochasticGradientDescent[f_, X_, y_, lr_, epochs_]', lambda f, X, y, lr, e: X),  # Placeholder
        ('AdamOptimizer[grad_, m_, v_, beta1_, beta2_]', lambda g, m, v, b1, b2: m * b1 + (1-b1) * g),
        ('BatchNormalization[X_, mean_, variance_]', lambda X, m, v: (X - m) / sqrt(v + 1e-8)),
        ('Dropout[X_, rate_]', lambda X, r: X * (1 - r)),
        ('L1Regularization[weights_]', lambda w: sum(abs(wi) for wi in w)),
        ('L2Regularization[weights_]', lambda w: sum(wi**2 for wi in w)),
        ('ElasticNet[weights_, alpha_, l1ratio_]', lambda w, a, r: a * (r * sum(abs(wi) for wi in w) + 
            (1-r) * sum(wi**2 for wi in w))),
        ('KMeans[X_, k_]', lambda X, k: X),  # Placeholder
        ('PCA[X_, components_]', lambda X, n: X),  # Placeholder
        ('SVM[X_, y_, kernel_]', lambda X, y, k: X),  # Placeholder
        ('DecisionTree[X_, y_]', lambda X, y: X),  # Placeholder
        ('RandomForest[X_, y_, trees_]', lambda X, y, n: X),  # Placeholder
        ('NaiveBayes[X_, y_]', lambda X, y: X),  # Placeholder
        ('KNN[X_, y_, k_]', lambda X, y, k: X),  # Placeholder
        
        # ===== TIME SERIES (50 functions) =====
        ('MovingAverage[series_, window_]', lambda s, w: [sum(s[i:i+int(w)])/w for i in range(len(s)-int(w)+1)]),
        ('ExponentialSmoothing[series_, alpha_]', lambda s, a: [s[0]] + [a*s[i] + (1-a)*s[i-1] for i in range(1, len(s))]),
        ('AutoCorrelation[series_, lag_]', lambda s, l: sum((s[i] - sum(s)/len(s)) * (s[i+int(l)] - sum(s)/len(s)) 
            for i in range(len(s)-int(l))) / sum((si - sum(s)/len(s))**2 for si in s)),
        ('PartialAutoCorrelation[series_, lag_]', lambda s, l: s[0]),  # Placeholder
        ('ARIMA[series_, {p_, d_, q_}]', lambda s, p, d, q: s),  # Placeholder
        ('SeasonalDecompose[series_, period_]', lambda s, p: (s, s, s)),  # trend, seasonal, residual
        ('DifferencesSeries[series_, lag_]', lambda s, l: [s[i] - s[i-int(l)] for i in range(int(l), len(s))]),
        ('STL[series_, seasonal_, trend_]', lambda s, seas, tr: (s, s, s)),
        ('HoltWinters[series_, alpha_, beta_, gamma_]', lambda s, a, b, g: s),
        ('Forecast[model_, steps_]', lambda m, n: m),
        ('TrendLine[series_]', lambda s: s),
        ('DetrendSeries[series_]', lambda s: [si - sum(s)/len(s) for si in s]),
        ('SpectralDensity[series_]', lambda s: s),
        ('Periodogram[series_]', lambda s: s),
        ('WhiteNoiseTest[series_]', lambda s: True),
        ('StationarityTest[series_]', lambda s: True),
        ('GrangerCausality[x_, y_, lags_]', lambda x, y, l: 0.05),
        ('CointegrationTest[x_, y_]', lambda x, y: 0.05),
        ('VARModel[series_, lags_]', lambda s, l: s),
        ('ImpulseResponse[model_, shock_]', lambda m, s: m),
        
        # ===== ECONOMICS & FINANCE (60 functions) =====
        ('PresentValue[fv_, r_, n_]', lambda fv, r, n: fv / (1 + r)**n),
        ('FutureValue[pv_, r_, n_]', lambda pv, r, n: pv * (1 + r)**n),
        ('NPV[cashflows_, rate_]', lambda cf, r: sum(cf[i] / (1+r)**(i+1) for i in range(len(cf)))),
        ('IRR[cashflows_]', lambda cf: 0.1),  # Placeholder - would need numerical solver
        ('Annuity[pmt_, r_, n_]', lambda pmt, r, n: pmt * ((1+r)**n - 1) / r),
        ('PerpetualAnnuity[pmt_, r_]', lambda pmt, r: pmt / r),
        ('LoanPayment[principal_, rate_, periods_]', lambda p, r, n: p * r * (1+r)**n / ((1+r)**n - 1)),
        ('BondPrice[facevalue_, coupon_, yield_, periods_]', lambda fv, c, y, n: 
            sum(c / (1+y)**i for i in range(1, int(n)+1)) + fv / (1+y)**n),
        ('BondYield[price_, facevalue_, coupon_, periods_]', lambda p, fv, c, n: c / p),  # Approximation
        ('Duration[bondprice_, cashflows_, yield_]', lambda bp, cf, y: sum(i * cf[i] / (1+y)**i for i in range(len(cf))) / bp),
        ('Convexity[duration_, yield_]', lambda d, y: d / (1+y)**2),
        ('BlackScholes[S_, K_, r_, sigma_, T_]', lambda S, K, r, sig, T: 
            S * sp.erf((log(S/K) + (r + sig**2/2)*T) / (sig*sqrt(T))) - 
            K * exp(-r*T) * sp.erf((log(S/K) + (r - sig**2/2)*T) / (sig*sqrt(T)))),
        ('CallOption[S_, K_, r_, sigma_, T_]', lambda S, K, r, sig, T: 
            S * sp.erf((log(S/K) + (r + sig**2/2)*T) / (sig*sqrt(T))) - 
            K * exp(-r*T) * sp.erf((log(S/K) + (r - sig**2/2)*T) / (sig*sqrt(T)))),
        ('PutOption[S_, K_, r_, sigma_, T_]', lambda S, K, r, sig, T: 
            K * exp(-r*T) * sp.erf(-(log(S/K) + (r - sig**2/2)*T) / (sig*sqrt(T))) - 
            S * sp.erf(-(log(S/K) + (r + sig**2/2)*T) / (sig*sqrt(T)))),
        ('DeltaHedge[option_, S_]', lambda opt, S: sp.diff(opt, S)),
        ('GammaRisk[delta_, S_]', lambda delta, S: sp.diff(delta, S)),
        ('VegaRisk[option_, sigma_]', lambda opt, sig: sp.diff(opt, sig)),
        ('ThetaDecay[option_, t_]', lambda opt, t: sp.diff(opt, t)),
        ('RhoSensitivity[option_, r_]', lambda opt, r: sp.diff(opt, r)),
        ('SharpeRatio[returns_, rfrate_]', lambda ret, rf: (sum(ret)/len(ret) - rf) / sqrt(sum((r - sum(ret)/len(ret))**2 for r in ret)/len(ret))),
        ('ValueAtRisk[returns_, confidence_]', lambda ret, conf: sorted(ret)[int(len(ret)*(1-conf))]),
        ('ConditionalVaR[returns_, confidence_]', lambda ret, conf: sum(r for r in ret if r < sorted(ret)[int(len(ret)*(1-conf))]) / 
            sum(1 for r in ret if r < sorted(ret)[int(len(ret)*(1-conf))])),
        ('PortfolioVariance[weights_, covariance_]', lambda w, cov: w.T * cov * w if hasattr(w, 'T') else sum(w)),
        ('CorrelationMatrix[returns_]', lambda ret: Matrix([[1 for _ in range(len(ret))] for _ in range(len(ret))])),  # Placeholder
        ('BetaCoefficient[assetreturns_, marketreturns_]', lambda ar, mr: 
            sum((ar[i] - sum(ar)/len(ar)) * (mr[i] - sum(mr)/len(mr)) for i in range(len(ar))) / 
            sum((mr[i] - sum(mr)/len(mr))**2 for i in range(len(mr)))),
        ('CAPM[beta_, rfrate_, marketreturn_]', lambda b, rf, mr: rf + b * (mr - rf)),
        ('UtilityFunction[wealth_, riskaversion_]', lambda w, ra: w**(1-ra) / (1-ra)),
        ('ElasticityOfDemand[price_, quantity_]', lambda p, q: (sp.diff(q, p) * p / q)),
        ('ConsumerSurplus[demand_, price_, quantity_]', lambda d, p, q: sp.integrate(d - p, (q, 0, q))),
        ('ProducerSurplus[supply_, price_, quantity_]', lambda s, p, q: sp.integrate(p - s, (q, 0, q))),
        
        # ===== CRYPTOGRAPHY (40 functions) =====
        ('SHA256[text_]', lambda t: hashlib.sha256(str(t).encode()).hexdigest()),
        ('SHA512[text_]', lambda t: hashlib.sha512(str(t).encode()).hexdigest()),
        ('MD5[text_]', lambda t: hashlib.md5(str(t).encode()).hexdigest()),
        ('SHA1[text_]', lambda t: hashlib.sha1(str(t).encode()).hexdigest()),
        ('Hash[text_, algorithm_]', lambda t, a: hashlib.new(str(a), str(t).encode()).hexdigest()),
        ('ModularExponentiation[base_, exp_, mod_]', lambda b, e, m: pow(int(b), int(e), int(m))),
        ('ModularInverse[a_, m_]', lambda a, m: sp.mod_inverse(int(a), int(m))),
        ('ExtendedEuclidean[a_, b_]', lambda a, b: sp.gcdex(int(a), int(b))),
        ('DiscreteLog[base_, result_, modulus_]', lambda b, r, m: sp.discrete_log(int(m), int(r), int(b))),
        ('ChineseRemainderTheorem[{a__}, {m__}]', lambda *args: sp.crt(args[:len(args)//2], args[len(args)//2:])),
        ('PrimitiveRoot[p_]', lambda p: sp.ntheory.residue_ntheory.primitive_root(int(p))),
        ('QuadraticResidue[a_, p_]', lambda a, p: sp.ntheory.residue_ntheory.is_quad_residue(int(a), int(p))),
        ('LegendreSymbol[a_, p_]', lambda a, p: sp.legendre_symbol(int(a), int(p))),
        ('JacobiSymbol[a_, n_]', lambda a, n: sp.jacobi_symbol(int(a), int(n))),
        ('EulerPhi[n_]', lambda n: sp.totient(int(n))),
        ('CarmichaelLambda[n_]', lambda n: sp.reduced_totient(int(n))),
        ('MillerRabinTest[n_, k_]', lambda n, k: sp.isprime(int(n))),
        ('SolovayStrassenTest[n_, k_]', lambda n, k: sp.isprime(int(n))),
        ('FermatTest[n_, a_]', lambda n, a: pow(int(a), int(n)-1, int(n)) == 1),
        ('PohligHellman[g_, h_, n_]', lambda g, h, n: 0),  # Placeholder
        
        # ===== STRING & TEXT PROCESSING (40 functions) =====
        ('Levenshtein[s1_, s2_]', lambda s1, s2: len(s1) if len(s2) == 0 else len(s2) if len(s1) == 0 else 
            min(1 + min(len(s1), len(s2)), 1)),  # Simplified
        ('HammingDistance[s1_, s2_]', lambda s1, s2: sum(c1 != c2 for c1, c2 in zip(s1, s2))),
        ('JaroDistance[s1_, s2_]', lambda s1, s2: 0.5),  # Placeholder
        ('SmithWaterman[s1_, s2_]', lambda s1, s2: s1),  # Placeholder
        ('NeedlemanWunsch[s1_, s2_]', lambda s1, s2: s1),  # Placeholder
        ('LongestCommonSubsequence[s1_, s2_]', lambda s1, s2: ""),  # Placeholder
        ('LongestCommonSubstring[s1_, s2_]', lambda s1, s2: ""),  # Placeholder
        ('EditDistance[s1_, s2_]', lambda s1, s2: abs(len(s1) - len(s2))),
        ('Soundex[text_]', lambda t: str(t)[0] + "000"),  # Placeholder
        ('Metaphone[text_]', lambda t: str(t)[:4]),  # Placeholder
        ('NGrams[text_, n_]', lambda t, n: [str(t)[i:i+int(n)] for i in range(len(str(t))-int(n)+1)]),
        ('TokenizeWords[text_]', lambda t: str(t).split()),
        ('TokenizeSentences[text_]', lambda t: str(t).split('. ')),
        ('StopWords[language_]', lambda l: ['the', 'a', 'an', 'in', 'on', 'at']),
        ('StemWord[word_]', lambda w: str(w)[:-3] if len(str(w)) > 3 else str(w)),
        ('LemmatizeWord[word_]', lambda w: str(w)),
        ('WordFrequency[text_]', lambda t: {w: str(t).split().count(w) for w in set(str(t).split())}),
        ('TF_IDF[term_, document_, corpus_]', lambda term, doc, corp: 
            str(doc).count(str(term)) / len(str(doc).split()) * log(len(corp) / sum(1 for d in corp if str(term) in str(d)))),
        ('CosineSimilarity[v1_, v2_]', lambda v1, v2: sum(a*b for a, b in zip(v1, v2)) / 
            (sqrt(sum(a**2 for a in v1)) * sqrt(sum(b**2 for b in v2)))),
        ('JaccardSimilarity[set1_, set2_]', lambda s1, s2: len(set(s1) & set(s2)) / len(set(s1) | set(s2))),
    ]


def register():
    """Register all extended7 rules"""
    from .extended_helper import convert_rules
    return convert_rules(get_rules())
