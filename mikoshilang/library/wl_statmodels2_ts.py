"""Wolfram batch X02 (part 3): hidden Markov processes (HiddenMarkovProcess, FindHiddenMarkovStates; LogLikelihood,
EstimatedProcess, StationaryDistribution, RandomFunction and slices through the W25 process framework), time-series
models (TimeSeriesModelFit, TimeSeriesModel, AdjustTimeSeriesForecast), time / event series utilities
(TimeSeriesSummary, TimeSeriesStructure, TimeSeriesEvents, EventSeriesAccumulate, the TimeObservationWindow
option) and Jackson / Gordon-Newell queueing networks (QueueingNetworkProcess).

Methods are the textbook ones and are named in each function's notes: Viterbi and forward-backward decoding and
the forward-algorithm likelihood (Rabiner 1989), Baum-Welch EM, exact Gaussian ARMA maximum likelihood by the
Kalman filter (Harvey 1989) with the Jones (1980) stationarity / invertibility reparametrisation, Box-Jenkins
forecast updating, Jackson's product form and Gordon-Newell normalisation.  Where Wolfram's automatic choice is
undocumented, the choice made here is stated, never claimed to match.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import sympy as sp

from ..expr import Expr, Symbol
from .. import kernel as K
from .. import kernel_lists as KL
from .core import Unevaluated, fn, L, R, rule, S, is_list, has_float
from . import wl_statmodels as SM
from . import wl_stats as W
from . import wl_processes as P
from . import wl_processes_ts as T

AREA = "statistical models II (X02)"
UNH = K.UNHANDLED
ev = SM.ev


def _sym(x, *names):
    return isinstance(x, Symbol) and (not names or x.name in names)


def _int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def _isnum(v):
    try:
        s = S(v)
    except Exception:
        return False
    return isinstance(s, sp.Basic) and s.is_number and s.is_real is not False


def _missing(v):
    return isinstance(v, Expr) and v.head == "Missing"


# ─────────────────────────── hidden Markov processes ───────────────────────────

class HMM(P.Proc):
    """HiddenMarkovProcess[i0 | p0, m, em | {dist1, ...}]"""
    dtime = True
    has_cov = False

    def parse(self, a, fname):
        if len(a) == 2:
            if fname == self.name:
                raise Unevaluated()
            raise NotImplementedError(f"{fname}: HiddenMarkovProcess[n, edist] is an estimation template; only "
                                      "EstimatedProcess accepts it")
        if len(a) == 1:
            raise NotImplementedError(f"{fname}: HiddenMarkovProcess[proc] (removing silent states) is not "
                                      "supported")
        if len(a) != 3:
            raise ValueError(f"{fname}: expected HiddenMarkovProcess[i0 or p0, m, em or {{dist1, ...}}]")
        m = P._matrix(a[1], fname)
        n = m.rows
        if m.cols != n:
            raise ValueError(f"{fname}: the transition matrix must be square")
        P.Markov._check_stochastic(m, fname)
        self.n, self.m = n, m
        e = a[2]
        if is_list(e) and e.args and all(W.is_dist(d) or _sym(d, "None") for d in e.args):
            if len(e.args) != n:
                raise ValueError(f"{fname}: give one emission distribution per hidden state ({n})")
            if any(_sym(d, "None") for d in e.args):
                raise NotImplementedError(f"{fname}: silent states (None emissions) are not supported")
            self.dists = list(e.args)
            dds = [W.dist_of(d, fname) for d in self.dists]
            kinds = {bool(getattr(dd, "discrete", False)) for dd in dds}
            if len(kinds) > 1:
                raise ValueError(f"{fname}: the emission distributions must be all discrete or all continuous")
            for d in self.dists:
                if W.is_dist(d) and d.head in ("MultinormalDistribution", "BinormalDistribution",
                                               "MultivariateTDistribution", "DirichletDistribution",
                                               "MultinomialDistribution", "ProductDistribution"):
                    raise NotImplementedError(f"{fname}: multivariate emissions are not supported")
            self.em = None
            emp = []
        else:
            em = P._matrix(e, fname, "emission matrix")
            if em.rows != n:
                raise ValueError(f"{fname}: the emission matrix needs one row per hidden state ({n})")
            for i in range(n):
                row = list(em.row(i))
                for v in row:
                    if P._num(v) and not (v.is_real and 0 <= v <= 1):
                        raise ValueError(f"{fname}: emission probabilities must be between 0 and 1")
                if all(P._num(v) for v in row):
                    s = sum(row)
                    if P._close(s, 0):
                        raise NotImplementedError(f"{fname}: silent states (zero emission rows) are not supported")
                    if not P._close(s, 1):
                        raise ValueError(f"{fname}: the rows of the emission matrix must sum to 1")
            self.em, self.dists = em, None
            emp = list(em)
        p0 = a[0]
        if is_list(p0):
            pv = P._plist(p0, fname, "initial probability list")
            if len(pv) != n:
                raise ValueError(f"{fname}: the initial probability vector must have length {n}")
            for v in pv:
                P._prob(v, fname, "each initial probability")
            if all(P._num(v) for v in pv) and not P._close(sum(pv), 1):
                raise ValueError(f"{fname}: the initial probabilities must sum to 1")
            self.p0 = sp.Matrix([pv])
        elif _int(p0):
            if not 1 <= p0 <= n:
                raise ValueError(f"{fname}: the initial state must be between 1 and {n}")
            self.p0 = sp.Matrix([[1 if j == p0 - 1 else 0 for j in range(n)]])
        elif isinstance(p0, Symbol):
            raise Unevaluated()
        else:
            raise TypeError(f"{fname}: the initial state must be an integer or a probability vector")
        self.p = list(self.p0) + list(self.m) + emp

    def check_time(self, t, fname):
        P._int_time(t, fname, 0)

    def numeric(self):
        ok = P._numeric_all(self.p)
        if self.dists is not None:
            ok = ok and not any(isinstance(v, Symbol) for d in self.dists for v in d.args)
        return ok

    def state_probs(self, t):
        if not P._num(t):
            raise Unevaluated()
        pt = self.p0 * self.m ** int(t)
        return [sp.simplify(v) if not P._inexact(v) else v for v in pt]

    def slice(self, t):
        pt = self.state_probs(t)
        return self._mix(pt)

    def _mix(self, pt):
        if self.em is not None:
            q = (sp.Matrix([pt]) * self.em)
            q = [sp.simplify(v) if not P._inexact(v) else v for v in q]
            return Expr("EmpiricalDistribution", rule(R(q), L(*range(1, self.em.cols + 1))))
        return Expr("MixtureDistribution", R(pt), L(*self.dists))

    def stationary_distribution(self, fname):
        mk = P.Markov(Expr("DiscreteMarkovProcess", R(list(self.p0)), R(self.m)), fname)
        d = mk.stationary_dist()
        pi = [S(w) for w in d.args[0].args[0].args]
        return self._mix(pi)

    def path(self, rng, times):
        if not self.numeric():
            raise Unevaluated()
        tmax = max(times)
        Pm = np.array([[float(v) for v in self.m.row(i)] for i in range(self.n)])
        p0 = np.array([float(v) for v in self.p0])
        st = [int(rng.choice(self.n, p=p0 / p0.sum()))]
        for _ in range(tmax):
            row = Pm[st[-1]]
            st.append(int(rng.choice(self.n, p=row / row.sum())))
        out = []
        prng = W._rng()
        for t in times:
            s = st[t]
            if self.em is not None:
                row = np.array([float(v) for v in self.em.row(s)])
                out.append(int(rng.choice(len(row), p=row / row.sum())) + 1)
            else:
                v = W.dist_of(self.dists[s], "RandomFunction").sample(prng)
                out.append(v)
        return out

    # numeric emission likelihoods
    def emission_matrix(self, ys, fname):
        """n_obs x n_states array of emission probabilities / densities (1 for missing values)"""
        out = np.ones((len(ys), self.n))
        if self.em is not None:
            E = np.array([[float(v) for v in self.em.row(i)] for i in range(self.n)])
            for t, y in enumerate(ys):
                if y is None:
                    continue
                if isinstance(y, sp.Basic) and y.is_number and float(y) == int(float(y)):
                    y = int(float(y))
                if not (_int(y) and 1 <= y <= E.shape[1]):
                    raise ValueError(f"{fname}: emissions must be integers between 1 and {E.shape[1]}, not {y}")
                out[t] = E[:, y - 1]
            return out
        dds = [W.dist_of(d, fname) for d in self.dists]
        cache = {}
        for t, y in enumerate(ys):
            if y is None:
                continue
            key = S(y)
            if key not in cache:
                cache[key] = [float(S(W.pdf_value(dd, key))) for dd in dds]
            out[t] = cache[key]
        return out


def _register_hmm():
    P.CLASSES["HiddenMarkovProcess"] = HMM


_register_hmm()


@fn("HiddenMarkovProcess", area=AREA, status="added",
    notes="HiddenMarkovProcess[i0 | p0, m, em] (emission matrix, emissions 1..r) and HiddenMarkovProcess[i0 | p0, "
          "m, {dist1, ...}] (univariate emission distributions, all discrete or all continuous): a discrete-time "
          "process whose hidden state starts from p0 at time 0 and moves with the transition matrix m, emitting "
          "y[t] ~ dist_(x[t]). Slices: EmpiricalDistribution[(p0.m^t).em -> Range[r]] or MixtureDistribution["
          "p0.m^t, dists]; StationaryDistribution uses the stationary distribution of m; RandomFunction simulates "
          "the emissions; LogLikelihood[hmm, data] is the forward algorithm (exact rationals for exact parameters "
          "and an emission matrix, else machine numbers; Missing values are skipped); EstimatedProcess[data, "
          "HiddenMarkovProcess[n, s | \"Gaussian\"]] fits by Baum-Welch. Silent states, HiddenMarkovProcess[proc], "
          "multivariate emissions and the IncludeHiddenStates method raise NotImplementedError; invalid matrices "
          "ValueError. Check: forward-algorithm likelihoods against brute-force enumeration of all state paths "
          "(tests) and simulated frequencies against the slice distribution.")
def _hmm_ctor(*args: Expr) -> Expr:
    del P._CONDS[:]
    HMM(Expr("HiddenMarkovProcess", *args), "HiddenMarkovProcess")
    raise Unevaluated()


def _hmm_of(x, fname):
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not (isinstance(x, Expr) and x.head == "HiddenMarkovProcess"):
        raise TypeError(f"{fname}: {x} is not a HiddenMarkovProcess")
    pr = HMM(x, fname)
    if not pr.numeric():
        raise Unevaluated()
    return pr


def _emission_paths(data, fname):
    """list of (times or None, values (None for missing), kind) from a list, list of lists, TimeSeries or
    TemporalData"""
    if isinstance(data, Symbol):
        raise Unevaluated()
    if isinstance(data, Expr) and data.head in ("TimeSeries", "EventSeries"):
        pairs, _ = T.pairs_of(data, fname)
        return [([t for t, _ in pairs], [_val(v, fname) for _, v in pairs], data.head)]
    if isinstance(data, Expr) and data.head == "TemporalData":
        paths, _ = T.td_paths(data)
        return [([t for t, _ in p], [_val(v, fname) for _, v in p], "TemporalData") for p in paths]
    if is_list(data):
        if data.args and all(is_list(p) for p in data.args) and not all(
                len(p.args) == 2 and _isnum(p.args[0]) for p in data.args):
            return [(None, [_val(v, fname) for v in p.args], "List") for p in data.args]
        return [(None, [_val(v, fname) for v in data.args], "List")]
    raise TypeError(f"{fname}: the data must be a list of emissions, a TimeSeries or TemporalData")


def _val(v, fname):
    if _missing(v):
        return None
    if isinstance(v, Symbol):
        raise Unevaluated()
    if _int(v):
        return v
    if not _isnum(v):
        raise TypeError(f"{fname}: emissions must be numbers, not {v}")
    s = S(v)
    return int(s) if s.is_Integer else s


def _forward(pr, ys, fname):
    """log-likelihood by the scaled forward algorithm"""
    B = pr.emission_matrix(ys, fname)
    A = np.array([[float(v) for v in pr.m.row(i)] for i in range(pr.n)])
    a = np.array([float(v) for v in pr.p0]) * B[0]
    ll = 0.0
    for t in range(len(ys)):
        if t > 0:
            a = (a @ A) * B[t]
        s = a.sum()
        if s <= 0:
            return -math.inf
        ll += math.log(s)
        a = a / s
    return ll


def _forward_exact(pr, ys):
    a = [pr.p0[j] * (pr.em[j, ys[0] - 1] if ys[0] is not None else 1) for j in range(pr.n)]
    for t in range(1, len(ys)):
        a = [sum(a[i] * pr.m[i, j] for i in range(pr.n)) * (pr.em[j, ys[t] - 1] if ys[t] is not None else 1)
             for j in range(pr.n)]
    return sp.nsimplify(sum(a))


def hmm_loglikelihood(proc, data):
    fname = "LogLikelihood"
    pr = _hmm_of(proc, fname)
    paths = _emission_paths(data, fname)
    exact = pr.em is not None and not P._inexact(*pr.p) and all(
        y is None or _int(y) for _, ys, _ in paths for y in ys)
    if exact:
        tot = sp.Integer(0)
        for _, ys, _ in paths:
            for y in ys:
                if y is not None and not 1 <= y <= pr.em.cols:
                    raise ValueError(f"{fname}: emissions must be integers between 1 and {pr.em.cols}")
            lik = _forward_exact(pr, ys)
            if lik == 0:
                return R(-sp.oo)
            tot += sp.log(lik)
        return R(sp.expand_log(tot, force=True))
    tot = 0.0
    for _, ys, _ in paths:
        tot += _forward(pr, ys, fname)
    return tot if math.isfinite(tot) else R(-sp.oo)


SM.PROCESS_LOGLIKELIHOOD["HiddenMarkovProcess"] = hmm_loglikelihood


def _viterbi(pr, ys, fname):
    B = pr.emission_matrix(ys, fname)
    with np.errstate(divide="ignore"):
        lA = np.log(np.array([[float(v) for v in pr.m.row(i)] for i in range(pr.n)]))
        lB = np.log(B)
        d = np.log(np.array([float(v) for v in pr.p0])) + lB[0]
    back = []
    for t in range(1, len(ys)):
        sc = d[:, None] + lA
        back.append(np.argmax(sc, axis=0))
        d = np.max(sc, axis=0) + lB[t]
    if not np.isfinite(np.max(d)):
        raise ValueError(f"{fname}: the emissions have zero probability under the model")
    s = [int(np.argmax(d))]
    for bp in reversed(back):
        s.append(int(bp[s[-1]]))
    return [v + 1 for v in reversed(s)]


def _posterior(pr, ys, fname):
    B = pr.emission_matrix(ys, fname)
    A = np.array([[float(v) for v in pr.m.row(i)] for i in range(pr.n)])
    n = len(ys)
    al = np.zeros((n, pr.n))
    c = np.zeros(n)
    a = np.array([float(v) for v in pr.p0]) * B[0]
    for t in range(n):
        if t > 0:
            a = (al[t - 1] @ A) * B[t]
        c[t] = a.sum()
        if c[t] <= 0:
            raise ValueError(f"{fname}: the emissions have zero probability under the model")
        al[t] = a / c[t]
    be = np.ones((n, pr.n))
    for t in range(n - 2, -1, -1):
        be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be
    return g / g.sum(axis=1, keepdims=True)


@fn("FindHiddenMarkovStates", area=AREA, status="added",
    notes="FindHiddenMarkovStates[data, hmm] is Viterbi decoding (the most probable state path, log-space dynamic "
          "programming, ties to the lower state); FindHiddenMarkovStates[data, hmm, \"PosteriorDecoding\"] gives "
          "the most probable state at each time from the forward-backward posteriors (Rabiner 1989). The first "
          "emission is at time 0 with state distribution p0. Data: a list of emissions (returns a list), a "
          "TimeSeries (returns a TimeSeries of states), TemporalData or a list of paths (one state path each); "
          "Missing values contribute no emission factor. Emissions with zero probability under the model raise "
          "ValueError; other criteria ValueError. Check: brute-force maximisation over all state paths for small "
          "models (tests) and the Wolfram documentation example.")
def _find_hmm_states(data: Expr, proc: Expr, crit: String = None) -> Expr:
    fname = "FindHiddenMarkovStates"
    pr = _hmm_of(proc, fname)
    c = "ViterbiDecoding" if crit is None else crit
    if c not in ("ViterbiDecoding", "PosteriorDecoding"):
        raise ValueError(f'{fname}: the criterion must be "ViterbiDecoding" or "PosteriorDecoding"')
    paths = _emission_paths(data, fname)
    outs = []
    for ts, ys, kind in paths:
        if not ys:
            raise ValueError(f"{fname}: the data must not be empty")
        if c == "ViterbiDecoding":
            st = _viterbi(pr, ys, fname)
        else:
            st = [int(np.argmax(r)) + 1 for r in _posterior(pr, ys, fname)]
        outs.append((ts, st, kind))
    if len(outs) == 1 and outs[0][2] == "List" and not (is_list(data) and data.args and is_list(data.args[0])):
        return L(*outs[0][1])
    if outs[0][2] in ("TimeSeries", "EventSeries"):
        return T.make_ts(list(zip(outs[0][0], outs[0][1])), None, outs[0][2])
    if outs[0][2] == "TemporalData":
        return T.make_td([list(zip(ts, st)) for ts, st, _ in outs])
    return L(*(L(*st) for _, st, _ in outs))


def _baum_welch(data, proc, est, fname, template_ok):
    """EstimatedProcess / FindProcessParameters hook for HiddenMarkovProcess[n, edist]"""
    a = list(proc.args)
    if len(a) != 2:
        raise NotImplementedError(f"{fname}: estimation needs the template HiddenMarkovProcess[n, edist]")
    if est not in ("Automatic", "BaumWelch"):
        raise NotImplementedError(f"{fname}: ProcessEstimator -> {est!r} is not supported for HiddenMarkovProcess "
                                  "(Automatic or \"BaumWelch\")")
    n, ed = a
    if not (_int(n) and n >= 1):
        raise ValueError(f"{fname}: the number of hidden states must be a positive integer")
    if fname == "FindProcessParameters":
        raise NotImplementedError(f"{fname}: use EstimatedProcess for hidden Markov processes")
    gaussian = ed == "Gaussian" or (isinstance(ed, Expr) and ed.head == "NormalDistribution" and
                                     all(isinstance(v, Symbol) for v in ed.args))
    if not gaussian and not (_int(ed) and ed >= 1):
        raise NotImplementedError(f"{fname}: emission templates other than s (discrete) and \"Gaussian\" are not "
                                  "supported")
    paths = _emission_paths(data, fname)
    seqs = [ys for _, ys, _ in paths]
    if any(y is None for ys in seqs for y in ys):
        raise NotImplementedError(f"{fname}: Missing values are not supported in estimation")
    allv = np.array([float(y) for ys in seqs for y in ys])
    if len(allv) < 2 * n:
        raise ValueError(f"{fname}: too few observations for {n} hidden states")
    if not gaussian:
        if not all(_int(y) and 1 <= y <= ed for ys in seqs for y in ys):
            raise ValueError(f"{fname}: discrete emissions must be integers between 1 and {ed}")
    # initialisation (stated): sorted-quantile state clustering, sticky transitions
    order = np.argsort(allv, kind="stable")
    lab = np.empty(len(allv), dtype=int)
    for k, chunk in enumerate(np.array_split(order, n)):
        lab[chunk] = k
    A = np.full((n, n), 0.3 / (n - 1) if n > 1 else 1.0)
    np.fill_diagonal(A, 0.7 if n > 1 else 1.0)
    pi = np.full(n, 1.0 / n)
    if gaussian:
        mu = np.array([allv[lab == k].mean() for k in range(n)])
        sd = np.array([max(allv[lab == k].std(), 1e-3 * (allv.std() or 1.0)) for k in range(n)])
    else:
        E = np.ones((n, ed))
        for v, k in zip(allv, lab):
            E[k, int(v) - 1] += 1
        E /= E.sum(axis=1, keepdims=True)
    prev = -math.inf
    for it in range(2000):
        An = np.zeros((n, n))
        pin = np.zeros(n)
        gsum = np.zeros(n)
        if gaussian:
            s1, s2 = np.zeros(n), np.zeros(n)
        else:
            En = np.zeros((n, ed))
        ll = 0.0
        for ys in seqs:
            y = np.array([float(v) for v in ys])
            if gaussian:
                B = np.exp(-0.5 * ((y[:, None] - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))
            else:
                B = E[:, y.astype(int) - 1].T
            m_ = len(y)
            al = np.zeros((m_, n))
            c = np.zeros(m_)
            for t in range(m_):
                a_ = (pi if t == 0 else al[t - 1] @ A) * B[t]
                c[t] = a_.sum()
                if c[t] <= 0 or not np.isfinite(c[t]):
                    raise ArithmeticError(f"{fname}: Baum-Welch broke down (zero likelihood)")
                al[t] = a_ / c[t]
            be = np.ones((m_, n))
            for t in range(m_ - 2, -1, -1):
                be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
            g = al * be
            g /= g.sum(axis=1, keepdims=True)
            for t in range(m_ - 1):
                xi = al[t][:, None] * A * (B[t + 1] * be[t + 1])[None, :] / c[t + 1]
                An += xi
            pin += g[0]
            gsum += g.sum(axis=0)
            if gaussian:
                s1 += g.T @ y
                s2 += g.T @ (y ** 2)
            else:
                for t in range(m_):
                    En[:, int(y[t]) - 1] += g[t]
            ll += float(np.sum(np.log(c)))
        A = An / An.sum(axis=1, keepdims=True)
        pi = pin / len(seqs)
        if gaussian:
            mu = s1 / gsum
            sd = np.sqrt(np.maximum(s2 / gsum - mu ** 2, 1e-12))
        else:
            E = En / En.sum(axis=1, keepdims=True)
        if ll - prev < 1e-10 * max(1.0, abs(ll)):
            break
        prev = ll
    else:
        raise ArithmeticError(f"{fname}: Baum-Welch did not converge in 2000 iterations")
    p0 = L(*(float(v) for v in pi))
    m = L(*(L(*(float(v) for v in r)) for r in A))
    if gaussian:
        em = L(*(Expr("NormalDistribution", float(a_), float(b_)) for a_, b_ in zip(mu, sd)))
    else:
        em = L(*(L(*(float(v) for v in r)) for r in E))
    return {}, Expr("HiddenMarkovProcess", p0, m, em)


P.ESTIMATE_HOOKS["HiddenMarkovProcess"] = _baum_welch


# ─────────────────────────── ARMA maximum likelihood ───────────────────────────

def _pacf_to_coef(u):
    """unconstrained u -> coefficients of a stationary AR polynomial (Jones 1980 / Monahan 1984)"""
    r = np.tanh(np.asarray(u, dtype=float))
    phi = np.zeros(0)
    for k, rk in enumerate(r):
        new = np.empty(k + 1)
        new[:k] = phi - rk * phi[::-1]
        new[k] = rk
        phi = new
    return phi


def _coef_to_pacf(phi):
    phi = np.array(phi, dtype=float)
    p = len(phi)
    r = np.zeros(p)
    for k in range(p - 1, -1, -1):
        rk = phi[k]
        if abs(rk) >= 1:
            raise ValueError
        r[k] = rk
        if k:
            phi = (phi[:k] + rk * phi[:k][::-1]) / (1 - rk ** 2)
    return np.arctanh(np.clip(r, -0.999999, 0.999999))


def arma_kalman(y, phi, theta):
    """exact Gaussian ARMA likelihood pieces (zero mean): innovations v, variances F (scale 1)"""
    p, q = len(phi), len(theta)
    r = max(p, q + 1)
    Tm = np.zeros((r, r))
    Tm[:p, 0] = phi
    if r > 1:
        Tm[:-1, 1:] = np.eye(r - 1)
    Rv = np.zeros(r)
    Rv[0] = 1
    Rv[1:q + 1] = theta
    Q = np.outer(Rv, Rv)
    M = np.eye(r * r) - np.kron(Tm, Tm)
    Pm = np.linalg.solve(M, Q.reshape(-1)).reshape(r, r)
    a = np.zeros(r)
    n = len(y)
    v = np.zeros(n)
    F = np.zeros(n)
    for t in range(n):
        v[t] = y[t] - a[0]
        F[t] = Pm[0, 0]
        Kg = Tm @ Pm[:, 0] / F[t]
        a = Tm @ a + Kg * v[t]
        Pm = Tm @ Pm @ Tm.T + Q - np.outer(Kg, Kg) * F[t]
    return v, F


def _nelder_mead(f, x0, tol=1e-10, maxit=None):
    x0 = np.asarray(x0, dtype=float)
    d = len(x0)
    if d == 0:
        return x0, f(x0)
    maxit = maxit or 4000 * d
    pts = [x0] + [x0 + np.eye(d)[i] * (0.25 if x0[i] == 0 else 0.25 * max(1.0, abs(x0[i]))) for i in range(d)]
    vals = [f(p) for p in pts]
    for _ in range(maxit):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) <= tol * (abs(vals[0]) + 1e-12) and \
                max(np.max(np.abs(p - pts[0])) for p in pts[1:]) <= 1e-9:
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1])
        fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1])
            fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c) if fr >= vals[-1] else c + 0.5 * (xr - c)
            fc = f(xc)
            if fc < min(fr, vals[-1]):
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0]] + [pts[0] + 0.5 * (p - pts[0]) for p in pts[1:]]
                vals = [vals[0]] + [f(p) for p in pts[1:]]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def arma_ml(w, p, q, const):
    """exact Gaussian ML of ARMA(p, q) (+ mean): returns dict(phi, theta, mu, sigma2, ll, v, F)"""
    w = np.asarray(w, dtype=float)
    n = len(w)

    def unpack(x):
        phi = _pacf_to_coef(x[:p])
        theta = -_pacf_to_coef(x[p:p + q])
        mu = x[p + q] if const else 0.0
        return phi, theta, mu

    def negll(x):
        phi, theta, mu = unpack(x)
        try:
            v, F = arma_kalman(w - mu, phi, theta)
        except np.linalg.LinAlgError:
            return 1e300
        if np.any(F <= 0) or not np.all(np.isfinite(v)):
            return 1e300
        s2 = float(np.sum(v ** 2 / F)) / n
        if s2 <= 0:
            return 1e300
        return 0.5 * n * (math.log(2 * math.pi * s2) + 1) + 0.5 * float(np.sum(np.log(F)))
    starts = [np.zeros(p + q + (1 if const else 0))]
    if const:
        starts[0][-1] = float(np.mean(w))
    for sgn in (0.5, -0.5):
        s = starts[0].copy()
        s[:p + q] = sgn
        starts.append(s)
    best = None
    for s in starts:
        x, fv = _nelder_mead(negll, s)
        x, fv = _nelder_mead(negll, x)
        if best is None or fv < best[1]:
            best = (x, fv)
    x, fv = best
    if fv >= 1e299:
        raise ArithmeticError("TimeSeriesModelFit: the likelihood could not be evaluated")
    phi, theta, mu = unpack(x)
    v, F = arma_kalman(w - mu, phi, theta)
    s2 = float(np.sum(v ** 2 / F)) / n
    return dict(phi=phi, theta=theta, mu=mu, sigma2=s2, ll=-fv, v=v, F=F, p=p, q=q, const=const)


def _adf_d(x, fname):
    """integration order by sequential ADF tests (Dickey-Fuller t with a constant, MacKinnon p-values; difference
    while p >= 0.05, at most twice -- a stated choice)"""
    w = np.asarray(x, dtype=float)
    for d in range(2):
        if len(w) < 8 or _adf_pvalue(w) < 0.05:
            return d
        w = np.diff(w)
    return 2


def _adf_pvalue(w):
    v = ev(Expr("UnitRootTest", L(*(float(a) for a in w)), "Constant", "DickeyFullerT"))
    return float(S(v))


# ─────────────────────────── TimeSeriesModelFit ───────────────────────────

_FAMILIES = ("AR", "MA", "ARMA", "ARIMA")
_CRITERIA = ("AIC", "AICc", "BIC", "SBC")


def _orders(v, fname, auto):
    if _sym(v, "Automatic"):
        return list(auto)
    if _int(v) and v >= 0:
        return [v]
    if is_list(v) and v.args and all(_int(a) and a >= 0 for a in v.args):
        return list(v.args)
    if isinstance(v, Expr) and v.head == "Span" and all(_int(a) for a in v.args) and len(v.args) in (2, 3):
        st = v.args[2] if len(v.args) == 3 else 1
        return list(range(v.args[0], v.args[1] + 1, st))
    raise ValueError(f"{fname}: invalid order specification {v}")


def _tsm_spec(mspec, fname):
    """(family, [(p, d, q)], d_auto)"""
    if mspec is None or _sym(mspec, "Automatic"):
        return "Automatic", None
    fam, par = (mspec, None) if isinstance(mspec, str) else (None, None)
    if is_list(mspec) and len(mspec.args) == 2 and isinstance(mspec.args[0], str):
        fam, par = mspec.args
    if fam is None:
        raise TypeError(f"{fname}: invalid model specification {mspec}")
    if fam in ("SARMA", "SARIMA", "ARCH", "GARCH"):
        raise NotImplementedError(f'{fname}: the "{fam}" family is not supported (AR, MA, ARMA, ARIMA)')
    if fam not in _FAMILIES:
        raise ValueError(f'{fname}: unknown model family "{fam}"')
    return fam, par


def _candidates(fam, par, fname):
    """list of (p, d, q) with d None for 'choose by ADF'"""
    A = Symbol("Automatic")
    if fam == "AR":
        ps = _orders(par.args[0] if is_list(par) and len(par.args) == 1 else (par if par is not None else A), fname,
                     range(0, 6))
        return [(p, 0, 0) for p in ps]
    if fam == "MA":
        qs = _orders(par.args[0] if is_list(par) and len(par.args) == 1 else (par if par is not None else A), fname,
                     range(0, 6))
        return [(0, 0, q) for q in qs]
    if fam == "ARMA":
        if par is None:
            par = L(A, A)
        if not (is_list(par) and len(par.args) == 2):
            raise ValueError(f"{fname}: ARMA orders must be {{p, q}}")
        return [(p, 0, q) for p in _orders(par.args[0], fname, range(0, 4))
                for q in _orders(par.args[1], fname, range(0, 4))]
    if par is None:
        par = L(A, A, A)
    if not (is_list(par) and len(par.args) == 3):
        raise ValueError(f"{fname}: ARIMA orders must be {{p, d, q}}")
    dv = par.args[1]
    if _sym(dv, "Automatic"):
        ds = [None]
    else:
        ds = _orders(dv, fname, [0])
        if len(ds) > 1:
            raise NotImplementedError(f"{fname}: selecting among several integration orders is not supported "
                                      "(information criteria are not comparable across differencing orders)")
    return [(p, ds[0], q) for p in _orders(par.args[0], fname, range(0, 4))
            for q in _orders(par.args[2], fname, range(0, 4))]


def _crit_value(fit, crit, n):
    k = fit["p"] + fit["q"] + (1 if fit["const"] else 0) + 1
    ll = fit["ll"]
    if crit == "AIC":
        return -2 * ll + 2 * k
    if crit == "AICc":
        return -2 * ll + 2 * k + (2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else math.inf)
    return -2 * ll + k * math.log(n)


def _process_expr(fit, d, inexact=True):
    phi = [float(v) for v in fit["phi"]]
    theta = [float(v) for v in fit["theta"]]
    v = float(fit["sigma2"])
    c = float(fit["mu"]) * (1 - sum(phi)) if fit["const"] else None
    cst = [c] if c is not None else []
    if d > 0:
        return Expr("ARIMAProcess", *cst, L(*phi), d, L(*theta), v)
    if fit["q"] == 0 and fit["p"] > 0:
        return Expr("ARProcess", *cst, L(*phi), v)
    if fit["p"] == 0 and fit["q"] > 0:
        return Expr("MAProcess", *cst, L(*theta), v)
    if fit["p"] == 0 and fit["q"] == 0:
        return Expr("ARMAProcess", *cst, L(), L(), v)
    return Expr("ARMAProcess", *cst, L(*phi), L(*theta), v)


TSMF_OPTS = ("ConfidenceLevel", "IncludeConstantBasis", "Method", "WorkingPrecision")


def _tsmf_data(data, fname):
    paths = P._series_data(data, fname)
    if len(paths) != 1:
        raise NotImplementedError(f"{fname}: several paths are not supported")
    path = paths[0]
    ts = [t for t, _ in path]
    xs = [x for _, x in path]
    if any(not P._num(x) for x in xs):
        raise Unevaluated()
    if len(ts) > 2:
        dt = ts[1] - ts[0]
        for a, b in zip(ts, ts[1:]):
            if sp.simplify(b - a - dt) != 0:
                raise NotImplementedError(f"{fname}: irregularly sampled data is not supported")
    return path


@fn("TimeSeriesModelFit", area=AREA, status="added",
    notes="TimeSeriesModelFit[data, mspec] fits AR, MA, ARMA and ARIMA models by exact Gaussian maximum "
          "likelihood (Kalman filter on the stationary state-space form, Harvey 1989; stationarity and "
          "invertibility enforced by the Jones 1980 partial-autocorrelation reparametrisation; Nelder-Mead from "
          "three starting points) and returns a TimeSeriesModel. mspec: Automatic (= {\"ARIMA\", {Automatic, "
          "Automatic, Automatic}}), \"family\", {\"AR\", p}, {\"MA\", q}, {\"ARMA\", {p, q}}, {\"ARIMA\", {p, d, q}} "
          "with orders n, lists, spans or Automatic. Automatic orders (stated choices; Wolfram's are undocumented): "
          "p or q in 0..5 for AR / MA, 0..3 each for ARMA / ARIMA; an Automatic integration order is chosen by "
          "sequential augmented Dickey-Fuller tests (difference while the DickeyFullerT p-value with a constant is "
          ">= 0.05, at most twice). All candidates are fitted (grid search) and ranked by the selection criterion: "
          "Method -> {\"GridSearch\" | Automatic, \"SelectionCriterion\" -> \"AIC\" (default) | \"AICc\" | \"BIC\" | "
          "\"SBC\"}; AIC = -2 LogLikelihood + 2 k with k counting the coefficients, the constant and the noise "
          "variance; the reported noise variance is the ML estimate. IncludeConstantBasis Automatic includes a "
          "constant when d = 0 (documented). SARMA, SARIMA, ARCH, GARCH, several integration orders in one "
          "search, the \"Stepwise\" method, irregular or multi-path data raise NotImplementedError. Check: "
          "estimates, log-likelihoods and AIC compared with statsmodels ARIMA(order, trend) exact ML on the same "
          "data (tests).")
def _tsmf(data: Expr, mspec: String | ListOf[Expr] | Expr = None, *rest: Option) -> Expr:
    fname = "TimeSeriesModelFit"
    rest = list(rest)
    if mspec is not None and SM._is_rule(mspec):
        rest.insert(0, mspec)
        mspec = None
    _, o = SM.opts(rest, TSMF_OPTS, fname)
    if isinstance(data, Symbol):
        raise Unevaluated()
    crit = "AIC"
    meth = o.get("Method", Symbol("Automatic"))
    if not _sym(meth, "Automatic"):
        items = list(meth.args) if is_list(meth) else [meth]
        for it in items:
            if it in ("GridSearch",) or _sym(it, "Automatic"):
                continue
            if it == "Stepwise":
                raise NotImplementedError(f'{fname}: the "Stepwise" method is not supported (grid search is used)')
            if isinstance(it, Expr) and it.head == "Rule" and it.args[0] == "SelectionCriterion":
                if it.args[1] not in _CRITERIA:
                    raise ValueError(f"{fname}: the selection criterion must be one of {', '.join(_CRITERIA)}")
                crit = it.args[1]
            else:
                raise ValueError(f"{fname}: invalid Method setting {it}")
    wp = o.get("WorkingPrecision", Symbol("Automatic"))
    if not _sym(wp, "Automatic", "MachinePrecision"):
        raise NotImplementedError(f"{fname}: WorkingPrecision is not supported")
    icb = o.get("IncludeConstantBasis", Symbol("Automatic"))
    if not (icb in (True, False) or _sym(icb, "Automatic")):
        raise ValueError(f"{fname}: IncludeConstantBasis must be True, False or Automatic")
    path = _tsmf_data(data, fname)
    x = np.array([float(S(v)) for _, v in path])
    fam, par = _tsm_spec(mspec, fname)
    if fam == "Automatic":
        fam, par = "ARIMA", None
    cands = _candidates(fam, par, fname)
    d_auto = None
    fits = []
    for p, d, q in cands:
        if d is None:
            if d_auto is None:
                d_auto = _adf_d(x, fname)
            d = d_auto
        w = np.diff(x, n=d) if d else x
        const = (d == 0) if _sym(icb, "Automatic") else bool(icb)
        nfree = p + q + (1 if const else 0) + 1
        if len(w) <= nfree + 1:
            continue
        try:
            fit = arma_ml(w, p, q, const)
        except (ArithmeticError, np.linalg.LinAlgError, ValueError):
            continue
        fit["d"] = d
        fit["n"] = len(w)
        fit["crit"] = _crit_value(fit, crit, len(w))
        fits.append(fit)
    if not fits:
        raise ValueError(f"{fname}: too few observations to fit any candidate model")
    fits.sort(key=lambda f: f["crit"])
    best = fits[0]
    fam_out = "ARIMA" if best["d"] > 0 else ("AR" if best["q"] == 0 and best["p"] > 0 else
                                            ("MA" if best["p"] == 0 and best["q"] > 0 else "ARMA"))
    resid = list(zip([t for t, _ in path][best["d"]:], best["v"]))
    a = Expr("Association",
             rule("BestFit", _process_expr(best, best["d"])),
             rule("ModelFamily", fam_out),
             rule("SelectionCriterion", crit),
             rule("CandidateModels", L(*(_process_expr(f, f["d"]) for f in fits))),
             rule("CandidateModelSelectionValues", L(*(float(f["crit"]) for f in fits))),
             rule("LogLikelihood", float(best["ll"])),
             rule("ErrorVariance", float(best["sigma2"])),
             rule("AIC", float(_crit_value(best, "AIC", best["n"]))),
             rule("AICc", float(_crit_value(best, "AICc", best["n"]))) if math.isfinite(
                 _crit_value(best, "AICc", best["n"])) else rule("AICc", Symbol("Infinity")),
             rule("BIC", float(_crit_value(best, "BIC", best["n"]))),
             rule("Residuals", L(*(L(t, float(v)) for t, v in resid))),
             rule("Standardized", L(*(float(v / math.sqrt(f * best["sigma2"])) for v, f in
                                      zip(best["v"], best["F"])))),
             rule("Data", L(*(L(t, v) for t, v in path))),
             rule("ConfidenceLevel", o.get("ConfidenceLevel", R(sp.Rational(95, 100)))))
    return Expr("TimeSeriesModel", a)


def _tsm_assoc(x):
    if isinstance(x, Expr) and x.head == "TimeSeriesModel" and len(x.args) == 1 and \
            isinstance(x.args[0], Expr) and x.args[0].head == "Association":
        return x.args[0]
    return None


def _get(a, k):
    for r in a.args:
        if isinstance(r, Expr) and r.head == "Rule" and r.args[0] == k:
            return r.args[1]
    raise ValueError(f'TimeSeriesModel: missing "{k}"')


_TSM_PROPS = ("BestFit", "BestFitParameters", "ErrorVariance", "FitResiduals", "StandardizedResiduals",
              "TemporalData", "CandidateModels", "CandidateModelSelectionValues", "ModelFamily",
              "SelectionCriterion", "AIC", "AICc", "BIC", "SBC", "LogLikelihood", "ACFValues", "PACFValues",
              "LjungBoxValues", "Properties")


def _acf(r, lags):
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    c0 = float(r @ r) / len(r)
    return [float(r[:len(r) - k] @ r[k:]) / len(r) / c0 for k in range(1, lags + 1)]


def _pacf(r, lags):
    rho = [1.0] + _acf(r, lags)
    out = []
    phi = []
    for k in range(1, lags + 1):
        if k == 1:
            a = rho[1]
            phi = [a]
        else:
            num = rho[k] - sum(phi[j] * rho[k - 1 - j] for j in range(k - 1))
            den = 1 - sum(phi[j] * rho[j + 1] for j in range(k - 1))
            a = num / den
            phi = [phi[j] - a * phi[k - 2 - j] for j in range(k - 1)] + [a]
        out.append(a)
    return out


def _tsm_property(a, prop, lagmax=10):
    if prop == "Properties":
        return L(*_TSM_PROPS)
    if prop in ("BestFit", "ModelFamily", "SelectionCriterion", "CandidateModels", "CandidateModelSelectionValues",
                "ErrorVariance", "AIC", "AICc", "BIC", "LogLikelihood"):
        return _get(a, prop)
    if prop == "SBC":
        return _get(a, "BIC")
    if prop == "BestFitParameters":
        return L(*_get(a, "BestFit").args)
    if prop == "FitResiduals":
        return T.make_ts([(r.args[0], r.args[1]) for r in _get(a, "Residuals").args])
    if prop == "StandardizedResiduals":
        res = _get(a, "Residuals").args
        return T.make_ts([(r.args[0], v) for r, v in zip(res, _get(a, "Standardized").args)])
    if prop == "TemporalData":
        return T.make_td([[(r.args[0], r.args[1]) for r in _get(a, "Data").args]])
    if prop in ("ACFValues", "PACFValues", "LjungBoxValues"):
        res = [float(S(r.args[1])) for r in _get(a, "Residuals").args]
        lags = min(lagmax, len(res) - 1)
        if prop == "ACFValues":
            return L(*(L(k, v) for k, v in zip(range(1, lags + 1), _acf(res, lags))))
        if prop == "PACFValues":
            return L(*(L(k, v) for k, v in zip(range(1, lags + 1), _pacf(res, lags))))
        n = len(res)
        rho = _acf(res, lags)
        out = []
        for h in range(1, lags + 1):
            Q = n * (n + 2) * sum(rho[k - 1] ** 2 / (n - k) for k in range(1, h + 1))
            out.append(L(h, float(SM.chi2_sf(Q, h))))
        return L(*out)
    if prop in ("ACFPlot", "PACFPlot", "LjungBoxPlot", "CandidateSelectionTable", "CandidateSelectionTableEntries",
                "ParameterTable", "ParameterTableEntries", "CovarianceMatrix", "InformationMatrix",
                "ParameterConfidenceIntervals", "ParameterStandardErrors", "PredictionLimits"):
        raise NotImplementedError(f'TimeSeriesModel: the property "{prop}" is not supported')
    raise ValueError(f'TimeSeriesModel: unknown property "{prop}"')


def tsm_forecast(a, ks):
    data = _get(a, "Data").args
    vals = L(*(r.args[1] for r in data))
    res = ev(Expr("TimeSeriesForecast", _get(a, "BestFit"), vals, L(L(*ks))))
    return [p.args[1] for p in res.args[0].args[0].args]


def _tsm_value(a, t):
    data = [(r.args[0], r.args[1]) for r in _get(a, "Data").args]
    if not _isnum(t):
        return UNH
    tv = S(t)
    last = S(data[-1][0])
    if tv <= last:
        return T.value_at(data, {}, t, "TimeSeriesModel", True)
    dt = S(data[1][0]) - S(data[0][0]) if len(data) > 1 else sp.Integer(1)
    k = sp.nsimplify((tv - last) / dt) if not P._inexact(tv) else (tv - last) / dt
    if not (abs(float(k) - round(float(k))) < 1e-9 and round(float(k)) >= 1):
        raise ValueError("TimeSeriesModel: forecasts are available at whole time steps after the data")
    return tsm_forecast(a, [int(round(float(k)))])[0]


@fn("TimeSeriesModel", area=AREA, status="added",
    notes="TimeSeriesModel[<|...|>] is returned by TimeSeriesModelFit. model[t] gives the data value for t in the "
          "data range (interpolated between samples) and the forecast (TimeSeriesForecast of the fitted process) "
          "at whole steps after it; model[\"prop\"] gives \"BestFit\", \"BestFitParameters\" (the process "
          "arguments), \"ErrorVariance\" (ML), \"FitResiduals\" and \"StandardizedResiduals\" (Kalman innovations, "
          "TimeSeries), \"TemporalData\", \"CandidateModels\" and \"CandidateModelSelectionValues\" (sorted), "
          "\"ModelFamily\", \"SelectionCriterion\", \"AIC\", \"AICc\", \"BIC\", \"SBC\" (= BIC), \"LogLikelihood\", "
          "\"ACFValues\", \"PACFValues\" and \"LjungBoxValues\" ({lag, value} for lags 1..10, or model[prop, "
          "\"LagMax\" -> m]) and \"Properties\"; Normal is not overloaded (use \"BestFit\"). Plots, tables, "
          "parameter covariances / standard errors and \"PredictionLimits\" raise NotImplementedError. Check: "
          "residual ACF and Ljung-Box values compared with statsmodels acf / acorr_ljungbox (tests).")
def _tsm(a: Expr) -> Expr:
    if not (isinstance(a, Expr) and a.head == "Association"):
        raise TypeError("TimeSeriesModel: expected TimeSeriesModel[<|...|>] (use TimeSeriesModelFit)")
    raise Unevaluated()


def _tsm_apply(head, args, ev_):
    a = _tsm_assoc(head)
    if a is None or not args:
        return UNH
    p = args[0]
    if isinstance(p, str):
        lag = 10
        for r in args[1:]:
            if isinstance(r, Expr) and r.head == "Rule" and r.args[0] == "LagMax" and _int(r.args[1]) and \
                    r.args[1] >= 1:
                lag = r.args[1]
            else:
                raise ValueError(f"TimeSeriesModel: invalid property option {r}")
        return _tsm_property(a, p, lag)
    if len(args) == 1 and is_list(p) and p.args and all(isinstance(q, str) for q in p.args):
        return L(*(_tsm_property(a, q) for q in p.args))
    if len(args) == 1:
        if is_list(p):
            return L(*(_tsm_value(a, t) for t in p.args))
        return _tsm_value(a, p)
    return UNH


KL.COMPOUND_HEAD_HOOKS["TimeSeriesModel"] = _tsm_apply


def _tsm_forecast_hook(model, spec, rest):
    a = _tsm_assoc(model)
    ks, scalar = P._steps(spec, "TimeSeriesForecast")
    preds = tsm_forecast(a, ks)
    if scalar:
        return preds[0]
    data = _get(a, "Data").args
    last = S(data[-1].args[0])
    dt = S(data[1].args[0]) - S(data[0].args[0]) if len(data) > 1 else sp.Integer(1)
    return T.make_td([[(R(last + k * dt), v) for k, v in zip(ks, preds)]])


P.TSM_FORECAST_HOOK = _tsm_forecast_hook


# ─────────────────────────── AdjustTimeSeriesForecast ───────────────────────────

def psi_weights(pr, n):
    full = P._polymul(P._ar_poly(pr.phi), pr.diff)
    A = [-c for c in full[1:]]
    th = list(pr.theta)
    psi = [sp.Integer(1)]
    for j in range(1, n + 1):
        v = (th[j - 1] if j <= len(th) else 0) + sum(A[i - 1] * psi[j - i] for i in range(1, min(j, len(A)) + 1))
        psi.append(v)
    return psi


def _fc_pairs(x, fname):
    """(times or None, values, kind, opts)"""
    if isinstance(x, Expr) and x.head in ("TimeSeries", "EventSeries"):
        pairs, o = T.pairs_of(x, fname)
        return [t for t, _ in pairs], [v for _, v in pairs], x.head
    if isinstance(x, Expr) and x.head == "TemporalData":
        paths, _ = T.td_paths(x)
        if len(paths) != 1:
            raise NotImplementedError(f"{fname}: several paths are not supported")
        return [t for t, _ in paths[0]], [v for _, v in paths[0]], "TemporalData"
    if is_list(x):
        if x.args and all(is_list(p) and len(p.args) == 2 for p in x.args):
            return [p.args[0] for p in x.args], [p.args[1] for p in x.args], "Pairs"
        return None, list(x.args), "List"
    if isinstance(x, Symbol):
        raise Unevaluated()
    raise TypeError(f"{fname}: expected a list, TimeSeries, EventSeries or TemporalData, not {x}")


@fn("AdjustTimeSeriesForecast", area=AREA, status="added",
    notes="AdjustTimeSeriesForecast[tproc, forecast, newdata] updates the forecast of an ARMA-family process "
          "(ARProcess, MAProcess, ARMAProcess, ARIMAProcess, SARMAProcess, SARIMAProcess or a TimeSeriesModel) "
          "with new observations by the Box-Jenkins updating formula x^_(t+1)(l) = x^_t(l + 1) + psi_l a_(t+1), "
          "applied sequentially (each new error a = new value - the forecast already updated by the earlier new "
          "values; psi the MA(infinity) weights of the full AR polynomial including differencing). Note: the "
          "Wolfram documentation example ARProcess[{1/3}, 1] with two new points shows errors taken against the "
          "unadjusted forecast instead; this implementation follows the textbook sequential formula, so results "
          "agree for one new point and can differ from Wolfram's for two or more. The new values replace the "
          "forecast at their times; the result has the forecast's form (list, TimeSeries, TemporalData). A vector "
          "forecast takes new data in order (timestamps ignored); timestamped forecasts align new data by "
          "timestamp (a list of new values starts at the first forecast time); new data outside the forecast "
          "times raise ValueError. Symbolic values are allowed. Check: Wolfram documentation examples "
          "(MAProcess[{1/10}, 1] -> {1.3, 0.03, 0.}; ARMAProcess[{.1}, {.2}, 1] with one symbolic point) and the "
          "AR(1) identity that the adjusted forecasts equal fresh forecasts from the new last value (tests).")
def _adjust_forecast(proc: Expr, forecast: Expr, newdata: Expr) -> Expr:
    fname = "AdjustTimeSeriesForecast"
    a = _tsm_assoc(proc)
    if a is not None:
        proc = _get(a, "BestFit")
    pr = P.proc_of(proc, fname)
    if not isinstance(pr, P.ARMA) or pr.frac is not None:
        raise TypeError(f"{fname}: {pr.name} is not an ARMA-type time series process")
    if not pr.numeric():
        raise Unevaluated()
    ft, fv, fkind = _fc_pairs(forecast, fname)
    nt, nv, nkind = _fc_pairs(newdata, fname)
    H = len(fv)
    if fkind == "List" or nt is None:
        idx = list(range(len(nv)))
    else:
        keys = [S(t) for t in ft]
        idx = []
        for t in nt:
            st = S(t)
            hit = [i for i, k in enumerate(keys) if sp.simplify(k - st) == 0]
            if not hit:
                raise ValueError(f"{fname}: the new data time {t} is not a forecast time")
            idx.append(hit[0])
    if any(i >= H for i in idx):
        raise ValueError(f"{fname}: more new data points than forecast steps")
    psi = psi_weights(pr, H)
    cur = [S(v) for v in fv]
    exact = not P._inexact(*cur, *[S(v) for v in nv], *pr.flat())
    for i, x in sorted(zip(idx, nv), key=lambda q: q[0]):
        e = S(x) - cur[i]
        for l in range(i + 1, H):
            cur[l] = cur[l] + psi[l - i] * e
        cur[i] = S(x)
    out = [R(sp.simplify(v) if exact else (sp.Float(float(v)) if v.is_number else sp.expand(v))) for v in cur]
    if fkind == "List":
        return L(*out)
    if fkind == "Pairs":
        return L(*(L(t, v) for t, v in zip(ft, out)))
    if fkind in ("TimeSeries", "EventSeries"):
        return T.make_ts(list(zip(ft, out)), None, fkind)
    return T.make_td([list(zip(ft, out))])


# ─────────────────────────── time / event series utilities ───────────────────────────

def _series_values(x, fname, heads=("TimeSeries", "EventSeries")):
    if isinstance(x, Symbol):
        raise Unevaluated()
    if not (isinstance(x, Expr) and x.head in heads):
        raise TypeError(f"{fname}: expected a {' or '.join(heads)}")
    pairs, _ = T.pairs_of(x, fname)
    if any(is_list(v) for _, v in pairs):
        raise NotImplementedError(f"{fname}: vector-valued (multi-component) series are not supported")
    return pairs


def _tabular(pairs, with_time):
    rows = L(*(L(t, v) if with_time else L(v) for t, v in pairs))
    keys = L("Timestamp", "Value") if with_time else L("Value")
    return ev(Expr("Tabular", rows, keys))


def _value_comp(c, fname):
    if c is None or _sym(c, "All") or c == "Value" or (is_list(c) and list(c.args) == ["Value"]):
        return True
    raise ValueError(f'{fname}: a series without named components has the single component "Value"')


@fn("TimeSeriesSummary", area=AREA, status="added",
    notes="TimeSeriesSummary[tes], [tes, coms], [tes, coms -> props] and [tes, {coms1 -> props1, ...}] for a "
          "scalar-valued TimeSeries or EventSeries: the TabularSummary of its single value component \"Value\" "
          "(same property names and defaults as TabularSummary). Multi-component (vector-valued) series raise "
          "NotImplementedError; component names other than \"Value\" / All ValueError. Check: equals "
          "TabularSummary of Tabular[values, {\"Value\"}] (tests).")
def _ts_summary(tes: Expr, spec: Expr = None) -> Expr:
    fname = "TimeSeriesSummary"
    pairs = _series_values(tes, fname)
    tab = _tabular(pairs, False)
    if spec is None:
        return ev(Expr("TabularSummary", tab))
    rules = [spec] if isinstance(spec, Expr) and spec.head == "Rule" else (
        list(spec.args) if is_list(spec) and spec.args and all(isinstance(r, Expr) and r.head == "Rule"
                                                               for r in spec.args) else None)
    if rules is None:
        _value_comp(spec, fname)
        return ev(Expr("TabularSummary", tab, "Value" if not _sym(spec, "All") else spec))
    out = []
    for r in rules:
        _value_comp(r.args[0], fname)
        out.append(Expr("Rule", L("Value"), r.args[1]))
    return ev(Expr("TabularSummary", tab, out[0] if len(out) == 1 else L(*out)))


@fn("TimeSeriesStructure", area=AREA, status="added",
    notes="TimeSeriesStructure[tes], [tes, comps], [tes, comps, props] for a scalar-valued TimeSeries or "
          "EventSeries: the TabularStructure of Tabular[{{t, v}, ...}, {\"Timestamp\", \"Value\"}] (structural "
          "information about the timestamps and the value component; comps and props as TabularStructure). "
          "Multi-component series raise NotImplementedError. Check: equals TabularStructure of that Tabular "
          "(tests).")
def _ts_structure(tes: Expr, comps: Expr = None, props: Expr = None) -> Expr:
    fname = "TimeSeriesStructure"
    pairs = _series_values(tes, fname)
    tab = _tabular(pairs, True)
    args = [tab] + ([comps] if comps is not None else []) + ([props] if props is not None else [])
    return ev(Expr("TabularStructure", *args))


def _apply_fn(f, v, t):
    """apply fun / cond: an association <|"Value" -> v, "Timestamp" -> t|> when it uses #Value / #Timestamp"""
    uses = not (_truth(ev(Expr("FreeQ", f, Expr("Slot", "Value")))) and
                _truth(ev(Expr("FreeQ", f, Expr("Slot", "Timestamp")))))
    arg = Expr("Association", rule("Value", v), rule("Timestamp", t)) if uses else v
    if isinstance(f, Symbol):
        f = f.name
    return ev(Expr(f, arg))


def _truth(x):
    return x is True or _sym(x, "True")


@fn("TimeSeriesEvents", area=AREA, status="added",
    notes="TimeSeriesEvents[ts, crit, fun, cond] treats a scalar numeric TimeSeries as the piecewise-linear "
          "function through its samples. Crossings (\"Crossing\", \"CrossingFromBelow\", \"CrossingFromAbove\", "
          "{crossing, value}; default value 0) are found where fun[value] - level changes sign between "
          "consecutive samples, at the linearly interpolated time (exact for exact data); a sample exactly at the "
          "level counts once when the neighbouring non-level samples lie on opposite sides. Extrema "
          "(\"LocalExtremum\", \"LocalMaximum\", \"LocalMinimum\") are strict interior sample extrema of fun[value] "
          "(a plateau counts once, at its first sample; end points are not events -- stated choices). The result "
          "is EventSeries[{{t, v}, ...}] with v the series value at the event time, or Missing[\"NotFound\"] when "
          "there are no events (an empty EventSeries is not representable here). fun and cond receive the value, "
          "or <|\"Value\" -> v, \"Timestamp\" -> t|> when they use #Value / #Timestamp. Date timestamps and "
          "multi-component series raise NotImplementedError. Check: hand-computed crossings and extrema (tests).")
def _ts_events(ts: Expr, crit: String | ListOf[Expr], fun: Expr = None, cond: Expr = None) -> Expr:
    fname = "TimeSeriesEvents"
    pairs = _series_values(ts, fname, ("TimeSeries",))
    if any(not _isnum(t) for t, _ in pairs) or any(not _isnum(v) for _, v in pairs):
        raise NotImplementedError(f"{fname}: only numeric times and values are supported")
    ts_ = [S(t) for t, _ in pairs]
    vs = [S(v) for _, v in pairs]
    gv = [S(_apply_fn(fun, R(v), R(t))) if fun is not None else v for t, v in zip(ts_, vs)]
    if any(not (isinstance(g, sp.Basic) and g.is_number) for g in gv):
        raise ValueError(f"{fname}: the function must give numbers")
    level = sp.Integer(0)
    c = crit
    if is_list(crit) and len(crit.args) == 2 and isinstance(crit.args[0], str):
        c = crit.args[0]
        if not _isnum(crit.args[1]):
            raise TypeError(f"{fname}: the crossing value must be a number")
        level = S(crit.args[1])
        if c not in ("Crossing", "CrossingFromBelow", "CrossingFromAbove"):
            raise ValueError(f"{fname}: {{crit, value}} needs a crossing criterion")
    events = []
    inexact = any(P._inexact(v) for v in ts_ + gv)
    if c in ("Crossing", "CrossingFromBelow", "CrossingFromAbove"):
        h = [g - level for g in gv]
        sgn = [0 if (abs(float(x)) == 0) else (1 if x > 0 else -1) for x in h]
        n = len(h)
        for i in range(n - 1):
            if sgn[i] != 0 and sgn[i + 1] != 0 and sgn[i] != sgn[i + 1]:
                frac = -h[i] / (h[i + 1] - h[i])
                t = ts_[i] + (ts_[i + 1] - ts_[i]) * frac
                v = vs[i] + (vs[i + 1] - vs[i]) * frac
                events.append((t, v, 1 if sgn[i] < 0 else -1))
        for i in range(n):
            if sgn[i] == 0:
                lo = next((sgn[j] for j in range(i - 1, -1, -1) if sgn[j] != 0), 0)
                hi = next((sgn[j] for j in range(i + 1, n) if sgn[j] != 0), 0)
                first_zero = i == 0 or sgn[i - 1] != 0
                if lo and hi and lo != hi and first_zero:
                    events.append((ts_[i], vs[i], 1 if lo < 0 else -1))
        if c == "CrossingFromBelow":
            events = [e for e in events if e[2] > 0]
        elif c == "CrossingFromAbove":
            events = [e for e in events if e[2] < 0]
    elif c in ("LocalExtremum", "LocalMaximum", "LocalMinimum"):
        n = len(gv)
        i = 1
        while i < n - 1:
            j = i
            while j + 1 < n and gv[j + 1] == gv[i]:
                j += 1
            if j < n - 1:
                if gv[i] > gv[i - 1] and gv[i] > gv[j + 1] and c != "LocalMinimum":
                    events.append((ts_[i], vs[i], 0))
                if gv[i] < gv[i - 1] and gv[i] < gv[j + 1] and c != "LocalMaximum":
                    events.append((ts_[i], vs[i], 0))
            i = j + 1
    else:
        raise ValueError(f"{fname}: unknown criterion {crit}")
    events.sort(key=lambda e: float(e[0]))
    if cond is not None:
        events = [e for e in events if _truth(_apply_fn(cond, R(e[1]), R(e[0])))]
    if not events:
        return Expr("Missing", "NotFound")
    fix = (lambda x: sp.Float(float(x))) if inexact else sp.nsimplify
    return T.make_ts([(R(fix(t)), R(fix(v))) for t, v, _ in events], None, "EventSeries")


@fn("EventSeriesAccumulate", area=AREA, status="added",
    notes="EventSeriesAccumulate[es] gives TimeSeries[{{t1, 1}, {t2, 2}, ...}] (cumulative event counts); "
          "EventSeriesAccumulate[es -> \"Value\"] (or {\"Value\"}) the cumulative sums of the values. "
          "Multi-component series raise NotImplementedError; non-EventSeries input TypeError. Check: exact "
          "(Accumulate of the values; tests).")
def _es_accumulate(es: Expr) -> Expr:
    fname = "EventSeriesAccumulate"
    comp = None
    if isinstance(es, Expr) and es.head == "Rule" and len(es.args) == 2:
        es, comp = es.args
        _value_comp(comp, fname)
    pairs = _series_values(es, fname, ("EventSeries",))
    if comp is None:
        return T.make_ts([(t, k + 1) for k, (t, _) in enumerate(pairs)])
    acc = ev(Expr("Accumulate", L(*(v for _, v in pairs))))
    return T.make_ts(list(zip([t for t, _ in pairs], acc.args)))


@fn("TimeObservationWindow", area=AREA, status="added",
    notes="TimeObservationWindow -> {min, max} or Interval[{min, max}] is an option of TimeSeries and EventSeries "
          "giving the observation window; it is validated (numeric, min <= max, all timestamps inside, else "
          "ValueError), stored on the object as Interval[{min, max}] and returned by "
          "series[\"TimeObservationWindow\"] (default Interval[{first time, last time}]). Derived series (shifts, "
          "windows, maps) do not carry it over. DateInterval windows raise NotImplementedError. Inert as an "
          "expression. Check: exact (tests).")
def _tow(*args: Expr) -> Expr:
    raise Unevaluated()


def tow_check(v, times, fname):
    """validated Interval[{a, b}] for a TimeObservationWindow setting"""
    if isinstance(v, Expr) and v.head == "DateInterval":
        raise NotImplementedError(f"{fname}: DateInterval observation windows are not supported")
    if isinstance(v, Expr) and v.head == "Interval" and len(v.args) == 1 and is_list(v.args[0]):
        v = v.args[0]
    if not (is_list(v) and len(v.args) == 2 and all(_isnum(a) for a in v.args)):
        raise ValueError(f"{fname}: TimeObservationWindow must be {{min, max}} or Interval[{{min, max}}]")
    a, b = S(v.args[0]), S(v.args[1])
    if a > b:
        raise ValueError(f"{fname}: the observation window needs min <= max")
    for t in times:
        if _isnum(t) and not (a <= S(t) <= b):
            raise ValueError(f"{fname}: the timestamp {t} lies outside the observation window")
    return Expr("Interval", L(v.args[0], v.args[1]))


T.TOW_CHECK = tow_check


# ─────────────────────────── queueing networks ───────────────────────────

class QNet(P.Proc):
    jump = True
    has_cov = False

    def parse(self, a, fname):
        if len(a) not in (4, 5):
            raise ValueError(f"{fname}: expected QueueingNetworkProcess[gamma, r, mu, c] or [gamma, r, mu, c, k]")
        g = P._plist(a[0], fname, "arrival rate vector")
        r = P._matrix(a[1], fname, "routing matrix")
        mu = P._plist(a[2], fname, "service rate vector")
        c = a[3]
        n = len(g)
        if not (r.rows == n and r.cols == n and len(mu) == n):
            raise ValueError(f"{fname}: gamma, r, mu and c must describe the same number of nodes")
        if not (is_list(c) and len(c.args) == n and all(_int(v) and v >= 1 for v in c.args)):
            if is_list(c) and any(isinstance(v, Symbol) for v in c.args):
                raise Unevaluated()
            raise ValueError(f"{fname}: c must be a list of {n} positive integers")
        for v in g:
            if P._num(v) and not v >= 0:
                raise ValueError(f"{fname}: arrival rates must be non-negative")
        for v in mu:
            P._pos(v, fname, "each service rate")
        for i in range(n):
            for v in r.row(i):
                if P._num(v) and not (v.is_real and 0 <= v <= 1):
                    raise ValueError(f"{fname}: routing probabilities must be between 0 and 1")
            if all(P._num(v) for v in r.row(i)) and sum(r.row(i)) > 1 and not P._close(sum(r.row(i)), 1):
                raise ValueError(f"{fname}: the routing probabilities out of a node must sum to at most 1")
        self.g, self.r, self.mu, self.c, self.n = g, r, mu, list(c.args), n
        self.k = None
        if len(a) == 5:
            k = a[4]
            if isinstance(k, Symbol):
                raise Unevaluated()
            if not (_int(k) and k >= 1):
                raise ValueError(f"{fname}: the number of jobs k must be a positive integer")
            self.k = k
            if any(P._num(v) and v != 0 for v in g):
                raise NotImplementedError(f"{fname}: closed networks with external arrivals (mixed networks) are "
                                          "not supported")
            for i in range(n):
                if all(P._num(v) for v in r.row(i)) and not P._close(sum(r.row(i)), 1):
                    raise ValueError(f"{fname}: in a closed network the routing rows must sum to 1")
        self.p = list(g) + list(r) + list(mu)

    def check_time(self, t, fname):
        P._nonneg_time(t, fname)

    def traffic(self, fname):
        if not P._numeric_all(self.p):
            raise Unevaluated()
        I = sp.eye(self.n)
        M = I - self.r
        if M.det() == 0:
            raise ValueError(f"{fname}: some customers never leave the open network (I - r is singular)")
        lam = sp.Matrix([self.g]) * M.inv()
        return [sp.nsimplify(v) if not P._inexact(v) else v for v in lam]

    def node_queue(self, i, fname):
        lam = self.traffic(fname)[i]
        return lam, Expr("QueueingProcess", R(lam), R(self.mu[i]), self.c[i])

    def gordon_newell(self, fname):
        """state probabilities {state tuple: prob} of the closed network"""
        if not P._numeric_all(self.p):
            raise Unevaluated()
        n, k = self.n, self.k
        # visit ratios: v = v r, v_1 = 1
        A = (self.r.T - sp.eye(n))
        A = A.row_insert(n, sp.Matrix([[1] + [0] * (n - 1)]))
        b = sp.Matrix([0] * n + [1])
        v = A.solve_least_squares(b) if P._inexact(*self.p) else A.gauss_jordan_solve(b)[0]
        v = [sp.nsimplify(x) if not P._inexact(x) else x for x in v]
        if math.comb(n + k - 1, k) > 200000:
            raise NotImplementedError(f"{fname}: more than 200000 network states are not supported")
        states, ws = [], []
        for comb in itertools.combinations_with_replacement(range(n), k):
            st = [0] * n
            for j in comb:
                st[j] += 1
            w = sp.Integer(1)
            for i in range(n):
                ci = self.c[i]
                beta = sp.factorial(st[i]) if st[i] <= ci else sp.factorial(ci) * sp.Integer(ci) ** (st[i] - ci)
                w *= (v[i] / self.mu[i]) ** st[i] / beta
            states.append(tuple(st))
            ws.append(w)
        Z = sum(ws)
        return {s: (sp.simplify(w / Z) if not P._inexact(w) else w / Z) for s, w in zip(states, ws)}

    def stationary_distribution(self, fname):
        if self.k is not None:
            raise NotImplementedError(f"{fname}: the stationary distribution of a closed network is a multivariate "
                                      "empirical distribution, which is not supported (use QueueProperties[{net, "
                                      "i}, ...] for node measures)")
        parts = []
        for i in range(self.n):
            lam, q = self.node_queue(i, fname)
            if lam == 0:
                parts.append(Expr("EmpiricalDistribution", rule(L(1), L(0))))
                continue
            if not sp.simplify(lam - self.c[i] * self.mu[i]) < 0:
                raise ValueError(f"{fname}: node {i + 1} is not stable (traffic >= capacity)")
            parts.append(P.proc_of(q, fname).stationary_dist(fname))
        return Expr("ProductDistribution", *parts)

    def check_time(self, t, fname):
        if t == sp.oo:
            return
        super().check_time(t, fname)

    def mean(self, t):        # vector valued: a sympy Tuple so the shared moment code can simplify it
        return sp.Tuple(*(sp.sympify(v) for v in S(ev(Expr("Mean", self.slice(t))))))

    def var(self, t):
        return sp.Tuple(*(sp.sympify(v) for v in S(ev(Expr("Variance", self.slice(t))))))

    def slice(self, t):
        if isinstance(R(t), Symbol) and R(t).name == "Infinity" or t == sp.oo:
            return self.stationary_distribution("SliceDistribution")
        raise NotImplementedError("SliceDistribution: transient distributions of queueing networks are not "
                                  "available in closed form")

    def jumps(self, rng, t0, t1):
        if not P._numeric_all(self.p):
            raise Unevaluated()
        n = self.n
        g = np.array([float(v) for v in self.g])
        mu = np.array([float(v) for v in self.mu])
        Rm = np.array([[float(v) for v in self.r.row(i)] for i in range(n)])
        c = np.array(self.c)
        st = np.zeros(n, dtype=int)
        if self.k is not None:
            st[0] = self.k
        t = t0
        out = [(t0, L(*(int(v) for v in st)))]
        steps = 0
        while True:
            rates = np.concatenate([g, mu * np.minimum(st, c)])
            tot = rates.sum()
            if tot <= 0:
                break
            t += rng.exponential(1 / tot)
            if t > t1:
                break
            ev_ = int(rng.choice(len(rates), p=rates / tot))
            if ev_ < n:
                st[ev_] += 1
            else:
                i = ev_ - n
                st[i] -= 1
                pr = np.append(Rm[i], max(0.0, 1 - Rm[i].sum()))
                j = int(rng.choice(n + 1, p=pr / pr.sum()))
                if j < n:
                    st[j] += 1
            out.append((t, L(*(int(v) for v in st))))
            steps += 1
            if steps > 10 ** 6:
                raise NotImplementedError("RandomFunction: more than 10^6 network events are not supported")
        return out


P.CLASSES["QueueingNetworkProcess"] = QNet


@fn("QueueingNetworkProcess", area=AREA, status="added",
    notes="QueueingNetworkProcess[gamma, r, mu, c] (open Jackson network: Poisson arrivals gamma_i, routing "
          "matrix r with row sums <= 1, exponential services mu_i, c_i servers) and QueueingNetworkProcess[gamma, "
          "r, mu, c, k] (closed Gordon-Newell network with k jobs, routing rows summing to 1, gamma = 0). The "
          "process value is the vector of node populations. Open networks: traffic equations lambda = gamma.(I - "
          "r)^-1 (exact), node measures of M/M/c_i queues with arrival rate lambda_i via QueueProperties[{net, i}] "
          "(and prop), network measures QueueProperties[net, prop] for \"MeanSystemSize\", \"MeanQueueSize\" "
          "(sums), \"Throughput\" (Total[gamma]), \"MeanSystemTime\", \"MeanQueueTime\" (Little's law), "
          "\"NodeCount\", \"StationarySystem\"; StationaryDistribution (and net[Infinity]) is the ProductDistribution "
          "of the node distributions (Jackson's theorem). Closed networks: exact Gordon-Newell probabilities by "
          "enumeration (visit ratios v = v.r, v_1 = 1) giving node \"MeanSystemSize\", \"MeanQueueSize\", "
          "\"Throughput\", \"UtilizationFactor\", \"MeanSystemTime\", \"MeanQueueTime\". RandomFunction simulates "
          "the continuous-time Markov chain (open networks start empty; closed networks start with all k jobs at "
          "node 1 -- a stated choice). Closed-network StationaryDistribution (multivariate empirical), transient "
          "slices and mixed networks raise NotImplementedError; invalid parameters ValueError. Check: Jackson / "
          "Burke product form against the documentation examples, Gordon-Newell probabilities summing to 1 and "
          "mean value analysis (tests).")
def _qnet_ctor(*args: Expr) -> Expr:
    del P._CONDS[:]
    QNet(Expr("QueueingNetworkProcess", *args), "QueueingNetworkProcess")
    raise Unevaluated()


_NODE_PROPS = ("ArrivalRate", "ServiceRate", "ServiceChannels", "UtilizationFactor", "Throughput", "MeanSystemSize",
               "MeanQueueSize", "MeanSystemTime", "MeanQueueTime", "StationarySystem")
_NET_PROPS = ("NodeCount", "StationarySystem", "MeanSystemSize", "MeanQueueSize", "Throughput", "MeanSystemTime",
              "MeanQueueTime")


def _closed_node(pr, i, prop, fname):
    probs = pr.gordon_newell(fname)
    ci, mui = pr.c[i], pr.mu[i]
    L_ = sum(p * s[i] for s, p in probs.items())
    Lq = sum(p * max(0, s[i] - ci) for s, p in probs.items())
    X = sum(p * min(s[i], ci) for s, p in probs.items()) * mui
    fix = (lambda v: sp.simplify(v)) if not P._inexact(*pr.p) else (lambda v: v)
    tab = {"ArrivalRate": X, "ServiceRate": mui, "ServiceChannels": ci, "Throughput": X,
           "UtilizationFactor": X / (ci * mui), "MeanSystemSize": L_, "MeanQueueSize": Lq,
           "MeanSystemTime": L_ / X if X != 0 else sp.oo, "MeanQueueTime": Lq / X if X != 0 else sp.oo,
           "StationarySystem": True}
    v = tab[prop]
    return v if isinstance(v, bool) or _int(v) else R(fix(v))


def _open_node(pr, i, prop, fname):
    lam, q = pr.node_queue(i, fname)
    if prop == "ArrivalRate":
        return R(lam)
    if prop == "ServiceRate":
        return R(pr.mu[i])
    if prop == "ServiceChannels":
        return pr.c[i]
    if lam == 0:
        return True if prop == "StationarySystem" else 0
    qp = P.proc_of(q, fname)
    return P._queue_prop(qp, prop, fname)


def qnet_props(q, prop, extra):
    """QueueProperties hook: QueueingNetworkProcess and {net, i}; None for other input"""
    fname = "QueueProperties"
    node = None
    if is_list(q) and len(q.args) == 2 and isinstance(q.args[0], Expr) and \
            q.args[0].head == "QueueingNetworkProcess":
        net, node = q.args
        if not (_int(node)):
            raise TypeError(f"{fname}: the node must be a positive integer")
    elif isinstance(q, Expr) and q.head == "QueueingNetworkProcess":
        net = q
    else:
        return None
    if extra is not None:
        raise NotImplementedError(f"{fname}: property descriptions are not supported")
    pr = P.proc_of(net, fname)
    if node is not None and not 1 <= node <= pr.n:
        raise ValueError(f"{fname}: the network has nodes 1 to {pr.n}")
    props = _NODE_PROPS if node is not None else _NET_PROPS

    def one(p):
        if p not in props:
            raise ValueError(f"{fname}: unknown or unsupported property {p!r} (supported: {', '.join(props)})")
        if node is not None:
            if pr.k is not None:
                return _closed_node(pr, node - 1, p, fname)
            return _open_node(pr, node - 1, p, fname)
        if p == "NodeCount":
            return pr.n
        if pr.k is not None:
            if p == "StationarySystem":
                return True
            if p == "MeanSystemSize":
                return pr.k
            if p == "MeanQueueSize":
                return R(sum(S(_closed_node(pr, i, "MeanQueueSize", fname)) for i in range(pr.n)))
            raise NotImplementedError(f"{fname}: {p!r} is not defined for a closed network as a whole (use the "
                                      "node form {net, i})")
        if p == "StationarySystem":
            return all(_open_node(pr, i, "StationarySystem", fname) is True for i in range(pr.n))
        tot_g = sum(pr.g)
        Ls = sum(S(_open_node(pr, i, "MeanSystemSize", fname)) for i in range(pr.n))
        Lqs = sum(S(_open_node(pr, i, "MeanQueueSize", fname)) for i in range(pr.n))
        val = {"MeanSystemSize": Ls, "MeanQueueSize": Lqs, "Throughput": tot_g,
               "MeanSystemTime": Ls / tot_g, "MeanQueueTime": Lqs / tot_g}[p]
        return R(sp.simplify(val) if not P._inexact(val) else val)
    if prop is None:
        out = []
        for k in props:
            try:
                out.append(rule(k, one(k)))
            except (ValueError, NotImplementedError):
                out.append(rule(k, Expr("Missing", "NotAvailable")))
        return Expr("Association", *out)
    if prop == "Properties":
        return L(*props)
    if not isinstance(prop, str):
        raise TypeError(f"{fname}: the property must be a string")
    return one(prop)


P.QPROPS_HOOK = qnet_props
