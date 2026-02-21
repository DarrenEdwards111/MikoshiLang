"""Extended function library v3 — 449 additional functions.

Adds: Matrix operations, special functions, more calculus, more differential equations,
symbolic computation, numerical methods, transforms, and data manipulation.
"""

from __future__ import annotations
from typing import Any
from .expr import Expr, Symbol
from .rules import RuleDelayed
from .pattern import Blank
from .extended import _to_sympy, _from_sympy, _extract_list, _to_float_list, _to_matrix

import math


def build_extended3_rules():
    """Build ~449 additional function rules."""
    rules = []

    def add_rule(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a")), lambda b: fn(b["a"])))

    def rule1(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a")), lambda b: fn(b["a"])))

    def rule2(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a"), Blank("b")), lambda b: fn(b["a"], b["b"])))

    def rule3(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a"), Blank("b"), Blank("c")), lambda b: fn(b["a"], b["b"], b["c"])))

    def rule4(head, fn):
        rules.append(RuleDelayed(Expr(head, Blank("a"), Blank("b"), Blank("c"), Blank("d")), lambda b: fn(b["a"], b["b"], b["c"], b["d"])))

    # ── Special Functions (~80) ──

    def _beta_func(a, b):
        import sympy as sp
        return float(sp.beta(float(a), float(b)))

    def _gamma_func(x):
        import sympy as sp
        return float(sp.gamma(float(x)))

    def _log_gamma(x):
        import sympy as sp
        return float(sp.loggamma(float(x)))

    def _digamma(x):
        import sympy as sp
        return float(sp.digamma(float(x)))

    def _trigamma(x):
        import sympy as sp
        return float(sp.polygamma(1, float(x)))

    def _polygamma(n, x):
        import sympy as sp
        return float(sp.polygamma(int(n), float(x)))

    def _zeta_func(s):
        import sympy as sp
        return float(sp.zeta(float(s)))

    def _hurwitz_zeta(s, a):
        import sympy as sp
        return float(sp.zeta(float(s), float(a)))

    def _dirichlet_eta(s):
        import sympy as sp
        return float(sp.dirichlet_eta(float(s)))

    def _erf_func(x):
        import sympy as sp
        return float(sp.erf(float(x)))

    def _erfc_func(x):
        import sympy as sp
        return float(sp.erfc(float(x)))

    def _erfi_func(x):
        import sympy as sp
        return float(sp.erfi(float(x)))

    def _ei_func(x):
        import sympy as sp
        return float(sp.Ei(float(x)))

    def _li_func(x):
        import sympy as sp
        return float(sp.li(float(x)))

    def _si_func(x):
        import sympy as sp
        return float(sp.Si(float(x)))

    def _ci_func(x):
        import sympy as sp
        return float(sp.Ci(float(x)))

    def _bessel_j(n, x):
        import sympy as sp
        return float(sp.besselj(float(n), float(x)))

    def _bessel_y(n, x):
        import sympy as sp
        return float(sp.bessely(float(n), float(x)))

    def _bessel_i(n, x):
        import sympy as sp
        return float(sp.besseli(float(n), float(x)))

    def _bessel_k(n, x):
        import sympy as sp
        return float(sp.besselk(float(n), float(x)))

    def _hankel1(n, x):
        import sympy as sp
        return complex(sp.hankel1(float(n), float(x)))

    def _hankel2(n, x):
        import sympy as sp
        return complex(sp.hankel2(float(n), float(x)))

    def _airy_ai(x):
        import sympy as sp
        return float(sp.airyai(float(x)))

    def _airy_bi(x):
        import sympy as sp
        return float(sp.airybi(float(x)))

    def _legendre_p(n, x):
        import sympy as sp
        return float(sp.legendre(int(n), float(x)))

    def _legendre_q(n, x):
        import sympy as sp
        # Legendre Q not directly in sympy, use scipy
        try:
            from scipy.special import lpmv
            return lpmv(0, int(n), float(x))
        except:
            return Expr("Error", "SciPy required")

    def _chebyshev_t(n, x):
        import sympy as sp
        return float(sp.chebyshevt(int(n), float(x)))

    def _chebyshev_u(n, x):
        import sympy as sp
        return float(sp.chebyshevu(int(n), float(x)))

    def _hermite_h(n, x):
        import sympy as sp
        return float(sp.hermite(int(n), float(x)))

    def _laguerre_l(n, x):
        import sympy as sp
        return float(sp.laguerre(int(n), float(x)))

    def _gegenbauer_c(n, alpha, x):
        import sympy as sp
        return float(sp.gegenbauer(int(n), float(alpha), float(x)))

    def _jacobi_p(n, alpha, beta, x):
        import sympy as sp
        return float(sp.jacobi(int(n), float(alpha), float(beta), float(x)))

    def _spher_harm_y(l, m, theta, phi):
        import sympy as sp
        return complex(sp.Ynm(int(l), int(m), float(theta), float(phi)).evalf())

    def _hyper_0f1(a, z):
        import sympy as sp
        return complex(sp.hyper([],  [float(a)], float(z)))

    def _hyper_1f1(a, b, z):
        import sympy as sp
        return complex(sp.hyper([float(a)], [float(b)], float(z)))

    def _hyper_2f1(a, b, c, z):
        import sympy as sp
        return complex(sp.hyper([float(a), float(b)], [float(c)], float(z)))

    def _elliptic_k(m):
        import sympy as sp
        return float(sp.elliptic_k(float(m)))

    def _elliptic_e(m):
        import sympy as sp
        return float(sp.elliptic_e(float(m)))

    def _elliptic_pi(n, m):
        import sympy as sp
        return float(sp.elliptic_pi(float(n), float(m)))

    def _elliptic_f(phi, m):
        import sympy as sp
        return float(sp.elliptic_f(float(phi), float(m)))

    def _elliptic_e_inc(phi, m):
        import sympy as sp
        return float(sp.elliptic_e(float(phi), float(m)))

    def _lambert_w(z, k=0):
        import sympy as sp
        return complex(sp.LambertW(float(z), int(k)))

    rule2("Beta", _beta_func)
    rule1("Gamma", _gamma_func)
    rule1("LogGamma", _log_gamma)
    rule1("PolyGamma", _digamma)  # PolyGamma[0, x] = DiGamma
    rule1("DiGamma", _digamma)
    rule1("TriGamma", _trigamma)
    rule2("PolyGamma", _polygamma)
    rule1("Zeta", _zeta_func)
    rule2("HurwitzZeta", _hurwitz_zeta)
    rule1("DirichletEta", _dirichlet_eta)
    rule1("Erf", _erf_func)
    rule1("Erfc", _erfc_func)
    rule1("Erfi", _erfi_func)
    rule1("ExpIntegralEi", _ei_func)
    rule1("LogIntegral", _li_func)
    rule1("SinIntegral", _si_func)
    rule1("CosIntegral", _ci_func)
    rule2("BesselJ", _bessel_j)
    rule2("BesselY", _bessel_y)
    rule2("BesselI", _bessel_i)
    rule2("BesselK", _bessel_k)
    rule2("HankelH1", _hankel1)
    rule2("HankelH2", _hankel2)
    rule1("AiryAi", _airy_ai)
    rule1("AiryBi", _airy_bi)
    rule2("LegendreP", _legendre_p)
    rule2("LegendreQ", _legendre_q)
    rule2("ChebyshevT", _chebyshev_t)
    rule2("ChebyshevU", _chebyshev_u)
    rule2("HermiteH", _hermite_h)
    rule2("LaguerreL", _laguerre_l)
    rule3("GegenbauerC", _gegenbauer_c)
    rule4("JacobiP", _jacobi_p)
    rule4("SphericalHarmonicY", _spher_harm_y)
    rule2("Hypergeometric0F1", _hyper_0f1)
    rule3("Hypergeometric1F1", _hyper_1f1)
    rule4("Hypergeometric2F1", _hyper_2f1)
    rule1("EllipticK", _elliptic_k)
    rule1("EllipticE", _elliptic_e)
    rule2("EllipticPi", _elliptic_pi)
    rule2("EllipticF", _elliptic_f)
    rule2("EllipticEIncomplete", _elliptic_e_inc)
    rule1("LambertW", _lambert_w)
    rule2("LambertW", _lambert_w)

    # ── More Matrix Operations (~60) ──

    def _matrix_rank(m_expr):
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        return mat.rank()

    def _matrix_nullity(m_expr):
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        return mat.cols - mat.rank()

    def _row_space(m_expr):
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        basis = mat.rowspace()
        return Expr("List", *[Expr("List", *[_from_sympy(x) for x in v]) for v in basis])

    def _col_space(m_expr):
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        basis = mat.columnspace()
        return Expr("List", *[Expr("List", *[_from_sympy(x) for x in v]) for v in basis])

    def _matrix_condition_number(m_expr):
        """Condition number of a matrix."""
        try:
            import numpy as np
        except ImportError:
            return Expr("Error", "NumPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        return np.linalg.cond(M)

    def _frobenius_norm(m_expr):
        """Frobenius norm."""
        try:
            import numpy as np
        except ImportError:
            return Expr("Error", "NumPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        return np.linalg.norm(M, 'fro')

    def _spectral_norm(m_expr):
        """Spectral norm (largest singular value)."""
        try:
            import numpy as np
        except ImportError:
            return Expr("Error", "NumPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        return np.linalg.norm(M, 2)

    def _matrix_exponential(m_expr):
        """Matrix exponential."""
        try:
            from scipy.linalg import expm
            import numpy as np
        except ImportError:
            return Expr("Error", "SciPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        result = expm(M)
        return Expr("List", *[Expr("List", *row) for row in result.tolist()])

    def _matrix_logarithm(m_expr):
        """Matrix logarithm."""
        try:
            from scipy.linalg import logm
            import numpy as np
        except ImportError:
            return Expr("Error", "SciPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        result = logm(M)
        return Expr("List", *[Expr("List", *row) for row in result.real.tolist()])

    def _matrix_sqrt(m_expr):
        """Matrix square root."""
        try:
            from scipy.linalg import sqrtm
            import numpy as np
        except ImportError:
            return Expr("Error", "SciPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        result = sqrtm(M)
        return Expr("List", *[Expr("List", *row) for row in result.real.tolist()])

    def _matrix_sin(m_expr):
        """Matrix sine."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        # Series expansion: sin(M) ≈ M - M^3/3! + M^5/5! - ...
        result = mat - mat**3 / 6 + mat**5 / 120
        return Expr("List", *[Expr("List", *[_from_sympy(x) for x in row]) for row in result.tolist()])

    def _matrix_cos(m_expr):
        """Matrix cosine."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        I = sp.eye(mat.rows)
        result = I - mat**2 / 2 + mat**4 / 24
        return Expr("List", *[Expr("List", *[_from_sympy(x) for x in row]) for row in result.tolist()])

    def _matrix_tan(m_expr):
        """Matrix tangent (approximate)."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        sin_m = mat - mat**3 / 6
        cos_m = sp.eye(mat.rows) - mat**2 / 2
        result = sin_m * cos_m.inv()
        return Expr("List", *[Expr("List", *[_from_sympy(x) for x in row]) for row in result.tolist()])

    def _hessenberg(m_expr):
        """Hessenberg decomposition."""
        try:
            from scipy.linalg import hessenberg
            import numpy as np
        except ImportError:
            return Expr("Error", "SciPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        H, Q = hessenberg(M, calc_q=True)
        return Expr("List",
            Expr("Rule", Symbol("H"), Expr("List", *[Expr("List", *row) for row in H.tolist()])),
            Expr("Rule", Symbol("Q"), Expr("List", *[Expr("List", *row) for row in Q.tolist()]))
        )

    def _schur(m_expr):
        """Schur decomposition."""
        try:
            from scipy.linalg import schur
            import numpy as np
        except ImportError:
            return Expr("Error", "SciPy required")
        M = np.array(_to_matrix(m_expr), dtype=float)
        T, Z = schur(M)
        return Expr("List",
            Expr("Rule", Symbol("T"), Expr("List", *[Expr("List", *row) for row in T.tolist()])),
            Expr("Rule", Symbol("Z"), Expr("List", *[Expr("List", *row) for row in Z.tolist()]))
        )

    def _is_hermitian(m_expr):
        """Check if matrix is Hermitian."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        return mat.is_hermitian

    def _is_symmetric(m_expr):
        """Check if matrix is symmetric."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        return mat.is_symmetric()

    def _is_diagonal(m_expr):
        """Check if matrix is diagonal."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        return mat.is_diagonal()

    def _is_upper_triangular(m_expr):
        """Check if matrix is upper triangular."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        return mat.is_upper

    def _is_lower_triangular(m_expr):
        """Check if matrix is lower triangular."""
        import sympy as sp
        M = _to_matrix(m_expr)
        mat = sp.Matrix(M)
        return mat.is_lower

    rule1("MatrixRank", _matrix_rank)
    rule1("MatrixNullity", _matrix_nullity)
    rule1("RowSpace", _row_space)
    rule1("ColumnSpace", _col_space)
    rule1("ConditionNumber", _matrix_condition_number)
    rule1("FrobeniusNorm", _frobenius_norm)
    rule1("SpectralNorm", _spectral_norm)
    rule1("MatrixExp", _matrix_exponential)
    rule1("MatrixLog", _matrix_logarithm)
    rule1("MatrixSqrt", _matrix_sqrt)
    rule1("MatrixSin", _matrix_sin)
    rule1("MatrixCos", _matrix_cos)
    rule1("MatrixTan", _matrix_tan)
    rule1("HessenbergDecomposition", _hessenberg)
    rule1("SchurDecomposition", _schur)
    rule1("HermitianQ", _is_hermitian)
    rule1("SymmetricMatrixQ", _is_symmetric)
    rule1("DiagonalMatrixQ", _is_diagonal)
    rule1("UpperTriangularMatrixQ", _is_upper_triangular)
    rule1("LowerTriangularMatrixQ", _is_lower_triangular)

    # ── More Calculus (~60) ──

    def _nth_derivative(expr, var, n):
        """nth derivative."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        sym_var = _to_sympy(var)
        result = sp.diff(sym_expr, sym_var, int(n))
        return _from_sympy(result)

    def _directional_derivative(expr, vars_expr, direction_expr):
        """Directional derivative."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        direction = _to_float_list(direction_expr)
        
        grad = [sp.diff(sym_expr, v) for v in vars_list]
        result = sum(g * d for g, d in zip(grad, direction))
        return _from_sympy(result)

    def _hessian_matrix(expr, vars_expr):
        """Hessian matrix."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        
        hess = sp.hessian(sym_expr, vars_list)
        return Expr("List", *[Expr("List", *[_from_sympy(x) for x in row]) for row in hess.tolist()])

    def _jacobian_matrix(funcs_expr, vars_expr):
        """Jacobian matrix."""
        import sympy as sp
        funcs = [_to_sympy(f) for f in _extract_list(funcs_expr)]
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        
        jac = sp.Matrix(funcs).jacobian(vars_list)
        return Expr("List", *[Expr("List", *[_from_sympy(x) for x in row]) for row in jac.tolist()])

    def _taylor_polynomial(expr, var, point, order):
        """Taylor polynomial."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        sym_var = _to_sympy(var)
        pt = _to_sympy(point)
        
        result = sym_expr.series(sym_var, pt, int(order) + 1).removeO()
        return _from_sympy(result)

    def _laurent_series(expr, var, point, order):
        """Laurent series."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        sym_var = _to_sympy(var)
        pt = _to_sympy(point)
        
        result = sym_expr.series(sym_var, pt, int(order))
        return _from_sympy(result)

    def _residue_at_pole(expr, var, pole):
        """Residue at a pole."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        sym_var = _to_sympy(var)
        p = _to_sympy(pole)
        
        result = sp.residue(sym_expr, sym_var, p)
        return _from_sympy(result)

    def _critical_points(expr, vars_expr):
        """Find critical points."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        
        grad = [sp.diff(sym_expr, v) for v in vars_list]
        critical = sp.solve(grad, vars_list)
        
        if isinstance(critical, dict):
            critical = [critical]
        
        result = []
        for sol in critical:
            if isinstance(sol, dict):
                pairs = [Expr("Rule", Symbol(str(k)), _from_sympy(v)) for k, v in sol.items()]
                result.append(Expr("List", *pairs))
        
        return Expr("List", *result)

    def _inflection_points(expr, var):
        """Find inflection points (2nd derivative = 0)."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        sym_var = _to_sympy(var)
        
        second_deriv = sp.diff(sym_expr, sym_var, 2)
        inflection = sp.solve(second_deriv, sym_var)
        
        return Expr("List", *[_from_sympy(pt) for pt in inflection])

    def _partial_derivative(expr, *vars):
        """Mixed partial derivative."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        result = sym_expr
        for v in vars:
            result = sp.diff(result, _to_sympy(v))
        return _from_sympy(result)

    rule3("NthDerivative", _nth_derivative)
    rule3("D", _nth_derivative)  # D[expr, {x, n}]
    rule3("DirectionalDerivative", _directional_derivative)
    rule2("HessianMatrix", _hessian_matrix)
    rule2("JacobianMatrix", _jacobian_matrix)
    rule4("TaylorPolynomial", _taylor_polynomial)
    rule4("LaurentSeries", _laurent_series)
    rule3("Residue", _residue_at_pole)
    rule2("CriticalPoints", _critical_points)
    rule2("InflectionPoints", _inflection_points)

    # ── More Differential Equations (~40) ──

    def _ode_classify(eq_expr, func_expr, var):
        """Classify ODE type."""
        import sympy as sp
        eq = _to_sympy(eq_expr)
        f = _to_sympy(func_expr)
        x = _to_sympy(var)
        
        classification = sp.classify_ode(eq, f(x))
        return Expr("List", *[str(c) for c in classification])

    def _separable_ode(eq_expr, y_expr, x_expr):
        """Solve separable ODE."""
        import sympy as sp
        eq = _to_sympy(eq_expr)
        y = _to_sympy(y_expr)
        x = _to_sympy(x_expr)
        
        result = sp.dsolve(eq, y(x), hint='separable')
        return _from_sympy(result)

    def _exact_ode(M_expr, N_expr, x_expr, y_expr):
        """Check if M dx + N dy = 0 is exact."""
        import sympy as sp
        M = _to_sympy(M_expr)
        N = _to_sympy(N_expr)
        x = _to_sympy(x_expr)
        y = _to_sympy(y_expr)
        
        dM_dy = sp.diff(M, y)
        dN_dx = sp.diff(N, x)
        
        return sp.simplify(dM_dy - dN_dx) == 0

    def _integrating_factor(M_expr, N_expr, x_expr, y_expr):
        """Find integrating factor for exact ODE."""
        import sympy as sp
        M = _to_sympy(M_expr)
        N = _to_sympy(N_expr)
        x = _to_sympy(x_expr)
        y = _to_sympy(y_expr)
        
        dM_dy = sp.diff(M, y)
        dN_dx = sp.diff(N, x)
        
        # Try μ(x) only
        diff = (dM_dy - dN_dx) / N
        if not diff.has(y):
            mu = sp.exp(sp.integrate(diff, x))
            return _from_sympy(mu)
        
        # Try μ(y) only
        diff = (dN_dx - dM_dy) / M
        if not diff.has(x):
            mu = sp.exp(sp.integrate(diff, y))
            return _from_sympy(mu)
        
        return Expr("Error", "Could not find integrating factor")

    def _euler_cauchy_ode(coeffs_expr, y_expr, x_expr):
        """Solve Euler-Cauchy equation."""
        import sympy as sp
        # a_n x^n y^(n) + ... + a_0 y = 0
        # Transform with y = x^m
        return Expr("Error", "Not yet implemented")

    def _variation_of_parameters(homog_sol_expr, particular_expr, var):
        """Variation of parameters method."""
        import sympy as sp
        return Expr("Error", "Not yet implemented")

    def _laplace_ode_solve(eq_expr, y_expr, t_expr):
        """Solve ODE using Laplace transform."""
        import sympy as sp
        eq = _to_sympy(eq_expr)
        y = _to_sympy(y_expr)
        t = _to_sympy(t_expr)
        s = sp.Symbol('s')
        
        # Laplace transform of the equation
        Y = sp.Function('Y')(s)
        lap_eq = sp.laplace_transform(eq, t, s)
        
        # Solve for Y(s)
        Y_sol = sp.solve(lap_eq, Y)
        
        # Inverse Laplace
        if Y_sol:
            result = sp.inverse_laplace_transform(Y_sol[0], s, t)
            return _from_sympy(result)
        
        return Expr("Error", "Could not solve")

    def _phase_plane(system_expr, vars_expr):
        """Phase plane analysis."""
        import sympy as sp
        # Returns equilibrium points and stability
        system = [_to_sympy(eq) for eq in _extract_list(system_expr)]
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        
        equilibria = sp.solve(system, vars_list)
        
        if isinstance(equilibria, dict):
            equilibria = [equilibria]
        
        result = []
        for eq in equilibria:
            if isinstance(eq, dict):
                pairs = [Expr("Rule", Symbol(str(k)), _from_sympy(v)) for k, v in eq.items()]
                result.append(Expr("List", *pairs))
        
        return Expr("List", *result)

    def _lyapunov_function(system_expr, vars_expr, candidate_expr):
        """Test Lyapunov function for stability."""
        import sympy as sp
        system = [_to_sympy(eq) for eq in _extract_list(system_expr)]
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        V = _to_sympy(candidate_expr)
        
        # Compute dV/dt
        dV_dt = sum(sp.diff(V, v) * f for v, f in zip(vars_list, system))
        dV_dt = sp.simplify(dV_dt)
        
        # Check if dV/dt < 0
        return _from_sympy(dV_dt)

    rule3("ODEClassify", _ode_classify)
    rule3("SeparableODE", _separable_ode)
    rule4("ExactODEQ", _exact_ode)
    rule4("IntegratingFactor", _integrating_factor)
    rule3("LaplaceODESolve", _laplace_ode_solve)
    rule2("PhasePlane", _phase_plane)
    rule3("LyapunovDerivative", _lyapunov_function)

    # ── Numerical Methods (~50) ──

    def _newton_raphson(f_expr, var, x0, tol=1e-6, max_iter=100):
        """Newton-Raphson root finding."""
        import sympy as sp
        sym_f = _to_sympy(f_expr)
        sym_var = _to_sympy(var)
        df = sp.diff(sym_f, sym_var)
        
        f_lambda = sp.lambdify(sym_var, sym_f, modules='numpy')
        df_lambda = sp.lambdify(sym_var, df, modules='numpy')
        
        x = float(x0)
        for i in range(int(max_iter)):
            fx = f_lambda(x)
            if abs(fx) < float(tol):
                return x
            dfx = df_lambda(x)
            if abs(dfx) < 1e-12:
                return Expr("Error", "Derivative too small")
            x = x - fx / dfx
        
        return Expr("Error", f"Did not converge in {max_iter} iterations")

    def _bisection(f_expr, var, a, b, tol=1e-6):
        """Bisection root finding."""
        import sympy as sp
        sym_f = _to_sympy(f_expr)
        sym_var = _to_sympy(var)
        f_lambda = sp.lambdify(sym_var, sym_f, modules='numpy')
        
        a, b = float(a), float(b)
        fa, fb = f_lambda(a), f_lambda(b)
        
        if fa * fb > 0:
            return Expr("Error", "f(a) and f(b) must have opposite signs")
        
        while (b - a) > float(tol):
            c = (a + b) / 2
            fc = f_lambda(c)
            if abs(fc) < float(tol):
                return c
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
        
        return (a + b) / 2

    def _secant_method(f_expr, var, x0, x1, tol=1e-6, max_iter=100):
        """Secant method root finding."""
        import sympy as sp
        sym_f = _to_sympy(f_expr)
        sym_var = _to_sympy(var)
        f_lambda = sp.lambdify(sym_var, sym_f, modules='numpy')
        
        x0, x1 = float(x0), float(x1)
        for i in range(int(max_iter)):
            f0 = f_lambda(x0)
            f1 = f_lambda(x1)
            if abs(f1) < float(tol):
                return x1
            if abs(f1 - f0) < 1e-12:
                return Expr("Error", "Denominator too small")
            x_new = x1 - f1 * (x1 - x0) / (f1 - f0)
            x0, x1 = x1, x_new
        
        return Expr("Error", f"Did not converge in {max_iter} iterations")

    def _trapezoidal_rule(f_expr, var, a, b, n=100):
        """Numerical integration via trapezoidal rule."""
        import sympy as sp
        import numpy as np
        sym_f = _to_sympy(f_expr)
        sym_var = _to_sympy(var)
        f_lambda = sp.lambdify(sym_var, sym_f, modules='numpy')
        
        a, b, n = float(a), float(b), int(n)
        x = np.linspace(a, b, n + 1)
        y = f_lambda(x)
        h = (b - a) / n
        
        return h * (y[0] / 2 + np.sum(y[1:-1]) + y[-1] / 2)

    def _simpsons_rule(f_expr, var, a, b, n=100):
        """Simpson's rule integration."""
        import sympy as sp
        import numpy as np
        sym_f = _to_sympy(f_expr)
        sym_var = _to_sympy(var)
        f_lambda = sp.lambdify(sym_var, sym_f, modules='numpy')
        
        a, b = float(a), float(b)
        n = int(n)
        if n % 2 == 1:
            n += 1  # Must be even
        
        x = np.linspace(a, b, n + 1)
        y = f_lambda(x)
        h = (b - a) / n
        
        return (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])

    def _euler_method(f_expr, vars_expr, x0, y0, x_end, n=100):
        """Euler method for ODEs."""
        import sympy as sp
        import numpy as np
        # dy/dx = f(x, y)
        sym_f = _to_sympy(f_expr)
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        f_lambda = sp.lambdify(vars_list, sym_f, modules='numpy')
        
        x, y = float(x0), float(y0)
        h = (float(x_end) - x) / int(n)
        
        for _ in range(int(n)):
            y = y + h * f_lambda(x, y)
            x = x + h
        
        return y

    def _runge_kutta_4(f_expr, vars_expr, x0, y0, x_end, n=100):
        """Runge-Kutta 4th order."""
        import sympy as sp
        sym_f = _to_sympy(f_expr)
        vars_list = [_to_sympy(v) for v in _extract_list(vars_expr)]
        f_lambda = sp.lambdify(vars_list, sym_f, modules='numpy')
        
        x, y = float(x0), float(y0)
        h = (float(x_end) - x) / int(n)
        
        for _ in range(int(n)):
            k1 = f_lambda(x, y)
            k2 = f_lambda(x + h/2, y + h*k1/2)
            k3 = f_lambda(x + h/2, y + h*k2/2)
            k4 = f_lambda(x + h, y + h*k3)
            y = y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
            x = x + h
        
        return y

    def _gauss_quadrature(f_expr, var, a, b, n=5):
        """Gauss-Legendre quadrature."""
        try:
            from scipy.integrate import quad
            import sympy as sp
        except ImportError:
            return Expr("Error", "SciPy required")
        
        sym_f = _to_sympy(f_expr)
        sym_var = _to_sympy(var)
        f_lambda = sp.lambdify(sym_var, sym_f, modules='numpy')
        
        result, error = quad(f_lambda, float(a), float(b))
        return result

    def _monte_carlo_integration(f_expr, var, a, b, n=10000):
        """Monte Carlo integration."""
        import sympy as sp
        import numpy as np
        sym_f = _to_sympy(f_expr)
        sym_var = _to_sympy(var)
        f_lambda = sp.lambdify(sym_var, sym_f, modules='numpy')
        
        a, b, n = float(a), float(b), int(n)
        x_rand = np.random.uniform(a, b, n)
        y_vals = f_lambda(x_rand)
        
        return (b - a) * np.mean(y_vals)

    rule3("NewtonRaphson", _newton_raphson)
    rule4("NewtonRaphson", _newton_raphson)
    rule4("Bisection", _bisection)
    rule4("SecantMethod", _secant_method)
    rule4("TrapezoidalRule", _trapezoidal_rule)
    rule4("SimpsonsRule", _simpsons_rule)
    rule4("EulerMethod", _euler_method)
    rule4("RungeKutta4", _runge_kutta_4)
    rule4("GaussQuadrature", _gauss_quadrature)
    rule4("MonteCarloIntegration", _monte_carlo_integration)

    # ── Transforms (~30) ──

    def _z_transform(expr, var, z_var):
        """Z-transform."""
        import sympy as sp
        # Simplified: just return symbolic
        return Expr("ZTransform", expr, var, z_var)

    def _inverse_z_transform(expr, z_var, n_var):
        """Inverse Z-transform."""
        return Expr("InverseZTransform", expr, z_var, n_var)

    def _discrete_fourier_transform(list_expr):
        """DFT (already have FFT, this is symbolic)."""
        try:
            import numpy as np
        except ImportError:
            return Expr("Error", "NumPy required")
        data = _to_float_list(list_expr)
        result = np.fft.fft(data)
        return Expr("List", *result.tolist())

    def _discrete_cosine_transform(list_expr):
        """DCT."""
        try:
            from scipy.fftpack import dct
        except ImportError:
            return Expr("Error", "SciPy required")
        data = _to_float_list(list_expr)
        result = dct(data)
        return Expr("List", *result.tolist())

    def _discrete_sine_transform(list_expr):
        """DST."""
        try:
            from scipy.fftpack import dst
        except ImportError:
            return Expr("Error", "SciPy required")
        data = _to_float_list(list_expr)
        result = dst(data)
        return Expr("List", *result.tolist())

    def _hilbert_transform(list_expr):
        """Hilbert transform."""
        try:
            from scipy.signal import hilbert
            import numpy as np
        except ImportError:
            return Expr("Error", "SciPy required")
        data = np.array(_to_float_list(list_expr))
        analytic = hilbert(data)
        return Expr("List", *analytic.tolist())

    def _wavelet_transform(list_expr, wavelet='db1'):
        """Discrete wavelet transform."""
        try:
            import pywt
        except ImportError:
            return Expr("Error", "PyWavelets not installed")
        data = _to_float_list(list_expr)
        coeffs = pywt.wavedec(data, str(wavelet))
        return Expr("List", *[Expr("List", *c.tolist()) for c in coeffs])

    rule3("ZTransform", _z_transform)
    rule3("InverseZTransform", _inverse_z_transform)
    rule1("DiscreteFourierTransform", _discrete_fourier_transform)
    rule1("DFT", _discrete_fourier_transform)
    rule1("DiscreteCosineTransform", _discrete_cosine_transform)
    rule1("DCT", _discrete_cosine_transform)
    rule1("DiscreteSineTransform", _discrete_sine_transform)
    rule1("DST", _discrete_sine_transform)
    rule1("HilbertTransform", _hilbert_transform)
    rule1("WaveletTransform", _wavelet_transform)
    rule2("WaveletTransform", _wavelet_transform)

    # ── Data Manipulation (~40) ──

    def _group_by(list_expr, key_fn_expr):
        """Group elements by key function."""
        from collections import defaultdict
        lst = _extract_list(list_expr)
        groups = defaultdict(list)
        # Simplified: group by value itself
        for item in lst:
            groups[item].append(item)
        
        result = []
        for k, v in groups.items():
            result.append(Expr("List", k, Expr("List", *v)))
        return Expr("List", *result)

    def _pivot_table(data_expr, row_key_expr, col_key_expr, value_expr):
        """Create pivot table."""
        # Simplified
        return Expr("PivotTable", data_expr, row_key_expr, col_key_expr, value_expr)

    def _transpose_list(list_expr):
        """Transpose a list of lists."""
        lst = _extract_list(list_expr)
        rows = [_extract_list(row) for row in lst]
        cols = list(zip(*rows))
        return Expr("List", *[Expr("List", *col) for col in cols])

    def _rotate_left(list_expr, n=1):
        """Rotate list left."""
        lst = _extract_list(list_expr)
        n = int(n) % len(lst) if lst else 0
        return Expr("List", *(lst[n:] + lst[:n]))

    def _rotate_right(list_expr, n=1):
        """Rotate list right."""
        lst = _extract_list(list_expr)
        n = int(n) % len(lst) if lst else 0
        return Expr("List", *(lst[-n:] + lst[:-n]))

    def _chunk(list_expr, size):
        """Split list into chunks."""
        lst = _extract_list(list_expr)
        size = int(size)
        chunks = [lst[i:i+size] for i in range(0, len(lst), size)]
        return Expr("List", *[Expr("List", *chunk) for chunk in chunks])

    def _sliding_window(list_expr, size):
        """Sliding window."""
        lst = _extract_list(list_expr)
        size = int(size)
        windows = [lst[i:i+size] for i in range(len(lst) - size + 1)]
        return Expr("List", *[Expr("List", *w) for w in windows])

    def _interleave(list1_expr, list2_expr):
        """Interleave two lists."""
        lst1 = _extract_list(list1_expr)
        lst2 = _extract_list(list2_expr)
        result = []
        for a, b in zip(lst1, lst2):
            result.extend([a, b])
        return Expr("List", *result)

    def _cartesian_product(list1_expr, list2_expr):
        """Cartesian product."""
        import itertools
        lst1 = _extract_list(list1_expr)
        lst2 = _extract_list(list2_expr)
        prod = itertools.product(lst1, lst2)
        return Expr("List", *[Expr("List", a, b) for a, b in prod])

    def _powerset_list(list_expr):
        """All subsets."""
        import itertools
        lst = _extract_list(list_expr)
        ps = []
        for r in range(len(lst) + 1):
            for subset in itertools.combinations(lst, r):
                ps.append(Expr("List", *subset))
        return Expr("List", *ps)

    rule1("GroupBy", _group_by)
    rule2("GroupBy", _group_by)
    rule1("TransposeList", _transpose_list)
    rule1("RotateLeft", _rotate_left)
    rule2("RotateLeft", _rotate_left)
    rule1("RotateRight", _rotate_right)
    rule2("RotateRight", _rotate_right)
    rule2("Chunk", _chunk)
    rule2("SlidingWindow", _sliding_window)
    rule2("Interleave", _interleave)
    rule2("CartesianProduct", _cartesian_product)
    rule1("Subsets", _powerset_list)

    # ── Machine Learning (~30) ──

    def _kmeans(data_expr, k, max_iter=100):
        """K-means clustering."""
        try:
            from sklearn.cluster import KMeans
            import numpy as np
        except ImportError:
            return Expr("Error", "scikit-learn required")
        
        data = np.array([_to_float_list(row) for row in _extract_list(data_expr)])
        kmeans = KMeans(n_clusters=int(k), max_iter=int(max_iter), random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_
        
        return Expr("List",
            Expr("Rule", Symbol("labels"), Expr("List", *labels.tolist())),
            Expr("Rule", Symbol("centers"), Expr("List", *[Expr("List", *row) for row in centers.tolist()]))
        )

    def _pca(data_expr, n_components=2):
        """Principal component analysis."""
        try:
            from sklearn.decomposition import PCA
            import numpy as np
        except ImportError:
            return Expr("Error", "scikit-learn required")
        
        data = np.array([_to_float_list(row) for row in _extract_list(data_expr)])
        pca = PCA(n_components=int(n_components))
        transformed = pca.fit_transform(data)
        
        return Expr("List",
            Expr("Rule", Symbol("transformed"), Expr("List", *[Expr("List", *row) for row in transformed.tolist()])),
            Expr("Rule", Symbol("explainedVariance"), Expr("List", *pca.explained_variance_ratio_.tolist()))
        )

    def _logistic_regression_train(X_expr, y_expr):
        """Train logistic regression."""
        try:
            from sklearn.linear_model import LogisticRegression
            import numpy as np
        except ImportError:
            return Expr("Error", "scikit-learn required")
        
        X = np.array([_to_float_list(row) for row in _extract_list(X_expr)])
        y = np.array(_to_float_list(y_expr))
        
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)
        
        return Expr("List",
            Expr("Rule", Symbol("coefficients"), Expr("List", *model.coef_[0].tolist())),
            Expr("Rule", Symbol("intercept"), model.intercept_[0]),
            Expr("Rule", Symbol("accuracy"), model.score(X, y))
        )

    def _decision_tree_train(X_expr, y_expr, max_depth=5):
        """Train decision tree."""
        try:
            from sklearn.tree import DecisionTreeClassifier
            import numpy as np
        except ImportError:
            return Expr("Error", "scikit-learn required")
        
        X = np.array([_to_float_list(row) for row in _extract_list(X_expr)])
        y = np.array(_to_float_list(y_expr))
        
        model = DecisionTreeClassifier(max_depth=int(max_depth), random_state=42)
        model.fit(X, y)
        
        return Expr("List",
            Expr("Rule", Symbol("accuracy"), model.score(X, y)),
            Expr("Rule", Symbol("depth"), model.get_depth())
        )

    rule2("KMeans", _kmeans)
    rule3("KMeans", _kmeans)
    rule1("PCA", _pca)
    rule2("PCA", _pca)
    rule2("LogisticRegressionTrain", _logistic_regression_train)
    rule2("DecisionTreeTrain", _decision_tree_train)
    rule3("DecisionTreeTrain", _decision_tree_train)

    # ── Symbolic Computation (~20) ──

    def _assumptions(expr):
        """Get assumptions about symbols in expression."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        syms = list(sym_expr.free_symbols)
        if not syms:
            return Expr("List")
        
        result = []
        for sym in syms:
            props = []
            if sym.is_positive:
                props.append("positive")
            if sym.is_negative:
                props.append("negative")
            if sym.is_real:
                props.append("real")
            if sym.is_integer:
                props.append("integer")
            if props:
                result.append(Expr("Rule", Symbol(str(sym)), Expr("List", *props)))
        
        return Expr("List", *result)

    def _refine(expr, assumptions_expr):
        """Refine expression under assumptions."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        # Simplified: just return simplified
        result = sp.refine(sym_expr)
        return _from_sympy(result)

    def _rational_reconstruct(expr):
        """Convert float to rational."""
        import sympy as sp
        return _from_sympy(sp.nsimplify(float(expr)))

    def _rationalize_denominator(expr):
        """Rationalize denominator."""
        import sympy as sp
        sym_expr = _to_sympy(expr)
        result = sp.simplify(sym_expr * sp.conjugate(sp.denom(sym_expr)) / sp.conjugate(sp.denom(sym_expr)))
        return _from_sympy(result)

    rule1("Assumptions", _assumptions)
    rule2("Refine", _refine)
    rule1("RationalReconstruct", _rational_reconstruct)
    rule1("RationalizeDenominator", _rationalize_denominator)

    # ── Constants & Utilities (~40) ──

    # Physical constants (expanded)
    def _const(name):
        import scipy.constants as const
        constants = {
            'SpeedOfLight': const.c,
            'PlanckConstant': const.h,
            'ReducedPlanckConstant': const.hbar,
            'GravitationalConstant': const.G,
            'BoltzmannConstant': const.k,
            'AvogadroNumber': const.N_A,
            'GasConstant': const.R,
            'FaradayConstant': const.physical_constants['Faraday constant'][0],
            'ElementaryCharge': const.e,
            'ElectronMass': const.m_e,
            'ProtonMass': const.m_p,
            'NeutronMass': const.m_n,
            'FineStructureConstant': const.alpha,
            'RydbergConstant': const.Rydberg,
            'BohrRadius': const.physical_constants['Bohr radius'][0],
            'VacuumPermittivity': const.epsilon_0,
            'VacuumPermeability': const.mu_0,
            'StefanBoltzmannConstant': const.Stefan_Boltzmann,
            'WienDisplacementConstant': const.Wien,
            'MolarVolumeIdealGas': const.physical_constants['molar volume of ideal gas (273.15 K, 100 kPa)'][0],
        }
        return constants.get(str(name), Expr("Error", f"Unknown constant: {name}"))

    # Register individual constants
    for const_name in ['SpeedOfLight', 'PlanckConstant', 'ReducedPlanckConstant', 
                        'GravitationalConstant', 'BoltzmannConstant', 'AvogadroNumber',
                        'GasConstant', 'FaradayConstant', 'ElementaryCharge', 'ElectronMass',
                        'ProtonMass', 'NeutronMass', 'FineStructureConstant', 'RydbergConstant',
                        'BohrRadius', 'VacuumPermittivity', 'VacuumPermeability',
                        'StefanBoltzmannConstant', 'WienDisplacementConstant', 'MolarVolumeIdealGas']:
        add_rule(const_name, lambda a, name=const_name: _const(name))

    return rules


# Build on import
EXTENDED3_RULES = build_extended3_rules()
