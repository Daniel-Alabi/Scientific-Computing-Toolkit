"""Tests for numethods, checked against analytic solutions."""

import math

import pytest

from numethods import roots, linalg, ode, integrate, interpolate, optimize


# ------------------------------------------------------------------ roots
class TestRoots:
    def test_bisection_sqrt2(self):
        r = roots.bisection(lambda x: x * x - 2, 0, 2)
        assert abs(r - math.sqrt(2)) < 1e-9

    def test_newton_cubic(self):
        r = roots.newton(lambda x: x ** 3 - x - 2, 1.5, dfdx=lambda x: 3 * x * x - 1)
        assert abs(r ** 3 - r - 2) < 1e-10

    def test_newton_numeric_derivative(self):
        r = roots.newton(lambda x: math.cos(x) - x, 0.5)
        assert abs(math.cos(r) - r) < 1e-10

    def test_secant(self):
        r = roots.secant(lambda x: math.exp(x) - 3, 0, 2)
        assert abs(r - math.log(3)) < 1e-10

    def test_fixed_point(self):
        r = roots.fixed_point(lambda x: math.cos(x), 1.0)
        assert abs(math.cos(r) - r) < 1e-10

    def test_brent(self):
        r = roots.brent(lambda x: x ** 3 - 2 * x - 5, 2, 3)
        assert abs(r ** 3 - 2 * r - 5) < 1e-9

    def test_bisection_bad_bracket(self):
        with pytest.raises(ValueError):
            roots.bisection(lambda x: x * x + 1, -1, 1)


# ----------------------------------------------------------------- linalg
class TestLinalg:
    A = [[4.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 4.0]]
    b = [11.0, -16.0, 17.0]

    def test_gauss_solve(self):
        x = linalg.gauss_solve(self.A, self.b)
        r = linalg.matvec(self.A, x)
        assert all(abs(r[i] - self.b[i]) < 1e-10 for i in range(3))

    def test_lu_solve(self):
        L, U, perm = linalg.lu_decompose(self.A)
        x = linalg.lu_solve(L, U, perm, self.b)
        r = linalg.matvec(self.A, x)
        assert all(abs(r[i] - self.b[i]) < 1e-10 for i in range(3))

    def test_jacobi_matches_direct(self):
        x_direct = linalg.gauss_solve(self.A, self.b)
        x_jac = linalg.jacobi(self.A, self.b, tol=1e-12)
        assert all(abs(x_direct[i] - x_jac[i]) < 1e-8 for i in range(3))

    def test_gauss_seidel_matches_direct(self):
        x_direct = linalg.gauss_solve(self.A, self.b)
        x_gs = linalg.gauss_seidel(self.A, self.b, tol=1e-12)
        assert all(abs(x_direct[i] - x_gs[i]) < 1e-8 for i in range(3))

    def test_determinant_2x2(self):
        assert abs(linalg.determinant([[1.0, 2.0], [3.0, 4.0]]) - (-2.0)) < 1e-12

    def test_determinant_singular(self):
        assert abs(linalg.determinant([[1.0, 2.0], [2.0, 4.0]])) < 1e-12

    def test_matmul_identity(self):
        I = [[1.0, 0.0], [0.0, 1.0]]
        M = [[5.0, 7.0], [-3.0, 2.0]]
        assert linalg.matmul(I, M) == M

    def test_norms(self):
        v = [3.0, -4.0]
        assert abs(linalg.norm(v, 2) - 5.0) < 1e-12
        assert abs(linalg.norm(v, 1) - 7.0) < 1e-12
        assert linalg.norm(v, float("inf")) == 4.0

    def test_singular_raises(self):
        with pytest.raises(ValueError):
            linalg.gauss_solve([[1.0, 2.0], [2.0, 4.0]], [1.0, 2.0])


# -------------------------------------------------------------------- ode
class TestODE:
    """y' = -y, y(0)=1 -> y(t) = exp(-t)."""

    def test_euler_first_order(self):
        _, ys = ode.euler(lambda t, y: -y, 0, 1.0, 1.0, 1000)
        assert abs(ys[-1] - math.exp(-1)) < 1e-3

    def test_rk4_accuracy(self):
        _, ys = ode.rk4(lambda t, y: -y, 0, 1.0, 1.0, 100)
        assert abs(ys[-1] - math.exp(-1)) < 1e-9

    def test_heun_between(self):
        _, ys = ode.heun(lambda t, y: -y, 0, 1.0, 1.0, 100)
        assert abs(ys[-1] - math.exp(-1)) < 1e-4

    def test_rkf45_adaptive(self):
        ts, ys = ode.rkf45(lambda t, y: -y, 0, 1.0, 1.0, tol=1e-10)
        assert abs(ts[-1] - 1.0) < 1e-12
        assert abs(ys[-1] - math.exp(-1)) < 1e-7

    def test_system_harmonic_oscillator(self):
        # y'' = -y as system; y(0)=1, y'(0)=0 -> y(t)=cos(t)
        f = lambda t, s: [s[1], -s[0]]
        _, ys = ode.rk4(f, 0, [1.0, 0.0], math.pi, 200)
        assert abs(ys[-1][0] - (-1.0)) < 1e-8

    def test_solve_ivp_dispatch(self):
        _, ys = ode.solve_ivp(lambda t, y: -y, (0, 1), 1.0, method="rk4", n_steps=100)
        assert abs(ys[-1] - math.exp(-1)) < 1e-9


# -------------------------------------------------------------- integrate
class TestIntegrate:
    def test_trapezoid_linear_exact(self):
        assert abs(integrate.trapezoid(lambda x: 2 * x + 1, 0, 1, 10) - 2.0) < 1e-12

    def test_simpson_cubic_exact(self):
        # Simpson is exact for cubics
        assert abs(integrate.simpson(lambda x: x ** 3, 0, 2, 2) - 4.0) < 1e-12

    def test_gauss_legendre_quintic(self):
        # 3-point GL exact through degree 5
        val = integrate.gauss_legendre(lambda x: x ** 5, 0, 1, n_points=3)
        assert abs(val - 1 / 6) < 1e-13

    def test_romberg_sin(self):
        assert abs(integrate.romberg(math.sin, 0, math.pi) - 2.0) < 1e-10

    def test_adaptive_simpson(self):
        val = integrate.adaptive_simpson(lambda x: math.exp(-x * x), 0, 1, tol=1e-12)
        assert abs(val - 0.7468241328124271) < 1e-10

    def test_differentiate(self):
        assert abs(integrate.differentiate(math.sin, 0.0) - 1.0) < 1e-8
        assert abs(integrate.differentiate(math.sin, 0.0, order=2)) < 1e-4


# ------------------------------------------------------------ interpolate
class TestInterpolate:
    def test_lagrange_recovers_quadratic(self):
        p = interpolate.lagrange([0, 1, 2], [1, 2, 5])  # 1 + x^2? p(x)=x^2+1 fits
        assert abs(p(1.5) - (1.5 ** 2 + 1)) < 1e-12

    def test_newton_divided_matches_lagrange(self):
        xs, ys = [0, 1, 3, 4], [1, 3, 2, 5]
        pl = interpolate.lagrange(xs, ys)
        pn = interpolate.newton_divided(xs, ys)
        for x in [0.5, 2.0, 3.7]:
            assert abs(pl(x) - pn(x)) < 1e-10

    def test_spline_interpolates_knots(self):
        xs = [0, 1, 2, 3]
        ys = [0, 1, 4, 9]
        s = interpolate.CubicSpline(xs, ys)
        for x, y in zip(xs, ys):
            assert abs(s(x) - y) < 1e-12

    def test_spline_smooth_sine(self):
        xs = [i * math.pi / 8 for i in range(9)]
        s = interpolate.CubicSpline(xs, [math.sin(x) for x in xs])
        assert abs(s(1.0) - math.sin(1.0)) < 1e-3

    def test_polyfit_exact_quadratic(self):
        xs = [-2, -1, 0, 1, 2]
        ys = [4 - 2 * x + 3 * x * x for x in xs]
        c = interpolate.polyfit(xs, ys, 2)
        assert abs(c[0] - 4) < 1e-8 and abs(c[1] + 2) < 1e-8 and abs(c[2] - 3) < 1e-8

    def test_linear_regression(self):
        xs = [0, 1, 2, 3, 4]
        ys = [1, 3, 5, 7, 9]  # y = 1 + 2x
        a, b, r2 = interpolate.linear_regression(xs, ys)
        assert abs(a - 1) < 1e-12 and abs(b - 2) < 1e-12 and abs(r2 - 1) < 1e-12


# --------------------------------------------------------------- optimize
class TestOptimize:
    def test_golden_section_parabola(self):
        x = optimize.golden_section(lambda x: (x - 2.5) ** 2, 0, 5)
        assert abs(x - 2.5) < 1e-7

    def test_gradient_descent_quadratic(self):
        f = lambda p: (p[0] - 1) ** 2 + 2 * (p[1] + 3) ** 2
        x = optimize.gradient_descent(f, [0, 0], lr=0.1)
        assert abs(x[0] - 1) < 1e-4 and abs(x[1] + 3) < 1e-4

    def test_nelder_mead_rosenbrock(self):
        f = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
        x = optimize.nelder_mead(f, [-1.2, 1.0], tol=1e-14, max_iter=20_000)
        assert abs(x[0] - 1) < 1e-3 and abs(x[1] - 1) < 1e-3

    def test_minimize_scalar(self):
        x = optimize.minimize_scalar(lambda x: math.cos(x), (0, 2 * math.pi))
        assert abs(x - math.pi) < 1e-6
