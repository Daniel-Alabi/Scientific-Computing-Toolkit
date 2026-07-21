"""Interpolation and curve fitting."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple


def lagrange(xs: Sequence[float], ys: Sequence[float]) -> Callable[[float], float]:
    """Return the Lagrange interpolating polynomial through (xs, ys)."""
    xs, ys = list(xs), list(ys)
    n = len(xs)

    def p(x: float) -> float:
        total = 0.0
        for i in range(n):
            li = 1.0
            for j in range(n):
                if j != i:
                    li *= (x - xs[j]) / (xs[i] - xs[j])
            total += ys[i] * li
        return total

    return p


def newton_divided(xs: Sequence[float], ys: Sequence[float]) -> Callable[[float], float]:
    """Newton divided-difference interpolation (efficient repeated evaluation)."""
    xs = list(xs)
    n = len(xs)
    coef = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coef[i] = (coef[i] - coef[i - 1]) / (xs[i] - xs[i - j])

    def p(x: float) -> float:
        result = coef[-1]
        for i in range(n - 2, -1, -1):
            result = result * (x - xs[i]) + coef[i]
        return result

    return p


class CubicSpline:
    """Natural cubic spline interpolant.

    C2-continuous piecewise cubic with zero second derivative at endpoints.
    Call the instance like a function: ``s(x)``.
    """

    def __init__(self, xs: Sequence[float], ys: Sequence[float]):
        xs, ys = list(map(float, xs)), list(map(float, ys))
        if len(xs) != len(ys) or len(xs) < 3:
            raise ValueError("need at least 3 points")
        if any(xs[i + 1] <= xs[i] for i in range(len(xs) - 1)):
            raise ValueError("xs must be strictly increasing")
        n = len(xs) - 1
        h = [xs[i + 1] - xs[i] for i in range(n)]
        # solve tridiagonal system for second derivatives (natural BCs)
        alpha = [0.0] * (n + 1)
        for i in range(1, n):
            alpha[i] = 3 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1])
        l = [1.0] * (n + 1)
        mu = [0.0] * (n + 1)
        z = [0.0] * (n + 1)
        for i in range(1, n):
            l[i] = 2 * (xs[i + 1] - xs[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
        c = [0.0] * (n + 1)
        b = [0.0] * n
        d = [0.0] * n
        for i in range(n - 1, -1, -1):
            c[i] = z[i] - mu[i] * c[i + 1]
            b[i] = (ys[i + 1] - ys[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
            d[i] = (c[i + 1] - c[i]) / (3 * h[i])
        self.xs, self.ys = xs, ys
        self.b, self.c, self.d = b, c[:-1], d

    def __call__(self, x: float) -> float:
        xs = self.xs
        if x <= xs[0]:
            i = 0
        elif x >= xs[-1]:
            i = len(xs) - 2
        else:  # binary search
            lo, hi = 0, len(xs) - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if xs[mid] <= x:
                    lo = mid
                else:
                    hi = mid
            i = lo
        dx = x - xs[i]
        return self.ys[i] + dx * (self.b[i] + dx * (self.c[i] + dx * self.d[i]))


def polyfit(xs: Sequence[float], ys: Sequence[float], degree: int) -> List[float]:
    """Least-squares polynomial fit; returns coefficients [c0, c1, ...] for
    p(x) = c0 + c1*x + ... solved via normal equations."""
    from .linalg import gauss_solve

    n = degree + 1
    # normal equations: (V^T V) c = V^T y
    A = [[sum(x ** (i + j) for x in xs) for j in range(n)] for i in range(n)]
    b = [sum(ys[k] * xs[k] ** i for k in range(len(xs))) for i in range(n)]
    return gauss_solve(A, b)


def polyval(coeffs: Sequence[float], x: float) -> float:
    """Evaluate polynomial with coefficients [c0, c1, ...] at x (Horner)."""
    result = 0.0
    for c in reversed(list(coeffs)):
        result = result * x + c
    return result


def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, float]:
    """Ordinary least squares line y = a + b*x.

    Returns (intercept, slope, r_squared).
    """
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        raise ValueError("xs are all identical")
    b = sxy / sxx
    a = my - b * mx
    r2 = (sxy * sxy) / (sxx * syy) if syy > 0 else 1.0
    return a, b, r2
