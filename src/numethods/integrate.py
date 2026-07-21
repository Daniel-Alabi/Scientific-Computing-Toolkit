"""Numerical quadrature: definite integrals of f on [a, b]."""

from __future__ import annotations

import math
from typing import Callable


def trapezoid(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Composite trapezoid rule with n subintervals. Error O(h^2)."""
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b)) + sum(f(a + i * h) for i in range(1, n))
    return s * h


def simpson(f: Callable[[float], float], a: float, b: float, n: int = 1000) -> float:
    """Composite Simpson's rule (n must be even). Error O(h^4)."""
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    s += 4 * sum(f(a + i * h) for i in range(1, n, 2))
    s += 2 * sum(f(a + i * h) for i in range(2, n, 2))
    return s * h / 3


def romberg(f: Callable[[float], float], a: float, b: float, tol: float = 1e-12, max_levels: int = 20) -> float:
    """Romberg integration: Richardson-extrapolated trapezoid rule."""
    R = [[0.5 * (b - a) * (f(a) + f(b))]]
    for k in range(1, max_levels):
        n = 2 ** k
        h = (b - a) / n
        # trapezoid refinement using only new points
        new = 0.5 * R[k - 1][0] + h * sum(f(a + (2 * i - 1) * h) for i in range(1, n // 2 + 1))
        row = [new]
        for j in range(1, k + 1):
            row.append(row[j - 1] + (row[j - 1] - R[k - 1][j - 1]) / (4 ** j - 1))
        R.append(row)
        if abs(row[-1] - R[k - 1][-1]) < tol:
            return row[-1]
    return R[-1][-1]


# Nodes/weights for Gauss-Legendre on [-1, 1], n = 2..5
_GL = {
    2: ([-0.5773502691896257, 0.5773502691896257], [1.0, 1.0]),
    3: ([-0.7745966692414834, 0.0, 0.7745966692414834],
        [5 / 9, 8 / 9, 5 / 9]),
    4: ([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526],
        [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]),
    5: ([-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640],
        [0.2369268850561891, 0.4786286704993665, 128 / 225, 0.4786286704993665, 0.2369268850561891]),
}


def gauss_legendre(
    f: Callable[[float], float], a: float, b: float, n_points: int = 5, n_panels: int = 1
) -> float:
    """Composite Gauss-Legendre quadrature (2-5 points per panel).

    Exact for polynomials of degree <= 2*n_points - 1 on each panel.
    """
    if n_points not in _GL:
        raise ValueError("n_points must be 2, 3, 4, or 5")
    xs, ws = _GL[n_points]
    total = 0.0
    width = (b - a) / n_panels
    for p in range(n_panels):
        lo = a + p * width
        mid, half = lo + width / 2, width / 2
        total += half * sum(w * f(mid + half * x) for x, w in zip(xs, ws))
    return total


def adaptive_simpson(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-10, max_depth: int = 50
) -> float:
    """Adaptive Simpson quadrature with recursive interval refinement."""

    def _simpson(fa, fm, fb, a_, b_):
        return (b_ - a_) / 6 * (fa + 4 * fm + fb)

    def _recurse(a_, b_, fa, fm, fb, whole, tol_, depth):
        m = 0.5 * (a_ + b_)
        lm, rm = 0.5 * (a_ + m), 0.5 * (m + b_)
        flm, frm = f(lm), f(rm)
        left = _simpson(fa, flm, fm, a_, m)
        right = _simpson(fm, frm, fb, m, b_)
        if depth >= max_depth or abs(left + right - whole) <= 15 * tol_:
            return left + right + (left + right - whole) / 15
        return _recurse(a_, m, fa, flm, fm, left, tol_ / 2, depth + 1) + _recurse(
            m, b_, fm, frm, fb, right, tol_ / 2, depth + 1
        )

    fa, fb = f(a), f(b)
    m = 0.5 * (a + b)
    fm = f(m)
    return _recurse(a, b, fa, fm, fb, _simpson(fa, fm, fb, a, b), tol, 0)


def differentiate(f: Callable[[float], float], x: float, h: float = 1e-6, order: int = 1) -> float:
    """Central finite-difference derivative (order 1 or 2)."""
    if order == 1:
        return (f(x + h) - f(x - h)) / (2 * h)
    if order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h)
    raise ValueError("order must be 1 or 2")
