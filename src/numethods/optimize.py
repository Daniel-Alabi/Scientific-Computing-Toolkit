"""Optimization: 1-D line methods and derivative-free/gradient ND methods."""

from __future__ import annotations

import math
from typing import Callable, List, Sequence, Tuple

_GOLDEN = (math.sqrt(5) - 1) / 2  # ~0.618


def golden_section(
    f: Callable[[float], float], a: float, b: float, tol: float = 1e-10, max_iter: int = 500
) -> float:
    """Minimize unimodal f on [a, b] by golden-section search."""
    x1 = b - _GOLDEN * (b - a)
    x2 = a + _GOLDEN * (b - a)
    f1, f2 = f(x1), f(x2)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if f1 > f2:
            a, x1, f1 = x1, x2, f2
            x2 = a + _GOLDEN * (b - a)
            f2 = f(x2)
        else:
            b, x2, f2 = x2, x1, f1
            x1 = b - _GOLDEN * (b - a)
            f1 = f(x1)
    return 0.5 * (a + b)


def gradient_descent(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    grad: Callable[[Sequence[float]], List[float]] | None = None,
    lr: float = 0.01,
    tol: float = 1e-8,
    max_iter: int = 50_000,
    h: float = 1e-7,
) -> List[float]:
    """Gradient descent with optional numerical gradient and backtracking.

    Parameters
    ----------
    grad : callable, optional
        Analytic gradient; approximated by central differences if omitted.
    lr : float
        Initial step size (backtracked by halving when the step increases f).
    """
    x = list(map(float, x0))
    n = len(x)

    def num_grad(p: List[float]) -> List[float]:
        g = []
        for i in range(n):
            q1, q2 = p[:], p[:]
            q1[i] += h
            q2[i] -= h
            g.append((f(q1) - f(q2)) / (2 * h))
        return g

    g_fn = grad if grad is not None else num_grad
    fx = f(x)
    for _ in range(max_iter):
        g = list(g_fn(x))
        gnorm = math.sqrt(sum(v * v for v in g))
        if gnorm < tol:
            return x
        step = lr
        while step > 1e-16:
            x_new = [x[i] - step * g[i] for i in range(n)]
            f_new = f(x_new)
            if f_new < fx:
                break
            step /= 2
        else:
            return x  # cannot make progress
        x, fx = x_new, f_new
    return x


def nelder_mead(
    f: Callable[[Sequence[float]], float],
    x0: Sequence[float],
    tol: float = 1e-10,
    max_iter: int = 10_000,
    initial_step: float = 0.5,
) -> List[float]:
    """Nelder-Mead downhill simplex (derivative-free) minimization."""
    n = len(x0)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    # build initial simplex
    simplex = [list(map(float, x0))]
    for i in range(n):
        p = list(map(float, x0))
        p[i] += initial_step if p[i] == 0 else initial_step * abs(p[i])
        simplex.append(p)
    fvals = [f(p) for p in simplex]

    for _ in range(max_iter):
        order = sorted(range(n + 1), key=lambda i: fvals[i])
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            return simplex[0]
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        # reflection
        xr = [centroid[j] + alpha * (centroid[j] - simplex[-1][j]) for j in range(n)]
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[0]:  # expansion
            xe = [centroid[j] + gamma * (xr[j] - centroid[j]) for j in range(n)]
            fe = f(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        else:  # contraction
            xc = [centroid[j] + rho * (simplex[-1][j] - centroid[j]) for j in range(n)]
            fc = f(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:  # shrink
                for i in range(1, n + 1):
                    simplex[i] = [
                        simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)
                    ]
                    fvals[i] = f(simplex[i])
    return simplex[0]


def minimize_scalar(
    f: Callable[[float], float], bracket: Tuple[float, float], tol: float = 1e-10
) -> float:
    """Convenience wrapper: golden-section minimize over a bracket."""
    return golden_section(f, bracket[0], bracket[1], tol=tol)
