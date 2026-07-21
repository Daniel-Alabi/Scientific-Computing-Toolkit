"""Root-finding algorithms for scalar nonlinear equations f(x) = 0."""

from __future__ import annotations

from typing import Callable, Optional


class ConvergenceError(RuntimeError):
    """Raised when an iterative method fails to converge."""


def bisection(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """Find a root of f in [a, b] by bisection.

    Requires f(a) and f(b) to have opposite signs. Converges linearly
    but is unconditionally robust.

    Parameters
    ----------
    f : callable
        Continuous function of one variable.
    a, b : float
        Bracketing interval endpoints, f(a)*f(b) < 0.
    tol : float
        Absolute tolerance on the interval half-width.
    max_iter : int
        Maximum number of bisections.

    Returns
    -------
    float
        Approximate root.
    """
    fa, fb = f(a), f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if fm == 0.0 or 0.5 * (b - a) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    raise ConvergenceError(f"bisection did not converge in {max_iter} iterations")


def newton(
    f: Callable[[float], float],
    x0: float,
    dfdx: Optional[Callable[[float], float]] = None,
    tol: float = 1e-12,
    max_iter: int = 100,
    h: float = 1e-7,
) -> float:
    """Newton-Raphson iteration.

    Quadratic convergence near a simple root. If ``dfdx`` is omitted, the
    derivative is approximated by a central finite difference with step ``h``.
    """
    x = float(x0)
    for _ in range(max_iter):
        fx = f(x)
        d = dfdx(x) if dfdx is not None else (f(x + h) - f(x - h)) / (2 * h)
        if d == 0:
            raise ZeroDivisionError("derivative vanished during Newton iteration")
        step = fx / d
        x -= step
        if abs(step) < tol:
            return x
    raise ConvergenceError(f"newton did not converge in {max_iter} iterations")


def secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    tol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """Secant method: superlinear convergence, no derivative required."""
    f0, f1 = f(x0), f(x1)
    for _ in range(max_iter):
        if f1 == f0:
            raise ZeroDivisionError("flat secant encountered")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            return x2
        x0, f0, x1, f1 = x1, f1, x2, f(x2)
    raise ConvergenceError(f"secant did not converge in {max_iter} iterations")


def fixed_point(
    g: Callable[[float], float],
    x0: float,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> float:
    """Fixed-point iteration x_{k+1} = g(x_k).

    Converges when |g'(x)| < 1 near the fixed point.
    """
    x = float(x0)
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    raise ConvergenceError(f"fixed_point did not converge in {max_iter} iterations")


def brent(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Brent's method: combines bisection, secant, and inverse quadratic
    interpolation. Robust like bisection, fast like secant."""
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have opposite signs")
    if abs(fa) < abs(fb):
        a, b, fa, fb = b, a, fb, fa
    c, fc = a, fa
    mflag = True
    d = c
    for _ in range(max_iter):
        if fb == 0 or abs(b - a) < tol:
            return b
        if fa != fc and fb != fc:  # inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:  # secant
            s = b - fb * (b - a) / (fb - fa)
        cond = (
            not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
            or (mflag and abs(s - b) >= abs(b - c) / 2)
            or (not mflag and abs(s - b) >= abs(c - d) / 2)
            or (mflag and abs(b - c) < tol)
            or (not mflag and abs(c - d) < tol)
        )
        if cond:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d, c, fc = c, b, fb
        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        if abs(fa) < abs(fb):
            a, b, fa, fb = b, a, fb, fa
    raise ConvergenceError(f"brent did not converge in {max_iter} iterations")
