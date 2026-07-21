"""Initial-value problem solvers for systems y' = f(t, y).

All solvers accept scalar or vector (list) state. They return
``(ts, ys)`` where ``ts`` is a list of times and ``ys`` a list of states.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple, Union

State = Union[float, List[float]]
RHS = Callable[[float, State], State]


def _is_scalar(y) -> bool:
    return not isinstance(y, (list, tuple))


def _axpy(a: float, x: State, y: State) -> State:
    """y + a*x elementwise."""
    if _is_scalar(x):
        return y + a * x
    return [y[i] + a * x[i] for i in range(len(x))]


def _combo(y: State, terms: Sequence[Tuple[float, State]]) -> State:
    """y + sum(c*k for c, k in terms)."""
    out = y
    for c, k in terms:
        out = _axpy(c, k, out)
    return out


def euler(f: RHS, t0: float, y0: State, t_end: float, n_steps: int):
    """Explicit (forward) Euler. First-order accurate: error O(h)."""
    h = (t_end - t0) / n_steps
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for _ in range(n_steps):
        y = _axpy(h, f(t, y), y)
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


def heun(f: RHS, t0: float, y0: State, t_end: float, n_steps: int):
    """Heun's method (explicit trapezoid). Second-order accurate."""
    h = (t_end - t0) / n_steps
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h, _axpy(h, k1, y))
        y = _combo(y, [(h / 2, k1), (h / 2, k2)])
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


def rk4(f: RHS, t0: float, y0: State, t_end: float, n_steps: int):
    """Classic fourth-order Runge-Kutta. Error O(h^4)."""
    h = (t_end - t0) / n_steps
    ts, ys = [t0], [y0]
    t, y = t0, y0
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h / 2, _axpy(h / 2, k1, y))
        k3 = f(t + h / 2, _axpy(h / 2, k2, y))
        k4 = f(t + h, _axpy(h, k3, y))
        y = _combo(y, [(h / 6, k1), (h / 3, k2), (h / 3, k3), (h / 6, k4)])
        t += h
        ts.append(t)
        ys.append(y)
    return ts, ys


def rkf45(
    f: RHS,
    t0: float,
    y0: State,
    t_end: float,
    tol: float = 1e-8,
    h0: float = 0.01,
    h_min: float = 1e-12,
    h_max: float | None = None,
):
    """Adaptive Runge-Kutta-Fehlberg 4(5) with step-size control.

    Steps are accepted when the embedded error estimate is below ``tol``.
    """
    if h_max is None:
        h_max = (t_end - t0) / 4 or 1.0

    # Fehlberg coefficients
    A = [
        [],
        [1 / 4],
        [3 / 32, 9 / 32],
        [1932 / 2197, -7200 / 2197, 7296 / 2197],
        [439 / 216, -8, 3680 / 513, -845 / 4104],
        [-8 / 27, 2, -3544 / 2565, 1859 / 4104, -11 / 40],
    ]
    C = [0, 1 / 4, 3 / 8, 12 / 13, 1, 1 / 2]
    B4 = [25 / 216, 0, 1408 / 2565, 2197 / 4104, -1 / 5, 0]
    B5 = [16 / 135, 0, 6656 / 12825, 28561 / 56430, -9 / 50, 2 / 55]

    ts, ys = [t0], [y0]
    t, y, h = t0, y0, h0
    while t < t_end:
        h = min(h, t_end - t, h_max)
        ks = []
        for i in range(6):
            yi = _combo(y, [(h * A[i][j], ks[j]) for j in range(i)])
            ks.append(f(t + C[i] * h, yi))
        y4 = _combo(y, [(h * B4[i], ks[i]) for i in range(6)])
        y5 = _combo(y, [(h * B5[i], ks[i]) for i in range(6)])
        if _is_scalar(y4):
            err = abs(y5 - y4)
        else:
            err = max(abs(y5[i] - y4[i]) for i in range(len(y4)))
        if err <= tol or h <= h_min:
            t += h
            y = y5
            ts.append(t)
            ys.append(y)
        # step-size update (safety factor 0.9)
        scale = 0.9 * (tol / err) ** 0.2 if err > 0 else 2.0
        h = max(h_min, min(h * min(max(scale, 0.1), 4.0), h_max))
    return ts, ys


def solve_ivp(f: RHS, t_span, y0: State, method: str = "rk4", **kwargs):
    """Convenience dispatcher.

    method in {"euler", "heun", "rk4"} takes ``n_steps`` (default 1000);
    method "rkf45" takes ``tol`` and step-size options.
    """
    t0, t_end = t_span
    if method == "rkf45":
        return rkf45(f, t0, y0, t_end, **kwargs)
    n_steps = kwargs.pop("n_steps", 1000)
    solver = {"euler": euler, "heun": heun, "rk4": rk4}[method]
    return solver(f, t0, y0, t_end, n_steps)
