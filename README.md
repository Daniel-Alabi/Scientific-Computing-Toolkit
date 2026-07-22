# numethods

Pure-Python numerical methods toolkit. No dependencies (pytest only for tests).

## Install

```bash
pip install -e .          # from this directory
pip install -e ".[test]"  # with pytest
```

## Modules

| Module | Contents |
|---|---|
| `roots` | bisection, newton, secant, fixed_point, brent |
| `linalg` | gauss_solve, lu_decompose/lu_solve, jacobi, gauss_seidel, determinant, matmul, norms |
| `ode` | euler, heun, rk4, adaptive rkf45, solve_ivp dispatcher |
| `integrate` | trapezoid, simpson, romberg, gauss_legendre, adaptive_simpson, differentiate |
| `interpolate` | lagrange, newton_divided, CubicSpline, polyfit, linear_regression |
| `optimize` | golden_section, gradient_descent, nelder_mead, minimize_scalar |

## Quick start

```python
from numethods import roots, ode, integrate

# root of x^2 = 2
r = roots.brent(lambda x: x*x - 2, 0, 2)

# integrate sin from 0 to pi
import math
area = integrate.adaptive_simpson(math.sin, 0, math.pi)

# solve y' = -y, y(0) = 1
ts, ys = ode.solve_ivp(lambda t, y: -y, (0, 5), 1.0, method="rkf45", tol=1e-9)
```

ODE solvers accept vector state as plain lists, e.g. `f = lambda t, s: [s[1], -s[0]]` for a harmonic oscillator.

## Tests

```bash
pytest
```
