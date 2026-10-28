# numethods

Pure-Python numerical methods toolkit. No runtime dependencies (pytest only for tests). Requires Python 3.9+.

## Install

```bash
pip install -e .          # from this directory
pip install -e ".[test]"  # with pytest
```

## Package code

- [src/numethods/](src/numethods/) — numerical methods listed below.
- [src/engutils/](src/engutils/) — engineering utilities for units, materials, beams, fluids, and heat transfer.
- [pyproject.toml](pyproject.toml) — package metadata, setuptools discovery under `src/`, and the `test` extra.

### numethods modules

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

## Demonstrations and supporting material

The following files are separate from the Python packages in `src/`:

- [Stepwise.html](Stepwise.html) — process planner for breaking work into steps and tracking completion; the embedded code uses browser local storage and JSON backup/restore.
- [radix-bench.html](radix-bench.html) — integer conversion between binary, octal, decimal, and hexadecimal, with selectable bit widths and two's-complement views.
- [examples/demo.py](examples/demo.py) — one example for each `numethods` module; after installation, run `python examples/demo.py` from the repository root.
- [morse.py](morse.py) — standalone Morse/text encoder and decoder with command-line and stdin input; usage and conventions are in its docstring.

Browser compatibility and deployment instructions for the HTML tools are undocumented.

## Tests

[tests/test_numethods.py](tests/test_numethods.py) checks numerical methods against analytic results and selected error cases. [tests/test_engutils.py](tests/test_engutils.py) checks engineering utilities against reference values, formulas, and selected error cases. The test suite contains no tests for the HTML tools or `morse.py`.

```bash
pytest
```
