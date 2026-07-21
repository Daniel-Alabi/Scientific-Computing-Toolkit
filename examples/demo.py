"""numethods demo: one example per module."""

import math

from numethods import roots, linalg, ode, integrate, interpolate, optimize

print("=== roots: solve x^3 - x - 2 = 0 ===")
r = roots.brent(lambda x: x ** 3 - x - 2, 1, 2)
print(f"  root = {r:.12f}  (check f(r) = {r**3 - r - 2:.2e})")

print("\n=== linalg: solve 3x3 system ===")
A = [[4.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 4.0]]
b = [11.0, -16.0, 17.0]
x = linalg.gauss_solve(A, b)
print(f"  x = {[f'{v:.6f}' for v in x]}")
print(f"  det(A) = {linalg.determinant(A):.4f}")

print("\n=== ode: damped oscillator y'' + 0.2y' + y = 0 ===")
f = lambda t, s: [s[1], -0.2 * s[1] - s[0]]
ts, ys = ode.rkf45(f, 0, [1.0, 0.0], 20.0, tol=1e-9)
print(f"  {len(ts)} adaptive steps; y(20) = {ys[-1][0]:+.6f}")

print("\n=== integrate: erf-style integral ===")
val = integrate.adaptive_simpson(lambda x: math.exp(-x * x), 0, 1, tol=1e-12)
print(f"  int_0^1 exp(-x^2) dx = {val:.12f}")

print("\n=== interpolate: cubic spline through sin samples ===")
xs = [i * math.pi / 6 for i in range(7)]
s = interpolate.CubicSpline(xs, [math.sin(v) for v in xs])
print(f"  spline(1.0) = {s(1.0):.6f}  vs  sin(1.0) = {math.sin(1.0):.6f}")

print("\n=== optimize: Rosenbrock minimum via Nelder-Mead ===")
rosen = lambda p: (1 - p[0]) ** 2 + 100 * (p[1] - p[0] ** 2) ** 2
xmin = optimize.nelder_mead(rosen, [-1.2, 1.0], tol=1e-14, max_iter=20000)
print(f"  minimum at ({xmin[0]:.6f}, {xmin[1]:.6f})  (exact: (1, 1))")
