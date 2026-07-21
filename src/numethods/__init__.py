"""numethods: pure-Python numerical methods toolkit.

Modules
-------
roots        : bisection, newton, secant, fixed-point
linalg       : Gaussian elimination, LU, Jacobi, Gauss-Seidel, norms
ode          : euler, heun, rk4, adaptive RKF45
integrate    : trapezoid, simpson, gauss_legendre, romberg
interpolate  : Lagrange, Newton divided differences, cubic splines, polyfit
optimize     : golden section, Nelder-Mead (1D/ND), gradient descent
"""

from . import roots, linalg, ode, integrate, interpolate, optimize

__version__ = "0.1.0"
__all__ = ["roots", "linalg", "ode", "integrate", "interpolate", "optimize"]
