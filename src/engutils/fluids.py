"""Incompressible pipe flow and external flow utilities (SI units).

Water at 20 C: rho = 998.2 kg/m^3, mu = 1.002e-3 Pa*s.
Air at 20 C, 1 atm: rho = 1.204 kg/m^3, mu = 1.825e-5 Pa*s.
"""

from __future__ import annotations

import math

# convenient property constants
WATER_20C = {"rho": 998.2, "mu": 1.002e-3}
AIR_20C = {"rho": 1.204, "mu": 1.825e-5}
G = 9.80665  # m/s^2


def reynolds(rho: float, v: float, L: float, mu: float) -> float:
    """Reynolds number Re = rho*v*L/mu (L = pipe diameter for internal flow)."""
    return rho * v * L / mu


def flow_regime(re: float) -> str:
    """Classify internal pipe flow: laminar (<2300), transitional, turbulent (>4000)."""
    if re < 2300:
        return "laminar"
    if re < 4000:
        return "transitional"
    return "turbulent"


def friction_factor(re: float, rel_roughness: float = 0.0) -> float:
    """Darcy friction factor.

    Laminar: 64/Re. Turbulent: Swamee-Jain explicit approximation of the
    Colebrook equation (accurate to ~1% for 5e3 < Re < 1e8,
    1e-6 < eps/D < 1e-2).
    """
    if re <= 0:
        raise ValueError("Re must be positive")
    if re < 2300:
        return 64.0 / re
    return 0.25 / math.log10(rel_roughness / 3.7 + 5.74 / re ** 0.9) ** 2


def pressure_drop_pipe(
    rho: float, v: float, L: float, D: float, mu: float, roughness: float = 0.0
) -> float:
    """Darcy-Weisbach pressure drop (Pa) in a straight pipe.

    Parameters: density, mean velocity, pipe length, diameter, dynamic
    viscosity, absolute roughness (m).
    """
    re = reynolds(rho, v, D, mu)
    f = friction_factor(re, roughness / D)
    return f * (L / D) * 0.5 * rho * v * v


def velocity_from_flow(Q: float, D: float) -> float:
    """Mean velocity (m/s) from volumetric flow Q (m^3/s) in circular pipe."""
    return Q / (math.pi * D * D / 4)


def bernoulli_pressure(
    p1: float, v1: float, z1: float, v2: float, z2: float, rho: float, h_loss: float = 0.0
) -> float:
    """Solve Bernoulli (with optional head loss, m) for downstream pressure p2."""
    return p1 + 0.5 * rho * (v1 ** 2 - v2 ** 2) + rho * G * (z1 - z2) - rho * G * h_loss


def drag_force(rho: float, v: float, cd: float, area: float) -> float:
    """Aerodynamic/hydrodynamic drag F = 0.5*rho*v^2*Cd*A."""
    return 0.5 * rho * v * v * cd * area


def dynamic_pressure(rho: float, v: float) -> float:
    """q = 0.5*rho*v^2 (Pa)."""
    return 0.5 * rho * v * v


def hydraulic_diameter(area: float, wetted_perimeter: float) -> float:
    """D_h = 4A/P for non-circular ducts."""
    return 4.0 * area / wetted_perimeter


def orifice_flow(cd: float, area: float, dp: float, rho: float) -> float:
    """Volumetric flow (m^3/s) through an orifice: Q = Cd*A*sqrt(2*dp/rho)."""
    if dp < 0:
        raise ValueError("pressure drop must be non-negative")
    return cd * area * math.sqrt(2.0 * dp / rho)


def pump_power(Q: float, dp: float, efficiency: float = 1.0) -> float:
    """Shaft power (W) to pump flow Q (m^3/s) against pressure rise dp (Pa)."""
    if not 0 < efficiency <= 1:
        raise ValueError("efficiency must be in (0, 1]")
    return Q * dp / efficiency
