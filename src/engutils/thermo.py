"""Heat transfer and basic thermodynamics (SI units)."""

from __future__ import annotations

import math

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m^2 K^4)
R_UNIVERSAL = 8.314462618          # J/(mol K)
R_AIR = 287.052874                 # J/(kg K)


# ------------------------------------------------------------- conduction
def conduction_1d(k: float, A: float, dT: float, L: float) -> float:
    """Steady 1-D conduction through a plane wall: Q = kA dT / L (W)."""
    return k * A * dT / L


def conduction_cylindrical(k: float, length: float, r_in: float, r_out: float, dT: float) -> float:
    """Radial conduction through a cylinder wall (W)."""
    if r_out <= r_in:
        raise ValueError("r_out must exceed r_in")
    return 2 * math.pi * k * length * dT / math.log(r_out / r_in)


def thermal_resistance_wall(L: float, k: float, A: float) -> float:
    """R = L/(kA), K/W."""
    return L / (k * A)


def thermal_resistance_convection(h: float, A: float) -> float:
    """R = 1/(hA), K/W."""
    return 1.0 / (h * A)


def composite_wall_heat(dT: float, resistances) -> float:
    """Heat rate through series thermal resistances: Q = dT / sum(R)."""
    return dT / sum(resistances)


# ------------------------------------------------------------- convection
def convection(h: float, A: float, dT: float) -> float:
    """Newton's law of cooling: Q = hA dT (W)."""
    return h * A * dT


def nusselt_dittus_boelter(re: float, pr: float, heating: bool = True) -> float:
    """Dittus-Boelter correlation for turbulent pipe flow.

    Nu = 0.023 Re^0.8 Pr^n, n = 0.4 heating / 0.3 cooling.
    Valid: Re > 1e4, 0.6 < Pr < 160, L/D > 10.
    """
    n = 0.4 if heating else 0.3
    return 0.023 * re ** 0.8 * pr ** n


def nusselt_laminar_pipe(constant_wall_temp: bool = True) -> float:
    """Fully developed laminar pipe flow: Nu = 3.66 (const T_w) or 4.36 (const q)."""
    return 3.66 if constant_wall_temp else 4.36


def h_from_nusselt(nu: float, k: float, L: float) -> float:
    """Convection coefficient h = Nu*k/L (W/m^2K)."""
    return nu * k / L


# -------------------------------------------------------------- radiation
def radiation(emissivity: float, A: float, T_surface: float, T_surroundings: float) -> float:
    """Net radiative exchange with large surroundings (temperatures in K):
    Q = eps*sigma*A*(Ts^4 - Tsur^4)."""
    if T_surface < 0 or T_surroundings < 0:
        raise ValueError("temperatures must be absolute (K)")
    return emissivity * STEFAN_BOLTZMANN * A * (T_surface ** 4 - T_surroundings ** 4)


# ---------------------------------------------------------- heat exchangers
def lmtd(dT1: float, dT2: float) -> float:
    """Log-mean temperature difference between the two ends of an exchanger."""
    if dT1 * dT2 <= 0:
        raise ValueError("dT1 and dT2 must have the same sign and be nonzero")
    if dT1 == dT2:
        return dT1
    return (dT1 - dT2) / math.log(dT1 / dT2)


def heat_exchanger_q(U: float, A: float, dT1: float, dT2: float) -> float:
    """Q = U A LMTD (W)."""
    return U * A * lmtd(dT1, dT2)


def effectiveness_ntu_counterflow(ntu: float, c_ratio: float) -> float:
    """Counterflow effectiveness from NTU and C_min/C_max ratio."""
    if not 0 <= c_ratio <= 1:
        raise ValueError("c_ratio must be in [0, 1]")
    if c_ratio == 1:
        return ntu / (1 + ntu)
    e = math.exp(-ntu * (1 - c_ratio))
    return (1 - e) / (1 - c_ratio * e)


# -------------------------------------------------------------- ideal gas
def ideal_gas_pressure(n_mol: float, T: float, V: float) -> float:
    """p = nRT/V (Pa), T in K, V in m^3."""
    return n_mol * R_UNIVERSAL * T / V


def ideal_gas_density(p: float, T: float, R_specific: float = R_AIR) -> float:
    """rho = p/(R T), kg/m^3."""
    return p / (R_specific * T)


def sensible_heat(mass: float, cp: float, dT: float) -> float:
    """Q = m cp dT (J)."""
    return mass * cp * dT


def thermal_expansion(L0: float, alpha: float, dT: float) -> float:
    """Linear expansion dL = alpha L0 dT (m)."""
    return alpha * L0 * dT
