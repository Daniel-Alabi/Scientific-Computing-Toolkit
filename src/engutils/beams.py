"""Beam bending and cross-section properties (Euler-Bernoulli theory).

All inputs SI: lengths m, forces N, distributed loads N/m, E in Pa, I in m^4.
Deflections positive downward for downward loads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


# ---------------------------------------------------------------- sections
@dataclass(frozen=True)
class Section:
    """Cross-section properties: area A (m^2), second moment I (m^4) about the
    horizontal centroidal axis, section modulus S (m^3), extreme fiber c (m)."""
    A: float
    I: float
    c: float

    @property
    def S(self) -> float:
        return self.I / self.c


def rectangle(b: float, h: float) -> Section:
    """Solid rectangle, width b, height h."""
    return Section(A=b * h, I=b * h ** 3 / 12, c=h / 2)


def circle(d: float) -> Section:
    """Solid circular bar, diameter d."""
    return Section(A=math.pi * d ** 2 / 4, I=math.pi * d ** 4 / 64, c=d / 2)


def hollow_circle(d_out: float, d_in: float) -> Section:
    """Tube: outer diameter d_out, inner diameter d_in."""
    if d_in >= d_out:
        raise ValueError("d_in must be < d_out")
    A = math.pi * (d_out ** 2 - d_in ** 2) / 4
    I = math.pi * (d_out ** 4 - d_in ** 4) / 64
    return Section(A=A, I=I, c=d_out / 2)


def i_beam(b_f: float, t_f: float, h: float, t_w: float) -> Section:
    """Symmetric I-beam: flange width b_f, flange thickness t_f,
    overall depth h, web thickness t_w."""
    h_w = h - 2 * t_f
    if h_w <= 0:
        raise ValueError("flanges too thick for overall depth")
    A = 2 * b_f * t_f + t_w * h_w
    I = (b_f * h ** 3 - (b_f - t_w) * h_w ** 3) / 12
    return Section(A=A, I=I, c=h / 2)


# ------------------------------------------------------- classic load cases
def cantilever_end_load(P: float, L: float, E: float, I: float) -> dict:
    """Cantilever, point load P at free end."""
    return {
        "max_deflection": P * L ** 3 / (3 * E * I),
        "max_moment": P * L,
        "max_shear": P,
        "end_slope": P * L ** 2 / (2 * E * I),
    }


def cantilever_udl(w: float, L: float, E: float, I: float) -> dict:
    """Cantilever, uniform load w (N/m) over full length."""
    return {
        "max_deflection": w * L ** 4 / (8 * E * I),
        "max_moment": w * L ** 2 / 2,
        "max_shear": w * L,
        "end_slope": w * L ** 3 / (6 * E * I),
    }


def simply_supported_center_load(P: float, L: float, E: float, I: float) -> dict:
    """Simply supported beam, point load P at midspan."""
    return {
        "max_deflection": P * L ** 3 / (48 * E * I),
        "max_moment": P * L / 4,
        "max_shear": P / 2,
    }


def simply_supported_udl(w: float, L: float, E: float, I: float) -> dict:
    """Simply supported beam, uniform load w (N/m)."""
    return {
        "max_deflection": 5 * w * L ** 4 / (384 * E * I),
        "max_moment": w * L ** 2 / 8,
        "max_shear": w * L / 2,
    }


def fixed_fixed_udl(w: float, L: float, E: float, I: float) -> dict:
    """Fixed-fixed beam, uniform load w (N/m)."""
    return {
        "max_deflection": w * L ** 4 / (384 * E * I),
        "max_moment": w * L ** 2 / 12,  # at supports
        "midspan_moment": w * L ** 2 / 24,
        "max_shear": w * L / 2,
    }


# ------------------------------------------------------------------ stress
def bending_stress(M: float, section: Section) -> float:
    """Max bending stress sigma = M*c/I."""
    return M * section.c / section.I


def axial_stress(P: float, section: Section) -> float:
    """Normal stress sigma = P/A."""
    return P / section.A


def euler_buckling(E: float, I: float, L: float, k: float = 1.0) -> float:
    """Euler critical buckling load P_cr = pi^2 EI/(kL)^2.

    k: effective length factor (1.0 pinned-pinned, 0.5 fixed-fixed,
    0.7 fixed-pinned, 2.0 fixed-free).
    """
    return math.pi ** 2 * E * I / (k * L) ** 2


def safety_factor(yield_strength: float, applied_stress: float) -> float:
    """Static safety factor against yield."""
    if applied_stress <= 0:
        raise ValueError("applied stress must be positive")
    return yield_strength / applied_stress
