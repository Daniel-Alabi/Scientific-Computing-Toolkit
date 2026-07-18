"""Material property database (typical room-temperature values, SI units).

Properties per material:
    density        kg/m^3
    E              Young's modulus, Pa
    yield_strength Pa (None where not meaningful)
    poisson        Poisson's ratio
    k              thermal conductivity, W/(m*K)
    cp             specific heat, J/(kg*K)
    alpha          thermal expansion coefficient, 1/K

Values are typical/nominal — verify against supplier data for design work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Material:
    name: str
    density: float          # kg/m^3
    E: float                # Pa
    yield_strength: Optional[float]  # Pa
    poisson: float
    k: float                # W/(m K)
    cp: float               # J/(kg K)
    alpha: float            # 1/K

    @property
    def shear_modulus(self) -> float:
        """G = E / (2(1+nu))."""
        return self.E / (2 * (1 + self.poisson))

    @property
    def bulk_modulus(self) -> float:
        """K = E / (3(1-2nu))."""
        return self.E / (3 * (1 - 2 * self.poisson))

    def thermal_diffusivity(self) -> float:
        """alpha_t = k / (rho * cp), m^2/s."""
        return self.k / (self.density * self.cp)


_DB = {
    "steel_a36":        Material("ASTM A36 steel",        7850, 200e9, 250e6, 0.26, 45,   486,  11.7e-6),
    "steel_4140":       Material("AISI 4140 steel",       7850, 205e9, 655e6, 0.29, 42.6, 473,  12.3e-6),
    "stainless_304":    Material("304 stainless steel",   8000, 193e9, 215e6, 0.29, 16.2, 500,  17.3e-6),
    "aluminum_6061":    Material("Aluminum 6061-T6",      2700, 68.9e9, 276e6, 0.33, 167,  896,  23.6e-6),
    "aluminum_7075":    Material("Aluminum 7075-T6",      2810, 71.7e9, 503e6, 0.33, 130,  960,  23.4e-6),
    "copper":           Material("Copper (annealed)",     8960, 117e9, 70e6,  0.34, 401,  385,  16.5e-6),
    "brass":            Material("Brass (70/30)",         8530, 100e9, 200e6, 0.34, 109,  380,  19.9e-6),
    "titanium_6al4v":   Material("Ti-6Al-4V",             4430, 113.8e9, 880e6, 0.342, 6.7, 526, 8.6e-6),
    "cast_iron_gray":   Material("Gray cast iron",        7200, 110e9, None,  0.26, 53,   490,  10.8e-6),
    "concrete":         Material("Concrete (normal)",     2400, 30e9,  None,  0.20, 1.7,  880,  10e-6),
    "douglas_fir":      Material("Douglas fir (parallel)", 530, 13e9,  None,  0.29, 0.12, 1700, 3.5e-6),
    "abs":              Material("ABS plastic",           1060, 2.3e9, 40e6,  0.35, 0.25, 1400, 90e-6),
    "nylon_66":         Material("Nylon 6/6",             1140, 2.9e9, 82e6,  0.39, 0.25, 1670, 80e-6),
    "pla":              Material("PLA (3D printed)",      1250, 3.5e9, 50e6,  0.36, 0.13, 1800, 68e-6),
}


def get(name: str) -> Material:
    """Look up a material by key (e.g. 'steel_a36', 'aluminum_6061')."""
    key = name.lower().strip()
    if key not in _DB:
        raise KeyError(f"unknown material {name!r}. Available: {sorted(_DB)}")
    return _DB[key]


def available() -> list:
    """List of available material keys."""
    return sorted(_DB)
