"""engutils: engineering utilities toolkit.

Modules
-------
units      : unit conversion between SI/imperial (length, mass, pressure, ...)
materials  : property database for common engineering materials
beams      : bending stress, deflection, section properties
fluids     : Reynolds number, friction factor, pressure drop, drag
thermo     : conduction, convection, radiation, LMTD, ideal gas
"""

from . import units, materials, beams, fluids, thermo

__version__ = "0.1.0"
__all__ = ["units", "materials", "beams", "fluids", "thermo"]
