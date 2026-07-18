"""Unit conversion.

``convert(value, "ft", "m")`` converts between any two units of the same
quantity. Temperatures use ``convert_temperature`` (offset scales).
"""

from __future__ import annotations

# Conversion factors TO the SI base unit of each quantity.
_FACTORS = {
    # length -> m
    "m": 1.0, "km": 1e3, "cm": 1e-2, "mm": 1e-3, "um": 1e-6,
    "in": 0.0254, "ft": 0.3048, "yd": 0.9144, "mi": 1609.344,
    # mass -> kg
    "kg": 1.0, "g": 1e-3, "mg": 1e-6, "tonne": 1e3,
    "lbm": 0.45359237, "oz": 0.028349523125, "slug": 14.59390294,
    # force -> N
    "N": 1.0, "kN": 1e3, "MN": 1e6, "lbf": 4.4482216152605, "kip": 4448.2216152605, "kgf": 9.80665,
    # pressure -> Pa
    "Pa": 1.0, "kPa": 1e3, "MPa": 1e6, "GPa": 1e9, "bar": 1e5,
    "atm": 101325.0, "psi": 6894.757293168, "ksi": 6894757.293168, "torr": 133.32236842105263,
    "mmHg": 133.322387415, "inH2O": 249.0889,
    # energy -> J
    "J": 1.0, "kJ": 1e3, "MJ": 1e6, "Wh": 3600.0, "kWh": 3.6e6,
    "cal": 4.184, "kcal": 4184.0, "Btu": 1055.05585262, "ft-lbf": 1.3558179483314,
    # power -> W
    "W": 1.0, "kW": 1e3, "MW": 1e6, "hp": 745.6998715822702, "Btu/h": 0.29307107,
    # volume -> m^3
    "m3": 1.0, "L": 1e-3, "mL": 1e-6, "gal": 3.785411784e-3, "qt": 9.46352946e-4,
    "ft3": 0.028316846592, "in3": 1.6387064e-5,
    # flow -> m^3/s
    "m3/s": 1.0, "L/s": 1e-3, "L/min": 1e-3 / 60, "gpm": 3.785411784e-3 / 60, "cfm": 0.028316846592 / 60,
    # velocity -> m/s
    "m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704, "ft/s": 0.3048, "knot": 0.514444,
    # area -> m^2
    "m2": 1.0, "cm2": 1e-4, "mm2": 1e-6, "ft2": 0.09290304, "in2": 6.4516e-4,
    # angle -> rad
    "rad": 1.0, "deg": 3.141592653589793 / 180, "rev": 2 * 3.141592653589793,
    # time -> s
    "s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0,
}

_QUANTITY = {}
for _group in (
    ("m", "km", "cm", "mm", "um", "in", "ft", "yd", "mi"),
    ("kg", "g", "mg", "tonne", "lbm", "oz", "slug"),
    ("N", "kN", "MN", "lbf", "kip", "kgf"),
    ("Pa", "kPa", "MPa", "GPa", "bar", "atm", "psi", "ksi", "torr", "mmHg", "inH2O"),
    ("J", "kJ", "MJ", "Wh", "kWh", "cal", "kcal", "Btu", "ft-lbf"),
    ("W", "kW", "MW", "hp", "Btu/h"),
    ("m3", "L", "mL", "gal", "qt", "ft3", "in3"),
    ("m3/s", "L/s", "L/min", "gpm", "cfm"),
    ("m/s", "km/h", "mph", "ft/s", "knot"),
    ("m2", "cm2", "mm2", "ft2", "in2"),
    ("rad", "deg", "rev"),
    ("s", "min", "h", "day"),
):
    for _u in _group:
        _QUANTITY[_u] = _group[0]


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Convert ``value`` from ``from_unit`` to ``to_unit``.

    >>> convert(100, "psi", "kPa")
    689.4757293168
    """
    try:
        qf, qt = _QUANTITY[from_unit], _QUANTITY[to_unit]
    except KeyError as e:
        raise ValueError(f"unknown unit: {e.args[0]}. Known: {sorted(_FACTORS)}") from None
    if qf != qt:
        raise ValueError(f"incompatible units: {from_unit} ({qf}) vs {to_unit} ({qt})")
    return value * _FACTORS[from_unit] / _FACTORS[to_unit]


def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert temperature between C, F, K, R (Rankine)."""
    u = {"C": "C", "F": "F", "K": "K", "R": "R"}
    f, t = u.get(from_unit.upper().rstrip("°")), u.get(to_unit.upper().rstrip("°"))
    if f is None or t is None:
        raise ValueError("temperature units must be C, F, K, or R")
    # to kelvin
    k = {
        "C": lambda v: v + 273.15,
        "F": lambda v: (v - 32) * 5 / 9 + 273.15,
        "K": lambda v: v,
        "R": lambda v: v * 5 / 9,
    }[f](value)
    # from kelvin
    return {
        "C": lambda v: v - 273.15,
        "F": lambda v: (v - 273.15) * 9 / 5 + 32,
        "K": lambda v: v,
        "R": lambda v: v * 9 / 5,
    }[t](k)


def si_prefix(value: float) -> str:
    """Format a value with an appropriate SI prefix.

    >>> si_prefix(4.2e6)
    '4.2 M'
    """
    prefixes = [
        (1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""),
        (1e-3, "m"), (1e-6, "u"), (1e-9, "n"), (1e-12, "p"),
    ]
    if value == 0:
        return "0 "
    mag = abs(value)
    for factor, prefix in prefixes:
        if mag >= factor:
            return f"{value / factor:g} {prefix}"
    return f"{value:g} "
