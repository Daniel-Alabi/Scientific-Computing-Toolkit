# engutils

Engineering utilities toolkit, SI units throughout. Pure Python, no dependencies.

## Install

```bash
pip install -e .          # from this directory
pip install -e ".[test]"  # with pytest
```

## Modules

| Module | Contents |
|---|---|
| `units` | `convert(v, "psi", "kPa")` across length/mass/force/pressure/energy/power/volume/flow/velocity/area/angle/time; `convert_temperature` for C/F/K/R |
| `materials` | property database (steel, aluminum, titanium, copper, plastics, ...): density, E, yield, Poisson, k, cp, alpha + derived G, K, diffusivity |
| `beams` | section properties (rectangle, circle, tube, I-beam), classic load cases (cantilever, simply supported, fixed-fixed), bending stress, Euler buckling, safety factor |
| `fluids` | Reynolds, friction factor (Swamee-Jain), Darcy-Weisbach pressure drop, Bernoulli, drag, orifice flow, pump power |
| `thermo` | conduction (plane/cylindrical), convection + Nusselt correlations, radiation, LMTD, effectiveness-NTU, ideal gas, thermal expansion |

## Quick start

```python
from engutils import units, materials, beams, fluids

# unit conversion
stress_mpa = units.convert(36_000, "psi", "MPa")

# material lookup
al = materials.get("aluminum_6061")
print(al.E, al.yield_strength, al.shear_modulus)

# beam check: 2 m simply supported steel beam, 10 kN center load
s = beams.rectangle(0.05, 0.10)
r = beams.simply_supported_center_load(10e3, 2.0, 200e9, s.I)
sigma = beams.bending_stress(r["max_moment"], s)
sf = beams.safety_factor(materials.get("steel_a36").yield_strength, sigma)

# pipe pressure drop: water, 2 m/s, 20 m of 50 mm pipe
dp = fluids.pressure_drop_pipe(998.2, 2.0, 20, 0.05, 1.002e-3, roughness=4.5e-5)
```

Material data are typical room-temperature values — verify against supplier datasheets for real design work.

## Tests

```bash
pytest
```
