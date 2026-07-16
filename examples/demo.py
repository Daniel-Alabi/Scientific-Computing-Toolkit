"""engutils demo: a small end-to-end design check."""

from engutils import units, materials, beams, fluids, thermo

print("=== units ===")
print(f"  100 psi = {units.convert(100, 'psi', 'kPa'):.2f} kPa")
print(f"  60 mph  = {units.convert(60, 'mph', 'km/h'):.2f} km/h")
print(f"  350 F   = {units.convert_temperature(350, 'F', 'C'):.1f} C")

print("\n=== beam design check ===")
# 3 m simply supported A36 steel beam, 5 kN/m distributed load, 100x200 mm section
steel = materials.get("steel_a36")
sec = beams.rectangle(0.100, 0.200)
res = beams.simply_supported_udl(5000, 3.0, steel.E, sec.I)
sigma = beams.bending_stress(res["max_moment"], sec)
sf = beams.safety_factor(steel.yield_strength, sigma)
print(f"  max deflection = {res['max_deflection']*1000:.3f} mm")
print(f"  max moment     = {res['max_moment']/1000:.2f} kN*m")
print(f"  bending stress = {sigma/1e6:.2f} MPa")
print(f"  safety factor  = {sf:.2f}")

print("\n=== pipe flow ===")
# water at 2 m/s through 20 m of 50 mm commercial steel pipe
rho, mu = fluids.WATER_20C["rho"], fluids.WATER_20C["mu"]
re = fluids.reynolds(rho, 2.0, 0.05, mu)
dp = fluids.pressure_drop_pipe(rho, 2.0, 20, 0.05, mu, roughness=4.5e-5)
Q = 2.0 * 3.14159265 * 0.05**2 / 4
print(f"  Re = {re:.0f} ({fluids.flow_regime(re)})")
print(f"  pressure drop = {dp/1000:.2f} kPa")
print(f"  pump power (70% eff) = {fluids.pump_power(Q, dp, 0.7):.1f} W")

print("\n=== heat loss through a wall ===")
# 10 m^2 wall: 20 cm concrete + inside/outside convection, dT = 25 K
concrete = materials.get("concrete")
r_wall = thermo.thermal_resistance_wall(0.20, concrete.k, 10)
r_in = thermo.thermal_resistance_convection(8, 10)
r_out = thermo.thermal_resistance_convection(25, 10)
q = thermo.composite_wall_heat(25, [r_in, r_wall, r_out])
print(f"  heat loss = {q:.0f} W")
