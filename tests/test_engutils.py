"""Tests for engutils against handbook values."""

import math

import pytest

from engutils import units, materials, beams, fluids, thermo


# ------------------------------------------------------------------ units
class TestUnits:
    def test_length(self):
        assert abs(units.convert(1, "ft", "m") - 0.3048) < 1e-12
        assert abs(units.convert(1, "mi", "km") - 1.609344) < 1e-9

    def test_pressure(self):
        assert abs(units.convert(1, "atm", "psi") - 14.6959) < 1e-3
        assert abs(units.convert(100, "psi", "MPa") - 0.6894757) < 1e-6

    def test_roundtrip(self):
        v = units.convert(units.convert(123.4, "kW", "hp"), "hp", "kW")
        assert abs(v - 123.4) < 1e-9

    def test_incompatible_raises(self):
        with pytest.raises(ValueError):
            units.convert(1, "kg", "m")

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            units.convert(1, "furlong", "m")

    def test_temperature(self):
        assert units.convert_temperature(0, "C", "F") == 32
        assert abs(units.convert_temperature(100, "C", "K") - 373.15) < 1e-12
        assert abs(units.convert_temperature(32, "F", "C")) < 1e-12
        assert abs(units.convert_temperature(0, "C", "R") - 491.67) < 1e-9


# -------------------------------------------------------------- materials
class TestMaterials:
    def test_lookup(self):
        steel = materials.get("steel_a36")
        assert steel.E == 200e9
        assert steel.yield_strength == 250e6

    def test_shear_modulus(self):
        al = materials.get("aluminum_6061")
        expected = al.E / (2 * (1 + al.poisson))
        assert abs(al.shear_modulus - expected) < 1e-6

    def test_thermal_diffusivity_copper(self):
        cu = materials.get("copper")
        # copper ~1.16e-4 m^2/s
        assert 1.0e-4 < cu.thermal_diffusivity() < 1.3e-4

    def test_unknown_material(self):
        with pytest.raises(KeyError):
            materials.get("unobtanium")

    def test_available_nonempty(self):
        assert "steel_a36" in materials.available()


# ------------------------------------------------------------------ beams
class TestBeams:
    def test_rectangle_section(self):
        s = beams.rectangle(0.1, 0.2)
        assert abs(s.A - 0.02) < 1e-12
        assert abs(s.I - 0.1 * 0.2 ** 3 / 12) < 1e-15
        assert abs(s.S - s.I / 0.1) < 1e-15

    def test_circle_vs_hollow(self):
        solid = beams.circle(0.05)
        tube = beams.hollow_circle(0.05, 0.04)
        assert tube.I < solid.I
        assert tube.A < solid.A

    def test_simply_supported_center_load(self):
        # steel beam: P=10kN, L=2m, E=200GPa, I=1e-6 m^4
        r = beams.simply_supported_center_load(10e3, 2.0, 200e9, 1e-6)
        assert abs(r["max_deflection"] - 10e3 * 8 / (48 * 200e9 * 1e-6)) < 1e-12
        assert abs(r["max_moment"] - 5e3) < 1e-9

    def test_cantilever_udl(self):
        r = beams.cantilever_udl(1000, 3.0, 200e9, 1e-6)
        assert abs(r["max_moment"] - 4500) < 1e-9
        assert abs(r["max_deflection"] - 1000 * 81 / (8 * 200e9 * 1e-6)) < 1e-12

    def test_bending_stress(self):
        s = beams.rectangle(0.05, 0.1)
        sigma = beams.bending_stress(1000.0, s)
        assert abs(sigma - 1000 * 0.05 / s.I) < 1e-6

    def test_euler_buckling_pinned(self):
        P = beams.euler_buckling(200e9, 1e-8, 2.0)
        assert abs(P - math.pi ** 2 * 200e9 * 1e-8 / 4) < 1e-6

    def test_safety_factor(self):
        assert abs(beams.safety_factor(250e6, 100e6) - 2.5) < 1e-12


# ----------------------------------------------------------------- fluids
class TestFluids:
    def test_reynolds_water(self):
        re = fluids.reynolds(998.2, 1.0, 0.05, 1.002e-3)
        assert abs(re - 998.2 * 0.05 / 1.002e-3) < 1e-6

    def test_regimes(self):
        assert fluids.flow_regime(1000) == "laminar"
        assert fluids.flow_regime(3000) == "transitional"
        assert fluids.flow_regime(10000) == "turbulent"

    def test_laminar_friction(self):
        assert abs(fluids.friction_factor(1600) - 0.04) < 1e-12

    def test_turbulent_friction_smooth(self):
        f = fluids.friction_factor(1e5)
        assert 0.017 < f < 0.019  # ~0.018 for smooth pipe at Re=1e5

    def test_pressure_drop_laminar_hagen_poiseuille(self):
        # laminar: dp = 32 mu L v / D^2
        rho, v, L, D, mu = 998.2, 0.01, 10.0, 0.05, 1.002e-3
        dp = fluids.pressure_drop_pipe(rho, v, L, D, mu)
        assert abs(dp - 32 * mu * L * v / D ** 2) / dp < 1e-9

    def test_velocity_from_flow(self):
        v = fluids.velocity_from_flow(0.001, 0.05)
        assert abs(v - 0.001 / (math.pi * 0.0025 / 4)) < 1e-12

    def test_drag(self):
        # car: Cd=0.3, A=2.2 m^2, 30 m/s in air
        F = fluids.drag_force(1.204, 30, 0.3, 2.2)
        assert abs(F - 0.5 * 1.204 * 900 * 0.66) < 1e-9

    def test_pump_power(self):
        assert abs(fluids.pump_power(0.01, 100e3, 0.7) - 0.01 * 100e3 / 0.7) < 1e-9

    def test_orifice_negative_dp(self):
        with pytest.raises(ValueError):
            fluids.orifice_flow(0.6, 1e-4, -10, 1000)


# ----------------------------------------------------------------- thermo
class TestThermo:
    def test_conduction_wall(self):
        # 10 cm brick wall, k=0.7, A=10 m^2, dT=20 K -> 1400 W
        assert abs(thermo.conduction_1d(0.7, 10, 20, 0.1) - 1400) < 1e-9

    def test_composite_wall(self):
        r1 = thermo.thermal_resistance_wall(0.1, 0.7, 10)
        r2 = thermo.thermal_resistance_convection(10, 10)
        q = thermo.composite_wall_heat(20, [r1, r2])
        assert abs(q - 20 / (r1 + r2)) < 1e-9

    def test_radiation_equilibrium_zero(self):
        assert thermo.radiation(0.9, 1.0, 300, 300) == 0

    def test_radiation_sign(self):
        assert thermo.radiation(0.9, 1.0, 400, 300) > 0
        assert thermo.radiation(0.9, 1.0, 300, 400) < 0

    def test_lmtd_symmetric(self):
        assert abs(thermo.lmtd(10, 10) - 10) < 1e-12

    def test_lmtd_value(self):
        val = thermo.lmtd(40, 20)
        assert abs(val - 20 / math.log(2)) < 1e-9

    def test_lmtd_invalid(self):
        with pytest.raises(ValueError):
            thermo.lmtd(10, -5)

    def test_dittus_boelter(self):
        nu = thermo.nusselt_dittus_boelter(1e4, 3.0, heating=True)
        assert abs(nu - 0.023 * 1e4 ** 0.8 * 3.0 ** 0.4) < 1e-9

    def test_effectiveness_ntu_limits(self):
        assert abs(thermo.effectiveness_ntu_counterflow(2.0, 1.0) - 2 / 3) < 1e-12
        # c_ratio=0 reduces to 1 - exp(-NTU)
        assert abs(thermo.effectiveness_ntu_counterflow(2.0, 0.0) - (1 - math.exp(-2))) < 1e-12

    def test_ideal_gas_air_density(self):
        rho = thermo.ideal_gas_density(101325, 293.15)
        assert abs(rho - 1.204) < 0.01

    def test_sensible_heat(self):
        # 1 kg water raised 10 K: ~41.86 kJ
        assert abs(thermo.sensible_heat(1, 4186, 10) - 41860) < 1e-9

    def test_thermal_expansion(self):
        # 10 m steel rail, 50 K rise: ~5.85 mm
        dl = thermo.thermal_expansion(10, 11.7e-6, 50)
        assert abs(dl - 5.85e-3) < 1e-12
