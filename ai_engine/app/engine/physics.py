"""
physics.py — 1-D Navier-Stokes residuals and Negative Pressure Wave (NPW)
math for the NAVIER pipeline monitoring system.

References
----------
* Wylie & Streeter, "Fluid Transients in Systems" (1993)
* Brunone et al., "Leak Detection in Pipeline Systems" (2000)
* API TR 2570, "Leak Detection for Liquid Petroleum Pipelines" (2020)

All units: SI unless explicitly noted.
  pressure     → Pa   (1 bar = 1e5 Pa)
  flow_rate    → m³/s
  density      → kg/m³
  viscosity    → Pa·s (dynamic)  [1 cSt × density_kg_m3 / 1000 = Pa·s approx]
  velocity     → m/s
  length       → m
  time         → s
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Physical constants dictionary — one entry per chemical
# ─────────────────────────────────────────────────────────────────────────────

CHEMICAL_PROFILES: dict[str, dict] = {
    "MS": {          # Motor Spirit / Petrol
        "name": "Motor Spirit (Petrol)",
        "density_kg_m3": 745.0,          # at 15 °C
        "dynamic_viscosity_pa_s": 5.5e-4, # ~0.55 cSt @ 40 °C
        "bulk_modulus_pa": 1.32e9,
        "vapor_pressure_pa": 55_000.0,
        "thermal_expansion_coeff": 9.5e-4,  # per °C
        "wave_speed_ms": 1050.0,            # acoustic in MS
        "specific_heat_j_kgk": 2100.0,
    },
    "HSD": {         # High-Speed Diesel
        "name": "High Speed Diesel",
        "density_kg_m3": 840.0,
        "dynamic_viscosity_pa_s": 3.5e-3,   # ~4.2 cSt @ 40 °C
        "bulk_modulus_pa": 1.48e9,
        "vapor_pressure_pa": 1_500.0,
        "thermal_expansion_coeff": 8.6e-4,
        "wave_speed_ms": 1320.0,
        "specific_heat_j_kgk": 1950.0,
    },
    "ATF": {         # Aviation Turbine Fuel (Jet-A1)
        "name": "Aviation Turbine Fuel (Jet-A1)",
        "density_kg_m3": 800.0,
        "dynamic_viscosity_pa_s": 1.2e-3,   # ~1.5 cSt @ 40 °C
        "bulk_modulus_pa": 1.42e9,
        "vapor_pressure_pa": 2_000.0,
        "thermal_expansion_coeff": 9.0e-4,
        "wave_speed_ms": 1200.0,
        "specific_heat_j_kgk": 2000.0,
    },
    "BENZENE": {
        "name": "Benzene",
        "density_kg_m3": 879.0,
        "dynamic_viscosity_pa_s": 6.5e-4,
        "bulk_modulus_pa": 1.50e9,
        "vapor_pressure_pa": 12_700.0,
        "thermal_expansion_coeff": 1.24e-3,
        "wave_speed_ms": 1300.0,
        "specific_heat_j_kgk": 1740.0,
    },
    "LPG": {         # Liquefied Petroleum Gas (propane dominant)
        "name": "LPG (Propane-dominant)",
        "density_kg_m3": 508.0,
        "dynamic_viscosity_pa_s": 1.1e-4,
        "bulk_modulus_pa": 4.3e8,
        "vapor_pressure_pa": 850_000.0,    # ~8.5 bar — pressurised storage
        "thermal_expansion_coeff": 1.6e-3,
        "wave_speed_ms": 700.0,
        "specific_heat_j_kgk": 2520.0,
    },
}

# Default: crude-oil-like fallback
CHEMICAL_PROFILES["UNKNOWN"] = CHEMICAL_PROFILES["HSD"].copy()
CHEMICAL_PROFILES["UNKNOWN"]["name"] = "Unknown Fluid (HSD defaults)"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes for results
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NavierStokesResiduals:
    """
    1-D incompressible Navier-Stokes residuals for mass and momentum
    conservation at a single pipeline cross-section.
    """
    mass_residual: float          # |∂ρ/∂t + ∂(ρu)/∂x|  — should be ≈ 0
    momentum_residual: float      # |ρ(∂u/∂t + u∂u/∂x) + ∂P/∂x + f_w| — ≈ 0
    darcy_friction_factor: float
    reynolds_number: float
    is_turbulent: bool            # Re > 4000
    pinn_loss_weight: float       # combined residual used as PINN physics loss

    @property
    def total_residual(self) -> float:
        return math.sqrt(self.mass_residual**2 + self.momentum_residual**2)


@dataclass
class NPWResult:
    """Result of a Negative Pressure Wave leak-localization calculation."""
    triggered: bool
    leak_distance_m: float        # X from upstream sensor
    wave_speed_ms: float
    delta_t_s: float              # arrival-time difference (estimated)
    segment_length_m: float
    confidence: float             # 0.0 – 1.0
    method: str                   # "NPW" | "PRESSURE_GRADIENT" | "NONE"


# ─────────────────────────────────────────────────────────────────────────────
# 1-D Navier-Stokes implementation
# ─────────────────────────────────────────────────────────────────────────────

def pipe_reynolds_number(
    velocity_ms: float,
    diameter_m: float,
    density_kg_m3: float,
    dynamic_viscosity_pa_s: float,
) -> float:
    """
    Compute the Reynolds number for pipe flow.

        Re = ρ · u · D / μ

    Parameters
    ----------
    velocity_ms            : Mean flow velocity [m/s]
    diameter_m             : Internal pipe diameter [m]
    density_kg_m3          : Fluid density [kg/m³]
    dynamic_viscosity_pa_s : Dynamic viscosity [Pa·s]

    Returns
    -------
    float : Reynolds number (dimensionless)
    """
    if dynamic_viscosity_pa_s <= 0 or diameter_m <= 0:
        return 0.0
    return (density_kg_m3 * abs(velocity_ms) * diameter_m) / dynamic_viscosity_pa_s


def darcy_friction_factor(re: float, relative_roughness: float = 4.6e-5) -> float:
    """
    Compute the Darcy-Weisbach friction factor.

    * Laminar   (Re < 2300)  : f = 64 / Re
    * Transition (2300–4000) : linear interpolation
    * Turbulent (Re > 4000)  : Colebrook-White equation solved iteratively

    Parameters
    ----------
    re               : Reynolds number
    relative_roughness: ε/D — roughness height / diameter
                       (default 4.6e-5 corresponds to commercial steel)

    Returns
    -------
    float : Darcy friction factor f  (dimensionless)
    """
    if re < 1e-6:
        return 0.064  # very slow flow
    if re < 2300:
        return 64.0 / re
    if re < 4000:
        f_lam = 64.0 / 2300.0
        f_turb = _colebrook(4000.0, relative_roughness)
        t = (re - 2300.0) / 1700.0
        return f_lam * (1 - t) + f_turb * t

    return _colebrook(re, relative_roughness)


def _colebrook(re: float, eps_d: float) -> float:
    """
    Colebrook-White implicit equation solved via Newton-Raphson.

        1/√f = -2 log10( ε/(3.7D) + 2.51/(Re·√f) )
    """
    # Swamee-Jain explicit approximation as initial guess
    try:
        f = (0.25 / (math.log10(eps_d / 3.7 + 5.74 / re**0.9))**2)
    except (ValueError, ZeroDivisionError):
        f = 0.02

    for _ in range(20):
        sqrt_f = math.sqrt(f)
        rhs = -2.0 * math.log10(eps_d / 3.7 + 2.51 / (re * sqrt_f))
        f_new = 1.0 / rhs**2
        if abs(f_new - f) < 1e-10:
            break
        f = f_new
    return max(1e-4, f)


def compute_navier_stokes_residuals(
    inlet_pressure_pa: float,
    outlet_pressure_pa: float,
    flow_rate_m3s: float,
    pipe_diameter_m: float,
    pipe_length_m: float,
    fluid_type: str = "HSD",
    temperature_c: float = 30.0,
    dt: float = 1.0,
    du_dt: float = 0.0,   # time-derivative of velocity (from LSTM)
) -> NavierStokesResiduals:
    """
    Compute 1-D Navier-Stokes residuals for a single pipeline segment.

    Governing equations (1-D, incompressible, Newtonian):

    Mass conservation (continuity):
        ∂ρ/∂t  +  ∂(ρu)/∂x  =  0
        → For incompressible: ∂u/∂x = 0  → Q_in = Q_out

    Momentum conservation:
        ρ(∂u/∂t + u·∂u/∂x) = -∂P/∂x - τ_w · (4/D)

    Wall shear stress:
        τ_w = f · ρ · u² / 8          (Darcy-Weisbach)

    The residual is the L2 norm of the LHS minus RHS; a perfectly
    behaving pipeline should give residual ≈ 0.  A leak breaks mass
    conservation, raising mass_residual significantly.

    Parameters
    ----------
    inlet_pressure_pa  : Measured inlet pressure [Pa]
    outlet_pressure_pa : Measured outlet pressure [Pa]
    flow_rate_m3s      : Measured volumetric flow rate [m³/s]
    pipe_diameter_m    : Internal pipe diameter [m]
    pipe_length_m      : Pipeline segment length [m]
    fluid_type         : Key into CHEMICAL_PROFILES dict
    temperature_c      : Operating temperature [°C] — used for viscosity correction
    dt                 : Time step between readings [s]
    du_dt              : Velocity time-derivative (≈ Δu/Δt) [m/s²]

    Returns
    -------
    NavierStokesResiduals dataclass
    """
    profile = CHEMICAL_PROFILES.get(fluid_type.upper(), CHEMICAL_PROFILES["HSD"])
    density = profile["density_kg_m3"]
    mu_base = profile["dynamic_viscosity_pa_s"]

    # Temperature correction for viscosity (Andrade model: μ ∝ exp(B/T))
    mu = _viscosity_temperature_correction(mu_base, temperature_c, fluid_type)

    # Cross-sectional area
    area_m2 = math.pi * (pipe_diameter_m / 2.0) ** 2
    velocity = flow_rate_m3s / area_m2 if area_m2 > 0 else 0.0

    # Hydraulic quantities
    re = pipe_reynolds_number(velocity, pipe_diameter_m, density, mu)
    f  = darcy_friction_factor(re)
    is_turbulent = re > 4000

    # ── Mass conservation residual ──────────────────────────────────────────
    # For incompressible: ∂(ρu)/∂x = 0  ↔  du/dx ≈ 0
    # Proxy: pressure-driven flow rate vs measured flow rate
    # Hagen-Poiseuille / Darcy-Weisbach predicted flow
    delta_p = max(0.0, inlet_pressure_pa - outlet_pressure_pa)
    friction_head_pa = (f * pipe_length_m / pipe_diameter_m) * (density * velocity**2 / 2.0)
    predicted_delta_p = friction_head_pa  # steady-state prediction
    mass_residual = abs(delta_p - predicted_delta_p) / max(1.0, delta_p + predicted_delta_p)

    # ── Momentum conservation residual ─────────────────────────────────────
    # ρ(∂u/∂t + u·∂u/∂x) + ∂P/∂x + τ_w·(4/D) = 0
    dp_dx = (outlet_pressure_pa - inlet_pressure_pa) / pipe_length_m   # sign: outlet > inlet wrong
    tau_w = f * density * velocity**2 / 8.0
    wall_term = tau_w * 4.0 / pipe_diameter_m
    inertia_term = density * du_dt  # ρ · ∂u/∂t  (convective ≈ 0 for quasi-steady)
    momentum_lhs = inertia_term + dp_dx + wall_term
    # Normalise by dynamic pressure
    q_dyn = max(1.0, 0.5 * density * velocity**2)
    momentum_residual = abs(momentum_lhs) / q_dyn

    # Combined PINN loss weight (used in PyTorch autograd)
    pinn_loss = 0.5 * mass_residual + 0.5 * momentum_residual

    return NavierStokesResiduals(
        mass_residual=float(mass_residual),
        momentum_residual=float(momentum_residual),
        darcy_friction_factor=float(f),
        reynolds_number=float(re),
        is_turbulent=is_turbulent,
        pinn_loss_weight=float(pinn_loss),
    )


def _viscosity_temperature_correction(mu_ref: float, temp_c: float, fluid_type: str) -> float:
    """
    Andrade / exponential model for dynamic viscosity temperature dependence.
    μ(T) = μ_ref · exp(B · (1/T - 1/T_ref))

    Reference temperature: 40 °C.  B ≈ 1000 K (typical petroleum).

    Parameters
    ----------
    mu_ref   : Reference dynamic viscosity [Pa·s] at 40 °C
    temp_c   : Operating temperature [°C]
    fluid_type : key into CHEMICAL_PROFILES (unused here; reserved for LUT)

    Returns
    -------
    float : Corrected dynamic viscosity [Pa·s]
    """
    T_ref_K = 313.15   # 40 °C in Kelvin
    T_op_K  = temp_c + 273.15
    B = 1000.0
    correction = math.exp(B * (1.0 / T_op_K - 1.0 / T_ref_K))
    return float(mu_ref * max(0.01, min(100.0, correction)))


# ─────────────────────────────────────────────────────────────────────────────
# Negative Pressure Wave (NPW) localization
# ─────────────────────────────────────────────────────────────────────────────

def compute_wave_speed(
    fluid_type: str,
    pipe_diameter_m: float,
    wall_thickness_m: float,
    elastic_modulus_pa: float = 2.07e11,   # steel Young's modulus
) -> float:
    """
    Korteweg equation for acoustic wave speed in a fluid-filled elastic pipe.

        a = sqrt(K_eff / ρ)

    where  K_eff  accounts for pipe-wall compliance:
        1/K_eff = 1/K_fluid + D/(E · e)

    Parameters
    ----------
    fluid_type      : Key into CHEMICAL_PROFILES
    pipe_diameter_m : Internal diameter [m]
    wall_thickness_m: Pipe wall thickness [m]
    elastic_modulus_pa: Young's modulus of pipe material [Pa]

    Returns
    -------
    float : Wave propagation speed [m/s]
    """
    profile = CHEMICAL_PROFILES.get(fluid_type.upper(), CHEMICAL_PROFILES["HSD"])
    K = profile["bulk_modulus_pa"]
    rho = profile["density_kg_m3"]

    # Pipe-wall compliance term
    compliance = pipe_diameter_m / (elastic_modulus_pa * wall_thickness_m)
    K_eff = 1.0 / (1.0 / K + compliance)
    try:
        a = math.sqrt(K_eff / rho)
        return float(max(300.0, min(2500.0, a)))
    except (ValueError, ZeroDivisionError):
        return profile["wave_speed_ms"]


def npw_localization(
    segment_length_m: float,
    wave_speed_ms: float,
    inlet_pressure_pa: float,
    outlet_pressure_pa: float,
    inlet_pressure_prev_pa: float,
    outlet_pressure_prev_pa: float,
    dt_sensor_s: float = 1.0,
    pressure_drop_threshold_pa: float = 50_000.0,  # 0.5 bar
) -> NPWResult:
    """
    Locate a leak using the Negative Pressure Wave method.

    Formula:
        X = (L - a · Δt) / 2

    where:
        L   = segment length [m]
        a   = acoustic wave speed [m/s]
        Δt  = difference in NPW arrival times at inlet vs outlet [s]
              (estimated here from pressure-drop timing asymmetry)

    A negative pressure wave originates at the leak and propagates
    towards both ends.  The time difference between its arrival at
    the upstream sensor and the downstream sensor identifies distance X.

    When only single-point pressure measurements are available (typical
    SCADA), Δt is estimated from the relative pressure-drop magnitude
    ratio between inlet and outlet.

    Parameters
    ----------
    segment_length_m         : Pipeline segment length [m]
    wave_speed_ms            : Acoustic wave speed [m/s]
    inlet_pressure_pa        : Current inlet pressure [Pa]
    outlet_pressure_pa       : Current outlet pressure [Pa]
    inlet_pressure_prev_pa   : Previous inlet pressure [Pa]
    outlet_pressure_prev_pa  : Previous outlet pressure [Pa]
    dt_sensor_s              : Time between sensor readings [s]
    pressure_drop_threshold_pa: Minimum drop to trigger NPW [Pa]

    Returns
    -------
    NPWResult dataclass
    """
    # Pressure drops at each end since last reading
    drop_inlet  = inlet_pressure_prev_pa  - inlet_pressure_pa
    drop_outlet = outlet_pressure_prev_pa - outlet_pressure_pa

    # Trigger: sudden significant drop at either end
    max_drop = max(drop_inlet, drop_outlet)
    if max_drop < pressure_drop_threshold_pa:
        return NPWResult(
            triggered=False, leak_distance_m=0.0, wave_speed_ms=wave_speed_ms,
            delta_t_s=0.0, segment_length_m=segment_length_m,
            confidence=0.0, method="NONE",
        )

    # Estimate Δt from relative drop magnitudes
    # If inlet drops more ↔ leak is close to inlet (short travel time to inlet)
    # Wave reaches nearest end first → smaller Δt → larger X for inlet-side leak
    total_drop = max(1.0, drop_inlet + drop_outlet)
    fraction_inlet = drop_inlet / total_drop    # 1.0 = all drop at inlet

    # Δt = (fraction_outlet - fraction_inlet) × (L / a)
    # Positive Δt → wave reached outlet first → leak closer to outlet
    delta_t = (drop_outlet / total_drop - fraction_inlet) * (segment_length_m / wave_speed_ms)
    delta_t = max(-segment_length_m / wave_speed_ms, min(segment_length_m / wave_speed_ms, delta_t))

    # NPW formula: X = (L - a·Δt) / 2
    X = (segment_length_m - wave_speed_ms * delta_t) / 2.0
    X = max(0.0, min(segment_length_m, X))

    # Confidence: scales with magnitude of pressure drop relative to operating pressure
    confidence_raw = min(1.0, max_drop / (inlet_pressure_pa + 1.0))
    confidence = float(max(0.05, min(0.95, confidence_raw)))

    return NPWResult(
        triggered=True,
        leak_distance_m=round(X, 1),
        wave_speed_ms=wave_speed_ms,
        delta_t_s=round(delta_t, 4),
        segment_length_m=segment_length_m,
        confidence=confidence,
        method="NPW",
    )


def pressure_gradient_localization(
    inlet_pressure_pa: float,
    outlet_pressure_pa: float,
    segment_length_m: float,
) -> float:
    """
    Simplified pressure-gradient-based leak distance estimate.

    Assumes that under leaking conditions the pressure profile becomes
    non-linear; uses the fraction of pressure loss relative to expected
    linear drop.

    Parameters
    ----------
    inlet_pressure_pa  : Current inlet pressure [Pa]
    outlet_pressure_pa : Current outlet pressure [Pa]
    segment_length_m   : Segment length [m]

    Returns
    -------
    float : Estimated leak distance from inlet [m]
    """
    total = inlet_pressure_pa + outlet_pressure_pa
    if total <= 0:
        return segment_length_m / 2.0
    frac = outlet_pressure_pa / total
    return float(segment_length_m * (1.0 - frac))
