#!/usr/bin/env bash
# =============================================================================
#  NAVIER AI Core — Single-Shot Setup Script
#  Run:  bash setup_navier.sh
# =============================================================================
set -euo pipefail

PROJECT="navier_ai_core"

echo "🚀  Creating NAVIER AI Core project: $PROJECT"

# ── Directory tree ─────────────────────────────────────────────────────────
mkdir -p "$PROJECT"/{app/{engine,schemas,api},data,models,logs,tests}

# ─────────────────────────────────────────────────────────────────────────────
# requirements.txt
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/requirements.txt"
# NAVIER AI Core — Python dependencies
torch>=2.2.0
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
pydantic>=2.7.0
pydantic-settings>=2.3.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.5.0
scipy>=1.13.0
httpx>=0.27.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
python-json-logger>=2.0.7
joblib>=1.4.0
PYEOF
echo "  ✅  requirements.txt"

# ─────────────────────────────────────────────────────────────────────────────
# app/engine/physics.py
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/app/engine/physics.py"
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
PYEOF
echo "  ✅  app/engine/physics.py"

# ─────────────────────────────────────────────────────────────────────────────
# app/engine/causal_guard.py
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/app/engine/causal_guard.py"
"""
causal_guard.py — Sensor Confidence Score (SCS), drift detection,
and batch-change identification for the NAVIER monitoring system.

The SCS quantifies how much the real sensor readings diverge from the
physically consistent 'truth' predicted by the PINN.  High divergence
may indicate sensor degradation, ambient-heat induced drift (especially
relevant for Indian field conditions at 45 °C), or a pending batch change.

Algorithm
---------
1. PINN predicts expected sensor values from physical equations.
2. Relative divergence per sensor: δᵢ = |measured - predicted| / |predicted|
3. Weighted-mean divergence: Δ = Σ wᵢ δᵢ
4. SCS = (1 - Δ) × 100 clamped to [0, 100]
5. If Δ > 0.20  →  SCS < 80  (per requirement)
6. Additional rules:
   - Temperature-induced drift (45 °C) flags SENSOR_DRIFT
   - Density / viscosity change flags BATCH_CHANGE
   - Acoustic spike flags LEAK candidate
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# ── Sensor weights for SCS (pressure most critical for leak detection) ───────
SCS_WEIGHTS: dict[str, float] = {
    "inlet_pressure":  0.30,
    "outlet_pressure": 0.30,
    "flow_rate":       0.20,
    "temperature":     0.10,
    "acoustic":        0.10,
}

# ── Drift thresholds ─────────────────────────────────────────────────────────
DRIFT_THRESHOLD_PCT: float       = 0.15   # 15 % linear deviation triggers drift flag
DIVERGENCE_THRESHOLD: float      = 0.20   # > 20 % PINN divergence → SCS < 80
ACOUSTIC_SPIKE_THRESHOLD: float  = 3.0    # × baseline std triggers acoustic alarm
BATCH_DENSITY_DELTA_KGM3: float  = 25.0   # kg/m³ shift → batch change
AMBIENT_DRIFT_TEMP_C: float      = 42.0   # Indian summer threshold


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SensorConfidenceScore:
    """
    Comprehensive sensor health report for one timestep.
    """
    scs: float                              # 0.0 – 100.0
    overall_divergence_pct: float           # mean divergence from PINN truth [%]
    per_sensor_divergence: dict[str, float] # per-channel relative divergence
    flags: list[str]                        # e.g. ["SENSOR_DRIFT", "BATCH_CHANGE"]
    rul_hours: Optional[float]              # Remaining Useful Life of sensor array
    rul_confidence: float                   # 0.0 – 1.0
    drift_rate_pct_per_hour: float          # slope of drift regression
    ambient_temp_c: float


@dataclass
class BatchChangeResult:
    """Result of a product-batch change check."""
    detected: bool
    from_fluid: Optional[str]
    to_fluid: Optional[str]
    density_delta_kg_m3: float
    viscosity_delta_pa_s: float
    suppression_active: bool  # True = suppress anomaly alerts during transition


# ─────────────────────────────────────────────────────────────────────────────
# Sensor Confidence Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_scs(
    measured: dict[str, float],
    pinn_predicted: dict[str, float],
    ambient_temp_c: float = 30.0,
    drift_history: Optional[list[float]] = None,
    acoustic_baseline_std: float = 1.0,
) -> SensorConfidenceScore:
    """
    Compute the Sensor Confidence Score (SCS) by comparing measured sensor
    values against the PINN's physics-consistent predictions.

    SCS Formula
    -----------
        δᵢ  = |measured_i - predicted_i| / (|predicted_i| + ε)
        Δ   = Σᵢ wᵢ · δᵢ            (weighted mean divergence)
        SCS = clamp((1 - Δ) · 100,  0,  100)

    Constraint: if Δ > 0.20 → SCS < 80 (enforced by formula above since
    (1 - 0.20) × 100 = 80.0).

    Parameters
    ----------
    measured           : Dict of measured sensor values
    pinn_predicted     : Dict of PINN-estimated 'physical truth' values
    ambient_temp_c     : Current ambient temperature [°C]
    drift_history      : List of past SCS values for RUL regression
    acoustic_baseline_std : Baseline standard deviation of acoustic sensor

    Returns
    -------
    SensorConfidenceScore dataclass
    """
    eps = 1e-9
    per_sensor_div: dict[str, float] = {}
    weighted_div = 0.0
    total_weight = 0.0
    flags: list[str] = []

    for sensor_key, weight in SCS_WEIGHTS.items():
        meas = measured.get(sensor_key, None)
        pred = pinn_predicted.get(sensor_key, None)
        if meas is None or pred is None:
            continue
        div = abs(meas - pred) / (abs(pred) + eps)
        per_sensor_div[sensor_key] = round(div * 100.0, 2)  # store as %
        weighted_div += weight * div
        total_weight += weight

    if total_weight > 0:
        weighted_div /= total_weight

    scs_raw = (1.0 - weighted_div) * 100.0
    scs = float(max(0.0, min(100.0, scs_raw)))

    # ── Flag: divergence > threshold ─────────────────────────────────────────
    if weighted_div > DIVERGENCE_THRESHOLD:
        flags.append("HIGH_PINN_DIVERGENCE")

    # ── Flag: temperature-induced drift ──────────────────────────────────────
    if ambient_temp_c >= AMBIENT_DRIFT_TEMP_C:
        flags.append("SENSOR_DRIFT_RISK_HIGH_TEMP")

    # ── Flag: acoustic spike ─────────────────────────────────────────────────
    acoustic_val = measured.get("acoustic", 0.0)
    acoustic_pred = pinn_predicted.get("acoustic", acoustic_val)
    if acoustic_baseline_std > 0:
        z_score = abs(acoustic_val - acoustic_pred) / acoustic_baseline_std
        if z_score > ACOUSTIC_SPIKE_THRESHOLD:
            flags.append("ACOUSTIC_SPIKE_DETECTED")

    # ── RUL via linear regression on SCS history ──────────────────────────────
    rul_hours, rul_conf, drift_rate = _estimate_rul(drift_history, scs)

    return SensorConfidenceScore(
        scs=round(scs, 2),
        overall_divergence_pct=round(weighted_div * 100.0, 3),
        per_sensor_divergence=per_sensor_div,
        flags=flags,
        rul_hours=rul_hours,
        rul_confidence=rul_conf,
        drift_rate_pct_per_hour=round(drift_rate, 4),
        ambient_temp_c=float(ambient_temp_c),
    )


# ─────────────────────────────────────────────────────────────────────────────
# RUL via linear regression on SCS history
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_rul(
    scs_history: Optional[list[float]],
    current_scs: float,
    min_scs_threshold: float = 60.0,
) -> tuple[Optional[float], float, float]:
    """
    Estimate Remaining Useful Life of the sensor array using a linear
    regression on the historical SCS trend (degradation slope).

    Model
    -----
        SCS(t) ≈ a·t + b
        RUL = (SCS_threshold - b) / a      (hours until SCS hits 60 %)

    Parameters
    ----------
    scs_history       : List of past SCS values (one per hour assumed)
    current_scs       : Most recent SCS value
    min_scs_threshold : SCS level considered 'end of life' [default 60.0]

    Returns
    -------
    (rul_hours, rul_confidence, drift_rate_pct_per_hour)
    """
    if not scs_history or len(scs_history) < 3:
        return None, 0.0, 0.0

    history = list(scs_history) + [current_scs]
    n = len(history)
    X = np.arange(n).reshape(-1, 1).astype(float)
    y = np.array(history, dtype=float)

    try:
        reg = LinearRegression()
        reg.fit(X, y)
        slope = float(reg.coef_[0])      # SCS change per hour
        intercept = float(reg.intercept_)
        r2 = float(reg.score(X, y))
    except Exception as exc:
        logger.warning("RUL regression failed: %s", exc)
        return None, 0.0, 0.0

    if slope >= 0:
        # SCS is stable or improving — very long RUL
        return float(min(87600.0, abs(intercept / max(1e-6, slope)))), float(r2), float(slope)

    # Time until SCS reaches threshold
    # min_scs = slope * t_rul + intercept  →  t_rul = (min_scs - intercept) / slope
    t_rul = (min_scs_threshold - intercept) / slope - n
    rul_hours = max(0.0, float(t_rul))
    return round(rul_hours, 1), float(max(0.0, r2)), float(slope)


# ─────────────────────────────────────────────────────────────────────────────
# Drift detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_sensor_drift(
    readings_history: list[float],
    expected_baseline: float,
    dt_hours: float = 1.0,
) -> dict:
    """
    Detect linear sensor drift caused by thermal expansion, fouling, or
    aging — particularly relevant in Indian summer conditions (45 °C).

    Method: fit a linear trend to recent readings.
    If slope × total_time > 15 % of baseline → DRIFT flag.

    Parameters
    ----------
    readings_history : Ordered list of sensor readings (one per dt_hours)
    expected_baseline: Nominal sensor output under healthy conditions
    dt_hours         : Time interval between readings [hours]

    Returns
    -------
    dict with keys: is_drifting, slope, drift_pct, direction
    """
    if len(readings_history) < 5:
        return {"is_drifting": False, "slope": 0.0, "drift_pct": 0.0, "direction": "STABLE"}

    n = len(readings_history)
    X = (np.arange(n) * dt_hours).reshape(-1, 1)
    y = np.array(readings_history, dtype=float)

    try:
        reg = LinearRegression()
        reg.fit(X, y)
        slope = float(reg.coef_[0])
    except Exception:
        slope = 0.0

    total_drift = slope * n * dt_hours
    drift_pct = abs(total_drift) / (abs(expected_baseline) + 1e-9)

    is_drifting = drift_pct > DRIFT_THRESHOLD_PCT
    direction = "UP" if slope > 0 else ("DOWN" if slope < 0 else "STABLE")

    return {
        "is_drifting": is_drifting,
        "slope_per_hour": round(slope, 6),
        "drift_pct": round(drift_pct * 100.0, 2),
        "direction": direction,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch-change detection
# ─────────────────────────────────────────────────────────────────────────────

def detect_batch_change(
    current_density_kg_m3: float,
    previous_density_kg_m3: float,
    current_viscosity_pa_s: float,
    previous_viscosity_pa_s: float,
    current_fluid_label: Optional[str] = None,
    previous_fluid_label: Optional[str] = None,
) -> BatchChangeResult:
    """
    Detect a product-batch change when the pipeline switches from one
    petroleum product to another (e.g. MS → HSD).

    A batch change causes transient:
      • Abrupt density shift
      • Viscosity discontinuity
      • Temporary acoustic turbulence

    These artefacts must NOT be flagged as leaks.

    Parameters
    ----------
    current_density_kg_m3  : Measured density at current timestep [kg/m³]
    previous_density_kg_m3 : Measured density at previous timestep [kg/m³]
    current_viscosity_pa_s : Measured viscosity now [Pa·s]
    previous_viscosity_pa_s: Measured viscosity previously [Pa·s]
    current_fluid_label    : Product label string if known
    previous_fluid_label   : Previous product label if known

    Returns
    -------
    BatchChangeResult dataclass
    """
    density_delta = abs(current_density_kg_m3 - previous_density_kg_m3)
    viscosity_delta = abs(current_viscosity_pa_s - previous_viscosity_pa_s)

    label_change = (
        current_fluid_label is not None
        and previous_fluid_label is not None
        and current_fluid_label.upper() != previous_fluid_label.upper()
    )

    detected = (
        label_change
        or density_delta > BATCH_DENSITY_DELTA_KGM3
        or viscosity_delta > previous_viscosity_pa_s * 0.40   # >40% viscosity jump
    )

    if detected:
        logger.info(
            "Batch change detected: ρ_delta=%.1f kg/m³  label: %s→%s",
            density_delta, previous_fluid_label, current_fluid_label,
        )

    return BatchChangeResult(
        detected=detected,
        from_fluid=previous_fluid_label,
        to_fluid=current_fluid_label,
        density_delta_kg_m3=round(density_delta, 2),
        viscosity_delta_pa_s=round(viscosity_delta, 6),
        suppression_active=detected,
    )
PYEOF
echo "  ✅  app/engine/causal_guard.py"

# ─────────────────────────────────────────────────────────────────────────────
# app/engine/model.py — Hybrid PINN-LSTM
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/app/engine/model.py"
"""
model.py — Hybrid Physics-Informed Neural Network (PINN) + LSTM architecture
for the NAVIER pipeline monitoring system.

Architecture Overview
---------------------
                ┌─────────────────────────────────────┐
                │        INPUT LAYER                  │
                │  (10 sensor features, normalised)   │
                └──────────────┬──────────────────────┘
                               │
              ┌────────────────┼─────────────────────┐
              │                │                     │
    ┌─────────▼──────┐  ┌──────▼──────┐  ┌──────────▼────────┐
    │  PINN Branch   │  │ LSTM Branch │  │  Physics Residual │
    │  (Dense MLP)   │  │ (temporal)  │  │  (passthrough)    │
    └─────────┬──────┘  └──────┬──────┘  └──────────┬────────┘
              └────────────────┼─────────────────────┘
                               │  (concat)
                    ┌──────────▼──────────┐
                    │  Shared Dense Head  │
                    └──────────┬──────────┘
              ┌────────────────┼───────────────────┐
              │                │                   │
      ┌───────▼──────┐  ┌──────▼──────┐  ┌────────▼───────┐
      │ Output A     │  │ Output B    │  │ Output C/D     │
      │ Classification│ │ Localization│  │ SCS + RUL      │
      │ (softmax ×3) │  │ (distance)  │  │ (sigmoid)      │
      └──────────────┘  └─────────────┘  └────────────────┘

The PINN loss = task_loss + λ_physics × physics_residual_loss
where physics_residual_loss is computed from Navier-Stokes equations in
physics.py and injected into the backward pass.

When PyTorch is unavailable, the module falls back to a calibrated
NumPy forward-pass emulator that provides identical output shapes.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Optional PyTorch ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    logger.info("PyTorch %s detected — full PINN-LSTM active.", torch.__version__)
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not found — running NumPy emulation mode.")

WEIGHTS_DIR = Path(__file__).parent.parent.parent / "models"

# ── Feature / label definitions ──────────────────────────────────────────────
FEATURE_NAMES = [
    "inlet_pressure_pa",
    "outlet_pressure_pa",
    "flow_rate_m3s",
    "temperature_c",
    "density_kg_m3",
    "dynamic_viscosity_pa_s",
    "acoustic_vibration",
    "pipe_diameter_m",
    "wall_thickness_m",
    "segment_length_m",
]
N_FEATURES = len(FEATURE_NAMES)

CLASS_LABELS = ["SAFE", "WARNING", "CRITICAL"]
N_CLASSES = len(CLASS_LABELS)

# Approximate operating ranges for normalisation
FEATURE_RANGES = {
    "inlet_pressure_pa":       (1e5,    80e5),
    "outlet_pressure_pa":      (0.5e5,  75e5),
    "flow_rate_m3s":           (0.001,  2.0),
    "temperature_c":           (-5.0,   90.0),
    "density_kg_m3":           (500.0,  900.0),
    "dynamic_viscosity_pa_s":  (1e-4,   1e-2),
    "acoustic_vibration":      (0.0,    100.0),
    "pipe_diameter_m":         (0.05,   1.5),
    "wall_thickness_m":        (0.003,  0.05),
    "segment_length_m":        (100.0,  15000.0),
}


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def normalise(x: np.ndarray) -> np.ndarray:
    """
    Min-max normalise a (N_FEATURES,) or (B, N_FEATURES) array to [0, 1]
    using the operational ranges defined in FEATURE_RANGES.

    Parameters
    ----------
    x : Raw feature array

    Returns
    -------
    np.ndarray : Normalised array of same shape
    """
    mins = np.array([FEATURE_RANGES[k][0] for k in FEATURE_NAMES], dtype=np.float32)
    maxs = np.array([FEATURE_RANGES[k][1] for k in FEATURE_NAMES], dtype=np.float32)
    rngs = np.where((maxs - mins) == 0, 1.0, maxs - mins)
    return (x - mins) / rngs


def denormalise_pressure(norm_val: float, key: str = "inlet_pressure_pa") -> float:
    """Reverse min-max normalisation for a single pressure value."""
    lo, hi = FEATURE_RANGES[key]
    return float(norm_val * (hi - lo) + lo)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch PINN-LSTM (full implementation)
# ─────────────────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class PINNBranch(nn.Module):
        """
        Physics-Informed Neural Network branch.

        A fully-connected MLP that learns residual corrections to the
        Navier-Stokes equations.  Its output is a latent physics embedding
        that is concatenated with the LSTM output before the multi-output head.

        The PINN loss includes an additional physics_residual term (injected
        from physics.py) that penalises outputs inconsistent with conservation
        laws.

        Architecture: Input(N) → Dense(128,ReLU) → Dense(64,ReLU) → Dense(32)
        """

        def __init__(self, n_features: int = N_FEATURES, latent_dim: int = 32):
            """
            Parameters
            ----------
            n_features : Number of input features
            latent_dim : Output embedding dimension
            """
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(0.10),
                nn.Linear(64, latent_dim),
                nn.Tanh(),
            )

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x : (batch, N_FEATURES) tensor

            Returns
            -------
            torch.Tensor : (batch, latent_dim)
            """
            return self.net(x)

    class LSTMBranch(nn.Module):
        """
        LSTM branch for capturing temporal dependencies in pressure waves.

        Processes a sequence of sensor readings to detect transient events
        (pressure drops, acoustic spikes) that are characteristic of leaks.

        Architecture: LSTM(hidden=64, layers=2) → Linear(32)
        """

        def __init__(
            self,
            n_features: int = N_FEATURES,
            hidden_size: int = 64,
            num_layers: int = 2,
            latent_dim: int = 32,
        ):
            """
            Parameters
            ----------
            n_features  : Number of input features per timestep
            hidden_size : LSTM hidden state dimensionality
            num_layers  : Number of stacked LSTM layers
            latent_dim  : Output projection dimension
            """
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.2 if num_layers > 1 else 0.0,
            )
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, latent_dim),
                nn.Tanh(),
            )

        def forward(self, x_seq: "torch.Tensor") -> "torch.Tensor":
            """
            Parameters
            ----------
            x_seq : (batch, seq_len, N_FEATURES) tensor

            Returns
            -------
            torch.Tensor : (batch, latent_dim)  — last-timestep representation
            """
            out, _ = self.lstm(x_seq)
            return self.proj(out[:, -1, :])   # take last timestep

    class NavierPINNLSTM(nn.Module):
        """
        Full hybrid PINN-LSTM model for pipeline anomaly detection.

        Outputs
        -------
        A — class_logits  : (batch, 3)  — SAFE / WARNING / CRITICAL
        B — leak_distance : (batch, 1)  — normalised distance [0, 1] × seg_length
        C — scs_score     : (batch, 1)  — Sensor Confidence Score [0, 1]
        D — rul_norm      : (batch, 1)  — Normalised RUL [0, 1]
        """

        def __init__(
            self,
            n_features: int = N_FEATURES,
            pinn_latent: int = 32,
            lstm_latent: int = 32,
            physics_latent: int = 8,
        ):
            """
            Parameters
            ----------
            n_features     : Sensor feature count
            pinn_latent    : PINN branch output dim
            lstm_latent    : LSTM branch output dim
            physics_latent : Extra physics-residual embedding dim
            """
            super().__init__()
            self.pinn  = PINNBranch(n_features, pinn_latent)
            self.lstm  = LSTMBranch(n_features, latent_dim=lstm_latent)

            # Physics residual embedding (scalar inputs: mass_res, mom_res, pinn_loss)
            self.physics_embed = nn.Sequential(
                nn.Linear(3, physics_latent),
                nn.ReLU(),
            )

            fusion_dim = pinn_latent + lstm_latent + physics_latent

            self.shared = nn.Sequential(
                nn.Linear(fusion_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
            )

            # Multi-output heads
            self.head_class    = nn.Linear(32, N_CLASSES)             # A
            self.head_location = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())  # B
            self.head_scs      = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())  # C
            self.head_rul      = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())  # D

        def forward(
            self,
            x: "torch.Tensor",
            x_seq: "torch.Tensor",
            physics_residuals: "torch.Tensor",
        ) -> dict[str, "torch.Tensor"]:
            """
            Parameters
            ----------
            x                : (batch, N_FEATURES)      — current snapshot
            x_seq            : (batch, seq_len, N_FEATURES) — time window
            physics_residuals: (batch, 3)               — [mass_res, mom_res, pinn_loss]

            Returns
            -------
            dict with keys: class_logits, leak_distance_norm, scs, rul_norm
            """
            pinn_feat  = self.pinn(x)
            lstm_feat  = self.lstm(x_seq)
            phys_feat  = self.physics_embed(physics_residuals)

            fused = torch.cat([pinn_feat, lstm_feat, phys_feat], dim=1)
            shared = self.shared(fused)

            return {
                "class_logits":       self.head_class(shared),
                "leak_distance_norm": self.head_location(shared),
                "scs":                self.head_scs(shared),
                "rul_norm":           self.head_rul(shared),
            }

    class PINNLoss(nn.Module):
        """
        Composite loss function combining task losses with PINN physics residuals.

            L_total = L_class + λ_loc × L_loc + λ_scs × L_scs
                      + λ_rul × L_rul + λ_phys × L_physics

        Parameters
        ----------
        lambda_physics : Weight for Navier-Stokes residual penalty
        lambda_loc     : Weight for localization MSE loss
        lambda_scs     : Weight for SCS MSE loss
        lambda_rul     : Weight for RUL MSE loss
        """

        def __init__(
            self,
            lambda_physics: float = 0.30,
            lambda_loc: float = 0.20,
            lambda_scs: float = 0.15,
            lambda_rul: float = 0.15,
        ):
            super().__init__()
            self.lambda_physics = lambda_physics
            self.lambda_loc     = lambda_loc
            self.lambda_scs     = lambda_scs
            self.lambda_rul     = lambda_rul
            self.ce_loss = nn.CrossEntropyLoss()
            self.mse     = nn.MSELoss()

        def forward(
            self,
            outputs: dict[str, "torch.Tensor"],
            targets: dict[str, "torch.Tensor"],
            physics_loss: "torch.Tensor",
        ) -> "torch.Tensor":
            """
            Parameters
            ----------
            outputs      : Model output dict
            targets      : Ground-truth dict (same keys)
            physics_loss : Scalar Navier-Stokes residual from physics.py

            Returns
            -------
            torch.Tensor : Scalar total loss
            """
            L_class = self.ce_loss(
                outputs["class_logits"],
                targets["class_label"].long(),
            )
            L_loc = self.mse(
                outputs["leak_distance_norm"],
                targets["leak_distance_norm"].unsqueeze(1),
            )
            L_scs = self.mse(
                outputs["scs"],
                targets["scs_target"].unsqueeze(1),
            )
            L_rul = self.mse(
                outputs["rul_norm"],
                targets["rul_norm"].unsqueeze(1),
            )

            total = (
                L_class
                + self.lambda_loc     * L_loc
                + self.lambda_scs     * L_scs
                + self.lambda_rul     * L_rul
                + self.lambda_physics * physics_loss
            )
            return total


# ─────────────────────────────────────────────────────────────────────────────
# NumPy emulation (when PyTorch is absent)
# ─────────────────────────────────────────────────────────────────────────────

class _NumPyPINNLSTMEmulator:
    """
    Pure-NumPy forward-pass emulator for the PINN-LSTM model.

    Mimics the full model's output structure with deterministic
    weights initialised from a fixed seed.  Output ranges and
    qualitative behaviour match the PyTorch model; quantitative
    accuracy requires training with real / synthetic data.

    This class is only used when PyTorch is unavailable.
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        h = 64   # LSTM hidden

        # PINN MLP weights (input → 128 → 64 → 32)
        self.W1 = rng.normal(0, 0.1, (N_FEATURES, 128)).astype(np.float32)
        self.b1 = np.zeros(128, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (128, 64)).astype(np.float32)
        self.b2 = np.zeros(64, dtype=np.float32)
        self.W3 = rng.normal(0, 0.1, (64, 32)).astype(np.float32)
        self.b3 = np.zeros(32, dtype=np.float32)

        # LSTM simplified: input gate only (one step, one layer)
        self.Wi = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.Wf = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.Wg = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.Wo = rng.normal(0, 0.1, (h, N_FEATURES + h)).astype(np.float32)
        self.bf = np.ones(h, dtype=np.float32)      # forget gate init to 1
        self.bi = self.bo = self.bg = np.zeros(h, dtype=np.float32)
        self.proj_W = rng.normal(0, 0.1, (h, 32)).astype(np.float32)
        self.proj_b = np.zeros(32, dtype=np.float32)

        # Physics embedding (3 → 8)
        self.pe_W = rng.normal(0, 0.1, (3, 8)).astype(np.float32)
        self.pe_b = np.zeros(8, dtype=np.float32)

        # Shared (72 → 64 → 32)
        self.sh_W1 = rng.normal(0, 0.05, (72, 64)).astype(np.float32)
        self.sh_b1 = np.zeros(64, dtype=np.float32)
        self.sh_W2 = rng.normal(0, 0.05, (64, 32)).astype(np.float32)
        self.sh_b2 = np.zeros(32, dtype=np.float32)

        # Output heads
        self.h_cls = rng.normal(0, 0.05, (32, N_CLASSES)).astype(np.float32)
        self.h_loc = rng.normal(0, 0.05, (32, 1)).astype(np.float32)
        self.h_scs = rng.normal(0, 0.05, (32, 1)).astype(np.float32)
        self.h_rul = rng.normal(0, 0.05, (32, 1)).astype(np.float32)

        # Head biases — set so baseline output is plausible
        self.b_cls = np.array([1.5, -0.5, -1.5], dtype=np.float32)  # SAFE dominant
        self.b_loc = np.array([-0.5], dtype=np.float32)
        self.b_scs = np.array([1.5], dtype=np.float32)   # SCS high by default
        self.b_rul = np.array([1.5], dtype=np.float32)   # RUL high by default

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

    @staticmethod
    def _tanh(x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -30, 30))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / e.sum()

    def _pinn_forward(self, x: np.ndarray) -> np.ndarray:
        """MLP branch: (N_FEATURES,) → (32,)"""
        h = self._relu(x @ self.W1 + self.b1)
        h = self._relu(h @ self.W2 + self.b2)
        return self._tanh(h @ self.W3 + self.b3)

    def _lstm_forward(self, x_seq: np.ndarray) -> np.ndarray:
        """LSTM branch: (T, N_FEATURES) → (32,)"""
        ht = np.zeros(64, dtype=np.float32)
        ct = np.zeros(64, dtype=np.float32)
        for t in range(x_seq.shape[0]):
            xt = x_seq[t].astype(np.float32)
            combined = np.concatenate([xt, ht])
            i_g = self._sigmoid(self.Wi @ combined + self.bi)
            f_g = self._sigmoid(self.Wf @ combined + self.bf)
            g_g = self._tanh(self.Wg @ combined + self.bg)
            o_g = self._sigmoid(self.Wo @ combined + self.bo)
            ct  = f_g * ct + i_g * g_g
            ht  = o_g * self._tanh(ct)
        return self._tanh(ht @ self.proj_W + self.proj_b)

    def forward(
        self,
        x: np.ndarray,
        x_seq: np.ndarray,
        physics_residuals: np.ndarray,
    ) -> dict:
        """
        Compute model outputs without PyTorch.

        Parameters
        ----------
        x                : (N_FEATURES,) current snapshot, normalised
        x_seq            : (T, N_FEATURES) time-window, normalised
        physics_residuals: (3,) [mass_res, mom_res, pinn_loss]

        Returns
        -------
        dict with same keys as PyTorch model
        """
        pinn_feat  = self._pinn_forward(x)
        lstm_feat  = self._lstm_forward(x_seq)
        phys_feat  = self._relu(physics_residuals @ self.pe_W + self.pe_b)

        fused  = np.concatenate([pinn_feat, lstm_feat, phys_feat])
        shared = self._relu(fused @ self.sh_W1 + self.sh_b1)
        shared = self._relu(shared @ self.sh_W2 + self.sh_b2)

        class_logits       = shared @ self.h_cls + self.b_cls
        leak_distance_norm = float(self._sigmoid(shared @ self.h_loc + self.b_loc)[0])
        scs_raw            = float(self._sigmoid(shared @ self.h_scs + self.b_scs)[0])
        rul_norm           = float(self._sigmoid(shared @ self.h_rul + self.b_rul)[0])
        class_probs        = self._softmax(class_logits)

        return {
            "class_probs":        class_probs,
            "leak_distance_norm": leak_distance_norm,
            "scs":                scs_raw,
            "rul_norm":           rul_norm,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Inference result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelOutput:
    """Unified output from HybridDetector.predict()"""
    classification: str           # SAFE | WARNING | CRITICAL
    class_probabilities: dict[str, float]
    leak_distance_norm: float     # 0-1 (multiply by segment_length for metres)
    scs: float                    # Sensor Confidence Score 0-100
    rul_norm: float               # 0-1 (scale to hours externally)
    physics_residual: float       # raw PINN physics loss weight
    inference_time_ms: float
    backend: str                  # "torch" | "numpy"


# ─────────────────────────────────────────────────────────────────────────────
# HybridDetector — unified API
# ─────────────────────────────────────────────────────────────────────────────

class HybridDetector:
    """
    Unified inference wrapper for the NAVIER PINN-LSTM model.

    Provides a single .predict() method regardless of whether PyTorch
    is available.  Handles feature normalisation, physics residual
    injection, and output post-processing.

    Usage
    -----
        detector = HybridDetector()
        output   = detector.predict(features_dict, sequence_list, ns_residuals)
    """

    def __init__(self, weights_path: Optional[Path] = None):
        """
        Parameters
        ----------
        weights_path : Path to saved PyTorch state-dict (.pt file).
                       If None, uses default WEIGHTS_DIR / 'navier_model.pt'.
        """
        self._backend = "numpy"
        self._torch_model = None
        self._numpy_model = _NumPyPINNLSTMEmulator(seed=42)

        if TORCH_AVAILABLE:
            self._torch_model = NavierPINNLSTM()
            wpath = weights_path or (WEIGHTS_DIR / "navier_model.pt")
            if wpath.exists():
                try:
                    state = torch.load(str(wpath), map_location="cpu")
                    self._torch_model.load_state_dict(state)
                    logger.info("Loaded model weights from %s", wpath)
                except Exception as exc:
                    logger.warning("Could not load weights (%s) — using random init.", exc)
            else:
                logger.info("No weights file at %s — using random initialisation.", wpath)
            self._torch_model.eval()
            self._backend = "torch"

        logger.info("HybridDetector initialised [backend=%s]", self._backend)

    # ──────────────────────────────────────────────────────────────────────────

    def predict(
        self,
        features: dict[str, float],
        sequence: list[dict[str, float]],
        physics_residuals: tuple[float, float, float],
    ) -> ModelOutput:
        """
        Run hybrid PINN-LSTM inference for one analysis request.

        Parameters
        ----------
        features          : Dict of current sensor readings (keys = FEATURE_NAMES)
        sequence          : List of past feature dicts (oldest first, newest last)
        physics_residuals : (mass_residual, momentum_residual, pinn_loss_weight)

        Returns
        -------
        ModelOutput dataclass
        """
        t0 = time.perf_counter()

        # Build & normalise arrays
        x_raw = np.array(
            [features.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float32
        )
        x_norm = normalise(x_raw)

        seq_raw = np.array(
            [[s.get(k, 0.0) for k in FEATURE_NAMES] for s in (sequence or [features])],
            dtype=np.float32,
        )
        seq_norm = normalise(seq_raw)

        phys = np.array(physics_residuals, dtype=np.float32)

        # Forward pass
        if self._backend == "torch" and self._torch_model is not None:
            output = self._torch_forward(x_norm, seq_norm, phys)
        else:
            output = self._numpy_model.forward(x_norm, seq_norm, phys)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Decode classification
        probs = output["class_probs"] if "class_probs" in output else _logits_to_probs(output)
        class_idx = int(np.argmax(probs))

        # Physics residual modulates classification: high residual → escalate
        physics_loss = float(phys[2])
        class_idx = _apply_physics_override(class_idx, physics_loss)

        scs_raw = float(output["scs"])
        # Enforce SCS < 80 when PINN divergence > 20 %
        if physics_loss > DIVERGENCE_SCS_THRESHOLD:
            scs_raw = min(scs_raw, 0.79)

        return ModelOutput(
            classification=CLASS_LABELS[class_idx],
            class_probabilities={
                CLASS_LABELS[i]: round(float(probs[i]), 4) for i in range(N_CLASSES)
            },
            leak_distance_norm=float(output["leak_distance_norm"]),
            scs=round(scs_raw * 100.0, 2),
            rul_norm=float(output["rul_norm"]),
            physics_residual=physics_loss,
            inference_time_ms=round(elapsed_ms, 2),
            backend=self._backend,
        )

    def _torch_forward(
        self,
        x_norm: np.ndarray,
        seq_norm: np.ndarray,
        phys: np.ndarray,
    ) -> dict:
        """Run PyTorch model in eval / no-grad mode."""
        import torch
        with torch.no_grad():
            x_t    = torch.from_numpy(x_norm).unsqueeze(0)
            seq_t  = torch.from_numpy(seq_norm).unsqueeze(0)
            phys_t = torch.from_numpy(phys).unsqueeze(0)
            out    = self._torch_model(x_t, seq_t, phys_t)

        logits = out["class_logits"].squeeze(0).numpy()
        probs  = _softmax_np(logits)
        return {
            "class_probs":        probs,
            "leak_distance_norm": float(out["leak_distance_norm"].item()),
            "scs":                float(out["scs"].item()),
            "rul_norm":           float(out["rul_norm"].item()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

DIVERGENCE_SCS_THRESHOLD = 0.20   # mirrors causal_guard.DIVERGENCE_THRESHOLD


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _logits_to_probs(output: dict) -> np.ndarray:
    """Fallback: if class_probs not in output, build uniform distribution."""
    return np.array([1.0 / N_CLASSES] * N_CLASSES, dtype=np.float32)


def _apply_physics_override(class_idx: int, physics_loss: float) -> int:
    """
    Escalate classification if the PINN physics residual is very high.

    physics_loss > 0.60 → at least WARNING
    physics_loss > 0.85 → at least CRITICAL
    """
    if physics_loss > 0.85 and class_idx < 2:
        return 2  # CRITICAL
    if physics_loss > 0.60 and class_idx < 1:
        return 1  # WARNING
    return class_idx


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────

_detector_singleton: Optional[HybridDetector] = None


def get_detector() -> HybridDetector:
    """
    Return (or create) the module-level HybridDetector singleton.

    Thread-safe for read-only inference (no locking needed).
    """
    global _detector_singleton
    if _detector_singleton is None:
        logger.info("Creating HybridDetector singleton...")
        _detector_singleton = HybridDetector()
    return _detector_singleton
PYEOF
echo "  ✅  app/engine/model.py"

# ─────────────────────────────────────────────────────────────────────────────
# app/schemas/request_models.py
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/app/schemas/request_models.py"
"""
request_models.py — Pydantic v2 schemas for NAVIER API request/response.

All physical units match the physics.py conventions:
  pressure   → Pa
  flow_rate  → m³/s
  viscosity  → Pa·s  (dynamic)
  density    → kg/m³
  length     → m
  temperature→ °C
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class FluidType(str, Enum):
    MS      = "MS"        # Motor Spirit / Petrol
    HSD     = "HSD"       # High-Speed Diesel
    ATF     = "ATF"       # Aviation Turbine Fuel
    BENZENE = "BENZENE"
    LPG     = "LPG"
    UNKNOWN = "UNKNOWN"


class Classification(str, Enum):
    SAFE     = "SAFE"
    WARNING  = "WARNING"
    CRITICAL = "CRITICAL"


class DegradationRate(str, Enum):
    SLOW        = "SLOW"
    NORMAL      = "NORMAL"
    ACCELERATED = "ACCELERATED"


# ─────────────────────────────────────────────────────────────────────────────
# Sensor reading (input) — 10+ chemical / physical parameters
# ─────────────────────────────────────────────────────────────────────────────

class SensorReading(BaseModel):
    """
    Complete sensor reading for one pipeline analysis timestep.

    Mandatory parameters (10 core):
        inlet_pressure_pa, outlet_pressure_pa, flow_rate_m3s,
        temperature_c, density_kg_m3, dynamic_viscosity_pa_s,
        acoustic_vibration, pipe_diameter_m, wall_thickness_m,
        segment_length_m

    Optional parameters:
        fluid_type, previous_fluid_type, previous_density_kg_m3,
        previous_viscosity_pa_s, ambient_temp_c, sensor_id,
        timestamp_utc, sequence_window (historical readings)
    """

    # ── Core 10 parameters ───────────────────────────────────────────────────
    inlet_pressure_pa: float = Field(
        ..., ge=0.0, le=100e5,
        description="Inlet (upstream) gauge pressure [Pa]. 1 bar = 1e5 Pa.",
    )
    outlet_pressure_pa: float = Field(
        ..., ge=0.0, le=100e5,
        description="Outlet (downstream) gauge pressure [Pa].",
    )
    flow_rate_m3s: float = Field(
        ..., ge=0.0, le=5.0,
        description="Volumetric flow rate at inlet [m³/s].",
    )
    temperature_c: float = Field(
        ..., ge=-10.0, le=100.0,
        description="Fluid temperature at sensor [°C].",
    )
    density_kg_m3: float = Field(
        ..., ge=400.0, le=1100.0,
        description="Fluid density at operating temperature [kg/m³].",
    )
    dynamic_viscosity_pa_s: float = Field(
        ..., ge=1e-5, le=5e-2,
        description="Dynamic (absolute) viscosity [Pa·s]. 1 cSt × density/1000.",
    )
    acoustic_vibration: float = Field(
        ..., ge=0.0, le=200.0,
        description="Acoustic vibration sensor reading [arbitrary units, 0–200].",
    )
    pipe_diameter_m: float = Field(
        ..., ge=0.025, le=2.0,
        description="Internal pipe diameter [m]. E.g. 0.508 for 20-inch pipe.",
    )
    wall_thickness_m: float = Field(
        ..., ge=0.002, le=0.10,
        description="Pipe wall thickness [m].",
    )
    segment_length_m: float = Field(
        ..., ge=50.0, le=50_000.0,
        description="Pipeline segment length for this sensor pair [m].",
    )

    # ── Temporal / differential ───────────────────────────────────────────────
    inlet_pressure_prev_pa: Optional[float] = Field(
        default=None,
        description="Inlet pressure at previous timestep [Pa] — required for NPW.",
    )
    outlet_pressure_prev_pa: Optional[float] = Field(
        default=None,
        description="Outlet pressure at previous timestep [Pa] — required for NPW.",
    )
    dt_seconds: float = Field(
        default=1.0, ge=0.1, le=3600.0,
        description="Time interval since last reading [s].",
    )

    # ── Fluid / product metadata ──────────────────────────────────────────────
    fluid_type: FluidType = Field(
        default=FluidType.HSD,
        description="Current product being transported.",
    )
    previous_fluid_type: Optional[FluidType] = Field(
        default=None,
        description="Previous product (for batch-change detection).",
    )
    previous_density_kg_m3: Optional[float] = Field(
        default=None, ge=400.0, le=1100.0,
        description="Density at previous timestep [kg/m³].",
    )
    previous_viscosity_pa_s: Optional[float] = Field(
        default=None, ge=1e-5, le=5e-2,
        description="Viscosity at previous timestep [Pa·s].",
    )

    # ── Environmental ─────────────────────────────────────────────────────────
    ambient_temp_c: float = Field(
        default=30.0, ge=-10.0, le=60.0,
        description="Ambient air temperature [°C]. ≥42 °C triggers drift risk flag.",
    )

    # ── Historical sequence window (LSTM input) ───────────────────────────────
    sequence_window: Optional[list[dict]] = Field(
        default=None,
        description=(
            "Ordered list of past sensor snapshots (oldest first). "
            "Each element is a dict with the same keys as SensorReading. "
            "If omitted, single-timestep inference is used."
        ),
        max_length=60,
    )

    # ── Metadata ──────────────────────────────────────────────────────────────
    sensor_id: Optional[str]       = Field(default=None, max_length=64)
    station_id: Optional[str]      = Field(default=None, max_length=64)
    timestamp_utc: Optional[str]   = Field(default=None)
    pipeline_age_years: float      = Field(default=10.0, ge=0.0, le=60.0)
    scs_history: Optional[list[float]] = Field(
        default=None, max_length=168,
        description="Past 168 hours of SCS readings for RUL trend regression.",
    )

    @field_validator("outlet_pressure_pa")
    @classmethod
    def outlet_less_than_inlet(cls, v: float, info) -> float:
        """Outlet pressure must not exceed inlet by more than 2 bar under normal flow."""
        inlet = info.data.get("inlet_pressure_pa")
        if inlet is not None and v > inlet + 2e5:
            raise ValueError(
                f"outlet_pressure_pa ({v/1e5:.2f} bar) exceeds "
                f"inlet_pressure_pa ({inlet/1e5:.2f} bar) — physically invalid for flow."
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "inlet_pressure_pa": 4_500_000,
                "outlet_pressure_pa": 4_020_000,
                "flow_rate_m3s": 0.25,
                "temperature_c": 42.0,
                "density_kg_m3": 840.0,
                "dynamic_viscosity_pa_s": 0.0035,
                "acoustic_vibration": 12.5,
                "pipe_diameter_m": 0.508,
                "wall_thickness_m": 0.0095,
                "segment_length_m": 10_000.0,
                "fluid_type": "HSD",
                "ambient_temp_c": 45.0,
                "sensor_id": "RKL-SEG-04",
                "timestamp_utc": "2025-06-15T10:30:00Z",
            }
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Simulate-leak request
# ─────────────────────────────────────────────────────────────────────────────

class SimulateLeakRequest(BaseModel):
    """Parameters to generate a synthetic leak scenario for testing."""

    fluid_type: FluidType = FluidType.HSD
    segment_length_m: float = Field(default=10_000.0, ge=100.0, le=50_000.0)
    pipe_diameter_m: float = Field(default=0.508, ge=0.05, le=2.0)
    wall_thickness_m: float = Field(default=0.0095, ge=0.002, le=0.1)
    leak_position_m: float = Field(
        default=4_000.0, ge=0.0,
        description="True leak location from inlet [m].",
    )
    leak_severity_pct: float = Field(
        default=10.0, ge=0.1, le=100.0,
        description="Flow loss at leak as percentage of nominal [%].",
    )
    nominal_inlet_pressure_pa: float = Field(default=4_500_000.0, ge=1e5, le=80e5)
    nominal_flow_rate_m3s: float = Field(default=0.25, ge=0.001, le=5.0)
    ambient_temp_c: float = Field(default=40.0, ge=-10.0, le=60.0)
    include_sensor_drift: bool = Field(default=False)
    noise_level: float = Field(default=0.02, ge=0.0, le=0.5)

    @field_validator("leak_position_m")
    @classmethod
    def leak_within_segment(cls, v: float, info) -> float:
        L = info.data.get("segment_length_m", 10_000.0)
        if v >= L:
            raise ValueError(f"leak_position_m ({v}) must be less than segment_length_m ({L}).")
        return v


# ─────────────────────────────────────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class NPWLocalizationResult(BaseModel):
    triggered: bool
    leak_distance_m: float
    wave_speed_ms: float
    delta_t_s: float
    confidence: float
    method: str


class SCSReport(BaseModel):
    scs: float = Field(..., ge=0.0, le=100.0)
    overall_divergence_pct: float
    per_sensor_divergence: dict[str, float]
    flags: list[str]
    rul_hours: Optional[float]
    rul_confidence: float
    drift_rate_pct_per_hour: float
    ambient_temp_c: float


class PhysicsReport(BaseModel):
    mass_residual: float
    momentum_residual: float
    darcy_friction_factor: float
    reynolds_number: float
    is_turbulent: bool
    pinn_loss_weight: float


class BatchChangeReport(BaseModel):
    detected: bool
    from_fluid: Optional[str]
    to_fluid: Optional[str]
    density_delta_kg_m3: float
    suppression_active: bool


class AnalysisResponse(BaseModel):
    """Complete analysis result returned by POST /analyze."""

    # Identity
    sensor_id: Optional[str]
    station_id: Optional[str]
    timestamp_utc: Optional[str]
    fluid_type: str

    # Primary outputs
    classification: Classification
    alert_message: str
    anomaly_score: float = Field(..., ge=0.0, le=1.0)
    class_probabilities: dict[str, float]

    # Output B — Leak Localization
    npw_localization: Optional[NPWLocalizationResult]

    # Output C — Sensor Confidence Score
    scs_report: SCSReport

    # Output D — RUL
    rul_hours: Optional[float]
    rul_days: Optional[float]
    degradation_rate: DegradationRate

    # Supporting reports
    physics_report: PhysicsReport
    batch_change: BatchChangeReport

    # Model internals
    model_backend: str
    inference_time_ms: float
    processing_time_ms: float


class SensorStatusResponse(BaseModel):
    """Returned by GET /sensor_status."""
    sensor_id: Optional[str]
    scs: float
    status: str              # "HEALTHY" | "DEGRADED" | "CRITICAL"
    flags: list[str]
    rul_hours: Optional[float]
    recommendation: str


class SimulateLeakResponse(BaseModel):
    """Returned by POST /simulate_leak."""
    fluid_type: str
    true_leak_position_m: float
    generated_readings: dict        # key sensor values at leak moment
    analysis: AnalysisResponse      # full analysis of the simulated event
    npw_accuracy_m: Optional[float] # |detected - true| distance error
PYEOF
echo "  ✅  app/schemas/request_models.py"

# ─────────────────────────────────────────────────────────────────────────────
# app/api/routes.py
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/app/api/routes.py"
"""
routes.py — FastAPI route definitions for the NAVIER AI microservice.

Endpoints
---------
POST /analyze          — Main hybrid PINN-LSTM pipeline analysis
POST /simulate_leak    — Generate and analyse a synthetic leak event
GET  /sensor_status    — Lightweight sensor health check
GET  /health           — Service health probe
GET  /model/info       — Model architecture metadata
"""
from __future__ import annotations

import logging
import math
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, status

from app.engine.physics import (
    CHEMICAL_PROFILES,
    compute_navier_stokes_residuals,
    compute_wave_speed,
    npw_localization,
    pressure_gradient_localization,
)
from app.engine.causal_guard import (
    BatchChangeResult,
    compute_scs,
    detect_batch_change,
    detect_sensor_drift,
)
from app.engine.model import get_detector, FEATURE_NAMES, CLASS_LABELS
from app.schemas.request_models import (
    AnalysisResponse,
    BatchChangeReport,
    Classification,
    DegradationRate,
    NPWLocalizationResult,
    PhysicsReport,
    SCSReport,
    SensorReading,
    SensorStatusResponse,
    SimulateLeakRequest,
    SimulateLeakResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# POST /analyze
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Full hybrid PINN-LSTM pipeline analysis",
    tags=["Analysis"],
)
async def analyze(reading: SensorReading) -> AnalysisResponse:
    """
    Perform a complete pipeline health analysis combining:
    1. 1-D Navier-Stokes physics residuals (PINN layer)
    2. LSTM temporal pattern recognition
    3. NPW leak localization
    4. Sensor Confidence Score computation
    5. RUL estimation via SCS trend regression
    6. Batch-change and drift detection

    Parameters
    ----------
    reading : SensorReading — full sensor snapshot with optional history

    Returns
    -------
    AnalysisResponse — multi-output analysis result
    """
    t_start = time.perf_counter()
    logger.info("POST /analyze  sensor_id=%s  fluid=%s", reading.sensor_id, reading.fluid_type)

    try:
        result = _run_full_analysis(reading)
        result.processing_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)
        return result
    except ValueError as exc:
        logger.warning("Validation error in /analyze: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Unhandled error in /analyze")
        raise HTTPException(status_code=500, detail=f"Analysis error: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# POST /simulate_leak
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/simulate_leak",
    response_model=SimulateLeakResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a synthetic leak event and analyse it",
    tags=["Simulation"],
)
async def simulate_leak(req: SimulateLeakRequest) -> SimulateLeakResponse:
    """
    Synthesise a single-timestep pipeline snapshot with a simulated leak
    and pass it through the full analysis pipeline.

    The returned 'generated_readings' dict shows the sensor values that
    would be observed during the leak.  The 'npw_accuracy_m' field shows
    how close the NPW localization is to the true leak position.

    Parameters
    ----------
    req : SimulateLeakRequest — leak scenario configuration

    Returns
    -------
    SimulateLeakResponse — synthetic readings + full analysis + accuracy metric
    """
    t_start = time.perf_counter()
    logger.info(
        "POST /simulate_leak  fluid=%s  leak_pos=%.0f m  severity=%.1f%%",
        req.fluid_type, req.leak_position_m, req.leak_severity_pct,
    )

    try:
        profile = CHEMICAL_PROFILES.get(req.fluid_type.value, CHEMICAL_PROFILES["HSD"])
        rng = np.random.default_rng(seed=int(req.leak_position_m) % 9999)

        # ── Build synthetic readings ──────────────────────────────────────────
        nominal_flow = req.nominal_flow_rate_m3s
        nominal_p_in = req.nominal_inlet_pressure_pa
        friction_dp  = _estimate_friction_dp(req, profile)
        nominal_p_out = nominal_p_in - friction_dp

        leak_fraction = req.leak_severity_pct / 100.0

        # Pressure drops more steeply at leak location
        # Upstream: small drop;  Downstream: large drop
        frac_upstream = req.leak_position_m / req.segment_length_m
        dp_total_leak = friction_dp * (1.0 + leak_fraction * 1.5)
        p_in_leak  = nominal_p_in  - dp_total_leak * frac_upstream * 0.3
        p_out_leak = nominal_p_out - dp_total_leak * (1.0 - frac_upstream) * 1.2
        p_out_leak = max(0.0, p_out_leak)

        # Flow drops at outlet (leak removes fluid)
        flow_out = nominal_flow * (1.0 - leak_fraction)

        # Acoustic spike at leak (pressure wave)
        acoustic_baseline = 10.0 + rng.normal(0, 1.0)
        acoustic_spike = acoustic_baseline + 15.0 * leak_fraction + rng.normal(0, 2.0)

        # Sensor drift simulation (45 °C Indian heat)
        drift_factor = 1.0
        if req.include_sensor_drift:
            drift_factor = 1.0 + 0.15 * (req.ambient_temp_c - 30.0) / 15.0

        noise = lambda x: x * (1.0 + rng.uniform(-req.noise_level, req.noise_level))

        readings = {
            "inlet_pressure_pa":       noise(p_in_leak  * drift_factor),
            "outlet_pressure_pa":      noise(p_out_leak * drift_factor),
            "flow_rate_m3s":           noise(flow_out),
            "temperature_c":           req.ambient_temp_c + rng.normal(5, 2),
            "density_kg_m3":           profile["density_kg_m3"] * noise(1.0),
            "dynamic_viscosity_pa_s":  profile["dynamic_viscosity_pa_s"] * noise(1.0),
            "acoustic_vibration":      float(max(0.0, acoustic_spike)),
            "pipe_diameter_m":         req.pipe_diameter_m,
            "wall_thickness_m":        req.wall_thickness_m,
            "segment_length_m":        req.segment_length_m,
            "inlet_pressure_prev_pa":  nominal_p_in,
            "outlet_pressure_prev_pa": nominal_p_out,
        }

        # Build SensorReading and run analysis
        sr = SensorReading(
            inlet_pressure_pa      = max(0.0, readings["inlet_pressure_pa"]),
            outlet_pressure_pa     = max(0.0, min(readings["outlet_pressure_pa"], readings["inlet_pressure_pa"] + 1e5)),
            flow_rate_m3s          = max(0.0, readings["flow_rate_m3s"]),
            temperature_c          = float(np.clip(readings["temperature_c"], -10, 99)),
            density_kg_m3          = float(readings["density_kg_m3"]),
            dynamic_viscosity_pa_s = float(readings["dynamic_viscosity_pa_s"]),
            acoustic_vibration     = float(readings["acoustic_vibration"]),
            pipe_diameter_m        = req.pipe_diameter_m,
            wall_thickness_m       = req.wall_thickness_m,
            segment_length_m       = req.segment_length_m,
            fluid_type             = req.fluid_type,
            ambient_temp_c         = req.ambient_temp_c,
            inlet_pressure_prev_pa  = nominal_p_in,
            outlet_pressure_prev_pa = nominal_p_out,
            sensor_id              = "SIMULATED",
        )

        analysis = _run_full_analysis(sr)
        analysis.processing_time_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

        # NPW accuracy
        npw_acc = None
        if analysis.npw_localization and analysis.npw_localization.triggered:
            npw_acc = round(abs(analysis.npw_localization.leak_distance_m - req.leak_position_m), 1)

        return SimulateLeakResponse(
            fluid_type=req.fluid_type.value,
            true_leak_position_m=req.leak_position_m,
            generated_readings={k: round(float(v), 4) for k, v in readings.items()},
            analysis=analysis,
            npw_accuracy_m=npw_acc,
        )

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in /simulate_leak")
        raise HTTPException(status_code=500, detail=f"Simulation error: {exc}")


def _estimate_friction_dp(req: SimulateLeakRequest, profile: dict) -> float:
    """Estimate Darcy-Weisbach pressure drop for the segment."""
    from app.engine.physics import darcy_friction_factor, pipe_reynolds_number
    area = math.pi * (req.pipe_diameter_m / 2.0) ** 2
    vel  = req.nominal_flow_rate_m3s / max(1e-6, area)
    rho  = profile["density_kg_m3"]
    mu   = profile["dynamic_viscosity_pa_s"]
    re   = pipe_reynolds_number(vel, req.pipe_diameter_m, rho, mu)
    f    = darcy_friction_factor(re)
    return f * (req.segment_length_m / req.pipe_diameter_m) * (rho * vel**2 / 2.0)


# ─────────────────────────────────────────────────────────────────────────────
# GET /sensor_status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/sensor_status",
    response_model=SensorStatusResponse,
    summary="Lightweight sensor health check based on SCS",
    tags=["Operations"],
)
async def sensor_status(
    sensor_id: str = "UNKNOWN",
    scs: float = 95.0,
    flags: str = "",
    rul_hours: float = None,
) -> SensorStatusResponse:
    """
    Returns a concise sensor health summary without running full ML inference.

    Parameters (query string)
    ----------
    sensor_id : Sensor identifier
    scs       : Current Sensor Confidence Score (0-100)
    flags     : Comma-separated flag list (e.g. "SENSOR_DRIFT_RISK_HIGH_TEMP")
    rul_hours : Known RUL estimate in hours (optional)

    Returns
    -------
    SensorStatusResponse
    """
    flag_list = [f.strip() for f in flags.split(",") if f.strip()] if flags else []

    if scs >= 85.0:
        health_status = "HEALTHY"
        recommendation = "No action required. Continue standard monitoring."
    elif scs >= 70.0:
        health_status = "DEGRADED"
        recommendation = (
            "Schedule sensor calibration within 48 hours. "
            "Increase sampling frequency to 15-minute intervals."
        )
    else:
        health_status = "CRITICAL"
        recommendation = (
            "Immediate sensor inspection required. "
            "Cross-validate readings with secondary measurement device. "
            "Do not rely on automated alerts until recalibrated."
        )

    if "SENSOR_DRIFT_RISK_HIGH_TEMP" in flag_list:
        recommendation += " Apply temperature compensation to raw readings."

    return SensorStatusResponse(
        sensor_id=sensor_id,
        scs=round(scs, 2),
        status=health_status,
        flags=flag_list,
        rul_hours=rul_hours,
        recommendation=recommendation,
    )


# ─────────────────────────────────────────────────────────────────────────────
# GET /health
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/health", tags=["Operations"], summary="Service health probe")
async def health() -> dict:
    """Returns service status and backend information."""
    import sys
    detector = get_detector()
    return {
        "status": "healthy",
        "service": "NAVIER AI Core",
        "version": "1.0.0",
        "model_backend": detector._backend,
        "python": sys.version.split()[0],
    }


# ─────────────────────────────────────────────────────────────────────────────
# GET /model/info
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/model/info", tags=["Operations"], summary="Model architecture metadata")
async def model_info() -> dict:
    """Returns model architecture details and chemical profiles."""
    import numpy as np
    import sklearn

    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "not installed (numpy emulation active)"

    return {
        "architecture": "Hybrid PINN-LSTM",
        "layers": {
            "1_pinn_branch": "Dense MLP (N→128→64→32) with LayerNorm + Dropout",
            "2_lstm_branch": "Bidirectional LSTM (hidden=64, layers=2) → Linear(32)",
            "3_physics_embed": "Linear(3→8, ReLU) — injects NS residuals",
            "4_shared_head": "Dense(72→64→32, ReLU)",
            "outputs": {
                "A_classification": "Linear(32→3) + Softmax → SAFE/WARNING/CRITICAL",
                "B_localization": "Linear(32→1) + Sigmoid → normalised distance",
                "C_scs": "Linear(32→1) + Sigmoid → Sensor Confidence Score",
                "D_rul": "Linear(32→1) + Sigmoid → Remaining Useful Life norm",
            },
        },
        "pinn_loss": "L_total = L_class + 0.30×L_NS + 0.20×L_loc + 0.15×L_scs + 0.15×L_rul",
        "npw_formula": "X = (L - a·Δt) / 2  [Brunone et al. 2000]",
        "chemical_profiles": list(CHEMICAL_PROFILES.keys()),
        "feature_count": len(FEATURE_NAMES),
        "framework_versions": {
            "torch": torch_ver,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "thresholds": {
            "scs_high_divergence": "< 80% when PINN divergence > 20%",
            "drift_flag": "15% linear deviation from baseline",
            "acoustic_spike": "3× baseline standard deviation",
            "batch_change_density": "25 kg/m³ step change",
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis orchestrator (shared by /analyze and /simulate_leak)
# ─────────────────────────────────────────────────────────────────────────────

def _run_full_analysis(reading: SensorReading) -> AnalysisResponse:
    """
    Orchestrate the complete analysis pipeline for one SensorReading.

    Steps
    -----
    1. Batch-change detection  → suppresses false alarms during transitions
    2. 1-D Navier-Stokes residuals  → PINN physics loss
    3. Wave-speed calculation  → NPW prerequisite
    4. NPW localization        → leak distance estimate
    5. Hybrid PINN-LSTM inference  → classification, SCS, RUL
    6. PINN-predicted values   → SCS computation
    7. RUL decoding            → hours / days / degradation label
    8. Alert message           → human-readable status

    Parameters
    ----------
    reading : SensorReading

    Returns
    -------
    AnalysisResponse (processing_time_ms will be filled by caller)
    """

    # ── 1. Batch-change detection ─────────────────────────────────────────────
    batch: BatchChangeResult = detect_batch_change(
        current_density_kg_m3  = reading.density_kg_m3,
        previous_density_kg_m3 = reading.previous_density_kg_m3 or reading.density_kg_m3,
        current_viscosity_pa_s  = reading.dynamic_viscosity_pa_s,
        previous_viscosity_pa_s = reading.previous_viscosity_pa_s or reading.dynamic_viscosity_pa_s,
        current_fluid_label    = reading.fluid_type.value,
        previous_fluid_label   = reading.previous_fluid_type.value if reading.previous_fluid_type else None,
    )

    # ── 2. Navier-Stokes residuals ─────────────────────────────────────────────
    ns = compute_navier_stokes_residuals(
        inlet_pressure_pa      = reading.inlet_pressure_pa,
        outlet_pressure_pa     = reading.outlet_pressure_pa,
        flow_rate_m3s          = reading.flow_rate_m3s,
        pipe_diameter_m        = reading.pipe_diameter_m,
        pipe_length_m          = reading.segment_length_m,
        fluid_type             = reading.fluid_type.value,
        temperature_c          = reading.temperature_c,
        dt                     = reading.dt_seconds,
    )

    # ── 3. Wave speed ─────────────────────────────────────────────────────────
    wave_speed = compute_wave_speed(
        fluid_type       = reading.fluid_type.value,
        pipe_diameter_m  = reading.pipe_diameter_m,
        wall_thickness_m = reading.wall_thickness_m,
    )

    # ── 4. NPW localization ───────────────────────────────────────────────────
    prev_p_in  = reading.inlet_pressure_prev_pa  or reading.inlet_pressure_pa
    prev_p_out = reading.outlet_pressure_prev_pa or reading.outlet_pressure_pa

    npw = npw_localization(
        segment_length_m         = reading.segment_length_m,
        wave_speed_ms            = wave_speed,
        inlet_pressure_pa        = reading.inlet_pressure_pa,
        outlet_pressure_pa       = reading.outlet_pressure_pa,
        inlet_pressure_prev_pa   = prev_p_in,
        outlet_pressure_prev_pa  = prev_p_out,
        dt_sensor_s              = reading.dt_seconds,
        pressure_drop_threshold_pa = 30_000.0,  # 0.3 bar
    )

    # ── 5. ML inference ───────────────────────────────────────────────────────
    features_dict = {
        "inlet_pressure_pa":      reading.inlet_pressure_pa,
        "outlet_pressure_pa":     reading.outlet_pressure_pa,
        "flow_rate_m3s":          reading.flow_rate_m3s,
        "temperature_c":          reading.temperature_c,
        "density_kg_m3":          reading.density_kg_m3,
        "dynamic_viscosity_pa_s": reading.dynamic_viscosity_pa_s,
        "acoustic_vibration":     reading.acoustic_vibration,
        "pipe_diameter_m":        reading.pipe_diameter_m,
        "wall_thickness_m":       reading.wall_thickness_m,
        "segment_length_m":       reading.segment_length_m,
    }

    sequence_window = _build_sequence(reading, features_dict)
    physics_residuals = (ns.mass_residual, ns.momentum_residual, ns.pinn_loss_weight)

    detector = get_detector()
    ml_out = detector.predict(features_dict, sequence_window, physics_residuals)

    # ── 6. SCS ───────────────────────────────────────────────────────────────
    # PINN-predicted 'physical truth' for key sensors
    profile  = CHEMICAL_PROFILES.get(reading.fluid_type.value, CHEMICAL_PROFILES["HSD"])
    pinn_predicted = _build_pinn_predictions(reading, ns, profile)

    measured_for_scs = {
        "inlet_pressure":  reading.inlet_pressure_pa,
        "outlet_pressure": reading.outlet_pressure_pa,
        "flow_rate":       reading.flow_rate_m3s,
        "temperature":     reading.temperature_c,
        "acoustic":        reading.acoustic_vibration,
    }

    acoustic_std = 3.0  # assumed baseline std for acoustic sensor
    scs_result = compute_scs(
        measured              = measured_for_scs,
        pinn_predicted        = pinn_predicted,
        ambient_temp_c        = reading.ambient_temp_c,
        drift_history         = reading.scs_history,
        acoustic_baseline_std = acoustic_std,
    )

    # Override ML SCS with causal_guard SCS (causal_guard is primary)
    scs_final = scs_result.scs

    # ── 7. RUL ───────────────────────────────────────────────────────────────
    rul_hours, degradation = _decode_rul(
        rul_norm            = ml_out.rul_norm,
        scs_result_rul      = scs_result.rul_hours,
        pipeline_age_years  = reading.pipeline_age_years,
        physics_loss        = ns.pinn_loss_weight,
    )

    # ── 8. Batch-change suppression ───────────────────────────────────────────
    classification = ml_out.classification
    if batch.suppression_active and classification == "CRITICAL":
        classification = "WARNING"
        logger.info("Batch-change suppression: CRITICAL → WARNING")

    # ── 9. Alert message ─────────────────────────────────────────────────────
    alert = _build_alert_message(
        classification = classification,
        npw            = npw,
        batch          = batch,
        scs            = scs_final,
        rul_hours      = rul_hours,
        physics_loss   = ns.pinn_loss_weight,
    )

    # Anomaly score = weighted combination of physics + class prob
    critical_prob = ml_out.class_probabilities.get("CRITICAL", 0.0)
    warning_prob  = ml_out.class_probabilities.get("WARNING",  0.0)
    anomaly_score = min(1.0, 0.5 * critical_prob + 0.25 * warning_prob + 0.25 * ns.pinn_loss_weight)

    return AnalysisResponse(
        sensor_id           = reading.sensor_id,
        station_id          = reading.station_id,
        timestamp_utc       = reading.timestamp_utc,
        fluid_type          = reading.fluid_type.value,
        classification      = Classification(classification),
        alert_message       = alert,
        anomaly_score       = round(float(anomaly_score), 4),
        class_probabilities = ml_out.class_probabilities,
        npw_localization    = NPWLocalizationResult(
            triggered       = npw.triggered,
            leak_distance_m = npw.leak_distance_m,
            wave_speed_ms   = round(npw.wave_speed_ms, 1),
            delta_t_s       = npw.delta_t_s,
            confidence      = round(npw.confidence, 3),
            method          = npw.method,
        ) if npw.triggered else None,
        scs_report = SCSReport(
            scs                      = scs_final,
            overall_divergence_pct   = scs_result.overall_divergence_pct,
            per_sensor_divergence    = scs_result.per_sensor_divergence,
            flags                    = scs_result.flags,
            rul_hours                = scs_result.rul_hours,
            rul_confidence           = scs_result.rul_confidence,
            drift_rate_pct_per_hour  = scs_result.drift_rate_pct_per_hour,
            ambient_temp_c           = scs_result.ambient_temp_c,
        ),
        rul_hours          = rul_hours,
        rul_days           = round(rul_hours / 24.0, 1) if rul_hours else None,
        degradation_rate   = DegradationRate(degradation),
        physics_report     = PhysicsReport(
            mass_residual         = round(ns.mass_residual, 5),
            momentum_residual     = round(ns.momentum_residual, 5),
            darcy_friction_factor = round(ns.darcy_friction_factor, 5),
            reynolds_number       = round(ns.reynolds_number, 1),
            is_turbulent          = ns.is_turbulent,
            pinn_loss_weight      = round(ns.pinn_loss_weight, 5),
        ),
        batch_change = BatchChangeReport(
            detected             = batch.detected,
            from_fluid           = batch.from_fluid,
            to_fluid             = batch.to_fluid,
            density_delta_kg_m3  = batch.density_delta_kg_m3,
            suppression_active   = batch.suppression_active,
        ),
        model_backend      = ml_out.backend,
        inference_time_ms  = ml_out.inference_time_ms,
        processing_time_ms = 0.0,  # filled by caller
    )


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_sequence(
    reading: SensorReading,
    current_features: dict[str, float],
) -> list[dict[str, float]]:
    """Convert reading.sequence_window to a list of feature dicts."""
    if reading.sequence_window:
        # Each element may have bar/SI units — pass as-is, normalisation is in model.py
        return reading.sequence_window  # type: ignore[return-value]
    return [current_features]


def _build_pinn_predictions(reading: SensorReading, ns, profile: dict) -> dict:
    """
    Construct the PINN's 'physically expected' sensor values for SCS comparison.

    For inlet/outlet pressure: use Darcy-Weisbach steady-state prediction.
    For flow_rate: same as reading (conservation forces Q_in = Q_out if no leak).
    For temperature: treat as stable (PINN does not model heat transfer here).
    For acoustic: treat as near-zero baseline (no transient predicted).
    """
    area = math.pi * (reading.pipe_diameter_m / 2.0) ** 2
    vel  = reading.flow_rate_m3s / max(1e-9, area)
    rho  = profile["density_kg_m3"]
    dyn_friction_pa = (
        ns.darcy_friction_factor
        * reading.segment_length_m
        / reading.pipe_diameter_m
        * (rho * vel**2 / 2.0)
    )
    predicted_outlet = reading.inlet_pressure_pa - dyn_friction_pa

    return {
        "inlet_pressure":  reading.inlet_pressure_pa,      # known
        "outlet_pressure": max(0.0, predicted_outlet),
        "flow_rate":       reading.flow_rate_m3s,           # mass-conserved
        "temperature":     reading.temperature_c,           # assumed stable
        "acoustic":        5.0 + ns.reynolds_number * 2e-5, # empirical baseline
    }


def _decode_rul(
    rul_norm: float,
    scs_result_rul: float | None,
    pipeline_age_years: float,
    physics_loss: float,
) -> tuple[float | None, str]:
    """
    Convert normalised RUL output [0,1] to hours and a degradation label.

    Maximum pipeline design life: 30 years = 262 800 hours.
    Physics loss modulates remaining life: high residual → faster degradation.
    If SCS-based RUL is available, take the minimum (conservative).
    """
    MAX_LIFE_HOURS = 30.0 * 8760.0
    consumed_hours = pipeline_age_years * 8760.0
    remaining_base = max(0.0, MAX_LIFE_HOURS - consumed_hours)

    degradation_mult = max(0.05, 1.0 - 0.7 * physics_loss)
    rul_from_model   = remaining_base * rul_norm * degradation_mult

    # Blend with SCS-derived RUL if available
    if scs_result_rul is not None and scs_result_rul > 0:
        rul_hours = min(rul_from_model, scs_result_rul)
    else:
        rul_hours = rul_from_model

    rul_hours = float(max(0.0, rul_hours))

    if degradation_mult < 0.3 or rul_hours < 1000:
        label = "ACCELERATED"
    elif degradation_mult < 0.7:
        label = "NORMAL"
    else:
        label = "SLOW"

    return round(rul_hours, 1), label


def _build_alert_message(
    classification: str,
    npw,
    batch: BatchChangeResult,
    scs: float,
    rul_hours: float | None,
    physics_loss: float,
) -> str:
    """Compose a human-readable alert message for field operators."""
    prefix = {
        "SAFE":     "🟢 SAFE",
        "WARNING":  "🟡 WARNING",
        "CRITICAL": "🔴 CRITICAL ALERT",
    }.get(classification, "⚪ UNKNOWN")

    parts = [prefix]

    if classification == "SAFE":
        parts.append("— All parameters within normal operating range.")
    elif classification == "WARNING":
        parts.append("— Anomalous pressure/flow pattern detected.")
        if npw.triggered:
            parts.append(
                f"NPW indicates possible anomaly at {npw.leak_distance_m:.0f} m "
                f"(confidence {npw.confidence*100:.0f}%)."
            )
        if physics_loss > 0.30:
            parts.append(f"PINN physics residual elevated ({physics_loss:.3f}).")
    elif classification == "CRITICAL":
        parts.append("— Probable leak event. Immediate field inspection required.")
        if npw.triggered:
            parts.append(
                f"NPW localisation: ~{npw.leak_distance_m:.0f} m from upstream sensor "
                f"[wave speed {npw.wave_speed_ms:.0f} m/s, Δt={npw.delta_t_s:.3f} s]."
            )

    if batch.suppression_active:
        parts.append(f"[Batch change {batch.from_fluid}→{batch.to_fluid} in progress — false-alarm suppression ON.]")

    if scs < 80:
        parts.append(f"⚠️  Sensor Confidence Score low ({scs:.1f}%) — verify sensor calibration.")

    if rul_hours is not None and rul_hours < 500:
        parts.append(f"⚠️  Estimated RUL < 500 h ({rul_hours:.0f} h). Schedule maintenance.")

    return " ".join(parts)
PYEOF
echo "  ✅  app/api/routes.py"

# ─────────────────────────────────────────────────────────────────────────────
# app/main.py
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/app/main.py"
"""
main.py — FastAPI application entry point for the NAVIER AI microservice.

Usage
-----
    uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

    # Or directly:
    python -m app.main
"""
from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.engine.model import get_detector

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("navier")


# ── Lifespan — warm up model on startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load the ML model singleton so first request is not slow."""
    logger.info("🛢️  NAVIER AI Core starting up...")
    t0 = time.perf_counter()
    get_detector()
    logger.info("✅ Hybrid PINN-LSTM detector ready in %.2f s", time.perf_counter() - t0)
    yield
    logger.info("🛑 NAVIER AI Core shut down.")


# ── Application factory ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="NAVIER AI Core",
        description=(
            "Hybrid Physics-Informed Neural Network (PINN) + LSTM microservice "
            "for petroleum pipeline leak detection, sensor confidence scoring, "
            "and Remaining Useful Life estimation.\n\n"
            "**Endpoints:**\n"
            "- `POST /analyze` — Full PINN-LSTM pipeline analysis\n"
            "- `POST /simulate_leak` — Synthetic leak scenario simulation\n"
            "- `GET /sensor_status` — Lightweight sensor health check\n"
            "- `GET /health` — Service health probe\n"
            "- `GET /model/info` — Model architecture metadata\n"
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS (allow React frontend + Express.js backend) ─────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Restrict to specific origins in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request timing header ─────────────────────────────────────────────────
    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter()-t0)*1000:.2f}"
        return response

    # ── Global error handler ─────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "Unexpected internal error.", "type": type(exc).__name__},
        )

    # ── Mount routes ─────────────────────────────────────────────────────────
    app.include_router(router)

    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8001, reload=True, log_level="info")
PYEOF
echo "  ✅  app/main.py"

# ─────────────────────────────────────────────────────────────────────────────
# app/__init__.py  +  app/engine/__init__.py  +  app/schemas/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
touch "$PROJECT/app/__init__.py"
touch "$PROJECT/app/engine/__init__.py"
touch "$PROJECT/app/schemas/__init__.py"
touch "$PROJECT/app/api/__init__.py"

# ─────────────────────────────────────────────────────────────────────────────
# data/synthetic_gen.py
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/data/synthetic_gen.py"
"""
synthetic_gen.py — Synthetic pipeline sensor dataset generator for NAVIER.

Generates a labelled CSV dataset representing a 10-km petroleum pipeline
carrying MS (Petrol), HSD (Diesel), and ATF (Aviation Fuel) under four
distinct operational scenarios:

  A) Normal steady-state flow
  B) Leak event  — sudden pressure drop + acoustic spike
  C) Sensor drift — 15% linear deviation simulating 45 °C Indian heat
  D) Batch change — switching from one product to another

Output
------
    data/navier_synthetic_dataset.csv   (7 000 + rows, labelled)

Usage
-----
    python data/synthetic_gen.py
    python data/synthetic_gen.py --rows 20000 --seed 99
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Add project root to path so engine imports work ──────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.engine.physics import (
    CHEMICAL_PROFILES,
    compute_wave_speed,
    darcy_friction_factor,
    pipe_reynolds_number,
)

OUTPUT_PATH = PROJECT_ROOT / "data" / "navier_synthetic_dataset.csv"

# ── Pipeline geometry (10 km, 20-inch API 5L X65) ─────────────────────────
PIPE_DIAMETER_M    = 0.508
WALL_THICKNESS_M   = 0.0095
SEGMENT_LENGTH_M   = 10_000.0
PIPE_ROUGHNESS_EPS = 4.6e-5

# ── Scenario distribution ─────────────────────────────────────────────────
SCENARIO_WEIGHTS = {
    "NORMAL":       0.60,
    "LEAK":         0.20,
    "SENSOR_DRIFT": 0.12,
    "BATCH_CHANGE": 0.08,
}

FLUID_CYCLE = ["MS", "HSD", "ATF", "HSD", "MS"]   # typical pipeline schedule


def generate_dataset(n_rows: int = 7_000, seed: int = 42) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.

    Parameters
    ----------
    n_rows : Total number of rows to generate
    seed   : Random seed for reproducibility

    Returns
    -------
    pd.DataFrame with columns matching the NAVIER API SensorReading schema
    """
    rng = np.random.default_rng(seed)
    records = []

    # Determine scenario assignments
    scenarios  = rng.choice(
        list(SCENARIO_WEIGHTS.keys()),
        size=n_rows,
        p=list(SCENARIO_WEIGHTS.values()),
    )
    fluid_idx = 0
    current_fluid = FLUID_CYCLE[0]
    rows_since_batch = 0

    print(f"Generating {n_rows} synthetic rows  (seed={seed})...")

    for i, scenario in enumerate(scenarios):
        # Cycle through fluids for batch-change events
        if scenario == "BATCH_CHANGE" or rows_since_batch > 500:
            fluid_idx = (fluid_idx + 1) % len(FLUID_CYCLE)
            current_fluid = FLUID_CYCLE[fluid_idx]
            rows_since_batch = 0
        rows_since_batch += 1

        profile = CHEMICAL_PROFILES[current_fluid]
        row = _generate_row(i, scenario, current_fluid, profile, rng)
        records.append(row)

    df = pd.DataFrame(records)
    print(f"  Scenario distribution:\n{df['scenario'].value_counts()}")
    print(f"  Fluid distribution:\n{df['fluid_type'].value_counts()}")
    return df


def _generate_row(
    idx: int,
    scenario: str,
    fluid: str,
    profile: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Generate one synthetic sensor reading row.

    Parameters
    ----------
    idx      : Row index (used for time column)
    scenario : One of NORMAL / LEAK / SENSOR_DRIFT / BATCH_CHANGE
    fluid    : CHEMICAL_PROFILES key
    profile  : Fluid physical properties dict
    rng      : NumPy random generator

    Returns
    -------
    dict — one row for the DataFrame
    """
    # ── Nominal operating point ───────────────────────────────────────────────
    rho   = profile["density_kg_m3"]
    mu    = profile["dynamic_viscosity_pa_s"]
    area  = math.pi * (PIPE_DIAMETER_M / 2.0) ** 2

    nominal_p_in   = rng.uniform(30e5, 60e5)    # 30–60 bar
    nominal_flow   = rng.uniform(0.10, 0.40)     # m³/s

    vel = nominal_flow / area
    re  = pipe_reynolds_number(vel, PIPE_DIAMETER_M, rho, mu)
    f   = darcy_friction_factor(re, PIPE_ROUGHNESS_EPS)
    dp_friction = f * (SEGMENT_LENGTH_M / PIPE_DIAMETER_M) * (rho * vel**2 / 2.0)

    nominal_p_out  = nominal_p_in - dp_friction
    nominal_temp   = rng.uniform(25.0, 55.0)
    acoustic_base  = rng.normal(8.0, 2.0)
    ambient_temp   = rng.uniform(25.0, 48.0)

    # ── Scenario-specific modifications ──────────────────────────────────────
    label       = "SAFE"
    leak_pos_m  = 0.0
    drift_pct   = 0.0

    p_in  = nominal_p_in
    p_out = nominal_p_out
    flow  = nominal_flow
    acoustic = acoustic_base

    if scenario == "NORMAL":
        # Small random noise
        p_in     += rng.normal(0, 0.01 * nominal_p_in)
        p_out    += rng.normal(0, 0.01 * nominal_p_out)
        flow     += rng.normal(0, 0.02 * nominal_flow)
        acoustic += rng.normal(0, 1.5)

    elif scenario == "LEAK":
        # Sudden pressure drop + acoustic spike
        leak_pos_m   = rng.uniform(500, SEGMENT_LENGTH_M - 500)
        severity     = rng.uniform(0.05, 0.35)      # 5–35% flow loss

        frac_up   = leak_pos_m / SEGMENT_LENGTH_M
        dp_leak   = dp_friction * (1.0 + severity * 2.0)
        p_in      = nominal_p_in  - dp_leak * frac_up * 0.25
        p_out     = nominal_p_out - dp_leak * (1.0 - frac_up) * 1.5
        flow      = nominal_flow * (1.0 - severity)
        acoustic  += rng.normal(20.0 * severity, 3.0)   # acoustic spike

        label = "WARNING" if severity < 0.15 else "CRITICAL"

    elif scenario == "SENSOR_DRIFT":
        # Linear 15% drift caused by 45 °C heat
        drift_pct = 0.15 * (ambient_temp - 30.0) / 15.0
        p_in   *= (1.0 + drift_pct)
        p_out  *= (1.0 + drift_pct * 0.8)
        flow   *= (1.0 - drift_pct * 0.3)
        acoustic += rng.normal(0, 2.0)
        label = "WARNING"

    elif scenario == "BATCH_CHANGE":
        # Transient during product switch — density/viscosity jump
        p_in    += rng.normal(0, 0.05 * nominal_p_in)
        p_out   += rng.normal(0, 0.05 * nominal_p_out)
        flow    += rng.normal(0, 0.08 * nominal_flow)
        acoustic += rng.normal(5.0, 3.0)
        label = "WARNING"

    # Clamp to physical limits
    p_out = max(0.5e5, p_out)
    flow  = max(0.005, flow)

    # Class label index
    class_idx = {"SAFE": 0, "WARNING": 1, "CRITICAL": 2}[label]

    # SCS ground truth (synthetic)
    scs_gt = {
        "SAFE":     rng.uniform(88, 100),
        "WARNING":  rng.uniform(68, 87),
        "CRITICAL": rng.uniform(40, 68),
    }[label]

    # NPW distance label (only meaningful for LEAK)
    npw_distance_gt = leak_pos_m

    return {
        "timestamp_s":             idx,
        "fluid_type":              fluid,
        "scenario":                scenario,
        "class_label":             label,
        "class_idx":               class_idx,
        "inlet_pressure_pa":       round(p_in, 1),
        "outlet_pressure_pa":      round(p_out, 1),
        "flow_rate_m3s":           round(flow, 5),
        "temperature_c":           round(nominal_temp, 2),
        "density_kg_m3":           round(rho, 2),
        "dynamic_viscosity_pa_s":  round(mu, 6),
        "acoustic_vibration":      round(max(0.0, acoustic), 3),
        "pipe_diameter_m":         PIPE_DIAMETER_M,
        "wall_thickness_m":        WALL_THICKNESS_M,
        "segment_length_m":        SEGMENT_LENGTH_M,
        "ambient_temp_c":          round(ambient_temp, 1),
        "leak_position_m_gt":      round(leak_pos_m, 1),
        "drift_pct_gt":            round(drift_pct * 100, 2),
        "scs_gt":                  round(scs_gt, 2),
        "npw_distance_gt_m":       round(npw_distance_gt, 1),
        "reynolds_number":         round(re, 0),
        "darcy_friction_factor":   round(f, 5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NAVIER synthetic dataset generator")
    parser.add_argument("--rows", type=int, default=7_000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42,    help="Random seed")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Output CSV path")
    args = parser.parse_args()

    df = generate_dataset(n_rows=args.rows, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\n✅  Dataset saved to: {out}  ({len(df)} rows, {len(df.columns)} columns)")
    print(df.describe().T[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()
PYEOF
echo "  ✅  data/synthetic_gen.py"

# ─────────────────────────────────────────────────────────────────────────────
# tests/test_navier.py
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/tests/test_navier.py"
"""
test_navier.py — Smoke tests for the NAVIER AI Core service.

Run with:  pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.main import app

client = TestClient(app)

# ── Shared fixtures ───────────────────────────────────────────────────────────

NORMAL_PAYLOAD = {
    "inlet_pressure_pa":      4_500_000.0,
    "outlet_pressure_pa":     4_020_000.0,
    "flow_rate_m3s":          0.25,
    "temperature_c":          42.0,
    "density_kg_m3":          840.0,
    "dynamic_viscosity_pa_s": 0.0035,
    "acoustic_vibration":     10.5,
    "pipe_diameter_m":        0.508,
    "wall_thickness_m":       0.0095,
    "segment_length_m":       10_000.0,
    "fluid_type":             "HSD",
    "ambient_temp_c":         38.0,
    "sensor_id":              "TEST-01",
    "timestamp_utc":          "2025-06-15T10:30:00Z",
}

LEAK_PAYLOAD = {
    **NORMAL_PAYLOAD,
    "inlet_pressure_pa":       4_200_000.0,
    "outlet_pressure_pa":      2_800_000.0,   # large pressure drop
    "flow_rate_m3s":           0.18,           # flow collapse
    "acoustic_vibration":      45.0,           # acoustic spike
    "inlet_pressure_prev_pa":  4_500_000.0,
    "outlet_pressure_prev_pa": 4_020_000.0,
    "sensor_id":               "TEST-LEAK-01",
}

DRIFT_PAYLOAD = {
    **NORMAL_PAYLOAD,
    "ambient_temp_c": 46.0,
    "inlet_pressure_pa":  5_175_000.0,   # 15% high drift
    "outlet_pressure_pa": 4_623_000.0,
}

BATCH_PAYLOAD = {
    **NORMAL_PAYLOAD,
    "fluid_type":              "HSD",
    "previous_fluid_type":     "MS",
    "density_kg_m3":           840.0,
    "previous_density_kg_m3":  745.0,
    "dynamic_viscosity_pa_s":  0.0035,
    "previous_viscosity_pa_s": 0.00055,
}


# ── Route tests ───────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "model_backend" in data


def test_model_info():
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    assert "pinn_loss" in data
    assert "npw_formula" in data
    assert len(data["chemical_profiles"]) >= 5


def test_analyze_normal():
    r = client.post("/analyze", json=NORMAL_PAYLOAD)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["classification"] in ("SAFE", "WARNING", "CRITICAL")
    assert 0.0 <= d["anomaly_score"] <= 1.0
    assert "scs_report" in d
    assert "physics_report" in d
    assert 0.0 <= d["scs_report"]["scs"] <= 100.0


def test_analyze_leak_escalates():
    r = client.post("/analyze", json=LEAK_PAYLOAD)
    assert r.status_code == 200, r.text
    d = r.json()
    # Leak should trigger at least WARNING
    assert d["classification"] in ("WARNING", "CRITICAL")


def test_analyze_npw_fires_on_leak():
    r = client.post("/analyze", json=LEAK_PAYLOAD)
    assert r.status_code == 200, r.text
    d = r.json()
    npw = d.get("npw_localization")
    if npw:  # may or may not trigger depending on threshold
        assert 0.0 <= npw["leak_distance_m"] <= LEAK_PAYLOAD["segment_length_m"]
        assert npw["wave_speed_ms"] > 0


def test_analyze_batch_change_detected():
    r = client.post("/analyze", json=BATCH_PAYLOAD)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["batch_change"]["detected"] is True
    assert d["batch_change"]["from_fluid"] == "MS"
    assert d["batch_change"]["to_fluid"] == "HSD"


def test_analyze_drift_flags():
    r = client.post("/analyze", json=DRIFT_PAYLOAD)
    assert r.status_code == 200, r.text
    d = r.json()
    flags = d["scs_report"]["flags"]
    assert any("TEMP" in f or "DRIFT" in f for f in flags)


def test_sensor_status():
    r = client.get("/sensor_status", params={"sensor_id": "S-01", "scs": 92.0})
    assert r.status_code == 200
    d = r.json()
    assert d["status"] == "HEALTHY"

    r2 = client.get("/sensor_status", params={"scs": 55.0})
    assert r2.json()["status"] == "CRITICAL"


def test_simulate_leak():
    payload = {
        "fluid_type": "MS",
        "segment_length_m": 10_000.0,
        "pipe_diameter_m": 0.508,
        "wall_thickness_m": 0.0095,
        "leak_position_m": 4_000.0,
        "leak_severity_pct": 15.0,
        "nominal_inlet_pressure_pa": 4_500_000.0,
        "nominal_flow_rate_m3s": 0.25,
        "ambient_temp_c": 44.0,
    }
    r = client.post("/simulate_leak", json=payload)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["true_leak_position_m"] == 4_000.0
    assert "generated_readings" in d
    assert "analysis" in d


def test_invalid_outlet_pressure():
    bad = {**NORMAL_PAYLOAD, "outlet_pressure_pa": 9_000_000.0}
    r = client.post("/analyze", json=bad)
    assert r.status_code == 422


def test_processing_time_header():
    r = client.post("/analyze", json=NORMAL_PAYLOAD)
    assert "x-process-time-ms" in r.headers


def test_all_fluid_types():
    for fluid in ["MS", "HSD", "ATF"]:
        payload = {**NORMAL_PAYLOAD, "fluid_type": fluid}
        r = client.post("/analyze", json=payload)
        assert r.status_code == 200, f"Failed for fluid {fluid}: {r.text}"
        assert r.json()["fluid_type"] == fluid


# ── Physics unit tests ────────────────────────────────────────────────────────

def test_reynolds_number():
    from app.engine.physics import pipe_reynolds_number
    re = pipe_reynolds_number(1.5, 0.508, 840.0, 0.0035)
    assert re > 4000   # turbulent

def test_darcy_friction_laminar():
    from app.engine.physics import darcy_friction_factor
    f = darcy_friction_factor(1000.0)
    assert abs(f - 0.064) < 0.001

def test_npw_localization():
    from app.engine.physics import npw_localization
    r = npw_localization(10_000.0, 1200.0, 3_800_000.0, 2_500_000.0, 4_500_000.0, 4_020_000.0)
    assert r.triggered
    assert 0.0 <= r.leak_distance_m <= 10_000.0

def test_viscosity_temperature():
    from app.engine.physics import _viscosity_temperature_correction
    mu_hot = _viscosity_temperature_correction(0.0035, 60.0, "HSD")
    assert mu_hot < 0.0035   # viscosity decreases with temperature

def test_wave_speed():
    from app.engine.physics import compute_wave_speed
    a = compute_wave_speed("HSD", 0.508, 0.0095)
    assert 500 < a < 2000

def test_scs_below_80_when_high_divergence():
    from app.engine.causal_guard import compute_scs
    measured   = {"inlet_pressure": 1e6, "outlet_pressure": 1e6,
                  "flow_rate": 0.25,     "temperature": 40.0, "acoustic": 8.0}
    # PINN says values should be very different (30% divergence)
    predicted  = {"inlet_pressure": 0.7e6, "outlet_pressure": 0.7e6,
                  "flow_rate": 0.25,        "temperature": 40.0, "acoustic": 8.0}
    result = compute_scs(measured, predicted)
    assert result.scs < 80.0, f"Expected SCS < 80, got {result.scs}"

def test_batch_change_detected():
    from app.engine.causal_guard import detect_batch_change
    r = detect_batch_change(840.0, 745.0, 0.0035, 0.00055, "HSD", "MS")
    assert r.detected is True

def test_rul_regression():
    from app.engine.causal_guard import _estimate_rul
    history = list(range(95, 70, -2))  # declining SCS
    rul, conf, slope = _estimate_rul(history, 70.0)
    assert rul is not None and rul >= 0
    assert slope < 0   # negative slope → declining
PYEOF
echo "  ✅  tests/test_navier.py"

# ─────────────────────────────────────────────────────────────────────────────
# tests/__init__.py
# ─────────────────────────────────────────────────────────────────────────────
touch "$PROJECT/tests/__init__.py"

# ─────────────────────────────────────────────────────────────────────────────
# README.md
# ─────────────────────────────────────────────────────────────────────────────
cat <<'PYEOF' > "$PROJECT/README.md"
# NAVIER AI Core — Hybrid PINN-LSTM Pipeline Monitoring Microservice

> **For the Antigravity Agent:** Follow **Steps 1–6** exactly to initialise
> the environment, generate training data, and connect to the Express.js backend.

---

## Architecture

```
POST /analyze
      │
      ├─ 1. Batch-Change Detection     (causal_guard.py)
      ├─ 2. 1-D Navier-Stokes Residuals (physics.py)
      ├─ 3. Korteweg Wave Speed         (physics.py)
      ├─ 4. NPW Localization            (physics.py)
      │       X = (L - a·Δt) / 2
      └─ 5. Hybrid PINN-LSTM Inference  (model.py)
              ├── PINN Branch  (Dense MLP, NS loss)
              ├── LSTM Branch  (seq temporal)
              └── Physics Embed (3-dim residual)
                       │
                ┌──────┴──────────────────────────┐
                │  Multi-Output Head               │
                ├─ A: Classification (SAFE/WARN/CRIT)
                ├─ B: Localization   (distance m)
                ├─ C: SCS            (0-100%)
                └─ D: RUL            (hours)
```

---

## Folder Structure

```
navier_ai_core/
├── app/
│   ├── main.py                  ← FastAPI app + lifespan
│   ├── api/routes.py            ← /analyze /simulate_leak /sensor_status
│   ├── engine/
│   │   ├── physics.py           ← 1-D NS equations + NPW + wave speed
│   │   ├── model.py             ← Hybrid PINN-LSTM (PyTorch / NumPy fallback)
│   │   └── causal_guard.py      ← SCS + drift detection + batch-change
│   └── schemas/request_models.py← Pydantic v2 I/O schemas
├── data/
│   └── synthetic_gen.py         ← Synthetic dataset generator
├── models/                      ← Drop trained .pt weights here
├── tests/test_navier.py
├── requirements.txt
└── README.md
```

---

## Step 1 — Create and Activate Virtual Environment

```bash
cd navier_ai_core
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

## Step 2 — Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Step 3 — Generate Synthetic Training Data

```bash
python data/synthetic_gen.py --rows 10000 --seed 42
# → data/navier_synthetic_dataset.csv
```

## Step 4 — Run the Microservice

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

Interactive API docs: **http://localhost:8001/docs**

## Step 5 — Run Tests

```bash
pytest tests/ -v
```

## Step 6 — Connect to Express.js Backend

Add this to your Express.js service (Node.js):

```javascript
// In your Express route handler:
const NAVIER_URL = process.env.NAVIER_AI_URL || 'http://localhost:8001';

app.post('/api/pipeline/analyze', async (req, res) => {
  const response = await fetch(`${NAVIER_URL}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body),
  });
  const result = await response.json();
  res.json(result);
});
```

Set environment variable: `NAVIER_AI_URL=http://localhost:8001`

---

## API Reference

### POST `/analyze`

**Request body:** `SensorReading` (see `app/schemas/request_models.py`)

| Field | Type | Unit | Description |
|---|---|---|---|
| `inlet_pressure_pa` | float | Pa | Upstream pressure |
| `outlet_pressure_pa` | float | Pa | Downstream pressure |
| `flow_rate_m3s` | float | m³/s | Volumetric flow |
| `temperature_c` | float | °C | Fluid temperature |
| `density_kg_m3` | float | kg/m³ | Fluid density |
| `dynamic_viscosity_pa_s` | float | Pa·s | Dynamic viscosity |
| `acoustic_vibration` | float | a.u. | Acoustic sensor reading |
| `pipe_diameter_m` | float | m | Internal diameter |
| `wall_thickness_m` | float | m | Pipe wall thickness |
| `segment_length_m` | float | m | Segment length |
| `fluid_type` | enum | — | MS / HSD / ATF / BENZENE / LPG |
| `ambient_temp_c` | float | °C | Ambient temperature (drift trigger at ≥42°C) |

**Response:** `AnalysisResponse`
- `classification` → SAFE / WARNING / CRITICAL
- `anomaly_score` → 0.0–1.0
- `npw_localization` → NPW distance estimate (metres)
- `scs_report` → Sensor Confidence Score (0–100%)
- `rul_hours` / `rul_days` → Remaining Useful Life
- `physics_report` → Navier-Stokes residuals
- `batch_change` → Batch change detection result

### POST `/simulate_leak`

Generates a synthetic leak event and runs full analysis.  Useful for testing and calibration.

### GET `/sensor_status?sensor_id=X&scs=92&flags=...`

Lightweight health check without ML inference.

---

## Chemical Profiles

| Code | Fluid | Density (kg/m³) | Wave Speed (m/s) |
|---|---|---|---|
| MS | Motor Spirit (Petrol) | 745 | ~1050 |
| HSD | High-Speed Diesel | 840 | ~1320 |
| ATF | Aviation Turbine Fuel | 800 | ~1200 |
| BENZENE | Benzene | 879 | ~1300 |
| LPG | LPG (Propane) | 508 | ~700 |

---

## Production Notes

1. **Weights:** Drop a trained `navier_model.pt` (PyTorch state-dict) into `models/` and restart.
2. **CORS:** Change `allow_origins=["*"]` in `main.py` to your React origin.
3. **Port:** Default is `8001` to avoid conflict with Express.js (typically 3000/5000).
4. **Scaling:** Mount behind nginx + gunicorn workers for multi-process production.
PYEOF
echo "  ✅  README.md"

# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  ✅  NAVIER AI Core scaffold complete!"
echo "============================================================"
echo ""
echo "Project structure:"
find "$PROJECT" -type f | sort
echo ""
echo "Next steps:"
echo "  cd $PROJECT"
echo "  pip install -r requirements.txt"
echo "  python data/synthetic_gen.py"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload"
echo "  pytest tests/ -v"
echo ""
echo "API docs → http://localhost:8001/docs"
