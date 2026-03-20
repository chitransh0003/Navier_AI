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
