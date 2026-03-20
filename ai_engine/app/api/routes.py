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
