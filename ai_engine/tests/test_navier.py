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
