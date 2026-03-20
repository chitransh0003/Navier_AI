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
