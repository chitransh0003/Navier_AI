"""
interactive_test.py — Manual testing tool for NAVIER AI Core
=============================================================
Run this file directly WITHOUT needing a running server:

    python interactive_test.py

It runs 6 hand-crafted scenarios and prints a full human-readable
report for each one so you can see exactly what the model produces
for different input values.

You can edit the INPUT VALUES sections below to test your own numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Suppress torch warning for cleaner output
import logging
logging.basicConfig(level=logging.WARNING)

from app.api.routes import _run_full_analysis
from app.schemas.request_models import SensorReading, FluidType

# ─────────────────────────────────────────────────────────────────────────────
# Colour helpers (works on Linux/Mac terminal; falls back gracefully on Windows)
# ─────────────────────────────────────────────────────────────────────────────
try:
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
except Exception:
    GREEN = YELLOW = RED = CYAN = BOLD = RESET = ""

def _colour(text: str, classification: str) -> str:
    c = {"SAFE": GREEN, "WARNING": YELLOW, "CRITICAL": RED}.get(classification, "")
    return f"{c}{BOLD}{text}{RESET}"

def _bar(value: float, max_val: float = 100.0, width: int = 30, fill: str = "█") -> str:
    filled = int((value / max_val) * width)
    return fill * filled + "░" * (width - filled)

def print_result(scenario_name: str, reading: SensorReading) -> None:
    """Run the full analysis and print a formatted report."""
    result = _run_full_analysis(reading)

    cls   = result.classification.value
    score = result.anomaly_score

    print(f"\n{'='*65}")
    print(f"  {BOLD}{CYAN}SCENARIO:{RESET} {scenario_name}")
    print(f"{'='*65}")

    # ── Primary result ────────────────────────────────────────────────────────
    print(f"\n  {BOLD}STATUS:{RESET}        {_colour(cls, cls)}")
    print(f"  {BOLD}ALERT:{RESET}         {result.alert_message}")
    print(f"  {BOLD}ANOMALY SCORE:{RESET} {score:.4f}  [{_bar(score * 100, fill='▓')}]")

    # Class probabilities
    probs = result.class_probabilities
    print(f"\n  {BOLD}CLASS PROBABILITIES:{RESET}")
    for label, prob in probs.items():
        bar_char = {"SAFE": "░", "WARNING": "▒", "CRITICAL": "█"}[label]
        print(f"    {label:<10} {prob*100:5.1f}%  [{_bar(prob * 100, fill=bar_char)}]")

    # ── Sensor Confidence Score ───────────────────────────────────────────────
    scs = result.scs_report
    scs_colour = GREEN if scs.scs >= 85 else (YELLOW if scs.scs >= 70 else RED)
    print(f"\n  {BOLD}SENSOR CONFIDENCE SCORE (SCS):{RESET}")
    print(f"    Score:         {scs_colour}{scs.scs:.1f}%{RESET}  [{_bar(scs.scs)}]")
    print(f"    PINN Divergence: {scs.overall_divergence_pct:.2f}%", end="")
    if scs.overall_divergence_pct > 20:
        print(f"  {YELLOW}⚠️  >20% → SCS forced below 80%{RESET}", end="")
    print()
    print(f"    Ambient Temp:  {scs.ambient_temp_c:.1f}°C", end="")
    if scs.ambient_temp_c >= 42:
        print(f"  {YELLOW}🌡️  Indian heat drift risk{RESET}", end="")
    print()
    if scs.flags:
        print(f"    Flags:         {', '.join(scs.flags)}")
    print(f"    Per-sensor divergence:")
    for k, v in scs.per_sensor_divergence.items():
        status = "🔴" if v > 20 else ("🟡" if v > 10 else "🟢")
        print(f"      {k:<25} {v:6.2f}%  {status}")

    # ── NPW Localization ──────────────────────────────────────────────────────
    print(f"\n  {BOLD}NPW LEAK LOCALIZATION:{RESET}")
    if result.npw_localization:
        npw = result.npw_localization
        print(f"    Triggered:      YES ⚡")
        print(f"    Leak Distance:  {npw.leak_distance_m:.0f} m from upstream sensor")
        print(f"    Wave Speed:     {npw.wave_speed_ms:.0f} m/s  (Korteweg eq.)")
        print(f"    Δt estimated:   {npw.delta_t_s:.4f} s")
        print(f"    Confidence:     {npw.confidence*100:.0f}%  [{_bar(npw.confidence*100)}]")
        print(f"    Method:         {npw.method}")
    else:
        print(f"    Triggered:      NO  (pressure drop below threshold)")

    # ── RUL ───────────────────────────────────────────────────────────────────
    print(f"\n  {BOLD}REMAINING USEFUL LIFE (RUL):{RESET}")
    if result.rul_hours:
        rul_pct = min(100.0, result.rul_hours / (30 * 8760) * 100)
        rul_colour = GREEN if result.degradation_rate.value == "SLOW" else (
                     YELLOW if result.degradation_rate.value == "NORMAL" else RED)
        print(f"    Estimated RUL:  {result.rul_hours:,.0f} hours  ({result.rul_days:,.0f} days)")
        print(f"    Degradation:    {rul_colour}{result.degradation_rate.value}{RESET}  [{_bar(rul_pct)}]")

    # ── Physics (Navier-Stokes) ───────────────────────────────────────────────
    ph = result.physics_report
    print(f"\n  {BOLD}PHYSICS REPORT (1-D Navier-Stokes):{RESET}")
    print(f"    Reynolds Number:       {ph.reynolds_number:,.0f}  {'(turbulent)' if ph.is_turbulent else '(laminar)'}")
    print(f"    Darcy Friction Factor: {ph.darcy_friction_factor:.5f}")
    print(f"    Mass Conservation Res: {ph.mass_residual:.5f}  {'🔴 HIGH' if ph.mass_residual > 0.3 else '🟢 OK'}")
    print(f"    Momentum Residual:     {ph.momentum_residual:.5f}")
    print(f"    PINN Physics Loss:     {ph.pinn_loss_weight:.5f}")

    # ── Batch change ──────────────────────────────────────────────────────────
    bc = result.batch_change
    if bc.detected:
        print(f"\n  {BOLD}BATCH CHANGE:{RESET} {YELLOW}DETECTED ⚡{RESET}")
        print(f"    {bc.from_fluid} → {bc.to_fluid}  (Δρ = {bc.density_delta_kg_m3:.1f} kg/m³)")
        print(f"    False-alarm suppression: {'ACTIVE' if bc.suppression_active else 'OFF'}")

    print(f"\n  {BOLD}MODEL:{RESET}  backend={result.model_backend}  "
          f"inference={result.inference_time_ms:.1f}ms  "
          f"total={result.processing_time_ms:.1f}ms")
    print()


# =============================================================================
# ════════════════════════════════════════════════════════════════════════════
#  SCENARIO DEFINITIONS  ← Edit these values to test your own inputs!
# ════════════════════════════════════════════════════════════════════════════
# =============================================================================

def main():
    print(f"\n{BOLD}{CYAN}{'═'*65}")
    print("  NAVIER AI CORE — Interactive Manual Test Runner")
    print(f"{'═'*65}{RESET}")
    print("  Running 6 scenarios. Edit this file to change input values.\n")

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO 1: Normal HSD pipeline — everything healthy
    # ─────────────────────────────────────────────────────────────────────────
    print_result(
        "1. Normal HSD (Diesel) — Healthy Pipeline",
        SensorReading(
            # ── Core 10 sensor values ──────────────────────────────────────
            inlet_pressure_pa       = 4_500_000,   # 45 bar  ← EDIT ME
            outlet_pressure_pa      = 4_020_000,   # 40.2 bar
            flow_rate_m3s           = 0.25,         # m³/s
            temperature_c           = 42.0,         # °C
            density_kg_m3           = 840.0,        # HSD density
            dynamic_viscosity_pa_s  = 0.0035,       # Pa·s
            acoustic_vibration      = 10.5,         # normal baseline
            pipe_diameter_m         = 0.508,        # 20-inch pipe
            wall_thickness_m        = 0.0095,       # 9.5mm wall
            segment_length_m        = 10_000.0,     # 10 km segment
            # ── Metadata ──────────────────────────────────────────────────
            fluid_type              = FluidType.HSD,
            ambient_temp_c          = 35.0,
            sensor_id               = "RKL-SEG-04",
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO 2: CRITICAL leak — massive pressure drop + acoustic spike
    # ─────────────────────────────────────────────────────────────────────────
    print_result(
        "2. Critical Leak — Large Pressure Drop + Acoustic Spike",
        SensorReading(
            inlet_pressure_pa       = 3_800_000,   # dropped from 45 bar
            outlet_pressure_pa      = 2_400_000,   # massive downstream drop
            flow_rate_m3s           = 0.17,         # flow collapsed (32% loss)
            temperature_c           = 45.0,
            density_kg_m3           = 840.0,
            dynamic_viscosity_pa_s  = 0.0035,
            acoustic_vibration      = 52.0,         # ← big acoustic spike!
            pipe_diameter_m         = 0.508,
            wall_thickness_m        = 0.0095,
            segment_length_m        = 10_000.0,
            # Previous readings (enables NPW Δt calculation)
            inlet_pressure_prev_pa  = 4_500_000,   # what it was before
            outlet_pressure_prev_pa = 4_020_000,
            fluid_type              = FluidType.HSD,
            ambient_temp_c          = 42.0,
            sensor_id               = "RKL-SEG-04",
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO 3: MS (Petrol) — Indian summer 46°C sensor drift
    # ─────────────────────────────────────────────────────────────────────────
    print_result(
        "3. MS (Petrol) — 46°C Indian Summer Sensor Drift",
        SensorReading(
            inlet_pressure_pa       = 5_175_000,   # 15% high drift (should be 45 bar)
            outlet_pressure_pa      = 4_623_000,   # drifted outlet
            flow_rate_m3s           = 0.24,         # slightly low
            temperature_c           = 52.0,         # hot fluid
            density_kg_m3           = 745.0,        # MS (petrol) density
            dynamic_viscosity_pa_s  = 0.00055,      # MS is less viscous than HSD
            acoustic_vibration      = 11.2,
            pipe_diameter_m         = 0.508,
            wall_thickness_m        = 0.0095,
            segment_length_m        = 10_000.0,
            fluid_type              = FluidType.MS,
            ambient_temp_c          = 46.0,         # ← Indian summer heat
            sensor_id               = "MUM-SEG-12",
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO 4: Batch change MS (Petrol) → HSD (Diesel)
    # ─────────────────────────────────────────────────────────────────────────
    print_result(
        "4. Batch Change — Switching from MS (Petrol) to HSD (Diesel)",
        SensorReading(
            inlet_pressure_pa       = 4_500_000,
            outlet_pressure_pa      = 4_020_000,
            flow_rate_m3s           = 0.25,
            temperature_c           = 40.0,
            density_kg_m3           = 840.0,        # ← now HSD density (was 745)
            dynamic_viscosity_pa_s  = 0.0035,       # ← now HSD viscosity (was 0.00055)
            acoustic_vibration      = 18.0,         # slight turbulence during switch
            pipe_diameter_m         = 0.508,
            wall_thickness_m        = 0.0095,
            segment_length_m        = 10_000.0,
            fluid_type              = FluidType.HSD,
            previous_fluid_type     = FluidType.MS, # ← TRIGGERS batch change detection
            previous_density_kg_m3  = 745.0,        # what it was before
            previous_viscosity_pa_s = 0.00055,
            ambient_temp_c          = 38.0,
            sensor_id               = "DEL-SEG-03",
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO 5: ATF (Aviation Fuel) — small warning-level leak
    # ─────────────────────────────────────────────────────────────────────────
    print_result(
        "5. ATF (Aviation Fuel) — Minor Leak (Warning Level)",
        SensorReading(
            inlet_pressure_pa       = 4_300_000,   # slight drop
            outlet_pressure_pa      = 3_700_000,   # noticeable downstream loss
            flow_rate_m3s           = 0.22,         # minor flow reduction
            temperature_c           = 38.0,
            density_kg_m3           = 800.0,        # ATF density
            dynamic_viscosity_pa_s  = 0.0012,       # ATF viscosity
            acoustic_vibration      = 28.0,         # moderate acoustic signal
            pipe_diameter_m         = 0.508,
            wall_thickness_m        = 0.0095,
            segment_length_m        = 10_000.0,
            inlet_pressure_prev_pa  = 4_500_000,
            outlet_pressure_prev_pa = 4_150_000,
            fluid_type              = FluidType.ATF,
            ambient_temp_c          = 38.0,
            sensor_id               = "CHN-SEG-07",
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # SCENARIO 6: YOUR OWN VALUES — edit anything below!
    # ─────────────────────────────────────────────────────────────────────────
    print_result(
        "6. 🔧 YOUR CUSTOM VALUES — Edit these in interactive_test.py!",
        SensorReading(
            # ↓↓↓ CHANGE THESE TO YOUR OWN FIELD READINGS ↓↓↓
            inlet_pressure_pa       = 3_500_000,   # ← your inlet pressure in Pa (bar × 1e5)
            outlet_pressure_pa      = 3_100_000,   # ← your outlet pressure
            flow_rate_m3s           = 0.20,         # ← your flow rate m³/s
            temperature_c           = 50.0,         # ← your fluid temperature °C
            density_kg_m3           = 840.0,        # ← your fluid density kg/m³
            dynamic_viscosity_pa_s  = 0.0035,       # ← viscosity in Pa·s
            acoustic_vibration      = 15.0,         # ← acoustic sensor reading
            pipe_diameter_m         = 0.508,        # ← pipe internal diameter m
            wall_thickness_m        = 0.0095,       # ← wall thickness m
            segment_length_m        = 8_000.0,      # ← segment length m
            # Optional: add previous readings for NPW
            inlet_pressure_prev_pa  = None,         # ← previous inlet pressure (or None)
            outlet_pressure_prev_pa = None,         # ← previous outlet pressure (or None)
            fluid_type              = FluidType.HSD, # ← MS / HSD / ATF / BENZENE / LPG
            ambient_temp_c          = 40.0,         # ← ambient air temperature
            sensor_id               = "MY-SENSOR",
        )
    )

    # ─────────────────────────────────────────────────────────────────────────
    # Quick unit conversion reminder
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'─'*65}")
    print("  Unit Conversion Quick Reference")
    print(f"{'─'*65}{RESET}")
    print("  Pressure:   1 bar  = 100,000 Pa   →  45 bar = 4,500,000 Pa")
    print("  Pressure:   1 psi  =   6,895 Pa   →  650 psi ≈ 4,482,350 Pa")
    print("  Flow Rate:  1 m³/h = 1/3600 m³/s  →  900 m³/h = 0.25 m³/s")
    print("  Viscosity:  1 cSt × density/1000 = Pa·s  →  4.2 cSt HSD ≈ 0.0035 Pa·s")
    print("  Diameter:   20 inch = 508 mm = 0.508 m")
    print()


if __name__ == "__main__":
    main()
