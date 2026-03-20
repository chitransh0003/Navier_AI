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
