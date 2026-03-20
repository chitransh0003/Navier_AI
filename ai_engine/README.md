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
