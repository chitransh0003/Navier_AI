#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

activate_venv() {
  if [ -f "$ROOT/venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source "$ROOT/venv/bin/activate"
  elif [ -f "$ROOT/venv/Scripts/activate" ]; then
    # shellcheck source=/dev/null
    source "$ROOT/venv/Scripts/activate"
  else
    echo "No venv found at $ROOT/venv — run Python setup first."
    exit 1
  fi
}

echo "Starting NAVIER AI Model on port 8001..."
(
  cd "$ROOT/ai_engine"
  activate_venv
  uvicorn app.main:app --host 0.0.0.0 --port 8001
) &

echo "Starting Express backend on port 5000..."
(cd "$ROOT/backend" && node server.js) &

echo "Starting React frontend..."
(cd "$ROOT/frontend/navier-insights-main/navier-insights-main" && npm run dev) &

echo "All services started."
echo "Frontend:   http://localhost:5173"
echo "Backend:    http://localhost:5000"
echo "AI Model:   http://localhost:8001"
wait
