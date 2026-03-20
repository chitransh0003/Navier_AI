# Navier_AI

What is this?

NAVIER is an AI-powered petroleum pipeline monitoring system. It detects leaks, estimates where the leak is happening on the pipe, scores sensor health, and estimates how long before the pipeline needs maintenance — all in real time. It is built as a microservice that connects to a React frontend and an Express.js backend.


How it works
You send 10 sensor readings (pressure, flow, temperature, density, viscosity, acoustic vibration, pipe dimensions) to the API. The system runs them through three layers in sequence:

Physics layer — checks if the readings make physical sense using 1-D Navier-Stokes equations (mass and momentum conservation). If the measured pressure drop is far higher than what friction alone would cause, the physics residual score goes up. This is the deterministic safety net.

AI model — a hybrid neural network runs on top of the physics layer. It has three branches that run in parallel and combine their outputs. The results feed into a shared head that produces four outputs simultaneously — classification, leak distance, sensor confidence, and remaining useful life.

Causal guard — before and after the AI runs, a rule-based layer catches things the model might miss: sensor drift caused by high ambient temperature (relevant for 45°C Indian field conditions), batch changes when the pipeline switches from one fluid to another (which would otherwise look like a leak), and false alarms caused by PINN divergence.


Project structure

navier_ai_core/       ← AI model (FastAPI)

backend/              ← Express.js proxy server

frontend/             ← React website

venv/                 ← Python virtual environment


Requirements

Python 3.9+

Node.js 18+

pip and npm


Setup — first time only

bash# Python dependencies

cd navier_ai_core

pip install -r requirements.txt


# Backend dependencies

cd ../backend

npm install


# Frontend dependencies

cd ../frontend

npm install


Running the project

Open 3 terminals and run one command in each:

Terminal 1 — AI Model

bashcd navier_ai_core

source ../venv/bin/activate

uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

Terminal 2 — Backend

bashcd backend

node server.js

Terminal 3 — Frontend

bashcd frontend

npm run dev

Then open http://localhost:5173 in your browser.


API quick test

Once the AI model is running, open http://localhost:8001/docs in your browser. You will see the full Swagger UI where you can test the API directly without any frontend.