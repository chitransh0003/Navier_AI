## How to Run NAVIER

### Start everything at once:

From the project root (where `start.sh` lives), in Git Bash or WSL:

```bash
chmod +x start.sh
./start.sh
```

On Windows, if you use Git Bash, the script activates either `venv/bin/activate` (Linux/macOS/WSL) or `venv/Scripts/activate` (Windows).

### Or start each part manually in 3 separate terminals:

**Terminal 1 — AI Model:**

```bash
cd ai_engine
# Linux / macOS / WSL:
source ../venv/bin/activate
# Windows (cmd): ..\venv\Scripts\activate.bat
# Windows (PowerShell): ..\venv\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 — Express Backend:**

```bash
cd backend
node server.js
```

**Terminal 3 — React Frontend:**

```bash
cd frontend/navier-insights-main/navier-insights-main
npm run dev
```

### URLs:

- Website:  http://localhost:5173
- Backend:  http://localhost:5000/api/health
- AI Model: http://localhost:8001/docs

### Notes:

- The AI model (`ai_engine/`) proxies through the Express backend; the frontend calls `http://localhost:5000/api/...`, not port 8001 directly.
- Express is a thin proxy to FastAPI; CORS is enabled for `http://localhost:5173`.
- Ensure `backend/.env` sets `NAVIER_AI_URL=http://localhost:8001` if you change the AI service URL.
