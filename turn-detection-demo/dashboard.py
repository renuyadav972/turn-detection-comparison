"""
Turn Detection Metrics Dashboard
=================================
FastAPI server that serves the dashboard UI and provides a REST API for
session data collected by MetricsCollectorObserver.

Run:
    python dashboard.py
    # Dashboard at http://localhost:8080
"""

import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sessions"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Turn Detection Metrics Dashboard")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _read_session(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _session_summary(session: dict) -> dict:
    """Return a lightweight summary for the session list."""
    return {
        "session_id": session["session_id"],
        "mode": session["mode"],
        "started_at": session["started_at"],
        "ended_at": session.get("ended_at"),
        "config": session.get("config", {}),
        "summary": session.get("summary", {}),
    }


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))


@app.get("/api/sessions")
async def list_sessions():
    if not DATA_DIR.exists():
        return []
    sessions = []
    for p in sorted(DATA_DIR.glob("*.json")):
        s = _read_session(p)
        if s:
            sessions.append(_session_summary(s))
    return sessions


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    session = _read_session(path)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to read session")
    return session


@app.get("/api/compare")
async def compare_sessions(ids: str):
    id_list = [i.strip() for i in ids.split(",") if i.strip()]
    if len(id_list) < 2 or len(id_list) > 5:
        raise HTTPException(status_code=400, detail="Provide 2-5 session IDs")
    results = []
    for sid in id_list:
        path = DATA_DIR / f"{sid}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Session {sid} not found")
        s = _read_session(path)
        if not s:
            raise HTTPException(status_code=500, detail=f"Failed to read session {sid}")
        results.append(s)
    return results


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    print(f"Dashboard: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
