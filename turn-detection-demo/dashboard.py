"""
Turn Detection Metrics Dashboard
=================================
FastAPI server that serves the dashboard UI and provides a REST API for
session data collected by MetricsCollectorObserver.

Run standalone:
    python dashboard.py
    # Dashboard at http://localhost:8080

Or embed in another app:
    from dashboard import register_dashboard
    register_dashboard(your_app)
"""

import json
import os
from pathlib import Path

import uvicorn
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "sessions"
RECORDINGS_DIR = BASE_DIR / "data" / "recordings"
STATIC_DIR = BASE_DIR / "static"

router = APIRouter()


def _read_session(path: Path) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _recording_key(session: dict) -> str | None:
    """Derive recording filename key from session mode + config."""
    mode = session.get("mode")
    if mode in ("semantic", "smart_turn"):
        return "smart_turn"
    if mode == "flux_semantic":
        return "flux_semantic"
    if mode == "vad_only":
        vad_secs = session.get("config", {}).get("vad_stop_secs")
        if vad_secs is not None:
            return f"vad{round(vad_secs * 1000)}"
    return None


def _recording_url(session: dict) -> str | None:
    """Return the recording URL if the WAV file exists on disk."""
    key = _recording_key(session)
    if not key:
        return None
    wav_path = RECORDINGS_DIR / f"{key}_combined.wav"
    if wav_path.exists():
        return f"/recordings/{key}_combined.wav"
    return None


def _session_summary(session: dict) -> dict:
    """Return a lightweight summary for the session list."""
    _backfill_dead_air(session)
    return {
        "session_id": session["session_id"],
        "mode": session["mode"],
        "started_at": session["started_at"],
        "ended_at": session.get("ended_at"),
        "config": session.get("config", {}),
        "summary": session.get("summary", {}),
    }


@router.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "dashboard.html"))


@router.get("/api/sessions")
async def list_sessions():
    if not DATA_DIR.exists():
        return []
    sessions = []
    for p in sorted(DATA_DIR.glob("*.json")):
        s = _read_session(p)
        if s:
            sessions.append(_session_summary(s))
    return sessions


@router.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    session = _read_session(path)
    if not session:
        raise HTTPException(status_code=500, detail="Failed to read session")
    # Prefer Plivo recording URL stored in session JSON, fall back to local WAV
    if not session.get("recording_url"):
        url = _recording_url(session)
        if url:
            session["recording_url"] = url
    # Backfill incorrect dead air for old sessions missing the field
    _backfill_dead_air(session)
    return session


def _backfill_dead_air(session: dict):
    """Compute incorrect dead air metrics for sessions missing them."""
    dead_air_ms = 0
    total_unnecessary = 0
    for t in session.get("turns", []):
        lat = t.get("response_latency_ms", 0)
        p = t.get("pipeline", {})
        pipeline_ms = (p.get("stt_ttfb_ms") or 0) + (p.get("llm_ttfb_ms") or 0) + (p.get("tts_ttfb_ms") or 0)
        unnecessary = max(lat - pipeline_ms, 0)
        t.setdefault("pipeline_ms", round(pipeline_ms, 1))
        t.setdefault("unnecessary_dead_air_ms", round(unnecessary, 1))
        t.setdefault("pct_incorrect_dead_air", round(unnecessary / lat * 100, 1) if lat > 0 else 0)
        if lat > 0:
            dead_air_ms += lat
            total_unnecessary += unnecessary
    summary = session.get("summary", {})
    summary.setdefault("unnecessary_dead_air_ms", round(total_unnecessary))
    summary.setdefault("pct_incorrect_dead_air", round(total_unnecessary / dead_air_ms * 100, 1) if dead_air_ms > 0 else 0)


def _mode_key(session: dict) -> str | None:
    """Return a grouping key like 'vad300', 'vad700', 'smart_turn', 'flux_semantic'."""
    mode = session.get("mode")
    if mode in ("semantic", "smart_turn"):
        return "smart_turn"
    if mode == "flux_semantic":
        return "flux_semantic"
    if mode == "vad_only":
        vad_secs = session.get("config", {}).get("vad_stop_secs")
        if vad_secs is not None:
            return f"vad{round(vad_secs * 1000)}"
    return None


def _mode_label(key: str) -> str:
    labels = {
        "semantic": "Smart Turn",  # legacy
        "smart_turn": "Smart Turn",
        "flux_semantic": "Flux Semantic",
    }
    if key in labels:
        return labels[key]
    if key.startswith("vad"):
        return f"VAD {key[3:]}ms"
    return key


@router.get("/api/modes/summary")
async def modes_summary():
    if not DATA_DIR.exists():
        return []
    # Read all sessions and group by mode key
    groups: dict[str, list[dict]] = {}
    for p in DATA_DIR.glob("*.json"):
        s = _read_session(p)
        if not s:
            continue
        _backfill_dead_air(s)
        key = _mode_key(s)
        if key:
            groups.setdefault(key, []).append(s)

    result = []
    for key, sessions_list in sorted(groups.items()):
        n = len(sessions_list)
        summaries = [s.get("summary", {}) for s in sessions_list]

        def avg_field(field):
            vals = [sm[field] for sm in summaries if sm.get(field) is not None]
            return round(sum(vals) / len(vals), 2) if vals else None

        # Check for recording file for this mode key
        wav_path = RECORDINGS_DIR / f"{key}_combined.wav"
        recording_url = f"/recordings/{key}_combined.wav" if wav_path.exists() else None

        entry = {
            "mode_key": key,
            "label": _mode_label(key),
            "session_count": n,
            "recording_url": recording_url,
            "avg_response_latency_ms": avg_field("avg_response_latency_ms"),
            "avg_interruption_rate": avg_field("interruption_rate"),
            "avg_false_endpoint_rate": avg_field("false_endpoint_rate"),
            "avg_reprompt_rate": avg_field("reprompt_rate"),
            "avg_dead_air_s": avg_field("dead_air_s"),
            "avg_pct_incorrect_dead_air": avg_field("pct_incorrect_dead_air"),
        }
        result.append(entry)

    return result


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    path = DATA_DIR / f"{session_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found")
    path.unlink()
    return {"message": "deleted"}


@router.get("/api/compare")
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
        url = _recording_url(s)
        if url:
            s["recording_url"] = url
        results.append(s)
    return results


def register_dashboard(target_app: FastAPI):
    """Register all dashboard routes and static file mounts on the given app."""
    target_app.include_router(router)
    target_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    if RECORDINGS_DIR.exists():
        target_app.mount("/recordings", StaticFiles(directory=str(RECORDINGS_DIR)), name="recordings")


# Standalone app for `python dashboard.py`
app = FastAPI(title="Turn Detection Metrics Dashboard")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
register_dashboard(app)


if __name__ == "__main__":
    port = int(os.getenv("DASHBOARD_PORT", "8080"))
    print(f"Dashboard: http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
