"""Phase 7 - Session Recording routes.

Endpoints for exporting the current live recording and reading back stored
session recordings (the structured JSON that Phase 8 turns into a report).
"""

import glob
import json
import os

from fastapi import APIRouter

from fastapi import HTTPException

from backend.app.services.session_logger_service import (
    interpret_from_file,
    report_from_file,
    save_session,
    save_session_json,
    session_interpretation,
    session_report,
    session_status
)

router = APIRouter()

# outputs/session_logs (project root is four levels up from this file).
SESSION_LOG_DIR = os.path.join(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    ),
    "outputs",
    "session_logs"
)


@router.post("/session/export")
async def export_session():
    """Persist the flat CSV log (backward compatible)."""

    filename = await save_session()

    return {
        "saved": filename is not None,
        "file": filename
    }


@router.post("/session/export/json")
async def export_session_json():
    """Persist the structured JSON recording of the current session."""

    filename = await save_session_json()

    return {
        "saved": filename is not None,
        "file": filename
    }


@router.get("/session/status")
async def get_session_status():
    """Snapshot of the in-progress recording (id, frame count, ...)."""

    return await session_status()


@router.get("/session/recordings")
def list_recordings():
    """List stored JSON recordings, newest first."""

    files = glob.glob(
        os.path.join(SESSION_LOG_DIR, "*.json")
    )

    files.sort(key=os.path.getctime, reverse=True)

    return {
        "recordings": [
            os.path.basename(path) for path in files
        ]
    }


@router.get("/session/recordings/latest")
def latest_recording():
    """Return the most recent stored JSON recording."""

    files = glob.glob(
        os.path.join(SESSION_LOG_DIR, "*.json")
    )

    if not files:
        return {"error": "No session recordings found"}

    latest = max(files, key=os.path.getctime)

    with open(latest, "r", encoding="utf-8") as file:
        return json.load(file)


# ----------------------------------------------------------------------
# Phase 8 - Automatic Report Generator
# ----------------------------------------------------------------------


@router.get("/session/report")
async def get_session_report():
    """Interview report for the current (live / in-memory) recording."""

    return await session_report()


@router.post("/session/report/export")
async def export_session_report():
    """Persist the current session's report to ``<session_id>_report.json``."""

    report = await session_report()

    if report.get("frame_count", 0) == 0:
        return {"saved": False, "file": None}

    filename = os.path.join(
        SESSION_LOG_DIR,
        f"{report['session_id']}_report.json"
    )

    with open(filename, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    return {"saved": True, "file": filename}


@router.get("/session/report/{filename}")
def get_stored_report(filename: str):
    """Interview report for a stored JSON recording (by file name)."""

    # Guard against path traversal - only a bare file name is accepted.
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid file name")

    path = os.path.join(SESSION_LOG_DIR, filename)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Recording not found")

    return report_from_file(path)


# ----------------------------------------------------------------------
# Phase 9 - LLM Interpretation
# ----------------------------------------------------------------------


@router.post("/session/interpret")
async def interpret_session():
    """Local LLM (Ollama) interpretation of the current live recording.

    POST because it triggers an LLM generation. Always returns a structured
    interpretation; ``llm_available: false`` marks the rule-based fallback.
    """

    return await session_interpretation()


@router.post("/session/interpret/{filename}")
def interpret_stored_session(filename: str):
    """LLM interpretation of a stored JSON recording (by file name)."""

    # Guard against path traversal - only a bare file name is accepted.
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid file name")

    path = os.path.join(SESSION_LOG_DIR, filename)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Recording not found")

    return interpret_from_file(path)
