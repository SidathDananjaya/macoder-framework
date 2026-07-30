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

SESSION_LOG_DIR = os.path.join(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    ),
    "outputs",
    "session_logs"
)


@router.post("/session/export")
async def export_session():
    filename = await save_session()

    return {
        "saved": filename is not None,
        "file": filename
    }


@router.post("/session/export/json")
async def export_session_json():
    filename = await save_session_json()

    return {
        "saved": filename is not None,
        "file": filename
    }


@router.get("/session/status")
async def get_session_status():
    return await session_status()


@router.get("/session/recordings")
def list_recordings():
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
    files = glob.glob(
        os.path.join(SESSION_LOG_DIR, "*.json")
    )

    if not files:
        return {"error": "No session recordings found"}

    latest = max(files, key=os.path.getctime)

    with open(latest, "r", encoding="utf-8") as file:
        return json.load(file)


@router.get("/session/report")
async def get_session_report():
    return await session_report()


@router.post("/session/report/export")
async def export_session_report():
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
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid file name")

    path = os.path.join(SESSION_LOG_DIR, filename)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Recording not found")

    return report_from_file(path)


@router.post("/session/interpret")
async def interpret_session():
    return await session_interpretation()


@router.post("/session/interpret/{filename}")
def interpret_stored_session(filename: str):
    if os.path.basename(filename) != filename:
        raise HTTPException(status_code=400, detail="Invalid file name")

    path = os.path.join(SESSION_LOG_DIR, filename)

    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Recording not found")

    return interpret_from_file(path)
