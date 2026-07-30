from fastapi import APIRouter, UploadFile, File
import shutil
import os

from backend.app.services.stress_inference_service import (
    predict_stress
)

router = APIRouter()

UPLOAD_DIR = "temp_uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/predict-stress")
async def predict_stress_api(
    file: UploadFile = File(...)
):

    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    result = predict_stress(file_path)

    return {
        "filename": file.filename,
        "prediction": result
    }
