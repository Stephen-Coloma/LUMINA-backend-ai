from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from app.controllers.ai_controllers import process_zip_dicom

router = APIRouter()

@router.post("/predict/")
async def predict_images(file: UploadFile = File(...)):
    try:
        with TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / file.filename
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)

            # Preprocess the data and get the prediction
            classification, confidence = process_zip_dicom(zip_path)

        return {
            "classification": classification,
            "confidence": confidence,
            "message": "Prediction successful"
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/diagnose/")
async def diagnose_symptoms():
    #Preprocess first
    #Predict using the model
    #Postprocess the result
    #Return the result
    return {"message": "Diagnose symptoms"}