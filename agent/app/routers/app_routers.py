from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from app.controllers.ai_controllers import process_zip_dicom
from app.schemas.symptoms import Symptoms
from app.models.model import data_sci_model
import pandas as pd

router = APIRouter()

@router.post("/predict")
async def predict_images(file: UploadFile = File(...)):
    try:
        with TemporaryDirectory() as tmp_dir:
            zip_path = Path(tmp_dir) / file.filename
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            
            # Preprocess the data and get the prediction
            classification, confidence = process_zip_dicom(zip_path)
            print(f"Data processed. CT volume and PET volume retrieved.")

        return {
            "classification": classification,
            "confidence": confidence,  
        }

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/diagnose")
async def diagnose_symptoms(data: Symptoms):
    gender = 1 if data.gender == "female" else 0

    symptoms_dict = {
        "MALE": [gender],
        "AGE": [data.age],
        "SMOKING": [int(data.smoking)],
        "YELLOW_FINGERS": [int(data.yellowFingers)],
        "ANXIETY": [int(data.anxiety)],
        "PEER_PRESSURE": [int(data.peerPressure)],
        "CHRONIC DISEASE": [int(data.chronicDisease)],
        "FATIGUE": [int(data.fatigue)],
        "ALLERGY": [int(data.allergy)],
        "WHEEZING": [int(data.wheezing)],
        "ALCOHOL CONSUMING": [int(data.alcohol)],
        "COUGHING": [int(data.coughing)],
        "SHORTNESS OF BREATH": [int(data.shortnessOfBreath)],
        "SWALLOWING DIFFICULTY": [int(data.swallowingDifficulty)],
        "CHEST PAIN": [int(data.chestPain)],
    }

    symptoms_df = pd.DataFrame(symptoms_dict)
    prediction = data_sci_model.predict(symptoms_df)
    confidence = data_sci_model.predict_proba(symptoms_df)[0]
    
    if(prediction == 0):
        confidence = confidence[0]
    else:
        confidence = confidence[1]

    return {
        "prediction": int(prediction[0]),
        "confidence": float(confidence)
    }