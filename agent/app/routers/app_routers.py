from fastapi import APIRouter, HTTPException, UploadFile, File
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
from app.controllers.ai_controllers import process_zip_dicom
from app.schemas import Symptoms
from app.models.model import data_sci_model

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

# Route for Data Science
@router.get("/diagnose")
async def diagnose_symptoms(data: Symptoms):
   gender = 1 if data.gender == "female" else 0
   
   input_features = [[
      gender,
      data.age,
      int(data.smoking),
      int(data.yellowFingers),
      int(data.anxiety),
      int(data.peerPressure),
      int(data.chronicDisease),
      int(data.fatigue),
      int(data.allergy),
      int(data.wheezing),
      int(data.alcohol),
      int(data.coughing),
      int(data.shortnessOfBreath),
      int(data.swallowingDifficulty),
      int(data.chestPain)
    ]]
   
   prediction = data_sci_model.predict(input_features)
   
   return {"prediction": int(prediction[0])}
