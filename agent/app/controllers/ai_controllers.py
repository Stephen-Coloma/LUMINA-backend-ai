from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
from app.preprocess.preprocessing import preprocess_patient_data
from app.models.model import ai_model

async def process_zip_dicom(zip_path: Path):
    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Not a valid zip file.")

    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir_path)

        ct_dir = tmp_dir_path / 'CT'
        pet_dir = tmp_dir_path / 'PET'

        # Preprocess the Data
        ct_volume, pet_volume = preprocess_patient_data(ct_dir, pet_dir) 

        # Predict using the model
        prediction = ai_model(ct_volume, pet_volume)

        # Postprocess the result
        confidence, index = prediction.max(dim=0)
        index = index.item()

        labels = { "Ardenocarnicoma", "Squamous Cell Carcinoma", "Small Cell Lung Cancer"}
        classification = labels[index]
        confidence = confidence.item()

        # Return the result in variable format
        return classification, confidence
        
