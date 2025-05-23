from pathlib import Path
from tempfile import TemporaryDirectory
import zipfile
from app.preprocess.preprocessing import preprocess_patient_data
from app.models.model import ai_model
import torch.nn.functional as F # type: ignore

def process_zip_dicom(zip_path: Path):

    if not zipfile.is_zipfile(zip_path):
        raise ValueError("Not a valid zip file.")
    
    with TemporaryDirectory() as tmp_dir:
        tmp_dir_path = Path(tmp_dir)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir_path)
        
        ct_dir = tmp_dir_path / 'CT'
        pet_dir = tmp_dir_path / 'PET'
        
        # Preprocess the data
        ct_volume, pet_volume = preprocess_patient_data(ct_dir, pet_dir)
        print(f"CT volume shape: {ct_volume.shape}, PET volume shape: {pet_volume.shape}")

        # Convert ct_volume and pet_volume to tensors

        # Predict using the model
        prediction = ai_model(ct_volume, pet_volume)

        probabilities = F.softmax(prediction, dim=1)

        print('PROBABILITIES:', probabilities)
        print('TEST:', probabilities[0].max(dim=0))

        # Postprocess the resuls
        confidence, index = probabilities[0].max(dim=0)
        index = index.item()

        labels = ["Adenocarnicoma", "Small Cell Lung Cancer", "Squamous Cell Carcinoma"]
        classification = labels[index]
        confidence = confidence.item()

        # Return the result in variable format
        return classification, confidence

