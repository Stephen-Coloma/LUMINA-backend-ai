from app.preprocess.ai.dicom_converter import DicomConverter
from app.preprocess.ai.volume_processing import VolumeProcessor
from app.preprocess.ai.intensity_processing import IntensityProcessor
from pathlib import Path
import torch.nn.functional as F
import torch

# Define preprocessing for AI
def preprocess_patient_data(ct_dir: Path, pet_dir: Path):

    try:
        # Convert slices to 2D NumPy arrays
        ct_slices = DicomConverter().to_2d_array(ct_dir)
        pet_slices = DicomConverter().to_2d_array(pet_dir)

        # Apply pixel intensity conversion and normalization
        ct_slices = IntensityProcessor(ct_slices, True).convert()
        pet_slices = IntensityProcessor(pet_slices, True).convert()

        # Stack the 2D NumPy arrays to a 3D shape
        ct_volume = DicomConverter.to_3d_array(ct_slices)
        pet_volume = DicomConverter.to_3d_array(pet_slices)

        # Apply standardization (resize depth)
        ct_volume = VolumeProcessor(ct_volume).resize_depth(32)
        pet_volume = VolumeProcessor(pet_volume).resize_depth(32)

        print(f"CT volume shape after depth resize: {ct_volume.shape}")
        print(f"PET volume shape after depth resize: {pet_volume.shape}")

        ct_volume = torch.tensor(ct_volume, dtype=torch.float32)
        pet_volume = torch.tensor(pet_volume, dtype=torch.float32)

        print(f"CT volume shape after tensor conversion: {ct_volume.shape}")
        print(f"PET volume shape after tensor conversion: {pet_volume.shape}")

        if ct_volume.ndim == 3:
            ct_volume = ct_volume.unsqueeze(0).unsqueeze(0)
        if pet_volume.ndim == 3:
            pet_volume = pet_volume.unsqueeze(0).unsqueeze(0)
        
        print(f"CT volume shape after unsqueeze: {ct_volume.shape}")
        print(f"PET volume shape after unsqueeze: {pet_volume.shape}")

        # ct_volume = F.interpolate(ct_volume, size=(200, 200), mode='bilinear', align_corners=False)
        # pet_volume = F.interpolate(pet_volume, size=(200, 200), mode='bilinear', align_corners=False)

        print(f"CT volume shape after interpolation: {ct_volume.shape}")
        return ct_volume, pet_volume

    except Exception as e:
        raise RuntimeError("Preprocessing failed.") from e


# Define preprocessing for Data Sci