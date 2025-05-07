from app.preprocess.ai.dicom_converter import DicomConverter
from app.preprocess.ai.volume_processing import VolumeProcessor
from app.preprocess.ai.intensity_processing import IntensityProcessor
from pathlib import Path

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
        ct_volume = VolumeProcessor(ct_volume).resize_depth(15)
        pet_volume = VolumeProcessor(pet_volume).resize_depth(15)


        return ct_volume, pet_volume

    except Exception as e:
        raise RuntimeError("Preprocessing failed.") from e


# Define preprocessing for Data Sci