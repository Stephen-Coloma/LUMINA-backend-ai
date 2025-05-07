from preprocess.ai.dicom_augmentor import DicomAugmentor
from preprocess.ai.dicom_converter import DicomConverter
from preprocess.ai.volume_processing import VolumeProcessor
from preprocess.ai.intensity_processing import IntensityProcessor
from preprocess.ai.roi_slice_filter import DicomROIFilter
from preprocess.ai.dicom_converter import save_arrays


# Define preprocessing for AI
# TODO: Change the parameters to accomodate the new data structure
def preprocess_patient_data(patient_dir: Path):
    patient_dicom_ct = patient_dir / 'CT'
    patient_dicom_pet = patient_dir / 'PET'

    try:

        # Check if there are DICOM files in both the CT and PET directories after filtering
        ct_files = list(patient_dicom_ct.glob('*.dcm'))
        pet_files = list(patient_dicom_pet.glob('*.dcm'))

        # Convert slices to 2D NumPy arrays
        ct_slices = DicomConverter().to_2d_array(patient_dicom_ct)
        pet_slices = DicomConverter().to_2d_array(patient_dicom_pet)

        # Apply pixel intensity conversion and normalization
        ct_slices = IntensityProcessor(ct_slices, True).convert()
        pet_slices = IntensityProcessor(pet_slices, True).convert()

        # Stack the 2D NumPy arrays to a 3D shape
        ct_volume = DicomConverter.to_3d_array(ct_slices)
        pet_volume = DicomConverter.to_3d_array(pet_slices)

        # Apply standardization (resize depth)
        ct_volume = VolumeProcessor(ct_volume).resize_depth(15)
        pet_volume = VolumeProcessor(pet_volume).resize_depth(15)

        return {
            'ct_volume': ct_volume,
            'pet_volume': pet_volume,
        }


    except Exception as e:
        raise
# Define preprocessing for Data Sci