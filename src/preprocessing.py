from src.preprocessing.roi_slice_filter import DicomROIFilter
from src.preprocessing.dicom_converter import DicomConverter
from src.preprocessing.volume_processing import VolumeProcessor
from src.preprocessing.intensity_processing import IntensityProcessor
from src.preprocessing.dicom_converter import save_array
from tqdm import tqdm
from pathlib import Path
import logging

def setup_logger():
    logger = logging.getLogger('Preprocessing')
    logger.setLevel(logging.INFO)
    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File Handler
    file_handler = logging.FileHandler('preprocessing.log')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def preprocess_patient_data(anno_path, output_path, patient_dir, logger):
    patient_num = patient_dir.name.split('-')[-1]
    patient_dicom_ct = patient_dir / 'CT'
    patient_dicom_pet = patient_dir / 'PET'
    patient_anno = anno_path / patient_num

    try:
        # Apply ROI filter
        roi_filter = DicomROIFilter(patient_dir, patient_anno, patient_num)
        roi_filter.filter_slices()

        # Check if there are DICOM files in both the CT and PET directories after filtering
        ct_files = list(patient_dicom_ct.glob('*.dcm'))
        pet_files = list(patient_dicom_pet.glob('*.dcm'))

        # Skip the patient if either CT or PET directory does not contain DICOM files
        if not ct_files or not pet_files:
            logger.info(f'Skipping patient {patient_num} due to missing DICOM files in CT or PET directory.')
            return  # Skip this patient

        # Convert slices to 2D NumPy arrays
        ct_slices = DicomConverter().to_2d_array(patient_dicom_ct)
        pet_slices = DicomConverter().to_2d_array(patient_dicom_pet)

        # Apply normalization (pixel intensity and normalization from 0 to 1)
        ct_slices = IntensityProcessor(ct_slices, True).convert()
        pet_slices = IntensityProcessor(pet_slices, True).convert()

        # Stack the 2D NumPy arrays to a 3D shape
        ct_volume = DicomConverter.to_3d_array(ct_slices)
        pet_volume = DicomConverter.to_3d_array(pet_slices)

        # Apply data augmentation if current patient is a minority class
        # TODO: Apply data augmentation if current patient is a minority class

        # Apply standardization (resize depth)
        ct_slices = VolumeProcessor(ct_volume).resize_depth(15)
        pet_slices = VolumeProcessor(pet_volume).resize_depth(15)
        # TODO: Augmented data should also be resized

        # Save the processed slices
        save_array(output_path, patient_num, ct_slices, 'CT')
        save_array(output_path, patient_num, pet_slices, 'PET')
        logger.info(f'Successfully processed patient {patient_num}')

    except Exception as e:
        logger.error(f'Error processing patient {patient_num}: {e}')
        raise


def main():
    dicom_path = Path(r'D:\Datasets\TEST')
    anno_path = Path(r'D:\Datasets\Annotation')
    output_path = Path(r'D:\Datasets\Output')

    logger = setup_logger()

    directories = dicom_path.iterdir()
    for patient_dir in tqdm(directories, desc='Preprocessing'):
        if patient_dir.is_dir():
            preprocess_patient_data(anno_path, output_path, patient_dir, logger)

if __name__ == '__main__':
    main()