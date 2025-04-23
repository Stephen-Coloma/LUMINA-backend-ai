from src.preprocessing.roi_slice_filter import DicomROIFilter
from src.preprocessing.dicom_converter import DicomConverter
from src.preprocessing.volume_processing import VolumeProcessor
from src.preprocessing.intensity_processing import IntensityProcessor
from src.preprocessing.dicom_augmentor import Augmentator
from src.preprocessing.dicom_converter import save_arrays
from tqdm import tqdm
from pathlib import Path
from src.utils.logger import setup_logger

def augment_data(input_path):
    augmentor = Augmentator()
    tally = augmentor.get_minority_classes(input_path)
    target = max(tally.values())
    minority_classes = {k: target - v for k, v in tally.items() if v < target}

    for label, num_augmentations in minority_classes.items():
        augmentor.augment_class_x_times(
            label=label,
            num_augmentations=num_augmentations,
            npy_dir=input_path,
            output_dir='path/to/save/augmented/data'
        )
        print(f"Augmented {num_augmentations} samples for class {label}.")
    
    print("Data augmentation completed.")

def preprocess_patient_data(anno_path, output_path, patient_dir, logger):
    patient_num = patient_dir.name.split('-')[-1]
    patient_dicom_ct = patient_dir / 'CT'
    patient_dicom_pet = patient_dir / 'PET'
    patient_anno = anno_path / patient_num

    logger.info(f'=====< PRE-PROCESSING PATIENT {patient_num} >=====')
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
            return  # Skip patient

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
        
        # Save the processed slices
        label = patient_num[:1]
        save_arrays(output_path, patient_num, ct_volume, pet_volume, label)
        logger.info(
            '---------------------------------------------'
            f'Successfully processed patient {patient_num}'
        )

    except Exception as e:
        logger.error(f'Error processing patient {patient_num}: {e}')
        raise

def main():
    dicom_path = Path(r'D:\Datasets\TEST')
    anno_path = Path(r'D:\Datasets\Annotation')
    output_path = Path(r'D:\Datasets\Output')
 

    logger = setup_logger(Path('../logs'), 'preprocessing.log', 'PreprocessingLogger')

    directories = dicom_path.iterdir()
    for patient_dir in tqdm(directories, desc='Preprocessing'):
        if patient_dir.is_dir():
            preprocess_patient_data(anno_path, output_path, patient_dir, logger)
    
    augment_data(output_path)

if __name__ == '__main__':
    main()