# ==== Standard Imports ====
import logging
from pathlib import Path

# ==== Third Party Imports ====
import numpy as np
from pydicom import dcmread


def save_arrays(output_path: Path, filename: str, ct_volume: np.ndarray, pet_volume: np.ndarray, label: str):
    """
    Saves the 3D NumPy arrays for the CT and PET scans as a .npy file.

    Args:
    :param output_path: Path to save the created .npy file.
    :param filename: Name of the file.
    :param ct_volume: 3D pixel value array representing the CT scan series.
    :param pet_volume: 3D pixel value array representing the PET scan series.
    :param label: An uppercase character representing the non-small cell lung cancer type.
    """
    logger = logging.getLogger('PreprocessingLogger')
    output_path.mkdir(parents=True, exist_ok=True)
    try:
        file_path = output_path / f'{filename}.npy'
        np.save(file_path, {'label': label, 'CT': ct_volume, 'PET': pet_volume})
        logger.info(f'CT and PET volumes successfully saved to {file_path}')
    except Exception as e:
        logger.error(f'Error saving arrays: {e}')

class DicomConverter:
    """
    A class to handle the conversion of DICOM files into 2D or
    3D image arrays. This class supports loading DICOM files
    from a specified directory, sorting them, and converting
    them into 2D arrays or stacking them into a 3D volume.
    """

    def __init__(self):
        """
        Initializes DicomConverter object.
        """
        self.logger = logging.getLogger('PreprocessingLogger')

    def to_2d_array(self, dicom_path: Path) -> list:
        """
        Loads DICOM files from a folder and returns a sorted list of (pixel_array, metadata) tuples.

        Args:
        :param dicom_path: The path containing the .dcm files
        :return: A List of tuples (pixel_array, metadata)
        """
        slices = []
        files = list(dicom_path.glob('*.dcm'))
        self.logger.info(f'Found {len(files)} DICOM files in {dicom_path}')

        for file in files:
            try:
                ds = dcmread(file)
                slices.append((ds.InstanceNumber, ds.pixel_array, ds))
            except Exception as e:
                self.logger.warning(f'Skipped {file.name}: {e}')

        slices.sort(key=lambda x: x[0])
        self.logger.info(f'Successfully loaded and sorted {len(slices)} slices.')

        return [(pixel_array, metadata) for _, pixel_array, metadata in slices]

    @staticmethod
    def to_3d_array(slices: list) -> np.ndarray:
        """
        Converts a list of 2D image arrays into a 3D shape.

        Args:
        :param slices: The list of images to be converted.
        :return: A 3D image array.
        """
        return np.stack(slices, axis=0)