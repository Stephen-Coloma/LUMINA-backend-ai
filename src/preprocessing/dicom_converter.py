import numpy as np
from pydicom import dcmread
from pathlib import Path
import logging

def save_array(output_path: Path, filename: str, volume: np.ndarray, modality: str):
    """
    Saves a 3D NumPy array as a .npy file

    :param output_path: Path to save the created .npy file
    :param filename: Name of the file
    :param volume: 3D pixel value array representing the scan series
    :param modality: 'CT' or 'PET' to tag the filename
    """
    logger = logging.getLogger('DicomConverter')
    try:
        file_path = output_path / f'{filename}_{modality}.npy'
        np.save(file_path, volume)
        logger.info(f'3D array successfully saved to {file_path}')
    except Exception as e:
        logger.error(f'Error saving array: {e}')


class DicomConverter:
    def __init__(self):
        self.logger = logging.getLogger('DicomConverter')

    def to_2d_array(self, dicom_path: Path) -> list:
        """
        Loads DICOM files from a folder and returns a sorted list of (pixel_array, metadata) tuples.

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

        :param slices: The list of images to be converted.
        :return: A 3D image array.
        """
        return np.stack(slices, axis=0)