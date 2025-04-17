import numpy as np
from pydicom import dcmread
from pathlib import Path
from tqdm import tqdm


def save_array(output_path: Path, filename: str, ct_volume: np.ndarray, pet_volume: np.ndarray):
    """
    Saves the 3D NumPy arrays for CT and PET to separate .npy files.

    :param output_path: Path to save the .npy files
    :param filename: The name of the file
    :param ct_volume: The 3D CT volume array
    :param pet_volume: The 3D PET volume array
    """
    try:
        np.save(output_path / f'{filename}_CT.npy', ct_volume)
        np.save(output_path / f'{filename}_PET.npy', pet_volume)
        print(f'Arrays successfully saved to {output_path}\\{filename}.pny and {output_path}\\{filename}.pny\n')
    except Exception as e:
        print(f'Error saving arrays: {e}\n')


class DicomConverter:
    def __init__(self, dicom_path: Path):
        self.dicom_path = dicom_path

    def convert_to_array(self):
        """
        Converts a folder of DICOM slices into a 3D NumPy array

        :return: A tuple of two 3D NumPy arrays
        """
        ct_slices = []
        pet_slices = []

        for modality in ['CT', 'PET']:
            modality_path = self.dicom_path / modality
            if not modality_path.exists():
                continue

            files = list(modality_path.glob('*.dcm'))
            for file in tqdm(files, desc=f'Converting {modality} slices to NumPy array'):
                try:
                    ds = dcmread(file)
                    dicom_slice = (ds.InstanceNumber, ds.pixel_array)

                    if modality == 'CT':
                        ct_slices.append(dicom_slice)
                    elif modality == 'PET':
                        pet_slices.append(dicom_slice)
                except Exception as e:
                    print(f'Skipped {file.name}: {e}')

        ct_slices.sort(key=lambda x: x[0])
        pet_slices.sort(key=lambda x: x[0])

        ct_volume = np.stack([slice_[1] for slice_ in ct_slices], axis=0)
        pet_volume = np.stack([slice_[1] for slice_ in pet_slices], axis=0)

        return ct_volume, pet_volume

