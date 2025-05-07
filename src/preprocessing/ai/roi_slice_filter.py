from pydicom import dcmread
from pathlib import Path
import logging

class DicomROIFilter:
    def __init__(self, dicom_path: Path, anno_path: Path, patient_num: str):
        self.dicom_path = dicom_path
        self.anno_path = anno_path
        self.patient_num = patient_num
        self.roi_slices = []
        self.logger = logging.getLogger('PreprocessingLogger')
        self._get_roi_slices()

    def filter_slices(self):
        """
        Filters DICOM slices to retain only those that are a part of the ROI (Region of Interest).
        Slices that are not a part of the ROI are deleted.
        """
        for modality in ['CT', 'PET']:
            modality_path = self.dicom_path / modality
            if not modality_path.exists():
                self.logger.warning(f'{modality_path} does not exist.')
                continue

            files = list(modality_path.glob('*.dcm'))
            for file in files:
                if file.name not in self.roi_slices:
                    try:
                        file.unlink()
                    except Exception as e:
                        self.logger.error(f'Could not delete {file}: {e}')

            self.logger.info(f'Successfully filtered slices for {modality} scans')

    def _get_uid_paths(self) -> dict:
        """
        Retrieves the DICOM slice SOPInstanceUIDs to their file names from the DICOM directory.

        :return: A dictionary mapping SOPInstanceUID to DICOM file names.
        """
        dicom_dict = {}

        for modality in ['CT', 'PET']:
            modality_path = self.dicom_path / modality
            if not modality_path.exists():
                self.logger.warning(f'{modality_path} does not exist.')
                continue

            for dicom_file in modality_path.iterdir():
                if dicom_file.suffix.lower() == '.dcm':
                    try:
                        ds = dcmread(dicom_file)
                        uid = ds.SOPInstanceUID
                        dicom_dict[uid] = dicom_file.name
                    except Exception as e:
                        self.logger.error(f'Error reading {dicom_file}: {e}')
                        continue

        return dicom_dict

    def _get_roi_slices(self):
        """
        Populates the roi_slices list with DICOM file names that are a part of the ROI (Region of Interest).
        """
        dicom_dict = self._get_uid_paths()

        for anno_file in self.anno_path.iterdir():
            uid_key = anno_file.stem
            dicom_name = dicom_dict.get(uid_key)
            if dicom_name:
                self.roi_slices.append(dicom_name)

        print(self.roi_slices)