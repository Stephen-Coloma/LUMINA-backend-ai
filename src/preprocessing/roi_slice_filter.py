from tqdm import tqdm
from pydicom import dcmread
from pathlib import Path

class DicomROIFilter:
    def __init__(self, dicom_path: Path, anno_path: Path, patient_num: str):
        self.dicom_path = dicom_path
        self.anno_path = anno_path
        self.patient_num = patient_num
        self.roi_slices = []
        self._get_roi_slices()

    def _get_uid_paths(self):
        """
        Retrieves the DICOM slice SOPInstanceUIDs to their file names from the DICOM directory.

        :return: A dictionary mapping SOPInstanceUID to DICOM file names.
        """
        dicom_dict = {}

        for modality in ['CT', 'PET']:
            modality_path = self.dicom_path / modality
            if not modality_path.exists():
                continue

            for dicom_file in modality_path.iterdir():
                if dicom_file.suffix.lower() == '.dcm':
                    try:
                        ds = dcmread(dicom_file)
                        uid = ds.SOPInstanceUID
                        dicom_dict[uid] = dicom_file.name
                    except Exception as e:
                        print(f'Error reading {dicom_file}: {e}')
                        continue

        return dicom_dict

    def filter_slices(self):
        """
        Filters DICOM slices to retain only those that are a part of the ROI (Region of Interest). Slices that are
        not a part of the ROI are deleted.
        """
        for modality in ['CT', 'PET']:
            modality_path = self.dicom_path / modality
            if not modality_path.exists():
                continue

            files = modality_path.iterdir()
            for file in tqdm(files, desc=f'Filtering {modality} for {self.patient_num}'):
                if file.suffix.lower() == '.dcm' and file.name not in self.roi_slices:
                    try:
                        file.unlink()
                    except Exception as e:
                        print(f'Could not delete {file}: {e}')

    def _get_roi_slices(self):
        """
        Populates the roi_slices list with DICOM file names that are a part of the ROI (Region of Interest)
        """
        dicom_dict = self._get_uid_paths()

        for anno_file in self.anno_path.iterdir():
            uid_key = anno_file.stem
            dicom_name = dicom_dict.get(uid_key)
            if dicom_name:
                self.roi_slices.append(dicom_name)