import pydicom as dicomio
import os

def load_file_info(filename):
    ds = dicomio.dcmread(filename)
    return ds.SOPInstanceUID

def get_uid_path(dicom_path):
    dicom_dict = {}

    for modality in ['CT', 'PET']:
        modality_path = os.path.join(dicom_path, modality)
        if not os.path.exists(modality_path):
            continue

        for dicom_file in os.listdir(modality_path):
            if dicom_file.lower().endswith('.dcm'):
                dicom_path = os.path.join(modality_path, dicom_file)
                try:
                    uid = load_file_info(dicom_path)
                    dicom_dict[uid] = dicom_file
                except Exception:
                    continue

    return dicom_dict