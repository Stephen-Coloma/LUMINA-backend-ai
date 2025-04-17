from src.utils.annotation_helper import get_uid_path
from tqdm import tqdm
import os

def filter_slices(dicom_path, anno_path):
    patients = os.listdir(dicom_path)

    for patient_num in patients:
        patient_dicom = os.path.join(dicom_path, patient_num)
        patient_anno = os.path.join(anno_path, patient_num)

        roi_slices = get_roi_slices(patient_dicom, patient_anno)
        delete_non_roi_slices(patient_dicom, roi_slices)

def get_roi_slices(dicom_path, anno_path):
    dicom_dict = get_uid_path(dicom_path)
    anno_files = os.listdir(anno_path)

    roi_slices = []
    for filename in anno_files:
        dcm_name = dicom_dict.get(filename[:-4])
        if dcm_name:
            roi_slices.append(dcm_name)

    return roi_slices

def delete_non_roi_slices(dicom_path, roi_slices):
    for modality in ['CT', 'PET']:
        modality_path = os.path.join(dicom_path, modality)
        if not os.path.exists(modality_path):
            continue

        for file in tqdm(os.listdir(modality_path), desc=f'Deleting {modality} slices for {os.path.basename(dicom_path)}'):
            file_path = os.path.join(modality_path, file)

            if file.endswith('.dcm') and file not in roi_slices:
                os.remove(file_path)