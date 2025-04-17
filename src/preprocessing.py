from src.preprocessing.roi_slice_filter import DicomROIFilter
from src.preprocessing.dicom_converter import DicomConverter
from src.preprocessing.dicom_converter import save_array
from pathlib import Path

def main():
    dicom_path = Path(r'D:\Datasets\Test')
    anno_path = Path(r'D:\Datasets\Annotation')
    output_path = Path(r'D:\Datasets\Output')

    for patient_dir in dicom_path.iterdir():
        patient_num = patient_dir.name
        patient_dicom = dicom_path / patient_num
        patient_anno = anno_path / patient_num

        # Apply ROI filter
        roi_filter = DicomROIFilter(patient_dicom, patient_anno, patient_num)
        roi_filter.filter_slices()

        # convert slices to a 3D NumPy array
        converter = DicomConverter(patient_dicom)
        ct_volume, pet_volume = converter.convert_to_array()

        # save the array to an output directory
        save_array(output_path, patient_num, ct_volume, pet_volume)

if __name__ == '__main__':
    main()