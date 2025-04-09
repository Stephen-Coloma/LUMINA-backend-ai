from utils import *


# path = 'Data/02-06-2020/DICOM/Lung_Dx-G0011'
# path = '/home/wangshuo/Desktop/test_data/Lung-PET-CT-Dx/G0011'
# path = 'Data/02-06-2020/DICOM/Lung_Dx-G0011/04-29-2009-LUNGC-51228/2.000000-A phase 5mm Stnd SS50-53792/2-017.dcm'


def getUID_path(path):
    dicom_dict = {}

    # Walk through all subdirectories and files in the given path
    for root, dirs, files in os.walk(path):
        for dicom_file in files:
            if dicom_file.lower().endswith('.dcm'):  # Check if the file is a DICOM file
                dicom_path = os.path.join(root, dicom_file)
                try:
                    # Assuming `loadFileInformation` gives you the DICOM number (UID)
                    info = loadFileInformation(dicom_path)
                    dicom_dict[info['dicom_num']] = (dicom_path, dicom_file)
                except Exception as e:
                    print(f"Error loading DICOM info for {dicom_path}: {e}")
                    continue

    return dicom_dict


def getUID_file(path):
    try:
        info = loadFileInformation(path)
        UID = info['dicom_num']
        return UID
    except Exception as e:
        print(f"Error loading DICOM info for {path}: {e}")
        return None