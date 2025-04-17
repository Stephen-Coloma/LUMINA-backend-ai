from slice_selection import filter_slices

def main():
    dicom_path = r"D:\Datasets\Test"
    anno_path = r"D:\Datasets\Annotation"

    filter_slices(dicom_path, anno_path)

if __name__ == '__main__':
    main()