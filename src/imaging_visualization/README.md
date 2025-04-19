##### Usage:

1. Create an anaconda environment:

`conda create -n dcm-vis python=3.7`

2. Installation:

`pip install -r requirements.txt`

3. Visualization:

For single: `python visualization.py --dicom-mode CT --dicom-path path/to/DICOM/folder --annotation-path path/to/ANNOTATION/file.xml --classfile category.txt`

For folder: `python visualization.py --dicom-mode PET --dicom-path D:\Datasets\TEST\Lung_Dx-A0166 --annotation-path D:\Datasets\Annotation\A0166 --classfile category.txt`

Press `ESC` to show next one