# 🫁 LUMINA - LUng Multimodal Integrated Network Assistant

This backend project is a **AI Model** in classifying subtypes of lung cancer based on CT and PET scans and Regression Model in detecting the presence of lung cancer through symptoms. 
This is part of our final project for on Artificcial Intelligence and Data Science courses A.Y. 2024-2025.

## 🧾 About
The `Lung Cancer DenseNet Classifier` is a deep learning model designed to classify
subtypes of lung cancer using 3D medical imaging data. This project leverages a 3D
DenseNet-based architecture for detecting subtle patterns within medical scans.

This model takes 3D CT and PET scan volumes as input and classifies them into distinct
lung cancer subtypes. It is built using PyTorch and includes preprocessing, feature extraction,
and classification components that handle volumetric data. 

Moreover, a regression model was developed in order to detect presence of lung cancer based on symptoms. 
Uses the library scikit-learn to implement the model. Users can input the symptoms they are experiencing and the 
regression will detect the likelihood of lung cancer based on these symptoms. 

## 📌 Key Features
### 3D DenseNet Architecture
Utilizes DenseNet to maintain efficient information flow through deep layers,
which is crucial for 3D volumetric medical imaging tasks.

### Multi-modal Input
Designed to take in both CT and PET scans for richer feature extraction.

### Regression Model
Uses regression techniques to predict continuous clinical outcomes based on extracted symptom features.

### Symptom Input
Incorporates patient symptom data alongside imaging inputs to improve prediction accuracy.

### End-to-End Solution
Contains the entire pipeline for lung cancer subtype classification from data
preprocessing to model training and evaluation.

### Somewhat Easy Parameter Tuning
A configuration file is integrated where hyperparameters can be changed along with the type of optimizers
and loss functions.

## Getting Started
### 📦 First, clone the project, change directory to 'agent' then create venv directory

```bash
cd agent

python -m venv venv
```

### 🏃🛠️ Install the dependencies

```bash
pip install -r requirements.txt
```

### 🏃‍♂️ Then run the backend server

```bash
venv\Scripts\activate
uvicorn app.main:app --reload
```

---

## 💻 Developers
1. BALOGO, Renuel Jeremi V.  
2. COLOMA, Stephen M.
3. GUZMAN, Sanchie Earl M.
4. LEUNG, Leonhard T.
5. NONATO, Marius Glenn M.
6. RAGUDOS, Hannah T.
7. RAMOS, Jerwin Kyle R.