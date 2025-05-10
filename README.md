# Lung Cancer DenseNet Classifier

## About
The `Lung Cancer DenseNet Classifier` is a deep learning model designed to classify
subtypes of lung cancer using 3D medical imaging data. This project leverages a 3D
DenseNet-based architecture for detecting subtle patterns within medical scans.

This model takes 3D CT and PET scan volumes as input and classifies them into distinct
lung cancer subtypes. It is built using PyTorch and includes preprocessing, feature extraction,
and classification components that handle volumetric data.

## Key Features
### 3D DenseNet Architecture
Utilizes DenseNet to maintain efficient information flow through deep layers,
which is crucial for 3D volumetric medical imaging tasks.
### Multi-modal Input
Designed to take in both CT and PET scans for richer feature extraction.
### End-to-End Solution
Contains the entire pipeline for lung cancer subtype classification from data
preprocessing to model training and evaluation.
### Somewhat Easy Parameter Tuning
A configuration file is integrated where hyperparameters can be changed along with the type of optimizers
and loss functions.

## Agent
A website is created as the agent for integrating the deep learning model.