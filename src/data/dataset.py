# ==== Standard Imports ====
from pathlib import Path
import logging

# ==== Third Party Imports ====
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class MedicalDataset(Dataset):
    """
    A custom PyTorch dataset class for loading medical imaging
    data stored in .npy format. The file contains a dictionary
    that composes of the label, CT volume, and PET volume.
    """

    def __init__(self, dataset_path: Path):
        """
        Initializes MedicalDataset object.

        Args:
        :param dataset_path: The path of the dataset containing .npy files.
        """
        self.logger = logging.getLogger('TrainingLogger')
        self.dataset = []

        # label mapping
        self.label_mapping = {
            'A': 0,
            'B': 1,
            'G': 2,
        }

        for data_file in dataset_path.glob('**/*.npy'):
            try:
                _ = np.load(data_file, allow_pickle=True).item()
                self.dataset.append(data_file)
            except Exception as e:
                self.logger.warning(f'Failed to load {data_file.name}: {e}')

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        :return: The total number of samples.
        """
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Retrieves one sample from the dataset.

        Args:
        :param index: The index of the sample to retrieve.
        :return: A tuple of label, ct, and pet tensors.
        """
        data_file = self.dataset[index]
        data = np.load(data_file, allow_pickle=True).item()

        # convert label string to integer index
        label_str = data['label']
        label = self.label_mapping[label_str]
        label = torch.tensor(label, dtype=torch.long)

        ct_volume = torch.tensor(data['CT'], dtype=torch.float32)
        pet_volume = torch.tensor(data['PET'], dtype=torch.float32)

        # add channel dim if it is missing
        if ct_volume.ndim == 3:
            ct_volume = ct_volume.unsqueeze(0)
        if pet_volume.ndim == 3:
            pet_volume = pet_volume.unsqueeze(0)

        # downsample spatial dimensions from 512x512 to 256x256
        ct_volume = F.interpolate(ct_volume, size=(200, 200), mode='bilinear', align_corners=False)
        pet_volume = F.interpolate(pet_volume, size=(200, 200), mode='bilinear', align_corners=False)

        return label, ct_volume, pet_volume