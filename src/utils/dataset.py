import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path
import logging

class MedicalDataset(Dataset):
    def __init__(self, dataset_path: Path):
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
                item = np.load(data_file, allow_pickle=True).item()
                self.dataset.append(item)
            except Exception as e:
                self.logger.warning(f'Failed to load {data_file.name}: {e}')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        # convert label string to integer index
        label_str = data['label']
        label = self.label_mapping[label_str]
        label = torch.tensor(label, dtype=torch.uint8)

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