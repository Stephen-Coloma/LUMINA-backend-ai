import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import logging

class MedicalDataset(Dataset):
    def __init__(self, dataset_path: Path):
        self.logger = logging.getLogger('TrainingLogger')
        self.dataset = []

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
        label = torch.tensor(data['label'], dtype=torch.long)

        ct_volume = torch.tensor(data['CT'], dtype=torch.float32)
        pet_volume = torch.tensor(data['PET'], dtype=torch.float32)

        # add channel dim if it is missing
        if ct_volume.ndim == 3:
            ct_volume = ct_volume.unsqueeze(0)
        if pet_volume.ndim == 3:
            pet_volume = pet_volume.unsqueeze(0)

        return label, ct_volume, pet_volume