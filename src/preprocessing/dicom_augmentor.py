import torch
import torchio as tio
import numpy as np
from pathlib import Path
from collections import defaultdict
from random import choice

class Augmentator:

    def __init__(self, use_default_transforms=True, custom_transforms=None):
        """
        Initialize the augmentor with either default or custom transforms.
        """
        if custom_transforms:
            self.transform = custom_transforms
        elif use_default_transforms:
            self.transform = self._default_transforms()
        else:
            self.transform = tio.Compose([]) 

    def get_minority_classes(self, path: str) -> dict:
        """
        Get the class distribution from a directory of .npy files.
        """
        label_tally = defaultdict(int)
        input_path = Path(path)

        for npy_file in input_path.glob('*.npy'):
            try:
                data = np.load(npy_file, allow_pickle=True).item()
                label = data.get('label')
                if label is not None:
                    label_tally[label] += 1
                else:
                    print(f"Label missing in file: {npy_file.name}")
            except Exception as e:
                print(f"Error loading {npy_file.name}: {e}")

        return dict(label_tally)

    def _default_transforms(self):
        """
        Default 3D medical image augmentations.
        """
        return tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(),
            tio.RandomFlip(),
            tio.RandomNoise(p=0.25),
            tio.RandomGamma(p=0.5),
        ])

    def augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the transformation to a 3D image (numpy array).
        """
        tio_image = tio.ScalarImage(tensor=torch.tensor(image).unsqueeze(0).unsqueeze(0).float())
        augmented = self.transform(tio_image)
        return augmented.data.squeeze().numpy()

    def augment_class_x_times(self, label: str, num_augmentations: int, npy_dir: str, output_dir: str):
        """
        Augment a specific class X times using random samples from that class and save results
        using the format: {'label': ..., 'CT': ..., 'PET': ...}
        """
        npy_path = Path(npy_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        samples = [
            np.load(f, allow_pickle=True).item()
            for f in npy_path.glob('*.npy')
            if np.load(f, allow_pickle=True).item().get('label') == label
        ]

        if not samples:
            print(f"No samples found for label '{label}' in {npy_path}")
            return

        for i in range(num_augmentations):
            seed = choice(samples)

            try:
                augmented_ct = self.augment(seed['CT'])
                augmented_pet = self.augment(seed['PET'])

                filename = f"{label}_aug_{i}"
                file_path = output_path / f'{filename}.npy'
                np.save(file_path, {
                    'label': label,
                    'CT': augmented_ct,
                    'PET': augmented_pet
                })
            except Exception as e:
                print(f'Error augmenting or saving for label {label}, index {i}: {e}')