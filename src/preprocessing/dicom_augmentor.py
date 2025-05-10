# ==== Standard Imports ====
from pathlib import Path
from collections import defaultdict
from random import choice
import logging

# ==== Third Party Imports ====
import torch
import torchio as tio
import numpy as np


class Augmenter:
    """
    A class that handles image augmentation operations on medical
    imaging data. It can apply a series of default or custom
    augmentation transforms on the data and also provide functionality
    to augment specific classes by a specified number of times.
    """

    def __init__(self, use_default_transforms=True, custom_transforms=None):
        """
        Initialize the augmentor with either default or custom transforms.

        Args:
        :param use_default_transforms: Flag to decide whether to use default transforms or not.
        :param custom_transforms: A list or composition of custom transformations to apply.
        """
        if custom_transforms:
            self.transform = custom_transforms
        elif use_default_transforms:
            self.transform = self._default_transforms()
        else:
            self.transform = tio.Compose([])

        self.logger = logging.getLogger('PreprocessingLogger')

    def get_minority_classes(self, path: Path) -> dict:
        """
        Scans a directory for .npy files, counting occurrences of
        each class label.

        Args:
        :param path: The directory containing the .npy files.
        :return: A dictionary where keys are labels and values are their counts.
        """
        label_tally = defaultdict(int)
        input_path = path

        for npy_file in input_path.glob('*.npy'):
            try:
                data = np.load(npy_file, allow_pickle=True).item()
                label = data.get('label')
                if label is not None:
                    label_tally[label] += 1
                else:
                    self.logger.warning(f'Label missing in file: {npy_file.name}')
            except Exception as e:
                self.logger.error(f'Error loading {npy_file}: {e}')

        return dict(label_tally)

    @staticmethod
    def _default_transforms():
        """
        Defines a set of default augmentations to be applied to
        the images.

        :return: A composition of default transformations to be applied to the images.
        """
        return tio.Compose([
            tio.RandomAnisotropy(p=0.25),
            tio.RandomAffine(),
            tio.RandomFlip(),
            tio.RandomNoise(p=0.25),
            tio.RandomGamma(p=0.5),
        ])

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """
        Apply augmentation transformations to a single image.

        Args:
        :param image: The input image to be augmented.
        :return: The augmented image.
        """
        tio_image = tio.ScalarImage(tensor = torch.tensor(image).unsqueeze(0).float())
        augmented = self.transform(tio_image)
        return augmented.data.squeeze().numpy()

    def augment_class_x_times(self, label: str, num_augmentations: int, npy_dir: Path, output_dir: Path):
        """
        Augments a specific class a set number of times, saving
        the augmented images in an output directory.

        Args:
        :param label: The label of the class to augment.
        :param num_augmentations: The number of augmented samples to create.
        :param npy_dir: The directory containing the original .npy files.
        :param output_dir: The output path where the augmented files will be saved.
        :return:
        """
        # """
        # Augment a specific class X times using random samples from that class and save results
        # using the format: {'label': ..., 'CT': ..., 'PET': ...}
        # """
        npy_path = npy_dir
        output_path = output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        samples = [
            np.load(f, allow_pickle=True).item()
            for f in npy_path.glob('*.npy')
            if np.load(f, allow_pickle=True).item().get('label') == label
        ]

        if not samples:
            self.logger.warning(f"No samples found for label '{label}' in {npy_path}")
            return

        for i in range(num_augmentations):
            seed = choice(samples)

            try:
                augmented_ct = self._augment(seed['CT'])
                augmented_pet = self._augment(seed['PET'])

                filename = f"{label}_aug_{i}"
                file_path = output_path / f'{filename}.npy'
                np.save(file_path, {
                    'label': label,
                    'CT': augmented_ct,
                    'PET': augmented_pet
                })
            except Exception as e:
                self.logger.error(f"Error augmenting or saving the label '{label}', index {i}: {e}")