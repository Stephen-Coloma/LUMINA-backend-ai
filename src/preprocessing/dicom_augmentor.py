import torch
import torchio as tio
import numpy as np

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

    def _default_transforms(self):

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
