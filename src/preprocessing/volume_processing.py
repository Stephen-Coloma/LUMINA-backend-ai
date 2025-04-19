import numpy as np
import logging

class VolumeProcessor:
    def __init__(self, input_array: np.ndarray):
        self.input_array = input_array
        self.logger = logging.getLogger('VolumeProcessor')

    def resize_depth(self, target_depth: int = 15) -> np.ndarray:
        """
        Resizes the depth of a 3D image array to a targeted depth.
        1. If the input depth is greater than the target depth, the edges are cropped.
        2. If the input depth is less than the target depth, a padding will be applied.

        :param target_depth: The desired number of slices.
        :return: A 3D image array with a resized depth.
        """
        current_depth = self.input_array.shape[0]
        self.logger.info(f'Resizing slices from {current_depth} to {target_depth}')

        if current_depth > target_depth:
            middle_index = current_depth // 2
            start = middle_index - target_depth // 2
            end = start + target_depth
            self.input_array = self.input_array[start:end]
        elif current_depth < target_depth:
            pad_size = target_depth - current_depth
            self.input_array = np.pad(self.input_array, ((0, pad_size), (0, 0), (0, 0)), mode='edge')

        return self.input_array