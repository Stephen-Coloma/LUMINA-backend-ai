import numpy as np
import logging

class IntensityProcessor:
    def __init__(self, slices: list, normalize: bool = True):
        self.slices = slices
        self.normalize = normalize
        self.logger = logging.getLogger('PreprocessingLogger')

    def convert(self) -> list:
        """
        Converts the intensity of each slice depending on its modality.
        If normalization is true, images are normalized.

        :return: A list containing the processed array images.
        """
        converted_slices = []
        for image, metadata in self.slices:
            if metadata.Modality == 'CT':
                converted_img = self._convert_to_hu(image, metadata)
            else:
                converted_img = self._convert_to_suv(image, metadata)

            if self.normalize:
                converted_img = self._normalize(converted_img)

            converted_slices.append(converted_img)
        return converted_slices

    def _convert_to_hu(self, image: np.ndarray, metadata) -> np.ndarray:
        """
        Converts the image intensity to Hounsfield Units (HU) based on the DICOM metadata.

        :param image: The image array to be converted.
        :param metadata: A DICOM metadata associated with the image.
        :return: An image converted into Hounsfield Units.
        """
        try:
            rescale_slope = metadata.RescaleSlope
            rescale_intercept = metadata.RescaleIntercept
            hu = image * rescale_slope + rescale_intercept
            return hu
        except Exception as e:
            self.logger.error(f'Could not convert to Hu: {e}')
            return image

    def _convert_to_suv(self, image: np.ndarray, metadata) -> np.ndarray:
        """
        Convert the image intensity to Standardized Uptake Value (SUV) based on the DICOM metadata.

        :param image: The image array to be converted.
        :param metadata: A DICOM metadata associated with the image.
        :return: An image converted into Standardized Uptake Value.
        """
        try:
            weight = float(metadata.PatientWeight)
            rph_info = metadata.RadiopharmaceuticalInformationSequence[0]
            dose = float(rph_info.RadionuclideTotalDose) / 1e6
            suv = (image.astype(np.float32) * weight) / dose
            return suv
        except Exception as e:
            self.logger.error(f'Could not convert to SUV: {e}')
            return image

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        """
        Normalizes the image array to a range from 0 to 1.

        :param image: The image to be normalized
        :return: A Normalized image
        """
        try:
            return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        except Exception as e:
            logging.getLogger('PreprocessingLogger').error(f'Normalization failed: {e}')
            return image