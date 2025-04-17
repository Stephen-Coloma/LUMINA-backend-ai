import numpy as np

def resize_slices(image_array, target_depth=15):
    current_depth = image_array.shape[0]

    if current_depth > target_depth:
        middle_index = current_depth // 2
        start = middle_index - target_depth // 2
        end = start + target_depth
        resized_image = image_array[start:end]
    elif current_depth < target_depth:
        pad_size = target_depth - current_depth
        padded_image = np.pad(image_array, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
        resized_image = padded_image
    else:
        resized_image = image_array

    return resized_image