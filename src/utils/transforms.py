import numpy as np
import cv2
from PIL import Image as PILImage
from fastai.vision.augment import Transform
import random

def normalize_mask(mask, mapping):
    """
    Normalize a mask image based on a color-to-class mapping.

    This function converts a color mask (RGB) into a class mask where each pixel's class is determined
    by its color according to the provided mapping.

    Parameters:
    - mask (numpy.ndarray): An RGB image of the mask (3-channel).
    - mapping (dict): A dictionary mapping RGB color tuples to class IDs.

    Returns:
    - numpy.ndarray: A 2D array where each pixel's value represents its class.

    Raises:
    - ValueError: If the mask is not a 3-channel RGB image.
    """
    if mask.ndim != 3 or mask.shape[-1] != 3:
        raise ValueError(
            "La máscara debe ser una imagen RGB con 3 canales de color.")
    h, w, _ = mask.shape
    mask_class = np.zeros((h, w), dtype=np.uint8)
    for color, c in mapping.items():
        mask_class[(mask == color).all(axis=-1)] = c
    return mask_class

def denormalize_mask(mask_class, mapeo):
    """
    Denormalize a class mask back to a color mask using the provided mapping.

    This function converts a class mask, where each pixel's value represents its class, back to a
    color mask (RGB) using an inverse mapping of class IDs to RGB colors.

    Parameters:
    - mask_class (numpy.ndarray): A 2D array where each pixel's value represents its class.
    - mapeo (dict): A dictionary mapping class IDs to RGB color tuples.

    Returns:
    - numpy.ndarray: An RGB image of the mask (3-channel).
    """
    inverse_map = {v: k for k, v in mapeo.items()}
    h, w = mask_class.shape
    mask_color = np.zeros((h, w, 3), dtype=np.uint8)
    for c in np.unique(mask_class):
        color = inverse_map[c]
        mask_color[mask_class == c] = color
    return mask_color

def add_shadow(image, num_shadows, min_opacity, max_opacity):
    """
    Add random shadow effects to an image.

    This function generates a specified number of shadow polygons with random shapes, sizes,
    and opacities, and then blends them onto the original image.

    Parameters:
    - image (numpy.ndarray): The original image to which shadows will be added.
    - num_shadows (int): The number of shadows to add.
    - min_opacity (float): The minimum opacity of the shadows (0 to 1).
    - max_opacity (float): The maximum opacity of the shadows (0 to 1).

    Returns:
    - numpy.ndarray: The image with added shadow effects.
    """
    height, width = image.shape[:2]
    shadow_image = image.copy()

    for _ in range(num_shadows):
        # Randomly generate the shadow polygon
        x1, y1 = random.randint(0, width), random.randint(0, height)
        x2, y2 = random.randint(0, width), random.randint(0, height)
        x3, y3 = random.randint(0, width), random.randint(0, height)
        x4, y4 = random.randint(0, width), random.randint(0, height)

        # Create a black polygon to simulate the shadow
        poly = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
        cv2.fillPoly(shadow_image, [poly], (0, 0, 0))

        # Blending the shadow with the image
        alpha = random.uniform(min_opacity, max_opacity)
        mask = shadow_image.astype(bool)
        shadow_image = cv2.addWeighted(shadow_image, alpha, image, 1 - alpha, 0, image)
        image[mask] = shadow_image[mask]

    return image

class ShadowTransform(Transform):
    """
    A custom Transform class for adding shadows to images in a FastAI pipeline.

    This class is used to create a transformation that can be applied to images during data augmentation
    in a FastAI DataLoader. It adds a specified number of shadows with random opacities to each image.

    Parameters:
    - num_shadows (int): The number of shadows to add to each image.
    - min_opacity (float): The minimum opacity of the shadows (0 to 1).
    - max_opacity (float): The maximum opacity of the shadows (0 to 1).
    """
    def __init__(self, num_shadows=3, min_opacity=0.5, max_opacity=0.75):
        self.num_shadows = num_shadows
        self.min_opacity = min_opacity
        self.max_opacity = max_opacity

    def encodes(self, x: PILImage):
        return add_shadow(x, self.num_shadows, self.min_opacity, self.max_opacity)