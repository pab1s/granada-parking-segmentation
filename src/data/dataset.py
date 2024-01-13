import numpy as np
from fastai.vision.all import get_image_files, PILImage, PILMask
from src.utils.transforms import normalize_mask
from pathlib import Path
import yaml

def get_y_fn(x):
    """
    Obtain the corresponding mask file path for a given source image.

    This function takes the path of an image file and modifies it to point to the corresponding mask file. 
    It assumes that the mask files are in a different directory ('gt') and have a different file extension ('.png').

    Parameters:
    - x (Pathlib.Path or str): The path to the source image file.

    Returns:
    - Pathlib.Path: The path to the corresponding mask file.
    """
    return Path(str(x).replace('src', 'gt').replace('.JPG', '.png'))


def get_items(path):
    """
    Retrieve a list of all image files in the 'src' directory of a given path.

    This function scans the specified directory (path/'src') and returns a list of all image files.
    It is primarily used to collect source images for further processing or model training.

    Parameters:
    - path (Pathlib.Path): The path to the directory containing the 'src' folder.

    Returns:
    - List[Pathlib.Path]: A list of paths to the image files.
    """
    return get_image_files(Path(path)/'src')

def get_mask(item):
    """
    Retrieve the normalized mask for a given image file.

    This function reads the mask file corresponding to a given image, normalizes it based on a predefined
    color-to-class mapping defined in 'config.yaml', and returns the normalized mask.

    Parameters:
    - item (Pathlib.Path or str): The path to the source image file.

    Returns:
    - PILMask: A PILMask object representing the normalized mask.
    """
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    mapping_class_color = {tuple(item['color']): item['class'] for item in config['data']['mapping_class_color']}

    msk_path = get_y_fn(item)
    msk = np.array(PILImage.create(msk_path))
    return PILMask.create(normalize_mask(msk, mapping_class_color))