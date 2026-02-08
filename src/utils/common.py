"""
Common utility functions for the MNIST research project.
"""

import numpy as np
from typing import Tuple, Union


def validate_image_array(image: np.ndarray) -> bool:
    """
    Validate that input is a proper image array.
    
    Args:
        image: Input array to validate
        
    Returns:
        bool: True if valid image array
    """
    if not isinstance(image, np.ndarray):
        raise TypeError("Input must be a numpy array")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image must be 2D or 3D array")
    
    return True


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to [0, 1] range.
    
    Args:
        image: Input image array
        
    Returns:
        Normalized image array
    """
    validate_image_array(image)
    return image.astype(np.float32) / 255.0


def crop_perimeter(image: np.ndarray, crop_size: int = 2) -> np.ndarray:
    """
    Remove pixels from the perimeter of an image.
    
    Args:
        image: Input image array
        crop_size: Number of pixels to remove from each side
        
    Returns:
        Cropped image array
    """
    validate_image_array(image)
    
    if len(image.shape) == 2:
        return image[crop_size:-crop_size, crop_size:-crop_size]
    elif len(image.shape) == 3:
        return image[crop_size:-crop_size, crop_size:-crop_size, :]
    else:
        raise ValueError("Unsupported image dimensions")
