"""
Input/output utilities for loading and saving images and results.
"""

import numpy as np
from typing import Union
import torch


def load_image(path: str) -> np.ndarray:
    """
    Load an image from file.
    
    TODO: Implement image loading using PIL, imageio, or similar.
    For now, this is a placeholder that should be implemented
    based on expected input formats.
    
    Args:
        path: Path to image file
        
    Returns:
        img: Image array in (H, W, C) format, uint8 or float
    """
    raise NotImplementedError("Image loading not yet implemented")


def save_image(img: np.ndarray, path: str) -> None:
    """
    Save an image array to file.
    
    TODO: Implement image saving using PIL, imageio, or similar.
    
    Args:
        img: Image array in (H, W, C) format
        path: Output file path
    """
    raise NotImplementedError("Image saving not yet implemented")


def ensure_rgb(img: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Ensure image has 3 channels (RGB).
    
    If grayscale (single channel), convert to RGB by repeating channels.
    
    Args:
        img: Input image, shape (H, W) or (H, W, C) or (C, H, W)
        
    Returns:
        rgb_img: Image with shape (H, W, 3) or (C, H, W) with C=3
    """
    if isinstance(img, torch.Tensor):
        if img.dim() == 2:  # (H, W)
            return img.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        elif img.dim() == 3 and img.shape[0] == 1:  # (1, H, W)
            return img.repeat(3, 1, 1)  # (3, H, W)
        elif img.dim() == 3 and img.shape[2] == 1:  # (H, W, 1)
            return img.repeat(1, 1, 3)  # (H, W, 3)
        return img
    else:  # numpy
        if img.ndim == 2:  # (H, W)
            return np.repeat(img[:, :, np.newaxis], 3, axis=2)  # (H, W, 3)
        elif img.ndim == 3 and img.shape[2] == 1:  # (H, W, 1)
            return np.repeat(img, 3, axis=2)  # (H, W, 3)
        return img

