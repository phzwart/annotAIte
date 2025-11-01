"""
Utility functions for type conversions, seeding, and color mapping.
"""

import numpy as np
import torch
from typing import Dict, Tuple
import random


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across numpy, torch, and Python random.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def numpy_to_torch(arr: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to torch tensor.
    
    Args:
        arr: Numpy array
        
    Returns:
        tensor: Torch tensor with same dtype (mapped appropriately)
    """
    return torch.from_numpy(arr)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch tensor to numpy array.
    
    Args:
        tensor: Torch tensor
        
    Returns:
        arr: Numpy array (detached and moved to CPU if needed)
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.numpy()


def make_label_colormap(k: int) -> Dict[int, Tuple[float, float, float, float]]:
    """
    Produce a deterministic RGBA color map for integer labels 0..k-1,
    values in [0,1]. This is used in napari to force stable coloring.
    
    Uses a deterministic color generation scheme to ensure the same labels
    get the same colors across different runs.
    
    Args:
        k: Number of labels (clusters)
        
    Returns:
        color_map: Dictionary mapping label_id (int) to (r, g, b, a) tuple,
                   all values in [0, 1]
    """
    color_map = {}
    
    # Generate deterministic colors using a hash-like approach
    # or use a fixed palette for first N colors, then cycle
    # Simple approach: use HSV space and convert to RGB
    
    for label_id in range(k):
        # Use golden angle for even spacing in color space
        hue = (label_id * 0.618033988749895) % 1.0  # Golden ratio - 1
        saturation = 0.7 + (label_id % 3) * 0.1  # Vary saturation slightly
        value = 0.8 + (label_id % 2) * 0.2  # Vary brightness
        
        # Convert HSV to RGB
        from colorsys import hsv_to_rgb
        r, g, b = hsv_to_rgb(hue, saturation, value)
        a = 1.0  # Full opacity
        
        color_map[label_id] = (float(r), float(g), float(b), float(a))
    
    return color_map

