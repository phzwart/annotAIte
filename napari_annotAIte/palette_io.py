"""
Import/export utilities for cluster palette (centers) persistence.
"""

import numpy as np
from typing import Optional


def save_palette(path: str, centers: np.ndarray) -> None:
    """
    Save reference cluster centers to .npy so sessions/annotators
    can share a harmonized palette.
    
    Args:
        path: Output file path (should have .npy extension)
        centers: (k, 3) float32 array of cluster centroids in RGB space
    """
    centers = np.asarray(centers, dtype=np.float32)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError(f"Expected (k, 3) array, got shape {centers.shape}")
    np.save(path, centers)


def load_palette(path: str) -> Optional[np.ndarray]:
    """
    Load reference cluster centers from .npy.
    
    Args:
        path: Input file path (.npy file)
        
    Returns:
        centers: (k, 3) float32 array of cluster centroids, or None on failure
    """
    try:
        centers = np.load(path)
        centers = np.asarray(centers, dtype=np.float32)
        if centers.ndim != 2 or centers.shape[1] != 3:
            raise ValueError(f"Loaded array has invalid shape: {centers.shape}")
        return centers
    except Exception as e:
        print(f"Failed to load palette from {path}: {e}")
        return None

