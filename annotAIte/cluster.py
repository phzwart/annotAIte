"""
Clustering functions for segmenting false-color images.
"""

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from typing import Tuple


def cluster_image(
    falsecolor_img: np.ndarray,
    k: int,
    batch_size: int = 2048,
    random_state: int = 42,
    n_init: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster pixels in the false-color image using MiniBatchKMeans.
    
    Treats each pixel's multi-channel value as a feature vector and clusters them into k groups.
    This produces a segmentation map where pixels with similar false-color values
    are assigned the same cluster label.
    
    Args:
        falsecolor_img: (H, W, C) float32 array in [0,1], false-color image with C channels.
                       Can be RGB (C=3) or multi-channel stacked representation (C>3).
        k: number of clusters
        batch_size: batch size for MiniBatchKMeans (for memory efficiency)
        random_state: random seed for reproducibility
        n_init: number of random initializations (best result is kept)
        
    Returns:
        labels: (H, W) int32 array, segmentation map with values in [0, k-1]
        centers: (k, C) float32 array, cluster centroids in multi-channel color space
    """
    falsecolor_img = np.asarray(falsecolor_img, dtype=np.float32)
    
    if falsecolor_img.ndim != 3:
        raise ValueError(
            f"Expected 3D array (H, W, C), got shape {falsecolor_img.shape}"
        )
    
    H, W, C = falsecolor_img.shape
    
    if C < 1:
        raise ValueError(f"Expected at least 1 channel, got {C}")
    
    # Reshape to (H*W, C) for clustering
    pixels = falsecolor_img.reshape(-1, C)  # (H*W, C)
    
    # Apply MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=batch_size,
        random_state=random_state,
        n_init=n_init,
        verbose=0,
    )
    
    labels_flat = kmeans.fit_predict(pixels)  # (H*W,)
    
    # Reshape labels back to image shape
    labels = labels_flat.reshape(H, W).astype(np.int32)
    
    # Get cluster centers (in multi-channel space)
    centers = kmeans.cluster_centers_.astype(np.float32)  # (k, C)
    
    # Ensure centers are in [0, 1] (they should be, but clip to be safe)
    centers = np.clip(centers, 0.0, 1.0)
    
    return labels, centers

