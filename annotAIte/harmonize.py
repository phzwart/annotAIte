"""
Label harmonization functions for consistent cluster mapping across images.
"""

import numpy as np
from typing import Dict, Tuple
from scipy.optimize import linear_sum_assignment


def harmonize_labels(
    ref_centers: np.ndarray,
    new_centers: np.ndarray,
    new_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Align new_labels to ref_centers using Hungarian algorithm.
    
    We:
    - Compute pairwise L2 distances between ref_centers and new_centers
    - Solve linear_sum_assignment for minimal total distance
    - Build a mapping {new_label -> ref_label}
    - Remap new_labels via that mapping
    
    This ensures that clusters with similar RGB centroids are assigned the same
    label ID across different images, enabling consistent annotation workflows.
    
    Args:
        ref_centers: (k, 3) float32 array from reference image, cluster centroids in RGB space
        new_centers: (k, 3) float32 array from current image, cluster centroids in RGB space
        new_labels: (H, W) int32 array in [0, k-1] from current image
        
    Returns:
        harmonized_labels: (H, W) int32 array, labels remapped into ref semantic space
        mapping: dict mapping new_label_idx -> ref_label_idx
    """
    ref_centers = np.asarray(ref_centers, dtype=np.float32)
    new_centers = np.asarray(new_centers, dtype=np.float32)
    new_labels = np.asarray(new_labels, dtype=np.int32)
    
    k_ref = ref_centers.shape[0]
    k_new = new_centers.shape[0]
    
    if k_ref != k_new:
        raise ValueError(
            f"Number of clusters must match: ref has {k_ref}, new has {k_new}"
        )
    
    # Compute pairwise L2 distances: (k_ref, k_new)
    # distances[i, j] = ||ref_centers[i] - new_centers[j]||
    distances = np.zeros((k_ref, k_new), dtype=np.float32)
    for i in range(k_ref):
        for j in range(k_new):
            diff = ref_centers[i] - new_centers[j]
            distances[i, j] = np.sqrt(np.sum(diff ** 2))
    
    # Solve assignment problem using Hungarian algorithm
    # linear_sum_assignment finds the minimum cost assignment
    # Returns (row_indices, col_indices) where row_indices[i] is matched to col_indices[i]
    row_indices, col_indices = linear_sum_assignment(distances)
    
    # Build mapping: new_label -> ref_label
    # col_indices[j] is the new label index, row_indices[j] is the ref label it maps to
    mapping = {}
    for ref_idx, new_idx in zip(row_indices, col_indices):
        mapping[int(new_idx)] = int(ref_idx)
    
    # Ensure all labels in [0, k_new-1] have a mapping (should be all of them)
    # Fill any missing labels with identity mapping (shouldn't happen if k matches)
    for label in range(k_new):
        if label not in mapping:
            # If somehow a label is missing, map to itself (or closest ref)
            mapping[label] = label
    
    # Remap labels
    harmonized_labels = np.zeros_like(new_labels)
    for new_label in range(k_new):
        mask = new_labels == new_label
        harmonized_labels[mask] = mapping[new_label]
    
    return harmonized_labels, mapping

