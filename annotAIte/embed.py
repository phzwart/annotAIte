"""
UMAP embedding functions for converting patches to low-dimensional representations.
"""

import numpy as np
import torch
from typing import Tuple, Optional
import umap


def embed_umap(
    patches: torch.Tensor,
    n_components: int = 3,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    max_patches: Optional[int] = None,
) -> np.ndarray:
    """
    Flatten patches and run UMAP to embed them into n_components (default=3).
    
    This reduces each patch from (C, P, P) dimensions to a single n_components-dimensional
    vector, which can then be normalized to RGB for false-color visualization.
    
    To speed up UMAP on large numbers of patches, use max_patches to subsample
    patches for fitting. The model is then used to transform all patches.
    
    Args:
        patches: (N, C, P, P) torch.Tensor float32, where N is number of patches,
                 C is channels, P is patch size
        n_components: embedding dimensionality (3 maps directly to RGB)
        n_neighbors: number of neighbors for UMAP local neighborhood graph
        min_dist: minimum distance between points in UMAP embedding space
        random_state: random seed for reproducibility
        max_patches: Optional maximum number of patches to use for fitting UMAP.
                     If None, uses all patches. If specified and N > max_patches,
                     randomly samples max_patches patches for fitting, then transforms all.
                     This can significantly speed up UMAP on large images.
        
    Returns:
        embedding: (N, n_components) float32 numpy array
    """
    # Convert to numpy if needed
    if isinstance(patches, torch.Tensor):
        patches_np = patches.detach().cpu().numpy()
    else:
        patches_np = np.asarray(patches)
    
    # Flatten patches: (N, C, P, P) -> (N, C*P*P)
    N = patches_np.shape[0]
    patches_flat = patches_np.reshape(N, -1)
    
    # Subsample patches for fitting if max_patches is specified and we have more patches
    if max_patches is not None and N > max_patches:
        # Set random seed for reproducibility
        rng = np.random.RandomState(random_state)
        
        # Randomly sample indices
        sample_indices = rng.choice(N, size=max_patches, replace=False)
        sample_indices = np.sort(sample_indices)  # Sort for cache efficiency
        
        # Fit UMAP on subsampled patches
        patches_sample = patches_flat[sample_indices]
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=min(n_neighbors, max_patches - 1),  # Ensure n_neighbors < sample size
            min_dist=min_dist,
            random_state=random_state,
            verbose=False,
        )
        
        # Fit on subset, then transform all patches
        reducer.fit(patches_sample)
        embedding = reducer.transform(patches_flat)
    else:
        # Fit and transform on all patches
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            verbose=False,
        )
        
        embedding = reducer.fit_transform(patches_flat)
    
    return embedding.astype(np.float32)


def normalize_embedding_to_rgb(
    embedding: np.ndarray,
) -> np.ndarray:
    """
    Min-max normalize each column of embedding independently to [0,1].
    
    Output rows correspond to patches, columns to (R,G,B).
    This ensures the embedding values can be used directly as RGB color values.
    
    Args:
        embedding: (N, 3) float32 array, raw UMAP embedding
        
    Returns:
        colors: (N, 3) float32 array in [0,1], normalized per-dimension
    """
    embedding = np.asarray(embedding, dtype=np.float32)
    
    # Handle edge case: all values in a dimension are the same
    min_vals = embedding.min(axis=0, keepdims=True)
    max_vals = embedding.max(axis=0, keepdims=True)
    
    # Avoid division by zero
    ranges = max_vals - min_vals
    ranges = np.where(ranges == 0, 1.0, ranges)
    
    # Min-max normalization: (x - min) / (max - min)
    normalized = (embedding - min_vals) / ranges
    
    # Clip to [0, 1] to handle any floating point issues
    normalized = np.clip(normalized, 0.0, 1.0)
    
    return normalized.astype(np.float32)

