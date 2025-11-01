"""
High-level pipeline function for end-to-end false-color mapping and clustering.
"""

import numpy as np
import torch
from typing import Optional

from .types import AnnotAIteResult
from .preprocess import preprocess_image
from .patches import extract_patches
from .embed import embed_umap, normalize_embedding_to_rgb
from .reconstruct import reconstruct_falsecolor
from .cluster import cluster_image
from .harmonize import harmonize_labels


def run_falsecolor_pipeline(
    img: np.ndarray,
    patch_size: int = 7,
    stride: Optional[int] = None,
    gaussian_sigma: float = 1.0,
    k: int = 10,
    ref_centers: Optional[np.ndarray] = None,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_random_state: int = 42,
    umap_max_patches: Optional[int] = 10000,
    device: str = "cpu",
) -> AnnotAIteResult:
    """
    High-level, single-call pipeline for false-color mapping and clustering.
    
    Steps:
    1. Preprocess image (blur, convert to torch, shape to (1,C,H,W))
    2. Extract overlapping patches with qlty
    3. UMAP-embed patches -> (N,3)
    4. Normalize per-dimension to [0,1] to get per-patch RGB triplets
    5. Reconstruct full false-color RGB image using quilt stitch/overlap
    6. Run MiniBatchKMeans on reconstructed RGB to get label map and centers
    7. If ref_centers is provided:
           - harmonize labels so clusters match ref palette
       Else:
           - keep labels as-is
    8. Return AnnotAIteResult(falsecolor_img, labels, centers, mapping)
    
    Args:
        img: Input image as numpy array (H,W,3) RGB assumed, uint8 or float
        patch_size: Patch edge length in pixels (typ 5,7,11)
        stride: Overlap step. If None, stride = patch_size
        gaussian_sigma: Blur sigma before patching/embedding
        k: Cluster count for MiniBatchKMeans
        ref_centers: (k,3) array from a previous reference run; if given, harmonize to it
        umap_neighbors: Number of neighbors for UMAP local structure preservation
        umap_min_dist: Minimum distance parameter for UMAP embedding
        umap_random_state: Random seed for UMAP and clustering
        umap_max_patches: Maximum number of patches to use for UMAP fitting.
                          If None, uses all patches. Default 10000 speeds up UMAP
                          on large images by fitting on a subset then transforming all.
        device: "cpu" or "cuda" for tensor ops (currently all ops are on CPU)
        
    Returns:
        AnnotAIteResult containing:
            - falsecolor_img: (H,W,3) float32 in [0,1]
            - cluster_labels: (H,W) int32 in [0,k-1]
            - centers: (k,3) float32 cluster centroids
            - mapping: dict[new_label -> ref_label] if harmonized, else None
    """
    # Step 1: Preprocess image
    img_tensor = preprocess_image(img, gaussian_sigma=gaussian_sigma)
    # Note: device parameter is for future GPU support, currently operations are CPU
    
    # Step 2: Extract patches
    patches, quilt_meta = extract_patches(
        img_tensor,
        patch_size=patch_size,
        stride=stride,
    )
    
    # Step 3: UMAP embedding
    embedding = embed_umap(
        patches,
        n_components=3,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=umap_random_state,
        max_patches=umap_max_patches,
    )
    
    # Step 4: Normalize to RGB
    colors = normalize_embedding_to_rgb(embedding)
    
    # Step 5: Reconstruct false-color image
    falsecolor_img = reconstruct_falsecolor(colors, patches, quilt_meta)
    
    # Step 6: Cluster pixels
    cluster_labels, centers = cluster_image(
        falsecolor_img,
        k=k,
        random_state=umap_random_state,
    )
    
    # Step 7: Harmonize if reference centers provided
    mapping = None
    if ref_centers is not None:
        ref_centers = np.asarray(ref_centers, dtype=np.float32)
        if ref_centers.shape != (k, 3):
            raise ValueError(
                f"ref_centers must have shape ({k}, 3), got {ref_centers.shape}"
            )
        cluster_labels, mapping = harmonize_labels(
            ref_centers=ref_centers,
            new_centers=centers,
            new_labels=cluster_labels,
        )
    
    # Step 8: Return result
    return AnnotAIteResult(
        falsecolor_img=falsecolor_img,
        cluster_labels=cluster_labels,
        centers=centers,
        mapping=mapping,
    )

