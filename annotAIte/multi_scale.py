"""
Multi-scale false-color mapping and clustering workflow.

Runs the false-color pipeline with multiple parameter combinations,
stacks the results, and performs clustering on the combined representation.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from .pipeline import run_falsecolor_pipeline
from .types import AnnotAIteResult
from .cluster import cluster_image


@dataclass
class MultiScaleResult:
    """
    Results from multi-scale false-color pipeline.
    
    Attributes:
        individual_results: List of AnnotAIteResult for each parameter combination
        stacked_falsecolor: Stacked false-color representation (H, W, N*3) where
                           N is number of parameter combinations
        final_labels: Final cluster labels from k-means on stacked representation (H, W)
        final_centers: Cluster centers in stacked false-color space (k, N*3)
        parameter_configs: List of parameter dictionaries used for each run
    """
    individual_results: List[AnnotAIteResult]
    stacked_falsecolor: np.ndarray  # (H, W, N*3)
    final_labels: np.ndarray  # (H, W)
    final_centers: np.ndarray  # (k, N*3)
    parameter_configs: List[Dict]


def run_multi_scale_falsecolor(
    img: np.ndarray,
    parameter_configs: List[Dict],
    final_k: int = 10,
    ref_centers: Optional[np.ndarray] = None,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    umap_random_state: int = 42,
    umap_max_patches: Optional[int] = 10000,
    cluster_random_state: int = 42,
) -> MultiScaleResult:
    """
    Run false-color pipeline with multiple parameter combinations and cluster the stacked results.
    
    This workflow:
    1. Runs the false-color pipeline for each parameter configuration
    2. Stacks all false-color images together (each adds 3 RGB channels)
    3. Runs k-means clustering on the stacked false-color representation
    4. Returns individual results and the final clustered output
    
    Args:
        img: Input image as numpy array (H,W,3) RGB assumed, uint8 or float
        parameter_configs: List of dictionaries, each containing parameters for one run.
                          Each dict can contain:
                          - patch_size (int): Patch size for this run
                          - stride (int, optional): Stride for this run
                          - gaussian_sigma (float): Blur sigma for this run
                          Example: [{"patch_size": 5, "gaussian_sigma": 0.5},
                                    {"patch_size": 7, "gaussian_sigma": 1.0},
                                    {"patch_size": 11, "gaussian_sigma": 1.5}]
        final_k: Number of clusters for final k-means on stacked representation
        ref_centers: (k, N*3) array for harmonization (where N is number of configs).
                     If provided, harmonizes final labels.
        umap_neighbors: Number of neighbors for UMAP (used in each individual run)
        umap_min_dist: Minimum distance for UMAP (used in each individual run)
        umap_random_state: Random seed for UMAP and reproducibility
        umap_max_patches: Maximum patches for UMAP fitting (speeds up each run)
        cluster_random_state: Random seed for final k-means clustering
        
    Returns:
        MultiScaleResult containing:
            - individual_results: List of AnnotAIteResult for each parameter combination
            - stacked_falsecolor: (H, W, N*3) stacked false-color representation
            - final_labels: (H, W) final cluster labels
            - final_centers: (k, N*3) cluster centers in stacked space
            - parameter_configs: Copy of input parameter_configs
    """
    if not parameter_configs:
        raise ValueError("parameter_configs must contain at least one configuration")
    
    # Run pipeline for each parameter configuration
    individual_results = []
    falsecolor_images = []
    
    for config in parameter_configs:
        # Extract parameters with defaults
        patch_size = config.get("patch_size", 7)
        stride = config.get("stride", None)
        gaussian_sigma = config.get("gaussian_sigma", 1.0)
        
        # Run false-color pipeline for this configuration
        result = run_falsecolor_pipeline(
            img=img,
            patch_size=patch_size,
            stride=stride,
            gaussian_sigma=gaussian_sigma,
            k=final_k,  # Use final_k for individual clustering (could also be separate)
            ref_centers=None,  # No harmonization on individual runs
            umap_neighbors=umap_neighbors,
            umap_min_dist=umap_min_dist,
            umap_random_state=umap_random_state,
            umap_max_patches=umap_max_patches,
        )
        
        individual_results.append(result)
        falsecolor_images.append(result.falsecolor_img)
    
    # Stack all false-color images along channel dimension
    # Each falsecolor_img is (H, W, 3), stacking gives (H, W, N*3)
    stacked_falsecolor = np.concatenate(falsecolor_images, axis=2)
    
    # Run k-means clustering on the stacked false-color representation
    final_labels, final_centers = cluster_image(
        stacked_falsecolor,
        k=final_k,
        random_state=cluster_random_state,
    )
    
    # Optionally harmonize final labels if reference centers provided
    if ref_centers is not None:
        from .harmonize import harmonize_labels
        ref_centers = np.asarray(ref_centers, dtype=np.float32)
        if ref_centers.shape != final_centers.shape:
            raise ValueError(
                f"ref_centers shape {ref_centers.shape} must match "
                f"final_centers shape {final_centers.shape}"
            )
        final_labels, _ = harmonize_labels(
            ref_centers=ref_centers,
            new_centers=final_centers,
            new_labels=final_labels,
        )
    
    return MultiScaleResult(
        individual_results=individual_results,
        stacked_falsecolor=stacked_falsecolor,
        final_labels=final_labels,
        final_centers=final_centers,
        parameter_configs=parameter_configs.copy(),
    )


def create_parameter_grid(
    patch_sizes: List[int],
    gaussian_sigmas: List[float],
    strides: Optional[List[Optional[int]]] = None,
) -> List[Dict]:
    """
    Create a grid of parameter configurations for multi-scale analysis.
    
    Args:
        patch_sizes: List of patch sizes to test
        gaussian_sigmas: List of Gaussian blur sigmas to test
        strides: Optional list of stride values. If None, uses None (patch_size) for all.
                 Length should match patch_sizes if provided.
        
    Returns:
        List of parameter dictionaries for use with run_multi_scale_falsecolor
        
    Example:
        configs = create_parameter_grid(
            patch_sizes=[5, 7, 11],
            gaussian_sigmas=[0.5, 1.0, 1.5],
        )
        # Creates 9 combinations (3x3)
    """
    configs = []
    
    for patch_size in patch_sizes:
        for sigma in gaussian_sigmas:
            config = {
                "patch_size": patch_size,
                "gaussian_sigma": sigma,
            }
            
            # Add stride if provided
            if strides is not None:
                if len(strides) == len(patch_sizes):
                    # Match stride to patch_size by index
                    idx = patch_sizes.index(patch_size)
                    config["stride"] = strides[idx]
                else:
                    raise ValueError(
                        f"strides length {len(strides)} must match patch_sizes length {len(patch_sizes)}"
                    )
            else:
                # Default: no explicit stride (will use patch_size)
                config["stride"] = None
            
            configs.append(config)
    
    return configs

