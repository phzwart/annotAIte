"""
Core data types for the annotAIte package.
"""

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np


@dataclass
class AnnotAIteResult:
    """
    Container for the output of the false-color mapping and clustering pipeline.
    
    Attributes:
        falsecolor_img: False-color RGB image (H, W, 3), float32 in [0,1]
        cluster_labels: Cluster label map (H, W), int32 in [0, k-1]
        centers: Cluster centroids in RGB color space (k, 3), float32
        mapping: Optional mapping from new_label -> ref_label for harmonization,
                 or None if no harmonization was performed
    """
    falsecolor_img: np.ndarray        # (H, W, 3), float32 in [0,1]
    cluster_labels: np.ndarray        # (H, W), int32 label map
    centers: np.ndarray               # (k, 3), float32 cluster centroids in color space
    mapping: Optional[Dict[int, int]]  # mapping from new_label -> ref_label, or None


@dataclass
class QuiltMetadata:
    """
    Metadata for patch extraction and reconstruction using qlty.
    
    Stores information needed to reconstruct full images from overlapping patches.
    
    Attributes:
        height: Original image height
        width: Original image width
        patch_size: Size of square patches (patch_size x patch_size)
        stride: Step size between patch starts
        quilt_obj: The qlty quilt object that can be used to stitch patches back together
    """
    height: int
    width: int
    patch_size: int
    stride: int
    quilt_obj: object  # qlty quilt object for stitching

