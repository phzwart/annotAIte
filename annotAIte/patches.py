"""
Patch extraction and management using qlty for overlapping patch generation.
"""

import torch
from typing import Tuple
from .types import QuiltMetadata


def extract_patches(
    img_tensor: torch.Tensor,
    patch_size: int,
    stride: int | None = None,
) -> Tuple[torch.Tensor, QuiltMetadata]:
    """
    Slice the image into overlapping patches using qlty (quilting).
    
    Uses qlty2D.NCYXQuilt to generate patches with overlap-aware indexing.
    
    Args:
        img_tensor: (1, C, H, W) torch.Tensor, float32 in [0,1]
        patch_size: size P of square patches (P x P)
        stride: step between patch starts. If None, stride = patch_size.
        
    Returns:
        patches: (N, C, P, P) torch.Tensor float32
        quilt_meta: QuiltMetadata storing original size, stride, and qlty object
    """
    if stride is None:
        stride = patch_size
    
    # Get image dimensions
    _, C, H, W = img_tensor.shape
    
    # Import qlty2D submodule - handle potential import errors gracefully
    try:
        import qlty.qlty2D as qlty2D
    except ImportError:
        # Try alternative import pattern
        try:
            from qlty import qlty2D
        except ImportError:
            raise ImportError(
                "qlty package with qlty2D submodule is required for patch extraction. "
                "Install with: pip install qlty"
            )
    
    # Create quilt object for patch extraction
    # qlty2D.NCYXQuilt expects: X=width, Y=height, window=(patch_size, patch_size), 
    # step=(stride, stride), border=(0,0), border_weight=0
    quilt = qlty2D.NCYXQuilt(
        X=W,
        Y=H,
        window=(patch_size, patch_size),
        step=(stride, stride),
        border=(0, 0),
        border_weight=0,
    )
    
    # Convert tensor to numpy for qlty (expects numpy arrays)
    # qlty expects 4D tensor (N, C, H, W) where N is batch size
    # Keep the batch dimension (1, C, H, W)
    img_np = img_tensor.detach().cpu().numpy()  # (1, C, H, W)
    
    # Create a dummy target tensor for unstitch_data_pair
    # qlty expects (data, target) pairs but we only have data
    dummy_target = img_np  # Same shape as data (1, C, H, W)
    
    # Extract patches using qlty
    patches_data, _ = quilt.unstitch_data_pair(img_np, dummy_target)
    
    # patches_data should be (N, C, P, P) where N is number of patches
    # Convert back to torch tensor
    patches_tensor = torch.from_numpy(patches_data).float()
    
    # Create metadata
    quilt_meta = QuiltMetadata(
        height=H,
        width=W,
        patch_size=patch_size,
        stride=stride,
        quilt_obj=quilt,
    )
    
    return patches_tensor, quilt_meta

