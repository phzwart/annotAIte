"""
Reconstruction functions for combining patches back into full-resolution images.
"""

import numpy as np
import torch
from .types import QuiltMetadata


def reconstruct_falsecolor(
    colors: np.ndarray,
    patches: torch.Tensor,
    quilt_meta: QuiltMetadata,
) -> np.ndarray:
    """
    Reconstruct a full-resolution RGB false-color image from per-patch colors.
    
    Steps:
    - For each patch i, take colors[i] = [r,g,b] and broadcast to (3, P, P)
    - Create a tensor shaped like patches but filled with those RGB values
    - Call quilt_meta.quilt_obj.stitch(...) to combine patches back into (1,C,H,W)
      (qlty should average overlaps at boundaries)
    - Permute to HxWx3, ensure float32 in [0,1]
    
    Args:
        colors: (N, 3) float32, normalized RGB per patch
        patches: (N, C, P, P) torch.Tensor, just for shape reference (C, P, P)
        quilt_meta: QuiltMetadata (contains original H, W and quilt_obj)
        
    Returns:
        falsecolor_img: (H, W, 3) float32 in [0,1]
    """
    colors = np.asarray(colors, dtype=np.float32)
    N, _ = colors.shape  # N patches, 3 RGB channels
    
    # Get patch shape from reference patches tensor
    if isinstance(patches, torch.Tensor):
        _, C_ref, P, P_patch = patches.shape
    else:
        _, C_ref, P, P_patch = patches.shape
    
    if P != P_patch:
        raise ValueError(f"Expected square patches, got {P}x{P_patch}")
    
    # Create RGB patches by broadcasting colors to patch size
    # Each patch i gets RGB values colors[i] broadcasted to (3, P, P)
    rgb_patches_list = []
    for i in range(N):
        r, g, b = colors[i]
        # Create (3, P, P) patch with constant RGB values
        rgb_patch = np.zeros((3, P, P), dtype=np.float32)
        rgb_patch[0, :, :] = r  # R channel
        rgb_patch[1, :, :] = g  # G channel
        rgb_patch[2, :, :] = b  # B channel
        rgb_patches_list.append(rgb_patch)
    
    # Stack into (N, 3, P, P)
    rgb_patches = np.stack(rgb_patches_list, axis=0)
    
    # Convert to torch tensor - qlty.stitch expects torch tensors
    rgb_patches_tensor = torch.from_numpy(rgb_patches).float()
    
    # Use qlty quilt object to stitch patches back together
    # quilt.stitch expects torch tensor in the same format as unstitch produced
    # The stitch method should handle overlap averaging automatically
    try:
        stitched_result = quilt_meta.quilt_obj.stitch(rgb_patches_tensor)
    except Exception as e:
        raise RuntimeError(
            f"Failed to stitch patches using qlty: {e}. "
            f"Ensure quilt object is valid and patches match quilt configuration."
        ) from e
    
    # Handle return value - may be a single tensor or a tuple
    if isinstance(stitched_result, tuple):
        # If tuple, take the first element (main stitched result)
        stitched = stitched_result[0]
    else:
        stitched = stitched_result
    
    # stitched should be a torch tensor: (3, H, W) or (1, 3, H, W)
    # Convert to numpy for processing
    if isinstance(stitched, torch.Tensor):
        stitched = stitched.detach().cpu().numpy()
    
    if stitched.ndim == 4:
        stitched = stitched.squeeze(0)  # Remove batch dimension if present
    
    # Permute from (C, H, W) to (H, W, C)
    if stitched.shape[0] == 3:
        falsecolor_img = np.transpose(stitched, (1, 2, 0))  # (H, W, 3)
    else:
        raise ValueError(f"Expected 3-channel RGB, got {stitched.shape[0]} channels")
    
    # Ensure values are in [0, 1] and float32
    falsecolor_img = np.clip(falsecolor_img, 0.0, 1.0).astype(np.float32)
    
    # Verify output dimensions match metadata
    H_expected, W_expected = quilt_meta.height, quilt_meta.width
    H_actual, W_actual = falsecolor_img.shape[:2]
    
    if H_actual != H_expected or W_actual != W_expected:
        raise ValueError(
            f"Reconstructed image size mismatch: expected ({H_expected}, {W_expected}), "
            f"got ({H_actual}, {W_actual})"
        )
    
    return falsecolor_img

