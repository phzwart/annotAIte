"""
Tests for patch reconstruction using qlty stitching.
"""

import numpy as np
import torch
import pytest
from annotAIte.types import QuiltMetadata
from annotAIte.reconstruct import reconstruct_falsecolor


def test_reconstruct_basic():
    """
    Test reconstruction with fake patches and colors.
    
    Creates a simple pattern where patches have known colors and verifies
    that reconstruction produces expected output dimensions.
    """
    # Skip if qlty is not available
    try:
        import qlty.qlty2D as qlty2D
    except ImportError:
        try:
            from qlty import qlty2D
        except ImportError:
            pytest.skip("qlty not available, skipping reconstruction test")
    
    H, W = 32, 32
    patch_size = 8
    stride = 4
    
    # Create a fake image tensor
    img_tensor = torch.rand(1, 3, H, W)
    
    # Create quilt object
    quilt = qlty2D.NCYXQuilt(
        X=W,
        Y=H,
        window=(patch_size, patch_size),
        step=(stride, stride),
        border=(0, 0),
        border_weight=0,
    )
    
    # Extract patches to get actual number
    # qlty expects 4D tensor (N, C, H, W), keep batch dimension
    img_np = img_tensor.detach().cpu().numpy()  # (1, C, H, W)
    patches_data, _ = quilt.unstitch_data_pair(img_np, img_np)
    N = patches_data.shape[0]
    
    # Create fake colors array - assign colors based on position
    colors = np.zeros((N, 3), dtype=np.float32)
    for i in range(N):
        # Simple pattern: vary color based on patch index
        colors[i, 0] = (i % 10) / 10.0  # Red
        colors[i, 1] = ((i * 2) % 10) / 10.0  # Green
        colors[i, 2] = ((i * 3) % 10) / 10.0  # Blue
    
    # Create metadata
    quilt_meta = QuiltMetadata(
        height=H,
        width=W,
        patch_size=patch_size,
        stride=stride,
        quilt_obj=quilt,
    )
    
    # Create fake patches tensor (just for shape reference)
    patches = torch.from_numpy(patches_data).float()
    
    # Reconstruct
    falsecolor_img = reconstruct_falsecolor(colors, patches, quilt_meta)
    
    # Assert output dimensions
    assert falsecolor_img.shape == (H, W, 3), \
        f"Expected shape ({H}, {W}, 3), got {falsecolor_img.shape}"
    
    # Assert values are in [0, 1]
    assert np.all(falsecolor_img >= 0) and np.all(falsecolor_img <= 1), \
        "False-color image values must be in [0, 1]"
    
    # Assert dtype
    assert falsecolor_img.dtype == np.float32, \
        f"Expected float32, got {falsecolor_img.dtype}"


def test_reconstruct_dimensions():
    """Test that reconstruction respects metadata dimensions."""
    try:
        import qlty.qlty2D as qlty2D
    except ImportError:
        try:
            from qlty import qlty2D
        except ImportError:
            pytest.skip("qlty not available")
    
    H, W = 64, 64
    patch_size = 16
    stride = 8
    
    img_tensor = torch.rand(1, 3, H, W)
    # qlty expects 4D tensor (N, C, H, W), keep batch dimension
    img_np = img_tensor.detach().cpu().numpy()  # (1, C, H, W)
    
    quilt = qlty2D.NCYXQuilt(
        X=W,
        Y=H,
        window=(patch_size, patch_size),
        step=(stride, stride),
        border=(0, 0),
        border_weight=0,
    )
    
    patches_data, _ = quilt.unstitch_data_pair(img_np, img_np)
    N = patches_data.shape[0]
    
    colors = np.random.rand(N, 3).astype(np.float32)
    patches = torch.from_numpy(patches_data).float()
    
    quilt_meta = QuiltMetadata(
        height=H,
        width=W,
        patch_size=patch_size,
        stride=stride,
        quilt_obj=quilt,
    )
    
    falsecolor_img = reconstruct_falsecolor(colors, patches, quilt_meta)
    
    # Check exact dimensions match metadata
    assert falsecolor_img.shape[0] == H
    assert falsecolor_img.shape[1] == W
    assert falsecolor_img.shape[2] == 3


if __name__ == "__main__":
    test_reconstruct_basic()
    test_reconstruct_dimensions()
    print("All reconstruction tests passed!")

