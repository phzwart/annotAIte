"""
Tests for multi-scale false-color pipeline.
"""

import numpy as np
import pytest
from annotAIte.multi_scale import (
    run_multi_scale_falsecolor,
    create_parameter_grid,
    MultiScaleResult,
)


def test_create_parameter_grid():
    """Test parameter grid creation."""
    patch_sizes = [5, 7]
    gaussian_sigmas = [0.5, 1.0]
    
    configs = create_parameter_grid(patch_sizes, gaussian_sigmas)
    
    assert len(configs) == 4, f"Expected 4 configs (2x2), got {len(configs)}"
    assert all("patch_size" in c and "gaussian_sigma" in c for c in configs)
    
    # Check all combinations are present
    patch_values = [c["patch_size"] for c in configs]
    sigma_values = [c["gaussian_sigma"] for c in configs]
    
    assert set(patch_values) == set(patch_sizes)
    assert set(sigma_values) == set(gaussian_sigmas)


def test_multi_scale_basic():
    """Test basic multi-scale pipeline execution."""
    # Create a small synthetic RGB test image
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Add gradient patterns
    for i in range(H):
        for j in range(W):
            img[i, j, 0] = int(255 * (i / H))
            img[i, j, 1] = int(255 * (j / W))
            img[i, j, 2] = int(255 * ((i + j) / (H + W)))
    
    # Add noise
    noise = np.random.randint(0, 50, size=(H, W, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Create parameter configurations
    configs = [
        {"patch_size": 5, "gaussian_sigma": 0.5},
        {"patch_size": 7, "gaussian_sigma": 1.0},
    ]
    
    # Run multi-scale pipeline
    result = run_multi_scale_falsecolor(
        img=img,
        parameter_configs=configs,
        final_k=5,
        umap_max_patches=5000,  # Small for speed
    )
    
    # Check results structure
    assert isinstance(result, MultiScaleResult)
    assert len(result.individual_results) == 2
    assert result.stacked_falsecolor.shape == (H, W, 6)  # 2 configs * 3 channels
    assert result.final_labels.shape == (H, W)
    assert result.final_centers.shape == (5, 6)  # k=5, 6 channels
    assert len(result.parameter_configs) == 2
    
    # Check individual results
    for ind_result in result.individual_results:
        assert isinstance(ind_result, type(result.individual_results[0]))  # AnnotAIteResult type
        assert ind_result.falsecolor_img.shape == (H, W, 3)
        assert ind_result.cluster_labels.shape == (H, W)
    
    # Check final labels are in valid range
    assert np.all(result.final_labels >= 0)
    assert np.all(result.final_labels < 5)


if __name__ == "__main__":
    test_create_parameter_grid()
    test_multi_scale_basic()
    print("All multi-scale tests passed!")

