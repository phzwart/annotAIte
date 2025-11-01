"""
Tests for the main false-color pipeline.
"""

import numpy as np
import pytest
from annotAIte.pipeline import run_falsecolor_pipeline


def test_pipeline_basic():
    """Test basic pipeline execution with synthetic image."""
    # Create a small synthetic RGB test image (64x64 gradient + noise)
    H, W = 64, 64
    img = np.zeros((H, W, 3), dtype=np.uint8)
    
    # Add gradient patterns
    for i in range(H):
        for j in range(W):
            img[i, j, 0] = int(255 * (i / H))  # Red gradient
            img[i, j, 1] = int(255 * (j / W))  # Green gradient
            img[i, j, 2] = int(255 * ((i + j) / (H + W)))  # Blue gradient
    
    # Add some noise
    noise = np.random.randint(0, 50, size=(H, W, 3), dtype=np.uint8)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Run pipeline with small parameters for testing
    result = run_falsecolor_pipeline(
        img=img,
        patch_size=5,
        k=3,
        gaussian_sigma=0.5,
    )
    
    # Assert output shapes
    assert result.falsecolor_img.shape == (H, W, 3), \
        f"Expected falsecolor shape ({H}, {W}, 3), got {result.falsecolor_img.shape}"
    
    assert result.cluster_labels.shape == (H, W), \
        f"Expected labels shape ({H}, {W}), got {result.cluster_labels.shape}"
    
    # Assert values are in expected ranges
    assert np.all(result.falsecolor_img >= 0) and np.all(result.falsecolor_img <= 1), \
        "False-color image values must be in [0, 1]"
    
    assert np.all(result.cluster_labels >= 0) and np.all(result.cluster_labels < 3), \
        f"Cluster labels must be in [0, {3-1}], got range [{result.cluster_labels.min()}, {result.cluster_labels.max()}]"
    
    # Assert centers shape
    assert result.centers.shape == (3, 3), \
        f"Expected centers shape (3, 3), got {result.centers.shape}"


def test_pipeline_with_harmonization():
    """Test pipeline with reference centers for harmonization."""
    H, W = 64, 64
    img = np.random.randint(0, 255, size=(H, W, 3), dtype=np.uint8)
    
    # First run to get reference centers
    result1 = run_falsecolor_pipeline(
        img=img,
        patch_size=5,
        k=3,
    )
    
    # Second run with harmonization
    result2 = run_falsecolor_pipeline(
        img=img,
        patch_size=5,
        k=3,
        ref_centers=result1.centers,
    )
    
    # Check that harmonization mapping exists
    assert result2.mapping is not None, "Harmonization mapping should exist when ref_centers provided"
    assert isinstance(result2.mapping, dict), "Mapping should be a dictionary"
    
    # Labels should still be in valid range
    assert np.all(result2.cluster_labels >= 0) and np.all(result2.cluster_labels < 3)


if __name__ == "__main__":
    test_pipeline_basic()
    test_pipeline_with_harmonization()
    print("All pipeline tests passed!")

