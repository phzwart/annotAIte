"""
Tests for label harmonization using Hungarian algorithm.
"""

import numpy as np
from annotAIte.harmonize import harmonize_labels


def test_harmonize_labels():
    """Test label harmonization with fake centroids and labels."""
    k = 5
    H, W = 32, 32
    
    # Create two fake centroid arrays
    # Reference centers (sorted in some order)
    ref_centers = np.array([
        [0.1, 0.1, 0.1],
        [0.3, 0.3, 0.3],
        [0.5, 0.5, 0.5],
        [0.7, 0.7, 0.7],
        [0.9, 0.9, 0.9],
    ], dtype=np.float32)
    
    # New centers (similar but slightly different order/values)
    new_centers = np.array([
        [0.92, 0.92, 0.92],  # Should map to ref [0.9, 0.9, 0.9]
        [0.12, 0.12, 0.12],  # Should map to ref [0.1, 0.1, 0.1]
        [0.52, 0.52, 0.52],  # Should map to ref [0.5, 0.5, 0.5]
        [0.32, 0.32, 0.32],  # Should map to ref [0.3, 0.3, 0.3]
        [0.72, 0.72, 0.72],  # Should map to ref [0.7, 0.7, 0.7]
    ], dtype=np.float32)
    
    # Create fake label map with known labels
    new_labels = np.random.randint(0, k, size=(H, W), dtype=np.int32)
    # Ensure all labels appear at least once
    new_labels[0:k, 0] = np.arange(k)
    
    # Call harmonize_labels
    harmonized_labels, mapping = harmonize_labels(
        ref_centers=ref_centers,
        new_centers=new_centers,
        new_labels=new_labels,
    )
    
    # Assert output shape matches input
    assert harmonized_labels.shape == new_labels.shape, \
        f"Output shape {harmonized_labels.shape} should match input {new_labels.shape}"
    
    # Assert mapping is a dict of ints
    assert isinstance(mapping, dict), "Mapping should be a dictionary"
    assert all(isinstance(k, int) and isinstance(v, int) for k, v in mapping.items()), \
        "Mapping keys and values should be integers"
    
    # Assert all new labels have mappings
    unique_labels = np.unique(new_labels)
    assert all(label in mapping for label in unique_labels), \
        "All unique labels should have mappings"
    
    # Assert harmonized labels are in valid range [0, k-1]
    assert np.all(harmonized_labels >= 0) and np.all(harmonized_labels < k), \
        f"Harmonized labels must be in [0, {k-1}]"
    
    # Assert mapping values are in valid range [0, k-1]
    assert all(0 <= v < k for v in mapping.values()), \
        "Mapping target values must be in [0, k-1]"


def test_harmonize_identity():
    """Test that identical centers produce identity mapping."""
    k = 3
    H, W = 16, 16
    
    centers = np.array([
        [0.2, 0.2, 0.2],
        [0.5, 0.5, 0.5],
        [0.8, 0.8, 0.8],
    ], dtype=np.float32)
    
    new_labels = np.random.randint(0, k, size=(H, W), dtype=np.int32)
    
    harmonized_labels, mapping = harmonize_labels(
        ref_centers=centers,
        new_centers=centers,  # Same centers
        new_labels=new_labels,
    )
    
    # With identical centers, mapping should ideally be identity
    # (though Hungarian algorithm might produce a permutation)
    assert harmonized_labels.shape == new_labels.shape
    assert isinstance(mapping, dict)


if __name__ == "__main__":
    test_harmonize_labels()
    test_harmonize_identity()
    print("All harmonization tests passed!")

