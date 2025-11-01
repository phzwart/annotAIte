# annotAIte

Self-supervised false-color mapping and harmonized clustering for assisting manual image annotation in microscopy and remote sensing.

## Overview

`annotAIte` is a Python package that implements a workflow for generating false-color representations of images and performing harmonized clustering. The package consists of two components:

1. **Core Library (`annotAIte`)**: A headless Python library that processes images without GUI dependencies
2. **Napari Plugin (`napari-annotAIte`)**: An interactive plugin for the napari image viewer

### Key Features

- **False-Color Mapping**: Uses UMAP to embed image patches into 3D space, normalizes to RGB, and reconstructs full-resolution false-color images
- **Overlap-Aware Stitching**: Uses qlty for intelligent patch-based reconstruction with overlap averaging
- **Clustering**: MiniBatchKMeans clustering on false-color representations for segmentation
- **Label Harmonization**: Hungarian algorithm-based matching to maintain consistent cluster labels across multiple images/sessions
- **Napari Integration**: Interactive widget for parameter tuning and visualization

## Installation

### Core Package

```bash
pip install annotAIte
```

### With Napari Plugin

```bash
pip install annotAIte[napari]
```

## Quick Start

### Command-Line / Headless Usage

```python
import numpy as np
from annotAIte import run_falsecolor_pipeline

# Load your image (H, W, 3) RGB array
img = np.random.randint(0, 255, size=(512, 512, 3), dtype=np.uint8)

# Run the pipeline
result = run_falsecolor_pipeline(
    img=img,
    patch_size=7,
    k=10,
    gaussian_sigma=1.0,
)

# Access results
falsecolor_img = result.falsecolor_img      # (H, W, 3) float32 in [0,1]
cluster_labels = result.cluster_labels      # (H, W) int32 in [0, k-1]
cluster_centers = result.centers            # (k, 3) float32 RGB centroids
harmonization_mapping = result.mapping      # dict or None
```

### Napari Plugin Usage

1. Install with napari dependencies:
   ```bash
   pip install annotAIte[napari]
   ```

2. Launch napari and load an image:
   ```python
   import napari
   viewer = napari.Viewer()
   viewer.open("path/to/your/image.tif")
   ```

3. Open the annotAIte widget:
   - The widget should appear in the napari plugin menu
   - Or access via: `viewer.window.add_dock_widget(napari_annotAIte.falsecolor_widget)`

4. Configure parameters:
   - **patch_size**: Size of square patches for embedding (typically 5, 7, or 11)
   - **k**: Number of clusters for segmentation
   - **blur_sigma**: Gaussian blur standard deviation
   - **harmonize_to_session_palette**: Enable label harmonization across runs

5. Click "Generate False-Color" to:
   - Generate a false-color RGB image layer
   - Generate a cluster labels layer with harmonized segmentation

## Architecture

### Core Modules

- `preprocess.py`: Image preprocessing (blur, normalization, format conversion)
- `patches.py`: Patch extraction using qlty
- `embed.py`: UMAP embedding and RGB normalization
- `reconstruct.py`: Overlap-aware patch reconstruction
- `cluster.py`: MiniBatchKMeans clustering
- `harmonize.py`: Hungarian algorithm-based label harmonization
- `pipeline.py`: High-level end-to-end workflow

### Data Types

The main output type is `AnnotAIteResult`:

```python
@dataclass
class AnnotAIteResult:
    falsecolor_img: np.ndarray        # (H, W, 3), float32 in [0,1]
    cluster_labels: np.ndarray       # (H, W), int32 label map
    centers: np.ndarray              # (k, 3), float32 cluster centroids
    mapping: Optional[Dict[int, int]] # harmonization mapping
```

## Parameters

### Pipeline Parameters

- `patch_size` (int): Size of square patches. Smaller values capture finer details but increase computation.
- `stride` (int, optional): Step size between patches. Defaults to `patch_size` (no overlap).
- `gaussian_sigma` (float): Standard deviation for Gaussian blur preprocessing. Higher values reduce noise but blur details.
- `k` (int): Number of clusters for segmentation. More clusters capture finer distinctions.
- `ref_centers` (np.ndarray, optional): Reference cluster centers for harmonization. Shape (k, 3).

### UMAP Parameters

- `umap_neighbors` (int): Number of neighbors for local structure preservation (default: 15)
- `umap_min_dist` (float): Minimum distance in embedding space (default: 0.1)
- `umap_random_state` (int): Random seed for reproducibility (default: 42)

## Harmonization

Label harmonization ensures that clusters with similar RGB centroids are assigned the same label ID across different images. This is crucial for consistent annotation workflows.

```python
# First run - establish reference palette
result1 = run_falsecolor_pipeline(img1, k=10)

# Subsequent runs - harmonize to reference
result2 = run_falsecolor_pipeline(
    img2,
    k=10,
    ref_centers=result1.centers,  # Use first run as reference
)

# result2.mapping contains {new_label: ref_label} dictionary
```

In napari, enable "harmonize_to_session_palette" to automatically harmonize across all runs in a session.

## Dependencies

### Core
- Python >= 3.10
- numpy
- scipy
- scikit-learn
- umap-learn
- torch
- qlty

### Napari Plugin (Optional)
- napari
- magicgui
- qtpy
- PyQt5 / PyQt6

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/annotAIte.git
cd annotAIte

# Install in development mode
pip install -e ".[napari]"

# Run tests
pytest tests/
```

## License

BSD-3-Clause License

## Citation

If you use annotAIte in your research, please cite:

```bibtex
@software{annotAIte2025,
  title = {annotAIte: Self-supervised false-color mapping and harmonized clustering},
  author = {Petrus Zwart & Aritro Dasgupta},
  year = {2025},
  url = {https://github.com/phzwart/annotAIte}
}
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

