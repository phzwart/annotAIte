"""
Napari widget for interactive false-color mapping and clustering.
"""

from magicgui import magicgui
from napari.types import ImageData
from typing import Optional
import numpy as np

# Import from core package
from annotAIte.pipeline import run_falsecolor_pipeline
from annotAIte.utils import make_label_colormap
from .palette_io import save_palette, load_palette

# Global session cache for reference palette
_REF_CENTERS_CACHE: Optional[np.ndarray] = None


@magicgui(
    call_button="Generate False-Color",
    patch_size={"min": 3, "max": 31, "step": 2},
    k={"min": 2, "max": 50, "step": 1},
    blur_sigma={"min": 0.0, "max": 5.0, "step": 0.1},
    umap_max_patches={"min": 1000, "max": 100000, "step": 1000},
)
def falsecolor_widget(
    viewer: "napari.Viewer",
    image: ImageData,
    patch_size: int = 7,
    k: int = 10,
    blur_sigma: float = 1.0,
    harmonize_to_session_palette: bool = True,
    umap_max_patches: Optional[int] = 10000,
) -> None:
    """
    Run the falsecolor pipeline on the selected image layer,
    add result layers to napari.
    
    Args:
        viewer: Napari viewer instance
        image: Image data from napari layer (H, W) or (H, W, C) numpy array
        patch_size: Size of square patches for embedding (odd numbers recommended)
        k: Number of clusters for segmentation
        blur_sigma: Standard deviation for Gaussian blur preprocessing
        harmonize_to_session_palette: If True, harmonize labels to match previous runs
        umap_max_patches: Max patches for UMAP fitting (None=use all). Lower values speed up UMAP.
    """
    global _REF_CENTERS_CACHE
    
    # Run pipeline
    result = run_falsecolor_pipeline(
        img=image,
        patch_size=patch_size,
        gaussian_sigma=blur_sigma,
        k=k,
        ref_centers=_REF_CENTERS_CACHE if harmonize_to_session_palette else None,
        umap_max_patches=umap_max_patches,
    )
    
    # Cache centers after first call if harmonization is enabled
    if harmonize_to_session_palette and _REF_CENTERS_CACHE is None:
        _REF_CENTERS_CACHE = result.centers.copy()
    
    # Add the reconstructed false-color RGB image
    rgb_uint8 = (result.falsecolor_img * 255).astype(np.uint8)
    viewer.add_image(
        rgb_uint8,
        name=f"falsecolor_p{patch_size}_k{k}",
        rgb=True,
        blending="additive",
    )
    
    # Add the cluster label map with deterministic colors
    color_map = make_label_colormap(result.centers.shape[0])
    viewer.add_labels(
        result.cluster_labels.astype(np.int32),
        name=f"clusters_p{patch_size}_k{k}",
        color=color_map,
    )


def save_session_palette(path: str) -> None:
    """
    Save the current session palette (reference centers) to disk.
    
    This can be called programmatically or via a future button in the widget.
    
    Args:
        path: Output file path (.npy extension recommended)
    """
    global _REF_CENTERS_CACHE
    if _REF_CENTERS_CACHE is None:
        raise ValueError("No session palette available to save. Run the pipeline first.")
    save_palette(path, _REF_CENTERS_CACHE)


def load_session_palette(path: str) -> None:
    """
    Load a reference palette from disk and set it as the session palette.
    
    This can be called programmatically or via a future button in the widget.
    
    Args:
        path: Input file path (.npy file)
    """
    global _REF_CENTERS_CACHE
    centers = load_palette(path)
    if centers is not None:
        _REF_CENTERS_CACHE = centers
        print(f"Loaded palette with {centers.shape[0]} clusters from {path}")
    else:
        raise ValueError(f"Failed to load palette from {path}")

