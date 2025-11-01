"""
Configuration constants and default parameters for annotAIte.
"""

# Default pipeline parameters
DEFAULT_PATCH_SIZE = 7
DEFAULT_STRIDE = None  # Will default to patch_size if None
DEFAULT_GAUSSIAN_SIGMA = 1.0
DEFAULT_K = 10

# Default UMAP parameters
DEFAULT_UMAP_N_COMPONENTS = 3
DEFAULT_UMAP_N_NEIGHBORS = 15
DEFAULT_UMAP_MIN_DIST = 0.1
DEFAULT_UMAP_RANDOM_STATE = 42
DEFAULT_UMAP_MAX_PATCHES = 10000  # Subsample for speed if more patches than this

# Default clustering parameters
DEFAULT_CLUSTER_BATCH_SIZE = 2048
DEFAULT_CLUSTER_N_INIT = 10
DEFAULT_CLUSTER_RANDOM_STATE = 42

