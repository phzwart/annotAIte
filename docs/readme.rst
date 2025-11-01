Overview
========

annotAIte is a Python package that implements a workflow for generating false-color representations of images and performing harmonized clustering. The package consists of two components:

1. **Core Library (`annotAIte`)**: A headless Python library that processes images without GUI dependencies
2. **Napari Plugin (`napari-annotAIte`)**: An interactive plugin for the napari image viewer

Key Features
------------

* **False-Color Mapping**: Uses UMAP to embed image patches into 3D space, normalizes to RGB, and reconstructs full-resolution false-color images
* **Overlap-Aware Stitching**: Uses qlty for intelligent patch-based reconstruction with overlap averaging
* **Clustering**: MiniBatchKMeans clustering on false-color representations for segmentation
* **Label Harmonization**: Hungarian algorithm-based matching to maintain consistent cluster labels across multiple images/sessions
* **Napari Integration**: Interactive widget for parameter tuning and visualization
* **Multi-Scale Analysis**: Run multiple parameter configurations and cluster the stacked results

Installation
------------

Core Package
~~~~~~~~~~~~

.. code-block:: bash

   pip install annotAIte

With Napari Plugin
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install annotAIte[napari]

Quick Start
-----------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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

Multi-Scale Workflow
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from annotAIte import run_multi_scale_falsecolor, create_parameter_grid

   # Create parameter grid
   configs = create_parameter_grid(
       patch_sizes=[5, 7, 11],
       gaussian_sigmas=[0.5, 1.0, 1.5],
   )

   # Run multi-scale pipeline
   result = run_multi_scale_falsecolor(
       img=img,
       parameter_configs=configs,
       final_k=10,
   )

   # Access results
   stacked_falsecolor = result.stacked_falsecolor  # (H, W, N*3)
   final_labels = result.final_labels              # (H, W)
   
   # Access individual results
   for ind_result in result.individual_results:
       falsecolor = ind_result.falsecolor_img
       labels = ind_result.cluster_labels

