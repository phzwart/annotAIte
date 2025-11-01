User Guide
==========

Core Concepts
-------------

False-Color Mapping
~~~~~~~~~~~~~~~~~~~

The false-color mapping process converts image patches into a 3D embedding space using UMAP, then normalizes to RGB for visualization. This creates a visual representation where similar patches appear with similar colors.

Clustering
~~~~~~~~~~

MiniBatchKMeans clustering is applied to the false-color representation, grouping pixels with similar false-color values. This produces a segmentation map.

Label Harmonization
~~~~~~~~~~~~~~~~~~~

The Hungarian algorithm aligns cluster labels across different images based on cluster centroids in RGB space. This ensures consistent labeling for annotation workflows.

Multi-Scale Analysis
~~~~~~~~~~~~~~~~~~~~

Run multiple parameter configurations and stack their false-color representations for comprehensive analysis.

Basic Usage
-----------

Single Scale Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from annotAIte import run_falsecolor_pipeline

   result = run_falsecolor_pipeline(
       img=img,
       patch_size=7,
       stride=5,
       k=10,
       gaussian_sigma=1.0,
   )

Parameters
~~~~~~~~~~

* ``patch_size``: Size of square patches (typically 5, 7, or 11)
* ``stride``: Step between patches (controls overlap)
* ``k``: Number of clusters
* ``gaussian_sigma``: Blur strength
* ``umap_max_patches``: Max patches for UMAP fitting (speeds up on large images)

Multi-Scale Analysis
--------------------

Parameter Grid
~~~~~~~~~~~~~~

.. code-block:: python

   from annotAIte import create_parameter_grid

   configs = create_parameter_grid(
       patch_sizes=[5, 7, 11],
       gaussian_sigmas=[0.5, 1.0, 1.5],
   )

Running Multi-Scale
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from annotAIte import run_multi_scale_falsecolor

   result = run_multi_scale_falsecolor(
       img=img,
       parameter_configs=configs,
       final_k=10,
   )

Harmonization
-------------

Across Sessions
~~~~~~~~~~~~~~~

.. code-block:: python

   # First run - establish reference
   result1 = run_falsecolor_pipeline(img1, k=10)

   # Subsequent runs - harmonize
   result2 = run_falsecolor_pipeline(
       img2,
       k=10,
       ref_centers=result1.centers,
   )

In Napari Plugin
~~~~~~~~~~~~~~~~

Enable "harmonize_to_session_palette" to automatically harmonize across all runs.

Visualization
-------------

Using Matplotlib
~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt

   # Display false-color image
   plt.imshow(result.falsecolor_img)
   plt.show()

   # Display cluster labels
   plt.imshow(result.cluster_labels, cmap='tab20')
   plt.colorbar()
   plt.show()

Using Napari
~~~~~~~~~~~~

The napari plugin provides interactive visualization and parameter tuning.

