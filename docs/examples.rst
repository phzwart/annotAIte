Examples
========

Basic Pipeline
--------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from annotAIte import run_falsecolor_pipeline

   # Create test image
   img = np.random.randint(0, 255, size=(256, 256, 3), dtype=np.uint8)

   # Run pipeline
   result = run_falsecolor_pipeline(
       img=img,
       patch_size=7,
       k=10,
   )

   # Visualize
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   axes[0].imshow(result.falsecolor_img)
   axes[0].set_title("False-Color Image")
   axes[1].imshow(result.cluster_labels, cmap='tab20')
   axes[1].set_title("Cluster Labels")
   plt.show()

Multi-Scale Analysis
--------------------

.. code-block:: python

   from annotAIte import run_multi_scale_falsecolor, create_parameter_grid

   # Create parameter configurations
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

   # Visualize individual results
   n_configs = len(configs)
   fig, axes = plt.subplots(1, n_configs, figsize=(5*n_configs, 5))
   for i, ind_result in enumerate(result.individual_results):
       axes[i].imshow(ind_result.falsecolor_img)
       axes[i].set_title(f"Config {i+1}")
   plt.show()

   # Visualize final result
   plt.figure(figsize=(10, 5))
   plt.imshow(result.final_labels, cmap='tab20')
   plt.title("Final Multi-Scale Clustering")
   plt.colorbar()
   plt.show()

Harmonization Example
---------------------

.. code-block:: python

   # First image - establish reference
   result1 = run_falsecolor_pipeline(img1, k=10, patch_size=7)

   # Second image - harmonize to first
   result2 = run_falsecolor_pipeline(
       img2,
       k=10,
       patch_size=7,
       ref_centers=result1.centers,
   )

   # Compare harmonization mapping
   print(f"Harmonization mapping: {result2.mapping}")

   # Visualize side by side
   fig, axes = plt.subplots(1, 2, figsize=(12, 5))
   axes[0].imshow(result1.cluster_labels, cmap='tab20')
   axes[0].set_title("Reference")
   axes[1].imshow(result2.cluster_labels, cmap='tab20')
   axes[1].set_title("Harmonized")
   plt.show()

Napari Plugin
-------------

.. code-block:: python

   import napari
   from napari_annotAIte import falsecolor_widget

   # Launch napari
   viewer = napari.Viewer()

   # Load image
   viewer.open("path/to/your/image.tif")

   # Add widget
   viewer.window.add_dock_widget(falsecolor_widget)

   # Adjust parameters and click "Generate False-Color"

