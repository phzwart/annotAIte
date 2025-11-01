Installation
============

Requirements
------------

* Python >= 3.10
* numpy
* scipy
* scikit-learn
* umap-learn
* torch
* qlty

Optional Dependencies
---------------------

* napari (for interactive widget)
* magicgui
* qtpy
* PyQt5 / PyQt6

Installation Methods
--------------------

Using pip
~~~~~~~~~

Install the core package:

.. code-block:: bash

   pip install annotAIte

Install with napari support:

.. code-block:: bash

   pip install annotAIte[napari]

Install from source
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/phzwart/annotAIte.git
   cd annotAIte
   pip install -e ".[napari]"

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/phzwart/annotAIte.git
   cd annotAIte
   pip install -e ".[napari]"
   pip install -e .[dev]  # Install development dependencies

Verification
------------

After installation, verify the installation:

.. code-block:: python

   import annotAIte
   print(annotAIte.__version__)

