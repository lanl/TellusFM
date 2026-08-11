Installation
============

This section describes how to install TellusFM and its dependencies.

Creating a Python environment
-----------------------------

Option A: using `venv` (built-in Python)

.. code-block:: bash

   python -m venv tellusfm-env
   source tellusfm-env/bin/activate

Option B: using `conda`

.. code-block:: bash

   conda create -n tellusfm-env python=3.12
   conda activate tellusfm-env

Option C: using `uv`

.. code-block:: bash

   uv venv tellusfm-env
   source tellusfm-env/bin/activate

Installing dependencies
-----------------------

After activating the environment, install required packages and the TellusFM package in editable mode:

.. code-block:: bash

   pip install -r requirements.txt
   pip install -e .

For `uv` users:

.. code-block:: bash

   uv pip install -r requirements.txt
   uv pip install -e .

Validating your setup
---------------------

Validate the default example configs before running training or tests:

.. code-block:: bash

   python example/validate_config.py --show-paths

This checks that all required sections and relative paths are resolved properly.
