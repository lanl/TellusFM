Preprocessing and Data Preparation
=================================

TellusFM uses HDF5 external link tables to organize simulation data without duplicating large files.
The preprocessing pipeline includes:

- Embedding creation for fracture and material descriptions
- External link table generation for phase-field and rule-based datasets

Phase-field link tables
-----------------------

The phase-field preprocessing scripts are located in `preprocessing/src_link_tables/`.

Create phase-field training and validation link tables:

.. code-block:: bash

   cd preprocessing/src_link_tables
   python create_links_phase_train_val.py

This generates:

- `data/phase_field/external_links/train_link.h5`
- `data/phase_field/external_links/validation_link.h5`

Create phase-field test set link tables:

.. code-block:: bash

   cd preprocessing/src_link_tables
   python create_links_phase_testset.py

The test-set script scans `data/phase_field/test_set/PHASE-FIELD/` and creates one HDF5 link table per test directory.

Embedding generation
--------------------

Embedding scripts are available under `preprocessing/src_embeddings/`.
See `preprocessing/src_embeddings/README.md` for details on generating embeddings and storing the output.

Workflow summary
----------------

1. Clone and install the repository.
2. Download datasets into `data/`.
3. Generate embeddings if required.
4. Generate external link tables.
5. Run model training or testing with the example configs.
