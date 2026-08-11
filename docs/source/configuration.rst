Configuration Reference
=======================

TellusFM uses YAML configuration files for experiments.
Supported example configs live in `example/`.

Available configs
-----------------

- `example/train_rule_based.yaml`
- `example/train_phase_field.yaml`
- `example/train_mixed.yaml`
- `example/test_phase_field.yaml`
- `example/test_rule_based.yaml`

Validating configs
------------------

Validate a config file from the repository root:

.. code-block:: bash

   python example/validate_config.py example/<config>.yaml

Skip path validation if the data is not yet downloaded:

.. code-block:: bash

   python example/validate_config.py --skip-path-check example/<config>.yaml

Show normalized paths:

.. code-block:: bash

   python example/validate_config.py --show-paths example/<config>.yaml

Top-level sections
-------------------

TellusFM YAML supports the following top-level sections:

- `model_params`
- `checkpoints`
- `datasets`
- `embeddings`
- `encoder`
- `decoder`
- `rule_based_params`

Common model params
-------------------

- `path_working_directory`
- `num_workers`
- `num_nodes`
- `devices`
- `seed`
- `batch_pixels`
- `lr`
- `accum_grads`
- `warmup_steps`
- `max_steps`
- `gradient_clip_val`
- `weight_decay`
- `sim_type`
- `sim_weights`
- `accelerator`
- `log_every_n_steps`
- `check_val_every_n_epoch`

Dataset paths
-------------

- `path_phase`
- `path_val_phase`
- `path_hoss`
- `path_val_hoss`
- `path_test`
- `unstructured_mesh_scale`

Embedding paths
---------------

- `emb_rule_based`
- `emb_phase`
- `emb_hoss`
- `num_embedding_variants`

Notes
-----

Relative paths in YAML are resolved relative to the config file location. This makes it easier to keep path references portable across environments.
