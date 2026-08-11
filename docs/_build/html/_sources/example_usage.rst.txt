Example Usage
=============

The `example/` directory contains the main entry points for training, validation, and testing.

Running training
----------------

Start rule-based training:

.. code-block:: bash

   python example/main.py --config example/train_rule_based.yaml --log_filename example/rule_based_run

Start phase-field training:

.. code-block:: bash

   python example/main.py --config example/train_phase_field.yaml --log_filename example/phase_field_run

Start mixed training:

.. code-block:: bash

   python example/main.py --config example/train_mixed.yaml --log_filename example/mixed_run

Training outputs
----------------

Outputs and logs are saved under the directory specified by `model_params.path_working_directory` in the config.
Typical outputs include:

- `lightning_logs/` for TensorBoard and checkpoint history
- `logs/` for CSV training and validation metrics
- text log files with summaries and configuration details

Testing
-------

Run a single phase-field test config:

.. code-block:: bash

   python example/main.py --config example/test_phase_field.yaml --log_filename example/test_run

Run the phase-field sweep helper:

.. code-block:: bash

   python example/run_all_phase_tests.py

The helper script writes results to `example/test_runs/<timestamp>/`.

Select specific tests by pattern:

.. code-block:: bash

   python example/run_all_phase_tests.py --pattern "lowDensity|highDensity"

Additional options:

.. code-block:: bash

   python example/run_all_phase_tests.py --dry-run
   python example/run_all_phase_tests.py --limit 5
   python example/run_all_phase_tests.py --stop-on-error
   python example/run_all_phase_tests.py --test-bc vertical
