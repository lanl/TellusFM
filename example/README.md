# TellusFM Example YAML Options

This document describes the YAML options supported by the example configs in `example/`, based on `tellusfm/read_input_yaml_file.py`.

## Example YAML files present in `example/`

- `train_rule_based.yaml`
- `train_phase_field.yaml`
- `train_mixed.yaml`
- `test_phase_field.yaml`
- `test_rule_based.yaml`

## Current example file usage

- `example/main.py` loads a YAML via `tfm.read_yaml_config(args.config)` and validates it.
- `example/validate_config.py` validates config structure and path requirements. Its default example configs are:
  - `example/train_rule_based.yaml`
  - `example/train_phase_field.yaml`
- `example/run_all_phase_tests.py` defaults to `example/test_phase_field.yaml` as its base config.

## How to use and validate YAML configs

### Validate a config file

Run this from the repository root:

```bash
python example/validate_config.py example/<config>.yaml
```

To validate structure without checking whether dataset and embedding paths exist:

```bash
python example/validate_config.py --skip-path-check example/<config>.yaml
```

To show normalized file paths during validation:

```bash
python example/validate_config.py --show-paths example/<config>.yaml
```

If no config file is provided, `example/validate_config.py` validates:

- `example/train_rule_based.yaml`
- `example/train_phase_field.yaml`

### Run a config with the example entrypoint

Run training or testing using the main example driver:

```bash
python example/main.py --config example/<config>.yaml --log_filename <name>
```

The example driver will:

- read the YAML with `tfm.read_yaml_config(...)`
- merge values with built-in defaults
- validate the merged config with `tfm.validate_config(...)`
- build section configs for `datasets`, `encoder`, `decoder`, `model_params`, `embeddings`, `checkpoints`, and `rule_based_params`

### Run phase-field tests with generated configs

The phase-field sweep helper uses `example/test_phase_field.yaml` by default:

```bash
python example/run_all_phase_tests.py
```

Override the base config with:

```bash
python example/run_all_phase_tests.py --base-config example/<config>.yaml
```

## Supported top-level sections

The loader merges user YAML with built-in defaults. The supported sections are:

- `model_params`
- `checkpoints`
- `datasets`
- `embeddings`
- `encoder`
- `decoder`
- `rule_based_params`

Additionally, YAML may include a shared `default` or `defaults` mapping. Keys from that mapping are copied into each required section if missing.

## `model_params`

Common keys:

- `path_working_directory`: working directory for logs, checkpoints, outputs.
- `num_workers`: number of data loading workers.
- `num_nodes`: number of distributed nodes.
- `devices`: number of devices to use.
- `seed`: random seed.
- `batch_pixels`: batch size measured in pixels.
- `lr`: learning rate.
- `accum_grads`: gradient accumulation steps.
- `warmup_steps`: scheduler warmup steps.
- `max_steps`: maximum training/testing steps.
- `gradient_clip_val`: gradient clipping value.
- `weight_decay`: optimizer weight decay.
- `sim_type`: active simulation mode. Allowed values:
  - `RB`
  - `PHASE`
  - `HOSS`
  - `MIXED`
- `sim_weights`: required when `sim_type: MIXED`. Must be a dict with supported simulation names and numeric weights.
- `accelerator`: training accelerator, typically `cpu`.
- `space_bands`: model-specific band count.
- `log_every_n_steps`: how often to log training progress.
- `check_val_every_n_epoch`: validation frequency in epochs.
- `every_n_epochs`: checkpoint frequency in epochs.
- `every_n_train_steps`: checkpoint frequency in training steps.
- `flush_logs_every_n_steps`: logger flush frequency.
- `number_samples_per_epoch`: number of samples per epoch.
- `single_loss_scale`: default value exists in code but is optional in YAML.
- `test_bc`: boundary condition selector for phase tests. Allowed values:
  - `auto`
  - `vertical`
  - `horizontal`
  - `combined`
  - `all`

### Notes for `MIXED`

- `sim_type: MIXED` requires `sim_weights`.
- `sim_weights` may include any supported simulation names, e.g. `RB`, `PHASE`, `HOSS`.
- At least one weight must be positive.

## `checkpoints`

Supported keys:

- `load_model_num`: checkpoint reference. Can be:
  - `null` / missing: no checkpoint loaded.
  - numeric: treated as a Lightning checkpoint version index.
  - string: treated as a checkpoint path relative to the YAML file.
- `run_type`: either `train` or `test`.

## `datasets`

Supported keys:

- `path_phase`: path to PHASE training data.
- `path_val_phase`: path to PHASE validation data.
- `path_hoss`: path to HOSS training data.
- `path_val_hoss`: path to HOSS validation data.
- `path_test`: path to a test HDF5 link file.
- `unstructured_mesh_scale`: numeric scale factor for unstructured mesh input.

### Validation rules

- For `RB`: `emb_rule_based` is required in `embeddings`.
- For `PHASE`: `path_phase`, `path_val_phase`, and `emb_phase` are required.
- For `HOSS`: `path_hoss`, `path_val_hoss`, and `emb_hoss` are required.
- For `run_type: test`:
  - `path_test` is required when active sim type is `PHASE` or `HOSS`.
  - `path_phase` / `path_val_phase` are not required for PHASE test if `path_test` is provided.

## `embeddings`

Supported keys:

- `emb_rule_based`: path to RB embedding directory.
- `emb_phase`: path to PHASE embedding directory.
- `emb_hoss`: path to HOSS embedding directory.
- `num_embedding_variants`: number of embedding variants.

## `encoder`

Supported keys:

- `enc_preproc_ch`
- `num_latents`
- `enc_num_latent_channels`
- `num_layers`
- `num_cross_attention_heads`
- `enc_num_self_attention_heads`
- `num_self_attention_layers_per_block`
- `dropout`

## `decoder`

Supported keys:

- `dec_preproc_ch`
- `dec_num_latent_channels`
- `dec_num_cross_attention_heads`
- `latent_size`
- `llm_embedding_dim` (present in defaults; optional if not overridden)

## `rule_based_params`

Supported keys:

- `n`
- `m`
- `numfractures`
- `numtimesteps`
- `num_sims`
- `material`

## Path handling

- Relative paths in YAML are resolved relative to the YAML file’s directory.
- The loader accepts `null` for path-like values when a path is intentionally omitted.
