# TellusFM — A Pretrained Transferable Multitask Model for Material Fracture Prediction


## Overview

TellusFM is a Python package accompanying the manuscript on pretrained transferable transformer models for material fracture prediction. The repository provides model code, example configurations, preprocessing utilities, pretrained weights, and small test datasets for reproducing the fracture-prediction workflows described in the paper.

---

## Quick Start

Clone the repository

```bash
git clone git@github.com:lanl/TellusFM.git
cd TellusFM
```

Install a new environment
(for other options such as `conda` or `uv`, see [DETAILED_GUIDE.md](DETAILED_GUIDE.md)) 

```bash
python3 -m venv tellusfm-env
source tellusfm-env/bin/activate
```

Install the required packages

```bash
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -e .
```

(Optional) Validate the paths used in the examples

```bash
python example/validate_config.py --show-paths
```

Download the pretrained fracture checkpoint from Huggingface 

```bash
mkdir -p models
cd models
hf download smartFRACs/material_fracturing_model material_fracturing_model.ckpt --local-dir .
cd ..
```

Run a test using the test datasets included in the git repository

```bash
python example/run_all_phase_tests.py --pattern "ortho|curved"
```


Results are saved in `example/test_runs/<timestamp>/`:
- `summary.csv` — metrics for each test
- `summary.json` — detailed results in JSON format
- `<test_name>/` — per-test directory with logs and outputs

For more information on where to find the other training, validation and test sets, see [DETAILED_GUIDE.md](DETAILED_GUIDE.md)) or contact Agnese Marcato ([agnese.marcato@polito.it](mailto:agnese.marcato@polito.it)) or Aleksandra Pachalieva ([apachalieva@lanl.gov](mailto:apachalieva@lanl.gov)).



## Fine-Tuning Tasks

Besides the pretrained model, the repository contains the configuration needed to fine-tune the model on
additional data and to evaluate the result. The first task is **HOSS** (finite-discrete element simulations on an
unstructured mesh).

### HOSS

#### 1. Data

HOSS simulations are one HDF5 file per simulation, named `frac_pull_z_<id>.h5`, containing the datasets
`model <id> fracture initial`, `model <id> fracture final` and `model <id> break time`.
They are not included in the repository. Put them in a folder of your choice, for example `../data/hoss/`.

Create a link table (one external link per simulation) for each split you want to use:

```bash
# validation / test split: all simulations in a folder
python preprocessing/src_link_tables/create_links_hoss.py \
    --source-dir ../data/hoss --output data/hoss/external_links/validation_link.h5 --check

# training split: optionally restrict to the simulations listed in a text file (one file name or id per line)
python preprocessing/src_link_tables/create_links_hoss.py \
    --source-dir /path/to/hoss_train --ids-file train_ids.txt \
    --output data/hoss/external_links/train_link.h5 --check
```

`--check` opens every simulation and verifies that the expected datasets exist. The link table stores paths
relative to itself, so keep the link table and the data in the same relative location.

The text embeddings go in one folder, named `<material>_finite-discrete element method_<orientation>_<i>.torch`
with `i` from 0 to 19, for example `pbx_finite-discrete element method_horizontal_7.torch`.
The default location is `data/hoss/embeddings/`.

You also need a checkpoint, for example `models/hoss_finetuning.ckpt` for evaluation, or the pretrained
`models/material_fracturing_model.ckpt` as the starting point of a fine-tune.

#### 2. Test (evaluate a checkpoint)

```bash
python example/validate_config.py example/test_hoss.yaml      # check the paths first
python example/main.py --config example/test_hoss.yaml --log_filename test_hoss
```

`example/test_hoss.yaml` evaluates the checkpoint in `checkpoints.load_model_num` on the simulations in
`datasets.path_test`. The summary (MAE, MSE; SSIM is not defined for unstructured meshes) is printed at the end,
and one plot per simulation is written to `model_params.test_output_dir` (default `test_samples`, relative to
`path_working_directory`, which itself is relative to the folder of the YAML file).

#### 3. Train (fine-tune)

```bash
python example/validate_config.py example/train_hoss.yaml
python example/main.py --config example/train_hoss.yaml --log_filename train_hoss
```

`example/train_hoss.yaml` loads the weights from `checkpoints.load_model_num` and starts a fresh optimizer
(`checkpoints.resume_training: false`). Training uses `datasets.path_hoss` and validates after every epoch on
`datasets.path_val_hoss`. Outputs, relative to `path_working_directory`:

- `lightning_logs/version_<job id>/checkpoints/`: checkpoints,
- `logs/training_metrics/` and `logs/val_metrics/`: metrics as CSV,
- `logs/val_metrics/version_<job id>/validation_samples/`: validation plots and the `.npy` arrays behind them.

The example is written for a GPU node (`accelerator: gpu`, `devices: 8`). For a quick local test set
`accelerator: cpu`, `devices: 1` and a small `number_samples_per_epoch`.

#### 4. HOSS options

| Option | Meaning |
|---|---|
| `model_params.loss_type` | `relu_mean`: MSE of `relu(output)`, averaged over points (HOSS). `sigmoid_sum`: MSE of `sigmoid(output)`, summed (default, original TellusFM loss). Use the one the checkpoint was trained with. |
| `model_params.loss_type_by_sim` | Per-simulation override, e.g. `{HOSS: relu_mean}` in a `MIXED` run. |
| `model_params.trainable` | `null` trains all parameters. A list of parameter-name substrings freezes everything else, e.g. `[output_field, postproc_field]`. |
| `model_params.test_output_dir` | Folder for the test plots. |
| `model_params.val_samples_per_sim` | Validation passes per simulation (use `1` with a single orientation). |
| `checkpoints.resume_training` | `false`: load the weights only. `true`: also resume optimizer, step and epoch from the checkpoint. |
| `datasets.hoss_materials` | Materials used for the text embedding label, e.g. `[pbx]`. Must match the data in the link table. |
| `datasets.hoss_orientations` | `[horizontal]`, `[vertical]` or both. `vertical` swaps x and y. |
| `datasets.hoss_mesh_displacement` | Maximum random displacement (metres) added to training meshes, `0` to switch off. |
| `datasets.hoss_val_data_aug` | Validation/test mesh flip: `0` horizontal, `1` vertical, `2` both, `3` none, `null` default. |
| `embeddings.emb_hoss`, `embeddings.num_embedding_variants` | Embedding folder and number of variants per material and orientation (20). |

In training a random embedding variant is drawn for every sample. Validation and test use variant
`sample index % num_embedding_variants`, so the results are reproducible.

### New phase-field materials (titanium, concrete)

Fine-tunes the pretrained model on a material that is not part of the pretraining data, using the same
phase-field file format and reader as the existing materials (`titanium`, `concrete`; aluminum, pbx, shale, steel
and tungsten are supported as before).

#### 1. Data

Put the simulation files in folders named `<material>_train/` and `<material>_validation/` below one source
folder, for example `ti_train/frac_pull_z_<id>.h5`, `ti_validation/frac_pull_z_<id>.h5` (`frac_pull_xz_<id>.h5` for
the combined loading). A file holds `fracture initial` and `fracture final` (or `model <id> fracture initial` and
`model <id> fracture final`) as 2D arrays; the image size can differ from the pretraining data (titanium is 128 x 128).
Use the folder prefix `ti` or `titanium` for titanium, and `concrete` for concrete. Then build the link tables:

```bash
python preprocessing/src_link_tables/create_links_phase_train_val.py \
    --source-dir /path/to/source --output-dir data/new_material/external_links
```

Every `<material>_train` / `<material>_validation` folder below `--source-dir` ends up in the same
`train_link.h5` / `validation_link.h5`, so use a separate source folder for each fine-tuning task.

The embeddings are named `<material>_phase-field method_<orientation>_<i>.torch` (orientation `horizontal`,
`vertical` and, for the combined loading, `combined`; `i` from 0 to 19). `titanium` and `concrete` have been added to
the material list of `preprocessing/src_embeddings/embeddings.py`.

#### 2. Train

Set the paths in `example/train_new_material.yaml` (`datasets.path_phase`, `datasets.path_val_phase`,
`embeddings.emb_phase`, `checkpoints.load_model_num`), then

```bash
python example/validate_config.py example/train_new_material.yaml
python example/main.py --config example/train_new_material.yaml --log_filename train_new_material
```

The configuration reproduces the DI_fractures `new_material` setup: loss `sigmoid_sum`, learning rate 1e-4,
gradient accumulation 64, warmup 1000, one epoch of 100000 samples (`max_epochs: 1`), only the field head and the
decoder cross-attention blocks trained (`trainable`), a fresh optimizer (`resume_training: false`) and random
mirroring of the training images (`phase_flip_augmentation: legacy`). Set `trainable: null` to train all parameters.

#### 3. Test

```bash
python example/main.py --config example/test_new_material.yaml --log_filename test_new_material
```

Set `checkpoints.load_model_num` to the fine-tuned checkpoint and `datasets.path_test` to the validation link table.
For phase-field data the vertical/horizontal orientation of each simulation is controlled by `model_params.test_bc`
(`auto`, `vertical`, `horizontal`, `combined`, `all`).

#### 4. Options

| Option | Meaning |
|---|---|
| `model_params.max_epochs` | Stop after this many epochs (`null`: only `max_steps` limits the run). |
| `model_params.trainable` | List of parameter-name substrings that stay trainable; `null` trains everything. `preproc_field` matches no parameter (as in the original branch), add `decoder_1.preproc` to train the decoder input projection too. |
| `datasets.phase_flip_augmentation` | Mirroring of training images. `none`; `full`: mirror columns, rows, both or neither with probability 1/4 each; `legacy`: the original branch behaviour, which mirrors the columns with probability 1/2. In the branch the "horizontal flip" was applied to the size-1 channel axis and had no effect, so `legacy` is not the same as `full`. |

### Notice of Copyright Assertion (O4924):

*This program is Open-Source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:*
- *Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.*
- *Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.*
- *Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.*

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.




