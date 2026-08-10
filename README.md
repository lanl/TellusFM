# TellusFM — *Transformer Fracture Model*

> *“From the forge of Vulcan and the soil of Tellus — intelligence that understands the Earth.”*

---

## Overview

**TellusFM** is a **Python-based transformer model** for simulating and learning the coupled processes of **flow, fracture, and material evolution**.  
It bridges **machine learning**, **physics-informed modeling**, and **computational materials science** to enable *data-driven discovery* of emergent behaviors in complex materials.

TellusFM draws inspiration from **Roman mythology**, where *Tellus*, the goddess of Earth, embodies strength, structure, and renewal — the perfect metaphor for a model designed to uncover the hidden language of matter.

---

## Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:lanl/TellusFM.git
cd TellusFM
```

### 2. Create and activate a virtual environment

Choose one of the following approaches:

#### Option A: Using `venv` (built-in Python)

```bash
python -m venv tellusfm-env
source tellusfm-env/bin/activate
```

#### Option B: Using `conda`

```bash
conda create -n tellusfm-env python=3.12
conda activate tellusfm-env
```

#### Option C: Using `uv` (faster and recommended)

```bash
uv venv tellusfm-env
source tellusfm-env/bin/activate
```

You should see the environment name in your terminal prompt when activated.

### 4. Install dependencies

After activating your environment, install the required packages:

**For `venv` or `conda`:**
```bash
pip install -r requirements.txt
pip install -e .
```

**For `uv`:**
```bash
uv pip install -r requirements.txt
uv pip install -e .
```

### 5. Validate your setup

Before running training or tests, validate the default example configs:

```bash
python example/validate_config.py --show-paths
```

This checks that all required sections and paths are in place.

---

## Configuration

TellusFM experiments are configured with YAML files. Example configs are in `example/`:

| Config | Purpose |
| --- | --- |
| `example/train_rule_based.yaml` | Rule-based training config. |
| `example/train_phase_field.yaml` | Phase-field training config. |
| `example/train_mixed.yaml` | Mixed rule-based + phase-field training config. |
| `example/test_phase_field.yaml` | Phase-field testing config. |
| `example/test_rule_based.yaml` | Rule-based testing config. |

See [example/YAML_OPTIONS.md](example/YAML_OPTIONS.md) for detailed documentation of all supported config options.

### Data & Datasets

This repository includes the **orthogonal and curved test sets** (both data and external link tables) used to create results S4–S6 in the manuscript.

**Additional test data** is available on Hugging Face:
- High density test set
- Low density test set  
- Random test set

To use these additional datasets:

1. Download them from the Hugging Face model repository
2. Use the preprocessing scripts in `preprocessing/src_link_tables/` to generate the external link tables
3. Update your YAML config with the paths to the downloaded data

Refer to the preprocessing documentation for details on generating external link tables.


## Set up Hugging Face authentication

Create a Hugging Face access token with permission to read the model:

1. Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token with read access
3. Set it as an environment variable:

```bash
export HUGGINGFACE_API_TOKEN="hf_..."
```

### Path handling

Relative paths in YAML are resolved relative to the YAML file's directory:

```yaml
model_params:
  path_working_directory: "."

datasets:
  path_phase: "../data/phase_field/external_links/train_link.h5"
  path_val_phase: "../data/phase_field/external_links/validation_link.h5"

embeddings:
  emb_rule_based: "../data/rule_based/embeddings"
  emb_phase: "../data/phase_field/embeddings"
```

### Validate Configs

Before running training, validate your config:

```bash
python example/validate_config.py example/train_phase_field.yaml
```

To validate without checking that paths exist (useful before downloading data):

```bash
python example/validate_config.py --skip-path-check example/train_phase_field.yaml
```

To see the normalized paths that will be used:

```bash
python example/validate_config.py --show-paths example/train_phase_field.yaml
```

---

## Training

### Basic training run

Start training with a rule-based config:

```bash
python example/main.py --config example/train_rule_based.yaml --log_filename rule_based_run
```

For phase-field training:

```bash
python example/main.py --config example/train_phase_field.yaml --log_filename phase_field_run
```

For mixed training (both rule-based and phase-field):

```bash
python example/main.py --config example/train_mixed.yaml --log_filename mixed_run
```

### Training outputs

Logs and checkpoints are saved to the directory specified by `model_params.path_working_directory` in your config (default: `.`).

Training generates:
- `lightning_logs/` — TensorBoard logs and checkpoint history
- `logs/` — CSV training and validation metrics
- Log files with results summary

---

## Testing

### Run a single test config

Test with a single config file:

```bash
python example/main.py --config example/test_phase_field.yaml --log_filename test_run
```

### Run all phase-field tests

The `example/run_all_phase_tests.py` script automatically runs tests on all phase-field test link files and produces MAE/MSE/SSIM summaries.

#### Basic usage

Run all tests using the default base config:

```bash
python example/run_all_phase_tests.py
```

Results are saved in `example/test_runs/<timestamp>/`:
- `summary.csv` — metrics for each test
- `summary.json` — detailed results in JSON format
- `<test_name>/` — per-test directory with logs and outputs

#### Filter tests by pattern

Run only low and high density tests:

```bash
python example/run_all_phase_tests.py --pattern "lowDensity|highDensity"
```

Run only ortho and curved tests:

```bash
python example/run_all_phase_tests.py --pattern "ortho|curved"
```

Supply patterns separately (equivalent to OR semantics):

```bash
python example/run_all_phase_tests.py --pattern "lowDensity" --pattern "highDensity"
```

#### Advanced options

Dry-run to see which tests will be executed without running them:

```bash
python example/run_all_phase_tests.py --dry-run
```

Limit the number of tests:

```bash
python example/run_all_phase_tests.py --limit 5
```

Run with verbose output:

```bash
python example/run_all_phase_tests.py --verbose
```

Override the base config:

```bash
python example/run_all_phase_tests.py --base-config example/test_rule_based.yaml
```

Stop after the first failure:

```bash
python example/run_all_phase_tests.py --stop-on-error
```

Specify a test boundary condition (auto, vertical, horizontal, combined, all):

```bash
python example/run_all_phase_tests.py --test-bc vertical
```

---

## Building Documentation

```bash
$ sphinx-build -b html docs docs/_build/html
```

Then open the docs:
```
docs/_build/html/index.html
```


### Notice of Copyright Assertion (O4924):

*This program is Open-Source under the BSD-3 License.
Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:*
- *Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.*
- *Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.*
- *Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.*

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


## Citation

If you use **TellusFM** in your research, please cite:
```
Marcato, Agnese, Aleksandra Pachalieva, Ryley G. Hill, Kai Gao, Xiaoyu Wang, Esteban Rougier, Zhou Lei et al. "A foundation model for material fracture prediction." arXiv preprint arXiv:2507.23077 (2025).

```
