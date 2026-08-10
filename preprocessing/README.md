# Preprocessing for TellusFM

This folder contains preprocessing scripts for downloading data, generating embeddings, and creating HDF5 external link tables used by TellusFM.

## Overview

The preprocessing pipeline consists of two main components:

1. **Embeddings** — Generate LLM-based embeddings for fracture and material descriptions
2. **Link Tables** — Create HDF5 external link tables that organize simulation data for efficient loading

## Directory structure

```
preprocessing/
├── src_embeddings/          # Embedding generation scripts
│   ├── embeddings.py        # Generate LLM embeddings for prompts
│   └── README.md            # Embeddings documentation
└── src_link_tables/         # Link table creation scripts
    ├── create_links.py                      # Create rule-based link tables
    ├── create_links_phase.py                # Create phase-field train/val link tables
    └── create_links_phase_testset.py        # Create all phase-field test link tables
```

## Data download

### Included datasets

This repository includes the **orthogonal and random test sets**:
- Raw HDF5 files
- Pre-generated external link tables

No download needed for these datasets.

### Additional datasets on Hugging Face

The following datasets are available on Hugging Face and must be downloaded separately:

- **High density test set** — Phase-field test data with high material density
- **Low density test set** — Phase-field test data with low material density
- **Curved test set** — Phase-field test data with curved fracture paths
- **Training data** — Phase-field and rule-based training datasets
- **Validation data** — Phase-field and rule-based validation datasets

#### Download instructions

1. Visit the Hugging Face model repository: `https://huggingface.co/...`
2. Download the desired dataset files
3. Extract them into the `data/` directory, maintaining the folder structure:

```
data/
├── phase_field/
│   ├── train/              # Phase-field training data
│   ├── validation/         # Phase-field validation data
│   └── test_set/           # Phase-field test sets (curved, random, ortho, etc.)
└── rule_based/
    ├── train/              # Rule-based training data
    └── validation/         # Rule-based validation data
```

## Creating link tables

HDF5 external link tables provide a unified interface to simulation data without duplicating files. They enable efficient data loading during training and testing.

### Phase-field training and validation link tables

Use this to create link tables for phase-field training and validation data:

```bash
cd preprocessing/src_link_tables
python create_links_phase.py
```

This creates:
- `data/phase_field/external_links/train_link.h5`
- `data/phase_field/external_links/validation_link.h5`

These files contain external links pointing to individual HDF5 files for materials: `al`, `pbx`, `shale`, `steel`, `tungsten`.

### Phase-field test set link tables

Create link tables for all phase-field test sets:

```bash
cd preprocessing/src_link_tables
python create_links_phase_testset.py
```

**Note:** Ensure the corresponding test set data exists in `data/phase_field/test_set/` before running.

The script discovers every directory under `data/phase_field/test_set/PHASE-FIELD/`
that directly contains `.h5` files and creates one link table for that directory.
It overwrites existing tables so that they cannot retain links to files from a
different source folder. Use `--dry-run` to preview the tables without writing them.

This creates test-specific link files:
- `al_ortho_axial_link.h5`, `pbx_ortho_axial_link.h5`, etc. (orthogonal tests)
- `al_random_axial_link.h5`, `pbx_random_biaxial_link.h5`, etc. (random tests)
- `al_lowDensity_*.h5`, `al_highDensity_*.h5`, etc. (density tests)

### Rule-based training and validation link tables

Create link tables for rule-based (FWB) data:

```bash
cd preprocessing/src_link_tables
python create_links.py
```

This creates:
- `data/rule_based/external_links/train_link.h5`
- `data/rule_based/external_links/validation_link.h5`

## Embeddings generation

LLM-based embeddings enrich the model with semantic information about fracture types, materials, and stress directions.

See [src_embeddings/README.md](src_embeddings/README.md) for detailed instructions on:
- How embeddings work
- What prompts are used
- How to run the embedding generation script
- Output format and storage

## Workflow summary

### For a fresh TellusFM setup with all data:

1. **Clone and install** (see main README.md Quick Start)
2. **Download datasets** from Hugging Face into `data/` folder
3. **Generate embeddings** (optional, but recommended):
   ```bash
   python src_embeddings/embeddings.py
   ```
4. **Create link tables**:
   ```bash
   cd src_link_tables
   python create_links_phase.py
   python create_links.py
   python create_links_phase_testset.py
   python create_links_phase_testset_random.py
   python create_links_phase_testset_density.py
   ```
5. **Validate and run** (see main README.md Training and Testing sections)

### For quick testing with included data:

1. **Clone and install** (see main README.md Quick Start)
2. Link tables are already provided; skip to **Validate and run**

## Link table structure

Each link table is an HDF5 file containing external links to individual data files. This structure:
- Keeps the original data files intact and uncorrupted
- Enables efficient lazy loading during training
- Provides a unified interface to heterogeneous datasets

Example structure inside `train_link.h5`:
```
link_al_z_frac_pull_z_1
link_al_z_frac_pull_z_2
link_pbx_z_frac_pull_z_1
...
```

Each link points to a specific HDF5 dataset in the original simulation output files.

## Troubleshooting

**"HDF5 file not found" error:**
- Verify data files exist in `data/` directory
- Check paths in the link table creation scripts (they use relative paths)

**Link tables already exist:**
- The phase-field test-set script replaces its existing tables when regenerating them

**Large data downloads on Hugging Face:**
- Use `huggingface-cli` for resumable downloads:
  ```bash
  huggingface-cli download <repo-id> --repo-type model --local-dir data/
  ```
