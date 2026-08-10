# Embeddings Preprocessing

This folder contains a helper script for generating attention-based embeddings from a large language model and saving them for later use.

## Purpose

The current script is designed for the two TellusFM dataset types used in this repository:

- **Phase field dataset** — prompt templates reference phase-field fracture behavior.
- **Rule-based dataset** — prompt templates reference rule-based fracture modes (`T-fracture` and `X-fracture`).

It uses prompt variations to produce embeddings that capture textual descriptions of fracture type, material, and stress orientation.

## What this adds to the model

Instead of directly training on raw simulation data, this preprocessing step produces learned signal embeddings from a language model. These embeddings can be used as additional features or representations for the TellusFM model, helping it encode:

- fracture type semantics
- material-specific fracture behavior
- directional stress context
- prompt variation robustness

By collecting activations from multiple transformer layers, the code captures richer, higher-level representations than a single token output alone.

## Inputs

The script uses the following inputs:

- `MODEL_ID`: currently set to `meta-llama/Meta-Llama-3.1-8B-Instruct`
- `HUGGINGFACE_API_TOKEN`: must be set in the environment
- 20 prompt templates in `prompt_templates`
- `material_types`: `steel`, `pbx`, `aluminum`, `shale`, `tungsten`
- `fracture_types`: `phase-field method`, `T-fracture rule-based mode`, `X-fracture rule-based mode`
- `directions`: `horizontal`, `combined`, `vertical`
- device selection via PyTorch auto-detection:
  - `cuda` if available
  - else `mps` if available
  - else `cpu`

## Outputs

For every generated prompt combination, the script saves a PyTorch tensor file into `emb_folder/`.

Saved files are named like:

```text
emb_folder/<material>_<fracture>_<direction>_<i>.pt
```

Each `.pt` file contains a stacked tensor of self-attention activations captured from the model layers specified by `layers_to_hook`.

## What has been saved

The script captures self-attention activations from the following layers:

- layer 0
- layer 7
- layer 15
- layer -1 (final layer)

For each prompt, it registers a forward hook on `model.model.layers[layer].self_attn` and stores the attention output vector for the last token.

## How to use

1. Install required dependencies if not already available:

```bash
pip install torch transformers
```

2. Set your Hugging Face token:

```bash
export HUGGINGFACE_API_TOKEN="hf_..."
```

3. Run the script from the `preprocessing/src_embeddings` directory:

```bash
cd preprocessing/src_embeddings
python embeddings.py
```

4. Check the generated embeddings in `emb_folder/`.

## Important notes

- The script currently processes only the first prompt template because it uses `prompt_templates[:1]`.
- To generate embeddings for all 20 templates, change that line to `for i, template in enumerate(prompt_templates):`.
- The output tensor shape depends on the attention head output dimension of the loaded model.

## Recommended improvements

- Add a command-line flag to choose how many templates to run.
- Add a configurable output directory instead of the hard-coded `emb_folder/`.
- Add error handling for missing or invalid `HUGGINGFACE_API_TOKEN` values.

## Summary

This script is a preprocessing helper that converts structured fracture/ material prompts into attention-based embeddings for phase field and rule-based datasets. It is useful when you want to augment the TellusFM pipeline with LLM-derived representations.