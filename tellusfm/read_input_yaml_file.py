import os
import yaml
from copy import deepcopy
from typing import Any, Dict, Mapping


CONFIG_DIR_KEY = "_config_dir"

REQUIRED_SECTIONS = ["model_params", "datasets", "embeddings", "encoder", "decoder"]

SECTION_PATH_KEYS = {
    "model_params": ("path_working_directory",),
    "datasets": ("path_phase", "path_val_phase", "path_hoss", "path_val_hoss", "path_test"),
    "embeddings": ("emb_rule_based", "emb_phase", "emb_hoss"),
    "checkpoints": ("load_model_num",),
}

SIM_PATH_REQUIREMENTS = {
    "RB": {"embeddings": ("emb_rule_based",)},
    "PHASE": {"datasets": ("path_phase", "path_val_phase"), "embeddings": ("emb_phase",)},
    "HOSS": {"datasets": ("path_hoss", "path_val_hoss"), "embeddings": ("emb_hoss",)},
}

SUPPORTED_SIM_TYPES = set(SIM_PATH_REQUIREMENTS)


def normalize_config_path(value, base_dir=None):
    """
    Expand and absolutize a config path while preserving null values.

    Relative paths are resolved against ``base_dir``. When ``base_dir`` is not
    supplied, the current working directory is used.
    """
    if value is None:
        return None
    if not isinstance(value, (str, os.PathLike)):
        return value

    path = os.path.expandvars(os.path.expanduser(os.fspath(value)))
    if not path:
        return path
    if not os.path.isabs(path):
        anchor = base_dir or os.getcwd()
        path = os.path.join(anchor, path)
    return os.path.normpath(os.path.abspath(path))


def _get_config_dir(config):
    config_dir = config.get(CONFIG_DIR_KEY)
    if isinstance(config_dir, str) and config_dir:
        return config_dir
    return os.getcwd()


def _normalize_section_paths(section_config, section_name, base_dir):
    for key in SECTION_PATH_KEYS.get(section_name, ()):
        if key in section_config:
            section_config[key] = normalize_config_path(section_config[key], base_dir)
    return section_config


def get_default_config():
    """
    Generate the default experiment configuration aligned with the YAML structure.

    Returns
    -------
    dict
        A dictionary where top-level keys match the sections
        of the reshuffled input.yaml file:
        model_params, checkpoints, datasets, embeddings,
        encoder, decoder, and rule_based_params.
    """

    return {
        "model_params": {
            "path_working_directory": os.getcwd(),
            "num_workers": 1,
            "num_nodes": 1,
            "devices": 4,
            "seed": 42,
            "batch_pixels": 10000,
            "lr": 0.0001,
            "accum_grads": 32,
            "warmup_steps": 10000,
            "max_steps": 100000000,
            "gradient_clip_val": 0.5,
            "weight_decay": 0.01,
            "sim_type": "RB",
            "sim_weights": {
                "RB": 1.0,
                #"PHASE": 0.8,
                #"HOSS": 0.8,
            },
            "accelerator": "cpu",
            "space_bands": 64,
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 1,
            "every_n_epochs": 1,
            "every_n_train_steps": 1000,
            "flush_logs_every_n_steps": 10000,
            "number_samples_per_epoch": 100000,
            #"single_loss_scale": 10000,
            "test_bc": "auto",
        },

        "checkpoints": {
            "load_model_num": None,
            "run_type": "train",
        },

        "datasets": {
            "path_phase": None,
            "path_val_phase": None,
            "path_hoss": None,
            "path_val_hoss": None,
            "path_test": None,
            "unstructured_mesh_scale": 1000.0,
        },

        "embeddings": {
            "emb_rule_based": None,
            "emb_phase": None,
            "emb_hoss": None,
            "num_embedding_variants": 20,
        },

        "encoder": {
            "enc_preproc_ch": 64,
            "num_latents": 2048,
            "enc_num_latent_channels": 256,
            "num_layers": 3,
            "num_cross_attention_heads": 2,
            "enc_num_self_attention_heads": 2,
            "num_self_attention_layers_per_block": 3,
            "dropout": 0.0,
        },

        "decoder": {
            "dec_preproc_ch": 64,
            "dec_num_latent_channels": 256,
            "dec_num_cross_attention_heads": 2,
            "latent_size": 1,
            "llm_embedding_dim": 4096,
        },

        "rule_based_params": {
            "n": 100,
            "m": 100,
            "numfractures": 20,
            "numtimesteps": 100,
            "num_sims": 50,
            "material": "pbx",
        },
    }


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base without mutating inputs.
    Dicts are merged, non-dicts replace. Lists in override replace base."""
    a = deepcopy(base)
    b = override or {}
    for k, v in b.items():
        if k in a and isinstance(a[k], Mapping) and isinstance(v, Mapping):
            a[k] = deep_merge(a[k], v)
        else:
            a[k] = deepcopy(v)
    return a


def read_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Reads a YAML configuration file, applying defaults and structural checks.

    Returns
    -------
    dict: Parsed configuration with defaults merged.
    """
    config_path = normalize_config_path(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"YAML file not found: {config_path}")

    with open(config_path, "r") as file:
        user_cfg = yaml.safe_load(file) or {}

    if not isinstance(user_cfg, dict):
        raise ValueError("YAML content is not a valid dictionary")

    # Start from full default baseline generated in code
    baseline = get_default_config()

    # Merge user config over the baseline (user wins)
    merged = deep_merge(baseline, user_cfg)

    # Accept both 'default' and 'defaults' from user YAML
    shared_defaults = {}
    if isinstance(user_cfg.get("default"), dict):
        shared_defaults.update(user_cfg["default"])
    if isinstance(user_cfg.get("defaults"), dict):
        shared_defaults.update(user_cfg["defaults"])

    # If a section is missing in user YAML entirely, it is already present from baseline.
    # Now fill any missing keys from shared_defaults (without overwriting user values).
    if shared_defaults:
        for section_name in REQUIRED_SECTIONS:
            section = merged.get(section_name)
            if not isinstance(section, dict):
                # If something odd happened, re-establish section as dict
                section = {}
                merged[section_name] = section
            for k, v in shared_defaults.items():
                section.setdefault(k, v)

    # Final sanity check
    for section in REQUIRED_SECTIONS:
        if section not in merged or not isinstance(merged[section], dict):
            raise KeyError(f"Missing or invalid section '{section}' after merging defaults")

    merged[CONFIG_DIR_KEY] = os.path.dirname(config_path)

    return merged


def _active_sim_types(model_config):
    sim_type = model_config.get("sim_type")
    errors = []

    if not isinstance(sim_type, str):
        return [], ["model_params.sim_type must be a string"]

    if sim_type == "MIXED":
        sim_weights = model_config.get("sim_weights")
        if not isinstance(sim_weights, dict):
            return [], ["model_params.sim_weights must be a dictionary when sim_type is MIXED"]

        active = []
        for name, weight in sim_weights.items():
            if name not in SUPPORTED_SIM_TYPES:
                errors.append(f"Unknown simulation type in model_params.sim_weights: {name}")
                continue
            try:
                numeric_weight = float(weight)
            except (TypeError, ValueError):
                errors.append(f"Simulation weight for {name} must be numeric")
                continue
            if numeric_weight < 0:
                errors.append(f"Simulation weight for {name} must be non-negative")
            elif numeric_weight > 0:
                active.append(name)

        if not active:
            errors.append("At least one simulation weight must be greater than zero")
        return active, errors

    if sim_type not in SUPPORTED_SIM_TYPES:
        return [], [f"model_params.sim_type must be one of {sorted(SUPPORTED_SIM_TYPES | {'MIXED'})}"]
    return [sim_type], errors


def validate_config(config, check_paths=True):
    """
    Validate a merged TellusFM configuration.

    Parameters
    ----------
    config : dict
        Full configuration dictionary returned by ``read_yaml_config``.
    check_paths : bool, optional
        If True, required dataset files and embedding directories must exist.

    Returns
    -------
    dict
        Summary containing active simulation types, checked paths, and warnings.

    Raises
    ------
    ValueError
        If the configuration is structurally invalid or required paths are missing.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")

    errors = []
    warnings = []

    for section in REQUIRED_SECTIONS:
        if section not in config or not isinstance(config[section], dict):
            errors.append(f"Missing or invalid section: {section}")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))

    base_dir = _get_config_dir(config)
    model_config = _normalize_section_paths(dict(config["model_params"]), "model_params", base_dir)
    data_config = _normalize_section_paths(dict(config["datasets"]), "datasets", base_dir)
    embeddings_config = _normalize_section_paths(dict(config["embeddings"]), "embeddings", base_dir)
    checkpoints_config = dict(config.get("checkpoints", {}))
    run_type = str(checkpoints_config.get("run_type", "train")).lower()
    test_bc = str(model_config.get("test_bc", "auto")).lower()
    allowed_test_bc = {"auto", "vertical", "horizontal", "combined", "all"}
    if test_bc not in allowed_test_bc:
        errors.append(
            "model_params.test_bc must be one of "
            f"{sorted(allowed_test_bc)}"
        )

    sim_types, sim_errors = _active_sim_types(model_config)
    errors.extend(sim_errors)

    checked_paths = {}
    path_sections = {
        "model_params": model_config,
        "datasets": data_config,
        "embeddings": embeddings_config,
    }

    for section_name, section_config in path_sections.items():
        for key in SECTION_PATH_KEYS.get(section_name, ()):
            value = section_config.get(key)
            if value is not None and not isinstance(value, str):
                errors.append(f"{section_name}.{key} must be a string path or null")

    working_directory = model_config.get("path_working_directory")
    if isinstance(working_directory, str) and working_directory:
        checked_paths["model_params.path_working_directory"] = working_directory
        if os.path.exists(working_directory) and not os.path.isdir(working_directory):
            errors.append("model_params.path_working_directory exists but is not a directory")
        elif check_paths and not os.path.exists(working_directory):
            warnings.append(f"Working directory does not exist yet: {working_directory}")

    for sim_type in sim_types:
        requirements = SIM_PATH_REQUIREMENTS[sim_type]
        for section_name, keys in requirements.items():
            if run_type == "test" and section_name == "datasets":
                keys = () if sim_type == "RB" else ("path_test",)

            section_config = path_sections[section_name]
            for key in keys:
                value = section_config.get(key)
                label = f"{section_name}.{key}"
                if not value:
                    errors.append(f"{label} is required when {sim_type} data is active")
                    continue
                if not isinstance(value, str):
                    continue
                checked_paths[label] = value
                if not check_paths:
                    continue
                if section_name == "embeddings":
                    if not os.path.isdir(value):
                        errors.append(f"{label} must point to an existing directory: {value}")
                elif not os.path.isfile(value):
                    errors.append(f"{label} must point to an existing file: {value}")

    if errors:
        raise ValueError("Invalid configuration:\n- " + "\n- ".join(errors))

    return {
        "sim_types": sim_types,
        "checked_paths": checked_paths,
        "warnings": warnings,
    }


def create_section_config(config, section_name, verbose=False, extras=None):
    """
    Build a configuration dictionary for a given section of the full config.

    This function extracts all key-value pairs from the specified section
    (e.g. 'encoder', 'decoder', 'datasets') and optionally merges in
    additional cross-section parameters.

    Parameters
    ----------
    config : dict
        Full configuration dictionary (after merging defaults and YAML).
    section_name : str
        Name of the section to extract (e.g. 'encoder', 'decoder').
    verbose : bool, optional
        If True, pretty-prints the resulting section configuration.
    extras : dict, optional
        Extra key-value pairs to inject (e.g. {"space_bands": config["model_params"]["space_bands"]}).

    Returns
    -------
    dict
        Dictionary of parameters for the specified section.
    """

    print(f"Parsing Config file based on {section_name}")
    if section_name not in config:
        raise KeyError(f"Section '{section_name}' not found in configuration")

    section_config = {k: v for k, v in config[section_name].items()}

    # Merge in any extras (e.g., space_bands from model_params)
    if extras:
        section_config.update(extras)

    section_config = _normalize_section_paths(
        section_config,
        section_name,
        _get_config_dir(config),
    )

    if verbose:
        from tellusfm.helper_functions import print_config

        print_config(section_config, title=f"{section_name.capitalize()} Config")

    return section_config
