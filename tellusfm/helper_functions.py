import json
import torch
import argparse
import numpy as np 
from tellusfm.logfile import local_print_log 

def prepare_model_inputs(initial_values, target_values, coords, single_value=None, dtype=torch.float32):
    """
    Prepare batched and concatenated input/target tensors for a neural model.

    This function reshapes scalar, vector, and coordinate inputs into
    PyTorch tensors with a consistent batch-first format, suitable for
    neural network inputs of shape ``(batch, N, features)``.

    Parameters
    ----------
    initial_values : array-like
        1D array or list of initial scalar values for each spatial position.
        Shape: (N,)
    target_values : array-like
        1D array or list of target/output values for each spatial position.
        Shape: (N,)
    coords : torch.Tensor
        Tensor of positional encodings or spatial coordinates.
        Expected shape: (N, F) or (1, N, F), where F = number of features.
    single_value : float or None, optional
        Optional scalar value to include (e.g., a global parameter).
        If provided, will be shaped to match the batch format: (1, 1, 1).
    dtype : torch.dtype, optional
        Tensor data type (default: torch.float32).

    Returns
    -------
    tuple of torch.Tensor
        (input_values, target_values, single_value)
        - input_values: Tensor of shape (1, N, F + 1)
          Concatenation of initial_values and coords.
        - target_values: Tensor of shape (1, N, 1)
          Target output tensor for training.
        - single_value: Tensor of shape (1, 1, 1) or None
          Optional scalar parameter, if provided.

    Examples
    --------
    >>> initial = [0.1, 0.2, 0.3]
    >>> target = [1.0, 1.1, 1.2]
    >>> coords = torch.randn(1, 3, 4)
    >>> input_t, target_t, single_t = prepare_model_inputs(initial, target, coords, single_value=5.0)
    >>> input_t.shape, target_t.shape, single_t.shape
    (torch.Size([1, 3, 5]), torch.Size([1, 3, 1]), torch.Size([1, 1, 1]))
    """

    # --- Handle optional single_value ---
    single_tensor = None
    #if single_value is not None:
    single_tensor = torch.tensor([single_value], dtype=dtype)
    single_tensor = single_tensor[None, :, None]  # (1, 1, 1)

    # --- Convert to float tensors ---
    initial_values = torch.as_tensor(initial_values, dtype=dtype)
    target_values = torch.as_tensor(target_values, dtype=dtype)

    # --- Add batch and feature dimensions ---
    initial_values = initial_values.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
    target_values = target_values.unsqueeze(0).unsqueeze(-1)    # (1, N, 1)

    # --- Concatenate initial values with positional encodings ---
    input_values = torch.cat([initial_values, coords], axis=-1)  # (1, N, 1+F)

    return input_values, target_values, coords, single_tensor


def build_sample_metadata(dataset_key, method_label, orientation, mat, bc=None, is_unstructured=False, **extra):
    metadata = {
        "dataset_key": dataset_key,
        "method_label": method_label,
        "orientation": orientation,
        "mat": str(mat),
        "bc": bc,
        "sample_label": f"{mat}_{method_label}_{orientation}",
        "is_unstructured": is_unstructured,
    }
    metadata.update(extra)
    return metadata


def get_sample_prefix(metadata, train):
    mode = "train" if train else "val"
    return f"{mode}/{metadata['dataset_key']}"

# def prepare_model_inputs(initial_values, target_values, coords, single_value=None, dtype=torch.float32):
#     print("\n--- DEBUG: Checking input shapes ---")
#     print(f"initial_values (before): {type(initial_values)}, len={len(initial_values) if hasattr(initial_values, '__len__') else 'scalar'}")
#     print(f"target_values (before): {type(target_values)}, len={len(target_values) if hasattr(target_values, '__len__') else 'scalar'}")
#     print(f"coords (before): {type(coords)}, shape={getattr(coords, 'shape', 'N/A')}")

#     # --- Handle optional single_value ---
#     single_tensor = None
#     if single_value is not None:
#         single_tensor = torch.tensor([single_value], dtype=dtype)
#         single_tensor = single_tensor[None, :, None]  # (1, 1, 1)
#         print(f"single_value -> tensor shape: {single_tensor.shape}")

#     # --- Convert to float tensors ---
#     initial_values = torch.as_tensor(initial_values, dtype=dtype)
#     target_values = torch.as_tensor(target_values, dtype=dtype)
#     print(f"initial_values -> tensor shape: {initial_values.shape}")
#     print(f"target_values -> tensor shape: {target_values.shape}")

#     # --- Add batch and feature dimensions ---
#     initial_values = initial_values.unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
#     target_values = target_values.unsqueeze(0).unsqueeze(-1)    # (1, N, 1)
#     print(f"initial_values after unsqueeze: {initial_values.shape}")
#     print(f"target_values after unsqueeze: {target_values.shape}")

#     # --- Ensure coords are batch-shaped ---
#     if coords.ndim == 2:
#         coords = torch.tensor(coords, dtype=dtype)
#         coords = coords.unsqueeze(0)  # (1, N, F)
#         print(f"coords after unsqueeze: {coords.shape}")
#     else:
#         print(f"coords already batched: {coords.shape}")

#     # --- Concatenate initial values with positional encodings ---
#     input_values = torch.cat([initial_values, coords], dim=-1)  # (1, N, 1+F)
#     print(f"input_values concatenated shape: {input_values.shape}")

#     print("--- DEBUG END ---\n")
#     return input_values, target_values, coords, single_tensor

def print_config(config, title="Experiment Configuration"):
    """
    Pretty-print the experiment configuration in a formatted and readable way.

    Uses standard JSON formatting for readability instead of pprint,
    and routes all output through ``local_print_log`` for unified logging.

    Parameters
    ----------
    config : dict
        The configuration dictionary to display.
    title : str, optional
        A string to display above the configuration printout.
        Default is ``"Experiment Configuration"``.

    Returns
    -------
    None
        This function only prints/logs to stdout and the log file.
    """
    border = "=" * 50
    local_print_log(f"\n{border}", "info")
    local_print_log(f"{title}", "info")
    local_print_log(f"{border}", "info")
    # Use json.dumps for nice formatting
    config_str = json.dumps(config, indent=2)
    local_print_log(config_str, "info")
    local_print_log(f"{border}\n", "info")
    

def frac_threshold(phi):
    w = phi
    w = np.clip(w, 0, 1)
    w = 1 - (1 - w)**0.2
    w[w >= 0.5] = 1
    w[w < 0.5] = 0
    return w

def midpoint(x1,y1,x2,y2):
    xm = (x1+x2)/2
    ym = (y1+y2)/2
    midpoints = np.stack([xm, ym], axis=1)
    return midpoints

def flip_horizontal_mesh(mesh: torch.Tensor) -> torch.Tensor:
    """ Flip mesh left-to-right based on x range (coordinate-based). """
    x_min, x_max = mesh[:, 0].min(), mesh[:, 0].max()
    flipped_x = x_max - (mesh[:, 0] - x_min)
    return torch.stack((flipped_x, mesh[:, 1]), dim=1)

def flip_vertical_mesh(mesh: torch.Tensor) -> torch.Tensor:
    """ Flip mesh top-to-bottom based on y range (coordinate-based). """
    y_min, y_max = mesh[:, 1].min(), mesh[:, 1].max()
    flipped_y = y_max - (mesh[:, 1] - y_min)
    return torch.stack((mesh[:, 0], flipped_y), dim=1)

def flip_both_mesh(mesh: torch.Tensor) -> torch.Tensor:
    """ Flip mesh both horizontally and vertically. """
    return flip_vertical_mesh(flip_horizontal_mesh(mesh))


def parse_command_line():
    """
    Parse command-line arguments for configuration and logging.

    This function defines and parses command-line arguments used to load
    experiment settings from a YAML file, enable verbose logging, and
    configure the output log filename.

    Parameters
    ----------
    None
        Command-line arguments are parsed directly from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        An object containing the parsed arguments with the following attributes:

        - **config** : str  
          Path to the YAML configuration file. (Required)

        - **verbose** : bool  
          Flag to enable verbose logging. Default is False.

        - **log_filename** : str  
          Name of the logfile (without extension). Defaults to ``"sample"``.

    Examples
    --------
    From the command line:

    .. code-block:: bash

       # Run training with a config file and verbose logging
       python main.py --config config.yaml --verbose --log_filename run1

    Programmatic usage:

    >>> args = parse_command_line()
    >>> args.config
    'config.yaml'
    >>> args.verbose
    True
    >>> args.log_filename
    'run1'
    """
    from tellusfm.logo import print_logo 
    print_logo()
    
    parser = argparse.ArgumentParser(
        description="YAML-based configuration loader")
    
    parser.add_argument("--config",
                        type=str,
                        required=True,
                        help="Path to YAML config file")
    
    parser.add_argument(
        "--verbose",
        action="store_true",  # makes it a boolean flag (default False)
        help="Enable verbose logging")
    
    parser.add_argument("--log_filename",
                        type=str,
                        required=False,
                        default="sample",
                        help="Name of logfile") 

    parser.add_argument("--test-bc",
                        type=str,
                        required=False,
                        default=None,
                        choices=["auto", "vertical", "horizontal", "combined", "all"],
                        help="PHASE test BC/orientation to run")

    args = parser.parse_args()
    return args
