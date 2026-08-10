import math
import torch
import torch.nn as nn
from einops import rearrange, repeat


def positional_encoder(image_shape, num_frequency_bands, max_frequencies=None):
    """
    Generate sinusoidal positional encodings for an N-dimensional spatial grid.

    This function creates sinusoidal positional encodings (using sine and cosine
    functions) across spatial dimensions of an input grid, similar to those used
    in transformer models or neural radiance fields (NeRF). The encodings provide
    each spatial location with a unique, continuous representation that helps 
    neural networks learn spatial relationships.

    Parameters
    ----------
    image_shape : tuple of int
        The shape of the input tensor (e.g., `(H, W, C)` for a 2D image or `(D, H, W, C)`
        for a 3D volume). The last dimension is assumed to be channels.
    num_frequency_bands : int
        Number of frequency bands to use for encoding each spatial dimension.
        Larger values capture higher-frequency spatial variations.
    max_frequencies : sequence of float or None, optional
        Maximum frequency values for each spatial dimension. If None, defaults
        to the size of each spatial dimension (i.e., `image_shape[:-1]`).

    Returns
    -------
    torch.Tensor
        A flattened tensor of shape `(N, 2 * num_frequency_bands * D)` where
        `D` is the number of spatial dimensions and `N` is the total number of
        spatial positions (product of all spatial dimensions).
        The tensor contains concatenated sine and cosine positional encodings.

    Notes
    -----
    - The encoding follows the formulation:
      .. math::
         [\sin(\pi x f), \cos(\pi x f)]_{f \in F}
      where `x` are normalized coordinates in [-1, 1] and `F` is the set of
      sampled frequency bands per spatial dimension.
    - The encodings are flattened along spatial dimensions for convenience.

    Examples
    --------
    >>> enc = positional_encoder((64, 64, 3), num_frequency_bands=6)
    >>> enc.shape
    torch.Size([4096, 12])  # (64x64 positions, 2 * 6 frequency bands)
    """

    # Unpack the spatial dimensions (ignore channel dimension)
    *spatial_shape, _ = image_shape

    # Generate coordinate grids for each spatial dimension in the range [-1, 1]
    coords = [torch.linspace(-1, 1, steps=s) for s in spatial_shape]

    # Create a meshgrid representing all spatial positions
    pos = torch.stack(torch.meshgrid(*coords, indexing='ij'), dim=len(spatial_shape))

    # If no maximum frequency is specified, use the spatial size of each dimension
    if max_frequencies is None:
        max_frequencies = pos.shape[:-1]

    # Create a set of frequency values for each spatial dimension
    frequencies = [
        torch.linspace(1.0, max_freq / 2.0, num_frequency_bands)
        for max_freq in max_frequencies
    ]

    # Compute coordinate-frequency products for each spatial dimension
    frequency_grids = []
    for i, frequencies_i in enumerate(frequencies):
        # Multiply coordinates by frequencies (broadcasting)
        frequency_grids.append(pos[..., i:i+1] * frequencies_i[None, ...])

    # Compute sine and cosine components for each frequency grid
    encodings = []
    encodings.extend([torch.sin(math.pi * fgrid) for fgrid in frequency_grids])
    encodings.extend([torch.cos(math.pi * fgrid) for fgrid in frequency_grids])

    # Concatenate encodings along the channel dimension
    enc = torch.cat(encodings, axis=-1)

    # Flatten encodings along spatial dimensions (e.g., H×W×D → N)
    enc = rearrange(enc, "... c -> (...) c")

    return enc


def positional_encoder_unstructured(mesh, num_frequency_bands, max_frequency=None, coordinate_scale=1000.0):
    """
    Compute sinusoidal positional encodings for an unstructured 2D mesh.

    This function encodes arbitrary (x, y) spatial coordinates with sinusoidal
    position features across multiple frequency bands. It is particularly useful
    when the spatial domain is *unstructured* — i.e., points are not aligned on
    a uniform grid (e.g., mesh nodes or scattered samples).

    Parameters
    ----------
    mesh : torch.Tensor
        A tensor of shape ``(N, 2)`` containing the unstructured coordinates.
        Each row is a 2D point: ``(x, y)``.
    num_frequency_bands : int
        Number of frequency bands used per dimension.
        Higher values increase the spatial frequency resolution.
    max_frequency : sequence of float or None, optional
        Maximum frequency to use per dimension ``[fx, fy]``.
        If ``None``, these values are derived from the maximum
        coordinate extents of the input mesh.

    Returns
    -------
    torch.Tensor
        A tensor of shape ``(N, 4 * num_frequency_bands)`` containing concatenated
        sine and cosine encodings for each frequency and dimension.
        (2 dimensions × 2 trigonometric functions × num_frequency_bands)

    Notes
    -----
    The encoding follows the standard formulation used in transformer
    architectures and NeRF positional encodings:

    .. math::
        \gamma(p) = [\sin(\pi p f), \cos(\pi p f)]_{f \in F}

    where ``p`` are normalized coordinates in ``[-1, 1]`` and ``F`` is a set
    of frequency bands sampled between 1 and ``max_frequency / 2``.

    Examples
    --------
    >>> mesh = torch.rand(100, 2) * 10.0
    >>> enc = positional_encoder_unstructured(mesh, num_frequency_bands=6)
    >>> enc.shape
    torch.Size([100, 24])
    """

    # --- Extract x, y coordinates ---
    x_sens = mesh[:, 0] * coordinate_scale
    y_sens = mesh[:, 1] * coordinate_scale
    # print("Applying Positional Encoder")

    # --- Compute coordinate scaling factors (maximum extent) ---
    x_max = int(torch.ceil(x_sens.max()))
    y_max = int(torch.ceil(y_sens.max()))

    # --- Normalize coordinates to [0, 1] ---
    x_norm = x_sens / x_max
    y_norm = y_sens / y_max

    # --- Map coordinates to [-1, 1] for symmetric encoding ---
    x_norm = 2 * x_norm - 1
    y_norm = 2 * y_norm - 1

    # Combine into a single position tensor of shape [N, 2]
    pos = torch.stack((x_norm, y_norm), dim=-1)

    # --- Determine max frequencies per dimension ---
    if max_frequency is None:
        max_frequency = [x_max, y_max]

    # --- Create evenly spaced frequencies per dimension ---
    frequencies = [
        torch.linspace(1.0, max_freq / 2, num_frequency_bands)
        for max_freq in max_frequency
    ]
    # Alternatively, use logarithmic spacing for broader scale coverage:
    # frequencies = [
    #     torch.logspace(math.log10(1), math.log10(max_freq / 2), num_frequency_bands)
    #     for max_freq in max_frequency
    # ]

    # --- Stack frequency bands into a tensor of shape [D, B] (2 x num_bands) ---
    freqs = torch.stack(frequencies, dim=0)

    # --- Broadcast to [N, D, B] to apply all frequencies to all points ---
    scaled = pos.unsqueeze(-1) * freqs.unsqueeze(0)

    # --- Compute sine and cosine components for each frequency band ---
    sin_part = torch.sin(math.pi * scaled).reshape(pos.size(0), -1)  # [N, D*B]
    cos_part = torch.cos(math.pi * scaled).reshape(pos.size(0), -1)  # [N, D*B]

    # --- Concatenate sine and cosine encodings along the last dimension ---
    enc = torch.cat([sin_part, cos_part], axis=-1)  # [N, 2*D*B]

    # --- Flatten for convenience (though it’s already [N, C]) ---
    enc = rearrange(enc, "... c -> (...) c")

    return enc

def apply_positional_encoder(
    mesh,
    num_frequency_bands,
    pixels=None,
    structured=True,
    max_frequency=None,
    coordinate_scale=1000.0,
):
    """
    Apply the appropriate positional encoder (structured or unstructured)
    to a spatial mesh or grid.

    This function dispatches to either `positional_encoder` (for structured
    image-like data) or `positional_encoder_unstructured` (for arbitrary
    unstructured coordinates), returning the encoded positional features
    in a batch-like tensor format.

    Parameters
    ----------
    mesh : torch.Tensor or tuple
        Input spatial data to be encoded.
        - For structured grids: typically an `image_shape` tuple, e.g. `(H, W, C)`.
        - For unstructured meshes: a tensor of shape `(N, D)` representing N
          spatial points with D coordinates.
    num_frequency_bands : int
        Number of frequency bands to use in the positional encoding.
        Higher values increase the representational frequency range.
    structured : bool, optional
        Whether to treat the input as a structured grid (`True`) or an
        unstructured mesh (`False`). Defaults to `True`.
    max_frequency : sequence of float or None, optional
        Maximum frequency values per dimension. If `None`, these values
        are inferred automatically from the mesh or grid size.

    Returns
    -------
    torch.Tensor
        A tensor containing the positional encodings with an added batch
        dimension (shape: `(1, N, C)`), where:
        - `N` is the total number of spatial positions.
        - `C` is the number of encoded channels.

    Examples
    --------
    >>> # Structured image grid (e.g., for CNN features)
    >>> enc = apply_positional_encoder((64, 64, 3), num_frequency_bands=6, structured=True)
    >>> enc.shape
    torch.Size([1, 4096, 12])

    >>> # Unstructured mesh (e.g., for point cloud)
    >>> points = torch.rand(100, 2) * 10.0
    >>> enc = apply_positional_encoder(points, num_frequency_bands=6, structured=False)
    >>> enc.shape
    torch.Size([1, 100, 24])
    """

    # Select which encoder to apply based on mesh structure
    if structured:
        pos_encodings = positional_encoder(mesh, num_frequency_bands, max_frequency)
        # Add a batch dimension for compatibility with models expecting [B, N, C]
        coords = pos_encodings[pixels,][None,]
    else:
        pos_encodings = positional_encoder_unstructured(
            mesh,
            num_frequency_bands,
            max_frequency,
            coordinate_scale=coordinate_scale,
        )
        # Add a batch dimension for compatibility with models expecting [B, N, C]
        coords = pos_encodings[None, :]
   

    return coords
