import random 
import numpy as np
import torch 
import re
from tellusfm.helper_functions import build_sample_metadata

mat_h5_type = {
               'steel': 'steel',
               'pbx': 'pbx',
               'al': 'aluminum',
               'shale': 'shale',
               'tungsten': 'tungsten',
               # materials added by the fine-tuning tasks (link-table keys use either spelling)
               'ti': 'titanium',
               'titanium': 'titanium',
               'concrete': 'concrete',
           }


def _flip_pair(data, target, mode):
    """Random mirror augmentation of an (H, W, 1) input / target pair (training only).

    mode "full"  : one of {mirror columns, mirror rows, both, none}, each with probability 1/4.
    mode "legacy": reproduces the DI_fractures ``new_material`` branch. There the "horizontal flip" used
                   ``torch.flip(dims=[-1])``, which for an (H, W, 1) tensor flips the size-1 channel axis (a no-op),
                   and the "vertical" and "both" flips both used ``dims=[-2]``/``[-2, -1]``, i.e. they mirror the
                   columns. Net effect: the columns are mirrored with probability 1/2.
    """
    choice = random.choice([0, 1, 2, 3])
    if mode == "legacy":
        dims = {0: [], 1: [1], 2: [1], 3: []}[choice]
    else:
        dims = {0: [1], 1: [0], 2: [0, 1], 3: []}[choice]
    if dims:
        data, target = torch.flip(data, dims=dims), torch.flip(target, dims=dims)
    return data, target

def _load_phase_field(self):
    method_label = 'phase-field method'
    validation_sample = getattr(self, "_active_validation_sample", None)
    if validation_sample is None:
        fracture_key = random.choice(self.simulation_list)
    else:
        fracture_key = validation_sample["fracture_key"]

    match = re.match(r"link_(\w+)_([a-z]+)_([a-z]+)_(\d+)", fracture_key)

    if match:
        material, direction, bc, idx = match.groups()
    else:
        raise ValueError(f"Could not parse phase-field fracture key: {fracture_key}")
    
    material = mat_h5_type[material]
    hdf5_file = self._get_hdf5_file("PHASE")
    frac_data=hdf5_file[fracture_key]

    try:
        data = np.expand_dims(np.array(frac_data[f"fracture initial"][:]), axis=-1)
    except (KeyError, IndexError):
        data = np.expand_dims(np.array(frac_data[f"model {idx} fracture initial"][:]), axis=-1)
    data = torch.tensor(data, dtype=torch.float32)

    try:
        target = np.expand_dims(np.array(frac_data[f"fracture final"][:]), axis=-1)
    except (KeyError, IndexError):
        target = np.expand_dims(np.array(frac_data[f"model {idx} fracture final"][:]), axis=-1)
    target = torch.tensor(target, dtype=torch.float32)

    time_to_failure = [[[0.0]]]
    time_to_failure=torch.tensor(time_to_failure, dtype=torch.float32)

    requested_orientation = None
    if validation_sample is not None:
        requested_orientation = validation_sample.get("test_bc")

    if requested_orientation is not None:
        orientation = requested_orientation
    elif bc == "z":
        #print("Validation sample: ", validation_sample)
        if validation_sample is None:
            orientation = random.choice(["vertical", "horizontal"])
        else:
            orientations = ["vertical", "horizontal"]
            orientation = orientations[validation_sample["variant_index"] % len(orientations)]
    else:
        orientation = "combined"


    source_orientation = direction if direction in {"vertical", "horizontal"} else orientation
    if bc == "z" and orientation in {"vertical", "horizontal"} and orientation != source_orientation:
        data = data.transpose(0, 1)
        target = target.transpose(0, 1)

    if validation_sample is None and self.phase_flip_augmentation != "none":
        data, target = _flip_pair(data, target, self.phase_flip_augmentation)

    metadata = build_sample_metadata(
        dataset_key="phase_field",
        method_label=method_label,
        orientation=orientation,
        mat=material,
        bc=bc,
        is_unstructured=False,
        fracture_key=fracture_key,
        sample_index=idx,
        source_orientation=source_orientation,
    )
    if validation_sample is not None:
        metadata.update(
            validation_index=validation_sample["validation_index"],
            validation_sample_index=validation_sample["sample_index"],
            validation_variant_index=validation_sample["variant_index"],
        )

    return data, target, time_to_failure, metadata
