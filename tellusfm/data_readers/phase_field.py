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
               'tungsten': 'tungsten'
           }

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
