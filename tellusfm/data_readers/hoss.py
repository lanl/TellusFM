import random 
import numpy as np 
import re 
import torch 

from tellusfm.helper_functions import build_sample_metadata, midpoint, flip_horizontal_mesh, flip_vertical_mesh, flip_both_mesh


def _load_hoss(self):
    method_label = 'finite-discrete element method'
    validation_sample = getattr(self, "_active_validation_sample", None)
    if validation_sample is None:
        material = random.choice(self.hoss_materials)
        orientation = random.choice(self.hoss_orientations)
        data_aug = random.choice([0,1,2,3])
        fracture_key = random.choice(self.simulation_list)
    else:
        variant_index = validation_sample["variant_index"]
        materials = self.hoss_materials
        orientations = self.hoss_orientations
        material = materials[(variant_index // 8) % len(materials)]
        orientation = orientations[variant_index % len(orientations)]
        data_aug = (variant_index // 2) % 4
        if self.hoss_val_data_aug is not None:
            data_aug = self.hoss_val_data_aug
        fracture_key = validation_sample["fracture_key"]


    # Link-table keys are either "Sample_<n>_frac_<m>" or the older "link_frac_pull_z_<m>"
    match = re.match(r"Sample_(\d+)_frac_(\d+)", fracture_key)
    if match:
        fracture_num = int(match.group(2))
    else:
        match = re.match(r"link_frac_pull_z_(\d+)$", fracture_key)
        if not match:
            raise ValueError(f"Invalid fracture key format: {fracture_key}")
        fracture_num = int(match.group(1))

    hoss_file = self._get_hdf5_file("HOSS")
    linked_frac = hoss_file[fracture_key]
        
    data = linked_frac[f"model {fracture_num} fracture initial"][:]
    target = linked_frac[f"model {fracture_num} fracture final"][:]
    time_to_failure = linked_frac[f"model {fracture_num} break time"][()]

    assert(np.all(np.all(data[:, :6] == target[:, :6], axis=1)) == True)

    x1, y1 = data[:, 0], data[:, 1]
    x2, y2 = data[:, 3], data[:, 4]
    mask = (x1 > 0.001) & (x1 < 0.248) & (y1 > 0.001) & (y1 < 0.25) & \
            (x2 > 0.001) & (x2 < 0.248) & (y2 > 0.001) & (y2 < 0.25)
    data = data[mask]
    target = target[mask]

    mesh = midpoint(data[:, 0], data[:, 1], data[:, 3], data[:, 4])
    # Half-vector of each line segment (midpoint -/+ half = the two endpoints); used for plotting
    segment_half = np.stack([(data[:, 3] - data[:, 0]) / 2, (data[:, 4] - data[:, 1]) / 2], axis=1)

    # Small random displacement of the mesh (training only), as in the DI_fractures hoss branch.
    # The displacement has random direction and a random length up to hoss_mesh_displacement.
    if validation_sample is None and self.hoss_mesh_displacement > 0:
        disp = np.random.randn(2)
        disp /= np.linalg.norm(disp)
        mesh = mesh + disp * (np.random.rand() * self.hoss_mesh_displacement)
    mesh = torch.tensor(mesh, dtype=torch.float32)

    initial_values = data[:, 6]
    initial_values[initial_values == 2.0] = 1.0
    target_values = target[:, 6]
    target_values[target_values == 2.0] = 1.0

    if orientation == "vertical":
        mesh = mesh[:, [1, 0]]
        
    if data_aug == 0:
        mesh=flip_horizontal_mesh(mesh)
    elif data_aug == 1:
        mesh = flip_vertical_mesh(mesh)
    elif data_aug==2:
        mesh = flip_both_mesh(mesh)

    # apply the same orientation / flip to the segment half-vectors (a flip negates one component)
    if orientation == "vertical":
        segment_half = segment_half[:, [1, 0]]
    if data_aug in (0, 2):
        segment_half[:, 0] *= -1
    if data_aug in (1, 2):
        segment_half[:, 1] *= -1

    order = np.lexsort((mesh[:, 1], mesh[:, 0]))   # primary key = y, secondary = x
    mesh = mesh[order]  
    segment_half = segment_half[order].astype(np.float32)
    initial_values=initial_values[order]
    target_values=target_values[order]

    metadata = build_sample_metadata(
        dataset_key="hoss",
        method_label=method_label,
        orientation=orientation,
        mat=material,
        is_unstructured=True,
        data_aug=data_aug,
        fracture_key=fracture_key,
        fracture_num=fracture_num,
        segment_half=segment_half,
    )
    if validation_sample is not None:
        metadata.update(
            validation_index=validation_sample["validation_index"],
            validation_sample_index=validation_sample["sample_index"],
            validation_variant_index=validation_sample["variant_index"],
        )

    return initial_values, target_values, mesh, time_to_failure, metadata