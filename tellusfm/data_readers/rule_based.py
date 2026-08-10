import numpy as np
import networkx as nx
import random 
import torch 
from tellusfm.helper_functions import build_sample_metadata


class FractureTip:
    def __init__(self, i, j, di, dj, freeze_time):
        self.i = i
        self.j = j
        self.di = di
        self.dj = dj
        self.freeze_time = freeze_time

class Material:
    def __init__(self, n, m):
        self.isfractured = np.zeros((n, m), dtype=int)

def inside(m, tip):
    return 0 < tip.i <= m.isfractured.shape[0] and 0 < tip.j <= m.isfractured.shape[1]

def timestep(m, tips, tmode, freeze_steps=10):
    for tip in tips:
        tip.freeze_time -= 1
        if inside(m, tip) and tip.freeze_time <= 0:
            m.isfractured[tip.i - 1, tip.j - 1] = 1
        if tip.freeze_time <= 0:
            tip.i += tip.di
            tip.j += tip.dj
            if not tmode and inside(m, tip) and m.isfractured[tip.i - 1, tip.j - 1] == 1:
                tip.freeze_time = freeze_steps
        if tmode and inside(m, tip) and m.isfractured[tip.i - 1, tip.j - 1] == 1:
            tip.freeze_time = float('inf')

def simulate(n, m, numfractures, numtimesteps, tmode=False, p=0.5, freeze_steps=10, rng=None):
    rng = rng or np.random
    mat = Material(n, m)
    fracs = np.zeros((n, m, numtimesteps), dtype=int)
    tips = []
    for k in range(numfractures):
        i = rng.randint(1, n+1)
        j = rng.randint(1, m+1)
        if rng.rand() > p:
            tips.append(FractureTip(i, j, 1, 0, 0))
            tips.append(FractureTip(i, j, -1, 0, 0))
        else:
            tips.append(FractureTip(i, j, 0, 1, 0))
            tips.append(FractureTip(i, j, 0, -1, 0))
    for k in range(numtimesteps):
        timestep(mat, tips, tmode, freeze_steps=freeze_steps)
        fracs[:, :, k] = mat.isfractured
    return fracs, mat, tips

def simulate_material(mat, fracs, tips, numtimesteps, tmode=False, freeze_steps=10):
    for k in range(numtimesteps):
        timestep(mat, tips, tmode, freeze_steps=freeze_steps)
        fracs[:, :, k] = mat.isfractured
    return fracs, mat, tips

def build_graph_from_2d_array(arr):
    
    nrows, ncols = arr.shape
    g = nx.Graph()
    coord_to_id = lambda row, col: (row - 1) * ncols + col
    for row in range(1, nrows + 1):
        for col in range(1, ncols + 1):
            if arr[row - 1, col - 1] == 1:
                for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nrow, ncol = row + drow, col + dcol
                    if 1 <= nrow <= nrows and 1 <= ncol <= ncols and arr[nrow - 1, ncol - 1] == 1:
                        g.add_edge(coord_to_id(row, col), coord_to_id(nrow, ncol))
    for row in range(1, nrows + 1):
        g.add_edge(nrows * ncols + 1, coord_to_id(row, 1))
        g.add_edge(nrows * ncols + 2, coord_to_id(row, ncols))
    return g

def isfailed(frac):
    g = build_graph_from_2d_array(frac)
    return nx.has_path(g, frac.size + 1, frac.size + 2)

def get_failure(fracs):
    for i in range(fracs.shape[2]):
        if isfailed(fracs[:, :, i]):
            return i, fracs[:, :, i]
    return fracs.shape[2] - 1, fracs[:, :, -1]


def load_data_rule_based(n, m, numfractures, numtimesteps, failure_time_scale, tmode=True, rng=None):
    fracs, mat, tips = simulate(n, m, numfractures, 10, tmode=tmode, rng=rng)
    tips = list(filter(lambda x: x.di == 0, tips))
    fracs, mat, tips = simulate_material(mat, np.zeros((n,m,numtimesteps)), tips, numtimesteps,tmode=tmode)
    failure_time, failure_pattern = get_failure(fracs)

    return  torch.as_tensor( fracs[:,:,0], dtype=torch.float ).unsqueeze(2), torch.as_tensor( failure_pattern, dtype=torch.float ).unsqueeze(2), torch.Tensor([failure_time/failure_time_scale])

def _load_rule_based(self):
    material = self.rule_based_material
    validation_sample = getattr(self, "_active_validation_sample", None)
    fracture_variants = [
        ("T", "horizontal"),
        ("T", "vertical"),
        ("X", "horizontal"),
    ]
    if validation_sample is None:
        fracture_mode, orientation = random.choice(fracture_variants)
        rng = None
    else:
        fracture_mode, orientation = fracture_variants[
            validation_sample["variant_index"] % len(fracture_variants)
        ]
        rng = np.random.RandomState(validation_sample["seed"])

    tmode = fracture_mode == "T"
    method_label = 'T-fracture rule-based model' if tmode else 'X-fracture rule-based model'

    failure_time_scale = 10
    single_value = 100
    while single_value == 100:
        data, target, single_value = load_data_rule_based(
            self.n,
            self.m,
            self.numfractures,
            self.numtimesteps,
            failure_time_scale = failure_time_scale,
            tmode=tmode,
            rng=rng,
        )
        single_value /= failure_time_scale

    if orientation == "vertical":
        data = data.transpose(0, 1)
        target = target.transpose(0, 1)

    metadata = build_sample_metadata(
        dataset_key="rule_based",
        method_label=method_label,
        orientation=orientation,
        mat=material,
        is_unstructured=False,
        fracture_mode=fracture_mode,
        tmode=tmode,
    )
    if validation_sample is not None:
        metadata.update(
            validation_index=validation_sample["validation_index"],
            validation_sample_index=validation_sample["sample_index"],
            validation_variant_index=validation_sample["variant_index"],
            validation_seed=validation_sample["seed"],
        )

    return data, target, single_value, metadata
