import os
import math
import h5py
import random
import re
from pathlib import Path
import torch 
from torch.utils.data import DataLoader, Dataset
from tellusfm.helper_functions import prepare_model_inputs
from tellusfm.positional import apply_positional_encoder

class SenseiverLoaderTraining(Dataset):

    from tellusfm.data_readers.rule_based import _load_rule_based
    from tellusfm.data_readers.phase_field import _load_phase_field 
    from tellusfm.data_readers.hoss import _load_hoss

    SUPPORTED_SIM_TYPES = {"RB", "PHASE", "HOSS"}

    def __init__(self, train, model_config, embeddings_config, data_config, rule_based_params = None, run_type = None):

        self.mode = train
        self.run_type = str(run_type or model_config.get("run_type", "train" if train else "val")).lower()
        self.is_test = not self.mode and self.run_type == "test"
        self.sim_types, self.sim_weights = self._build_sim_mix(model_config)

        print("RB parameters: ", rule_based_params)
        if "RB" in self.sim_types:
            if rule_based_params is None:
                raise ValueError("rule_based_params must be provided when using rule based data")
            self.n = rule_based_params['n']
            self.m = rule_based_params['m']
            self.numfractures = rule_based_params['numfractures']
            self.numtimesteps = rule_based_params['numtimesteps']
            self.rule_based_num_sims = rule_based_params.get("num_sims", 200)
            self.rule_based_material = rule_based_params.get("material", "pbx")

        self.space_bands = model_config['space_bands']
        self.hdf5_files = {}
        self.seed = int(model_config.get("seed", 0))

        self.batch_pixels = model_config['batch_pixels']
        self.number_samples_per_epoch = model_config['number_samples_per_epoch']
        self.val_samples_per_sim = int(model_config.get("val_samples_per_sim", 2))
        if self.val_samples_per_sim < 1:
            raise ValueError("model_config['val_samples_per_sim'] must be at least 1")
        self.test_bc = self._normalize_test_bc(model_config.get("test_bc", "auto"))
        self.num_embedding_variants = embeddings_config.get("num_embedding_variants", 20)
        if self.num_embedding_variants < 1:
            raise ValueError("embeddings_config['num_embedding_variants'] must be at least 1")
        self.unstructured_mesh_scale = data_config.get("unstructured_mesh_scale", 1000.0)

        self.path_emb_by_sim = {
            "RB": embeddings_config.get("emb_rule_based"),
            "PHASE": embeddings_config.get("emb_phase"),
            "HOSS": embeddings_config.get("emb_hoss"),
        }

        if self.mode:
            path_phase = data_config.get("path_phase")
            path_hoss = data_config.get("path_hoss")
        elif self.is_test:
            path_phase = data_config.get("path_test")
            path_hoss = data_config.get("path_test")
        else:
            path_phase = data_config.get("path_val_phase")
            path_hoss = data_config.get("path_val_hoss")

        self.path_data_by_sim = {
            "RB": None,
            "PHASE": path_phase,
            "HOSS": path_hoss,
        }

        self.simulation_lists = {}
        self.num_sims_by_sim = {}
        for sim_type in self.sim_types:
            self._load_simulation_list(sim_type)

        self.num_sims = max(self.num_sims_by_sim.values())
        print(f"Total simulations across all types: {self.num_sims_by_sim.values()}")
        self.test_samples = self._build_test_samples() if self.is_test else []
        self.validation_samples = [] if self.mode or self.is_test else self._build_validation_samples()
        self._activate_sim(self.sim_types[0])

        im_ch = model_config.get("im_ch", 1)
        for sim_type in self.sim_types:
            if sim_type == "RB":
                image_size = [self.n, self.m]
                break
            if sim_type == "PHASE":
                fracture_key = self.simulation_lists["PHASE"][0]
                with h5py.File(self.path_data_by_sim["PHASE"], 'r') as hdf5_file:
                    frac_data = hdf5_file[fracture_key]
                    initial_name = next(
                        (name for name in frac_data.keys() if name.endswith("fracture initial")),
                        None,
                    )
                    if initial_name is None:
                        raise KeyError(f"No initial fracture dataset found for {fracture_key}")
                    image_size = list(frac_data[initial_name].shape)
                break
        else:
            image_size = model_config.get("image_size", [1, 1])

        model_config['image_size'] = image_size
        model_config['im_ch'] = im_ch
        model_config['num_batches'] = max(1, math.ceil(math.prod(image_size) / self.batch_pixels))
        print(f'{model_config["num_batches"]} Batches of data per epoch\n')

        print(f"Mode: {self.mode}")
        mode_type = "Training" if self.mode else ("Test" if self.is_test else "Validation")
        print(f"Simulation mix: {dict(zip(self.sim_types, self.sim_weights))}")
        for sim_type in self.sim_types:
            print(f"{sim_type} embeddings: {self.path_emb_by_sim[sim_type]}")
            print(f"{sim_type} {mode_type} Dataset: {self.path_data_by_sim[sim_type]}")
            print(f"There are {self.num_sims_by_sim[sim_type]} {sim_type} simulations in the {mode_type} set")


    def _build_sim_mix(self, model_config):
        if model_config["sim_type"] == "MIXED":
            sim_weights = model_config.get("sim_weights")
            if not isinstance(sim_weights, dict):
                raise ValueError("model_config['sim_weights'] must be a dictionary, e.g. {'RB': 0.2, 'PHASE': 0.8}")
        else:
            sim_weights = {model_config["sim_type"]: 1.0}

        sim_types = []
        weights = []
        valid_sim_types = {"RB", "PHASE", "HOSS"}
        for sim_type, weight in sim_weights.items():
            if sim_type not in valid_sim_types:
                raise ValueError(f"Unknown sim_to_run: {sim_type}")
            weight = float(weight)
            if weight < 0:
                raise ValueError(f"Simulation weight for {sim_type} must be non-negative")
            if weight == 0:
                continue
            sim_types.append(sim_type)
            weights.append(weight)

        if not sim_types:
            raise ValueError("At least one simulation weight must be greater than zero")

        return sim_types, weights


    def _load_simulation_list(self, sim_type):
        if sim_type == "RB":
            self.simulation_lists[sim_type] = None
            self.num_sims_by_sim[sim_type] = self.rule_based_num_sims
            return

        if not self.path_data_by_sim[sim_type]:
            path_name = "path_test" if self.is_test else f"path_{sim_type.lower()}"
            if not self.mode and not self.is_test:
                path_name = f"path_val_{sim_type.lower()}"
            raise ValueError(f"No {self.run_type} dataset path configured for {sim_type}; set datasets.{path_name}")

        with h5py.File(self.path_data_by_sim[sim_type], 'r') as f:
            all_keys = list(f.keys())
        self.simulation_lists[sim_type] = all_keys[:]
        self.num_sims_by_sim[sim_type] = len(self.simulation_lists[sim_type])
        if self.num_sims_by_sim[sim_type] == 0:
            raise ValueError(
                f"No {sim_type} samples found in {self.path_data_by_sim[sim_type]}. "
                "The HDF5 file opened successfully, but it contains no top-level "
                "sample groups or external links. Regenerate this link file from "
                "a non-empty source directory, or skip it in test sweeps."
            )

    def _build_validation_samples(self):
        total_samples = self.num_sims * self.val_samples_per_sim
        sim_sequence = self._deterministic_sim_sequence(total_samples)
        sample_positions = {sim_type: 0 for sim_type in self.sim_types}
        validation_samples = []

        for validation_index, sim_type in enumerate(sim_sequence):
            sample_position = sample_positions[sim_type]
            sample_positions[sim_type] += 1
            sample_spec = self._build_validation_sample_spec(sim_type, sample_position)
            sample_spec["validation_index"] = validation_index
            validation_samples.append(sample_spec)

        return validation_samples

    def _build_test_samples(self):
        sim_sequence = self._deterministic_sim_sequence(self.num_sims)
        sample_positions = {sim_type: 0 for sim_type in self.sim_types}
        test_samples = []
        test_index = 0

        for sim_type in sim_sequence:
            sample_position = sample_positions[sim_type]
            sample_positions[sim_type] += 1
            sample_spec = self._build_validation_sample_spec(sim_type, sample_position)

            for orientation in self._test_orientations_for_sample(sim_type, sample_spec):
                test_spec = sample_spec.copy()
                test_spec["validation_index"] = test_index
                test_spec["validation_sample_index"] = sample_position
                if orientation is not None:
                    test_spec["test_bc"] = orientation
                test_samples.append(test_spec)
                test_index += 1

        return test_samples

    @staticmethod
    def _normalize_test_bc(test_bc):
        normalized = str(test_bc or "auto").strip().lower()
        allowed = {"auto", "vertical", "horizontal", "combined", "all"}
        if normalized not in allowed:
            raise ValueError(
                "model_config['test_bc'] must be one of "
                f"{sorted(allowed)}; got {test_bc!r}"
            )
        return normalized

    @staticmethod
    def _phase_key_direction_and_bc(fracture_key):
        match = re.match(r"link_(\w+)_([a-z]+)_([a-z]+)_(\d+)", fracture_key)
        if match:
            _, direction, bc, _ = match.groups()
            return direction, bc
        if SenseiverLoaderTraining._phase_key_has_bc_z(fracture_key):
            return "vertical", "z"
        return None, None

    def _test_orientations_for_sample(self, sim_type, sample_spec):
        if sim_type != "PHASE":
            return [None]

        direction, bc = self._phase_key_direction_and_bc(sample_spec.get("fracture_key", ""))
        if bc != "z" or direction == "combined":
            return ["combined"]

        if direction not in {"vertical", "horizontal"}:
            direction = "vertical"

        if self.test_bc == "all":
            return ["vertical", "horizontal"]
        if self.test_bc in {"vertical", "horizontal"}:
            return [self.test_bc]
        return [direction]

    @staticmethod
    def _phase_key_has_bc_z(fracture_key):
        return bool(re.search(r"_z(?:_|$)", fracture_key))

    def _deterministic_sim_sequence(self, total_samples):
        if len(self.sim_types) == 1:
            return [self.sim_types[0]] * total_samples

        weights_by_sim = dict(zip(self.sim_types, self.sim_weights))
        total_weight = sum(self.sim_weights)
        running_weights = {sim_type: 0.0 for sim_type in self.sim_types}
        sim_sequence = []

        for _ in range(total_samples):
            for sim_type in self.sim_types:
                running_weights[sim_type] += weights_by_sim[sim_type]

            selected_sim = max(self.sim_types, key=lambda sim_type: running_weights[sim_type])
            sim_sequence.append(selected_sim)
            running_weights[selected_sim] -= total_weight

        return sim_sequence

    def _build_validation_sample_spec(self, sim_type, sample_position):
        sample_spec = {
            "sim_type": sim_type,
            "sample_position": sample_position,
            "variant_index": sample_position,
        }

        if sim_type == "RB":
            sample_spec["sample_index"] = sample_position
            sample_spec["seed"] = (self.seed + sample_position * 1000003) % (2**32 - 1)
            return sample_spec

        simulation_list = self.simulation_lists[sim_type]
        if not simulation_list:
            raise ValueError(f"No validation samples available for {sim_type}")

        sample_index = sample_position % len(simulation_list)
        sample_spec["sample_index"] = sample_index
        sample_spec["variant_index"] = sample_position // len(simulation_list)
        sample_spec["fracture_key"] = simulation_list[sample_index]
        return sample_spec

    def _get_hdf5_file(self, sim_type):
        path = self.path_data_by_sim[sim_type]
        if path is None:
            raise ValueError(f"No HDF5 path configured for {sim_type}")

        hdf5_file = self.hdf5_files.get(sim_type)
        if hdf5_file is None or not hdf5_file.id.valid:
            self.hdf5_files[sim_type] = h5py.File(path, 'r')
        return self.hdf5_files[sim_type]

    def close_hdf5_files(self):
        for hdf5_file in self.hdf5_files.values():
            try:
                if hdf5_file.id.valid:
                    hdf5_file.close()
            except Exception:
                pass
        self.hdf5_files.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["hdf5_files"] = {}
        return state

    def __del__(self):
        try:
            self.close_hdf5_files()
        except Exception:
            pass

    def _activate_sim(self, sim_type):
        self.sim_to_run = sim_type
        self.path_emb = self.path_emb_by_sim[sim_type]
        self.path_data = self.path_data_by_sim[sim_type]
        self.file_path = self.path_data
        self.simulation_list = self.simulation_lists[sim_type]

    def _select_sim_to_run(self):
        if len(self.sim_types) == 1:
            return self.sim_types[0]
        return random.choices(self.sim_types, weights=self.sim_weights, k=1)[0]

    def _get_embedding_filename(self, metadata):
        return (
            f'{metadata["mat"]}_{metadata["method_label"]}_'
            f'{metadata["orientation"]}_{metadata["embedding_index"]}.torch'
        )

    def _sample_embedding_index(self, sample_idx=None):
        if self.mode:
            return torch.randint(0, self.num_embedding_variants, (1,)).item()
        else:
            return sample_idx % self.num_embedding_variants
            #return 0  # Fixed embedding for deterministic validation

    def __len__(self):
        if self.mode:
            return self.number_samples_per_epoch
        if self.is_test:
            return len(self.test_samples)
        return len(self.validation_samples)

    def _get_embedding(self,embedding_filename):
        file_path = os.path.join(self.path_emb, embedding_filename.lstrip(os.sep))
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Missing embedding file for sample: {file_path}")
        llm = torch.load(file_path, map_location=torch.device('cpu'))
        llm = llm.to(torch.float32)
        return llm 

    def __getitem__(self, idx):
        idx = int(idx)
        if self.mode:
            sim_to_run = self._select_sim_to_run()
            self._active_validation_sample = None
        else:
            if self.is_test:
                validation_sample = self.test_samples[idx]
            else:
                validation_sample = self.validation_samples[idx]
            sim_to_run = validation_sample["sim_type"]
            self._active_validation_sample = validation_sample

        self._activate_sim(sim_to_run)

        if sim_to_run == "RB":
            mesh = None 
            data, target, single_value, metadata = self._load_rule_based()
            # Apply encoder
            *image_size, im_ch = data.shape
            pixels = torch.arange(0, math.prod(image_size), step=1)
            coords = apply_positional_encoder(data.shape[0:], self.space_bands, pixels, structured=True)
            
            # Flatten arrays 
            input_values = data.flatten(start_dim=0, end_dim=-2)[pixels,][None,]
            input_values= torch.cat([input_values,coords], axis=-1)
            target_values = target.flatten(start_dim=0, end_dim=-2)[pixels,][None,]
            #single_value = single_value[None,:,None]

            # Embeddings
            num = self._sample_embedding_index(idx)
            metadata["embedding_index"] = num
            embedding_filename = self._get_embedding_filename(metadata)
            llm = self._get_embedding(embedding_filename)

            return input_values, target_values, coords, mesh, single_value, llm, metadata
        
        elif sim_to_run == "PHASE":
            mesh = None
            data, target, single_value, metadata = self._load_phase_field()
            
            *image_size, im_ch = data.shape
            pixels = torch.arange(0, math.prod(image_size), step=1)
            coords = apply_positional_encoder(data.shape[0:], self.space_bands, pixels, structured=True)
            
            input_values = data.flatten(start_dim=0, end_dim=-2)[pixels,][None,]
            input_values= torch.cat([input_values,coords], axis=-1)
            target_values = target.flatten(start_dim=0, end_dim=-2)[pixels,][None,]
            #single_value = single_value[None,:,None]

            num = self._sample_embedding_index(idx)
            metadata["embedding_index"] = num
            embedding_filename = self._get_embedding_filename(metadata)
            llm = self._get_embedding(embedding_filename)

            return input_values, target_values, coords, mesh, single_value, llm, metadata

        elif sim_to_run=='HOSS':
            data, target, mesh, single_value, metadata = self._load_hoss()

            coords = apply_positional_encoder(
                mesh,
                self.space_bands,
                structured=False,
                coordinate_scale=self.unstructured_mesh_scale,
            )
            input_values, target_values, coords, single_value = prepare_model_inputs(
                data,
                target,
                coords,
                single_value,
            )
        
            num = self._sample_embedding_index(idx)
            metadata["embedding_index"] = num
            embedding_filename = self._get_embedding_filename(metadata)
            llm = self._get_embedding(embedding_filename)

            return input_values, target_values, coords, mesh, single_value, llm, metadata
        else:
            raise ValueError(f"Unknown sim_to_run: {sim_to_run}")
   

def create_senseiver_dataloader(train, model_config, embeddings_config, data_config, rule_based_params = None, run_type = None):
    """
    Create a PyTorch DataLoader for the Senseiver dataset.

    This function instantiates and configures a DataLoader for the
    `SenseiverLoaderTraining` dataset used in Darcy flow or PFLOTRAN-based
    DFNWorks simulations. It automatically detects the device backend
    (CUDA, MPS, or CPU) to set optimal DataLoader flags such as
    `pin_memory` and optionally `persistent_workers`.

    Parameters
    ----------
    data_config : dict
        Configuration dictionary passed to the dataset class
        Should include all paths and runtime parameters required
        by `SenseiverLoaderTraining`.
    num_workers : int, optional
        Number of worker processes for background data loading.
        Defaults to 72.
    train : bool, optional
        If True, creates a DataLoader for training (shuffled, persistent workers).
        If False, creates one for validation/testing (no shuffle).
        Defaults to True.

    Returns
    -------
    torch.utils.data.DataLoader
        A DataLoader instance for the given mode (train or validation).

    Notes
    -----
    - On Apple M-series (MPS backend), pinned memory is automatically disabled,
      since it’s unsupported on MPS.
    - Consider setting `persistent_workers=True` for improved epoch-to-epoch
      performance when `num_workers > 0` and dataset fits in memory.
    - `batch_size=None` assumes that the dataset provides pre-batched tensors.

    Example
    -------
    >>> config = {"path_data": "/data/pflotran/output.h5", "mode": "train"}
    >>> train_loader = create_senseiver_dataloader(config, num_workers=32, train=True)
    >>> val_loader = create_senseiver_dataloader(config, num_workers=32, train=False)
    """

    use_mps = torch.backends.mps.is_available()

    # Select dataset class (training/validation mode)
    dataset = SenseiverLoaderTraining(train, model_config, embeddings_config, data_config, rule_based_params, run_type=run_type)

    # Common DataLoader settings
    common_args = dict(
        dataset=dataset,
        batch_size=None,
        num_workers=model_config['num_workers'],
        pin_memory=not use_mps,
    )

    if train:
        # Training loader: shuffled, persistent workers enabled for speed
        return DataLoader(
            shuffle=True,
            persistent_workers=(model_config['num_workers'] > 0 and not use_mps),
            **common_args,
        )
    else:
        # Validation loader: deterministic order, no shuffle
        return DataLoader(
            shuffle=False,
            persistent_workers=(model_config['num_workers'] > 0 and not use_mps),
            **common_args,
        )
