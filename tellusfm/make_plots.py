import os
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pytorch_lightning.loggers import TensorBoardLogger

mpl.rcParams['figure.max_open_warning'] = 0

def _save_validation_patterns(self, inputs, mesh, targets, predictions, batch_idx, metadata):

    log_dir = self.trainer.loggers[2].log_dir
    val_dir = os.path.join(log_dir, "validation_samples")
    os.makedirs(val_dir, exist_ok=True)

    rank = self.trainer.global_rank
    num_nodes = self.trainer.world_size
    val_idx = int(metadata.get("validation_index", batch_idx * num_nodes + rank))

    data = inputs.cpu().detach().numpy()
    data = data.squeeze()
    target = targets.cpu().detach().numpy()
    target = target.squeeze()
    prediction = torch.sigmoid(predictions).cpu().detach().numpy()
    prediction = prediction.squeeze()
    if not metadata["is_unstructured"]:
        side_length = int(np.sqrt(data.shape[0]))

        data = data[:,:1].reshape(side_length, side_length, 1)
        target = target.reshape(side_length, side_length, 1)
        prediction = prediction.reshape(side_length, side_length, 1)
    else:
        data = data[:, :1]
        mesh = mesh.cpu().detach().numpy()

    residual = target - prediction
    sample_label = metadata["sample_label"].replace(os.sep, "_").replace(" ", "_")

    fig = plt.figure(figsize=(20, 8), dpi=400)
    gs = fig.add_gridspec(2, 4, height_ratios=[3, 1], hspace=0.25)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_meta = fig.add_subplot(gs[1, :])

    if not metadata["is_unstructured"]:
        axs[0].imshow(data.squeeze())
        axs[0].set_title("Input")
        axs[0].axis('off')

        axs[1].imshow(target.squeeze())
        axs[1].set_title("Target")
        axs[1].axis('off')

        axs[2].imshow(prediction.squeeze())
        axs[2].set_title("Predicted")
        axs[2].axis('off')

        im = axs[3].imshow(residual.squeeze(), cmap='seismic', vmin=-1, vmax=1)
        axs[3].set_title("Residual (Target-Predicted)")
        axs[3].axis('off')
        fig.colorbar(im, ax=axs[3], fraction=0.046, pad=0.04)
    else:
        custom_cmap = plt.get_cmap('inferno_r')
        cmap_array = custom_cmap(np.linspace(0, 1, 256))
        cmap_array[0] = np.array([1.0, 1.0, 1.0, 1.0])
        custom_cmap = mcolors.ListedColormap(cmap_array)

        scatter1 = axs[0].scatter(mesh[:, 0], mesh[:, 1], c=data, s=0.7, cmap=custom_cmap)
        fig.colorbar(scatter1, ax=axs[0], fraction=0.046, pad=0.04)
        axs[0].set_title('Input')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_aspect('equal', 'box')

        scatter2 = axs[1].scatter(mesh[:, 0], mesh[:, 1], c=target, s=0.7, cmap=custom_cmap, vmax=1)
        fig.colorbar(scatter2, ax=axs[1], fraction=0.046, pad=0.04)
        axs[1].set_title('Target')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_aspect('equal', 'box')

        scatter3 = axs[2].scatter(mesh[:, 0], mesh[:, 1], c=prediction, s=0.7, cmap=custom_cmap)
        fig.colorbar(scatter3, ax=axs[2], fraction=0.046, pad=0.04)
        axs[2].set_title('Predicted')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_aspect('equal', 'box')

        scatter4 = axs[3].scatter(mesh[:, 0], mesh[:, 1], c=residual, s=0.7, cmap='seismic', vmin=-1, vmax=1)
        fig.colorbar(scatter4, ax=axs[3], fraction=0.046, pad=0.04)
        axs[3].set_title('Residual (Target-Predicted)')
        axs[3].set_xticks([])
        axs[3].set_yticks([])
        axs[3].set_aspect('equal', 'box')

    ax_meta.axis('off')
    lines = [
        f"Simulation type: {metadata.get('method_label', 'unknown')}",
        f"Material: {metadata.get('mat', 'unknown')}",
        f"BC: {metadata.get('orientation', 'unknown')}"
    ]
    meta = " | ".join(lines)
    fig.subplots_adjust(bottom=0.25, top=0.97)
    ax_meta.text(0.5, 1.1, meta, ha="center", va="center", fontsize=10)
    plt.savefig(f"{val_dir}/{val_idx:04d}_validation_{sample_label}.png")

    # Save the same validation index to tensorboard every epoch.
    if _log_validation_plot(self, metadata, val_idx):
        _save_val_pattern_tb(self, fig, sample_label, val_idx, metadata)
    plt.close()


def _log_validation_plot(self, metadata, val_idx):
    if not hasattr(self, "_chosen_samples"):
        self._chosen_samples = {}

    dataset_key = str(metadata.get("dataset_key", "unknown")).replace(" ", "_")
    sample_label = str(metadata.get("sample_label", "unknown"))
    sample_key = (val_idx, sample_label)

    chosen = self._chosen_samples.setdefault(dataset_key, sample_key)
    return chosen == sample_key


def _get_tensorboard_logger(self):
    logger = self.logger
    if isinstance(logger, TensorBoardLogger):
        return logger

    if hasattr(logger, "loggers"):
        for child_logger in logger.loggers:
            if isinstance(child_logger, TensorBoardLogger):
                return child_logger

    return None

def _save_val_pattern_tb(self, fig, val_name, val_idx, metadata):
    tb_logger = _get_tensorboard_logger(self)
    if tb_logger is None or not hasattr(tb_logger, "experiment"):
        return

    dataset_key = str(metadata.get("dataset_key", "unknown")).replace(" ", "_")
    tb_logger.experiment.add_figure(
        f"validation/{dataset_key}/{val_idx:04d}_{val_name}",
        fig,
        global_step=self.global_step,
    )


def _plot_losses(self):
    val_dir = self.trainer.loggers[2].log_dir
    os.makedirs(val_dir, exist_ok=True)

    def average_losses(losses, avg_num):
        return [np.mean(losses[i:i+avg_num]) for i in range(0, len(losses), avg_num)]

    def normalize_losses(losses, target_length):
        x_original = np.linspace(0, 1, len(losses))
        x_target = np.linspace(0, 1, target_length)
        return np.interp(x_target, x_original, losses)

    n = 100
    train_losses_avg = average_losses(self.train_losses, n)
    val_losses_avg = average_losses(self.val_losses, n)

    if not train_losses_avg or not val_losses_avg:
        #print("Averaged train or validation losses are empty. Skipping plot.")
        return

    min_loss = min(min(train_losses_avg), min(val_losses_avg))
    max_loss = max(max(train_losses_avg), max(val_losses_avg))
    buffer = 0.05 * (max_loss - min_loss)  # Add 5% buffer
    min_loss -= buffer
    max_loss += buffer

    max_length = max(len(train_losses_avg), len(val_losses_avg))
    train_losses_normalized = normalize_losses(train_losses_avg, max_length)
    val_losses_normalized = normalize_losses(val_losses_avg, max_length)

    # Plot training and validation losses
    plt.figure(dpi=400, figsize=(10, 6))
    plt.plot(train_losses_normalized, label="Training Loss", linewidth=2)
    plt.plot(val_losses_normalized, label="Validation Loss", linewidth=2)
    plt.xlabel("Normalized Number of Samples")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss [Epoch {self.current_epoch}]")
    plt.legend()
    plt.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(os.path.join(val_dir, f"losses_norm.png"))
    plt.close()

def _save_test_patterns(self, inputs, mesh, targets, predictions, batch_idx, metadata, output_root=None):

    # Determine base output directory without relying on Trainer (which may be absent)
    base_dir = None
    try:
        if output_root is not None:
            base_dir = output_root
        else:
            # Try trainer-backed logger location first (may raise if no trainer attached)
            try:
                base_dir = self.trainer.loggers[2].log_dir
            except Exception:
                base_dir = getattr(self.hparams, 'path_working_directory', '.')
    except Exception:
        base_dir = getattr(self.hparams, 'path_working_directory', '.')

    test_dir = os.path.join(base_dir, "test_samples")
    os.makedirs(test_dir, exist_ok=True)

    data = inputs.cpu().detach().numpy()
    data = data.squeeze()
    target = targets.cpu().detach().numpy()
    target = target.squeeze()
    prediction = torch.sigmoid(predictions).cpu().detach().numpy()
    prediction = prediction.squeeze()

    if not metadata.get("is_unstructured", False):
        side_length = int(np.sqrt(data.shape[0]))

        data = data[:,:1].reshape(side_length, side_length, 1)
        target = target.reshape(side_length, side_length, 1)
        prediction = prediction.reshape(side_length, side_length, 1)
    else:
        data = data[:, :1]
        mesh = mesh.cpu().detach().numpy()

    # Create a 1x4 layout: Input | Target | Predicted | Metadata
    fig, axs = plt.subplots(1, 4, figsize=(16, 4), dpi=400)
    if not metadata.get("is_unstructured", False):
        axs[0].imshow(data.squeeze())
        axs[0].set_title("Input")
        axs[0].axis('off')

        axs[1].imshow(target.squeeze())
        axs[1].set_title("Target")
        axs[1].axis('off')

        axs[2].imshow(prediction.squeeze())
        axs[2].set_title(f"Predicted")
        axs[2].axis('off')
    else:
        custom_cmap = plt.get_cmap('inferno_r')
        cmap_array = custom_cmap(np.linspace(0, 1, 256))
        cmap_array[0] = np.array([1.0, 1.0, 1.0, 1.0])
        custom_cmap = mcolors.ListedColormap(cmap_array)

        scatter1 = axs[0].scatter(mesh[:, 0], mesh[:, 1], c=data, s=0.7, cmap=custom_cmap)
        fig.colorbar(scatter1, ax=axs[0])
        axs[0].set_title('Input')
        axs[0].set_xticks([])
        axs[0].set_yticks([])
        axs[0].set_aspect('equal', 'box')

        scatter2 = axs[1].scatter(mesh[:, 0], mesh[:, 1], c=target, s=0.7, cmap=custom_cmap, vmax=1)
        fig.colorbar(scatter2, ax=axs[1])
        axs[1].set_title('Target')
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        axs[1].set_aspect('equal', 'box')

        scatter3 = axs[2].scatter(mesh[:, 0], mesh[:, 1], c=prediction, s=0.7, cmap=custom_cmap)
        fig.colorbar(scatter3, ax=axs[2])
        axs[2].set_title(f'Predicted')
        axs[2].set_xticks([])
        axs[2].set_yticks([])
        axs[2].set_aspect('equal', 'box')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # Compute simple per-sample metrics (flattened)
    try:
        t_arr = np.ravel(target.astype(np.float32))
    except Exception:
        t_arr = np.ravel(target)
    try:
        p_arr = np.ravel(prediction.astype(np.float32))
    except Exception:
        p_arr = np.ravel(prediction)

    # Ensure equal length
    min_len = min(t_arr.size, p_arr.size)
    if min_len == 0:
        mse = float('nan')
        mae = float('nan')
    else:
        mse = float(np.mean((t_arr[:min_len] - p_arr[:min_len]) ** 2))
        mae = float(np.mean(np.abs(t_arr[:min_len] - p_arr[:min_len])))

    sample_label = metadata.get("sample_label", str(batch_idx)).replace(os.sep, "_").replace(" ", "_")
    # Add metadata and metrics text below the figure
    meta_lines = [f"sample: {sample_label}"]
    for k in ("dataset_key", "sample_index", "method_label", "embedding_index", "orientation", "mat", "bc"):
        if k in metadata:
            meta_lines.append(f"{k}: {metadata[k]}")
    meta_lines.append(f"MSE: {mse:.6g}")
    meta_lines.append(f"MAE: {mae:.6g}")

    # Render metadata/metrics in the 4th panel
    try:
        axs[3].axis('off')
        textstr = "\n".join(meta_lines)
        axs[3].text(0.5, 0.5, textstr, ha='center', va='center', fontsize=8, wrap=True)
    except Exception:
        # Fallback to figure text if axs[3] not available
        fig.text(0.5, 0.01, " | ".join(meta_lines), ha='center', fontsize=8)

    plt.savefig(f"{test_dir}/{batch_idx:04d}_test_{sample_label}.png")
    plt.close()