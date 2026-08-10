import numpy as np
import math
import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from skimage.metrics import structural_similarity as ssim

from tellusfm.model import Encoder, Decoder
from tellusfm.data_readers.rule_based import *
from tellusfm.helper_functions import get_sample_prefix
from tellusfm.senseiver_loader import create_senseiver_dataloader

class Senseiver(pl.LightningModule):

    from tellusfm.make_plots import _save_validation_patterns, _plot_losses, _save_test_patterns

    def __init__(self,**kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(self.hparams)

        self.train_losses = []
        self.val_losses = []
        #self.single_loss_scale = getattr(self.hparams, "single_loss_scale", 10000)
        self.validation_plot_interval = getattr(self.hparams, "validation_plot_interval", 1)

        pos_encoder_ch = self.hparams.space_bands*len(self.hparams.image_size)*2

        self.encoder = Encoder(
            input_ch = self.hparams.im_ch+pos_encoder_ch,
            preproc_ch = self.hparams.enc_preproc_ch,
            num_latents = self.hparams.num_latents,
            num_latent_channels = self.hparams.enc_num_latent_channels,
            num_layers = self.hparams.num_layers,
            num_cross_attention_heads = self.hparams.num_cross_attention_heads,
            num_self_attention_heads = self.hparams.enc_num_self_attention_heads,
            num_self_attention_layers_per_block = self.hparams.num_self_attention_layers_per_block,
            dropout = self.hparams.dropout,
        )

        self.decoder_1 = Decoder(
            ff_channels = pos_encoder_ch,
            preproc_ch = self.hparams.dec_preproc_ch,  # latent bottleneck
            num_latent_channels = self.hparams.dec_num_latent_channels,  # hyperparam
            latent_size = self.hparams.latent_size,  # collapse from n_sensors to 1
            num_output_channels = self.hparams.im_ch,
            num_latents = self.hparams.num_latents,
            llm_embedding_dim = getattr(self.hparams, "llm_embedding_dim", 4096),
            num_cross_attention_heads = self.hparams.dec_num_cross_attention_heads,
            dropout = self.hparams.dropout,
        )

        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        self.num_params = sum([np.prod(p.size()) for p in model_parameters])
        print(f'\nThe model has {self.num_params} params \n')

    def forward(self, input_values, coords, llm):

        out = self.encoder(input_values)
        return self.decoder_1(out, coords, llm)

    @staticmethod
    def _compute_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
        if img1.shape != img2.shape:
            raise ValueError("SSIM inputs must have the same shape")

        gt = img1.detach().cpu().numpy()
        pred = img2.detach().cpu().numpy()

        def compute_single_ssim(a: np.ndarray, b: np.ndarray) -> float:
            data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
            data_range = max(data_range, 1e-8)
            return float(ssim(a, b, data_range=data_range))

        if gt.ndim == 3:
            values = [compute_single_ssim(g, p) for g, p in zip(gt, pred)]
            return float(np.mean(values))

        return compute_single_ssim(gt, pred)

    def test(self, data_config, checkpoints_config, rule_based_config, test_loader=None):
        """Run a simple evaluation pass using the current model and dataset config."""
        self.eval()

        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.to(device)
        run_type = str(checkpoints_config.get("run_type", "test")).lower() if isinstance(checkpoints_config, dict) else "test"

        model_config = {
            "sim_type": getattr(self.hparams, "sim_type", "RB"),
            "sim_weights": getattr(self.hparams, "sim_weights", {getattr(self.hparams, "sim_type", "RB"): 1.0}),
            "run_type": run_type,
            "seed": getattr(self.hparams, "seed", 0),
            "space_bands": getattr(self.hparams, "space_bands", 32),
            "batch_pixels": getattr(self.hparams, "batch_pixels", 10000),
            "number_samples_per_epoch": getattr(self.hparams, "number_samples_per_epoch", 1000),
            "num_workers": getattr(self.hparams, "num_workers", 0),
            "im_ch": getattr(self.hparams, "im_ch", 1),
            "num_embedding_variants": getattr(self.hparams, "num_embedding_variants", 20),
            "test_bc": getattr(self.hparams, "test_bc", "auto"),
        }

        embeddings_config = {
            "emb_rule_based": getattr(self.hparams, "emb_rule_based", None),
            "emb_phase": getattr(self.hparams, "emb_phase", None),
            "emb_hoss": getattr(self.hparams, "emb_hoss", None),
            "num_embedding_variants": getattr(self.hparams, "num_embedding_variants", 20),
        }

        # Prefer rule_based_config if provided, otherwise fall back to hparams or checkpoints_config
        rule_based_params = None
        if rule_based_config is not None and isinstance(rule_based_config, dict):
            rule_based_params = rule_based_config
        elif hasattr(self.hparams, "n") and hasattr(self.hparams, "m"):
            rule_based_params = {
                "n": getattr(self.hparams, "n"),
                "m": getattr(self.hparams, "m"),
                "numfractures": getattr(self.hparams, "numfractures", None),
                "numtimesteps": getattr(self.hparams, "numtimesteps", None),
                "num_sims": getattr(self.hparams, "num_sims", 200),
                "material": getattr(self.hparams, "material", "pbx"),
            }
        elif rule_based_params is None and isinstance(checkpoints_config, dict):
            rule_based_params = checkpoints_config.get("rule_based_params")

        if rule_based_params is not None:
            missing = [k for k in ("n", "m", "numfractures", "numtimesteps") if rule_based_params.get(k) is None]
            if missing:
                raise ValueError(f"Missing rule based parameters for test: {missing}")

        if test_loader is None:
            test_loader = create_senseiver_dataloader(
                False,
                model_config=model_config,
                embeddings_config=embeddings_config,
                data_config=data_config,
                rule_based_params=rule_based_params,
                run_type=run_type,
            )

        # Progress helpers: try to use tqdm if available, otherwise fall back to periodic prints
        try:
            from tqdm import tqdm
            use_tqdm = True
            total_batches = len(test_loader)
        except Exception:
            tqdm = None
            use_tqdm = False
            try:
                total_batches = len(test_loader)
            except Exception:
                total_batches = None

        print(f"Starting test: device={device}, total_batches={total_batches}")

        summary = {
            "loss_target": 0.0,
            #"loss_single": 0.0,
            "mae_sum": 0.0,
            "mae_count": 0,
            "mse_sum": 0.0,
            "mse_sample_sum": 0.0,
            "ssim_sum": 0.0,
            "ssim_count": 0,
            "mae_sample_sum": 0.0,
            "samples": 0,
        }

        with torch.no_grad():
            iterator = (tqdm(test_loader, desc="Testing", total=total_batches) if use_tqdm else test_loader)
            for batch_idx, batch in enumerate(iterator):
                input_values, target_values, coords, mesh, single_value, llm, metadata = batch
                input_values = input_values.to(device)
                target_values = target_values.to(device)
                coords = coords.to(device)
                llm = llm.to(device)
                #single_value = single_value.to(device)

                pred_values, _ = self(input_values, coords, llm)
                pred_probs = torch.sigmoid(pred_values)

                # Save visualizations of test samples
                try:
                    out_root = getattr(self.hparams, 'path_working_directory', '.')
                    self._save_test_patterns(input_values, mesh, target_values, pred_values, batch_idx, metadata, output_root=out_root)
                except Exception as e:
                    print(f"Warning: failed to save test plot for batch {batch_idx}: {e}")

                loss_target = F.mse_loss(pred_probs, target_values, reduction="sum")
                abs_error = torch.abs(pred_probs - target_values)
                sample_mae = float(abs_error.mean().item())
                sample_mse = float(((target_values - pred_probs) ** 2).mean().item())

                image_size = getattr(self.hparams, "image_size", None)
                if image_size is not None and len(image_size) >= 2:
                    height, width = image_size[:2]
                    pred_image = pred_probs.view(-1, height, width)
                    target_image = target_values.view(-1, height, width)
                    batch_ssim = self._compute_ssim(target_image, pred_image)
                else:
                    batch_ssim = float("nan")
    
                summary["loss_target"] += loss_target.item()
                summary["mae_sum"] += abs_error.sum().item()
                summary["mae_count"] += abs_error.numel()
                summary["mse_sum"] += loss_target.item()
                summary["mse_sample_sum"] += sample_mse
                summary["ssim_sum"] += batch_ssim
                summary["ssim_count"] += 1
                summary["mae_sample_sum"] += sample_mae
                summary["samples"] += 1

                material = metadata.get("mat", "unknown")
                bc = metadata.get("bc") or metadata.get("orientation", "unknown")
                sample_label = metadata.get("sample_label", f"batch_{batch_idx}")
                sample_msg = (
                    f"Test sample {batch_idx}: material={material}, "
                    f"bc={bc}, sample={sample_label}, mae={sample_mae:.6g}"
                )
                if use_tqdm:
                    tqdm.write(sample_msg)
                else:
                    print(sample_msg)

                # Periodic feedback when tqdm isn't available
                if not use_tqdm and (batch_idx + 1) % 10 == 0:
                    avg_loss_target = summary["loss_target"] / max(1, summary["samples"])
                    print(f"Processed {batch_idx+1}{('/' + str(total_batches)) if total_batches else ''} batches — avg_loss_target={avg_loss_target:.4f}")

        results = {
            "loss_target": summary["loss_target"],
            #"loss_single": summary["loss_single"],
            "samples": summary["samples"],
            "mae": summary["mae_sum"] / max(1, summary["mae_count"]),
            "mse": summary["mse_sum"] / max(1, summary["mae_count"]),
            "mse_per_sample": summary["mse_sample_sum"] / max(1, summary["samples"]),
            "ssim": summary["ssim_sum"] / max(1, summary["ssim_count"]),
            "mae_per_sample": summary["mae_sample_sum"] / max(1, summary["samples"]),
        }

        print("Test summary:", results)
        return results

    def training_step(self, batch, batch_idx):
        input_values, target_values, coords, mesh, single_value, llm, metadata = batch
        # forward
        pred_values, _ = self(input_values, coords, llm)

        # Losses
        loss_target = F.mse_loss(F.sigmoid(pred_values), target_values, reduction='sum')

        self.train_losses.append(loss_target.detach().cpu().item())
        loss = loss_target

        # Log metrics via Lightning so they are forwarded to all loggers
        self.log("train/loss_target", loss_target, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        prefix = get_sample_prefix(metadata, train = True)
        self.log(f"train/loss_target_{prefix}", loss_target, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log("train/lr", lr, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)

        return loss


    def validation_step(self, batch, batch_idx):
        input_values, target_values, coords, mesh, single_value, llm, metadata = batch
        pred_values, _ = self(input_values, coords, llm)

        # Losses
        val_loss_target = F.mse_loss(F.sigmoid(pred_values), target_values, reduction='sum')
        self.val_losses.append(val_loss_target.detach().cpu().item())

        val_loss = val_loss_target

        # Log validation metrics via Lightning so they are forwarded to all loggers
        self.log("validation/loss_target", val_loss_target, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("validation/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        prefix = get_sample_prefix(metadata, train=False)

        self.log(f"validation/loss_target_{prefix}", val_loss_target, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        # Save visualizations
        if batch_idx % self.validation_plot_interval == 0:
           self._save_validation_patterns(input_values, mesh, target_values, pred_values, batch_idx, metadata)


        return val_loss

    def on_validation_epoch_end(self, outputs=None):
        self._plot_losses()

    def configure_optimizers(self):
        # Freezing the trainable parameters
        # for name, param in self.named_parameters():
        #     param.requires_grad = False  # Freeze everything by default

        # Only unfreeze single-value parameters
        # for name, param in self.named_parameters():
        #     if (
        #         "output_single" in name or
        #         "preproc_single" in name or
        #         "postproc_single" in name
        #         ):
        #         param.requires_grad = True

        # Printing the parameters:      
        # for name, param in self.named_parameters():
        #      print(f"{name} — requires_grad={param.requires_grad}")
        # trainable_params = 0
        # for name, param in self.named_parameters():
        #   if param.requires_grad:
        #     layer_count = param.numel()
        #     trainable_params += layer_count
        #     print(f"{name}: {layer_count} params")
        # print(f"Total trainable parameters: {trainable_params}")

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = CosineWithWarmupLR(optimizer, training_steps=self.hparams.max_steps, warmup_steps=self.hparams.warmup_steps)
        if self.hparams.warmup_steps==0:
           return {"optimizer": optimizer}
        else:
           return {
                 "optimizer": optimizer,
                 "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
            }

class CosineWithWarmupLR(LambdaLR):
    def __init__(
        self,
        optimizer: Optimizer,
        training_steps: int = 0,
        warmup_steps: int = 0,
        num_cycles: float = 0.5,
        min_fraction: float = 0.0,
        last_epoch: int = -1,
    ):
        # Can be updated after instantiation
        self.training_steps = training_steps

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, self.training_steps - warmup_steps))
            return min_fraction + max(
                0.0, 0.5 * (1.0 - min_fraction) * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
            )

        super().__init__(optimizer, lr_lambda, last_epoch=last_epoch)
