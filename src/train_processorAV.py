import json
from copy import deepcopy
from typing import Tuple

import mlflow
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchmetrics.functional import structural_similarity_index_measure as ssim

from data_handlers import ProcessorAVDataset
from x_net.controller.Processor_av import ProcessorAV
from utils.common import read_json
from torchvision import transforms
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_loss = float("-inf")
best_step_loss = float("inf")
best_step = None
############################################################################################
#                                   Loss Calculation                                       #
############################################################################################


def compute_kernel(x1, x2, kernel_type="rbf", z_var=1.0, eps=1e-7):
    """Computes kernel function for MMD"""
    D = x2.size(-1)
    if kernel_type == "rbf":
        sigma = 2.0 * D * z_var
        return torch.exp(
            -((x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2).mean(-1) / sigma)
        )
    elif kernel_type == "imq":
        C = 2 * D * z_var
        kernel = C / (
            eps + C + (x1.unsqueeze(-2) - x2.unsqueeze(-3)).pow(2).sum(dim=-1)
        )
        return kernel.sum() - kernel.diag().sum()
    else:
        raise ValueError("Undefined kernel type.")


def compute_mmd(z, reg_weight, kernel_type="rbf", z_var=1.0):
    """Computes MMD loss between encoded z and prior"""
    prior_z = torch.randn_like(z)
    prior_z_kernel = compute_kernel(prior_z, prior_z, kernel_type, z_var)
    z_kernel = compute_kernel(z, z, kernel_type, z_var)
    priorz_z_kernel = compute_kernel(prior_z, z, kernel_type, z_var)

    return (
        reg_weight * prior_z_kernel.mean()
        + reg_weight * z_kernel.mean()
        - 2 * reg_weight * priorz_z_kernel.mean()
    )


def mmd_vae_loss_function(recon_x, x, z, epoch, **kwargs):
    """
    Computes the loss for an MMD-VAE:
    - MSE loss
    - SSIM loss
    - VGG perceptual loss
    - MMD regularization loss
    """
    reg_weight = kwargs.get("reg_weight", 1.0)
    kernel_type = kwargs.get("kernel_type", "rbf")
    z_var = kwargs.get("z_var", 1.0)
    beta = kwargs.get("beta", 0.1)

    # MSE loss
    val_mse_loss = F.mse_loss(recon_x, x, reduction="mean")

    recon_batch = (recon_x + 1) / 2.0
    targets = (x + 1) / 2.0

    val_ssim_loss = 1 - ssim(recon_batch, targets).item()

    # Total reconstruction loss
    recon_loss = val_mse_loss + val_ssim_loss

    # MMD loss
    mmd_loss = compute_mmd(z, reg_weight, kernel_type, z_var)

    return recon_loss, mmd_loss * beta


############################################################################################
#                                   Loss Calculation                                       #
############################################################################################


def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    max_val = torch.tensor(max_val, device=pred.device, dtype=pred.dtype)
    return 20 * torch.log10(max_val) - 10 * torch.log10(mse + 1e-8)


def setup_logger_mlflow(experiment_name: str = "train_processorAV") -> mlflow:
    """Sets up MLflow for experiment tracking."""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    return mlflow


def prepare_dataloader(
    data_path, batch_size, shuffle=True, transformation=None
) -> DataLoader:
    """Prepares the DataLoader for the dataset."""
    dataset = ProcessorAVDataset(data_path, transform=transformation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def prepare_model(
    training_params, model_params, device="cuda"
) -> Tuple[ProcessorAV, optim.Optimizer, lr_scheduler._LRScheduler]:
    """Initializes the model, optimizer, and scheduler."""
    vae = ProcessorAV(model_params).to(device)
    optimizer_cls = getattr(optim, training_params["optimizer"], optim.AdamW)
    optimizer = optimizer_cls(vae.parameters(), **training_params["optimizer_params"])

    # Default scheduler parameters if not provided
    scheduler_cls = getattr(
        lr_scheduler, training_params["lr_scheduler"], lr_scheduler.StepLR
    )
    scheduler = scheduler_cls(optimizer, **training_params["lr_scheduler_params"])

    return vae, optimizer, scheduler


def validate_on_batch(vae, val_dataloader, device, loss_args, epoch, verbose=False):
    """
    Runs validation on a single batch for quick checkpoint evaluation during training.

    Returns a dictionary with validation metrics for logging.
    """
    vae.eval()
    val_total_loss = 0
    val_recon_loss_total = 0
    val_mmd_loss_total = 0
    val_ssim_total = 0
    val_psnr_total = 0

    with torch.no_grad():
        for val_batch in val_dataloader:
            input_batch = val_batch.to(device)
            targets = deepcopy(input_batch).to(device)

            # Forward pass
            recon_batch, z = vae(input_batch)
            recon_loss, mmd_loss = mmd_vae_loss_function(
                recon_batch, targets, z, epoch, **loss_args
            )
            loss = recon_loss + mmd_loss

            # Accumulate losses
            val_total_loss += loss.item()
            val_recon_loss_total += recon_loss.item()
            val_mmd_loss_total += mmd_loss.item()

            recon_batch = (recon_batch + 1) / 2.0
            targets = (targets + 1) / 2.0

            val_ssim_total += ssim(recon_batch, targets).item()
            val_psnr_total += psnr(recon_batch, targets).item()

            break  # Only run one batch

    return {
        "step_val_total_loss": val_total_loss,
        "step_val_recon_loss": val_recon_loss_total,
        "step_val_mmd_loss": val_mmd_loss_total,
        "step_val_ssim": val_ssim_total,
        "step_val_psnr": val_psnr_total,
    }


def train_one_epoch(
    vae,
    dataloader,
    optimizer,
    device,
    loss_params,
    epoch_number,
    scheduler,
    val_dataloader=None,
    val_every_n_steps=None,
    scheduler_step=None,
    verbose=False,
) -> Tuple[float, float, float]:
    """Trains the model for one epoch and returns losses."""
    vae.train()
    total_loss = 0
    recon_loss_total = 0
    total_mmd_loss = 0
    for i, batch in enumerate(dataloader):
        input_batch = batch
        targets = deepcopy(input_batch)

        input_batch = input_batch.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        recon_batch, z = vae(input_batch)
        recon_loss, mmd_loss = mmd_vae_loss_function(
            recon_batch, targets, z, epoch_number, **loss_params
        )

        loss = recon_loss + mmd_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        recon_loss_total += recon_loss.item()
        total_mmd_loss += mmd_loss.item()

        if (i + 1) % 10 == 0:
            mlflow.log_metrics(
                {
                    "batch_recon_loss": recon_loss.item(),
                    "batch_mmd_loss": mmd_loss.item(),
                    "batch_total_loss": loss.item(),
                },
                step=i + 1,
            )
            if verbose:
                print(
                    f"Batch {i+1}/{len(dataloader)} - "
                    f"Recon Loss: {recon_loss.item():.6f}, "
                    f"MMD Loss: {mmd_loss.item():.6f}, "
                    f"Total Loss: {loss.item():.6f}"
                )

        if val_every_n_steps and val_dataloader and (i + 1) % val_every_n_steps == 0:
            metrics = validate_on_batch(
                vae, val_dataloader, device, loss_params, epoch_number
            )
            mlflow.log_metrics(metrics, step=epoch_number * len(dataloader) + i + 1)
            current_step = epoch_number * len(dataloader) + i + 1
            current_val_loss = metrics["step_val_total_loss"]
            current_ssim = metrics["step_val_ssim"]
            global best_step_loss
            is_better_step = current_val_loss < best_step_loss

            if is_better_step:
                print(
                    f"[Step {current_step}] New best step validation loss: {current_val_loss:.6f}. SSIM: {current_ssim}. Saving best_step.pt"
                )
                torch.save(vae.state_dict(), "best_step.pt")
                mlflow.log_artifact("best_step.pt")
                best_step_loss = current_val_loss
                best_step = current_step

        if scheduler_step and val_dataloader and (i + 1) % scheduler_step == 0:
            scheduler.step()

    if verbose:
        print(
            f"Epoch finished - Total Loss: {total_loss:.6f}, "
            f"Recon Loss: {recon_loss_total:.6f}, "
            f"MMD Loss: {total_mmd_loss:.6f}"
        )
    return total_loss, recon_loss_total, total_mmd_loss


def train_model(params_path="params.json", verbose=False):
    # Load parameters from a single JSON file
    params = read_json(params_path)
    train_params = params.get("training_parameters", {})
    model_params = params.get("model_parameters", {})

    # Training parameters
    batch_size = train_params.get("batch_size", 32)
    experiment_name = train_params.get("experiment_name", "Default experiment")
    epochs = train_params.get("epochs", 30)
    early_stop_patience = train_params.get("early_stop_patience", 10)
    minimize_metric = train_params.get("early_stop_mode", "min") == "min"
    val_every_n_steps = train_params.get("val_every_n_steps", False)
    schduler_steps = train_params.get("scheduler_step_size", 100)

    if verbose:
        print(f"Setting up MLflow for experiment: {experiment_name}")
    mlflow = setup_logger_mlflow(experiment_name)
    mlflow.log_params({**train_params, **model_params})
    mlflow.log_artifact(params_path)

    # Prepare transformations if specified in the JSON config
    train_transformations = None
    val_transformations = None
    if "transformations" in train_params:
        train_transform_list = []
        val_transform_list = []
        for t_name, t_params in train_params["transformations"].items():
            transform_cls = getattr(transforms, t_name, None)
            params = t_params if isinstance(t_params, dict) else {}
            use_in_val = params.pop("use_in_val", False)
            if transform_cls is not None:
                train_transform_list.append(transform_cls(**params))
                if use_in_val:
                    val_transform_list.append(transform_cls(**params))
            else:
                print(
                    f"Warning: Transformation '{t_name}' not found in torchvision.transforms."
                )
        if train_transform_list:
            train_transformations = transforms.Compose(train_transform_list)
        if val_transform_list:
            val_transformations = transforms.Compose(val_transform_list)

    if verbose:
        print("Preparing dataloaders...")
    train_dataloader = prepare_dataloader(
        train_params["train_data_path"],
        batch_size,
        shuffle=True,
        transformation=train_transformations,
    )

    val_dataloader = prepare_dataloader(
        train_params["val_data_path"],
        batch_size,
        shuffle=False,
        transformation=val_transformations,
    )

    if verbose:
        print("Initializing model, optimizer, and scheduler...")
    vae, optimizer, scheduler = prepare_model(train_params, model_params)

    dataset_len = len(train_dataloader.dataset)
    val_dataset_len = len(val_dataloader.dataset)
    early_stop_metric = train_params["early_stop_metric"]
    epochs_no_improve = 0
    best_metric_values = {}

    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Training...")
        # Training loop
        total_loss, recon_loss_total, total_mmd_loss = train_one_epoch(
            vae,
            train_dataloader,
            optimizer,
            device,
            train_params["loss_args"],
            epoch,
            scheduler,
            val_dataloader=val_dataloader,
            val_every_n_steps=val_every_n_steps,
            scheduler_step=schduler_steps,
            verbose=verbose,
        )
        scheduler.step()

        avg_recon_loss = recon_loss_total / dataset_len
        avg_mmd_loss = total_mmd_loss / dataset_len
        avg_total_loss = total_loss / dataset_len

        mlflow.log_metrics(
            {
                "epoch_recon_loss": avg_recon_loss,
                "epoch_mmd_loss": avg_mmd_loss,
                "epoch_total_loss": avg_total_loss,
            },
            step=epoch + 1,
        )

        if verbose:
            print(
                f"Epoch {epoch+1} - Training Loss: {avg_total_loss:.6f}, Recon Loss: {avg_recon_loss:.6f}, MMD Loss: {avg_mmd_loss:.6f}"
            )

        # Evaluation loop
        vae.eval()
        val_total_loss = 0
        val_recon_loss_total = 0
        val_mmd_loss_total = 0
        val_ssim_total = 0
        val_psnr_total = 0

        if verbose:
            print(f"Epoch {epoch+1} - Evaluating on validation set...")
        with torch.no_grad():
            for val_batch in val_dataloader:
                input_batch = val_batch.to(device)
                targets = deepcopy(input_batch).to(device)

                recon_batch, z = vae(input_batch)
                recon_loss, mmd_loss = mmd_vae_loss_function(
                    recon_batch, targets, z, epoch, **train_params["loss_args"]
                )
                loss = recon_loss + mmd_loss

                val_total_loss += loss.item()
                val_recon_loss_total += recon_loss.item()
                val_mmd_loss_total += mmd_loss.item()

                recon_batch = (recon_batch + 1) / 2.0
                targets = (targets + 1) / 2.0

                # Validation Metrics
                val_ssim_total += ssim(recon_batch, targets).item()
                val_psnr_total += psnr(recon_batch, targets).item()

        avg_val_recon_loss = val_recon_loss_total / val_dataset_len
        avg_val_mmd_loss = val_mmd_loss_total / val_dataset_len
        avg_val_total_loss = val_total_loss / val_dataset_len
        avg_val_ssim = val_ssim_total / len(val_dataloader)
        avg_val_psnr = val_psnr_total / len(val_dataloader)

        metrics_dict = {
            "val_epoch_recon_loss": avg_val_recon_loss,
            "val_epoch_mmd_loss": avg_val_mmd_loss,
            "val_epoch_total_loss": avg_val_total_loss,
            "val_epoch_ssim": avg_val_ssim,
            "val_epoch_psnr": avg_val_psnr,
        }

        mlflow.log_metrics(metrics_dict, step=epoch + 1)

        if verbose:
            print(
                f"Epoch {epoch+1} - Validation Loss: {avg_val_total_loss:.6f}, Recon Loss: {avg_val_recon_loss:.6f}, MMD Loss: {avg_val_mmd_loss:.6f}, SSIM: {avg_val_ssim:.4f}, PSNR: {avg_val_psnr:.2f}"
            )

        # Determine if the early stop metric should be minimized or maximized
        current_metric = metrics_dict[early_stop_metric]
        global best_loss
        is_better = (
            (current_metric < best_loss)
            if minimize_metric
            else (current_metric > best_loss)
        )

        if is_better:
            if verbose:
                print(
                    f"Epoch {epoch+1} - New best {early_stop_metric}: {current_metric:.6f}. Saving model checkpoint."
                )
            torch.save(vae.state_dict(), "best.pt")
            mlflow.log_artifact("best.pt")
            best_loss = current_metric
            epochs_no_improve = 0
            best_metric_values = deepcopy(metrics_dict)
        else:
            epochs_no_improve += 1
            if verbose:
                print(
                    f"Epoch {epoch+1} - No improvement in {early_stop_metric}. ({epochs_no_improve}/{early_stop_patience})"
                )

        if epochs_no_improve >= early_stop_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if verbose:
        print("Training finished. Saving final model.")
    torch.save(vae.state_dict(), "final_model.pt")
    mlflow.log_artifact("final_model.pt")
    mlflow.end_run()

    if verbose:
        print("Best validation metrics:", best_metric_values)
    return best_metric_values


if __name__ == "__main__":
    if len(sys.argv) > 1:
        training_parameters = sys.argv[1]
    else:
        training_parameters = "src/x_net/model_configs/processorAV.json"

    metric_values = train_model(training_parameters, True)

    print(json.dumps(metric_values))
