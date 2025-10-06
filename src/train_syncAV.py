import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow
import torch.nn.functional as F

from data_handlers.dataset import SyncAVDataset
from x_net.controller.Sync_AV import SyncAV
from utils.common import read_json
from data_handlers.dataset import load_UCF_dataset
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_dataloader(data_path, batch_size, shuffle=True):
    dataset = SyncAVDataset(data_path, load_pre_processed=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def compute_binary_metrics(logits, labels):
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).int()
    labels = labels.int()
    correct = (preds == labels).sum().item()
    total = labels.numel()
    accuracy = correct / total
    return accuracy


def train_model(config_path, verbose=True):
    config = read_json(config_path)
    training_params = config.get("training_parameters", {})
    model_params = config.get("model_parameters", {})

    train_paths, _, val_paths, _ = load_UCF_dataset(training_params["data_path"])

    if training_params["use_pre_processed"]:
        train_processed_data = []
        val_processed_data = []
        all_processed_data = glob.glob(
            f"{training_params['pre_processed_data_path']}/*.npy"
        )

        train_filenames = [os.path.basename(file) for file in train_paths]
        val_filename = [os.path.basename(file) for file in val_paths]

        for file in all_processed_data:
            filename = os.path.basename(file).removesuffix(".npy")

            if filename in train_filenames:
                train_processed_data.append(file)
            elif filename in val_filename:
                val_processed_data.append(file)
            else:
                print(f"[ ERROR ] {filename} not in any split")
                raise ValueError

        train_paths = train_processed_data
        val_paths = val_processed_data

    # Preparing dataset
    train_loader = prepare_dataloader(
        data_path=train_paths,
        batch_size=training_params.get("batch_size", 8),
        shuffle=training_params.get("shuffle", True),
    )

    val_loader = prepare_dataloader(
        data_path=val_paths,
        batch_size=training_params.get("batch_size", 8),
        shuffle=False,
    )

    model = SyncAV(model_params).to(device)

    # Training configuration setup
    epochs = training_params.get("epochs", 5000)
    save_path = training_params.get("save_path", "best_model.pt")
    early_stop_patience = training_params.get("early_stop_patience", 10)
    experiment_name = training_params.get("experiment_name", "SyncAV Classification")

    # Optimizer setup
    optimizer_name = training_params.get("optimizer", "Adam")
    optimizer_params = training_params.get("optimizer_params", {"lr": 1e-4})
    optimizer_cls = getattr(optim, optimizer_name, optim.Adam)
    optimizer = optimizer_cls(model.parameters(), **optimizer_params)

    criterion = nn.BCEWithLogitsLoss()

    # Scheduler setup
    scheduler_name = training_params.get("lr_scheduler", "StepLR")
    scheduler_params = training_params.get(
        "lr_scheduler_params", {"step_size": 10, "gamma": 0.5}
    )
    scheduler_cls = getattr(
        optim.lr_scheduler, scheduler_name, optim.lr_scheduler.StepLR
    )
    scheduler = scheduler_cls(optimizer, **scheduler_params)

    best_val_acc = float("-inf")
    epochs_no_improve = 0
    best_metrics = {}

    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    mlflow.log_params(training_params)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for training_item in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
        ):
            video = training_item["video_frames"]
            audio = training_item["audio_frames"]
            label = training_item["labels"]

            video, audio, label = (
                video.to(device),
                audio.to(device),
                label.float().to(device).unsqueeze(1),
            )

            audio = audio.permute(0, 3, 1, 2).float()
            video = video.permute(0, 4, 1, 2, 3).float()

            optimizer.zero_grad()
            output = model(video, audio)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * video.size(0)
            correct += (
                ((torch.sigmoid(output) >= 0.8).int() == label.int()).sum().item()
            )
            total += label.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        train_acc = correct / total

        if verbose:
            print(
                f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for validation_item in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"
            ):
                video = validation_item["video_frames"]
                audio = validation_item["audio_frames"]
                label = validation_item["labels"]

                video, audio, label = (
                    video.to(device),
                    audio.to(device),
                    label.float().to(device).unsqueeze(1),
                )

                audio = audio.permute(0, 3, 1, 2).float()
                video = video.permute(0, 4, 1, 2, 3).float()
                output = model(video, audio)
                loss = criterion(output, label)
                val_loss += loss.item() * video.size(0)
                correct += (
                    ((torch.sigmoid(output) >= 0.7).int() == label.int()).sum().item()
                )
                total += label.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        if verbose:
            print(
                f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        mlflow.log_metrics(
            {
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            },
            step=epoch + 1,
        )

        scheduler.step()

        # Checkpointing and Early Stopping
        if val_acc > best_val_acc:
            if verbose:
                print(
                    f"Validation acc improved from {best_val_acc:.4f} to {val_acc:.4f}. Saving model..."
                )
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path)
            best_metrics = {
                "train_loss": avg_loss,
                "val_loss": avg_val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
            }
        else:
            epochs_no_improve += 1
            if verbose:
                print(
                    f"No improvement in validation loss. Patience: {epochs_no_improve}/{early_stop_patience}"
                )

        if epochs_no_improve >= early_stop_patience:
            if verbose:
                print("Early stopping triggered.")
            break

    mlflow.end_run()
    if verbose:
        print(f"Training completed. Best model saved to {save_path}")
    return best_metrics


if __name__ == "__main__":
    config_path = (
        sys.argv[1] if len(sys.argv) > 1 else "src/x_net/model_configs/sync_av.json"
    )
    metrics = train_model(config_path)
    print(json.dumps(metrics, indent=2))
