import os
import glob
import random
import torch
import mlflow
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from data_handlers.dataset import VideoAudioDataset, load_UCF_dataset
from x_net import XNet
from utils.common import read_json

###############################################
#                COIN Utils                   #
###############################################


def load_coin_dataset(data_path):

    # Define data paths
    train_videos_path = os.path.join(data_path, "train")
    val_videos_path = os.path.join(data_path, "val")

    train_videos = glob.glob(f"{train_videos_path}/*/*")
    val_videos = glob.glob(f"{val_videos_path}/*/*")

    train_labels = [int(video_path.split("/")[-2]) for video_path in train_videos]
    val_labels = [int(video_path.split("/")[-2]) for video_path in val_videos]

    return train_videos, train_labels, val_videos, val_labels


###############################################
#                COIN Utils                   #
###############################################


def set_seed(seed):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def step(self, val_loss):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def build_optimizer(config, model):
    opt_type = config["optimizer"]
    model_params = model.parameters()
    opt_params = config["optimizer_params"]

    return getattr(torch.optim, opt_type)(model_params, **opt_params)


def build_scheduler(config, optmizer):
    sched_type = config["lr_scheduler"]
    params = config["lr_scheduler_params"]

    return getattr(torch.optim.lr_scheduler, sched_type)(optmizer, **params)


def build_loss(config):
    loss_type = config["loss"]
    params = config["loss_params"]

    return getattr(nn, loss_type)(**params)


def train_from_embeddings(config_path):
    config = read_json(config_path)
    training_config = config["training_parameters"]
    set_seed(training_config.get("seed", 42))

    model = XNet(config["model_parameters"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_parameters = training_config["data_parameters"]
    data_path = data_parameters["dataset_path"]
    video_processor_path = data_parameters["video_processor_path"]
    audio_processor_path = data_parameters["audio_processor_path"]

    video_processor = AutoImageProcessor.from_pretrained(video_processor_path)
    audio_processor = AutoImageProcessor.from_pretrained(audio_processor_path)

    train_paths, train_labels, val_paths, val_labels = load_coin_dataset(data_path)

    if data_parameters["load_pre_processed_data"]:
        train_processed_data = []
        val_processed_data = []
        all_processed_data = glob.glob(
            f"{data_parameters['pre_processed_data_path']}/*.npy"
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

    train_loader = DataLoader(
        VideoAudioDataset(
            train_paths,
            train_labels,
            video_processor,
            audio_processor,
            load_pre_processed=data_parameters["load_pre_processed_data"],
        ),
        batch_size=data_parameters["batch_size"],
        shuffle=data_parameters["shuffle"],
    )

    val_loader = DataLoader(
        VideoAudioDataset(
            val_paths,
            val_labels,
            video_processor,
            audio_processor,
            load_pre_processed=data_parameters["load_pre_processed_data"],
        ),
        batch_size=data_parameters["batch_size"],
        shuffle=False,
    )

    optimizer = build_optimizer(training_config, model)
    scheduler = build_scheduler(training_config, optimizer)
    criterion = build_loss(training_config)
    early_stopper = EarlyStopping(patience=training_config["early_stop_patience"])

    # Mlflow setup
    logger_cfg = training_config["logger"]
    mlflow.set_experiment(logger_cfg["experiment_name"])
    mlflow.start_run(run_name=logger_cfg["run_name"])
    mlflow.log_artifact(config_path)
    mlflow.log_params(training_config)
    mlflow.log_params(config["model_parameters"])

    scheduler_is_one_cycle = training_config["lr_scheduler"] == "OneCycleLR"
    data_lenght = train_loader.__len__()
    val_data_lenght = val_loader.__len__()

    print(f"[ INFO ] Train size: {data_lenght}")
    print(f"[ INFO ] Validation size: {val_data_lenght}")

    best_val_loss = float("inf")
    for epoch in range(training_config["epochs"]):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for _, input_dict in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            video_input, audio_input, frames, spectograms = (
                input_dict["video_pixel_values"],
                input_dict["audio_pixel_values"],
                input_dict["frames"],
                input_dict["spectogram"],
            )

            y = input_dict["labels"].to(device)
            video_input = video_input.to(device)
            audio_input = audio_input.to(device)
            frames = frames.to(device)
            spectograms = spectograms.to(device)

            out = model(video_input, audio_input, frames, spectograms)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            # For One Cycle scheduler, step at each optimizer step
            if scheduler_is_one_cycle:
                scheduler.step()

            optimizer.zero_grad()
            total_loss += loss.item()

        # For other scheduler, step at end of each epoch
        if not scheduler_is_one_cycle:
            scheduler.step()

        mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for input_dict in val_loader:
                video_input, audio_input, frames, spectograms = (
                    input_dict["video_pixel_values"],
                    input_dict["audio_pixel_values"],
                    input_dict["frames"],
                    input_dict["spectogram"],
                )

                y = input_dict["labels"].to(device)
                video_input = video_input.to(device)
                audio_input = audio_input.to(device)
                frames = frames.to(device)
                spectograms = spectograms.to(device)

                out = model(video_input, audio_input, frames, spectograms)
                val_loss += criterion(out, y).item()
                preds = out.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        total_loss /= data_lenght
        val_loss /= val_data_lenght

        print(
            f"Epoch {epoch+1} | Train Loss: {total_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
        mlflow.log_metrics(
            {
                "train_loss": total_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            step=epoch,
        )

        if val_loss < best_val_loss:
            print(
                f"Saving new best model at {training_config['best_model_save_path']} | Val loss {val_loss:.4f} | Val acc {val_acc:.4f}"
            )
            torch.save(model.state_dict(), training_config["best_model_save_path"])
            mlflow.log_artifact(training_config["best_model_save_path"])

            best_val_loss = val_loss

        if early_stopper.step(val_loss):
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), training_config["final_model_save_path"])
    mlflow.log_artifact(training_config["final_model_save_path"])
    print("Training complete. Final model saved.")


if __name__ == "__main__":
    import sys

    train_from_embeddings("src/x_net/model_configs/x_net.json")
