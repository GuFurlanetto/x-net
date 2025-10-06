import optuna
import json
import subprocess
import tempfile
import sys
import os

import optunahub

CONFIG_PATH = "src/x_net/model_configs/sync_av.json"
TRAIN_SCRIPT = "src/train_syncAV.py"


def syncav_architecture_objective(trial):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # CNN architecture: generate shallower variants of the base model
    video_layers = []
    audio_layers = []
    num_layers = trial.suggest_int("num_layers", 2, 4)

    in_channels = 3
    for i in range(num_layers):
        out_channels = trial.suggest_categorical(
            f"video_out_channels_{i}", [64, 128, 256, 512]
        )
        video_layers.append(
            {
                "out_channels": out_channels,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": 2,
            }
        )
        in_channels = out_channels

    in_channels = 1
    for i in range(num_layers):
        out_channels = trial.suggest_categorical(
            f"audio_out_channels_{i}", [64, 128, 256, 512]
        )
        audio_layers.append(
            {
                "out_channels": out_channels,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool": 2,
            }
        )
        in_channels = out_channels

    embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 512])
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

    config["model_parameters"] = {
        "video_cnn": {"in_channels": 3, "embed_dim": embed_dim, "layers": video_layers},
        "audio_cnn": {"in_channels": 1, "embed_dim": embed_dim, "layers": audio_layers},
        "predictor": {"hidden_dim": hidden_dim},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(config, tmp, indent=2)
        tmp.flush()
        tmp_config_path = tmp.name

        result = subprocess.run(
            ["python", TRAIN_SCRIPT, tmp_config_path],
            capture_output=True,
            text=True,
        )

        try:
            metrics = json.loads(result.stdout.strip())
            val_acc = metrics.get("val_acc", float("-inf"))
        except Exception:
            print("Failed to parse output. STDERR:", result.stderr)
            val_acc = float("-inf")

    return val_acc


def syncav_optimizer_objective(trial):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_params = {"lr": lr, "weight_decay": weight_decay}

    if optimizer in ["Adam", "AdamW"]:
        optimizer_params["amsgrad"] = trial.suggest_categorical(
            "amsgrad", [True, False]
        )
    elif optimizer == "SGD":
        optimizer_params["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)

    config["training_parameters"]["optimizer"] = optimizer
    config["training_parameters"]["optimizer_params"] = optimizer_params

    scheduler = trial.suggest_categorical(
        "lr_scheduler", ["StepLR", "ExponentialLR", "OneCycleLR"]
    )
    if scheduler == "StepLR":
        scheduler_params = {
            "step_size": trial.suggest_int("step_size", 2, 10),
            "gamma": trial.suggest_float("gamma", 0.2, 0.9),
        }
    elif scheduler == "ExponentialLR":
        scheduler_params = {"gamma": trial.suggest_float("gamma", 0.5, 0.9)}
    else:
        scheduler_params = {
            "max_lr": trial.suggest_float("max_lr", 1e-4, 5e-3, log=True),
            "steps_per_epoch": 100,
            "epochs": 100,
        }

    config["training_parameters"]["lr_scheduler"] = scheduler
    config["training_parameters"]["lr_scheduler_params"] = scheduler_params

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(config, tmp, indent=2)
        tmp.flush()
        tmp_config_path = tmp.name

        result = subprocess.run(
            ["python", TRAIN_SCRIPT, tmp_config_path],
            capture_output=True,
            text=True,
        )

        try:
            metrics = json.loads(result.stdout.strip())
            val_acc = metrics.get("val_acc", float("-inf"))
        except Exception:
            print("Failed to parse output. STDERR:", result.stderr)
            val_acc = float("-inf")

    return val_acc


if __name__ == "__main__":
    study_name = str(sys.argv[1])
    study_to_use = int(sys.argv[2])
    num_trials = int(sys.argv[3])
    n_jobs = int(sys.argv[4])

    objectives_avaliable = {
        1: syncav_architecture_objective,
        2: syncav_optimizer_objective,
    }

    os.makedirs(f"Studies/{study_name}", exist_ok=True)
    study_storage_path = f"sqlite:///Studies/{study_name}/study.db"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=study_storage_path,
        load_if_exists=True,
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
    )

    objective = objectives_avaliable[study_to_use]

    study.optimize(objective, n_trials=num_trials, n_jobs=n_jobs)

    print("\nBest trial:")
    best = study.best_trial
    print(f"  Value: {best.value}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")
