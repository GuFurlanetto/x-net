import optuna
import json
import subprocess
import tempfile
import sys
import optunahub
import os

CONFIG_PATH = "src/x_net/model_configs/processorAV.json"
TRAIN_SCRIPT = "src/train_processorAV_cross.py"


def arc_study_objective(trial):
    # Load current config
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Suggest model architecture
    latent_dim = trial.suggest_categorical("latent_dim", [32, 64, 128, 256])
    num_layers = trial.suggest_int("num_layers", 2, 5)

    layer_sizes = [
        trial.suggest_categorical(f"layer_{i}_size", [32, 64, 128, 256])
        for i in range(num_layers)
    ]
    kernel_sizes = [
        trial.suggest_categorical(f"kernel_size_{i}", [2, 3, 4])
        for i in range(num_layers)
    ]
    strides = [
        trial.suggest_categorical(f"stride_{i}", [1, 2]) for i in range(num_layers)
    ]
    paddings = [
        trial.suggest_categorical(f"padding_{i}", [0, 1]) for i in range(num_layers)
    ]

    config["model_parameters"]["latent_dim"] = latent_dim
    config["model_parameters"]["layers"] = layer_sizes
    config["model_parameters"]["kernel_sizes"] = kernel_sizes
    config["model_parameters"]["kernel_strides"] = strides
    config["model_parameters"]["kernel_paddings"] = paddings

    # Create a unique temp config file for this trial
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(config, tmp, indent=2)
        tmp.flush()
        tmp_config_path = tmp.name

        # Call the training function (assumed to be a script that reads CONFIG_PATH and prints val_loss)
        result = subprocess.run(
            ["python", "-W", "ignore", TRAIN_SCRIPT, tmp_config_path],
            capture_output=True,
            text=True,
        )
        # Parse the output for validation loss (assume last line is val_loss)
        try:
            metrics = json.loads(result.stdout.strip())
            ssim = metrics["val_epoch_ssim"]
            psnr = metrics["val_epoch_psnr"]

        except Exception:
            print("Failed to parse validation loss. Output was:", result.stderr)
            ssim = float("-inf")
            psnr = float("-inf")

    return ssim, psnr


def optimizer_study_objective(trial):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Suggest optimizer and optimizer-specific parameters
    optimizer = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    optimizer_params = {"lr": lr, "weight_decay": weight_decay}
    if optimizer == "Adam" or optimizer == "AdamW":
        optimizer_params["amsgrad"] = trial.suggest_categorical(
            "amsgrad", [True, False]
        )
    elif optimizer == "SGD":
        optimizer_params["momentum"] = trial.suggest_float("momentum", 0.5, 0.99)

    config["training_parameters"]["optimizer"] = optimizer
    config["training_parameters"]["optimizer_params"] = optimizer_params

    # Suggest LR scheduler
    scheduler = trial.suggest_categorical("lr_scheduler", ["StepLR", "ExponentialLR"])
    if scheduler == "StepLR":
        scheduler_params = {
            "step_size": trial.suggest_int("step_size", 2, 10),
            "gamma": trial.suggest_float("gamma", 0.2, 0.9),
        }
    elif scheduler == "ExponentialLR":
        scheduler_params = {"gamma": trial.suggest_float("gamma", 0.5, 0.9)}

    config["training_parameters"]["lr_scheduler"] = scheduler
    config["training_parameters"]["lr_scheduler_params"] = scheduler_params

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(config, tmp, indent=2)
        tmp.flush()
        tmp_config_path = tmp.name

        result = subprocess.run(
            ["python", "-W", "ignore", TRAIN_SCRIPT, tmp_config_path],
            capture_output=True,
            text=True,
        )

        try:
            metrics = json.loads(result.stdout.strip())
            ssim = metrics["val_epoch_ssim"]
            psnr = metrics["val_epoch_psnr"]

        except Exception:
            print("Failed to parse validation loss. Output was:", result.stderr)
            ssim = float("-inf")
            psnr = float("-inf")

    return ssim, psnr


def loss_weight_study_objective(trial):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Suggest MMD loss parameters
    beta = trial.suggest_float("beta", 0.1, 5.0)
    reg_weight = trial.suggest_float("reg_weight", 0.0, 1000.0)
    z_var = trial.suggest_float("z_var", 0.1, 2.0)
    kernel_type = trial.suggest_categorical("kernel_type", ["rbf"])

    config["training_parameters"]["loss_args"]["beta"] = beta
    config["training_parameters"]["loss_args"]["reg_weight"] = reg_weight
    config["training_parameters"]["loss_args"]["z_var"] = z_var
    config["training_parameters"]["loss_args"]["kernel_type"] = kernel_type

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(config, tmp, indent=2)
        tmp.flush()
        tmp_config_path = tmp.name

        result = subprocess.run(
            ["python", TRAIN_SCRIPT, tmp_config_path], capture_output=True, text=True
        )

        try:
            metrics = json.loads(result.stdout.strip())
            ssim = metrics["val_epoch_ssim"]
            psnr = metrics["val_epoch_psnr"]
        except Exception:
            print("Failed to parse output. STDERR:", result.stderr)
            ssim = float("-inf")
            psnr = float("-inf")

    return ssim, psnr


def fine_tune_study_objective(trial):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    # Architecture (refined ranges)
    latent_dim = trial.suggest_categorical("latent_dim", [128, 256])
    num_layers = trial.suggest_int("num_layers", 3, 5)
    layer_sizes = [
        trial.suggest_categorical(f"layer_{i}_size", [64, 128, 256])
        for i in range(num_layers)
    ]
    kernel_sizes = [
        trial.suggest_categorical(f"kernel_size_{i}", [3, 4]) for i in range(num_layers)
    ]
    strides = [
        trial.suggest_categorical(f"stride_{i}", [1, 2]) for i in range(num_layers)
    ]
    paddings = [
        trial.suggest_categorical(f"padding_{i}", [0, 1]) for i in range(num_layers)
    ]

    config["model_parameters"].update(
        {
            "latent_dim": latent_dim,
            "layers": layer_sizes,
            "kernel_sizes": kernel_sizes,
            "kernel_strides": strides,
            "kernel_paddings": paddings,
        }
    )

    # Optimizer
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    config["training_parameters"]["optimizer_params"].update(
        {"lr": lr, "weight_decay": weight_decay}
    )

    # Loss
    beta = trial.suggest_float("beta", 0.5, 2.0)
    reg_weight = trial.suggest_float("reg_weight", 0.5, 5.0)
    config["training_parameters"]["loss_args"].update(
        {"beta": beta, "reg_weight": reg_weight}
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp:
        json.dump(config, tmp, indent=2)
        tmp.flush()
        tmp_config_path = tmp.name

        result = subprocess.run(
            ["python", TRAIN_SCRIPT, tmp_config_path], capture_output=True, text=True
        )

        try:
            metrics = json.loads(result.stdout.strip())
            ssim = metrics["val_epoch_ssim"]
            psnr = metrics["val_epoch_psnr"]
        except Exception:
            print("Failed to parse output. STDERR:", result.stderr)
            ssim = float("-inf")
            psnr = float("-inf")

    return ssim, psnr


if __name__ == "__main__":
    study_name = str(sys.argv[1])
    study_to_use = int(sys.argv[2])
    num_trials = int(sys.argv[3])
    n_jobs = int(sys.argv[4])

    os.makedirs(f"Studies/{study_name}", exist_ok=True)

    study_storage_path = f"sqlite:///Studies/{study_name}/study.db"
    study = optuna.create_study(
        sampler=optunahub.load_module("samplers/auto_sampler").AutoSampler(),
        directions=["maximize", "maximize"],
        storage=study_storage_path,
        load_if_exists=True,
    )

    if study_to_use == 0:
        study_objective = arc_study_objective
    elif study_to_use == 1:
        study_objective = optimizer_study_objective
    elif study_to_use == 2:
        study_objective = loss_weight_study_objective
    elif study_to_use == 3:
        study_objective = fine_tune_study_objective

    study.optimize(study_objective, n_trials=num_trials, n_jobs=n_jobs)

    # Print Pareto front trials
    print("\nBest trials:")
    for i, t in enumerate(study.best_trials):
        print(f"Trial #{t.number}")
        print(f"  Objectives: {t.values}")
        print(f"  Params: {t.params}\n")
