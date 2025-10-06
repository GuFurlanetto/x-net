import os
import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from x_net import XNet
from utils.common import read_json
from train_xnet import PrecomputedFeatureDataset


def evaluate(config_path):
    config = read_json(config_path)
    model = XNet(config["model_parameters"])
    model.load_state_dict(
        torch.load(config["training_parameters"]["best_model_save_path"])
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    val_dir = glob.glob(
        config["training_parameters"]["data_parameters"]["val_features"]
    )
    dataset = PrecomputedFeatureDataset(val_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            all_preds.append(pred.item())
            all_labels.append(y.item())
            print(f"Pred = {pred.item()} | GT = {y.item()}")

    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_preds)
    print(f"\nFinal Accuracy: {acc:.4f}")


if __name__ == "__main__":
    import sys

    evaluate(sys.argv[1])
