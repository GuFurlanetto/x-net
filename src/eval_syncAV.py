import os
import torch
import matplotlib.pyplot as plt
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_handlers.dataset import SyncAVDataset
from x_net.controller import SyncAV
from utils.common import read_json, save_results


def compute_accuracy(preds, targets, threshold=0.5):
    predicted_labels = (preds >= threshold).float()
    correct = (predicted_labels == targets).float().sum()
    return (correct / targets.size(0)).item()


def evaluate_model(config_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Load config
    config = read_json(config_path)
    test_data_path = config["training_parameters"]["test_data_path"]
    checkpoint_path = config["training_parameters"].get(
        "save_path", "best_sync_av_model.pt"
    )

    # Prepare dataset
    test_loader = DataLoader(SyncAVDataset(test_data_path), batch_size=1, shuffle=False)

    # Load model
    model = SyncAV(config["model_parameters"])
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()

    predictions = []
    targets = []

    for idx, (video, audio, label) in enumerate(tqdm(test_loader, desc="Evaluating")):
        with torch.no_grad():
            logits = model(video, audio)
            pred_prob = torch.sigmoid(logits).item()

        label_val = label.item()

        predictions.append(pred_prob)
        targets.append(label_val)

        # Save output (video + shifted audio)
        video_np = video.squeeze(0).permute(1, 2, 3, 0).numpy()  # T x H x W x C
        audio_np = audio.squeeze(0).squeeze(0).numpy()  # (T,)
        save_path = os.path.join(
            output_dir, f"sample_{idx}_label_{int(label_val)}_pred_{pred_prob:.2f}.mp4"
        )

        save_results(video_np, audio_np, save_path)

    acc = compute_accuracy(torch.tensor(predictions), torch.tensor(targets))
    print(f"[ INFO ] Accuracy: {acc:.4f}")

    # Plot prediction vs ground truth
    plt.figure(figsize=(8, 6))
    plt.scatter(targets, predictions, alpha=0.6)
    plt.xlabel("Ground Truth Label")
    plt.ylabel("Predicted Probability")
    plt.title("Predicted Probability vs. Ground Truth")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "classification_results.png"))
    plt.close()

    return {"predictions": predictions, "targets": targets}


if __name__ == "__main__":
    model_config = str(sys.argv[1])
    output_path = "eval_results" if len(sys.argv) < 3 else sys.argv[2]

    evaluate_model(model_config, output_path)
