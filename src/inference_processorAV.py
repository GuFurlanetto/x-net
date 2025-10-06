import os
from PIL import Image
import torch
from x_net.controller.Processor_av import ProcessorAV
import numpy as np
import cv2
import matplotlib.pyplot as plt  # ⬅️ NEW
from torchvision import transforms
from train_processorAV import prepare_dataloader
from utils.common import read_json


def make_image_grid(images, grid_shape, output_path, title=""):  # ⬅️ NEW
    fig, axes = plt.subplots(
        *grid_shape, figsize=(grid_shape[1] * 2, grid_shape[0] * 2)
    )
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
        ax.imshow(images[i])
        ax.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def inference_on_folder(
    model_path, folder_path, model_params, batch_size=32, device="cuda", save_dir=None
):
    train_params = read_json(model_params)["training_parameters"]
    val_transformations = None
    if "transformations" in train_params:
        val_transform_list = []
        for t_name, t_params in train_params["transformations"].items():
            transform_cls = getattr(transforms, t_name, None)
            params = t_params if isinstance(t_params, dict) else {}
            use_in_val = params.pop("use_in_val", False)
            if transform_cls is not None and use_in_val:
                val_transform_list.append(transform_cls(**params))
        if val_transform_list:
            val_transformations = transforms.Compose(val_transform_list)

    dataloader = prepare_dataloader(
        folder_path,
        batch_size=batch_size,
        shuffle=False,
        transformation=val_transformations,
    )

    model_params = read_json(model_params)["model_parameters"]
    model = ProcessorAV(model_params).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    results = []
    filenames = (
        dataloader.dataset.image_files
        if hasattr(dataloader.dataset, "image_files")
        else None
    )

    # ➕ Collect for 4x10 visualization
    orig_images = []
    recon_images = []

    with torch.no_grad():
        idx = 0
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) > 1:
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)

            for i in range(inputs.size(0)):
                input_img = inputs[i].cpu()
                recon_img = outputs[0][i].cpu()

                input_np = input_img.permute(1, 2, 0).numpy()
                recon_np = recon_img.permute(1, 2, 0).numpy()

                input_np = (input_np * 255).clip(0, 255).astype(np.uint8)
                recon_np = (recon_np * 255).clip(0, 255).astype(np.uint8)

                if len(orig_images) < 40:
                    orig_images.append(input_np)
                    recon_images.append(recon_np)

                # Optional: Save to folder
                if save_dir is not None and filenames is not None:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(
                        save_dir, f"generated_{os.path.basename(filenames[idx])}"
                    )
                    cv2.imwrite(save_path, cv2.cvtColor(recon_np, cv2.COLOR_RGB2BGR))
                idx += 1

                if len(orig_images) >= 40:
                    break
            if len(orig_images) >= 40:
                break

    # ➕ Save grid images
    os.makedirs("visualization", exist_ok=True)
    make_image_grid(
        orig_images, (4, 10), "visualization/original_grid.png", "Originals"
    )
    make_image_grid(
        recon_images,
        (4, 10),
        "visualization/reconstruction_grid.png",
        "Reconstructions",
    )

    return results


if __name__ == "__main__":
    input_folder = "/media/gustavo-furlanetto/KINGSTON/datasets/processorAV_training_datasett/Audio/Train"
    output_folder = "/media/gustavo-furlanetto/KINGSTON/datasets/processorAV_training_datasett/Image_cleam/val_recon"
    model_path = "best_models/processorAV/audio/best.pt"
    model_params_path = "best_models/processorAV/audio/processorAV.json"

    inference_on_folder(
        model_path=model_path,
        model_params=model_params_path,
        folder_path=input_folder,
        save_dir=output_folder,
    )
