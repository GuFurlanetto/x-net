import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from x_net.controller.Processor_av import ProcessorAV
from train_processorAV import prepare_dataloader
from utils.common import read_json
from torchaudio.transforms import InverseMelScale, GriffinLim
import torchaudio
import torchaudio.functional as F


def spectrogram_image_to_tensor(image_np, n_mels=128):
    img_gray = np.array(Image.fromarray(image_np).convert("L")) / 255.0
    img_tensor = torch.tensor(img_gray, dtype=torch.float32)  # shape: [H, W]
    img_tensor = transforms.Resize((n_mels, img_tensor.shape[1]))(
        img_tensor.unsqueeze(0)
    )
    return img_tensor  # shape: [1, n_mels, time]


def mel_to_waveform(mel_db, sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128):
    """
    mel_db: Tensor of shape [1, n_mels, time] in dB
    Returns waveform: Tensor [samples]
    """

    # 1. Convert dB to amplitude
    mel_amplitude = F.DB_to_amplitude(
        mel_db, ref=1.0, power=2.0
    )  # ⬅️ Matches AmplitudeToDB(power=2.0)

    # 2. Approximate linear STFT magnitude from mel
    inv_mel = InverseMelScale(
        n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate
    )
    linear_mag = inv_mel(mel_amplitude)

    # 3. Griffin-Lim to recover waveform
    griffin = GriffinLim(n_fft=n_fft, hop_length=hop_length)
    waveform = griffin(linear_mag)

    return waveform.squeeze(0)


def inference_and_audio_reconstruction(
    model_path,
    folder_path,
    model_params,
    sample_rate=16000,
    batch_size=32,
    device="cuda",
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

    model_cfg = read_json(model_params)["model_parameters"]
    model = ProcessorAV(model_cfg).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    os.makedirs("audio_outputs/original_wav", exist_ok=True)
    os.makedirs("audio_outputs/reconstructed_wav", exist_ok=True)

    saved = 0

    with torch.no_grad():
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

                input_np = ((input_np + 1 / 2) * 255).clip(0, 255).astype(np.uint8)
                recon_np = ((recon_np + 1 / 2) * 255).clip(0, 255).astype(np.uint8)

                # Convert back to waveform
                input_tensor = spectrogram_image_to_tensor(input_np)
                recon_tensor = spectrogram_image_to_tensor(recon_np)

                input_waveform = mel_to_waveform(input_tensor.unsqueeze(0), sample_rate)
                recon_waveform = mel_to_waveform(recon_tensor.unsqueeze(0), sample_rate)

                torchaudio.save(
                    f"audio_outputs/original_wav/sample_{saved:03d}.wav",
                    input_waveform,
                    sample_rate,
                )
                torchaudio.save(
                    f"audio_outputs/reconstructed_wav/sample_{saved:03d}.wav",
                    recon_waveform,
                    sample_rate,
                )

                saved += 1
                if saved >= 40:
                    return


if __name__ == "__main__":
    input_folder = "/media/gustavo-furlanetto/KINGSTON/datasets/processorAV_training_datasett/Audio/Train"
    model_path = "best_models/processorAV/audio/best.pt"
    model_params_path = "best_models/processorAV/audio/processorAV.json"

    inference_and_audio_reconstruction(
        model_path=model_path,
        model_params=model_params_path,
        folder_path=input_folder,
        sample_rate=16000,
    )
