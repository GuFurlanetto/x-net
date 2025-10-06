import torch
import torchaudio
import torchvision.transforms as T
from pathlib import Path
import numpy as np
import av
import cv2
import random
from moviepy import VideoFileClip
from PIL import Image
import sys

from utils.common import read_json
from x_net.controller.Sync_AV import SyncAV

# --- Constants ---
N_FRAMES = 16
TARGET_SR = 16000
FRAME_RATE = 16
N_MELS = 128
RESIZE = T.Resize((224, 224))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---


def sample_frame_indices(container, n_frames=N_FRAMES, frame_rate=FRAME_RATE):
    total_frames = container.streams.video[0].frames
    converted_len = n_frames * frame_rate

    if converted_len > total_frames:
        end_idx = total_frames
        start_idx = 0
    else:
        end_idx = min(np.random.randint(converted_len, total_frames), total_frames - 1)
        start_idx = max(0, end_idx - converted_len)

    indices = np.linspace(start_idx, end_idx, num=n_frames)
    return np.clip(indices, start_idx, end_idx - 1).astype(np.int64)


def read_video_pyav(path, times, num_frames=16, size=(224, 224)):
    with av.open(path) as container:
        stream = container.streams.video[0]
        duration = stream.duration

        print(f"[ INFO ] Video duration: {duration}")

        frames = []
        for t in times:
            container.seek(int(t / stream.time_base), stream=stream)
            frame = next(container.decode(video=0), None)
            if frame is not None:
                img = frame.to_ndarray(format="rgb24")
                resized = cv2.resize(img, size)
                tensor = torch.tensor(resized).permute(2, 0, 1).float()
                frames.append(tensor)

        if len(frames) < num_frames:
            pad = frames[-1] if frames else torch.zeros(3, *size)
            frames += [pad] * (num_frames - len(frames))

        return torch.stack(frames, dim=1)  # [3, T, H, W]


def extract_audio(path, start_sec=None, end_sec=None, duration=9.0):
    try:
        with VideoFileClip(str(path)) as clip:
            audio = clip.audio
            if audio is None:
                raise ValueError("No audio track")

            if start_sec and end_sec:
                audio = audio.subclipped(start_sec, end_sec)

            array = audio.to_soundarray(fps=TARGET_SR).mean(axis=1)
    except:
        array = np.zeros(int(duration * TARGET_SR) * 2)
    return array


def waveform_to_mel_image(waveform):
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)  # [1, N]
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR,
        n_fft=1024,
        hop_length=512,
        n_mels=N_MELS,
    )(
        waveform
    )  # [1, n_mels, T]

    mel_db = torchaudio.transforms.AmplitudeToDB()(mel).squeeze(0)  # [n_mels, T]
    img = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)
    img = (img * 255).clamp(0, 255).byte().cpu().numpy()
    img = np.uint8(img * 255)
    img = Image.fromarray(img).convert("RGB")
    return RESIZE(img)


# --- Inference Entry ---


def run_syncav(config_path, weights_path, video_path, return_embedding=False):
    config = read_json(config_path)["model_parameters"]
    model = SyncAV(config).to(DEVICE)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.eval()

    # Process input
    with av.open(video_path) as container:
        indices = sample_frame_indices(container)

    video_tensor = (
        read_video_pyav(video_path, indices).unsqueeze(0).to(DEVICE)
    )  # [1, 3, T, H, W]

    # Calculate timestamps based on frame indices and frame rate
    start_frame = indices[0]
    end_frame = indices[-1]

    if end_frame <= FRAME_RATE * N_FRAMES:
        # Used full clip extraction
        start_sec = None
        end_sec = None
    else:
        start_sec = start_frame / 30
        end_sec = end_frame / 30

    waveform = extract_audio(video_path, start_sec=start_sec, end_sec=end_sec)
    spectrogram_img = waveform_to_mel_image(waveform)
    spectrogram_tensor = (
        torch.tensor(np.array(spectrogram_img)).permute(2, 0, 1).unsqueeze(0).float()
    )
    spectrogram_tensor = spectrogram_tensor.to(DEVICE)  # [1, 3, 224, 224]

    with torch.no_grad():
        if return_embedding:
            out = model(video_tensor, spectrogram_tensor, return_embedding=True)
            print("Sync embedding:", out.squeeze().cpu().numpy())
        else:
            out = model(video_tensor, spectrogram_tensor)
            prob = (torch.sigmoid(out) >= 0.7).int().item()

            print(f"Sync probability: {prob}")


# --- CLI ---
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(
            "Usage: python run_syncav_on_video.py <config.json> <weights.pt> <video.mp4> [--embed]"
        )
        sys.exit(1)

    config_path = Path(sys.argv[1])
    weights_path = Path(sys.argv[2])
    video_path = Path(sys.argv[3])
    return_embedding = "--embed" in sys.argv

    run_syncav(config_path, weights_path, video_path, return_embedding)
