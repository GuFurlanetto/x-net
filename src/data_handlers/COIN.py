import cv2
import torch
import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram
import random
import torch
import cv2
import numpy as np
import torchaudio
from pydub import AudioSegment
from pathlib import Path
import torch
import torchaudio
from torchaudio.transforms import InverseMelScale, GriffinLim
import librosa
import librosa.display
import numpy as np
import soundfile as sf


class COIN(Dataset):
    def __init__(self, data_path: Path, split: str, raw_data_path: Path):
        super().__init__()
        self.split = split
        self.data_path = data_path / split
        self.raw_data_path = raw_data_path

        self.video_files = os.listdir(self.data_path)
        self.audio_sample_rate = 22050  # Desired audio sample rate
        self.mel_transform = MelSpectrogram(
            sample_rate=self.audio_sample_rate, n_fft=1024, hop_length=256, n_mels=128
        )  # MelSpectrogram with 224 Mel bins

        self.max_delay = 5  # Max delay/advance in seconds

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, index):
        # Get the path of the video file
        video_file = self.data_path / self.video_files[index]

        # Open the video file
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {video_file}")

        # Get the frame rate of the video
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = fps // 10  # Interval for extracting 10 frames per second

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Only keep frames at the defined interval
            try:
                if frame_count % frame_interval == 0:
                    # Convert frame from BGR to RGB and resize (optional)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = (
                        cv2.resize(frame, (224, 224)) / 255
                    )  # Resize to a fixed size, if needed
                    frames.append(frame)
            except:
                pass

            frame_count += 1

        cap.release()

        # Ensure we have exactly 50 frames by padding with black frames
        while len(frames) < 50:
            frames.append(np.zeros((224, 224), dtype=np.uint8))  # Black frame

        # Extract the audio for the video
        filename = os.path.basename(video_file).split("_segment")[0] + ".mp4"
        audio_file = str(self.raw_data_path / filename)
        audio_waveform, sample_rate = torchaudio.load(audio_file)

        # Convert to mono by averaging channels
        if audio_waveform.size(0) > 1:  # If multi-channel
            audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)

        # Resample audio to the desired sample rate
        if sample_rate != self.audio_sample_rate:
            resampler = Resample(orig_freq=sample_rate, new_freq=self.audio_sample_rate)
            audio_waveform = resampler(audio_waveform)

        # Randomly delay or advance the audio
        delay_seconds = random.uniform(
            -self.max_delay, self.max_delay
        )  # Random delay/advance in seconds
        delay_samples = int(
            delay_seconds * self.audio_sample_rate
        )  # Convert to samples

        # Apply the delay to the audio
        if delay_samples > 0:
            # Advance the audio by shifting it forward (pad the end with zeros)
            audio_waveform = torch.cat(
                [audio_waveform[:, delay_samples:], torch.zeros(1, delay_samples)],
                dim=1,
            )
        elif delay_samples < 0:
            # Delay the audio by shifting it backward (pad the beginning with zeros)
            audio_waveform = torch.cat(
                [torch.zeros(1, -delay_samples), audio_waveform], dim=1
            )

        # Cut video segment
        max_audio_samples = self.audio_sample_rate * 5  # 5 seconds of audio
        current_segment = os.path.basename(video_file).split("_segment_")[1][:-4]
        sample_start = int(current_segment) * max_audio_samples
        sample_end = sample_start + max_audio_samples

        audio_waveform = audio_waveform[:, sample_start:sample_end]

        # Convert audio to Mel spectrogram
        mel_spectrogram = self.mel_transform(audio_waveform)  # Shape: (1, 224, Time)

        # Resize Mel spectrogram to (224, 224)
        mel_spectrogram = torch.nn.functional.interpolate(
            mel_spectrogram.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        ).squeeze(
            0
        )  # Shape: (1, 224, 224)
        # Convert frames to a 3D tensor

        frames_tensor = torch.tensor(
            np.array(frames[:50]), dtype=torch.float32
        ).unsqueeze(
            1
        )  # Shape: (50, 1, 224, 224)

        frames = frames_tensor.squeeze().numpy()
        frames_with_audio = torch.cat(
            [mel_spectrogram.unsqueeze(0), frames_tensor], dim=0
        )

        return frames_with_audio.squeeze(), torch.tensor(
            [delay_seconds], dtype=torch.float32
        )


def mel_to_wav(
    mel_spectrogram, sample_rate=22050, n_fft=1024, n_mels=128, hop_length=256
):
    """
    Converts a Mel spectrogram tensor to a WAV audio waveform.

    Parameters:
        mel_spectrogram (torch.Tensor): Input Mel spectrogram, shape (n_mels, time_steps).
        sample_rate (int): Target sample rate of the output audio.
        n_fft (int): Size of FFT used to create the spectrogram.
        n_mels (int): Number of Mel bands used in the Mel spectrogram.
        hop_length (int): Hop length used to create the Mel spectrogram.

    Returns:
        torch.Tensor: Reconstructed audio waveform.
    """
    # Ensure input is 2D (n_mels x time_steps)

    # Example: Convert PyTorch Mel spectrogram to NumPy
    mel_spectrogram_np = mel_spectrogram.cpu().numpy()

    # Inverse Mel spectrogram using librosa
    linear_spectrogram = librosa.feature.inverse.mel_to_stft(
        mel_spectrogram_np, sr=sample_rate, n_fft=n_fft
    )
    waveform = librosa.griffinlim(linear_spectrogram, hop_length=hop_length)

    # Save to WAV
    return waveform


def generate_video_with_delayed_audio(
    dataset, index, output_video_path, output_audio_path
):
    # Extract frames and delayed audio from the dataset
    frames_tensor, delay_seconds = dataset[index]

    # Convert frames_tensor to numpy (for OpenCV)
    frames = frames_tensor[1:].numpy()  # Shape: (51, 224, 224)

    # Video parameters
    fps = 10  # Since we extracted 10 frames per second

    # Define video writer to save the frames as a video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (224, 224), 0)

    # Write frames to the video
    for frame in frames:
        video_writer.write(
            frame.astype(np.uint8)
        )  # Ensure frame is in uint8 format for OpenCV

    # Release the video writer
    video_writer.release()

    # Extract the audio waveform
    spectogram = frames_tensor[0]
    waveform = mel_to_wav(spectogram)

    # Save to WAV
    sf.write("output.wav", waveform, 22050)

    print(f"Audio delayed by {delay_seconds} seconds")


if __name__ == "__main__":

    # Example usage:
    # Assuming 'dataset' is your COIN dataset instance
    index = 0  # Index of the sample you want to process
    output_video_path = "video_output.mp4"
    output_audio_path = "output_audio_with_delay.wav"
    dataset = COIN(
        Path("data/coin/training_dataset"), "val", Path("data/coin/raw_videos")
    )

    generate_video_with_delayed_audio(
        dataset, index, output_video_path, output_audio_path
    )
