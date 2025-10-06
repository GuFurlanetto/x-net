import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import glob
import tqdm
from moviepy import VideoFileClip
from PIL import Image
import torchaudio
import torchvision.transforms as T
import tempfile
import subprocess
import json
from transformers import AutoImageProcessor, VideoMAEImageProcessor
import av
import random


#############################################################################################################
#                                     Sync AV functions                                                     #
#############################################################################################################


class SyncAVDataset(Dataset):
    def __init__(
        self,
        video_paths,
        target_sr=16000,
        duration=30,
        n_frames=16,
        n_mels=128,
        shift_range=(-3.0, 3.0),  # in seconds
        shift_prob=0.5,
        frame_rate=16,
        load_pre_processed=False,
    ):
        self.video_paths = video_paths
        self.target_sr = target_sr
        self.duration = duration
        self.max_samples = int(duration * target_sr)
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.shift_range = shift_range
        self.shift_prob = shift_prob
        self.frame_rate = frame_rate
        self.resize = T.Resize((224, 224))
        self.load_pre_processed = load_pre_processed

    def count_video_frames_ffprobe(self, path):
        """
        Uses ffprobe to quickly count video packets (frames) using the command from StackOverflow.
        Compatible with .avi, .mp4, .mkv, .webv, etc.
        """
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                path,
            ]
            output = (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            )
            return int(output)
        except Exception as e:
            print(f"[FrameCountError] Failed to count frames for {path}: {e}")
            return -1  # fallback signal

    def sample_indices(self, container, video_path):
        stream = container.streams.video[0]

        try:
            total_frames = stream.frames
        except:
            total_frames = None

        if not total_frames:
            try:
                total_frames = self.count_video_frames_ffprobe(video_path)
            except:
                total_frames = self.n_frames * self.frame_rate + 1  # minimum viable

        converted_len = self.n_frames * self.frame_rate

        if converted_len >= total_frames:
            # For less than 16 seconds videos, extract 16 frames evenly spaced
            end_idx = total_frames
            start_idx = 0
        else:
            # For higher than 16 seconds video, extract a random clip from it
            end_idx = np.random.randint(converted_len, total_frames)
            start_idx = end_idx - converted_len

        indices = np.linspace(start_idx, end_idx, num=self.n_frames)
        return np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    def read_video_pyav(self, path, indices):
        with av.open(path) as container:
            container.seek(0)
            frames = []
            for i, frame in enumerate(container.decode(video=0)):
                if i > indices[-1]:
                    break
                if i in indices:
                    frame = frame.to_ndarray(format="rgb24")
                    resized_frame = cv2.resize(frame, (224, 224))
                    frames.append(resized_frame)
        if len(frames) < self.n_frames:
            last_frame = (
                frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            )
            frames += [last_frame] * (self.n_frames - len(frames))
        return frames

    def extract_audio(self, path):
        try:
            with VideoFileClip(path) as clip:
                audio = clip.audio
                if audio is None:
                    raise ValueError("No audio track")
                array = audio.to_soundarray(fps=self.target_sr).mean(axis=1)
        except:
            array = np.zeros(self.max_samples * 2)
        return array

    def shift_audio(self, waveform, shift_label):
        if shift_label == 0:
            return waveform[: self.max_samples]

        # Ensure shift is at least ±1 second
        while True:
            shift_sec = random.uniform(*self.shift_range)
            if abs(shift_sec) >= 1.0:
                break

        shift_samples = int(shift_sec * self.target_sr)
        total_needed = abs(shift_samples) + self.max_samples

        if total_needed > len(waveform):
            pad_len = total_needed - len(waveform)

            if shift_samples > 0:
                # Forward shift: pad at the end
                pad_source = waveform[-pad_len:]
            else:
                # Backward shift: pad at the start
                pad_source = waveform[:pad_len]

            pad_source = np.pad(
                pad_source, (0, max(0, pad_len - len(pad_source))), mode="wrap"
            )
            noise = np.random.normal(0, np.std(waveform) * 0.1, size=pad_len)
            pad = 0.9 * pad_source + 0.1 * noise

            waveform = np.concatenate([waveform, pad])

        if shift_samples > 0:
            # Forward shift: trim start
            return waveform[shift_samples : shift_samples + self.max_samples]
        else:
            # Backward shift: trim end
            return waveform[0 : self.max_samples] + [-shift_samples]

    def waveform_to_image(self, waveform):
        tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels,
        )(tensor)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        img = mel_db.squeeze(0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = np.uint8(img * 255)
        img = Image.fromarray(img).convert("RGB")
        return self.resize(img)

    def __getitem__(self, idx):
        path = self.video_paths[idx]

        if self.load_pre_processed:
            return np.load(self.video_paths[idx], allow_pickle=True).item()

        # Label: 0 = aligned, 1 = shifted
        shift_label = int(random.random() < self.shift_prob)

        with av.open(path) as container:
            indices = self.sample_indices(container, path)
        frames = self.read_video_pyav(path, indices)

        waveform = self.extract_audio(path)
        shifted = self.shift_audio(waveform, shift_label)
        spectrogram_img = self.waveform_to_image(shifted)

        return {
            "video_frames": np.array(frames, dtype=float),
            "audio_frames": np.array(spectrogram_img, dtype=float),
            "labels": torch.tensor(shift_label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.video_paths)


#############################################################################################################
#                                     Processor AV functions                                                #
#############################################################################################################


class ProcessorAVDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = glob.glob(f"{image_dir}/*.png")
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(img_path)

        if self.transform:
            image = self.transform(image)

        return image.float()


#############################################################################################################
#                               Cross Processor AV functions                                                #
#############################################################################################################


class CrossProcessorAVDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_files = sorted(glob.glob(f"{image_dir}/*_frame.png"))
        self.spec_files = sorted(glob.glob(f"{image_dir}/*_audio.png"))
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        audio_path = self.spec_files[idx]

        image = cv2.imread(img_path)
        audio_spectogram = cv2.imread(audio_path)

        if self.transform:
            image = self.transform(image)
            audio_spectogram = self.transform(audio_spectogram)

        return image.float(), audio_spectogram.float()


#############################################################################################################
#                                                                                                           #
#############################################################################################################


#############################################################################################################
#                                            X-net functions                                                #
#############################################################################################################


def load_UCF_dataset(data_path):

    # Define file paths
    video_path = os.path.join(data_path, "videos")
    class2id_path = os.path.join(data_path, "class2id.json")
    test_split_path = os.path.join(data_path, "testlist01.txt")

    class2id = json.load(open(class2id_path, "r"))
    videos = os.listdir(video_path)
    test_videos = open(test_split_path, "r").readlines()

    test_videos = [video_name.split("/")[1] for video_name in test_videos]

    val_data = []
    val_label = []
    train_data = []
    train_label = []
    for data_name in test_videos:
        checkup_list = [data_name.split(".")[0] in train_name for train_name in videos]
        if any(checkup_list):
            video_idx = np.where(checkup_list)[0][0]
            video_class = videos[video_idx].split("_")[1]
            val_data.append(os.path.join(video_path, videos.pop(video_idx)))
            val_label.append(class2id[video_class])

    train_data.extend([os.path.join(video_path, video_name) for video_name in videos])
    train_label.extend([class2id[video_name.split("_")[1]] for video_name in videos])

    return (train_data, train_label, val_data, val_label)


def load_coin_dataset(data_path):

    # Define data paths
    train_videos_path = os.path.join(data_path, "train/videos")
    val_videos_path = os.path.join(data_path, "eval/videos")

    train_videos = glob.glob(f"{train_videos_path}/*/*")
    val_videos = glob.glob(f"{val_videos_path}/*/*")

    train_labels = [int(video_path.split("/")[-2]) for video_path in train_videos]
    val_labels = [int(video_path.split("/")[-2]) for video_path in val_videos]

    return train_videos, train_labels, val_videos, val_labels


class VideoAudioDataset(Dataset):
    def __init__(
        self,
        video_paths,
        labels,
        video_processor,
        audio_processor,
        target_sr=16000,
        duration=30,
        n_frames=16,
        n_mels=128,
        frame_rate=16,
        load_pre_processed=False,
        full_xnet_training=True,
    ):
        self.video_paths = video_paths
        self.labels = labels
        self.video_processor = video_processor
        self.audio_processor = audio_processor
        self.target_sr = target_sr
        self.duration = duration
        self.n_frames = n_frames
        self.n_mels = n_mels
        self.max_samples = int(duration * target_sr)
        self.frame_rate = frame_rate
        self.resize = T.Resize((224, 224))  # For spectrogram image
        self.load_pre_processed = load_pre_processed
        self.full_xnet_training = full_xnet_training

    def count_video_frames_ffprobe(self, path):
        """
        Uses ffprobe to quickly count video packets (frames) using the command from StackOverflow.
        Compatible with .avi, .mp4, .mkv, .webv, etc.
        """
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_packets",
                "-show_entries",
                "stream=nb_read_packets",
                "-of",
                "csv=p=0",
                path,
            ]
            output = (
                subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
            )
            return int(output)
        except Exception as e:
            print(f"[FrameCountError] Failed to count frames for {path}: {e}")
            return -1  # fallback signal

    def sample_indices(self, container, video_path):
        stream = container.streams.video[0]

        try:
            total_frames = stream.frames
        except:
            total_frames = None

        if not total_frames or total_frames < self.n_frames * self.frame_rate:
            try:
                total_frames = self.count_video_frames_ffprobe(video_path)
            except:
                total_frames = self.n_frames * self.frame_rate + 1  # minimum viable

        converted_len = self.n_frames * self.frame_rate

        if True or converted_len >= total_frames:
            # For less than 16 seconds videos, extract 16 frames evenly spaced
            end_idx = total_frames
            start_idx = 0
        else:
            # For higher than 16 seconds video, extract a random clip from it
            end_idx = np.random.randint(converted_len, total_frames)
            start_idx = end_idx - converted_len

        indices = np.linspace(start_idx, end_idx, num=self.n_frames)
        return np.clip(indices, start_idx, end_idx - 1).astype(np.int64)

    def read_video_cv2(self, path, indices):
        cap = cv2.VideoCapture(path)
        frames = []

        if not cap.isOpened():
            print(f"[VideoOpenError] Failed to open {path}")
            return [np.zeros((224, 224, 3), dtype=np.uint8)] * self.n_frames

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()

        # Pad in case of missing frames
        if len(frames) < self.n_frames:
            last = frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8)
            frames += [last] * (self.n_frames - len(frames))

        return frames

    def extract_audio_with_moviepy(self, path, start_sec=None, end_sec=None):
        with VideoFileClip(path) as clip:
            audio = clip.audio
            if audio is None:
                raise ValueError("No audio track")

            if start_sec is not None and end_sec is not None:
                audio = audio.subclipped(start_sec, end_sec)

            audio_array = audio.to_soundarray(fps=self.target_sr)
            audio_array = audio_array.mean(axis=1)  # to mono
            return audio_array

    def extract_audio_with_torchaudio(self, path, start_sec=None, end_sec=None):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                path,
                "-ar",
                str(self.target_sr),
                "-ac",
                "1",
            ]
            if start_sec is not None and end_sec is not None:
                duration = end_sec - start_sec
                cmd += ["-ss", str(start_sec), "-t", str(duration)]

            cmd += ["-f", "wav", tmp.name]

            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            waveform, _ = torchaudio.load(tmp.name)
            return waveform.squeeze(0).numpy()

    def get_audio(self, path, start_sec=None, end_sec=None):
        try:
            return self.extract_audio_with_moviepy(path, start_sec, end_sec)
        except Exception:
            return self.extract_audio_with_torchaudio(path, start_sec, end_sec)

    def waveform_to_image(self, waveform):
        tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=512,
            n_mels=self.n_mels,
        )(tensor)
        mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
        img = mel_db.squeeze(0).numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img = np.uint8(img * 255)
        img = Image.fromarray(img).convert("RGB")
        return self.resize(img)

    def __getitem__(self, idx):

        if self.load_pre_processed:
            return np.load(self.video_paths[idx], allow_pickle=True).item()

        path = self.video_paths[idx]
        label = self.labels[idx]

        # Video frames
        with av.open(path) as container:
            indices = self.sample_indices(container, path)

        frames = self.read_video_cv2(path, indices)

        # Calculate timestamps based on frame indices and frame rate
        start_frame = indices[0]
        end_frame = indices[-1]

        if end_frame <= self.frame_rate * self.n_frames:
            # Used full clip extraction
            start_sec = None
            end_sec = None
        else:
            start_sec = start_frame / 30
            end_sec = end_frame / 30

        # Extract audio waveform for the same segment as the video clip
        try:
            waveform = self.get_audio(path, start_sec=start_sec, end_sec=end_sec)
        except:
            waveform = np.zeros(int(16 * self.target_sr))

        audio_image = self.waveform_to_image(waveform)

        audio_input = self.audio_processor(audio_image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        video_input = self.video_processor(frames, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        if self.full_xnet_training:
            return {
                "video_pixel_values": video_input,
                "audio_pixel_values": audio_input,
                "frames": np.array(frames, dtype=np.float32),
                "spectogram": np.array(audio_image, dtype=np.float32),
                "labels": torch.tensor(label, dtype=torch.long),
            }

        return {
            "video_pixel_values": video_input,
            "audio_pixel_values": audio_input,
            "labels": torch.tensor(label, dtype=torch.long),
        }

    def __len__(self):
        return len(self.video_paths)


if __name__ == "__main__":

    data_path = "/media/gustavo-furlanetto/KINGSTON/datasets/coin"

    output_path = "/media/gustavo-furlanetto/KINGSTON/datasets/coin/features"

    train_paths, train_labels, val_paths, val_labels = load_coin_dataset(data_path)

    video_processor = VideoMAEImageProcessor.from_pretrained(
        "OpenGVLab/VideoMAEv2-Base"
    )
    audio_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224-in21k"
    )

    train_dataset = VideoAudioDataset(
        train_paths, train_labels, video_processor, audio_processor
    )

    # dataset = VideoAudioDataset(
    #     glob.glob(f"{output_path}/*npy"),
    #     val_labels,
    #     video_processor,
    #     audio_processor,
    #     load_pre_processed=True,
    # )

    # print("[ INFO ] CHecking if everything is ok")
    # for i in tqdm.tqdm(range(len(dataset))):
    #     try:
    #         _ = dataset[i]
    #     except:
    #         print(f"[ ERRO ] Unable to load {dataset.video_paths[i]}")
    #         os.remove(dataset.video_paths[i])

    dataset = VideoAudioDataset(
        val_paths,
        val_labels,
        video_processor,
        audio_processor,
    )

    print("Replacing data")
    os.makedirs(f"{output_path}/train", exist_ok=True)
    os.makedirs(f"{output_path}/val", exist_ok=True)
    for i in tqdm.tqdm(range(len(train_dataset)), desc="train dataset"):
        try:
            file_name = os.path.basename(train_dataset.video_paths[i])
            class_idx = train_dataset.labels[i]
            output_filename = f"{output_path}/train/{class_idx}/{file_name}.npy"

            if os.path.exists(output_filename):
                continue

            os.makedirs(f"{output_path}/train/{class_idx}", exist_ok=True)

            training_item = train_dataset.__getitem__(i)

            if not training_item:
                raise ValueError

            np.save(output_filename, np.array([training_item]))

        except:
            print(f"ERROR: Error processing: {train_dataset.video_paths[i]}")
    for i in tqdm.tqdm(range(len(dataset)), desc="Validation dataset"):
        try:

            file_name = os.path.basename(dataset.video_paths[i])
            class_idx = dataset.labels[i]
            output_filename = f"{output_path}/val/{class_idx}/{file_name}.npy"

            if os.path.exists(output_filename):
                continue

            os.makedirs(f"{output_path}/val/{class_idx}", exist_ok=True)

            training_item = dataset.__getitem__(i)

            if not training_item:
                raise ValueError

            np.save(output_filename, np.array([training_item]))
        except:
            print(f"ERROR: Error processing: {dataset.video_paths[i]}")
