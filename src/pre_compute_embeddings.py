import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from itertools import zip_longest
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from x_net import XNet
from x_net.model_utils.video_utils import video_loader
from utils.common import read_json


def collect_and_split_videos_by_size(video_dir, short_ratio=0.6):
    valid_exts = (".mp4", ".mkv", ".avi", ".webm")
    class_videos = defaultdict(list)

    for root, _, files in os.walk(video_dir):
        for fname in files:
            if fname.endswith(valid_exts):
                class_idx = os.path.relpath(root, video_dir)
                full_path = os.path.join(root, fname)
                class_videos[class_idx].append(full_path)

    interleaved = []
    for grouped in zip_longest(*class_videos.values()):
        for item in grouped:
            if item is not None:
                interleaved.append(item)

    interleaved_sorted = sorted(interleaved, key=os.path.getsize)
    cutoff = int(len(interleaved_sorted) * short_ratio)
    short_videos = interleaved_sorted[:cutoff]
    long_videos = interleaved_sorted[cutoff:]

    return short_videos, long_videos


def process_video(full_video_path):
    try:
        rel_path = os.path.relpath(full_video_path, VIDEO_DIR)
        output_path = os.path.join(OUTPUT_DIR, os.path.splitext(rel_path)[0] + ".npy")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if os.path.exists(output_path):
            return f"Skipped: {rel_path}"

        model = XNet(read_json(CONFIG_PATH)["model_parameters"]).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model.eval()

        frames, waveform = video_loader(full_video_path, use_whole_video=True)

        feats = model.extract_video_features(frames, waveform).cpu().numpy()
        np.save(output_path, feats)

        return f"Done: {rel_path}"
    except Exception as e:
        return f"Failed: {full_video_path} - {e}"


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage:")
        print(
            "  python process_videos_balanced.py <VIDEO_DIR> <OUTPUT_DIR> <NUM_WORKERS> <CONFIG_PATH> <short|long>"
        )
        sys.exit(1)

    VIDEO_DIR = str(sys.argv[1])
    OUTPUT_DIR = str(sys.argv[2])
    NUM_WORKERS = min(int(sys.argv[3]), cpu_count())
    CONFIG_PATH = str(sys.argv[4])
    MODE = str(sys.argv[5]).lower()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Split video files
    short_videos, long_videos = collect_and_split_videos_by_size(VIDEO_DIR)
    if MODE == "short":
        video_files = short_videos
    elif MODE == "long":
        video_files = long_videos
    else:
        raise ValueError("Invalid mode. Use 'short' or 'long'.")

    print(
        f"[INFO] Processing {len(video_files)} {MODE} videos with {NUM_WORKERS} worker(s)."
    )

    with Pool(processes=NUM_WORKERS) as pool:
        for result in tqdm(
            pool.imap_unordered(process_video, video_files),
            total=len(video_files),
            desc=f"Processing {MODE} videos",
        ):
            print(result)
