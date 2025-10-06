import av
import librosa
import numpy as np
import torch
import math
import subprocess
import json
import tempfile

from moviepy import VideoFileClip


def get_video_duration_and_fps(path):
    """Return (duration in seconds, fps) using ffprobe format + stream info."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    info = json.loads(result.stdout)

    # Fallback duration from format (container level)
    duration = float(info["format"]["duration"])

    # r_frame_rate is guaranteed to be in stream[0]
    fps_parts = info["streams"][0]["r_frame_rate"].split("/")
    fps = float(fps_parts[0]) / float(fps_parts[1])
    return duration, fps


def video_loader(
    file_path,
    segment_duration=5.0,
    sample_rate=22050,
    num_frames=16,
    use_whole_video=False,
):
    """
    Yields (frames, waveform) segments from a video file.

    If use_whole_video is True, the entire video is returned as a single segment.
    """
    duration, fps = get_video_duration_and_fps(file_path)
    container = av.open(file_path)

    if use_whole_video:
        segment_indices = [(0, duration)]
    else:
        total_segments = math.floor(duration / segment_duration)
        segment_indices = [
            (i * segment_duration, (i + 1) * segment_duration)
            for i in range(total_segments)
        ]

    # Load full audio waveform
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp_audio:
            clip = VideoFileClip(file_path)
            clip.audio.write_audiofile(
                tmp_audio.name, fps=22050, verbose=False, logger=None
            )
            full_audio, _ = librosa.load(tmp_audio.name, sr=sample_rate, mono=True)
    except:
        full_audio = np.zeros((int((duration + 1) * sample_rate),))

    for start_time, end_time in segment_indices:
        # --- Extract audio chunk ---
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = full_audio[start_sample:end_sample]
        audio_segment = audio_segment * (2**-23)  # match SoundNet preprocessing
        audio_tensor = torch.tensor(audio_segment, dtype=torch.float32)
        audio_tensor = audio_tensor.view(1, 1, -1, 1)  # [1, 1, T, 1]

        # --- Extract video frames ---
        container.seek(int(start_time * av.time_base))
        frames = []
        frame_indices = np.linspace(
            start_time * fps, (end_time - 0.5) * fps, num=num_frames
        ).astype(int)

        i = 0
        for frame in container.decode(video=0):
            if i > frame_indices[-1]:
                break
            if i in frame_indices:
                frames.append(frame.to_ndarray(format="rgb24"))
            i += 1

        if len(frames) < num_frames:
            continue  # skip incomplete segments

    return frames, audio_tensor


if __name__ == "__main__":
    frames, tensor = video_loader(
        "/media/gustavo-furlanetto/KINGSTON/datasets/coin/eval/videos/0/-8NaVGEccgc.mp4.mkv",
        use_whole_video=True,
    )
