import json
import csv
import yaml


import moviepy as mpy
import tempfile
import os
import soundfile as sf

import librosa


def save_results(video_frames, audio_spectrogram, output_path, fps=10, sr=16000):
    """
    Saves reconstructed video with attached audio.

    Args:
        video_frames (List[np.ndarray]): List of HxWx3 video frames.
        audio_spectrogram (np.ndarray): Audio spectrogram to be converted back to waveform.
        output_path (str): Path to save the final video file.
        fps (int): Frame rate of the video.
        sr (int): Sample rate of audio.
    """

    # Convert spectrogram back to waveform
    audio_db = librosa.db_to_amplitude(audio_spectrogram)
    audio_waveform = librosa.griffinlim(audio_db)

    # Save temporary audio file
    tmp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(tmp_dir, "temp_audio.wav")
    sf.write(audio_path, audio_waveform, sr)

    # Create video clip
    video_frames = [frame * 255.0 for frame in video_frames]
    clip = mpy.ImageSequenceClip(video_frames, fps=fps)
    audio = mpy.AudioFileClip(audio_path)
    clip = clip.with_audio(audio)

    # Write final video
    clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    # Cleanup
    clip.close()
    audio.close()


# JSON Functions
def read_json(file_path):
    """
    Read a JSON file and return the data as a dictionary.

    Parameters:
    file_path (str): Path to the JSON file.

    Returns:
    dict: Data contained in the JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def write_json(data, file_path, indent=4):
    """
    Write data to a JSON file with pretty printing.

    Parameters:
    data (dict): Data to be written to the JSON file.
    file_path (str): Path to the JSON file.
    indent (int, optional): Number of spaces for indentation. Default is 4.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)


# CSV Functions
def read_csv(file_path):
    """
    Read a CSV file and return the data as a list of dictionaries.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    list of dict: Data contained in the CSV file, where each row is a dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [row for row in reader]


def write_csv(data, file_path, fieldnames):
    """
    Write a list of dictionaries to a CSV file.

    Parameters:
    data (list of dict): Data to be written to the CSV file, where each dictionary represents a row.
    file_path (str): Path to the CSV file.
    fieldnames (list of str): List of field names for the CSV file header.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


# Text Functions
def read_text(file_path):
    """
    Read a text file and return its content as a string.

    Parameters:
    file_path (str): Path to the text file.

    Returns:
    str: Content of the text file.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def write_text(data, file_path):
    """
    Write a string to a text file.

    Parameters:
    data (str): String to be written to the text file.
    file_path (str): Path to the text file.
    """
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(data)


# Append to Text File
def append_text(data, file_path):
    """
    Append a string to a text file.

    Parameters:
    data (str): String to be appended to the text file.
    file_path (str): Path to the text file.
    """
    with open(file_path, "a", encoding="utf-8") as file:
        file.write(data + "\n")


def load_yaml(file):
    """
    Load a yaml file from path

    Args:
        file -> str: Path to the yaml file

    Return:
        yaml_file -> dict: Yaml file loaded


    """

    with open(file, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)

    return config


def save_yaml(file, yaml_obj):
    """
    Save a yaml file from path

    Args:
        file -> str: Path to the yaml file

        yaml_obj -> dict: Dict with the yaml content

    Return:
        success -> boll: Flag indicating success or failure
    """

    try:
        yaml.dump(yaml_obj, open(file, "w"), indent=2)
    except:
        print("Failed to save the YAML file")
        return False

    return True
