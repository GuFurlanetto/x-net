import librosa
import soundfile as sf


def process_audio_results(generated_audio, output_file, sample_rate, n_fft, hop_lenght):
    """
    Takes the generated spectograms and produces a audio file at the output folder

    Args:
        generated_audios -> np.array: Spectogram

        output_folder -> Str: Path to output folder

    Return:
        Status -> bool: True if process was completed with no errors
    """

    try:
        # Invert the Mel spectrogram to the linear scale
        audio_signal = librosa.feature.inverse.mel_to_audio(
            generated_audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_lenght,
            norm="slaney",
        )

        # Save the audio to a file
        sf.write(output_file, audio_signal, sample_rate, "PCM_24")

        return True
    except:
        return False
