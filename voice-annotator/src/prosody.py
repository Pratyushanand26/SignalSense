import librosa
import numpy as np
# You might need to install librosa and a package like praat-parselmouth for jitter.
# pip install librosa praat-parselmouth

def extract_prosody_features(audio_file_path):
    """
    Extracts mean pitch, mean intensity, and jitter from an audio file.
    """
    # Load audio file. librosa can load various formats like .mp3, .wav
    y, sr = librosa.load(audio_file_path, sr=None)

    # Calculate pitch (F0) using a pitch tracker
    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C5'))
    pitch = np.mean(f0[voiced_flag])

    # Calculate intensity (RMS energy)
    rms = librosa.feature.rms(y=y)
    intensity = np.mean(rms)

    # For jitter, a more specialized tool like parselmouth is better.
    # Placeholder for jitter calculation.
    # Jitter calculation typically requires converting to a Sound object and using specific Praat functions.
    jitter = 0.0 # This needs to be implemented using a library like parselmouth

    return {
        "pitch": pitch,
        "intensity": intensity,
        "jitter": jitter
    }

if __name__ == "__main__":
    # This is an example of how the function can be called
    audio_file = "../audio/sample_audio.wav"
    features = extract_prosody_features(audio_file)
    print(features)