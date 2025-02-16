import librosa
import numpy as np
import os
import random
from pydub import AudioSegment
import soundfile as sf
from scipy.io.wavfile import write

# Directory paths
RAW_AUDIO_DIR = "D:/MELD/MELD.Raw/"
AUGMENTED_AUDIO_DIR = "D:/MELD/processed/augmented_audio"

# List of audio files (example: .wav or .mp3)
audio_files = [os.path.join(RAW_AUDIO_DIR, f) for f in os.listdir(RAW_AUDIO_DIR) if f.endswith('.wav')]

# Ensure output directory exists
os.makedirs(AUGMENTED_AUDIO_DIR, exist_ok=True)

# Function to add noise to the audio
def add_noise(audio, noise_level=0.005):
    """Add random noise to an audio signal."""
    noise = np.random.randn(len(audio))
    audio = audio + noise_level * noise
    audio = np.clip(audio, -1.0, 1.0)  # Ensure values stay within valid range
    return audio

# Function to pitch shift the audio
def pitch_shift(audio, sr, n_steps=2):
    """Shift the pitch of the audio."""
    return librosa.effects.pitch_shift(audio, sr, n_steps=n_steps)

# Function to time stretch the audio
def time_stretch(audio, rate=1.2):
    """Stretch the audio by a given rate."""
    return librosa.effects.time_stretch(audio, rate)

# Function to change volume (adjust gain)
def change_volume(audio, gain_db=6):
    """Change the volume of the audio by a certain number of decibels."""
    return audio * (10 ** (gain_db / 20))

# Function to process and augment the audio
def augment_audio(audio_file):
    """Augment audio by applying pitch shifting, time stretching, and noise addition."""
    # Load audio using librosa (it supports wav, mp3, etc.)
    audio, sr = librosa.load(audio_file, sr=None)

    # Apply audio augmentations
    augmented_audio = audio
    
    # 1. Add noise
    augmented_audio = add_noise(augmented_audio)

    # 2. Apply pitch shift
    augmented_audio = pitch_shift(augmented_audio, sr)

    # 3. Apply time stretching
    augmented_audio = time_stretch(augmented_audio)

    # 4. Adjust volume
    augmented_audio = change_volume(augmented_audio, gain_db=random.uniform(-3, 3))  # Randomize the volume change

    return augmented_audio, sr

# Function to save the augmented audio
def save_augmented_audio(augmented_audio, sr, original_audio_file, output_dir):
    """Save the augmented audio to the output directory."""
    base_filename = os.path.basename(original_audio_file)
    augmented_filename = f"augmented_{base_filename}"

    # Save augmented audio as WAV file using soundfile
    augmented_path = os.path.join(output_dir, augmented_filename)
    sf.write(augmented_path, augmented_audio, sr)

    print(f"Saved augmented audio: {augmented_path}")

# Main process: Loop through audio files and augment them
for audio_file in audio_files:
    print(f"Processing {audio_file}...")
    
    # Perform audio augmentation
    augmented_audio, sr = augment_audio(audio_file)
    
    # Save the augmented audio
    save_augmented_audio(augmented_audio, sr, audio_file, AUGMENTED_AUDIO_DIR)

print("Audio augmentation complete!")
