import os
import numpy as np
import librosa
import soundfile as sf
import json
from tqdm import tqdm
import subprocess  # For extracting audio from mp4

# 📁 Paths
VIDEO_BASE_DIR = "D:/MELD/MELD.Raw/"
OUTPUT_DIR = "D:/MELD/processed/audio_features/"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.json")
COMBINED_NPY_FILE = os.path.join(OUTPUT_DIR, "all_audio_features.npy")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📂 Dataset Paths
DATASET_PATHS = {
    "train": "train/train_splits",
    "dev": "dev/dev_splits_complete"
}

# 🎵 **Feature Extraction Function**
def extract_audio_features(audio_path, sr=16000, n_mfcc=13):
    """Extracts MFCCs, pitch, and energy features from an audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=sr)

        # 🔹 Extract Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        energy = librosa.feature.rms(y=y)
        pitch, _ = librosa.piptrack(y=y, sr=sr)

        # 🔹 Flatten & Combine Features
        feature_vector = np.concatenate([
            np.mean(mfccs, axis=1),  # MFCCs
            np.mean(energy, axis=1),  # Energy
            np.mean(pitch, axis=1)    # Pitch
        ])
        return feature_vector
    except Exception as e:
        print(f"❌ ERROR processing {audio_path}: {str(e)}")
        return None

# 🎥 **Extract Audio from Video**
def extract_audio_from_video(video_path, output_audio_path):
    """Extracts audio from an mp4 file using FFmpeg."""
    try:
        command = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{output_audio_path}" -y -loglevel error'
        subprocess.run(command, shell=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR extracting audio from {video_path}: {str(e)}")
        return False

# 🚀 **Process All Videos**
audio_log = {}
all_audio_features = []

for dataset, subfolder in DATASET_PATHS.items():
    video_folder = os.path.join(VIDEO_BASE_DIR, subfolder)
    output_folder = os.path.join(OUTPUT_DIR, dataset)
    os.makedirs(output_folder, exist_ok=True)

    video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
    print(f"\n🔍 Found {len(video_files)} videos in {dataset}")

    for idx, video in enumerate(video_files):
        print(f"🎵 Processing {dataset}: {idx+1}/{len(video_files)} - {video}")

        video_path = os.path.join(video_folder, video)
        audio_path = os.path.join(output_folder, os.path.splitext(video)[0] + ".wav")
        feature_path = os.path.join(output_folder, os.path.splitext(video)[0] + ".npy")

        try:
            # 🎙️ Step 1: Extract Audio
            if not os.path.exists(audio_path):
                success = extract_audio_from_video(video_path, audio_path)
                if not success:
                    audio_log[video] = {"error": "Failed to extract audio"}
                    continue

            # 🎼 Step 2: Extract Features
            features = extract_audio_features(audio_path)
            if features is not None:
                np.save(feature_path, features)
                all_audio_features.append(features)

                # 📝 Log audio details
                audio_log[video] = {
                    "feature_shape": features.shape
                }
            else:
                audio_log[video] = {"error": "No features extracted"}

        except Exception as e:
            print(f"❌ ERROR processing {video}: {str(e)}")
            audio_log[video] = {"error": str(e)}

# 📝 Save log file
with open(LOG_FILE, "w") as log_file:
    json.dump(audio_log, log_file, indent=4)

# 📌 Combine all features into one file
if all_audio_features:
    combined_features = np.stack(all_audio_features)
    np.save(COMBINED_NPY_FILE, combined_features)
    print(f"✅ Combined audio features saved: {COMBINED_NPY_FILE}")

print("🎉 Audio processing complete!")
