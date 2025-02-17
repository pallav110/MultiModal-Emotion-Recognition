import pandas as pd
import numpy as np
import random
import os
import time
import multiprocessing as mp
from tqdm import tqdm
from nltk.corpus import wordnet
import spacy
from deep_translator import GoogleTranslator
import nltk

# Load spaCy with limited components to save memory
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])  # Load only tokenizer
nltk.download('wordnet')

# Paths
CSV_BASE_DIR = "D:/MELD/MELD.Raw/csv"
OUTPUT_DIR = "D:/MELD/processed/augmented_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset File
TRAIN_FILE = "train_sent_emo.csv"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "augmented_train.csv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "checkpoint.csv")

# Read CSV in Chunks to Save Memory
print("Loading training dataset in chunks...")
chunksize = 5000  # Adjust based on available RAM
chunk_list = []

for chunk in pd.read_csv(os.path.join(CSV_BASE_DIR, TRAIN_FILE), chunksize=chunksize, dtype={"Emotion": "category"}):
    chunk_list.append(chunk)

train_df = pd.concat(chunk_list, ignore_index=True)
del chunk_list  # Free up memory

print(f"Training dataset loaded. Shape: {train_df.shape}")

emotions = train_df["Emotion"].unique()

# Load checkpoint if available
if os.path.exists(CHECKPOINT_FILE):
    print("Resuming from last checkpoint...")
    augmented_data = pd.read_csv(CHECKPOINT_FILE).to_dict("records")
else:
    augmented_data = []

# Function for Synonym Replacement
def synonym_replacement(text):
    words = text.split()
    augmented_words = [
        random.choice(wordnet.synsets(word)[0].lemmas()).name()
        if wordnet.synsets(word) else word for word in words
    ]
    return " ".join(augmented_words)

# Function for Back Translation with retries
def back_translate_text(text, mid_lang="fr", retries=3):
    for _ in range(retries):
        try:
            translated = GoogleTranslator(source="auto", target=mid_lang).translate(text)
            return GoogleTranslator(source=mid_lang, target="en").translate(translated)
        except Exception:
            time.sleep(1)
    return text  # Return original text if translation fails

# Function to augment a single sample
def augment_sample(args):
    text, emotion, row = args
    try:
        augmented_text = back_translate_text(text)
        augmented_text = synonym_replacement(augmented_text)
        return {**row, "Utterance": augmented_text, "Emotion": emotion}
    except Exception as e:
        print(f"Error processing: {e}")
        return None

# 🛠 Ensure script runs properly on Windows
if __name__ == "__main__":
    print("Starting data augmentation...")

    # Determine maximum class count
    max_count = train_df["Emotion"].value_counts().max()
    augmentation_tasks = []

    for emotion in emotions:
        emotion_data = train_df[train_df["Emotion"] == emotion].reset_index(drop=True)
        num_to_generate = max_count - len(emotion_data)

        if num_to_generate > 0:
            sampled_rows = np.random.choice(len(emotion_data), num_to_generate, replace=True)
            augmentation_tasks.extend(
                [(emotion_data.iloc[i]["Utterance"], emotion, emotion_data.iloc[i].to_dict()) for i in sampled_rows]
            )

    # **Multiprocessing with Limited CPU Usage**
    print(f"Generating {len(augmentation_tasks)} samples using multiprocessing...")
    with mp.Pool(processes=4) as pool:  # Limit to 4 processes to save memory
        augmented_samples = list(tqdm(pool.imap(augment_sample, augmentation_tasks), total=len(augmentation_tasks)))

    # **Filter out failed augmentations**
    augmented_data.extend([s for s in augmented_samples if s])

    # **Convert to DataFrame**
    augmented_df = pd.DataFrame(augmented_data)
    augmented_train_df = pd.concat([train_df, augmented_df], ignore_index=True)

    # **Save Data**
    print("Saving augmented dataset...")
    augmented_train_df.to_csv(OUTPUT_FILE, index=False)
    pd.DataFrame(augmented_data).to_csv(CHECKPOINT_FILE, index=False)  # Save checkpoint
    print(f"✅ Augmented dataset saved at {OUTPUT_FILE}")
