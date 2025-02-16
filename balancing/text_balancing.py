import pandas as pd
import numpy as np
import random
from nltk.corpus import wordnet
import spacy
import os
from pybacktrans import BackTranslator
import nltk
from tqdm import tqdm
import concurrent.futures  # For parallel processing
import threading

# Load spaCy model for synonym replacement
nlp = spacy.load("en_core_web_sm")
nltk.download('wordnet')

# Paths
CSV_BASE_DIR = "D:/MELD/MELD.Raw/csv"
OUTPUT_DIR = "D:/MELD/processed/augmented_text"

# 📄 Dataset Files
TRAIN_FILE = "train_sent_emo.csv"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "augmented_train.csv")

# Load Train Dataset
print("Loading the training dataset...")
train_df = pd.read_csv(os.path.join(CSV_BASE_DIR, TRAIN_FILE))
print(f"Training dataset loaded. Shape: {train_df.shape}")

# List of emotions in the dataset
emotions = train_df["Emotion"].unique()
print(f"List of emotions found in the dataset: {emotions}")

# Create a dictionary to store the augmented data
augmented_data = []

# Function for Synonym Replacement
def synonym_replacement(text):
    """Replace random words in the text with their synonyms."""
    words = text.split()
    augmented_words = []
    
    for word in words:
        # Get synonyms from WordNet
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym = random.choice(synonyms).lemmas()[0].name()
            if synonym != word:  # Replace only if synonym is different
                augmented_words.append(synonym)
            else:
                augmented_words.append(word)
        else:
            augmented_words.append(word)
    
    return " ".join(augmented_words)

# Function for Back Translation (using backtranslate library)
def back_translate_text(text):
    """Translate text to another language and back to generate a paraphrased version."""
    back_translator = BackTranslator()  # Instantiate the BackTranslator
    back_translated = back_translator.backtranslate(text)  # Get the back-translated object
    return str(back_translated)  # Return the string version of the back-translated text

# Checkpoint function to save progress
def save_checkpoint(index):
    with open('checkpoint.txt', 'w') as f:
        f.write(str(index))

# Load checkpoint (last processed index)
def load_checkpoint():
    try:
        with open('checkpoint.txt', 'r') as f:
            return int(f.read())
    except FileNotFoundError:
        return 0  # If no checkpoint exists, start from the beginning

# Balance the dataset using back-translation and synonym replacement
print("Starting data augmentation...")

# Load checkpoint to resume from where it left off
start_index = load_checkpoint()
print(f"Resuming from index {start_index}...")

# Parallel processing function
def process_sample(index, emotion_data, augmented_data, emotion):
    """Process a single sample (this will be run in parallel)."""
    original_text = random.choice(emotion_data["Utterance"])
    
    # Apply back translation
    augmented_text = back_translate_text(original_text)
    
    # Apply synonym replacement
    augmented_text = synonym_replacement(augmented_text)
    
    # Append the augmented example
    augmented_data.append({
        "Utterance": augmented_text,
        "Emotion": emotion,
        "Sentiment": emotion_data["Sentiment"].iloc[0],  # Copying the sentiment
        "Speaker": emotion_data["Speaker"].iloc[0],  # Copying the speaker
        "Dialogue_ID": emotion_data["Dialogue_ID"].iloc[0],  # Copying the dialogue ID
        "Utterance_ID": emotion_data["Utterance_ID"].iloc[0],  # Copying the utterance ID
        "Season": emotion_data["Season"].iloc[0],  # Copying the season
        "Episode": emotion_data["Episode"].iloc[0],  # Copying the episode
        "StartTime": emotion_data["StartTime"].iloc[0],  # Copying the start time
        "EndTime": emotion_data["EndTime"].iloc[0],  # Copying the end time
    })

    # Save checkpoint after each successful augmentation
    save_checkpoint(index)

# Parallelize the augmentation process
def augment_emotion_data(emotion, emotion_data):
    emotion_count = len(emotion_data)
    max_count = train_df['Emotion'].value_counts().max()
    num_to_generate = max_count - emotion_count
    
    if num_to_generate > 0:
        print(f"Generating {num_to_generate} more samples for {emotion}...")
        augmented_data = []

        # Use ThreadPoolExecutor to process samples in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {executor.submit(process_sample, i, emotion_data, augmented_data, emotion): i
                               for i in range(start_index, num_to_generate)}
            for future in concurrent.futures.as_completed(future_to_index):
                # Process completed samples
                future.result()

        return augmented_data
    else:
        return []

# Process each emotion
for emotion in emotions:
    emotion_data = train_df[train_df["Emotion"] == emotion]
    emotion_data = emotion_data.reset_index(drop=True)

    # Augment the emotion data
    augmented_data_for_emotion = augment_emotion_data(emotion, emotion_data)
    augmented_data.extend(augmented_data_for_emotion)

# Add augmented data to the original train dataframe
augmented_df = pd.DataFrame(augmented_data)
augmented_train_df = pd.concat([train_df, augmented_df], ignore_index=True)
print(f"Augmented dataset created. New shape: {augmented_train_df.shape}")

# Save augmented dataset
print(f"Saving augmented dataset to {OUTPUT_FILE}...")
augmented_train_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Augmented train dataset saved: {OUTPUT_FILE}")
