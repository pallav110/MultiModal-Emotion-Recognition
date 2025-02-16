import pandas as pd
import numpy as np
import random
from nltk.corpus import wordnet
import spacy
import os
from pybacktrans import BackTranslator
import nltk
from tqdm import tqdm  # Import tqdm for progress bar

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
    print(f"Performing synonym replacement on text: {text[:50]}...")  # Preview the first 50 chars of the text
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
    print(f"Performing back translation on text: {text[:50]}...")  # Preview the first 50 chars of the text
    back_translator = BackTranslator()  # Instantiate the BackTranslator
    back_translated = back_translator.backtranslate(text)  # Get the back-translated object
    return str(back_translated)  # Return the string version of the back-translated text

# Balance the dataset using back-translation and synonym replacement
print("Starting data augmentation...")

for emotion in emotions:
    emotion_data = train_df[train_df["Emotion"] == emotion]
    emotion_count = len(emotion_data)
    print(f"\nOriginal count for {emotion}: {emotion_count}")

    # Reset the index to ensure valid indices for random sampling
    emotion_data = emotion_data.reset_index(drop=True)
    print(f"Data for {emotion} has been reset. Shape: {emotion_data.shape}")

    # Determine how many more samples are needed for underrepresented classes
    max_count = train_df['Emotion'].value_counts().max()
    num_to_generate = max_count - emotion_count
    print(f"Max count in the dataset: {max_count}. Need to generate {num_to_generate} more samples for {emotion}.")

    if num_to_generate > 0:
        print(f"Generating {num_to_generate} more samples for {emotion}...")

        # Use tqdm to show the progress bar for sample generation
        for i in tqdm(range(num_to_generate), desc=f"Processing {emotion} samples", unit="sample"):
            original_text = random.choice(emotion_data["Utterance"])
            
            # Apply back translation
            augmented_text = back_translate_text(original_text)
            print(f"Back-translated text: {augmented_text[:50]}...")  # Preview the first 50 chars of augmented text
            
            # Apply synonym replacement
            augmented_text = synonym_replacement(augmented_text)
            print(f"Synonym replaced text: {augmented_text[:50]}...")  # Preview the first 50 chars of augmented text

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

# Add augmented data to the original train dataframe
augmented_df = pd.DataFrame(augmented_data)
augmented_train_df = pd.concat([train_df, augmented_df], ignore_index=True)
print(f"Augmented dataset created. New shape: {augmented_train_df.shape}")

# Save augmented dataset
print(f"Saving augmented dataset to {OUTPUT_FILE}...")
augmented_train_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Augmented train dataset saved: {OUTPUT_FILE}")
