import os
import pandas as pd
import numpy as np
import re
import json
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel

# 📂 Paths
CSV_BASE_DIR = "D:/MELD/MELD.Raw/csv"
OUTPUT_DIR = "D:/MELD/processed/text_features/"
LOG_FILE = os.path.join(OUTPUT_DIR, "log.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 📄 Dataset Files
DATASETS = {
    "train": "train_sent_emo.csv",
    "dev": "dev_sent_emo.csv",
    "test": "test_sent_emo.csv"
}

# Load BERT Tokenizer & Model
print("🚀 Loading BERT Tokenizer and Model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
print("✅ BERT Model Loaded Successfully!")

def clean_text(text):
    """Clean text by removing special characters and extra spaces."""
    print(f"🔹 Cleaning text: {text[:50]}...")  # Debug: Show first 50 chars
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"\s+", " ", text)  # Remove extra spaces
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    text = text.strip()  # Remove leading/trailing spaces
    return text

def extract_tfidf_features(texts):
    """Convert text into numerical features using TF-IDF."""
    print("🚀 Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"✅ TF-IDF Extraction Complete! Shape: {tfidf_matrix.shape}")
    return tfidf_matrix.toarray()

def extract_bert_embeddings(texts):
    """Convert text into numerical features using BERT embeddings."""
    print("🚀 Extracting BERT embeddings...")
    embeddings = []
    
    for text in tqdm(texts, desc="🔄 Processing BERT Embeddings"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=50)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        embeddings.append(cls_embedding)

    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"✅ BERT Embeddings Extraction Complete! Shape: {embeddings.shape}")
    return embeddings

# 📌 Process Each Dataset
text_log = {}

for dataset, filename in DATASETS.items():
    print(f"\n📂 Processing `{dataset}` dataset...")

    # Load CSV
    csv_path = os.path.join(CSV_BASE_DIR, filename)
    print(f"📥 Loading CSV file: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ CSV Loaded! Total Samples: {len(df)}")

    # Extract text & labels
    print(f"🔍 Cleaning text for {dataset}...")
    texts = [clean_text(text) for text in tqdm(df["Utterance"].astype(str), desc="🧼 Cleaning Text")]
    labels = df["Emotion"].tolist()

    # TF-IDF Features
    print(f"\n🔹 Extracting TF-IDF features for `{dataset}`...")
    tfidf_features = extract_tfidf_features(texts)
    tfidf_path = os.path.join(OUTPUT_DIR, f"{dataset}_tfidf.npy")
    np.save(tfidf_path, tfidf_features)
    print(f"💾 TF-IDF Features Saved: {tfidf_path}")

    # BERT Features
    print(f"\n🔹 Extracting BERT embeddings for `{dataset}`...")
    bert_features = extract_bert_embeddings(texts)
    bert_path = os.path.join(OUTPUT_DIR, f"{dataset}_bert.npy")
    np.save(bert_path, bert_features)
    print(f"💾 BERT Embeddings Saved: {bert_path}")

    # Log dataset details
    text_log[dataset] = {
        "samples": len(texts),
        "tfidf_shape": tfidf_features.shape,
        "bert_shape": bert_features.shape
    }

# Save Log
with open(LOG_FILE, "w") as log_file:
    json.dump(text_log, log_file, indent=4)

print("\n🎉 ✅ Text preprocessing complete! Features saved.")
