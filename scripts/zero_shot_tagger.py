import pandas as pd
from transformers import pipeline
from tqdm import tqdm  # optional, for showing progress

# Load cleaned data
df = pd.read_csv("../data/cleaned_youtube_data.csv")

# Use a lighter zero-shot model
model_name = "typeform/distilbert-base-uncased-mnli"
classifier = pipeline("zero-shot-classification", model=model_name)

# Candidate tags (customizable)
candidate_tags = [
    "artificial intelligence", "machine learning", "deep learning", "data science",
    "python", "google", "youtube", "vlog", "review", "unboxing",
    "music", "tutorial", "tech", "comedy", "finance", "fitness"
]

# Function to get top 3 predicted tags
def get_zero_shot_tags(text):
    try:
        result = classifier(text, candidate_labels=candidate_tags, multi_label=True)
        top_indices = sorted(range(len(result['scores'])), key=lambda i: result['scores'][i], reverse=True)[:3]
        return [result['labels'][i] for i in top_indices]
    except Exception as e:
        print(f"⚠️ Error processing text: {text[:50]}... -> {e}")
        return []

# Apply to full dataset with progress bar
tqdm.pandas(desc="🔍 Generating tags")
df['zero_shot_tags'] = df['text'].progress_apply(get_zero_shot_tags)

# Save results
df.to_csv("../data/zero_shot_tagged_data.csv", index=False)
print("✅ Zero-shot tag generation complete! Saved to 'data/zero_shot_tagged_data.csv'")
