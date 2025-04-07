import pandas as pd
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("../data/kaggle_channel_meta.csv")

# Combine title and description
df['text'] = df['video_title'].fillna('') + ' ' + df['video_description'].fillna('')

# Clean function
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    filtered = [word for word in tokens if word not in stop_words]
    return ' '.join(filtered)

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Save cleaned data
df.to_csv("../data/cleaned_youtube_data.csv", index=False)

print("✅ Preprocessing complete! Cleaned data saved to 'cleaned_youtube_data.csv'")
