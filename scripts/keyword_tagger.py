import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load cleaned data
df = pd.read_csv("../data/cleaned_youtube_data.csv")

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # You can increase limit if needed
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])

# Get feature (word) names
feature_names = vectorizer.get_feature_names_out()

# Function to extract top N keywords from a row
def extract_keywords(row_idx, top_n=5):
    row = tfidf_matrix[row_idx].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]  # Indices of top N values
    top_keywords = [feature_names[i] for i in top_indices]
    return top_keywords

# Apply to all rows
df['tfidf_tags'] = [extract_keywords(i) for i in range(tfidf_matrix.shape[0])]

# Save to CSV
df.to_csv("../data/tagged_youtube_data.csv", index=False)

print("✅ TF-IDF tag generation complete! Saved to 'data/tagged_youtube_data.csv'")
