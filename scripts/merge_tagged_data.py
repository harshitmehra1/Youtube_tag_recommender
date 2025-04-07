import pandas as pd
import ast

# Load both tagged files from the data folder
keyword_df = pd.read_csv("../data/tagged_youtube_data.csv")          # from TF-IDF
zero_shot_df = pd.read_csv("../data/zero_shot_tagged_data.csv")      # from zero-shot

# Rename tag columns
keyword_df.rename(columns={"tags": "tfidf_tags"}, inplace=True)
zero_shot_df.rename(columns={"tags": "zero_shot_tags"}, inplace=True)

# Ensure 'text' column exists in both (create it if needed)
if 'text' not in keyword_df.columns:
    keyword_df['text'] = keyword_df['video_title'].fillna('') + ' ' + keyword_df['video_description'].fillna('')
if 'text' not in zero_shot_df.columns:
    zero_shot_df['text'] = zero_shot_df['video_title'].fillna('') + ' ' + zero_shot_df['video_description'].fillna('')

# Merge on 'text'
merged_df = pd.merge(keyword_df[['text', 'tfidf_tags']], zero_shot_df[['text', 'zero_shot_tags']], on='text', how='inner')

# Convert stringified lists to actual lists
merged_df['tfidf_tags'] = merged_df['tfidf_tags'].apply(ast.literal_eval)
merged_df['zero_shot_tags'] = merged_df['zero_shot_tags'].apply(ast.literal_eval)

# Merge both tag lists and remove duplicates
def merge_lists(list1, list2):
    return list(set(list1 + list2))

merged_df['combined_tags'] = merged_df.apply(lambda row: merge_lists(row['tfidf_tags'], row['zero_shot_tags']), axis=1)

# Convert lists back to strings before saving (so ast.literal_eval will work when loading)
merged_df['tfidf_tags'] = merged_df['tfidf_tags'].apply(repr)
merged_df['zero_shot_tags'] = merged_df['zero_shot_tags'].apply(repr)
merged_df['combined_tags'] = merged_df['combined_tags'].apply(repr)

# Save final merged file to the data folder
merged_df.to_csv("../data/final_merged_tagged_data.csv", index=False)
print("✅ Final merged file saved as 'data/final_merged_tagged_data.csv'")
