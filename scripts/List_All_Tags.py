import pandas as pd
import ast
import re
from collections import defaultdict

# Load data
df = pd.read_csv("../data/final_merged_tagged_data.csv")

# Function to safely evaluate and clean tags
def safe_eval(val):
    try:
        return ast.literal_eval(val) if isinstance(val, str) else []
    except:
        return []

def clean_tag(tag):
    tag = tag.strip().lower()
    if (
        not tag 
        or tag.isdigit()
        or len(tag) < 3
        or "http" in tag 
        or "www" in tag
        or re.search(r"[^\w\s]", tag)  # remove tags with special chars
    ):
        return None
    return tag

# Process tags
df["combined_tags"] = df["combined_tags"].apply(safe_eval)
all_tags = set()

for tags in df["combined_tags"]:
    for tag in tags:
        cleaned = clean_tag(tag)
        if cleaned:
            all_tags.add(cleaned)

# Sort alphabetically
sorted_tags = sorted(all_tags)

# Group tags by starting letter
grouped_tags = defaultdict(list)
for tag in sorted_tags:
    first_letter = tag[0].upper()
    grouped_tags[first_letter].append(tag)

# Display grouped tags
print("🎯 Cleaned Unique Tags:", len(sorted_tags))
print("\n📚 Tags Grouped Alphabetically:\n")

for letter in sorted(grouped_tags.keys()):
    print(f"\n🔠 {letter}:")
    tags = grouped_tags[letter]
    for i in range(0, len(tags), 5):
        print(", ".join(tags[i:i+5]))






# Optional: save to file
# with open("../data/all_cleaned_tags.txt", "w") as f:
#     for tag in sorted_tags:
#         f.write(f"{tag}\n")
