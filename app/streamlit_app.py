import streamlit as st
import pandas as pd
import ast

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="YouTube Tag Recommender", layout="wide")

# ----------------------------
# Safe conversion from string to list
# ----------------------------
def safe_literal_eval(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return []
    return val

# ----------------------------
# Load and preprocess data
# ----------------------------
df = pd.read_csv("../data/final_merged_tagged_data.csv")  # 🔁 updated path

# Convert stringified lists safely
df["tfidf_tags"] = df["tfidf_tags"].apply(safe_literal_eval)
df["zero_shot_tags"] = df["zero_shot_tags"].apply(safe_literal_eval)
df["combined_tags"] = df["combined_tags"].apply(safe_literal_eval)

# Combine tags into a string for search
df["all_tags"] = df["combined_tags"].apply(lambda tags: ", ".join(tags))

# ----------------------------
# UI Header
# ----------------------------
st.title("🎬 YouTube Video Tag Recommender System")
st.markdown("Search video summaries and explore AI-generated tag recommendations.")
st.markdown("### 🔎 Search Panel")

# ----------------------------
# Search input
# ----------------------------
search_query = st.text_input("Enter a tag, keyword, or topic:")

# ----------------------------
# Search and Results
# ----------------------------
if search_query:
    filtered_df = df[df["all_tags"].str.contains(search_query, case=False, na=False)]

    if not filtered_df.empty:
        st.success(f"✅ Found {len(filtered_df)} matching result(s). Scroll to explore.")
        st.markdown("---")

        for idx, row in filtered_df.iterrows():
            with st.expander(f"📄 Video {idx + 1}: Click to expand"):
                st.markdown("#### 📝 Summary")
                st.write(row["text"])

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("🧠 **TF-IDF Tags**")
                    st.code(", ".join(row["tfidf_tags"]), language="")

                with col2:
                    st.markdown("🤖 **Zero-shot Tags**")
                    st.code(", ".join(row["zero_shot_tags"]), language="")

                with col3:
                    st.markdown("🌐 **Combined Tags**")
                    st.code(", ".join(row["combined_tags"]), language="")

            st.markdown("---")
    else:
        st.warning("❌ No results found. Try a different keyword.")
else:
    st.info("💡 Enter a keyword above to begin searching.")
