# 🎬 YouTube Tag Recommender System

An AI-powered web app to generate smart, SEO-friendly tags for YouTube videos using TF-IDF and Zero-Shot Learning techniques. Built with Streamlit and HuggingFace Transformers.

---

## 📁 Project Structure

```
.
├── app/
│   └── app.py                      # Streamlit frontend
├── data/
│   └── final_merged_tagged_data.csv
├── scripts/
│   ├── clean_data.py
│   ├── extract_all_tags.py
│   ├── merge_tagged_data.py
│   ├── tfidf_tagging.py
│   └── zero_shot_tagging.py
├── README.md
```

---

## 🚀 Features

- 🔍 Search by any keyword or tag
- 🧠 Generate TF-IDF based tags from video summaries
- 🤖 Predict tags using Zero-Shot Classification (NLP)
- 🌐 View all combined tags in a clean UI
- 📊 Keyword-based and model-based tag comparison

---

## 🛠️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/youtube-tag-recommender.git
cd youtube-tag-recommender
```

### 2. Create and activate a virtual environment (optional but recommended)

```bash
python -m venv venv
# Activate:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
cd app
streamlit run app.py
```

---

## 🧠 Models & Techniques

- **TF-IDF** for extracting top keywords
- **Zero-Shot Learning** using `distilbert-base-uncased-mnli`
- **Combined Tagging** for broader coverage

---

## 📦 Dependencies

- `pandas`
- `scikit-learn`
- `transformers`
- `streamlit`
- `tqdm`

---

## 💡 Sample Tags

| Tag Type      | Example Tags                           |
|---------------|----------------------------------------|
| TF-IDF        | `python`, `data`, `model`              |
| Zero-Shot     | `data science`, `tutorial`, `fitness`  |
| Combined      | All of the above + manually added tags |

---

## 🔮 Future Improvements

- Upload and tag new video summaries dynamically
- Support larger Zero-Shot models
- Deploy on Streamlit Cloud / HuggingFace Spaces

