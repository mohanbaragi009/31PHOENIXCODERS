import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load data
df = pd.read_csv("fixed_mutual_funds_data.csv")
df = df.fillna("")

# Combine metadata for embedding
def combine_metadata(row):
    return f"{row['schemeName']} {row['amcName']} {row['category']} {row['subCategory']} {row['riskOMeter']} {row['objective']}"

df["combined_text"] = df.apply(combine_metadata, axis=1)

# Load model and embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def generate_embeddings():
    return model.encode(df["combined_text"].tolist(), show_progress_bar=True)

embeddings = generate_embeddings()

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Streamlit App
st.title("🧠 Mutual Fund Search")
st.markdown("Search mutual funds using natural language.")

query = st.text_input("Enter your query (e.g. 'high growth in banking sector')", "")

if query:
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), 5)

    st.subheader("🔍 Top Matches")
    for i in I[0]:
        st.markdown(f"""
        ### 🎯 {df.iloc[i]['schemeName']}
        - 🏢 **AMC**: {df.iloc[i]['amcName']}
        - 📂 **Category**: {df.iloc[i]['category']} > {df.iloc[i]['subCategory']}
        - ⚠️ **Risk**: {df.iloc[i]['riskOMeter']}
        - 🧠 **Match Preview**: `{df.iloc[i]['combined_text'][:150]}...`
        """)
