# app.py
import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import openai

# Load
model = SentenceTransformer('D:/Vijayi_Tasks/Task-2/RAG_QUOTE_RETRIEVEL_GUIDE/fine_tuned_model')
index = faiss.read_index("D:/Vijayi_Tasks/Task-2/RAG_QUOTE_RETRIEVEL_GUIDE/quote_index.faiss")
df = pd.read_pickle("D:/Vijayi_Tasks/Task-2/RAG_QUOTE_RETRIEVEL_GUIDE/quotes_data.pkl")

# UI
st.title("📚 Quote Retrieval App")

query = st.text_input("Enter your quote query")

if st.button("Search"):
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), k=5)
    
    retrieved = df.iloc[I[0]]
    
    st.json({
        "results": [
            {
                "quote": row["quote"],
                "author": row["author"],
                "tags": eval(row["tags"]),
                "score": float(D[0][i])
            }
            for i, (_, row) in enumerate(retrieved.iterrows())
        ]
    })
