# build_rag.py
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

# Load fine-tuned model and data
model = SentenceTransformer('fine_tuned_model')
df = pd.read_csv("cleaned_quotes.csv")

# Encode quotes
embeddings = model.encode(df['quote'].tolist(), show_progress_bar=True)

# FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# Save index and data
faiss.write_index(index, "quote_index.faiss")
df.to_pickle("quotes_data.pkl")
