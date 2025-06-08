# data_preparation.py
from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("Abirate/english_quotes", split="train")

# Convert to pandas
df = pd.DataFrame(dataset)

# Clean text
df = df.dropna()
df['quote'] = df['quote'].str.lower().str.strip()
df['author'] = df['author'].str.lower().str.strip()
df['tags'] = df['tags'].apply(lambda x: [tag.lower() for tag in x])

df.to_csv("cleaned_quotes.csv", index=False)
