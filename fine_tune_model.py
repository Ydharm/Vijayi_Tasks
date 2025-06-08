# fine_tune_model.py
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd

df = pd.read_csv("cleaned_quotes.csv")

# Create sentence pairs for training
train_data = [InputExample(texts=[row['quote'], f"{row['author']} {' '.join(eval(row['tags']))}"]) for _, row in df.iterrows()]

model = SentenceTransformer('all-MiniLM-L6-v2')  # Base model
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
train_loss = losses.MultipleNegativesRankingLoss(model)

# Fine-tune
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
model.save("fine_tuned_model")
