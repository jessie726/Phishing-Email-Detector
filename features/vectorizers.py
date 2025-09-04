from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Classical vectorizers
def tfidf(max_features=1000, ngram_range=(1,2)):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    return tfidf

def bow(max_features=1000, ngram_range=(1,2)):
    bow = CountVectorizer(max_features=max_features, ngram_range=ngram_range) 
    return bow


def get_bert_embeddings(texts, batch_size=32, max_len=128, device=None):
    """
    Generate BERT embeddings in batches to avoid memory issues.
    Uses GPU if available, else falls back to CPU.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.to(device)
    bert_model.eval()

    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :]

        all_embeddings.append(batch_embeddings.cpu().numpy())

        # free memory
        del inputs, outputs, batch_embeddings
        torch.cuda.empty_cache()

    return np.vstack(all_embeddings)