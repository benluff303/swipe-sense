from sentence_transformers import SentenceTransformer
import numpy as np
import torch
import pandas as pd

def embed_reviews_and_keywords(df: pd.DataFrame, keywords: list, save_embeddings: bool = True):
    """
    Embed reviews and keyword groups using SBERT.

    Args:
        df: pd.DataFrame containing 'review_text'.
        keywords: list of lists of keyword strings.
        save_embeddings: whether to save embeddings as .npy files.

    Returns:
        review_embeddings: torch.Tensor
        keyword_embeddings: torch.Tensor
        keyword_strings: list of joined keywords
    """

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Convert keyword lists to strings
    keyword_strings = [" ".join(k) if isinstance(k, list) else str(k) for k in keywords]

    # Prepare reviews
    reviews = df['review_text'].fillna("").astype(str).tolist()

    # Compute embeddings
    review_embeddings = model.encode(reviews)
    keyword_embeddings = model.encode(keyword_strings)

    if save_embeddings:
        np.save('review_embeddings.npy', review_embeddings)
        np.save('keyword_embeddings.npy', keyword_embeddings)

    # Load from file and convert to tensors (to match your original code)
    review_embeddings = torch.tensor(np.load("review_embeddings.npy"))
    keyword_embeddings = torch.tensor(np.load("keyword_embeddings.npy"))

    return review_embeddings, keyword_embeddings, keyword_strings
