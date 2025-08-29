from sentence_transformers import SentenceTransformer, util
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import torch

def top_locations(df, keywords):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    keyword_strings = [" ".join(k) if isinstance(k, list) else str(k) for k in keywords]
    list_reviews = df['review_text'].tolist()
    reviews = [str(r) if r is not None else "" for r in list_reviews]
    review_embeddings = model.encode(reviews)
    keyword_embeddings = model.encode(keyword_strings)
    np.save('review_embeddings.npy', review_embeddings)
    np.save('keyword_embeddings.npy', keyword_embeddings)
    review_embeddings_np = np.load("review_embeddings.npy")
    keyword_embeddings_np = np.load("keyword_embeddings.npy")

    # Convert to tensors
    review_embeddings = torch.tensor(review_embeddings_np)
    keyword_embeddings = torch.tensor(keyword_embeddings_np)

    # Collect results
    results = []

    for i, query_emb in enumerate(keyword_embeddings):
        similarities = util.cos_sim(query_emb, review_embeddings)[0]

        # Aggregate similarities by location
        location_sims = defaultdict(list)
        for idx, sim in enumerate(similarities):
            location = df.iloc[idx]['location']
            location_sims[location].append((sim.item(), idx))

        # Compute average similarity per location
        location_avg_scores = []
        for location, sims in location_sims.items():
            avg_score = sum([s[0] for s in sims]) / len(sims)
            location_avg_scores.append((location, avg_score, sims))

        # Sort locations by average similarity
        location_avg_scores_sorted = sorted(location_avg_scores, key=lambda x: x[1], reverse=True)[:5]

        # For each top location, pick top review
        for location, avg_score, sims in location_avg_scores_sorted:
                results.append({
                    'keywords': keyword_strings[i],
                    'location': location,
                    'location_score': avg_score
                })

    return pd.DataFrame(results)
