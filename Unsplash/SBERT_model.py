from sentence_transformers import SentenceTransformer, util
import re
import pandas as pd
import numpy as np
from collections import defaultdict
import torch

def top_locations(df, keywords):
    """This function takes in a DataFrame with locations and various reviews,
    along with a list of groups of keywords based on users 'swiping' profile and
    outputs a DataFrame listing the top five locations and associated similarity
    score for each group of keywords"""
    # Instantiate model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define variables to embed
    keyword_strings = [" ".join(k) if isinstance(k, list) else str(k) for k in keywords]
    list_reviews = df['review_text'].tolist()
    reviews = [str(r) if r is not None else "" for r in list_reviews]

    # Embedding
    review_embeddings = model.encode(reviews)
    keyword_embeddings = model.encode(keyword_strings)

    # Save embeddings
    np.save('review_embeddings.npy', review_embeddings)
    np.save('keyword_embeddings.npy', keyword_embeddings)
    review_embeddings_np = np.load("review_embeddings.npy")
    keyword_embeddings_np = np.load("keyword_embeddings.npy")

    # Convert to tensors
    review_embeddings = torch.tensor(review_embeddings_np)
    keyword_embeddings = torch.tensor(keyword_embeddings_np)

    # Collect results
    results = []

    # Compute the cosine similarity between keyword group embeddings and all review embeddings
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

        # Sort locations by average similarity and pick top 5
        location_avg_scores_sorted = sorted(location_avg_scores, key=lambda x: x[1], reverse=True)[:5]

        # Append top locations
        for location, avg_score in location_avg_scores_sorted:
                results.append({
                    'keywords': keyword_strings[i],
                    'location': location,
                    'location_score': avg_score
                })

    return pd.DataFrame(results)
