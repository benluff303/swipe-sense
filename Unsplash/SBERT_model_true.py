from sentence_transformers import util
from collections import defaultdict
import pandas as pd

def top_locations_from_embeddings(
    review_embeddings,
    locations,
    keyword_embeddings,
    keyword_strings,
    top_k: int = 5
):
    """
    Compute top locations per keyword group using precomputed embeddings.

    Args:
        review_embeddings: torch.Tensor
        locations: list of strings corresponding to review_embeddings
        keyword_embeddings: torch.Tensor
        keyword_strings: list of keyword strings
        top_k: number of top locations to return per keyword group

    Returns:
        pd.DataFrame with columns ['keywords', 'location', 'location_score']
    """

    results = []

    for i, kw_emb in enumerate(keyword_embeddings):
        similarities = util.cos_sim(kw_emb, review_embeddings)[0]

        # Aggregate by location
        location_sims = defaultdict(list)
        for idx, sim in enumerate(similarities):
            location_sims[locations[idx]].append(sim.item())

        # Average per location
        location_avg_scores = [(loc, sum(sims)/len(sims)) for loc, sims in location_sims.items()]

        # Top K locations
        top_locations = sorted(location_avg_scores, key=lambda x: x[1], reverse=True)[:top_k]

        for loc, score in top_locations:
            results.append({
                'keywords': keyword_strings[i],
                'location': loc,
                'location_score': score
            })

    return pd.DataFrame(results)
