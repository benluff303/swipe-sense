# ranks the top locations for our topk keywords

#from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer

@dataclass(frozen=True)
class LocationRankConfig:
    emb_npy_path: str = "embeddings/Review_embeddings/review_embeddings.npy"
    meta_csv_path: str = "embeddings/Review_embeddings/meta.csv"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    similarity_threshold: float = 0.35
    top_n_per_location: int = 10
    min_reviews_per_location: int = 5
    k_top_locations: int = 5
    location_col: str = "location"

def _l2_normalize(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)

def _embed_phrases(model: SentenceTransformer, phrases: List[str]) -> np.ndarray:
    vecs = model.encode(
        phrases, batch_size=32, convert_to_numpy=True,
        show_progress_bar=False, normalize_embeddings=False
    ).astype(np.float32)
    return _l2_normalize(vecs)

def _load_review_embeddings_and_meta(emb_npy: str, meta_csv: str) -> Tuple[np.ndarray, pd.DataFrame]:
    review_vecs = np.load(emb_npy).astype(np.float32)
    meta = pd.read_csv(meta_csv)
    if len(meta) != len(review_vecs):
        raise ValueError(f"Embeddings rows ({len(review_vecs)}) != meta rows ({len(meta)})")
    return _l2_normalize(review_vecs), meta

def _similarity_matrix(phrase_vecs: np.ndarray, review_vecs: np.ndarray) -> np.ndarray:
    return phrase_vecs @ review_vecs.T  # cosine when L2-normalized

def _aggregate_by_location(
    sims_row: np.ndarray,
    meta: pd.DataFrame,
    loc_col: str,
    threshold: float,
    top_n: int,
    min_reviews: int,
) -> Dict[str, float]:
    mask = sims_row >= threshold
    if not mask.any():
        return {}
    sims_kept = sims_row[mask]
    locs_kept = meta.loc[mask, loc_col].to_numpy()

    order = np.argsort(-sims_kept)  # desc
    sims_sorted = sims_kept[order]
    locs_sorted = locs_kept[order]

    counts: Dict[str, int] = {}
    buckets: Dict[str, List[float]] = {}
    for sim, loc in zip(sims_sorted, locs_sorted):
        c = counts.get(loc, 0)
        if c < top_n:
            buckets.setdefault(str(loc), []).append(float(sim))
            counts[loc] = c + 1

    return {loc: float(np.mean(vals)) for loc, vals in buckets.items() if len(vals) >= min_reviews}

def _normalize_scores_per_phrase(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    v = np.array(list(scores.values()), dtype=np.float32)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax <= vmin:
        return {k: 0.0 for k in scores}
    return {k: (scores[k] - vmin) / (vmax - vmin) for k in scores}

def _combine_phrase_scores(per_phrase_norm: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_phrase_norm:
        return {}
    all_locs = set().union(*[d.keys() for d in per_phrase_norm])
    return {loc: float(np.mean([d.get(loc, 0.0) for d in per_phrase_norm])) for loc in all_locs}

def rank_locations_for_phrases(
    phrases: List[str],
    cfg: LocationRankConfig = LocationRankConfig(),
) -> pd.DataFrame:
    if not phrases:
        raise ValueError("No phrases provided.")

    model = SentenceTransformer(cfg.model_name)
    phrase_vecs = _embed_phrases(model, phrases)

    review_vecs, meta = _load_review_embeddings_and_meta(cfg.emb_npy_path, cfg.meta_csv_path)
    if cfg.location_col not in meta.columns:
        raise KeyError(f"location_col '{cfg.location_col}' not found. Meta columns: {list(meta.columns)}")

    S = _similarity_matrix(phrase_vecs, review_vecs)
    per_phrase_scores = [
        _aggregate_by_location(S[i], meta, cfg.location_col,
                               cfg.similarity_threshold,
                               cfg.top_n_per_location,
                               cfg.min_reviews_per_location)
        for i in range(S.shape[0])
    ]

    per_phrase_norm = [_normalize_scores_per_phrase(d) for d in per_phrase_scores]
    combined = _combine_phrase_scores(per_phrase_norm)

    rows = []
    for loc, score in combined.items():
        row = {"location": loc, "combined_score": score}
        for idx, d in enumerate(per_phrase_norm, start=1):
            row[f"phrase_{idx}_score"] = d.get(loc, 0.0)
        rows.append(row)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        cols = ["rank", "location", "combined_score"] + [f"phrase_{i+1}_score" for i in range(len(phrases))]
        return pd.DataFrame(columns=cols)

    result_df = result_df.sort_values("combined_score", ascending=False).reset_index(drop=True)
    result_df.insert(0, "rank", np.arange(1, len(result_df) + 1))
    return result_df.head(cfg.k_top_locations)
