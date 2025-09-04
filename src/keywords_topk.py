# Top-K Keyword phrases from a user vector

#from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Tuple, Literal, Union

import numpy as np
import pandas as pd


ReturnType = Literal["df", "list"]

# def _load_user_vec_from_api(user_id: str, api_url: str = "http://localhost:8000") -> np.ndarray:
#     resp = requests.get(f"{api_url}/user_profile/{user_id}", params={"save": "false"})
#     data = resp.json()
#     arr = np.array(data["preference"], dtype=np.float32)
#     if arr.ndim == 1:
#         arr = arr[None, :]
#     return arr



@dataclass(frozen=True)

class KeywordTopKConfig:
    kw_mat_path: str = "embeddings/keywords_embeddings.npy"
    phrases_json_path: str = "json_files/keywords_index.json"
    top_k: int = 5

def _l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        n = np.linalg.norm(X)
        return (X / max(n, eps))[None, :]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, eps)

def load_keywords(cfg: KeywordTopKConfig) -> Tuple[np.ndarray, List[str]]:
    kw_mat = np.load(cfg.kw_mat_path).astype(np.float32)
    with open(cfg.phrases_json_path, "r") as f:
        data = json.load(f)
    phrase_texts = [x["text"] for x in data]
    if len(phrase_texts) != kw_mat.shape[0]:
        raise ValueError("JSON/text count must match embedding rows.")
    return kw_mat, phrase_texts

def topk_phrases_for_user(
    user_pref_vec: np.ndarray,
    cfg: KeywordTopKConfig = KeywordTopKConfig(),
    return_type: ReturnType = "df",
) -> Union[pd.DataFrame, List[str]]:

    # user_pref_vec = np.load("user_vec.npy")
    df = topk_phrases_for_user(user_pref_vec)

    """
    Inputs:  user_pref_vec shape (512,) or (1,512); artifacts from cfg.
    Output:  either a DataFrame with rank/index/Phrase/cosine, or a list[str] of phrases.
    """
    kw_mat, phrase_texts = load_keywords(cfg)
    kw_norm = _l2_normalize_rows(kw_mat)
    user_vec = _l2_normalize_rows(user_pref_vec)

    sims = (user_vec @ kw_norm.T).ravel()
    k = int(min(cfg.top_k, sims.size))
    topk_idx = np.argpartition(sims, -k)[-k:]
    topk_idx = topk_idx[np.argsort(sims[topk_idx])[::-1]]
    topk_scores = sims[topk_idx]

    if return_type == "list":
        return [phrase_texts[i] for i in topk_idx]

    return pd.DataFrame({
        "rank":   np.arange(1, k+1, dtype=int),
        "index":  topk_idx.astype(int),
        "Phrase": [phrase_texts[i] for i in topk_idx],
        "cosine": topk_scores.astype(float),
    })
