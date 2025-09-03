# Pipeline from user preference file to gpt JSON package


import argparse, json, numpy as np
from src.keywords_topk import KeywordTopKConfig, topk_phrases_for_user
from src.location_ranker import LocationRankConfig, rank_locations_for_phrases
from src.outputs import to_api_payload

default_kw = KeywordTopKConfig()
default_loc = LocationRankConfig()

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run user→phrases→locations pipeline and emit JSON.")
    p.add_argument("--user_vec_npy", type=str, default=None)

    # KeywordTopK with defaults from dataclass
    p.add_argument("--kw_mat_path", type=str, default=default_kw.kw_mat_path)
    p.add_argument("--phrases_json_path", type=str, default=default_kw.phrases_json_path)
    p.add_argument("--top_k", type=int, default=default_kw.top_k)

    # LocationRank with defaults from dataclass
    p.add_argument("--review_emb_npy", type=str, default=default_loc.emb_npy_path)
    p.add_argument("--meta_csv", type=str, default=default_loc.meta_csv_path)
    p.add_argument("--similarity_threshold", type=float, default=default_loc.similarity_threshold)
    p.add_argument("--top_n_per_location", type=int, default=default_loc.top_n_per_location)
    p.add_argument("--min_reviews_per_location", type=int, default=default_loc.min_reviews_per_location)
    p.add_argument("--k_top_locations", type=int, default=default_loc.k_top_locations)
    p.add_argument("--location_col", type=str, default=default_loc.location_col)

    return p.parse_args()

def _load_user_vec(path: str | None) -> np.ndarray:
    if path:
        return np.load(path).astype(np.float32)
    rng = np.random.default_rng(42)
    return rng.standard_normal(512, dtype=np.float32)

if __name__ == "__main__":
    args = _parse_args()

    # Build configs directly from args

    kw_cfg = KeywordTopKConfig(
        kw_mat_path=args.kw_mat_path,
        phrases_json_path=args.phrases_json_path,
        top_k=args.top_k,
    )
    loc_cfg = LocationRankConfig(
        emb_npy_path=args.review_emb_npy,
        meta_csv_path=args.meta_csv,
        similarity_threshold=args.similarity_threshold,
        top_n_per_location=args.top_n_per_location,
        min_reviews_per_location=args.min_reviews_per_location,
        k_top_locations=args.k_top_locations,
        location_col=args.location_col,
    )

    user_vec = _load_user_vec(args.user_vec_npy)

    phrases = topk_phrases_for_user(user_vec, cfg=kw_cfg, return_type="list")
    ranked = rank_locations_for_phrases(phrases, cfg=loc_cfg)
    payload = to_api_payload(ranked, phrases)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
