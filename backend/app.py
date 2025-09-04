# app.py
# FastAPI service: batch pipeline on startup, then /next and /feedback endpoints.

from __future__ import annotations
from typing import List, Tuple, Dict
import random, json
from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import FastAPI, Response
from pydantic import BaseModel
from backend.src.keywords_topk import topk_phrases_for_user
from backend.src.location_ranker import rank_locations_for_phrases

from backend.config import (
    USE_GCS, GCS_BUCKET, GCS_PREFIX, LOCAL_DIRS, MAX_IMAGES, SEED,
    USER_DB_PATH, STATE_DIR,
    MODEL_ID,
    MIN_EDGE_DEFAULT, MIN_W_DEFAULT, MIN_H_DEFAULT,
    DETECT_ZOOM_DEFAULT, ZOOM_CENTER_RATIO_DEFAULT,
    DETECT_CUTOUT_DEFAULT, SOLID_BG_FRAC_DEFAULT, ALPHA_FRAC_DEFAULT,
    USE_NEG_SEM_DEFAULT, WEIRD_THRESH_DEFAULT,
    PLACES_ONLY_DEFAULT, PLACE_MIN_DEFAULT,
    ALPHA_DEFAULT, ETA_INIT_DEFAULT, ETA_MIN_DEFAULT, ETA_DECAY_SPAN_DEFAULT, USE_DECAY_DEFAULT,
    RECENT_K_DEFAULT, RECENT_W_DEFAULT, FOCUS_GAMMA_DEFAULT,
    DIVERSITY_LAST_K_DEFAULT, DIVERSITY_MIN_COS_DEFAULT, HIDE_EXACT_DUPES_DEFAULT,
    POOL_K_DEFAULT, LAMBDA_DIV_DEFAULT, NEAR_DUPE_THR_DEFAULT, QUALITY_BOOST_DEFAULT,
)
from backend.storage_io import list_gcs_images, list_local_images, load_image_any
from backend.embeddings import load_or_compute_E
from backend.recommender import ImageRecommender
from backend.quality import (
    inspect_images, build_ahash_groups,
    build_or_load_qf_meta, quality_scores_from_qf,
    build_or_load_neg_emb, build_or_load_places_emb,
    compute_quality_mask_v2, place_scores
)
from backend.utils import l2_normalize

# from keywords_topk import topk_phrases_for_user
# from location_ranker import rank_locations_for_phrases

# ---------------------------
# Global runtime state
# ---------------------------
app = FastAPI(title="SwipeSense Recommender API")

paths: List[str] = []
meta = []
ahash_groups: Dict[int, set[int]] = {}
E: np.ndarray | None = None
Q_NORM: np.ndarray | None = None
rec: ImageRecommender | None = None

# user accum (long-term) persistence
user_db: Dict[str, dict] = {}

# ---------------------------
# Persistence helpers
# ---------------------------
def load_user_db() -> Dict[str, dict]:
    try:
        return json.loads(USER_DB_PATH.read_text())
    except Exception:
        return {}

def save_user_db(db: Dict[str, dict]):
    USER_DB_PATH.parent.mkdir(exist_ok=True)
    USER_DB_PATH.write_text(json.dumps(db))

def commit_user_accum(user_id: str, e_vec: np.ndarray, feedback: float):
    """Only positive feedback contributes to long-term accum."""
    global user_db
    ent = user_db.get(user_id, {"accum_sum": None, "accum_weight": 0.0})
    if ent["accum_sum"] is None:
        ent["accum_sum"] = np.zeros(e_vec.shape[0], np.float32).tolist()
    if feedback > 0:
        s = np.array(ent["accum_sum"], dtype=np.float32) + feedback * e_vec
        w = float(ent["accum_weight"] + feedback)
        ent["accum_sum"] = s.astype(np.float32).tolist()
        ent["accum_weight"] = w
        user_db[user_id] = ent
        save_user_db(user_db)

def load_user_profile_into_rec(user_id: str):
    """Set recommender preference to stored accum (if any)."""
    global rec, user_db
    if rec is None:
        return
    ent = user_db.get(user_id, {})
    s = np.array(ent.get("accum_sum", []), dtype=np.float32)
    w = float(ent.get("accum_weight", 0.0))
    if s.size == rec.D and w > 0:
        rec.preference = l2_normalize(s / max(w, 1e-9))
    else:
        base = rec.E.mean(axis=0) + np.random.default_rng(42).normal(0.0, 0.05, size=rec.D)
        rec.preference = l2_normalize(base).astype(np.float32)

#louis
def get_user_preference_from_db(user_id: str):
    """Get the current user profile from the recommender (if any).
        returns a numpy array or None if no profile found.
    """
    global rec, user_db
    if rec is None:
        print("rec is None")
        return None
    ent = user_db.get(user_id, {})
    s = np.array(ent.get("accum_sum", []), dtype=np.float32)
    w = float(ent.get("accum_weight", 0.0))
    if s.size == rec.D and w > 0:
        pref = l2_normalize(s / max(w, 1e-9))
        return pref
    return None

# ---------------------------
# Startup: build the batch pipeline
# ---------------------------
@app.on_event("startup")
def startup_event():
    global paths, meta, ahash_groups, E, Q_NORM, rec, user_db

    random.seed(SEED); np.random.seed(SEED)
    print("starting up")
    # 1) Discover dataset
    if USE_GCS:
        paths = list_gcs_images(GCS_BUCKET, GCS_PREFIX, limit=None)
        print(f"loaded {len(paths)} image paths from bucket") #human
    else:
        paths = list_local_images(LOCAL_DIRS, limit=None)

    #human - commented out the below which was causing issies with path ordering
    # random.shuffle(paths)
    if MAX_IMAGES is not None and len(paths) > MAX_IMAGES:
        print("cutting down to MAX_IMAGES", MAX_IMAGES) #human
        paths = paths[:MAX_IMAGES]
    assert len(paths) > 0, "No images found. Check bucket/prefix or local dirs."
    print(f"[startup] Found images: {len(paths)}")

    # 2) Inspect (quality quick features & duplicates)
    #each meta is of type ImageMeta, made in utils.py
    # meta = inspect_images(paths)
    # # reorder paths according to items we could inspect (drop failures)
    # print("building idx map and paths list from meta")
    # idx_map = {m.path: i for i, m in enumerate(meta)}
    # paths = [m.path for m in meta]
    # ahash_groups = build_ahash_groups(meta)
    # print(f"[startup] Inspected: {len(meta)}")

    # 3) Embeddings: load or compute
    #human -> fingerprint unused, leaving in to avoid refactor
    fingerprint = {"model": MODEL_ID, "use_gcs": USE_GCS, "bucket": GCS_BUCKET if USE_GCS else None, "prefix": GCS_PREFIX if USE_GCS else None, "count": len(paths)}
    print("loading or computing embeddings")
    E, loaded = load_or_compute_E(paths, fingerprint)
    E = l2_normalize(E, axis=1).astype(np.float32)
    N, D = E.shape
    print(f"[startup] E shape: {E.shape}")

    # # 4) Quality: quick features + quality scores
    # print("image quality scores start")
    # qf = build_or_load_qf_meta(paths)
    # Q_NORM = quality_scores_from_qf(qf)
    # print("image quality scores end")

    # # 5) Semantic gates
    # print("negative embeddings start")
    # neg_emb = build_or_load_neg_emb()
    # pos_places, neg_places = build_or_load_places_emb()
    # print("negative embeddings end")

    # # 6) Build quality mask
    # mask = compute_quality_mask_v2(
    #     E, qf,
    #     use_negative_semantics=USE_NEG_SEM_DEFAULT,
    #     weird_thresh=WEIRD_THRESH_DEFAULT,
    #     min_edge=MIN_EDGE_DEFAULT, min_w=MIN_W_DEFAULT, min_h=MIN_H_DEFAULT,
    #     detect_zoom=DETECT_ZOOM_DEFAULT, zoom_center_ratio=ZOOM_CENTER_RATIO_DEFAULT,
    #     detect_cutout=DETECT_CUTOUT_DEFAULT, solid_bg_frac=SOLID_BG_FRAC_DEFAULT, alpha_frac=ALPHA_FRAC_DEFAULT,
    #     neg_txt_emb=neg_emb
    # )

    # if PLACES_ONLY_DEFAULT:
    #     ps = place_scores(E, pos_places, neg_places)
    #     mask &= (ps >= PLACE_MIN_DEFAULT)

    # 7) Recommender
    # rec = ImageRecommender(
    #     E, quality_mask=mask, quality_scores=Q_NORM, ahash_groups=ahash_groups,
    #     alpha=ALPHA_DEFAULT, eta=ETA_INIT_DEFAULT, warmup_n=5,
    #     recent_k=RECENT_K_DEFAULT, recent_weight=RECENT_W_DEFAULT,
    #     focus_gamma=FOCUS_GAMMA_DEFAULT, diversity_last_k=DIVERSITY_LAST_K_DEFAULT, diversity_min_cos=DIVERSITY_MIN_COS_DEFAULT,
    #     hide_exact_dupes=HIDE_EXACT_DUPES_DEFAULT
    # )
    #trying without quality mask/ ahash groups
    #setting quility mask, scores, ahash group + hide exact dupes to None
    rec = ImageRecommender(
        E, quality_mask=None,
        quality_scores=None,
        ahash_groups=None,
        alpha=ALPHA_DEFAULT, eta=ETA_INIT_DEFAULT, warmup_n=5,
        recent_k=RECENT_K_DEFAULT, recent_weight=RECENT_W_DEFAULT,
        focus_gamma=FOCUS_GAMMA_DEFAULT, diversity_last_k=DIVERSITY_LAST_K_DEFAULT, diversity_min_cos=DIVERSITY_MIN_COS_DEFAULT,
        hide_exact_dupes=None
    )


    # Decay setup
    rec.eta0 = ETA_INIT_DEFAULT; rec.eta_min = ETA_MIN_DEFAULT; rec.eta_decay_span = ETA_DECAY_SPAN_DEFAULT; rec.use_decay = USE_DECAY_DEFAULT

    user_db = load_user_db()
    print(f"[startup] Backend ready: N={rec.N} D={rec.D}")

# ---------------------------
# API models
# ---------------------------
class NextRequest(BaseModel):
    user_id: str = "default"

class Feedback(BaseModel):
    user_id: str = "default"
    idx: int
    action: str  # "like" or "swipe"
    dwell: float | None = None

# ---------------------------
# Helpers
# ---------------------------
def compute_feedback_value(action: str, dwell_seconds: float | None) -> float:
    from backend.config import LIKE_BASE, SWIPE_FAST, DWELL_THRESHOLD, DWELL_BONUS
    base = LIKE_BASE if action == "like" else SWIPE_FAST
    # dwell bonus only for swipes (NOT likes)
    if action != "like" and dwell_seconds is not None and dwell_seconds >= DWELL_THRESHOLD:
        base += DWELL_BONUS
    return float(base)

# ---------------------------
# Endpoints
# ---------------------------
@app.post("/next")
def get_next(req: NextRequest):
    global rec, paths
    assert rec is not None
    # Load user's accum profile into the recommender
    load_user_profile_into_rec(req.user_id)

    idx, score = rec.recommend_next_smart(
      pool_k=POOL_K_DEFAULT,
      lambda_div=LAMBDA_DIV_DEFAULT,
      quality_boost=QUALITY_BOOST_DEFAULT,
      near_dupe_thr=NEAR_DUPE_THR_DEFAULT
    )
    if idx is None:
        return {"status": "end"}
    return {"status": "ok", "idx": int(idx), "path": paths[int(idx)], "sim_score": float(score) if score is not None else None}

@app.post("/feedback")
def send_feedback(fb: Feedback):
    global rec, paths, E
    assert rec is not None and E is not None
    idx = int(fb.idx)
    # commit accum (if positive only)
    val = compute_feedback_value(fb.action, fb.dwell)
    commit_user_accum(fb.user_id, rec.E[idx], val)

    # update online preference
    rec._update(idx, float(val))
    return {"status": "ok", "eta_now": float(rec.eta)}

@app.get("/image/{idx}")
def get_image(idx: int):
    """Serve the image bytes (JPEG)."""
    global paths
    path = paths[int(idx)]
    im = load_image_any(path).convert("RGB")
    buf = BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return Response(content=buf.getvalue(), media_type="image/jpeg")

@app.get("/stats")
def stats():
    global rec
    return {
        "N": int(rec.N) if rec else 0,
        "D": int(rec.D) if rec else 0,
        "seen": int(len(rec.seen)) if rec else 0,
    }

#human louis added
@app.get("/user_profile/{user_id}")
def get_user_profile(user_id: str, save=True):
    """
        saves user profile to a numpy file if save=True
        can load this profile for next steps
    """
    print("Getting user preference for user_id:", user_id)
    user_prefs = get_user_preference_from_db(user_id)
    print("User prefs type:", type(user_prefs))
    print("User prefs:", user_prefs)
    if user_prefs is not None:
        if save:
            user_filename = f"user_prefs_{user_id}.npy"
            save_path = STATE_DIR / user_filename
            print("Saving user preferences to file:", save_path)
            np.save(save_path, user_prefs)

        return {"status": "ok", "preference": user_prefs.tolist()}
    else:
        return {"status": "no_profile", "preference": None}




@app.get("/user_to_keywords")
def run_flow_from_user_vector(user_id: str = "louis", use_local_npy: bool = False):
    print("running user vector flow with user:", user_id)
    #TODO make this live swipe info
    #get user vector from local file (ideally will get new vector)
    if use_local_npy:
        print("Loading user vector from local npy file")
        user_vector = np.load("users/user_prefs_louis.npy")
    else:
        print("calculating user preference vector...")
        user_vector = get_user_preference_from_db(user_id)
        if user_vector is None:
            print("something went wrong getting user vector from db?")
            print("have you swiped any/enough images?")
            return {"status": "no_profile", "message": f"No user profile found for user_id {user_id}"}

    # user_vector = np.load("users/user_prefs_louis.npy")
    #cosime similary with keykwords phrases - keywords_topktopk_phrases_for_user()
    top_phrases = topk_phrases_for_user(user_vector)
    print(top_phrases) #is a df with phrases column


    phrases_list = top_phrases['Phrase'].tolist()
    print("phrases list", phrases_list)

    #embed 5 phrases with sbert
    #run similarity with reviews (pre embedded with sbert), lcation ranker py
    print("ranking locations for phrases DF")
    locations_df = rank_locations_for_phrases(phrases_list)
    print(locations_df)

    phrases_out = phrases_list
    locations_out = locations_df[['location', 'combined_score']].to_dict(orient='records')

    print(locations_out)

    return {
        'locations' : locations_out,
        'phrases' : phrases_out,
        'user_id' : user_id,
    }



if __name__ == "__main__":
    # print("running startup_event() manually: ")
    # startup_event()
    run_flow_from_user_vector()

print('end of file app.py')
