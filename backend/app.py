# app.py
# FastAPI service: batch pipeline on startup, then /next and /feedback endpoints.

### Othman Update

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

from config import (
    USE_GCS, GCS_BUCKET, GCS_PREFIX, LOCAL_DIRS, MAX_IMAGES, SEED,
    USER_DB_PATH,
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
from storage_io import list_gcs_images, list_local_images, load_image_any
from embeddings import load_or_compute_E
from recommender import ImageRecommender
from quality import (
    inspect_images, build_ahash_groups,
    build_or_load_qf_meta, quality_scores_from_qf,
    build_or_load_neg_emb, build_or_load_places_emb,
    compute_quality_mask_v2, place_scores
)
from utils import l2_normalize

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

# map from image path/url -> index (for frontend that sends path)
path2idx: Dict[str, int] = {}

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

# ---------------------------
# Startup: build the batch pipeline
# ---------------------------
@app.on_event("startup")
def startup_event():
    global paths, meta, ahash_groups, E, Q_NORM, rec, user_db, path2idx

    random.seed(SEED); np.random.seed(SEED)
    print("starting up")

    # 1) Discover dataset
    if USE_GCS:
        paths = list_gcs_images(GCS_BUCKET, GCS_PREFIX, limit=None)
    else:
        paths = list_local_images(LOCAL_DIRS, limit=None)

    if MAX_IMAGES is not None and len(paths) > MAX_IMAGES:
        paths = paths[:MAX_IMAGES]
    assert len(paths) > 0, "No images found. Check bucket/prefix or local dirs."
    print(f"[startup] Found images: {len(paths)}")

    # 2) Inspect (quick features & dup a-hash)
    meta = inspect_images(paths)
    paths = [m.path for m in meta]            # keep only readable ones
    ahash_groups = build_ahash_groups(meta)
    print(f"[startup] Inspected: {len(meta)}")

    # 3) Embeddings: load or compute (MUST match paths 1:1)
    fingerprint = {
        "model": MODEL_ID,
        "use_gcs": USE_GCS,
        "bucket": GCS_BUCKET if USE_GCS else None,
        "prefix": GCS_PREFIX if USE_GCS else None,
        "count": len(paths),
    }
    E_loaded, loaded = load_or_compute_E(paths, fingerprint)
    E = l2_normalize(E_loaded, axis=1).astype(np.float32)
    assert E.shape[0] == len(paths), f"Embedding/paths mismatch: E={E.shape[0]} vs paths={len(paths)}"
    print(f"[startup] E shape: {E.shape} | loaded_from_cache={loaded}")

    # fast lookup by path for /feedback using URLs
    path2idx = {p: i for i, p in enumerate(paths)}

    # 4) Quality quick features + scores
    print("image quality scores start")
    qf = build_or_load_qf_meta(paths)
    Q_NORM = quality_scores_from_qf(qf)
    print("image quality scores end")

    # 5) Semantic gates (negatives & places)
    print("negative embeddings start")
    neg_emb = build_or_load_neg_emb()
    pos_places, neg_places = build_or_load_places_emb()
    print("negative embeddings end")

    # 6) Build quality mask
    mask = compute_quality_mask_v2(
        E, qf,
        use_negative_semantics=USE_NEG_SEM_DEFAULT,
        weird_thresh=WEIRD_THRESH_DEFAULT,
        min_edge=MIN_EDGE_DEFAULT, min_w=MIN_W_DEFAULT, min_h=MIN_H_DEFAULT,
        detect_zoom=DETECT_ZOOM_DEFAULT, zoom_center_ratio=ZOOM_CENTER_RATIO_DEFAULT,
        detect_cutout=DETECT_CUTOUT_DEFAULT, solid_bg_frac=SOLID_BG_FRAC_DEFAULT, alpha_frac=ALPHA_FRAC_DEFAULT,
        neg_txt_emb=neg_emb
    )
    if PLACES_ONLY_DEFAULT:
        ps = place_scores(E, pos_places, neg_places)
        mask &= (ps >= PLACE_MIN_DEFAULT)

    # 7) Recommender
    rec = ImageRecommender(
        E, quality_mask=mask, quality_scores=Q_NORM, ahash_groups=ahash_groups,
        alpha=ALPHA_DEFAULT, eta=ETA_INIT_DEFAULT, warmup_n=5,
        recent_k=RECENT_K_DEFAULT, recent_weight=RECENT_W_DEFAULT,
        focus_gamma=FOCUS_GAMMA_DEFAULT, diversity_last_k=DIVERSITY_LAST_K_DEFAULT, diversity_min_cos=DIVERSITY_MIN_COS_DEFAULT,
        hide_exact_dupes=HIDE_EXACT_DUPES_DEFAULT
    )
    # Decay setup
    rec.eta0 = ETA_INIT_DEFAULT
    rec.eta_min = ETA_MIN_DEFAULT
    rec.eta_decay_span = ETA_DECAY_SPAN_DEFAULT
    rec.use_decay = USE_DECAY_DEFAULT

    user_db = load_user_db()
    print(f"[startup] Backend ready: N={rec.N} D={rec.D}")

# ---------------------------
# API models
# ---------------------------
class NextRequest(BaseModel):
    user_id: str = "default"

class Feedback(BaseModel):
    user_id: str = "default"
    idx: int | None = None
    path: str | None = None
    action: str  # "like" | "swipe" | "next" | "dislike"
    dwell: float | None = None

# ---------------------------
# Helpers
# ---------------------------
def compute_feedback_value(action: str, dwell_seconds: float | None) -> float:
    from config import LIKE_BASE, SWIPE_FAST, DWELL_THRESHOLD, DWELL_BONUS
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
    return {
        "status": "ok",
        "idx": int(idx),
        "path": paths[int(idx)],
        "sim_score": float(score) if score is not None else None
    }

@app.post("/feedback")
def send_feedback(fb: Feedback):
    global rec, paths, E, path2idx
    assert rec is not None and E is not None

    # Resolve index from idx OR path
    if fb.idx is not None:
        idx = int(fb.idx)
    elif fb.path is not None:
        idx = int(path2idx.get(fb.path, -1))
        if idx < 0:
            return {"status": "error", "msg": "path not found"}
    else:
        return {"status": "error", "msg": "idx or path is required"}

    # Normalize action: "next"/"dislike" -> "swipe"
    action = (fb.action or "").lower()
    if action in ("next", "dislike"):
        action = "swipe"

    # Commit accum (positive only) + online update
    val = compute_feedback_value(action, fb.dwell)
    commit_user_accum(fb.user_id, rec.E[idx], val)
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




# from __future__ import annotations
# from typing import List, Tuple, Dict
# import random, json
# from io import BytesIO

# import numpy as np
# from PIL import Image
# from fastapi import FastAPI, Response
# from pydantic import BaseModel

# from config import (
#     USE_GCS, GCS_BUCKET, GCS_PREFIX, LOCAL_DIRS, MAX_IMAGES, SEED,
#     USER_DB_PATH,
#     MODEL_ID,
#     MIN_EDGE_DEFAULT, MIN_W_DEFAULT, MIN_H_DEFAULT,
#     DETECT_ZOOM_DEFAULT, ZOOM_CENTER_RATIO_DEFAULT,
#     DETECT_CUTOUT_DEFAULT, SOLID_BG_FRAC_DEFAULT, ALPHA_FRAC_DEFAULT,
#     USE_NEG_SEM_DEFAULT, WEIRD_THRESH_DEFAULT,
#     PLACES_ONLY_DEFAULT, PLACE_MIN_DEFAULT,
#     ALPHA_DEFAULT, ETA_INIT_DEFAULT, ETA_MIN_DEFAULT, ETA_DECAY_SPAN_DEFAULT, USE_DECAY_DEFAULT,
#     RECENT_K_DEFAULT, RECENT_W_DEFAULT, FOCUS_GAMMA_DEFAULT,
#     DIVERSITY_LAST_K_DEFAULT, DIVERSITY_MIN_COS_DEFAULT, HIDE_EXACT_DUPES_DEFAULT,
#     POOL_K_DEFAULT, LAMBDA_DIV_DEFAULT, NEAR_DUPE_THR_DEFAULT, QUALITY_BOOST_DEFAULT,
# )
# from storage_io import list_gcs_images, list_local_images, load_image_any
# from embeddings import load_or_compute_E
# from recommender import ImageRecommender
# from quality import (
#     inspect_images, build_ahash_groups,
#     build_or_load_qf_meta, quality_scores_from_qf,
#     build_or_load_neg_emb, build_or_load_places_emb,
#     compute_quality_mask_v2, place_scores
# )
# from utils import l2_normalize

# # ---------------------------
# # Global runtime state
# # ---------------------------
# app = FastAPI(title="SwipeSense Recommender API")

# paths: List[str] = []
# meta = []
# ahash_groups: Dict[int, set[int]] = {}
# E: np.ndarray | None = None
# Q_NORM: np.ndarray | None = None
# rec: ImageRecommender | None = None

# # user accum (long-term) persistence
# user_db: Dict[str, dict] = {}

# # ---------------------------
# # Persistence helpers
# # ---------------------------
# def load_user_db() -> Dict[str, dict]:
#     try:
#         return json.loads(USER_DB_PATH.read_text())
#     except Exception:
#         return {}

# def save_user_db(db: Dict[str, dict]):
#     USER_DB_PATH.parent.mkdir(exist_ok=True)
#     USER_DB_PATH.write_text(json.dumps(db))

# def commit_user_accum(user_id: str, e_vec: np.ndarray, feedback: float):
#     """Only positive feedback contributes to long-term accum."""
#     global user_db
#     ent = user_db.get(user_id, {"accum_sum": None, "accum_weight": 0.0})
#     if ent["accum_sum"] is None:
#         ent["accum_sum"] = np.zeros(e_vec.shape[0], np.float32).tolist()
#     if feedback > 0:
#         s = np.array(ent["accum_sum"], dtype=np.float32) + feedback * e_vec
#         w = float(ent["accum_weight"] + feedback)
#         ent["accum_sum"] = s.astype(np.float32).tolist()
#         ent["accum_weight"] = w
#         user_db[user_id] = ent
#         save_user_db(user_db)

# def load_user_profile_into_rec(user_id: str):
#     """Set recommender preference to stored accum (if any)."""
#     global rec, user_db
#     if rec is None:
#         return
#     ent = user_db.get(user_id, {})
#     s = np.array(ent.get("accum_sum", []), dtype=np.float32)
#     w = float(ent.get("accum_weight", 0.0))
#     if s.size == rec.D and w > 0:
#         rec.preference = l2_normalize(s / max(w, 1e-9))
#     else:
#         base = rec.E.mean(axis=0) + np.random.default_rng(42).normal(0.0, 0.05, size=rec.D)
#         rec.preference = l2_normalize(base).astype(np.float32)


# ---------------------------
# # Startup: build the batch pipeline
# # ---------------------------
# @app.on_event("startup")
# def startup_event():
#     global paths, meta, ahash_groups, E, Q_NORM, rec, user_db

#     random.seed(SEED); np.random.seed(SEED)
#     print("starting up")
#     # 1) Discover dataset
#     if USE_GCS:
#         paths = list_gcs_images(GCS_BUCKET, GCS_PREFIX, limit=None)
#         # print("found images:", paths) #human
#     else:
#         paths = list_local_images(LOCAL_DIRS, limit=None)

#     #TODO
#     #REMOVE THIS
#     # random.shuffle(paths)
#     if MAX_IMAGES is not None and len(paths) > MAX_IMAGES:
#         paths = paths[:MAX_IMAGES]
#     assert len(paths) > 0, "No images found. Check bucket/prefix or local dirs."
#     print(f"[startup] Found images: {len(paths)}")

#     # 2) Inspect (quality quick features & duplicates)
#     #each meta is of type ImageMeta, made in utils.py
#     meta = inspect_images(paths)
#     # reorder paths according to items we could inspect (drop failures)
#     idx_map = {m.path: i for i, m in enumerate(meta)}
#     paths = [m.path for m in meta]
#     ahash_groups = build_ahash_groups(meta)
#     print(f"[startup] Inspected: {len(meta)}")

#     # 3) Embeddings: load or compute
#     fingerprint = {"model": MODEL_ID, "use_gcs": USE_GCS, "bucket": GCS_BUCKET if USE_GCS else None, "prefix": GCS_PREFIX if USE_GCS else None, "count": len(paths)}
#     E, loaded = load_or_compute_E(paths, fingerprint)
#     E = l2_normalize(E, axis=1).astype(np.float32)
#     N, D = E.shape
#     print(f"[startup] E shape: {E.shape}")

#     # 4) Quality: quick features + quality scores
#     print("image quality scores start")
#     qf = build_or_load_qf_meta(paths)
#     Q_NORM = quality_scores_from_qf(qf)
#     print("image quality scores end")

#     # 5) Semantic gates
#     print("negative embeddings start")
#     neg_emb = build_or_load_neg_emb()
#     pos_places, neg_places = build_or_load_places_emb()
#     print("negative embeddings end")

#     # 6) Build quality mask
#     mask = compute_quality_mask_v2(
#         E, qf,
#         use_negative_semantics=USE_NEG_SEM_DEFAULT,
#         weird_thresh=WEIRD_THRESH_DEFAULT,
#         min_edge=MIN_EDGE_DEFAULT, min_w=MIN_W_DEFAULT, min_h=MIN_H_DEFAULT,
#         detect_zoom=DETECT_ZOOM_DEFAULT, zoom_center_ratio=ZOOM_CENTER_RATIO_DEFAULT,
#         detect_cutout=DETECT_CUTOUT_DEFAULT, solid_bg_frac=SOLID_BG_FRAC_DEFAULT, alpha_frac=ALPHA_FRAC_DEFAULT,
#         neg_txt_emb=neg_emb
#     )
#     if PLACES_ONLY_DEFAULT:
#         ps = place_scores(E, pos_places, neg_places)
#         mask &= (ps >= PLACE_MIN_DEFAULT)

#     # 7) Recommender
#     rec = ImageRecommender(
#         E, quality_mask=mask, quality_scores=Q_NORM, ahash_groups=ahash_groups,
#         alpha=ALPHA_DEFAULT, eta=ETA_INIT_DEFAULT, warmup_n=5,
#         recent_k=RECENT_K_DEFAULT, recent_weight=RECENT_W_DEFAULT,
#         focus_gamma=FOCUS_GAMMA_DEFAULT, diversity_last_k=DIVERSITY_LAST_K_DEFAULT, diversity_min_cos=DIVERSITY_MIN_COS_DEFAULT,
#         hide_exact_dupes=HIDE_EXACT_DUPES_DEFAULT
#     )
#     # Decay setup
#     rec.eta0 = ETA_INIT_DEFAULT; rec.eta_min = ETA_MIN_DEFAULT; rec.eta_decay_span = ETA_DECAY_SPAN_DEFAULT; rec.use_decay = USE_DECAY_DEFAULT

#     user_db = load_user_db()
#     print(f"[startup] Backend ready: N={rec.N} D={rec.D}")

# # ---------------------------
# # API models
# # ---------------------------
# class NextRequest(BaseModel):
#     user_id: str = "default"

# class Feedback(BaseModel):
#     user_id: str = "default"
#     idx: int
#     action: str  # "like" or "swipe"
#     dwell: float | None = None

# # ---------------------------
# # Helpers
# # ---------------------------
# def compute_feedback_value(action: str, dwell_seconds: float | None) -> float:
#     from config import LIKE_BASE, SWIPE_FAST, DWELL_THRESHOLD, DWELL_BONUS
#     base = LIKE_BASE if action == "like" else SWIPE_FAST
#     # dwell bonus only for swipes (NOT likes)
#     if action != "like" and dwell_seconds is not None and dwell_seconds >= DWELL_THRESHOLD:
#         base += DWELL_BONUS
#     return float(base)

# # ---------------------------
# # Endpoints
# # ---------------------------
# @app.post("/next")
# def get_next(req: NextRequest):
#     global rec, paths
#     assert rec is not None
#     # Load user's accum profile into the recommender
#     load_user_profile_into_rec(req.user_id)

#     idx, score = rec.recommend_next_smart(
#       pool_k=POOL_K_DEFAULT,
#       lambda_div=LAMBDA_DIV_DEFAULT,
#       quality_boost=QUALITY_BOOST_DEFAULT,
#       near_dupe_thr=NEAR_DUPE_THR_DEFAULT
#     )
#     if idx is None:
#         return {"status": "end"}
#     return {"status": "ok", "idx": int(idx), "path": paths[int(idx)], "sim_score": float(score) if score is not None else None}

# @app.post("/feedback")
# def send_feedback(fb: Feedback):
#     global rec, paths, E
#     assert rec is not None and E is not None
#     idx = int(fb.idx)
#     # commit accum (if positive only)
#     val = compute_feedback_value(fb.action, fb.dwell)
#     commit_user_accum(fb.user_id, rec.E[idx], val)

#     # update online preference
#     rec._update(idx, float(val))
#     return {"status": "ok", "eta_now": float(rec.eta)}

# @app.get("/image/{idx}")
# def get_image(idx: int):
#     """Serve the image bytes (JPEG)."""
#     global paths
#     path = paths[int(idx)]
#     im = load_image_any(path).convert("RGB")
#     buf = BytesIO()
#     im.save(buf, format="JPEG", quality=90)
#     return Response(content=buf.getvalue(), media_type="image/jpeg")

# @app.get("/stats")
# def stats():
#     global rec
#     return {
#         "N": int(rec.N) if rec else 0,
#         "D": int(rec.D) if rec else 0,
#         "seen": int(len(rec.seen)) if rec else 0,
#     }

# print('end of file app.py')
# print("running startup method manually: startup_event()")

# startup_event()
