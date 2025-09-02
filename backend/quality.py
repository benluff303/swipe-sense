# quality.py
# Quality features, semantic gates (negative prompts, places), tags/labels,
# duplicates inspection, and mask building.

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import ImageMeta, l2_normalize, minmax01
from storage_io import load_image_any, upload_to_gcs
from config import EMB_DIR
from embeddings import embed_text_templates

# -----------------------------
# Lightweight image descriptors
# -----------------------------
def _edge_energy(im: Image.Image) -> float:
    g = im.convert("L").resize((128, 128), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    dx = np.abs(np.diff(arr, axis=1)).mean()
    dy = np.abs(np.diff(arr, axis=0)).mean()
    return float(dx + dy)

def _ahash(im: Image.Image, size=8) -> int:
    g = im.convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(g, dtype=np.float32)
    thr = arr.mean()
    bits = (arr > thr).astype(np.uint8).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return int(h)

def inspect_images(paths: List[str]) -> List[ImageMeta]:
    print("inspecting images", len(paths))
    out: list[ImageMeta] = []
    for p in paths:
        try:
            im = load_image_any(p)
            w, h = im.size
            out.append(ImageMeta(p, w, h, _edge_energy(im), _ahash(im)))
        except Exception:
            continue
    return out

def build_ahash_groups(meta: List[ImageMeta]) -> Dict[int, set[int]]:
    """Group near-exact duplicates by a-hash; return index->set(group) map."""
    print("building ahash groups", len(meta))
    from collections import defaultdict
    hash2idxs: Dict[int, List[int]] = defaultdict(list)
    for i, m in enumerate(meta):
        hash2idxs[m.ahash].append(i)
    dupe_groups = [v for v in hash2idxs.values() if len(v) > 1]
    groups = {i: set() for i in range(len(meta))}
    for group in dupe_groups:
        s = set(group)
        for i in group:
            groups[i] = s
    return groups

# -----------------------------
# Quick features for quality v2
# -----------------------------
def _quick_features_for_path(p: str, thumb=128) -> Tuple[int,int,float,float,float,bool]:
    try:
        im = load_image_any(p)
        has_alpha = (im.mode in ("LA", "RGBA", "PA"))
        alpha_cut = False
        if has_alpha:
            im = im.convert("RGBA")
            a = np.asarray(im)[:, :, 3].astype(np.float32) / 255.0
            alpha_cut = (a < 0.05).mean() > 0.10
            im = im.convert("RGB")
        else:
            im = im.convert("RGB")
        w, h = im.size
        im2 = im.copy(); im2.thumbnail((thumb, thumb))
        arr = np.asarray(im2).astype(np.float32) / 255.0
        gray = (0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2])
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, [0]]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[[0], :]))
        edge = gx + gy
        edge_energy = float(edge.mean() * 100.0)
        H, W = gray.shape; m1, m2 = int(H * 0.25), int(H * 0.75); n1, n2 = int(W * 0.25), int(W * 0.75)
        center = edge[m1:m2, n1:n2].mean() + 1e-8
        border = np.concatenate([edge[:m1, :], edge[m2:, :], edge[m1:m2, :n1], edge[m1:m2, n2:]], axis=None).mean() + 1e-8
        center_edge_ratio = float(center / border)
        std = arr.std(axis=2); white = (arr.mean(axis=2) > 0.92) & (std < 0.03)
        white_bg_frac = float(white.mean())
        return w, h, edge_energy, center_edge_ratio, white_bg_frac, alpha_cut
    except Exception:
        return 0, 0, 0.0, 1.0, 0.0, False

def build_or_load_qf_meta(paths: List[str]) -> dict:
    """Build or load quick features cache."""
    QF_META = EMB_DIR / "quality_meta_v2.npz"
    if QF_META.exists():
        dat = np.load(QF_META, allow_pickle=False)
        if int(dat["N"]) == len(paths):
            return {k: dat[k] for k in dat.files}
    feats = [_quick_features_for_path(p) for p in paths]
    w, h, ee, cer, wbg, ac = map(np.array, zip(*feats))
    np.savez_compressed(QF_META, N=len(paths), w=w, h=h, edge=ee, center_ratio=cer, whitebg=wbg, alpha_cut=ac)
    try:
        upload_to_gcs(str(QF_META))
    except Exception:
        pass
    return {"N": len(paths), "w": w, "h": h, "edge": ee, "center_ratio": cer, "whitebg": wbg, "alpha_cut": ac}

def quality_scores_from_qf(qf_meta: dict) -> np.ndarray:
    """Combine edge + megapixels into a normalized quality score (0..1)."""
    mp = (qf_meta["w"] * qf_meta["h"]).astype(np.float32) / 1e6
    q_edge = minmax01(qf_meta["edge"])
    q_mp = minmax01(mp)
    return 0.5 * q_edge + 0.5 * q_mp

# -----------------------------
# Negative semantics (bad stuff)
# -----------------------------
NEG_PROMPTS = [
    "isolated object on white background","isolated animal on white background",
    "product cutout on white background","plain solid background",
    "macro close-up crop","extreme zoom crop","texture pattern close-up",
    "blurry out of focus photo","logo or icon or clipart",
    "screenshot of user interface","meme with big text","watermark text overlay"
]
NEG_TEMPLATES = ["a photo of {}", "high quality {}", "{}"]
NEG_EMB_PATH = EMB_DIR / "neg_txt_emb.npy"
NEG_LABS_PATH = EMB_DIR / "neg_txt_labels.json"

def build_or_load_neg_emb() -> np.ndarray:
    if NEG_EMB_PATH.exists() and NEG_LABS_PATH.exists():
        labs = __load_json(NEG_LABS_PATH)
        if labs == NEG_PROMPTS:
            return np.load(NEG_EMB_PATH)
    neg = embed_text_templates(NEG_PROMPTS, NEG_TEMPLATES)
    np.save(NEG_EMB_PATH, neg)
    __save_json(NEG_LABS_PATH, NEG_PROMPTS)
    try:
        upload_to_gcs(str(NEG_EMB_PATH)); upload_to_gcs(str(NEG_LABS_PATH))
    except Exception:
        pass
    return neg

# -----------------------------
# Places gate (pos/neg)
# -----------------------------
PL_POS = [
    "wide landscape view","mountain landscape","desert landscape","beach coastline",
    "forest landscape","waterfall in nature","aerial cityscape","city skyline","old town street",
    "architecture exterior","modern building exterior","bridge over river",
    "interior of hotel lobby","modern interior of living room","luxury resort exterior",
    "street scene with buildings","harbor with boats","park landscape","temple exterior","castle exterior"
]
PL_NEG = [
    "portrait of a person","selfie close-up face","close-up of object","macro product",
    "bokeh portrait","face with blurry background","food close-up","hand holding phone",
    "animal close-up face","extreme close-up crop"
]
PL_TEMPLATES = ["{}", "a photo of {}", "high quality {}"]
PL_EMB_PATH = EMB_DIR / "places_pos_neg.npz"

def build_or_load_places_emb() -> Tuple[np.ndarray, np.ndarray]:
    if PL_EMB_PATH.exists():
        dat = np.load(PL_EMB_PATH)
        return dat["pos"], dat["neg"]
    pos = embed_text_templates(PL_POS, PL_TEMPLATES)
    neg = embed_text_templates(PL_NEG, PL_TEMPLATES)
    np.savez_compressed(PL_EMB_PATH, pos=pos, neg=neg)
    try:
        upload_to_gcs(str(PL_EMB_PATH))
    except Exception:
        pass
    return pos, neg

def place_scores(E_unit: np.ndarray, pos_emb: np.ndarray, neg_emb: np.ndarray) -> np.ndarray:
    s_pos = E_unit @ pos_emb.T
    s_neg = E_unit @ neg_emb.T
    return s_pos.max(axis=1) - s_neg.max(axis=1)

# -----------------------------
# Tags/labels (100 vocab) — for explanations/orientation
# -----------------------------
label_vocab = [
    "supercars","sports cars","classic cars","luxury cars","off-road trucks",
    "motorcycles","motorsport","yacht","private jet","racing car",
    "luxury watches","mechanical watch","chronograph","jewelry","diamonds",
    "rings","necklaces","handbags","sneakers","menswear","womenswear","streetwear",
    "portraits","studio portrait","headshot","close-up face","street photography","fashion photography",
    "architecture","modern interiors","luxury interiors","minimal interiors",
    "skyscrapers","luxury hotel","resort","spa",
    "landscapes","mountains","beaches","desert","forests","waterfalls",
    "sunsets","night sky","milky way","aurora","aerial",
    "wildlife","dogs","cats","birds","horses","lions","tigers","elephants","wolves","foxes",
    "cityscapes","old town","alleyway","night city","street market",
    "train station","airport lounge","harbor","bridge","tower",
    "food","desserts","coffee","latte art","tea","sushi","pizza","burgers",
    "steak","pasta","salad","breakfast","fine dining",
    "football","basketball","tennis","boxing","golf","gym fitness",
    "cycling","running","swimming","skiing",
    "technology","gadgets","smartphones","laptops","gaming setup","workstation","headphones","camera gear",
    "abstract art","minimalism"
]
assert len(label_vocab) == 100
TEMPLATES = ["a photo of {}", "a high quality photo of {}", "aesthetic {}", "premium {}", "close-up of {}", "{}"]
TXT_EMB_PATH = EMB_DIR / "txt_emb_tags100.npy"
TXT_LABS_PATH = EMB_DIR / "txt_labels_tags100.json"

def build_or_load_label_emb():
    if TXT_EMB_PATH.exists() and TXT_LABS_PATH.exists():
        txt_emb = np.load(TXT_EMB_PATH)
        labs = __load_json(TXT_LABS_PATH)
        if labs == label_vocab and txt_emb.shape[0] == len(label_vocab):
            return txt_emb
    txt_emb = embed_text_templates(label_vocab, TEMPLATES)
    np.save(TXT_EMB_PATH, txt_emb)
    __save_json(TXT_LABS_PATH, label_vocab)
    try:
        upload_to_gcs(str(TXT_EMB_PATH)); upload_to_gcs(str(TXT_LABS_PATH))
    except Exception:
        pass
    return txt_emb

def image_labels_from_idx(E_unit: np.ndarray, txt_emb: np.ndarray, idx: int, top_k: int = 6):
    e = E_unit[idx]; scores = (txt_emb @ e); J = np.argsort(-scores)[:top_k]
    return [(label_vocab[j], float(scores[j])) for j in J]

def orientation_from_pref(pref: np.ndarray, txt_emb: np.ndarray, top_k: int = 8):
    scores = (txt_emb @ pref); J = np.argsort(-scores)[:top_k]
    return [(label_vocab[j], float(scores[j])) for j in J]

# -----------------------------
# Quality mask v2
# -----------------------------
def compute_quality_mask_v2(
    E_unit: np.ndarray,
    qf_meta: dict,
    use_negative_semantics: bool = True,
    weird_thresh: float = 0.45,
    min_edge: float = 1.0, min_w: int = 256, min_h: int = 256,
    detect_zoom: bool = True, zoom_center_ratio: float = 2.2,
    detect_cutout: bool = True, solid_bg_frac: float = 0.60, alpha_frac: float = 0.10,
    neg_txt_emb: np.ndarray | None = None
) -> np.ndarray:
    w = qf_meta["w"]; h = qf_meta["h"]; edge = qf_meta["edge"]
    cer = qf_meta["center_ratio"]; wbg = qf_meta["whitebg"]; ac = qf_meta["alpha_cut"].astype(np.float32)

    ok = (w >= min_w) & (h >= min_h) & (edge >= min_edge)
    if detect_zoom: ok &= (cer <= zoom_center_ratio)
    if detect_cutout: ok &= ~((wbg >= solid_bg_frac) | (ac >= alpha_frac))
    if use_negative_semantics and neg_txt_emb is not None:
        sims = E_unit @ neg_txt_emb.T
        worst = sims.max(axis=1)
        ok &= (worst < weird_thresh)
    return ok.astype(bool)

# -----------------------------
# Small JSON helpers (avoid circulars)
# -----------------------------
def __save_json(path: Path, obj):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))

def __load_json(path: Path):
    import json
    return json.loads(path.read_text())
