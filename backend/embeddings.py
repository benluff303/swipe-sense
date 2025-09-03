# embeddings.py
# CLIP model init + image/text embeddings + caching

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel

from config import MODEL_ID, EMB_DIR, MAX_IMAGES
from utils import l2_normalize
from storage_io import load_image_any

# -------- Model init (module-global singletons) --------
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = CLIPProcessor.from_pretrained(MODEL_ID, use_fast=False)
model = CLIPModel.from_pretrained(MODEL_ID).to(device)
model.eval()

# -------- Image embeddings --------
def get_image_features_batch(batch_imgs: List) -> np.ndarray:
    inputs = processor(images=batch_imgs, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_image_features(**inputs).float()
    emb = emb.cpu().numpy()
    return l2_normalize(emb, axis=1).astype(np.float32)

def encode_all(paths: List[str], batch_size: int = 32) -> np.ndarray:
    """Compute CLIP embeddings for all image paths. Returns (N, D) float32, L2-normalized."""
    E_parts: list[np.ndarray] = []
    batch: list = []
    for i, p in enumerate(paths, 1):
        try:
            batch.append(load_image_any(p))
        except Exception:
            # fallback black image if unreadable
            import PIL.Image as PILI
            batch.append(PILI.new("RGB", (224, 224), (0, 0, 0)))
        if len(batch) == batch_size or i == len(paths):
            E_parts.append(get_image_features_batch(batch))
            batch.clear()
        if i % 200 == 0:
            print(f"[embeddings] encoded {i}/{len(paths)}")
    E = np.vstack(E_parts).astype(np.float32)
    return l2_normalize(E, axis=1)

# -------- Text embeddings --------
def embed_text_templates(labels: List[str], templates: List[str]) -> np.ndarray:
    """
    For each label, average CLIP text features of multiple template prompts.
    Returns (len(labels), D) float32, L2-normalized.
    """
    rows = []
    with torch.no_grad():
        for lab in labels:
            prompts = [t.format(lab) for t in templates]
            tin = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            tfeat = model.get_text_features(**tin).float().cpu().numpy().mean(axis=0)
            rows.append(tfeat)
    return l2_normalize(np.stack(rows, axis=0), axis=1).astype(np.float32)

# -------- Cache helpers --------
def save_numpy(path: Path, arr: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)

def load_numpy(path: Path) -> np.ndarray:
    return np.load(path)

def save_json(path: Path, obj):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False))

def load_json(path: Path):
    import json
    return json.loads(path.read_text())

def save_fingerprint(meta_path: Path, fingerprint: dict):
    save_json(meta_path, fingerprint)

def load_fingerprint(meta_path: Path) -> dict | None:
    try:
        return load_json(meta_path)
    except Exception:
        return None

def load_or_compute_E(paths: List[str], fingerprint: dict) -> Tuple[np.ndarray, bool]:
    """
    Load embeddings if cache matches fingerprint; otherwise compute and save.
    Returns (E, loaded_from_cache).
    """
    E_PATH = EMB_DIR / "embed_array_20k.npy"
    P_PATH = EMB_DIR / "paths20k.npy"
    M_PATH = EMB_DIR / "meta20k.json" #adjust the count

    # E_PATH = EMB_DIR / "E_fp32.npy"
    # P_PATH = EMB_DIR / "paths.npy"
    # M_PATH = EMB_DIR / "meta.json"

    print(E_PATH)
    print(P_PATH)
    print(M_PATH)

    #louis hunch
    E = load_numpy(E_PATH)
    return E, True

    # Try load
    if E_PATH.exists() and P_PATH.exists() and M_PATH.exists():
        print("step 1. all cached embed paths exist")
        meta_f = load_fingerprint(M_PATH) or {}
        print("meta_f, fingerprint:", meta_f)
        if meta_f.get("model") == fingerprint.get("model") and meta_f.get("use_gcs") == fingerprint.get("use_gcs"):
            E = load_numpy(E_PATH)
            p = np.load(P_PATH, allow_pickle=True).tolist()[:MAX_IMAGES] #human MAX_IMAGES
            print("step 2, trying loading paths and embeddings from cache")
            print(len(paths))
            print(len(p))
            print(paths)
            print(p)
            if len(p) == len(paths) and p == paths:
                print("[embeddings] Loaded embeddings from cache. (not scratch) Means that the paths match")
                return E.astype(np.float32), True
            else:
                print("paths mismatch, need to recompute")
                print(p[:5], paths[:5])

    # Compute fresh
    # print("[embeddings] Computing embeddings from scratch…")
    # E = encode_all(paths, batch_size=32).astype(np.float32)
    # save_numpy(E_PATH, E)
    # np.save(P_PATH, np.array(paths, dtype=object))
    # save_fingerprint(M_PATH, fingerprint)
    # print("[embeddings] Saved embeddings to", EMB_DIR)
    print("[embeddings] WRONG!! Computing embeddings from scratch…")

    return None
    # return E, False
