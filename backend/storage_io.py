# storage_io.py
# GCS & local image I/O (lists, read, optional uploads, small cache)

from __future__ import annotations
from pathlib import Path
from io import BytesIO
from typing import List, Dict

from PIL import Image
import numpy as np

from config import USE_GCS, GCS_BUCKET, ARTIFACTS_PREFIX, MIRROR_TO_GCS, IMG_EXTS, LOCAL_DIRS
# GCS imports are optional; import lazily
from google.cloud import storage
import gcsfs

import os

GCP_CREDS = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
GCP_PROJECT = os.environ.get("GCP_PROJECT")

_gcs_client = None
_gcs_fs = None

def _ensure_gcs():
    """Initialize GCS client/filesystem lazily."""
    global _gcs_client, _gcs_fs
    if _gcs_client is None:
        _gcs_client = storage.Client(project=GCP_PROJECT)
    if _gcs_fs is None:
        _gcs_fs = gcsfs.GCSFileSystem(token=GCP_CREDS)

def upload_to_gcs(local_path: str, dst_name: str | None = None):
    """Mirror a local artifact to GCS if enabled."""
    if not (USE_GCS and MIRROR_TO_GCS):
        return
    _ensure_gcs()
    dst_name = dst_name or (ARTIFACTS_PREFIX.rstrip('/') + '/' + Path(local_path).name)
    bucket = _gcs_client.bucket(GCS_BUCKET)
    blob = bucket.blob(dst_name)
    blob.upload_from_filename(local_path)

def list_gcs_images(bucket: str, prefix: str = "", limit: int | None = None) -> List[str]:
    """List image objects in a GCS bucket/prefix."""
    _ensure_gcs()
    blobs = _gcs_client.list_blobs(bucket, prefix=prefix or None)
    #human
    print("found blobs", blobs)
    # for blob in blobs:
    #     print(blob)
    # names = [blob.name for blob in blobs]
    # print("found n blobs", len(names))

    # Print them all at once
    # print("\n".join(names))
    #end human
    out = []
    for b in blobs:
        name = b.name
        if name.lower().endswith(IMG_EXTS):
            out.append(f"gs://{bucket}/{name}")
            if limit is not None and len(out) >= limit:
                break
    return out

def list_local_images(dirs: list[Path], limit: int | None = None) -> List[str]:
    files: list[str] = []
    for d in dirs:
        if not Path(d).exists():
            continue
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
            files += [str(p) for p in Path(d).rglob(ext)]
    return files[:limit] if limit is not None else files

def load_image_any(path: str) -> Image.Image:
    """Load an image from GCS (gs://) or local path, always RGB."""
    if path.startswith("gs://"):
        _ensure_gcs()
        no_scheme = path[5:]
        with _gcs_fs.open(no_scheme, 'rb') as f:
            data = f.read()
        return Image.open(BytesIO(data)).convert("RGB")
    return Image.open(path).convert("RGB")

# Small cache for display/serving
_img_cache: Dict[str, Image.Image] = {}

def load_image_cached(path: str, max_side: int = 720) -> Image.Image:
    """Cached image load with quick downscale for display."""
    im = _img_cache.get(path)
    if im is not None:
        return im
    im = load_image_any(path)
    w, h = im.size
    s = min(max_side / w, max_side / h, 1.0)
    if s < 1.0:
        im = im.resize((int(w * s), int(h * s)))
    _img_cache[path] = im
    return im
