# config.py
# Central configuration & defaults

from pathlib import Path

# -----------------------
# Data & storage settings
# -----------------------
USE_GCS: bool = True                      # If False, read from LOCAL_DIRS

GCS_BUCKET = "swipe-bucket"          # <<< put your bucket name here
GCS_PREFIX = ""                      # optional subfolder in the bucket

# Optional local dirs to search for images when USE_GCS=False
LOCAL_DIRS = [Path.cwd() / "images", Path.cwd() / "data", Path.cwd()]

# Limit dataset size for speed during dev
MAX_IMAGES: int | None = 100
MAX_IMAGES: int | None = 20888

# Reproducibility
SEED: int = 123

# Artifacts & persistence paths
EMB_DIR = Path("./emb_cache_20k"); EMB_DIR.mkdir(exist_ok=True)
STATE_DIR = Path("./users"); STATE_DIR.mkdir(exist_ok=True)

# User DB file (accumulated long-term profile)
USER_DB_PATH = STATE_DIR / "users.json"

# Whether to mirror local artifacts to GCS (if creds available)
MIRROR_TO_GCS: bool = True
ARTIFACTS_PREFIX: str = f"artifacts_{str(MAX_IMAGES)}/"      # subfolder in bucket

# -----------------------
# Model settings
# -----------------------
MODEL_ID = "openai/clip-vit-base-patch32"  # 512-dim CLIP

# -----------------------
# Feedback mapping (fixed)
# -----------------------
LIKE_BASE = 0.30
SWIPE_FAST = -0.10
DWELL_THRESHOLD = 3.0
DWELL_BONUS = 0.10  # used for swipes only (NOT likes)

# -----------------------
# Default quality controls
# -----------------------
MIN_EDGE_DEFAULT = 2.0
MIN_W_DEFAULT = 256
MIN_H_DEFAULT = 256
DETECT_ZOOM_DEFAULT = True
ZOOM_CENTER_RATIO_DEFAULT = 1.8
DETECT_CUTOUT_DEFAULT = True
SOLID_BG_FRAC_DEFAULT = 0.60
ALPHA_FRAC_DEFAULT = 0.10
USE_NEG_SEM_DEFAULT = True
WEIRD_THRESH_DEFAULT = 0.32

# Places-only gate
PLACES_ONLY_DEFAULT = True
PLACE_MIN_DEFAULT = 0.05

# -----------------------
# Recommender params
# -----------------------
ALPHA_DEFAULT = 0.85
ETA_INIT_DEFAULT = 1.6
ETA_MIN_DEFAULT = 0.10
ETA_DECAY_SPAN_DEFAULT = 50
USE_DECAY_DEFAULT = True

RECENT_K_DEFAULT = 50
RECENT_W_DEFAULT = 0.80
FOCUS_GAMMA_DEFAULT = 1.30
DIVERSITY_LAST_K_DEFAULT = 20
DIVERSITY_MIN_COS_DEFAULT = 0.92
HIDE_EXACT_DUPES_DEFAULT = True

POOL_K_DEFAULT = 15000
LAMBDA_DIV_DEFAULT = 0.50
NEAR_DUPE_THR_DEFAULT = 0.95
QUALITY_BOOST_DEFAULT = 0.30

# Image file extensions
IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")
