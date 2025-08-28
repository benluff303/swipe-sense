from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
import open_clip
import matplotlib.pyplot as plt

# -------- Paths --------
IMG_DIR = Path("data/images_cache")
IMG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = Path("gallery_paths.csv")

# -------- Helpers --------
def list_images(img_dir: Path = IMG_DIR):
    exts = {".jpg",".jpeg",".png",".webp",".bmp"}
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in exts])

def load_model(model_id="ViT-B-32", pretrained="openai", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    m, preprocess, _ = open_clip.create_model_and_transforms(model_id, pretrained=pretrained)
    tok = open_clip.get_tokenizer(model_id)
    return m.to(device).eval(), preprocess, tok, device

@torch.no_grad()
def encode_images(paths, model, preprocess, device, bs=64):
    feats=[]
    for i in range(0, len(paths), bs):
        batch=[preprocess(Image.open(p).convert("RGB")) for p in paths[i:i+bs]]
        x=torch.stack(batch).to(device)
        z=model.encode_image(x)
        z=F.normalize(z, dim=-1).cpu().float()
        feats.append(z)
    return torch.cat(feats, 0).numpy()

def build_or_load_embeds(model_id, pretrained, paths, out_path, bs=64, device=None):
    out=Path(out_path)
    if out.exists():
        return np.load(out)
    m, pre, _, device = load_model(model_id, pretrained, device)
    emb = encode_images(paths, m, pre, device, bs=bs)
    np.save(out, emb)
    return emb

@torch.no_grad()
def encode_text(query, model, tokenizer, device):
    tok = tokenizer([query]).to(device)
    z = model.encode_text(tok)
    return F.normalize(z, dim=-1).cpu().numpy()[0]

def show_grid_local(df, title=None, base: Path = IMG_DIR, max_cols=5):
    n=len(df); cols=min(n, max_cols); rows=(n+cols-1)//cols
    plt.figure(figsize=(3.2*cols, 3.2*rows))
    for i, r in enumerate(df.itertuples(), 1):
        p = base / getattr(r, "image_path")
        img = Image.open(p).convert("RGB")
        ax = plt.subplot(rows, cols, i)
        ax.imshow(img); ax.axis("off")
        if hasattr(r, "score"):
            ax.set_title(f"{getattr(r,'score'):.2f}", fontsize=9)
    if title: plt.suptitle(title, y=0.99)
    plt.tight_layout(); plt.show()
