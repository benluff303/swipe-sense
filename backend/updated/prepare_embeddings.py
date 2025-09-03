# prepare_embeddings.py
import os, json, re
import numpy as np
import pandas as pd
from urllib.parse import urlparse

# === 1) عدّل المسارات حسب جهازك (هذه حسب لقطة الشاشة) ===
EMB_CSV   = "/mnt/c/Users/lenovo/Downloads/outputs_classified_bucket_with_embeddings.csv"
LINKS_CSV = "/mnt/c/Users/lenovo/Downloads/photo_lookup_table.csv"

# أين نكتب ملفات الكاش التي سيقرأها backend لاحقاً
OUT_DIR = "emb_cache"
os.makedirs(OUT_DIR, exist_ok=True)

# === 2) دوال مساعدة لتوحيد المعرفات ===
def norm_name(x: str) -> str:
    """حوّل الاسم لبسيط: بدون مسارات/استعلامات/امتدادات وبأحرف صغيرة."""
    if x is None:
        return ""
    x = str(x).strip()
    # خذ آخر جزء فقط لو فيه مسار
    x = os.path.basename(x)
    # شِل أي استعلامات ?size=.. الخ
    x = x.split("?")[0].split("#")[0]
    # خذ بدون الامتداد
    x = re.sub(r"\.(jpg|jpeg|png|webp)$", "", x, flags=re.IGNORECASE)
    return x.lower()

def filename_from_url(u: str) -> str:
    """استخرج اسم الملف من URL ثم طبّق norm_name."""
    if not isinstance(u, str) or not u:
        return ""
    path = urlparse(u).path or u  # احتياط
    return norm_name(path)

def parse_embedding(s) -> np.ndarray:
    """حوّل النص '[0.1, 0.2, ...]' إلى numpy array."""
    if s is None:
        return np.array([], dtype=np.float32)
    s = str(s).strip()
    if s == "" or s == "[]":
        return np.array([], dtype=np.float32)
    # أزل الأقواس لو موجودة
    s = s.strip("[]")
    arr = np.fromstring(s, sep=",", dtype=np.float32)
    return arr

# === 3) اقرأ الملفين ===
df_emb   = pd.read_csv(EMB_CSV)
df_links = pd.read_csv(LINKS_CSV)

print("Embeddings CSV columns:", df_emb.columns.tolist())
print("Links CSV columns:", df_links.columns.tolist())

# توقعنا الأعمدة التالية من لقطاتك:
# df_emb:  image_id, predicted_type, embedding
# df_links: photo_id, photo_url, photo_image_url, ...

# طبّع مثال للتأكد
print("Sample image_id:", df_emb["image_id"].head())
if "photo_image_url" in df_links.columns:
    print("Sample photo_image_url:", df_links["photo_image_url"].head())
if "photo_url" in df_links.columns:
    print("Sample photo_url:", df_links["photo_url"].head())

# === 4) جهّز مفاتيح الدمج ===
# طَبع اسم الملف من image_id (بدون الامتداد)
df_emb["image_key"] = df_emb["image_id"].apply(norm_name)

# استخرج اسم الملف من أي عمود URL متوفر
if "photo_image_url" in df_links.columns:
    df_links["file_from_url"] = df_links["photo_image_url"].apply(filename_from_url)
elif "photo_url" in df_links.columns:
    df_links["file_from_url"] = df_links["photo_url"].apply(filename_from_url)
else:
    raise ValueError("لم أجد عمود photo_image_url ولا photo_url في جدول الروابط.")

# قد يكون هناك صفوف بلا URL؛ نحذفها
df_links = df_links[df_links["file_from_url"] != ""].copy()

# === 5) دمج عبر اسم الملف الموحّد ===
df = pd.merge(
    df_emb,
    df_links,
    left_on="image_key",
    right_on="file_from_url",
    how="inner",
)

print(f"Merged rows: {len(df)}")
if len(df) == 0:
    raise ValueError(
        "الدمج رجّع 0 صفوف. يبدو أن أسماء الملفات لا تتطابق.\n"
        "افحص مثالاً:\n"
        f"image_id -> image_key مثال:\n{df_emb[['image_id','image_key']].head()}\n"
        f"photo_image_url/photo_url -> file_from_url مثال:\n{df_links[['file_from_url']].head()}"
    )

# === 6) حوّل embeddings إلى مصفوفة ثابتة الأبعاد ===
vecs = [parse_embedding(x) for x in df["embedding"]]
# تخلّص من أي صف فاضي أو غير 512
vecs = [v for v in vecs if v.size > 0]
# تأكد الأبعاد متساوية
dims = [v.size for v in vecs]
D = max(set(dims), key=dims.count)  # الأكثر تكراراً (غالباً 512)
vecs = [v for v in vecs if v.size == D]
if len(vecs) == 0:
    raise ValueError("لا توجد متجهات صالحة بعد التحويل. تحقق من صيغة عمود embedding.")
E = np.vstack(vecs).astype(np.float32)

# ملاحظة: بعد ترشيح vecs، لازم نرُتّب paths بنفس الترتيب
valid_mask = df["embedding"].apply(lambda x: parse_embedding(x).size == D).values
paths_series = df.loc[valid_mask]

# اختر العمود الذي تريد استعماله كمسار للصورة في الخلفية (backend)
# النُصح: استخدم photo_image_url إن كان مباشر للصورة؛ وإلا photo_url
PATH_COL = "photo_image_url" if "photo_image_url" in df.columns else "photo_url"
paths = paths_series[PATH_COL].astype(str).tolist()

# تحقّق أخير
assert E.shape[0] == len(paths), f"عدم تطابق: E={E.shape[0]} vs paths={len(paths)}"

# === 7) خزّن الملفات التي يقرأها backend ===
np.save(os.path.join(OUT_DIR, "E_fp32.npy"), E)
np.save(os.path.join(OUT_DIR, "paths.npy"), np.array(paths, dtype=object))

meta = {
    "model": "openai/clip-vit-base-patch32",
    "use_gcs": True,
    "bucket": "swipe-bucket",
    "prefix": "",
    "count": int(len(paths)),
}
with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f)

print("✅ Done.")
print("E shape:", E.shape)
print("paths:", len(paths))
print("Saved to:", OUT_DIR)
