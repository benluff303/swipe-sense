
import argparse, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- helpers ----------
POSSIBLE_IMG_COLS = ["image_path","path","image","filepath","file","img","img_path"]
POSSIBLE_SCORE_COLS = ["score","similarity","cosine","cos","sim"]

def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".csv",".tsv"]:
        sep = "," if ext==".csv" else "\t"
        return pd.read_csv(path, sep=sep)
    elif ext in [".parquet",".pq"]:
        return pd.read_parquet(path)
    elif ext in [".json",".jsonl"]:
        return pd.read_json(path, lines=True)
    else:
        raise ValueError(f"Unsupported file: {path}")

def pick_col(df: pd.DataFrame, candidates) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return cols[cand]
    # fallback: try partial
    for c in df.columns:
        cl = c.lower()
        if any(cand in cl for cand in candidates):
            return c
    raise ValueError(f"Could not find a matching column among {candidates} in {list(df.columns)}")

def normalize_models(x: str) -> str:
    if x is None: return None
    s = x.lower().replace("vit","").replace(" ", "").replace("-", "").replace("/", "")
    if "b32" in s: return "B32"
    if "l14" in s: return "L14"
    return x

def reduce_per_image(df: pd.DataFrame, img_col: str, score_col: str) -> pd.DataFrame:
    # keep the max score per image (robust if file has many rows per image)
    return df[[img_col, score_col]].groupby(img_col, as_index=False)[score_col].max()

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Compare ViT-B/32 vs ViT-L/14 scores and plot hist+scatter.")
    ap.add_argument("--b32", type=Path, help="File with B/32 scores (csv/parquet/jsonl) OR a unified file with a 'model' column")
    ap.add_argument("--l14", type=Path, help="File with L/14 scores if using two separate files")
    ap.add_argument("--out", type=Path, default=Path("out"), help="Output directory")
    ap.add_argument("--img-col", type=str, default=None, help="Image column name (auto-detect if omitted)")
    ap.add_argument("--score-col", type=str, default=None, help="Score column name (auto-detect if omitted)")
    ap.add_argument("--threshold", type=float, default=None, help="Optional rejection threshold on score")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Load
    df_a = read_any(args.b32)

    # Auto-detect columns
    img_col = args.img_col or pick_col(df_a, POSSIBLE_IMG_COLS)
    score_col = args.score_col or pick_col(df_a, POSSIBLE_SCORE_COLS)

    if args.l14 is None:
        # Single file with a 'model' column — pivot to B32/L14
        if "model" not in df_a.columns:
            raise ValueError("Single-file mode requires a 'model' column. Otherwise pass --l14 second file.")
        df_a = df_a.copy()
        df_a["model_norm"] = df_a["model"].map(normalize_models)
        piv = (df_a
               .groupby([img_col,"model_norm"], as_index=False)[score_col].max()
               .pivot(index=img_col, columns="model_norm", values=score_col)
               .reset_index())
        if "B32" not in piv or "L14" not in piv:
            raise ValueError("Could not find both B32 and L14 after pivot. Check 'model' values.")
        merged = piv.rename(columns={"B32":"score_B32","L14":"score_L14"})
    else:
        # Two files: B32 + L14
        df_b = reduce_per_image(df_a, img_col, score_col).rename(columns={score_col: "score_B32"})
        df_l = read_any(args.l14)
        img_col_l = args.img_col or pick_col(df_l, POSSIBLE_IMG_COLS)
        score_col_l = args.score_col or pick_col(df_l, POSSIBLE_SCORE_COLS)
        df_l = reduce_per_image(df_l, img_col_l, score_col_l).rename(columns={score_col_l: "score_L14", img_col_l: img_col})
        merged = pd.merge(df_b, df_l, on=img_col, how="inner")

    # Basic stats
    summary = {
        "count": len(merged),
        "B32_mean": float(np.nanmean(merged["score_B32"])),
        "L14_mean": float(np.nanmean(merged["score_L14"])),
        "B32_median": float(np.nanmedian(merged["score_B32"])),
        "L14_median": float(np.nanmedian(merged["score_L14"])),
    }
    pd.Series(summary).to_json(args.out / "summary.json", indent=2)

    # Histograms
    plt.figure(figsize=(10,4))
    plt.hist(merged["score_B32"].dropna(), bins=40, alpha=0.6, label="B/32")
    plt.hist(merged["score_L14"].dropna(), bins=40, alpha=0.6, label="L/14")
    plt.xlabel("Cosine score"); plt.ylabel("Count"); plt.title("Score Distribution"); plt.legend()
    plt.tight_layout(); plt.savefig(args.out / "score_hist.png", dpi=180); plt.close()

    # Scatter
    lims = [
        np.nanmin([merged["score_B32"].min(), merged["score_L14"].min()]),
        np.nanmax([merged["score_B32"].max(), merged["score_L14"].max()])
    ]
    plt.figure(figsize=(5,5))
    plt.scatter(merged["score_B32"], merged["score_L14"], s=8)
    plt.plot(lims, lims)
    plt.xlabel("B/32 score"); plt.ylabel("L/14 score"); plt.title("Per-image score comparison")
    plt.tight_layout(); plt.savefig(args.out / "scatter.png", dpi=180); plt.close()

    # Gains
    merged["delta"] = merged["score_L14"] - merged["score_B32"]
    gains = merged.sort_values("delta", ascending=False)
    gains[ [img_col, "score_B32", "score_L14", "delta"] ].to_csv(args.out / "top_gains.csv", index=False)

    # Threshold (optional)
    if args.threshold is not None:
        rej = merged[(merged["score_L14"] < args.threshold) & (merged["score_B32"] < args.threshold)]
        rej.to_csv(args.out / "rejected.csv", index=False)

    # Console preview
    print(f"✅ wrote: {args.out}/score_hist.png, {args.out}/scatter.png, {args.out}/top_gains.csv")
    print("Top 20 L/14 gains:")
    print(gains[[img_col, "score_B32","score_L14","delta"]].head(20).to_string(index=False))

if __name__ == "__main__":
    main()
