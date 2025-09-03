import argparse
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

COLUMNS = ["Vibes", "Activities", "Destination Features"]

def load_text_cells(df, columns):
    """
    Collect (column_name, row_idx, text) for each non-empty text cell, as-is.
    """
    items = []
    for col in columns:
        if col not in df.columns:
            continue
        col_series = df[col].astype(str)
        for i, raw in col_series.items():
            text = raw.strip()
            if text and text.lower() != "nan":
                items.append((col, int(i), text))
    return items

def main():
    ap = argparse.ArgumentParser(description="Embed CSV keyword phrases with CLIP-ViT-B-32 and save NumPy arrays + index.")
    ap.add_argument("--csv", required=True, help="Path to input CSV (expects columns: 'Vibes','Activities','Destination Features').")
    ap.add_argument("--model", default="clip-ViT-B-32", help="SentenceTransformers model name (default: clip-ViT-B-32).")
    ap.add_argument("--out_prefix", default="keywords", help="Output file prefix (default: 'keywords').")
    ap.add_argument("--no_per_column", action="store_true", help="If set, skip saving per-column .npy files.")
    args = ap.parse_args()

    # 1) Load CSV
    df = pd.read_csv(args.csv)

    # 2) Collect texts as-is, keeping (column, row_idx, text)
    items = load_text_cells(df, COLUMNS)
    if not items:
        raise ValueError("No non-empty cells found in the expected columns. Check your CSV and column names.")

    # 3) Embed with SentenceTransformers CLIP text model
    model = SentenceTransformer(args.model)
    texts = [t for (_, _, t) in items]

    # Convert to numpy, normalize, float32
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        dtype=np.float32
    )  # shape: (N, 512)

    # 4) Save a single stacked array + index
    np.save(f"{args.out_prefix}_embeddings.npy", embeddings)
    index_records = [
        {"column": col, "row_idx": row_idx, "text": text, "emb_idx": i}
        for i, (col, row_idx, text) in enumerate(items)
    ]
    with open(f"{args.out_prefix}_index.json", "w", encoding="utf-8") as f:
        json.dump(index_records, f, ensure_ascii=False, indent=2)

    # 5) (Optional) Save per-column arrays + labels
    if not args.no_per_column:
        for col in COLUMNS:
            col_indices = [i for i, (c, _, _) in enumerate(items) if c == col]
            if not col_indices:
                continue
            col_emb = embeddings[col_indices]
            col_texts = [items[i][2] for i in col_indices]
            np.save(f"{args.out_prefix}_{col.lower().replace(' ', '_')}.npy", col_emb)
            with open(f"{args.out_prefix}_{col.lower().replace(' ', '_')}_labels.txt", "w", encoding="utf-8") as f:
                for t in col_texts:
                    f.write(t + "\n")

    print("Done")
    print(" - Stacked embeddings:", f"{args.out_prefix}_embeddings.npy", embeddings.shape)
    print(" - Index:", f"{args.out_prefix}_index.json")
    if not args.no_per_column:
        print(" - Per-column files saved where data was present.")

if __name__ == "__main__":
    main()
