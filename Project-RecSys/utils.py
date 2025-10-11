from pathlib import Path
from datetime import datetime
import shutil
import json
import pandas as pd
import os, json, hashlib
import numpy as np

# --- Helper methods for saving and loading BERT embeddings ---
def _normalize_text_list(content_list):
    # Normalize to stable strings for hashing
    return [(" ".join(str(s).split())).strip() for s in content_list]

def _fingerprint(content_list, model_name="distilbert-base-uncased"):
    norm = _normalize_text_list(content_list)
    payload = (model_name + "\n" + "\n".join(norm)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]

def _cache_paths(cache_dir, model_name, fp):
    base = os.path.join(cache_dir, model_name.replace("/", "_"))
    os.makedirs(base, exist_ok=True)
    return (
        os.path.join(base, f"{fp}.npz"),   # embeddings + content
        os.path.join(base, f"{fp}.meta"),  # small meta json (optional)
    )

def save_embeddings_cache(embeddings, content, cache_dir="bert_cache", model_name="distilbert-base-uncased"):
    if embeddings is None:
        return None
    fp = _fingerprint(content, model_name)
    npz_path, meta_path = _cache_paths(cache_dir, model_name, fp)

    # store compactly
    np.savez_compressed(npz_path, embeddings=embeddings.astype(np.float32, copy=False),
                        content=np.array(_normalize_text_list(content), dtype=object))

    meta = {"model_name": model_name, "embedding_dim": int(embeddings.shape[1]), "n": int(embeddings.shape[0])}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)

    return npz_path

def load_embeddings_cache(content, cache_dir="bert_cache", model_name="distilbert-base-uncased"):
    fp = _fingerprint(content, model_name)
    npz_path, meta_path = _cache_paths(cache_dir, model_name, fp)
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)
    emb = data["embeddings"]
    cached_content = data["content"].tolist()

    # sanity: make sure order matches exactly
    if _normalize_text_list(content) != cached_content:
        return None
    return emb


# --- Helper methods for saving and loading BERT embeddings ---
def save_similarity_matrix(
    df: pd.DataFrame,
    name: str,                       # e.g. "user" or "item"
    out_dir: str = "artifacts/similarity",
    formats=("parquet", "csv", "pickle"),
    float_fmt="%.6f"
) -> dict:
    """
    Save a similarity matrix DataFrame in multiple formats with versioning.
    Returns dict of written paths.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = out / f"{name}_sim_{df.shape[0]}x{df.shape[1]}_{ts}"

    written = {}

    if "parquet" in formats:
        try:
            pq_path = base.with_suffix(".parquet")
            df.to_parquet(pq_path)  
            shutil.copy2(pq_path, out / f"{name}_sim_latest.parquet")
            written["parquet"] = str(pq_path)
        except Exception as e:
            print(f"[save_similarity_matrix] Parquet failed: {e}")

    if "csv" in formats:
        csv_path = base.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=True, compression="gzip", float_format=float_fmt)
        shutil.copy2(csv_path, out / f"{name}_sim_latest.csv.gz")
        written["csv"] = str(csv_path)

    if "pickle" in formats:
        pkl_path = base.with_suffix(".pkl")
        df.to_pickle(pkl_path)
        shutil.copy2(pkl_path, out / f"{name}_sim_latest.pkl")
        written["pickle"] = str(pkl_path)

    meta = {
        "name": name,
        "shape": df.shape,
        "created_at": ts,
        "formats": list(written.keys()),
        "index_type": str(type(df.index)),
        "columns_type": str(type(df.columns)),
        "dtype_summary": df.dtypes.astype(str).unique().tolist(),
        "sparsity": float((df.values == 0).sum() / df.size),
    }
    meta_path = base.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    shutil.copy2(meta_path, out / f"{name}_sim_latest.meta.json")
    written["meta"] = str(meta_path)

    return written

def load_similarity_matrix(path: str) -> pd.DataFrame:
    """Convenience loader based on file extension."""
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in {".gz", ".csv"}:
        return pd.read_csv(p, index_col=0)
    if p.suffix == ".pkl":
        return pd.read_pickle(p)
    raise ValueError(f"Unsupported format: {p.suffix}")
