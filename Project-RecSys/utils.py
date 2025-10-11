import joblib
from pathlib import Path
from datetime import datetime
import shutil
import json
import pandas as pd
import os, json, hashlib
import numpy as np
import pickle


# --- Helper methods for saving and loading hybrid model ---
def save_hybrid(hybrid, path="artifacts/hybrid_model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid, path)
    print(f"[Hybrid] Saved to {path}")


def load_hybrid(path="artifacts/hybrid_model.joblib"):
    hybrid = joblib.load(path)
    print(f"[Hybrid] Loaded from {path}")
    return hybrid


def save_embeddings(emb, path):
    if not path.endswith(".pkl"):
        path += ".pkl"
    with open(path, "wb") as f:
        pickle.dump(emb, f)
    print(f"Embeddings saved to {path}")


def load_embeddings(path):
    with open(path, "rb") as f:
        emb = pickle.load(f)
    print(f"Embeddings loaded from {path}")
    return emb
