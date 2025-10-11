import joblib
from pathlib import Path
from datetime import datetime
import shutil
import json
import pandas as pd
import os, json, hashlib
import numpy as np

# --- Helper methods for saving and loading hybrid model ---
def save_hybrid(hybrid, path="artifacts/hybrid_model.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(hybrid, path)
    print(f"[Hybrid] Saved to {path}")

def load_hybrid(path="artifacts/hybrid_model.joblib"):
    hybrid = joblib.load(path)
    print(f"[Hybrid] Loaded from {path}")
    return hybrid