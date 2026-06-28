"""
Pre-computation step — run this ONCE before ranker.py.
Downloads BAAI/bge-small-en-v1.5 (~130MB) from HuggingFace and
caches it locally in models/bge-small-en-v1.5/ so ranker.py
can run fully offline with no network access.

Usage:
    python download_model.py
"""

import os
from sentence_transformers import SentenceTransformer

MODEL_NAME  = "BAAI/bge-small-en-v1.5"
LOCAL_PATH  = os.path.join(os.path.dirname(__file__), "models", "bge-small-en-v1.5")

def main():
    if os.path.exists(LOCAL_PATH) and os.path.exists(os.path.join(LOCAL_PATH, "model.safetensors")):
        print(f"✅ Model already cached at: {LOCAL_PATH}")
        return

    print(f"⬇️  Downloading {MODEL_NAME} from HuggingFace Hub (~130 MB)...")
    model = SentenceTransformer(MODEL_NAME)
    model.save(LOCAL_PATH)
    print(f"✅ Model saved to: {LOCAL_PATH}")
    print("   You can now run ranker.py with no network access.")

if __name__ == "__main__":
    main()
