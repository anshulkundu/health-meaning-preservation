from pathlib import Path
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data/annotation_candidates.csv"
OUTPUT = ROOT / "data/auto_labeled_candidates.csv"

NEGATION_TERMS = [
    "not", "no", "never", "don't", "do not", "cannot", "can't", "without",
    "नहीं", "न", "मत", "बिना", "नहीं करतीं", "नहीं करें"
]

def contains_negation(text):
    s = str(text).lower()
    return any(term in s for term in NEGATION_TERMS)

def critical_flag(a, b):
    # This is only a coarse review flag, not a clinical validator.
    na, nb = contains_negation(a), contains_negation(b)
    if na != nb:
        return "REVIEW_NEGATION_MISMATCH"
    return ""

def score_to_label(score):
    if score >= 8.5:
        return "provisionally_high_similarity"
    if score >= 6.0:
        return "provisionally_medium_similarity"
    return "provisionally_low_similarity"

def main():
    if not INPUT.exists():
        raise SystemExit(f"Missing input CSV: {INPUT}")
    df = pd.read_csv(INPUT)
    required = ["pair_id", "original_text", "comparison_text", "language", "task_type"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns: {missing}")
    if df.empty:
        raise SystemExit("Input CSV has no rows.")
    a = df["original_text"].astype(str).tolist()
    b = df["comparison_text"].astype(str).tolist()

    # TF-IDF is included as a lightweight reference score.
    vec = TfidfVectorizer(ngram_range=(1, 2))
    mat = vec.fit_transform(a + b)
    n = len(a)
    tfidf = cosine_similarity(mat[:n], mat[n:]).diagonal()

    # Multilingual semantic similarity.
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    print(f"Loading {model_name}; first run downloads model weights.")
    model = SentenceTransformer(model_name)
    ea = model.encode(a, normalize_embeddings=True, show_progress_bar=True)
    eb = model.encode(b, normalize_embeddings=True, show_progress_bar=True)
    semantic = np.sum(ea * eb, axis=1)
    # Map cosine [-1,1] to a display range [1,10]; not a calibrated LMP score.
    auto_score = np.clip(1 + 9 * ((semantic + 1) / 2), 1, 10)

    df["tfidf_similarity"] = tfidf
    df["embedding_cosine"] = semantic
    df["auto_score"] = np.round(auto_score, 2)
    df["auto_label"] = [score_to_label(x) for x in auto_score]
    df["review_flag"] = [
        critical_flag(x, y) or "REVIEW_HEALTH_CONTENT_AND_SOURCE"
        for x, y in zip(a, b)
    ]
    df["label_source"] = "automatic_similarity_proxy_not_human_annotation"
    df.to_csv(OUTPUT, index=False)
    print(f"Wrote {len(df)} provisional rows to: {OUTPUT}")
    print("IMPORTANT: scores are similarity proxies, not validated human LMP scores.")
    print("Review all flagged rows and verify every source before research use.")

if __name__ == "__main__":
    main()
