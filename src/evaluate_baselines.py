from pathlib import Path
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error


# Project paths
ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data/processed/test.csv"
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(exist_ok=True)

# Multilingual Sentence Transformer
SENTENCE_MODEL_NAME = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


def calculate_metrics(name, y_true, y_pred):

    corr_ok = np.std(y_true) > 0 and np.std(y_pred) > 0

    return {
        "model": name,
        "n": len(y_true),

        "pearson_vs_pseudo": (
            float(pearsonr(y_true, y_pred).statistic)
            if corr_ok else float("nan")
        ),

        "spearman_vs_pseudo": (
            float(spearmanr(y_true, y_pred).statistic)
            if corr_ok else float("nan")
        ),

        "mae_vs_pseudo": float(
            mean_absolute_error(y_true, y_pred)
        ),

        "rmse_vs_pseudo": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),

        "reference_labels": "automatic_pseudo_labels_not_human_verified"
    }


def main():

    if not TEST_PATH.exists():
        raise SystemExit(
            "Test file not found. Run: python src/prepare_data.py"
        )

    df = pd.read_csv(TEST_PATH)

    required = {
        "original_text",
        "comparison_text",
        "pseudo_score"
    }

    missing = required - set(df.columns)

    if missing:
        raise SystemExit(
            f"Missing required columns: {sorted(missing)}"
        )

    # Prepare reference scores
    y = pd.to_numeric(
        df["pseudo_score"],
        errors="coerce"
    ).to_numpy(dtype=float)

    valid = np.isfinite(y)

    df = df.loc[valid].reset_index(drop=True)
    y = y[valid]

    if len(df) < 2:
        raise SystemExit("Need at least two valid test rows.")

    texts_a = (
        df["original_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    texts_b = (
        df["comparison_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # ==========================================
    # MODEL 1: TF-IDF COSINE SIMILARITY
    # ==========================================

    print("\nRunning TF-IDF model...")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        analyzer="char",
        min_df=1
    )

    matrix = vectorizer.fit_transform(
        texts_a + texts_b
    )

    tfidf_similarity = cosine_similarity(
        matrix[:len(texts_a)],
        matrix[len(texts_a):]
    ).diagonal()

    tfidf_predictions = np.clip(
        1 + 9 * tfidf_similarity,
        1,
        10
    )

    # ==========================================
    # MODEL 2: MULTILINGUAL SENTENCE TRANSFORMER
    # ==========================================

    print("\nLoading Multilingual Sentence Transformer...")

    try:
        from sentence_transformers import SentenceTransformer

    except ImportError:
        raise SystemExit(
            "Install sentence-transformers first:\n"
            "pip install sentence-transformers"
        )

    encoder = SentenceTransformer(
        SENTENCE_MODEL_NAME
    )

    print("\nGenerating embeddings...")

    embeddings_a = encoder.encode(
        texts_a,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    embeddings_b = encoder.encode(
        texts_b,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    # Cosine similarity
    sentence_similarity = np.sum(
        embeddings_a * embeddings_b,
        axis=1
    )

    # Convert similarity from [-1, 1] to [1, 10]
    sentence_predictions = np.clip(
        1 + 9 * ((sentence_similarity + 1) / 2),
        1,
        10
    )

    # ==========================================
    # CALCULATE METRICS
    # ==========================================

    metrics_rows = [

        calculate_metrics(
            "Character n-gram TF-IDF cosine",
            y,
            tfidf_predictions
        ),

        calculate_metrics(
            "Multilingual Sentence Transformer cosine",
            y,
            sentence_predictions
        )

    ]

    # ==========================================
    # SAVE PREDICTIONS
    # ==========================================

    pair_ids = (
        df["pair_id"]
        if "pair_id" in df.columns
        else pd.Series(range(len(df)))
    )

    languages = (
        df["language"]
        if "language" in df.columns
        else pd.Series(["unknown"] * len(df))
    )

    predictions = pd.DataFrame({

        "pair_id": pair_ids,

        "language": languages,

        "pseudo_score": y,

        "tfidf_predicted_score": tfidf_predictions,

        "sentence_transformer_cosine_similarity":
            sentence_similarity,

        "sentence_transformer_predicted_score":
            sentence_predictions

    })

    # Save CSV files
    metrics_df = pd.DataFrame(metrics_rows)

    metrics_df.to_csv(
        RESULTS_DIR / "baseline_metrics.csv",
        index=False
    )

    predictions.to_csv(
        RESULTS_DIR / "baseline_predictions.csv",
        index=False
    )

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================

    print("\n========== BASELINE MODEL COMPARISON ==========")

    print(
        metrics_df.to_string(index=False)
    )

    print("\nFiles saved successfully:")

    print("results/baseline_metrics.csv")

    print("results/baseline_predictions.csv")

    print("\nNote: Evaluation uses automatic pseudo-labels,")
    print("not human-verified annotations.")


if __name__ == "__main__":
    main()