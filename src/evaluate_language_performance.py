from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. PROJECT PATHS
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

BASELINE_FILE = RESULTS / "baseline_predictions.csv"
XLM_FILE = RESULTS / "proposed_model_predictions.csv"


# ==========================================
# 2. CHECK FILES
# ==========================================

for file in [BASELINE_FILE, XLM_FILE]:
    if not file.exists():
        raise SystemExit(f"Required file not found: {file}")


# ==========================================
# 3. LOAD DATA
# ==========================================

baseline = pd.read_csv(BASELINE_FILE)
xlm = pd.read_csv(XLM_FILE)

print("Baseline columns:", baseline.columns.tolist())
print("XLM-R columns:", xlm.columns.tolist())


# ==========================================
# 4. VALIDATE REQUIRED COLUMNS
# ==========================================

required_baseline = [
    "language",
    "pseudo_score",
    "tfidf_predicted_score",
    "sentence_transformer_predicted_score",
]

for col in required_baseline:
    if col not in baseline.columns:
        raise SystemExit(f"Missing baseline column: {col}")

required_xlm = [
    "language",
    "pseudo_score",
    "predicted_score",
]

for col in required_xlm:
    if col not in xlm.columns:
        raise SystemExit(f"Missing XLM-R column: {col}")


# ==========================================
# 5. METRIC FUNCTION
# ==========================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    valid = np.isfinite(actual) & np.isfinite(predicted)

    actual = actual[valid]
    predicted = predicted[valid]

    if len(actual) < 2:
        return {
            "n": len(actual),
            "pearson": np.nan,
            "spearman": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
        }

    pearson = np.nan
    spearman = np.nan

    if np.std(actual) > 0 and np.std(predicted) > 0:
        pearson = pearsonr(actual, predicted).statistic
        spearman = spearmanr(actual, predicted).statistic

    return {
        "n": len(actual),
        "pearson": pearson,
        "spearman": spearman,
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
    }


# ==========================================
# 6. NORMALIZE LANGUAGE NAMES
# ==========================================

def normalize_language(value):

    value = str(value).strip().lower()

    if value in ["english", "en"]:
        return "English"

    if value in ["hindi", "hi"]:
        return "Hindi"

    return None


baseline["language"] = baseline["language"].apply(
    normalize_language
)

xlm["language"] = xlm["language"].apply(
    normalize_language
)


# ==========================================
# 7. EVALUATE BASELINE MODELS
# ==========================================

records = []

baseline_models = {
    "TF-IDF": "tfidf_predicted_score",
    "Sentence Transformer": "sentence_transformer_predicted_score",
}

for language in ["English", "Hindi"]:

    subset = baseline[
        baseline["language"] == language
    ].copy()

    for model_name, prediction_column in baseline_models.items():

        subset[prediction_column] = pd.to_numeric(
            subset[prediction_column],
            errors="coerce"
        )

        actual = pd.to_numeric(
            subset["pseudo_score"],
            errors="coerce"
        )

        metrics = calculate_metrics(
            actual,
            subset[prediction_column]
        )

        records.append({
            "language": language,
            "model": model_name,
            **metrics,
        })


# ==========================================
# 8. EVALUATE XLM-RoBERTa
# ==========================================

for language in ["English", "Hindi"]:

    subset = xlm[
        xlm["language"] == language
    ].copy()

    actual = pd.to_numeric(
        subset["pseudo_score"],
        errors="coerce"
    )

    predicted = pd.to_numeric(
        subset["predicted_score"],
        errors="coerce"
    )

    metrics = calculate_metrics(
        actual,
        predicted
    )

    records.append({
        "language": language,
        "model": "XLM-RoBERTa",
        **metrics,
    })


# ==========================================
# 9. CREATE RESULTS TABLE
# ==========================================

results = pd.DataFrame(records)

results = results[
    [
        "language",
        "model",
        "n",
        "pearson",
        "spearman",
        "mae",
        "rmse",
    ]
]

# Round numerical metrics
results[["pearson", "spearman", "mae", "rmse"]] = (
    results[["pearson", "spearman", "mae", "rmse"]].round(4)
)


# ==========================================
# 10. SAVE RESULTS
# ==========================================

output_file = RESULTS / "language_wise_model_metrics.csv"

results.to_csv(
    output_file,
    index=False
)


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n==========================================")
print("RQ3: ENGLISH VS HINDI MODEL EVALUATION")
print("==========================================")

print(results.to_string(index=False))

print("\nResults saved to:")
print(output_file)

print("\nIMPORTANT:")
print("Evaluation uses automatic pseudo-labels.")
print("These are not human-verified results.")
print("Language-wise results are exploratory only.")

print("\nRQ3 EVALUATION COMPLETED SUCCESSFULLY!")
