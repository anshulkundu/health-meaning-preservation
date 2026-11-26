from pathlib import Path

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL = ROOT / "models" / "health_xlmr_pseudo"
TEST = ROOT / "data" / "processed" / "test.csv"
OUTPUT = ROOT / "results"


# --------------------------------------------------
# 2. CHECK FILES
# --------------------------------------------------

if not MODEL.exists():
    raise SystemExit("Trained model not found. Train the model first.")

if not TEST.exists():
    raise SystemExit("Test dataset not found. Prepare the data first.")


# --------------------------------------------------
# 3. LOAD TEST DATA
# --------------------------------------------------

df = pd.read_csv(TEST)

required_columns = [
    "original_text",
    "comparison_text",
    "pseudo_score",
]

for col in required_columns:
    if col not in df.columns:
        raise SystemExit(f"Missing column: {col}")

df["pseudo_score"] = pd.to_numeric(
    df["pseudo_score"],
    errors="coerce"
)

df = df.dropna(subset=required_columns).copy()

df["original_text"] = df["original_text"].astype(str)
df["comparison_text"] = df["comparison_text"].astype(str)

df = df[
    (df["original_text"].str.strip() != "")
    & (df["comparison_text"].str.strip() != "")
].copy()

if len(df) < 2:
    raise SystemExit("Not enough valid test rows.")

print("Test samples:", len(df))


# --------------------------------------------------
# 4. LOAD TOKENIZER
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(str(MODEL))


# --------------------------------------------------
# 5. CREATE DATASET WITHOUT LABELS
# --------------------------------------------------

eval_df = df[
    ["original_text", "comparison_text"]
].copy()

dataset = Dataset.from_pandas(
    eval_df,
    preserve_index=False
)


# --------------------------------------------------
# 6. TOKENIZATION
# --------------------------------------------------

def tokenize_function(batch):

    return tokenizer(
        batch["original_text"],
        batch["comparison_text"],
        truncation=True,
        max_length=256,
        padding=False,
    )


dataset = dataset.map(
    tokenize_function,
    batched=True
)


# --------------------------------------------------
# 7. REMOVE UNNECESSARY COLUMNS
# --------------------------------------------------

columns_to_keep = [
    "input_ids",
    "attention_mask",
    "token_type_ids",
]

columns_to_keep = [
    col for col in columns_to_keep
    if col in dataset.column_names
]

columns_to_remove = [
    col for col in dataset.column_names
    if col not in columns_to_keep
]

dataset = dataset.remove_columns(columns_to_remove)


# --------------------------------------------------
# 8. DYNAMIC PADDING
# --------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)


# --------------------------------------------------
# 9. LOAD TRAINED MODEL
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL)
)

model.config.problem_type = "regression"
model.config.num_labels = 1


# --------------------------------------------------
# 10. TRAINER CONFIGURATION
# --------------------------------------------------

training_args = TrainingArguments(
    output_dir=str(OUTPUT / "eval_tmp"),
    per_device_eval_batch_size=4,
    report_to="none",
    dataloader_pin_memory=False,
    fp16=False,
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
)


# --------------------------------------------------
# 11. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nEvaluating XLM-RoBERTa model...\n")

prediction_output = trainer.predict(dataset)

predictions = prediction_output.predictions

if isinstance(predictions, tuple):
    predictions = predictions[0]

predicted_scores = np.asarray(
    predictions,
    dtype=np.float32
).reshape(-1)

predicted_scores = np.clip(
    predicted_scores,
    1,
    10
)


# --------------------------------------------------
# 12. ACTUAL REFERENCE SCORES
# --------------------------------------------------

actual_scores = df["pseudo_score"].to_numpy(
    dtype=np.float32
)


# --------------------------------------------------
# 13. CALCULATE METRICS
# --------------------------------------------------

pearson = float("nan")
spearman = float("nan")

if (
    np.std(actual_scores) > 0
    and np.std(predicted_scores) > 0
):
    pearson = float(
        pearsonr(actual_scores, predicted_scores).statistic
    )

    spearman = float(
        spearmanr(actual_scores, predicted_scores).statistic
    )


mae = float(
    mean_absolute_error(
        actual_scores,
        predicted_scores
    )
)

rmse = float(
    np.sqrt(
        mean_squared_error(
            actual_scores,
            predicted_scores
        )
    )
)


# --------------------------------------------------
# 14. CREATE RESULTS
# --------------------------------------------------

result = {
    "model": "XLM-RoBERTa Regression",
    "n": len(actual_scores),
    "pearson_vs_pseudo": pearson,
    "spearman_vs_pseudo": spearman,
    "mae_vs_pseudo": mae,
    "rmse_vs_pseudo": rmse,
    "reference_labels": "automatic_pseudo_labels_not_human_verified",
}


metrics_df = pd.DataFrame([result])


# --------------------------------------------------
# 15. SAVE RESULTS
# --------------------------------------------------

OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)

metrics_df.to_csv(
    OUTPUT / "proposed_model_metrics.csv",
    index=False
)

predictions_df = df.copy()

predictions_df["predicted_score"] = predicted_scores

predictions_df.to_csv(
    OUTPUT / "proposed_model_predictions.csv",
    index=False
)


# --------------------------------------------------
# 16. DISPLAY RESULTS
# --------------------------------------------------

print("\n======================================")
print("XLM-RoBERTa EVALUATION RESULTS")
print("======================================")

print(metrics_df.to_string(index=False))

print("\nFiles saved:")

print(OUTPUT / "proposed_model_metrics.csv")
print(OUTPUT / "proposed_model_predictions.csv")

print("\nEVALUATION COMPLETED SUCCESSFULLY!")