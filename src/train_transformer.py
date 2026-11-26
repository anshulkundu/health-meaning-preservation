from pathlib import Path
import numpy as np
import pandas as pd
import torch

from datasets import Dataset, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# --------------------------------------------------
# 1. PROJECT PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models" / "health_xlmr_pseudo"

TRAIN_FILE = DATA_DIR / "train.csv"

# Find the validation file
VALIDATION_OPTIONS = [
    DATA_DIR / "validation.csv",
    DATA_DIR / "val.csv",
    DATA_DIR / "valid.csv",
]

VALID_FILE = next(
    (p for p in VALIDATION_OPTIONS if p.exists()),
    None
)

BASE_MODEL = "FacebookAI/xlm-roberta-base"

MAX_LENGTH = 256
BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 2e-5


# --------------------------------------------------
# 2. CHECK DATA FILES
# --------------------------------------------------

if not TRAIN_FILE.exists():
    raise SystemExit(
        f"Training file not found: {TRAIN_FILE}\n"
        "Run prepare_data.py first."
    )

if VALID_FILE is None:
    raise SystemExit(
        "Validation CSV not found.\n"
        "Expected validation.csv, val.csv or valid.csv "
        "inside data/processed."
    )


# --------------------------------------------------
# 3. LOAD DATA
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_FILE)
valid_df = pd.read_csv(VALID_FILE)

print("\nTraining rows:", len(train_df))
print("Validation rows:", len(valid_df))


# --------------------------------------------------
# 4. IDENTIFY THE SCORE COLUMN
# --------------------------------------------------

possible_score_columns = [
    "pseudo_score",
    "meaning_score",
    "score",
    "labels",
]

score_column = next(
    (
        col for col in possible_score_columns
        if col in train_df.columns
        and col in valid_df.columns
    ),
    None
)

if score_column is None:
    raise SystemExit(
        "Could not find a score column.\n"
        f"Available columns: {list(train_df.columns)}"
    )


# --------------------------------------------------
# 5. CLEAN AND CONVERT SCORES
# --------------------------------------------------

for df in [train_df, valid_df]:

    required_columns = [
        "original_text",
        "comparison_text",
        score_column,
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise SystemExit(
            f"Missing columns: {missing}"
        )

    df[score_column] = pd.to_numeric(
        df[score_column],
        errors="coerce"
    )

    df.dropna(
        subset=required_columns,
        inplace=True
    )

    # IMPORTANT: Regression labels must be floating-point.
    df[score_column] = df[score_column].astype(
        np.float32
    )

    # Keep scores within the intended 1-10 range.
    df[score_column] = df[score_column].clip(1, 10)


if len(train_df) == 0 or len(valid_df) == 0:
    raise SystemExit(
        "No usable training or validation rows remain."
    )

print("\nUsable training rows:", len(train_df))
print("Usable validation rows:", len(valid_df))
print("Score column:", score_column)


# --------------------------------------------------
# 6. RENAME SCORE COLUMN TO LABELS
# --------------------------------------------------

train_df = train_df.rename(
    columns={score_column: "labels"}
)

valid_df = valid_df.rename(
    columns={score_column: "labels"}
)


# --------------------------------------------------
# 7. LOAD TOKENIZER
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL
)


# --------------------------------------------------
# 8. TOKENIZATION FUNCTION
# --------------------------------------------------

def tokenize_function(batch):

    result = tokenizer(
        batch["original_text"],
        batch["comparison_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
    )

    # Ensure regression labels are float32.
    result["labels"] = [
        np.float32(x)
        for x in batch["labels"]
    ]

    return result


# --------------------------------------------------
# 9. CONVERT PANDAS TO HUGGING FACE DATASETS
# --------------------------------------------------

train_ds = Dataset.from_pandas(
    train_df[
        ["original_text", "comparison_text", "labels"]
    ],
    preserve_index=False,
)

valid_ds = Dataset.from_pandas(
    valid_df[
        ["original_text", "comparison_text", "labels"]
    ],
    preserve_index=False,
)


# --------------------------------------------------
# 10. TOKENIZE DATASETS
# --------------------------------------------------

train_ds = train_ds.map(
    tokenize_function,
    batched=True,
)

valid_ds = valid_ds.map(
    tokenize_function,
    batched=True,
)


# --------------------------------------------------
# 11. FORCE FLOAT32 LABELS
# --------------------------------------------------

train_ds = train_ds.cast_column(
    "labels",
    Value("float32"),
)

valid_ds = valid_ds.cast_column(
    "labels",
    Value("float32"),
)


# --------------------------------------------------
# 12. DYNAMIC PADDING
# --------------------------------------------------

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)


# --------------------------------------------------
# 13. LOAD XLM-ROBERTA REGRESSION MODEL
# --------------------------------------------------

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    problem_type="regression",
)

model.config.problem_type = "regression"
model.config.num_labels = 1


# --------------------------------------------------
# 14. TRAINING ARGUMENTS
# --------------------------------------------------

training_args = TrainingArguments(

    output_dir=str(ROOT / "results" / "xlmr_checkpoints"),

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    logging_steps=10,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",

    greater_is_better=False,

    report_to="none",

    dataloader_pin_memory=False,

    fp16=False,

    seed=42,
)


# --------------------------------------------------
# 15. INITIALIZE TRAINER
# --------------------------------------------------

trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_ds,

    eval_dataset=valid_ds,

    tokenizer=tokenizer,

    data_collator=data_collator,
)


# --------------------------------------------------
# 16. TRAIN MODEL
# --------------------------------------------------

print("\nStarting XLM-RoBERTa training...\n")

trainer.train()


# --------------------------------------------------
# 17. SAVE TRAINED MODEL
# --------------------------------------------------

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

trainer.save_model(str(MODEL_DIR))

tokenizer.save_pretrained(str(MODEL_DIR))


# --------------------------------------------------
# 18. FINAL VALIDATION
# --------------------------------------------------

print("\nRunning final validation...\n")

metrics = trainer.evaluate()

print("\nFINAL VALIDATION METRICS")

for key, value in metrics.items():

    print(f"{key}: {value}")


# --------------------------------------------------
# 19. SAVE METRICS
# --------------------------------------------------

results_dir = ROOT / "results"

results_dir.mkdir(
    parents=True,
    exist_ok=True,
)

pd.DataFrame([metrics]).to_csv(
    results_dir / "training_metrics.csv",
    index=False,
)


print("\n-----------------------------------")
print("TRAINING COMPLETED")
print("-----------------------------------")

print("Model saved at:")
print(MODEL_DIR)

print("\nMetrics saved at:")
print(results_dir / "training_metrics.csv")