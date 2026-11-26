from pathlib import Path
import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "annotation_worksheet_500_auto_assisted.csv"
OUT = DATA / "processed"

OUT.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    INPUT = DATA / "dataset_500_candidates.csv"

if not INPUT.exists():
    raise FileNotFoundError("Could not find the 500-pair dataset.")

df = pd.read_csv(INPUT).fillna("")

# Convert automatic scores into numeric pseudo-labels.
df["pseudo_score"] = pd.to_numeric(
    df["auto_suggested_score_1_to_10"], errors="coerce"
)

# Keep rows with valid automatic scores.
df = df[df["pseudo_score"].between(1, 10)].copy()

if len(df) < 10:
    raise ValueError("Not enough rows with valid automatic scores.")

# Preserve the distinction between pseudo-labels and human labels.
df["label_source"] = "automatic_pseudo_label"
df["human_verified"] = False

df["target_category"] = df["auto_suggested_category"]

df["target_critical_change"] = (
    df["auto_critical_change_flag"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

# Group variants of the same original sentence to reduce leakage.
def normalize(text):
    return re.sub(r"\s+", " ", str(text).lower()).strip()

df["split_group"] = df["original_text"].apply(normalize)

# Use pair ID if an original sentence is missing.
df.loc[df["split_group"] == "", "split_group"] = (
    df.loc[df["split_group"] == "", "pair_id"].astype(str)
)

# 80% train+validation, 20% test.
split1 = GroupShuffleSplit(
    n_splits=1, test_size=0.20, random_state=42
)

train_val_idx, test_idx = next(
    split1.split(df, groups=df["split_group"])
)

train_val = df.iloc[train_val_idx].copy()
test = df.iloc[test_idx].copy()

# Split remaining data into 75% train and 25% validation.
split2 = GroupShuffleSplit(
    n_splits=1, test_size=0.25, random_state=43
)

train_idx, val_idx = next(
    split2.split(train_val, groups=train_val["split_group"])
)

train = train_val.iloc[train_idx].copy()
val = train_val.iloc[val_idx].copy()

# Save datasets.
train.to_csv(OUT / "train.csv", index=False)
val.to_csv(OUT / "validation.csv", index=False)
test.to_csv(OUT / "test.csv", index=False)

print("\nDATA PREPARATION COMPLETED")
print("--------------------------")
print("Input:", INPUT.name)
print("Total pseudo-labelled rows:", len(df))
print("Training rows:", len(train))
print("Validation rows:", len(val))
print("Testing rows:", len(test))

print("\nLanguage distribution:")
print(df["language"].value_counts())

print("\nPseudo-score distribution:")
print(df["pseudo_score"].value_counts().sort_index())

print("\nAll labels are automatic pseudo-labels.")
print("They are NOT human-verified annotations.")
print("\nOutput folder:", OUT)