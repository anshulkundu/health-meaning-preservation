from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INPUT_FILE = DATA_DIR / "source_verification_checklist.csv"
OUTPUT_FILE = DATA_DIR / "dataset_500_candidates.csv"

# Read existing candidate pairs
df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

required = [
    "pair_id",
    "language",
    "original_text",
    "comparison_text",
    "source_name",
    "source_url",
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[required].copy()

# Normalize language labels
df["language"] = df["language"].str.strip()

# Keep complete pairs only
df = df[
    df["original_text"].str.strip().ne("")
    & df["comparison_text"].str.strip().ne("")
    & df["language"].isin(["English", "Hindi"])
].copy()

# Remove exact duplicate pairs
df = df.drop_duplicates(
    subset=["language", "original_text", "comparison_text"]
).copy()

# Preserve original examples
df["candidate_origin"] = "existing_candidate"
df["candidate_type"] = "existing_pair"
df["meaning_score_1_to_10"] = ""
df["meaning_category"] = ""
df["critical_change"] = ""
df["review_status"] = "pending"
df["reviewer_notes"] = ""

# Do not manufacture new medical claims.
# Until additional source-backed pairs are collected, report the
# actual number available instead of pretending there are 500.
counts = df["language"].value_counts().to_dict()

print("\nEXISTING CANDIDATE DATASET")
print("-------------------------")
print(f"Unique pairs: {len(df)}")
print(f"English pairs: {counts.get('English', 0)}")
print(f"Hindi pairs: {counts.get('Hindi', 0)}")
print(f"Remaining to reach 500: {max(0, 500 - len(df))}")

if len(df) < 500:
    print("\nThe dataset currently has fewer than 500 unique pairs.")
    print("No fake or duplicated examples will be added.")
    print("New source-backed candidate pairs must be collected and reviewed.")

df.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved: {OUTPUT_FILE}")
print("Human scores remain blank.")
print("All pairs remain pending review.")