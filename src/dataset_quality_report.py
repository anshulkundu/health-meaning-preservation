from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
INPUT = DATA / "dataset_500_candidates.csv"
OUTPUT = ROOT / "results" / "dataset_quality_report.txt"

if not INPUT.exists():
    raise SystemExit(f"Dataset not found: {INPUT}")

df = pd.read_csv(INPUT, dtype=str).fillna("")
report = []

def add(text=""):
    report.append(str(text))

add("HEALTH MEANING PRESERVATION — DATASET QUALITY REPORT")
add("=" * 58)
add(f"Total rows: {len(df)}")
add(f"Unique pair IDs: {df['pair_id'].nunique()}")
add()

add("LANGUAGE DISTRIBUTION")
add(df["language"].value_counts().to_string())
add()

add("DATA COMPLETENESS")
for col in [
    "original_text", "comparison_text", "source_name",
    "source_url", "review_status"
]:
    if col in df.columns:
        missing = df[col].astype(str).str.strip().eq("").sum()
        add(f"{col}: {len(df) - missing}/{len(df)} filled")

add()
add("DUPLICATE CHECKS")
add(
    "Duplicate sentence pairs: "
    + str(df.duplicated(
        subset=["language", "original_text", "comparison_text"]
    ).sum())
)
add(
    "Duplicate pair IDs: "
    + str(df["pair_id"].duplicated().sum())
)
add()

add("REVIEW STATUS")
add(df["review_status"].value_counts(dropna=False).to_string())
add()

if "meaning_score_1_to_10" in df.columns:
    scores = pd.to_numeric(
        df["meaning_score_1_to_10"], errors="coerce"
    )
    add(f"Rows with numeric meaning scores: {scores.notna().sum()}")
else:
    add("Rows with numeric meaning scores: column unavailable")

add()
add("IMPORTANT LIMITATIONS")
add("- Candidate pairs are not automatically validated research labels.")
add("- Source URLs require checking against each individual claim.")
add("- Hindi text requires fluent-speaker review.")
add("- Human meaning scores must come from genuine annotation.")
add("- Do not train or evaluate supervised models on invented labels.")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text("\n".join(report), encoding="utf-8")

print("\n".join(report))
print(f"\nReport saved to: {OUTPUT}")