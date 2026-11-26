from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

INPUT = DATA / "dataset_500_candidates.csv"
OUTPUT = DATA / "dataset_500_candidates_expanded.csv"

TARGET = 500

if not INPUT.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT}")

df = pd.read_csv(INPUT, dtype=str).fillna("")

required = {
    "pair_id",
    "language",
    "original_text",
    "comparison_text",
    "source_name",
    "source_url",
    "candidate_origin",
    "candidate_type",
    "meaning_score_1_to_10",
    "meaning_category",
    "critical_change",
    "review_status",
    "reviewer_notes",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

# Preserve existing rows. Do not invent human scores.
existing = df.copy()

# These transformations are candidate-generation heuristics only.
# They are NOT guaranteed to preserve meaning or medical accuracy.
english_rules = [
    ("before", "prior to"),
    ("after", "following"),
    ("regularly", "on a regular basis"),
    ("generally", "usually"),
    ("can help", "may help"),
]

hindi_rules = [
    ("पहले", "से पूर्व"),
    ("बाद", "के पश्चात"),
    ("नियमित रूप से", "लगातार"),
    ("आमतौर पर", "सामान्यतः"),
    ("मदद कर सकता है", "सहायता कर सकता है"),
]

def make_variants(text, language):
    rules = english_rules if language == "English" else hindi_rules
    variants = []

    for old, new in rules:
        if old in text:
            candidate = text.replace(old, new, 1)
            if candidate != text:
                variants.append(candidate)

    return list(dict.fromkeys(variants))


new_rows = []
existing_keys = set(
    zip(
        existing["language"],
        existing["original_text"],
        existing["comparison_text"],
    )
)

next_id = 1
while f"AUG{next_id:04d}" in set(existing["pair_id"]):
    next_id += 1

for _, row in existing.iterrows():
    if len(existing) + len(new_rows) >= TARGET:
        break

    language = row["language"]
    original = row["original_text"]

    # Generate only candidates whose text can actually be transformed.
    variants = make_variants(original, language)

    for variant in variants:
        key = (language, original, variant)

        if key in existing_keys:
            continue

        if len(existing) + len(new_rows) >= TARGET:
            break

        new_rows.append({
            "pair_id": f"AUG{next_id:04d}",
            "language": language,
            "original_text": original,
            "comparison_text": variant,
            "source_name": row["source_name"],
            "source_url": row["source_url"],
            "candidate_origin": "synthetic_augmentation_pending_review",
            "candidate_type": "automatic_paraphrase_candidate",
            "meaning_score_1_to_10": "",
            "meaning_category": "",
            "critical_change": "",
            "review_status": "pending",
            "reviewer_notes": (
                "Automatically generated candidate. "
                "Verify source support, meaning preservation, and fluency "
                "before research use. Not a human annotation."
            ),
        })

        existing_keys.add(key)
        next_id += 1

if new_rows:
    additions = pd.DataFrame(new_rows)
    expanded = pd.concat([existing, additions], ignore_index=True)
else:
    expanded = existing.copy()

expanded.to_csv(OUTPUT, index=False)

print("\nDATASET EXPANSION REPORT")
print("=" * 35)
print("Existing rows preserved:", len(existing))
print("New candidates generated:", len(new_rows))
print("Total rows:", len(expanded))
print("English:", (expanded["language"] == "English").sum())
print("Hindi:", (expanded["language"] == "Hindi").sum())
print("Output:", OUTPUT)

if len(expanded) < TARGET:
    print(
        f"\nTarget not reached: {TARGET - len(expanded)} "
        "additional candidates are still needed."
    )
    print(
        "More diverse, source-reviewed seed sentences are required. "
        "The script will not fabricate medical content to fill the gap."
    )
else:
    print(
        "\n500 rows reached. Automatically generated candidates "
        "still require review before research use."
    )