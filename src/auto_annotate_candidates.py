import re
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher

ROOT = Path(__file__).resolve().parent.parent
candidates = [
    ROOT / "data" / "annotation_worksheet_500_clean.csv",
    ROOT / "data" / "dataset_500_candidates.csv",
    ROOT / "annotation_worksheet_500_clean.csv",
]
source = next((p for p in candidates if p.exists()), None)
if source is None:
    raise FileNotFoundError(
        "Could not find data/annotation_worksheet_500_clean.csv or "
        "data/dataset_500_candidates.csv. Put this script in the project root."
    )

df = pd.read_csv(source, encoding="utf-8-sig")

required = {"original_text", "comparison_text"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

def norm(s):
    return re.sub(r"\s+", " ", str(s).lower()).strip()

def has_negation(s):
    # English and common Hindi negation markers; this is a screening heuristic.
    return bool(re.search(
        r"\b(not|never|no|don't|do not|doesn't|didn't|cannot|can't|नहीं|मत)\b",
        norm(s)
    ))

def number_unit_signature(s):
    s = norm(s)
    return re.findall(
        r"\b\d+(?:\.\d+)?\s*(?:mg|g|kg|ml|l|tablet|tablets|times|hours?|days?|weeks?)?\b",
        s
    )

def critical_flag(a, b):
    if has_negation(a) != has_negation(b):
        return "Yes", "Possible negation change; verify manually"
    na, nb = number_unit_signature(a), number_unit_signature(b)
    if na != nb:
        return "Yes", "Possible number/dose/time change; verify manually"
    critical_terms = [
        "emergency", "urgent", "immediately", "severe", "stop",
        "continue", "share", "children", "expiry", "expired",
        "antibiotic", "viral", "prescription", "prescribed",
        "आपातकाल", "तुरंत", "बंद", "बच्चों", "समाप्ति", "दवा"
    ]
    changed = [w for w in critical_terms if (w in norm(a)) != (w in norm(b))]
    if changed:
        return "Review", "Safety-related term differs: " + ", ".join(changed[:4])
    return "No", "No simple critical-change trigger detected; still verify"

def assess(row):
    a, b = norm(row["original_text"]), norm(row["comparison_text"])
    ratio = SequenceMatcher(None, a, b).ratio()
    flag, note = critical_flag(a, b)

    if flag == "Yes":
        score, category = 1, "Critical change"
    elif flag == "Review":
        score, category = 5, "Needs review"
    elif ratio >= 0.88:
        score, category = 9, "Likely preserved"
    elif ratio >= 0.72:
        score, category = 7, "Possible minor change"
    elif ratio >= 0.50:
        score, category = 5, "Meaning may have changed"
    else:
        score, category = 3, "Major difference; review"

    return pd.Series([score, category, flag, note])

assessments = df.apply(assess, axis=1)
assessments.columns = [
    "auto_suggested_score_1_to_10",
    "auto_suggested_category",
    "auto_critical_change_flag",
    "auto_review_note",
]
out = pd.concat([df.reset_index(drop=True), assessments], axis=1)

# Do not falsely mark machine-generated suggestions as human annotations.
if "review_status" in out.columns:
    out["review_status"] = "auto_assisted_pending_human_review"

destination = ROOT / "data" / "annotation_worksheet_500_auto_assisted.csv"
destination.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(destination, index=False, encoding="utf-8-sig")

print(f"Loaded: {source}")
print(f"Rows processed: {len(out)}")
print(f"Saved: {destination}")
print("IMPORTANT: These are heuristic suggestions, NOT human-verified labels.")
print("Review flagged rows and verify medical claims/source URLs before research use.")
