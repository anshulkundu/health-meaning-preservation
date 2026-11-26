from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
test_path = ROOT / "data/processed/test.csv"
if not test_path.exists():
    raise SystemExit("Run python src/prepare_data.py first.")
df = pd.read_csv(test_path)
if len(df) < 2:
    raise SystemExit("Need at least two test rows.")
if "pseudo_score" not in df.columns:
    raise SystemExit("Expected pseudo_score. Re-run src/prepare_data.py.")
y = pd.to_numeric(df.pseudo_score, errors="coerce").to_numpy(float)
valid = np.isfinite(y)
df = df.loc[valid].reset_index(drop=True); y = y[valid]
a = df.original_text.astype(str).tolist(); b = df.comparison_text.astype(str).tolist()
def metrics(name, pred):
    return {"model": name, "n": len(y), "pearson_vs_pseudo": float(pearsonr(y,pred).statistic) if np.std(pred)>0 and np.std(y)>0 else float("nan"), "spearman_vs_pseudo": float(spearmanr(y,pred).statistic) if np.std(pred)>0 and np.std(y)>0 else float("nan"), "mae_vs_pseudo": float(mean_absolute_error(y,pred)), "rmse_vs_pseudo": float(np.sqrt(mean_squared_error(y,pred))), "reference_labels": "automatic_pseudo_labels_not_human_verified"}
vec = TfidfVectorizer(ngram_range=(1,2), analyzer="char", min_df=1)
mat = vec.fit_transform(a+b)
sim = cosine_similarity(mat[:len(a)], mat[len(a):]).diagonal()
pred = np.clip(1 + 9*sim, 1, 10)
rows = [metrics("Character n-gram TF-IDF cosine", pred)]
out = ROOT / "results"; out.mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(out/"baseline_metrics.csv", index=False)
pd.DataFrame({"pair_id":df.get("pair_id", pd.Series(range(len(df)))), "language":df.get("language", ""), "pseudo_score":y, "tfidf_predicted_score":pred}).to_csv(out/"baseline_predictions.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("Saved results/baseline_metrics.csv. Metrics are against automatic pseudo-labels, not human ratings.")
