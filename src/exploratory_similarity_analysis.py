from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "exploratory_pseudo_labeled_500.csv"
RESULTS = ROOT / "results"

RESULTS.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(DATA)

print("\n========== DATASET OVERVIEW ==========")
print("Total pairs:", len(df))
print("\nLanguages:")
print(df["language"].value_counts())

# Calculate TF-IDF similarity
original = df["original_text"].fillna("").astype(str)
comparison = df["comparison_text"].fillna("").astype(str)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
matrix = vectorizer.fit_transform(
    pd.concat([original, comparison], ignore_index=True)
)

n = len(df)
similarities = cosine_similarity(
    matrix[:n], matrix[n:]
).diagonal()

df["tfidf_similarity"] = similarities

# Exploratory statistics
print("\n========== SIMILARITY STATISTICS ==========")
print(df["tfidf_similarity"].describe())

print("\n========== PSEUDO CATEGORIES ==========")
print(df["pseudo_category"].value_counts())

print("\n========== CRITICAL CHANGE FLAGS ==========")
print(df["pseudo_critical_change"].value_counts())

# Save detailed analysis
df.to_csv(
    RESULTS / "exploratory_similarity_results.csv",
    index=False
)

# Language distribution chart
df["language"].value_counts().plot(
    kind="bar", title="Dataset Language Distribution"
)
plt.xlabel("Language")
plt.ylabel("Number of Pairs")
plt.tight_layout()
plt.savefig(RESULTS / "language_distribution.png")
plt.close()

# Category distribution chart
df["pseudo_category"].value_counts().plot(
    kind="bar", title="Automatic Pseudo-Label Distribution"
)
plt.xlabel("Category")
plt.ylabel("Number of Pairs")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
plt.savefig(RESULTS / "category_distribution.png")
plt.close()

# Similarity distribution
plt.hist(df["tfidf_similarity"], bins=20)
plt.title("TF-IDF Similarity Distribution")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(RESULTS / "similarity_distribution.png")
plt.close()

# Save summary
summary = [
    "EXPLORATORY SIMILARITY ANALYSIS",
    f"Total pairs: {len(df)}",
    f"Average TF-IDF similarity: {similarities.mean():.4f}",
    f"Minimum similarity: {similarities.min():.4f}",
    f"Maximum similarity: {similarities.max():.4f}",
    "",
    "IMPORTANT: Pseudo-labels are automatically generated.",
    "They are not human-verified research labels.",
    "Similarity scores are exploratory, not model accuracy."
]

(RESULTS / "exploratory_analysis_summary.txt").write_text(
    "\n".join(summary), encoding="utf-8"
)

print("\nAnalysis completed successfully!")
print("Results saved in:", RESULTS)
print("Generated 3 charts, a CSV and a summary.")