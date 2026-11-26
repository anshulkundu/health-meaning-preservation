# Health Meaning Preservation (English + Hindi)

Scope: general health education, English/Hindi. One proposed model (fine-tuned XLM-RoBERTa regression) compared with three existing approaches: TF-IDF cosine, multilingual Sentence Transformer cosine, and JUDGEBERT where compatible released code/weights can be run.

## Important
`data/demo_examples.csv` is synthetic pipeline-test data. Its scores are illustrative, not human annotations or research results. Replace it with real, sourced and human-annotated examples before reporting results.

## Run in VS Code (Python 3.10/3.11)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp data/annotation_template.csv data/health_dataset.csv
# Add real annotated data, then:
python src/prepare_data.py
python src/evaluate_baselines.py
python src/train_transformer.py --epochs 2
python src/evaluate_transformer.py
```
The transformer needs sufficient compute and downloads model weights on first run.

Dataset columns: pair_id, original_text, comparison_text, language, task_type, human_score, label, error_type, source, annotator_id, split_group. Keep related paraphrases/translations in the same `split_group` to reduce data leakage.

JUDGEBERT was developed for legal meaning preservation and is not validated for health text. Include it as a comparator only if the original implementation and artifacts can be run with a documented protocol; do not label a substitute model as JUDGEBERT.

This is an NLP research prototype, not a medical decision system.
