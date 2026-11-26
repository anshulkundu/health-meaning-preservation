# Run the project (macOS)

Open Terminal in this project folder. Use a fresh virtual environment (do not reuse the archived `.venv`):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Prepare the data and create leakage-aware splits:

```bash
python src/prepare_data.py
python src/evaluate_baselines.py
```

Train the exploratory XLM-R model (first run downloads model weights and needs internet; training may take time):

```bash
python src/train_transformer.py --epochs 2 --batch-size 4
python src/evaluate_transformer.py
```

Run the dashboard in another terminal (activate the same environment first):

```bash
streamlit run app.py
```

## Interpretation / research integrity

The current 500-row worksheet has automatically suggested scores. These are **pseudo-labels**, not human-verified annotations. The training script uses `pseudo_score` and names the model `health_xlmr_pseudo` to make this explicit. All model metrics from this pipeline are comparisons against pseudo-labels and must not be presented as human-validated meaning-preservation performance, clinical correctness, or medical safety.

For a defensible final research evaluation, replace/add actual independently assigned human scores in the annotation worksheet, then update the preparation pipeline to split on those human scores. Keep annotator identity, adjudication and source verification documented.

The dashboard's critical-change flags are rule-based review cues. They are not a diagnosis, treatment recommendation, or guarantee that every clinically important change will be detected.
