from pathlib import Path
import pandas as pd
import streamlit as st
from src.analysis_engine import analyze_pair

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data' / 'dataset_500_candidates.csv'
st.set_page_config(page_title='Health Meaning Preservation', page_icon='🩺', layout='wide')
st.title('Multilingual Health Information Meaning Preservation')
st.caption('Research prototype • English and Hindi • Not a medical decision system')
st.warning('This is a research prototype. Similarity and model scores are not human meaning ratings or medical safety guarantees. Current training labels are automatic pseudo-labels unless independently human-annotated. Always have health instructions reviewed by a qualified person.')

if DATA.exists():
    try:
        df = pd.read_csv(DATA)
        c1,c2,c3 = st.columns(3)
        c1.metric('Candidate pairs', len(df))
        c2.metric('English pairs', int((df['language']=='English').sum()))
        c3.metric('Hindi pairs', int((df['language']=='Hindi').sum()))
    except Exception:
        df = pd.DataFrame()
else:
    df = pd.DataFrame()

st.subheader('Compare two health-related sentences')
lang = st.selectbox('Language', ['English', 'Hindi'])
col1,col2 = st.columns(2)
with col1:
    original = st.text_area('Original sentence', 'Do not share prescription medicines with other people.', height=130)
with col2:
    comparison = st.text_area('Comparison sentence', 'Share your prescription medicine with other people.', height=130)
if st.button('Analyze meaning differences', type='primary'):
    if not original.strip() or not comparison.strip():
        st.error('Please enter both sentences.')
    else:
        result = analyze_pair(original, comparison, lang)
        st.metric('Text similarity (baseline)', f"{result['similarity_score']}%")
        st.info(result['interpretation'])
        if result['flags']:
            st.subheader('Potential critical changes')
            for f in result['flags']:
                st.error(f"{f['type']} — {f['severity']}\n\n{f['detail']}")
        else:
            st.success('No configured critical-change pattern was detected.')
        x,y = st.columns(2)
        with x:
            st.markdown('**Terms only in original**')
            st.write(', '.join(result['removed_terms']) or 'None detected')
        with y:
            st.markdown('**Terms only in comparison**')
            st.write(', '.join(result['added_terms']) or 'None detected')
        st.caption('Flags are rule-based cues for human review. The system may miss changes or flag harmless wording differences.')
        model_dir = ROOT / 'models' / 'health_xlmr_pseudo'
        if model_dir.exists():
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                import torch
                @st.cache_resource
                def load_xlmr(path):
                    return AutoTokenizer.from_pretrained(path), AutoModelForSequenceClassification.from_pretrained(path)
                tokenizer, model = load_xlmr(str(model_dir))
                encoded = tokenizer(original, comparison, return_tensors='pt', truncation=True, max_length=256)
                with torch.no_grad():
                    prediction = float(model(**encoded).logits.reshape(-1)[0].item())
                st.metric('XLM-RoBERTa predicted pseudo-score (1–10)', f'{max(1,min(10,prediction)):.2f}')
                st.caption('Exploratory model trained on automatic pseudo-labels; not human-validated.')
            except Exception as exc:
                st.info(f'XLM-R model files exist but could not be loaded: {exc}')
        else:
            st.info('XLM-RoBERTa model has not been trained yet. Follow RUN_PROJECT.md to train it.')

if not df.empty:
    with st.expander('Inspect dataset examples'):
        sample = df[['pair_id','language','original_text','comparison_text','candidate_type','review_status']].head(10)
        st.dataframe(sample, use_container_width=True)
        st.caption('The current 500 rows are candidate pairs. The quality report records zero human meaning scores; they are not verified ground-truth labels.')

st.subheader('Research status')
st.write('Implemented in this prototype: dataset overview, character n-gram TF-IDF similarity, and rule-based cues for negation, quantities, timing, and warning language.')
st.write('Not yet validated: human-rated meaning scores, clinical correctness, and cross-language fairness.')

metrics_file = ROOT / 'results' / 'baseline_metrics.csv'
if metrics_file.exists():
    st.subheader('Baseline evaluation (against automatic pseudo-labels)')
    st.dataframe(pd.read_csv(metrics_file), use_container_width=True)
    st.caption('These are exploratory agreement metrics against automatic labels, not human ground truth.')
model_metrics = ROOT / 'results' / 'proposed_model_metrics.csv'
if model_metrics.exists():
    st.subheader('XLM-RoBERTa evaluation (against automatic pseudo-labels)')
    st.dataframe(pd.read_csv(model_metrics), use_container_width=True)
