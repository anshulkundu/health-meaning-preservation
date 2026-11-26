"""Exploratory RQ4 critical-change evaluation. Run: python src/evaluate_rq4_critical_changes.py
Evaluates a simple heuristic against provisional AI-assisted labels; not human-validated.
"""
import os, re
import pandas as pd
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df=pd.read_csv(os.path.join(ROOT,"data","dataset_500_provisional_annotations.csv"))
VOCAB={
"Negation":{"en":["no","not","never","without","don't","doesn't","can't","cannot","shouldn't","mustn't"],"hi":["नहीं","मत","बिना","न"]},
"Timing / frequency":{"en":["daily","day","days","weekly","week","monthly","month","morning","evening","night","before","after","until","every","hour","hours"],"hi":["दिन","रोज","प्रतिदिन","सुबह","शाम","रात","पहले","बाद","घंटे","हफ्ते","महीने"]},
"Warnings / severity":{"en":["warning","severe","serious","emergency","urgent","danger","avoid","stop","immediately","critical"],"hi":["चेतावनी","गंभीर","आपातकाल","तुरंत","खतरा","बचें","रोकें"]},
"Actor":{"en":["patient","doctor","healthcare professional","caregiver","child","adult","you","your"],"hi":["मरीज","रोगी","डॉक्टर","चिकित्सक","देखभालकर्ता","बच्चा","वयस्क","आप"]}}
def detect(r):
 lang="hi" if str(r.language).lower().startswith("h") else "en"; a=str(r.original_text).lower(); b=str(r.comparison_text).lower(); out=[]
 rx=r"\\d+(?:\\.\\d+)?\\s*(?:mg|mcg|g|kg|ml|l|tablets?|capsules?|%|मिलीलीटर|ग्राम|गोली|टैबलेट)?"
 if re.findall(rx,a)!=re.findall(rx,b): out.append("Dosage / quantity")
 for label,v in VOCAB.items():
  terms=v[lang]; ha=any(t in a for t in terms); hb=any(t in b for t in terms)
  if ha!=hb: out.append(label)
  elif label=="Actor":
   aa=[t for t in terms if t in a]; bb=[t for t in terms if t in b]
   if aa!=bb and (aa or bb): out.append(label)
 return out
def gold(v):
 if pd.isna(v): return []
 s=str(v).lower(); m={"Negation":["negation"],"Dosage / quantity":["quantity","dosage","required component"],"Timing / frequency":["timing","frequency"],"Warnings / severity":["warning","severity","safety"],"Actor":["actor"]}
 return [k for k,ts in m.items() if any(t in s for t in ts)]
pred=df.apply(detect,axis=1); ref=df["critical_change_annotation"].apply(gold); rows=[]
for label in ["Negation","Dosage / quantity","Timing / frequency","Warnings / severity","Actor"]:
 yt=[int(label in x) for x in ref]; yp=[int(label in x) for x in pred]
 tp=sum(t and p for t,p in zip(yt,yp)); fp=sum((not t) and p for t,p in zip(yt,yp)); fn=sum(t and (not p) for t,p in zip(yt,yp)); tn=sum((not t) and (not p) for t,p in zip(yt,yp))
 pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
 rows.append({"critical_change_type":label,"provisional_positive_examples":sum(yt),"heuristic_predicted_positive":sum(yp),"TP":tp,"FP":fp,"FN":fn,"TN":tn,"precision":pr,"recall":re,"f1":2*pr*re/(pr+re) if pr+re else 0})
os.makedirs(os.path.join(ROOT,"results"),exist_ok=True)
pd.DataFrame(rows).to_csv(os.path.join(ROOT,"results","rq4_critical_change_metrics_exploratory.csv"),index=False)
out=df[["pair_id","language","original_text","comparison_text","critical_change_annotation","validation_status","annotation_method"]].copy()
out["heuristic_detected_types"]=["; ".join(x) if x else "None detected" for x in pred]
out.to_csv(os.path.join(ROOT,"results","rq4_critical_change_predictions_exploratory.csv"),index=False)
print(pd.DataFrame(rows).to_string(index=False))
print("Exploratory only: reference labels are provisional and pending human validation.")
