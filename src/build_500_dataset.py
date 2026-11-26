from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
INPUTS = [
    DATA / "dataset_500_candidates_expanded.csv",
    DATA / "dataset_500_candidates.csv",
]
OUTPUT = DATA / "dataset_500_candidates.csv"
TARGET = 500

# Source-backed seed statements. Sources support the general topic/base claim;
# every generated comparison remains a candidate and must be reviewed.
SEEDS = [
    ("handwashing", "Wash your hands with soap and water before preparing food.", "खाना बनाने से पहले साबुन और पानी से हाथ धोएँ।", "CDC - About Handwashing", "https://www.cdc.gov/clean-hands/about/index.html"),
    ("handwashing", "Wash your hands before eating.", "खाना खाने से पहले हाथ धोएँ।", "CDC - About Handwashing", "https://www.cdc.gov/clean-hands/about/index.html"),
    ("handwashing", "Wash your hands after using the toilet.", "शौचालय का उपयोग करने के बाद हाथ धोएँ।", "CDC - About Handwashing", "https://www.cdc.gov/clean-hands/about/index.html"),
    ("handwashing", "Scrub your hands with soap for at least 20 seconds.", "अपने हाथों को साबुन से कम से कम 20 सेकंड तक रगड़ें।", "CDC - About Handwashing", "https://www.cdc.gov/clean-hands/about/index.html"),
    ("handwashing", "Use hand sanitizer containing at least 60% alcohol when soap and water are unavailable.", "जब साबुन और पानी उपलब्ध न हों, तो कम से कम 60% अल्कोहल वाला हैंड सैनिटाइज़र इस्तेमाल करें।", "CDC - About Handwashing", "https://www.cdc.gov/clean-hands/about/index.html"),
    ("antibiotics", "Antibiotics do not treat viral infections.", "एंटीबायोटिक दवाएँ वायरल संक्रमणों का इलाज नहीं करतीं।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("antibiotics", "Take antibiotics exactly as prescribed by your healthcare professional.", "एंटीबायोटिक दवाएँ स्वास्थ्य पेशेवर के निर्देशानुसार ही लें।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("antibiotics", "Do not share your antibiotics with other people.", "अपनी एंटीबायोटिक दवाएँ दूसरों के साथ साझा न करें।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("antibiotics", "Do not save leftover antibiotics for a future illness.", "बची हुई एंटीबायोटिक दवाएँ भविष्य में होने वाली बीमारी के लिए बचाकर न रखें।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("antibiotics", "Ask a healthcare professional if you have questions about antibiotic treatment.", "एंटीबायोटिक उपचार के बारे में प्रश्न हों, तो स्वास्थ्य पेशेवर से पूछें।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("healthy_living", "Eat a balanced and healthy diet.", "संतुलित और स्वस्थ आहार लें।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/ency/article/002393.htm"),
    ("healthy_living", "Regular exercise can support overall health.", "नियमित व्यायाम समग्र स्वास्थ्य में मदद कर सकता है।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/ency/article/002393.htm"),
    ("healthy_living", "Use medicines as recommended by your healthcare provider.", "दवाओं का उपयोग स्वास्थ्य सेवा प्रदाता की सलाह के अनुसार करें।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/ency/article/002393.htm"),
    ("healthy_living", "Avoid smoking to protect your health.", "अपने स्वास्थ्य की रक्षा के लिए धूम्रपान से बचें।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/ency/article/002393.htm"),
    ("healthy_living", "Get enough sleep each night.", "हर रात पर्याप्त नींद लें।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/healthyliving.html"),
    ("healthy_living", "Eat a variety of healthy foods.", "विभिन्न प्रकार के स्वस्थ खाद्य पदार्थ खाएँ।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/healthyliving.html"),
    ("healthy_living", "Keep up to date with recommended vaccines.", "अनुशंसित टीकों के बारे में अद्यतन रहें और समय पर टीकाकरण कराएँ।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/healthyliving.html"),
    ("healthy_living", "Protect your skin from too much sun exposure.", "त्वचा को बहुत अधिक धूप के संपर्क से बचाएँ।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/healthyliving.html"),
    ("healthy_living", "Physical activity, even in small amounts, can be helpful.", "थोड़ी मात्रा में शारीरिक गतिविधि भी लाभकारी हो सकती है।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/healthyliving.html"),
    ("healthy_living", "Work on managing stress and supporting mental well-being.", "तनाव को संभालने और मानसिक स्वास्थ्य को सहारा देने पर काम करें।", "MedlinePlus - Healthy Living", "https://medlineplus.gov/healthyliving.html"),
    ("infection_prevention", "Cover your mouth and nose when coughing or sneezing.", "खाँसते या छींकते समय मुँह और नाक ढकें।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("infection_prevention", "Avoid touching your eyes, nose, and mouth with unwashed hands.", "बिना धुले हाथों से आँखों, नाक और मुँह को छूने से बचें।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("infection_prevention", "Stay home when you are sick, when possible.", "बीमार होने पर, जहाँ संभव हो, घर पर रहें।", "CDC - Antibiotic Do's and Don'ts", "https://www.cdc.gov/antibiotic-use/about/index.html"),
    ("infection_prevention", "Clean your hands after coughing or sneezing.", "खाँसने या छींकने के बाद हाथ साफ करें।", "CDC - About Handwashing", "https://www.cdc.gov/clean-hands/about/index.html"),
    ("infection_prevention", "Wash your hands after touching garbage.", "कचरे को छूने के बाद हाथ धोएँ।", "CDC - About Handwashing", "https://www.cdc.gov/clean-hands/about/index.html"),
]

def candidates(en, hi, lang):
    if lang == "English":
        return [
            (en, "meaning_preserving_candidate"),
            ("Do not " + en[0].lower() + en[1:] if en.startswith(("Wash ", "Scrub ", "Use ", "Eat ", "Get ", "Keep ", "Protect ", "Work ", "Cover ", "Avoid ", "Clean ", "Stay ", "Take ", "Ask ")) else "It is false that " + en[0].lower() + en[1:], "negation_change_candidate"),
            (en.replace("at least 20 seconds", "for 5 seconds").replace("at least 60%", "at least 20%"), "quantity_change_candidate"),
            (en.replace("before", "after", 1), "timing_change_candidate") if "before" in en else (en.replace("after", "before", 1), "timing_change_candidate") if "after" in en else (en.replace("regular", "rare", 1), "frequency_change_candidate"),
            ("It is not necessary to " + en[0].lower() + en[1:], "negation_change_candidate"),
            (en.replace("can", "cannot", 1), "negation_change_candidate") if "can" in en else (en.replace("Do not", "Do", 1), "negation_change_candidate"),
            ("Guidance: " + en, "formatting_variant_candidate"),
            (en + " This is general health information.", "context_added_candidate"),
            ("Remember that " + en[0].lower() + en[1:], "paraphrase_candidate"),
            ("The recommendation is to " + en[0].lower() + en[1:], "paraphrase_candidate"),
            ("Please remember to " + en[0].lower() + en[1:], "paraphrase_candidate"),
            (en.replace("healthcare professional", "friend", 1), "actor_change_candidate"),
        ]
    return [
        (hi, "meaning_preserving_candidate"),
        ("यह सलाह न मानें: " + hi, "negation_change_candidate"),
        (hi.replace("कम से कम 20 सेकंड", "5 सेकंड").replace("कम से कम 60%", "20%"), "quantity_change_candidate"),
        (hi.replace("पहले", "बाद", 1), "timing_change_candidate") if "पहले" in hi else (hi.replace("बाद", "पहले", 1), "timing_change_candidate") if "बाद" in hi else (hi.replace("नियमित", "कभी-कभार", 1), "frequency_change_candidate"),
        ("यह आवश्यक नहीं है कि " + hi, "negation_change_candidate"),
        (hi.replace("कर सकता है", "नहीं कर सकता है", 1), "negation_change_candidate") if "कर सकता है" in hi else (hi.replace("न करें", "करें", 1), "negation_change_candidate"),
        ("कृपया याद रखें: " + hi, "paraphrase_candidate"),
        (hi.replace("स्वास्थ्य पेशेवर", "मित्र", 1), "actor_change_candidate"),
        ("सलाह: " + hi, "formatting_variant_candidate"),
        (hi + " यह सामान्य स्वास्थ्य जानकारी है।", "context_added_candidate"),
        ("ध्यान रखें कि " + hi, "paraphrase_candidate"),
        ("सिफारिश यह है: " + hi, "paraphrase_candidate"),
    ]

# Load existing data if available.
source_file = next((p for p in INPUTS if p.exists()), None)
if source_file is None:
    raise FileNotFoundError("Could not find dataset_500_candidates_expanded.csv or dataset_500_candidates.csv in data/")

existing = pd.read_csv(source_file, dtype=str).fillna("")
columns = [
    "pair_id", "language", "original_text", "comparison_text", "source_name",
    "source_url", "candidate_origin", "candidate_type", "meaning_score_1_to_10",
    "meaning_category", "critical_change", "review_status", "reviewer_notes"
]
for col in columns:
    if col not in existing.columns:
        existing[col] = ""

rows = existing[columns].to_dict("records")
seen = set((r["language"], r["original_text"], r["comparison_text"]) for r in rows)
next_id = 1

for topic, en, hi, source, url in SEEDS:
    for language, original in (("English", en), ("Hindi", hi)):
        for comparison, kind in candidates(en, hi, language):
            if len(rows) >= TARGET:
                break
            key = (language, original, comparison)
            if key in seen or not comparison.strip():
                continue
            rows.append({
                "pair_id": f"GEN{next_id:04d}",
                "language": language,
                "original_text": original,
                "comparison_text": comparison,
                "source_name": source,
                "source_url": url,
                "candidate_origin": "template_generated_pending_review",
                "candidate_type": f"{topic}_{kind}",
                "meaning_score_1_to_10": "",
                "meaning_category": "",
                "critical_change": "",
                "review_status": "pending",
                "reviewer_notes": "Template-generated candidate. Check language, meaning, and source support; not a human label."
            })
            seen.add(key)
            next_id += 1
        if len(rows) >= TARGET:
            break
    if len(rows) >= TARGET:
        break

out = pd.DataFrame(rows, columns=columns)
out = out.drop_duplicates(subset=["language", "original_text", "comparison_text"], keep="first")

# If the templates did not provide 500 genuinely unique pairs, fail honestly rather than duplicate rows.
out.to_csv(OUTPUT, index=False)
print("DATASET BUILD REPORT")
print("====================")
print("Input:", source_file)
print("Rows written:", len(out))
print("Unique pairs:", out.drop_duplicates(["language", "original_text", "comparison_text"]).shape[0])
print("English:", (out["language"] == "English").sum())
print("Hindi:", (out["language"] == "Hindi").sum())
print("New template candidates:", (out["candidate_origin"] == "template_generated_pending_review").sum())
print("Output:", OUTPUT)
if len(out) < TARGET:
    print(f"Target not reached: {TARGET - len(out)} more unique pairs are needed.")
    print("Do not treat template-generated rows as verified or human-labelled data.")
else:
    print("500 rows reached. All generated candidates still require human/source review.")
