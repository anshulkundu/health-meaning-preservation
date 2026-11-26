from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

INPUT = DATA / "dataset_500_candidates.csv"
OUTPUT = DATA / "dataset_500_candidates.csv"

# Each record contains a source-backed health claim and
# comparison candidates requiring human review.
topics = [
    {
        "topic": "handwashing",
        "source": "WHO - Community hand hygiene",
        "url": "https://www.ncbi.nlm.nih.gov/books/NBK619535/",
        "en": (
            "Wash your hands with soap and water before preparing food.",
            [
                "Before preparing food, wash your hands using soap and water.",
                "Wash your hands with water before preparing food.",
                "Do not wash your hands before preparing food.",
                "Wash your hands with soap and water after preparing food.",
                "Use soap and water to wash your hands before making food."
            ]
        ),
        "hi": (
            "खाना बनाने से पहले साबुन और पानी से हाथ धोएँ।",
            [
                "भोजन तैयार करने से पहले साबुन और पानी से अपने हाथ धो लें।",
                "खाना बनाने से पहले केवल पानी से हाथ धोएँ।",
                "खाना बनाने से पहले हाथ न धोएँ।",
                "खाना बनाने के बाद साबुन और पानी से हाथ धोएँ।",
                "भोजन बनाने से पहले साबुन और पानी से हाथ साफ करें।"
            ]
        )
    },
    {
        "topic": "antibiotics",
        "source": "WHO - Patient information on antibiotic resistance",
        "url": "https://cdn.who.int/media/docs/default-source/integrated-health-services-%28ihs%29/ipc/hh-institutional-safety-climate/patient-tips.pdf?ua=1",
        "en": (
            "Antibiotics generally do not work against viral infections.",
            [
                "Antibiotics generally do not work for infections caused by viruses.",
                "Antibiotics always work against viral infections.",
                "Antibiotics generally work against viral infections.",
                "Antibiotics are not generally effective against viruses.",
                "Viral infections should always be treated with antibiotics."
            ]
        ),
        "hi": (
            "एंटीबायोटिक दवाएँ आमतौर पर वायरल संक्रमणों पर असर नहीं करतीं।",
            [
                "एंटीबायोटिक दवाएँ आमतौर पर वायरस से होने वाले संक्रमणों पर असर नहीं करतीं।",
                "एंटीबायोटिक दवाएँ हमेशा वायरल संक्रमणों पर असर करती हैं।",
                "एंटीबायोटिक दवाएँ आमतौर पर वायरल संक्रमणों पर असर करती हैं।",
                "एंटीबायोटिक दवाएँ वायरस के खिलाफ आमतौर पर प्रभावी नहीं होतीं।",
                "वायरल संक्रमणों का इलाज हमेशा एंटीबायोटिक दवाओं से करना चाहिए।"
            ]
        )
    },
    {
        "topic": "physical_activity",
        "source": "WHO - Physical activity",
        "url": "https://www.who.int/health-topics/physical-activity",
        "en": (
            "Regular physical activity can benefit overall health.",
            [
                "Being physically active regularly can support general health.",
                "Regular physical activity has no health benefits.",
                "Physical activity is beneficial only when done once a month.",
                "Regular physical activity can support health and well-being.",
                "Avoid all physical activity to maintain good health."
            ]
        ),
        "hi": (
            "नियमित शारीरिक गतिविधि समग्र स्वास्थ्य के लिए लाभकारी हो सकती है।",
            [
                "नियमित रूप से शारीरिक रूप से सक्रिय रहना सामान्य स्वास्थ्य के लिए लाभकारी हो सकता है।",
                "नियमित शारीरिक गतिविधि से स्वास्थ्य को कोई लाभ नहीं होता।",
                "शारीरिक गतिविधि केवल महीने में एक बार करने पर लाभकारी होती है।",
                "नियमित शारीरिक गतिविधि स्वास्थ्य और कल्याण में मदद कर सकती है।",
                "अच्छे स्वास्थ्य के लिए सभी शारीरिक गतिविधियों से बचें।"
            ]
        )
    },
    {
        "topic": "medicine_safety",
        "source": "WHO - Patient information on antibiotic resistance",
        "url": "https://cdn.who.int/media/docs/default-source/integrated-health-services-%28ihs%29/ipc/hh-institutional-safety-climate/patient-tips.pdf?ua=1",
        "en": (
            "Do not share your prescribed medicines with other people.",
            [
                "Never share medicines prescribed for you with other people.",
                "Share your prescribed medicines with other people.",
                "You should share prescribed medicines whenever someone asks.",
                "Avoid sharing your prescribed medicines with others.",
                "Give your prescription medicines to anyone who needs them."
            ]
        ),
        "hi": (
            "अपने लिए निर्धारित दवाएँ दूसरों के साथ साझा न करें।",
            [
                "अपने लिए लिखी गई दवाएँ किसी दूसरे व्यक्ति को न दें।",
                "अपने लिए निर्धारित दवाएँ दूसरों के साथ साझा करें।",
                "माँगने पर अपनी निर्धारित दवाएँ दूसरों को दे दें।",
                "अपनी निर्धारित दवाएँ दूसरों के साथ साझा करने से बचें।",
                "अपनी पर्चे वाली दवाएँ जरूरतमंद किसी भी व्यक्ति को दे दें।"
            ]
        )
    },
    {
        "topic": "healthy_lifestyle",
        "source": "MedlinePlus - Healthy living",
        "url": "https://medlineplus.gov/ency/article/002393.htm",
        "en": (
            "A balanced and healthy diet supports good health.",
            [
                "Eating a healthy, balanced diet can support overall health.",
                "A balanced diet has no role in maintaining health.",
                "Only one type of food is needed for a healthy diet.",
                "A healthy balanced diet can contribute to well-being.",
                "Avoid eating a variety of foods to stay healthy."
            ]
        ),
        "hi": (
            "संतुलित और स्वस्थ आहार अच्छे स्वास्थ्य में मदद करता है।",
            [
                "स्वस्थ और संतुलित आहार समग्र स्वास्थ्य को बेहतर बनाए रखने में मदद कर सकता है।",
                "स्वास्थ्य बनाए रखने में संतुलित आहार की कोई भूमिका नहीं है।",
                "स्वस्थ आहार के लिए केवल एक प्रकार का भोजन पर्याप्त है।",
                "स्वस्थ और संतुलित आहार कल्याण में योगदान दे सकता है।",
                "स्वस्थ रहने के लिए विभिन्न प्रकार के खाद्य पदार्थ खाने से बचें।"
            ]
        )
    }
]

new_rows = []

for topic in topics:
    for language in ("English", "Hindi"):
        key = "en" if language == "English" else "hi"
        original, comparisons = topic[key]

        for index, comparison in enumerate(comparisons, start=1):
            new_rows.append({
                "pair_id": "",
                "language": language,
                "original_text": original,
                "comparison_text": comparison,
                "source_name": topic["source"],
                "source_url": topic["url"],
                "candidate_origin": "research_candidate_pending_review",
                "candidate_type": f"{topic['topic']}_variant_{index}",
                "meaning_score_1_to_10": "",
                "meaning_category": "",
                "critical_change": "",
                "review_status": "pending",
                "reviewer_notes": (
                    "Check source support and sentence meaning. "
                    "Hindi requires fluent-speaker review."
                )
            })

new_df = pd.DataFrame(new_rows)

if INPUT.exists():
    old_df = pd.read_csv(INPUT, dtype=str).fillna("")
else:
    raise FileNotFoundError(
        f"Existing dataset not found: {INPUT}"
    )

# Ensure all expected columns exist
all_columns = list(dict.fromkeys(
    list(old_df.columns) + list(new_df.columns)
))

for col in all_columns:
    if col not in old_df.columns:
        old_df[col] = ""
    if col not in new_df.columns:
        new_df[col] = ""

# Assign unique IDs to newly added rows
used_ids = set(old_df["pair_id"].astype(str))
next_number = 1

for i in new_df.index:
    while f"NEW{next_number:04d}" in used_ids:
        next_number += 1

    new_id = f"NEW{next_number:04d}"
    new_df.loc[i, "pair_id"] = new_id
    used_ids.add(new_id)
    next_number += 1

combined = pd.concat(
    [old_df[all_columns], new_df[all_columns]],
    ignore_index=True
)

# Remove exact duplicates only; preserve meaning-changing variants
combined = combined.drop_duplicates(
    subset=["language", "original_text", "comparison_text"],
    keep="first"
)

combined.to_csv(OUTPUT, index=False)

print("CANDIDATE EXPANSION COMPLETE")
print("----------------------------")
print("New candidates created:", len(new_df))
print("Total unique pairs:", len(combined))
print(
    "English:",
    (combined["language"] == "English").sum()
)
print(
    "Hindi:",
    (combined["language"] == "Hindi").sum()
)
print("Output:", OUTPUT)
print("All new candidates require review.")
print("Human scores remain blank.")