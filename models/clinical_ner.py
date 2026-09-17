import spacy
from spacy.pipeline import EntityRuler


def create_clinical_ner():
    """
    Create a custom clinical NER pipeline for CareBot.
    """

    nlp = spacy.blank("en")

    ruler = nlp.add_pipe("entity_ruler")

    patterns = [
        # Diagnoses
        {"label": "DIAGNOSIS", "pattern": "pneumonia"},
        {"label": "DIAGNOSIS", "pattern": "hypertension"},
        {"label": "DIAGNOSIS", "pattern": "type 2 diabetes"},
        {"label": "DIAGNOSIS", "pattern": "iron deficiency anemia"},
        {"label": "DIAGNOSIS", "pattern": "anemia"},
        {"label": "DIAGNOSIS", "pattern": "acute bronchitis"},
        {"label": "DIAGNOSIS", "pattern": "migraine"},
        {"label": "DIAGNOSIS", "pattern": "allergic rhinitis"},
        {"label": "DIAGNOSIS", "pattern": "asthma"},
        {"label": "DIAGNOSIS", "pattern": "urinary tract infection"},
        {"label": "DIAGNOSIS", "pattern": "hypothyroidism"},
        {"label": "DIAGNOSIS", "pattern": "gastritis"},
        {"label": "DIAGNOSIS", "pattern": "sinusitis"},
        {"label": "DIAGNOSIS", "pattern": "eczema"},
        {"label": "DIAGNOSIS", "pattern": "hyperlipidemia"},
        {"label": "DIAGNOSIS", "pattern": "depression"},
        {"label": "DIAGNOSIS", "pattern": "osteoarthritis"},
        {"label": "DIAGNOSIS", "pattern": "GERD"},
        {"label": "DIAGNOSIS", "pattern": "bacterial conjunctivitis"},

        # Medications
        {"label": "MEDICATION", "pattern": "amoxicillin"},
        {"label": "MEDICATION", "pattern": "lisinopril"},
        {"label": "MEDICATION", "pattern": "metformin"},
        {"label": "MEDICATION", "pattern": "iron supplementation"},
        {"label": "MEDICATION", "pattern": "azithromycin"},
        {"label": "MEDICATION", "pattern": "sumatriptan"},
        {"label": "MEDICATION", "pattern": "cetirizine"},
        {"label": "MEDICATION", "pattern": "salbutamol"},
        {"label": "MEDICATION", "pattern": "nitrofurantoin"},
        {"label": "MEDICATION", "pattern": "levothyroxine"},
        {"label": "MEDICATION", "pattern": "omeprazole"},
        {"label": "MEDICATION", "pattern": "hydrocortisone"},
        {"label": "MEDICATION", "pattern": "atorvastatin"},
        {"label": "MEDICATION", "pattern": "sertraline"},
        {"label": "MEDICATION", "pattern": "paracetamol"},
        {"label": "MEDICATION", "pattern": "pantoprazole"},
        {"label": "MEDICATION", "pattern": "antibiotic eye drops"},
        {"label": "MEDICATION", "pattern": "doxycycline"},
        {"label": "MEDICATION", "pattern": "amlodipine"},
        {"label": "MEDICATION", "pattern": "oral iron tablets"},
        {"label": "MEDICATION", "pattern": "propranolol"},
        {"label": "MEDICATION", "pattern": "budesonide"},
        {"label": "MEDICATION", "pattern": "ciprofloxacin"},

        # Follow-up periods
        {"label": "FOLLOW_UP", "pattern": "7 days"},
        {"label": "FOLLOW_UP", "pattern": "5 days"},
        {"label": "FOLLOW_UP", "pattern": "10 days"},
        {"label": "FOLLOW_UP", "pattern": "2 weeks"},
        {"label": "FOLLOW_UP", "pattern": "4 weeks"},
        {"label": "FOLLOW_UP", "pattern": "6 weeks"},
        {"label": "FOLLOW_UP", "pattern": "1 month"},
        {"label": "FOLLOW_UP", "pattern": "3 months"},
    ]

    ruler.add_patterns(patterns)

    return nlp


if __name__ == "__main__":
    nlp = create_clinical_ner()

    text = (
        "Patient has pneumonia. "
        "Start amoxicillin 500 mg three times daily. "
        "Follow up in 7 days."
    )

    doc = nlp(text)

    print("Clinical NER Test")
    print("-----------------")

    for ent in doc.ents:
        print(f"{ent.label_}: {ent.text}")