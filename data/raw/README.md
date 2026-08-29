# CareBot Healthcare AI

## Clinical Note Summarization Pipeline

CareBot Healthcare AI is an AI/ML project designed to process doctor voice transcripts represented as text and extract important clinical information such as diagnosis, medication, and follow-up instructions.

The extracted information will be converted into structured JSON and used to generate a concise clinical summary for patient records.

## Project Objective

The main objectives of CareBot are to:

- Process doctor transcripts as text
- Extract diagnosis information
- Extract medication information
- Extract follow-up instructions
- Generate structured JSON output
- Generate a concise clinical summary
- Evaluate the pipeline using 30 test clinical notes

## Proposed Architecture

Doctor Transcript
↓
Text Preprocessing
↓
Custom spaCy NER
↓
Diagnosis / Medication / Follow-Up Extraction
↓
Structured JSON
↓
LLM Summarization
↓
Clinical Note Summary

## Technologies

- Python 3.11
- spaCy
- Natural Language Processing (NLP)
- Named Entity Recognition (NER)
- Large Language Models (LLMs)
- Pandas
- NumPy
- Scikit-learn
- Git & GitHub

## Evaluation Metrics

The system will be evaluated using:

- Precision
- Recall
- F1-score
- JSON validity
- Entity extraction accuracy
- Summary completeness

## Dataset

The project uses the provided healthcare datasets:

- `aiml_healthcare_symptoms.csv`
- `aiml_healthcare_test_cases.txt`

No real patient data will be used in this project.

## Project Structure

```text
CareBot-Healthcare-AI/
│
├── data/
│   ├── raw/
│   │   ├── aiml_healthcare_symptoms.csv
│   │   └── aiml_healthcare_test_cases.txt
│   └── processed/
│
├── notebooks/
│
├── src/
│   ├── preprocessing.py
│   ├── ner.py
│   ├── summarizer.py
│   └── pipeline.py
│
├── models/
│
├── evaluation/
│   └── evaluate.py
│
├── tests/
│   └── test_pipeline.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Week 1 – Research and Architecture

- [x] Project folder created
- [x] Virtual environment configured
- [x] Required basic ML packages installed
- [x] Provided datasets added
- [ ] Review relevant clinical NLP projects
- [ ] Finalize architecture
- [ ] Finalize evaluation metrics
- [ ] Complete GitHub repository setup

## Future Development

1. Implement text preprocessing.
2. Develop the custom spaCy NER component.
3. Extract diagnosis, medication, and follow-up information.
4. Generate structured JSON.
5. Integrate an LLM for clinical note summarization.
6. Test the pipeline using 30 test notes.
7. Calculate evaluation metrics.
8. Prepare the final GitHub submission.
