
 Week 1 Research Review

 CareBot Healthcare AI

 Research Paper 1 — Clinical Note Summarization

Title: Development and evaluation of a clinical note summarization system using large language models

Year: 2025

Source: Communications Medicine

Research Focus:

The study investigates the use of large language models for generating clinical summaries from detailed clinical notes. The researchers developed and evaluated a discharge-summary system and considered feedback from physicians and patients when evaluating the generated summaries.

Key Finding:

The study demonstrates the potential of LLMs to interpret clinical information and assist with generating clinical summaries. It also emphasizes the importance of evaluating generated summaries against human assessments.

Relevance to CareBot:

This research supports the LLM component of CareBot. After extracting diagnosis, medication and follow-up information, CareBot can use an LLM to generate a concise clinical summary.



 Research Paper 2 — Medication Extraction and Clinical NER

Title: Detection of Medication Mentions and Medication Change Events in Clinical Notes Using Transformer-Based Models

Authors: Yuting Guo, Yao Ge and Abeed Sarker

Year: 2024

Research Focus:

The study addresses medication extraction, medication-event classification and clinical context classification from clinical notes. The researchers developed a Named Entity Recognition model using BioClinicalBERT and also used dictionary-based fuzzy matching to identify medication mentions.

Key Finding:

The study used 9,012 annotated medication mentions from 500 clinical notes. Its best event-classification model achieved a 0.926 micro-averaged F1-score.

Relevance to CareBot:

This research directly supports the medication-extraction component of CareBot. Our system will use NER to identify medication information from doctor transcripts and convert the extracted information into structured data.

The research also shows that medication information can have important context such as action, negation, temporality, certainty and actor. These can be considered as future improvements to CareBot.



 Research Paper 3 — LLM Clinical Summarization

Title: Adapted large language models can outperform medical experts in clinical text summarization

Authors: Dave Van Veen and colleagues

Year: 2024

Source: Nature Medicine

Research Focus:

The researchers evaluated adapted LLMs across several clinical summarization tasks, including radiology reports, patient questions, progress notes and doctor-patient dialogue. They evaluated the generated summaries using quantitative NLP metrics and a clinical reader study involving physicians.

Key Finding:

In the physician evaluation, summaries from the best-adapted LLMs were considered equivalent to expert summaries in 45% of cases and superior in 36% of cases. The study also identified safety concerns involving fabricated information, showing that clinical LLM outputs need careful evaluation.

Relevance to CareBot:

This research supports the use of an LLM for generating clinical summaries from structured information. It also highlights the importance of evaluating the correctness and completeness of generated summaries rather than relying only on fluent text.



 Research-Based Architecture

Based on the reviewed research, the proposed CareBot architecture is:

Doctor Transcript
↓
Text Preprocessing
↓
Custom Clinical NER
↓
Diagnosis / Medication / Follow-up Extraction
↓
Structured JSON
↓
LLM Summarization
↓
Clinical Summary
↓
Evaluation

The NER component will focus on extracting the three fields required by the internship:

* Diagnosis
* Medication
* Follow-up

The LLM will then use the structured information to generate a concise summary.



 Evaluation Metrics

The following metrics will be used:

 1. Precision

Measures how many extracted entities are correct.

 2. Recall

Measures how many relevant entities were successfully extracted.

 3. F1-score

Combines precision and recall into a single metric.

 4. JSON Validity

Checks whether the system produces valid structured JSON containing the required fields.

 5. Summary Completeness

Checks whether the generated summary contains the important clinical information, especially diagnosis, medication and follow-up.

 6. Summary Correctness

Checks whether the generated summary accurately represents the extracted clinical information.



 Research Conclusion

The three reviewed studies support the main components of the CareBot system.

The first study supports using LLMs for clinical-note summarization. The second directly supports medication extraction using clinical NER. The third supports the use of adapted LLMs for clinical summarization while highlighting the need for safety and evaluation.

Therefore, CareBot will use a hybrid NLP architecture:

Clinical NER → Structured Information → LLM Summarization → Evaluation

The project will use only the provided test data and will not use real patient data.
