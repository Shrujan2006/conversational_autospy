# Conversational Autopsy

## Overview
Conversational Autopsy is an experimental, explainable NLP pipeline for analyzing text conversations between identified speakers. It tracks turn order, estimates message-level emotion, assigns lightweight rule-based dialogue acts, summarizes local context, and highlights turns with heuristic tension or escalation signals.

It is a research prototype, not a clinical, legal, or definitive conflict detector. Emotion predictions and rule-based signals can be wrong, especially with sarcasm, cultural variation, short messages, and missing context.

## Implemented components
- **Input:** pasted text, UTF-8 TXT, or CSV with `Speaker` and `Message` columns.
- **Emotion classification:** `j-hartmann/emotion-english-distilroberta-base`.
- **Sentence embeddings:** `sentence-transformers/all-MiniLM-L6-v2`; adjacent-turn semantic shift is descriptive and is not used alone as conflict evidence.
- **Dialogue acts:** transparent keyword/rule baseline.
- **Context:** preceding two turns and speaker-switch count.
- **Conflict indicators:** weighted heuristic using emotion, conflict-oriented dialogue acts, same-speaker emotion change, and recent reciprocal conflict-oriented turns.
- **Outputs:** turn-level analysis, first tension/escalation localization, speaker summaries, explanations, and Streamlit visualizations.

## Method limitations
The conflict score is a manually designed heuristic, not a trained classifier. It must not be presented as validated accuracy. Emotion classification is not equivalent to detecting intent or interpersonal conflict. Dialogue-act rules are language- and phrase-dependent. The current implementation operates on text and does not perform audio transcription, speaker diarization, or psychological diagnosis.

## Installation
```bash
git clone https://github.com/Shrujan2006/conversational_autospy.git
cd conversational_autospy
python -m venv .venv
```

Activate the environment (Windows PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The first analysis may download pretrained model weights and requires an internet connection. The current model inference configuration uses CPU.

## Run tests
```bash
python -m unittest discover -s tests -v
```

## Research and evaluation
Report empirical metrics only after evaluating against a documented, appropriately consented and annotated dataset. Record annotation guidelines, annotator agreement, class distribution, train/test separation (if applicable), per-class precision/recall/F1, macro-F1, and confusion matrix. Do not treat the heuristic score as ground-truth labels.

## Privacy
Use anonymized data and obtain appropriate consent before processing private conversations. Avoid uploading sensitive personal information.
