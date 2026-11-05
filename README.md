# AI-Powered Conversation Analysis and Escalation Detection

## Overview
An AI-powered conversation analysis application designed to understand conversational flow, identify individual speakers, classify conversational states, and detect potential tension or escalation during interactions.

The application supports pasted conversations and uploaded TXT/CSV files.

## Key Features
- **Speaker Identification:** Tracks individual speakers throughout a conversation.
- **Emotion Analysis:** Identifies emotional signals in conversational text.
- **Dialogue Analysis:** Examines conversational intent and dialogue patterns.
- **Context Analysis:** Evaluates conversational flow and context.
- **Conflict Detection:** Identifies potential tension and escalation.
- **Conversation State Distribution:** Visualizes the distribution of conversational states.
- **Escalation Timeline:** Displays changes in conversational tension over time.
- **Explainability:** Provides insights into the analysis results.
- **Interactive Dashboard:** Streamlit-based interface with separate navigation sections.

## Technology Stack
- Python
- Streamlit
- Pandas
- Plotly
- Hugging Face Transformers
- Sentence Transformers
- Scikit-learn
- PyTorch

## AI Models
| Model | Purpose |
|---|---|
| `j-hartmann/emotion-english-distilroberta-base` | Emotion classification |
| `sentence-transformers/all-MiniLM-L6-v2` | Text embeddings and semantic similarity |

## Architecture
1. Input conversation through text or file upload.
2. Process and organize speaker messages.
3. Perform emotion and dialogue analysis.
4. Analyze context and conversational patterns.
5. Detect potential conflicts and escalation.
6. Generate conversation states and analytical results.
7. Visualize the results through the dashboard.

## Installation

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd conversation-analysis

python -m venv venv
```

Activate the environment:

Windows:
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Status
Working prototype. Detection results are based on the implemented models and analysis rules and require further validation.

## Privacy
Use anonymized conversation data for testing. Do not upload private conversations or sensitive personal information.

## License
See the LICENSE file.
