import io
import json
from functools import lru_cache

import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.dialogue_act import detect_dialogue_act
from src.context_analyzer import analyze_context
from src.conflict_detector import analyze_conflicts
from src.explainability import locate_breakdown, generate_explanations

EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_models():
    """Load the two pretrained NLP models once per process."""
    emotion_model = pipeline("text-classification", model=EMOTION_MODEL_NAME, device=-1)
    semantic_model = SentenceTransformer(SEMANTIC_MODEL_NAME, device="cpu")
    return emotion_model, semantic_model


def parse_conversation_text(text):
    """Parse one 'Speaker: message' record per non-empty line."""
    if not isinstance(text, str):
        raise ValueError("Conversation input must be text.")
    conversation = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Line {line_number} is invalid. Use 'Speaker: Message'.")
        speaker, message = line.split(":", 1)
        speaker, message = speaker.strip(), message.strip()
        if not speaker or not message:
            raise ValueError(f"Line {line_number} has a missing speaker or message.")
        conversation.append((speaker, message))
    if not conversation:
        raise ValueError("No conversation messages were found.")
    return conversation


def parse_uploaded_file(uploaded_file):
    """Read TXT or CSV uploads; missing CSV cells are skipped safely."""
    filename = uploaded_file.name.lower()
    content = uploaded_file.getvalue()
    if filename.endswith(".txt"):
        try:
            return parse_conversation_text(content.decode("utf-8-sig"))
        except UnicodeDecodeError as exc:
            raise ValueError("TXT file must use UTF-8 encoding.") from exc
    if not filename.endswith(".csv"):
        raise ValueError("Please upload a .txt or .csv file.")
    try:
        frame = pd.read_csv(io.BytesIO(content))
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc
    columns = {str(c).strip().lower(): c for c in frame.columns}
    if "speaker" in columns and "message" in columns:
        result = []
        for _, row in frame.iterrows():
            speaker, message = row[columns["speaker"]], row[columns["message"]]
            if pd.isna(speaker) or pd.isna(message):
                continue
            speaker, message = str(speaker).strip(), str(message).strip()
            if speaker and message:
                result.append((speaker, message))
        if not result:
            raise ValueError("The CSV contains no valid messages.")
        return result
    if len(frame.columns) == 1:
        lines = frame.iloc[:, 0].dropna().astype(str).tolist()
        return parse_conversation_text("\n".join(lines))
    raise ValueError("CSV must contain Speaker and Message columns, or one column of 'Speaker: Message' lines.")


def analyze_conversation(conversation):
    """Analyze emotions, dialogue acts, context, and heuristic conflict signals."""
    if not conversation:
        raise ValueError("Conversation must contain at least one message.")
    normalized = []
    for index, item in enumerate(conversation, start=1):
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError(f"Conversation item {index} must be a (speaker, message) pair.")
        speaker, message = str(item[0]).strip(), str(item[1]).strip()
        if not speaker or not message:
            raise ValueError(f"Conversation item {index} has an empty speaker or message.")
        normalized.append((speaker, message))

    emotion_model, semantic_model = load_models()
    speakers = [s for s, _ in normalized]
    messages = [m for _, m in normalized]
    predictions = emotion_model(messages, top_k=None)
    records, previous_emotion = [], {}

    for index, prediction in enumerate(predictions):
        if isinstance(prediction, dict):
            prediction = [prediction]
        best = max(prediction, key=lambda item: item["score"])
        speaker = speakers[index]
        prior = previous_emotion.get(speaker)
        emotion = str(best["label"]).lower()
        changed = None if prior is None else prior != emotion
        previous_emotion[speaker] = emotion
        scores = {str(p["label"]): float(p["score"]) for p in prediction}
        records.append({
            "Turn": index + 1, "Speaker": speaker, "Message": messages[index],
            "Emotion": emotion, "Confidence": float(best["score"]),
            "Previous Same-Speaker Emotion": prior, "Emotion Changed": changed,
            "Emotion Scores": json.dumps(scores)
        })
    df = pd.DataFrame(records)

    # Semantic shift is retained as a descriptive feature only. Since adjacent
    # turns often belong to different speakers, it is not treated as conflict evidence.
    embeddings = semantic_model.encode(messages)
    shifts = [None]
    for index in range(1, len(embeddings)):
        similarity = cosine_similarity([embeddings[index - 1]], [embeddings[index]])[0][0]
        shifts.append(float(1.0 - similarity))
    df["Semantic Shift"] = shifts
    df["Dialogue Act"] = [detect_dialogue_act(message) for message in messages]

    context_rows = analyze_context(normalized, window=2)
    context_df = pd.DataFrame(context_rows)
    for column in ["Previous Messages Count", "Previous Speaker Switches", "Has Context", "Previous Context"]:
        df[column] = context_df[column].values

    conflict_df = analyze_conflicts(df)
    for column in conflict_df.columns:
        df[column] = conflict_df[column].values

    breakdown = locate_breakdown(df)
    explanation_rows = generate_explanations(df)
    explanation_map = {row["Turn"]: row["Explanation"] for row in explanation_rows}
    df["Explanation"] = df["Turn"].map(explanation_map)

    summaries = []
    for speaker, group in df.groupby("Speaker", sort=False):
        distribution = group["Emotion"].value_counts().to_dict()
        summaries.append({
            "Speaker": speaker,
            "Total Messages": len(group),
            "Dominant Emotion": group["Emotion"].mode().iloc[0],
            "Emotion Distribution": json.dumps(distribution),
            "Emotion Changes": int(group["Emotion Changed"].eq(True).sum())
        })
    return df, pd.DataFrame(summaries), breakdown
