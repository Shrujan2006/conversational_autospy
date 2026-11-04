
import io
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
SEMANTIC_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def load_models():
    """Load models once and reuse them."""
    emotion_model = pipeline(
        "text-classification",
        model=EMOTION_MODEL_NAME,
        device=-1
    )

    semantic_model = SentenceTransformer(
        SEMANTIC_MODEL_NAME,
        device="cpu"
    )

    return emotion_model, semantic_model


def parse_conversation_text(text):
    """Parse lines such as Person A: Hello."""

    conversation = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()

        if not line:
            continue

        if ":" not in line:
            raise ValueError(
                f"Line {line_number} is invalid. "
                "Use the format: Speaker: Message"
            )

        speaker, message = line.split(":", 1)

        speaker = speaker.strip()
        message = message.strip()

        if not speaker or not message:
            raise ValueError(
                f"Line {line_number} has a missing speaker or message."
            )

        conversation.append((speaker, message))

    if not conversation:
        raise ValueError("No conversation messages were found.")

    return conversation


def parse_uploaded_file(uploaded_file):
    """Read TXT or CSV conversation files."""

    filename = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if filename.endswith(".txt"):
        text = file_bytes.decode("utf-8-sig")
        return parse_conversation_text(text)

    if filename.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))

        # Match column names without case sensitivity.
        columns = {
            str(column).strip().lower(): column
            for column in df.columns
        }

        if "speaker" in columns and "message" in columns:
            speaker_column = columns["speaker"]
            message_column = columns["message"]

            conversation = []

            for _, row in df.iterrows():
                speaker = str(row[speaker_column]).strip()
                message = str(row[message_column]).strip()

                if speaker and message and speaker.lower() != "nan":
                    conversation.append((speaker, message))

            if not conversation:
                raise ValueError("The CSV contains no valid messages.")

            return conversation

        # Also allow a CSV containing one column of dialogue lines.
        if len(df.columns) == 1:
            lines = df.iloc[:, 0].dropna().astype(str).tolist()
            return parse_conversation_text("\n".join(lines))

        raise ValueError(
            "CSV must contain 'Speaker' and 'Message' columns, "
            "or one column containing 'Speaker: Message' lines."
        )

    raise ValueError("Please upload a .txt or .csv file.")


def detect_dialogue_act(message):
    text = message.lower().strip()

    if text.endswith("?"):
        if any(word in text for word in ["why", "how dare", "what is wrong"]):
            return "Challenge"
        return "Question"

    if any(word in text for word in ["sorry", "my apologies", "i apologize"]):
        return "Apology"

    if any(word in text for word in [
        "i agree", "exactly", "you're right",
        "that is true", "absolutely"
    ]):
        return "Agreement"

    if any(word in text for word in [
        "i disagree", "not true", "you're wrong",
        "i don't think so"
    ]):
        return "Disagreement"

    if any(word in text for word in [
        "you always", "you never", "your fault",
        "you are blaming"
    ]):
        return "Accusation"

    if any(word in text for word in [
        "i will", "i'll", "i promise",
        "i am going to", "i'm going to"
    ]):
        return "Commitment"

    if any(word in text for word in [
        "stop criticizing", "not fair",
        "i am tired of", "you said that"
    ]):
        return "Complaint"

    if any(word in text for word in [
        "please", "could you", "can you", "would you"
    ]):
        return "Request"

    if any(word in text for word in [
        "hello", "hi", "good morning", "hey"
    ]):
        return "Greeting"

    return "Statement"


def analyze_conversation(conversation):
    """Run the complete NLP analysis pipeline."""

    emotion_model, semantic_model = load_models()

    messages = [message for speaker, message in conversation]
    speakers = [speaker for speaker, message in conversation]

    # 1. Emotion detection
    predictions = emotion_model(messages, top_k=None)

    records = []
    previous_emotion = {}

    for i, prediction in enumerate(predictions):
        if isinstance(prediction, dict):
            prediction = [prediction]

        best_prediction = max(
            prediction,
            key=lambda item: item["score"]
        )

        emotion = best_prediction["label"]
        confidence = float(best_prediction["score"])

        speaker = speakers[i]
        old_emotion = previous_emotion.get(speaker)

        emotion_changed = (
            None if old_emotion is None
            else old_emotion != emotion
        )

        previous_emotion[speaker] = emotion

        emotion_scores = {
            item["label"]: float(item["score"])
            for item in prediction
        }

        records.append({
            "Turn": i + 1,
            "Speaker": speaker,
            "Message": messages[i],
            "Emotion": emotion,
            "Confidence": confidence,
            "Previous Same-Speaker Emotion": old_emotion,
            "Emotion Changed": emotion_changed,
            "Emotion Scores": json.dumps(emotion_scores)
        })

    df = pd.DataFrame(records)

    # 2. Semantic shift
    embeddings = semantic_model.encode(messages)

    semantic_shifts = [None]

    for i in range(1, len(embeddings)):
        similarity = cosine_similarity(
            [embeddings[i - 1]],
            [embeddings[i]]
        )[0][0]

        semantic_shifts.append(float(1 - similarity))

    df["Semantic Shift"] = semantic_shifts

    # 3. Dialogue-act detection
    df["Dialogue Act"] = [
        detect_dialogue_act(message)
        for message in messages
    ]

    # 4. Context analysis
    previous_counts = []
    speaker_switches = []
    previous_contexts = []

    for i in range(len(conversation)):
        start = max(0, i - 2)
        previous = conversation[start:i]

        previous_speakers = [speaker for speaker, _ in previous]

        switches = sum(
            previous_speakers[j] != previous_speakers[j - 1]
            for j in range(1, len(previous_speakers))
        )

        previous_counts.append(len(previous))
        speaker_switches.append(switches)

        previous_contexts.append(
            " ".join(f"{s}: {m}" for s, m in previous)
        )

    df["Previous Messages Count"] = previous_counts
    df["Previous Speaker Switches"] = speaker_switches
    df["Has Context"] = df["Previous Messages Count"] > 0
    df["Previous Context"] = previous_contexts

    # 5. Conflict and escalation detection
    conflict_scores = []
    conversation_states = []
    evidence_list = []

    negative_emotions = ["anger", "disgust", "sadness"]
    conflict_acts = [
        "Accusation", "Complaint", "Challenge", "Disagreement"
    ]

    for _, row in df.iterrows():
        score = 0
        evidence = []

        if row["Emotion"].lower() in negative_emotions:
            score += 2
            evidence.append("Negative emotion detected")

        if row["Dialogue Act"] in conflict_acts:
            score += 2
            evidence.append("Conflict-related dialogue act")

        if row["Emotion Changed"] is True:
            score += 1
            evidence.append("Speaker emotion changed")

        shift = row["Semantic Shift"]

        if pd.notna(shift) and float(shift) >= 0.65:
            score += 1
            evidence.append("High semantic shift")

        if score >= 5:
            state = "Escalation"
        elif score >= 2:
            state = "Tension"
        else:
            state = "Normal"

        conflict_scores.append(score)
        conversation_states.append(state)
        evidence_list.append(evidence)

    df["Conflict Score"] = conflict_scores
    df["Conversation State"] = conversation_states

    # 6. Breakdown localization
    tension_turns = df[
        df["Conversation State"].isin(["Tension", "Escalation"])
    ]

    escalation_turns = df[
        df["Conversation State"] == "Escalation"
    ]

    first_tension = (
        int(tension_turns.iloc[0]["Turn"])
        if not tension_turns.empty else None
    )

    first_escalation = (
        int(escalation_turns.iloc[0]["Turn"])
        if not escalation_turns.empty else None
    )

    breakdown = {
        "First Tension Turn": first_tension,
        "First Escalation Turn": first_escalation,
        "Breakdown Status": (
            "Tension" if first_tension is not None
            else "No deterioration detected"
        )
    }

    # 7. Explainability
    explanations = []

    for i, row in df.iterrows():
        evidence = []

        if row["Emotion"].lower() in negative_emotions:
            evidence.append(f"Negative emotion: {row['Emotion']}")

        if row["Dialogue Act"] in conflict_acts:
            evidence.append(
                f"Conflict-related dialogue act: {row['Dialogue Act']}"
            )

        if row["Emotion Changed"] is True:
            evidence.append("Speaker's predicted emotion changed")

        if row["Conversation State"] == "Escalation":
            evidence.append(
                "Combined signals crossed escalation threshold"
            )

        if not evidence:
            evidence.append(
                "No strong conflict indicators identified"
            )

        explanations.append("; ".join(evidence))

    df["Explanation"] = explanations

    # Speaker-wise summary
    summary = []

    for speaker, group in df.groupby("Speaker", sort=False):
        emotion_distribution = (
            group["Emotion"].value_counts().to_dict()
        )

        changes = group["Emotion Changed"].eq(True).sum()

        summary.append({
            "Speaker": speaker,
            "Total Messages": len(group),
            "Dominant Emotion": group["Emotion"].mode().iloc[0],
            "Emotion Distribution": json.dumps(emotion_distribution),
            "Emotion Changes": int(changes)
        })

    summary_df = pd.DataFrame(summary)

    return df, summary_df, breakdown
