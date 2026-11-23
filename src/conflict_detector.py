"""Transparent, rule-based turn-level conflict signal aggregation."""

import pandas as pd

STRONG_CONFLICT_ACTS = {"Accusation", "Complaint", "Challenge"}
NEGATIVE_EMOTIONS = {"anger", "disgust", "sadness", "fear"}


def detect_conflict(row, reciprocal_conflict=False):
    """Return a heuristic score; this is not a trained conflict classifier."""
    score, evidence = 0, []
    emotion = str(row.get("Emotion", "")).lower()
    act = str(row.get("Dialogue Act", ""))
    changed = row.get("Emotion Changed", False)

    if emotion in NEGATIVE_EMOTIONS:
        score += 1
        evidence.append(f"Negative emotional signal ({emotion})")
    if act in STRONG_CONFLICT_ACTS:
        score += 2
        evidence.append(f"Conflict-oriented dialogue act ({act})")
    if act == "Disagreement":
        evidence.append("Disagreement detected; not counted as conflict by itself")
    if pd.notna(changed) and bool(changed):
        score += 1
        evidence.append("Same speaker's predicted emotion changed")
    if reciprocal_conflict:
        score += 1
        evidence.append("Recent reciprocal conflict-oriented turn")

    state = "Escalation" if score >= 4 else "Tension" if score >= 2 else "Normal"
    return {
        "Conflict Score": score,
        "Conversation State": state,
        "Evidence": "; ".join(evidence) if evidence else "No strong conflict indicators"
    }


def analyze_conflicts(df, lookback=3):
    """Add a bounded reciprocal-conflict signal from preceding turns."""
    results = []
    strong_acts = STRONG_CONFLICT_ACTS
    for index, row in df.iterrows():
        start = max(0, index - lookback)
        recent = df.iloc[start:index]
        reciprocal = bool(
            not recent.empty
            and any(
                prior["Speaker"] != row["Speaker"]
                and prior["Dialogue Act"] in strong_acts
                for _, prior in recent.iterrows()
            )
            and row["Dialogue Act"] in strong_acts
        )
        results.append(detect_conflict(row, reciprocal_conflict=reciprocal))
    return pd.DataFrame(results, index=df.index)
