
def detect_conflict(row):

    score = 0
    evidence = []

    # 1. Emotion signals
    negative_emotions = ["anger", "disgust", "sadness"]

    if str(row["Emotion"]).lower() in negative_emotions:
        score += 2
        evidence.append("Negative emotion detected")

    # 2. Dialogue act signals
    conflict_acts = [
        "accusation",
        "complaint",
        "challenge",
        "disagreement"
    ]

    if str(row["Dialogue Act"]).lower() in conflict_acts:
        score += 2
        evidence.append("Conflict-related dialogue act")

    # 3. Emotional change
    if row["Emotion Changed"] is True:
        score += 1
        evidence.append("Speaker emotion changed")

    # 4. Semantic shift
    shift = row["Semantic Shift"]

    if shift is not None and str(shift) != "nan":
        if float(shift) >= 0.65:
            score += 1
            evidence.append("High semantic shift")

    # 5. Determine conversation state
    if score >= 5:
        state = "Escalation"

    elif score >= 2:
        state = "Tension"

    else:
        state = "Normal"

    return {
        "Conflict Score": score,
        "Conversation State": state,
        "Evidence": "; ".join(evidence) if evidence else "No strong signals"
    }


def analyze_conflicts(df):

    conflict_results = df.apply(
        detect_conflict,
        axis=1,
        result_type="expand"
    )

    return conflict_results
