
def locate_breakdown(df):

    tension_turns = df[
        df["Conversation State"].isin(
            ["Tension", "Escalation"]
        )
    ]

    escalation_turns = df[
        df["Conversation State"] == "Escalation"
    ]

    if tension_turns.empty:
        breakdown_turn = None
        breakdown_state = "No deterioration detected"
    else:
        first = tension_turns.iloc[0]
        breakdown_turn = int(first["Turn"])
        breakdown_state = first["Conversation State"]

    if escalation_turns.empty:
        escalation_turn = None
    else:
        escalation_turn = int(
            escalation_turns.iloc[0]["Turn"]
        )

    return {
        "First Tension Turn": breakdown_turn,
        "First Escalation Turn": escalation_turn,
        "Breakdown Status": breakdown_state
    }


def generate_explanations(df):

    explanations = []

    for _, row in df.iterrows():

        evidence = []

        if str(row["Emotion"]).lower() in [
            "anger", "disgust", "sadness"
        ]:
            evidence.append(
                f"Negative emotion: {row['Emotion']}"
            )

        if row["Dialogue Act"] in [
            "Accusation", "Complaint",
            "Challenge", "Disagreement"
        ]:
            evidence.append(
                f"Conflict-related dialogue act: {row['Dialogue Act']}"
            )

        if row["Emotion Changed"] is True:
            evidence.append(
                "Speaker's predicted emotion changed"
            )

        if row["Conversation State"] == "Escalation":
            evidence.append(
                "Combined signals crossed escalation threshold"
            )

        if not evidence:
            evidence.append(
                "No strong conflict indicators identified"
            )

        explanations.append({
            "Turn": int(row["Turn"]),
            "Explanation": "; ".join(evidence)
        })

    return explanations
