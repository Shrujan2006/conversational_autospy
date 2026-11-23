import pandas as pd

def locate_breakdown(df):
    """Locate the first tension and escalation turns without conflating them."""
    tension = df[df["Conversation State"].isin(["Tension", "Escalation"])]
    escalation = df[df["Conversation State"] == "Escalation"]
    first_tension = int(tension.iloc[0]["Turn"]) if not tension.empty else None
    first_escalation = int(escalation.iloc[0]["Turn"]) if not escalation.empty else None
    if first_escalation is not None:
        status = "Escalation"
    elif first_tension is not None:
        status = "Tension"
    else:
        status = "No deterioration detected"
    return {
        "First Tension Turn": first_tension,
        "First Escalation Turn": first_escalation,
        "Breakdown Status": status
    }


def generate_explanations(df):
    """Use the same evidence recorded by the conflict scoring component."""
    output = []
    for _, row in df.iterrows():
        evidence = row.get("Evidence", "No strong conflict indicators")
        if pd.isna(evidence) or not str(evidence).strip():
            evidence = "No strong conflict indicators"
        output.append({"Turn": int(row["Turn"]), "Explanation": str(evidence)})
    return output
