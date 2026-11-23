"""Lightweight, transparent rule-based dialogue-act baseline."""


def detect_dialogue_act(message):
    text = str(message).lower().strip()
    if text.endswith("?"):
        if any(term in text for term in ("why", "how dare", "what is wrong")):
            return "Challenge"
        return "Question"
    if any(term in text for term in ("sorry", "my apologies", "i apologize")):
        return "Apology"
    if any(term in text for term in ("i agree", "exactly", "you're right", "that is true", "absolutely")):
        return "Agreement"
    if any(term in text for term in ("i disagree", "not true", "you're wrong", "i don't think so")):
        return "Disagreement"
    if any(term in text for term in ("you always", "you never", "your fault", "you are blaming")):
        return "Accusation"
    if any(term in text for term in ("i will", "i'll", "i promise", "i am going to", "i'm going to")):
        return "Commitment"
    if any(term in text for term in ("stop criticizing", "not fair", "i am tired of", "you said that")):
        return "Complaint"
    if any(term in text for term in ("please", "could you", "can you", "would you")):
        return "Request"
    if any(term in text for term in ("hello", "hi", "good morning", "hey")):
        return "Greeting"
    return "Statement"


def analyze_dialogue_acts(messages):
    return [
        {"Turn": turn, "Speaker": speaker, "Message": message,
         "Dialogue Act": detect_dialogue_act(message)}
        for turn, (speaker, message) in enumerate(messages, start=1)
    ]
