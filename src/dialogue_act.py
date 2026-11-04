# detect dialogue acts in a conversation
import re


def detect_dialogue_act(message):

    text = message.lower().strip()

    # Question
    if text.endswith("?"):
        if any(word in text for word in [
            "why", "how dare", "what is wrong"
        ]):
            return "Challenge"

        return "Question"

    # Apology
    if any(word in text for word in [
        "sorry", "my apologies", "i apologize"
    ]):
        return "Apology"

    # Agreement
    if any(word in text for word in [
        "i agree", "exactly", "you're right",
        "that is true", "absolutely"
    ]):
        return "Agreement"

    # Disagreement
    if any(word in text for word in [
        "i disagree", "not true", "you're wrong",
        "i don't think so"
    ]):
        return "Disagreement"

    # Accusation
    if any(word in text for word in [
        "you always", "you never",
        "your fault", "you are blaming"
    ]):
        return "Accusation"

    # Commitment
    if any(word in text for word in [
        "i will", "i'll", "i promise",
        "i am going to", "i'm going to"
    ]):
        return "Commitment"

    # Complaint
    if any(word in text for word in [
        "stop criticizing", "not fair",
        "i am tired of", "you said that"
    ]):
        return "Complaint"

    # Request
    if any(word in text for word in [
        "please", "could you", "can you",
        "would you"
    ]):
        return "Request"

    # Greeting
    if any(word in text for word in [
        "hello", "hi", "good morning", "hey"
    ]):
        return "Greeting"

    return "Statement"


def analyze_dialogue_acts(messages):

    results = []

    for turn, (speaker, message) in enumerate(messages, start=1):

        dialogue_act = detect_dialogue_act(message)

        results.append({
            "Turn": turn,
            "Speaker": speaker,
            "Message": message,
            "Dialogue Act": dialogue_act
        })

    return results
