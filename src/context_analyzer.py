
def analyze_context(conversation, window=2):

    results = []

    for i, (speaker, message) in enumerate(conversation):

        # Get previous messages
        start = max(0, i - window)

        previous_messages = conversation[start:i]

        # Count previous messages from other speakers
        previous_speakers = [
            s for s, m in previous_messages
        ]

        previous_messages_count = len(previous_messages)

        speaker_switches = sum(
            1 for j in range(1, len(previous_speakers))
            if previous_speakers[j] != previous_speakers[j - 1]
        )

        # Create conversational context
        context_text = " ".join(
            f"{s}: {m}" for s, m in previous_messages
        )

        # Identify whether this is the first message
        has_context = len(previous_messages) > 0

        results.append({

            "Turn": i + 1,
            "Previous Messages Count": previous_messages_count,
            "Previous Speaker Switches": speaker_switches,
            "Has Context": has_context,
            "Previous Context": context_text,
            "Current Speaker": speaker,
            "Current Message": message

        })

    return results
