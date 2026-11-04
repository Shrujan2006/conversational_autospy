
from src.analysis_engine import (
    parse_conversation_text,
    analyze_conversation
)

sample = """
Person A: Did you finish the assignment?
Person B: Not yet. I'll do it tonight.
Person A: You said that yesterday.
Person B: Why are you always blaming me?
Person A: Because you never take this seriously.
Person B: I am trying my best. Stop criticizing me.
"""

conversation = parse_conversation_text(sample)

df, summary_df, breakdown = analyze_conversation(conversation)

print("\nCONVERSATION ANALYSIS\n")
print(df[
    ["Turn", "Speaker", "Emotion", "Dialogue Act",
     "Conflict Score", "Conversation State"]
].to_string(index=False))

print("\nBREAKDOWN LOCALIZATION\n")
print(breakdown)

print("\nSPEAKER SUMMARY\n")
print(summary_df.to_string(index=False))

from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(output_dir / "conversation_analysis.csv", index=False)
summary_df.to_csv(output_dir / "speaker_summary.csv", index=False)

print("\nAnalysis completed!")
