
import streamlit as st
import pandas as pd
import plotly.express as px

from src.analysis_engine import (
    parse_conversation_text,
    parse_uploaded_file,
    analyze_conversation
)

st.set_page_config(
    page_title="Conversation Autopsy",
    page_icon="🔍",
    layout="wide"
)

# ---------------- PAGE HEADER ----------------

st.title("🔍 Conversation Autopsy")

st.caption(
    "An Explainable NLP Framework for Detecting, "
    "Localizing, and Analyzing Conversational Breakdown"
)

# ---------------- INPUT SECTION ----------------

with st.expander("＋ Add or Upload Conversation", expanded=True):

    input_method = st.radio(
        "Input method",
        ["Paste Conversation", "Upload File"],
        horizontal=True
    )

    conversation = None

    if input_method == "Paste Conversation":

        conversation_text = st.text_area(
            "Enter conversation",
            height=180,
            placeholder=(
                "Person A: Hello!\n"
                "Person B: Hey, how are you?\n"
                "Person A: I am doing well."
            )
        )

        if st.button("Analyze Conversation", type="primary"):
            try:
                conversation = parse_conversation_text(
                    conversation_text
                )

                with st.spinner("Analyzing conversation..."):
                    df, summary_df, breakdown = analyze_conversation(
                        conversation
                    )

                st.session_state["analysis_df"] = df
                st.session_state["summary_df"] = summary_df
                st.session_state["breakdown"] = breakdown

                st.success("Analysis completed successfully!")

            except Exception as error:
                st.error(str(error))

    else:

        uploaded_file = st.file_uploader(
            "Upload TXT or CSV",
            type=["txt", "csv"]
        )

        if uploaded_file is not None:

            if st.button("Analyze Uploaded File", type="primary"):
                try:
                    conversation = parse_uploaded_file(
                        uploaded_file
                    )

                    with st.spinner("Analyzing conversation..."):
                        df, summary_df, breakdown = analyze_conversation(
                            conversation
                        )

                    st.session_state["analysis_df"] = df
                    st.session_state["summary_df"] = summary_df
                    st.session_state["breakdown"] = breakdown

                    st.success("Analysis completed successfully!")

                except Exception as error:
                    st.error(str(error))


# ---------------- CHECK ANALYSIS ----------------

if "analysis_df" not in st.session_state:

    st.info("Enter or upload a conversation to begin.")
    st.stop()

df = st.session_state["analysis_df"]
summary_df = st.session_state["summary_df"]
breakdown = st.session_state["breakdown"]


# ---------------- HORIZONTAL NAVIGATION ----------------

st.divider()

selected_page = st.radio(
    "Navigation",
    [
        "Overview",
        "Emotion Analysis",
        "Escalation Timeline",
        "State Distribution",
        "Speaker Analysis",
        "Explainability",
        "Full Report"
    ],
    horizontal=True,
    label_visibility="collapsed"
)

st.divider()


# ---------------- OVERVIEW ----------------

if selected_page == "Overview":

    st.header("Conversation Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Turns", len(df))
    col2.metric("Speakers", df["Speaker"].nunique())

    col3.metric(
        "Tension Turns",
        int((df["Conversation State"] == "Tension").sum())
    )

    col4.metric(
        "Escalation Turns",
        int((df["Conversation State"] == "Escalation").sum())
    )

    st.subheader("Breakdown Localization")

    col1, col2 = st.columns(2)

    with col1:
        value = breakdown["First Tension Turn"]

        st.metric(
            "First Tension",
            f"Turn {value}" if value is not None else "Not detected"
        )

    with col2:
        value = breakdown["First Escalation Turn"]

        st.metric(
            "First Escalation",
            f"Turn {value}" if value is not None else "Not detected"
        )

    st.subheader("Conversation Preview")

    st.dataframe(
        df[[
            "Turn", "Speaker", "Message",
            "Emotion", "Conversation State"
        ]],
        use_container_width=True,
        hide_index=True
    )


# ---------------- EMOTION ANALYSIS ----------------

elif selected_page == "Emotion Analysis":

    st.header("Emotional Progression")

    emotion_fig = px.line(
        df,
        x="Turn",
        y="Confidence",
        color="Speaker",
        markers=True,
        hover_data=["Emotion", "Message"],
        title="Emotion Confidence Across Turns"
    )

    emotion_fig.update_layout(
        yaxis_title="Prediction Confidence",
        xaxis_title="Conversation Turn",
        yaxis_range=[0, 1]
    )

    st.plotly_chart(
        emotion_fig,
        use_container_width=True
    )

    st.subheader("Emotion Predictions")

    st.dataframe(
        df[[
            "Turn", "Speaker", "Emotion",
            "Confidence", "Emotion Changed"
        ]],
        use_container_width=True,
        hide_index=True
    )


# ---------------- ESCALATION TIMELINE ----------------

elif selected_page == "Escalation Timeline":

    st.header("Conflict and Escalation Timeline")

    conflict_fig = px.bar(
        df,
        x="Turn",
        y="Conflict Score",
        color="Conversation State",
        hover_data=[
            "Speaker", "Message", "Emotion", "Dialogue Act"
        ],
        title="Conflict Score Across Turns"
    )

    conflict_fig.update_layout(
        xaxis_title="Conversation Turn",
        yaxis_title="Conflict Score"
    )

    st.plotly_chart(
        conflict_fig,
        use_container_width=True
    )

    st.subheader("Turn-wise Conflict States")

    st.dataframe(
        df[[
            "Turn", "Speaker", "Conflict Score",
            "Conversation State", "Explanation"
        ]],
        use_container_width=True,
        hide_index=True
    )


# ---------------- STATE DISTRIBUTION ----------------

elif selected_page == "State Distribution":

    st.header("Conversation State Distribution")

    state_counts = (
        df["Conversation State"]
        .value_counts()
        .rename_axis("State")
        .reset_index(name="Count")
    )

    state_fig = px.pie(
        state_counts,
        names="State",
        values="Count",
        hole=0.4,
        title="Distribution of Conversation States"
    )

    st.plotly_chart(
        state_fig,
        use_container_width=True
    )

    st.subheader("State Counts")

    st.dataframe(
        state_counts,
        use_container_width=True,
        hide_index=True
    )


# ---------------- SPEAKER ANALYSIS ----------------

elif selected_page == "Speaker Analysis":

    st.header("Speaker-wise Analysis")

    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Speaker Emotion Distribution")

    emotion_distribution = (
        df.groupby(["Speaker", "Emotion"])
        .size()
        .reset_index(name="Count")
    )

    fig = px.bar(
        emotion_distribution,
        x="Speaker",
        y="Count",
        color="Emotion",
        barmode="group",
        title="Emotion Distribution by Speaker"
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------- EXPLAINABILITY ----------------

elif selected_page == "Explainability":

    st.header("Turn-wise Explainability")

    for _, row in df.iterrows():

        with st.expander(
            f"Turn {int(row['Turn'])} | "
            f"{row['Speaker']} | "
            f"{row['Conversation State']}"
        ):

            st.write(f"**Message:** {row['Message']}")
            st.write(f"**Emotion:** {row['Emotion']}")
            st.write(f"**Dialogue Act:** {row['Dialogue Act']}")
            st.write(f"**Conflict Score:** {row['Conflict Score']}")
            st.write(f"**Explanation:** {row['Explanation']}")


# ---------------- FULL REPORT ----------------

elif selected_page == "Full Report":

    st.header("Complete Conversation Analysis")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Analysis CSV",
        data=csv_data,
        file_name="conversation_autopsy_analysis.csv",
        mime="text/csv"
    )

    summary_csv = summary_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download Speaker Summary",
        data=summary_csv,
        file_name="speaker_summary.csv",
        mime="text/csv"
    )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Conversation Autopsy | Experimental NLP Analysis | "
    "Conflict detection uses heuristic rules and requires validation."
)
