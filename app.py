import streamlit as st
import pandas as pd
import torch
from transformers import pipeline

# MUST BE FIRST
st.set_page_config(page_title="📞 SDR Calls Summary Dashboard", layout="wide")

# Load summarization model
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="google/flan-t5-large", tokenizer="google/flan-t5-large", framework="pt", device=0 if torch.cuda.is_available() else -1)
    return summarizer

summarizer = load_model()

# Streamlit app starts
st.title("📞 SDR Full Day Summary Dashboard (No OpenAI Needed!)")

uploaded_file = st.file_uploader("Upload your Nooks CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if {'User Name', 'Call Notes'}.issubset(df.columns):
        st.success("✅ File successfully uploaded and processed!")

        grouped = df.groupby('User Name')
        summaries = []

        with st.spinner("🔎 Analyzing all SDRs... please wait"):
            for user_name, group in grouped:
                all_notes = " ".join(group['Call Notes'].dropna().astype(str))

                if not all_notes.strip():
                    continue

                try:
                    # Build custom prompt
                    prompt = (
                        f"You are a professional sales evaluator. Based on the following call notes for SDR {user_name}, "
                        "analyze:\n"
                        "- Overall tone across all calls (Friendly, Neutral, Rushed)\n"
                        "- Frequency of objection handling (High/Medium/Low) with examples\n"
                        "- Meeting success (How many meetings booked)\n"
                        "- Key strengths of the SDR\n"
                        "- Areas to improve\n"
                        "- Provide a 1-paragraph overall evaluation\n\n"
                        f"CALL NOTES:\n{all_notes}"
                    )

                    # flan-t5 expects an input to summarize
                    result = summarizer(prompt, max_length=300, min_length=100, do_sample=False)

                    summary = result[0]['summary_text']

                    summaries.append({
                        'User Name': user_name,
                        'Calls Made': len(group),
                        'Summary': summary
                    })

                except Exception as e:
                    st.error(f"⚠️ Error analyzing {user_name}: {e}")

        # Sort summaries by User Name
        summaries = sorted(summaries, key=lambda x: x['User Name'])

        # Display results
        for sdr_summary in summaries:
            st.markdown(f"### 📋 {sdr_summary['User Name']}")
            st.markdown(f"**Calls Made:** {sdr_summary['Calls Made']}")
            st.markdown("📝 **Full Day Summary:**")
            st.success(sdr_summary['Summary'])

        if summaries:
            report_text = ""
            for s in summaries:
                report_text += f"SDR Name: {s['User Name']}\nCalls Made: {s['Calls Made']}\nFull Day Summary:\n{s['Summary']}\n\n"

            st.download_button(
                label="📥 Download Full Report",
                data=report_text,
                file_name="sdr_full_day_summary.txt",
                mime="text/plain"
            )

    else:
        st.error("ERROR.")
else:
    st.info("📂 Please upload a CSV file to get started.")
