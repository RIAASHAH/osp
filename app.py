import streamlit as st
import pandas as pd
import torch
from transformers import pipeline

# Load summarization model
@st.cache_resource
def load_model():
    summarizer = pipeline("summarization", model="google/flan-t5-large", tokenizer="google/flan-t5-large", framework="pt", device=0 if torch.cuda.is_available() else -1)
    return summarizer

summarizer = load_model()

st.set_page_config(page_title="📞 SDR Calls Summary Dashboard", layout="wide")
st.title("📞 SDR Calls Summary Dashboard (Free, No OpenAI!)")

uploaded_file = st.file_uploader("Upload your Nooks CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if {'User Name', 'Prospect Name', 'Call Notes'}.issubset(df.columns):
        st.success("File successfully uploaded and processed!")

        grouped = df.groupby(['User Name', 'Prospect Name'])
        summaries = []

        with st.spinner("Analyzing calls... please wait"):
            for (user_name, client_name), group in grouped:
                all_notes = " ".join(group['Call Notes'].dropna().astype(str))

                if not all_notes.strip():
                    continue

                try:
                    short_text = all_notes[:3000]  # Limit to avoid memory errors
                    result = summarizer(short_text, max_length=150, min_length=50, do_sample=False)

                    summary = result[0]['summary_text']

                    summaries.append({
                        'User Name': user_name,
                        'Client': client_name,
                        'Calls Made': len(group),
                        'Summary': summary
                    })

                except Exception as e:
                    st.error(f"Error analyzing {user_name} - {client_name}: {e}")

        summaries = sorted(summaries, key=lambda x: x['User Name'])

        for sdr_summary in summaries:
            st.markdown(f"### 📋 {sdr_summary['User Name']}")
            st.markdown(f"**Client:** {sdr_summary['Client']}")
            st.markdown(f"**Calls Made:** {sdr_summary['Calls Made']}")
            st.markdown("📝 **Summary:**")
            st.success(sdr_summary['Summary'])

        if summaries:
            report_text = ""
            for s in summaries:
                report_text += f"SDR Name: {s['User Name']}\nClient: {s['Client']}\nCalls Made: {s['Calls Made']}\nSummary:\n{s['Summary']}\n\n"

            st.download_button(
                label="📥 Download Full Report",
                data=report_text,
                file_name="sdr_call_summary.txt",
                mime="text/plain"
            )
    else:
        st.error("Your file must have 'User Name', 'Prospect Name', and 'Call Notes' columns.")
else:
    st.info("Please upload a CSV file to get started.")
