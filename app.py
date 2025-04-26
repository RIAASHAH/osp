import streamlit as st
import pandas as pd
import openai

# Set your OpenAI API key here
openai.api_key = st.secrets["OPENAI_API_KEY"]
st.set_page_config(page_title="SDR Calls Summary Dashboard", layout="wide")

st.title("📞 SDR Calls Summary Dashboard")

uploaded_file = st.file_uploader("Upload your Nooks CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Ensure the necessary columns are present
    if {'User Name', 'Prospect Name', 'Call Notes'}.issubset(df.columns):
        st.success("File successfully uploaded and processed!")

        # Group by SDR Name and Client
        grouped = df.groupby(['User Name', 'Prospect Name'])

        summaries = []

        with st.spinner("Analyzing calls... this may take a minute!"):
            for (user_name, client_name), group in grouped:
                all_notes = " ".join(group['Call Notes'].dropna().astype(str))

                if all_notes.strip() == "":
                    continue  # Skip if no notes

                prompt = f"""
You are a sales coach. Summarize the behavior of the SDR across multiple calls for the client below.
Client: {client_name}
SDR: {user_name}
Call Notes:
{all_notes}

Please return:
- Tone (Friendly, Neutral, Rushed)
- Key objections raised (if any)
- Meeting success (Yes/No/Partial)
- Client sentiment
- One coaching tip
- One paragraph summary of all calls
"""

                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-4-turbo",
                        messages=[
                            {"role": "system", "content": "You are a professional sales performance analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        max_tokens=600
                    )

                    summary = response['choices'][0]['message']['content'].strip()

                    summaries.append({
                        'User Name': user_name,
                        'Client': client_name,
                        'Calls Made': len(group),
                        'Summary': summary
                    })

                except Exception as e:
                    st.error(f"Error analyzing {user_name} - {client_name}: {e}")

        # Sort summaries by User Name
        summaries = sorted(summaries, key=lambda x: x['User Name'])

        # Display nicely
        for sdr_summary in summaries:
            st.markdown(f"### 📋 {sdr_summary['User Name']}")
            st.markdown(f"**Client:** {sdr_summary['Client']}")
            st.markdown(f"**Calls Made:** {sdr_summary['Calls Made']}")
            st.markdown("📝 **Summary:**")
            st.info(sdr_summary['Summary'])

        # Download report
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
