import streamlit as st
import os
from utils import (
    transcribe_audio,
    summarize_text,
    extract_action_points,
    generate_pdf,
    analyze_sentiment,
    embed_transcript,
    ask_question
)

# Ensure folders exist
os.makedirs("sample_audio", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Custom page styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 36px;
        color: #2E8B57;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🎙️ AI Meeting Minutes Generator</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Transcribe audio, summarize key points, extract actions, analyze tone & ask questions about your meeting.</div>", unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("Upload a meeting recording in mp3, wav, m4a, or mp4 format.")
    st.markdown("Results will include summary, action items, sentiment, and chatbot Q&A.")

# Upload section
uploaded_file = st.file_uploader("📤 Upload your audio file", type=["mp3", "wav", "m4a", "mp4"])

if uploaded_file is not None:
    audio_path = os.path.join("sample_audio", uploaded_file.name)

    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded successfully!")

    with st.spinner("🔄 Transcribing audio..."):
        transcript = transcribe_audio(audio_path)

    st.subheader("🔤 Transcript")
    st.write(transcript)

    with st.spinner("📚 Summarizing transcript..."):
        summary = summarize_text(transcript)

    st.subheader("📌 Summary")
    st.info(summary)

    with st.spinner("🧠 Extracting action items..."):
        action_points = extract_action_points(transcript)

    st.subheader("✅ Action Items")
    st.code(action_points, language="markdown")

    with st.spinner("📊 Analyzing meeting sentiment..."):
        sentiment = analyze_sentiment(transcript)

    st.subheader("📈 Sentiment Analysis")
    if "Positive" in sentiment:
        st.success(sentiment)
    elif "Negative" in sentiment:
        st.error(sentiment)
    else:
        st.warning(sentiment)

    # Save as .txt
    output_file = os.path.join("output", f"{uploaded_file.name}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("TRANSCRIPTION:\n" + transcript + "\n\n")
        f.write("SUMMARY:\n" + summary + "\n\n")
        f.write("ACTION ITEMS:\n" + action_points + "\n\n")
        f.write("SENTIMENT:\n" + sentiment)

    st.success("✅ All processing complete! Results saved to output/ folder.")

    # PDF download
    st.markdown("### 📥 Download PDF Report")
    pdf_file = generate_pdf(summary, transcript, action_points, sentiment)
    st.download_button(
        label="⬇️ Download PDF",
        data=pdf_file,
        file_name="meeting_minutes.pdf",
        mime="application/pdf"
    )

    # Chatbot section
    st.markdown("### 🤖 Ask Questions About the Meeting")
    chunks, embeddings, index, embed_model = embed_transcript(transcript)
    user_question = st.text_input("What would you like to ask?", placeholder="e.g. What are the next steps?")

    if user_question:
        with st.spinner("Thinking..."):
            answer = ask_question(user_question, chunks, embeddings, index, embed_model)
            st.success("Answer:")
            st.write(answer)
