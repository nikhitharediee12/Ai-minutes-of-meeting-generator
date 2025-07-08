# 🎙️ AI Meeting Minutes Generator

An AI-powered tool that transcribes your meeting audio, summarizes key points, extracts action items, analyzes sentiment, and generates a professional PDF report — all in one click.

> 🔥 Built with Whisper, BART, BERT, and Streamlit — works locally with no external API needed.

---

## 🚀 Live Demo
👉 (https://nikhitharediee12-ai-minutes-of-meeting-generator.streamlit.app)
## 🚀 Features

- 🎤 **Audio Transcription** — powered by [Whisper](https://github.com/openai/whisper)
- 📌 **Summary Generation** — using BART via Hugging Face Transformers
- ✅ **Action Item Extraction** — rule-based NLP logic
- 📈 **Sentiment Analysis** — with multilingual BERT model
- 🧾 **PDF Report Generation** — creates a downloadable summary
- 🖼️ **Streamlit UI** — upload file, view results, and download report

---

## 🛠️ Tech Stack

| Layer           | Tool/Library                                      |
|-----------------|--------------------------------------------------|
| AI/ML Models    | OpenAI Whisper, BART (`facebook/bart-large-cnn`), BERT (`nlptown/bert-base-multilingual-uncased-sentiment`) |
| NLP             | Transformers, Torch                              |
| Frontend / UI   | Streamlit                                        |
| Report Output   | ReportLab (PDF generation)                       |
| Deployment Ready| GitHub + Streamlit Cloud                         |
| Language        | Python                                           |

---

## 🧪 Sample Use Case

> Upload your `.mp3` or `.wav` meeting file and instantly get:
> - A clean summary  
> - Action items  
> - Sentiment of the conversation  
> - Downloadable PDF report  

---

## 📦 Installation

```bash
git clone https://github.com/nikhitharediee12/Ai-minutes-of-meeting-generator
cd Ai-minutes-of-meeting-generator
python -m venv venv
venv\Scripts\activate     # On Windows
pip install -r requirements.txt
streamlit run app.py
