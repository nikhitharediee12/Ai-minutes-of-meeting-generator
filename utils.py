import whisper
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


# Transcribe audio using Whisper
def transcribe_audio(file_path):
    model = whisper.load_model("base")  # Options: base, small, medium, large
    result = model.transcribe(file_path)
    return result["text"]


# Summarize transcript using BART
def summarize_text(text, max_len=200):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=max_len, min_length=40, do_sample=False)
    return summary[0]['summary_text']


# Extract action items using simple rule-based keywords
def extract_action_points(text):
    keywords = ['should', 'need to', 'must', 'ensure', 'make sure', 'responsible for', 'to do', 'required to']
    lines = text.split('.')
    actions = [line.strip() for line in lines if any(k in line.lower() for k in keywords)]
    if not actions:
        return "No clear action items detected."
    return "\n".join(f"- {a.strip()}" for a in actions if len(a.strip()) > 10)


# Sentiment analysis using BERT (nlptown)
def analyze_sentiment(text):
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    scores = probs.detach().numpy()[0]

    rating = scores.argmax() + 1  # 1 to 5
    label_map = {
        1: "Very Negative 😠",
        2: "Negative 😕",
        3: "Neutral 😐",
        4: "Positive 🙂",
        5: "Very Positive 😄"
    }
    return label_map[rating]


# Generate PDF report
def generate_pdf(summary, transcript, action_points, sentiment):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    text_object = c.beginText(40, height - 40)
    text_object.setFont("Helvetica", 12)

    def split_text(text, max_chars=90):
        return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

    def write_multiline(title, content):
        text_object.textLine(f"{title}:")
        for line in content.split("\n"):
            for subline in split_text(line):
                text_object.textLine(subline)
        text_object.textLine("")

    write_multiline("📌 Summary", summary)
    write_multiline("✅ Action Items", action_points)
    write_multiline("📈 Sentiment", sentiment)
    write_multiline("📝 Transcription", transcript)

    c.drawText(text_object)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer
