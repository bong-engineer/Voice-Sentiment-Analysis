import nltk
nltk.download('vader_lexicon')
!pip install -q gradio
!pip install -q openai-whisper
import gradio as gr
import whisper as ow
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the Whisper model
model = ow.load_model("large")

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

def transcribe_and_analyze(audio):
    # Transcribe the audio using Whisper
    transcription = model.transcribe(audio)["text"]

    # Analyze sentiment using VADER
    sentiment_scores = sia.polarity_scores(transcription)
    compound_score = sentiment_scores["compound"]

    # Map sentiment score to emojis
    if compound_score >= 0.05:
        sentiment_emoji = "😄"  # Positive
    elif compound_score <= -0.05:
        sentiment_emoji = "😞"  # Negative
    else:
        sentiment_emoji = "😐"  # Neutral

    return f"Transcription: {transcription}\nSentiment: {sentiment_emoji}"

# Create the Gradio interface
gr.Interface(
    fn=transcribe_and_analyze,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    live=True
).launch()
