from youtube_transcript_api import YouTubeTranscriptApi
import nltk
import re
import time
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import pipeline

# Ensure required NLTK data is downloaded
nltk.download('punkt')


def get_video_transcript(video_id):
    """Fetches the transcript of a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        return transcript_text
    except Exception as e:
        print("Error fetching transcript:", e)
        return None


def clean_transcript(text):
    """Cleans the transcript by removing timestamps and unwanted characters."""
    text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)  # Remove timestamps ss
    text = re.sub(r'[^a-zA-Z0-9\s.,?!]', '', text)  # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


def summarize_text(text, num_sentences=5):
    """Summarizes text using TextRank algorithm (Sumy)."""
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, num_sentences)
        return ' '.join([str(sentence) for sentence in summary])
    except Exception as e:
        print("Error in Sumy summarization:", e)
        return None


def summarize_large_text(text, chunk_size=1024):
    """Summarizes long text using Facebook's BART model by splitting into chunks."""
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        summaries = []
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)
            summaries.append(summary[0]['summary_text'])
        return " ".join(summaries)
    except Exception as e:
        print("Error in GPT summarization:", e)
        return None


# Example usage
video_id = "f2b0q2qKNNU"  # Replace with actual video ID
transcript_text = get_video_transcript(video_id)

if transcript_text:
    # Save raw transcript
    with open("transcript.txt", "w") as file:
        file.write(transcript_text)

    # Clean transcript
    cleaned_transcript = clean_transcript(transcript_text)
    with open("cleaned_data.txt", "w") as file:
        file.write(cleaned_transcript)

    # Sumy TextRank Summarization
    summary_sumy = summarize_text(cleaned_transcript)
    if summary_sumy:
        print("\n===== Sumy Summary =====\n")
        print(summary_sumy)
        with open("summary_sumy.txt", "w") as file:
            file.write(summary_sumy)

    # GPT-based Summarization (Handles Long Texts)
    summary_gpt = summarize_large_text(cleaned_transcript)
    if summary_gpt:
        print("\n===== GPT Summary =====\n")
        print(summary_gpt)
        with open("summary_gpt.txt", "w") as file:
            file.write(summary_gpt)

    # Add a small delay to prevent API rate limits
    time.sleep(2)
