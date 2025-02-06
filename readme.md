# YouTube Video Summarizer

This project extracts transcripts from YouTube videos, cleans the text, and generates summaries using two methods: TextRank (Sumy) and a pre-trained deep learning model (BART). The summaries can be used to quickly grasp the key points of a video without watching the entire content.

## Features
- Fetches YouTube video transcripts using `youtube_transcript_api`.
- Cleans transcript by removing timestamps and unwanted characters.
- Summarizes text using:
  - **TextRank Algorithm (Sumy)** – Extractive summarization.
  - **BART Model (Hugging Face Transformers)** – Abstractive summarization for longer texts.
- Saves raw, cleaned, and summarized text in separate files.

## Requirements
Ensure you have Python installed along with the following dependencies:

```bash
pip install youtube-transcript-api nltk sumy transformers torch
```

## How to Use
1. **Set the YouTube Video ID**  
   Replace `video_id` in the script with the actual YouTube video ID you want to summarize.

2. **Run the script**  
   Execute the script to fetch, clean, and summarize the transcript:
   
   ```bash
   python script.py
   ```

3. **Output Files**  
   - `transcript.txt` – Raw transcript from YouTube.
   - `cleaned_data.txt` – Cleaned version of the transcript.
   - `summary_sumy.txt` – Summary using TextRank (Sumy).
   - `summary_gpt.txt` – Summary using BART Model.

## Code Overview
- `get_video_transcript(video_id)`: Fetches transcript from YouTube.
- `clean_transcript(text)`: Cleans transcript by removing timestamps and special characters.
- `summarize_text(text, num_sentences)`: Summarizes text using Sumy's TextRank.
- `summarize_large_text(text, chunk_size)`: Uses BART to summarize long texts in chunks.

## Example
Output for a sample YouTube video:

```
===== Sumy Summary =====
Key sentences extracted from the transcript...

===== GPT Summary =====
AI-generated concise summary of the transcript...
```

## Notes
- BART-based summarization may take longer and require a GPU for faster processing.
- Avoid exceeding API rate limits when fetching transcripts.

## License
This project is open-source and available under the MIT License.

---
Feel free to modify the script to suit your specific use case!

