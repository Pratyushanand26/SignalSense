The Truth Weaver: A Deception Analysis Agent
Project Overview
The Truth Weaver is an automated pipeline designed to solve the "Whispering Shadows Mystery" problem. It takes raw audio testimonies from a subject across multiple sessions, processes them to extract a variety of linguistic and emotional cues, and uses a Large Language Model (LLM) to identify contradictions and synthesize the most likely truth. The final output is a structured JSON file and a comprehensive CSV dataset ready for analysis.

Features
Multi-Modal Analysis: Integrates data from both audio (prosody) and transcribed text (behavioral cues).

Automated Pipeline: A single run.sh script automates the entire process, from audio conversion to final data output.

Speech-to-Text: Uses OpenAI's Whisper model for high-accuracy audio transcription.

Forced Alignment: Leverages Gentle to align words with their exact timestamps, enabling precise per-word prosody analysis.

Agentic Reasoning: Employs Gemini to perform the core contradiction detection and truth synthesis.

Incremental Data Generation: The pipeline appends data for each new audio file to a central CSV, allowing for the building of a large dataset.

Setup and Installation
1. Dependencies
First, ensure you have Python 3.10 or later installed. It is highly recommended to use a virtual environment.

# Create a virtual environment
python -m venv venv
# Activate the environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
# Install required Python packages
pip install -r requirements.txt

2. External Tools
FFmpeg: Ensure FFmpeg is installed and added to your system's PATH. This is required for audio format conversion.

Docker: Docker must be installed and running to use the Gentle forced aligner.

Usage
To run the full pipeline, place all your audio files (.mp3 or .wav) in the audio/ directory. Then, execute the run.sh script from your terminal and provide the path to a single audio file as an argument.

# Example: Process a single audio file
./run.sh audio/subject_session.mp3

The script will process the file and append the results to data/processed_data.csv. Repeat this command for each audio file in your dataset to build a complete CSV.

Project Structure
.
├── audio/                      # Raw audio files go here
├── configs/                    # Configuration files
├── data/                       # All generated data is stored here
│   ├── alignments/             # Forced alignment data
│   ├── annotated/              # Annotated transcripts
│   ├── ml_ready/               # Final 5-line text files
│   ├── prosody/                # Per-word prosody data
│   ├── transcripts_raw/        # Raw ASR output
│   └── processed_data.csv      # The final CSV dataset
├── models/                     # For trained models (if any)
├── scripts/                    # Helper shell scripts (e.g., convert_to_wav.sh)
├── src/                        # Python source code for each pipeline step
│   ├── 01_asr_whisper_simple.py
│   ├── ...
│   ├── 04_prosody.py
│   ├── 08_create_csv.py
├── .env.example                # Example environment variables
├── requirements.txt            # Python dependencies
├── run.sh                      # Main pipeline execution script
└── README.md                   # This file

Workflow
The pipeline operates as an automated workflow, with each step feeding into the next:

Audio Conversion & ASR: run.sh converts the input audio to .wav and uses Whisper for speech-to-text.

VAD & Forced Alignment: The transcript is processed for voice activity and aligned to the audio by Gentle.

Feature Extraction: 04_prosody.py and other scripts extract prosody and disfluency features based on the aligned data.

Data Annotation: Scripts merge all features into a final annotated text file.

ML-Ready File: The annotated text is split into a 5-line, ML-ready file.

Data Aggregation: 08_create_csv.py takes the ml_ready text and the prosody data to append a new row of features to processed_data.csv.

Authors
Pratyush Anand,Aditya Dharpal

Optimal