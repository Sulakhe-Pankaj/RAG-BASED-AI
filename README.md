# RAG-Based AI Teaching Assistant (Video to LLM Pipeline)

This project implements a Retrieval-Augmented Generation (RAG) system that converts video-based learning content into searchable vector embeddings and uses them to answer user queries with a Large Language Model.

This is a sequential pipeline. Break the order and the system fails.

---

## Overview

The workflow converts raw videos into text, transforms the text into embeddings, stores them for retrieval, and dynamically builds prompts for an LLM.

Pipeline:
Videos → Audio → Transcripts → Embeddings → Retrieval → LLM Response


---

## Requirements

- Python 3.8+
- ffmpeg
- Ollama
- Required Python libraries:
  - pandas
  - numpy
  - joblib
  - sentence-transformers
  - requests or ollama-python (depending on implementation)

Install dependencies before running anything.

---

## Ollama Installation and Model Setup

This project uses **Ollama** to run LLMs locally. Cloud APIs are not required.

### Step 1: Install Ollama

#### Linux / macOS

```bash
curl -fsSL https://ollama.com/install.sh | sh
```
---

## Usage Instructions

### Step 1: Add Video Files

Place all source video files inside the `videos/` directory.

Do not mix unrelated files here.

---

### Step 2: Convert Videos to MP3 

Run:

```bash
python video_to_mp3.py
```
### Step 3: Convert MP3 to JSON
```
python mp3_to_json.py
```
### Step 4: Generate Vector Embeddings
```
python preprocess_json.py
```
### Step 5: Query the System
```
python process_incoming.py
```

## Notes

Steps must be executed in order.

API keys for transcription and LLM access must be configured.

Large video files will increase processing time.

Re-run preprocessing only when source data changes.

## Intended Use

This project is designed for:

Building course assistants

Searching lecture content

Question-answering over video material




