# YouTube Audio Transcription Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A minimal, production-grade Python pipeline for automated audio acquisition, preprocessing, and transcription of YouTube content. This project is designed for robustness, idempotency, and high-quality speech data extraction.

## 🚀 Features

- **Automated Acquisition**: Reliable downloads via `yt-dlp` with automatic retries.
- **Audio Standardization**: Seamless preprocessing using `FFmpeg` with automatic retries.
- **Local Transcription**: High-accuracy speech-to-text using OpenAI's `Whisper` models with automatic retries.
- **Structured Output**: Machine-readable JSON including timestamped segments and metadata.
- **Batch Processing**: Orchestrate processing for hundreds of URLs via text file input.
- **Production Ready**: Comprehensive logging, error handling, and idempotency checks.

## 🛠 Prerequisites

### System Dependencies
Ensure the following are installed and available in your `PATH`:
- **FFmpeg**: [Download](https://ffmpeg.org/download.html)
- **yt-dlp**: [Installation Guide](https://github.com/yt-dlp/yt-dlp#installation)

### Python Environment
```bash
# Clone the repository
git clone https://github.com/yourusername/yturl.git
cd yturl

# Install dependencies
pip install -r requirements.txt
```

## 📖 Usage

### Single Video Processing
```bash
python main.py --url "https://www.youtube.com/watch?v=..."
```

### Batch Processing
Create a `urls.txt` with one URL per line:
```bash
python main.py --input urls.txt --model small
```

### Options
- `--url`: Process a single YouTube URL.
- `--input`: Process a list of URLs from a text file.
- `--model`: Whisper model size (`tiny`, `base`, `small`, `medium`, `large`). Default is `base`.
- `--dry-run`: Preview the plan without executing downloads or transcription.

## 📂 Project Structure

```text
yturl/
├── data/
│   ├── raw/                # Original downloaded media
│   └── processed_audio/    # Standardized 16kHz WAV files
├── outputs/
│   └── json/               # Final structured transcripts
├── logs/                   # Pipeline execution logs
├── main.py                 # CLI Entry point & Orchestration
├── pipeline.py             # Core pipeline logic
└── requirements.txt        # Python dependencies
```

## 🏗 Pipeline Architecture

1. **Check**: Validate system dependencies (`ffmpeg`, `yt-dlp`).
2. **Download**: `yt-dlp` fetches audio; retries up to 3 times on failure.
3. **Process**: `FFmpeg` converts media to standardized WAV (Mono, 16kHz).
4. **Transcribe**: `Whisper` generates full text and timestamped segments.
5. **Output**: Metadata and transcript are serialized to a structured JSON file.

## ⚖️ License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
