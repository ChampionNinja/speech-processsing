import os
import json
import logging
import shutil
import subprocess
from typing import Dict, List, Optional
import yt_dlp
import whisper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Validates required system dependencies."""
    dependencies = ["ffmpeg", "yt-dlp"]
    for dep in dependencies:
        if not shutil.which(dep):
            logger.error(f"{dep} not found. Please install it and add it to PATH.")
            exit(1)
    logger.info("System dependencies checked successfully.")

def download_audio(url: str, output_dir: str = "data/raw") -> Optional[Dict]:
    """Downloads best available audio using yt-dlp."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    for attempt in range(3):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                return {
                    "video_id": info["id"],
                    "title": info["title"],
                    "url": url,
                    "duration_seconds": info.get("duration"),
                    "filepath": ydl.prepare_filename(info),
                    "ext": info["ext"]
                }
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {e}")
            if attempt == 2:
                logger.error(f"Failed to download {url} after 3 attempts.")
    return None

def process_audio(input_path: str, output_dir: str = "data/processed_audio") -> Optional[str]:
    """Standardizes audio to WAV, mono, 16kHz using FFmpeg."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return None
    
    filename = os.path.basename(input_path)
    video_id = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{video_id}.wav")
    
    if os.path.exists(output_path):
        logger.info(f"Processed audio already exists: {output_path}")
        return output_path

    command = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000",
        output_path
    ]
    
    for attempt in range(3):
        try:
            subprocess.run(command, check=True, capture_output=True)
            logger.info(f"Successfully processed audio: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.warning(f"FFmpeg attempt {attempt + 1} failed for {input_path}: {e.stderr.decode()}")
            if attempt == 2:
                logger.error(f"FFmpeg failed after 3 attempts: {input_path}")
    return None

def transcribe_audio(audio_path: str, model_name: str = "base") -> Optional[Dict]:
    """Runs transcription using local Whisper model."""
    for attempt in range(3):
        try:
            logger.info(f"Attempt {attempt + 1}: Loading Whisper model: {model_name}")
            model = whisper.load_model(model_name)
            logger.info(f"Transcribing: {audio_path}")
            result = model.transcribe(audio_path)
            return {
                "full_text": result["text"],
                "segments": [
                    {"start": s["start"], "end": s["end"], "text": s["text"]}
                    for s in result["segments"]
                ]
            }
        except Exception as e:
            logger.warning(f"Transcription attempt {attempt + 1} failed for {audio_path}: {e}")
            if attempt == 2:
                logger.error(f"Transcription failed after 3 attempts: {audio_path}")
    return None

def save_output(video_metadata: Dict, transcript: Dict, output_dir: str = "outputs/json"):
    """Generates a structured JSON output file."""
    video_id = video_metadata["video_id"]
    output_path = os.path.join(output_dir, f"{video_id}_output.json")
    
    output_data = {
        "video_metadata": {
            "video_id": video_id,
            "title": video_metadata["title"],
            "url": video_metadata["url"],
            "duration_seconds": video_metadata["duration_seconds"]
        },
        "audio_metadata": {
            "sample_rate": 16000,
            "channels": 1
        },
        "transcript": transcript
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved structured JSON: {output_path}")
