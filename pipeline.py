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

def transcribe_audio(audio_path: str, model) -> Optional[Dict]:
    """Runs transcription using pre-loaded Whisper model."""
    for attempt in range(3):
        try:
            logger.info(f"Attempt {attempt + 1}: Transcribing: {audio_path}")
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

def add_speaker_labels(audio_path: str, whisper_segments: List[Dict], hf_token: str) -> List[Dict]:
    """Runs diarization and matches speakers to Whisper segments using timestamp overlap."""
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        logger.error("pyannote.audio not installed. Diarization fallback to 'unknown'. Run: pip install pyannote.audio")
        for s in whisper_segments:
            s["speaker"] = "unknown"
        return whisper_segments

    try:
        logger.info("Loading pyannote diarization pipeline...")
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        if pipeline is None:
            raise ValueError("Pipeline could not be loaded. Check your HF_TOKEN and model access permissions.")

        # Check if we have GPU
        import torch
        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
            logger.info("Using GPU for diarization.")

        logger.info(f"Running diarization on: {audio_path}")
        diarization = pipeline(audio_path)
        
       # Convert diarization results to a readable list
        diar_segments = []

        for segment, speaker in diarization.speaker_diarization:
            diar_segments.append({
                "start": segment.start,
                "end": segment.end,
                "speaker": speaker
            })

        if not diar_segments:
            logger.warning(f"No speakers detected by diarization for: {audio_path}")
        # Match each whisper segment to the speaker with the most overlap
        labeled_segments = []
        for ws in whisper_segments:
            ws_start, ws_end = ws["start"], ws["end"]
            speaker_overlaps = {}

            for ds in diar_segments:
                overlap = max(0, min(ws_end, ds["end"]) - max(ws_start, ds["start"]))
                if overlap > 0:
                    speaker_overlaps[ds["speaker"]] = speaker_overlaps.get(ds["speaker"], 0) + overlap

            if speaker_overlaps:
                best_speaker = max(speaker_overlaps, key=speaker_overlaps.get)
                ws["speaker"] = best_speaker
            else:
                ws["speaker"] = "unknown"
            labeled_segments.append(ws)

        logger.info(f"Speaker labeling complete for: {audio_path}")
        return labeled_segments

    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        logger.info("Falling back to 'unknown' speaker labels.")
        for s in whisper_segments:
            s["speaker"] = "unknown"
        return whisper_segments

def format_timestamp(seconds: float, srt: bool = True) -> str:
    """Formats seconds to SRT (HH:MM:SS,mmm) or VTT (HH:MM:SS.mmm) format."""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    
    if ms >= 1000:
        ms -= 1000
        secs += 1
        if secs >= 60:
            secs -= 60
            mins += 1
            if mins >= 60:
                mins -= 60
                hrs += 1
                
    sep = "," if srt else "."
    return f"{hrs:02d}:{mins:02d}:{secs:02d}{sep}{ms:03d}"

def export_subtitles(segments: List[Dict], video_id: str, output_dir: str = "outputs/subtitles"):
    """Generates and saves SRT and VTT files from transcript segments."""
    os.makedirs(output_dir, exist_ok=True)
    
    srt_path = os.path.join(output_dir, f"{video_id}.srt")
    vtt_path = os.path.join(output_dir, f"{video_id}.vtt")
    
    # Write SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"], srt=True)
            end = format_timestamp(seg["end"], srt=True)
            speaker_prefix = f"[{seg.get('speaker', 'unknown')}] " if seg.get("speaker") and seg["speaker"] != "unknown" else ""
            text = seg["text"].strip()
            f.write(f"{idx}\n{start} --> {end}\n{speaker_prefix}{text}\n\n")
            
    # Write VTT
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for idx, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"], srt=False)
            end = format_timestamp(seg["end"], srt=False)
            speaker_prefix = f"[{seg.get('speaker', 'unknown')}] " if seg.get("speaker") and seg["speaker"] != "unknown" else ""
            text = seg["text"].strip()
            f.write(f"{idx}\n{start} --> {end}\n{speaker_prefix}{text}\n\n")
            
    logger.info(f"Subtitles exported successfully: {srt_path} & {vtt_path}")

def calculate_speaker_stats(segments: List[Dict]) -> List[Dict]:
    """Calculates statistics for each speaker in the transcription."""
    stats = {}
    total_time = 0.0
    
    for seg in segments:
        speaker = seg.get("speaker", "unknown")
        duration = max(0.0, seg["end"] - seg["start"])
        word_count = len(seg["text"].split())
        total_time += duration
        
        if speaker not in stats:
            stats[speaker] = {"talk_time": 0.0, "word_count": 0, "segments": 0}
            
        stats[speaker]["talk_time"] += duration
        stats[speaker]["word_count"] += word_count
        stats[speaker]["segments"] += 1

    # Finalize stats calculations
    speaker_analytics = []
    for speaker, data in stats.items():
        talk_time = data["talk_time"]
        word_count = data["word_count"]
        pct = (talk_time / total_time * 100) if total_time > 0 else 0.0
        wpm = (word_count / (talk_time / 60)) if talk_time > 0 else 0.0
        
        speaker_analytics.append({
            "speaker": speaker,
            "talk_time_seconds": round(talk_time, 2),
            "percentage": round(pct, 1),
            "word_count": word_count,
            "words_per_minute": round(wpm, 1),
            "segment_count": data["segments"]
        })
        
    return speaker_analytics

def print_speaker_summary(speaker_analytics: List[Dict]):
    """Prints a beautiful summary table of speaker analytics to the terminal."""
    logger.info("\n" + "="*70 + "\n" + "                   SPEAKER DIARIZATION STATISTICS\n" + "="*70)
    logger.info(f"{'Speaker':<15} | {'Talk Time (s)':<13} | {'Speech %':<8} | {'Word Count':<10} | {'WPM':<6}")
    logger.info("-" * 70)
    for spk in sorted(speaker_analytics, key=lambda x: x["talk_time_seconds"], reverse=True):
        logger.info(
            f"{spk['speaker']:<15} | "
            f"{spk['talk_time_seconds']:<13.2f} | "
            f"{spk['percentage']:>7.1f}% | "
            f"{spk['word_count']:<10} | "
            f"{spk['words_per_minute']:<6.1f}"
        )
    logger.info("="*70 + "\n")

def save_output(video_metadata: Dict, transcript: Dict, output_dir: str = "outputs/json"):
    """Generates a structured JSON output file with transcript statistics."""
    video_id = video_metadata["video_id"]
    output_path = os.path.join(output_dir, f"{video_id}_output.json")
    
    # Calculate statistics
    full_text = transcript["full_text"]
    segments = transcript["segments"]
    
    word_count = len(full_text.split())
    segment_count = len(segments)
    
    # Duration based on segments if available, otherwise fallback to metadata
    audio_duration_seconds = segments[-1]["end"] if segments else video_metadata["duration_seconds"]
    
    # Calculate speaker analytics
    speaker_analytics = calculate_speaker_stats(segments)
    print_speaker_summary(speaker_analytics)
    
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
        "transcript": {
            **transcript,
            "speaker_analytics": speaker_analytics,
            "statistics": {
                "word_count": word_count,
                "segment_count": segment_count,
                "audio_duration_seconds": round(audio_duration_seconds, 2)
            }
        }
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved structured JSON with stats: {output_path}")
    
    # Export subtitles
    export_subtitles(segments, video_id)
