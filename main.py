import argparse
import os
import logging
import whisper
from dotenv import load_dotenv
import pipeline

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("pipeline")

def run_pipeline(url: str, model, hf_token: str):
    """Orchestrates the full pipeline for a single URL."""
    logger.info(f"Starting pipeline for: {url}")
    
    # 1. Download
    metadata = pipeline.download_audio(url)
    if not metadata:
        return

    video_id = metadata["video_id"]
    output_path = f"outputs/json/{video_id}_output.json"
    
    if os.path.exists(output_path):
        logger.info(f"Skipping {video_id}: already processed.")
        return

    # 2. Process
    audio_path = pipeline.process_audio(metadata["filepath"])
    if not audio_path:
        return

    # 3. Transcribe
    transcript_result = pipeline.transcribe_audio(audio_path, model)
    if not transcript_result:
        return

    # 4. Diarization (Optional step, depends on hf_token availability)
    if hf_token:
        transcript_result["segments"] = pipeline.add_speaker_labels(
            audio_path, transcript_result["segments"], hf_token
        )
    else:
        logger.warning("No HF_TOKEN provided. Skipping speaker diarization.")
        for s in transcript_result["segments"]:
            s["speaker"] = "unknown"

    # 5. Output
    pipeline.save_output(metadata, transcript_result)
    logger.info(f"Pipeline completed successfully for: {video_id}")

def main():
    parser = argparse.ArgumentParser(description="YouTube Audio Transcription Pipeline")
    parser.add_argument("--url", help="Single YouTube URL")
    parser.add_argument("--input", help="Path to text file containing YouTube URLs (one per line)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    parser.add_argument("--dry-run", action="store_true", help="Print plan without executing")
    parser.add_argument("--hf-token", help="HuggingFace token for diarization")
    
    args = parser.parse_args()

    # 1. Dependency Check
    pipeline.check_dependencies()
    try:
        import pyannote.audio
    except ImportError:
        logger.warning("pyannote.audio not installed. Diarization will be disabled.")

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    urls = []
    if args.url:
        urls.append(args.url)
    
    if args.input:
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            return
        with open(args.input, "r") as f:
            urls.extend([line.strip() for line in f if line.strip()])

    if not urls:
        logger.warning("No URLs provided. Use --url or --input.")
        return

    if args.dry_run:
        logger.info(f"[DRY RUN] Would process {len(urls)} URLs using model '{args.model}':")
        if hf_token:
            logger.info(" - Speaker Diarization: ENABLED")
        else:
            logger.info(" - Speaker Diarization: DISABLED (No HF Token)")
        for url in urls:
            logger.info(f" - Plan: {url}")
        return

    logger.info(f"Processing {len(urls)} URLs...")
    
    # Load model once
    logger.info(f"Loading Whisper model: {args.model}")
    model = whisper.load_model(args.model)
    
    for url in urls:
        try:
            run_pipeline(url, model, hf_token)
        except Exception as e:
            logger.error(f"Critical failure processing {url}: {e}")
            continue

if __name__ == "__main__":
    main()
