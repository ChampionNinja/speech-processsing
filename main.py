import argparse
import os
import logging
import pipeline

logger = logging.getLogger("pipeline")

def run_pipeline(url: str, model_name: str):
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
    transcript = pipeline.transcribe_audio(audio_path, model_name)
    if not transcript:
        return

    # 4. Output
    pipeline.save_output(metadata, transcript)
    logger.info(f"Pipeline completed successfully for: {video_id}")

def main():
    parser = argparse.ArgumentParser(description="YouTube Audio Transcription Pipeline")
    parser.add_argument("--url", help="Single YouTube URL")
    parser.add_argument("--input", help="Path to text file containing YouTube URLs (one per line)")
    parser.add_argument("--model", default="base", choices=["tiny", "base", "small", "medium", "large"], help="Whisper model size")
    
    args = parser.parse_args()

    # 1. Dependency Check
    pipeline.check_dependencies()

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

    logger.info(f"Processing {len(urls)} URLs...")
    for url in urls:
        try:
            run_pipeline(url, args.model)
        except Exception as e:
            logger.error(f"Critical failure processing {url}: {e}")
            continue

if __name__ == "__main__":
    main()
