"""
Simple test script to process a video with a hardcoded path.
"""

import sys
from pathlib import Path

from main import main
import logging 
logger = logging.getLogger(__name__)
# Hardcoded video path - change this to your video file
VIDEO_PATH = "/home/adminpc/test_vids/dr1.mp4"

# Optional: specify output path (leave as None to auto-generate)
OUTPUT_PATH = None  # Will default to {video_name}_roughcut.mp4

# Whether to re-encode the video
REENCODE = True


if __name__ == "__main__":
    # Update the hardcoded path here
    video_path = VIDEO_PATH
    
    # Validate video exists
    if not Path(video_path).exists():
        logger.error(f"❌ Video file not found: {video_path}")
        logger.info("Please update VIDEO_PATH in the script with a valid video file path")
        sys.exit(1)
    
    logger.info("🎬 Starting video processing test...")
    logger.info(f"   📹 Input: {video_path}")
    
    try:
        result_path = main(
            video_path=video_path,
            output_path=OUTPUT_PATH,
            reencode=REENCODE
        )
        logger.info(f"\n✅ SUCCESS: Processed video saved to: {result_path}")
    except Exception as e:
        logger.exception(f"❌ Processing failed: {e}")
        sys.exit(1)

