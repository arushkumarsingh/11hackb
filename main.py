
"""
Main pipeline for rough cut video creation.
Takes a video, extracts frames, gets descriptions from OpenAI, 
uses an agent to define cuts, and stitches the final video.
"""

import os
import json
import sys
import tempfile
from pathlib import Path
from loguru import logger

from src.ffmpeg import extract_frames_every_second
from src.keyframe_analysis import describe_image, define_cuts
from src.define_cuts import TimeLineCutList, TimeCut
from src.stitch import cut_and_stitch_timeline


def main(video_path: str, output_path: str | None = None, reencode: bool = True):
    """
    Main pipeline: Extract frames -> Describe with OpenAI -> Define cuts -> Stitch video.
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video (default: input_name_roughcut.mp4)
        reencode: Whether to re-encode the output video
    """
    video_path = str(video_path).strip('"\'')
    
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Set default output path if not provided
    if output_path is None:
        base_name = Path(video_path).stem
        output_dir = Path(video_path).parent
        output_path = str(output_dir / f"{base_name}_roughcut.mp4")
    
    output_path = str(output_path).strip('"\'')
    
    logger.info("🎬 Starting rough cut pipeline...")
    logger.info(f"   📹 Input video: {video_path}")
    logger.info(f"   📤 Output video: {output_path}")
    
    # Step 1: Extract frames using ffmpeg
    logger.info("📸 Step 1: Extracting frames from video...")
    with tempfile.TemporaryDirectory() as temp_dir:
        extract_frames_every_second(video_path, temp_dir)
        
        # Get list of frame files, sorted by name (which corresponds to time)
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_") and f.endswith(".png")])
        logger.info(f"   ✅ Extracted {len(frame_files)} frames")
        
        # Step 2: Send each frame for image description to OpenAI
        logger.info("🤖 Step 2: Getting image descriptions from OpenAI...")
        descriptions = {}
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(temp_dir, frame_file)
            # Extract time in seconds from filename, e.g., frame_0001.png -> 0, frame_0002.png -> 1
            time_sec = int(frame_file.split("_")[1].split(".")[0]) - 1  # frame_0001 is 0s, 0002 is 1s, etc.
            logger.info(f"   📸 Processing frame {i+1}/{len(frame_files)} (t={time_sec}s)...")
            description = describe_image(frame_path)
            descriptions[str(time_sec)] = description
        
        logger.info(f"   ✅ Got descriptions for {len(descriptions)} frames")
        
        # Save descriptions to JSON for debugging
        output_json = str(Path(video_path).with_suffix("")) + "_descriptions.json"
        with open(output_json, "w") as f:
            json.dump(descriptions, f, indent=4)
        logger.info(f"   💾 Descriptions saved to {output_json}")
        
        # Step 3: Agent call to define cuts
        logger.info("✂️ Step 3: Agent analyzing descriptions to define cuts...")
        cuts = define_cuts(descriptions)
        logger.info(f"   ✅ Agent identified {len(cuts)} cuts")
        
        if not cuts:
            logger.warning("   ⚠️ No cuts defined! Creating full video copy...")
            # If no cuts, create a single cut for the entire video
            # We'll need to get video duration, but for now, use a large number
            cuts = [{"start": 0, "end": 999999, "reason": "no cuts defined"}]
        
        # Log cuts
        for i, cut in enumerate(cuts):
            logger.info(f"   ✂️ Cut {i+1}: {cut.get('start', 0)}s to {cut.get('end', 0)}s - {cut.get('reason', 'N/A')}")
        
        # Convert cuts to TimeLineCutList format
        timeline_cuts = [
            TimeCut(start=float(cut.get("start", 0)), end=float(cut.get("end", 0)))
            for cut in cuts
        ]
        timeline = TimeLineCutList(timeline=timeline_cuts)
        
        # Step 4: Call stitch to cut and save final video
        logger.info("🔗 Step 4: Cutting and stitching video segments...")
        result = cut_and_stitch_timeline(
            timeline=timeline,
            video_path=video_path,
            output_path=output_path,
            reencode=reencode
        )
        
        logger.info(f"   ✅ {result}")
        logger.info("🎉 Rough cut pipeline completed!")
        
        return output_path


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path> [output_path] [--no-reencode]")
        print("  video_path: Path to input video file")
        print("  output_path: (optional) Path to save output video")
        print("  --no-reencode: (optional) Skip re-encoding for faster processing")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else None
    reencode = "--no-reencode" not in sys.argv
    
    try:
        result_path = main(video_path, output_path, reencode)
        print(f"\n✅ SUCCESS: Rough cut video saved to: {result_path}")
    except Exception as e:
        logger.exception(f"❌ Pipeline failed: {e}")
        sys.exit(1)
