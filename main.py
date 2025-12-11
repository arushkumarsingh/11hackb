

import os
import json
import tempfile
from src.ffmpeg import extract_frames_every_second
from src.keyframe_analysis import describe_image, define_cuts

def main(video_path):
    # Create a temporary directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract frames every second
        extract_frames_every_second(video_path, temp_dir)

        # Get list of frame files, sorted by name (which corresponds to time)
        frame_files = sorted([f for f in os.listdir(temp_dir) if f.startswith("frame_") and f.endswith(".png")])

        descriptions = {}
        for frame_file in frame_files:
            frame_path = os.path.join(temp_dir, frame_file)
            # Extract time in seconds from filename, e.g., frame_0001.png -> 0, frame_0002.png -> 1
            time_sec = int(frame_file.split("_")[1].split(".")[0]) - 1  # frame_0001 is 0s, 0002 is 1s, etc.
            description = describe_image(frame_path)
            descriptions[str(time_sec)] = description

        # Save to JSON
        output_json = video_path.rsplit(".", 1)[0] + "_descriptions.json"
        with open(output_json, "w") as f:
            json.dump(descriptions, f, indent=4)

        print(f"Descriptions saved to {output_json}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]
    main(video_path)
