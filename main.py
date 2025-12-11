import ffmpeg
import os

def extract_frames_every_second(video_path, output_dir):
    """
    Extracts frames from a video at every second using FFmpeg.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save the extracted frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        ValueError: If the output directory is not writable or invalid.
        RuntimeError: If FFmpeg fails to process the video.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    elif not os.access(output_dir, os.W_OK):
        raise ValueError(f"Output directory is not writable: {output_dir}")

    output_pattern = os.path.join(output_dir, "frame_%04d.png")

    try:
        (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=1)
            .output(output_pattern)
            .run(quiet=True)
        )
        print(f"Frames extracted successfully to {output_dir}")
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode()}") from e

def main():
    print("Hello from 11hackb!")
    # Example usage (replace with actual paths)
    # extract_frames_every_second("path/to/video.mp4", "output/frames")

if __name__ == "__main__":
    main()
