"""
Functions for stitching videos with custom cropping.
stitch_videos(crop_map, output_path, reencode=True): crops multiple videos
with custom coordinates and merges them in one FFmpeg pass using filter_complex.
stitch_tool(crop_map: list[dict], output_path: str, reencode: bool = True): 
tool wrapper with logging and error handling that calls stitch_videos.
stitch_tool_from_config(config: StitchConfig): tool wrapper that accepts StitchConfig model
with automatic validation of crop dimensions.
Each function includes: comprehensive error handling, validation of input files,
logging of commands executed, and support for re-encoding or stream copying.
Uses FFmpeg concat filter to merge cropped video segments sequentially.
All cropped videos must have the same dimensions for successful stitching.
"""

import subprocess
from pathlib import Path

from loguru import logger


from .ffmpeg_tools import get_video_dimensions
from .video_analysis import get_face_centroid


def stitch_tool_from_config(config: StitchConfig) -> str:
    """
    Stitch videos using StitchConfig Pydantic model with automatic validation.
    
    Args:
        config: StitchConfig model with videos, output_path, and reencode settings
    
    Returns:
        Success or error message string
    
    Note:
        The StitchConfig model automatically validates that:
        - All video files exist
        - All crop coordinates are valid (x2 > x1, y2 > y1)
        - All cropped videos have the same dimensions (required for FFmpeg concat)
    """
    try:
        output_path = config.output_path.strip('"\'')
        
        logger.info("🔗 STITCH: Starting stitch process (using StitchConfig)")
        logger.info(f"   📹 Number of videos: {len(config.videos)}")
        logger.info(f"   📤 Output video: {output_path}")
        logger.info(f"   🔧 Re-encode: {config.reencode}")
        
        # Get dimensions from first video (all should be the same after validation)
        first_dimensions = config.get_crop_dimensions()
        logger.info(
            f"   📐 Cropped dimensions: {first_dimensions[0]}x{first_dimensions[1]} (width x height)"
        )
        
        for i, video in enumerate(config.videos):
            logger.info(
                f"   📹 Video {i + 1}: {video.path} (crop: {video.x1},{video.y1} to {video.x2},{video.y2})"
            )
        
        # Convert to dict format for stitch_videos
        crop_map = config.to_crop_map()
        
        stitch_videos(crop_map, output_path, config.reencode)

        logger.info("   ✅ Stitch completed successfully")
        return f"SUCCESS: Merged video saved to {output_path}"

    except Exception as e:
        error_msg = f"❌ STITCH FAILED: {e!s}"
        logger.exception(error_msg)
        return error_msg

def stitch_videos(crop_map, output_path, reencode=True):
    """
    Crop multiple videos with custom coordinates and merge in one FFmpeg pass.

    Args:
        crop_map (list[dict]): List of dicts with keys:
            - path (str): video path
            - x1, y1, x2, y2 (int): crop coordinates (pixels)
        output_path (str): Path to save merged output video.
        reencode (bool): Whether to re-encode videos for uniform output.
    """
    if not crop_map:
        raise ValueError("No crop map provided.")

    for v in crop_map:
        if not Path(v["path"]).exists():
            raise FileNotFoundError(f"Video not found: {v['path']}")

    # FFmpeg input args
    input_args = []
    crop_filters = []
    concat_inputs = []

    for i, vid in enumerate(crop_map):
        path = vid["path"]
        x1, y1 = vid.get("x1", 0), vid.get("y1", 0)
        x2, y2 = vid.get("x2"), vid.get("y2")

        if x2 is None or y2 is None:
            raise ValueError(f"x2/y2 missing for video {path}")

        crop_width = x2 - x1
        crop_height = y2 - y1

        input_args += ["-i", path]
        crop_filters.append(f"[{i}:v]crop={crop_width}:{crop_height}:{x1}:{y1}[v{i}]")
        concat_inputs.append(f"[v{i}][{i}:a]")

    # Build filter_complex
    filter_complex = (
        ";".join(crop_filters)
        + f";{''.join(concat_inputs)}concat=n={len(crop_map)}:v=1:a=1[outv][outa]"
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
    ]

    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]

    cmd += [output_path]

    logger.info(f"🎬 Running FFmpeg command:\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"✅ Merged video saved at: {output_path}")


def concatenate_videos(video_paths: list[str], output_path: str, reencode: bool = True) -> None:
    """
    Concatenate videos without cropping. Scales all videos to match the first video's dimensions.
    
    Args:
        video_paths (list[str]): List of video file paths
        output_path (str): Path to save merged output video
        reencode (bool): Whether to re-encode videos for uniform output
    """
    if not video_paths:
        raise ValueError("No video paths provided.")
    
    for video_path in video_paths:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Get dimensions of first video (target dimensions)
    target_width, target_height = get_video_dimensions(video_paths[0])
    
    # FFmpeg input args
    input_args = []
    scale_filters = []
    concat_inputs = []
    
    for i, video_path in enumerate(video_paths):
        input_args += ["-i", video_path]
        # Scale all videos to match first video's dimensions
        scale_filters.append(f"[{i}:v]scale={target_width}:{target_height}[v{i}]")
        concat_inputs.append(f"[v{i}][{i}:a]")
    
    # Build filter_complex
    filter_complex = (
        ";".join(scale_filters)
        + f";{''.join(concat_inputs)}concat=n={len(video_paths)}:v=1:a=1[outv][outa]"
    )
    
    cmd = [
        "ffmpeg",
        "-y",
        "-v", "error",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "[outa]",
    ]
    
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    
    cmd += [output_path]
    
    logger.info(f"🎬 Running FFmpeg concatenation command:\n{' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"✅ Concatenated video saved at: {output_path}")


def validate_video_aspect_ratio(video_path: str) -> tuple[int, int]:
    """
    Validate that video dimensions support 9:16 cropping (portrait orientation).
    Accepts videos that are portrait (aspect ratio <= 9:16) or can be cropped to 9:16.
    
    Args:
        video_path (str): Path to video file
    
    Returns:
        tuple: (width, height) of the video
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video cannot support 9:16 cropping
    """
    width, height = get_video_dimensions(video_path)
    aspect_ratio = width / height if height > 0 else 0
    target_aspect_ratio = 9 / 16  # ~0.5625 (portrait)
    
    # Accept videos that are portrait (aspect_ratio <= 9:16) or can be cropped to 9:16
    # For landscape videos, we need width >= height * (9/16) to crop to 9:16
    min_width_for_9_16 = height * target_aspect_ratio
    
    if aspect_ratio > target_aspect_ratio and width < min_width_for_9_16:
        raise ValueError(
            f"Video dimensions ({width}x{height}, aspect ratio: {aspect_ratio:.2f}) "
            f"cannot support 9:16 cropping. Need at least {min_width_for_9_16:.0f}px width "
            f"to crop to 9:16 aspect ratio."
        )
    
    logger.debug(f"Video {video_path} validated: {width}x{height} (aspect ratio: {aspect_ratio:.2f})")
    return width, height


def stitch_tool(
    video_paths: list[str],
    output_path: str, 
    reencode: bool = True,
    crop_height: int | None = None
) -> str:
    """
    Stitch videos with custom cropping by auto-detecting centroids and generating crop maps.
    
    Automatically extracts centroids from each video using face detection, then generates
    crop maps centered around the centroids with 9:16 aspect ratio, and stitches the videos.
    
    Args:
        video_paths (list[str]): List of video file paths
        output_path (str): Path to save merged output video
        reencode (bool): Whether to re-encode videos for uniform output
        crop_height (int, optional): Desired crop height. If None, uses maximum possible height.
    
    Returns:
        str: Success or error message
    
    Example:
        result = stitch_tool(
            video_paths=["video1.mp4", "video2.mp4"],
            output_path="output.mp4",
            crop_height=1080
        )
    """
    try:
        if not video_paths:
            raise ValueError("video_paths is required and cannot be empty")
        
        if output_path is None:
            raise ValueError("output_path is required")
        
        output_path = output_path.strip('"\'')
        
        logger.info("🔗 STITCH: Auto-detecting centroids and generating crop maps...")
        
        # Extract centroids and generate crop maps
        generated_crop_map = []
        crop_dimensions = None
        
        for i, video_path in enumerate(video_paths):
            video_path = str(video_path).strip('"\'')
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video not found: {video_path}")
            
            # Get centroid from video using face detection
            logger.info(f"   📹 Video {i + 1}: Extracting face centroid from {video_path}...")
            centroid = get_face_centroid(video_path)
            
            if centroid is None:
                raise ValueError(f"Failed to extract face centroid from {video_path}")
            
            centroid_x, centroid_y = centroid
            logger.info(f"   ✅ Face centroid found: ({centroid_x:.2f}, {centroid_y:.2f})")
            
            # Generate crop coordinates from centroid
            crop = generate_crop_from_centroid(
                video_path=video_path,
                centroid_x=centroid_x,
                crop_height=crop_height
            )
            
            # Store crop dimensions from first video
            if crop_dimensions is None:
                crop_dimensions = (crop["x2"] - crop["x1"], crop["y2"] - crop["y1"])
            
            # Validate all crops have same dimensions
            current_crop_dimensions = (crop["x2"] - crop["x1"], crop["y2"] - crop["y1"])
            if current_crop_dimensions != crop_dimensions:
                logger.warning(
                    f"Video {i+1} crop dimensions {current_crop_dimensions} differ from first video "
                    f"{crop_dimensions}. This may cause stitching issues."
                )
            
            generated_crop_map.append(crop)
        
        logger.info("🔗 STITCH: Starting stitch process")
        logger.info(f"   📹 Number of videos: {len(generated_crop_map)}")
        logger.info(f"   📤 Output video: {output_path}")
        logger.info(f"   🔧 Re-encode: {reencode}")
        
        if crop_dimensions:
            logger.info(
                f"   📐 Cropped dimensions: {crop_dimensions[0]}x{crop_dimensions[1]} (width x height)"
            )
        
        for i, vid in enumerate(generated_crop_map):
            logger.info(
                f"   📹 Video {i + 1}: {vid['path']} (crop: {vid['x1']},{vid['y1']} to {vid['x2']},{vid['y2']})"
            )

        stitch_videos(generated_crop_map, output_path, reencode)

        logger.info("   ✅ Stitch completed successfully")
        return f"SUCCESS: Merged video saved to {output_path}"

    except Exception as e:
        error_msg = f"❌ STITCH FAILED: {e!s}"
        logger.exception(error_msg)
        return error_msg


def cut_and_stitch_timeline(timeline: TimeLineCutList, video_path: str, output_path: str, reencode: bool = True) -> str:
    """
    Cut segments from a video based on timeline and stitch them together.

    Args:
        timeline: TimeLineCutList model with timeline cuts
        video_path: Path to source video file
        output_path: Path to save output video
        reencode: Whether to re-encode videos for uniform output

    Returns:
        Success or error message string

    Note:
        Uses FFmpeg select filter to extract time-based segments and concatenate them.
        All cuts must be from the same video file.
    """
    try:
        logger.info("✂️ TIMELINE CUT: Starting timeline-based cutting and stitching...")
        logger.info(f"   📹 Source video: {video_path}")
        logger.info(f"   📤 Output video: {output_path}")
        logger.info(f"   🔧 Re-encode: {reencode}")
        logger.info(f"   📋 Number of cuts: {len(timeline.timeline)}")

        for i, cut in enumerate(timeline.timeline):
            logger.info(
                f"   ✂️ Cut {i + 1}: {cut.start}s to {cut.end}s"
            )

        # Build FFmpeg filter for selecting time ranges
        select_filters = []
        concat_inputs = []

        # Create select filter for each time range
        for i, cut in enumerate(timeline.timeline):
            select_filters.append(f"select='between(t,{cut.start},{cut.end})'")
            concat_inputs.append(f"[{i}:v][{i}:a]")

        # For multiple segments, we need to use segment muxing or complex filter
        # Since select filter can produce multiple segments, we'll use segment approach
        # Actually, for this use case, it's better to cut each segment separately and then concatenate

        # First, cut each segment to temporary files
        temp_segments = []
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            for i, cut in enumerate(timeline.timeline):
                temp_segment = str(Path(temp_dir) / f"segment_{i}.mp4")

                cmd = [
                    "ffmpeg",
                    "-y",
                    "-v", "error",
                    "-i", video_path,
                    "-ss", str(cut.start),
                    "-t", str(cut.end - cut.start),
                    "-c", "copy" if not reencode else "libx264",
                ]

                if reencode:
                    cmd += ["-preset", "fast", "-crf", "23", "-c:a", "aac"]

                cmd += [temp_segment]

                logger.debug(f"Cutting segment {i+1}: {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
                temp_segments.append(temp_segment)

            # Now concatenate the segments
            concatenate_videos(temp_segments, output_path, reencode)

        logger.info("   ✅ Timeline cutting and stitching completed successfully")
        return f"SUCCESS: Cut and stitched video saved to {output_path}"

    except Exception as e:
        error_msg = f"❌ TIMELINE CUT FAILED: {e!s}"
        logger.exception(error_msg)
        return error_msg

if __name__ == "__main__":
    # Example: Stitch videos with auto-detected centroids
    # Note: Requires at least 2 videos for stitching
    video_paths = ["/home/adminpc/drive-download-20251118T083328Z-1-001/1.MP4", "/home/adminpc/drive-download-20251118T083328Z-1-001/2.MP4", "/home/adminpc/drive-download-20251118T083328Z-1-001/3.MP4", "/home/adminpc/drive-download-20251118T083328Z-1-001/4.MP4"]
    output_path = "output_stitched.mp4"
    crop_height = 1080
    
    result = stitch_tool(
        video_paths=video_paths,
        output_path=output_path,
        # crop_height=crop_height
    )
    logger.info(result)