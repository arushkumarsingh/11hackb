"""
FastAPI application for video processing API.
Provides endpoint to process videos using the main pipeline.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
import logging 
logger = logging.getLogger(__name__)

from main import main


app = FastAPI(title="Video Processing API", version="1.0.0")


class ProcessRequest(BaseModel):
    """Request model for video processing."""
    video_path: str
    output_path: Optional[str] = None
    reencode: bool = True


class ProcessResponse(BaseModel):
    """Response model for video processing."""
    message: str
    video_path: str
    output_path: Optional[str] = None
    status: str


def process_video_background(video_path: str, output_path: Optional[str], reencode: bool):
    """
    Background task to process video.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path to save output video
        reencode: Whether to re-encode the output video
    """
    try:
        logger.info(f"Starting background processing for: {video_path}")
        result_path = main(video_path, output_path, reencode)
        logger.info(f"Background processing completed: {result_path}")
    except Exception as e:
        logger.exception(f"Background processing failed for {video_path}: {e}")


@app.post("/process", response_model=ProcessResponse)
async def process_video(
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    """
    Process a video file using the main pipeline.
    
    This endpoint accepts a local video path and starts processing it in the background.
    The processing includes:
    - Extracting frames from the video
    - Getting descriptions from OpenAI
    - Using an agent to define cuts
    - Stitching the final video
    
    Args:
        request: ProcessRequest containing video_path, optional output_path, and reencode flag
        background_tasks: FastAPI background tasks for async processing
    
    Returns:
        ProcessResponse with status and message
    
    Raises:
        HTTPException: If video file doesn't exist or path is invalid
    """
    video_path = str(request.video_path).strip('"\'')
    
    # Validate video path exists
    if not Path(video_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Video file not found: {video_path}"
        )
    
    # Validate it's a file, not a directory
    if not Path(video_path).is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a file: {video_path}"
        )
    
    # Set default output path if not provided
    output_path = request.output_path
    if output_path is None:
        base_name = Path(video_path).stem
        output_dir = Path(video_path).parent
        output_path = str(output_dir / f"{base_name}_roughcut.mp4")
    else:
        output_path = str(output_path).strip('"\'')
    
    # Add background task to process video
    background_tasks.add_task(
        process_video_background,
        video_path,
        output_path,
        request.reencode
    )
    
    logger.info(f"Video processing started for: {video_path}")
    logger.info(f"Output will be saved to: {output_path}")
    
    return ProcessResponse(
        message="Video processing started successfully",
        video_path=video_path,
        output_path=output_path,
        status="processing"
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Video Processing API",
        "version": "1.0.0",
        "endpoints": {
            "POST /process": "Start processing a video file"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

