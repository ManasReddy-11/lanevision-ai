from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ...core.logging import get_logger
from ...schemas.video_detection import VideoDetectionResponse
from ...services.video_processing_service import VideoProcessingService
from ...utils.video_validation import VideoValidationError

logger = get_logger("lanevision.video_detection")
router = APIRouter()
service = VideoProcessingService()


@router.post("/detect/video", response_model=VideoDetectionResponse, status_code=status.HTTP_200_OK)
async def detect_video(file: UploadFile = File(...), algorithm: str = Form(default="classic")) -> VideoDetectionResponse:
    """Process an uploaded road video with the selected lane detector and temporal smoothing."""
    if algorithm not in {"classic", "advanced", "ai"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported algorithm")
    try:
        return service.process_upload(file, algorithm=algorithm)
    except VideoValidationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.exception("Video detection request failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Video processing failed") from exc
