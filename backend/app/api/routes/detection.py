from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ...core.config import get_settings
from ...core.logging import get_logger
from ...schemas.detection import DetectionResponse
from ...services.detection_service import DetectionService

logger = get_logger("lanevision.detection")
settings = get_settings()
router = APIRouter()
service = DetectionService()


@router.post("/detect/image", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def detect_image(file: UploadFile = File(...), algorithm: str = Form(default="classic")) -> DetectionResponse:
    """Process an uploaded road image with the selected lane detector."""
    if algorithm not in {"classic", "advanced", "ai"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported algorithm")
    try:
        return await service.process_upload(file, algorithm=algorithm)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Detection request failed")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive catch
        logger.exception("Unexpected detection failure")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Detection processing failed") from exc
