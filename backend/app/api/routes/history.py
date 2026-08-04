from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...core.config import get_settings
from ...services.history_service import HistoryService

router = APIRouter()
settings = get_settings()


def get_history_service() -> HistoryService:
    return HistoryService(database_url=settings.database_url, upload_dir=settings.upload_dir, processed_dir=settings.processed_dir)


@router.get("/detections")
def list_detections(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    media_type: str | None = Query(default=None),
    algorithm: str | None = Query(default=None),
    status: str | None = Query(default=None),
    date_from: str | None = Query(default=None),
    date_to: str | None = Query(default=None),
    service: HistoryService = Depends(get_history_service),
) -> dict[str, Any]:
    try:
        return service.list_detections(page=page, page_size=page_size, media_type=media_type, algorithm=algorithm, status=status, date_from=date_from, date_to=date_to)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc


@router.get("/detections/{detection_id}")
def get_detection(detection_id: str, service: HistoryService = Depends(get_history_service)) -> dict[str, Any]:
    item = service.get_detection(detection_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")
    return {
        "id": item.id,
        "media_type": item.media_type,
        "algorithm": item.algorithm,
        "status": item.status,
        "original_filename": item.original_filename,
        "original_media_path": item.original_media_path,
        "processed_media_path": item.processed_media_path,
        "created_at": item.created_at.isoformat(),
        "processing_time_ms": item.processing_time_ms,
        "confidence": item.confidence,
        "width": item.width,
        "height": item.height,
        "left_lane_detected": item.left_lane_detected,
        "right_lane_detected": item.right_lane_detected,
        "video_fps": item.video_fps,
        "detection_success_rate": item.detection_success_rate,
        "both_lane_detection_rate": item.both_lane_detection_rate,
        "metrics": item.metrics_json or {},
        "warnings": item.warnings_json or [],
        "original_media_url": f"/media/uploads/{Path(item.original_media_path).name}" if item.original_media_path else None,
        "processed_media_url": f"/media/processed/{Path(item.processed_media_path).name}" if item.processed_media_path else None,
        "media_available": bool(item.original_media_path) and bool(item.processed_media_path),
    }


@router.delete("/detections/{detection_id}")
def delete_detection(detection_id: str, service: HistoryService = Depends(get_history_service)) -> dict[str, Any]:
    deleted = service.delete_detection(detection_id, delete_files=True)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Detection not found")
    return {"deleted": True, "id": detection_id}


@router.get("/analytics")
def analytics(service: HistoryService = Depends(get_history_service)) -> dict[str, Any]:
    return service.get_analytics()
