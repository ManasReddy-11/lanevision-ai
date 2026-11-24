from __future__ import annotations

from pathlib import Path

import pytest

from app.services.history_service import HistoryService


def test_detection_record_can_be_created_and_retrieved(tmp_path):
    service = HistoryService(database_url=f"sqlite:///{tmp_path / 'persist.sqlite3'}")
    service.initialize()

    detection_id = service.record_detection(
        media_type="image",
        algorithm="classic",
        status="success",
        original_filename="sample.jpg",
        original_media_path="/media/uploads/sample.jpg",
        processed_media_path="/media/processed/sample.jpg",
        processing_time_ms=120.5,
        confidence=0.81,
        width=640,
        height=480,
        left_lane_detected=True,
        right_lane_detected=False,
        metrics_json={"detected_segments": 4},
        warnings_json=["minor warning"],
    )

    record = service.get_detection(detection_id)
    assert record is not None
    assert record.media_type == "image"
    assert record.algorithm == "classic"
    assert record.metrics_json["detected_segments"] == 4
    assert record.warnings_json == ["minor warning"]


def test_list_detections_supports_pagination_and_filters(tmp_path):
    service = HistoryService(database_url=f"sqlite:///{tmp_path / 'filters.sqlite3'}")
    service.initialize()

    service.record_detection(media_type="image", algorithm="classic", status="success", original_filename="a.jpg", original_media_path="/uploads/a.jpg", processed_media_path="/processed/a.jpg", processing_time_ms=100.0, confidence=0.6)
    service.record_detection(media_type="video", algorithm="classic", status="success", original_filename="b.mp4", original_media_path="/uploads/b.mp4", processed_media_path="/processed/b.mp4", processing_time_ms=200.0, confidence=0.7)
    service.record_detection(media_type="video", algorithm="advanced", status="failed", original_filename="c.mp4", original_media_path="/uploads/c.mp4", processed_media_path="/processed/c.mp4", processing_time_ms=300.0, confidence=0.3)

    result = service.list_detections(page=1, page_size=1, media_type="video", algorithm="classic", status="success")

    assert result["total"] == 1
    assert result["page"] == 1
    assert result["page_size"] == 1
    assert len(result["items"]) == 1
    assert result["items"][0].original_filename == "b.mp4"


def test_delete_detection_only_removes_safe_media_files(tmp_path):
    media_root = tmp_path / "media"
    upload_dir = media_root / "uploads"
    processed_dir = media_root / "processed"
    upload_dir.mkdir(parents=True)
    processed_dir.mkdir(parents=True)

    safe_file = upload_dir / "delete-me.jpg"
    safe_file.write_bytes(b"safe")
    outside_file = tmp_path / "outside.jpg"
    outside_file.write_bytes(b"outside")

    service = HistoryService(
        database_url=f"sqlite:///{tmp_path / 'delete.sqlite3'}",
        upload_dir=str(upload_dir),
        processed_dir=str(processed_dir),
    )
    service.initialize()

    detection_id = service.record_detection(
        media_type="image",
        algorithm="classic",
        status="success",
        original_filename="delete-me.jpg",
        original_media_path=str(safe_file),
        processed_media_path=str(processed_dir / "delete-me.jpg"),
        processing_time_ms=10.0,
        confidence=0.5,
    )

    service.delete_detection(detection_id, delete_files=True)

    assert not safe_file.exists()
    assert outside_file.exists()


def test_analytics_and_activity_timeline(tmp_path):
    service = HistoryService(database_url=f"sqlite:///{tmp_path / 'analytics.sqlite3'}")
    service.initialize()

    service.record_detection(media_type="image", algorithm="classic", status="success", original_filename="i1.jpg", original_media_path="/uploads/i1.jpg", processed_media_path="/processed/i1.jpg", processing_time_ms=100.0, confidence=0.8)
    service.record_detection(media_type="video", algorithm="advanced", status="success", original_filename="v1.mp4", original_media_path="/uploads/v1.mp4", processed_media_path="/processed/v1.mp4", processing_time_ms=200.0, confidence=0.9, fps=24.0, detection_success_rate=0.95, both_lane_detection_rate=0.89)
    service.record_detection(media_type="image", algorithm="advanced", status="failed", original_filename="i2.jpg", original_media_path="/uploads/i2.jpg", processed_media_path="/processed/i2.jpg", processing_time_ms=50.0, confidence=0.1)

    analytics = service.get_analytics()

    assert analytics["total_detections"] == 3
    assert analytics["total_images"] == 2
    assert analytics["total_videos"] == 1
    assert analytics["classic_detections"] == 1
    assert analytics["advanced_detections"] == 2
    assert analytics["successful_detections"] == 2
    assert analytics["failed_detections"] == 1
    # Averages of floats: the rows come back in no particular order and float addition is not associative
    # (0.8 + 0.9 + 0.1 != 0.1 + 0.8 + 0.9 in binary floating point), so compare with a tight relative tolerance.
    assert analytics["average_confidence"] == pytest.approx(0.6, rel=1e-12)
    assert analytics["average_processing_time_ms"] == pytest.approx(350.0 / 3.0, rel=1e-12)
    assert analytics["average_processing_fps"] == pytest.approx(24.0, rel=1e-12)
    assert analytics["average_detection_success_rate"] == pytest.approx(0.95, rel=1e-12)
    assert analytics["activity_timeline"][0]["count"] >= 1
    assert len(analytics["recent_detections"]) == 3
