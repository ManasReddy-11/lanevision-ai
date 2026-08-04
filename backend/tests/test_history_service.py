from __future__ import annotations

from app.services.history_service import HistoryService


def test_history_service_persists_and_lists_runs(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'history.sqlite3'}"
    service = HistoryService(database_url=database_url)
    service.initialize()

    run_id = service.record_detection(
        kind="image",
        algorithm="classic",
        status="success",
        filename="sample.jpg",
        original_url="/media/uploads/sample.jpg",
        result_url="/media/processed/sample.jpg",
        metrics={"confidence": 0.82, "processing_time_ms": 120.5},
        warnings=["minor warning"],
    )

    runs = service.list_runs(limit=10)

    assert len(runs) == 1
    assert runs[0].id == run_id
    assert runs[0].algorithm == "classic"
    assert runs[0].metrics["confidence"] == 0.82
    assert runs[0].warnings == ["minor warning"]


def test_history_service_builds_analytics_summary(tmp_path):
    database_url = f"sqlite:///{tmp_path / 'analytics.sqlite3'}"
    service = HistoryService(database_url=database_url)
    service.initialize()

    service.record_detection(kind="image", algorithm="classic", status="success", filename="a.jpg", original_url="/uploads/a.jpg", result_url="/processed/a.jpg", metrics={"confidence": 0.7, "processing_time_ms": 100.0})
    service.record_detection(kind="video", algorithm="advanced", status="success", filename="b.mp4", original_url="/uploads/b.mp4", result_url="/processed/b.mp4", metrics={"confidence": 0.9, "processing_time_ms": 400.0})
    service.record_detection(kind="image", algorithm="classic", status="failed", filename="c.jpg", original_url="/uploads/c.jpg", result_url="/processed/c.jpg", metrics={"confidence": 0.0, "processing_time_ms": 50.0})

    analytics = service.get_analytics()

    assert analytics.total_runs == 3
    assert analytics.successful_runs == 2
    assert analytics.failed_runs == 1
    assert analytics.by_algorithm["classic"] == 2
    assert analytics.by_algorithm["advanced"] == 1
    assert analytics.average_confidence == 0.5333333333333333
