from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, func, inspect, insert
from sqlalchemy.orm import sessionmaker

from ..db.session import Base, get_engine


class DetectionRun(Base):
    __tablename__ = "detection_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    kind = Column(String, nullable=False, default="image")
    media_type = Column(String, nullable=False, default="image")
    algorithm = Column(String, nullable=False, default="classic")
    status = Column(String, nullable=False, default="success")
    filename = Column(String, nullable=False, default="", server_default="")
    original_filename = Column(String, nullable=False, default="")
    original_media_path = Column(String, nullable=False, default="")
    original_url = Column(String, nullable=False, default="", server_default="")
    processed_media_path = Column(String, nullable=False, default="")
    result_url = Column(String, nullable=False, default="", server_default="")
    processing_time_ms = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    left_lane_detected = Column(Boolean, nullable=True)
    right_lane_detected = Column(Boolean, nullable=True)
    video_fps = Column(Float, nullable=True)
    detection_success_rate = Column(Float, nullable=True)
    both_lane_detection_rate = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True, default=dict)
    warnings = Column(JSON, nullable=True, default=list)

    @property
    def metrics_json(self) -> dict[str, Any]:
        return self.metrics or {}

    @property
    def warnings_json(self) -> list[str]:
        return self.warnings or []


class HistoryService:
    def __init__(self, database_url: str | None = None, upload_dir: str | None = None, processed_dir: str | None = None) -> None:
        self.database_url = database_url or "sqlite:///./lanevision-history.db"
        self.upload_dir = Path(upload_dir or "uploads").resolve()
        self.processed_dir = Path(processed_dir or "processed").resolve()
        self.engine = get_engine(self.database_url)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)
        self._ensure_schema_columns()

    def _ensure_schema_columns(self) -> None:
        inspector = inspect(self.engine)
        if not inspector.has_table("detection_runs"):
            return
        columns = {column["name"] for column in inspector.get_columns("detection_runs")}
        with self.engine.begin() as connection:
            if "kind" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN kind VARCHAR")
            if "media_type" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN media_type VARCHAR")
            if "algorithm" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN algorithm VARCHAR")
            if "status" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN status VARCHAR")
            if "filename" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN filename VARCHAR NOT NULL DEFAULT ''")
            if "original_filename" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN original_filename VARCHAR")
            if "original_media_path" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN original_media_path VARCHAR")
            if "original_url" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN original_url VARCHAR NOT NULL DEFAULT ''")
            if "processed_media_path" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN processed_media_path VARCHAR")
            if "result_url" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN result_url VARCHAR NOT NULL DEFAULT ''")
            if "processing_time_ms" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN processing_time_ms FLOAT")
            if "confidence" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN confidence FLOAT")
            if "width" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN width INTEGER")
            if "height" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN height INTEGER")
            if "left_lane_detected" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN left_lane_detected BOOLEAN")
            if "right_lane_detected" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN right_lane_detected BOOLEAN")
            if "video_fps" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN video_fps FLOAT")
            if "detection_success_rate" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN detection_success_rate FLOAT")
            if "both_lane_detection_rate" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN both_lane_detection_rate FLOAT")
            if "metrics" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN metrics JSON")
            if "warnings" not in columns:
                connection.exec_driver_sql("ALTER TABLE detection_runs ADD COLUMN warnings JSON")

    def _coerce_record(self, *, media_type: str | None = None, kind: str | None = None, algorithm: str | None = None, status: str | None = None, filename: str | None = None, original_filename: str | None = None, original_media_path: str | None = None, processed_media_path: str | None = None, processing_time_ms: float | None = None, confidence: float | None = None, width: int | None = None, height: int | None = None, left_lane_detected: bool | None = None, right_lane_detected: bool | None = None, video_fps: float | None = None, detection_success_rate: float | None = None, both_lane_detection_rate: float | None = None, metrics: dict[str, Any] | None = None, metrics_json: dict[str, Any] | None = None, warnings: list[str] | None = None, warnings_json: list[str] | None = None) -> dict[str, Any]:
        resolved_media_type = media_type or ("video" if kind == "video" else "image" if kind == "image" else "image")
        resolved_filename = filename or original_filename or ""
        resolved_metrics = metrics_json or metrics or {}
        resolved_confidence = confidence
        if resolved_confidence is None and isinstance(resolved_metrics, dict):
            resolved_confidence = resolved_metrics.get("confidence")
        resolved_processing_time_ms = processing_time_ms
        if resolved_processing_time_ms is None and isinstance(resolved_metrics, dict):
            resolved_processing_time_ms = resolved_metrics.get("processing_time_ms")
        resolved_detection_success_rate = detection_success_rate
        if resolved_detection_success_rate is None and isinstance(resolved_metrics, dict):
            resolved_detection_success_rate = resolved_metrics.get("detection_success_rate")
        return {
            "kind": kind or ("video" if resolved_media_type == "video" else "image"),
            "media_type": resolved_media_type,
            "algorithm": algorithm or "classic",
            "status": status or "success",
            "filename": resolved_filename,
            "original_filename": resolved_filename,
            "original_media_path": original_media_path or "",
            "original_url": original_media_path or "",
            "processed_media_path": processed_media_path or "",
            "result_url": processed_media_path or "",
            "processing_time_ms": resolved_processing_time_ms,
            "confidence": resolved_confidence,
            "width": width,
            "height": height,
            "left_lane_detected": left_lane_detected,
            "right_lane_detected": right_lane_detected,
            "video_fps": video_fps,
            "detection_success_rate": resolved_detection_success_rate,
            "both_lane_detection_rate": both_lane_detection_rate,
            "metrics": metrics_json or resolved_metrics,
            "warnings": warnings_json or warnings or [],
        }

    def record_detection(
        self,
        *,
        media_type: str | None = None,
        kind: str | None = None,
        algorithm: str | None = None,
        status: str | None = None,
        original_filename: str | None = None,
        filename: str | None = None,
        original_media_path: str | None = None,
        original_url: str | None = None,
        processed_media_path: str | None = None,
        result_url: str | None = None,
        processing_time_ms: float | None = None,
        confidence: float | None = None,
        width: int | None = None,
        height: int | None = None,
        left_lane_detected: bool | None = None,
        right_lane_detected: bool | None = None,
        video_fps: float | None = None,
        fps: float | None = None,
        detection_success_rate: float | None = None,
        both_lane_detection_rate: float | None = None,
        metrics: dict[str, Any] | None = None,
        metrics_json: dict[str, Any] | None = None,
        warnings: list[str] | None = None,
        warnings_json: list[str] | None = None,
    ) -> str:
        values = self._coerce_record(
            media_type=media_type,
            kind=kind,
            algorithm=algorithm,
            status=status,
            filename=filename or original_filename,
            original_filename=original_filename or filename,
            original_media_path=original_media_path or original_url,
            processed_media_path=processed_media_path or result_url,
            processing_time_ms=processing_time_ms,
            confidence=confidence,
            width=width,
            height=height,
            left_lane_detected=left_lane_detected,
            right_lane_detected=right_lane_detected,
            video_fps=video_fps if video_fps is not None else fps,
            detection_success_rate=detection_success_rate,
            both_lane_detection_rate=both_lane_detection_rate,
            metrics=metrics,
            metrics_json=metrics_json,
            warnings=warnings,
            warnings_json=warnings_json,
        )
        with self.SessionLocal() as session:
            run_id = values.get("id") or str(uuid4())
            values.setdefault("id", run_id)
            values.setdefault("created_at", datetime.now(timezone.utc))
            columns = {column["name"] for column in inspect(self.engine).get_columns("detection_runs")}
            values.setdefault("original_url", values.get("original_media_path", ""))
            values.setdefault("result_url", values.get("processed_media_path", ""))
            insert_payload = {}
            for key, value in values.items():
                if key in {"id", "created_at", "kind", "media_type", "algorithm", "status", "filename", "original_filename", "original_media_path", "processed_media_path", "processing_time_ms", "confidence", "width", "height", "left_lane_detected", "right_lane_detected", "video_fps", "detection_success_rate", "both_lane_detection_rate"}:
                    if key in columns:
                        insert_payload[key] = value
                elif key == "metrics" and "metrics" in columns:
                    insert_payload["metrics"] = value
                elif key == "warnings" and "warnings" in columns:
                    insert_payload["warnings"] = value
                elif key == "metrics_json" and "metrics" in columns:
                    insert_payload["metrics"] = value
                elif key == "warnings_json" and "warnings" in columns:
                    insert_payload["warnings"] = value
            if "original_url" in columns:
                insert_payload["original_url"] = values.get("original_url", values.get("original_media_path", ""))
            if "result_url" in columns:
                insert_payload["result_url"] = values.get("result_url", values.get("processed_media_path", ""))
            if not insert_payload:
                insert_payload = {"id": run_id}
            session.execute(insert(DetectionRun).values(insert_payload))
            session.commit()
            return run_id

    def get_detection(self, detection_id: str) -> DetectionRun | None:
        with self.SessionLocal() as session:
            return session.get(DetectionRun, detection_id)

    def list_runs(self, limit: int = 20) -> list[DetectionRun]:
        with self.SessionLocal() as session:
            return session.query(DetectionRun).order_by(DetectionRun.created_at.desc()).limit(limit).all()

    def list_detections(self, *, page: int = 1, page_size: int = 20, media_type: str | None = None, algorithm: str | None = None, status: str | None = None, date_from: str | None = None, date_to: str | None = None) -> dict[str, Any]:
        if page < 1:
            raise ValueError("page must be >= 1")
        if page_size < 1 or page_size > 100:
            raise ValueError("page_size must be between 1 and 100")

        with self.SessionLocal() as session:
            query = self._build_query(session=session, media_type=media_type, algorithm=algorithm, status=status, date_from=date_from, date_to=date_to)
            total = query.count()
            items = query.order_by(DetectionRun.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
            total_pages = (total + page_size - 1) // page_size if total else 0
            return {"items": items, "page": page, "page_size": page_size, "total": total, "total_pages": total_pages}

    def _build_query(self, *, session, media_type: str | None = None, algorithm: str | None = None, status: str | None = None, date_from: str | None = None, date_to: str | None = None):
        query = session.query(DetectionRun)
        if media_type is not None:
            if media_type not in {"image", "video"}:
                raise ValueError("media_type must be image or video")
            query = query.filter(DetectionRun.media_type == media_type)
        if algorithm is not None:
            if algorithm not in {"classic", "advanced"}:
                raise ValueError("algorithm must be classic or advanced")
            query = query.filter(DetectionRun.algorithm == algorithm)
        if status is not None:
            if status not in {"success", "failed"}:
                raise ValueError("status must be success or failed")
            query = query.filter(DetectionRun.status == status)
        if date_from is not None:
            try:
                from_dt = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("date_from must be an ISO datetime string") from exc
            query = query.filter(DetectionRun.created_at >= from_dt)
        if date_to is not None:
            try:
                to_dt = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("date_to must be an ISO datetime string") from exc
            query = query.filter(DetectionRun.created_at <= to_dt)
        return query

    def delete_detection(self, detection_id: str, *, delete_files: bool = True) -> bool:
        with self.SessionLocal() as session:
            run = session.get(DetectionRun, detection_id)
            if run is None:
                return False
            if delete_files:
                self._delete_associated_media(run)
            session.delete(run)
            session.commit()
            return True

    def _delete_associated_media(self, run: DetectionRun) -> None:
        for media_path in [run.original_media_path, run.processed_media_path]:
            if not media_path:
                continue
            candidate = self._resolve_safe_media_path(media_path)
            if candidate and candidate.exists() and candidate.is_file():
                candidate.unlink()

    def _resolve_safe_media_path(self, media_path: str) -> Path | None:
        candidate = Path(media_path).expanduser()
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
        else:
            candidate = candidate.resolve()

        allowed_roots = [self.upload_dir.resolve(), self.processed_dir.resolve()]
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
                return candidate
            except ValueError:
                continue
        return None

    def get_analytics(self) -> dict[str, Any]:
        with self.SessionLocal() as session:
            runs = session.query(DetectionRun).all()
            total_detections = len(runs)
            total_images = sum(1 for run in runs if run.media_type == "image")
            total_videos = sum(1 for run in runs if run.media_type == "video")
            classic_detections = sum(1 for run in runs if run.algorithm == "classic")
            advanced_detections = sum(1 for run in runs if run.algorithm == "advanced")
            successful_detections = sum(1 for run in runs if run.status == "success")
            failed_detections = total_detections - successful_detections
            confs = [float(run.confidence) for run in runs if run.confidence is not None]
            pctimes = [float(run.processing_time_ms) for run in runs if run.processing_time_ms is not None]
            video_runs = [run for run in runs if run.media_type == "video"]
            fps_values = [float(run.video_fps) for run in video_runs if run.video_fps is not None]
            success_rates = [float(run.detection_success_rate) for run in video_runs if run.detection_success_rate is not None]
            both_lane_rates = [float(run.both_lane_detection_rate) for run in video_runs if run.both_lane_detection_rate is not None]
            activity_timeline = []
            for row in (
                session.query(func.date(DetectionRun.created_at), func.count(DetectionRun.id))
                .group_by(func.date(DetectionRun.created_at))
                .order_by(func.date(DetectionRun.created_at))
                .all()
            ):
                date_value = row[0]
                if hasattr(date_value, "strftime"):
                    date_text = date_value.strftime("%Y-%m-%d")
                else:
                    date_text = str(date_value)
                activity_timeline.append({"date": date_text, "count": row[1]})
            recent_detections = [
                {
                    "id": run.id,
                    "media_type": run.media_type,
                    "algorithm": run.algorithm,
                    "status": run.status,
                    "original_filename": run.original_filename,
                    "created_at": run.created_at.isoformat(),
                }
                for run in session.query(DetectionRun).order_by(DetectionRun.created_at.desc()).limit(5).all()
            ]
            payload = {
                "total_detections": total_detections,
                "total_images": total_images,
                "total_videos": total_videos,
                "classic_detections": classic_detections,
                "advanced_detections": advanced_detections,
                "successful_detections": successful_detections,
                "failed_detections": failed_detections,
                "average_confidence": sum(confs) / len(confs) if confs else 0.0,
                "average_processing_time_ms": sum(pctimes) / len(pctimes) if pctimes else 0.0,
                "average_processing_fps": sum(fps_values) / len(fps_values) if fps_values else 0.0,
                "average_detection_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0.0,
                "average_both_lane_detection_rate": sum(both_lane_rates) / len(both_lane_rates) if both_lane_rates else 0.0,
                "per_algorithm": {
                    "classic": {
                        "count": classic_detections,
                        "average_confidence": sum(float(run.confidence) for run in runs if run.algorithm == "classic" and run.confidence is not None) / len([run for run in runs if run.algorithm == "classic" and run.confidence is not None]) if any(run.algorithm == "classic" and run.confidence is not None for run in runs) else 0.0,
                        "average_processing_time_ms": sum(float(run.processing_time_ms) for run in runs if run.algorithm == "classic" and run.processing_time_ms is not None) / len([run for run in runs if run.algorithm == "classic" and run.processing_time_ms is not None]) if any(run.algorithm == "classic" and run.processing_time_ms is not None for run in runs) else 0.0,
                    },
                    "advanced": {
                        "count": advanced_detections,
                        "average_confidence": sum(float(run.confidence) for run in runs if run.algorithm == "advanced" and run.confidence is not None) / len([run for run in runs if run.algorithm == "advanced" and run.confidence is not None]) if any(run.algorithm == "advanced" and run.confidence is not None for run in runs) else 0.0,
                        "average_processing_time_ms": sum(float(run.processing_time_ms) for run in runs if run.algorithm == "advanced" and run.processing_time_ms is not None) / len([run for run in runs if run.algorithm == "advanced" and run.processing_time_ms is not None]) if any(run.algorithm == "advanced" and run.processing_time_ms is not None for run in runs) else 0.0,
                    },
                },
                "activity_timeline": activity_timeline,
                "recent_detections": recent_detections,
            }
            payload.setdefault("total_runs", payload["total_detections"])
            payload.setdefault("successful_runs", payload["successful_detections"])
            payload.setdefault("failed_runs", payload["failed_detections"])
            payload.setdefault("by_algorithm", {"classic": classic_detections, "advanced": advanced_detections})
            return AnalyticsSummary(payload)


class AnalyticsSummary(dict):
    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
