from pathlib import Path

from app.core import config


def test_media_dirs_resolve_from_project_root(monkeypatch):
    config.get_settings.cache_clear()
    monkeypatch.delenv("UPLOAD_DIR", raising=False)
    monkeypatch.delenv("PROCESSED_DIR", raising=False)

    settings = config.get_settings()
    expected_root = Path(config.__file__).resolve().parents[3]

    assert Path(settings.upload_dir) == expected_root / "uploads"
    assert Path(settings.processed_dir) == expected_root / "processed"
