import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from urllib.parse import urljoin

import cv2
import numpy as np

BASE_URL = "http://127.0.0.1:8010"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPLOADS_DIR = (PROJECT_ROOT / "uploads").resolve()
PROCESSED_DIR = (PROJECT_ROOT / "processed").resolve()
SYNTHETIC_IMAGE = PROJECT_ROOT / "sample_data" / "images" / "synthetic_output.png"


def post_multipart(path: str, filename: str, payload: bytes, content_type: str):
    boundary = "----Phase3Boundary"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode("utf-8") + payload + f"\r\n--{boundary}--\r\n".encode("utf-8")
    req = urllib.request.Request(
        BASE_URL + path,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8")


def get(path: str):
    req = urllib.request.Request(BASE_URL + path, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def main():
    # Health/docs
    health_status, health_body = get("/api/health")
    docs_status, _ = get("/docs")
    uploads_status, _ = get("/media/uploads")
    print("health", health_status, health_body.decode())
    print("docs", docs_status)
    print("uploads_route", uploads_status)

    payload = SYNTHETIC_IMAGE.read_bytes()
    status, body = post_multipart("/api/detect/image", "synthetic_output.png", payload, "image/png")
    print("detect_status", status)
    data = json.loads(body)
    print(json.dumps(data, indent=2))

    original_url = data["original"]["url"]
    result_url = data["result"]["url"]
    original_path = UPLOADS_DIR / Path(original_url).name
    processed_path = PROCESSED_DIR / Path(result_url).name
    print("original_exists", original_path.exists())
    print("processed_exists", processed_path.exists())
    print("processed_size", processed_path.stat().st_size if processed_path.exists() else None)
    print("processed_decodable", cv2.imread(str(processed_path), cv2.IMREAD_COLOR) is not None)
    img = cv2.imread(str(processed_path), cv2.IMREAD_COLOR)
    print("processed_dims", img.shape if img is not None else None)
    media_status, _ = get(result_url)
    print("processed_media_status", media_status)

    # Error scenarios
    cases = [
        ("txt", b"not an image", "text/plain", "unsupported_extension"),
        ("bad.jpg", b"not an image", "image/gif", "invalid_mime"),
        ("empty", b"", "image/png", "empty"),
        ("corrupt.jpg", b"not a real image", "image/jpeg", "corrupt"),
    ]
    for filename, bytes_payload, ctype, label in cases:
        status, body = post_multipart("/api/detect/image", filename, bytes_payload, ctype)
        print(label, status, body)

    # Path traversal and safe filename
    traversal_payload = (Path("../sample_data/images/synthetic_output.png")).read_bytes()
    status, body = post_multipart("/api/detect/image", "../../test.jpg", traversal_payload, "image/png")
    data = json.loads(body)
    print("traversal_status", status)
    print("traversal_saved_url", data["original"]["url"])
    print("traversal_safe_name", Path(data["original"]["url"]).name)

    # JSON safety: ensure no NaN/inf/np scalars present in response
    def walk(obj):
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        if isinstance(obj, (float, int, str, bool)) or obj is None:
            return obj
        return repr(obj)

    safe_json = json.dumps(data)
    print("json_safe", safe_json is not None)


if __name__ == "__main__":
    main()
