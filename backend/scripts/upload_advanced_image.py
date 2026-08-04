from pathlib import Path

import cv2
import httpx
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
IMAGE_DIR = BASE_DIR / "sample_data" / "images"
VIDEO_DIR = BASE_DIR / "sample_data" / "videos"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def ensure_sample_image(path: Path) -> Path:
    if path.exists():
        return path

    height, width = 480, 720
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (30, 30, 30)
    cv2.line(image, (0, height - 60), (width, height - 60), (255, 255, 255), 3)
    cv2.line(image, (40, height - 80), (width - 60, height - 120), (255, 255, 255), 3)
    cv2.line(image, (80, height - 140), (width - 80, height - 180), (255, 255, 255), 3)
    cv2.imwrite(str(path), image)
    return path


image_path = ensure_sample_image(IMAGE_DIR / "curved_road.png")
video_path = VIDEO_DIR / "synthetic_lanes.mp4"
if not video_path.exists():
    raise SystemExit(f"Expected video sample not found: {video_path}")

for endpoint, file_path, file_name in [
    ("/api/detect/image", image_path, image_path.name),
    ("/api/detect/video", video_path, video_path.name),
]:
    with open(file_path, "rb") as fh:
        files = {"file": (file_name, fh, "image/png" if file_path.suffix.lower() == ".png" else "video/mp4")}
        data = {"algorithm": "advanced"}
        with httpx.Client(timeout=240.0) as client:
            response = client.post(
                f"http://127.0.0.1:8012{endpoint}",
                files=files,
                data=data,
                headers={"accept": "application/json"},
            )
        print(f"{endpoint} -> {response.status_code}")
        print(response.text[:4000])
        print("-" * 80)
