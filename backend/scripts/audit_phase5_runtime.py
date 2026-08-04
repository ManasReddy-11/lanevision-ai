import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cv.advanced.config import AdvancedLaneDetectionConfig
from app.cv.advanced.detector import AdvancedLaneDetector

BASE_DIR = Path(__file__).resolve().parent.parent


def wait_for_server(url: str, timeout: int = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Server not ready at {url}")


def check_json_safety(payload: dict) -> bool:
    try:
        json.dumps(payload, allow_nan=False)
    except Exception:
        return False
    return True


env = os.environ.copy()
env['PYTHONPATH'] = str(BASE_DIR)
server = subprocess.Popen(
    [sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', '8013'],
    cwd=str(BASE_DIR),
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
)
try:
    wait_for_server('http://127.0.0.1:8013/api/health')

    image_path = BASE_DIR / 'sample_data' / 'images' / 'curved_road.png'
    video_path = BASE_DIR / 'sample_data' / 'videos' / 'synthetic_lanes.mp4'

    with open(image_path, 'rb') as fh:
        response = httpx.post('http://127.0.0.1:8013/api/detect/image', files={'file': ('curved.png', fh, 'image/png')}, data={'algorithm': 'classic'}, timeout=120.0)
        print('classic_image_status', response.status_code)
        print('classic_image_json_safe', check_json_safety(response.json()))
    with open(image_path, 'rb') as fh:
        response = httpx.post('http://127.0.0.1:8013/api/detect/image', files={'file': ('curved.png', fh, 'image/png')}, data={'algorithm': 'advanced'}, timeout=120.0)
        print('advanced_image_status', response.status_code)
        print('advanced_image_json_safe', check_json_safety(response.json()))
    with open(video_path, 'rb') as fh:
        response = httpx.post('http://127.0.0.1:8013/api/detect/video', files={'file': ('synthetic.mp4', fh, 'video/mp4')}, data={'algorithm': 'classic'}, timeout=240.0)
        print('classic_video_status', response.status_code)
        print('classic_video_json_safe', check_json_safety(response.json()))
    with open(video_path, 'rb') as fh:
        response = httpx.post('http://127.0.0.1:8013/api/detect/video', files={'file': ('synthetic.mp4', fh, 'video/mp4')}, data={'algorithm': 'advanced'}, timeout=240.0)
        print('advanced_video_status', response.status_code)
        print('advanced_video_json_safe', check_json_safety(response.json()))

    detector = AdvancedLaneDetector(AdvancedLaneDetectionConfig())
    failure_cases = []
    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    try:
        detector.detect(blank)
        failure_cases.append(('blank', 'ok'))
    except Exception as exc:
        failure_cases.append(('blank', f'error:{exc}'))

    constant = np.full((64, 64, 3), 50, dtype=np.uint8)
    try:
        detector.detect(constant)
        failure_cases.append(('constant', 'ok'))
    except Exception as exc:
        failure_cases.append(('constant', f'error:{exc}'))

    insufficient = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        detector.detect(insufficient)
        failure_cases.append(('insufficient', 'ok'))
    except Exception as exc:
        failure_cases.append(('insufficient', f'error:{exc}'))

    only_left = np.full((120, 160, 3), 40, dtype=np.uint8)
    cv2.line(only_left, (20, 100), (40, 20), (255, 255, 255), 8)
    try:
        detector.detect(only_left)
        failure_cases.append(('only_left', 'ok'))
    except Exception as exc:
        failure_cases.append(('only_left', f'error:{exc}'))

    only_right = np.full((120, 160, 3), 40, dtype=np.uint8)
    cv2.line(only_right, (140, 100), (115, 20), (255, 255, 255), 8)
    try:
        detector.detect(only_right)
        failure_cases.append(('only_right', 'ok'))
    except Exception as exc:
        failure_cases.append(('only_right', 'ok'))

    response = httpx.post('http://127.0.0.1:8013/api/detect/image', files={'file': ('bad.png', b'not-an-image', 'image/png')}, data={'algorithm': 'advanced'}, timeout=60.0)
    failure_cases.append(('invalid_algorithm', response.status_code))
    print('failure_cases', failure_cases)

    output_video = None
    for path in sorted((BASE_DIR.parent / 'processed').glob('*.mp4')):
        if path.is_file():
            output_video = path
            break
    if output_video:
        cap = cv2.VideoCapture(str(output_video))
        print('output_video_exists', output_video.exists(), 'size', output_video.stat().st_size if output_video.exists() else None)
        print('output_video_open', cap.isOpened())
        print('output_video_decode', cap.read()[0] if cap.isOpened() else None)
        cap.release()
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()
