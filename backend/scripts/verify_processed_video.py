import os
import sys
from pathlib import Path

import cv2

sys.path.insert(0, os.getcwd())
from app.core.config import get_settings

processed_dir = Path(get_settings().processed_dir)
files = sorted(processed_dir.glob('*.mp4'))
print('processed_dir', processed_dir)
print('files', [p.name for p in files[-3:]])
for p in files[-3:]:
    cap = cv2.VideoCapture(str(p))
    print('file', p.name, 'exists', p.exists(), 'size', p.stat().st_size if p.exists() else None, 'opened', cap.isOpened())
    if cap.isOpened():
        ok, frame = cap.read()
        print('first_frame', ok, 'shape', frame.shape if ok else None)
    cap.release()
