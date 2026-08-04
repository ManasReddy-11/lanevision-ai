import os
import sys
import time
from pathlib import Path

from fastapi import UploadFile

sys.path.insert(0, os.getcwd())
from app.services.video_processing_service import VideoProcessingService

path = Path('sample_data/videos/synthetic_lanes.mp4')
with path.open('rb') as fh:
    upload = UploadFile(filename=path.name, file=fh)
    service = VideoProcessingService()
    start = time.perf_counter()
    try:
        res = service.process_upload(upload, algorithm='advanced')
        print('done_ms', round((time.perf_counter() - start) * 1000, 2))
        print('result_url', res.result.url)
        processed = Path('..') / 'processed'
        matches = sorted([x.name for x in processed.glob('*.mp4') if x.is_file()])
        print('processed_matches', matches[-5:])
        print('processed_dir', processed.resolve())
    except Exception as exc:
        print(type(exc).__name__, exc)
        import traceback
        traceback.print_exc()
