from pathlib import Path

import cv2
import numpy as np

output_path = Path(__file__).resolve().parents[1] / "sample_data" / "videos" / "synthetic_lanes.mp4"
output_path.parent.mkdir(parents=True, exist_ok=True)

width, height = 1280, 720
fps = 30.0
frames = 90
writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

for idx in range(frames):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.line(frame, (0, int(height * 0.72)), (width, int(height * 0.72) + idx // 10), (255, 255, 255), 3)
    cv2.line(frame, (0, int(height * 0.68)), (width, int(height * 0.68) + idx // 12), (255, 255, 255), 3)
    cv2.putText(frame, "Synthetic Road", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    writer.write(frame)

writer.release()
print(output_path)
