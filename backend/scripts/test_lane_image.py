from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.cv.config import LaneDetectionConfig
from app.cv.detector import ClassicLaneDetector
from app.cv.utils.validation import CVException, validate_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the classical lane detector on an input image")
    parser.add_argument("input_image", help="Path to the input image")
    parser.add_argument("output_image", help="Path to write the processed image")
    parser.add_argument("--debug", action="store_true", help="Write intermediate debug images")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_image)
    output_path = Path(args.output_image)

    if not input_path.exists():
        print("Error: missing image")
        return 1

    if input_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        print("Error: unsupported input")
        return 1

    try:
        frame = cv2.imread(str(input_path))
        if frame is None:
            raise CVException("invalid image")
        validate_frame(frame)
        detector = ClassicLaneDetector(config=LaneDetectionConfig())
        result = detector.detect(frame, debug=args.debug)
    except CVException as exc:
        print(f"Error: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"Error: {exc}")
        return 1

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), result.processed_frame)
    if not success:
        print("Error: write failure")
        return 1

    if args.debug and result.debug_output:
        debug_dir = output_path.parent / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, value in result.debug_output.items():
            cv2.imwrite(str(debug_dir / f"{name}.png"), value)

    print("Detection successful")
    print(f"Processing time: {result.processing_time_ms:.2f} ms")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Segments detected: {result.detected_segment_count}")
    print(f"Lane center: {result.lane_center_x}")
    print(f"Vehicle offset: {result.vehicle_offset_pixels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
