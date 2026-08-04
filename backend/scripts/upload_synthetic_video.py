from pathlib import Path
from urllib.request import Request, urlopen

video_path = Path(__file__).resolve().parent.parent / "sample_data" / "videos" / "synthetic_lanes.mp4"
if not video_path.exists():
    raise SystemExit(f"missing video: {video_path}")

boundary = "----LaneVisionBoundary"
body = []
body.append(f"--{boundary}\r\n".encode())
body.append(b'Content-Disposition: form-data; name="file"; filename="synthetic_lanes.mp4"\r\n')
body.append(b"Content-Type: video/mp4\r\n\r\n")
body.append(video_path.read_bytes())
body.append(f"\r\n--{boundary}--\r\n".encode())

req = Request("http://127.0.0.1:8011/api/detect/video", data=b"".join(body), method="POST")
req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
with urlopen(req, timeout=180) as resp:
    print(resp.status)
    print(resp.read().decode("utf-8")[:4000])
