# LaneVision AI

LaneVision AI is an incremental portfolio project focused on road-lane detection and analysis using both traditional computer-vision pipelines and an experimental neural segmentation mode.

## Current status

The project currently supports:
- FastAPI backend image and video detection APIs
- Classic and Advanced traditional computer-vision lane detectors
- A new AI segmentation mode using a PyTorch U-Net architecture
- History and analytics persistence
- Analytics dashboard and React/Vite frontend

## AI architecture

The AI mode uses a lightweight PyTorch U-Net for semantic lane segmentation. The pipeline includes:
- deterministic preprocessing (BGR → RGB, resize, normalization, tensor conversion)
- model inference with `model.eval()` and `torch.inference_mode()`
- mask postprocessing and connected-component filtering
- lane geometry estimation for overlay and metrics

Classic and Advanced modes remain traditional computer-vision approaches and are not replaced by the AI mode.

## Model requirements

The AI path requires a trained checkpoint in PyTorch `.pth` format. A checkpoint is not shipped in this repository by default.

Expected placement:
- backend/models/lane_segmentation.pth

The checkpoint path can be configured with environment variables:

```bash
AI_ENABLED=true
AI_MODEL_PATH=backend/models/lane_segmentation.pth
AI_DEVICE=cpu
AI_MASK_THRESHOLD=0.5
```

## Local installation

To start the local demo with Docker, run `docker compose up --build` from the repository root and open
<http://localhost:5173>. For AI mode, put a checkpoint under `backend/models/` and set
`AI_MODEL_PATH=/app/models/<checkpoint>.pth` when starting Compose.

### Private staging with C1

The local C1 staging package is `backend/models/lanevision-c1-seed42-epoch12.pth`.
It contains the inference weights and model metadata. Git ignores the checkpoint, so copy it
into `backend/models/` when setting up another checkout. Verify the package SHA-256 before loading it:

```text
d8ab1d76d87fb7e8cb5e5eb733609461e14ca44d188b9be96d61e6aa66c3386c
```

The packaged model uses the existing lane-extraction defaults, a probability threshold of
0.30 and a maximum gap of 12 sampled rows. `AI_MASK_THRESHOLD` controls the pixel mask separately.

From the repository root in PowerShell, use the local backend environment and a separate staging
database and media directory:

```powershell
New-Item -ItemType Directory -Force dataset/staging | Out-Null
$env:DATABASE_URL = 'sqlite:///./dataset/staging/lanevision.db'
$env:UPLOAD_DIR = 'dataset/staging/uploads'
$env:PROCESSED_DIR = 'dataset/staging/processed'
$env:AI_ENABLED = 'true'
$env:AI_DEVICE = 'cpu'
$env:AI_MODEL_PATH = (Resolve-Path backend/models/lanevision-c1-seed42-epoch12.pth).Path
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal, run `npm ci` and `npm run dev -- --host 127.0.0.1` from `frontend/`.
Open <http://localhost:5173>, select **AI Segmentation**, and upload a short MP4.
To roll back to Classic in staging, stop the API, set `AI_ENABLED=false`, restart it,
and select **Classic OpenCV** in the frontend.

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Install FFmpeg with H.264 encoding support and put it on PATH for processed video playback in browsers.
The Docker backend includes FFmpeg. Without FFmpeg, the API still returns the OpenCV video export.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Running AI mode

1. Place a compatible PyTorch checkpoint at the configured `AI_MODEL_PATH`.
2. Start the backend.
3. Send an image or video upload with `algorithm=ai`.
4. If no usable checkpoint is configured, the API returns HTTP 503 for AI requests. Select Classic or Advanced to continue.

Important: the current AI mode is architecture-ready and safe for integration, but it is not a verified real-neural-inference path until a real trained checkpoint is provided and successfully loaded.

## Testing

```bash
cd backend
pytest
```

## Frontend verification

```bash
cd frontend
npm ci
npm test
npm run build
```

Open **Analyze a video**, upload a short MP4, and run Classic. Play the annotated result,
then open **View history** and follow the result link for that run. Check that Analytics includes
the run and that choosing a new video clears the previous result.

## Notes

- Classic and Advanced modes are the stable traditional CV pipelines.
- AI mode uses neural segmentation and is designed to work with a real trained checkpoint.
- Real inference verification is pending until a trained checkpoint is available.
