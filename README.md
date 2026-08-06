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
AI_THRESHOLD=0.5
```

## Local installation

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

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
4. If no checkpoint is configured, the API will return a clear warning and the classic/advanced paths will remain available.

Important: the current AI mode is architecture-ready and safe for integration, but it is not a verified real-neural-inference path until a real trained checkpoint is provided and successfully loaded.

## Testing

```bash
cd backend
pytest
```

## Frontend verification

```bash
cd frontend
npm run build
```

## Notes

- Classic and Advanced modes are the stable traditional CV pipelines.
- AI mode uses neural segmentation and is designed to work with a real trained checkpoint.
- Real inference verification is pending until a trained checkpoint is available.
