# LaneVision AI

LaneVision AI is an incremental portfolio project focused on real-time lane detection and road analysis.

## Current status

This phase establishes the foundation for the project:
- FastAPI backend with a health endpoint
- React + Vite frontend starter
- Basic environment configuration
- Initial project structure

## Planned functionality

The following features are now implemented or in progress:
- image and video upload
- classical computer vision lane detection
- history and analytics persistence
- analytics dashboard for recent detections and trends
- production deployment

## Tech stack

- Backend: FastAPI, Uvicorn, Pydantic, SQLAlchemy
- Computer vision: OpenCV, NumPy
- Frontend: React, Vite, Tailwind CSS, React Router, Axios

## Project structure

- backend/: FastAPI application and Python modules
- frontend/: React/Vite application
- models/: placeholder for trained model assets
- sample_data/: sample images and videos
- docs/: project documentation

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
