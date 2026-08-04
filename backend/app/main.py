from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api.routes.detection import router as detection_router
from .api.routes.health import router as health_router
from .api.routes.history import router as history_router
from .api.routes.video_detection import router as video_detection_router
from .core.config import get_settings
from .core.logging import get_logger
from .services.history_service import HistoryService

logger = get_logger("lanevision.main")
settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Handle application startup and shutdown logging."""
    logger.info("Application startup")
    logger.info("Environment: %s", settings.environment)
    history_service = HistoryService(database_url=settings.database_url)
    history_service.initialize()
    yield
    logger.info("Application shutdown")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    debug=settings.debug,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

frontend_origins = [origin.strip() for origin in settings.frontend_url.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=frontend_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(detection_router, prefix=settings.api_prefix)
app.include_router(video_detection_router, prefix=settings.api_prefix)
app.include_router(history_router, prefix=settings.api_prefix)

upload_dir = Path(settings.upload_dir).resolve()
processed_dir = Path(settings.processed_dir).resolve()

upload_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(parents=True, exist_ok=True)

app.mount("/media/uploads", StaticFiles(directory=str(upload_dir)), name="uploads")
app.mount("/media/processed", StaticFiles(directory=str(processed_dir)), name="processed")
