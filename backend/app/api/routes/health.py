from fastapi import APIRouter

from ...schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return the health status for the API service."""
    return HealthResponse(
        status="healthy",
        service="LaneVision AI API",
        version="1.0.0",
    )
