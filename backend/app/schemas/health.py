from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for the health endpoint response."""

    status: str = Field(..., json_schema_extra={"example": "healthy"})
    service: str = Field(..., json_schema_extra={"example": "LaneVision AI API"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
