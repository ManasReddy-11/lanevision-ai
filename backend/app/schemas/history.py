from __future__ import annotations

from pydantic import BaseModel, Field


class HistoryRun(BaseModel):
    id: str
    kind: str
    algorithm: str
    status: str
    filename: str
    original_url: str
    result_url: str
    metrics: dict
    warnings: list[str] = Field(default_factory=list)
    created_at: str


class AnalyticsSummaryResponse(BaseModel):
    total_runs: int
    successful_runs: int
    failed_runs: int
    by_algorithm: dict[str, int]
    average_confidence: float
