"""HTTP API for the local domain-suggestion Transformer."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from inference.generate import DomainSuggestionGenerator


class SuggestRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1_000)


class SuggestResponse(BaseModel):
    reply: str
    model: str
    generation_ms: int


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.generator = DomainSuggestionGenerator()
    yield


app = FastAPI(title="Domain Suggestion Transformer API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"], allow_methods=["GET", "POST"], allow_headers=["Content-Type"])


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "domain-transformer"}


@app.post("/api/suggest", response_model=SuggestResponse)
def suggest(request: SuggestRequest) -> SuggestResponse:
    try:
        started_at = time.perf_counter()
        reply = app.state.generator.suggest(request.message)
        return SuggestResponse(reply=reply, model="domain-transformer", generation_ms=round((time.perf_counter() - started_at) * 1_000))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Model generation failed") from error
