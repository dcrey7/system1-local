"""HTTP routes for local decisions."""

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException

from system1.backend import GemmaBackend
from system1.core import MODEL, LowCoverageError, SystemOne
from system1.schema import Request


def create_app(engine: SystemOne | None = None) -> FastAPI:
    backend = GemmaBackend() if engine is None else None
    decider = engine if engine is not None else SystemOne(backend)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        if backend is not None:
            backend.close()

    app = FastAPI(title="System One", lifespan=lifespan)

    @app.post("/v1/systemone")
    async def decide(request: Request) -> dict:
        # The model's single slot processes one complete decision at a time.
        try:
            return decider.decide(request)
        except (LowCoverageError, httpx.HTTPError, KeyError, IndexError) as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=500, detail=str(error)) from error

    @app.get("/v1/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [{"id": MODEL, "object": "model", "owned_by": "local"}],
        }

    return app
