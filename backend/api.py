import asyncio
import re
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field

from backend.status import pending_progress


class SpeechRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    text: str = Field(min_length=1, max_length=400)
    seed: int = Field(default=42, ge=0, le=2147483647, strict=True)


def create_api(worker, resolve_call, expired_errors, progress_store=None):
    api = FastAPI(title="Rumik TTS", version="2.0", docs_url=None, redoc_url=None, openapi_url=None)

    @api.exception_handler(HTTPException)
    async def http_error(_request, exc):
        return JSONResponse({"message": exc.detail}, status_code=exc.status_code, headers=exc.headers)

    @api.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc):
        # Do not echo submitted speech text in validation errors.
        return JSONResponse({"message": "Enter 1–400 characters and an integer seed between 0 and 2147483647."}, status_code=422)

    @api.middleware("http")
    async def no_cache(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @api.get("/api/health")
    async def health():
        return {"status": "online", "scope": "web-api", "model": "rumik-ai/rumik-oss-1", "gpu": "H100"}

    async def read_progress(key):
        if progress_store is None:
            return None
        try:
            return await asyncio.wait_for(progress_store.get.aio(key, None), timeout=2)
        except Exception:
            return None

    async def worker_snapshot():
        # All class methods share one Modal service function and its pool counts.
        stats = await asyncio.wait_for(worker().synthesize.get_current_stats.aio(), timeout=5)
        backlog = int(stats.backlog)
        running = int(stats.num_running_inputs)
        runners = int(stats.num_total_runners)
        # Pool counts are a snapshot, not a promise that a warm worker stays alive.
        gpu = "running" if running else "warming" if backlog and not runners else "ready" if runners else "inactive"
        lifecycle = await read_progress("worker-lifecycle")
        if isinstance(lifecycle, dict) and lifecycle.get("phase") == "warming" and (running or backlog):
            try:
                age = (datetime.now(timezone.utc) - datetime.fromisoformat(lifecycle["updated_at"])).total_seconds()
                if 0 <= age < 900:
                    gpu = "warming"
            except (KeyError, TypeError, ValueError):
                pass
        return {"deployment": "online", "gpu": gpu, "backlog": backlog,
                "runners": runners, "running_inputs": running}

    @api.get("/api/status")
    async def status():
        try:
            return await worker_snapshot()
        except Exception:
            raise HTTPException(503, "GPU status is temporarily unavailable.") from None

    async def submit(method, *args):
        try:
            call = await method.spawn.aio(*args)
        except Exception as exc:
            print(f"Job submission failed: {type(exc).__name__}", flush=True)
            raise HTTPException(503, "The GPU job could not be started.") from None
        return JSONResponse({"status": "accepted", "call_id": call.object_id}, status_code=202,
                            headers={"Retry-After": "2"})

    @api.post("/api/warmup", status_code=202)
    async def warmup():
        return await submit(worker().warmup)

    @api.post("/api/synthesize", status_code=202)
    async def synthesize(payload: SpeechRequest):
        return await submit(worker().synthesize, payload.text, payload.seed)

    @api.get("/api/jobs/{call_id}")
    async def poll(call_id: str):
        if not re.fullmatch(r"fc-[A-Za-z0-9_-]{3,128}", call_id):
            raise HTTPException(422, "Invalid job ID.")
        try:
            return await resolve_call(call_id).get.aio(timeout=0)
        except TimeoutError:
            progress = await read_progress(f"job:{call_id}")
            try:
                snapshot = await worker_snapshot()
            except Exception:
                snapshot = {"gpu": "unknown"}
            return JSONResponse({"status": "running", **snapshot,
                                 **pending_progress(progress, snapshot)},
                                status_code=202, headers={"Retry-After": "2"})
        except expired_errors:
            raise HTTPException(404, "This job has expired. Submit a new request.") from None
        except Exception as exc:
            print(f"GPU job failed: {type(exc).__name__}", flush=True)
            raise HTTPException(500, "Speech generation failed. Try a shorter text or warm up again.") from None

    return api
