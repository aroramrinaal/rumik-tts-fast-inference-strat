import os
from pathlib import Path

import modal

from backend.status import progress_record


app = modal.App("rumik-tts-fast-inference")
progress_store = modal.Dict.from_name("rumik-tts-web-progress", create_if_missing=True)


def write_progress(phase, *, lifecycle=False):
    # Progress is advisory; a metadata outage must not fail a GPU job.
    try:
        key = "worker-lifecycle" if lifecycle else f"job:{modal.current_function_call_id()}"
        progress_store.put(key, progress_record(phase))
    except Exception:
        print("Progress update unavailable", flush=True)


model_volume = modal.Volume.from_name(
    os.environ.get("RUMIK_MODEL_VOLUME", "rumik-models"), create_if_missing=False
)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .uv_pip_install(
        "torch==2.9.1", "transformers==5.17.0", "accelerate==1.15.0",
        "safetensors==0.8.0", "soundfile==0.14.0", "numpy==2.5.3",
    )
    .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "PYTHONPATH": "/workspace"})
    .add_local_dir(
        Path(__file__).resolve().parents[1] / "inference_source",
        remote_path="/workspace/inference_source",
    )
    .add_local_python_source("backend")
)
web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("fastapi[standard]>=0.115,<1")
    .add_local_python_source("backend")
)


@app.cls(
    image=image,
    gpu="H100!",
    volumes={"/models": model_volume.with_mount_options(read_only=True)},
    cpu=8,
    memory=32768,
    timeout=180,
    startup_timeout=900,
    max_containers=1,
    min_containers=0,
    buffer_containers=0,
    scaledown_window=120,
)
class RumikWorker:
    @modal.enter()
    def load(self):
        write_progress("warming", lifecycle=True)
        try:
            self._load()
        except Exception:
            write_progress("failed", lifecycle=True)
            raise
        write_progress("ready", lifecycle=True)

    def _load(self):
        import time
        import torch
        from inference_source.fast_decode import apply
        from inference_source.rumik_tts import RumikTTS

        started = time.perf_counter()
        self.engine = apply(RumikTTS(Path("/models/rumik-oss-1")))
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        self.engine.synthesize(
            text="Hello from Rumik.", speaker="Ira",
            description="neutral, Indian English accent, steady pace",
            max_new_tokens=128,
        )
        torch.cuda.synchronize()
        self.warmup_seconds = time.perf_counter() - started

    @modal.method()
    def warmup(self):
        write_progress("complete")
        return {
            "status": "complete", "kind": "warmup", "gpu": "H100",
            "load_seconds": self.load_seconds, "warmup_seconds": self.warmup_seconds,
        }

    @modal.method()
    def synthesize(self, text: str, seed: int = 42):
        try:
            return self._synthesize(text, seed)
        except Exception:
            write_progress("failed")
            raise

    def _synthesize(self, text: str, seed: int):
        import base64
        import io
        import time
        import soundfile as sf
        import torch

        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 400:
            raise ValueError("Text must contain between 1 and 400 characters")
        if type(seed) is not int or not 0 <= seed <= 2147483647:
            raise ValueError("Seed must be an integer between 0 and 2147483647")
        write_progress("generating")
        torch.cuda.synchronize()
        started = time.perf_counter()
        waveform, frames = self.engine.synthesize(
            text=text.strip(), speaker="Ira",
            description="neutral, Indian English accent, steady pace",
            seed=seed, max_new_tokens=2048, temperature=0.8, top_k=30,
        )
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - started
        audio_seconds = waveform.numel() / self.engine.sample_rate
        if audio_seconds <= 0 or not bool(torch.isfinite(waveform).all()):
            raise RuntimeError("The model did not produce a valid waveform")
        write_progress("encoding")
        output = io.BytesIO()
        sf.write(output, waveform.numpy(), self.engine.sample_rate, format="WAV", subtype="PCM_16")
        write_progress("complete")
        return {
            "status": "complete", "kind": "speech", "gpu": "H100",
            "audio_base64": base64.b64encode(output.getvalue()).decode("ascii"),
            "sample_rate": self.engine.sample_rate, "audio_seconds": audio_seconds,
            "inference_seconds": inference_seconds,
            "rtf": inference_seconds / audio_seconds,
            "realtime_factor": audio_seconds / inference_seconds,
            "frames": frames, "seed": seed,
            "load_seconds": self.load_seconds, "warmup_seconds": self.warmup_seconds,
        }


@app.function(image=web_image, cpu=1, memory=1024, timeout=120, max_containers=2, scaledown_window=60)
@modal.concurrent(max_inputs=20)
@modal.asgi_app(requires_proxy_auth=True)
def api():
    from backend.api import create_api

    return create_api(
        RumikWorker,
        modal.FunctionCall.from_id,
        (modal.exception.OutputExpiredError, modal.exception.NotFoundError),
        progress_store,
    )
