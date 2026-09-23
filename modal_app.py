# Device-controlled CUDA graphs combine Hopper PDL kernels with INT8 Transformer weights.
# A BF16 phase-aware audio head and fused sampling feed the Mimi waveform decoder.

import base64
from pathlib import Path

from backend.modal_web_app import app, RumikWorker


@app.local_entrypoint()
def main(
    text: str = "Hello, this is Rumik text to speech.",
    output: str = "speech.wav",
    seed: int = 42,
):
    result = RumikWorker().synthesize.remote(text, seed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(base64.b64decode(result["audio_base64"]))
    print(f"{destination.resolve()} | {result['inference_seconds']:.3f}s | RTF {result['rtf']:.3f}")
