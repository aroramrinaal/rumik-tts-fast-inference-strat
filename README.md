# Rumik TTS fast inference

An H100-specialized inference stack for [Rumik OSS 1](https://huggingface.co/rumik-ai/rumik-oss-1), with a Modal backend and a Next.js App Router frontend on Cloudflare Workers through OpenNext.

[Live demo](https://aroramrinaal.com/rumik-tts-fast-inference) · [Technical write-up](https://aroramrinaal.com/ai/rumik-tts-fast-inference)

## Inference stack

The retained experiment-74 stack combines INT8 weight-only decode projections, fused Triton kernels, packed projections, active-cache attention, Hopper programmatic dependent launch, and a device-controlled conditional CUDA WHILE graph. A phase-restricted BF16 audio head and fused sampler produce audio tokens; Mimi decodes them into a waveform.

This implementation targets a single Hopper H100 request at a time, with a maximum of 2,304 prompt-plus-output tokens. The demo uses Ira, seed 42, temperature 0.8, top-k 30, and a 2,048-token output cap. Prefill, activations, the audio head, and Mimi remain BF16. Quantization, phase restriction, and replacement sampling can change outputs; this is not a claim of stock-equivalent quality. Benchmark methodology and quality-screen limitations are in the write-up.

## Repository

- `inference_source/`: retained decode stack, CUDA graph control, and Triton kernels.
- `backend/`: Modal worker, FastAPI endpoints, and progress reporting.
- `modal_app.py`: command-line entry point for synthesis.
- `web/`: Next.js frontend and server-side proxy. Credentials never go to the browser.

## Backend setup

Use Python 3.12 and install `backend/requirements.txt` into a virtual environment. Authenticate the Modal CLI with your own account.

The backend expects an existing Modal volume named `rumik-models` (override with `RUMIK_MODEL_VOLUME`) containing the complete model snapshot under `rumik-oss-1/`, including its custom Python files, tokenizer, weight shards, and `codec/` directory. It mounts this volume read-only at `/models` and loads offline. Obtain the model separately from the original repository, respect its license, and pin the revision for reproducibility. Model weights are not included here.

```sh
python3.12 -m venv .venv
.venv/bin/pip install -r backend/requirements.txt
.venv/bin/modal setup
.venv/bin/modal deploy modal_app.py
```

Modal builds the GPU image with the pinned dependencies in `backend/modal_web_app.py`. GPU startup loads the model and captures the graphs; the worker scales down after two minutes idle. The HTTP endpoint requires Modal proxy authentication.

## Frontend setup

Use Node.js and pnpm (version pinned in `web/package.json`). Copy `web/.dev.vars.example` to `web/.dev.vars` and supply your Modal endpoint and proxy credentials. For production, set `MODAL_ENDPOINT`, `MODAL_PROXY_TOKEN_ID`, and `MODAL_PROXY_TOKEN_SECRET` as Cloudflare secrets.

```sh
cd web
pnpm install --frozen-lockfile
pnpm dev
```

Update `web/wrangler.jsonc` for your own Worker name, domain routes, and self-reference service. Keep the `TTS_RATE_LIMITER` binding. The app's base path is configured in `web/lib/config.ts`; the bundled font URLs and static header path also use that prefix.

The original local DIN Next font files are excluded because redistribution permission is not documented in this repository. The UI falls back to Arial/Helvetica; you can supply your own licensed font files or change the font-face definitions.

## Checks

```sh
python3 -B -m unittest backend.test_status
cd web
pnpm typecheck
pnpm lint
pnpm test:proxy
```

The proxy tests use an isolated local Worker runtime, mock credentials, and mock upstream responses. They do not start the Next.js app or submit GPU jobs.

## License and attribution

Project code is available under the MIT License. The Rumik model, its custom model implementation, model weights, Mimi assets, dependencies, and fonts retain their respective upstream licenses; this repository's license does not relicense them. See the [original model](https://huggingface.co/rumik-ai/rumik-oss-1) and [research post](https://rumik.ai/research/rumik-oss). UI styling was adapted from the author's Before It Codes project.
