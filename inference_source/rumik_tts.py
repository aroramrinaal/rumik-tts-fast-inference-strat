from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer, MimiModel


DEFAULT_MODEL_PATH = Path("/models/rumik-oss-1")


class RumikTTS:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self._validate_environment()
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True, local_files_only=True,
            dtype=self.dtype,
            attn_implementation="sdpa",
        ).eval().to(self.device)
        codec_path = model_path / "codec"
        self.mimi = MimiModel.from_pretrained(codec_path, dtype=self.dtype, local_files_only=True).eval().to(self.device)
        extractor = AutoFeatureExtractor.from_pretrained(codec_path, local_files_only=True)
        self.sample_rate = int(extractor.sampling_rate)
        self.speakers = tuple(self.model.config.speakers)

    def _validate_environment(self) -> None:
        required = (
            "config.json",
            "model.safetensors.index.json",
            "modeling_rumik_oss.py",
            "tokenizer.json",
            "codec/model.safetensors",
        )
        missing = [name for name in required if not (self.model_path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"missing model files: {', '.join(missing)}")
        if not torch.cuda.is_available():
            raise RuntimeError("rumik-oss-1 inference requires an NVIDIA CUDA GPU")

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        speaker: str,
        description: str,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_k: int = 30,
        seed: int = 42,
    ) -> tuple[torch.Tensor, int]:
        if speaker not in self.speakers:
            raise ValueError(f"speaker must be one of: {', '.join(self.speakers)}")
        torch.manual_seed(seed)
        prompt = f'<text>{speaker}: <description="{description}"> {text}<audio>'
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate_audio(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=8,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
        )
        audio_tokens = generated[0, inputs.input_ids.shape[1] :].tolist()
        codes = self.model.audio_tokens_to_codes(audio_tokens).to(self.device)
        waveform = self.mimi.decode(codes).audio_values[0, 0].float().cpu().clamp(-1, 1)
        return waveform, codes.shape[-1]
