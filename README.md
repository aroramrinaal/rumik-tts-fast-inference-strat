# Rumik TTS fast inference

[Rumik OSS 1](https://huggingface.co/rumik-ai/rumik-oss-1) extends Tiny Aya Fire to generate audio tokens autoregressively; Mimi decodes them into speech.
Eight codebook tokens form each audio frame.

Our H100 inference stack combines INT8 weight-only decode, fused Triton kernels, Hopper PDL, and a GPU-controlled CUDA WHILE graph with a phase-restricted BF16 audio head.
[Read the write-up](https://aroramrinaal.com/ai/rumik-tts-fast-inference).
