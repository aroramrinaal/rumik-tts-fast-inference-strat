import torch
import triton
import triton.language as tl

@triton.jit
def _audio_gemv(X, W, S, Phase, Y, N: tl.constexpr, K: tl.constexpr,
                ROWS: tl.constexpr, PHASED: tl.constexpr,
                QUANTIZED: tl.constexpr, LOGIT_SCALE: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, K)
    phase = tl.load(Phase) if PHASED else 0
    base = phase * N * K if PHASED else 0
    x = tl.load(X + cols).to(tl.float32)
    weight = tl.load(W + base + rows[:, None] * K + cols[None, :],
                     rows[:, None] < N, other=0).to(tl.float32)
    result = tl.sum(weight * x[None, :], axis=1)
    if QUANTIZED:
        scale_base = phase * N if PHASED else 0
        result *= tl.load(S + scale_base + rows, rows < N, other=0)
    else:
        result *= LOGIT_SCALE
    tl.store(Y + rows, result, rows < N)


def _head(hidden, weight, scales, phase, n, quantized, logit_scale):
    if hidden.shape != (1, 1, 2048) or hidden.dtype != torch.bfloat16:
        raise ValueError("Audio head requires one contiguous BF16 hidden state")
    if not hidden.is_cuda or not hidden.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Audio head tensors must be contiguous CUDA tensors")
    output = torch.empty((1, 1, n), device=hidden.device, dtype=torch.bfloat16)
    _audio_gemv[(triton.cdiv(n, 4),)](
        hidden, weight, scales, phase, output, n, 2048, 4,
        weight.ndim == 3, quantized, float(logit_scale), num_warps=4,
    )
    return output


def _phase_layout(model):
    config = model.config
    first, last = int(config.first_unit_id), int(config.last_unit_id)
    if last - first + 1 != 2048 * 8 or int(config.num_quantizers) != 8:
        raise ValueError("Expected released 2,048-code x 8-quantizer vocabulary")
    codes = torch.arange(2048, device=model.device, dtype=torch.long)
    ids = first + codes[:, None] * 8 + torch.arange(8, device=model.device)
    ids = ids.transpose(0, 1).contiguous()
    eos = torch.full((8, 1), int(config.audio_end_token_id), device=model.device,
                     dtype=torch.long)
    phase_ids = torch.cat((ids, eos), dim=1)
    weights = model.lm_head.weight.index_select(0, phase_ids.flatten()).reshape(
        8, 2049, 2048).contiguous()
    return phase_ids, weights
