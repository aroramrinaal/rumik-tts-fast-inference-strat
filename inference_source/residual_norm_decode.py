import torch
import triton
import triton.language as tl
from transformers.models.cohere2.modeling_cohere2 import Cohere2LayerNorm

@triton.jit
def _residual_norm(R, B, W, Updated, Y, EPS: tl.constexpr, ADD: tl.constexpr):
    i = tl.arange(0, 2048)
    value = tl.load(R + i).to(tl.float32)
    if ADD:
        branch = tl.load(B + i).to(tl.float32)

        value = (value + branch).to(tl.bfloat16).to(tl.float32)
        tl.store(Updated + i, value)
    mean = tl.sum(value, axis=0) / 2048
    centered = value - mean
    variance = tl.sum(centered * centered, axis=0) / 2048
    normalized = centered * tl.rsqrt(variance + EPS)
    weight = tl.load(W + i).to(tl.float32)
    tl.store(Y + i, weight * normalized)


def residual_norm(residual, branch, norm):
    if (residual.shape != (1, 1, 2048) or residual.dtype != torch.bfloat16
            or not residual.is_cuda or not residual.is_contiguous()):
        raise ValueError("Residual normalization requires contiguous CUDA BF16 [1,1,2048]")
    if branch is not None and (branch.shape != residual.shape
            or branch.dtype != residual.dtype or not branch.is_contiguous()
            or branch.device != residual.device):
        raise ValueError("Branch must match the residual shape, device and dtype")
    if norm.weight.shape != (2048,) or not norm.weight.is_contiguous():
        raise ValueError("Expected one contiguous width-2048 norm weight")
    updated = torch.empty_like(residual) if branch is not None else residual
    normalized = torch.empty_like(residual)
    _residual_norm[(1,)](
        residual, residual if branch is None else branch, norm.weight,
        updated, normalized, norm.variance_epsilon, branch is not None,
        num_warps=4, enable_fp_fusion=False)
    return updated, normalized
