import torch
import triton
import triton.language as tl
from transformers.cache_utils import StaticLayer
from .rope_attention import _partials


LAST_COMPILED = {}


@triton.jit
def _residual_norm(R, B, W, Updated, Y, EPS: tl.constexpr, ADD: tl.constexpr, USE_GDC: tl.constexpr):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
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


def residual_norm(residual, branch, norm, pdl=True):
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
    LAST_COMPILED["norm"] = _residual_norm[(1,)](
        residual, residual if branch is None else branch, norm.weight,
        updated, normalized, norm.variance_epsilon, branch is not None, pdl,
        num_warps=4, enable_fp_fusion=False, launch_pdl=pdl)
    return updated, normalized


@triton.jit
def _gemv(X, W, S, Y, N: tl.constexpr, K: tl.constexpr,
          ROWS: tl.constexpr, BLOCK_K: tl.constexpr, USE_GDC: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_K)
    w = tl.load(W + rows[:, None] * K + cols[None, :],
                (rows[:, None] < N) & (cols[None, :] < K), other=0).to(tl.float32)
    scale = tl.load(S + rows, rows < N, other=0)

    if USE_GDC:
        tl.extra.cuda.gdc_wait()
    x = tl.load(X + cols, cols < K, other=0).to(tl.float32)
    result = tl.sum(w * x[None, :], axis=1) * scale
    tl.store(Y + rows, result, rows < N)


def linear(x, weight, scale, pdl=True):
    n, k = weight.shape
    if x.shape != (1, 1, k) or x.dtype != torch.bfloat16 or x.stride(-1) != 1:
        raise ValueError("INT8 GEMV requires contiguous BF16 [1,1,K] activations")
    if weight.dtype != torch.int8 or not weight.is_contiguous():
        raise ValueError("INT8 GEMV requires contiguous INT8 row-major weights")
    if scale.shape != (n,) or scale.dtype != torch.float32 or not scale.is_contiguous():
        raise ValueError("INT8 GEMV requires one contiguous FP32 scale per row")
    if (n, k) not in ((25088, 2048), (2048, 13056), (23296, 2048), (2048, 12160)):
        raise ValueError("Unexpected packed projection dimensions (released or slim-10112)")
    rows, warps = (4, 4) if k == 2048 else (1, 8)
    output = torch.empty((1, 1, n), device=x.device, dtype=x.dtype)
    side = "ingress" if k == 2048 else "egress"
    LAST_COMPILED[side] = _gemv[(triton.cdiv(n, rows),)](
        x, weight, scale, output, n, k, rows, triton.next_power_of_2(k), pdl,
        num_warps=warps, launch_pdl=pdl)
    return output


@triton.jit
def _merge_gate(Partial, Lse, Gate, Up, Out, Position,
                SPLITS: tl.constexpr, PAD: tl.constexpr, SEG: tl.constexpr,
                USE_GDC: tl.constexpr):
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    head = tl.program_id(0)
    s = tl.arange(0, PAD)
    d = tl.arange(0, 128)
    lse = tl.load(Lse + head * SPLITS + s, s < SPLITS, other=-float("inf"))
    weights = tl.exp(lse - tl.max(lse, axis=0))
    weights = weights / tl.sum(weights, axis=0)
    values = tl.load(Partial + (head * SPLITS + s[:, None]) * 128 + d[None, :],
                     s[:, None] < SPLITS, other=0)
    tl.store(Out + head * 128 + d, tl.sum(values * weights[:, None], axis=0))


    i = tl.arange(0, 1024)
    j = head * SEG + i
    gate = tl.load(Gate + j, i < SEG, other=0).to(tl.float32)
    up = tl.load(Up + j, i < SEG, other=0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
    tl.store(Out + 2048 + j, activated * up, i < SEG)


    if head == 0:
        tl.store(Position, tl.load(Position) + 1)


def packed_attention(q, k, v, gate, up, cos, sin, cache, rotary, scale=128 ** -0.5, pdl=True):
    if type(cache) is not StaticLayer:
        raise ValueError("Expected plain StaticLayer")
    if not cache.is_initialized:
        cache.lazy_initialization(k, v)
    if q.shape != (1, 16, 1, 128) or k.shape != (1, 4, 1, 128) or v.shape != k.shape:
        raise ValueError("Expected released single-token attention layout")
    if gate.shape != (1, 1, up.shape[-1]) or gate.dim() != 3:
        raise ValueError("Expected single-token gate/up with a shared MLP width")
    width = int(gate.shape[-1])
    if width % 16 != 0 or width > 16 * 1024:
        raise ValueError("Slimmed MLP width must be a multiple of 16 within one 1024 tile")
    if any(x.dtype != torch.bfloat16 or not x.is_contiguous() for x in (q, k, v, gate, up)):
        raise ValueError("Expected contiguous BF16 projections")
    if any(x.numel() != 128 or not x.is_contiguous() for x in (cos, sin)):
        raise ValueError("Expected one-position rotary tables")
    cap = cache.max_cache_len
    if cap != 2304 or not cache.keys.is_contiguous() or not cache.values.is_contiguous():
        raise ValueError("Expected contiguous 2304-position cache")
    splits = triton.cdiv(cap, 256)
    partial = torch.empty((16, splits, 128), device=q.device, dtype=torch.float32)
    lse = torch.empty((16, splits), device=q.device, dtype=torch.float32)
    output = torch.empty((1, 1, 2048 + width), device=q.device, dtype=q.dtype)
    _partials[(16, splits)](q, k, v, cos, sin, cache.keys, cache.values,
        cache.cumulative_length, partial, lse, cap, splits, scale, rotary, 256,
        num_warps=8, enable_fp_fusion=False)
    LAST_COMPILED["merge"] = _merge_gate[(16,)](partial, lse, gate, up, output, cache.cumulative_length,
                       splits, triton.next_power_of_2(splits), width // 16, pdl,
                       num_warps=4, launch_pdl=pdl)
    return output
