import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.models.cohere2.modeling_cohere2 import apply_rotary_pos_emb

@triton.jit
def _partials(Q, K, V, Mask, Partial, Lse,
              QH: tl.constexpr, KH: tl.constexpr, KS: tl.constexpr,
              VH: tl.constexpr, VS: tl.constexpr,
              LENGTH: tl.constexpr, SPLITS: tl.constexpr,
              SCALE: tl.constexpr, BLOCK: tl.constexpr):
    head = tl.program_id(0)
    split = tl.program_id(1)
    d = tl.arange(0, 128)
    start = split * BLOCK
    output = tl.full((128,), 0, tl.float32)
    lse = tl.full((), -float("inf"), tl.float32)
    active = tl.load(Mask + start, start < LENGTH, other=0)
    if active:
        t = start + tl.arange(0, BLOCK)
        valid = tl.load(Mask + t, t < LENGTH, other=0)
        q = tl.load(Q + head * QH + d).to(tl.float32)
        k = tl.load(K + (head // 4) * KH + t[:, None] * KS + d[None, :],
                    valid[:, None], other=0).to(tl.float32)
        scores = tl.sum(k * q[None, :], axis=1) * SCALE
        scores = tl.where(valid, scores, -float("inf"))
        maximum = tl.max(scores, axis=0)
        probabilities = tl.exp(scores - maximum)
        denominator = tl.sum(probabilities, axis=0)
        v = tl.load(V + (head // 4) * VH + t[:, None] * VS + d[None, :],
                    valid[:, None], other=0).to(tl.float32)
        output = tl.sum(v * probabilities[:, None], axis=0) / denominator
        lse = maximum + tl.log(denominator)
    tl.store(Partial + (head * SPLITS + split) * 128 + d, output)
    tl.store(Lse + head * SPLITS + split, lse)


@triton.jit
def _merge(Partial, Lse, Out, SPLITS: tl.constexpr, PAD: tl.constexpr):
    head = tl.program_id(0)
    s = tl.arange(0, PAD)
    d = tl.arange(0, 128)
    lse = tl.load(Lse + head * SPLITS + s, s < SPLITS, other=-float("inf"))
    weights = tl.exp(lse - tl.max(lse, axis=0))
    weights = weights / tl.sum(weights, axis=0)
    values = tl.load(Partial + (head * SPLITS + s[:, None]) * 128 + d[None, :],
                     s[:, None] < SPLITS, other=0)
    output = tl.sum(values * weights[:, None], axis=0)
    tl.store(Out + head * 128 + d, output)


def attend(q, k, v, mask, scale=128 ** -0.5):

    if q.shape != (1, 16, 1, 128) or k.shape[0:2] != (1, 4):
        raise ValueError("Active GQA requires batch one, 16 Q heads and four KV heads")
    if k.shape != v.shape or k.shape[-1] != 128 or mask.dtype != torch.bool:
        raise ValueError("Unexpected static cache or mask")
    if mask.numel() != k.shape[-2] or not mask.is_contiguous():
        raise ValueError("Expected contiguous one-query prefix mask")
    if q.stride(-1) != 1 or k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Head dimension must be contiguous")
    length, block = k.shape[-2], 256
    splits = triton.cdiv(length, block)
    partial = torch.empty((16, splits, 128), dtype=torch.float32, device=q.device)
    lse = torch.empty((16, splits), dtype=torch.float32, device=q.device)
    output = torch.empty((1, 16, 1, 128), dtype=q.dtype, device=q.device)
    _partials[(16, splits)](
        q, k, v, mask, partial, lse, q.stride(1), k.stride(1), k.stride(2),
        v.stride(1), v.stride(2), length, splits, scale, block, num_warps=8,
    )
    _merge[(16,)](partial, lse, output, splits, triton.next_power_of_2(splits), num_warps=4)
    return output


def _forward(self, hidden_states, position_embeddings=None, attention_mask=None,
             past_key_values=None, use_cache=False, **kwargs):
    if hidden_states.shape[:2] != (1, 1) or past_key_values is None:
        return self._active_gqa_parent(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, **kwargs,
        )
    residual = hidden_states
    normalized = self.input_layernorm(hidden_states)
    q, k, v, gate, up = F.linear(normalized, self.packed_in_weight).split(
        self.packed_in_splits, dim=-1)
    attention = self.self_attn
    q = q.view(1, 1, 16, 128).transpose(1, 2)
    k = k.view(1, 1, 4, 128).transpose(1, 2)
    v = v.view(1, 1, 4, 128).transpose(1, 2)
    if attention.sliding_window is not None:
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    k, v = past_key_values.update(k, v, attention.layer_idx)
    attended = attend(q, k, v, attention_mask, attention.scaling).reshape(1, 1, 2048)
    gated = self.mlp.act_fn(gate) * up
    branches = F.linear(torch.cat((attended, gated), dim=-1), self.packed_out_weight)
    return residual + branches


def install(model):
    if model.training or int(model.config.sliding_window) < 2304:
        raise ValueError("Expected eval mode and local window >= static capacity 2304")
    for layer in model.model.layers:
        if not hasattr(layer, "packed_in_weight"):
            raise ValueError("Pack BF16 layers before installing active GQA")
        layer._active_gqa_parent = layer.forward
        layer.forward = types.MethodType(_forward, layer)
