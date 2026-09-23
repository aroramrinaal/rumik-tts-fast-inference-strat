import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.cache_utils import StaticLayer
from transformers.models.cohere2.modeling_cohere2 import apply_rotary_pos_emb
from .active_gqa import attend

@triton.jit
def _rope_write(Q, K, V, Cos, Sin, Out, KC, VC, Position,
                CAPACITY: tl.constexpr, ROTARY: tl.constexpr):


    i = tl.arange(0, 2048)
    d = i % 128
    position = tl.load(Position)
    q = tl.load(Q + i).to(tl.float32)
    k = tl.load(K + i, i < 512, other=0).to(tl.float32)
    if ROTARY:
        cos = tl.load(Cos + d).to(tl.float32)
        sin = tl.load(Sin + d).to(tl.float32)
        qp = tl.load(Q + (i ^ 1)).to(tl.float32)
        kp = tl.load(K + (i ^ 1), i < 512, other=0).to(tl.float32)
        qp = tl.where(d % 2 == 0, -qp, qp)
        kp = tl.where(d % 2 == 0, -kp, kp)
        q = q * cos + qp * sin
        k = k * cos + kp * sin
    tl.store(Out + i, q)
    v = tl.load(V + i, i < 512, other=0)
    destination = (i // 128) * CAPACITY * 128 + position * 128 + d
    valid = (i < 512) & (position >= 0) & (position < CAPACITY)
    tl.store(KC + destination, k, valid)
    tl.store(VC + destination, v, valid)
    tl.store(Position, position + 1)


def rope_write(q, k, v, cos, sin, cache, rotary):
    if type(cache) is not StaticLayer:
        raise ValueError("Fused RoPE cache requires a plain Transformers StaticLayer")
    if not cache.is_initialized:
        cache.lazy_initialization(k, v)
    if q.shape != (1, 16, 1, 128) or k.shape != (1, 4, 1, 128) or v.shape != k.shape:
        raise ValueError("Unexpected fused RoPE Q/K/V shapes")
    if any(x.dtype != torch.bfloat16 or not x.is_contiguous() for x in (q, k, v)):
        raise ValueError("Fused RoPE requires contiguous BF16 Q/K/V")
    if cos.numel() != 128 or sin.numel() != 128 or not cos.is_contiguous() or not sin.is_contiguous():
        raise ValueError("Fused RoPE requires contiguous single-position rotary tables")
    output = torch.empty_like(q)
    _rope_write[(1,)](q, k, v, cos, sin, output, cache.keys, cache.values,
                      cache.cumulative_length, cache.max_cache_len, rotary,
                      num_warps=4, enable_fp_fusion=False)
    return output, cache.keys, cache.values


def _forward(self, hidden_states, position_embeddings=None, attention_mask=None,
             past_key_values=None, use_cache=False, **kwargs):
    if (hidden_states.shape[:2] != (1, 1) or past_key_values is None
            or type(past_key_values.layers[self.self_attn.layer_idx]) is not StaticLayer):
        return self._fused_rope_parent(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, **kwargs)
    residual = hidden_states
    normalized = self.input_layernorm(hidden_states)
    q, k, v, gate, up = F.linear(normalized, self.packed_in_weight).split(
        self.packed_in_splits, dim=-1)
    attention = self.self_attn
    q = q.view(1, 1, 16, 128).transpose(1, 2)
    k = k.view(1, 1, 4, 128).transpose(1, 2)
    v = v.view(1, 1, 4, 128).transpose(1, 2)
    q, k, v = rope_write(q, k, v, *position_embeddings,
                         past_key_values.layers[attention.layer_idx],
                         attention.sliding_window is not None)
    attended = attend(q, k, v, attention_mask, attention.scaling).reshape(1, 1, 2048)
    gated = self.mlp.act_fn(gate) * up
    branches = F.linear(torch.cat((attended, gated), dim=-1), self.packed_out_weight)
    return residual + branches


def install(model):
    if model.training:
        raise ValueError("Fused RoPE cache requires evaluation mode")
    for layer in model.model.layers:
        if not hasattr(layer, "_active_gqa_parent"):
            raise ValueError("Install active GQA before fused RoPE cache")
        layer._fused_rope_parent = layer.forward
        layer.forward = types.MethodType(_forward, layer)
