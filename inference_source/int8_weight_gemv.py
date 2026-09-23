import types
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from transformers.cache_utils import StaticLayer
from .active_gqa import attend
from .fused_rope_cache import rope_write

@triton.jit
def _gemv(X, W, S, Y, N: tl.constexpr, K: tl.constexpr,
          ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK_K)
    x = tl.load(X + cols, cols < K, other=0).to(tl.float32)
    w = tl.load(W + rows[:, None] * K + cols[None, :],
                (rows[:, None] < N) & (cols[None, :] < K), other=0).to(tl.float32)
    scale = tl.load(S + rows, rows < N, other=0)
    result = tl.sum(w * x[None, :], axis=1) * scale
    tl.store(Y + rows, result, rows < N)


def quantize(weight):

    fp = weight.float()
    maximum = fp.abs().amax(dim=1)
    scale = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    packed = (fp / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return packed.contiguous(), scale.contiguous()


def linear(x, weight, scale):
    n, k = weight.shape
    if x.shape != (1, 1, k) or x.dtype != torch.bfloat16 or x.stride(-1) != 1:
        raise ValueError("INT8 GEMV requires contiguous BF16 [1,1,K] activations")
    if weight.dtype != torch.int8 or not weight.is_contiguous():
        raise ValueError("INT8 GEMV requires contiguous INT8 row-major weights")
    if scale.shape != (n,) or scale.dtype != torch.float32 or not scale.is_contiguous():
        raise ValueError("INT8 GEMV requires one contiguous FP32 scale per row")
    if (n, k) not in ((25088, 2048), (2048, 13056), (23296, 2048), (2048, 12160),
                       (21504, 2048), (2048, 11264)):
        raise ValueError("Unexpected packed projection dimensions (released, slim-10112 or selective-9216)")
    rows, warps = (4, 4) if k == 2048 else (1, 8)
    output = torch.empty((1, 1, n), device=x.device, dtype=x.dtype)
    _gemv[(triton.cdiv(n, rows),)](
        x, weight, scale, output, n, k, rows, triton.next_power_of_2(k),
        num_warps=warps)
    return output


def _forward(self, hidden_states, position_embeddings=None, attention_mask=None,
             past_key_values=None, use_cache=False, **kwargs):
    if (hidden_states.shape[:2] != (1, 1) or past_key_values is None
            or type(past_key_values.layers[self.self_attn.layer_idx]) is not StaticLayer):
        return self._int8_weight_parent(
            hidden_states, position_embeddings=position_embeddings,
            attention_mask=attention_mask, past_key_values=past_key_values,
            use_cache=use_cache, **kwargs)
    residual = hidden_states
    normalized = self.input_layernorm(hidden_states)
    q, k, v, gate, up = linear(normalized, self.int8_in_weight, self.int8_in_scale).split(
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
    branches = linear(torch.cat((attended, gated), dim=-1),
                      self.int8_out_weight, self.int8_out_scale)
    return residual + branches


@torch.inference_mode()
def install(model):
    if model.training:
        raise ValueError("INT8 weight GEMV requires evaluation mode")
    for layer in model.model.layers:
        if not hasattr(layer, "_fused_rope_parent"):
            raise ValueError("Install fused RoPE/cache before INT8 weight GEMV")
        if hasattr(layer, "_int8_weight_parent"):
            raise ValueError("INT8 weight GEMV already installed")
        for side in ("in", "out"):
            packed, scales = quantize(getattr(layer, f"packed_{side}_weight"))
            layer.register_buffer(f"int8_{side}_weight", packed, persistent=False)
            layer.register_buffer(f"int8_{side}_scale", scales, persistent=False)
        layer._int8_weight_parent = layer.forward
        layer.forward = types.MethodType(_forward, layer)
