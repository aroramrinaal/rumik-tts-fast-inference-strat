import types
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.cohere2.modeling_cohere2 import apply_rotary_pos_emb, eager_attention_forward

def _packed_forward(
    self, hidden_states, position_embeddings=None, attention_mask=None,
    past_key_values=None, use_cache=False, **kwargs,
):
    residual = hidden_states
    normalized = self.input_layernorm(hidden_states)
    q, k, v, gate, up = F.linear(normalized, self.packed_in_weight).split(
        self.packed_in_splits, dim=-1
    )

    attention = self.self_attn
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    q = q.view(hidden_shape).transpose(1, 2)
    k = k.view(hidden_shape).transpose(1, 2)
    v = v.view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    if attention.sliding_window is not None:
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, attention.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        attention.config._attn_implementation, eager_attention_forward
    )
    attended, _ = attention_interface(
        attention, q, k, v, attention_mask,
        dropout=0.0 if not self.training else attention.attention_dropout,
        scaling=attention.scaling, sliding_window=attention.sliding_window,
        **kwargs,
    )
    attended = attended.reshape(*input_shape, -1).contiguous()
    gated = self.mlp.act_fn(gate) * up
    branches = F.linear(torch.cat((attended, gated), dim=-1), self.packed_out_weight)
    return residual + branches


def _pack_layer(layer):
    if hasattr(layer, "packed_in_weight"):
        raise ValueError("Layer is already packed")
    attention, mlp = layer.self_attn, layer.mlp
    ingress = (attention.q_proj, attention.k_proj, attention.v_proj,
               mlp.gate_proj, mlp.up_proj)
    if any(module.bias is not None for module in ingress + (attention.o_proj, mlp.down_proj)):
        raise ValueError("Packed projection requires the released bias-free layers")
    in_weight = torch.cat([module.weight for module in ingress], dim=0).contiguous()
    out_weight = torch.cat((attention.o_proj.weight, mlp.down_proj.weight), dim=1).contiguous()
    layer.register_parameter("packed_in_weight", nn.Parameter(in_weight, requires_grad=False))
    layer.register_parameter("packed_out_weight", nn.Parameter(out_weight, requires_grad=False))
    layer.packed_in_splits = tuple(module.out_features for module in ingress)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attention, name, nn.Identity())
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(mlp, name, nn.Identity())
    layer.forward = types.MethodType(_packed_forward, layer)
