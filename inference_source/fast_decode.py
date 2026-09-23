import time
import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers.cache_utils import Cache, StaticLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
from .audio_head import _head, _phase_layout
from .sampling import _tail_kernel, _tail_launch, BLOCK, N_LOGITS, EOS_LOCAL, MASK31
from .residual_norm_decode import residual_norm as final_norm
from .pdl_kernels import residual_norm, linear, packed_attention
from .packed_projection_graph import _pack_layer
from .active_gqa import install as install_active
from .fused_rope_cache import install as install_rope
from .int8_weight_gemv import install as install_int8, quantize


class GraphDecode:
    @torch.inference_mode()
    def __init__(self, model, capacity=2304):
        if capacity > int(model.config.sliding_window):
            raise ValueError("Fixed cache must fit entirely inside the local attention window")
        if model.training or model.config._attn_implementation != "sdpa":
            raise ValueError("Graph decode requires eval mode and SDPA")
        self.model = model
        self.original_forward = model.forward
        self.original_generate = model.generate_audio
        self.capacity = capacity
        self.cache = Cache(layers=[StaticLayer(max_cache_len=capacity)
                                   for _ in range(model.config.num_hidden_layers)])
        self.token = torch.zeros((1, 1), dtype=torch.long, device=model.device)
        self.position = torch.zeros((1, 1), dtype=torch.long, device=model.device)
        self.key_positions = torch.arange(capacity, device=model.device).view(1, 1, 1, -1)
        self.next_position = 0
        self.replays = 0
        start = time.perf_counter()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for i in range(3):
                self.position.fill_(i)
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        self.cache.reset()
        self.position.zero_()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._step()
        torch.cuda.synchronize()
        self.cache.reset()
        self.capture_s = time.perf_counter() - start

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, past_key_values=None, **kwargs):
        if input_ids.shape[0] != 1:
            raise ValueError("Graph decode supports one request at a time")
        if past_key_values is None:

            out = self.original_forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            length = input_ids.shape[1]
            if length >= self.capacity:
                raise ValueError("Prompt exceeds graph cache capacity")
            self.cache.reset()
            for source, target in zip(out.past_key_values.layers, self.cache.layers, strict=True):
                target.keys[:, :, :length].copy_(source.keys)
                target.values[:, :, :length].copy_(source.values)
                target.cumulative_length.fill_(length)
            self.next_position = length
            out.past_key_values = self.cache
            return out
        if past_key_values is not self.cache or input_ids.shape != (1, 1):
            raise ValueError("Unexpected decode cache or input shape")
        if self.next_position >= self.capacity:
            raise ValueError("Graph decode cache capacity exceeded")
        self.token.copy_(input_ids)
        self.position.fill_(self.next_position)
        self.graph.replay()
        self.next_position += 1
        self.replays += 1
        return self.output


class AudioVocabGraph(GraphDecode):
    @torch.inference_mode()
    def __init__(self, model):
        started = time.perf_counter()
        self.audio_ids = model.audio_token_ids(device=model.device)
        self.audio_weight = model.lm_head.weight.index_select(0, self.audio_ids).contiguous()
        self.eos_index = self.audio_ids.numel() - 1
        if int(self.audio_ids[-1]) != int(model.config.audio_end_token_id):
            raise ValueError("Expected EOS at the end of the released audio vocabulary")
        if model.lm_head.bias is not None:
            raise ValueError("This strategy requires the released bias-free output head")
        super().__init__(model)
        torch.cuda.synchronize()
        self.setup_s = time.perf_counter() - started

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, past_key_values=None, **kwargs):
        out = super().forward(input_ids, attention_mask, past_key_values, **kwargs)
        if past_key_values is None:

            out.logits = out.logits[:, -1:, :].index_select(-1, self.audio_ids)
        return out


class BodySetup(AudioVocabGraph):
    @torch.inference_mode()
    def __init__(self, model):
        if len(model.model.layers) != 36 or model.config.hidden_size != 2048:
            raise ValueError("Expected the released 36-layer Rumik architecture")
        for layer in model.model.layers:
            if not hasattr(layer, "int8_in_weight"):
                raise ValueError("Install INT8 projections before initializing the decoder")
        super().__init__(model)


class PhaseVocabHeadRunner(BodySetup):
    quantized = False

    def __init__(self, model):
        self.phase_ids, weights = _phase_layout(model)
        if self.quantized:
            packed, scales = quantize(weights.reshape(-1, 2048))
            self.head_weight = packed.reshape(8, 2049, 2048).contiguous()
            self.head_scales = scales.reshape(8, 2049).contiguous()
            self.head_scales.mul_(float(model.logit_scale))
        else:
            self.head_weight = weights
            self.head_scales = torch.ones((8, 2049), device=model.device,
                                          dtype=torch.float32)
        self.head_phase = torch.zeros((), device=model.device, dtype=torch.int32)
        self.phase_index = 0

        self.prefill_select = torch.cat((
            torch.arange(0, 16384, 8, device=model.device),
            torch.tensor([16384], device=model.device),
        ))
        super().__init__(model)

    def _step(self):
        hidden = self._hidden_step()
        logits = _head(hidden, self.head_weight, self.head_scales,
                       self.head_phase, 2049, self.quantized,
                       self.model.logit_scale)
        return CausalLMOutputWithPast(logits=logits, past_key_values=self.cache,
                                     hidden_states=(hidden,))

    @torch.inference_mode()
    def forward(self, input_ids, attention_mask, past_key_values=None, **kwargs):
        if past_key_values is None:
            self.phase_index = 0
            self.head_phase.zero_()
            out = super().forward(input_ids, attention_mask, past_key_values, **kwargs)
            out.logits = out.logits.index_select(-1, self.prefill_select)
            return out
        self.phase_index = (self.phase_index + 1) % 8
        self.head_phase.fill_(self.phase_index)
        return super().forward(input_ids, attention_mask, past_key_values, **kwargs)


class FusedTailRunner(PhaseVocabHeadRunner):
    @torch.inference_mode()
    def __init__(self, model):
        stop = model.stop_predictor
        if (not isinstance(stop, nn.Sequential) or len(stop) != 4
                or not isinstance(stop[0], nn.LayerNorm)
                or tuple(stop[1].weight.shape) != (512, 2048)
                or tuple(stop[3].weight.shape) != (1, 512)):
            raise ValueError("Expected the released 2048->512->1 stop head")
        self._stop_buf = torch.zeros((), device=model.device, dtype=torch.float32)
        super().__init__(model)
        self.history = torch.zeros((1, self.capacity), device=model.device,
                                   dtype=torch.int64)
        self.sampled = torch.zeros((1, 1), device=model.device, dtype=torch.int64)
        self.tail_action = torch.zeros((), device=model.device, dtype=torch.int32)

    def _step(self):
        out = super()._step()
        hidden = out.hidden_states[0]
        stop_logit = self.model.stop_predictor(hidden).float().reshape(())
        self._stop_buf.copy_(stop_logit)
        return out


class PDLRunner(FusedTailRunner):
    def _hidden_step(self):
        backbone = self.model.model
        residual = backbone.embed_tokens(self.token)
        embeddings = backbone.rotary_emb(residual, self.position)
        branch = None
        for layer in backbone.layers:
            residual, normalized = residual_norm(residual, branch, layer.input_layernorm)
            q, k, v, gate, up = linear(normalized, layer.int8_in_weight,
                layer.int8_in_scale).split(layer.packed_in_splits, dim=-1)
            attn = layer.self_attn
            packed = packed_attention(q.view(1, 16, 1, 128), k.view(1, 4, 1, 128),
                v.view(1, 4, 1, 128), gate, up, *embeddings,
                self.cache.layers[attn.layer_idx], attn.sliding_window is not None,
                attn.scaling)
            branch = linear(packed, layer.int8_out_weight, layer.int8_out_scale)
        _, hidden = final_norm(residual, branch, backbone.norm)
        return hidden


@triton.jit
def _finish(Sampled, Token, Position, Phase, Controls, History, Action,
            EOS: tl.constexpr):
    pos = tl.load(Controls)
    action = tl.load(Action)
    token = tl.load(Sampled)
    if action == 1:
        tl.store(History + pos, EOS)
    tl.store(Token, token)
    tl.store(Position, tl.load(Position) + 1)
    tl.store(Phase, (tl.load(Phase) + 1) % 8)
    tl.store(Controls, pos + 1)
    tl.store(Controls + 1, tl.load(Controls + 1) + 1)
    tl.store(Controls + 5, pos + 1)


class ConditionalRunner(PDLRunner):
    @torch.inference_mode()
    def __init__(self, model):
        from .conditional_graph import ConditionalGraph
        super().__init__(model)

        self.controls = torch.tensor([1, 1, 8, 42, 2, 1], device=model.device,
                                     dtype=torch.int64)
        self.temperature = torch.tensor([.8], device=model.device)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._generation_step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.cache.reset(); self.position.zero_(); self.head_phase.zero_()
        self.body_graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(self.body_graph):
            self._generation_step()
        torch.cuda.synchronize()
        self.conditional_graph = ConditionalGraph(self.body_graph, self.tail_action,
                                                  self.controls)
        self.cache.reset(); self.position.zero_(); self.head_phase.zero_()

    def _generation_step(self):
        output = self._step()
        _tail_kernel[(1,)](output.logits, self.phase_ids, self._stop_buf,
            self.sampled, self.history, self.tail_action,
            self.controls[0:1], self.controls[1:2], self.controls[2:3],
            self.temperature, self.controls[3:4], 30, True, True,
            BLOCK, N_LOGITS, EOS_LOCAL, MASK31,
            PhaseOffset=self.head_phase, DEVICE_CONTROL=True, num_warps=32)
        _finish[(1,)](self.sampled, self.token, self.position, self.head_phase,
            self.controls, self.history, self.tail_action,
            int(self.model.config.audio_end_token_id), num_warps=1)

    @torch.inference_mode()
    def generate_audio(self, input_ids, attention_mask, allowed_ids=None,
                       max_new_tokens=2048, min_new_tokens=8, temperature=.8,
                       top_k=30, do_sample=True):
        if allowed_ids is not None or input_ids.shape[0] != 1 or not bool((attention_mask == 1).all()):
            raise ValueError('Conditional decode requires one unpadded phase-vocabulary request')
        if min_new_tokens < 1 or top_k != 30 or temperature != .8 or not do_sample:
            raise ValueError('Conditional decode requires temperature=0.8, top_k=30, and sampling enabled')
        prefix, count = input_ids.shape[1], int(max_new_tokens)
        if count < 1 or prefix + count > self.capacity:
            raise ValueError('Expected positive token cap and total context <=2304')
        output = self.forward(input_ids, attention_mask, use_cache=True,
                              output_hidden_states=True, return_dict=True)
        seed = int(torch.initial_seed() & MASK31)
        _tail_launch(output.logits.reshape(-1), self.phase_ids[0], self._stop_buf,
            self.sampled, self.history, self.tail_action, prefix, 0,
            min_new_tokens, .8, seed, 30, True, 32)

        self.token.copy_(self.sampled)
        self.position.fill_(prefix)
        self.head_phase.fill_(1)
        self.controls.copy_(torch.tensor([prefix+1, 1, min_new_tokens, seed,
                                         count, prefix+1], dtype=torch.int64))
        self.conditional_graph.replay()
        end = int(self.controls[5].item())
        return torch.cat((input_ids, self.history[:, prefix:end]), dim=1)


@torch.inference_mode()
def apply(engine):
    if torch.cuda.get_device_capability()[0] != 9:
        raise ValueError("This stack requires a Hopper H100 GPU")
    for layer in engine.model.model.layers:
        _pack_layer(layer)
    install_active(engine.model)
    install_rope(engine.model)
    install_int8(engine.model)
    runner = ConditionalRunner(engine.model)
    engine.model.generate_audio = runner.generate_audio
    engine.runner = runner
    return engine
